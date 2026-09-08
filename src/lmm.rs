//! Profiled REML/ML for linear mixed models with one grouping factor.
//!
//! Follows the lme4 formulation (Bates, Machler, Bolker & Walker 2015, JSS 67(1)):
//! the fixed effects `beta` and the residual variance `sigma^2` are eliminated
//! analytically, so the optimiser only ever sees `theta` -- the `q(q+1)/2`
//! entries of the relative covariance factor `Lambda`.
//!
//! With one grouping factor, `Lambda' Z'Z Lambda + I` is block diagonal: `m`
//! independent `q x q` blocks. No general sparse solver is needed, and every
//! block is independent, so the whole thing parallelises cleanly.
//!
//! # The analytic gradient
//!
//! `lme4` and `MixedModels.jl` both optimise `theta` derivative-free (BOBYQA).
//! We derive and evaluate the gradient instead, which cuts the number of
//! objective evaluations substantially. Writing `D_k = dLambda/dtheta_k` (a
//! single-entry matrix with a 1 at `(r_k, c_k)`), `M_i = Z_i'Z_i Lambda`,
//! `A_i = Lambda' Z_i'Z_i Lambda + I`, `W_i = Lambda' Z_i'X`, `B_i = A_i^-1 W_i`
//! and `P = (X'X - sum_i W_i' A_i^-1 W_i)^-1`:
//!
//! ```text
//! d(ldL2)/dtheta_k  = 2 sum_i (M_i A_i^-1)[r_k, c_k]
//!
//! d(pwrss)/dtheta_k = -2 sum_i u_i[c_k] * t_i[r_k]
//!                     with t_i = Z_i'y - Z_i'X beta - M_i u_i   (envelope theorem)
//!
//! d(ldRX2)/dtheta_k = -2 sum_i [ (Z_i'X)[r_k,:] . (B_i P)[c_k,:]
//!                                - M_i[r_k,:] . (B_i P B_i')[:,c_k] ]
//! ```
//!
//! and finally
//!
//! ```text
//! d(dev)/dtheta_k = d(ldL2) + [d(ldRX2)] + (dfree / pwrss) * d(pwrss)
//! ```
//!
//! The `pwrss` term uses the envelope theorem: `beta` and `u` minimise the
//! penalised least squares problem at fixed `theta`, so only the explicit
//! `theta` dependence contributes.
//!
//! Every one of these is checked against central finite differences in the test
//! suite, because this derivation is the one place the project goes beyond its
//! references.

use crate::linalg::*;
use rayon::prelude::*;

const LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// Theta-independent cross-products. Formed once, before the optimiser starts.
pub struct LmmData {
    pub n: usize,
    pub p: usize,
    pub q: usize,
    pub m: usize,
    pub xtx: Vec<f64>, // p*p
    pub xty: Vec<f64>, // p
    pub yty: f64,
    pub ztz: Vec<f64>, // m*q*q
    pub ztx: Vec<f64>, // m*q*p
    pub zty: Vec<f64>, // m*q
}

/// Result of one objective evaluation.
pub struct Eval {
    pub deviance: f64,
    pub grad: Vec<f64>,
    pub beta: Vec<f64>,
    pub u: Vec<f64>, // m*q, spherical scale
    pub sigma2: f64,
    pub pwrss: f64,
    pub ldl2: f64,
    pub ldrx2: f64,
    /// `(X'X - sum W' A^-1 W)^-1`; `cov(beta) = sigma2 * this`.
    pub rxtrx_inv: Vec<f64>,
}

/// Packed lower-triangle index -> `(row, col)`, column-major within the
/// triangle, which is lme4's convention for `theta`.
pub fn theta_index(q: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            v.push((r, c));
        }
    }
    v
}

pub fn n_theta(q: usize) -> usize {
    q * (q + 1) / 2
}

pub fn theta_to_lambda(theta: &[f64], q: usize) -> Vec<f64> {
    let mut lam = vec![0.0; q * q];
    for (k, &(r, c)) in theta_index(q).iter().enumerate() {
        lam[r * q + c] = theta[k];
    }
    lam
}

/// Per-group quantities carried from the first pass to the second.
struct Block {
    ainv: Vec<f64>,  // q*q
    m_mat: Vec<f64>, // q*q  (= Z'Z Lambda)
    b: Vec<f64>,     // q*p  (= A^-1 W)
    rzx: Vec<f64>,   // q*p
    cu: Vec<f64>,    // q
    l: Vec<f64>,     // q*q
}

struct Pass1 {
    block: Block,
    ldl2: f64,
    acc_pp: Vec<f64>,
    acc_p: Vec<f64>,
}

struct Pass2 {
    u: Vec<f64>,
    uv: f64,
    grad_ld: Vec<f64>,
    grad_pw: Vec<f64>,
}

/// Evaluate the profiled criterion, and optionally its gradient, at `theta`.
///
/// Returns `None` when `theta` is infeasible (a Cholesky fails, or the penalised
/// residual sum of squares is non-positive) -- the optimiser reads that as `+inf`.
pub fn evaluate(d: &LmmData, theta: &[f64], reml: bool, want_grad: bool) -> Option<Eval> {
    let (n, p, q, m) = (d.n, d.p, d.q, d.m);
    let lam = theta_to_lambda(theta, q);
    let tix = theta_index(q);
    let nth = tix.len();

    // ---- Pass 1: factorise each block, accumulate the fixed-effect system.
    let pass1: Option<Vec<Pass1>> = (0..m)
        .into_par_iter()
        .map(|i| -> Option<Pass1> {
            let ztz_i = &d.ztz[i * q * q..(i + 1) * q * q];
            let ztx_i = &d.ztx[i * q * p..(i + 1) * q * p];
            let zty_i = &d.zty[i * q..(i + 1) * q];

            // M = Z'Z * Lambda           (q x q)
            let mut m_mat = vec![0.0; q * q];
            matmul(ztz_i, &lam, &mut m_mat, q, q, q);

            // A = Lambda' * M + I        (q x q)
            let mut a = vec![0.0; q * q];
            matmul_at(&lam, &m_mat, &mut a, q, q, q);
            for j in 0..q {
                a[j * q + j] += 1.0;
            }

            let mut l = a;
            cholesky(&mut l, q)?;
            let ldl2 = 2.0 * log_diag_sum(&l, q);

            let mut ainv = vec![0.0; q * q];
            chol_inverse(&l, &mut ainv, q);

            // W = Lambda' * Z'X          (q x p)
            let mut w = vec![0.0; q * p];
            matmul_at(&lam, ztx_i, &mut w, q, q, p);

            // B = A^-1 * W               (q x p)
            let mut b = vec![0.0; q * p];
            matmul(&ainv, &w, &mut b, q, q, p);

            // RZX = L^-1 * W             (q x p)
            let mut rzx = w;
            trsm_lower(&l, &mut rzx, q, p);

            // cu = L^-1 * Lambda' * Z'y  (q)
            let mut cu = vec![0.0; q];
            matmul_at(&lam, zty_i, &mut cu, q, q, 1);
            trsm_lower(&l, &mut cu, q, 1);

            let mut acc_pp = vec![0.0; p * p];
            matmul_at(&rzx, &rzx, &mut acc_pp, q, p, p);
            let mut acc_p = vec![0.0; p];
            matmul_at(&rzx, &cu, &mut acc_p, q, p, 1);

            Some(Pass1 {
                block: Block { ainv, m_mat, b, rzx, cu, l },
                ldl2,
                acc_pp,
                acc_p,
            })
        })
        .collect();
    let pass1 = pass1?;

    let mut ldl2 = 0.0;
    let mut sum_pp = vec![0.0; p * p];
    let mut sum_p = vec![0.0; p];
    for r in &pass1 {
        ldl2 += r.ldl2;
        for j in 0..p * p {
            sum_pp[j] += r.acc_pp[j];
        }
        for j in 0..p {
            sum_p[j] += r.acc_p[j];
        }
    }

    // ---- Fixed effects: (X'X - sum RZX'RZX) beta = X'y - sum RZX'cu
    let mut rxtrx = vec![0.0; p * p];
    for j in 0..p * p {
        rxtrx[j] = d.xtx[j] - sum_pp[j];
    }
    let mut rx = rxtrx;
    cholesky(&mut rx, p)?;
    let ldrx2 = 2.0 * log_diag_sum(&rx, p);

    let mut beta: Vec<f64> = (0..p).map(|j| d.xty[j] - sum_p[j]).collect();
    trsm_lower(&rx, &mut beta, p, 1);
    trsm_lower_t(&rx, &mut beta, p, 1);

    let mut rxtrx_inv = vec![0.0; p * p];
    chol_inverse(&rx, &mut rxtrx_inv, p);

    // ---- Pass 2: random effects, pwrss, and both gradient contributions.
    //
    // The pwrss gradient is accumulated UNSCALED here; the dfree/pwrss factor is
    // applied afterwards, since pwrss is not known until this pass has summed.
    let pass2: Vec<Pass2> = (0..m)
        .into_par_iter()
        .map(|i| {
            let blk = &pass1[i].block;
            let ztx_i = &d.ztx[i * q * p..(i + 1) * q * p];
            let zty_i = &d.zty[i * q..(i + 1) * q];

            // u = L^-T (cu - RZX beta)
            let mut u = vec![0.0; q];
            for r in 0..q {
                let mut s = blk.cu[r];
                for c in 0..p {
                    s -= blk.rzx[r * p + c] * beta[c];
                }
                u[r] = s;
            }
            trsm_lower_t(&blk.l, &mut u, q, 1);

            // pwrss contribution: u . (Lambda' Z'y)
            let mut v = vec![0.0; q];
            matmul_at(&lam, zty_i, &mut v, q, q, 1);
            let uv = (0..q).map(|r| u[r] * v[r]).sum::<f64>();

            let mut grad_ld = vec![0.0; if want_grad { nth } else { 0 }];
            let mut grad_pw = vec![0.0; if want_grad { nth } else { 0 }];

            if want_grad {
                // G = M * A^-1                     -> d(ldL2)
                let mut g = vec![0.0; q * q];
                matmul(&blk.m_mat, &blk.ainv, &mut g, q, q, q);

                // t = Z'y - Z'X beta - M u         -> d(pwrss)
                let mut t = vec![0.0; q];
                for r in 0..q {
                    let mut s = zty_i[r];
                    for c in 0..p {
                        s -= ztx_i[r * p + c] * beta[c];
                    }
                    for c in 0..q {
                        s -= blk.m_mat[r * q + c] * u[c];
                    }
                    t[r] = s;
                }

                // REML-only: BP = B P (q x p), BPBt = BP B' (q x q)
                let mut bp = Vec::new();
                let mut bpbt = Vec::new();
                if reml {
                    bp = vec![0.0; q * p];
                    matmul(&blk.b, &rxtrx_inv, &mut bp, q, p, p);
                    bpbt = vec![0.0; q * q];
                    for r in 0..q {
                        for c in 0..q {
                            let mut s = 0.0;
                            for x in 0..p {
                                s += bp[r * p + x] * blk.b[c * p + x];
                            }
                            bpbt[r * q + c] = s;
                        }
                    }
                }

                for (k, &(r, c)) in tix.iter().enumerate() {
                    let mut gk = 2.0 * g[r * q + c];
                    if reml {
                        let mut s1 = 0.0;
                        for x in 0..p {
                            s1 += ztx_i[r * p + x] * bp[c * p + x];
                        }
                        let mut s2 = 0.0;
                        for x in 0..q {
                            s2 += blk.m_mat[r * q + x] * bpbt[x * q + c];
                        }
                        gk -= 2.0 * (s1 - s2);
                    }
                    grad_ld[k] = gk;
                    grad_pw[k] = -2.0 * u[c] * t[r];
                }
            }

            Pass2 { u, uv, grad_ld, grad_pw }
        })
        .collect();

    let mut u_all = vec![0.0; m * q];
    let mut sum_uv = 0.0;
    let mut grad_ld = vec![0.0; nth];
    let mut grad_pw = vec![0.0; nth];
    for (i, r) in pass2.iter().enumerate() {
        u_all[i * q..(i + 1) * q].copy_from_slice(&r.u);
        sum_uv += r.uv;
        if want_grad {
            for k in 0..nth {
                grad_ld[k] += r.grad_ld[k];
                grad_pw[k] += r.grad_pw[k];
            }
        }
    }

    let beta_xty: f64 = (0..p).map(|j| beta[j] * d.xty[j]).sum();
    let pwrss = d.yty - beta_xty - sum_uv;
    if !(pwrss > 0.0) || !pwrss.is_finite() {
        return None;
    }

    let dfree = if reml { (n - p) as f64 } else { n as f64 };
    let deviance = ldl2
        + if reml { ldrx2 } else { 0.0 }
        + dfree * (1.0 + LOG_2PI + (pwrss / dfree).ln());

    let mut grad = vec![0.0; nth];
    if want_grad {
        let scale = dfree / pwrss;
        for k in 0..nth {
            grad[k] = grad_ld[k] + scale * grad_pw[k];
        }
    }

    Some(Eval {
        deviance,
        grad,
        beta,
        u: u_all,
        sigma2: pwrss / dfree,
        pwrss,
        ldl2,
        ldrx2: if reml { ldrx2 } else { 0.0 },
        rxtrx_inv,
    })
}
