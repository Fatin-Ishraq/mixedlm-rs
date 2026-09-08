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

/// Per-group state lives in one flat buffer of `m * BlockLayout::stride`
/// doubles, allocated once per evaluation rather than as ~10 small `Vec`s per
/// group. At 125,000 groups and 10 objective evaluations that is the difference
/// between a handful of allocations and roughly twelve million of them.
#[derive(Clone, Copy)]
struct BlockLayout {
    qq: usize,
    qp: usize,
    stride: usize,
}

impl BlockLayout {
    fn new(q: usize, p: usize) -> Self {
        let (qq, qp) = (q * q, q * p);
        // m_mat | l | rzx | cu
        //
        // A^-1 and B = A^-1 W are deliberately NOT stored. Because A = L L',
        //     A^-1 = L^-T L^-1   and   B = A^-1 W = L^-T (L^-1 W) = L^-T rzx,
        // so both are recoverable in pass 2 from `l` and `rzx` alone. The extra
        // arithmetic is trivial and this evaluation is memory-bound, so trading
        // q*q + q*p of traffic per group for a triangular solve is a clear win.
        // A deviance-only call now never forms A^-1 at all.
        Self { qq, qp, stride: 2 * qq + qp + q }
    }

    /// Split one group's slice into its four named pieces.
    fn split<'a>(
        &self,
        buf: &'a mut [f64],
    ) -> (&'a mut [f64], &'a mut [f64], &'a mut [f64], &'a mut [f64]) {
        let (m_mat, rest) = buf.split_at_mut(self.qq);
        let (l, rest) = rest.split_at_mut(self.qq);
        let (rzx, cu) = rest.split_at_mut(self.qp);
        (m_mat, l, rzx, cu)
    }

    fn split_ref<'a>(
        &self,
        buf: &'a [f64],
    ) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        let (m_mat, rest) = buf.split_at(self.qq);
        let (l, rest) = rest.split_at(self.qq);
        let (rzx, cu) = rest.split_at(self.qp);
        (m_mat, l, rzx, cu)
    }
}

/// Fold accumulator for pass 1. One per rayon thread, not one per group.
struct Acc1 {
    ok: bool,
    ldl2: f64,
    pp: Vec<f64>,
    p: Vec<f64>,
}

/// Fold accumulator for pass 2.
struct Acc2 {
    uv: f64,
    grad_ld: Vec<f64>,
    grad_pw: Vec<f64>,
    ainv: Vec<f64>,
    b: Vec<f64>,
    bp: Vec<f64>,
    bpbt: Vec<f64>,
    g: Vec<f64>,
    t: Vec<f64>,
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
    //
    // Every intermediate is written straight into the group's slice of `blocks`,
    // so no per-group allocation happens at all. The accumulators live in the
    // fold state, which rayon creates once per thread.
    let lay = BlockLayout::new(q, p);
    let mut blocks = vec![0.0f64; m * lay.stride];

    let acc = blocks
        .par_chunks_mut(lay.stride)
        .enumerate()
        .fold(
            || Acc1 { ok: true, ldl2: 0.0, pp: vec![0.0; p * p], p: vec![0.0; p] },
            |mut acc, (i, buf)| {
                if !acc.ok {
                    return acc;
                }
                let ztz_i = &d.ztz[i * q * q..(i + 1) * q * q];
                let ztx_i = &d.ztx[i * q * p..(i + 1) * q * p];
                let zty_i = &d.zty[i * q..(i + 1) * q];
                let (m_mat, l, rzx, cu) = lay.split(buf);

                // M = Z'Z * Lambda
                matmul(ztz_i, &lam, m_mat, q, q, q);
                // A = Lambda' * M + I, built directly in `l` and factorised there
                matmul_at(&lam, m_mat, l, q, q, q);
                for j in 0..q {
                    l[j * q + j] += 1.0;
                }
                if cholesky(l, q).is_none() {
                    acc.ok = false;
                    return acc;
                }
                acc.ldl2 += 2.0 * log_diag_sum(l, q);

                // W = Lambda' Z'X staged in `rzx`, then rzx <- L^-1 W in place
                matmul_at(&lam, ztx_i, rzx, q, q, p);
                trsm_lower(l, rzx, q, p);

                // cu = L^-1 Lambda' Z'y, in place
                matmul_at(&lam, zty_i, cu, q, q, 1);
                trsm_lower(l, cu, q, 1);

                // Accumulate RZX'RZX and RZX'cu without forming a temporary.
                for a in 0..p {
                    let mut sp = 0.0;
                    for r in 0..q {
                        sp += rzx[r * p + a] * cu[r];
                    }
                    acc.p[a] += sp;
                    for bb in a..p {
                        let mut s = 0.0;
                        for r in 0..q {
                            s += rzx[r * p + a] * rzx[r * p + bb];
                        }
                        acc.pp[a * p + bb] += s;
                        if bb != a {
                            acc.pp[bb * p + a] += s;
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || Acc1 { ok: true, ldl2: 0.0, pp: vec![0.0; p * p], p: vec![0.0; p] },
            |mut a, b| {
                a.ok &= b.ok;
                a.ldl2 += b.ldl2;
                for j in 0..p * p {
                    a.pp[j] += b.pp[j];
                }
                for j in 0..p {
                    a.p[j] += b.p[j];
                }
                a
            },
        );

    if !acc.ok {
        return None;
    }
    let ldl2 = acc.ldl2;
    let sum_pp = acc.pp;
    let sum_p = acc.p;

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
    // As in pass 1, scratch lives in the fold state -- one set per thread rather
    // than one per group.
    let mut u_all = vec![0.0f64; m * q];

    let acc2 = blocks
        .par_chunks(lay.stride)
        .zip(u_all.par_chunks_mut(q))
        .enumerate()
        .fold(
            || Acc2 {
                uv: 0.0,
                grad_ld: vec![0.0; nth],
                grad_pw: vec![0.0; nth],
                ainv: vec![0.0; q * q],
                b: vec![0.0; q * p],
                bp: vec![0.0; q * p],
                bpbt: vec![0.0; q * q],
                g: vec![0.0; q * q],
                t: vec![0.0; q],
            },
            |mut acc, (i, (buf, u))| {
                let ztx_i = &d.ztx[i * q * p..(i + 1) * q * p];
                let zty_i = &d.zty[i * q..(i + 1) * q];
                let (m_mat, l, rzx, cu) = lay.split_ref(buf);

                // u = L^-T (cu - RZX beta)
                for r in 0..q {
                    let mut s = cu[r];
                    for c in 0..p {
                        s -= rzx[r * p + c] * beta[c];
                    }
                    u[r] = s;
                }
                trsm_lower_t(l, u, q, 1);

                // pwrss contribution: u . (Lambda' Z'y), formed without a temp
                for r in 0..q {
                    let mut v = 0.0;
                    for x in 0..q {
                        v += lam[x * q + r] * zty_i[x];
                    }
                    acc.uv += u[r] * v;
                }

                if want_grad {
                    // A^-1 = L^-T L^-1, rebuilt here rather than stored
                    chol_inverse(l, &mut acc.ainv, q);
                    // G = M * A^-1                     -> d(ldL2)
                    matmul(m_mat, &acc.ainv, &mut acc.g, q, q, q);

                    // t = Z'y - Z'X beta - M u         -> d(pwrss)
                    for r in 0..q {
                        let mut s = zty_i[r];
                        for c in 0..p {
                            s -= ztx_i[r * p + c] * beta[c];
                        }
                        for c in 0..q {
                            s -= m_mat[r * q + c] * u[c];
                        }
                        acc.t[r] = s;
                    }

                    if reml {
                        // B = A^-1 W = L^-T (L^-1 W) = L^-T rzx
                        acc.b.copy_from_slice(rzx);
                        trsm_lower_t(l, &mut acc.b, q, p);
                        matmul(&acc.b, &rxtrx_inv, &mut acc.bp, q, p, p);
                        for r in 0..q {
                            for c in 0..q {
                                let mut s = 0.0;
                                for x in 0..p {
                                    s += acc.bp[r * p + x] * acc.b[c * p + x];
                                }
                                acc.bpbt[r * q + c] = s;
                            }
                        }
                    }

                    for (k, &(r, c)) in tix.iter().enumerate() {
                        let mut gk = 2.0 * acc.g[r * q + c];
                        if reml {
                            let mut s1 = 0.0;
                            for x in 0..p {
                                s1 += ztx_i[r * p + x] * acc.bp[c * p + x];
                            }
                            let mut s2 = 0.0;
                            for x in 0..q {
                                s2 += m_mat[r * q + x] * acc.bpbt[x * q + c];
                            }
                            gk -= 2.0 * (s1 - s2);
                        }
                        acc.grad_ld[k] += gk;
                        acc.grad_pw[k] += -2.0 * u[c] * acc.t[r];
                    }
                }
                acc
            },
        )
        .reduce(
            || Acc2 {
                uv: 0.0,
                grad_ld: vec![0.0; nth],
                grad_pw: vec![0.0; nth],
                ainv: Vec::new(),
                b: Vec::new(),
                bp: Vec::new(),
                bpbt: Vec::new(),
                g: Vec::new(),
                t: Vec::new(),
            },
            |mut a, b| {
                a.uv += b.uv;
                for k in 0..nth {
                    a.grad_ld[k] += b.grad_ld[k];
                    a.grad_pw[k] += b.grad_pw[k];
                }
                a
            },
        );

    let sum_uv = acc2.uv;
    let grad_ld = acc2.grad_ld;
    let grad_pw = acc2.grad_pw;

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
