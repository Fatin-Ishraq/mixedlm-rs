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
//! The gradient of the profiled criterion is not itself novel: Bates et al.
//! derive the ML version in the lme4 paper (eq. 46-48), and MixedModels.jl
//! documents derivative support. What is here is a REML gradient specialised
//! to the single-grouping-factor block structure and evaluated in the two
//! passes that already produce the criterion. What differs in practice is
//! that the optimiser *uses* it, where both of those optimise derivative-free.
//!
//! # The augmented Schur form, and the kernels built on it
//!
//! Write `v_i = [Z_i'X | Z_i'y]` (`q x (p+1)`), `S_i = Z_i'Z_i`, and
//! `P_i = Lambda A_i^-1 Lambda'`. Then the profiled `(p+1) x (p+1)` system is
//!
//! ```text
//! K = [X y]'[X y] - sum_i v_i' P_i v_i  =  [ H   h ]
//!                                          [ h'  c ]
//! beta = H^-1 h,   pwrss = c - h' beta,   ldRX2 = log|H|,
//! ```
//!
//! and, because `beta` minimises `e' K e` over `e = [-beta, 1]`,
//!
//! ```text
//! d(pwrss)/dtheta_k = e' dK_k e,       d(ldRX2)/dtheta_k = tr(H^-1 dH_k),
//! dP_k = G_k = F_k + F_k',   F_k = e_r bc' - bc (S P)[r,:],   bc = (Lambda A^-1)[:,c].
//! ```
//!
//! `P_i` depends on the group only through `S_i`. Groups with bit-identical
//! `S_i` therefore contribute `sum_ab P_ab T[a,b]` with `T[a,b] = sum_i
//! v_i[a]' v_i[b]` summed over the class -- an exact algebraic rearrangement,
//! not an approximation. For a balanced random intercept there is exactly one
//! class, and the per-evaluation cost stops depending on the number of groups.
//!
//! Three kernels evaluate the same criterion:
//!
//! * [`Kernel::Blocks`] -- the two-pass evaluator above, storing one block per
//!   group between the passes. The general-purpose default.
//! * [`Kernel::Classes`] -- the class aggregation, chosen automatically when
//!   the number of distinct `Z_i'Z_i` is small enough to pay for itself.
//! * [`Kernel::Streaming`] -- the augmented form one group at a time: a single
//!   pass accumulating `K` and every `dK_k`, with no per-group buffer at all.
//!   More arithmetic per group, so it is opt-in rather than dispatched.
//!
//! Every term is checked against central finite differences -- and against an
//! independent dense implementation in tests/test_dense_reference.py, which
//! shares no code with this one -- and the kernels are checked against each
//! other in tests/test_kernels.py.

use crate::linalg::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Mutex;

const LOG_2PI: f64 = 1.837_877_066_409_345_5;

/// Approximate floating-point work in one chunk of groups.
///
/// Chunks are sized from the problem dimensions alone, never from the thread
/// count, and their results are summed in index order. The criterion is
/// therefore bit-for-bit identical on 1 thread and on 64, which the previous
/// rayon `fold`/`reduce` did not guarantee: its reduction tree followed the
/// work-stealing splits, and on one fixture a thread-count change moved the
/// optimiser from 9 evaluations to 20.
const CHUNK_FLOPS: usize = 1 << 15;
/// Below this much total work a direct serial loop beats scheduling tasks.
const PAR_MIN_FLOPS: usize = 1 << 18;
/// Idle block buffers kept for reuse between evaluations on one core.
const MAX_POOLED: usize = 4;

/// What an evaluation must produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The criterion alone: one pass, no per-group storage, no inverses.
    Criterion,
    /// Criterion and gradient, as the optimiser needs them.
    Gradient,
    /// Everything, including every group's random effects and `cov(beta)`.
    Full,
}

impl Mode {
    fn grad(self) -> bool {
        !matches!(self, Mode::Criterion)
    }
}

/// Which evaluator computes the criterion. All three are exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Blocks,
    Streaming,
    Classes,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Blocks => "blocks",
            Kernel::Streaming => "streaming",
            Kernel::Classes => "aggregated",
        }
    }
}

/// How the constructor should pick a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRequest {
    Auto,
    Force(Kernel),
}

/// Groups collapsed by bit-identical `Z_i'Z_i`.
pub struct Classes {
    pub k: usize,
    /// Number of groups in each class.
    pub count: Vec<f64>,
    /// `k * q * q`: the shared `Z'Z`.
    pub s: Vec<f64>,
    /// `k * q*q * (p+1)*(p+1)`: `T[a,b] = sum_i v_i[a]' v_i[b]`, upper triangle
    /// in the trailing `(p+1) x (p+1)` block (the lower one is left zero).
    pub t: Vec<f64>,
    /// `m`: the class of each group, kept so a new response can reuse it.
    pub of_group: Vec<u32>,
}

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
    pub classes: Option<Classes>,
    pub kernel: Kernel,
    tix: Vec<(usize, usize)>,
    pool: Mutex<Vec<Vec<f64>>>,
    codes_digest: u64,
}

/// Result of one objective evaluation.
pub struct Eval {
    pub deviance: f64,
    /// Empty in [`Mode::Criterion`].
    pub grad: Vec<f64>,
    pub beta: Vec<f64>,
    /// `m*q`, spherical scale. Empty unless [`Mode::Full`].
    pub u: Vec<f64>,
    pub sigma2: f64,
    pub pwrss: f64,
    pub ldl2: f64,
    pub ldrx2: f64,
    /// `(X'X - sum W' A^-1 W)^-1`; `cov(beta) = sigma2 * this`. Empty unless
    /// [`Mode::Full`].
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

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

/// `(groups per chunk, run in parallel)` for `units` items of `per_unit` work.
fn plan(units: usize, per_unit: usize) -> (usize, bool) {
    let per = per_unit.max(1);
    // At least 64 units: every chunk carries its own accumulators, which are
    // p x p for the fixed-effect system, and tiny chunks spend more on those
    // than on the groups themselves.
    let chunk = (CHUNK_FLOPS / per).clamp(64, 1 << 16);
    let parallel = units > chunk
        && units.saturating_mul(per) >= PAR_MIN_FLOPS
        && rayon::current_num_threads() > 1;
    (chunk, parallel)
}

/// Map `parts` in order, in parallel or not. The output order is the input
/// order either way, so summing it afterwards is deterministic.
fn map_parts<P, R, F>(parallel: bool, parts: Vec<P>, f: F) -> Vec<R>
where
    P: Send,
    R: Send,
    F: Fn(P) -> R + Sync + Send,
{
    if parallel {
        parts.into_par_iter().map(f).collect()
    } else {
        parts.into_iter().map(f).collect()
    }
}

/// Run `f(q_literal)` with `q` as a compile-time constant for the common sizes,
/// so the inlined `q x q` loops unroll. The arithmetic is identical in every
/// arm; only the code generation differs.
macro_rules! specialise_q {
    ($q:expr, |$qq:ident| $body:expr) => {
        match $q {
            1 => {
                let $qq: usize = 1;
                $body
            }
            2 => {
                let $qq: usize = 2;
                $body
            }
            3 => {
                let $qq: usize = 3;
                $body
            }
            _ => {
                let $qq: usize = $q;
                $body
            }
        }
    };
}

/// A Neumaier-compensated running sum.
///
/// The one-pass criterion is `y'y - sum ||cu_i||^2 - h' H^-1 h`: large sums
/// that cancel at the very end. Plain accumulation leaves a few ulps of
/// theta-dependent rounding in the deviance, which is enough to turn the last
/// sub-ulp step of a converged line search uphill and make L-BFGS-B report an
/// abnormal stop after dozens of wasted evaluations. Compensation keeps the
/// accumulated error near one ulp of the result.
#[derive(Clone, Copy, Default)]
struct Csum {
    s: f64,
    c: f64,
}

impl Csum {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        let t = self.s + x;
        let z = t - self.s;
        self.c += (self.s - (t - z)) + (x - z);
        self.s = t;
    }
    #[inline(always)]
    fn merge(&mut self, o: Csum) {
        self.add(o.s);
        self.c += o.c;
    }
    #[inline(always)]
    fn get(self) -> f64 {
        self.s + self.c
    }
    /// `a - self`, subtracting the leading part first so that a nearly equal
    /// `a` cancels exactly.
    #[inline(always)]
    fn subtracted_from(self, a: f64) -> f64 {
        (a - self.s) - self.c
    }
}

fn merge_into(dst: &mut [Csum], src: &[Csum]) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge(*s);
    }
}

fn add_into(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

fn mirror_upper(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            a[i * n + j] = a[j * n + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// A small multiplicative hasher for fixed-width bit patterns; SipHash is
/// needlessly slow for keys nobody controls adversarially.
#[derive(Default)]
struct BitsHasher(u64);

impl Hasher for BitsHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Keys are arrays of u64, which hash as one byte slice: take them a
        // word at a time rather than a byte at a time.
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(w);
            self.write_u64(u64::from_le_bytes(buf));
        }
        for &b in words.remainder() {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }
}

type BitsMap<K> = HashMap<K, u32, BuildHasherDefault<BitsHasher>>;

/// Class id per group, and one representative per class.
///
/// `None` once more than `limit` classes appear. With a finite `limit` it also
/// gives up once a sizeable prefix of the groups is mostly distinct, so a
/// design with no repetition -- a continuous random slope -- costs a few
/// thousand insertions rather than `limit`. Only the choice of kernel depends
/// on either rule; every kernel computes the same criterion.
fn assign_classes<K: Hash + Eq>(
    m: usize,
    limit: usize,
    key: impl Fn(usize) -> K,
) -> Option<(Vec<u32>, Vec<usize>)> {
    // A balanced design -- every group alike -- needs no map at all.
    let first = key(0);
    let mut run = 1;
    while run < m && key(run) == first {
        run += 1;
    }
    let mut of_group = vec![0u32; run];
    let mut reps = vec![0usize];
    if run == m {
        return Some((of_group, reps));
    }
    let mut map: BitsMap<K> = HashMap::default();
    map.insert(first, 0);
    of_group.reserve(m - run);
    let bounded = limit != usize::MAX;
    for i in run..m {
        if bounded && i >= 4096 && i % 1024 == 0 && reps.len() * 4 > i {
            return None;
        }
        let next = reps.len() as u32;
        let id = *map.entry(key(i)).or_insert(next);
        if id == next {
            reps.push(i);
            if reps.len() > limit {
                return None;
            }
        }
        of_group.push(id);
    }
    Some((of_group, reps))
}

/// Rough flops per group for one gradient evaluation of each kernel. Used for
/// dispatch and chunk sizing only; nothing numerical depends on it.
fn blocks_cost(q: usize, p: usize) -> usize {
    3 * q * q * q + 3 * q * q * p + 2 * q * p * p + n_theta(q) * (p + q) + 8
}

fn class_cost(q: usize, p: usize) -> usize {
    let p1 = p + 1;
    4 * q * q * q + (1 + n_theta(q)) * (q * q * (p1 * (p1 + 1) / 2) + 2 * q * q) + 8
}

fn streaming_cost(q: usize, p: usize) -> usize {
    let p1 = p + 1;
    4 * q * q * q + (1 + n_theta(q)) * (q * q * p1 + q * p1 * (p1 + 1) / 2 + 2 * q * q) + 8
}

impl Classes {
    fn build(
        q: usize,
        p: usize,
        m: usize,
        ztz: &[f64],
        ztx: &[f64],
        zty: &[f64],
        limit: usize,
    ) -> Option<Classes> {
        let qq = q * q;
        // Z'Z is exactly symmetric (the constructor mirrors one triangle), so
        // the lower triangle identifies it.
        let (of_group, reps) = if q <= 3 {
            assign_classes(m, limit, |i| {
                let mut key = [0u64; 6];
                let mut j = 0;
                for c in 0..q {
                    for r in c..q {
                        key[j] = ztz[i * qq + r * q + c].to_bits();
                        j += 1;
                    }
                }
                key
            })
        } else {
            assign_classes(m, limit, |i| {
                let mut key = Vec::with_capacity(n_theta(q));
                for c in 0..q {
                    for r in c..q {
                        key.push(ztz[i * qq + r * q + c].to_bits());
                    }
                }
                key
            })
        }?;
        Some(Classes::with_assignment(
            q, p, ztz, ztx, zty, of_group, &reps,
        ))
    }

    /// Sum the class statistics for a known assignment.
    #[allow(clippy::too_many_arguments)]
    fn with_assignment(
        q: usize,
        p: usize,
        ztz: &[f64],
        ztx: &[f64],
        zty: &[f64],
        of_group: Vec<u32>,
        reps: &[usize],
    ) -> Classes {
        let qq = q * q;
        let k = reps.len();
        let p1 = p + 1;
        let blk = qq * p1 * p1;
        let m = of_group.len();
        let mut s = vec![0.0; k * qq];
        for (c, &i) in reps.iter().enumerate() {
            s[c * qq..(c + 1) * qq].copy_from_slice(&ztz[i * qq..(i + 1) * qq]);
        }

        // Fixed chunks of groups, each summed in order and the chunk totals
        // added in order: the same bits serially or in parallel. A per-chunk
        // copy of the class statistics is only affordable while classes are few.
        let (chunk, parallel) = plan(m, qq * p1 * (p1 + 1) / 2 + q * p1);
        let parallel = parallel && k * blk <= 1 << 16;
        let starts: Vec<usize> = (0..m).step_by(chunk).collect();
        let of = &of_group;
        let parts = map_parts(parallel, starts, |g0| {
            let g1 = (g0 + chunk).min(m);
            specialise_q!(q, |qc| class_sums(of, ztx, zty, g0, g1, k, qc, p))
        });
        let mut t = vec![0.0; k * blk];
        let mut count = vec![0.0; k];
        for (pt, pc) in &parts {
            add_into(&mut t, pt);
            add_into(&mut count, pc);
        }
        drop(parts);
        Classes {
            k,
            count,
            s,
            t,
            of_group,
        }
    }

    /// One representative group per class.
    fn representatives(&self) -> Vec<usize> {
        let mut reps = vec![usize::MAX; self.k];
        for (i, &c) in self.of_group.iter().enumerate() {
            let c = c as usize;
            if reps[c] == usize::MAX {
                reps[c] = i;
            }
        }
        reps
    }
}

/// Class statistics `(T, count)` for groups `g0..g1`, summed in order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn class_sums(
    of_group: &[u32],
    ztx: &[f64],
    zty: &[f64],
    g0: usize,
    g1: usize,
    k: usize,
    q: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let p1 = p + 1;
    let blk = q * q * p1 * p1;
    let mut t = vec![0.0; k * blk];
    let mut count = vec![0.0; k];
    let mut v = vec![0.0; q * p1];
    for (i, &c) in of_group.iter().enumerate().take(g1).skip(g0) {
        let c = c as usize;
        count[c] += 1.0;
        fill_v(&mut v, ztx, zty, i, q, p);
        let tc = &mut t[c * blk..(c + 1) * blk];
        for a in 0..q {
            let va = &v[a * p1..(a + 1) * p1];
            for b in 0..q {
                let vb = &v[b * p1..(b + 1) * p1];
                let dst = &mut tc[(a * q + b) * p1 * p1..(a * q + b + 1) * p1 * p1];
                for (x, &vax) in va.iter().enumerate() {
                    let row = &mut dst[x * p1 + x..(x + 1) * p1];
                    for (dv, vby) in row.iter_mut().zip(&vb[x..]) {
                        *dv += vax * vby;
                    }
                }
            }
        }
    }
    (t, count)
}

fn digest_codes(codes: &[i64]) -> u64 {
    let mut h = BitsHasher(codes.len() as u64);
    for &c in codes {
        h.write_u64(c as u64);
    }
    h.finish()
}

/// `X'X` (one triangle, mirrored; skipped unless `with_xtx`), `X'y` and `y'y`,
/// summed over fixed-size row chunks in order.
fn fixed_products(y: &[f64], x: &[f64], p: usize, with_xtx: bool) -> (Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let row_work = p * (p + 1) / 2 + p + 1;
    let (row_chunk, par_rows) = plan(n, row_work);
    let ranges: Vec<(usize, usize)> = (0..n)
        .step_by(row_chunk)
        .map(|r0| (r0, (r0 + row_chunk).min(n)))
        .collect();
    let parts = map_parts(par_rows, ranges, |(r0, r1)| {
        let mut xtx = vec![0.0; if with_xtx { p * p } else { 0 }];
        let mut xty = vec![0.0; p];
        let mut yty = 0.0;
        for r in r0..r1 {
            let yr = y[r];
            yty += yr * yr;
            let xrow = &x[r * p..r * p + p];
            for a in 0..p {
                let xa = xrow[a];
                xty[a] += xa * yr;
                if with_xtx {
                    let dst = &mut xtx[a * p + a..a * p + p];
                    for (dv, xb) in dst.iter_mut().zip(&xrow[a..]) {
                        *dv += xa * xb;
                    }
                }
            }
        }
        (xtx, xty, yty)
    });
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    let mut yty = 0.0;
    for (a, b, c) in parts {
        for j in 0..a.len() {
            xtx[j] += a[j];
        }
        for j in 0..p {
            xty[j] += b[j];
        }
        yty += c;
    }
    mirror_upper(&mut xtx, p);
    (xtx, xty, yty)
}

/// `v_i = [Z_i'X | Z_i'y]`, row-major `q x (p+1)`.
#[inline(always)]
fn fill_v(v: &mut [f64], ztx: &[f64], zty: &[f64], i: usize, q: usize, p: usize) {
    let p1 = p + 1;
    for a in 0..q {
        v[a * p1..a * p1 + p].copy_from_slice(&ztx[i * q * p + a * p..i * q * p + (a + 1) * p]);
        v[a * p1 + p] = zty[i * q + a];
    }
}

impl LmmData {
    /// Form every cross-product in one pass over the rows.
    ///
    /// Row-major `x` (`n x p`) and `z` (`n x q`). The fixed-effect products are
    /// summed over fixed-size row chunks; the group products are accumulated
    /// group by group, in row order within a group -- in parallel over disjoint
    /// ranges of groups when the data are large enough. Neither depends on the
    /// thread count.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rows(
        y: &[f64],
        x: &[f64],
        z: &[f64],
        codes: &[i64],
        p: usize,
        q: usize,
        m: usize,
        request: KernelRequest,
    ) -> Result<LmmData, String> {
        let n = y.len();
        // The codes are validated and fingerprinted inside whichever pass
        // first reads them, not in a pass of their own.
        let mut digest = BitsHasher(n as u64);
        let out_of_range = |g: i64| format!("group code {g} out of range 0..{m}");

        // ---- Group products, one triangle of Z'Z.
        let (qq, qp) = (q * q, q * p);
        let mut ztz = vec![0.0; m * qq];
        let mut ztx = vec![0.0; m * qp];
        let mut zty = vec![0.0; m * q];
        let grp_work = q * (q + 1) / 2 + qp + q;

        let accumulate_row = |r: usize, ztz_g: &mut [f64], ztx_g: &mut [f64], zty_g: &mut [f64]| {
            let yr = y[r];
            let xrow = &x[r * p..r * p + p];
            let zrow = &z[r * q..r * q + q];
            for a in 0..q {
                let za = zrow[a];
                zty_g[a] += za * yr;
                for (dv, zb) in ztz_g[a * q + a..(a + 1) * q].iter_mut().zip(&zrow[a..]) {
                    *dv += za * zb;
                }
                for (dv, xb) in ztx_g[a * p..(a + 1) * p].iter_mut().zip(xrow) {
                    *dv += za * xb;
                }
            }
        };

        let row_work = p * (p + 1) / 2 + p + 1;
        let serial = n.saturating_mul(grp_work + row_work) < PAR_MIN_FLOPS
            || rayon::current_num_threads() == 1;
        let (xtx, xty, yty) = if serial {
            // One pass over the rows, reading each row once. The fixed-effect
            // sums use the same row chunks as `fixed_products`, so this path
            // and the parallel one below give identical bits.
            let (row_chunk, _) = plan(n, row_work);
            let mut xtx = vec![0.0; p * p];
            let mut xty = vec![0.0; p];
            let mut yty = 0.0;
            let mut cx = vec![0.0; p * p];
            let mut cy = vec![0.0; p];
            for r0 in (0..n).step_by(row_chunk) {
                cx.fill(0.0);
                cy.fill(0.0);
                let mut cyy = 0.0;
                for r in r0..(r0 + row_chunk).min(n) {
                    let yr = y[r];
                    cyy += yr * yr;
                    let xrow = &x[r * p..r * p + p];
                    for a in 0..p {
                        let xa = xrow[a];
                        cy[a] += xa * yr;
                        let dst = &mut cx[a * p + a..a * p + p];
                        for (dv, xb) in dst.iter_mut().zip(&xrow[a..]) {
                            *dv += xa * xb;
                        }
                    }
                    let g = codes[r];
                    if g < 0 || (g as usize) >= m {
                        return Err(out_of_range(g));
                    }
                    digest.write_u64(g as u64);
                    let g = g as usize;
                    accumulate_row(
                        r,
                        &mut ztz[g * qq..(g + 1) * qq],
                        &mut ztx[g * qp..(g + 1) * qp],
                        &mut zty[g * q..(g + 1) * q],
                    );
                }
                add_into(&mut xtx, &cx);
                add_into(&mut xty, &cy);
                yty += cyy;
            }
            mirror_upper(&mut xtx, p);
            (xtx, xty, yty)
        } else {
            // Stable counting sort of rows by group: rows stay in ascending
            // order within a group, so each group's sums are exactly the ones
            // the serial loop forms.
            let mut start = vec![0usize; m + 1];
            for &g in codes {
                if g < 0 || (g as usize) >= m {
                    return Err(out_of_range(g));
                }
                digest.write_u64(g as u64);
                start[g as usize + 1] += 1;
            }
            let fixed = fixed_products(y, x, p, true);
            for g in 0..m {
                start[g + 1] += start[g];
            }
            let mut fill = start.clone();
            let mut order = vec![0usize; n];
            for (r, &g) in codes.iter().enumerate() {
                let g = g as usize;
                order[fill[g]] = r;
                fill[g] += 1;
            }

            let target = (CHUNK_FLOPS / grp_work.max(1)).max(64);
            let mut parts = Vec::new();
            let (mut rest_tz, mut rest_tx, mut rest_ty) =
                (&mut ztz[..], &mut ztx[..], &mut zty[..]);
            let mut g0 = 0;
            while g0 < m {
                let mut g1 = g0 + 1;
                while g1 < m && start[g1] - start[g0] < target {
                    g1 += 1;
                }
                let ng = g1 - g0;
                let (a, b) = std::mem::take(&mut rest_tz).split_at_mut(ng * qq);
                let (c, d) = std::mem::take(&mut rest_tx).split_at_mut(ng * qp);
                let (e, f) = std::mem::take(&mut rest_ty).split_at_mut(ng * q);
                parts.push((g0, g1, a, c, e));
                rest_tz = b;
                rest_tx = d;
                rest_ty = f;
                g0 = g1;
            }
            let start = &start;
            let order = &order;
            map_parts(true, parts, |(g0, g1, tz, tx, ty)| {
                for g in g0..g1 {
                    let j = g - g0;
                    let (tz_g, tx_g, ty_g) = (
                        &mut tz[j * qq..(j + 1) * qq],
                        &mut tx[j * qp..(j + 1) * qp],
                        &mut ty[j * q..(j + 1) * q],
                    );
                    for &r in &order[start[g]..start[g + 1]] {
                        accumulate_row(r, tz_g, tx_g, ty_g);
                    }
                }
            });
            fixed
        };
        for g in 0..m {
            mirror_upper(&mut ztz[g * qq..(g + 1) * qq], q);
        }

        let mut data = LmmData::from_products(n, p, q, m, xtx, xty, yty, ztz, ztx, zty, request);
        data.codes_digest = digest.finish();
        Ok(data)
    }

    /// The same design with a new response.
    ///
    /// Only `X'y`, `y'y`, `Z_i'y` and the response rows of the class
    /// statistics are formed; `X'X`, every `Z_i'Z_i` and `Z_i'X`, and the class
    /// assignment are reused. `x`, `z` and `codes` must be the arrays this core
    /// was built from: the codes are checked against a digest, and the designs
    /// are the caller's responsibility (they are needed only for `X'y` and
    /// `Z'y`). The result is bit-for-bit the core a fresh build would give.
    pub fn with_response(
        &self,
        y: &[f64],
        x: &[f64],
        z: &[f64],
        codes: &[i64],
    ) -> Result<LmmData, String> {
        let (n, p, q, m) = (self.n, self.p, self.q, self.m);
        if y.len() != n || x.len() != n * p || z.len() != n * q || codes.len() != n {
            return Err(format!(
                "the prepared design has {n} rows, {p} fixed and {q} random columns; \
                 the arrays passed do not match it"
            ));
        }
        if digest_codes(codes) != self.codes_digest {
            return Err("the group codes differ from the ones the design was prepared with".into());
        }
        let (_, xty, yty) = fixed_products(y, x, p, false);
        let mut zty = vec![0.0; m * q];
        for r in 0..n {
            let g = codes[r] as usize;
            let yr = y[r];
            for a in 0..q {
                zty[g * q + a] += z[r * q + a] * yr;
            }
        }
        let classes = self.classes.as_ref().map(|c| {
            let reps = c.representatives();
            Classes::with_assignment(q, p, &self.ztz, &self.ztx, &zty, c.of_group.clone(), &reps)
        });
        Ok(LmmData {
            n,
            p,
            q,
            m,
            xtx: self.xtx.clone(),
            xty,
            yty,
            ztz: self.ztz.clone(),
            ztx: self.ztx.clone(),
            zty,
            classes,
            kernel: self.kernel,
            tix: self.tix.clone(),
            pool: Mutex::new(Vec::new()),
            codes_digest: self.codes_digest,
        })
    }

    /// Assemble from already-formed products and choose the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn from_products(
        n: usize,
        p: usize,
        q: usize,
        m: usize,
        xtx: Vec<f64>,
        xty: Vec<f64>,
        yty: f64,
        ztz: Vec<f64>,
        ztx: Vec<f64>,
        zty: Vec<f64>,
        request: KernelRequest,
    ) -> LmmData {
        let (classes, kernel) = match request {
            KernelRequest::Force(Kernel::Classes) => (
                Classes::build(q, p, m, &ztz, &ztx, &zty, usize::MAX),
                Kernel::Classes,
            ),
            KernelRequest::Force(k) => (None, k),
            KernelRequest::Auto => {
                // Aggregate only when the classes are cheap enough to beat the
                // block kernel by a clear margin, counting the class storage.
                let limit = (m * blocks_cost(q, p)) / (2 * class_cost(q, p));
                match Classes::build(q, p, m, &ztz, &ztx, &zty, limit) {
                    Some(c) => (Some(c), Kernel::Classes),
                    None => (None, Kernel::Blocks),
                }
            }
        };
        LmmData {
            n,
            p,
            q,
            m,
            xtx,
            xty,
            yty,
            ztz,
            ztx,
            zty,
            classes,
            kernel,
            tix: theta_index(q),
            pool: Mutex::new(Vec::new()),
            codes_digest: 0,
        }
    }

    fn take_buffer(&self, len: usize) -> Vec<f64> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        match pool.iter().position(|b| b.len() == len) {
            // Every element is overwritten by pass 1 before it is read, so a
            // reused buffer needs no clearing.
            Some(i) => pool.swap_remove(i),
            None => vec![0.0; len],
        }
    }

    fn give_buffer(&self, buf: Vec<f64>) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < MAX_POOLED {
            pool.push(buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared tail: from the augmented system to the criterion and its gradient
// ---------------------------------------------------------------------------

struct Profiled {
    beta: Vec<f64>,
    rx: Vec<f64>,
    ldrx2: f64,
    pwrss: f64,
    dfree: f64,
    deviance: f64,
}

/// Solve the fixed-effect system `H beta = h` and profile out sigma^2.
///
/// `beta` comes in holding `h` and leaves holding `H^-1 h`.
fn profile(
    d: &LmmData,
    hmat: Vec<f64>,
    mut beta: Vec<f64>,
    c: f64,
    ldl2: f64,
    reml: bool,
) -> Option<Profiled> {
    let p = d.p;
    let mut rx = hmat;
    cholesky(&mut rx, p)?;
    let ldrx2 = 2.0 * log_diag_sum(&rx, p);
    trsm_lower(&rx, &mut beta, p, 1);
    // pwrss = c - h' H^-1 h = c - ||L^-1 h||^2. The squared norm comes from the
    // half-solve, so it is non-negative by construction, and the whole
    // criterion needs no second pass over the groups.
    let half: f64 = beta.iter().map(|v| v * v).sum();
    trsm_lower_t(&rx, &mut beta, p, 1);
    let pwrss = c - half;
    // NaN is caught by the first clause; see the note in linalg::cholesky.
    if !pwrss.is_finite() || pwrss <= 0.0 {
        return None;
    }
    let dfree = if reml { (d.n - p) as f64 } else { d.n as f64 };
    let deviance =
        ldl2 + if reml { ldrx2 } else { 0.0 } + dfree * (1.0 + LOG_2PI + (pwrss / dfree).ln());
    Some(Profiled {
        beta,
        rx,
        ldrx2,
        pwrss,
        dfree,
        deviance,
    })
}

/// Evaluate the profiled criterion at `theta`, with whatever `mode` asks for.
///
/// Returns `None` when `theta` is infeasible (a Cholesky fails, or the penalised
/// residual sum of squares is non-positive) -- the optimiser reads that as `+inf`.
pub fn evaluate(d: &LmmData, theta: &[f64], reml: bool, mode: Mode) -> Option<Eval> {
    match d.kernel {
        Kernel::Blocks => evaluate_blocks(d, theta, reml, mode),
        Kernel::Classes => match &d.classes {
            Some(c) => evaluate_augmented(d, &ClassSource { d, c }, theta, reml, mode),
            None => evaluate_blocks(d, theta, reml, mode),
        },
        Kernel::Streaming => evaluate_augmented(d, &GroupSource { d }, theta, reml, mode),
    }
}

// ---------------------------------------------------------------------------
// Kernel::Blocks
// ---------------------------------------------------------------------------

/// Per-group state lives in one flat buffer of `m * stride` doubles, taken
/// from the core's pool rather than allocated per evaluation.
///
/// Layout: `m_mat | l | rzx | cu`. A^-1 and B = A^-1 W are deliberately NOT
/// stored: since A = L L', both are recoverable in pass 2 from `l` and `rzx`.
#[inline(always)]
fn stride(q: usize, p: usize) -> usize {
    2 * q * q + q * p + q
}

struct Acc1 {
    ok: bool,
    ldl2: Csum,
    cu2: Csum,
    pp: Vec<f64>,
    pv: Vec<Csum>,
}

/// Pass 1 over groups `g0..g0+ng`: factorise, accumulate the fixed-effect
/// system. With `store`, each group's intermediates are kept in `bufs`;
/// without, `bufs` is one stride of scratch reused by every group.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn blocks_pass1(
    d: &LmmData,
    lam: &[f64],
    q: usize,
    p: usize,
    g0: usize,
    ng: usize,
    bufs: &mut [f64],
    store: bool,
) -> Acc1 {
    let (qq, qp) = (q * q, q * p);
    let st = stride(q, p);
    let mut acc = Acc1 {
        ok: true,
        ldl2: Csum::default(),
        cu2: Csum::default(),
        pp: vec![0.0; p * p],
        pv: vec![Csum::default(); p],
    };
    for j in 0..ng {
        let i = g0 + j;
        let buf = if store {
            &mut bufs[j * st..(j + 1) * st]
        } else {
            &mut bufs[..st]
        };
        let ztz_i = &d.ztz[i * qq..(i + 1) * qq];
        let ztx_i = &d.ztx[i * qp..(i + 1) * qp];
        let zty_i = &d.zty[i * q..(i + 1) * q];
        let (m_mat, rest) = buf.split_at_mut(qq);
        let (l, rest) = rest.split_at_mut(qq);
        let (rzx, cu) = rest.split_at_mut(qp);

        // M = Z'Z * Lambda
        matmul(ztz_i, lam, m_mat, q, q, q);
        // A = Lambda' * M + I, built directly in `l` and factorised there
        matmul_at(lam, m_mat, l, q, q, q);
        for r in 0..q {
            l[r * q + r] += 1.0;
        }
        if cholesky(l, q).is_none() {
            acc.ok = false;
            return acc;
        }
        acc.ldl2.add(2.0 * log_diag_sum(l, q));

        // W = Lambda' Z'X staged in `rzx`, then rzx <- L^-1 W in place
        matmul_at(lam, ztx_i, rzx, q, q, p);
        trsm_lower(l, rzx, q, p);
        // cu = L^-1 Lambda' Z'y, in place
        matmul_at(lam, zty_i, cu, q, q, 1);
        trsm_lower(l, cu, q, 1);

        // Accumulate RZX'RZX (upper triangle), RZX'cu and ||cu||^2.
        for v in cu.iter() {
            acc.cu2.add(v * v);
        }
        for a in 0..p {
            let mut sp = 0.0;
            for r in 0..q {
                sp += rzx[r * p + a] * cu[r];
            }
            acc.pv[a].add(sp);
            for bb in a..p {
                let mut s = 0.0;
                for r in 0..q {
                    s += rzx[r * p + a] * rzx[r * p + bb];
                }
                acc.pp[a * p + bb] += s;
            }
        }
    }
    acc
}

struct Acc2 {
    grad_ld: Vec<f64>,
    grad_pw: Vec<f64>,
}

/// Pass 2 over one chunk: random effects and both gradient contributions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn blocks_pass2(
    d: &LmmData,
    q: usize,
    p: usize,
    g0: usize,
    blocks: &[f64],
    u_out: Option<&mut [f64]>,
    beta: &[f64],
    rxtrx_inv: &[f64],
    reml: bool,
) -> Acc2 {
    let (qq, qp) = (q * q, q * p);
    let st = stride(q, p);
    let ng = blocks.len() / st;
    let nth = d.tix.len();
    let mut acc = Acc2 {
        grad_ld: vec![0.0; nth],
        grad_pw: vec![0.0; nth],
    };
    // All scratch for the chunk in one allocation.
    let mut scratch = vec![0.0; 3 * qq + 2 * qp + 2 * q];
    let (ainv, rest) = scratch.split_at_mut(qq);
    let (g, rest) = rest.split_at_mut(qq);
    let (bpbt, rest) = rest.split_at_mut(qq);
    let (b, rest) = rest.split_at_mut(qp);
    let (bp, rest) = rest.split_at_mut(qp);
    let (t, u_local) = rest.split_at_mut(q);
    let mut u_out = u_out;

    for j in 0..ng {
        let i = g0 + j;
        let buf = &blocks[j * st..(j + 1) * st];
        let ztx_i = &d.ztx[i * qp..(i + 1) * qp];
        let zty_i = &d.zty[i * q..(i + 1) * q];
        let (m_mat, rest) = buf.split_at(qq);
        let (l, rest) = rest.split_at(qq);
        let (rzx, cu) = rest.split_at(qp);
        let u: &mut [f64] = match u_out.as_deref_mut() {
            Some(all) => &mut all[j * q..(j + 1) * q],
            None => &mut u_local[..],
        };

        // u = L^-T (cu - RZX beta)
        for r in 0..q {
            let mut s = cu[r];
            for c in 0..p {
                s -= rzx[r * p + c] * beta[c];
            }
            u[r] = s;
        }
        trsm_lower_t(l, u, q, 1);

        // A^-1 = L^-T L^-1, rebuilt here rather than stored
        chol_inverse(l, ainv, q);
        // G = M * A^-1                     -> d(ldL2)
        matmul(m_mat, ainv, g, q, q, q);

        // t = Z'y - Z'X beta - M u         -> d(pwrss)
        for r in 0..q {
            let mut s = zty_i[r];
            for c in 0..p {
                s -= ztx_i[r * p + c] * beta[c];
            }
            for c in 0..q {
                s -= m_mat[r * q + c] * u[c];
            }
            t[r] = s;
        }

        if reml {
            // B = A^-1 W = L^-T (L^-1 W) = L^-T rzx
            b.copy_from_slice(rzx);
            trsm_lower_t(l, b, q, p);
            matmul(b, rxtrx_inv, bp, q, p, p);
            for r in 0..q {
                for c in 0..q {
                    let mut s = 0.0;
                    for x in 0..p {
                        s += bp[r * p + x] * b[c * p + x];
                    }
                    bpbt[r * q + c] = s;
                }
            }
        }

        for (k, &(r, c)) in d.tix.iter().enumerate() {
            let mut gk = 2.0 * g[r * q + c];
            if reml {
                let mut s1 = 0.0;
                for x in 0..p {
                    s1 += ztx_i[r * p + x] * bp[c * p + x];
                }
                let mut s2 = 0.0;
                for x in 0..q {
                    s2 += m_mat[r * q + x] * bpbt[x * q + c];
                }
                gk -= 2.0 * (s1 - s2);
            }
            acc.grad_ld[k] += gk;
            acc.grad_pw[k] += -2.0 * u[c] * t[r];
        }
    }
    acc
}

fn evaluate_blocks(d: &LmmData, theta: &[f64], reml: bool, mode: Mode) -> Option<Eval> {
    let (p, q, m) = (d.p, d.q, d.m);
    let lam = theta_to_lambda(theta, q);
    let nth = d.tix.len();
    let st = stride(q, p);
    let (chunk, parallel) = plan(m, blocks_cost(q, p));
    let store = mode.grad();

    // ---- Pass 1: factorise each block, accumulate the fixed-effect system.
    let mut blocks = if store {
        d.take_buffer(m * st)
    } else {
        Vec::new()
    };
    let accs: Vec<Acc1> = if store {
        let parts: Vec<(usize, &mut [f64])> = blocks
            .chunks_mut(chunk * st)
            .enumerate()
            .map(|(c, s)| (c * chunk, s))
            .collect();
        map_parts(parallel, parts, |(g0, bufs)| {
            let ng = bufs.len() / st;
            specialise_q!(q, |qc| blocks_pass1(d, &lam, qc, p, g0, ng, bufs, true))
        })
    } else {
        let parts: Vec<usize> = (0..m).step_by(chunk).collect();
        map_parts(parallel, parts, |g0| {
            let ng = chunk.min(m - g0);
            let mut scratch = vec![0.0; st];
            specialise_q!(q, |qc| blocks_pass1(
                d,
                &lam,
                qc,
                p,
                g0,
                ng,
                &mut scratch,
                false
            ))
        })
    };

    let mut ok = true;
    let (mut ldl2, mut cu2) = (Csum::default(), Csum::default());
    let mut sum_pp = vec![0.0; p * p];
    let mut sum_pv = vec![Csum::default(); p];
    for a in &accs {
        ok &= a.ok;
        ldl2.merge(a.ldl2);
        cu2.merge(a.cu2);
        add_into(&mut sum_pp, &a.pp);
        merge_into(&mut sum_pv, &a.pv);
    }
    let ldl2 = ldl2.get();
    drop(accs);
    let give_back = |blocks: Vec<f64>| {
        if store {
            d.give_buffer(blocks)
        }
    };
    if !ok {
        give_back(blocks);
        return None;
    }

    // ---- Fixed effects: (X'X - sum RZX'RZX) beta = X'y - sum RZX'cu
    let mut hmat = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            hmat[a * p + b] = d.xtx[a * p + b] - sum_pp[a * p + b];
        }
    }
    mirror_upper(&mut hmat, p);
    let hv: Vec<f64> = (0..p)
        .map(|j| sum_pv[j].subtracted_from(d.xty[j]))
        .collect();
    let c = cu2.subtracted_from(d.yty);
    let prof = match profile(d, hmat, hv, c, ldl2, reml) {
        Some(v) => v,
        None => {
            give_back(blocks);
            return None;
        }
    };

    let mut grad = Vec::new();
    let mut u_all = Vec::new();
    let mut rxtrx_inv = Vec::new();
    if store {
        if reml || mode == Mode::Full {
            rxtrx_inv = vec![0.0; p * p];
            chol_inverse(&prof.rx, &mut rxtrx_inv, p);
        }
        if mode == Mode::Full {
            u_all = vec![0.0; m * q];
        }

        // ---- Pass 2: random effects and both gradient contributions.
        //
        // The pwrss gradient is accumulated UNSCALED here; the dfree/pwrss
        // factor is applied afterwards.
        let beta = &prof.beta;
        let rinv = &rxtrx_inv;
        let accs2: Vec<Acc2> = if mode == Mode::Full {
            let parts: Vec<(usize, &[f64], &mut [f64])> = blocks
                .chunks(chunk * st)
                .zip(u_all.chunks_mut(chunk * q))
                .enumerate()
                .map(|(c, (b, u))| (c * chunk, b, u))
                .collect();
            map_parts(parallel, parts, |(g0, b, u)| {
                specialise_q!(q, |qc| blocks_pass2(
                    d,
                    qc,
                    p,
                    g0,
                    b,
                    Some(&mut *u),
                    beta,
                    rinv,
                    reml
                ))
            })
        } else {
            let parts: Vec<(usize, &[f64])> = blocks
                .chunks(chunk * st)
                .enumerate()
                .map(|(c, b)| (c * chunk, b))
                .collect();
            map_parts(parallel, parts, |(g0, b)| {
                specialise_q!(q, |qc| blocks_pass2(
                    d, qc, p, g0, b, None, beta, rinv, reml
                ))
            })
        };
        let mut grad_ld = vec![0.0; nth];
        let mut grad_pw = vec![0.0; nth];
        for a in &accs2 {
            for k in 0..nth {
                grad_ld[k] += a.grad_ld[k];
                grad_pw[k] += a.grad_pw[k];
            }
        }
        let scale = prof.dfree / prof.pwrss;
        grad = (0..nth).map(|k| grad_ld[k] + scale * grad_pw[k]).collect();
        if mode != Mode::Full {
            rxtrx_inv = Vec::new();
        }
    }
    give_back(blocks);

    Some(Eval {
        deviance: prof.deviance,
        grad,
        beta: prof.beta,
        u: u_all,
        sigma2: prof.pwrss / prof.dfree,
        pwrss: prof.pwrss,
        ldl2,
        ldrx2: if reml { prof.ldrx2 } else { 0.0 },
        rxtrx_inv,
    })
}

// ---------------------------------------------------------------------------
// Kernel::Classes and Kernel::Streaming -- the augmented Schur form
// ---------------------------------------------------------------------------

/// Where the augmented kernels read their units from: collapsed classes, or
/// the groups themselves.
trait Source: Sync {
    fn len(&self) -> usize;
    fn weight(&self, i: usize) -> f64;
    fn s(&self, i: usize) -> &[f64];
    /// `out[x, y] += scale * sum_ab w[a,b] T_i[a,b][x,y]` for `x <= y`,
    /// accumulated with compensation.
    /// `v` is `q x (p+1)` scratch and `wv` another.
    #[allow(clippy::too_many_arguments)]
    fn contract(
        &self,
        i: usize,
        w: &[f64],
        scale: f64,
        out: &mut [Csum],
        v: &mut [f64],
        wv: &mut [f64],
        q: usize,
        p: usize,
    );
    /// Load unit `i`'s `v` into scratch, when the contraction needs it.
    fn load(&self, i: usize, v: &mut [f64], q: usize, p: usize);
}

struct ClassSource<'a> {
    d: &'a LmmData,
    c: &'a Classes,
}

struct GroupSource<'a> {
    d: &'a LmmData,
}

impl Source for ClassSource<'_> {
    fn len(&self) -> usize {
        self.c.k
    }
    #[inline(always)]
    fn weight(&self, i: usize) -> f64 {
        self.c.count[i]
    }
    #[inline(always)]
    fn s(&self, i: usize) -> &[f64] {
        let qq = self.d.q * self.d.q;
        &self.c.s[i * qq..(i + 1) * qq]
    }
    #[inline(always)]
    fn load(&self, _i: usize, _v: &mut [f64], _q: usize, _p: usize) {}
    #[inline(always)]
    fn contract(
        &self,
        i: usize,
        w: &[f64],
        scale: f64,
        out: &mut [Csum],
        _v: &mut [f64],
        _wv: &mut [f64],
        q: usize,
        p: usize,
    ) {
        let p1 = p + 1;
        let blk = p1 * p1;
        let base = i * q * q * blk;
        for a in 0..q {
            for b in 0..q {
                let wab = w[a * q + b] * scale;
                if wab == 0.0 {
                    continue;
                }
                let tab = &self.c.t[base + (a * q + b) * blk..base + (a * q + b + 1) * blk];
                for x in 0..p1 {
                    for y in x..p1 {
                        out[x * p1 + y].add(wab * tab[x * p1 + y]);
                    }
                }
            }
        }
    }
}

impl Source for GroupSource<'_> {
    fn len(&self) -> usize {
        self.d.m
    }
    #[inline(always)]
    fn weight(&self, _i: usize) -> f64 {
        1.0
    }
    #[inline(always)]
    fn s(&self, i: usize) -> &[f64] {
        let qq = self.d.q * self.d.q;
        &self.d.ztz[i * qq..(i + 1) * qq]
    }
    #[inline(always)]
    fn load(&self, i: usize, v: &mut [f64], q: usize, p: usize) {
        fill_v(v, &self.d.ztx, &self.d.zty, i, q, p);
    }
    #[inline(always)]
    fn contract(
        &self,
        _i: usize,
        w: &[f64],
        scale: f64,
        out: &mut [Csum],
        v: &mut [f64],
        wv: &mut [f64],
        q: usize,
        p: usize,
    ) {
        // Rank-one T: sum_ab w_ab v[a]' v[b] = V' W V.
        let p1 = p + 1;
        matmul(w, v, wv, q, q, p1);
        for x in 0..p1 {
            for y in x..p1 {
                let mut s = 0.0;
                for a in 0..q {
                    s += v[a * p1 + x] * wv[a * p1 + y];
                }
                out[x * p1 + y].add(scale * s);
            }
        }
    }
}

struct AccA {
    ok: bool,
    ldl2: Csum,
    sub: Vec<Csum>,
    dld: Vec<f64>,
    dsub: Vec<Csum>,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn augmented_chunk<S: Source>(
    src: &S,
    lam: &[f64],
    tix: &[(usize, usize)],
    q: usize,
    p: usize,
    i0: usize,
    i1: usize,
    grad: bool,
) -> AccA {
    let qq = q * q;
    let p1 = p + 1;
    let nth = tix.len();
    let mut acc = AccA {
        ok: true,
        ldl2: Csum::default(),
        sub: vec![Csum::default(); p1 * p1],
        dld: vec![0.0; if grad { nth } else { 0 }],
        dsub: vec![Csum::default(); if grad { nth * p1 * p1 } else { 0 }],
    };
    let mut scratch = vec![0.0; 8 * qq + 2 * q * p1];
    let (mm, rest) = scratch.split_at_mut(qq);
    let (l, rest) = rest.split_at_mut(qq);
    let (ainv, rest) = rest.split_at_mut(qq);
    let (bt, rest) = rest.split_at_mut(qq);
    let (pm, rest) = rest.split_at_mut(qq);
    let (sb, rest) = rest.split_at_mut(qq);
    let (sp, rest) = rest.split_at_mut(qq);
    let (gm, rest) = rest.split_at_mut(qq);
    let (v, wv) = rest.split_at_mut(q * p1);

    for i in i0..i1 {
        let s = src.s(i);
        let h = src.weight(i);
        // A = Lambda' S Lambda + I
        matmul(s, lam, mm, q, q, q);
        matmul_at(lam, mm, l, q, q, q);
        for r in 0..q {
            l[r * q + r] += 1.0;
        }
        if cholesky(l, q).is_none() {
            acc.ok = false;
            return acc;
        }
        acc.ldl2.add(h * 2.0 * log_diag_sum(l, q));
        chol_inverse(l, ainv, q);
        // bt = Lambda A^-1;  P = bt Lambda'
        matmul(lam, ainv, bt, q, q, q);
        for a in 0..q {
            for b in 0..q {
                let mut acc_ab = 0.0;
                for x in 0..q {
                    acc_ab += bt[a * q + x] * lam[b * q + x];
                }
                pm[a * q + b] = acc_ab;
            }
        }
        src.load(i, v, q, p);
        src.contract(i, pm, 1.0, &mut acc.sub, v, wv, q, p);

        if grad {
            // sb = S Lambda A^-1 = M A^-1;  sp = S P = sb Lambda'
            matmul(s, bt, sb, q, q, q);
            for a in 0..q {
                for b in 0..q {
                    let mut acc_ab = 0.0;
                    for x in 0..q {
                        acc_ab += sb[a * q + x] * lam[b * q + x];
                    }
                    sp[a * q + b] = acc_ab;
                }
            }
            for (k, &(r, c)) in tix.iter().enumerate() {
                acc.dld[k] += h * 2.0 * sb[r * q + c];
                // F[a,b] = [a == r] bt[b,c] - bt[a,c] sp[r,b];  G = F + F'
                for a in 0..q {
                    for b in 0..=a {
                        let fab = if a == r { bt[b * q + c] } else { 0.0 }
                            - bt[a * q + c] * sp[r * q + b];
                        let fba = if b == r { bt[a * q + c] } else { 0.0 }
                            - bt[b * q + c] * sp[r * q + a];
                        gm[a * q + b] = fab + fba;
                        gm[b * q + a] = fab + fba;
                    }
                }
                src.contract(
                    i,
                    gm,
                    1.0,
                    &mut acc.dsub[k * p1 * p1..(k + 1) * p1 * p1],
                    v,
                    wv,
                    q,
                    p,
                );
            }
        }
    }
    acc
}

/// `e' B e` for symmetric `B` given by its upper triangle (`p1 x p1`).
#[inline]
fn sym_quad(e: &[f64], upper: &[f64], p1: usize) -> f64 {
    let mut s = 0.0;
    for x in 0..p1 {
        s += e[x] * e[x] * upper[x * p1 + x];
        for y in x + 1..p1 {
            s += 2.0 * e[x] * e[y] * upper[x * p1 + y];
        }
    }
    s
}

fn evaluate_augmented<S: Source>(
    d: &LmmData,
    src: &S,
    theta: &[f64],
    reml: bool,
    mode: Mode,
) -> Option<Eval> {
    let (p, q) = (d.p, d.q);
    let p1 = p + 1;
    let lam = theta_to_lambda(theta, q);
    let nth = d.tix.len();
    let grad = mode.grad();
    let units = src.len();
    let unit_cost = match d.kernel {
        Kernel::Streaming => streaming_cost(q, p),
        _ => class_cost(q, p),
    };
    let (chunk, parallel) = plan(units, unit_cost);
    let parts: Vec<usize> = (0..units).step_by(chunk).collect();
    let tix = &d.tix[..];
    let accs = map_parts(parallel, parts, |i0| {
        let i1 = (i0 + chunk).min(units);
        specialise_q!(q, |qc| augmented_chunk(src, &lam, tix, qc, p, i0, i1, grad))
    });

    let mut ok = true;
    let mut ldl2 = Csum::default();
    let mut sub = vec![Csum::default(); p1 * p1];
    let mut dld = vec![0.0; if grad { nth } else { 0 }];
    let mut dsub = vec![Csum::default(); if grad { nth * p1 * p1 } else { 0 }];
    for a in &accs {
        ok &= a.ok;
        ldl2.merge(a.ldl2);
        merge_into(&mut sub, &a.sub);
        add_into(&mut dld, &a.dld);
        merge_into(&mut dsub, &a.dsub);
    }
    let ldl2 = ldl2.get();
    let dsub: Vec<f64> = dsub.iter().map(|v| v.get()).collect();
    drop(accs);
    if !ok {
        return None;
    }

    let mut hmat = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            hmat[a * p + b] = sub[a * p1 + b].subtracted_from(d.xtx[a * p + b]);
        }
    }
    mirror_upper(&mut hmat, p);
    let hv: Vec<f64> = (0..p)
        .map(|j| sub[j * p1 + p].subtracted_from(d.xty[j]))
        .collect();
    let c = sub[p * p1 + p].subtracted_from(d.yty);
    let prof = profile(d, hmat, hv, c, ldl2, reml)?;

    let mut rxtrx_inv = Vec::new();
    if (grad && reml) || mode == Mode::Full {
        rxtrx_inv = vec![0.0; p * p];
        chol_inverse(&prof.rx, &mut rxtrx_inv, p);
    }

    let mut g = Vec::new();
    if grad {
        let mut e = vec![1.0; p1];
        for (ej, b) in e.iter_mut().zip(&prof.beta) {
            *ej = -b;
        }
        let scale = prof.dfree / prof.pwrss;
        g = vec![0.0; nth];
        for k in 0..nth {
            let dk = &dsub[k * p1 * p1..(k + 1) * p1 * p1];
            // dK_k = -dsub_k
            let d_pwrss = -sym_quad(&e, dk, p1);
            let mut gk = dld[k] + scale * d_pwrss;
            if reml {
                // tr(H^-1 dH_k), dH_k = -(dsub_k)[:p, :p]
                let mut tr = 0.0;
                for x in 0..p {
                    tr += rxtrx_inv[x * p + x] * dk[x * p1 + x];
                    for y in x + 1..p {
                        tr += 2.0 * rxtrx_inv[x * p + y] * dk[x * p1 + y];
                    }
                }
                gk -= tr;
            }
            g[k] = gk;
        }
    }

    let mut u_all = Vec::new();
    if mode == Mode::Full {
        u_all = random_effects(d, &lam, &prof.beta);
    } else if mode == Mode::Gradient {
        rxtrx_inv = Vec::new();
    }

    Some(Eval {
        deviance: prof.deviance,
        grad: g,
        beta: prof.beta,
        u: u_all,
        sigma2: prof.pwrss / prof.dfree,
        pwrss: prof.pwrss,
        ldl2,
        ldrx2: if reml { prof.ldrx2 } else { 0.0 },
        rxtrx_inv,
    })
}

/// `u_i = A_i^-1 Lambda' (Z_i'y - Z_i'X beta)` for every group.
fn random_effects(d: &LmmData, lam: &[f64], beta: &[f64]) -> Vec<f64> {
    let (p, q, m) = (d.p, d.q, d.m);
    let (qq, qp) = (q * q, q * p);
    let mut u_all = vec![0.0; m * q];
    let (chunk, parallel) = plan(m, 2 * q * q * q + qp + 4);
    let parts: Vec<(usize, &mut [f64])> = u_all
        .chunks_mut(chunk * q)
        .enumerate()
        .map(|(c, u)| (c * chunk, u))
        .collect();
    map_parts(parallel, parts, |(g0, u)| {
        let mut scratch = vec![0.0; 2 * qq + q];
        let (mm, rest) = scratch.split_at_mut(qq);
        let (l, r) = rest.split_at_mut(qq);
        for j in 0..u.len() / q {
            let i = g0 + j;
            matmul(&d.ztz[i * qq..(i + 1) * qq], lam, mm, q, q, q);
            matmul_at(lam, mm, l, q, q, q);
            for a in 0..q {
                l[a * q + a] += 1.0;
            }
            // A is Lambda'SLambda + I and was factorised successfully when the
            // criterion was evaluated at this theta.
            let _ = cholesky(l, q);
            for (a, ra) in r.iter_mut().enumerate() {
                let mut s = d.zty[i * q + a];
                for (x, bx) in beta.iter().enumerate() {
                    s -= d.ztx[i * qp + a * p + x] * bx;
                }
                *ra = s;
            }
            let ui = &mut u[j * q..(j + 1) * q];
            matmul_at(lam, r, ui, q, q, 1);
            trsm_lower(l, ui, q, 1);
            trsm_lower_t(l, ui, q, 1);
        }
    });
    u_all
}

/// `Var(b_i | y) = sigma^2 Lambda A_i^-1 Lambda'` for every group, `m*q*q`.
///
/// This is the Woodbury form of `G - G Z_i'(Z_i G Z_i' + sigma^2 I)^-1 Z_i G`
/// with `G = sigma^2 Lambda Lambda'`, and remains valid for singular `Lambda`.
pub fn conditional_covariances(d: &LmmData, theta: &[f64], sigma2: f64) -> Option<Vec<f64>> {
    let (q, m) = (d.q, d.m);
    let qq = q * q;
    let lam = theta_to_lambda(theta, q);
    let mut out = vec![0.0; m * qq];
    let (chunk, parallel) = plan(m, 4 * q * q * q + 4);
    let parts: Vec<(usize, &mut [f64])> = out
        .chunks_mut(chunk * qq)
        .enumerate()
        .map(|(c, o)| (c * chunk, o))
        .collect();
    let oks = map_parts(parallel, parts, |(g0, o)| {
        let mut scratch = vec![0.0; 4 * qq];
        let (mm, rest) = scratch.split_at_mut(qq);
        let (l, rest) = rest.split_at_mut(qq);
        let (ainv, bt) = rest.split_at_mut(qq);
        for j in 0..o.len() / qq {
            let i = g0 + j;
            matmul(&d.ztz[i * qq..(i + 1) * qq], &lam, mm, q, q, q);
            matmul_at(&lam, mm, l, q, q, q);
            for a in 0..q {
                l[a * q + a] += 1.0;
            }
            if cholesky(l, q).is_none() {
                return false;
            }
            chol_inverse(l, ainv, q);
            matmul(&lam, ainv, bt, q, q, q);
            let oi = &mut o[j * qq..(j + 1) * qq];
            for a in 0..q {
                for b in 0..q {
                    let mut s = 0.0;
                    for x in 0..q {
                        s += bt[a * q + x] * lam[b * q + x];
                    }
                    oi[a * q + b] = sigma2 * s;
                }
            }
        }
        true
    });
    if oks.into_iter().all(|v| v) {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Analytic second derivative for a scalar random effect
// ---------------------------------------------------------------------------

struct AccH {
    ld: [Csum; 3],
    sub: [Vec<Csum>; 3],
}

/// `(deviance, d/dtheta, d2/dtheta2)` for `q = 1`, or `None` if infeasible.
///
/// With `t = theta^2` and `s` a group's (or class's) `Z'Z`, the augmented
/// system and its `t`-derivatives are
///
/// ```text
/// K   = C - sum t/(1+ts)   T,     K'  = -sum 1/(1+ts)^2 T,    K'' = 2 sum s/(1+ts)^3 T
/// ldL2' = sum h s/(1+ts),         ldL2'' = -sum h s^2/(1+ts)^2
/// pwrss' = e'K'e,                 pwrss'' = e'K''e - 2 g'H^-1 g,  g = K'[:p,:] e
/// ldRX2' = tr(H^-1 H'),           ldRX2'' = tr(H^-1 H'') - tr(H^-1 H' H^-1 H')
/// ```
///
/// and the chain rule through `t = theta^2` gives `D'' = 2 f' + 4 t f''`.
pub fn scalar_hessian(d: &LmmData, theta: f64, reml: bool) -> Option<(f64, f64, f64)> {
    if d.q != 1 {
        return None;
    }
    match (&d.classes, d.kernel) {
        (Some(c), Kernel::Classes) => scalar_hessian_src(d, &ClassSource { d, c }, theta, reml),
        _ => scalar_hessian_src(d, &GroupSource { d }, theta, reml),
    }
}

fn scalar_hessian_src<S: Source>(
    d: &LmmData,
    src: &S,
    theta: f64,
    reml: bool,
) -> Option<(f64, f64, f64)> {
    let p = d.p;
    let p1 = p + 1;
    let t = theta * theta;
    let units = src.len();
    let (chunk, parallel) = plan(units, 3 * p1 * p1 + 8);
    let parts: Vec<usize> = (0..units).step_by(chunk).collect();
    let one = [1.0];
    let accs = map_parts(parallel, parts, |i0| {
        let i1 = (i0 + chunk).min(units);
        let zero = vec![Csum::default(); p1 * p1];
        let mut acc = AccH {
            ld: [Csum::default(); 3],
            sub: [zero.clone(), zero.clone(), zero],
        };
        let mut v = vec![0.0; p1];
        let mut wv = vec![0.0; p1];
        for i in i0..i1 {
            let s = src.s(i)[0];
            let h = src.weight(i);
            let den = 1.0 + t * s;
            acc.ld[0].add(h * den.ln());
            acc.ld[1].add(h * s / den);
            acc.ld[2].add(-h * s * s / (den * den));
            src.load(i, &mut v, 1, p);
            src.contract(i, &one, t / den, &mut acc.sub[0], &mut v, &mut wv, 1, p);
            src.contract(
                i,
                &one,
                1.0 / (den * den),
                &mut acc.sub[1],
                &mut v,
                &mut wv,
                1,
                p,
            );
            src.contract(
                i,
                &one,
                s / (den * den * den),
                &mut acc.sub[2],
                &mut v,
                &mut wv,
                1,
                p,
            );
        }
        acc
    });
    let mut ld_acc = [Csum::default(); 3];
    let zero = vec![Csum::default(); p1 * p1];
    let mut sub_acc = [zero.clone(), zero.clone(), zero];
    for a in &accs {
        for j in 0..3 {
            ld_acc[j].merge(a.ld[j]);
            merge_into(&mut sub_acc[j], &a.sub[j]);
        }
    }
    let ld = ld_acc.map(|v| v.get());
    let mut sub = sub_acc.map(|v| v.iter().map(|c| c.get()).collect::<Vec<f64>>());
    for s in sub.iter_mut() {
        mirror_upper(s, p1);
    }
    if !ld[0].is_finite() {
        return None;
    }
    let ldl2 = ld[0];

    let mut hmat = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            hmat[a * p + b] = d.xtx[a * p + b] - sub[0][a * p1 + b];
        }
    }
    let hv: Vec<f64> = (0..p).map(|j| d.xty[j] - sub[0][j * p1 + p]).collect();
    let c = d.yty - sub[0][p * p1 + p];
    let prof = profile(d, hmat, hv, c, ldl2, reml)?;
    let mut hinv = vec![0.0; p * p];
    chol_inverse(&prof.rx, &mut hinv, p);

    let mut e = vec![1.0; p1];
    for (ej, b) in e.iter_mut().zip(&prof.beta) {
        *ej = -b;
    }
    // K' = -sub1, K'' = 2 sub2
    let k1 = |x: usize, y: usize| -sub[1][x * p1 + y];
    let k2 = |x: usize, y: usize| 2.0 * sub[2][x * p1 + y];
    let mut pw1 = 0.0;
    let mut pw2 = 0.0;
    for x in 0..p1 {
        for y in 0..p1 {
            pw1 += e[x] * k1(x, y) * e[y];
            pw2 += e[x] * k2(x, y) * e[y];
        }
    }
    let gvec: Vec<f64> = (0..p)
        .map(|x| (0..p1).map(|y| k1(x, y) * e[y]).sum())
        .collect();
    let mut ghg = 0.0;
    for x in 0..p {
        for y in 0..p {
            ghg += gvec[x] * hinv[x * p + y] * gvec[y];
        }
    }
    pw2 -= 2.0 * ghg;

    let (mut tr1, mut tr2) = (0.0, 0.0);
    // hk1 = H^-1 H'
    let mut hk1 = vec![0.0; p * p];
    for x in 0..p {
        for y in 0..p {
            let mut s = 0.0;
            for z in 0..p {
                s += hinv[x * p + z] * k1(z, y);
            }
            hk1[x * p + y] = s;
        }
    }
    for x in 0..p {
        for y in 0..p {
            tr1 += hinv[x * p + y] * k1(y, x);
            tr2 += hinv[x * p + y] * k2(y, x);
            tr2 -= hk1[x * p + y] * hk1[y * p + x];
        }
    }

    let pw = prof.pwrss;
    let df = prof.dfree;
    let r = if reml { 1.0 } else { 0.0 };
    let f1 = ld[1] + r * tr1 + df * pw1 / pw;
    let f2 = ld[2] + r * tr2 + df * (pw2 / pw - (pw1 / pw) * (pw1 / pw));
    Some((prof.deviance, 2.0 * theta * f1, 2.0 * f1 + 4.0 * t * f2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small unbalanced design with a repeated Z'Z pattern, built row by row.
    type Fixture = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>, usize, usize, usize);

    fn fixture(q: usize, reps: bool) -> Fixture {
        let p = 3;
        let m = 23;
        let mut state = 12345u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let (mut y, mut x, mut z, mut codes) = (vec![], vec![], vec![], vec![]);
        for g in 0..m {
            let ni = 2 + (g % 4);
            for j in 0..ni {
                x.push(1.0);
                x.push(rnd());
                x.push(rnd());
                z.push(1.0);
                if q > 1 {
                    z.push(if reps { j as f64 } else { rnd() });
                }
                if q > 2 {
                    z.push(if reps { (j * j) as f64 } else { rnd() });
                }
                y.push(rnd() + 0.3 * g as f64);
                codes.push(g as i64);
            }
        }
        (y, x, z, codes, p, q, m)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn kernels_agree() {
        for q in 1..=3 {
            for reps in [true, false] {
                let (y, x, z, codes, p, q, m) = fixture(q, reps);
                let mk = |k| {
                    LmmData::from_rows(&y, &x, &z, &codes, p, q, m, KernelRequest::Force(k))
                        .unwrap()
                };
                let (db, ds, dc) = (
                    mk(Kernel::Blocks),
                    mk(Kernel::Streaming),
                    mk(Kernel::Classes),
                );
                let nth = n_theta(q);
                for (ti, scale) in [0.0, 0.3, 1.0, 4.0].iter().enumerate() {
                    let theta: Vec<f64> = (0..nth)
                        .map(|k| {
                            if theta_index(q)[k].0 == theta_index(q)[k].1 {
                                *scale
                            } else {
                                0.1 * ti as f64
                            }
                        })
                        .collect();
                    for reml in [true, false] {
                        let eb = evaluate(&db, &theta, reml, Mode::Full).unwrap();
                        for other in [&ds, &dc] {
                            for mode in [Mode::Criterion, Mode::Gradient, Mode::Full] {
                                let eo = evaluate(other, &theta, reml, mode).unwrap();
                                assert!(
                                    close(eb.deviance, eo.deviance, 1e-11),
                                    "{} {}",
                                    eb.deviance,
                                    eo.deviance
                                );
                                if mode.grad() {
                                    for k in 0..nth {
                                        assert!(
                                            close(eb.grad[k], eo.grad[k], 1e-8),
                                            "q={q} k={k} {:?} {:?}",
                                            eb.grad,
                                            eo.grad
                                        );
                                    }
                                }
                                if mode == Mode::Full {
                                    for j in 0..m * q {
                                        assert!(close(eb.u[j], eo.u[j], 1e-9));
                                    }
                                    for j in 0..p * p {
                                        assert!(close(eb.rxtrx_inv[j], eo.rxtrx_inv[j], 1e-9));
                                    }
                                }
                            }
                        }
                        let ec = evaluate(&db, &theta, reml, Mode::Criterion).unwrap();
                        assert_eq!(ec.deviance.to_bits(), eb.deviance.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_hessian_matches_differenced_gradient() {
        let (y, x, z, codes, p, q, m) = fixture(1, true);
        for k in [Kernel::Blocks, Kernel::Classes] {
            let d =
                LmmData::from_rows(&y, &x, &z, &codes, p, q, m, KernelRequest::Force(k)).unwrap();
            for th in [0.0, 0.2, 0.9, 3.0] {
                for reml in [true, false] {
                    let (dev, g, hess) = scalar_hessian(&d, th, reml).unwrap();
                    let e = evaluate(&d, &[th], reml, Mode::Gradient).unwrap();
                    assert!(close(dev, e.deviance, 1e-11));
                    assert!(close(g, e.grad[0], 1e-8), "{g} {}", e.grad[0]);
                    let hstep = 1e-5;
                    let gp = evaluate(&d, &[th + hstep], reml, Mode::Gradient)
                        .unwrap()
                        .grad[0];
                    let gm = evaluate(&d, &[th - hstep], reml, Mode::Gradient)
                        .unwrap()
                        .grad[0];
                    let fd = (gp - gm) / (2.0 * hstep);
                    assert!(close(hess, fd, 1e-5), "th={th} {hess} {fd}");
                }
            }
        }
    }

    #[test]
    fn balanced_intercepts_collapse_to_one_class() {
        let p = 2;
        let m = 50;
        let (mut y, mut x, mut z, mut codes) = (vec![], vec![], vec![], vec![]);
        for g in 0..m {
            for j in 0..4 {
                x.push(1.0);
                x.push((g * 7 + j) as f64 % 5.0);
                z.push(1.0);
                y.push(((g * 13 + j * 3) % 11) as f64);
                codes.push(g as i64);
            }
        }
        let d = LmmData::from_rows(&y, &x, &z, &codes, p, 1, m, KernelRequest::Auto).unwrap();
        assert_eq!(d.kernel, Kernel::Classes);
        assert_eq!(d.classes.as_ref().unwrap().k, 1);
    }
}
