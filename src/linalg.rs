//! Small dense linear algebra on row-major slices.
//!
//! Everything here operates on matrices that are either `q x q` (q is the number
//! of random effects per group, typically 1-3) or `p x p` (fixed effects,
//! typically under 50). At those sizes a hand-rolled implementation beats the
//! call overhead of a general BLAS, and it keeps the dependency surface at zero.

/// Lower-triangular Cholesky factor of a symmetric positive-definite matrix.
///
/// `a` is `n x n` row-major and is overwritten with `L` in its lower triangle;
/// the strict upper triangle is zeroed. Returns `None` if `a` is not positive
/// definite, which the caller treats as "this theta is infeasible".
pub fn cholesky(a: &mut [f64], n: usize) -> Option<()> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            let v = a[j * n + k];
            d -= v * v;
        }
        if !(d > 0.0) || !d.is_finite() {
            return None;
        }
        let ljj = d.sqrt();
        a[j * n + j] = ljj;
        let inv = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s * inv;
        }
    }
    for j in 0..n {
        for i in 0..j {
            a[i * n + j] = 0.0;
        }
    }
    Some(())
}

/// Sum of `log` of the Cholesky diagonal. Never forms a determinant.
#[inline]
pub fn log_diag_sum(l: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += l[i * n + i].ln();
    }
    s
}

/// Solve `L * X = B` in place for lower-triangular `L` (`n x n`).
/// `b` is `n x k` row-major and is overwritten with `X`.
pub fn trsm_lower(l: &[f64], b: &mut [f64], n: usize, k: usize) {
    for i in 0..n {
        let inv = 1.0 / l[i * n + i];
        for c in 0..k {
            let mut s = b[i * k + c];
            for j in 0..i {
                s -= l[i * n + j] * b[j * k + c];
            }
            b[i * k + c] = s * inv;
        }
    }
}

/// Solve `L' * X = B` in place for lower-triangular `L` (`n x n`).
pub fn trsm_lower_t(l: &[f64], b: &mut [f64], n: usize, k: usize) {
    for i in (0..n).rev() {
        let inv = 1.0 / l[i * n + i];
        for c in 0..k {
            let mut s = b[i * k + c];
            for j in (i + 1)..n {
                s -= l[j * n + i] * b[j * k + c];
            }
            b[i * k + c] = s * inv;
        }
    }
}

/// `c = a * b`, with `a` being `m x k` and `b` being `k x n`, all row-major.
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for v in c.iter_mut().take(m * n) {
        *v = 0.0;
    }
    for i in 0..m {
        for t in 0..k {
            let aik = a[i * k + t];
            if aik == 0.0 {
                continue;
            }
            let brow = &b[t * n..t * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += aik * brow[j];
            }
        }
    }
}

/// `c = a' * b`, with `a` being `k x m` and `b` being `k x n`, all row-major.
pub fn matmul_at(a: &[f64], b: &[f64], c: &mut [f64], k: usize, m: usize, n: usize) {
    for v in c.iter_mut().take(m * n) {
        *v = 0.0;
    }
    for t in 0..k {
        let arow = &a[t * m..t * m + m];
        let brow = &b[t * n..t * n + n];
        for i in 0..m {
            let a_ti = arow[i];
            if a_ti == 0.0 {
                continue;
            }
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += a_ti * brow[j];
            }
        }
    }
}

/// Full inverse of a symmetric positive-definite matrix from its Cholesky factor.
/// `l` is lower-triangular `n x n`; `out` receives the `n x n` inverse.
pub fn chol_inverse(l: &[f64], out: &mut [f64], n: usize) {
    // Build the identity, then apply L^{-1} and L^{-T}.
    for v in out.iter_mut().take(n * n) {
        *v = 0.0;
    }
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    trsm_lower(l, out, n, n);
    trsm_lower_t(l, out, n, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_reconstructs() {
        // A = [[4,2],[2,3]]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        cholesky(&mut a, 2).unwrap();
        // L L' should recover A
        let mut lt = vec![0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                lt[i * 2 + j] = a[j * 2 + i];
            }
        }
        let mut prod = vec![0.0; 4];
        matmul(&a, &lt, &mut prod, 2, 2, 2);
        for (got, want) in prod.iter().zip([4.0, 2.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&mut a, 2).is_none());
    }

    #[test]
    fn triangular_solves_round_trip() {
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        cholesky(&mut a, 2).unwrap();
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = orig.clone();
        trsm_lower(&a, &mut b, 2, 2);
        // Multiply back by L
        let mut back = vec![0.0; 4];
        matmul(&a, &b, &mut back, 2, 2, 2);
        for (got, want) in back.iter().zip(orig.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_is_inverse() {
        let a0 = vec![4.0, 2.0, 2.0, 3.0];
        let mut l = a0.clone();
        cholesky(&mut l, 2).unwrap();
        let mut inv = vec![0.0; 4];
        chol_inverse(&l, &mut inv, 2);
        let mut prod = vec![0.0; 4];
        matmul(&a0, &inv, &mut prod, 2, 2, 2);
        for (i, v) in prod.iter().enumerate() {
            let want = if i % 3 == 0 { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-12, "{v} vs {want}");
        }
    }
}
