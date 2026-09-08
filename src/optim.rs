//! Projected L-BFGS for box-constrained minimisation of the profiled criterion.
//!
//! The feasible set is a box: diagonal entries of `Lambda` are constrained
//! non-negative (matching lme4's lower bounds), off-diagonals are free. The
//! dimension is `q(q+1)/2` -- 1, 3 or 6 in practice -- so a compact two-loop
//! L-BFGS with a projected-arc backtracking line search is both sufficient and
//! very robust.
//!
//! We can afford a gradient-based method because the analytic gradient is
//! available (see `lmm.rs`). lme4 and MixedModels.jl use derivative-free BOBYQA
//! here, which needs materially more objective evaluations.

pub struct OptResult {
    pub x: Vec<f64>,
    pub fx: f64,
    pub grad_inf_norm: f64,
    pub iterations: usize,
    pub fev: usize,
    pub converged: bool,
    pub message: String,
}

pub struct OptSettings {
    pub max_iter: usize,
    pub gtol: f64,
    pub ftol: f64,
    pub memory: usize,
}

impl Default for OptSettings {
    fn default() -> Self {
        Self { max_iter: 300, gtol: 1e-8, ftol: 1e-12, memory: 10 }
    }
}

#[inline]
fn project(x: &mut [f64], lower: &[f64]) {
    for i in 0..x.len() {
        if lower[i].is_finite() && x[i] < lower[i] {
            x[i] = lower[i];
        }
    }
}

/// Infinity norm of the projected gradient -- the correct stationarity measure
/// for a box-constrained problem. A component pinned at its bound with the
/// gradient pushing further into the bound is *not* a violation.
fn projected_grad_norm(x: &[f64], g: &[f64], lower: &[f64]) -> f64 {
    let mut norm: f64 = 0.0;
    for i in 0..x.len() {
        let gi = if lower[i].is_finite() && x[i] <= lower[i] && g[i] > 0.0 {
            0.0
        } else {
            g[i]
        };
        norm = norm.max(gi.abs());
    }
    norm
}

/// Minimise `f` subject to `x >= lower` (componentwise, `-inf` meaning free).
///
/// `f` returns `None` for infeasible points; the line search treats that as
/// `+inf` and backtracks.
pub fn minimize<F>(
    mut f: F,
    x0: &[f64],
    lower: &[f64],
    s: &OptSettings,
) -> OptResult
where
    F: FnMut(&[f64]) -> Option<(f64, Vec<f64>)>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    project(&mut x, lower);

    let mut fev = 0usize;
    let (mut fx, mut g) = match f(&x) {
        Some(v) => {
            fev += 1;
            v
        }
        None => {
            return OptResult {
                x,
                fx: f64::INFINITY,
                grad_inf_norm: f64::INFINITY,
                iterations: 0,
                fev,
                converged: false,
                message: "initial point infeasible".into(),
            }
        }
    };

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut iterations = 0usize;
    let mut message = String::from("maximum iterations reached");
    let mut converged = false;

    for it in 0..s.max_iter {
        iterations = it + 1;

        let pg = projected_grad_norm(&x, &g, lower);
        if pg < s.gtol {
            converged = true;
            message = "projected gradient below tolerance".into();
            break;
        }

        // ---- Two-loop recursion for the search direction.
        let mut dir: Vec<f64> = g.iter().map(|v| -v).collect();
        let k = s_hist.len();
        let mut alpha = vec![0.0; k];
        for i in (0..k).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &dir);
            alpha[i] = a;
            axpy(-a, &y_hist[i], &mut dir);
        }
        if k > 0 {
            let last = k - 1;
            let sy = dot(&s_hist[last], &y_hist[last]);
            let yy = dot(&y_hist[last], &y_hist[last]);
            if yy > 0.0 {
                let gamma = sy / yy;
                for v in dir.iter_mut() {
                    *v *= gamma;
                }
            }
        }
        for i in 0..k {
            let b = rho_hist[i] * dot(&y_hist[i], &dir);
            axpy(alpha[i] - b, &s_hist[i], &mut dir);
        }

        // Zero out directions that would immediately leave the box at a bound.
        for i in 0..n {
            if lower[i].is_finite() && x[i] <= lower[i] && dir[i] < 0.0 {
                dir[i] = 0.0;
            }
        }

        let dg = dot(&dir, &g);
        if dg >= 0.0 {
            // Not a descent direction: reset the memory and take steepest descent.
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            dir = g.iter().map(|v| -v).collect();
            for i in 0..n {
                if lower[i].is_finite() && x[i] <= lower[i] && dir[i] < 0.0 {
                    dir[i] = 0.0;
                }
            }
        }

        // ---- Backtracking Armijo line search along the projected arc.
        let mut step = 1.0;
        let c1 = 1e-4;
        let g_dot_d = dot(&g, &dir);
        let mut accepted = false;
        let mut x_new = x.clone();
        let mut fx_new = fx;
        let mut g_new = g.clone();

        for _ in 0..40 {
            for i in 0..n {
                x_new[i] = x[i] + step * dir[i];
            }
            project(&mut x_new, lower);

            match f(&x_new) {
                Some((fv, gv)) => {
                    fev += 1;
                    // Armijo against the actual displacement (the arc may be shorter
                    // than the raw step once projection has clipped it).
                    let mut real_dot = 0.0;
                    for i in 0..n {
                        real_dot += g[i] * (x_new[i] - x[i]);
                    }
                    if fv <= fx + c1 * real_dot || (fv < fx && real_dot.abs() < 1e-16) {
                        fx_new = fv;
                        g_new = gv;
                        accepted = true;
                        break;
                    }
                }
                None => {
                    fev += 1;
                }
            }
            step *= 0.5;
        }

        if !accepted {
            if s_hist.is_empty() {
                message = "line search failed".into();
                break;
            }
            // One retry from a clean steepest-descent state.
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            continue;
        }

        let df = (fx - fx_new).abs();
        let denom = fx.abs().max(1.0);

        let mut s_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        for i in 0..n {
            s_vec[i] = x_new[i] - x[i];
            y_vec[i] = g_new[i] - g[i];
        }
        let sy = dot(&s_vec, &y_vec);
        if sy > 1e-12 {
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
            if s_hist.len() > s.memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
        }

        x = x_new;
        fx = fx_new;
        g = g_new;
        let _ = g_dot_d;

        if df / denom < s.ftol {
            let pg2 = projected_grad_norm(&x, &g, lower);
            converged = true;
            message = if pg2 < 1e-4 {
                "relative change below tolerance".into()
            } else {
                "relative change below tolerance (gradient not small)".into()
            };
            break;
        }
    }

    let grad_inf_norm = projected_grad_norm(&x, &g, lower);
    if !converged && grad_inf_norm < s.gtol {
        converged = true;
        message = "projected gradient below tolerance".into();
    }

    OptResult { x, fx, grad_inf_norm, iterations, fev, converged, message }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_a_quadratic() {
        // f(x) = (x0-3)^2 + (x1+2)^2, unconstrained optimum (3, -2)
        let f = |x: &[f64]| {
            let v = (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
            Some((v, vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)]))
        };
        let r = minimize(f, &[0.0, 0.0], &[f64::NEG_INFINITY; 2], &OptSettings::default());
        assert!(r.converged, "{}", r.message);
        assert!((r.x[0] - 3.0).abs() < 1e-6);
        assert!((r.x[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn respects_a_lower_bound() {
        // Same function but x1 >= 0, so the optimum moves to (3, 0)
        let f = |x: &[f64]| {
            let v = (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
            Some((v, vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)]))
        };
        let r = minimize(f, &[1.0, 1.0], &[f64::NEG_INFINITY, 0.0], &OptSettings::default());
        assert!((r.x[0] - 3.0).abs() < 1e-6, "x0 = {}", r.x[0]);
        assert!(r.x[1].abs() < 1e-9, "x1 = {}", r.x[1]);
        assert!(r.converged, "{}", r.message);
    }

    #[test]
    fn handles_rosenbrock() {
        let f = |x: &[f64]| {
            let (a, b) = (1.0, 100.0);
            let v = (a - x[0]).powi(2) + b * (x[1] - x[0] * x[0]).powi(2);
            let g = vec![
                -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0] * x[0]),
                2.0 * b * (x[1] - x[0] * x[0]),
            ];
            Some((v, g))
        };
        let s = OptSettings { max_iter: 2000, ..Default::default() };
        let r = minimize(f, &[-1.2, 1.0], &[f64::NEG_INFINITY; 2], &s);
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x = {:?} ({})", r.x, r.message);
        assert!((r.x[1] - 1.0).abs() < 1e-3, "x = {:?}", r.x);
    }
}
