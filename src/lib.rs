//! PyO3 bindings for the profiled-REML linear mixed model core.
//!
//! The whole fit is a single call into Rust with the GIL released. Nothing
//! iterates back into Python, so a model with 100,000 groups costs one FFI
//! crossing rather than one per group per iteration.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod linalg;
mod lmm;
mod optim;

use lmm::{evaluate, n_theta, theta_index, theta_to_lambda, LmmData};
use optim::{minimize, OptSettings};

/// Pre-computed cross-products for one grouping factor.
#[pyclass(name = "LmmCore", module = "mixedlm_rs._mixedlm_rs")]
pub struct LmmCore {
    data: LmmData,
}

#[pymethods]
impl LmmCore {
    /// Build the theta-independent cross-products in a single pass over the rows.
    ///
    /// `codes` must be contiguous group labels in `0..n_groups`.
    #[new]
    fn new(
        y: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
        z: PyReadonlyArray2<f64>,
        codes: PyReadonlyArray1<i64>,
        n_groups: usize,
    ) -> PyResult<Self> {
        let y = y.as_slice()?;
        let xs = x.as_array();
        let zs = z.as_array();
        let codes = codes.as_slice()?;

        let n = y.len();
        let p = xs.shape()[1];
        let q = zs.shape()[1];
        let m = n_groups;

        // Flat row-major slices: the accumulation below is O(n * (p^2 + qp + q^2))
        // scalar work, and going through bounds-checked 2-D ndarray indexing for
        // every element of it costs several times the arithmetic.
        let xv = xs
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("X must be C-contiguous"))?;
        let zv = zs
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Z must be C-contiguous"))?;

        if xs.shape()[0] != n || zs.shape()[0] != n || codes.len() != n {
            return Err(PyValueError::new_err(
                "y, X, Z and group codes must agree on the row count",
            ));
        }
        if m == 0 {
            return Err(PyValueError::new_err("n_groups must be positive"));
        }
        // A zero-width design reaches a zero-size rayon chunk downstream, and a
        // q of zero makes `theta` empty, which every evaluate() path then
        // indexes. Both used to abort the process rather than raise, because
        // this crate is built with panic="abort". Refuse them at the boundary.
        if p == 0 {
            return Err(PyValueError::new_err("X must have at least one column"));
        }
        if q == 0 {
            return Err(PyValueError::new_err("Z must have at least one column"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("y must not be empty"));
        }
        // m * q * p is the largest buffer; on a 32-bit usize a big model could
        // wrap and under-allocate. Check rather than trust.
        let too_big = m
            .checked_mul(q)
            .and_then(|v| v.checked_mul(p.max(q)))
            .is_none();
        if too_big {
            return Err(PyValueError::new_err(
                "model dimensions overflow the address space",
            ));
        }

        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        let mut yty = 0.0;
        let mut ztz = vec![0.0; m * q * q];
        let mut ztx = vec![0.0; m * q * p];
        let mut zty = vec![0.0; m * q];

        for r in 0..n {
            let g = codes[r];
            if g < 0 || (g as usize) >= m {
                return Err(PyValueError::new_err(format!(
                    "group code {g} out of range 0..{m}"
                )));
            }
            let g = g as usize;
            let yr = y[r];
            yty += yr * yr;

            let xrow = &xv[r * p..r * p + p];
            let zrow = &zv[r * q..r * q + q];

            for a in 0..p {
                let xa = xrow[a];
                xty[a] += xa * yr;
                let dst = &mut xtx[a * p..a * p + p];
                for b in 0..p {
                    dst[b] += xa * xrow[b];
                }
            }
            let ztz_g = &mut ztz[g * q * q..(g + 1) * q * q];
            let zty_g = &mut zty[g * q..(g + 1) * q];
            for a in 0..q {
                let za = zrow[a];
                zty_g[a] += za * yr;
                let dst = &mut ztz_g[a * q..a * q + q];
                for b in 0..q {
                    dst[b] += za * zrow[b];
                }
            }
            let ztx_g = &mut ztx[g * q * p..(g + 1) * q * p];
            for a in 0..q {
                let za = zrow[a];
                let dst = &mut ztx_g[a * p..a * p + p];
                for b in 0..p {
                    dst[b] += za * xrow[b];
                }
            }
        }

        Ok(Self {
            data: LmmData {
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
            },
        })
    }

    #[getter]
    fn n(&self) -> usize {
        self.data.n
    }
    #[getter]
    fn p(&self) -> usize {
        self.data.p
    }
    #[getter]
    fn q(&self) -> usize {
        self.data.q
    }
    #[getter]
    fn m(&self) -> usize {
        self.data.m
    }
    #[getter]
    fn n_theta(&self) -> usize {
        n_theta(self.data.q)
    }

    /// Profiled deviance at `theta`. Returns `inf` for an infeasible `theta`.
    ///
    /// A `theta` of the wrong length is a caller error, not an infeasible
    /// point: too short used to index out of bounds and abort the process,
    /// too long used to silently ignore the tail.
    #[pyo3(signature = (theta, reml=true))]
    fn deviance(&self, py: Python<'_>, theta: Vec<f64>, reml: bool) -> PyResult<f64> {
        self.check_theta(&theta)?;
        Ok(py.detach(|| {
            evaluate(&self.data, &theta, reml, false)
                .map(|e| e.deviance)
                .unwrap_or(f64::INFINITY)
        }))
    }

    /// Profiled deviance and its analytic gradient at `theta`.
    #[pyo3(signature = (theta, reml=true))]
    fn deviance_grad(
        &self,
        py: Python<'_>,
        theta: Vec<f64>,
        reml: bool,
    ) -> PyResult<(f64, Vec<f64>)> {
        self.check_theta(&theta)?;
        let nth = n_theta(self.data.q);
        Ok(
            py.detach(|| match evaluate(&self.data, &theta, reml, true) {
                Some(e) => (e.deviance, e.grad),
                None => (f64::INFINITY, vec![f64::NAN; nth]),
            }),
        )
    }

    /// Lower bounds on `theta`: diagonal entries >= 0, off-diagonals free.
    fn lower_bounds(&self) -> Vec<f64> {
        theta_index(self.data.q)
            .into_iter()
            .map(|(r, c)| if r == c { 0.0 } else { f64::NEG_INFINITY })
            .collect()
    }

    /// lme4's starting value: the identity relative covariance factor.
    fn default_theta(&self) -> Vec<f64> {
        theta_index(self.data.q)
            .into_iter()
            .map(|(r, c)| if r == c { 1.0 } else { 0.0 })
            .collect()
    }

    /// Fit by minimising the profiled criterion over `theta` alone.
    ///
    /// `starts` is a flat concatenation of candidate starting vectors; the best
    /// result wins. Multi-start costs almost nothing and removes the boundary
    /// cases where a single start stalls.
    #[pyo3(signature = (starts, reml=true, max_iter=300, gtol=1e-8, ftol=1e-12))]
    fn fit(
        &self,
        py: Python<'_>,
        starts: Vec<f64>,
        reml: bool,
        max_iter: usize,
        gtol: f64,
        ftol: f64,
    ) -> PyResult<Py<PyDict>> {
        let nth = n_theta(self.data.q);
        if let Some(pos) = starts.iter().position(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "starts[{pos}] is not finite"
            )));
        }
        if nth == 0 || starts.is_empty() || starts.len() % nth != 0 {
            return Err(PyValueError::new_err(
                "starts must be a non-empty multiple of the theta length",
            ));
        }
        let lower = self.lower_bounds();
        let settings = OptSettings {
            max_iter,
            gtol,
            ftol,
            memory: 10,
        };

        let best = py.detach(|| {
            let mut best: Option<optim::OptResult> = None;
            for chunk in starts.chunks(nth) {
                let r = minimize(
                    |t: &[f64]| evaluate(&self.data, t, reml, true).map(|e| (e.deviance, e.grad)),
                    chunk,
                    &lower,
                    &settings,
                );
                let better = match &best {
                    None => true,
                    Some(b) => r.fx < b.fx,
                };
                if better {
                    best = Some(r);
                }
            }
            best
        });

        let best =
            best.ok_or_else(|| PyRuntimeError::new_err("optimisation produced no result"))?;
        self.solution_dict(py, &best.x, reml, Some(&best))
    }

    /// Every derived quantity at a given `theta`, without re-optimising.
    #[pyo3(signature = (theta, reml=true))]
    fn solution(&self, py: Python<'_>, theta: Vec<f64>, reml: bool) -> PyResult<Py<PyDict>> {
        self.check_theta(&theta)?;
        self.solution_dict(py, &theta, reml, None)
    }
}

impl LmmCore {
    /// `theta` must have exactly `q(q+1)/2` entries, all finite.
    fn check_theta(&self, theta: &[f64]) -> PyResult<()> {
        let nth = n_theta(self.data.q);
        if theta.len() != nth {
            return Err(PyValueError::new_err(format!(
                "theta must have {} entries for q = {}, got {}",
                nth,
                self.data.q,
                theta.len()
            )));
        }
        if let Some(pos) = theta.iter().position(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!("theta[{pos}] is not finite")));
        }
        Ok(())
    }

    fn solution_dict(
        &self,
        py: Python<'_>,
        theta: &[f64],
        reml: bool,
        opt: Option<&optim::OptResult>,
    ) -> PyResult<Py<PyDict>> {
        let d = &self.data;
        let e = evaluate(d, theta, reml, true)
            .ok_or_else(|| PyRuntimeError::new_err("theta is infeasible"))?;

        let q = d.q;
        let p = d.p;
        let lam = theta_to_lambda(theta, q);

        // cov_re = sigma^2 * Lambda Lambda'
        let mut cov_re = vec![0.0; q * q];
        for a in 0..q {
            for b in 0..q {
                let mut s = 0.0;
                for k in 0..q {
                    s += lam[a * q + k] * lam[b * q + k];
                }
                cov_re[a * q + b] = e.sigma2 * s;
            }
        }
        // cov_beta = sigma^2 * (X'X - sum W'A^-1W)^-1
        let cov_beta: Vec<f64> = e.rxtrx_inv.iter().map(|v| v * e.sigma2).collect();
        // random effects on the data scale: b_i = Lambda u_i
        let mut b_all = vec![0.0; d.m * q];
        for i in 0..d.m {
            for a in 0..q {
                let mut s = 0.0;
                for k in 0..q {
                    s += lam[a * q + k] * e.u[i * q + k];
                }
                b_all[i * q + a] = s;
            }
        }

        let out = PyDict::new(py);
        out.set_item("theta", PyArray1::from_vec(py, theta.to_vec()))?;
        out.set_item("deviance", e.deviance)?;
        out.set_item("grad", PyArray1::from_vec(py, e.grad.clone()))?;
        out.set_item("beta", PyArray1::from_vec(py, e.beta.clone()))?;
        out.set_item(
            "cov_beta",
            PyArray2::from_vec2(py, &reshape(&cov_beta, p, p))?,
        )?;
        out.set_item("cov_re", PyArray2::from_vec2(py, &reshape(&cov_re, q, q))?)?;
        // These are m x q, so building them through Vec<Vec<f64>> would allocate
        // one small Vec per group. Reshaping a flat buffer avoids that entirely.
        out.set_item(
            "random_effects",
            PyArray1::from_vec(py, b_all).reshape([d.m, q])?,
        )?;
        out.set_item("u", PyArray1::from_vec(py, e.u.clone()).reshape([d.m, q])?)?;
        out.set_item("sigma2", e.sigma2)?;
        out.set_item("sigma", e.sigma2.sqrt())?;
        out.set_item("pwrss", e.pwrss)?;
        out.set_item("ldl2", e.ldl2)?;
        out.set_item("ldrx2", e.ldrx2)?;
        out.set_item("reml", reml)?;
        out.set_item("n", d.n)?;
        out.set_item("p", p)?;
        out.set_item("q", q)?;
        out.set_item("m", d.m)?;

        if let Some(o) = opt {
            out.set_item("converged", o.converged)?;
            out.set_item("iterations", o.iterations)?;
            out.set_item("fev", o.fev)?;
            out.set_item("grad_inf_norm", o.grad_inf_norm)?;
            out.set_item("message", o.message.clone())?;
        }
        Ok(out.into())
    }
}

fn reshape(v: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| v[r * cols..(r + 1) * cols].to_vec())
        .collect()
}

#[pymodule]
fn _mixedlm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LmmCore>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
