# Changelog

## 0.1.0 — unreleased

First release. Linear mixed-effects models with one grouping factor, matching
`statsmodels.MixedLM`'s API.

- **Profiled REML/ML** in the `lme4` formulation (Bates et al. 2015): the fixed
  effects and residual variance are eliminated analytically, so the optimiser
  sees only the 1–3 covariance parameters instead of all of them jointly.
- **Block-diagonal Cholesky.** With one grouping factor the penalised system
  decomposes into `m` independent `q x q` blocks; no general sparse solver is
  used, and the blocks are factorised in a flat `rayon`-parallel loop.
- **Analytic gradient** of the profiled criterion, which `lme4` and
  `MixedModels.jl` do not use (both optimise derivative-free with BOBYQA). Cuts
  objective evaluations from 72–148 to 10–12, and is verified against central
  finite differences across `q = 1..3`, `p = 1,2,5,10`, both criteria and the
  variance-zero boundary.
- **Boundary handling.** `theta = 0` is a stationary point of the profiled
  criterion for *any* data, because every gradient term vanishes at `Lambda = 0`.
  Diagonal entries landing on the bound are probed and restarted from, so a
  genuine singular fit is reported as converged while a spurious one is escaped.
- **Exact internal rescaling** of the random-effects design to unit column RMS,
  which leaves the criterion surface identical but makes `theta = I` a sensible
  start whatever units the data is in.
- **Flat per-group storage.** Every intermediate for a group lives in one
  preallocated buffer rather than ~10 small `Vec`s, with rayon fold accumulators
  instead of per-group temporaries. A single objective evaluation is 8-12x
  faster as a result, which is most of a fit.
- **Variance-component standard errors are computed on first access**, not
  during the fit: the profiled Hessian costs `2 * n_theta` extra gradient
  evaluations and most callers only read the fixed effects.
- **Lazy multi-start.** Extra starting values are tried only when the first fails
  to converge; measured across 120 randomised fixtures, an unconditional 3-way
  multi-start produced identical outcomes for 1.4x the objective evaluations.
- `install()` / `uninstall()` alias only the mixed-model entry points; the rest
  of statsmodels is left untouched.
- Wheels: one `abi3` wheel per platform covering Python 3.10 through 3.14.

Verified against lme4's published fits for `sleepstudy`, `Dyestuff` and the
singular `Dyestuff2`, under both REML and ML. 206 Python tests, 7 Rust tests.

Not implemented, and raising rather than ignored: variance components
(`vc_formula` / `exog_vc`), `fe_pen`, `cov_pen`, `free`, `profile_re`,
`bootstrap`, `get_distribution`. GLMMs are out of scope. See
`docs/LIMITATIONS.md`.
