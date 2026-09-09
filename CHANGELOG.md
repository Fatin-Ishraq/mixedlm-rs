# Changelog

## 0.1.0 — unreleased

First release. Linear mixed-effects models with one grouping factor, matching
`statsmodels.MixedLM`'s API.

- **Profiled REML/ML** in the `lme4` formulation (Bates et al. 2015): the fixed
  effects and residual variance are eliminated analytically, so the optimiser
  sees only the 1–3 covariance parameters.
- **Block-diagonal Cholesky.** With one grouping factor the penalised system
  decomposes into `m` independent `q x q` blocks; no general sparse solver is
  used, and the blocks are factorised in a flat `rayon`-parallel loop. This is
  the largest single contributor to the speed, by a wide margin.
- **Analytic gradient** of the profiled criterion, which `lme4` and
  `MixedModels.jl` do not use (both optimise derivative-free with BOBYQA). Cuts
  objective evaluations from 44–64 to 11–13 against this package's own
  finite-difference stage, and is verified against central finite differences
  over the full product of `q = 1..3`, `p = 1,2,5,10`, both criteria, and the
  variance-zero boundary.
- **Exact conditioning of both designs.** The random-effects columns are scaled
  to unit RMS, and the response is offset by its OLS fit with the fixed-effect
  columns RMS-scaled. All are exact reparameterisations — the criterion surface
  is identical — and together they make results invariant to response
  translation and sensible to start from whatever units the data is in.
- **Boundary handling.** `theta = 0` is a stationary point of the profiled
  criterion for *any* data, because every gradient term vanishes at `Lambda = 0`.
  Diagonal entries landing on the bound are probed and restarted from, so a
  genuine singular fit is reported as converged while a spurious one is escaped.
- **Convergence is certified by stationarity**, never by the optimiser's own
  success flag, and the certificate drives a retry rather than annotating the
  result. Singular fits are reported separately, via `results.singular`.
- **Flat per-group storage.** Every intermediate for a group lives in one
  preallocated buffer rather than ~10 small `Vec`s, with rayon fold accumulators
  instead of per-group temporaries. A single objective evaluation is 8–12x
  faster as a result, which is most of a fit.
- **Variance-component standard errors are computed on first access**, not
  during the fit: the profiled Hessian costs `2 * n_theta` extra gradient
  evaluations and most callers only read the fixed effects.
- **Starts are chosen per optimiser, and both counts are measured.** scipy's
  L-BFGS-B takes one -- across the 120 fuzz fixtures, 1, 2 and 3 starts give
  identical outcomes for 27% more evaluations. The in-crate optimiser takes
  five, because with one it settles on a strictly worse stationary point on 2
  of 20 seeds.
- **Identifiability check.** On designs where it is plausible, the fit probes
  whether the criterion is flat along the variance split and warns if it is.
  `lme4` refuses such models; `statsmodels` fits them silently.
- `install()` / `uninstall()` alias only the mixed-model entry points; the rest
  of statsmodels is left untouched.
- Hypothesis tests on the fixed effects (`t_test`, `wald_test`, `f_test`),
  `params_object`, `df_resid`, labelled parameter views, and pickling.
- Wheels: one `abi3` wheel per platform covering Python 3.10 through 3.14.

Verified against lme4's published fits for `sleepstudy`, `Dyestuff` and the
singular `Dyestuff2`, under both REML and ML. 520 Python tests, 7 Rust tests.

Not implemented, and raising rather than ignored: variance components
(`vc_formula` / `exog_vc`), `fe_pen`, `cov_pen`, `free`, `fit_regularized`,
`profile_re`, `bootstrap`, `get_distribution`. GLMMs are out of scope. See
`docs/LIMITATIONS.md`.

### Release readiness

A second pass over the package as a shipped artifact rather than as an
algorithm. The theme is that a claim needs a check behind it, or it should not
be a claim.

**Correctness of the supported family**

- **`subset` selects by index label**, which is pandas' and statsmodels'
  meaning. It treated a non-boolean subset as row *positions*, so
  `subset=[10, 20]` on a frame indexed 100..199 quietly selected rows 10 and 20
  instead of raising, and on a string-labelled frame selected nothing.
  Selection is now by membership rather than `.loc`, so a duplicated index
  label returns each matching row once and an externally supplied `groups`
  stays aligned. `subset` and `groups` are resolved together against the
  unsubset frame, so they cannot end up half-aligned.
- **`fit()` validates its arguments before the optimiser runs.** Unknown
  keywords were rejected *after* the fit, so a caller waited out a full
  optimisation to be told the keyword was never read; tolerances, iteration
  counts and start counts were not checked at all. The tests time the
  rejection against a real fit, so "before the optimiser" is under test.
- **`method="rust"` is experimental** and warns. A sweep of all 120 fuzz seeds
  found it reaching a criterion 649 and 704 deviance units worse than the
  default path on seeds 32 and 54; both are kept as regression fixtures. It
  reports `converged=False` and warns, but — as with `statsmodels` — it still
  returns the estimate, and the documentation now says so.
- **`results.diagnostics`** carries what a failed fit needs to leave behind:
  optimiser, projected gradient, the tolerance required, evaluations,
  iterations, starts, and the optimiser's message.
- **Formula prediction survives `save`/`load`.** patsy cannot pickle a
  `DesignInfo`, so it is rebuilt from the frame the model was fitted on, which
  is what reproduces the state its transforms learned. `save(with_data=False)`
  drops that frame, and the error then names the real cause — it previously
  advised passing raw new data, which is exactly what had just failed.

**What is promised**

- **`docs/COMPATIBILITY.md`** replaces "change the import, that is the whole
  migration". Four sections — compatible, different type or coordinates,
  unsupported and raising, additions — produced by diffing the two results
  objects rather than from memory, and pinned by 72 tests that assert both
  halves of each claim.

**Evidence**

- **`bench/baseline.json`** is one recorded run — commit, environment, seeds,
  counts, individual losses, warning categories — and `tests/test_baseline.py`
  checks the documents against it. The 120-fixture counts had drifted to
  disagree across three documents (19/75 in one, 42/52 in two).
- **`tests/test_dense_reference.py`** checks the block-diagonal Cholesky
  against a dense implementation of the same criterion that shares no code
  with it, including the gradient by differencing the dense form.
- **`tests/test_review_regressions.py`** pins each specific wrong answer the
  review found, with the old behaviour written down.
- Benchmarks share `bench/tolerances.py`; the staged one no longer gates on a
  relative tolerance over a criterion carrying an arbitrary additive constant.

**Supply chain and platforms**

- **PyO3 0.23.5 → 0.29.2, rust-numpy 0.23 → 0.29**, clearing two RustSec
  advisories against the locked graph. `cargo audit` and `pip-audit` run in CI
  and fail the build; `SECURITY.md` records a reachability assessment for
  anything that remains.
- **The declared Rust floor was 1.74 and was wrong by nine minor versions.**
  pyo3 and numpy require 1.83, rayon 1.80. Raised to 1.83 and built with it.
- The Python floor job pinned `numpy==1.23.*`, which tests a release series
  rather than the advertised minimum; it now pins exact versions.
- **Typing is checked rather than counted.** `typing.get_type_hints()` failed
  on 18 members because `ArrayLike` was imported under `TYPE_CHECKING`;
  `mypy --strict` now accepts a representative-usage file in CI, and the
  compiled module has a stub.
- **`scripts/verify_release.py`** builds both artifacts, installs the exact
  wheel into a clean venv outside the checkout, rebuilds a wheel from the
  unpacked sdist, and runs the documented example, numerical smoke tests and
  save/load against each.

### Fixed before release, following an external review

The review is worth recording, because most of these were wrong answers rather
than missing features, and several were being reported as successes.

**Numerical**

- `pwrss` was computed as `y'y - beta'X'y - u'Lambda'Z'y`, a difference of large
  nearly equal quantities. With a response around 1e8 the fit returned a
  confidently converged wrong answer. On a 200-row random-intercept fixture,
  shifting `y` by 1e8 moved the log-likelihood from -174.042190 to -455.702427
  and the residual variance from 0.243 to 4.408, still reporting
  `converged=True`; lme4 and statsmodels are both invariant to the shift.
  Fixed by the OLS response offset above, after which all four shifts tested
  (0, 1e4, 1e6, 1e8) agree to the last printed digit.
- Convergence was `optimiser_flag or stationary`, so a loose `ftol` reported
  success at a projected gradient of 19.
- The stationarity tolerance keyed off the deviance, which shifts by a constant
  under rescaling; it is now scaled by the residual degrees of freedom.
- Boundary escape and certification ran only on the scipy path, so
  `method="rust"` still returned false zero variances marked converged.
- Extra starts were tried only after a reported failure, so a successful stop at
  a worse optimum was never challenged.
- Rank-deficient fixed effects surfaced as `RuntimeError: theta is infeasible`,
  an error naming the wrong thing entirely. Exact and numerical rank deficiency
  are now both detected before the optimiser starts, and reported distinctly.
- An infeasible point reached by the optimiser on a near-singular design let a
  bare `RuntimeError` escape `fit()`. Infeasibility is now handled: the driver
  retreats to points that are feasible by construction, and only raises -- with
  a message naming the design -- when nothing at all can be evaluated.
- `bse_re` inverted the profiled Hessian without checking it. At a boundary
  optimum it need not be positive definite, and the result looked like standard
  errors without being any. Non-positive or numerically flat directions now
  return NaN.
- The identifiability rule was wrong in both directions: `n <= q*m` neither
  implies a divergent likelihood (the criterion is *flat*) nor catches a
  confounded single group.
- Random starting values were built by multiplying the identity theta, so every
  off-diagonal stayed exactly zero and no start ever explored a correlated
  random-effects structure.

**Safety**

- An empty, oversized or non-finite `theta`, and zero-width `X` or `Z`, reached
  unchecked indexing in the core. Built with `panic="abort"`, that terminated
  the host process rather than raising. All now raise `ValueError`, and
  `tests/test_native_safety.py` checks each in a child process.

**Results**

- `random_effects_cov` returned the population covariance `G` for every group
  where the conditional covariance given that group's data was asked for, and
  every group shared one mutable frame.
- `summary()` printed `cov_re_unscaled` under the label "Group Var": 0.935 on
  `sleepstudy` where lme4 and statsmodels both report 612.1.
- `bse_re` returned the packed unscaled errors rather than the reference's
  `sqrt(scale)`-weighted ones — and the test compared against the reference's
  `bse` tail, so it passed while the attribute was wrong.

**API fidelity**

- Multi-column responses (`y1 + y2 ~ x`) silently fitted the first column.
- Missing rows were dropped independently for `endog`/`exog` and `exog_re`, and
  not at all for `groups`, so a missing group label became a group of its own.
- `.loc`-based alignment multiplied rows on a frame with duplicate index labels,
  and `subset` with an external `groups` array raised a length mismatch.
- `MixedLMParams` could not be passed to `fit`.
- `loglike`/`score`/`hessian` always evaluated REML, rejected the covariance-only
  packing the reference uses, and ignored `profile_fe`.
- `predict` could not rebuild a design for new data.
- `do_cg`, `full_output`, unknown optimiser names and unknown keyword arguments
  were all silently ignored.
- `VCSpec`'s positional order did not match the reference; `get_packed`
  defaulted to including the fixed effects and raised on a singular covariance.
- `group_list` was a property returning labels, not a method splitting an array.
- Fitted results could not be pickled.
- Missing: `t_test`, `wald_test`, `f_test`, `df_resid`, `params_object`.

**Documentation and benchmarks**

- `docs/DESIGN.md` claimed statsmodels optimises everything jointly and
  recomputes constants in its loop. Its own docstring says the likelihood is
  profiled over the scale and the fixed effects, its scores are analytic, and
  its cross-products are precomputed. Rewritten.
- The staged benchmark handed S3–S5 a pre-built core while charging S1/S2 for
  building one, timed S0/S1 once against best-of-three for the rest, and called
  `S0/S4` an end-to-end ratio although S0 alone parses a formula and computes
  inference. All corrected; `S6` is the like-for-like row.
- Benchmarks printed agreement without enforcing it. They now abort rather than
  report timings when the criterion or the coefficients disagree.
- The gradient was described as going beyond both references; Bates et al.
  derive the profiled ML gradient in the lme4 paper. Corrected.
- pymer4's overhead was attributed entirely to rpy2 marshalling, although its
  fit also routes through lmerTest.
- The 400-case adversarial sweep was not committed and dismissed two of its
  three losses using the incorrect divergence argument. It is now
  `bench/stress_sweep.py`, and every loss is reported with its absolute
  deviance gap.
- The source archive shipped the GPL-2 lme4 fixtures while the README said the
  fixtures are not distributed. Excluded from both artifacts.
- `patsy` was an optional extra although the documented first example needs it.
- CI never ran lint, covered 3 of the 5 declared Python versions, and did not
  test the declared dependency floor, the MSRV, the sdist contents or a clean
  install.
