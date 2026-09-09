# Limitations

Stated plainly, because a drop-in that hides its gaps is worse than one that
does not have them. Everything below either raises `NotImplementedError`, is
documented as accepted-and-ignored with a warning, or is named here as a known
divergence. Nothing is silently dropped.

## The big one: one grouping factor

This release fits models with **a single grouping factor**. Crossed and nested
random effects raise `NotImplementedError` rather than silently fitting a
different model.

This matters, and it is worth being concrete about who it excludes. It covers
repeated measurements within one cluster hierarchy — subjects measured over
time, patients within one clinic layer, plots within sites — which is a large
share of applied use. It does **not** cover:

- **Crossed subject and item effects**, `(1|subject) + (1|item)`. This is the
  standard design in psycholinguistics and much of experimental psychology, and
  the "keep it maximal" literature is specifically about it.
- **Three-level nesting**, students within classrooms within schools, which is
  the ordinary shape of education research.
- **Multisite longitudinal designs** with both a site effect and a subject
  effect.
- **Independent variance structures** that need `free` to hold parameters
  fixed.

If your model has two `(...|...)` terms, this package cannot fit it today. Use
`lme4` or `statsmodels` for those.

One correction to an earlier version of this document: it said nested effects
"break the block-diagonal structure". That is too strong. Independent top-level
clusters retain block structure, just with larger blocks, and a general sparse
Cholesky with a fill-reducing ordering is the engineering choice made for the
general case — not a mathematical necessity in every nested design.

## Not implemented, and refused rather than ignored

| what | why it raises |
|---|---|
| **Variance components** (`vc_formula`, `exog_vc`, `VCSpec` passed to a model) | See above. This is the largest piece of unfinished work. |
| **`fe_pen`, `cov_pen`** | Penalised fitting is a separate code path. Accepting the argument and ignoring it would report a different model's numbers under your specification. |
| **`free`** | Same reasoning: it constrains which parameters move. |
| **`MixedLM.fit_regularized`** | L1-penalised fixed effects; a different estimator, not a faster path to this one. |
| **`profile_re`** | Likelihood profiling for confidence intervals; needs repeated refits with one component held fixed. |
| **`bootstrap`** | Post-fit resampling, not yet ported. |
| **`get_distribution`** (model and results) | Post-fit machinery not yet ported. |
| **`MixedLM.get_scale`** | An internal statsmodels helper tied to its parameterisation. Use `MixedLMResults.scale`. |
| **`loglike(..., profile_fe=False)`** | The criterion here eliminates the fixed effects analytically, so there is no un-profiled surface to evaluate. |

`VCSpec` can still be *constructed* and imported, so code that references the
name keeps working; only passing one to a model raises.

## Generalised linear mixed models

**Out of scope.** Binomial and Poisson random-effects models (`cbpp`-style data)
are a substantially harder problem — there is no closed-form profiled criterion,
and the standard approaches (Laplace approximation, adaptive Gauss-Hermite
quadrature) are a separate project. This package is linear mixed models only,
which is what `statsmodels.MixedLM` is.

## Inference: what the standard errors do and do not cover

- **Fixed-effect covariance is conditional on the fitted variance parameters.**
  `cov_params()[:k_fe, :k_fe]` is the GLS covariance
  `sigma^2 (X' V(theta_hat)^-1 X)^-1`. That is what lme4 and statsmodels report,
  but it does not propagate uncertainty in `theta`, so it is not in general the
  fixed-effect block of the inverse full observed information, and it is mildly
  anti-conservative in small samples. **Kenward–Roger and Satterthwaite
  corrections are not implemented.** If your design is small and the *p*-values
  matter, that is a real gap; `lmerTest` in R does this and `pymer4` exposes it.
- **`cov_params()` has no cross-block terms.** The variance-component rows carry
  delta-method variances on the diagonal and `NaN` off-diagonal. Joint inference
  across fixed effects and variance parameters is not available.
- **`bse_re` follows statsmodels' definition**, which is `sqrt(scale)` times the
  standard errors of the *unscaled* covariance parameters that `bse` reports.
  Note the reference's own inconsistency, preserved here for compatibility: the
  value tabulated next to those errors in `summary()` is `cov_re`, which is
  `scale` times the unscaled parameter. `bse_cov_re` gives errors on the same
  scale as `cov_re`.
- **Wald intervals for variance parameters are unreliable near zero.** Their
  sampling distribution is skewed and degenerate at the boundary. lme4 declines
  to report these at all. `results.singular` flags when a component is at the
  boundary; treat those rows as point estimates, not as inputs to a *z*-test.
- **The optimum is not certified global.** The fit verifies that it stopped at a
  stationary point and restarts when it did not, and it probes away from the
  boundary where the criterion has a stationary point for any data at all. That
  is a local certificate, not a global one: the profiled criterion can be
  multimodal and nothing here proves the reported optimum is the best one.
  Additional random starts are available via `n_starts=`, though across the 120
  randomised fuzz fixtures they changed no answer.

## Numerical scope

- **Rank-deficient fixed effects are refused**, not pivoted away. This covers
  both exact collinearity — a duplicated column, a redundant categorical
  coding, a constant term alongside the intercept — and *numerical* rank
  deficiency, where two predictors agree to within eight digits after the
  columns are scaled to unit norm. Both raise a `ValueError` that names which
  case it is and reports the singular-value ratio. statsmodels and lme4 drop
  columns instead; here you drop the redundant one yourself.

  The numerical threshold exists because the normal equations square the
  condition number: a design that is collinear to 1e-9 is past what a double
  can represent once squared, and the penalised Cholesky then fails inside the
  optimiser and surfaces as `theta is infeasible` — an error naming entirely
  the wrong thing. A design collinear to about 1e-6 still fits, and may report
  `converged=False`, which is the honest answer for a criterion that is nearly
  flat in one direction.
- **Precision is bounded by the data, not the algorithm.** The response is
  offset by its OLS fit and the fixed-effect columns are RMS-scaled before any
  cross-product is formed, so results are invariant to response translation. But
  a response stored as `1e8 + O(1)` only determines its own residuals to about
  `2e-8` absolute, and no rearrangement recovers digits that are not in the
  input. lme4 sits at the same limit.
- **Identifiability is checked, not enforced.** When the count of random effects
  makes it plausible, the fit probes whether the criterion is flat along the
  variance split and warns if it is. A criterion that is unbounded rather than
  flat shows up as a convergence warning instead. Neither is refused, because
  refusing would break the drop-in contract.

## Accepted and ignored, with a warning

- **`niter_sa`** — describes statsmodels' simulated-annealing warm-up, which this
  optimiser does not have.
- **`do_cg`**, **`full_output`** — accepted for signature compatibility; passing
  a non-default value warns.
- **`use_sparse`** in `from_formula` — accepted; the block-diagonal path is
  already the efficient one for a single grouping factor.
- **statsmodels optimiser names** (`"bfgs"`, `"lbfgs"`, `"cg"`, …) — accepted and
  warned about, because the criterion being optimised is not the same one. An
  unrecognised string raises.

Unknown keyword arguments to `MixedLM(...)` and `fit(...)` raise `TypeError`.

## API surface that is present but not identical

- **Formula fits return plain ndarrays**, where statsmodels returns labelled
  pandas objects for `fe_params`, `cov_re` and friends. Name-based access is
  available through `fe_params_labelled`, `params_labelled` and `param_names`.
- **`summary()` is text only.** No `.tables`, no `.as_html()`, no `.as_latex()`.
- **`t_test` / `wald_test` / `f_test` cover the fixed effects only**, and return
  a minimal result object, not statsmodels' `ContrastResults`.
- **`use_t` is stored and honoured by `t_test`**, but `pvalues` are always
  normal-based, as in the reference.
- **`score` and `hessian` are in the internal `theta` coordinates** — the
  entries of the relative covariance factor — not statsmodels' packed covariance
  parameters, and `hessian` is `(k_re2, k_re2)`, not the reference's full square.
  They are not term-by-term comparable with the reference's.
- **Pickling drops patsy's `DesignInfo`.** patsy declines to pickle it
  (pydata/patsy#26), so a formula-fitted model that has been through a pickle
  can no longer rebuild a design from raw new data; `predict` on an
  already-built design matrix still works. `MixedLMResults.save`/`.load` are
  provided for the common case.

## Behavioural differences

- **`method=`** selects the optimiser driver, not statsmodels' solver names.
  The default is scipy's L-BFGS-B over the Rust objective and analytic gradient;
  `method="rust"` uses the in-Rust projected L-BFGS, which needs roughly three
  times as many objective evaluations and exists for callers who want no scipy
  in the loop. Both paths run the same boundary escape and the same
  stationarity certification.

  They do **not** get the same number of starts, and that is deliberate. The
  weaker line search finds worse basins: with a single start it settles on a
  strictly worse *stationary* point on 2 of 20 fuzz seeds, one of them
  legitimately reported as converged, because a local optimum is what it is.
  Certification cannot fix that — the certificate is local by construction. So
  the rust path defaults to five starts and the scipy path to one, both
  measured. Even so, `method="rust"` is the fallback, not the recommendation.
- **`install()` aliases only the mixed-model entry points** — `MixedLM`,
  `MixedLMResults`, `MixedLMParams` and `smf.mixedlm`. It deliberately does not
  shadow the rest of statsmodels, which does far more than mixed models and must
  keep working untouched.
- **`__version__`** reports this package's version, not a statsmodels one. The
  API level being emulated is `__statsmodels_version__`.
- **`profile=True`-style profiling of the fit** no longer shows which internal
  step is slow: a cProfile trace shows one opaque call into Rust.

## Typing

The public API carries **no type annotations** and ships no `py.typed` marker.
Type checkers will treat this package as untyped and say so, which is accurate.

## Platforms actually tested

Built and tested on **Windows 11, Python 3.11 and 3.14**. The wheel is `abi3`
for Python ≥ 3.10 and the package declares support for 3.10–3.14, but Linux,
macOS and Python 3.10/3.12/3.13 have not been exercised. The CI workflow that
would cover them is committed but has never run, because the repository has no
remote. Treat the support range as declared-but-unverified until it has.

## Scale

Tested up to **125,066 groups / 500,264 observations** (about 0.18 s; see
docs/BENCHMARKS.md). There is no hard limit; memory is `O(m q^2 + m q p)` for
the per-group cross-products, which at that size is a few hundred megabytes.

`q` (random effects per group) is expected to be small — 1 to 4. The per-group
work is `O(q^3 + q^2 p + q p^2)`, so very large `q` is not what this structure is
for.
