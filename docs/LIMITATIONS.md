# Limitations

Stated plainly, because a drop-in that hides its gaps is worse than one that
does not have them. Everything below either raises `NotImplementedError` or is
documented as accepted-and-ignored with a warning. Nothing is silently dropped.

## Not implemented, and refused rather than ignored

| what | why it raises |
|---|---|
| **Variance components** (`vc_formula`, `exog_vc`, `VCSpec` passed to a model) | Crossed and nested random effects break the block-diagonal structure this release is built on and need a genuine sparse Cholesky with a fill-reducing ordering. This is the largest piece of unfinished work. |
| **`fe_pen`, `cov_pen`** | Penalised fitting is a separate code path. Accepting the argument and ignoring it would report a different model's numbers under your specification. |
| **`free`** | Same reasoning: it constrains which parameters move. |
| **`profile_re`** | Likelihood profiling for confidence intervals; needs repeated refits with one component held fixed. |
| **`bootstrap`, `get_distribution`** | Post-fit machinery not yet ported. |
| **`MixedLM.get_scale`** | An internal statsmodels helper tied to its parameterisation. Use `MixedLMResults.scale`. |

`VCSpec` can still be *constructed* and imported, so code that references the
name keeps working; only passing one to a model raises.

## Generalised linear mixed models

**Out of scope.** Binomial and Poisson random-effects models (`cbpp`-style data)
are a substantially harder problem — there is no closed-form profiled criterion,
and the standard approaches (Laplace approximation, adaptive Gauss-Hermite
quadrature) are a separate project. This package is linear mixed models only,
which is what `statsmodels.MixedLM` is.

## Accepted and ignored, with a warning

- **`niter_sa`** — describes statsmodels' simulated-annealing warm-up, which this
  optimiser does not have.
- **`do_cg`**, **`full_output`** — accepted for signature compatibility.
- **`use_sparse`** in `from_formula` — accepted; the block-diagonal path is
  already the efficient one for a single grouping factor.

## Behavioural differences

- **`method=`** selects the optimiser driver, not statsmodels' solver names.
  The default is scipy's L-BFGS-B over the Rust objective and analytic gradient;
  `method="rust"` uses the in-Rust projected L-BFGS, which needs roughly three
  times as many objective evaluations and exists for callers who want no scipy
  in the loop.
- **`install()` aliases only the mixed-model entry points** — `MixedLM`,
  `MixedLMResults`, `MixedLMParams` and `smf.mixedlm`. It deliberately does not
  shadow the rest of statsmodels, which does far more than mixed models and must
  keep working untouched.
- **`__version__`** reports this package's version, not a statsmodels one. The
  API level being emulated is `__statsmodels_version__`.
- **`profile=True`-style profiling of the fit** no longer shows which internal
  step is slow: a cProfile trace shows one opaque call into Rust.

## Scale

Tested up to **125,066 groups / 500,264 observations** (5.8 s). There is no
hard limit; memory is `O(m q^2 + m q p)` for the per-group cross-products, which
at that size is a few hundred megabytes.

`q` (random effects per group) is expected to be small — 1 to 4. The per-group
work is `O(q^3 + q^2 p + q p^2)`, so very large `q` is not what this structure is
for.
