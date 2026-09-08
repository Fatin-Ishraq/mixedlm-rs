<div align="center">

# mixedlm-rs

**`lmer` for Python.**
Linear mixed-effects models with `lme4` speed, the `statsmodels` API, and no R required.

[![PyPI](https://img.shields.io/pypi/v/mixedlm-rs.svg)](https://pypi.org/project/mixedlm-rs/)
[![Python](https://img.shields.io/badge/python-3.10%20–%203.14-blue.svg)](https://pypi.org/project/mixedlm-rs/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## What this is for

You measured the same subjects more than once. Or students inside classrooms,
patients inside clinics, plots inside sites, trials inside people. The
observations are not independent, and a plain regression will tell you things
that are not true.

**Mixed-effects models** are the standard answer, and they are the workhorse of
experimental science — clinical trials, psychology, neuroscience, ecology,
linguistics, education research, pharmacology.

Python's implementation has been the weak link. This is a drop-in replacement
for it.

```bash
pip install mixedlm-rs
```

```python
import pandas as pd
import mixedlm_rs as mlm

sleep = pd.read_csv("sleepstudy.csv")     # reaction times over 10 days of sleep deprivation

model = mlm.mixedlm("Reaction ~ Days", sleep,
                    groups=sleep["Subject"],   # each subject measured 10 times
                    re_formula="~Days")        # and each has their own slope
print(model.fit().summary())
```

```
                    Mixed Linear Model Regression Results
==============================================================================
Model:                MixedLM           Dependent Variable:                  y
No. Observations:     180               Method:                           REML
No. Groups:           18                Scale:                        654.9410
Min. group size:      10                Log-Likelihood:              -871.8141
Max. group size:      10                Converged:                         Yes
------------------------------------------------------------------------------
                         Coef.  Std.Err.        z    P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
Intercept              251.405     6.825   36.838    0.000   238.029   264.781
Days                    10.467     1.546    6.771    0.000     7.438    13.497
Group Var                0.935     0.464
Group x Days Cov         0.015     0.071
Days Var                 0.054     0.024
==============================================================================
```

Those are `lme4`'s published numbers for this dataset, to every digit it prints.

## Already using statsmodels?

Change the import. That is the whole migration.

```diff
- from statsmodels.regression.mixed_linear_model import MixedLM
+ from mixedlm_rs import MixedLM
```

Same classes, same arguments, same `params` packing, same `summary()` layout.

If you cannot edit the code that imports it — someone else's library, a notebook
you were handed:

```python
import mixedlm_rs
mixedlm_rs.install()      # before anything imports statsmodels' MixedLM

import statsmodels.api as sm       # sm.MixedLM is now ours
import statsmodels.formula.api as smf   # smf.mixedlm is now ours
```

`install()` aliases **only** the mixed-model entry points. statsmodels does far
more than mixed models, and the rest of it is left untouched.

## It is not just faster. It is more often right.

This is the part that matters more than the speed.

`statsmodels.MixedLM` does not merely take a long time — on ordinary inputs it
**fails to converge and returns estimates anyway**, after retrying `bfgs`, then
`lbfgs`, then `cg`, and giving up with a gradient norm in the hundreds.

When that happens, a researcher either does not notice, or starts deleting
random-effects terms until the fit converges. That second response is a
documented source of anti-conservative *p*-values — the "keep it maximal"
literature in psycholinguistics exists because of exactly this.

Across **120 randomised fixtures**, comparing against `statsmodels`:

| outcome | count |
|---|---:|
| statsmodels did not converge | **26** |
| mixedlm-rs found a strictly **better** optimum | **19** |
| same optimum | 75 |
| mixedlm-rs found a **worse** optimum | **0** |

On 37.5% of fixtures the reference either failed or landed somewhere worse.

Singular fits — a variance component genuinely at zero — are reported as
converged, because a boundary optimum **is** a converged fit. That distinction
is what stops people from mangling their model to silence a warning.

## Speed

| n | groups | statsmodels | mixedlm-rs | |
|---:|---:|---:|---:|---|
| 10,000 | 500 | 1.72 s | **0.033 s** | 52x |
| 40,000 | 5,000 | 11.80 s | **0.107 s** | 110x |
| 100,000 | 20,000 | 41.98 s | **0.638 s** | 66x |
| 200,000 | 50,000 | 106.03 s | **2.019 s** | 53x |
| 500,264 | **125,066** | — | **5.840 s** | |

Every row is checked for agreement before it is timed.

That last row is the size from
[statsmodels#9097](https://github.com/statsmodels/statsmodels/issues/9097),
where a user reported waiting **41 minutes** for a fit that R's `lmer` did in
1–2 seconds.

**Where the win comes from — honestly.** It is mostly not Rust:

| step | multiplier |
|---|---:|
| profiled REML alone | 2.1x |
| + batched block-diagonal Cholesky | **33.6x** |
| + Rust core | 1.5x |
| + analytic gradient | 6.4x |

The formulation is `lme4`'s: eliminate the fixed effects and `sigma^2`
analytically so the optimiser sees only the 1–3 covariance parameters, and
factorise the block-diagonal penalised system instead of applying a dense
Sherman-Morrison-Woodbury update per group per iteration. Stages 1 and 2 are
reproducible by anyone in NumPy — `proto/preml.py` is that implementation, in
about 200 lines. [Full tables →](docs/BENCHMARKS.md)

## One thing lme4 does not do

`lme4` and `MixedModels.jl` both optimise the covariance parameters
**derivative-free** (BOBYQA). We derive and evaluate the analytic gradient of the
profiled criterion instead, which cuts objective evaluations from 72–148 down to
10–12.

Because a wrong gradient does not crash — it converges quietly to the wrong
answer — it is checked against central finite differences across every
combination of random-effect count, fixed-effect count, both criteria, and at
the variance-zero boundary. [The derivation →](docs/DESIGN.md)

## What is included

| | |
|---|---|
| **Models** | `MixedLM`, `MixedLM.from_formula`, `mixedlm` |
| **Results** | `fe_params`, `cov_re`, `cov_re_unscaled`, `scale`, `params`, `bse`, `bse_fe`, `bse_re`, `tvalues`, `pvalues`, `llf`, `aic`, `bic`, `random_effects`, `random_effects_cov`, `fittedvalues`, `resid`, `conf_int`, `cov_params`, `predict`, `summary` |
| **Parameters** | `MixedLMParams` with `from_packed` / `get_packed` / `from_components` |
| **Criteria** | REML (default) and ML |
| **Aliasing** | `install()` / `uninstall()` |

**Scoped to one grouping factor.** Crossed and nested random effects
(`vc_formula`) break the block-diagonal structure this is built on and need a
sparse Cholesky with a fill-reducing ordering — that is the next release. They
raise `NotImplementedError` rather than silently fitting a different model, as
do `fe_pen`, `cov_pen` and `free`. GLMMs are out of scope.
[Every gap, stated plainly →](docs/LIMITATIONS.md)

## Is it actually the same?

That is the only question that matters for a drop-in, so it is what the test
suite is built around — **206 Python tests and 7 Rust tests**.

The primary oracle is **lme4's published fits**, not statsmodels, because
statsmodels is the thing that is wrong on some inputs. `sleepstudy`, `Dyestuff`
and the singular `Dyestuff2` are reproduced to every published digit, under both
REML and ML.

Differential testing found two real defects during development:

- **`fittedvalues` returned the marginal fit.** statsmodels' is the *conditional*
  fit, including the random effects.
- **`theta = 0` is a stationary point for any data whatsoever.** At `Lambda = 0`
  every term of the gradient vanishes identically, so a gradient-based optimiser
  that reaches the bound stops there and reports success — even when the true
  optimum is an ordinary non-zero variance. Fixed by probing away from the bound
  and restarting.

[What is verified, and every divergence →](docs/CORRECTNESS.md)

## Development

```bash
pip install maturin pytest numpy scipy pandas patsy statsmodels
python -m maturin build --release --out dist
pip install --force-reinstall --no-deps --no-index --find-links dist mixedlm-rs
pytest tests/ -q
cargo test --lib
python bench/scaling.py
```

## Licence and credit

MIT.

The algorithm is Douglas Bates and colleagues'. This package implements the
formulation published in:

> Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting Linear
> Mixed-Effects Models Using lme4.* Journal of Statistical Software, 67(1),
> 1–48.

The API mirrors `statsmodels` (BSD-3), by the statsmodels developers.

The `lme4` datasets used to verify correctness (`sleepstudy`, `Dyestuff`,
`Dyestuff2` and others) are GPL-2 and live in `data/` as **test fixtures only**.
They are not part of the distributed package — the wheel contains nothing but
the Python module and the compiled extension. See [data/README.md](data/README.md).
