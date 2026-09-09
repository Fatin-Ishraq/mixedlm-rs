<div align="center">

# mixedlm-rs

**`lmer` for Python.**
Linear mixed-effects models with `lme4` speed, the `statsmodels` API, and no R required.

[![CI](https://github.com/Fatin-Ishraq/mixedlm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/mixedlm-rs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20–%203.14-blue.svg)](https://github.com/Fatin-Ishraq/mixedlm-rs)
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

**Not on PyPI yet.** Build it from a checkout — you need a Rust toolchain:

```bash
git clone https://github.com/Fatin-Ishraq/mixedlm-rs
cd mixedlm-rs
pip install maturin && maturin build --release --out dist
pip install --find-links dist mixedlm-rs
```

The build is `abi3`, so one wheel per platform will cover Python 3.10–3.14 once
these are published, and there will be nothing to compile.

**Currently verified:** the Windows x86-64 wheel and the sdist, both built and
installed into clean environments outside the source tree
(`python scripts/verify_release.py`, 69/69 checks). The Linux and macOS wheels
come out of the same `maturin` configuration and their CI jobs are written; read
the CI badge above for whether those jobs have gone green. Until they have,
treat non-Windows wheels as expected to work rather than as tested.

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
Model:                MixedLM           Dependent Variable:           Reaction
No. Observations:     180               Method:                           REML
No. Groups:           18                Scale:                        654.9410
Min. group size:      10                Log-Likelihood:              -871.8141
Max. group size:      10                Converged:                         Yes
Mean group size:      10.0              Singular fit:                       No
------------------------------------------------------------------------------
                         Coef.  Std.Err.        z    P>|z|    [0.025    0.975]
------------------------------------------------------------------------------
Intercept              251.405     6.825   36.838    0.000   238.029   264.781
Days                    10.467     1.546    6.771    0.000     7.438    13.497
Group Var              612.090    11.881
Group x Days Cov         9.604     1.821
Days Var                35.072     0.610
==============================================================================
```

`lme4` reports this fit as `sd(Intercept) = 24.741`, `sd(Days) = 5.922`,
`corr = 0.066`, `sd(Residual) = 25.592`, REML criterion `1743.6284`. Squaring
those standard deviations gives 612.12 and 35.07, and the criterion is
`-2 x -871.8141 = 1743.6282`. The variance rows are printed on the same scale
statsmodels prints them, so the two summaries are directly comparable.

## Already using statsmodels?

For most code, changing the import is the whole migration.

```diff
- from statsmodels.regression.mixed_linear_model import MixedLM
+ from mixedlm_rs import MixedLM
```

Same classes, same `params` packing, same `summary()` layout. The
arguments match too, with the exceptions catalogued in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md): a few statsmodels
arguments are accepted and warned about rather than honoured, and some
attributes are plain arrays where statsmodels returns a DataFrame.

**Three things decide whether that holds for you**, and it is worth two minutes
to check rather than finding out later:

1. **One grouping factor.** Crossed or nested random effects — two `(...|...)`
   terms — are not supported and raise. This is the biggest gap in the package.
2. **Result containers are ndarrays, not Series.** `result.fe_params["x"]`
   raises; `result.fe_params[1]` works. Use `params_labelled` or
   `param_names` for name-based access.
3. **Some methods are absent**, including `summary().tables` / `.as_html()`,
   `wald_test_terms`, `t_test_pairwise` and the `bsejac` family.

Everything else — every attribute, every type difference, every method that
raises and why — is enumerated in
**[the compatibility contract](docs/COMPATIBILITY.md)**, which was produced by
diffing the two results objects rather than from memory.

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
| mixedlm-rs found a strictly **better** optimum | **42** |
| same optimum | 52 |
| mixedlm-rs found a **worse** optimum | **0** |

On 56.7% of fixtures the reference either failed or landed somewhere worse.
Reproduce the counts with `python bench/differential_table.py`.

These are deliberately difficult fixtures — that is what they are for — so the
rate is not an estimate of how often statsmodels fails on ordinary published
datasets, and should not be read as one.

Singular fits — a variance component genuinely at zero — are reported as
converged, because a boundary optimum **is** a converged fit. That is what stops
people from mangling their model to silence a warning. They are also flagged
*separately*, as `results.singular`, because the estimate being legitimate does
not make the usual Wald intervals around it legitimate: at the boundary the
sampling distribution of a variance parameter is degenerate, and `bse_re`
returns `NaN` there rather than a number that would be read as a standard error.
`lme4` draws the same distinction with `isSingular`.

**Convergence is certified, not reported.** The optimiser's own success flag is
never taken as the answer: it says its termination rule fired, not that the
point is stationary. The projected gradient is checked against a tolerance that
does not depend on the units, and a failed check triggers a restart rather than
a warning.

## Speed

| n | groups | statsmodels | mixedlm-rs | |
|---:|---:|---:|---:|---|
| 10,000 | 500 | 2.31 s | **0.011 s** | 218x |
| 40,000 | 5,000 | 16.59 s | **0.019 s** | 870x |
| 100,000 | 20,000 | 61.00 s | **0.042 s** | 1454x |
| 200,000 | 50,000 | 160.48 s | **0.087 s** | 1836x |
| 500,264 | **125,066** | — | **0.197 s** | |

Agreement is **enforced**, not reported: the benchmark aborts rather than print
a timing if the fixed effects differ by more than 0.05 of a standard error, the
variance components by more than 2%, or our criterion falls below the
reference's. On every row above it does not — the largest fixed-effect
difference is 3.8e-05 standard errors, and our log-likelihood is never lower.

That last row matches the *size* reported in
[statsmodels#9097](https://github.com/statsmodels/statsmodels/issues/9097) —
125,066 groups — where a user reported waiting **41 minutes** for a fit that R's
`lmer` did in 1–2 seconds. To be clear about what that is and is not: the 41
minutes is their report on their own data, not a measurement made here. This
row is synthetic data at the same group count, so it shows that the size is not
the obstacle. It is not a reproduction of their categorical-design dataset, and
should not be read as a measured 41-minutes-to-0.18-seconds result.

### Faster than lme4 itself

`statsmodels` is what this replaces, but `lme4` in R is the strongest
implementation in the market, and `pymer4` — which calls `lme4` through rpy2 —
is the only way a Python user gets genuine `lme4` results today. Same models,
byte-identical data:

| n | groups | mixedlm-rs | lme4 (R) | pymer4 | vs lme4 | vs pymer4 |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 500 | **0.010 s** | 0.080 s | 1.71 s | 8x | 166x |
| 40,000 | 5,000 | **0.017 s** | 0.330 s | 11.67 s | 19x | 683x |
| 100,000 | 20,000 | **0.038 s** | 1.120 s | 118.10 s | 30x | 3,127x |
| 200,000 | 50,000 | **0.078 s** | 2.510 s | 747.28 s | 32x | 9,554x |
| 500,264 | **125,066** | **0.169 s** | 7.330 s | — | **43x** | — |

**The log-likelihood matches `lme4` to six decimals on every fixture.** Same
answer, 8–43x faster, and the margin widens with the group count.

`pymer4` *is* `lme4`, so the gap between those two columns is what a Python
caller pays on top of the fit — and it compounds: 21x at 10,000 rows, **298x at
200,000**, where 2.51 s of the 747.28 s is the `lmer` fit. The remaining 744.77 s
is *not* a measurement of marshalling alone: `pymer4` also runs lmerTest for
Satterthwaite degrees of freedom and extracts the results as R objects, and the
benchmark does not separate those from serialisation. It also still needs R,
Rtools and a writable R library wherever your code runs.
[Full tables and caveats →](docs/BENCHMARKS.md)

**Where the win comes from — honestly.** The largest single factor is
structural, not the language:

| step | multiplier |
|---|---:|
| profiled REML — *upper bound only* | 1.7x |
| + batched block-diagonal Cholesky | **30.8x** |
| + Rust core | 14.6x |
| + analytic gradient | 2.8x |
| **end to end, like for like** | **182x** |

That last row compares the public API against `statsmodels` on the same
DataFrame, both parsing a formula, fitting, and computing inference. An earlier
version of this table published **1844x** for the same row, which was not a
like-for-like comparison: it measured the bare Rust objective against
statsmodels doing all of that extra work. [What was wrong, in
detail →](docs/BENCHMARKS.md)

The formulation is `lme4`'s: eliminate the fixed effects and `sigma^2`
analytically so the optimiser sees only the 1–3 covariance parameters, and
factorise the block-diagonal penalised system instead of applying a dense
Sherman-Morrison-Woodbury update per group per iteration. Stages 1 and 2 are
reproducible by anyone in NumPy — `proto/preml.py` is that implementation, in
about 200 lines. [Full tables →](docs/BENCHMARKS.md)

## The optimiser uses a gradient; lme4's does not

`lme4` and `MixedModels.jl` both optimise the covariance parameters
**derivative-free** (BOBYQA). Here the analytic gradient of the profiled REML
criterion is evaluated alongside the criterion itself and handed to L-BFGS-B,
which cuts objective evaluations from 44–64 to 11–13 on the benchmark
fixtures.

Two caveats worth stating plainly. The gradient of the profiled criterion is
not new — Bates et al. derive the ML version in the lme4 paper (eq. 46–48), and
`MixedModels.jl` documents derivative support; what is here is a REML gradient
specialised to the block structure, computed in the passes that already produce
the criterion. And the 44–64 figure is *this* package's own finite-difference
stage, not BOBYQA: no claim is made about lme4's evaluation count, which was
not measured.

Because a wrong gradient does not crash — it converges quietly to the wrong
answer — it is checked against central finite differences over the full product
of random-effect count, fixed-effect count and criterion, and at the
variance-zero boundary. [The derivation →](docs/DESIGN.md)

## What is included

| | |
|---|---|
| **Models** | `MixedLM`, `MixedLM.from_formula`, `mixedlm` |
| **Results** | `fe_params`, `cov_re`, `cov_re_unscaled`, `scale`, `params`, `bse`, `bse_fe`, `bse_re`, `bse_cov_re`, `tvalues`, `pvalues`, `llf`, `aic`, `bic`, `df_resid`, `random_effects`, `random_effects_cov`, `fittedvalues`, `resid`, `conf_int`, `cov_params`, `predict`, `summary`, `converged`, `singular` |
| **Tests** | `t_test`, `wald_test`, `f_test` (fixed effects) |
| **Persistence** | pickling, `save` / `load` |
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
suite is built around — **573 Python tests and 7 Rust tests**.

The primary oracle is **lme4's published fits**, not statsmodels, because
statsmodels is the thing that is wrong on some inputs. `sleepstudy`, `Dyestuff`
and the singular `Dyestuff2` are reproduced to every published digit, under both
REML and ML.

A further 400-case adversarial sweep — tiny groups, extreme imbalance,
predictors spanning six orders of magnitude, near-collinear fixed effects,
heavy outliers — raised **zero exceptions**, and found a better optimum than
statsmodels 131 times against **3 losses**. Those three are near-ties, worse by
2.7e-06, 5.6e-05 and 5.7e-05 in deviance on a criterion whose own scale is in
the hundreds, and they are reported as losses rather than explained away. One
case of 400 could not be certified as a stationary point, and says so. The sweep is committed as `bench/stress_sweep.py`, so the classification
can be checked.

Testing found real defects, and they are listed rather than quietly fixed.
Differential testing against statsmodels found four during development:

- **`fittedvalues` returned the marginal fit.** statsmodels' is the *conditional*
  fit, including the random effects.
- **`theta = 0` is a stationary point for any data whatsoever.** At `Lambda = 0`
  every term of the gradient vanishes identically, so a gradient-based optimiser
  that reaches the bound stops there and reports success — even when the true
  optimum is an ordinary non-zero variance. Fixed by probing away from the bound
  and restarting.
- **That probe ladder was then too coarse.** Its smallest step was 0.05, so a
  true optimum at `theta = 0.028` was still missed — every probe overshot it.
- **Unidentifiable models were fitted silently.** Now detected — though the
  original reasoning for the check was itself wrong, and the correction is
  below.

An external review then found several more, which are fixed and documented:

- **Catastrophic cancellation in the residual sum of squares.** The criterion
  was computed as `y'y - beta'X'y - u'Lambda'Z'y`, a difference of large nearly
  equal quantities. With a response around 1e8 — prices in minor units, epoch
  timestamps, populations — the fit returned a confidently converged wrong
  answer. The response is now offset by its OLS fit before any cross-product is
  formed, which makes results invariant to response translation.
- **Convergence was `optimiser_flag or stationary`.** A loose tolerance let the
  optimiser's own success flag overrule the gradient check, reporting success
  at a projected gradient of 19. Stationarity is now the only test, and it
  drives a retry rather than just annotating the answer.
- **`summary()` printed the wrong quantity.** It labelled `cov_re_unscaled` as
  "Group Var" — 0.935 on `sleepstudy` where both lme4 and statsmodels report
  612.1.
- **`random_effects_cov` returned the population covariance** for every group,
  where the conditional covariance given that group's data was asked for.
- **Malformed input to the compiled core killed the process.** An empty or
  oversized `theta` reached unchecked indexing, and `panic="abort"` turned that
  into a process abort rather than a catchable error.
- **The identifiability rule was wrong in both directions.** `n <= q*m` neither
  implies a divergent likelihood nor catches a confounded single group. The
  check now measures whether the criterion is flat instead of counting.

[What is verified, and every divergence →](docs/CORRECTNESS.md)

## Building from source

Needs a Rust toolchain — **1.83 or newer**, which is what the locked
dependencies require and what CI builds with.

```bash
pip install maturin
python -m maturin build --release --out dist
pip install --force-reinstall --no-deps --no-index --find-links dist mixedlm-rs
```

### Development

```bash
pip install maturin pytest ruff mypy numpy scipy pandas patsy statsmodels
python -m maturin build --release --out dist
pip install --force-reinstall --no-deps --no-index --find-links dist mixedlm-rs

pytest tests/ -q          # the suite
cargo test --lib          # the linear algebra and the optimiser
ruff check .              # rules pinned in pyproject.toml
mypy                      # configured in pyproject.toml
cargo clippy --all-targets -- -D warnings
cargo audit               # Rust advisories
python -m pip_audit --requirement requirements-runtime.txt
```

Benchmarks and the recorded numerical baseline:

```bash
python bench/scaling.py            # scaling against statsmodels
python bench/stages.py             # where the speed comes from
python bench/stress_sweep.py       # 400 adversarial cases
python bench/baseline.py           # re-record bench/baseline.json
```

Every benchmark refuses to print a timing unless the fits agree; the
tolerances and the reasoning behind each are in `bench/tolerances.py`. The
counts quoted in the documentation come from `bench/baseline.json` and are
checked against it by `tests/test_baseline.py`, so a stale table fails the
suite.

## Support

Only the latest release is supported; before 1.0 there are no backports. See
[SECURITY.md](SECURITY.md) for the reporting process and the dependency-audit
policy, and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for what is
linked, what is depended on, and what is deliberately not distributed.

- **Issues and questions:** <https://github.com/Fatin-Ishraq/mixedlm-rs/issues>
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Migration contract:** [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)
- **Known gaps:** [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

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
They are excluded from both distributed artifacts — neither the wheel nor the
source archive contains them, so nothing GPL-2 is redistributed under this
project's MIT licence. They are present in the git repository, and the tests
that use them skip when they are absent. See [data/README.md](data/README.md).
