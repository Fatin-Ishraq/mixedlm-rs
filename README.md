<div align="center">

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/logo.svg" alt="mixedlm-rs logo" width="96" height="96">

# mixedlm-rs

**Fast linear mixed-effects models for Python, powered by Rust.**

[![PyPI](https://img.shields.io/pypi/v/mixedlm-rs.svg)](https://pypi.org/project/mixedlm-rs/)
[![CI](https://github.com/Fatin-Ishraq/mixedlm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/mixedlm-rs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)](https://pypi.org/project/mixedlm-rs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/LICENSE)

</div>

`mixedlm-rs` fits Gaussian linear mixed-effects models using lme4's profiled
ML/REML formulation and a compiled Rust core. It provides a statsmodels-style
formula and array API, with no R installation required.

Use it for repeated measurements within subjects, patients within clinics, or
other data with **one grouping factor**, including correlated random intercepts
and slopes. The implementation is designed to make models with many groups
fast while checking convergence explicitly.

**This is beta software.** It supports a subset of `statsmodels.MixedLM`;
crossed effects, multiple levels of nesting and GLMMs are outside its scope.
See [compatibility](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/COMPATIBILITY.md)
before migrating an existing analysis.

## Install

```bash
python -m pip install mixedlm-rs
```

Python 3.10 or later is required. NumPy, SciPy, pandas and Patsy are installed
as dependencies. Neither statsmodels nor R is needed for normal use.

Binary wheels are available on [PyPI](https://pypi.org/project/mixedlm-rs/#files):

| Platform | Architectures |
|---|---|
| Windows | x86-64 |
| macOS | Intel x86-64, Apple Silicon ARM64 |
| Linux, glibc 2.17+ | x86-64, ARM64 |

The wheels use Python's stable ABI (`cp310-abi3`). On a matching platform,
installation needs no Rust compiler. Building from source requires Rust 1.83+
and a compatible native build toolchain; pip installs the Maturin build backend.
There are no published wheels for 32-bit systems or musl-based Linux.

**Currently verified:** the project's CI matrix covers Python 3.10–3.14 on
Windows, macOS and Linux, with separate checks for the declared dependency
floors and minimum Rust version. Release checks install each platform's wheel
outside the source tree and test a wheel rebuilt from the source archive.
See the [CI runs](https://github.com/Fatin-Ishraq/mixedlm-rs/actions/workflows/ci.yml)
for the status of a particular commit. Python versions beyond 3.14 are not yet
part of that test matrix.

## Quick start

This example simulates 18 subjects measured on 10 days. Each subject has their
own baseline and rate of change. It runs after installation, with **no data file to fetch**.

<!-- readme-example -->
```python
import numpy as np
import pandas as pd
import mixedlm_rs as mlm

rng = np.random.default_rng(0)
n_subjects, n_days = 18, 10
subject = np.repeat(np.arange(n_subjects), n_days)
day = np.tile(np.arange(n_days), n_subjects)

intercept = rng.normal(250, 25, n_subjects)[subject]
slope = rng.normal(10, 6, n_subjects)[subject]
reaction = intercept + slope * day + rng.normal(0, 25, subject.size)
data = pd.DataFrame({"reaction": reaction, "day": day, "subject": subject})

model = mlm.mixedlm(
    "reaction ~ day",
    data,
    groups="subject",
    re_formula="~day",
)
result = model.fit()  # REML by default

print(result.summary())
print(f"fixed effects: {result.fe_params}")
print(f"converged: {result.converged}")
print(f"singular: {result.singular}")
```
<!-- /readme-example -->

`reaction ~ day` specifies the fixed effects. `groups="subject"` identifies
independent clusters; `re_formula="~day"` gives each subject a random intercept
and slope, with their covariance estimated. Omit `re_formula` for a random
intercept only. Group sizes do not have to be equal.

To use maximum likelihood instead, call `.fit(reml=False)` on a new model.
Use ML when comparing likelihoods of models with different fixed-effect
specifications fitted to the same observations; their REML criteria are not
directly comparable.

### Inspect estimates and predict

Continuing the example:

```python
print(result.fe_params_labelled)  # Fixed effects as a named pandas Series
print(result.bse_fe)             # Fixed-effect standard errors
print(result.cov_re)             # Random-effect covariance matrix
print(result.scale)              # Residual variance
print(result.diagnostics)        # Optimizer and numerical diagnostics

new_data = pd.DataFrame({"day": [0, 5, 10]})
print(result.predict(new_data))
```

`predict()` returns the **population-level, fixed-effects prediction**. It does
not add fitted subject effects, so this example needs no subject column.
`result.fittedvalues` includes the fitted random effects for the training
observations. `result.random_effects` contains the estimated effects by group.

The main numerical attributes, including `fe_params` and `cov_re`, are NumPy
arrays even for formula fits. Use `fe_params_labelled` or `params_labelled`
when you need names.

### Use arrays directly

The equivalent model can be constructed without a formula:

```python
X = np.column_stack([np.ones(len(data)), data["day"].to_numpy()])
array_result = mlm.MixedLM(
    endog=data["reaction"].to_numpy(),
    exog=X,
    groups=data["subject"].to_numpy(),
    exog_re=X,
).fit()
```

Include an intercept column explicitly in array inputs when the model needs
one. `exog` defines fixed effects; `exog_re` defines random effects within each
group.

### Save and load a fit

```python
result.save("fit.pkl")
restored = mlm.MixedLMResults.load("fit.pkl")
print(restored.predict(new_data))
```

The default save retains the training data needed to rebuild supported formula
designs. Only load files you trust: this uses Python pickle. Custom formula
functions and reduced-data saves have additional
[persistence limitations](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/LIMITATIONS.md).

## Moving from statsmodels

For a supported model, start by changing the formula API import:

```diff
- import statsmodels.formula.api as smf
+ import mixedlm_rs as smf
```

The `mixedlm(...)`, `MixedLM(...)` and `MixedLM.from_formula(...)` entry points
follow familiar statsmodels conventions. Existing code still needs review:
some return types differ, optimizer arguments do not select the same
algorithms, and several methods are unimplemented. The
[compatibility guide](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/COMPATIBILITY.md)
lists the supported contract, warnings and exceptions.

## Convergence and inference

Inspect both `result.converged` and `result.singular`:

| Attribute | Meaning |
|---|---|
| `converged` | The fit passed the implementation's numerical convergence checks. |
| `singular` | The estimated random-effect covariance is at or near a singular boundary. |

A singular fit can be a valid converged optimum, for example when a random
intercept variance is zero. Conversely, returning estimates does not establish
convergence. This package can also return `converged=False` with a warning.

The optimizer checks local stationarity and probes away from boundary
solutions. These checks do **not** prove that it found a global optimum or that
the model is identifiable. Examine warnings and `result.diagnostics` before
using an uncertain fit.

Reported fixed-effect tests use normal/Wald inference. There are no
Satterthwaite or Kenward–Roger small-sample corrections. Fixed-effect covariance
is conditional on the fitted variance parameters; joint covariance across
fixed and variance parameters is not available. Wald intervals for variance
parameters are especially unreliable near zero.

## Numerical validation

The primary reference is **lme4**, with checks against published fits and
recorded R results for `sleepstudy`, `Dyestuff` and `Dyestuff2`, under ML and
REML. These cover correlated random slopes and a zero-variance boundary fit.
The reference datasets are GPL-2 test fixtures kept in the Git repository;
they are **not shipped** in the wheels or source archive. See
[dataset provenance](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/data/README.md).

Agreement is assessed at the precision and tolerances of the recorded
references. Additional tests compare against statsmodels, check derivatives
with finite differences, and probe the objective independently of the analytic
gradient. A convergence warning from another package alone does not establish
that its estimates are wrong.

The recorded comparison on **120 randomised fixtures** reports:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/outcomes-dark.svg">
  <img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/outcomes-light.svg"
       alt="The same 120 fixtures as one bar: 42 better optimum, 52 same optimum, 26 statsmodels did not converge, 0 worse."
       width="100%">
</picture>

| Outcome | Count |
|---|---:|
| statsmodels did not converge | **26** |
| mixedlm-rs found a strictly **better** optimum | **42** |
| same optimum | 52 |
| mixedlm-rs found a **worse** optimum | **0** |

“Better” and “worse” refer to the fitted likelihood criterion on the same
model and data, using the experiment's absolute deviance tolerance. They do
not mean better prediction or establish which model is scientifically appropriate.

A separate **400-case adversarial sweep** recorded 99 statsmodels
nonconvergences, 130 better optima for mixedlm-rs, 168 ties and **3 worse
optima**. Those losses have absolute deviance gaps of approximately
`2.74e-6` to `6.71e-5`. Every case passed the sweep's independent stationarity
check. These are finite test suites, not estimates of failure rates in general use.

Both experiments are recorded in
[`bench/baseline.json`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/baseline.json)
at commit `83c01d5`. The
[correctness report](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/CORRECTNESS.md)
provides the reference fits, thresholds, generators and known difficult cases.

## Performance

The recorded benchmark below uses synthetic data with a random intercept and
slope, fitted through each package's formula API on the same machine. Timings
include model construction and fitting; they exclude data generation and
printing a summary.

| Observations | Groups | statsmodels | mixedlm-rs | Speedup |
|---:|---:|---:|---:|---:|
| 2,000 | 100 | 0.353 s | 0.0070 s | 50x |
| 10,000 | 500 | 1.813 s | 0.0085 s | 215x |
| 20,000 | 1,000 | 2.666 s | 0.0107 s | 250x |
| 40,000 | 5,000 | 11.472 s | 0.0141 s | 814x |
| 100,000 | 20,000 | 43.358 s | 0.0244 s | 1777x |
| 200,000 | 50,000 | 108.354 s | 0.0456 s | 2375x |
| 500,264 | 125,066 | Not measured | 0.124 s | — |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/scaling-dark.svg">
  <img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/scaling-light.svg"
       alt="Fit time against grouping factor levels, log-log. statsmodels rises from 0.35 to 108 seconds; mixedlm-rs stays between 0.007 and 0.046 seconds."
       width="100%">
</picture>

**Measurement conditions:** Windows 11, AMD Ryzen 5 5600G, 12 logical CPUs,
Python 3.14.3 and statsmodels 0.15.0. mixedlm-rs reports the minimum of three
runs. statsmodels also uses three runs through 40,000 observations, but only
**one run** for the 100,000- and 200,000-observation cases. Thread counts were
not pinned: Rayon used the available logical CPUs, and OMP/OpenBLAS thread
settings were unset. These are wall-clock comparisons with those defaults,
not comparisons at a fixed CPU budget. Speedups use unrounded timings.

Both fits reported convergence in every timed comparison, and their estimates
and likelihoods passed the benchmark's agreement checks. The workload favors
many small groups; speedups depend on model shape, data, dependencies and
hardware, and should not be assumed for every analysis.

The measurements were recorded on 2026-09-19 at commit `d01ab61` (0.2.0).
[`bench/performance.json`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/performance.json)
contains all repetitions, dependency versions, thread settings and the
extension hash. Reproduce the comparison with
[`bench/performance.py`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/performance.py).

The largest case uses a group count motivated by
[statsmodels issue #9097](https://github.com/statsmodels/statsmodels/issues/9097).
Its reported 41-minute timing is not a measurement made here. Our synthetic
case is not a reproduction of their dataset and should not be read as a measured
speedup over that report.

### Against lme4

[`bench/three_way.py`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/three_way.py) fits byte-identical CSV
files with all three packages in one session on the same machine, with lme4
2.0.6 run through R 4.6.1 and timed inside R. Each Python fit is checked
against lme4's before its time is reported.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/three-way-scaling-dark.svg">
  <img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/three-way-scaling-light.svg"
       alt="Fit time against groups, log-log, for three packages. At 125,066 groups mixedlm-rs takes 110 ms and lme4 7.03 s; statsmodels takes 108 s at 50,000 groups."
       width="100%">
</picture>

| Observations | Groups | lme4 | statsmodels | mixedlm-rs | vs lme4 |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 100 | 0.030 s | 0.351 s | 0.0075 s | 4x |
| 10,000 | 500 | 0.075 s | 1.739 s | 0.0083 s | 9x |
| 20,000 | 1,000 | 0.163 s | 2.636 s | 0.0130 s | 13x |
| 40,000 | 5,000 | 0.322 s | 11.041 s | 0.0160 s | 20x |
| 100,000 | 20,000 | 1.135 s | 43.134 s | 0.0273 s | 42x |
| 200,000 | 50,000 | 2.409 s | 107.936 s | 0.0468 s | 51x |
| 500,264 | 125,066 | 7.033 s | Not measured | 0.110 s | 64x |

Accuracy is measured on the same 120 randomised fixtures as above, this
time against lme4's optimum:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/lme4-agreement-dark.svg">
  <img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/lme4-agreement-light.svg"
       alt="Outcomes on 120 fixtures relative to lme4. mixedlm-rs: 113 match, 7 higher likelihood. statsmodels: 74 match, 3 higher, 26 lower and flagged not converged, 17 lower but reported converged."
       width="100%">
</picture>

mixedlm-rs matched lme4's optimum on 113 fixtures and reached a higher
likelihood on 7. It did not finish below lme4 on any fixture. statsmodels finished below lme4 on 43, and on 17 of
those it still reported convergence. lme4 is a reference here, not ground
truth: its optimiser can stop short too. Both experiments were recorded on
2026-09-19 at commit `7677222` (0.2.0) in
[`bench/three_way.json`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/three_way.json), under the same
measurement conditions and caveats as above.

The earlier lme4 and pymer4 timings, including pymer4's, are kept in the
[benchmark report](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/BENCHMARKS.md) as historical records. The pymer4
path also performs lmerTest inference and result extraction, so its elapsed
time covers different work from a bare model fit.

### Against the previous release

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/release-speedup-dark.svg">
  <img src="https://raw.githubusercontent.com/Fatin-Ishraq/mixedlm-rs/main/docs/assets/release-speedup-light.svg"
       alt="Full-fit speedup of 0.2.0 over 0.1.1 per case, at 1 and 12 threads: 1.4x to 2.2x on continuous random slopes, 4.0x to 6.4x on one thread where groups share a design."
       width="100%">
</picture>

0.2.0 fits 1.4x to 6.4x faster than 0.1.1 on the same machine. The largest
gains are on designs where many groups share the same random-effects design,
such as a random intercept or a slope over a shared visit schedule. There the work
scales with the number of distinct group designs, not the number of groups.
Recorded with [`bench/phases.py`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/phases.py) in
[`bench/phases.json`](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/bench/phases.json), which alternates the two
builds and keeps every sample.

## How it works

The Rust core evaluates a profiled ML/REML criterion using penalised least
squares. With one grouping factor, the random-effect system separates into
small per-group blocks. The implementation precomputes group statistics,
factorises those blocks in Rust, and uses Rayon for parallel work.

An analytic gradient reuses the factorisation work. The default optimizer is
SciPy's L-BFGS-B, calling the Rust objective and gradient. In the recorded
implementation-stage comparison, replacing this project's finite differences
with its analytic gradient reduced objective evaluations from **44–64** to
**11–13**. That is a comparison between stages of this implementation, not a
claim that analytic gradients are new or that statsmodels lacks them.

See the [design and gradient derivation](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/DESIGN.md)
for the equations, implementation details and validation.

## Scope and limitations

| Area | Current boundary |
|---|---|
| Grouping | One grouping factor. No crossed subject/item effects or separate effects at multiple nesting levels. |
| Model family | Gaussian linear mixed models only; no binomial or Poisson GLMMs. |
| Covariance structures | No variance-component formulas, `free` masks or covariance penalties. |
| Additional fitting methods | No `fit_regularized`, `profile_re`, `bootstrap` or `get_distribution`. |
| Fixed-effect design | Rank-deficient or numerically rank-deficient designs are rejected; redundant columns are not dropped automatically. |
| Inference | Normal/Wald inference, with the limitations described above. |
| Optimization | Local convergence checks; no guarantee of a global optimum. `method="rust"` is experimental. |

These restrictions matter for designs such as crossed participant/item
experiments or students within classrooms within schools. Combining IDs into
one grouping variable does not reproduce separate random effects at each level.
Read the full [limitations](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/LIMITATIONS.md)
and [API compatibility guide](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/docs/COMPATIBILITY.md)
for argument handling, result types, numerical edge cases and persistence.

## Development

Building a checkout requires Rust 1.83+ and Python 3.10+. Use a virtual
environment, then install the package and test tools:

```bash
git clone https://github.com/Fatin-Ishraq/mixedlm-rs.git
cd mixedlm-rs
python -m pip install ".[test,lint]"
python -m pytest tests/ -q
cargo test --lib --locked
python -m ruff check .
```

The Git checkout includes the numerical reference fixtures and benchmark
scripts omitted from published distributions. CI also runs type checking,
Rust linting, dependency-floor checks and artifact validation.

For bug reports, include a minimal example, package and dependency versions,
platform, warnings and `result.diagnostics` when a fit is available. Use
[GitHub issues](https://github.com/Fatin-Ishraq/mixedlm-rs/issues).
See the [changelog](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/CHANGELOG.md)
for release history.

## License and acknowledgements

mixedlm-rs is [MIT licensed](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/LICENSE).
Bundled dependencies retain their own licenses; see
[third-party notices](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/THIRD-PARTY-NOTICES.md)
and [license texts](https://github.com/Fatin-Ishraq/mixedlm-rs/blob/main/THIRD-PARTY-LICENSES.md).

The statistical formulation follows lme4 and the work of Bates, Mächler,
Bolker and Walker, *Fitting Linear Mixed-Effects Models Using lme4* (2015).
The Python API follows statsmodels. This is an independent implementation;
neither upstream project maintains or endorses it.
