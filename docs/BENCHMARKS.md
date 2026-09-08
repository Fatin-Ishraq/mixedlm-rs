# Benchmarks

Every number here was measured on the machine described below. Agreement is
checked **before** timing, so a fast wrong answer cannot appear in a table.
Convergence status is reported for every row, because on several fixtures the
reference does not converge and a speedup ratio would be comparing against a fit
that did not happen.

Reproduce with `python bench/scaling.py` and `python bench/stages.py`.

## Machine

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | AMD Ryzen 5 5600G, 6 cores / 12 threads, AVX2, no GPU |
| Python | 3.14.3 |
| numpy / scipy / statsmodels | 2.5.2 / 1.18.1 / 0.15.0 |

## Scaling in the group count

`y ~ x1 + x2` with a random intercept and slope (`re_formula="~x1"`).

| n | groups | statsmodels | conv | mixedlm-rs | conv | speedup | agreement |
|---:|---:|---:|:---:|---:|:---:|---:|---:|
| 2,000 | 100 | 0.35 s | True | **0.025 s** | True | **14x** | 6.6e-06 |
| 10,000 | 500 | 1.72 s | True | **0.033 s** | True | **52x** | 4.4e-07 |
| 20,000 | 1,000 | 3.32 s | True | **0.046 s** | True | **72x** | 3.3e-07 |
| 40,000 | 5,000 | 11.80 s | True | **0.107 s** | True | **110x** | 2.0e-06 |
| 100,000 | 20,000 | 41.98 s | True | **0.638 s** | True | **66x** | 3.8e-05 |
| 200,000 | 50,000 | 106.03 s | True | **2.019 s** | True | **53x** | 1.2e-05 |
| 500,264 | **125,066** | not run | — | **5.840 s** | True | — | — |

*Agreement* is the largest fixed-effect difference expressed in units of its own
standard error.

The last row is the scale from
[statsmodels#9097](https://github.com/statsmodels/statsmodels/issues/9097),
where a user reported `mixedlm` taking **41 minutes** on 125,066 groups against
1–2 seconds for R's `lmer`. It is not run against the reference here because
that is the point.

## Where the win actually comes from

Reporting a single speedup number would misattribute it. Each stage below was
measured on the same fixtures:

| stage | what it is | isolates |
|---|---|---|
| S0 | `statsmodels.MixedLM` | baseline |
| S1 | profiled REML, pure NumPy, per-group Python loop | the algorithmic win |
| S2 | S1 + batched block-diagonal linear algebra | the structural win |
| S3 | Rust core, numeric gradient | the language win |
| S4 | Rust core, analytic gradient | the beyond-lme4 win |
| S5 | Rust core, in-Rust optimiser | owning the whole loop |

| n | groups | S0 | conv | S1 | S2 | S3 | S4 | S5 | S4/S0 |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 500 | 5.83 s | True | 2.655 s | 0.097 s | 0.0646 s | **0.0060 s** | 0.0143 s | 972x |
| 20,000 | 1,000 | 20.79 s | **False** | 2.110 s | 0.052 s | 0.0552 s | **0.0086 s** | 0.0229 s | 2426x |
| 20,000 | 2,000 | 5.21 s | True | 3.583 s | 0.106 s | 0.0726 s | **0.0135 s** | 0.0641 s | 387x |
| 40,000 | 5,000 | 14.97 s | True | 7.273 s | 0.293 s | 0.1402 s | **0.0219 s** | 0.0753 s | 684x |
| 100,000 | 20,000 | 44.34 s | True | 36.744 s | 0.895 s | 0.6774 s | **0.1418 s** | 0.4296 s | 313x |

**Median stage-to-stage multipliers:**

| step | multiplier |
|---|---:|
| S0 → S1 profiled REML alone | **2.1x** |
| S1 → S2 + batched block Cholesky | **33.6x** |
| S2 → S3 + Rust core | **1.5x** |
| S3 → S4 + analytic gradient | **6.4x** |
| S0 → S4 end to end | **684x** |

The honest headline is that **most of the win is structural, not from Rust**.
Profiling `beta` and `sigma^2` out is worth only 2.1x on its own. Exploiting the
block-diagonal structure so the whole system is a handful of batched operations
is worth 33.6x. The Rust port then adds just 1.5x over batched NumPy, and the
analytic gradient another 6.4x.

S1 and S2 are reproducible by anyone in NumPy — `proto/preml.py` is the
implementation, in about 200 lines. Saying so is what makes the rest of the
table credible.

## What the analytic gradient buys

Objective evaluations for a whole fit:

| n | groups | S3 numeric | S4 analytic | S5 in-Rust optimiser |
|---:|---:|---:|---:|---:|
| 10,000 | 500 | 148 | **11** | 35 |
| 20,000 | 1,000 | 88 | **11** | 36 |
| 20,000 | 2,000 | 80 | **11** | 64 |
| 40,000 | 5,000 | 80 | **10** | 38 |
| 100,000 | 20,000 | 72 | **12** | 40 |

`lme4` and `MixedModels.jl` both optimise `theta` derivative-free (BOBYQA), so
they pay the numeric-gradient evaluation count rather than the analytic one.

## An honest negative result

The hand-rolled projected L-BFGS in Rust (S5) needs roughly **three times** the
objective evaluations of scipy's L-BFGS-B (S4) — 35–64 against 10–12 — because
scipy's line search is better. S4 is therefore the default path, and the Rust
optimiser stays available as `method="rust"` for callers who want no scipy in
the loop. Competing with a mature Fortran line search was not where the value
was.

## Convergence quality

Across 120 randomised fixtures (see CORRECTNESS.md):

| outcome | count |
|---|---:|
| statsmodels did not converge | **26** |
| mixedlm-rs found a strictly better optimum | **19** |
| same optimum | 75 |
| mixedlm-rs found a worse optimum | **0** |
