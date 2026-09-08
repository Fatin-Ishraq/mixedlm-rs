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
| R / rpy2 / pymer4 | 4.6.1 / 3.6.7 / 0.9.2 |

## Scaling in the group count

`y ~ x1 + x2` with a random intercept and slope (`re_formula="~x1"`).

| n | groups | statsmodels | conv | mixedlm-rs | conv | speedup | agreement |
|---:|---:|---:|:---:|---:|:---:|---:|---:|
| 2,000 | 100 | 0.34 s | True | **0.011 s** | True | **30x** | 6.6e-06 |
| 10,000 | 500 | 1.73 s | True | **0.011 s** | True | **153x** | 4.5e-07 |
| 20,000 | 1,000 | 2.63 s | True | **0.012 s** | True | **211x** | 3.4e-07 |
| 40,000 | 5,000 | 11.64 s | True | **0.017 s** | True | **684x** | 2.0e-06 |
| 100,000 | 20,000 | 41.86 s | True | **0.041 s** | True | **1032x** | 3.8e-05 |
| 200,000 | 50,000 | 103.80 s | True | **0.072 s** | True | **1449x** | 1.2e-05 |
| 500,264 | **125,066** | not run | — | **0.180 s** | True | — | — |

*Agreement* is the largest fixed-effect difference expressed in units of its own
standard error.

The last row is the scale from
[statsmodels#9097](https://github.com/statsmodels/statsmodels/issues/9097),
where a user reported `mixedlm` taking **41 minutes** on 125,066 groups against
1–2 seconds for R's `lmer`. It is not run against the reference here because
that is the point.

## Against lme4 and pymer4

`statsmodels` is the package this replaces, but it is not the strongest thing in
the market. That is **`lme4`** in R -- the reference implementation, and the
oracle this package's correctness is checked against -- and **`pymer4`**, which
is the only way a Python user gets genuine `lme4` results today.

All three fitted the same models on byte-identical CSVs. Reproduce with:

```bash
python bench/vs_lme4.py --write            # fixtures + mixedlm-rs timings
Rscript bench/vs_lme4.R bench/fixtures     # lme4, timed inside R
python -u bench/time_pymer4.py             # pymer4, in its own process
python bench/vs_lme4.py --report
```

| fixture | n | groups | mixedlm-rs | lme4 (R) | pymer4 | vs lme4 | vs pymer4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sleepstudy-like | 180 | 18 | **0.008 s** | 0.010 s | 0.39 s | 1x | 47x |
| small | 10,000 | 500 | **0.010 s** | 0.080 s | 1.71 s | 8x | 166x |
| mid | 20,000 | 1,000 | **0.011 s** | 0.160 s | 3.34 s | 14x | 293x |
| mid | 40,000 | 5,000 | **0.017 s** | 0.330 s | 11.67 s | 19x | 683x |
| intercept-only | 40,000 | 5,000 | **0.011 s** | 0.190 s | 6.69 s | 17x | 600x |
| ML (not REML) | 40,000 | 5,000 | **0.017 s** | 0.320 s | 11.29 s | 19x | 653x |
| large | 100,000 | 20,000 | **0.038 s** | 1.120 s | 118.10 s | 30x | 3,127x |
| large | 200,000 | 50,000 | **0.078 s** | 2.510 s | 747.28 s | 32x | **9,554x** |
| huge | 500,264 | **125,066** | **0.169 s** | 7.330 s | *not run* | **43x** | — |

**The log-likelihood agrees with `lme4` to 0.000000 on all nine fixtures.** Same
optimum, same model, between 1x and 43x faster -- and the margin grows with the
group count, which is what you would expect if the win is structural rather than
a constant factor.

The `sleepstudy` row is close to a tie because at 180 observations neither
implementation is doing meaningful work; fixed overhead dominates on both sides.

### The rpy2 tax

`pymer4` *is* `lme4` -- it calls it through `rpy2`. So the gap between the two
columns is pure bridge overhead: marshalling the data frame into R, and the
fitted object back out.

| n | lme4 | pymer4 | overhead |
|---:|---:|---:|---:|
| 10,000 | 0.080 s | 1.71 s | 21x |
| 40,000 | 0.330 s | 11.67 s | 35x |
| 100,000 | 1.120 s | 118.10 s | 105x |
| 200,000 | 2.510 s | 747.28 s | **298x** |

It is not a constant. At 200,000 rows the bridge costs **744 of the 747
seconds** -- the statistics is 2.5 s of it.

That is the substantive competitive point. `pymer4` is not a slower alternative
you might accept in order to avoid reimplementing `lme4`; at any real data size
it is a different order of magnitude, *and* it still needs R, Rtools, rpy2 and a
writable R library present wherever the code runs.

### Caveats

- The 500,264-row row is `mixedlm-rs` and `lme4` only. `pymer4` was stopped
  there deliberately; extrapolating the 298x overhead it would have needed
  roughly 40 minutes.
- `pymer4` 0.9.2 exposes its fit statistics differently than the benchmark
  expected, so its `logLik` came back `NaN` and is not compared. The correctness
  comparison rests on the `lme4` column, which is the same fit.
- Versions: R 4.6.1, lme4 1.1-x from CRAN, rpy2 3.6.7, pymer4 0.9.2.

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
| 10,000 | 500 | 6.02 s | True | 2.621 s | 0.098 s | 0.0130 s | **0.0030 s** | 0.0041 s | 2023x |
| 20,000 | 1,000 | 21.55 s | **False** | 2.130 s | 0.052 s | 0.0114 s | **0.0034 s** | 0.0072 s | 6369x |
| 20,000 | 2,000 | 5.35 s | True | 3.528 s | 0.106 s | 0.0131 s | **0.0041 s** | 0.0093 s | 1301x |
| 40,000 | 5,000 | 13.72 s | True | 7.196 s | 0.294 s | 0.0275 s | **0.0096 s** | 0.0187 s | 1426x |
| 100,000 | 20,000 | 45.07 s | True | 39.211 s | 0.925 s | 0.0801 s | **0.0244 s** | 0.0705 s | 1844x |

**Median stage-to-stage multipliers:**

| step | multiplier |
|---|---:|
| S0 → S1 profiled REML alone | **1.9x** |
| S1 → S2 + batched block Cholesky | **33.4x** |
| S2 → S3 + Rust core | **8.0x** |
| S3 → S4 + analytic gradient | **3.3x** |
| S0 → S4 end to end | **1844x** |

The largest single factor is still **structural, not the language**: profiling
`beta` and `sigma^2` out is worth only 1.9x on its own, while exploiting the
block-diagonal structure is worth 33.4x.

The Rust core's contribution was originally measured at 1.5x. It is now 8.0x —
not because the language changed, but because the first implementation allocated
about ten small `Vec`s per group per objective evaluation, which at 125,000
groups is roughly a million allocations per evaluation. Moving every per-group
intermediate into one preallocated flat buffer, with rayon fold accumulators
instead of per-group temporaries, made a single objective evaluation 8–12x
faster and revealed what the compiled core was actually worth.

S1 and S2 are reproducible by anyone in NumPy — `proto/preml.py` is the
implementation, in about 200 lines. Saying so is what makes the rest of the
table credible.

## What the analytic gradient buys

Objective evaluations for a whole fit:

| n | groups | S3 numeric | S4 analytic | S5 in-Rust optimiser |
|---:|---:|---:|---:|---:|
| 10,000 | 500 | 60 | **11** | 35 |
| 20,000 | 1,000 | 60 | **14** | 36 |
| 20,000 | 2,000 | 44 | **11** | 36 |
| 40,000 | 5,000 | 80 | **16** | 61 |
| 100,000 | 20,000 | 48 | **11** | 39 |

`lme4` and `MixedModels.jl` both optimise `theta` derivative-free (BOBYQA), so
they pay the numeric-gradient evaluation count rather than the analytic one.

## Objective evaluation cost

The measurement that drove the flat-buffer rewrite:

| groups | n | one evaluation, before | after | |
|---:|---:|---:|---:|---|
| 5,000 | 40,000 | 3.31 ms | **0.42 ms** | 7.9x |
| 20,000 | 100,000 | 19.69 ms | **1.63 ms** | 12.1x |
| 125,066 | 500,264 | 89.34 ms | **9.74 ms** | 9.2x |

Objective evaluations are the largest single component of a fit, so this carries
straight through to the end-to-end numbers.

### Memory per evaluation

Two of the six per-group blocks were then removed entirely. Since `A = L L'`,
both `A^-1 = L^-T L^-1` and `B = A^-1 W = L^-T rzx` are recoverable in pass 2
from `l` and `rzx`, so neither needs storing. The evaluation is memory-bound, so
trading traffic for a triangular solve wins; a deviance-only call now never forms
`A^-1` at all.

| | doubles per group | at 125,066 groups |
|---|---:|---:|
| before | `3q² + 2qp + q` = 26 | 26.0 MB |
| after | `2q² + qp + q` = 16 | **16.0 MB** |

Deviance-only evaluation at 125,066 groups: 8.41 ms → **7.07 ms**.

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
