# Benchmarks

Every number here was measured on the machine described below. Agreement is
checked **before** timing, so a fast wrong answer cannot appear in a table.
Convergence status is reported for every row, because on several fixtures the
reference does not converge and a speedup ratio would be comparing against a fit
that did not happen.

Reproduce with `python bench/scaling.py` and `python bench/stages.py`. Both
refuse to print a timing unless the fits agree; the tolerances, and the reason
each is what it is, are in `bench/tolerances.py`.

**Counts can move with the environment.** The optimiser on the default path is
SciPy's, so a different SciPy build can change which fixtures land on which
side of a tie. An independent re-run on Python 3.11 with SciPy 1.17.1 produced
130 wins and 168 ties on the 400-case sweep where the recorded baseline has 131
and 167 — the same picture, one case moved. That is why `bench/baseline.json`
records the environment alongside the counts rather than treating it as a
footnote, and why the reproduction instructions name it.

## Where these numbers come from

Every figure here is recorded, not remembered. `python bench/performance.py`
re-measures this package against `statsmodels` on the candidate build and
writes **`bench/performance.json`**, which carries the commit, the extension's
sha256, the machine, the dependency versions, the repetition count, the thread
settings, and every individual repetition rather than only the reported
minimum. `tests/test_performance_evidence.py` holds this document to that file.

Two kinds of number appear below and they are **not** interchangeable:

| | how to read it |
|---|---|
| **measured** | `mixedlm-rs` against `statsmodels`, re-run on the candidate build. Agreement is checked *before* any timing is reported -- a fast wrong answer is not a benchmark result -- against the thresholds in `bench/tolerances.py`. |
| **historical** | the **`lme4`** and **`pymer4`** columns. These need R, `rpy2` and a writable R library, which the recording machine for the current candidate does not have. They are carried forward verbatim from the run named in `performance.json`, tagged with the environment that produced them, and were **not** re-measured on this build. |

The reported figure is the **minimum** of the repetitions, not the mean: a
timing distribution is bounded below by the true cost and has an unbounded
right tail made of scheduler noise. Every repetition is in the JSON so the
spread is inspectable rather than something you have to take on trust.

## Machine

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| CPU | AMD Ryzen 5 5600G, 6 cores / 12 threads, AVX2, no GPU |
| Python | 3.14.3 |
| numpy / scipy / statsmodels | 2.5.2 / 1.18.1 / 0.15.0 |
| R / lme4 | 4.6.1 / 2.0.6 |
| rpy2 / pymer4 | 3.6.7 / 0.9.2 |
| threads | rayon default (12); no thread pinning |

## Scaling in the group count

`y ~ x1 + x2` with a random intercept and slope (`re_formula="~x1"`).

| n | groups | statsmodels | conv | mixedlm-rs | conv | speedup | fe (SEs) | re (rel) | dlogLik |
|---:|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|
| 2,000 | 100 | 0.39 s | True | **0.012 s** | True | **33x** | 6.6e-06 | 7.6e-05 | +1.2e-06 |
| 10,000 | 500 | 2.31 s | True | **0.011 s** | True | **218x** | 4.5e-07 | 1.4e-05 | +2.8e-07 |
| 20,000 | 1,000 | 4.00 s | True | **0.018 s** | True | **227x** | 3.4e-07 | 2.6e-05 | +1.1e-06 |
| 40,000 | 5,000 | 16.59 s | True | **0.019 s** | True | **870x** | 2.0e-06 | 1.8e-05 | +2.3e-06 |
| 100,000 | 20,000 | 61.00 s | True | **0.042 s** | True | **1454x** | 3.8e-05 | 4.3e-05 | +5.1e-05 |
| 200,000 | 50,000 | 160.48 s | True | **0.087 s** | True | **1836x** | 1.2e-05 | 3.7e-05 | +6.9e-05 |
| 500,264 | **125,066** | not run | — | **0.197 s** | True | — | — | — | — |

- *fe (SEs)* is the largest fixed-effect difference in units of its own
  standard error.
- *re (rel)* is the largest `cov_re` difference relative to its own scale.
  Comparing only the fixed effects, as this table used to, lets two fits agree
  on the mean model while disagreeing about the thing a mixed model is for.
- *dlogLik* is ours minus theirs. It is non-negative on every row.

**Every one of these is enforced, not printed.** The script aborts and refuses to
report timings if a row exceeds 0.05 SEs on the fixed effects, 2% on `cov_re`,
or falls below the reference's criterion. A benchmark that prints an agreement
column without checking it can publish a fast wrong answer, and this one used
to.

The last row is the scale from
[statsmodels#9097](https://github.com/statsmodels/statsmodels/issues/9097),
where a user reported `mixedlm` taking **41 minutes** on 125,066 groups against
1–2 seconds for R's `lmer`. It is synthetic data at that group count, not their
categorical-design dataset, and the 41 minutes is their report rather than a
measurement made here — so this row shows that the size is not the obstacle, and
nothing more than that.

## Against lme4 and pymer4

`statsmodels` is the package this replaces, but it is not the strongest thing in
the market. That is **`lme4`** in R -- the reference implementation, and the
oracle this package's correctness is checked against -- and **`pymer4`**, which
is the only way a Python user gets genuine `lme4` results today.

All three fitted the same models on byte-identical CSVs.

> **Historical, not re-measured.** The `lme4` and `pymer4` columns in this
> section come from the run recorded in `bench/performance.json` under
> `historical`, on a machine with the R toolchain installed. The candidate
> build was not re-timed against them, because R, `rpy2` and a writable R
> library are not present on the machine that recorded it -- and a machine
> without them can produce no number rather than a wrong one. The
> `mixedlm-rs` column in the same table is from that same historical run, so
> the ratios are internally consistent; they are not a claim about the current
> build's timings. To refresh them, run the commands below on a machine with R.

Reproduce with:

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

### What calling lme4 from Python costs

`pymer4` *is* `lme4` -- it calls it through `rpy2`. The gap between the two
columns is what a Python caller pays on top of the fit itself.

| n | lme4 | pymer4 | ratio |
|---:|---:|---:|---:|
| 10,000 | 0.080 s | 1.71 s | 21x |
| 40,000 | 0.330 s | 11.67 s | 35x |
| 100,000 | 1.120 s | 118.10 s | 105x |
| 200,000 | 2.510 s | 747.28 s | **298x** |

It is not a constant. At 200,000 rows, 2.51 s of the 747.28 s is the `lmer` fit
and the other 744.77 s is everything `pymer4` does around it.

**That 744.77 s is not a measurement of marshalling.** An earlier revision of
this file called it "pure bridge overhead", and that was wrong -- it attributed
to serialisation a number that was never decomposed. `pymer4.models.lmer.fit`
also routes through `lmerTest` for Satterthwaite degrees of freedom, and pulls
fixed effects, random effects and fit statistics back as R objects. The 744.77 s
covers marshalling *and* that additional inference and extraction, and nothing
in `bench/time_pymer4.py` separates them; doing so would need a profiler
decomposition that has not been run. The benchmark script has always said this,
and the claim here now matches it.

The competitive point does not depend on the split. `pymer4` is not a slower
alternative you might accept in order to avoid reimplementing `lme4`; at any
real data size it is a different order of magnitude end to end, *and* it still
needs R, Rtools, rpy2 and a writable R library present wherever the code runs.

### Caveats

- The 500,264-row row is `mixedlm-rs` and `lme4` only. `pymer4` was stopped
  there deliberately; extrapolating the 298x overhead it would have needed
  roughly 40 minutes.
- `pymer4` 0.9.2 exposes its fit statistics differently than the benchmark
  expected, so its `logLik` came back `NaN` and is not compared. The correctness
  comparison rests on the `lme4` column, which is the same fit.
- Versions: R 4.6.1, lme4 **2.0.6**, rpy2 3.6.7, pymer4 0.9.2. An earlier
  revision of this file recorded "lme4 1.1-x", which was a guess rather
  than a reading of the installed package.

## Where the win actually comes from

Reporting a single speedup number would misattribute it. Each stage below is
measured on the same fixtures, and — after the corrections described at the end
of this section — on the same amount of work.

| stage | what it is | isolates |
|---|---|---|
| S0 | `statsmodels.MixedLM`, from a DataFrame | baseline |
| S1 | profiled REML, pure NumPy, per-group Python loop | the algorithmic step |
| S2 | S1 + batched block-diagonal linear algebra | the structural win |
| S3 | Rust core, numeric gradient | the language win |
| S4 | Rust core, analytic gradient | what the gradient buys |
| S5 | Rust core, in-Rust optimiser | owning the whole loop |
| S6 | **the public API, from a DataFrame** | the like-for-like end-to-end row |

**What is comparable to what.** S1–S5 all start from the same
`(y, X, Z, codes)` and each builds what it needs from there, including the
cross-products. S0 and S6 both start from a DataFrame, parse a formula, fit, and
compute inference. **S0 is not comparable to S4** — it does work S4 never
does — so the end-to-end ratio is `S0/S6`, not `S0/S4`.

| n | groups | S0 | conv | S1 | S2 | S3 | S4 | S5 | **S6** | S0/S6 |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 500 | 8.69 s | True | 2.773 s | 0.099 s | 0.0182 s | **0.0045 s** | 0.0062 s | 0.0236 s | 368x |
| 20,000 | 1,000 | 30.58 s | **False** | 3.263 s | 0.076 s | 0.0120 s | **0.0045 s** | 0.0071 s | 0.0476 s | 643x |
| 20,000 | 2,000 | 5.37 s | True | 3.646 s | 0.108 s | 0.0144 s | **0.0049 s** | 0.0111 s | 0.0563 s | 95x |
| 40,000 | 5,000 | 14.36 s | True | 7.463 s | 0.521 s | 0.0241 s | **0.0104 s** | 0.0335 s | 0.0853 s | 168x |
| 100,000 | 20,000 | 48.17 s | True | 63.304 s | 1.433 s | 0.0656 s | **0.0250 s** | 0.0766 s | 0.2449 s | 197x |

Best of three for **every** stage, including the slow ones.

**Median stage-to-stage multipliers**, over the four fixtures where the baseline
converged — the fifth is excluded, because a speedup against a fit that did not
happen is not a speedup:

| step | multiplier |
|---|---:|
| S0 → S1 profiled REML — **upper bound only** | 1.7x |
| S1 → S2 + batched block Cholesky | **30.8x** |
| S2 → S3 + Rust core | 14.6x |
| S3 → S4 + analytic gradient | 2.8x |
| **S0 → S6 end to end, like for like** | **182.5x** |

The largest single factor is **structural, not the language**: exploiting the
block-diagonal structure is worth 30.8x, and it is reproducible by anyone in
NumPy — `proto/preml.py` is that implementation, in about 200 lines. Saying so
is what makes the rest of the table credible.

`S0 → S1` is labelled an upper bound because S0 also parses a formula and
computes inference that S1 skips entirely, so some non-algorithmic work is
attributed to profiling. It cannot be read as "profiling alone is worth 1.7x".

### What was wrong with this table before

Four things, all of which inflated it:

1. **S3–S5 were handed a pre-built `LmmCore`** while S1 and S2 built their own
   cross-products inside the timed region. The "language win" was therefore
   partly credited with work the compiled stages were simply not charged for.
   Every stage now builds what it needs.
2. **S0 and S1 were timed once**, the rest best-of-three.
3. **`S0/S4` was reported as "end to end", at 1844x.** It is not an end-to-end
   comparison, and that number should not have been published. The honest
   figure is 182.5x.
4. **A non-converged S0 row contributed a 6369x ratio** to the medians, despite
   the surrounding prose saying such comparisons are inappropriate.

Agreement is now *enforced* rather than printed: the script aborts instead of
reporting timings if any stage's criterion is worse than S4's, or if the
coefficients disagree at the same optimum. Where we reach a **better** optimum
than statsmodels the coefficients legitimately differ, and that is reported
rather than treated as a failure — on the 10,000-row fixture we reach a
deviance 152.5 lower.

## What the analytic gradient buys

Objective evaluations for a whole fit:

| n | groups | S3 numeric | S4 analytic | S5 in-Rust optimiser |
|---:|---:|---:|---:|---:|
| 10,000 | 500 | 64 | **11** | 35 |
| 20,000 | 1,000 | 44 | **11** | 36 |
| 20,000 | 2,000 | 56 | **11** | 36 |
| 40,000 | 5,000 | 60 | **13** | 59 |
| 100,000 | 20,000 | 56 | **11** | 39 |

**What this does and does not measure.** S3 is *this package's own*
finite-difference L-BFGS-B, so the 44-64 column is the cost of not having a
gradient inside this optimiser. It is **not** a measurement of BOBYQA, which is
a different algorithm with a different evaluation profile, and no claim is made
here about lme4's or MixedModels.jl's evaluation counts -- neither was measured.

What is true is the qualitative point: `lme4` and `MixedModels.jl` both optimise
`theta` derivative-free, so neither uses a gradient of the profiled criterion,
while this package does. The gradient itself is not novel -- Bates et al. derive
the profiled ML version in the lme4 paper (eq. 46-48), and `MixedModels.jl`
documents derivative support. What is here is a REML gradient specialised to the
block structure and computed in the passes that already produce the criterion.

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
objective evaluations of scipy's L-BFGS-B (S4) — 35–59 against 11–13 — because
scipy's line search is better. S4 is therefore the default path, and the Rust
optimiser stays available as `method="rust"` for callers who want no scipy in
the loop. Competing with a mature Fortran line search was not where the value
was.

It is worse in a second way, found in review and worth stating: a weaker line
search does not merely take longer to reach the same place, it settles in worse
basins. With a single starting value it lands on a strictly worse *stationary*
point on 2 of 20 fuzz seeds — one of them correctly reported as converged, since
a local optimum is exactly what it is. Certification does not help, because the
certificate is local by construction. The rust path therefore defaults to five
starts where the scipy path takes one, which recovers the scipy answer on both
seeds. Neither number is a guess; both are measured.

## Convergence quality

Across 120 randomised fixtures (see CORRECTNESS.md):

| outcome | count |
|---|---:|
| statsmodels did not converge | **26** |
| mixedlm-rs found a strictly better optimum | **42** |
| same optimum | 52 |
| mixedlm-rs found a worse optimum | **0** |

These counts, and every other one quoted across the documentation, come from
`bench/baseline.json` — one recorded run with its commit, environment and
seeds. They are checked against it by `tests/test_baseline.py`.

That file exists because this table said **19** wins and **75** ties for a long
time after README.md and CORRECTNESS.md had moved to 42 and 52. Both were true
at some commit; nothing tied either to a run, so nothing caught the drift. Now
a stale number fails the test suite.
