# Correctness

## Why statsmodels cannot be the only oracle

The packages that preceded this one were verified by *identical output*: run the
reference and the replacement on the same input, demand the same numbers,
bug-for-bug.

That standard cannot be applied wholesale here, because on a substantial
fraction of ordinary inputs **statsmodels does not converge and returns
estimates anyway**. Reproducing it exactly would mean reproducing fits that are
wrong.

So the bar is split:

1. Where statsmodels converges, we must agree with it.
2. Where it does not, we must converge — and that is asserted, not compared.
3. Where we find a strictly better optimum, the parameters legitimately differ,
   and demanding agreement would be asserting that we reproduce a worse fit.
4. The primary oracle is **lme4's published fits**, not statsmodels.

## Primary oracle: lme4's published datasets

From Bates, Mächler, Bolker & Walker (2015), *Fitting Linear Mixed-Effects
Models Using lme4*, JSS 67(1), and confirmed against a live R 4.6.1 + lme4
install. Reproduced exactly:

| dataset | model | quantity | lme4 | mixedlm-rs |
|---|---|---|---:|---:|
| `sleepstudy` | `Reaction ~ Days + (Days\|Subject)` | REML criterion | 1743.6 | **1743.6284** |
| | | (Intercept) | 251.405 | **251.4051** |
| | | Days | 10.467 | **10.4673** |
| | | sd(Intercept) | 24.741 | **24.7410** |
| | | sd(Days) | 5.922 | **5.9221** |
| | | corr | 0.066 | **0.0656** |
| | | sd(Residual) | 25.592 | **25.5918** |
| | | ML deviance | 1751.9 | **1751.9393** |
| `Dyestuff` | `Yield ~ 1 + (1\|Batch)` | REML criterion | 319.65 | **319.6543** |
| | | (Intercept) | 1527.5 | **1527.5000** |
| | | sd(Batch) | 42.00 | **42.0006** |
| | | sd(Residual) | 49.51 | **49.5101** |
| `Dyestuff2` | `Yield ~ 1 + (1\|Batch)` | REML criterion | 161.8283 | **161.8283** |
| | | (Intercept) | 5.6656 | **5.6656** |
| | | var(Batch) | 0 (singular) | **0.0 exactly** |
| | | sd(Residual) | 3.715684 | **3.715684** |

`Dyestuff2` is the important one: the optimum sits exactly on the boundary. This
is where statsmodels emits `Random effects covariance is singular` and `The MLE
may be on the boundary of the parameter space`. **A boundary optimum is a
converged fit, not a failure**, and we report `converged=True` — which matters,
because a scientist who sees a convergence warning starts deleting
random-effects terms, and that is a documented source of anti-conservative
p-values.

## Differential testing against statsmodels

120 randomised fixtures varying group count (5–120), group size (2–30), balance,
signal-to-noise, predictor scale over four orders of magnitude, the number of
random-effect terms, and REML vs ML:

| outcome | count |
|---|---:|
| statsmodels did not converge | **26** |
| we found a strictly **better** optimum | **19** |
| same optimum, parameters agree | 75 |
| we found a **worse** optimum | **0** |

On 45 of 120 fixtures — 37.5% — statsmodels either failed to converge or landed
on a worse optimum. We were never worse on any fixture.

Where both converge to the same optimum, fixed effects are compared **on the
scale of their own standard errors**, and agree to under 2% of one SE. That is
the scale that means anything scientifically: two optimisers stopping at
slightly different points on a flat likelihood is not a disagreement between
implementations.

Variance-component standard errors agree with statsmodels to within 5%, and on
`sleepstudy` to about five decimal places
(0.464239 / 0.071136 / 0.023817 against 0.464246 / 0.071137 / 0.023817).

## Adversarial stress sweep

A wider sweep than the committed fuzz suite: 400 cases across six deliberately
hostile shapes -- many groups of exactly two observations, extreme imbalance
(one huge group among tiny ones), a handful of very large groups, predictors
spanning six orders of magnitude, near-collinear fixed effects, and heavy
outliers.

| outcome | count |
|---|---:|
| we raised an exception | **0** |
| statsmodels failed or did not converge | 71 |
| we found a strictly better optimum | **106** |
| same optimum | 220 |
| we found a worse optimum | 3 |

Of the three: two are models where `n = q * m` exactly, which are unidentifiable
and now warn (see below) -- the likelihood diverges there rather than attaining
a maximum, so comparing optima is meaningless. The third is a genuine near-tie
on a near-collinear surface, differing by 8.6e-06 in relative terms.

This sweep found two defects that the narrower 120-case suite did not.

**The boundary-escape ladder was too coarse.** Its smallest probe was 0.05, so a
true optimum at `theta = 0.028` was unreachable: every probe overshot it and the
optimiser slid back into the stationary point at zero. The ladder now reaches
down to 1e-3.

**Unidentifiable models were fitted silently.** See below.

## Identifiability

With at least as many random effects as observations (`n <= q * m`), every group
is fitted perfectly, the residual variance is driven towards zero and the
profiled likelihood **diverges** instead of attaining a maximum. Two
implementations will simply stop at different points along that path, and
neither answer means anything.

`lme4` refuses such models outright. `statsmodels` fits them silently. Refusing
would break the drop-in contract, so we warn -- which is the part that actually
protects the user.

A related non-finding, worth recording because it looks like a bug and is not:
a variance component estimated as **exactly zero is often correct**. The REML
estimate legitimately sits on the boundary whenever the observed between-group
spread is no larger than sampling noise would produce. On one such fixture
statsmodels returns exactly 0.0 as well, with a likelihood identical to ours to
eight decimal places. The property worth asserting is the criterion, not the
parameter.

## The analytic gradient

This is the one place the project goes beyond its references, so it is the most
heavily tested code here. Checked against central finite differences across
`q = 1, 2, 3`, `p = 1, 2, 5, 10`, five values of `theta`, both criteria, and at
the variance-zero boundary — 22 tests. A wrong gradient does not crash; it
converges quietly to the wrong answer.

## Documented divergences from statsmodels

**D-1. `fittedvalues` includes the random effects.** Matching statsmodels:
`fittedvalues` is the conditional fit `X beta + Z b`, while `predict()` remains
marginal (fixed effects only). This was a real bug caught by differential
testing.

**D-2. Variance-component standard errors come from a different
parameterisation.** statsmodels differentiates its own full-parameter
likelihood; we apply the delta method to the Hessian of the *profiled*
criterion, obtained by central-differencing the analytic gradient. The two agree
to a few percent. lme4 declines to report these at all, on the grounds that
their sampling distribution is poorly behaved near the boundary — a caution
worth taking seriously.

**D-3. `cov_params()` is exact only in the fixed-effect block.** The
variance-component rows carry the delta-method variances on the diagonal and
`NaN` off-diagonal, rather than a fabricated full covariance.

**D-4. Convergence is certified independently of the optimiser's flag.** A
component pinned at its lower bound with the gradient pushing further into the
bound is stationary, not a failure. We check the projected gradient ourselves.

**D-5. `__version__` reports this package's version.** The statsmodels API level
being emulated is published separately as `__statsmodels_version__`. Claiming to
be statsmodels 0.15.0 would fix one version check and lie to every other.

## Open questions

**OQ-1 — RESOLVED.** This previously recorded a suspected discrepancy in the
`Dyestuff2` residual standard deviation: the value 3.653 was believed to be in
the lme4 literature, against 3.7157 computed here and by statsmodels. Checked
against a live R 4.6.1 install with lme4:

```
REML criterion: 161.8283      sigma: 3.715684      residual var: 13.80631
Batch var: 0                  intercept: 5.6656    SST/(n-1): 13.80631
```

lme4 reports **3.715684**, agreeing with this package exactly. There was no
discrepancy -- the 3.653 figure was a misremembering, and it corresponds to
`SST/n` rather than the REML divisor `SST/(n-1)`. The row is now in the oracle
table above.

**OQ-2 — statsmodels#9097.** The issue body (41 minutes vs 1–2 seconds for
`lmer`, 125,066 groups) and its open status were verified; the comment thread
was not retrieved, so it is unknown whether maintainers replied or consider it
fixable in-tree.

## Running the suite

```bash
pip install maturin pytest numpy scipy pandas patsy statsmodels
python -m maturin build --release --out dist
pip install --force-reinstall --no-deps --no-index --find-links dist mixedlm-rs
pytest tests/ -q      # 222 tests
cargo test --lib      # 7 tests
```

The differential and fuzz tests skip cleanly when statsmodels is absent — they
just check less.
