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
2. Where it does not, there is nothing to compare against — so the result is
   certified *independently*, from the deviance alone: a finite-difference
   projected gradient and a spread of perturbations, neither of which uses the
   analytic gradient or our own convergence flag. Asserting our own flag here,
   which is what this used to do, is circular precisely where independence
   matters most.
3. Where we find a strictly better optimum, the parameters legitimately differ,
   and demanding agreement would be asserting that we reproduce a worse fit.
4. The primary oracle is **lme4's published fits**, not statsmodels.

One caveat on the framing above, which the reviewer was right to press: a
competing optimiser's convergence *warning* does not by itself mean its
estimates are wrong. What justifies preferring our answer in those cases is the
criterion — where we report a higher likelihood on the same model and data, that
is a checkable fact, and it is what the tables below classify on.

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

Under **ML**, from the same R 4.6.1 + lme4 install:

| dataset | quantity | lme4 | mixedlm-rs |
|---|---|---:|---:|
| `Dyestuff` | ML deviance | 327.327060 | **327.327060** |
| | sd(Batch) | 37.260345 | **37.260345** |
| | sd(Residual) | 49.510100 | **49.510100** |
| `Dyestuff2` | ML deviance | 162.873037 | **162.873037** |
| | sd(Batch) | 0 (singular) | **0.0 exactly** |
| | sd(Residual) | 3.653231 | **3.653231** |
| `sleepstudy` | ML deviance | 1751.939344 | **1751.939344** |
| | sd(Intercept) / sd(Days) | 23.779760 / 5.716799 | **23.779760 / 5.716799** |
| | corr | 0.081321 | **0.081321** |
| | sd(Residual) | 25.591907 | **25.591907** |

And a model shape the table did not previously cover, `Reaction ~ Days +
(1|Subject)` under REML: criterion 1786.465085, sd(Subject) 37.123827,
sd(Residual) 30.991234 — all reproduced.

`Dyestuff2` is the important one: the optimum sits exactly on the boundary. This
is where statsmodels emits `Random effects covariance is singular` and `The MLE
may be on the boundary of the parameter space`. **A boundary optimum is a
converged fit, not a failure**, and we report `converged=True` — which matters,
because a scientist who sees a convergence warning starts deleting
random-effects terms, and that is a documented source of anti-conservative
p-values.

## Differential testing against statsmodels

120 randomised fixtures varying group count (5-120), group size (2-30), balance,
signal-to-noise, predictor scale over four orders of magnitude, the number of
random-effect terms, and REML vs ML. Reproduce the counts with
`python bench/differential_table.py`; the same fixtures are asserted on by
`pytest tests/test_fuzz.py`.

| outcome | count |
|---|---:|
| statsmodels did not converge | **26** |
| we found a strictly **better** optimum | **42** |
| same optimum | 52 |
| we found a **worse** optimum | **0** |

On 68 of 120 fixtures — 56.7% — statsmodels either failed to converge or landed
on a worse optimum. We were never worse on any fixture.

Classification is on the **criterion**, in deviance units, which is what the two
optimisers are competing on. Where both reach the same optimum, fixed effects
are compared on the scale of their own standard errors, and agree to under 2% of
one SE. That is the scale that means anything scientifically: two optimisers
stopping at slightly different points on a flat likelihood is not a
disagreement between implementations.

Where statsmodels does **not** converge there is nothing to compare against, and
asserting our own `converged` flag there would be circular. That branch instead
certifies the result independently, from the deviance alone: a central
finite-difference projected gradient, and a spread of random perturbations none
of which may improve the criterion. Neither uses the analytic gradient or the
convergence flag, both of which are the package's own claims.

Variance-component standard errors agree with statsmodels to within 5%, and on
`sleepstudy` to about five decimal places
(0.464239 / 0.071136 / 0.023817 against 0.464246 / 0.071137 / 0.023817).

## Adversarial stress sweep

400 cases across six deliberately hostile shapes — many groups of exactly two
observations, extreme imbalance (one huge group among tiny ones), a handful of
very large groups, predictors spanning six orders of magnitude, near-collinear
fixed effects, and heavy outliers.

This sweep used to live outside the repository, so its classifications could not
be checked. It is committed as `bench/stress_sweep.py`; reproduce with
`python bench/stress_sweep.py --cases 400 --seed 0`.

| outcome | count |
|---|---:|
| we raised an exception | **0** |
| statsmodels did not converge | 99 |
| we found a strictly **better** optimum | **131** |
| same optimum | 167 |
| we found a **worse** optimum | 3 |
| we failed to certify a stationary point | 1 |

The one case where stationarity could not be certified is reported as
`converged=False` with the projected gradient in the message, which is the
honest answer for a near-collinear design whose criterion is nearly flat in one
direction. It is not silently reported as a success.

**The three losses, stated rather than argued away.** An earlier version
dismissed two of them as unidentifiable models where "the likelihood diverges",
which was wrong on the mathematics (see below) and unverifiable, since neither
the seeds nor the script were committed. The current sweep reports each loss
with its **absolute** deviance gap — relative tolerances on a log-likelihood are
meaningless, because the criterion carries an arbitrary additive constant:

| case | shape | n | groups | deviance worse by |
|---:|---|---:|---:|---:|
| 100 | near-collinear | 1,035 | 72 | 5.619e-05 |
| 214 | near-collinear | 186 | 15 | 5.710e-05 |
| 282 | groups of two | 264 | 132 | 2.737e-06 |

All three are near-ties on hard surfaces, on a criterion whose own scale is in
the hundreds. They are losses nonetheless, and are recorded as losses.

The sweep also reclassifies a loss as *unidentified* only when the criterion is
**measured** to be flat along the variance split, never on a counting rule. No
case in this run met that condition.

This sweep is what found the boundary-escape ladder being too coarse: its
smallest probe was 0.05, so a true optimum at `theta = 0.028` was unreachable —
every probe overshot it and the optimiser slid back into the stationary point at
zero. The ladder now reaches down to 1e-3.

## Identifiability

**This section was wrong and has been rewritten.** It previously argued that
`n <= q * m` implies every group is fitted perfectly, the residual variance is
driven to zero, and the profiled likelihood *diverges*. Both halves of that are
false, and the counterexamples are simple:

- **The rule over-fires.** With `q = 2` random effects and two observations per
  group, `n == q * m` exactly — and the model is perfectly well identified, with
  an interior optimum and a positive residual variance. The old check warned
  about it purely on the row count. `tests/test_degenerate.py::
  test_saturated_but_identified_model_does_not_warn` pins this.
- **The rule under-fires.** A single group, whose random intercept is exactly
  confounded with the fixed intercept, is completely unidentified, and
  `n <= q * m` is false there (60 observations against 1 random effect).
- **And when it does fire, the reason given was wrong.** One random intercept
  per singleton observation gives `V = (tau^2 + sigma^2) I`. The criterion is
  finite and exactly *constant* along the trade-off between the two variances —
  **flat, not divergent**. Measured directly on the core: `94.395392420` at
  `theta` = 0, 1, 10 and 100.

What the code does now: the structural count decides only *where to look*, and
the claim is settled by asking the criterion. On a suspect design the fit probes
whether the deviance changes along the variance split, and warns only if it does
not — saying the criterion is flat and the reported split is one arbitrary point
on a ridge. A criterion that is unbounded rather than flat has no stationary
point, so it surfaces as a convergence warning instead.

`lme4` refuses such models outright. `statsmodels` fits them silently. Refusing
would break the drop-in contract, so we warn.

A related non-finding, worth recording because it looks like a bug and is not:
a variance component estimated as **exactly zero is often correct**. The REML
estimate legitimately sits on the boundary whenever the observed between-group
spread is no larger than sampling noise would produce. On one such fixture
statsmodels returns exactly 0.0 as well, with a likelihood identical to ours to
eight decimal places. The property worth asserting is the criterion, not the
parameter.

## Singular fits are reported as such

A boundary optimum is a converged fit — but it is not an ordinary one, and the
two should not be conflated in either direction. `results.singular` is True when
a variance component sits on its bound, `summary()` says so in the header and
adds a note, and `docs/LIMITATIONS.md` states that Wald intervals for variance
parameters do not apply there.

This follows `lme4`, which reports singularity separately from convergence
(`isSingular`). Reporting a boundary fit as a *convergence failure*, as
statsmodels effectively does, is what drives people to delete random-effect
terms until the warning goes away — an anti-conservative practice. Reporting it
as an unremarkable success would be the opposite error: the estimate is fine,
the inference around it is not.

## The analytic gradient

A wrong gradient does not crash; it converges quietly to the wrong answer. So it
is the most heavily tested code here: checked against central finite differences
over the **full product** of `q = 1, 2, 3` and `p = 1, 2, 5, 10`, both criteria,
four random feasible thetas each, plus five fixed thetas and the variance-zero
boundary.

Two things about that sweep were fixed after review. It previously varied `q`
with `p` pinned at 2, and `p` with `q` pinned at 2, so no case with `q = 3` and
`p = 10` was ever evaluated -- and the interaction between the two is exactly
where a stride error in the block layout would show up. And thetas whose
criterion came back non-finite were skipped with `continue`, silently removing
cases from the sweep; the generator must now produce feasible points, and it is
asserted that every generated theta was actually checked.

At `theta = 0` the test asserts more than a finite deviance. Every term of the
analytic gradient vanishes identically at `Lambda = 0`, so "the gradient is
zero" there proves nothing on its own. The forward difference at step `h`
measures `f'(0) + (1/2) f''(0) h`, so at a true stationary point it does not
vanish -- it shrinks *linearly in h*. That is what is asserted: a 100x smaller
step must give a ~100x smaller slope, which distinguishes a genuine stationary
point from a lucky zero.

**On novelty.** An earlier version of this document called the gradient "the one
place the project goes beyond its references". That overstates it. Bates et al.
derive the profiled ML gradient in the lme4 paper (eq. 46-48), and
`MixedModels.jl` documents derivative support of its own. What is here is a REML
gradient specialised to the single-grouping-factor block structure and evaluated
in the passes that already produce the criterion. What is different in practice
is that the optimiser *uses* it, where lme4 and MixedModels.jl both run
derivative-free BOBYQA.

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

**D-3. `cov_params()` carries only the fixed-effect block.** The
variance-component rows carry the delta-method variances on the diagonal and
`NaN` off-diagonal, rather than a fabricated full covariance.

An earlier version called that block "exact". That overstates it. It is
`sigma^2 (X' V(theta_hat)^-1 X)^-1`, the GLS covariance **conditional on the
fitted variance parameters** — the standard mixed-model quantity, and the one
both lme4 and statsmodels report, but not in general the fixed-effect block of
the inverse full observed information: estimating `theta` can introduce
cross-block terms. It is therefore mildly anti-conservative in small samples,
which is exactly what Kenward-Roger and Satterthwaite corrections exist to fix,
and neither is implemented here. See docs/LIMITATIONS.md.

**D-6. `bse_re` follows the reference's definition, including its quirk.**
statsmodels computes `sqrt(scale * diag(cov_params())[k_fe:])`, i.e.
`sqrt(scale)` times the standard errors of the unscaled covariance parameters
that `bse` reports — while the value tabulated beside them in `summary()` is
`cov_re`, which is `scale` times that parameter. Estimate and standard error in
that table therefore differ by a further `sqrt(scale)`. We reproduce it, because
a drop-in that silently redefines an attribute is worse than one that documents
an inherited wart. `bse_cov_re` gives errors on the same scale as `cov_re`.

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

lme4 reports **3.715684** for the REML fit, agreeing with this package exactly.

A later check found where 3.653 actually comes from, and it is not a
misremembering: it is the **ML** residual standard deviation for the same model.
Running `lmer(Yield ~ 1 + (1|Batch), Dyestuff2, REML=FALSE)` gives sigma
**3.653231**. Both numbers are real, and the difference is the divisor -- REML
uses `n - p`, ML uses `n`, which here is 29 against 30. This package reproduces
both, and `test_dyestuff2_ml_residual_is_the_3653_figure` pins them together so
the confusion cannot recur.

Full ML oracles for all three datasets, from the same live R install, are in the
table above and in `tests/test_lme4_oracles.py`. Previously only `sleepstudy`'s
ML deviance was checked.

**OQ-2 — statsmodels#9097.** The issue body (41 minutes vs 1–2 seconds for
`lmer`, 125,066 groups) and its open status were verified; the comment thread
was not retrieved, so it is unknown whether maintainers replied or consider it
fixable in-tree.

## Running the suite

```bash
pip install maturin pytest numpy scipy pandas patsy statsmodels
python -m maturin build --release --out dist
pip install --force-reinstall --no-deps --no-index --find-links dist mixedlm-rs
pytest tests/ -q      # 505 tests
cargo test --lib      # 7 tests
```

The differential and fuzz tests skip cleanly when statsmodels is absent — they
just check less.
