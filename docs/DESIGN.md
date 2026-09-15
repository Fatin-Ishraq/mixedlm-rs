# Why it is faster

`statsmodels.MixedLM` is slow for four separate reasons. They compound, and only
one of them is "it is written in Python".

## Where the time goes in the reference

`cProfile`, `n = 10,000`, 500 groups, random intercept and slope — a fit that
takes 9.87 s and reports `converged=False`:

```
ncalls    tottime  cumtime  function
479,500     2.483        -   mixed_linear_model.py:538  solver
    126     2.072    6.960  mixed_linear_model.py:1813 score_full
276,785     1.939    4.169  numpy/linalg/_linalg.py:374 solve
276,500     1.592    6.111  mixed_linear_model.py:472  _smw_solver
    149     1.046    7.863  mixed_linear_model.py:1588 loglike
550,500     0.737    0.876  mixed_linear_model.py:174  _dot
```

276,785 calls to `numpy.linalg.solve` — roughly one tiny numpy call per group per
iteration, where the fixed dispatch overhead of each call dwarfs its own
arithmetic.

## Where the time actually goes

An earlier version of this document claimed that `statsmodels` "optimises over
everything jointly" -- fixed effects, residual variance and covariance
parameters together -- and that it "recomputes constants inside the loop".
**Both claims are false**, and its own source says so. Its module docstring
states that the likelihood "is profiled over both the scale parameter (a
scalar) and the fixed [effects]"; `loglike(params, profile_fe=True)` is the
default, `score`/`score_full`/`score_sqrt` are analytic, and the per-group
random-effect cross-products are precomputed once into `_aex_r`/`_aex_r2`.

So the difference is **not** profiling, and the 1.7x measured for the S0 -> S1
step is not an isolated measurement of what profiling buys -- S0 also parses a
formula and computes inference that S1 does not. It is reported as an upper
bound, and labelled that way in `bench/stages.py`.

The real differences are these.

**1. Dense Sherman-Morrison-Woodbury, per group, per iteration.**
`_smw_solver` and `_smw_logdet` apply a dense update once per group per
iteration. With one grouping factor the penalised system
`Lambda' Z'Z Lambda + I` is exactly block diagonal -- `m` independent `q x q`
blocks -- so one Cholesky per block does the same work with none of the
per-group solve. This is the largest single factor by a wide margin.

**2. A Python loop over groups, in the inner loop.** The profile above shows
276,785 calls to `numpy.linalg.solve` -- roughly one tiny call per group per
iteration, where the fixed dispatch overhead of each call dwarfs its own
arithmetic. Batching every step across groups removes the interpreter from the
hot path entirely, and it is worth ~33x on its own *in NumPy*, before any Rust.
`proto/preml.py` demonstrates exactly this, in about 200 lines.

**3. A different parameterisation for the optimiser.** `statsmodels` optimises
its own covariance entries; the criterion here is a function of the relative
covariance factor `theta`, whose diagonal has a simple lower bound of zero.
That makes the boundary -- where a variance component is genuinely zero -- an
ordinary constrained optimum rather than a region the optimiser has to be
nursed through. The analytic gradient below is in those coordinates.

**4. The sparsity pattern is constant.** As `theta` varies, only the *values*
of `Lambda' Z'Z Lambda + I` change, never the pattern, so the factorisation
structure is settled once before the optimisation starts.

## What replaces it

The formulation is `lme4`'s (Bates, Mächler, Bolker & Walker 2015, JSS 67(1)).
Reparameterise with spherical random effects `u`, where `b = Lambda u`. For a
given `theta`, solve the penalised least squares problem
`min ||y - X beta - Z Lambda u||^2 + ||u||^2` through the Cholesky of
`Lambda' Z'Z Lambda + I`, then

```
ML:    d(theta) = log|L|^2 + n     (1 + log(2 pi r^2 / n))
REML:  d(theta) = log|L|^2 + log|RX|^2 + (n-p) (1 + log(2 pi r^2 / (n-p)))
```

`beta` and `sigma^2` fall out in closed form at the optimum. With `c_u` the
block-wise `L^-1 Lambda' Z'y` and `h = X'y - sum RZX'c_u`, the residual is
`r^2 = y'y - ||c_u||^2 - ||R_X^-T h||^2`, which needs nothing from a second pass
over the groups; see *Evaluation modes* below.

### The block-diagonal shortcut

With **one grouping factor**, `Lambda' Z'Z Lambda + I` is block diagonal: `m`
independent `q x q` blocks, where `q` is 1 or 2 in practice. No general sparse
solver is needed — and none is used. The whole factorisation is `m` tiny
Choleskys in a flat, contiguous, `rayon`-parallel loop.

This is worth more than anything else in the project. See BENCHMARKS.md.

### The analytic gradient

**What is and is not new here.** The gradient of the profiled criterion is not
a new idea, and this document should not have implied otherwise. Bates et al.
derive the profiled ML gradient in the lme4 paper (equations 46-48), and
`MixedModels.jl` documents derivative support of its own. What is implemented
here is a REML gradient specialised to the single-grouping-factor block
structure, evaluated in the same two passes that produce the criterion, at
essentially no extra cost per evaluation.

What is genuinely different is that it is *used by the optimiser*: both `lme4`
and `MixedModels.jl` optimise `theta` derivative-free, with BOBYQA, and so pay
a derivative-free evaluation count. Note that the comparison in BENCHMARKS.md
is against a finite-difference L-BFGS-B (stage S3), not against BOBYQA -- so it
measures what the analytic gradient buys *this* optimiser, not what it would
buy lme4. No claim is made about BOBYQA's evaluation count, which was not
measured.

Writing `D_k = dLambda/dtheta_k` (a single-entry
matrix with a 1 at `(r_k, c_k)`), `M_i = Z_i'Z_i Lambda`,
`A_i = Lambda' Z_i'Z_i Lambda + I`, `W_i = Lambda' Z_i'X`, `B_i = A_i^-1 W_i`,
and `P = (X'X - sum_i W_i' A_i^-1 W_i)^-1`:

```
d(ldL2)/dtheta_k  = 2 sum_i (M_i A_i^-1)[r_k, c_k]

d(pwrss)/dtheta_k = -2 sum_i u_i[c_k] * t_i[r_k]
                    with t_i = Z_i'y - Z_i'X beta - M_i u_i

d(ldRX2)/dtheta_k = -2 sum_i [ (Z_i'X)[r_k,:] . (B_i P)[c_k,:]
                               - M_i[r_k,:] . (B_i P B_i')[:,c_k] ]

d(dev)/dtheta_k   = d(ldL2) + [d(ldRX2)] + (dfree / pwrss) * d(pwrss)
```

The `pwrss` term uses the envelope theorem: `beta` and `u` minimise the penalised
least squares problem at fixed `theta`, so only the explicit `theta` dependence
contributes.

This cuts objective evaluations from 44–64 down to 11–13 against this
package's own finite-difference stage (see BENCHMARKS.md). The earlier
figure of "72–148 down to 10–12" quoted here did not match the table it
referred to.

Because a wrong gradient converges quietly to the wrong answer rather than
crashing, it is checked against central finite differences over the full
product of `q = 1..3` and `p = 1,2,5,10`, both criteria, four random feasible
thetas each, plus five fixed thetas and the variance-zero boundary.

## Two numerical traps, both found by fuzzing

**`theta = 0` is a stationary point for any data whatsoever.** At `Lambda = 0`
every gradient term vanishes identically: `M = 0` so `d(ldL2) = 0`; `u = 0` so
`d(pwrss) = 0`; `B = 0` so `d(ldRX2) = 0`. A gradient-based optimiser that
reaches the bound stops there and reports a zero projected gradient — even when
the true optimum has an ordinary non-zero variance. We were silently returning
`converged=True` on a worse fit.

Singular fits are real and must stay reportable (`Dyestuff2` genuinely has a
between-batch variance of zero), so the bound cannot be forbidden. Instead,
whenever a diagonal entry lands on it, a few positive values along that
coordinate are probed and the fit restarted if any is better.

For a scalar random effect the escape works in `t = theta^2` instead, where
the bound is *not* stationary: the one-sided derivative `f'(0) = D''(0)/2` is
negative exactly when moving off zero improves the fit, and `D''` is
available analytically. The criterion is scanned over eight decades of
`theta` (criterion-only calls, about a microsecond each on an aggregated
core), every grid minimum better than the bound is re-optimised, and the
probe ladder above remains the fallback when the scan cannot vouch for the
result. Across the 120 differential and 400 stress fixtures it reached the
ladder's optimum on every scalar fit, to within the criterion's own rounding on
near-collinear designs, and cut the optimiser's evaluations on scalar fits
from 15 to 2 at a boundary optimum and from 27-35 to 12-13 at an interior
one.

**Predictor scaling.** The random-effects design is internally rescaled to unit
column RMS. This is an *exact* reparameterisation, not an approximation:
`Z -> Z D^-1` with `Lambda -> D Lambda` leaves `Lambda' Z'Z Lambda + I`
unchanged, so the criterion surface is identical and only the coordinates move.
What it buys is that `theta = I` is a sensible starting point whatever units the
data is in. Results are transformed back before being returned.

## Allocation, and why the Rust core is worth 8x rather than 1.5x

The first working version allocated about ten small `Vec`s per group per
objective evaluation — `ainv`, `m_mat`, `l`, `b`, `rzx`, `cu`, plus per-group
accumulator temporaries. At 125,000 groups that is roughly a million allocations
per evaluation, and around ten million per fit.

Every per-group intermediate now lives in one flat buffer of
`m * (3q^2 + 2qp + q)` doubles, split per group with `split_at_mut`, and taken
from a small pool on the core rather than allocated per evaluation.
Intermediates are staged in place: `A` is built directly in the slot that
will hold its Cholesky factor, and `W` is staged in the slot that becomes
`L^-1 W`. Accumulators belong to fixed-size chunks of groups — not to each
group, and not to rayon's work-stealing splits, which is what an earlier
version of this paragraph wrongly called "one set per thread" — and
`RZX'RZX` is accumulated with a direct loop rather than by forming a `p x p`
temporary.

| groups | one objective evaluation, before | after | |
|---:|---:|---:|---|
| 5,000 | 3.31 ms | **0.42 ms** | 7.9x |
| 20,000 | 19.69 ms | **1.63 ms** | 12.1x |
| 125,066 | 89.34 ms | **9.74 ms** | 9.2x |

Two of the six blocks were then dropped altogether. Because `A = L L'`, both
`A^-1 = L^-T L^-1` and `B = A^-1 W = L^-T (L^-1 W) = L^-T rzx` can be rebuilt in
pass 2 from `l` and `rzx`. The stride falls from `3q^2 + 2qp + q` to
`2q^2 + qp + q` — 26 doubles per group to 16 at `q = 2, p = 3`, or 26.0 MB to
16.0 MB per evaluation at 125,066 groups. A criterion-only call now uses no
per-group buffer at all; see below.

Objective evaluations are 80%+ of a fit, so this is most of the end-to-end
number. It also changed the story the staged benchmark tells: the Rust core was
measured at 1.5x over batched NumPy before this change, and 14.6x on the
current, fairer staging. The allocator had been hiding what the compiled core
was worth.

## Exact aggregation, evaluation modes and deterministic scheduling

These came out of `.review/OPTIMIZATION-ROADMAP.md`. Each is an exact
rearrangement of the same criterion, checked against the independent dense
implementation and against the other kernels in `tests/test_kernels.py`, and
measured phase by phase with `bench/phases.py` (results in
`bench/phases.json`: this build against the pre-change one, alternating
rounds, 1 and 12 rayon threads, BLAS pinned to one).

**The augmented Schur form.** With `v_i = [Z_i'X | Z_i'y]` and
`P_i = Lambda A_i^-1 Lambda'`, the profiled system is
`K = [X y]'[X y] - sum_i v_i' P_i v_i`. Its leading block is the fixed-effect
system `H`, and `pwrss = c - h'H^-1 h`. Because `beta` minimises `e'Ke` over
`e = [-beta, 1]`, `d(pwrss) = e' dK e` and `d(ldRX2) = tr(H^-1 dH)`, with
`dP_k = F_k + F_k'`, `F_k = e_r (Lambda A^-1)[:,c]' - (Lambda A^-1)[:,c] (S P)[r,:]`.

**Aggregation (`evaluator="aggregated"`).** `P_i` depends on a group only
through `S_i = Z_i'Z_i`, so groups with bit-identical `S_i` contribute
`sum_ab P_ab T[a,b]`, where `T[a,b] = sum_i v_i[a]' v_i[b]` is summed once, at
construction. An evaluation then costs the number of *distinct* `Z_i'Z_i`,
not the number of groups. That is one class for a balanced random intercept,
and at most about `sqrt(2n)` for any random intercept (the distinct group
sizes cannot sum to more than `n`). It is one class too for a random slope
over a repeated visit schedule. The core detects classes when it is built and
uses them when they are few enough to pay; a continuous random slope has no
repetition, gives up after a few thousand groups, and keeps the block kernel.

| 125,066 groups, 1 thread | gradient, before | after | whole fit, before | after |
|---|---:|---:|---:|---:|
| balanced random intercept | 18.8 ms | 0.0014 ms | 204 ms | 47 ms |
| random intercept, 1-8 rows per group | 18.5 ms | 0.0019 ms | 226 ms | 56 ms |
| intercept + slope in visit number | 33.8 ms | 0.0022 ms | 442 ms | 69 ms |

The whole fit does not shrink with the kernel: what remains is the design
conditioning, the core's construction and the one full solution.

**Streaming (`evaluator="streaming"`).** The same form one group at a time:
a single pass accumulating `K` and every `dK_k`, with no per-group buffer. It
does more arithmetic per group than the block kernel: level with it at `q = 2,
p = 3` on 12 threads (3.4 ms against 3.6 ms per gradient, 125,066 groups),
slower on one thread (22.3 ms against 17.2 ms), and four times slower at
`p = 30` (80 ms against 20 ms), where its `(p+1)^2` accumulators per `theta`
entry dominate. It is available but never chosen automatically.

**Evaluation modes.** A criterion-only call (`deviance`) makes one pass, stores
nothing per group and forms no inverse; a gradient call skips the random
effects and forms `(X'V^-1X)^-1` only under REML; only `solution` forms
everything. The one-pass residual is a difference of large sums, so its
accumulators are Neumaier-compensated. Without that, a fixture's last line
search went uphill by a few ulps and L-BFGS-B stopped "abnormally" after 38
evaluations instead of 10.

**Scheduling.** Work is split into chunks sized from the dimensions alone,
and chunk results are summed in index order, so the criterion is
bit-for-bit identical on 1 thread and on 12. Below about 250,000 flops of
total work it runs serially. The previous rayon `fold`/`reduce` followed the
work-stealing splits: on the degenerate stress case 252 it returned a
different optimum on each of seven runs, and on one fixture a thread-count change moved
the optimiser from 9 evaluations to 20. The `q = 1, 2, 3` kernels are compiled
with `q` as a constant; larger `q` uses the generic loops.

| continuous random slope, 125,066 groups | before | after |
|---|---:|---:|
| gradient, 1 thread | 34.3 ms | 16.9 ms |
| gradient, 12 threads | 7.7 ms | 3.9 ms |
| whole fit, 12 threads | 145 ms | 98 ms |
| 18 groups, gradient, 12 threads | 0.033 ms | 0.004 ms |

**Construction.** One triangle of `X'X` and `Z_i'Z_i`, then mirrored; with
more than one thread, rows are counting-sorted by group and accumulated over
disjoint ranges of groups, in row order within a group, so the sums match
the serial loop's. 13.2 ms to 10.4 ms at 125,066 groups on 12 threads, and
17.7 ms to 9.1 ms at `p = 30`. On one thread it is level with before, except
that finding classes on a repeated design costs 2.5-7 ms, repaid by the
first evaluation.

**Reuse.** `MixedLM` keeps the response-independent work -- column scales,
the scaled design, and a template core -- keyed on a checksum of its design
arrays, so an ML refit after a REML fit rebuilds nothing, and
`model.with_endog(y)` builds a core for a new response from `X'y`, `Z_i'y`
and the class assignment alone (`LmmCore.with_response`, bit-identical to a
fresh build). A refit for a new response is 2-3x faster on a continuous slope
and 6-30x on an aggregated design.

**Results.** The analytic scalar Hessian (`LmmCore.deviance_hessian`) replaces
two differenced gradients for `q = 1`; first access to `bse` falls from
37.5 ms to 0.3 ms at 125,066 groups. `q > 1` still differences the analytic
gradient -- `2 * n_theta` calls, now each about half the cost.
`random_effects_cov` is computed in the core from the cross-products it
already holds, once, and `random_effects_cov_array` returns it without
building a DataFrame per group: 2.8 s to 3-11 ms at 20,000 groups.

## What is compiled, and what is not

The compiled surface is deliberately small — the same discipline as the earlier
packages in this family. Only what runs many times per fit is in Rust:
cross-product construction, the block-diagonal Cholesky, the profiled criterion,
and the analytic gradient. The whole fit is one call with the GIL released.

**The optimiser is scipy's L-BFGS-B, called from Python.** A hand-rolled
projected L-BFGS is included and measurably worse — 35–64 objective evaluations
against scipy's 10–12 for the same fits. This is not the "own the loop"
anti-pattern: that rule matters when you would cross the FFI boundary once per
group per iteration, whereas here a whole fit crosses it 10–12 times no matter
how many groups there are. Paying a few microseconds to use a mature Fortran
line search is the right trade. `method="rust"` selects the in-Rust optimiser for
callers who want no scipy in the loop.

```
src/linalg.rs   dense Cholesky, triangular solves, small matmuls (no BLAS dependency)
src/lmm.rs      cross-products, the block, aggregated and streaming kernels,
                analytic gradient and scalar Hessian
src/optim.rs    projected L-BFGS (the method="rust" path)
src/lib.rs      PyO3 bindings

python/mixedlm_rs/_fit.py                the fitting driver, multi-start, boundary escape
python/mixedlm_rs/mixed_linear_model.py  the statsmodels-shaped API
python/mixedlm_rs/_install.py            aliasing for code you cannot edit
```

## What is left on the table

- **Crossed and nested random effects** break block-diagonality and need a real
  sparse Cholesky with a fill-reducing ordering. That is the largest piece of
  unfinished work and the reason this release is scoped to one grouping factor.
- **The Python-side preparation** is now most of a fit on an aggregated
  design: the rank-revealing least-squares solve (12 ms at 500,264 rows) and
  the group coding. The solve is not replaced by the normal equations, which
  would square the condition number the rank check exists to measure.
- **The in-Rust optimiser** is worse than scipy's and stays a fallback.
- **An analytic Hessian for `q > 1`**, and a second-order optimiser built on
  it. The scalar case is done; the matrix case adds per-group work to every
  evaluation and is unmeasured.
