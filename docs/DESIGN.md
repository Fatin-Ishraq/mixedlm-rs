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

## The four mistakes

**1. It optimises over everything jointly.** `statsmodels` hands the optimiser
the fixed effects, the residual variance *and* the covariance parameters
together — about 7 parameters for a random-slope model. `lme4` profiles the
fixed effects and `sigma^2` out analytically, so the optimiser sees 3. This is
the primary cause of both the speed gap and the convergence failures: a smaller,
better-conditioned search space converges, a larger badly-scaled one reports
`|grad| = 166`.

**2. Dense Sherman-Morrison-Woodbury, per group, per iteration.** `_smw_solver`
and `_smw_logdet` apply a dense update once per group per iteration. The correct
structure is a single Cholesky of the whole penalised system.

**3. It ignores that the sparsity pattern is constant.** As `theta` varies, only
the *values* of `Lambda' Z'Z Lambda + I` change — never the pattern. The
factorisation structure can be settled once, before the optimisation starts.

**4. It recomputes constants inside the loop.** `Z'Z`, `X'X`, `X'Z`, `X'y` and
`Z'y` do not depend on `theta` at all. Here they are formed exactly once.

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

`beta` and `sigma^2` fall out in closed form at the optimum, via the
normal-equation identity `r^2 = y'y - beta'(X'y) - u'(Lambda' Z'y)`.

### The block-diagonal shortcut

With **one grouping factor**, `Lambda' Z'Z Lambda + I` is block diagonal: `m`
independent `q x q` blocks, where `q` is 1 or 2 in practice. No general sparse
solver is needed — and none is used. The whole factorisation is `m` tiny
Choleskys in a flat, contiguous, `rayon`-parallel loop.

This is worth more than anything else in the project. See BENCHMARKS.md.

### The analytic gradient

`lme4` and `MixedModels.jl` both optimise `theta` derivative-free, with BOBYQA.
We derive the gradient instead. Writing `D_k = dLambda/dtheta_k` (a single-entry
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

This cuts objective evaluations from 72–148 down to 10–12. Because a wrong
gradient converges quietly to the wrong answer rather than crashing, it is
checked against central finite differences across `q = 1..3`, `p = 1,2,5,10`,
both criteria, five thetas, and at the variance-zero boundary.

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
`m * (3q^2 + 2qp + q)` doubles, allocated once per evaluation and split per group
with `split_at_mut`. Intermediates are staged in place: `A` is built directly in
the slot that will hold its Cholesky factor, and `W` is staged in the slot that
becomes `L^-1 W`. Accumulators live in rayon `fold` state — one set per thread,
not one per group — and `RZX'RZX` is accumulated with a direct loop rather than
by forming a `p x p` temporary.

| groups | one objective evaluation, before | after | |
|---:|---:|---:|---|
| 5,000 | 3.31 ms | **0.42 ms** | 7.9x |
| 20,000 | 19.69 ms | **1.63 ms** | 12.1x |
| 125,066 | 89.34 ms | **9.74 ms** | 9.2x |

Objective evaluations are 80%+ of a fit, so this is most of the end-to-end
number. It also changed the story the staged benchmark tells: the Rust core was
measured at 1.5x over batched NumPy before this change and 8.0x after. The
allocator had been hiding what the compiled core was worth.

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
src/lmm.rs      cross-products, profiled criterion, analytic gradient
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
- **Specialising the `q = 1` and `q = 2` cases** with fixed-size arithmetic
  instead of the generic loops, now that allocation no longer dominates.
- **The in-Rust optimiser** is worse than scipy's and stays a fallback.
- **The Hessian for variance-component standard errors** is computed by
  differencing the analytic gradient. An analytic second derivative is derivable
  and would be both faster and more accurate.
