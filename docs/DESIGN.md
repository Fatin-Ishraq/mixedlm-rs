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

`beta` and `sigma^2` fall out in closed form at the optimum, via the
normal-equation identity `r^2 = y'y - beta'(X'y) - u'(Lambda' Z'y)`.

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

Two of the six blocks were then dropped altogether. Because `A = L L'`, both
`A^-1 = L^-T L^-1` and `B = A^-1 W = L^-T (L^-1 W) = L^-T rzx` can be rebuilt in
pass 2 from `l` and `rzx`. The stride falls from `3q^2 + 2qp + q` to
`2q^2 + qp + q` — 26 doubles per group to 16 at `q = 2, p = 3`, or 26.0 MB to
16.0 MB per evaluation at 125,066 groups. A deviance-only call now never forms
`A^-1` at all, which is why the criterion-only evaluation drops a further 16%.

Objective evaluations are 80%+ of a fit, so this is most of the end-to-end
number. It also changed the story the staged benchmark tells: the Rust core was
measured at 1.5x over batched NumPy before this change, and 14.6x on the
current, fairer staging. The allocator had been hiding what the compiled core
was worth.

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
  instead of the generic loops. Measured headroom is real but bounded: per group
  the evaluation moves ~300 bytes for ~150 flops, so it is close to memory-bound
  and the remaining gain is perhaps 2x, not 10x.
- **Parallelising cross-product construction**, currently 16 ms of a 180 ms fit
  at 125,066 groups.
- **The in-Rust optimiser** is worse than scipy's and stays a fallback.
- **The Hessian for variance-component standard errors** is computed by
  differencing the analytic gradient. An analytic second derivative is derivable
  and would be both faster and more accurate.
