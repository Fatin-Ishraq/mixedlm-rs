"""The fitting driver: builds the Rust core, optimises theta, derives everything.

Design note -- why scipy drives the optimiser
---------------------------------------------
The objective and its analytic gradient are in Rust. The *optimiser* is scipy's
L-BFGS-B, called from Python, because it measurably beats the hand-rolled
projected L-BFGS: 10-12 objective evaluations against 35-64 for the same fits.

This is not the "own the loop" anti-pattern. That rule matters when you would
cross the FFI boundary once per group per iteration; here we cross it 10-12
times in total for a whole fit, no matter how many groups there are. Paying a
handful of microseconds to use a mature Fortran line search is the right trade.

``method="rust"`` runs the in-Rust optimiser instead, for callers who want no
scipy in the loop.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

from ._mixedlm_rs import LmmCore

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

__all__ = ["ConvergenceWarning", "fit_core"]


class ConvergenceWarning(UserWarning):
    """Raised when the optimiser cannot certify a stationary point."""


def _theta_index(q):
    return [(r, c) for c in range(q) for r in range(c, q)]


def theta_to_lambda(theta: ArrayLike, q: int) -> np.ndarray:
    lam = np.zeros((q, q))
    for k, (r, c) in enumerate(_theta_index(q)):
        lam[r, c] = theta[k]
    return lam


def _starts(core, start_params, n_starts):
    """Starting values for theta.

    The first is lme4's: the identity relative covariance factor. Extra starts
    cost almost nothing (each fit is milliseconds) and remove the boundary cases
    where a single start stalls -- exactly the situation in which statsmodels
    retries bfgs, then lbfgs, then cg, and gives up.

    Diagonal and off-diagonal entries are perturbed differently, and they have
    to be. The default theta is the identity, so its off-diagonals are exactly
    zero, and a purely *multiplicative* perturbation -- which is what this did
    -- leaves every one of them at zero no matter how many starts are drawn. No
    start ever explored a correlated random-effects structure, which is the one
    case where a second start is most likely to help. Off-diagonals are
    therefore perturbed additively, and both signs are reachable: the
    correlation between a random intercept and a random slope is negative at
    least as often as it is positive.

    This is a heuristic search and not a certificate. Nothing here proves the
    result is the global optimum of a criterion that genuinely can be
    multimodal; see docs/LIMITATIONS.md.
    """
    base = np.asarray(core.default_theta(), float)
    out = []
    if start_params is not None:
        out.append(np.asarray(start_params, float).ravel())
    out.append(base)
    if n_starts > 1:
        rng = np.random.default_rng(0)
        lower = np.asarray(core.lower_bounds(), float)
        bounded = np.isfinite(lower)          # the diagonal entries
        for _ in range(n_starts - 1):
            cand = base.copy()
            cand[bounded] = np.maximum(
                base[bounded] * rng.uniform(0.25, 3.0, int(bounded.sum())), 1e-3)
            free = ~bounded
            if free.any():
                cand[free] = base[free] + rng.uniform(-0.6, 0.6, int(free.sum()))
            out.append(cand)
    return out


def _profiled_hessian(core, theta, reml, h=1e-5):
    """Hessian of the profiled criterion, by central-differencing the analytic
    gradient. Cheap: theta has 1, 3 or 6 entries.

    Used only for standard errors of the variance components. Differencing an
    analytic gradient is far more accurate than differencing the objective.
    """
    theta = np.asarray(theta, float)
    k = theta.size
    H = np.zeros((k, k))
    for j in range(k):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        _, gp = core.deviance_grad(list(tp), reml)
        _, gm = core.deviance_grad(list(tm), reml)
        H[:, j] = (np.asarray(gp) - np.asarray(gm)) / (2 * h)
    return 0.5 * (H + H.T)


def _cov_re_jacobian(theta, q):
    """d vech(Lambda Lambda') / d theta, for the delta method."""
    tix = _theta_index(q)
    k = len(tix)
    vech = [(a, b) for b in range(q) for a in range(b, q)]
    J = np.zeros((len(vech), k))
    lam = theta_to_lambda(theta, q)
    for kk, (r, c) in enumerate(tix):
        D = np.zeros((q, q))
        D[r, c] = 1.0
        dS = D @ lam.T + lam @ D.T
        for vi, (a, b) in enumerate(vech):
            J[vi, kk] = dS[a, b]
    return J, vech


def _condition_design(y, X):
    """Put the fixed-effect system on a well-conditioned, well-centred footing.

    Two exact reparameterisations, applied before any cross-product is formed:

    **Column scaling.** `X -> X D^-1`, `beta -> D beta`. The criterion surface is
    identical; only the coordinates move.

    **Response offset.** The profiled criterion depends on `y` only through

        pwrss = min over beta, u of ||y - X beta - Z Lambda u||^2 + ||u||^2,

    and replacing `y` by `y - X c` merely shifts the minimising `beta` by `-c`.
    The minimum itself -- and therefore the deviance, `sigma^2`, `theta` and the
    random effects -- is unchanged for *any* `c`. Choosing `c` to be the OLS
    coefficients makes the working response the OLS residual, which matters
    because the core evaluates

        pwrss = y'y - beta'X'y - u'Lambda'Z'y,

    a difference of large nearly-equal quantities. On raw `y` that cancellation
    is catastrophic: with a response around 1e8 -- prices in minor units, epoch
    timestamps, populations -- `y'y` is around 1e18, a double resolves it to
    about 256, and a residual sum of squares of order 1 is entirely noise. The
    fit then returns a confidently converged wrong answer.

    After the offset, `X'y` is zero to rounding and `y'y` is the OLS residual
    sum of squares, which is the same order as `pwrss` itself. What remains is
    the precision the data actually carries: a response stored as `1e8 + O(1)`
    only determines its own residuals to about 2e-8 absolute, and no
    rearrangement of the arithmetic can recover digits that are not in the
    input. `lme4` sits at the same limit for the same reason.

    Returns `(y_work, X_work, beta_offset, xscale)` with

        beta_original = (beta_offset + beta_fitted) / xscale.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    xscale = np.sqrt(np.mean(np.square(X), axis=0))
    xscale[~np.isfinite(xscale) | (xscale <= 0)] = 1.0
    Xs = X if np.all(xscale == 1.0) else np.ascontiguousarray(X / xscale)

    # lstsq is rank-revealing, so the rank check below is free. `rcond` is set
    # explicitly rather than left to the machine-precision default: the columns
    # have already been scaled to unit RMS, so a surviving singular-value ratio
    # below 1e-8 means two predictors agree to eight digits. The normal
    # equations square that, which is past what a double can represent, and the
    # penalised system's Cholesky then fails somewhere inside the optimiser --
    # surfacing as "theta is infeasible", an error that names the wrong thing
    # entirely. Catching it here says what is actually wrong.
    RCOND = 1e-8
    beta0, _, rank, svals = np.linalg.lstsq(Xs, y, rcond=RCOND)
    if rank < Xs.shape[1]:
        ratio = (svals[-1] / svals[0]) if svals.size and svals[0] > 0 else 0.0
        # An exactly duplicated column still leaves a singular value around
        # 1e-17 rather than a clean zero, so "exact" is a tolerance, not a test
        # for equality.
        exact = ratio < 1e-14
        raise ValueError(
            f"the fixed-effects design matrix is "
            f"{'rank deficient' if exact else 'numerically rank deficient'}: "
            f"rank {rank} of {Xs.shape[1]} columns"
            + ("." if exact else
               f", with a singular-value ratio of {ratio:.2e} after scaling "
               f"the columns to unit norm.")
            + " Some predictor is "
            + ("an exact linear combination of the others -- a duplicated "
               "column, a redundant categorical coding, or a constant term "
               "alongside the intercept."
               if exact else
               "a linear combination of the others to within eight digits, so "
               "their separate coefficients are not determined by the data.")
            + " Drop or combine the redundant column; the model as written "
              "does not identify its own fixed effects.")

    y_work = y - Xs @ beta0
    return np.ascontiguousarray(y_work), Xs, beta0, xscale


def _column_scales(Z):
    """Root-mean-square norm of each random-effects column, for conditioning.

    Poorly scaled predictors are the commonest real-world cause of a mixed model
    failing to converge: with a slope variable measured in the hundreds and an
    intercept column of ones, the two variance components live on scales that
    differ by four orders of magnitude, and a single starting value cannot serve
    both.
    """
    d = np.sqrt(np.mean(np.square(Z), axis=0))
    d[~np.isfinite(d) | (d <= 0)] = 1.0
    return d


def fit_core(
    y: ArrayLike,
    X: ArrayLike,
    Z: ArrayLike,
    codes: ArrayLike,
    n_groups: int,
    reml: bool = True,
    start_params: ArrayLike | None = None,
    method: str | None = None,
    n_starts: int | None = None,
    maxiter: int = 500,
    gtol: float = 1e-8,
    ftol: float = 1e-12,
    want_se_re: bool = False,
) -> dict[str, Any]:
    """Fit one grouping factor and return every derived quantity.

    Returns a plain dict; the estimator classes wrap it.

    Three exact reparameterisations are applied before anything is computed,
    and undone before anything is returned, so none of them is visible to the
    caller:

    1. **Random-effects columns to unit RMS.** Substituting `Z -> Z D^-1` and
       `Lambda -> D Lambda` leaves `Lambda' Z'Z Lambda + I` unchanged, so the
       criterion surface is identical and only the coordinates move. It makes
       `theta = I` a sensible start whatever units the predictors are in.
    2. **Fixed-effect columns to unit RMS**, for the conditioning of the
       fixed-effect solve. This one *does* shift the REML criterion, by
       `2 * sum(log d)` through `log|X'V^-1 X|`; the constant is restored below.
    3. **Response offset by its OLS fit**, which is what makes the criterion
       computable at all when the response is far from zero. See
       `_condition_design`.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)

    # ---- Identifiability of the variance split.
    #
    # The earlier version of this check asserted that `n <= q * m` means every
    # group is interpolated, the residual variance is driven to zero and the
    # profiled likelihood *diverges*. That is not true, and the counterexample
    # is the simplest possible case: one random intercept per singleton
    # observation gives V = (tau^2 + sigma^2) I, so the criterion is finite and
    # exactly *constant* along the ridge that trades tau^2 against sigma^2 --
    # flat, not divergent. Conversely the count rule misses real
    # non-identifiability, such as a single group whose random intercept is
    # perfectly confounded with the fixed intercept.
    #
    # So the structural count is used only to decide *where to look*, and the
    # claim itself is settled empirically below, by asking the criterion.
    n_obs, q_re = Z.shape
    n_groups = int(n_groups)
    suspect = (n_obs <= q_re * n_groups) or (n_groups == 1)

    dscale = _column_scales(Z)
    # When the columns are already on comparable scales -- which includes the
    # very common intercept-only and standardised-predictor cases -- the
    # rescaling is a no-op and copying an n x q array to perform it is not free
    # at 500,000 rows. Only divide when it actually changes something.
    if np.all(dscale == 1.0):
        Zs = Z
    else:
        Zs = np.ascontiguousarray(Z / dscale)

    y_work, Xs, beta_offset, xscale = _condition_design(y, X)

    core = LmmCore(y_work, Xs, Zs,
                   np.ascontiguousarray(codes, dtype=np.int64),
                   int(n_groups))

    q = core.q
    use_rust = str(method).lower() == "rust"

    # The number of starts is a property of the optimiser, not a taste.
    #
    # scipy's L-BFGS-B needs one: across the 120 randomised fuzz fixtures,
    # n_starts of 1, 2 and 3 give identical outcomes (52 same optimum, 42
    # better than statsmodels, 26 where the reference does not converge, 0
    # worse) at 54.5, 59.1 and 69.4 mean objective evaluations. Extra starts
    # changed no answer.
    #
    # The in-crate projected L-BFGS needs more, and this is measured too: with
    # a single start it settles on a strictly worse *stationary* point on 2 of
    # 20 fuzz seeds -- one of them reported as converged, which it legitimately
    # is, being a local optimum. A weaker line search finds worse basins, and
    # no amount of certification fixes that, because the certificate is local.
    # Five starts recover the scipy answer on both.
    if n_starts is None:
        n_starts = 5 if use_rust else 1

    lower = np.asarray(core.lower_bounds(), float)
    bounds = [(0.0, None) if np.isfinite(v) else (None, None) for v in lower]
    tix = _theta_index(q)
    diag_k = [k for k, (r, c) in enumerate(tix) if r == c]
    state = {"nfev": 0, "nit": 0}

    class _Res:
        """The bit of a scipy OptimizeResult the driver below actually uses."""

        __slots__ = ("fun", "message", "nit", "success", "x")

        def __init__(self, x, fun, success, message, nit):
            self.x, self.fun = np.asarray(x, float), float(fun)
            self.success, self.message, self.nit = bool(success), str(message), int(nit)

    if use_rust:
        def run(th0):
            sol = core.fit(list(np.asarray(th0, float)), reml, maxiter, gtol, ftol)
            state["nfev"] += int(sol["fev"])
            state["nit"] += int(sol["iterations"])
            return _Res(np.asarray(sol["theta"], float), float(sol["deviance"]),
                        bool(sol["converged"]), str(sol["message"]),
                        int(sol["iterations"]))
    else:
        def obj(t):
            f, g = core.deviance_grad(list(t), reml)
            if not np.isfinite(f):
                # A huge finite value with a zero gradient looks stationary to
                # the optimiser, which is how an infeasible start used to be
                # reported as a successful fit. Point the gradient back towards
                # the feasible region instead, and never call this a success.
                return 1e300, -np.asarray(t, float)
            return f, np.asarray(g, float)

        def run(th0):
            res = minimize(obj, np.asarray(th0, float), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"ftol": ftol, "gtol": gtol,
                                    "maxiter": maxiter})
            state["nfev"] += int(res.nfev)
            state["nit"] += int(res.nit)
            return _Res(res.x, res.fun, res.success, res.message, res.nit)

    # ---- Optimise, and compare every start that is run.
    #
    # This is a heuristic search, not a certificate: nothing here proves the
    # result is the global optimum of a criterion that can genuinely be
    # multimodal. What makes it trustworthy is the stationarity check below,
    # which verifies the stopping point and *restarts* when it fails.
    # docs/LIMITATIONS.md states the distinction.
    #
    # `n_starts` defaults to 1, and that is a measured choice rather than a
    # cost-saving one. Across the 120 randomised fuzz fixtures, n_starts of 1,
    # 2 and 3 give *identical* outcomes -- 52 same optimum, 42 better than
    # statsmodels, 26 where the reference does not converge, 0 worse -- at 54.5,
    # 59.1 and 69.4 mean objective evaluations. The extra starts changed no
    # answer on any fixture. What does the work is the boundary escape below
    # and the certified retry after it, and both are targeted: they cost
    # nothing at all until something is actually wrong.
    if start_params is not None:
        # A caller's theta describes Lambda in the *data* coordinates. The core
        # sees Z D^-1, where the matching factor is D Lambda, so scale the rows
        # of the packed lower triangle by D before handing it over. Without
        # this a supplied start silently meant a different covariance than the
        # one the caller wrote.
        lam0 = theta_to_lambda(np.asarray(start_params, float), q) * dscale[:, None]
        start_params = np.array([lam0[r, c] for c in range(q) for r in range(c, q)])

    all_starts = _starts(core, start_params, max(1, n_starts))
    best = run(all_starts[0])
    # Every start that is run is *compared*. An earlier version ran the extra
    # starts only after a reported failure, so a confident stop at a worse
    # interior optimum was never challenged -- the one case where a second
    # start would have helped most.
    for th0 in all_starts[1:]:
        res = run(th0)
        if res.fun < best.fun - 1e-10:
            best = res

    # ---- Boundary escape.
    #
    # At Lambda = 0 every term of the gradient vanishes identically: M = 0,
    # so d(ldL2) = 0; u = 0, so d(pwrss) = 0; B = 0, so d(ldRX2) = 0. theta = 0
    # is therefore a *stationary point of the profiled criterion whatever the
    # data*, and a gradient-based optimiser that reaches it stops there and
    # reports a zero projected gradient -- even when the true optimum has a
    # perfectly ordinary non-zero variance.
    #
    # Singular fits are real and must be reportable (Dyestuff2 genuinely has
    # a between-batch variance of zero), so we cannot just forbid the bound.
    # Instead, whenever a diagonal entry lands on it, probe a few positive
    # values along that coordinate and re-optimise if any of them is better.
    # Fits are milliseconds, so this costs almost nothing.
    # Each trial point gets its own full re-optimisation. Two cheaper
    # variants were tried and both regressed quality on the fuzz suite:
    # evaluating the four trials and re-optimising only from one that beats
    # the incumbent gave 5 of 120 fixtures a worse optimum, and re-optimising
    # only from the lowest-valued trial still gave 4. Near the bound the
    # criterion can be *higher* at a probe point and still lead downhill into
    # a better basin, so the probe cannot be used to choose between trials.
    #
    # The ladder reaches down to 1e-3 because a variance component can be
    # genuinely small rather than zero. With trials starting at 0.05, a true
    # optimum at theta = 0.028 was missed: every probe overshot it, and the
    # optimiser slid back into the stationary point at the bound. Found by the
    # wider stress sweep, not by the committed fuzz suite.
    #
    # This costs nothing on well-behaved data: the escape only runs when a
    # variance component actually lands on the bound, and on clean fixtures a
    # whole fit is still 10-11 objective evaluations.
    for _ in range(3):
        at_bound = [k for k in diag_k if best.x[k] <= lower[k] + 1e-10]
        if not at_bound:
            break
        improved = False
        for k in at_bound:
            for trial in (1e-3, 1e-2, 0.05, 0.2, 0.6, 1.5):
                cand = np.array(best.x, float)
                cand[k] = trial
                res = run(cand)
                if res.fun < best.fun - 1e-10:
                    best = res
                    improved = True
        if not improved:
            break

    # ---- Certify, and retry while the certificate fails.
    #
    # Stationarity is the acceptance test, so it drives the search rather than
    # merely annotating its result. If the incumbent is not stationary, restart
    # from perturbed points and keep the best certified one. This matters most
    # for method="rust", whose line search is weaker than scipy's: it used to
    # stop early at a point tens of deviance units worse and, before the flag
    # was fixed, report success there.
    gtol_abs = 1e-5 * max(1.0, float(n_obs) - (float(X.shape[1]) if reml else 0.0))

    def certify(th):
        """Evaluate at `th` and say whether it is a stationary point.

        Returns `(None, inf, False)` for an infeasible `th`. The optimiser can
        legitimately finish on one -- the objective reports 1e300 there rather
        than raising, so L-BFGS-B may stop at the edge of the feasible region on
        a near-singular design. Calling `solution()` on such a point raises
        `RuntimeError: theta is infeasible`, which used to escape `fit()` as an
        opaque error naming the wrong thing: the design is the problem, not
        theta.
        """
        th = np.asarray(th, float)
        try:
            sol = core.solution(list(th), reml)
        except RuntimeError:
            return None, float("inf"), False
        grad = np.asarray(sol["grad"], float)
        if not np.all(np.isfinite(grad)):
            return None, float("inf"), False
        pg = grad.copy()
        on_bound = np.isfinite(lower) & (th <= lower + 1e-12) & (grad > 0)
        pg[on_bound] = 0.0
        gn = float(np.max(np.abs(pg))) if pg.size else 0.0
        return sol, gn, gn <= gtol_abs

    sol, grad_norm, stationary = certify(best.x)
    if sol is None:
        # The incumbent is infeasible, so there is nothing to certify and
        # nothing to report. Retreat to points that are feasible by
        # construction -- Lambda = I, then progressively smaller diagonals,
        # ending at Lambda = 0, where A = I and the factorisation cannot fail.
        for fallback in (core.default_theta(), None, None, None):
            if fallback is None:
                continue
            cand = np.asarray(fallback, float)
            res = run(cand)
            sol, grad_norm, stationary = certify(res.x)
            if sol is not None:
                best = res
                break
        if sol is None:
            for scale_down in (0.1, 0.01, 0.0):
                cand = np.asarray(core.default_theta(), float) * scale_down
                sol, grad_norm, stationary = certify(cand)
                if sol is not None:
                    best = _Res(cand, float(sol["deviance"]), False,
                                "retreated to a feasible point", 0)
                    break
        if sol is None:
            raise ValueError(
                "the model could not be evaluated anywhere in the parameter "
                "space: the penalised system is singular even at a zero "
                "random-effects covariance. This is a design problem rather "
                "than an optimisation one -- most often a fixed-effect design "
                "that is collinear to within machine precision.")

    if not stationary:
        rng = np.random.default_rng(0)
        off_diag = [k for k in range(len(lower)) if k not in diag_k]
        for attempt in range(4):
            cand = np.array(best.x, float)
            # Diagonal entries are scale parameters, so perturb them
            # multiplicatively and keep them inside their bound. Off-diagonals
            # are correlations that are frequently exactly zero, where a
            # multiplicative perturbation is the identity -- they get an
            # additive kick instead, in both directions.
            spread = 0.5 + 0.5 * attempt
            for k in diag_k:
                cand[k] = max(cand[k] * np.exp(rng.normal(scale=spread)), 1e-3)
            for k in off_diag:
                cand[k] += rng.normal(scale=spread)
            res = run(cand)
            s_new, gn_new, ok_new = certify(res.x)
            if s_new is None:            # infeasible: not a candidate at all
                continue

            # Never trade away likelihood for a certificate. A strictly better
            # criterion is always taken; a certified point is taken over an
            # uncertified incumbent only when it does not cost anything.
            # Preferring "certified" outright would let a worse local optimum
            # be reported as the answer purely because it was stationary.
            take = (res.fun < best.fun - 1e-10
                    or (ok_new and not stationary
                        and res.fun <= best.fun + 1e-10))
            if take:
                best, sol, grad_norm, stationary = res, s_new, gn_new, ok_new
            if stationary:
                break

    if suspect:
        # Is the criterion actually flat along theta here? A ridge on which the
        # deviance does not change is a variance split the data cannot resolve:
        # every point on it fits identically, so the reported split is one
        # arbitrary point and means nothing. Two probes, only on designs the
        # structural check already flagged.
        d0 = float(core.deviance(list(best.x), reml))
        flat = True
        for factor in (4.0, 0.25):
            probe = np.array(best.x, float)
            probe[diag_k] = np.maximum(probe[diag_k] * factor, 1e-3)
            dp = float(core.deviance(list(probe), reml))
            if not np.isfinite(dp) or abs(dp - d0) > 1e-9 * max(1.0, abs(d0)):
                flat = False
                break
        if flat:
            warnings.warn(
                f"the variance split is not identifiable: {n_obs} observations "
                f"against {q_re * n_groups} random effects ({q_re} per group x "
                f"{n_groups} groups). The profiled criterion is flat in the "
                "random-effects variance -- every split between it and the "
                "residual variance fits the data equally well -- so the "
                "reported split is one arbitrary point on that ridge. The "
                "fixed effects and the total variance are still estimated; the "
                "decomposition is not. Use fewer random-effect terms, or more "
                "observations per group.",
                ConvergenceWarning, stacklevel=3)

    theta = np.asarray(best.x, float)
    converged = bool(stationary)
    message = str(best.message)
    nit = state["nit"]
    nfev = state["nfev"]

    # Stationarity is the *only* convergence test: an optimiser's own success
    # flag reports that its termination rule fired, not that the point is
    # stationary. L-BFGS-B with a loose ftol, and the in-Rust optimiser's
    # relative-change rule, both report success at points with a projected
    # gradient in the tens. Taking `converged or stationary` -- as this once
    # did -- let that flag overrule the gradient, which is exactly the failure
    # mode the package exists to prevent.
    #
    # The tolerance is invariant to things that do not change the problem. The
    # deviance is not such a thing: rescaling the response, or the fixed-effect
    # columns under REML, adds a constant to the criterion while leaving its
    # theta-gradient identical, so `1e-6 * abs(deviance)` would accept or
    # reject the same stationary point depending on the units. It is scaled by
    # dfree instead, which is what the criterion's theta-dependence is measured
    # in.
    if not converged:
        warnings.warn(
            f"optimisation did not reach a stationary point "
            f"(max |projected gradient| = {grad_norm:.3g}, tolerance "
            f"{gtol_abs:.3g}); {message}",
            ConvergenceWarning, stacklevel=3)

    # A variance component sitting exactly on its bound is a *singular* fit: a
    # legitimate boundary optimum, but one where the usual Wald intervals for
    # the variance parameters do not apply, because the estimate is on the edge
    # of the parameter space and its sampling distribution is not normal. lme4
    # reports this separately from convergence (`isSingular`), and so do we --
    # conflating the two is what drives people to delete random-effect terms.
    tix_all = _theta_index(q)
    singular = bool(any(
        theta[k] <= lower[k] + 1e-10
        for k, (r, c) in enumerate(tix_all) if r == c and np.isfinite(lower[k])))

    # Undo the internal column scaling. With Z -> Z D^-1 the fitted random
    # effects and their covariance are on the scaled coordinates: b = D^-1 b~
    # and cov(b) = D^-1 cov(b~) D^-1. beta, sigma^2 and the deviance are
    # invariant, because the reparameterisation leaves the criterion identical.
    Dinv = 1.0 / dscale
    cov_re = np.asarray(sol["cov_re"], float) * np.outer(Dinv, Dinv)
    re_modes = np.asarray(sol["random_effects"], float) * Dinv

    # Undo the fixed-effect conditioning. The response offset shifts beta back
    # by the OLS coefficients it removed; the column scaling divides through.
    beta = (np.asarray(sol["beta"], float) + beta_offset) / xscale
    cov_beta = np.asarray(sol["cov_beta"], float) / np.outer(xscale, xscale)

    # The REML criterion is *not* invariant to the column scaling, because it
    # carries log|X'V^-1 X|: with X -> X D^-1 that determinant loses a factor
    # det(D)^2, so the criterion is short by 2 * sum(log d). theta, sigma^2 and
    # the random effects are unaffected, and the ML criterion has no such term.
    # Restoring the constant is what keeps llf comparable with lme4 and
    # statsmodels, and keeps REML deviances comparable between models fitted
    # here in different units.
    deviance = float(sol["deviance"])
    ldrx2_shift = 2.0 * float(np.sum(np.log(xscale))) if reml else 0.0
    deviance += ldrx2_shift

    out = {
        "core": core,
        "theta": theta,
        "scale_factors": dscale,
        "converged": converged,
        "singular": singular,
        "message": message,
        "grad_norm": grad_norm,
        "nfev": nfev,
        "nit": nit,
        "deviance": deviance,
        "beta": beta,
        "cov_beta": cov_beta,
        "cov_re": cov_re,
        "random_effects": re_modes,
        "u": np.asarray(sol["u"], float),
        "scale": float(sol["sigma2"]),
        "sigma": float(sol["sigma"]),
        "pwrss": float(sol["pwrss"]),
        "reml": bool(reml),
        "n": int(sol["n"]), "p": int(sol["p"]),
        "q": int(sol["q"]), "m": int(sol["m"]),
        "lambda": theta_to_lambda(theta, q) * Dinv[:, None],
    }

    # cov_re_unscaled = cov_re / scale  (statsmodels' parameterisation)
    out["cov_re_unscaled"] = cov_re / out["scale"]

    # Standard errors for the variance components, via the delta method on the
    # profiled Hessian. lme4 declines to report these at all; statsmodels does,
    # so a drop-in has to. They are documented as coming from the profiled
    # parameterisation and may differ slightly from statsmodels' own.
    #
    # This is deferred rather than computed here: the Hessian costs 2 * n_theta
    # extra gradient evaluations -- about 28% of a whole fit -- and many callers
    # only ever look at the fixed effects. MixedLMResults calls it on first
    # access to bse_re and caches the result.
    def _compute_bse_re():
        """Delta-method standard errors for the variance components.

        The Hessian is *checked*, not merely inverted. At a genuine interior
        minimum the profiled criterion's Hessian is positive definite, and its
        inverse is a covariance. At a boundary optimum -- where a variance
        component sits at zero, which is common and legitimate -- it need not
        be, and inverting an indefinite matrix there yields numbers that look
        like standard errors and are not. Those entries come back NaN.

        A near-singular Hessian is treated the same way: it means the criterion
        is nearly flat in some direction, so the corresponding variance
        parameter is barely determined and reporting a small standard error for
        it would be backwards.
        """
        if q == 0:
            return None
        nan_matrix = np.full((q, q), np.nan)
        try:
            H = _profiled_hessian(core, theta, reml)
            if not np.all(np.isfinite(H)):
                return nan_matrix

            # Symmetrise: H comes from differencing a gradient, so the two
            # off-diagonal estimates differ by rounding.
            H = 0.5 * (H + H.T)
            evals = np.linalg.eigvalsh(H)
            # Positive definite, and not so ill-conditioned that the inverse is
            # noise. 1e-10 relative is generous -- it only rejects directions in
            # which the criterion is flat to ten digits.
            if evals[0] <= 0 or evals[0] <= 1e-10 * evals[-1]:
                return nan_matrix

            # deviance = -2 logL, so the observed information is H/2.
            cov_theta = 2.0 * np.linalg.inv(H)
            J, vech = _cov_re_jacobian(theta, q)
            # The Jacobian is in the internally scaled coordinates; map it back,
            # since cov_re_unscaled[a, b] carries a factor Dinv[a] * Dinv[b].
            for vi, (a, b) in enumerate(vech):
                J[vi, :] *= Dinv[a] * Dinv[b]
            cov_vech = J @ cov_theta @ J.T
            dg = np.diag(cov_vech)
            se = np.sqrt(np.where(dg > 0, dg, np.nan))
            M = nan_matrix.copy()
            for vi, (a, b) in enumerate(vech):
                M[a, b] = M[b, a] = se[vi]
            return M
        except (np.linalg.LinAlgError, ValueError):
            return nan_matrix

    out["compute_bse_re"] = _compute_bse_re
    out["bse_re_unscaled"] = _compute_bse_re() if want_se_re else None

    return out
