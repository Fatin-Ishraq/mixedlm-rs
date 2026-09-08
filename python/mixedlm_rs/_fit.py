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

import numpy as np
from scipy.optimize import minimize

from ._mixedlm_rs import LmmCore

__all__ = ["fit_core", "ConvergenceWarning"]


class ConvergenceWarning(UserWarning):
    """Raised when the optimiser cannot certify a stationary point."""


def _theta_index(q):
    return [(r, c) for c in range(q) for r in range(c, q)]


def theta_to_lambda(theta, q):
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
    """
    base = np.asarray(core.default_theta(), float)
    out = []
    if start_params is not None:
        out.append(np.asarray(start_params, float).ravel())
    out.append(base)
    if n_starts > 1:
        rng = np.random.default_rng(0)
        lower = np.asarray(core.lower_bounds(), float)
        for _ in range(n_starts - 1):
            cand = base * rng.uniform(0.25, 3.0, base.size)
            cand[np.isfinite(lower)] = np.maximum(cand[np.isfinite(lower)], 1e-3)
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


def fit_core(y, X, Z, codes, n_groups, reml=True, start_params=None,
             method=None, n_starts=3, maxiter=500, gtol=1e-8, ftol=1e-12,
             want_se_re=True):
    """Fit one grouping factor and return every derived quantity.

    Returns a plain dict; the estimator classes wrap it.

    The random-effects design is internally rescaled to unit column RMS. This is
    an *exact* reparameterisation, not an approximation: substituting Z -> Z D^-1
    and Lambda -> D Lambda leaves Lambda' Z'Z Lambda + I unchanged, so the whole
    criterion surface is identical and only the coordinates move. What it buys is
    conditioning -- theta = I becomes a sensible starting point whatever units
    the predictors are in. Results are transformed back before being returned,
    so nothing about this is visible to the caller.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    dscale = _column_scales(Z)
    Zs = np.ascontiguousarray(Z / dscale)

    core = LmmCore(np.ascontiguousarray(y, dtype=np.float64),
                   np.ascontiguousarray(X, dtype=np.float64),
                   Zs,
                   np.ascontiguousarray(codes, dtype=np.int64),
                   int(n_groups))

    q = core.q
    use_rust = str(method).lower() == "rust"

    if use_rust:
        flat = np.concatenate(_starts(core, start_params, n_starts))
        sol = core.fit(list(flat), reml, maxiter, gtol, ftol)
        theta = np.asarray(sol["theta"], float)
        converged = bool(sol["converged"])
        message = str(sol["message"])
        nfev = int(sol["fev"])
        nit = int(sol["iterations"])
    else:
        lower = np.asarray(core.lower_bounds(), float)
        bounds = [(0.0, None) if np.isfinite(v) else (None, None) for v in lower]
        tix = _theta_index(q)
        diag_k = [k for k, (r, c) in enumerate(tix) if r == c]

        def obj(t):
            f, g = core.deviance_grad(list(t), reml)
            if not np.isfinite(f):
                return 1e300, np.zeros(len(t))
            return f, np.asarray(g, float)

        state = {"nfev": 0}

        def run(th0):
            res = minimize(obj, np.asarray(th0, float), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options={"ftol": ftol, "gtol": gtol,
                                    "maxiter": maxiter})
            state["nfev"] += int(res.nfev)
            return res

        # ---- Optimise from the primary start.
        #
        # Extra starts are tried only if the first one fails to converge. An
        # unconditional multi-start was measured to be pure overhead: across the
        # 120 randomised fuzz fixtures, n_starts=1 and n_starts=3 produce
        # *identical* outcomes (0 worse, 19 better, 26 where the reference does
        # not converge) while n_starts=3 costs 68.4 mean objective evaluations
        # against 47.5. The robustness comes from the boundary escape below, not
        # from the extra starts.
        all_starts = _starts(core, start_params, max(1, n_starts))
        best = run(all_starts[0])
        if not best.success:
            for th0 in all_starts[1:]:
                res = run(th0)
                if res.fun < best.fun:
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
        # This costs nothing on well-behaved data: the escape only runs when a
        # variance component actually lands on the bound, and on clean fixtures a
        # whole fit is still 10-11 objective evaluations.
        for _ in range(3):
            at_bound = [k for k in diag_k if best.x[k] <= lower[k] + 1e-10]
            if not at_bound:
                break
            improved = False
            for k in at_bound:
                for trial in (0.05, 0.2, 0.6, 1.5):
                    cand = np.array(best.x, float)
                    cand[k] = trial
                    res = run(cand)
                    if res.fun < best.fun - 1e-10:
                        best = res
                        improved = True
            if not improved:
                break

        theta = np.asarray(best.x, float)
        converged = bool(best.success)
        message = str(best.message)
        nit = int(best.nit)
        nfev = state["nfev"]

    sol = core.solution(list(theta), reml)

    # Certify stationarity ourselves rather than trusting the optimiser's flag:
    # a component pinned at its lower bound with the gradient pushing further
    # into the bound is stationary, not a failure.
    grad = np.asarray(sol["grad"], float)
    lower = np.asarray(core.lower_bounds(), float)
    pg = grad.copy()
    at_bound = np.isfinite(lower) & (theta <= lower + 1e-12) & (grad > 0)
    pg[at_bound] = 0.0
    grad_norm = float(np.max(np.abs(pg))) if pg.size else 0.0
    # Scale-free: the criterion is on a deviance scale, so compare relatively.
    stationary = grad_norm <= max(1e-4, 1e-6 * abs(float(sol["deviance"])))
    converged = bool(converged or stationary)
    if not converged:
        warnings.warn(
            f"optimisation did not reach a stationary point "
            f"(max |projected gradient| = {grad_norm:.3g}); {message}",
            ConvergenceWarning, stacklevel=3)

    # Undo the internal column scaling. With Z -> Z D^-1 the fitted random
    # effects and their covariance are on the scaled coordinates: b = D^-1 b~
    # and cov(b) = D^-1 cov(b~) D^-1. beta, sigma^2 and the deviance are
    # invariant, because the reparameterisation leaves the criterion identical.
    Dinv = 1.0 / dscale
    cov_re = np.asarray(sol["cov_re"], float) * np.outer(Dinv, Dinv)
    re_modes = np.asarray(sol["random_effects"], float) * Dinv

    out = {
        "core": core,
        "theta": theta,
        "scale_factors": dscale,
        "converged": converged,
        "message": message,
        "grad_norm": grad_norm,
        "nfev": nfev,
        "nit": nit,
        "deviance": float(sol["deviance"]),
        "beta": np.asarray(sol["beta"], float),
        "cov_beta": np.asarray(sol["cov_beta"], float),
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
    out["bse_re_unscaled"] = None
    if want_se_re and q > 0:
        try:
            H = _profiled_hessian(core, theta, reml)
            # deviance = -2 logL, so the observed information is H/2.
            cov_theta = 2.0 * np.linalg.inv(H)
            J, vech = _cov_re_jacobian(theta, q)
            # The Jacobian is in the internally scaled coordinates; map it back,
            # since cov_re_unscaled[a, b] carries a factor Dinv[a] * Dinv[b].
            for vi, (a, b) in enumerate(vech):
                J[vi, :] *= Dinv[a] * Dinv[b]
            cov_vech = J @ cov_theta @ J.T
            d = np.diag(cov_vech)
            se = np.sqrt(np.where(d > 0, d, np.nan))
            M = np.full((q, q), np.nan)
            for vi, (a, b) in enumerate(vech):
                M[a, b] = M[b, a] = se[vi]
            out["bse_re_unscaled"] = M
        except (np.linalg.LinAlgError, ValueError):
            out["bse_re_unscaled"] = np.full((q, q), np.nan)

    return out
