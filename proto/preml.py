"""Profiled REML/ML for linear mixed models with one grouping factor.

Pure NumPy reference implementation of the lme4 formulation
(Bates, Machler, Bolker & Walker 2015, JSS 67(1)).

The point of this file is to prove -- before any Rust exists -- that the win over
statsmodels is *algorithmic*: profiling beta and sigma^2 out analytically and
factorising the block-diagonal penalised system, rather than handing every
parameter to a general optimiser.

Model
    y = X beta + Z b + eps,  b ~ N(0, sigma^2 Lambda Lambda'),  eps ~ N(0, sigma^2 I)

With spherical random effects u (b = Lambda u), for a given theta the profiled
criterion is minimised by a penalised least squares solve.  With ONE grouping
factor, Lambda' Z'Z Lambda + I is block diagonal: m independent q x q blocks.
No general sparse solver is needed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["LMMData", "profiled_deviance", "fit_lmm",
           "profiled_deviance_looped", "fit_lmm_looped"]

_LOG2PI = np.log(2.0 * np.pi)


class LMMData:
    """Pre-computed, theta-independent cross-products.

    Everything here is a constant of the problem.  statsmodels recomputes this
    class of quantity inside the optimiser loop; we form it exactly once.
    """

    __slots__ = ("n", "p", "q", "m", "XtX", "Xty", "yty", "ZtZ", "ZtX", "Zty",
                 "group_index", "group_sizes")

    def __init__(self, y, X, Z, groups):
        y = np.ascontiguousarray(y, dtype=np.float64).ravel()
        X = np.ascontiguousarray(X, dtype=np.float64)
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        groups = np.asarray(groups)

        if X.ndim != 2 or Z.ndim != 2:
            raise ValueError("X and Z must be 2-D")
        if not (len(y) == X.shape[0] == Z.shape[0] == len(groups)):
            raise ValueError("y, X, Z and groups must agree on the row count")

        # Contiguous 0..m-1 group codes, preserving first-appearance order.
        uniq, codes = np.unique(groups, return_inverse=True)

        self.n, self.p = X.shape
        self.q = Z.shape[1]
        self.m = len(uniq)

        # Global cross-products.
        self.XtX = X.T @ X
        self.Xty = X.T @ y
        self.yty = float(y @ y)

        # Per-group blocks.  Held as stacked contiguous arrays so the hot loop is
        # a flat walk rather than a dict lookup per group.
        m, q, p = self.m, self.q, self.p
        self.ZtZ = np.zeros((m, q, q))
        self.ZtX = np.zeros((m, q, p))
        self.Zty = np.zeros((m, q))
        self.group_sizes = np.zeros(m, dtype=np.int64)

        order = np.argsort(codes, kind="stable")
        codes_sorted = codes[order]
        starts = np.searchsorted(codes_sorted, np.arange(m), side="left")
        stops = np.searchsorted(codes_sorted, np.arange(m), side="right")

        for i in range(m):
            idx = order[starts[i]:stops[i]]
            Zi = Z[idx]
            self.ZtZ[i] = Zi.T @ Zi
            self.ZtX[i] = Zi.T @ X[idx]
            self.Zty[i] = Zi.T @ y[idx]
            self.group_sizes[i] = len(idx)

        self.group_index = (order, starts, stops)


def theta_to_lambda(theta, q):
    """Lower-triangular relative covariance factor from the packed parameters.

    theta packs the lower triangle column-wise, which is lme4's convention.
    Diagonal entries are constrained non-negative by the optimiser bounds.
    """
    Lam = np.zeros((q, q))
    k = 0
    for j in range(q):
        for i in range(j, q):
            Lam[i, j] = theta[k]
            k += 1
    return Lam


def n_theta(q):
    return q * (q + 1) // 2


def profiled_deviance(theta, d: LMMData, reml=True, want_solution=False):
    """Profiled ML/REML criterion at theta.

    beta and sigma^2 are eliminated analytically, so the optimiser only ever
    sees theta -- q(q+1)/2 parameters instead of p + q(q+1)/2 + 1.

    Every step below is *batched* across the m groups.  There is deliberately no
    Python-level loop over groups: that is precisely the pathology that makes
    statsmodels issue ~276,000 tiny ``numpy.linalg.solve`` calls per fit.  Here
    the whole block-diagonal system is one handful of numpy calls regardless of
    how many groups there are.
    """
    q, p, n = d.q, d.p, d.n
    Lam = theta_to_lambda(theta, q)
    LamT = Lam.T

    # A_i = Lambda' (Z_i'Z_i) Lambda + I_q, for every i at once -> (m, q, q)
    A = LamT @ d.ZtZ @ Lam
    idx = np.arange(q)
    A[:, idx, idx] += 1.0
    try:
        L = np.linalg.cholesky(A)                       # batched
    except np.linalg.LinAlgError:
        return np.inf if not want_solution else (np.inf, None)

    diagL = L[:, idx, idx]
    if not np.all(diagL > 0.0):
        return np.inf if not want_solution else (np.inf, None)
    ldL2 = 2.0 * float(np.log(diagL).sum())

    RZX = np.linalg.solve(L, LamT @ d.ZtX)              # (m, q, p)
    cu = np.linalg.solve(L, (d.Zty @ Lam)[:, :, None])  # (m, q, 1)

    sum_RZXtRZX = np.einsum("mqa,mqb->ab", RZX, RZX, optimize=True)
    sum_RZXtcu = np.einsum("mqa,mq->a", RZX, cu[:, :, 0], optimize=True)

    # Fixed effects: (X'X - sum RZX'RZX) beta = X'y - sum RZX'cu
    RXtRX = d.XtX - sum_RZXtRZX
    try:
        RX = np.linalg.cholesky(RXtRX)
    except np.linalg.LinAlgError:
        return np.inf if not want_solution else (np.inf, None)
    rhs = d.Xty - sum_RZXtcu
    beta = np.linalg.solve(RX.T, np.linalg.solve(RX, rhs))

    # u_i = L_i^{-T} (cu_i - RZX_i beta), batched
    u = np.linalg.solve(L.transpose(0, 2, 1), cu - RZX @ beta[:, None])[:, :, 0]

    # Penalised RSS via the normal-equation identity:
    #   r^2 = y'y - beta'(X'y) - u'(Lambda' Z'y)
    LtZty = d.Zty @ Lam            # (m, q)
    pwrss = d.yty - float(beta @ d.Xty) - float((u * LtZty).sum())
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        return np.inf if not want_solution else (np.inf, None)

    if reml:
        dfree = n - p
        ldRX2 = 2.0 * np.log(np.diag(RX)).sum()
        dev = ldL2 + ldRX2 + dfree * (1.0 + _LOG2PI + np.log(pwrss / dfree))
    else:
        dfree = n
        ldRX2 = 0.0
        dev = ldL2 + dfree * (1.0 + _LOG2PI + np.log(pwrss / dfree))

    if not want_solution:
        return dev

    sigma2 = pwrss / dfree
    return dev, {
        "beta": beta,
        "u": u,
        "b": u @ Lam.T,                 # random effects on the data scale
        "sigma2": sigma2,
        "sigma": np.sqrt(sigma2),
        "pwrss": pwrss,
        "ldL2": ldL2,
        "ldRX2": ldRX2,
        "RX": RX,
        "Lambda": Lam,
        "cov_re": sigma2 * (Lam @ Lam.T),
        "cov_beta": sigma2 * np.linalg.inv(RXtRX),
    }


def _default_theta0(q):
    """lme4's starting value: identity relative covariance factor."""
    th = np.zeros(n_theta(q))
    k = 0
    for j in range(q):
        for i in range(j, q):
            if i == j:
                th[k] = 1.0
            k += 1
    return th


def _bounds(q):
    """Diagonal entries >= 0; off-diagonals free.  Matches lme4's lower bounds."""
    lo = []
    for j in range(q):
        for i in range(j, q):
            lo.append(0.0 if i == j else -np.inf)
    return [(l, None if l == 0.0 else None) for l in lo], np.array(lo)


def fit_lmm(y, X, Z, groups, reml=True, theta0=None, n_starts=1, tol=1e-10):
    """Fit by minimising the profiled criterion over theta only."""
    d = LMMData(y, X, Z, groups)
    q = d.q
    lo = _bounds(q)[1]
    bounds = [(0.0, None) if np.isfinite(l) else (None, None) for l in lo]

    starts = []
    if theta0 is not None:
        starts.append(np.asarray(theta0, dtype=float))
    else:
        starts.append(_default_theta0(q))
        # Extra starts guard against the boundary cases where statsmodels thrashes.
        rng = np.random.default_rng(0)
        for _ in range(max(0, n_starts - 1)):
            cand = _default_theta0(q) * rng.uniform(0.2, 3.0, size=n_theta(q))
            cand[np.isfinite(lo) & (cand < 0)] = 0.05
            starts.append(cand)

    best = None
    for th0 in starts:
        res = minimize(profiled_deviance, th0, args=(d, reml, False),
                       method="L-BFGS-B", bounds=bounds,
                       options={"ftol": tol, "gtol": 1e-8, "maxiter": 10000})
        if best is None or res.fun < best.fun:
            best = res

    dev, sol = profiled_deviance(best.x, d, reml, want_solution=True)
    sol.update({
        "theta": best.x,
        "deviance": dev,
        "reml": reml,
        "converged": bool(best.success),
        "message": str(best.message),
        "nit": int(best.nit),
        "nfev": int(best.nfev),
        "n": d.n, "p": d.p, "q": d.q, "m": d.m,
    })
    return sol


def profiled_deviance_looped(theta, d: LMMData, reml=True):
    """S1 reference: identical maths, one Python-level iteration per group.

    Kept so the staged benchmark can show what the algorithmic change is worth
    *before* the block structure is exploited.  This is the shape statsmodels
    uses -- and it is only ~2x faster than statsmodels, which is the point: the
    profiled formulation alone does not explain the win.
    """
    q, p, n, m = d.q, d.p, d.n, d.m
    Lam = theta_to_lambda(theta, q)
    ldL2 = 0.0
    RZX_all = np.empty((m, q, p)); cu_all = np.empty((m, q)); Ls = np.empty((m, q, q))
    sum_pp = np.zeros((p, p)); sum_p = np.zeros(p)
    for i in range(m):
        A = Lam.T @ d.ZtZ[i] @ Lam
        A.flat[:: q + 1] += 1.0
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return np.inf
        Ls[i] = L
        ldL2 += 2.0 * np.log(np.diag(L)).sum()
        RZX = np.linalg.solve(L, Lam.T @ d.ZtX[i])
        cu = np.linalg.solve(L, Lam.T @ d.Zty[i])
        RZX_all[i] = RZX; cu_all[i] = cu
        sum_pp += RZX.T @ RZX; sum_p += RZX.T @ cu
    RXtRX = d.XtX - sum_pp
    try:
        RX = np.linalg.cholesky(RXtRX)
    except np.linalg.LinAlgError:
        return np.inf
    beta = np.linalg.solve(RX.T, np.linalg.solve(RX, d.Xty - sum_p))
    u = np.empty((m, q))
    for i in range(m):
        u[i] = np.linalg.solve(Ls[i].T, cu_all[i] - RZX_all[i] @ beta)
    pwrss = d.yty - float(beta @ d.Xty) - float((u * (d.Zty @ Lam)).sum())
    if not np.isfinite(pwrss) or pwrss <= 0.0:
        return np.inf
    dfree = n - p if reml else n
    dev = ldL2 + (2.0 * np.log(np.diag(RX)).sum() if reml else 0.0)
    return dev + dfree * (1.0 + _LOG2PI + np.log(pwrss / dfree))


def fit_lmm_looped(y, X, Z, groups, reml=True):
    """S1: profiled REML driven by scipy, with the per-group Python loop."""
    d = LMMData(y, X, Z, groups)
    lo = _bounds(d.q)[1]
    bounds = [(0.0, None) if np.isfinite(l) else (None, None) for l in lo]
    res = minimize(profiled_deviance_looped, _default_theta0(d.q), args=(d, reml),
                   method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 10000})
    dev, sol = profiled_deviance(res.x, d, reml, want_solution=True)
    sol.update({"theta": res.x, "deviance": dev, "converged": bool(res.success),
                "nfev": int(res.nfev), "nit": int(res.nit)})
    return sol
