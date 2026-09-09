"""An independent dense implementation of the criterion, and its gradient.

Everything else in this suite checks the compiled core against itself, against
statsmodels, or against lme4's published numbers. This file checks it against
*arithmetic*: the profiled criterion written out over the full n-by-n marginal
covariance, in about fifteen lines of NumPy, sharing no code and no structure
with the implementation.

That matters because the fast path is a block-diagonal Cholesky over
`Lambda' Z'Z Lambda + I` with a flat per-group buffer and rayon accumulators.
An indexing or stride error there would be invisible to a test that compares
one theta against another, and could survive agreement with statsmodels on
easy fixtures. The dense form has none of that machinery: it forms
`V = Z Lambda Lambda' Z' + I` and inverts it.

With `W = Z Lambda Lambda' Z' + I`, `dfree = n - p` under REML and `n` under
ML:

    beta   = (X' W^-1 X)^-1 X' W^-1 y
    pwrss  = (y - X beta)' W^-1 (y - X beta)
    dev    = log|W| + [log|X' W^-1 X|] + dfree * (1 + log(2 pi pwrss / dfree))

The gradient is then taken by central differences *of the dense deviance*, so
it never touches the analytic derivation it is checking.
"""

from __future__ import annotations

import numpy as np
import pytest
from mixedlm_rs import LmmCore


def theta_index(q):
    return [(r, c) for c in range(q) for r in range(c, q)]


def theta_to_lambda(theta, q):
    lam = np.zeros((q, q))
    for k, (r, c) in enumerate(theta_index(q)):
        lam[r, c] = theta[k]
    return lam


def dense_deviance(y, X, Z, codes, m, theta, reml):
    """The profiled criterion, written out densely. No block structure."""
    n, p = X.shape
    q = Z.shape[1]
    lam = theta_to_lambda(theta, q)

    # The full random-effects design: one block of q columns per group.
    zfull = np.zeros((n, m * q))
    for i, g in enumerate(codes):
        zfull[i, g * q:(g + 1) * q] = Z[i]

    lam_big = np.kron(np.eye(m), lam)
    a = zfull @ lam_big
    w = a @ a.T + np.eye(n)

    sign, logdet_w = np.linalg.slogdet(w)
    if sign <= 0:
        return np.inf

    winv_x = np.linalg.solve(w, X)
    xtwx = X.T @ winv_x
    sign_x, logdet_x = np.linalg.slogdet(xtwx)
    if sign_x <= 0:
        return np.inf

    beta = np.linalg.solve(xtwx, X.T @ np.linalg.solve(w, y))
    resid = y - X @ beta
    pwrss = float(resid @ np.linalg.solve(w, resid))
    if not np.isfinite(pwrss) or pwrss <= 0:
        return np.inf

    dfree = (n - p) if reml else n
    dev = logdet_w + dfree * (1.0 + np.log(2 * np.pi * pwrss / dfree))
    if reml:
        dev += logdet_x
    return float(dev)


def make(m=12, nper=6, p=3, q=2, seed=0):
    """Small enough for a dense n-by-n inverse, varied enough to catch strides."""
    rng = np.random.default_rng(seed)
    n = m * nper
    codes = np.repeat(np.arange(m), nper).astype(np.int64)
    X = np.column_stack([np.ones(n)]
                        + [rng.standard_normal(n) for _ in range(p - 1)])
    Z = np.column_stack([np.ones(n)]
                        + [rng.standard_normal(n) for _ in range(q - 1)])
    b = rng.standard_normal((m, q)) * 0.8
    y = (X @ rng.standard_normal(p)
         + np.einsum("nq,nq->n", Z, b[codes])
         + rng.standard_normal(n) * 0.5)
    X, Z = np.ascontiguousarray(X), np.ascontiguousarray(Z)
    return y, X, Z, codes, m


def feasible_theta(q, rng):
    theta = rng.uniform(-0.4, 0.8, q * (q + 1) // 2)
    for k, (r, c) in enumerate(theta_index(q)):
        if r == c:
            theta[k] = abs(theta[k]) + 0.2
    return theta


# ------------------------------------------------------------------ criterion
@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2, 4])
@pytest.mark.parametrize("reml", [True, False])
def test_criterion_matches_the_dense_form(q, p, reml):
    y, X, Z, codes, m = make(p=p, q=q, seed=10 * q + p)
    core = LmmCore(y, X, Z, codes, m)
    rng = np.random.default_rng(100 + 10 * q + p)

    for _ in range(4):
        theta = feasible_theta(q, rng)
        fast = core.deviance(list(theta), reml)
        slow = dense_deviance(y, X, Z, codes, m, theta, reml)
        assert np.isfinite(fast) and np.isfinite(slow)
        assert abs(fast - slow) < 1e-8 * max(1.0, abs(slow)), (
            f"q={q} p={p} reml={reml} theta={theta}: block form {fast!r} "
            f"against dense form {slow!r}")


def test_the_dense_form_is_actually_independent():
    """A guard on the guard.

    If the dense reference silently agreed with anything, it would prove
    nothing. Perturbing theta must move it, and by the same amount the fast
    path moves.
    """
    y, X, Z, codes, m = make()
    core = LmmCore(y, X, Z, codes, m)
    base = np.array([1.0, 0.0, 1.0])
    bumped = np.array([1.3, 0.0, 1.0])

    d_fast = core.deviance(list(bumped), True) - core.deviance(list(base), True)
    d_slow = (dense_deviance(y, X, Z, codes, m, bumped, True)
              - dense_deviance(y, X, Z, codes, m, base, True))
    assert abs(d_fast) > 1e-6, "the perturbation must actually change something"
    assert abs(d_fast - d_slow) < 1e-8 * max(1.0, abs(d_slow))


# ------------------------------------------------------------------- gradient
@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("reml", [True, False])
def test_analytic_gradient_matches_the_dense_finite_difference(q, reml):
    """The analytic gradient against differences of the *dense* criterion.

    tests/test_gradient.py differences the compiled criterion, so it verifies
    the derivation against the same block-Cholesky the derivation runs on.
    This differences a form that shares no code with either.
    """
    y, X, Z, codes, m = make(p=3, q=q, seed=7 * q)
    core = LmmCore(y, X, Z, codes, m)
    rng = np.random.default_rng(500 + q)

    for _ in range(3):
        theta = feasible_theta(q, rng)
        _, analytic = core.deviance_grad(list(theta), reml)
        analytic = np.asarray(analytic)

        h = 1e-6
        numeric = np.zeros_like(theta)
        for k in range(theta.size):
            up, dn = theta.copy(), theta.copy()
            up[k] += h
            dn[k] -= h
            numeric[k] = (dense_deviance(y, X, Z, codes, m, up, reml)
                          - dense_deviance(y, X, Z, codes, m, dn, reml)) / (2 * h)

        scale = np.maximum(np.abs(numeric), 1.0)
        assert np.all(np.abs(analytic - numeric) / scale < 1e-5), (
            f"q={q} reml={reml} theta={theta}\n"
            f" analytic={analytic}\n dense fd={numeric}")


# ------------------------------------------------------- derived quantities
@pytest.mark.parametrize("reml", [True, False])
def test_beta_and_sigma_match_the_dense_solution(reml):
    """The profiled quantities, not just the criterion they are profiled from."""
    y, X, Z, codes, m = make(p=3, q=2, seed=21)
    core = LmmCore(y, X, Z, codes, m)
    theta = np.array([0.9, 0.2, 0.7])
    sol = core.solution(list(theta), reml)

    n, p = X.shape
    q = Z.shape[1]
    lam = theta_to_lambda(theta, q)
    zfull = np.zeros((n, m * q))
    for i, g in enumerate(codes):
        zfull[i, g * q:(g + 1) * q] = Z[i]
    a = zfull @ np.kron(np.eye(m), lam)
    w = a @ a.T + np.eye(n)

    winv_x = np.linalg.solve(w, X)
    beta = np.linalg.solve(X.T @ winv_x, X.T @ np.linalg.solve(w, y))
    resid = y - X @ beta
    pwrss = float(resid @ np.linalg.solve(w, resid))
    dfree = (n - p) if reml else n

    assert np.allclose(np.asarray(sol["beta"]), beta, rtol=1e-8, atol=1e-10)
    assert float(sol["pwrss"]) == pytest.approx(pwrss, rel=1e-9)
    assert float(sol["sigma2"]) == pytest.approx(pwrss / dfree, rel=1e-9)


def test_unbalanced_groups_match_the_dense_form():
    """Equal group sizes would hide a stride error in the block layout."""
    rng = np.random.default_rng(3)
    sizes = np.array([2, 9, 4, 15, 3, 7, 11, 5])
    codes = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    n, m = len(codes), len(sizes)
    X = np.ascontiguousarray(
        np.column_stack([np.ones(n), rng.standard_normal(n)]))
    Z = np.ascontiguousarray(
        np.column_stack([np.ones(n), rng.standard_normal(n)]))
    y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)

    core = LmmCore(y, X, Z, codes, m)
    for reml in (True, False):
        for theta in ([1.0, 0.0, 1.0], [0.4, -0.3, 1.2], [1.5, 0.6, 0.2]):
            theta = np.array(theta)
            fast = core.deviance(list(theta), reml)
            slow = dense_deviance(y, X, Z, codes, m, theta, reml)
            assert abs(fast - slow) < 1e-8 * max(1.0, abs(slow)), (
                f"unbalanced, reml={reml}, theta={theta}")
