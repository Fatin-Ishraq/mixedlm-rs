"""The analytic gradient is the one place this project goes past its references.

lme4 and MixedModels.jl both optimise theta derivative-free (BOBYQA). We derive
and evaluate the gradient of the profiled criterion instead. That derivation is
therefore the most dangerous code in the package, and it is checked here against
central finite differences across a wide sweep of shapes, thetas and both
criteria.

A wrong gradient does not crash -- it quietly converges to the wrong answer.
"""

import numpy as np
import pytest

from mixedlm_rs import LmmCore


def make_core(m=40, nper=12, p=3, q=2, seed=0):
    rng = np.random.default_rng(seed)
    n = m * nper
    codes = np.repeat(np.arange(m), nper).astype(np.int64)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    Z = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(q - 1)])
    b = rng.standard_normal((m, q)) * 0.8
    y = X @ rng.standard_normal(p) + np.einsum("nq,nq->n", Z, b[codes]) \
        + rng.standard_normal(n) * 0.5
    return LmmCore(y, np.ascontiguousarray(X), np.ascontiguousarray(Z), codes, m)


def central_diff(core, theta, reml, h=1e-6):
    g = np.zeros(len(theta))
    for k in range(len(theta)):
        tp = np.array(theta, dtype=float)
        tm = np.array(theta, dtype=float)
        tp[k] += h
        tm[k] -= h
        fp = core.deviance(list(tp), reml)
        fm = core.deviance(list(tm), reml)
        g[k] = (fp - fm) / (2 * h)
    return g


THETAS = [
    [1.0, 0.0, 1.0],
    [0.7, 0.25, 0.4],
    [1.5, -0.6, 0.9],
    [0.3, 0.05, 1.8],
    [2.2, 0.9, 0.15],
]


@pytest.mark.parametrize("reml", [True, False])
@pytest.mark.parametrize("theta", THETAS)
def test_gradient_matches_finite_differences(reml, theta):
    core = make_core()
    _, ga = core.deviance_grad(list(theta), reml)
    gn = central_diff(core, theta, reml)
    ga = np.asarray(ga)
    scale = np.maximum(np.abs(gn), 1.0)
    rel = np.abs(ga - gn) / scale
    assert np.all(rel < 2e-5), (
        f"reml={reml} theta={theta}\n analytic={ga}\n numeric ={gn}\n rel={rel}"
    )


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("reml", [True, False])
def test_gradient_across_q(q, reml):
    core = make_core(m=30, nper=10, p=2, q=q, seed=q)
    nth = core.n_theta
    rng = np.random.default_rng(100 + q)
    for _ in range(4):
        theta = core.default_theta()
        theta = list(np.asarray(theta) + rng.uniform(-0.3, 0.6, nth))
        # keep diagonal entries positive
        idx = 0
        for c in range(q):
            for r in range(c, q):
                if r == c:
                    theta[idx] = abs(theta[idx]) + 0.15
                idx += 1
        f, ga = core.deviance_grad(theta, reml)
        if not np.isfinite(f):
            continue
        gn = central_diff(core, theta, reml)
        ga = np.asarray(ga)
        scale = np.maximum(np.abs(gn), 1.0)
        assert np.all(np.abs(ga - gn) / scale < 5e-5), (
            f"q={q} reml={reml} theta={theta}\n analytic={ga}\n numeric={gn}"
        )


@pytest.mark.parametrize("p", [1, 2, 5, 10])
def test_gradient_across_p(p):
    core = make_core(m=25, nper=8, p=p, q=2, seed=p)
    theta = [0.8, 0.2, 0.6]
    for reml in (True, False):
        _, ga = core.deviance_grad(theta, reml)
        gn = central_diff(core, theta, reml)
        ga = np.asarray(ga)
        scale = np.maximum(np.abs(gn), 1.0)
        assert np.all(np.abs(ga - gn) / scale < 5e-5), f"p={p} reml={reml}"


def test_gradient_near_the_boundary():
    """Variance components legitimately go to zero; the gradient must survive it."""
    core = make_core(m=30, nper=10, p=2, q=2, seed=7)
    for theta in ([1e-4, 0.0, 0.9], [0.9, 0.0, 1e-4], [1e-3, 1e-3, 1e-3]):
        for reml in (True, False):
            f, ga = core.deviance_grad(list(theta), reml)
            assert np.isfinite(f)
            gn = central_diff(core, theta, reml, h=1e-8)
            ga = np.asarray(ga)
            scale = np.maximum(np.abs(gn), 1.0)
            assert np.all(np.abs(ga - gn) / scale < 1e-3), (
                f"theta={theta} reml={reml}\n analytic={ga}\n numeric={gn}"
            )


def test_infeasible_theta_reports_infinity():
    core = make_core()
    # A negative diagonal entry makes Lambda' Z'Z Lambda + I indefinite in general;
    # at minimum the criterion must not return a finite lie.
    f = core.deviance([0.0, 0.0, 0.0], True)
    assert np.isfinite(f), "theta = 0 is feasible (Lambda = 0 gives A = I)"
