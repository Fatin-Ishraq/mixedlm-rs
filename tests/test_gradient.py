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


def _random_feasible_theta(q, rng):
    """A random theta with a strictly positive diagonal."""
    nth = q * (q + 1) // 2
    theta = rng.uniform(-0.3, 0.6, nth)
    idx = 0
    for c in range(q):
        for r in range(c, q):
            if r == c:
                theta[idx] = abs(theta[idx]) + 0.15
            idx += 1
    return theta


# The full product of shapes and criteria, not one dimension at a time. The
# earlier pair of tests varied q with p fixed at 2, and p with q fixed at 2, so
# no case with q = 3 and p = 10 was ever evaluated -- and the interaction
# between the two is exactly where an index or stride error in the block
# layout would show up.
@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 2, 5, 10])
@pytest.mark.parametrize("reml", [True, False])
def test_gradient_across_shapes(q, p, reml):
    core = make_core(m=30, nper=10, p=p, q=q, seed=10 * q + p)
    rng = np.random.default_rng(100 + 10 * q + p)
    checked = 0
    for _ in range(4):
        theta = list(_random_feasible_theta(q, rng))
        f, ga = core.deviance_grad(theta, reml)
        # A non-finite criterion here would mean the *generator* produced an
        # infeasible point, which it must not: skipping silently, as this once
        # did with `continue`, would let the whole case vanish from the sweep.
        assert np.isfinite(f), f"generated an infeasible theta: {theta}"
        gn = central_diff(core, theta, reml)
        ga = np.asarray(ga)
        scale = np.maximum(np.abs(gn), 1.0)
        assert np.all(np.abs(ga - gn) / scale < 5e-5), (
            f"q={q} p={p} reml={reml} theta={theta}\n"
            f" analytic={ga}\n numeric={gn}"
        )
        checked += 1
    assert checked == 4, "every generated theta must actually be checked"


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


@pytest.mark.parametrize("reml", [True, False])
def test_gradient_at_exactly_zero(reml):
    """theta = 0 is a stationary point of the profiled criterion for ANY data.

    At Lambda = 0 every term of the analytic gradient vanishes identically, so
    this is the one point where "the gradient is zero" proves nothing about the
    derivation being right -- and it is also the trap the optimiser has to be
    steered out of. Checking that the analytic gradient *agrees with finite
    differences* there is what distinguishes a correct zero from a lucky one:
    the one-sided numeric derivative into the feasible region must be zero too.

    This case previously asserted only that the deviance was finite.
    """
    core = make_core(m=30, nper=10, p=2, q=2, seed=11)
    zero = [0.0, 0.0, 0.0]
    f, ga = core.deviance_grad(zero, reml)
    assert np.isfinite(f)
    ga = np.asarray(ga)
    assert np.allclose(ga, 0.0, atol=1e-10), f"analytic gradient at 0: {ga}"

    # One-sided differences, since the diagonal cannot go negative. A forward
    # difference at step h measures f'(0) + (1/2) f''(0) h, so at a true
    # stationary point it does not vanish -- it shrinks *linearly in h*. That
    # is the property to assert: halving h must halve the measured slope. A
    # genuinely non-zero derivative would leave it constant instead.
    for k in range(len(zero)):
        # Steps stay above the rounding floor: the deviance is O(1e3), so at
        # h = 1e-8 the difference is pure cancellation noise.
        slopes = []
        for h in (1e-4, 1e-5, 1e-6):
            up = np.array(zero)
            up[k] += h
            slopes.append((core.deviance(list(up), reml) - f) / h)
        big, small = abs(slopes[0]), abs(slopes[2])
        # A 100x smaller step must give a ~100x smaller slope.
        assert small <= big * 0.02 + 1e-12, (
            f"direction {k}: slopes {slopes} do not shrink in proportion to h, "
            "which means a genuine non-zero first derivative rather than "
            "curvature -- the analytic zero would then be wrong")
