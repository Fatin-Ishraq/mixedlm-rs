"""Invariances the fit must have, and did not.

A linear mixed model has exact equivariances under rescaling and translating
its inputs. They are worth testing directly, because a violation is not a
crash: it is a plausible-looking wrong answer, reported as converged.

Two real defects lived here:

* The criterion evaluated ``pwrss = y'y - beta'X'y - u'Lambda'Z'y``, a
  difference of large nearly equal quantities. A response near 1e8 -- prices in
  minor units, epoch timestamps, populations, anything with a large offset --
  destroyed it entirely, and the fit still said ``converged=True``.
* The stationarity tolerance was ``max(1e-4, 1e-6 * abs(deviance))``. Rescaling
  the response, or the fixed-effect columns under REML, adds a constant to the
  criterion without changing its theta-gradient, so the same stationary point
  could be accepted in one set of units and rejected in another.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mixedlm_rs as mlm


def fixture(seed=7, m=20, nper=10):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), nper)
    n = len(g)
    x = rng.normal(size=n)
    b = rng.normal(scale=0.6, size=m)
    y = 1.0 + 0.5 * x + b[g] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlm.mixedlm("y ~ x", df, groups=df["g"], **kw).fit()


# ------------------------------------------------------- response translation
@pytest.mark.parametrize("shift", [1e2, 1e4, 1e6, 1e8, -1e8])
def test_response_translation_leaves_everything_but_the_intercept(shift):
    """y -> y + c shifts only the intercept. Everything else is identical.

    At 1e8 this used to move the log-likelihood by 280 and the residual
    variance by a factor of 18, while still reporting convergence.
    """
    df = fixture()
    base = fit(df)
    moved = fit(df.assign(y=df["y"] + shift))

    assert moved.llf == pytest.approx(base.llf, abs=1e-6)
    assert moved.scale == pytest.approx(base.scale, rel=1e-7)
    assert moved.cov_re[0, 0] == pytest.approx(base.cov_re[0, 0], rel=1e-7)
    assert moved.converged

    # The intercept absorbs the shift; the slope does not move.
    assert moved.fe_params[0] == pytest.approx(base.fe_params[0] + shift,
                                               rel=1e-12, abs=1e-6)
    assert moved.fe_params[1] == pytest.approx(base.fe_params[1], rel=1e-7)


def test_translation_does_not_silently_degrade_before_it_breaks():
    """The failure was gradual: three digits were already gone at 1e6."""
    df = fixture()
    base = fit(df)
    for shift in (1e3, 1e4, 1e5, 1e6, 1e7):
        moved = fit(df.assign(y=df["y"] + shift))
        assert moved.llf == pytest.approx(base.llf, abs=1e-6), \
            f"log-likelihood drifted at a shift of {shift:g}"


# ---------------------------------------------------------------- rescaling
@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_response_rescaling_is_equivariant(c):
    """y -> c*y scales beta and every variance by c and c^2 respectively.

    The theta-gradient is exactly invariant under this: pwrss picks up c^2 and
    the dfree/pwrss factor cancels it, while the log-determinant terms do not
    involve y at all. So the *stationarity verdict* must not change either.
    """
    df = fixture()
    base = fit(df)
    scaled = fit(df.assign(y=df["y"] * c))

    assert scaled.converged == base.converged
    assert np.allclose(scaled.fe_params, base.fe_params * c, rtol=1e-7)
    assert scaled.scale == pytest.approx(base.scale * c * c, rel=1e-7)
    assert scaled.cov_re[0, 0] == pytest.approx(base.cov_re[0, 0] * c * c,
                                                rel=1e-6)


@pytest.mark.parametrize("c", [1e-5, 1e-2, 1e2, 1e5])
def test_predictor_rescaling_is_equivariant(c):
    """x -> c*x scales that coefficient by 1/c and leaves the variances alone."""
    df = fixture()
    base = fit(df)
    scaled = fit(df.assign(x=df["x"] * c))

    assert scaled.converged == base.converged
    assert scaled.scale == pytest.approx(base.scale, rel=1e-7)
    assert scaled.cov_re[0, 0] == pytest.approx(base.cov_re[0, 0], rel=1e-6)
    assert scaled.fe_params[0] == pytest.approx(base.fe_params[0], rel=1e-7)
    assert scaled.fe_params[1] == pytest.approx(base.fe_params[1] / c, rel=1e-6)


@pytest.mark.parametrize("c", [1e-5, 1e-2, 1e2, 1e5])
def test_reml_criterion_shifts_by_exactly_two_log_c(c):
    """Rescaling a fixed-effect column shifts the REML criterion by 2*log(c).

    This is a genuine property of REML, not a defect: the criterion carries
    ``log|X' V^-1 X|``, so it depends on the basis chosen for X. ML has no such
    term and is exactly invariant. Both are checked, because the internal
    conditioning *also* rescales the fixed-effect columns, and if the constant
    it introduces were not restored the reported REML criterion would silently
    be in the internal basis rather than the caller's.

    statsmodels shifts by the same 2*log(c) on the same data, which is what
    makes this the right behaviour rather than merely a self-consistent one.
    """
    df = fixture()
    scaled_df = df.assign(x=df["x"] * c)

    reml_base, reml_scaled = fit(df), fit(scaled_df)
    shift = -2 * (reml_scaled.llf - reml_base.llf)
    assert shift == pytest.approx(2 * np.log(c), abs=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_base = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit(reml=False)
        ml_scaled = mlm.mixedlm("y ~ x", scaled_df,
                                groups=scaled_df["g"]).fit(reml=False)
    assert ml_scaled.llf == pytest.approx(ml_base.llf, abs=1e-6), \
        "the ML criterion has no log|X'V^-1 X| term and must not move"


def test_the_stationarity_verdict_is_the_same_in_every_unit_system():
    """The gradient norm itself must not depend on the units.

    This is the property the old tolerance broke: it compared the projected
    gradient against a fraction of the *deviance*, which shifts by a constant
    under rescaling while the gradient does not.
    """
    df = fixture()
    thetas, verdicts = [], []
    for cy in (1.0, 1e3, 1e6):
        for cx in (1.0, 1e4):
            r = fit(df.assign(y=df["y"] * cy, x=df["x"] * cx))
            thetas.append(float(np.asarray(r._res["theta"])[0]))
            verdicts.append(bool(r.converged))

    assert all(verdicts), "the same optimum was rejected in some unit system"
    assert max(thetas) - min(thetas) < 1e-7, \
        f"theta depends on the units: {thetas}"


# ------------------------------------------------- random-effects rescaling
def test_random_slope_rescaling_is_equivariant():
    """Scaling a random-effects column rescales its variance component only."""
    rng = np.random.default_rng(3)
    m, nper = 30, 12
    g = np.repeat(np.arange(m), nper)
    n = len(g)
    x = rng.normal(size=n)
    b0 = rng.normal(scale=0.8, size=m)
    b1 = rng.normal(scale=0.4, size=m)
    y = 1.0 + 2.0 * x + b0[g] + b1[g] * x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": g})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = mlm.mixedlm("y ~ x", df, groups=df["g"], re_formula="~x").fit()
        c = 1e3
        scaled = mlm.mixedlm("y ~ x", df.assign(x=df["x"] * c),
                             groups=df["g"], re_formula="~x").fit()

    # x appears in both designs here, so the REML criterion picks up the
    # 2*log(c) from the fixed-effect column while every variance component
    # transforms exactly.
    assert -2 * (scaled.llf - base.llf) == pytest.approx(2 * np.log(c), abs=1e-4)
    assert scaled.scale == pytest.approx(base.scale, rel=1e-5)
    assert scaled.cov_re[0, 0] == pytest.approx(base.cov_re[0, 0], rel=1e-5)
    assert scaled.cov_re[1, 1] == pytest.approx(base.cov_re[1, 1] / c**2,
                                                rel=1e-5)
