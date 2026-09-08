"""Degenerate and near-degenerate models.

These are the regimes where a mixed-model fitter actually fails, and every test
here corresponds to a defect found by a stress sweep rather than to a
hypothetical.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mixedlm_rs as mlm
from mixedlm_rs import ConvergenceWarning


# --------------------------------------------------------------- tiny variance
def _tiny_variance_data(seed, re_sd):
    """A genuine random effect, but a very small one."""
    rng = np.random.default_rng(seed)
    ngroups, nper = 120, 20
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    b0 = rng.standard_normal(ngroups) * re_sd
    y = 1.0 + 2.0 * x1 + b0[codes] + rng.standard_normal(n) * 1.0
    return pd.DataFrame(dict(y=y, x1=x1, g=codes))


@pytest.mark.parametrize("re_sd", [0.3, 0.1, 0.05, 0.02, 0.01])
def test_small_variance_still_produces_a_valid_fit(re_sd):
    """theta = 0 is a stationary point for ANY data, so an optimiser that
    reaches the bound stops there. The boundary escape probes away from it --
    and its ladder has to reach small values, because the true optimum can be
    smaller than the smallest probe.

    With a ladder starting at 0.05, a true optimum near theta = 0.028 was
    missed: every probe overshot it and the optimiser slid back to zero.

    Note what this does *not* assert. An estimate of exactly zero is not by
    itself a bug: the REML estimate legitimately sits on the boundary when the
    observed between-group spread is no larger than sampling noise would
    produce. On the re_sd=0.02 fixture below, statsmodels also returns exactly
    0.0, with a likelihood identical to ours. The property that matters is the
    criterion, which is what the next test checks.
    """
    df = _tiny_variance_data(seed=int(re_sd * 1000) + 7, re_sd=re_sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert r.converged
    assert np.isfinite(r.llf)
    assert r.scale > 0
    assert r.cov_re[0, 0] >= 0.0


@pytest.mark.parametrize("re_sd", [0.3, 0.1, 0.05])
def test_clearly_identifiable_variance_is_not_collapsed_to_zero(re_sd):
    """Where the signal is unambiguous, a zero estimate would be a real failure."""
    df = _tiny_variance_data(seed=int(re_sd * 1000) + 7, re_sd=re_sd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert r.cov_re[0, 0] > 0.0, (
        f"true re_sd={re_sd} collapsed to a zero variance estimate")


def test_beats_or_matches_statsmodels_on_small_variance():
    smf = pytest.importorskip("statsmodels.formula.api")
    for re_sd in (0.05, 0.02, 0.01):
        df = _tiny_variance_data(seed=int(re_sd * 1000) + 7, re_sd=re_sd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
            theirs = smf.mixedlm("y ~ x1", df, groups=df["g"]).fit()
        if not theirs.converged:
            continue
        tol = 1e-6 * max(abs(theirs.llf), 1.0)
        assert ours.llf >= theirs.llf - tol, (
            f"re_sd={re_sd}: our optimum is worse, {ours.llf} < {theirs.llf}")


# ------------------------------------------------------------ identifiability
def _saturated(q):
    """n == q * m exactly: every group perfectly fitted by its own random effects."""
    rng = np.random.default_rng(0)
    ngroups, nper = 60, q
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes] + rng.standard_normal(n) * 0.5
    return pd.DataFrame(dict(y=y, x1=x1, g=codes))


@pytest.mark.parametrize("q,re_formula", [(1, None), (2, "~x1")])
def test_unidentifiable_model_warns(q, re_formula):
    """With as many random effects as observations, the residual variance and
    the variance components cannot be separated: the profiled likelihood
    diverges as sigma^2 -> 0 rather than attaining a maximum.

    lme4 refuses these outright. statsmodels fits them silently, so refusing
    would break the drop-in contract -- but the user must be told.
    """
    df = _saturated(q)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula=re_formula).fit()
    msgs = [str(x.message) for x in w
            if issubclass(x.category, ConvergenceWarning)]
    assert any("not identifiable" in m for m in msgs), \
        f"expected an identifiability warning, got {msgs}"


def test_well_identified_model_does_not_warn():
    """The check must not fire on ordinary data."""
    rng = np.random.default_rng(1)
    ngroups, nper = 40, 12
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes] + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(dict(y=y, x1=x1, g=codes))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    assert not any("not identifiable" in str(x.message) for x in w)


# ------------------------------------------------------------- other corners
def test_groups_of_size_one():
    """Singleton groups are legal and common in unbalanced designs."""
    rng = np.random.default_rng(3)
    sizes = np.array([1] * 40 + [8] * 40)
    codes = np.repeat(np.arange(len(sizes)), sizes)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(len(sizes))[codes] + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(dict(y=y, x1=x1, g=codes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert np.isfinite(r.llf)
    assert r.scale > 0


def test_near_collinear_fixed_effects():
    """Nearly collinear predictors must not produce NaNs or a failed Cholesky."""
    rng = np.random.default_rng(4)
    ngroups, nper = 50, 20
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    x2 = x1 * 0.999 + rng.standard_normal(n) * 1e-3
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(ngroups)[codes] \
        + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=codes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"]).fit()
    assert np.all(np.isfinite(r.fe_params))
    assert np.isfinite(r.llf) and r.scale > 0


def test_extreme_predictor_scaling():
    """Predictors spanning six orders of magnitude still fit."""
    for scale in (1e-3, 1.0, 1e3):
        rng = np.random.default_rng(5)
        ngroups, nper = 50, 15
        codes = np.repeat(np.arange(ngroups), nper)
        n = len(codes)
        x1 = rng.standard_normal(n) * scale
        y = (1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes]
             + rng.standard_normal(n) * 0.5)
        df = pd.DataFrame(dict(y=y, x1=x1, g=codes))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
        assert np.all(np.isfinite(r.fe_params)), f"scale={scale}"
        ev = np.linalg.eigvalsh(r.cov_re)
        assert np.all(ev >= -1e-8), f"scale={scale}: cov_re not PSD"


def test_heavy_outliers_do_not_break_the_fit():
    rng = np.random.default_rng(6)
    ngroups, nper = 60, 15
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes] + rng.standard_normal(n) * 0.5
    y[rng.integers(0, n, n // 50)] += rng.standard_normal(n // 50) * 100
    df = pd.DataFrame(dict(y=y, x1=x1, g=codes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert np.isfinite(r.llf) and r.scale > 0
