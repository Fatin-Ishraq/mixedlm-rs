"""Degenerate and near-degenerate models.

These are the regimes where a mixed-model fitter actually fails, and every test
here corresponds to a defect found by a stress sweep rather than to a
hypothetical.
"""

import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest
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
    return pd.DataFrame({"y": y, "x1": x1, "g": codes})


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
#
# The counting rule `n <= q * m` was once treated as proof that a model is
# unidentifiable, on the reasoning that every group is interpolated and the
# likelihood diverges. Both halves are false, and these tests pin the two
# counterexamples:
#
#   * q = 1 with one observation per group really is unidentifiable, but the
#     criterion is *flat* along the variance split, not divergent -- V is
#     (tau^2 + sigma^2) I, so only the total is determined.
#   * q = 2 with two observations per group satisfies the same count rule and
#     is perfectly well identified: it has an interior optimum with a positive
#     residual variance, and the old rule warned about it for no reason.
#
# So the warning is now driven by asking the criterion whether it is flat,
# not by counting.
def _singleton_groups():
    """One observation per group: the variance split is genuinely unidentified."""
    rng = np.random.default_rng(0)
    n = 60
    codes = np.arange(n)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "x1": x1, "g": codes})


def _two_per_group_random_slope():
    """n == q * m, but identified: the criterion has an interior optimum."""
    rng = np.random.default_rng(0)
    ngroups, nper = 60, 2
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = (1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes]
         + rng.standard_normal(n) * 0.5)
    return pd.DataFrame({"y": y, "x1": x1, "g": codes})


def _warnings_from(df, re_formula):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = mlm.mixedlm("y ~ x1", df, groups=df["g"],
                          re_formula=re_formula).fit()
    return res, [str(x.message) for x in w
                 if issubclass(x.category, ConvergenceWarning)]


def test_flat_variance_split_warns():
    """One observation per group: warn, and say the criterion is flat."""
    _, msgs = _warnings_from(_singleton_groups(), None)
    assert any("not identifiable" in m for m in msgs), \
        f"expected an identifiability warning, got {msgs}"
    assert any("flat" in m for m in msgs), \
        "the warning must say the criterion is flat, not that it diverges"


def test_flat_criterion_really_is_flat():
    """The premise of the warning above, checked directly against the core."""
    df = _singleton_groups()
    model = mlm.mixedlm("y ~ x1", df, groups=df["g"])
    core = model._core()
    values = [core.deviance([t], True) for t in (0.0, 1.0, 10.0, 100.0)]
    spread = (max(values) - min(values)) / max(1.0, abs(values[0]))
    assert spread < 1e-9, f"expected a flat criterion, got {values}"


def test_saturated_but_identified_model_does_not_warn():
    """n == q * m and yet perfectly well identified -- the count rule is wrong.

    This model has an interior optimum with a positive residual variance. The
    old rule warned here purely because of the row count.
    """
    res, msgs = _warnings_from(_two_per_group_random_slope(), "~x1")
    assert not any("not identifiable" in m for m in msgs), \
        f"false identifiability warning on a well-identified model: {msgs}"
    assert res.scale > 0
    assert np.isfinite(res.llf)


def test_single_group_is_caught_although_the_count_rule_misses_it():
    """m == 1: the random intercept is confounded with the fixed intercept.

    `n <= q * m` is false here (n is 60, q * m is 1), so counting never flags
    it, but the variance split is exactly as unidentified as the flat case.
    """
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({"y": rng.standard_normal(n),
                       "x1": rng.standard_normal(n),
                       "g": np.zeros(n)})
    _, msgs = _warnings_from(df, None)
    assert any("not identifiable" in m for m in msgs), \
        f"expected an identifiability warning for a single group, got {msgs}"


def test_well_identified_model_does_not_warn():
    """The check must not fire on ordinary data."""
    rng = np.random.default_rng(1)
    ngroups, nper = 40, 12
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(ngroups)[codes] + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"y": y, "x1": x1, "g": codes})
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
    df = pd.DataFrame({"y": y, "x1": x1, "g": codes})
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
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": codes})
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
        df = pd.DataFrame({"y": y, "x1": x1, "g": codes})
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
    df = pd.DataFrame({"y": y, "x1": x1, "g": codes})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert np.isfinite(r.llf) and r.scale > 0


def test_boundary_fits_decline_to_invent_variance_standard_errors():
    """At a singular fit the profiled Hessian need not be positive definite.

    Inverting it anyway produces numbers that look like standard errors and
    are not: the estimate is on the edge of the parameter space, where the
    usual asymptotics do not hold. `bse_re` returns NaN there instead, and
    `singular` says why.
    """
    import pathlib
    data = pathlib.Path(__file__).resolve().parents[1] / "data"
    if not data.is_dir():
        pytest.skip("lme4 fixtures are repository-only")
    df = pd.read_csv(data / "Dyestuff2.csv")
    r = mlm.mixedlm("Yield ~ 1", df, groups=df["Batch"]).fit()

    assert r.singular, "Dyestuff2 is the canonical boundary fit"
    assert r.converged, "a boundary optimum is a converged fit"
    assert np.all(np.isnan(np.asarray(r.bse_re))), \
        "a standard error at the boundary would be fabricated"


def test_interior_fits_still_report_variance_standard_errors():
    """The check above must not suppress legitimate standard errors."""
    import pathlib
    data = pathlib.Path(__file__).resolve().parents[1] / "data"
    if not data.is_dir():
        pytest.skip("lme4 fixtures are repository-only")
    df = pd.read_csv(data / "sleepstudy.csv")
    r = mlm.mixedlm("Reaction ~ Days", df, groups=df["Subject"],
                    re_formula="~Days").fit()

    assert not r.singular
    se = np.asarray(r.bse_re)
    assert np.all(np.isfinite(se)) and np.all(se > 0)
    # statsmodels reports 11.881 / 1.821 / 0.610 for this fit.
    assert np.allclose(se, [11.881, 1.821, 0.610], rtol=5e-3)
