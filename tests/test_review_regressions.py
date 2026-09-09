"""Defects found in external review, each pinned by the case that found it.

The rest of the suite tests properties. This file tests *history*: every entry
below is a specific wrong answer this package once produced, kept as a fixture
so it cannot come back quietly. Each test says what the old behaviour was, so
a future failure is legible without digging through the log.

Some of these also live elsewhere as properties -- response translation in
test_invariance.py, the boundary escape in test_degenerate.py, malformed
native input in test_native_safety.py. Those are the general statements; these
are the individual counterexamples.
"""

from __future__ import annotations

import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest
from mixedlm_rs import ConvergenceWarning


def frame(n=200, m=20, seed=7):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = (1.0 + 0.5 * x + rng.normal(scale=0.6, size=m)[g]
         + rng.normal(scale=0.5, size=n))
    return pd.DataFrame({"y": y, "x": x, "g": g})


def fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlm.mixedlm("y ~ x", df, groups=df["g"], **kw).fit(**kw.pop("fit_kw", {}))


# ============================================================ false convergence
#
# Convergence was `optimiser_flag or stationary`, so the optimiser's own
# success flag could overrule the gradient check. With a loose tolerance it
# reported success at a projected gradient of 19 against a threshold of
# 0.000416.
def test_a_loose_ftol_does_not_buy_a_false_success():
    df = frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit(ftol=0.1, gtol=0.1)
        tight = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit()

    if loose.converged:
        # Allowed only if it genuinely landed on the same optimum.
        assert loose.llf == pytest.approx(tight.llf, abs=1e-4), (
            "converged=True at a loose tolerance must mean the point really "
            "is stationary, not that the optimiser stopped early")
    d = loose.diagnostics
    assert (d["converged"]
            == (d["max_abs_projected_gradient"] <= d["gradient_tolerance"])), (
        "the convergence flag and the gradient check must not disagree")


def test_convergence_always_matches_the_certificate():
    """Across a spread of tolerances, the flag is the certificate. Never a flag."""
    df = frame()
    for gtol, ftol in [(1e-8, 1e-12), (1e-4, 1e-8), (1e-2, 1e-4), (0.5, 0.5)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit(gtol=gtol, ftol=ftol)
        d = r.diagnostics
        assert r.converged == (
            d["max_abs_projected_gradient"] <= d["gradient_tolerance"])


def test_a_failed_fit_warns_rather_than_only_setting_a_flag():
    """A caller who does not read `converged` must still be told."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from test_fuzz import random_case

    df, re_formula, reml = random_case(np.random.default_rng(10_032))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                        re_formula=re_formula).fit(reml=reml, method="rust")
    if not r.converged:
        assert any(issubclass(x.category, ConvergenceWarning) for x in w)


# ================================================ conditional random effects cov
#
# `random_effects_cov` returned the population covariance G for every group,
# where the conditional covariance given that group's data was asked for. On a
# 20-group fixture it returned 3.304 where statsmodels returned 0.027.
def test_conditional_covariance_matches_the_closed_form():
    """For a random intercept the answer is `1 / (1/tau^2 + n_i/sigma^2)`."""
    df = frame()
    r = fit(df)

    tau2 = float(r.cov_re[0, 0])
    sigma2 = float(r.scale)
    sizes = df["g"].value_counts().sort_index().to_numpy()

    for label, size in zip(sorted(df["g"].unique()), sizes, strict=True):
        got = float(np.asarray(r.random_effects_cov[label])[0, 0])
        want = 1.0 / (1.0 / tau2 + size / sigma2)
        assert got == pytest.approx(want, rel=1e-8), f"group {label}"


def test_conditional_covariance_is_smaller_than_the_population_covariance():
    """The property that makes it conditional at all: data shrinks it."""
    r = fit(frame())
    population = float(r.cov_re[0, 0])
    for cov in r.random_effects_cov.values():
        assert float(np.asarray(cov)[0, 0]) < population


def test_conditional_covariance_shrinks_with_group_size():
    """A bigger group is better determined, so its posterior is tighter."""
    rng = np.random.default_rng(3)
    sizes = np.array([2, 5, 20, 100])
    codes = np.repeat(np.arange(len(sizes)), sizes)
    n = len(codes)
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.normal(scale=0.8, size=len(sizes))[codes] \
        + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": codes})
    r = fit(df)

    got = [float(np.asarray(r.random_effects_cov[g])[0, 0])
           for g in sorted(df["g"].unique())]
    assert got == sorted(got, reverse=True), (
        f"conditional variance should fall as group size rises, got {got}")


def test_each_group_gets_its_own_object():
    """Every group shared one mutable frame, so editing one edited all."""
    r = fit(frame())
    keys = list(r.random_effects_cov)
    assert r.random_effects_cov[keys[0]] is not r.random_effects_cov[keys[1]]


# ============================================================== missing data
#
# Missing rows were dropped independently for endog/exog and exog_re, and not
# at all for groups -- so a missing group label survived to become a group of
# its own, and the arrays ended up misaligned.
def test_missing_group_label_is_dropped_not_made_into_a_level():
    df = frame()
    df.loc[df.index[:5], "g"] = np.nan          # 5 of group 0's 10 rows
    r = fit(df, missing="drop")

    assert r.nobs == 195
    # Group 0 keeps its other 5 rows, so 20 levels is right. What must not
    # happen is a 21st level made of the rows whose group is unknown.
    assert r.model.n_groups == df["g"].nunique() == 20
    assert not any(pd.isna(lab) for lab in r.model.group_labels), (
        "a missing group label became a level of its own")


def test_a_wholly_missing_group_disappears():
    df = frame()
    df.loc[df.index[:10], "g"] = np.nan         # all of group 0
    r = fit(df, missing="drop")
    assert r.nobs == 190
    assert r.model.n_groups == 19


def test_missing_random_effect_predictor_drops_the_same_rows():
    df = frame()
    df.loc[df.index[:7], "x"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ 1", df, groups=df["g"], re_formula="~x",
                        missing="drop").fit()
    assert r.nobs == 193
    assert len(r.fittedvalues) == 193


def test_rows_stay_aligned_after_a_mixed_drop():
    """Missing in three different places at once; alignment is the point."""
    df = frame()
    df.loc[df.index[0], "y"] = np.nan
    df.loc[df.index[50], "x"] = np.nan
    df.loc[df.index[100], "g"] = np.nan
    r = fit(df, missing="drop")
    assert r.nobs == 197

    # Compare against dropping by hand, which is unambiguous.
    clean = df.dropna()
    want = fit(clean.reset_index(drop=True))
    assert r.llf == pytest.approx(want.llf, abs=1e-9)
    assert np.allclose(r.fe_params, want.fe_params, rtol=1e-9)


def test_missing_data_without_drop_is_refused():
    df = frame()
    df.loc[df.index[0], "x"] = np.nan
    # patsy raises PatsyError; the point is that it is not silently dropped.
    from patsy import PatsyError
    with pytest.raises((PatsyError, ValueError)):
        fit(df)


# ================================================ variance standard errors
#
# `bse_re` inverted the profiled Hessian unchecked. At a boundary optimum it
# need not be positive definite, and the result looked like standard errors
# without being any.
def test_singular_fit_returns_nan_standard_errors():
    rng = np.random.default_rng(11)
    n, m = 200, 20
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.normal(scale=0.5, size=n)      # no group effect
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    r = fit(df)
    if r.singular:
        assert np.all(np.isnan(np.asarray(r.bse_re)))


def test_interior_fit_returns_finite_standard_errors():
    r = fit(frame())
    assert not r.singular
    assert np.all(np.isfinite(np.asarray(r.bse_re)))
    assert np.all(np.asarray(r.bse_re) > 0)


def test_bse_re_and_bse_cov_re_differ_by_the_documented_factor():
    """`bse_re` is the reference's sqrt(scale)-weighted quantity; the other
    is on the same scale as `cov_re`. Confusing them was the original bug."""
    r = fit(frame())
    packed = np.asarray(r.bse)[r.k_fe:r.k_fe + r.k_re2]
    assert np.allclose(np.asarray(r.bse_re), np.sqrt(r.scale) * packed)
    assert np.allclose(np.asarray(r.bse_cov_re), r.scale * packed)


# ====================================================== summary display
#
# summary() printed cov_re_unscaled under the label "Group Var": 0.935 on
# sleepstudy where lme4 and statsmodels both report 612.1.
def test_summary_prints_the_covariance_not_the_ratio():
    r = fit(frame())
    line = next(ln for ln in str(r.summary()).splitlines()
                if ln.startswith("Group Var"))
    shown = float(line.split()[2])
    assert shown == pytest.approx(r.cov_re[0, 0], abs=5e-3)
    assert shown != pytest.approx(r.cov_re_unscaled[0, 0], abs=5e-3), (
        "the two must differ here, or the test proves nothing")


# ================================================================ rank checks
def test_duplicated_predictor_names_the_design_not_theta():
    """It surfaced as `RuntimeError: theta is infeasible`."""
    df = frame()
    df["xd"] = df["x"]
    with pytest.raises(ValueError, match="rank deficient"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlm.mixedlm("y ~ x + xd", df, groups=df["g"]).fit()


def test_near_collinear_design_still_fits():
    """The threshold must refuse the unrepresentable, not the merely awkward."""
    rng = np.random.default_rng(5)
    df = frame()
    df["xn"] = df["x"] * (1 - 1e-3) + rng.normal(size=len(df)) * 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x + xn", df, groups=df["g"]).fit()
    assert np.isfinite(r.llf)
