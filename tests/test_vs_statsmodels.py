"""Differential tests against statsmodels.

The standard used by the earlier packages in this family -- "identical output,
proven bug-for-bug" -- cannot be applied wholesale here, because on some inputs
statsmodels does not converge and returns estimates anyway. So the bar is split:

  * where statsmodels converges, we must agree with it closely;
  * where it does not, we must converge, and we assert that rather than
    comparing against a fit that did not happen.

Every fixture records which branch it took, so the split is visible rather than
hidden behind a loose tolerance.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mixedlm_rs as mlm

warnings.filterwarnings("ignore")
smf = pytest.importorskip("statsmodels.formula.api")


def make(ngroups, nper, seed, q=2, re_sd=1.0, resid_sd=0.5, unbalanced=False):
    rng = np.random.default_rng(seed)
    if unbalanced:
        sizes = rng.integers(max(2, nper // 2), nper * 2, ngroups)
    else:
        sizes = np.full(ngroups, nper)
    codes = np.repeat(np.arange(ngroups), sizes)
    n = len(codes)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    b0 = rng.standard_normal(ngroups) * re_sd
    y = 1 + 2 * x1 - 0.5 * x2 + b0[codes] + rng.standard_normal(n) * resid_sd
    if q == 2:
        b1 = rng.standard_normal(ngroups) * re_sd * 0.6
        y = y + b1[codes] * x1
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=codes))


def fit_both(df, re_formula="~x1", reml=True):
    ours = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                       re_formula=re_formula).fit(reml=reml)
    theirs = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                         re_formula=re_formula).fit(reml=reml)
    return ours, theirs


CASES = [
    # (ngroups, nper, seed, re_formula, unbalanced)
    (30, 10, 0, "~x1", False),
    (50, 8, 1, "~x1", False),
    (40, 12, 2, None, False),
    (60, 6, 3, None, False),
    (25, 20, 4, "~x1", True),
    (80, 5, 5, "~x1", True),
    (100, 4, 6, None, True),
    (35, 15, 7, "~x1", False),
]


@pytest.mark.parametrize("ngroups,nper,seed,re_formula,unbal", CASES)
@pytest.mark.parametrize("reml", [True, False])
def test_agrees_where_statsmodels_converges(ngroups, nper, seed, re_formula,
                                            unbal, reml):
    q = 2 if re_formula else 1
    df = make(ngroups, nper, seed, q=q, unbalanced=unbal)
    ours, theirs = fit_both(df, re_formula, reml)

    if not theirs.converged:
        # The reference failed. We must not.
        assert ours.converged, (
            "statsmodels did not converge and neither did we -- this is the "
            "case the package exists to fix")
        pytest.skip("statsmodels did not converge; nothing to compare against")

    assert ours.converged
    assert np.allclose(ours.fe_params, theirs.fe_params, rtol=1e-4, atol=1e-5), \
        f"fe_params {ours.fe_params} vs {theirs.fe_params}"
    assert ours.scale == pytest.approx(theirs.scale, rel=1e-4)
    # Compare on the scale of the matrix itself. A relative tolerance is
    # meaningless for a near-zero covariance term, and the two optimisers stop
    # at slightly different points on what is a very flat surface there.
    tcov = np.asarray(theirs.cov_re, float)
    tol = 1e-4 * max(float(np.max(np.abs(np.diag(tcov)))), 1.0)
    assert np.allclose(ours.cov_re, tcov, rtol=1e-3, atol=tol), \
        f"cov_re\n{ours.cov_re}\nvs\n{tcov}  (atol={tol:.2e})"
    assert ours.llf == pytest.approx(theirs.llf, rel=1e-6)


@pytest.mark.parametrize("ngroups,nper,seed,re_formula,unbal", CASES[:5])
def test_standard_errors_agree(ngroups, nper, seed, re_formula, unbal):
    q = 2 if re_formula else 1
    df = make(ngroups, nper, seed, q=q, unbalanced=unbal)
    ours, theirs = fit_both(df, re_formula)
    if not theirs.converged:
        pytest.skip("statsmodels did not converge")
    # Fixed-effect SEs are exact on both sides.
    assert np.allclose(ours.bse_fe, np.asarray(theirs.bse)[:ours.k_fe],
                       rtol=2e-3), f"{ours.bse_fe} vs {np.asarray(theirs.bse)[:ours.k_fe]}"
    # Variance-component SEs come from different parameterisations; require
    # agreement to a few percent rather than to machine precision.
    ours_re = ours.bse_re
    theirs_re = np.asarray(theirs.bse)[ours.k_fe:ours.k_fe + len(ours_re)]
    good = np.isfinite(ours_re) & np.isfinite(theirs_re)
    if good.any():
        assert np.allclose(ours_re[good], theirs_re[good], rtol=0.05), \
            f"{ours_re} vs {theirs_re}"


@pytest.mark.parametrize("ngroups,nper,seed,re_formula,unbal", CASES[:4])
def test_params_packing_matches(ngroups, nper, seed, re_formula, unbal):
    """The packed params vector must have statsmodels' layout and length."""
    q = 2 if re_formula else 1
    df = make(ngroups, nper, seed, q=q, unbalanced=unbal)
    ours, theirs = fit_both(df, re_formula)
    if not theirs.converged:
        pytest.skip("statsmodels did not converge")
    assert ours.params.shape == np.asarray(theirs.params).shape
    assert np.allclose(ours.params, np.asarray(theirs.params),
                       rtol=1e-3, atol=1e-6), \
        f"{ours.params} vs {np.asarray(theirs.params)}"


def test_random_effects_agree():
    df = make(40, 10, 11, q=2)
    ours, theirs = fit_both(df)
    if not theirs.converged:
        pytest.skip("statsmodels did not converge")
    ore = ours.random_effects
    tre = theirs.random_effects
    assert set(ore) == set(tre)
    for k in list(ore)[:20]:
        assert np.allclose(np.asarray(ore[k], float),
                           np.asarray(tre[k], float), atol=1e-4), \
            f"group {k}: {ore[k].to_numpy()} vs {np.asarray(tre[k])}"


def test_fitted_and_resid_agree():
    df = make(30, 10, 12, q=2)
    ours, theirs = fit_both(df)
    if not theirs.converged:
        pytest.skip("statsmodels did not converge")
    assert np.allclose(ours.fittedvalues, theirs.fittedvalues, atol=1e-5)
    assert np.allclose(ours.resid, theirs.resid, atol=1e-5)


def test_we_converge_where_the_reference_does_not():
    """The headline claim, asserted rather than described.

    This fixture is the n=20,000 / 1,000-group case from the project brief: the
    reference retries bfgs, lbfgs and cg, then gives up with a gradient norm in
    the hundreds and returns estimates anyway.
    """
    df = make(1000, 20, 0, q=2)
    ours = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                       re_formula="~x1").fit()
    theirs = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                         re_formula="~x1").fit()
    assert ours.converged, "we must converge on the fixture that motivated the project"
    if theirs.converged:
        pytest.skip("statsmodels converged here on this platform/version")
    # The estimates should still be close: statsmodels lands near the optimum,
    # it just cannot certify it.
    assert np.allclose(ours.fe_params, theirs.fe_params, atol=5e-3)
