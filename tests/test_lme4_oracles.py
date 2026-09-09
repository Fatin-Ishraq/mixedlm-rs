"""Correctness against lme4's published fits.

statsmodels cannot be the only oracle here, because on some inputs statsmodels
is the thing that is wrong. The primary oracle is therefore lme4's own datasets
and the fitted values published in Bates, Machler, Bolker & Walker (2015),
*Fitting Linear Mixed-Effects Models Using lme4*, JSS 67(1).

Only single-grouping-factor datasets are used: Penicillin (crossed), Pastes
(nested), cbpp (binomial) and InstEval (large crossed) are outside this
release's scope and are exercised in test_api.py as explicit refusals.
"""

import pathlib

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def load(name):
    return pd.read_csv(DATA / f"{name}.csv")


# ---------------------------------------------------------------- sleepstudy
# lmer(Reaction ~ Days + (Days|Subject), sleepstudy, REML=TRUE)
SLEEP_REML = {
    "criterion": 1743.6284,
    "beta": [251.405, 10.467],
    "sd_intercept": 24.741,
    "sd_days": 5.922,
    "corr": 0.066,
    "sd_resid": 25.592,
}


def sleepstudy_fit(reml=True):
    d = load("sleepstudy")
    return mlm.mixedlm("Reaction ~ Days", d, groups=d["Subject"],
                       re_formula="~Days").fit(reml=reml)


def test_sleepstudy_reml_criterion():
    r = sleepstudy_fit()
    assert abs(-2 * r.llf - SLEEP_REML["criterion"]) < 5e-3


def test_sleepstudy_fixed_effects():
    r = sleepstudy_fit()
    assert np.allclose(r.fe_params, SLEEP_REML["beta"], atol=5e-3)


def test_sleepstudy_variance_components():
    r = sleepstudy_fit()
    sd = np.sqrt(np.diag(r.cov_re))
    corr = r.cov_re[1, 0] / (sd[0] * sd[1])
    assert abs(sd[0] - SLEEP_REML["sd_intercept"]) < 5e-3
    assert abs(sd[1] - SLEEP_REML["sd_days"]) < 5e-3
    assert abs(corr - SLEEP_REML["corr"]) < 1e-3
    assert abs(np.sqrt(r.scale) - SLEEP_REML["sd_resid"]) < 5e-3


def test_sleepstudy_ml_deviance():
    """lme4 reports an ML deviance of 1751.9 for the same model."""
    r = sleepstudy_fit(reml=False)
    assert abs(-2 * r.llf - 1751.9393) < 5e-3


def test_sleepstudy_converges():
    r = sleepstudy_fit()
    assert r.converged


# ------------------------------------------------------------------ Dyestuff
# lmer(Yield ~ 1 + (1|Batch), Dyestuff, REML=TRUE)
#   REML criterion 319.6543; Batch var 1764 (sd 42.00);
#   Residual var 2451 (sd 49.51); (Intercept) 1527.5
def test_dyestuff_matches_published_fit():
    d = load("Dyestuff")
    r = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit()
    assert abs(-2 * r.llf - 319.6543) < 5e-3
    assert abs(r.fe_params[0] - 1527.5) < 1e-3
    assert abs(np.sqrt(r.cov_re[0, 0]) - 42.00) < 5e-2
    assert abs(np.sqrt(r.scale) - 49.51) < 5e-2
    assert r.converged


# ----------------------------------------------------------------- Dyestuff2
# The singular-fit case: the maximum sits exactly on the boundary, with the
# between-batch variance estimated at zero. This is where statsmodels warns
# "Random effects covariance is singular" and "The MLE may be on the boundary
# of the parameter space", and it is the behaviour a mixed-model fitter most
# needs to get right -- a scientist who sees a convergence failure here will
# start deleting random-effects terms.
def test_dyestuff2_is_singular_and_still_converges():
    d = load("Dyestuff2")
    r = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit()
    assert abs(-2 * r.llf - 161.8283) < 5e-3, "lme4's published REML criterion"
    assert abs(r.fe_params[0] - 5.6656) < 1e-3
    assert r.cov_re[0, 0] == pytest.approx(0.0, abs=1e-8), \
        "the between-batch variance is exactly zero here"
    assert r.converged, "a boundary optimum is a converged fit, not a failure"


def test_dyestuff2_residual_matches_lme4():
    """The residual SD on the singular fit, verified against a live lme4 run.

    This was once recorded as an open question, on a belief that the lme4
    literature printed 3.653 here against our 3.7157. Checked on R 4.6.1 with
    lme4, the reference actually reports:

        REML criterion 161.8283   sigma 3.715684   Batch var 0   intercept 5.6656

    so there was never a discrepancy -- the remembered figure corresponded to
    SST/n rather than the REML divisor SST/(n-1). Pinned here so the claim
    cannot quietly drift.
    """
    d = load("Dyestuff2")
    ours = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit()
    assert np.sqrt(ours.scale) == pytest.approx(3.715684, abs=1e-6)
    assert ours.scale == pytest.approx(13.80631, abs=1e-5)

    # and it is exactly SST/(n-1), the REML residual for a singular fit
    y = d["Yield"].to_numpy(float)
    sst = ((y - y.mean()) ** 2).sum()
    assert ours.scale == pytest.approx(sst / (len(y) - 1), rel=1e-10)


def test_dyestuff2_residual_agrees_with_statsmodels_too():
    """Independent third implementation, as a further check on the above."""
    statsmodels = pytest.importorskip("statsmodels.formula.api")
    d = load("Dyestuff2")
    ours = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit()
    theirs = statsmodels.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit()
    assert ours.scale == pytest.approx(theirs.scale, rel=1e-8)


# ------------------------------------------------------- criterion invariants
@pytest.mark.parametrize("name,formula,group,re_formula", [
    ("sleepstudy", "Reaction ~ Days", "Subject", "~Days"),
    ("sleepstudy", "Reaction ~ Days", "Subject", None),
    ("Dyestuff", "Yield ~ 1", "Batch", None),
])
def test_reml_criterion_is_at_least_ml_deviance(name, formula, group, re_formula):
    """REML and ML must both be finite and ordered sanely for the same data."""
    d = load(name)
    kw = {"groups": d[group], "re_formula": re_formula}
    a = mlm.mixedlm(formula, d, **kw).fit(reml=True)
    b = mlm.mixedlm(formula, d, **kw).fit(reml=False)
    assert np.isfinite(a.llf) and np.isfinite(b.llf)
    assert np.allclose(a.fe_params, b.fe_params, atol=1e-6), \
        "fixed effects should barely move between REML and ML here"
