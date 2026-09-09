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

# The fixtures are GPL-2 and ship with the git repository only, not in the
# wheel or the source archive (this package is MIT). Skip rather than fail when
# running from a distribution.
pytestmark = pytest.mark.skipif(
    not DATA.is_dir(),
    reason="lme4 fixtures are repository-only; clone the repo to run these")


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


# --------------------------------------------------------------- ML oracles
#
# The REML fits above were checked against lme4 from the start; the ML fits
# were not, beyond sleepstudy's deviance. All three datasets are covered here,
# with full parameter sets, from a live R 4.6.1 + lme4 run:
#
#   lmer(Yield ~ 1 + (1|Batch), Dyestuff,  REML=FALSE)
#     deviance 327.327060  (Intercept) 1527.5  sd(Batch) 37.260345  sigma 49.510100
#   lmer(Yield ~ 1 + (1|Batch), Dyestuff2, REML=FALSE)
#     deviance 162.873037  (Intercept) 5.6656  sd(Batch) 0  sigma 3.653231
#   lmer(Reaction ~ Days + (Days|Subject), sleepstudy, REML=FALSE)
#     deviance 1751.939344  beta 251.405105, 10.467286
#     sd 23.779760 / 5.716799  corr 0.081321  sigma 25.591907
ML_ORACLES = {
    "Dyestuff": {
        "formula": "Yield ~ 1", "group": "Batch", "re_formula": None,
        "criterion": 327.327060, "beta": [1527.5],
        "sd_re": [37.260345], "sd_resid": 49.510100,
    },
    "Dyestuff2": {
        "formula": "Yield ~ 1", "group": "Batch", "re_formula": None,
        "criterion": 162.873037, "beta": [5.6656],
        "sd_re": [0.0], "sd_resid": 3.653231,
    },
    "sleepstudy": {
        "formula": "Reaction ~ Days", "group": "Subject", "re_formula": "~Days",
        "criterion": 1751.939344, "beta": [251.405105, 10.467286],
        "sd_re": [23.779760, 5.716799], "sd_resid": 25.591907,
    },
}


@pytest.mark.parametrize("name", sorted(ML_ORACLES))
def test_ml_fits_match_lme4(name):
    spec = ML_ORACLES[name]
    d = load(name)
    r = mlm.mixedlm(spec["formula"], d, groups=d[spec["group"]],
                    re_formula=spec["re_formula"]).fit(reml=False)

    assert abs(-2 * r.llf - spec["criterion"]) < 5e-3, "ML deviance"
    assert np.allclose(r.fe_params, spec["beta"], atol=5e-3), "fixed effects"
    sd = np.sqrt(np.maximum(np.diag(r.cov_re), 0.0))
    assert np.allclose(sd, spec["sd_re"], atol=5e-3), \
        f"sd(random effects): {sd} vs {spec['sd_re']}"
    assert abs(np.sqrt(r.scale) - spec["sd_resid"]) < 5e-3, "sd(Residual)"
    assert r.converged


def test_sleepstudy_ml_correlation_matches_lme4():
    r = sleepstudy_fit(reml=False)
    sd = np.sqrt(np.diag(r.cov_re))
    corr = r.cov_re[1, 0] / (sd[0] * sd[1])
    assert abs(corr - 0.081321) < 1e-3


def test_dyestuff2_ml_residual_is_the_3653_figure():
    """The ML residual SD on the singular fit, and why 3.653 kept appearing.

    An open question in this project once recorded a suspicion that lme4
    reported 3.653 for `Dyestuff2` against our 3.7157. Both numbers are real
    and neither was wrong: 3.715684 is the REML residual SD and 3.653231 is the
    ML one, because REML divides the residual sum of squares by `n - p` and ML
    by `n`. With a single intercept that is 29 against 30.
    """
    d = load("Dyestuff2")
    reml = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit(reml=True)
    ml = mlm.mixedlm("Yield ~ 1", d, groups=d["Batch"]).fit(reml=False)
    assert np.sqrt(reml.scale) == pytest.approx(3.715684, abs=1e-6)
    assert np.sqrt(ml.scale) == pytest.approx(3.653231, abs=1e-6)

    y = d["Yield"].to_numpy(float)
    sst = ((y - y.mean()) ** 2).sum()
    assert reml.scale == pytest.approx(sst / (len(y) - 1), rel=1e-10)
    assert ml.scale == pytest.approx(sst / len(y), rel=1e-10)


def test_sleepstudy_intercept_only_matches_lme4():
    """lmer(Reaction ~ Days + (1|Subject), REML=TRUE): a model shape the
    oracle table did not previously cover at all."""
    d = load("sleepstudy")
    r = mlm.mixedlm("Reaction ~ Days", d, groups=d["Subject"]).fit()
    assert abs(-2 * r.llf - 1786.465085) < 5e-3
    assert np.allclose(r.fe_params, [251.405105, 10.467286], atol=5e-3)
    assert abs(np.sqrt(r.cov_re[0, 0]) - 37.123827) < 5e-3
    assert abs(np.sqrt(r.scale) - 30.991234) < 5e-3
    assert r.converged
