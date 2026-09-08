"""API surface, aliasing, and the things this release deliberately refuses.

A drop-in that hides its gaps is worse than one that does not have them, so the
unimplemented paths are tested to *raise* rather than to silently fit a
different model.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import mixedlm_rs as mlm
from mixedlm_rs import MixedLM, MixedLMParams, MixedLMResults, VCSpec

warnings.filterwarnings("ignore")


def toy(ngroups=20, nper=8, seed=0, q=2):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    y = 1 + 2 * x1 + rng.standard_normal(ngroups)[codes] + rng.standard_normal(n) * 0.5
    return pd.DataFrame(dict(y=y, x1=x1, g=codes))


# ------------------------------------------------------------------ surface
def test_public_names_exist():
    for name in ["MixedLM", "MixedLMResults", "MixedLMParams", "VCSpec",
                 "mixedlm", "install", "uninstall", "is_installed",
                 "ConvergenceWarning", "__version__"]:
        assert hasattr(mlm, name), name


def test_results_attributes_match_the_reference_surface():
    """Every MixedLMResults attribute we claim to support must exist and work."""
    df = toy()
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    for attr in ["fe_params", "cov_re", "cov_re_unscaled", "scale", "params",
                 "bse", "bse_fe", "bse_re", "tvalues", "pvalues", "llf",
                 "aic", "bic", "converged", "random_effects",
                 "random_effects_cov", "fittedvalues", "resid", "nobs",
                 "k_fe", "k_re", "k_re2", "k_vc", "method", "df_modelwc"]:
        getattr(r, attr)
    assert r.conf_int().shape == (r.params.size, 2)
    assert r.cov_params().shape == (r.params.size, r.params.size)
    assert isinstance(str(r.summary()), str)
    assert r.predict().shape == (len(df),)


def test_model_attributes_exist():
    df = toy()
    m = MixedLM.from_formula("y ~ x1", df, groups=df["g"], re_formula="~x1")
    for attr in ["endog", "exog", "exog_re", "groups", "k_fe", "k_re", "k_re2",
                 "k_vc", "nobs", "exog_names", "n_groups", "group_labels",
                 "group_list", "endog_names"]:
        getattr(m, attr)


def test_reml_gives_nan_aic_bic_like_the_reference():
    df = toy()
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit(reml=True)
    assert np.isnan(r.aic) and np.isnan(r.bic)
    rml = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit(reml=False)
    assert np.isfinite(rml.aic) and np.isfinite(rml.bic)


def test_summary_renders_expected_labels():
    df = toy()
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    txt = str(r.summary())
    for token in ["Mixed Linear Model Regression Results", "Coef.", "Std.Err.",
                  "Intercept", "x1", "Group Var", "No. Groups:", "Converged:"]:
        assert token in txt, token


# --------------------------------------------------------- MixedLMParams
def test_params_packing_round_trip():
    cov = np.array([[2.0, 0.3], [0.3, 1.5]])
    fe = np.array([1.0, -2.0, 0.5])
    p = MixedLMParams.from_components(fe_params=fe, cov_re=cov)
    packed = p.get_packed(use_sqrt=True, has_fe=True)
    back = MixedLMParams.from_packed(packed, k_fe=3, k_re=2, use_sqrt=True)
    assert np.allclose(back.fe_params, fe)
    assert np.allclose(back.cov_re, cov)
    assert p.k_fe == 3 and p.k_re == 2 and p.k_re2 == 3


def test_params_copy_is_independent():
    p = MixedLMParams.from_components(fe_params=np.ones(2), cov_re=np.eye(2))
    c = p.copy()
    c.fe_params[0] = 99.0
    assert p.fe_params[0] == 1.0


# ----------------------------------------------------- deliberate refusals
def test_variance_components_are_refused_not_ignored():
    df = toy()
    with pytest.raises(NotImplementedError, match="variance components"):
        MixedLM(df["y"], np.ones((len(df), 1)), df["g"],
                exog_vc=VCSpec(["a"], [np.ones((len(df), 1))]))
    with pytest.raises(NotImplementedError, match="vc_formula"):
        MixedLM.from_formula("y ~ x1", df, groups=df["g"], vc_formula={"a": "0 + C(g)"})


@pytest.mark.parametrize("kw", ["fe_pen", "cov_pen", "free"])
def test_penalty_arguments_are_refused(kw):
    df = toy()
    m = MixedLM.from_formula("y ~ x1", df, groups=df["g"])
    with pytest.raises(NotImplementedError, match=kw):
        m.fit(**{kw: object()})


@pytest.mark.parametrize("meth", ["profile_re", "bootstrap", "get_distribution"])
def test_unimplemented_results_methods_raise(meth):
    df = toy()
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    with pytest.raises(NotImplementedError):
        getattr(r, meth)()


def test_vcspec_can_be_constructed():
    """The name must import and construct so dependent code keeps working."""
    v = VCSpec(["a"], [np.ones((3, 1))])
    assert v.names == ["a"]


# ------------------------------------------------------------ input checks
def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MixedLM(np.zeros(10), np.ones((10, 1)), np.arange(9))


def test_non_finite_values_are_reported_not_silently_dropped():
    y = np.zeros(10)
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        MixedLM(y, np.ones((10, 1)), np.repeat(np.arange(5), 2))


def test_missing_drop_removes_rows():
    y = np.arange(10.0)
    y[3] = np.nan
    m = MixedLM(y, np.ones((10, 1)), np.repeat(np.arange(5), 2), missing="drop")
    assert m.nobs == 9


def test_intercept_only_random_effect_works():
    df = toy(q=1)
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert r.k_re == 1 and r.cov_re.shape == (1, 1)
    assert r.converged


def test_single_group_is_handled():
    """Degenerate but legal: one group means the random effect is unidentified."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(dict(y=rng.standard_normal(n), x1=rng.standard_normal(n),
                           g=np.zeros(n, dtype=int)))
    r = mlm.mixedlm("y ~ x1", df, groups=df["g"]).fit()
    assert np.isfinite(r.llf)


# ----------------------------------------------------------------- install
def test_install_and_uninstall_round_trip():
    smm = pytest.importorskip("statsmodels.regression.mixed_linear_model")
    import statsmodels.formula.api as smf

    original = smm.MixedLM
    original_fn = smf.mixedlm
    try:
        assert mlm.install() is True
        assert mlm.is_installed()
        assert smm.MixedLM is mlm.MixedLM
        assert smf.mixedlm is mlm.mixedlm
        assert mlm.install() is False, "installing twice is a no-op"
    finally:
        mlm.uninstall()
    assert smm.MixedLM is original
    assert smf.mixedlm is original_fn
    assert not mlm.is_installed()


def test_install_leaves_the_rest_of_statsmodels_alone():
    """We alias mixed-model entry points only; statsmodels does much more."""
    sm = pytest.importorskip("statsmodels.api")
    ols_before = sm.OLS
    try:
        mlm.install()
        assert sm.OLS is ols_before
        assert sm.GLM is not None
    finally:
        mlm.uninstall()


# ------------------------------------------------------------- reproducibility
def test_fit_is_deterministic():
    df = toy(30, 10, seed=3)
    a = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    b = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    assert np.array_equal(a.params, b.params)
    assert a.llf == b.llf


def test_rust_optimiser_path_agrees_with_the_default():
    df = toy(30, 10, seed=4)
    a = mlm.mixedlm("y ~ x1", df, groups=df["g"], re_formula="~x1").fit()
    b = mlm.mixedlm("y ~ x1", df, groups=df["g"],
                    re_formula="~x1").fit(method="rust")
    assert np.allclose(a.fe_params, b.fe_params, atol=1e-6)
    assert a.llf == pytest.approx(b.llf, rel=1e-8)
