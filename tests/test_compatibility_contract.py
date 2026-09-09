"""The compatibility contract in docs/COMPATIBILITY.md, as executable claims.

A migration table is only useful if it is true, and a table maintained by hand
drifts the moment someone adds an attribute. Each claim below corresponds to a
row of that document, so the document fails with the code rather than after it.

The statsmodels comparisons skip cleanly when statsmodels is absent.
"""

from __future__ import annotations

import pathlib
import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "COMPATIBILITY.md"


def frame(n=200, m=20, seed=1):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(size=m)[g] + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "x": x, "g": g})


DF = frame()


def ours():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit()


def theirs():
    smf = pytest.importorskip("statsmodels.formula.api")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm("y ~ x", DF, groups=DF["g"]).fit()


# ------------------------------------------------- 2. different return types
NDARRAY_WHERE_SERIES = [
    "fe_params", "params", "bse", "bse_fe", "bse_re",
    "tvalues", "pvalues", "fittedvalues", "resid",
]


@pytest.mark.parametrize("attr", NDARRAY_WHERE_SERIES)
def test_documented_ndarray_returns(attr):
    """Documented as ndarray where the reference returns a Series."""
    assert isinstance(getattr(ours(), attr), np.ndarray), (
        f"{attr} is documented as returning an ndarray")


@pytest.mark.parametrize("attr", NDARRAY_WHERE_SERIES)
def test_the_reference_really_does_return_a_series(attr):
    """The other half of the claim: if this changed, the doc is now wrong."""
    value = getattr(theirs(), attr)
    assert isinstance(value, pd.Series), (
        f"COMPATIBILITY.md says statsmodels returns a Series for {attr}, but "
        f"this version returns {type(value).__name__}")


@pytest.mark.parametrize("attr", ["cov_re", "cov_re_unscaled"])
def test_covariance_containers(attr):
    assert isinstance(getattr(ours(), attr), np.ndarray)
    assert isinstance(getattr(theirs(), attr), pd.DataFrame)


def test_name_based_access_raises_as_documented():
    r = ours()
    with pytest.raises((IndexError, KeyError, ValueError, TypeError)):
        _ = r.fe_params["x"]


def test_the_documented_replacements_work():
    r = ours()
    assert r.fe_params_labelled["x"] == pytest.approx(r.fe_params[1])
    assert r.params_labelled["Intercept"] == pytest.approx(r.params[0])
    assert r.param_names[:2] == ["Intercept", "x"]


def test_scalars_are_plain_python_types():
    r = ours()
    assert type(r.llf) is float
    assert type(r.scale) is float
    assert type(r.df_resid) is int


def test_summary_is_text_only():
    s = ours().summary()
    assert hasattr(s, "as_text")
    for absent in ("tables", "as_html", "as_latex"):
        assert not hasattr(s, absent), (
            f"summary() gained {absent}; COMPATIBILITY.md says it is text-only")


def test_score_and_hessian_are_in_theta_coordinates():
    """Documented as *not* comparable to the reference term by term."""
    model = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    theta = np.array([1.0])
    assert model.score(theta).shape == (model.k_re2,)
    assert model.hessian(theta).shape == (model.k_re2, model.k_re2)


def test_cov_params_has_nan_off_diagonal_in_the_variance_block():
    r = ours()
    cp = r.cov_params()
    p = r.k_fe
    assert np.all(np.isfinite(cp[:p, :p])), "the fixed-effect block is exact"
    off = cp[p:, :p]
    assert np.all(np.isnan(off)), (
        "COMPATIBILITY.md says there are no cross-block terms")


# ------------------------------------------------------------ 3. unsupported
@pytest.mark.parametrize("name", [
    "fit_regularized", "get_distribution", "get_scale",
])
def test_unsupported_model_methods_raise(name):
    model = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    with pytest.raises(NotImplementedError):
        getattr(model, name)()


@pytest.mark.parametrize("name", ["profile_re", "bootstrap", "get_distribution"])
def test_unsupported_result_methods_raise(name):
    with pytest.raises(NotImplementedError):
        getattr(ours(), name)()


ABSENT_ON_RESULTS = [
    "bsejac", "bsejhj", "covjac", "covjhj", "hessv", "hist", "score_obsv",
    "normalized_cov_params", "k_constant", "use_sqrt", "freepat", "cov_pen",
    "remove_data", "get_nlfun", "t_test_pairwise", "wald_test_terms",
]

ABSENT_ON_MODEL = [
    "get_fe_params", "score_full", "score_sqrt", "k_params", "k_constant",
    "n_totobs", "fe_pen", "re_pen", "row_indices",
    "endog_li", "exog_li", "exog_re_li", "exog_re2_li",
]


@pytest.mark.parametrize("name", ABSENT_ON_RESULTS)
def test_documented_absent_result_attributes_are_absent(name):
    """If one of these appears, the document is out of date -- not the code."""
    assert not hasattr(ours(), name), (
        f"{name} now exists; add it to the compatible section of "
        "COMPATIBILITY.md instead of leaving it listed as absent")


@pytest.mark.parametrize("name", ABSENT_ON_MODEL)
def test_documented_absent_model_attributes_are_absent(name):
    model = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    assert not hasattr(model, name)


def test_every_absent_name_is_actually_listed_in_the_document():
    text = DOC.read_text(encoding="utf-8")
    for name in ABSENT_ON_RESULTS + ABSENT_ON_MODEL:
        assert f"`{name}`" in text, (
            f"{name} is asserted absent here but not listed in "
            "COMPATIBILITY.md")


# ------------------------------------------------------------- 4. additions
@pytest.mark.parametrize("name", [
    "singular", "diagnostics", "bse_cov_re", "param_names",
    "params_labelled", "fe_params_labelled",
])
def test_documented_additions_exist(name):
    assert hasattr(ours(), name)


def test_additions_are_documented():
    text = DOC.read_text(encoding="utf-8")
    for name in ("singular", "diagnostics", "bse_cov_re", "param_names"):
        assert f"`results.{name}`" in text or f"`{name}`" in text


# ------------------------------------------------------ 1. compatible values
def test_the_compatible_section_really_is_compatible():
    """Values, not containers: the numbers must match the reference."""
    a, b = ours(), theirs()
    assert a.llf == pytest.approx(b.llf, abs=1e-6)
    assert np.allclose(a.fe_params, np.asarray(b.fe_params), rtol=1e-5)
    assert a.scale == pytest.approx(b.scale, rel=1e-5)
    assert np.allclose(np.asarray(a.cov_re), np.asarray(b.cov_re), rtol=1e-4)
    assert np.allclose(a.params, np.asarray(b.params), rtol=1e-4)
    assert a.nobs == b.nobs
    assert a.k_fe == b.k_fe and a.k_re == b.k_re and a.k_re2 == b.k_re2


def test_aic_and_bic_are_nan_under_reml_as_in_the_reference():
    a, b = ours(), theirs()
    assert np.isnan(a.aic) and np.isnan(b.aic)
    assert np.isnan(a.bic) and np.isnan(b.bic)


def test_version_reports_this_package_not_statsmodels():
    assert mlm.__version__ == "0.1.0"
    assert hasattr(mlm, "__statsmodels_version__")
