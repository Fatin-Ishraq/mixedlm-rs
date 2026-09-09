"""Representative public usage, type-checked rather than executed.

`py.typed` claims this package's annotations are usable. That claim needs a
checker run against real calls, not a count of annotated functions -- the count
was the mistake the first time round, when `py.typed` shipped against 0 of 42
annotated functions and merely suppressed mypy's "missing stubs" warning.

`mypy tests/typing_usage.py --strict` is run by the lint CI job. This file is
also imported by test_typing.py so that a syntax or attribute error here fails
the ordinary suite too.
"""

from __future__ import annotations

import mixedlm_rs as mlm
import numpy as np
import pandas as pd


def build_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, m = 200, 20
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(size=m)[g] + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def formula_path(df: pd.DataFrame) -> float:
    model = mlm.mixedlm("y ~ x", df, groups=df["g"], re_formula="~x")
    result = model.fit(reml=True, maxiter=300, gtol=1e-8, ftol=1e-12)
    return float(result.llf)


def array_path(df: pd.DataFrame) -> np.ndarray:
    endog = df["y"].to_numpy()
    exog = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    model = mlm.MixedLM(endog, exog, df["g"].to_numpy())
    return model.fit().fe_params


def results_surface(df: pd.DataFrame) -> None:
    result = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit()

    _fe: np.ndarray = result.fe_params
    _cov: np.ndarray = result.cov_re
    _scale: float = result.scale
    _params: np.ndarray = result.params
    _bse: np.ndarray = result.bse
    _bse_fe: np.ndarray = result.bse_fe
    _bse_re: np.ndarray = result.bse_re
    _bse_cov: np.ndarray = result.bse_cov_re
    _t: np.ndarray = result.tvalues
    _p: np.ndarray = result.pvalues
    _llf: float = result.llf
    _aic: float = result.aic
    _bic: float = result.bic
    _df: int = result.df_resid
    _dfm: int = result.df_modelwc
    _fitted: np.ndarray = result.fittedvalues
    _resid: np.ndarray = result.resid
    _names: list[str] = result.param_names
    _labelled: pd.Series = result.params_labelled
    _obj: mlm.MixedLMParams = result.params_object

    _ci: np.ndarray = result.conf_int(alpha=0.05)
    _cp: np.ndarray = result.cov_params()
    _pred: np.ndarray = result.predict(pd.DataFrame({"x": [1.0]}))
    _pred2: np.ndarray = result.predict(np.ones((2, 2)), transform=False)

    ranef: dict[object, pd.Series] = result.random_effects
    ranef_cov: dict[object, pd.DataFrame] = result.random_effects_cov
    assert len(ranef) == len(ranef_cov)

    diagnostics: dict[str, object] = result.diagnostics
    assert "converged" in diagnostics


def hypothesis_tests(df: pd.DataFrame) -> None:
    result = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit()
    contrast = np.eye(2)
    result.t_test(contrast)
    result.wald_test(contrast, use_f=False)
    result.f_test(contrast)
    result.t_test("x = 0")


def parameter_containers() -> mlm.MixedLMParams:
    params = mlm.MixedLMParams.from_components(
        fe_params=np.zeros(2), cov_re=np.eye(1))
    packed: np.ndarray = params.get_packed(use_sqrt=True, has_fe=False)
    restored = mlm.MixedLMParams.from_packed(
        packed, k_fe=2, k_re=1, use_sqrt=True, has_fe=False)
    return restored.copy()


def likelihood_surface(df: pd.DataFrame) -> None:
    model = mlm.mixedlm("y ~ x", df, groups=df["g"])
    theta = np.array([1.0])
    _v: float = model.loglike(theta)
    _g: np.ndarray = model.score(theta)
    _h: np.ndarray = model.hessian(theta)
    _i: np.ndarray = model.information(theta)


def model_metadata(df: pd.DataFrame) -> None:
    model = mlm.mixedlm("y ~ x", df, groups=df["g"])
    _xn: list[str] = model.data.xnames
    _yn: str = model.data.ynames
    _rn: list[str] = model.exog_re_names
    _pn: list[str] = model.data.param_names
    _split: list[np.ndarray] = model.group_list(df["y"].to_numpy())


def persistence(df: pd.DataFrame, path: str) -> None:
    result = mlm.mixedlm("y ~ x", df, groups=df["g"]).fit()
    result.save(path)
    result.save(path, with_data=False)
    restored: mlm.MixedLMResults = mlm.MixedLMResults.load(path)
    assert restored.converged in (True, False)


def aliasing() -> None:
    installed: bool = mlm.install(strict=False)
    if installed:
        mlm.uninstall()
    _state: bool = mlm.is_installed()


def native_core(df: pd.DataFrame) -> float:
    """The low-level interface, which the stub has to describe usefully."""
    endog = np.ascontiguousarray(df["y"].to_numpy(dtype=np.float64))
    exog = np.ascontiguousarray(
        np.column_stack([np.ones(len(df)), df["x"].to_numpy()]))
    re_design = np.ascontiguousarray(np.ones((len(df), 1)))
    codes = np.ascontiguousarray(df["g"].to_numpy(dtype=np.int64))

    core = mlm.LmmCore(endog, exog, re_design, codes, int(df["g"].nunique()))
    n_theta: int = core.n_theta
    assert n_theta == 1
    value, gradient = core.deviance_grad(core.default_theta(), True)
    _bounds: list[float] = core.lower_bounds()
    assert len(gradient) == n_theta
    return float(value)
