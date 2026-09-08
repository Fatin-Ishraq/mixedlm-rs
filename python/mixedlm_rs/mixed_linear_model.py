"""Drop-in replacement for ``statsmodels.regression.mixed_linear_model``.

The public shapes -- ``MixedLM``, ``MixedLMResults``, ``MixedLMParams``, the
``params`` packing, ``bse``, ``summary()`` -- follow statsmodels so that changing
the import is the whole migration. The fitting underneath is the lme4 profiled
REML formulation over a block-diagonal Cholesky (see ``_fit.py`` and
``docs/DESIGN.md``).

Where this deliberately differs from statsmodels, it is because statsmodels is
wrong rather than merely different; every such case is documented in
``docs/CORRECTNESS.md`` and pinned by a test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._fit import ConvergenceWarning, fit_core, theta_to_lambda

__all__ = [
    "MixedLM", "MixedLMResults", "MixedLMParams", "VCSpec",
    "ConvergenceWarning",
]


class VCSpec:
    """Variance-component specification.

    Present so that code importing the name keeps working. Variance components
    beyond a single grouping factor are not implemented in this release; see
    ``docs/LIMITATIONS.md``. Constructing one is fine; passing it to ``MixedLM``
    raises ``NotImplementedError`` rather than silently fitting a different model.
    """

    def __init__(self, names, mats, colnames=None):
        self.names = list(names)
        self.mats = list(mats)
        self.colnames = colnames


class MixedLMParams:
    """Packed parameter container, matching statsmodels' layout.

    The packed vector is ``[fe_params, vech_row(cov_re_unscaled), vcomp]`` where
    ``vech_row`` walks the lower triangle by rows: (0,0), (1,0), (1,1), ...
    """

    def __init__(self, k_fe, k_re, k_vc):
        self.k_fe = int(k_fe)
        self.k_re = int(k_re)
        self.k_re2 = int(k_re * (k_re + 1) // 2)
        self.k_vc = int(k_vc)
        self.fe_params = np.zeros(self.k_fe)
        self.cov_re = np.eye(self.k_re) if self.k_re else np.zeros((0, 0))
        self.vcomp = np.zeros(self.k_vc)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_components(cls, fe_params=None, cov_re=None, cov_re_sqrt=None,
                        vcomp=None):
        if cov_re is None and cov_re_sqrt is not None:
            cov_re = np.asarray(cov_re_sqrt) @ np.asarray(cov_re_sqrt).T
        fe_params = np.zeros(0) if fe_params is None else np.asarray(fe_params, float)
        cov_re = np.zeros((0, 0)) if cov_re is None else np.asarray(cov_re, float)
        vcomp = np.zeros(0) if vcomp is None else np.asarray(vcomp, float)
        obj = cls(len(fe_params), cov_re.shape[0], len(vcomp))
        obj.fe_params = fe_params
        obj.cov_re = cov_re
        obj.vcomp = vcomp
        return obj

    @classmethod
    def from_packed(cls, params, k_fe, k_re, use_sqrt=True, has_fe=True):
        params = np.asarray(params, float)
        k_re2 = k_re * (k_re + 1) // 2
        i = 0
        if has_fe:
            fe = params[:k_fe]
            i = k_fe
        else:
            fe = np.zeros(k_fe)
        tri = params[i:i + k_re2]
        vcomp = params[i + k_re2:]
        mat = np.zeros((k_re, k_re))
        k = 0
        for r in range(k_re):
            for c in range(r + 1):
                mat[r, c] = tri[k]
                k += 1
        cov = mat @ mat.T if use_sqrt else mat + mat.T - np.diag(np.diag(mat))
        return cls.from_components(fe_params=fe, cov_re=cov, vcomp=vcomp)

    def get_packed(self, use_sqrt=True, has_fe=True):
        cov = np.asarray(self.cov_re, float)
        if use_sqrt and cov.size:
            mat = np.linalg.cholesky(cov)
        else:
            mat = cov
        tri = np.array([mat[r, c] for r in range(self.k_re) for c in range(r + 1)])
        parts = []
        if has_fe:
            parts.append(np.asarray(self.fe_params, float))
        parts.append(tri)
        parts.append(np.asarray(self.vcomp, float))
        return np.concatenate([p for p in parts if p.size or True])

    def copy(self):
        return MixedLMParams.from_components(
            fe_params=self.fe_params.copy(),
            cov_re=self.cov_re.copy(),
            vcomp=self.vcomp.copy(),
        )


def _vech_row(mat):
    k = mat.shape[0]
    return np.array([mat[r, c] for r in range(k) for c in range(r + 1)])


def _codes_from_groups(groups):
    g = np.asarray(groups)
    uniq, codes = np.unique(g, return_inverse=True)
    return uniq, codes.astype(np.int64)


class MixedLM:
    """Linear mixed effects model.

    Parameters mirror ``statsmodels.regression.mixed_linear_model.MixedLM``.
    """

    def __init__(self, endog, exog, groups, exog_re=None, exog_vc=None,
                 use_sqrt=True, missing="none", **kwargs):
        if exog_vc is not None:
            raise NotImplementedError(
                "variance components (exog_vc / vc_formula) are not implemented "
                "in this release. Refusing rather than silently fitting a "
                "different model; see docs/LIMITATIONS.md."
            )

        self.exog_names = kwargs.pop("exog_names", None)
        self._endog_name = kwargs.pop("endog_name", None) or "y"
        self.data_frame = kwargs.pop("_data_frame", None)
        self.formula = kwargs.pop("formula", None)
        self.re_formula = kwargs.pop("re_formula", None)
        self._exog_re_names = kwargs.pop("exog_re_names", None)

        endog = np.asarray(endog, float).ravel()
        exog = np.asarray(exog, float)
        if exog.ndim == 1:
            exog = exog[:, None]

        if exog_re is None:
            exog_re = np.ones((len(endog), 1))
        exog_re = np.asarray(exog_re, float)
        if exog_re.ndim == 1:
            exog_re = exog_re[:, None]

        groups = np.asarray(groups)
        if missing == "drop":
            ok = np.isfinite(endog) & np.all(np.isfinite(exog), 1) \
                & np.all(np.isfinite(exog_re), 1)
            endog, exog, exog_re, groups = endog[ok], exog[ok], exog_re[ok], groups[ok]
        elif not (np.all(np.isfinite(endog)) and np.all(np.isfinite(exog))
                  and np.all(np.isfinite(exog_re))):
            raise ValueError("endog/exog contain non-finite values; "
                             "pass missing='drop' to remove those rows")

        if not (len(endog) == exog.shape[0] == exog_re.shape[0] == len(groups)):
            raise ValueError("endog, exog, exog_re and groups must agree on length")

        self.endog = endog
        self.exog = exog
        self.exog_re = exog_re
        self.exog_vc = None
        self.groups = groups
        self.use_sqrt = use_sqrt

        self.group_labels, self._codes = _codes_from_groups(groups)
        self.n_groups = len(self.group_labels)

        self.k_fe = exog.shape[1]
        self.k_re = exog_re.shape[1]
        self.k_re2 = self.k_re * (self.k_re + 1) // 2
        self.k_vc = 0
        self.nobs = len(endog)

        if self.exog_names is None:
            self.exog_names = [f"x{i}" for i in range(self.k_fe)]
        if self._exog_re_names is None:
            self._exog_re_names = [f"z{i}" for i in range(self.k_re)]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None,
                     subset=None, use_sparse=False, missing="none",
                     *args, **kwargs):
        if vc_formula is not None:
            raise NotImplementedError(
                "vc_formula (variance components) is not implemented in this "
                "release; see docs/LIMITATIONS.md."
            )
        try:
            from patsy import dmatrices, dmatrix
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the formula interface needs patsy: pip install 'mixedlm-rs[formula]'"
            ) from exc

        groups = kwargs.pop("groups", None)
        if groups is None:
            raise ValueError("from_formula requires groups=")
        if subset is not None:
            data = data.loc[subset]

        if isinstance(groups, str):
            groups_arr = np.asarray(data[groups])
        else:
            groups_arr = np.asarray(groups)

        na_action = "drop" if missing == "drop" else "raise"
        y, X = dmatrices(formula, data, return_type="dataframe",
                         NA_action=na_action)
        if re_formula is None or str(re_formula).strip() in ("1", "~1", ""):
            Zdf = pd.DataFrame({"Group": np.ones(len(y))}, index=y.index)
        else:
            rf = str(re_formula)
            Zdf = dmatrix(rf, data, return_type="dataframe", NA_action=na_action)
            Zdf = Zdf.rename(columns={"Intercept": "Group"})

        Zdf = Zdf.loc[y.index]
        if len(groups_arr) != len(y):
            groups_arr = pd.Series(groups_arr, index=data.index).loc[y.index].to_numpy()

        model = cls(y.iloc[:, 0].to_numpy(float), X.to_numpy(float), groups_arr,
                    exog_re=Zdf.to_numpy(float), missing="none",
                    endog_name=str(y.columns[0]),
                    exog_names=list(X.columns),
                    exog_re_names=list(Zdf.columns),
                    formula=formula, re_formula=re_formula,
                    _data_frame=data, **kwargs)
        return model

    # -- fitting ------------------------------------------------------------
    def fit(self, start_params=None, reml=True, niter_sa=0, do_cg=True,
            fe_pen=None, cov_pen=None, free=None, full_output=False,
            method=None, **fit_kwargs):
        for name, val in (("fe_pen", fe_pen), ("cov_pen", cov_pen), ("free", free)):
            if val is not None:
                raise NotImplementedError(
                    f"{name}= is not implemented in this release; see "
                    "docs/LIMITATIONS.md. Refusing rather than ignoring it, "
                    "because honouring it by ignoring it would report a "
                    "different model's numbers under your specification."
                )
        if niter_sa:
            warnings.warn("niter_sa is accepted for signature compatibility and "
                          "ignored: this optimiser does not use simulated annealing.",
                          UserWarning, stacklevel=2)

        theta0 = None
        if start_params is not None:
            sp = np.asarray(start_params, float)
            if isinstance(start_params, MixedLMParams):
                cov = np.asarray(start_params.cov_re, float)
                theta0 = _vech_col(np.linalg.cholesky(cov))
            elif sp.size == self.k_re2:
                theta0 = sp
            # a full packed vector: take the covariance block
            elif sp.size >= self.k_fe + self.k_re2:
                tri = sp[self.k_fe:self.k_fe + self.k_re2]
                mat = np.zeros((self.k_re, self.k_re))
                k = 0
                for r in range(self.k_re):
                    for c in range(r + 1):
                        mat[r, c] = tri[k]
                        k += 1
                theta0 = _vech_col(mat)

        n_starts = int(fit_kwargs.pop("n_starts", 3))
        res = fit_core(self.endog, self.exog, self.exog_re, self._codes,
                       self.n_groups, reml=reml, start_params=theta0,
                       method=method, n_starts=n_starts,
                       maxiter=int(fit_kwargs.pop("maxiter", 500)),
                       gtol=float(fit_kwargs.pop("gtol", 1e-8)),
                       ftol=float(fit_kwargs.pop("ftol", 1e-12)))
        return MixedLMResults(self, res)

    # -- likelihood surface (for compatibility and testing) -----------------
    def loglike(self, params, profile_fe=True):
        """Log-likelihood at a packed parameter vector."""
        from ._fit import fit_core as _fc  # noqa: F401
        theta = self._theta_from_packed(params)
        core = self._core()
        dev = core.deviance(list(theta), True)
        return -0.5 * dev

    def _core(self):
        from ._mixedlm_rs import LmmCore
        if getattr(self, "_core_cache", None) is None:
            self._core_cache = LmmCore(
                np.ascontiguousarray(self.endog),
                np.ascontiguousarray(self.exog),
                np.ascontiguousarray(self.exog_re),
                np.ascontiguousarray(self._codes),
                self.n_groups)
        return self._core_cache

    def _theta_from_packed(self, params):
        sp = np.asarray(params, float)
        tri = sp[self.k_fe:self.k_fe + self.k_re2]
        mat = np.zeros((self.k_re, self.k_re))
        k = 0
        for r in range(self.k_re):
            for c in range(r + 1):
                mat[r, c] = tri[k]
                k += 1
        return _vech_col(mat)

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        exog = np.asarray(exog, float)
        fe = np.asarray(params, float)[:self.k_fe]
        return exog @ fe

    @property
    def endog_names(self):
        return self._endog_name

    def initialize(self):
        return None

    def score(self, params, profile_fe=True):
        theta = self._theta_from_packed(params)
        _, g = self._core().deviance_grad(list(theta), True)
        return -0.5 * np.asarray(g, float)

    def hessian(self, params):
        from ._fit import _profiled_hessian
        theta = self._theta_from_packed(params)
        return -0.5 * _profiled_hessian(self._core(), theta, True)

    def information(self, params):
        return -self.hessian(params)

    def get_scale(self, fe_params=None, cov_re=None, vcomp=None):
        raise NotImplementedError(
            "get_scale is an internal statsmodels helper tied to its "
            "parameterisation; use MixedLMResults.scale instead."
        )

    @property
    def group_list(self):
        return list(self.group_labels)


def _vech_col(mat):
    """Column-major lower-triangle packing -- the theta convention."""
    q = mat.shape[0]
    return np.array([mat[r, c] for c in range(q) for r in range(c, q)])


class MixedLMResults:
    """Results of a :class:`MixedLM` fit."""

    def __init__(self, model, res):
        self.model = model
        self._res = res

        self.fe_params = res["beta"]
        self.cov_re = res["cov_re"]
        self.cov_re_unscaled = res["cov_re_unscaled"]
        self.scale = res["scale"]
        self.vcomp = np.zeros(0)
        self.converged = res["converged"]
        self.reml = res["reml"]
        self.nobs = res["n"]
        self.k_fe = res["p"]
        self.k_re = res["q"]
        self.k_re2 = model.k_re2
        self.k_vc = 0
        self.method = "REML" if res["reml"] else "ML"
        self.use_t = False

        self._cov_beta = res["cov_beta"]
        self._random_effects = res["random_effects"]
        self._deviance = res["deviance"]
        self._bse_re_unscaled = res["bse_re_unscaled"]

    # -- parameter vector, statsmodels packing ------------------------------
    @property
    def params(self):
        return np.concatenate([self.fe_params, _vech_row(self.cov_re_unscaled),
                               self.vcomp])

    @property
    def bse_fe(self):
        return np.sqrt(np.diag(self._cov_beta))

    @property
    def bse_re(self):
        m = self._bse_re_unscaled
        if m is None:
            return np.full(self.k_re2, np.nan)
        return _vech_row(m)

    @property
    def bse(self):
        return np.concatenate([self.bse_fe, self.bse_re])

    @property
    def tvalues(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.params / self.bse

    @property
    def pvalues(self):
        from scipy import stats
        with np.errstate(invalid="ignore", divide="ignore"):
            return 2 * stats.norm.sf(np.abs(self.tvalues))

    @property
    def llf(self):
        return -0.5 * self._deviance

    @property
    def df_modelwc(self):
        return self.k_fe + self.k_re2 + self.k_vc

    @property
    def aic(self):
        if self.reml:
            return np.nan
        return -2 * (self.llf - (self.params.size + 1))

    @property
    def bic(self):
        if self.reml:
            return np.nan
        df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    @property
    def fittedvalues(self):
        """Conditional fit: X*beta + Z*b, including the random effects.

        This matches statsmodels, whose ``fittedvalues`` adds each group's
        conditional modes. ``predict()`` remains marginal (fixed effects only),
        also matching the reference.
        """
        fit = self.model.exog @ self.fe_params
        b = self._random_effects            # (m, q)
        codes = self.model._codes
        fit = fit + np.einsum("nq,nq->n", self.model.exog_re, b[codes])
        return fit

    @property
    def resid(self):
        return self.model.endog - self.fittedvalues

    @property
    def random_effects(self):
        """Conditional modes, one entry per group, as statsmodels returns them."""
        names = self.model._exog_re_names
        return {lab: pd.Series(self._random_effects[i], index=names)
                for i, lab in enumerate(self.model.group_labels)}

    @property
    def random_effects_cov(self):
        cov = self.cov_re
        names = self.model._exog_re_names
        frame = pd.DataFrame(cov, index=names, columns=names)
        return {lab: frame for lab in self.model.group_labels}

    def conf_int(self, alpha=0.05, cols=None):
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.params - z * self.bse
        hi = self.params + z * self.bse
        out = np.column_stack([lo, hi])
        return out if cols is None else out[cols]

    def cov_params(self):
        """Covariance of the fixed effects.

        statsmodels returns the covariance of the whole packed vector; here only
        the fixed-effect block is exact, so the variance-component rows are
        filled with the delta-method variances on the diagonal and NaN
        off-diagonal. Documented in docs/CORRECTNESS.md.
        """
        k = self.params.size
        out = np.full((k, k), np.nan)
        p = self.k_fe
        out[:p, :p] = self._cov_beta
        bre = self.bse_re
        for i in range(len(bre)):
            out[p + i, p + i] = bre[i] ** 2
        return out

    def predict(self, exog=None):
        return self.model.predict(self.params, exog=exog)

    # -- reporting ----------------------------------------------------------
    def summary(self, yname=None, xname_fe=None, xname_re=None, title=None,
                alpha=0.05):
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        fe_names = list(xname_fe or self.model.exog_names)
        re_names = list(xname_re or self.model._exog_re_names)

        lines = []
        title = title or "Mixed Linear Model Regression Results"
        lines.append(f"{title:^78s}")
        lines.append("=" * 78)
        left = [
            ("Model:", "MixedLM"),
            ("No. Observations:", f"{self.nobs}"),
            ("No. Groups:", f"{self.model.n_groups}"),
            ("Min. group size:", f"{int(np.bincount(self.model._codes).min())}"),
            ("Max. group size:", f"{int(np.bincount(self.model._codes).max())}"),
            ("Mean group size:", f"{self.nobs / self.model.n_groups:.1f}"),
        ]
        right = [
            ("Dependent Variable:", yname or self.model.endog_names),
            ("Method:", self.method),
            ("Scale:", f"{self.scale:.4f}"),
            ("Log-Likelihood:", f"{self.llf:.4f}"),
            ("Converged:", "Yes" if self.converged else "No"),
            ("", ""),
        ]
        for (la, lv), (ra, rv) in zip(left, right):
            lines.append(f"{la:<22s}{lv:<18s}{ra:<22s}{rv:>16s}")
        lines.append("-" * 78)
        lines.append(f"{'':<20s}{'Coef.':>10s}{'Std.Err.':>10s}{'z':>9s}"
                     f"{'P>|z|':>9s}{'[' + format(alpha / 2, '.3f'):>10s}"
                     f"{format(1 - alpha / 2, '.3f') + ']':>10s}")
        lines.append("-" * 78)

        for i, nm in enumerate(fe_names):
            c = self.fe_params[i]
            se = self.bse_fe[i]
            zv = c / se if se > 0 else np.nan
            pv = 2 * stats.norm.sf(abs(zv)) if np.isfinite(zv) else np.nan
            lines.append(f"{nm[:20]:<20s}{c:10.3f}{se:10.3f}{zv:9.3f}{pv:9.3f}"
                         f"{c - z * se:10.3f}{c + z * se:10.3f}")

        # Variance components, on the unscaled (statsmodels) scale.
        vn = _re_param_names(re_names)
        vals = _vech_row(self.cov_re_unscaled)
        ses = self.bse_re
        for i, nm in enumerate(vn):
            se = ses[i] if i < len(ses) else np.nan
            se_s = f"{se:10.3f}" if np.isfinite(se) else f"{'':>10s}"
            lines.append(f"{nm[:20]:<20s}{vals[i]:10.3f}{se_s}"
                         f"{'':>9s}{'':>9s}{'':>10s}{'':>10s}")
        lines.append("=" * 78)
        return _SummaryText("\n".join(lines))

    def __str__(self):
        return str(self.summary())

    # -- explicitly unimplemented -------------------------------------------
    def profile_re(self, *args, **kwargs):
        raise NotImplementedError(
            "profile_re is not implemented in this release; see docs/LIMITATIONS.md"
        )

    def bootstrap(self, *args, **kwargs):
        raise NotImplementedError(
            "bootstrap is not implemented in this release; see docs/LIMITATIONS.md"
        )

    def get_distribution(self, *args, **kwargs):
        raise NotImplementedError(
            "get_distribution is not implemented in this release; "
            "see docs/LIMITATIONS.md"
        )


def _re_param_names(re_names):
    """statsmodels' variance-component labels: 'Group Var', 'Group x D Cov', 'D Var'."""
    out = []
    for r in range(len(re_names)):
        for c in range(r + 1):
            if r == c:
                out.append(f"{re_names[r]} Var")
            else:
                out.append(f"{re_names[c]} x {re_names[r]} Cov")
    return out


class _SummaryText:
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text

    def __repr__(self):
        return self._text

    def as_text(self):
        return self._text


def mixedlm(formula, data, groups, re_formula=None, vc_formula=None,
            subset=None, use_sparse=False, missing="none", *args, **kwargs):
    """Formula interface, matching ``statsmodels.formula.api.mixedlm``."""
    return MixedLM.from_formula(formula, data, re_formula=re_formula,
                                vc_formula=vc_formula, subset=subset,
                                use_sparse=use_sparse, missing=missing,
                                groups=groups, *args, **kwargs)
