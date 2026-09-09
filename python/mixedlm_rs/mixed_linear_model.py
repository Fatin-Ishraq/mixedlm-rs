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

import os
import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._fit import ConvergenceWarning, fit_core

__all__ = [
    "ConvergenceWarning",
    "MixedLM",
    "MixedLMParams",
    "MixedLMResults",
    "VCSpec",
]


class VCSpec:
    """Variance-component specification.

    Present so that code importing the name keeps working. Variance components
    beyond a single grouping factor are not implemented in this release; see
    ``docs/LIMITATIONS.md``. Constructing one is fine; passing it to ``MixedLM``
    raises ``NotImplementedError`` rather than silently fitting a different model.
    """

    def __init__(
        self,
        names: Sequence[str],
        colnames: Sequence[Sequence[str]],
        mats: Sequence[ArrayLike],
    ) -> None:
        # Positional order follows statsmodels exactly -- (names, colnames,
        # mats). This once read (names, mats, colnames), which silently swapped
        # two of the three arguments for anyone constructing one positionally.
        self.names = names
        self.colnames = colnames
        self.mats = mats


class MixedLMParams:
    """Packed parameter container, matching statsmodels' layout.

    The packed vector is ``[fe_params, vech_row(cov_re_unscaled), vcomp]`` where
    ``vech_row`` walks the lower triangle by rows: (0,0), (1,0), (1,1), ...
    """

    def __init__(self, k_fe: int, k_re: int, k_vc: int) -> None:
        self.k_fe: int = int(k_fe)
        self.k_re: int = int(k_re)
        self.k_re2: int = int(k_re * (k_re + 1) // 2)
        self.k_vc: int = int(k_vc)
        self.fe_params: np.ndarray = np.zeros(self.k_fe)
        self.cov_re: np.ndarray = (
            np.eye(self.k_re) if self.k_re else np.zeros((0, 0)))
        self.vcomp: np.ndarray = np.zeros(self.k_vc)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        fe_params: ArrayLike | None = None,
        cov_re: ArrayLike | None = None,
        cov_re_sqrt: ArrayLike | None = None,
        vcomp: ArrayLike | None = None,
    ) -> MixedLMParams:
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
    def from_packed(
        cls,
        params: ArrayLike,
        k_fe: int,
        k_re: int,
        use_sqrt: bool = True,
        has_fe: bool = True,
    ) -> MixedLMParams:
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
        # Variance components are stored as variances; under use_sqrt the packed
        # vector holds their square roots, so square on the way in. Matching the
        # reference here matters even though vcomp is always empty in this
        # release: round-tripping a packed vector must be the identity.
        vcomp = np.asarray(vcomp, float) ** 2 if use_sqrt else np.asarray(vcomp, float)
        return cls.from_components(fe_params=fe, cov_re=cov, vcomp=vcomp)

    def get_packed(self, use_sqrt: bool = True,
                   has_fe: bool = False) -> np.ndarray:
        """Pack into statsmodels' layout.

        ``has_fe`` defaults to False, as in the reference: the packed vector the
        optimiser passes around is covariance-only.
        """
        cov = np.asarray(self.cov_re, float)
        if use_sqrt and cov.size:
            try:
                mat = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # A singular or indefinite covariance has no Cholesky factor.
                # The reference falls back to the diagonal square root rather
                # than raising, and callers rely on that for boundary fits,
                # where an exactly-zero variance component is the common case.
                mat = np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))
        else:
            mat = cov
        tri = np.array([mat[r, c] for r in range(self.k_re) for c in range(r + 1)])
        vcomp = np.asarray(self.vcomp, float)
        if use_sqrt:
            vcomp = np.sqrt(np.maximum(vcomp, 0.0))
        parts = []
        if has_fe:
            parts.append(np.asarray(self.fe_params, float))
        parts.append(tri)
        parts.append(vcomp)
        return np.concatenate(parts) if parts else np.zeros(0)

    def copy(self) -> MixedLMParams:
        return MixedLMParams.from_components(
            fe_params=self.fe_params.copy(),
            cov_re=self.cov_re.copy(),
            vcomp=self.vcomp.copy(),
        )


def _vech_row(mat):
    k = mat.shape[0]
    return np.array([mat[r, c] for r in range(k) for c in range(r + 1)])


def _designs(formula, re_formula, data, missing, eval_env):
    """Build the fixed and random designs, dropping missing rows once.

    Returns ``(y, X, Z, names, kept)`` where ``kept`` indexes the rows of
    ``data`` that survived, so the group vector can be aligned by position.

    patsy can drop rows for missing data, but it does not report which rows it
    dropped, and the ``design_info`` it attaches to a DataFrame result does not
    survive current pandas. So missingness is resolved here, once, across both
    formulas: build on the complete cases and hand patsy data it never needs to
    filter.
    """
    from patsy import PatsyError, dmatrices, dmatrix

    def build(frame: pd.DataFrame, na_action: str) -> tuple:
        y, X = dmatrices(formula, frame, return_type="matrix",
                         NA_action=na_action, eval_env=eval_env)
        if y.shape[1] != 1:
            # patsy happily builds a multi-column response for `y1 + y2 ~ x` or
            # a categorical left-hand side, and taking column 0 would fit a
            # different response than the one written. The reference rejects
            # these; so do we.
            raise ValueError(
                f"the left-hand side of {formula!r} produced {y.shape[1]} "
                "columns. A mixed model needs exactly one response; a "
                "categorical or multi-column left-hand side is not supported.")
        if re_formula is None or str(re_formula).strip() in ("1", "~1", ""):
            Z, re_info, re_names = np.ones((y.shape[0], 1)), None, ["Group"]
        else:
            Zm = dmatrix(str(re_formula), frame, return_type="matrix",
                         NA_action=na_action, eval_env=eval_env)
            Z, re_info = np.asarray(Zm, float), Zm.design_info
            re_names = ["Group" if c == "Intercept" else c
                        for c in re_info.column_names]
        return y, X, Z, re_info, re_names

    try:
        y, X, Z, re_info, re_names = build(data, "raise")
        kept = np.arange(len(data), dtype=np.intp)
    except PatsyError:
        if missing != "drop":
            raise
        # Find the complete cases with patsy's own NA rules, then rebuild on
        # exactly those rows so both designs describe the same observations.
        frames = [dmatrices(formula, data, return_type="dataframe",
                            NA_action="drop", eval_env=eval_env)[0]]
        if re_formula is not None and str(re_formula).strip() not in ("1", "~1", ""):
            frames.append(dmatrix(str(re_formula), data,
                                  return_type="dataframe", NA_action="drop",
                                  eval_env=eval_env))
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.intersection(f.index)
        kept = np.asarray(idx, dtype=np.intp)
        y, X, Z, re_info, re_names = build(data.iloc[kept], "raise")

    names = {
        "endog": str(y.design_info.column_names[0]),
        "exog": list(X.design_info.column_names),
        "exog_re": re_names,
    }
    return (np.asarray(y, float)[:, 0], np.asarray(X, float), Z,
            names, kept, X.design_info, re_info)


# Defaults for the optimiser controls `fit()` accepts, with the rule each one
# has to satisfy. `n_starts=None` means "let the driver choose per optimiser":
# 1 for scipy's L-BFGS-B and 5 for the in-crate one, both measured.
_FIT_OPTIONS = {
    "maxiter": (500, "a positive integer"),
    "gtol": (1e-8, "a positive finite float"),
    "ftol": (1e-12, "a positive finite float"),
    "n_starts": (None, "a positive integer, or None to choose per optimiser"),
}


def _validate_fit_options(kwargs):
    """Check the optimiser controls, and reject anything unrecognised.

    Returns the validated options. Raises before any fitting happens, so a
    mistake costs a millisecond rather than a whole optimisation.
    """
    unknown = set(kwargs) - set(_FIT_OPTIONS)
    if unknown:
        raise TypeError(
            "unexpected keyword argument(s) for fit: "
            + ", ".join(sorted(unknown))
            + ". Accepted optimiser controls are: "
            + ", ".join(sorted(_FIT_OPTIONS))
            + ". statsmodels options that exist but are not implemented here "
              "raise NotImplementedError with a pointer to "
              "docs/LIMITATIONS.md.")

    out = {}
    for name, (default, rule) in _FIT_OPTIONS.items():
        value = kwargs.get(name, default)
        if name == "n_starts" and value is None:
            out[name] = None
            continue

        if name in ("maxiter", "n_starts"):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"{name}={value!r} must be {rule}, not "
                    f"{type(value).__name__}")
            value = int(value)
            if value < 1:
                raise ValueError(f"{name}={value} must be {rule}")
        else:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name}={value!r} must be {rule}, not "
                    f"{type(value).__name__}") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}={value!r} must be {rule}")
        out[name] = value

    # A tolerance looser than the certificate it is checked against is a
    # contradiction: the optimiser would stop early and the stationarity check
    # would then reject the point it was told to stop at.
    if out["gtol"] > 1e-3:
        warnings.warn(
            f"gtol={out['gtol']:g} is looser than the stationarity tolerance "
            "this fit is certified against, so the optimiser will stop early "
            "and the certificate will fail. Expect converged=False.",
            UserWarning, stacklevel=3)
    return out


def _subset_positions(data, subset):
    """Positional row indices selected by ``subset``.

    Two accepted forms, and they are distinguished by dtype rather than by
    guessing:

    * A **boolean mask** of the same length as ``data``. If it is a Series its
      index must match the frame's, so that a mask built from a different
      frame cannot silently select the wrong rows.
    * A collection of **index labels**. This is pandas' and statsmodels'
      meaning of ``subset``, and it is what this used to get wrong: labels were
      treated as positions, so ``subset=[10, 20]`` on a frame indexed
      ``100..199`` selected rows 10 and 20 rather than raising, and on a
      string-labelled frame it selected nothing.

    Selection is by membership, not by ``.loc``. With a duplicated index label
    ``.loc`` returns every matching row *per occurrence in the subset*, which
    multiplies rows; membership returns each matching row once, in the frame's
    own order. That keeps a positionally-aligned external ``groups`` array
    aligned, which ``.loc`` does not.

    Returns a sorted array of positions.
    """
    if subset is None:
        return np.arange(len(data), dtype=np.intp)

    if isinstance(subset, pd.Series) and subset.dtype == bool:
        if not subset.index.equals(data.index):
            raise ValueError(
                "a boolean `subset` Series must have the same index as `data`; "
                "got an index of length "
                f"{len(subset.index)} against {len(data.index)}. Pass a plain "
                "array if you mean positional selection.")
        return _nonempty(np.flatnonzero(subset.to_numpy()))

    arr = np.asarray(subset)
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.size != len(data):
            raise ValueError(
                f"a boolean `subset` must be one-dimensional with {len(data)} "
                f"entries, one per row of `data`; got shape {arr.shape}")
        return _nonempty(np.flatnonzero(arr))

    if arr.ndim != 1:
        raise ValueError(
            f"`subset` must be one-dimensional; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("`subset` selected no rows")

    labels = pd.Index(arr)
    missing = labels[~labels.isin(data.index)]
    if len(missing):
        shown = list(dict.fromkeys(missing.tolist()))[:5]
        raise ValueError(
            f"`subset` contains {len(set(missing.tolist()))} label(s) that are "
            f"not in the index of `data`: {shown}"
            + (" ..." if len(set(missing.tolist())) > len(shown) else "")
            + ". `subset` selects by index label, not by row position -- pass a "
              "boolean mask if you meant positions.")

    return _nonempty(np.flatnonzero(data.index.isin(labels)))


def _nonempty(positions):
    """An empty selection is a caller error, and must say so here.

    Left to flow through, it reaches the design construction and surfaces as a
    rank-deficiency error about the fixed effects, which names the wrong thing.
    """
    if positions.size == 0:
        raise ValueError(
            "`subset` selected no rows. Check the mask or the labels: an "
            "empty selection cannot be fitted, and the error further down "
            "would blame the design matrix instead.")
    return positions


def _index_is_positional(index, length):
    """True when ``index`` carries no information beyond row position.

    A default ``RangeIndex(0, n)`` is what you get from a Series built out of a
    bare array, so its labels say nothing the position does not already say.
    Any other index is the caller telling us which rows the values belong to,
    and we must not silently override it.
    """
    if not isinstance(index, pd.RangeIndex):
        return False
    return index.start == 0 and index.step == 1 and len(index) == length


def _align_groups(groups, data, positions):
    """Resolve ``groups`` against ``data`` and apply the subset selection.

    ``groups`` may be a column name, a Series, or an array.

    A **Series is aligned by index label** whenever its index carries meaning
    and can address the frame, which is what pandas itself would do and what
    the caller means when they hand over a labelled object. A Series whose
    index carries no information -- a default ``RangeIndex`` -- is aligned by
    position. Input that is neither is rejected rather than guessed at. See
    :func:`_align_group_series` for why that order matters.

    Aligning a labelled Series positionally is a silent-wrong-answer bug: a
    caller who subsets and then sorts or shuffles their group Series gets a
    model fitted on scrambled cluster membership, which returns a plausible
    number rather than an error.

    An array is positional; it has no labels to align by.
    """
    if isinstance(groups, str):
        if groups not in data.columns:
            raise ValueError(
                f"groups={groups!r} is not a column of `data`. Columns are: "
                f"{list(data.columns)[:12]}")
        return np.asarray(data[groups])[positions]

    if isinstance(groups, pd.Series):
        return _align_group_series(groups, data, positions)

    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(
            f"groups must be one-dimensional, got shape {arr.shape}")
    if arr.size == len(data):
        return arr[positions]
    if arr.size == len(positions):
        return arr
    raise ValueError(
        f"groups has length {arr.size}, which matches neither `data` "
        f"({len(data)} rows) nor the subset selection ({len(positions)} rows)")


def _align_group_series(groups, data, positions):
    """Align a group Series to the selected rows of ``data``.

    The order of these branches is the whole point, and it is not the obvious
    order. Position is tried first, but *only* for an index that carries no
    information; everything else is aligned by label.

    A default ``RangeIndex`` is what a Series built from a bare array has, so
    its labels say nothing the position does not. Treating those as labels
    breaks the ordinary positional call: a 50-element ``pd.Series(arr)`` handed
    alongside a 100-row frame has a ``RangeIndex(0, 50)``, which *is* a subset
    of that frame's ``RangeIndex(0, 100)`` and would silently be read as
    selecting rows 0-49.

    Any other index is the caller saying which rows the values belong to, and
    aligning that positionally is a silent-wrong-answer bug: a caller who
    subsets and then sorts or shuffles their group Series gets a model fitted
    on scrambled cluster membership, which returns a plausible number rather
    than an error. Input that is neither is rejected.
    """
    if groups.index.equals(data.index):
        # Same labels in the same order: position and label agree.
        return np.asarray(groups.to_numpy())[positions]

    # An uninformative index: align by position, as the caller meant.
    for length, subset_after in ((len(data), True), (len(positions), False)):
        if len(groups) == length and _index_is_positional(groups.index, length):
            values = np.asarray(groups.to_numpy())
            return values[positions] if subset_after else values

    selected = data.index[positions]
    label_addressable = (
        data.index.is_unique
        and groups.index.is_unique
        and bool(groups.index.isin(data.index).all())
    )
    if label_addressable:
        absent = selected[~selected.isin(groups.index)]
        if len(absent):
            shown = list(absent[:5])
            raise ValueError(
                f"`groups` is missing {len(absent)} of the {len(selected)} "
                f"selected row label(s), for example {shown}"
                + (" ..." if len(absent) > len(shown) else "")
                + ". A group Series carrying a meaningful index is aligned to "
                  "`data` by that index, and every selected row needs a "
                  "group. Pass a plain array if you meant row order.")
        return np.asarray(groups.reindex(selected).to_numpy())

    # Neither addressable by label nor positional. Falling back to row order
    # here would discard labels the caller meant, which is the bug this
    # function exists to prevent, so say what is wrong instead.
    if not data.index.is_unique:
        raise ValueError(
            "`data` has a non-unique index, so a `groups` Series cannot be "
            "aligned to it by label. Pass `groups` as a column name of "
            "`data`, or as a plain array in row order.")
    if not groups.index.is_unique:
        raise ValueError(
            "`groups` has duplicate index labels, so it cannot be aligned to "
            "`data` by label. Pass a column name of `data`, or a plain array "
            "in row order.")

    stray = groups.index[~groups.index.isin(data.index)]
    raise ValueError(
        f"`groups` has {len(stray)} index label(s) that are not in the index "
        f"of `data`, for example {list(stray[:5])}"
        + (" ..." if len(stray) > 5 else "")
        + f", so it cannot be aligned by label; and its length ({len(groups)}) "
          f"with a non-positional index cannot be aligned by row order "
          f"either. `data` has {len(data)} rows and the selection has "
          f"{len(positions)}. Pass a column name of `data`, a Series indexed "
          "like `data`, or a plain array in row order.")


_PY_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _formula_namespace(formula, re_formula, eval_env):
    """Capture the caller-defined names a formula depends on.

    patsy resolves ``custom(x)`` out of the caller's namespace at fit time. The
    fitted model outlives that namespace, so the design cannot be rebuilt after
    a pickle round-trip unless the objects come along.

    Only names the formula actually mentions are captured, only when they are
    not columns of the data, and only when they pickle. A module-level function
    -- the ordinary case -- pickles by qualified name and costs nothing. A
    lambda or a closure does not pickle; it is left behind deliberately, and
    :meth:`_ensure_design_info` reports it by name rather than guessing.
    """
    if eval_env is None:
        return {}
    text = str(formula) + " " + ("" if re_formula is None else str(re_formula))
    wanted = set(_PY_IDENTIFIER.findall(text))
    if not wanted:
        return {}

    import pickle
    from types import ModuleType

    captured = {}
    namespaces = list(getattr(eval_env, "_namespaces", ()))
    for name in sorted(wanted):
        for ns in namespaces:
            try:
                if name not in ns:
                    continue
                value = ns[name]
            except Exception:
                continue
            if isinstance(value, ModuleType):
                break
            try:
                pickle.loads(pickle.dumps(value))
            except Exception:
                # Not portable across the pickle boundary. Recording the name
                # with no value lets the restore path say which transform is
                # missing instead of blaming the retained training frame.
                captured[name] = _Unpicklable(name, type(value).__name__)
            else:
                captured[name] = value
            break
    return captured


class _Unpicklable:
    """Placeholder for a formula name that could not be carried through pickle.

    Held so the failure can be *named*. Without it the restore path only sees
    that the formula stopped evaluating and has to guess at the reason.
    """

    __slots__ = ("kind", "name")

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind


def _group_is_present(groups):
    """Row mask for group labels that are actually observed.

    A missing group label is not a group. Retaining it silently created an
    extra level made entirely of rows whose cluster is unknown.
    """
    g = np.asarray(groups)
    if g.dtype.kind in "fc":
        return np.isfinite(g)
    return ~pd.isna(g)


def _codes_from_groups(groups):
    g = np.asarray(groups)
    if g.ndim != 1:
        # Catching this here turns a cryptic failure deep in the extension
        # module into a message that names the actual mistake -- most often
        # MixedLM(y, X, Z, groups), where the third positional argument is
        # `groups`, not `exog_re`.
        raise ValueError(
            f"groups must be one-dimensional, got shape {g.shape}. "
            "Note the signature is MixedLM(endog, exog, groups, exog_re=...) "
            "-- the third positional argument is the grouping variable."
        )
    uniq, codes = np.unique(g, return_inverse=True)
    return uniq, np.ascontiguousarray(codes.ravel(), dtype=np.int64)


class MixedLM:
    """Linear mixed effects model.

    Parameters mirror ``statsmodels.regression.mixed_linear_model.MixedLM``.
    """

    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike,
        groups: ArrayLike,
        exog_re: ArrayLike | None = None,
        exog_vc: VCSpec | None = None,
        use_sqrt: bool = True,
        missing: str = "none",
        **kwargs: Any,
    ) -> None:
        if exog_vc is not None:
            raise NotImplementedError(
                "variance components (exog_vc / vc_formula) are not implemented "
                "in this release. Refusing rather than silently fitting a "
                "different model; see docs/LIMITATIONS.md."
            )

        self.exog_names = kwargs.pop("exog_names", None)
        self._endog_name = kwargs.pop("endog_name", None) or "y"
        self.data_frame = kwargs.pop("_data_frame", None)
        # Which rows of `data_frame` the design was actually built on.
        # Missing-data handling can drop rows across *both* formulas at
        # once, so rebuilding the design later has to reproduce that same
        # selection or the stateful transforms relearn different state.
        self._design_rows: np.ndarray | None = kwargs.pop(
            "_design_rows", None)
        self._formula_namespace: dict[str, Any] | None = kwargs.pop(
            "_formula_namespace", None)
        self._design_rebuild_error: str | None = None
        self.formula = kwargs.pop("formula", None)
        self.re_formula = kwargs.pop("re_formula", None)
        self._exog_re_names = kwargs.pop("exog_re_names", None)
        self._design_info = kwargs.pop("_design_info", None)
        self._re_design_info = kwargs.pop("_re_design_info", None)
        if kwargs:
            # Silently swallowing an argument is how a caller ends up believing
            # a model honoured an option it never saw. statsmodels options that
            # exist but are not implemented here raise NotImplementedError with
            # a pointer to docs/LIMITATIONS.md; anything unrecognised is a typo
            # and gets the ordinary TypeError.
            raise TypeError(
                "unexpected keyword argument(s) for MixedLM: "
                + ", ".join(sorted(kwargs)))

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
        if not (len(endog) == exog.shape[0] == exog_re.shape[0] == len(groups)):
            raise ValueError(
                f"endog ({len(endog)}), exog ({exog.shape[0]}), exog_re "
                f"({exog_re.shape[0]}) and groups ({len(groups)}) must agree "
                "on length")

        # Missingness is resolved across *all four* inputs at once. Dropping
        # them independently -- endog/exog by one rule, exog_re by another,
        # groups by none at all -- left the arrays misaligned, and a missing
        # group label survived to become a group of its own.
        if missing == "drop":
            ok = (np.isfinite(endog) & np.all(np.isfinite(exog), 1)
                  & np.all(np.isfinite(exog_re), 1) & _group_is_present(groups))
            endog, exog, exog_re, groups = endog[ok], exog[ok], exog_re[ok], groups[ok]
            if len(endog) == 0:
                raise ValueError("every row was dropped as missing")
        else:
            bad = int((~np.isfinite(endog)).sum() + (~np.isfinite(exog)).sum()
                      + (~np.isfinite(exog_re)).sum()
                      + (~_group_is_present(groups)).sum())
            if bad:
                raise ValueError(
                    "endog/exog/exog_re/groups contain missing or non-finite "
                    "values; pass missing='drop' to remove those rows")

        self.endog: np.ndarray = endog
        self.exog: np.ndarray = exog
        self.exog_re: np.ndarray = exog_re
        self.exog_vc: VCSpec | None = None
        self.groups: np.ndarray = groups
        self.use_sqrt: bool = use_sqrt
        # The criterion the model was last fitted with; loglike/score/hessian
        # default to it rather than assuming REML.
        self.reml = True

        self.group_labels, self._codes = _codes_from_groups(groups)
        self.n_groups: int = len(self.group_labels)

        self.k_fe: int = exog.shape[1]
        self.k_re: int = exog_re.shape[1]
        self.k_re2: int = self.k_re * (self.k_re + 1) // 2
        self.k_vc: int = 0
        self.nobs: int = len(endog)

        if self.exog_names is None:
            self.exog_names = [f"x{i}" for i in range(self.k_fe)]
        if self._exog_re_names is None:
            self._exog_re_names = [f"z{i}" for i in range(self.k_re)]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        re_formula: str | None = None,
        vc_formula: Any = None,
        subset: ArrayLike | None = None,
        use_sparse: bool = False,
        missing: str = "none",
        *args: Any,
        **kwargs: Any,
    ) -> MixedLM:
        if vc_formula is not None:
            raise NotImplementedError(
                "vc_formula (variance components) is not implemented in this "
                "release; see docs/LIMITATIONS.md."
            )
        if use_sparse:
            # Accepted for signature compatibility and then ignored, like
            # `niter_sa` and `do_cg` in fit(). Those warn; this one did not,
            # while the documentation promised that it did. A silently ignored
            # argument is the specific thing a compatibility layer must not do.
            warnings.warn(
                "use_sparse=True is accepted for statsmodels compatibility "
                "and ignored: this implementation factorises the "
                "block-diagonal system directly, which is already the "
                "efficient path for a single grouping factor. There is no "
                "dense fallback to switch away from. See "
                "docs/LIMITATIONS.md.",
                UserWarning, stacklevel=2)
        from patsy import EvalEnvironment

        groups = kwargs.pop("groups", None)
        if groups is None:
            raise ValueError("from_formula requires groups=")
        eval_env = kwargs.pop("eval_env", 0)
        if isinstance(eval_env, int):
            # +1 for this frame, so a formula referring to the caller's locals
            # resolves the way it does in statsmodels.
            eval_env = EvalEnvironment.capture(eval_env + 1)

        # `subset` and `groups` are resolved together against the *unsubset*
        # frame, so the two can never end up half-aligned. Everything after
        # this point is positional: `.loc`-based alignment multiplies rows on a
        # frame with duplicate index labels, and breaks outright when `groups`
        # is an external Series carrying the unsubset index.
        data = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
        positions = _subset_positions(data, subset)
        groups_arr = _align_groups(groups, data, positions)
        data = data.iloc[positions]

        if len(groups_arr) != len(data):          # unreachable; kept as a guard
            raise ValueError(
                f"groups has length {len(groups_arr)} but the selected data "
                f"has {len(data)} rows")

        # Work on a positional index so patsy's NA handling and the group array
        # can be re-aligned by position rather than by label.
        work = data.reset_index(drop=True)
        y_arr, X_arr, Z, names, kept, fe_design, re_design = _designs(
            formula, re_formula, work, missing, eval_env)

        return cls(y_arr, X_arr, groups_arr[kept],
                   exog_re=Z, missing=missing,
                   endog_name=names["endog"],
                   exog_names=names["exog"],
                   exog_re_names=names["exog_re"],
                   formula=formula, re_formula=re_formula,
                   _design_info=fe_design,
                   _re_design_info=re_design,
                   _data_frame=data,
                   _design_rows=kept,
                   _formula_namespace=_formula_namespace(
                       formula, re_formula, eval_env),
                   **kwargs)

    # -- fitting ------------------------------------------------------------
    def fit(
        self,
        start_params: MixedLMParams | ArrayLike | None = None,
        reml: bool = True,
        niter_sa: int = 0,
        do_cg: bool = True,
        fe_pen: Any = None,
        cov_pen: Any = None,
        free: Any = None,
        full_output: bool = False,
        method: str | None = None,
        **fit_kwargs: Any,
    ) -> MixedLMResults:
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

        theta0 = self._start_theta(start_params, use_sqrt=self.use_sqrt)

        for name in ("do_cg", "full_output"):
            val = {"do_cg": do_cg, "full_output": full_output}[name]
            if val != {"do_cg": True, "full_output": False}[name]:
                warnings.warn(
                    f"{name}={val!r} is accepted for signature compatibility "
                    "and has no effect: this optimiser is L-BFGS-B over the "
                    "profiled criterion, not statsmodels' steepest-descent / "
                    "conjugate-gradient chain.", UserWarning, stacklevel=2)
        if method is not None and str(method).lower() not in ("rust", "lbfgs",
                                                              "l-bfgs-b", "bfgs",
                                                              "cg", "powell",
                                                              "nm", "newton"):
            raise ValueError(
                f"unknown optimisation method {method!r}. This package accepts "
                "None (scipy L-BFGS-B over the profiled criterion, the default) "
                "or 'rust' (the in-crate projected L-BFGS). statsmodels' "
                "optimiser names are accepted and ignored, because the "
                "criterion being optimised is not the same one.")
        if method is not None and str(method).lower() != "rust":
            warnings.warn(
                f"method={method!r} is accepted for statsmodels compatibility "
                "and ignored; the profiled criterion is optimised with "
                "L-BFGS-B. Pass method='rust' for the in-crate optimiser.",
                UserWarning, stacklevel=2)

        self.reml = bool(reml)

        # Everything is validated before the optimiser starts. Rejecting a
        # typo'd keyword *after* the fit -- which is what this used to do --
        # means a caller waits out a full optimisation to be told the argument
        # they passed was never read.
        opts = _validate_fit_options(fit_kwargs)

        res = fit_core(self.endog, self.exog, self.exog_re, self._codes,
                       self.n_groups, reml=reml, start_params=theta0,
                       method=method, **opts)
        return MixedLMResults(self, res)

    def _start_theta(self, start_params, use_sqrt=True):
        """Translate a user starting value into the internal `theta`.

        Accepts a :class:`MixedLMParams`, a covariance-only packed vector, or a
        full packed vector. Returns None to mean "use the default start".

        The isinstance test comes first: converting to an array up front raised
        TypeError on the documented parameter-container form before the branch
        that handles it was ever reached.
        """
        if start_params is None:
            return None

        if isinstance(start_params, MixedLMParams):
            cov = np.asarray(start_params.cov_re, float)
        else:
            sp = np.asarray(start_params, float).ravel()
            full = self.k_fe + self.k_re2 + self.k_vc
            if sp.size == self.k_re2 + self.k_vc:
                tri = sp[:self.k_re2]
            elif sp.size == full:
                tri = sp[self.k_fe:self.k_fe + self.k_re2]
            else:
                raise ValueError(
                    f"start_params has {sp.size} entries; expected "
                    f"{self.k_re2 + self.k_vc} (covariance only) or {full} "
                    "(fixed effects then covariance). A length matching "
                    "neither used to fall through to the default start, so a "
                    "mis-sized start was silently ignored.")
            # statsmodels packs the lower triangle by ROWS; theta is packed by
            # columns. Below q = 3 the two orders coincide, which is why this
            # was invisible until a three-term random-effects model was tried.
            mat = np.zeros((self.k_re, self.k_re))
            k = 0
            for r in range(self.k_re):
                for c in range(r + 1):
                    mat[r, c] = tri[k]
                    k += 1
            if use_sqrt:
                # The packed triangle is already a Cholesky factor.
                return _vech_col(mat)
            cov = mat + mat.T - np.diag(np.diag(mat))

        if cov.shape != (self.k_re, self.k_re):
            raise ValueError(
                f"start_params covariance is {cov.shape}, expected "
                f"({self.k_re}, {self.k_re})")
        try:
            fac = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            fac = np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))
        return _vech_col(fac)

    # -- likelihood surface (for compatibility and testing) -----------------
    def loglike(self, params: MixedLMParams | ArrayLike,
                profile_fe: bool = True, reml: bool | None = None) -> float:
        """Profiled log-likelihood at a packed parameter vector.

        ``params`` may be a :class:`MixedLMParams`, a covariance-only packed
        vector of length ``k_re2``, or a full packed vector. ``reml`` defaults
        to the model's own criterion rather than always REML -- evaluating the
        REML criterion for a model the caller fitted by ML reports a number
        that does not correspond to any fit.

        ``profile_fe`` is accepted for signature compatibility. This criterion
        is profiled over the fixed effects by construction, so ``False`` is not
        available; it raises rather than silently returning the profiled value.
        """
        if not profile_fe:
            raise NotImplementedError(
                "profile_fe=False is not available: the criterion implemented "
                "here eliminates the fixed effects analytically, so there is "
                "no un-profiled surface to evaluate. See docs/DESIGN.md.")
        theta = self._theta_from_packed(params)
        return -0.5 * self._core().deviance(list(theta), self._reml_flag(reml))

    def _reml_flag(self, reml):
        return bool(self.reml if reml is None else reml)

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
        """Packed vector (any accepted form) -> internal `theta`.

        A covariance-only vector of length ``k_re2`` is accepted, which is what
        the optimiser actually passes around; requiring a fixed-effect prefix
        made ``loglike`` raise IndexError on the reference's own convention.
        """
        theta = self._start_theta(params, use_sqrt=self.use_sqrt)
        if theta is None:
            raise ValueError("params must not be None")
        return theta

    def predict(self, params: MixedLMParams | ArrayLike,
                exog: ArrayLike | pd.DataFrame | None = None,
                transform: bool = True) -> np.ndarray:
        """Marginal prediction ``X beta``, matching the reference.

        With a formula-fitted model and ``transform=True``, ``exog`` may be a
        DataFrame of new data: the stored patsy design is applied to it, so
        transformations, categorical codings and the intercept are rebuilt the
        same way they were at fit time. Passing raw new data used to be a shape
        error, because only the numeric design was retained.
        """
        raw = isinstance(exog, (pd.DataFrame, dict))
        if exog is None:
            X = self.exog
        elif transform and raw and self._ensure_design_info():
            from patsy import dmatrix
            X = np.asarray(dmatrix(self._design_info, exog,
                                   return_type="matrix"), dtype=float)
        elif transform and raw and self.formula is not None:
            # Formula-fitted, but the design metadata is gone and cannot be
            # rebuilt -- the frame it was fitted on was not retained. Telling
            # the caller to "pass a DataFrame and leave transform=True" here,
            # as this once did, advises exactly the thing that just failed.
            reason = getattr(self, "_design_rebuild_error", None)
            if reason == "no-frame" or reason is None:
                cause = (
                    "it is normally rebuilt from the rows the model was "
                    "fitted on, and those were dropped -- by "
                    "save(with_data=False), or by a pickle written before "
                    "this was supported.")
            else:
                cause = (
                    "it is normally rebuilt by re-running the formula against "
                    "the rows the model was fitted on, and that re-run failed: "
                    f"{reason}. The training rows were retained, so this is "
                    "the formula itself no longer evaluating -- most often a "
                    "transform defined in the session that saved the model "
                    "(a lambda or a closure cannot be pickled; a "
                    "module-level function can).")
            raise ValueError(
                "this model was fitted from the formula "
                f"{self.formula!r}, but its patsy design metadata is not "
                "available, so raw new data cannot be converted into a design "
                f"matrix. patsy cannot pickle a DesignInfo (pydata/patsy#26); "
                f"{cause} Either re-fit the model, or build the design "
                "yourself and pass it with transform=False -- but build "
                "it against the *training* frame first so any stateful "
                "transform learns the training state: "
                "`d = patsy.dmatrix(<rhs>, training_data).design_info` then "
                "`result.predict(patsy.dmatrix(d, new_data), "
                "transform=False)`. Building `patsy.dmatrix(<rhs>, new_data)` "
                "directly re-learns centring means, spline knots and "
                "categorical levels from the new data, which silently fits "
                "the prediction to a different design.")
        else:
            X = np.asarray(exog, float)
            if X.ndim == 1:
                X = X[:, None]
        if X.shape[1] != self.k_fe:
            hint = ""
            if self.formula is not None and not raw:
                hint = (" This model was fitted from a formula: pass a "
                        "DataFrame of new data with transform=True to have "
                        "the design rebuilt for you.")
            raise ValueError(
                f"exog has {X.shape[1]} columns, expected {self.k_fe}.{hint}")
        if isinstance(params, MixedLMParams):
            fe = np.asarray(params.fe_params, float)
        else:
            fe = np.asarray(params, float).ravel()[:self.k_fe]
        return X @ fe

    @property
    def endog_names(self) -> str:
        return self._endog_name

    def initialize(self) -> None:
        return None

    def score(self, params: MixedLMParams | ArrayLike,
              profile_fe: bool = True,
              reml: bool | None = None) -> np.ndarray:
        """Gradient of :meth:`loglike`, in the internal ``theta`` coordinates.

        These are the entries of the relative covariance factor, not
        statsmodels' packed covariance parameters, so the two are not
        numerically comparable term by term. docs/LIMITATIONS.md says so
        explicitly; the coordinates are documented rather than translated
        because the chain rule through a Cholesky factor is not invertible at
        the boundary, which is exactly where these are most often inspected.
        """
        if not profile_fe:
            raise NotImplementedError(
                "profile_fe=False is not available; see loglike().")
        theta = self._theta_from_packed(params)
        _, g = self._core().deviance_grad(list(theta), self._reml_flag(reml))
        return -0.5 * np.asarray(g, float)

    def hessian(self, params: MixedLMParams | ArrayLike,
                reml: bool | None = None) -> np.ndarray:
        """Hessian of :meth:`loglike` in the internal ``theta`` coordinates.

        Shape is ``(k_re2, k_re2)`` -- the profiled criterion has no fixed-effect
        or scale coordinates, so this is not the reference's full
        ``(k_fe + k_re2 + 1)`` square. Use ``MixedLMResults.cov_params()`` for
        fixed-effect inference.
        """
        from ._fit import _profiled_hessian
        theta = self._theta_from_packed(params)
        return -0.5 * _profiled_hessian(self._core(), theta,
                                        self._reml_flag(reml))

    def information(self, params: MixedLMParams | ArrayLike,
                    reml: bool | None = None) -> np.ndarray:
        return -self.hessian(params, reml=reml)

    # -- persistence --------------------------------------------------------
    def __getstate__(self):
        """Drop what cannot be pickled, and keep what can rebuild it.

        The compiled core and patsy's ``DesignInfo`` both go. The frame the
        model was fitted on stays, because :meth:`_ensure_design_info` rebuilds
        the design from it on the other side, which is what keeps ``predict``
        working on raw new data after a round trip.

        ``MixedLMResults.save(..., with_data=False)`` drops that frame for
        callers who would rather have a small file, and ``predict`` then raises
        an error saying precisely that instead of advising the impossible.
        """
        state = self.__dict__.copy()
        state["_core_cache"] = None
        state["_design_info"] = None
        state["_re_design_info"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("_design_info", None)
        self.__dict__.setdefault("_re_design_info", None)
        # Pickles written by earlier versions carry neither, and a model whose
        # design was built without missing-data filtering does not need them.
        self.__dict__.setdefault("_design_rows", None)
        self.__dict__.setdefault("_formula_namespace", None)
        self.__dict__.setdefault("_design_rebuild_error", None)

    def _training_frame(self):
        """The exact rows the design was fitted on.

        Not ``data_frame`` itself: when ``missing="drop"`` removed rows, it
        removed them across the fixed *and* random formulas together, and
        ``data_frame`` still holds them. Rebuilding on the unfiltered frame
        lets a stateful transform relearn different state -- ``center()`` over
        a different mean, ``C()`` over an extra level -- so predictions move,
        or the design comes back with the wrong number of columns.
        """
        frame = self.data_frame
        if frame is None:
            return None
        rows = self._design_rows
        if rows is None:
            return frame
        return frame.iloc[np.asarray(rows, dtype=np.intp)]

    def _restore_eval_env(self):
        """Rebuild an eval environment for the captured formula names.

        Returns ``(eval_env, missing_names)``. Anything that did not survive
        the pickle is reported by name so the caller learns which transform is
        gone instead of being told the training data was discarded.
        """
        ns = self._formula_namespace
        if not ns:
            return None, []
        missing = sorted(v.name for v in ns.values()
                         if isinstance(v, _Unpicklable))
        usable = {k: v for k, v in ns.items()
                  if not isinstance(v, _Unpicklable)}
        if not usable:
            return None, missing
        from patsy import EvalEnvironment
        return EvalEnvironment([usable]), missing

    def _ensure_design_info(self) -> bool:
        """Rebuild patsy's design metadata after a pickle round-trip.

        patsy declines to pickle a ``DesignInfo`` (pydata/patsy#26). Dropping it
        and stopping there breaks ``predict`` on raw new data across a pickle
        boundary -- and the resulting error told the caller to pass raw new
        data, which is exactly what had just failed.

        So the design is *reconstructed* instead: the formula is re-run against
        the rows the model was fitted on, which reproduces the identical
        ``DesignInfo``, stateful transforms included (``C()`` levels,
        ``center()`` means, spline knots). Both halves of that matter -- the
        same formula *and* the same rows.

        Returns True when a fixed-effect design is available afterwards; the
        reason for a False is kept in ``_design_rebuild_error`` for predict().
        """
        if self._design_info is not None:
            return True
        if self.formula is None:
            self._design_rebuild_error = "no formula"
            return False
        frame = self._training_frame()
        if frame is None:
            self._design_rebuild_error = "no-frame"
            return False

        from patsy import dmatrices, dmatrix

        eval_env, missing_names = self._restore_eval_env()
        kwargs = {} if eval_env is None else {"eval_env": eval_env}
        try:
            # NA_action="raise": these rows are the complete cases already, so
            # anything patsy would drop here is a bug worth surfacing, not
            # silently filtering a second time.
            _, design = dmatrices(self.formula, frame, return_type="matrix",
                                  NA_action="raise", **kwargs)
            fe_info = design.design_info
            re_info = self._re_design_info
            rf = self.re_formula
            if rf is not None and str(rf).strip() not in ("1", "~1", ""):
                re_info = dmatrix(str(rf), frame, return_type="matrix",
                                  NA_action="raise", **kwargs).design_info

            # The rebuild has to reproduce the design that was *fitted*, not
            # merely produce a design. A stateful transform that relearned
            # different state shows up here as a different column count or
            # different column names, and predicting through it would return
            # plausible numbers from the wrong model.
            self._check_rebuilt(fe_info, np.asarray(design).shape[0])
        except Exception as exc:
            # Assigned only on full success. Keeping a half-rebuilt design
            # leaves the model in a state where the fixed part was relearned
            # and the random part was not, which is worse than no design.
            self._design_rebuild_error = (
                f"{type(exc).__name__}: {exc}"
                + (f" (formula names not carried through the pickle: "
                   f"{missing_names})" if missing_names else ""))
            return False
        self._design_info = fe_info
        self._re_design_info = re_info
        return True

    def _check_rebuilt(self, fe_info, n_rows) -> None:
        """Assert the rebuilt design matches the one that was fitted."""
        names = list(fe_info.column_names)
        if len(names) != self.k_fe:
            raise ValueError(
                f"rebuilding the design from the training rows produced "
                f"{len(names)} fixed-effect columns, but the model was fitted "
                f"with {self.k_fe} ({names} against "
                f"{list(self.exog_names or [])}). A stateful transform in "
                f"{self.formula!r} relearned different state; the rebuilt "
                "design describes a different model and is discarded.")
        expected = list(self.exog_names or [])
        if expected and names != expected:
            raise ValueError(
                f"rebuilding the design produced different column names: "
                f"{names} against the fitted {expected}.")
        if n_rows != self.exog.shape[0]:
            raise ValueError(
                f"rebuilding the design produced {n_rows} rows, but the model "
                f"was fitted on {self.exog.shape[0]}.")

    def fit_regularized(self, *args: Any, **kwargs: Any) -> MixedLMResults:
        raise NotImplementedError(
            "fit_regularized (L1-penalised fixed effects) is not implemented "
            "in this release; see docs/LIMITATIONS.md. Refusing rather than "
            "silently fitting the unpenalised model.")

    def get_distribution(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "get_distribution is not implemented in this release; see "
            "docs/LIMITATIONS.md.")

    def get_scale(self, fe_params: ArrayLike | None = None,
                  cov_re: ArrayLike | None = None,
                  vcomp: ArrayLike | None = None) -> float:
        raise NotImplementedError(
            "get_scale is an internal statsmodels helper tied to its "
            "parameterisation; use MixedLMResults.scale instead."
        )

    @property
    def data(self) -> _ModelData:
        """Minimal stand-in for statsmodels' model data namespace.

        Code that reaches for ``model.data.xnames`` or ``exog_re_names`` to
        label output is common enough that its absence breaks otherwise
        portable plotting and reporting helpers.
        """
        return _ModelData(self)

    @property
    def exog_re_names(self) -> list[str]:
        return list(self._exog_re_names)

    def group_list(self, array: ArrayLike) -> list[np.ndarray]:
        """Split ``array`` by group, as the reference does.

        This was a list-valued property returning the labels, which is a
        different thing under the same name: code calling
        ``model.group_list(resid)`` got a TypeError.
        """
        arr = np.asarray(array)
        return [arr[self._codes == i] for i in range(self.n_groups)]

    @property
    def df_resid(self) -> int:
        return self.nobs - self.k_fe

    @property
    def df_modelwc(self) -> int:
        return self.k_fe + self.k_re2 + self.k_vc


class _ModelData:
    """The subset of statsmodels' `model.data` that labelling code actually uses."""

    __slots__ = ("_model",)

    def __init__(self, model: MixedLM) -> None:
        self._model = model

    @property
    def xnames(self) -> list[str]:
        return list(self._model.exog_names)

    @property
    def ynames(self) -> str:
        return self._model.endog_names

    @property
    def exog_re_names(self) -> list[str]:
        return list(self._model._exog_re_names)

    @property
    def param_names(self) -> list[str]:
        return (list(self._model.exog_names)
                + _re_param_names(list(self._model._exog_re_names)))

    @property
    def endog(self) -> np.ndarray:
        return self._model.endog

    @property
    def exog(self) -> np.ndarray:
        return self._model.exog


def _vech_col(mat):
    """Column-major lower-triangle packing -- the theta convention."""
    q = mat.shape[0]
    return np.array([mat[r, c] for c in range(q) for r in range(c, q)])


class MixedLMResults:
    """Results of a :class:`MixedLM` fit."""

    def __init__(self, model: MixedLM, res: dict[str, Any]) -> None:
        # Annotated one by one rather than left to inference: `res` is a
        # dict[str, Any], so every attribute taken from it would otherwise be
        # Any, and a checker would accept nonsense on all of them while
        # py.typed advertised the opposite.
        self.model: MixedLM = model
        self._res: dict[str, Any] = res

        self.fe_params: np.ndarray = res["beta"]
        self.cov_re: np.ndarray = res["cov_re"]
        self.cov_re_unscaled: np.ndarray = res["cov_re_unscaled"]
        self.scale: float = res["scale"]
        self.vcomp: np.ndarray = np.zeros(0)
        self.converged: bool = res["converged"]
        self.singular: bool = res["singular"]
        self.reml: bool = res["reml"]
        self.nobs: int = res["n"]
        self.k_fe: int = res["p"]
        self.k_re: int = res["q"]
        self.k_re2: int = model.k_re2
        self.k_vc: int = 0
        self.method: str = "REML" if res["reml"] else "ML"
        self.use_t: bool = False

        self._cov_beta: np.ndarray = res["cov_beta"]
        self._random_effects: np.ndarray = res["random_effects"]
        self._deviance: float = res["deviance"]
        self._bse_re_unscaled: np.ndarray | None = res["bse_re_unscaled"]
        self._compute_bse_re: Any = res.get("compute_bse_re")

    # -- persistence --------------------------------------------------------
    #
    # The fit result carries the compiled core and a closure over it for the
    # lazy standard errors; neither can be pickled, which made a fitted model
    # unusable with multiprocessing, joblib caching or a saved analysis.
    # Forcing the standard errors and dropping the core leaves a fully
    # self-contained result -- everything MixedLMResults exposes is already
    # materialised by then.
    def __getstate__(self):
        _ = self._bse_re_packed          # realise before discarding the core
        state = self.__dict__.copy()
        state["_compute_bse_re"] = None
        res = dict(state["_res"])
        res.pop("core", None)
        res.pop("compute_bse_re", None)
        res["bse_re_unscaled"] = self._bse_re_unscaled
        state["_res"] = res
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, path: str | os.PathLike[str],
             with_data: bool = True) -> None:
        """Pickle this result to ``path``.

        ``with_data`` keeps the frame the model was fitted on, which is what
        lets :meth:`predict` rebuild a patsy design for raw new data after the
        file is loaded again. Pass False for a small file when only the fitted
        numbers are wanted; prediction from a pre-built design matrix still
        works, and prediction from raw data then raises an error that says why.
        """
        import pickle
        if with_data:
            with open(path, "wb") as fh:
                pickle.dump(self, fh)
            return

        frame = self.model.data_frame
        self.model.data_frame = None
        try:
            with open(path, "wb") as fh:
                pickle.dump(self, fh)
        finally:
            self.model.data_frame = frame

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> MixedLMResults:
        """Load a result written by :meth:`save`."""
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # -- parameter vector, statsmodels packing ------------------------------
    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.fe_params, _vech_row(self.cov_re_unscaled),
                               self.vcomp])

    @property
    def bse_fe(self) -> np.ndarray:
        return np.sqrt(np.diag(self._cov_beta))

    @property
    def _bse_re_packed(self) -> np.ndarray:
        """Standard errors of the *unscaled* covariance parameters.

        This is what goes in the tail of ``params`` and ``bse``, matching the
        reference's packing. Computed on first access, not during the fit: the
        profiled Hessian costs 2 * n_theta extra gradient evaluations, and most
        callers only look at the fixed effects.
        """
        if self._bse_re_unscaled is None and self._compute_bse_re is not None:
            self._bse_re_unscaled = self._compute_bse_re()
            self._compute_bse_re = None
        m = self._bse_re_unscaled
        if m is None:
            return np.full(self.k_re2, np.nan)
        return _vech_row(m)

    @property
    def bse_re(self) -> np.ndarray:
        """Standard errors of the variance parameters, as the reference defines them.

        statsmodels computes ``sqrt(scale * diag(cov_params())[k_fe:])``, i.e.
        ``sqrt(scale)`` times the standard errors of the *unscaled* covariance
        parameters that ``bse`` reports. That is a different quantity from
        ``bse[k_fe:]``, and this property used to return the latter -- so a
        caller reading ``bse_re`` got numbers a factor of ``sqrt(scale)`` out.

        Note the reference's own inconsistency, preserved here for
        compatibility: the value displayed alongside these errors in
        ``summary()`` is ``cov_re``, which is ``scale`` times the unscaled
        parameter, so the tabulated estimate and its standard error differ by a
        further factor of ``sqrt(scale)``. Use :attr:`bse_cov_re` for standard
        errors on the same scale as :attr:`cov_re`.

        The sampling distribution of a variance parameter is strongly skewed
        unless the sample size is large, and degenerate at the boundary; see
        :attr:`singular`.
        """
        return np.sqrt(self.scale) * self._bse_re_packed

    @property
    def bse_cov_re(self) -> np.ndarray:
        """Standard errors on the same scale as :attr:`cov_re`.

        Not a statsmodels attribute. ``cov_re = scale * cov_re_unscaled``, so
        these are ``scale`` times the unscaled standard errors -- the internally
        consistent pairing that :attr:`bse_re` does not provide.
        """
        return self.scale * self._bse_re_packed

    @property
    def bse(self) -> np.ndarray:
        """Standard errors of the packed parameter vector.

        The tail is on the unscaled covariance parameterisation, matching
        :attr:`params` and the reference's ``bse``.
        """
        return np.concatenate([self.bse_fe, self._bse_re_packed])

    @property
    def tvalues(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.params / self.bse

    @property
    def pvalues(self) -> np.ndarray:
        from scipy import stats
        with np.errstate(invalid="ignore", divide="ignore"):
            return 2 * stats.norm.sf(np.abs(self.tvalues))

    @property
    def llf(self) -> float:
        return -0.5 * self._deviance

    @property
    def df_modelwc(self) -> int:
        return self.k_fe + self.k_re2 + self.k_vc

    @property
    def aic(self) -> float:
        if self.reml:
            return np.nan
        return -2 * (self.llf - (self.params.size + 1))

    @property
    def bic(self) -> float:
        if self.reml:
            return np.nan
        df = self.params.size + 1
        return -2 * self.llf + np.log(self.nobs) * df

    @property
    def fittedvalues(self) -> np.ndarray:
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
    def resid(self) -> np.ndarray:
        return self.model.endog - self.fittedvalues

    @property
    def random_effects(self) -> dict[Any, pd.Series]:
        """Conditional modes, one entry per group, as statsmodels returns them."""
        names = self.model._exog_re_names
        return {lab: pd.Series(self._random_effects[i], index=names)
                for i, lab in enumerate(self.model.group_labels)}

    @property
    def random_effects_cov(self) -> dict[Any, pd.DataFrame]:
        """Conditional covariance of each group's random effects, given the data.

        This is *not* the population covariance ``cov_re``: it is
        ``Var(b_i | y_i)``, which shrinks as a group accumulates observations.
        Returning ``cov_re`` for every group -- as this once did -- reported the
        prior where the posterior was asked for, and for a random intercept the
        two differ by the whole factor ``1 / (1 + n_i * tau^2 / sigma^2)``.

        With ``G = cov_re`` and ``s = scale``,

            Var(b_i | y_i) = G - G Z_i' (Z_i G Z_i' + s I)^-1 Z_i G
                           = G - G (s I + Z_i'Z_i G)^-1 Z_i'Z_i G,

        the second form needing only ``q x q`` work per group. It is used
        because it stays valid when ``G`` is singular, which is exactly the
        boundary case a mixed model most often lands on.
        """
        G = np.asarray(self.cov_re, float)
        q = G.shape[0]
        names = list(self.model._exog_re_names)
        m = self.model.n_groups
        codes = self.model._codes
        Z = self.model.exog_re

        # Z_i'Z_i for every group at once: q^2 weighted bincounts rather than a
        # Python loop over groups.
        ztz = np.empty((m, q, q))
        for a in range(q):
            for b in range(a + 1):
                v = np.bincount(codes, weights=Z[:, a] * Z[:, b], minlength=m)
                ztz[:, a, b] = v
                ztz[:, b, a] = v

        gz = ztz @ G                                  # (m, q, q)
        lhs = self.scale * np.eye(q) + gz
        try:
            covs = G - G @ np.linalg.solve(lhs, gz)
        except np.linalg.LinAlgError:                 # pragma: no cover
            covs = np.stack([G - G @ np.linalg.pinv(lhs[i]) @ gz[i]
                             for i in range(m)])
        # A fresh frame per group: these used to be the same mutable object,
        # so editing one group's table edited every group's.
        return {lab: pd.DataFrame(covs[i], index=names, columns=names)
                for i, lab in enumerate(self.model.group_labels)}

    def conf_int(self, alpha: float = 0.05,
                 cols: ArrayLike | None = None) -> np.ndarray:
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.params - z * self.bse
        hi = self.params + z * self.bse
        out = np.column_stack([lo, hi])
        return out if cols is None else out[cols]

    def cov_params(self, r_matrix: ArrayLike | None = None,
                   column: ArrayLike | None = None,
                   scale: float | None = None,
                   cov_p: ArrayLike | None = None) -> np.ndarray:
        """Covariance of the packed parameter vector.

        The fixed-effect block is the GLS covariance
        ``sigma^2 (X' V(theta_hat)^-1 X)^-1``, evaluated at the fitted variance
        parameters. That is the standard mixed-model fixed-effect covariance and
        what both lme4 and statsmodels report, but it is *conditional on*
        ``theta_hat``: it does not propagate uncertainty in the variance
        parameters, so it is not in general the fixed-effect block of the
        inverse full observed information, and it is mildly anti-conservative in
        small samples. Kenward-Roger and Satterthwaite corrections exist for
        exactly this reason and are not implemented here; see
        docs/LIMITATIONS.md.

        The variance-component rows carry the delta-method variances on the
        diagonal and NaN off-diagonal, rather than a fabricated full covariance.

        ``r_matrix``, ``column``, ``scale`` and ``cov_p`` follow the reference.
        """
        k = self.params.size
        out = np.full((k, k), np.nan)
        p = self.k_fe
        out[:p, :p] = self._cov_beta if cov_p is None else np.asarray(cov_p)[:p, :p]
        bre = self._bse_re_packed
        for i in range(len(bre)):
            out[p + i, p + i] = bre[i] ** 2
        if scale is not None:
            out = out * scale
        if r_matrix is not None:
            R = np.atleast_2d(np.asarray(r_matrix, float))
            if R.shape[1] == k and not np.any(R[:, p:]):
                # The variance-component block is structurally NaN off the
                # diagonal, and 0 * NaN is NaN, so a contrast that touches only
                # the fixed effects would otherwise come back all-NaN.
                R = R[:, :p]
            if R.shape[1] == p:
                return R @ out[:p, :p] @ R.T
            return R @ out @ R.T
        if column is not None:
            col = np.atleast_1d(column)
            return out[np.ix_(col, col)] if col.size > 1 else out[col[0], col[0]]
        return out

    # -- hypothesis tests ---------------------------------------------------
    #
    # Restricted to the fixed effects, as in the reference: the variance
    # parameters sit on a bounded space and their Wald statistics are not
    # chi-square distributed near the boundary.
    def _fe_contrast(self, r_matrix):
        if isinstance(r_matrix, str):
            R = _parse_constraints(r_matrix, list(self.model.exog_names))
        else:
            R = np.atleast_2d(np.asarray(r_matrix, float))
        if R.shape[1] == self.k_fe + 1:
            R, qv = R[:, :-1], R[:, -1]
        else:
            qv = np.zeros(R.shape[0])
        if R.shape[1] != self.k_fe:
            raise ValueError(
                f"constraint matrix has {R.shape[1]} columns, expected "
                f"{self.k_fe} (one per fixed effect), optionally plus a "
                "right-hand-side column")
        return R, qv

    def t_test(self, r_matrix: str | ArrayLike,
               use_t: bool | None = None) -> Any:
        """Wald test of ``R beta = q`` on the fixed effects, term by term."""
        from scipy import stats
        R, qv = self._fe_contrast(r_matrix)
        eff = R @ self.fe_params - qv
        se = np.sqrt(np.diag(R @ self._cov_beta @ R.T))
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = eff / se
        use_t = self.use_t if use_t is None else bool(use_t)
        if use_t:
            pv = 2 * stats.t.sf(np.abs(stat), self.df_resid)
        else:
            pv = 2 * stats.norm.sf(np.abs(stat))
        return _ContrastResults(eff, se, stat, pv, use_t, self.df_resid)

    def wald_test(self, r_matrix: str | ArrayLike, use_f: bool = False,
                  scalar: bool = True) -> Any:
        """Joint Wald test of ``R beta = q`` on the fixed effects."""
        from scipy import stats
        R, qv = self._fe_contrast(r_matrix)
        eff = R @ self.fe_params - qv
        V = R @ self._cov_beta @ R.T
        stat = float(eff @ np.linalg.solve(V, eff))
        df = R.shape[0]
        if use_f:
            fval = stat / df
            return _ContrastResults(eff, None, fval,
                                    float(stats.f.sf(fval, df, self.df_resid)),
                                    True, self.df_resid, df_num=df)
        return _ContrastResults(eff, None, stat,
                                float(stats.chi2.sf(stat, df)),
                                False, self.df_resid, df_num=df)

    def f_test(self, r_matrix: str | ArrayLike) -> Any:
        """Joint F test of ``R beta = q`` on the fixed effects."""
        return self.wald_test(r_matrix, use_f=True)

    # -- labelled views -----------------------------------------------------
    @property
    def diagnostics(self) -> dict[str, Any]:
        """What the optimiser did, and how the result was certified.

        Not a statsmodels attribute. It exists because a fit that fails to
        certify needs to leave the caller something to act on: which optimiser
        ran, the projected gradient it stopped at, the tolerance that was
        required, and how many evaluations it took.
        """
        return dict(self._res["diagnostics"])

    @property
    def param_names(self) -> list[str]:
        """Names for every entry of :attr:`params`."""
        return (list(self.model.exog_names)
                + _re_param_names(list(self.model._exog_re_names)))

    @property
    def fe_params_labelled(self) -> pd.Series:
        """:attr:`fe_params` as a named Series, for name-based access."""
        return pd.Series(self.fe_params, index=list(self.model.exog_names))

    @property
    def params_labelled(self) -> pd.Series:
        """:attr:`params` as a named Series."""
        return pd.Series(self.params, index=self.param_names)

    @property
    def params_object(self) -> MixedLMParams:
        """The fit as a :class:`MixedLMParams`, as the reference exposes it."""
        return MixedLMParams.from_components(
            fe_params=np.asarray(self.fe_params, float),
            cov_re=np.asarray(self.cov_re_unscaled, float),
            vcomp=np.asarray(self.vcomp, float))

    @property
    def df_resid(self) -> int:
        return self.nobs - self.k_fe

    def predict(self, exog: ArrayLike | pd.DataFrame | None = None,
                transform: bool = True) -> np.ndarray:
        return self.model.predict(self.params, exog=exog, transform=transform)

    # -- reporting ----------------------------------------------------------
    def summary(
        self,
        yname: str | None = None,
        xname_fe: Sequence[str] | None = None,
        xname_re: Sequence[str] | None = None,
        title: str | None = None,
        alpha: float = 0.05,
    ) -> Any:
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
            ("Singular fit:", "Yes" if self.singular else "No"),
        ]
        for (la, lv), (ra, rv) in zip(left, right, strict=False):
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

        # Variance components. The displayed value is `cov_re` -- the actual
        # estimated covariance -- exactly as the reference displays it. This
        # once printed `cov_re_unscaled`, the ratio to the residual variance,
        # under the label "Group Var": on sleepstudy that showed 0.935 where
        # both lme4 and statsmodels report 612.1.
        vn = _re_param_names(re_names)
        vals = _vech_row(self.cov_re)
        ses = self.bse_re
        for i, nm in enumerate(vn):
            se = ses[i] if i < len(ses) else np.nan
            se_s = f"{se:10.3f}" if np.isfinite(se) else f"{'':>10s}"
            lines.append(f"{nm[:20]:<20s}{vals[i]:10.3f}{se_s}"
                         f"{'':>9s}{'':>9s}{'':>10s}{'':>10s}")
        lines.append("=" * 78)
        if self.singular:
            lines.append(
                "Singular fit: a variance component is estimated at the "
                "boundary (zero). The estimate is legitimate, but Wald")
            lines.append(
                "standard errors and p-values for the variance parameters "
                "do not apply there --")
            lines.append(
                "this is a boundary optimum, not a convergence failure.")
        return _SummaryText("\n".join(lines))

    def __str__(self):
        return str(self.summary())

    # -- explicitly unimplemented -------------------------------------------
    def profile_re(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "profile_re is not implemented in this release; see docs/LIMITATIONS.md"
        )

    def bootstrap(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "bootstrap is not implemented in this release; see docs/LIMITATIONS.md"
        )

    def get_distribution(self, *args: Any, **kwargs: Any) -> Any:
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


def _parse_constraints(spec, names):
    """Turn "x1 = 0, x2 - x3 = 0" into a contrast matrix over `names`."""
    rows = []
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        lhs, _, rhs = clause.partition("=")
        row = np.zeros(len(names) + 1)
        row[-1] = float(rhs) if rhs.strip() else 0.0
        for piece in lhs.replace("-", "+-").split("+"):
            piece = piece.strip()
            if not piece:
                continue
            sign = -1.0 if piece.startswith("-") else 1.0
            piece = piece.lstrip("-").strip()
            coef, _, nm = piece.rpartition("*")
            nm = nm.strip()
            if nm not in names:
                raise ValueError(f"unknown term {nm!r} in constraint {spec!r}")
            row[names.index(nm)] += sign * (float(coef) if coef.strip() else 1.0)
        rows.append(row)
    if not rows:
        raise ValueError(f"no constraints parsed from {spec!r}")
    return np.array(rows)


class _ContrastResults:
    """Minimal stand-in for statsmodels' ContrastResults."""

    def __init__(self, effect: ArrayLike, sd: ArrayLike | None,
                 statistic: Any, pvalue: Any, use_t: bool, df_denom: int,
                 df_num: int | None = None) -> None:
        self.effect = np.atleast_1d(effect)
        self.sd = sd
        self.statistic = statistic
        self.pvalue = pvalue
        self.use_t = use_t
        self.df_denom = df_denom
        self.df_num = df_num
        self.tvalue = statistic
        self.fvalue = statistic if (use_t and df_num is not None) else None

    def summary(self) -> str:
        return str(self)

    def __str__(self):
        if self.df_num is not None:
            kind = "F" if self.use_t else "chi2"
        else:
            kind = "t" if self.use_t else "z"
        tail = f", df_num={self.df_num}" if self.df_num is not None else ""
        return (f"<{kind}={np.round(self.statistic, 4)}, "
                f"p={np.round(self.pvalue, 4)}, df_denom={self.df_denom}{tail}>")

    __repr__ = __str__


class _SummaryText:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self):
        return self._text

    def __repr__(self):
        return self._text

    def as_text(self) -> str:
        return self._text


def mixedlm(
    formula: str,
    data: pd.DataFrame,
    groups: ArrayLike | str,
    re_formula: str | None = None,
    vc_formula: Any = None,
    subset: ArrayLike | None = None,
    use_sparse: bool = False,
    missing: str = "none",
    *args: Any,
    **kwargs: Any,
) -> MixedLM:
    """Formula interface, matching ``statsmodels.formula.api.mixedlm``."""
    return MixedLM.from_formula(formula, data, *args, re_formula=re_formula,
                                vc_formula=vc_formula, subset=subset,
                                use_sparse=use_sparse, missing=missing,
                                groups=groups, **kwargs)
