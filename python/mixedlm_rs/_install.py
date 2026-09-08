"""Alias ``statsmodels``' mixed-model entry points to this package.

For code you cannot edit -- a pipeline that reaches for
``statsmodels.regression.mixed_linear_model.MixedLM`` deep inside a dependency,
or a notebook you were handed.

    import mixedlm_rs
    mixedlm_rs.install()

    import statsmodels.api as sm          # MixedLM is now ours
    import statsmodels.formula.api as smf # smf.mixedlm is now ours

Unlike the other packages in this family, this does *not* replace the whole of
``statsmodels`` -- statsmodels does far more than mixed models, and shadowing all
of it would be reckless. Only ``MixedLM``, ``MixedLMResults``, ``MixedLMParams``
and ``smf.mixedlm`` are redirected; everything else in statsmodels is untouched
and keeps working.
"""

from __future__ import annotations

import warnings

_ORIGINALS = {}
_INSTALLED = False


def install(strict=True):
    """Redirect statsmodels' mixed-model names to this package.

    Parameters
    ----------
    strict : bool
        When True (default), raise if statsmodels is missing. When False, a
        missing statsmodels is a no-op -- useful in environments that only ever
        use this package directly.

    Returns
    -------
    bool
        True if the aliases were installed by this call.
    """
    global _INSTALLED
    if _INSTALLED:
        return False

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import statsmodels.regression.mixed_linear_model as smm
    except ImportError:
        if strict:
            raise ImportError(
                "install() aliases statsmodels' mixed-model entry points, so "
                "statsmodels must be importable. Pass strict=False to make this "
                "a no-op."
            )
        return False

    from .mixed_linear_model import (
        MixedLM, MixedLMParams, MixedLMResults, mixedlm,
    )

    targets = [
        (smm, "MixedLM", MixedLM),
        (smm, "MixedLMResults", MixedLMResults),
        (smm, "MixedLMParams", MixedLMParams),
        (sm, "MixedLM", MixedLM),
        (smf, "mixedlm", mixedlm),
    ]
    for mod, name, repl in targets:
        if hasattr(mod, name):
            _ORIGINALS[(mod.__name__, name)] = getattr(mod, name)
        setattr(mod, name, repl)

    _INSTALLED = True
    return True


def uninstall():
    """Put statsmodels' own implementations back."""
    global _INSTALLED
    if not _INSTALLED:
        return False
    import importlib

    for (modname, name), orig in _ORIGINALS.items():
        try:
            setattr(importlib.import_module(modname), name, orig)
        except ImportError:  # pragma: no cover
            warnings.warn(f"could not restore {modname}.{name}", stacklevel=2)
    _ORIGINALS.clear()
    _INSTALLED = False
    return True


def is_installed():
    return _INSTALLED
