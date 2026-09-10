"""mixedlm-rs: fast linear mixed-effects models for Python.

lme4 speed, statsmodels API, no R required.

    - import statsmodels.api as sm            ->  import mixedlm_rs as mlm
    - from statsmodels.regression.mixed_linear_model import MixedLM
    + from mixedlm_rs import MixedLM
"""

from ._fit import ConvergenceWarning, ExperimentalWarning
from ._install import install, is_installed, uninstall
from ._mixedlm_rs import LmmCore
from .mixed_linear_model import (
    MixedLM,
    MixedLMParams,
    MixedLMResults,
    VCSpec,
    mixedlm,
)

__version__ = "0.1.1"
__statsmodels_version__ = "0.15.0"

__all__ = [
    "ConvergenceWarning",
    "ExperimentalWarning",
    "LmmCore",
    "MixedLM",
    "MixedLMParams",
    "MixedLMResults",
    "VCSpec",
    "__statsmodels_version__",
    "__version__",
    "install",
    "is_installed",
    "mixedlm",
    "uninstall",
]
