"""mixedlm-rs: fast linear mixed-effects models for Python.

lme4 speed, statsmodels API, no R required.

    - import statsmodels.api as sm            ->  import mixedlm_rs as mlm
    - from statsmodels.regression.mixed_linear_model import MixedLM
    + from mixedlm_rs import MixedLM
"""

from ._mixedlm_rs import LmmCore
from ._fit import ConvergenceWarning
from .mixed_linear_model import (
    MixedLM,
    MixedLMParams,
    MixedLMResults,
    VCSpec,
    mixedlm,
)
from ._install import install, uninstall, is_installed

__version__ = "0.1.0"
__statsmodels_version__ = "0.15.0"

__all__ = [
    "MixedLM", "MixedLMResults", "MixedLMParams", "VCSpec", "mixedlm",
    "LmmCore", "ConvergenceWarning", "install", "uninstall", "is_installed",
    "__version__", "__statsmodels_version__",
]
