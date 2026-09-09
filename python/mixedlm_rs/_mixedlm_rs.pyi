"""Type stubs for the compiled core.

``LmmCore`` is a **low-level interface**. It is exported so that the benchmarks
and the test suite can measure and check the criterion directly, and so that a
caller who wants the profiled deviance without the estimator wrapper can have
it. It is not the drop-in API: it takes prepared numeric designs, it does no
conditioning, no boundary escape and no convergence certification, and its
argument conventions can change between releases.

Use :class:`mixedlm_rs.MixedLM` unless you specifically want the raw criterion.

Every method validates its input and raises ``ValueError`` rather than
panicking: the crate is built with ``panic = "abort"``, so an unchecked index
would terminate the interpreter rather than raise. See
``tests/test_native_safety.py``.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

class LmmCore:
    """Theta-independent cross-products for one grouping factor.

    Parameters
    ----------
    y
        Response, shape ``(n,)``, C-contiguous float64.
    x
        Fixed-effects design, shape ``(n, p)``, C-contiguous float64.
    z
        Random-effects design, shape ``(n, q)``, C-contiguous float64.
    codes
        Group index per row, shape ``(n,)``, int64, values in ``0..n_groups-1``.
    n_groups
        Number of groups, positive.

    Raises
    ------
    ValueError
        If the shapes disagree, a design has zero columns, ``y`` is empty,
        ``n_groups`` is zero, or a group code is out of range.
    """

    def __new__(
        cls,
        y: NDArray[np.float64],
        x: NDArray[np.float64],
        z: NDArray[np.float64],
        codes: NDArray[np.int64],
        n_groups: int,
    ) -> LmmCore: ...
    @property
    def n(self) -> int:
        """Number of observations."""

    @property
    def p(self) -> int:
        """Number of fixed-effect columns."""

    @property
    def q(self) -> int:
        """Number of random-effect columns per group."""

    @property
    def m(self) -> int:
        """Number of groups."""

    @property
    def n_theta(self) -> int:
        """Length of ``theta``: ``q * (q + 1) // 2``."""

    def deviance(self, theta: Any, reml: bool = True) -> float:
        """Profiled deviance at ``theta``; ``inf`` where ``theta`` is infeasible.

        Raises ``ValueError`` if ``theta`` has the wrong length or is not
        finite -- a caller error, as distinct from an infeasible point.
        """

    def deviance_grad(
        self, theta: Any, reml: bool = True
    ) -> tuple[float, list[float]]:
        """Profiled deviance and its analytic gradient at ``theta``.

        Returns ``(inf, [nan, ...])`` for an infeasible ``theta``.
        """

    def lower_bounds(self) -> list[float]:
        """Lower bounds on ``theta``: 0 on the diagonal, ``-inf`` elsewhere."""

    def default_theta(self) -> list[float]:
        """lme4's starting value: the identity relative covariance factor."""

    def fit(
        self,
        starts: list[float],
        reml: bool = True,
        max_iter: int = 300,
        gtol: float = 1e-8,
        ftol: float = 1e-12,
    ) -> dict[str, Any]:
        """Minimise the profiled criterion with the in-crate projected L-BFGS.

        ``starts`` is a flat concatenation of candidate starting vectors, each
        of length ``n_theta``. Experimental; see ``docs/LIMITATIONS.md``.
        """

    def solution(self, theta: Any, reml: bool = True) -> dict[str, Any]:
        """Every derived quantity at ``theta``, without re-optimising.

        Raises ``RuntimeError`` if ``theta`` is infeasible.
        """
