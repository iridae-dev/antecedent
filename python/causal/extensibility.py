"""Typing Protocols for DESIGN §25.4 slow-path Python callbacks.

These are documentation / type-checking aids. Native bridges accept any
callable matching the shapes below; they reacquire the GIL and force serial
execution (non-native performance).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class CiBatchTest(Protocol):
    """Batch conditional-independence test.

    Parameters
    ----------
    columns:
        List of 1-d float64 columns.
    queries:
        List of ``(x, y, z_idxs)`` where ``z_idxs`` is a list of conditioning
        column indexes.
    """

    def __call__(
        self,
        columns: Sequence[NDArray[np.float64]],
        queries: Sequence[tuple[int, int, list[int]]],
    ) -> Sequence[tuple[float, float]]:
        """Return ``(statistic, p_value)`` per query."""


@runtime_checkable
class MechanismWrapper(Protocol):
    """Per-node mechanism override for GCM sampling."""

    def sample_noise(self, n: int) -> NDArray[np.float64]:
        """Draw structural noise of length ``n``."""

    def evaluate(
        self,
        parents: Sequence[NDArray[np.float64]],
        noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map parents + noise → child values (length ``noise``)."""


@runtime_checkable
class UtilityFn(Protocol):
    """Batch utility for decision evaluation."""

    def __call__(
        self,
        actions: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return flat utilities of length ``len(actions) * len(outcomes)``."""


@runtime_checkable
class EffectValidator(Protocol):
    """Custom effect refuter returning a report dict."""

    def __call__(
        self,
        *,
        ate: float,
        se_analytic: float,
        method: str,
        adjustment_set: list[str],
    ) -> dict:
        """Must include ``passed: bool``; optional ``refuted_ate``, ``comparison``."""


__all__ = [
    "CiBatchTest",
    "EffectValidator",
    "MechanismWrapper",
    "UtilityFn",
]
