"""Sampling uncertainty of the joint mechanism sensitivity range.

Claim ``sensitivity.joint_uncertainty``. The interval is the conservative
endpoint percentile bootstrap of the joint range. The assumption range itself
is ``joint_mechanism_sensitivity`` (``_joint_sensitivity``) and does not carry
this interval.
"""

from __future__ import annotations

from typing import Any

from .._native import z_transport_joint_sensitivity_interval as _joint_interval
from ._impl import _optional_non_negative
from ._joint_sensitivity import JointDeviation, _deviation, _stage

__all__ = ["joint_mechanism_sensitivity_interval"]


def joint_mechanism_sensitivity_interval(
    stage: Any,
    deviation: JointDeviation,
    *,
    replicates: int = 199,
    level: float = 0.95,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> tuple[float, float, str]:
    """Conservative endpoint percentile bootstrap of the joint range.

    Returns ``(lower, upper, coverage_target)``. Only level ``0.95`` and
    replicate counts from 199 through 2000 are admitted
    (``joint_sensitivity.bounds_exceeded``).
    """
    deviation = _deviation(deviation)
    lower, upper, coverage_target = _joint_interval(
        _stage(stage),
        list(deviation._pairs),
        replicates=replicates,
        level=level,
        decision_threshold=deviation.decision_threshold,
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    return (lower, upper, coverage_target)
