"""Sampling uncertainty of the joint mechanism sensitivity range.

Claim ``sensitivity.joint_uncertainty``. The interval is the conservative
endpoint percentile bootstrap of the joint range. The assumption range itself
is ``joint_mechanism_sensitivity`` (``_joint_sensitivity``) and does not carry
this interval.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .._native import z_transport_joint_sensitivity_interval as _joint_interval
from ..execution import Execution
from ._controls import call_native
from ._joint_sensitivity import JointDeviation, _deviation, _stage

__all__ = ["JointSensitivityInterval", "joint_mechanism_sensitivity_interval"]


@dataclass(frozen=True)
class JointSensitivityInterval:
    """The sampling interval of the joint range with its calibration annotation.

    Iterating yields ``(lower, upper, coverage_target)``. ``calibration`` is the slot
    (``status`` is ``calibrated``, ``outside_measured_scope``, ``measured_below_nominal`` or
    ``not_measured``); it annotates the interval and never withholds it.
    """

    lower: float
    upper: float
    coverage_target: str
    level: float
    replicates_ok: int
    replicates_failed: int
    calibration: dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        return iter((self.lower, self.upper, self.coverage_target))


def joint_mechanism_sensitivity_interval(
    stage: Any,
    deviation: JointDeviation,
    *,
    replicates: int = 199,
    level: float = 0.95,
    execution: Execution | None = None,
) -> JointSensitivityInterval:
    """Conservative endpoint percentile bootstrap of the joint range.

    Returns the interval with its calibration slot. A level other than ``0.95`` or fewer than 199
    replicates is computed too; its slot says ``not_measured`` or ``outside_measured_scope``.
    Replicate counts outside 2 through 2000 are refused (``joint_sensitivity.bounds_exceeded``).
    """
    deviation = _deviation(deviation)
    lower, upper, coverage_target, calibration, ok, failed = call_native(
        execution,
        _joint_interval,
        _stage(stage),
        list(deviation._pairs),
        replicates=replicates,
        level=level,
        decision_threshold=deviation.decision_threshold,
        cancel="cancel",
        memory="memory_bytes",
    )
    return JointSensitivityInterval(
        lower, upper, coverage_target, level, ok, failed, json.loads(calibration)
    )
