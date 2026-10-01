"""Sampling uncertainty of the joint mechanism sensitivity range: closed in 2.2.

Record ``2.2B.X3.joint_sensitivity_uncertainty`` (carried forward). The one
declared composition, the conservative endpoint percentile bootstrap, has a
one-sided coverage target that the coverage harness cannot measure yet, so the
interval is not offered and :func:`joint_mechanism_sensitivity_interval` always
refuses with ``cell_not_licensed``. The joint assumption range itself is
``joint_mechanism_sensitivity`` (``_joint_sensitivity``).
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
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> None:
    """Sampling-uncertainty interval of the joint range: closed.

    Always raises with ``reason_code == "cell_not_licensed"``; the assumption
    range is not a confidence interval and sampling uncertainty is not offered
    in 2.2.
    """
    deviation = _deviation(deviation)
    _joint_interval(
        _stage(stage),
        list(deviation._pairs),
        decision_threshold=deviation.decision_threshold,
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
