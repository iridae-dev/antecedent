"""Joint mechanism deviations of the registered surrogate z-transport formula.

A prepared z stage (``identify_z_transport(...).prepare_exact(...)``) whose checked
formula is the registered surrogate ``sum_w P(Y | w, X, do(Z)) P(w | do(Z))`` can
perturb two of its factors together: the outcome kernel (``"outcome_kernel"``) and
the shared parent marginal (``"shared_parent_marginal"``). Each is contaminated
independently within its own fraction bound (a box, never a total budget). The
result is the exact assumption range over that box, the axis tipping points (the
2.1 one-factor values), and a tipping frontier with certified brackets.

The range is an assumption range, never a confidence interval. Sampling
uncertainty has one declared composition (the conservative endpoint percentile
bootstrap), and its interval route is closed: :func:`joint_mechanism_sensitivity_interval`
always refuses with ``cell_not_licensed``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._native import (
    consume_z_transport_joint_sensitivity_artifact as _consume_joint_artifact,
)
from .._native import export_z_transport_joint_sensitivity as _export_joint
from .._native import z_transport_joint_sensitivity as _joint
from .._native import z_transport_joint_sensitivity_interval as _joint_interval
from ..errors import CausalTypeError, CausalValueError
from ._impl import _non_negative, _optional_non_negative


@dataclass(frozen=True, slots=True)
class JointDeviation:
    """A declared joint deviation set and its analysis settings.

    ``fractions`` maps a factor name to its largest contamination fraction.
    Only ``outcome_kernel`` and ``shared_parent_marginal`` are supported; the
    other names (``treatment_mechanism``, ``fixed_graph_parent``,
    ``fixed_graph_conditional``, ``source_target_discrepancy``) exist so that their requests refuse with a
    reason code instead of being ignored. ``total_budget`` (a coupled budget)
    always refuses.
    """

    fractions: Mapping[str, float]
    decision_threshold: float | None = None
    total_budget: float | None = None
    tolerance: float = 1e-9
    frontier_points: int = 17
    max_operations: int = 100_000
    max_depth: int = 64
    max_memory_bytes: int = 64 * 1024 * 1024
    _pairs: tuple[tuple[str, float], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fractions, Mapping):
            raise CausalTypeError("fractions must map factor names to fraction bounds")
        pairs = []
        for name, value in self.fractions.items():
            if not isinstance(name, str):
                raise CausalTypeError("factor names must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalTypeError(f"fraction bound of {name!r} must be a number")
            pairs.append((name, float(value)))
        if self.decision_threshold is not None and not math.isfinite(self.decision_threshold):
            raise CausalValueError("decision_threshold must be finite")
        object.__setattr__(self, "_pairs", tuple(pairs))

    def _kwargs(self) -> dict[str, Any]:
        return {
            "decision_threshold": self.decision_threshold,
            "total_budget": self.total_budget,
            "tolerance": float(self.tolerance),
            "frontier_points": _non_negative("frontier_points", self.frontier_points),
            "max_operations": _non_negative("max_operations", self.max_operations),
            "max_depth": _non_negative("max_depth", self.max_depth),
            "max_memory_bytes": _non_negative("max_memory_bytes", self.max_memory_bytes),
        }


def _stage(stage: Any) -> Any:
    if not hasattr(stage, "export_sensitivity"):
        raise CausalTypeError(
            "stage must be a prepared z-transport stage (prepare_exact/prepare_empirical)"
        )
    return stage


def _deviation(deviation: JointDeviation) -> JointDeviation:
    if not isinstance(deviation, JointDeviation):
        raise CausalTypeError("deviation must be a JointDeviation")
    return deviation


def joint_mechanism_sensitivity(
    stage: Any,
    deviation: JointDeviation,
    *,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """Exact joint assumption range of a prepared stage's retained surrogate formula.

    Nothing is re-identified: the stage's checked functional and retained laws
    are evaluated. The result carries ``inference_claim == "assumption_range"``
    and ``uncertainty["status"] == "withheld"``.
    """
    deviation = _deviation(deviation)
    return _joint(
        _stage(stage),
        list(deviation._pairs),
        **deviation._kwargs(),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def export_joint_mechanism_sensitivity(
    stage: Any,
    deviation: JointDeviation,
    *,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> bytes:
    """Export the stage's last execution with a checked joint analysis (artifact v3).

    Call ``stage.estimate()`` first: the artifact embeds that execution's point
    artifact as its baseline.
    """
    deviation = _deviation(deviation)
    return bytes(
        _export_joint(
            _stage(stage),
            list(deviation._pairs),
            **deviation._kwargs(),
            memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
            cancel=cancel,
        )
    )


def consume_joint_mechanism_sensitivity_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    max_search_operations: int = 100_000,
    max_search_depth: int = 64,
    max_memory_bytes: int = 64 * 1024 * 1024,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """Independently verify a joint sensitivity artifact under the consumer's limits.

    Stored search limits above ``max_search_operations`` / ``max_search_depth`` /
    ``max_memory_bytes`` refuse before any replay (``joint_sensitivity.consumer_limits``).
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_joint_artifact(
        artifact,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_optional_non_negative("max_support_rows", max_support_rows),
        max_laws=_optional_non_negative("max_laws", max_laws),
        max_law_cells=_optional_non_negative("max_law_cells", max_law_cells),
        max_search_operations=_non_negative("max_search_operations", max_search_operations),
        max_search_depth=_non_negative("max_search_depth", max_search_depth),
        max_memory_bytes=_non_negative("max_memory_bytes", max_memory_bytes),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def joint_mechanism_sensitivity_interval(
    stage: Any,
    deviation: JointDeviation,
    *,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> None:
    """Sampling-uncertainty interval of the joint range: closed.

    Always raises with ``reason_code == "cell_not_licensed"``; the assumption
    range is not a confidence interval and its sampling interval is unmeasured.
    """
    deviation = _deviation(deviation)
    _joint_interval(
        _stage(stage),
        list(deviation._pairs),
        decision_threshold=deviation.decision_threshold,
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
