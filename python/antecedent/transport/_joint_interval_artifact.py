"""Replayable endpoint-bootstrap interval; separate from range artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .. import _native
from .._artifact_protocol import ExportedArtifact, artifact_consumer, container_bytes
from ..execution import Execution
from ._consumption import TransportConsumption
from ._controls import call_native, resolve
from ._joint_sensitivity import JointDeviation, _deviation, _stage

__all__ = [
    "JointSensitivityIntervalArtifact",
    "export_joint_sensitivity_interval",
    "consume_joint_sensitivity_interval",
]


@dataclass(frozen=True, slots=True)
class JointSensitivityIntervalArtifact(ExportedArtifact[Any]):
    """Portable bytes. Replay derives current calibration independently of stored history."""

    def replay(
        self,
        *,
        max_replicates: int = 2000,
        max_bytes: int = 67_108_864,
        max_search_operations: int = 100_000,
        max_search_depth: int = 64,
        max_memory_bytes: int = 67_108_864,
        execution: Execution | None = None,
    ) -> TransportConsumption:
        return consume_joint_sensitivity_interval(
            self,
            max_replicates=max_replicates,
            max_bytes=max_bytes,
            max_search_operations=max_search_operations,
            max_search_depth=max_search_depth,
            max_memory_bytes=max_memory_bytes,
            execution=execution,
        )


def export_joint_sensitivity_interval(
    stage: Any,
    deviation: JointDeviation,
    *,
    seed: int,
    replicates: int = 199,
    level: float = 0.95,
    execution: Execution | None = None,
) -> JointSensitivityIntervalArtifact:
    """Execute the existing interval procedure on the last executed baseline and seal it.

    The complete deviation box/settings, seed, counts, endpoints and producer calibration
    annotation are retained. No range artifact changes meaning and no new coverage is earned.
    """
    deviation = _deviation(deviation)
    policy = replace(resolve(execution), seed=seed)
    data = call_native(
        policy,
        _native.export_z_transport_joint_interval,
        _stage(stage),
        list(deviation._pairs),
        seed="seed",
        replicates=replicates,
        level=level,
        **deviation._kwargs(policy),
        cancel="cancel",
        memory="memory_bytes",
    )
    return JointSensitivityIntervalArtifact(bytes(data), "joint_sensitivity_interval_v1")


@artifact_consumer(
    proof="JointSensitivityIntervalArtifact::consume_with_limits replays the range and seeded endpoint bootstrap and derives current calibration",
    replayable=True,
    resumable=False,
    identity=(),
)
def consume_joint_sensitivity_interval(
    artifact: bytes | ExportedArtifact[Any],
    *,
    max_replicates: int = 2000,
    max_bytes: int = 67_108_864,
    max_search_operations: int = 100_000,
    max_search_depth: int = 64,
    max_memory_bytes: int = 67_108_864,
    execution: Execution | None = None,
) -> TransportConsumption:
    """Independently replay the range and interval under hard consumer maxima.

    ``recorded_calibration`` is historical producer metadata. ``calibration`` is
    resolved now from the replayed procedure and the current coverage classifier.
    """
    return TransportConsumption.from_checked_mapping(
        "joint_sensitivity_interval",
        call_native(
            execution,
            _native.consume_z_transport_joint_interval,
            container_bytes(artifact),
            max_replicates=max_replicates,
            max_bytes=max_bytes,
            max_search_operations=max_search_operations,
            max_search_depth=max_search_depth,
            max_memory_bytes=max_memory_bytes,
            cancel="cancel",
            memory="memory_bytes",
        ),
    )
