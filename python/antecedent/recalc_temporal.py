"""Checked empirical binary two-step histories; point-only temporal recalculation.

The source-backed artifact retains complete repeated-unit ownership and runs
actual checked proof/mechanism reconstruction in an independent consumer.
Dependent-sampling interval publication remains frozen until calibration.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from . import _native
from .errors import CausalSerializationError, CausalValueError
from .recalc import Decision, Law, RecalcPlan, RecalcReceipt, Utility, _raise, _seed
from .recalc_static import _Session


@dataclass(frozen=True, slots=True)
class TemporalHistory:
    time_id: int
    s0: int
    a1: int
    l2: int
    a2: int
    y: float


@dataclass(frozen=True, slots=True)
class TemporalUnit:
    unit_id: int
    histories: Sequence[TemporalHistory]


@dataclass(frozen=True, slots=True)
class TemporalResponse:
    sequence: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TemporalEffect:
    active: tuple[int, int]
    control: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TemporalRequest:
    """Actual binary histories and a declared target initial-state law.

    Variable ids are fixed: s0=0, a1=1, l2=2, a2=3, y=4. The current checked
    proof supports selection at root s0 with a directed graph. Raw histories
    preserve unit ownership and ascending within-unit time order.
    """

    edges: Sequence[tuple[int, int]]
    units: Sequence[TemporalUnit]
    period: tuple[int, int]
    snapshot_id: str
    initial_state: tuple[float, float]
    initial_state_id: str
    functional: TemporalResponse | TemporalEffect
    utility: Utility
    bidirected: Sequence[tuple[int, int]] = ()
    selection_targets: Sequence[int] = (0,)
    horizon: int = 2
    lag_alignment: tuple[int, int, int, int, int] = (0, 1, 2, 2, 2)

    def _wire(self) -> dict[str, Any]:
        if len(self.units) > 4096 or sum(len(unit.histories) for unit in self.units) > 100_000:
            raise CausalValueError(
                "recalc.limits_exceeded: temporal unit/history bounds",
                reason_code="invalid_argument",
            )
        functional = asdict(self.functional)
        functional["kind"] = (
            "response" if isinstance(self.functional, TemporalResponse) else "effect"
        )
        return {
            "edges": self.edges,
            "bidirected": self.bidirected,
            "selection_targets": self.selection_targets,
            "horizon": self.horizon,
            "lag_alignment": self.lag_alignment,
            "period": self.period,
            "units": [asdict(unit) for unit in self.units],
            "snapshot_id": self.snapshot_id,
            "initial_state": self.initial_state,
            "initial_state_id": self.initial_state_id,
            "functional": functional,
            "benefit_per_unit": self.utility.benefit_per_unit,
            "cost": self.utility.cost,
        }


@dataclass(frozen=True, slots=True)
class TemporalResult:
    plan: RecalcPlan
    receipt: RecalcReceipt
    functional: TemporalResponse | TemporalEffect
    means: tuple[float, ...]
    law: Law
    decision: Decision


def _result(payload: tuple[str | None, bytes | None, str | None]) -> TemporalResult:
    result, artifact, error = payload
    _raise(error)
    if result is None or artifact is None:
        raise CausalSerializationError("temporal execution returned no result")
    wire = json.loads(result)
    functional = wire["functional"]
    declaration = (
        TemporalResponse(tuple(functional["sequence"]))
        if functional["kind"] == "response"
        else TemporalEffect(tuple(functional["active"]), tuple(functional["control"]))
    )
    return TemporalResult(
        RecalcPlan.from_wire(wire["plan"]),
        RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
        declaration,
        tuple(wire["means"]),
        Law(**wire["law"]),
        Decision(**wire["decision"]),
    )


class TemporalSession(_Session):
    __slots__ = ()

    @staticmethod
    def _native_type() -> Any:
        return _native.TemporalSessionHandle

    def plan(
        self, request: TemporalRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(json.dumps(request._wire()), seed=_seed(seed), threads=threads)
            )
        )

    def execute(
        self, request: TemporalRequest, *, seed: int = 1, threads: int | None = None
    ) -> TemporalResult:
        return _result(
            self._handle.execute(json.dumps(request._wire()), seed=_seed(seed), threads=threads)
        )

    def dependent_interval(self) -> None:
        """Refuse the frozen public dependent-interval route with its exact reason."""
        _raise(self._handle.dependent_interval())


@dataclass(frozen=True, slots=True)
class TemporalReplay:
    session: TemporalSession
    result: TemporalResult


def consume_temporal_result(
    artifact: bytes, *, seed: int = 1, threads: int | None = None
) -> TemporalReplay:
    """Fresh actual checked execution from full source-bound histories, not a receipt."""
    native, result, receipt, error = _native.TemporalSessionHandle.consume(
        artifact, seed=_seed(seed), threads=threads
    )
    _raise(error)
    if native is None:
        raise CausalSerializationError("temporal replay returned no native state")
    session = TemporalSession.__new__(TemporalSession)
    session._handle = native
    return TemporalReplay(session, _result((result, receipt, None)))


__all__ = [
    "TemporalHistory",
    "TemporalUnit",
    "TemporalResponse",
    "TemporalEffect",
    "TemporalRequest",
    "TemporalResult",
    "TemporalSession",
    "TemporalReplay",
    "consume_temporal_result",
]
