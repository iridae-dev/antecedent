"""Checked empirical binary two-step histories; point-only temporal recalculation.

The source-backed artifact retains complete repeated-unit ownership and runs
actual checked proof/mechanism reconstruction in an independent consumer.
Measured dependent-sampling intervals preserve their checked source and require current calibration evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._measured_inference import MeasuredInference
from ._recalc_bounds import _admit, _check_resources
from ._refusal import raise_refusal
from .errors import CausalSerializationError, CausalTypeError, CausalValueError
from .outcomes import outcome_operation
from .recalc import Decision, Law, RecalcPlan, RecalcReceipt, Utility, _refusal_type, _seed
from .recalc_static import _Session
from .transport._candidate_data import detached, freeze


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

    def _history_columns(self) -> tuple[np.ndarray, ...]:
        """The histories as arrays for the bridge: per unit its id and window count, then per
        window its time id, the four binary states and the outcome."""
        try:
            units = self.units
            ids = np.fromiter((unit.unit_id for unit in units), dtype=np.uint64, count=len(units))
            lengths = np.fromiter(
                (len(unit.histories) for unit in units), dtype=np.uint64, count=len(units)
            )
            rows = [h for unit in units for h in unit.histories]
            time_ids = np.fromiter((h.time_id for h in rows), dtype=np.uint64, count=len(rows))
            states = [
                np.fromiter((getattr(h, name) for h in rows), dtype=np.uint32, count=len(rows))
                for name in ("s0", "a1", "l2", "a2")
            ]
            y = np.fromiter((h.y for h in rows), dtype=np.float64, count=len(rows))
        except (OverflowError, TypeError, ValueError) as error:
            raise CausalValueError(
                "recalc.invalid_declaration: histories must be non-negative integers and a "
                "numeric outcome",
                reason_code="invalid_argument",
            ) from error
        return (ids, lengths, time_ids, *states, y)

    def _wire(self) -> dict[str, Any]:
        """The whole request as recorded in a source report, histories included."""
        return {**self._declaration(), "units": [asdict(unit) for unit in self.units]}

    def _declaration(self) -> dict[str, Any]:
        """The declaration sent to the bridge: the histories travel as columns instead."""
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
            "units": [],
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
    raise_refusal(error, _refusal_type)
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

    _standing_kind = "temporal"

    @staticmethod
    def _native_type() -> Any:
        return _native.TemporalSessionHandle

    def plan(
        self,
        request: TemporalRequest,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> RecalcPlan:
        _admit_histories(request, memory_limit_bytes, cancel)
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(
                    json.dumps(request._declaration()),
                    request._history_columns(),
                    seed=_seed(seed),
                    threads=threads,
                    memory_limit_bytes=memory_limit_bytes,
                    cancel=cancel,
                )
            )
        )

    @outcome_operation
    def execute(
        self,
        request: TemporalRequest,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> TemporalResult:
        """Run the invalidated stages under an optional memory limit and cancellation token."""
        _admit_histories(request, memory_limit_bytes, cancel)
        return _result(
            self._handle.execute(
                json.dumps(request._declaration()),
                request._history_columns(),
                seed=_seed(seed),
                threads=threads,
                memory_limit_bytes=memory_limit_bytes,
                cancel=cancel,
            )
        )

    def interval_candidate(
        self,
        *,
        config: CheckedTemporalIntervalConfig,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> CheckedTemporalIntervalCandidate:
        """Original checked source/proof whole-unit interval; unmeasured and feature-only."""
        if not isinstance(config, CheckedTemporalIntervalConfig):
            raise CausalTypeError("config must be a CheckedTemporalIntervalConfig")
        _check_resources(memory_limit_bytes, cancel)
        if not isinstance(self._handle, _native.TemporalSessionHandle):
            raise CausalTypeError("interval candidate requires original checked temporal session")
        native = getattr(self._handle, "interval_candidate", None)
        if native is None:
            raise_refusal(self._handle.dependent_interval(), _refusal_type)
            raise CausalSerializationError("closed interval route returned without a refusal")
        result, error = native(
            json.dumps(config._wire(), allow_nan=False),
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        raise_refusal(error, _refusal_type)
        return CheckedTemporalIntervalCandidate._from_native(result)

    def dependent_interval(
        self,
        *,
        config: CheckedTemporalIntervalConfig | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> MeasuredInference:
        """Measured whole-unit interval from the original causal source and both effect arms.

        Defaults to 500 studentized draws at 95%. The native producer enforces
        the measured binary two-step protocol and current evidence; it refuses
        every adjacent unmeasured method, target law or functional.
        """
        if config is None:
            config = CheckedTemporalIntervalConfig(method="studentized")
        if not isinstance(config, CheckedTemporalIntervalConfig):
            raise CausalTypeError("config must be a CheckedTemporalIntervalConfig")
        _check_resources(memory_limit_bytes, cancel)
        if not isinstance(self._handle, _native.TemporalSessionHandle):
            raise CausalTypeError("interval requires original checked temporal session")
        native, error = self._handle.measured_interval(
            json.dumps(config._wire(), allow_nan=False),
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        raise_refusal(error, _refusal_type)
        return MeasuredInference._from_native(native)


@dataclass(frozen=True, slots=True)
class TemporalReplay:
    session: TemporalSession
    result: TemporalResult


@outcome_operation
def consume_temporal_result(
    artifact: bytes, *, seed: int = 1, threads: int | None = None
) -> TemporalReplay:
    """Fresh actual checked execution from full source-bound histories, not a receipt."""
    artifact = container_bytes(artifact)
    native, result, receipt, error = _native.TemporalSessionHandle.consume(
        artifact, seed=_seed(seed), threads=threads
    )
    raise_refusal(error, _refusal_type)
    if native is None:
        raise CausalSerializationError("temporal replay returned no native state")
    session = TemporalSession.__new__(TemporalSession)
    session._handle = native
    return TemporalReplay(session, _result((result, receipt, None)))


def _interval_integer(name: str, value: int, lower: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise CausalValueError(f"{name} must be in {lower}..={upper}")


# A history window carries its time id and five binary states.
_HISTORY_VALUES = 6


def _admit_histories(
    request: TemporalRequest,
    memory_limit_bytes: int | None,
    cancel: _native.CancellationToken | None,
) -> None:
    _check_resources(memory_limit_bytes, cancel)
    rows = sum(len(unit.histories) for unit in request.units)
    _admit(rows * _HISTORY_VALUES, memory_limit_bytes)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckedTemporalIntervalConfig:
    """Explicit whole-unit method; bootstrap seed differs from the producing seed.

    Studentization requires the original checked balanced binary histories. No method
    borrows coverage from a plain panel or a different source/functional.
    """

    method: Literal["percentile", "basic", "studentized"]
    level: float = 0.95
    replicates: int = 500
    bootstrap_seed: int = 0
    min_units: int = 20
    max_failed_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.method not in ("percentile", "basic", "studentized"):
            raise CausalValueError("method must be percentile, basic or studentized")
        _interval_integer("replicates", self.replicates, 20, 2000)
        _interval_integer("bootstrap_seed", self.bootstrap_seed, 0, 2**64 - 1)
        _interval_integer("min_units", self.min_units, 2, 4096)
        for name, value in (
            ("level", self.level),
            ("max_failed_fraction", self.max_failed_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CausalTypeError(f"{name} must be real")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise CausalValueError(f"{name} must be finite")
        if not 0 < self.level < 1:
            raise CausalValueError("level must lie strictly between zero and one")
        if not 0 <= self.max_failed_fraction < 1:
            raise CausalValueError("max_failed_fraction must be in [0, 1)")

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> CheckedTemporalIntervalConfig:
        return cls(
            method=wire["method"],
            level=wire["level"],
            replicates=wire["replicates"],
            bootstrap_seed=wire["seed"],
            min_units=wire["min_units"],
            max_failed_fraction=wire["max_failed_fraction"],
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "level": self.level,
            "replicates": self.replicates,
            "seed": self.bootstrap_seed,
            "min_units": self.min_units,
            "max_failed_fraction": self.max_failed_fraction,
        }


def _interval_functional(value: Any) -> TemporalResponse | TemporalEffect:
    def sequence(values: Any) -> tuple[int, int]:
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise CausalValueError("functional must declare binary two-step sequences")
        for coordinate in values:
            _interval_integer("sequence coordinate", coordinate, 0, 1)
        return (values[0], values[1])

    if isinstance(value, TemporalResponse):
        return TemporalResponse(sequence(value.sequence))
    if isinstance(value, TemporalEffect):
        return TemporalEffect(sequence(value.active), sequence(value.control))
    raise CausalTypeError("functional must be TemporalResponse or TemporalEffect")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckedTemporalIntervalIdentity:
    """Caller-retained original source, target, functional and complete candidate seal."""

    seal: str
    source_data_digest: str
    source_premises_digest: str
    snapshot_id: str
    initial_state_id: str
    functional: TemporalResponse | TemporalEffect
    producing_seed: int
    config: CheckedTemporalIntervalConfig
    panel_digest: str

    def __post_init__(self) -> None:
        for name in ("seal", "source_data_digest", "source_premises_digest"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise CausalValueError(f"{name} requires 64 lowercase hexadecimal digits")
        for name in ("snapshot_id", "initial_state_id"):
            value = getattr(self, name)
            try:
                valid = isinstance(value, str) and bool(value) and len(value.encode()) <= 4096
            except UnicodeError:
                valid = False
            if not valid:
                raise CausalValueError(f"{name} must be a nonempty bounded identity")
        if not isinstance(self.config, CheckedTemporalIntervalConfig):
            raise CausalTypeError("config must be a CheckedTemporalIntervalConfig")
        if (
            not isinstance(self.panel_digest, str)
            or len(self.panel_digest) != 16
            or any(c not in "0123456789abcdef" for c in self.panel_digest)
        ):
            raise CausalValueError("panel_digest requires 16 lowercase hexadecimal digits")
        _interval_integer("producing_seed", self.producing_seed, 0, 2**64 - 1)
        coerce_fields(self, functional=_interval_functional(self.functional))

    def _wire(self) -> dict[str, Any]:
        wire = asdict(self)
        wire["functional"]["kind"] = (
            "response" if isinstance(self.functional, TemporalResponse) else "effect"
        )
        wire["config"] = self.config._wire()
        return wire

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> CheckedTemporalIntervalIdentity:
        functional = wire["functional"]
        declaration: TemporalResponse | TemporalEffect
        if functional["kind"] == "response":
            declaration = TemporalResponse(tuple(functional["sequence"]))
        elif functional["kind"] == "effect":
            declaration = TemporalEffect(tuple(functional["active"]), tuple(functional["control"]))
        else:
            raise CausalSerializationError("unknown checked temporal functional")
        return cls(
            **{
                key: wire[key]
                for key in (
                    "seal",
                    "source_data_digest",
                    "source_premises_digest",
                    "snapshot_id",
                    "initial_state_id",
                    "producing_seed",
                    "panel_digest",
                )
            },
            functional=declaration,
            config=CheckedTemporalIntervalConfig._from_wire(wire["config"]),
        )


@dataclass(frozen=True, slots=True, init=False)
class CheckedTemporalIntervalCandidate:
    """Immutable original checked source/proof candidate; calibration is unmeasured.

    Constructed by TemporalSession.interval_candidate or independent load. Both arms
    of an effect retain their original declarations and share every whole-unit draw.
    """

    _native: Any
    _body: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise CausalTypeError(
            "use TemporalSession.interval_candidate or load with retained identity"
        )

    @classmethod
    def _from_native(cls, native: Any) -> CheckedTemporalIntervalCandidate:
        kind = getattr(_native, "NativeCheckedTemporalIntervalCandidate", None)
        if kind is None or not isinstance(native, kind):
            raise CausalTypeError("candidate requires original checked temporal native authority")
        body = json.loads(native.payload())
        if (
            body.get("version") != 1
            or body.get("required_features") != ["checked_temporal_interval_candidate_v1"]
            or body.get("calibration") != "unmeasured"
            or body["result"].get("calibration") != "unmeasured"
            or body.get("claim") != "dependence_preserving_calibration_unmeasured"
            or body["result"].get("claim") != "dependence_preserving_calibration_unmeasured"
        ):
            raise CausalSerializationError("unsupported checked temporal interval standing")
        identity = CheckedTemporalIntervalIdentity._from_wire(body["identity"])
        if (
            identity.config._wire() != body["config"]
            or identity.panel_digest != body["result"]["panel_digest"]
        ):
            raise CausalSerializationError("checked temporal interval identity and receipt differ")
        result = object.__new__(cls)
        coerce_fields(result, _native=native)
        coerce_fields(result, _body=freeze(body))
        return result

    @property
    def calibration(self) -> Literal["unmeasured"]:
        return "unmeasured"

    @property
    def expected_identity(self) -> CheckedTemporalIntervalIdentity:
        return CheckedTemporalIntervalIdentity._from_wire(self._body["identity"])

    @property
    def functional(self) -> TemporalResponse | TemporalEffect:
        return self.expected_identity.functional

    @property
    def config(self) -> CheckedTemporalIntervalConfig:
        body = self._body["config"]
        return CheckedTemporalIntervalConfig(
            method=body["method"],
            level=body["level"],
            replicates=body["replicates"],
            bootstrap_seed=body["seed"],
            min_units=body["min_units"],
            max_failed_fraction=body["max_failed_fraction"],
        )

    @property
    def point(self) -> float:
        return float(self._body["result"]["point"])

    @property
    def interval(self) -> tuple[float, float]:
        result = self._body["result"]
        return float(result["lower"]), float(result["upper"])

    @property
    def standard_error(self) -> float | None:
        value = self._body["result"].get("studentization")
        return None if value is None else float(value["standard_error"])

    @property
    def unit_scores(self) -> tuple[float, ...] | None:
        value = self._body["result"].get("studentization")
        return None if value is None else tuple(float(x) for x in value["unit_scores"])

    def inspect(self) -> dict[str, Any]:
        """Detached original source identities and complete interval draw/SE/pivot receipt."""
        return cast(dict[str, Any], detached(self._body))

    @artifact_export
    def export(self) -> ExportedArtifact:
        return artifact_from_receipt(self, bytes(self._native.export()))

    @classmethod
    def load(
        cls,
        artifact: bytes,
        *,
        expected: CheckedTemporalIntervalIdentity,
        max_units: int = 4096,
        max_histories: int = 100_000,
        max_replicates: int = 2000,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> CheckedTemporalIntervalCandidate:
        """Fresh original proof/mechanism reconstruction and paired whole-unit replay."""
        artifact = container_bytes(artifact)
        if not isinstance(artifact, bytes):
            raise CausalTypeError("artifact must be bytes")
        if len(artifact) > 24 * 1024 * 1024:
            raise CausalValueError("checked temporal interval artifact exceeds 24 MiB")
        if not isinstance(expected, CheckedTemporalIntervalIdentity):
            raise CausalTypeError("expected must be a retained CheckedTemporalIntervalIdentity")
        for name, value, bound in (
            ("max_units", max_units, 4096),
            ("max_histories", max_histories, 100_000),
            ("max_replicates", max_replicates, 2000),
        ):
            _interval_integer(name, value, 1, bound)
        _check_resources(memory_limit_bytes, cancel)
        consume = getattr(_native, "consume_checked_temporal_interval_candidate", None)
        if consume is None:
            from .recalc import RecalcRefusal

            raise RecalcRefusal(
                {
                    "code": "cell_not_licensed",
                    "detail": "temporal_interval.route_frozen",
                    "stage": "inference",
                    "message": "checked temporal interval calibration is unmeasured",
                }
            )
        native, error = consume(
            artifact,
            json.dumps(expected._wire(), allow_nan=False),
            max_units=max_units,
            max_histories=max_histories,
            max_replicates=max_replicates,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        raise_refusal(error, _refusal_type)
        return cls._from_native(native)


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
    "CheckedTemporalIntervalConfig",
    "CheckedTemporalIntervalIdentity",
    "CheckedTemporalIntervalCandidate",
]
