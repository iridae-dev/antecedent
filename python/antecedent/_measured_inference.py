"""Measured scalar inference authorized by retained native source and calibration records."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from ._transport_results import _freeze as freeze
from ._transport_results import _thaw as detached
from .errors import (
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


def _integer(name: str, value: int, lower: int, upper: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise CausalValueError(f"{name} must be in {lower}..={upper}")


def _level(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError("level must be real")
    try:
        valid = math.isfinite(value) and 0 < value < 1
    except OverflowError:
        valid = False
    if not valid:
        raise CausalValueError("level must be finite and strictly between zero and one")


def _production_limits(
    memory_limit_bytes: int | None, cancel: _native.CancellationToken | None
) -> None:
    if memory_limit_bytes is not None:
        _integer("memory_limit_bytes", memory_limit_bytes, 0, 2**64 - 1)
    if cancel is not None and not isinstance(cancel, _native.CancellationToken):
        raise CausalTypeError("cancel must be a CancellationToken")


def _bounded_name(name: str, value: str) -> None:
    try:
        valid = isinstance(value, str) and bool(value.strip()) and len(value.encode()) <= 4096
    except UnicodeError:
        valid = False
    if not valid:
        raise CausalValueError(f"{name} must be a nonempty bounded string")


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasuredInferenceIdentity:
    """Independent caller expectation; constructing it supplies no calibration authority."""

    route: str
    candidate_digest: str
    premises_digest: str
    data_digest: str
    level: float
    scalars: tuple[str, ...]
    seal: str

    def __post_init__(self) -> None:
        _bounded_name("route", self.route)
        for name in ("candidate_digest", "premises_digest", "data_digest", "seal"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise CausalValueError(f"{name} requires 64 lowercase hexadecimal digits")
        _level(self.level)
        if not isinstance(self.scalars, (tuple, list)):
            raise CausalTypeError("scalars must be a sequence of names")
        if not 1 <= len(self.scalars) <= 32:
            raise CausalValueError("scalars must declare 1..=32 outputs")
        for name in self.scalars:
            _bounded_name("scalar", name)
        if len(set(self.scalars)) != len(self.scalars):
            raise CausalValueError("scalar names must be unique")
        object.__setattr__(self, "scalars", tuple(self.scalars))

    def _wire(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "candidate_digest": self.candidate_digest,
            "premises_digest": self.premises_digest,
            "data_digest": self.data_digest,
            "level": self.level,
            "scalars": list(self.scalars),
            "seal": self.seal,
        }

    @classmethod
    def _from_wire(cls, body: Mapping[str, Any]) -> MeasuredInferenceIdentity:
        return cls(
            **{
                name: body[name]
                for name in (
                    "route",
                    "candidate_digest",
                    "premises_digest",
                    "data_digest",
                    "level",
                    "scalars",
                    "seal",
                )
            }
        )


@dataclass(frozen=True, slots=True, init=False)
class MeasuredScalar:
    """One native-authorized scalar; its record applies only to this output and scope."""

    _authority: Any = field(repr=False)
    _body: Mapping[str, Any] = field(repr=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise CausalTypeError("obtain measured scalars from an original MeasuredInference")

    @property
    def name(self) -> str:
        return str(self._body["name"])

    @property
    def point(self) -> float:
        return float(self._body["point"])

    @property
    def interval(self) -> tuple[float, float]:
        return float(self._body["lower"]), float(self._body["upper"])

    @property
    def level(self) -> float:
        return float(self._body["level"])

    @property
    def calibration(self) -> Literal["calibrated"]:
        return "calibrated"

    @property
    def record_id(self) -> str:
        return str(self._body["calibration"]["record_id"])

    @property
    def calibration_sha(self) -> str:
        """Git commit of the governing measurement; this is not a content checksum."""
        return str(self._body["calibration"]["calibration_sha"])

    @property
    def basis(self) -> dict[str, Any]:
        return dict(detached(self._body["calibration"]["basis"]))

    def inspect(self) -> dict[str, Any]:
        return dict(detached(self._body))


@dataclass(frozen=True, slots=True, init=False)
class MeasuredInference:
    """Immutable measured scalar carrier, separate from its original unmeasured source."""

    _native: Any = field(repr=False)
    _body: Mapping[str, Any] = field(repr=False)
    _identity: MeasuredInferenceIdentity = field(repr=False)
    _scalars: tuple[MeasuredScalar, ...] = field(repr=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise CausalTypeError("use a measured native producer or load with retained identity")

    @classmethod
    def _from_native(cls, native: Any) -> MeasuredInference:
        kind = getattr(_native, "NativeMeasuredInference", None)
        if kind is None or not isinstance(native, kind):
            raise CausalTypeError("measured inference requires original native authority")
        body = json.loads(native.payload())
        if body.get("version") != 1 or body.get("calibration") != "calibrated":
            raise CausalSerializationError("unsupported measured inference standing")
        identity = MeasuredInferenceIdentity._from_wire(body["identity"])
        rows = body["scalars"]
        if (
            body["route"] != identity.route
            or tuple(row["name"] for row in rows) != identity.scalars
        ):
            raise CausalSerializationError("measured scalar declarations and identity differ")
        scalars = []
        for row in rows:
            slot = row["calibration"]
            if (
                slot.get("status") != "calibrated"
                or not slot.get("record_id")
                or not slot.get("calibration_sha")
                or not slot.get("basis")
                or row["level"] != identity.level
            ):
                raise CausalSerializationError("measured scalar lacks its governing native record")
            scalar = object.__new__(MeasuredScalar)
            object.__setattr__(scalar, "_authority", native)
            object.__setattr__(scalar, "_body", freeze(row))
            scalars.append(scalar)
        result = object.__new__(cls)
        object.__setattr__(result, "_native", native)
        object.__setattr__(result, "_body", freeze(body))
        object.__setattr__(result, "_identity", identity)
        object.__setattr__(result, "_scalars", tuple(scalars))
        return result

    @property
    def route(self) -> str:
        return self._identity.route

    @property
    def calibration(self) -> Literal["calibrated"]:
        return "calibrated"

    @property
    def expected_identity(self) -> MeasuredInferenceIdentity:
        return self._identity

    @property
    def scalars(self) -> tuple[MeasuredScalar, ...]:
        return self._scalars

    def scalar(self, name: str) -> MeasuredScalar:
        for scalar in self._scalars:
            if scalar.name == name:
                return scalar
        raise CausalValueError(f"no measured scalar {name!r}")

    @property
    def validated_scope(self) -> dict[str, Any]:
        return dict(detached(self._body["validated_scope"]))

    def inspect(self) -> dict[str, Any]:
        return dict(detached(self._body))

    def export(self) -> bytes:
        return bytes(self._native.export())

    def source_report(self) -> dict[str, Any]:
        """Detached original candidate report; its calibration standing is unchanged."""
        return dict(json.loads(self._native.source_report()))

    def source_artifact(self) -> bytes:
        """Original candidate bytes, retaining their own unmeasured calibration label."""
        return bytes(self._native.source_artifact())

    @classmethod
    def load(
        cls,
        artifact: bytes,
        *,
        expected: MeasuredInferenceIdentity,
        max_bytes: int = MAX_ARTIFACT_BYTES,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> MeasuredInference:
        if not isinstance(artifact, bytes):
            raise CausalTypeError("artifact must be bytes")
        if not isinstance(expected, MeasuredInferenceIdentity):
            raise CausalTypeError("expected must be a retained MeasuredInferenceIdentity")
        _integer("max_bytes", max_bytes, 1, MAX_ARTIFACT_BYTES)
        if len(artifact) > max_bytes:
            raise CausalValueError("measured inference artifact exceeds the byte bound")
        _production_limits(memory_limit_bytes, cancel)
        consume = getattr(_native, "consume_measured_inference", None)
        if consume is None:
            raise CausalUnsupportedError(
                "measured inference native authority unavailable",
                reason_code="cell_not_licensed",
            )
        native = consume(
            artifact,
            json.dumps(expected._wire(), allow_nan=False),
            max_bytes=max_bytes,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        return cls._from_native(native)


__all__ = ["MeasuredInference", "MeasuredInferenceIdentity", "MeasuredScalar"]
