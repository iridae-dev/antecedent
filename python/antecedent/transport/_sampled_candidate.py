"""Whole-row recovery acceptance lifecycle; candidate inference stays unmeasured."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import BuiltinFunctionType
from typing import Any, Literal, cast

from .. import _native
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ._candidate_data import detached, freeze


@dataclass(frozen=True, slots=True)
class SampledRecoveryIdentity:
    """Consumer-retained scientific premises and original row identity."""

    premises_digest: str
    data_digest: str

    def __post_init__(self) -> None:
        for value in (self.premises_digest, self.data_digest):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise CausalValueError("sampled recovery identity requires 64 lowercase hex digits")


@dataclass(frozen=True, slots=True, init=False)
class SampledRecoveryCandidate:
    """Checked whole-row bootstrap candidate, with no measured coverage license.

    Created through the private candidate hook in a calibration-internal build.
    The standard public producer returns native-authorized measured inference.
    Export/load reruns original recovery and every replicate under retained identity.
    """

    _body: Mapping[str, Any]
    _artifact: bytes

    def __init__(self) -> None:
        raise CausalTypeError("use the internal candidate hook or load with retained identity")

    @classmethod
    def _from_native(cls, payload: str, artifact: bytes) -> SampledRecoveryCandidate:
        consume = getattr(_native, "consume_sampled_recovery_candidate", None)
        if (
            not isinstance(consume, BuiltinFunctionType)
            or consume.__module__ != _native.__name__
            or consume.__name__ != "consume_sampled_recovery_candidate"
        ):
            raise CausalUnsupportedError(
                "sampled_recovery.route_frozen: private candidate factory requires "
                "the original calibration-internal native hook",
                reason_code="cell_not_licensed",
            )
        decoded = json.loads(payload)
        if decoded.get("calibration") != "unmeasured":
            raise CausalValueError("sampled recovery candidate must remain unmeasured")
        result = object.__new__(cls)
        object.__setattr__(result, "_body", freeze(decoded))
        object.__setattr__(result, "_artifact", bytes(artifact))
        return result

    @property
    def calibration(self) -> str:
        return "unmeasured"

    @property
    def interval_method(self) -> Literal["bootstrap_bca"]:
        """Actual supported method, bound to the independently replayed BCa receipt."""
        value = self._body["receipt"]["config"]["interval_method"]
        return cast(Literal["bootstrap_bca"], value)

    @property
    def effect(self) -> float:
        return float(self._body["result"]["effect"])

    @property
    def effect_standard_error(self) -> float:
        return float(self._body["result"]["effect_standard_error"])

    @property
    def interval(self) -> tuple[float, float]:
        interval = self._body["result"]["interval"]
        return float(interval["lower"]), float(interval["upper"])

    @property
    def level(self) -> float:
        return float(self._body["result"]["interval"]["level"])

    @property
    def expected_identity(self) -> SampledRecoveryIdentity:
        payload = self._body
        return SampledRecoveryIdentity(payload["premises_digest"], payload["data_digest"])

    def inspect(self) -> dict[str, Any]:
        """Detached original law, covariance, interval, diagnostics and work receipt."""
        return dict(detached(self._body))

    def export(self) -> bytes:
        return self._artifact

    @classmethod
    def load(
        cls,
        artifact: bytes,
        *,
        expected: SampledRecoveryIdentity,
        max_rows: int = 100_000,
        max_replicates: int = 2000,
        memory_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> SampledRecoveryCandidate:
        """Independently re-identify and rerun whole-row bootstrap under expectations."""
        if not isinstance(artifact, bytes):
            raise CausalTypeError("artifact must be bytes")
        if not isinstance(expected, SampledRecoveryIdentity):
            raise CausalTypeError("expected must be a retained SampledRecoveryIdentity")
        for name, value, ceiling in (
            ("max_rows", max_rows, 100_000),
            ("max_replicates", max_replicates, 2000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= ceiling:
                raise CausalValueError(f"{name} must be an integer in 1..={ceiling}")
        if memory_bytes is not None and (
            isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, int)
            or not 0 <= memory_bytes <= 2**64 - 1
        ):
            raise CausalValueError("memory_bytes must fit an unsigned 64-bit integer")
        if cancel is not None and not isinstance(cancel, _native.CancellationToken):
            raise CausalTypeError("cancel must be a CancellationToken")
        consume = getattr(_native, "consume_sampled_recovery_candidate", None)
        if consume is None:
            raise CausalUnsupportedError(
                "sampled_recovery.route_frozen: private candidate replay requires calibration-internal",
                reason_code="cell_not_licensed",
            )
        payload = consume(
            artifact,
            expected.premises_digest,
            expected.data_digest,
            max_rows=max_rows,
            max_replicates=max_replicates,
            memory_bytes=memory_bytes,
            cancel=cancel,
        )
        return cls._from_native(payload, artifact)
