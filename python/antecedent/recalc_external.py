"""Executing bounded foreign mean callbacks under exact declared dependencies.

Policies are provider declarations. Outputs remain externally attested; invocation
receipts confer neither native causal identification nor uncertainty calibration.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._recalc_bounds import _columns
from .errors import CausalSerializationError, CausalTypeError, CausalValueError
from .external import BoundExternalClaim, ExternalSpec, ProviderObject, Response
from .program_claims import ProgramBinding
from .recalc import RecalcPlan, RecalcReceipt, RecalcRefusal, _seed

CallbackPolicy = Literal["deterministic", "seeded", "stateful", "side_effecting", "unknown"]


@dataclass(frozen=True, slots=True)
class CallbackDescriptor:
    provider: ProviderObject
    environment_id: str
    policy: CallbackPolicy = "unknown"
    idempotency_supported: bool = False

    def _wire(self) -> dict[str, Any]:
        return {
            "provider": self.provider._wire(),
            "environment_id": self.environment_id,
            "policy": self.policy,
            "idempotency_supported": self.idempotency_supported,
        }


@dataclass(frozen=True, slots=True)
class CallbackInputs:
    """Copied native request columns and bounded metadata supplied to one invocation."""

    data: Mapping[str, NDArray[np.float64]]
    model_parameters: Mapping[str, float]
    doses: tuple[float, ...]
    seed: int
    branch: int
    idempotency_key: str | None
    cancellation: _native.CancellationToken


class CallbackProvider:
    """Actual callable and its declared implementation/environment/replay policy."""

    __slots__ = ("descriptor", "_callback", "_handle")

    def __init__(
        self, callback: Callable[[CallbackInputs], Response], descriptor: CallbackDescriptor
    ):
        if not callable(callback):
            raise CausalTypeError("callback must be callable")
        self.descriptor, self._callback = descriptor, callback
        self._handle = _native.ExternalCallbackProviderHandle(
            json.dumps(descriptor._wire()), self._invoke
        )

    def _invoke(self, payload: dict[str, Any]) -> str:
        response = self._callback(
            CallbackInputs(
                MappingProxyType(payload["data"]),
                MappingProxyType(payload["model_parameters"]),
                tuple(payload["doses"]),
                payload["seed"],
                payload["branch"],
                payload["idempotency_key"],
                payload["cancellation"],
            )
        )
        if not isinstance(response, Response):
            raise CausalTypeError("callback must return external.Response")
        if (
            len(response.values) > 1024
            or len(response.evidence) > 1024
            or len(response.assumptions) > 1024
        ):
            raise CausalValueError(
                "external_recalc.limits_exceeded: response exceeds bounds",
                reason_code="invalid_argument",
            )
        quantities = (
            payload["quantities"]
            if response.quantities is None
            else [q._wire() for q in response.quantities]
        )
        if len(quantities) > 1024 or len(response.probes) > 1024:
            raise CausalValueError(
                "external_recalc.limits_exceeded: response metadata exceeds bounds",
                reason_code="invalid_argument",
            )
        wire = {
            "provider": response.provider._wire(),
            "graph_id": response.graph_id or payload["graph_id"],
            "quantities": quantities,
            "values": response.values,
            "evidence_ids": response.evidence,
            "assumption_ids": response.assumptions,
            "attestor": response.attested_by,
            "probes": [p._wire() for p in response.probes] if response.probes else None,
            "uncertainty_method": response.uncertainty_method,
            "point_support": response.support,
        }
        result = json.dumps(wire)
        if len(result.encode()) > 1024 * 1024:
            raise CausalValueError(
                "external_recalc.limits_exceeded: response exceeds byte bound",
                reason_code="invalid_argument",
            )
        return result


@dataclass(frozen=True, slots=True, eq=False)
class ExternalCallbackRequest:
    program: ProgramBinding
    spec: ExternalSpec
    descriptor: CallbackDescriptor
    data: Mapping[str, ArrayLike] = field(default_factory=dict)
    model_parameters: Mapping[str, float] = field(default_factory=dict)
    seed: int = 1
    branch: int = 0
    idempotency_key: str | None = None

    def _wire(self) -> str:
        if len(self.model_parameters) > 256:
            raise CausalValueError(
                "external_recalc.limits_exceeded: parameter count exceeds bound",
                reason_code="invalid_argument",
            )
        return json.dumps(
            {
                "program": self.program._wire(),
                "claim": self.spec._program_claim(),
                "contract": self.spec._contract_wire(),
                "descriptor": self.descriptor._wire(),
                "model_parameters": sorted(self.model_parameters.items()),
                "seed": _seed(self.seed),
                "branch": self.branch,
                "idempotency_key": self.idempotency_key,
            }
        )


class ExternalCallbackRefusal(RecalcRefusal):
    """A failed attempt reports actual invocations without issuing output authority."""

    def __init__(self, wire: Mapping[str, Any]):
        super().__init__(wire)
        self.attempt: Mapping[str, Any] | None = wire.get("attempt")


def _raise(error: str | None) -> None:
    if error is not None:
        raise ExternalCallbackRefusal(json.loads(error))


@dataclass(frozen=True, slots=True)
class ExternalCallbackResult:
    plan: RecalcPlan
    receipt: RecalcReceipt
    claim: BoundExternalClaim


class ExternalCallbackSession:
    __slots__ = ("_handle",)

    def __init__(self) -> None:
        self._handle = _native.ExternalCallbackSessionHandle()

    @property
    def is_live(self) -> bool:
        return bool(self._handle.is_live())

    def plan(
        self, request: ExternalCallbackRequest, *, provider: CallbackProvider | None = None
    ) -> RecalcPlan:
        names, columns = _columns(request.data)
        result, error = self._handle.plan(names, columns, request._wire(), provider is not None)
        _raise(error)
        assert result is not None
        return RecalcPlan.from_wire(json.loads(result))

    def execute(
        self,
        request: ExternalCallbackRequest,
        *,
        provider: CallbackProvider | None = None,
        threads: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> ExternalCallbackResult:
        names, columns = _columns(request.data)
        wire, receipt, claim, error = self._handle.execute(
            names,
            columns,
            request._wire(),
            None if provider is None else provider._handle,
            threads=threads,
            cancel=cancel,
        )
        _raise(error)
        if wire is None or receipt is None or claim is None:
            raise CausalSerializationError("external callback returned no issued output")
        body = json.loads(wire)
        return ExternalCallbackResult(
            RecalcPlan.from_wire(body["plan"]),
            RecalcReceipt._from_wire(body["receipt"], body["plan"], receipt, loaded=False),
            BoundExternalClaim(claim, request.spec.statement),
        )

    def export_output(self) -> bytes:
        result, error = self._handle.export_output()
        _raise(error)
        if result is None:
            raise CausalSerializationError("external callback has no portable output")
        return bytes(result)

    @classmethod
    def resume(cls, artifact: bytes, request: ExternalCallbackRequest) -> ExternalCallbackSession:
        names, columns = _columns(request.data)
        handle, error = _native.ExternalCallbackSessionHandle.resume(
            artifact, names, columns, request._wire()
        )
        _raise(error)
        if handle is None:
            raise CausalSerializationError("external callback replay returned no session")
        session = cls.__new__(cls)
        session._handle = handle
        return session


__all__ = [
    "CallbackDescriptor",
    "CallbackInputs",
    "CallbackProvider",
    "ExternalCallbackRequest",
    "ExternalCallbackResult",
    "ExternalCallbackSession",
    "ExternalCallbackRefusal",
]
