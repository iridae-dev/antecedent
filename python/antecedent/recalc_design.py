"""Shared selective execution of checked IV, sharp RD and linear front-door designs.

Utility changes reuse the checked estimate. Changed supplied inputs invalidate
its identification or numerical fits; the existing public estimators do the
actual work. Weak-IV decisions refuse while retaining the attempted point and
actual Anderson–Rubin diagnostics. Reuse grants no calibrated interval license.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import ArrayLike

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._recalc_bounds import _columns
from ._refusal import raise_refusal
from .errors import CausalSerializationError, CausalTypeError
from .execution import Execution
from .outcomes import outcome_operation
from .recalc import (
    Decision,
    Law,
    RecalcPlan,
    RecalcReceipt,
    RecalcRefusal,
    RecalcResult,
    Utility,
    _execution,
    _native_controls,
    _refusal_type,
)
from .recalc_static import _Session


@dataclass(frozen=True, slots=True)
class IvModel:
    """One checked binary instrument (0/1); two-stage least squares."""

    instrument: str


@dataclass(frozen=True, slots=True)
class RdModel:
    """Checked sharp cutoff with a finite symmetric local-linear window."""

    running_variable: str
    cutoff: float
    bandwidth: float


@dataclass(frozen=True, slots=True)
class FrontdoorModel:
    """Checked linear path product through one observed mediator."""

    mediator: str


@dataclass(frozen=True, slots=True, eq=False)
class DesignRequest:
    """Full supplied frame, causal graph, design roles and utility."""

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatment: str
    outcome: str
    model: IvModel | RdModel | FrontdoorModel
    utility: Utility


class DesignWeakInstrument(RecalcRefusal):
    """Unavailable IV decision with actual attempted point, fits and diagnostics.

    The diagnostics retain the native AR set/withheld reason. They do not become
    a decision bound or a new calibrated interval through this wrapper.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        super().__init__(refusal)
        fields = refusal["fields"]
        self.point: float = fields["point"]
        self.model_fits: int = fields["model_fits"]
        self.diagnostics: Mapping[str, Any] = fields["diagnostics"]


def _design_refusal_type(wire: Mapping[str, Any]) -> type[RecalcRefusal]:
    if wire["detail"] == "recalc.iv_decision_unavailable":
        return DesignWeakInstrument
    return _refusal_type(wire)


def _raise_design(error: str | None) -> None:
    raise_refusal(error, _design_refusal_type)


def _spec(request: DesignRequest) -> str:
    model = request.model
    if isinstance(model, IvModel):
        config: dict[str, Any] = {"kind": "iv", "instrument": model.instrument}
    elif isinstance(model, RdModel):
        config = {
            "kind": "rd",
            "running_variable": model.running_variable,
            "cutoff": model.cutoff,
            "bandwidth": model.bandwidth,
        }
    elif isinstance(model, FrontdoorModel):
        config = {"kind": "frontdoor", "mediator": model.mediator}
    else:
        raise CausalTypeError(
            "model must be IvModel, RdModel or FrontdoorModel", reason_code="invalid_argument"
        )
    return json.dumps(
        {
            "edges": list(request.edges),
            "treatment": request.treatment,
            "outcome": request.outcome,
            "model": config,
            "benefit_per_unit": request.utility.benefit_per_unit,
            "cost": request.utility.cost,
        }
    )


class DesignSession(_Session):
    """Native checked design state; readable receipts and flags do not supply it."""

    __slots__ = ()

    _standing_kind = "design"

    @staticmethod
    def _native_type() -> Any:
        return _native.DesignSessionHandle

    def plan(
        self,
        request: DesignRequest,
        *,
        execution: Execution | None = None,
    ) -> RecalcPlan:
        run = _execution(execution)
        ctl = _native_controls(run)
        names, columns = _columns(
            request.data, memory_limit_bytes=ctl["memory_limit_bytes"], cancel=ctl["cancel"]
        )
        wire = run._call(self._handle.plan, names, columns, _spec(request), **ctl)
        return RecalcPlan.from_wire(json.loads(wire))

    @outcome_operation
    def execute(
        self,
        request: DesignRequest,
        *,
        execution: Execution | None = None,
    ) -> RecalcResult:
        """Run the invalidated stages under the execution's memory limit and cancellation."""
        run = _execution(execution)
        ctl = _native_controls(run)
        names, columns = _columns(
            request.data, memory_limit_bytes=ctl["memory_limit_bytes"], cancel=ctl["cancel"]
        )
        result, artifact, error = run._call(
            self._handle.execute, names, columns, _spec(request), **ctl
        )
        _raise_design(error)
        if result is None or artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("design recalculation returned no result")
        wire = json.loads(result)
        return RecalcResult(
            RecalcPlan.from_wire(wire["plan"]),
            RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
            Law(**wire["law"]),
            Decision(**wire["decision"]),
        )

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Original result artifact; neither new inference nor resumed state.

        IV and front-door consumers verify their existing scientific program.
        RD remains readable with its declared checked-operation dependency;
        fresh RD execution requires raw inputs and a new checked preparation.
        """
        artifact, error = _execution(execution)._call(self._handle.export_result)
        _raise_design(error)
        if artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("design recalculation exported no artifact")
        return artifact_from_receipt(self, bytes(artifact))


@dataclass(frozen=True, slots=True)
class DesignReplay:
    """Independent raw-input replay, its actual live session and work receipt."""

    session: DesignSession
    result: RecalcResult
    artifact_digest: str


@outcome_operation
def consume_design_result(
    artifact: bytes,
    request: DesignRequest | None = None,
    *,
    execution: Execution | None = None,
) -> DesignReplay:
    """Verify an original artifact and rerun its checked design from actual data.

    The supplied inputs and producing seed must reproduce the full scientific
    body and contract. This discharges RD's declared checked-operation dependency
    through real execution, while preserving existing reader behavior. Missing
    or mismatched data refuse rather than becoming portable fitted state.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes", reason_code="invalid_argument")
    run = _execution(execution)
    ctl = _native_controls(run)
    names, columns = (
        (None, None)
        if request is None
        else _columns(
            request.data, memory_limit_bytes=ctl["memory_limit_bytes"], cancel=ctl["cancel"]
        )
    )
    specification = None if request is None else _spec(request)
    native, result, receipt_bytes, digest, error = run._call(
        _native.DesignSessionHandle.consume, artifact, names, columns, specification, **ctl
    )
    _raise_design(error)
    if native is None or result is None or receipt_bytes is None or digest is None:
        raise CausalSerializationError("design replay returned no checked result")
    session = DesignSession.__new__(DesignSession)
    session._handle = native
    wire = json.loads(result)
    replayed = RecalcResult(
        RecalcPlan.from_wire(wire["plan"]),
        RecalcReceipt._from_wire(wire["receipt"], wire["plan"], receipt_bytes, loaded=False),
        Law(**wire["law"]),
        Decision(**wire["decision"]),
    )
    return DesignReplay(session, replayed, digest)


__all__ = [
    "DesignReplay",
    "consume_design_result",
    "DesignRequest",
    "DesignSession",
    "DesignWeakInstrument",
    "FrontdoorModel",
    "IvModel",
    "RdModel",
]
