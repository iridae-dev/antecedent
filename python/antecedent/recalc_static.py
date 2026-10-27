"""Selective finite ADMG response and multi-source exact-law transport.

The native shared planner checks every graph, query, data, regime and catalog
identity. Retained numerical work does not license a new population or proof.
Results expose point means and contrasts; no uncertainty or calibration claim is
created. Existing independent consumers verify the original engine artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from numpy.typing import ArrayLike

from . import _native
from ._artifact_protocol import ExportedArtifact, artifact_export, artifact_from_receipt
from ._recalc_bounds import _columns
from ._refusal import raise_refusal
from .errors import CausalSerializationError, CausalTypeError
from .execution import Execution
from .graph import Admg
from .outcomes import outcome_operation
from .recalc import (
    Capabilities,
    Decision,
    RecalcPlan,
    RecalcReceipt,
    ResumeContext,
    SessionKind,
    SessionStanding,
    Stage,
    Utility,
    _declared,
    _declared_json,
    _execution,
    _native_controls,
    _refusal_type,
    session_standing,
)
from .transport._impl import EvidenceCatalog, ExactDiscreteLaw
from .transport._multi_source import MultiSourceZTransportQuery


@dataclass(frozen=True, slots=True, eq=False)
class StaticResponseRequest:
    """Finite observed ADMG mean responses and a selected point contrast.

    The checked general-ID graph retains at least one bidirected edge.
    ``support`` declares at least two finite points in strictly increasing order.
    ``actions`` selects or reorders those points without declaring extrapolation. The contrast is
    the selected active mean minus baseline mean.
    """

    data: Mapping[str, ArrayLike]
    graph: Admg
    treatment: str
    outcome: str
    support: Sequence[float]
    utility: Utility
    actions: Sequence[float] | None = None
    baseline: int = 0
    active: int = 1


@dataclass(frozen=True, slots=True, eq=False)
class MultiSourceRequest:
    """Catalog-bound finite multi-source transport, preserving source regimes.

    Laws use the existing ExactDiscreteLaw contract. ``assignments`` contains
    the intervention requests; each has one mean outcome coordinate. Changing
    any source proof or snapshot requires the native route to recheck it.
    """

    graph: Admg
    query: MultiSourceZTransportQuery
    catalog: EvidenceCatalog
    laws: Sequence[ExactDiscreteLaw]
    assignments: Sequence[Mapping[str, float]]
    utility: Utility
    baseline: int = 0
    active: int = 1
    search_operations: int = 4096
    search_depth: int = 24
    evaluation_operations: int = 10_000_000
    evaluation_depth: int = 256


@dataclass(frozen=True, slots=True)
class StaticResult:
    """Executed point means, selected contrast, utility and measured receipt."""

    plan: RecalcPlan
    receipt: RecalcReceipt
    means: tuple[float, ...]
    contrast: float
    decision: Decision

    def to_dict(self) -> dict[str, object]:
        return {
            "means": list(self.means),
            "contrast": self.contrast,
            "decision": {"net_benefit": self.decision.net_benefit, "treat": self.decision.treat},
            "uncertainty": {"status": "unavailable", "reason": "point_only"},
        }


def _utility(request: StaticResponseRequest | MultiSourceRequest) -> dict[str, Any]:
    return {
        "baseline": request.baseline,
        "active": request.active,
        "benefit_per_unit": float(request.utility.benefit_per_unit),
        "cost": float(request.utility.cost),
    }


def _response_spec(request: StaticResponseRequest) -> str:
    return json.dumps(
        {
            **_utility(request),
            "treatment": request.treatment,
            "outcome": request.outcome,
            "support": list(request.support),
            "actions": list(request.support if request.actions is None else request.actions),
        }
    )


def _transport_spec(request: MultiSourceRequest) -> str:
    query = request.query
    if not isinstance(query, MultiSourceZTransportQuery):
        raise CausalTypeError(
            "query must be MultiSourceZTransportQuery", reason_code="invalid_argument"
        )
    return json.dumps(
        {
            **_utility(request),
            "target": query.target,
            "outcomes": list(query.outcomes),
            "treatments": list(query.treatments),
            "sources": [
                {
                    "population": source.population,
                    "controllable": list(source.controllable),
                    "selections": list(source.selections),
                    "experiment_assignment": dict(source.experiment_assignment),
                }
                for source in query.sources
            ],
            "assignments": [dict(assignment) for assignment in request.assignments],
            "search_operations": request.search_operations,
            "search_depth": request.search_depth,
            "evaluation_operations": request.evaluation_operations,
            "evaluation_depth": request.evaluation_depth,
        }
    )


def _result(payload: tuple[str | None, bytes | None, str | None]) -> StaticResult:
    result, artifact, error = payload
    raise_refusal(error, _refusal_type)
    if result is None or artifact is None:  # pragma: no cover - native invariant
        raise CausalSerializationError("static recalculation returned no result")
    wire = json.loads(result)
    return StaticResult(
        RecalcPlan.from_wire(wire["plan"]),
        RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False),
        tuple(wire["means"]),
        wire["contrast"],
        Decision(**wire["decision"]),
    )


class _Session:
    __slots__ = ("_handle",)

    _standing_kind: ClassVar[SessionKind]

    @staticmethod
    def _native_type() -> Any:
        raise NotImplementedError

    def __init__(self) -> None:
        self._handle = self._native_type()()

    @classmethod
    def resume(
        cls, previous: Mapping[Stage, str] | RecalcReceipt, context: ResumeContext | None = None
    ) -> Any:
        """Historical identities only; supplied compatible raw inputs require refitting."""
        declared = previous.requested if isinstance(previous, RecalcReceipt) else previous
        session = cls.__new__(cls)
        session._handle = cls._native_type().resume(
            _declared_json(declared), json.dumps((context or ResumeContext()).to_wire())
        )
        return session

    @property
    def is_live(self) -> bool:
        return bool(self._handle.is_live())

    @property
    def identities(self) -> dict[Stage, str]:
        return _declared(json.loads(self._handle.identities_json()))

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities.from_wire(json.loads(self._handle.capabilities_json()))

    @property
    def standing(self) -> SessionStanding:
        """The standing inputs this session family's stage identities declare."""
        return session_standing(self._standing_kind)

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Export the retained original engine artifact, independently consumable.

        This result artifact supplies no general executable-state resume and no
        calibrated interval license. The shared work receipt exports separately.
        """
        run = _execution(execution)
        artifact, error = run._call(
            self._handle.export_result, **_native_controls(run, resources=False)
        )
        raise_refusal(error, _refusal_type)
        if artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("static recalculation exported no artifact")
        return artifact_from_receipt(self, bytes(artifact))


class StaticResponseSession(_Session):
    """Retained checked finite ADMG response program and evaluated response grid."""

    __slots__ = ()

    _standing_kind = "static"

    # Resolve lazily so importing this module does not instantiate native state.
    @staticmethod
    def _native_type() -> Any:
        return _native.StaticResponseSessionHandle

    @property
    def response(self) -> Any:
        """Original full-support response from retained native execution, with zero work."""
        from .estimation import _wrap_prepared_response
        from .query import ResponseCurve

        raw = self._handle.response()
        if raw is None:
            return None
        basis = json.loads(raw.program_basis_json())
        view = _wrap_prepared_response(
            raw, ResponseCurve(basis["treatment"], basis["outcome"], grid=basis["grid"])
        )
        return view.model_copy(
            update={"data_snapshot_id": basis["snapshot_id"], "program_id": basis["program_id"]}
        )

    def plan(
        self,
        request: StaticResponseRequest,
        *,
        execution: Execution | None = None,
    ) -> RecalcPlan:
        run = _execution(execution)
        ctl = _native_controls(run)
        names, columns = _columns(
            request.data, memory_limit_bytes=ctl["memory_limit_bytes"], cancel=ctl["cancel"]
        )
        return RecalcPlan.from_wire(
            json.loads(
                run._call(
                    self._handle.plan, names, columns, request.graph, _response_spec(request), **ctl
                )
            )
        )

    @outcome_operation
    def execute(
        self,
        request: StaticResponseRequest,
        *,
        execution: Execution | None = None,
    ) -> StaticResult:
        """Run the invalidated stages under the execution's memory limit and cancellation."""
        run = _execution(execution)
        ctl = _native_controls(run)
        names, columns = _columns(
            request.data, memory_limit_bytes=ctl["memory_limit_bytes"], cancel=ctl["cancel"]
        )
        return _result(
            run._call(
                self._handle.execute,
                names,
                columns,
                request.graph,
                _response_spec(request),
                **ctl,
            )
        )


class MultiSourceSession(_Session):
    """Retained checked multi-source proof, catalog, exact laws and compiled plans."""

    __slots__ = ()

    _standing_kind = "static"

    @staticmethod
    def _native_type() -> Any:
        return _native.MultiSourceSessionHandle

    def plan(
        self, request: MultiSourceRequest, *, execution: Execution | None = None
    ) -> RecalcPlan:
        run = _execution(execution)
        return RecalcPlan.from_wire(
            json.loads(
                run._call(
                    self._handle.plan,
                    request.graph,
                    request.catalog,
                    request.laws,
                    _transport_spec(request),
                    **_native_controls(run, resources=False),
                )
            )
        )

    @outcome_operation
    def execute(
        self, request: MultiSourceRequest, *, execution: Execution | None = None
    ) -> StaticResult:
        run = _execution(execution)
        return _result(
            run._call(
                self._handle.execute,
                request.graph,
                request.catalog,
                request.laws,
                _transport_spec(request),
                **_native_controls(run, resources=False),
            )
        )


__all__ = [
    "MultiSourceRequest",
    "MultiSourceSession",
    "StaticResponseRequest",
    "StaticResponseSession",
    "StaticResult",
]
