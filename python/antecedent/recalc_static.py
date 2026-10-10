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
from typing import Any

from numpy.typing import ArrayLike

from . import _native
from ._recalc_bounds import _columns
from .errors import CausalSerializationError, CausalTypeError
from .graph import Admg
from .recalc import (
    Capabilities,
    Decision,
    RecalcPlan,
    RecalcReceipt,
    ResumeContext,
    Stage,
    Utility,
    _declared,
    _declared_json,
    _raise,
    _seed,
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
    _raise(error)
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

    def export_result(self, *, seed: int = 1, threads: int | None = None) -> bytes:
        """Export the retained original engine artifact, independently consumable.

        This result artifact supplies no general executable-state resume and no
        calibrated interval license. The shared work receipt exports separately.
        """
        artifact, error = self._handle.export_result(seed=_seed(seed), threads=threads)
        _raise(error)
        if artifact is None:  # pragma: no cover - native invariant
            raise CausalSerializationError("static recalculation exported no artifact")
        return bytes(artifact)


class StaticResponseSession(_Session):
    """Retained checked finite ADMG response program and evaluated response grid."""

    __slots__ = ()

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
        self, request: StaticResponseRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        names, columns = _columns(request.data)
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(
                    names,
                    columns,
                    request.graph,
                    _response_spec(request),
                    seed=_seed(seed),
                    threads=threads,
                )
            )
        )

    def execute(
        self, request: StaticResponseRequest, *, seed: int = 1, threads: int | None = None
    ) -> StaticResult:
        names, columns = _columns(request.data)
        return _result(
            self._handle.execute(
                names,
                columns,
                request.graph,
                _response_spec(request),
                seed=_seed(seed),
                threads=threads,
            )
        )


class MultiSourceSession(_Session):
    """Retained checked multi-source proof, catalog, exact laws and compiled plans."""

    __slots__ = ()

    @staticmethod
    def _native_type() -> Any:
        return _native.MultiSourceSessionHandle

    def plan(
        self, request: MultiSourceRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        return RecalcPlan.from_wire(
            json.loads(
                self._handle.plan(
                    request.graph,
                    request.catalog,
                    request.laws,
                    _transport_spec(request),
                    seed=_seed(seed),
                    threads=threads,
                )
            )
        )

    def execute(
        self, request: MultiSourceRequest, *, seed: int = 1, threads: int | None = None
    ) -> StaticResult:
        return _result(
            self._handle.execute(
                request.graph,
                request.catalog,
                request.laws,
                _transport_spec(request),
                seed=_seed(seed),
                threads=threads,
            )
        )


__all__ = [
    "MultiSourceRequest",
    "MultiSourceSession",
    "StaticResponseRequest",
    "StaticResponseSession",
    "StaticResult",
]
