"""General cross-world selective execution over a declared finite Markovian SCM.

Counterfactual and natural / path-specific quantities are computed from **coupled worlds**:
each unit's exogenous posterior given its factual evidence is enumerated exactly once, and every
named world is evaluated under that same posterior state. The pipeline keeps its stages separate
(mechanism fitting, factual abduction, exogenous draws, action-specific prediction, summary):

* an **action-only change** (new worlds, routes or quantity) reuses the retained abduction, and
  the observed work counters (:attr:`ExecutionResult.work`) show it was not redone;
* changed factual evidence, a changed mechanism parameter, graph, population or history rebuilds
  it;
* independently drawn world noise is never paired as a counterfactual joint law: a cross-world
  event with ``coupling="independent_world_noise"`` or an undeclared coupling raises
  ``joint_law_required``; latent confounding, another population or an incompatible history are
  scientific refusals naming what would license them (:func:`capability_matrix`, :func:`decide`).

Every value is a ``point_only`` result of exact enumeration under the declared model (mechanism
adequacy is a premise, not a check). Intervals, posteriors and a sampled abduction are closed:
:meth:`CrossWorldExecution.interval` and :meth:`CrossWorldExecution.posterior` always raise
``cell_not_licensed``. :meth:`CrossWorldExecution.to_bytes` writes ``cross_world_execution_v1``;
:func:`consume_cross_world_execution_artifact` replays everything with a fresh executor and
refuses any edit. Every number, identity and refusal is computed in Rust.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "CapabilityRow",
    "ConsumedCrossWorldExecution",
    "CrossWorldExecution",
    "CrossWorldExecutionRefusal",
    "CrossWorldExecutor",
    "CrossWorldRequest",
    "ExecutionResult",
    "ExogenousTerm",
    "FactualUnit",
    "FiniteSCM",
    "FiniteVariable",
    "World",
    "WorldRoute",
    "capability_matrix",
    "consume_cross_world_execution_artifact",
    "decide",
    "joint_probability",
    "mean_contrast",
    "probability",
    "seal",
]


class CrossWorldExecutionRefusal(StructuredRefusal):
    """A typed scientific refusal of a cross-world request.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``cross_world_not_identified``, ``joint_law_required``, ``cell_not_licensed``,
    ``route_not_supported``, ``invalid_argument``) and a ``cross_world_execution.*`` or
    ``joint_law_propagation.*`` ``detail``; ``remedy`` names the declaration that would license
    the request and ``missing`` the unmet prerequisites.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: Unmet prerequisites the refusal names (empty when none applies).
        self.missing: tuple[str, ...] = tuple(refusal.get("missing", ()))


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, CrossWorldExecutionRefusal)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExogenousTerm:
    """A finite exogenous term with its state probabilities."""

    name: str
    pmf: tuple[float, ...]


@dataclass(frozen=True)
class FiniteVariable:
    """A finite variable and its mechanism table.

    ``table[u * n_cfg + cfg]`` is the level for exogenous state ``u`` and parent configuration
    ``cfg``, the mixed-radix index of the parent levels (first parent most significant).
    ``parents`` and ``exogenous`` may be names or indices.
    """

    name: str
    levels: int
    parents: tuple[str | int, ...]
    exogenous: str | int
    table: tuple[int, ...]


@dataclass(frozen=True)
class FiniteSCM:
    """A static Markovian SCM over finite variables, declared for one population and history."""

    population: str
    history: str
    exogenous: tuple[ExogenousTerm, ...]
    variables: tuple[FiniteVariable, ...]


@dataclass(frozen=True)
class FactualUnit:
    """One unit's factual evidence as ``{variable: level}`` (possibly empty)."""

    unit: str
    observed: Mapping[str | int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldRoute:
    """A cross-world route: ``child`` reads ``parent`` from world ``source``."""

    parent: str | int
    child: str | int
    source: str | int


@dataclass(frozen=True)
class World:
    """A named world: hard interventions ``{variable: level}`` and cross-world routes."""

    name: str
    interventions: Mapping[str | int, int] = field(default_factory=dict)
    routes: tuple[WorldRoute, ...] = ()


@dataclass(frozen=True)
class _Quantity:
    kind: str
    atoms: tuple[tuple[str | int, str | int, int], ...] = ()
    plus: tuple[str | int, str | int] | None = None
    minus: tuple[str | int, str | int] | None = None


def probability(world: str | int, variable: str | int, level: int) -> _Quantity:
    """``P(variable = level in world | evidence)``."""
    return _Quantity("probability", atoms=((world, variable, level),))


def mean_contrast(
    plus: tuple[str | int, str | int], minus: tuple[str | int, str | int]
) -> _Quantity:
    """``E[V in world a] - E[V in world b]`` with the level index as the value."""
    return _Quantity("mean_contrast", plus=plus, minus=minus)


def joint_probability(atoms: Sequence[tuple[str | int, str | int, int]]) -> _Quantity:
    """The probability that every ``(world, variable, level)`` atom holds at once."""
    return _Quantity("joint_probability", atoms=tuple(atoms))


@dataclass(frozen=True)
class CrossWorldRequest:
    """A cross-world request over a finite Markovian SCM.

    ``coupling`` is ``shared_abduced_exogenous`` (every world reads the unit's one exogenous
    posterior state), ``independent_world_noise`` or ``undeclared``. ``reading`` is ``point``;
    ``interval`` and ``posterior`` are refused. ``evidence_history`` defaults to the SCM's.
    """

    scm: FiniteSCM
    units: tuple[FactualUnit, ...]
    worlds: tuple[World, ...]
    quantity: _Quantity
    coupling: str = "shared_abduced_exogenous"
    reading: str = "point"
    exogenous_inference: str = "exact_enumeration"
    target_population: str | None = None
    evidence_history: str | None = None

    def to_json(self) -> str:
        """The wire form handed to Rust (names resolved to indices)."""
        v_names = [v.name for v in self.scm.variables]
        e_names = [e.name for e in self.scm.exogenous]
        w_names = [w.name for w in self.worlds]

        def index(ref: str | int, names: list[str], what: str) -> int:
            if isinstance(ref, bool):
                raise CausalTypeError(f"{what} must be a name or an index")
            if isinstance(ref, int):
                return ref
            if ref in names:
                return names.index(ref)
            raise CausalValueError(f"unknown {what} `{ref}`")

        def v(ref: str | int) -> int:
            return index(ref, v_names, "variable")

        def w(ref: str | int) -> int:
            return index(ref, w_names, "world")

        def level(value: object, what: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
                raise CausalValueError(f"{what} must be a small non-negative integer")
            return value

        quantity = self.quantity
        wire: dict[str, Any] = {
            "scm": {
                "population": _text(self.scm.population, "population"),
                "history": _text(self.scm.history, "history"),
                "exogenous": [
                    {"name": e.name, "pmf": [float(p) for p in e.pmf]} for e in self.scm.exogenous
                ],
                "variables": [
                    {
                        "name": var.name,
                        "levels": var.levels,
                        "parents": [v(p) for p in var.parents],
                        "exogenous": index(var.exogenous, e_names, "exogenous term"),
                        "table": [level(t, "table entry") for t in var.table],
                    }
                    for var in self.scm.variables
                ],
            },
            "evidence_history": self.evidence_history or self.scm.history,
            "units": [
                {
                    "unit": u.unit,
                    "observed": sorted(
                        [v(k), level(val, "observed level")] for k, val in u.observed.items()
                    ),
                }
                for u in self.units
            ],
            "worlds": [
                {
                    "name": wd.name,
                    "interventions": sorted(
                        [v(k), level(val, "intervention level")]
                        for k, val in wd.interventions.items()
                    ),
                    "routes": [
                        {"parent": v(r.parent), "child": v(r.child), "source": w(r.source)}
                        for r in wd.routes
                    ],
                }
                for wd in self.worlds
            ],
            "quantity": {
                "kind": quantity.kind,
                "atoms": [
                    {"world": w(a[0]), "variable": v(a[1]), "level": level(a[2], "level")}
                    for a in quantity.atoms
                ],
                "plus": None
                if quantity.plus is None
                else [w(quantity.plus[0]), v(quantity.plus[1])],
                "minus": None
                if quantity.minus is None
                else [w(quantity.minus[0]), v(quantity.minus[1])],
            },
            "coupling": self.coupling,
            "exogenous": self.exogenous_inference,
            "target_population": self.target_population,
            "reading": self.reading,
        }
        return json.dumps(wire, allow_nan=False)


@dataclass(frozen=True)
class CapabilityRow:
    """One row of the theorem-scoped capability matrix."""

    id: str
    quantity: str
    model: str
    conditioning: tuple[str, ...]
    history: tuple[str, ...]
    evidence: tuple[str, ...]
    population: str
    coupling: str
    status: str
    route: str
    theorem: str
    refusal_code: str | None
    refusal_detail: str | None
    remedy: str


def capability_matrix() -> tuple[CapabilityRow, ...]:
    """The capability matrix: what is executed, what keeps its own route, what is closed."""
    rows = json.loads(_native.cross_world_capability_matrix())
    return tuple(
        CapabilityRow(
            **{
                **row,
                "conditioning": tuple(row["conditioning"]),
                "history": tuple(row["history"]),
                "evidence": tuple(row["evidence"]),
            }
        )
        for row in rows
    )


def decide(
    *,
    quantity: str,
    model: str = "finite_markovian_scm",
    conditioning: str = "factual_evidence",
    interventions: str = "hard",
    population: str = "single",
    history: str = "no_history",
    evidence: str = "declared_mechanisms",
) -> dict[str, str]:
    """Decide a matrix cell.

    Returns ``{"id", "status", "route", "coupling"}`` of the licensed row.

    Raises:
        CrossWorldExecutionRefusal: the request is outside every executed row; the refusal names
            the closed row or dimension and what would open it.
    """
    row, refusal = _native.cross_world_decide(
        quantity, model, conditioning, interventions, population, history, evidence
    )
    _refuse(refusal)
    assert row is not None
    return dict(json.loads(row))


@dataclass(frozen=True)
class UnitAnswer:
    """One unit's value and ``P(evidence)``."""

    unit: str
    value: float
    evidence_probability: float


@dataclass(frozen=True)
class ExecutionResult:
    """The answer of one execution.

    ``work`` holds the observed counters of *this* call (``abductions``, ``abductions_reused``,
    ``world_predictions``, ``shared_draws``, ``mechanism_fits``, ...); ``stages`` the identity and
    ``declared`` / ``computed`` / ``reused`` status of each stage.
    """

    matrix_row: str
    value: float
    label: str
    units: tuple[UnitAnswer, ...]
    stages: tuple[Mapping[str, str], ...]
    work: Mapping[str, int]
    receipt: Mapping[str, str]

    @classmethod
    def _from(cls, info: Mapping[str, Any]) -> ExecutionResult:
        return cls(
            matrix_row=info["matrix_row"],
            value=info["value"],
            label=info["label"],
            units=tuple(UnitAnswer(**u) for u in info["units"]),
            stages=tuple(info["stages"]),
            work=dict(info["work"]),
            receipt=dict(info["receipt"]),
        )


class CrossWorldExecutor:
    """A selective executor that retains abductions across calls.

    A request whose mechanisms and factual evidence are unchanged reuses the retained abduction
    and shared exogenous states; ``execute(...).work["abductions"]`` is then ``0``.
    """

    def __init__(self, _native_executor: Any = None) -> None:
        self._native = (
            _native_executor if _native_executor is not None else _native.CrossWorldExecutor()
        )

    def execute(self, request: CrossWorldRequest, *, cancel: Any = None) -> ExecutionResult:
        """Execute a request, reusing matching retained abductions."""
        if not isinstance(request, CrossWorldRequest):
            raise CausalTypeError("request must be a CrossWorldRequest")
        payload, refusal = self._native.execute(request.to_json(), cancel)
        _refuse(refusal)
        assert payload is not None
        return ExecutionResult._from(json.loads(payload))

    def retained_len(self) -> int:
        """Number of retained abductions."""
        return int(self._native.retained_len())

    def counters(self) -> Mapping[str, int]:
        """Cumulative observed work of successful executions."""
        return dict(json.loads(self._native.counters()))

    def clear(self) -> None:
        """Drop every retained abduction (a process boundary)."""
        self._native.clear()

    @classmethod
    def from_artifact(
        cls, data: bytes, *, expected_identity: Mapping[str, str] | None = None, cancel: Any = None
    ) -> CrossWorldExecutor:
        """An executor warmed by an artifact's verified abductions (a licensed resume)."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise CausalTypeError("data must be bytes")
        identity = None if expected_identity is None else json.dumps(dict(expected_identity))
        parsed, refusal = _native.CrossWorldExecutor.from_artifact(bytes(data), identity, cancel)
        _refuse(refusal)
        assert parsed is not None
        return cls(parsed[0])


@dataclass(frozen=True)
class ConsumedCrossWorldExecution:
    """A consumed artifact: replayed result, identity and shared exogenous states."""

    result: ExecutionResult
    identity: Mapping[str, str]
    matrix_row: str
    inference_claim: str
    shared_draws: tuple[Mapping[str, Any], ...]


def _consumed(summary: str) -> ConsumedCrossWorldExecution:
    info = json.loads(summary)
    return ConsumedCrossWorldExecution(
        result=ExecutionResult._from(info["result"]),
        identity=dict(info["identity"]),
        matrix_row=info["matrix_row"],
        inference_claim=info["inference_claim"],
        shared_draws=tuple(info["shared_draws"]),
    )


class CrossWorldExecution:
    """A sealed execution: the answer, its identities and the exportable artifact."""

    def __init__(self, data: bytes, consumed: ConsumedCrossWorldExecution) -> None:
        self._data = data
        self._consumed = consumed

    @property
    def result(self) -> ExecutionResult:
        """The answer."""
        return self._consumed.result

    @property
    def value(self) -> float:
        """The point value (mean over units)."""
        return self._consumed.result.value

    @property
    def identity(self) -> Mapping[str, str]:
        """Identity digests of the declarations and stages."""
        return self._consumed.identity

    @property
    def shared_draws(self) -> tuple[Mapping[str, Any], ...]:
        """Each unit's shared exogenous states."""
        return self._consumed.shared_draws

    def to_bytes(self) -> bytes:
        """The ``cross_world_execution_v1`` artifact bytes."""
        return self._data

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: no whole-path coverage record exists."""
        _refuse(_native.cross_world_closed_reading("interval"))

    def posterior(self) -> None:
        """Always raises ``cell_not_licensed``: no whole-path calibration record exists."""
        _refuse(_native.cross_world_closed_reading("posterior"))


def seal(
    request: CrossWorldRequest, *, artifact_id: str = "cross-world-execution", cancel: Any = None
) -> CrossWorldExecution:
    """Execute with a fresh executor and seal the answer, declarations and shared states.

    Raises:
        CrossWorldExecutionRefusal: the request is outside the matrix or lacks a licensed
            coupling (see :func:`decide`).
    """
    if not isinstance(request, CrossWorldRequest):
        raise CausalTypeError("request must be a CrossWorldRequest")
    parts, refusal = _native.cross_world_seal(
        request.to_json(), _text(artifact_id, "artifact_id"), cancel
    )
    _refuse(refusal)
    assert parts is not None
    data, summary = parts
    if not math.isfinite(json.loads(summary)["result"]["value"]):
        raise CausalValueError("non-finite cross-world value")
    return CrossWorldExecution(bytes(data), _consumed(summary))


def consume_cross_world_execution_artifact(
    data: bytes, *, expected_identity: Mapping[str, str] | None = None, cancel: Any = None
) -> ConsumedCrossWorldExecution:
    """Replay an artifact with a fresh executor and refuse any difference.

    Raises:
        CrossWorldExecutionRefusal: an edited, truncated or foreign artifact, a retained
            identity that differs, or a declaration the matrix or the joint-law rule no longer
            licenses.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError("data must be bytes")
    identity = None if expected_identity is None else json.dumps(dict(expected_identity))
    summary, refusal = _native.cross_world_consume(bytes(data), identity, cancel)
    _refuse(refusal)
    assert summary is not None
    return _consumed(summary)
