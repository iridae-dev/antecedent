"""Ordered-response observation recovery (2.3 B2): a response chain ``R -> R``.

A second, separate observation-recovery row. The 2.2 route
(:mod:`antecedent.transport` ``identify_observation_recovery``) refuses every response-to-response
edge; this row is exactly the graphs it refuses because of **one** response chain, with a
different identifying order: the head response first, then the tail response given the head.

The m-graph is over binary ``X1, X2`` (partially observed), their response indicators ``R1, R2``
(1 observed, 0 missing) and deterministic proxies ``X*1, X*2`` (``X*_i = X_i`` when ``R_i = 1``,
``"?"`` when ``R_i = 0``), nothing else: any substantive edge between ``X1`` and ``X2`` (or
none), the proxy wiring ``X_i -> X*_i <- R_i``, exactly one response edge ``R_h -> R_t`` and,
optionally, ``X_h -> R_t``. ``R_h`` has no parent other than a possible self-censoring ``X_h``.

The observed margin is the nine-cell pattern law :class:`PatternLaw`
``P(R1, R2, X*1, X*2)``; the target is the full law ``P(X1, X2)``. With the head ``h`` and
tail ``t`` the recovered cell is ``both(x_h, x_t) / (P(R_h=1) q(x_h))`` where ``q(x_h)`` is
``P(R_h=1, R_t=1, X*_h=x_h) / P(R_h=1, X*_h=x_h)`` when the tail response depends on the head
variable and ``P(R_h=1, R_t=1) / P(R_h=1)`` otherwise. This is graph-licensed recovery: not
complete-case analysis, MAR or inverse-probability weighting.

A **self-censoring edge** (``X_i -> R_i``) with every other edge in the class is refused as
nonrecoverable with an exactly verified witness (:class:`ChainWitness`): two models Markov to
the m-graph that agree on all nine observed cells and differ on the target. The witness models
are degenerate, so nothing is claimed under faithful or generic parameters. An edge outside the
row is ``route_not_supported`` (neither recoverable nor nonrecoverable is claimed); a missing
cell is ``transport_support_failure``. Exact laws only: a sampled provider composed on this row
is calibrated separately, and that calibration is unmeasured and closed.

:meth:`RecoveredChain.export` writes an artifact; :func:`consume_recovery_chain_artifact`
re-decides, re-evaluates and re-verifies the witness and refuses any change, even one resealed
with fresh digests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._native import consume_recovery_chain_artifact as _consume
from ._native import recover_recovery_chain as _recover
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .graph import Admg

__all__ = [
    "ChainDecision",
    "ChainPlan",
    "ChainRoles",
    "ChainWitness",
    "PatternLaw",
    "RecoveredChain",
    "RecoveryChainQuery",
    "RecoveryChainRefusal",
    "consume_recovery_chain_artifact",
    "decide_chain",
    "recover_chain",
]


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty name")
    return value


@dataclass(frozen=True, slots=True)
class ChainRoles:
    """A partially observed binary variable with its response indicator and proxy."""

    variable: str
    response: str
    proxy: str

    def __post_init__(self) -> None:
        names = tuple(_name(getattr(self, f), f) for f in ("variable", "response", "proxy"))
        if len(set(names)) != 3:
            raise CausalValueError("variable, response and proxy must be three distinct nodes")

    def _wire(self) -> tuple[str, str, str]:
        return (self.variable, self.response, self.proxy)


@dataclass(frozen=True, slots=True)
class RecoveryChainQuery:
    """Recover ``P(X1, X2)``; the declaration order fixes the axis order of every law."""

    first: ChainRoles
    second: ChainRoles

    def __post_init__(self) -> None:
        for item in (self.first, self.second):
            if not isinstance(item, ChainRoles):
                raise CausalTypeError("first and second must be ChainRoles")


@dataclass(frozen=True, slots=True)
class PatternLaw:
    """The exact observed pattern law over the nine proxy cells.

    ``both[x1][x2] = P(R1=1, R2=1, X*1=x1, X*2=x2)``, ``only_first[x1] = P(R1=1, R2=0, X*1=x1)``,
    ``only_second[x2] = P(R1=0, R2=1, X*2=x2)`` and ``neither = P(R1=0, R2=0)``. The cells must
    be non-negative, finite and sum to one (``invalid_argument``,
    ``recovery_chain.invalid_observed_law``).
    """

    both: tuple[tuple[float, float], tuple[float, float]]
    only_first: tuple[float, float]
    only_second: tuple[float, float]
    neither: float

    def __post_init__(self) -> None:
        try:
            both = tuple(tuple(float(c) for c in row) for row in self.both)
            only_first = tuple(float(c) for c in self.only_first)
            only_second = tuple(float(c) for c in self.only_second)
            neither = float(self.neither)
        except (TypeError, ValueError) as error:
            raise CausalTypeError("the pattern law cells must be real numbers") from error
        if (
            len(both) != 2
            or any(len(row) != 2 for row in both)
            or len(only_first) != 2
            or len(only_second) != 2
        ):
            raise CausalValueError("both is 2 x 2 and only_first / only_second have two cells")
        object.__setattr__(self, "both", both)
        object.__setattr__(self, "only_first", only_first)
        object.__setattr__(self, "only_second", only_second)
        object.__setattr__(self, "neither", neither)

    def _cells(self) -> list[float]:
        return [*(c for row in self.both for c in row), *self.only_first, *self.only_second]

    def _wire(self) -> str:
        if not all(math.isfinite(c) for c in [*self._cells(), self.neither]):
            raise _refusal_of(
                "invalid_argument",
                "recovery_chain.invalid_observed_law",
                "an observed pattern cell is negative or not finite",
            )
        return json.dumps(
            {
                "both": [list(row) for row in self.both],
                "only_first": list(self.only_first),
                "only_second": list(self.only_second),
                "neither": self.neither,
            },
            allow_nan=False,
        )

    @classmethod
    def from_mapping(cls, cells: Mapping[str, Any]) -> PatternLaw:
        """Build a law from a mapping of ``both``, ``only_first``, ``only_second``, ``neither``."""
        try:
            return cls(
                both=cells["both"],
                only_first=cells["only_first"],
                only_second=cells["only_second"],
                neither=cells["neither"],
            )
        except KeyError as error:
            raise CausalValueError(f"the pattern law is missing {error}") from error


@dataclass(frozen=True, slots=True)
class ChainPlan:
    """The checked recovery plan."""

    head: int
    tail_depends_on_head_variable: bool
    rule_version: str
    premises: tuple[str, ...]
    formula: str
    operations_consumed: int


@dataclass(frozen=True, slots=True)
class ChainWitness:
    """A verified nonrecoverability witness.

    ``self_censoring_edge`` is ``(X_i, R_i)``. Each model lists, for ``X1, X2, R1, R2`` in order,
    ``{"node", "parents", "p_one_times_60"}`` (``P(node = 1 | parents) = k / 60`` per parent
    configuration, first parent most significant). The models agree on every one of
    ``observed_cells_equal`` observed pattern cells and differ at ``differing_target_cell`` of the
    target law, where their masses over ``denominator`` (``60**4``) are ``target_masses``. The
    verification is exact integer enumeration.
    """

    self_censoring_edge: tuple[str, str]
    first_model: tuple[Mapping[str, Any], ...]
    second_model: tuple[Mapping[str, Any], ...]
    observed_cells_equal: int
    differing_target_cell: tuple[int, int]
    target_masses: tuple[int, int]
    denominator: int


class RecoveryChainRefusal(CausalUnsupportedError):
    """A :class:`CausalUnsupportedError` carrying the structured Rust refusal fields.

    ``detail`` is the namespaced ``recovery_chain.*`` slot (``.nonrecoverable_witness``,
    ``.unsupported_mechanism``, ``.invalid_query``, ``.positivity``, ``.invalid_observed_law``,
    ``.invalid_derivation``, ``.budget`` or a replay mismatch). A nonrecoverable refusal carries
    its verified ``witness`` and the exportable ``artifact`` of the decision.
    """

    def __init__(
        self,
        refusal: Mapping[str, Any],
        *,
        witness: ChainWitness | None = None,
        artifact: bytes | None = None,
    ) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        text = f"{detail}: {message}" if message else detail
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``recovery_chain.*`` detail.
        self.detail: str = detail
        #: Offending field, when there is one.
        self.offending: str | None = refusal.get("offending")
        #: The verified witness of a nonrecoverable decision.
        self.witness: ChainWitness | None = witness
        #: The exportable artifact of a nonrecoverable decision.
        self.artifact: bytes | None = artifact


def _refusal_of(code: str, detail: str, message: str) -> RecoveryChainRefusal:
    return RecoveryChainRefusal(
        {"code": code, "stage": "recovery_chain", "detail": detail, "message": message}
    )


def _raise_refusal(payload: str | None) -> None:
    if payload is not None:
        raise RecoveryChainRefusal(json.loads(payload))


def _witness(report: Mapping[str, Any]) -> ChainWitness:
    witness = report["witness"]
    check = report["witness_check"]
    return ChainWitness(
        self_censoring_edge=(witness["self_censoring_edge"][0], witness["self_censoring_edge"][1]),
        first_model=tuple(witness["first_model"]),
        second_model=tuple(witness["second_model"]),
        observed_cells_equal=int(check["observed_cells_equal"]),
        differing_target_cell=(
            int(check["differing_target_cell"][0]),
            int(check["differing_target_cell"][1]),
        ),
        target_masses=(int(check["target_masses"][0]), int(check["target_masses"][1])),
        denominator=int(check["denominator"]),
    )


def _plan(plan: Mapping[str, Any]) -> ChainPlan:
    return ChainPlan(
        head=int(plan["head"]),
        tail_depends_on_head_variable=bool(plan["tail_depends_on_head_variable"]),
        rule_version=plan["rule_version"],
        premises=tuple(plan["premises"]),
        formula=plan["formula"],
        operations_consumed=int(plan["operations_consumed"]),
    )


def _witness_refusal(report: Mapping[str, Any], artifact: bytes | None) -> RecoveryChainRefusal:
    witness = _witness(report)
    edge = witness.self_censoring_edge
    return RecoveryChainRefusal(
        {
            "code": report["reason"],
            "stage": "recovery_chain",
            "detail": report["detail"],
            "message": (
                f"self-censoring edge {edge[0]} -> {edge[1]}: a verified witness shows the target "
                "is not recoverable for every model Markov to the m-graph"
            ),
        },
        witness=witness,
        artifact=artifact,
    )


@dataclass(frozen=True, slots=True)
class ChainDecision:
    """A decided query, without evaluation: a checked plan or a verified witness."""

    outcome: str
    plan: ChainPlan | None
    witness: ChainWitness | None
    artifact: bytes | None = field(default=None, repr=False, compare=False)

    def export(self) -> bytes:
        """The artifact of a nonrecoverable decision (a recovered one has none until evaluated)."""
        if self.artifact is None:
            raise _refusal_of(
                "invalid_argument",
                "recovery_chain.not_evaluated",
                "evaluate the recovered decision on an observed pattern law before exporting",
            )
        return self.artifact


@dataclass(frozen=True, slots=True)
class RecoveredChain:
    """The recovered law ``P(X1, X2)`` with the plan and artifact that license it."""

    plan: ChainPlan
    observed: PatternLaw
    #: ``cells[x1][x2] = P(X1 = x1, X2 = x2)`` in the query's declaration order.
    cells: tuple[tuple[float, float], tuple[float, float]]
    rule_version: str
    premises_digest: str
    data_digest: str
    artifact: bytes = field(repr=False, compare=False)

    @property
    def law(self) -> NDArray[np.float64]:
        """The recovered law as a ``2 x 2`` array, axes in declaration order."""
        return np.asarray(self.cells, dtype=np.float64)

    def export(self) -> bytes:
        """The artifact: graph, query, plan, observed law, recovered law and both digests."""
        return self.artifact

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping of the result."""
        return {
            "outcome": "recovered",
            "cells": [list(row) for row in self.cells],
            "rule_version": self.rule_version,
            "formula": self.plan.formula,
            "head": self.plan.head,
            "tail_depends_on_head_variable": self.plan.tail_depends_on_head_variable,
            "interval": {"available": False, "status": "point_only"},
        }


def _graph_parts(
    graph: Admg | Mapping[str, Any],
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    if isinstance(graph, Admg):
        nodes = list(graph.nodes())
        directed = [(n, c) for n in nodes for c in graph.children(n)]
        position = {n: i for i, n in enumerate(nodes)}
        bidirected = [
            (n, m)
            for n in nodes
            for m in graph.bidirected_neighbors(n)
            if position[n] < position[m]
        ]
        return nodes, directed, bidirected
    if isinstance(graph, Mapping):
        nodes = [_name(n, "node") for n in graph["nodes"]]
        directed = [(str(a), str(b)) for a, b in graph.get("edges", ())]
        bidirected = [(str(a), str(b)) for a, b in graph.get("bidirected", ())]
        return nodes, directed, bidirected
    raise CausalTypeError(
        "graph must be an Admg or a mapping with 'nodes', 'edges' and optional 'bidirected'"
    )


def _query(query: RecoveryChainQuery) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if not isinstance(query, RecoveryChainQuery):
        raise CausalTypeError("query must be a RecoveryChainQuery")
    return query.first._wire(), query.second._wire()


def _law(pattern_law: PatternLaw | Mapping[str, Any] | None) -> str | None:
    if pattern_law is None:
        return None
    if isinstance(pattern_law, Mapping):
        pattern_law = PatternLaw.from_mapping(pattern_law)
    if not isinstance(pattern_law, PatternLaw):
        raise CausalTypeError("pattern_law must be a PatternLaw or a mapping of its cells")
    return pattern_law._wire()


def _call(
    graph: Admg | Mapping[str, Any],
    query: RecoveryChainQuery,
    law_json: str | None,
    *,
    seed: int,
    memory_bytes: int | None,
    cancel: Any,
) -> tuple[Mapping[str, Any], bytes | None]:
    nodes, directed, bidirected = _graph_parts(graph)
    first, second = _query(query)
    report, artifact, refusal = _recover(
        nodes,
        directed,
        bidirected,
        first,
        second,
        law_json,
        seed=seed,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native recovery returned neither a result nor a refusal")
    return json.loads(report), None if artifact is None else bytes(artifact)


def _recovered(report: Mapping[str, Any], artifact: bytes) -> RecoveredChain:
    observed = report["observed"]
    recovered = report["recovered"]
    cells = recovered["cells"]
    return RecoveredChain(
        plan=_plan(report["plan"]),
        observed=PatternLaw(
            both=observed["both"],
            only_first=observed["only_first"],
            only_second=observed["only_second"],
            neither=observed["neither"],
        ),
        cells=((float(cells[0][0]), float(cells[0][1])), (float(cells[1][0]), float(cells[1][1]))),
        rule_version=recovered["rule_version"],
        premises_digest=report["premises_digest"],
        data_digest=report["data_digest"],
        artifact=artifact,
    )


def decide_chain(
    graph: Admg | Mapping[str, Any],
    query: RecoveryChainQuery,
    *,
    seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ChainDecision:
    """Decide whether ``P(X1, X2)`` is recoverable, without evaluating a law.

    A recovered decision carries its checked :class:`ChainPlan`; a nonrecoverable one carries its
    verified :class:`ChainWitness` and an exportable artifact (it is a result, not an exception).
    A graph or role outside the row raises :class:`RecoveryChainRefusal`
    (``recovery_chain.unsupported_mechanism`` / ``.invalid_query``).
    """
    report, artifact = _call(
        graph, query, None, seed=seed, memory_bytes=memory_bytes, cancel=cancel
    )
    if report["outcome"] == "nonrecoverable":
        return ChainDecision("nonrecoverable", None, _witness(report), artifact)
    return ChainDecision("recovered", _plan(report["plan"]), None, artifact)


def recover_chain(
    graph: Admg | Mapping[str, Any],
    query: RecoveryChainQuery,
    pattern_law: PatternLaw | Mapping[str, Any],
    *,
    seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> RecoveredChain:
    """Recover ``P(X1, X2)`` from the observed nine-cell ``pattern_law`` on the m-graph.

    ``graph`` is an :class:`~antecedent.graph.Admg` or a mapping ``{"nodes", "edges"}`` over the
    six role nodes. Returns the :class:`RecoveredChain`, or raises
    :class:`RecoveryChainRefusal` whose ``witness`` is the verified nonrecoverability witness
    (``transport_proven_non_transportable``, ``recovery_chain.nonrecoverable_witness``) when the
    graph has a self-censoring edge. Other refusals: ``route_not_supported`` /
    ``recovery_chain.unsupported_mechanism`` (an edge outside the row),
    ``transport_support_failure`` / ``recovery_chain.positivity`` (a required cell or denominator
    has no mass) and ``invalid_argument`` (``.invalid_query``, ``.invalid_observed_law``).
    """
    law_json = _law(pattern_law)
    report, artifact = _call(
        graph, query, law_json, seed=seed, memory_bytes=memory_bytes, cancel=cancel
    )
    if report["outcome"] == "nonrecoverable":
        raise _witness_refusal(report, artifact)
    if artifact is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native recovery returned no artifact for a recovered law")
    return _recovered(report, artifact)


def consume_recovery_chain_artifact(
    artifact: bytes,
    *,
    seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> RecoveredChain:
    """Re-decide, re-evaluate and re-verify an exported artifact; accept only an identical one.

    The graph is re-decided under this consumer's own budget, the stored plan must equal the
    re-decided plan, the observed law is re-validated and the recovered law recomputed and
    compared bit for bit. A nonrecoverable artifact re-verifies its witness by exact enumeration
    and raises :class:`RecoveryChainRefusal` carrying it (as :func:`recover_chain` does). A changed
    edge, role, plan, witness model or cell is refused even when the digests were resealed
    (``invalid_argument``, ``recovery_chain.premises_mismatch`` / ``.data_identity_mismatch`` /
    ``.decision_replay_mismatch`` / ``.recovered_replay_mismatch``). Corruption and unknown
    versions raise :class:`~antecedent.errors.CausalSerializationError`.
    """
    if not isinstance(artifact, bytes | bytearray | memoryview):
        raise CausalTypeError("artifact must be bytes")
    data = bytes(artifact)
    report, refusal = _consume(data, seed=seed, memory_bytes=memory_bytes, cancel=cancel)
    _raise_refusal(refusal)
    if report is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native consumer returned neither a result nor a refusal")
    parsed = json.loads(report)
    if parsed["outcome"] == "nonrecoverable":
        raise _witness_refusal(parsed, data)
    return _recovered(parsed, data)
