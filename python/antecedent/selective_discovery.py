"""Selective discovery: reuse conditional-independence statistics across discovery requests.

Discovery is rerun for reasons that leave most of its work unchanged: a different threshold,
added background knowledge, a different orientation budget, a different lag window. A
:class:`SelectiveDiscovery` adapter retains the statistics a run computed and serves a later
request from them at exactly the input coordinates it needs; the decisions, search and
orientation are rerun on top::

    from antecedent import selective_discovery as sd

    contract = sd.StaticContract(variables=("a", "b", "c"), algorithm="pc", seed=0)
    adapter = sd.SelectiveDiscovery("static")
    first = adapter.run(contract, {"a": a, "b": b, "c": c})
    stricter = adapter.run(sd.replace(contract, alpha=0.01), {"a": a, "b": b, "c": c})
    stricter.work.ci_tests_reused      # statistics served from the first run
    adapter.outcome.unresolved_edges   # edges still undetermined within the equivalence class

What is claimed, and what is not:

* **Algorithmic match.** A selective run returns what a full run of the same request returns,
  bit for bit: a statistic is retained only at an identical input coordinate (test, settings,
  seed, kernel policy, columns and their content).
* **New data invalidates the statistics.** A previous graph is never the start of a search.
* **No recovery guarantee.** The discovered structure is an estimate. No recovery, false-discovery
  or error-control statement is made, and none is licensed: :meth:`SelectiveDiscovery.request_recovery_claim`
  and :meth:`SelectiveDiscovery.request_downstream_interval` refuse with ``cell_not_licensed``.
* **A discovered structure is not a selection diagram.**
  :meth:`SelectiveDiscovery.request_selection_diagram` refuses and names every unresolved edge
  and exhausted search bound.
* **One population.** Multi-population discovery is closed (``route_not_supported``).

Work is observed at the algorithm boundaries: CI tests handed to the test (attempted and
completed), requests answered from a retained statistic (reused), orientation rounds and applied
orientations, and search passes. Score-based search is not delivered: ``score_evaluations`` is
always zero.

A run's state exports to a checksummed artifact (:meth:`SelectiveDiscovery.export_state`) holding
the contract identity, statistic receipts and graph identities, never raw data. Resuming
(:meth:`SelectiveDiscovery.resume`) needs the raw data and the same contract: the algorithm is
replayed over the receipts only and must reproduce the recorded result. Refusals are
:class:`SelectiveDiscoveryRefusal` (a :class:`~antecedent.errors.CausalUnsupportedError`) with a
registered ``reason_code``, a ``selective_discovery.*`` ``detail`` and a ``remedy``. A failure of
the discovery algorithm itself raises :class:`~antecedent.errors.CausalDiscoveryError`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "ALGORITHMS",
    "Edge",
    "ExportedState",
    "Fdr",
    "Link",
    "Node",
    "Outcome",
    "PlanEntry",
    "RunReport",
    "SelectiveDiscovery",
    "SelectiveDiscoveryRefusal",
    "Significance",
    "StaticContract",
    "TemporalContract",
    "WorkUnits",
    "consume_selective_discovery_artifact",
    "inspect_selective_discovery_artifact",
    "replace",
    "request_score_search_reuse",
]

#: The algorithms of each kind.
ALGORITHMS: Mapping[str, tuple[str, ...]] = {
    "static": ("pc", "fci"),
    "temporal": ("pcmci", "pcmci_plus"),
}

_DEFAULT: Any = object()
_METHODS = ("bh", "by", "bonferroni", "holm")


class SelectiveDiscoveryRefusal(StructuredRefusal):
    """A typed refusal of selective discovery.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``invalid_argument``, ``route_not_supported``, ``cell_not_licensed``,
    ``discovery_statistics_invalid``, ``discovered_structure_unresolved``) and a
    ``selective_discovery.*`` ``detail``; ``remedy`` names what would license the request.
    """


def _raise(payload: str | None) -> None:
    raise_refusal(payload, SelectiveDiscoveryRefusal)


# --------------------------------------------------------------------------- declarations


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalValueError(f"{what} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class Link:
    """A directed lagged link ``source[t - source_lag] -> target[t - target_lag]`` by name."""

    source: str
    target: str
    source_lag: int = 0
    target_lag: int = 0


@dataclass(frozen=True, slots=True)
class Fdr:
    """A multiple-testing adjustment: ``method`` is ``bh``, ``by``, ``bonferroni`` or ``holm``."""

    method: str = "bh"
    exclude_contemporaneous: bool = False

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise CausalValueError(f"method must be one of {_METHODS}, got {self.method!r}")

    def to_wire(self) -> dict[str, Any]:
        return {"method": self.method, "exclude_contemporaneous": self.exclude_contemporaneous}


@dataclass(frozen=True, slots=True)
class Significance:
    """How CI p-values are obtained: analytic, or a block-shuffle null."""

    kind: str = "analytic"
    replicates: int = 0
    block_size: int = 1

    @classmethod
    def block_shuffle(cls, replicates: int, block_size: int = 1) -> Significance:
        """A permutation null with ``replicates`` shuffles of blocks of ``block_size`` rows."""
        return cls(
            "block_shuffle", _count(replicates, "replicates"), _count(block_size, "block_size")
        )

    def to_wire(self) -> dict[str, Any]:
        if self.kind == "analytic":
            return {"kind": "analytic"}
        return {"kind": self.kind, "replicates": self.replicates, "block_size": self.block_size}


def _common_wire(contract: Any, kind: str) -> dict[str, Any]:
    names = tuple(_name(v, "variable") for v in contract.variables)
    if len(set(names)) != len(names) or not names:
        raise CausalValueError("variables must be a non-empty sequence of distinct names")
    index = {name: i for i, name in enumerate(names)}

    def at(name: str) -> int:
        try:
            return index[name]
        except KeyError:
            raise CausalValueError(f"{name!r} is not one of the contract's variables") from None

    def link(item: Link) -> list[int]:
        return [
            at(item.source),
            _count(item.source_lag, "lag"),
            at(item.target),
            _count(item.target_lag, "lag"),
        ]

    wire: dict[str, Any] = {
        "kind": kind,
        "algorithm": contract.algorithm,
        "ci_test": _name(contract.ci_test, "ci_test"),
        "seed": _count(contract.seed, "seed"),
        "population": _name(contract.population, "population"),
        "additional_populations": [_name(p, "population") for p in contract.additional_populations],
        "alpha": float(contract.alpha),
        "significance": contract.significance.to_wire(),
        "mask_type": contract.mask_type,
        "required": [link(item) for item in contract.required],
        "forbidden": [link(item) for item in contract.forbidden],
        "tiers": [[at(v) for v in tier] for tier in contract.tiers],
    }
    if contract.max_cond_size is not None:
        wire["max_cond_size"] = _count(contract.max_cond_size, "max_cond_size")
    if contract.max_parents is not None:
        wire["max_parents"] = _count(contract.max_parents, "max_parents")
    if contract.fdr is not _DEFAULT:
        wire["fdr"] = None if contract.fdr is None else contract.fdr.to_wire()
    return wire


@dataclass(frozen=True, slots=True)
class StaticContract:
    """The frozen contract of a static (tabular) discovery run.

    Everything about a run except the data: the searched variables in search order, the algorithm
    (``pc`` or ``fci``), the CI test, the threshold ``alpha`` and multiplicity control ``fdr``
    (omit for the algorithm's default, ``None`` for none), the conditioning-set limit, background
    knowledge (``required``, ``forbidden``, ``tiers``), the Possible-D-Sep and discriminating-path
    budgets (FCI), the single declared population and the random seed.
    """

    variables: tuple[str, ...]
    algorithm: str = "pc"
    ci_test: str = "parcorr"
    alpha: float = 0.05
    max_cond_size: int | None = None
    max_parents: int | None = None
    fdr: Fdr | None = _DEFAULT
    significance: Significance = field(default_factory=Significance)
    mask_type: str = "yxz"
    required: tuple[Link, ...] = ()
    forbidden: tuple[Link, ...] = ()
    tiers: tuple[tuple[str, ...], ...] = ()
    possible_d_sep_max_nodes: int | None = None
    discriminating_path_max_paths: int | None = None
    discriminating_path_max_len: int | None = None
    population: str = "population"
    additional_populations: tuple[str, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS["static"]:
            raise CausalValueError(f"algorithm must be one of {ALGORITHMS['static']}")

    def to_wire(self) -> dict[str, Any]:
        wire = _common_wire(self, "static")
        if self.possible_d_sep_max_nodes is not None:
            wire["possible_d_sep_max_nodes"] = _count(self.possible_d_sep_max_nodes, "budget")
        budget: dict[str, int] = {}
        if self.discriminating_path_max_paths is not None:
            budget["max_paths_per_edge"] = _count(self.discriminating_path_max_paths, "budget")
        if self.discriminating_path_max_len is not None:
            budget["max_len"] = _count(self.discriminating_path_max_len, "budget")
        if budget:
            wire["discriminating_paths"] = budget
        return wire


@dataclass(frozen=True, slots=True)
class TemporalContract:
    """The frozen contract of a temporal (lagged) discovery run.

    As :class:`StaticContract`, with the algorithm ``pcmci`` (lagged links only) or ``pcmci_plus``
    (contemporaneous links oriented up to an equivalence class) and the lag window
    ``min_lag``..``max_lag`` (omit for the algorithm's default). The maximum lag sets the lagged
    frame; changing it retains no statistic.
    """

    variables: tuple[str, ...]
    algorithm: str = "pcmci"
    ci_test: str = "parcorr"
    alpha: float = 0.05
    max_cond_size: int | None = None
    max_parents: int | None = None
    min_lag: int | None = None
    max_lag: int | None = None
    fdr: Fdr | None = _DEFAULT
    significance: Significance = field(default_factory=Significance)
    mask_type: str = "yxz"
    required: tuple[Link, ...] = ()
    forbidden: tuple[Link, ...] = ()
    tiers: tuple[tuple[str, ...], ...] = ()
    population: str = "population"
    additional_populations: tuple[str, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS["temporal"]:
            raise CausalValueError(f"algorithm must be one of {ALGORITHMS['temporal']}")

    def to_wire(self) -> dict[str, Any]:
        wire = _common_wire(self, "temporal")
        if self.min_lag is not None:
            wire["min_lag"] = _count(self.min_lag, "min_lag")
        if self.max_lag is not None:
            wire["max_lag"] = _count(self.max_lag, "max_lag")
        return wire


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class Node:
    """A node of a discovered structure: a variable, at a lag for temporal structures."""

    variable: str
    lag: int | None

    def __str__(self) -> str:
        return self.variable if self.lag is None else f"{self.variable}[t-{self.lag}]"


@dataclass(frozen=True, slots=True)
class Edge:
    """A marked edge between two nodes; ``unresolved_reason`` is ``None`` for a directed edge.

    ``at_a`` and ``at_b`` are the marks (``tail``, ``arrow``, ``circle``, ``conflict``) at ``a``
    and ``b``.
    """

    a: Node
    b: Node
    at_a: str
    at_b: str
    middle: int
    unresolved_reason: str | None

    @property
    def is_directed(self) -> bool:
        """Whether the edge is ``tail -> arrow`` in one direction."""
        return (self.at_a, self.at_b) in (("tail", "arrow"), ("arrow", "tail"))

    @property
    def directed(self) -> tuple[Node, Node] | None:
        """``(parent, child)`` when the edge is directed, else ``None``."""
        if (self.at_a, self.at_b) == ("tail", "arrow"):
            return (self.a, self.b)
        if (self.at_a, self.at_b) == ("arrow", "tail"):
            return (self.b, self.a)
        return None


@dataclass(frozen=True, slots=True)
class WorkUnits:
    """Work observed at algorithm boundaries; a retained statistic is not work."""

    ci_tests_attempted: int
    ci_tests_completed: int
    ci_tests_reused: int
    ci_tests_consumed: int
    search_passes: int
    orientation_runs: int
    orientation_rounds: int
    orientations_applied: int
    score_evaluations: int
    algorithm_executions: int

    @property
    def is_pure_reuse(self) -> bool:
        """Whether nothing was computed and the algorithm was not entered."""
        return self.ci_tests_attempted == 0 and self.algorithm_executions == 0


@dataclass(frozen=True, slots=True)
class PlanEntry:
    """One stage's reuse decision: ``reused`` or ``recomputed`` because of the named inputs."""

    stage: str
    status: str
    because: tuple[str, ...]
    identity: str


@dataclass(frozen=True, slots=True)
class RunReport:
    """What one request did: the stage plan, the observed work and what was retained."""

    stages: tuple[PlanEntry, ...]
    work: WorkUnits
    outcome_changed: bool
    retained_statistics: int

    def stage(self, name: str) -> PlanEntry:
        """The entry of stage ``name`` (statistics, search, orientation or graph_output)."""
        for entry in self.stages:
            if entry.stage == name:
                return entry
        raise CausalValueError(f"no stage {name!r}")

    @property
    def pure_reuse(self) -> bool:
        """Whether every stage was reused."""
        return all(e.status == "reused" for e in self.stages)


@dataclass(frozen=True, slots=True)
class Outcome:
    """A discovered structure with its unresolved edges, evidence and bounded-search receipts."""

    structure_class: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    unresolved_edges: tuple[Edge, ...]
    evidence: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[tuple[str, str], ...]
    bounded_search_receipts: tuple[tuple[str, str], ...]
    search_incomplete: bool
    orientation_conflict: bool
    structure_identity: str
    result_identity: str

    @property
    def directed_edges(self) -> tuple[tuple[Node, Node], ...]:
        """Every directed edge as ``(parent, child)``, in canonical order."""
        return tuple(d for e in self.edges if (d := e.directed) is not None)

    @property
    def fully_directed(self) -> bool:
        """Whether every edge is directed. A discovered estimate even then."""
        return all(e.is_directed for e in self.edges)


@dataclass(frozen=True, slots=True)
class ExportedState:
    """An exported run: artifact bytes, their identity and a summary (no raw data)."""

    artifact: bytes
    identity: str
    summary: Mapping[str, Any]


def _node(wire: Mapping[str, Any], names: Sequence[str]) -> Node:
    variable = names[int(wire["variable"])]
    kind = int(wire["kind"])
    if kind == 1:
        return Node(variable, int(wire["aux"]))
    if kind == 2:
        return Node(variable, -int(wire["aux"]))
    return Node(variable, None)


def _edge(wire: Mapping[str, Any], names: Sequence[str]) -> Edge:
    return Edge(
        _node(wire["a"], names),
        _node(wire["b"], names),
        str(wire["at_a"]),
        str(wire["at_b"]),
        int(wire.get("middle", 4)),
        wire.get("unresolved_reason", wire.get("reason")),
    )


def _outcome(text: str, names: Sequence[str]) -> Outcome:
    wire = json.loads(text)
    structure = wire["structure"]
    evidence = []
    for item in wire["evidence"]:
        entry = dict(item)
        entry["source"] = names[int(item["source"])]
        entry["target"] = names[int(item["target"])]
        evidence.append(entry)
    return Outcome(
        structure_class=str(structure["class"]),
        nodes=tuple(_node(n, names) for n in structure["nodes"]),
        edges=tuple(_edge(e, names) for e in structure["edges"]),
        unresolved_edges=tuple(_edge(e, names) for e in wire["unresolved_edges"]),
        evidence=tuple(evidence),
        diagnostics=tuple((str(c), str(m)) for c, m in wire["diagnostics"]),
        bounded_search_receipts=tuple((str(c), str(m)) for c, m in wire["bounded_search_receipts"]),
        search_incomplete=bool(wire["search_incomplete"]),
        orientation_conflict=bool(wire["orientation_conflict"]),
        structure_identity=str(wire["structure_identity"]),
        result_identity=str(wire["result_identity"]),
    )


def _report(text: str) -> RunReport:
    wire = json.loads(text)
    return RunReport(
        stages=tuple(
            PlanEntry(e["stage"], e["status"], tuple(e["because"]), e["identity"])
            for e in wire["plan"]["stages"]
        ),
        work=WorkUnits(**wire["work"]),
        outcome_changed=bool(wire["outcome_changed"]),
        retained_statistics=int(wire["retained_statistics"]),
    )


def _columns(contract: Any, data: Mapping[str, Any]) -> list[np.ndarray]:
    if not isinstance(data, Mapping):
        raise CausalTypeError("data must map variable names to 1-D arrays")
    out = []
    for name in contract.variables:
        if name not in data:
            raise CausalValueError(f"data has no column {name!r}")
        column = np.ascontiguousarray(np.asarray(data[name], dtype=np.float64))
        if column.ndim != 1:
            raise CausalValueError(f"column {name!r} must be 1-D")
        out.append(column)
    return out


def _kind_of(contract: object) -> str:
    if isinstance(contract, StaticContract):
        return "static"
    if isinstance(contract, TemporalContract):
        return "temporal"
    raise CausalTypeError("contract must be a StaticContract or a TemporalContract")


# --------------------------------------------------------------------------- the adapter


class SelectiveDiscovery:
    """A selective-discovery adapter: retains the last successful run and reuses its statistics.

    ``kind`` is ``"static"`` (PC, FCI over a table) or ``"temporal"`` (PCMCI, PCMCI+ over a
    series). A refusal, an algorithm failure or a cancellation leaves the retained state, outcome
    and identities exactly as they were.
    """

    def __init__(self, kind: str) -> None:
        if kind not in ALGORITHMS:
            raise CausalValueError(f"kind must be one of {tuple(ALGORITHMS)}, got {kind!r}")
        self._kind = kind
        self._handle: Any = _native.SelectiveDiscoveryHandle(kind)
        self._names: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        """``"static"`` or ``"temporal"``."""
        return self._kind

    @property
    def retained_statistics(self) -> int:
        """Statistics currently retained."""
        return int(self._handle.retained_statistics())

    @property
    def outcome(self) -> Outcome | None:
        """The retained outcome, or ``None`` before the first run."""
        text = self._handle.outcome_json()
        return None if text is None else _outcome(text, self._names)

    def _call(self, which: str, contract: Any, data: Mapping[str, Any], **options: Any) -> str:
        if _kind_of(contract) != self._kind:
            raise CausalTypeError(f"a {self._kind} adapter needs a {self._kind} contract")
        wire = json.dumps(contract.to_wire(), sort_keys=True)
        columns = _columns(contract, data)
        text, refusal = getattr(self._handle, which)(
            wire, list(contract.variables), columns, **options
        )
        _raise(refusal)
        self._names = tuple(contract.variables)
        assert text is not None
        return str(text)

    def run(
        self,
        contract: StaticContract | TemporalContract,
        data: Mapping[str, Any],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> RunReport:
        """Discover on ``data`` under ``contract``, reusing retained statistics where it may.

        ``data`` maps every searched variable to a 1-D float array. The context's seed is the
        contract's seed.

        Raises:
            SelectiveDiscoveryRefusal: a refused contract (``multi_population``,
                ``contemporaneous_links``, ``invalid_contract``).
            CausalDiscoveryError: the algorithm failed or was cancelled; retained state is
                unchanged.
        """
        return _report(self._call("run", contract, data, threads=threads, cancel=cancel))

    def plan(
        self,
        contract: StaticContract | TemporalContract,
        data: Mapping[str, Any],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> tuple[PlanEntry, ...]:
        """What :meth:`run` would reuse, without executing anything."""
        wire = json.loads(self._call("plan", contract, data, threads=threads, cancel=cancel))
        return tuple(
            PlanEntry(e["stage"], e["status"], tuple(e["because"]), e["identity"])
            for e in wire["stages"]
        )

    def export_state(self, artifact_id: str = "selective_discovery") -> ExportedState:
        """Export the retained run as a ``selective_discovery_v1`` artifact (no raw data).

        Raises:
            SelectiveDiscoveryRefusal: ``invalid_argument`` when nothing is retained,
                ``discovery_statistics_invalid`` for a state over the portable bound.
        """
        _name(artifact_id, "artifact_id")
        made, refusal = self._handle.export(artifact_id)
        _raise(refusal)
        if made is None:
            raise SelectiveDiscoveryRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "selective_discovery",
                    "detail": "selective_discovery.nothing_retained",
                    "message": "no discovery has run",
                    "remedy": "run discovery before exporting its state",
                }
            )
        summary, artifact, identity = made
        return ExportedState(bytes(artifact), str(identity), json.loads(summary))

    @classmethod
    def resume(
        cls,
        contract: StaticContract | TemporalContract,
        data: Mapping[str, Any] | None,
        artifact: bytes,
        expected_identity: str | None = None,
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> tuple[SelectiveDiscovery, RunReport]:
        """Resume from an exported artifact: replay the algorithm over its receipts only.

        ``data`` is the raw data the statistics were computed from; ``None`` is refused. The
        replayed result must have the identity the artifact recorded.

        Raises:
            SelectiveDiscoveryRefusal: ``discovery_statistics_invalid`` for missing or different
                data, another contract, an unusable receipt or a replay that diverges;
                ``route_not_supported`` for a retained-identity mismatch.
        """
        kind = _kind_of(contract)
        if not isinstance(artifact, (bytes, bytearray, memoryview)):
            raise CausalTypeError("artifact must be bytes")
        columns = None if data is None else _columns(contract, data)
        handle, report, refusal = _native.selective_discovery_resume(
            kind,
            json.dumps(contract.to_wire(), sort_keys=True),
            list(contract.variables),
            columns,
            bytes(artifact),
            expected_identity,
            threads=threads,
            cancel=cancel,
        )
        _raise(refusal)
        assert handle is not None and report is not None
        adapter = cls.__new__(cls)
        adapter._kind = kind
        adapter._handle = handle
        adapter._names = tuple(contract.variables)
        return adapter, _report(report)

    def request_recovery_claim(self) -> None:
        """Ask for a recovery or error-control statement about the retained structure.

        Always raises :class:`SelectiveDiscoveryRefusal` (``cell_not_licensed``): matching a full
        run is an algorithmic fact, not a statement about the true graph.
        """
        _raise(self._handle.request_recovery_claim())

    def request_downstream_interval(self) -> None:
        """Ask for an interval on a downstream estimate resting on the retained structure.

        Always raises :class:`SelectiveDiscoveryRefusal` (``cell_not_licensed``).
        """
        _raise(self._handle.request_downstream_interval())

    def request_selection_diagram(self, targets: Sequence[str]) -> None:
        """Ask for a selection diagram built from the retained structure.

        Always raises :class:`SelectiveDiscoveryRefusal` with ``discovered_structure_unresolved``;
        the refusal's ``unresolved_edges`` and ``bounded_search_receipts`` say what is undecided.
        """
        ids = []
        for name in targets:
            if name not in self._names:
                raise CausalValueError(f"{name!r} is not a searched variable")
            ids.append(self._names.index(name))
        payload = self._handle.request_selection_diagram(ids)
        wire = json.loads(payload)
        refusal = SelectiveDiscoveryRefusal(wire)
        refusal.unresolved_edges = tuple(wire.get("unresolved_edges", ()))  # type: ignore[attr-defined]
        refusal.bounded_search_receipts = tuple(  # type: ignore[attr-defined]
            wire.get("bounded_search_receipts", ())
        )
        raise refusal


# --------------------------------------------------------------------------- artifacts


def inspect_selective_discovery_artifact(artifact: bytes) -> Mapping[str, Any]:
    """The stored metadata of an artifact; needs no raw data and replays nothing.

    Raises:
        SelectiveDiscoveryRefusal: for corrupt or foreign bytes.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    text, refusal = _native.selective_discovery_inspect(bytes(artifact))
    _raise(refusal)
    assert text is not None
    return dict(json.loads(text))


def consume_selective_discovery_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> Mapping[str, Any]:
    """Validate an artifact (every identity recomputed) and return its identity and summary.

    ``expected_identity`` is an identity the consumer retained independently; a consistently
    resealed artifact of another run is refused.

    Raises:
        SelectiveDiscoveryRefusal: ``discovery_statistics_invalid`` for an unusable receipt or an
            inconsistent identity, ``invalid_argument`` for corrupt bytes.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    text, refusal = _native.selective_discovery_consume(bytes(artifact), expected_identity)
    _raise(refusal)
    assert text is not None
    return dict(json.loads(text))


def request_score_search_reuse() -> None:
    """Ask to reuse local scores of a score-based search: always refused (``route_not_supported``)."""
    _raise(_native.selective_discovery_score_search_refusal())
