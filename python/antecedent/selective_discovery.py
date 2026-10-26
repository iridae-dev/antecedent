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
* **No per-dataset recovery guarantee.** The discovered structure is an estimate. Whether it is
  the true graph rests on faithfulness and causal sufficiency, which cannot be tested from the
  table discovery ran on, so no recovery, false-discovery or error-control statement is made
  about one dataset, and none can be licensed: :meth:`SelectiveDiscovery.request_recovery_claim`
  refuses with ``cell_not_licensed``. How often the method recovers the graph over a family of
  generating processes is a measured table in the documentation, not a license.
* **A reused row cannot carry an interval.**
  :meth:`SelectiveDiscovery.request_downstream_interval` refuses with ``cell_not_licensed``;
  the interval that can be licensed is the one :func:`sample_split_effect` computes on rows
  discovery never saw.
* **A discovered structure is not a selection diagram.**
  :meth:`SelectiveDiscovery.request_selection_diagram` refuses and names every unresolved edge
  and exhausted search bound.
* **One population per run.** A pooled or context-variable search is refused
  (``route_not_supported``); several populations are discovered one at a time with
  :class:`PerPopulationDiscovery`, nothing pooled and no invariance claimed.
* **Discovery then analysis.** :func:`discover_on_split` partitions exchangeable rows into a
  discovery part and an analysis part by a declared rule, discovers on the first only and binds
  the second to the discovery through a receipt. The binding licenses a statement conditional on
  the selected graph. :func:`sample_split_effect` runs the one construction whose whole-method
  coverage of the true effect is measured (a seeded half split, PC with the partial-correlation
  test, a resolved structure, parent adjustment on the analysis half) and its
  :meth:`SplitEffect.measure` attests the interval from the coverage records.
* **Effect after discovery on a treatment parent set.** :func:`sample_split_parent_set_effect`
  accepts the discovered class when every edge incident to the treatment is oriented, whatever
  the state of the unrelated edges, and reports the total effect of a linear study or the
  intervention mean ratio of a log-linear study. The analysis scale is declared before the data
  are touched; undirected edges elsewhere are listed on the result and never completed; an
  undirected edge at the treatment is refused by name.

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
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import encode_identifier
from ._freeze import freeze
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import count as _shared_count
from .discovery import BackgroundKnowledge, Link
from .errors import CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact
    from .composition_verification import CompositionGraph
    from .evidence_ledger import SourceDeclaration

__all__ = [
    "ALGORITHMS",
    "AnalysisBinding",
    "DiscoveryClaimKeys",
    "DiscoveryClaims",
    "PerPopulationClaims",
    "PerPopulationDiscovery",
    "PerPopulationReport",
    "PopulationEdge",
    "PopulationResult",
    "SampleSplit",
    "ParentSetEffect",
    "SplitEffect",
    "SplitReceipt",
    "discover_on_split",
    "sample_split_effect",
    "sample_split_parent_set_effect",
    "Edge",
    "ExportedState",
    "ConsumedSelectiveDiscovery",
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
    "discovered_graph_source",
    "discovery_data_source",
    "discovery_rows_source",
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


def _count(value: object, what: str, **kwargs: Any) -> int:
    kwargs.setdefault("minimum", 0)
    kwargs.setdefault("maximum", 2**64 - 1)
    return _shared_count(value, what, **kwargs)


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
    knowledge = BackgroundKnowledge(
        required=contract.required,
        forbidden=contract.forbidden,
        tiers=contract.tiers,
        max_parents=contract.max_parents,
    )
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
        **knowledge.to_wire(names),
    }
    if contract.max_cond_size is not None:
        wire["max_cond_size"] = _count(contract.max_cond_size, "max_cond_size")
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


@dataclass(repr=False, frozen=True, slots=True)
class RunReport(ScientificResult):
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


@dataclass(repr=False, frozen=True, slots=True)
class Outcome(ScientificResult):
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


class SelectiveDiscovery(ScientificResult):
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

    @artifact_export
    def export(
        self, artifact_id: str = "selective_discovery"
    ) -> ExportedArtifact[ConsumedSelectiveDiscovery]:
        return artifact_from_receipt(self, self.export_state(artifact_id).artifact)

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
        artifact = container_bytes(artifact)
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

    def add_claims(
        self,
        graph: CompositionGraph | None = None,
        prefix: str = "discovery",
        *,
        completion: tuple[str, str] | None = None,
    ) -> DiscoveryClaims:
        """Add this run's data, statistics and structure to a composition-verification graph.

        Adds the nodes ``{prefix}.data``, ``{prefix}.statistics`` and ``{prefix}.graph`` (and
        ``{prefix}.completion`` when ``completion`` is given) to ``graph`` (an empty graph when
        ``None``) and returns the extended graph with the node keys. New evidence at an unchanged
        structure recomputes the numbers and keeps the structure; a search that exhausted a bound
        makes the structure node unsupported. Downstream claims depend on
        :attr:`DiscoveryClaimKeys.downstream_anchor` with a ``structural_derivation`` edge.

        ``completion`` is a ``(label, identity)`` pair naming the declared completion of the
        equivalence class (``identity`` is 64 hexadecimal characters).

        Raises:
            CausalValueError: discovery has not run, or ``completion`` is malformed.
            SelectiveDiscoveryRefusal: a key that already exists in ``graph``, or a graph that
                does not validate.
        """
        label, identity = (None, None) if completion is None else _completion(completion)
        text, refusal = self._handle.add_claims(
            _graph_text(graph), _name(prefix, "prefix"), label, identity
        )
        _raise(refusal)
        assert text is not None
        wire = json.loads(text)
        return DiscoveryClaims(_graph_from_wire(wire["graph"]), _claim_keys(wire["keys"]))

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
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    text, refusal = _native.selective_discovery_inspect(bytes(artifact))
    _raise(refusal)
    assert text is not None
    return ConsumedSelectiveDiscovery(freeze(json.loads(text)), bytes(artifact))


@dataclass(repr=False, frozen=True)
class ConsumedSelectiveDiscovery(ScientificResult, Mapping[str, Any]):
    """Validated retained discovery receipts; resumption remains explicit."""

    summary: Mapping[str, Any]
    _artifact_bytes: bytes

    def __getitem__(self, key: str) -> Any:
        return self.summary[key]

    def __iter__(self) -> Any:
        return iter(self.summary)

    def __len__(self) -> int:
        return len(self.summary)

    def to_dict(self) -> dict[str, Any]:
        from ._freeze import thaw

        return thaw(self.summary)


@artifact_consumer(
    proof="selective_discovery.consume_selective_discovery_artifact: independent semantic consumer checks declarations, identities and stored result",
    replayable=False,
)
def consume_selective_discovery_artifact(
    artifact: bytes, expected_identity: str | None = None
) -> ConsumedSelectiveDiscovery:
    """Validate an artifact (every identity recomputed) and return its identity and summary.

    ``expected_identity`` is an identity the consumer retained independently; a consistently
    resealed artifact of another run is refused.

    Raises:
        SelectiveDiscoveryRefusal: ``discovery_statistics_invalid`` for an unusable receipt or an
            inconsistent identity, ``invalid_argument`` for corrupt bytes.
    """
    artifact = container_bytes(artifact)
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    text, refusal = _native.selective_discovery_consume(bytes(artifact), expected_identity)
    _raise(refusal)
    assert text is not None
    return ConsumedSelectiveDiscovery(freeze(json.loads(text)), bytes(artifact))


def request_score_search_reuse() -> None:
    """Ask to reuse local scores of a score-based search: always refused (``route_not_supported``)."""
    _raise(_native.selective_discovery_score_search_refusal())


# --------------------------------------------------------------------------- per population


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """What discovery on one population's rows returned and retained.

    ``data_identity`` covers that population's searched columns, so two populations with
    content-identical rows are refused before any work. ``receipts_identity`` is ``None`` when
    the retained statistics exceed what a portable state carries.
    """

    population: str
    rows: int
    data_identity: str
    structure_identity: str
    result_identity: str
    receipts_identity: str | None
    retained_statistics: int
    search_incomplete: bool
    unresolved_edges: int
    report: RunReport


@dataclass(frozen=True, slots=True)
class PopulationEdge:
    """One edge compared across populations.

    ``marks`` maps each population to its ``(at_a, at_b)`` marks, or ``None`` when its structure
    has no such edge. A difference is a difference between two estimates; it is not a statement
    that a mechanism differs.
    """

    a: Node
    b: Node
    marks: Mapping[str, tuple[str, str] | None]
    agrees: bool


@dataclass(repr=False, frozen=True, slots=True)
class PerPopulationReport(ScientificResult):
    """Per-population structures lined up edge by edge. Descriptive only.

    No selection diagram, invariance test or error rate is derived from the comparison.
    """

    populations: tuple[PopulationResult, ...]
    edges: tuple[PopulationEdge, ...]
    identity: str

    @property
    def label(self) -> str:
        """The standing of the comparison: ``descriptive_comparison``."""
        return "descriptive_comparison"

    @property
    def differing_edges(self) -> tuple[PopulationEdge, ...]:
        """Edges present in some populations only, or marked differently."""
        return tuple(e for e in self.edges if not e.agrees)

    def population(self, name: str) -> PopulationResult:
        """The result of population ``name``."""
        for result in self.populations:
            if result.population == name:
                return result
        raise CausalValueError(f"no population {name!r}")


def _per_population_report(text: str, names: Sequence[str]) -> PerPopulationReport:
    wire = json.loads(text)
    populations = tuple(
        PopulationResult(
            population=str(p["population"]),
            rows=int(p["rows"]),
            data_identity=str(p["data_identity"]),
            structure_identity=str(p["structure_identity"]),
            result_identity=str(p["result_identity"]),
            receipts_identity=p["receipts_identity"],
            retained_statistics=int(p["retained_statistics"]),
            search_incomplete=bool(p["search_incomplete"]),
            unresolved_edges=int(p["unresolved_edges"]),
            report=_report(json.dumps(p["report"])),
        )
        for p in wire["populations"]
    )
    edges = tuple(
        PopulationEdge(
            a=_node(e["a"], names),
            b=_node(e["b"], names),
            marks={
                str(m["population"]): (
                    None
                    if m["marks"] is None
                    else (str(m["marks"]["at_a"]), str(m["marks"]["at_b"]))
                )
                for m in e["marks"]
            },
            agrees=bool(e["agrees"]),
        )
        for e in wire["edges"]
    )
    return PerPopulationReport(populations, edges, str(wire["identity"]))


class PerPopulationDiscovery(ScientificResult):
    """Discovery run separately on each population's rows, with one retained store per population.

    One declared static contract is applied to every population. Each population has its own
    statistic store, data identity and result identity; no row or statistic crosses populations.
    A refused or failed population leaves that population's retained state as it was, and the
    set-level report is replaced only by a request in which every population succeeded::

        driver = sd.PerPopulationDiscovery()
        report = driver.run(contract, {"north": north_data, "south": south_data})
        report.differing_edges          # descriptive; not a selection diagram
        driver.outcome("north")         # that population's structure
    """

    def __init__(self) -> None:
        self._handle: Any = _native.PerPopulationDiscoveryHandle()
        self._names: tuple[str, ...] = ()

    def run(
        self,
        contract: StaticContract,
        populations: Mapping[str, Mapping[str, Any]],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> PerPopulationReport:
        """Run ``contract`` on each population's columns.

        ``populations`` maps a population label to a mapping of every searched variable to a
        1-D float array. ``contract.population`` is replaced by each label.

        Raises:
            SelectiveDiscoveryRefusal: fewer than two populations, a blank label, a pooled or
                context-variable declaration (``selective_discovery.pooled_population_search``),
                content-identical populations
                (``selective_discovery.pooled_population_data``) or a refused contract.
            CausalDiscoveryError: the algorithm failed or was cancelled.
        """
        if not isinstance(contract, StaticContract):
            raise CausalTypeError("contract must be a StaticContract")
        if not isinstance(populations, Mapping):
            raise CausalTypeError("populations must map a label to a column mapping")
        labels = [_name(label, "population") for label in populations]
        columns = [_columns(contract, populations[label]) for label in labels]
        text, refusal = self._handle.run(
            json.dumps(contract.to_wire(), sort_keys=True),
            labels,
            list(contract.variables),
            columns,
            threads=threads,
            cancel=cancel,
        )
        _raise(refusal)
        self._names = tuple(contract.variables)
        assert text is not None
        return _per_population_report(str(text), self._names)

    @property
    def report(self) -> PerPopulationReport | None:
        """The last report in which every population succeeded, or ``None``."""
        text = self._handle.report_json()
        return None if text is None else _per_population_report(text, self._names)

    def outcome(self, population: str) -> Outcome | None:
        """One population's retained outcome, or ``None`` before it has run."""
        text = self._handle.outcome_json(_name(population, "population"))
        return None if text is None else _outcome(text, self._names)

    def retained_statistics(self, population: str) -> int:
        """Statistics retained for ``population``."""
        return int(self._handle.retained_statistics(_name(population, "population")))

    def add_claims(
        self, graph: CompositionGraph | None = None, prefix: str = "discovery"
    ) -> PerPopulationClaims:
        """Add every population's discovery stages of the last committed report to ``graph``.

        Each population is added under the prefix ``{prefix}.{population}`` with no completion,
        so a downstream claim anchors on the discovered structure of its own population; the
        population label is part of every node's coordinate. A driver with no committed report
        adds nothing and returns an empty key mapping.

        Raises:
            SelectiveDiscoveryRefusal: a key that already exists in ``graph``, or a graph that
                does not validate.
        """
        text, refusal = self._handle.add_claims(_graph_text(graph), _name(prefix, "prefix"))
        _raise(refusal)
        assert text is not None
        wire = json.loads(text)
        keys = {label: _claim_keys(item) for label, item in wire["keys"].items()}
        return PerPopulationClaims(_graph_from_wire(wire["graph"]), freeze(keys))

    def request_selection_targets(self, candidates: Sequence[str] = ()) -> None:
        """Ask for selection targets read from the edges that differ between populations.

        Always raises :class:`SelectiveDiscoveryRefusal`
        (``selective_discovery.differences_are_not_selection_targets``): a difference between two
        discovered structures is a difference between two estimates, and selection targets are
        declared by the caller.
        """
        ids = []
        for name in candidates:
            if name not in self._names:
                raise CausalValueError(f"{name!r} is not a searched variable")
            ids.append(self._names.index(name))
        _raise(self._handle.request_selection_targets(ids))


# --------------------------------------------------------------------------- sample split


@dataclass(frozen=True, slots=True)
class SplitReceipt:
    """The record of one row split: enough to re-derive the partition and check a table.

    Rows are filtered before they are split: first by the analysis mask (``masked_out_rows``),
    then to complete cases over the searched variables (``incomplete_rows``); both lists hold
    original row indices and those rows belong to neither part. ``weights`` is ``None`` for an
    unweighted source, else the digests of the source, discovery and analysis weights and the
    Kish effective sample size of each part's weights; ``mask_identity`` is the digest of the
    source mask when it had one.

    ``whole_method_calibration`` is always ``unmeasured`` for an analysis you run yourself on the
    analysis part. The one whole-method construction with a measured coverage record is
    :func:`sample_split_effect`, which attests its own interval.
    """

    rule: Mapping[str, Any]
    n_rows: int
    masked_out_rows: tuple[int, ...]
    incomplete_rows: tuple[int, ...]
    n_discovery: int
    n_analysis: int
    discovery_rows_identity: str
    analysis_rows_identity: str
    mask_identity: str | None
    weights: Mapping[str, Any] | None
    source_data_identity: str
    discovery_data_identity: str
    analysis_data_identity: str
    identity: str
    conditional_on_selected_graph: bool
    whole_method_calibration: str


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """What an analysis table passed the split's check under.

    The binding licenses a statement conditional on the selected graph, and no more: the
    probability that discovery returned a graph that licenses the question is not controlled.
    """

    receipt_identity: str
    structure_identity: str
    result_identity: str
    analysis_data_identity: str
    identity: str
    conditional_on_selected_graph: bool
    whole_method_calibration: str


def _weights(weights: Sequence[float] | None) -> list[float] | None:
    if weights is None:
        return None
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1:
        raise CausalValueError("weights must be a 1-D array")
    return [float(w) for w in array]


def _mask(mask: Sequence[bool] | None) -> list[bool] | None:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim != 1 or array.dtype != np.bool_:
        raise CausalValueError("mask must be a 1-D boolean array (True keeps the row)")
    return [bool(m) for m in array]


class SampleSplit(ScientificResult):
    """A row split, the discovery run on its discovery part and the receipt tying them together.

    ``discovery`` and ``analysis`` map every column of the split table to that part's values.
    Only ``analysis`` may be used by the downstream analysis; :meth:`check_analysis_table`
    refuses any other rows.

    Rows were filtered (analysis mask, then complete cases over the searched variables) before
    they were split; the excluded rows are in :attr:`receipt` and in neither part.
    :attr:`analysis_row_ids` gives the original row of each analysis row, and
    :attr:`analysis_weights` its observation weight exactly as the source held it
    (unnormalized: normalizing is the consumer's, by what the weights mean).
    :attr:`analysis_effective_sample_size` is the Kish size ``(sum w)^2 / sum w^2``, which an
    interval for a weighted analysis uses instead of the row count.
    """

    def __init__(
        self,
        handle: Any,
        names: Sequence[str],
        report: RunReport,
        discovery: Mapping[str, np.ndarray],
        analysis: Mapping[str, np.ndarray],
    ) -> None:
        self._handle = handle
        self._names = tuple(names)
        self._report = report
        self.discovery: Mapping[str, np.ndarray] = discovery
        self.analysis: Mapping[str, np.ndarray] = analysis

    @property
    def discovery_row_ids(self) -> np.ndarray:
        """Original row index of each discovery row, in the part's row order."""
        return np.asarray(self._handle.discovery_row_ids(), dtype=np.int64)

    @property
    def analysis_row_ids(self) -> np.ndarray:
        """Original row index of each analysis row, in the part's row order."""
        return np.asarray(self._handle.analysis_row_ids(), dtype=np.int64)

    def participants(
        self, data: Mapping[str, Any], column: str, *, require_disjoint: bool = True
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Original participant labels of the ``(discovery, analysis)`` rows, in part order.

        ``column`` of ``data`` (the source table that was split) holds the participant of
        every original row, strings or whole numbers; each part's labels are read at its
        original row ids. A participant with rows on both sides would let the analysis part
        reuse what discovery saw, so by default that is refused.

        Raises:
            CausalValueError: the table does not have the split's row count, a label is
                missing, or a participant appears in both parts.
        """
        binding, _ = encode_identifier(data, column, what="participant", keep_rows=True)
        if binding.rows != self.receipt.n_rows:
            raise CausalValueError(
                f"participant column {column!r} labels {binding.rows} rows; the split was made "
                f"on {self.receipt.n_rows}",
                reason_code="invalid_argument",
            )
        labels = binding.row_labels
        discovery = tuple(labels[int(row)] for row in self.discovery_row_ids)
        analysis = tuple(labels[int(row)] for row in self.analysis_row_ids)
        shared = sorted(set(discovery) & set(analysis))
        if require_disjoint and shared:
            raise CausalValueError(
                f"participants {shared[:5]} have rows in both the discovery and the analysis "
                "part; split by participant so that no participant is on both sides",
                reason_code="invalid_argument",
            )
        return discovery, analysis

    @property
    def analysis_weights(self) -> np.ndarray | None:
        """Weights of the analysis rows, unnormalized; ``None`` for an unweighted source."""
        weights = self._handle.analysis_weights()
        return None if weights is None else np.asarray(weights, dtype=np.float64)

    @property
    def analysis_effective_sample_size(self) -> float | None:
        """Kish effective sample size of the analysis weights; ``None`` when unweighted."""
        weights = self.receipt.weights
        return None if weights is None else float(weights["analysis_effective_sample_size"])

    @property
    def report(self) -> RunReport:
        """What the discovery request did (stage plan, observed work)."""
        return self._report

    @property
    def receipt(self) -> SplitReceipt:
        """The split receipt."""
        wire = json.loads(self._handle.receipt_json())
        return SplitReceipt(
            rule=freeze(wire["rule"]),
            n_rows=int(wire["n_rows"]),
            masked_out_rows=tuple(int(r) for r in wire["masked_out_rows"]),
            incomplete_rows=tuple(int(r) for r in wire["incomplete_rows"]),
            n_discovery=int(wire["n_discovery"]),
            n_analysis=int(wire["n_analysis"]),
            discovery_rows_identity=str(wire["discovery_rows_identity"]),
            analysis_rows_identity=str(wire["analysis_rows_identity"]),
            mask_identity=None if wire["mask_identity"] is None else str(wire["mask_identity"]),
            weights=None if wire["weights"] is None else freeze(wire["weights"]),
            source_data_identity=str(wire["source_data_identity"]),
            discovery_data_identity=str(wire["discovery_data_identity"]),
            analysis_data_identity=str(wire["analysis_data_identity"]),
            identity=str(wire["identity"]),
            conditional_on_selected_graph=bool(wire["conditional_on_selected_graph"]),
            whole_method_calibration=str(wire["whole_method_calibration"]),
        )

    @property
    def outcome(self) -> Outcome:
        """The structure discovery returned on the discovery part."""
        return _outcome(self._handle.outcome_json(), self._names)

    def verify(self) -> None:
        """Re-derive the partition and check the receipt against both parts.

        The partition is re-derived from the rule and the recorded exclusions; the stored
        identities, the parts' original row ids, masks, weights and content are checked against
        the receipt.

        Raises:
            SelectiveDiscoveryRefusal: ``split_not_disjoint``, ``split_receipt_mismatch``,
                ``split_rows_mismatch``, ``split_mask_mismatch`` or ``split_weights_mismatch``.
        """
        _raise(self._handle.verify())

    def verify_source(
        self,
        data: Mapping[str, Any],
        *,
        weights: Sequence[float] | None = None,
        mask: Sequence[bool] | None = None,
    ) -> None:
        """Replay the split on the source table and require it to be the one that was split.

        ``data``, ``weights`` and ``mask`` are the source table as it is now. A different mask or
        different weights are refused by name before the split is replayed.

        Raises:
            SelectiveDiscoveryRefusal: ``split_mask_mismatch``, ``split_weights_mismatch``,
                ``split_source_mismatch`` or a refusal of the replayed split.
        """
        columns = self._table_columns(data)
        _raise(
            self._handle.verify_source(list(self._names), columns, _weights(weights), _mask(mask))
        )

    def _table_columns(self, data: Mapping[str, Any]) -> list[np.ndarray]:
        if not isinstance(data, Mapping):
            raise CausalTypeError("data must map variable names to 1-D arrays")
        columns = []
        for name in self._names:
            if name not in data:
                raise CausalValueError(f"data has no column {name!r}")
            column = np.ascontiguousarray(np.asarray(data[name], dtype=np.float64))
            if column.ndim != 1:
                raise CausalValueError(f"column {name!r} must be 1-D")
            columns.append(column)
        return columns

    def evidence_sources(self, snapshot_id: bytes) -> tuple[SourceDeclaration, SourceDeclaration]:
        """The two parts as evidence-ledger sources, ``(discovery, analysis)``.

        Observation identities are keyed by ``snapshot_id`` and the row index, so the two parts
        are independent of each other and either overlaps the unsplit table (declared with
        :func:`discovery_data_source` under the same ``snapshot_id``) on exactly the rows it holds.

        Raises:
            SelectiveDiscoveryRefusal: the receipt no longer verifies.
        """
        snapshot_id = container_bytes(snapshot_id)
        text, refusal = self._handle.evidence_sources(_snapshot(snapshot_id))
        _raise(refusal)
        assert text is not None
        wire = json.loads(text)
        return _declaration(wire["discovery"]), _declaration(wire["analysis"])

    def check_analysis_table(
        self, data: Mapping[str, Any], *, weights: Sequence[float] | None = None
    ) -> AnalysisBinding:
        """Check that ``data`` (and its ``weights``) are exactly the analysis part.

        Raises:
            SelectiveDiscoveryRefusal: ``selective_discovery.analysis_rows_not_the_split_part``
                for the discovery rows, the unsplit table, re-weighted rows or any other rows.
        """
        columns = self._table_columns(data)
        text, refusal = self._handle.check_analysis_table(
            list(self._names), columns, _weights(weights)
        )
        _raise(refusal)
        assert text is not None
        wire = json.loads(text)
        return AnalysisBinding(
            receipt_identity=str(wire["receipt_identity"]),
            structure_identity=str(wire["structure_identity"]),
            result_identity=str(wire["result_identity"]),
            analysis_data_identity=str(wire["analysis_data_identity"]),
            identity=str(wire["identity"]),
            conditional_on_selected_graph=bool(wire["conditional_on_selected_graph"]),
            whole_method_calibration=str(wire["whole_method_calibration"]),
        )

    def request_unsplit_interval(self) -> None:
        """Ask for an interval on account of discovery for an analysis that reused its rows.

        Always raises :class:`SelectiveDiscoveryRefusal` (``cell_not_licensed``).
        """
        _raise(self._handle.request_unsplit_interval())


def discover_on_split(
    contract: StaticContract,
    data: Mapping[str, Any],
    *,
    discovery_fraction: float | None = None,
    seed: int | None = None,
    discovery_rows: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
    mask: Sequence[bool] | None = None,
    threads: int | None = None,
    cancel: Any = None,
) -> SampleSplit:
    """Split the rows of ``data`` and discover on the discovery part only.

    Declare the split either as ``discovery_rows`` (original row indices the discovery part
    takes; every other eligible row is an analysis row) or as ``discovery_fraction`` with
    ``seed`` (a seeded permutation of the eligible rows; the first ``round(fraction * n)`` of
    them are discovery rows). ``data`` must hold every searched variable as a 1-D float array;
    rows are assumed exchangeable.

    Rows are filtered before they are split: first by ``mask`` (a boolean array, ``True`` keeps
    the row), then to the rows with no missing (``NaN``) value in a searched variable. The
    excluded rows are recorded in the receipt and belong to neither part, and the rule only
    ever sees the rows that remain. ``weights`` (one non-negative weight per row) are carried to
    the part of their row unchanged and unnormalized; use ``ci_test="weighted_parcorr"`` so
    that discovery reads the discovery part's weights (weights under any other test, or that
    test without weights, are refused).

    Raises:
        SelectiveDiscoveryRefusal: non-numeric rows
            (``selective_discovery.split_unsupported_rows``), a rule that leaves a part empty,
            repeats, exceeds or names an excluded row, a part without positive weight, weights
            the declared test does not read, or a refused contract.
        CausalDiscoveryError: the algorithm failed or was cancelled.
    """
    if not isinstance(contract, StaticContract):
        raise CausalTypeError("contract must be a StaticContract")
    if discovery_rows is not None:
        if discovery_fraction is not None or seed is not None:
            raise CausalValueError(
                "declare discovery_rows or discovery_fraction with seed, not both"
            )
        rule: dict[str, Any] = {
            "kind": "explicit",
            "discovery_rows": [_count(r, "row") for r in discovery_rows],
        }
    else:
        if discovery_fraction is None or seed is None:
            raise CausalValueError("declare discovery_rows, or discovery_fraction with seed")
        rule = {
            "kind": "seeded_permutation",
            "seed": _count(seed, "seed"),
            "discovery_fraction": float(discovery_fraction),
        }
    columns = _columns(contract, data)
    handle, report, refusal = _native.sample_split_discover(
        json.dumps(contract.to_wire(), sort_keys=True),
        list(contract.variables),
        columns,
        json.dumps(rule, sort_keys=True),
        weights=_weights(weights),
        mask=_mask(mask),
        threads=threads,
        cancel=cancel,
    )
    _raise(refusal)
    assert handle is not None and report is not None
    names = tuple(contract.variables)
    discovery = dict(zip(names, handle.discovery_columns(), strict=True))
    analysis = dict(zip(names, handle.analysis_columns(), strict=True))
    return SampleSplit(handle, names, _report(report), discovery, analysis)


@dataclass(frozen=True, slots=True)
class SplitEffect:
    """The linear effect of a treatment on an outcome estimated after discovery on a row split.

    ``effect`` is the treatment's least-squares coefficient on the analysis rows given its
    parents in the accepted graph, and ``lower``/``upper`` its ``level`` interval. ``basis`` is
    the calibration key the interval is looked up under: ``structure`` is always
    ``discovered_sample_split``, never the key of a declared graph. The interval is a computed
    value until :meth:`measure` attests it; the attestation is a statement about the whole
    method over the measured family of generating processes, never about this dataset's graph.
    """

    effect: float
    lower: float
    upper: float
    level: float
    standard_error: float
    analysis_rows: int
    discovery_rows: int
    adjustment_set: tuple[str, ...]
    receipt_identity: str
    structure_identity: str
    basis: Mapping[str, Any]
    _source: bytes = field(repr=False)
    _seed: int = field(repr=False)

    def measure(self, *, threads: int | None = None, cancel: Any = None) -> Any:
        """Replay the source and attest the interval from the current coverage records.

        Returns:
            A :class:`antecedent.MeasuredInference` carrying the effect, its interval and the
            governing record.

        Raises:
            SelectiveDiscoveryRefusal: ``cell_not_licensed`` when no current attesting,
                non-boundary record measured this construction at this analysis row count.
        """
        from ._measured_inference import MeasuredInference

        native, refusal = _native.sample_split_effect_measure(
            self._source, self._seed, threads=threads, cancel=cancel
        )
        _raise(refusal)
        assert native is not None
        return MeasuredInference._from_native(native)


def _effect_columns(
    data: Mapping[str, Any], treatment: str, outcome: str
) -> tuple[list[str], list[np.ndarray]]:
    """Variable names and contiguous float columns of ``data``, with the roles checked."""
    if not isinstance(data, Mapping):
        raise CausalTypeError("data must map variable names to 1-D arrays")
    names = [_name(name, "variable") for name in data]
    for role, label in ((treatment, "treatment"), (outcome, "outcome")):
        if role not in data:
            raise CausalValueError(f"{label} {role!r} is not a column of data")
    columns = []
    for name in names:
        column = np.ascontiguousarray(np.asarray(data[name], dtype=np.float64))
        if column.ndim != 1:
            raise CausalValueError(f"column {name!r} must be 1-D")
        columns.append(column)
    return names, columns


def sample_split_effect(
    data: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    split_seed: int,
    seed: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> SplitEffect:
    """Discover on half of the rows and estimate the treatment's effect on the other half.

    ``data`` maps five to seven searched variables to finite 1-D float arrays of exchangeable
    rows. The rows are split by a seeded permutation (half to discovery), PC with the
    partial-correlation test (``alpha = 0.01``) runs on the discovery half, the structure is
    accepted only when the search finished and it is one fully directed graph, and the effect is
    estimated on the analysis half adjusting for the treatment's parents, with a 0.95
    least-squares interval.

    Raises:
        SelectiveDiscoveryRefusal: a table outside five to seven variables
            (``split_effect.variable_count_not_measured``), an unresolved structure
            (``split_effect.structure_unresolved``), an unfinished search, or an outcome oriented
            into the treatment.
        CausalDiscoveryError: the discovery algorithm failed or was cancelled.
    """
    names, columns = _effect_columns(data, treatment, outcome)
    text, source, refusal = _native.sample_split_effect(
        names,
        columns,
        names.index(treatment),
        names.index(outcome),
        _count(split_seed, "split_seed"),
        _count(seed, "seed"),
        threads=threads,
        cancel=cancel,
    )
    _raise(refusal)
    assert text is not None and source is not None
    wire = json.loads(text)
    return SplitEffect(
        effect=float(wire["effect"]),
        lower=float(wire["lower"]),
        upper=float(wire["upper"]),
        level=float(wire["level"]),
        standard_error=float(wire["standard_error"]),
        analysis_rows=int(wire["analysis_rows"]),
        discovery_rows=int(wire["discovery_rows"]),
        adjustment_set=tuple(names[int(i)] for i in wire["adjustment_set"]),
        receipt_identity=str(wire["receipt_identity"]),
        structure_identity=str(wire["structure_identity"]),
        basis=freeze(wire["basis"]),
        _source=bytes(source),
        _seed=_count(seed, "seed"),
    )


@dataclass(frozen=True, slots=True)
class ParentSetEffect:
    """The effect of a treatment on an outcome after discovery on a row split, accepted on the
    treatment's parent set.

    ``effect`` with ``lower``/``upper`` at ``level`` is the total effect per unit of the treatment
    (``scalar == "total_effect"``, a linear study) or the intervention mean ratio of multiplying
    the treatment by ``exp(log_shift)`` (``scalar == "intervention_mean_ratio"``, a log-linear
    study; ``standard_error`` is then the delta-method image of the log-scale coefficient's).
    ``coefficient`` is the treatment's least-squares coefficient on the declared analysis scale.

    The graph's uncertainty is part of the result. ``treatment_parents`` is the adjustment set and
    ``treatment_children`` the treatment's directed children; ``unresolved_edges`` lists, by
    variable name, every edge elsewhere in the class that is not a definite directed edge, as
    ``(a, b, reason)``. They are reported, never completed. ``acceptance`` records why the class
    was accepted: every edge incident to the treatment is oriented. ``basis`` is the calibration
    key the interval is looked up under. The interval is a computed value until :meth:`measure`
    attests it; the attestation is a statement about the whole method over the measured family,
    never about this dataset's graph.
    """

    scalar: str
    effect: float
    lower: float
    upper: float
    level: float
    standard_error: float
    coefficient: float
    transformation: str
    analysis_rows: int
    discovery_rows: int
    treatment_parents: tuple[str, ...]
    treatment_children: tuple[str, ...]
    unresolved_edges: tuple[tuple[str, str, str], ...]
    edge_count: int
    acceptance: str
    receipt_identity: str
    structure_identity: str
    basis: Mapping[str, Any]
    _source: bytes = field(repr=False)
    _seed: int = field(repr=False)

    def measure(self, *, threads: int | None = None, cancel: Any = None) -> Any:
        """Replay the source and attest the interval from the current coverage records.

        Returns:
            A :class:`antecedent.MeasuredInference` carrying the effect, its interval, the
            acceptance reason, the unresolved edges and the governing record.

        Raises:
            SelectiveDiscoveryRefusal: ``cell_not_licensed`` when no current attesting,
                non-boundary record measured this construction at this variable count, degree
                and analysis row count.
        """
        from ._measured_inference import MeasuredInference

        native, refusal = _native.sample_split_parent_set_effect_measure(
            self._source, self._seed, threads=threads, cancel=cancel
        )
        _raise(refusal)
        assert native is not None
        return MeasuredInference._from_native(native)


def sample_split_parent_set_effect(
    data: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    transformation: str,
    split_seed: int,
    max_degree: int = 4,
    log_shift: float | None = None,
    required: Sequence[tuple[str, str]] = (),
    forbidden: Sequence[tuple[str, str]] = (),
    tiers: Sequence[Sequence[str]] = (),
    seed: int = 0,
    threads: int | None = None,
    cancel: Any = None,
) -> ParentSetEffect:
    """Discover on half of the rows and estimate the treatment's effect on the other half,
    accepting the class when the treatment's parent set is unambiguous.

    ``data`` maps ten to twenty searched variables to finite 1-D float arrays of exchangeable
    rows. ``transformation`` is declared before the data are touched: ``"linear"``, or
    ``"log_linear"``, which requires every column to be positive and analyses its natural
    logarithm (``log_shift`` is then the log-scale size of the prespecified intervention, default
    one, so the treatment is multiplied by ``e`` and the reported ratio is ``exp(beta)``; a
    linear study takes none). ``max_degree`` (at most four) is the declared maximum graph degree
    and sets PC's conditioning-set size and parent limit.

    Background knowledge handed to discovery is optional and part of the calibration key:
    ``required`` and ``forbidden`` are ``(cause, effect)`` name pairs (an edge that must, or must
    not, run in that direction) and ``tiers`` lists of names (a later tier never causes an
    earlier one). The statements are the caller's assertions about the process; discovery
    honours them and does not test them.

    The rows are split by a seeded permutation (half to discovery), PC with the partial
    correlation test (``alpha = 0.01``) runs on the discovery half, the class is accepted when
    the search finished and every edge incident to the treatment is oriented, and the effect is
    estimated on the analysis half adjusting for the treatment's parents.

    Raises:
        SelectiveDiscoveryRefusal: an undeclared transformation
            (``parent_set_effect.transformation_undeclared``, ``invalid_argument``), a
            non-positive log-linear value (``parent_set_effect.log_linear_input_not_positive``,
            naming the column), a degree above four
            (``parent_set_effect.degree_above_supported``), a table outside ten to twenty
            variables, an unfinished search, an undirected edge at the treatment
            (``parent_set_effect.treatment_parents_ambiguous``, naming the edges), or an outcome
            oriented into the treatment.
        CausalDiscoveryError: the discovery algorithm failed or was cancelled.
    """
    if not isinstance(transformation, str):
        raise CausalTypeError("transformation must be declared as 'linear' or 'log_linear'")
    names, columns = _effect_columns(data, treatment, outcome)
    if transformation == "log_linear" and log_shift is None:
        log_shift = 1.0
    shift = None if log_shift is None else float(log_shift)

    def position(name: str) -> int:
        if name not in data:
            raise CausalValueError(f"knowledge names {name!r}, which is not a column of data")
        return names.index(name)

    knowledge = None
    if required or forbidden or tiers:
        knowledge = json.dumps(
            {
                "required": [[position(a), position(b)] for a, b in required],
                "forbidden": [[position(a), position(b)] for a, b in forbidden],
                "tiers": [[position(v) for v in tier] for tier in tiers],
            }
        )
    text, source, refusal = _native.sample_split_parent_set_effect(
        names,
        columns,
        names.index(treatment),
        names.index(outcome),
        _count(split_seed, "split_seed"),
        _count(seed, "seed"),
        transformation,
        _count(max_degree, "max_degree"),
        shift,
        knowledge,
        threads=threads,
        cancel=cancel,
    )
    _raise(refusal)
    assert text is not None and source is not None
    wire = json.loads(text)
    return ParentSetEffect(
        scalar=str(wire["scalar"]),
        effect=float(wire["effect"]),
        lower=float(wire["lower"]),
        upper=float(wire["upper"]),
        level=float(wire["level"]),
        standard_error=float(wire["standard_error"]),
        coefficient=float(wire["coefficient"]),
        transformation=str(wire["transformation"]),
        analysis_rows=int(wire["analysis_rows"]),
        discovery_rows=int(wire["discovery_rows"]),
        treatment_parents=tuple(names[int(i)] for i in wire["treatment_parents"]),
        treatment_children=tuple(names[int(i)] for i in wire["treatment_children"]),
        unresolved_edges=tuple(
            (names[int(e["a"])], names[int(e["b"])], str(e["reason"]))
            for e in wire["unresolved_edges"]
        ),
        edge_count=int(wire["edge_count"]),
        acceptance=str(wire["acceptance"]),
        receipt_identity=str(wire["receipt_identity"]),
        structure_identity=str(wire["structure_identity"]),
        basis=freeze(wire["basis"]),
        _source=bytes(source),
        _seed=_count(seed, "seed"),
    )


# --------------------------------------------------------------------------- claims and evidence


@dataclass(frozen=True, slots=True)
class DiscoveryClaimKeys:
    """The node keys :meth:`SelectiveDiscovery.add_claims` created."""

    data: str
    statistics: str
    graph: str
    completion: str | None
    downstream_anchor: str


@dataclass(frozen=True, slots=True)
class DiscoveryClaims:
    """A composition graph extended with one discovery run, and the keys it added."""

    graph: CompositionGraph
    keys: DiscoveryClaimKeys


@dataclass(frozen=True, slots=True)
class PerPopulationClaims:
    """A composition graph extended with each population's discovery, keyed by population."""

    graph: CompositionGraph
    keys: Mapping[str, DiscoveryClaimKeys]


def _snapshot(snapshot_id: object) -> bytes:
    if not isinstance(snapshot_id, (bytes, bytearray)) or not snapshot_id:
        raise CausalTypeError("snapshot_id must be non-empty bytes")
    return bytes(snapshot_id)


def _completion(completion: object) -> tuple[str, str]:
    if (
        not isinstance(completion, tuple)
        or len(completion) != 2
        or not all(isinstance(part, str) for part in completion)
    ):
        raise CausalTypeError("completion must be a (label, identity) pair of strings")
    return completion[0], completion[1]


def _claim_keys(wire: Mapping[str, Any]) -> DiscoveryClaimKeys:
    completion = wire["completion"]
    return DiscoveryClaimKeys(
        data=str(wire["data"]),
        statistics=str(wire["statistics"]),
        graph=str(wire["graph"]),
        completion=None if completion is None else str(completion),
        downstream_anchor=str(wire["downstream_anchor"]),
    )


def _graph_text(graph: CompositionGraph | None) -> str | None:
    if graph is None:
        return None
    from .composition_verification import CompositionGraph as _Graph

    if not isinstance(graph, _Graph):
        raise CausalTypeError("graph must be a composition_verification.CompositionGraph or None")
    return json.dumps(graph.to_wire(), sort_keys=True)


def _graph_from_wire(wire: Mapping[str, Any]) -> CompositionGraph:
    from . import composition_verification as cv

    def calibration(item: Mapping[str, Any]) -> cv.Calibration:
        kind = str(item["kind"])
        if kind == "held":
            return cv.Calibration.held(str(item["record"]), str(item["coordinate"]))
        if kind == "inherited":
            return cv.Calibration.inherited(
                str(item["from"]), str(item["record"]), str(item["coordinate"])
            )
        return cv.Calibration.required() if kind == "required" else cv.Calibration.point_only()

    nodes = tuple(
        cv.Node(
            key=str(n["key"]),
            kind=str(n["kind"]),
            identity=str(n["identity"]),
            structure=n.get("structure"),
            representation=n.get("representation"),
            coordinate=n.get("coordinate"),
            unsupported=tuple(n.get("unsupported", ())),
            trust=str(n["trust"]),
            calibration=calibration(n["calibration"]),
            assumptions=tuple(n["assumptions"]),
            transformation=n.get("transformation"),
            evidence=tuple(n["evidence"]),
        )
        for n in wire["nodes"]
    )
    edges = tuple(
        cv.Edge(str(e["downstream"]), str(e["upstream"]), str(e["kind"])) for e in wire["edges"]
    )
    return cv.CompositionGraph(nodes=nodes, edges=edges)


def _declaration(text: str | Mapping[str, Any]) -> SourceDeclaration:
    from .evidence_ledger import Lineage, SourceDeclaration

    wire = json.loads(text) if isinstance(text, str) else text
    ids = wire.get("observation_ids")
    return SourceDeclaration(
        label=str(wire["label"]),
        kind=str(wire["kind"]),
        lineage=Lineage(dict(wire["lineage"])),
        observation_ids=None if ids is None else tuple(str(i) for i in ids),
    )


def discovery_data_source(label: str, snapshot_id: bytes, rows: range) -> SourceDeclaration:
    """Declare the rows a discovery run used as an evidence-ledger source.

    Observation identities are keyed by ``snapshot_id`` and the row index, never by which
    columns were searched. Declare the later analysis's rows with the same ``snapshot_id``: rows
    both used overlap (the ledger reports the shared observations) and a disjoint row range is
    independent. A range larger than the ledger's per-source bound is declared without
    observations, which the ledger treats as unknown overlap, never as independent.

    ``rows`` is a ``range`` with step 1.
    """
    snapshot_id = container_bytes(snapshot_id)
    if not isinstance(rows, range) or rows.step != 1:
        raise CausalTypeError("rows must be a range with step 1")
    stop = max(rows.start, rows.stop)
    if rows.start < 0:
        raise CausalValueError("rows must not be negative")
    text = _native.selective_discovery_data_source(
        _name(label, "label"), _snapshot(snapshot_id), rows.start, stop
    )
    return _declaration(text)


def discovery_rows_source(label: str, snapshot_id: bytes, rows: Sequence[int]) -> SourceDeclaration:
    """Declare an explicit set of rows (not necessarily contiguous) as an evidence source.

    As :func:`discovery_data_source`, for a row set such as one part of a sample split.
    """
    snapshot_id = container_bytes(snapshot_id)
    items = [_count(row, "row", minimum=0) for row in rows]
    text = _native.selective_discovery_rows_source(
        _name(label, "label"), _snapshot(snapshot_id), items
    )
    return _declaration(text)


def discovered_graph_source(label: str, data_sources: Sequence[str]) -> SourceDeclaration:
    """Declare a discovered graph as a fitted-model source built from data already used.

    ``data_sources`` are the ledger source ids (64 hexadecimal characters) of the discovery
    rows. A later analysis that treats the graph as evidence is counted as the rows it came
    from; the declaration makes that lineage explicit.
    """
    ids = [str(source) for source in data_sources]
    text = _native.selective_discovery_graph_source(_name(label, "label"), ids)
    return _declaration(text)
