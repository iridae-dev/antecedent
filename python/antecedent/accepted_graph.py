"""Versioned accepted-graph session handle for artifact-first interactive UX.

Discover once (or accept a reviewed graph), hold the artifact here, then run
many estimate clicks via :meth:`analyze` / :meth:`prepare`. Rediscovery is
**explicit only** — changing bootstrap, prior scale, treatment levels, or
latency never re-enters discovery.

Contrast with one-shot ``analyze(..., discovery=...)`` (script convenience;
refused under ``latency="interactive"``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ._native import CausalUnsupportedError
from .discovery import (
    LPCMCI,
    PCMCI,
    DiscoveryResult,
    PCMCIPlus,
    StaticDiscovery,
    TemporalDiscovery,
    run_static_discovery,
    run_temporal_discovery,
)
from .errors import CausalTypeError, CausalValueError, PendingEdge
from .graph import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalCpdag,
    TemporalDag,
    TemporalPag,
    discovery_to_dag,
)

_AnyDiscovery = StaticDiscovery | TemporalDiscovery

_GraphTypes = (
    Dag
    | Cpdag
    | Pag
    | Admg
    | TemporalDag
    | TemporalCpdag
    | TemporalPag
    | Sequence[tuple[str, str]]
    | Sequence[tuple[str, int, str, int]]
)


def _result_to_graph(result: DiscoveryResult, algorithm_id: str) -> Dag | Cpdag | Pag:
    """Coerce a static discovery result into a holdable graph artifact."""
    try:
        return discovery_to_dag(result)
    except ValueError:
        pass

    names: list[str] = []
    seen: set[str] = set()
    for e in result.graph_edges:
        for n in (e.source, e.target):
            if n not in seen:
                seen.add(n)
                names.append(n)

    if algorithm_id in ("fci", "rfci"):
        marked: list[tuple[str, str, str, str]] = [
            (e.source, e.target, e.at_source, e.at_target) for e in result.graph_edges
        ]
        return Pag.from_marked_edges(names, marked)

    directed: list[tuple[str, str]] = []
    undirected: list[tuple[str, str]] = []
    for e in result.graph_edges:
        if e.at_source == "tail" and e.at_target == "arrow":
            directed.append((e.source, e.target))
        elif e.at_source == "arrow" and e.at_target == "tail":
            directed.append((e.target, e.source))
        elif e.at_source == "tail" and e.at_target == "tail":
            a, b = e.source, e.target
            undirected.append((a, b) if a <= b else (b, a))
        else:
            raise CausalValueError(
                f"cannot hold edge {e.source}->{e.target} "
                f"({e.at_source}/{e.at_target}) as Dag/Cpdag; "
                "orient under review or use a PAG constructor"
            )
    return Cpdag.from_directed_undirected(names, directed, undirected)


def _temporal_names_from_graph_edges(result: DiscoveryResult) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for e in result.graph_edges:
        for n in (e.source, e.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
    if not names:
        cpdag_nodes = getattr(result, "cpdag_nodes", None)
        names = list(cpdag_nodes) if cpdag_nodes else ["x", "y"]
    return names


def _plain_pcmci_temporal_dag(result: DiscoveryResult) -> TemporalDag:
    """Build a ``TemporalDag`` from a plain-PCMCI result's ``links``.

    ``graph_edges`` is always empty for lagged-only PCMCI (see
    ``python/src/discovery_api.rs``) — every retained link is already a
    directed, time-ordered edge, with no CPDAG/PAG-style mark to resolve.
    """
    names: list[str] = []
    seen: set[str] = set()
    directed: list[tuple[str, int, str, int]] = []
    for link in result.links:
        for n in (link.source, link.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
        directed.append((link.source, int(link.source_lag), link.target, int(link.target_lag)))

    if not names:
        cpdag_nodes = getattr(result, "cpdag_nodes", None)
        names = list(cpdag_nodes) if cpdag_nodes else ["x", "y"]

    # TemporalDag is fully oriented by construction, matching PCMCI having no
    # CPDAG/PAG-style mark to resolve.
    return TemporalDag.from_lagged_edges(names, directed)


def _result_to_temporal_graph(
    result: DiscoveryResult, algorithm_id: str
) -> TemporalDag | TemporalCpdag | TemporalPag:
    """Coerce a PCMCI-family discovery result into a holdable temporal artifact.

    Mirrors ``_result_to_graph``'s directed/undirected split, using
    ``algorithm_id`` (rather than discarding it) because ``graph_edges`` is
    populated differently per algorithm on the native side
    (``python/src/discovery_api.rs``):

    - ``"lpcmci"``: ``graph_edges`` carries genuine ``circle`` marks — LPCMCI's
      native result is itself a ``TemporalPag`` (an ancestral graph with
      unresolved circle endpoints). Always held as a ``TemporalPag`` — this
      mirrors how ``_result_to_graph`` always holds FCI/RFCI as a ``Pag`` even
      when every mark happens to already be resolved — so incomplete
      orientations surface through :attr:`AcceptedGraph.pending` instead of
      silently reporting "nothing to review" (the gate this fixes).
    - ``"pcmci+"``: ``graph_edges`` carries ``("tail", "arrow")`` /
      ``("arrow", "tail")`` (oriented) and ``("tail", "tail")`` (unresolved
      contemporaneous) marks — never ``circle`` (the native ``TemporalCpdag``
      type structurally rejects it). Held as a ``TemporalDag`` when fully
      oriented, else a ``TemporalCpdag`` so the unresolved edges are visible
      via :attr:`AcceptedGraph.pending`.
    - ``"pcmci"`` (plain): ``graph_edges`` is always empty for lagged-only
      PCMCI — every retained link is already directed and time-ordered, with
      no CPDAG/PAG-style mark to resolve. Built from ``result.links`` instead,
      as before; ``TemporalDag`` is already fully oriented by construction, so
      there is nothing to gate here.
    """
    if algorithm_id == "lpcmci":
        names = _temporal_names_from_graph_edges(result)
        marked: list[tuple[str, int, str, int, str, str]] = [
            (e.source, int(e.source_lag), e.target, int(e.target_lag), e.at_source, e.at_target)
            for e in result.graph_edges
        ]
        return TemporalPag.from_marked_lagged_edges(names, marked)

    if algorithm_id == "pcmci+":
        names = _temporal_names_from_graph_edges(result)
        directed: list[tuple[str, int, str, int]] = []
        undirected: list[tuple[str, int, str, int]] = []
        for e in result.graph_edges:
            if e.at_source == "tail" and e.at_target == "arrow":
                directed.append((e.source, int(e.source_lag), e.target, int(e.target_lag)))
            elif e.at_source == "arrow" and e.at_target == "tail":
                directed.append((e.target, int(e.target_lag), e.source, int(e.source_lag)))
            elif e.at_source == "tail" and e.at_target == "tail":
                undirected.append((e.source, int(e.source_lag), e.target, int(e.target_lag)))
            else:
                raise CausalValueError(
                    f"cannot hold temporal edge {e.source}@{e.source_lag}->"
                    f"{e.target}@{e.target_lag} ({e.at_source}/{e.at_target}) as "
                    "TemporalDag/TemporalCpdag; orient under review or use LPCMCI's "
                    "TemporalPag constructor"
                )
        if undirected:
            return TemporalCpdag.from_lagged_edges(names, directed, undirected)
        return TemporalDag.from_lagged_edges(names, directed)

    # "pcmci" (plain) and any other temporal-family caller.
    return _plain_pcmci_temporal_dag(result)


class AcceptedGraph:
    """Versioned accepted CPDAG/PAG/DAG/temporal completion for estimate-only clicks.

    Estimate / prepare / refresh never call discovery. Only :meth:`rediscover`
    or constructing a new handle may replace structure and bump :attr:`version`.
    """

    __slots__ = ("_graph", "_version", "_algorithm_id")

    def __init__(
        self,
        graph: _GraphTypes,
        *,
        version: int = 1,
        algorithm_id: str | None = None,
    ) -> None:
        if version < 1:
            raise CausalValueError("version must be >= 1")
        self._graph = graph
        self._version = int(version)
        self._algorithm_id = algorithm_id

    @property
    def graph(self) -> _GraphTypes:
        return self._graph

    @property
    def version(self) -> int:
        return self._version

    @property
    def algorithm_id(self) -> str | None:
        return self._algorithm_id

    @classmethod
    def from_graph(
        cls,
        graph: _GraphTypes,
        *,
        algorithm_id: str | None = None,
        version: int = 1,
    ) -> AcceptedGraph:
        """Hold a reviewed or hand-authored graph artifact."""
        return cls(graph, version=version, algorithm_id=algorithm_id)

    @classmethod
    def from_discovery(
        cls,
        result: DiscoveryResult,
        *,
        algorithm_id: str,
        version: int = 1,
    ) -> AcceptedGraph:
        """Accept a standalone ``discover_*`` result into a session artifact."""
        if not algorithm_id:
            raise CausalValueError("algorithm_id is required for discovery provenance")
        algo = algorithm_id.lower().replace("pcmci_plus", "pcmci+")
        if algo in ("pcmci", "pcmci+", "lpcmci"):
            graph: _GraphTypes = _result_to_temporal_graph(result, algo)
        else:
            graph = _result_to_graph(result, algo)
        return cls(graph, version=version, algorithm_id=algo)

    @classmethod
    def asserted(
        cls,
        graph: _GraphTypes,
        *,
        algorithm_id: str | None = None,
        version: int = 1,
    ) -> AcceptedGraph:
        """Hold a hand-authored structure the caller is asserting — no discovery provenance.

        The documented spelling for what :meth:`from_graph` does; ``from_graph``
        stays as a thin alias (other call sites, including
        ``discovery.Config.accept()``, still spell it that way).
        """
        return cls.from_graph(graph, algorithm_id=algorithm_id, version=version)

    @classmethod
    def accepted(
        cls,
        result: DiscoveryResult,
        *,
        algorithm_id: str,
        version: int = 1,
    ) -> AcceptedGraph:
        """Accept a discovered structure the caller has reviewed.

        The documented spelling for what :meth:`from_discovery` does;
        ``from_discovery`` stays as a thin alias for existing callers.
        """
        return cls.from_discovery(result, algorithm_id=algorithm_id, version=version)

    def replace(
        self,
        graph: _GraphTypes,
        *,
        algorithm_id: str | None = None,
    ) -> AcceptedGraph:
        """Explicit structure replace (bumps version). Returns a new handle."""
        return AcceptedGraph(
            graph,
            version=self._version + 1,
            algorithm_id=algorithm_id if algorithm_id is not None else self._algorithm_id,
        )

    def rediscover(
        self,
        data: Any,
        discovery: _AnyDiscovery,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AcceptedGraph:
        """User-triggered rediscovery; never called by estimate / prepare."""
        if isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI)):
            result, algo = run_temporal_discovery(data, discovery, seed=seed, threads=threads)
            graph: _GraphTypes = _result_to_temporal_graph(result, algo)
        else:
            result, algo = run_static_discovery(data, discovery, seed=seed, threads=threads)
            graph = _result_to_graph(result, algo)
        return AcceptedGraph(graph, version=self._version + 1, algorithm_id=algo)

    @property
    def pending(self) -> tuple[PendingEdge, ...]:
        """Edges still needing orientation review (undirected marks / PAG circles).

        Empty for a fully oriented graph (``Dag``, ``TemporalDag``, or a plain
        edge list — none of those can carry an unresolved mark by
        construction). For ``Cpdag``, each undirected pair is reported as
        ``PendingEdge(source, target, at_source="tail", at_target="tail")``.
        For ``Pag``, each edge with a circle mark at either end is reported
        with its actual current marks (``at_source``/``at_target`` in
        ``{"tail", "arrow", "circle", "conflict"}``).

        ``Admg`` has no pending-review concept in this codebase — bidirected
        edges are a modeling choice, not an unresolved orientation — so it
        always reports empty. ``TemporalCpdag``/``TemporalPag`` expose no
        edge accessor in the native layer (only ``node_count()``); this
        raises :class:`antecedent.errors.CausalUnsupportedError` for those two
        graph kinds rather than silently reporting an empty tuple that would
        misrepresent an actually-incomplete graph as fully reviewed —
        *except* when ``TemporalCpdag.try_into_temporal_dag()`` succeeds,
        which proves there is nothing pending.
        """
        return _pending_edges(self._graph)

    def review(self, marks: Mapping[tuple[str, str], tuple[str, str]]) -> AcceptedGraph:
        """Apply orientation decisions to pending edges; returns a new, version-bumped handle.

        ``marks`` keys are ``(source, target)`` pairs exactly as they appear
        in :attr:`pending`; values are the new ``(at_source, at_target)``
        marks (``"tail"`` / ``"arrow"``, or ``"circle"`` on a ``Pag`` to leave
        that end still pending). A mark for an edge that isn't in
        :attr:`pending` raises ``ValueError`` naming the offending edge; on a
        ``Cpdag``, a mark that would create a directed cycle also raises
        ``ValueError`` naming the offending edge (``Pag`` orientation validity
        — the FCI soundness rules — is a harder problem than DAG acyclicity
        and isn't checked here).

        Only ``Cpdag`` and ``Pag`` currently support review (the two graph
        kinds :attr:`pending` can enumerate); a fully reviewed ``Cpdag``
        collapses to a plain ``Dag`` (via the directed/undirected edge lists,
        not the native ``try_into_dag()``, since the resolution already
        guarantees full orientation) so it's immediately usable by
        :meth:`prepare`.
        """
        return _review_graph(self, marks)

    def analyze(self, data: Any, query: Any, **kwargs: Any) -> Any:
        """Estimate on the held graph (default ``latency="interactive"``).

        Rejects caller ``discovery=``. Does not bump :attr:`version`.
        """
        from ._analyze import analyze

        if "discovery" in kwargs and kwargs["discovery"] is not None:
            raise CausalUnsupportedError(
                "AcceptedGraph.analyze rejects discovery=; structure is already accepted "
                "(call rediscover() for an explicit structure refresh)"
            )
        kwargs.pop("discovery", None)
        kwargs.setdefault("latency", "interactive")
        kwargs["graph"] = self
        return analyze(data, query=query, **kwargs)

    def prepare(self, data: Any, *, query: Any, **kwargs: Any) -> Any:
        """Compile-once prepared handle on the held static DAG/edges."""
        from .estimation import PreparedAnalysis

        if isinstance(self._graph, (Cpdag, Pag, Admg, TemporalCpdag, TemporalPag)):
            raise CausalUnsupportedError(
                "PreparedAnalysis requires a fully oriented Dag/TemporalDag (or edge list); "
                "complete CPDAG/PAG review first, then AcceptedGraph.from_graph(...)"
            )
        kwargs.setdefault("latency", "interactive")
        return PreparedAnalysis.prepare(data, query=query, graph=self, **kwargs)

    def to_json(self) -> str:
        """Serialize for durable hold (JSON interchange, not CBOR wire)."""
        kind, payload = _encode_graph(self._graph)
        return json.dumps(
            {
                "format": "causal.AcceptedGraph/v1",
                "version": self._version,
                "algorithm_id": self._algorithm_id,
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, s: str) -> AcceptedGraph:
        """Restore from :meth:`to_json`."""
        obj = json.loads(s)
        if obj.get("format") != "causal.AcceptedGraph/v1":
            raise CausalValueError(f"unsupported AcceptedGraph format: {obj.get('format')!r}")
        graph = _decode_graph(obj["kind"], obj["payload"])
        return cls(
            graph,
            version=int(obj["version"]),
            algorithm_id=obj.get("algorithm_id"),
        )

    def __len__(self) -> int:
        """Node count of the held graph."""
        return _node_count(self._graph)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the held graph's edges.

        Yields whatever shape the underlying graph kind's own edge listing
        would: 2-tuples for ``Dag``/edge lists, 3-tuples ``(source, target,
        kind)`` for ``Cpdag``, 4-tuples for ``TemporalDag``/lagged edge lists,
        and marked ``(source, target, at_source, at_target)`` tuples for
        ``Pag``/``Admg`` (the latter reconstructed from ``parents()`` /
        ``bidirected_neighbors()`` — it has no native ``.edges()``). Raises
        ``TypeError`` for ``TemporalCpdag``/``TemporalPag``, which expose no
        edge accessor at all in the native layer.
        """
        yield from _iter_edges(self._graph)

    def __contains__(self, item: Any) -> bool:
        """``True`` if ``item`` is a node name held by the graph, or an edge (as a 2-tuple).

        A plain string is checked against node names. A 2-tuple ``(source,
        target)`` is checked against edge endpoints *in that order* — mark /
        kind components beyond the first two are ignored, but direction is
        not: ``("a", "b") in accepted`` and ``("b", "a") in accepted`` are
        independent checks, matching how ``("a", "b") in dag.edges()`` already
        reads elsewhere in this codebase.
        """
        if isinstance(item, str):
            return item in _node_names(self._graph)
        if isinstance(item, tuple) and len(item) == 2:
            a, b = item
            return any(e[0] == a and e[1] == b for e in _iter_edges(self._graph))
        return False

    def __repr__(self) -> str:
        try:
            pending_count = len(self.pending)
        except CausalUnsupportedError:
            pending_count = -1  # unknown — native layer can't enumerate this graph kind
        return (
            f"AcceptedGraph(kind={type(self._graph).__name__!r}, "
            f"nodes={_node_count(self._graph)}, version={self._version}, "
            f"algorithm_id={self._algorithm_id!r}, pending={pending_count})"
        )

    # No __eq__: the native graph objects (Dag/Cpdag/Pag/Admg/Temporal*) have
    # no equality of their own, and two of the seven kinds (TemporalCpdag,
    # TemporalPag) expose no edge accessor to compare structurally either —
    # an __eq__ that's honest for five kinds and silently wrong (or raises)
    # for two is worse than no __eq__ at all.


def accept_discovery(
    config: _AnyDiscovery,
    data: Any,
    *,
    seed: int = 1,
    threads: int = 1,
) -> AcceptedGraph:
    """Run a discovery config and accept its result as a session artifact.

    Shared body for the nine discovery configs (PC, PCMCI, PCMCIPlus, LPCMCI,
    GES, LiNGAM, NOTEARS, FCI, RFCI) whose ``accept()`` methods all run
    ``config.run(data, seed=seed, threads=threads)`` then pass the result to
    :meth:`AcceptedGraph.from_discovery`. Lives here (rather than duplicated on
    each ``discovery.Config`` class) because this module already owns
    ``from_discovery``; ``discovery.py`` reaches this via a function-local
    import in each ``accept()`` to avoid the module-level import cycle (this
    module already imports from ``.discovery`` at module scope).
    """
    result = config.run(data, seed=seed, threads=threads)
    return AcceptedGraph.from_discovery(result, algorithm_id=config.algorithm_id)


def _encode_graph(graph: _GraphTypes) -> tuple[str, Any]:
    if isinstance(graph, TemporalDag):
        return "temporal_dag", {
            "names": _temporal_names(graph),
            "edges": [list(e) for e in graph.edges()],
        }
    if isinstance(graph, TemporalCpdag):
        # Prefer oriented TemporalDag when possible; otherwise names + empty edges.
        try:
            dag = graph.try_into_temporal_dag()
            return "temporal_dag", {
                "names": _temporal_names(dag),
                "edges": [list(e) for e in dag.edges()],
            }
        except Exception:
            return "temporal_cpdag", {
                "names": [f"v{i}" for i in range(graph.node_count())],
                "directed": [],
                "undirected": [],
            }
    if isinstance(graph, TemporalPag):
        return "temporal_pag", {
            "names": [f"v{i}" for i in range(graph.node_count())],
            "edges": [],
        }
    if isinstance(graph, Dag):
        return "dag", {"nodes": list(graph.nodes()), "edges": [list(e) for e in graph.edges()]}
    if isinstance(graph, Cpdag):
        return "cpdag", graph.to_json()
    if isinstance(graph, Pag):
        return "pag", graph.to_json()
    if isinstance(graph, Admg):
        return "admg", graph.to_json()
    # Edge list — static pairs or lagged quadruples.
    edges = [list(e) for e in graph]
    if edges and len(edges[0]) == 4:
        return "temporal_edges", {"edges": edges}
    return "edges", {"edges": edges}


def _temporal_names(graph: TemporalDag) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for name, _lag in graph.nodes():
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _decode_graph(kind: str, payload: Any) -> _GraphTypes:
    if kind == "temporal_dag":
        names = list(payload["names"])
        lagged = [(str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["edges"]]
        return TemporalDag.from_lagged_edges(names, lagged)
    if kind == "temporal_cpdag":
        names = list(payload["names"])
        directed = [(str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["directed"]]
        undirected = [
            (str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload.get("undirected", [])
        ]
        return TemporalCpdag.from_lagged_edges(names, directed, undirected or None)
    if kind == "temporal_pag":
        names = list(payload["names"])
        marked = [
            (str(a), int(sa), str(b), int(tb), str(ma), str(mb))
            for a, sa, b, tb, ma, mb in payload.get("edges", [])
        ]
        return TemporalPag.from_marked_lagged_edges(names, marked)
    if kind == "temporal_edges":
        return [(str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["edges"]]
    if kind == "dag":
        nodes = list(payload["nodes"])
        static = [(str(a), str(b)) for a, b in payload["edges"]]
        return Dag.from_edges(nodes, static)
    if kind == "cpdag":
        return Cpdag.from_json(payload)
    if kind == "pag":
        return Pag.from_json(payload)
    if kind == "admg":
        return Admg.from_json(payload)
    if kind == "edges":
        return [(str(a), str(b)) for a, b in payload["edges"]]
    raise CausalValueError(f"unknown AcceptedGraph kind: {kind!r}")


def _node_names(graph: _GraphTypes) -> list[str]:
    if isinstance(graph, TemporalDag):
        return _temporal_names(graph)
    if isinstance(graph, (TemporalCpdag, TemporalPag)):
        raise CausalUnsupportedError(
            f"{type(graph).__name__} exposes no node-name accessor in the native layer "
            "(only node_count())"
        )
    if isinstance(graph, (Dag, Cpdag, Pag, Admg)):
        return list(graph.nodes())
    # Plain edge list — static pairs or lagged quadruples.
    names: list[str] = []
    seen: set[str] = set()
    for e in graph:
        a, b = (e[0], e[2]) if len(e) == 4 else (e[0], e[1])
        for n in (a, b):
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _node_count(graph: _GraphTypes) -> int:
    if isinstance(graph, (Dag, Cpdag, Pag, Admg, TemporalDag, TemporalCpdag, TemporalPag)):
        return int(graph.node_count())
    return len(_node_names(graph))


def _pag_edge_doc(graph: Pag) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse `Pag.to_json()`'s `{node_count, edges: [{a, b, at_a, at_b}], variable_names}`."""
    doc = json.loads(graph.to_json())
    names = list(doc.get("variable_names") or (str(i) for i in range(doc["node_count"])))
    return names, list(doc["edges"])


def _pag_pending(graph: Pag) -> list[PendingEdge]:
    names, edges = _pag_edge_doc(graph)
    return [
        PendingEdge(
            source=names[e["a"]],
            target=names[e["b"]],
            at_source=e["at_a"],
            at_target=e["at_b"],
        )
        for e in edges
        if e["at_a"] == "circle" or e["at_b"] == "circle"
    ]


def _iter_edges(graph: _GraphTypes) -> Iterator[Any]:
    if isinstance(graph, (Dag, TemporalDag, Cpdag)):
        yield from graph.edges()
        return
    if isinstance(graph, Pag):
        names, edges = _pag_edge_doc(graph)
        for e in edges:
            yield (names[e["a"]], names[e["b"]], e["at_a"], e["at_b"])
        return
    if isinstance(graph, Admg):
        nodes = list(graph.nodes())
        for n in nodes:
            for p in graph.parents(n):
                yield (p, n, "directed")
        seen: set[frozenset[str]] = set()
        for n in nodes:
            for m in graph.bidirected_neighbors(n):
                pair = frozenset((n, m))
                if pair not in seen:
                    seen.add(pair)
                    a, b = sorted(pair)
                    yield (a, b, "bidirected")
        return
    if isinstance(graph, (TemporalCpdag, TemporalPag)):
        raise CausalTypeError(
            f"{type(graph).__name__} exposes no edge accessor in the native layer; "
            "iteration is unsupported for this graph kind"
        )
    # Plain edge list.
    yield from graph


def _pending_edges(graph: _GraphTypes) -> tuple[PendingEdge, ...]:
    if isinstance(graph, (Dag, TemporalDag, Admg)):
        return ()  # fully oriented by construction; Admg has no "pending" concept here
    if isinstance(graph, Cpdag):
        return tuple(
            PendingEdge(source=src, target=tgt, at_source="tail", at_target="tail")
            for src, tgt, kind in graph.edges()
            if kind != "directed"
        )
    if isinstance(graph, Pag):
        return tuple(_pag_pending(graph))
    if isinstance(graph, TemporalCpdag):
        try:
            graph.try_into_temporal_dag()
        except Exception as exc:  # noqa: BLE001 — surfacing "can't enumerate", not orientation
            raise CausalUnsupportedError(
                "TemporalCpdag has undirected marks but exposes no edge accessor to list "
                "them from Python; orient via try_into_temporal_dag() success, or hold the "
                "discovery result's graph_edges directly for review"
            ) from exc
        return ()
    if isinstance(graph, TemporalPag):
        raise CausalUnsupportedError(
            "TemporalPag exposes no edge accessor in the native layer; pending circle "
            "marks cannot be enumerated from Python"
        )
    # Plain edge list — always fully oriented.
    return ()


def _find_cycle_edge(edges: Sequence[tuple[str, str]]) -> tuple[str, str] | None:
    """DFS cycle detection; returns the back-edge that closes a cycle, else None."""
    adjacency: dict[str, list[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, [])
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(adjacency, WHITE)

    def visit(node: str) -> tuple[str, str] | None:
        color[node] = GRAY
        for neighbor in adjacency[node]:
            if color[neighbor] == GRAY:
                return (node, neighbor)
            if color[neighbor] == WHITE:
                found = visit(neighbor)
                if found is not None:
                    return found
        color[node] = BLACK
        return None

    for node in list(adjacency):
        if color[node] == WHITE:
            found = visit(node)
            if found is not None:
                return found
    return None


def _review_cpdag(
    accepted: AcceptedGraph, graph: Cpdag, marks: Mapping[tuple[str, str], tuple[str, str]]
) -> AcceptedGraph:
    directed: list[tuple[str, str]] = []
    undirected_pairs: set[frozenset[str]] = set()
    for src, tgt, kind in graph.edges():
        if kind == "directed":
            directed.append((src, tgt))
        else:
            undirected_pairs.add(frozenset((src, tgt)))

    resolved: dict[frozenset[str], tuple[str, str]] = {}
    for edge, mark in marks.items():
        src, tgt = edge
        key = frozenset((src, tgt))
        if key not in undirected_pairs:
            raise CausalValueError(f"({src!r}, {tgt!r}) is not a pending edge on this Cpdag")
        at_source, at_target = mark
        if (at_source, at_target) == ("tail", "arrow"):
            resolved[key] = (src, tgt)
        elif (at_source, at_target) == ("arrow", "tail"):
            resolved[key] = (tgt, src)
        else:
            raise CausalValueError(
                f"mark {mark!r} for ({src!r}, {tgt!r}) is not a valid Cpdag orientation; "
                'use ("tail", "arrow") or ("arrow", "tail")'
            )

    new_directed = list(directed)
    new_undirected: list[tuple[str, str]] = []
    # Sorted for deterministic output — sets have no stable iteration order.
    for pair in sorted(undirected_pairs, key=lambda p: tuple(sorted(p))):
        if pair in resolved:
            new_directed.append(resolved[pair])
        else:
            a, b = tuple(sorted(pair))
            new_undirected.append((a, b))

    cycle_edge = _find_cycle_edge(new_directed)
    if cycle_edge is not None:
        raise CausalValueError(
            f"orienting {cycle_edge[0]!r}->{cycle_edge[1]!r} would create a directed cycle"
        )

    names = list(graph.nodes())
    new_graph: _GraphTypes
    if new_undirected:
        new_graph = Cpdag.from_directed_undirected(names, new_directed, new_undirected)
    else:
        # Fully resolved — collapse to a Dag so .prepare() works immediately.
        new_graph = Dag.from_edges(names, new_directed)
    return accepted.replace(new_graph)


def _review_pag(
    accepted: AcceptedGraph, graph: Pag, marks: Mapping[tuple[str, str], tuple[str, str]]
) -> AcceptedGraph:
    names, edges = _pag_edge_doc(graph)
    index = {name: i for i, name in enumerate(names)}
    pending_pairs = {frozenset((e.source, e.target)) for e in _pag_pending(graph)}

    updates: dict[tuple[int, int], tuple[str, str]] = {}
    for edge, mark in marks.items():
        src, tgt = edge
        if frozenset((src, tgt)) not in pending_pairs:
            raise CausalValueError(f"({src!r}, {tgt!r}) is not a pending edge on this Pag")
        if src not in index or tgt not in index:
            raise CausalValueError(f"({src!r}, {tgt!r}) is not a known edge on this Pag")
        at_source, at_target = mark
        if at_source not in ("tail", "arrow", "circle") or at_target not in (
            "tail",
            "arrow",
            "circle",
        ):
            raise CausalValueError(
                f"mark {mark!r} for ({src!r}, {tgt!r}) must use tail/arrow/circle endpoints"
            )
        updates[(index[src], index[tgt])] = (at_source, at_target)

    new_edges: list[tuple[str, str, str, str]] = []
    for e in edges:
        a, b, at_a, at_b = e["a"], e["b"], e["at_a"], e["at_b"]
        if (a, b) in updates:
            at_a, at_b = updates[(a, b)]
        elif (b, a) in updates:
            at_b, at_a = updates[(b, a)]
        new_edges.append((names[a], names[b], at_a, at_b))

    # No cycle check here: PAG orientation validity (FCI soundness rules) is a
    # harder problem than DAG acyclicity and isn't implemented — see
    # AcceptedGraph.review's docstring.
    new_graph = Pag.from_marked_edges(names, new_edges)
    return accepted.replace(new_graph)


def _review_graph(
    accepted: AcceptedGraph, marks: Mapping[tuple[str, str], tuple[str, str]]
) -> AcceptedGraph:
    graph = accepted.graph
    if isinstance(graph, Cpdag):
        return _review_cpdag(accepted, graph, marks)
    if isinstance(graph, Pag):
        return _review_pag(accepted, graph, marks)
    if not marks:
        return accepted.replace(graph)
    edge = next(iter(marks))
    raise CausalValueError(
        f"{edge!r} is not a pending edge on {type(graph).__name__} (pending={accepted.pending!r})"
    )


__all__ = ["AcceptedGraph"]
