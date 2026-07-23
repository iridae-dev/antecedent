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
from typing import Any, Sequence

from ._native import CausalUnsupportedError
from .discovery import (
    FCI,
    GES,
    LPCMCI,
    LiNGAM,
    NOTEARS,
    PC,
    PCMCI,
    PCMCIPlus,
    RFCI,
    DiscoveryResult,
    discover_fci,
    discover_ges,
    discover_lingam,
    discover_lpcmci,
    discover_notears,
    discover_pc,
    discover_pcmci,
    discover_pcmci_plus,
    discover_rfci,
    discovery_to_dag,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag

_StaticDiscovery = PC | GES | LiNGAM | NOTEARS | FCI | RFCI
_TemporalDiscovery = PCMCI | PCMCIPlus | LPCMCI
_AnyDiscovery = _StaticDiscovery | _TemporalDiscovery

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


def _run_static_discovery(
    data: Any,
    discovery: _StaticDiscovery,
    *,
    seed: int,
    threads: int,
) -> tuple[DiscoveryResult, str]:
    if isinstance(discovery, PC):
        return (
            discover_pc(
                data,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                ci=discovery.ci if isinstance(discovery.ci, str) else "parcorr",
                max_cond_size=discovery.max_cond_size,
            ),
            "pc",
        )
    if isinstance(discovery, GES):
        return (
            discover_ges(
                data,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
            ),
            "ges",
        )
    if isinstance(discovery, LiNGAM):
        return discover_lingam(data, seed=seed, threads=threads), "lingam"
    if isinstance(discovery, NOTEARS):
        return discover_notears(data, seed=seed, threads=threads), "notears"
    if isinstance(discovery, FCI):
        return (
            discover_fci(
                data,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                max_cond_size=discovery.max_cond_size,
            ),
            "fci",
        )
    if isinstance(discovery, RFCI):
        return (
            discover_rfci(
                data,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                max_cond_size=discovery.max_cond_size,
            ),
            "rfci",
        )
    raise TypeError(f"unsupported static discovery type for AcceptedGraph: {type(discovery)!r}")


def _run_temporal_discovery(
    data: Any,
    discovery: _TemporalDiscovery,
    *,
    seed: int,
    threads: int,
) -> tuple[DiscoveryResult, str]:
    if isinstance(discovery, PCMCI):
        return (
            discover_pcmci(
                data=data,
                max_lag=discovery.max_lag,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                ci=discovery.ci if isinstance(discovery.ci, str) else "parcorr",
            ),
            "pcmci",
        )
    if isinstance(discovery, PCMCIPlus):
        return (
            discover_pcmci_plus(
                data=data,
                max_lag=discovery.max_lag,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                ci=discovery.ci if isinstance(discovery.ci, str) else "parcorr",
            ),
            "pcmci+",
        )
    if isinstance(discovery, LPCMCI):
        return (
            discover_lpcmci(
                data=data,
                max_lag=discovery.max_lag,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
                ci=discovery.ci if isinstance(discovery.ci, str) else "parcorr",
            ),
            "lpcmci",
        )
    raise TypeError(f"unsupported temporal discovery type for AcceptedGraph: {type(discovery)!r}")


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
            raise ValueError(
                f"cannot hold edge {e.source}->{e.target} "
                f"({e.at_source}/{e.at_target}) as Dag/Cpdag; "
                "orient under review or use a PAG constructor"
            )
    return Cpdag.from_directed_undirected(names, directed, undirected)


def _result_to_temporal_graph(
    result: DiscoveryResult, algorithm_id: str
) -> TemporalDag | TemporalCpdag | TemporalPag:
    """Coerce a PCMCI-family discovery result into a holdable temporal artifact."""
    del algorithm_id  # reserved for PAG/LPCMCI mark handling
    names: list[str] = []
    seen: set[str] = set()
    directed: list[tuple[str, int, str, int]] = []
    for link in result.links:
        for n in (link.source, link.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
        directed.append(
            (link.source, int(link.source_lag), link.target, int(link.target_lag))
        )

    if not names:
        cpdag_nodes = getattr(result, "cpdag_nodes", None)
        if cpdag_nodes:
            names = list(cpdag_nodes)
        else:
            names = ["x", "y"]
    # Hold as TemporalDag from retained links. Incomplete PCMCI+ orientations
    # should be completed under review before estimate; directed links still form
    # a usable completion artifact for pulse estimates on identified paths.
    return TemporalDag.from_lagged_edges(names, directed)


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
            raise ValueError("version must be >= 1")
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
            raise ValueError("algorithm_id is required for discovery provenance")
        algo = algorithm_id.lower().replace("pcmci_plus", "pcmci+")
        if algo in ("pcmci", "pcmci+", "lpcmci"):
            graph: _GraphTypes = _result_to_temporal_graph(result, algo)
        else:
            graph = _result_to_graph(result, algo)
        return cls(graph, version=version, algorithm_id=algo)

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
            result, algo = _run_temporal_discovery(
                data, discovery, seed=seed, threads=threads
            )
            graph: _GraphTypes = _result_to_temporal_graph(result, algo)
        else:
            result, algo = _run_static_discovery(
                data, discovery, seed=seed, threads=threads
            )
            graph = _result_to_graph(result, algo)
        return AcceptedGraph(graph, version=self._version + 1, algorithm_id=algo)

    def analyze(self, data: Any, query: Any, **kwargs: Any) -> Any:
        """Estimate on the held graph (default ``latency="interactive"``).

        Rejects caller ``discovery=``. Does not bump :attr:`version`.
        """
        from .estimation import analyze

        if "discovery" in kwargs and kwargs["discovery"] is not None:
            raise CausalUnsupportedError(
                "AcceptedGraph.analyze rejects discovery=; structure is already accepted "
                "(call rediscover() for an explicit structure refresh)"
            )
        kwargs.pop("discovery", None)
        kwargs.setdefault("latency", "interactive")
        kwargs["graph"] = self._graph
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
        return PreparedAnalysis.prepare(data, query=query, graph=self._graph, **kwargs)  # type: ignore[arg-type]

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
            raise ValueError(f"unsupported AcceptedGraph format: {obj.get('format')!r}")
        graph = _decode_graph(obj["kind"], obj["payload"])
        return cls(
            graph,
            version=int(obj["version"]),
            algorithm_id=obj.get("algorithm_id"),
        )


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
        edges = [
            (str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["edges"]
        ]
        return TemporalDag.from_lagged_edges(names, edges)
    if kind == "temporal_cpdag":
        names = list(payload["names"])
        directed = [
            (str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["directed"]
        ]
        undirected = [
            (str(a), int(sa), str(b), int(tb))
            for a, sa, b, tb in payload.get("undirected", [])
        ]
        return TemporalCpdag.from_lagged_edges(names, directed, undirected or None)
    if kind == "temporal_pag":
        names = list(payload["names"])
        edges = [
            (str(a), int(sa), str(b), int(tb), str(ma), str(mb))
            for a, sa, b, tb, ma, mb in payload.get("edges", [])
        ]
        return TemporalPag.from_marked_lagged_edges(names, edges)
    if kind == "temporal_edges":
        return [
            (str(a), int(sa), str(b), int(tb)) for a, sa, b, tb in payload["edges"]
        ]
    if kind == "dag":
        nodes = list(payload["nodes"])
        edges = [(str(a), str(b)) for a, b in payload["edges"]]
        return Dag.from_edges(nodes, edges)
    if kind == "cpdag":
        return Cpdag.from_json(payload)
    if kind == "pag":
        return Pag.from_json(payload)
    if kind == "admg":
        return Admg.from_json(payload)
    if kind == "edges":
        return [(str(a), str(b)) for a, b in payload["edges"]]
    raise ValueError(f"unknown AcceptedGraph kind: {kind!r}")

__all__ = ["AcceptedGraph"]
