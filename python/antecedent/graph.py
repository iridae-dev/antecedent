"""Graph types and interchange helpers.

Use the class methods for interchange (``Dag.from_dot``, ``Dag.to_json``,
``Dag.from_networkx_adjacency``, …, and the equivalents on ``Cpdag``,
``Pag``, ``Admg``). They return/accept the class's own name-keyed
representation — no separate integer-index free-function codecs.
"""

from __future__ import annotations

from ._native import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalCpdag,
    TemporalDag,
    TemporalPag,
)
from ._native import PcmciDiscoveryResult as _PcmciDiscoveryResult
from .errors import CausalTypeError, CausalValueError


def discovery_to_dag(result: _PcmciDiscoveryResult) -> Dag:
    """Build a ``Dag`` from a discovery result's directed ``graph_edges``.

    Raises ``ValueError`` if any undirected/circle marks remain.
    """
    names: list[str] = []
    seen: set[str] = set()
    directed: list[tuple[str, str]] = []
    for e in result.graph_edges:
        for n in (e.source, e.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
        if e.at_source == "tail" and e.at_target == "arrow":
            directed.append((e.source, e.target))
        elif e.at_source == "arrow" and e.at_target == "tail":
            directed.append((e.target, e.source))
        else:
            raise CausalValueError(
                f"cannot coerce edge {e.source}->{e.target} "
                f"({e.at_source}/{e.at_target}) into a DAG; "
                "use graph_edges or a CPDAG/PAG constructor"
            )
    return Dag.from_edges(names, directed)


def cpdag_oriented_edges(cpdag: Cpdag, *, require_oriented: bool = True) -> list[tuple[str, str]]:
    """Return directed edges from a CPDAG; error if undirected remain when required."""
    if not isinstance(cpdag, Cpdag):
        raise CausalTypeError(f"expected Cpdag, got {type(cpdag)!r}")
    directed: list[tuple[str, str]] = []
    undirected = 0
    for src, tgt, kind in cpdag.edges():
        if kind == "directed":
            directed.append((src, tgt))
        elif kind == "undirected":
            undirected += 1
        else:
            undirected += 1
    if undirected and require_oriented:
        raise CausalValueError(
            f"CPDAG has {undirected} undirected/ambiguous edge(s); orient before "
            "PathSpecific/Interventional queries (require_oriented=True)"
        )
    if require_oriented:
        try:
            dag = cpdag.try_into_dag()
        except Exception as exc:  # noqa: BLE001
            raise CausalValueError(
                "CPDAG is not fully oriented; cannot coerce to DAG for path/distribution queries"
            ) from exc
        return list(dag.edges())
    return directed


__all__ = [
    "Admg",
    "Cpdag",
    "cpdag_oriented_edges",
    "Dag",
    "discovery_to_dag",
    "Pag",
    "TemporalCpdag",
    "TemporalDag",
    "TemporalPag",
]
