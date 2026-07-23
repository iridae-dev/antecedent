"""Graph types and interchange helpers.

Prefer class methods (``Dag.from_dot``, ``Dag.to_json``, …). Free
``dag_from_*`` / ``dag_to_*`` functions remain thin aliases that return
``(node_count, edges[, names])`` for low-level codecs.
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
    dag_from_dot,
    dag_from_gml,
    dag_from_json,
    dag_from_networkx_adjacency,
    dag_from_networkx_node_link,
    dag_to_dot,
    dag_to_gml,
    dag_to_json,
    dag_to_networkx_adjacency,
    dag_to_networkx_node_link,
)

__all__ = [
    "Admg",
    "Cpdag",
    "Dag",
    "Pag",
    "TemporalCpdag",
    "TemporalDag",
    "TemporalPag",
    "dag_from_dot",
    "dag_from_gml",
    "dag_from_json",
    "dag_from_networkx_adjacency",
    "dag_from_networkx_node_link",
    "dag_to_dot",
    "dag_to_gml",
    "dag_to_json",
    "dag_to_networkx_adjacency",
    "dag_to_networkx_node_link",
]
