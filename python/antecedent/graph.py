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

__all__ = [
    "Admg",
    "Cpdag",
    "Dag",
    "Pag",
    "TemporalCpdag",
    "TemporalDag",
    "TemporalPag",
]
