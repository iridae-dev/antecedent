"""Shared estimation refusals and static-graph normalization.

``estimation``, ``batch`` and ``_prepare_route`` all need these. They live
here so those modules can import them without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .discovery import cpdag_oriented_edges
from .errors import CausalUnsupportedError, CausalValueError
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from .ids import Estimator


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise CausalValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, (Pag, TemporalDag, TemporalCpdag, TemporalPag, Admg)):
        raise CausalValueError(
            f"this query reads a fully oriented static Dag; got {type(graph).__name__}"
        )
    if isinstance(graph, Cpdag):
        # Compatibility normalization for fixed-DAG query families: a fully
        # oriented CPDAG has a unique DAG and reports that Dag coordinate.
        # Incomplete CPDAGs refuse; this does not license class execution.
        return cpdag_oriented_edges(graph, require_oriented=True)
    return [(str(a), str(b)) for a, b in graph]


def _refused(reason_code: str, message: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(message, reason_code=reason_code)


def _refused_with_remedy(reason_code: str, message: str, remedy: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(message, reason_code=reason_code, remedy=remedy)


def _not_applicable(option: str, route: str) -> CausalUnsupportedError:
    return _refused("option_not_applicable", f"{option} does not apply to {route}")


def _unwrap_estimator(
    estimator: Any, estimator_config: Mapping[str, Any] | None
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Split a typed estimator config into its id and wire configuration."""
    if estimator is None or isinstance(estimator, str):
        return estimator, estimator_config
    if isinstance(estimator, Estimator):
        return str(estimator), estimator_config
    if estimator_config is not None:
        raise ValueError(
            "estimator= already carries its configuration; do not also pass estimator_config="
        )
    return estimator.estimator_id, estimator._wire()
