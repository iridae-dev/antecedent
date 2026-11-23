"""Shared estimation refusals and static-graph normalization.

``estimation``, ``batch`` and ``_prepare_route`` all need these. They live
here so those modules can import them without importing each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .discovery import cpdag_oriented_edges
from .errors import CausalUnsupportedError, CausalValueError
from .graph import (
    Admg,
    Cpdag,
    Dag,
    Pag,
    TemporalAdmg,
    TemporalCpdag,
    TemporalDag,
    TemporalPag,
)
from .ids import Estimator


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """The directed edges of the explicit DAG a DAG-only route reads.

    Routes with a class executor (the derivatives, mediation, counterfactuals, path-specific
    effects and interventional distributions, and the class-aware averages and responses) bind
    a ``Cpdag``, ``Pag``, ``Admg`` or ``TemporalAdmg`` as itself and never call this.
    """
    if graph is None:
        raise CausalValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, (Pag, TemporalDag, TemporalCpdag, TemporalPag, Admg, TemporalAdmg)):
        raise CausalValueError(
            f"this query reads a fully oriented static Dag; got {type(graph).__name__}"
        )
    if isinstance(graph, Cpdag):
        # A DAG-only route reads a fully oriented CPDAG as its unique DAG and reports that Dag
        # coordinate; an incomplete CPDAG refuses.
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
