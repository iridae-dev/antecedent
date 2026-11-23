"""Graphical causal models fitted on a discovered structure, and attribution on them.

Attribution never discovers structure on its own. The ``*_discovered`` helpers run one
tabular discovery (:class:`~antecedent.discovery.PC`, :class:`~antecedent.discovery.GES`,
:class:`~antecedent.discovery.LiNGAM` or :class:`~antecedent.discovery.NOTEARS`), accept its
structure as a DAG through the review gate, fit the graphical causal model on it, and run one
attribution. The structure is accepted from the discovery's own route, so every attribution on
the fitted model states the discovery's provenance in its ``contract``. A PAG from
:class:`~antecedent.discovery.FCI` or :class:`~antecedent.discovery.RFCI`, or a CPDAG with
unoriented edges, is not a DAG and raises ``ValueError``; orientations are never invented.

The module also re-exports graph falsification (:func:`evaluate_graph`) and intrinsic
influence (:func:`intrinsic_influence`, :func:`fitted_intrinsic_influence`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._data import as_columns
from .counterfactual import _fit_gcm_accepted
from .discovery import (
    FCI,
    GES,
    NOTEARS,
    PC,
    RFCI,
    LiNGAM,
    discovery_to_dag,
    run_static_discovery,
)
from .execution import Execution
from .graph_falsification import GraphFalsification, MarkovImplicationTest, evaluate_graph
from .intrinsic_influence import (
    ComponentUncertainty,
    IntrinsicInfluence,
    MechanismPosterior,
    SourceInfluence,
    SourcePosterior,
    fitted_intrinsic_influence,
    intrinsic_influence,
)


def _run_static_discovery(data, discovery, *, execution: Execution | None):
    if isinstance(discovery, (FCI, RFCI)):
        algo = "fci" if isinstance(discovery, FCI) else "rfci"
        raise ValueError(
            f"{algo}: fit_gcm_discovered requires a fully oriented DAG; "
            "use PC/GES/LiNGAM/NOTEARS, or orient the PAG and call fit_gcm directly"
        )
    return run_static_discovery(data, discovery, execution=execution)


def fit_gcm_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    execution: Execution | None = None,
):
    """Discover structure, accept it as a DAG through the review gate, then fit.

    Returns ``(fitted_gcm, graph_edges)``. Incomplete CPDAG/PAG marks raise
    ``ValueError`` (orientations are never invented). The model is fitted on the
    structure the discovery's compiled route accepted, so the contract of every
    attribution on it carries ``structure_provenance.*`` (method, inference, query,
    data snapshot, program and claim) — attribution does not discover.
    """
    result, _algo = _run_static_discovery(data, discovery, execution=execution)
    dag = discovery_to_dag(result)
    names, columns = as_columns(data)
    edges = list(dag.edges())
    fitted, _accepted_edges = _fit_gcm_accepted(names, columns, result, execution=execution)
    return fitted, edges


def attribute_paths_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    sources: Sequence[str],
    outcome: str,
    max_paths: int = 64,
    max_len: int = 16,
    execution: Execution | None = None,
):
    """:func:`fit_gcm_discovered`, then the fitted model's ``attribute_paths``.

    Attributes the effect of ``sources`` on ``outcome`` to the causal paths between them,
    enumerating at most ``max_paths`` paths of at most ``max_len`` edges. The model is fitted
    once and reused. Returns ``(result, graph_edges)``.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, execution=execution)
    result = fitted.attribute_paths(
        list(sources),
        outcome,
        max_paths=max_paths,
        max_len=max_len,
        execution=execution,
    )
    return result, edges


def anomaly_attribution_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    outcomes: Sequence[str],
    max_units: int = 0,
    execution: Execution | None = None,
):
    """:func:`fit_gcm_discovered`, then the fitted model's ``anomaly_attribution``.

    Attributes the values of ``outcomes`` in each unit to upstream causes; ``max_units``
    bounds the units attributed (``0``, the default, attributes every row). Returns
    ``(result, graph_edges)``.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, execution=execution)
    result = fitted.anomaly_attribution(list(outcomes), max_units=max_units)
    return result, edges


def attribute_distribution_change_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    n_samples: int = 500,
    execution: Execution | None = None,
):
    """:func:`fit_gcm_discovered`, then the fitted model's ``attribute_distribution_change``.

    Attributes the change in the distribution of ``outcome`` between the baseline rows
    ``[baseline_start, baseline_end)`` and the comparison rows
    ``[comparison_start, comparison_end)`` to the causal mechanisms, by Monte Carlo Shapley
    allocation with ``n_samples`` samples. Returns ``(result, graph_edges)``.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, execution=execution)
    result = fitted.attribute_distribution_change(
        outcome,
        baseline_start,
        baseline_end,
        comparison_start,
        comparison_end,
        n_samples=n_samples,
        execution=execution,
    )
    return result, edges


__all__ = [
    "ComponentUncertainty",
    "GraphFalsification",
    "IntrinsicInfluence",
    "MarkovImplicationTest",
    "MechanismPosterior",
    "SourceInfluence",
    "SourcePosterior",
    "anomaly_attribution_discovered",
    "attribute_distribution_change_discovered",
    "attribute_paths_discovered",
    "evaluate_graph",
    "fit_gcm_discovered",
    "fitted_intrinsic_influence",
    "intrinsic_influence",
]
