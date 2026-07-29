"""Discover-then-fit GCM composition helpers.

Attribution never discovers structure internally (ADR 0012/0015). These helpers
compose ``discover_*`` → ``discovery_to_dag`` → ``fit_gcm`` / ``attribute_*``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._data import as_columns
from ._native import fit_gcm
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


def _run_static_discovery(data, discovery, *, seed: int, threads: int):
    if isinstance(discovery, (FCI, RFCI)):
        algo = "fci" if isinstance(discovery, FCI) else "rfci"
        raise ValueError(
            f"{algo}: fit_gcm_discovered requires a fully oriented DAG; "
            "use PC/GES/LiNGAM/NOTEARS, or orient the PAG and call fit_gcm directly"
        )
    return run_static_discovery(data, discovery, seed=seed, threads=threads)


def fit_gcm_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    seed: int = 1,
    threads: int = 1,
):
    """Discover structure, coerce to a DAG, then ``fit_gcm``.

    Returns ``(fitted_gcm, graph_edges)``. Incomplete CPDAG/PAG marks raise
    ``ValueError`` (orientations are never invented). Structure provenance is
    the caller-supplied ``discovery`` algorithm — attribution does not discover.
    """
    result, _algo = _run_static_discovery(data, discovery, seed=seed, threads=threads)
    dag = discovery_to_dag(result)
    names, columns = as_columns(data)
    edges = list(dag.edges())
    fitted = fit_gcm(names, columns, edges, threads=threads)
    return fitted, edges


def attribute_paths_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    sources: Sequence[str],
    outcome: str,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 1,
    threads: int = 1,
):
    """``fit_gcm_discovered`` then ``FittedGcm.attribute_paths``. Returns ``(result, graph_edges)``.

    Calls the fitted model's own ``attribute_paths`` instead of the
    module-level ``attribute_paths(names, columns, edges, ...)`` native
    function, which would otherwise silently re-fit the GCM from scratch —
    ``fit_gcm_discovered`` already paid for one fit; this reuses it.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, seed=seed, threads=threads)
    result = fitted.attribute_paths(
        list(sources),
        outcome,
        max_paths=max_paths,
        max_len=max_len,
        seed=seed,
        threads=threads,
    )
    return result, edges


def anomaly_attribution_discovered(
    data: Any,
    *,
    discovery: PC | GES | LiNGAM | NOTEARS,
    outcomes: Sequence[str],
    max_units: int = 0,
    seed: int = 1,
    threads: int = 1,
):
    """``fit_gcm_discovered`` then ``FittedGcm.anomaly_attribution``. Returns ``(result, graph_edges)``.

    See :func:`attribute_paths_discovered` for why this calls the fitted
    model's own method rather than the module-level free function.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, seed=seed, threads=threads)
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
    seed: int = 1,
    threads: int = 1,
):
    """Compose discover → DAG → ``FittedGcm.attribute_distribution_change``.

    See :func:`attribute_paths_discovered` for why this calls the fitted
    model's own method rather than the module-level free function.
    """
    fitted, edges = fit_gcm_discovered(data, discovery=discovery, seed=seed, threads=threads)
    result = fitted.attribute_distribution_change(
        outcome,
        baseline_start,
        baseline_end,
        comparison_start,
        comparison_end,
        n_samples=n_samples,
        seed=seed,
        threads=threads,
    )
    return result, edges


__all__ = [
    "anomaly_attribution_discovered",
    "attribute_distribution_change_discovered",
    "attribute_paths_discovered",
    "fit_gcm_discovered",
]
