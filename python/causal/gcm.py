"""Discover-then-fit GCM composition helpers.

Attribution never discovers structure internally (ADR 0012/0015). These helpers
compose ``discover_*`` → ``discovery_to_dag`` → ``fit_gcm`` / ``attribute_*``.
"""

from __future__ import annotations

from typing import Any, Sequence

from ._data import as_columns
from ._native import (
    anomaly_attribution,
    attribute_distribution_change,
    attribute_paths,
    fit_gcm,
)
from .discovery import (
    FCI,
    GES,
    LiNGAM,
    NOTEARS,
    PC,
    RFCI,
    discover_fci,
    discover_ges,
    discover_lingam,
    discover_notears,
    discover_pc,
    discover_rfci,
    discovery_to_dag,
)


def _run_static_discovery(data, discovery, *, seed: int, threads: int):
    if isinstance(discovery, PC):
        return discover_pc(
            data, alpha=discovery.alpha, fdr=discovery.fdr, seed=seed, threads=threads
        ), "pc"
    if isinstance(discovery, GES):
        return discover_ges(
            data, alpha=discovery.alpha, fdr=discovery.fdr, seed=seed, threads=threads
        ), "ges"
    if isinstance(discovery, LiNGAM):
        return discover_lingam(data, seed=seed, threads=threads), "lingam"
    if isinstance(discovery, NOTEARS):
        return discover_notears(data, seed=seed, threads=threads), "notears"
    if isinstance(discovery, (FCI, RFCI)):
        algo = "fci" if isinstance(discovery, FCI) else "rfci"
        raise ValueError(
            f"{algo}: fit_gcm_discovered requires a fully oriented DAG; "
            "use PC/GES/LiNGAM/NOTEARS, or orient the PAG and call fit_gcm directly"
        )
    raise TypeError(f"unsupported discovery type for GCM compose: {type(discovery)!r}")


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
    """``fit_gcm_discovered`` then ``attribute_paths``. Returns ``(result, graph_edges)``."""
    fitted, edges = fit_gcm_discovered(
        data, discovery=discovery, seed=seed, threads=threads
    )
    _ = fitted
    names, columns = as_columns(data)
    result = attribute_paths(
        names,
        columns,
        edges,
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
    """``fit_gcm_discovered`` then ``anomaly_attribution``. Returns ``(result, graph_edges)``."""
    fitted, edges = fit_gcm_discovered(
        data, discovery=discovery, seed=seed, threads=threads
    )
    _ = fitted
    names, columns = as_columns(data)
    result = anomaly_attribution(
        names, columns, edges, list(outcomes), max_units=max_units
    )
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
    """Compose discover → DAG → ``attribute_distribution_change``."""
    fitted, edges = fit_gcm_discovered(
        data, discovery=discovery, seed=seed, threads=threads
    )
    _ = fitted
    names, columns = as_columns(data)
    result = attribute_distribution_change(
        names,
        columns,
        edges,
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
