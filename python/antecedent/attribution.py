"""Attribution helpers.

The change, structure-change, path, feature-relevance, anomaly and unit-change attributions run
on mechanisms fitted to the data. Each accepts ``posterior_draws`` (the posterior over the
mechanisms: that many Dirichlet-weighted refits of every mechanism, the families held at the ones
selected on the data) or ``bootstrap_replicates`` (the mechanism-refit row bootstrap), and then
returns an :class:`AttributionWithUncertainty` holding the point result and the uncertainty of
every quantity it reports. Both are uncalibrated and unlicensed. Without either, the functions
return the point result as before. Mechanism-change detection has no posterior over the
mechanisms: it is a two-sample test, not a smooth functional of the refitted mechanisms.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._native import (
    AnomalyScores,
    ChangeAttributionResult,
    Contribution,
    FeatureRelevance,
    MechanismChangeDetection,
    attribute_paths,
    mechanism_change_detection,
    rank_root_causes,
)


@dataclass(frozen=True)
class PosteriorOverMechanisms:
    """Posterior of each reported quantity over the Dirichlet-weighted mechanism refits."""

    draws: int
    level: float
    names: tuple[str, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    ci_low: tuple[float, ...]
    ci_high: tuple[float, ...]

    def quantity(self, name: str) -> tuple[float, float, float, float]:
        """``(mean, sd, ci_low, ci_high)`` of ``name``."""
        index = self.names.index(name)
        return self.mean[index], self.sd[index], self.ci_low[index], self.ci_high[index]


@dataclass(frozen=True)
class RefitBootstrap:
    """Standard error of each reported quantity over the mechanism-refit row bootstrap.

    A quantity whose replicates were too few or failed has a standard error of ``None``.
    """

    replicates_ok: int
    replicates_failed: int
    se: dict[str, float | None]


@dataclass(frozen=True)
class AttributionUncertainty:
    """The point value of each quantity and its posterior or bootstrap uncertainty."""

    kind: str
    point: dict[str, float]
    posterior: PosteriorOverMechanisms | None
    bootstrap: RefitBootstrap | None


@dataclass(frozen=True)
class AttributionWithUncertainty:
    """A point attribution result with the uncertainty of its quantities."""

    result: Any
    uncertainty: AttributionUncertainty


def _uncertainty(
    kind: str,
    names: Sequence[str],
    columns: Sequence[Any],
    edges: Sequence[tuple[str, str]],
    params: dict[str, Any],
    *,
    posterior_draws: int,
    bootstrap_replicates: int,
    seed: int,
    threads: int | None,
) -> AttributionUncertainty:
    raw = json.loads(
        _native.attribution_uncertainty(
            list(names),
            list(columns),
            [(str(a), str(b)) for a, b in edges],
            kind,
            json.dumps(params),
            posterior_draws=posterior_draws,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
            threads=threads,
        )
    )
    posterior = raw.get("posterior")
    bootstrap = raw.get("bootstrap")
    return AttributionUncertainty(
        kind=raw["kind"],
        point={entry["name"]: entry["value"] for entry in raw["point"]},
        posterior=(
            None
            if posterior is None
            else PosteriorOverMechanisms(
                draws=posterior["draws"],
                level=posterior["level"],
                names=tuple(posterior["names"]),
                mean=tuple(posterior["mean"]),
                sd=tuple(posterior["sd"]),
                ci_low=tuple(posterior["ci_low"]),
                ci_high=tuple(posterior["ci_high"]),
            )
        ),
        bootstrap=(
            None
            if bootstrap is None
            else RefitBootstrap(
                replicates_ok=bootstrap["replicates_ok"],
                replicates_failed=bootstrap["replicates_failed"],
                se=dict(bootstrap["se"]),
            )
        ),
    )


def _wants(posterior_draws: int, bootstrap_replicates: int) -> bool:
    if posterior_draws and bootstrap_replicates:
        raise ValueError("ask for posterior_draws or bootstrap_replicates, not both")
    return bool(posterior_draws or bootstrap_replicates)


def _combine(point: Any, uncertainty: AttributionUncertainty) -> AttributionWithUncertainty:
    return AttributionWithUncertainty(result=point, uncertainty=uncertainty)


def attribute_distribution_change(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
    root_causes: bool = False,
) -> Any:
    """Shapley attribution of the change of ``outcome`` between two row ranges to mechanisms.

    With ``posterior_draws`` or ``bootstrap_replicates`` the result is an
    :class:`AttributionWithUncertainty` (``root_causes=True`` asks for the same quantities as
    the ranked root-cause attribution).
    """
    point = _native.attribute_distribution_change(
        names, columns, edges, outcome, baseline_start, baseline_end, comparison_start,
        comparison_end, n_samples=n_samples, seed=seed, threads=threads,
    )  # fmt: skip
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {
        "outcome": outcome,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
        "n_samples": n_samples,
    }
    uncertainty = _uncertainty(
        "root_causes" if root_causes else "distribution_change",
        names, columns, edges, params,
        posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
        seed=seed, threads=threads,
    )  # fmt: skip
    return _combine(point, uncertainty)


def attribute_distribution_change_robust(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Robust distribution-change attribution; see :func:`attribute_distribution_change`."""
    point = _native.attribute_distribution_change_robust(
        names, columns, edges, outcome, baseline_start, baseline_end, comparison_start,
        comparison_end, n_samples=n_samples, seed=seed, threads=threads,
    )  # fmt: skip
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {
        "outcome": outcome,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
        "n_samples": n_samples,
    }
    return _combine(
        point,
        _uncertainty(
            "robust_change", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def attribute_structure_change(
    names: list[str],
    columns: Sequence[Any],
    baseline_edges: list[tuple[str, str]],
    comparison_edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Structure-change attribution; see :func:`attribute_distribution_change`."""
    point = _native.attribute_structure_change(
        names, columns, baseline_edges, comparison_edges, outcome, baseline_start,
        baseline_end, comparison_start, comparison_end, n_samples=n_samples, seed=seed,
        threads=threads,
    )  # fmt: skip
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {
        "outcome": outcome,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
        "n_samples": n_samples,
        "comparison_edges": [[str(a), str(b)] for a, b in comparison_edges],
    }
    return _combine(
        point,
        _uncertainty(
            "structure_change", names, columns, baseline_edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def attribute_path_specific(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    path_nodes: list[str] | None = None,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Path decomposition of ``treatment`` on ``outcome``; see
    :func:`attribute_distribution_change` for the uncertainty options."""
    point = _native.attribute_path_specific(
        names, columns, edges, treatment, outcome, path_nodes=path_nodes,
        max_paths=max_paths, max_len=max_len, seed=seed, threads=threads,
    )  # fmt: skip
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {
        "sources": [treatment],
        "outcome": outcome,
        "through": list(path_nodes or []),
        "max_paths": max_paths,
        "max_len": max_len,
    }
    return _combine(
        point,
        _uncertainty(
            "paths", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def attribute_feature_relevance(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcome: str,
    *,
    delta: float = 1.0,
    n_samples: int = 200,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Interventional feature relevance; see :func:`attribute_distribution_change`."""
    point = _native.attribute_feature_relevance(
        names, columns, edges, outcome, delta=delta, n_samples=n_samples, seed=seed,
        threads=threads,
    )  # fmt: skip
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {"outcome": outcome, "delta": delta, "n_samples": n_samples}
    return _combine(
        point,
        _uncertainty(
            "feature_relevance", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def anomaly_attribution(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcomes: list[str],
    *,
    max_units: int = 0,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Anomaly scores of ``outcomes``; with an uncertainty option the result also carries the
    posterior or bootstrap of each target's mean anomaly score."""
    point = _native.anomaly_attribution(names, columns, edges, outcomes, max_units=max_units)
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {"outcomes": list(outcomes), "max_units": max_units}
    return _combine(
        point,
        _uncertainty(
            "anomaly", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def attribute_unit_change(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcome: str,
    *,
    max_units: int = 0,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Unit-level change attribution; the uncertainty is that of each input's mean
    contribution. See :func:`attribute_distribution_change`."""
    point = _native.attribute_unit_change(
        names, columns, edges, outcome, max_units=max_units, seed=seed, threads=threads
    )
    if not _wants(posterior_draws, bootstrap_replicates):
        return point
    params = {"outcome": outcome, "max_units": max_units}
    return _combine(
        point,
        _uncertainty(
            "unit_change", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


def attribution_uncertainty(
    kind: str,
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    params: dict[str, Any],
    *,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
    seed: int = 0,
    threads: int | None = None,
) -> AttributionUncertainty:
    """The uncertainty of an attribution of ``kind`` (``distribution_change``,
    ``root_causes``, ``robust_change``, ``structure_change``, ``paths``, ``feature_relevance``,
    ``anomaly`` or ``unit_change``) with the parameters ``params`` that kind takes."""
    if not _wants(posterior_draws, bootstrap_replicates):
        raise ValueError("ask for posterior_draws or bootstrap_replicates")
    return _uncertainty(
        kind, names, columns, edges, params,
        posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
        seed=seed, threads=threads,
    )  # fmt: skip


def attribute_root_causes(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int | None = None,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
) -> Any:
    """Root causes of the change of ``outcome``: the change attribution and its ranking.

    Without an uncertainty option this returns the ranked contributions; with one it returns an
    :class:`AttributionWithUncertainty` whose ``result`` is that ranking.
    """
    attribution = _native.attribute_distribution_change(
        names, columns, edges, outcome, baseline_start, baseline_end, comparison_start,
        comparison_end, n_samples=n_samples, seed=seed, threads=threads,
    )  # fmt: skip
    ranks = _native.rank_root_causes(attribution, seed=seed, threads=threads)
    if not _wants(posterior_draws, bootstrap_replicates):
        return ranks
    params = {
        "outcome": outcome,
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "comparison_start": comparison_start,
        "comparison_end": comparison_end,
        "n_samples": n_samples,
    }
    return _combine(
        ranks,
        _uncertainty(
            "root_causes", names, columns, edges, params,
            posterior_draws=posterior_draws, bootstrap_replicates=bootstrap_replicates,
            seed=seed, threads=threads,
        ),
    )  # fmt: skip


__all__ = [
    "AnomalyScores",
    "AttributionUncertainty",
    "AttributionWithUncertainty",
    "ChangeAttributionResult",
    "Contribution",
    "FeatureRelevance",
    "MechanismChangeDetection",
    "PosteriorOverMechanisms",
    "RefitBootstrap",
    "anomaly_attribution",
    "attribute_distribution_change",
    "attribute_distribution_change_robust",
    "attribute_feature_relevance",
    "attribute_path_specific",
    "attribute_paths",
    "attribute_root_causes",
    "attribute_structure_change",
    "attribute_unit_change",
    "attribution_uncertainty",
    "mechanism_change_detection",
    "rank_root_causes",
]
