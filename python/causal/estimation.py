"""High-level estimation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ._native import (
    AnalysisResult as NativeAnalysisResult,
    AteAnalysisResult,
    analyze as _analyze_temporal,
    analyze_ate as _analyze_ate,
    analyze_temporal_discover as _analyze_temporal_discover,
)
from .discovery import JPCMCIPlus, LPCMCI, PCMCI, PCMCIPlus, RPCMCI
from .inference import Bayesian, Frequentist
from .query import AverageEffect, PulseEffect, SustainedEffect


@dataclass(frozen=True)
class IdentificationView:
    status: str
    method: str
    adjustment_set: list[str]
    assumption_count: int
    derivation_step_count: int


@dataclass(frozen=True)
class EstimateView:
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    estimator_id: str
    method: str
    overlap_ess: float | None = None
    overlap_propensity_min: float | None = None


@dataclass(frozen=True)
class PosteriorView:
    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    n_draws: int | None
    p_below_zero: float | None
    backend: str | None
    artifact: list[int] | None = None


@dataclass(frozen=True)
class ValidationView:
    passed: bool
    ran: bool
    count: int


@dataclass(frozen=True)
class PerformanceView:
    plan_id: str | None = None
    modality: str | None = None
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Nested analysis result matching the Rust facade sections."""

    identification: IdentificationView
    estimate: EstimateView
    posterior: PosteriorView | None
    validation: ValidationView
    performance: PerformanceView
    diagnostics: list[str]
    provenance: dict[str, Any]
    _raw: Any = None

    @property
    def ate(self) -> float:
        return self.estimate.ate


def _as_columns(
    data: Mapping[str, Any] | Any,
) -> tuple[list[str], list[NDArray[np.float64]]]:
    if isinstance(data, Mapping):
        names = list(data.keys())
        cols = [_to_f64(data[n]) for n in names]
        return names, cols
    if hasattr(data, "columns") and hasattr(data, "to_numpy"):
        names = [str(c) for c in data.columns]
        cols = [_to_f64(data[c].to_numpy()) for c in data.columns]
        return names, cols
    raise TypeError(
        "data must be a mapping of name→array or a pandas DataFrame; "
        f"got {type(data)!r}"
    )


def _to_f64(arr: Any) -> NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"expected 1-d column, got shape {a.shape}")
    if a.dtype == object:
        raise TypeError("object-dtype columns are not supported")
    return a


def _wrap_ate(raw: AteAnalysisResult) -> AnalysisResult:
    posterior = None
    if raw.posterior_n_draws is not None:
        posterior = PosteriorView(
            effect_mean=raw.posterior_effect_mean,
            effect_sd=raw.posterior_effect_sd,
            q025=raw.posterior_q025,
            q975=raw.posterior_q975,
            n_draws=raw.posterior_n_draws,
            p_below_zero=raw.posterior_p_below_zero,
            backend=raw.posterior_backend,
            artifact=raw.posterior_artifact,
        )
    return AnalysisResult(
        identification=IdentificationView(
            status=raw.identification_status,
            method=raw.method,
            adjustment_set=list(raw.adjustment_set),
            assumption_count=raw.assumption_count,
            derivation_step_count=raw.derivation_step_count,
        ),
        estimate=EstimateView(
            ate=raw.ate,
            se_analytic=raw.se_analytic,
            se_bootstrap=raw.se_bootstrap,
            estimator_id=raw.estimator_id,
            method=raw.method,
            overlap_ess=raw.overlap_ess,
            overlap_propensity_min=raw.overlap_propensity_min,
        ),
        posterior=posterior,
        validation=ValidationView(
            passed=raw.refutation_passed,
            ran=raw.refutation_ran,
            count=raw.refutation_count,
        ),
        performance=PerformanceView(
            plan_id=raw.plan_id,
            modality=raw.modality,
            peak_memory_bytes=raw.peak_memory_bytes,
        ),
        diagnostics=list(raw.diagnostics),
        provenance={"node_count": raw.provenance_node_count},
        _raw=raw,
    )


def _wrap_temporal(raw: NativeAnalysisResult) -> AnalysisResult:
    return AnalysisResult(
        identification=IdentificationView(
            status=raw.identification_status,
            method=raw.method,
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        ),
        estimate=EstimateView(
            ate=raw.ate,
            se_analytic=raw.se_analytic,
            se_bootstrap=raw.se_bootstrap,
            estimator_id="",
            method=raw.method,
        ),
        posterior=None,
        validation=ValidationView(
            passed=True,
            ran=raw.refutation_count > 0,
            count=raw.refutation_count,
        ),
        performance=PerformanceView(
            plan_id=raw.plan_id,
            modality=raw.modality,
            peak_memory_bytes=raw.peak_memory_bytes,
        ),
        diagnostics=list(raw.diagnostics),
        provenance={"node_count": raw.provenance_node_count},
        _raw=raw,
    )


def _discovery_algorithm(discovery: Any) -> tuple[str, int, float, bool]:
    if isinstance(discovery, PCMCI):
        return "pcmci", discovery.max_lag, discovery.alpha, discovery.fdr
    if isinstance(discovery, PCMCIPlus):
        return "pcmci_plus", discovery.max_lag, discovery.alpha, discovery.fdr
    if isinstance(discovery, LPCMCI):
        return "lpcmci", discovery.max_lag, discovery.alpha, discovery.fdr
    if isinstance(discovery, (JPCMCIPlus, RPCMCI)):
        raise NotImplementedError(
            f"{type(discovery).__name__} via analyze(discovery=...) is not wired; "
            "call discover_jpcmci_plus / discover_rpcmci directly"
        )
    raise TypeError(f"unsupported discovery config: {type(discovery)!r}")


def analyze(
    data: Mapping[str, Any] | Any,
    *,
    query: AverageEffect | PulseEffect | SustainedEffect,
    graph: Sequence[tuple[str, str]] | Sequence[tuple[str, int, str, int]] | None = None,
    discovery: Any | None = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | None = None,
    estimator: str | None = None,
    refute: bool = True,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AnalysisResult:
    """Identify then estimate a causal effect.

    Parameters
    ----------
    data:
        Mapping of column name → 1-d float array, or a pandas ``DataFrame``.
    query:
        ``AverageEffect`` (static) or ``PulseEffect`` / ``SustainedEffect`` (temporal).
    graph:
        Supplied edges. For temporal queries without ``discovery``, lagged edges
        ``(from, from_lag, to, to_lag)`` are required.
    discovery:
        ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` for temporal discover→estimate
        (auto-accepts oriented edges when ``accept_discovered=True``).
    inference:
        ``Frequentist()`` (default) or ``Bayesian(...)``.
    """
    if discovery is not None and isinstance(query, AverageEffect):
        raise ValueError("discovery= requires a temporal PulseEffect/SustainedEffect query")

    names, columns = _as_columns(data)
    inference = inference or Frequentist()

    if isinstance(query, AverageEffect):
        if graph is None:
            raise ValueError("graph= edge list is required for AverageEffect")
        edges = [(str(a), str(b)) for a, b in graph]  # type: ignore[misc]
        inference_s = None
        n_draws = 1000
        prior_scale = 10.0
        if isinstance(inference, Bayesian):
            inference_s = "bayesian"
            n_draws = inference.n_draws
            prior_scale = inference.prior_scale
        raw = _analyze_ate(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            identifier=identifier,
            estimator=estimator,
            inference=inference_s,
            n_draws=n_draws,
            prior_scale=prior_scale,
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_ate(raw)

    if isinstance(query, (PulseEffect, SustainedEffect)):
        if discovery is not None:
            algo, max_lag, alpha, fdr = _discovery_algorithm(discovery)
            raw = _analyze_temporal_discover(
                names,
                columns,
                query.treatment,
                query.outcome,
                algorithm=algo,
                max_lag=max_lag,
                alpha=alpha,
                fdr=fdr,
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
            )
            return _wrap_temporal(raw)
        if graph is None:
            raise ValueError(
                "temporal queries require graph= lagged edges or discovery=PCMCI(...)"
            )
        lagged = [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph]  # type: ignore[misc]
        raw = _analyze_temporal(
            names,
            columns,
            lagged,
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_temporal(raw)

    raise TypeError(f"unsupported query type: {type(query)!r}")


__all__ = [
    "AnalysisResult",
    "AteAnalysisResult",
    "EstimateView",
    "IdentificationView",
    "NativeAnalysisResult",
    "PerformanceView",
    "PosteriorView",
    "ValidationView",
    "analyze",
]
