"""High-level estimation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ._data import as_columns, as_multi_env_columns, try_as_arrow_c_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
    AteAnalysisResult,
    analyze as _analyze_temporal,
    analyze_ate as _analyze_ate,
    analyze_ate_admg as _analyze_ate_admg,
    analyze_ate_arrow_c as _analyze_ate_arrow_c,
    analyze_ate_cpdag as _analyze_ate_cpdag,
    analyze_ate_discover as _analyze_ate_discover,
    analyze_ate_pag as _analyze_ate_pag,
    analyze_distribution as _analyze_distribution,
    analyze_events as _analyze_events,
    analyze_panel as _analyze_panel,
    analyze_panel_discover as _analyze_panel_discover,
    analyze_path_specific as _analyze_path_specific,
    analyze_temporal_discover as _analyze_temporal_discover,
    analyze_temporal_pag as _analyze_temporal_pag,
)
from .data import EventFrame, MultiEnvFrame, PanelFrame
from .discovery import (
    FCI,
    GES,
    JPCMCIPlus,
    LPCMCI,
    LiNGAM,
    NOTEARS,
    PC,
    PCMCI,
    PCMCIPlus,
    RFCI,
    RPCMCI,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from .inference import Bayesian, Frequentist
from .query import (
    AverageEffect,
    InterventionalDistribution,
    PathSpecificEffect,
    PulseEffect,
    SustainedEffect,
)

# Preferred name for the native temporal DTO.
NativeAnalysisResult = TemporalAnalysisResult


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
    artifact: bytes | list[int] | None = None


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


def _wrap_temporal(raw: TemporalAnalysisResult) -> AnalysisResult:
    # Mirror static ate_result_from_analysis: never claim pass when nothing ran.
    ran = raw.refutation_count > 0
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
            passed=False if not ran else True,
            ran=ran,
            count=raw.refutation_count,
        ),
        performance=PerformanceView(
            plan_id=raw.plan_id,
            modality=raw.modality,
            peak_memory_bytes=raw.peak_memory_bytes,
        ),
        diagnostics=list(raw.diagnostics),
        provenance={
            "node_count": raw.provenance_node_count,
            "worker_threads": getattr(raw, "worker_threads", None),
            "expected_python_crossings": getattr(raw, "expected_python_crossings", None),
        },
        _raw=raw,
    )


_STATIC_DISCOVERY = (PC, GES, LiNGAM, NOTEARS, FCI, RFCI)
_TEMPORAL_DISCOVERY = (PCMCI, PCMCIPlus, LPCMCI, JPCMCIPlus, RPCMCI)


def _discovery_algorithm(discovery: Any) -> dict[str, Any]:
    if isinstance(discovery, PCMCI):
        return {
            "algorithm": "pcmci",
            "max_lag": discovery.max_lag,
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
        }
    if isinstance(discovery, PCMCIPlus):
        return {
            "algorithm": "pcmci_plus",
            "max_lag": discovery.max_lag,
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
        }
    if isinstance(discovery, LPCMCI):
        return {
            "algorithm": "lpcmci",
            "max_lag": discovery.max_lag,
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
        }
    if isinstance(discovery, JPCMCIPlus):
        return {
            "algorithm": "jpcmci_plus",
            "max_lag": discovery.max_lag,
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "context_names": list(discovery.context_names),
            "include_space_dummy": discovery.include_space_dummy,
            "include_time_dummy": discovery.include_time_dummy,
            "space_dummy_ci": discovery.space_dummy_ci,
            "time_dummy_encoding": discovery.time_dummy_encoding,
            "time_dummy_ci": discovery.time_dummy_ci,
        }
    if isinstance(discovery, RPCMCI):
        return {
            "algorithm": "rpcmci",
            "max_lag": discovery.max_lag,
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
        }
    if isinstance(discovery, PC):
        return {
            "algorithm": "pc",
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "max_cond_size": discovery.max_cond_size,
        }
    if isinstance(discovery, GES):
        return {
            "algorithm": "ges",
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "max_cond_size": discovery.max_cond_size,
        }
    if isinstance(discovery, LiNGAM):
        return {
            "algorithm": "lingam",
            "prune_threshold": discovery.prune_threshold,
            "max_cond_size": discovery.max_cond_size,
            "alpha": 0.05,
            "fdr": True,
            "ci": "parcorr",
        }
    if isinstance(discovery, NOTEARS):
        return {
            "algorithm": "notears",
            "lambda": discovery.l1,
            "threshold": discovery.threshold,
            "standardize": discovery.standardize,
            "max_cond_size": discovery.max_cond_size,
            "alpha": 0.05,
            "fdr": True,
            "ci": "parcorr",
        }
    if isinstance(discovery, FCI):
        return {
            "algorithm": "fci",
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "max_cond_size": discovery.max_cond_size,
        }
    if isinstance(discovery, RFCI):
        return {
            "algorithm": "rfci",
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "max_cond_size": discovery.max_cond_size,
        }
    raise TypeError(f"unsupported discovery config: {type(discovery)!r}")


def _static_edges(
    graph: Dag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise ValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    return [(str(a), str(b)) for a, b in graph]  # type: ignore[misc]


def _lagged_edges(
    graph: TemporalDag | Sequence[tuple[str, int, str, int]] | None,
) -> list[tuple[str, int, str, int]]:
    if graph is None:
        raise ValueError("graph= lagged edges are required")
    if isinstance(graph, TemporalDag):
        return [
            (str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph.edges()
        ]
    return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph]  # type: ignore[misc]


def _refute_requested(refute: bool | str) -> bool:
    """True when the caller asked for any non-empty refute suite."""
    if isinstance(refute, bool):
        return refute
    key = str(refute).strip().lower()
    return key not in ("", "none", "off", "false", "0")


def _reject_unsupported_temporal(
    *,
    inference: Frequentist | Bayesian | None,
    refute: bool | str,
    validators: Sequence[Any] | None,
) -> None:
    if isinstance(inference, Bayesian):
        raise TypeError(
            "inference=Bayesian(...) is not supported for temporal queries; "
            "omit inference or use Frequentist()"
        )
    if _refute_requested(refute):
        raise TypeError(
            "refute= is not supported for temporal queries yet; pass refute=False"
        )
    if validators is not None:
        raise TypeError(
            "validators= is not supported for temporal queries yet"
        )


def analyze(
    data: Mapping[str, Any] | Any | Sequence[Mapping[str, Any] | Any],
    *,
    query: (
        AverageEffect
        | PulseEffect
        | SustainedEffect
        | InterventionalDistribution
        | PathSpecificEffect
    ),
    graph: (
        Dag
        | Cpdag
        | Pag
        | Admg
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
        | Sequence[tuple[str, str]]
        | Sequence[tuple[str, int, str, int]]
        | None
    ) = None,
    discovery: Any | None = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | None = None,
    estimator: str | None = None,
    refute: bool | Literal["full", "placebo", "none"] = True,
    validators: Sequence[Any] | None = None,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
    regimes: Sequence[int] | None = None,
) -> AnalysisResult:
    """Identify then estimate a causal effect.

    Parameters
    ----------
    data:
        Mapping of column name → 1-d float array, a pandas ``DataFrame``,
        Arrow CDI exporters (PyArrow columns / table), or a
        ``causal.data`` frame (``EventFrame`` / ``PanelFrame`` / ``MultiEnvFrame``).
        For ``discovery=JPCMCIPlus(...)``, pass a sequence of environment frames
        or a ``MultiEnvFrame``.
    query:
        ``AverageEffect``, ``PulseEffect`` / ``SustainedEffect``,
        ``InterventionalDistribution``, or ``PathSpecificEffect``.
    graph:
        ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` / ``TemporalDag`` /
        ``TemporalCpdag`` / ``TemporalPag``, or an edge list. Lagged edges
        ``(from, from_lag, to, to_lag)`` are required for temporal queries
        without ``discovery``. Fully oriented CPDAGs run as DAGs; incomplete
        CPDAGs require review. ADMGs without bidirected edges coerce to DAGs;
        ADMGs with latents use general ID + functional effect.
    discovery:
        Static: ``PC`` / ``GES`` / ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI``.
        Temporal: ``PCMCI`` / ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``.
    """
    inference = inference or Frequentist()

    if isinstance(query, InterventionalDistribution):
        if discovery is not None:
            raise ValueError("InterventionalDistribution does not support discovery=")
        names, columns = as_columns(data)  # type: ignore[arg-type]
        edges = _static_edges(graph)  # type: ignore[arg-type]
        raw = _analyze_distribution(
            names,
            columns,
            edges,
            query.outcome,
            dict(query.interventions),
            conditioning=list(query.conditioning) or None,
            seed=seed,
            threads=threads,
        )
        return _wrap_ate(raw)

    if isinstance(query, PathSpecificEffect):
        if discovery is not None:
            raise ValueError("PathSpecificEffect does not support discovery=")
        names, columns = as_columns(data)  # type: ignore[arg-type]
        edges = _static_edges(graph)  # type: ignore[arg-type]
        raw = _analyze_path_specific(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            path_nodes=list(query.path_nodes) if query.path_nodes is not None else None,
            max_paths=query.max_paths,
            max_len=query.max_len,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_ate(raw)

    if discovery is not None and isinstance(discovery, _STATIC_DISCOVERY):
        if not isinstance(query, AverageEffect):
            raise ValueError(
                f"discovery={type(discovery).__name__}(...) requires AverageEffect"
            )
        names, columns = as_columns(data)  # type: ignore[arg-type]
        cfg = _discovery_algorithm(discovery)
        inference_s = None
        n_draws = 1000
        prior_scale = 10.0
        if isinstance(inference, Bayesian):
            inference_s = "bayesian"
            n_draws = inference.n_draws
            prior_scale = inference.prior_scale
        raw = _analyze_ate_discover(
            names,
            columns,
            query.treatment,
            query.outcome,
            algorithm=cfg["algorithm"],
            alpha=cfg.get("alpha", 0.05),
            fdr=cfg.get("fdr", True),
            max_cond_size=cfg.get("max_cond_size", 2),
            prune_threshold=cfg.get("prune_threshold", 0.0),
            l1=cfg.get("lambda", 0.1),
            threshold=cfg.get("threshold", 0.3),
            standardize=cfg.get("standardize", True),
            accept_discovered=accept_discovered,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=identifier,
            estimator=estimator,
            inference=inference_s,
            n_draws=n_draws,
            prior_scale=prior_scale,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            ci=cfg.get("ci"),
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_ate(raw)

    if discovery is not None and isinstance(query, AverageEffect):
        raise ValueError(
            "AverageEffect with discovery= requires a static algorithm "
            "(PC/GES/LiNGAM/NOTEARS/FCI/RFCI); temporal discovery needs "
            "PulseEffect/SustainedEffect"
        )

    if isinstance(query, AverageEffect):
        inference_s = None
        n_draws = 1000
        prior_scale = 10.0
        if isinstance(inference, Bayesian):
            inference_s = "bayesian"
            n_draws = inference.n_draws
            prior_scale = inference.prior_scale
        common = dict(
            treatment=query.treatment,
            outcome=query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=identifier,
            estimator=estimator,
            inference=inference_s,
            n_draws=n_draws,
            prior_scale=prior_scale,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        if isinstance(graph, Pag):
            names, columns = as_columns(data)  # type: ignore[arg-type]
            return _wrap_ate(_analyze_ate_pag(names, columns, graph, **common))
        if isinstance(graph, Cpdag):
            names, columns = as_columns(data)  # type: ignore[arg-type]
            return _wrap_ate(_analyze_ate_cpdag(names, columns, graph, **common))
        if isinstance(graph, Admg):
            names, columns = as_columns(data)  # type: ignore[arg-type]
            return _wrap_ate(_analyze_ate_admg(names, columns, graph, **common))
        edges = _static_edges(graph)  # type: ignore[arg-type]
        arrow = try_as_arrow_c_columns(data)
        ate_kwargs = dict(edges=edges, **common)
        if arrow is not None:
            names, columns = arrow
            raw = _analyze_ate_arrow_c(names, columns, **ate_kwargs)
        else:
            names, columns = as_columns(data)  # type: ignore[arg-type]
            raw = _analyze_ate(names, columns, **ate_kwargs)
        return _wrap_ate(raw)

    if isinstance(query, (PulseEffect, SustainedEffect)):
        policy = "sustained" if isinstance(query, SustainedEffect) else "pulse"
        _reject_unsupported_temporal(
            inference=inference, refute=refute, validators=validators
        )
        if isinstance(data, EventFrame):
            if discovery is not None:
                raise ValueError("EventFrame does not support discovery= yet; supply a TemporalDag")
            lagged = _lagged_edges(graph)  # type: ignore[arg-type]
            raw = _analyze_events(
                data.names,
                data.columns,
                data.event_times_ns.tolist(),
                data.align_interval_ns,
                lagged,
                query.treatment,
                query.outcome,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
            )
            return _wrap_temporal(raw)
        if isinstance(data, PanelFrame):
            if discovery is not None:
                if not isinstance(discovery, JPCMCIPlus):
                    raise TypeError(
                        "PanelFrame discovery currently supports only JPCMCIPlus(...)"
                    )
                cfg = _discovery_algorithm(discovery)
                raw = _analyze_panel_discover(
                    data.names,
                    data.unit_columns,
                    data.unit_ids,
                    query.treatment,
                    query.outcome,
                    max_lag=cfg["max_lag"],
                    alpha=cfg["alpha"],
                    fdr=cfg["fdr"],
                    accept_discovered=accept_discovered,
                    treatment_lag=query.treatment_lag,
                    horizon_steps=query.horizon_steps,
                    active_level=query.active_level,
                    policy=policy,
                    seed=seed,
                    bootstrap=bootstrap,
                    threads=threads,
                    context_names=cfg["context_names"],
                    include_space_dummy=cfg["include_space_dummy"],
                    include_time_dummy=cfg["include_time_dummy"],
                    space_dummy_ci=cfg["space_dummy_ci"] in ("multivariate", "multivariate_block", "block", True),
                    time_dummy_encoding=cfg["time_dummy_encoding"],
                    time_dummy_ci=cfg["time_dummy_ci"] in ("multivariate", "multivariate_block", "block", True),
                )
                return _wrap_temporal(raw)
            lagged = _lagged_edges(graph)  # type: ignore[arg-type]
            raw = _analyze_panel(
                data.names,
                data.unit_columns,
                data.unit_ids,
                lagged,
                query.treatment,
                query.outcome,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
            )
            return _wrap_temporal(raw)
        if isinstance(data, MultiEnvFrame):
            if discovery is None or not isinstance(discovery, JPCMCIPlus):
                raise TypeError(
                    "MultiEnvFrame requires discovery=JPCMCIPlus(...)"
                )
            cfg = _discovery_algorithm(discovery)
            raw = _analyze_temporal_discover(
                data.names,
                data.env_columns[0],
                query.treatment,
                query.outcome,
                algorithm="jpcmci_plus",
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                env_columns=data.env_columns,
                context_names=cfg["context_names"],
                include_space_dummy=cfg["include_space_dummy"],
                include_time_dummy=cfg["include_time_dummy"],
                space_dummy_ci=cfg["space_dummy_ci"],
                time_dummy_encoding=cfg["time_dummy_encoding"],
                time_dummy_ci=cfg["time_dummy_ci"],
                ci=cfg.get("ci"),
            )
            return _wrap_temporal(raw)
        if discovery is not None:
            if not isinstance(discovery, _TEMPORAL_DISCOVERY):
                raise TypeError(
                    f"temporal discovery expects PCMCI-family config, got {type(discovery)!r}"
                )
            cfg = _discovery_algorithm(discovery)
            algo = cfg["algorithm"]
            if algo == "jpcmci_plus":
                if not isinstance(data, Sequence) or isinstance(data, (str, bytes, Mapping)):
                    raise TypeError(
                        "discovery=JPCMCIPlus(...) requires data as a sequence of "
                        "environment mappings/DataFrames"
                    )
                names, env_columns = as_multi_env_columns(data)
                raw = _analyze_temporal_discover(
                    names,
                    env_columns[0],
                    query.treatment,
                    query.outcome,
                    algorithm=algo,
                    max_lag=cfg["max_lag"],
                    alpha=cfg["alpha"],
                    fdr=cfg["fdr"],
                    accept_discovered=accept_discovered,
                    treatment_lag=query.treatment_lag,
                    horizon_steps=query.horizon_steps,
                    active_level=query.active_level,
                    policy=policy,
                    seed=seed,
                    bootstrap=bootstrap,
                    threads=threads,
                    env_columns=env_columns,
                    context_names=cfg["context_names"],
                    include_space_dummy=cfg["include_space_dummy"],
                    include_time_dummy=cfg["include_time_dummy"],
                    space_dummy_ci=cfg["space_dummy_ci"],
                    time_dummy_encoding=cfg["time_dummy_encoding"],
                    time_dummy_ci=cfg["time_dummy_ci"],
                    ci=cfg.get("ci"),
                )
                return _wrap_temporal(raw)
            if algo == "rpcmci":
                if regimes is None:
                    raise ValueError("discovery=RPCMCI(...) requires regimes=[…] labels")
                names, columns = as_columns(data)  # type: ignore[arg-type]
                raw = _analyze_temporal_discover(
                    names,
                    columns,
                    query.treatment,
                    query.outcome,
                    algorithm=algo,
                    max_lag=cfg["max_lag"],
                    alpha=cfg["alpha"],
                    fdr=cfg["fdr"],
                    accept_discovered=accept_discovered,
                    treatment_lag=query.treatment_lag,
                    horizon_steps=query.horizon_steps,
                    active_level=query.active_level,
                    policy=policy,
                    seed=seed,
                    bootstrap=bootstrap,
                    threads=threads,
                    regimes=list(regimes),
                    ci=cfg.get("ci"),
                )
                return _wrap_temporal(raw)
            names, columns = as_columns(data)  # type: ignore[arg-type]
            raw = _analyze_temporal_discover(
                names,
                columns,
                query.treatment,
                query.outcome,
                algorithm=algo,
                max_lag=cfg["max_lag"],
                alpha=cfg["alpha"],
                fdr=cfg["fdr"],
                accept_discovered=accept_discovered,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
                ci=cfg.get("ci"),
            )
            return _wrap_temporal(raw)
        names, columns = as_columns(data)  # type: ignore[arg-type]
        if isinstance(graph, TemporalPag):
            raw = _analyze_temporal_pag(
                names,
                columns,
                graph,
                query.treatment,
                query.outcome,
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                policy=policy,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
            )
            return _wrap_temporal(raw)
        if isinstance(graph, TemporalCpdag):
            try:
                graph = graph.try_into_temporal_dag()
            except Exception as exc:  # noqa: BLE001 — surface orientation failures
                raise ValueError(
                    "TemporalCpdag has undirected/conflict marks; orient edges "
                    "(try_into_temporal_dag) before analyze, or use discovery review"
                ) from exc
        lagged = _lagged_edges(graph)  # type: ignore[arg-type]
        raw = _analyze_temporal(
            names,
            columns,
            lagged,
            query.treatment,
            query.outcome,
            treatment_lag=query.treatment_lag,
            horizon_steps=query.horizon_steps,
            active_level=query.active_level,
            policy=policy,
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
    "TemporalAnalysisResult",
    "ValidationView",
    "analyze",
]
