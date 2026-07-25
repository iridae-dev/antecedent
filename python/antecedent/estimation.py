"""High-level estimation entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._data import as_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
)
from ._native import (
    AteAnalysisResult,
)
from ._native import (
    PreparedAnalysis as _NativePreparedAnalysis,
)
from ._native import (
    analyze_ate_many as _analyze_ate_many,
)
from ._native import (
    identify_ate as _identify_ate,
)
from .discovery import (
    FCI,
    GES,
    NOTEARS,
    PC,
    RFCI,
    CiScreenedPosterior,
    ExactDagPosterior,
    LiNGAM,
    OrderMcmc,
    StructureMcmc,
    cpdag_oriented_edges,
    discovery_algorithm,
    discovery_to_dag,
    graph_posterior_map_edges,
    run_static_discovery,
)
from .graph import Cpdag, Dag, TemporalDag
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .query import (
    AverageEffect,
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
class MediationView:
    total: float | None
    direct: float | None
    mediated: float | None


@dataclass(frozen=True)
class EstimateView:
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    estimator_id: str
    method: str
    overlap_ess: float | None = None
    overlap_propensity_min: float | None = None
    mediation: MediationView | None = None


@dataclass(frozen=True)
class ConflictSummaryView:
    """Applied external-prior alphas after conflict shrink."""

    source_ids: list[str]
    alphas_requested: list[float]
    alphas_applied: list[float]


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
    unidentified_mass: float | None = None
    envelope: EffectEnvelope | None = None
    conflict: ConflictSummaryView | None = None


@dataclass(frozen=True)
class EffectEnvelope:
    """Mixture effect posterior over weighted graphs (PAG / graph-posterior path)."""

    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    unidentified_mass: float
    n_draws: int | None
    backend: str | None = None


@dataclass(frozen=True)
class PredictiveCheckReport:
    """Prior or posterior predictive check summary."""

    kind: str
    observed: float
    predictive_mean: float
    predictive_sd: float
    p_value: float
    n_sims: int


@dataclass(frozen=True)
class PriorSensitivityReport:
    """Prior sensitivity grid (Bayesian + ``refute="full"``).

    Isotropic mode fills ``scales``; external prior-bank mode fills ``alphas``
    (multipliers on post-conflict applied α). Exactly one mode is active.
    """

    scales: list[float]
    effect_means: list[float]
    effect_sds: list[float]
    alphas: list[float] | None = None


@dataclass(frozen=True)
class ValidationView:
    passed: bool
    ran: bool
    count: int
    prior_predictive: PredictiveCheckReport | None = None
    posterior_predictive: PredictiveCheckReport | None = None
    prior_sensitivity: PriorSensitivityReport | None = None


@dataclass(frozen=True)
class PerformanceView:
    plan_id: str | None = None
    modality: str | None = None
    peak_memory_bytes: int | None = None
    latency_mode: str | None = None
    wall_time_ns: int | None = None
    bootstrap_replicates_requested: int | None = None
    bootstrap_replicates_ok: int | None = None
    n_draws: int | None = None
    cancelled: bool = False
    early_stopped: bool = False
    stage_timings: dict[str, int] | None = None


@dataclass(frozen=True)
class PlanView:
    """Logical-plan summary (semantics; inspect before/after estimate)."""

    plan_id: str
    modality: str | None = None
    discovery_algorithm: str | None = None
    graph_review_required: bool = False
    identifier: str | None = None
    estimator: str | None = None
    validation_suite: str | None = None


@dataclass(frozen=True)
class PhysicalPlanView:
    """Physical-plan highlights from prepare (layouts / threads / kernels)."""

    plan_id: str
    estimated_peak_memory_bytes: int | None = None
    workspace_bytes: int | None = None
    batch_size: int | None = None
    worker_threads: int = 0
    expected_python_crossings: int = 0
    deterministic_reductions: bool = True
    kernels: str | None = None


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
    mediation: MediationView | None = None
    plan: PlanView | None = None
    _raw: Any = None
    _prepared: Any = None

    @property
    def effect(self) -> float:
        """Primary scalar effect (mediation total when present, else estimate ATE/mean)."""
        if self.mediation is not None and self.mediation.total is not None:
            return float(self.mediation.total)
        if self.estimate.mediation is not None and self.estimate.mediation.total is not None:
            return float(self.estimate.mediation.total)
        return self.estimate.ate

    @property
    def ate(self) -> float:
        """Alias for :attr:`effect` (prefer ``effect`` for non-ATE queries)."""
        return self.effect

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Re-estimate on new data via the retained prepared handle.

        Only results from :meth:`PreparedAnalysis.estimate` / ``refresh`` support
        this. One-shot :func:`analyze` results raise ``TypeError``.
        """
        if self._prepared is None:
            raise TypeError(
                "AnalysisResult.refresh requires a result from PreparedAnalysis; "
                "use PreparedAnalysis.prepare(...) then estimate/refresh"
            )
        return self._prepared.estimate(data, seed=seed, threads=threads)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int = 1,
        threads: int = 1,
        cancel: Any | None = None,
    ) -> AnalysisResult:
        """Second-click refute via the retained prepared handle."""
        if self._prepared is None:
            raise TypeError(
                "AnalysisResult.refute requires a result from PreparedAnalysis; "
                "use PreparedAnalysis.prepare(...) then estimate"
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        return self._prepared.refute(data, suite, seed=seed, threads=threads, cancel=cancel)


def _plan_from_raw(raw: Any) -> PlanView:
    return PlanView(
        plan_id=str(getattr(raw, "plan_id", "") or ""),
        modality=getattr(raw, "modality", None),
        discovery_algorithm=getattr(raw, "discovery_algorithm", None),
        graph_review_required=bool(getattr(raw, "graph_review_required", False)),
        identifier=getattr(raw, "plan_identifier", None),
        estimator=getattr(raw, "plan_estimator", None)
        or (getattr(raw, "estimator_id", None) or None),
        validation_suite=getattr(raw, "validation_suite", None),
    )


def _wrap_ate(raw: AteAnalysisResult, prepared: Any | None = None) -> AnalysisResult:
    def _conflict_from_raw(r: AteAnalysisResult) -> ConflictSummaryView | None:
        ids = getattr(r, "conflict_source_ids", None)
        if ids is None:
            return None
        return ConflictSummaryView(
            source_ids=list(ids),
            alphas_requested=list(r.conflict_alphas_requested or []),
            alphas_applied=list(r.conflict_alphas_applied or []),
        )

    posterior = None
    if raw.posterior_n_draws is not None:
        mass = getattr(raw, "posterior_unidentified_mass", None)
        envelope = None
        if mass is not None and float(mass) > 0.0:
            envelope = EffectEnvelope(
                effect_mean=raw.posterior_effect_mean,
                effect_sd=raw.posterior_effect_sd,
                q025=raw.posterior_q025,
                q975=raw.posterior_q975,
                unidentified_mass=float(mass),
                n_draws=raw.posterior_n_draws,
                backend=raw.posterior_backend,
            )
        posterior = PosteriorView(
            effect_mean=raw.posterior_effect_mean,
            effect_sd=raw.posterior_effect_sd,
            q025=raw.posterior_q025,
            q975=raw.posterior_q975,
            n_draws=raw.posterior_n_draws,
            p_below_zero=raw.posterior_p_below_zero,
            backend=raw.posterior_backend,
            artifact=raw.posterior_artifact,
            unidentified_mass=None if mass is None else float(mass),
            envelope=envelope,
            conflict=_conflict_from_raw(raw),
        )
    prior_predictive = None
    if raw.prior_ppc_p_value is not None:
        assert raw.prior_ppc_observed is not None
        assert raw.prior_ppc_predictive_mean is not None
        assert raw.prior_ppc_predictive_sd is not None
        assert raw.prior_ppc_n_sims is not None
        prior_predictive = PredictiveCheckReport(
            kind="prior_predictive",
            observed=float(raw.prior_ppc_observed),
            predictive_mean=float(raw.prior_ppc_predictive_mean),
            predictive_sd=float(raw.prior_ppc_predictive_sd),
            p_value=float(raw.prior_ppc_p_value),
            n_sims=int(raw.prior_ppc_n_sims),
        )
    posterior_predictive = None
    if raw.posterior_ppc_p_value is not None:
        assert raw.posterior_ppc_observed is not None
        assert raw.posterior_ppc_predictive_mean is not None
        assert raw.posterior_ppc_predictive_sd is not None
        assert raw.posterior_ppc_n_sims is not None
        posterior_predictive = PredictiveCheckReport(
            kind="posterior_predictive",
            observed=float(raw.posterior_ppc_observed),
            predictive_mean=float(raw.posterior_ppc_predictive_mean),
            predictive_sd=float(raw.posterior_ppc_predictive_sd),
            p_value=float(raw.posterior_ppc_p_value),
            n_sims=int(raw.posterior_ppc_n_sims),
        )
    prior_sensitivity = None
    means = raw.prior_sensitivity_means
    if means is not None:
        alphas_raw = raw.prior_sensitivity_alphas
        scales_raw = raw.prior_sensitivity_scales
        sds = raw.prior_sensitivity_sds
        prior_sensitivity = PriorSensitivityReport(
            scales=list(scales_raw or ()),
            effect_means=list(means),
            effect_sds=list(sds or ()),
            alphas=None if alphas_raw is None else list(alphas_raw),
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
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
            prior_sensitivity=prior_sensitivity,
        ),
        performance=PerformanceView(
            plan_id=raw.plan_id,
            modality=raw.modality,
            peak_memory_bytes=raw.peak_memory_bytes,
            latency_mode=getattr(raw, "latency_mode", None),
            wall_time_ns=getattr(raw, "wall_time_ns", None),
            bootstrap_replicates_requested=getattr(raw, "bootstrap_replicates_requested", None),
            bootstrap_replicates_ok=getattr(raw, "bootstrap_replicates_ok", None),
            n_draws=getattr(raw, "n_draws_effort", None),
            cancelled=bool(getattr(raw, "cancelled", False)),
            early_stopped=bool(getattr(raw, "early_stopped", False)),
            stage_timings={str(k): int(v) for k, v in (getattr(raw, "stage_timings", None) or [])}
            or None,
        ),
        diagnostics=list(raw.diagnostics),
        provenance={"node_count": raw.provenance_node_count},
        plan=_plan_from_raw(raw),
        _raw=raw,
        _prepared=prepared,
    )


def _resolve_latency_budget(
    latency: Latency | Literal["interactive", "standard", "report"] | str | None,
    bootstrap: int | None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | str,
) -> tuple[int, bool | Literal["full", "placebo", "none", "cheap"]]:
    """Map latency tier to bootstrap/refute; explicit bootstrap wins when set."""
    if isinstance(latency, Latency):
        latency = str(latency)
    if isinstance(refute, Refute):
        refute = str(refute)
    if latency is None:
        return (50 if bootstrap is None else bootstrap, refute)  # type: ignore[return-value]
    key = str(latency).strip().lower()
    mapped_boot: int
    mapped_refute: bool | str
    if key == "interactive":
        mapped_boot, mapped_refute = 0, "cheap"
    elif key == "standard":
        mapped_boot, mapped_refute = 50, True
    elif key == "report":
        mapped_boot, mapped_refute = 200, "full"
    else:
        raise ValueError(f"unknown latency={latency!r}; use interactive|standard|report")
    # refute default True means "use mode mapping" when latency is set unless
    # the caller chose a non-default refute value.
    out_refute: bool | Literal["full", "placebo", "none", "cheap"] = (
        mapped_refute if refute is True else refute  # type: ignore[assignment]
    )
    out_boot = mapped_boot if bootstrap is None else bootstrap
    return out_boot, out_refute


def _discovery_algorithm(discovery: Any) -> dict[str, Any]:
    return discovery_algorithm(discovery)


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise ValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, Cpdag):
        # PathSpecific / Interventional need a fully oriented DAG; incomplete
        # CPDAGs fail closed with a clear undirected-count message.
        return cpdag_oriented_edges(graph, require_oriented=True)
    return [(str(a), str(b)) for a, b in graph]


def _lagged_edges(
    graph: TemporalDag | Sequence[tuple[str, int, str, int]] | None,
) -> list[tuple[str, int, str, int]]:
    if graph is None:
        raise ValueError("graph= lagged edges are required")
    if isinstance(graph, TemporalDag):
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph.edges()]
    return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph]


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
    # Bayesian, refute, and validators are supported on series Pulse/Sustained.
    _ = (inference, refute, validators)
    return


def _bayesian_inference_kwargs(inference: Bayesian) -> dict[str, Any]:
    backend = str(inference.backend).strip().lower()
    if backend == "laplace":
        inference_s = "bayesian"
    elif backend == "conjugate":
        inference_s = "conjugate"
    elif backend == "hmc":
        inference_s = "hmc"
    else:
        raise ValueError(
            f"unknown Bayesian backend {inference.backend!r}; use laplace|conjugate|hmc"
        )
    kw: dict[str, Any] = {
        "inference": inference_s,
        "n_draws": inference.n_draws,
        "prior_scale": inference.prior_scale,
    }
    prior_from = inference.prior_from
    if prior_from is not None:
        # Local import avoids circular import with prior_bank ↔ estimation.
        from .prior_bank import ComposedPrior

        if isinstance(prior_from, ComposedPrior):
            kw["composed_prior"] = prior_from.to_native_dict()
        else:
            kw["prior_artifact"] = bytes(prior_from)
    if inference.mapping is not None:
        kw["prior_mapping"] = inference.mapping.to_dict()
    return kw


def _temporal_inference_kwargs(
    inference: Frequentist | Bayesian | None,
) -> dict[str, Any]:
    if isinstance(inference, Bayesian):
        return _bayesian_inference_kwargs(inference)
    if isinstance(inference, Frequentist) or inference is None:
        return {}
    return {}


def _wrap_temporal(raw: TemporalAnalysisResult) -> AnalysisResult:
    # Mirror static ate_result_from_analysis: never claim pass when nothing ran.
    ran = raw.refutation_count > 0
    mediation = None
    if (
        getattr(raw, "mediation_total", None) is not None
        or getattr(raw, "mediation_mediated", None) is not None
    ):
        mediation = MediationView(
            total=getattr(raw, "mediation_total", None),
            direct=getattr(raw, "mediation_direct", None),
            mediated=getattr(raw, "mediation_mediated", None),
        )
    posterior = None
    if getattr(raw, "posterior_n_draws", None) is not None:
        mass = getattr(raw, "posterior_unidentified_mass", None)
        envelope = None
        if mass is not None and float(mass) > 0.0:
            envelope = EffectEnvelope(
                effect_mean=raw.posterior_effect_mean,
                effect_sd=raw.posterior_effect_sd,
                q025=raw.posterior_q025,
                q975=raw.posterior_q975,
                unidentified_mass=float(mass),
                n_draws=raw.posterior_n_draws,
                backend=raw.posterior_backend,
            )
        posterior = PosteriorView(
            effect_mean=raw.posterior_effect_mean,
            effect_sd=raw.posterior_effect_sd,
            q025=raw.posterior_q025,
            q975=raw.posterior_q975,
            n_draws=raw.posterior_n_draws,
            p_below_zero=raw.posterior_p_below_zero,
            backend=raw.posterior_backend,
            artifact=raw.posterior_artifact,
            unidentified_mass=None if mass is None else float(mass),
            envelope=envelope,
        )
    return AnalysisResult(
        identification=IdentificationView(
            status=raw.identification_status,
            method=raw.method,
            adjustment_set=list(getattr(raw, "adjustment_set", []) or []),
            assumption_count=int(getattr(raw, "assumption_count", 0) or 0),
            derivation_step_count=int(getattr(raw, "derivation_step_count", 0) or 0),
        ),
        estimate=EstimateView(
            ate=raw.ate,
            se_analytic=raw.se_analytic,
            se_bootstrap=raw.se_bootstrap,
            estimator_id=str(getattr(raw, "estimator_id", "") or ""),
            method=raw.method,
            mediation=mediation,
        ),
        posterior=posterior,
        mediation=mediation,
        validation=ValidationView(
            passed=ran,
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
        plan=_plan_from_raw(raw),
        _raw=raw,
    )


def _resolve_static_discovery_edges(
    data, discovery, accept_discovered: bool, seed: int, threads: int
):
    """Run static discovery and return oriented DAG edge list.

    When ``accept_discovered`` is True, incomplete CPDAG/PAG marks raise
    ``ValueError`` (auto-accept cannot invent orientations). When False,
    raises ``CausalReviewError`` with structured attrs.
    """
    from . import discovery as disc
    from ._native import CausalReviewError

    def _require_oriented(result, *, kind: str, algorithm: str):
        try:
            return list(discovery_to_dag(result).edges())
        except ValueError as exc:
            pending = sum(
                1
                for e in result.graph_edges
                if not (
                    (e.at_source == "tail" and e.at_target == "arrow")
                    or (e.at_source == "arrow" and e.at_target == "tail")
                )
            )
            if accept_discovered:
                raise ValueError(
                    f"{algorithm}: accept_discovered=True but graph is incomplete "
                    f"({pending} non-directed marks); cannot invent orientations. {exc}"
                ) from exc
            err = CausalReviewError("cannot execute while graph review is required")
            err.kind = kind
            err.algorithm = algorithm
            err.pending_edge_count = pending
            err.hint = (
                "orient remaining edges into a Dag, or use finish_*_review / supply graph= edges"
            )
            err.message = str(err)
            raise err from exc

    if isinstance(discovery, ExactDagPosterior):
        return graph_posterior_map_edges(disc.discover_exact_dag_posterior(data))
    if isinstance(discovery, OrderMcmc):
        return graph_posterior_map_edges(
            disc.discover_order_mcmc(
                data,
                n_warmup=discovery.n_warmup,
                n_draws=discovery.n_draws,
                seed=seed,
                threads=threads,
            )
        )
    if isinstance(discovery, StructureMcmc):
        return graph_posterior_map_edges(
            disc.discover_structure_mcmc(
                data,
                n_warmup=discovery.n_warmup,
                n_draws=discovery.n_draws,
                seed=seed,
                threads=threads,
            )
        )
    if isinstance(discovery, CiScreenedPosterior):
        return graph_posterior_map_edges(
            disc.discover_ci_screened_posterior(
                data,
                alpha=discovery.alpha,
                fdr=discovery.fdr,
                seed=seed,
                threads=threads,
            )
        )
    if isinstance(discovery, (FCI, RFCI)):
        algo = "fci" if isinstance(discovery, FCI) else "rfci"
        if not accept_discovered:
            err = CausalReviewError(
                "FCI/RFCI PathSpecific/Interventional queries require a fully "
                "oriented DAG; accept_discovered=False leaves PAG review open"
            )
            err.kind = "static_pag"
            err.algorithm = algo
            err.pending_edge_count = 0
            err.hint = (
                "orient the PAG to a Dag (or use PC/GES/LiNGAM/NOTEARS); "
                "PathSpecific/Interventional do not run generalized PAG adjustment"
            )
            err.message = str(err)
            raise err
        raise ValueError(
            f"{algo}: PathSpecific/Interventional require a fully oriented DAG; "
            "use PC/GES/LiNGAM/NOTEARS or supply graph= edges "
            "(accept_discovered cannot invent PAG orientations)"
        )
    if isinstance(discovery, (PC, GES, LiNGAM, NOTEARS)):
        result, algo = run_static_discovery(data, discovery, seed=seed, threads=threads)
        kind = "static_dag" if algo in ("lingam", "notears") else "static_cpdag"
        return _require_oriented(result, kind=kind, algorithm=algo)
    raise TypeError(f"unsupported discovery type for path/distribution: {type(discovery)!r}")


def analyze_many(
    data: Mapping[str, Any] | Any,
    *,
    graph: Dag | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect],
    identifier: str | None = None,
    estimator: str | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] = True,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Literal["interactive", "standard", "report"] | None = None,
) -> list[AnalysisResult]:
    """Estimate many average effects on one shared table ingest.

    Parameters
    ----------
    data:
        Column mapping / DataFrame (ingested once).
    graph:
        Static DAG or edge list shared by every query.
    queries:
        Non-empty sequence of ``AverageEffect`` queries.
    """
    if not queries:
        raise ValueError("analyze_many requires at least one query")
    if not all(isinstance(q, AverageEffect) for q in queries):
        raise TypeError("analyze_many currently supports AverageEffect queries only")
    bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)
    names, columns = as_columns(data)
    edges = _static_edges(graph)
    specs = [
        (q.treatment, q.outcome, float(q.control_level), float(q.active_level)) for q in queries
    ]
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    if latency is not None:
        kwargs["latency"] = latency
    raws = _analyze_ate_many(names, columns, edges, specs, **kwargs)
    return [_wrap_ate(r) for r in raws]


@dataclass(frozen=True)
class IdentifyResult:
    """Identify-only result (no estimate)."""

    status: str
    method: str
    adjustment_set: list[str]


def identify(
    *,
    graph: Dag | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> IdentifyResult:
    """Identify without estimating.

    Pass ``names`` when ``graph`` is an edge list (variable order). With a
    ``Dag``, names are taken from ``graph.nodes()``.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(graph, Dag):
        node_names = list(graph.nodes())
        edges = list(graph.edges())
    else:
        if names is None:
            raise ValueError("identify(edge_list) requires names=")
        node_names = list(names)
        edges = list(graph)
    status, method, adjustment = _identify_ate(
        node_names,
        edges,
        query.treatment,
        query.outcome,
        identifier=identifier,
    )
    return IdentifyResult(status=status, method=method, adjustment_set=list(adjustment))


class PreparedAnalysis:
    """Compile-once / re-estimate-many handle for static AverageEffect on a DAG.

    Use for interactive sessions: prepare with a fixed graph/query/estimator,
    then call :meth:`estimate` or :meth:`refresh` when the table changes
    (same schema). Prefer this over fresh :func:`analyze` on every click.
    For streaming append + incremental OLS, use :class:`antecedent.CausalState`.
    """

    def __init__(self, native: Any) -> None:
        self._native = native

    @classmethod
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        query: AverageEffect,
        graph: Dag | Sequence[tuple[str, str]],
        inference: Frequentist | Bayesian | None = None,
        identifier: str | Identifier | None = None,
        estimator: str | Estimator | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] = False,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int = 1,
        latency: Latency | Literal["interactive", "standard", "report"] | None = "interactive",
    ) -> PreparedAnalysis:
        """Compile a durable plan for static ATE on a supplied DAG."""
        if not isinstance(query, AverageEffect):
            raise TypeError("PreparedAnalysis supports AverageEffect only")
        inference = inference or Frequentist()
        if isinstance(identifier, Identifier):
            identifier = str(identifier)
        if isinstance(estimator, Estimator):
            estimator = str(estimator)
        if isinstance(latency, Latency):
            latency = str(latency)  # type: ignore[assignment]
        if isinstance(refute, Refute):
            refute = str(refute)  # type: ignore[assignment]
        bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)
        names, columns = as_columns(data)
        edges = _static_edges(graph)
        bayes_kw: dict[str, Any] = {}
        if isinstance(inference, Bayesian):
            bayes_kw = _bayesian_inference_kwargs(inference)
            inference_mode = str(bayes_kw.pop("inference"))
        else:
            inference_mode = "frequentist"
        native = _NativePreparedAnalysis.prepare(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=identifier,
            estimator=estimator,
            inference=inference_mode,
            n_draws=int(bayes_kw.get("n_draws", 1000)),
            prior_scale=float(bayes_kw.get("prior_scale", 10.0)),
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            latency=latency,
        )
        return cls(native)

    @property
    def plan(self) -> PhysicalPlanView:
        """Physical-plan summary retained from prepare."""
        raw = self._native.plan_summary()
        return PhysicalPlanView(
            plan_id=str(raw.get("plan_id", "")),
            estimated_peak_memory_bytes=(
                int(raw["estimated_peak_memory_bytes"])
                if "estimated_peak_memory_bytes" in raw
                else None
            ),
            workspace_bytes=(int(raw["workspace_bytes"]) if "workspace_bytes" in raw else None),
            batch_size=int(raw["batch_size"]) if "batch_size" in raw else None,
            worker_threads=int(raw.get("worker_threads", 0)),
            expected_python_crossings=int(raw.get("expected_python_crossings", 0)),
            deterministic_reductions=str(raw.get("deterministic_reductions", "true")).lower()
            in ("1", "true"),
            kernels=raw.get("kernels") or None,
        )

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Re-estimate without recompiling (same schema as prepare)."""
        names, columns = as_columns(data)
        raw = self._native.estimate(names, columns, seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Replace retained data and re-estimate."""
        names, columns = as_columns(data)
        raw = self._native.refresh(names, columns, seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int = 1,
        threads: int = 1,
        cancel: Any | None = None,
    ) -> AnalysisResult:
        """Second-click refute against the last :meth:`estimate` / :meth:`refresh`.

        Interactive first clicks typically use ``refute=False`` or ``cheap``;
        call this with ``suite="placebo"`` or ``"full"`` for the deferred suite.
        """
        if isinstance(suite, Refute):
            suite = str(suite)
        names, columns = as_columns(data)
        kwargs: dict[str, Any] = dict(seed=seed, threads=threads)
        if cancel is not None:
            kwargs["cancel"] = cancel
        raw = self._native.refute(names, columns, suite, **kwargs)
        return _wrap_ate(raw, prepared=self)


__all__ = [
    "AnalysisResult",
    "ConflictSummaryView",
    "EstimateView",
    "MediationView",
    "IdentificationView",
    "IdentifyResult",
    "NativeAnalysisResult",
    "PerformanceView",
    "PhysicalPlanView",
    "PlanView",
    "PosteriorView",
    "PredictiveCheckReport",
    "PreparedAnalysis",
    "PriorSensitivityReport",
    "TemporalAnalysisResult",
    "ValidationView",
    "analyze_many",
    "identify",
]
