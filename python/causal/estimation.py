"""High-level estimation entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ._data import as_columns, as_multi_env_columns, try_as_arrow_c_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
    AteAnalysisResult,
    CausalUnsupportedError,
    PreparedAnalysis as _NativePreparedAnalysis,
    analyze as _analyze_temporal,
    analyze_ate as _analyze_ate,
    analyze_ate_admg as _analyze_ate_admg,
    analyze_ate_arrow_c as _analyze_ate_arrow_c,
    analyze_ate_cpdag as _analyze_ate_cpdag,
    analyze_ate_discover as _analyze_ate_discover,
    analyze_ate_many as _analyze_ate_many,
    analyze_ate_pag as _analyze_ate_pag,
    analyze_conditional as _analyze_conditional,
    analyze_distribution as _analyze_distribution,
    analyze_events as _analyze_events,
    analyze_mediation as _analyze_mediation,
    analyze_panel as _analyze_panel,
    analyze_panel_discover as _analyze_panel_discover,
    analyze_path_specific as _analyze_path_specific,
    analyze_temporal_discover as _analyze_temporal_discover,
    analyze_temporal_mediation as _analyze_temporal_mediation,
    analyze_temporal_pag as _analyze_temporal_pag,
    identify_ate as _identify_ate,
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
    CiScreenedPosterior,
    DbnPosterior,
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    cpdag_oriented_edges,
    discovery_to_dag,
    graph_posterior_map_edges,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from .inference import Bayesian, Frequentist
from .query import (
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    InterventionalDistribution,
    MediationEffect,
    PathSpecificEffect,
    PulseEffect,
    SustainedEffect,
    TemporalMediationEffect,
)
from .ids import Estimator, Identifier, Latency, Refute

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
        return self._prepared.refute(
            data, suite, seed=seed, threads=threads, cancel=cancel
        )


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
            alphas_requested=list(r.conflict_alphas_requested),
            alphas_applied=list(r.conflict_alphas_applied),
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
    if getattr(raw, "prior_ppc_p_value", None) is not None:
        prior_predictive = PredictiveCheckReport(
            kind="prior_predictive",
            observed=float(raw.prior_ppc_observed),
            predictive_mean=float(raw.prior_ppc_predictive_mean),
            predictive_sd=float(raw.prior_ppc_predictive_sd),
            p_value=float(raw.prior_ppc_p_value),
            n_sims=int(raw.prior_ppc_n_sims),
        )
    posterior_predictive = None
    if getattr(raw, "posterior_ppc_p_value", None) is not None:
        posterior_predictive = PredictiveCheckReport(
            kind="posterior_predictive",
            observed=float(raw.posterior_ppc_observed),
            predictive_mean=float(raw.posterior_ppc_predictive_mean),
            predictive_sd=float(raw.posterior_ppc_predictive_sd),
            p_value=float(raw.posterior_ppc_p_value),
            n_sims=int(raw.posterior_ppc_n_sims),
        )
    prior_sensitivity = None
    means = getattr(raw, "prior_sensitivity_means", None)
    if means is not None:
        alphas_raw = getattr(raw, "prior_sensitivity_alphas", None)
        scales_raw = getattr(raw, "prior_sensitivity_scales", None)
        prior_sensitivity = PriorSensitivityReport(
            scales=list(scales_raw or ()),
            effect_means=list(means),
            effect_sds=list(raw.prior_sensitivity_sds),
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
            bootstrap_replicates_requested=getattr(
                raw, "bootstrap_replicates_requested", None
            ),
            bootstrap_replicates_ok=getattr(raw, "bootstrap_replicates_ok", None),
            n_draws=getattr(raw, "n_draws_effort", None),
            cancelled=bool(getattr(raw, "cancelled", False)),
            early_stopped=bool(getattr(raw, "early_stopped", False)),
            stage_timings={
                str(k): int(v) for k, v in (getattr(raw, "stage_timings", None) or [])
            }
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
    if key == "interactive":
        mapped_boot, mapped_refute = 0, "cheap"
    elif key == "standard":
        mapped_boot, mapped_refute = 50, True
    elif key == "report":
        mapped_boot, mapped_refute = 200, "full"
    else:
        raise ValueError(
            f"unknown latency={latency!r}; use interactive|standard|report"
        )
    # refute default True means "use mode mapping" when latency is set unless
    # the caller chose a non-default refute value.
    out_refute: bool | Literal["full", "placebo", "none", "cheap"]
    if refute is True:
        out_refute = mapped_refute  # type: ignore[assignment]
    else:
        out_refute = refute  # type: ignore[assignment]
    out_boot = mapped_boot if bootstrap is None else bootstrap
    return out_boot, out_refute


_STATIC_DISCOVERY = (PC, GES, LiNGAM, NOTEARS, FCI, RFCI)
_GRAPH_POSTERIOR_DISCOVERY = (
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    CiScreenedPosterior,
)
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
    if isinstance(discovery, ExactDagPosterior):
        return {"algorithm": "exact_dag_posterior"}
    if isinstance(discovery, OrderMcmc):
        return {
            "algorithm": "order_mcmc",
            "n_chains": discovery.n_chains,
            "n_warmup": discovery.n_warmup,
            "mcmc_draws": discovery.n_draws,
            "thin": discovery.thin,
            "require_diagnostics_gate": discovery.require_diagnostics_gate,
        }
    if isinstance(discovery, StructureMcmc):
        return {
            "algorithm": "structure_mcmc",
            "n_chains": discovery.n_chains,
            "n_warmup": discovery.n_warmup,
            "mcmc_draws": discovery.n_draws,
            "thin": discovery.thin,
        }
    if isinstance(discovery, CiScreenedPosterior):
        return {
            "algorithm": "ci_screened_posterior",
            "alpha": discovery.alpha,
            "fdr": discovery.fdr,
            "ci": discovery.ci,
            "max_cond_size": discovery.max_cond_size,
            "soft_weight": discovery.soft_weight,
            "n_chains": discovery.n_chains,
            "n_warmup": discovery.n_warmup,
            "mcmc_draws": discovery.n_draws,
            "thin": discovery.thin,
        }
    if isinstance(discovery, DbnPosterior):
        return {
            "algorithm": "dbn_posterior",
            "max_lag": discovery.max_lag,
            "force_mcmc": discovery.force_mcmc,
            "n_chains": discovery.n_chains,
            "n_warmup": discovery.n_warmup,
            "mcmc_draws": discovery.n_draws,
        }
    raise TypeError(f"unsupported discovery config: {type(discovery)!r}")


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
            f"unknown Bayesian backend {inference.backend!r}; "
            "use laplace|conjugate|hmc"
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
    if getattr(raw, "mediation_total", None) is not None or getattr(raw, "mediation_mediated", None) is not None:
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
        plan=_plan_from_raw(raw),
        _raw=raw,
    )



def _resolve_static_discovery_edges(data, discovery, accept_discovered: bool, seed: int, threads: int):
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
            err = CausalReviewError(
                "cannot execute while graph review is required"
            )
            err.kind = kind
            err.algorithm = algorithm
            err.pending_edge_count = pending
            err.hint = (
                "orient remaining edges into a Dag, or use finish_*_review / "
                "supply graph= edges"
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
    if isinstance(discovery, PC):
        result = disc.discover_pc(
            data, alpha=discovery.alpha, fdr=discovery.fdr, seed=seed, threads=threads
        )
        return _require_oriented(result, kind="static_cpdag", algorithm="pc")
    if isinstance(discovery, GES):
        result = disc.discover_ges(
            data, alpha=discovery.alpha, fdr=discovery.fdr, seed=seed, threads=threads
        )
        return _require_oriented(result, kind="static_cpdag", algorithm="ges")
    if isinstance(discovery, LiNGAM):
        result = disc.discover_lingam(data, seed=seed, threads=threads)
        return _require_oriented(result, kind="static_dag", algorithm="lingam")
    if isinstance(discovery, NOTEARS):
        result = disc.discover_notears(data, seed=seed, threads=threads)
        return _require_oriented(result, kind="static_dag", algorithm="notears")
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
    names, columns = as_columns(data)  # type: ignore[arg-type]
    edges = _static_edges(graph)  # type: ignore[arg-type]
    specs = [
        (q.treatment, q.outcome, float(q.control_level), float(q.active_level))
        for q in queries
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


def analyze(
    data: Mapping[str, Any] | Any | Sequence[Mapping[str, Any] | Any],
    *,
    query: (
        AverageEffect
        | PulseEffect
        | SustainedEffect
        | InterventionalDistribution
        | PathSpecificEffect
        | ConditionalEffect
        | MediationEffect
        | Counterfactual
        | TemporalMediationEffect
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
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] = True,
    validators: Sequence[Any] | None = None,
    accept_discovered: bool = True,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    regimes: Sequence[int] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    population_registry: Any | None = None,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    return_posterior_artifact: bool = False,
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
        ``InterventionalDistribution``, ``PathSpecificEffect``,
        ``MediationEffect``, ``Counterfactual``, or ``TemporalMediationEffect``.
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
        One-shot script convenience — discovery runs at compile time. For
        interactive / spreadsheet estimate clicks, discover once into
        :class:`causal.AcceptedGraph` (or hold a reviewed graph) and pass
        ``graph=`` with ``latency="interactive"`` instead. Combining
        ``discovery=`` with ``latency="interactive"`` raises
        :class:`CausalUnsupportedError`.
    latency:
        Optional compute tier (``interactive`` / ``standard`` / ``report``).
        Maps to known-equivalent bootstrap / refute / draws; explicit
        ``bootstrap=`` / ``refute=`` always win. Interactive refuses inline
        ``discovery=`` (artifact-first UX).
    cancel:
        Optional ``CancellationToken`` from ``causal._native``.
    on_progress:
        Optional ``(fraction: float, stage: str) -> None`` callback.
    on_stage:
        Optional ``(stage: str, payload: dict) -> None`` progressive stage
        callback (identify → estimate_point → uncertainty → validate).
    return_posterior_artifact:
        When ``True`` and inference is Bayesian, attach full posterior draw
        bytes on ``result.posterior.artifact`` (for download / sequential-prior
        hydrate). Default ``False``: UI summaries only.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(estimator, Estimator):
        estimator = str(estimator)
    if isinstance(latency, Latency):
        latency = str(latency)  # type: ignore[assignment]
    if isinstance(refute, Refute):
        refute = str(refute)  # type: ignore[assignment]
    inference = inference or Frequentist()
    bootstrap, refute = _resolve_latency_budget(latency, bootstrap, refute)

    if discovery is not None and latency == "interactive":
        raise CausalUnsupportedError(
            "discovery= is not on the interactive estimate path; "
            "call discover_* once, accept into AcceptedGraph, then "
            "analyze(graph=..., latency='interactive')"
        )

    if isinstance(query, ConditionalEffect):
        if isinstance(inference, Bayesian):
            raise TypeError("ConditionalEffect does not support inference=Bayesian(...)")
        if discovery is not None:
            raise ValueError("ConditionalEffect does not support discovery=")
        names, columns = as_columns(data)  # type: ignore[arg-type]
        edges = _static_edges(graph)  # type: ignore[arg-type]
        raw = _analyze_conditional(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            query.modifier,
            control_level=query.control_level,
            active_level=query.active_level,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_ate(raw)

    if isinstance(query, TemporalMediationEffect):
        if isinstance(inference, Bayesian):
            raise TypeError("TemporalMediationEffect does not support inference=Bayesian(...)")
        if discovery is not None:
            raise ValueError("TemporalMediationEffect does not support discovery=")
        names, columns = as_columns(data)  # type: ignore[arg-type]
        lagged = _lagged_edges(graph)  # type: ignore[arg-type]
        raw = _analyze_temporal_mediation(
            names,
            columns,
            lagged,
            query.treatment,
            query.mediator,
            query.outcome,
            contrast=query.contrast,
            control_level=query.control_level,
            active_level=query.active_level,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_temporal(raw)

    if isinstance(query, MediationEffect):
        if discovery is not None:
            raise ValueError("MediationEffect does not support discovery=")
        edges = _static_edges(graph)  # type: ignore[arg-type]
        names, columns = as_columns(data)  # type: ignore[arg-type]
        raw = _analyze_mediation(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            list(query.mediators),
            contrast=query.contrast,
            control_level=query.control_level,
            active_level=query.active_level,
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_ate(raw)

    if isinstance(query, Counterfactual):
        from ._native import counterfactual_ite

        if discovery is not None:
            raise ValueError("Counterfactual does not support discovery=")
        edges = _static_edges(graph)  # type: ignore[arg-type]
        names, columns = as_columns(data)  # type: ignore[arg-type]
        ite = counterfactual_ite(
            names,
            columns,
            edges,
            query.treatment,
            query.outcome,
            query.active_level,
            query.control_level,
            seed=seed,
            threads=threads,
        )
        return AnalysisResult(
            identification=IdentificationView(
                status="gcm.parametric",
                method="counterfactual.ite",
                adjustment_set=[],
                assumption_count=0,
                derivation_step_count=0,
            ),
            estimate=EstimateView(
                ate=float(ite.mean_ite),
                se_analytic=float("nan"),
                se_bootstrap=None,
                estimator_id="gcm.ite",
                method="counterfactual.ite",
            ),
            posterior=None,
            validation=ValidationView(passed=False, ran=False, count=0),
            performance=PerformanceView(
                plan_id="counterfactual.ite",
                modality="static",
                peak_memory_bytes=0,
            ),
            diagnostics=[],
            provenance={"noise_inference": getattr(ite, "noise_inference", None)},
            _raw=ite,
        )

    if isinstance(query, InterventionalDistribution):
        if discovery is not None:
            edges = _resolve_static_discovery_edges(
                data, discovery, accept_discovered, seed, threads
            )
        else:
            edges = _static_edges(graph)  # type: ignore[arg-type]
        names, columns = as_columns(data)  # type: ignore[arg-type]
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
            edges = _resolve_static_discovery_edges(
                data, discovery, accept_discovered, seed, threads
            )
        else:
            edges = _static_edges(graph)  # type: ignore[arg-type]
        names, columns = as_columns(data)  # type: ignore[arg-type]
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

    if discovery is not None and isinstance(
        discovery, _STATIC_DISCOVERY + _GRAPH_POSTERIOR_DISCOVERY
    ):
        if not isinstance(query, AverageEffect):
            raise ValueError(
                f"discovery={type(discovery).__name__}(...) requires AverageEffect"
            )
        if isinstance(discovery, _GRAPH_POSTERIOR_DISCOVERY) and not isinstance(
            inference, Bayesian
        ):
            raise TypeError(
                "graph-posterior discovery requires inference=Bayesian(...) "
                "for effect mixture"
            )
        names, columns = as_columns(data)  # type: ignore[arg-type]
        cfg = _discovery_algorithm(discovery)
        bayes_kw: dict[str, Any] = {}
        if isinstance(inference, Bayesian):
            bayes_kw = _bayesian_inference_kwargs(inference)
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
            refute=refute,
            validators=list(validators) if validators is not None else None,
            ci=cfg.get("ci"),
            n_chains=cfg.get("n_chains", 2),
            n_warmup=cfg.get("n_warmup", 100),
            mcmc_draws=cfg.get("mcmc_draws", 200),
            thin=cfg.get("thin", 1),
            soft_weight=cfg.get("soft_weight", "none"),
            require_diagnostics_gate=cfg.get("require_diagnostics_gate", True),
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            **bayes_kw,
        )
        return _wrap_ate(raw)

    if discovery is not None and isinstance(query, AverageEffect):
        raise ValueError(
            "AverageEffect with discovery= requires a static algorithm "
            "(PC/GES/LiNGAM/NOTEARS/FCI/RFCI); temporal discovery needs "
            "PulseEffect/SustainedEffect"
        )

    if isinstance(query, AverageEffect):
        bayes_kw: dict[str, Any] = {}
        if isinstance(inference, Bayesian):
            bayes_kw = _bayesian_inference_kwargs(inference)
        if estimator == "rd.sharp" or any(
            v is not None for v in (running_variable, cutoff, bandwidth)
        ):
            if running_variable is None or cutoff is None or bandwidth is None:
                raise ValueError(
                    "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth"
                )
            if estimator is None:
                estimator = "rd.sharp"
            if identifier is None:
                identifier = "rd.sharp"
        common = dict(
            treatment=query.treatment,
            outcome=query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=identifier,
            estimator=estimator,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            **bayes_kw,
        )
        if return_posterior_artifact:
            common["return_posterior_artifact"] = True
        if latency is not None:
            common["latency"] = latency
        if cancel is not None:
            common["cancel"] = cancel
        if on_progress is not None:
            common["on_progress"] = on_progress
        if on_stage is not None:
            common["on_stage"] = on_stage
        from .population import coerce_target_population, registry_wire

        pop = coerce_target_population(
            getattr(query, "target_population", None)
        )
        preds, dists = registry_wire(population_registry)
        pop_kw: dict[str, Any] = {}
        if pop is not None:
            pop_kw["target_population"] = pop
        if preds:
            pop_kw["population_predicates"] = preds
        if dists:
            pop_kw["population_distributions"] = dists
        if pop_kw and isinstance(graph, (Pag, Cpdag, Admg)):
            raise ValueError(
                "target_population / population_registry currently require a Dag "
                "(or edge list); PAG/CPDAG/ADMG analyze paths do not accept them yet"
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
        ate_kwargs = dict(edges=edges, **common, **pop_kw)
        # Prefer Arrow CDI (zero-copy float64) when available. Population kwargs
        # still require the NumPy path until that surface is wired on CDI.
        use_arrow = arrow is not None and not pop_kw
        if use_arrow:
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
        bayes_kw = _temporal_inference_kwargs(inference)
        if isinstance(data, EventFrame):
            if discovery is not None:
                if isinstance(discovery, JPCMCIPlus):
                    raise TypeError(
                        "EventFrame does not support discovery=JPCMCIPlus(...); "
                        "use MultiEnvFrame or PanelFrame for multi-environment discovery"
                    )
                if isinstance(discovery, DbnPosterior):
                    if not isinstance(inference, Bayesian):
                        raise TypeError(
                            "EventFrame discovery=DbnPosterior(...) requires inference=Bayesian(...)"
                        )
                elif not isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI, RPCMCI)):
                    raise TypeError(
                        f"EventFrame discovery expects PCMCI/PCMCIPlus/LPCMCI/RPCMCI/DbnPosterior, "
                        f"got {type(discovery)!r}"
                    )
                cfg = _discovery_algorithm(discovery)
                raw = _analyze_events(
                    data.names,
                    data.columns,
                    data.event_times_ns.tolist(),
                    data.align_interval_ns,
                    [],  # discovery path ignores edges
                    query.treatment,
                    query.outcome,
                    treatment_lag=query.treatment_lag,
                    horizon_steps=query.horizon_steps,
                    active_level=query.active_level,
                    policy=policy,
                    **bayes_kw,
                    refute=refute,
                    validators=list(validators) if validators is not None else None,
                    seed=seed,
                    bootstrap=bootstrap,
                    threads=threads,
                    algorithm=cfg["algorithm"],
                    max_lag=cfg.get("max_lag", 1),
                    alpha=cfg.get("alpha", 0.05),
                    fdr=cfg.get("fdr", True),
                    accept_discovered=accept_discovered,
                    regimes=list(regimes) if regimes is not None else None,
                    **{
                        k: cfg[k]
                        for k in ("n_chains", "n_warmup", "mcmc_draws", "force_mcmc", "ci")
                        if k in cfg
                    },
                )
                return _wrap_temporal(raw)
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
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
                seed=seed,
                bootstrap=bootstrap,
                threads=threads,
            )
            return _wrap_temporal(raw)
        if isinstance(data, PanelFrame):
            if discovery is not None:
                if isinstance(discovery, JPCMCIPlus):
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
                        **bayes_kw,
                        refute=refute,
                        validators=list(validators) if validators is not None else None,
                        seed=seed,
                        bootstrap=bootstrap,
                        threads=threads,
                        context_names=cfg["context_names"],
                        include_space_dummy=cfg["include_space_dummy"],
                        include_time_dummy=cfg["include_time_dummy"],
                        space_dummy_ci=cfg["space_dummy_ci"]
                        in ("multivariate", "multivariate_block", "block", True),
                        time_dummy_encoding=cfg["time_dummy_encoding"],
                        time_dummy_ci=cfg["time_dummy_ci"]
                        in ("multivariate", "multivariate_block", "block", True),
                    )
                    return _wrap_temporal(raw)
                if isinstance(discovery, (PCMCI, PCMCIPlus, LPCMCI)):
                    cfg = _discovery_algorithm(discovery)
                    # Pooled-units discovery: treat panel as multi-env without JPCMCI+ context.
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
                        **bayes_kw,
                        refute=refute,
                        validators=list(validators) if validators is not None else None,
                        seed=seed,
                        bootstrap=bootstrap,
                        threads=threads,
                        algorithm=cfg["algorithm"],
                    )
                    return _wrap_temporal(raw)
                raise TypeError(
                    "PanelFrame discovery supports JPCMCIPlus, PCMCI, PCMCIPlus, or LPCMCI"
                )
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
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
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
                **bayes_kw,
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
            if isinstance(discovery, DbnPosterior):
                if not isinstance(inference, Bayesian):
                    raise TypeError(
                        "discovery=DbnPosterior(...) requires inference=Bayesian(...) "
                        "for temporal effect mixture"
                    )
                cfg = _discovery_algorithm(discovery)
                names, columns = as_columns(data)  # type: ignore[arg-type]
                raw = _analyze_temporal_discover(
                    names,
                    columns,
                    query.treatment,
                    query.outcome,
                    algorithm="dbn_posterior",
                    max_lag=cfg["max_lag"],
                    accept_discovered=accept_discovered,
                    treatment_lag=query.treatment_lag,
                    horizon_steps=query.horizon_steps,
                    active_level=query.active_level,
                    policy=policy,
                    **bayes_kw,
                    n_chains=cfg["n_chains"],
                    n_warmup=cfg["n_warmup"],
                    mcmc_draws=cfg["mcmc_draws"],
                    force_mcmc=cfg["force_mcmc"],
                    seed=seed,
                    bootstrap=bootstrap,
                    threads=threads,
                )
                return _wrap_temporal(raw)
            if not isinstance(discovery, _TEMPORAL_DISCOVERY):
                raise TypeError(
                    f"temporal discovery expects PCMCI-family or DbnPosterior, got {type(discovery)!r}"
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
                    **bayes_kw,
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
                    **bayes_kw,
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
                **bayes_kw,
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
                **bayes_kw,
                refute=refute,
                validators=list(validators) if validators is not None else None,
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
            **bayes_kw,
            refute=refute,
            validators=list(validators) if validators is not None else None,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        return _wrap_temporal(raw)

    raise TypeError(f"unsupported query type: {type(query)!r}")


class PreparedAnalysis:
    """Compile-once / re-estimate-many handle for static AverageEffect on a DAG.

    Use for interactive sessions: prepare with a fixed graph/query/estimator,
    then call :meth:`estimate` or :meth:`refresh` when the table changes
    (same schema). Prefer this over fresh :func:`analyze` on every click.
    For streaming append + incremental OLS, use :class:`causal.CausalState`.
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
        names, columns = as_columns(data)  # type: ignore[arg-type]
        edges = _static_edges(graph)  # type: ignore[arg-type]
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
            workspace_bytes=(
                int(raw["workspace_bytes"]) if "workspace_bytes" in raw else None
            ),
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
        names, columns = as_columns(data)  # type: ignore[arg-type]
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
        names, columns = as_columns(data)  # type: ignore[arg-type]
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
        names, columns = as_columns(data)  # type: ignore[arg-type]
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
    "analyze",
    "analyze_many",
    "identify",
]
