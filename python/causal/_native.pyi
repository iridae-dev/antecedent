"""Type stubs for the native extension module ``causal._native``."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

CiArg = str | Callable[..., Any] | None

__version__: str

class CausalError(Exception): ...
class CausalIdentifyError(CausalError): ...
class CausalEstimateError(CausalError): ...
class CausalValidateError(CausalError): ...
class CausalDiscoveryError(CausalError): ...
class CausalModelError(CausalError): ...
class CausalCounterfactualError(CausalError): ...
class CausalAttributionError(CausalError): ...
class CausalDataError(CausalError): ...
class CausalGraphError(CausalError): ...
class CausalDesignError(CausalError): ...
class CausalStateError(CausalError): ...
class CausalSerializationError(CausalError): ...
class CausalCompileError(CausalError): ...
class CausalResourceError(CausalError): ...
class CausalReviewError(CausalError):
    kind: str
    algorithm: str | None
    pending_edge_count: int
    hint: str
    message: str
class CausalUnsupportedError(CausalError): ...
class CausalCancelledError(CausalError): ...

class CancellationToken:
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...

class ArrowLoadInfo:
    row_count: int
    column_count: int
    bytes_copied: int
    bytes_borrowed: int
    diagnostic_count: int
    column_names: list[str]

class AteAnalysisResult:
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    bootstrap_replicates_failed: int | None
    adjustment_set: list[str]
    identification_status: str
    refutation_passed: bool
    refutation_ran: bool
    refutation_count: int
    assumption_count: int
    derivation_step_count: int
    method: str
    estimator_id: str
    overlap_ess: float | None
    overlap_propensity_min: float | None
    posterior_effect_mean: float | None
    posterior_effect_sd: float | None
    posterior_q025: float | None
    posterior_q975: float | None
    posterior_n_draws: int | None
    posterior_p_below_zero: float | None
    posterior_backend: str | None
    posterior_artifact: list[int] | None
    diagnostics: list[str]
    provenance_node_count: int
    plan_id: str
    modality: str
    peak_memory_bytes: int | None
    worker_threads: int
    expected_python_crossings: int
    prior_sensitivity_scales: list[float] | None
    prior_sensitivity_alphas: list[float] | None
    prior_sensitivity_means: list[float] | None
    prior_sensitivity_sds: list[float] | None
    conflict_source_ids: list[str] | None
    conflict_alphas_requested: list[float] | None
    conflict_alphas_applied: list[float] | None
    posterior_unidentified_mass: float | None
    latency_mode: str | None
    wall_time_ns: int | None
    bootstrap_replicates_requested: int | None
    bootstrap_replicates_ok: int | None
    n_draws_effort: int | None
    cancelled: bool
    early_stopped: bool
    stage_timings: list[tuple[str, int]]

class PosteriorArtifact:
    n_draws: int
    mean: list[float]
    sd: list[float]
    q025: list[float]
    q975: list[float]
    draws: list[float]
    backend_id: str
    identification: str
    unidentified_mass: float
    converged: bool
    hessian_condition: float
    quantity_names: list[str]

class DiscoveredLink:
    source: str
    source_lag: int
    target: str
    target_lag: int
    statistic: float
    p_value: float
    adjusted_p_value: float | None

class GraphEdge:
    source: str
    source_lag: int
    target: str
    target_lag: int
    at_source: str
    at_target: str

class PcmciDiscoveryResult:
    links: list[DiscoveredLink]
    algorithm_id: str
    algorithm_config: str
    ci_tests: int
    links_retained: int
    pending_edge_count: int
    lagged_frame_bytes: int
    worker_threads: int
    ci_name: str
    cpdag_nodes: int
    cpdag_directed_edges: int
    cpdag_undirected_edges: int
    graph_edges: list[GraphEdge]

class RpcmciDiscoverySummary:
    algorithm: str
    n_regimes: int
    regime_ids: list[int]
    directed_edges: list[int]
    undirected_edges: list[int]

class GraphPosterior:
    names: list[str]
    n_vars: int
    n_graphs: int
    weights: list[float]
    adjacency: list[int]
    edge_marginals: list[float]
    orientation_marginals: list[float]
    ess: float
    rejected_invalid: int
    converged: bool
    lagged_edge_marginals: list[float] | None
    max_lag: int | None
    def to_weighted_samples(self) -> dict[str, object]: ...
    def edge_marginal_matrix(self) -> list[list[float]]: ...

class MediationEffectsSummary:
    total: float
    direct: float
    mediated: float

class PredictSummary:
    mean_prediction: float
    n: int

class AnalysisResult:
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    plan_id: str
    modality: str
    peak_memory_bytes: int | None
    identification_status: str
    method: str
    diagnostics: list[str]
    provenance_node_count: int
    refutation_count: int
    worker_threads: int
    expected_python_crossings: int

TemporalAnalysisResult = AnalysisResult

class PreparedAnalysis:
    @staticmethod
    def prepare(
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatment: str,
        outcome: str,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        identifier: str | None = None,
        estimator: str | None = None,
        inference: str | None = None,
        n_draws: int = 1000,
        prior_scale: float = 10.0,
        refute: bool | str | None = None,
        seed: int = 1,
        bootstrap: int = 50,
        threads: int = 1,
        latency: str | None = None,
    ) -> PreparedAnalysis: ...
    def estimate(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AteAnalysisResult: ...
    def refresh(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AteAnalysisResult: ...
    @property
    def names(self) -> list[str]: ...

class GcmIteResult:
    mean_ite: float
    n_units: int
    noise_inference: str
    n_assignments: int
    unit_effects: NDArray[np.float64]

class GcmSampleResult:
    column_means: list[float]
    n_draws: int
    n_nodes: int
    draws: NDArray[np.float64]

class Contribution:
    name: str
    score: float

class ChangeAttributionResult:
    total_change: float
    contributions: list[tuple[str, float]]
    path_breakdown: list[tuple[list[str], float]]

class AnomalyScores:
    outcome: str
    mean_score: float
    n_units: int

class MechanismChangeDetection:
    node: str
    statistic: float
    p_value: float
    changed: bool

class FeatureRelevance:
    feature: str
    score: float

class RankedDesign:
    candidate_index: int
    kind: str
    tag: int
    score: float
    stderr: float
    rank: int
    rank_uncertain: bool

class DesignConstraintViolation:
    candidate_index: int
    constraint: str
    detail: str

class DesignRanking:
    best_index: int
    scores: list[float]
    mc_samples: int
    early_stopped: bool
    ranked: list[RankedDesign]
    violations: list[DesignConstraintViolation]

class DecisionEvaluation:
    expected_utility: float
    posterior_regret: float
    chosen_action: int | None

class FittedGcm:
    names: list[str]
    n_assignments: int
    def sample_do(
        self,
        interventions: dict[str, float],
        n: int,
        *,
        seed: int = 0,
        threads: int = 1,
    ) -> GcmSampleResult: ...
    def counterfactual_ite(
        self,
        treatment: str,
        outcome: str,
        active: float,
        control: float,
        *,
        seed: int = 0,
        threads: int = 1,
    ) -> GcmIteResult: ...
    def attribute_path_specific(
        self,
        treatment: str,
        outcome: str,
        *,
        path_nodes: list[str] | None = None,
        max_paths: int = 64,
        max_len: int = 16,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_paths(
        self,
        sources: list[str],
        outcome: str,
        *,
        max_paths: int = 64,
        max_len: int = 16,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_distribution_change(
        self,
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_distribution_change_robust(
        self,
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_structure_change(
        self,
        comparison_edges: list[tuple[str, str]],
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        n_samples: int = 500,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_unit_change(
        self,
        outcome: str,
        *,
        max_units: int = 0,
        seed: int = 0,
        threads: int = 1,
    ) -> ChangeAttributionResult: ...
    def attribute_feature_relevance(
        self,
        outcome: str,
        *,
        delta: float = 1.0,
        n_samples: int = 200,
        seed: int = 0,
        threads: int = 1,
    ) -> list[FeatureRelevance]: ...
    def anomaly_attribution(
        self,
        outcomes: list[str],
        *,
        max_units: int = 0,
    ) -> list[AnomalyScores]: ...
    def mechanism_change_detection(
        self,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        seed: int = 0,
        threads: int = 1,
    ) -> list[MechanismChangeDetection]: ...
    def rank_root_causes(
        self,
        attribution: ChangeAttributionResult,
        *,
        seed: int = 0,
        threads: int = 1,
    ) -> list[Contribution]: ...

def fit_gcm(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    *,
    threads: int = 1,
) -> FittedGcm: ...

def load_float64_columns(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
) -> ArrowLoadInfo: ...

def load_float64_arrow_c_columns(
    names: list[str],
    columns: Sequence[object],
) -> ArrowLoadInfo: ...

def analyze_ate(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    prior_artifact: bytes | None = None,
    prior_mapping: dict[str, Any] | None = None,
    composed_prior: dict[str, Any] | None = None,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
    target_population: dict[str, Any] | None = None,
    population_predicates: dict[str, list[int]] | None = None,
    population_distributions: dict[int, list[float]] | None = None,
    latency: str | None = None,
    cancel: CancellationToken | None = None,
    on_progress: Callable[[float, str], Any] | None = None,
    on_stage: Callable[[str, dict[str, Any]], Any] | None = None,
    return_posterior_artifact: bool = False,
) -> AteAnalysisResult: ...

def analyze_ate_arrow_c(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    prior_artifact: bytes | None = None,
    prior_mapping: dict[str, Any] | None = None,
    composed_prior: dict[str, Any] | None = None,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
    latency: str | None = None,
    cancel: CancellationToken | None = None,
    on_progress: Callable[[float, str], Any] | None = None,
    return_posterior_artifact: bool = False,
) -> AteAnalysisResult: ...

def analyze(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, int, str, int]],
    treatment: str,
    outcome: str,
    *,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
) -> AnalysisResult: ...

def analyze_temporal_pag(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    graph: TemporalPag,
    treatment: str,
    outcome: str,
    *,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
) -> AnalysisResult: ...

def analyze_events(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    event_times_ns: Sequence[int],
    align_interval_ns: int,
    edges: list[tuple[str, int, str, int]],
    treatment: str,
    outcome: str,
    *,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    prior_artifact: bytes | None = None,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
    algorithm: str | None = None,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    accept_discovered: bool = True,
    regimes: list[int] | None = None,
    n_chains: int = 1,
    n_warmup: int = 500,
    mcmc_draws: int = 1000,
    force_mcmc: bool = False,
    ci: CiArg = None,
) -> AnalysisResult: ...

def analyze_panel(
    names: list[str],
    unit_columns: Sequence[Sequence[NDArray[np.float64]]],
    unit_ids: Sequence[int],
    edges: list[tuple[str, int, str, int]],
    treatment: str,
    outcome: str,
    *,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    prior_artifact: bytes | None = None,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
) -> AnalysisResult: ...

def analyze_panel_discover(
    names: list[str],
    unit_columns: Sequence[Sequence[NDArray[np.float64]]],
    unit_ids: Sequence[int],
    treatment: str,
    outcome: str,
    *,
    algorithm: str = "jpcmci_plus",
    max_lag: int = 3,
    alpha: float = 0.05,
    fdr: bool = True,
    accept_discovered: bool = True,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    prior_artifact: bytes | None = None,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
    context_names: list[str] | None = None,
    include_space_dummy: bool = True,
    include_time_dummy: bool = False,
    space_dummy_ci: bool = False,
    time_dummy_encoding: str = "integer",
    time_dummy_ci: bool = False,
) -> AnalysisResult: ...

def analyze_distribution(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    interventions: dict[str, float],
    *,
    conditioning: list[str] | None = None,
    seed: int = 1,
    threads: int = 1,
) -> AteAnalysisResult: ...

def analyze_path_specific(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    path_nodes: list[str] | None = None,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...


def analyze_conditional(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    modifier: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...

def analyze_temporal_mediation(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, int, str, int]],
    treatment: str,
    mediator: str,
    outcome: str,
    *,
    contrast: str = "mediated",
    control_level: float = 0.0,
    active_level: float = 1.0,
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
) -> AnalysisResult: ...


def analyze_ate_discover(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    outcome: str,
    *,
    algorithm: str = "pc",
    alpha: float = 0.05,
    fdr: bool = True,
    max_cond_size: int = 2,
    prune_threshold: float = 0.0,
    l1: float = 0.1,
    threshold: float = 0.3,
    standardize: bool = True,
    accept_discovered: bool = True,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    refute: bool | str = True,
    validators: list[Callable[..., Any]] | None = None,
    ci: CiArg = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...

def analyze_temporal_discover(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    outcome: str,
    *,
    algorithm: str = "pcmci",
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    accept_discovered: bool = True,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    policy: str = "pulse",
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
    env_columns: list[Sequence[NDArray[np.float64]]] | None = None,
    regimes: list[int] | None = None,
    context_names: list[str] | None = None,
    include_space_dummy: bool = True,
    include_time_dummy: bool = False,
    space_dummy_ci: str = "scalar",
    time_dummy_encoding: str = "integer",
    time_dummy_ci: str = "scalar",
    ci: CiArg = None,
) -> AnalysisResult: ...

def discover_pcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_pcmci_plus(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_pc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_ges(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    max_cond_size: int = 2,
    threads: int = 1,
    screen_pc: bool = False,
    max_subset: int | None = None,
) -> PcmciDiscoveryResult: ...

def discover_lingam(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    prune_threshold: float = 0.05,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_notears(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    l1: float = 0.1,
    threshold: float = 0.3,
    standardize: bool = True,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_fci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_rfci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    max_cond_size: int = 2,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_lpcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_jpcmci_plus(
    names: list[str],
    env_columns: list[Sequence[NDArray[np.float64]]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: list[float] | None = None,
    threads: int = 1,
    context_names: list[str] | None = None,
    include_space_dummy: bool = True,
    include_time_dummy: bool = False,
    space_dummy_ci: str = "scalar",
    time_dummy_encoding: str = "integer",
    time_dummy_ci: str = "scalar",
) -> PcmciDiscoveryResult: ...

def discover_rpcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    regimes: list[int],
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: list[float] | None = None,
    threads: int = 1,
) -> RpcmciDiscoverySummary: ...

def two_regime_half_split(series_len: int) -> list[int]: ...

def discover_exact_dag_posterior(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior: ...

def discover_order_mcmc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    require_diagnostics_gate: bool = True,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior: ...

def discover_structure_mcmc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior: ...

def discover_ci_screened_posterior(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    alpha: float = 0.05,
    fdr: bool = True,
    ci: str | None = None,
    max_cond_size: int = 2,
    soft_weight: str = "none",
    n_chains: int = 2,
    n_warmup: int = 300,
    n_draws: int = 600,
    thin: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior: ...

def discover_dbn_posterior(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    force_mcmc: bool = False,
    n_chains: int = 2,
    n_warmup: int = 200,
    n_draws: int = 400,
    thin: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> GraphPosterior: ...

def mediation_effects_summary(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    mediator: str,
    outcome: str,
    *,
    seed: int = 1,
    threads: int = 1,
) -> MediationEffectsSummary: ...

def predict_intervened_summary(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    target: str,
    parent: str,
    *,
    parent_lag: int = 1,
    level: float = 1.0,
) -> PredictSummary: ...

def counterfactual_ite(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    active: float,
    control: float,
    *,
    seed: int = 0,
    threads: int = 1,
) -> GcmIteResult: ...

def sample_do(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    *,
    seed: int = 0,
    threads: int = 1,
    mechanism_wrappers: dict[str, Any] | None = None,
) -> GcmSampleResult: ...

def sample_interventional_distribution(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    outcome: str | None = None,
    *,
    seed: int = 0,
    threads: int = 1,
) -> GcmSampleResult: ...

def attribute_path_specific(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    path_nodes: list[str] | None = None,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 0,
    threads: int = 1,
) -> ChangeAttributionResult: ...

def attribute_paths(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    sources: list[str],
    outcome: str,
    *,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 0,
    threads: int = 1,
) -> ChangeAttributionResult: ...

def attribute_distribution_change(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int = 1,
) -> ChangeAttributionResult: ...

def attribute_distribution_change_robust(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    n_samples: int = 500,
    seed: int = 0,
    threads: int = 1,
) -> ChangeAttributionResult: ...

def attribute_structure_change(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
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
    threads: int = 1,
) -> ChangeAttributionResult: ...

def anomaly_attribution(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcomes: list[str],
    *,
    max_units: int = 0,
) -> list[AnomalyScores]: ...

def attribute_unit_change(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    *,
    max_units: int = 0,
    seed: int = 0,
    threads: int = 1,
) -> ChangeAttributionResult: ...

def attribute_feature_relevance(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    *,
    delta: float = 1.0,
    n_samples: int = 200,
    seed: int = 0,
    threads: int = 1,
) -> list[FeatureRelevance]: ...

def mechanism_change_detection(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    *,
    seed: int = 0,
    threads: int = 1,
) -> list[MechanismChangeDetection]: ...

def rank_root_causes(
    attribution: ChangeAttributionResult,
    *,
    seed: int = 0,
    threads: int = 1,
) -> list[Contribution]: ...

def rank_designs(
    graph_weights: list[float],
    identified: list[int],
    graph_keys: list[int],
    candidates: list[dict[str, Any]],
    objective: str | dict[str, Any] | None = None,
    *,
    query_id: int | None = None,
    model_ids: list[int] | None = None,
    decision_id: int | None = None,
    query_id_unlock: list[tuple[int, list[int]]] | None = None,
    env_id_unlock: list[tuple[int, list[int]]] | None = None,
    identified_under_intervention: list[int] | None = None,
    graph_features: list[int] | None = None,
    effect_width: dict[str, Any] | None = None,
    model_loglik: dict[str, Any] | None = None,
    max_cost: float | None = None,
    max_sample_budget: int | None = None,
    min_batches: int = 2,
    max_batches: int = 64,
    batch_size: int = 8,
    rank_uncertainty_threshold: float = 0.05,
    seed: int = 0,
    threads: int = 1,
) -> DesignRanking: ...

def evaluate_decision_py(
    actions: list[float],
    outcomes: list[float],
    utility: Callable[..., Any],
) -> DecisionEvaluation: ...

def decode_posterior_artifact(bytes: list[int] | bytes) -> PosteriorArtifact: ...
def encode_posterior_artifact(artifact: PosteriorArtifact) -> bytes: ...
class Dag:
    @classmethod
    def from_edges(cls, names: list[str], edges: list[tuple[str, str]]) -> Dag: ...
    @classmethod
    def from_dot(cls, dot: str) -> Dag: ...
    def nodes(self) -> list[str]: ...
    def edges(self) -> list[tuple[str, str]]: ...
    def parents(self, name: str) -> list[str]: ...
    def children(self, name: str) -> list[str]: ...
    def node_count(self) -> int: ...
    def to_dot(self) -> str: ...

class Cpdag:
    @classmethod
    def from_directed_undirected(
        cls,
        names: list[str],
        directed: list[tuple[str, str]],
        undirected: list[tuple[str, str]] | None = None,
    ) -> Cpdag: ...
    @classmethod
    def from_edges(
        cls,
        names: list[str],
        edges: list[tuple[str, str, str]],
    ) -> Cpdag: ...
    def nodes(self) -> list[str]: ...
    def edges(self) -> list[tuple[str, str, str]]: ...
    def parents(self, name: str) -> list[str]: ...
    def children(self, name: str) -> list[str]: ...
    def undirected_neighbors(self, name: str) -> list[str]: ...
    def try_into_dag(self) -> Dag: ...
    def node_count(self) -> int: ...
    @classmethod
    def from_dot(cls, dot: str) -> Cpdag: ...
    def to_dot(self) -> str: ...
    @classmethod
    def from_json(cls, json: str) -> Cpdag: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_gml(cls, gml: str) -> Cpdag: ...
    def to_gml(self) -> str: ...
    @classmethod
    def from_networkx_node_link(cls, json: str) -> Cpdag: ...
    def to_networkx_node_link(self) -> str: ...

class Pag:
    @classmethod
    def from_marked_edges(
        cls,
        names: list[str],
        edges: list[tuple[str, str, str, str]],
    ) -> Pag: ...
    def nodes(self) -> list[str]: ...
    def neighbors(self, name: str) -> list[tuple[str, str, str]]: ...
    def directed_children(self, name: str) -> list[str]: ...
    def node_count(self) -> int: ...
    @classmethod
    def from_dot(cls, dot: str) -> Pag: ...
    def to_dot(self) -> str: ...
    @classmethod
    def from_json(cls, json: str) -> Pag: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_gml(cls, gml: str) -> Pag: ...
    def to_gml(self) -> str: ...
    @classmethod
    def from_networkx_node_link(cls, json: str) -> Pag: ...
    def to_networkx_node_link(self) -> str: ...

class Admg:
    @classmethod
    def from_edges(
        cls,
        names: list[str],
        directed: list[tuple[str, str]],
        bidirected: list[tuple[str, str]] | None = None,
    ) -> Admg: ...
    def nodes(self) -> list[str]: ...
    def parents(self, name: str) -> list[str]: ...
    def children(self, name: str) -> list[str]: ...
    def bidirected_neighbors(self, name: str) -> list[str]: ...
    def node_count(self) -> int: ...
    @classmethod
    def from_dot(cls, dot: str) -> Admg: ...
    def to_dot(self) -> str: ...
    @classmethod
    def from_json(cls, json: str) -> Admg: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_gml(cls, gml: str) -> Admg: ...
    def to_gml(self) -> str: ...
    @classmethod
    def from_networkx_node_link(cls, json: str) -> Admg: ...
    def to_networkx_node_link(self) -> str: ...

class TemporalDag:
    @classmethod
    def from_lagged_edges(
        cls,
        names: list[str],
        edges: list[tuple[str, int, str, int]],
    ) -> TemporalDag: ...
    def nodes(self) -> list[tuple[str, int]]: ...
    def edges(self) -> list[tuple[str, int, str, int]]: ...
    def node_count(self) -> int: ...

class TemporalCpdag:
    @classmethod
    def from_lagged_edges(
        cls,
        names: list[str],
        directed: list[tuple[str, int, str, int]],
        undirected: list[tuple[str, int, str, int]] | None = None,
    ) -> TemporalCpdag: ...
    def try_into_temporal_dag(self) -> TemporalDag: ...
    def node_count(self) -> int: ...

class TemporalPag:
    @classmethod
    def from_marked_lagged_edges(
        cls,
        names: list[str],
        edges: list[tuple[str, int, str, int, str, str]],
    ) -> TemporalPag: ...
    def node_count(self) -> int: ...

def analyze_ate_pag(
    names: list[str],
    columns: list[object],
    graph: Pag,
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    refute: bool | str = True,
    validators: list[object] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...
def analyze_ate_cpdag(
    names: list[str],
    columns: list[object],
    graph: Cpdag,
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    refute: bool | str = True,
    validators: list[object] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...
def analyze_ate_admg(
    names: list[str],
    columns: list[object],
    graph: Admg,
    treatment: str,
    outcome: str,
    *,
    control_level: float = 0.0,
    active_level: float = 1.0,
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    refute: bool | str = True,
    validators: list[object] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
) -> AteAnalysisResult: ...

def dag_from_dot(dot: str) -> tuple[int, list[tuple[int, int]]]: ...
def dag_to_dot(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def dag_from_json(json: str) -> tuple[int, list[tuple[int, int]], list[str] | None]: ...
def dag_to_json(
    node_count: int,
    edges: list[tuple[int, int]],
    variable_names: list[str] | None = None,
) -> str: ...
def dag_from_gml(gml: str) -> tuple[int, list[tuple[int, int]]]: ...
def dag_to_gml(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def dag_from_networkx_node_link(json: str) -> tuple[int, list[tuple[int, int]]]: ...
def dag_to_networkx_node_link(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def dag_from_networkx_adjacency(json: str) -> tuple[int, list[tuple[int, int]]]: ...
def dag_to_networkx_adjacency(
    node_count: int,
    edges: list[tuple[int, int]],
    variable_names: list[str] | None = None,
) -> str: ...
class CausalState:
    def __init__(self, cache_bytes: int = 1_048_576) -> None: ...
    @property
    def version(self) -> int: ...
    @property
    def data_version(self) -> int: ...
    def stale_query_count(self) -> int: ...
    def stale_queries(self) -> list[int]: ...
    def batch_ids(self) -> list[str]: ...
    def append_data(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
    ) -> int: ...
    def replace_data(
        self,
        names: list[str] | None = None,
        columns: Sequence[NDArray[np.float64]] | None = None,
    ) -> int: ...
    def get_batch(
        self, batch_id: str
    ) -> tuple[list[str], list[NDArray[np.float64]]]: ...
    def batch_nrows(self, batch_id: str) -> int: ...
    def add_graph_evidence(
        self, evidence_id: str, fingerprint: int, bytes: int
    ) -> int: ...
    def graph_evidence(self) -> list[tuple[str, int, int]]: ...
    def add_constraint(self, constraint_id: str, fingerprint: int) -> int: ...
    def remove_constraint(self, constraint_id: str) -> int: ...
    def constraints(self) -> list[tuple[str, int]]: ...
    def update_assumption(self, kind: str) -> int: ...
    def register_average_effect(
        self, treatment: int, outcome: int
    ) -> tuple[int, int]: ...
    def record_intervention(self, intervention_id: str, fingerprint: int) -> int: ...
    def refresh_results(self, entries: list[tuple[int, int, int]]) -> None: ...
    def ols_ensure(self, key: str, ncols: int) -> None: ...
    def ols_append_row(self, key: str, row: list[float], y: float) -> None: ...
    def ols_get(self, key: str) -> dict[str, Any]: ...
    def cov_ensure(self, key: str, dim: int) -> None: ...
    def cov_update(self, key: str, row: list[float]) -> None: ...
    def cov_get(self, key: str) -> dict[str, Any]: ...
    def particle_filter_init(
        self,
        key: str,
        n_particles: int,
        *,
        a: float = 0.9,
        process_std: float = 0.3,
        obs_std: float = 0.5,
        seed: int = 1,
    ) -> None: ...
    def particle_filter_step(self, key: str, y: float) -> None: ...
    def particle_filter_get(self, key: str) -> dict[str, Any]: ...

def causal_state_append(n_appends: int = 2, cache_bytes: int = 1_048_576) -> tuple[int, int]: ...
def encode_model_bundle(
    variable_names: list[str],
    edges: list[tuple[int, int]],
    mechanisms: list[tuple[str, float | None, list[float] | None, float | None]],
) -> bytes: ...

def validate_pcmci_block_bootstrap(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_pcmci_false_positive(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    transform: str = "permute",
    replicates: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_pcmci_alpha_sensitivity(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    alphas: list[float],
    *,
    max_lag: int = 1,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_pcmci_lag_sensitivity(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    max_lags: list[int],
    *,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_pcmci_ci_sensitivity(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    ci_names: list[str],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_pcmci_plus_orientation(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_synthetic_null_calibration(
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_sim: int = 20,
    n_obs: int = 100,
    n_vars: int = 3,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_environment_holdout(
    names: list[str],
    env_columns: list[list[NDArray[np.float64]]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_discovery: int = 1,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...

def validate_regime_stability(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    regimes: list[int],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 10,
    block_size: int = 20,
    seed: int = 1,
    threads: int = 1,
) -> dict[str, Any]: ...
def decode_model_bundle(bytes: list[int] | bytes) -> tuple[list[str], list[tuple[int, int]], int]: ...
