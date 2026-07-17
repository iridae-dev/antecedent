"""Type stubs for the native extension module ``causal._native``."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

class CausalError(Exception): ...
class CausalIdentifyError(CausalError): ...
class CausalEstimateError(CausalError): ...
class CausalValidateError(CausalError): ...
class CausalDiscoveryError(CausalError): ...
class CausalModelError(CausalError): ...
class CausalCounterfactualError(CausalError): ...
class CausalDataError(CausalError): ...
class CausalGraphError(CausalError): ...
class CausalSerializationError(CausalError): ...
class CausalCompileError(CausalError): ...
class CausalResourceError(CausalError): ...
class CausalReviewError(CausalError): ...
class CausalUnsupportedError(CausalError): ...

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
    a: str
    a_lag: int
    b: str
    b_lag: int
    at_a: str
    at_b: str

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
    identifier: str | None = None,
    estimator: str | None = None,
    inference: str | None = None,
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    refute: bool = True,
    seed: int = 1,
    bootstrap: int = 50,
    threads: int = 1,
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
    seed: int = 1,
    bootstrap: int = 0,
    threads: int = 1,
) -> AnalysisResult: ...

def discover_pcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
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
    ci: str = "parcorr",
    weights: list[float] | None = None,
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
    ci: str = "parcorr",
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
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> PcmciDiscoveryResult: ...

def discover_rpcmci(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str = "parcorr",
    weights: list[float] | None = None,
    threads: int = 1,
) -> RpcmciDiscoverySummary: ...

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

def gcm_counterfactual_ite(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    active: float,
    control: float,
    seed: int = 0,
    threads: int = 1,
) -> GcmIteResult: ...

def gcm_sample_do(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    seed: int = 0,
    threads: int = 1,
) -> GcmSampleResult: ...

def gcm_sample_interventional_distribution(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    do_value: float,
    n_draws: int,
    outcome: str | None = None,
    seed: int = 0,
    threads: int = 1,
) -> GcmSampleResult: ...

def gcm_attribute_path_specific(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    path_nodes: list[str] | None = None,
    max_paths: int = 64,
    max_len: int = 16,
    seed: int = 0,
    threads: int = 1,
) -> tuple[float, list[tuple[list[str], float]]]: ...

def gcm_distribution_change(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    outcome: str,
    baseline_start: int,
    baseline_end: int,
    comparison_start: int,
    comparison_end: int,
    n_samples: int = 500,
    seed: int = 0,
    threads: int = 1,
) -> tuple[float, list[tuple[str, float]]]: ...

def rank_design_eig(
    graph_weights: list[float],
    identified: list[int],
    graph_keys: list[int],
    measure_var_ids: list[int],
    sampling_increments: list[int],
    seed: int = 0,
    threads: int = 1,
) -> tuple[int, list[float], int]: ...

def causal_state_append_demo(
    n_appends: int = 2,
    cache_bytes: int = 1_048_576,
) -> tuple[int, int]: ...

def decode_posterior_artifact(bytes: list[int] | bytes) -> PosteriorArtifact: ...
def encode_posterior_artifact(artifact: PosteriorArtifact) -> bytes: ...
def parse_dag_dot(dot: str) -> tuple[int, list[tuple[int, int]]]: ...
def format_dag_dot(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def parse_dag_json(json: str) -> tuple[int, list[tuple[int, int]], list[str] | None]: ...
def format_dag_json(
    node_count: int,
    edges: list[tuple[int, int]],
    variable_names: list[str] | None = None,
) -> str: ...
def parse_dag_gml(gml: str) -> tuple[int, list[tuple[int, int]]]: ...
def format_dag_gml(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def parse_dag_networkx_node_link(json: str) -> tuple[int, list[tuple[int, int]]]: ...
def format_dag_networkx_node_link(node_count: int, edges: list[tuple[int, int]]) -> str: ...
def encode_model_bundle(
    variable_names: list[str],
    edges: list[tuple[int, int]],
    mechanisms: list[tuple[str, float | None, list[float] | None, float | None]],
) -> bytes: ...
def decode_model_bundle(bytes: list[int] | bytes) -> tuple[list[str], list[tuple[int, int]], int]: ...
