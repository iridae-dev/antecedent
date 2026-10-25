"""Type stubs for the native extension module ``antecedent._native``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .transport._impl import ZTransportSensitivityResult

CiArg = str | Callable[..., Any] | None

__version__: str
__build_optimized__: bool

def temporal_response_spec() -> dict[str, Any]:
    """Licensed temporal-response query policy from Rust ``TemporalResponseSpec::license``."""

def runtime_refusal_codes() -> list[str]:
    """Registered runtime-refusal reason codes (``parity/reason_codes.toml``)."""

def identification_refusal_codes() -> list[str]:
    """Reason codes of the identification-class refusals the descriptive fallback converts."""

def host_schema_text() -> str:
    """The committed host-integration JSON Schema, exactly as generated."""

def host_status_of_claim(
    kind: str, identification: str, support: str, evaluated: str
) -> str | None:
    """Host status of a claim from its wire names; ``None`` for an unregistered name."""

def host_status_of_operation(applicability: str, readiness: str | None = None) -> str | None:
    """Host status of an operation from its wire names; ``None`` for an unregistered name."""

class CausalError(Exception):
    # What the caller can change to get past a refusal, when it names one;
    # ``None`` otherwise (a class default every subclass inherits).
    remedy: str | None
    # Structured refusal fields (stage, subject, counts, implicated columns, remedy)
    # when the refusal carries them; ``None`` otherwise.
    refusal_fields: dict[str, object] | None
    # Route-native diagnostics the refusal carries: one dict per record with ``route``,
    # ``outcome``, ``failed`` and ``unevaluable_mass`` (``None`` when the route did not
    # report a mass); an empty sequence otherwise.
    route_diagnostics: Sequence[dict[str, object]]

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

class CausalPendingEdge:
    """One unreviewed edge carried on a review error (native-constructed)."""

    source: str
    target: str
    at_source: str
    at_target: str

class CausalReviewError(CausalError):
    kind: str
    algorithm: str | None
    pending_edge_count: int
    # Native raises carry CausalPendingEdge; errors built on the Python side
    # carry antecedent.errors.PendingEdge. Both expose the same four fields.
    pending_edges: Sequence[Any]
    hint: str
    message: str

def set_review_error_class(cls: type) -> None:
    """Register the Python class native code instantiates for review errors."""

def set_unsupported_error_class(cls: type) -> None:
    """Register the Python class native code instantiates for refusals."""

def set_value_error_class(cls: type) -> None:
    """Register the Python class native code instantiates for invalid input."""

def set_not_identified_error_class(cls: type) -> None:
    """Register the Python class native code instantiates for not-identified refusals."""

def accept_rpcmci(
    names: list[str],
    columns: Sequence[Any],
    *,
    regimes: list[int],
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: CiArg = None,
    weights: Sequence[Any] | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
    accept_discovered: bool = True,
) -> Any:
    """Run RPCMCI over labelled regimes and return the reviewed accepted graph."""

class CausalUnsupportedError(CausalError): ...
class CausalCancelledError(CausalError): ...

class CancellationToken:
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...

class AdmittedNumericBuffer:
    """Owner of an admitted numeric allocation; NumPy views keep it alive as their base."""

    @property
    def admitted_bytes(self) -> int: ...

class ArrowLoadInfo:
    row_count: int
    column_count: int
    bytes_copied: int
    bytes_borrowed: int
    diagnostic_count: int
    column_names: list[str]

class NativeTableFile:
    """Parquet file or dataset, or Arrow IPC file or stream, read in bounded chunks."""

    @staticmethod
    def open(path: str, format: str | None = None) -> NativeTableFile: ...
    @property
    def format(self) -> str: ...
    @property
    def paths(self) -> list[str]: ...
    @property
    def declared_rows(self) -> int | None: ...
    def columns(self) -> list[tuple[str, str, bool, str, str | None, bool]]: ...
    def quote(
        self, columns: list[str] | None = None, *, chunk_rows: int | None = None
    ) -> tuple[int | None, int | None, int, int]: ...
    def read(
        self,
        columns: list[str] | None = None,
        *,
        chunk_rows: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> dict[str, Any]: ...
    def content_identity(
        self,
        columns: list[str] | None = None,
        *,
        chunk_rows: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str, int, int]: ...
    def summarize_weighted_gaussian(
        self,
        covariates: list[str],
        response: str,
        *,
        weight: str | None = None,
        intercept: bool = True,
        missing: str = "refuse",
        batch_rows: int = 65536,
        ceiling_bytes: int | None = None,
        control: str = "glm",
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> dict[str, Any]: ...

class AteAnalysisResult:
    def effect_source_json(
        self, units: str, population: str, supplied_value: float
    ) -> tuple[str | None, str | None]: ...
    missing_data_json: str | None
    structural_weight_basis: str | None
    structural_identified_mass: float | None
    structural_unidentified_mass: float | None
    structural_unevaluable_mass: float | None
    structural_identified_set: tuple[float, float] | None
    structural_identified_set_interval: tuple[float, float] | None
    structural_identified_set_interval_level: float | None
    structural_identified_set_interval_method: str | None
    structural_identified_set_interval_truncated: bool | None
    certificate_json: str | None
    unit_effects: list[float] | None
    unit_effect_intervals: list[tuple[float, float]] | None
    unit_effect_intervals_level: float | None
    unit_effect_intervals_method: str | None
    unit_extrapolative: list[bool] | None
    assumptions: list[str]
    support_diagnostics: list[str]
    ate: float
    se_analytic: float
    se_bootstrap: float | None
    bootstrap_replicates_failed: int | None
    adjustment_set: list[str]
    identification_status: str
    refutation_passed: bool
    refutation_ran: bool
    refutation_count: int
    refutations: list[RefutationReportView]
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
    # One ``(route, outcome, failed, unevaluable_mass)`` row per ``route.*`` diagnostic; the
    # mass is ``None`` when the route did not report one.
    route_diagnostics: list[tuple[str, str, bool, float | None]]
    provenance_node_count: int
    plan_id: str
    modality: str
    discovery_algorithm: str | None
    structure_source: str
    graph_review_required: bool
    plan_identifier: str | None
    plan_estimator: str | None
    validation_suite: str | None
    peak_memory_bytes: int | None
    worker_threads: int
    expected_python_crossings: int
    prior_sensitivity_scales: list[float] | None
    prior_sensitivity_alphas: list[float] | None
    prior_sensitivity_means: list[float] | None
    prior_sensitivity_sds: list[float] | None
    prior_sensitivity_family: str | None
    prior_sensitivity_variance_multipliers: list[float] | None
    conflict_source_ids: list[str] | None
    conflict_alphas_requested: list[float] | None
    conflict_alphas_applied: list[float] | None
    posterior_unidentified_mass: float | None
    posterior_subsampled_out_mass: float | None
    latency_mode: str | None
    wall_time_ns: int | None
    bootstrap_replicates_requested: int | None
    bootstrap_replicates_ok: int | None
    n_draws_effort: int | None
    cancelled: bool
    early_stopped: bool
    stage_timings: list[tuple[str, int]]
    prior_ppc_p_value: float | None
    prior_ppc_observed: float | None
    prior_ppc_predictive_mean: float | None
    prior_ppc_predictive_sd: float | None
    prior_ppc_n_sims: int | None
    posterior_ppc_p_value: float | None
    posterior_ppc_observed: float | None
    posterior_ppc_predictive_mean: float | None
    posterior_ppc_predictive_sd: float | None
    posterior_ppc_n_sims: int | None
    identification: IdentificationSection
    estimate: EstimateSection
    posterior: PosteriorSection
    validation: ValidationSection
    performance: PerformanceSection
    mediation_total: float | None
    mediation_direct: float | None
    mediation_mediated: float | None
    mediation_horizons: list[int]
    mediation_effects: list[float]
    mediation_totals: list[float]
    mediation_directs: list[float]
    mediation_mediated_effects: list[float]
    mediation_identification_statuses: list[str]
    mediation_methods: list[str]
    mediation_adjustments: list[list[tuple[int, int]]]
    mediation_uncertainty_kinds: list[str]
    mediation_standard_deviations: list[float | None]
    mediation_q025: list[float | None]
    mediation_q975: list[float | None]
    mediation_identified_lower: list[float | None]
    mediation_identified_upper: list[float | None]
    mediation_joint_posterior: bool | None
    transport: TransportSection | None
    interference: InterferenceSection | None
    randomized_effect: RandomizedEffectSection | None
    panel_did: PanelDidSection | None
    synthetic_control: SyntheticControlSection | None
    synthetic_did: SyntheticDidSection | None
    local_polynomial_ratio: LocalPolynomialRatioSection | None
    survival: SurvivalSection | None
    posterior_row: PosteriorRowSection | None
    longitudinal_regime: LongitudinalRegimeSection | None
    sequential_contrast: SequentialContrastSection | None
    history_match: HistoryMatchSection | None
    temporal_design: TemporalDesignSection | None
    iv_bounds: IvBoundsSection | None
    association: AssociationSection | None
    quasi_rollout: QuasiRolloutSection | None
    policy_value: PolicyValueSection | None
    continuous_dose_response: ContinuousDoseResponseSection | None
    anomaly: list[AnomalyScores] | None
    change_attribution: ChangeAttributionResult | None
    evidence_status: str | None
    allowlist_reason: str | None
    allowlist_parent: str | None

class ResponseAnalysisResult:
    def refutation_reports(
        self,
    ) -> list[tuple[str, bool, bool, float, float, str | None]]: ...
    def program_basis_json(self) -> str | None: ...
    def horizon_draws_json(self, dose_index: int) -> tuple[str | None, str | None]: ...
    missing_data_json: str | None
    certificate_json: str | None
    treatments: list[str]
    outcomes: list[str]
    points: list[list[float]]
    values: list[list[float]]
    scalar: float | None
    matrix: list[list[float]] | None
    uncertainty_kind: str
    interval_interpretation: str | None
    lower: list[list[float]] | None
    upper: list[list[float]] | None
    level: float | None
    standard_error: float | None
    replicates: int | None
    artifact_id: str | None
    support_status: str
    support_minima: list[float]
    support_maxima: list[float]
    support_point_status: list[str]
    diagnostic_ids: list[str]
    diagnostic_values: list[list[float]]
    diagnostic_details: list[str]
    diagnostic_scopes: list[str]
    warnings: list[str]
    identification: str
    adjustment_set: list[str]
    horizon_adjustment_sets: list[list[str]]
    assumptions: list[str]
    provenance_id: str
    identified_mass: float | None
    unidentified_mass: float | None
    unevaluable_mass: float | None
    subsampled_out_mass: float | None
    completion_count: int | None
    truncated_completions: int | None
    enumeration_capped: bool | None
    mass_scope: str | None
    weight_basis: str | None
    atom_keys: list[int]
    atom_weights: list[float]
    atom_statuses: list[str]
    atom_values: list[list[float]]
    evidence_status: str | None
    allowlist_reason: str | None
    allowlist_parent: str | None
    diagnostics: list[str]
    identifier: str | None
    def response_coordinates(
        self, outcome_units: str, population: str = "target", transform: str = "identity"
    ) -> tuple[str | None, str | None]: ...

class TransportIdentificationResult:
    outcome: str
    transportable: bool
    formula_kind: str | None
    rule: str | None
    reason: str | None
    message: str | None
    selection_targets: list[str]
    marginalize: list[str]
    factor_populations: list[str]
    factor_variables: list[list[str]]
    factor_conditioned_on: list[list[str]]
    factor_interventions: list[list[str]]
    factor_regimes: list[int | None]
    expr_pretty: str | None
    expr_latex: str | None
    leaf_populations: list[str]
    leaf_regimes: list[int | None]
    expr_root: int | None
    expr_wire_json: str | None
    def reload_checked_program(self) -> CheckedTransportProgram: ...
    def load_checked_program(self, wire_json: str) -> CheckedTransportProgram: ...

class CheckedTransportProgram:
    source_root: int
    executable_root: int
    def to_wire_json(self) -> str: ...
    def evaluate_exact(
        self,
        catalog: object | None = None,
        laws: object | None = None,
        at: Mapping[str, float] | None = None,
        *,
        max_support_rows: int = 100_000,
        max_operations: int = 1_000_000,
    ) -> float: ...

class TrialTransportResult:
    rule: str
    ipw: float
    aipw: float | None
    selection_probability_min: float
    selection_probability_max: float
    selection_effective_sample_size: float
    selection_extreme_weight_count: int
    treatment_probability_min: float
    treatment_probability_max: float
    treatment_effective_sample_size: float
    treatment_extreme_weight_count: int

class InterferenceAnalysisResult:
    horvitz_thompson: float
    hajek: float
    conservative_variance: float
    from_probability_method: str
    to_probability_method: str
    minimum_exposure_probability: float

class TransportSection:
    ipw: float
    selection_probability_min: float
    selection_probability_max: float
    selection_effective_sample_size: float
    selection_extreme_weight_count: int
    treatment_probability_min: float
    treatment_probability_max: float
    treatment_effective_sample_size: float
    treatment_extreme_weight_count: int

class InterferenceSection:
    horvitz_thompson: float
    hajek: float
    conservative_variance: float
    from_probability_method: str
    to_probability_method: str
    minimum_exposure_probability: float
    from_exposed_units: int | None
    to_exposed_units: int | None
    from_exposed_clusters: int | None
    to_exposed_clusters: int | None
    clusters: int | None
    maximum_exposure_probability: float | None
    pointwise_interval: tuple[float, float, float, float, int, int, str] | None
    interval_unavailable_reason: str | None
    graphless_support_status: str | None

class RandomizedEffectSection:
    effect: float
    estimand: str
    intention_to_treat_effect: float | None
    first_stage_effect: float | None
    received_treatment: list[bool] | None
    randomization_p_value: float | None
    randomization_allocations: int | None
    second_factor_effect: float | None
    factorial_interaction: float | None
    second_factor_variance: float | None
    factorial_interaction_variance: float | None
    multi_arm_values: list[tuple[str, float, float, int]]
    variance_upper_bound: float
    standard_error: float | None
    interval_95: tuple[float, float] | None
    second_factor_interval_95: tuple[float, float] | None
    factorial_interaction_interval_95: tuple[float, float] | None
    multi_arm_intervals_95: list[tuple[float, float] | None]
    minimum_assignment_probability: float
    assignment_design: str
    blocks: list[str]
    periods: list[str]
    control_units: int
    treatment_units: int
    uncertainty: str
    graphless_support_status: str | None
    assignment_units: list[str]
    outcome_units: list[str]
    control_arm: str
    treatment_arm: str

class PanelDidSection:
    effect: float
    standard_error: float | None
    interval_95: tuple[float, float] | None
    treated_subjects: int
    comparison_subjects: int
    clusters: int
    uncertainty: str
    graphless_support_status: str | None
    event_time_effects: list[tuple[int, int, int, float, int, int, float, int]]
    event_time_intervals_95: list[tuple[float, float] | None]
    augmented: tuple[float, float, float, bool] | None

class SyntheticControlSection:
    effect: float
    pre_treatment_rmse: float
    donor_weights: list[tuple[str, float]]
    placebo_effects: list[float]
    placebo_rank: float
    effective_donors: float
    n_pre_periods: int
    n_post_periods: int
    uncertainty: str
    randomization_p_value: float | None
    randomization_null_effect: float | None
    randomization_statistics: list[tuple[str, float]]
    unadjusted_effect: float | None
    outcome_model_correction: float | None
    augmentation_ridge: float | None

class SyntheticDidSection:
    effect: float
    pre_treatment_rmse: float
    donor_weights: list[tuple[str, float]]
    time_weights: list[tuple[int, float]]
    n_donors: int
    n_pre_periods: int
    n_post_periods: int
    uncertainty: str
    randomization_p_value: float | None
    randomization_null_effect: float | None
    randomization_statistics: list[tuple[str, float]]

class LocalPolynomialRatioSection:
    effect: float
    reduced_form: float
    first_stage: float
    cutoff: float
    bandwidth: float
    kink: bool
    n_left: int
    n_right: int
    standard_error: float
    ci_lower: float | None
    ci_upper: float | None
    reduced_form_standard_error: float
    first_stage_standard_error: float
    uncertainty: str
    graphless_support_status: str | None

class DesignEffectSection:
    kind: str
    group: int | None
    period: int | None
    event_time: int | None
    unit: str | None
    estimate: float
    standard_error: float | None
    interval: tuple[float, float] | None
    band: tuple[float, float] | None
    treated: int
    comparison: int

class QuasiRolloutSection:
    design: str
    method: str
    effects: list[DesignEffectSection]
    uncertainty: str
    calibration: str
    replicates_requested: int
    replicates_ok: int
    replicates_failed: int
    critical_value: float | None
    diagnostics: list[tuple[str, float]]
    licensed: bool

class SurvivalSection:
    times: list[float]
    control: list[float]
    treated: list[float]
    rmst_control: float | None
    rmst_treated: float | None
    target_cause: int | None
    tau: float
    minimum_event_risk_set: int | None
    uncertainty: str
    rmst_difference_interval: list[float] | None
    difference_at_tau_interval: list[float] | None
    bootstrap_replicates_requested: int | None
    bootstrap_replicates_ok: int | None
    assignment_counts: tuple[int, int]
    censoring_survival_provenance: str | None
    difference_band: (
        tuple[list[float], list[float], list[float], list[float], int, str | None] | None
    )
    band_unavailable_reason: str | None

class PosteriorRowSection:
    estimator: str
    target: str
    uncertainty: str
    calibration: str
    licensed: bool
    coordinates: list[str]
    mean: list[float]
    lower: list[float]
    upper: list[float]
    covariance: list[float] | None
    exact_mean: list[float] | None
    exact_covariance: list[float] | None
    draws: int
    extras: dict[str, float]
    vectors: dict[str, list[float]]

class LongitudinalRegimeSection:
    method: str
    graphless_support_status: str | None
    rule_id: str | None
    rule_version: str | None
    rule_provenance: str | None
    value: float
    value_standard_error: float | None
    value_interval_95: tuple[float, float] | None
    period_intervals_95: list[tuple[float, float]] | None
    interval_reason: str | None
    effective_sample_size: float
    matched_observed_fraction: float
    maximum_weight: float
    minimum_action_probability: float
    minimum_censoring_probability: float
    uncertainty: str
    probability_ownership: str
    period_effects: list[float] | None
    standard_errors: list[float] | None
    stabilizing_numerator_probabilities: list[float] | None
    observed_subjects: int | None

class SequentialContrastSection:
    active_regime: tuple[int, int]
    control_regime: tuple[int, int]
    contrast: float
    active_mean: float
    control_mean: float
    evaluated_rows: int
    minimum_action_probability: float
    min_probability: float
    standard_error: float | None
    interval: tuple[float, float] | None
    coverage: float
    active_scores: list[float]
    control_scores: list[float]
    contrast_scores: list[float]
    uncertainty: str
    fits_computed: int
    fits_reused: int

class HistoryMatchSection:
    contrast: float
    standard_error: float | None
    weight_total: float
    matched_queries: list[int]
    matched_donors: list[int]
    differences: list[float]
    pair_weights: list[float]
    unmatched_treated: list[int]
    periods_matched: int
    caliper: float | None
    uncertainty: str

class IvBoundsIntervalSection:
    level: float
    lower: float
    upper: float
    lower_se: float
    upper_se: float
    critical_value: float
    width_retained: bool
    replicates: int
    infeasible: int
    method: str

class IvBoundsPosteriorSection:
    n_draws: int
    infeasible_draws: int
    lower_mean: float
    upper_mean: float
    lower_sd: float
    upper_sd: float

class IvBoundsSection:
    lower: float
    upper: float
    cells: list[float]
    arm_weights: list[float]
    rows: int
    rows_dropped: int
    interval: IvBoundsIntervalSection | None
    posterior: IvBoundsPosteriorSection | None

class AssociationTermSection:
    label: str
    scale: str
    estimate: float
    standard_error: float | None
    standard_error_method: str | None
    spread_lower: float | None
    spread_upper: float | None
    posterior_mean: float | None
    posterior_sd: float | None
    posterior_lower: float | None
    posterior_upper: float | None
    rows: int

class AssociationResamplingSection:
    method: str
    unit: str
    replicates_requested: int
    replicates_ok: int
    block_length: int | None
    spread_level: float

class AssociationPosteriorSection:
    method: str
    draws: int
    level: float

class AssociationPredictionSection:
    folds: int
    fold_design: str
    cases: int
    score: str
    mean_score: float
    baseline_score: float
    mean_crps: float | None
    calibration: str

class AssociationSection:
    analysis: str
    rule: str
    requested: str
    outcome_kind: str
    modality: str
    terms: list[AssociationTermSection]
    rows: int
    rows_dropped: int
    resampling: AssociationResamplingSection | None
    posterior: AssociationPosteriorSection | None
    prediction: AssociationPredictionSection | None
    assumptions: list[str]
    refused_after: str | None
    uncertainty: str
    calibration: str
    licensed: bool
    ranks_actions: bool
    is_intervention_effect: bool

class TemporalDesignSection:
    design: str
    functional: str
    coordinates: list[str]
    estimate: list[float]
    standard_error: list[float]
    lower: list[float]
    upper: list[float]
    covariance: list[float] | None
    band_coordinates: list[int]
    band_critical_value: float | None
    band_lower: list[float] | None
    band_upper: list[float] | None
    holm_adjusted_p: list[float] | None
    smoothing_bias_bound: list[float] | None
    pairs: int
    clusters: int
    unpaired: int
    minimum_probability: float
    replicates: tuple[int, int] | None
    draws: int | None
    uncertainty: str
    calibration: str
    licensed: bool

class PolicyValueSection:
    policy_value: float
    reference_value: float
    incremental_value: float
    relative_value_gap: float
    treatment_rate: float
    total_cost: float
    policy_standard_error: float
    reference_standard_error: float
    incremental_standard_error: float
    policy_interval_95: tuple[float, float] | None
    incremental_interval_95: tuple[float, float] | None
    prediction_ownership: str
    propensity_min: float
    propensity_max: float
    uncertainty: str
    graphless_support_status: str | None
    uplift_bins: list[tuple[int, float, float, int, tuple[float, float] | None]]
    multi_action_cate: list[
        tuple[str, str, float, int, int, int, float, tuple[float, float] | None]
    ]
    regret: tuple[float, tuple[float, float], list[float], list[float], int] | None

class ContinuousDoseResponseSection:
    points: list[tuple[str, float, float, int, float, float, float, float]]
    bandwidth: float
    density_provenance: str
    uncertainty: str
    fixed_policy: DosePolicyValueSection | None
    graphless_support_status: str | None

class DosePolicyValueSection:
    policy_doses: list[tuple[str, float]]
    reference_doses: list[tuple[str, float]]
    policy_value: float
    reference_value: float
    incremental_value: float
    policy_variance: float
    reference_variance: float
    incremental_variance: float
    policy_interval_95: tuple[float, float] | None
    reference_interval_95: tuple[float, float] | None
    incremental_interval_95: tuple[float, float] | None
    minimum_local_rows: int
    minimum_effective_sample_size: float
    maximum_normalized_weight: float
    minimum_dose_density: float

class ObservationAdjustedOutcomeResult:
    values: list[float]
    weights: list[float]
    method: str

class ObservationResponseResult:
    treatments: list[str]
    outcomes: list[str]
    points: list[list[float]]
    values: list[list[float]]
    scalar: float | None
    matrix: list[list[float]] | None
    uncertainty_kind: str
    lower: list[list[float]] | None
    upper: list[list[float]] | None
    level: float | None
    standard_error: float | None
    replicates: int | None
    artifact_id: str | None
    support_status: str
    support_minima: list[float]
    support_maxima: list[float]
    support_point_status: list[str]
    diagnostic_ids: list[str]
    diagnostic_values: list[list[float]]
    diagnostic_details: list[str]
    diagnostic_scopes: list[str]
    warnings: list[str]
    identification: str
    adjustment_set: list[str]
    assumptions: list[str]
    provenance_id: str

class RefutationReportView:
    refuter: str
    original_ate: float
    refuted_ate: float
    comparison: float
    informative: bool
    passed: bool
    failure_condition: str | None
    replicates: int

class IdentificationSection:
    """Nested identification section, shared shape on both `AteAnalysisResult` and
    `AnalysisResult` (temporal populates every field)."""

    status: str
    method: str
    adjustment_set: list[str]
    assumption_count: int
    derivation_step_count: int

class ScoreTableSection:
    """Cross-fitted AIPW score-table payload on an estimate section."""

    n_rows: int
    n_folds: int
    provenance: str
    columns: list[tuple[int, float | None]]
    scores: list[float]
    row_index: list[int]
    fold_ids: list[int]
    adjustment_set: list[int]
    observed_arm: list[int]
    propensities: list[float]
    observed_outcome: list[float]
    treatment: int
    intervened: list[int]

class ScoreInferenceSection:
    raw_means: list[float]
    lower: list[float]
    upper: list[float]
    level: float
    critical_value: float
    event_n_eff: list[float]
    threshold_supported: list[bool]
    n_eff: float
    n_eff_by_arm: list[float]
    propensity_range: tuple[float, float] | None
    overlap_ok: bool

class EstimateSection:
    """Nested estimate section (top-level scalar fields only)."""

    ate: float | None
    se_analytic: float
    se_bootstrap: float | None
    estimator_id: str
    method: str
    overlap_ess: float | None
    overlap_propensity_min: float | None
    functional_means: list[float] | None
    exceedance_cdf: list[float] | None
    monotone_rearranged: bool
    interaction_structurally_zero: bool | None
    unit_effects_homogeneous: bool | None
    score_table: ScoreTableSection | None
    joint_covariance: list[list[float]] | None
    score_inference: ScoreInferenceSection | None
    sampling_weights: tuple[int, int, float, float] | None
    scenario_effects: list[float] | None
    scenario_intervals: list[tuple[float, float]] | None
    simultaneous_interval: tuple[float, float, float] | None
    adjusted_p_values: tuple[float, float] | None
    family_contrast: tuple[float, float] | None
    family_contrast_interval: tuple[float, float, float] | None
    family_contrast_posterior: tuple[float, float, tuple[float, float], list[float]] | None
    candidate_selection: CandidateSelectionSection | None
    evalue: float | None
    evalue_threshold: float | None
    distribution_atoms: list[DistributionAtomSection] | None
    mean_interval: ProbabilityIntervalSection | None

    cate: list[float] | None
    cate_se: list[float] | None
    cate_leaf_dispersion: list[float] | None
    learner_provenance: list[tuple[str, str, str]]
    outcome_oof_r2: float | None
    treatment_oof_logloss: float | None
    crossfit_folds: int | None
    crossfit_seed: int | None
    penalized_fallback: tuple[str, int, str, str, str] | None
    penalized_support: list[tuple[int, list[str]]]
    penalized_bootstrap: (
        tuple[str, float | None, int, int, int, bool, list[tuple[int, list[float]]], float] | None
    )

class ProbabilityIntervalSection:
    """Bounded interval for one interventional probability, or why none exists."""

    level: float | None
    lower: float | None
    upper: float | None
    unavailable: str | None

class DistributionAtomSection:
    """One interventional-distribution atom and its probability interval."""

    outcomes: list[tuple[str, float | None]]
    conditioning: list[tuple[str, float | None]]
    probability: float
    se_bootstrap: float | None
    interval: ProbabilityIntervalSection | None

class CandidateSelectionSection:
    screen_id: str
    procedure: str
    winner_index: int | None
    family_size: int
    screen_rows: list[int]
    estimate_rows: list[int]
    disjoint: bool

class PosteriorSection:
    """Nested posterior section. Every field is `None` when no posterior was
    computed (frequentist inference)."""

    effect_mean: float | None
    effect_sd: float | None
    q025: float | None
    q975: float | None
    n_draws: int | None
    p_below_zero: float | None
    backend: str | None
    artifact: bytes | list[int] | None
    unidentified_mass: float | None
    subsampled_out_mass: float | None

class ValidationFailureSection:
    validator: str
    reason: str

class ValidationSection:
    """Nested validation section. `passed`/`ran` follow the shared aggregate rule:
    `ran` is whether any refuter ran, `passed` is `ran and all(r.passed for r in reports)`
    — never `True` when nothing ran."""

    passed: bool
    ran: bool
    count: int
    reports: list[RefutationReportView]
    computation_failures: list[ValidationFailureSection]

class PerformanceSection:
    """Nested performance section. Most fields mirror `StudyResult.performance`,
    which both static and temporal execution paths populate. `bootstrap_replicates_ok`,
    `n_draws` (posterior draw effort), and `stage_timings` are the exception: no
    temporal execution path currently records them, so they read `None`/empty on
    the temporal DTO."""

    native_execution: ExecutionFacts | None
    plan_id: str
    modality: str
    peak_memory_bytes: int | None
    latency_mode: str | None
    wall_time_ns: int | None
    bootstrap_replicates_requested: int | None
    bootstrap_replicates_ok: int | None
    n_draws: int | None
    cancelled: bool
    early_stopped: bool
    stage_timings: list[tuple[str, int]]
    bytes_borrowed: int | None
    resampling: ResamplingSection | None

class ResamplingSection:
    """The complete resampling procedure behind a requested bootstrap interval."""

    level: float
    wave_size: int
    min_replicates: int
    max_replicates: int
    se_rel_tol: float | None
    half_width_rel_tol: float | None
    interval: tuple[float, float] | None
    stop: str
    attempted: int
    waves: list[tuple[int, int, int, int, int, int, int]]
    se_rel_mc_error: float | None
    half_width_rel_mc_error: float | None

class PosteriorArtifact:
    n_draws: int
    mean: list[float]
    sd: list[float]
    q025: list[float]
    q975: list[float]
    draws: NDArray[np.float64]
    backend_id: str
    identification: str
    unidentified_mass: float
    subsampled_out_mass: float
    converged: bool
    hessian_condition: float
    quantity_names: list[str]
    treatment_contrast: float | None
    def __init__(
        self,
        n_draws: int,
        mean: list[float],
        sd: list[float],
        q025: list[float],
        q975: list[float],
        draws: list[float],
        backend_id: str,
        identification: str,
        quantity_names: list[str],
        unidentified_mass: float = 0.0,
        converged: bool = True,
        hessian_condition: float = ...,
        treatment_contrast: float | None = None,
        subsampled_out_mass: float = 0.0,
    ) -> None: ...
    def __array__(
        self,
        dtype: Any = None,
        copy: Any = None,
    ) -> NDArray[np.float64]: ...
    @staticmethod
    def from_moments(
        n_draws: int,
        mean: list[float],
        sd: list[float],
        q025: list[float],
        q975: list[float],
        backend_id: str,
        identification: str,
        quantity_names: list[str],
        unidentified_mass: float = 0.0,
        converged: bool = True,
        hessian_condition: float = ...,
        treatment_contrast: float | None = None,
        subsampled_out_mass: float = 0.0,
    ) -> PosteriorArtifact: ...

class DiscoveredLink:
    source: str
    source_lag: int
    target: str
    target_lag: int
    statistic: float
    p_value: float
    adjusted_p_value: float | None
    conditioning_set: list[tuple[str, int]]

class GraphEdge:
    source: str
    source_lag: int
    target: str
    target_lag: int
    at_source: str
    at_target: str

class StabilityMark:
    at_source: str
    at_target: str
    frequency: float

class StabilityEdge:
    source: str
    source_lag: int
    target: str
    target_lag: int
    frequency: float
    in_reference: bool
    marks: list[StabilityMark]

class StabilityStructure:
    frequency: float
    count: int
    edges: list[GraphEdge]

class StructureStabilityResult:
    algorithm_id: str
    base_algorithm: str
    structure_class: str
    variable_names: list[str]
    replicates_requested: int
    replicates_succeeded: int
    replicates_failed: int
    reference_frequency: float
    reference_edges: list[GraphEdge]
    edges: list[StabilityEdge]
    structures: list[StabilityStructure]
    @property
    def route(self) -> Any: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...
    def adjacency_frequency(
        self, source: str, target: str, *, source_lag: int = 0, target_lag: int = 0
    ) -> float: ...

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
    variable_names: list[str]
    background_knowledge: str | None
    row_accounting: dict[str, Any] | None
    @property
    def route(self) -> CompiledRoute | None: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...
    def accepted_graph(self, *, accept_discovered: bool = True) -> Any:
        """The reviewed accepted graph; refuses while edges are pending."""

class CompiledRoute:
    """An executed compiled route: its contract, claim and inspection record."""

    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...
    @property
    def program_identity(self) -> str | None: ...
    def structure_provenance(self) -> dict[str, str] | None: ...
    def accepted_graph(
        self, *, accept_discovered: bool = True
    ) -> tuple[Any, dict[str, str] | None]: ...

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
    lag_masks: list[int] | None
    algorithm: str | None
    atom_kind: str
    mark_masks: list[int] | None
    edge_mcse: list[float] | None
    edge_rhat: list[float] | None
    edge_ess: list[float] | None
    @property
    def route(self) -> CompiledRoute | None: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...
    def accept_posterior(self) -> dict[str, str]: ...
    def to_weighted_samples(self) -> dict[str, object]: ...
    def edge_marginal_matrix(self) -> list[list[float]]: ...
    @classmethod
    def from_atoms(
        cls,
        names: list[str],
        weights: list[float],
        adjacency: list[int],
        *,
        edge_marginals: list[float] | None = None,
        orientation_marginals: list[float] | None = None,
        lagged_edge_marginals: list[float] | None = None,
        lag_masks: list[int] | None = None,
        max_lag: int | None = None,
        ess: float | None = None,
        atom_kind: str = "Dag",
        mark_masks: list[int] | None = None,
    ) -> GraphPosterior: ...
    @classmethod
    def from_graphs(
        cls,
        names: list[str],
        weights: list[float],
        graphs: Sequence[Any],
        *,
        ess: float | None = None,
    ) -> GraphPosterior: ...

class MediationEffectsSummary:
    total: float
    direct: float
    mediated: float

class PredictSummary:
    mean_prediction: float
    n: int

class AnalysisResult:
    certificate_json: str | None
    ate: float | None
    se_analytic: float
    se_bootstrap: float | None
    plan_id: str
    modality: str
    discovery_algorithm: str | None
    structure_source: str
    graph_review_required: bool
    plan_identifier: str | None
    plan_estimator: str | None
    validation_suite: str | None
    peak_memory_bytes: int | None
    identification_status: str
    method: str
    diagnostics: list[str]
    provenance_node_count: int
    refutation_count: int
    refutations: list[RefutationReportView]
    worker_threads: int
    expected_python_crossings: int
    posterior_effect_mean: float | None
    posterior_effect_sd: float | None
    posterior_q025: float | None
    posterior_q975: float | None
    posterior_n_draws: int | None
    posterior_p_below_zero: float | None
    posterior_backend: str | None
    posterior_artifact: bytes | None
    posterior_unidentified_mass: float | None
    posterior_subsampled_out_mass: float | None
    mediation_total: float | None
    mediation_direct: float | None
    mediation_mediated: float | None
    mediation_horizons: list[int]
    mediation_effects: list[float]
    mediation_totals: list[float]
    mediation_directs: list[float]
    mediation_mediated_effects: list[float]
    mediation_identification_statuses: list[str]
    mediation_methods: list[str]
    mediation_adjustments: list[list[tuple[int, int]]]
    mediation_uncertainty_kinds: list[str]
    mediation_standard_deviations: list[float | None]
    mediation_q025: list[float | None]
    mediation_q975: list[float | None]
    mediation_identified_lower: list[float | None]
    mediation_identified_upper: list[float | None]
    mediation_joint_posterior: bool | None
    identification: IdentificationSection
    estimate: EstimateSection
    posterior: PosteriorSection
    validation: ValidationSection
    performance: PerformanceSection
    evidence_status: str | None
    allowlist_reason: str | None
    allowlist_parent: str | None
    structural_weight_basis: str | None
    structural_identified_mass: float | None
    structural_unidentified_mass: float | None
    structural_unevaluable_mass: float | None
    structural_identified_set: tuple[float, float] | None
    structural_identified_set_interval: tuple[float, float] | None
    structural_identified_set_interval_level: float | None
    structural_identified_set_interval_method: str | None
    structural_identified_set_interval_truncated: bool | None
    adjustment_set: list[str]
    assumption_count: int
    derivation_step_count: int
    estimator_id: str

class PreparedAnalysis:
    def identification_json(self) -> str: ...
    def staleness(
        self,
        names: list[str],
        columns: list[Any],
        *,
        policy_json: str | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def export_staleness_baseline(
        self, *, policy_json: str | None = None, seed: int = 1, threads: int | None = None
    ) -> bytes: ...
    def staleness_plan_json(self) -> str: ...
    def refute_target(
        self,
        names: list[str],
        columns: list[Any],
        *,
        suite: str = "full",
        replicates: int = 30,
        alpha: float = 0.05,
        subset_fraction: float = 0.8,
        negative_control_exposure: str | None = None,
        negative_control_outcome: str | None = None,
        mediator: str | None = None,
        mediation_rho_bar: float = 0.1,
        common_cause_strength: float = 0.0,
        prior_scale_factors: list[float] | None = None,
        derivative_step: float | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def refute_bound(
        self, suite: Any, *, seed: int = 1, threads: int | None = None, cancel: Any = None
    ) -> AteAnalysisResult: ...
    def diagnose_json(self, *, seed: int = 1, threads: int | None = None) -> str: ...
    def diagnose_fit_json(self, *, seed: int = 1, threads: int | None = None) -> str: ...
    def plan_rank_drop_json(
        self,
        priority: list[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def estimate_cost_json(self) -> str: ...
    def contract(
        self,
    ) -> dict[str, str]: ...
    def plan_view_json(self) -> str: ...
    def report(self, format: str = "markdown") -> str: ...
    def estimate(
        self,
        names: list[str],
        columns: Sequence[Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> AteAnalysisResult: ...
    def estimate_bound(
        self,
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> AteAnalysisResult: ...
    def estimate_frame(
        self,
        names: list[str],
        columns: Sequence[Any],
        frame: dict[str, Any] | None,
        *,
        response: bool = False,
        refresh: bool = False,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> Any: ...
    def estimate_response(
        self,
        names: list[str],
        columns: Sequence[Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> ResponseAnalysisResult: ...
    def estimate_response_bound(
        self,
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> ResponseAnalysisResult: ...
    def has_execution_contract(self) -> bool: ...
    def execution_contract(
        self,
    ) -> dict[str, str]: ...
    def tier_diagnostics_json(self, *, cancel: CancellationToken | None = None) -> str: ...
    def export_artifact(
        self,
        *,
        artifact_id: str = "prepared-result",
        payload: str = "result",
    ) -> bytes: ...
    def export_contracted_artifact(
        self,
        *,
        artifact_id: str = "prepared-contract",
        quantities_json: str | None = None,
    ) -> bytes: ...
    @staticmethod
    def identify_existing(
        names: list[str],
        edges: list[tuple[str, str]],
        kind: Any,
        treatments: list[str],
        outcomes: list[str],
        *,
        mediators: list[str] = ...,
        contrast: str = "mediated",
        control_level: float = 0.0,
        active_level: float = 1.0,
        at: list[float] | None = None,
        direction: list[float] | None = None,
        order: int = 1,
        scale: str = "identity",
        weighting: str = "observed",
    ) -> tuple[str, str, list[str], str]: ...
    def inspect(
        self,
    ) -> dict[str, str]: ...
    @property
    def names(self) -> list[str]: ...
    def plan_summary(
        self,
    ) -> dict[str, str]: ...
    @staticmethod
    def prepare(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        treatment: Any,
        outcome: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        identifier: str | None = None,
        estimator: str | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        outcome_functional: dict[str, Any] | None = None,
        running_variable: str | None = None,
        cutoff: float | None = None,
        bandwidth: float | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_class_ate(
        names: list[str],
        columns: Sequence[Any],
        graph: Any,
        treatment: Any,
        outcome: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        identifier: str | None = None,
        estimator: str | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        outcome_functional: dict[str, Any] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_class_response(
        names: list[str],
        columns: Sequence[Any],
        graph: Any,
        kind: Any,
        treatments: list[str],
        outcomes: list[str],
        *,
        grid: list[float] | None = None,
        intervention_kinds: list[str] | None = None,
        intervention_parameters: list[list[float]] | None = None,
        identifier: str | None = None,
        estimator: str | None = None,
        response_options: dict[str, Any] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_conditional(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        treatment: Any,
        outcome: Any,
        modifier: Any,
        *,
        graph: Any = None,
        control_level: float = 0.0,
        active_level: float = 1.0,
        identifier: str | None = None,
        estimator: str | None = None,
        outcome_functional: dict[str, Any] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_dbn_posterior_mediation(
        names: list[str],
        columns: Sequence[Any],
        treatment: Any,
        mediator: Any,
        outcome: Any,
        *,
        contrast: str = "mediated",
        control_level: float = 0.0,
        active_level: float = 1.0,
        horizons: list[int] | None = None,
        max_lag: int = 1,
        force_mcmc: bool = False,
        n_chains: int = 2,
        n_warmup: int = 200,
        mcmc_draws: int = 400,
        posterior: GraphPosterior | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_dbn_posterior_temporal(
        names: list[str],
        columns: Sequence[Any],
        treatment: Any,
        outcome: Any,
        *,
        policy: str = "pulse",
        window: tuple[int, int] | None = None,
        treatment_lag: int = 1,
        horizon_steps: int = 1,
        control_level: float = 0.0,
        active_level: float = 1.0,
        max_history_lag: int | None = None,
        max_lag: int = 1,
        force_mcmc: bool = False,
        n_chains: int = 2,
        n_warmup: int = 200,
        mcmc_draws: int = 400,
        posterior: GraphPosterior | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_dbn_posterior_response(
        names: list[str],
        columns: Sequence[Any],
        kind: Any,
        treatments: list[str],
        outcomes: list[str],
        *,
        grid: list[float] | None = None,
        intervention_kinds: list[str] | None = None,
        intervention_parameters: list[list[float]] | None = None,
        horizons: list[int],
        policy: str = ...,
        treatment_lag: int = ...,
        max_history_lag: int | None = None,
        max_lag: int = 1,
        force_mcmc: bool = False,
        n_chains: int = 2,
        n_warmup: int = 200,
        mcmc_draws: int = 400,
        posterior: GraphPosterior | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_derivative(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        kind: Any,
        treatments: list[str],
        outcomes: list[str],
        *,
        at: list[float] | None = None,
        direction: list[float] | None = None,
        order: int = 1,
        scale: str = "identity",
        weighting: str = "observed",
        response_options: dict[str, Any] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_distribution(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        outcome: Any,
        interventions: dict[str, float],
        *,
        graph: Any = None,
        conditioning: list[str] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_transport(
        names: list[str],
        columns: Sequence[Any],
        graph: Admg,
        selections: list[str],
        source_population: str,
        target_population: str,
        source_experiments: list[str],
        kind: str,
        treatments: list[str],
        outcomes: list[str],
        trial: str,
        selection_probability: str,
        treatment_probability: str,
        *,
        grid: list[float] | None = None,
        at: list[float] | None = None,
        direction: list[float] | None = None,
        order: int = 1,
        scale: str = "identity",
        weighting: str = "observed",
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
        catalog: object | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_interference(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        outcome: str,
        network: list[tuple[int, int, float]],
        realized_assignment: list[bool],
        assignment_kind: str,
        assignment_probabilities: list[float],
        treated: int,
        clusters: list[int],
        treated_clusters: int,
        exposure: str,
        from_level: tuple[float, float],
        to_level: tuple[float, float],
        *,
        probability_draws: int = 10_000,
        low_probability: float = 0.0,
        high_probability: float = 0.0,
        realized_saturation: list[float] | None = None,
        propensity_from: list[float] | None = None,
        propensity_to: list[float] | None = None,
        propensity_provenance: str = "known",
        assume_network_exchangeability: bool = False,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_randomized_effect(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        realized_assignment: list[bool],
        assignment_probabilities: list[float],
        assignment_units: list[str],
        outcome_units: list[str],
        treatment_arms: tuple[str, str],
        design_kind: str,
        treated_units: int | None = None,
        blocks: list[str] | None = None,
        treated_per_row: list[int] | None = None,
        *,
        treated_clusters: int | None = None,
        fixed_cuped: tuple[str, float] | None = None,
        ancova_covariates: list[str] | None = None,
        periods: list[str] | None = None,
        received_treatment: list[bool] | None = None,
        treatment_on_treated: bool = False,
        exact_randomization_test: bool = False,
        second_factor_assignment: list[bool] | None = None,
        factorial_cell_counts: tuple[int, int, int, int] | None = None,
        second_factor_arms: tuple[str, str] | None = None,
        multi_arm_labels: list[str] | None = None,
        multi_arm_assignment: list[int] | None = None,
        multi_arm_probabilities: list[list[float]] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_panel_did(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        treated: list[bool],
        post: list[bool],
        subjects: list[str],
        clusters: list[str],
        *,
        repeated_cross_section: bool = False,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_augmented_panel_did(
        names: list[str],
        columns: Sequence[Any],
        outcome_pre: str,
        outcome_post: str,
        propensity: str,
        untreated_change_prediction: str,
        treated: list[bool],
        subjects: list[str],
        clusters: list[str],
        predictions_cross_fitted: bool,
        *,
        resampling: str | None = None,
        replicates: int = 999,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_staggered_group_time(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        subjects: list[str],
        clusters: list[str],
        periods: list[int],
        cohorts: list[int],
        target_cohort: int,
        target_period: int,
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_staggered_event_study(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        subjects: list[str],
        clusters: list[str],
        periods: list[int],
        cohorts: list[int],
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_synthetic_control(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        units: list[str],
        periods: list[int],
        treated_unit: str,
        intervention_period: int,
        *,
        difference_in_differences: bool = False,
        uniform_unit_randomization: bool = False,
        sharp_null_effect: float | None = None,
        augmentation_ridge: float | None = None,
        conformal: bool = False,
        adoptions: list[tuple[str, int]] | None = None,
        placebo_replicates: int = 0,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_rollout_did(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        subjects: list[str],
        clusters: list[str],
        periods: list[int],
        cohorts: list[int],
        *,
        estimator: str = "callaway_santanna",
        comparison: str = "never_treated",
        nuisance: str = "doubly_robust",
        base_period: str = "varying",
        aggregation: str = "simple",
        anticipation: int = 0,
        covariates: list[str] = [],
        resampling: str | None = None,
        replicates: int = 999,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_covariate_adjusted_did(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        treated: list[bool],
        post: list[bool],
        subjects: list[str],
        clusters: list[str],
        covariates: list[str],
        *,
        nuisance: str = "doubly_robust",
        resampling: str | None = None,
        replicates: int = 999,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_continuous_dose_response(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        dose: str,
        dose_density: str,
        baseline_groups: list[str],
        target_doses: list[float],
        bandwidth: float,
        density_provenance: str,
        *,
        min_local_support: int = 3,
        policy_doses: list[tuple[str, float]] | None = None,
        reference_doses: list[tuple[str, float]] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_local_polynomial_ratio(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        treatment: str,
        running: str,
        cutoff: float,
        bandwidth: float,
        *,
        kink: bool = False,
        bandwidth_rule: str = "declared",
        temporal_replicates: int | None = None,
        density_test: bool = False,
        balance_covariates: list[str] = [],
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_survival(
        names: list[str],
        columns: Sequence[Any],
        duration: str,
        event: str,
        treatment: str,
        tau: float,
        target_cause: int | None = None,
        delayed_entry: str | None = None,
        *,
        independent_given: list[str] = [],
        censoring_times: list[float] = [],
        censoring_columns: list[str] = [],
        censoring_probability_floor: float | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_longitudinal_regime(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        periods: int,
        treatment_history: list[bool],
        regime_actions: list[bool],
        treatment_probabilities: list[float],
        censoring_probabilities: list[float],
        outcome_observed: list[bool],
        subject_ids: list[str],
        fold_ids: list[int],
        excluded_fold_predictions: bool,
        probabilities_known_by_design: bool,
        minimum_probability: float,
        *,
        method: str = "ipw",
        period_outcome_predictions: list[float] = [],
        known_fixed_outcome_predictions: bool = False,
        stabilizing_numerator_probabilities: list[float] = [],
        q_predictions: list[float] = [],
        observation_history: list[bool] = [],
        prediction_fold_ids: list[int] = [],
        rule_id: str | None = None,
        rule_version: str | None = None,
        rule_provenance: str | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_sequential_contrast(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        first_treatment: str,
        second_treatment: str,
        baseline: list[str],
        intermediate: list[str],
        active: tuple[int, int],
        control: tuple[int, int],
        fold_of_row: list[int],
        fold_count: int,
        min_probability: float,
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_history_match(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        treated: str,
        history: list[str],
        period: str | None = None,
        weight: str | None = None,
        caliper: float | None = None,
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_iv_bounds(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        treatment: str,
        instrument: str,
        weight: str | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_association(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        predictors: list[str],
        conditioning: list[str],
        functional: str,
        *,
        folds: int | None = None,
        outcome_scale: str | None = None,
        weight: str | None = None,
        refused_after: str | None = None,
        series: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_temporal_design(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        units: list[str],
        periods: list[int],
        design: str,
        *,
        lag: int = 0,
        quantile_level: float | None = None,
        min_probability: float = 0.05,
        misclassification: tuple[int, int, int, int] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_policy_value(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        assignment: list[bool],
        propensity: list[float],
        actions: list[bool],
        reference: list[bool],
        mu0: list[float],
        mu1: list[float],
        costs: list[float],
        reference_costs: list[float],
        evaluation_subject_ids: list[str],
        disjoint_training_subjects: bool,
        crossfit_fold_ownership_valid: bool,
        *,
        uplift_bins: list[int] = ...,
        uplift_bin_count: int = 0,
        uplift_training_subject_ids: list[str] = ...,
        global_constraints_present: bool = False,
        regret_candidates: list[list[bool]] = ...,
        regret_selected_index: int | None = None,
        regret_training_subject_ids: list[str] = ...,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_multi_action_policy_value(
        names: list[str],
        columns: Sequence[Any],
        outcome: str,
        action_labels: list[str],
        assignment: list[int],
        propensities: list[float],
        actions: list[int],
        reference: list[int],
        costs: list[float],
        reference_costs: list[float],
        available: list[bool],
        capacities: list[int],
        reference_capacities: list[int],
        budget: float | None,
        reference_budget: float | None,
        evaluation_subject_ids: list[str],
        *,
        cate_groups: list[str] = ...,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_anomaly_attribution(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        targets: list[str],
        max_units: int,
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
        reference: tuple[float, float] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_change_attribution(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        outcome: str,
        baseline_start: int,
        baseline_end: int,
        comparison_start: int,
        comparison_end: int,
        *,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_ate(
        names: list[str],
        columns: Sequence[Any],
        treatment: Any,
        outcome: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        estimator_config: Mapping[str, Any] | None = None,
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_conditional(
        names: list[str],
        columns: Sequence[Any],
        treatment: Any,
        outcome: Any,
        modifier: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        outcome_functional: dict[str, Any] | None = None,
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_kind(
        names: list[str],
        columns: Sequence[Any],
        kind: str,
        *,
        treatment: str | None = None,
        outcome: str | None = None,
        mediators: list[str] = ...,
        contrast: str = "mediated",
        control_level: float = 0.0,
        active_level: float = 1.0,
        path_nodes: list[str] | None = None,
        max_paths: int = 64,
        max_len: int = 16,
        interventions: dict[str, float] | None = None,
        conditioning: list[str] | None = None,
        targets: list[str] = ...,
        max_units: int = 0,
        reference: tuple[float, float] | None = None,
        baseline_start: int = 0,
        baseline_end: int = 0,
        comparison_start: int = 0,
        comparison_end: int = 0,
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_transport(
        names: list[str],
        columns: Sequence[Any],
        selections: list[str],
        source_population: str,
        target_population: str,
        source_experiments: list[str],
        kind: str,
        treatments: list[str],
        outcomes: list[str],
        trial: str,
        selection_probability: str,
        treatment_probability: str,
        *,
        grid: list[float] | None = None,
        at: list[float] | None = None,
        direction: list[float] | None = None,
        order: int = 1,
        scale: str = "identity",
        weighting: str = "observed",
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_intervention_response(
        names: list[str],
        columns: Sequence[Any],
        outcome: Any,
        treatments: list[str],
        intervention_kinds: list[str] | None,
        intervention_parameters: list[list[float]] | None,
        *,
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_graph_posterior_response(
        names: list[str],
        columns: Sequence[Any],
        treatment: Any,
        outcome: Any,
        grid: list[float] | None,
        *,
        response_options: dict[str, Any] | None = None,
        posterior: GraphPosterior | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_intervention_response(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        outcome: Any,
        treatments: list[str],
        intervention_kinds: list[str] | None,
        intervention_parameters: list[list[float]] | None,
        *,
        identifier: str | None = None,
        estimator: str | None = None,
        outcome_functional: dict[str, Any] | None = None,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_path_specific(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        treatment: Any,
        outcome: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        path_nodes: list[str] | None = None,
        max_paths: int = 64,
        max_len: int = 16,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_response(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        treatment: Any,
        outcome: Any,
        grid: list[float] | None,
        *,
        identifier: str | None = None,
        estimator: str | None = None,
        accepted: bool = False,
        response_options: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_static_kind(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, str]],
        kind: Any,
        treatment: Any,
        outcome: Any,
        *,
        mediators: list[str] = ...,
        contrast: str = "mediated",
        control_level: float = 0.0,
        active_level: float = 1.0,
        accepted: bool = False,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_temporal_effect(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, int, str, int]],
        treatment: Any,
        outcome: Any,
        *,
        policy: str = "pulse",
        window: tuple[int, int] | None = None,
        treatment_lag: int = 1,
        horizon_steps: int = 1,
        control_level: float = 0.0,
        active_level: float = 1.0,
        max_history_lag: int | None = None,
        modifier: str | None = None,
        accepted: bool = False,
        class_graph: Any = None,
        class_prior_ordered: list[float] | None = None,
        class_prior_pairs: list[tuple[int, float]] | None = None,
        max_completions: str | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
        as_average_effect: bool = False,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_temporal_mediation(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, int, str, int]],
        treatment: Any,
        mediator: Any,
        outcome: Any,
        *,
        contrast: str = "mediated",
        control_level: float = 0.0,
        active_level: float = 1.0,
        horizons: list[int] | None = None,
        accepted: bool = False,
        class_graph: Any = None,
        class_prior_ordered: list[float] | None = None,
        class_prior_pairs: list[tuple[int, float]] | None = None,
        max_completions: str | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_temporal_response(
        names: list[str],
        columns: Sequence[Any],
        edges: list[tuple[str, int, str, int]],
        kind: Any,
        treatments: list[str],
        outcomes: list[str],
        *,
        grid: list[float] | None = None,
        intervention_kinds: list[str] | None = None,
        intervention_parameters: list[list[float]] | None = None,
        horizons: list[int],
        policy: Any = ...,
        treatment_lag: Any = ...,
        max_history_lag: str | None = None,
        feedback: tuple[int, float, float, float, float, str | None, float] | None = None,
        accepted: bool = False,
        class_graph: Any = None,
        class_prior_ordered: list[float] | None = None,
        class_prior_pairs: list[tuple[int, float]] | None = None,
        max_completions: str | None = None,
        observation_kind: str | None = None,
        latent: str | None = None,
        observed: str | None = None,
        censoring: str | None = None,
        event: str | None = None,
        lower: str | None = None,
        upper: str | None = None,
        indicator: str | None = None,
        assumption_kind: str | None = None,
        assumption_variables: list[str] = ...,
        structural_model: str | None = None,
        frame: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_tiered(
        names: list[str],
        columns: Sequence[Any],
        tiers: list[list[str]],
        within_tier: Any,
        treatment: Any,
        outcome: Any,
        *,
        control_level: float = 0.0,
        active_level: float = 1.0,
        estimator: str | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        outcome_functional: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    @staticmethod
    def prepare_tiered_intervention_response(
        names: list[str],
        columns: Sequence[Any],
        tiers: list[list[str]],
        within_tier: Any,
        outcome: Any,
        treatments: list[str],
        intervention_kinds: list[str] | None,
        intervention_parameters: list[list[float]] | None,
        *,
        outcome_functional: dict[str, Any] | None = None,
        seed: int = 1,
        threads: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> PreparedAnalysis: ...
    def preview_transform(
        self,
        intent: Any,
    ) -> dict[str, str]: ...
    def rebind_validators(
        self,
        validators: Any,
    ) -> None: ...
    def reexecute_retarget(
        self,
        artifact: bytes,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> AteAnalysisResult: ...
    def refresh(
        self,
        names: list[str],
        columns: Sequence[Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> AteAnalysisResult: ...
    def refresh_response(
        self,
        names: list[str],
        columns: Sequence[Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
        on_progress: Callable[..., Any] | None = None,
        on_stage: Callable[..., Any] | None = None,
    ) -> ResponseAnalysisResult: ...
    def refute(
        self,
        names: list[str],
        columns: Sequence[Any],
        suite: Any,
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> AteAnalysisResult: ...
    def refute_arrow_c(
        self,
        names: list[str],
        columns: Sequence[Any],
        suite: Any,
        *,
        seed: int = 1,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> AteAnalysisResult: ...
    def retarget(
        self,
        weights: list[float],
        depends_on: list[str],
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> AteAnalysisResult: ...
    def temporal_treatment_score(
        self,
        histories: list[list[float]],
        *,
        penalty: float | None = None,
        intercept_sd: float = 1.0,
        slope_sd: float = 1.0,
        draws: int = 4000,
        seed: int = 0,
    ) -> tuple[str, bytes]: ...
    def treatment_score_posterior(
        self,
        *,
        prior_sd: float = 2.0,
        draws: int = 4000,
        seed: int = 0,
        clip: float = 0.05,
        summary_rows: list[int] | None = None,
    ) -> tuple[str, bytes]: ...
    def snapshot(
        self,
    ) -> PreparedAnalysis: ...
    def streams_stages(
        self,
    ) -> bool: ...
    def validator_names(
        self,
    ) -> list[str]: ...

class GcmIteResult:
    mean_ite: float
    n_units: int
    noise_inference: str
    n_assignments: int
    unit_effects: NDArray[np.float64]
    @property
    def route(self) -> CompiledRoute | None: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...

class GcmSampleResult:
    column_means: list[float]
    n_draws: int
    n_nodes: int
    draws: NDArray[np.float64]
    @property
    def route(self) -> CompiledRoute | None: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...

class Contribution:
    name: str
    score: float

class ChangeAttributionResult:
    total_change: float
    contributions: list[tuple[str, float]]
    path_breakdown: list[tuple[list[str], float]]
    @property
    def route(self) -> CompiledRoute | None: ...
    @property
    def contract(self) -> dict[str, str]: ...
    @property
    def claim(self) -> dict[str, str]: ...
    def inspect(self) -> dict[str, str]: ...

class AnomalyScores:
    outcome: str
    mean_score: float
    n_units: int
    scores: list[float]
    rows: list[int]
    top_row: int | None

class MechanismChangeDetection:
    node: str
    statistic: float
    p_value: float
    adjusted_p_value: float | None
    changed: bool
    n_permutations: int | None
    p_value_floor: float | None

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
    implemented_functional: str
    evaluation: str

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
    chosen_action: int

class FittedGcm:
    names: list[str]
    n_assignments: int
    @property
    def last_route(self) -> CompiledRoute | None: ...
    def structure_provenance(self) -> dict[str, str] | None: ...
    def mechanism_kinds(self) -> list[tuple[str, str]]: ...
    def mechanism_selection(self) -> list[dict[str, Any]]: ...
    def sample_do(
        self,
        interventions: dict[str, float],
        n: int,
        *,
        shifts: dict[str, float] | None = None,
        seed: int = 0,
        threads: int | None = None,
        programs: list[tuple[str, str, BatchCallbackHandle, list[str | None]]] | None = None,
    ) -> GcmSampleResult: ...
    def sample_noise(
        self,
        interventions: dict[str, float],
        n: int,
        *,
        shifts: dict[str, float] | None = None,
        seed: int = 0,
        threads: int | None = None,
        programs: list[tuple[str, str, BatchCallbackHandle, list[str | None]]] | None = None,
    ) -> tuple[GcmSampleResult, NDArray[np.float64]]: ...
    def replay_noise(
        self,
        noise: NDArray[np.float64],
        interventions: dict[str, float] | None = None,
        *,
        shifts: dict[str, float] | None = None,
        policy_salt: int = 0,
        seed: int = 0,
        threads: int | None = None,
        programs: list[tuple[str, str, BatchCallbackHandle, list[str | None]]] | None = None,
    ) -> GcmSampleResult: ...
    def attribute_intrinsic_influence(
        self,
        target: str,
        *,
        independent_noise: bool = False,
        bidirected_edges: list[tuple[str, str]] | None = None,
        allocation: str = "auto",
        n_permutations: int = 200,
        outer_draws: int = 200,
        inner_draws: int = 40,
        replicates: int = 4,
        fit_bootstrap: int = 0,
        posterior_draws: int = 0,
        level: float = 0.95,
        max_sources: int = 64,
        seed: int = 0,
        threads: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def counterfactual_ite(
        self,
        treatment: str,
        outcome: str,
        active: float,
        control: float,
        *,
        seed: int = 0,
        threads: int | None = None,
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
        threads: int | None = None,
    ) -> ChangeAttributionResult: ...
    def attribute_paths(
        self,
        sources: list[str],
        outcome: str,
        *,
        max_paths: int = 64,
        max_len: int = 16,
        seed: int = 0,
        threads: int | None = None,
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
        threads: int | None = None,
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
        threads: int | None = None,
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
        threads: int | None = None,
    ) -> ChangeAttributionResult: ...
    def attribute_unit_change(
        self,
        outcome: str,
        *,
        max_units: int = 0,
        seed: int = 0,
        threads: int | None = None,
    ) -> ChangeAttributionResult: ...
    def attribute_feature_relevance(
        self,
        outcome: str,
        *,
        delta: float = 1.0,
        n_samples: int = 200,
        seed: int = 0,
        threads: int | None = None,
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
        threads: int | None = None,
    ) -> list[MechanismChangeDetection]: ...
    def rank_root_causes(
        self,
        attribution: ChangeAttributionResult,
        *,
        seed: int = 0,
        threads: int | None = None,
    ) -> list[Contribution]: ...

def fit_gcm(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    *,
    threads: int | None = None,
    candidates: list[str] | None = None,
    constraints: dict[str, dict[str, Any]] | None = None,
    splits: str = "shuffled",
    purge: int = 0,
) -> FittedGcm: ...
def fit_gcm_accepted(
    names: list[str],
    columns: Sequence[Any],
    discovery: PcmciDiscoveryResult,
    *,
    threads: int | None = None,
    candidates: list[str] | None = None,
    constraints: dict[str, dict[str, Any]] | None = None,
    splits: str = "shuffled",
    purge: int = 0,
) -> tuple[FittedGcm, list[tuple[str, str]]]: ...

class BatchCallbackHandle:
    def __init__(
        self,
        function: Callable[..., Any],
        name: str,
        inputs: list[str],
        output_kind: str,
        output_params: list[float],
        dependencies: list[tuple[str, str]],
        source_sha256: str | None = None,
        entry_point: str | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def inputs(self) -> list[str]: ...
    @property
    def recoverable(self) -> bool: ...
    def recovery_record(self) -> dict[str, Any] | None: ...

def callback_export_refusal(
    callbacks: list[BatchCallbackHandle],
) -> tuple[str, str] | None: ...
def encode_callback_policy_program(
    names: list[str],
    items: list[
        tuple[
            str | None,
            tuple[str, str, BatchCallbackHandle, list[str | None]] | None,
        ]
    ],
) -> bytes: ...
def policy_program_callback_records(data: bytes) -> list[dict[str, Any]]: ...
def decode_callback_policy_program(
    names: list[str],
    data: bytes,
    supplied: list[BatchCallbackHandle],
) -> list[tuple[str, str, str | None, str | None, list[str | None] | None]]: ...
def verify_callback_recovery(
    name: str,
    inputs: list[str],
    output_kind: str,
    output_params: list[float],
    source_sha256: str,
    entry_point: str,
    dependencies: list[tuple[str, str]],
    supplied: BatchCallbackHandle,
) -> None: ...
def evaluate_graph(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    *,
    ci_test: str = "parcorr",
    alpha: float = 0.05,
    significance: float = 0.05,
    n_permutations: int = 200,
    min_permutations: int = 20,
    max_equivalent_share: float | None = None,
    max_ci_tests: int | None = None,
    ci_replicates: int = 199,
    permutation_seed: int | None = None,
    seed: int = 0,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def attribute_intrinsic_influence(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    target: str,
    *,
    independent_noise: bool = False,
    bidirected_edges: list[tuple[str, str]] | None = None,
    allocation: str = "auto",
    n_permutations: int = 200,
    outer_draws: int = 200,
    inner_draws: int = 40,
    replicates: int = 4,
    fit_bootstrap: int = 0,
    posterior_draws: int = 0,
    level: float = 0.95,
    max_sources: int = 64,
    seed: int = 0,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def sample_policy(
    fitted: FittedGcm,
    policies: str,
    n: int,
    seed: int,
    threads: int | None = None,
) -> GcmSampleResult: ...
def counterfactual_policy(
    fitted: FittedGcm,
    policies: str,
    outcomes: list[str],
    seed: int,
    threads: int | None = None,
) -> dict[str, NDArray[np.float64]]: ...
def sample_temporal_policy(
    fitted: FittedGcm,
    policies: str,
    n_rows: int,
    n_steps: int,
    window: int,
    seed: int,
    start: int = 0,
    history: list[dict[str, list[float]]] | None = None,
    threads: int | None = None,
) -> list[GcmSampleResult]: ...
def counterfactual_temporal_policy(
    fitted: FittedGcm,
    policies: str,
    steps: list[dict[str, NDArray[np.float64]]],
    window: int,
    seed: int,
    start: int = 0,
    history: list[dict[str, list[float]]] | None = None,
    threads: int | None = None,
) -> list[GcmSampleResult]: ...
def policy_support(fitted: FittedGcm, policies: str) -> dict[str, dict[str, Any]]: ...
def policy_value(
    fitted: FittedGcm,
    policies: str,
    outcome: str,
    n: int,
    seed: int,
    threads: int | None = None,
) -> dict[str, Any]: ...

class PreparedPolicy:
    def __init__(self, fitted: FittedGcm, policies: str) -> None: ...
    @property
    def identity(self) -> str | None: ...
    @property
    def program_compilations(self) -> int: ...
    @property
    def model_bindings(self) -> int: ...
    def support(self) -> dict[str, dict[str, Any]]: ...
    def rebind(self, fitted: FittedGcm) -> PreparedPolicy: ...
    def sample(self, n: int, seed: int, threads: int | None = None) -> GcmSampleResult: ...
    def counterfactual(
        self, outcomes: list[str], seed: int, threads: int | None = None
    ) -> dict[str, NDArray[np.float64]]: ...
    def value(
        self, outcome: str, n: int, seed: int, threads: int | None = None
    ) -> dict[str, Any]: ...

def encode_policy_program(names: list[str], policies: str) -> bytes: ...
def decode_policy_program(names: list[str], data: bytes) -> str: ...
def policy_program_identity(names: list[str], policies: str) -> str: ...
def load_float64_columns(
    names: list[str],
    columns: Sequence[NDArray[Any]],
) -> ArrowLoadInfo: ...
def load_float64_panel_columns(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    unit_lengths: list[int],
) -> ArrowLoadInfo: ...
def load_float64_arrow_c_columns(
    names: list[str],
    columns: Sequence[object],
) -> ArrowLoadInfo: ...
def analyze_ate_many(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    queries: list[
        tuple[str, str, float, float, dict[str, object] | None, dict[str, object] | None]
    ],
    *,
    identifier: str | None = None,
    estimator: str | None = None,
    estimator_config: dict[str, object] | None = None,
    refute: bool | str | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: str | None = None,
    screen_id: str | None = None,
    screen_procedure: str | None = None,
    screen_rows: list[int] | None = None,
    estimate_rows: list[int] | None = None,
    tiers: list[list[str]] | None = None,
    within_tier: str | None = None,
) -> list[AteAnalysisResult]: ...

class PreparedBatch:
    def diagnose_json(self, *, seed: int = 1, threads: int | None = None) -> str: ...
    def diagnose_fit_json(self, *, seed: int = 1, threads: int | None = None) -> str: ...
    def plan_rank_drop_json(
        self,
        priority: list[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def estimate_cost_json(self) -> str: ...
    def n_plans(self) -> int: ...
    def shares_covariates(self) -> bool: ...
    def shared_n_folds(self) -> int | None: ...
    def shared_fold_ids(self) -> list[int] | None: ...
    def shared_adjustment_set(self) -> list[int] | None: ...
    def estimate(
        self,
        names: list[str],
        columns: Sequence[Any],
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> list[AteAnalysisResult]: ...
    def retarget_rows(self) -> list[int] | None: ...
    def retarget_scores_source(self) -> str: ...
    def retarget_snapshot(self) -> str | None: ...
    def retarget_family(
        self,
        claims: list[tuple[str, int, list[float], list[str]]],
        contrasts: list[tuple[str, list[tuple[str, float]]]],
        *,
        expected_snapshot: str | None = None,
        simultaneous_level: float = 0.95,
        simultaneous_seed: int = 0,
        simultaneous_draws: int = 100_000,
    ) -> dict[str, Any]: ...

def prepare_ate_batch(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    queries: list[
        tuple[str, str, float, float, dict[str, object] | None, dict[str, object] | None]
    ],
    *,
    identifier: str | None = None,
    estimator: str | None = None,
    estimator_config: dict[str, object] | None = None,
    refute: bool | str | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: str | None = None,
    screen_id: str | None = None,
    screen_procedure: str | None = None,
    screen_rows: list[int] | None = None,
    estimate_rows: list[int] | None = None,
    tiers: list[list[str]] | None = None,
    within_tier: str | None = None,
) -> PreparedBatch: ...
def prepare_cells_batch(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    queries: list[tuple[str, list[str], list[str], list[list[float]], dict[str, object] | None]],
    *,
    identifier: str | None = None,
    estimator: str | None = None,
    estimator_config: dict[str, object] | None = None,
    refute: bool | str | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: str | None = None,
    screen_id: str | None = None,
    screen_procedure: str | None = None,
    screen_rows: list[int] | None = None,
    estimate_rows: list[int] | None = None,
    tiers: list[list[str]] | None = None,
    within_tier: str | None = None,
    family_contrast: str | None = "cell_minus_control",
    family_posterior_draws: int | None = None,
) -> PreparedBatch: ...
def identify_transport(
    graph: Admg,
    selections: list[str],
    source_population: str,
    target_population: str,
    source_experiments: list[str],
    kind: str,
    treatments: list[str],
    outcomes: list[str],
    *,
    grid: list[float] | None = None,
    at: list[float] | None = None,
    direction: list[float] | None = None,
    order: int = 1,
    scale: str = "identity",
    weighting: str = "observed",
    catalog: object | None = None,
) -> TransportIdentificationResult: ...
def roundtrip_expr_arena(
    wire_json: str,
    root: int,
) -> tuple[str, str, list[str], list[int | None], list[int]]: ...
def estimate_trial_transport(
    identification: TransportIdentificationResult,
    outcome: NDArray[np.float64],
    treatment: list[bool],
    trial: list[bool],
    selection_probability: NDArray[np.float64],
    treatment_probability: NDArray[np.float64],
    *,
    mu0: NDArray[np.float64] | None = None,
    mu1: NDArray[np.float64] | None = None,
) -> TrialTransportResult: ...
def estimate_network_interference(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    edges: list[tuple[int, int, float]],
    assignment_kind: str,
    assignment_probabilities: list[float],
    treated: int,
    clusters: list[int],
    treated_clusters: int,
    exposure: str,
    from_level: tuple[float, float],
    to_level: tuple[float, float],
    *,
    probability_draws: int = 10_000,
    seed: int = 1,
) -> InterferenceAnalysisResult: ...
def observation_adjusted_outcome(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    outcome: str,
    observation_kind: str,
    latent: str,
    *,
    observed: str | None = None,
    censoring: str | None = None,
    event: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    indicator: str | None = None,
    assumption_kind: str,
    assumption_variables: list[str] = ...,
    structural_model: str | None = None,
    delayed_entry: str | None = None,
    correction: str = "aipw",
    observation_probability_floor: float = 0.01,
    censoring_survival_floor: float = 0.01,
    crossfit_folds: int = 5,
) -> ObservationAdjustedOutcomeResult: ...
def prepare_observation_response(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    grid: list[float],
    observation_kind: str,
    latent: str,
    *,
    observed: str | None = None,
    censoring: str | None = None,
    event: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    indicator: str | None = None,
    assumption_kind: str,
    assumption_variables: list[str] = ...,
    structural_model: str | None = None,
    delayed_entry: str | None = None,
    correction: str = "aipw",
    observation_probability_floor: float = 0.01,
    censoring_survival_floor: float = 0.01,
    crossfit_folds: int = 5,
    accepted: bool = False,
    seed: int = 1,
    threads: int | None = None,
    options: dict[str, Any] | None = None,
) -> PreparedAnalysis: ...
def gaussian_observation_log_likelihood(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    outcome: str,
    observation_kind: str,
    latent: str,
    means: list[float],
    sigma: float,
    *,
    observed: str | None = None,
    censoring: str | None = None,
    event: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    indicator: str | None = None,
    assumption_kind: str,
    assumption_variables: list[str] = ...,
    structural_model: str | None = None,
) -> float: ...
def _identify_ate(
    names: list[str],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    identifier: str | None = None,
) -> tuple[str, str, list[str]]: ...
def _identify_structure(
    graph: Dag | Cpdag | Pag | TemporalDag | TemporalCpdag | TemporalPag,
    query_kind: str,
    treatment: str,
    outcome: str,
    *,
    identifier: str | None = None,
    modifier: str | None = None,
    policy: str | None = None,
    treatment_lag: int = 1,
    horizon_steps: int = 1,
    active_level: float = 1.0,
    window: tuple[int, int] | None = None,
    treatments: list[str] | None = None,
    include_details: bool = False,
    horizons: list[int] | None = None,
    max_history_lag: int | None = None,
    response_grid: list[float] | None = None,
    response_steps: list[tuple[str, str, list[float]]] | None = None,
) -> tuple[str, str, list[str]] | str: ...
def _identify_ate_admg(
    names: list[str],
    graph: Admg,
    treatment: str,
    outcome: str,
    *,
    identifier: str | None = None,
) -> tuple[str, str, list[str]]: ...
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
    unit_lengths: list[int] | None = None,
    background_knowledge: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
    unit_lengths: list[int] | None = None,
    background_knowledge: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    background_knowledge: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    screen_pc: bool = False,
    max_subset: int | None = None,
    background_knowledge: str | None = None,
) -> PcmciDiscoveryResult: ...
def discover_lingam(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    prune_threshold: float = 0.05,
    seed: int = 1,
    max_cond_size: int = 8,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    missing: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    background_knowledge: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    background_knowledge: str | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
    unit_lengths: list[int] | None = None,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    context_names: list[str] | None = None,
    include_space_dummy: bool = True,
    include_time_dummy: bool = False,
    space_dummy_ci: str = "scalar",
    time_dummy_encoding: str = "integer",
    time_dummy_ci: str = "scalar",
    max_cond_size: int = 2,
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
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
) -> RpcmciDiscoverySummary: ...
def two_regime_half_split(series_len: int) -> list[int]: ...
def discover_exact_dag_posterior(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    seed: int = 1,
    threads: int | None = None,
    equivalence_classes: bool = False,
) -> GraphPosterior: ...
def discover_pag_mcmc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 2000,
    thin: int = 1,
    seed: int = 1,
    threads: int | None = None,
) -> GraphPosterior: ...
def discover_exact_pag_posterior(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    seed: int = 1,
    threads: int | None = None,
) -> GraphPosterior: ...
def discover_exact_dag_marginals(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    max_parents: int | None = None,
    seed: int = 1,
    threads: int | None = None,
) -> tuple[list[float], float]: ...
def discover_bootstrap_stability(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    base: str = "pc",
    replicates: int = 200,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str | None = None,
    max_cond_size: int = 2,
    max_lag: int = 1,
    screen_pc: bool = False,
    max_subset: int | None = None,
    prune_threshold: float = 0.05,
    l1: float = 0.1,
    threshold: float = 0.3,
    standardize: bool = True,
    missing: str | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    unit_lengths: list[int] | None = None,
    background_knowledge: str | None = None,
) -> StructureStabilityResult: ...

class RegimeStabilityResult:
    regimes: list[int]
    tables: list[StructureStabilityResult]
    def table(self, regime: int) -> StructureStabilityResult: ...

def discover_jpcmci_plus_stability(
    names: list[str],
    env_columns: Sequence[Sequence[NDArray[np.float64]]],
    *,
    replicates: int = 200,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    context_names: list[str] | None = None,
    max_cond_size: int = 2,
) -> StructureStabilityResult: ...
def discover_rpcmci_stability(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    regimes: list[int],
    replicates: int = 200,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = True,
    seed: int = 1,
    ci: str | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
    max_cond_size: int = 2,
    min_regime_len: int | None = None,
) -> RegimeStabilityResult: ...
def discover_order_mcmc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    require_diagnostics_gate: bool = True,
    max_parents: int | None = None,
    seed: int = 1,
    threads: int | None = None,
    equivalence_classes: bool = False,
    noise: str = "gaussian",
) -> GraphPosterior: ...
def discover_structure_mcmc(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_draws: int = 1000,
    thin: int = 1,
    max_parents: int | None = None,
    seed: int = 1,
    threads: int | None = None,
    equivalence_classes: bool = False,
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
    threads: int | None = None,
    equivalence_classes: bool = False,
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
    threads: int | None = None,
    unit_lengths: list[int] | None = None,
    equivalence_classes: bool = False,
) -> GraphPosterior: ...
def mediation_effects_summary(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    treatment: str,
    mediator: str,
    outcome: str,
    *,
    seed: int = 1,
    threads: int | None = None,
) -> MediationEffectsSummary: ...
def predict_conditional_summary(
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
    threads: int | None = None,
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
    threads: int | None = None,
    mechanism_wrappers: dict[str, Any] | None = None,
    shift: bool = False,
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
    threads: int | None = None,
    shift: bool = False,
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
    threads: int | None = None,
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
    threads: int | None = None,
) -> ChangeAttributionResult: ...
def temporal_model_query(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    lagged_edges: list[tuple[str, int, str, int]],
    kind: str,
    params_json: str,
    *,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
    seed: int = 0,
    threads: int | None = None,
) -> str: ...
def compose_horizon_draws(
    horizons: list[int],
    columns: Sequence[NDArray[np.float64]],
    evidence_id: str,
    alignment_id: str,
    operation: str,
    *,
    coefficients: list[float] | None = None,
    map: Any = None,
    seed: int = 0,
) -> str: ...
def attribution_uncertainty(
    names: list[str],
    columns: Sequence[NDArray[np.float64]],
    edges: list[tuple[str, str]],
    kind: str,
    params_json: str,
    *,
    posterior_draws: int = 0,
    bootstrap_replicates: int = 0,
    seed: int = 0,
    threads: int | None = None,
) -> str: ...
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
) -> list[MechanismChangeDetection]: ...
def rank_root_causes(
    attribution: ChangeAttributionResult,
    *,
    seed: int = 0,
    threads: int | None = None,
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
    decision: dict[str, Any] | None = None,
    max_cost: float | None = None,
    max_sample_budget: int | None = None,
    min_batches: int = 2,
    max_batches: int = 64,
    batch_size: int = 8,
    rank_uncertainty_threshold: float = 0.05,
    seed: int = 0,
    threads: int | None = None,
) -> DesignRanking: ...
def evaluate_decision_py(
    actions: list[float],
    outcomes: list[float],
    utility: Callable[..., Any],
) -> DecisionEvaluation: ...
def evaluate_binary_policy(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    actions: list[bool],
    propensity: NDArray[np.float64],
    *,
    reference: list[bool] | None = None,
    costs: list[float] = ...,
    reference_costs: list[float] | None = None,
    available: list[bool] | None = None,
    capacity: int | None = None,
    budget: float | None = None,
    reference_capacity: int | None = None,
    reference_budget: float | None = None,
) -> tuple[float, float, float, float, float, float]: ...
def evaluate_binary_policy_doubly_robust(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    actions: list[bool],
    propensity: NDArray[np.float64],
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
    *,
    reference: list[bool] | None = None,
    costs: list[float] = ...,
    reference_costs: list[float] | None = None,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]: ...
def evaluate_longitudinal_regime_value(
    outcome: NDArray[np.float64],
    treatment: NDArray[np.bool_],
    regime: NDArray[np.bool_],
    treatment_probability: NDArray[np.float64],
    outcome_observed: NDArray[np.bool_],
    censoring_survival: NDArray[np.float64],
) -> tuple[float, float, float, float]: ...
def evaluate_sequential_gformula(
    period_outcome_predictions: NDArray[np.float64],
    regime_actions: NDArray[np.bool_],
    treatment_probability: NDArray[np.float64],
    censoring_survival: NDArray[np.float64],
    subject_ids: list[str],
    fold_ids: NDArray[np.int64] | list[int],
    minimum_probability: float,
) -> tuple[float, int, float, float]: ...
def evaluate_sequential_doubly_robust(
    outcome: NDArray[np.float64],
    outcome_observed: NDArray[np.bool_],
    observation_history: NDArray[np.bool_],
    treatment: NDArray[np.bool_],
    regime: NDArray[np.bool_],
    q_prediction: NDArray[np.float64],
    treatment_probability: NDArray[np.float64],
    censoring_probability: NDArray[np.float64],
    subject_ids: list[str],
    fold_ids: NDArray[np.int64] | list[int],
    prediction_fold_ids: NDArray[np.int64] | list[int],
    minimum_probability: float,
) -> tuple[float, int, float, float]: ...
def evaluate_multi_action_policy(
    outcome: NDArray[np.float64],
    assigned: list[int],
    actions: list[int],
    probabilities: NDArray[np.float64],
    *,
    reference: list[int] | None = None,
    costs: list[float] | None = None,
    reference_costs: list[float] | None = None,
    available: list[list[bool]] | None = None,
    capacities: list[int] | None = None,
    reference_capacities: list[int] | None = None,
    budget: float | None = None,
    reference_budget: float | None = None,
) -> tuple[float, float, float, float, float]: ...
def uplift_by_score(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    score_bin: list[int],
    propensity: NDArray[np.float64],
    bin_count: int,
) -> list[tuple[int, float, float, int]]: ...

class FactorialNativeResult:
    cell_means: list[float]
    cell_support: list[int]
    factor_a_effect: float
    factor_b_effect: float
    interaction_effect: float
    factor_a_variance_bound: float
    factor_b_variance_bound: float
    interaction_variance_bound: float

def estimate_factorial_2x2(
    outcome: NDArray[np.float64],
    factor_a: list[bool],
    factor_b: list[bool],
    probability_a: NDArray[np.float64],
    probability_b: NDArray[np.float64],
) -> FactorialNativeResult: ...
def difference_in_differences(
    outcome: NDArray[np.float64], treated: list[bool], post: list[bool]
) -> float: ...
def estimate_cuped_effect(
    outcome: NDArray[np.float64],
    covariate: NDArray[np.float64],
    assignment: list[bool],
    propensity: NDArray[np.float64],
) -> tuple[float, float, float]: ...
def estimate_stratified_effect(
    outcome: NDArray[np.float64], assignment: list[bool], blocks: list[str]
) -> tuple[float, float, float, int]: ...
def exact_randomization_test(
    outcome: NDArray[np.float64], assignment: list[bool], propensity: NDArray[np.float64]
) -> tuple[float, float, int]: ...
def estimate_switchback_effect(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    sequences: list[str],
    propensity: NDArray[np.float64],
) -> tuple[float, float, float, int]: ...
def estimate_ancova_effect(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    covariates: NDArray[np.float64],
) -> tuple[float, float, list[float], int, int]: ...
def estimate_saturation_interference(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    edges: list[tuple[int, int, float]],
    clusters: list[int],
    realized_saturation: list[float],
    low_saturation: float,
    high_saturation: float,
    high_clusters: int,
    exposure: str,
    reference_neighbors: float,
    low_neighbors: float,
    high_neighbors: float,
) -> tuple[
    tuple[float, float, float, int, int, int, int, float],
    tuple[float, float, float, int, int, int, int, float],
    tuple[float, float, float, int, int, int, int, float],
]: ...
def estimate_observational_network_exposure(
    outcome: NDArray[np.float64],
    assignment: list[bool],
    edges: list[tuple[int, int, float]],
    clusters: list[int],
    exposure: str,
    from_level: tuple[float, float],
    to_level: tuple[float, float],
    propensity_from: NDArray[np.float64],
    propensity_to: NDArray[np.float64],
    propensity_provenance: str,
) -> tuple[float, float, float, int, int, int, int, float, float, int]: ...
def randomized_survival_ipcw(
    duration: NDArray[np.float64],
    event_observed: list[bool],
    treated: list[bool],
    times: NDArray[np.float64] | list[float],
    censoring_survival: NDArray[np.float64],
    tau: float,
    minimum_probability: float,
) -> tuple[list[float], list[float], list[float], float, float, int | None, int | None, float]: ...
def randomized_cumulative_incidence_ipcw(
    duration: NDArray[np.float64],
    event_cause: list[int],
    treated: list[bool],
    times: NDArray[np.float64] | list[float],
    censoring_survival: NDArray[np.float64],
    tau: float,
    target_cause: int,
    minimum_probability: float,
) -> tuple[list[float], list[float], list[float], int | None, int | None, float]: ...
def group_time_att(
    outcome: NDArray[np.float64],
    subjects: list[str],
    periods: list[int],
    cohorts: list[int],
    clusters: list[str],
) -> list[tuple[int, int, float, int, int, float, int]]: ...
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
    @classmethod
    def from_json(cls, json: str) -> Dag: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_gml(cls, gml: str) -> Dag: ...
    def to_gml(self) -> str: ...
    @classmethod
    def from_networkx_node_link(cls, json: str) -> Dag: ...
    def to_networkx_node_link(self) -> str: ...
    @classmethod
    def from_networkx_adjacency(cls, json: str) -> Dag: ...
    def to_networkx_adjacency(self) -> str: ...
    def d_separated(self, x: str, y: str, z: list[str] | None = None) -> bool: ...
    def latent_project(self, observed: list[str]) -> Admg: ...

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
    def m_separated(
        self,
        x: str,
        y: str,
        z: list[str] | None = None,
        *,
        max_paths: int = 32,
        max_len: int = 6,
    ) -> bool: ...
    def m_separation_status(
        self,
        x: str,
        y: str,
        z: list[str] | None = None,
        *,
        max_paths: int = 32,
        max_len: int = 6,
    ) -> Literal["separated", "connected", "undetermined"]: ...

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
    def m_separated(self, x: str, y: str, z: list[str] | None = None) -> bool: ...

class TemporalAdmg:
    @classmethod
    def from_lagged_edges(
        cls,
        names: list[str],
        directed: list[tuple[str, int, str, int]],
        bidirected: list[tuple[str, int, str, int]] | None = None,
    ) -> TemporalAdmg: ...
    def node_count(self) -> int: ...
    def variable_names(self) -> list[str]: ...
    def edges(
        self,
    ) -> tuple[list[tuple[str, int, str, int]], list[tuple[str, int, str, int]]]: ...

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
    def get_batch(self, batch_id: str) -> tuple[list[str], list[NDArray[np.float64]]]: ...
    def batch_nrows(self, batch_id: str) -> int: ...
    def add_graph_evidence(self, evidence_id: str, fingerprint: int, bytes: int) -> int: ...
    def graph_evidence(self) -> list[tuple[str, int, int]]: ...
    def add_constraint(self, constraint_id: str, fingerprint: int) -> int: ...
    def remove_constraint(self, constraint_id: str) -> int: ...
    def constraints(self) -> list[tuple[str, int]]: ...
    def update_assumption(self, kind: str) -> int: ...
    def register_average_effect(self, treatment: int, outcome: int) -> tuple[int, int]: ...
    def register_response_curve(
        self,
        treatment: int,
        outcome: int,
        *,
        grid: list[float],
        horizons: list[int] | None = None,
        policy: str = ...,
        treatment_lag: int = ...,
        max_history_lag: int | None = None,
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

def antecedent_state_append(n_appends: int = 2, cache_bytes: int = ...) -> tuple[int, int]: ...
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
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
    threads: int | None = None,
) -> dict[str, Any]: ...
def decode_model_bundle(
    bytes: list[int] | bytes,
) -> tuple[list[str], list[tuple[int, int]], int]: ...
def prior_catalog_filter(
    sources: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[dict[str, Any]]: ...
def prior_catalog_rank(
    reports: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[dict[str, Any]]: ...
def encode_prior_source_meta(meta: dict[str, Any]) -> bytes: ...
def decode_prior_source_meta(bytes: bytes | list[int]) -> dict[str, Any]: ...
def compose_external_priors(
    sources: list[dict[str, Any]],
    baseline_mean: list[float],
    baseline_variance: list[float],
    *,
    conflict: dict[str, Any] | None = None,
    conflict_signals: list[dict[str, Any]] | None = None,
    transport: str | None = None,
    target_population: str | None = None,
    source_populations: list[str | None] | None = None,
    unit_effects: list[float] | None = None,
    transport_weights: list[float] | None = None,
    coef_index: int | None = None,
) -> dict[str, Any]: ...
def conflict_shrink_alpha(
    alpha: float,
    *,
    p_value: float | None = None,
    kl: float | None = None,
    p_min: float = 0.05,
    kl_scale: float = 1.0,
) -> float: ...
def beta_from_moments(
    mean: float,
    variance: float,
) -> tuple[float, float]: ...
def beta_from_mean_and_ess(
    mean: float,
    ess: float,
) -> tuple[float, float]: ...
def gamma_from_moments(
    mean: float,
    variance: float,
) -> tuple[float, float]: ...
def gamma_from_mean_and_ess(
    mean: float,
    ess: float,
) -> tuple[float, float]: ...

class ExternalClaimArtifact:
    source_evidence: SourceEvidenceHandle
    metadata_json: str
    provenance_label: str
    @staticmethod
    def load(data: bytes, expected_identity_json: str) -> ExternalClaimArtifact: ...
    def values_copy(self) -> NDArray[np.float64]: ...
    def export(self, artifact_id: str) -> bytes: ...

def bind_external_response(
    contract_json: str, response_json: str, causal_contract_id: str
) -> tuple[ExternalClaimArtifact | None, str | None]: ...

class ExternalColumnClaimArtifact:
    metadata_json: str
    provenance_label: str
    @staticmethod
    def load(data: bytes, expected_identity_json: str) -> ExternalColumnClaimArtifact: ...
    def covers(self, values: NDArray[np.float64]) -> bool: ...
    def export(self, artifact_id: str) -> bytes: ...

def bind_external_column(
    contract_json: str,
    column_json: str,
    values: NDArray[np.float64],
    causal_contract_id: str,
) -> tuple[ExternalColumnClaimArtifact | None, str | None]: ...

class NativeResponseClaim:
    source_evidence: SourceEvidenceHandle
    execution_program_id: str | None
    coordinates_json: str
    means: list[float]
    point_status: list[str]
    support_status: str
    trust: str
    calibration: str
    program_identity: str
    provenance_id: str
    has_joint_law: bool
    def joint_artifact(
        self, contract_json: str
    ) -> tuple[JointDistributionArtifact | None, str | None]: ...
    def decision_source(self, contract_json: str) -> tuple[str | None, str | None]: ...

def program_binding_identity(binding_json: str) -> tuple[str | None, str | None]: ...
def check_external_program(binding_json: str, claim_json: str) -> tuple[str | None, str | None]: ...
def bind_external_to_program(
    binding_json: str,
    claim_json: str,
    contract_json: str,
    response_json: str,
    causal_contract_id: str,
) -> tuple[ExternalClaimArtifact | None, str | None]: ...
def native_response_claim(
    raw: ResponseAnalysisResult | None,
    projection_json: str,
    binding_json: str,
    snapshot_id: str | None = None,
    rng_id: str | None = None,
    calibration: str | None = None,
    premises_json: str | None = None,
) -> tuple[NativeResponseClaim | None, str | None]: ...

class JointDistributionArtifact:
    source_evidence: SourceEvidenceHandle | None
    semantic: str
    axes: list[str]
    shape: tuple[int, int]
    n_draws: int
    metadata_json: str
    def __init__(self, metadata_json: str, draws: NDArray[np.float64]) -> None: ...
    @staticmethod
    def load(data: bytes, expected_identity_json: str) -> JointDistributionArtifact: ...
    def draws_copy(self) -> NDArray[np.float64]: ...
    def mean(self, coordinate: int) -> float: ...
    def covariance(self, left: int, right: int) -> float: ...
    def joint_product_expectation(self, left: int, right: int) -> float: ...
    def export(self, artifact_id: str) -> bytes: ...

class DecodedCausalArtifact:
    @property
    def artifact_id(self) -> str: ...
    @property
    def format_major(self) -> int: ...
    @property
    def format_minor(self) -> int: ...
    @property
    def payload_kind(self) -> str: ...
    @property
    def variable_names(self) -> list[str]: ...
    @property
    def payload_json(self) -> str: ...
    @property
    def contract_json(self) -> str | None: ...

def encode_causal_artifact(
    payload_kind: str,
    variable_names: list[str],
    payload_json: str,
    artifact_id: str,
    contract_json: str | None = None,
) -> bytes: ...
def decode_causal_artifact(bytes: bytes) -> DecodedCausalArtifact: ...
def accept_analysis_result_contract(bytes: bytes) -> dict[str, str]: ...
def encode_external_estimate_claim(
    *,
    learner: str,
    config: bytes,
    payload: bytes,
    names: list[str],
    treatment: str,
    outcome: str,
    identifier: str,
    status: str,
    confounders: list[str],
    identification: str | None = None,
    data_snapshot: str | None = None,
    snapshot_payload: bytes | None = None,
    scalar_value: float | None = None,
    value_types: list[str] | None = None,
    contrast: tuple[float, float] | None = None,
    modifiers: list[str] | None = None,
) -> bytes: ...
def omitted_defaults() -> dict[str, Any]:
    """Budgets the builder applies when a caller omits them (bootstrap, refute, n_draws, ...)."""

def default_user_threads() -> int:
    """Default product `threads` (`available_parallelism`, capped)."""

def identification_status_names() -> list[str]:
    """Every native identification status string, exhaustively (the verdict table's key set)."""

class ClassicalTransportStage:
    def export(self) -> bytes: ...
    def certificate_json(self) -> str: ...
    def proof_graph_json(
        self, catalog: Any, *, max_steps: int = 100_000, max_depth: int = 256
    ) -> str: ...
    @property
    def identification_status(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    @property
    def outcomes(self) -> list[str]: ...
    @property
    def formula(self) -> str | None: ...
    @property
    def rules(self) -> list[str]: ...
    def evaluate_exact(
        self,
        catalog: Any,
        laws: Any,
        assignments: dict[str, float],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        max_support_rows: int = 1_000_000,
        memory_bytes: int | None = None,
    ) -> tuple[list[list[float]], list[float], str, list[str]]: ...
    def prepare_exact(
        self,
        catalog: Any,
        laws: Any,
        assignments: dict[str, float],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        max_support_rows: int = 1_000_000,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> PreparedExactStage: ...
    def catalog_search(
        self,
        catalog: Any,
        *,
        max_steps: int = 100_000,
        max_depth: int = 256,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...

def identify_classical_transport_stage(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ClassicalTransportStage: ...

class PreparedExactStage:
    @property
    def outcomes(self) -> list[str]: ...
    def estimate(
        self, execution: str | None = None, cancel: CancellationToken | None = None
    ) -> tuple[list[list[float]], list[float], str, list[str]]: ...
    def refresh(
        self, laws: Any, cancel: CancellationToken | None = None
    ) -> tuple[list[list[float]], list[float], str, list[str]]: ...
    def replace_snapshot(self, laws: Any, cancel: CancellationToken | None = None) -> None: ...
    def last_result(self) -> tuple[list[list[float]], list[float], str, list[str]]: ...
    def plan_summary(self) -> dict[str, str]: ...
    def freeze(self) -> PreparedExactStage: ...
    def inspection_json(self) -> str: ...
    def export(self) -> bytes: ...
    def preview_transform(self, intent: str) -> dict[str, str]: ...
    def mechanism_sensitivity(
        self,
        outcome_values: list[float],
        parent_cardinalities: list[int],
        treatment_levels: tuple[int, int],
        max_fraction: float,
        source_kernel_regime: int,
        source_kernel_snapshot: str,
        target_parent_regime: int,
        target_parent_snapshot: str,
        source_kernel: list[tuple[list[int], list[float]]],
        source_parent_law: list[tuple[list[int], float]],
        target_parent_law: list[tuple[int, list[int], float]],
        decision_threshold: float | None = None,
        perturbed_treatment_level: int | None = None,
        perturbed_root_mechanism: str | None = None,
        perturbed_conditional_mechanism: str | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...

class ZTransportStage:
    @property
    def identification_status(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    @property
    def reason(self) -> str | None: ...
    def prepare_exact(
        self,
        catalog: Any,
        laws: Any,
        assignments: dict[str, float],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        max_support_rows: int | None = None,
        memory_bytes: int | None = None,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedZTransportStage: ...
    def prepare_empirical(
        self,
        catalog: Any,
        laws: Any,
        assignments: dict[str, float],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        max_support_rows: int | None = None,
        memory_bytes: int | None = None,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedZTransportStage: ...
    def plan_evidence(
        self,
        catalog: Any,
        candidates: list[Any],
        failure_snapshot: bytes | None = None,
        *,
        max_evaluated: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str, list[ZTransportProposalStage]]: ...
    def failure_snapshot(
        self, catalog: Any, *, cancel: CancellationToken | None = None
    ) -> bytes: ...
    def inspect_proof(self, catalog: Any) -> dict[str, Any]: ...
    def not_certified_inspection(self) -> dict[str, Any] | None: ...
    def decide(
        self,
        catalog: Any,
        *,
        max_steps: int | None = None,
        max_depth: int | None = None,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> Any: ...

class ZTransportProposalStage:
    def export(self) -> bytes: ...
    def replay(self) -> None: ...
    def receive(
        self,
        catalog: Any,
        laws: Any,
        assignments: dict[str, float],
        provider_snapshot: str,
        *,
        empirical: bool = False,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        max_support_rows: int | None = None,
        memory_bytes: int | None = None,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedZTransportStage: ...

class PreparedZTransportStage:
    def estimate(
        self,
        estimator: str | None = None,
        posterior_draws: int | None = None,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
        seed: int | None = None,
    ) -> str: ...
    def export(self) -> bytes: ...
    def mechanism_sensitivity(
        self,
        max_fraction: float,
        decision_threshold: float | None = None,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> ZTransportSensitivityResult: ...
    def export_sensitivity(
        self,
        max_fraction: float,
        decision_threshold: float | None = None,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    @property
    def interval_type(self) -> str: ...
    @property
    def interval_method(self) -> str | None: ...
    @property
    def interval_reason(self) -> str: ...
    @property
    def seed(self) -> int: ...

class PreparedTransportScenariosStage:
    def estimate(
        self, *, memory_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> str: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def aggregate_interval(self) -> None: ...
    def plan_summary(self) -> list[tuple[str, str]]: ...
    def export(self) -> bytes: ...

def prepare_transport_scenarios_stage(
    scenarios: list[
        tuple[
            str,
            Admg,
            list[str],
            float | None,
            list[tuple[str, str, int | None, str | None]] | None,
        ]
    ],
    coordinates: list[tuple[str, str, int | None, str | None]],
    outcomes: list[str],
    treatments: list[str],
    source: str,
    target: str,
    catalog: Any,
    laws: Any,
    assignments: dict[str, float],
    *,
    conditioned_on: list[str] = ...,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedTransportScenariosStage: ...
def consume_transport_scenarios_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class CpdagScenarioRun:
    @property
    def payload_json(self) -> str: ...
    def export(self) -> tuple[bytes | None, str | None]: ...

class ScenarioCovarianceRun:
    @property
    def payload_json(self) -> str: ...
    def export(self) -> tuple[bytes | None, str | None]: ...

def cpdag_completion_scenarios_stage(
    cpdag: Cpdag,
    coordinates: list[tuple[str, str, int | None, str | None]],
    outcomes: list[str],
    treatments: list[str],
    source: str,
    target: str,
    evidence_mode: str,
    evidence: list[tuple[str | None, list[tuple[str, str]] | None, str, Any, str | None]],
    laws: Any,
    assignments: dict[str, float],
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[CpdagScenarioRun | None, str | None]: ...
def consume_cpdag_scenarios_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def scenario_shared_covariance_stage(
    spec_json: str,
    *,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[ScenarioCovarianceRun | None, str | None]: ...
def consume_scenario_covariance_artifact(
    artifact: bytes,
    *,
    max_rows: int = 10_000,
    max_columns: int = 64,
    max_replicates: int = 2000,
    max_compositions: int = 4_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...

class PreparedTemporalTransportStage:
    def estimate(
        self, *, memory_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> str: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def plan(self) -> str: ...
    def interval(self) -> None: ...
    def export(self) -> bytes: ...

def unroll_two_slice_template(
    template: TemporalDag,
    covariates: list[str],
    action: str,
    outcome: str,
    baseline: list[str],
    baseline_effects: list[tuple[str, str]],
    latent_pairs: list[tuple[str, str]],
) -> str: ...
def prepare_temporal_transport_stage(
    graph: Admg,
    selections: list[str],
    slots: tuple[list[str], list[list[str]], list[str], str],
    coordinates: list[tuple[str, str, int | None, str | None]],
    horizon: int,
    sequence: list[float],
    source: str,
    target: str,
    catalog: Any,
    laws: Any,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedTemporalTransportStage: ...
def consume_temporal_transport_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class NativeTemporalInitialState:
    """A prepared target-marginal initial-state result and its observation window."""

    def payload(self) -> str: ...
    def export(self) -> bytes: ...
    def export_refresh(self) -> tuple[bytes | None, str | None]: ...
    def refresh(
        self,
        snapshot_id: str,
        units: list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None,
        window: tuple[
            int,
            list[tuple[str, int]],
            list[str],
            list[str],
            int,
            int,
            str | None,
            str | None,
        ],
        interval_existed: bool,
    ) -> tuple[NativeTemporalInitialState | None, str | None]: ...

def temporal_initial_state_prepare(
    snapshot_id: str,
    units: list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None,
    sequence: tuple[int, int],
    law: tuple[str, str, list[tuple[int, float]]] | None,
    point: int | None,
    fixed_state: int | None,
    premises: tuple[str, list[str], str, str, str, str],
    window: tuple[
        int, list[tuple[str, int]], list[str], list[str], int, int, str | None, str | None
    ]
    | None,
) -> tuple[NativeTemporalInitialState | None, str | None]: ...
def consume_temporal_initial_state_artifact(
    artifact: bytes, *, max_summary_rows: int = 1_000_000, max_units: int = 100_000
) -> tuple[str | None, str | None]: ...
def consume_temporal_refresh_artifact(
    artifact: bytes, *, max_summary_rows: int = 1_000_000, max_units: int = 100_000
) -> tuple[str | None, str | None]: ...

_PeriodPrior = tuple[list[float], list[float], float, float]
_PeriodWindow = tuple[
    str,
    int,
    list[tuple[str, int]],
    list[str],
    list[str],
    list[str],
    int,
    int,
    list[str],
    str,
    str,
]

class NativePeriodRefresh:
    """An autoregressive mechanism kept across observation periods."""

    @staticmethod
    def start(
        series_id: str,
        lags: int,
        invariant: bool,
        prior: _PeriodPrior,
        window: _PeriodWindow,
        cadence_tick: int,
        initial_law: str,
        values: list[float],
    ) -> tuple[NativePeriodRefresh | None, str | None]: ...
    def refresh(
        self,
        window: _PeriodWindow,
        cadence_tick: int,
        initial_law: str,
        values: list[float],
    ) -> tuple[str | None, str | None]: ...
    def payload(self) -> tuple[str | None, str | None]: ...
    def export_ledger(self) -> tuple[bytes | None, str | None]: ...
    def evict_row(self, id: int) -> tuple[str | None, str | None]: ...
    def summary_under(self, prior: _PeriodPrior) -> tuple[str | None, str | None]: ...
    def reprior(self, prior: _PeriodPrior) -> tuple[bool | None, str | None]: ...
    def check_against_fresh(self) -> tuple[bool | None, str | None]: ...
    def export_summary_only(self) -> tuple[bytes | None, str | None]: ...

class NativePeriodRefreshSummary:
    """A least-squares summary read from its summary-only artifact."""

    @staticmethod
    def consume(artifact: bytes) -> tuple[NativePeriodRefreshSummary | None, str | None]: ...
    def payload(self) -> str: ...
    def refit_from_rows(self) -> tuple[bool | None, str | None]: ...
    def evict_row(self, id: int) -> tuple[bool | None, str | None]: ...
    def set_mask(self, id: int, masked: bool) -> tuple[bool | None, str | None]: ...
    def replay_rows(self) -> tuple[bool | None, str | None]: ...

def consume_period_refresh_artifact(artifact: bytes) -> tuple[str | None, str | None]: ...
def staleness_from_artifact(
    artifact: bytes,
    names: list[str],
    columns: list[Any],
    *,
    policy_json: str | None = None,
    seed: int = 1,
    threads: int | None = None,
) -> str: ...
def staleness_policy(policy_json: str | None = None) -> str: ...
def sensitivity_evalue_vector(
    estimates: list[float],
    labels: list[str],
    *,
    lowers: list[float] | None = None,
    uppers: list[float] | None = None,
    standard_errors: list[float] | None = None,
    scale: str = "risk_ratio",
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> str: ...
def sensitivity_evalue_posterior(
    draws: list[list[float]],
    labels: list[str],
    *,
    scale: str = "risk_ratio",
    level: float = 0.95,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> str: ...
def sensitivity_ovb_vector(
    estimates: list[float],
    standard_errors: list[float],
    dofs: list[float],
    labels: list[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> str: ...
def sensitivity_ovb_posterior(
    coefficients: list[list[float]],
    bias_scale: list[list[float]],
    labels: list[str],
    *,
    level: float = 0.95,
    r2yz_extreme: float = 1.0,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> str: ...
def sensitivity_rosenbaum_vector(
    differences: list[list[float]],
    labels: list[str],
    *,
    gammas: list[float] | None = None,
    alpha: float = 0.05,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def sensitivity_evalue(
    estimate: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    standard_error: float | None = None,
    scale: str = "risk_ratio",
    target_source: str | None = None,
) -> str: ...
def sensitivity_omitted_variable_bias(
    estimate: float,
    standard_error: float,
    dof: float,
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    benchmarks: list[tuple[str, float, float]] | None = None,
    kd: float = 1.0,
    ky: float | None = None,
    target_source: str | None = None,
) -> str: ...
def sensitivity_omitted_variable_bias_fit(
    outcome: list[float],
    treatment: list[float],
    covariates: list[list[float]],
    covariate_names: list[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    kd: float = 1.0,
    ky: float | None = None,
    target_source: str | None = None,
) -> str: ...
def sensitivity_rosenbaum(
    differences: list[float],
    *,
    gammas: list[float] | None = None,
    alpha: float = 0.05,
    target_source: str | None = None,
) -> str: ...
def sensitivity_matched_sets(
    sets: list[tuple[list[float], int]],
    *,
    score: str = "aligned_ranks",
    gammas: list[float] | None = None,
    alpha: float = 0.05,
) -> str: ...
def sensitivity_amplify(gamma: float, lam: float) -> float | None: ...
def sensitivity_mediation(
    treatment: list[float],
    mediator: list[float],
    outcome: list[float],
    covariates: list[list[float]],
    *,
    target_source: str | None = None,
) -> str: ...
def sensitivity_mediation_vector(
    treatment: list[float],
    mediator: list[float],
    outcomes: list[list[float]],
    covariates: list[list[float]],
    labels: list[str],
    *,
    component: str = "indirect",
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def sensitivity_mediation_paths(
    treatment: list[float],
    mediator: list[float],
    outcomes: list[list[float]],
    covariates: list[list[float]],
    labels: list[str],
    *,
    path: str = "indirect",
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def sensitivity_mediation_posterior(
    beta_mediator: list[list[float]],
    sigma_total: list[list[float]],
    sigma_mediator: list[list[float]],
    rho_tilde: list[list[float]],
    total_effect: list[list[float]],
    labels: list[str],
    *,
    component: str = "indirect",
    level: float = 0.95,
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def sensitivity_msm_parameters(
    outcome: list[float],
    exposures: list[list[float]],
    weights: list[float],
    labels: list[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def sensitivity_msm_parameters_refit(
    outcome: list[float],
    exposures: list[list[float]],
    treatments: list[list[float]],
    numerators: list[list[list[float]]],
    denominators: list[list[list[float]]],
    labels: list[str],
    *,
    replicates: int = 200,
    seed: int = 1,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
) -> str: ...
def target_refutation_plan(
    family: str,
    *,
    temporal: bool = False,
    bayesian: bool = False,
    suite: str = "full",
    labels: list[str] | None = None,
) -> str: ...
def target_derivative_consistency(
    response: Any,
    points: list[float],
    reported: list[float],
    *,
    step: float = 1e-2,
    tolerance: float = 1e-4,
) -> str: ...
def target_local_linearity(
    response: Any,
    points: list[float],
    *,
    step: float = 1e-1,
    tolerance: float = 0.1,
    slope_floor: float = 1e-12,
) -> str: ...
def target_prior_sensitivity(
    labels: list[str],
    baseline: list[list[float]],
    alternatives: list[tuple[str, list[list[float]]]],
    *,
    tolerance: float = 0.5,
    family: str = "world_valued",
    temporal: bool = False,
) -> str: ...
def target_posterior_predictive(
    labels: list[str],
    observed: list[float],
    draws: list[list[float]],
    replicated: list[list[float]],
    *,
    alpha: float = 0.05,
    family: str = "world_valued",
    temporal: bool = False,
) -> str: ...
def temporal_dependent_interval_closed(
    snapshot_id: str,
    units: list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None,
    sequence: tuple[int, int],
    estimand: str,
    fixed_state: int | None,
    law: tuple[str, str, list[tuple[int, float]]] | None,
    point: int | None,
    replicates: int,
    seed: int,
    level: float,
    method: Literal["studentized"],
    min_units: int,
    max_failed_fraction: float,
) -> str: ...

# Available only in calibration-internal acceptance builds; no released license.
class NativeTemporalIntervalCandidate:
    def payload(self) -> str: ...
    def export(self) -> bytes: ...

def temporal_dependent_interval_candidate(
    snapshot_id: str,
    units: list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None,
    sequence: tuple[int, int],
    estimand: str,
    fixed_state: int | None,
    law: tuple[str, str, list[tuple[int, float]]] | None,
    point: int | None,
    replicates: int,
    seed: int,
    level: float,
    method: Literal["studentized"],
    min_units: int,
    max_failed_fraction: float,
) -> tuple[NativeTemporalIntervalCandidate | None, str | None]: ...
def consume_temporal_interval_candidate(
    artifact: bytes,
    expected_identity: str,
) -> tuple[NativeTemporalIntervalCandidate | None, str | None]: ...
def evaluate_cross_world_edge_contrast(
    names: list[str],
    columns: Sequence[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    control: float,
    active: float,
    intervened_edges: list[tuple[str, str]],
    *,
    mechanism: str = "linear_gaussian",
    interval_requested: bool = False,
    seed: int = 0,
) -> tuple[str, bytes]: ...
def consume_cross_world_edge_contrast_artifact(
    artifact: bytes,
    *,
    seed: int = 0,
) -> tuple[str, bytes]: ...

class NativePreparedCounterfactualId:
    def evaluate(
        self,
        *,
        probabilities: list[float] | None = None,
        counts: list[int] | None = None,
        seed: int = 0,
    ) -> tuple[str, bytes]: ...
    @property
    def derivation(self) -> str: ...
    @property
    def search(self) -> str: ...

def prepare_counterfactual_id_native(
    names: list[str],
    directed: list[tuple[str, str]],
    bidirected: list[tuple[str, str]],
    levels: list[list[float]],
    treatment: str,
    active: float,
    observed: float,
    outcome: str,
    outcome_level: float,
    *,
    operations: int = 20_000,
    depth: int = 48,
    interval_requested: bool = False,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> NativePreparedCounterfactualId: ...
def consume_counterfactual_id_artifact_native(
    artifact: bytes,
    *,
    seed: int = 0,
) -> tuple[str, bytes]: ...

class MixedSourceStage:
    @property
    def identification_status(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    def decision(self) -> dict[str, Any]: ...
    def prepare_exact(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedMixedSourceStage: ...
    def prepare_empirical(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedMixedSourceStage: ...

class PreparedMixedSourceStage:
    def estimate(
        self,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
        seed: int | None = None,
    ) -> str: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def export(self) -> bytes: ...
    def plan(self) -> dict[str, Any]: ...
    @property
    def cited_studies(self) -> list[str]: ...
    @property
    def seed(self) -> int: ...

def identify_mixed_source_transport_stage(
    graph: Admg,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    sources: list[tuple[str, list[str], dict[str, float], list[str]]],
    catalog: Any,
    *,
    max_operations: int = 20_000,
    max_depth: int = 16,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> MixedSourceStage: ...
def consume_mixed_source_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 20_000,
    max_search_depth: int = 16,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class StudyPlanStage:
    @property
    def outcome(self) -> str: ...
    def plan(self) -> dict[str, Any]: ...
    def receive(self, rank: int, catalog: Any, provider_snapshot: str) -> dict[str, Any]: ...
    def export(self) -> bytes: ...

def plan_studies_stage(
    graph: Admg,
    route: str,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    sources: list[tuple[str, list[str], dict[str, float], list[str]]],
    catalog: Any,
    candidates: list[
        tuple[
            str,
            str,
            list[str],
            list[dict[str, float]] | None,
            list[str],
            str,
            int,
            int,
            list[str],
            list[str],
            list[str],
        ]
    ],
    *,
    max_operations: int,
    max_depth: int,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> StudyPlanStage: ...
def replay_study_plan(
    artifact: bytes,
    *,
    max_operations: int,
    max_depth: int,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class RepairContractStage:
    @property
    def family(self) -> str: ...
    @property
    def contract_id(self) -> str: ...
    def obligations(self, quantities_json: str | None = None) -> list[dict[str, Any]]: ...
    def repair(
        self,
        candidates_json: str,
        objective: str,
        *,
        max_operations: int,
        max_depth: int,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> RepairStage: ...

class RepairStage:
    @property
    def outcome(self) -> str: ...
    def report(self) -> dict[str, Any]: ...
    def export(self) -> bytes: ...

def repair_transport_contract(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    catalog: Any,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> RepairContractStage: ...
def repair_z_transport_contract(
    names: list[str],
    failure_snapshot: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> RepairContractStage: ...
def repair_backdoor_contract(
    graph: Dag,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
    assumptions: list[tuple[str, str, str | None]],
) -> RepairContractStage: ...
def repair_frontdoor_contract(
    graph: Dag,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
) -> RepairContractStage: ...
def repair_instrument_contract(
    graph: Dag,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
) -> RepairContractStage: ...
def repair_hedge_contract(
    graph: Admg,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
) -> RepairContractStage: ...
def repair_sensitivity_contract(
    graph: Dag,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
    method: str,
    strength: float,
    robust: bool,
) -> RepairContractStage: ...
def consume_repair_artifact(
    artifact: bytes,
    *,
    max_operations: int,
    max_depth: int,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> RepairStage: ...

class FailureStage:
    @property
    def family(self) -> str: ...
    @property
    def contract_id(self) -> str: ...
    @property
    def routes(self) -> list[str]: ...
    @property
    def variable_names(self) -> list[str]: ...
    def obligations(self, quantities_json: str | None = None) -> list[dict[str, Any]]: ...
    def templates(self) -> dict[str, Any]: ...
    def next_actions(self) -> list[dict[str, Any]]: ...
    def capability_report(self) -> dict[str, Any]: ...
    def plan(
        self,
        candidates_json: str,
        objective: str,
        *,
        max_operations: int,
        max_depth: int,
        memory_bytes: int | None = None,
        value_json: str | None = None,
        cancel: CancellationToken | None = None,
    ) -> EvidencePlanStage: ...

class EvidencePlanStage:
    @property
    def outcome(self) -> str: ...
    def report(self) -> dict[str, Any]: ...
    def export(self) -> bytes: ...

def evidence_failure_dag(
    graph: Dag,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
    sensitivity: tuple[str, float, bool] | None = None,
) -> FailureStage: ...
def evidence_failure_admg(
    graph: Admg,
    treatment: str,
    outcome: str,
    population: str,
    observed: list[str],
    sensitivity: tuple[str, float, bool] | None = None,
) -> FailureStage: ...
def consume_evidence_plan(
    artifact: bytes,
    *,
    max_operations: int,
    max_depth: int,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> EvidencePlanStage: ...
def identify_structure_posterior(
    graphs: list[Admg],
    weights: list[float],
    keys: list[int],
    treatment: str,
    outcome: str,
    interventions: list[str] | None = None,
) -> tuple[list[bool], list[bool] | None]: ...
def sensitivity_confounder_value(
    artifact: SensitivityArtifact, prior_json: str
) -> tuple[str | None, str | None]: ...
def classify_inverse_outcome_stage(
    query_kind: str,
    threshold: float,
    direction: str,
    forward: tuple[
        str,
        str,
        tuple[str | None, str | None, str | None],
        bool,
        bool,
        list[list[float]],
        list[float],
        list[str],
        str,
        tuple[str, str, float, list[float], list[float]] | None,
        list[str],
    ],
    actions: list[tuple[str, list[float], float, list[tuple[str, bool]]]],
    *,
    probability: float | None = None,
    level: float | None = None,
    budget: float | None = None,
    tolerance: float = 1e-9,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]: ...
def matched_case_control_odds_ratio(
    stratum: list[str],
    case: list[float],
    exposed: list[float],
    *,
    sampling: str,
    estimand: str,
    interval: bool = False,
    cancel: CancellationToken | None = None,
) -> tuple[float, float, int, int, int, int, int]: ...
def derived_treatment_check_json(
    names: list[str],
    columns: list[Any],
    declaration: str,
    outcome: str,
    adjustment: list[str],
    *,
    seed: int = 1,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def factorized_joint_cells_json(
    names: list[str],
    columns: list[Any],
    declaration: str,
    outcome: str,
    adjustment: list[str],
    *,
    ordering: list[str] | None = None,
    all_orderings: bool = True,
    nuisance: str = "ridge_logistic",
    penalties: list[float] | None = None,
    inner_folds: int = 5,
    folds: int = 5,
    seed: int = 1,
    clip: float = 0.01,
    min_cell_ess: float = 10.0,
    normalization_tolerance: float = 1e-9,
    ordering_tolerance_sd: float = 0.05,
    contrasts: list[str] | None = None,
    interval: bool = False,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def joint_cell_cost_json(
    components: int,
    *,
    ordering: list[int] | None = None,
    all_orderings: bool = True,
    nuisance: str = "ridge_logistic",
    penalties: list[float] | None = None,
    inner_folds: int = 5,
    folds: int = 5,
    rows: int | None = None,
) -> str: ...
def raw_vs_adjusted(
    outcome: list[float],
    treatment: list[float],
    adjusted_estimate: float,
    *,
    adjusted_se: float | None = None,
    scale: str,
    active: float,
    control: float,
    all_observed: bool,
    cancel: CancellationToken | None = None,
) -> tuple[
    tuple[int, float, float],
    tuple[int, float, float],
    float,
    float | None,
    float,
]: ...
def attribute_gap_to_columns() -> None: ...
def transform_mean_pair(
    mean_active: float,
    mean_control: float,
    scales: list[str],
    *,
    covariance: tuple[float, float, float] | None = None,
    interval: bool = False,
) -> tuple[list[float], list[tuple[float, float]], list[float] | None]: ...
def transform_raw_arms(
    outcome: list[float],
    treatment: list[float],
    scales: list[str],
    *,
    interval: bool = False,
    cancel: CancellationToken | None = None,
) -> tuple[
    tuple[int, float, float],
    tuple[int, float, float],
    tuple[list[float], list[tuple[float, float]], list[float] | None],
]: ...
def candidate_screen_from_units(
    screen_id: str,
    procedure: str,
    *,
    screen_fraction: float,
    seed: int,
    entity_ids: list[int] | None = None,
    first: list[int] | None = None,
    second: list[int] | None = None,
) -> tuple[str, list[int], list[int], int, str, int, int, int, str]: ...
def joint_bayesian_transport_closed(
    graph_class: str,
    dependence: str,
    varying: str,
    sharing: str,
    features: int,
    sources: int,
    has_target: bool,
    draws: int,
) -> None: ...
def binary_nested_markov_closed(
    variables: list[str],
    directed: list[tuple[int, int]],
    bidirected: list[tuple[int, int]],
    regimes: list[tuple[list[int] | None, list[int], list[float]]],
) -> None: ...
def sampled_observation_recovery_closed(
    graph_recoverable: bool,
    partially_observed: int,
    fully_observed: int,
    replicates: int,
    rows: list[tuple[int, int, int, int]],
) -> None: ...
def learned_joint_transport_closed(
    graph_class: str,
    dependence: str,
    varying: str,
    sharing: str,
    features: int,
    basis_degree: int,
    sources: int,
    has_target: bool,
    draws: int,
) -> None: ...
def msm_sensitivity_run(
    strata: list[tuple[float, float, list[float], list[float], list[float], list[float]]],
    lambda_max: float,
    grid_points: int,
    decision_threshold: float | None,
    tolerance: float,
    sampling_composition: str | None = None,
) -> tuple[str | None, str | None]: ...
def msm_sensitivity_artifact_run(
    strata: list[tuple[float, float, list[float], list[float], list[float], list[float]]],
    lambda_max: float,
    grid_points: int,
    decision_threshold: float | None,
    tolerance: float,
    effect_json: str,
    point_quantities_json: str,
    actions_json: str,
    causal_contract_id: str,
    artifact_id: str,
) -> tuple[bytes | None, str | None]: ...
def mechanism_discrepancy_summarize(
    label: str,
    node: str,
    node_unit: str,
    parents: list[tuple[str, str]],
    protocol_id: str,
    outcome: list[float],
    parent_values: list[list[float]],
) -> tuple[str | None, str | None]: ...
def mechanism_discrepancy_run(
    request_json: str,
    source_ids: list[str],
    target_ids: list[str],
    artifact_id: str,
) -> tuple[str | None, bytes | None, str | None]: ...
def mechanism_discrepancy_consume(
    data: bytes, expected_identity_json: str | None = None
) -> tuple[str | None, str | None]: ...

class ObservationRecoveryStage:
    def sampled_measured(
        self,
        population: str,
        observed_regime: str,
        partial: list[tuple[str, str, str]],
        fully: list[str],
        rows: list[tuple[int, int, int, int]],
        snapshot: str,
        replicates: int,
        seed: int,
        interval_method: str = "bootstrap_bca",
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> NativeMeasuredInference: ...
    # Optional: calibration-internal acceptance builds only.
    def sampled_candidate(
        self,
        population: str,
        observed_regime: str,
        partial: list[tuple[str, str, str]],
        fully: list[str],
        rows: list[tuple[int, int, int, int]],
        snapshot: str,
        replicates: int,
        seed: int,
        interval_method: str = "bootstrap_bca",
    ) -> tuple[str, list[int]]: ...
    @property
    def outcome(self) -> str: ...
    def decision(self) -> dict[str, Any]: ...
    def prepare_exact(
        self,
        law: Any,
        requests: list[dict[str, float]] | None = None,
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedObservationRecoveryStage: ...
    def prepare_empirical(
        self,
        law: Any,
        requests: list[dict[str, float]] | None = None,
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedObservationRecoveryStage: ...

class PreparedObservationRecoveryStage:
    def estimate(
        self,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def refresh(
        self,
        law: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def export(self) -> bytes: ...
    def plan(self) -> dict[str, Any]: ...

def identify_observation_recovery_stage(
    graph: Admg,
    population: str,
    observed_regime: str,
    partially_observed: list[tuple[str, str, str]],
    fully_observed: list[str],
    catalog: Any,
    *,
    effect_outcomes: list[str] | None = None,
    effect_treatments: list[str] | None = None,
    effect_graph: Admg | None = None,
    max_operations: int = 50_000,
    max_depth: int = 32,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ObservationRecoveryStage: ...
def consume_observation_recovery_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 50_000,
    max_search_depth: int = 32,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_requests: int = 64,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class AdmgConditionalTransportStage:
    @property
    def identification_status(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    def decision(self) -> dict[str, Any]: ...
    def prepare_exact(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedAdmgConditionalTransportStage: ...
    def prepare_empirical(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedAdmgConditionalTransportStage: ...
    def export_obstruction(self) -> bytes: ...

class PreparedAdmgConditionalTransportStage:
    def estimate(
        self,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
        seed: int | None = None,
    ) -> str: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def export(self) -> bytes: ...
    def plan(self) -> dict[str, Any]: ...
    @property
    def seed(self) -> int: ...

def identify_admg_conditional_transport_stage(
    graph: Admg,
    source: str,
    target: str,
    selections: list[str],
    outcomes: list[str],
    treatments: list[str],
    conditioned_on: list[str],
    catalog: Any,
    *,
    max_operations: int = 4096,
    max_depth: int = 24,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> AdmgConditionalTransportStage: ...
def consume_admg_conditional_transport_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 4096,
    max_search_depth: int = 24,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def consume_admg_conditional_obstruction_artifact(
    artifact: bytes,
    *,
    cancel: CancellationToken | None = None,
) -> str: ...

class MultiSourceZTransportStage:
    @property
    def identification_status(self) -> str: ...
    @property
    def outcome(self) -> str: ...
    def decision(self) -> dict[str, Any]: ...
    def prepare_exact(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedMultiSourceZTransportStage: ...
    def prepare_empirical(
        self,
        laws: Any,
        assignments: dict[str, float] | Sequence[dict[str, float]],
        *,
        max_operations: int = 10_000_000,
        max_depth: int = 256,
        seed: int = 0,
        cancel: CancellationToken | None = None,
    ) -> PreparedMultiSourceZTransportStage: ...

class PreparedMultiSourceZTransportStage:
    def estimate(
        self,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
        seed: int | None = None,
    ) -> str: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...
    def export(self) -> bytes: ...
    def plan(self) -> dict[str, Any]: ...
    @property
    def cited_sources(self) -> list[str]: ...
    @property
    def seed(self) -> int: ...

def identify_multi_source_z_transport_stage(
    graph: Admg,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    sources: list[tuple[str, list[str], dict[str, float], list[str]]],
    catalog: Any,
    *,
    max_operations: int = 4096,
    max_depth: int = 24,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> MultiSourceZTransportStage: ...
def consume_multi_source_z_transport_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 4096,
    max_search_depth: int = 24,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class CounterfactualTransportDecisionNative:
    @property
    def identification_status(self) -> str: ...
    def decision(self) -> dict[str, Any]: ...
    def prepared(self) -> PreparedCounterfactualTransportNative: ...
    def export(self) -> bytes: ...
    def identity(self) -> tuple[str | None, str | None]: ...

class PreparedCounterfactualTransportNative:
    @property
    def bound(self) -> bool: ...
    def derivation(self) -> dict[str, Any]: ...
    def evaluate(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> CounterfactualTransportResultNative: ...
    def refresh(
        self,
        laws: Any,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> PreparedCounterfactualTransportNative: ...
    def estimate(
        self,
        *,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> CounterfactualTransportResultNative: ...

class CounterfactualTransportResultNative:
    def summary(self) -> dict[str, Any]: ...
    def export(self) -> bytes: ...
    def identity(self) -> tuple[str | None, str | None]: ...

class SampledCounterfactualTransportResultNative:
    def summary(self) -> dict[str, Any]: ...
    def export(self) -> bytes: ...
    def identity(self) -> tuple[str | None, str | None]: ...
    def measure_interval(
        self, *, memory_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> NativeMeasuredInference: ...

def identify_counterfactual_transport_native(
    graph: Admg,
    levels: list[tuple[str, list[float]]],
    worlds: list[tuple[dict[str, float], list[tuple[str, str, int]]]],
    event: list[tuple[int, str, float]],
    given: list[tuple[int, str, float]],
    target: str,
    sources: list[tuple[str, list[str], list[str]]],
    catalog: Any,
    *,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_eval_operations: int | None = None,
    max_eval_depth: int | None = None,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> CounterfactualTransportDecisionNative: ...
def estimate_sampled_counterfactual_transport_native(
    prepared: PreparedCounterfactualTransportNative,
    tables: Any,
    overlap: tuple[str, list[tuple[Any, Any, str]]],
    unseen_rule: tuple[str, float | None],
    *,
    min_table_total: int | None = None,
    max_unseen_fraction: float | None = None,
    max_envelope_evaluations: int | None = None,
    max_operations: int | None = None,
    max_depth: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> SampledCounterfactualTransportResultNative: ...
def consume_counterfactual_transport_artifact_native(
    artifact: bytes,
    *,
    expected_premises_digest: str | None = None,
    expected_data_digest: str | None = None,
    max_search_operations: int | None = None,
    max_search_depth: int | None = None,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[
    str,
    CounterfactualTransportResultNative | None,
    CounterfactualTransportDecisionNative | None,
]: ...
def consume_sampled_counterfactual_transport_artifact_native(
    artifact: bytes,
    *,
    expected_premises_digest: str | None = None,
    expected_data_digest: str | None = None,
    max_search_operations: int | None = None,
    max_search_depth: int | None = None,
    max_operations: int | None = None,
    max_depth: int | None = None,
    max_tables: int | None = None,
    max_table_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> SampledCounterfactualTransportResultNative: ...
def identify_z_transport_stage(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    controllable: list[str],
    experiment_assignment: dict[str, float],
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ZTransportStage: ...
def decide_two_source_z_transport_stage(
    graph: Admg,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    sources: list[tuple[str, list[str], dict[str, float], list[str]]],
    catalogs: list[Any],
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> Any: ...
def consume_z_transport_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def consume_z_transport_sensitivity_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ZTransportSensitivityResult: ...
def z_transport_joint_sensitivity(
    stage: PreparedZTransportStage,
    factors: list[tuple[str, float]],
    *,
    decision_threshold: float | None = None,
    total_budget: float | None = None,
    tolerance: float = 1e-9,
    frontier_points: int = 17,
    max_operations: int = 100_000,
    max_depth: int = 64,
    max_memory_bytes: int = 67_108_864,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]: ...
def export_z_transport_joint_sensitivity(
    stage: PreparedZTransportStage,
    factors: list[tuple[str, float]],
    *,
    decision_threshold: float | None = None,
    total_budget: float | None = None,
    tolerance: float = 1e-9,
    frontier_points: int = 17,
    max_operations: int = 100_000,
    max_depth: int = 64,
    max_memory_bytes: int = 67_108_864,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> bytes: ...
def consume_z_transport_joint_sensitivity_artifact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    max_search_operations: int = 100_000,
    max_search_depth: int = 64,
    max_memory_bytes: int = 67_108_864,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> dict[str, Any]: ...
def z_transport_joint_sensitivity_interval(
    stage: PreparedZTransportStage,
    factors: list[tuple[str, float]],
    *,
    replicates: int = 199,
    level: float = 0.95,
    decision_threshold: float | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[float, float, str]: ...
def consume_z_transport_failure_snapshot(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> Any: ...
def replay_z_transport_proposal(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> None: ...
def consume_exact_transport(
    bytes: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedExactStage: ...

class PreparedStatisticalStage:
    @property
    def outcomes(self) -> list[str]: ...
    @property
    def bayesian_posterior(self) -> str | None: ...
    def estimate_grid(
        self, assignments: list[dict[str, float]], cancel: CancellationToken | None = None
    ) -> list[PreparedStatisticalStage]: ...
    def estimate(
        self, execution: str | None = None, cancel: CancellationToken | None = None
    ) -> tuple[list[list[float]], list[float], str, list[str], str]: ...
    def refresh(
        self, payload: Any, cancel: CancellationToken | None = None
    ) -> tuple[list[list[float]], list[float], str, list[str], str]: ...
    def replace_snapshot(self, payload: Any, cancel: CancellationToken | None = None) -> None: ...
    def contrast(self, reference: PreparedStatisticalStage, outcome: str) -> str: ...
    def last_result(self) -> tuple[list[list[float]], list[float], str, list[str], str]: ...
    def plan_summary(self) -> dict[str, str]: ...
    def freeze(self) -> PreparedStatisticalStage: ...
    def inspection_json(self) -> str: ...
    def export(self) -> bytes: ...
    def preview_transform(self, intent: str) -> dict[str, str]: ...

def prepare_statistical_transport(
    stage: ClassicalTransportStage,
    catalog: Any,
    payload: Any,
    assignments: dict[str, float],
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
    bootstrap: int = 199,
    posterior_draws: int = 199,
    coverage_level: float = 0.95,
    estimator: object = None,
    seed: int = 1,
) -> PreparedStatisticalStage: ...
def consume_statistical_transport(
    bytes: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
    seed: int = 1,
) -> PreparedStatisticalStage: ...

class PreparedTransportGridStage:
    def freeze(self) -> PreparedTransportGridStage: ...
    def estimate(self, cancel: CancellationToken | None = None) -> str: ...
    def refresh(self, data: Any, cancel: CancellationToken | None = None) -> str: ...
    def replace_snapshot(self, data: Any, cancel: CancellationToken | None = None) -> None: ...
    def last_result(self) -> str: ...
    def export(self) -> bytes: ...
    def contrast(self, left: int, right: int, outcome: str) -> str: ...
    def scalar_projection(self, point: int, outcome: str) -> str: ...
    def inspection_json(self) -> str: ...
    def plan_summary(self) -> dict[str, str]: ...
    def preview_transform(self, intent: str) -> dict[str, str]: ...

def prepare_transport_grid(
    stage: ClassicalTransportStage,
    catalog: Any,
    data: Any,
    at: list[dict[str, float]],
    *,
    statistical: bool = False,
    estimator: object = None,
    bootstrap: int = 199,
    coverage_level: float = 0.95,
    seed: int = 1,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedTransportGridStage: ...
def consume_transport_grid(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedTransportGridStage: ...
def transport_identification_statuses() -> tuple[list[str], list[tuple[str, str]]]: ...
def identify_meta_transport_stage(
    graph: Admg,
    catalog: Any,
    target: str,
    outcomes: list[str],
    treatments: list[str],
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ClassicalTransportStage: ...
def consume_transport_certificate(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> ClassicalTransportStage: ...

class FittedEffectModel:
    features: list[str]
    parent_claim: str
    @staticmethod
    def load(bytes: bytes) -> FittedEffectModel: ...
    def predict(self, columns: list[NDArray[np.float64]], nrows: int) -> NDArray[np.float64]: ...
    def export(self) -> bytes: ...

class PreparedLearnedTrial:
    def preview_transform(self, intent: str) -> dict[str, str]: ...
    def estimate_summary(self, cancel: CancellationToken | None = None) -> str: ...
    def last_result_summary(self) -> str: ...
    def refresh_summary(self, data: Any, cancel: CancellationToken | None = None) -> str: ...
    def estimate(self, cancel: CancellationToken | None = None) -> str: ...
    def refresh(self, data: Any, cancel: CancellationToken | None = None) -> str: ...
    def replace_snapshot(self, data: Any, cancel: CancellationToken | None = None) -> None: ...
    def last_result(self) -> str: ...
    def freeze(self) -> PreparedLearnedTrial: ...
    def inspection_json(self) -> str: ...
    def plan_summary(self) -> dict[str, str]: ...
    def export(self) -> bytes: ...

def prepare_learned_trial(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    treatment: str,
    outcome: str,
    data: Any,
    options: str,
    *,
    seed: int = 1,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedLearnedTrial: ...
def consume_learned_trial(bytes: bytes) -> PreparedLearnedTrial: ...

class PreparedLearnedContinuousNative:
    def estimate(self, cancel: CancellationToken | None = None) -> str: ...
    def interval(self, cancel: CancellationToken | None = None) -> None: ...
    def refresh(self, data: Any, cancel: CancellationToken | None = None) -> None: ...
    def export(self) -> bytes: ...
    def estimator_menu(self) -> str: ...
    def last_result(self) -> str: ...

def prepare_learned_continuous(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    treatment: str,
    outcome: str,
    data: Any,
    options: str,
    target_name: str,
    *,
    seed: int = 1,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedLearnedContinuousNative: ...
def learned_continuous_estimator_menu(
    graph: Admg,
    selections: list[str],
    source: str,
    target: str,
    treatment: str,
    outcome: str,
    learners: str | None = None,
) -> str: ...
def consume_learned_continuous(
    artifact: bytes,
    *,
    max_rows: int | None = None,
    max_features: int | None = None,
) -> str: ...

class PreparedSmoothedDoseNative:
    def estimate(self, cancel: CancellationToken | None = None) -> str: ...
    def interval(self, cancel: CancellationToken | None = None) -> None: ...
    def refresh(self, data: Any, cancel: CancellationToken | None = None) -> None: ...
    def export(self) -> bytes: ...
    def estimator_menu(self) -> str: ...
    def last_result(self) -> str: ...
    def posterior(self, model: str) -> tuple[str, bytes]: ...

def prepare_smoothed_dose(
    graph: Admg,
    selections: list[str],
    query: str,
    data: Any,
    options: str,
    target_name: str,
    *,
    seed: int = 1,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> PreparedSmoothedDoseNative: ...
def smoothed_dose_estimator_menu(
    graph: Admg,
    selections: list[str],
    query: str,
    options: str | None = None,
) -> str: ...
def consume_smoothed_dose_posterior(
    artifact: bytes, *, max_rows: int = 100_000, max_points: int = 1_024
) -> str: ...
def consume_smoothed_dose(
    artifact: bytes,
    *,
    max_rows: int | None = None,
    max_features: int | None = None,
    max_memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class ProviderVerificationReceipt:
    @staticmethod
    def verify(requests_json: str) -> tuple[ProviderVerificationReceipt | None, str | None]: ...
    @staticmethod
    def load(data: bytes, expected_digest: str) -> ProviderVerificationReceipt: ...
    @property
    def wire_json(self) -> str: ...
    @property
    def digest(self) -> str: ...
    def export(self, artifact_id: str) -> bytes: ...

def canonical_json_digest_hex(domain: str, json: str) -> str: ...
def blake3_digest_hex(data: bytes) -> str: ...
def seal_provider_result(header_json: str, external_artifact: bytes | None) -> bytes: ...
def open_provider_result(
    bytes: bytes,
    *,
    verified_spec_digest: str | None = None,
    verified_request_digest: str | None = None,
    verified_evidence_digest: str | None = None,
) -> tuple[str, bytes | None]: ...
def preflight_json(
    names: list[str],
    columns: Sequence[Any],
    treatment: str,
    outcome: str,
    adjustment: list[str],
    control: float = 0.0,
    active: float = 1.0,
    *,
    seed: int = 1,
    threads: int | None = None,
) -> str: ...
def fit_diagnostics_json(
    names: list[str],
    columns: Sequence[Any],
    treatment: str,
    outcome: str,
    adjustment: list[str],
    control: float = 0.0,
    active: float = 1.0,
    *,
    seed: int = 1,
    threads: int | None = None,
) -> str: ...
def rank_drop_json(
    names: list[str],
    columns: Sequence[Any],
    treatment: str,
    outcome: str,
    adjustment: list[str],
    priority: list[str] | None = None,
    control: float = 0.0,
    active: float = 1.0,
    *,
    seed: int = 1,
    threads: int | None = None,
) -> str: ...
def estimate_with_rank_drop_json(
    names: list[str],
    columns: Sequence[Any],
    treatment: str,
    outcome: str,
    adjustment: list[str],
    estimator: str,
    priority: list[str] | None = None,
    control: float = 0.0,
    active: float = 1.0,
    *,
    seed: int = 1,
    threads: int | None = None,
) -> str: ...
def decision_contract_normalize(contract_json: str) -> tuple[str | None, str | None]: ...
def evaluate_decision(
    contract_json: str, source: JointDistributionArtifact
) -> tuple[str | None, str | None]: ...
def export_decision_contract(contract_json: str, artifact_id: str) -> bytes: ...
def load_decision_contract(data: bytes, expected_identity: str) -> str: ...
def export_decision_result(
    contract_json: str, source: JointDistributionArtifact, artifact_id: str
) -> bytes: ...
def replay_decision_result(
    data: bytes, contract_json: str, source: JointDistributionArtifact
) -> tuple[str | None, str | None]: ...
def decision_source_digest(source: JointDistributionArtifact) -> str: ...
def evaluate_decision_means(
    contract_json: str,
    coordinates_json: str,
    means: list[float],
    provider_id: str,
    snapshot_id: str,
    causal_contract_id: str,
) -> tuple[str | None, str | None]: ...
def composition_lineage(links_json: str) -> str: ...

class TrustEvidence:
    @staticmethod
    def none() -> TrustEvidence: ...
    @staticmethod
    def attested(attestor: str) -> TrustEvidence: ...
    @staticmethod
    def from_verification(receipt: ProviderVerificationReceipt) -> TrustEvidence: ...
    @property
    def kind(self) -> str: ...

class DecisionInput:
    def _retain_external_source(self, claim: ExternalClaimArtifact) -> None: ...
    has_source_evidence: bool
    source_evidence: SourceEvidenceHandle | None
    @property
    def id(self) -> str: ...
    def summary_json(self) -> str: ...

def composition_input_from_native_claim(
    input_id: str,
    claim: NativeResponseClaim,
    contract_json: str,
) -> tuple[DecisionInput | None, str | None]: ...
def composition_input_from_distribution(
    input_id: str,
    artifact: JointDistributionArtifact,
    evidence: TrustEvidence,
    requirement: str,
) -> tuple[DecisionInput | None, str | None]: ...
def composition_input_from_external_claim(
    input_id: str,
    claim: ExternalClaimArtifact,
    requirement: str,
) -> tuple[DecisionInput | None, str | None]: ...
def composition_input_from_means(
    input_id: str,
    coordinates_json: str,
    means: list[float],
    statuses: list[str],
    provider_id: str,
    snapshot_id: str,
    causal_contract_id: str,
    evidence: TrustEvidence,
    requirement: str,
) -> tuple[DecisionInput | None, str | None]: ...
def composition_input_from_scalar(
    input_id: str,
    coordinate_json: str,
    value: float,
    status: str,
    provider_id: str,
    snapshot_id: str,
    causal_contract_id: str,
    evidence: TrustEvidence,
    requirement: str,
) -> tuple[DecisionInput | None, str | None]: ...
def composition_evaluate(
    contract_json: str,
    inputs: list[DecisionInput],
    unsupported: str,
    weakest_support: str,
) -> tuple[str | None, str | None]: ...
def composition_functional(
    contract_json: str,
    action_id: str,
    functional_json: str,
    input: DecisionInput,
    weakest_support: str,
) -> tuple[str | None, str | None]: ...
def composition_check(
    inputs: list[DecisionInput], relations_json: str, operation: str | None = None
) -> tuple[str | None, str | None]: ...
def composition_check_paired_draws(
    inputs: list[DecisionInput], relations_json: str
) -> tuple[str | None, str | None]: ...
def composition_check_atoms(atoms_json: str, combination: str) -> tuple[str | None, str | None]: ...
def scenario_identified_utilities(
    declaration_json: str,
    kind: str,
    result: PreparedTransportScenariosStage | CpdagScenarioRun,
    binding_json: str,
    support_json: str,
    route: str,
) -> tuple[str | None, str | None]: ...
def interval_identified_action_utilities(
    declaration_json: str, intervals_json: str, reading: str, support_json: str
) -> tuple[str | None, str | None]: ...
def decide_scenario_claims(
    declaration_json: str,
    kind: str,
    result: PreparedTransportScenariosStage | CpdagScenarioRun,
    binding_json: str,
    policy: str,
    weights_json: str | None = None,
    expected_causal: str | None = None,
) -> tuple[str | None, str | None]: ...

class ProposalBundle:
    @staticmethod
    def from_json(json: str) -> ProposalBundle: ...
    def to_json(self) -> str: ...
    @property
    def identity(self) -> str: ...
    def verify(self, repair: bytes, ranking: bytes) -> str | None: ...
    def on_arrival(
        self, contract: RepairContractStage, candidate_json: str, arrival_json: str
    ) -> tuple[str | None, str | None]: ...

def proposal_bundle_build(
    repair: bytes, ranking: bytes
) -> tuple[ProposalBundle | None, str | None]: ...
def proposal_ranked_candidates(ranking: bytes) -> list[str]: ...
def admissible_contract_normalize(declaration_json: str) -> tuple[str | None, str | None]: ...
def export_admissible_contract(declaration_json: str, artifact_id: str) -> bytes: ...
def load_admissible_contract(data: bytes, expected_identity: str) -> str: ...
def evaluate_robust_decision(
    declaration_json: str,
    kind: str,
    claims_json: str,
    sources: list[JointDistributionArtifact | None],
    receipts_json: str,
) -> tuple[str | None, str | None]: ...
def export_robust_decision(
    declaration_json: str,
    kind: str,
    claims_json: str,
    sources: list[JointDistributionArtifact | None],
    receipts_json: str,
    artifact_id: str,
) -> tuple[bytes | None, str | None]: ...
def replay_robust_decision(
    data: bytes,
    declaration_json: str,
    kind: str,
    claims_json: str,
    sources: list[JointDistributionArtifact | None],
) -> tuple[str | None, str | None]: ...
def evaluate_identified_set_decision(
    declaration_json: str, utilities_json: str
) -> tuple[str | None, str | None]: ...
def identified_utility_interval(
    declaration_json: str, action_id: str, intervals: list[tuple[float, float]]
) -> tuple[tuple[float, float] | None, str | None]: ...
def evaluate_effect_constancy(
    request_json: str, artifact_id: str
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_effect_constancy_artifact(
    artifact: bytes, expected_identity_json: str | None = None
) -> tuple[str | None, str | None]: ...
def plan_recalculation(previous_json: str, requested_json: str, capabilities_json: str) -> str: ...
def recalc_stage_identity(label: str, parts: list[bytes]) -> str: ...
def consume_recalc_receipt(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[str | None, str | None]: ...

class RecalcSessionHandle:
    def __init__(self, retarget: str = "licensed") -> None: ...
    @staticmethod
    def resume(
        previous_json: str, resume_json: str, retarget: str = "licensed"
    ) -> RecalcSessionHandle: ...
    def set_retarget_support(self, retarget: str) -> None: ...
    def set_request_support(self, request: str, licensed_route: str | None = None) -> None: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def score_contrast(self) -> list[float] | None: ...
    def plan(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatment: str,
        outcome: str,
        benefit_per_unit: float,
        cost: float,
        *,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[str] | None = None,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatment: str,
        outcome: str,
        benefit_per_unit: float,
        cost: float,
        *,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[str] | None = None,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...

class CellSessionHandle:
    def __init__(self, retarget: str = "licensed") -> None: ...
    def set_retarget_support(self, retarget: str) -> None: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def score_columns(self) -> list[tuple[int, list[float]]] | None: ...
    def plan(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatments: list[str],
        outcome: str,
        adjustment: list[str],
        quantity: str,
        benefit_per_unit: float,
        cost: float,
        *,
        arm: int | None = None,
        folds: int | None = None,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[str] | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatments: list[str],
        outcome: str,
        adjustment: list[str],
        quantity: str,
        benefit_per_unit: float,
        cost: float,
        *,
        arm: int | None = None,
        folds: int | None = None,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[str] | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_frozen_scores(self) -> tuple[tuple[str, bytes] | None, str | None]: ...

class CrossfitSessionHandle:
    def __init__(self) -> None: ...
    def is_live(self) -> bool: ...
    def execute(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        edges: list[tuple[str, str]],
        treatment: str,
        outcome: str,
        benefit_per_unit: float,
        cost: float,
        *,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[str] | None = None,
        seed: int = 1,
        threads: int | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_frozen_scores(self) -> tuple[tuple[str, bytes] | None, str | None]: ...

class ScoreResumeHandle:
    @staticmethod
    def from_bytes(
        artifact: bytes, expected_identity: str | None = None
    ) -> tuple[ScoreResumeHandle | None, str | None]: ...
    def artifact_identity(self) -> str: ...
    def n_rows(self) -> int: ...
    def row_ids(self) -> list[int]: ...
    def score_columns(self) -> list[tuple[int, list[float]]]: ...
    def has_run(self) -> bool: ...
    def identities_json(self) -> str: ...
    def plan(
        self,
        n_variables: int,
        edges: list[tuple[int, int]],
        quantity: str,
        benefit_per_unit: float,
        cost: float,
        *,
        arm: int | None = None,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[int] | None = None,
        target_row_ids: list[int] | None = None,
        changed_inputs: list[tuple[str, str]] | None = None,
    ) -> tuple[str | None, str | None]: ...
    def execute_retarget(
        self,
        n_variables: int,
        edges: list[tuple[int, int]],
        quantity: str,
        benefit_per_unit: float,
        cost: float,
        *,
        arm: int | None = None,
        target_weights: NDArray[np.float64] | None = None,
        target_depends_on: list[int] | None = None,
        target_row_ids: list[int] | None = None,
        changed_inputs: list[tuple[str, str]] | None = None,
    ) -> tuple[str | None, str | None]: ...

def evaluate_design_ranking(
    request_json: str, artifact_id: str, checked_prior: CheckedPriorSignal | None = None
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_design_ranking(
    artifact: bytes, expectation_json: str | None = None
) -> tuple[str | None, str | None]: ...
def rank_structural_designs(candidates_json: str) -> tuple[str | None, str | None]: ...
def plan_experiment_py(
    request_json: str, threads: int = 1, cancel: CancellationToken | None = None
) -> str: ...
def planning_program_identity_py(request_json: str) -> str: ...
def consume_array_fit_artifact(
    artifact: bytes,
    *,
    max_numbers: int = 4_000_000,
    max_draws: int = 200_000,
    max_columns: int = 1_024,
) -> tuple[str, str]: ...
def evaluate_dose_grid_functional(
    request_json: str,
    dose: list[float],
    outcome: list[float],
    *,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, bytes | None, str | None]: ...
def evaluate_dose_grid_posterior(
    request_json: str,
    dose: list[float],
    outcome: list[float],
    *,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, bytes | None, str | None, ExecutionFacts]: ...
def consume_dose_grid_posterior_artifact(
    artifact: bytes,
    *,
    max_rows: int = 100_000,
    max_grid: int = 1_024,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None, ExecutionFacts]: ...
def consume_dose_grid_artifact(
    artifact: bytes,
    *,
    max_rows: int = 100_000,
    max_grid: int = 1_024,
    threads: int | None = None,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def dose_support_labels(
    dose: list[float], points: list[float], bandwidth: float, minimum_local_ess: float
) -> tuple[str | None, str | None]: ...
def recover_recovery_chain(
    nodes: list[str],
    directed: list[tuple[str, str]],
    bidirected: list[tuple[str, str]],
    first: tuple[str, str, str],
    second: tuple[str, str, str],
    law_json: str | None = None,
    *,
    seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_recovery_chain_artifact(
    artifact: bytes,
    *,
    seed: int = 0,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None]: ...
def evaluate_vector_treatment(
    outcome: NDArray[np.float64] | list[float],
    adjustment_columns: list[NDArray[np.float64] | list[float]],
    treatment_columns: list[NDArray[np.float64] | list[float]],
    declaration_json: str,
    artifact_id: str,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_vector_treatment_artifact(
    artifact: bytes,
    expected_identity_json: str | None = None,
    *,
    memory_limit_bytes: int | None = None,
) -> tuple[str | None, str | None]: ...
def evaluate_categorical_treatment(
    outcome: NDArray[np.float64] | list[float],
    adjustment_columns: list[NDArray[np.float64] | list[float]],
    levels: list[str],
    declaration_json: str,
    artifact_id: str,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_categorical_treatment_artifact(
    artifact: bytes,
    expected_identity_json: str | None = None,
    *,
    memory_limit_bytes: int | None = None,
) -> tuple[str | None, str | None]: ...
def evaluate_multi_level_treatment(
    outcome: NDArray[np.float64] | list[float],
    treatment: NDArray[np.float64] | list[float],
    adjustment_names: list[str],
    adjustment_columns: list[NDArray[np.float64] | list[float]],
    declaration_json: str,
) -> str: ...
def evaluate_observational_survival(
    duration: NDArray[np.float64] | list[float],
    event: NDArray[np.float64] | list[float],
    treatment: NDArray[np.float64] | list[float],
    adjustment_names: list[str],
    adjustment_columns: list[NDArray[np.float64] | list[float]],
    declaration_json: str,
) -> str: ...
def evaluate_observational_policy(
    outcome: NDArray[np.float64] | list[float],
    treatment: NDArray[np.float64] | list[float],
    adjustment_names: list[str],
    adjustment_columns: list[NDArray[np.float64] | list[float]],
    declaration_json: str,
) -> str: ...
def build_compact_export(
    spec_json: str, artifact_id: str
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_compact_export(
    artifact: bytes, expected_identity: str, max_bytes: int | None = None
) -> tuple[str | None, str | None]: ...
def evaluate_compact_export(
    artifact: bytes,
    expected_identity: str,
    queries_json: str,
    max_bytes: int | None = None,
) -> tuple[str | None, str | None]: ...
def build_cohort_decision_source(
    model_kind: str,
    model_bytes: bytes,
    expected_model: str | None,
    cohort_json: str,
    outcome_units: str,
    options_json: str,
    artifact_id: str,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_cohort_decision_source(
    artifact: bytes, expectation_json: str
) -> tuple[str | None, str | None]: ...
def validate_cohort_binding(cohort_json: str) -> tuple[str | None, str | None]: ...
def normalize_cohort_weights(weights: list[float]) -> tuple[str | None, str | None]: ...
def cohort_decision_law_refusal(kind: str) -> str: ...
def count_fitted_observation_build(
    nodes_json: str,
    rows: int,
    target: str,
    target_level: str,
    evidence_json: str,
    seed: int | None = None,
    draws: int | None = None,
) -> tuple[bytes | None, str | None]: ...
def count_fitted_observation_point(bytes: bytes) -> tuple[float | None, str | None]: ...
def count_fitted_observation_interval(
    bytes: bytes, level: float, *, cancel: Any = None
) -> tuple[NativeMeasuredInference | None, str | None]: ...
def fit_native_cohort_effect(
    fit_json: str,
) -> tuple[str | None, bytes | None, str | None]: ...
def cohort_decision_interval(
    artifact: bytes,
    expectation_json: str,
    fit_json: str | None,
    level: float,
) -> tuple[NativeMeasuredInference | None, str | None]: ...
def cohort_decision_forward_claim(
    artifact: bytes, expectation_json: str
) -> tuple[str | None, str | None]: ...
def transport_cohort_effect(
    model_bytes: bytes,
    expected_claim: str | None,
    source_population: str,
    cohort_json: str,
    outcome_units: str,
    shift_json: str | None,
    artifact_id: str,
) -> tuple[str | None, str | None, bytes | None, str | None]: ...
def transport_cohort_effect_verify(
    model_bytes: bytes,
    expected_claim: str | None,
    source_population: str,
    cohort_json: str,
    shift_json: str | None,
    expected_identity: str,
) -> tuple[str | None, str | None]: ...
def transported_cohort_refusal(kind: str) -> str: ...
def evaluate_nonlinear_mediation(
    treatment: NDArray[np.float64],
    mediator: NDArray[np.float64],
    outcome: NDArray[np.float64],
    covariates: list[NDArray[np.float64]],
    covariate_names: list[str],
    premises_json: str,
    config_json: str,
    artifact_id: str,
    *,
    digest_bound: bool = False,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_nonlinear_mediation_artifact(
    artifact: bytes,
    expected_identity_json: str | None = None,
    *,
    memory_limit_bytes: int | None = None,
    data: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        list[str],
        list[NDArray[np.float64]],
    ]
    | None = None,
) -> tuple[str | None, str | None]: ...
def evaluate_latent_class_effects(
    outcome: NDArray[np.float64],
    treatment: NDArray[np.float64],
    covariates: list[NDArray[np.float64]],
    covariate_names: list[str],
    config_json: str,
    artifact_id: str,
    *,
    digest_bound: bool = False,
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_latent_class_artifact(
    artifact: bytes,
    expected_identity_json: str | None = None,
    *,
    memory_limit_bytes: int | None = None,
    data: tuple[NDArray[np.float64], NDArray[np.float64], list[str], list[NDArray[np.float64]]]
    | None = None,
) -> tuple[str | None, str | None]: ...
def evaluate_temporal_counterfactual(
    request_json: str, artifact_id: str, *, seed: int = 0
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_temporal_counterfactual_artifact(
    artifact: bytes, expected_identity_json: str | None = None, *, seed: int = 0
) -> tuple[str | None, str | None]: ...
def transported_path_specific_refusal(
    transport_license: bool,
    fixed_population_license: bool,
    cross_population_assumptions: bool,
    required_factors: list[tuple[str, str]],
    supplied_factors: list[tuple[str, str, str]],
) -> str: ...
def evaluate_transported_counterfactual(
    request_json: str, artifact_id: str
) -> tuple[str | None, bytes | None, str | None]: ...
def consume_transported_counterfactual_artifact(
    artifact: bytes, expected_identity_json: str | None = None
) -> tuple[str | None, str | None]: ...
def scenario_invariance_report(stage: Any) -> tuple[str | None, str | None]: ...

class SensitivityArtifact:
    summary_json: str
    def export(self, artifact_id: str) -> bytes: ...
    @staticmethod
    def consume(
        data: bytes, expected_identity_json: str | None = None
    ) -> tuple[SensitivityArtifact | None, str | None]: ...
    def outcome_json(
        self, range: tuple[float, float] | None = None
    ) -> tuple[str | None, str | None]: ...

def sensitivity_artifact_from_surface(
    surface_json: str,
) -> tuple[SensitivityArtifact | None, str | None]: ...
def sensitivity_artifact_from_z_joint(
    stage: PreparedZTransportStage,
    factors: list[tuple[str, float]],
    effect_json: str,
    grid_points: int,
    actions_json: str,
    causal_contract_id: str,
    *,
    decision_threshold: float | None = None,
    total_budget: float | None = None,
    tolerance: float = 1e-9,
    frontier_points: int = 17,
    max_operations: int = 100_000,
    max_depth: int = 64,
    max_memory_bytes: int = 67_108_864,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[SensitivityArtifact | None, str | None]: ...
def sensitivity_contract(
    artifact: SensitivityArtifact, policy: str
) -> tuple[str | None, str | None]: ...
def sensitivity_decide(
    contract_json: str, artifact: SensitivityArtifact, spec_json: str
) -> tuple[str | None, str | None]: ...

class InverseQueryArtifact:
    source_evidence: list[SourceEvidenceHandle]
    result_json: str
    identity_json: str
    query_json: str
    contract_json: str
    forward_origin_json: str | None
    @staticmethod
    def build(
        contract_json: str,
        query_json: str,
        point: JointDistributionArtifact | str | None = None,
        interval_region: tuple[
            JointDistributionArtifact | str, JointDistributionArtifact | str, bool
        ]
        | None = None,
        identified_set: tuple[list[tuple[str, float | None, str, object]], bool] | None = None,
        scenarios: list[tuple[str, float | None, str, object]] | None = None,
    ) -> tuple[InverseQueryArtifact | None, str | None]: ...
    @staticmethod
    def consume(
        data: bytes, expected_identity_json: str | None = None
    ) -> tuple[InverseQueryArtifact | None, str | None]: ...
    def export(self, artifact_id: str) -> bytes: ...

def inverse_query_baseline(
    points_json: str, target: float, comparison: str, tolerance: float
) -> tuple[str | None, str | None]: ...

class CompositionBundle:
    identity: str
    nodes_json: str
    edges_json: str
    def export(self, artifact_id: str) -> tuple[bytes | None, str | None]: ...

class CompositionBundleBuilder:
    def __init__(self) -> None: ...
    def add_artifact(
        self, kind: str, data: bytes, node_id: str | None = None
    ) -> tuple[str | None, str | None]: ...
    def add_reference(
        self,
        node_id: str,
        kind: str,
        identity: str,
        requires_json: str,
        facts_json: str | None = None,
        inspected_json: str | None = None,
    ) -> str | None: ...
    def connect(self, upstream: str, dependent: str) -> str | None: ...
    def relate(self, left: str, right: str, relationship: str) -> tuple[str | None, str | None]: ...
    def build(self) -> tuple[CompositionBundle | None, str | None]: ...

def consume_composition_bundle(
    data: bytes, expected_identity: str, supplied_json: str | None = None
) -> tuple[str | None, str | None]: ...
def composition_describe_artifact(kind: str, data: bytes) -> tuple[str | None, str | None]: ...
def composition_mean_decision(
    contract_json: str, claim_data: bytes, artifact_id: str
) -> tuple[bytes | None, str | None]: ...
def composition_detect_kind(data: bytes) -> str | None: ...

class AdjustedSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> AdjustedSessionHandle: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def prediction_columns(self) -> list[str] | None: ...
    def predict(
        self,
        rows: list[list[float]],
        columns: list[str],
        *,
        memory_limit_bytes: int | None = None,
    ) -> tuple[str | None, str | None]: ...
    def plan(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...

class DrSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> DrSessionHandle: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def prediction_columns(self) -> list[str] | None: ...
    def row_ids(self) -> list[int] | None: ...
    def score_contrast(self) -> list[float] | None: ...
    def plan(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: Sequence[NDArray[np.float64]],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def predict(
        self,
        rows: list[list[float]],
        columns: list[str],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
    ) -> tuple[str | None, str | None]: ...
    def export_scores(self) -> tuple[tuple[str, bytes] | None, str | None]: ...
    def export_predictor(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[bytes | None, str | None]: ...

class StaticResponseSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> StaticResponseSessionHandle: ...
    def is_live(self) -> bool: ...
    def response(self) -> ResponseAnalysisResult | None: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def plan(
        self,
        names: list[str],
        columns: list[Any],
        graph: Admg,
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: list[Any],
        graph: Admg,
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_result(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[bytes | None, str | None]: ...

class MultiSourceSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> MultiSourceSessionHandle: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def plan(
        self,
        graph: Admg,
        catalog: Any,
        laws: Any,
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> str: ...
    def execute(
        self,
        graph: Admg,
        catalog: Any,
        laws: Any,
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_result(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[bytes | None, str | None]: ...

class DesignSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> DesignSessionHandle: ...
    @staticmethod
    def consume(
        artifact: bytes,
        names: list[str] | None = None,
        columns: list[Any] | None = None,
        specification: str | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[DesignSessionHandle | None, str | None, bytes | None, str | None, str | None]: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def plan(
        self,
        names: list[str],
        columns: list[Any],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: list[Any],
        specification: str,
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_result(self) -> tuple[bytes | None, str | None]: ...

class BayesianSessionHandle:
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> BayesianSessionHandle: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def plan(
        self,
        names: list[str],
        columns: list[Any],
        specification: str,
        inference: dict[str, Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        names: list[str],
        columns: list[Any],
        specification: str,
        inference: dict[str, Any],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_prior_source(self) -> tuple[bytes | None, str | None]: ...
    def effect_draws(self) -> list[float] | None: ...
    def export_result(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[bytes | None, str | None]: ...

class TemporalSessionHandle:
    def measured_interval(
        self,
        config_json: str,
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[NativeMeasuredInference | None, str | None]: ...
    def __init__(self) -> None: ...
    @staticmethod
    def resume(previous_json: str, resume_json: str) -> TemporalSessionHandle: ...
    def is_live(self) -> bool: ...
    def identities_json(self) -> str: ...
    def capabilities_json(self) -> str: ...
    def plan(
        self,
        specification: str,
        histories: tuple[NDArray[np.uint64], ...],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def execute(
        self,
        specification: str,
        histories: tuple[NDArray[np.uint64], ...],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def export_result(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[bytes | None, str | None]: ...
    @staticmethod
    def consume(
        artifact: bytes, *, seed: int = 1, threads: int | None = None
    ) -> tuple[TemporalSessionHandle | None, str | None, bytes | None, str | None]: ...
    # Optional calibration-internal checked-source lifecycle only.
    def interval_candidate(
        self,
        config_json: str,
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[NativeCheckedTemporalIntervalCandidate | None, str | None]: ...
    def dependent_interval(self) -> str: ...

# Finite source-state terminal decision and study-ranking artifact.
def export_rollout(
    source: bytes,
    ranking: bytes,
    binding_json: str,
    artifact_id: str,
    source_evidence: bytes | None = None,
) -> tuple[str, bytes]: ...
def consume_rollout(artifact: bytes, expectation_json: str) -> str: ...

class ExternalCallbackProviderHandle:
    def __init__(self, text: str, callback: Any) -> None: ...

class ExternalCallbackSessionHandle:
    def __init__(self) -> None: ...
    def is_live(self) -> bool: ...
    def plan(
        self,
        names: list[str],
        columns: list[Any],
        text: str,
        supplied_provider: bool,
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...
    def execute(
        self,
        names: list[str],
        columns: list[Any],
        text: str,
        provider: ExternalCallbackProviderHandle | None = None,
        *,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, ExternalClaimArtifact | None, str | None]: ...
    def export_output(self) -> tuple[bytes | None, str | None]: ...
    @staticmethod
    def resume(
        data: bytes,
        names: list[str],
        columns: list[Any],
        text: str,
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[ExternalCallbackSessionHandle | None, str | None]: ...

def estimate_proposal_arrival(
    graph: Admg,
    catalog: Any,
    base_laws: Any,
    arrived_laws: Any,
    repair: bytes,
    ranking: bytes,
    request_json: str,
    seed: int,
) -> tuple[str, bytes]: ...
def consume_proposal_arrival(artifact: bytes, expected_proposal: str, seed: int) -> str: ...
def review_effect_constancy(artifact: bytes, identity_json: str, request_json: str) -> str: ...

class SourceEvidenceHandle:
    summary_json: str
    def diagnostics_at(self, quantity_json: str) -> str: ...
    def resolve_with(self, actual: NativeResponseClaim) -> SourceEvidenceHandle: ...
    def project(self, contract_json: str, actions: list[str]) -> SourceEvidenceHandle: ...
    def export(self) -> bytes: ...
    def original_artifact(self) -> bytes: ...
    @staticmethod
    def consume(bytes: bytes) -> SourceEvidenceHandle: ...

class CompositeSessionHandle:
    def __init__(self, native: StaticResponseSessionHandle, payload: dict[str, object]) -> None: ...
    def plan(
        self,
        payload: dict[str, object],
        *,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...
    def execute(
        self,
        payload: dict[str, object],
        providers: dict[int, ExternalCallbackProviderHandle],
        *,
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def rank_conditional_studies(
        self, policy_json: str
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def consume_conditional_studies(
        self, artifact: bytes, expected_identity: str | None = None
    ) -> tuple[str | None, bytes | None, str | None]: ...
    def native_response(self) -> ResponseAnalysisResult | None: ...
    def export_native(self) -> bytes: ...
    def export_callback(self, branch: int) -> bytes: ...

# Original-source projections and opaque actual execution resolution.
def produce_source_projection(
    source: bytes, native: bool, projection_json: str
) -> tuple[str, list[int]]: ...
def consume_source_projection(data: bytes, expected_identity: str) -> str: ...
def consume_source_projection_bundle(
    data: bytes,
    expected_identity: str,
    supplied_json: str | None,
    actual_claims: dict[str, NativeResponseClaim],
) -> tuple[str | None, str | None]: ...
def export_law_functional_source(
    contract_json: str, action_id: str, functional_json: str, input: DecisionInput
) -> tuple[str, list[int]]: ...
def consume_law_functional_source(bytes: bytes, expected_identity: str | None = None) -> str: ...
def rollout_ranking(artifact: bytes, expectation_json: str) -> list[int]: ...
def sensitivity_source_summary(artifact: SensitivityArtifact) -> str | None: ...
def original_sensitivity_source(artifact: SensitivityArtifact) -> bytes | None: ...
def export_sensitivity_source(artifact: SensitivityArtifact) -> bytes: ...
def consume_sensitivity_source(data: bytes) -> SensitivityArtifact: ...
def resolve_law_functional_source(bytes: bytes, actual: NativeResponseClaim) -> str: ...

class SourceBackedMeanClaim: ...

def source_backed_native_mean(
    original: NativeResponseClaim, declaration: str
) -> SourceBackedMeanClaim: ...
def observe_native_attempts(
    operation: Callable[[], Any],
) -> tuple[Any | None, BaseException | None, str]:
    """Execute once and separately observe original synchronous native component work."""

class CheckedPriorSignal:
    @property
    def diagnostics_json(self) -> str: ...

def adapt_prior_signal(
    sources: list[dict[str, Any]], request_json: str
) -> tuple[CheckedPriorSignal | None, str | None]: ...

# Feature-only nested Fisher candidate lifecycle; absent from normal wheels.
class NativeNestedFisherCandidate:
    @property
    def identity(self) -> str: ...
    def payload(self) -> str: ...
    def export(self) -> bytes: ...

def nested_markov_fisher_candidate(
    variables: list[str],
    directed: list[tuple[int, int]],
    bidirected: list[tuple[int, int]],
    regimes: list[tuple[list[int] | None, list[int], list[float]]],
    nominal_level: float,
    max_iterations: int,
    tolerance: float,
) -> NativeNestedFisherCandidate: ...
def consume_nested_markov_fisher_candidate(
    bytes: bytes,
    expected_identity: str,
) -> NativeNestedFisherCandidate: ...

# Present only on internal candidate feature builds, never a release activation switch.
def joint_transport_candidate(
    proof: TransportIdentificationResult, request_json: str, learned: bool
) -> tuple[str, bytes]: ...
def consume_joint_transport_candidate(data: bytes, expectation_json: str, learned: bool) -> str: ...

# Feature-only original continuous Verma posterior; absent from normal wheels.
class NativeNestedMarkovPosteriorCandidate:
    @property
    def identity(self) -> str: ...
    def payload(self) -> str: ...
    def export(self) -> bytes: ...

def nested_markov_posterior_candidate(
    variables: list[str],
    directed: list[tuple[int, int]],
    bidirected: list[tuple[int, int]],
    regimes: list[tuple[list[int] | None, list[int], list[float]]],
    max_iterations: int,
    tolerance: float,
    alpha: list[float],
    beta: list[float],
    seed: int,
) -> NativeNestedMarkovPosteriorCandidate: ...
def consume_nested_markov_posterior_candidate(
    bytes: bytes,
    expected_identity: str,
) -> NativeNestedMarkovPosteriorCandidate: ...

# Optional: calibration-internal acceptance builds only.
def consume_sampled_recovery_candidate(
    artifact: bytes,
    premises_digest: str,
    data_digest: str,
    *,
    max_rows: int = 100_000,
    max_replicates: int = 2000,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

# Optional calibration-internal original checked TemporalSession source/proof route.
class NativeCheckedTemporalIntervalCandidate:
    def payload(self) -> str: ...
    def export(self) -> bytes: ...

def consume_checked_temporal_interval_candidate(
    artifact: bytes,
    expected_identity_json: str,
    *,
    max_units: int = 4096,
    max_histories: int = 100_000,
    max_replicates: int = 2000,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[NativeCheckedTemporalIntervalCandidate | None, str | None]: ...

class NativeMeasuredInference:
    def source_report(self) -> str: ...
    def payload(self) -> str: ...
    def export(self) -> bytes: ...
    def source_artifact(self) -> bytes: ...

def joint_transport_measured(
    identification: TransportIdentificationResult,
    request_json: str,
    learned: bool,
    *,
    level: float = 0.95,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> NativeMeasuredInference: ...
def consume_measured_inference(
    artifact: bytes,
    expected_identity_json: str,
    *,
    max_bytes: int = 67108864,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> NativeMeasuredInference: ...
def nested_markov_fisher_measured(
    variables: list[str],
    directed: list[tuple[int, int]],
    bidirected: list[tuple[int, int]],
    regimes: list[tuple[list[int] | None, list[int], list[float]]],
    nominal_level: float,
    max_iterations: int,
    tolerance: float,
    *,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> NativeMeasuredInference: ...
def nested_markov_posterior_measured(
    variables: list[str],
    directed: list[tuple[int, int]],
    bidirected: list[tuple[int, int]],
    regimes: list[tuple[list[int] | None, list[int], list[float]]],
    max_iterations: int,
    tolerance: float,
    alpha: list[float],
    beta: list[float],
    seed: int,
    *,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> NativeMeasuredInference: ...
def temporal_dependent_interval_measured(
    snapshot_id: str,
    units: list[tuple[int, list[tuple[int, int, int, int, int, float]]]] | None,
    sequence: tuple[int, int],
    estimand: str,
    fixed_state: int | None,
    law: tuple[str, str, list[tuple[int, float]]] | None,
    point: int | None,
    replicates: int,
    seed: int,
    level: float,
    method: Literal["studentized"],
    min_units: int,
    max_failed_fraction: float,
    *,
    memory_limit_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[NativeMeasuredInference | None, str | None]: ...

class PreparedMeasurementHandle:
    def model_json(self) -> str: ...
    def binding_json(self) -> str: ...
    def identities_json(self) -> str: ...
    def route_card_json(self) -> str: ...
    def check_operation(self, operation: str) -> str | None: ...
    def interval(self) -> str: ...
    def estimate(
        self, *, cancel: CancellationToken | None = None
    ) -> tuple[MeasurementEstimateHandle | None, str | None]: ...
    def estimate_continuous(
        self, input_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def refresh(
        self, data_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[PreparedMeasurementHandle | None, MeasurementEstimateHandle | None, str | None]: ...
    def with_validation_parameters(
        self,
        parameters_json: str,
        evidence_json: str,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[PreparedMeasurementHandle | None, MeasurementEstimateHandle | None, str | None]: ...

class MeasurementEstimateHandle:
    def summary_json(self) -> str: ...
    def detail_json(self) -> str: ...
    def interval(self) -> str: ...
    def downstream_linear_functional(
        self, weights: list[float]
    ) -> tuple[tuple[float, float] | None, str | None]: ...
    def export(self, artifact_id: str) -> tuple[bytes | None, str | None]: ...

class ConsumedMeasurementHandle:
    def summary_json(self) -> str: ...
    def covers_data_identity(self, identity_hex: str) -> bool: ...
    def check_use(self, binding_json: str, operation: str) -> str | None: ...

def measurement_model_prepare(
    model_json: str,
    binding_json: str,
    data_json: str,
    provider_json: str | None = None,
    receipt: ProviderVerificationReceipt | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[PreparedMeasurementHandle | None, str | None]: ...
def measurement_model_consume(
    artifact: bytes, expected_model_identity: str
) -> tuple[ConsumedMeasurementHandle | None, str | None]: ...
def measurement_model_identities(model_json: str) -> str: ...
def measurement_model_transport(
    models_json: str,
    structure_json: str | None,
    binding_json: str,
    data_json: str,
    provider_json: str | None = None,
    receipt: ProviderVerificationReceipt | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[MeasurementEstimateHandle | None, str | None, str | None]: ...
def measurement_model_transport_verify(
    models_json: str,
    structure_json: str | None,
    binding_json: str,
    expected_identity: str,
) -> tuple[str | None, str | None]: ...

class PooledDifferentialSection:
    target_population: str
    prevalence: float
    interval: tuple[float, float]
    sensitivity: tuple[float, float, float, float, float]
    specificity: tuple[float, float, float, float, float]
    discarded_draws: int
    truncated_draws: int
    draws: list[float]
    calibration: str

def measurement_pooled_differential_transport(
    sources: list[tuple[str, int, int, int, int]],
    target_population: str,
    observed_positive: int,
    observed_total: int,
    *,
    target_validation: tuple[int, int, int, int] | None = None,
    draws: int = 4000,
    seed: int = 1,
    threads: int | None = None,
) -> PooledDifferentialSection: ...

class PreparedBridgeHandle:
    def contract_json(self) -> str: ...
    def contract_identity(self) -> str: ...
    def transported(self) -> bool: ...
    def derive(self, source_json: str) -> tuple[BridgedClaimHandle | None, str | None]: ...
    def with_contract(
        self, contract_json: str
    ) -> tuple[PreparedBridgeHandle | None, str | None]: ...

class BridgedClaimHandle:
    def summary_json(self) -> str: ...
    def source_json(self) -> str: ...
    def require_value_kind(self, kind: str) -> str | None: ...
    def interval(self) -> str: ...
    def bridged_interval(self, level: float) -> tuple[str | None, str | None]: ...
    def propagated_uncertainty(self) -> tuple[str | None, str | None]: ...
    def freshness_json(self, source_withdrawn: bool, source_superseded: bool) -> str: ...
    def export(self, artifact_id: str) -> tuple[tuple[bytes, str] | None, str | None]: ...

class ConsumedBridgeHandle:
    def summary_json(self) -> str: ...
    def source_json(self) -> str: ...
    def require_value_kind(self, kind: str) -> str | None: ...
    def propagated_uncertainty(self) -> tuple[str | None, str | None]: ...
    def interval(self) -> str: ...
    def freshness_json(self, source_withdrawn: bool, source_superseded: bool) -> str: ...

def scientific_bridge_prepare(
    contract_json: str,
) -> tuple[PreparedBridgeHandle | None, str | None]: ...
def scientific_bridge_contract_identity(contract_json: str) -> str: ...
def scientific_bridge_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[ConsumedBridgeHandle | None, str | None]: ...

class PreparedDistributionBridgeHandle:
    def contract_json(self) -> str: ...
    def contract_identity(self) -> str: ...
    def transported(self) -> bool: ...
    def derive(
        self,
        distribution: JointDistributionArtifact,
        seed: int,
        claim_kind: str | None = None,
    ) -> tuple[BridgedDistributionHandle | None, str | None]: ...
    def with_contract(
        self, contract_json: str
    ) -> tuple[PreparedDistributionBridgeHandle | None, str | None]: ...

class BridgedDistributionHandle:
    def summary_json(self) -> str: ...
    def distribution(self) -> JointDistributionArtifact: ...
    def source_distribution(self) -> JointDistributionArtifact: ...
    def require_value_kind(self, kind: str) -> str | None: ...
    def interval(self) -> str: ...
    def bridged_interval(self, level: float) -> tuple[str | None, str | None]: ...
    def chain_stage_json(self, stage_id: str) -> tuple[str | None, str | None]: ...
    def freshness_json(self, source_withdrawn: bool, source_superseded: bool) -> str: ...
    def export(self, artifact_id: str) -> tuple[tuple[bytes, str] | None, str | None]: ...

def scientific_bridge_distribution_prepare(
    contract_json: str,
) -> tuple[PreparedDistributionBridgeHandle | None, str | None]: ...
def scientific_bridge_distribution_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[BridgedDistributionHandle | None, str | None]: ...
def scientific_bridge_shape(artifact: bytes) -> tuple[str | None, str | None]: ...

class CompositionVerificationHandle:
    def summary_json(self) -> str: ...
    def view_json(self, key: str) -> tuple[str | None, str | None]: ...
    def require_valid(self, key: str) -> str | None: ...
    def export(self, artifact_id: str) -> tuple[tuple[bytes, str] | None, str | None]: ...

def composition_verification_verify(
    request_json: str,
) -> tuple[CompositionVerificationHandle | None, str | None]: ...
def composition_verification_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[CompositionVerificationHandle | None, str | None]: ...
def composition_registry_record_identity(record_id: str) -> str: ...
def composition_registry_record_attests(record: str) -> bool: ...
def composition_registry_resolved_calibration(calibration_json: str) -> str: ...
def composition_registry_unresolved(
    graph_json: str,
) -> tuple[list[str] | None, str | None]: ...
def interface_contract_identity(contract_json: str) -> tuple[str | None, str | None]: ...
def interface_contract_check_connection(
    producer_json: str, produced_port: str, consumer_json: str, consumed_port: str
) -> tuple[str | None, str | None]: ...
def interface_contract_check_composition(
    components_json: str, connections_json: str
) -> tuple[str | None, str | None]: ...
def interface_contract_check_composition_with_ledger(
    components_json: str, connections_json: str, ledger: EvidenceLedgerHandle
) -> tuple[str | None, str | None]: ...
def interface_contract_check_composition_with_cyclic(
    components_json: str,
    connections_json: str,
    cyclic: list[tuple[str, _PreparedCyclicFamily | _PreparedCyclicSystem]],
    ledger: EvidenceLedgerHandle | None = None,
) -> tuple[str | None, str | None]: ...
def interface_contract_export(
    contract_json: str, artifact_id: str
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def interface_contract_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[str | None, str | None]: ...
def interface_contract_inspect(artifact: bytes) -> tuple[str | None, str | None]: ...

class CallbackToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...

class CallbackSessionHandle:
    def __init__(self) -> None: ...
    def invocations(self) -> int: ...
    def state_probes(self) -> int: ...
    def request_attempts(self, request_identity: str) -> int: ...
    def history_len(self) -> int: ...
    def record_json(self, index: int) -> str: ...
    def dependents_of(self, identity: str) -> list[int]: ...
    def forecast_law(self, index: int) -> tuple[str | None, str | None]: ...
    def plan(
        self, declaration_json: str, request_json: str, current_state: str | None = None
    ) -> tuple[str | None, str | None]: ...
    def execute(
        self,
        identity_json: str,
        invoke: object,
        state: object | None,
        declaration_json: str,
        request_json: str,
        inputs_json: str,
        policy_json: str,
        token: CallbackToken,
    ) -> tuple[str | None, str | None]: ...
    def resume(
        self,
        artifact: bytes,
        expected_identity: str | None,
        identity_json: str | None,
        invoke: object | None,
        state: object | None,
        inputs_json: str,
        policy_json: str,
        token: CallbackToken,
    ) -> tuple[str | None, str | None]: ...
    def export_record(
        self, index: int, artifact_id: str
    ) -> tuple[tuple[bytes, str] | None, str | None]: ...

def callback_declaration_identity(declaration_json: str) -> tuple[str | None, str | None]: ...
def callback_request_identity(declaration_json: str, request_json: str) -> str: ...
def callback_values_digest(values: list[float]) -> str: ...
def callback_result_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[str | None, str | None]: ...
def callback_result_inspect(artifact: bytes) -> tuple[str | None, str | None]: ...
def callback_closed_refusal(route: str) -> str: ...
def callback_predictive_law_source(
    declaration_json: str, verification_confirmed: bool
) -> tuple[str | None, str | None]: ...
def callback_check_claim_connection(
    producer_json: str, produced_port: str, callback_json: str, input: str
) -> tuple[str | None, str | None]: ...
def callback_register_source(
    ledger: EvidenceLedgerHandle, declaration_json: str, inner_sources_json: str
) -> tuple[str | None, str | None]: ...
def callback_supersede_source(
    ledger: EvidenceLedgerHandle, old: str, replacement_json: str, inner_sources_json: str
) -> tuple[str | None, str | None]: ...

class PreparedCompositionHandle:
    def plan_identity(self) -> str: ...
    def family(self) -> str: ...
    def variables(self) -> list[str]: ...
    def declared_mechanisms(self) -> list[str]: ...
    def interval(self) -> str: ...
    def intervention_law(self, interventions_json: str) -> tuple[str | None, str | None]: ...
    def mean_contrast(self, contrast_json: str) -> tuple[float | None, str | None]: ...
    def mean_contrast_uncertainty(self, contrast_json: str) -> tuple[str | None, str | None]: ...
    def sample(
        self,
        interventions_json: str,
        n_rows: int,
        seed: int,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[tuple[list[str], list[list[float]]] | None, str | None]: ...
    def finite_law(
        self, interventions_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def refresh_mechanisms(
        self,
        refits_json: str,
        dependence_json: str | None = None,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[PreparedCompositionHandle | None, str | None]: ...
    def export(
        self, query_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
    def transport_to(
        self, transport_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[tuple[TransportedCompositionHandle, str] | None, str | None]: ...
    def verify_transport(
        self,
        transport_json: str,
        expected_identity: str,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...

class TransportedCompositionHandle:
    def identity(self) -> str: ...
    def receipt_json(self) -> str: ...
    def mean_contrast(self, contrast_json: str) -> tuple[float | None, str | None]: ...
    def mean_contrast_uncertainty(self, contrast_json: str) -> tuple[str | None, str | None]: ...
    def interval(self) -> str: ...
    def export(
        self, query_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[tuple[bytes, str, str] | None, str | None]: ...

def mechanism_composition_prepare(
    composition_json: str, *, cancel: CancellationToken | None = None
) -> tuple[PreparedCompositionHandle | None, str | None]: ...
def mechanism_composition_consume(
    artifact: bytes,
    expected_premises_digest: str | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def mechanism_composition_fitted_source(
    model: FittedGcm, variable: str
) -> tuple[str | None, str | None]: ...
def observation_query_observe(
    handle: PreparedCompositionHandle,
    request_json: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_observe_with_uncertainty(
    handle: PreparedCompositionHandle,
    request_json: str,
    uncertainty_json: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_parameter_interval(
    artifact: bytes,
    quantity: str,
    construction: str,
    evidence_rows: int,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def observation_query_pair(
    handle: PreparedCompositionHandle,
    outcome: str,
    assignment_json: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_threshold(
    handle: PreparedCompositionHandle,
    request_json: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_finite_posterior(
    model_json: str,
    observations: list[str],
    basis: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_conjugate_posterior(
    alpha: float,
    beta: float,
    successes: int,
    failures: int,
    basis: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_required_bayes_factor(
    prior: float,
    level: float,
    direction: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_required_observations_binary(
    prior: float,
    level: float,
    direction: str,
    rate_given_hypothesis: float,
    rate_given_complement: float,
    max_observations: int,
    basis: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_required_observations_conjugate(
    alpha: float,
    beta: float,
    level: float,
    direction: str,
    max_observations: int,
    basis: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, bytes] | None, str | None]: ...
def observation_query_closed(route: str) -> str: ...
def observation_query_consume(
    artifact: bytes,
    expected_identity: str | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...

class EvidenceLedgerHandle:
    def __init__(self) -> None: ...
    def head(self) -> str: ...
    def state_digest(self) -> str: ...
    def ledger_json(self) -> str: ...
    def result_json(self, result: str) -> str | None: ...
    def standing(self, result: str) -> str | None: ...
    def add_source(self, declaration_json: str) -> tuple[str | None, str | None]: ...
    def declare_from_result(
        self, result: str, label: str, kind: str
    ) -> tuple[str | None, str | None]: ...
    def account_observations(self, sources_json: str) -> tuple[str | None, str | None]: ...
    def pool_summaries(
        self,
        request_json: str,
        dependence_json: str,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...
    def random_effects(
        self, request_json: str, method: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def hierarchical_borrowing(
        self, request_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def dependence_range(
        self, request_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def transport_claim(
        self,
        label: str,
        transported_json: str,
        native_json: str,
        replaces: str | None = None,
    ) -> tuple[str | None, str | None]: ...
    def reuse_result(self, label: str, of: str) -> tuple[str | None, str | None]: ...
    def repeat_observations(
        self, label: str, sources_json: str, stage: str
    ) -> tuple[str | None, str | None]: ...
    def revise_source(self, revision_json: str) -> tuple[str | None, str | None]: ...
    def revalidate_downstream(self) -> tuple[str | None, str | None]: ...
    def pooled_interval(self, result: str) -> tuple[str | None, str | None]: ...
    def synthesis(self, result: str) -> tuple[str | None, str | None]: ...
    def refresh(
        self, changes_json: str
    ) -> tuple[tuple[EvidenceLedgerHandle, str, str] | None, str | None]: ...
    def export(
        self, artifact_id: str
    ) -> tuple[tuple[bytes, str, list[str], list[tuple[str, str]]] | None, str | None]: ...

class ConsumedEvidenceLedgerHandle:
    def summary_json(self) -> str: ...

def evidence_ledger_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[ConsumedEvidenceLedgerHandle | None, str | None]: ...
def evidence_ledger_observation_id(domain: str, key: bytes) -> str: ...

class PredictiveAssessmentHandle:
    def report_json(self) -> str: ...
    def identity_json(self) -> str: ...
    def export(self, artifact_id: str) -> tuple[bytes | None, str | None]: ...
    def score_interval(
        self, predictor: str, reference: str | None, score: str, level: float
    ) -> tuple[str | None, str | None]: ...
    def request_score_interval(self) -> str: ...
    def request_causal_validity(self) -> str: ...
    def request_score_as_probability(self) -> str: ...

def predictive_assessment_assess(
    spec_json: str, *, seed: int = 0, cancel: CancellationToken | None = None
) -> tuple[PredictiveAssessmentHandle | None, str | None]: ...
def predictive_assessment_consume(
    artifact: bytes,
    expected_identity_json: str | None = None,
    max_cases: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[PredictiveAssessmentHandle | None, str | None]: ...
def predictive_assessment_score_case(
    kind: str, form: str, law_json: str, outcome_json: str
) -> tuple[float | None, str | None]: ...
def predictive_assessment_exact_expected_score(
    kind: str, law_json: str, truth_json: str
) -> tuple[float | None, str | None]: ...
def predictive_assessment_drift(
    reference: list[float], comparison: list[float], bins: int
) -> tuple[str | None, str | None]: ...
def predictive_assessment_observation_key(domain: str, key: bytes) -> str: ...
def predictive_assessment_ledger_fit(
    ledger: EvidenceLedgerHandle, sources_json: str
) -> tuple[str | None, str | None]: ...
def predictive_assessment_ledger_observations(
    ledger: EvidenceLedgerHandle, source: str
) -> tuple[list[str] | None, str | None]: ...

class AdaptiveInferenceHandle:
    def contract_identity(self) -> str: ...
    def summary_json(self) -> str: ...
    def analyse(
        self, request_json: str, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def fixed_sample(self) -> str | None: ...
    def interval(self) -> str | None: ...
    def measured_interval(
        self, request_json: str, cancel: CancellationToken | None = None
    ) -> tuple[NativeMeasuredInference | None, str | None]: ...
    def interim_evidence(self) -> tuple[str | None, str | None]: ...
    def export_contract(
        self, paradigm: str, artifact_id: str
    ) -> tuple[bytes | None, str | None]: ...
    def export_contract_with_procedure(
        self, paradigm: str, procedure: ParticipantProcedureHandle, artifact_id: str
    ) -> tuple[bytes | None, str | None]: ...
    def export_result(
        self, request_json: str, artifact_id: str, cancel: CancellationToken | None = None
    ) -> tuple[bytes | None, str | None]: ...

class ConsumedMonitoringContractHandle:
    def summary_json(self) -> str: ...

class ConsumedAdaptiveResultHandle:
    def summary_json(self) -> str: ...

def adaptive_inference_prepare(
    declaration_json: str,
) -> tuple[AdaptiveInferenceHandle | None, str | None]: ...
def adaptive_monitoring_contract_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[ConsumedMonitoringContractHandle | None, str | None]: ...
def adaptive_inference_result_consume(
    artifact: bytes,
    expected_identity: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[ConsumedAdaptiveResultHandle | None, str | None]: ...
def adaptive_procedure_matrix() -> str: ...

class AdaptiveProcedureHandle:
    def identity(self) -> str: ...
    def summary_json(self) -> str: ...
    def law(
        self, truth: list[float], cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def simulate(
        self, truth: list[float], request_json: str, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def report(
        self, request_json: str, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...

class RepeatedMeasuresProcedureHandle:
    def identity(self) -> str: ...
    def cell_json(self) -> str: ...
    def report(
        self, request_json: str, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def posterior(self, request_json: str) -> tuple[str | None, str | None]: ...
    def posterior_artifact(self, report_json: str) -> tuple[str | None, str | None]: ...
    def load_posterior_artifact(self, artifact: str) -> tuple[str | None, str | None]: ...

def adaptive_procedure_declare(
    declaration_json: str,
) -> tuple[AdaptiveProcedureHandle | None, str | None]: ...
def adaptive_procedure_from_contract(
    session: AdaptiveInferenceHandle,
    elements_json: str,
    cancel: CancellationToken | None = None,
) -> tuple[AdaptiveProcedureHandle | None, str | None]: ...
def repeated_measures_procedure_declare(
    declaration_json: str,
) -> tuple[RepeatedMeasuresProcedureHandle | None, str | None]: ...

class ParticipantProcedureHandle:
    def identity(self) -> str: ...
    def summary_json(self) -> str: ...
    def simulate(
        self,
        truth: list[float],
        request_json: str,
        memory_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...
    def simulate_history(
        self,
        truth: list[float],
        seed: int,
        true_correlation: float | None = None,
        cancel: CancellationToken | None = None,
    ) -> tuple[str | None, str | None]: ...
    def read_history(
        self, history_json: str, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def amend_borrowing(
        self, borrowing_json: str, looks_performed: int
    ) -> tuple[ParticipantProcedureHandle | None, str | None]: ...
    def prepare_session(
        self,
        label: str,
        arm_labels: list[str],
        history_json: str,
        cancel: CancellationToken | None = None,
    ) -> tuple[AdaptiveInferenceHandle | None, str | None]: ...

def participant_procedure_declare(
    declaration_json: str,
) -> tuple[ParticipantProcedureHandle | None, str | None]: ...
def decision_reliability_assess(
    draws_json: str,
    validation_json: str | None,
    request_json: str,
    artifact_id: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def decision_reliability_consume(
    artifact: bytes,
    expected_json: str | None = None,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[str, str] | None, str | None]: ...
def decision_reliability_rank(
    draws_json: str,
    request_json: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def decision_reliability_digests(
    draws_json: str,
) -> tuple[tuple[str, str] | None, str | None]: ...
def decision_reliability_publish(route: str, coordinate_json: str) -> str | None: ...
def decision_reliability_derive(
    draws_json: str,
    validation_json: str | None,
    request_json: str,
    interval_json: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def decision_reliability_consume_derived(
    artifact: bytes,
    expected_json: str | None,
    interval_json: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def decision_reliability_verify_derived(
    artifact: bytes,
    derived_json: str,
    expected_json: str | None,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> str | None: ...
def decision_reliability_publish_derived(route: str, derived_json: str) -> str | None: ...
def decision_reliability_coverage_design_json() -> tuple[str | None, str | None]: ...
def decision_reliability_reach_carrier(output_json: str) -> tuple[str | None, str | None]: ...
def decision_reliability_assess_producer(
    output_json: str,
    validation_json: str | None,
    request_json: str,
    artifact_id: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def decision_reliability_derive_producer(
    output_json: str,
    validation_json: str | None,
    request_json: str,
    interval_json: str,
    seed: int = 0,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def joint_law_column(
    input_id: str, artifact: JointDistributionArtifact, column: int
) -> tuple[str | None, str | None]: ...
def joint_law_aligned_functional(
    operation_name: str,
    x_id: str,
    x: JointDistributionArtifact,
    x_column: int,
    y_id: str,
    y: JointDistributionArtifact,
    y_column: int,
    functional: str,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_aligned_linear(
    operation_name: str,
    a: float,
    b: float,
    x_id: str,
    x: JointDistributionArtifact,
    x_column: int,
    y_id: str,
    y: JointDistributionArtifact,
    y_column: int,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_linear_moments(
    operation_name: str,
    inputs_json: str,
    coefficients: list[float],
    covariance: list[float],
    dependence: str,
    covariance_id: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_mixture(
    components_json: str, cancel: CancellationToken | None = None
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_mixture_undeclared(
    ids_json: str, weights: list[float], means: list[float], variances: list[float]
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_envelope(
    operation_name: str,
    query_json: str,
    x_id: str,
    x_evidence: str,
    x_values: list[float],
    y_id: str,
    y_evidence: str,
    y_values: list[float],
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_correlation_range(
    operation_name: str,
    x_id: str,
    x_evidence: str,
    y_id: str,
    y_evidence: str,
    x_mean: float,
    x_sd: float,
    y_mean: float,
    y_sd: float,
    a: float,
    b: float,
    rho_lo: float,
    rho_hi: float,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_consume(
    artifact: bytes,
    expected_identity: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def joint_law_closed_reading(reading: str) -> str: ...
def joint_law_linear_moments_spread(
    operation_name: str,
    inputs_json: str,
    coefficients: list[float],
    covariance: list[float],
    dependence: str,
    covariance_id: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_spread(
    operation_name: str,
    x_id: str,
    x: JointDistributionArtifact,
    x_column: int,
    y_id: str,
    y: JointDistributionArtifact,
    y_column: int,
    functional: str,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_aligned_linear_spread(
    operation_name: str,
    a: float,
    b: float,
    x_id: str,
    x: JointDistributionArtifact,
    x_column: int,
    y_id: str,
    y: JointDistributionArtifact,
    y_column: int,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_composed_exceedance(
    operation_name: str,
    x_id: str,
    x: JointDistributionArtifact,
    x_column: int,
    y_id: str,
    y: JointDistributionArtifact,
    y_column: int,
    functional: str,
    threshold: float,
    cutoff: float,
    band_level: float,
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_mixture_exceedance(
    components_json: str, threshold: float, cancel: CancellationToken | None = None
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_transported_spread_aligned(
    source_id: str,
    source: JointDistributionArtifact,
    source_column: int,
    target_id: str,
    target: JointDistributionArtifact,
    target_column: int,
    functional_id: str,
    functional: JointDistributionArtifact,
    functional_column: int,
    coefficients: list[float],
    cancel: CancellationToken | None = None,
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def joint_law_check_composed_interval(
    artifact: bytes,
    construction: str,
    evidence_rows: int,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def joint_law_coordinate_id(artifact: bytes) -> tuple[str | None, str | None]: ...
def propagation_chain_seal(
    record_json: str, cancel: CancellationToken | None = None
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def propagation_chain_change(
    record_json: str, change_json: str, cancel: CancellationToken | None = None
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def propagation_chain_plan_view(
    record_json: str, cancel: CancellationToken | None = None
) -> tuple[str | None, str | None]: ...
def propagation_chain_consume(
    artifact: bytes,
    expected_identity: str | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def propagation_chain_dependents(
    record_json: str, source: str
) -> tuple[list[str] | None, str | None]: ...
def propagation_chain_product_table(
    columns: list[list[float]], max_rows: int
) -> tuple[list[list[float]] | None, str | None]: ...
def propagation_chain_family_table(
    alignment: str,
    estimates: list[
        tuple[str, float, float, str, list[tuple[str, str]], bool, str | None]
    ],
    dependence: str,
    covariance: list[float] | None = None,
    ledger: EvidenceLedgerHandle | None = None,
    rows: int = 1597,
    seed: int = 0,
) -> tuple[tuple[list[str], list[float], list[tuple[str, str]]] | None, str | None]: ...
def finite_discrete_adjustment_contrast(
    counts: list[int],
    draws: list[tuple[list[int], float]],
    cancel: CancellationToken | None = None,
) -> tuple[float, float, float, float, int, int]: ...

class LogisticProvider:
    def __init__(self, id: int, name: str) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def capabilities(self) -> list[str]: ...

def recipe_logistic_standardized_effect(
    columns: list[tuple[str, list[float]]],
    treatment: str,
    outcome: str,
    adjustment: list[str],
    provider: LogisticProvider,
    cancel: CancellationToken | None = None,
) -> tuple[float, float, float, list[float], list[float], int, int]: ...
def cohort_checked_projection(
    closure: tuple[list[str], list[str], list[str], list[str], list[str], list[str]],
    observed: list[tuple[str, list[bool]]],
    keep: list[str],
) -> list[int]: ...
def cohort_checked_push_filter(
    closure: tuple[list[str], list[str], list[str], list[str], list[str], list[str]],
    observed: list[tuple[str, list[bool]]],
    column: str,
    drop_row: list[bool],
) -> list[int]: ...
def score_published_decision(
    demand: str,
    mean: float,
    posterior_mean: float | None = None,
    slope: float | None = None,
    intercept: float | None = None,
) -> tuple[tuple[float, str] | None, str | None]: ...
def compile_design_ranking_for_study(
    request_json: str,
    names: list[str],
    columns: list[list[float]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    posterior: bool = True,
    n_draws: int = 256,
    prior_scale: float = 1.0e6,
    checked_prior: CheckedPriorSignal | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def evaluate_design_ranking_for_study(
    request_json: str,
    artifact_id: str,
    names: list[str],
    columns: list[list[float]],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    posterior: bool = True,
    n_draws: int = 256,
    prior_scale: float = 1.0e6,
    checked_prior: CheckedPriorSignal | None = None,
    memory_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, bytes | None, str | None, str | None]: ...
def discovered_transport_trial_estimate_run(
    contract_json: str,
    names: list[str],
    source_columns: list[NDArray[np.float64]],
    target_columns: list[NDArray[np.float64]],
    query_json: str,
    selection_targets: list[int],
    alpha: float,
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    *,
    discovery_columns: list[NDArray[np.float64]] | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None, str | None, str | None, bytes | None]: ...
def verify_trial_compatible_transport_estimate_artifact(
    artifact: bytes, *, seed: int = 0
) -> str: ...
def discovered_transport_trial_confidence_set_run(
    contract_json: str,
    names: list[str],
    source_columns: list[NDArray[np.float64]],
    target_columns: list[NDArray[np.float64]],
    query_json: str,
    selection_targets: list[int],
    alpha: float,
    support_lower: float,
    support_upper: float,
    member_bound: int | None = None,
    bootstrap_replicates: int = 200,
    *,
    discovery_columns: list[NDArray[np.float64]] | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None, str | None, str | None, bytes | None]: ...
def verify_trial_confidence_set_against_inputs(
    receipt: bytes,
    contract_json: str,
    names: list[str],
    source_columns: list[NDArray[np.float64]],
    target_columns: list[NDArray[np.float64]],
    query_json: str,
    selection_targets: list[int],
    alpha: float,
    support_lower: float,
    support_upper: float,
    member_bound: int | None = None,
    bootstrap_replicates: int = 200,
    *,
    discovery_columns: list[NDArray[np.float64]] | None = None,
    threads: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None, str | None, str | None, bytes | None]: ...

class SequentialBeliefSession:
    @staticmethod
    def start(
        model_json: str, session: str
    ) -> tuple[SequentialBeliefSession | None, str | None]: ...
    def observe(self, action: int, symbol: int, record: str) -> tuple[str | None, str | None]: ...
    @property
    def belief(self) -> list[float]: ...
    @property
    def history_json(self) -> str: ...
    @property
    def time(self) -> int: ...
    @property
    def resource_used(self) -> float: ...
    @property
    def model_identity(self) -> str: ...
    @property
    def ledger_state(self) -> str: ...
    @property
    def recorded(self) -> int: ...

class SequentialEvaluationArtifact:
    @staticmethod
    def seal(
        model_json: str,
        policy_json: str,
        off_policy_json: str | None = None,
        rollout_json: str | None = None,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[SequentialEvaluationArtifact | None, str | None]: ...
    @staticmethod
    def consume(
        data: bytes,
        expected_json: str | None = None,
        *,
        cancel: CancellationToken | None = None,
    ) -> tuple[SequentialEvaluationArtifact | None, str | None]: ...
    @property
    def value(self) -> float: ...
    @property
    def identities_json(self) -> str: ...
    @property
    def summary_json(self) -> str: ...
    @property
    def model_json(self) -> str: ...
    @property
    def policy_json(self) -> str: ...
    def export(self, artifact_id: str) -> bytes: ...

def sequential_model_check(model_json: str) -> tuple[str | None, str | None]: ...
def sequential_evaluate(
    model_json: str,
    policy_json: str,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def sequential_off_policy(
    model_json: str,
    policy_json: str,
    behaviour_json: str,
    log_json: str,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def sequential_rollout(
    model_json: str,
    policy_json: str,
    seed: int,
    rollouts: int,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def sequential_value_of_evidence(
    model_json: str,
    voi_request_json: str,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def sequential_compare(
    model_json: str,
    comparisons_json: str,
    voi_request_json: str | None = None,
    request_json: str | None = None,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[str | None, str | None]: ...
def sequential_check_request(model_json: str, request_json: str) -> str | None: ...
def sequential_resolve_policy(spec_json: str) -> tuple[str | None, str | None]: ...
def sequential_request_interval() -> str: ...
def sequential_stream_uniform(seed: int, rollout: int, time: int, role: str) -> float: ...

class SelectiveDiscoveryHandle:
    def __init__(self, kind: str) -> None: ...
    def run(
        self,
        contract_json: str,
        names: list[str],
        columns: list[Any],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> tuple[str | None, str | None]: ...
    def plan(
        self,
        contract_json: str,
        names: list[str],
        columns: list[Any],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> tuple[str | None, str | None]: ...
    def outcome_json(self) -> str | None: ...
    def retained_statistics(self) -> int: ...
    def add_claims(
        self,
        graph_json: str | None,
        prefix: str,
        completion_label: str | None = None,
        completion_identity: str | None = None,
    ) -> tuple[str | None, str | None]: ...
    def export(self, artifact_id: str) -> tuple[tuple[str, bytes, str] | None, str | None]: ...
    def request_recovery_claim(self) -> str: ...
    def request_downstream_interval(self) -> str: ...
    def request_selection_diagram(self, targets: list[int]) -> str: ...
    def transport_across_class(
        self,
        query_json: str,
        selection_targets: list[int],
        member_bound: int | None = None,
    ) -> tuple[str | None, str | None]: ...
    def transport_across_temporal_class(
        self,
        query_json: str,
        selection_targets: list[int],
        horizon_steps: int,
        history: int | None = None,
        member_bound: int | None = None,
    ) -> tuple[str | None, str | None]: ...

class PerPopulationDiscoveryHandle:
    def __init__(self) -> None: ...
    def run(
        self,
        contract_json: str,
        populations: list[str],
        names: list[str],
        columns: list[list[Any]],
        *,
        threads: int | None = None,
        cancel: Any = None,
    ) -> tuple[str | None, str | None]: ...
    def report_json(self) -> str | None: ...
    def add_claims(self, graph_json: str | None, prefix: str) -> tuple[str | None, str | None]: ...
    def outcome_json(self, population: str) -> str | None: ...
    def retained_statistics(self, population: str) -> int: ...
    def request_selection_targets(self, candidates: list[int]) -> str: ...

class SampleSplitDiscoveryHandle:
    def receipt_json(self) -> str: ...
    def outcome_json(self) -> str: ...
    def discovery_columns(self) -> list[Any]: ...
    def analysis_columns(self) -> list[Any]: ...
    def evidence_sources(self, snapshot_id: bytes) -> tuple[str | None, str | None]: ...
    def verify(self) -> str | None: ...
    def discovery_row_ids(self) -> list[int]: ...
    def analysis_row_ids(self) -> list[int]: ...
    def analysis_weights(self) -> list[float] | None: ...
    def verify_source(
        self,
        names: list[str],
        columns: list[Any],
        weights: list[float] | None,
        mask: list[bool] | None,
    ) -> str | None: ...
    def check_analysis_table(
        self, names: list[str], columns: list[Any], weights: list[float] | None
    ) -> tuple[str | None, str | None]: ...
    def request_unsplit_interval(self) -> str: ...

def selective_discovery_resume(
    kind: str,
    contract_json: str,
    names: list[str],
    columns: list[Any] | None,
    artifact: bytes,
    expected_identity: str | None = None,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[SelectiveDiscoveryHandle | None, str | None, str | None]: ...
def selective_discovery_inspect(artifact: bytes) -> tuple[str | None, str | None]: ...
def selective_discovery_consume(
    artifact: bytes, expected_identity: str | None = None
) -> tuple[str | None, str | None]: ...
def selective_discovery_score_search_refusal() -> str: ...
def selective_discovery_data_source(
    label: str, snapshot_id: bytes, start: int, stop: int
) -> str: ...
def selective_discovery_rows_source(label: str, snapshot_id: bytes, rows: list[int]) -> str: ...
def selective_discovery_graph_source(label: str, data_sources: list[str]) -> str: ...
def sample_split_discover(
    contract_json: str,
    names: list[str],
    columns: list[Any],
    rule_json: str,
    *,
    weights: list[float] | None = None,
    mask: list[bool] | None = None,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[SampleSplitDiscoveryHandle | None, str | None, str | None]: ...
def sample_split_effect(
    names: list[str],
    columns: list[Any],
    treatment: int,
    outcome: int,
    split_seed: int,
    seed: int,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, bytes | None, str | None]: ...
def sample_split_effect_measure(
    source: bytes,
    seed: int,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[NativeMeasuredInference | None, str | None]: ...
def sample_split_parent_set_effect(
    names: list[str],
    columns: list[Any],
    treatment: int,
    outcome: int,
    split_seed: int,
    seed: int,
    transformation: str,
    max_degree: int,
    log_shift: float | None = None,
    knowledge: str | None = None,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, bytes | None, str | None]: ...
def sample_split_parent_set_effect_measure(
    source: bytes,
    seed: int,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[NativeMeasuredInference | None, str | None]: ...
def discovered_transport_run(
    contract_json: str,
    names: list[str],
    columns: list[Any],
    query_json: str,
    selection_targets: list[int],
    member_bound: int | None = None,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None, str | None, str | None]: ...
def discovered_transport_estimate_run(
    contract_json: str,
    names: list[str],
    source_columns: list[Any],
    target_columns: list[Any],
    query_json: str,
    selection_targets: list[int],
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None, str | None, str | None, bytes | None]: ...
def verify_discovered_transport_estimate_artifact(artifact: bytes, *, seed: int = 0) -> str: ...
def discovered_temporal_graph_transport(
    graph: Any,
    horizon_steps: int,
    query_json: str,
    selection_targets: list[int],
    history: int | None = None,
    member_bound: int | None = None,
    exhausted_search_bounds: int = 0,
) -> tuple[str | None, str | None]: ...
def discovered_temporal_transport_estimate_run(
    contract_json: str,
    names: list[str],
    source_columns: list[Any],
    target_columns: list[Any],
    query_json: str,
    selection_targets: list[int],
    horizon_steps: int,
    history: int | None = None,
    member_bound: int | None = None,
    bootstrap_replicates: int = 0,
    block_length: int | None = None,
    *,
    threads: int | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None, str | None, str | None, bytes | None]: ...

class CrossWorldExecutor:
    def __init__(self) -> None: ...
    def execute(self, request_json: str, cancel: Any = None) -> tuple[str | None, str | None]: ...
    def retained_len(self) -> int: ...
    def counters(self) -> str: ...
    def clear(self) -> None: ...
    @staticmethod
    def from_artifact(
        data: bytes,
        expected_identity_json: str | None = None,
        cancel: Any = None,
    ) -> tuple[tuple[CrossWorldExecutor, str] | None, str | None]: ...

def cross_world_capability_matrix() -> str: ...
def cross_world_decide(
    quantity: str,
    model: str,
    conditioning: str,
    interventions: str,
    population: str,
    history: str,
    evidence: str,
) -> tuple[str | None, str | None]: ...
def cross_world_seal(
    request_json: str, artifact_id: str, cancel: Any = None
) -> tuple[tuple[bytes, str] | None, str | None]: ...
def cross_world_consume(
    data: bytes,
    expected_identity_json: str | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None]: ...
def cross_world_closed_reading(reading: str) -> str: ...
def inspect_artifact_container(bytes: bytes) -> str: ...
def estimate_transport_response_grid(
    identification: Any,
    *,
    outcome: Sequence[float],
    trial: Sequence[bool],
    selection_probability: Sequence[float],
    grid: Sequence[float],
    observed_regression: Sequence[float],
    target_regression: Sequence[float],
    response_weight: Sequence[float],
) -> str: ...

class TransportedResponseHandle:
    def receipt_json(self) -> str: ...
    def value_json(self) -> str: ...
    def identity(self) -> str: ...

def transported_response_declared_point(
    coordinates_json: str, means: list[float], declaration_json: str
) -> tuple[TransportedResponseHandle | None, str | None]: ...
def transported_response_declared_law(
    law: JointDistributionArtifact, declaration_json: str
) -> tuple[TransportedResponseHandle | None, str | None]: ...
def transported_response_from_shifted_cohort(
    artifact: bytes, expected_identity: str
) -> tuple[TransportedResponseHandle | None, str | None]: ...
def transported_decision_evaluate(
    prepared: PreparedAnalysis,
    declaration_json: str,
    response: TransportedResponseHandle,
    demand: str,
    unsupported: str,
    weakest_support: str,
) -> tuple[str | None, str | None]: ...
def transported_history_alphabet(model_json: str) -> tuple[str | None, str | None]: ...
def transported_policy_construct(
    model_json: str,
    policy_json: str,
    target_population: str,
    target_initial: list[float] | None,
    diagram_json: str | None,
    coordinate_json: str,
    provider_id: str,
    max_nodes: int | None = None,
    target_utility_json: str | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[TransportedResponseHandle | None, str | None, str | None]: ...
def temporal_policy_construct(
    model_json: str,
    policy_json: str,
    target_json: str,
    target_initial: list[float] | None,
    invariance_json: str | None,
    coordinate_json: str,
    provider_id: str,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[TransportedResponseHandle | None, str | None, str | None]: ...
def adaptive_trial_construct(
    design_json: str | None,
    trial_population: str,
    unit_strata: list[int],
    stratum_labels: list[str],
    target_stratum_law: list[float] | None,
    diagram_json: str | None,
    treatment_arm: int,
    control_arm: int,
    value: str,
    coordinate_json: str,
    provider_id: str,
) -> tuple[TransportedResponseHandle | None, str | None, str | None]: ...
def transported_policy_verify(
    model_json: str,
    policy_json: str,
    target_population: str,
    target_initial: list[float] | None,
    diagram_json: str | None,
    stored_json: str,
    max_nodes: int | None = None,
    target_utility_json: str | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> str | None: ...
def temporal_policy_verify(
    model_json: str,
    policy_json: str,
    target_json: str,
    target_initial: list[float] | None,
    invariance_json: str | None,
    stored_json: str,
    max_nodes: int | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> str | None: ...
def adaptive_trial_verify(
    design_json: str | None,
    trial_population: str,
    unit_strata: list[int],
    stratum_labels: list[str],
    target_stratum_law: list[float] | None,
    diagram_json: str | None,
    treatment_arm: int,
    control_arm: int,
    stored_json: str,
) -> str | None: ...
def transported_synthesis_construct(
    ledger: EvidenceLedgerHandle,
    label: str,
    contributions_json: str,
    method_json: str,
    native_population: str | None,
    coordinate_json: str,
    provider_id: str,
) -> tuple[TransportedResponseHandle | None, str | None, str | None]: ...
def surrogate_bridge_construct(
    bridge_json: str, claim_json: str, provider_id: str
) -> tuple[TransportedResponseHandle | None, str | None, str | None]: ...
def transport_sensitivity_build(
    response: TransportedResponseHandle,
    family: str,
    bound: float,
    grid_points: int,
    reference_json: str,
    reference_value: float,
    *,
    coordinate: int = 0,
    route: str | None = None,
) -> tuple[SensitivityArtifact | None, str | None, str | None]: ...

class ForwardSourceHandle:
    def origin_json(self) -> str: ...

class ObservationalScenarioHandle:
    def summary_json(self) -> str: ...
    def verify(self, law: JointDistributionArtifact) -> str | None: ...

def forward_source_composed(
    handle: PreparedCompositionHandle,
    actions_json: str,
    transport_json: str | None,
    accepts_transported: bool,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[ForwardSourceHandle | None, str | None]: ...
def forward_source_cohort(
    artifact: bytes, expected_identity: str, accepts_transported: bool
) -> tuple[ForwardSourceHandle | None, str | None]: ...
def forward_source_transported(
    response: TransportedResponseHandle, accepts_transported: bool
) -> tuple[ForwardSourceHandle | None, str | None]: ...
def forward_source_evaluate(
    source: ForwardSourceHandle, contract_json: str, query_json: str, interval_region: bool
) -> tuple[str | None, str | None]: ...
def forward_source_seal(
    source: ForwardSourceHandle, contract_json: str, query_json: str, interval_region: bool
) -> tuple[InverseQueryArtifact | None, str | None]: ...
def forward_source_seal_per_action(
    source: ForwardSourceHandle, contract_json: str, query_json: str, interval_region: bool
) -> tuple[list[tuple[str, InverseQueryArtifact]] | None, str | None]: ...
def observational_scenarios_evaluate(
    law: JointDistributionArtifact, query_json: str
) -> tuple[ObservationalScenarioHandle | None, str | None]: ...

class TransportSceneHandle:
    def summary_json(self) -> str: ...

class TransportRevalidationHandle:
    def inspect_json(self) -> str: ...
    def require_valid(self, key: str) -> tuple[str | None, str | None]: ...
    def identity(self) -> str: ...
    def export(self, artifact_id: str) -> tuple[bytes | None, str | None]: ...

class TransportClaimStateHandle:
    def __init__(self) -> None: ...
    def observe(
        self, current: TransportSceneHandle, effects_json: str | None = None
    ) -> tuple[TransportRevalidationHandle | None, str | None]: ...
    def scene(self) -> TransportSceneHandle | None: ...

def transport_scene_target_identity(target_population: str, target_data_identity: str) -> str: ...
def transport_scene_build(
    scene_json: str, responses: list[TransportedResponseHandle]
) -> tuple[TransportSceneHandle | None, str | None]: ...
def transport_revalidate(
    previous: TransportSceneHandle | None,
    current: TransportSceneHandle,
    effects_json: str | None = None,
) -> tuple[TransportRevalidationHandle | None, str | None]: ...

class DiscretizedCompositionHandle:
    def identity(self) -> str: ...
    def inspect_json(self) -> str: ...
    def mean_contrast(
        self, contrast_json: str, *, cancel: CancellationToken | None = None
    ) -> tuple[str | None, str | None]: ...
    def uncertainty(self, paradigm: str) -> str: ...
    def refresh(
        self, refit: PreparedCompositionHandle, *, cancel: CancellationToken | None = None
    ) -> tuple[DiscretizedCompositionHandle | None, str | None]: ...
    def export(self) -> tuple[bytes | None, str | None]: ...

def discretized_composition_prepare(
    handle: PreparedCompositionHandle,
    plan_json: str,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[DiscretizedCompositionHandle | None, str | None]: ...
def discretized_composition_consume(
    artifact: bytes,
    expected_identity: str | None = None,
    *,
    cancel: CancellationToken | None = None,
) -> tuple[DiscretizedCompositionHandle | None, str | None]: ...
def cross_world_whole_path(
    data: bytes,
    unit_sampling: str,
    expected_identity_json: str | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None]: ...
def cross_world_measured_interval(
    data: bytes,
    unit_sampling: str,
    expected_identity_json: str | None = None,
    cancel: Any = None,
) -> tuple[str | None, str | None]: ...
def cross_world_whole_path_verify(
    data: bytes,
    unit_sampling: str,
    account_identity: str,
    receipt_digest: str,
    expected_identity_json: str | None = None,
    cancel: Any = None,
) -> str | None: ...
def transport_handoff_seal(
    offer_json: str,
    callback_json: str,
    input: str,
    accepts_transported: bool,
    artifact_id: str,
) -> tuple[tuple[bytes, str, str] | None, str | None]: ...
def transport_handoff_consume(
    artifact: bytes, callback_json: str, expected_identity: str | None = None
) -> tuple[str | None, str | None]: ...

class RetentionPolicy:
    """Raw-history requirement. ``require`` raises when the result mode cannot serve the operation."""

    @classmethod
    def raw_history(cls) -> RetentionPolicy: ...
    @classmethod
    def bounded_window(cls, max_rows: int) -> RetentionPolicy: ...
    @classmethod
    def sufficient_statistics(cls) -> RetentionPolicy: ...
    def kind(self) -> str: ...
    def result_mode(self) -> RetentionMode: ...
    def max_retained_rows(self) -> int | None: ...
    def no_longer_serves(self) -> list[RetainedOperation]: ...
    def require(self, operation: RetainedOperation) -> None: ...
    def refusal(self, operation: RetainedOperation) -> RetentionRefusal | None: ...

class RetentionMode:
    """How a result is kept. The default is full reusable state."""

    @classmethod
    def full_reusable(cls) -> RetentionMode: ...
    @classmethod
    def summary_only(cls) -> RetentionMode: ...
    @classmethod
    def disk_backed(cls) -> RetentionMode: ...
    @classmethod
    def default(cls) -> RetentionMode: ...
    def as_str(self) -> str: ...
    def serves(self, operation: RetainedOperation) -> bool: ...
    def no_longer_serves(self) -> list[RetainedOperation]: ...
    def require(self, operation: RetainedOperation) -> None: ...
    def refusal(self, operation: RetainedOperation) -> RetentionRefusal | None: ...

class RetentionRefusal:
    """A reduced retention mode's refusal of one later operation, held as a value."""

    def mode(self) -> RetentionMode: ...
    def operation(self) -> RetainedOperation: ...
    def no_longer_serves(self) -> list[RetainedOperation]: ...
    def message(self) -> str: ...
    def raise_error(self) -> None: ...

class RetainedOperation:
    @classmethod
    def row_reconstruction(cls) -> RetainedOperation: ...
    @classmethod
    def refit_from_retained_rows(cls) -> RetainedOperation: ...
    @classmethod
    def exact_row_eviction(cls) -> RetainedOperation: ...
    @classmethod
    def in_place_column_insert(cls) -> RetainedOperation: ...
    @classmethod
    def in_place_weight_or_mask_update(cls) -> RetainedOperation: ...
    @classmethod
    def reuse_full_state(cls) -> RetainedOperation: ...
    @classmethod
    def reuse_summary(cls) -> RetainedOperation: ...
    @classmethod
    def read_disk_artifact(cls) -> RetainedOperation: ...
    def as_str(self) -> str: ...

class PreservedIdentities:
    def __init__(
        self,
        sample_selection: str,
        folds: str,
        population: str,
        history: str,
        dependence: str,
        uncertainty: str,
    ) -> None: ...
    def identity(self, axis: str) -> str: ...

class FrameRewrite:
    """A declared rewrite. ``check`` refuses a changed scientific frame and does not execute it."""

    @classmethod
    def column_filter_pushdown(
        cls, cohort: PreservedIdentities, pushed: PreservedIdentities, effect: str
    ) -> FrameRewrite: ...
    @classmethod
    def shared_row_design_preparation(
        cls,
        preparation: PreservedIdentities,
        left: PreservedIdentities,
        right: PreservedIdentities,
        effect: str,
    ) -> FrameRewrite: ...
    @classmethod
    def nuisance_reuse(
        cls, fit: PreservedIdentities, consumer: PreservedIdentities, effect: str
    ) -> FrameRewrite: ...
    @classmethod
    def sufficient_statistics(
        cls, rows: PreservedIdentities, summary: PreservedIdentities, effect: str
    ) -> FrameRewrite: ...
    @classmethod
    def tiled_evaluation(
        cls, full: PreservedIdentities, tile: PreservedIdentities, effect: str
    ) -> FrameRewrite: ...
    def name(self) -> str: ...
    def with_matching_frame(
        self, row_ids: list[str], fold_ids: list[str], population: str, history: str
    ) -> FrameRewrite: ...
    def check(self) -> None: ...

class FrequentistSummary:
    @classmethod
    def estimate_covariance(
        cls, name: str, estimate: float, covariance: float
    ) -> FrequentistSummary: ...
    @classmethod
    def interval(cls, name: str) -> FrequentistSummary: ...
    def kind(self) -> str: ...
    def name(self) -> str: ...
    def as_bayesian_likelihood(self) -> None: ...

class SummaryObservationModel:
    """Covariance adapted with sampling approximation, selection, dependence, and origin."""

    @classmethod
    def try_adapt(
        cls,
        summary: FrequentistSummary,
        sampling_approximation: str,
        selection: str,
        dependence: str,
        origin: str,
    ) -> SummaryObservationModel: ...
    def summary_kind(self) -> str: ...
    def summary_name(self) -> str: ...
    def sampling_approximation(self) -> str: ...
    def selection(self) -> str: ...
    def dependence(self) -> str: ...
    def origin(self) -> str: ...
    def validate_use(self, requested: str) -> LikelihoodUpdateAdmission: ...

class LikelihoodUpdateAdmission:
    def summary_kind(self) -> str: ...
    def summary_name(self) -> str: ...

class FeatureVersion:
    def __init__(self, name: str, version: int) -> None: ...
    def name(self) -> str: ...
    def version(self) -> int: ...

class FeatureRead:
    """Container format ``0.5`` is compatible. Unknown features and package SemVer are refused."""

    def kind(self) -> str: ...
    def name(self) -> str: ...
    def version(self) -> int: ...
    def source_major(self) -> int | None: ...
    def source_minor(self) -> int | None: ...
    @classmethod
    def classify(
        cls,
        container_major: int,
        container_minor: int,
        name: str,
        version: int,
        accepted: list[FeatureVersion],
    ) -> FeatureRead: ...
    @classmethod
    def classify_package(
        cls,
        container_major: int,
        container_minor: int,
        major: int,
        minor: int,
        patch: int,
        accepted: list[FeatureVersion],
    ) -> FeatureRead: ...

class ExecutionCounts:
    """Counts a run actually performed. ``try_add`` refuses a sum that does not fit in ``u64``."""

    def __init__(self, scans: int, copies: int, fits: int, tests: int, cache_hits: int) -> None: ...
    def scans(self) -> int: ...
    def copies(self) -> int: ...
    def fits(self) -> int: ...
    def tests(self) -> int: ...
    def cache_hits(self) -> int: ...
    def try_add(self, extra: ExecutionCounts) -> ExecutionCounts: ...

class ExecutionResources:
    """Threads, memory, and spill bytes. Spill is a count, not a path."""

    def __init__(self, threads: int, memory_bytes: int, spill_bytes: int) -> None: ...
    def threads(self) -> int: ...
    def memory_bytes(self) -> int: ...
    def spill_bytes(self) -> int: ...
    def try_add(self, extra: ExecutionResources) -> ExecutionResources: ...

class NumericalReplay:
    """Recorded measurements. Reading them does not re-execute the run or invoke a callback."""

    def counts(self) -> ExecutionCounts: ...
    def resources(self) -> ExecutionResources: ...
    def retention(self) -> str: ...
    def backend_name(self) -> str: ...
    def kernel(self) -> str: ...
    def status(self) -> str: ...
    def contract_changed(self) -> bool: ...

class ExecutionFactStanding:
    """Physical integrity, stored scientific digest, and numerical record."""

    def integrity(self) -> bytes: ...
    def semantic_verification(self) -> bytes | None: ...
    def numerical_replay(self) -> NumericalReplay: ...

class ExecutionFacts:
    """Measured execution-receipt facts.

    Physical identity is the measurements. The scientific digest is stored beside
    them and is not recomputed by ``rebuild``. The numerical-policy flag changes
    only through ``declare_numerical_policy_changes_scientific_contract``.
    """

    @property
    def is_native(self) -> bool: ...
    @property
    def observed_scans(self) -> int | None: ...
    @property
    def observed_copies(self) -> int | None: ...
    @property
    def observed_copied_bytes(self) -> int | None: ...
    @property
    def observed_retained_bytes(self) -> int | None: ...
    @property
    def observed_model_fits(self) -> int | None: ...
    @property
    def peak_admitted_bytes(self) -> int | None: ...
    @property
    def peak_admitted_threads(self) -> int | None: ...
    @property
    def observed_spill_bytes(self) -> int | None: ...
    @property
    def observed_allocations(self) -> int | None: ...
    @property
    def observed_allocated_bytes(self) -> int | None: ...
    def accounting(
        self,
        planned_peak_memory_bytes: int | None = None,
        planned_copy_bytes: int | None = None,
        planned_workspace_bytes: int | None = None,
        planned_worker_threads: int | None = None,
    ) -> list[tuple[str, str, int | None]]: ...
    def __init__(
        self,
        counts: ExecutionCounts,
        retention: str,
        resources: ExecutionResources,
        backend_name: str,
        kernel: str,
        status: str,
        scientific_identity: bytes | None,
        numerical_policy_changes_scientific_contract: bool,
    ) -> None: ...
    def counts(self) -> ExecutionCounts: ...
    def retention(self) -> str: ...
    def resources(self) -> ExecutionResources: ...
    def backend_name(self) -> str: ...
    def kernel(self) -> str: ...
    def status(self) -> str: ...
    def scientific_identity(self) -> bytes | None: ...
    def scientific_contract_changed(self) -> bool: ...
    def publishes_claim(self) -> bool: ...
    def physical_identity(self) -> bytes: ...
    def try_add_counts(self, extra: ExecutionCounts) -> None: ...
    def try_add_resources(self, extra: ExecutionResources) -> None: ...
    def rebuild(
        self,
        counts: ExecutionCounts,
        retention: str,
        resources: ExecutionResources,
        backend_name: str,
        kernel: str,
        status: str,
    ) -> ExecutionFacts: ...
    def with_scientific_identity(self, scientific_identity: bytes | None) -> ExecutionFacts: ...
    def declare_numerical_policy_changes_scientific_contract(
        self, changes: bool
    ) -> ExecutionFacts: ...
    def standing(self) -> ExecutionFactStanding: ...

class InspectionStanding:
    """Scientific identity of the target copied into the inspection."""

    def semantic_verification(self) -> bytes: ...

class InspectionInputs:
    """Held inspection parts. Absence is not an empty list or a default policy.

    ``inspect`` copies those parts. It takes no callback and does not fit, scan,
    or simulate.
    """

    def __init__(self) -> None: ...
    def with_assumptions(self, names: list[str]) -> InspectionInputs: ...
    def with_realization(self, name: str) -> InspectionInputs: ...
    def with_evidence(self, names: list[str]) -> InspectionInputs: ...
    def with_uncertainty(self, joint: bool, level: float) -> InspectionInputs: ...
    def with_reuse(self, name: str, frozen_fit: bool) -> InspectionInputs: ...
    def with_materialization(self, name: str, copy_bytes: int) -> InspectionInputs: ...
    def with_resources(
        self, memory_limit_bytes: int | None, estimated_bytes: int
    ) -> InspectionInputs: ...
    def with_backend(self, kernel: str) -> InspectionInputs: ...
    def with_read_operators(
        self, operators: list[tuple[int, int, int, int]]
    ) -> InspectionInputs: ...
    def inspect(self) -> PlanInspection: ...

class PlanInspection:
    """Plan record copied from values the caller already held."""

    def target_identity(self) -> bytes: ...
    def target_inference(self) -> str: ...
    def assumptions(self) -> list[str]: ...
    def realization(self) -> str: ...
    def evidence(self) -> list[str]: ...
    def uncertainty_joint(self) -> bool: ...
    def uncertainty_level(self) -> float: ...
    def uncertainty_level_bits(self) -> int: ...
    def reuse_name(self) -> str: ...
    def reuse_frozen_fit(self) -> bool: ...
    def materialization_name(self) -> str: ...
    def materialization_copy_bytes(self) -> int: ...
    def memory_limit_bytes(self) -> int | None: ...
    def estimated_bytes(self) -> int: ...
    def backend(self) -> str: ...
    def operators(self) -> list[tuple[str, list[int], int | None, int | None, int | None]]: ...
    def standing(self) -> InspectionStanding: ...

def require_ordinary_temporal_curve_posterior_native(serial_error_model: str) -> None: ...
def trajectory_contrast_native(
    subjects: int,
    periods: int,
    treatment: list[bool],
    event: list[bool],
    at_risk: list[bool],
    treatment_probability: list[float],
    censoring_weight: list[float],
    minimum_probability: float,
    active: list[bool],
    control: list[bool],
) -> tuple[float, float]: ...
def trajectory_bootstrap_native(
    subjects: int,
    periods: int,
    treatment: list[bool],
    event: list[bool],
    at_risk: list[bool],
    treatment_probability: list[float],
    censoring_weight: list[float],
    minimum_probability: float,
    active: list[bool],
    control: list[bool],
    replicates: int,
    seed: int,
) -> tuple[float, float, float, float, int, int]: ...
def trajectory_credible_interval_native(
    subjects: int,
    periods: int,
    treatment: list[bool],
    event: list[bool],
    at_risk: list[bool],
    treatment_probability: list[float],
    censoring_weight: list[float],
    minimum_probability: float,
    active: list[bool],
    control: list[bool],
) -> tuple[float, float]: ...
def execute_cyclic_system(
    request_json: str, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
) -> tuple[str, bytes, ExecutionFacts]: ...
def consume_cyclic_system(
    bytes: bytes,
    expected_identity: str,
    *,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str, ExecutionFacts]: ...
def cyclic_parameter_posterior(
    request_json: str,
    prior_sd: float,
    chains: int,
    warmup: int,
    draws: int,
    proposal_scale: float,
    seed: int,
    *,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def execute_cyclic_family(
    request_json: str, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
) -> tuple[str, bytes, ExecutionFacts]: ...
def consume_cyclic_family(
    bytes: bytes,
    expected_identity: str,
    *,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[str, ExecutionFacts]: ...

class _PreparedCyclicSystem:
    def execute(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> tuple[str, bytes, ExecutionFacts]: ...
    def execute_buffers(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> tuple[str, bytes, NDArray[np.float64], NDArray[np.float64], int, ExecutionFacts]: ...
    def request_json(self) -> str: ...
    def initial_state(self) -> list[float]: ...
    def inspect(self) -> str: ...
    def retarget(
        self, worlds_json: str, noise_json: str, initial_json: str
    ) -> _PreparedCyclicSystem: ...
    def retarget_buffers(
        self, worlds_json: str, noise: NDArray[np.float64], initial: list[float]
    ) -> _PreparedCyclicSystem: ...
    def refresh_buffers(
        self,
        rows_json: str,
        observations: NDArray[np.float64],
        *,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _PreparedCyclicSystem: ...
    def to_family(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> _PreparedCyclicFamily: ...
    def contract(self) -> str: ...

def prepare_cyclic_system(
    request_json: str, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
) -> _PreparedCyclicSystem: ...
def prepare_cyclic_system_buffers(
    metadata: str,
    noise: NDArray[np.float64],
    observations: NDArray[np.float64],
    *,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedCyclicSystem: ...

class _PreparedCyclicFamily:
    def rebind(
        self,
        request_json: str,
        *,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _PreparedCyclicFamily: ...
    def evidence(
        self,
        schema_json: str,
        selectors_json: str,
        *,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _CyclicEvidence: ...
    def execute(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> tuple[str, bytes, ExecutionFacts]: ...
    def interventions(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> tuple[str, ExecutionFacts]: ...
    def inspect(self) -> str: ...
    def contract(self) -> str: ...

def prepare_cyclic_family(
    request_json: str, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
) -> _PreparedCyclicFamily: ...
def prepare_cyclic_experimental_recovery(
    request_json: str, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
) -> _PreparedCyclicFamily: ...
def execute_local_cyclic(
    request_json: str, callbacks: list[Any], *, max_bytes: int | None = None, cancel: Any = None
) -> str: ...

class _CyclicEvidence:
    def report(self) -> str: ...
    def export(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> bytes: ...
    def joint_law(
        self, *, max_bytes: int | None = None, cancel: CancellationToken | None = None
    ) -> JointDistributionArtifact: ...
    def decide_means(self, contract_json: str) -> str: ...
    def decide_enclosures(self, contract_json: str) -> str: ...

def consume_cyclic_evidence(
    bytes: bytes,
    expected_identity: str,
    *,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _CyclicEvidence: ...

class _JointPanelResult:
    @property
    def execution_facts(self) -> ExecutionFacts: ...
    def experiment_quantity(self, functional_json: str) -> str: ...
    def condition_experiment(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _JointPanelExperiment: ...
    def report(self) -> str: ...
    def scheduled_contrast(self, functional_json: str, level: float) -> str: ...
    def certify_program(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _PanelProgramCertificate: ...
    def search_program(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def decompose_reference(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def export(
        self,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...

class _PreparedJointPanel:
    def inspect(self) -> str: ...
    def estimate(
        self,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _JointPanelResult: ...
    def retarget(
        self,
        worlds_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _PreparedJointPanel: ...

def prepare_joint_panel(
    declaration_json: str,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedJointPanel: ...
def consume_joint_panel(
    bytes: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def consume_panel_program_certificate(
    bytes: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class _PanelProgramCertificate:
    def report(self) -> str: ...
    def export(
        self,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

class _PreparedRandomizedProgramSelection:
    def execute(
        self,
        *,
        repetitions: int = 199,
        tie_tolerance: float = 0.0,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _RandomizedProgramSelection: ...
    def simultaneous_intervals(
        self,
        *,
        repetitions: int = 199,
        level: float = 0.95,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...

class _RandomizedProgramSelection:
    def report(self) -> str: ...
    def export(self) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

class _InstrumentedProgramSelection:
    def report(self) -> str: ...
    def export(self) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

def prepare_randomized_program_selection(
    declaration_json: str,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedRandomizedProgramSelection: ...
def consume_randomized_program_selection(
    bytes: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def select_instrumented_program(
    declaration_json: str,
    *,
    repetitions: int = 199,
    tie_tolerance: float = 0.0,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _InstrumentedProgramSelection: ...
def consume_instrumented_selection(
    bytes: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class _JointPanelExperiment:
    def report(self, ledger: EvidenceLedgerHandle) -> str: ...
    def decide(self, contract_json: str, ledger: EvidenceLedgerHandle) -> str: ...
    def rank_next_studies(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        replicates: int = 199,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def rank_next_samples(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        replicates: int = 199,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def export(
        self,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

class _JointDesignExperiment:
    def report(self, ledger: EvidenceLedgerHandle) -> str: ...
    def decide(self, contract_json: str, ledger: EvidenceLedgerHandle) -> str: ...
    def rank_next_samples(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        replicates: int = 199,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def export(
        self,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

def joint_panel_experiment_acceptance_matrix() -> str: ...

class _PreparedJointDesign:
    def proof(self) -> str: ...
    def execute(
        self,
        inference_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _JointDesignResult: ...
    def consume_current(
        self,
        artifact: bytes,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _JointDesignResult: ...

class _JointDesignResult:
    def export_reference_decomposition(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def consume_reference_decomposition_current(
        self,
        artifact: bytes,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def report(self) -> str: ...
    def posterior_quantities(self) -> str: ...
    def export(self) -> bytes: ...
    def experiment_quantity(self, functional_json: str) -> str: ...
    def condition_experiment(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _JointDesignExperiment: ...
    def decompose_population_reference(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def export_population_reference(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def consume_population_reference_current(
        self,
        artifact: bytes,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def decompose_reference_components(
        self,
        declaration_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...

class _PreparedNativeJointDesign:
    def identification(self) -> str: ...
    def execute(
        self,
        inference_json: str,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _NativeJointDesignResult: ...
    def consume_current(
        self,
        artifact: bytes,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _NativeJointDesignResult: ...

class _NativeJointDesignResult:
    def experiment_quantity(self, binding_json: str) -> str: ...
    def condition_native_experiment(
        self,
        binding_json: str,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> _NativeJointExperiment: ...
    def report(self) -> str: ...
    def export(self) -> bytes: ...

def prepare_joint_design(
    target_json: str,
    proof_json: str,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedJointDesign: ...
def prepare_joint_design_for_panel(
    panel_result: _JointPanelResult,
    proof_json: str,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedJointDesign: ...
def replay_joint_design(
    artifact: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def prepare_native_joint_design(
    recipe_json: str,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> _PreparedNativeJointDesign: ...
def replay_native_joint_design(
    artifact: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def replay_population_reference(
    artifact: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...

class _NativeJointExperiment:
    def report(self, ledger: EvidenceLedgerHandle) -> str: ...
    def decide(self, contract_json: str, ledger: EvidenceLedgerHandle) -> str: ...
    def rank_next_samples(
        self,
        declaration_json: str,
        ledger: EvidenceLedgerHandle,
        *,
        replicates: int = 199,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> str: ...
    def export(
        self,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> bytes: ...
    def verify_current(
        self,
        bytes: bytes,
        ledger: EvidenceLedgerHandle,
        *,
        seed: int = 0,
        threads: int = 1,
        max_bytes: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> None: ...

def replay_reference_decomposition(
    artifact: bytes,
    *,
    seed: int = 0,
    threads: int = 1,
    max_bytes: int | None = None,
    cancel: CancellationToken | None = None,
) -> str: ...
def current_score_row_digest_bytes(row_ids: list[int]) -> bytes: ...
def narrow_score_row_digest_bytes(row_ids: list[int]) -> bytes: ...
def declared_target_digest(level: float) -> bytes: ...
def consume_previous_release(
    artifact: bytes, level: float
) -> tuple[bytes, bytes, str, list[tuple[str, str]]]: ...
def consume_previous_routes(
    artifact: bytes, methods: list[tuple[str, str, str, str, str, str]]
) -> str: ...
