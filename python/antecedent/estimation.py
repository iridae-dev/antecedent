"""High-level estimation entry points."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, cast

from ._api import describe_refusal
from ._coerce import coerce_latency, coerce_query, coerce_refute
from ._data import as_columns, ingest_columns, try_as_arrow_c_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
)
from ._native import (
    AteAnalysisResult,
    MediationEffectsSummary,
    mediation_effects_summary,
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
from ._native import (
    identify_ate_admg as _identify_ate_admg,
)
from ._native import (
    prepare_ate_batch as _prepare_ate_batch,
)
from ._native import (
    prepare_cells_batch as _prepare_cells_batch,
)
from .discovery import (
    DbnPosterior,
    ExactDagPosterior,
    GraphPosterior,
    cpdag_oriented_edges,
)
from .errors import (
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import (
    Bayesian,
    ClassPrior,
    Frequentist,
    _class_prior_kwargs,
    _max_completions_kwargs,
)
from .query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .results import (
    AnalysisResult,
    CausalResponseView,
    ConflictSummaryView,
    DistributionAtomView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    ProbabilityIntervalView,
    ReasoningSlots,
    RefutationReport,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
    TemporalMediationGridView,
    TemporalMediationSliceView,
    ValidationView,
)
from .results.response import SupportStatus, UncertaintyKind

# Preferred name for the native temporal DTO.
NativeAnalysisResult = TemporalAnalysisResult


def _refutation_reports_from_raw(validation: Any) -> list[RefutationReport]:
    """One :class:`RefutationReport` per entry in a nested ``validation.reports``."""
    return [
        RefutationReport(
            refuter=r.refuter,
            original_ate=r.original_ate,
            refuted_ate=r.refuted_ate,
            comparison=r.comparison,
            informative=r.informative,
            passed=r.passed,
            failure_condition=r.failure_condition,
            replicates=r.replicates,
        )
        for r in getattr(validation, "reports", None) or ()
    ]


def _plan_from_raw(raw: Any) -> PlanView:
    return PlanView(
        plan_id=str(getattr(raw, "plan_id", "") or ""),
        modality=getattr(raw, "modality", None),
        discovery_algorithm=getattr(raw, "discovery_algorithm", None),
        structure_source=getattr(raw, "structure_source", None),
        graph_review_required=bool(getattr(raw, "graph_review_required", False)),
        identifier=getattr(raw, "plan_identifier", None),
        estimator=getattr(raw, "plan_estimator", None)
        or (getattr(raw, "estimator_id", None) or None),
        validation_suite=getattr(raw, "validation_suite", None),
    )


# --- Nested-section resolution, with a flat-field fallback ------------------------
#
# Every real native DTO now carries `identification`/`estimate`/`posterior`/
# `validation`/`performance` (see `antecedent._native`), so `_wrap_ate` reads
# those directly in the common case. The `_section_*` helpers below exist only
# for test doubles that pre-date the nested sections (e.g. the `SimpleNamespace`
# stand-in in `test_wrap_temporal_refutation.py`, which exercises the
# ran-but-failing-refuter aggregation bug fix against a minimal object exposing
# only the historical flat attributes): when a raw object has no `.identification`
# etc., these reconstruct an equivalent section from the flat fields it does have,
# so `_wrap_ate` never needs an `isinstance`/shape check of its own.
def _section_identification(raw: Any) -> Any:
    sec = getattr(raw, "identification", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        status=getattr(raw, "identification_status", "") or "",
        method=getattr(raw, "method", "") or "",
        adjustment_set=list(getattr(raw, "adjustment_set", None) or []),
        assumption_count=int(getattr(raw, "assumption_count", 0) or 0),
        derivation_step_count=int(getattr(raw, "derivation_step_count", 0) or 0),
    )


def _optional_finite_ate(value: Any) -> float | None:
    """Omit non-finite sentinels so function-valued results have no scalar ate."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _section_estimate(raw: Any) -> Any:
    sec = getattr(raw, "estimate", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        ate=_optional_finite_ate(getattr(raw, "ate", None)),
        se_analytic=raw.se_analytic,
        se_bootstrap=raw.se_bootstrap,
        estimator_id=str(getattr(raw, "estimator_id", "") or ""),
        method=getattr(raw, "method", "") or "",
        overlap_ess=getattr(raw, "overlap_ess", None),
        overlap_propensity_min=getattr(raw, "overlap_propensity_min", None),
        functional_means=getattr(raw, "functional_means", None),
        exceedance_cdf=getattr(raw, "exceedance_cdf", None),
        monotone_rearranged=bool(getattr(raw, "monotone_rearranged", False)),
        interaction_structurally_zero=getattr(raw, "interaction_structurally_zero", None),
        unit_effects_homogeneous=getattr(raw, "unit_effects_homogeneous", None),
        score_table=getattr(raw, "score_table", None),
        simultaneous_interval=getattr(raw, "simultaneous_interval", None),
        adjusted_p_values=getattr(raw, "adjusted_p_values", None),
        family_contrast=getattr(raw, "family_contrast", None),
        family_contrast_interval=getattr(raw, "family_contrast_interval", None),
        candidate_selection=getattr(raw, "candidate_selection", None),
        evalue=getattr(raw, "evalue", None),
        evalue_threshold=getattr(raw, "evalue_threshold", None),
        joint_covariance=getattr(raw, "joint_covariance", None),
        score_inference=getattr(raw, "score_inference", None),
        scenario_effects=getattr(raw, "scenario_effects", None),
        scenario_intervals=getattr(raw, "scenario_intervals", None),
    )


def _section_posterior(raw: Any) -> Any:
    sec = getattr(raw, "posterior", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        effect_mean=getattr(raw, "posterior_effect_mean", None),
        effect_sd=getattr(raw, "posterior_effect_sd", None),
        q025=getattr(raw, "posterior_q025", None),
        q975=getattr(raw, "posterior_q975", None),
        n_draws=getattr(raw, "posterior_n_draws", None),
        p_below_zero=getattr(raw, "posterior_p_below_zero", None),
        backend=getattr(raw, "posterior_backend", None),
        artifact=getattr(raw, "posterior_artifact", None),
        unidentified_mass=getattr(raw, "posterior_unidentified_mass", None),
        subsampled_out_mass=getattr(raw, "posterior_subsampled_out_mass", None),
    )


def _section_validation(raw: Any) -> Any:
    sec = getattr(raw, "validation", None)
    if sec is not None:
        return sec
    # Mirror the shared Rust aggregate rule (see `ValidationSection::from_reports`
    # in `python/src/lib.rs`): never claim pass when nothing ran.
    reports = list(getattr(raw, "refutations", None) or ())
    ran = len(reports) > 0
    passed = ran and all(r.passed for r in reports)
    return SimpleNamespace(
        passed=passed,
        ran=ran,
        count=int(getattr(raw, "refutation_count", len(reports)) or 0),
        reports=reports,
    )


def _section_performance(raw: Any) -> Any:
    sec = getattr(raw, "performance", None)
    if sec is not None:
        return sec
    return SimpleNamespace(
        plan_id=getattr(raw, "plan_id", "") or "",
        modality=getattr(raw, "modality", "") or "",
        peak_memory_bytes=getattr(raw, "peak_memory_bytes", None),
        latency_mode=getattr(raw, "latency_mode", None),
        wall_time_ns=getattr(raw, "wall_time_ns", None),
        bootstrap_replicates_requested=getattr(raw, "bootstrap_replicates_requested", None),
        bootstrap_replicates_ok=getattr(raw, "bootstrap_replicates_ok", None),
        n_draws=getattr(raw, "n_draws_effort", None),
        cancelled=bool(getattr(raw, "cancelled", False)),
        early_stopped=bool(getattr(raw, "early_stopped", False)),
        stage_timings=getattr(raw, "stage_timings", None),
        bytes_borrowed=getattr(raw, "bytes_borrowed", None),
    )


def _probability_interval_from_raw(raw: Any) -> ProbabilityIntervalView | None:
    """Native ``ProbabilityIntervalSection`` → view (``None`` passes through)."""
    if raw is None:
        return None
    return ProbabilityIntervalView(
        level=raw.level,
        lower=raw.lower,
        upper=raw.upper,
        unavailable=raw.unavailable,
    )


def _unit_effect_intervals_from_raw(raw: Any) -> list[tuple[float, float]] | None:
    """Native per-unit ``(lower, upper)`` pairs → tuples (``None`` passes through)."""
    pairs = getattr(raw, "unit_effect_intervals", None)
    if pairs is None:
        return None
    return [(float(lower), float(upper)) for lower, upper in pairs]


def _distribution_atoms_from_raw(sec_estimate: Any) -> tuple[DistributionAtomView, ...] | None:
    """Native distribution atoms with their bounded probability intervals."""
    atoms = getattr(sec_estimate, "distribution_atoms", None)
    if atoms is None:
        return None
    return tuple(
        DistributionAtomView(
            outcomes=tuple((str(name), value) for name, value in atom.outcomes),
            conditioning=tuple((str(name), value) for name, value in atom.conditioning),
            probability=float(atom.probability),
            se_bootstrap=atom.se_bootstrap,
            interval=_probability_interval_from_raw(atom.interval),
        )
        for atom in atoms
    )


def _wrap_ate(
    raw: AteAnalysisResult | TemporalAnalysisResult,
    prepared: Any | None = None,
    *,
    query: Any = None,
) -> AnalysisResult:
    """Build the nested :class:`AnalysisResult` view from either native DTO.

    ``AteAnalysisResult`` (static) and ``AnalysisResult`` (temporal, aliased here
    as ``TemporalAnalysisResult``) both expose the same five nested sections —
    ``identification`` / ``estimate`` / ``posterior`` / ``validation`` /
    ``performance`` (see ``antecedent._native`` and the doc comments there) —
    built and kept in sync with their flat-field siblings on the Rust side. This
    one function reads those sections instead of the ~30 hand-written
    ``getattr(raw, "field_name", default)`` calls per DTO shape that used to live
    in two near-duplicate functions (``_wrap_ate`` / ``_wrap_temporal``). Fields
    the temporal DTO genuinely cannot supply already come through their section
    as ``None`` (see each section's doc comment in ``_native.pyi``), so no
    per-DTO branching is needed here beyond `getattr` presence gates for the
    handful of fields that only ever exist on the static DTO (predictive checks,
    prior sensitivity, external-prior conflict, mediation) — those `getattr`
    calls resolve to ``None`` on the temporal DTO exactly as the old
    `_wrap_temporal` left them.
    """
    execution = None if prepared is None else prepared._native.snapshot()
    slots = (
        None if execution is None else ReasoningSlots.from_contract(execution.execution_contract())
    )

    def _conflict_from_raw(r: Any) -> ConflictSummaryView | None:
        ids = getattr(r, "conflict_source_ids", None)
        if ids is None:
            return None
        return ConflictSummaryView(
            source_ids=list(ids),
            alphas_requested=list(getattr(r, "conflict_alphas_requested", None) or []),
            alphas_applied=list(getattr(r, "conflict_alphas_applied", None) or []),
        )

    sec_identification = _section_identification(raw)
    sec_estimate = _section_estimate(raw)
    sec_posterior = _section_posterior(raw)
    sec_validation = _section_validation(raw)
    sec_performance = _section_performance(raw)

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
    if sec_posterior.n_draws is not None:
        mass = sec_posterior.unidentified_mass
        skipped = float(getattr(sec_posterior, "subsampled_out_mass", None) or 0.0)
        envelope = None
        if (mass is not None and float(mass) > 0.0) or skipped > 0.0:
            envelope = EffectEnvelope(
                effect_mean=sec_posterior.effect_mean,
                effect_sd=sec_posterior.effect_sd,
                q025=sec_posterior.q025,
                q975=sec_posterior.q975,
                unidentified_mass=float(mass or 0.0),
                n_draws=sec_posterior.n_draws,
                backend=sec_posterior.backend,
                subsampled_out_mass=skipped,
            )
        posterior = PosteriorView(
            effect_mean=sec_posterior.effect_mean,
            effect_sd=sec_posterior.effect_sd,
            q025=sec_posterior.q025,
            q975=sec_posterior.q975,
            n_draws=sec_posterior.n_draws,
            p_below_zero=sec_posterior.p_below_zero,
            backend=sec_posterior.backend,
            artifact=sec_posterior.artifact,
            unidentified_mass=None if mass is None else float(mass),
            envelope=envelope,
            conflict=_conflict_from_raw(raw),
            subsampled_out_mass=skipped,
        )

    # The predictive-check fields are static-DTO only; the temporal DTO does not
    # declare them, so every read goes through `getattr` rather than asserting a
    # union member that may not have the attribute at all.
    def _ppc(prefix: str, kind: str) -> PredictiveCheckReport | None:
        p_value = getattr(raw, f"{prefix}_p_value", None)
        if p_value is None:
            return None
        observed = getattr(raw, f"{prefix}_observed", None)
        predictive_mean = getattr(raw, f"{prefix}_predictive_mean", None)
        predictive_sd = getattr(raw, f"{prefix}_predictive_sd", None)
        n_sims = getattr(raw, f"{prefix}_n_sims", None)
        if observed is None or predictive_mean is None:
            return None
        if predictive_sd is None or n_sims is None:
            return None
        return PredictiveCheckReport(
            kind=kind,
            observed=float(observed),
            predictive_mean=float(predictive_mean),
            predictive_sd=float(predictive_sd),
            p_value=float(p_value),
            n_sims=int(n_sims),
        )

    prior_predictive = _ppc("prior_ppc", "prior_predictive")
    posterior_predictive = _ppc("posterior_ppc", "posterior_predictive")
    prior_sensitivity = None
    means = getattr(raw, "prior_sensitivity_means", None)
    if means is not None:
        alphas_raw = getattr(raw, "prior_sensitivity_alphas", None)
        scales_raw = getattr(raw, "prior_sensitivity_scales", None)
        sds = getattr(raw, "prior_sensitivity_sds", None)
        multipliers_raw = getattr(raw, "prior_sensitivity_variance_multipliers", None)
        family = getattr(raw, "prior_sensitivity_family", None)
        prior_sensitivity = PriorSensitivityReport(
            scales=list(scales_raw or ()),
            effect_means=list(means),
            effect_sds=list(sds or ()),
            alphas=None if alphas_raw is None else list(alphas_raw),
            variance_multipliers=None if multipliers_raw is None else list(multipliers_raw),
            family=str(family) if family is not None else "isotropic_scale",
        )
    certificate_json = getattr(raw, "certificate_json", None)
    mediation_grid = None
    horizons = list(getattr(raw, "mediation_horizons", None) or ())
    if horizons:
        temporal_raw = cast(TemporalAnalysisResult, raw)
        effects = list(temporal_raw.mediation_effects)
        totals = list(temporal_raw.mediation_totals)
        directs = list(temporal_raw.mediation_directs)
        mediated_effects = list(temporal_raw.mediation_mediated_effects)
        statuses = list(temporal_raw.mediation_identification_statuses)
        methods = list(temporal_raw.mediation_methods)
        adjustments = list(temporal_raw.mediation_adjustments)
        uncertainty_kinds = list(temporal_raw.mediation_uncertainty_kinds)
        standard_deviations = list(temporal_raw.mediation_standard_deviations)
        q025 = list(temporal_raw.mediation_q025)
        q975 = list(temporal_raw.mediation_q975)
        identified_lower = list(temporal_raw.mediation_identified_lower)
        identified_upper = list(temporal_raw.mediation_identified_upper)
        slices = tuple(
            TemporalMediationSliceView(
                horizon=int(values[0]),
                effect=float(values[1]),
                total=float(values[2]),
                direct=float(values[3]),
                mediated=float(values[4]),
                identification_status=str(values[5]),
                method=str(values[6]),
                adjustment=tuple((int(variable), int(offset)) for variable, offset in values[7]),
                uncertainty_kind=str(values[8]),
                standard_deviation=None if values[9] is None else float(values[9]),
                q025=None if values[10] is None else float(values[10]),
                q975=None if values[11] is None else float(values[11]),
                identified_lower=None if values[12] is None else float(values[12]),
                identified_upper=None if values[13] is None else float(values[13]),
            )
            for values in zip(
                horizons,
                effects,
                totals,
                directs,
                mediated_effects,
                statuses,
                methods,
                adjustments,
                uncertainty_kinds,
                standard_deviations,
                q025,
                q975,
                identified_lower,
                identified_upper,
                strict=True,
            )
        )
        mediation_grid = TemporalMediationGridView(
            slices=slices,
            joint_posterior=bool(getattr(raw, "mediation_joint_posterior", False)),
        )
    return AnalysisResult(
        certificate=json.loads(certificate_json) if certificate_json else None,
        query=query if query is not None else getattr(prepared, "_query", None),
        identification=IdentificationView(
            status=sec_identification.status,
            method=sec_identification.method,
            adjustment_set=list(sec_identification.adjustment_set),
            assumption_count=sec_identification.assumption_count,
            derivation_step_count=sec_identification.derivation_step_count,
        ),
        estimate=EstimateView(
            ate=_optional_finite_ate(getattr(sec_estimate, "ate", None)),
            se_analytic=sec_estimate.se_analytic,
            se_bootstrap=sec_estimate.se_bootstrap,
            estimator_id=sec_estimate.estimator_id,
            method=sec_estimate.method,
            overlap_ess=sec_estimate.overlap_ess,
            overlap_propensity_min=sec_estimate.overlap_propensity_min,
            mediation=mediation,
            functional_means=tuple(sec_estimate.functional_means)
            if getattr(sec_estimate, "functional_means", None) is not None
            else None,
            exceedance_cdf=tuple(sec_estimate.exceedance_cdf)
            if getattr(sec_estimate, "exceedance_cdf", None) is not None
            else None,
            monotone_rearranged=bool(getattr(sec_estimate, "monotone_rearranged", False)),
            interaction_structurally_zero=getattr(
                sec_estimate, "interaction_structurally_zero", None
            ),
            unit_effects_homogeneous=getattr(sec_estimate, "unit_effects_homogeneous", None),
            score_table=getattr(sec_estimate, "score_table", None),
            joint_covariance=getattr(sec_estimate, "joint_covariance", None),
            score_inference=getattr(sec_estimate, "score_inference", None),
            scenario_effects=getattr(sec_estimate, "scenario_effects", None),
            scenario_intervals=getattr(sec_estimate, "scenario_intervals", None),
            simultaneous_interval=getattr(sec_estimate, "simultaneous_interval", None),
            adjusted_p_values=getattr(sec_estimate, "adjusted_p_values", None),
            family_contrast=getattr(sec_estimate, "family_contrast", None),
            family_contrast_interval=getattr(sec_estimate, "family_contrast_interval", None),
            candidate_selection=getattr(sec_estimate, "candidate_selection", None),
            evalue=getattr(sec_estimate, "evalue", None),
            evalue_threshold=getattr(sec_estimate, "evalue_threshold", None),
            distribution=_distribution_atoms_from_raw(sec_estimate),
            mean_interval=_probability_interval_from_raw(
                getattr(sec_estimate, "mean_interval", None)
            ),
        ),
        posterior=posterior,
        unit_effects=getattr(raw, "unit_effects", None),
        unit_effect_intervals=_unit_effect_intervals_from_raw(raw),
        unit_effect_intervals_level=getattr(raw, "unit_effect_intervals_level", None),
        unit_effect_intervals_method=getattr(raw, "unit_effect_intervals_method", None),
        unit_extrapolative=getattr(raw, "unit_extrapolative", None),
        assumptions=getattr(raw, "assumptions", None),
        support=getattr(raw, "support_diagnostics", None),
        mediation=mediation,
        mediation_grid=mediation_grid,
        validation=ValidationView(
            passed=sec_validation.passed,
            ran=sec_validation.ran,
            count=sec_validation.count,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
            prior_sensitivity=prior_sensitivity,
            reports=_refutation_reports_from_raw(sec_validation),
            computation_failures=list(getattr(sec_validation, "computation_failures", [])),
        ),
        performance=PerformanceView(
            plan_id=sec_performance.plan_id,
            modality=sec_performance.modality,
            peak_memory_bytes=sec_performance.peak_memory_bytes,
            latency_mode=sec_performance.latency_mode,
            wall_time_ns=sec_performance.wall_time_ns,
            bootstrap_replicates_requested=sec_performance.bootstrap_replicates_requested,
            bootstrap_replicates_ok=sec_performance.bootstrap_replicates_ok,
            n_draws=sec_performance.n_draws,
            cancelled=bool(sec_performance.cancelled),
            early_stopped=bool(sec_performance.early_stopped),
            stage_timings={str(k): int(v) for k, v in (sec_performance.stage_timings or [])}
            or None,
            bytes_borrowed=getattr(sec_performance, "bytes_borrowed", None),
        ),
        diagnostics=list(raw.diagnostics),
        provenance={
            "node_count": raw.provenance_node_count,
            "worker_threads": getattr(raw, "worker_threads", None),
            "expected_python_crossings": getattr(raw, "expected_python_crossings", None),
        },
        plan=_plan_from_raw(raw),
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        structural_weight_basis=getattr(raw, "structural_weight_basis", None),
        structural_identified_mass=getattr(raw, "structural_identified_mass", None),
        structural_unidentified_mass=getattr(raw, "structural_unidentified_mass", None),
        structural_unevaluable_mass=getattr(raw, "structural_unevaluable_mass", None),
        structural_identified_set=getattr(raw, "structural_identified_set", None),
        structural_identified_set_interval=getattr(raw, "structural_identified_set_interval", None),
        structural_identified_set_interval_level=getattr(
            raw, "structural_identified_set_interval_level", None
        ),
        structural_identified_set_interval_method=getattr(
            raw, "structural_identified_set_interval_method", None
        ),
        structural_identified_set_interval_truncated=getattr(
            raw, "structural_identified_set_interval_truncated", None
        ),
        _raw=raw,
        _prepared=prepared,
        _execution=execution,
        reasoning=slots,
        program_id=None if slots is None else slots.program_id,
        claim_id=None if slots is None else slots.claim_id,
        data_snapshot_id=None if slots is None else slots.data_snapshot_id,
    )


ADMG_DISTRIBUTION_RUST_ONLY = (
    "refused: ADMG InterventionalDistribution (unconditional finite-discrete tables, "
    "validation none) is licensed in the Rust Study API only; the Python "
    "distribution entry points take DAG edges and cannot carry bidirected edges"
)


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise CausalValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, (Pag, TemporalDag, TemporalCpdag, TemporalPag, Admg)):
        raise CausalValueError(
            f"this query reads a fully oriented static Dag; got {type(graph).__name__}"
        )
    if isinstance(graph, Cpdag):
        # PathSpecific / Interventional need a fully oriented DAG; incomplete
        # CPDAGs fail closed with a clear undirected-count message.
        return cpdag_oriented_edges(graph, require_oriented=True)
    return [(str(a), str(b)) for a, b in graph]


_RESPONSE_FAMILY = (
    ResponseCurve,
    InterventionResponse,
    AverageDerivative,
    PointDerivative,
    Elasticity,
    SemiElasticity,
    DirectionalDerivative,
    ResponseJacobian,
)
_ADMG_RESPONSE_REFUSED = (
    "refused: Admg response has no functional plug-in; licensed general-ID "
    "ATE does not estimate a curve."
)
_RESPONSE_CONFIG_KEYS = frozenset(
    {
        "bandwidth",
        "simultaneous_replicates",
        "confidence_level",
        "multiplier_seed",
        "export_row_diagnostics",
    }
)
_RESPONSE_ESTIMATORS = {
    "response_curve": "response.kennedy_dr",
    "point_derivative": "response.kennedy_dr",
    "elasticity": "response.kennedy_dr",
    "semi_elasticity": "response.kennedy_dr",
    "average_derivative": "response.riesz_ade",
    "directional_derivative": "response.gam_derivative",
    "response_jacobian": "response.gam_derivative",
    "intervention_response": "response.intervention_gcomp",
}


def _refuse_admg_response(graph: Any, query: Any) -> None:
    if isinstance(graph, Admg) and isinstance(query, _RESPONSE_FAMILY):
        raise CausalUnsupportedError(_ADMG_RESPONSE_REFUSED)
    if isinstance(graph, Admg) and isinstance(query, ConditionalEffect):
        raise CausalUnsupportedError(
            "refused: ConditionalEffect on Admg has no compile arm; "
            "Dag, Cpdag, and Pag are licensed."
        )


def _check_response_threads(threads: int) -> None:
    if threads != 1:
        raise ValueError("response queries currently require threads=1")


def _check_response_strategy(
    query: Any,
    *,
    graph: Any,
    identifier: str | None,
    estimator: str | None,
    inference: Frequentist | Bayesian | None,
) -> None:
    if (
        isinstance(query, InterventionResponse)
        and estimator == "cell.aipw"
        and not isinstance(inference, Bayesian)
    ):
        if identifier not in (None, "generalized.adjustment"):
            raise ValueError(
                f"{query.kind} requires identifier='generalized.adjustment'; got {identifier!r}"
            )
        return
    expected_identifier = (
        "generalized.adjustment" if isinstance(graph, (Pag, Cpdag)) else "response.backdoor"
    )
    if identifier not in (None, expected_identifier):
        raise ValueError(
            f"{query.kind} requires identifier={expected_identifier!r}; got {identifier!r}"
        )
    expected = _RESPONSE_ESTIMATORS[query.kind]
    allowed: tuple[str | None, ...] = (None, expected)
    if isinstance(query, InterventionResponse) and isinstance(inference, Bayesian):
        allowed = (None, expected, "response.bayesian")
    if estimator not in allowed:
        raise ValueError(f"{query.kind} requires estimator={expected!r}; got {estimator!r}")


def _parse_response_estimator_config(
    query: Any, estimator_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    if estimator_config is None:
        return {}
    unknown = set(estimator_config) - _RESPONSE_CONFIG_KEYS
    if unknown:
        raise ValueError("unknown response estimator_config keys: " + ", ".join(sorted(unknown)))
    options = dict(estimator_config)
    if "simultaneous_replicates" in options and "bandwidth" not in options:
        raise ValueError(
            "simultaneous response bands require an explicit estimator_config bandwidth"
        )
    if isinstance(query, (PointDerivative, Elasticity, SemiElasticity)):
        if "simultaneous_replicates" in options:
            raise ValueError("simultaneous response bands currently apply to ResponseCurve only")
        if "bandwidth" not in options:
            raise ValueError(
                "PointDerivative/Elasticity/SemiElasticity require estimator_config bandwidth"
            )
        extra = set(options) - {"bandwidth"}
        if extra:
            raise ValueError("prepared derivatives accept only bandwidth in estimator_config")
    elif isinstance(query, (AverageDerivative, DirectionalDerivative, ResponseJacobian)):
        extra = set(options) - {"bandwidth"}
        if extra:
            raise ValueError("prepared derivatives accept only bandwidth in estimator_config")
    elif not isinstance(query, ResponseCurve):
        raise ValueError(
            "response estimator_config currently applies to ResponseCurve and point derivatives only"
        )
    return options


def _lagged_edges(
    graph: TemporalDag | Sequence[tuple[str, int, str, int]] | None,
) -> list[tuple[str, int, str, int]]:
    if graph is None:
        raise CausalValueError("graph= lagged edges are required")
    if isinstance(graph, TemporalDag):
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph.edges()]
    return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in graph]


def _bayesian_inference_kwargs(inference: Bayesian) -> dict[str, Any]:
    backend = str(inference.backend).strip().lower()
    if backend == "laplace":
        inference_s = "bayesian"
    elif backend == "conjugate":
        inference_s = "conjugate"
    elif backend == "hmc":
        inference_s = "hmc"
    else:
        raise CausalValueError(
            f"unknown Bayesian backend {inference.backend!r}; use laplace|conjugate|hmc"
        )
    kw: dict[str, Any] = {
        "inference": inference_s,
        "prior_scale": inference.prior_scale,
    }
    if inference.n_draws_explicit:
        kw["n_draws"] = inference.n_draws
    prior_from = inference.prior_from
    if prior_from is not None:
        # Local import avoids circular import with priors ↔ estimation.
        from .priors import ComposedPrior

        if isinstance(prior_from, ComposedPrior):
            kw["composed_prior"] = prior_from.to_native_dict()
        else:
            kw["prior_artifact"] = bytes(prior_from)
    if inference.mapping is not None:
        kw["prior_mapping"] = inference.mapping.to_dict()
    return kw


def analyze_many(
    data: Mapping[str, Any] | Any,
    *,
    graph: Dag | TieredBackground | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect],
    identifier: str | None = None,
    estimator: str | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Literal["interactive", "standard", "report"] | None = None,
    candidate_screen: CandidateScreen | None = None,
) -> list[AnalysisResult]:
    """Estimate many average effects on one shared table ingest.

    Parameters
    ----------
    data:
        Column mapping / DataFrame (ingested once).
    graph:
        Static DAG, edge list, or ``TieredBackground`` shared by every query.
    queries:
        Non-empty sequence of ``AverageEffect`` queries. Outcome functionals
        are executed, not dropped to means.
    refute:
        ``False`` or a suite name; leave unset (``None``) for the default
        suite. Explicit ``refute=True`` raises ``TypeError`` — see
        :func:`antecedent._coerce.coerce_refute`.
    candidate_screen:
        Optional screen/estimate split recorded on every result.
    """
    if not queries:
        raise CausalValueError("analyze_many requires at least one query")
    if not all(isinstance(q, AverageEffect) for q in queries):
        raise CausalTypeError("analyze_many currently supports AverageEffect queries only")
    resolved_refute = None if refute is None else coerce_refute(refute)
    names, columns = ingest_columns(data)
    from .query import coerce_outcome_functional

    specs = [
        (
            q.treatment,
            q.outcome,
            float(q.control_level),
            float(q.active_level),
            coerce_outcome_functional(q.outcome_functional),
        )
        for q in queries
    ]
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        refute=resolved_refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    if latency is not None:
        kwargs["latency"] = latency
    kwargs.update(_screen_kwargs(candidate_screen))
    if isinstance(graph, TieredBackground):
        raws = _analyze_ate_many(
            names,
            columns,
            [],
            specs,
            tiers=[list(tier) for tier in graph.tiers],
            within_tier=str(graph.within_tier),
            **kwargs,
        )
    else:
        raws = _analyze_ate_many(names, columns, _static_edges(graph), specs, **kwargs)
    return [_wrap_ate(r, query=q) for r, q in zip(raws, queries, strict=True)]


@dataclass(frozen=True)
class CandidateScreen:
    """Declared screen/estimate split for a batch family."""

    screen_id: str
    procedure: Literal["max_t", "bh", "by", "unrecorded"]
    screen_rows: Sequence[int]
    estimate_rows: Sequence[int]


def _screen_kwargs(screen: CandidateScreen | None) -> dict[str, Any]:
    if screen is None:
        return {}
    return {
        "screen_id": screen.screen_id,
        "screen_procedure": screen.procedure,
        "screen_rows": [int(i) for i in screen.screen_rows],
        "estimate_rows": [int(i) for i in screen.estimate_rows],
    }


def _joint_cell_batch_specs(
    queries: Sequence[InterventionResponse],
) -> list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]]:
    from . import intervention as intervention_specs
    from .query import coerce_outcome_functional

    specs: list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]] = []
    for query in queries:
        if getattr(query, "is_temporal", False):
            raise CausalUnsupportedError(
                "PreparedBatch.prepare_cells is licensed for static joint InterventionResponse"
            )
        supplied = query.intervention
        interventions = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(interventions) < 2:
            raise CausalUnsupportedError("prepare_cells requires joint InterventionResponse")
        treatments: list[str] = []
        kinds: list[str] = []
        parameters: list[list[float]] = []
        for spec in interventions:
            if not isinstance(spec, intervention_specs.Set):
                raise CausalUnsupportedError("prepare_cells requires binary Set interventions")
            treatments.append(spec.variable)
            kinds.append("set")
            parameters.append([spec.value])
        specs.append(
            (
                query.outcome,
                treatments,
                kinds,
                parameters,
                coerce_outcome_functional(query.outcome_functional),
            )
        )
    return specs


@dataclass(frozen=True)
class SharedBatchDesign:
    """Fold assignment and covariate design frozen on a prepared batch.

    Folds and, when adjustment sets agree, the ``[1 | Z]`` matrix are shared.
    Propensity and outcome residualization remain per-query fits on that design.
    """

    n_folds: int
    fold_ids: tuple[int, ...]
    adjustment_set: tuple[int, ...] | None
    shares_covariates: bool
    shares_propensity: bool = False
    shares_outcome_residualization: bool = False


@dataclass
class PreparedBatch:
    """Compile-once batch of average-effect or joint-cell plans.

    ``prepare`` and ``prepare_cells`` freeze one fold-assignment object and,
    when every query shares a certified adjustment set, one covariate design.
    Propensity and outcome residualization are still fit per query.     A family of two or more average-effect claims attaches joint IF covariance
    and max-t / BH / BY on those contrasts. Joint-cell families keep
    ``simultaneous_interval`` on cell levels and, unless ``family_contrast``
    is ``None``, test a declared score-difference contrast (default
    ``cell_minus_control``) with max-t and ``family_contrast_interval`` on
    that contrast family.
    """

    _native: Any
    _queries: tuple[AverageEffect | InterventionResponse, ...]
    _names: tuple[str, ...]

    @property
    def shared_design(self) -> SharedBatchDesign | None:
        n_folds = self._native.shared_n_folds()
        if n_folds is None:
            return None
        folds = self._native.shared_fold_ids() or []
        adj = self._native.shared_adjustment_set()
        return SharedBatchDesign(
            n_folds=int(n_folds),
            fold_ids=tuple(int(i) for i in folds),
            adjustment_set=None if adj is None else tuple(int(i) for i in adj),
            shares_covariates=bool(self._native.shares_covariates()),
        )

    @classmethod
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[AverageEffect],
        identifier: str | None = None,
        estimator: str | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = False,
        seed: int = 1,
        bootstrap: int | None = 0,
        threads: int = 1,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
    ) -> PreparedBatch:
        if not queries:
            raise CausalValueError("PreparedBatch.prepare requires at least one query")
        if not all(isinstance(q, AverageEffect) for q in queries):
            raise CausalTypeError(
                "PreparedBatch.prepare supports AverageEffect queries only; "
                "use prepare_cells for joint InterventionResponse"
            )
        resolved_refute: bool | str = False if refute is None else coerce_refute(refute)
        names, columns = ingest_columns(data)
        from .query import coerce_outcome_functional

        specs = [
            (
                q.treatment,
                q.outcome,
                float(q.control_level),
                float(q.active_level),
                coerce_outcome_functional(q.outcome_functional),
            )
            for q in queries
        ]
        kwargs: dict[str, Any] = dict(
            identifier=identifier,
            estimator=estimator,
            refute=resolved_refute,
            seed=seed,
            bootstrap=0 if bootstrap is None else bootstrap,
            threads=threads,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_ate_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_ate_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    @classmethod
    def prepare_cells(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[InterventionResponse],
        identifier: str | None = None,
        estimator: str | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = False,
        seed: int = 1,
        bootstrap: int | None = 0,
        threads: int = 1,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
        family_contrast: Literal["cell_minus_control", "interaction"] | None = "cell_minus_control",
    ) -> PreparedBatch:
        """Compile discrete joint ``InterventionResponse`` cells into one batch.

        Licensed on a DAG and on CoDetermined ``TieredBackground``. Unknown-tier
        joint has no single ADMG and refuses. Pair families share folds and, when
        adjustment sets agree, the covariate design; estimation attaches joint
        IF covariance on cell **levels**. Family p-values / FDR use
        ``family_contrast`` (default ``cell_minus_control``, the same contrast
        as cell.aipw refuters). Max-t and ``family_contrast_interval`` are
        formed on that contrast family. ``family_contrast=None`` publishes no
        p-values. ``estimate.ate`` remains the requested cell level, not a
        contrast.
        """
        if not queries:
            raise CausalValueError("PreparedBatch.prepare_cells requires at least one query")
        if not all(isinstance(q, InterventionResponse) for q in queries):
            raise CausalTypeError("PreparedBatch.prepare_cells supports InterventionResponse only")
        if isinstance(graph, TieredBackground):
            if identifier not in (None, "generalized.adjustment"):
                raise CausalUnsupportedError(
                    "CoDetermined joint cells require identifier generalized.adjustment"
                )
            if estimator not in (None, "cell.aipw"):
                raise CausalUnsupportedError("CoDetermined joint cells require estimator cell.aipw")
        resolved_refute: bool | str = False if refute is None else coerce_refute(refute)
        names, columns = ingest_columns(data)
        specs = _joint_cell_batch_specs(queries)
        kwargs: dict[str, Any] = dict(
            identifier=None if isinstance(graph, TieredBackground) else identifier,
            estimator=estimator,
            refute=resolved_refute,
            seed=seed,
            bootstrap=0 if bootstrap is None else bootstrap,
            threads=threads,
            family_contrast=family_contrast,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_cells_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_cells_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> list[AnalysisResult]:
        names, columns = ingest_columns(data)
        raws = self._native.estimate(names, columns, seed=seed, threads=threads)
        return [_wrap_ate(r, query=q) for r, q in zip(raws, self._queries, strict=True)]


@dataclass(frozen=True)
class IdentifyResult:
    """Identify-only result (no estimate), retaining its certified query."""

    status: str
    method: str
    adjustment_set: list[str]
    query: Any = None


def identify(
    *,
    graph: Dag | Admg | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> IdentifyResult:
    """Identify without estimating.

    Accepts a ``Dag``, an ``Admg``, or an edge list. Pass ``names`` with an edge
    list (variable order); with a typed graph the names come from
    ``graph.nodes()``.

    ``identify(TieredBackground)`` stays Rust-only in 1.5
    (``identify_tiered`` / ``identify_tiered_joint``). Prepared identification
    already returns the certificate; this entry does not accept a tier rule.
    Pair-family joint cells use :meth:`PreparedBatch.prepare_cells`.

    Prefer an ``Admg`` whenever a confounder is unmeasured. A ``Dag`` has no way
    to say a variable cannot be observed, so a latent common cause flattened
    into one is treated as an ordinary adjustable node and the effect is
    reported identified by adjusting on something no study can measure.
    ``Dag.latent_project(observed)`` produces the ``Admg`` for that graph.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(graph, Admg):
        status, method, adjustment = _identify_ate_admg(
            list(graph.nodes()),
            graph,
            query.treatment,
            query.outcome,
            identifier=identifier,
        )
        return IdentifyResult(
            status=status, method=method, adjustment_set=list(adjustment), query=query
        )
    if isinstance(graph, Dag):
        node_names = list(graph.nodes())
        edges = list(graph.edges())
    else:
        if names is None:
            raise CausalValueError("identify(edge_list) requires names=")
        node_names = list(names)
        edges = list(graph)
    status, method, adjustment = _identify_ate(
        node_names,
        edges,
        query.treatment,
        query.outcome,
        identifier=identifier,
    )
    return IdentifyResult(
        status=status, method=method, adjustment_set=list(adjustment), query=query
    )


def _response_support_bounds(raw: Any) -> dict[str, tuple[float, float]]:
    """Map native support minima/maxima onto axis names.

    Static curves align one interval per treatment name. Temporal dose × horizon
    surfaces report a multi-axis query region wider than the treatment list.
    """
    minima = list(getattr(raw, "support_minima", ()) or ())
    maxima = list(getattr(raw, "support_maxima", ()) or ())
    treatments = list(getattr(raw, "treatments", ()) or ())
    if len(minima) == len(treatments):
        return {
            name: (lower, upper)
            for name, lower, upper in zip(treatments, minima, maxima, strict=True)
        }
    axis_names = (
        ["dose", "horizon"] if len(minima) == 2 else [f"axis_{i}" for i in range(len(minima))]
    )
    return {
        name: (lower, upper) for name, lower, upper in zip(axis_names, minima, maxima, strict=True)
    }


def _horizon_adjustment_sets(raw: Any) -> tuple[tuple[str, ...], ...] | None:
    sets = tuple(
        tuple(str(name) for name in group)
        for group in (getattr(raw, "horizon_adjustment_sets", ()) or ())
    )
    return sets or None


def _support_point_status(raw: Any) -> tuple[SupportStatus, ...] | None:
    """Native empty vec means a static curve with no per-cell support grid."""
    cells = tuple(getattr(raw, "support_point_status", ()) or ())
    return tuple(cast(SupportStatus, status) for status in cells) if cells else None


def _wrap_prepared_response(
    raw: Any,
    query: ResponseCurve | InterventionResponse | None = None,
    prepared: Any | None = None,
) -> CausalResponseView:
    """Build a :class:`CausalResponseView` from a prepared-response native DTO."""
    execution = None if prepared is None else prepared._native.snapshot()
    slots = (
        None if execution is None else ReasoningSlots.from_contract(execution.execution_contract())
    )
    from typing import cast

    response = (
        ResponseView(raw.treatments, raw.outcomes, raw.points, raw.values)
        if raw.points and raw.values
        else None
    )
    temporal = bool(getattr(query, "is_temporal", False))
    identifier = getattr(raw, "identifier", None)
    validation: ResponseValidationView | None
    if identifier == "generalized.adjustment":
        method = "generalized.adjustment"
        identify_op = "identify.generalized_adjustment"
        validation = None
    elif temporal:
        method = "temporal.backdoor.unfolded"
        identify_op = "identify.temporal_backdoor"
        validation = ResponseValidationView(
            (
                ResponseValidationCheck(
                    "refute.temporal_response.skipped",
                    "skipped",
                    None,
                    None,
                    "scalar ATE refuters are not applicable to a function-valued temporal response",
                ),
            )
        )
    else:
        method = "response.backdoor"
        identify_op = "identify.response"
        validation = None
    certificate_json = getattr(raw, "certificate_json", None)
    envelope = None
    if getattr(raw, "identified_mass", None) is not None:
        if raw.lower is None or raw.upper is None:
            raise RuntimeError("native structural response omitted its identified envelope")
        envelope = ResponseEnvelopeView(
            raw.treatments,
            raw.outcomes,
            raw.points,
            raw.lower,
            raw.upper,
            float(raw.identified_mass),
            float(raw.unidentified_mass),
            int(raw.completion_count),
            int(raw.truncated_completions),
            bool(raw.enumeration_capped),
            cast(Literal["full_class", "examined_completions"], raw.mass_scope),
            cast(
                Literal[
                    "posterior_probability",
                    "completion_enumeration",
                    "caller_supplied_class_prior",
                ],
                raw.weight_basis,
            ),
            tuple(raw.atom_keys),
            tuple(raw.atom_weights),
            tuple(raw.atom_statuses),
            tuple(tuple(values) for values in raw.atom_values),
            float(getattr(raw, "unevaluable_mass", None) or 0.0),
            float(getattr(raw, "subsampled_out_mass", None) or 0.0),
        )
    return CausalResponseView(
        certificate=json.loads(certificate_json) if certificate_json else None,
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=ResponseUncertainty(
            cast(UncertaintyKind, raw.uncertainty_kind),
            lower=raw.lower,
            upper=raw.upper,
            level=raw.level,
            standard_error=raw.standard_error,
            replicates=raw.replicates,
            artifact_id=raw.artifact_id,
        ),
        support=SupportReport(
            cast(SupportStatus, raw.support_status),
            _response_support_bounds(raw),
            [
                SupportDiagnostic(identifier, values, detail)
                for identifier, values, detail in zip(
                    raw.diagnostic_ids,
                    raw.diagnostic_values,
                    raw.diagnostic_details,
                    strict=True,
                )
            ],
            raw.warnings,
            _support_point_status(raw),
        ),
        identification=IdentificationView(
            status=raw.identification,
            method=method,
            adjustment_set=list(getattr(raw, "adjustment_set", ())),
            assumption_count=len(raw.assumptions),
            derivation_step_count=0,
            horizon_adjustment_sets=_horizon_adjustment_sets(raw),
        ),
        assumptions=raw.assumptions,
        provenance={
            "operation_id": raw.provenance_id,
            "operation_ids": [identify_op, raw.provenance_id],
        },
        envelope=envelope,
        validation=validation,
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        diagnostics=tuple(getattr(raw, "diagnostics", ()) or ()),
        _prepared=prepared,
        _execution=execution,
        reasoning=slots,
        program_id=None if slots is None else slots.program_id,
        claim_id=None if slots is None else slots.claim_id,
        data_snapshot_id=None if slots is None else slots.data_snapshot_id,
    )


def _prepared_columns(data: Any) -> tuple[list[str], list[Any], bool]:
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, columns = arrow
        return names, columns, True
    names, columns = as_columns(data)
    return names, columns, False


_PreparedQuery = (
    AverageEffect
    | ResponseCurve
    | ConditionalEffect
    | PathSpecificEffect
    | InterventionalDistribution
    | InterventionResponse
    | PulseEffect
    | SustainedEffect
    | MediationEffect
    | Counterfactual
    | PointDerivative
    | Elasticity
    | SemiElasticity
    | AverageDerivative
    | DirectionalDerivative
    | ResponseJacobian
    | TemporalMediationEffect
)


@dataclass(frozen=True)
class _Controls:
    """Execution controls a study applies to every click unless overridden.

    In-process only: none of them is part of the contract, program, or claim.
    """

    cancel: Any | None = None
    on_progress: Any | None = None
    on_stage: Any | None = None

    def kwargs(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in (
                ("cancel", self.cancel),
                ("on_progress", self.on_progress),
                ("on_stage", self.on_stage),
            )
            if value is not None
        }


_UNSET: Any = object()


def _refused(reason_code: str, message: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(message, reason_code=reason_code)


def _not_applicable(option: str, route: str) -> CausalUnsupportedError:
    return _refused("option_not_applicable", f"{option} does not apply to {route}")


def _unwrap_estimator(
    estimator: Any, estimator_config: Mapping[str, Any] | None
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Split a typed estimator config into its id and wire configuration."""
    if estimator is None or isinstance(estimator, str):
        return estimator, estimator_config
    if isinstance(estimator, Estimator):
        return str(estimator), estimator_config
    if estimator_config is not None:
        raise ValueError(
            "estimator= already carries its configuration; do not also pass estimator_config="
        )
    return estimator.estimator_id, estimator._wire()


def _frame_payload(data: Any) -> tuple[list[str], list[Any], dict[str, Any] | None]:
    """Columns plus the panel / multi-environment / event frame the natives rebuild."""
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    if isinstance(data, EventFrame):
        return (
            list(data.names),
            list(data.columns),
            {
                "kind": "events",
                "event_times_ns": [int(t) for t in data.event_times_ns],
                "align_interval_ns": int(data.align_interval_ns),
            },
        )
    if isinstance(data, PanelFrame):
        return (
            list(data.names),
            [],
            {
                "kind": "panel",
                "unit_columns": [list(cols) for cols in data.unit_columns],
                "unit_ids": [int(i) for i in data.unit_ids],
            },
        )
    if isinstance(data, MultiEnvFrame):
        return (
            list(data.names),
            [],
            {"kind": "multi_env", "env_columns": [list(cols) for cols in data.env_columns]},
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
        # A sequence of tables is multi-environment data (the J-PCMCI+ spelling).
        from ._data import as_multi_env_columns

        names, env_columns = as_multi_env_columns(list(data))
        return names, [], {"kind": "multi_env", "env_columns": env_columns}
    names, columns = ingest_columns(data)
    return names, columns, None


def _inference_wire(inference: Frequentist | Bayesian) -> dict[str, Any]:
    if isinstance(inference, Frequentist):
        return {"mode": "frequentist", "prior_scale": 10.0}
    kw = _bayesian_inference_kwargs(inference)
    if kw.get("prior_mapping") is not None and kw.get("prior_artifact") is None:
        raise CausalUnsupportedError(
            "prior_mapping names which estimand of a prior artifact to read; it has nothing "
            "to map without prior_from=<artifact>",
            reason_code="option_not_applicable",
        )
    return {
        "mode": kw.pop("inference"),
        "n_draws": kw.get("n_draws"),
        "prior_scale": kw["prior_scale"],
        "prior_artifact": kw.get("prior_artifact"),
        "prior_mapping": kw.get("prior_mapping"),
        "composed_prior": kw.get("composed_prior"),
    }


def _encode_interventions(
    supplied: Any, *, route: str, soft_hint: str
) -> tuple[list[str], list[str], list[list[float]]]:
    from . import intervention as intervention_specs

    interventions = (
        list(supplied)
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
        else [supplied]
    )
    if not interventions:
        raise CausalValueError("InterventionResponse requires at least one intervention")
    treatments: list[str] = []
    kinds: list[str] = []
    parameters: list[list[float]] = []
    for spec in interventions:
        if isinstance(spec, intervention_specs.Set):
            kind, params = "set", [spec.value]
        elif isinstance(spec, intervention_specs.Shift):
            kind, params = "shift", [spec.delta]
        elif isinstance(spec, intervention_specs.Bernoulli):
            kind, params = "bernoulli", [spec.p]
        elif isinstance(spec, intervention_specs.Gaussian):
            kind, params = "gaussian", [spec.mean, spec.variance]
        elif isinstance(spec, intervention_specs.Categorical):
            kind, params = "categorical", list(spec.probabilities)
        elif isinstance(spec, (intervention_specs.Soft, intervention_specs.Sequence)):
            raise CausalUnsupportedError(
                f"{type(spec).__name__} interventions {soft_hint} and are not estimable by {route}"
            )
        else:
            raise TypeError(
                "InterventionResponse.intervention must be an antecedent.intervention "
                "specification or a sequence of specifications"
            )
        treatments.append(spec.variable)
        kinds.append(kind)
        parameters.append(params)
    return treatments, kinds, parameters


def _accept_live_discovery(
    data: Any,
    discovery: Any,
    *,
    accept_discovered: bool,
    regimes: Sequence[int] | None,
    seed: int,
    threads: int,
    controls: _Controls,
) -> Any:
    """Run a live discovery config once and accept it through its review gate.

    Cancellation is checked at the discovery boundary and progress reports the
    discovery stage; estimation on the accepted structure honours both fully.
    """
    from .accepted_graph import AcceptedGraph

    token = controls.cancel
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled before discovery")
    if controls.on_progress is not None:
        controls.on_progress(0.0, "discovery")
    from .discovery import RPCMCI

    # Only RPCMCI labels regimes; prepare refuses `regimes=` for every other config.
    labelled = {"regimes": regimes} if isinstance(discovery, RPCMCI) else {}
    accepted = discovery.accept(
        data,
        seed=seed,
        threads=threads,
        accept_discovered=accept_discovered,
        **labelled,
    )
    if controls.on_progress is not None:
        controls.on_progress(1.0, "discovery")
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled after discovery")
    return accepted if isinstance(accepted, AcceptedGraph) else AcceptedGraph(accepted)


_GRAPH_POSTERIOR_TYPES = (ExactDagPosterior, DbnPosterior, GraphPosterior)


@dataclass
class _PrepareRoute:
    """One prepare request, routed to the native entry that compiles it."""

    names: list[str]
    columns: list[Any]
    frame: dict[str, Any] | None
    query: Any
    graph: Any
    discovery: Any
    inference: Frequentist | Bayesian
    identifier: str | None
    estimator: str | None
    estimator_config: Mapping[str, Any] | None
    refute: str | bool | None
    bootstrap: int | None
    threads: int
    seed: int
    class_prior: ClassPrior | None
    max_completions: int | None
    rd_args: tuple[str | None, float | None, float | None]
    accepted: bool
    options: dict[str, Any]
    #: Suite a route defers to the estimate's second click (see `_response`).
    deferred_suite: str | None = None

    # -- shared -----------------------------------------------------------
    def _common(self) -> dict[str, Any]:
        return {"seed": self.seed, "threads": self.threads, "options": self.options}

    def _refuse_rd(self, route: str) -> None:
        if any(value is not None for value in self.rd_args):
            raise _not_applicable(
                "running_variable / cutoff / bandwidth", f"{route} (rd.sharp needs a Dag)"
            )

    def _refuse_estimator_config(self, route: str) -> None:
        if self.estimator_config:
            raise _not_applicable("estimator_config", route)

    def _refuse_ids(self, route: str) -> None:
        if self.identifier is not None or self.estimator is not None:
            raise CausalUnsupportedError(
                f"{route} selects its identifier and estimator itself; custom "
                "identifier= / estimator= are not supported",
                reason_code="option_not_applicable",
            )

    def _explicit_refute(self) -> bool:
        return self.refute is not None and self.refute not in (False, "none")

    def compile(self) -> tuple[Any, Literal["average", "response_curve", "intervention_response"]]:
        query = self.query
        if self.discovery is not None:
            return self._graph_posterior()
        if self.graph is None:
            raise CausalValueError("PreparedAnalysis.prepare requires graph= or discovery=")
        temporal = isinstance(query, (PulseEffect, SustainedEffect, TemporalMediationEffect)) or (
            isinstance(query, (ResponseCurve, InterventionResponse))
            and getattr(query, "is_temporal", False)
        )
        if self.frame is not None and not temporal:
            raise CausalUnsupportedError(
                f"{type(query).__name__} is a static query; panel, multi-environment and "
                "event data are licensed for temporal queries (Pulse / Sustained / "
                "TemporalMediation / temporal response)",
                reason_code="data_modality_not_licensed",
            )
        if temporal:
            return self._temporal()
        if not isinstance(query, AverageEffect):
            self._refuse_rd(f"{type(query).__name__}")
        if isinstance(query, _RESPONSE_FAMILY):
            # A requested replicate count is refused before the structural
            # refusals: it is the caller's own request, not the cell's shape.
            self._refuse_response_bootstrap()
        _refuse_admg_response(self.graph, query)
        if isinstance(query, _RESPONSE_FAMILY):
            return self._response()
        if isinstance(query, AverageEffect):
            return self._average()
        if isinstance(query, ConditionalEffect):
            return self._conditional()
        if isinstance(query, InterventionalDistribution) and isinstance(self.graph, Admg):
            raise CausalUnsupportedError(ADMG_DISTRIBUTION_RUST_ONLY)
        edges = _static_edges(self.graph)
        if isinstance(query, (MediationEffect, Counterfactual)):
            return self._static_kind(edges)
        if isinstance(query, PathSpecificEffect):
            self._refuse_estimator_config("PathSpecificEffect")
            self._refuse_ids("PathSpecificEffect (path_specific.natural + functional.effect)")
            native = _NativePreparedAnalysis.prepare_path_specific(
                self.names,
                self.columns,
                edges,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                path_nodes=list(query.path_nodes) if query.path_nodes is not None else None,
                max_paths=query.max_paths,
                max_len=query.max_len,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        if isinstance(query, InterventionalDistribution):
            self._refuse_estimator_config("InterventionalDistribution")
            self._refuse_ids("InterventionalDistribution (general.id + functional.distribution)")
            native = _NativePreparedAnalysis.prepare_distribution(
                self.names,
                self.columns,
                edges,
                query.outcome,
                dict(query.interventions),
                conditioning=list(query.conditioning) or None,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        raise CausalTypeError(f"unsupported query type: {type(query)!r}")

    # -- graph posterior ------------------------------------------------------
    def _graph_posterior(self) -> tuple[Any, Any]:
        query, discovery = self.query, self.discovery
        self._refuse_rd("a graph-posterior mixture")
        self._refuse_ids("PreparedAnalysis graph-posterior discovery (per posterior atom)")
        if self.frame is not None and self.frame["kind"] != "events":
            raise CausalUnsupportedError(
                "graph-posterior cells are licensed on one table, one series, or one event "
                "stream; a panel or multi-environment mixture has no single atom weighting",
                reason_code="data_modality_not_licensed",
            )
        static = isinstance(discovery, (ExactDagPosterior, GraphPosterior))
        temporal = isinstance(discovery, (DbnPosterior, GraphPosterior))
        shared: dict[str, Any] = dict(self._common())
        if isinstance(discovery, GraphPosterior):
            shared["posterior"] = discovery
        dbn: dict[str, Any] = {"frame": self.frame} if temporal else {}
        dbn |= (
            {}
            if isinstance(discovery, GraphPosterior) or not isinstance(discovery, DbnPosterior)
            else {
                "max_lag": discovery.max_lag,
                "force_mcmc": discovery.force_mcmc,
                "n_chains": discovery.n_chains,
                "n_warmup": discovery.n_warmup,
                "mcmc_draws": discovery.n_draws,
            }
        )
        if static and isinstance(query, AverageEffect):
            native = _NativePreparedAnalysis.prepare_graph_posterior_ate(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                estimator_config=self.estimator_config,
                **shared,
            )
            return native, "average"
        self._refuse_estimator_config("a graph-posterior mixture other than AverageEffect")
        if static and isinstance(query, ConditionalEffect):
            from .query import coerce_outcome_functional

            native = _NativePreparedAnalysis.prepare_graph_posterior_conditional(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                query.modifier,
                control_level=query.control_level,
                active_level=query.active_level,
                outcome_functional=coerce_outcome_functional(
                    getattr(query, "outcome_functional", None)
                ),
                **shared,
            )
            return native, "average"
        if static and isinstance(query, ResponseCurve) and not query.is_temporal:
            native = _NativePreparedAnalysis.prepare_graph_posterior_response(
                self.names, self.columns, query.treatment, query.outcome, list(query.grid), **shared
            )
            return native, "response_curve"
        if static and isinstance(query, InterventionResponse) and not query.is_temporal:
            treatments, kinds, parameters = _encode_interventions(
                query.intervention,
                route="a graph-posterior response",
                soft_hint="require a structural/temporal model",
            )
            if any(kind not in ("set", "shift") for kind in kinds):
                raise CausalUnsupportedError(
                    "graph-posterior InterventionResponse is licensed for one Set or Shift "
                    "intervention coordinate",
                    reason_code="option_not_applicable",
                )
            native = _NativePreparedAnalysis.prepare_graph_posterior_intervention_response(
                self.names, self.columns, query.outcome, treatments, kinds, parameters, **shared
            )
            return native, "intervention_response"
        if temporal and isinstance(query, (PulseEffect, SustainedEffect)):
            native = _NativePreparedAnalysis.prepare_dbn_posterior_temporal(
                self.names,
                self.columns,
                query.treatment,
                query.outcome,
                policy=query.kind,
                window=getattr(query, "window", None),
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                **dbn,
                **shared,
            )
            return native, "average"
        if temporal and isinstance(query, TemporalMediationEffect):
            native = _NativePreparedAnalysis.prepare_dbn_posterior_mediation(
                self.names,
                self.columns,
                query.treatment,
                query.mediator,
                query.outcome,
                contrast=query.contrast,
                control_level=query.control_level,
                active_level=query.active_level,
                horizons=list(query.horizons or (1,)),
                **dbn,
                **shared,
            )
            return native, "average"
        raise CausalUnsupportedError(
            "graph-posterior structures are refused: a path, distribution, or mediation "
            "mixture is not a single estimand across posterior atoms. "
            "Licensed graph-posterior cells are AverageEffect / "
            "ConditionalEffect / static ResponseCurve / one-coordinate InterventionResponse on "
            "DAG atoms and Pulse / Sustained / TemporalMediationEffect on DBN atoms",
            reason_code="option_not_applicable",
        )

    # -- temporal -------------------------------------------------------------
    def _temporal(self) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        self._refuse_rd(f"{type(query).__name__}")
        self._refuse_estimator_config(f"{type(query).__name__} (fixed temporal estimator)")
        class_graph = graph if isinstance(graph, (TemporalCpdag, TemporalPag)) else None
        lagged = (
            []
            if class_graph is not None
            else _lagged_edges(cast("TemporalDag | Sequence[tuple[str, int, str, int]]", graph))
        )
        common: dict[str, Any] = {
            **self._common(),
            "accepted": self.accepted,
            "class_graph": class_graph,
            "frame": self.frame,
            **_class_prior_kwargs(self.class_prior),
            **_max_completions_kwargs(self.max_completions),
        }
        if isinstance(query, (PulseEffect, SustainedEffect)):
            self._refuse_ids("PulseEffect / SustainedEffect (fixed temporal backdoor estimator)")
            native = _NativePreparedAnalysis.prepare_temporal_effect(
                self.names,
                self.columns,
                lagged,
                query.treatment,
                query.outcome,
                policy=query.kind,
                window=getattr(query, "window", None),
                treatment_lag=query.treatment_lag,
                horizon_steps=query.horizon_steps,
                active_level=query.active_level,
                **common,
            )
            return native, "average"
        if isinstance(query, TemporalMediationEffect):
            self._refuse_ids("TemporalMediationEffect (fixed temporal mediation estimator)")
            native = _NativePreparedAnalysis.prepare_temporal_mediation(
                self.names,
                self.columns,
                lagged,
                query.treatment,
                query.mediator,
                query.outcome,
                contrast=query.contrast,
                control_level=query.control_level,
                active_level=query.active_level,
                horizons=list(query.horizons or (1,)),
                **common,
            )
            return native, "average"
        # Temporal ResponseCurve / InterventionResponse.
        if self.identifier not in (None, "temporal.backdoor.unfolded"):
            raise CausalValueError(
                "temporal response requires identifier='temporal.backdoor.unfolded'; "
                f"got {self.identifier!r}"
            )
        expected = (
            "response.temporal.bayesian"
            if isinstance(self.inference, Bayesian)
            else "temporal.response.gcomp"
        )
        if self.estimator not in (None, expected):
            raise CausalValueError(
                f"temporal response requires estimator={expected!r}; got {self.estimator!r}"
            )
        if self._explicit_refute():
            raise CausalUnsupportedError(
                "not_applicable: PreparedAnalysis temporal ResponseCurve/InterventionResponse "
                "cheap/full does not denote; cheap and full name the ATE-shaped scalar refuter "
                "suite and a function-valued estimand has no such state. Use refute='none'."
            )
        if isinstance(self.inference, Bayesian) and self.bootstrap:
            raise CausalUnsupportedError(
                "Bayesian responses use posterior intervals; bootstrap is unsupported"
            )
        from .intervention import encode_temporal_steps
        from .observation import (
            Complete,
            _ensure_latent_schema_column,
            _temporal_observation_kwargs,
        )

        names, columns = self.names, self.columns
        if getattr(query, "observation", None) is not None and not isinstance(
            query.observation, Complete
        ):
            if self.frame is not None:
                raise CausalUnsupportedError(
                    "an observation mechanism adds a latent schema column to one series; "
                    "panel, multi-environment and event frames carry fixed partitions",
                    reason_code="data_modality_not_licensed",
                )
            names, columns = _ensure_latent_schema_column(names, columns, query.observation)
        observation_kwargs = _temporal_observation_kwargs(query)
        if isinstance(query, InterventionResponse):
            supplied = query.intervention
            specs = (
                list(supplied)
                if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
                else [supplied]
            )
            treatments: list[str] = []
            kinds: list[str] = []
            parameters: list[list[float]] = []
            for spec in specs:
                for variable, kind, params in encode_temporal_steps(spec):
                    treatments.append(variable)
                    kinds.append(kind)
                    parameters.append(params)
            grid = None
            response_kind: Literal["response_curve", "intervention_response"] = (
                "intervention_response"
            )
            outcomes = [query.outcome]
        else:
            treatments, kinds, parameters = [query.treatment], None, None  # type: ignore[assignment]
            grid = list(query.grid)
            response_kind = "response_curve"
            outcomes = [query.outcome]
        native = _NativePreparedAnalysis.prepare_temporal_response(
            names,
            columns,
            lagged,
            query.kind,
            treatments,
            outcomes,
            grid=grid,
            intervention_kinds=kinds,
            intervention_parameters=parameters,
            horizons=list(query.horizons or ()),
            policy=query.policy,
            treatment_lag=query.treatment_lag,
            max_history_lag=query.max_history_lag,
            **common,
            **observation_kwargs,
        )
        return native, response_kind

    # -- static average / conditional ------------------------------------------
    def _average(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        functional = coerce_outcome_functional(getattr(query, "outcome_functional", None))
        if isinstance(graph, TieredBackground):
            self._refuse_rd("a TieredBackground")
            if self.identifier is not None:
                raise CausalUnsupportedError(
                    "TieredBackground selects its own identifier; omit identifier",
                    reason_code="option_not_applicable",
                )
            native = _NativePreparedAnalysis.prepare_tiered(
                self.names,
                self.columns,
                [list(tier) for tier in graph.tiers],
                str(graph.within_tier),
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                estimator=self.estimator,
                estimator_config=self.estimator_config,
                outcome_functional=functional,
                **self._common(),
            )
            return native, "average"
        if isinstance(graph, (Pag, Cpdag, Admg)):
            self._refuse_rd(f"a {type(graph).__name__}")
            native = _NativePreparedAnalysis.prepare_class_ate(
                self.names,
                self.columns,
                graph,
                query.treatment,
                query.outcome,
                control_level=query.control_level,
                active_level=query.active_level,
                identifier=self.identifier,
                estimator=self.estimator,
                estimator_config=self.estimator_config,
                outcome_functional=functional,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "average"
        running_variable, cutoff, bandwidth = self.rd_args
        native = _NativePreparedAnalysis.prepare(
            self.names,
            self.columns,
            _static_edges(graph),
            query.treatment,
            query.outcome,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=self.identifier,
            estimator=self.estimator,
            estimator_config=self.estimator_config,
            outcome_functional=functional,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _conditional(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        self._refuse_estimator_config("ConditionalEffect (fixed conditional estimator)")
        class_graph = isinstance(graph, (Pag, Cpdag))
        expected_identifier = "generalized.adjustment" if class_graph else "backdoor.adjustment"
        expected_estimator = (
            "conditional.bayesian"
            if isinstance(self.inference, Bayesian)
            else "conditional.linear.adjustment"
        )
        if self.identifier not in (None, expected_identifier) or self.estimator not in (
            None,
            expected_estimator,
        ):
            prefix = "Cpdag/Pag ConditionalEffect" if class_graph else "ConditionalEffect"
            raise CausalUnsupportedError(
                f"{prefix} requires {expected_identifier} and {expected_estimator}"
            )
        native = _NativePreparedAnalysis.prepare_conditional(
            self.names,
            self.columns,
            [] if class_graph else _static_edges(graph),
            query.treatment,
            query.outcome,
            query.modifier,
            graph=graph if class_graph else None,
            control_level=query.control_level,
            active_level=query.active_level,
            identifier=self.identifier,
            estimator=self.estimator,
            outcome_functional=coerce_outcome_functional(
                getattr(query, "outcome_functional", None)
            ),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    def _static_kind(self, edges: list[tuple[str, str]]) -> tuple[Any, Any]:
        query = self.query
        self._refuse_estimator_config(f"{type(query).__name__}")
        expected_id = (
            "path_specific.natural" if isinstance(query, MediationEffect) else "gcm.parametric"
        )
        expected_est = "mediation.linear" if isinstance(query, MediationEffect) else "gcm.fit"
        if self.identifier not in (None, expected_id) or self.estimator not in (None, expected_est):
            raise CausalUnsupportedError(f"{query.kind} requires {expected_id} and {expected_est}")
        if isinstance(query, Counterfactual) and self._explicit_refute():
            raise CausalUnsupportedError(
                "refused: Counterfactual cheap/full are not licensed; there is no "
                "native ITE refuter suite and ATE refuters do not apply."
            )
        if isinstance(query, Counterfactual) and self.bootstrap:
            raise CausalUnsupportedError("counterfactual sampling uncertainty is unavailable")
        native = _NativePreparedAnalysis.prepare_static_kind(
            self.names,
            self.columns,
            edges,
            query.kind,
            query.treatment,
            query.outcome,
            mediators=list(query.mediators) if isinstance(query, MediationEffect) else [],
            contrast=query.contrast if isinstance(query, MediationEffect) else "mediated",
            control_level=query.control_level,
            active_level=query.active_level,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "average"

    # -- static responses -------------------------------------------------------
    def _refuse_response_bootstrap(self) -> None:
        """Only Frequentist temporal surfaces resample; elsewhere refuse the request."""
        if not self.bootstrap:
            return
        raise CausalUnsupportedError(
            "Bayesian responses use posterior intervals; bootstrap is unsupported"
            if isinstance(self.inference, Bayesian)
            else "prepared static responses use analytic or influence-function "
            "uncertainty; bootstrap= applies to Frequentist temporal responses only"
        )

    def _response(self) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        # A scalar Dag InterventionResponse has an ATE-shaped state, so the
        # refuter suite denotes on it; a curve or a derivative is function
        # valued and cheap/full name nothing there.
        scalar_dag_response = isinstance(query, InterventionResponse) and isinstance(graph, Dag)
        if self._explicit_refute() and scalar_dag_response:
            # The response executor has no validation stage; the study runs the
            # suite as the second click of every estimate.
            self.deferred_suite = "placebo" if self.refute is True else str(self.refute)
            self.options.pop("refute", None)
        if self._explicit_refute() and not scalar_dag_response:
            raise CausalUnsupportedError(
                "not_applicable: a function-valued response has no ATE-shaped state for the "
                "cheap/full/placebo refuter suite; prepare it with refute='none'"
            )
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            _check_quantile_functional(query, self.inference, self.estimator, graph)
        response_options = _parse_response_estimator_config(query, self.estimator_config)
        if isinstance(query, (ResponseCurve, InterventionResponse)) and isinstance(
            graph, (Pag, Cpdag)
        ):
            return self._class_response(response_options)
        if isinstance(query, InterventionResponse) and isinstance(graph, TieredBackground):
            return self._tiered_response()
        _check_response_threads(self.threads)
        _check_response_strategy(
            query,
            graph=graph,
            identifier=self.identifier,
            estimator=self.estimator,
            inference=self.inference,
        )
        edges = _static_edges(graph)
        if isinstance(query, ResponseCurve):
            return self._response_curve(edges, response_options)
        if isinstance(query, InterventionResponse):
            return self._intervention_response(edges)
        return self._derivative(edges, response_options)

    def _class_response(self, response_options: Mapping[str, Any]) -> tuple[Any, Any]:
        query, graph = self.query, self.graph
        if response_options:
            raise _not_applicable("estimator_config", "a Cpdag/Pag response envelope")
        if self.identifier not in (None, "generalized.adjustment"):
            raise CausalUnsupportedError(
                "Cpdag/Pag response requires identifier='generalized.adjustment'"
            )
        if isinstance(self.inference, Bayesian):
            expected = "response.bayesian"
        elif isinstance(query, ResponseCurve):
            expected = "response.kennedy_dr"
        else:
            expected = "response.intervention_gcomp"
        if self.estimator not in (None, expected):
            raise CausalUnsupportedError(
                f"Cpdag/Pag response requires estimator={expected!r}; got {self.estimator!r}"
            )
        observation = getattr(query, "observation", None)
        if observation is not None:
            from .observation import Complete

            if not isinstance(observation, Complete):
                raise CausalUnsupportedError(
                    "an observation mechanism is estimated on one identified adjustment set; "
                    "a PAG/CPDAG envelope mixes several completions, each needing its own "
                    "observation model",
                    reason_code="option_not_applicable",
                )
        if isinstance(query, ResponseCurve):
            native = _NativePreparedAnalysis.prepare_class_response(
                self.names,
                self.columns,
                graph,
                query.kind,
                [query.treatment],
                [query.outcome],
                grid=list(query.grid),
                identifier=self.identifier,
                estimator=self.estimator,
                accepted=self.accepted,
                **self._common(),
            )
            return native, "response_curve"
        treatments, kinds, parameters = _encode_interventions(
            query.intervention,
            route="response.intervention_gcomp",
            soft_hint="require a structural/temporal model",
        )
        native = _NativePreparedAnalysis.prepare_class_response(
            self.names,
            self.columns,
            graph,
            "intervention_response",
            treatments,
            [query.outcome],
            intervention_kinds=kinds,
            intervention_parameters=parameters,
            identifier=self.identifier,
            estimator=self.estimator,
            accepted=self.accepted,
            **self._common(),
        )
        return native, "intervention_response"

    def _tiered_response(self) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query, graph = self.query, self.graph
        if not isinstance(self.inference, Frequentist):
            raise CausalUnsupportedError("CoDetermined joint cells are Frequentist cell.aipw")
        if self.identifier not in (None, "generalized.adjustment"):
            raise CausalUnsupportedError(
                "CoDetermined joint cells require identifier generalized.adjustment"
            )
        if self.estimator not in (None, "cell.aipw"):
            raise CausalUnsupportedError("CoDetermined joint cells require estimator cell.aipw")
        from . import intervention as intervention_specs

        supplied = query.intervention
        specs = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(specs) < 2:
            raise CausalUnsupportedError(
                "cell-AIPW on a tiered background requires joint InterventionResponse"
            )
        if not all(isinstance(spec, intervention_specs.Set) for spec in specs):
            raise CausalUnsupportedError(
                "CoDetermined joint cells require binary Set interventions"
            )
        native = _NativePreparedAnalysis.prepare_tiered_intervention_response(
            self.names,
            self.columns,
            [list(tier) for tier in graph.tiers],
            str(graph.within_tier),
            query.outcome,
            [spec.variable for spec in specs],
            ["set"] * len(specs),
            [[spec.value] for spec in specs],
            outcome_functional=coerce_outcome_functional(query.outcome_functional),
            **self._common(),
        )
        return native, "intervention_response"

    def _response_curve(
        self, edges: list[tuple[str, str]], response_options: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        query = self.query
        from .observation import (
            Complete,
            _assumption_kwargs,
            _ensure_latent_schema_column,
            _mechanism_kwargs,
        )

        if query.observation is not None and not isinstance(query.observation, Complete):
            from ._native import prepare_observation_response

            if not isinstance(self.inference, Frequentist):
                raise CausalUnsupportedError("observation-adjusted responses require Frequentist")
            if len(query.observation_assumptions) != 1:
                raise ValueError("observation response requires exactly one explicit assumption")
            if response_options:
                raise _not_applicable("estimator_config", "an observation-adjusted response")
            if self.identifier not in (None, "response.backdoor") or self.estimator not in (
                None,
                "response.kennedy_dr",
            ):
                raise CausalUnsupportedError(
                    "observation response requires response.backdoor and response.kennedy_dr"
                )
            names, columns = _ensure_latent_schema_column(
                self.names, self.columns, query.observation
            )
            kwargs = _mechanism_kwargs(query.observation)
            kwargs.update(_assumption_kwargs(query.observation_assumptions[0]))
            native = prepare_observation_response(
                names,
                columns,
                edges,
                query.treatment,
                query.outcome,
                list(query.grid),
                accepted=self.accepted,
                **cast(dict[str, Any], kwargs),
                **self._common(),
            )
            return native, "response_curve"
        native = _NativePreparedAnalysis.prepare_response(
            self.names,
            self.columns,
            edges,
            query.treatment,
            query.outcome,
            list(query.grid),
            identifier=self.identifier,
            estimator=self.estimator,
            accepted=self.accepted,
            bandwidth=response_options.get("bandwidth"),
            simultaneous_replicates=response_options.get("simultaneous_replicates"),
            confidence_level=response_options.get("confidence_level", 0.95),
            multiplier_seed=response_options.get("multiplier_seed", self.seed),
            export_row_diagnostics=bool(response_options.get("export_row_diagnostics", False)),
            **self._common(),
        )
        return native, "response_curve"

    def _intervention_response(self, edges: list[tuple[str, str]]) -> tuple[Any, Any]:
        from .query import coerce_outcome_functional

        query = self.query
        bayesian = isinstance(self.inference, Bayesian)
        if bayesian and self._explicit_refute():
            raise CausalUnsupportedError(
                "not_applicable: Bayesian InterventionResponse cheap/full does not denote"
            )
        treatments, kinds, parameters = _encode_interventions(
            query.intervention,
            route="response.intervention_gcomp",
            soft_hint="require a temporal response cell (set horizons=...)",
        )
        native = _NativePreparedAnalysis.prepare_intervention_response(
            self.names,
            self.columns,
            edges,
            query.outcome,
            treatments,
            kinds,
            parameters,
            estimator=self.estimator,
            outcome_functional=coerce_outcome_functional(query.outcome_functional),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "intervention_response"

    def _derivative(
        self, edges: list[tuple[str, str]], response_options: Mapping[str, Any]
    ) -> tuple[Any, Any]:
        from .observation import Complete

        query = self.query
        if getattr(query, "observation", None) is not None and not isinstance(
            query.observation, Complete
        ):
            raise CausalUnsupportedError("derivatives require complete observations")
        if query.observation_assumptions:
            raise CausalUnsupportedError("derivatives require the unweighted observed population")
        if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
            treatments = list(query.treatments)
            outcomes = list(query.outcomes)
        else:
            treatments = [query.treatment]
            outcomes = [query.outcome]
        at = getattr(query, "at", None)
        if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
            at = [at[t] for t in treatments] if isinstance(at, Mapping) else list(query.at)
        elif at is not None:
            at = [at]
        direction = getattr(query, "direction", None)
        if direction is not None:
            direction = (
                [direction[t] for t in treatments]
                if isinstance(direction, Mapping)
                else list(direction)
            )
        scale = (
            "log_log"
            if isinstance(query, Elasticity)
            else "log_" + query.log_scale
            if isinstance(query, SemiElasticity)
            else "identity"
        )
        native = _NativePreparedAnalysis.prepare_derivative(
            self.names,
            self.columns,
            edges,
            query.kind,
            treatments,
            outcomes,
            at=at,
            direction=direction,
            order=getattr(query, "order", 1),
            scale=scale,
            weighting=getattr(query, "weighting", None) or "observed",
            bandwidth=response_options.get("bandwidth"),
            accepted=self.accepted,
            **self._common(),
        )
        return native, "response_curve"


def _check_quantile_functional(
    query: Any, inference: Any, estimator: str | None, graph: Any
) -> None:
    from .query import coerce_outcome_functional

    functional = coerce_outcome_functional(getattr(query, "outcome_functional", None))
    if (
        functional is not None
        and functional.get("kind") == "quantile"
        and (
            not isinstance(query, InterventionResponse)
            or getattr(query, "is_temporal", False)
            or isinstance(inference, Bayesian)
            or (
                str(estimator) != "cell.aipw"
                and not (estimator is None and isinstance(graph, TieredBackground))
            )
        )
    ):
        raise CausalUnsupportedError(
            "quantiles require Frequentist AllObserved AIPW AverageEffect, binary "
            "ConditionalEffect with one modifier, or cell-AIPW joint response; use prepare + "
            "retarget for score-table target weights"
        )


def _check_response_observation(query: Any, inference: Any) -> None:
    """Observation mechanism, its assumptions, and the target population of a response."""
    from .observation import Complete
    from .population import coerce_target_population

    if query.target_population is not None and coerce_target_population(
        query.target_population
    ) != {"kind": "all"}:
        raise _refused(
            "population_not_estimable",
            "a prepared response surface estimates the AllObserved population; declare "
            "another target with prepare + retarget on a frozen score table",
        )
    if query.observation is not None and not isinstance(query.observation, Complete):
        if isinstance(inference, Bayesian) and not getattr(query, "is_temporal", False):
            raise CausalUnsupportedError("these response cells require complete observations")
        if not getattr(query, "is_temporal", False) and isinstance(query, InterventionResponse):
            raise CausalUnsupportedError("these response cells require complete observations")
    if query.observation_assumptions and (
        query.observation is None or isinstance(query.observation, Complete)
    ):
        raise CausalUnsupportedError(
            "observation_assumptions require an explicit observation mechanism"
        )


class PreparedAnalysis:
    """Compile-once / re-estimate-many handle for licensed analysis cells.

    **Frozen at prepare:** schema (names, types, order); graph, accepted graph,
    or licensed graph-posterior atoms and weights; query identity; identifier;
    estimator and its configuration; validation suite, custom validators and
    replicate / draw budgets; observation / transport / interference
    assumptions; panel, multi-environment, or event data layout.

    **Estimate click:** same-schema data (or the retained data); seeds /
    threads; the study's execution controls (``cancel`` / ``on_progress`` /
    ``on_stage``), which a click may override. Does not re-identify or
    recompile the logical plan.

    **Refute click:** AverageEffect and scalar Dag ``InterventionResponse``.
    :meth:`refute` raises ``CausalUnsupportedError``, prefixed with the
    matrix's own wire id for the cell: ``not_applicable:`` for ResponseCurve
    and for temporal / class-aware InterventionResponse (those remain
    function-valued or envelope-mixed surfaces). ConditionalEffect and Dag
    InterventionResponse cheap/full run the scalar ATE-shaped suite.

    **Re-prepare required:** any frozen field change, including schema mismatch.

    ``analyze`` is ``prepare(...).estimate()``; its result retains this handle
    as ``result.study``. For streaming append + incremental OLS, use
    :class:`antecedent.CausalState`.
    """

    def __init__(
        self,
        native: Any,
        *,
        kind: Literal["average", "response_curve", "intervention_response"] = "average",
        query: _PreparedQuery | None = None,
        seed: int = 1,
        threads: int = 1,
        controls: _Controls | None = None,
        deferred_suite: str | None = None,
        snapshot_data: Any = None,
    ) -> None:
        self._native = native
        self._kind = kind
        self._query = query
        # A scalar Dag InterventionResponse runs its refuter suite as the
        # second click of every estimate, so `refute=` is honoured on the
        # prepared route exactly as it is on a one-shot analyze.
        self._deferred_suite = deferred_suite
        self._snapshot_data = snapshot_data
        self._seed = seed
        self._threads = threads
        self._controls = controls or _Controls()
        self._cancelled = False

    def _frozen(self, execution: Any) -> PreparedAnalysis:
        """A handle over one retained execution with this study's seed and controls."""
        return PreparedAnalysis(
            execution,
            kind=self._kind,
            query=self._query,
            seed=self._seed,
            threads=self._threads,
            controls=self._controls,
        )

    @property
    def validator_names(self) -> tuple[str, ...]:
        """Attested custom-validator names frozen at prepare (their claim identity)."""
        return tuple(self._native.validator_names())

    def rebind_validators(self, validators: Mapping[str, Any]) -> None:
        """Re-bind caller custom validators by their exact attested names.

        Validators are in-process callbacks: a claim carries their attested
        results, never the callables. The names must equal
        :attr:`validator_names`; otherwise the rebind is refused.
        """
        self._native.rebind_validators(dict(validators))

    @classmethod
    @describe_refusal
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        query: _PreparedQuery,
        graph: Dag | Sequence[tuple[str, str]] | Any | None = None,
        discovery: Any | None = None,
        inference: Frequentist | Bayesian | None = None,
        identifier: str | Identifier | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int = 1,
        latency: Latency | Literal["interactive", "standard", "report"] | None = None,
        class_prior: ClassPrior | None = None,
        max_completions: int | None = None,
        population_registry: Any | None = None,
        cancel: Any | None = None,
        on_progress: Any | None = None,
        on_stage: Any | None = None,
        validators: Sequence[Any] | Mapping[str, Any] | None = None,
        accept_discovered: bool = True,
        regimes: Sequence[int] | None = None,
        running_variable: str | None = None,
        cutoff: float | None = None,
        bandwidth: float | None = None,
    ) -> PreparedAnalysis:
        """Compile a durable plan for a licensed analysis cell.

        Every argument either reaches the compiled study or is refused with a
        reason code; none is dropped. Omitted ``refute`` / ``bootstrap`` /
        ``latency`` reach the Rust study builder as omitted, so its one
        omitted-default table, latency-tier mapping and refute downgrade apply
        (``antecedent._defaults.OMITTED`` reads that table).

        Supports ``AverageEffect``, ``ResponseCurve``, ``ConditionalEffect``,
        ``PathSpecificEffect``, ``InterventionalDistribution``,
        ``InterventionResponse``, ``PulseEffect``, ``SustainedEffect``,
        ``TemporalMediationEffect``, static ``MediationEffect``,
        ``Counterfactual``, and the six Frequentist derivative query types
        on an explicit graph or accepted wrapper.
        ``AverageEffect`` also prepares on a ``Pag`` or bidirected ``Admg``; the
        generalized-adjustment envelope or general-ID result is frozen at
        prepare and reused by every estimate click (``exec.identify.cached``).
        Temporal ``ResponseCurve`` / ``InterventionResponse`` (keyword ``horizons``)
        prepare on a ``TemporalDag`` or lagged edge list; their Frequentist
        ``bootstrap`` is the joint circular-block replicate count behind the
        pointwise and simultaneous bands. An omitted ``bootstrap`` takes the
        ``latency`` tier's replicate count from the omitted-default table
        (``antecedent._native.omitted_defaults()``), as Pulse / Sustained do;
        ``0`` publishes the point surface with no band and an
        ``estimate.temporal_response.band_withheld`` warning. Every other
        prepared response refuses a positive ``bootstrap``. Pulse / Sustained /
        TemporalMediation prepare on a ``TemporalDag`` or lagged edge list; a
        Frequentist ``TemporalMediationEffect`` with ``bootstrap=None`` follows the
        same ``latency`` tier for its shared circular-block replicates.
        ``discovery=ExactDagPosterior()`` / ``DbnPosterior()`` / a constructed
        ``GraphPosterior`` compiles the licensed graph-posterior cells
        (AverageEffect / ConditionalEffect / ResponseCurve / one-coordinate
        InterventionResponse on DAG atoms; Pulse / Sustained /
        TemporalMediationEffect on a DBN posterior). Frequentist DBN mixtures
        take their shared circular-block replicates from the ``latency`` tier
        (or an explicit ``bootstrap``).

        ``graph=`` takes a ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` /
        ``TieredBackground`` / ``TemporalDag`` / ``TemporalCpdag`` /
        ``TemporalPag``, an edge list, or an :class:`antecedent.AcceptedGraph`.
        ``discovery=`` also takes a live configuration (``PC`` / ``GES`` /
        ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI`` / ``PCMCI`` /
        ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``), which runs
        once and is accepted through its review gate (``accept_discovered=False``
        raises ``ReviewRequired`` while anything is pending).

        ``data`` may be a table, a series, or an ``antecedent.data`` panel /
        multi-environment / event frame; frames compile on the matching Rust
        data modality for every query it licenses and are refused otherwise.

        ``cancel`` / ``on_progress`` apply to prepare-time identification and to
        every estimate click; ``on_stage`` streams the stages of each click on
        routes that have them. The handle retains all three; a click may
        override them.
        """
        from .population import coerce_target_population, registry_wire

        coerce_query(query)
        if isinstance(identifier, Identifier):
            identifier = str(identifier)
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        if latency is not None:
            latency = str(coerce_latency(latency))  # type: ignore[assignment]
        # The native side takes a suite name or False; the parameter keeps its
        # richer public type.
        suite: str | bool | None = None
        if refute is not None:
            coerced = coerce_refute(refute)
            suite = str(coerced) if isinstance(coerced, Refute) else coerced
        if bootstrap is not None and (
            isinstance(bootstrap, bool)
            or not isinstance(bootstrap, numbers.Integral)
            or bootstrap < 0
        ):
            raise CausalValueError("bootstrap must be a non-negative integer or None")
        inference = inference or Frequentist()
        if not isinstance(inference, (Frequentist, Bayesian)):
            raise CausalTypeError("inference must be Frequentist or Bayesian")
        controls = _Controls(cancel=cancel, on_progress=on_progress, on_stage=on_stage)
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            _check_response_observation(query, inference)

        from .accepted_graph import AcceptedGraph
        from .discovery import RPCMCI

        live_discovery = discovery is not None and not isinstance(discovery, _GRAPH_POSTERIOR_TYPES)
        if isinstance(graph, AcceptedGraph) and discovery is not None:
            raise CausalUnsupportedError(
                "graph=AcceptedGraph(...) rejects discovery=; the structure artifact is "
                "already accepted (call rediscover() explicitly to replace it)"
            )
        if graph is not None and discovery is not None:
            raise CausalValueError("PreparedAnalysis.prepare rejects both graph= and discovery=")
        if regimes is not None and not isinstance(discovery, RPCMCI):
            raise _not_applicable("regimes", "a discovery other than RPCMCI")
        if not accept_discovered and not live_discovery:
            raise _not_applicable(
                "accept_discovered=False", "a study without live discovery (nothing to review)"
            )
        if live_discovery and latency == "interactive":
            raise CausalUnsupportedError(
                "discovery= is not on the interactive estimate path; run discovery once "
                "(Config.accept(data) -> AcceptedGraph), then prepare(graph=..., "
                "latency='interactive')",
                reason_code="option_not_applicable",
            )
        if live_discovery:
            graph = _accept_live_discovery(
                data,
                discovery,
                accept_discovered=accept_discovered,
                regimes=regimes,
                seed=seed,
                threads=threads,
                controls=controls,
            )
            discovery = None
        structure_accepted = isinstance(graph, AcceptedGraph)
        discovery_algorithm = None
        if structure_accepted:
            discovery_algorithm = cast(AcceptedGraph, graph).algorithm_id
            graph = cast(AcceptedGraph, graph).graph
        if class_prior is not None and (
            not isinstance(graph, (TemporalCpdag, TemporalPag))
            or not isinstance(inference, Bayesian)
        ):
            raise CausalUnsupportedError(
                "class_prior requires Bayesian inference on TemporalCpdag or TemporalPag",
                reason_code="option_not_applicable",
            )
        if max_completions is not None and not isinstance(graph, (TemporalCpdag, TemporalPag)):
            raise _not_applicable("max_completions", "a structure without class completions")
        rd_args = (running_variable, cutoff, bandwidth)

        names, columns, frame = _frame_payload(data)
        predicates, distributions = registry_wire(population_registry)
        population = coerce_target_population(getattr(query, "target_population", None))
        options: dict[str, Any] = {
            "refute": suite,
            "bootstrap": bootstrap,
            "latency": latency,
            "validators": validators,
            "target_population": population,
            "population_predicates": predicates or None,
            "population_distributions": distributions or None,
            "cancel": cancel,
            "on_progress": on_progress,
            "inference": _inference_wire(inference),
            "discovery_algorithm": discovery_algorithm,
        }
        route = _PrepareRoute(
            names=names,
            columns=columns,
            frame=frame,
            query=query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=suite,
            bootstrap=bootstrap,
            threads=threads,
            seed=seed,
            class_prior=class_prior,
            max_completions=max_completions,
            rd_args=rd_args,
            accepted=structure_accepted,
            options=options,
        )
        native, kind = route.compile()
        prepared = cls(
            native,
            kind=kind,
            query=query,
            seed=seed,
            threads=threads,
            controls=controls,
            deferred_suite=route.deferred_suite,
            snapshot_data=data if route.deferred_suite else None,
        )
        if on_stage is not None and not native.streams_stages():
            raise CausalUnsupportedError(
                "on_stage streams identify, estimate_point, uncertainty and validate for one "
                "identification and one scalar effect whose point is fitted before its "
                "uncertainty; this route fits its point and uncertainty together or mixes "
                "several identifications, so it has no such stages (use on_progress)",
                reason_code="stage_stream_unavailable",
            )
        return prepared

    def export_artifact(
        self,
        *,
        artifact_id: str = "prepared-result",
        payload: Literal["query", "result"] = "result",
    ) -> bytes:
        """Export the last full posterior/response, or its query, without refitting.

        ``payload="query"`` also supports functional path/distribution queries.

        Decode Bayesian scalar results with ``inference.decode_posterior_artifact``;
        decode response results with ``artifacts.loads``. Posterior artifacts retain
        draws, quantity names, identification and backend metadata. Response
        artifacts also retain support and assumptions; retain the analysis result
        separately for posterior assumptions and validation reports.
        """
        return self._native.export_artifact(artifact_id=artifact_id, payload=payload)

    def export(self, *, artifact_id: str = "analysis-result") -> bytes:
        """Export the last execution as a contracted ``analysis_result``, or refuse
        if no claim was produced."""
        if getattr(self, "_cancelled", False):
            raise CausalUnsupportedError(
                "Cancelled estimate produced no claim.",
                reason_code="cancelled_no_claim",
            )
        return self._native.export_contracted_artifact(artifact_id=artifact_id)

    @property
    def structure_source(self) -> str:
        """Support-matrix structure axis frozen at prepare (`explicit` or `accepted`)."""
        return str(self._native.plan_summary().get("structure_source", "explicit"))

    @property
    def evidence_status(self) -> str | None:
        """`licensed` or `allowed_unlicensed`, or ``None`` if the query is off-axis."""
        raw = self._native.plan_summary().get("evidence_status")
        return str(raw) if raw is not None else None

    @property
    def allowlist_reason(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_reason")
        return str(raw) if raw is not None else None

    @property
    def allowlist_parent(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_parent")
        return str(raw) if raw is not None else None

    def inspect(self) -> ReasoningSlots:
        """Everything known about this study, including cached identification.

        ``to_dict()`` gives the structured report; its ``contract`` field is the
        study's domain-separated identities and reasoning slots (ADR 0022), and
        its ``calibration`` is unavailable (``not_executed``) until an estimate
        runs. :meth:`preflight` is the cheap structural-only view.
        """
        from dataclasses import replace

        from .results._execution import Answer, CalibrationInfo

        return replace(
            ReasoningSlots.from_contract(dict(self._native.contract())),
            answer=Answer("unavailable", detail="not_executed"),
            calibration=CalibrationInfo(status="unavailable", reason="not_executed"),
        )

    def preflight(self) -> ReasoningSlots:
        """Cheap structural-only inspection; identification and fitting are not run."""
        return ReasoningSlots.from_contract(self._native.inspect())

    def preview_transform(self, intent: str) -> dict[str, str]:
        """Pure preview of a transformation of this study; nothing is re-executed.

        ``intent`` is one of ``display_precision``, ``filter_display``,
        ``compatible_data_replace``, ``retarget``, ``filter_population``,
        ``new_conditional_query``, ``change_graph``, ``change_prior``,
        ``change_physical_policy`` or ``average_unweighted_class``. The report
        names the intent, whether it is refused, its obligations, and the frozen
        input identities it would carry under ``input_<domain>`` keys.
        """
        return dict(self._native.preview_transform(intent))

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

    def _click_controls(self, cancel: Any, on_progress: Any, on_stage: Any) -> dict[str, Any]:
        """The study's retained controls, with any per-click override applied."""
        controls = self._controls
        chosen = _Controls(
            cancel=controls.cancel if cancel is _UNSET else cancel,
            on_progress=controls.on_progress if on_progress is _UNSET else on_progress,
            on_stage=controls.on_stage if on_stage is _UNSET else on_stage,
        )
        return chosen.kwargs()

    def _run_click(self, call: Any) -> Any:
        from .errors import CausalCancelledError

        try:
            raw = call()
        except CausalCancelledError:
            # The handle still holds the previous execution; it must not export as this one.
            self._cancelled = True
            raise
        self._cancelled = False
        return raw

    def _click_payload(self, data: Any) -> tuple[list[str], list[Any], dict[str, Any] | None]:
        query = self._query
        if isinstance(query, ResponseCurve) and query.observation is not None:
            from .observation import Complete, _ensure_latent_schema_column

            if not isinstance(query.observation, Complete):
                names, columns = ingest_columns(data)
                names, columns = _ensure_latent_schema_column(names, columns, query.observation)
                return names, columns, None
        return _frame_payload(data)

    def _wrap(self, raw: Any) -> AnalysisResult | CausalResponseView:
        if self._kind in ("response_curve", "intervention_response"):
            query = self._query if isinstance(self._query, _RESPONSE_FAMILY) else None
            return _wrap_prepared_response(raw, query=cast(Any, query), prepared=self)
        return _wrap_ate(raw, prepared=self)

    def _click(
        self,
        data: Any,
        *,
        refresh: bool,
        seed: int | None,
        threads: int | None,
        controls: dict[str, Any],
    ) -> AnalysisResult | CausalResponseView:
        seed = self._seed if seed is None else seed
        threads = self._threads if threads is None else threads
        response = self._kind in ("response_curve", "intervention_response")
        native = self._native
        if data is None:
            bound = native.estimate_response_bound if response else native.estimate_bound
            raw = self._run_click(lambda: bound(seed=seed, threads=threads, **controls))
            return self._with_deferred_suite(
                self._wrap(raw), self._snapshot_data, seed=seed, threads=threads
            )
        names, columns, frame = self._click_payload(data)
        if frame is not None:
            raw = self._run_click(
                lambda: native.estimate_frame(
                    names,
                    columns,
                    frame,
                    response=response,
                    refresh=refresh,
                    seed=seed,
                    threads=threads,
                    **controls,
                )
            )
            return self._wrap(raw)
        if refresh:
            fn = native.refresh_response if response else native.refresh
        else:
            fn = native.estimate_response if response else native.estimate
        raw = self._run_click(lambda: fn(names, columns, seed=seed, threads=threads, **controls))
        return self._with_deferred_suite(self._wrap(raw), data, seed=seed, threads=threads)

    def _with_deferred_suite(self, result: Any, data: Any, *, seed: int, threads: int) -> Any:
        """Run the study's refuter suite as this click's second click.

        A scalar ``InterventionResponse`` on a Dag is estimated by the response
        executor, which fits its point and uncertainty together and has no
        validation stage of its own; the ATE-shaped suite denotes on it and
        runs against the estimate just produced.
        """
        if self._deferred_suite is None or data is None:
            return result
        from dataclasses import replace

        refuted = self.refute(data, suite=self._deferred_suite, seed=seed, threads=threads)
        # The suite ran against this estimate, so the click's claim is the
        # refuted one; the estimate itself is the response executor's.
        return replace(
            result,
            validation=refuted.validation,
            certificate=refuted.certificate,
            reasoning=refuted.reasoning,
            evidence_status=refuted.evidence_status,
            claim_id=refuted.claim_id,
        )

    @describe_refusal
    def estimate(
        self,
        data: Mapping[str, Any] | Any = None,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> AnalysisResult | CausalResponseView:
        """Re-estimate without recompiling.

        With no ``data`` this re-executes the prepared program on the retained
        data. New ``data`` must match the prepared schema (and, for a panel /
        multi-environment / event study, its frame kind). ``seed`` / ``threads``
        default to the values given at prepare; ``cancel`` / ``on_progress`` /
        ``on_stage`` default to the study's retained controls and may be
        overridden (``None`` turns one off for this click).
        """
        return self._click(
            data,
            refresh=False,
            seed=seed,
            threads=threads,
            controls=self._click_controls(cancel, on_progress, on_stage),
        )

    @describe_refusal
    def retarget(
        self,
        weights: Any,
        depends_on: Sequence[str],
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Estimate a declared target population from frozen scores. No refit.

        Requires a prepared AllObserved iid AIPW or cell-AIPW score table.
        Nonempty ``depends_on`` needs a directed graph (DAG or ADMG) so
        descendant closure can be checked. Nonconstant weights require a
        nonempty ``depends_on``.
        """
        import numpy as np

        raw = self._native.retarget(
            np.asarray(weights, dtype=float).tolist(),
            list(depends_on),
            seed=seed,
            threads=threads,
        )
        return _wrap_ate(raw, prepared=self)

    @describe_refusal
    def reexecute_retarget(
        self,
        artifact: bytes,
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AnalysisResult:
        """Re-execute the row-weight retarget an exported contract carries.

        The weights travel in the artifact with the identity that binds them to
        one data snapshot and one score table. Re-executing them on a study
        that holds a different snapshot raises ``CausalUnsupportedError`` with
        ``reason_code="row_weights_bound_to_snapshot"`` instead of silently
        reweighting other rows.
        """
        raw = self._native.reexecute_retarget(bytes(artifact), seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    @describe_refusal
    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> AnalysisResult | CausalResponseView:
        """Replace retained data and re-estimate (controls as in :meth:`estimate`)."""
        return self._click(
            data,
            refresh=True,
            seed=seed,
            threads=threads,
            controls=self._click_controls(cancel, on_progress, on_stage),
        )

    @describe_refusal
    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
    ) -> AnalysisResult:
        """Second-click refute against the last :meth:`estimate` / :meth:`refresh`.

        Interactive first clicks typically use ``refute=False`` or ``cheap``;
        call this with ``suite="placebo"`` or ``"full"`` for the deferred suite.
        ``seed`` / ``threads`` / ``cancel`` default to the study's own.
        """
        if self._kind == "response_curve":
            raise CausalUnsupportedError(
                "not_applicable: PreparedAnalysis.refute is AverageEffect and scalar "
                "Dag InterventionResponse; ResponseCurve cheap/full/placebo do not denote."
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        names, columns, arrow = _prepared_columns(data)
        kwargs: dict[str, Any] = dict(
            seed=self._seed if seed is None else seed,
            threads=self._threads if threads is None else threads,
        )
        token = self._controls.cancel if cancel is _UNSET else cancel
        if token is not None:
            kwargs["cancel"] = token
        fn = self._native.refute_arrow_c if arrow else self._native.refute
        raw = fn(names, columns, suite, **kwargs)
        return _wrap_ate(raw, prepared=self)


__all__ = [
    "AnalysisResult",
    "ConflictSummaryView",
    "EffectEnvelope",
    "EstimateView",
    "MediationView",
    "IdentificationView",
    "IdentifyResult",
    "MediationEffectsSummary",
    "PerformanceView",
    "PhysicalPlanView",
    "PlanView",
    "PosteriorView",
    "PredictiveCheckReport",
    "PreparedAnalysis",
    "PreparedBatch",
    "SharedBatchDesign",
    "CandidateScreen",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "analyze_many",
    "identify",
    "mediation_effects_summary",
]
