"""Public-member snapshot of the objects the five-line API touches.

``antecedent`` (the package root), ``AnalysisResult``, ``PreparedAnalysis`` and
``LoadedResult`` are what ``analyze`` → ``.study`` → ``refresh`` →
``inspect().to_dict()`` → ``load(export())`` hands a caller. Their public names
are listed here in full, so adding or removing one fails this test until the
snapshot is edited in the same change: growth of the surface is a reviewed act,
not an accident.

The names are read in a fresh interpreter, because other tests import stage
modules lazily and would otherwise add attributes to the package root.
A dataclass contributes its public fields as well as its class attributes.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_DUMP = """
import dataclasses, json, sys
import antecedent
from antecedent import AnalysisResult
from antecedent._workflow import LoadedResult
from antecedent.estimation import PreparedAnalysis
from antecedent import estimators
from antecedent import design
from antecedent.transport import advanced as transport

def members(obj):
    names = {n for n in dir(obj) if not n.startswith("_")}
    if dataclasses.is_dataclass(obj):
        names |= {f.name for f in dataclasses.fields(obj) if not f.name.startswith("_")}
    try:
        from pydantic import BaseModel
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            names |= {n for n in obj.model_fields if not n.startswith("_")}
    except Exception:
        pass
    return sorted(names)

json.dump(
    {
        "transport.JointTransportIdentity": members(transport.JointTransportIdentity),
        "transport.GaussianTransportPrior": members(transport.GaussianTransportPrior),
        "transport.JointTransportPriors": members(transport.JointTransportPriors),
        "transport.JointTransportSource": members(transport.JointTransportSource),
        "transport.JointTransportTarget": members(transport.JointTransportTarget),
        "transport.JointTransportPosterior": members(transport.JointTransportPosterior),
        "transport.TemporalIntervalCandidate": members(transport.TemporalIntervalCandidate),
        "transport.NestedFisherCandidate": members(transport.NestedFisherCandidate),
        "transport.NestedMarkovPrior": members(transport.NestedMarkovPrior),
        "transport.NestedMarkovPosteriorCandidate": members(transport.NestedMarkovPosteriorCandidate),
        "transport.SampledRecoveryCandidate": members(transport.SampledRecoveryCandidate),
        "transport.SampledRecoveryIdentity": members(transport.SampledRecoveryIdentity),
        "design.CheckedPriorSignal": members(design.CheckedPriorSignal),
        "design.PriorSignalSource": members(design.PriorSignalSource),
        "antecedent": members(antecedent),
        "AnalysisResult": members(AnalysisResult),
        "PreparedAnalysis": members(PreparedAnalysis),
        "LoadedResult": members(LoadedResult),
        "Bayesian": members(antecedent.Bayesian),
        "PulseEffect": members(antecedent.PulseEffect),
        "SustainedEffect": members(antecedent.SustainedEffect),
        "estimators": sorted(estimators.__all__),
        "estimators.Overlap": members(estimators.Overlap),
        "estimators.Aipw": members(estimators.Aipw),
    },
    sys.stdout,
)
"""

# `annotations` is the `from __future__ import annotations` binding every module
# carries; it is not API. Pydantic machinery on analyze result models is not
# Antecedent API; callers use ``to_dict()``.
_NOT_API = {"annotations"}
_PYDANTIC_SURFACE = {
    "construct",
    "copy",
    "dict",
    "from_orm",
    "json",
    "model_computed_fields",
    "model_config",
    "model_construct",
    "model_copy",
    "model_dump",
    "model_dump_json",
    "model_extra",
    "model_fields",
    "model_fields_set",
    "model_json_schema",
    "model_parametrized_name",
    "model_post_init",
    "model_rebuild",
    "model_validate",
    "model_validate_json",
    "model_validate_strings",
    "parse_file",
    "parse_obj",
    "parse_raw",
    "schema",
    "schema_json",
    "update_forward_refs",
    "validate",
}

SNAPSHOT: dict[str, set[str]] = {
    "transport.JointTransportIdentity": {"datum_ids", "snapshot_digest"},
    "transport.GaussianTransportPrior": {"bank_id", "consumed", "covariance", "mean"},
    "transport.JointTransportPriors": {"invariant", "varying"},
    "transport.JointTransportSource": {"data", "id", "identity", "noise_variance", "population"},
    "transport.JointTransportTarget": {"data", "identity", "population"},
    "transport.JointTransportPosterior": {
        "calibration",
        "diagnostics",
        "draw_names",
        "draws",
        "expectation",
        "export",
        "kind",
        "parameter_names",
        "posterior_covariance",
        "posterior_mean",
        "release_status",
        "target_effect_mean",
        "target_effect_variance",
        "to_dict",
    },
    "transport.TemporalIntervalCandidate": {
        "calibration",
        "claim",
        "export",
        "identity",
        "level",
        "load",
        "lower",
        "method",
        "panel_digest",
        "point",
        "replicate_digest",
        "seed",
        "snapshot_id",
        "to_dict",
        "upper",
    },
    "transport.NestedMarkovPrior": {
        "coordinates",
        "a",
        "c0",
        "c1",
        "q20",
        "q21",
        "q40",
        "q41",
        "g00",
        "g01",
        "g10",
        "g11",
    },
    "transport.NestedMarkovPosteriorCandidate": {
        "prior",
        "coordinate_order",
        "parameter_mean",
        "values",
        "covariance",
        "credible_intervals",
        "samples",
        "diagnostics",
        "seed",
        "calibration",
        "inference",
        "identification",
        "identity",
        "export",
        "to_dict",
        "load",
    },
    "transport.NestedFisherCandidate": {
        "calibration",
        "covariance",
        "export",
        "identification",
        "identity",
        "inference",
        "intervals",
        "load",
        "method",
        "nominal_level",
        "to_dict",
        "values",
    },
    "transport.SampledRecoveryCandidate": {
        "calibration",
        "effect",
        "effect_standard_error",
        "expected_identity",
        "export",
        "inspect",
        "interval",
        "level",
        "load",
    },
    "transport.SampledRecoveryIdentity": {"data_digest", "premises_digest"},
    "design.CheckedPriorSignal": {"diagnostics", "prior", "prior_id", "signal"},
    "design.PriorSignalSource": {
        "artifact_id",
        "conflict_shrinkage",
        "intercept",
        "lineage",
        "observation_ids",
        "prior_strength",
        "quantity",
        "slope",
        "source_population",
        "weight",
    },
    "antecedent": {
        # Verbs
        "analyze",
        "estimate",
        "identify",
        "load",
        "prepare",
        # Structure, results and selectors
        "AcceptedGraph",
        "Admg",
        "Analysis",
        "AnalysisResult",
        "Bayesian",
        "CausalError",
        "ClassPrior",
        "Cpdag",
        "Dag",
        "Estimator",
        "Frequentist",
        "Identification",
        "Identifier",
        "Latency",
        "Pag",
        "Refute",
        "ReviewRequired",
        "TemporalDag",
        # Queries
        "AnomalyAttribution",
        "AnomalyReference",
        "AverageDerivative",
        "AverageEffect",
        "ChangeAttribution",
        "ConditionalEffect",
        "Counterfactual",
        "DirectionalDerivative",
        "Elasticity",
        "InterventionResponse",
        "InterventionalDistribution",
        "MediationEffect",
        "NestedCounterfactual",
        "PathSpecificEffect",
        "PointDerivative",
        "PulseEffect",
        "ResponseCurve",
        "ResponseJacobian",
        "SemiElasticity",
        "SustainedEffect",
        "TemporalMediationEffect",
        # Design queries: licensed cells run on analyze and retain a study
        "InterferenceQuery",
        # Stage modules
        "accepted_graph",
        "artifacts",
        "attribution",
        "counterfactual",
        "data",
        "decision",
        "derived",
        "design",
        "discovery",
        "errors",
        "experiment",
        "estimation",
        "estimators",
        "extensibility",
        "factorial",
        "gcm",
        "graph",
        "handoff",
        "ids",
        "inference",
        "interference",
        "intervention",
        "external",
        "joint_distribution",
        "matched",
        "model",
        "observation",
        "policy",
        "preflight",
        "regimes",
        "population",
        "prediction",
        "priors",
        "quasi",
        "query",
        "results",
        "state",
        "survival",
        "transport",
        "learners",
        "validation",
        # Lazy 2.3 stage modules (`_LAZY_MODULES`): listed by ``dir``, imported on first access
        # (`joint_distribution` and `preflight` above are also in this family)
        "effect_constancy_review",
        "execution_attempt",
        "functional_source",
        "proposal_arrival",
        "recalc_bayesian",
        "recalc_composite",
        "recalc_design",
        "recalc_external",
        "recalc_static",
        "recalc_temporal",
        "source_evidence",
        "source_projection",
        "categorical_treatment",
        "compact_export",
        "composition",
        "composition_bundle",
        "counterfactual_id",
        "cross_world",
        "decision_robust",
        "descriptive",
        "dose_grid",
        "inverse",
        "inverse_query",
        "latent_class",
        "mechanism_discrepancy",
        "msm_sensitivity",
        "nonlinear_mediation",
        "program_claims",
        "proposals",
        "recalc",
        "recalc_adjusted",
        "recalc_capabilities",
        "recalc_cell",
        "recalc_dr",
        "recovery_chain",
        "repair",
        "scenario_decision",
        "scenario_invariance",
        "sensitivity_decision",
        "temporal",
        "temporal_counterfactual",
        "transported_counterfactual",
        "vector_treatment",
    },
    "AnalysisResult": {
        "fitted_model",
        # The five-line API (`ResultAPI`)
        "answer",
        "as_point",
        "as_response",
        "msm_sensitivity",
        "calibration",
        "claim",
        "export",
        "inspect",
        "study",
        # Identities
        "claim_id",
        "data_snapshot_id",
        "program_id",
        # Sections
        "assumptions",
        "certificate",
        "diagnostics",
        "estimate",
        "evidence_status",
        "allowlist_parent",
        "allowlist_reason",
        "identification",
        "mediation",
        "mediation_grid",
        "performance",
        "panel_did",
        "synthetic_control",
        "synthetic_did",
        "local_polynomial_ratio",
        "plan",
        "policy_value",
        "continuous_dose_response",
        "survival",
        "longitudinal_regime",
        "posterior",
        "provenance",
        "query",
        "reasoning",
        "rendering_limitation",
        "support",
        "validation",
        # Historical scalars and result-level clicks
        "ate",
        "effect",
        "mean_ite",
        "refresh",
        "refute",
        "to_dict",
        # Structural uncertainty
        "structural_identified_mass",
        "structural_identified_set",
        "structural_identified_set_interval",
        "structural_identified_set_interval_level",
        "structural_identified_set_interval_method",
        "structural_identified_set_interval_truncated",
        "structural_unevaluable_mass",
        "structural_unidentified_mass",
        "structural_weight_basis",
        # Unit-level counterfactual effects
        "unit_effect_intervals",
        "unit_effect_intervals_level",
        "unit_effect_intervals_method",
        "unit_effects",
        "unit_extrapolative",
        # Design-cell sections beside the scalar estimate
        "interference",
        "randomized_effect",
        "transport",
        "transport_overlap",
        # Licensed GCM attribution cells
        "anomaly",
        "change_attribution",
    },
    "PreparedAnalysis": {
        "replace_snapshot",
        # One-factor mechanism sensitivity on the retained fixed-graph plan
        "mechanism_sensitivity",
        # Compile, execute, and the two views
        "prepare",
        "estimate",
        "refresh",
        "inspect",
        "preflight",
        # Data-reading diagnostics, cost counts and the opt-in rank-drop plan (2.2 E1)
        "diagnose",
        "diagnose_fit",
        "plan_rank_drop",
        "estimate_cost",
        "preview_transform",
        "export",
        # Second clicks on the retained execution
        "refute",
        "retarget",
        "reexecute_retarget",
        # Callback validators frozen at prepare
        "rebind_validators",
        "validator_names",
        # Plan coordinates
        "allowlist_parent",
        "allowlist_reason",
        "evidence_status",
        "plan",
        "structure_source",
        # Posterior / response payload for prior transfer
        "export_artifact",
        "checked_conditional_effect_info",
        "checked_static_mediation_info",
        "checked_bayesian_dag_ate_info",
        "checked_static_dag_response_info",
        "checked_graph_posterior_effect_info",
        "checked_temporal_effect_info",
        "checked_temporal_response_info",
    },
    "LoadedResult": {
        "fitted_model",
        "acceptance",
        "answer",
        "artifact",
        "as_point",
        "as_response",
        "msm_sensitivity",
        "ate",
        "calibration",
        "claim",
        "claim_id",
        "effect",
        "export",
        "inspect",
        "program_id",
        "refresh",
        "refute",
        "study",
    },
    # Estimator options: the Bayesian likelihood, the temporal history cap and
    # the propensity overlap policy are reviewed surface, like the verbs.
    "Bayesian": {
        "backend",
        "kind",
        "likelihood",
        "mapping",
        "n_draws",
        "n_draws_explicit",
        "prior_from",
        "prior_scale",
    },
    "PulseEffect": {
        "active_level",
        "control_level",
        "horizon_steps",
        "kind",
        "max_history_lag",
        "outcome",
        "target_population",
        "treatment",
        "treatment_lag",
    },
    "SustainedEffect": {
        "active_level",
        "control_level",
        "horizon_steps",
        "kind",
        "max_history_lag",
        "outcome",
        "target_population",
        "treatment",
        "treatment_lag",
        "window",
    },
    "estimators": {
        "Aipw",
        "CausalForest",
        "ClusterDml",
        "DML",
        "DRLearner",
        "DistanceMatching",
        "FitKind",
        "FrontdoorLinearTwoStage",
        "GlmAdjustment",
        "GlmFamilyName",
        "GlmOptions",
        "Iv2Sls",
        "IvWald",
        "LinearAdjustment",
        "Overlap",
        "PropensityMatching",
        "PropensityPenalty",
        "PropensityStratification",
        "PropensityWeighting",
        "SeKind",
        "UNSET",
    },
    "estimators.Overlap": {"clip", "trim"},
    "estimators.Aipw": {
        "bootstrap",
        "cluster_dml",
        "cluster_ids",
        "estimator_id",
        "glm_options",
        "multiway_ids",
        "nuisance_fallback",
        "overlap",
        "panel_times",
        "propensity_penalty",
        "se",
        "se_lag",
    },
}


@pytest.fixture(scope="module")
def live() -> dict[str, set[str]]:
    completed = subprocess.run(
        [sys.executable, "-c", _DUMP], check=True, text=True, capture_output=True
    )
    return {
        name: {
            item
            for item in names
            if item not in _NOT_API
            and item not in _PYDANTIC_SURFACE
            and not item.startswith("model_")
        }
        for name, names in json.loads(completed.stdout).items()
    }


@pytest.mark.parametrize("surface", sorted(SNAPSHOT))
def test_public_members_match_the_snapshot(surface: str, live: dict[str, set[str]]) -> None:
    added = sorted(live[surface] - SNAPSHOT[surface])
    removed = sorted(SNAPSHOT[surface] - live[surface])
    assert not added and not removed, (
        f"{surface}: public surface changed (added {added}, removed {removed}); "
        "update SNAPSHOT in this file in the same change if that is intended"
    )


def test_duplicate_spellings_stay_retired(live: dict[str, set[str]]) -> None:
    """The five lines have one spelling each; their retired duplicates do not return."""
    assert (
        not {"export_contracted_artifact", "contract", "reasoning", "calibration"}
        & (live["PreparedAnalysis"])
    )
    assert "preview_intent" not in live["PreparedAnalysis"]
    assert not {"display_effect", "display_mass", "data_version"} & live["AnalysisResult"]


def test_only_rpcmci_accept_takes_regimes() -> None:
    import inspect

    from antecedent import discovery

    takes = {
        name
        for name in discovery.__all__
        if isinstance(config := getattr(discovery, name), type)
        and callable(getattr(config, "accept", None))
        and "regimes" in inspect.signature(config.accept).parameters
    }
    assert takes == {"RPCMCI"}
