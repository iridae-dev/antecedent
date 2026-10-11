//! Versioned artifact IO for Antecedent.
//!
//! This crate holds the artifacts that replay through an estimator or a
//! validator, and re-exports `antecedent-io-core` (the container, wires,
//! digests and estimator-free artifacts) at its root, so every public path is
//! the one it was before the split.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use antecedent_io_core::*;

/// Analysis / estimate / identification / refutation / diagnostic wire types.
pub mod analysis_wire;
mod claim_identity;
mod estimate_wire;
/// Canonical encoding and domain-separated digests for contract identities.
pub mod identity;
mod transport_effect_wire;

pub use analysis_wire::{
    CandidateSelectionWire, ClipSensitivityWire, EffectEstimateWire, FirstStageDiagnosticsWire,
    JointCovarianceWire, OverlapReportWire, PropensityIntervalWire, ScoreColumnWire,
    ScoreInferenceWire, ScoreTableWire, effect_estimate_from_wire, effect_estimate_to_wire,
    refutation_from_wire, refutation_to_wire,
};
pub use identity::{ClaimIdentityWire, claim_digest};

pub mod admg_conditional_transport_artifact;
pub mod analysis_result_artifact;
pub mod calibration;
mod contract_reason;
pub mod contract_section;
pub mod coverage_records_data;
pub mod cpdag_completion_artifact;
mod distribution_replay;
pub use distribution_replay::distribution_factor_laws_to_wire;
pub mod external_estimate;
pub mod frozen_scores_artifact;
pub mod graphless_support;
pub mod joint_bayesian_transport_artifact;
pub mod learned_joint_transport_artifact;
pub mod measured_inference;
pub mod mixed_source_artifact;
pub mod model_bundle;
pub mod mz_transport_artifact;
pub mod nested_markov_artifact;
pub mod nested_markov_bayesian_artifact;
pub mod posterior_convert;
pub mod recovery_artifact;
pub mod sampled_recovery_artifact;
pub mod scenario_covariance_artifact;
pub mod temporal_initial_state_artifact;
pub mod temporal_interval_artifact;
pub mod temporal_refresh_artifact;
pub mod temporal_transport_artifact;
pub mod transport_interference_wire;
pub mod transport_scenario_artifact;
pub mod z_transport_artifact;
pub use analysis_result_artifact::{
    AnalysisResultHeader, AnalysisResultWire, DistributionAtomWire,
    IdentifiedSetIntervalMethodWire, IdentifiedSetIntervalWire, InterferenceInferenceWire,
    InterferencePointwiseIntervalWire, InterventionalDistributionWire, LocalPolynomialRatioWire,
    LongitudinalRegimeWire, MediationPosteriorSummaryWire, PanelDidWire, PolicyValueWire,
    RandomizedEffectWire, StructuralResponseAtomWire, StructuralResponseMixtureWire,
    StructuralWeightBasisWire, SurvivalDifferenceBandWire, SurvivalWire, SyntheticControlWire,
    SyntheticDidWire, TemporalIdentificationWire, TemporalMediationGridWire,
    TemporalMediationSliceWire, TemporalMediationUncertaintyWire, UnitEffectIntervalsWire,
    UnitEffectsWire, UpliftBinWire, decode_analysis_result_artifact,
    encode_analysis_result_artifact, encode_analysis_result_artifact_with_contract,
    identified_set_interval_from_wire, identified_set_interval_to_wire,
};
pub use contract_section::TargetWeightsSectionWire;
pub use contract_section::{
    AnalysisResultConsumption, AnalysisResultContractWire, AssumptionSlotWire,
    AttestedEvidenceWire, CONTRACT_SECTION, CONTRACT_SECTION_FORMAT, CalibrationSlotWire,
    CheckedAipwLoweringWire, CheckedAipwRowsWire, CheckedIvLoweringWire, ClaimDomainsWire,
    ClaimHostProjection, ClaimSectionWire, ContractIdentitiesWire, ContractReason,
    IdentificationSlotWire, ObligationSectionWire, ReasoningSectionWire, SlotSectionWire,
    SupportSlotWire, UncertaintyComponentWire, UncertaintySlotWire, accept_claim,
    checked_aipw_rows_digest, claim_domains, claim_kind_name, consume_analysis_result,
    contract_seal, decode_analysis_result_contract, project_claim_host, project_lossy_scalar,
    result_digest, support_empirical, validate_contract_section, validate_mixture_masses,
    verify_contract_against_body, weight_basis_name,
};
pub use external_estimate::{
    ExternalEstimateAttach, attested_external_estimate, decode_external_estimate_claim,
    encode_external_estimate_claim, parse_digest_hex,
};
pub use model_bundle::{
    ModelBundle, ModelBundleEncode, ModelBundleHeaderWire, decode_model_bundle, encode_model_bundle,
};
pub use posterior_convert::{
    PosteriorPayload, decode_causal_posterior_bytes, encode_causal_posterior,
    encode_causal_posterior_bytes, encode_causal_posterior_bytes_with_payload,
    encode_causal_posterior_with_payload,
};
pub use transport_interference_wire::{
    ExposureProbabilityMethodWire, InterferenceEstimateWire, NotCertifiedCertificateWire,
    PopulationFactorWire, RandomizationContrastWire, TransportCertificateWire,
    TransportEffectEstimateWire, TransportFormulaWire, TransportIdentificationWire,
    TransportOverlapDiagnosticWire, interference_estimate_from_wire, interference_estimate_to_wire,
    transport_effect_from_wire, transport_effect_to_wire, transport_identification_from_wire,
    transport_identification_to_wire,
};

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, VERSION, ValueType,
    };
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;

    #[test]
    fn schema_and_dag_artifact_round_trip() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();

        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

        let schema_bytes = to_cbor(&schema_to_wire(&schema)).unwrap();
        let dag_bytes = to_cbor(&dag_to_wire(&dag).unwrap()).unwrap();

        let schema_desc = section_descriptor("schema", "application/cbor", &schema_bytes);
        let dag_desc = section_descriptor("dag", "application/cbor", &dag_bytes);

        let artifact = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: STABLE_FORMAT,
                minimum_reader_version: STABLE_FORMAT,
                artifact_kind: ArtifactKind::SchemaGraph,
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "test-schema-graph".into(),
                sections: vec![schema_desc, dag_desc],
                provenance: ProvenanceWire { note: "roundtrip".into() },
            },
            sections: vec![
                SectionBytes::new("schema", schema_bytes),
                SectionBytes::new("dag", dag_bytes),
            ],
        };

        let mut buf = Vec::new();
        artifact.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        assert_eq!(decoded.manifest.artifact_id, "test-schema-graph");

        let schema_wire: SchemaWire = from_cbor(&decoded.sections[0].data).unwrap();
        assert_eq!(schema_wire.variable_names(), vec!["x", "y"]);
        let dag_wire: DagWire = from_cbor(&decoded.sections[1].data).unwrap();
        let dag2 = dag_from_wire(&dag_wire).unwrap();
        assert!(dag2.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn analysis_trace_artifact_round_trips_assumptions_and_derivation() {
        use std::sync::Arc;

        use antecedent_core::{
            Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
            AssumptionStatus,
        };

        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::CausalMarkov,
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("backdoor") },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        let trace = AnalysisTraceWire {
            assumptions: assumptions_to_wire(&assumptions),
            derivation: vec![DerivationStepWire {
                rule: "backdoor.criterion".into(),
                detail: "Z blocks all backdoor paths".into(),
            }],
            method: "backdoor.adjustment".into(),
            adjustment_set: vec![2],
            support_status: None,
            allowlist_reason: None,
            allowlist_parent: None,
        };
        let bytes = to_cbor(&trace).unwrap();
        let desc = section_descriptor("analysis.trace", "application/cbor", &bytes);
        let artifact = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: STABLE_FORMAT,
                minimum_reader_version: STABLE_FORMAT,
                artifact_kind: ArtifactKind::AnalysisTrace,
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "test-analysis-trace".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "trace".into() },
            },
            sections: vec![SectionBytes::new("analysis.trace", bytes)],
        };
        let mut buf = Vec::new();
        artifact.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        let round: AnalysisTraceWire = from_cbor(&decoded.sections[0].data).unwrap();
        assert_eq!(round.method, "backdoor.adjustment");
        assert_eq!(round.adjustment_set, vec![2]);
        assert_eq!(round.assumptions.len(), 1);
        assert_eq!(round.assumptions[0].assumption, AssumptionTagWire::CausalMarkov);
        assert_eq!(round.derivation[0].rule, "backdoor.criterion");
    }
}

/// Transport grid artifact records and encoding.
pub mod transport_grid_wire;

/// Categorical-treatment artifact (`categorical_treatment_v1`) and its recomputing consumer.
pub mod categorical_treatment_artifact;
/// Dose-grid functional artifact (`dose_grid_functional_row_v1`) and its recomputing consumer.
pub mod dose_grid_functional_artifact;
/// Effect-constancy artifact (`effect_constancy_v1`) and its recomputing consumer.
pub mod effect_constancy_artifact;
/// Latent-class regime-effects artifact (`latent_class_effects_v1`) and its refitting consumer.
pub mod latent_class_artifact;
/// Learned continuous-outcome trial transport artifact verification.
pub mod learned_continuous_artifact;
/// Learner-backed trial transport artifact verification.
pub mod learned_trial_wire;
/// Mechanism discrepancy artifact (`mechanism_discrepancy_v1`) and its recomputing consumer.
pub mod mechanism_discrepancy_artifact;
/// Adapter from a marginal sensitivity model result into the sensitivity artifact.
pub mod msm_sensitivity_adapter;
/// Nonlinear-mediation artifact (`nonlinear_mediation_v1`) and its re-estimating consumer.
pub mod nonlinear_mediation_artifact;
/// Ordered-response recovery artifact (`ordered_response_recovery_v1`) and its re-deciding consumer.
pub mod recovery_chain_artifact;
/// Sensitivity-surface artifact (`sensitivity_decision_v1`) and its recomputing consumer.
pub mod sensitivity_artifact;
/// Smoothed dose-response transport grid artifact verification.
pub mod smoothed_dose_artifact;
/// Joint vector-treatment artifact (`vector_treatment_v1`) and its recomputing consumer.
pub mod vector_treatment_artifact;

#[cfg(test)]
mod closed_wire_fields {
    use serde::Deserialize;

    fn zeros() -> serde_json::Value {
        serde_json::Value::Array(vec![serde_json::json!(0); 32])
    }

    fn accept_then_reject_unknown<T: for<'de> Deserialize<'de>>(historical: &serde_json::Value) {
        serde_json::from_value::<T>(historical.clone()).expect("historical payload");
        let mut with_unknown = historical.clone();
        with_unknown
            .as_object_mut()
            .expect("object payload")
            .insert("not_a_historical_field".to_owned(), serde_json::Value::Null);
        assert!(
            serde_json::from_value::<T>(with_unknown).is_err(),
            "unknown field must be rejected"
        );
    }

    #[test]
    fn historical_payloads_decode_and_unknown_fields_do_not() {
        let digest = zeros();
        accept_then_reject_unknown::<crate::ContractIdentitiesWire>(&serde_json::json!({
            "target": digest.clone(),
            "identification": digest.clone(),
            "program": digest.clone(),
            "inference_binding": digest.clone(),
            "observation": digest.clone(),
            "data_snapshot": digest,
        }));
        accept_then_reject_unknown::<crate::wire::GraphCpdagWire>(&serde_json::json!({
            "node_count": 2,
            "directed": [[0, 1]],
            "undirected": [],
        }));
        accept_then_reject_unknown::<crate::analysis_wire::ScoreColumnWire>(&serde_json::json!({
            "arm": 1,
            "threshold": null,
        }));
        accept_then_reject_unknown::<crate::plan_wire::LogicalAnalysisPlanWire>(
            &serde_json::json!({
                "plan_id": "p",
                "data_classification": "public",
                "discovery_algorithm": null,
                "graph_review_required": false,
                "identifier": null,
                "estimator": null,
                "validation_suite": null,
                "query_variables": [],
            }),
        );
        accept_then_reject_unknown::<crate::trace::DerivationStepWire>(&serde_json::json!({
            "rule": "backdoor",
            "detail": "adjust",
        }));
        let temporal: crate::prior_bank::TemporalCoordinates =
            serde_json::from_value(serde_json::json!({
                "treatment_lags": [1],
                "horizon": 1,
            }))
            .expect("omitted coefficient_names");
        assert!(temporal.coefficient_names.is_empty());
        accept_then_reject_unknown::<crate::prior_bank::TemporalCoordinates>(&serde_json::json!({
            "treatment_lags": [1],
            "horizon": 1,
        }));
        accept_then_reject_unknown::<crate::mechanism_wire::ParentBasisWire>(&serde_json::json!({
            "n_parents": 0,
            "centers": [],
            "scales": [],
            "knots": [],
            "terms": [],
        }));
        accept_then_reject_unknown::<crate::response_wire::SupportRegionWire>(&serde_json::json!({
            "minima": [0.0],
            "maxima": [1.0],
        }));
        let diagnostic: crate::response_wire::SupportDiagnosticWire =
            serde_json::from_value(serde_json::json!({
                "id": "overlap",
                "values": [1.0],
                "detail": "ok",
            }))
            .expect("omitted scope");
        assert!(diagnostic.scope.is_none());
        accept_then_reject_unknown::<crate::response_wire::SupportDiagnosticWire>(
            &serde_json::json!({
                "id": "overlap",
                "values": [1.0],
                "detail": "ok",
            }),
        );
    }

    #[test]
    fn artifact_kind_keeps_the_forward_compat_other_tag() {
        let known: crate::wire::ArtifactKind = serde_json::from_str(r#""schema_graph""#).unwrap();
        assert_eq!(known, crate::wire::ArtifactKind::SchemaGraph);
        let future: crate::wire::ArtifactKind =
            serde_json::from_str(r#"{"other":"future_kind"}"#).unwrap();
        assert_eq!(future, crate::wire::ArtifactKind::Other("future_kind".into()));
    }
}
