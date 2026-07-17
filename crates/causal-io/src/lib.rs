//! Versioned artifact IO for causal-library.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod analysis_wire;
pub mod arrow_section;
pub mod container;
pub mod contrast_wire;
pub mod convert;
pub mod discovery_wire;
pub mod error;
pub mod expr_wire;
pub mod graph_dot;
pub mod graph_gml;
pub mod graph_json;
pub mod graph_networkx;
pub mod mechanism_wire;
pub mod migrate;
pub mod model_bundle;
pub mod plan_wire;
pub mod posterior;
pub mod posterior_convert;
pub mod provenance_wire;
pub mod query_wire;
pub mod trace;
pub mod wire;

pub use analysis_wire::{
    DiagnosticWire, EffectEstimateWire, IdentifiedEstimandWire, IdentificationResultWire,
    RefutationReportWire, diagnostic_from_wire, diagnostic_to_wire, effect_estimate_from_wire,
    effect_estimate_to_wire, identification_from_wire, identification_to_wire,
    refutation_from_wire, refutation_to_wire,
};
pub use arrow_section::{ARROW_IPC_CONTENT_TYPE, arrow_ipc_section};
pub use container::{
    AUTO_COMPRESS_MAX_RATIO, AUTO_COMPRESS_MIN_BYTES, ArtifactManifest, COMPRESSION_ZSTD,
    CONTAINER_VERSION, CompressPolicy, EncodedArtifact, MAGIC, SectionBytes, pack_section,
    section_descriptor, section_descriptor_with_policy,
};
pub use contrast_wire::{ContrastBundleWire, RecordedContrastWire};
pub use convert::{
    dag_from_wire, dag_to_wire, from_cbor, schema_from_wire, schema_to_wire, schema_wire_from_v01,
    to_cbor,
};
pub use discovery_wire::{
    DiscoveryHeaderWire, EdgeEvidenceWire, LaggedLinkWire, TemporalGraphWire, TemporalNodeKeyWire,
    discovery_dag_from_sections, discovery_dag_sections, temporal_dag_from_wire,
    temporal_dag_to_wire,
};
pub use error::IoError;
pub use expr_wire::{ExprArenaWire, ExprNodeWire, expr_arena_from_wire, expr_arena_to_wire};
pub use graph_dot::{dag_from_dot, dag_to_dot, dag_wire_from_dot, dag_wire_to_dot};
pub use graph_gml::{dag_from_gml, dag_to_gml, dag_wire_from_gml, dag_wire_to_gml};
pub use graph_json::{DagJson, dag_from_json, dag_json_from_str, dag_to_json};
pub use graph_networkx::{
    NetworkXAdjacency, NetworkXNodeLink, dag_from_networkx_adjacency, dag_from_networkx_node_link,
    dag_to_networkx_adjacency, dag_to_networkx_node_link,
};
pub use mechanism_wire::{
    MechanismSlotWire, MechanismStoreWire, ModelKindWire, mechanisms_from_wire, mechanisms_to_wire,
};
pub use migrate::{
    STABLE_FORMAT, SUPPORTED_SOURCE_FORMATS, is_supported_source, migrate_artifact,
    read_and_migrate,
};
pub use model_bundle::{
    ModelBundle, ModelBundleEncode, ModelBundleHeaderWire, decode_model_bundle,
    encode_model_bundle,
};
pub use plan_wire::{
    ExecutionPerformanceWire, LogicalAnalysisPlanWire, PhysicalExecutionPlanWire,
    logical_plan_from_wire, logical_plan_to_wire, performance_from_wire, performance_to_wire,
    physical_plan_from_wire, physical_plan_to_wire,
};
pub use posterior::{
    CausalPosteriorWire, PosteriorQuantityWire, decode_posterior_artifact,
    encode_posterior_artifact,
};
pub use posterior_convert::{
    decode_causal_posterior_bytes, encode_causal_posterior, encode_causal_posterior_bytes,
};
pub use provenance_wire::{
    ProvenanceGraphWire, ProvenanceNodeWire, provenance_from_wire, provenance_to_wire,
};
pub use query_wire::{
    CausalQueryWire, InterventionalDistributionQueryWire, InterventionWire,
    PathSpecificEffectQueryWire, SetInterventionWire, TargetPopulationWire, ValueWire,
    causal_query_from_wire, causal_query_to_wire, interventional_distribution_from_wire,
    interventional_distribution_to_wire, path_specific_from_wire, path_specific_to_wire,
};
pub use trace::{
    AnalysisTraceWire, AssumptionRecordWire, AssumptionTagWire, DerivationStepWire,
    assumptions_to_wire,
};
pub use wire::{
    ArtifactKind, DagWire, FormatVersion, MeasurementSpecWire, ProvenanceWire, SchemaWire,
    SchemaWireV01, SectionDescriptor, SemanticVersion, ValueTypeWire, VariableSchemaWire,
};

#[cfg(test)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, VERSION, ValueType,
    };
    use causal_graph::{Dag, DenseNodeId};

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
                SectionBytes { id: "schema".into(), data: schema_bytes },
                SectionBytes { id: "dag".into(), data: dag_bytes },
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

        use causal_core::{
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
            sections: vec![SectionBytes { id: "analysis.trace".into(), data: bytes }],
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
