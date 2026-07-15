//! artifact migration conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, VERSION,
    ValueType,
};
use causal_graph::{Dag, DenseNodeId};
use causal_io::{
    AnalysisTraceWire, ArtifactKind, ArtifactManifest, CausalPosteriorWire, DerivationStepWire,
    EncodedArtifact, FormatVersion, PosteriorQuantityWire, ProvenanceWire, STABLE_FORMAT,
    SectionBytes, SemanticVersion, assumptions_to_wire, dag_to_wire, encode_posterior_artifact,
    from_cbor, migrate_artifact, read_and_migrate, schema_to_wire, section_descriptor, to_cbor,
};
use serde_json::Value;
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/interchange/artifact_migrate")
}

#[test]
fn conformance_migrate_three_kinds() {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(v["stable_format"]["major"], 0);
    assert_eq!(v["stable_format"]["minor"], 1);

    for art in [schema_graph_artifact(), analysis_trace_artifact(), posterior_artifact()] {
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let migrated = read_and_migrate(buf.as_slice()).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(migrated.sections.len(), art.sections.len());
        for (a, b) in art.sections.iter().zip(migrated.sections.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.data, b.data);
        }
        let again = migrate_artifact(migrated).unwrap();
        assert_eq!(again.manifest.format_version, FormatVersion { major: 0, minor: 1 });
    }
}

fn schema_graph_artifact() -> EncodedArtifact {
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
    EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::SchemaGraph,
            library_version: SemanticVersion::from_crate_version(VERSION).expect("CARGO_PKG_VERSION"),
            artifact_id: "p12-schema".into(),
            sections: vec![
                section_descriptor("schema", "application/cbor", &schema_bytes),
                section_descriptor("dag", "application/cbor", &dag_bytes),
            ],
            provenance: ProvenanceWire { note: "release".into() },
        },
        sections: vec![
            SectionBytes { id: "schema".into(), data: schema_bytes },
            SectionBytes { id: "dag".into(), data: dag_bytes },
        ],
    }
}

fn analysis_trace_artifact() -> EncodedArtifact {
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
            detail: "Z blocks".into(),
        }],
        method: "backdoor.adjustment".into(),
        adjustment_set: vec![2],
    };
    let bytes = to_cbor(&trace).unwrap();
    EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::AnalysisTrace,
            library_version: SemanticVersion::from_crate_version(VERSION).expect("CARGO_PKG_VERSION"),
            artifact_id: "p12-trace".into(),
            sections: vec![section_descriptor("analysis.trace", "application/cbor", &bytes)],
            provenance: ProvenanceWire { note: "release".into() },
        },
        sections: vec![SectionBytes { id: "analysis.trace".into(), data: bytes }],
    }
}

fn posterior_artifact() -> EncodedArtifact {
    let meta = CausalPosteriorWire {
        quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
        n_draws: 2,
        mean: vec![1.0],
        sd: vec![0.1],
        q025: vec![0.9],
        q975: vec![1.1],
        identification: "NonparametricallyIdentified".into(),
        unidentified_mass: 0.0,
        backend_id: "laplace".into(),
        converged: true,
        hessian_condition: 1.0,
        draws_encoding: "f64_le_colmajor".into(),
    };
    encode_posterior_artifact(&meta, &[1.0, 1.0], "p12-post", VERSION).unwrap()
}

#[test]
fn wire_round_trip_still_decodes() {
    let art = schema_graph_artifact();
    let mut buf = Vec::new();
    art.write_to(&mut buf).unwrap();
    let migrated = read_and_migrate(buf.as_slice()).unwrap();
    let _: causal_io::SchemaWire = from_cbor(&migrated.sections[0].data).unwrap();
}
