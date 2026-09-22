//! artifact migration conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, VERSION,
    ValueType,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::{
    AnalysisTraceWire, ArtifactKind, ArtifactManifest, CausalPosteriorWire, DerivationStepWire,
    EncodedArtifact, FormatVersion, ModelBundleEncode, ModelBundleHeaderWire, ModelKindWire,
    PosteriorQuantityWire, ProvenanceWire, STABLE_FORMAT, SchemaWire, SchemaWireV01, SectionBytes,
    SemanticVersion, assumptions_to_wire, dag_to_wire, decode_model_bundle, encode_model_bundle,
    encode_posterior_artifact, from_cbor, migrate_artifact, read_and_migrate, schema_to_wire,
    section_descriptor, to_cbor,
};
use antecedent_model::{CompiledMechanismStore, MechanismSlot};
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/interchange/artifact_migrate")
}

#[test]
fn conformance_migrate_stable_kinds_round_trip() {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    // Validate the fixture against the live constant rather than a second hardcoded
    // copy of it. The previous form asserted `minor == 2` against a fixture that also
    // said 2, so it agreed with itself while both drifted a full version behind
    // `STABLE_FORMAT` (now 0.4) — the drift this fixture exists to catch.
    assert_eq!(v["stable_format"]["major"], u64::from(STABLE_FORMAT.major));
    assert_eq!(v["stable_format"]["minor"], u64::from(STABLE_FORMAT.minor));

    let artifacts = [
        ("schema_graph", schema_graph_artifact()),
        ("analysis_trace", analysis_trace_artifact()),
        ("causal_posterior", posterior_artifact()),
        ("model_bundle", model_bundle_artifact()),
    ];
    let fixture_kinds: Vec<&str> =
        v["kinds"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
    let tested_kinds: Vec<&str> = artifacts.iter().map(|(kind, _)| *kind).collect();
    assert_eq!(fixture_kinds, tested_kinds);

    for (_, art) in artifacts {
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
        assert_eq!(again.manifest.format_version, STABLE_FORMAT);
    }
}

#[test]
fn conformance_migrate_0_1_schema_to_stable() {
    let v01 = SchemaWireV01 { variable_names: vec!["x".into(), "y".into()] };
    let payload = to_cbor(&v01).unwrap();
    let art = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: FormatVersion { major: 0, minor: 1 },
            minimum_reader_version: FormatVersion { major: 0, minor: 1 },
            artifact_kind: ArtifactKind::SchemaGraph,
            library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
            artifact_id: "v01-schema".into(),
            sections: vec![section_descriptor("schema", "application/cbor", &payload)],
            provenance: ProvenanceWire { note: "v01".into() },
        },
        sections: vec![SectionBytes::new("schema", payload)],
    };
    let migrated = migrate_artifact(art).unwrap();
    assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
    let schema: SchemaWire = from_cbor(&migrated.sections[0].data).unwrap();
    assert_eq!(schema.variable_names(), vec!["x".to_string(), "y".to_string()]);
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
            library_version: SemanticVersion::from_crate_version(VERSION)
                .expect("CARGO_PKG_VERSION"),
            artifact_id: "p12-schema".into(),
            sections: vec![
                section_descriptor("schema", "application/cbor", &schema_bytes),
                section_descriptor("dag", "application/cbor", &dag_bytes),
            ],
            provenance: ProvenanceWire { note: "release".into() },
        },
        sections: vec![
            SectionBytes::new("schema", schema_bytes),
            SectionBytes::new("dag", dag_bytes),
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
        support_status: None,
        allowlist_reason: None,
        allowlist_parent: None,
    };
    let bytes = to_cbor(&trace).unwrap();
    EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::AnalysisTrace,
            library_version: SemanticVersion::from_crate_version(VERSION)
                .expect("CARGO_PKG_VERSION"),
            artifact_id: "p12-trace".into(),
            sections: vec![section_descriptor("analysis.trace", "application/cbor", &bytes)],
            provenance: ProvenanceWire { note: "release".into() },
        },
        sections: vec![SectionBytes::new("analysis.trace", bytes)],
    }
}

fn posterior_artifact() -> EncodedArtifact {
    let meta = CausalPosteriorWire {
        quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
        n_draws: 3,
        mean: vec![1.0],
        sd: vec![0.1],
        // Type-7 quantiles of the draws (0.9, 1.0, 1.1).
        q025: vec![0.905],
        q975: vec![1.095],
        identification: "NonparametricallyIdentified".into(),
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        backend_id: "laplace".into(),
        converged: true,
        hessian_condition: 1.0,
        draws_encoding: "f64_le_colmajor".into(),
        treatment_contrast: None,
    };
    encode_posterior_artifact(&meta, &[0.9, 1.0, 1.1], "p12-post", VERSION).unwrap()
}

fn model_bundle_artifact() -> EncodedArtifact {
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
    let mechanisms = CompiledMechanismStore {
        slots: vec![
            MechanismSlot::Constant { value: 0.0 },
            MechanismSlot::LinearGaussian {
                intercept: 0.1,
                coeffs: Arc::from([0.5f64]),
                sigma: 1.0,
            },
        ]
        .into(),
    };
    encode_model_bundle(&ModelBundleEncode {
        header: ModelBundleHeaderWire {
            model_kind: ModelKindWire::Scm,
            label: Some("p12-bundle".into()),
        },
        schema: &schema,
        dag: &dag,
        mechanisms: &mechanisms,
        artifact_id: "p12-bundle",
        contrast: None,
        query: None,
        analysis_trace: None,
        identification: None,
        estimate: None,
        refutations: None,
        logical_plan: None,
        physical_plan: None,
        performance: None,
        diagnostics: None,
        provenance: None,
        posterior: None,
        discovery: None,
    })
    .unwrap()
}

#[test]
fn conformance_migrate_0_2_model_bundle_to_stable() {
    let mut art = model_bundle_artifact();
    art.manifest.format_version = FormatVersion { major: 0, minor: 2 };
    art.manifest.minimum_reader_version = FormatVersion { major: 0, minor: 2 };
    let migrated = migrate_artifact(art).unwrap();
    assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
    assert_eq!(migrated.manifest.minimum_reader_version, STABLE_FORMAT);
    let bundle = decode_model_bundle(&migrated).unwrap();
    assert_eq!(bundle.header.model_kind, ModelKindWire::Scm);
    assert_eq!(bundle.schema.len(), 2);
    assert_eq!(bundle.dag.node_count(), 2);
    assert!(matches!(bundle.mechanisms.slots[1], MechanismSlot::LinearGaussian { .. }));
}

#[test]
fn wire_round_trip_still_decodes() {
    let art = schema_graph_artifact();
    let mut buf = Vec::new();
    art.write_to(&mut buf).unwrap();
    let migrated = read_and_migrate(buf.as_slice()).unwrap();
    let _: SchemaWire = from_cbor(&migrated.sections[0].data).unwrap();
}

/// The format-0.5 identified-set interval of a class-aware temporal Pulse
/// result survives the composite analysis-result artifact.
#[test]
fn class_aware_identified_set_interval_survives_the_result_artifact() {
    use antecedent::{InferenceMode, RefuteSuite, Study};
    use antecedent_core::{
        CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::TemporalCpdag;

    // t = 0.5 z + u, y_t = 0.8 t_{t-1} + 0.6 z_{t-1} + e; `z_{t-1} — t_{t-1}` is
    // undirected, so the completions adjust z (0.8) or not (a larger total).
    let n = 240;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut draw = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[allow(clippy::cast_precision_loss)]
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    };
    let z: Vec<f64> = (0..n).map(|_| 2.0 * draw()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.5 * z + 2.0 * draw()).collect();
    let mut y = vec![0.0; n];
    for i in 1..n {
        y[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + draw();
    }
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(z1, t1).unwrap();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let result = Study::series(data)
        .graph(graph)
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(60)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let interval = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set_interval)
        .expect("a two-completion class publishes an identified-set interval");

    let query_wire = serde_json::json!({"response": {
        "functional": {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
        "target_population": "all_observed", "observation": "complete", "observation_assumptions": []
    }});
    let mut wire: antecedent_io::AnalysisResultWire = serde_json::from_value(serde_json::json!({
        "query": query_wire,
        "identification": {"status": "nonparametrically_identified", "query": query_wire,
            "estimands": [], "arena": {"var_sets": [], "interventions": [], "lists": [], "nodes": []},
            "derivation": [], "required_assumptions": [], "diagnostics": [],
            "candidates_examined": 0, "sets_returned": 0},
        "estimate": null, "standard_error": null, "assumptions": [], "diagnostics": [],
        "refutations": [], "response": null, "posterior_artifact": null,
        "mediation_grid": null, "structural_response": {
            "weight_basis": "completion_enumeration", "atoms": [], "identified_mass": 1.0,
            "unidentified_mass": 0.0, "unevaluable_mass": 0.0, "identified_set": null,
            "conditional_on_identified": null, "full_mass_scope": true, "truncated_atoms": 0}
    }))
    .unwrap();
    wire.structural_response.as_mut().unwrap().identified_set_interval =
        Some(antecedent_io::identified_set_interval_to_wire(&interval).unwrap());
    let artifact = antecedent_io::encode_analysis_result_artifact(
        &wire,
        vec!["t".into(), "y".into(), "z".into()],
        "identified-set",
    )
    .unwrap();
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    let (migrated, _, decoded) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
    let restored = antecedent_io::identified_set_interval_from_wire(
        decoded.structural_response.as_ref().unwrap().identified_set_interval.as_ref().unwrap(),
    )
    .unwrap();
    assert_eq!(restored, interval);
}
