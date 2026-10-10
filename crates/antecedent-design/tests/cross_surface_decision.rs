//! Cross-surface decision contract, joint source and result: Python-built bytes
//! consumed here, and Rust-built bytes written for the Python consumer.
//!
//! Fixtures live in `conformance/cross_surface/`. A missing `py_*` fixture fails
//! the test: run `python python/tests/generate_cross_surface_fixtures.py`.
//! To refresh the `rust_*` fixtures:
//! `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test
//! cross_surface_decision -- --ignored regenerate_rust_fixtures`.
//!
//! Truth is the enumerated `P * Q` fixture of `decision_artifact.rs`: rows
//! `p = [1, 3, 2, 0]`, `q = [4, 0, 2, 6]`, safe = 3, so `EU(risky) = 2`,
//! `EU(safe) = 3`, `EVPI = 0.5` and the verdict is uniquely optimal `safe`.
//! Every consumer-side identity is built from constants here.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{Verdict, evaluate_contract};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cross_surface")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing cross-surface fixture {} ({error}); run \
             `python python/tests/generate_cross_surface_fixtures.py` from the repo root \
             (Rust-built fixtures: ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design \
             --test cross_surface_decision -- --ignored regenerate_rust_fixtures)",
            path.display()
        )
    })
}

fn flip_last_byte(bytes: &[u8]) -> Vec<u8> {
    let mut copy = bytes.to_vec();
    *copy.last_mut().expect("non-empty artifact") ^= 0xFF;
    copy
}

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn columns() -> Vec<ScientificQuantity> {
    vec![quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("safe", "do(a=0)")]
}

fn source_identity() -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

fn source() -> DistributionArtifact {
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: source_identity(),
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// The consumer re-declares the contract from constants.
fn contract() -> DecisionContract {
    let cols = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[0].clone(), cols[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Policy,
                inputs: vec![cols[2].clone()],
                utility: UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "q-cap".into(),
            expr: UtilityExpr::Input(1),
            bound: 5.0,
            min_probability: 0.75,
            units: "units".into(),
            applies_to: vec!["risky".into()],
        }],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn python_built_contract_source_and_result_load_and_replay() {
    let consumer_contract = contract();
    let identity = consumer_contract.identity().unwrap();

    let contract_artifact =
        DecisionContractArtifact::from_bytes(&read_fixture("py_decision_contract.bin"), &identity)
            .unwrap();
    assert_eq!(contract_artifact.contract(), &consumer_contract);

    let loaded_source = DistributionArtifact::from_bytes(
        &read_fixture("py_decision_source.bin"),
        &source_identity(),
    )
    .unwrap();
    let digest = source_digest(&loaded_source);
    assert_eq!(digest, source_digest(&source()));

    let result = DecisionResultArtifact::from_bytes(
        &read_fixture("py_decision_result.bin"),
        &identity,
        &digest,
    )
    .unwrap();
    result.replay(contract_artifact.contract(), &loaded_source).unwrap();
    assert_eq!(result.result().verdict, Verdict::UniquelyOptimal("safe".into()));
    assert!((result.result().evpi.unwrap() - 0.5).abs() < 1e-12);
    assert!((result.result().actions[0].expected_utility - 2.0).abs() < 1e-12);
    assert!((result.result().actions[1].expected_utility - 3.0).abs() < 1e-12);
}

#[test]
fn python_identity_file_agrees_with_rust_constants_but_is_never_trusted() {
    // A cross-check only: every load above and below uses identities built here.
    let json: serde_json::Value =
        serde_json::from_slice(&read_fixture("py_decision.identities.json")).unwrap();
    assert_eq!(json["contract_identity"], contract().identity().unwrap());
    assert_eq!(json["source_digest"], source_digest(&source()));
}

#[test]
fn python_built_artifacts_refuse_tampered_bytes_and_other_inputs() {
    let identity = contract().identity().unwrap();
    let contract_bytes = read_fixture("py_decision_contract.bin");
    let source_bytes = read_fixture("py_decision_source.bin");
    let result_bytes = read_fixture("py_decision_result.bin");
    let digest = source_digest(&source());

    assert!(
        DecisionContractArtifact::from_bytes(&flip_last_byte(&contract_bytes), &identity).is_err()
    );
    assert!(
        DecisionContractArtifact::from_bytes(&contract_bytes, "0".repeat(64).as_str()).is_err()
    );
    assert!(
        DistributionArtifact::from_bytes(&flip_last_byte(&source_bytes), &source_identity())
            .is_err()
    );
    assert!(
        DecisionResultArtifact::from_bytes(&flip_last_byte(&result_bytes), &identity, &digest)
            .is_err()
    );
    assert!(
        DecisionResultArtifact::from_bytes(&result_bytes, &identity, "0".repeat(64).as_str())
            .is_err()
    );
}

/// Writes the Rust-built fixtures the Python tests consume.
#[test]
#[ignore = "writes conformance/cross_surface; set ANTECEDENT_WRITE_FIXTURES=1"]
fn regenerate_rust_fixtures() {
    assert_eq!(
        std::env::var("ANTECEDENT_WRITE_FIXTURES").as_deref(),
        Ok("1"),
        "set ANTECEDENT_WRITE_FIXTURES=1 to write fixtures"
    );
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let c = contract();
    let s = source();
    let result = DecisionResultArtifact::new(evaluate_contract(&c, &s).unwrap(), &s);
    std::fs::write(
        dir.join("rust_decision_contract.bin"),
        DecisionContractArtifact::new(c).unwrap().to_bytes("decision-contract").unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("rust_decision_source.bin"), s.to_bytes("source").unwrap()).unwrap();
    std::fs::write(
        dir.join("rust_decision_result.bin"),
        result.to_bytes("decision-result").unwrap(),
    )
    .unwrap();
}
