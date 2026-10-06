//! F21 composition parity for the aligned joint distribution.
//!
//! Both surfaces build the enumerated two-coordinate law of two equally weighted
//! aligned draws (0, 0) and (1, 2): means (1/2, 1), covariance 1/2 and E[XY] = 1.
//! Python-built bytes are consumed here and Rust-built bytes are written for the
//! Python consumer. Every identity below is written from constants.
//!
//! A missing `py_joint_law.bin` fails the test: run
//! `python python/tests/generate_cross_surface_fixtures.py` from the repo root.
//! To refresh the Rust-built fixture:
//! `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-io --test
//! cross_surface_joint_law -- --ignored regenerate_rust_fixtures`.

use antecedent_core::{QuantityRole, ScientificQuantity};
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
             `python python/tests/generate_cross_surface_fixtures.py` from the repo root",
            path.display()
        )
    })
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn quantities(second: &str) -> Vec<ScientificQuantity> {
    ["x", second]
        .iter()
        .map(|name| ScientificQuantity {
            variable_id: (*name).into(),
            variable_name: (*name).into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        })
        .collect()
}

fn identity(
    meaning: DistributionMeaningWire,
    alignment: DrawAlignment,
    second: &str,
) -> DistributionIdentity {
    DistributionIdentity::new(
        meaning,
        &quantities(second),
        alignment,
        DistributionProvenance {
            source_id: "enumerated-law".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "law-snapshot".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

fn expected_identity() -> DistributionIdentity {
    identity(DistributionMeaningWire::InterventionalPredictive, DrawAlignment::Joint, "y")
}

fn law(identity: DistributionIdentity) -> DistributionArtifact {
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![0.0, 0.0, 1.0, 2.0],
    )
    .unwrap()
}

fn assert_enumerated_truth(artifact: &DistributionArtifact) {
    assert!(near(artifact.mean(0).unwrap(), 0.5));
    assert!(near(artifact.mean(1).unwrap(), 1.0));
    assert!(near(artifact.covariance(0, 1).unwrap(), 0.5));
    assert!(near(artifact.joint_expectation(0, 1, |x, y| x * y).unwrap(), 1.0));
}

/// F21 positive: the Python-built law reports the enumerated truth under an identity
/// built here from constants, and the Rust-built twin has the same identity.
#[test]
fn f21_python_built_joint_law_reports_the_enumerated_truth_under_constants() {
    let loaded =
        DistributionArtifact::from_bytes(&read_fixture("py_joint_law.bin"), &expected_identity())
            .unwrap();
    assert_enumerated_truth(&loaded);
    assert_eq!(loaded.metadata().identity, expected_identity());
    assert_eq!(loaded.metadata().trust, DistributionTrust::Unverified);
    assert_eq!(loaded.metadata().calibration, DistributionCalibration::Exact);
    let rust = law(expected_identity());
    assert_eq!(rust.metadata().identity, loaded.metadata().identity);
    assert_eq!(rust.draws(), loaded.draws());
}

/// F21 negative: a changed semantic ID or a wrong distribution meaning refuses, and
/// independent marginals refuse the joint operations with the shared detail.
#[test]
fn f21_changed_semantic_id_wrong_meaning_and_marginals_refuse() {
    let bytes = read_fixture("py_joint_law.bin");
    for changed in [
        identity(DistributionMeaningWire::InterventionalPredictive, DrawAlignment::Joint, "z"),
        identity(DistributionMeaningWire::Bootstrap, DrawAlignment::Joint, "y"),
        identity(DistributionMeaningWire::CausalFunctionalPosterior, DrawAlignment::Joint, "y"),
    ] {
        let error = DistributionArtifact::from_bytes(&bytes, &changed).unwrap_err();
        assert!(error.to_string().contains("aligned_joint_draws.identity_expected"), "{error}");
    }
    let marginals = law(identity(
        DistributionMeaningWire::InterventionalPredictive,
        DrawAlignment::IndependentMarginals,
        "y",
    ));
    let error = marginals.covariance(0, 1).unwrap_err();
    assert!(error.to_string().contains("aligned_joint_draws.marginals_not_joint"), "{error}");
}

/// F21 artifact: both directions round-trip through the one bounded wire format; the
/// Rust-built bytes are written for Python and the Python-built bytes survive here.
#[test]
fn f21_both_directions_round_trip_through_one_wire_format() {
    let python = read_fixture("py_joint_law.bin");
    let loaded = DistributionArtifact::from_bytes(&python, &expected_identity()).unwrap();
    let again = loaded.to_bytes("joint-law").unwrap();
    let reloaded = DistributionArtifact::from_bytes(&again, &expected_identity()).unwrap();
    assert_enumerated_truth(&reloaded);
    assert_eq!(reloaded.draws(), loaded.draws());
    // A flipped byte of the Python-built artifact is refused.
    let mut tampered = python;
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    assert!(DistributionArtifact::from_bytes(&tampered, &expected_identity()).is_err());
}

/// Writes the Rust-built fixture the Python tests consume.
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
    std::fs::write(
        dir.join("rust_joint_law.bin"),
        law(expected_identity()).to_bytes("joint-law").unwrap(),
    )
    .unwrap();
}
