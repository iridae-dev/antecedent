//! Independent-process consumption of a bounded 2.3 joint distribution artifact.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn expected_identity() -> DistributionIdentity {
    let quantities = ["x", "y"].map(|name| ScientificQuantity {
        variable_id: format!("schema:{name}"),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    });
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study".into(),
            provider_id: "provider".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snapshot".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

#[test]
fn fresh_process_consumer_recomputes_enumerated_joint_truth() {
    const PATH_ENV: &str = "ANTECEDENT_23_JOINT_TEST_PATH";
    if let Ok(path) = std::env::var(PATH_ENV) {
        let bytes = std::fs::read(path).unwrap();
        // Built from the independently retained contract constants above, not
        // from identity fields decoded from the artifact.
        let loaded = DistributionArtifact::from_bytes(&bytes, &expected_identity()).unwrap();
        assert!((loaded.mean(0).unwrap() - 0.5).abs() < 1e-12);
        assert!((loaded.mean(1).unwrap() - 1.0).abs() < 1e-12);
        assert!((loaded.covariance(0, 1).unwrap() - 0.5).abs() < 1e-12);
        assert!((loaded.joint_expectation(0, 1, |x, y| x * y).unwrap() - 1.0).abs() < 1e-12);
        return;
    }

    let identity = expected_identity();
    let artifact = DistributionArtifact::new(
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
        },
        vec![0.0, 0.0, 1.0, 2.0],
    )
    .unwrap();
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path =
        std::env::temp_dir().join(format!("antecedent-joint-{}-{unique}.bin", std::process::id()));
    std::fs::write(&path, artifact.to_bytes("joint-fixture").unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("fresh_process_consumer_recomputes_enumerated_joint_truth")
        .env(PATH_ENV, &path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}
