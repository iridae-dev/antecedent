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
        let mut changed = loaded.metadata().clone();
        changed.identity.snapshot_id = "other-snapshot".into();
        let resealed = DistributionArtifact::new(changed, loaded.draws().to_vec())
            .unwrap()
            .to_bytes("joint-fixture")
            .unwrap();
        assert!(DistributionArtifact::from_bytes(&resealed, &expected_identity()).is_err());
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
            legacy_bindings: None,
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

fn enumerated_metadata(identity: DistributionIdentity) -> DistributionMetadata {
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
    }
}

const ENUMERATED_DRAWS: [f64; 4] = [0.0, 0.0, 1.0, 2.0];

#[test]
fn f15_joint_moments_match_the_enumerated_law() {
    let artifact = DistributionArtifact::new(
        enumerated_metadata(expected_identity()),
        ENUMERATED_DRAWS.to_vec(),
    )
    .unwrap();
    let bytes = artifact.to_bytes("joint-fixture").unwrap();
    let loaded = DistributionArtifact::from_bytes(&bytes, &expected_identity()).unwrap();
    // Law: (X, Y) = (0, 0) and (1, 2) with probability one half each.
    assert!((loaded.mean(0).unwrap() - 0.5).abs() < 1e-12);
    assert!((loaded.mean(1).unwrap() - 1.0).abs() < 1e-12);
    assert!((loaded.covariance(0, 1).unwrap() - 0.5).abs() < 1e-12);
    assert!((loaded.joint_expectation(0, 1, |x, y| x * y).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn f15_independent_marginals_cannot_supply_covariance() {
    let mut identity = expected_identity();
    identity.alignment = DrawAlignment::IndependentMarginals;
    let marginals =
        DistributionArtifact::new(enumerated_metadata(identity), ENUMERATED_DRAWS.to_vec())
            .unwrap();
    for error in [
        marginals.covariance(0, 1).unwrap_err(),
        marginals.joint_expectation(0, 1, |x, y| x * y).unwrap_err(),
    ] {
        match error {
            antecedent_io::error::IoError::Refused { code, message } => {
                assert_eq!(code, "joint_law_required");
                assert!(
                    message.starts_with("aligned_joint_draws.marginals_not_joint:"),
                    "{message}"
                );
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
    // Per-coordinate means remain available; only the dependence is refused.
    assert!((marginals.mean(0).unwrap() - 0.5).abs() < 1e-12);
}

#[test]
fn f15_resealed_axes_alignment_and_meaning_changes_are_refused() {
    let expected = expected_identity();
    let reseal = |metadata: DistributionMetadata| {
        DistributionArtifact::new(metadata, ENUMERATED_DRAWS.to_vec())
            .unwrap()
            .to_bytes("joint-fixture")
            .unwrap()
    };
    let refused = |bytes: &[u8]| {
        DistributionArtifact::from_bytes(bytes, &expected).unwrap_err().reason_code()
    };
    // Control: the unchanged artifact is accepted.
    DistributionArtifact::from_bytes(&reseal(enumerated_metadata(expected.clone())), &expected)
        .unwrap();

    // Swapped physical axes cannot even be sealed by a conforming producer.
    let mut swapped_axes = enumerated_metadata(expected.clone());
    swapped_axes.axes = ["quantity".into(), "draw".into()];
    assert!(DistributionArtifact::new(swapped_axes, ENUMERATED_DRAWS.to_vec()).is_err());

    // Quantity (column) order of the axis pair changed and resealed.
    let mut reordered = expected.clone();
    reordered.quantities.reverse();
    assert_eq!(
        refused(&reseal(enumerated_metadata(reordered))),
        Some("quantity_semantics_mismatch")
    );

    // Draw alignment weakened from joint to independent marginals.
    let mut marginals = expected.clone();
    marginals.alignment = DrawAlignment::IndependentMarginals;
    assert_eq!(
        refused(&reseal(enumerated_metadata(marginals))),
        Some("quantity_semantics_mismatch")
    );

    // Quantity meaning changed in the semantic tag.
    for meaning in [
        DistributionMeaningWire::Bootstrap,
        DistributionMeaningWire::CausalFunctionalPosterior,
        DistributionMeaningWire::PosteriorPredictive,
    ] {
        let mut changed = expected.clone();
        changed.semantic = meaning;
        assert_eq!(
            refused(&reseal(enumerated_metadata(changed))),
            Some("quantity_semantics_mismatch"),
            "{meaning:?}"
        );
    }

    // Quantity meaning changed in a coordinate's functional.
    let mut functional = expected.clone();
    functional.quantities[1].functional_id = "mean".into();
    assert_eq!(
        refused(&reseal(enumerated_metadata(functional))),
        Some("quantity_semantics_mismatch")
    );

    // Snapshot changed.
    let mut snapshot = expected.clone();
    snapshot.snapshot_id = "other-snapshot".into();
    assert_eq!(
        refused(&reseal(enumerated_metadata(snapshot))),
        Some("quantity_semantics_mismatch")
    );
}
