//! The pairing identity of a distribution artifact: two joint artifacts pair
//! row by row only when they are draws of one joint law (same source, provider,
//! rng stream, snapshot, causal contract and draw count). A constant label would
//! pair artifacts of different simulations.

use antecedent_contract::{QuantityRole, ScientificQuantity, check_rows_pairable_across};
use antecedent_io_core::IoError;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn artifact(alignment: DrawAlignment, rng: &str, rows: usize) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[quantity("x"), quantity("y")],
        alignment,
        DistributionProvenance {
            source_id: "study".into(),
            provider_id: "provider".into(),
            rng_id: rng.into(),
            snapshot_id: "snapshot".into(),
            causal_contract_id: "contract".into(),
        },
    )
    .expect("identity");
    let draws: Vec<f64> = (0..rows * 2).map(|i| i as f64).collect();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("artifact")
}

#[test]
fn artifacts_of_one_joint_law_share_a_pairing_identity() {
    let a = artifact(DrawAlignment::Joint, "pcg:seed=1:stream=a", 4);
    let b = artifact(DrawAlignment::Joint, "pcg:seed=1:stream=a", 4);
    assert_eq!(a.alignment_identity().unwrap(), b.alignment_identity().unwrap());
    let (ia, ib) = (a.input_alignment().unwrap(), b.input_alignment().unwrap());
    check_rows_pairable_across(&[("a", &ia), ("b", &ib)]).expect("one joint law pairs");
}

#[test]
fn artifacts_of_different_simulations_are_not_paired_by_index() {
    let a = artifact(DrawAlignment::Joint, "pcg:seed=1:stream=a", 4);
    // Another rng stream; another draw count.
    for other in [
        artifact(DrawAlignment::Joint, "pcg:seed=2:stream=a", 4),
        artifact(DrawAlignment::Joint, "pcg:seed=1:stream=a", 5),
    ] {
        assert_ne!(a.alignment_identity().unwrap(), other.alignment_identity().unwrap());
        let (ia, io) = (a.input_alignment().unwrap(), other.input_alignment().unwrap());
        let refusal = check_rows_pairable_across(&[("a", &ia), ("other", &io)])
            .expect_err("different joint laws do not pair");
        assert_eq!(refusal.code, "joint_law_required");
        assert_eq!(refusal.detail, "joint_law_propagation.alignment_mismatch");
    }
}

#[test]
fn independent_marginals_have_no_pairing_identity() {
    let marginals = artifact(DrawAlignment::IndependentMarginals, "pcg:seed=1:stream=a", 4);
    match marginals.alignment_identity() {
        Err(IoError::Refused { code, .. }) => assert_eq!(code, "joint_law_required"),
        other => panic!("expected a joint_law_required refusal, got {other:?}"),
    }
    let joint = artifact(DrawAlignment::Joint, "pcg:seed=1:stream=a", 4);
    let (im, ij) = (marginals.input_alignment().unwrap(), joint.input_alignment().unwrap());
    assert!(check_rows_pairable_across(&[("m", &im), ("j", &ij)]).is_err());
}
