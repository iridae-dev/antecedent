//! Distribution-meaning boundary: only a declared interventional outcome law
//! answers an interventional outcome threshold.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::DistributionMeaningWire;

const DRAWS: [f64; 4] = [0.0, 0.0, 1.0, 2.0];

fn identity(meaning: DistributionMeaningWire) -> DistributionIdentity {
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
        meaning,
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

fn artifact(meaning: DistributionMeaningWire) -> DistributionArtifact {
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: identity(meaning),
            axes: ["draw".into(), "quantity".into()],
            shape: [2, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        DRAWS.to_vec(),
    )
    .unwrap()
}

fn assert_refused(error: IoError, expected_code: &str, expected_detail: &str) {
    match error {
        IoError::Refused { code, message } => {
            assert_eq!(code, expected_code);
            assert!(message.starts_with(&format!("{expected_detail}:")), "{message}");
        }
        other => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn f16_interventional_predictive_law_answers_its_own_outcome_threshold() {
    let law = artifact(DistributionMeaningWire::InterventionalPredictive);
    // Outcome Y takes 0 and 2 with probability one half each.
    assert!((law.interventional_probability_above(1, 1.0).unwrap() - 0.5).abs() < 1e-12);
    // Outcome X takes 0 and 1: P(X > 0.5) = 1/2; P(X > 1) = 0; P(X > -1) = 1.
    assert!((law.interventional_probability_above(0, 0.5).unwrap() - 0.5).abs() < 1e-12);
    assert!(law.interventional_probability_above(0, 1.0).unwrap().abs() < 1e-12);
    assert!((law.interventional_probability_above(0, -1.0).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn f16_effect_posterior_and_observational_law_refuse_an_interventional_threshold() {
    for meaning in [
        DistributionMeaningWire::CausalFunctionalPosterior,
        DistributionMeaningWire::ParameterPosterior,
        DistributionMeaningWire::EstimatorSampling,
        DistributionMeaningWire::Bootstrap,
    ] {
        let error = artifact(meaning).interventional_probability_above(1, 1.0).unwrap_err();
        assert_refused(
            error,
            "distribution_meaning_mismatch",
            "distribution_meaning.incompatible_operation",
        );
    }
    for meaning in
        [DistributionMeaningWire::PosteriorPredictive, DistributionMeaningWire::EmpiricalOutcome]
    {
        let error = artifact(meaning).interventional_probability_above(1, 1.0).unwrap_err();
        assert_refused(
            error,
            "quantity_semantics_mismatch",
            "distribution_meaning.observational_not_interventional",
        );
    }
}

#[test]
fn f16_resealed_meaning_change_is_refused_by_a_fresh_consumer() {
    // The consumer's retained contract names an interventional predictive law.
    let expected = identity(DistributionMeaningWire::InterventionalPredictive);
    let control = artifact(DistributionMeaningWire::InterventionalPredictive)
        .to_bytes("meaning-fixture")
        .unwrap();
    DistributionArtifact::from_bytes(&control, &expected).unwrap();
    for source in
        [DistributionMeaningWire::Bootstrap, DistributionMeaningWire::CausalFunctionalPosterior]
    {
        let resealed = artifact(source).to_bytes("meaning-fixture").unwrap();
        let error = DistributionArtifact::from_bytes(&resealed, &expected).unwrap_err();
        assert_refused(
            error,
            "quantity_semantics_mismatch",
            "aligned_joint_draws.identity_expected",
        );
    }
}
