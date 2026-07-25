//! Prior-domain validation against frozen SciPy support/normalization evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_prob::{
    ContrastCoding, EffectPrior, GaussianCoefficientPrior, GaussianVarianceModel, InvGammaPrior,
    PriorSet, PriorSpec,
};
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/bayesian/prior_support/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("prior-support fixture"))
        .expect("parse prior-support fixture")
}

fn number(value: &JsonValue) -> f64 {
    value.as_f64().unwrap_or_else(|| match value.as_str().unwrap() {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        other => panic!("unknown encoded number {other}"),
    })
}

#[test]
fn coefficient_effect_variance_and_inverse_gamma_support_matches_oracle() {
    let fixture = fixture();
    for case in fixture["coefficient_priors"].as_array().unwrap() {
        let mean: Vec<f64> = case["mean"].as_array().unwrap().iter().map(number).collect();
        let variance: Vec<f64> = case["variance"].as_array().unwrap().iter().map(number).collect();
        let prior =
            GaussianCoefficientPrior { mean: Arc::from(mean), variance: Arc::from(variance) };
        assert_eq!(prior.validate().is_ok(), case["valid"].as_bool().unwrap(), "{}", case["name"]);
        if case["valid"].as_bool().unwrap() {
            let expected: Vec<f64> =
                case["precision"].as_array().unwrap().iter().map(number).collect();
            assert_eq!(prior.precision(), expected);
            for integral in case["normal_integrals"].as_array().unwrap() {
                assert!((integral.as_f64().unwrap() - 1.0).abs() <= 2e-8);
            }
        }
    }

    for case in fixture["effect_priors"].as_array().unwrap() {
        let result = EffectPrior::new(number(&case["mean"]), number(&case["sd"]));
        assert_eq!(result.is_ok(), case["valid"].as_bool().unwrap());
    }
    for case in fixture["inverse_gamma_priors"].as_array().unwrap() {
        let prior = InvGammaPrior {
            shape: case["shape"].as_f64().unwrap(),
            scale: case["scale"].as_f64().unwrap(),
        };
        assert_eq!(prior.validate().is_ok(), case["valid"].as_bool().unwrap());
        if case["valid"].as_bool().unwrap() {
            assert!((case["integral"].as_f64().unwrap() - 1.0).abs() <= 2e-8);
        }
    }
    for case in fixture["known_variances"].as_array().unwrap() {
        let prior = PriorSpec::KnownResidualVariance(number(&case["value"]));
        assert_eq!(prior.validate().is_ok(), case["valid"].as_bool().unwrap());
    }
}

#[test]
fn prior_set_residual_and_contrast_rules_match_frozen_contract() {
    let mut duplicate = PriorSet::new();
    duplicate.push(PriorSpec::KnownResidualVariance(1.0));
    duplicate.push(PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 2.0, scale: 3.0 }));
    assert!(duplicate.validate().is_err());
    assert!(GaussianVarianceModel::from_prior_set(&duplicate).is_err());

    let empty = PriorSet::new();
    assert_eq!(
        GaussianVarianceModel::from_prior_set(&empty).unwrap(),
        GaussianVarianceModel::InvGamma { shape: 1e-3, scale: 1e-3 }
    );

    let mut categorical = PriorSet::weakly_informative(2);
    categorical.categorical.push(VariableId::from_raw(0));
    assert!(categorical.validate().is_err());
    categorical.contrast = Some(ContrastCoding::Treatment);
    assert!(categorical.validate().is_ok());
    categorical.contrast = Some(ContrastCoding::Sum);
    assert!(categorical.validate().is_ok());
}

#[test]
fn prior_fixture_records_scipy_pins_and_no_generator() {
    let fixture = fixture();
    assert_eq!(fixture["oracle"]["packages"]["scipy"]["version"], "1.14.1");
    assert_eq!(fixture["oracle"]["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(
        fixture["oracle"]["generation_location"],
        "temporary external harness; not retained"
    );
    for package in ["scipy", "numpy"] {
        assert_eq!(
            fixture["oracle"]["packages"][package]["metadata_sha256"].as_str().unwrap().len(),
            64
        );
    }
}
