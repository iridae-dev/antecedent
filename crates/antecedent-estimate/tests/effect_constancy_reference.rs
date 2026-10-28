//! The effect-constancy test against 60-digit evaluation of its definitions.
//!
//! `oracle_data/effect_constancy_reference.py` (frozen in the JSON beside it) computes Cochran's
//! Q, the Wald statistic of the contrasts against the first partition in label order, the
//! chi-square tail (regularized upper incomplete gamma), the two-sided normal p-value of every
//! contrast and the Holm step-down adjustment. Cases cover precisions over six decades,
//! chi-square tails down to 1e-276, effects of 1e150, correlated partitions supplied out of
//! label order and both contrast families.

use antecedent_estimate::effect_constancy::{
    ContrastFamily, EffectEstimandIdentity, PartitionDependence, PartitionEstimate,
    PartitionSupport, test_effect_constancy,
};
use serde_json::Value;

fn estimand() -> EffectEstimandIdentity {
    EffectEstimandIdentity {
        estimand: "ate_difference".into(),
        units: "outcome_units".into(),
        regime: "treat_vs_control".into(),
        population: "all_observed_h2".into(),
    }
}

fn agree(case: &str, what: &str, actual: f64, expected: f64) {
    let tolerance = 1e-9 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{case} {what}: {actual:.17e} vs {expected:.17e}"
    );
}

#[test]
fn statistics_tails_and_holm_values_match_the_exact_definitions() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/effect_constancy_reference.json")).unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let partitions: Vec<PartitionEstimate> = case["partitions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let label = p["label"].as_str().unwrap();
                PartitionEstimate {
                    label: label.into(),
                    coordinate: format!("period:{label}"),
                    support: PartitionSupport::Supported,
                    estimand: estimand(),
                    effect: p["effect"].as_f64().unwrap(),
                    standard_error: p["se"].as_f64().unwrap(),
                }
            })
            .collect();
        let dependence = match case["covariance"].as_array() {
            Some(c) => {
                PartitionDependence::Covariance(c.iter().map(|v| v.as_f64().unwrap()).collect())
            }
            None => PartitionDependence::Independent,
        };
        let family = match case["family"].as_str().unwrap() {
            "all" => ContrastFamily::AllPairs,
            reference => ContrastFamily::AgainstReference(reference.into()),
        };
        let result = test_effect_constancy(&partitions, &dependence, &family, 0.05).unwrap();
        assert_eq!(result.test.degrees_of_freedom as u64, case["df"].as_u64().unwrap(), "{name}");
        agree(name, "statistic", result.test.statistic, case["statistic"].as_f64().unwrap());
        agree(name, "p_value", result.test.p_value, case["p_value"].as_f64().unwrap());
        if let Some(pooled) = case["pooled"].as_f64() {
            agree(name, "pooled", result.pooled_effect.unwrap(), pooled);
        } else {
            assert!(result.pooled_effect.is_none(), "{name}: pooled effect with a covariance");
        }
        let expected = case["contrasts"].as_array().unwrap();
        assert_eq!(result.contrasts.len(), expected.len(), "{name}: contrast count");
        for (got, want) in result.contrasts.iter().zip(expected) {
            assert_eq!(got.left, want["left"].as_str().unwrap(), "{name}: left");
            assert_eq!(got.right, want["right"].as_str().unwrap(), "{name}: right");
            agree(name, "difference", got.difference, want["difference"].as_f64().unwrap());
            agree(name, "se", got.standard_error, want["se"].as_f64().unwrap());
            agree(name, "z", got.z, want["z"].as_f64().unwrap());
            agree(name, "p", got.p_value, want["p"].as_f64().unwrap());
            agree(name, "p_holm", got.p_holm, want["p_holm"].as_f64().unwrap());
        }
    }
}
