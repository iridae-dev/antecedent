//! The posterior of arm means under repeated measures per unit against 50-digit evaluation of the
//! multivariate normal model (`oracle_data/repeated_measures_posterior_reference.py`).
//!
//! The reference builds each unit's covariance explicitly (compound symmetric, autoregressive, or a
//! unit-level random effect), solves it with a general linear solve and conditions the arm mean on
//! every unit, instead of the closed-form generalized-least-squares weights of the product. It
//! covers two to thirty-two measurements per unit, correlations from -0.9 to 0.999999 (and one
//! within 1e-6 of the positive-definite limit), a between-unit variance from 0 to 1e6 times the
//! within variance, and priors from informative to variance 1e6.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    NormalMeanPrior, OutcomeStructure, RepeatedMeasuresPosteriorRequest, SelectionMechanism,
    UnitRecord, WithinUnitCorrelation, repeated_measures_posterior,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/repeated_measures_posterior_reference.json"))
        .expect("frozen reference")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

fn structure(value: &Value) -> OutcomeStructure {
    let m = value["m"].as_u64().expect("m") as u32;
    match value["kind"].as_str().expect("kind") {
        "cs" => OutcomeStructure::RepeatedMeasures {
            measurements: m,
            variance: number(&value["variance"]),
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric {
                rho: number(&value["rho"]),
            }),
        },
        "ar" => OutcomeStructure::RepeatedMeasures {
            measurements: m,
            variance: number(&value["variance"]),
            correlation: Some(WithinUnitCorrelation::Autoregressive { rho: number(&value["rho"]) }),
        },
        _ => OutcomeStructure::Hierarchical {
            members: m,
            between_variance: Some(number(&value["between"])),
            within_variance: Some(number(&value["within"])),
        },
    }
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-300 + relative * want.abs(), "{what}: {got} vs {want}");
}

#[test]
fn posteriors_match_the_multivariate_normal_reference() {
    let reference = reference();
    for case in reference["cases"].as_array().expect("cases") {
        let structure = structure(&case["structure"]);
        let priors: Vec<NormalMeanPrior> = case["priors"]
            .as_array()
            .expect("priors")
            .iter()
            .map(|p| NormalMeanPrior { mean: number(&p[0]), variance: number(&p[1]) })
            .collect();
        let measurements: Vec<Vec<f64>> = case["units"]
            .as_array()
            .expect("units")
            .iter()
            .map(|u| {
                u["measurements"].as_array().expect("measurements").iter().map(number).collect()
            })
            .collect();
        let units: Vec<UnitRecord<'_>> = case["units"]
            .as_array()
            .expect("units")
            .iter()
            .zip(&measurements)
            .map(|(u, m)| UnitRecord {
                arm: u["arm"].as_u64().expect("arm") as usize,
                measurements: m,
                assignment_probability: number(&u["probability"]),
            })
            .collect();
        let request = RepeatedMeasuresPosteriorRequest {
            structure: &structure,
            selection: SelectionMechanism::DependsOnObservedHistory,
            priors: &priors,
            units: &units,
            plus: case["plus"].as_u64().expect("plus") as usize,
            minus: case["minus"].as_u64().expect("minus") as usize,
            credible_level: number(&case["level"]),
        };
        let got = repeated_measures_posterior(&request)
            .unwrap_or_else(|e| panic!("{}: {e:?}", case["structure"]));
        let what = format!("{}", case["structure"]);
        let tolerance = 1e-9;
        close(
            got.unit_summary_variance,
            number(&case["unit_summary_variance"]),
            tolerance,
            &format!("summary variance {what}"),
        );
        for (g, w) in got.arms.iter().zip(case["arms"].as_array().expect("arms")) {
            assert_eq!(g.units as u64, w["units"].as_u64().expect("units"));
            let scale = number(&w["sd"]);
            assert!(
                (g.mean - number(&w["mean"])).abs()
                    <= tolerance * (number(&w["mean"]).abs() + scale),
                "mean {what}: {} vs {}",
                g.mean,
                w["mean"]
            );
            close(g.standard_deviation, scale, tolerance, &format!("sd {what}"));
        }
        let sd = number(&case["contrast_sd"]);
        let mean = number(&case["contrast_mean"]);
        assert!(
            (got.contrast.mean - mean).abs() <= tolerance * (mean.abs() + sd),
            "contrast {what}"
        );
        close(got.contrast.standard_deviation, sd, tolerance, &format!("contrast sd {what}"));
        for (g, w) in
            got.contrast.interval.iter().zip(case["interval"].as_array().expect("interval"))
        {
            let w = number(w);
            assert!((g - w).abs() <= tolerance * (w.abs() + sd), "interval {what}: {g} vs {w}");
        }
        let positive = number(&case["probability_positive"]);
        assert!((got.contrast.probability_positive - positive).abs() <= 1e-9, "positive {what}");
        assert!(got.evidence.adaptive);
    }
}
