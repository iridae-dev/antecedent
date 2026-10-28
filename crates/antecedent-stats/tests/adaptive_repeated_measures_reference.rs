//! Unit-summary variances, design effects and the exact fixed-allocation contrast law of a
//! repeated-measures or clustered procedure against exact rational arithmetic
//! (`oracle_data/procedure_repeated_reference.py`).
//!
//! The reference builds each correlation matrix explicitly (compound symmetry, AR(1), random
//! effect), inverts it exactly to obtain the generalized-least-squares weights and their variance
//! `s2 / (1' R^-1 1)` with no closed form assumed, across 2 to 32 measurements, correlations from
//! the edge of positive definiteness to 0.99 and negative values, and writes the binomial mixture
//! of the contrast with `Fraction` weights for up to 4000 revealed units. The mixture cases past
//! about a thousand units are the ones on which `p^n` leaves the range of a double.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, FinalAnalysis, OutcomeStructure, RepeatedMeasuresDeclaration,
    RepeatedMeasuresSpec, StoppingRule, UnitAssignment, UnitSummary, WithinUnitCorrelation,
    exact_fixed_contrast_law,
};
use serde_json::Value;
use support::frac;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_repeated_reference.json"))
        .expect("frozen reference")
}

fn relative(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!(
        (got - want).abs() <= tolerance * want.abs().max(1e-300),
        "{what}: got {got}, want {want}"
    );
}

fn structure(value: &Value, measurements: u32) -> OutcomeStructure {
    match value["kind"].as_str().unwrap() {
        "compound" => OutcomeStructure::RepeatedMeasures {
            measurements,
            variance: frac(&value["variance"]),
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric {
                rho: frac(&value["rho"]),
            }),
        },
        "autoregressive" => OutcomeStructure::RepeatedMeasures {
            measurements,
            variance: frac(&value["variance"]),
            correlation: Some(WithinUnitCorrelation::Autoregressive { rho: frac(&value["rho"]) }),
        },
        _ => OutcomeStructure::Hierarchical {
            members: measurements,
            between_variance: Some(frac(&value["between"])),
            within_variance: Some(frac(&value["within"])),
        },
    }
}

fn spec(
    structure: OutcomeStructure,
    summary: UnitSummary,
    units: u32,
    plus_probability: f64,
) -> RepeatedMeasuresSpec {
    RepeatedMeasuresSpec::new(RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(UnitAssignment::Fixed {
            probabilities: vec![1.0 - plus_probability, plus_probability],
        }),
        enrolment: Some(vec![units]),
        lag_looks: 0,
        structure: Some(structure),
        analysis: Some(FinalAnalysis {
            estimator: AnalysisEstimator::ClusterRobust { summary },
            plus: 1,
            minus: 0,
        }),
        stopping: Some(StoppingRule::NoStopping),
    })
    .expect("a valid declaration")
}

#[test]
fn unit_summary_variances_and_design_effects_match_the_exact_inverse() {
    let reference = reference();
    let rows = reference["summaries"].as_array().unwrap();
    assert!(rows.len() > 100);
    for row in rows {
        let m = row["measurements"].as_u64().unwrap() as u32;
        let summary = if row["summary"] == "gls" { UnitSummary::Gls } else { UnitSummary::Mean };
        let label = format!("{} m={m} {} {}", row["name"], row["summary"], row["structure"]);
        let spec = spec(structure(&row["structure"], m), summary, 4, 0.5);
        relative(
            spec.marginal_variance(),
            row["marginal_variance"].as_f64().unwrap(),
            1e-13,
            &label,
        );
        relative(
            spec.unit_summary_variance(),
            row["unit_summary_variance"].as_f64().unwrap(),
            1e-9,
            &label,
        );
        relative(spec.design_effect(), row["design_effect"].as_f64().unwrap(), 1e-9, &label);
    }
}

#[test]
fn the_fixed_allocation_mixture_matches_the_exact_binomial_weights() {
    let reference = reference();
    for case in reference["mixtures"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as u32;
        let p = frac(&case["plus_probability"]);
        let v = case["summary_variance"].as_f64().unwrap();
        let delta = case["delta"].as_f64().unwrap();
        // One measurement pair per unit with variance v * 4 / (2 (1 + rho)): choose rho = 0 and
        // marginal variance 2 v so the mean of two measurements has variance v.
        let structure = OutcomeStructure::RepeatedMeasures {
            measurements: 2,
            variance: 2.0 * v,
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.0 }),
        };
        let spec = spec(structure, UnitSummary::Mean, n, p);
        let law = exact_fixed_contrast_law(&spec, &[0.0, delta]).expect("a law");
        let label = format!("n={n} p={p}");
        assert_eq!(
            law.components.len(),
            case["component_count"].as_u64().unwrap() as usize,
            "{label}"
        );
        relative(law.defined_mass(), case["defined_mass"].as_f64().unwrap(), 1e-12, &label);
        assert!(
            (law.undefined_mass - case["undefined_mass"].as_f64().unwrap()).abs() <= 1e-12,
            "{label}: undefined {}",
            law.undefined_mass
        );
        relative(law.variance(), case["variance"].as_f64().unwrap(), 1e-9, &label);
        assert!((law.mean() - delta).abs() <= 1e-12, "{label}: mean");
        for (k, want) in case["components"].as_object().unwrap() {
            let k: usize = k.parse().unwrap();
            let component = &law.components[k - 1];
            let w = want["weight"].as_f64().unwrap();
            assert!(
                (component.weight - w).abs() <= 1e-12 + 1e-9 * w,
                "{label}: weight at k={k}: {} vs {w}",
                component.weight
            );
            relative(component.variance, want["variance"].as_f64().unwrap(), 1e-12, &label);
        }
        for point in case["cdf"].as_array().unwrap() {
            let x = point["x"].as_f64().unwrap();
            assert!(
                (law.cdf(x) - point["cdf"].as_f64().unwrap()).abs() <= 1e-9,
                "{label}: cdf at {x}"
            );
        }
    }
}
