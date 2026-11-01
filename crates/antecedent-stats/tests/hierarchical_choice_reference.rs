//! The hierarchical choice posterior against independent references
//! (`oracle_data/hierarchical_choice_reference.py`): mpmath double integration for two options;
//! a trapezoid rule in log-concentration coordinates and scipy's adaptive vector cubature over the
//! best-option regions for three. Also the structural properties the family states: the
//! homogeneous limit recovers the pooled Dirichlet posterior, the two-option kernel is the
//! beta-binomial, and unresolved or out-of-domain posteriors are refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    DirichletPosterior, HierarchicalChoicePosterior, HierarchicalChoicePrior,
};
use serde_json::Value;

fn never() -> bool {
    false
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(number).collect()
}

fn counts(value: &Value) -> Vec<Vec<u32>> {
    value
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| {
            row.as_array().expect("row").iter().map(|x| x.as_u64().expect("count") as u32).collect()
        })
        .collect()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/hierarchical_choice_reference.json"))
        .expect("frozen reference")
}

fn prior(case: &Value) -> HierarchicalChoicePrior {
    HierarchicalChoicePrior {
        shapes: numbers(&case["a"]),
        log_concentration_mean: number(&case["mu"]),
        log_concentration_sd: number(&case["sigma"]),
    }
}

fn close(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!((got - want).abs() <= tolerance, "{what}: {got} vs {want}");
}

#[test]
fn summaries_match_the_references() {
    for case in reference()["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let posterior = HierarchicalChoicePosterior::new(&prior(case), &counts(&case["counts"]))
            .expect("posterior");
        let s = posterior.summary(&never).expect("summary");
        let tol = 1e-7;
        close(s.log_marginal_likelihood, number(&case["log_evidence"]), tol, name);
        for (g, w) in s.mean.iter().zip(numbers(&case["mean"])) {
            close(*g, w, tol, &format!("{name} mean"));
        }
        for (g, w) in s.probability_best.iter().zip(numbers(&case["probability_best"])) {
            close(*g, w, tol, &format!("{name} best"));
        }
        for (g, w) in s.expected_loss.iter().zip(numbers(&case["expected_loss"])) {
            close(*g, w, tol, &format!("{name} loss"));
        }
        close(s.log_concentration_mean, number(&case["log_kappa_mean"]), tol, name);
        close(s.log_concentration_sd, number(&case["log_kappa_sd"]), tol, name);
        close(s.correlation_mean, number(&case["correlation_mean"]), tol, name);
        assert!(s.quadrature_error <= 1e-8, "{name}: {}", s.quadrature_error);
    }
}

#[test]
fn intervals_match_the_references() {
    for case in reference()["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let posterior = HierarchicalChoicePosterior::new(&prior(case), &counts(&case["counts"]))
            .expect("posterior");
        let want = numbers(&case["interval_option0_90"]);
        let (lo, hi) = posterior.share_interval(0, 0.9, &never).expect("interval");
        close(lo, want[0], 1e-7, &format!("{name} lower"));
        close(hi, want[1], 1e-7, &format!("{name} upper"));
    }
}

#[test]
fn a_nearly_homogeneous_population_recovers_the_pooled_posterior() {
    // ln kappa fixed near 13 (kappa about 4e5): respondents share one theta, so the population
    // shares are the pooled Dirichlet posterior of the summed counts.
    let rows: Vec<Vec<u32>> = vec![vec![3, 1, 0], vec![2, 1, 1], vec![3, 0, 1], vec![1, 2, 1]];
    let prior = HierarchicalChoicePrior {
        shapes: vec![1.0; 3],
        log_concentration_mean: 12.0,
        log_concentration_sd: 0.05,
    };
    let summary = HierarchicalChoicePosterior::new(&prior, &rows)
        .expect("posterior")
        .summary(&never)
        .expect("summary");
    let pooled = DirichletPosterior::from_counts(&[1.0; 3], &[9.0, 4.0, 3.0]).expect("pooled");
    for (g, w) in summary.mean.iter().zip(pooled.mean()) {
        close(*g, w, 5e-4, "pooled mean");
    }
    for (g, w) in
        summary.probability_best.iter().zip(pooled.best_option_probabilities(&never).expect("best"))
    {
        close(*g, w, 5e-4, "pooled best");
    }
}

#[test]
fn heterogeneity_widens_the_population_interval_over_the_pooled_one() {
    let rows: Vec<Vec<u32>> =
        vec![vec![4, 0], vec![0, 4], vec![4, 0], vec![0, 4], vec![3, 1], vec![1, 3]];
    let mixed = HierarchicalChoicePrior {
        shapes: vec![1.0; 2],
        log_concentration_mean: 0.0,
        log_concentration_sd: 1.0,
    };
    let (lo, hi) = HierarchicalChoicePosterior::new(&mixed, &rows)
        .expect("posterior")
        .share_interval(0, 0.9, &never)
        .expect("interval");
    // Pooled Beta(13, 13) 90% interval half-width is about 0.16; respondent heterogeneity
    // makes the population share genuinely less certain.
    assert!(hi - lo > 0.32, "{lo} {hi}");
}

#[test]
fn the_domain_and_bounds_are_refused() {
    let good = HierarchicalChoicePrior {
        shapes: vec![1.0; 3],
        log_concentration_mean: 1.0,
        log_concentration_sd: 1.0,
    };
    let rows: Vec<Vec<u32>> = vec![vec![2, 1, 0], vec![1, 1, 1]];
    assert!(HierarchicalChoicePosterior::new(&good, &rows).is_ok());
    let mut four = good.clone();
    four.shapes = vec![1.0; 4];
    assert!(HierarchicalChoicePosterior::new(&four, &[vec![1, 1, 0, 0]]).is_err());
    let mut sparse = good.clone();
    sparse.shapes = vec![0.5, 1.0, 1.0];
    assert!(HierarchicalChoicePosterior::new(&sparse, &rows).is_err());
    let mut flat = good.clone();
    flat.log_concentration_sd = 0.0;
    assert!(HierarchicalChoicePosterior::new(&flat, &rows).is_err());
    let singles: Vec<Vec<u32>> = vec![vec![1, 0, 0], vec![0, 1, 0]];
    assert!(HierarchicalChoicePosterior::new(&good, &singles).is_err());
    assert!(HierarchicalChoicePosterior::new(&good, &[vec![2, 1]]).is_err());
    let posterior = HierarchicalChoicePosterior::new(&good, &rows).expect("posterior");
    assert!(posterior.share_interval(3, 0.9, &never).is_err());
    assert!(posterior.share_interval(0, 1.0, &never).is_err());
    assert!(posterior.summary(&|| true).is_err());
}
