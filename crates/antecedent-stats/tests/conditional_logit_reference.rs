//! Conditional logit fits set against an independent 50-digit Newton solution.
//!
//! The frozen values come from `oracle_data/conditional_logit_reference.py` (its output is
//! `oracle_data/conditional_logit_reference.json`). That script works on the raw option features
//! with its own gradient and information code, solves in 50-digit decimal arithmetic, and covers
//! one to four features, sets of two to five options, 60 to 400 sets, clustered sets, and a
//! feature carrying a large per-set offset. Coefficients are compared to `1e-8` (relative to
//! `max(1, |beta|)`), the log-likelihoods to `1e-9` relative, and both covariances to `1e-7`
//! relative to their largest entry.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    ConditionalLogitDesign, ConditionalLogitOptions, conditional_logit_log_likelihood,
    fit_conditional_logit,
};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/conditional_logit_reference.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| usize::try_from(x.as_u64().unwrap()).unwrap()).collect()
}

fn normwise(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn relative(got: f64, want: f64) -> f64 {
    ((got - want) / want.abs()).abs()
}

#[test]
fn conditional_logit_matches_fifty_digit_newton() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let cases = doc.as_array().unwrap();
    assert_eq!(cases.len(), 5);
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let features = floats(&case["features"]);
        let set_starts = indices(&case["set_starts"]);
        let chosen = indices(&case["chosen"]);
        let clusters: Option<Vec<u32>> = case["clusters"]
            .as_array()
            .map(|a| a.iter().map(|x| u32::try_from(x.as_u64().unwrap()).unwrap()).collect());
        let design = ConditionalLogitDesign {
            features: &features,
            n_features: usize::try_from(case["n_features"].as_u64().unwrap()).unwrap(),
            set_starts: &set_starts,
            chosen: &chosen,
            clusters: clusters.as_deref(),
        };
        let fit = fit_conditional_logit(&design, ConditionalLogitOptions::default())
            .unwrap_or_else(|e| panic!("{name}: fit refused: {e}"));

        let want_beta = floats(&case["coefficients"]);
        for (k, (g, w)) in fit.coefficients.iter().zip(&want_beta).enumerate() {
            if (g - w).abs() > 1e-8 * w.abs().max(1.0) {
                failures.push(format!("{name}: coefficient {k}: {g} vs {w}"));
            }
        }
        let want_ll = case["log_likelihood"].as_f64().unwrap();
        if relative(fit.log_likelihood, want_ll) > 1e-9 {
            failures.push(format!("{name}: log-likelihood {} vs {want_ll}", fit.log_likelihood));
        }
        let at_reference = conditional_logit_log_likelihood(&design, &want_beta).unwrap();
        if relative(at_reference, want_ll) > 1e-12 {
            failures.push(format!("{name}: log-likelihood at reference {at_reference}"));
        }
        let want_null = case["null_log_likelihood"].as_f64().unwrap();
        if relative(fit.null_log_likelihood, want_null) > 1e-12 {
            failures.push(format!("{name}: null log-likelihood {}", fit.null_log_likelihood));
        }
        let cov = normwise(&fit.covariance, &floats(&case["covariance"]));
        if cov > 1e-7 {
            failures.push(format!("{name}: covariance off by {cov:e}"));
        }
        match (&fit.cluster_covariance, case["cluster_covariance"].as_array()) {
            (Some(got), Some(_)) => {
                let off = normwise(got, &floats(&case["cluster_covariance"]));
                if off > 1e-7 {
                    failures.push(format!("{name}: cluster covariance off by {off:e}"));
                }
            }
            (None, None) => {}
            _ => failures.push(format!("{name}: cluster covariance presence differs")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
