//! The posterior-mean gap of a whole procedure and the posterior report, against exact rational
//! path enumeration (`oracle_data/procedure_gap_reference.py`).
//!
//! The reference expands every path of the declared procedure under the prior predictive (a Polya
//! urn per delivered cell) in `Fraction` arithmetic. The gap is exactly zero for policies and
//! stopping rules that read observed history only with outcome-independent loss (Thompson with a
//! posterior-probability stop, lagged epsilon-greedy with administrative loss, futility stop with
//! posterior-mean selection, UCB with a cap), and the product must reproduce the reference's
//! nonzero gaps where a premise fails (outcome-dependent loss with fixed and adaptive assignment,
//! counts attributed to the assigned arm while exposure fails). The report rows check the Beta
//! shapes, the probability of being best and the expected loss under non-uniform priors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    BetaPrior, ProcedureSpec, bayesian_report, posterior_mean_gaps,
};
use serde_json::Value;
use support::{never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_gap_reference.json"))
        .expect("frozen reference")
}

fn priors(value: &Value) -> Vec<BetaPrior> {
    value
        .as_array()
        .expect("priors")
        .iter()
        .map(|p| BetaPrior { alpha: p[0].as_f64().unwrap(), beta: p[1].as_f64().unwrap() })
        .collect()
}

#[test]
fn posterior_mean_gaps_match_the_exact_prior_predictive_expectation() {
    let reference = reference();
    let cases = reference["gaps"].as_array().expect("cases");
    let (mut zero, mut nonzero) = (0, 0);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let gaps =
            posterior_mean_gaps(&spec(&case["declaration"]), &priors(&case["priors"]), &never)
                .expect("gaps");
        let cells = case["cells"].as_array().expect("cells");
        assert_eq!(gaps.len(), cells.len(), "{name}");
        for (got, want) in gaps.iter().zip(cells) {
            let (prior_mean, expected, gap) = (
                want["prior_mean"].as_f64().unwrap(),
                want["expected_posterior_mean"].as_f64().unwrap(),
                want["gap"].as_f64().unwrap(),
            );
            assert!((got.prior_mean - prior_mean).abs() <= 1e-15, "{name}: prior mean");
            assert!((got.expected_posterior_mean - expected).abs() <= 1e-12, "{name}: expectation");
            assert!(
                (got.gap - gap).abs() <= 1e-12,
                "{name} cell {}: gap {} vs {gap}",
                got.cell,
                got.gap
            );
            if want["gap_exact"] == "0/1" {
                zero += 1;
                assert!(got.gap.abs() <= 1e-12, "{name}: a martingale has no gap");
            } else {
                nonzero += 1;
                assert!(got.gap.abs() > 1e-3, "{name}: the witnessed gap is not small");
            }
        }
    }
    assert!(zero >= 8 && nonzero >= 5);
}

#[test]
fn posterior_reports_match_the_exact_beta_values() {
    let reference = reference();
    let rows = reference["report"].as_array().expect("rows");
    assert_eq!(rows.len(), 18);
    let declaration = serde_json::json!({
        "arms": 2,
        "assignment": {"kind": "fixed", "probabilities": ["1/2", "1/2"]},
        "enrolment": [2, 2],
        "stopping": {"kind": "none"},
        "selection": {"kind": "none"}
    });
    let spec: ProcedureSpec = support::spec(&declaration);
    for row in rows {
        let prior = priors(&Value::Array(vec![row["prior"].clone()]))[0];
        let counts: Vec<u16> =
            row["counts"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as u16).collect();
        let report = bayesian_report(&spec, prior, &counts, &never).expect("a report");
        for arm in 0..2 {
            let shape = &row["posteriors"][arm];
            assert!((report.posteriors[arm].alpha() - shape[0].as_f64().unwrap()).abs() < 1e-15);
            assert!((report.posteriors[arm].beta() - shape[1].as_f64().unwrap()).abs() < 1e-15);
            let (b, l) = (row["best"][arm].as_f64().unwrap(), row["loss"][arm].as_f64().unwrap());
            assert!(
                (report.probability_best[arm] - b).abs() <= 1e-9,
                "{counts:?} {prior:?}: best {} vs {b}",
                report.probability_best[arm]
            );
            assert!(
                (report.expected_loss[arm] - l).abs() <= 1e-9,
                "{counts:?} {prior:?}: loss {} vs {l}",
                report.expected_loss[arm]
            );
        }
    }
    // A prior shape below one is refused, as is a count vector of the wrong length.
    let bad = BetaPrior { alpha: 0.5, beta: 1.0 };
    assert!(format!("{:?}", bayesian_report(&spec, bad, &[0, 0, 0, 0], &never).err()).contains("Beta prior shape must be finite"), "Beta prior shape must be finite");
    assert!(format!("{:?}", bayesian_report(&spec, BetaPrior::UNIFORM, &[0, 0, 0], &never).err()).contains("observed counts must hold a success and a failure per cell"), "observed counts must hold a success and a failure per cell");
}
