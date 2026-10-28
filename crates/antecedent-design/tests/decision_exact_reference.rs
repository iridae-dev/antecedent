//! Bayes action, expected regret (EVPI) and exact binomial EVSI against an exact-rational
//! reference (`oracle_data/decision_exact_reference.py`, `fractions.Fraction`, dyadic inputs).
//!
//! 120 random problems of 1 to 7 state draws and 2 to 4 affine actions, with common utility
//! offsets up to 1e6 (the regret and EVSI are origin-free), exact action ties, a draw-by-draw
//! chance constraint whose satisfaction lands exactly on the threshold in some cases, empty
//! admissible sets, and weights spanning six orders of magnitude on the unconstrained
//! weighted route. EVSI is compared at 1, 2, 5, 12 and 60 binomial trials.

use std::sync::Arc;

use antecedent_design::{
    AffineUtility, BinomialSignal, DecisionConstraint, DecisionPrior, DecisionProblem, DesignError,
    PreposteriorAnalysis, evaluate_decision,
};
use serde_json::Value;

struct Cutoffs(Vec<f64>);

impl DecisionConstraint<usize, f64> for Cutoffs {
    fn name(&self) -> &str {
        "theta_at_most_cutoff"
    }

    fn satisfaction_batch(&self, actions: &[usize], outcomes: &[f64], out: &mut [f64]) {
        for (slot, &action) in out.iter_mut().zip(actions) {
            let cutoff = self.0[action];
            *slot =
                outcomes.iter().filter(|t| **t <= cutoff).count() as f64 / outcomes.len() as f64;
        }
    }
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn close(got: f64, want: f64, what: &str) {
    let tolerance = 1e-9 + 1e-12 * want.abs();
    assert!((got - want).abs() <= tolerance, "{what}: got {got}, reference {want}");
}

#[test]
fn decisions_and_binomial_evsi_match_the_exact_rational_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/decision_exact_reference.json"))
            .expect("frozen reference");
    let cases = root["discrete"].as_array().expect("discrete cases");
    let (mut constrained, mut weighted, mut refused, mut ties) = (0, 0, 0, 0);
    for (index, case) in cases.iter().enumerate() {
        let thetas = floats(&case["thetas"]);
        let intercepts = floats(&case["intercepts"]);
        let slopes = floats(&case["slopes"]);
        let actions: Vec<usize> = (0..intercepts.len()).collect();
        let mut constraints: Vec<Arc<dyn DecisionConstraint<usize, f64>>> = Vec::new();
        if !case["cutoffs"].is_null() {
            constrained += 1;
            constraints.push(Arc::new(Cutoffs(floats(&case["cutoffs"]))));
        }
        let mut problem = DecisionProblem::new(
            actions,
            Arc::new(AffineUtility::new(intercepts, slopes).expect("utility")),
            constraints,
        );
        problem.chance_threshold = case["threshold"].as_f64().unwrap();
        let ns: Vec<u64> =
            case["ns"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        let prior = DecisionPrior::Draws(thetas.clone());
        if case["expect"].is_null() {
            refused += 1;
            assert!(
                matches!(
                    evaluate_decision(&problem, &thetas),
                    Err(DesignError::NoAdmissibleAction(_))
                ),
                "case {index}"
            );
            assert!(matches!(
                PreposteriorAnalysis::new(&problem, &prior, &BinomialSignal).err(),
                Some(DesignError::NoAdmissibleAction(_))
            ));
            continue;
        }
        let want = &case["expect"];
        let signal = BinomialSignal;
        let weights = (!case["weights"].is_null()).then(|| floats(&case["weights"]));
        let analysis = match &weights {
            Some(w) => {
                weighted += 1;
                PreposteriorAnalysis::new_weighted(&problem, &thetas, w, &signal)
            }
            None => {
                let evaluation = evaluate_decision(&problem, &thetas).expect("decision");
                assert_eq!(
                    evaluation.chosen_action,
                    want["chosen"].as_u64().unwrap() as usize,
                    "case {index}"
                );
                close(
                    evaluation.expected_utility,
                    want["expected_utility"].as_f64().unwrap(),
                    "eu",
                );
                close(evaluation.posterior_regret, want["regret"].as_f64().unwrap(), "regret");
                if let Some(satisfaction) = want["satisfaction"].as_array() {
                    let chosen = want["chosen"].as_u64().unwrap() as usize;
                    assert_eq!(
                        evaluation.chance_constraint_probs.as_ref(),
                        &[satisfaction[chosen].as_f64().unwrap()]
                    );
                }
                PreposteriorAnalysis::new(&problem, &prior, &signal)
            }
        }
        .unwrap_or_else(|e| panic!("case {index}: {e:?}"));
        if want["admissible"].as_array().unwrap().len() < intercepts_len(case) {
            // some action was excluded by the constraint
        }
        if intercepts_tie(case) {
            ties += 1;
        }
        assert_eq!(
            analysis.bayes_action(),
            want["chosen"].as_u64().unwrap() as usize,
            "case {index}"
        );
        close(analysis.prior_expected_utility(), want["prior_eu"].as_f64().unwrap(), "prior eu");
        close(
            analysis.expected_value_of_perfect_information(),
            want["evpi"].as_f64().unwrap(),
            &format!("case {index} evpi"),
        );
        for n in ns {
            let got = analysis.exact_evsi(n).expect("evsi").expect("finite support");
            close(
                got,
                want["evsi"][n.to_string()].as_f64().unwrap(),
                &format!("case {index} evsi({n})"),
            );
            assert!(got <= analysis.expected_value_of_perfect_information() + 1e-9);
        }
    }
    assert!(constrained > 30 && weighted > 20 && refused >= 5 && ties > 10);
}

fn intercepts_len(case: &Value) -> usize {
    case["intercepts"].as_array().unwrap().len()
}

fn intercepts_tie(case: &Value) -> bool {
    case["intercepts"][0] == case["intercepts"][1] && case["slopes"][0] == case["slopes"][1]
}
