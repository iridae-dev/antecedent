//! The posterior probabilities and expected losses behind `expected_loss_stopping`, and the
//! stop-or-continue decision itself, against exact rational values on a grid of counts
//! (`antecedent-stats/tests/oracle_data/procedure_decision_reference.py`).
//!
//! The reference evaluates P(Y > X) for integer Beta shapes through the binomial tail of the
//! larger-shape arm and the Beta integral of each monomial, and the expected loss through
//! E[(Y - X)+] = E[Y; Y > X] - E[X; Y > X]; the product integrates numerically. Thresholds are set
//! just above and below each exact value, so a decision flips exactly when the exact value
//! crosses; the arm named best is the arm with the larger exact probability of being best, with
//! ties naming the first arm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::adaptive_inference::{
    ArmCounts, PosteriorRule, bayesian_posterior, expected_loss_stopping,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../antecedent-stats/tests/oracle_data/procedure_decision_reference.json"
    ))
    .expect("frozen reference")
}

fn two(value: &Value) -> [f64; 2] {
    let a = value.as_array().expect("pair");
    [a[0].as_f64().unwrap(), a[1].as_f64().unwrap()]
}

#[test]
fn posterior_probabilities_losses_and_decisions_match_the_exact_values() {
    let reference = reference();
    let ctx = ExecutionContext::for_tests(5);
    let rows = reference["posteriors"].as_array().expect("rows");
    assert_eq!(rows.len(), 100);
    for row in rows {
        let c: Vec<u64> =
            row["counts"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        let counts = [
            ArmCounts { successes: c[0], failures: c[1] },
            ArmCounts { successes: c[2], failures: c[3] },
        ];
        let posterior = bayesian_posterior(1.0, 1.0, &counts, &ctx).expect("a posterior");
        let (best, loss) = (two(&row["best"]), two(&row["loss"]));
        for arm in 0..2 {
            assert!(
                (posterior.arms[arm].probability_best - best[arm]).abs() <= 1e-9,
                "{c:?} arm {arm}: best {} vs {}",
                posterior.arms[arm].probability_best,
                best[arm]
            );
            assert!(
                (posterior.arms[arm].expected_loss - loss[arm]).abs() <= 1e-9,
                "{c:?} arm {arm}: loss {} vs {}",
                posterior.arms[arm].expected_loss,
                loss[arm]
            );
        }
        let exact_arm = usize::from(best[1] > best[0]);
        let exact_best = best[exact_arm];
        let exact_loss = loss[exact_arm];
        for (offset, stops) in [(-1e-6, true), (1e-6, false)] {
            let threshold = exact_best + offset;
            if threshold > 0.5 && threshold < 1.0 {
                let decision = expected_loss_stopping(
                    &posterior,
                    PosteriorRule::ProbabilityBest { threshold },
                )
                .expect("a decision");
                assert_eq!(decision.best_arm, exact_arm, "{c:?}: best arm");
                assert_eq!(decision.stop, stops, "{c:?}: probability-best stop at {threshold}");
            }
            let loss_threshold = exact_loss + offset;
            if loss_threshold > 0.0 {
                let decision = expected_loss_stopping(
                    &posterior,
                    PosteriorRule::ExpectedLoss { loss_threshold },
                )
                .expect("a decision");
                assert_eq!(decision.best_arm, exact_arm, "{c:?}: best arm for loss");
                assert_eq!(decision.stop, !stops, "{c:?}: loss stop at {loss_threshold}");
            }
        }
    }
}
