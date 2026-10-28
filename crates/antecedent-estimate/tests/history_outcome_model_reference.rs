//! The period-adjusted history outcome model against an exact Normal-Inverse-Gamma reference.
//!
//! `oracle_data/history_outcome_model_reference.py` (frozen in the JSON beside it) builds the
//! design of the declared model `y = a_period + g score + b t + d t (score - score_treated)` from
//! the units' histories, keeps only the periods that hold both arms, and gives the exact
//! Student-t posterior of `b` (the model-based effect on the treated, because the interaction is
//! centred at the treated mean score) in 40-digit arithmetic. The prepared problem fitted by the
//! conjugate Bayesian g-computation must reproduce its mean and standard deviation, the treated
//! units of a one-armed period must be reported rather than fitted, and unequal sampling
//! weights and mismatched history lengths must be refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IdentificationStatus};
use antecedent_estimate::history_match::HistoryUnit;
use antecedent_estimate::history_outcome_model::prepare_history_outcome_model;
use antecedent_estimate::{BayesianGCompWorkspace, BayesianGComputationAte};
use serde_json::Value;

const DRAWS: usize = 300_000;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/history_outcome_model_reference.json")).unwrap()
}

fn units(reference: &Value, weight: impl Fn(usize) -> f64) -> Vec<HistoryUnit> {
    reference["units"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let history: Vec<f64> =
                u["history"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            HistoryUnit::try_new(
                &history,
                u["outcome"].as_f64().unwrap(),
                u["treated"].as_bool().unwrap(),
                weight(i),
            )
            .unwrap()
            .in_period(u["period"].as_i64().unwrap())
        })
        .collect()
}

#[test]
fn the_effect_on_the_treated_matches_the_exact_posterior() {
    let reference = reference();
    let problem = prepare_history_outcome_model(&units(&reference, |_| 1.0)).unwrap();
    assert_eq!(problem.rows as u64, reference["rows"].as_u64().unwrap());
    assert_eq!(problem.treated_rows as u64, reference["treated_rows"].as_u64().unwrap());
    assert_eq!(problem.periods as u64, reference["periods"].as_u64().unwrap());
    let unused: Vec<u64> = reference["unused_treated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(problem.unused_treated.iter().map(|&i| i as u64).collect::<Vec<_>>(), unused);

    let posterior = BayesianGComputationAte::conjugate()
        .with_n_draws(DRAWS)
        .with_prior_scale(reference["prior_scale"].as_f64().unwrap())
        .with_seed(13)
        .fit(
            &problem.prepared,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let column = posterior.effect_column().unwrap();
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let (r_mean, r_sd) = (reference["mean"].as_f64().unwrap(), reference["sd"].as_f64().unwrap());
    let root = (DRAWS as f64).sqrt();
    assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "mean {mean} vs {r_mean}");
    assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "sd {sd} vs {r_sd}");
}

#[test]
fn unequal_sampling_weights_and_ragged_histories_are_refused() {
    let reference = reference();
    assert!(prepare_history_outcome_model(&units(&reference, |i| 1.0 + (i % 2) as f64)).is_err());
    let mut ragged = units(&reference, |_| 1.0);
    ragged[0] = HistoryUnit::try_new(&[0.1, 0.2], 1.0, true, 1.0).unwrap().in_period(10);
    assert!(prepare_history_outcome_model(&ragged).is_err());
}
