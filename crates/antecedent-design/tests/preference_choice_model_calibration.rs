//! Whole-method coverage of the choice-model parameter interval: the `theta_hat +- z se`
//! interval of a trade-off parameter, with the delta-method standard error from the inverse
//! observed information, scored against the trade-off the choices were simulated from.
//!
//! Nothing here is measured at implementation time: the test is `#[ignore]`d and runs only at
//! the final measurement step, and the interval reads as measured only once its record exists.
//!
//! Design (`choice_model_trade_off_one_parameter_l95`): one decision-maker, three options per
//! record, value `x0 + t * x1` with the fixed part `x0` as the unit of account. Each option's
//! `x0` is uniform on `(-2, 2)` and `x1` uniform on `(0, 2)`; the chosen option maximises
//! `lambda * V + Gumbel noise` with consistency `lambda = 1` and trade-off `t = 1.5`, so the
//! choices follow the logit model the estimate assumes. Records on the three-point grid around
//! 200. The record is keyed by the runtime construction through [`calibration_basis`], never by
//! a hand-declared key.
//!
//! The truth is fixed, so the reading is pointwise frequentist coverage of the trade-off at this
//! design; it says nothing about choices that do not follow the logit model.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

// The calibration helpers are the binary's own module (this file runs inside the
// decision_reliability_interval_calibration target).
use super::calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::preference_choice_model::{
    CALIBRATION_INTERVAL_METHOD, ChoiceModelOptions, CovarianceKind, calibration_basis,
    estimate_choice_model,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as P, PreferenceParameter, PreferenceSet, ValueModel,
};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim, uniform};

const LEVEL: f64 = 0.95;
const TRUE_TRADE_OFF: f64 = 1.5;
const CONSISTENCY: f64 = 1.0;
const OPTIONS: usize = 3;
const DGP: &str =
    "one_decider_three_options_uniform_attributes_logit_choices_consistency_1_trade_off_1.5";

fn model() -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "cost".into(), units: "usd".into() },
            Attribute { id: "benefit".into(), units: "events".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "t".into(),
                units: "usd/event".into(),
                lower: -10.0,
                upper: 10.0,
                assumption_id: "declared_trade_off".into(),
            }],
            Vec::new(),
        ),
        value: P::sum(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        utility_units: "usd".into(),
    }
}

/// `n` records whose choices follow the logit model at the true consistency and trade-off.
fn records(model: &ValueModel, n: usize, seed: u64) -> ChoiceRecordSet {
    let mut next = uniform(seed, 31);
    let list = (0..n)
        .map(|index| {
            let means: Vec<Vec<f64>> =
                (0..OPTIONS).map(|_| vec![4.0 * next() - 2.0, 2.0 * next()]).collect();
            let chosen = means
                .iter()
                .map(|option| {
                    let gumbel = -(-next().max(f64::MIN_POSITIVE).ln()).ln();
                    CONSISTENCY * (option[0] + TRUE_TRADE_OFF * option[1]) + gumbel
                })
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map_or(0, |(position, _)| position);
            ChoiceRecord {
                id: format!("r{index}"),
                decider: "decider".into(),
                options: means
                    .into_iter()
                    .enumerate()
                    .map(|(position, option)| ChoiceOption {
                        id: format!("o{position}"),
                        belief: OptionBelief::Means(option),
                    })
                    .collect(),
                chosen: format!("o{chosen}"),
                belief_source: BeliefSource::RecordedForecast { source_id: "simulated".into() },
                observation_rows: vec![format!("row-{index}")],
            }
        })
        .collect();
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "simulated decision makers".into(),
        records: list,
        pooling: Pooling::Pooled { assumption_id: "assumption.one_decider".into() },
    }
}

fn trade_off(test: &'static str, expected_id: &str) {
    let n = grid_n(200);
    let model = model();
    let options =
        ChoiceModelOptions { level: LEVEL, n_draws: 200, ..ChoiceModelOptions::default() };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: DGP, interval: CALIBRATION_INTERVAL_METHOD },
        LEVEL,
    )
    .measures("capability", "preference_learning.choice_model");
    let basis = calibration_basis(CovarianceKind::Model, 1, n as u64, LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x7072_0000 + replicate);
        let set = records(&model, n, seed);
        let estimate = *estimate_choice_model(&model, &set, options).ok()?.pooled?;
        if estimate.covariance_kind != CovarianceKind::Model || estimate.n_records != n {
            return None;
        }
        let parameter = estimate.parameters.first()?;
        Some(parameter.interval)
    });
    for result in results {
        let Some(interval) = result else {
            tally.skip();
            continue;
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some(interval), TRUE_TRADE_OFF);
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn choice_model_trade_off_one_parameter_l95() {
    trade_off(
        "choice_model_trade_off_one_parameter_l95",
        "cov.preference_trade_off_from_choices.not_applicable.frequentist.wald_delta_method.l95.choice_model_trade_off_one_parameter_l95.v2_validation_not_applicable",
    );
}
