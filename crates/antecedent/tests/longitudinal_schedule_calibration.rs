//! Coverage of an entire fixed contrast family and of learned schedule statements.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::{
    CausalQuery, ExecutionContext, LongitudinalRegimeMethod, LongitudinalRegimeQuery,
    LongitudinalScheduleFamily, LongitudinalScheduleTarget, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, gaussian, grid_n, map_replicates,
    n_sim, uniform,
};
use common::calibration_bind::{bind_reported, constructions};

fn evaluate(
    subjects: usize,
    means: [f64; 4],
    target: LongitudinalScheduleTarget,
    seed: u64,
) -> (antecedent::PreparedStudy, antecedent::StudyResult) {
    let mut assigned = uniform(seed, 211);
    let mut noise = gaussian(seed, 212);
    let mut histories = Vec::with_capacity(subjects * 2);
    let mut endpoint = Vec::with_capacity(subjects);
    for _ in 0..subjects {
        let first = assigned() < 0.5;
        let second = assigned() < 0.5;
        histories.extend([first, second]);
        endpoint.push(means[2 * usize::from(first) + usize::from(second)] + noise());
    }
    let query = LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: LongitudinalRegimeMethod::ScheduleFamily(LongitudinalScheduleFamily {
            schedules: 0b1111,
            target,
            draws: 199,
        }),
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: 2,
        treatment_history: histories.into(),
        regime_actions: vec![false; subjects * 2].into(),
        treatment_probabilities: vec![0.5; subjects * 2].into(),
        censoring_probabilities: vec![1.0; subjects * 2].into(),
        outcome_observed: vec![true; subjects].into(),
        subject_ids: (0..subjects).map(|i| Arc::from(format!("schedule-{i}"))).collect(),
        fold_ids: (0..subjects).map(|i| u32::try_from(i % 2).unwrap()).collect(),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    let table = TabularData::from_f64_columns([("endpoint", endpoint.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(table.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&table, &ctx).unwrap();
    (prepared, result)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn fixed_longitudinal_schedule_family_joint_coverage() {
    let subjects = grid_n(800);
    let means = [1.0, 2.0, 3.0, 4.0];
    let observations = map_replicates(n_sim(), |seed| {
        let (prepared, result) = evaluate(
            subjects,
            means,
            LongitudinalScheduleTarget::Contrasts { reference: 0, contrasts: 0b1110 },
            seed ^ 0x5319_1020,
        );
        let family = result.longitudinal_regime().unwrap().schedule_family.as_ref().unwrap();
        let statistic = family
            .contrasts
            .iter()
            .map(|contrast| {
                let truth = means[contrast.schedule as usize] - means[contrast.reference as usize];
                ((contrast.difference - truth) / contrast.standard_error).abs()
            })
            .fold(0.0_f64, f64::max);
        let reported = constructions(&prepared.contract_for_result(&result).unwrap(), &result);
        (family.critical_value, statistic, reported)
    });
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "fixed_longitudinal_schedule_family_joint_coverage",
            dgp: "independent_fair_coin_decisions_four_known_gaussian_schedule_values",
            interval: "simultaneous_band",
        },
        REPORTED_LEVEL,
    );
    for (critical, statistic, reported) in observations {
        bind_reported(&mut tally, &reported);
        // This event is exactly coverage of all three contrasts by the reported family.
        tally.record(Some((0.0, critical)), statistic);
    }
    tally.assert();
}

