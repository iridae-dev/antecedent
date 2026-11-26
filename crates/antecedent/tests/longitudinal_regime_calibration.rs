//! Repeated-sampling coverage and publication for four randomized decisions.
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{CausalError, PreparedStudy, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, LongitudinalCalibrationDesign, LongitudinalRegimeMethod,
    LongitudinalRegimeQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::error::EstimationError;
use antecedent_stats::StatsError;
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, gaussian, grid_n, map_replicates, n_sim, uniform,
};
use common::calibration_bind::bind_reported;
use std::sync::Arc;

const PERIODS: usize = 4;
const TRUTH: f64 = 6.0;

fn draw(
    subjects: usize,
    p: f64,
    c: f64,
    seed: u64,
) -> (TabularData, LongitudinalRegimeQuery, usize) {
    let mut action_draw = uniform(seed, 101);
    let mut censor_draw = uniform(seed, 102);
    let mut noise_draw = gaussian(seed ^ 103);
    let cells = subjects * PERIODS;
    let mut outcomes = Vec::with_capacity(subjects);
    let mut treatment = Vec::with_capacity(cells);
    let mut observation = Vec::with_capacity(cells);
    let mut outcome_observed = Vec::with_capacity(subjects);
    let mut predictions = Vec::with_capacity(cells);
    let mut matches = 0;
    let folds: Arc<[u32]> = (0..subjects).map(|i| u32::try_from(i % 5).unwrap()).collect();
    for &fold in &folds[..] {
        let mut at_risk = true;
        let mut treated_periods = 0;
        for t in 0..PERIODS {
            let treated = action_draw() < p;
            treated_periods += u32::from(treated);
            let retained = censor_draw() < c;
            at_risk &= retained;
            treatment.push(treated);
            observation.push(at_risk);
            // Frozen external predictions indexed by fold. Small deterministic offsets
            // give misspecified Q values without learning from any evaluation outcomes.
            predictions.push(TRUTH + (f64::from(fold) - 2.0) / 16.0 + (t as f64 - 1.5) / 32.0);
        }
        outcomes.push(2.0 + f64::from(treated_periods) + noise_draw());
        outcome_observed.push(at_risk);
        matches += usize::from(at_risk && treated_periods == 4);
    }
    let query = LongitudinalRegimeQuery {
        calibration_design: None,
        outcome: VariableId::from_raw(0),
        method: LongitudinalRegimeMethod::SequentialDoublyRobust,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: Some(predictions.into()),
        observation_history: Some(observation.into()),
        prediction_fold_ids: Some(folds.clone()),
        periods: PERIODS,
        treatment_history: treatment.into(),
        regime_actions: vec![true; cells].into(),
        treatment_probabilities: vec![p; cells].into(),
        censoring_probabilities: vec![c; cells].into(),
        outcome_observed: outcome_observed.into(),
        subject_ids: (0..subjects).map(|i| Arc::from(format!("subject-{i}"))).collect(),
        fold_ids: folds,
        excluded_fold_predictions: true,
        probabilities_known_by_design: true,
        minimum_probability: 0.01,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    (data, query, matches)
}

fn run(
    data: &TabularData,
    query: LongitudinalRegimeQuery,
    seed: u64,
) -> Result<(PreparedStudy, StudyResult), CausalError> {
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(data, &ctx)?;
    Ok((prepared, result))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn four_period_sequential_dr_coverage() {
    let subjects = grid_n(4_800);
    let datasets = map_replicates(n_sim(), |seed| {
        let (data, mut query, _) = draw(subjects, 0.5, 0.95, seed ^ 0x4911_0000);
        query.calibration_design = Some(LongitudinalCalibrationDesign::FourPeriodSequentialDr);
        let (prepared, result) = run(&data, query, seed).unwrap();
        let bounds = result.longitudinal_regime().unwrap().value_interval_95;
        let reported = bounds.map(|_| {
            let contract = prepared.contract_for_result(&result).unwrap();
            common::calibration_bind::constructions(&contract, &result)
        });
        (bounds, reported)
    });
    let mut coverage = CoverageTally::for_record(
        RecordKey {
            test: "four_period_sequential_dr_coverage",
            dgp: "four_randomized_fair_coin_decisions_monotone_censoring_additive_gaussian_value",
            interval: "analytic_se",
        },
        REPORTED_LEVEL,
    );
    let attempted = datasets.len();
    let mut published = 0;
    for (bounds, reported) in datasets {
        if let Some(reported) = &reported {
            bind_reported(&mut coverage, reported);
            published += 1;
        }
        coverage.record(bounds.map(|[lo, hi]| (lo, hi)), TRUTH);
    }
    eprintln!("four-period DR publication: {published}/{attempted}, subjects={subjects}");
    coverage.assert();
}

#[test]
fn four_period_sequential_dr_coverage_and_publication() {
    // Separate from the calibration key, these draws measure selective publication
    // and conditional-on-publication coverage across the lower support boundary.
    let subjects = 2_400;
    let replicates = 128;
    for (p, c, publication_range) in [
        (0.5, 1.0, 125..=128),
        (0.5, 0.95, 125..=128),
        (0.45, 0.95, 65..=120),
        (0.44, 0.95, 0..=0),
        (0.5, 0.94, 0..=0),
    ] {
        let mut published = 0;
        let mut covered = 0;
        let mut min_matches = subjects;
        let mut max_matches = 0;
        for replicate in 0..replicates {
            let seed = replicate ^ 0x4912_0000;
            let (data, query, matches) = draw(subjects, p, c, seed);
            let observed = query.outcome_observed.iter().filter(|&&on| on).count();
            let (_, result) = run(&data, query, seed).unwrap();
            let bounds = result.longitudinal_regime().unwrap().value_interval_95;
            min_matches = min_matches.min(matches);
            max_matches = max_matches.max(matches);
            assert_eq!(
                bounds.is_some(),
                p >= 0.45 && c >= 0.95 && matches >= 75 && observed >= 1_800
            );
            if let Some([lo, hi]) = bounds {
                published += 1;
                covered += usize::from(lo <= TRUTH && TRUTH <= hi);
            }
        }
        eprintln!(
            "p={p}, censoring survival={c}, observed regime matches={min_matches}..={max_matches}, published={published}/{replicates}, covered among published={covered}/{published}"
        );
        assert!(
            publication_range.contains(&published),
            "unexpected publication at p={p}, c={c}: {published}"
        );
        if published > 0 {
            assert!(
                covered * 100 >= 85 * published,
                "low coverage at p={p}, c={c}: {covered}/{published}"
            );
        }
    }
}

fn fitted_q_coverage(test: &'static str, external_training_fold: Option<u32>) {
    let subjects = grid_n(800);
    let datasets = map_replicates(n_sim(), |seed| {
        let seed = seed ^ 0x4913_0000;
        let mut assignment = uniform(seed, 121);
        let mut baseline_draw = uniform(seed, 122);
        let mut noise = gaussian(seed ^ 123);
        let mut baseline = Vec::with_capacity(subjects);
        let mut actions = Vec::with_capacity(subjects * 2);
        let mut outcomes = Vec::with_capacity(subjects);
        for _ in 0..subjects {
            let x = f64::from(baseline_draw() < 0.5);
            let a0 = assignment() < 0.5;
            let a1 = assignment() < 0.5;
            baseline.push(x);
            actions.extend([a0, a1]);
            outcomes.push(2.0 + 0.5 * f64::from(a0) + 0.75 * f64::from(a1) + x + noise());
        }
        let data = TabularData::from_f64_columns([
            ("outcome", outcomes.as_slice()),
            ("baseline", baseline.as_slice()),
        ])
        .unwrap();
        let query = LongitudinalRegimeQuery {
            calibration_design: Some(if external_training_fold.is_some() {
                LongitudinalCalibrationDesign::FittedQExternalSplit
            } else {
                LongitudinalCalibrationDesign::FittedQInternalCrossfit
            }),
            outcome: VariableId::from_raw(0),
            method: LongitudinalRegimeMethod::FittedGFormula(antecedent_core::FittedRegimeQ {
                baseline: VariableId::from_raw(1),
                external_training_fold,
                bootstrap_replicates: 199,
            }),
            period_outcome_predictions: None,
            known_fixed_outcome_predictions: false,
            stabilizing_numerator_probabilities: None,
            q_predictions: None,
            observation_history: None,
            prediction_fold_ids: None,
            periods: 2,
            treatment_history: actions.into(),
            regime_actions: vec![true; subjects * 2].into(),
            treatment_probabilities: vec![0.5; subjects * 2].into(),
            censoring_probabilities: vec![1.0; subjects * 2].into(),
            outcome_observed: vec![true; subjects].into(),
            subject_ids: (0..subjects).map(|i| Arc::from(format!("fitted-{i}"))).collect(),
            fold_ids: (0..subjects).map(|i| u32::try_from(i % 2).unwrap()).collect(),
            excluded_fold_predictions: false,
            probabilities_known_by_design: true,
            minimum_probability: 0.01,
            rule_id: None,
            rule_version: None,
            rule_provenance: None,
        };
        let (prepared, result) = match run(&data, query, seed) {
            Ok(fit) => fit,
            Err(CausalError::Estimate(EstimationError::Stats(StatsError::Unsupported {
                ..
            }))) => {
                return None;
            }
            Err(error) => panic!("unexpected fitted-Q calibration failure: {error}"),
        };
        let fit = result.longitudinal_regime().unwrap();
        // Only a full 199-refit interval is eligible for this coverage record.
        // Incomplete procedures still count against repeated-study coverage.
        if fit.fitted_q_bootstrap_replicates_ok != Some(199) {
            return None;
        }
        let contract = prepared.contract_for_result(&result).unwrap();
        Some((fit.value_interval_95, common::calibration_bind::constructions(&contract, &result)))
    });
    let mut coverage = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "two_fair_coin_decisions_binary_baseline_additive_normal_terminal_reward",
            interval: "percentile_bootstrap",
        },
        REPORTED_LEVEL,
    );
    for dataset in datasets {
        if let Some((bounds, reported)) = dataset {
            bind_reported(&mut coverage, &reported);
            coverage.record(bounds.map(|[lo, hi]| (lo, hi)), 3.75);
        } else {
            coverage.skip();
        }
    }
    coverage.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn two_period_fitted_q_internal_crossfit_coverage() {
    fitted_q_coverage("two_period_fitted_q_internal_crossfit_coverage", None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn two_period_fitted_q_external_split_coverage() {
    fitted_q_coverage("two_period_fitted_q_external_split_coverage", Some(0));
}
