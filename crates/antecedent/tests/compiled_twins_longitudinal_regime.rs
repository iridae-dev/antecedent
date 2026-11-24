//! The facade longitudinal regime evaluators and the compiled `LongitudinalRegime` route give
//! identical numbers.
//!
//! `evaluate_regime_value`, `evaluate_g_formula_value` and `evaluate_sequential_dr_value` are
//! the kernel calls the compiled route makes for its ipw, g-formula and sequential doubly
//! robust methods (the last through its subject-score variant, which the plain function
//! wraps). The assertions are on `to_bits()` of the value, its summary fields and, where the
//! support floors let the compiled route publish one, the standard error and interval bounds
//! against the kernel interval functions applied to the direct summary.
#![allow(deprecated, reason = "the deprecated facade evaluators are the paths under comparison")]
#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "bit-identical equality is the property under test; fixtures widen small integers"
)]

use std::sync::Arc;

use antecedent::estimate::longitudinal_regime::{
    SequentialDrInputs, SubjectHistories, evaluate_g_formula_value, evaluate_regime_value,
    evaluate_sequential_dr_value,
};
use antecedent::prelude::ExecutionContext;
use antecedent::{LongitudinalRegimeEstimate, Study};
use antecedent_core::{CausalQuery, LongitudinalRegimeMethod, LongitudinalRegimeQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::longitudinal_regime::{
    RegimeValueSummary, evaluate_sequential_dr_value_with_subject_scores,
    g_formula_fixed_q_pointwise_interval_95, ipw_pointwise_interval_95,
    sequential_dr_pointwise_interval_95,
};

const SUBJECTS: usize = 600;
const PERIODS: usize = 2;
const FLOOR: f64 = 0.01;

struct Fixture {
    outcome: Vec<f64>,
    treatment: Vec<bool>,
    actions: Vec<bool>,
    treatment_probability: Vec<f64>,
    censoring_probability: Vec<f64>,
    observed: Vec<bool>,
    observation_history: Vec<bool>,
    rewards: Vec<f64>,
    q_predictions: Vec<f64>,
    folds: Vec<u32>,
}

fn fixture() -> Fixture {
    let cells = SUBJECTS * PERIODS;
    let mut fixture = Fixture {
        outcome: Vec::new(),
        treatment: Vec::new(),
        actions: vec![true; cells],
        treatment_probability: Vec::new(),
        censoring_probability: Vec::new(),
        observed: vec![true; SUBJECTS],
        observation_history: vec![true; cells],
        rewards: Vec::new(),
        q_predictions: Vec::new(),
        folds: (0..SUBJECTS).map(|i| (i % 2) as u32).collect(),
    };
    for i in 0..SUBJECTS {
        let mut outcome = 1.0 + 0.01 * (i % 17) as f64;
        for t in 0..PERIODS {
            let treated = (i * (t + 3) + t * 5 + i / 4) % 5 < 3;
            fixture.treatment.push(treated);
            fixture.treatment_probability.push(0.4 + 0.02 * ((i + 2 * t) % 6) as f64);
            fixture.censoring_probability.push(1.0 - 0.03 * ((3 * i + t) % 5) as f64);
            fixture.rewards.push(0.5 + 0.2 * t as f64 + 0.005 * (i % 13) as f64);
            fixture.q_predictions.push(0.3 + 0.1 * t as f64 + 0.01 * (i % 11) as f64);
            outcome += if treated { 0.5 + 0.3 * t as f64 } else { 0.0 };
        }
        fixture.outcome.push(outcome);
    }
    fixture
}

fn histories(fixture: &Fixture) -> SubjectHistories<'_> {
    SubjectHistories {
        treatment: &fixture.treatment,
        treatment_probability: &fixture.treatment_probability,
        outcome_observed: &fixture.observed,
        censoring_probability: &fixture.censoring_probability,
        periods: PERIODS,
        minimum_probability: FLOOR,
    }
}

fn query(fixture: &Fixture, method: LongitudinalRegimeMethod) -> LongitudinalRegimeQuery {
    LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: None,
        observation_history: None,
        prediction_fold_ids: None,
        periods: PERIODS,
        treatment_history: Arc::from(fixture.treatment.as_slice()),
        regime_actions: Arc::from(fixture.actions.as_slice()),
        treatment_probabilities: Arc::from(fixture.treatment_probability.as_slice()),
        censoring_probabilities: Arc::from(fixture.censoring_probability.as_slice()),
        outcome_observed: Arc::from(fixture.observed.as_slice()),
        subject_ids: (0..SUBJECTS).map(|i| Arc::<str>::from(format!("s{i}"))).collect(),
        fold_ids: Arc::from(fixture.folds.as_slice()),
        excluded_fold_predictions: false,
        probabilities_known_by_design: true,
        minimum_probability: FLOOR,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    }
}

fn run(fixture: &Fixture, query: LongitudinalRegimeQuery) -> LongitudinalRegimeEstimate {
    let data = TabularData::from_f64_columns([("outcome", fixture.outcome.as_slice())]).unwrap();
    let context = ExecutionContext::for_tests(23);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    result.longitudinal_regime().unwrap().clone()
}

fn assert_summary_matches(compiled: &LongitudinalRegimeEstimate, direct: &RegimeValueSummary) {
    assert_eq!(compiled.value.to_bits(), direct.value.to_bits(), "value");
    assert_eq!(
        compiled.effective_sample_size.to_bits(),
        direct.effective_sample_size.to_bits(),
        "effective sample size"
    );
    assert_eq!(
        compiled.matched_observed_fraction.to_bits(),
        direct.matched_observed_fraction.to_bits(),
        "matched fraction"
    );
    assert_eq!(compiled.maximum_weight.to_bits(), direct.maximum_weight.to_bits(), "max weight");
    assert_eq!(
        compiled.minimum_action_probability.to_bits(),
        direct.minimum_action_probability.to_bits(),
        "minimum action probability"
    );
    assert_eq!(
        compiled.minimum_censoring_probability.to_bits(),
        direct.minimum_censoring_probability.to_bits(),
        "minimum censoring probability"
    );
}

fn bits(interval: Option<[f64; 2]>) -> Option<[u64; 2]> {
    interval.map(|bounds| bounds.map(f64::to_bits))
}

#[test]
fn compiled_ipw_regime_reproduces_the_direct_value_and_interval() {
    let fixture = fixture();
    let direct =
        evaluate_regime_value(&fixture.outcome, &histories(&fixture), &fixture.actions).unwrap();
    let compiled = run(&fixture, query(&fixture, LongitudinalRegimeMethod::Ipw));
    assert_eq!(&*compiled.method, "ipw");
    assert_summary_matches(&compiled, &direct);
    assert_eq!(
        compiled.value_standard_error.map(f64::to_bits),
        direct.score_standard_error.map(f64::to_bits)
    );
    // The interval is the kernel interval of the direct summary, or absent for both.
    assert_eq!(
        bits(compiled.value_interval_95),
        bits(ipw_pointwise_interval_95(&direct, SUBJECTS))
    );
    assert!(compiled.value_interval_95.is_some(), "the fixture must clear the interval floors");
}

#[test]
fn compiled_g_formula_regime_reproduces_the_direct_value_and_interval() {
    let fixture = fixture();
    let direct = evaluate_g_formula_value(
        &fixture.rewards,
        &fixture.actions,
        &fixture.treatment_probability,
        &fixture.censoring_probability,
        SUBJECTS,
        PERIODS,
        FLOOR,
    )
    .unwrap();
    let mut q = query(&fixture, LongitudinalRegimeMethod::GFormula);
    q.period_outcome_predictions = Some(Arc::from(fixture.rewards.as_slice()));
    q.known_fixed_outcome_predictions = true;
    let compiled = run(&fixture, q);
    assert_eq!(&*compiled.method, "g_formula");
    assert_summary_matches(&compiled, &direct);
    let kernel = g_formula_fixed_q_pointwise_interval_95(&direct, &fixture.rewards, SUBJECTS, 2);
    assert_eq!(bits(compiled.value_interval_95), bits(kernel.map(|(_, bounds)| bounds)));
    assert_eq!(compiled.value_standard_error.map(f64::to_bits), kernel.map(|(se, _)| se.to_bits()));
    assert!(compiled.value_interval_95.is_some(), "the fixture must clear the interval floors");
}

#[test]
fn compiled_sequential_dr_regime_reproduces_the_direct_value_and_interval() {
    let fixture = fixture();
    let inputs = SequentialDrInputs {
        actions: &fixture.actions,
        observation_history: &fixture.observation_history,
        q_predictions: &fixture.q_predictions,
    };
    let direct =
        evaluate_sequential_dr_value(&fixture.outcome, &histories(&fixture), &inputs).unwrap();
    let (with_scores, scores) = evaluate_sequential_dr_value_with_subject_scores(
        &fixture.outcome,
        &histories(&fixture),
        &inputs,
    )
    .unwrap();
    assert_eq!(with_scores.value.to_bits(), direct.value.to_bits());
    let mut q = query(&fixture, LongitudinalRegimeMethod::SequentialDoublyRobust);
    q.q_predictions = Some(Arc::from(fixture.q_predictions.as_slice()));
    q.observation_history = Some(Arc::from(fixture.observation_history.as_slice()));
    q.prediction_fold_ids = Some(Arc::from(fixture.folds.as_slice()));
    q.excluded_fold_predictions = true;
    let compiled = run(&fixture, q);
    assert_eq!(&*compiled.method, "sequential_dr");
    assert_summary_matches(&compiled, &direct);
    let matched = (0..SUBJECTS)
        .filter(|&i| {
            (0..PERIODS)
                .all(|t| fixture.treatment[i * PERIODS + t] == fixture.actions[i * PERIODS + t])
        })
        .count();
    let kernel = sequential_dr_pointwise_interval_95(&direct, &scores, PERIODS, SUBJECTS, matched);
    assert_eq!(bits(compiled.value_interval_95), bits(kernel.map(|(_, bounds)| bounds)));
    assert_eq!(compiled.value_standard_error.map(f64::to_bits), kernel.map(|(se, _)| se.to_bits()));
}
