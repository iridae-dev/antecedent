//! Model-based Bayesian effects on one arm, through the public `Study` flow.
//!
//! A nearest-neighbor matching contrast has no posterior: it is a rule applied to the sample, not
//! a model of the outcomes. The Bayesian counterpart of the matched effect on the treated is a
//! different estimand with a likelihood: the contrast of a declared Gaussian outcome model
//! averaged over the treated rows, with treatment-by-covariate interactions centred at the
//! treated means. Static data takes it through Bayesian g-computation with a `Treated` target
//! population; units with a history and a calendar period take it through the period-and-history
//! outcome model. Both are labelled as model-based, execute unlicensed with calibration
//! unmeasured, and leave the matching posterior refusal in place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    Assumption, AverageEffectQuery, CausalQuery, ExecutionContext, HistoryMatchQuery,
    TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2000).prior_scale(100.0))
}

fn has_parametric(result: &antecedent::StudyResult, id: &str) -> bool {
    result.posterior.as_ref().is_some_and(|posterior| {
        posterior.assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, Assumption::ParametricRestriction(p) if p.id.as_ref() == id)
        })
    })
}

/// Binary treatment confounded by `z`; the effect `1 + 2 z` grows in `z`, so the effects on the
/// treated and on the untreated differ. The treated truth is the mean of `1 + 2 z` over them.
fn static_fixture(n: usize) -> (TabularData, f64, f64) {
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 7.0).sin() * 1.5).collect();
    let t: Vec<f64> =
        (0..n).map(|i| if z[i] + (i as f64 / 3.0).cos() > 0.0 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> =
        (0..n).map(|i| t[i] * (1.0 + 2.0 * z[i]) + z[i] + (i as f64 / 5.0).sin() * 0.1).collect();
    let mean_effect = |arm: f64| {
        let rows: Vec<usize> = (0..n).filter(|&i| (t[i] - arm).abs() < 0.5).collect();
        rows.iter().map(|&i| 1.0 + 2.0 * z[i]).sum::<f64>() / rows.len() as f64
    };
    let (att, atc) = (mean_effect(1.0), mean_effect(0.0));
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    (data, att, atc)
}

fn static_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn static_effect(data: &TabularData, population: TargetPopulation) -> antecedent::StudyResult {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(population);
    Study::tabular(data.clone())
        .graph(static_dag())
        .query(query)
        .inference(bayesian())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(21))
        .unwrap()
}

#[test]
fn bayesian_gcomp_on_the_treated_and_untreated_recovers_each_arms_model_based_effect() {
    let (data, att, atc) = static_fixture(400);
    assert!((att - atc).abs() > 0.3, "the arms must differ for the check to bite");
    let treated = static_effect(&data, TargetPopulation::Treated);
    let untreated = static_effect(&data, TargetPopulation::Untreated);
    let ate_treated = treated.estimate().as_effect().unwrap().ate;
    let ate_untreated = untreated.estimate().as_effect().unwrap().ate;
    assert!((ate_treated - att).abs() < 0.06, "treated {ate_treated} vs {att}");
    assert!((ate_untreated - atc).abs() < 0.06, "untreated {ate_untreated} vs {atc}");
    for result in [&treated, &untreated] {
        assert!(result.posterior.is_some(), "the effect is a posterior, not a matching rule");
        assert!(has_parametric(result, "bayesian.gcomp.model_based_arm_effect"));
        assert!(has_parametric(result, "bayesian.gcomp.linear_gaussian_outcome"));
    }
    // The all-observed effect carries no arm label.
    let all = static_effect(&data, TargetPopulation::AllObserved);
    assert!(!has_parametric(&all, "bayesian.gcomp.model_based_arm_effect"));
}

#[test]
fn a_matching_estimator_under_bayesian_inference_keeps_its_typed_boundary() {
    let (data, _, _) = static_fixture(200);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let error = Study::tabular(data)
        .graph(static_dag())
        .query(query)
        .inference(bayesian())
        .estimator(EstimatorId::PropensityMatching)
        .bootstrap_replicates(0)
        .build()
        .expect_err("matching has no posterior");
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("Treated target population"), "{error}");
}

/// Three calendar periods of 60 units; the effect `1 + 2 * score` grows with the history mean,
/// treatment is likelier at a high score, and `weights` can break the equal-weight premise.
fn history_fixture(unequal_weights: bool) -> (TabularData, f64) {
    let (mut h0, mut h1, mut y, mut t, mut period, mut weight) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut effects = Vec::new();
    for p in 0..3_u32 {
        for k in 0..60_usize {
            let base = ((k as f64) * 0.37 + f64::from(p)).sin();
            let (a, b) = (base, base + 0.1 * ((k % 5) as f64 - 2.0));
            let score = (a + b) / 2.0;
            let treated = score + ((k as f64) * 0.11).cos() * 0.8 > 0.0;
            let effect = 1.0 + 2.0 * score;
            let wiggle = ((k as f64) * 1.3).sin() * 0.05;
            let outcome =
                1.5 * f64::from(p) + 0.5 * score + if treated { effect } else { 0.0 } + wiggle;
            if treated {
                effects.push(effect);
            }
            h0.push(a);
            h1.push(b);
            y.push(outcome);
            t.push(if treated { 1.0 } else { 0.0 });
            period.push(f64::from(p));
            weight.push(if unequal_weights && k % 2 == 0 { 2.0 } else { 1.0 });
        }
    }
    let truth = effects.iter().sum::<f64>() / effects.len() as f64;
    let data = TabularData::from_f64_columns([
        ("h0", h0.as_slice()),
        ("h1", h1.as_slice()),
        ("y", y.as_slice()),
        ("t", t.as_slice()),
        ("period", period.as_slice()),
        ("weight", weight.as_slice()),
    ])
    .unwrap();
    (data, truth)
}

fn history_query() -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: VariableId::from_raw(2),
        treated: VariableId::from_raw(3),
        history: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        period: Some(VariableId::from_raw(4)),
        weight: Some(VariableId::from_raw(5)),
        caliper: None,
    }
}

fn history_study(data: &TabularData, estimator: Option<EstimatorId>) -> antecedent::StudyBuilder {
    let builder = Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(history_query()))
        .inference(bayesian());
    match estimator {
        Some(id) => builder.estimator(id),
        None => builder,
    }
}

#[test]
fn the_history_outcome_model_reports_a_model_based_effect_on_the_treated() {
    let (data, truth) = history_fixture(false);
    let result = history_study(&data, Some(EstimatorId::BayesianPeriodHistoryOutcomeModel))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("bayesian.period_history_outcome_model")
    );
    assert_eq!(result.logical_plan.plan_id.as_ref(), "bayesian.period_history_outcome_model");
    let posterior = result.posterior.as_ref().expect("a posterior, not a matched contrast");
    let column = posterior.effect_column().expect("the effect on the treated");
    let ate = posterior.summaries.mean[column];
    assert!((ate - truth).abs() < 0.06, "posterior effect {ate} vs effect on treated {truth}");
    assert!(result.support_status().is_none(), "the route is unlicensed");
    let carries = |id: &str| {
        result.posterior.as_ref().unwrap().assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, Assumption::Custom { id: name, .. } if name.as_ref() == id)
        })
    };
    assert!(carries("bayesian.period_history_outcome_model.model_based_effect_on_treated"));
    assert!(carries("bayesian.period_history_outcome_model.linear_gaussian"));
}

#[test]
fn the_history_matching_posterior_is_refused_and_names_the_outcome_model_estimator() {
    let (data, _) = history_fixture(false);
    // No estimator: the matching contrast has no posterior.
    let error = history_study(&data, None).build().expect_err("matching has no posterior");
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    let text = error.to_string();
    assert!(text.contains("no likelihood"), "{text}");
    assert!(text.contains("bayesian.period_history_outcome_model"), "{text}");
    // The matching estimator selected under Bayesian inference is refused the same way.
    let error = history_study(&data, Some(EstimatorId::PeriodHistoryMatching))
        .build()
        .expect_err("matching has no posterior");
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    // The outcome-model estimator is Bayesian only.
    let error = Study::tabular(data)
        .query(CausalQuery::HistoryMatch(history_query()))
        .estimator(EstimatorId::BayesianPeriodHistoryOutcomeModel)
        .build()
        .expect_err("a posterior estimator under Frequentist inference");
    assert_eq!(error.reason_code(), Some("estimator_inference_mismatch"));
}

#[test]
fn the_history_outcome_model_refuses_unequal_sampling_weights_naming_the_next_attempt() {
    let (data, _) = history_fixture(true);
    let error = history_study(&data, Some(EstimatorId::BayesianPeriodHistoryOutcomeModel))
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(23)))
        .expect_err("sampling weights are not precisions");
    let text = error.to_string();
    assert!(text.contains("equal sampling weights"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}
