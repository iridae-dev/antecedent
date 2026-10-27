//! Horizon-one specializations whose design is a static route.
//!
//! Each number below is computed from the rows in this file: a mean contrast, a
//! structural coefficient, or the normal equations of the additive regression the
//! interaction functional is not. A route matches that reference. The two routes
//! are not checked by setting one equal to the other.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    CausalQuery, Dag, DenseNodeId, InferenceMode, RefuteSuite, Study, StudyResult, TemporalDag,
};
use antecedent_contract::DataClassification;
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, IntervalMethod, Lag,
    MediationContrast, MediationQuery, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::EstimandMethod;
use antecedent_graph::ensure_lagged;

const CONTAMINANT: f64 = 100.0;
const TOLERANCE: f64 = 1e-8;

fn id(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= TOLERANCE
}

/// Treated mean minus control mean. Arms are the 0/1 values in `treatment`.
fn mean_contrast(treatment: &[f64], outcome: &[f64]) -> f64 {
    let mut treated = (0.0, 0.0);
    let mut control = (0.0, 0.0);
    for (&arm, &outcome) in treatment.iter().zip(outcome) {
        let bucket = if arm > 0.5 { &mut treated } else { &mut control };
        bucket.0 += outcome;
        bucket.1 += 1.0;
    }
    treated.0 / treated.1 - control.0 / control.1
}

/// Classical homoskedastic SE of a binary mean contrast, `RSS / (n − 2)`.
fn homoskedastic_contrast_se(treatment: &[f64], outcome: &[f64]) -> f64 {
    let (mut n0, mut n1) = (0.0, 0.0);
    let (mut sum0, mut sum1) = (0.0, 0.0);
    for &arm in treatment {
        if arm > 0.5 {
            n1 += 1.0;
        } else {
            n0 += 1.0;
        }
    }
    for (&arm, &outcome) in treatment.iter().zip(outcome) {
        if arm > 0.5 {
            sum1 += outcome;
        } else {
            sum0 += outcome;
        }
    }
    let (mean0, mean1) = (sum0 / n0, sum1 / n1);
    let rss = treatment.iter().zip(outcome).fold(0.0, |sum, (&arm, &outcome)| {
        let mean = if arm > 0.5 { mean1 } else { mean0 };
        sum + (outcome - mean).powi(2)
    });
    let n = n0 + n1;
    (rss / (n - 2.0) * (1.0 / n0 + 1.0 / n1)).sqrt()
}

/// Static rows, then one extra time. Treatment-time columns keep the static
/// values and a trailing contaminant. Outcome-time columns start with a
/// contaminant and then the static values. Lag-one alignment drops both.
fn lag_one_series(columns: &[(&str, &[f64])], treatment_time: &[&str]) -> TimeSeriesData {
    let n = columns[0].1.len();
    let owned: Vec<(String, Vec<f64>)> = columns
        .iter()
        .map(|(name, values)| {
            let mut series = Vec::with_capacity(n + 1);
            if treatment_time.contains(name) {
                series.extend_from_slice(values);
                series.push(CONTAMINANT);
            } else {
                series.push(CONTAMINANT);
                series.extend_from_slice(values);
            }
            ((*name).to_string(), series)
        })
        .collect();
    let slices: Vec<(&str, &[f64])> =
        owned.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    TimeSeriesData::from_f64_columns(slices, 1).unwrap()
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

fn pulse(treatment: u32, outcome: u32) -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(id(treatment), id(outcome), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "callers build the study in the argument and this function drops it after the run"
)]
fn run(study: Study) -> StudyResult {
    study.run(&ExecutionContext::for_tests(1)).unwrap()
}

fn frequentist(builder: antecedent::StudyBuilder) -> Study {
    builder
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn no_crossfit_nuisance(result: &StudyResult) {
    let effect = result.estimate().as_effect().unwrap();
    assert!(effect.crossfit.folds.is_none());
    assert!(effect.crossfit.learner_provenance.is_empty());
    assert!(effect.diagnostics.first_stage.is_none());
    assert!(effect.diagnostics.penalized.is_none());
    assert!(result.posterior.is_none());
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

/// Twelve interleaved rows. Control outcomes `[1.2, 0.7, 1.5, 0.9, 1.1, 0.6]`,
/// treated outcomes `[3.3, 2.6, 3.1, 2.8, 3.4, 2.7]`. The contrast and its
/// homoskedastic SE are the two-sample formulas on those rows. Lag-one
/// alignment drops the contaminant times.
#[test]
fn horizon_one_pulse_reproduces_the_unadjusted_linear_contrast() {
    let control = [1.2, 0.7, 1.5, 0.9, 1.1, 0.6];
    let treated = [3.3, 2.6, 3.1, 2.8, 3.4, 2.7];
    let mut treatment = Vec::with_capacity(12);
    let mut outcome = Vec::with_capacity(12);
    for index in 0..6 {
        treatment.extend([0.0, 1.0]);
        outcome.extend([control[index], treated[index]]);
    }
    let contrast = mean_contrast(&treatment, &outcome);
    let static_se = homoskedastic_contrast_se(&treatment, &outcome);
    assert!(static_se > 0.0);

    let table =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", &outcome)]).unwrap();
    let static_result = run(frequentist(
        Study::tabular(table)
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1)))),
    ));
    let series = lag_one_series(&[("t", &treatment), ("y", &outcome)], &["t"]);
    let mut temporal_graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut temporal_graph, id(0), Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut temporal_graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    temporal_graph.insert_directed(treatment_lag, outcome_now).unwrap();
    let temporal_result = run(frequentist(
        Study::series(series).graph(temporal_graph).query(CausalQuery::TemporalEffect(pulse(0, 1))),
    ));

    for result in [&static_result, &temporal_result] {
        let effect = result.estimate().as_effect().unwrap();
        assert!(close(effect.ate, contrast), "{}", effect.ate);
        assert_eq!(effect.provenance.n_obs, Some(u64::try_from(treatment.len()).unwrap()));
        assert!(result.estimand.adjustment_set.is_empty());
        assert!(result.estimand.instruments.is_empty());
        assert!(result.estimand.mediators.is_empty());
        assert_eq!(result.treatment, Some(id(0)));
        assert_eq!(result.outcome, id(1));
        assert_eq!(result.logical_plan.query_variables.as_ref(), &[id(0), id(1)]);
        no_crossfit_nuisance(result);
    }
    assert_eq!(static_result.estimand.method_kind().unwrap(), EstimandMethod::BackdoorAdjustment);
    assert_eq!(
        temporal_result.estimand.method_kind().unwrap(),
        EstimandMethod::TemporalBackdoorUnfolded
    );
    assert_eq!(static_result.logical_plan.estimator.as_deref(), Some("linear.adjustment.ate"));
    assert_eq!(
        temporal_result.logical_plan.estimator.as_deref(),
        Some("temporal.linear.adjustment")
    );
    assert_eq!(static_result.logical_plan.data_classification, DataClassification::Tabular);
    assert_eq!(temporal_result.logical_plan.data_classification, DataClassification::Temporal);

    let static_effect = static_result.estimate().as_effect().unwrap();
    assert!(close(static_effect.se_analytic(), static_se), "{}", static_effect.se_analytic());
    let static_interval = static_result.primary_interval_binding(false);
    assert_eq!(static_interval.method, IntervalMethod::AnalyticSe);
    assert_eq!(static_interval.dependence, "iid");
    assert!(close(static_interval.level, 0.95));

    // One series does not publish the static iid interval. The point is the
    // same contrast; the sampling object is the withheld analytic SE.
    let temporal_effect = temporal_result.estimate().as_effect().unwrap();
    assert!(temporal_effect.se_analytic().is_nan());
    assert_eq!(temporal_result.primary_interval_binding(false).method, IntervalMethod::None);
    assert!(has_diagnostic(&temporal_result, "estimate.temporal.circular_block_se"));
    assert!(!has_diagnostic(&static_result, "estimate.temporal.circular_block_se"));
}

/// `Y = 1 + 2 T + 5 Z`, with `T` and `Z` associated. The partial coefficient
/// of `T` is 2. The unadjusted mean contrast is not.
#[test]
fn horizon_one_pulse_reproduces_the_confounded_linear_contrast() {
    let treatment = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let confounder = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let outcome: Vec<f64> =
        treatment.iter().zip(&confounder).map(|(arm, z)| 1.0 + 2.0 * arm + 5.0 * z).collect();
    let partial = 2.0;
    let crude = mean_contrast(&treatment, &outcome);
    assert!((crude - partial).abs() > 1.0, "crude contrast {crude} must need adjustment");

    let table = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap();
    let static_result = run(frequentist(
        Study::tabular(table)
            .graph(dag(3, &[(2, 0), (2, 1), (0, 1)]))
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1)))),
    ));
    let series =
        lag_one_series(&[("t", &treatment), ("y", &outcome), ("z", &confounder)], &["t", "z"]);
    let mut temporal_graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut temporal_graph, id(0), Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut temporal_graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    let confounder_lag = ensure_lagged(&mut temporal_graph, id(2), Lag::from_raw(1)).unwrap();
    temporal_graph.insert_directed(confounder_lag, treatment_lag).unwrap();
    temporal_graph.insert_directed(confounder_lag, outcome_now).unwrap();
    temporal_graph.insert_directed(treatment_lag, outcome_now).unwrap();
    let temporal_result = run(frequentist(
        Study::series(series).graph(temporal_graph).query(CausalQuery::TemporalEffect(pulse(0, 1))),
    ));

    for result in [&static_result, &temporal_result] {
        let effect = result.estimate().as_effect().unwrap();
        assert!(close(effect.ate, partial), "{} versus crude {crude}", effect.ate);
        assert!((effect.ate - crude).abs() > 1.0);
        assert_eq!(effect.provenance.n_obs, Some(u64::try_from(treatment.len()).unwrap()));
        assert_eq!(result.estimand.adjustment_set.len(), 1);
        assert!(result.estimand.instruments.is_empty());
        assert!(result.estimand.mediators.is_empty());
        assert_eq!(result.treatment, Some(id(0)));
        assert_eq!(result.outcome, id(1));
        no_crossfit_nuisance(result);
        assert!(result.posterior.is_none());
    }
    assert_eq!(static_result.logical_plan.estimator.as_deref(), Some("linear.adjustment.ate"));
    assert_eq!(
        temporal_result.logical_plan.estimator.as_deref(),
        Some("temporal.linear.adjustment")
    );
    // Exact fit: residual rounding leaves a sub-ulp homoskedastic SE, and
    // the static route still publishes that analytic interval. The
    // horizon-one fit withholds it.
    let static_se = static_result.estimate().as_effect().unwrap().se_analytic();
    assert!(static_se.abs() < 1e-12, "{static_se}");
    assert_eq!(static_result.primary_interval_binding(false).method, IntervalMethod::AnalyticSe);
    assert_eq!(static_result.primary_interval_binding(false).dependence, "iid");
    assert!(temporal_result.estimate().as_effect().unwrap().se_analytic().is_nan());
    assert_eq!(temporal_result.primary_interval_binding(false).method, IntervalMethod::None);
    assert!(has_diagnostic(&temporal_result, "estimate.temporal.circular_block_se"));
    assert!(!has_diagnostic(&static_result, "estimate.temporal.circular_block_se"));
}

/// `M = T / 2 + Z`, `Y = T / 4 + M / 2`, with `Z` varying inside treatment.
/// The path product is 1/4, the direct effect is 1/4, and the total effect
/// is 1/2. `Z` keeps the mediator column off the treatment column. Horizon
/// one reads treatment one step before the mediator and the outcome.
#[test]
fn horizon_one_mediation_reproduces_the_static_path_product() {
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let shift = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let mediator: Vec<f64> = treatment.iter().zip(&shift).map(|(arm, z)| arm / 2.0 + z).collect();
    let outcome: Vec<f64> =
        treatment.iter().zip(&mediator).map(|(arm, mediator)| arm / 4.0 + mediator / 2.0).collect();
    let (direct, mediated, total) = (0.25, 0.25, 0.5);
    assert!(close(direct + mediated, total));

    let query = MediationQuery::binary(id(0), id(2), [id(1)], MediationContrast::Mediated);
    let table = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("m", mediator.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap();
    let static_result = run(frequentist(
        Study::tabular(table)
            .graph(dag(3, &[(0, 1), (1, 2), (0, 2)]))
            .query(CausalQuery::Mediation(query.clone())),
    ));
    let series = lag_one_series(&[("t", &treatment), ("m", &mediator), ("y", &outcome)], &["t"]);
    let mut temporal_graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut temporal_graph, id(0), Lag::from_raw(1)).unwrap();
    let mediator_now = ensure_lagged(&mut temporal_graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    let outcome_now = ensure_lagged(&mut temporal_graph, id(2), Lag::CONTEMPORANEOUS).unwrap();
    temporal_graph.insert_directed(treatment_lag, mediator_now).unwrap();
    temporal_graph.insert_directed(treatment_lag, outcome_now).unwrap();
    temporal_graph.insert_directed(mediator_now, outcome_now).unwrap();
    let temporal_result = run(frequentist(
        Study::series(series).graph(temporal_graph).query(CausalQuery::Mediation(query)),
    ));

    for result in [&static_result, &temporal_result] {
        let decomposition = result.mediation().unwrap();
        assert!(close(decomposition.direct.unwrap(), direct));
        assert!(close(decomposition.mediated.unwrap(), mediated));
        assert!(close(decomposition.total.unwrap(), total));
        assert!(close(result.effect(), mediated));
        assert_eq!(result.treatment, Some(id(0)));
        assert_eq!(result.outcome, id(2));
        assert!(result.estimand.instruments.is_empty());
        assert!(result.estimate().as_effect().unwrap().se_analytic().is_nan());
        assert_eq!(result.primary_interval_binding(false).method, IntervalMethod::None);
        no_crossfit_nuisance(result);
    }
    assert_eq!(static_result.estimand.method_kind().unwrap(), EstimandMethod::PathSpecificNatural);
    assert_eq!(
        temporal_result.estimand.method_kind().unwrap(),
        EstimandMethod::TemporalMediationMediated
    );
    assert_eq!(static_result.logical_plan.estimator.as_deref(), Some("mediation.linear"));
    assert_eq!(temporal_result.logical_plan.estimator.as_deref(), Some("temporal.mediation"));
    assert!(temporal_result.estimand.adjustment_set.is_empty());
}

/// `Y = 1.5 T + 0.2 W + 0.8 T W` on eight rows. The marginal effect at the
/// mean of `W` is 9/5. The additive regression `Y ~ 1 + T + W` has treatment
/// coefficient 25/14, from the normal equations on these rows.
#[test]
fn horizon_one_conditional_pulse_reproduces_the_marginal_interaction() {
    let treatment = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let modifier = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let outcome: Vec<f64> = treatment
        .iter()
        .zip(&modifier)
        .map(|(arm, modifier)| 1.5 * arm + 0.2 * modifier + 0.8 * arm * modifier)
        .collect();
    let mean_modifier = modifier.iter().sum::<f64>() / modifier.len() as f64;
    let marginal = 1.5 + 0.8 * mean_modifier;
    let additive = 25.0 / 14.0;
    assert!(close(mean_modifier, 3.0 / 8.0));
    assert!(close(marginal, 9.0 / 5.0));
    assert!((marginal - additive).abs() > 0.01);

    let table = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("w", modifier.as_slice()),
    ])
    .unwrap();
    let static_query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(id(0), id(1)).with_effect_modifiers([id(2)]),
    )
    .unwrap();
    let static_result = run(frequentist(
        Study::tabular(table)
            .graph(dag(3, &[(0, 1), (2, 1)]))
            .query(CausalQuery::ConditionalEffect(static_query)),
    ));
    let series = lag_one_series(&[("t", &treatment), ("y", &outcome), ("w", &modifier)], &["t"]);
    let mut temporal_graph = TemporalDag::empty();
    let treatment_lag = ensure_lagged(&mut temporal_graph, id(0), Lag::from_raw(1)).unwrap();
    let outcome_now = ensure_lagged(&mut temporal_graph, id(1), Lag::CONTEMPORANEOUS).unwrap();
    let modifier_now = ensure_lagged(&mut temporal_graph, id(2), Lag::CONTEMPORANEOUS).unwrap();
    temporal_graph.insert_directed(treatment_lag, outcome_now).unwrap();
    temporal_graph.insert_directed(modifier_now, outcome_now).unwrap();
    let temporal_result = run(frequentist(
        Study::series(series)
            .graph(temporal_graph)
            .query(CausalQuery::TemporalEffect(pulse(0, 1).with_effect_modifiers([id(2)]))),
    ));

    for result in [&static_result, &temporal_result] {
        let effect = result.estimate().as_effect().unwrap();
        assert!(close(effect.ate, marginal), "{}", effect.ate);
        assert!((effect.ate - additive).abs() > 0.01);
        assert_eq!(result.treatment, Some(id(0)));
        assert_eq!(result.outcome, id(1));
        assert!(result.estimand.instruments.is_empty());
        assert!(result.posterior.is_none());
        no_crossfit_nuisance(result);
    }
    assert_eq!(
        static_result.logical_plan.estimator.as_deref(),
        Some("conditional.linear.adjustment")
    );
    assert_eq!(
        temporal_result.logical_plan.estimator.as_deref(),
        Some("temporal.linear.adjustment")
    );
    assert!(has_diagnostic(&temporal_result, "estimate.temporal.conditional_marginal"));
    let static_interval = static_result.primary_interval_binding(false);
    assert_eq!(static_interval.method, IntervalMethod::AnalyticSe);
    assert_eq!(static_interval.dependence, "iid");
    assert!(static_result.estimate().as_effect().unwrap().se_analytic().is_finite());
    assert!(temporal_result.estimate().as_effect().unwrap().se_analytic().is_nan());
    assert_eq!(temporal_result.primary_interval_binding(false).method, IntervalMethod::None);
}
