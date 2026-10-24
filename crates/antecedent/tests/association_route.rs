//! The associational route: a noncausal description of the observed law, compiled and executed
//! only from its retained lowered program.
//!
//! Known-truth checks compare the reported quantities with their closed forms, the selection
//! rule is exercised for determinism, and the claim is checked to be associational with
//! identification not claimed. The temporal counterpart resamples circular blocks under the
//! declared dependent-series design, and the Bayesian counterparts use the conjugate marginals
//! for a Gaussian outcome and the Bayesian bootstrap otherwise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "fixture indices convert exactly and exact constants are compared"
)]

use antecedent::analysis::RetainedOperation;
use antecedent::planner::RouteFamily;
use antecedent::planner::lowered::FunctionalKind;
use antecedent::planner::plan_view::IdentificationKind;
use antecedent::planner::statistical::{PosteriorAlgorithm, ResampleUnit, ResamplingMethod};
use antecedent::{
    AssociationFunctional, AssociationMeasure, AssociationQuery, BayesianConfig, CausalQuery,
    InferenceMode, Study,
};
use antecedent_contract::{ClaimKind, DomainStatus, IDENTIFICATION_NOT_CLAIMED, SlotAvailability};
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

/// Deterministic pseudo-noise in `[-0.5, 0.5)`.
fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

/// Columns x, y with `y = 1 + 2 x + noise` over 120 rows.
fn line() -> (TabularData, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..120).map(|i| (i as f64) / 12.0).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + noise(i)).collect();
    (TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap(), x, y)
}

/// Columns x, y over an AR(1) predictor and an outcome that follows it.
fn series(n: usize) -> TimeSeriesData {
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    for t in 1..n {
        x[t] = 0.8 * x[t - 1] + noise(t);
        y[t] = 0.5 * x[t] + 0.6 * y[t - 1] + noise(t + 1000);
    }
    TimeSeriesData::from_f64_columns([("x", &x[..]), ("y", &y[..])], 1_000_000_000).unwrap()
}

fn static_study(data: TabularData, query: AssociationQuery) -> Study {
    Study::tabular(data).query(CausalQuery::Association(query)).build().unwrap()
}

#[test]
fn a_static_association_executes_only_from_its_retained_program_and_is_noncausal() {
    let (data, x, y) = line();
    let query = AssociationQuery::new(v(1), [v(0)]);
    let study = static_study(data.clone(), query.clone());
    let plan = study.compile(&ctx(1)).unwrap();
    let program = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(program.route(), "AssociationStatic");
    assert!(matches!(
        program.scientific().target(),
        antecedent::planner::lowered::ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Association,
            outcome: Some(outcome),
            ..
        } if *outcome == v(1)
    ));
    assert_eq!(program.claim().kind(), Some(ClaimKind::Associational));
    let view = plan.plan_view().unwrap();
    assert_eq!(view.family(), RouteFamily::Associational);
    assert_eq!(view.identification().kind(), IdentificationKind::NotClaimed);
    assert_eq!(view.identification().refusal(), None);
    assert_eq!(view.identity(), program.identity());
    assert_eq!(program.statistical().models().len(), 1);
    assert_eq!(program.statistical().models()[0].family(), "linear_regression");

    // The description is the least-squares line, with the robust standard error.
    let prepared = study.prepare(&ctx(1)).unwrap();
    assert!(matches!(prepared.retained_operation(), RetainedOperation::Association));
    assert_eq!(prepared.retained_operation().name(), "association");
    assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
    let result = prepared.estimate(&data, &ctx(1)).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(program.identity()));
    let payload = result.association().expect("an association payload");
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let slope = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / sxx;
    assert_eq!(payload.estimate.terms.len(), 2);
    assert!((payload.estimate.terms[1].estimate - slope).abs() < 1e-8);
    assert!(payload.estimate.terms[1].standard_error.unwrap() > 0.0);
    assert!(payload.refused_after.is_none() && payload.scores.is_none());
    assert!(result.estimate.as_effect().is_none(), "an association is no effect estimate");

    // The claim is associational, never a point, with identification not claimed.
    let contract = prepared.contract().unwrap();
    let claim = result.claim(&contract, &ctx(1)).unwrap();
    assert_eq!(claim.kind, ClaimKind::Associational);
    assert_eq!(claim.value, None);
    assert_eq!(claim.domains.identification, DomainStatus::NotClaimed);
    assert!(matches!(
        &claim.reasoning.identification,
        SlotAvailability::Unavailable { reason, .. } if &**reason == IDENTIFICATION_NOT_CLAIMED
    ));

    // A lowered dispatch executes its retained program: it compiles no other study.
    let scoped = ctx(1).enter_lowered_dispatch(program.identity());
    let refusal = study.compile(&scoped).unwrap_err();
    assert!(refusal.to_string().contains("public_builder_reentry"), "{refusal}");
    // A plan lowered for the series cell is not the program of a table study.
    let temporal = Study::series(series(60))
        .query(CausalQuery::Association(query))
        .build()
        .unwrap()
        .compile(&ctx(1))
        .unwrap();
    let swapped = study.execute(&temporal, &ctx(1)).unwrap_err();
    assert!(swapped.to_string().contains("route_not_lowered"), "{swapped}");
    // A one-shot run compiles and executes the same program.
    let fresh = study.run(&ctx(1)).unwrap();
    assert_eq!(fresh.lowered_program_identity(), Some(program.identity()));
    assert_eq!(fresh.association().unwrap().estimate.terms, payload.estimate.terms);
}

#[test]
fn conditional_laws_and_the_selection_rule_follow_the_outcome_scale() {
    // 2x2 table: P(y = 1 | x = 0) = 0.2 and P(y = 1 | x = 1) = 0.6.
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (level, ones) in [(0.0, 20), (1.0, 60)] {
        for i in 0..100 {
            x.push(level);
            y.push(f64::from(i < ones));
        }
    }
    let data = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let strata =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional);
    let study = static_study(data.clone(), strata.clone());
    let result = study.run(&ctx(2)).unwrap();
    let association = result.association().unwrap();
    assert_eq!(association.estimate.selection.analysis.as_str(), "stratified_probability");
    assert!((association.estimate.terms[0].estimate - 0.2).abs() < 1e-12);
    assert!((association.estimate.terms[1].estimate - 0.6).abs() < 1e-12);
    // The selection rule is a deterministic function of the query and the data.
    let again = static_study(data.clone(), strata).run(&ctx(99)).unwrap();
    assert_eq!(again.association().unwrap().estimate, association.estimate);
    // A 0/1 outcome and a numeric predictor pick logistic regression; the log-odds ratio is the
    // closed form of the table.
    let auto = static_study(data, AssociationQuery::new(v(1), [v(0)])).run(&ctx(2)).unwrap();
    let logistic = auto.association().unwrap();
    assert_eq!(logistic.estimate.selection.analysis.as_str(), "logistic_regression");
    let log_odds = |p: f64| (p / (1.0 - p)).ln();
    assert!(
        (logistic.estimate.terms[1].estimate - (log_odds(0.6) - log_odds(0.2))).abs() < 1e-6,
        "{}",
        logistic.estimate.terms[1].estimate
    );
}

#[test]
fn an_association_after_a_refused_causal_query_retains_the_refusal_and_ranks_nothing() {
    let (data, _, _) = line();
    let query = AssociationQuery::new(v(1), [v(0)]).following_refusal("effect_not_identified");
    let study = static_study(data.clone(), query);
    let plan = study.compile(&ctx(3)).unwrap();
    let view = plan.plan_view().unwrap();
    assert_eq!(view.identification().kind(), IdentificationKind::NotClaimed);
    assert_eq!(view.identification().refusal(), Some("effect_not_identified"));
    let plain =
        static_study(data.clone(), AssociationQuery::new(v(1), [v(0)])).compile(&ctx(3)).unwrap();
    assert_ne!(plain.lowered_program_identity(), plan.lowered_program_identity());
    let prepared = study.prepare(&ctx(3)).unwrap();
    let result = prepared.estimate(&data, &ctx(3)).unwrap();
    assert_eq!(
        result.association().unwrap().refused_after.as_deref(),
        Some("effect_not_identified")
    );
}

#[test]
fn a_temporal_association_resamples_blocks_under_the_declared_design() {
    let data = series(150);
    let query = AssociationQuery::new(v(1), [v(0)]);
    let study = Study::series(data.clone())
        .query(CausalQuery::Association(query.clone()))
        .bootstrap_replicates(60)
        .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
        .build()
        .unwrap();
    let plan = study.compile(&ctx(4)).unwrap();
    let program = plan.lowered().unwrap();
    assert_eq!(program.route(), "AssociationTemporal");
    let resampling = program.statistical().resampling().expect("the study asked for replicates");
    assert_eq!(resampling.method(), ResamplingMethod::BlockBootstrap);
    assert_eq!(resampling.unit(), ResampleUnit::CalendarBlock);
    assert_eq!(resampling.replicates(), 60);
    let prepared = study.prepare(&ctx(4)).unwrap();
    let result = prepared.estimate_series(&data, &ctx(4)).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(program.identity()));
    let payload = result.association().unwrap();
    let record = payload.estimate.resampling.unwrap();
    assert_eq!((record.method, record.unit), ("block_bootstrap", "calendar_block"));
    assert!(record.block_length.unwrap() > 1, "dependent blocks, not rows");
    assert_eq!(payload.estimate.terms[1].standard_error_method, Some("bootstrap_circular_block"));
    // The same seed reproduces the replicates.
    let again = prepared.estimate_series(&data, &ctx(4)).unwrap();
    assert_eq!(again.association().unwrap().estimate, payload.estimate);

    // Independent subjects cannot describe one series; the refusal names the next attempt.
    let refused = Study::series(data)
        .query(CausalQuery::Association(query))
        .bootstrap_replicates(60)
        .sampling_design(DeclaredSamplingDesign::IndependentSubjectTrajectories)
        .build()
        .and_then(|study| study.compile(&ctx(4)).map(|_| ()))
        .unwrap_err();
    assert!(refused.to_string().contains("Next attempt"), "{refused}");
}

#[test]
fn a_bayesian_association_is_conjugate_for_a_gaussian_outcome_and_a_bootstrap_otherwise() {
    let (data, _, _) = line();
    let bayesian = |data: TabularData| {
        Study::tabular(data)
            .query(CausalQuery::Association(AssociationQuery::new(v(1), [v(0)])))
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400)))
            .build()
            .unwrap()
    };
    let study = bayesian(data.clone());
    let plan = study.compile(&ctx(5)).unwrap();
    let posterior = plan.lowered().unwrap().statistical().posterior().unwrap();
    assert_eq!(posterior.algorithm(), PosteriorAlgorithm::Conjugate);
    let result = study.run(&ctx(5)).unwrap();
    let payload = result.association().unwrap();
    assert_eq!(payload.estimate.posterior.unwrap().method, "conjugate_normal_inverse_gamma");
    let slope = payload.estimate.terms[1].posterior.unwrap();
    assert!((slope.mean - payload.estimate.terms[1].estimate).abs() < 0.02, "{slope:?}");
    assert!(slope.lower < slope.mean && slope.mean < slope.upper);

    // A 0/1 outcome has no conjugate form: the Bayesian bootstrap draws the posterior.
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (level, ones) in [(0.0, 25), (1.0, 55)] {
        for i in 0..100 {
            x.push(level);
            y.push(f64::from(i < ones));
        }
    }
    let binary = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let study = bayesian(binary);
    let plan = study.compile(&ctx(5)).unwrap();
    let posterior = plan.lowered().unwrap().statistical().posterior().unwrap();
    assert_eq!(posterior.algorithm(), PosteriorAlgorithm::BayesianBootstrap);
    let result = study.run(&ctx(5)).unwrap();
    let draws = result.association().unwrap();
    assert_eq!(draws.estimate.posterior.unwrap().method, "bayesian_bootstrap");
    let log_odds_ratio = draws.estimate.terms[1].posterior.unwrap();
    assert!(log_odds_ratio.lower < 1.3 && 1.3 < log_odds_ratio.upper, "{log_odds_ratio:?}");
}

#[test]
fn a_bayesian_series_association_draws_block_weights() {
    let data = series(120);
    let study = Study::series(data.clone())
        .query(CausalQuery::Association(AssociationQuery::new(v(1), [v(0)])))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .build()
        .unwrap();
    let plan = study.compile(&ctx(6)).unwrap();
    let posterior = plan.lowered().unwrap().statistical().posterior().unwrap();
    assert_eq!(posterior.algorithm(), PosteriorAlgorithm::BayesianBootstrap);
    assert!(plan.lowered().unwrap().statistical().resampling().is_none());
    let result = study.prepare(&ctx(6)).unwrap().estimate_series(&data, &ctx(6)).unwrap();
    let payload = result.association().unwrap();
    assert_eq!(payload.estimate.posterior.unwrap().method, "bayesian_bootstrap");
    assert_eq!(payload.estimate.selection.modality.as_str(), "series");
    assert!(payload.estimate.terms.iter().all(|term| term.posterior.is_some()));
}

#[test]
fn a_predictive_model_is_scored_on_held_out_folds_beside_its_baseline() {
    let (data, _, _) = line();
    let query = AssociationQuery::new(v(1), [v(0)])
        .with_functional(AssociationFunctional::Predictive { folds: 5 });
    let result = static_study(data, query).run(&ctx(7)).unwrap();
    let payload = result.association().unwrap();
    let scores = payload.scores.as_ref().expect("a predictive model is scored");
    assert_eq!(scores.cases, 120);
    assert_eq!(scores.score, "log_score_gaussian");
    assert!(scores.mean_score < scores.baseline_score, "the slope beats the training mean");
    assert!(scores.mean_crps.is_some());
}

#[test]
fn an_association_measure_needs_one_predictor_and_a_regression_declares_its_conditioning() {
    let (data, _, _) = line();
    let query = AssociationQuery::new(v(1), [v(0)])
        .with_functional(AssociationFunctional::Measure(AssociationMeasure::Pearson));
    let result = static_study(data.clone(), query).run(&ctx(8)).unwrap();
    let r = result.association().unwrap().estimate.terms[0].estimate;
    assert!(r > 0.99 && r <= 1.0, "{r}");
    let invalid = AssociationQuery::new(v(1), [v(0), v(0)]);
    assert!(Study::tabular(data).query(CausalQuery::Association(invalid)).build().is_err());
}
