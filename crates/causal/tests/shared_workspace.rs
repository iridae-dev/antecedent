//! Shared estimate→refute `EstimationWorkspace` (backlog C).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use causal::{
    CausalAnalysis,
    RefuteSuite,
};
use causal_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy};
use causal_graph::{Dag, DenseNodeId};
use causal_validate::{PlaceboTreatment, RefutationProblem};

fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        let yi = 2.0 * ti + zi + e;
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

#[test]
fn execute_static_refute_reuses_estimate_workspace() {
    let (data, dag, query) = confounded_scm(300, 3);
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(20)
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
    assert!(!result.refutations.is_empty());
    // Placebo/RCC must be informative under linear adjustment.
    assert!(result.refutations.iter().any(|r| r.informative));
}

#[test]
fn shared_workspace_placebo_parity_and_capacity() {
    let (data, dag, query) = confounded_scm(250, 5);
    let id_run = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag)
        .query(query.clone())
        .bootstrap_replicates(10)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();

    let estimand = id_run.estimand.clone();
    let estimate = id_run.estimate.clone();
    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &estimate,
        estimator: Some("linear.adjustment"),
        temporal: None,
    };

    let mut warmed = EstimationWorkspace::default();
    let mut est = LinearAdjustmentAte::new();
    est.bootstrap_replicates = 0;
    est.overlap = OverlapPolicy::ExplicitOverride;
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let _ = est
        .fit(&prep, &mut warmed, &ExecutionContext::for_tests(2), estimate.assumptions.clone())
        .unwrap();
    let warmed_cap = warmed.ols.scratch.capacity() + warmed.ols.rhs.capacity();
    let warmed_grows = warmed.ols.grow_count;
    assert!(warmed_cap > 0, "point fit must grow OLS scratch");
    assert!(warmed_grows >= 1);

    let placebo = PlaceboTreatment { replicates: 8, ..PlaceboTreatment::new() };
    let ctx = ExecutionContext::for_tests(11);
    let report_warm = placebo.refute(&problem, &mut warmed, &ctx).unwrap();
    let after_cap = warmed.ols.scratch.capacity() + warmed.ols.rhs.capacity();
    assert!(
        after_cap >= warmed_cap,
        "refute must reuse (not shrink) warmed OLS capacity"
    );
    // Grow count must not reset; may stay equal if capacity already sufficient.
    assert!(warmed.ols.grow_count >= warmed_grows);

    let mut fresh = EstimationWorkspace::default();
    let report_fresh = placebo.refute(&problem, &mut fresh, &ctx).unwrap();
    assert_eq!(report_warm.passed, report_fresh.passed);
    assert!((report_warm.refuted_ate - report_fresh.refuted_ate).abs() < 1e-9);
}

#[test]
fn propensity_workspace_reused_estimate_into_overlap() {
    use causal::strategy_table::{EstimatorId, StaticEstimateWorkspaces, estimate_static_effect};
    use causal::{
    CausalAnalysis,
    RefuteSuite,
};
    use causal_estimate::OverlapPolicy;
    use causal_validate::OverlapRefuter;

    let (data, dag, query) = confounded_scm(280, 7);
    let id_run = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag)
        .query(query.clone())
        .estimator(EstimatorId::PropensityWeighting)
        .bootstrap_replicates(10)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let estimand = id_run.estimand.clone();
    let estimate = id_run.estimate.clone();

    let mut ws = StaticEstimateWorkspaces::default();
    let _ = estimate_static_effect(
        EstimatorId::PropensityWeighting,
        &data,
        &estimand,
        &query,
        estimate.assumptions.clone(),
        10,
        Some(OverlapPolicy::require_diagnostics()),
        None,
        &ExecutionContext::for_tests(3),
        &mut ws,
    )
    .unwrap();
    let score_grows = ws.propensity.propensity.scores_grow_count;
    let score_cap = ws.propensity.propensity.scores.capacity();
    let ols_grows = ws.propensity.propensity.ols.grow_count;
    assert!(score_grows >= 1 && score_cap > 0, "point propensity fit must warm buffers");
    assert!(estimate.ate.is_finite());

    let problem = RefutationProblem {
        data: &data,
        estimand: &estimand,
        query: &query,
        original: &estimate,
        estimator: Some("propensity.weighting"),
        temporal: None,
    };
    let _ = OverlapRefuter::new()
        .refute_with_propensity(&problem, &mut ws.propensity.propensity)
        .unwrap();
    assert!(ws.propensity.propensity.scores.capacity() >= score_cap);
    assert!(ws.propensity.propensity.scores_grow_count >= score_grows);
    assert!(ws.propensity.propensity.ols.grow_count >= ols_grows);

    let mut ws2 = StaticEstimateWorkspaces::default();
    let aipw = estimate_static_effect(
        EstimatorId::Aipw,
        &data,
        &estimand,
        &query,
        estimate.assumptions,
        0,
        None,
        None,
        &ExecutionContext::for_tests(4),
        &mut ws2,
    )
    .unwrap();
    assert!(aipw.ate.is_finite());
    assert!(ws2.aipw.propensity.scores_grow_count >= 1 || ws2.aipw.propensity.ols.grow_count >= 1);
}
