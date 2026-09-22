//! Shared estimate→refute `EstimationWorkspace` (backlog C).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent::EstimatorSpec;
use antecedent::{RefuteSuite, Study};
use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy};
use antecedent_validate::{PlaceboTreatment, RefutationProblem};

mod common;

// The confounded static ATE study five suites run, in one owner.
use common::fixtures::confounded_scm;

#[test]
fn execute_static_refute_reuses_estimate_workspace() {
    let (data, dag, query) = confounded_scm(300, 3);
    let result = Study::tabular(data)
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
    let id_run = Study::tabular(data.clone())
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
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &estimate,
        Some("linear.adjustment"),
        None,
    );

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
    assert!(after_cap >= warmed_cap, "refute must reuse (not shrink) warmed OLS capacity");
    // Grow count must not reset; may stay equal if capacity already sufficient.
    assert!(warmed.ols.grow_count >= warmed_grows);

    let mut fresh = EstimationWorkspace::default();
    let report_fresh = placebo.refute(&problem, &mut fresh, &ctx).unwrap();
    assert_eq!(report_warm.passed, report_fresh.passed);
    assert!((report_warm.refuted_ate - report_fresh.refuted_ate).abs() < 1e-9);
}

#[test]
fn propensity_workspace_reused_estimate_into_overlap() {
    use antecedent::strategy_table::{
        EstimatorId, StaticEstimateWorkspaces, estimate_static_effect,
    };
    use antecedent::{RefuteSuite, Study};
    use antecedent_estimate::OverlapPolicy;
    use antecedent_validate::OverlapRefuter;

    let (data, dag, query) = confounded_scm(280, 7);
    let id_run = Study::tabular(data.clone())
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
        &EstimatorSpec::Default(EstimatorId::PropensityWeighting),
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

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &estimate,
        Some("propensity.weighting"),
        None,
    );
    let _ = OverlapRefuter::new()
        .refute_with_propensity(&problem, &mut ws.propensity.propensity)
        .unwrap();
    assert!(ws.propensity.propensity.scores.capacity() >= score_cap);
    assert!(ws.propensity.propensity.scores_grow_count >= score_grows);
    assert!(ws.propensity.propensity.ols.grow_count >= ols_grows);

    // ATT uses the legacy residualized fit and its reusable propensity workspace.
    // AllObserved iid AIPW now cross-fits per-fold nuisances instead.
    let query = query.with_target_population(antecedent_core::TargetPopulation::Treated);
    let mut ws2 = StaticEstimateWorkspaces::default();
    let aipw = estimate_static_effect(
        &EstimatorSpec::Default(EstimatorId::Aipw),
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
