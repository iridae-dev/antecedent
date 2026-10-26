//! Shared estimate→refute `EstimationWorkspace` (backlog C).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent::{RefuteSuite, Study};
use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy};
use antecedent_validate::{PlaceboTreatment, RefutationProblem};

#[path = "fixtures_common/mod.rs"]
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
    assert!(result.estimate.as_effect().unwrap().ate.is_finite());
    assert!(!result.refutations.is_empty());
    // Placebo and RCC are OLS-powerless sampling-stability diagnostics, not
    // claim falsifiers, and report informative: false under linear adjustment
    // ("fix: make validate refuters, sensitivity and stability checks say
    // only what they test").
    assert!(result.refutations.iter().all(|r| !r.informative));
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
        estimate.as_effect().unwrap(),
        Some("linear.adjustment"),
        None,
    );

    let mut warmed = EstimationWorkspace::default();
    let mut est = LinearAdjustmentAte::new();
    est.inference.bootstrap_replicates = 0;
    est.overlap = OverlapPolicy::ExplicitOverride;
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let _ = est
        .fit(
            &prep,
            &mut warmed,
            &ExecutionContext::for_tests(2),
            estimate.as_effect().unwrap().provenance.assumptions.clone(),
        )
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
