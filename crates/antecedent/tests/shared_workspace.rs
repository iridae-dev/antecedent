//! Shared estimate→refute `EstimationWorkspace` (backlog C).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent::{RefuteSuite, Study};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy};
use antecedent_validate::{PlaceboTreatment, RefutationProblem};

#[path = "fixtures_common/mod.rs"]
mod common;

// The confounded static ATE study five suites run, in one owner.
use common::fixtures::confounded_scm;

/// Coefficient of `t` in the least-squares regression of `y` on an intercept, `t` and `z`
/// (columns 0, 1, 2 of the fixture), solved here from the normal equations.
fn adjusted_coefficient(data: &antecedent_data::TabularData) -> f64 {
    let column = |raw: u32| data.float64_values(VariableId::from_raw(raw)).unwrap();
    let (t, y, z) = (column(0), column(1), column(2));
    let mut a = [[0.0_f64; 4]; 3];
    for i in 0..t.len() {
        let row = [1.0, t[i], z[i]];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += row[r] * row[c];
            }
            a[r][3] += row[r] * y[i];
        }
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs())).unwrap();
        a.swap(col, pivot);
        for r in 0..3 {
            if r != col {
                let factor = a[r][col] / a[col][col];
                for c in 0..4 {
                    let delta = factor * a[col][c];
                    a[r][c] -= delta;
                }
            }
        }
    }
    a[1][3] / a[1][1]
}

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
    // The point estimate is the confounder-adjusted least-squares coefficient (truth 2), not the
    // unadjusted contrast, which the confounder pushes well above 2.
    let ate = result.estimate().as_effect().unwrap().ate;
    let (data, _, _) = confounded_scm(300, 3);
    assert!((ate - adjusted_coefficient(&data)).abs() < 1e-8, "{ate}");
    assert!((ate - 2.0).abs() < 0.2, "{ate}");
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
    let estimate = id_run.estimate().clone();
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
    // A placebo treatment carries no effect: its refuted ATE is far below the true effect of 2.
    assert!(report_warm.refuted_ate.abs() < 0.1, "{}", report_warm.refuted_ate);
}
