#![allow(clippy::cast_precision_loss, clippy::many_single_char_names, clippy::similar_names)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! Natural direct, natural indirect and total effects, and unit-level counterfactuals, on one
//! confounded linear model.
//!
//! Question: how much of the effect of `a` on `y` passes through the mediator `m`, and what is
//! each unit's outcome change when `a` moves from 0.2 to 0.8?
//!
//! Data and truth: 500 deterministic rows with `m = 2 a + 0.5 z + ...` and
//! `y = 3 a + 4 m + 5 z + ...`; `z` confounds `a`, `m` and `y`. For control 0.2 against active
//! 0.8 the structural contrasts are: natural direct effect `3 * 0.6 = 1.8`, natural indirect
//! effect `2 * 4 * 0.6 = 4.8`, total effect and mean individual effect 6.6.
//!
//! Asserted: each mediation contrast is within 0.03 of its truth and is estimated by
//! `mediation.linear`; the counterfactual query returns 500 unit effects with mean within 0.03
//! of 6.6, estimated by `gcm.fit`.
//!
//! What this establishes: on a linear model with the confounder measured, the mediation and
//! counterfactual routes agree with the structural coefficients. The decomposition relies on
//! the linear mechanisms and on `z` being the only common cause; the unit effects are statements
//! under the fitted mechanisms. The bootstrap and refuters are off.
//!
//! Run: `cargo run -p antecedent --example staged_static_kinds`
//! Source: `examples/rust/staged_static_kinds.rs`

use std::sync::Arc;

use antecedent::prelude::*;
use antecedent::{CounterfactualQuery, MediationContrast, MediationQuery};

fn confounded_scm(n: usize) -> (TabularData, Dag) {
    let i = 0..n;
    let z: Vec<f64> = i.clone().map(|k| (k as f64 * 0.41).cos()).collect();
    let a: Vec<f64> = i.clone().map(|k| (k as f64 * 0.71).sin() + 0.4 * z[k]).collect();
    let m: Vec<f64> =
        i.clone().map(|k| 2.0 * a[k] + 0.5 * z[k] + (k as f64 * 1.13).cos()).collect();
    let y: Vec<f64> =
        i.map(|k| 3.0 * a[k] + 4.0 * m[k] + 5.0 * z[k] + 0.1 * (k as f64 * 0.31).sin()).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .expect("equal-length columns");
    let mut dag = Dag::with_variables(4);
    for (s, t) in [(0, 1), (0, 2), (1, 2), (3, 0), (3, 1), (3, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    (data, dag)
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let (data, dag) = confounded_scm(500);
    let control = 0.2;
    let active = 0.8;
    let ctx = ExecutionContext::for_tests(13);
    for (label, contrast, expected) in [
        ("natural_direct", MediationContrast::NaturalDirect, 1.8),
        ("natural_indirect", MediationContrast::NaturalIndirect, 4.8),
        ("total", MediationContrast::Total, 6.6),
    ] {
        let mut query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            Arc::from([VariableId::from_raw(1)]),
            contrast,
        );
        query.control = Intervention::set(query.treatment, Value::f64(control));
        query.active = Intervention::set(query.treatment, Value::f64(active));
        let result = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()?
            .run(&ctx)?;
        println!(
            "{label}={:.4} estimator={}",
            result.estimate().as_effect().unwrap().ate,
            result.logical_plan.estimator.as_deref().unwrap_or("?")
        );
        assert!(
            (result.estimate().as_effect().unwrap().ate - expected).abs() < 0.03,
            "{label}={}",
            result.estimate().as_effect().unwrap().ate
        );
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("mediation.linear"));
    }

    // The counterfactual query fits a mechanism per node and returns, for every observed unit,
    // the change in y when a is set to the active level instead of the control level.
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    )
    .with_control_level(control);
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ctx)?;
    let cf = result.counterfactual().expect("unit ITE");
    println!(
        "mean_ite={:.4} n={} estimator={}",
        cf.mean_ite,
        cf.unit_effects.len(),
        result.logical_plan.estimator.as_deref().unwrap_or("?")
    );
    assert!((cf.mean_ite - 6.6).abs() < 0.03, "mean_ite={}", cf.mean_ite);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
    assert_eq!(cf.unit_effects.len(), 500);
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
