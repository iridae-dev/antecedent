//! Routes that read their rows in their own way refuse every missing-data policy.
//!
//! `parity/missing_policy_support.toml` has no row for the synthetic-control, local polynomial
//! ratio and counterfactual routes, so any declaration (here complete-case and multiple imputation) is
//! refused at build with `missing_policy_not_applicable` instead of being ignored. Each design
//! first runs without a declaration and recovers its closed-form effect, so the refusal is of a
//! route that works.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{LocalPolynomialRatioQuery, Study, SyntheticControlQuery};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext as Ctx, Intervention, Value, VariableId,
};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};

fn synthetic() -> (TabularData, SyntheticControlQuery) {
    let donors = [
        ("a", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("b", [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
        ("c", [1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
    ];
    let (mut y, mut units, mut periods) = (vec![], vec![], vec![]);
    for (unit, trajectory) in donors {
        for (index, value) in trajectory.into_iter().enumerate() {
            units.push(Arc::<str>::from(unit));
            periods.push(index as i64 + 1);
            y.push(value);
        }
    }
    for index in 0..6 {
        units.push(Arc::<str>::from("treated"));
        periods.push(index as i64 + 1);
        let counterfactual =
            0.2 * donors[0].1[index] + 0.3 * donors[1].1[index] + 0.5 * donors[2].1[index];
        y.push(counterfactual + if index >= 4 { 5.0 } else { 0.0 });
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        5,
    );
    (data, query)
}

fn fuzzy_jump() -> (TabularData, LocalPolynomialRatioQuery) {
    let (mut x, mut treatment, mut outcome) = (vec![], vec![], vec![]);
    for step in 1..80 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 80.0;
            for replicate in 0..4 {
                let dose = if sign < 0.0 {
                    f64::from(replicate == 0)
                } else {
                    f64::from(replicate != 3)
                };
                x.push(score);
                treatment.push(dose);
                outcome.push(
                    1.0 + 2.0 * score
                        + 0.5 * score * score
                        + 3.0 * dose
                        + [-0.02, -0.01, 0.01, 0.02][replicate],
                );
            }
        }
    }
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap();
    let query = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff: 0.0,
        bandwidth: 1.0,
        kink: false,
        options: None,
    };
    (data, query)
}

fn declarations() -> [MissingDeclaration; 2] {
    [
        MissingDeclaration::new(MissingPolicy::CompleteCase),
        MissingDeclaration::new(MissingPolicy::MultipleImputation(ImputationSpec::new())),
    ]
}

#[test]
fn synthetic_control_runs_undeclared_and_refuses_every_missing_data_policy() {
    let (data, query) = synthetic();
    let ctx = ExecutionContext::for_tests(61);
    let fit = Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&ctx).unwrap();
    assert!((fit.synthetic_control().unwrap().effect - 5.0).abs() < 1e-5);
    for declaration in declarations() {
        let error = Study::tabular(data.clone())
            .query(query.clone())
            .missing(declaration)
            .build()
            .unwrap_err();
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
        assert!(error.to_string().contains("reads its rows in its own way"), "{error}");
    }
}

#[test]
fn the_local_polynomial_ratio_runs_undeclared_and_refuses_every_missing_data_policy() {
    let (data, query) = fuzzy_jump();
    let ctx = ExecutionContext::for_tests(62);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&ctx).unwrap();
    let effect = result.estimate().as_effect().unwrap().ate;
    assert!((effect - 3.0).abs() < 1e-8, "{effect}");
    for declaration in declarations() {
        let error = Study::tabular(data.clone())
            .query(query.clone())
            .missing(declaration)
            .build()
            .unwrap_err();
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
        assert!(error.to_string().contains("reads its rows in its own way"), "{error}");
    }
}

#[test]
fn the_counterfactual_runs_undeclared_and_refuses_every_missing_data_policy() {
    // y = 3 a + 4 m + z, m = 2 a + noise, so the mean unit effect of setting a to 1 is 11.
    let n = 300_usize;
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..n).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..n).map(|i| z[i] + (at(i) * 0.71).cos()).collect();
    let m: Vec<f64> = (0..n).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = antecedent_graph::Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)] {
        dag.insert_directed(
            antecedent_graph::DenseNodeId::from_raw(from),
            antecedent_graph::DenseNodeId::from_raw(to),
        )
        .unwrap();
    }
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(2),
            Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    let result = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .build()
        .unwrap()
        .run(&Ctx::for_tests(63))
        .unwrap();
    assert!((result.counterfactual().unwrap().mean_ite - 11.0).abs() < 0.1);
    for declaration in declarations() {
        let error = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .missing(declaration)
            .build()
            .unwrap_err();
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
        assert!(error.to_string().contains("reads its rows in its own way"), "{error}");
    }
}
