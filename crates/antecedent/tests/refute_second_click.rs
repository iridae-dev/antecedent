//! Second-click refute after prepared estimate (BACKLOG E).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, LatencyMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, CausalSchemaBuilder, ContinuousDomain,
    ExecutionContext, GridSpec, MeasurementSpec, ResponseFunctional, ResponseQuery, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};

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
        z.push(zi);
        t.push(ti);
        y.push(2.0 * ti + zi + e);
    }
    let mut b = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
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
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn prepared_refute_second_click_preserves_ate() {
    let (data, dag, query) = confounded_scm(400, 11);
    let ctx = ExecutionContext::for_tests(5);
    let prepared = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();

    let first = prepared.estimate(&data, &ctx).unwrap();
    assert!(first.refutations.is_empty());
    let ate = first.estimate.ate;

    let second = prepared.refute(&first, &data, RefuteSuite::PlaceboAndRcc, &ctx).unwrap();
    assert!((second.estimate.ate - ate).abs() < 1e-15);
    assert!(!second.refutations.is_empty());
    assert!(second.diagnostics.iter().any(|d| d.code.as_ref() == "exec.refute.second_click"));

    let one_shot = Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::PlaceboAndRcc)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(second.refutations.len(), one_shot.refutations.len());
}

#[test]
fn prepared_bayesian_refute_includes_predictive_checks() {
    let (data, dag, query) = confounded_scm(160, 17);
    let ctx = ExecutionContext::for_tests(7);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32));
    let prepared = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();

    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.refute(&first, &data, RefuteSuite::Cheap, &ctx).unwrap();
    assert_eq!(second.predictive_checks.len(), 2);
    assert!(second.refutations.iter().any(|report| report.refuter.as_ref() == "prior_predictive"));
    assert!(
        second.refutations.iter().any(|report| report.refuter.as_ref() == "posterior_predictive")
    );

    let one_shot = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let second_ids: Vec<&str> =
        second.refutations.iter().map(|report| report.refuter.as_ref()).collect();
    let one_shot_ids: Vec<&str> =
        one_shot.refutations.iter().map(|report| report.refuter.as_ref()).collect();
    assert_eq!(second_ids, one_shot_ids);
}

/// A caller who never touches `.refute(..)` and lands on a query where the
/// default suite (`RefuteSuite::PlaceboAndRcc`) is not supported must be
/// told validation was silently turned off, not left to infer it from an
/// empty `refutations` list. `ResponseCurve` on an explicit `Dag` is
/// licensed at `RefuteSuite::None` (the surface itself) but has no licensed
/// or allowlisted cell at `RefuteSuite::PlaceboAndRcc` — exactly the
/// requested-vs-none-suite gap `StudyBuilder::build` downgrades silently.
#[test]
fn implicit_default_refute_downgrade_is_diagnosed() {
    let n = 120;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.05)
        .collect();
    let data = antecedent_data::TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", z.as_slice()),
    ])
    .unwrap();

    let mut graph = antecedent_graph::Dag::with_variables(3);
    graph
        .insert_directed(
            antecedent_graph::DenseNodeId::from_raw(2),
            antecedent_graph::DenseNodeId::from_raw(0),
        )
        .unwrap();
    graph
        .insert_directed(
            antecedent_graph::DenseNodeId::from_raw(2),
            antecedent_graph::DenseNodeId::from_raw(1),
        )
        .unwrap();
    graph
        .insert_directed(
            antecedent_graph::DenseNodeId::from_raw(0),
            antecedent_graph::DenseNodeId::from_raw(1),
        )
        .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });

    // No `.refute(..)` call: the builder default (`PlaceboAndRcc`) is left in place.
    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(11)).unwrap();

    assert!(result.refutations.is_empty());
    let downgrade = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "exec.refute.default_suite_unsupported")
        .expect("silent refute downgrade must be diagnosed, not silent");
    assert!(
        downgrade
            .fields
            .iter()
            .any(|(k, v)| k.as_ref() == "requested_suite" && v.as_ref() == "placebo+rcc"),
        "{:?}",
        downgrade.fields
    );
    assert!(
        downgrade.fields.iter().any(|(k, v)| k.as_ref() == "applied_suite" && v.as_ref() == "none"),
        "{:?}",
        downgrade.fields
    );
}

/// Same shape as above, but on a family whose default suite IS supported
/// (`AverageEffect` on a confounded `Dag`, `RefuteSuite::PlaceboAndRcc` is
/// licensed) — no downgrade happens, so the diagnostic must not appear.
#[test]
fn implicit_default_refute_no_downgrade_is_not_diagnosed() {
    let (data, dag, query) = confounded_scm(200, 23);
    let ctx = ExecutionContext::for_tests(9);
    let study =
        Study::tabular(data).graph(dag).query(query).bootstrap_replicates(0).build().unwrap();
    let result = study.run(&ctx).unwrap();

    assert!(!result.refutations.is_empty());
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "exec.refute.default_suite_unsupported"),
        "{:?}",
        result.diagnostics
    );
}
