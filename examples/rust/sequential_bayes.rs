#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
//! Sequential Bayes: batch A posterior → batch B prior.
//!
//! Fits Bayesian ATE on batch A, encodes the posterior artifact, then re-analyzes
//! an independent batch B with `prior_from_artifact` on the same graph/design.
//!
//! Run: `cargo run -p antecedent --example sequential_bayes`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::io::encode_causal_posterior_bytes;
use antecedent::prelude::*;
use antecedent_core::{CausalRng, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
use antecedent_graph::DenseNodeId;

fn batch(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let e1 = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos();
        let ti = if zi + e1 > 0.0 { 1.0 } else { 0.0 };
        let e2 = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + e2;
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
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
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    (TabularData::new(storage), dag, query)
}

fn main() -> Result<(), CausalError> {
    let (data_a, dag, query) = batch(180, 1);
    let (_, dag_b, _) = batch(180, 2);
    let (data_b, _, _) = batch(180, 2);
    let _ = dag_b;

    let batch_a = CausalAnalysis::builder()
        .data(data_a)
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(11))?;
    let post_a = batch_a.posterior.as_ref().expect("batch A posterior");
    let mean_a = post_a.summaries.mean[post_a.effect_column().unwrap()];
    let artifact = encode_causal_posterior_bytes(post_a, "batch-a")?;

    let batch_b = CausalAnalysis::builder()
        .data(data_b)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(128).prior_from_artifact(artifact, None),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(12))?;
    let post_b = batch_b.posterior.as_ref().expect("batch B posterior");
    let mean_b = post_b.summaries.mean[post_b.effect_column().unwrap()];
    assert!(mean_b.is_finite());
    let assumptions = batch_b.identification.required_assumptions.entries.len();
    assert!(assumptions >= 1);

    println!("A effect_mean={mean_a:.4} B effect_mean={mean_b:.4} assumptions={assumptions}");
    Ok(())
}
