//! Sequential prior hydration from posterior artifacts carries the full
//! coefficient covariance (dense `V0`); summary-only artifacts stay diagonal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::io::encode_causal_posterior_bytes;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    Assumption, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::{PosteriorPayload, PriorMapping, encode_causal_posterior_bytes_with_payload};

/// `Z → T`, `Z → Y`, `T → Y` with `T` strongly driven by `Z`, so the
/// treatment and `Z` coefficients are correlated in the posterior.
fn confounded(n: usize, shift: usize) -> (TabularData, VariableId, VariableId) {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("Z", RoleHint::Context),
        ("T", RoleHint::TreatmentCandidate),
        ("Y", RoleHint::OutcomeCandidate),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let (z, t, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let (mut zv, mut tv, mut yv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let k = i + shift;
        let zk = ((k * 37 % 101) as f64) / 101.0;
        let u = ((k * 53 % 97) as f64) / 97.0;
        let tk = if zk + 0.25 * u > 0.6 { 1.0 } else { 0.0 };
        zv.push(zk);
        tv.push(tk);
        yv.push(2.0 * tk + 1.5 * zk + (((k * 29) % 11) as f64 - 5.0) * 0.12);
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    (TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap()), t, y)
}

fn dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    dag
}

fn run(
    data: TabularData,
    t: VariableId,
    y: VariableId,
    cfg: BayesianConfig,
) -> antecedent::StudyResult {
    Study::tabular(data)
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(cfg))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap()
}

#[test]
fn artifact_hydration_carries_dense_v0_and_summary_artifacts_stay_diagonal() {
    let (data_a, t, y) = confounded(60, 0);
    let result_a = run(data_a, t, y, BayesianConfig::conjugate().n_draws(2000).prior_scale(10.0));
    let post_a = result_a.posterior.as_ref().unwrap();
    let full = encode_causal_posterior_bytes(post_a, "source-a").unwrap();
    let summary =
        encode_causal_posterior_bytes_with_payload(post_a, "source-a", PosteriorPayload::Summary)
            .unwrap();

    // The facade's sequential path uses the dense prior and says so.
    let (data_b, _, _) = confounded(60, 500);
    let seq = run(
        data_b,
        t,
        y,
        BayesianConfig::conjugate()
            .n_draws(500)
            .prior_from_artifact(full, Some(PriorMapping::IdenticalCoefficientSubspace)),
    );
    let post = seq.posterior.as_ref().unwrap();
    assert!(post.assumptions.entries.iter().any(|a| matches!(
        &a.assumption,
        Assumption::PriorRestriction(pa) if pa.id.as_ref() == "hydrated_coefficient_covariance"
    )));
    assert!(post.assumptions.entries.iter().any(|a| matches!(
        &a.assumption,
        Assumption::PriorRestriction(pa) if pa.description.contains("dense V0")
    )));

    // A summary-only artifact has no draws, hence no covariance: the prior stays diagonal.
    let (data_c, _, _) = confounded(60, 500);
    let diagonal = run(
        data_c,
        t,
        y,
        BayesianConfig::conjugate()
            .n_draws(500)
            .prior_from_artifact(summary, Some(PriorMapping::IdenticalCoefficientSubspace)),
    );
    let diagonal_post = diagonal.posterior.as_ref().unwrap();
    assert!(!diagonal_post.assumptions.entries.iter().any(|a| matches!(
        &a.assumption,
        Assumption::PriorRestriction(pa) if pa.id.as_ref() == "hydrated_coefficient_covariance"
    )));
}
