//! The raw static estimate dispatcher warms and reuses the propensity workspace that the
//! overlap refuter then continues from.
//!
//! The dispatcher is crate-internal: only compiled programs reach it outside tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::OverlapPolicy;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_validate::{OverlapRefuter, RefutationProblem};

use super::{EstimatorId, StaticEstimateInput, StaticEstimateWorkspaces, estimate_static_effect};
use crate::{EstimatorSpec, RefuteSuite, Study};

fn unit_noise(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Confounded binary-treatment study over `(t, y, z)` with `z -> t`, `z -> y`, `t -> y`.
fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let zi = unit_noise(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.4 + 0.9 * zi)).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = unit_noise(&mut rng) * 0.4;
        z.push(zi);
        t.push(ti);
        y.push(2.0 * ti + zi + e);
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let column = |id: u32, values: Vec<f64>| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::from(values),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )
    };
    let columns = vec![column(0, t), column(1, y), column(2, z)];
    let data =
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn propensity_workspace_reused_estimate_into_overlap() {
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
        StaticEstimateInput {
            data: &data,
            estimand: &estimand,
            query: &query,
            bootstrap_replicates: 10,
            overlap_policy: Some(OverlapPolicy::require_diagnostics()),
            population_registry: None,
            sampling_design: None,
        },
        estimate.as_effect().unwrap().provenance.assumptions.clone(),
        &ExecutionContext::for_tests(3),
        &mut ws,
    )
    .unwrap();
    let score_grows = ws.propensity.propensity.scores_grow_count;
    let score_cap = ws.propensity.propensity.scores.capacity();
    let ols_grows = ws.propensity.propensity.ols.grow_count;
    assert!(score_grows >= 1 && score_cap > 0, "point propensity fit must warm buffers");
    assert!(estimate.as_effect().unwrap().ate.is_finite());

    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        estimate.as_effect().unwrap(),
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
    let query = query.with_target_population(TargetPopulation::Treated);
    let mut ws2 = StaticEstimateWorkspaces::default();
    let aipw = estimate_static_effect(
        &EstimatorSpec::Default(EstimatorId::Aipw),
        StaticEstimateInput {
            data: &data,
            estimand: &estimand,
            query: &query,
            bootstrap_replicates: 0,
            overlap_policy: None,
            population_registry: None,
            sampling_design: None,
        },
        estimate.as_effect().unwrap().provenance.assumptions.clone(),
        &ExecutionContext::for_tests(4),
        &mut ws2,
    )
    .unwrap();
    assert!(aipw.ate.is_finite());
    assert!(ws2.aipw.propensity.scores_grow_count >= 1 || ws2.aipw.propensity.ols.grow_count >= 1);
}
