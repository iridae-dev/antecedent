//! Propensity-weighting bootstrap benchmark .
//!
//! Times `PropensityWeighting::fit` with bootstrap replicates enabled, refitting the
//! propensity model each replicate while reusing `PropensityEstimationWorkspace` scratch.
#![allow(missing_docs, clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_estimate::{PropensityEstimationWorkspace, PropensityWeighting};
use causal_expr::ExprId;
use causal_expr::IdentifiedEstimand;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn standard_normal(rng: &mut causal_core::CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn confounded_scm(n: usize) -> (TabularData, IdentifiedEstimand) {
    let mut rng = ExecutionContext::for_tests(11).rng.stream(0x1234_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.5 + zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
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
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn bench_propensity_bootstrap(c: &mut Criterion) {
    let (data, estimand) = confounded_scm(800);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 50, ..PropensityWeighting::new() };
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);

    c.bench_function("propensity_weighting_ipw_bootstrap50_n800", |b| {
        // Warm once so workspace buffers are sized; then assert reuse across timed iters.
        let mut ws = PropensityEstimationWorkspace::default();
        let _ = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let grows_after_warm = ws.propensity.ols.grow_count;
        let scores_grows_after_warm = ws.propensity.scores_grow_count;
        let scratch_ptr = ws.propensity.ols.scratch.as_ptr();
        b.iter(|| {
            let effect = est.fit(black_box(&prep), &mut ws, &ctx, AssumptionSet::new()).unwrap();
            black_box(effect.se_bootstrap);
            assert_eq!(
                ws.propensity.ols.grow_count, grows_after_warm,
                "OLS scratch must not grow across bootstrap fits of fixed n"
            );
            assert_eq!(
                ws.propensity.scores_grow_count, scores_grows_after_warm,
                "propensity score buffer must not grow across bootstrap fits of fixed n"
            );
            assert_eq!(ws.propensity.ols.scratch.as_ptr(), scratch_ptr);
        });
    });
}

criterion_group!(benches, bench_propensity_bootstrap);
criterion_main!(benches);
