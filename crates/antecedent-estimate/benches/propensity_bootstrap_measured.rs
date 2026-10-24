//! Native measurement of the propensity bootstrap control: allocator bytes and calls,
//! live-byte peak, peak resident set and the engine's counts across complete fits.
//!
//! Same data, estimator (50 bootstrap replicates) and workspace reuse as
//! `propensity_bootstrap`; timing stays with the Criterion bench. Writes
//! `$ANTECEDENT_MEASUREMENT_OUT/propensity_bootstrap/measurement.json`.
//! `ANTECEDENT_LANE_ROWS` sets the row count (default 800).
#![allow(
    missing_docs,
    reason = "benchmark crate root: the allocator static is the only public item"
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{PropensityEstimationWorkspace, PropensityWeighting};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::native_probe::{CountingAllocator, measured_pass, set_counting};
use antecedent_kernels::standard_normal;

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator::new();

fn confounded_scm(n: usize) -> (TabularData, IdentifiedEstimand) {
    let mut rng =
        ExecutionContext::for_tests(11).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);
    let (mut z, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.5 + zi)).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + standard_normal(&mut rng) * 0.5;
    }
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
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
    let cols = [t, y, z]
        .into_iter()
        .enumerate()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(id as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn main() {
    set_counting(false);
    let n: usize =
        std::env::var("ANTECEDENT_LANE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(800);
    let (data, estimand) = confounded_scm(n);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = PropensityWeighting { bootstrap_replicates: 50, ..PropensityWeighting::new() };
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = PropensityEstimationWorkspace::default();
    est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    measured_pass("propensity_bootstrap", 5, &ctx, || {
        est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).map(|effect| effect.se_bootstrap())
    });
}
