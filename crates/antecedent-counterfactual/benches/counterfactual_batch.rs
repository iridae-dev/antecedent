//! Counterfactual batch / streaming equivalence bench .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::{
    ArithOp, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueExpr, ValueType, VariableId,
};
use antecedent_counterfactual::{
    AbductionMissingPolicy, CounterfactualEngine, CounterfactualWorld, ExogenousPosterior,
    NoiseInferenceKind,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, ModelView, NoiseTable, SelectionPolicy,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Rows of the batch policy configuration.
const POLICY_ROWS: usize = 1_000_000;
/// Most a policy world may cost relative to a world without one.
/// The program executor makes a fixed number of cache-resident passes per row block (load,
/// one per operator, the finiteness check and the store), so its marginal cost over a world
/// without the program is bounded by a small multiple of the same expression as a direct loop.
const MAX_REGRESSION: f64 = 6.0;
const REPEATS: usize = 11;

fn engine() -> (CounterfactualEngine, TabularData) {
    let n = 100usize;
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
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = t.iter().enumerate().map(|(i, ti)| 2.0 * ti + 0.01 * i as f64).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    (CounterfactualEngine::new(compiled.with_mechanisms(store)), data)
}

fn bench_cf(c: &mut Criterion) {
    let (eng, data) = engine();
    let exo =
        eng.abduct(&data, AbductionMissingPolicy::Error, &ExecutionContext::for_tests(1)).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    c.bench_function("counterfactual_predict_n100", |b| {
        b.iter(|| {
            let mut ws = MechanismWorkspace::default();
            let worlds = [CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(1.0),
                )]),
            }];
            let res = eng
                .predict(&exo, &worlds, &[VariableId::from_raw(1)], false, &mut ws, &ctx)
                .unwrap();
            assert!(res.finite_outcome_count(0, DenseNodeId::from_raw(1)) > 0);
            res
        });
    });
    c.bench_function("counterfactual_predict_n100_unit_rows", |b| {
        b.iter(|| {
            let mut ws = MechanismWorkspace::default();
            let rows: Arc<[usize]> = (0..10).collect();
            let worlds = [CounterfactualWorld {
                unit_rows: Some(rows),
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(1.0),
                )]),
            }];
            let res = eng
                .predict(&exo, &worlds, &[VariableId::from_raw(1)], false, &mut ws, &ctx)
                .unwrap();
            assert!(res.finite_outcome_count(0, DenseNodeId::from_raw(1)) > 0);
            res
        });
    });
}

/// `a -> b -> c`, linear Gaussian, with abduced noise for `POLICY_ROWS` units.
fn chain_exogenous() -> (CounterfactualEngine, ExogenousPosterior) {
    let mut g = Dag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let linear = |intercept: f64, coeffs: &[f64], sigma: f64| MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(coeffs.to_vec()),
        sigma,
    };
    let model = CompiledCausalModel::compile(g).unwrap().with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            linear(1.0, &[], 1.0),
            linear(0.5, &[2.0], 0.5),
            linear(-1.0, &[3.0], 2.0),
        ]),
    });
    let ctx = ExecutionContext::for_tests(3_076_196);
    let table = NoiseTable::sample(&ModelView::observational(&model), POLICY_ROWS, &ctx).unwrap();
    let exo = ExogenousPosterior {
        noise: Arc::from(table.values()),
        n_units: POLICY_ROWS,
        n_nodes: 3,
        kind: NoiseInferenceKind::Invertible,
        assumed_columns: Arc::from([false, false, false]),
    };
    (CounterfactualEngine::new(model), exo)
}

/// The price floor as a direct loop over one column: what the executor's marginal cost is
/// measured against. Use the public program arithmetic, including NaN propagation,
/// and its finite-output validation. `f64::max` alone has different semantics and omits
/// a required pass, making it an artificially cheap reference on some CPUs.
fn native_floor(column: &[f64], scratch: &mut [f64]) -> Duration {
    scratch.copy_from_slice(column);
    let started = Instant::now();
    for v in scratch.iter_mut() {
        *v = native_floor_value(*v);
    }
    let largest = scratch.iter().map(|v| v.to_bits() & !(1_u64 << 63)).max().unwrap_or(0);
    assert!(black_box(largest) < f64::INFINITY.to_bits(), "direct policy must be finite");
    let elapsed = started.elapsed();
    black_box(scratch);
    elapsed
}

fn native_floor_value(value: f64) -> f64 {
    ArithOp::Max.apply(1.0, ArithOp::Mul.apply(0.85, value))
}

/// Batch counterfactual execution of a value-program policy against a world without one.
///
/// Streaming (outcomes only) and retained results agree bit for bit, one context and
/// workspace serve every execution, and the policy costs at most the regression margin.
fn bench_cf_policy(c: &mut Criterion) {
    // Reference semantics, outside all measured regions. In particular, f64::max
    // would silently turn the NaN case into 1.0 instead of preserving the refusal.
    for (value, expected) in [(-0.0, 1.0_f64), (0.0, 1.0), (2.0, 1.7), (f64::NEG_INFINITY, 1.0)] {
        assert_eq!(native_floor_value(value).to_bits(), expected.to_bits());
    }
    assert!(native_floor_value(f64::NAN).is_nan());
    assert_eq!(native_floor_value(f64::INFINITY).to_bits(), f64::INFINITY.to_bits());
    let (eng, exo) = chain_exogenous();
    let ctx = ExecutionContext::for_tests(1);
    let outcome = VariableId::from_raw(2);
    let floor = ValueExpr::constant(1.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
    let policy = CounterfactualWorld {
        unit_rows: None,
        interventions: Arc::from([Intervention::transform(VariableId::from_raw(0), floor)]),
    };
    let plain = CounterfactualWorld { unit_rows: None, interventions: Arc::from([]) };
    let mut ws = MechanismWorkspace::default();
    let run = |world: &CounterfactualWorld, ws: &mut MechanismWorkspace| {
        let started = Instant::now();
        let result = eng
            .predict_retaining_outcomes(
                &exo,
                std::slice::from_ref(world),
                &[outcome],
                false,
                ws,
                &ctx,
            )
            .unwrap();
        let elapsed = started.elapsed();
        black_box(result);
        elapsed
    };

    let mut group = c.benchmark_group("counterfactual_policy");
    group.sample_size(10);
    group.bench_function("counterfactual_plain_n1000000", |b| b.iter(|| run(&plain, &mut ws)));
    group.bench_function("counterfactual_price_floor_program_n1000000", |b| {
        b.iter(|| run(&policy, &mut ws));
    });
    group.finish();

    // Streaming and retained results are the same bits.
    let node = DenseNodeId::from_raw(2);
    let retained =
        eng.predict(&exo, std::slice::from_ref(&policy), &[outcome], false, &mut ws, &ctx).unwrap();
    let streamed = eng
        .predict_retaining_outcomes(
            &exo,
            std::slice::from_ref(&policy),
            &[outcome],
            false,
            &mut ws,
            &ctx,
        )
        .unwrap();
    let retained_bits: Vec<u64> =
        retained.outcome_column(0, node).unwrap().iter().map(|v| v.to_bits()).collect();
    let streamed_bits: Vec<u64> =
        streamed.outcome_column(0, node).unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(retained_bits, streamed_bits, "streaming must equal retained");

    // Regression gate: pair the policy and plain worlds, alternate their order, and
    // measure the equivalent direct kernel alongside each pair. The median paired
    // ratio avoids subtracting independent minima or a single scheduler interruption.
    run(&policy, &mut ws);
    run(&plain, &mut ws);
    // The root mechanism generates 1 + noise, rather than the raw noise itself.
    let generated: Vec<f64> = exo.noise[..POLICY_ROWS].iter().map(|noise| 1.0 + noise).collect();
    let column = generated.as_slice();
    let mut scratch = column.to_vec();
    native_floor(column, &mut scratch);
    let mut samples = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        let (without, with) = if repeat % 2 == 0 {
            (run(&plain, &mut ws), run(&policy, &mut ws))
        } else {
            let with = run(&policy, &mut ws);
            (run(&plain, &mut ws), with)
        };
        let native = native_floor(column, &mut scratch);
        let marginal = with.saturating_sub(without);
        let ratio = marginal.as_secs_f64() / native.as_secs_f64().max(1e-9);
        samples.push((ratio, marginal, with, without, native));
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (ratio, marginal, with, without, native) = samples[REPEATS / 2];
    eprintln!("paired median policy overhead: {ratio:.2}x ({marginal:?}); direct {native:?}");
    assert!(
        ratio <= MAX_REGRESSION,
        "the policy program added {marginal:?} to a counterfactual world ({with:?} vs \
         {without:?}), {ratio:.2}x the same floor as a direct loop ({native:?})"
    );
}

criterion_group!(benches, bench_cf, bench_cf_policy);
criterion_main!(benches);
