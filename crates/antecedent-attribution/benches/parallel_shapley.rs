//! Shapley coalition values and per-unit rows on worker threads.
//!
//! The serial and threaded runs are bit-identical (unit-tested); this bench shows what
//! the second thread count buys. `NoFork` hides `fork`, which is how a payoff ran before
//! coalitions could be evaluated on workers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "bench scaffolding indexes with small literals"
)]

use std::sync::Arc;

use antecedent_attribution::{
    AttributionError, CoalitionPayoff, DistributionChangeOptions, distribution_change,
    estimate_shapley, unit_change,
};
use antecedent_core::{
    AllocationMethod, CausalSchemaBuilder, ChangeAttributionQuery, ComponentId, ExecutionContext,
    MeasurementSpec, NonZeroThreadCount, Parallelism, PopulationSelector, RoleHint, ShapleyConfig,
    SmallRoleSet, UnitChangeQuery, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// A coalition value that costs real arithmetic and depends only on the mask.
#[derive(Clone)]
struct Heavy {
    weights: Vec<f64>,
}

impl CoalitionPayoff for Heavy {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let mut acc = 0.0_f64;
        for round in 0..4_000 {
            for (i, &w) in self.weights.iter().enumerate() {
                if mask & (1u64 << i) != 0 {
                    acc += (w * f64::from(round) * 1e-3).sin();
                }
            }
        }
        Ok(acc)
    }

    fn fork<'a>(&'a self) -> Option<Box<dyn CoalitionPayoff + Send + 'a>> {
        Some(Box::new(self.clone()))
    }
}

struct NoFork(Heavy);

impl CoalitionPayoff for NoFork {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        self.0.value(mask)
    }
}

fn ctx(threads: u32) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(7);
    ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(threads).unwrap());
    ctx
}

fn heavy(n: usize) -> Heavy {
    Heavy { weights: (0..n).map(|i| 1.0 + 0.1 * i as f64).collect() }
}

fn bench_heavy_payoff(c: &mut Criterion) {
    let players: Vec<_> = (0..10u32).map(ComponentId::from_raw).collect();
    let exact = ShapleyConfig::exact().with_max_exact_components(12);
    let sampled = ShapleyConfig::monte_carlo(120).with_seed(3);
    for (name, cfg) in [("exact_10p", exact), ("mc_10p_120", sampled)] {
        c.bench_function(&format!("shapley_heavy_{name}_nofork"), |b| {
            b.iter(|| {
                let mut payoff = NoFork(heavy(10));
                black_box(estimate_shapley(&players, &cfg, &mut payoff, &ctx(4)).unwrap());
            });
        });
        for threads in [1u32, 4] {
            c.bench_function(&format!("shapley_heavy_{name}_fork_{threads}t"), |b| {
                b.iter(|| {
                    let mut payoff = heavy(10);
                    black_box(
                        estimate_shapley(&players, &cfg, &mut payoff, &ctx(threads)).unwrap(),
                    );
                });
            });
        }
    }
}

fn chain(n: usize, width: usize, fan_in: bool) -> (CompiledCausalModel, TabularData) {
    let mut b = CausalSchemaBuilder::new();
    for k in 0..width {
        let role = if k + 1 == width { RoleHint::OutcomeCandidate } else { RoleHint::Context };
        b.add_variable(
            format!("v{k}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut columns = vec![Vec::with_capacity(n); width];
    for i in 0..n {
        let shifted = if i < n / 2 { 0.0 } else { 1.0 };
        let mut previous = ((i as f64 + 1.0) * 0.37).sin();
        columns[0].push(previous);
        for (k, column) in columns.iter_mut().enumerate().skip(1) {
            let wobble = ((i as f64 + 1.0) * (0.7 + k as f64 * 0.31)).sin();
            let value = (0.7 + 0.3 * shifted) * previous + 0.3 * wobble + 0.2 * shifted;
            column.push(value);
            previous = value;
        }
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols: Vec<_> = columns
        .into_iter()
        .enumerate()
        .map(|(k, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(values),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(width as u32);
    for k in 1..width {
        g.insert_directed(DenseNodeId::from_raw(k as u32 - 1), DenseNodeId::from_raw(k as u32))
            .unwrap();
    }
    if fan_in {
        for k in 0..width.saturating_sub(2) {
            g.insert_directed(
                DenseNodeId::from_raw(k as u32),
                DenseNodeId::from_raw(width as u32 - 1),
            )
            .unwrap();
        }
    }
    (CompiledCausalModel::compile(g).unwrap(), data)
}

fn bench_distribution_change(c: &mut Criterion) {
    let (model, data) = chain(2_000, 8, false);
    let query = ChangeAttributionQuery::new(
        VariableId::from_raw(7),
        PopulationSelector::TimeRange { start: 0, end: 1_000 },
        PopulationSelector::TimeRange { start: 1_000, end: 2_000 },
    )
    .with_allocation(AllocationMethod::Shapley {
        approximation: ShapleyConfig::monte_carlo(40).with_seed(1),
    });
    let options = DistributionChangeOptions::default();
    for threads in [1u32, 4] {
        c.bench_function(&format!("distribution_change_mc_8p_40_{threads}t"), |b| {
            b.iter(|| {
                black_box(distribution_change(&model, &data, &query, &options, &ctx(threads)))
                    .unwrap();
            });
        });
    }
}

fn bench_unit_change(c: &mut Criterion) {
    // Every earlier variable also feeds the outcome, so the unit game has several players.
    let (compiled, data) = chain(400, 6, true);
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    let model = compiled.with_mechanisms(store);
    let query = UnitChangeQuery::new(VariableId::from_raw(5), 1_000).with_allocation(
        AllocationMethod::Shapley { approximation: ShapleyConfig::monte_carlo(200).with_seed(2) },
    );
    for threads in [1u32, 4] {
        c.bench_function(&format!("unit_change_400_rows_mc200_{threads}t"), |b| {
            b.iter(|| black_box(unit_change(&model, &data, &query, &ctx(threads)).unwrap()));
        });
    }
}

criterion_group!(benches, bench_heavy_payoff, bench_distribution_change, bench_unit_change);
criterion_main!(benches);
