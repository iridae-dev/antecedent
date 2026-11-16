#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! Estimate a lagged effect in a time series with a declared temporal graph.
//!
//! Question: by how much does a unit pulse in line pressure change the defect rate one step
//! later?
//!
//! Data and truth: 400 hourly points with `pressure[t] = sin(0.04 t)` and
//! `defect[t] = 0.9 pressure[t - 1]`, noise-free. The temporal graph has the single edge from
//! pressure at lag 1 to the contemporaneous defect rate, so the effect of the pulse is the
//! structural coefficient 0.9.
//!
//! Asserted: the estimate is within 0.05 of 0.9.
//!
//! What this establishes: with the lag structure declared, the pulse query returns the lag-1
//! coefficient. The series has no noise and no confounding, and the bootstrap and refuters are
//! off, so the example says nothing about uncertainty or about a misdeclared lag.
//!
//! Run: `cargo run -p antecedent --example manufacturing_temporal`
//! Source: `examples/rust/manufacturing_temporal.rs`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::prelude::*;
use antecedent_core::{Lag, MeasurementSpec, RoleHint, SmallRoleSet, TemporalPolicy, ValueType};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};
use antecedent_graph::ensure_lagged;

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let n = 400usize;
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 0..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build()?;
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )?;

    // The temporal graph states the lag: pressure one step back causes the current defect rate.
    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1))?;
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS)?;
    g.insert_directed(p1, d0)?;

    // A pulse of size 1 applied one step before the outcome time, read at a one-step horizon.
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let result = Study::series(series)
        .graph(g)
        .temporal_query(q)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()?
        .run(&ExecutionContext::for_tests(42))?;

    println!(
        "ATE={:.4} plan={} peak_mem={:?} method={}",
        result.estimate().as_effect().unwrap().ate,
        result.logical_plan.plan_id,
        result.physical_plan.estimated_peak_memory_bytes,
        result.estimand.method,
    );
    assert!(
        (result.estimate().as_effect().unwrap().ate - 0.9).abs() < 0.05,
        "ate={}",
        result.estimate().as_effect().unwrap().ate
    );
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
