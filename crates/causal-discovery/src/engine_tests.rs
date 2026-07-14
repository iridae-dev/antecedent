//! Engine unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, NonZeroThreadCount, Parallelism,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{
    Float64Column, LaggedFrame, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};

use super::*;
use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
use crate::pcmci::Pcmci;

fn var_series() -> (TimeSeriesData, Vec<VariableId>) {
    // Y_t = 0.8 X_{t-1} + noise; X_t = noise
    let n = 400usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.01).sin();
        y[t] = 0.8 * x[t - 1] + 0.01 * ((t as f64) * 0.03).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
}

fn constraints() -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::from_raw(1) },
        alpha: 0.05,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    }
}

#[test]
fn recovers_lagged_parent() {
    let (data, vars) = var_series();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(9);
    let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    let has = result.evidence.links.iter().any(|s| {
        s.link.source == VariableId::from_raw(0)
            && s.link.target == VariableId::from_raw(1)
            && s.link.source_lag.raw() == 1
    });
    assert!(has, "links={:?}", result.evidence.links);
    assert_eq!(result.review.pending_edges.len(), result.evidence.links.len());
    assert_eq!(result.review.algorithm.as_ref(), "pcmci.engine.pc_mci");
}

#[test]
fn pcmci_alpha_filters_unthresholded_engine() {
    let (data, vars) = var_series();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(9);
    let engine = PcmciEngine::new().with_constraints(constraints());
    let raw = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints());
    let filtered = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(raw.evidence.links.len() >= filtered.evidence.links.len());
    assert!(filtered.evidence.links.iter().all(|s| s.p_value < 0.05));
}

#[test]
fn fdr_adjusts_full_mci_family() {
    let (data, vars) = var_series();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let with_fdr = Pcmci::new().with_fdr(true).with_constraints(constraints());
    let without = Pcmci::new().with_fdr(false).with_constraints(constraints());
    let a = with_fdr.run(&data, &vars, &mut ws, &ctx).unwrap();
    let b = without.run(&data, &vars, &mut ws, &ctx).unwrap();
    // FDR is more conservative; retained set is a subset (or equal) of alpha-only.
    let set_a: std::collections::BTreeSet<_> = a.evidence.links.iter().map(|s| s.link).collect();
    let set_b: std::collections::BTreeSet<_> = b.evidence.links.iter().map(|s| s.link).collect();
    assert!(set_a.is_subset(&set_b));
}

#[test]
fn parallel_matches_serial_link_set() {
    let (data, vars) = var_series();
    let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let serial = ExecutionContext::for_tests(3);
    let mut parallel = ExecutionContext::for_tests(3);
    parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).expect("threads"));
    let a = pcmci.run(&data, &vars, &mut ws, &serial).unwrap();
    let b = pcmci.run(&data, &vars, &mut ws, &parallel).unwrap();
    let set_a: std::collections::BTreeSet<_> = a
        .evidence
        .links
        .iter()
        .map(|s| (s.link, s.statistic.to_bits(), s.p_value.to_bits()))
        .collect();
    let set_b: std::collections::BTreeSet<_> = b
        .evidence
        .links
        .iter()
        .map(|s| (s.link, s.statistic.to_bits(), s.p_value.to_bits()))
        .collect();
    assert_eq!(set_a, set_b);
    assert_eq!(b.performance.worker_threads, 4);
}

#[test]
fn phase2_ci_hot_path_no_scratch_growth() {
    let (data, vars) = var_series();
    let frame = LaggedFrame::from_series(&data, &vars, 2).unwrap();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    // Warm up capacities with the same query shape as the steady-state loop.
    let cond = [(VariableId::from_raw(0), Lag::from_raw(2))];
    let _ = engine
        .ci_statistic(
            &frame,
            VariableId::from_raw(0),
            Lag::from_raw(1),
            VariableId::from_raw(1),
            Lag::CONTEMPORANEOUS,
            &cond,
            &mut ws,
            &ctx,
        )
        .unwrap();
    let col_cap = ws.col_idxs.capacity();
    let z_cap = ws.z_flat.capacity();
    let ci_n = ws.ci.parcorr.capacity_n();
    let ci_p = ws.ci.parcorr.capacity_p();
    let col_ptr = ws.col_idxs.as_ptr();
    let z_ptr = ws.z_flat.as_ptr();
    let design_ptr = ws.ci.parcorr.design.as_ptr();
    let design_cap = ws.ci.parcorr.design.capacity();
    for _ in 0..200 {
        let _ = engine
            .ci_statistic(
                &frame,
                VariableId::from_raw(0),
                Lag::from_raw(1),
                VariableId::from_raw(1),
                Lag::CONTEMPORANEOUS,
                &cond,
                &mut ws,
                &ctx,
            )
            .unwrap();
        assert_eq!(ws.col_idxs.capacity(), col_cap);
        assert_eq!(ws.z_flat.capacity(), z_cap);
        assert_eq!(ws.ci.parcorr.capacity_n(), ci_n);
        assert_eq!(ws.ci.parcorr.capacity_p(), ci_p);
        assert_eq!(ws.col_idxs.as_ptr(), col_ptr);
        assert_eq!(ws.z_flat.as_ptr(), z_ptr);
        assert_eq!(ws.ci.parcorr.design.as_ptr(), design_ptr);
        assert_eq!(ws.ci.parcorr.design.capacity(), design_cap);
    }
}

#[test]
fn engine_accepts_oracle_ci() {
    use causal_stats::OracleCi;

    let (data, vars) = var_series();
    // Local batch always uses x=0,y=1; empty deps ⇒ every pair independent.
    let engine =
        PcmciEngine::new().with_constraints(constraints()).with_ci(Arc::new(OracleCi::new([])));
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(
        result.evidence.links.is_empty(),
        "oracle with no deps should drop all links, got {:?}",
        result.evidence.links
    );

    let engine_dep = PcmciEngine::new()
        .with_constraints(constraints())
        .with_ci(Arc::new(OracleCi::new([(0usize, 1usize)])));
    let kept = engine_dep.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(!kept.evidence.links.is_empty(), "oracle marking (0,1) dependent should retain links");
}

#[test]
fn mci_conditioning_shifts_source_parents_by_link_lag() {
    // Link X_{t-2} → Y_t: pa(X) = {(X,1)} keyed at lag 0 must condition as X_{t-3},
    // and pa(Y) entries pass through minus the link endpoints.
    let x = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let link = LaggedLink {
        source: x,
        source_lag: Lag::from_raw(2),
        target: y,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let parents_target = [(x, Lag::from_raw(2)), (y, Lag::from_raw(1))];
    let parents_source = [(x, Lag::from_raw(1))];
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &parents_source, &mut out);
    assert_eq!(dropped, 0);
    assert_eq!(out, vec![(y, Lag::from_raw(1)), (x, Lag::from_raw(3))]);
}

#[test]
fn mci_conditioning_keeps_shifted_autocorrelation_parent() {
    // Link X_{t-1} → Y_t with pa(X) = {(X,1)}: the unshifted parent would collide with
    // the link source and be dropped; the shifted parent X_{t-2} must be conditioned on.
    let x = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let link = LaggedLink {
        source: x,
        source_lag: Lag::from_raw(1),
        target: y,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let parents_target = [(x, Lag::from_raw(1))];
    let parents_source = [(x, Lag::from_raw(1))];
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &parents_source, &mut out);
    assert_eq!(dropped, 0);
    assert_eq!(out, vec![(x, Lag::from_raw(2))]);
}
