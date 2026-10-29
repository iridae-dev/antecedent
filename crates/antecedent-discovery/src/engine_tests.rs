//! Engine unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, NonZeroThreadCount, Parallelism,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, LaggedFrame, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};

use super::*;
use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
use crate::pcmci::Pcmci;
use crate::result::LaggedLink;

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
    // X is white noise: a smooth sinusoid makes the lagged columns collinear, which the CI
    // test (correctly) reports as untestable, so every link would score statistic 0, p = 1.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let x = uniform_draws(&mut state, n);
    let mut y = vec![0.0; n];
    for t in 1..n {
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

fn alpha_of(c: &DiscoveryConstraints) -> f64 {
    c.alpha
}

#[test]
fn recovers_lagged_parent() {
    let (data, vars) = var_series();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(9);
    let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    // The full constrained family is scored, so mere presence proves nothing: the true link
    // must be the strong one (y = 0.8 x_{t-1} + 0.01 cos, partial correlation 0.9996) and
    // white-noise x must have no significant lagged parent.
    let scored = |source: u32, lag: u32, target: u32| {
        result
            .evidence
            .links
            .iter()
            .find(|s| {
                s.link.source == VariableId::from_raw(source)
                    && s.link.target == VariableId::from_raw(target)
                    && s.link.source_lag.raw() == lag
            })
            .unwrap_or_else(|| panic!("link {source}@{lag}->{target} unscored"))
    };
    let true_link = scored(0, 1, 1);
    assert!(true_link.statistic > 0.999 && true_link.p_value < 1e-12, "{true_link:?}");
    for (source, lag) in [(0, 1), (0, 2), (1, 1), (1, 2)] {
        let spurious = scored(source, lag, 0);
        assert!(spurious.p_value > alpha_of(&constraints()), "x has no parent: {spurious:?}");
        assert!(spurious.statistic.abs() < 0.1, "{spurious:?}");
    }
    assert_eq!(result.review.pending_edges.len(), result.evidence.links.len());
    assert_eq!(result.review.algorithm.as_ref(), "pcmci.engine.pc_mci");
}

#[test]
fn recovers_lagged_parent_with_analysis_mask() {
    let (data, vars) = var_series();
    let n = data.row_count();
    // Hide every 7th row via analysis mask; masked MCI must still recover X→Y@1.
    let mut bytes = vec![0xFFu8; n.div_ceil(8)];
    for row in (0..n).step_by(7) {
        bytes[row / 8] &= !(1 << (row % 8));
    }
    let mask = ValidityBitmap::from_bytes(bytes, n).unwrap();
    let data = data.with_analysis_mask(mask).unwrap();
    assert!(
        !LaggedFrame::from_series(
            &data,
            &vars,
            2,
            &antecedent_core::KernelPolicy::default_policy()
        )
        .unwrap()
        .is_fully_valid()
    );

    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(9);
    let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    let true_link = result
        .evidence
        .links
        .iter()
        .find(|s| {
            s.link.source == VariableId::from_raw(0)
                && s.link.target == VariableId::from_raw(1)
                && s.link.source_lag.raw() == 1
        })
        .expect("the true link is scored");
    // Presence is guaranteed by the full-family scoring; the masked test must still find the
    // dependence (partial correlation 0.9996 on the 6/7 of the rows that are kept).
    assert!(true_link.statistic > 0.999 && true_link.p_value < 1e-12, "{true_link:?}");
    assert!(!ws.keep_cache.is_empty(), "keep-mask cache should be populated");
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
    assert!(filtered.evidence.links.iter().all(|s| s.p_value <= 0.05));
    // The alpha keep is exactly the definition: the raw links with p <= alpha, no more, no less.
    let by_definition: std::collections::BTreeSet<_> =
        raw.evidence.links.iter().filter(|s| s.p_value <= 0.05).map(|s| s.link).collect();
    let kept: std::collections::BTreeSet<_> =
        filtered.evidence.links.iter().map(|s| s.link).collect();
    assert_eq!(kept, by_definition);
    assert!(!kept.is_empty() && kept.len() < raw.evidence.links.len());
}

#[test]
fn fdr_adjusts_full_mci_family() {
    let (data, vars) = var_series();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let cons = constraints();
    let engine = PcmciEngine::new().with_constraints(cons.clone());
    let raw = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    // Full family: N targets × candidates per target (min_lag..=max_lag, skip self@0).
    let mut expected = 0usize;
    for &t in &vars {
        expected += cons.candidate_sources(&vars, t).len();
    }
    assert_eq!(
        raw.evidence.links.len(),
        expected,
        "MCI must score the full constrained family, got {:?}",
        raw.evidence.links.len()
    );

    let with_fdr = Pcmci::new().with_fdr(true).with_constraints(cons.clone());
    let without = Pcmci::new().with_fdr(false).with_constraints(cons.clone());
    let a = with_fdr.run(&data, &vars, &mut ws, &ctx).unwrap();
    let b = without.run(&data, &vars, &mut ws, &ctx).unwrap();
    // FDR is more conservative; retained set is a subset (or equal) of alpha-only.
    let set_a: std::collections::BTreeSet<_> = a.evidence.links.iter().map(|s| s.link).collect();
    let set_b: std::collections::BTreeSet<_> = b.evidence.links.iter().map(|s| s.link).collect();
    assert!(set_a.is_subset(&set_b));
    // Benjamini-Hochberg over the full family of m scored links, recomputed from the raw
    // p-values: adjusted_(i) = min over j >= i of p_(j) m / j, kept when adjusted <= alpha.
    let mut sorted: Vec<f64> = raw.evidence.links.iter().map(|s| s.p_value).collect();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len() as f64;
    let mut adjusted = vec![0.0; sorted.len()];
    let mut running = f64::INFINITY;
    for i in (0..sorted.len()).rev() {
        running = running.min(sorted[i] * m / (i as f64 + 1.0)).min(1.0);
        adjusted[i] = running;
    }
    let bh = |p: f64| adjusted[sorted.iter().position(|q| *q == p).unwrap()];
    for link in a.evidence.links.iter() {
        let expected = bh(link.p_value);
        let got = link.adjusted_p_value.expect("adjusted");
        assert!((got - expected).abs() < 1e-12, "{:?}: {got} vs {expected}", link.link);
    }
    let expected_kept: std::collections::BTreeSet<_> = raw
        .evidence
        .links
        .iter()
        .filter(|s| bh(s.p_value) <= 0.05)
        .map(|s| s.link)
        .collect();
    assert_eq!(set_a, expected_kept);
    // The kept set is pinned at levels straddling the corrected value of the spurious
    // x_{t-2} -> y link (raw 0.034, BH 0.034 * 8 / 4 = 0.068): kept at 0.07, dropped at 0.065.
    for (alpha, spurious_kept) in [(0.05, false), (0.065, false), (0.07, true)] {
        let c = DiscoveryConstraints { alpha, ..cons.clone() };
        let family = PcmciEngine::new()
            .with_constraints(c.clone())
            .run_pc_mci(&data, &vars, &mut ws, &ctx)
            .unwrap();
        let mut sorted: Vec<f64> = family.evidence.links.iter().map(|s| s.p_value).collect();
        sorted.sort_by(f64::total_cmp);
        let m = sorted.len();
        let mut adjusted = vec![0.0; m];
        let mut running = f64::INFINITY;
        for i in (0..m).rev() {
            running = running.min(sorted[i] * m as f64 / (i as f64 + 1.0)).min(1.0);
            adjusted[i] = running;
        }
        let expected: std::collections::BTreeSet<_> = family
            .evidence
            .links
            .iter()
            .filter(|s| adjusted[sorted.iter().position(|q| *q == s.p_value).unwrap()] <= alpha)
            .map(|s| s.link)
            .collect();
        let got = Pcmci::new()
            .with_fdr(true)
            .with_constraints(c)
            .run(&data, &vars, &mut ws, &ctx)
            .unwrap();
        let got: std::collections::BTreeSet<_> = got.evidence.links.iter().map(|s| s.link).collect();
        assert_eq!(got, expected, "alpha {alpha}");
        let spurious = LaggedLink {
            source: VariableId::from_raw(0),
            source_lag: Lag::from_raw(2),
            target: VariableId::from_raw(1),
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert_eq!(got.contains(&spurious), spurious_kept, "alpha {alpha}");
    }
    // The spurious x_{t-2} -> y link (raw p = 0.034) is exactly what the correction removes.
    assert!(set_b.len() > set_a.len(), "alpha-only {} vs FDR {}", set_b.len(), set_a.len());
    assert!(a.evidence.links.iter().all(|s| s.adjusted_p_value.is_some()));
    assert!(b.evidence.links.iter().all(|s| s.adjusted_p_value.is_none()));
    assert!(a.evidence.edge_evidence.iter().all(|e| e.adjusted_p_value.is_some()));
}

#[test]
fn mci_recovers_pc_false_negative() {
    // Three vars: Y ← X_{t-1} strongly, Z independent noise. Force PC to drop X→Y by
    // requiring a decoy parent that crowds out via max_parents=1, then confirm MCI still
    // scores X→Y (full family) and recovers it after alpha keep.
    let n = 500usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y", "z"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    // White-noise x and z: smooth sinusoids would make the lagged columns collinear.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let x = uniform_draws(&mut state, n);
    let z = uniform_draws(&mut state, n);
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = 0.9 * x[t - 1] + 0.01 * ((t as f64) * 0.03).cos();
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
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let vars = vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let x_id = VariableId::from_raw(0);
    let y_id = VariableId::from_raw(1);
    let z_id = VariableId::from_raw(2);

    let constraints = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
        alpha: 0.05,
        max_cond_size: 2,
        max_parents: Some(1),
        required: Arc::from([LaggedLink {
            source: z_id,
            source_lag: Lag::from_raw(1),
            target: y_id,
            target_lag: Lag::CONTEMPORANEOUS,
        }]),
        ..DiscoveryConstraints::default()
    };
    let engine = PcmciEngine::new().with_constraints(constraints.clone());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(31);
    let raw = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    let true_link = LaggedLink {
        source: x_id,
        source_lag: Lag::from_raw(1),
        target: y_id,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let scored = raw
        .evidence
        .links
        .iter()
        .find(|s| s.link == true_link)
        .unwrap_or_else(|| panic!("X→Y unscored; links={:?}", raw.evidence.links));
    // Scored is not recovered: the dependence itself must be found (partial correlation
    // ~0.9996) while the forced decoy z_{t-1} -> y is not significant.
    assert!(scored.statistic > 0.999 && scored.p_value < 1e-12, "{scored:?}");
    let decoy = raw.evidence.links.iter().find(|s| s.link.source == z_id).expect("decoy scored");
    assert!(decoy.p_value > 0.05, "{decoy:?}");
    let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints);
    let kept = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(
        kept.evidence.links.iter().any(|s| s.link == true_link && s.p_value <= 0.05),
        "alpha keep should recover X→Y after full-family MCI; kept={:?}",
        kept.evidence.links
    );
}

#[test]
fn parallel_matches_serial_link_set() {
    let (data, vars) = var_series();
    let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let serial = ExecutionContext::for_tests(3);
    let mut parallel = ExecutionContext::for_tests(3);
    parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).expect("threads"));
    parallel =
        parallel.with_admission(antecedent_core::execution::AdmissionPool::fixed(4, u64::MAX));
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
fn ci_hot_path_no_scratch_growth() {
    let (data, vars) = var_series();
    let frame =
        LaggedFrame::from_series(&data, &vars, 2, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    // Warm up capacities with the same query shape as the steady-state loop.
    let cond = [(VariableId::from_raw(0), Lag::from_raw(2))];
    let link = LaggedLink {
        source: VariableId::from_raw(0),
        source_lag: Lag::from_raw(1),
        target: VariableId::from_raw(1),
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let _ = engine.ci_statistic(&frame, link, &cond, &mut ws, &ctx).unwrap();
    let col_cap = ws.col_idxs.capacity();
    let z_cap = ws.z_flat.capacity();
    let ci_n = ws.ci.parcorr.capacity_n();
    let ci_p = ws.ci.parcorr.capacity_p();
    let col_ptr = ws.col_idxs.as_ptr();
    let z_ptr = ws.z_flat.as_ptr();
    let design_ptr = ws.ci.parcorr.design.as_ptr();
    let design_cap = ws.ci.parcorr.design.capacity();
    for _ in 0..200 {
        let _ = engine.ci_statistic(&frame, link, &cond, &mut ws, &ctx).unwrap();
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
    use antecedent_stats::OracleCi;

    let (data, vars) = var_series();
    // Local batch always uses x=0,y=1; empty deps ⇒ every pair independent.
    let engine =
        PcmciEngine::new().with_constraints(constraints()).with_ci(Arc::new(OracleCi::new([])));
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let result = engine.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    // Full-family MCI still emits scores; independence ⇒ all p > alpha.
    assert!(
        !result.evidence.links.is_empty(),
        "full-family MCI must score candidates even when PC finds no parents"
    );
    assert!(
        result.evidence.links.iter().all(|s| s.p_value > 0.05),
        "oracle with no deps should mark every pair independent, got {:?}",
        result.evidence.links
    );
    let pcmci = Pcmci::new()
        .with_fdr(false)
        .with_constraints(constraints())
        .with_ci(Arc::new(OracleCi::new([])));
    let filtered = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(
        filtered.evidence.links.is_empty(),
        "alpha keep should drop all independent links, got {:?}",
        filtered.evidence.links
    );

    let engine_dep = PcmciEngine::new()
        .with_constraints(constraints())
        .with_ci(Arc::new(OracleCi::new([(0usize, 1usize)])));
    let kept = engine_dep.run_pc_mci(&data, &vars, &mut ws, &ctx).unwrap();
    assert!(
        kept.evidence.links.iter().any(|s| s.p_value <= 0.05),
        "oracle marking (0,1) dependent should retain significant links"
    );
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
    let dropped = mci_conditioning(link, &parents_target, &parents_source, usize::MAX, &mut out);
    assert_eq!(dropped, 0);
    assert_eq!(out, vec![(y, Lag::from_raw(1)), (x, Lag::from_raw(3))]);
}

/// The budget must bind on the lagged block *before* a caller prepends its own conditions.
///
/// PCMCI+ appends a contemporaneous candidate set `S` to the MCI conditioning set. It used
/// to prepend `S` and then truncate the combined vector — but truncation drops from
/// the tail, which is the lagged MCI block. That block is the fixed control set that makes
/// the MCI test valid under autocorrelation, so the optional candidates were evicting the
/// mandatory ones. Reserving the budget up front keeps that from happening.
#[test]
fn mci_conditioning_reserves_room_for_caller_conditions() {
    let y = VariableId::from_raw(99);
    let link = LaggedLink {
        source: VariableId::from_raw(98),
        source_lag: Lag::from_raw(1),
        target: y,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    // 40 distinct lagged parents of the target: no fixed column cap truncates them.
    let parents_target: Vec<(VariableId, Lag)> =
        (0..40).map(|i| (VariableId::from_raw(i), Lag::from_raw(1))).collect();

    // Unbounded: all 40 conditions are kept and nothing is dropped.
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &[], usize::MAX, &mut out);
    assert_eq!(out.len(), 40);
    assert_eq!(dropped, 0);

    // Reserving 3 slots for `S` out of 30 leaves 27 lagged conditions and accounts for all 13
    // drops, so the caller's 3 conditions fit without evicting anything after the fact.
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &[], 27, &mut out);
    assert_eq!(out.len(), 27);
    assert_eq!(dropped, 13);
    assert_eq!(out[..], parents_target[..27]);

    // A budget above the set size drops nothing.
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &[], 1000, &mut out);
    assert_eq!(out.len(), 40);
    assert_eq!(dropped, 0);
}

/// Deterministic uniform draws on `(-0.5, 0.5)`.
fn uniform_draws(state: &mut u64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// Partial correlation of `x` and `y` given `z`, by residualising both on `[1, z…]` through
/// the normal equations (Gaussian elimination with partial pivoting). Independent of the
/// production kernel.
fn reference_partial_correlation(x: &[f64], y: &[f64], z: &[&[f64]]) -> f64 {
    let n = x.len();
    let centre = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|e| e - mean).collect::<Vec<f64>>()
    };
    let (x, y) = (centre(x), centre(y));
    let z: Vec<Vec<f64>> = z.iter().map(|c| centre(c)).collect();
    let p = z.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(u, v)| u * v).sum::<f64>();
    let residual = |target: &[f64]| {
        // Augmented normal equations [Z'Z | Z'target].
        let mut m = vec![vec![0.0; p + 1]; p];
        for i in 0..p {
            for j in 0..p {
                m[i][j] = dot(&z[i], &z[j]);
            }
            m[i][p] = dot(&z[i], target);
        }
        for col in 0..p {
            let pivot =
                (col..p).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs())).unwrap();
            m.swap(col, pivot);
            for row in (col + 1)..p {
                let factor = m[row][col] / m[col][col];
                for k in col..=p {
                    let v = m[col][k];
                    m[row][k] -= factor * v;
                }
            }
        }
        let mut beta = vec![0.0; p];
        for i in (0..p).rev() {
            let tail: f64 = ((i + 1)..p).map(|j| m[i][j] * beta[j]).sum();
            beta[i] = (m[i][p] - tail) / m[i][i];
        }
        (0..n)
            .map(|t| target[t] - (0..p).map(|j| beta[j] * z[j][t]).sum::<f64>())
            .collect::<Vec<f64>>()
    };
    let (rx, ry) = (residual(&x), residual(&y));
    dot(&rx, &ry) / (dot(&rx, &rx) * dot(&ry, &ry)).sqrt()
}

/// `2 + conditioning` columns on 400 rows: `x`, `y` and `conditioning` independent draws with
/// `x` and `y` both loading on the first few of them.
fn wide_conditioning_series(conditioning: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let n = 400usize;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let z: Vec<Vec<f64>> = (0..conditioning).map(|_| uniform_draws(&mut state, n)).collect();
    let noise_x = uniform_draws(&mut state, n);
    let noise_y = uniform_draws(&mut state, n);
    let x: Vec<f64> = (0..n).map(|t| 0.4 * z[0][t] + 0.3 * z[1][t] + noise_x[t]).collect();
    let y: Vec<f64> =
        (0..n).map(|t| 0.5 * x[t] + 0.2 * z[0][t] + 0.1 * z[2][t] + noise_y[t]).collect();
    let mut b = CausalSchemaBuilder::new();
    let names: Vec<String> = ["x".to_owned(), "y".to_owned()]
        .into_iter()
        .chain((0..conditioning).map(|j| format!("z{j}")))
        .collect();
    for name in &names {
        b.add_variable(
            name.as_str(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols: Vec<OwnedColumn> = [x, y]
        .into_iter()
        .chain(z)
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let vars = (0..names.len()).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    (data, vars)
}

#[test]
fn a_forty_column_conditioning_set_matches_an_independent_partial_correlation() {
    // 42 columns in one CI query: past the old 32-column ceiling.
    let conditioning = 40;
    let (data, vars) = wide_conditioning_series(conditioning);
    let frame =
        LaggedFrame::from_series(&data, &vars, 1, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let x = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let link = LaggedLink {
        source: x,
        source_lag: Lag::CONTEMPORANEOUS,
        target: y,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let cond: Vec<(VariableId, Lag)> = (0..conditioning)
        .map(|j| (VariableId::from_raw(u32::try_from(2 + j).unwrap()), Lag::CONTEMPORANEOUS))
        .collect();
    let (statistic, p_value) = engine
        .ci_statistic(&frame, link, &cond, &mut ws, &ctx)
        .unwrap()
        .expect("the 40-column conditioning set is testable");

    let column = |v: VariableId| frame.column(frame.column_index(v, Lag::CONTEMPORANEOUS).unwrap());
    let given: Vec<&[f64]> = cond.iter().map(|&(v, _)| column(v)).collect();
    let expected = reference_partial_correlation(column(x), column(y), &given);
    assert!(
        (statistic - expected).abs() < 1e-8,
        "engine partial correlation {statistic} vs independent {expected}"
    );
    // x drives y, so the dependence given the 40 others is strong.
    assert!(expected.abs() > 0.3 && p_value < 1e-6, "r={expected}, p={p_value}");

    // The batch path carries the same 40-column set.
    let sets: Vec<&[(VariableId, Lag)]> = vec![&cond, &cond[..39]];
    let batch = engine.ci_batch(&frame, link, &sets, &mut ws, &ctx).expect("batchable");
    assert!((batch[0].statistic - expected).abs() < 1e-8);
}

#[test]
fn a_conditioning_set_over_the_memory_limit_is_refused_not_truncated() {
    let (data, vars) = wide_conditioning_series(40);
    let frame =
        LaggedFrame::from_series(&data, &vars, 1, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let engine = PcmciEngine::new().with_constraints(constraints());
    let mut ws = DiscoveryWorkspace::default();
    let link = LaggedLink {
        source: VariableId::from_raw(0),
        source_lag: Lag::CONTEMPORANEOUS,
        target: VariableId::from_raw(1),
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let cond: Vec<(VariableId, Lag)> =
        (2..42u32).map(|raw| (VariableId::from_raw(raw), Lag::CONTEMPORANEOUS)).collect();
    let mut tight = ExecutionContext::for_tests(1);
    tight.set_memory_limit(Some(10_000));
    let error = engine.ci_statistic(&frame, link, &cond, &mut ws, &tight).unwrap_err();
    assert!(matches!(error, DiscoveryError::Resource(_)), "{error:?}");

    // The capacity is the largest conditioning set the limit admits.
    let rows = frame.column(0).len();
    let limit = 400_000;
    tight.set_memory_limit(Some(limit));
    let capacity = ci_conditioning_capacity(&frame, &tight);
    assert!(ci_query_bytes(rows, capacity) <= limit);
    assert!(ci_query_bytes(rows, capacity + 1) > limit);
    assert_eq!(ci_conditioning_capacity(&frame, &ExecutionContext::for_tests(1)), usize::MAX);
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
    let dropped = mci_conditioning(link, &parents_target, &parents_source, usize::MAX, &mut out);
    assert_eq!(dropped, 0);
    assert_eq!(out, vec![(x, Lag::from_raw(2))]);
}

#[test]
fn pc1_ranks_required_independent_edge_by_statistic_not_infinity() {
    // Y ← X (strong) plus independent noise Z. Force-require Z_{t-1} → Y so it survives
    // PC1 despite independence; it must rank *after* the strong parent, not first via +∞.
    let n = 500usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y", "z"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.017).sin();
        z[t] = ((t as f64) * 0.113).cos();
        y[t] = 0.9 * x[t - 1] + 0.01 * ((t as f64) * 0.03).cos();
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
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let vars = vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let x_id = VariableId::from_raw(0);
    let y_id = VariableId::from_raw(1);
    let z_id = VariableId::from_raw(2);

    let constraints = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
        alpha: 0.05,
        max_cond_size: 2,
        required: Arc::from([LaggedLink {
            source: z_id,
            source_lag: Lag::from_raw(1),
            target: y_id,
            target_lag: Lag::CONTEMPORANEOUS,
        }]),
        ..DiscoveryConstraints::default()
    };
    let engine = PcmciEngine::new().with_constraints(constraints.clone());
    let compiled = constraints.compile(&vars).unwrap();
    let frame =
        LaggedFrame::from_series(&data, &vars, 1, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(21);
    let (parents, _) =
        engine.select_parents(&frame, y_id, &vars, &compiled, &mut ws, &ctx).unwrap();

    let strong = (x_id, Lag::from_raw(1));
    let forced = (z_id, Lag::from_raw(1));
    assert!(parents.contains(&strong), "parents={parents:?}");
    assert!(parents.contains(&forced), "parents={parents:?}");
    let i_strong = parents.iter().position(|p| *p == strong).unwrap();
    let i_forced = parents.iter().position(|p| *p == forced).unwrap();
    assert!(
        i_strong < i_forced,
        "strong parent must rank before required-independent edge: parents={parents:?}"
    );
}

/// Four-variable series `[a, b, x, y]` where only `x_{t-1}` drives `y`; `a`, `b` are unrelated
/// oscillations at incommensurate frequencies.
fn four_var_series() -> (TimeSeriesData, Vec<VariableId>) {
    let n = 500usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["a", "b", "x", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut cols_v = vec![vec![0.0; n]; 4];
    for t in 1..n {
        let tf = t as f64;
        cols_v[0][t] = (tf * 0.113).cos();
        cols_v[1][t] = (tf * 0.291).sin();
        cols_v[2][t] = (tf * 0.017).sin() + 0.3 * (tf * 0.71).cos();
        cols_v[3][t] = 0.9 * cols_v[2][t - 1] + 0.01 * (tf * 0.03).cos();
    }
    let cols: Vec<OwnedColumn> = cols_v
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, (0..4).map(VariableId::from_raw).collect())
}

/// `max_parents` truncates the surviving, strength-ranked parents, not the candidate list.
/// Candidates are enumerated by variable order, so a candidate cut would only ever test
/// `a` and `b` and could never find the true parent `x_{t-1}` of `y`.
#[test]
fn max_parents_caps_survivors_not_candidates() {
    let (data, vars) = four_var_series();
    let y_id = VariableId::from_raw(3);
    let constraints = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
        alpha: 0.05,
        max_cond_size: 2,
        max_parents: Some(2),
        ..DiscoveryConstraints::default()
    };
    let engine = PcmciEngine::new().with_constraints(constraints.clone());
    let compiled = constraints.compile(&vars).unwrap();
    let frame =
        LaggedFrame::from_series(&data, &vars, 1, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(5);
    let (parents, _) =
        engine.select_parents(&frame, y_id, &vars, &compiled, &mut ws, &ctx).unwrap();
    assert!(parents.contains(&(VariableId::from_raw(2), Lag::from_raw(1))), "parents={parents:?}");
    assert!(parents.len() <= 2, "parents={parents:?}");
}

/// A cancelled run must fail with `Cancelled`, never return parents that were only partly
/// pruned (MCI would then condition on the unpruned sets and look like a real result).
#[test]
fn cancelled_parent_selection_is_an_error() {
    let (data, vars) = four_var_series();
    let constraints = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    };
    let engine = PcmciEngine::new().with_constraints(constraints.clone());
    let compiled = constraints.compile(&vars).unwrap();
    let frame =
        LaggedFrame::from_series(&data, &vars, 1, &antecedent_core::KernelPolicy::default_policy())
            .unwrap();
    let ctx = ExecutionContext::for_tests(5);
    ctx.cancellation.cancel();
    let mut ws = DiscoveryWorkspace::default();
    let serial =
        engine.select_parents(&frame, VariableId::from_raw(3), &vars, &compiled, &mut ws, &ctx);
    assert!(matches!(serial, Err(DiscoveryError::Cancelled)), "{serial:?}");
    let all = engine.select_parents_all(&frame, &vars, &compiled, &mut ws, &ctx, 2);
    assert!(matches!(all, Err(DiscoveryError::Cancelled)), "parallel path must not return Ok");
}

/// A lagged target's parents are shifted by the target lag, exactly as the source's are:
/// for `a_t ⊥ b_{t-2}` the conditioning set is `pa(a_t) ∪ pa(b_{t-2})`, not `pa(b_t)`.
#[test]
fn mci_conditioning_shifts_target_parents_by_target_lag() {
    let a = VariableId::from_raw(0);
    let b = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let w = VariableId::from_raw(3);
    let link = LaggedLink {
        source: a,
        source_lag: Lag::CONTEMPORANEOUS,
        target: b,
        target_lag: Lag::from_raw(2),
    };
    let parents_target = [(z, Lag::from_raw(1))];
    let parents_source = [(w, Lag::from_raw(1))];
    let mut out = Vec::new();
    let dropped = mci_conditioning(link, &parents_target, &parents_source, usize::MAX, &mut out);
    assert_eq!(dropped, 0);
    assert_eq!(out, vec![(z, Lag::from_raw(3)), (w, Lag::from_raw(1))]);
}

