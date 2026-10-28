//! Discovery-stability wrappers against an independent tabulation (`oracle_data/discovery_stability_reference.py`).
//!
//! Compiled into the library's test build because the gapped block layout, the lagged-link family
//! and the null-rate standard error are crate-internal. Each wrapper's perturbation is regenerated
//! from its library stream (asserted equal to the frozen record), the discovery engine is run
//! directly on every perturbed series (the engines are reviewed in their own records), and the
//! wrapper's report is compared with a tabulation, verdict and layout derived in exact rational
//! arithmetic from those frozen per-replicate results.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, Lag, RegimeId, VariableId};
use antecedent_data::{
    ColumnView, EnvHoldoutSplit, LaggedFrame, MultiEnvironmentData, TableView, TimeSeriesData,
    ValidityBitmap, surrogate_permute_columns, surrogate_phase_randomize,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, JpcmciPlus, LaggedLink, Pcmci, PcmciPlus,
    RegimeAssignment, Rpcmci, TemporalConstraints, ci_from_name,
};
use antecedent_graph::{Endpoint, NodeRef};
use antecedent_kernels::standard_normal;
use serde_json::{Value, json};

use crate::error::ValidationError;
use crate::stability::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity,
    DiscoveryStabilityReport, EnvironmentHoldout, FalsePositiveCheck, LagWindowSensitivity,
    NullTransform, OrientationStability, RegimeStability, SyntheticNullCalibration,
    gapped_block_bootstrap, lagged_link_family, null_rate_se,
};
use crate::test_support::series_with_missing;

fn reference() -> Value {
    serde_json::from_str(include_str!("../tests/oracle_data/discovery_stability_reference.json"))
        .expect("frozen reference")
}

// ---------------------------------------------------------------------------------------------
// Designs
// ---------------------------------------------------------------------------------------------

fn gauss_stream(seed: u64, stream: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream(stream)
}

fn ids(n: u32) -> Vec<VariableId> {
    (0..n).map(VariableId::from_raw).collect()
}

/// `x` AR(1), `y_t = 0.8 x_{t-1} + 0.6 e`, `w` independent noise.
fn linked3(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = gauss_stream(seed, 1);
    let (mut x, mut y, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        let prev = if t > 0 { x[t - 1] } else { 0.0 };
        x[t] = 0.5 * prev + standard_normal(&mut rng);
        y[t] = 0.8 * prev + 0.6 * standard_normal(&mut rng);
        w[t] = standard_normal(&mut rng);
    }
    series_with_missing(&[x, y, w], &[])
}

/// Contemporaneous collider `x -> z <- y` with independent `x`, `y`.
fn collider3(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = gauss_stream(seed, 2);
    let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        x[t] = standard_normal(&mut rng);
        y[t] = standard_normal(&mut rng);
        z[t] = 0.8 * x[t] + 0.8 * y[t] + 0.4 * standard_normal(&mut rng);
    }
    series_with_missing(&[x, y, z], &[])
}

/// `y` follows `x` with a coefficient that changes sign at the midpoint.
fn two_regime(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = gauss_stream(seed, 3);
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        let prev = if t > 0 { x[t - 1] } else { 0.0 };
        let coef = if t < n / 2 { 0.9 } else { -0.9 };
        x[t] = 0.5 * prev + standard_normal(&mut rng);
        y[t] = coef * prev + 0.5 * standard_normal(&mut rng);
    }
    series_with_missing(&[x, y], &[])
}

fn constraints(min_lag: u32, max_lag: u32, alpha: f64, cond: usize) -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(min_lag),
        },
        max_cond_size: cond,
        alpha,
        ..Default::default()
    }
}

fn pcmci(min_lag: u32, max_lag: u32, alpha: f64) -> Pcmci {
    Pcmci::new().with_fdr(false).with_constraints(constraints(min_lag, max_lag, alpha, 1))
}

fn link_key(l: &LaggedLink) -> Value {
    json!([l.source.raw(), l.source_lag.raw(), l.target.raw(), l.target_lag.raw()])
}

fn links_json(links: &[LaggedLink]) -> Value {
    let mut sorted = links.to_vec();
    sorted.sort();
    Value::Array(sorted.iter().map(link_key).collect())
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn discovered(engine: &Pcmci, data: &TimeSeriesData, vars: &[VariableId]) -> Vec<LaggedLink> {
    let result =
        engine.run(data, vars, &mut DiscoveryWorkspace::default(), &ctx(1)).expect("discovery");
    result.evidence.links.iter().map(|s| s.link).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-14 * want.abs().max(1.0),
        "{what}: got {got} vs reference {want}"
    );
}

fn as_f64(v: &Value) -> f64 {
    if v.is_null() { f64::NAN } else { v.as_f64().unwrap() }
}

fn assert_table(report: &DiscoveryStabilityReport, want: &Value, label: &str) {
    let rows = want["frequencies"].as_array().unwrap();
    assert_eq!(report.replicates as u64, want["replicates"].as_u64().unwrap(), "{label} cells");
    assert_eq!(report.block_size as u64, want["block_size"].as_u64().unwrap(), "{label} block");
    assert_eq!(report.frequencies.len(), rows.len(), "{label} link count");
    for (got, row) in report.frequencies.iter().zip(rows) {
        assert_eq!(link_key(&got.link), row[0], "{label} link order");
        close(got.frequency, as_f64(&row[1]), &format!("{label} frequency"));
    }
}

// ---------------------------------------------------------------------------------------------
// Record builders (library results frozen as the reference's inputs)
// ---------------------------------------------------------------------------------------------

fn layout_inputs() -> Value {
    let mut cases = Vec::new();
    for (n, block, max_lag, seed) in
        [(60_usize, 7_usize, 2_u32, 9_u64), (45, 45, 1, 3), (90, 10, 3, 5)]
    {
        let rows: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let neg: Vec<f64> = rows.iter().map(|v| -v - 1.0).collect();
        let data = series_with_missing(&[rows, neg], &[]);
        let gap = 2 * max_lag as usize;
        let mut rng = gauss_stream(seed, 9);
        let mut scratch = Vec::new();
        let boot = gapped_block_bootstrap(&data, block, gap, &mut rng, &mut scratch).unwrap();
        let len = boot.series.row_count();
        let ColumnView::Float64(c0) = boot.series.column(VariableId::from_raw(0)).unwrap() else {
            panic!("float64")
        };
        let ColumnView::Float64(c1) = boot.series.column(VariableId::from_raw(1)).unwrap() else {
            panic!("float64")
        };
        cases.push(json!({
            "n": n, "block": block, "gap": gap,
            "idx": scratch.iter().map(|g| g.get()).collect::<Vec<_>>(),
            "got_len": len,
            "got_source_rows": boot.source_rows,
            "got_valid0": (0..len).map(|i| c0.validity.is_valid(i)).collect::<Vec<_>>(),
            "got_valid1": (0..len).map(|i| c1.validity.is_valid(i)).collect::<Vec<_>>(),
            "got_values0": c0.values.as_slice(),
            "got_values1": c1.values.as_slice(),
        }));
    }
    Value::Array(cases)
}

struct BlockRun {
    data: TimeSeriesData,
    vars: Vec<VariableId>,
    engine: Pcmci,
    replicates: u32,
    block: usize,
}

fn block_run() -> BlockRun {
    BlockRun {
        data: linked3(240, 21),
        vars: ids(3),
        engine: pcmci(1, 2, 0.05),
        replicates: 8,
        block: 40,
    }
}

fn block_inputs(run: &BlockRun) -> Value {
    let mut rng = ctx(5).rng.stream(crate::streams::STABILITY_PCMCI);
    let mut scratch = Vec::new();
    let gap = 2 * 2;
    let mut reps = Vec::new();
    for _ in 0..run.replicates {
        let boot =
            gapped_block_bootstrap(&run.data, run.block, gap, &mut rng, &mut scratch).unwrap();
        reps.push(links_json(&discovered(&run.engine, &boot.series, &run.vars)));
    }
    json!({ "replicates": run.replicates, "block_size": run.block, "links": reps })
}

/// `x` and `y` independent noise, `y` carries `0.2 x` at the same time step.
fn weak2(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = gauss_stream(seed, 5);
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        x[t] = standard_normal(&mut rng);
        y[t] = 0.2 * x[t] + standard_normal(&mut rng);
    }
    series_with_missing(&[x, y], &[])
}

fn orientation_design(name: &str) -> (TimeSeriesData, OrientationStability, u32) {
    let (data, cond, nv) = match name {
        "collider" => (collider3(240, 31), 2, 3),
        _ => (weak2(240, 33), 1, 2),
    };
    let engine = PcmciPlus::new().with_fdr(false).with_constraints(constraints(0, 1, 0.05, cond));
    (data, OrientationStability { pcmci_plus: engine, replicates: 8, block_size: 40 }, nv)
}

fn orientation_inputs(name: &str) -> Value {
    let (data, stab, nv) = orientation_design(name);
    let engine = stab.pcmci_plus.clone();
    let mut rng = ctx(5).rng.stream(crate::streams::STABILITY_ORIENTATION);
    let mut scratch = Vec::new();
    let mut reps = Vec::new();
    for _ in 0..stab.replicates {
        let boot =
            gapped_block_bootstrap(&data, stab.block_size, 2, &mut rng, &mut scratch).unwrap();
        let result = engine
            .run(&boot.series, &ids(nv), &mut DiscoveryWorkspace::default(), &ctx(1))
            .unwrap();
        let graph = &result.evidence.graph;
        let lag0 = |id: antecedent_graph::DenseNodeId| match graph.nodes()[id.as_usize()] {
            NodeRef::Lagged { variable, lag } if lag.raw() == 0 => Some(variable.raw()),
            _ => None,
        };
        let (mut directed, mut undirected, mut conflict) = (Vec::new(), Vec::new(), false);
        for e in graph.edges() {
            let (Some(a), Some(b)) = (lag0(e.a), lag0(e.b)) else { continue };
            match (e.at_a, e.at_b) {
                (Endpoint::Conflict, Endpoint::Conflict) => conflict = true,
                (Endpoint::Tail, Endpoint::Tail) => undirected.push(json!([a.min(b), a.max(b)])),
                (Endpoint::Tail, Endpoint::Arrow) => directed.push(json!([a, 0, b, 0])),
                (Endpoint::Arrow, Endpoint::Tail) => directed.push(json!([b, 0, a, 0])),
                _ => {}
            }
        }
        directed.sort_by_key(|v| v.to_string());
        undirected.sort_by_key(|v| v.to_string());
        reps.push(json!({ "directed": directed, "undirected": undirected, "conflict": conflict }));
    }
    json!({ "replicates": stab.replicates, "block_size": stab.block_size, "reps": reps })
}

fn regime_setup() -> (TimeSeriesData, Rpcmci, RegimeAssignment, u32, usize) {
    let n = 200;
    let rpcmci = Rpcmci::new()
        .with_pcmci_plus(
            PcmciPlus::new().with_fdr(false).with_constraints(constraints(1, 1, 0.15, 1)),
        )
        .with_min_regime_len(40)
        .with_alternating_iters(0);
    (two_regime(n, 41), rpcmci, antecedent_discovery::two_regime_half_split(n), 5, 25)
}

const REGIME_MIN_LEN: usize = 40;

/// What a labelled-regime search is defined to return: for every regime with enough rows, PCMCI+
/// on the lagged frame restricted to the windows lying wholly inside that regime, each regime on
/// its own fresh workspace.
fn explicit_regime_links(
    rpcmci: &Rpcmci,
    series: &TimeSeriesData,
    vars: &[VariableId],
    labels: &[u32],
) -> Vec<(u32, Vec<LaggedLink>)> {
    let depth = 2 * rpcmci.pcmci_plus.engine().constraints.temporal.max_lag.raw() as usize;
    let n = series.row_count();
    let policy = ctx(1).kernel_policy;
    let frame = LaggedFrame::from_series(series, vars, depth as u32, &policy).unwrap();
    let regimes: BTreeSet<u32> = labels.iter().copied().collect();
    let mut out = Vec::new();
    for r in regimes {
        if labels.iter().filter(|&&l| l == r).count() < REGIME_MIN_LEN {
            continue;
        }
        let keep: Vec<bool> =
            (0..n - depth).map(|i| (0..=depth).all(|l| labels[i + depth - l] == r)).collect();
        let retained = keep.iter().filter(|&&k| k).count();
        if retained < REGIME_MIN_LEN.saturating_sub(depth).max(8) {
            continue;
        }
        let masked = frame.retain_effective(&ValidityBitmap::from_bools(&keep)).unwrap();
        let result = rpcmci
            .pcmci_plus
            .run_on_frame(&masked, vars, &mut DiscoveryWorkspace::default(), &ctx(1))
            .unwrap();
        out.push((r, result.evidence.links.iter().map(|s| s.link).collect()));
    }
    out
}

fn regime_replicates() -> Vec<(TimeSeriesData, Vec<u32>)> {
    let (data, _, labels, replicates, block) = regime_setup();
    let mut rng = ctx(3).rng.stream(crate::streams::STABILITY_REGIME);
    let mut scratch = Vec::new();
    let orig: Vec<u32> = labels.regimes.iter().map(|r| r.raw()).collect();
    (0..replicates)
        .map(|_| {
            let boot = gapped_block_bootstrap(&data, block, 2, &mut rng, &mut scratch).unwrap();
            (boot.series, boot.source_rows.iter().map(|&row| orig[row]).collect())
        })
        .collect()
}

fn regime_inputs() -> Value {
    let (_, rpcmci, labels, replicates, block) = regime_setup();
    let vars = ids(2);
    let orig: Vec<u32> = labels.regimes.iter().map(|r| r.raw()).collect();
    let reps: Vec<Value> = regime_replicates()
        .iter()
        .map(|(series, mapped)| {
            Value::Array(
                explicit_regime_links(&rpcmci, series, &vars, mapped)
                    .iter()
                    .map(|(r, links)| json!([r, links_json(links)]))
                    .collect(),
            )
        })
        .collect();
    json!({
        "replicates": replicates, "block_size": block,
        "regimes": orig.iter().copied().collect::<BTreeSet<_>>(),
        "reps": reps,
    })
}

fn false_positive_setup() -> (TimeSeriesData, Pcmci, u32) {
    (linked3(160, 51), pcmci(1, 2, 0.3), 10)
}

fn false_positive_inputs(transform: NullTransform) -> Value {
    let (data, engine, replicates) = false_positive_setup();
    let mut rng = ctx(6).rng.stream(crate::streams::STABILITY_FALSE_POSITIVE);
    let hits: Vec<u64> = (0..replicates)
        .map(|_| {
            let null = match transform {
                NullTransform::ColumnPermute => surrogate_permute_columns(&data, &mut rng),
                NullTransform::PhaseRandomize => surrogate_phase_randomize(&data, &mut rng),
            }
            .unwrap();
            discovered(&engine, &null, &ids(3)).len() as u64
        })
        .collect();
    json!({ "alpha": 0.3, "variables": 3, "min_lag": 1, "max_lag": 2, "hits": hits })
}

fn null_setup() -> (u32, usize, usize, Pcmci) {
    (12, 100, 2, pcmci(1, 2, 0.3))
}

fn null_inputs() -> Value {
    let (n_sim, n_obs, n_vars, engine) = null_setup();
    let mut rng = ctx(7).rng.stream(crate::streams::STABILITY_NULL);
    let vars = ids(n_vars as u32);
    let hits: Vec<u64> = (0..n_sim)
        .map(|_| {
            let cols: Vec<Vec<f64>> = (0..n_vars)
                .map(|_| (0..n_obs).map(|_| standard_normal(&mut rng)).collect())
                .collect();
            let data = series_with_missing(&cols, &[]);
            discovered(&engine, &data, &vars).len() as u64
        })
        .collect();
    json!({ "alpha": 0.3, "variables": n_vars, "min_lag": 1, "max_lag": 2, "hits": hits, "band_tol": 3.0 })
}

fn env_series(n: usize, seed: u64, ar: f64, coef: f64) -> TimeSeriesData {
    let mut rng = gauss_stream(seed, 4);
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        let prev = if t > 0 { x[t - 1] } else { 0.0 };
        x[t] = ar * prev + standard_normal(&mut rng);
        y[t] = coef * prev + 0.5 * standard_normal(&mut rng);
    }
    series_with_missing(&[x, y], &[])
}

fn env_setup() -> (MultiEnvironmentData, JpcmciPlus) {
    let multi = MultiEnvironmentData::try_new([
        env_series(150, 61, 0.4, 0.9),
        env_series(150, 62, 0.4, 0.9),
        env_series(150, 63, 0.4, 0.0),
        env_series(150, 64, 0.0, 0.0),
        env_series(150, 65, 0.0, 0.0),
    ])
    .unwrap();
    let jp = JpcmciPlus::new().with_fdr(false).with_constraints(constraints(1, 1, 0.1, 1));
    (multi, jp)
}

fn subset(multi: &MultiEnvironmentData, idxs: &[usize]) -> MultiEnvironmentData {
    MultiEnvironmentData::try_new(
        idxs.iter().map(|&i| multi.environment(i).unwrap().clone()).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn jp_links(jp: &JpcmciPlus, data: &MultiEnvironmentData) -> Vec<LaggedLink> {
    let result =
        jp.run(data, &ids(2), &mut DiscoveryWorkspace::default(), &ctx(1)).expect("jpcmci");
    result.evidence.links.iter().map(|s| s.link).collect()
}

fn env_inputs() -> Value {
    let (multi, jp) = env_setup();
    let mut cases = Vec::new();
    for (train, hold) in [
        (vec![0_usize, 1], vec![2_usize]),
        (vec![2], vec![0, 1]),
        (vec![0], vec![1]),
        (vec![3], vec![4]),
        (vec![3], vec![0]),
        (vec![0], vec![3]),
    ] {
        let a: BTreeSet<_> = jp_links(&jp, &subset(&multi, &train)).into_iter().collect();
        let b: BTreeSet<_> = jp_links(&jp, &subset(&multi, &hold)).into_iter().collect();
        cases.push(json!({
            "train_envs": train, "holdout_envs": hold,
            "train": links_json(&a.into_iter().collect::<Vec<_>>()),
            "holdout": links_json(&b.into_iter().collect::<Vec<_>>()),
        }));
    }
    Value::Array(cases)
}

fn grid_setup() -> (TimeSeriesData, Vec<VariableId>) {
    (linked3(200, 71), ids(3))
}

fn grid_inputs() -> Value {
    let (data, vars) = grid_setup();
    let base = pcmci(1, 2, 0.05);
    let cell = |engine: &Pcmci| links_json(&discovered(engine, &data, &vars));
    let alpha_cells: Vec<Value> = [0.001, 0.05, 0.4]
        .iter()
        .map(|&a| {
            let mut c = base.engine().constraints.clone();
            c.alpha = a;
            cell(&base.clone().with_constraints(c))
        })
        .collect();
    let lag_cells: Vec<Value> = [1_u32, 2, 3]
        .iter()
        .map(|&m| {
            let mut c = base.engine().constraints.clone();
            c.temporal.max_lag = Lag::from_raw(m);
            cell(&base.clone().with_constraints(c))
        })
        .collect();
    let ci_cells: Vec<Value> = ["parcorr", "robust_parcorr"]
        .iter()
        .map(|n| cell(&base.clone().with_ci(ci_from_name(n).unwrap())))
        .collect();
    json!({ "alpha": alpha_cells, "lag": lag_cells, "ci": ci_cells })
}

fn arithmetic_inputs() -> Value {
    json!({
        "hits_cases": [
            {"alpha": 0.05, "family": 9, "hits": [0, 0, 0, 0, 0, 0, 0, 0]},
            {"alpha": 0.05, "family": 9, "hits": [0, 9, 0, 9, 0, 9, 0, 9, 0, 9]},
            {"alpha": 0.1, "family": 4, "hits": [1, 0, 2, 0, 1, 3, 0, 1, 0, 0, 2, 1]},
            {"alpha": 0.5, "family": 6, "hits": [3]},
            {"alpha": 1.0, "family": 6, "hits": [6, 6, 6]},
            {"alpha": 0.01, "family": 18, "hits": [0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5]},
        ],
        "family_cases": [[3, 1, 2], [2, 2, 4], [5, 1, 1], [1, 3, 3], [4, 2, 7]],
    })
}

// ---------------------------------------------------------------------------------------------
// Full input record (used to (re)generate the fixture and to assert it is current)
// ---------------------------------------------------------------------------------------------

fn all_inputs() -> Value {
    let run = block_run();
    json!({
        "layout": layout_inputs(),
        "block": block_inputs(&run),
        "orientation": {
            "collider": orientation_inputs("collider"),
            "weak": orientation_inputs("weak"),
        },
        "regime": regime_inputs(),
        "false_positive": {
            "permute": false_positive_inputs(NullTransform::ColumnPermute),
            "phase": false_positive_inputs(NullTransform::PhaseRandomize),
        },
        "null": null_inputs(),
        "env": env_inputs(),
        "grid": grid_inputs(),
        "arithmetic": arithmetic_inputs(),
    })
}

#[test]
#[ignore = "writes the frozen inputs for the reference script; run by hand to refresh"]
fn discovery_stability_reference_dump_inputs() {
    let path = std::env::var("DISCOVERY_STABILITY_DUMP").expect("output path");
    std::fs::write(path, serde_json::to_string(&json!({ "inputs": all_inputs() })).unwrap())
        .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Defects
// ---------------------------------------------------------------------------------------------

#[test]
fn discovery_stability_synthetic_null_runs_discovery_at_the_declared_alpha() {
    // The calibration declares the type-I level it checks (`alpha`) and reports it as the level
    // used for discovery, so a PCMCI configured with a different level must still run at the
    // declared one: otherwise the empirical rate of one level is judged against another.
    let report_for = |engine_alpha: f64| {
        SyntheticNullCalibration::new(pcmci(1, 2, engine_alpha), 0.3, 12, 100, 2)
            .run(&mut DiscoveryWorkspace::default(), &ctx(7))
            .unwrap()
    };
    let (matched, mismatched) = (report_for(0.3), report_for(0.001));
    assert!((matched.empirical_fpr - mismatched.empirical_fpr).abs() < 1e-15);
    assert!((matched.se - mismatched.se).abs() < 1e-15);
    assert_eq!(matched.within_band, mismatched.within_band);
    assert!(matched.empirical_fpr > 0.0, "a 0.3-level test on noise must reject sometimes");
}

#[test]
fn discovery_stability_environment_holdout_refuses_overlapping_environments() {
    // A holdout that shares an environment with the discovery set is not held out.
    let (multi, jp) = env_setup();
    let overlapping = EnvHoldoutSplit {
        discovery_envs: Arc::from([0_usize, 1]),
        estimation_envs: Arc::from([1_usize, 2]),
    };
    let err = EnvironmentHoldout::new(jp.clone(), overlapping)
        .run(&multi, &ids(2), &mut DiscoveryWorkspace::default(), &ctx(1))
        .unwrap_err();
    assert!(matches!(err, ValidationError::NotApplicable { .. }), "{err:?}");
    let disjoint = EnvHoldoutSplit::try_prefix(3, 2).unwrap();
    EnvironmentHoldout::new(jp, disjoint)
        .run(&multi, &ids(2), &mut DiscoveryWorkspace::default(), &ctx(1))
        .unwrap();
}

// ---------------------------------------------------------------------------------------------
// Reference comparisons
// ---------------------------------------------------------------------------------------------

fn frozen(area: &[&str]) -> (Value, Value) {
    let r = reference();
    let (mut inp, mut exp) = (&r["inputs"], &r["expected"]);
    for key in area {
        inp = &inp[*key];
        exp = &exp[*key];
    }
    (inp.clone(), exp.clone())
}

fn run_ws() -> DiscoveryWorkspace {
    DiscoveryWorkspace::default()
}

#[test]
fn discovery_stability_gapped_block_layout_matches_reference() {
    let (inputs, expected) = frozen(&["layout"]);
    // The library's block draws and layout are regenerated and equal the frozen record.
    assert_eq!(layout_inputs(), inputs, "regenerated perturbation differs from the frozen one");
    for (got, want) in inputs.as_array().unwrap().iter().zip(expected.as_array().unwrap()) {
        assert!(want["junctions"].as_u64().unwrap() > 0 || got["block"] == got["n"]);
        assert_eq!(got["got_len"], want["len"], "gapped length");
        assert_eq!(got["got_source_rows"], want["source_rows"], "source rows");
        assert_eq!(got["got_valid0"], want["valid"], "validity of column 0");
        assert_eq!(got["got_valid1"], want["valid"], "validity of column 1");
        // A separator row is missing; whatever storage fills it with is not a value.
        for (key, want_key) in [("got_values0", "values0"), ("got_values1", "values1")] {
            let (g, w, v) = (
                got[key].as_array().unwrap(),
                want[want_key].as_array().unwrap(),
                want["valid"].as_array().unwrap(),
            );
            assert_eq!(g.len(), w.len(), "{key} length");
            for i in 0..g.len() {
                if v[i].as_bool().unwrap() {
                    assert_eq!(g[i], w[i], "{key} at {i}");
                }
            }
        }
    }
}

#[test]
fn discovery_stability_block_bootstrap_matches_reference() {
    let (inputs, expected) = frozen(&["block"]);
    let run = block_run();
    assert_eq!(block_inputs(&run), inputs, "regenerated replicate discoveries differ");
    let stab = BlockBootstrapStability {
        pcmci: run.engine.clone(),
        replicates: run.replicates,
        block_size: run.block,
    };
    let report = stab.run(&run.data, &run.vars, &mut run_ws(), &ctx(5)).unwrap();
    assert_table(&report, &expected, "block bootstrap");
}

#[test]
fn discovery_stability_orientation_matches_reference() {
    for name in ["collider", "weak"] {
        let (inputs, expected) = frozen(&["orientation", name]);
        assert_eq!(orientation_inputs(name), inputs, "{name}: regenerated replicates differ");
        let (data, stab, nv) = orientation_design(name);
        let report = stab.run(&data, &ids(nv), &mut run_ws(), &ctx(5)).unwrap();
        assert_eq!(report.replicates, stab.replicates);
        assert_eq!(report.block_size, stab.block_size);
        let directed = DiscoveryStabilityReport {
            frequencies: report.directed.clone(),
            replicates: report.replicates,
            block_size: report.block_size,
        };
        assert_table(&directed, &expected["directed"], name);
        let want = expected["undirected"].as_array().unwrap();
        assert_eq!(report.undirected.len(), want.len(), "{name} undirected count");
        for (got, row) in report.undirected.iter().zip(want) {
            assert_eq!(json!([got.a.raw(), got.b.raw()]), row[0], "{name} undirected order");
            close(got.frequency, as_f64(&row[1]), &format!("{name} undirected frequency"));
        }
        close(
            report.conflict_rate,
            as_f64(&expected["conflict_rate"]),
            &format!("{name} conflict"),
        );
    }
}

#[test]
fn discovery_stability_regime_matches_reference() {
    let (inputs, expected) = frozen(&["regime"]);
    assert_eq!(regime_inputs(), inputs, "regenerated regime replicates differ");
    let (data, rpcmci, labels, replicates, block) = regime_setup();
    let mut stab = RegimeStability::new(rpcmci, labels);
    stab.replicates = replicates;
    stab.block_size = block;
    let report = stab.run(&data, &ids(2), &mut run_ws(), &ctx(3)).unwrap();
    let want = expected.as_object().unwrap();
    assert_eq!(report.per_regime.len(), want.len());
    for (regime, table) in &report.per_regime {
        assert_table(table, &want[&regime.raw().to_string()], &format!("regime {}", regime.raw()));
    }
}

#[test]
fn discovery_stability_false_positive_check_matches_reference() {
    for (name, transform) in
        [("permute", NullTransform::ColumnPermute), ("phase", NullTransform::PhaseRandomize)]
    {
        let (inputs, expected) = frozen(&["false_positive", name]);
        assert_eq!(
            false_positive_inputs(transform),
            inputs,
            "{name}: regenerated surrogates differ"
        );
        let (data, engine, replicates) = false_positive_setup();
        let report = FalsePositiveCheck::new(engine, transform, replicates)
            .run(&data, &ids(3), &mut run_ws(), &ctx(6))
            .unwrap();
        close(report.mean_edge_count, as_f64(&expected["mean_edge_count"]), name);
        close(report.empirical_fpr, as_f64(&expected["empirical_fpr"]), name);
        close(report.se, as_f64(&expected["se"]), name);
        assert_eq!(report.passed, expected["passed"].as_bool().unwrap(), "{name} verdict");
    }
    // Both verdicts are exercised.
    let r = reference();
    assert_eq!(r["expected"]["false_positive"]["permute"]["passed"], true);
    assert_eq!(r["expected"]["false_positive"]["phase"]["passed"], false);
}

#[test]
fn discovery_stability_synthetic_null_matches_reference() {
    let (inputs, expected) = frozen(&["null"]);
    assert_eq!(null_inputs(), inputs, "regenerated noise discoveries differ");
    let (n_sim, n_obs, n_vars, engine) = null_setup();
    let report = SyntheticNullCalibration::new(engine, 0.3, n_sim, n_obs, n_vars)
        .run(&mut run_ws(), &ctx(7))
        .unwrap();
    close(report.empirical_fpr, as_f64(&expected["empirical_fpr"]), "null fpr");
    close(report.se, as_f64(&expected["se"]), "null se");
    assert_eq!(report.within_band, expected["within_band"].as_bool().unwrap());
}

#[test]
fn discovery_stability_environment_holdout_matches_reference() {
    let (inputs, expected) = frozen(&["env"]);
    assert_eq!(env_inputs(), inputs, "regenerated environment discoveries differ");
    let (multi, jp) = env_setup();
    for (case, want) in inputs.as_array().unwrap().iter().zip(expected.as_array().unwrap()) {
        let envs = |key: &str| -> Arc<[usize]> {
            case[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
        };
        let split = EnvHoldoutSplit {
            discovery_envs: envs("train_envs"),
            estimation_envs: envs("holdout_envs"),
        };
        let report = EnvironmentHoldout::new(jp.clone(), split)
            .run(&multi, &ids(2), &mut run_ws(), &ctx(1))
            .unwrap();
        assert_eq!(links_json(&report.discovery_links), case["train"]);
        assert_eq!(links_json(&report.holdout_links), case["holdout"]);
        for (got, key) in [(report.shared_frequency, "shared"), (report.jaccard, "jaccard")] {
            if want[key].is_null() {
                assert!(got.is_nan(), "{key} must be undefined for {case}");
            } else {
                close(got, as_f64(&want[key]), key);
            }
        }
    }
}

#[test]
fn discovery_stability_parameter_grids_match_reference() {
    let (inputs, expected) = frozen(&["grid"]);
    assert_eq!(grid_inputs(), inputs, "regenerated grid cells differ");
    let (data, vars) = grid_setup();
    let base = pcmci(1, 2, 0.05);
    let alpha = AlphaThresholdSensitivity::new(base.clone(), Arc::from([0.001_f64, 0.05, 0.4]))
        .run(&data, &vars, &mut run_ws(), &ctx(1))
        .unwrap();
    assert_table(&alpha, &expected["alpha"], "alpha grid");
    let lag = LagWindowSensitivity::new(base.clone(), Arc::from([1_u32, 2, 3]))
        .run(&data, &vars, &mut run_ws(), &ctx(1))
        .unwrap();
    assert_table(&lag, &expected["lag"], "lag grid");
    let names: Arc<[Arc<str>]> =
        Arc::from([Arc::<str>::from("parcorr"), Arc::<str>::from("robust_parcorr")]);
    let ci = CiTestSensitivity::new(base, names).run(&data, &vars, &mut run_ws(), &ctx(1)).unwrap();
    assert_table(&ci, &expected["ci"], "ci grid");
}

#[test]
fn discovery_stability_null_rate_arithmetic_matches_reference() {
    let (inputs, expected) = frozen(&["arithmetic"]);
    assert_eq!(arithmetic_inputs(), inputs);
    for (case, want) in
        inputs["hits_cases"].as_array().unwrap().iter().zip(expected["se"].as_array().unwrap())
    {
        let hits: Vec<u64> =
            case["hits"].as_array().unwrap().iter().map(|h| h.as_u64().unwrap()).collect();
        let got = null_rate_se(
            case["alpha"].as_f64().unwrap(),
            case["family"].as_u64().unwrap() as usize,
            &hits,
        );
        close(got, want.as_f64().unwrap(), &format!("null_rate_se {case}"));
    }
    for (case, want) in inputs["family_cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["family"].as_array().unwrap())
    {
        let c: Vec<u64> = case.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let got = lagged_link_family(c[0] as usize, c[1] as u32, c[2] as u32);
        match want.as_u64() {
            Some(w) => assert_eq!(got.unwrap() as u64, w, "family {case}"),
            None => assert!(got.is_err()),
        }
    }
    // Refusals the reference defines as "no family".
    assert!(lagged_link_family(3, 0, 2).is_err());
    assert!(lagged_link_family(3, 3, 2).is_err());
}

#[test]
fn discovery_stability_reused_workspace_does_not_replay_validity_masks() {
    // Each regime of a labelled search is PCMCI+ on the windows wholly inside that regime. A
    // workspace reused from one replicate to the next sees frames of the same shape whose
    // validity differs, so a mask cached for one replicate's frame must never be replayed on the
    // next one's.
    let mut shared = DiscoveryWorkspace::default();
    let (_, rpcmci, _, _, _) = regime_setup();
    let vars = ids(2);
    for (series, mapped) in regime_replicates() {
        let assignment = RegimeAssignment::try_new(
            mapped.iter().map(|&r| RegimeId::from_raw(r)).collect::<Vec<_>>(),
        )
        .unwrap();
        let result = rpcmci.run(&series, &vars, &assignment, &mut shared, &ctx(1));
        let Ok(result) = result else { continue };
        let want = explicit_regime_links(&rpcmci, &series, &vars, &mapped);
        assert_eq!(result.graphs.graphs.len(), want.len());
        for (idx, &(regime, _)) in result.graphs.graphs.iter().enumerate() {
            let got: Vec<LaggedLink> =
                result.per_regime[idx].evidence.links.iter().map(|s| s.link).collect();
            assert_eq!(links_json(&got), links_json(&want[idx].1), "regime {}", regime.raw());
        }
    }
}
