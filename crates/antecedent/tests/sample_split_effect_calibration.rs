//! Whole-method coverage of the true causal effect after discovery on a row split.
//!
//! The procedure under test is the facade's one construction
//! (`antecedent::analysis::sample_split_effect`): a seeded half split, PC with the
//! partial-correlation test on the discovery half, acceptance of a single fully directed
//! graph from a finished search, and the least-squares effect of the treatment on the outcome
//! adjusted for the treatment's parents on the analysis half. A replicate scores the 95%
//! interval against the true structural coefficient of the treatment on the outcome.
//!
//! # Gated design (`sample_split_discovered_effect_nominal_95_coverage`)
//!
//! Linear-Gaussian DAGs of five, six or seven variables (replicate `k` uses `5 + k % 3`),
//! each with a fully orientable equivalence class built from colliders:
//! `Z1 -> X <- W`, `X -> Y <- Z3`, `W -> Y` (and `Z2 -> X`, `Z4 -> Y` at six and seven
//! variables). Edge weights have magnitude in `[0.6, 1.0]` with random signs, unit-variance
//! noise, and the weights are redrawn until the `W -> Y` edge does not nearly cancel the
//! `W -> X -> Y` path (`|b_wy + b_wx b_xy| >= 0.4`), so faithfulness holds with margin.
//! Rows are `grid_n(400)`; each half has half of them.
//!
//! A replicate that reports no interval (an unresolved structure, a conflict) is not scored
//! as covered or missed: coverage is the coverage *of reported intervals*, and the share of
//! replicates that report one is asserted to be at least 0.9 and printed.
//!
//! # Named boundary designs
//!
//! Each runs the same facade procedure on a family outside the gated one, and is keyed under
//! its own label so it can never govern the gated record:
//!
//! * `weak_edges`: weights in `[0.05, 0.25]`; discovery misses edges.
//! * `cancelling_edge`: `b_wy = -b_wx b_xy`, an unfaithful distribution.
//! * `misspecified_test`: the confounder acts through `w^2 - 1`, which the partial
//!   correlation test cannot see.
//! * `full_row_reuse`: the gated family, but discovery and estimation share all rows (the
//!   post-selection construction), keyed under its own structure value.
//!
//! The harness also prints a recovery table (`info recovery`): the share of replicates whose
//! discovered graph equals the true graph. It is a measured property of the method over the
//! family and is never a license.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::single_match_else,
    reason = "a reported and an unreported replicate are scored differently"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use std::collections::BTreeSet;

use antecedent::analysis::sample_split_analysis::{SplitRule, discover_on_split};
use antecedent::analysis::sample_split_effect::{
    ALPHA, CI_TEST, DISCOVERY_FRACTION, LEVEL, estimate_effect_on_split,
};
use antecedent::analysis::selective_discovery::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
};
use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{DiscoveredStructure, Mark};
use antecedent_io::measured_inference::split_effect::{
    FUNCTIONAL, REUSE_DEPENDENCE, REUSE_STRUCTURE, effect_on_structure,
};
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, map_replicates, n_sim,
    stream_seed, uniform,
};

const BASE_ROWS: usize = 400;
const MIN_REPORTED_SHARE: f64 = 0.9;
const INTERVAL: &str = "analytic_se";

/// Edge weights of one replicate's DAG.
#[derive(Clone, Copy, Debug)]
struct Edges {
    z1x: f64,
    z2x: f64,
    wx: f64,
    xy: f64,
    wy: f64,
    z3y: f64,
    z4y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Licensed,
    WeakEdges,
    CancellingEdge,
    MisspecifiedTest,
}

/// Column positions of one replicate's table.
#[derive(Clone, Copy, Debug)]
struct Layout {
    variables: usize,
    w: usize,
    x: usize,
    y: usize,
}

fn layout(variables: usize) -> Layout {
    let w = 1 + usize::from(variables >= 6);
    Layout { variables, w, x: w + 1, y: w + 2 }
}

fn variable_count(rep: u64) -> usize {
    5 + (rep % 3) as usize
}

fn draw_edges(family: Family, seed: u64) -> Edges {
    let (low, high) = match family {
        Family::WeakEdges => (0.05, 0.25),
        _ => (0.6, 1.0),
    };
    let mut next = uniform(seed, 0x57_17);
    let mut weight = move || {
        let magnitude = low + (high - low) * next();
        let sign = if next() < 0.5 { -1.0 } else { 1.0 };
        sign * magnitude
    };
    loop {
        let mut edges = Edges {
            z1x: weight(),
            z2x: weight(),
            wx: weight(),
            xy: weight(),
            wy: weight(),
            z3y: weight(),
            z4y: weight(),
        };
        match family {
            Family::CancellingEdge => {
                edges.wy = -edges.wx * edges.xy;
                return edges;
            }
            Family::Licensed if (edges.wy + edges.wx * edges.xy).abs() < 0.4 => {}
            _ => return edges,
        }
    }
}

/// The table of one replicate, columns `z1 [z2] w x y z3 [z4]`.
fn family_data(family: Family, rep: u64, n: usize) -> (TabularData, Edges, Layout) {
    let seed = stream_seed(0x5911_7EFF, rep);
    let edges = draw_edges(family, seed);
    let shape = layout(variable_count(rep));
    let mut noise = gaussian(stream_seed(seed, 1));
    let has_z2 = shape.variables >= 6;
    let has_z4 = shape.variables == 7;
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n); shape.variables];
    for _ in 0..n {
        let z1 = noise();
        let z2 = if has_z2 { noise() } else { 0.0 };
        let w = noise();
        let z3 = noise();
        let z4 = if has_z4 { noise() } else { 0.0 };
        // The confounder acts through a symmetric transform in the misspecified family.
        let confounder = if family == Family::MisspecifiedTest { w * w - 1.0 } else { w };
        let x = edges.z1x * z1 + edges.z2x * z2 + edges.wx * confounder + noise();
        let y = edges.xy * x
            + edges.wy * confounder
            + edges.z3y * z3
            + if has_z4 { edges.z4y * z4 } else { 0.0 }
            + noise();
        let mut at = 0;
        let mut push = |value: f64| {
            columns[at].push(value);
            at += 1;
        };
        push(z1);
        if has_z2 {
            push(z2);
        }
        push(w);
        push(x);
        push(y);
        push(z3);
        if has_z4 {
            push(z4);
        }
    }
    let mut names = vec!["z1"];
    if has_z2 {
        names.push("z2");
    }
    names.extend(["w", "x", "y", "z3"]);
    if has_z4 {
        names.push("z4");
    }
    let data =
        TabularData::from_f64_columns(names.iter().copied().zip(columns.iter().map(Vec::as_slice)))
            .expect("table");
    (data, edges, shape)
}

fn licensed_family_data(rep: u64, n: usize) -> (TabularData, Edges, Layout) {
    family_data(Family::Licensed, rep, n)
}
fn weak_edge_family_data(rep: u64, n: usize) -> (TabularData, Edges, Layout) {
    family_data(Family::WeakEdges, rep, n)
}
fn cancelling_edge_family_data(rep: u64, n: usize) -> (TabularData, Edges, Layout) {
    family_data(Family::CancellingEdge, rep, n)
}
fn misspecified_test_family_data(rep: u64, n: usize) -> (TabularData, Edges, Layout) {
    family_data(Family::MisspecifiedTest, rep, n)
}

/// True directed edges as `(source, sink)` column positions.
fn true_edges(shape: Layout) -> BTreeSet<(u32, u32)> {
    let has_z2 = shape.variables >= 6;
    let has_z4 = shape.variables == 7;
    let at = |i: usize| u32::try_from(i).expect("small index");
    let (w, x, y) = (at(shape.w), at(shape.x), at(shape.y));
    let z3 = y + 1;
    let mut edges = BTreeSet::from([(0, x), (w, x), (w, y), (x, y), (z3, y)]);
    if has_z2 {
        edges.insert((1, x));
    }
    if has_z4 {
        edges.insert((z3 + 1, y));
    }
    edges
}

fn directed_pairs(structure: &DiscoveredStructure) -> BTreeSet<(u32, u32)> {
    structure
        .edges()
        .iter()
        .filter_map(|edge| match (edge.at_a, edge.at_b) {
            (Mark::Tail, Mark::Arrow) => Some((edge.a.variable, edge.b.variable)),
            (Mark::Arrow, Mark::Tail) => Some((edge.b.variable, edge.a.variable)),
            _ => None,
        })
        .collect()
}

fn contract(shape: Layout, seed: u64) -> StaticDiscoveryContract {
    let variables: Vec<VariableId> =
        (0..shape.variables).map(|i| VariableId::from_raw(i as u32)).collect();
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        variables,
        CiTestSpec::named(CI_TEST),
        seed,
    );
    contract.fdr = None;
    contract.constraints.alpha = ALPHA;
    contract
}

fn bound(basis: &CalibrationBasis, functional: &str) -> (Construction, ScopeFacts) {
    (
        Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: functional.to_owned(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    )
}

/// One replicate's report: the interval, the truth, its key and whether discovery recovered
/// the true graph. `None` when the procedure reported no interval.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: (Construction, ScopeFacts),
    recovered: bool,
}

fn recovered_on_split(
    data: &TabularData,
    shape: Layout,
    edges: &BTreeSet<(u32, u32)>,
    rep: u64,
) -> bool {
    let seed = stream_seed(0x000A_11CE, rep);
    let ctx = ExecutionContext::for_tests(seed);
    let rule = SplitRule::SeededPermutation {
        seed: stream_seed(seed, 2),
        discovery_fraction: DISCOVERY_FRACTION,
    };
    discover_on_split(&contract(shape, seed), data, rule, &ctx).is_ok_and(|split| {
        let structure = &split.outcome().structure;
        structure.is_fully_directed() && directed_pairs(structure) == *edges
    })
}

/// The facade's split construction on one replicate of `family`.
fn split_replicate(
    draw: fn(u64, usize) -> (TabularData, Edges, Layout),
    suffix: Option<&str>,
    rep: u64,
) -> Option<Scored> {
    let (data, edges, shape) = draw(rep, grid_n(BASE_ROWS));
    let seed = stream_seed(0x000A_11CE, rep);
    let ctx = ExecutionContext::for_tests(seed);
    let split_seed = stream_seed(seed, 2);
    let estimate = estimate_effect_on_split(
        &data,
        VariableId::from_raw(shape.x as u32),
        VariableId::from_raw(shape.y as u32),
        split_seed,
        &ctx,
    )
    .ok()?;
    let functional = suffix.map_or_else(
        || FUNCTIONAL.to_owned(),
        |label| format!("{FUNCTIONAL}.named_boundary:{label}"),
    );
    let recovered = recovered_on_split(&data, shape, &true_edges(shape), rep);
    Some(Scored {
        interval: estimate.interval(),
        truth: edges.xy,
        construction: bound(&estimate.keyed_basis(), &functional),
        recovered,
    })
}

/// Discovery and estimation on the same rows (the post-selection construction).
fn reuse_replicate(rep: u64) -> Option<Scored> {
    let (data, edges, shape) = licensed_family_data(rep, grid_n(BASE_ROWS));
    let seed = stream_seed(0x000A_11CE, rep);
    let ctx = ExecutionContext::for_tests(seed);
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&contract(shape, seed), &data, &ctx).ok()?;
    let outcome = adapter.outcome()?;
    if outcome.search_incomplete || outcome.orientation_conflict {
        return None;
    }
    let effect = effect_on_structure(
        &outcome.structure,
        &data,
        shape.x as u32,
        shape.y as u32,
        shape.variables,
        LEVEL,
    )
    .ok()?;
    let basis = effect.keyed_basis(REUSE_STRUCTURE, REUSE_DEPENDENCE);
    Some(Scored {
        interval: (effect.lower, effect.upper),
        truth: edges.xy,
        construction: bound(&basis, FUNCTIONAL),
        recovered: outcome.structure.is_fully_directed()
            && directed_pairs(&outcome.structure) == true_edges(shape),
    })
}

fn score(
    key: RecordKey,
    runs: &[Option<Scored>],
    gated: bool,
) -> (CoverageTally, CoverageTally, CoverageTally) {
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::new(format!("{} reported", key.test), LEVEL);
    let mut recovery = CoverageTally::new(format!("{} recovered", key.test), LEVEL);
    for run in runs {
        match run {
            Some(scored) => {
                reported.record(Some((0.0, 1.0)), 0.5);
                recovery.record(scored.recovered.then_some((0.0, 1.0)), 0.5);
                tally.bind(&scored.construction.0, scored.construction.1);
                tally.record(Some(scored.interval), scored.truth);
            }
            None => {
                reported.record(None, 0.5);
                recovery.record(None, 0.5);
            }
        }
    }
    reported.persist();
    recovery.persist();
    let share = f64::from(reported.covered()) / f64::from(reported.attempts().max(1));
    let recovered = f64::from(recovery.covered()) / f64::from(recovery.attempts().max(1));
    eprintln!(
        "info recovery {}: reported_share={share:.3} recovered_share={recovered:.3} ({} of {} replicates reported)",
        key.test,
        reported.covered(),
        reported.attempts()
    );
    if gated {
        assert!(
            share >= MIN_REPORTED_SHARE,
            "{}: only {share:.3} of replicates reported an interval (floor {MIN_REPORTED_SHARE})",
            key.test
        );
    }
    (tally, reported, recovery)
}

fn named_boundary(test: &'static str, dgp: &'static str, runs: &[Option<Scored>]) {
    let key = RecordKey { test, dgp, interval: INTERVAL };
    let (tally, ..) = score(key, runs, false);
    tally.emit_named_boundary();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sample_split_discovered_effect_nominal_95_coverage() {
    const TEST: &str = "sample_split_discovered_effect_nominal_95_coverage";
    let runs = map_replicates(n_sim(), |rep| split_replicate(licensed_family_data, None, rep));
    let key = RecordKey { test: TEST, dgp: "licensed_family_data", interval: INTERVAL };
    let (tally, ..) = score(key, &runs, true);
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sample_split_weak_edges_boundary_95() {
    let runs = map_replicates(n_sim(), |rep| {
        split_replicate(weak_edge_family_data, Some("weak_edges"), rep)
    });
    named_boundary("sample_split_weak_edges_boundary_95", "weak_edge_family_data", &runs);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sample_split_cancelling_edge_boundary_95() {
    let runs = map_replicates(n_sim(), |rep| {
        split_replicate(cancelling_edge_family_data, Some("cancelling_edge"), rep)
    });
    named_boundary(
        "sample_split_cancelling_edge_boundary_95",
        "cancelling_edge_family_data",
        &runs,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sample_split_misspecified_test_boundary_95() {
    let runs = map_replicates(n_sim(), |rep| {
        split_replicate(misspecified_test_family_data, Some("misspecified_test"), rep)
    });
    named_boundary(
        "sample_split_misspecified_test_boundary_95",
        "misspecified_test_family_data",
        &runs,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sample_split_full_row_reuse_boundary_95() {
    let runs = map_replicates(n_sim(), reuse_replicate);
    named_boundary("sample_split_full_row_reuse_boundary_95", "licensed_family_data", &runs);
}
