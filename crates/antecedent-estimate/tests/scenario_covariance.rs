//! 2.3A A1 (X2): shared-data covariance of scenario estimates over one row sample.
//!
//! The units are six complete rows `(z, x, y)` of a tiny discrete SCM sample. Every
//! scenario estimator is a plug-in functional of the resampled rows' cell proportions.
//! The oracles never use the library's enumerator: the linear pair has the closed-form
//! covariance of the empirical proportions under multinomial resampling,
//! `Cov(a, b) = (E[ab] - E[a]E[b]) / n`, and the nonlinear adjustment pair is enumerated by
//! walking all `n^n` equally likely row-index tuples.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "tiny fixtures with exact small integers"
)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::scenario_covariance::{
    CovarianceMethod, ExactEnumerationOptions, MAX_REPLICATES, MAX_SCENARIOS, RowDependence,
    ScenarioCovariance, ScenarioRowEstimator, SharedRowBootstrapOptions,
    exact_enumeration_covariance, shared_row_bootstrap_covariance,
};

type Row = (u8, u8, u8);

/// `(z, x, y)` of the six units.
const ROWS: [Row; 6] = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0)];
const N: usize = 6;
const SNAPSHOT: &str = "snapshot:rows-v1";

fn units(prefix: &str) -> Vec<Arc<str>> {
    (0..N)
        .map(|i| {
            let id: Arc<str> = Arc::from(format!("{prefix}{i}"));
            id
        })
        .collect()
}

fn score_a(r: Row) -> f64 {
    match (r.1, r.2) {
        (1, 1) => 1.0,
        (0, 1) => -1.0,
        _ => 0.0,
    }
}

fn score_b(r: Row) -> f64 {
    match (r.1, r.2) {
        (1, 1) => 1.0,
        (1, 0) => -1.0,
        _ => 0.0,
    }
}

fn score_c(r: Row) -> f64 {
    if (r.0, r.2) == (1, 1) { 1.0 } else { 0.0 }
}

fn plug_in(counts: &[u32], score: fn(Row) -> f64) -> f64 {
    ROWS.iter().zip(counts).map(|(r, &c)| f64::from(c) * score(*r)).sum::<f64>() / N as f64
}

fn linear(id: &str, snapshot: &str, unit_ids: Vec<Arc<str>>, score: fn(Row) -> f64) -> ScenarioRowEstimator {
    ScenarioRowEstimator::new(id, snapshot, unit_ids, move |c| Ok(plug_in(c, score)))
}

fn linear_by_name(name: &str) -> ScenarioRowEstimator {
    let score: fn(Row) -> f64 = match name {
        "a" => score_a,
        "b" => score_b,
        _ => score_c,
    };
    linear(name, SNAPSHOT, units("u"), score)
}

/// Closed form `Cov(a, b)` of the empirical-proportion functionals under multinomial
/// whole-row resampling.
fn closed_form(sa: fn(Row) -> f64, sb: fn(Row) -> f64) -> f64 {
    let n = N as f64;
    let a: Vec<f64> = ROWS.iter().copied().map(sa).collect();
    let b: Vec<f64> = ROWS.iter().copied().map(sb).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let ab: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    (mean(&ab) - mean(&a) * mean(&b)) / n
}

fn exact_options() -> ExactEnumerationOptions {
    ExactEnumerationOptions { max_compositions: 10_000, max_failure_mass: 0.0 }
}

fn boot_options(seed: u64) -> SharedRowBootstrapOptions {
    SharedRowBootstrapOptions { replicates: 2000, seed, max_failure_fraction: 0.0 }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn assert_refusal(
    result: Result<ScenarioCovariance, EstimationError>,
    code: &str,
    detail: &str,
) {
    match result {
        Err(EstimationError::Refused { code: c, message }) => {
            assert_eq!(c, code, "{message}");
            assert!(message.starts_with(&format!("{detail}: ")), "{message}");
        }
        other => panic!("expected refusal {detail}, got {other:?}"),
    }
}

// ---- test-side independent oracles ----------------------------------------------------

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw_counts(state: &mut u64) -> Vec<u32> {
    let mut counts = vec![0_u32; N];
    for _ in 0..N {
        counts[(next_u64(state) % N as u64) as usize] += 1;
    }
    counts
}

/// Sample covariance of two functionals where EACH gets its own independent resample.
fn independent_resample_covariance(sa: fn(Row) -> f64, sb: fn(Row) -> f64, r: usize) -> f64 {
    let (mut s1, mut s2) = (0xAAAA_AAAA_1111_1111_u64, 0x5555_5555_2222_2222_u64);
    let pairs: Vec<(f64, f64)> = (0..r)
        .map(|_| (plug_in(&draw_counts(&mut s1), sa), plug_in(&draw_counts(&mut s2), sb)))
        .collect();
    let m = r as f64;
    let ma = pairs.iter().map(|p| p.0).sum::<f64>() / m;
    let mb = pairs.iter().map(|p| p.1).sum::<f64>() / m;
    pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>() / (m - 1.0)
}

/// `[z][x][y]` cell counts of a resampled multiset.
fn cells(counts: &[u32]) -> [[[f64; 2]; 2]; 2] {
    let mut c = [[[0.0; 2]; 2]; 2];
    for (r, &k) in ROWS.iter().zip(counts) {
        c[usize::from(r.0)][usize::from(r.1)][usize::from(r.2)] += f64::from(k);
    }
    c
}

/// Backdoor-adjustment effect `sum_z p(z) [p(y|x=1,z) - p(y|x=0,z)]`.
fn adjusted(counts: &[u32]) -> Option<f64> {
    let c = cells(counts);
    let mut effect = 0.0;
    for stratum in &c {
        let n1 = stratum[1][0] + stratum[1][1];
        let n0 = stratum[0][0] + stratum[0][1];
        if n1 == 0.0 || n0 == 0.0 {
            return None;
        }
        let pz = (n1 + n0) / N as f64;
        effect += pz * (stratum[1][1] / n1 - stratum[0][1] / n0);
    }
    Some(effect)
}

/// Unadjusted contrast `p(y|x=1) - p(y|x=0)`.
fn crude(counts: &[u32]) -> Option<f64> {
    let c = cells(counts);
    let n1: f64 = (0..2).map(|z| c[z][1][0] + c[z][1][1]).sum();
    let n0: f64 = (0..2).map(|z| c[z][0][0] + c[z][0][1]).sum();
    if n1 == 0.0 || n0 == 0.0 {
        return None;
    }
    let y1: f64 = (0..2).map(|z| c[z][1][1]).sum();
    let y0: f64 = (0..2).map(|z| c[z][0][1]).sum();
    Some(y1 / n1 - y0 / n0)
}

fn fallible(id: &str, f: fn(&[u32]) -> Option<f64>) -> ScenarioRowEstimator {
    ScenarioRowEstimator::new(id, SNAPSHOT, units("u"), move |c| {
        f(c).ok_or(EstimationError::unsupported("empty stratum"))
    })
}

struct Oracle {
    cov: [[f64; 2]; 2],
    failed_fraction: f64,
}

/// Walk all `n^n` row-index tuples; drop a tuple if either estimator fails.
fn enumerate_tuples() -> Oracle {
    let total = N.pow(N as u32);
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for t in 0..total {
        let (mut x, mut counts) = (t, vec![0_u32; N]);
        for _ in 0..N {
            counts[x % N] += 1;
            x /= N;
        }
        if let (Some(a), Some(b)) = (adjusted(&counts), crude(&counts)) {
            kept.push((a, b));
        }
    }
    let m = kept.len() as f64;
    let ma = kept.iter().map(|p| p.0).sum::<f64>() / m;
    let mb = kept.iter().map(|p| p.1).sum::<f64>() / m;
    let caa = kept.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>() / m;
    let cbb = kept.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>() / m;
    let cab = kept.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>() / m;
    Oracle { cov: [[caa, cab], [cab, cbb]], failed_fraction: 1.0 - m / total as f64 }
}

// ---- positive: shared rows -------------------------------------------------------------

#[test]
fn x2_shared_rows_covariance_exact_matches_closed_form() {
    let family = vec![linear_by_name("a"), linear_by_name("b")];
    let out = exact_enumeration_covariance(&family, &exact_options(), &ctx()).unwrap();
    assert_eq!(out.method, CovarianceMethod::ExactEnumeration);
    assert_eq!(out.method.label(), "exact_enumeration");
    assert_eq!(out.replicates_total, 462, "C(11, 6) count vectors of six rows");
    assert_eq!(out.replicates_used, 462);
    assert_eq!(out.failed_replicates, 0);
    let ids: Vec<&str> = out.scenario_ids.iter().map(|s| &**s).collect();
    assert_eq!(ids, ["a", "b"]);

    let scores: [fn(Row) -> f64; 2] = [score_a, score_b];
    for (i, si) in scores.iter().enumerate() {
        for (j, sj) in scores.iter().enumerate() {
            let want = closed_form(*si, *sj);
            assert!((out.entry(i, j) - want).abs() < 1e-12, "({i},{j}): {} vs {want}", out.entry(i, j));
        }
    }
    assert!((out.entry(0, 0) - 17.0 / 216.0).abs() < 1e-12);
    assert!((out.entry(1, 1) - 17.0 / 216.0).abs() < 1e-12);
    assert!((out.entry(0, 1) - 11.0 / 216.0).abs() < 1e-12);
    assert_eq!(out.entry(0, 1).to_bits(), out.entry(1, 0).to_bits());
    assert!((out.means[0] - 1.0 / 6.0).abs() < 1e-12);
    assert!((out.means[1] - 1.0 / 6.0).abs() < 1e-12);
    assert!(out.entry(0, 1).abs() > 0.05, "shared rows give a nonzero off-diagonal");
}

#[test]
fn x2_shared_rows_covariance_bootstrap_matches_oracle_and_shared_design_matters() {
    let family = vec![linear_by_name("a"), linear_by_name("b")];
    let out = shared_row_bootstrap_covariance(&family, &boot_options(20_260_101), &ctx()).unwrap();
    assert_eq!(out.method.label(), "shared_row_bootstrap");
    assert_eq!(out.replicates_total, MAX_REPLICATES as u64);
    assert_eq!(out.replicate_ids.len(), MAX_REPLICATES);
    assert_eq!(out.failed_replicates, 0);
    assert_eq!(out.seed, Some(20_260_101));

    // Monte-Carlo tolerance: the sample covariance of R = 2000 draws has a standard
    // deviation near sqrt((1 + rho^2) / R) ~ 0.032 of sigma_i * sigma_j; the bound is 0.15.
    let scores: [fn(Row) -> f64; 2] = [score_a, score_b];
    for (i, si) in scores.iter().enumerate() {
        for (j, sj) in scores.iter().enumerate() {
            let want = closed_form(*si, *sj);
            let scale = (closed_form(*si, *si) * closed_form(*sj, *sj)).sqrt();
            assert!(
                (out.entry(i, j) - want).abs() < 0.15 * scale,
                "({i},{j}): {} vs {want}",
                out.entry(i, j)
            );
        }
    }
    let oracle_off = closed_form(score_a, score_b);
    assert!(oracle_off.abs() > 0.05);
    assert!(out.entry(0, 1) > 0.03, "shared replicate selection recovers the dependence");

    // Independent per-scenario resamples cannot estimate it.
    let independent = independent_resample_covariance(score_a, score_b, 2000);
    let scale = closed_form(score_a, score_a);
    assert!(independent.abs() < 0.15 * scale, "independent off-diagonal {independent}");
    assert!(out.entry(0, 1) - independent > 0.03, "shared {} vs independent {independent}", out.entry(0, 1));
}

#[test]
fn x2_shared_rows_covariance_adjustment_pair_with_joint_dropping_matches_tuple_enumeration() {
    let family = vec![fallible("adjusted", adjusted), fallible("crude", crude)];
    let oracle = enumerate_tuples();
    assert!(oracle.failed_fraction > 0.0, "empty strata must occur");
    assert!(oracle.cov[0][1].abs() > 1e-4, "oracle off-diagonal {}", oracle.cov[0][1]);

    let options = ExactEnumerationOptions { max_compositions: 10_000, max_failure_mass: 0.9 };
    let exact = exact_enumeration_covariance(&family, &options, &ctx()).unwrap();
    assert!(exact.failed_replicates > 0);
    assert_eq!(exact.replicates_used + exact.failed_replicates, 462);
    assert!((exact.failed_mass - oracle.failed_fraction).abs() < 1e-12);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (exact.entry(i, j) - oracle.cov[i][j]).abs() < 1e-12,
                "({i},{j}): {} vs {}",
                exact.entry(i, j),
                oracle.cov[i][j]
            );
        }
    }
    assert!(exact.entry(0, 1).abs() > 1e-4);

    let boot_opts =
        SharedRowBootstrapOptions { replicates: 2000, seed: 77, max_failure_fraction: 0.9 };
    let boot = shared_row_bootstrap_covariance(&family, &boot_opts, &ctx()).unwrap();
    assert!(boot.failed_replicates > 0);
    assert_eq!(boot.replicates_used + boot.failed_replicates, 2000);
    assert_eq!(boot.replicate_ids.len() as u64, boot.replicates_used);
    assert_eq!(boot.failed_replicate_ids.len() as u64, boot.failed_replicates);
    assert!((boot.failed_mass - oracle.failed_fraction).abs() < 0.05);
    for i in 0..2 {
        for j in 0..2 {
            let scale = (oracle.cov[0][0] * oracle.cov[1][1]).sqrt();
            assert!(
                (boot.entry(i, j) - oracle.cov[i][j]).abs() < 0.25 * scale,
                "({i},{j}): {} vs {}",
                boot.entry(i, j),
                oracle.cov[i][j]
            );
        }
    }
}

// ---- negative: incompatible / unknown dependence and bounds --------------------------

const UNKNOWN: (&str, &str) = ("route_not_supported", "scenario_covariance.unknown_dependence");

fn both_methods_refuse(family: &[ScenarioRowEstimator], code: &str, detail: &str) {
    assert_refusal(exact_enumeration_covariance(family, &exact_options(), &ctx()), code, detail);
    assert_refusal(shared_row_bootstrap_covariance(family, &boot_options(1), &ctx()), code, detail);
}

#[test]
fn x2_independent_rows_mismatched_snapshot_refuses() {
    let family = vec![
        linear("a", "snapshot:one", units("u"), score_a),
        linear("b", "snapshot:two", units("u"), score_b),
    ];
    both_methods_refuse(&family, UNKNOWN.0, UNKNOWN.1);
}

#[test]
fn x2_independent_rows_mismatched_or_reordered_units_refuse() {
    let mut reordered = units("u");
    reordered.swap(0, 1);
    let family = vec![
        linear("a", SNAPSHOT, units("u"), score_a),
        linear("b", SNAPSHOT, reordered, score_b),
    ];
    both_methods_refuse(&family, UNKNOWN.0, UNKNOWN.1);
    let family = vec![
        linear("a", SNAPSHOT, units("u"), score_a),
        linear("b", SNAPSHOT, units("other"), score_b),
    ];
    both_methods_refuse(&family, UNKNOWN.0, UNKNOWN.1);
}

#[test]
fn x2_independent_rows_declared_independent_or_unknown_dependence_refuses() {
    for dependence in [RowDependence::IndependentSample, RowDependence::Unknown] {
        let family = vec![
            linear("a", SNAPSHOT, units("u"), score_a),
            linear("b", SNAPSHOT, units("u"), score_b).with_dependence(dependence),
        ];
        both_methods_refuse(&family, UNKNOWN.0, UNKNOWN.1);
    }
}

#[test]
fn x2_independent_rows_duplicate_unit_ids_refuse_without_a_cluster_map() {
    let mut dup = units("u");
    dup[3] = dup[2].clone();
    let family = vec![
        linear("a", SNAPSHOT, dup.clone(), score_a),
        linear("b", SNAPSHOT, dup, score_b),
    ];
    both_methods_refuse(&family, UNKNOWN.0, UNKNOWN.1);
}

#[test]
fn x2_independent_rows_bounds_and_arguments_refuse() {
    // Too many scenarios.
    let many: Vec<ScenarioRowEstimator> = (0..=MAX_SCENARIOS)
        .map(|i| linear(&format!("s{i}"), SNAPSHOT, units("u"), score_a))
        .collect();
    both_methods_refuse(&many, "cell_not_licensed", "scenario_covariance.too_many_scenarios");

    // Too many / too few replicates.
    let family = vec![linear_by_name("a"), linear_by_name("b")];
    let over = SharedRowBootstrapOptions { replicates: MAX_REPLICATES + 1, seed: 1, max_failure_fraction: 0.0 };
    assert_refusal(
        shared_row_bootstrap_covariance(&family, &over, &ctx()),
        "cell_not_licensed",
        "scenario_covariance.too_many_replicates",
    );
    let under = SharedRowBootstrapOptions { replicates: 1, seed: 1, max_failure_fraction: 0.0 };
    assert_refusal(
        shared_row_bootstrap_covariance(&family, &under, &ctx()),
        "invalid_argument",
        "scenario_covariance.too_few_replicates",
    );

    // Too few rows.
    let one_row: Vec<Arc<str>> = vec![Arc::from("only")];
    let tiny = vec![ScenarioRowEstimator::new("a", SNAPSHOT, one_row, |_| Ok(0.0))];
    both_methods_refuse(&tiny, "invalid_argument", "scenario_covariance.too_few_rows");

    // Duplicate scenario id and no scenarios.
    let twins = vec![linear_by_name("a"), linear_by_name("a")];
    both_methods_refuse(&twins, "invalid_argument", "scenario_covariance.duplicate_scenario_id");
    both_methods_refuse(&[], "invalid_argument", "scenario_covariance.no_scenarios");

    // Exact enumeration above the declared cap.
    let capped = ExactEnumerationOptions { max_compositions: 461, max_failure_mass: 0.0 };
    assert_refusal(
        exact_enumeration_covariance(&family, &capped, &ctx()),
        "cell_not_licensed",
        "scenario_covariance.exact_enumeration_cap",
    );
}

#[test]
fn x2_independent_rows_failing_estimators_and_cancellation_refuse() {
    let failing = vec![
        linear_by_name("a"),
        ScenarioRowEstimator::new("broken", SNAPSHOT, units("u"), |_| {
            Err(EstimationError::unsupported("always fails"))
        }),
    ];
    let boot = SharedRowBootstrapOptions { replicates: 50, seed: 3, max_failure_fraction: 0.1 };
    assert_refusal(
        shared_row_bootstrap_covariance(&failing, &boot, &ctx()),
        "transport_numerical_failure",
        "scenario_covariance.too_many_failed_replicates",
    );
    assert_refusal(
        exact_enumeration_covariance(&failing, &exact_options(), &ctx()),
        "transport_numerical_failure",
        "scenario_covariance.too_many_failed_replicates",
    );

    let family = vec![linear_by_name("a"), linear_by_name("b")];
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    assert_refusal(
        shared_row_bootstrap_covariance(&family, &boot_options(1), &cancelled),
        "cancelled_no_claim",
        "scenario_covariance.cancelled",
    );
    assert_refusal(
        exact_enumeration_covariance(&family, &exact_options(), &cancelled),
        "cancelled_no_claim",
        "scenario_covariance.cancelled",
    );
}

// ---- permutation -----------------------------------------------------------------------

fn family_in_order(order: &[&str]) -> Vec<ScenarioRowEstimator> {
    order.iter().map(|n| linear_by_name(n)).collect()
}

fn assert_permuted(first: &ScenarioCovariance, second: &ScenarioCovariance, names: &[&str]) {
    for x in names {
        for y in names {
            let p1 = (first.position(x).unwrap(), first.position(y).unwrap());
            let p2 = (second.position(x).unwrap(), second.position(y).unwrap());
            assert!(
                (first.entry(p1.0, p1.1) - second.entry(p2.0, p2.1)).abs() < 1e-15,
                "entry ({x},{y}) changed under permutation"
            );
        }
        assert_eq!(
            first.means[first.position(x).unwrap()].to_bits(),
            second.means[second.position(x).unwrap()].to_bits()
        );
    }
}

#[test]
fn x2_covariance_permutation_reorders_rows_and_columns_identically() {
    let names = ["a", "b", "c"];
    let first_order = family_in_order(&["a", "b", "c"]);
    let second_order = family_in_order(&["c", "a", "b"]);

    let e1 = exact_enumeration_covariance(&first_order, &exact_options(), &ctx()).unwrap();
    let e2 = exact_enumeration_covariance(&second_order, &exact_options(), &ctx()).unwrap();
    assert_permuted(&e1, &e2, &names);
    assert_ne!(e1.replicate_digest, e2.replicate_digest, "scenario order is identity");
    assert!((e1.entry(0, 2) - closed_form(score_a, score_c)).abs() < 1e-12);

    let b1 = shared_row_bootstrap_covariance(&first_order, &boot_options(9), &ctx()).unwrap();
    let b2 = shared_row_bootstrap_covariance(&second_order, &boot_options(9), &ctx()).unwrap();
    assert_permuted(&b1, &b2, &names);
    assert_eq!(b1.replicate_ids, b2.replicate_ids, "row selection does not depend on scenario order");
    assert_ne!(b1.replicate_digest, b2.replicate_digest);
}

// ---- replay ----------------------------------------------------------------------------

#[test]
fn x2_covariance_replay_is_bit_identical_and_identity_sensitive() {
    let make = || family_in_order(&["a", "b"]);
    let first = shared_row_bootstrap_covariance(&make(), &boot_options(42), &ctx()).unwrap();
    let second = shared_row_bootstrap_covariance(&make(), &boot_options(42), &ctx()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.covariance, second.covariance);
    assert_eq!(first.replicate_ids, second.replicate_ids);
    assert_eq!(first.replicate_digest, second.replicate_digest);
    assert_eq!(first.replicate_digest.len(), 32);

    let exact_1 = exact_enumeration_covariance(&make(), &exact_options(), &ctx()).unwrap();
    let exact_2 = exact_enumeration_covariance(&make(), &exact_options(), &ctx()).unwrap();
    assert_eq!(exact_1, exact_2);

    // A changed row identity changes the row digest and the replicate digest.
    let renamed = vec![
        linear("a", SNAPSHOT, units("v"), score_a),
        linear("b", SNAPSHOT, units("v"), score_b),
    ];
    let changed_rows = shared_row_bootstrap_covariance(&renamed, &boot_options(42), &ctx()).unwrap();
    assert_ne!(changed_rows.row_identity_digest, first.row_identity_digest);
    assert_ne!(changed_rows.replicate_digest, first.replicate_digest);

    // A changed snapshot digest changes the replicate digest.
    let other_snapshot = vec![
        linear("a", "snapshot:rows-v2", units("u"), score_a),
        linear("b", "snapshot:rows-v2", units("u"), score_b),
    ];
    let changed_snapshot =
        shared_row_bootstrap_covariance(&other_snapshot, &boot_options(42), &ctx()).unwrap();
    assert_ne!(changed_snapshot.replicate_digest, first.replicate_digest);

    // A changed scenario order changes the digest.
    let swapped = family_in_order(&["b", "a"]);
    let reordered = shared_row_bootstrap_covariance(&swapped, &boot_options(42), &ctx()).unwrap();
    assert_ne!(reordered.replicate_digest, first.replicate_digest);

    // A changed seed changes the replicate ids.
    let reseeded = shared_row_bootstrap_covariance(&make(), &boot_options(43), &ctx()).unwrap();
    assert_ne!(reseeded.replicate_ids, first.replicate_ids);
    assert_ne!(reseeded.replicate_digest, first.replicate_digest);
}
