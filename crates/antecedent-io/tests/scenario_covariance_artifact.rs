//! 2.3A A1 (X2): independent recomputation of the shared-data covariance.
//!
//! The units are the six complete rows `(z, x, y)` the estimate-layer oracle uses. The
//! declared scenarios are linear plug-in scores (closed form `(E[ab] - E[a]E[b]) / n`
//! under multinomial resampling) and the adjusted and crude contrasts (checked against a
//! walk of all `n^n` row-index tuples). A fresh consumer must reproduce the matrix bit
//! for bit from the artifact alone and refuse every re-sealed change.
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "tiny fixtures with exact small integers"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::scenario_covariance::ScenarioCovariance;
use antecedent_io::IoError;
use antecedent_io::scenario_covariance_artifact::{
    CovarianceMethodWire, CovarianceScenarioWire, CovarianceSpec, FunctionalWire, RowTableWire,
    ScenarioCovarianceArtifactWire, ScenarioCovarianceConsumeLimits, ScoreTermWire,
};

/// `(z, x, y)` of the six units.
const ROWS: [[i64; 3]; 6] = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]];
const N: usize = 6;
const SNAPSHOT: &str = "snapshot:rows-v1";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn table() -> RowTableWire {
    RowTableWire {
        columns: vec!["z".into(), "x".into(), "y".into()],
        unit_ids: (0..N).map(|i| format!("u{i}")).collect(),
        rows: ROWS.iter().map(|r| r.to_vec()).collect(),
    }
}

fn term(coefficient: f64, pattern: &[(&str, i64)]) -> ScoreTermWire {
    ScoreTermWire {
        coefficient,
        pattern: pattern.iter().map(|(c, v)| ((*c).to_string(), *v)).collect(),
    }
}

fn functional(name: &str) -> FunctionalWire {
    match name {
        "a" => FunctionalWire::LinearScore {
            terms: vec![term(1.0, &[("x", 1), ("y", 1)]), term(-1.0, &[("x", 0), ("y", 1)])],
        },
        "b" => FunctionalWire::LinearScore {
            terms: vec![term(1.0, &[("x", 1), ("y", 1)]), term(-1.0, &[("x", 1), ("y", 0)])],
        },
        "c" => FunctionalWire::LinearScore { terms: vec![term(1.0, &[("z", 1), ("y", 1)])] },
        "adjusted" => FunctionalWire::AdjustedContrast {
            treatment: "x".into(),
            outcome: "y".into(),
            adjustment: vec!["z".into()],
            treated: 1,
            control: 0,
            outcome_value: 1,
        },
        _ => FunctionalWire::AdjustedContrast {
            treatment: "x".into(),
            outcome: "y".into(),
            adjustment: Vec::new(),
            treated: 1,
            control: 0,
            outcome_value: 1,
        },
    }
}

fn scenario(name: &str) -> CovarianceScenarioWire {
    CovarianceScenarioWire {
        id: name.into(),
        snapshot: SNAPSHOT.into(),
        dependence: "shared_rows".into(),
        unit_ids: None,
        functional: functional(name),
    }
}

fn exact(max_failure_mass: f64) -> CovarianceMethodWire {
    CovarianceMethodWire::ExactEnumeration { max_compositions: 10_000, max_failure_mass }
}

fn bootstrap(seed: u64) -> CovarianceMethodWire {
    CovarianceMethodWire::SharedRowBootstrap { replicates: 300, seed, max_failure_fraction: 0.0 }
}

fn spec(names: &[&str], method: CovarianceMethodWire) -> CovarianceSpec {
    CovarianceSpec {
        table: table(),
        scenarios: names.iter().map(|n| scenario(n)).collect(),
        method,
    }
}

fn produce(spec: &CovarianceSpec) -> (Vec<u8>, ScenarioCovariance) {
    let covariance = spec.compute(&ctx()).unwrap();
    let bytes =
        ScenarioCovarianceArtifactWire::checked(spec, &covariance).unwrap().export().unwrap();
    (bytes, covariance)
}

fn consume(bytes: &[u8]) -> Result<ScenarioCovariance, IoError> {
    ScenarioCovarianceArtifactWire::consume_with_limits(
        bytes,
        ScenarioCovarianceConsumeLimits::default(),
        &ctx(),
    )
    .map(|(_, covariance)| covariance)
}

fn refused(result: Result<ScenarioCovariance, IoError>) -> (&'static str, String) {
    match result {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|c| c.dimension())),
    }
}

fn edit(
    bytes: &[u8],
    change: impl FnOnce(&mut ScenarioCovarianceArtifactWire),
) -> ScenarioCovarianceArtifactWire {
    let mut wire = ScenarioCovarianceArtifactWire::decode(bytes).unwrap();
    change(&mut wire);
    wire
}

fn reseal(mut wire: ScenarioCovarianceArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

// ---- oracles (independent of the library's enumerator) ---------------------------------

fn score(row: &[i64; 3], name: &str) -> f64 {
    let (z, x, y) = (row[0], row[1], row[2]);
    match name {
        "a" => match (x, y) {
            (1, 1) => 1.0,
            (0, 1) => -1.0,
            _ => 0.0,
        },
        "b" => match (x, y) {
            (1, 1) => 1.0,
            (1, 0) => -1.0,
            _ => 0.0,
        },
        _ => f64::from(u8::from((z, y) == (1, 1))),
    }
}

/// Closed-form `Cov(a, b)` of the empirical-proportion functionals under multinomial resampling.
fn closed_form(first: &str, second: &str) -> f64 {
    let n = N as f64;
    let a: Vec<f64> = ROWS.iter().map(|r| score(r, first)).collect();
    let b: Vec<f64> = ROWS.iter().map(|r| score(r, second)).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let ab: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    (mean(&ab) - mean(&a) * mean(&b)) / n
}

fn cells(counts: &[u32]) -> [[[f64; 2]; 2]; 2] {
    let mut c = [[[0.0; 2]; 2]; 2];
    for (r, &k) in ROWS.iter().zip(counts) {
        let idx = |v: i64| usize::try_from(v).unwrap();
        c[idx(r[0])][idx(r[1])][idx(r[2])] += f64::from(k);
    }
    c
}

fn oracle_adjusted(counts: &[u32]) -> Option<f64> {
    let c = cells(counts);
    let mut effect = 0.0;
    for stratum in &c {
        let n1 = stratum[1][0] + stratum[1][1];
        let n0 = stratum[0][0] + stratum[0][1];
        if n1 == 0.0 || n0 == 0.0 {
            return None;
        }
        effect += (n1 + n0) / N as f64 * (stratum[1][1] / n1 - stratum[0][1] / n0);
    }
    Some(effect)
}

fn oracle_crude(counts: &[u32]) -> Option<f64> {
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

/// Walk all `n^n` row-index tuples; drop a tuple if either estimator fails.
fn enumerate_tuples() -> ([[f64; 2]; 2], f64) {
    let total = N.pow(u32::try_from(N).unwrap());
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for t in 0..total {
        let (mut x, mut counts) = (t, vec![0_u32; N]);
        for _ in 0..N {
            counts[x % N] += 1;
            x /= N;
        }
        if let (Some(a), Some(b)) = (oracle_adjusted(&counts), oracle_crude(&counts)) {
            kept.push((a, b));
        }
    }
    let m = kept.len() as f64;
    let ma = kept.iter().map(|p| p.0).sum::<f64>() / m;
    let mb = kept.iter().map(|p| p.1).sum::<f64>() / m;
    let caa = kept.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>() / m;
    let cbb = kept.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>() / m;
    let cab = kept.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>() / m;
    ([[caa, cab], [cab, cbb]], 1.0 - m / total as f64)
}

// ---- positive: the declared functionals match the oracles -------------------------------

#[test]
fn x2_covariance_artifact_exact_linear_scores_match_closed_form_and_replay() {
    let (bytes, original) = produce(&spec(&["a", "b", "c"], exact(0.0)));
    let names = ["a", "b", "c"];
    for (i, first) in names.iter().enumerate() {
        for (j, second) in names.iter().enumerate() {
            let want = closed_form(first, second);
            assert!((original.entry(i, j) - want).abs() < 1e-12, "({first},{second})");
        }
    }
    assert!((original.entry(0, 0) - 17.0 / 216.0).abs() < 1e-12);
    assert!((original.entry(0, 1) - 11.0 / 216.0).abs() < 1e-12);
    // Fresh consumer: the identical matrix, bit for bit, with the point-only label.
    let replayed = consume(&bytes).unwrap();
    assert_eq!(replayed, original);
    assert_eq!(replayed.covariance, original.covariance);
    assert_eq!(replayed.interpretation, original.interpretation);
    assert!(replayed.interpretation.contains("not_an_interval"));
    let wire = ScenarioCovarianceArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.result.scenario_ids, ["a", "b", "c"]);
    assert_eq!(wire.result.n_rows, N);
    assert_eq!(wire.result.snapshot_digest, SNAPSHOT);
    assert_eq!(wire.premises_digest, wire.expected_premises_digest().unwrap());
    assert_eq!(wire.data_digest, wire.expected_data_digest().unwrap());
    assert_eq!(wire.export().unwrap(), bytes);
}

#[test]
fn x2_covariance_artifact_adjusted_pair_matches_tuple_enumeration_with_joint_dropping() {
    let (oracle, failed_fraction) = enumerate_tuples();
    assert!(failed_fraction > 0.0, "empty strata must occur");
    let (bytes, exact) = produce(&spec(&["adjusted", "crude"], exact(0.9)));
    assert!(exact.failed_replicates > 0);
    assert!((exact.failed_mass - failed_fraction).abs() < 1e-12);
    for (i, oracle_row) in oracle.iter().enumerate().take(2) {
        for (j, expected) in oracle_row.iter().enumerate().take(2) {
            assert!((exact.entry(i, j) - expected).abs() < 1e-12, "({i},{j})");
        }
    }
    assert!(exact.entry(0, 1).abs() > 1e-4, "shared rows give a nonzero off-diagonal");
    assert_eq!(consume(&bytes).unwrap(), exact);
}

#[test]
fn x2_covariance_artifact_bootstrap_replays_bit_identically_and_is_seed_sensitive() {
    let (bytes, original) = produce(&spec(&["a", "b"], bootstrap(42)));
    assert_eq!(original.seed, Some(42));
    assert_eq!(original.replicates_total, 300);
    let replayed = consume(&bytes).unwrap();
    assert_eq!(replayed, original);
    assert_eq!(replayed.replicate_digest, original.replicate_digest);
    // A changed seed (re-sealed) replays to a different matrix and is refused.
    let (code, message) = refused(consume(&reseal(edit(&bytes, |w| w.method = bootstrap(43)))));
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("scenario_covariance.replay_mismatch"), "{message}");
}

#[test]
fn x2_covariance_artifact_scenario_permutation_reorders_rows_and_columns_identically() {
    let first = spec(&["a", "b", "c"], exact(0.0)).compute(&ctx()).unwrap();
    let second = spec(&["c", "a", "b"], exact(0.0)).compute(&ctx()).unwrap();
    for x in ["a", "b", "c"] {
        for y in ["a", "b", "c"] {
            let p1 = (first.position(x).unwrap(), first.position(y).unwrap());
            let p2 = (second.position(x).unwrap(), second.position(y).unwrap());
            assert!((first.entry(p1.0, p1.1) - second.entry(p2.0, p2.1)).abs() < 1e-15);
        }
    }
    assert_ne!(first.replicate_digest, second.replicate_digest, "scenario order is identity");
}

// ---- negative: resealed changes ----------------------------------------------------------

#[test]
fn x2_covariance_artifact_resealed_changes_are_refused() {
    let (bytes, _) = produce(&spec(&["a", "b"], exact(0.0)));
    let mismatch = |wire: ScenarioCovarianceArtifactWire| {
        let (code, message) = refused(consume(&reseal(wire)));
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with("scenario_covariance.replay_mismatch"), "{message}");
    };
    // A changed row snapshot: one cell of one row.
    mismatch(edit(&bytes, |w| w.table.rows[0][2] = 1));
    // A changed unit id.
    mismatch(edit(&bytes, |w| w.table.unit_ids[0] = "forged".into()));
    // A changed scenario order: the stored matrix keeps the original order.
    mismatch(edit(&bytes, |w| w.scenarios.swap(0, 1)));
    // A changed stored matrix entry.
    mismatch(edit(&bytes, |w| w.result.covariance[1] += 1e-9));
    // A changed scenario declaration.
    mismatch(edit(&bytes, |w| w.scenarios[0].functional = functional("c")));

    // Un-resealed, the digests refuse first.
    let (_, message) =
        refused(consume(&edit(&bytes, |w| w.table.rows[0][2] = 1).export().unwrap()));
    assert!(message.starts_with("scenario_covariance.data_identity_mismatch"), "{message}");
    let (_, message) =
        refused(consume(&edit(&bytes, |w| w.scenarios.swap(0, 1)).export().unwrap()));
    assert!(message.starts_with("scenario_covariance.premises_mismatch"), "{message}");
}

#[test]
fn x2_covariance_artifact_scenario_on_a_different_snapshot_refuses_unknown_dependence() {
    let (bytes, _) = produce(&spec(&["a", "b"], exact(0.0)));
    let other = |wire: &mut ScenarioCovarianceArtifactWire| {
        wire.scenarios[1].snapshot = "snapshot:rows-v2".into();
    };
    let (code, message) = refused(consume(&reseal(edit(&bytes, other))));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("scenario_covariance.unknown_dependence"), "{message}");
    // The producer refuses the same declaration before any artifact exists.
    let mut declared = spec(&["a", "b"], exact(0.0));
    declared.scenarios[1].snapshot = "snapshot:rows-v2".into();
    let (code, message) = refused(declared.compute(&ctx()));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("scenario_covariance.unknown_dependence"), "{message}");
    // Declared-independent or unknown dependence, and a different unit list, refuse alike.
    for dependence in ["independent_sample", "unknown"] {
        let mut declared = spec(&["a", "b"], exact(0.0));
        declared.scenarios[1].dependence = dependence.into();
        let (code, message) = refused(declared.compute(&ctx()));
        assert_eq!(code, "route_not_supported");
        assert!(message.starts_with("scenario_covariance.unknown_dependence"), "{message}");
    }
    let mut declared = spec(&["a", "b"], exact(0.0));
    declared.scenarios[1].unit_ids = Some((0..N).map(|i| format!("w{i}")).collect());
    let (code, message) = refused(declared.compute(&ctx()));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("scenario_covariance.unknown_dependence"), "{message}");
}

#[test]
fn x2_covariance_artifact_version_features_and_consumer_limits_refuse() {
    let (bytes, _) = produce(&spec(&["a", "b"], bootstrap(7)));
    let future = edit(&bytes, |w| w.version = 2).export().unwrap();
    assert!(matches!(
        ScenarioCovarianceArtifactWire::decode(&future),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert!(matches!(consume(&future), Err(IoError::UnsupportedVersion { version: 2 })));
    let foreign =
        edit(&bytes, |w| w.required_features.push("other_feature_v9".into())).export().unwrap();
    let (code, message) = refused(consume(&foreign));
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("scenario_covariance.unsupported_semantics"), "{message}");
    // A matrix claiming to be an interval is not this format.
    let interval = edit(&bytes, |w| w.result.interpretation = "confidence_interval".into());
    let (_, message) = refused(consume(&interval.export().unwrap()));
    assert!(message.starts_with("scenario_covariance.unsupported_semantics"), "{message}");

    let strict = |limits: ScenarioCovarianceConsumeLimits| {
        let result = ScenarioCovarianceArtifactWire::consume_with_limits(&bytes, limits, &ctx());
        refused(result.map(|(_, c)| c))
    };
    let base = ScenarioCovarianceConsumeLimits::default();
    for limits in [
        ScenarioCovarianceConsumeLimits { max_rows: N - 1, ..base },
        ScenarioCovarianceConsumeLimits { max_columns: 2, ..base },
        ScenarioCovarianceConsumeLimits { max_replicates: 299, ..base },
    ] {
        let (code, message) = strict(limits);
        assert_eq!(code, "cell_not_licensed");
        assert!(message.starts_with("scenario_covariance.consumer_limit_exceeded"), "{message}");
    }
}

#[test]
fn x2_covariance_artifact_invalid_declarations_refuse_with_typed_details() {
    let invalid = |change: &dyn Fn(&mut CovarianceSpec), detail: &str| {
        let mut declared = spec(&["a", "b"], exact(0.0));
        change(&mut declared);
        let (code, message) = refused(declared.compute(&ctx()));
        assert_eq!(code, "invalid_argument", "{message}");
        assert!(message.starts_with(detail), "{message}");
    };
    invalid(&|s| s.table.rows[0].pop().map_or((), |_| ()), "scenario_covariance.invalid_row_table");
    invalid(&|s| s.table.columns[1] = "z".into(), "scenario_covariance.invalid_row_table");
    invalid(
        &|s| s.scenarios[0].dependence = "maybe".into(),
        "scenario_covariance.invalid_dependence",
    );
    invalid(
        &|s| {
            s.scenarios[0].functional =
                FunctionalWire::LinearScore { terms: vec![term(1.0, &[("nope", 1)])] };
        },
        "scenario_covariance.invalid_functional",
    );
    invalid(&|s| s.scenarios[1].id = "a".into(), "scenario_covariance.duplicate_scenario_id");
}
