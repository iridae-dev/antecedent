//! Prepared lifecycle of a linear Gaussian mechanism across observation periods, through the
//! public facade: a prepared series study fits the autoregressive mechanism of one column by
//! least squares and by the conjugate normal-inverse-gamma posterior, extends or restarts it
//! when a later period arrives, retains the delta ledger as an artifact, and an independent
//! consumer recomputes everything from the retained blocks.
//!
//! The references are normal equations solved here by Gaussian elimination over the rows a
//! fresh fit would use, not the library's own solver.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::BTreeSet;

use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent::{
    AcceptedGraph, InferenceMode, PreparedStudy, RefuteSuite, Study,
    consume_period_refresh_artifact,
};
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity, WindowProduct,
};
use antecedent_io::period_refresh_artifact::{PeriodRefreshArtifactWire, PriorWire};
use antecedent_state::{ConjugateOlsPrior, RefreshOutcome};
use common::persistent_dgp::{autoregressive_pulse_dag, autoregressive_pulse_series, pulse};

const N: usize = 1200;
const SPLIT: usize = 600;
const LAGS: usize = 2;

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn window(start: i64, end: i64, snapshot: &str) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "ar_mechanism".into(),
        horizon: 2,
        lag_alignment: vec![("t".into(), 1)],
        intervention_history: vec!["a1=1".into(), "a2=0".into()],
        selection_targets: set(&["s_t1"]),
        regimes: set(&["observational"]),
        period: ObservationPeriod { start, end },
        unit_ids: set(&["u1", "u2"]),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn cadence() -> ObservationCadence {
    ObservationCadence::new(1).unwrap()
}

fn spec(standing: MechanismStanding) -> PeriodRefreshSpec<'static> {
    PeriodRefreshSpec {
        series_id: "t",
        cadence: cadence(),
        initial_law: "law_1",
        standing,
        lags: LAGS,
    }
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(
        vec![0.0; 3],
        vec![0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3],
        3.0,
        2.0,
    )
    .unwrap()
}

fn other_prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::isotropic(3, 1.0, 4.0, 1.5).unwrap()
}

fn prepared(data: &TimeSeriesData) -> PreparedStudy {
    let ctx = ExecutionContext::for_tests(11);
    Study::series(data.clone())
        .graph(AcceptedGraph::temporal_dag(autoregressive_pulse_dag(true, false)))
        .query(CausalQuery::TemporalEffect(pulse()))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
}

fn column(data: &TimeSeriesData, name: &str) -> Vec<f64> {
    data.float64_values(data.schema().id_of(name).unwrap()).unwrap()
}

/// The rows `[from, to)` of every column, as a series of the prepared study's schema.
fn slice(data: &TimeSeriesData, from: usize, to: usize) -> TimeSeriesData {
    let names = ["t", "y", "z"];
    let columns: Vec<Vec<f64>> =
        names.iter().map(|name| column(data, name)[from..to].to_vec()).collect();
    TimeSeriesData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, values)| (*name, values.as_slice())),
        1,
    )
    .unwrap()
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                let sub = factor * a[col][k];
                a[row][k] -= sub;
            }
            let sub = factor * b[col];
            b[row] -= sub;
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// `(x, y)` rows `[1, v_{i-1}, v_{i-2}] -> v_i` for the targets `i` in `targets`.
fn rows(values: &[f64], targets: std::ops::Range<usize>) -> Vec<(Vec<f64>, f64)> {
    targets.map(|i| (vec![1.0, values[i - 1], values[i - 2]], values[i])).collect()
}

struct Reference {
    n: u64,
    beta: Vec<f64>,
    mean: Vec<f64>,
    shape: f64,
    rate: f64,
}

/// Least squares and the normal-inverse-gamma posterior of `rows` under `prior`.
fn reference(rows: &[(Vec<f64>, f64)], prior: &ConjugateOlsPrior) -> Reference {
    let p = 3;
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    let mut yty = 0.0;
    for (x, y) in rows {
        for i in 0..p {
            xty[i] += x[i] * y;
            for j in 0..p {
                xtx[i][j] += x[i] * x[j];
            }
        }
        yty += y * y;
    }
    let beta = solve(xtx.clone(), xty.clone());
    let l0: Vec<Vec<f64>> =
        (0..p).map(|i| (0..p).map(|j| prior.precision()[i * p + j]).collect()).collect();
    let m0 = prior.mean();
    let eta0: Vec<f64> = (0..p).map(|i| (0..p).map(|j| l0[i][j] * m0[j]).sum()).collect();
    let lambda: Vec<Vec<f64>> =
        (0..p).map(|i| (0..p).map(|j| l0[i][j] + xtx[i][j]).collect()).collect();
    let eta: Vec<f64> = (0..p).map(|i| eta0[i] + xty[i]).collect();
    let mean = solve(lambda, eta.clone());
    let c: f64 = (0..p).map(|i| m0[i] * eta0[i]).sum::<f64>() + yty;
    let explained: f64 = (0..p).map(|i| mean[i] * eta[i]).sum();
    let n = rows.len();
    Reference {
        n: n as u64,
        beta,
        mean,
        shape: prior.shape() + n as f64 / 2.0,
        rate: prior.rate() + 0.5 * (c - explained),
    }
}

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-8 * (1.0 + expected.abs()),
        "{label}: {actual:.15e} vs {expected:.15e}"
    );
}

fn assert_matches(refresh: &PeriodRefresh, want: &Reference) {
    let fit = refresh.fit().snapshot();
    assert_eq!(fit.n, want.n);
    for (got, expected) in fit.beta.as_ref().unwrap().iter().zip(&want.beta) {
        close(*got, *expected, "least squares");
    }
    let posterior = refresh.summary().unwrap();
    assert_eq!(posterior.n, want.n);
    for (got, expected) in posterior.mean.iter().zip(&want.mean) {
        close(*got, *expected, "posterior mean");
    }
    close(posterior.shape, want.shape, "shape");
    close(posterior.rate, want.rate, "rate");
    refresh.check_against_fresh().expect("cached results equal a fresh reconstruction");
}

fn start(
    study: &PreparedStudy,
    data: &TimeSeriesData,
    standing: MechanismStanding,
) -> PeriodRefresh {
    let first = window(0, i64::try_from(SPLIT).unwrap(), "snap_1");
    study
        .start_period_refresh(&slice(data, 0, SPLIT), "t", &spec(standing), &first, &prior())
        .unwrap()
}

fn apply(
    study: &PreparedStudy,
    refresh: &mut PeriodRefresh,
    data: &TimeSeriesData,
    second: &TemporalWindowIdentity,
) -> Result<PeriodRefreshOutcome, antecedent::CausalError> {
    let to = usize::try_from(second.period.end).unwrap();
    study.apply_period_refresh(refresh, &slice(data, SPLIT, to), "t", second, cadence(), "law_1")
}

#[test]
fn a_declared_invariant_period_extends_and_equals_one_batch_fit() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let values = column(&data, "t");
    let mut refresh = start(&study, &data, MechanismStanding::DeclaredInvariant);
    assert_matches(&refresh, &reference(&rows(&values, 2..SPLIT), &prior()));

    let second = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_2");
    let outcome = apply(&study, &mut refresh, &data, &second).unwrap();
    assert_eq!(outcome, PeriodRefreshOutcome::Extended { contiguous: true });
    // The rows spanning the boundary are formed: the batch over every target of the series.
    assert_matches(&refresh, &reference(&rows(&values, 2..N), &prior()));
    assert_eq!(refresh.fit().recomputes(), 0, "an append is not a rebuild");
    assert_eq!(refresh.blocks().len(), 2);
}

#[test]
fn an_undeclared_mechanism_restarts_on_the_new_period_alone() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let values = column(&data, "t");
    let mut refresh = start(&study, &data, MechanismStanding::Undeclared);
    let second = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_2");
    let outcome = apply(&study, &mut refresh, &data, &second).unwrap();
    let PeriodRefreshOutcome::Restarted { invalidated } = outcome else {
        panic!("an undeclared mechanism must not carry its fit across periods");
    };
    assert!(invalidated.contains(&WindowProduct::Model));
    // Only the new period's own rows, under the originally stated prior.
    assert_matches(&refresh, &reference(&rows(&values, SPLIT + LAGS..N), &prior()));
    assert_eq!(refresh.blocks().len(), 1);
}

#[test]
fn an_overlapping_period_a_reused_snapshot_and_a_changed_prior_are_typed_refusals() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let mut refresh = start(&study, &data, MechanismStanding::DeclaredInvariant);
    let before = refresh.summary().unwrap();

    let overlap = window(300, 900, "snap_overlap");
    let error = study
        .apply_period_refresh(
            &mut refresh,
            &slice(&data, 300, 900),
            "t",
            &overlap,
            cadence(),
            "law_1",
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("period_refresh.overlap"), "{error}");
    assert!(error.contains("supply a later period of a new snapshot"), "{error}");
    let reused = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_1");
    let error = apply(&study, &mut refresh, &data, &reused).unwrap_err().to_string();
    assert!(error.contains("reuses its snapshot"), "{error}");
    assert_eq!(refresh.summary().unwrap(), before, "a refused period changes nothing");

    let error = refresh.summary_under(&other_prior()).unwrap_err().to_string();
    assert!(error.contains("different prior is a different posterior"), "{error}");
    assert!(error.contains("rebuild from every retained block under the new prior"), "{error}");

    // The named next attempt: rebuild from the retained blocks under the new prior.
    refresh.reprior(&other_prior()).unwrap();
    let values = column(&data, "t");
    assert_matches(&refresh, &reference(&rows(&values, 2..SPLIT), &other_prior()));
}

#[test]
fn another_schema_or_a_missing_column_is_refused_before_any_fit() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let narrow = TimeSeriesData::from_f64_columns(
        [("t", column(&data, "t")[..SPLIT].to_vec().as_slice())],
        1,
    )
    .unwrap();
    let first = window(0, i64::try_from(SPLIT).unwrap(), "snap_1");
    let standing = MechanismStanding::DeclaredInvariant;
    let error =
        study.start_period_refresh(&narrow, "t", &spec(standing), &first, &prior()).unwrap_err();
    assert!(error.to_string().contains("same schema"), "{error}");
    let error = study
        .start_period_refresh(&slice(&data, 0, SPLIT), "w", &spec(standing), &first, &prior())
        .unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn the_ledger_round_trips_and_an_independent_consumer_recomputes_it() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let values = column(&data, "t");
    let mut refresh = start(&study, &data, MechanismStanding::DeclaredInvariant);
    let second = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_2");
    apply(&study, &mut refresh, &data, &second).unwrap();

    let bytes = refresh.export_ledger().unwrap();
    let (wire, replay) = consume_period_refresh_artifact(&bytes).unwrap();
    let want = reference(&rows(&values, 2..N), &prior());
    assert_eq!(replay.posterior.n, want.n);
    for (got, expected) in replay.posterior.mean.iter().zip(&want.mean) {
        close(*got, *expected, "consumer posterior mean");
    }
    close(replay.posterior.rate, want.rate, "consumer rate");
    for (got, expected) in replay.least_squares.beta.iter().zip(&want.beta) {
        close(*got, *expected, "consumer least squares");
    }
    assert_eq!(replay.live_snapshots, ["snap_1", "snap_2"]);
    assert_eq!(wire.rows_used, want.n);
    assert_eq!(wire.events.len(), 2);
    assert_eq!(wire.events[1].outcome, "extended");
    assert!(wire.events[1].contiguous);

    // Evicting a row rebuilds both summaries from the rows that remain, and the ledger follows.
    let outcome = refresh.evict_row(100).unwrap();
    assert_eq!(outcome, RefreshOutcome::Recomputed);
    let mut kept = rows(&values, 2..N);
    kept.remove(98);
    assert_matches(&refresh, &reference(&kept, &prior()));
    let (_, replay) = consume_period_refresh_artifact(&refresh.export_ledger().unwrap()).unwrap();
    assert_eq!(replay.posterior.n, want.n - 1);
    assert!(refresh.evict_row(100).is_err(), "an evicted row is gone");

    // A restart ledger records the period that dropped the earlier rows.
    let mut restarted = start(&study, &data, MechanismStanding::Undeclared);
    apply(&study, &mut restarted, &data, &second).unwrap();
    let (wire, replay) =
        consume_period_refresh_artifact(&restarted.export_ledger().unwrap()).unwrap();
    assert_eq!(wire.events[1].outcome, "restarted");
    assert_eq!(replay.live_snapshots, ["snap_2"]);
}

#[test]
fn a_tampered_ledger_or_a_block_used_twice_is_refused_by_the_consumer() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared(&data);
    let mut refresh = start(&study, &data, MechanismStanding::DeclaredInvariant);
    let second = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_2");
    apply(&study, &mut refresh, &data, &second).unwrap();
    let good = PeriodRefreshArtifactWire::decode(&refresh.export_ledger().unwrap()).unwrap();

    let refused = |wire: PeriodRefreshArtifactWire, needle: &str| {
        let error = wire.checked().unwrap_err().to_string();
        assert!(error.contains(needle), "{needle}: {error}");
    };

    // An edit without resealing: the seal catches it.
    let mut edited = good.clone();
    edited.blocks[0].rows[0].y += 1.0;
    let error = consume_period_refresh_artifact(&edited.export().unwrap()).unwrap_err();
    assert!(error.to_string().contains("seal"), "{error}");

    // Resealed forgeries: each is caught by a recomputation, not by the seal.
    let mut forged = good.clone();
    forged.blocks[0].rows[0].y += 1.0;
    refused(forged, "block_digest");

    let mut doubled = good.clone();
    let duplicate = doubled.blocks[0].clone();
    doubled.blocks.push(duplicate);
    refused(doubled, "block_reused");

    let mut shared_row = good.clone();
    let row = shared_row.blocks[0].rows[0].clone();
    let rows = &mut shared_row.blocks[1].rows;
    rows.push(row);
    let label = shared_row.blocks[1].snapshot_id.clone();
    shared_row.blocks[1] = antecedent_io::period_refresh_artifact::BlockWire::new(
        label,
        shared_row.blocks[1].rows.clone(),
    );
    refused(shared_row, "block_reused");

    let mut posterior = good.clone();
    posterior.posterior.mean[1] += 0.01;
    refused(posterior, "posterior");

    let mut least_squares = good.clone();
    least_squares.least_squares.beta[1] += 0.01;
    refused(least_squares, "least_squares");

    let mut swapped_prior = good.clone();
    swapped_prior.prior.shape += 1.0;
    swapped_prior.prior_digest = swapped_prior.prior.digest();
    refused(swapped_prior, "posterior");

    let mut stale_digest = good.clone();
    stale_digest.prior.rate += 1.0;
    refused(stale_digest, "prior_digest");

    let mut wrong_outcome = good.clone();
    wrong_outcome.events[1].outcome = "restarted".into();
    refused(wrong_outcome, "outcome");

    let mut overlapping = good.clone();
    overlapping.events[1].window.period_start = 300;
    refused(overlapping, "overlap");

    let mut reused_snapshot = good.clone();
    reused_snapshot.events[1].window.snapshot_id = "snap_1".into();
    refused(reused_snapshot, "block_reused");

    // A different prior held now is a typed refusal naming the rebuild.
    let held = PriorWire {
        mean: other_prior().mean().to_vec(),
        precision: other_prior().precision().to_vec(),
        shape: other_prior().shape(),
        rate: other_prior().rate(),
    };
    let error = good.require_prior(&held).unwrap_err().to_string();
    assert!(error.contains("period_refresh.prior_changed"), "{error}");
    assert!(error.contains("rebuild from every retained block"), "{error}");
    good.require_prior(&good.prior).unwrap();
}
