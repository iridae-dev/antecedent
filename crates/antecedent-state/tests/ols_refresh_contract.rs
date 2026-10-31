//! Delta contract of the least-squares summary against reference fits.
//!
//! Reference values are row-wise numpy fits of the stated rows (design `[1, x]`).

use antecedent_state::{
    LinearOlsDelta, LinearOlsSuffStats, OlsRefresh, RefreshChange, RefreshDisposition,
    RefreshOutcome, RefreshRoute, RefreshRow, StateError, route_contract,
};

const X: [f64; 12] = [0.5, 1.2, -0.3, 2.0, 1.5, -1.0, 0.8, 2.2, 0.1, 1.7, -0.6, 0.3];
const Y: [f64; 12] = [1.1, 2.4, 0.2, 3.9, 3.1, -0.9, 1.7, 4.6, 0.9, 3.2, -0.1, 1.0];

fn row(id: usize) -> RefreshRow {
    RefreshRow::new(id as u64, vec![1.0, X[id]], Y[id])
}

fn rows(range: std::ops::Range<usize>) -> Vec<RefreshRow> {
    range.map(row).collect()
}

fn close(actual: f64, expected: f64, label: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{label}: {actual:.17e} vs {expected:.17e}");
}

fn assert_fit(fit: &OlsRefresh, n: u64, beta: [f64; 2], sigma2: f64, yty: f64) {
    let snapshot = fit.snapshot();
    assert_eq!(snapshot.n, n);
    let got = snapshot.beta.as_ref().expect("beta");
    close(got[0], beta[0], "intercept");
    close(got[1], beta[1], "slope");
    close(snapshot.residual_variance.expect("sigma2"), sigma2, "residual variance");
    close(snapshot.yty, yty, "yty");
    fit.check_against_fresh().expect("cached result equals the fresh rebuild");
}

fn full() -> OlsRefresh {
    OlsRefresh::from_rows("series_a", 2, rows(0..12)).unwrap()
}

#[test]
fn appended_rows_and_windows_equal_the_reference_without_a_rebuild() {
    let mut fit = OlsRefresh::new("series_a", 2).unwrap();
    for id in 0..5 {
        assert_eq!(fit.append_row("series_a", &row(id)).unwrap(), RefreshOutcome::Updated);
    }
    assert_eq!(fit.append_window("series_a", &rows(5..9)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(fit.append_window("series_a", &rows(9..12)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(fit.recomputes(), 0);
    assert_fit(
        &fit,
        12,
        [0.629_451_864_218_141_1, 1.612_687_813_021_702_4],
        0.049_203_811_908_736_7,
        68.75,
    );
    close(fit.snapshot().xtx[3], 17.860_000_000_000_003, "xtx");
    assert_eq!(fit.ledger().rows, 12);
}

#[test]
fn evicting_a_row_recomputes_and_equals_the_reference_on_the_rest() {
    let mut fit = full();
    assert_eq!(fit.evict_row(3).unwrap(), RefreshOutcome::Recomputed);
    assert_eq!(fit.recomputes(), 1);
    assert_fit(
        &fit,
        11,
        [0.629_022_421_524_663_8, 1.606_367_713_004_484],
        0.054_378_574_987_543_48,
        53.54,
    );
    assert_eq!(fit.ledger().rows, 11);
    assert!(
        matches!(fit.evict_row(3), Err(StateError::UnknownId(ref id)) if id == "row 3"),
        "the row is gone"
    );
}

#[test]
fn a_mask_recomputes_and_keeps_the_rows_retained() {
    let mut fit = full();
    fit.set_mask(0, true).unwrap();
    fit.set_mask(11, true).unwrap();
    assert_fit(
        &fit,
        10,
        [0.683_378_746_594_005_2, 1.600_817_438_692_098_1],
        0.043_079_019_073_569_37,
        66.54,
    );
    assert_eq!((fit.ledger().rows, fit.ledger().live_rows), (12, 10));
    fit.set_mask(0, false).unwrap();
    fit.set_mask(11, false).unwrap();
    assert_fit(
        &fit,
        12,
        [0.629_451_864_218_141_1, 1.612_687_813_021_702_4],
        0.049_203_811_908_736_7,
        68.75,
    );
}

#[test]
fn integer_count_weights_equal_row_repetition() {
    let mut fit = full();
    fit.set_weight(5, 3.0).unwrap();
    fit.set_count(7, 0).unwrap();
    assert_fit(
        &fit,
        13,
        [0.633_712_419_772_548_2, 1.538_509_176_894_493_2],
        0.022_458_721_888_403_223,
        49.21,
    );
}

#[test]
fn a_weighted_row_matches_actual_small_repetition() {
    let mut repeated_row = RefreshRow::new(0, vec![1.0, 0.3], 1.7);
    repeated_row.count = 13;
    let weighted = OlsRefresh::from_rows("weighted", 2, vec![repeated_row]).unwrap();
    let expanded = OlsRefresh::from_rows(
        "expanded",
        2,
        (0..13).map(|id| RefreshRow::new(id, vec![1.0, 0.3], 1.7)).collect(),
    )
    .unwrap();
    let a = weighted.snapshot();
    let b = expanded.snapshot();
    assert_eq!(a.n, b.n);
    for (x, y) in a.xtx.iter().zip(&b.xtx) {
        close(*x, *y, "xtx");
    }
    for (x, y) in a.xty.iter().zip(&b.xty) {
        close(*x, *y, "xty");
    }
    close(a.yty, b.yty, "yty");
    weighted.check_against_fresh().unwrap();
}

#[test]
fn maximum_count_is_bounded_and_survives_refresh_mutations() {
    let mut item = RefreshRow::new(1, vec![1.0, 2.0], 3.0);
    item.count = u32::MAX;
    let mut fit = OlsRefresh::new("counts", 2).unwrap();
    fit.append_row("counts", &item).unwrap();
    let n = u64::from(u32::MAX);
    let expected = f64::from(u32::MAX);
    let snapshot = fit.snapshot();
    assert_eq!(snapshot.n, n);
    assert_eq!(snapshot.ledger.effective_n, n);
    assert_eq!(snapshot.xtx, vec![expected, 2.0 * expected, 2.0 * expected, 4.0 * expected]);
    assert_eq!(snapshot.xty, vec![3.0 * expected, 6.0 * expected]);
    assert_eq!(snapshot.yty, 9.0 * expected);
    fit.check_against_fresh().unwrap();

    fit.set_mask(1, true).unwrap();
    assert_eq!(fit.snapshot().n, 0);
    fit.set_mask(1, false).unwrap();
    assert_eq!(fit.snapshot().n, n);
    fit.set_count(1, 0).unwrap();
    assert_eq!(fit.snapshot().n, 0);
    fit.set_count(1, u32::MAX).unwrap();
    fit.append_window("counts", &[RefreshRow::new(2, vec![1.0, 0.0], 1.0)]).unwrap();
    assert_eq!(fit.snapshot().n, n + 1);
    assert_eq!(fit.ledger().effective_n, n + 1);
    fit.check_against_fresh().unwrap();
    fit.evict_row(1).unwrap();
    assert_eq!(fit.snapshot().n, 1);
    fit.check_against_fresh().unwrap();
}

#[test]
fn a_fractional_weight_is_refused_and_leaves_the_result_unchanged() {
    let mut fit = full();
    let before = fit.snapshot();
    for weight in [0.5, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(fit.set_weight(2, weight), Err(StateError::InvalidEvent(_))));
    }
    assert_eq!(fit.snapshot(), before);
}

#[test]
fn rows_of_another_source_are_refused_and_nothing_changes() {
    let mut fit = OlsRefresh::from_rows("series_a", 2, rows(0..5)).unwrap();
    let before = fit.snapshot();
    let err = fit.append_row("series_b", &row(5)).unwrap_err();
    assert!(matches!(err, StateError::StaleCache(_)));
    assert!(matches!(fit.append_window("series_b", &rows(5..9)), Err(StateError::StaleCache(_))));
    assert_eq!(fit.snapshot(), before);
}

#[test]
fn replacing_the_source_keeps_nothing_of_the_old_one() {
    let mut fit = full();
    let old_digest = fit.snapshot().source_digest;
    assert_eq!(fit.replace_source("series_b", rows(5..9)).unwrap(), RefreshOutcome::Recomputed);
    let snapshot = fit.snapshot();
    assert_eq!(&*snapshot.source, "series_b");
    assert_eq!(snapshot.n, 4);
    assert_ne!(snapshot.source_digest, old_digest);
    fit.check_against_fresh().unwrap();
    // The same rows under the old name digest differently from the replaced source.
    let same_rows = OlsRefresh::from_rows("series_a", 2, rows(5..9)).unwrap();
    assert_ne!(same_rows.snapshot().source_digest, snapshot.source_digest);
}

#[test]
fn a_repeated_observation_is_refused_whether_single_or_in_a_window() {
    let mut fit = OlsRefresh::from_rows("series_a", 2, rows(0..5)).unwrap();
    let before = fit.snapshot();
    assert!(
        matches!(fit.append_row("series_a", &row(2)), Err(StateError::InvalidEvent(ref m)) if m.contains("is already in the summary"))
    );
    assert!(
        matches!(fit.append_window("series_a", &rows(4..8)), Err(StateError::InvalidEvent(ref m)) if m.contains("is already in the summary"))
    );
    assert!(
        matches!(fit.append_window("series_a", &[row(8), row(8)]), Err(StateError::InvalidEvent(ref m)) if m.contains("is already in the summary"))
    );
    assert_eq!(fit.snapshot(), before);
}

#[test]
fn a_malformed_row_is_refused_before_anything_is_touched() {
    let mut fit = OlsRefresh::from_rows("series_a", 2, rows(0..5)).unwrap();
    let before = fit.snapshot();
    assert!(matches!(
        fit.append_row("series_a", &RefreshRow::new(9, vec![1.0], 1.0)),
        Err(StateError::Shape(ref m)) if m == "row len 1 != ncols 2"
    ));
    assert!(matches!(
        fit.append_row("series_a", &RefreshRow::new(9, vec![1.0, f64::NAN], 1.0)),
        Err(StateError::Numerical(ref m)) if m.contains("must be finite")
    ));
    assert_eq!(fit.snapshot(), before);
    assert!(matches!(fit.insert_column(), Err(StateError::InvalidEvent(_))));
}

#[test]
fn summary_only_appends_but_refuses_every_change_that_needs_rows() {
    let mut fit = OlsRefresh::from_rows("series_a", 2, rows(0..5)).unwrap().into_summary_only();
    assert!(fit.rows().is_empty());
    fit.append_window("series_a", &rows(5..12)).unwrap();
    close(fit.snapshot().beta.unwrap()[1], 1.612_687_813_021_702_4, "slope");
    let before = fit.snapshot();
    assert!(
        matches!(fit.evict_row(1), Err(StateError::InvalidEvent(ref m)) if m.contains("summary-only cannot serve")),
        "summary-only keeps no rows"
    );
    assert!(
        matches!(fit.set_mask(1, true), Err(StateError::InvalidEvent(ref m)) if m.contains("summary-only cannot serve")),
        "summary-only keeps no rows"
    );
    assert!(
        matches!(fit.set_count(1, 2), Err(StateError::InvalidEvent(ref m)) if m.contains("summary-only cannot serve")),
        "summary-only keeps no rows"
    );
    assert!(
        matches!(fit.fresh_snapshot(), Err(StateError::InvalidEvent(ref m)) if m.contains("summary-only cannot serve")),
        "summary-only keeps no rows"
    );
    assert!(
        matches!(fit.check_against_fresh(), Err(StateError::InvalidEvent(ref m)) if m.contains("summary-only cannot serve")),
        "summary-only keeps no rows"
    );
    assert_eq!(fit.snapshot(), before);
}

#[test]
fn the_summary_type_agrees_with_the_route_contract_table() {
    let mut stats = LinearOlsSuffStats::new(2).expect("dimension");
    for id in 0..5 {
        stats.append_row(&[1.0, X[id]], Y[id]).unwrap();
    }
    let kept = stats.clone();
    let table = |change| route_contract(RefreshRoute::LinearOlsGram, change);
    assert_eq!(table(RefreshChange::AppendRows), RefreshDisposition::ExactUpdate);
    stats.apply_delta(LinearOlsDelta::AppendRow { row: &[1.0, X[5]], y: Y[5] }).unwrap();
    assert_eq!(stats.n(), 6);
    // Evict is not applied in place, and the table says recompute.
    let mut evicted = kept.clone();
    assert!(matches!(
        evicted.apply_delta(LinearOlsDelta::EvictRow { row: &[1.0, X[0]], y: Y[0] }),
        Err(StateError::Numerical(ref m)) if m.contains("cannot evict a summarized row")
    ));
    assert_eq!(evicted, kept);
    assert!(matches!(
        table(RefreshChange::EvictRow),
        RefreshDisposition::InvalidateAndRecompute { .. }
    ));
    for (delta, change) in [
        (LinearOlsDelta::WeightsChanged, RefreshChange::WeightChange),
        (LinearOlsDelta::MaskChanged, RefreshChange::MaskChange),
    ] {
        let mut weighted = kept.clone();
        assert!(matches!(
            weighted.apply_delta(delta),
            Err(StateError::Numerical(ref m)) if m.contains("invalidates the summary")
        ));
        assert!(weighted.is_invalidated(), "{change:?} must not leave a stale summary");
        assert!(matches!(table(change), RefreshDisposition::InvalidateAndRecompute { .. }));
    }
}
