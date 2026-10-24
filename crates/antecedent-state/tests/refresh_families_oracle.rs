//! Weighted, scatter and conjugate-block refresh against fresh reconstruction, and budgeted
//! disk-backed retention of each.
//!
//! References are closed forms computed in this file from the stated rows (design `[1, x]`),
//! not values read back from the code under test.

use std::path::PathBuf;

use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{
    ConjugateBlockRefresh, ConjugateOlsPosterior, ConjugateOlsPrior, CovarianceRefresh,
    CovarianceRow, LinearOlsSuffStats, OlsRefresh, RefreshChange, RefreshDisposition,
    RefreshOutcome, RefreshRoute, RefreshRow, SpillRegistry, StateError, WeightedOlsRefresh,
    WeightedRow, route_contract,
};

const X: [f64; 12] = [0.5, 1.2, -0.3, 2.0, 1.5, -1.0, 0.8, 2.2, 0.1, 1.7, -0.6, 0.3];
const Y: [f64; 12] = [1.1, 2.4, 0.2, 3.9, 3.1, -0.9, 1.7, 4.6, 0.9, 3.2, -0.1, 1.0];

fn weight(id: usize) -> f64 {
    0.5 + 0.25 * (id % 4) as f64
}

fn wrow(id: usize) -> WeightedRow {
    WeightedRow::weighted(id as u64, vec![1.0, X[id]], Y[id], weight(id))
}

fn wrows(range: std::ops::Range<usize>) -> Vec<WeightedRow> {
    range.map(wrow).collect()
}

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
        "{label}: {actual:.17e} vs {expected:.17e}"
    );
}

/// Closed-form weighted line `(intercept, slope, sigma2, live)` over `(x, y, w)` triples.
fn reference(points: &[(f64, f64, f64)]) -> (f64, f64, f64, usize) {
    let live: Vec<_> = points.iter().filter(|p| p.2 > 0.0).collect();
    let sw: f64 = live.iter().map(|p| p.2).sum();
    let sx: f64 = live.iter().map(|p| p.2 * p.0).sum();
    let sy: f64 = live.iter().map(|p| p.2 * p.1).sum();
    let sxx: f64 = live.iter().map(|p| p.2 * p.0 * p.0).sum();
    let sxy: f64 = live.iter().map(|p| p.2 * p.0 * p.1).sum();
    let slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
    let intercept = (sy - slope * sx) / sw;
    let sse: f64 = live.iter().map(|p| p.2 * (p.1 - intercept - slope * p.0).powi(2)).sum();
    (intercept, slope, sse / (live.len() as f64 - 2.0), live.len())
}

fn points(ids: &[usize], weights: impl Fn(usize) -> f64) -> Vec<(f64, f64, f64)> {
    ids.iter().map(|&i| (X[i], Y[i], weights(i))).collect()
}

fn assert_weighted(fit: &WeightedOlsRefresh, expected: (f64, f64, f64, usize)) {
    let snapshot = fit.snapshot();
    let beta = snapshot.beta.as_ref().expect("beta");
    close(beta[0], expected.0, "intercept");
    close(beta[1], expected.1, "slope");
    close(snapshot.residual_variance.expect("sigma2"), expected.2, "residual variance");
    assert_eq!(snapshot.ledger.live_rows as usize, expected.3);
    fit.check_against_fresh().expect("cached result equals the fresh rebuild");
}

fn scratch(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("antecedent-refresh-families-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

#[test]
fn fractional_weights_append_and_window_equal_the_closed_form_without_a_rebuild() {
    let mut fit = WeightedOlsRefresh::new("series_a", 2).unwrap();
    for id in 0..4 {
        assert_eq!(fit.append_row("series_a", &wrow(id)).unwrap(), RefreshOutcome::Updated);
    }
    assert_eq!(fit.append_window("series_a", &wrows(4..9)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(fit.append_window("series_a", &wrows(9..12)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(fit.recomputes(), 0);
    let all: Vec<usize> = (0..12).collect();
    assert_weighted(&fit, reference(&points(&all, weight)));
    let snapshot = fit.snapshot();
    let total: f64 = all.iter().map(|&i| weight(i)).sum();
    let squares: f64 = all.iter().map(|&i| weight(i).powi(2)).sum();
    close(snapshot.total_weight, total, "total weight");
    close(snapshot.effective_sample_size, total * total / squares, "effective sample size");
    let se = snapshot.beta_se.expect("standard errors");
    assert!(se.iter().all(|value| value.is_finite() && *value > 0.0));
}

#[test]
fn every_weight_and_update_form_recomputes_to_the_closed_form_on_the_rest() {
    let all: Vec<usize> = (0..12).collect();

    let mut fit = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    assert_eq!(fit.set_weight(3, 2.75).unwrap(), RefreshOutcome::Recomputed);
    assert_weighted(&fit, reference(&points(&all, |i| if i == 3 { 2.75 } else { weight(i) })));

    assert_eq!(fit.evict_row(5).unwrap(), RefreshOutcome::Recomputed);
    let rest: Vec<usize> = all.iter().copied().filter(|&i| i != 5).collect();
    assert_weighted(&fit, reference(&points(&rest, |i| if i == 3 { 2.75 } else { weight(i) })));
    assert!(fit.evict_row(5).is_err(), "the row is gone");

    let mut masked = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    masked.set_mask(0, true).unwrap();
    masked.set_weight(1, 0.0).unwrap();
    let live: Vec<usize> = (2..12).collect();
    assert_weighted(&masked, reference(&points(&live, weight)));
    assert_eq!((masked.ledger().rows, masked.ledger().live_rows), (12, 10));
    masked.set_mask(0, false).unwrap();
    masked.set_weight(1, weight(1)).unwrap();
    assert_weighted(&masked, reference(&points(&all, weight)));

    let mut windowed = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    assert_eq!(windowed.retain_last(7).unwrap(), RefreshOutcome::Recomputed);
    let tail: Vec<usize> = (5..12).collect();
    assert_weighted(&windowed, reference(&points(&tail, weight)));
    assert_eq!(windowed.rows().len(), 7);
    windowed.append_row("s", &wrow(0)).expect("an id that left the window may return");
    windowed.check_against_fresh().unwrap();

    let mut replaced = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    let new_weights: Vec<f64> = (0..12).map(|i| 1.0 + 0.1 * i as f64).collect();
    replaced.replace_weights(&new_weights).unwrap();
    assert_weighted(&replaced, reference(&points(&all, |i| 1.0 + 0.1 * i as f64)));
    assert!(replaced.replace_weights(&new_weights[..3]).is_err());
}

#[test]
fn scaling_every_weight_keeps_the_fit_and_scales_the_variance_and_the_weight_total() {
    let mut fit = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    let before = fit.snapshot();
    assert_eq!(fit.scale_weights(3.0).unwrap(), RefreshOutcome::Recomputed);
    fit.check_against_fresh().unwrap();
    let after = fit.snapshot();
    let (b0, b1) = (before.beta.as_ref().unwrap(), after.beta.as_ref().unwrap());
    close(b1[0], b0[0], "intercept");
    close(b1[1], b0[1], "slope");
    close(after.residual_variance.unwrap(), 3.0 * before.residual_variance.unwrap(), "sigma2");
    close(after.total_weight, 3.0 * before.total_weight, "total weight");
    close(after.effective_sample_size, before.effective_sample_size, "effective n");
    assert_ne!(after.source_digest, before.source_digest, "the weights are part of the identity");
    assert!(fit.scale_weights(-1.0).is_err());
    assert!(fit.scale_weights(f64::NAN).is_err());
}

#[test]
fn integer_weights_give_the_same_fit_as_the_count_form() {
    let mut counted = OlsRefresh::from_rows(
        "s",
        2,
        (0..12).map(|i| RefreshRow::new(i as u64, vec![1.0, X[i]], Y[i])).collect(),
    )
    .unwrap();
    counted.set_count(5, 3).unwrap();
    counted.set_count(7, 0).unwrap();
    let mut weighted = WeightedOlsRefresh::from_rows(
        "s",
        2,
        (0..12).map(|i| WeightedRow::new(i as u64, vec![1.0, X[i]], Y[i])).collect(),
    )
    .unwrap();
    weighted.set_weight(5, 3.0).unwrap();
    weighted.set_weight(7, 0.0).unwrap();
    let (a, b) = (counted.snapshot(), weighted.snapshot());
    for (left, right) in a.beta.unwrap().iter().zip(b.beta.unwrap().iter()) {
        close(*left, *right, "coefficient");
    }
    for (left, right) in a.xtx.iter().zip(&b.xtx) {
        close(*left, *right, "gram");
    }
    close(a.yty, b.yty, "yty");
    close(a.n as f64, b.total_weight, "count equals total weight");
}

#[test]
fn weighted_refresh_follows_the_route_contract_for_each_change() {
    let table = |change| route_contract(RefreshRoute::LinearOlsGram, change);
    let mut fit = WeightedOlsRefresh::from_rows("s", 2, wrows(0..8)).unwrap();
    assert_eq!(table(RefreshChange::AppendRows), RefreshDisposition::ExactUpdate);
    assert_eq!(fit.append_row("s", &wrow(8)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(table(RefreshChange::AppendWindow), RefreshDisposition::ExactUpdate);
    assert_eq!(fit.append_window("s", &wrows(9..12)).unwrap(), RefreshOutcome::Updated);
    let recompute =
        |change| matches!(table(change), RefreshDisposition::InvalidateAndRecompute { .. });
    assert!(recompute(RefreshChange::EvictRow));
    assert_eq!(fit.evict_row(0).unwrap(), RefreshOutcome::Recomputed);
    assert!(recompute(RefreshChange::WeightChange));
    assert_eq!(fit.set_weight(1, 0.9).unwrap(), RefreshOutcome::Recomputed);
    assert!(recompute(RefreshChange::MaskChange));
    assert_eq!(fit.set_mask(2, true).unwrap(), RefreshOutcome::Recomputed);
    assert!(recompute(RefreshChange::SourceChanged));
    assert_eq!(fit.replace_source("t", wrows(0..6)).unwrap(), RefreshOutcome::Recomputed);
    assert!(recompute(RefreshChange::ColumnInserted));
    assert!(matches!(fit.insert_column(), Err(StateError::InvalidEvent(_))));
    fit.check_against_fresh().unwrap();
    assert_eq!(fit.recomputes(), 4);
}

#[test]
fn weighted_refusals_leave_the_result_unchanged() {
    let mut fit = WeightedOlsRefresh::from_rows("a", 2, wrows(0..6)).unwrap();
    let before = fit.snapshot();
    for bad in [-0.5, f64::NAN, f64::INFINITY] {
        assert!(matches!(fit.set_weight(2, bad), Err(StateError::InvalidEvent(_))));
        assert!(fit.append_row("a", &WeightedRow::weighted(99, vec![1.0, 0.0], 1.0, bad)).is_err());
    }
    assert!(matches!(fit.append_row("b", &wrow(7)), Err(StateError::StaleCache(_))));
    assert!(matches!(fit.append_window("b", &wrows(7..9)), Err(StateError::StaleCache(_))));
    assert!(fit.append_row("a", &wrow(2)).is_err(), "repeated id");
    assert!(fit.append_window("a", &[wrow(8), wrow(8)]).is_err(), "repeated id in a window");
    assert!(fit.append_row("a", &WeightedRow::new(50, vec![1.0], 1.0)).is_err());
    assert!(fit.append_row("a", &WeightedRow::new(51, vec![1.0, f64::NAN], 1.0)).is_err());
    assert_eq!(fit.snapshot(), before);
    fit.check_against_fresh().unwrap();
}

#[test]
fn summary_only_weighted_state_appends_but_refuses_what_needs_rows() {
    let mut fit = WeightedOlsRefresh::from_rows("s", 2, wrows(0..5)).unwrap().into_summary_only();
    assert_eq!(fit.retention(), RetentionMode::SummaryOnly);
    assert!(fit.rows().is_empty());
    fit.append_window("s", &wrows(5..12)).unwrap();
    let expected = reference(&points(&(0..12).collect::<Vec<_>>(), weight));
    close(fit.snapshot().beta.unwrap()[1], expected.1, "slope");
    let before = fit.snapshot();
    assert!(fit.evict_row(1).is_err());
    assert!(fit.set_mask(1, true).is_err());
    assert!(fit.set_weight(1, 2.0).is_err());
    assert!(fit.scale_weights(2.0).is_err());
    assert!(fit.retain_last(3).is_err());
    assert!(fit.refresh().is_err());
    assert!(fit.fresh_snapshot().is_err());
    assert!(fit.check_against_fresh().is_err());
    let registry = SpillRegistry::new(scratch("summary_only"), 1 << 20).unwrap();
    assert!(fit.spill(&registry, "x").is_err(), "no rows to spill");
    assert_eq!(fit.snapshot(), before);
}

#[test]
fn a_spilled_weighted_state_reloads_to_the_same_result_and_keeps_refreshing() {
    let directory = scratch("weighted_roundtrip");
    let registry = SpillRegistry::new(&directory, 1 << 20).unwrap();
    let fit = WeightedOlsRefresh::from_rows("s", 2, wrows(0..12)).unwrap();
    let handle = fit.spill(&registry, "fit").unwrap();
    assert_eq!(handle.mode(), RetentionMode::DiskBacked);
    assert!(handle.bytes() > 0 && registry.stats().used_bytes == handle.bytes());
    let mut reloaded: WeightedOlsRefresh = handle.reload().unwrap();
    assert_eq!(reloaded.snapshot(), fit.snapshot());
    assert_eq!(reloaded.retention(), RetentionMode::FullReusable);
    reloaded.set_weight(4, 1.9).unwrap();
    reloaded.append_row("s", &WeightedRow::weighted(40, vec![1.0, 0.7], 1.5, 0.6)).unwrap();
    reloaded.check_against_fresh().unwrap();
    assert_ne!(reloaded.snapshot(), fit.snapshot());
    assert_eq!(registry.stats().reloads, 1);
}

#[test]
fn a_spilled_result_serves_only_its_artifact_read() {
    let directory = scratch("unsupported");
    let registry = SpillRegistry::new(&directory, 1 << 20).unwrap();
    let handle =
        WeightedOlsRefresh::from_rows("s", 2, wrows(0..6)).unwrap().spill(&registry, "k").unwrap();
    handle.require(RetainedOperation::ReadDiskArtifact).unwrap();
    for operation in RetentionMode::DiskBacked.no_longer_serves() {
        let error = handle.require(*operation).unwrap_err();
        assert!(error.to_string().contains("reload the artifact"), "{error}");
    }
    // Asking for the wrong kind of state is refused, not mis-decoded.
    assert!(registry.reload::<CovarianceRefresh>("k").is_err());
}

#[test]
fn spilled_results_are_evicted_under_the_byte_budget_and_the_evicted_one_must_be_rebuilt() {
    let directory = scratch("budget");
    let one = {
        let probe = SpillRegistry::new(scratch("budget_probe"), 1 << 20).unwrap();
        WeightedOlsRefresh::from_rows("s", 2, wrows(0..12))
            .unwrap()
            .spill(&probe, "probe")
            .unwrap()
            .bytes()
    };
    let registry = SpillRegistry::new(&directory, one * 2 + one / 2).unwrap();
    let make = |source: &str| WeightedOlsRefresh::from_rows(source, 2, wrows(0..12)).unwrap();
    let first = make("s0").spill(&registry, "s0").unwrap();
    let second = make("s1").spill(&registry, "s1").unwrap();
    // Reload s0 so s1 is the least recently used.
    let _: WeightedOlsRefresh = first.reload().unwrap();
    let third = make("s2").spill(&registry, "s2").unwrap();
    assert_eq!(third.evicted(), ["s1"]);
    assert!(registry.stats().used_bytes <= registry.budget_bytes());
    assert!(!second.on_disk());
    let error = second.reload::<WeightedOlsRefresh>().unwrap_err();
    assert!(matches!(error, StateError::UnknownId(_)), "{error}");
    assert!(error.to_string().contains("rebuild it from the source rows"));
    // The remaining two still reload to their own sources.
    assert_eq!(first.reload::<WeightedOlsRefresh>().unwrap().source(), "s0");
    assert_eq!(third.reload::<WeightedOlsRefresh>().unwrap().source(), "s2");
    // Rebuilt from the source rows, the evicted state is the same result as before.
    let rebuilt = make("s1");
    rebuilt.check_against_fresh().unwrap();
    assert_eq!(registry.stats().evictions, 1);
    // A state larger than the whole budget is refused and evicts nothing.
    let big = WeightedOlsRefresh::from_rows(
        "big",
        2,
        (0..400).map(|i| WeightedRow::new(i, vec![1.0, i as f64], 1.0 + (i % 5) as f64)).collect(),
    )
    .unwrap();
    assert!(matches!(big.spill(&registry, "big"), Err(StateError::CacheBudget { .. })));
    assert_eq!(registry.stats().evictions, 1);
    assert!(first.remove());
    assert!(registry.stats().used_bytes < one * 2 + one / 2);
}

#[test]
fn a_spilled_file_that_was_changed_or_truncated_is_refused() {
    let directory = scratch("tamper");
    let registry = SpillRegistry::new(&directory, 1 << 20).unwrap();
    let handle =
        WeightedOlsRefresh::from_rows("s", 2, wrows(0..8)).unwrap().spill(&registry, "k").unwrap();
    let path = std::fs::read_dir(&directory).unwrap().next().unwrap().unwrap().path();
    let original = std::fs::read(&path).unwrap();
    let mut changed = original.clone();
    let at = changed.len() / 3;
    changed[at] ^= 0x01;
    std::fs::write(&path, &changed).unwrap();
    assert!(matches!(handle.reload::<WeightedOlsRefresh>(), Err(StateError::StaleCache(_))));
    std::fs::write(&path, &original[..original.len() - 9]).unwrap();
    assert!(handle.reload::<WeightedOlsRefresh>().is_err());
    std::fs::write(&path, &original).unwrap();
    assert!(handle.reload::<WeightedOlsRefresh>().is_ok(), "the original bytes reload");
}

#[test]
fn spilled_count_form_state_reloads_and_matches_a_fresh_rebuild() {
    let registry = SpillRegistry::new(scratch("count_form"), 1 << 20).unwrap();
    let mut fit = OlsRefresh::from_rows(
        "s",
        2,
        (0..12).map(|i| RefreshRow::new(i as u64, vec![1.0, X[i]], Y[i])).collect(),
    )
    .unwrap();
    fit.set_count(2, 3).unwrap();
    fit.set_mask(4, true).unwrap();
    let handle = fit.spill(&registry, "ols").unwrap();
    let mut back: OlsRefresh = handle.reload().unwrap();
    assert_eq!(back.snapshot(), fit.snapshot());
    back.evict_row(0).unwrap();
    back.check_against_fresh().unwrap();
    let summary_only = fit.clone().into_summary_only();
    assert!(summary_only.spill(&registry, "none").is_err());
}

fn zrow(id: usize) -> CovarianceRow {
    CovarianceRow::new(id as u64, vec![X[id], Y[id], X[id] * Y[id]])
}

fn zrows(range: std::ops::Range<usize>) -> Vec<CovarianceRow> {
    range.map(zrow).collect()
}

/// Sample covariance of the selected rows, by the two-pass definition.
fn two_pass(ids: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let rows: Vec<[f64; 3]> = ids.iter().map(|&i| [X[i], Y[i], X[i] * Y[i]]).collect();
    let n = rows.len() as f64;
    let mean: Vec<f64> = (0..3).map(|c| rows.iter().map(|r| r[c]).sum::<f64>() / n).collect();
    let mut cov = vec![0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            cov[i * 3 + j] =
                rows.iter().map(|r| (r[i] - mean[i]) * (r[j] - mean[j])).sum::<f64>() / (n - 1.0);
        }
    }
    (mean, cov)
}

fn assert_scatter(fit: &CovarianceRefresh, ids: &[usize]) {
    let (mean, cov) = two_pass(ids);
    let snapshot = fit.snapshot();
    assert_eq!(snapshot.n as usize, ids.len());
    for (a, b) in snapshot.mean.iter().zip(&mean) {
        close(*a, *b, "mean");
    }
    for (a, b) in snapshot.sample_covariance.as_ref().unwrap().iter().zip(&cov) {
        close(*a, *b, "covariance");
    }
    fit.check_against_fresh().expect("cached scatter equals the fresh rebuild");
}

#[test]
fn scatter_refresh_matches_the_two_pass_definition_after_every_mutation() {
    let all: Vec<usize> = (0..12).collect();
    let mut fit = CovarianceRefresh::new("s", 3).unwrap();
    for id in 0..5 {
        assert_eq!(fit.append_row("s", &zrow(id)).unwrap(), RefreshOutcome::Updated);
    }
    assert_eq!(fit.append_window("s", &zrows(5..12)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(fit.recomputes(), 0);
    assert_scatter(&fit, &all);

    assert_eq!(fit.evict_row(3).unwrap(), RefreshOutcome::Recomputed);
    let rest: Vec<usize> = all.iter().copied().filter(|&i| i != 3).collect();
    assert_scatter(&fit, &rest);

    fit.set_mask(0, true).unwrap();
    let live: Vec<usize> = rest.iter().copied().filter(|&i| i != 0).collect();
    assert_scatter(&fit, &live);
    assert_eq!((fit.ledger().rows, fit.ledger().live_rows), (11, 10));

    fit.retain_last(6).unwrap();
    assert_scatter(&fit, &[6, 7, 8, 9, 10, 11]);

    assert_eq!(fit.replace_source("t", zrows(2..9)).unwrap(), RefreshOutcome::Recomputed);
    assert_scatter(&fit, &(2..9).collect::<Vec<_>>());
    assert!(matches!(fit.append_row("s", &zrow(11)), Err(StateError::StaleCache(_))));
    assert!(fit.append_row("t", &zrow(2)).is_err(), "repeated id");
    assert!(fit.append_row("t", &CovarianceRow::new(70, vec![1.0])).is_err());
    assert!(fit.evict_row(999).is_err());
    fit.check_against_fresh().unwrap();
}

#[test]
fn scatter_summary_only_and_spill_behave_as_declared() {
    let registry = SpillRegistry::new(scratch("scatter"), 1 << 20).unwrap();
    let fit = CovarianceRefresh::from_rows("s", 3, zrows(0..9)).unwrap();
    let handle = fit.spill(&registry, "cov").unwrap();
    let mut back: CovarianceRefresh = handle.reload().unwrap();
    assert_eq!(back.snapshot(), fit.snapshot());
    back.set_mask(2, true).unwrap();
    back.check_against_fresh().unwrap();

    let mut summary = fit.clone().into_summary_only();
    summary.append_window("s", &zrows(9..12)).unwrap();
    assert_eq!(summary.snapshot().n, 12);
    assert!(summary.evict_row(1).is_err());
    assert!(summary.set_mask(1, true).is_err());
    assert!(summary.check_against_fresh().is_err());
    assert!(summary.spill(&registry, "none").is_err());
}

fn block_rows(range: std::ops::Range<usize>) -> Vec<RefreshRow> {
    range.map(|i| RefreshRow::new(i as u64, vec![1.0, X[i]], Y[i])).collect()
}

fn pooled(ids: std::ops::Range<usize>) -> LinearOlsSuffStats {
    let mut stats = LinearOlsSuffStats::new(2).unwrap();
    for i in ids {
        stats.append_row(&[1.0, X[i]], Y[i]).unwrap();
    }
    stats
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::isotropic(2, 4.0, 3.0, 2.0).unwrap()
}

fn batch_summary(ids: std::ops::Range<usize>, under: &ConjugateOlsPrior) -> Vec<f64> {
    let mut posterior = ConjugateOlsPosterior::new(under);
    posterior.update(under, "all", &pooled(ids)).unwrap();
    let summary = posterior.summary(under).unwrap();
    let mut out = summary.mean.clone();
    out.extend([summary.shape, summary.rate]);
    out
}

fn flat(refresh: &ConjugateBlockRefresh) -> Vec<f64> {
    let summary = refresh.summary().unwrap();
    let mut out = summary.mean.clone();
    out.extend([summary.shape, summary.rate]);
    out
}

fn assert_flat(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        close(*a, *b, "posterior");
    }
}

#[test]
fn conjugate_blocks_update_in_place_and_rebuild_to_the_pooled_posterior() {
    let mut refresh = ConjugateBlockRefresh::new(prior());
    assert_eq!(refresh.add_block("b1", "s", block_rows(0..5)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(refresh.add_block("b2", "s", block_rows(5..9)).unwrap(), RefreshOutcome::Updated);
    assert_eq!(refresh.recomputes(), 0);
    assert_flat(&flat(&refresh), &batch_summary(0..9, &prior()));
    refresh.check_against_fresh().unwrap();

    assert_eq!(
        refresh.extend_block("b2", "s", &block_rows(9..12)).unwrap(),
        RefreshOutcome::Recomputed
    );
    assert_flat(&flat(&refresh), &batch_summary(0..12, &prior()));
    refresh.check_against_fresh().unwrap();

    assert_eq!(refresh.evict_row("b1", 0).unwrap(), RefreshOutcome::Recomputed);
    let mut expected = LinearOlsSuffStats::new(2).unwrap();
    for i in 1..12 {
        expected.append_row(&[1.0, X[i]], Y[i]).unwrap();
    }
    let mut posterior = ConjugateOlsPosterior::new(&prior());
    posterior.update(&prior(), "all", &expected).unwrap();
    let s = posterior.summary(&prior()).unwrap();
    assert_flat(&flat(&refresh), &[s.mean[0], s.mean[1], s.shape, s.rate]);

    refresh.set_count("b1", 1, 2).unwrap();
    refresh.set_mask("b2", 6, true).unwrap();
    refresh.check_against_fresh().unwrap();
    assert_eq!(refresh.summary().unwrap().n, 11 + 1 - 1);
    assert_eq!(refresh.labels().collect::<Vec<_>>(), ["b1", "b2"]);
}

#[test]
fn conjugate_reuse_prior_change_and_emptied_blocks_follow_the_contract() {
    let mut refresh = ConjugateBlockRefresh::new(prior());
    refresh.add_block("b1", "s", block_rows(0..6)).unwrap();
    let before = flat(&refresh);
    // The same label, and an empty block, are refused and change nothing.
    assert!(refresh.add_block("b1", "s", block_rows(6..9)).is_err());
    assert!(refresh.add_block("empty", "s", Vec::new()).is_err());
    assert!(refresh.extend_block("missing", "s", &block_rows(6..9)).is_err());
    assert!(matches!(
        refresh.extend_block("b1", "other", &block_rows(6..9)),
        Err(StateError::StaleCache(_))
    ));
    assert_flat(&flat(&refresh), &before);
    assert!(matches!(
        route_contract(RefreshRoute::ConjugateGaussianPosterior, RefreshChange::ObservationsReused),
        RefreshDisposition::Refuse { .. }
    ));

    // A changed prior rebuilds from the retained blocks and equals the batch under it.
    let other = ConjugateOlsPrior::isotropic(2, 0.5, 5.0, 1.0).unwrap();
    assert_eq!(refresh.change_prior(other.clone()).unwrap(), RefreshOutcome::Recomputed);
    assert_flat(&flat(&refresh), &batch_summary(0..6, &other));
    refresh.check_against_fresh().unwrap();
    assert!(refresh.change_prior(ConjugateOlsPrior::isotropic(3, 1.0, 2.0, 1.0).unwrap()).is_err());

    // Emptying the only block leaves the prior as the posterior, not a stale result.
    for id in 0..6 {
        refresh.evict_row("b1", id).unwrap();
    }
    assert_eq!(refresh.summary().unwrap().n, 0);
    refresh.check_against_fresh().unwrap();
}

#[test]
fn conjugate_state_spills_and_reloads_with_the_same_posterior_and_identity() {
    let registry = SpillRegistry::new(scratch("conjugate"), 1 << 20).unwrap();
    let mut refresh = ConjugateBlockRefresh::new(prior());
    refresh.add_block("b1", "s", block_rows(0..5)).unwrap();
    refresh.add_block("b2", "s", block_rows(5..12)).unwrap();
    let handle = refresh.spill(&registry, "posterior").unwrap();
    let mut back: ConjugateBlockRefresh = handle.reload().unwrap();
    assert_eq!(back.identity(), refresh.identity());
    assert_flat(&flat(&back), &flat(&refresh));
    back.evict_row("b2", 7).unwrap();
    back.check_against_fresh().unwrap();
    assert_ne!(back.identity(), refresh.identity());
}
