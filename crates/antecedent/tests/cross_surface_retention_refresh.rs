//! A facade consumer of refreshed, reduced-retention and spilled results: each mutation of a
//! prepared least-squares summary equals a fresh reconstruction, reduced retention refuses the
//! operations it cannot serve with the next attempt, and a spilled result reloads to the same
//! answer under a byte budget.

use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{
    RefreshOutcome, SpillRegistry, StateError, WeightedOlsRefresh, WeightedRow,
};

fn rows(n: usize) -> Vec<WeightedRow> {
    (0..n)
        .map(|i| {
            let x = 0.3 * i as f64 - 1.0;
            let y = 1.0 + 2.0 * x + ((i * 7) % 5) as f64 * 0.1;
            WeightedRow::weighted(i as u64, vec![1.0, x], y, 0.5 + (i % 3) as f64 * 0.5)
        })
        .collect()
}

#[test]
fn prepared_summary_mutations_equal_fresh_reconstruction_and_reduced_retention_refuses() {
    let mut fit = WeightedOlsRefresh::from_rows("prepared", 2, rows(10)).unwrap();
    assert_eq!(fit.append_window("prepared", &rows(16)[10..]).unwrap(), RefreshOutcome::Updated);
    fit.check_against_fresh().unwrap();
    assert_eq!(fit.set_weight(3, 4.5).unwrap(), RefreshOutcome::Recomputed);
    assert_eq!(fit.evict_row(1).unwrap(), RefreshOutcome::Recomputed);
    fit.check_against_fresh().unwrap();
    let fresh = fit.fresh_snapshot().unwrap();
    assert_eq!(fit.snapshot().source_digest, fresh.source_digest);

    let summary = fit.clone().into_summary_only();
    assert_eq!(summary.retention(), RetentionMode::SummaryOnly);
    let error = summary.clone().evict_row(2).unwrap_err();
    let StateError::InvalidEvent(text) = error else { panic!("expected a refusal") };
    assert!(text.contains("summary-only") && text.contains("rebuild"), "{text}");
    for operation in RetentionMode::SummaryOnly.no_longer_serves() {
        assert!(RetentionMode::SummaryOnly.require(*operation).is_err());
    }
    assert!(RetentionMode::SummaryOnly.require(RetainedOperation::ReuseSummary).is_ok());
}

#[test]
fn spilled_prepared_summary_reloads_under_a_budget_and_refuses_resident_operations() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("antecedent-facade-spill-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fit = WeightedOlsRefresh::from_rows("prepared", 2, rows(12)).unwrap();
    let probe = SpillRegistry::new(dir.join("probe"), 1 << 20).unwrap();
    let one = fit.spill(&probe, "p").unwrap().bytes();
    let registry = SpillRegistry::new(dir.join("budget"), one + one / 2).unwrap();
    let first = fit.spill(&registry, "first").unwrap();
    let second = fit.spill(&registry, "second").unwrap();
    assert_eq!(second.evicted(), ["first"]);
    assert!(first.reload::<WeightedOlsRefresh>().is_err(), "evicted: rebuild from the rows");
    let mut back: WeightedOlsRefresh = second.reload().unwrap();
    assert_eq!(back.snapshot(), fit.snapshot());
    back.set_weight(0, 2.0).unwrap();
    back.check_against_fresh().unwrap();
    let refusal = second.require(RetainedOperation::RefitFromRetainedRows).unwrap_err();
    assert!(refusal.to_string().contains("reload"), "{refusal}");
    drop((first, second, registry, probe));
    let _ = std::fs::remove_dir_all(&dir);
}
