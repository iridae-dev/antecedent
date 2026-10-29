//! `new_period_refresh`: semantic refresh of a held horizon-two result onto a new period.
//! The evaluator here is a deterministic stand-in (the mean of the snapshot's unit outcomes)
//! so the tests isolate the identity, invalidation and receipt contract.

use std::collections::BTreeSet;

use antecedent_estimate::EstimationError;
use antecedent_estimate::posterior_mean::{
    PosteriorRequest, ScalarNamespace, normal_mean_posterior,
};
use antecedent_estimate::temporal_refresh::{
    HeldTemporalResult, ObservationPeriod, RefreshDecision, RefreshInvalidation, RefreshReceipt,
    TemporalWindowIdentity, accept_refresh, decide_refresh, refresh_held,
};

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn identity() -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "two_slice_graph_v1".into(),
        horizon: 2,
        lag_alignment: vec![("x".into(), 1), ("a".into(), 1), ("y".into(), 2)],
        intervention_history: vec!["a1=1".into(), "a2=0".into()],
        selection_targets: set(&["s_t1", "s_t2"]),
        regimes: set(&["observational", "experiment"]),
        period: ObservationPeriod { start: 0, end: 10 },
        unit_ids: set(&["u1", "u2", "u3"]),
        snapshot_id: "snap_old".into(),
        proof_id: "proof_1".into(),
    }
}

fn replacement() -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        period: ObservationPeriod { start: 10, end: 20 },
        unit_ids: set(&["u1", "u2", "u4"]),
        snapshot_id: "snap_new".into(),
        ..identity()
    }
}

/// Deterministic evaluator: the mean outcome of the snapshot named by the identity.
#[allow(
    clippy::unnecessary_wraps,
    reason = "`refresh_held` requires an evaluator returning `Result`; this one cannot fail"
)]
fn evaluate(id: &TemporalWindowIdentity) -> Result<f64, EstimationError> {
    let outcomes: &[f64] = match id.snapshot_id.as_str() {
        "snap_old" => &[1.0, 2.0, 4.0],
        "snap_new" => &[3.0, 5.0, 10.0],
        _ => &[0.0],
    };
    Ok(outcomes.iter().sum::<f64>() / f64::from(u32::try_from(outcomes.len()).unwrap_or(1)))
}

fn held(interval: bool) -> HeldTemporalResult<f64> {
    let id = identity();
    let value = evaluate(&id).unwrap();
    HeldTemporalResult { identity: id, value, interval_present: interval }
}

fn invalidation_detail(result: Result<impl std::fmt::Debug, EstimationError>) -> String {
    let EstimationError::Refused { code, message } = result.unwrap_err() else {
        panic!("expected a coded refusal");
    };
    assert_eq!(code, "route_not_supported");
    message.into_string()
}

#[test]
fn period_replace_matches_clean_preparation_bit_for_bit() {
    let held = held(true);
    let (refreshed, receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    let clean = evaluate(&replacement()).unwrap();
    assert_eq!(refreshed.value.to_bits(), clean.to_bits());
    assert_eq!(refreshed.identity, replacement());
    // Re-evaluated, not copied: the value moved with the period.
    assert_ne!(refreshed.value.to_bits(), held.value.to_bits());
    assert_eq!(refreshed.value.to_bits(), 6.0_f64.to_bits());
    // The receipt binds both periods, snapshots and the unchanged proof.
    assert_eq!(receipt.old_period, ObservationPeriod { start: 0, end: 10 });
    assert_eq!(receipt.new_period, ObservationPeriod { start: 10, end: 20 });
    assert_eq!(receipt.old_snapshot_id, "snap_old");
    assert_eq!(receipt.new_snapshot_id, "snap_new");
    assert_eq!(receipt.proof_id, "proof_1");
    assert_eq!(receipt.inference_claim, "point_only");
    receipt.verify().unwrap();
    // A held interval did not survive.
    assert!(receipt.interval_invalidated);
    assert!(!refreshed.interval_present);
}

#[test]
fn refresh_posterior_mean_stays_at_the_new_period_response() {
    let outcomes = [3.0, 5.0, 10.0];
    let mean = normal_mean_posterior(
        &outcomes,
        PosteriorRequest::IndependentNormal { scale: 1e6 },
        ScalarNamespace::Refresh,
    )
    .unwrap();
    let refreshed = refresh_held(&held(false), replacement(), evaluate).unwrap().0.value;
    assert!((mean - refreshed).abs() < 1e-6, "{mean} vs {refreshed}");
}

#[test]
fn refresh_posterior_refuses_an_unmeasured_band() {
    let error = normal_mean_posterior(
        &[3.0, 5.0, 10.0],
        PosteriorRequest::UnmeasuredBand,
        ScalarNamespace::Refresh,
    )
    .unwrap_err();
    let EstimationError::Refused { message, .. } = error else { panic!("expected a refusal") };
    assert!(message.contains("refresh_posterior.unmeasured_band"), "{message}");
}

#[test]
fn refresh_posterior_consumer_replays_the_sample_mean() {
    let outcomes = [3.0, 5.0, 10.0];
    let produced = normal_mean_posterior(
        &outcomes,
        PosteriorRequest::IndependentNormal { scale: 10.0 },
        ScalarNamespace::Refresh,
    )
    .unwrap();
    let mean = 6.0;
    let variance = (9.0 + 1.0 + 16.0) / 2.0;
    let weight = (3.0 / variance) / (3.0 / variance + 1.0 / 100.0);
    assert!((produced - weight * mean).abs() < 1e-12);
}

#[test]
fn period_replace_without_prior_interval_records_none() {
    let (refreshed, receipt) = refresh_held(&held(false), replacement(), evaluate).unwrap();
    assert!(!receipt.interval_invalidated);
    assert!(!refreshed.interval_present);
    assert_eq!(decide_refresh(&identity(), &replacement()), RefreshDecision::Reusable);
}

#[test]
fn third_period_appended_slice_or_changed_horizon_invalidates() {
    for horizon in [3, 1] {
        let mut new = replacement();
        new.horizon = horizon;
        assert_eq!(
            decide_refresh(&identity(), &new),
            RefreshDecision::Invalidated(RefreshInvalidation::HorizonChanged)
        );
        let message = invalidation_detail(refresh_held(&held(true), new, evaluate));
        assert!(message.contains("temporal_refresh.horizon_changed"), "{message}");
    }
}

#[test]
fn third_period_changed_lag_alignment_invalidates() {
    let mut new = replacement();
    new.lag_alignment = vec![("x".into(), 1), ("a".into(), 2), ("y".into(), 2)];
    assert_eq!(
        decide_refresh(&identity(), &new),
        RefreshDecision::Invalidated(RefreshInvalidation::LagAlignmentChanged)
    );
    let message = invalidation_detail(refresh_held(&held(false), new, evaluate));
    assert!(message.contains("temporal_refresh.lag_alignment_changed"), "{message}");
}

#[test]
fn third_period_altered_intervention_history_invalidates() {
    let mut new = replacement();
    new.intervention_history = vec!["a1=1".into(), "a2=1".into()];
    assert_eq!(
        decide_refresh(&identity(), &new),
        RefreshDecision::Invalidated(RefreshInvalidation::InterventionHistoryChanged)
    );
    let message = invalidation_detail(refresh_held(&held(false), new, evaluate));
    assert!(message.contains("temporal_refresh.intervention_history_changed"), "{message}");
}

#[test]
fn third_period_changed_selection_regimes_graph_or_proof_invalidates() {
    let edits: [fn(&mut TemporalWindowIdentity); 4] = [
        |i| i.selection_targets = set(&["s_t1"]),
        |i| i.regimes = set(&["observational"]),
        |i| i.graph_id = "three_slice_graph".into(),
        |i| i.proof_id = "proof_2".into(),
    ];
    for edit in edits {
        let mut new = replacement();
        edit(&mut new);
        assert_eq!(
            decide_refresh(&identity(), &new),
            RefreshDecision::Invalidated(RefreshInvalidation::PremisesChanged)
        );
        let message = invalidation_detail(refresh_held(&held(false), new, evaluate));
        assert!(message.contains("temporal_refresh.premises_changed"), "{message}");
    }
}

#[test]
fn third_period_new_snapshot_with_identical_period_is_stale() {
    let mut new = replacement();
    new.period = identity().period;
    assert_eq!(
        decide_refresh(&identity(), &new),
        RefreshDecision::Invalidated(RefreshInvalidation::StaleSnapshot)
    );
    let message = invalidation_detail(refresh_held(&held(true), new, evaluate));
    assert!(message.contains("temporal_refresh.stale_snapshot"), "{message}");
    // A new period under the old snapshot id, and an unchanged identity, are stale too.
    let mut same_snapshot = replacement();
    same_snapshot.snapshot_id = identity().snapshot_id;
    assert_eq!(
        decide_refresh(&identity(), &same_snapshot),
        RefreshDecision::Invalidated(RefreshInvalidation::StaleSnapshot)
    );
    assert_eq!(
        decide_refresh(&identity(), &identity()),
        RefreshDecision::Invalidated(RefreshInvalidation::StaleSnapshot)
    );
}

#[test]
fn third_period_invalid_replacement_is_refused_before_evaluation() {
    let mut empty_period = replacement();
    empty_period.period = ObservationPeriod { start: 5, end: 5 };
    let mut no_units = replacement();
    no_units.unit_ids.clear();
    for bad in [empty_period, no_units] {
        let mut called = false;
        let error = refresh_held(&held(false), bad, |id| {
            called = true;
            evaluate(id)
        })
        .unwrap_err();
        assert!(!called);
        let EstimationError::Refused { code, message } = error else { panic!("coded refusal") };
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("temporal_refresh.invalid_replacement"));
    }
}

#[test]
fn third_period_invalidated_refresh_never_runs_the_evaluator() {
    let mut new = replacement();
    new.horizon = 3;
    let mut called = false;
    let result = refresh_held(&held(true), new, |id| {
        called = true;
        evaluate(id)
    });
    let message = invalidation_detail(result);
    assert!(
        message.starts_with("temporal_refresh.horizon_changed:"),
        "{message}"
    );
    assert!(!called, "a stale proof must not be re-evaluated");
}

#[test]
fn period_reseal_identity_digest_changes_with_every_single_input() {
    let base = identity().digest();
    let edits: [fn(&mut TemporalWindowIdentity); 11] = [
        |i| i.graph_id = "g2".into(),
        |i| i.horizon = 3,
        |i| i.lag_alignment[0].1 = 2,
        |i| i.intervention_history[1] = "a2=1".into(),
        |i| {
            i.selection_targets.insert("s_t3".into());
        },
        |i| {
            i.regimes.remove("experiment");
        },
        |i| i.period.start = 1,
        |i| i.period.end = 11,
        |i| {
            i.unit_ids.insert("u9".into());
        },
        |i| i.snapshot_id = "snap_other".into(),
        |i| i.proof_id = "proof_other".into(),
    ];
    let mut seen = BTreeSet::from([base.clone()]);
    for edit in edits {
        let mut changed = identity();
        edit(&mut changed);
        let digest = changed.digest();
        assert_ne!(digest, base);
        assert!(seen.insert(digest), "two single edits collided");
    }
    assert_eq!(identity().digest(), base, "digest is deterministic");
}

#[test]
fn period_reseal_receipt_digest_changes_with_period_snapshot_units_and_interval() {
    let (_, base) = refresh_held(&held(true), replacement(), evaluate).unwrap();
    let mut variants = Vec::new();
    let mut other_period = replacement();
    other_period.period = ObservationPeriod { start: 11, end: 20 };
    variants.push(refresh_held(&held(true), other_period, evaluate).unwrap().1);
    let mut other_snapshot = replacement();
    other_snapshot.snapshot_id = "snap_newer".into();
    variants.push(refresh_held(&held(true), other_snapshot, evaluate).unwrap().1);
    let mut other_units = replacement();
    other_units.unit_ids.insert("u9".into());
    variants.push(refresh_held(&held(true), other_units, evaluate).unwrap().1);
    variants.push(refresh_held(&held(false), replacement(), evaluate).unwrap().1);
    for variant in &variants {
        assert_ne!(variant.digest, base.digest);
    }
}

#[test]
fn period_reseal_unsealed_edit_of_a_receipt_is_refused() {
    let held = held(true);
    let (refreshed, mut receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    receipt.new_period = ObservationPeriod { start: 11, end: 21 };
    let message = invalidation_detail(accept_refresh(&held, &receipt, refreshed));
    assert!(message.contains("temporal_refresh.receipt_digest_mismatch"), "{message}");
}

#[test]
fn period_reseal_resealed_receipt_for_another_period_or_snapshot_is_refused() {
    let held = held(true);
    let (refreshed, receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    // Honest receipt is accepted.
    let accepted = accept_refresh(&held, &receipt, refreshed.clone()).unwrap();
    assert_eq!(accepted, refreshed);

    let reseal = |mut r: RefreshReceipt| {
        r.digest = r.compute_digest();
        r
    };
    let mut other_period = receipt.clone();
    other_period.new_period = ObservationPeriod { start: 12, end: 22 };
    let mut other_snapshot = receipt.clone();
    other_snapshot.new_snapshot_id = "snap_forged".into();
    let mut other_proof = receipt.clone();
    other_proof.proof_id = "proof_forged".into();
    let mut flipped_interval = receipt.clone();
    flipped_interval.interval_invalidated = false;
    for forged in [other_period, other_snapshot, other_proof, flipped_interval] {
        let message =
            invalidation_detail(accept_refresh(&held, &reseal(forged), refreshed.clone()));
        assert!(message.contains("temporal_refresh.receipt_mismatch"), "{message}");
    }
}

#[test]
fn period_reseal_receipt_for_a_different_held_result_is_refused() {
    let held_a = held(true);
    let (refreshed, receipt) = refresh_held(&held_a, replacement(), evaluate).unwrap();
    // A different held result (other snapshot id and period) than the receipt's old identity.
    let mut other = identity();
    other.snapshot_id = "snap_elsewhere".into();
    other.period = ObservationPeriod { start: -5, end: 0 };
    let held_b = HeldTemporalResult { identity: other, value: 0.0, interval_present: true };
    let message = invalidation_detail(accept_refresh(&held_b, &receipt, refreshed));
    assert!(message.contains("temporal_refresh.receipt_mismatch"), "{message}");
}

#[test]
fn period_reseal_refreshed_result_keeping_an_interval_is_refused() {
    let held = held(true);
    let (mut refreshed, receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    refreshed.interval_present = true;
    let message = invalidation_detail(accept_refresh(&held, &receipt, refreshed));
    assert!(message.contains("temporal_refresh.stale_interval"), "{message}");
}

#[test]
fn resealed_refresh_cannot_change_inference_or_validate_an_invalid_window() {
    let held = held(false);
    let (fresh, mut receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    receipt.inference_claim = "confidence_interval";
    receipt.digest = receipt.compute_digest();
    assert_eq!(invalidation_detail(accept_refresh(&held, &receipt, fresh)), "temporal_refresh.receipt_mismatch: the receipt changes the point-only inference claim");

    let (mut fresh, mut receipt) = refresh_held(&held, replacement(), evaluate).unwrap();
    fresh.identity.period.end = fresh.identity.period.start;
    receipt.new_identity_digest = fresh.identity.digest();
    receipt.new_period = fresh.identity.period;
    receipt.digest = receipt.compute_digest();
    let EstimationError::Refused { code, message } =
        accept_refresh(&held, &receipt, fresh).unwrap_err()
    else {
        panic!("expected a coded refusal");
    };
    assert_eq!(code, "invalid_argument");
    assert_eq!(message.as_ref(), "temporal_refresh.invalid_replacement: the replacement period is empty or reversed");
}

#[test]
fn window_identity_and_receipt_digests_match_the_independent_derivation() {
    // Golden vectors derived in Python from the documented canonical framing (a length-prefixed
    // domain tag, length-prefixed text, little-endian counts) and a from-scratch BLAKE3.
    assert_eq!(
        identity().digest(),
        "5848783d939097dfce9769d79f53987ea23f64bf4927168b2da904551be4ece0"
    );
    let receipt = antecedent_estimate::temporal_refresh::RefreshReceipt {
        old_identity_digest: "aaaaaaaa".into(),
        new_identity_digest: "bbbbbbbb".into(),
        old_period: ObservationPeriod { start: 0, end: 10 },
        new_period: ObservationPeriod { start: 10, end: 20 },
        old_snapshot_id: "snap_old".into(),
        new_snapshot_id: "snap_new".into(),
        proof_id: "proof_1".into(),
        interval_invalidated: true,
        inference_claim: "point_only",
        digest: String::new(),
    };
    assert_eq!(
        receipt.compute_digest(),
        "59b0a1813b03550346c33dbe5b232b1e3a3fe5e8c2aa137fba92c12bce299bde"
    );
}
