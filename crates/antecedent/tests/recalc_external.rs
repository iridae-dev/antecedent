//! Independent foreign mean callbacks, actual invocation receipts and portable replay.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_external::*;
use antecedent_core::ExecutionContext;
use antecedent_core::recalc::Stage;
use antecedent_learn::fit_counts::observe_resolved_fits;

#[path = "common/external_callback_fixture.rs"]
mod reference;
use reference::{Provider, request};
fn execute(
    session: &mut ExternalCallbackSession,
    request: &ExternalCallbackRequest,
    provider: &mut Provider,
) -> ExternalCallbackOutcome {
    let ((out, fits), calls) = count_external_invocations(|| {
        observe_resolved_fits(|| {
            session.execute(request, Some(provider), &ExecutionContext::for_tests(999)).unwrap()
        })
    });
    assert_eq!(fits, 0);
    assert_eq!(calls, out.receipt.totals().external_invocations);
    out
}

#[test]
fn actual_attested_mean_grid_and_branch_one_receipt_reuse() {
    let mut request = request(1, CallbackPolicy::Deterministic);
    let mut provider = Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    assert!(session.plan(&request).is_executable());
    assert_eq!(provider.calls, 0);
    let first = execute(&mut session, &request, &mut provider);
    assert_eq!(first.claim.values().unwrap(), [2.5, 3.0, 3.5]);
    assert_eq!(
        first
            .receipt
            .entry(Stage::ProviderRequest(request.branch))
            .unwrap()
            .counts
            .external_invocations,
        1
    );
    assert_eq!(first.receipt.totals().model_fits + first.receipt.totals().identifications, 0);
    let counts = first
        .receipt
        .entries()
        .iter()
        .map(|entry| {
            (
                entry.stage,
                antecedent_io::recalc_receipt_artifact::CountsWire {
                    external_invocations: entry.counts.external_invocations,
                    ..Default::default()
                },
            )
        })
        .collect();
    let sealed = antecedent_io::recalc_receipt_artifact::RecalcReceiptArtifact::seal(
        &first.previous,
        &first.requested,
        &first.capabilities,
        &counts,
    )
    .unwrap();
    assert_eq!(sealed.receipt_identity(), first.receipt.identity().to_hex());
    let ptr = std::ptr::from_ref(first.claim.as_ref());
    let same = execute(&mut session, &request, &mut provider);
    assert_eq!(same.receipt.totals().total(), 0);
    assert_eq!(std::ptr::from_ref(same.claim.as_ref()), ptr);
    request.columns[0].1 = vec![0.0, 2.0];
    let changed = execute(&mut session, &request, &mut provider);
    assert_eq!(changed.claim.values().unwrap(), [4.5, 5.0, 5.5]);
    assert_eq!(changed.receipt.totals().external_invocations, 1);
}
#[test]
fn seeded_callback_rng_mutations_and_actual_fresh_replay() {
    let mut request = request(0, CallbackPolicy::Seeded);
    let mut provider = Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    let original = execute(&mut session, &request, &mut provider);
    assert_eq!(original.claim.values().unwrap(), [2.58, 3.08, 3.58]);
    request.seed = 42;
    let changed = execute(&mut session, &request, &mut provider);
    assert_eq!(changed.claim.values().unwrap(), [2.59, 3.09, 3.59]);
    let bytes = session.export_output().unwrap();
    let mut fresh = ExternalCallbackSession::resume(&bytes, &request).unwrap();
    assert!(fresh.issued_claim().is_none());
    assert!(!fresh.plan(&request).is_executable());
    let ((), calls) = count_external_invocations(|| {
        assert!(fresh.execute(&request, None, &ExecutionContext::for_tests(1)).is_err());
    });
    assert_eq!(calls, 0);
    let replayed = execute(&mut fresh, &request, &mut provider);
    assert_eq!(replayed.claim.values(), changed.claim.values());
    assert_eq!(replayed.receipt.totals().external_invocations, 1);
    let mut replaced = Provider::new(&request);
    replaced.descriptor.identity.version_id = "v2".into();
    assert!(matches!(
        fresh.execute(&request, Some(&mut replaced), &ExecutionContext::for_tests(1)),
        Err(ExternalCallbackError::Request("external_recalc.provider_mismatch"))
    ));
    let mut stale = ExternalCallbackSession::resume(&bytes, &request).unwrap();
    provider.offset = 1.0;
    let (failure, calls) = count_external_invocations(|| {
        stale.execute(&request, Some(&mut provider), &ExecutionContext::for_tests(1))
    });
    assert!(matches!(
        failure,
        Err(ExternalCallbackError::Attempt { detail: "external_recalc.replay_mismatch", .. })
    ));
    assert_eq!(calls, 1);
    assert!(stale.issued_claim().is_none());
}
#[test]
fn failures_preserve_success_and_keep_actual_attempt_evidence() {
    let mut request = request(0, CallbackPolicy::Deterministic);
    let mut provider = Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    let original = execute(&mut session, &request, &mut provider);
    request.columns[0].1[0] = 0.0;
    provider.fail = true;
    let (failure, calls) = count_external_invocations(|| {
        session.execute(&request, Some(&mut provider), &ExecutionContext::for_tests(1))
    });
    assert!(matches!(
        failure,
        Err(ExternalCallbackError::Attempt {
            report: InvocationAttempt { invocations: 1, .. },
            detail: "external_recalc.callback_failed",
            ..
        })
    ));
    assert_eq!(calls, 1);
    assert_eq!(session.issued_claim().unwrap().values(), original.claim.values());
    provider.fail = false;
    provider.wrong_quantity = true;
    assert!(matches!(
        session.execute(&request, Some(&mut provider), &ExecutionContext::for_tests(1)),
        Err(ExternalCallbackError::BindingAttempt { .. })
    ));
    request.columns[0].1[0] = -1.0;
    provider.wrong_quantity = false;
    assert_eq!(execute(&mut session, &request, &mut provider).receipt.totals().total(), 0);
}
#[test]
fn stateful_recomputes_and_side_effect_failure_never_silently_retries() {
    let request = request(0, CallbackPolicy::Stateful);
    let mut provider = Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    execute(&mut session, &request, &mut provider);
    provider.offset = 1.0;
    let changed = execute(&mut session, &request, &mut provider);
    assert_eq!(changed.receipt.totals().external_invocations, 1);
    assert_eq!(changed.claim.values().unwrap(), [3.5, 4.0, 4.5]);
    assert!(ExternalCallbackSession::resume(&session.export_output().unwrap(), &request).is_err());
    let mut effects = request.clone();
    effects.descriptor.policy = CallbackPolicy::SideEffecting;
    let mut provider = Provider::new(&effects);
    provider.fail = true;
    assert!(
        session.execute(&effects, Some(&mut provider), &ExecutionContext::for_tests(1)).is_err()
    );
    let before = provider.calls;
    assert!(matches!(
        session.execute(&effects, Some(&mut provider), &ExecutionContext::for_tests(1)),
        Err(ExternalCallbackError::Request("external_recalc.retry_unsafe"))
    ));
    assert_eq!(provider.calls, before);
    effects.idempotency_key = Some("operation-one".into());
    assert!(matches!(
        session.execute(&effects, Some(&mut provider), &ExecutionContext::for_tests(1)),
        Err(ExternalCallbackError::Request("external_recalc.idempotency_unsupported"))
    ));
    effects.descriptor.idempotency_supported = true;
    provider.descriptor = effects.descriptor.clone();
    provider.fail = false;
    execute(&mut session, &effects, &mut provider);
}
#[test]
fn pure_invalid_planning_and_cooperative_budget_and_cancel_refusals() {
    let mut request = request(0, CallbackPolicy::Unknown);
    let session = ExternalCallbackSession::new();
    assert!(!session.plan(&request).is_executable());
    request.descriptor.policy = CallbackPolicy::Deterministic;
    let mut session = ExternalCallbackSession::new();
    let mut provider = Provider::new(&request);
    let mut budget = ExecutionContext::for_tests(1);
    budget.memory.hard_limit_bytes = Some(1);
    assert!(matches!(
        session.execute(&request, Some(&mut provider), &budget),
        Err(ExternalCallbackError::Request("recalc.memory_budget_exceeded"))
    ));
    assert_eq!(provider.calls, 0);
    provider.cancel = true;
    let ctx = ExecutionContext::for_tests(1);
    let (failure, calls) =
        count_external_invocations(|| session.execute(&request, Some(&mut provider), &ctx));
    assert!(matches!(
        failure,
        Err(ExternalCallbackError::Attempt { detail: "recalc.cancelled", .. })
    ));
    assert_eq!(calls, 1);
    assert!(session.issued_claim().is_none());
}

#[test]
fn idempotency_key_is_bound_to_actual_first_attempt_payload() {
    let mut request = request(0, CallbackPolicy::SideEffecting);
    request.descriptor.idempotency_supported = true;
    request.idempotency_key = Some("operation-1".into());
    let mut provider = Provider::new(&request);
    let mut session = ExternalCallbackSession::new();
    execute(&mut session, &request, &mut provider);
    execute(&mut session, &request, &mut provider);
    assert_eq!(provider.calls, 2);
    request.columns[0].1 = vec![0.0, 2.0];
    let (refusal, calls) = count_external_invocations(|| {
        session.execute(&request, Some(&mut provider), &ExecutionContext::for_tests(1))
    });
    assert!(matches!(
        refusal,
        Err(ExternalCallbackError::Request("external_recalc.idempotency_conflict"))
    ));
    assert_eq!(calls, 0);
    assert_eq!(provider.calls, 2);
}
