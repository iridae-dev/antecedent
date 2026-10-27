//! Tests for the execution context.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use super::{admission::*, budget::*, cancel::*, context::*, pipeline::*, request::*, rng::*};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn rng_streams_are_deterministic() {
    let factory = RngFactory::from_seed(42);
    let mut a1 = factory.stream_for(StreamDomain::Test, 0);
    let mut a2 = factory.stream_for(StreamDomain::Test, 0);
    let mut b = factory.stream_for(StreamDomain::Test, 1);
    let seq_a1: Vec<u64> = (0..8).map(|_| a1.next_u64()).collect();
    let seq_a2: Vec<u64> = (0..8).map(|_| a2.next_u64()).collect();
    let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
    assert_eq!(seq_a1, seq_a2);
    assert_ne!(seq_a1, seq_b);
}

#[test]
fn independent_factories_same_seed_match() {
    let f1 = RngFactory::from_seed(7);
    let f2 = RngFactory::from_seed(7);
    let mut s1 = f1.stream_for(StreamDomain::Test, 99);
    let mut s2 = f2.stream_for(StreamDomain::Test, 99);
    for _ in 0..32 {
        assert_eq!(s1.next_u64(), s2.next_u64());
    }
}

#[test]
fn derive_stream_rejects_additive_master_stream_alias() {
    // (s, k) must not equal (s − 1, k + 1).
    for s in [1u64, 42, 1000, u64::MAX / 2, u64::MAX] {
        for k in [0u64, 1, 999, 0xA7E0_0001_1000] {
            let a = RngFactory::from_seed(s).stream(k).state();
            let b = RngFactory::from_seed(s.wrapping_sub(1)).stream(k.wrapping_add(1)).state();
            assert_ne!(a, b, "alias at master={s} stream={k}");
            let af = RngFactory::from_seed(s).stream_for(StreamDomain::Resample, k).state();
            let bf = RngFactory::from_seed(s.wrapping_sub(1))
                .stream_for(StreamDomain::Resample, k.wrapping_add(1))
                .state();
            assert_ne!(af, bf, "stream_for alias at master={s} index={k}");
        }
    }
}

#[test]
fn stream_for_domains_separate_same_index() {
    let factory = RngFactory::from_seed(42);
    let a = factory.stream_for(StreamDomain::Resample, 7).state();
    let b = factory.stream_for(StreamDomain::TemporalBlock, 7).state();
    let c = factory.stream_for(StreamDomain::McmcStructure, 7).state();
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
}

#[test]
fn built_in_domain_tags_are_pinned() {
    // The tag is the whole domain contribution to a derived stream; these
    // are the values the closed enum carried, so they may never change.
    let pinned = [
        (StreamDomain::Resample, 1),
        (StreamDomain::TemporalBlock, 2),
        (StreamDomain::Bayesian, 3),
        (StreamDomain::McmcStructure, 4),
        (StreamDomain::McmcOrder, 5),
        (StreamDomain::McmcDbn, 6),
        (StreamDomain::Completion, 7),
        (StreamDomain::Counterfactual, 8),
        (StreamDomain::Attribution, 9),
        (StreamDomain::Design, 10),
        (StreamDomain::Transport, 11),
        (StreamDomain::StatsCi, 12),
        (StreamDomain::Learner, 13),
        (StreamDomain::Estimate, 14),
        (StreamDomain::Mediation, 15),
        (StreamDomain::Panel, 16),
        (StreamDomain::Execute, 17),
        (StreamDomain::Test, 18),
        (StreamDomain::Example, 19),
    ];
    for (domain, tag) in pinned {
        assert_eq!(domain.tag(), tag);
    }
    let factory = RngFactory::from_seed(42);
    let by_domain = factory.stream_for(StreamDomain::Transport, 5).state();
    let by_hand = factory.stream(11u64.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 5).state();
    assert_eq!(by_domain, by_hand);
}

#[test]
fn external_domains_stay_clear_of_built_in_tags() {
    const OURS: StreamDomain = StreamDomain::external(StreamDomain::EXTERNAL_BASE);
    assert_eq!(OURS.tag(), StreamDomain::EXTERNAL_BASE);
    assert_ne!(OURS, StreamDomain::Example);
    assert_eq!(format!("{:?}", StreamDomain::Resample), "Resample");
    let factory = RngFactory::from_seed(42);
    assert_ne!(
        factory.stream_for(OURS, 0).state(),
        factory.stream_for(StreamDomain::Example, 0).state()
    );
}

#[test]
#[should_panic(expected = "collides with a built-in domain")]
fn external_domain_below_the_base_is_refused() {
    let _ = StreamDomain::external(StreamDomain::Example.tag());
}

#[test]
fn cancellation_is_shared_across_clones() {
    let token = CancellationToken::new();
    let clone = token.clone();
    assert!(!token.is_cancelled());
    clone.cancel();
    assert!(token.is_cancelled());
}

#[test]
fn request_identity_conflicts_on_reused_key() {
    use crate::identity::{ContractIdentities, SemanticDigest};
    let digest = |byte: u8| SemanticDigest::from_bytes([byte; 32]);
    let identities = |target: u8, inference: u8, data: u8| {
        ContractIdentities::new(
            digest(target),
            digest(2),
            Some(digest(3)),
            Some(digest(4)),
            digest(inference),
            digest(6),
            digest(data),
        )
    };
    let key = digest(1);
    let execution = digest(8);
    let first = RequestIdentity::new(key, identities(10, 5, 7), execution);
    let same = RequestIdentity::new(key, identities(10, 5, 7), execution);
    assert!(!first.conflicts_with(&same));
    // Data snapshot, target population, and inference binding each change
    // the scientific request even when the program digest is shared.
    for changed in [identities(10, 5, 9), identities(11, 5, 7), identities(10, 12, 7)] {
        assert!(first.conflicts_with(&RequestIdentity::new(key, changed, execution)));
    }
    assert!(first.conflicts_with(&RequestIdentity::new(key, identities(10, 5, 7), digest(9))));
    assert!(!first.conflicts_with(&RequestIdentity::new(
        digest(99),
        identities(11, 12, 9),
        execution
    )));
    assert!(!ExecutionRequestState::Cancelled.publishes_claim());
    assert!(ExecutionRequestState::Completed.publishes_claim());
    let receipt = ExecutionReceipt::new(first, ExecutionRequestState::Cancelled);
    assert_eq!(receipt.state.as_str(), "cancelled");
}

#[test]
fn production_token_does_not_trip_and_with_admission_keeps_clones() {
    let token = CancellationToken::new();
    for _ in 0..8 {
        assert!(!token.is_cancelled());
    }
    assert!(!format!("{token:?}").contains("trip"));
    let tripping = CancellationToken::cancel_after_checks(0);
    assert!(tripping.is_cancelled());

    let original = AdmissionPool::fixed(2, 10);
    let study = ExecutionContext::for_tests(1).with_admission(original.clone());
    let clone = study.clone();
    let replacement = AdmissionPool::fixed(8, 80);
    let moved = clone.with_admission(replacement.clone());
    assert_eq!(study.admission().thread_budget(), 2);
    assert_eq!(moved.admission().thread_budget(), 8);
    let _lease = moved.acquire_admission(8, 80).unwrap();
    assert_eq!(original.available().threads(), 2);
    assert_eq!(replacement.available().threads(), 0);
}

#[test]
fn credit_overflow_is_a_typed_error_and_leaves_the_lease() {
    let mut state = AdmissionState {
        thread_budget: u32::MAX,
        byte_budget: 0,
        threads_free: u32::MAX,
        bytes_free: 0,
        leases: vec![LeaseSlot { observation: None, id: 7, parent: None, threads: 1, bytes: 0 }],
        next_id: 8,
        blocked_waiters: 0,
    };
    let refused = close_slot(&mut state, 0).expect_err("credit overflow must be reported");
    assert!(matches!(refused, AdmissionError::Overflow), "{refused}");
    assert_eq!(state.leases.len(), 1);
    assert_eq!(state.leases[0].id, 7);
}

#[test]
fn cancel_wakes_a_blocked_waiter() {
    let shared = AdmissionPool::fixed(1, 8);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let held = study.acquire_admission(1, 8).unwrap();
    let waiting_study = study.clone();
    let started = Instant::now();
    let waiting = thread::spawn(move || {
        waiting_study.acquire_admission_bounded(1, 8, Duration::from_secs(30))
    });
    while shared.blocked_waiters() == 0 {
        assert!(started.elapsed() < Duration::from_secs(2), "waiter did not block");
        thread::yield_now();
    }
    study.cancellation.cancel();
    let result = waiting.join().unwrap();
    assert!(started.elapsed() < Duration::from_secs(2), "cancel left the waiter sleeping");
    assert!(matches!(result, Err(AdmissionError::Cancelled)));
    drop(held);
    assert_eq!(shared.available(), AdmissionRemainder::new(1, 8));
}

#[test]
fn two_contexts_do_not_share_a_mutated_pool() {
    let original = AdmissionPool::fixed(2, 10);
    let first = ExecutionContext::for_tests(1).with_admission(original.clone());
    let sibling = first.clone();
    let replacement = AdmissionPool::fixed(4, 40);
    let moved = sibling.with_admission(replacement.clone());
    let held = first.acquire_admission(2, 10).unwrap();
    assert_eq!(original.available().threads(), 0);
    assert_eq!(original.available().bytes(), 0);
    assert_eq!(replacement.available().threads(), 4);
    let moved_hold = moved.acquire_admission(4, 40).unwrap();
    assert_eq!(replacement.available().threads(), 0);
    moved.cancellation.cancel();
    assert!(first.cancellation.is_requested());
    assert_eq!(first.admission().thread_budget(), 2);
    assert_eq!(moved.admission().thread_budget(), 4);
    drop(held);
    drop(moved_hold);
    assert_eq!(original.available().threads(), 2);
    assert_eq!(replacement.available().threads(), 4);
}

#[test]
fn closing_child_then_parent_preserves_other_lease_holdings() {
    let mut state = AdmissionState {
        thread_budget: 10,
        byte_budget: 100,
        threads_free: 4,
        bytes_free: 40,
        leases: vec![
            LeaseSlot { id: 3, parent: Some(1), threads: 1, bytes: 10, observation: None },
            LeaseSlot { id: 2, parent: None, threads: 3, bytes: 30, observation: None },
            LeaseSlot { id: 1, parent: None, threads: 2, bytes: 20, observation: None },
            LeaseSlot { id: 4, parent: Some(2), threads: 0, bytes: 0, observation: None },
        ],
        next_id: 5,
        blocked_waiters: 0,
    };
    close_slot(&mut state, 0).unwrap();
    let parent_index = lease_index(&state.leases, 1).unwrap();
    close_slot(&mut state, parent_index).unwrap();
    assert_eq!(state.leases.iter().map(|slot| slot.id).collect::<Vec<_>>(), vec![2, 4]);
    assert_eq!(state.leases[1].parent, Some(2));
    assert_eq!(state.threads_free, 7);
    assert_eq!(state.bytes_free, 70);
    let mut threads = state.threads_free;
    let mut bytes = state.bytes_free;
    for slot in &state.leases {
        threads += slot.threads;
        bytes += slot.bytes;
    }
    assert_eq!(threads, state.thread_budget);
    assert_eq!(bytes, state.byte_budget);
}

#[test]
fn cancelled_parent_and_child_hold_live_memory_until_cleanup() {
    let pool = AdmissionPool::fixed(2, 100);
    let owner = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let other = ExecutionContext::for_tests(2).with_admission(pool.clone());
    let parent = owner.acquire_admission(2, 100).unwrap();
    let child = parent.acquire_child(1, 40).unwrap();
    owner.cancellation.cancel();
    assert_eq!(pool.available(), AdmissionRemainder::zero());
    assert!(matches!(parent.acquire_child(0, 1), Err(AdmissionError::Cancelled)));
    assert!(other.acquire_admission(1, 1).is_err());
    drop(parent);
    assert_eq!(pool.available(), AdmissionRemainder::new(1, 60));
    let _replacement = other.acquire_admission(1, 60).unwrap();
    assert_eq!(pool.available(), AdmissionRemainder::zero());
    drop(child);
    assert_eq!(pool.available(), AdmissionRemainder::new(1, 40));
}

#[test]
fn map_indexed_with_scratch_reuses_one_buffer_per_worker() {
    let ctx = ExecutionContext::production(1, 4);
    let built = AtomicUsize::new(0);
    let out = ctx
        .map_indexed_with_scratch(
            8,
            || {
                built.fetch_add(1, Ordering::SeqCst);
                Vec::<usize>::new()
            },
            |index, inner, scratch| {
                assert_eq!(inner.parallelism.max_threads.get(), 1);
                scratch.push(index);
                Ok::<_, ()>(scratch.clone())
            },
        )
        .unwrap();
    assert!(built.load(Ordering::SeqCst) <= 4);
    assert!(built.load(Ordering::SeqCst) >= 1);
    assert_eq!(
        out.iter().map(|row| *row.last().expect("scratch recorded the index")).collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );
    assert!(out.iter().any(|row| row.len() > 1), "a worker reused its scratch");
}

#[test]
fn map_indexed_preserves_order_and_serializes_inner() {
    let ctx = ExecutionContext::production(1, 4);
    let out = ctx
        .map_indexed(8, |i, inner| {
            assert_eq!(inner.parallelism.max_threads.get(), 1);
            Ok::<_, ()>(i * 10)
        })
        .unwrap();
    assert_eq!(out, vec![0, 10, 20, 30, 40, 50, 60, 70]);
    let serial = ExecutionContext::for_tests(1);
    let inner_threads =
        serial.map_indexed(1, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get())).unwrap();
    assert_eq!(inner_threads, vec![1]);
    assert_eq!(serial.serial_inner().parallelism.max_threads.get(), 1);
}

#[test]
fn map_indexed_fallback_keeps_inner_parallelism_within_admission() {
    let ctx = ExecutionContext::production(1, 4).with_admission(AdmissionPool::fixed(1, u64::MAX));
    for n in [1, 8] {
        let threads = ctx
            .map_indexed(n, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get()))
            .unwrap();
        assert_eq!(threads, vec![1; n]);
    }
    let ctx = ExecutionContext::production(1, 4).with_admission(AdmissionPool::fixed(2, u64::MAX));
    let threads =
        ctx.map_indexed(1, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get())).unwrap();
    assert_eq!(threads, vec![2]);
    ctx.cancellation.cancel();
    let threads =
        ctx.map_indexed(8, |_, inner| Ok::<_, ()>(inner.parallelism.max_threads.get())).unwrap();
    assert_eq!(threads, vec![1; 8]);
}

#[test]
fn map_indexed_stops_at_a_failure_and_reports_the_lowest_index() {
    let ctx = ExecutionContext::production(1, 4);
    let evaluated = AtomicUsize::new(0);
    let n = 2000;
    // Index 0 fails immediately; its own chunk (500 items) must stop there.
    let result = ctx.map_indexed(n, |i, _| {
        evaluated.fetch_add(1, Ordering::Relaxed);
        if i == 0 { Err(i) } else { Ok(i) }
    });
    assert_eq!(result, Err(0));
    assert!(evaluated.load(Ordering::Relaxed) <= n - 499, "the failing chunk must not run on");
    // Two failures in different chunks: the lowest index wins on every run.
    for _ in 0..20 {
        let result = ctx.map_indexed(n, |i, _| if i == 700 || i == 1500 { Err(i) } else { Ok(i) });
        assert_eq!(result, Err(700));
    }
    // No failure: everything is evaluated, in order.
    let all = ctx.map_indexed(n, |i, _| Ok::<_, ()>(i * 2)).unwrap();
    assert_eq!(all, (0..n).map(|i| i * 2).collect::<Vec<_>>());
}

#[test]
fn adaptive_budgets_are_disabled_by_default() {
    assert_eq!(AdaptiveBootstrapBudget::default(), AdaptiveBootstrapBudget::disabled());
    assert_eq!(AdaptiveDrawBudget::default(), AdaptiveDrawBudget::disabled());
    let spread = AdaptiveBootstrapBudget { min_replicates: 5, ..Default::default() };
    assert!(!spread.enabled);
    assert_eq!(spread.required_replicates(), u32::MAX);
}

#[test]
fn unavailable_arch_simd_is_not_an_effective_request() {
    const { assert!(!ARCH_SIMD_COMPILED) };
    assert!(!KernelPolicy::default_policy().arch_simd_effective());
    assert!(!KernelPolicy::scalar_only().arch_simd_effective());
}

#[test]
fn default_user_threads_is_at_least_one_and_capped() {
    let n = default_user_threads();
    assert!(n >= 1);
    assert!(n <= DEFAULT_USER_THREAD_CAP);
    let ctx = ExecutionContext::production_default(1);
    assert_eq!(ctx.parallelism.max_threads.get(), n);
    assert_ne!(ctx.kernel_policy, KernelPolicy::scalar_only());
}

#[test]
fn production_and_test_contexts_evaluate_full_monte_carlo_effort() {
    let production = ExecutionContext::production(1, 4);
    let tests = ExecutionContext::for_tests(1);
    assert_eq!(production.adaptive_bootstrap, AdaptiveBootstrapBudget::disabled());
    assert_eq!(production.adaptive_draws, AdaptiveDrawBudget::disabled());
    assert_eq!(production.adaptive_bootstrap, tests.adaptive_bootstrap);
    assert_eq!(production.adaptive_draws, tests.adaptive_draws);
}

#[test]
fn bootstrap_budget_floor_is_the_monte_carlo_error_bound() {
    let at = |eps: f64, min: u32| {
        AdaptiveBootstrapBudget { enabled: true, min_replicates: min, se_rel_epsilon: eps }
            .required_replicates()
    };
    // 1/√(2(B−1)) ≤ ε  ⇔  B ≥ 1 + 1/(2ε²).
    assert_eq!(at(0.05, 0), 201);
    assert_eq!(at(0.10, 0), 51);
    assert_eq!(at(0.01, 0), 5_001);
    assert_eq!(at(0.5, 0), 3);
    assert_eq!(at(0.5, 10), 10, "explicit floor wins when larger");
    assert_eq!(at(0.0, 10), u32::MAX, "non-positive ε never stops");
    assert_eq!(at(f64::NAN, 10), u32::MAX);
    assert_eq!(at(1e-9, 0), u32::MAX, "saturates instead of truncating");
    assert_eq!(AdaptiveBootstrapBudget::disabled().required_replicates(), u32::MAX);
}

#[test]
fn f64_draws_are_in_unit_interval() {
    let mut rng = CausalRng::from_seed(123);
    for _ in 0..1000 {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x));
    }
}

#[test]
fn simultaneous_studies_compete_for_a_fixed_admission_budget() {
    let shared = AdmissionPool::fixed(4, 500);
    let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
    assert_eq!(study_a.memory_limit_bytes(), Some(500));
    assert_eq!(shared.thread_budget(), 4);
    assert_eq!(shared.byte_budget(), 500);
    let gate = Arc::new(Barrier::new(2));
    let start = |study: ExecutionContext, gate: Arc<Barrier>| {
        thread::spawn(move || {
            gate.wait();
            study.acquire_admission(3, 300)
        })
    };
    let left = start(study_a, Arc::clone(&gate));
    let right = start(study_b, gate);
    let left = left.join().unwrap();
    let right = right.join().unwrap();
    let ((Ok(_), Err(refusal)) | (Err(refusal), Ok(_))) = (&left, &right) else {
        panic!("one study must take the budget and the other must be refused");
    };
    assert!(matches!(
        refusal,
        AdmissionError::ExceedsRemainder {
            requested_threads: 3,
            requested_bytes: 300,
            remaining_threads: 1,
            remaining_bytes: 200,
        }
    ));
    assert_eq!(shared.available().threads(), 1);
    assert_eq!(shared.available().bytes(), 200);
    drop(left);
    drop(right);
    assert_eq!(shared.available().threads(), 4);
    assert_eq!(shared.available().bytes(), 500);
}

#[test]
fn cloned_context_shares_the_admission_decrement() {
    let study = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(2, 10));
    let clone = study.clone();
    let lease = study.acquire_admission(2, 10).unwrap();
    assert!(clone.acquire_admission(1, 0).is_err());
    assert_eq!(clone.admission().available().threads(), 0);
    assert_eq!(lease.release().threads(), 2);
    assert_eq!(clone.admission().available().threads(), 2);
    assert_eq!(clone.admission().available().bytes(), 10);
}

#[test]
fn child_lease_cannot_exceed_parent_remainder() {
    let shared = AdmissionPool::fixed(10, 1_000);
    let study = ExecutionContext::for_tests(7).with_admission(shared.clone());
    let parent = study.acquire_admission(4, 400).unwrap();
    assert_eq!(shared.available().threads(), 6);
    let too_wide = parent.acquire_child(5, 1);
    assert!(matches!(
        too_wide,
        Err(AdmissionError::ExceedsRemainder { remaining_threads: 4, remaining_bytes: 400, .. })
    ));
    assert_eq!(shared.available().threads(), 6);
    let child = parent.acquire_child(3, 100).unwrap();
    assert_eq!(parent.remainder().threads(), 1);
    assert_eq!(parent.remainder().bytes(), 300);
    assert_eq!(shared.available().threads(), 6);
    let returned = child.release();
    assert_eq!(returned.threads(), 3);
    assert_eq!(returned.bytes(), 100);
    assert_eq!(parent.remainder().threads(), 4);
    assert_eq!(parent.remainder().bytes(), 400);
    let child = parent.acquire_child(4, 400).unwrap();
    assert_eq!(parent.release().threads(), 0);
    assert_eq!(shared.available().threads(), 6);
    assert_eq!(child.remainder().threads(), 4);
    drop(child);
    assert_eq!(shared.available().threads(), 10);
    assert_eq!(shared.available().bytes(), 1_000);
}

#[test]
fn cancellation_keeps_admission_until_the_owner_releases() {
    let shared = AdmissionPool::fixed(4, 80);
    let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
    let lease = study_a.acquire_admission(4, 80).unwrap();
    study_a.cancellation.cancel();
    assert_eq!(shared.available(), AdmissionRemainder::zero());
    assert!(study_b.acquire_admission(4, 80).is_err());
    assert_eq!(lease.release(), AdmissionRemainder::new(4, 80));
    let taken = study_b.acquire_admission(4, 80).unwrap();
    assert_eq!(shared.available().threads(), 0);
    drop(taken);
    assert_eq!(shared.available().threads(), 4);
    study_b.cancellation.cancel();
    let refused = study_b.acquire_admission_bounded(1, 1, Duration::from_secs(30));
    assert!(matches!(refused, Err(AdmissionError::Cancelled)));
}

#[test]
fn byte_overflow_is_refused_instead_of_wrapped() {
    let shared = AdmissionPool::fixed(2, 32);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    assert!(checked_byte_product(u64::MAX, 2).is_none());
    assert!(checked_byte_sum(u64::MAX, 1).is_none());
    assert_eq!(u64::MAX.wrapping_add(1), 0);
    assert!(matches!(AdmissionRequest::try_sum(1, &[u64::MAX, 1]), Err(AdmissionError::Overflow)));
    assert!(matches!(
        study.acquire_admission_product(1, u64::MAX, 2),
        Err(AdmissionError::Overflow)
    ));
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), 32);
    let parent = study.acquire_admission(1, 32).unwrap();
    assert!(matches!(parent.acquire_child_product(1, u64::MAX, 2), Err(AdmissionError::Overflow)));
    assert_eq!(parent.remainder().bytes(), 32);
    assert_eq!(shared.available().bytes(), 0);
    drop(parent);
    assert_eq!(shared.available().bytes(), 32);
}

#[test]
fn bounded_wait_ends_on_release_and_on_the_bound() {
    let shared = AdmissionPool::fixed(2, 40);
    let study_a = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let study_b = ExecutionContext::for_tests(2).with_admission(shared.clone());
    let lease = study_a.acquire_admission(2, 40).unwrap();
    let early = study_b.acquire_admission_bounded(2, 40, Duration::from_millis(20));
    assert!(matches!(early, Err(AdmissionError::WaitBoundExceeded { .. })));
    assert_eq!(shared.available().threads(), 0);
    let waiting =
        thread::spawn(move || study_b.acquire_admission_bounded(2, 40, Duration::from_secs(5)));
    let started = Instant::now();
    while shared.blocked_waiters() == 0 {
        assert!(started.elapsed() < Duration::from_secs(2), "waiter did not block");
        thread::yield_now();
    }
    assert_eq!(lease.release().threads(), 2);
    let got = waiting.join().unwrap().unwrap();
    assert_eq!(got.remainder().threads(), 2);
    assert_eq!(shared.available().threads(), 0);
    drop(got);
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), 40);
}

fn parallel_context(pool: &AdmissionPool, seed: u64) -> ExecutionContext {
    ExecutionContext::production(seed, 2).with_admission(pool.clone())
}

struct HoldState {
    entered: usize,
    release: bool,
}

struct Hold {
    state: Mutex<HoldState>,
    cv: Condvar,
}

impl Hold {
    const fn new() -> Self {
        Self { state: Mutex::new(HoldState { entered: 0, release: false }), cv: Condvar::new() }
    }

    fn enter(&self) {
        let mut state = lock_mutex(&self.state);
        state.entered = state.entered.saturating_add(1);
        self.cv.notify_all();
    }

    fn entered(&self) -> usize {
        lock_mutex(&self.state).entered
    }

    fn wait_until_released(&self) {
        let mut state = lock_mutex(&self.state);
        while !state.release {
            state = self.cv.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn release(&self) {
        lock_mutex(&self.state).release = true;
        self.cv.notify_all();
    }
}

struct Release(Arc<Hold>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[test]
fn concurrent_map_indexed_runs_stay_within_the_pool() {
    let shared = AdmissionPool::fixed(2, 64);
    let hold = Arc::new(Hold::new());
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let start = |seed: u64| {
        let ctx = parallel_context(&shared, seed);
        let hold = Arc::clone(&hold);
        let inflight = Arc::clone(&inflight);
        let peak = Arc::clone(&peak);
        thread::spawn(move || {
            ctx.map_indexed(2, move |index, inner| {
                assert_eq!(inner.parallelism.max_threads.get(), 1);
                let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
                peak.fetch_max(now, Ordering::SeqCst);
                hold.enter();
                hold.wait_until_released();
                inflight.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, ()>(index)
            })
        })
    };
    let left = start(1);
    let right = start(2);
    let _release = Release(Arc::clone(&hold));
    let started = Instant::now();
    loop {
        let held = shared.available().threads() == 0;
        let waiting = shared.blocked_waiters() >= 1;
        if hold.entered() == 2 && held && waiting {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "second map_indexed did not wait for the admission lease"
        );
        thread::yield_now();
    }
    assert!(peak.load(Ordering::SeqCst) <= 2);
    hold.release();
    let left = left.join().unwrap().unwrap();
    let right = right.join().unwrap().unwrap();
    assert_eq!(left, vec![0, 1]);
    assert_eq!(right, vec![0, 1]);
    assert!(peak.load(Ordering::SeqCst) <= 2);
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
    assert_eq!(shared.blocked_waiters(), 0);
}

#[test]
fn serial_admitted_maps_share_capacity_and_nested_work_reuses_the_slot() {
    let pool = AdmissionPool::fixed(1, 64);
    let first = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let second = ExecutionContext::for_tests(2).with_admission(pool.clone());
    let entered = Arc::new(Barrier::new(2));
    let finish = Arc::new(Barrier::new(2));
    let worker_entered = Arc::clone(&entered);
    let worker_finish = Arc::clone(&finish);
    let a = thread::spawn(move || {
        first.try_map_indexed(1, |_, inner| {
            assert_eq!(inner.try_map_indexed(2, |i, _| Ok::<_, ()>(i)).unwrap(), vec![0, 1]);
            worker_entered.wait();
            worker_finish.wait();
            Ok::<_, ()>(1)
        })
    });
    entered.wait();
    let b = thread::spawn(move || second.try_map_indexed(1, |_, _| Ok::<_, ()>(2)));
    let deadline = Instant::now() + Duration::from_secs(5);
    while pool.blocked_waiters() == 0 {
        assert!(Instant::now() < deadline, "second serial map must wait for shared admission");
        thread::yield_now();
    }
    assert_eq!(pool.available().threads(), 0);
    finish.wait();
    assert_eq!(a.join().unwrap().unwrap(), vec![1]);
    assert_eq!(b.join().unwrap().unwrap(), vec![2]);
    assert_eq!(pool.available(), AdmissionRemainder::new(1, 64));
}

#[test]
fn admitted_map_refuses_cancellation_before_constructing_scratch() {
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let result = ctx.try_map_indexed_with_scratch(
        1,
        || panic!("cancelled work must not allocate scratch"),
        |_, _, (): &mut ()| Ok::<_, ()>(()),
    );
    assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::Cancelled))));
}

#[test]
fn cancelled_map_indexed_leaves_no_lease() {
    let shared = AdmissionPool::fixed(2, 64);
    let ctx = parallel_context(&shared, 1);
    let entered = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));
    let running = ctx.clone();
    let entered_worker = Arc::clone(&entered);
    let finish_worker = Arc::clone(&finish);
    let handle = thread::spawn(move || {
        running.map_indexed(2, move |index, inner| {
            let _reservation = inner.acquire_admission(0, 32).unwrap();
            let scratch = vec![0_u8; 32];
            entered_worker.wait();
            finish_worker.wait();
            std::hint::black_box(scratch);
            Ok::<_, ()>(index)
        })
    });
    entered.wait();
    assert_eq!(shared.available().threads(), 0);
    assert_eq!(shared.available().bytes(), 0);
    ctx.cancellation.cancel();
    assert_eq!(shared.available(), AdmissionRemainder::zero());
    let competing = ExecutionContext::for_tests(2).with_admission(shared.clone());
    assert!(competing.acquire_admission(1, 1).is_err());
    finish.wait();
    assert_eq!(handle.join().unwrap().unwrap(), vec![0, 1]);
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
    let again = ctx.map_indexed(2, |index, _inner| Ok::<_, ()>(index)).unwrap();
    assert_eq!(again, vec![0, 1]);
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 64));
    assert_eq!(shared.blocked_waiters(), 0);
}

#[test]
fn failed_map_indexed_returns_the_lease() {
    let shared = AdmissionPool::fixed(2, 32);
    let ctx = parallel_context(&shared, 1);
    let saw_hold = AtomicBool::new(false);
    let result = ctx.map_indexed(4, |index, _inner| {
        if shared.available() == AdmissionRemainder::new(0, 32) {
            saw_hold.store(true, Ordering::SeqCst);
        }
        if index == 0 { Err(index) } else { Ok(index) }
    });
    assert_eq!(result, Err(0));
    assert!(saw_hold.load(Ordering::SeqCst));
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
}

#[test]
fn nested_map_indexed_does_not_wait_on_the_outer_lease() {
    let shared = AdmissionPool::fixed(2, 32);
    let ctx = parallel_context(&shared, 1);
    let output = ctx
        .map_indexed(2, |index, inner| {
            let nested = inner.map_indexed(2, |inner_index, _inner| Ok::<_, ()>(inner_index))?;
            Ok::<_, ()>((index, nested))
        })
        .unwrap();
    assert_eq!(output, vec![(0, vec![0, 1]), (1, vec![0, 1])]);
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
}

#[test]
fn map_workers_stay_within_the_pool() {
    let shared = AdmissionPool::fixed(2, 1_000);
    let ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
    let barrier = Arc::new(Barrier::new(2));
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let out = ctx
        .map_indexed(8, |index, inner| {
            assert_eq!(inner.parallelism.max_threads.get(), 1);
            let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
            peak.fetch_max(now, Ordering::SeqCst);
            assert_eq!(shared.available().threads(), 0);
            let scratch = inner.acquire_admission(0, 32).unwrap();
            barrier.wait();
            assert_eq!(shared.available().bytes(), 936);
            barrier.wait();
            drop(scratch);
            barrier.wait();
            inflight.fetch_sub(1, Ordering::SeqCst);
            Ok::<_, ()>(index)
        })
        .unwrap();
    assert_eq!(out, (0..8).collect::<Vec<_>>());
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(ctx.memory_limit_bytes(), Some(1_000));
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 1_000));
}

#[test]
fn explicit_byte_account_refuses_before_the_pool_is_charged() {
    let shared = AdmissionPool::fixed(4, 100);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let refused = study.acquire_admission(1, 101).unwrap_err();
    assert!(matches!(
        refused,
        AdmissionError::ExceedsRemainder {
            requested_threads: 1,
            requested_bytes: 101,
            remaining_threads: 4,
            remaining_bytes: 100,
        }
    ));
    let product = study.acquire_admission_product(1, 50, 3).unwrap_err();
    assert!(matches!(
        product,
        AdmissionError::ExceedsRemainder { requested_bytes: 150, remaining_bytes: 100, .. }
    ));
    assert_eq!(shared.available(), AdmissionRemainder::new(4, 100));
    assert_eq!(study.memory_limit_bytes(), Some(100));
    let granted = study.acquire_admission(1, 100).unwrap();
    assert_eq!(granted.remainder().bytes(), 100);
    assert_eq!(shared.available().bytes(), 0);
    drop(granted);
    assert_eq!(shared.available().bytes(), 100);
    assert_eq!(study.memory_limit_bytes(), Some(100));
}

#[test]
fn pool_refuses_a_request_it_cannot_fund() {
    let shared = AdmissionPool::fixed(2, 40);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    assert_eq!(study.memory_limit_bytes(), Some(40));
    let threads = study.acquire_admission(3, 10).unwrap_err();
    assert!(matches!(
        threads,
        AdmissionError::ExceedsRemainder {
            requested_threads: 3,
            requested_bytes: 10,
            remaining_threads: 2,
            remaining_bytes: 40,
        }
    ));
    let bytes = study.acquire_admission(1, 41).unwrap_err();
    assert!(matches!(
        bytes,
        AdmissionError::ExceedsRemainder {
            requested_threads: 1,
            requested_bytes: 41,
            remaining_threads: 2,
            remaining_bytes: 40,
        }
    ));
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 40));
    assert_eq!(shared.blocked_waiters(), 0);
    assert_eq!(study.memory_limit_bytes(), Some(40));
}

#[test]
fn wrapped_byte_product_is_not_admitted() {
    let count = (u64::MAX / 2) + 1;
    assert_eq!(count.wrapping_mul(2), 0);
    let shared = AdmissionPool::fixed(2, 32);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    assert!(matches!(study.acquire_admission_product(1, count, 2), Err(AdmissionError::Overflow)));
    assert!(matches!(AdmissionRequest::try_product(1, count, 2), Err(AdmissionError::Overflow)));
    assert_eq!(shared.available(), AdmissionRemainder::new(2, 32));
    assert_eq!(study.memory_limit_bytes(), Some(32));
}

#[test]
fn concurrent_child_scratch_cannot_exceed_the_parent_lease() {
    let shared = AdmissionPool::fixed(4, 500);
    let study = ExecutionContext::for_tests(1).with_admission(shared.clone());
    let parent = study.acquire_admission(2, 100).unwrap();
    let first = parent.acquire_child(1, 60).unwrap();
    let second = parent.acquire_child(1, 50).unwrap_err();
    assert!(matches!(
        second,
        AdmissionError::ExceedsRemainder {
            requested_threads: 1,
            requested_bytes: 50,
            remaining_threads: 1,
            remaining_bytes: 40,
        }
    ));
    assert_eq!(first.remainder().bytes(), 60);
    assert_eq!(parent.remainder().bytes(), 40);
    assert_eq!(shared.available().bytes(), 400);
    drop(first);
    assert_eq!(parent.remainder(), AdmissionRemainder::new(2, 100));
    drop(parent);
    assert_eq!(shared.available(), AdmissionRemainder::new(4, 500));
}
