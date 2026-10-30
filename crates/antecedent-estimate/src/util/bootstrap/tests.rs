//! Tests of the bootstrap loops, the design dispatch and the interval procedure.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use antecedent_core::execution::AdmissionPool;
use antecedent_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};
use antecedent_data::{ResamplingPlan, resample::fill_resample_index_replicate};

use super::rows::bootstrap_se_clusters_with_scratch;
use super::*;

#[test]
fn a_whole_series_contiguous_block_is_refused_before_any_refit() {
    let context = ExecutionContext::for_tests(1);
    for length in [0, 8, 9] {
        let design = BootstrapDesign {
            clusters: None,
            series_block: Some(length),
            declared: SamplingDeclaration::Undeclared,
        };
        let estimate = |_: &mut (), _: &[usize]| -> Result<Option<f64>, EstimationError> {
            panic!("an invalid block must be refused before refitting")
        };
        assert!(bootstrap_by_design(design, 20, &context, 1, 8, || (), estimate).is_err());
        assert!(
            bootstrap_interval_by_design(
                design,
                &IntervalRequest::default().rule(20),
                &context,
                1,
                8,
                || (),
                estimate
            )
            .is_err()
        );
    }
}

#[test]
fn cancelled_bootstrap_retains_actual_failures() {
    let ctx = ExecutionContext::for_tests(7);
    let calls = AtomicU32::new(0);
    let result = bootstrap_se(20, &ctx, 1, 10, |_| {
        let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 4 {
            ctx.cancellation.cancel();
        }
        Ok(if n <= 2 { Some(f64::from(n)) } else { None })
    })
    .unwrap();
    assert!(result.cancelled);
    assert!(!result.early_stopped);
    assert_eq!(result.replicates_ok, 2);
    assert_eq!(result.replicates_failed, 2);
    // The SE is the sample standard deviation (n - 1) of the two retained estimates 1 and 2.
    assert!((result.se.unwrap() - 0.5_f64.sqrt()).abs() < 1e-12, "{:?}", result.se);
}

#[test]
fn early_stopped_bootstrap_retains_actual_failures() {
    let mut ctx = ExecutionContext::for_tests(7);
    // ε = 0.5 → floor ⌈1 + 1/(2·0.25)⌉ = 3 successes.
    ctx.adaptive_bootstrap =
        AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.5 };
    let calls = AtomicU32::new(0);
    let result = bootstrap_se(20, &ctx, 1, 10, |_| {
        let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(if n <= 4 { None } else { Some(1.0) })
    })
    .unwrap();
    assert!(result.early_stopped);
    assert_eq!(result.replicates_ok, 3);
    assert_eq!(result.replicates_failed, 4);
    assert!(result.se.is_none(), "majority failure must not be hidden by early stopping");
}

#[test]
fn bootstrap_nonfinite_estimates_are_failures() {
    let ctx = ExecutionContext::for_tests(7);
    let values = [Some(1.0), Some(f64::NAN), Some(f64::INFINITY), Some(2.0)];
    let i = AtomicU32::new(0);
    let result = bootstrap_se(4, &ctx, 1, 5, |_| {
        let value = values[i.fetch_add(1, Ordering::Relaxed) as usize];
        Ok(value)
    })
    .unwrap();
    assert_eq!(result.replicates_ok, 2);
    assert_eq!(result.replicates_failed, 2);
    // Only the finite estimates 1 and 2 enter the SE: sample sd = sqrt(1/2).
    assert!((result.se.unwrap() - 0.5_f64.sqrt()).abs() < 1e-12, "{:?}", result.se);
}

#[test]
fn bootstrap_cancellation_does_not_hide_unrelated_estimation_error() {
    let ctx = ExecutionContext::for_tests(7);
    let error = EstimationError::data_msg("unrelated malformed data");
    let result = bootstrap_se(4, &ctx, 1, 5, |_| {
        ctx.cancellation.cancel();
        Err(error.clone())
    });
    assert_eq!(result.unwrap_err(), error);
}

#[test]
fn bootstrap_checks_capacity_before_allocation() {
    let ctx = ExecutionContext::for_tests(7);
    let result = bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate"));
    assert!(result.unwrap_err().to_string().contains("capacity"));
    ctx.cancellation.cancel();
    let cancelled = bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate")).unwrap();
    assert!(cancelled.cancelled);
    assert_eq!(cancelled.replicates_failed, 0);
}

#[test]
fn finalize_refuses_se_when_too_few_or_too_many_failures() {
    assert!(finalize_bootstrap_se(&[], 10).se.is_none());
    assert!(finalize_bootstrap_se(&[1.0], 10).se.is_none());
    let many_fail = finalize_bootstrap_se(&[1.0, 1.1], 10);
    assert_eq!(many_fail.replicates_ok, 2);
    assert_eq!(many_fail.replicates_failed, 8);
    assert!(many_fail.se.is_none(), "80% failure must refuse SE");
    let ok = finalize_bootstrap_se(&[1.0, 1.2, 0.9, 1.1], 4);
    // mean 1.05, squared deviations sum to 0.05, sample variance 0.05 / 3.
    assert!((ok.se.unwrap() - (0.05_f64 / 3.0).sqrt()).abs() < 1e-12, "{:?}", ok.se);
    assert_eq!(ok.replicates_failed, 0);
    assert!(!ok.early_stopped);
}

#[test]
fn finalize_early_stop_does_not_count_unattempted_as_failures() {
    let r = finalize_bootstrap_se_ex(&[1.0, 1.1, 0.9, 1.05], 4, false, true);
    assert_eq!(r.replicates_ok, 4);
    assert_eq!(r.replicates_failed, 0);
    assert!(r.early_stopped);
    // mean 1.0125, squared deviations sum to 0.021875, sample variance 0.021875 / 3.
    assert!((r.se.unwrap() - (0.021_875_f64 / 3.0).sqrt()).abs() < 1e-12, "{:?}", r.se);
}

#[test]
fn adaptive_stop_is_the_monte_carlo_error_floor() {
    // ε = 5% needs 201 successes; the explicit floor of 4 is below that.
    let budget = AdaptiveBootstrapBudget { enabled: true, min_replicates: 4, se_rel_epsilon: 0.05 };
    assert!(!adaptive_bootstrap_should_stop(budget, 4));
    assert!(!adaptive_bootstrap_should_stop(budget, 200));
    assert!(adaptive_bootstrap_should_stop(budget, 201));
    // A loose ε defers to the explicit floor.
    let floored =
        AdaptiveBootstrapBudget { enabled: true, min_replicates: 10, se_rel_epsilon: 0.5 };
    assert!(!adaptive_bootstrap_should_stop(floored, 9));
    assert!(adaptive_bootstrap_should_stop(floored, 10));
    assert!(!adaptive_bootstrap_should_stop(AdaptiveBootstrapBudget::disabled(), u32::MAX));
}

#[test]
fn bootstrap_early_stop_count_is_pinned_by_the_budget() {
    let mut ctx = ExecutionContext::for_tests(42);
    // ε = 0.2 → ⌈1 + 1/(2·0.04)⌉ = 14 successes.
    ctx.adaptive_bootstrap =
        AdaptiveBootstrapBudget { enabled: true, min_replicates: 5, se_rel_epsilon: 0.2 };
    // A constant ATE would have tripped the old relative-change rule at
    // once; the floor ignores the trajectory.
    let r1 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    let r2 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    assert!(r1.early_stopped);
    assert_eq!(r1.replicates_ok, 14);
    assert_eq!(r1.replicates_ok, r2.replicates_ok);
    assert!(!r1.cancelled);
    // A request at or below the floor is evaluated in full, not "stopped".
    let exact = bootstrap_se(14, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    assert!(!exact.early_stopped);
    assert_eq!(exact.replicates_ok, 14);
    let below = bootstrap_se(10, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    assert!(!below.early_stopped);
    assert_eq!(below.replicates_ok, 10);
}

#[test]
fn enabled_default_budget_never_stops_below_its_floor() {
    let mut ctx = ExecutionContext::for_tests(42);
    ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::enabled_default();
    let r = bootstrap_se(199, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    assert!(!r.early_stopped);
    assert_eq!(r.replicates_ok, 199);
    let r = bootstrap_se(400, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
    assert!(r.early_stopped);
    assert_eq!(r.replicates_ok, 201);
}

#[test]
fn bootstrap_cancel_and_early_stop_are_independent() {
    let mut ctx = ExecutionContext::for_tests(7);
    ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::disabled();
    let token = CancellationToken::new();
    token.cancel();
    ctx.cancellation = token;
    let r = bootstrap_se(40, &ctx, 0x1111, 10, |_| Ok(Some(1.0))).unwrap();
    assert!(r.cancelled);
    assert!(!r.early_stopped);
    assert_eq!(r.replicates_ok, 0);
}

#[test]
fn cluster_bootstrap_replicates_are_unions_of_whole_clusters_with_variable_length() {
    // Clusters of 1, 2, 3 and 5 rows, interleaved in the data.
    let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
    let clusters = ClusterRows::new(&labels).unwrap();
    assert_eq!(clusters.len(), 4);
    let sizes = [1_usize, 2, 3, 5];
    let lengths = std::sync::Mutex::new(std::collections::BTreeSet::new());
    let ctx = ExecutionContext::for_tests(11);
    let run = || {
        bootstrap_se_clusters_with_scratch(
            200,
            &ctx,
            9,
            &clusters,
            || (),
            |(), rows| {
                // Every drawn cluster is present in full, as many times as it was drawn.
                let mut seen = [0_usize; 4];
                for &row in rows {
                    seen[labels[row] as usize] += 1;
                }
                let mut draws = 0;
                for (label, count) in seen.iter().enumerate() {
                    assert_eq!(count % sizes[label], 0, "cluster {label} was split");
                    draws += count / sizes[label];
                }
                assert_eq!(draws, 4, "a replicate draws as many clusters as the data has");
                lengths.lock().unwrap().insert(rows.len());
                Ok(Some(rows.len() as f64))
            },
        )
        .unwrap()
    };
    let first = run();
    assert_eq!(first.replicates_ok, 200);
    assert!(lengths.lock().unwrap().len() > 3, "replicate lengths must vary");
    // Same seed, same draws.
    assert_eq!(run().se, first.se);
    // One cluster is not resamplable.
    assert!(matches!(
        ClusterRows::new(&[7, 7, 7]),
        Err(EstimationError::Stats(antecedent_stats::StatsError::FewClusters {
            clusters: 1,
            minimum: 2,
            ..
        }))
    ));
}

#[test]
fn cluster_replicate_ceiling_covers_the_sample_and_is_enforced_when_skewed() {
    let mut labels = vec![0_u32; 50];
    labels.extend((1..=20_u32).flat_map(|label| [label, label]));
    let clusters = ClusterRows::new(&labels).unwrap();
    let ceiling = clusters.replicate_row_ceiling();
    assert!(ceiling >= clusters.row_count());
    assert!(ceiling <= clusters.len() * 50);
    let ctx = ExecutionContext::for_tests(7);
    let tight = ReplicateRows::Clusters {
        members: clusters.members(),
        ceiling: 49,
        ctx: &ctx,
        stream_base: 7,
    };
    let refused = tight.fill(0, &mut Vec::new(), &mut Vec::new()).unwrap_err();
    assert!(refused.to_string().contains("exceeds the admitted index buffer"), "{refused}");
    let admitted =
        ReplicateRows::Clusters { members: clusters.members(), ceiling, ctx: &ctx, stream_base: 7 };
    let mut out = Vec::new();
    admitted.fill(0, &mut out, &mut Vec::new()).unwrap();
    assert!(out.len() <= ceiling);
}

#[test]
fn cluster_members_follow_ascending_labels() {
    let labels = [2_u32, 0, 2, 0];
    let clusters = ClusterRows::new(&labels).unwrap();
    let ids = |slot: usize| clusters.members()[slot].iter().map(|id| id.get()).collect::<Vec<_>>();
    assert_eq!(ids(0), vec![1, 3]);
    assert_eq!(ids(1), vec![0, 2]);
    let ctx = ExecutionContext::for_tests(9);
    let stream = 0xA1D5_u64;
    let mut from_rows = Vec::new();
    ReplicateRows::Clusters {
        members: clusters.members(),
        ceiling: clusters.replicate_row_ceiling(),
        ctx: &ctx,
        stream_base: stream,
    }
    .fill(0, &mut from_rows, &mut Vec::new())
    .unwrap();
    let mut from_data = Vec::new();
    fill_resample_index_replicate(
        ResamplingPlan::ClusterBootstrap { cluster: antecedent_core::VariableId::from_raw(0) },
        labels.len(),
        Some(&labels),
        &ctx,
        stream,
        0,
        &mut from_data,
    )
    .unwrap();
    let data_rows: Vec<usize> =
        from_data.iter().map(|id| usize::try_from(id.get()).unwrap()).collect();
    assert_eq!(from_rows, data_rows);
}

#[test]
fn cluster_bootstrap_writes_one_replicate_per_worker() {
    let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
    let clusters = ClusterRows::new(&labels).unwrap();
    let groups = clusters.len();
    let replicates = 80u32;
    let workers = 4usize;
    let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
    let matrix_bytes = groups * usize::try_from(replicates).unwrap() * width;
    let retained_bytes = clusters.replicate_row_ceiling() * workers * width;
    assert!(retained_bytes < matrix_bytes);
    let ctx = ExecutionContext::production(11, u32::try_from(workers).unwrap()).with_admission(
        AdmissionPool::fixed(
            u32::try_from(workers).unwrap(),
            u64::try_from(retained_bytes).unwrap(),
        ),
    );
    let calls = AtomicU32::new(0);
    let result = bootstrap_se_clusters_with_scratch(
        replicates,
        &ctx,
        9,
        &clusters,
        || (),
        |(), rows| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert!(!rows.is_empty());
            Ok(Some(rows.len() as f64))
        },
    )
    .unwrap();
    assert_eq!(result.replicates_ok, replicates);
    assert_eq!(calls.load(Ordering::Relaxed), replicates);
    let refused = ExecutionContext::production(11, u32::try_from(workers).unwrap()).with_admission(
        AdmissionPool::fixed(
            u32::try_from(workers).unwrap(),
            u64::try_from(retained_bytes - 1).unwrap(),
        ),
    );
    let refused_calls = AtomicU32::new(0);
    let error = bootstrap_se_clusters_with_scratch(
        replicates,
        &refused,
        9,
        &clusters,
        || (),
        |(), _rows| {
            refused_calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(1.0_f64))
        },
    )
    .unwrap_err();
    assert_eq!(refused_calls.load(Ordering::Relaxed), 0);
    assert!(error.to_string().contains("cannot fund"), "{error}");
}

#[test]
fn bootstrap_workers_stay_within_the_pool() {
    let shared = AdmissionPool::fixed(2, 1 << 20);
    let ctx = ExecutionContext::production(3, 8).with_admission(shared.clone());
    let barrier = Arc::new(Barrier::new(2));
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let n = 5usize;
    let result = bootstrap_se(8, &ctx, 1, n, |idx| {
        assert_eq!(idx.len(), n);
        let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
        peak.fetch_max(now, Ordering::SeqCst);
        barrier.wait();
        inflight.fetch_sub(1, Ordering::SeqCst);
        Ok(Some(1.0))
    })
    .unwrap();
    let widest = peak.load(Ordering::SeqCst);
    assert_eq!(widest, 2);
    let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
    let scratch_bytes = n * widest * width;
    assert!(scratch_bytes <= usize::try_from(shared.byte_budget()).unwrap());
    assert_eq!(result.replicates_ok, 8);
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(ctx.memory_limit_bytes(), Some(1 << 20));
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), 1 << 20);
}

#[test]
fn bootstrap_scratch_products_use_checked_arithmetic() {
    let calls = AtomicUsize::new(0);
    let estimate = |_: &[usize]| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(1.0))
    };
    let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
    // One buffer is addressable but two workers' buffers are not.
    let ctx = ExecutionContext::production(1, 2);
    let n = usize::try_from(isize::MAX).unwrap() / width;
    let error = bootstrap_se(4, &ctx, 1, n, estimate).unwrap_err();
    assert!(error.to_string().contains("addressable capacity"), "{error}");
    // The byte count of a single buffer wraps to zero.
    let serial = ExecutionContext::for_tests(1);
    let n = usize::MAX / width + 1;
    let error = bootstrap_se(4, &serial, 1, n, estimate).unwrap_err();
    assert!(error.to_string().contains("addressable capacity"), "{error}");
    // The cluster path refuses an oversized row count before any replicate.
    let _clusters = ClusterRows::new(&[0, 0, 1, 1]).unwrap();
    let error = retained_index_slots(usize::MAX / width + 1, 1).unwrap_err();
    assert!(error.to_string().contains("addressable capacity"), "{error}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_concurrent_bootstrap_finds_the_pool_held_and_stays_on_its_caller() {
    let n = 5usize;
    let shared = AdmissionPool::fixed(2, 1 << 20);
    let ctx = ExecutionContext::production(3, 8).with_admission(shared.clone());
    let ready = Barrier::new(3);
    let done = Barrier::new(3);
    let second_threads = std::sync::Mutex::new(std::collections::HashSet::new());
    std::thread::scope(|scope| {
        let holder = scope.spawn(|| {
            bootstrap_se(2, &ctx, 1, n, |_| {
                ready.wait();
                done.wait();
                Ok(Some(1.0))
            })
            .unwrap();
        });
        // Both holder workers are inside the estimate, so the pool's slots are leased.
        ready.wait();
        assert_eq!(shared.available().threads(), 0);
        let result = bootstrap_se(8, &ctx, 2, n, |_| {
            second_threads.lock().unwrap().insert(std::thread::current().id());
            Ok(Some(1.0))
        })
        .unwrap();
        assert_eq!(result.replicates_ok, 8);
        done.wait();
        holder.join().unwrap();
    });
    assert_eq!(second_threads.lock().unwrap().len(), 1);
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), 1 << 20);
}

#[test]
fn a_pool_that_exactly_funds_the_bootstrap_scratch_runs_it() {
    let n = 8usize;
    let workers = 2usize;
    let bytes = u64::try_from(n * workers).unwrap() * u64::try_from(size_of::<u64>()).unwrap();
    let shared = AdmissionPool::fixed(u32::try_from(workers).unwrap(), bytes);
    let ctx = ExecutionContext::production(1, u32::try_from(workers).unwrap())
        .with_admission(shared.clone());
    let calls = AtomicUsize::new(0);
    let result = bootstrap_se(4, &ctx, 1, n, |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(1.0))
    })
    .unwrap();
    assert_eq!(result.replicates_ok, 4);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(ctx.memory_limit_bytes(), Some(bytes));
    assert_eq!(shared.available().bytes(), bytes);
}

#[test]
fn bootstrap_refuses_when_the_pool_cannot_fund_scratch() {
    let n = 8usize;
    let workers = 2usize;
    let bytes = u64::try_from(n * workers).unwrap() * u64::try_from(size_of::<u64>()).unwrap();
    let shared = AdmissionPool::fixed(u32::try_from(workers).unwrap(), bytes - 1);
    let ctx = ExecutionContext::production(1, u32::try_from(workers).unwrap())
        .with_admission(shared.clone());
    let calls = AtomicUsize::new(0);
    let error = bootstrap_se(4, &ctx, 1, n, |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(1.0))
    })
    .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(error.to_string().contains("cannot fund"), "{error}");
    assert_eq!(shared.available().threads(), u32::try_from(workers).unwrap());
    assert_eq!(shared.available().bytes(), bytes - 1);
    assert_eq!(ctx.memory_limit_bytes(), Some(bytes - 1));
}

fn design_run(
    design: BootstrapDesign<'_>,
    n: usize,
) -> (Result<BootstrapSeResult, EstimationError>, usize) {
    let ctx = ExecutionContext::for_tests(5);
    let calls = AtomicUsize::new(0);
    let result = bootstrap_by_design(
        design,
        8,
        &ctx,
        3,
        n,
        || (),
        |(), _rows| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(1.0))
        },
    );
    (result, calls.load(Ordering::Relaxed))
}

fn is_design_refusal(result: &Result<BootstrapSeResult, EstimationError>) -> bool {
    matches!(result, Err(EstimationError::Data(DataError::SamplingDesign(_))))
}

#[test]
fn a_row_level_bootstrap_is_refused_under_every_declared_design() {
    for design in DeclaredSamplingDesign::ALL {
        let (result, calls) = design_run(
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Declared(design),
            },
            12,
        );
        assert!(is_design_refusal(&result), "{design}: row draw was accepted");
        assert_eq!(calls, 0, "{design}: a refused draw evaluated replicates");
    }
}

#[test]
fn an_undeclared_design_takes_the_one_the_grouping_implies() {
    let undeclared = SamplingDeclaration::Undeclared;
    let labels = [0_u32, 0, 1, 1, 2, 2];
    let (rows, _) =
        design_run(BootstrapDesign { clusters: None, series_block: None, declared: undeclared }, 6);
    // All 8 requested replicates are drawn and kept, so the implied design was accepted, not skipped.
    assert_eq!(rows.expect("the grouping implies a design").replicates_ok, 8);
    let (clusters, _) = design_run(
        BootstrapDesign { clusters: Some(&labels), series_block: None, declared: undeclared },
        6,
    );
    // All 8 requested replicates are drawn and kept, so the implied design was accepted, not skipped.
    assert_eq!(clusters.expect("the grouping implies a design").replicates_ok, 8);
    let (blocks, _) = design_run(
        BootstrapDesign { clusters: None, series_block: Some(2), declared: undeclared },
        6,
    );
    // All 8 requested replicates are drawn and kept, so the implied design was accepted, not skipped.
    assert_eq!(blocks.expect("the grouping implies a design").replicates_ok, 8);
}

#[test]
fn a_trajectory_bootstrap_draws_whole_subjects_under_the_trajectory_design() {
    let subjects = [0_u32, 0, 0, 1, 1, 1, 2, 2, 2];
    let ctx = ExecutionContext::for_tests(5);
    let result = bootstrap_by_design(
        BootstrapDesign {
            clusters: Some(&subjects),
            series_block: None,
            declared: SamplingDeclaration::Declared(
                DeclaredSamplingDesign::IndependentSubjectTrajectories,
            ),
        },
        16,
        &ctx,
        3,
        subjects.len(),
        || (),
        |(), rows| {
            let mut seen = [0_usize; 3];
            for &row in rows {
                seen[subjects[row] as usize] += 1;
            }
            assert!(seen.iter().all(|count| count % 3 == 0), "a trajectory was split");
            Ok(Some(1.0))
        },
    )
    .expect("whole-subject draws are the trajectory rule");
    assert_eq!(result.replicates_ok, 16);
}

#[test]
fn a_declared_design_refuses_the_grouping_that_contradicts_it() {
    let labels = [0_u32, 0, 1, 1];
    let series = SamplingDeclaration::Declared(DeclaredSamplingDesign::DependentTimeSeries);
    let panel = SamplingDeclaration::Declared(DeclaredSamplingDesign::ClusteredPanel);
    let (clusters_under_series, _) = design_run(
        BootstrapDesign { clusters: Some(&labels), series_block: None, declared: series },
        4,
    );
    assert!(is_design_refusal(&clusters_under_series));
    let (blocks_under_panel, _) =
        design_run(BootstrapDesign { clusters: None, series_block: Some(2), declared: panel }, 4);
    assert!(is_design_refusal(&blocks_under_panel));
    let (blocks_under_series, _) =
        design_run(BootstrapDesign { clusters: None, series_block: Some(2), declared: series }, 4);
    assert!(blocks_under_series.is_ok());
}

/// Prints one wall-clock trial per line. Ignored unless the run asks for it.
mod bootstrap_clock_trials {
    use std::env;
    use std::time::Instant;

    use antecedent_core::ExecutionContext;

    use super::bootstrap_se_with_scratch;

    fn env_u64(name: &str, default: u64) -> u64 {
        env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    #[test]
    #[ignore = "prints bootstrap scratch wall-clock trials"]
    fn print_bootstrap_scratch_trials() {
        let n = usize::try_from(env_u64("N", 10_000)).unwrap_or(10_000);
        let replicates = u32::try_from(env_u64("REPLICATES", 50)).unwrap_or(50);
        let threads = u32::try_from(env_u64("THREADS", 1)).unwrap_or(1);
        let trials = usize::try_from(env_u64("TRIALS", 11)).unwrap_or(11);
        let ctx = ExecutionContext::production(11, threads);
        let run = || {
            bootstrap_se_with_scratch(
                replicates,
                &ctx,
                9,
                n,
                || (),
                |(), idx| Ok(Some(idx.len() as f64)),
            )
        };
        for _ in 0..2 {
            run().expect("warmup");
        }
        println!("n={n} replicates={replicates} threads={threads} trials={trials}");
        for trial in 0..trials {
            let started = Instant::now();
            let result = run().expect("trial");
            println!(
                "trial={trial} nanos={} replicates_ok={} cancelled={}",
                started.elapsed().as_nanos(),
                result.replicates_ok,
                result.cancelled
            );
        }
    }
}

mod interval_procedure_tests {
    use std::cell::RefCell;

    use super::*;
    use crate::adaptive_resample_waves::StopReason;
    use crate::statistical_transport::percentile_interval;

    const N: usize = 40;

    fn statistic(rows: &[usize]) -> f64 {
        rows.iter().map(|row| ((row * 37) % 11) as f64).sum::<f64>() / rows.len() as f64
    }

    fn labels() -> Vec<u32> {
        (0..N).map(|row| u32::try_from(row / 4).unwrap()).collect()
    }

    /// Every replicate-row design the dispatcher resolves: single rows, whole clusters and
    /// contiguous blocks.
    fn designs(labels: &[u32]) -> Vec<BootstrapDesign<'_>> {
        vec![
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            BootstrapDesign {
                clusters: Some(labels),
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            BootstrapDesign {
                clusters: None,
                series_block: Some(4),
                declared: SamplingDeclaration::Undeclared,
            },
        ]
    }

    fn rule(wave_size: u32, max_replicates: u32, se_rel_tol: f64) -> WaveStoppingRule {
        IntervalRequest {
            level: 0.9,
            wave_size,
            min_replicates: 0,
            se_rel_tol: Some(se_rel_tol),
            half_width_rel_tol: None,
        }
        .rule(max_replicates)
    }

    #[test]
    fn a_never_met_request_reproduces_the_fixed_count_bootstrap_on_every_design() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let fixed = bootstrap_by_design(
                design,
                60,
                &ctx,
                21,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let seen = RefCell::new(Vec::new());
            let run = bootstrap_interval_by_design(
                design,
                &rule(15, 60, 1e-12),
                &ctx,
                21,
                N,
                || (),
                |(), rows| {
                    let value = statistic(rows);
                    seen.borrow_mut().push(value);
                    Ok(Some(value))
                },
            )
            .unwrap();
            assert_eq!(run.stop, StopReason::MaxReplicates);
            assert_eq!((run.attempted, run.waves.len()), (60, 4));
            assert_eq!(run.se.replicates_ok, fixed.replicates_ok);
            assert_eq!(run.se.se.map(f64::to_bits), fixed.se.map(f64::to_bits));
            // The interval is the type-7 percentile of exactly the successes the fixed count draws.
            assert_eq!(run.interval, percentile_interval(&seen.borrow(), 0.9));
            assert!(run.se_rel_mc_error.is_some());
        }
    }

    #[test]
    fn a_met_tolerance_stops_on_a_wave_boundary_with_the_prefix_of_the_full_run() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let record = |rule: &WaveStoppingRule| {
                let seen = RefCell::new(Vec::new());
                let run = bootstrap_interval_by_design(
                    design,
                    rule,
                    &ctx,
                    21,
                    N,
                    || (),
                    |(), rows| {
                        let value = statistic(rows);
                        seen.borrow_mut().push(value);
                        Ok(Some(value))
                    },
                )
                .unwrap();
                (run, seen.into_inner())
            };
            let (full, all) = record(&rule(20, 400, 1e-12));
            assert_eq!(full.attempted, 400);
            let mut loose = rule(20, 400, 0.3);
            loose.min_replicates = 40;
            let (stopped, prefix) = record(&loose);
            assert_eq!(stopped.stop, StopReason::Converged);
            assert!(stopped.se.early_stopped);
            assert_eq!((stopped.attempted, stopped.waves.len()), (40, 2));
            assert_eq!(prefix, all[..40]);
            assert!(stopped.se_rel_mc_error.is_some_and(|error| error <= 0.3));
        }
    }

    #[test]
    fn the_dispatcher_without_a_request_is_the_fixed_count_bootstrap_bit_for_bit() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let fixed = bootstrap_by_design(
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let outcome = bootstrap_outcome_by_design(
                None,
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            assert_eq!(outcome.se, fixed);
            assert!(outcome.resampling.is_none());
            // A request reports the procedure beside the same replicate cap.
            let request = IntervalRequest { min_replicates: 10, ..IntervalRequest::default() };
            let requested = bootstrap_outcome_by_design(
                Some(&request),
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let report = requested.resampling.expect("a requested interval is reported");
            assert_eq!(report.rule.max_replicates, 50);
            assert_eq!(report.attempted, requested.se.replicates_ok);
            assert!(report.interval.is_some() && !report.waves.is_empty());
        }
    }

    #[test]
    fn failures_stay_in_the_attempts_and_withhold_the_interval_with_the_standard_error() {
        let ctx = ExecutionContext::for_tests(5);
        let calls = RefCell::new(0u32);
        let run = bootstrap_interval_by_design(
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            &rule(10, 40, 1e-12),
            &ctx,
            8,
            N,
            || (),
            |(), rows| {
                *calls.borrow_mut() += 1;
                Ok((*calls.borrow() % 4 == 0).then(|| statistic(rows)))
            },
        )
        .unwrap();
        assert_eq!(run.attempted, 40);
        assert_eq!((run.se.replicates_ok, run.se.replicates_failed), (10, 30));
        assert!(run.waves.iter().all(|wave| wave.declined == 7 || wave.declined == 8));
        // Three quarters failed: the fixed-count bootstrap withholds the SE, and the report
        // withholds the interval it would otherwise compute from the survivors.
        assert!(run.se.se.is_none());
        assert!(run.interval.is_some());
        assert!(run.report(rule(10, 40, 1e-12)).interval.is_none());
    }

    #[test]
    fn cancellation_returns_the_completed_waves_flagged() {
        let ctx = ExecutionContext::for_tests(5);
        ctx.cancellation.cancel();
        let run = bootstrap_interval_by_design(
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            &rule(10, 40, 1e-12),
            &ctx,
            8,
            N,
            || (),
            |(), rows| Ok(Some(statistic(rows))),
        )
        .unwrap();
        assert_eq!(run.stop, StopReason::Cancelled);
        assert!(run.se.cancelled && run.interval.is_none() && run.waves.is_empty());
    }
}
