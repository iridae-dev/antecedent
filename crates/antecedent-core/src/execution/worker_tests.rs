use super::pipeline::FAIL_NEXT_WORKER_SPAWN;
use super::{
    AdmissionError, AdmissionPool, AdmissionRemainder, ExecutionContext, MapExecutionError,
    PipelineStep,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn inline_pipeline_does_not_consume_after_producer_cancels() {
    let ctx = ExecutionContext::for_tests(1);
    let consumed = AtomicUsize::new(0);
    let result = ctx.pipeline_indexed(
        2,
        1,
        |index, inner| {
            inner.cancellation.cancel();
            Ok::<_, ()>(index)
        },
        |_, _| {
            consumed.fetch_add(1, Ordering::SeqCst);
            Ok(PipelineStep::Continue)
        },
    );
    assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::Cancelled))));
    assert_eq!(consumed.load(Ordering::SeqCst), 0);
}

#[test]
fn map_cancelled_before_scratch_and_if_free_refusal_are_typed() {
    let pool = AdmissionPool::fixed(1, 1024);
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let lease = ctx.acquire_admission(1, 0).unwrap();
    let made = AtomicUsize::new(0);
    let result = ctx.map_indexed_with_scratch_if_free(
        1,
        || {
            made.fetch_add(1, Ordering::SeqCst);
        },
        |_, _, ()| Ok::<_, ()>(()),
    );
    assert!(matches!(
        result,
        Err(MapExecutionError::Admission(AdmissionError::ExceedsRemainder { .. }))
    ));
    assert_eq!(made.load(Ordering::SeqCst), 0);
    drop(lease);
    ctx.cancellation.cancel();
    let result = ctx.map_indexed_with_scratch(
        1,
        || {
            made.fetch_add(1, Ordering::SeqCst);
        },
        |_, _, ()| Ok::<_, ()>(()),
    );
    assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::Cancelled))));
    assert_eq!(made.load(Ordering::SeqCst), 0);
}

#[test]
fn worker_spawn_failure_returns_typed_refusal_and_releases_lease() {
    let pool = AdmissionPool::fixed(2, 1024);
    let ctx = ExecutionContext::production(1, 2).with_admission(pool.clone());
    FAIL_NEXT_WORKER_SPAWN.with(|flag| flag.set(true));
    let result = ctx.map_indexed(2, |i, _| Ok::<_, ()>(i));
    assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::WorkerSpawn))));
    assert_eq!(pool.available(), AdmissionRemainder::new(2, 1024));
    FAIL_NEXT_WORKER_SPAWN.with(|flag| flag.set(true));
    let result =
        ctx.pipeline_indexed(2, 2, |i, _| Ok::<_, ()>(i), |_, _| Ok(PipelineStep::Continue));
    assert!(matches!(result, Err(MapExecutionError::Admission(AdmissionError::WorkerSpawn))));
    assert_eq!(pool.available(), AdmissionRemainder::new(2, 1024));
}

#[test]
fn zero_thread_budget_refuses_before_work() {
    let pool = AdmissionPool::fixed(0, 128);
    let ctx = ExecutionContext::production(1, 2).with_admission(pool.clone());
    let ran = AtomicUsize::new(0);
    let result = ctx.map_indexed(3, |_, _| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok::<_, ()>(())
    });
    assert!(matches!(
        result,
        Err(MapExecutionError::Admission(AdmissionError::ExceedsRemainder { .. }))
    ));
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}
