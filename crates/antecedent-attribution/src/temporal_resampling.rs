//! Run-aware temporal block resampling, v2.
//!
//! The primary generator consumes exactly one draw for each legacy block.
//! Whenever that block crosses a source-run boundary, each subsequent segment
//! uses a deterministic derived stream keyed by the primary draw, block index,
//! run index and segment index. A bootstrap segment starts within its source
//! run and wraps circularly within that run; an overall end-to-start wrap is a
//! distinct derived segment even for one run. Short runs are supported without
//! dropping samples. The effective weights deliberately differ from v1.

use crate::AttributionError;
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{CausalRng, ExecutionContext, TemporalBudget, derive_stream};
use std::ops::{Deref, Range};

/// Version of the complete-window run-aware block and substream law.
pub const TEMPORAL_BLOCK_LAW_VERSION: u16 = 2;
const SEGMENT_DOMAIN: u64 = 0x7465_6d70_5f73_6567;

/// Sampled weights and the admission retained for their lifetime.
pub struct RunWeights {
    values: Vec<f64>,
    _lease: AdmissionLease,
}
impl Deref for RunWeights {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Drawn compact row positions in block order and their retained admission.
pub struct RunRows {
    values: Vec<usize>,
    _lease: AdmissionLease,
}
impl Deref for RunRows {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

fn setup(
    n: usize,
    runs: &[Range<usize>],
    block: usize,
    budget: &mut TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(usize, RunWeights), AttributionError> {
    if ctx.cancellation.is_cancelled() {
        return Err(AttributionError::Cancelled);
    }
    let run_work = u64::try_from(runs.len())
        .map_err(|_| AttributionError::Budget { message: "temporal run count overflow".into() })?;
    budget
        .charge_work(run_work)
        .map_err(|e| AttributionError::Budget { message: e.to_string() })?;
    let mut end = 0;
    for run in runs {
        if ctx.cancellation.is_cancelled() {
            return Err(AttributionError::Cancelled);
        }
        if run.start != end || run.start >= run.end || run.end > n {
            return Err(AttributionError::InvalidInput {
                message: "temporal runs must partition the compact rows",
            });
        }
        end = run.end;
    }
    if n == 0 || end != n || block == 0 {
        return Err(AttributionError::InvalidInput {
            message: "temporal block resampling needs nonempty rows and positive block length",
        });
    }
    let width = (usize::BITS - runs.len().leading_zeros()) as u64;
    let work = u64::try_from(n).ok().and_then(|n| n.checked_mul(width + 8)).ok_or_else(|| {
        AttributionError::Budget { message: "temporal resampling work overflow".into() }
    })?;
    budget.charge_work(work).map_err(|e| AttributionError::Budget { message: e.to_string() })?;
    let per_row = (std::mem::size_of::<f64>() + std::mem::size_of::<usize>()) as u64;
    let bytes = u64::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(per_row))
        .and_then(|v| v.checked_add(1024))
        .ok_or_else(|| AttributionError::Budget {
            message: "temporal resampling bytes overflow".into(),
        })?;
    if bytes > budget.limits().max_bytes {
        return Err(AttributionError::Budget { message: "temporal resampling byte limit".into() });
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        AdmissionError::Cancelled => AttributionError::Cancelled,
        other => AttributionError::Budget { message: other.to_string() },
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| AttributionError::Budget {
        message: "temporal resampling allocation refused".into(),
    })?;
    for _ in 0..n {
        if ctx.cancellation.is_cancelled() {
            return Err(AttributionError::Cancelled);
        }
        values.push(0.0);
    }
    Ok((block.min(n), RunWeights { values, _lease: lease }))
}
fn segment_rng(
    primary: u64,
    replicate: u64,
    block: usize,
    run: usize,
    segment: usize,
) -> CausalRng {
    let mut key = derive_stream(SEGMENT_DOMAIN, replicate);
    key = derive_stream(key, primary);
    key = derive_stream(key, block as u64);
    key = derive_stream(key, run as u64);
    CausalRng::from_seed(derive_stream(key, segment as u64))
}
fn exponential(rng: &mut CausalRng) -> f64 {
    let mut value = [0.0];
    antecedent_kernels::fill_bayesian_bootstrap_weights(rng, &mut value, false);
    value[0]
}
fn run_at(runs: &[Range<usize>], row: usize) -> usize {
    runs.partition_point(|run| run.end <= row)
}

/// Bayesian block weights with separate draws for disconnected segments.
///
/// # Errors
/// Invalid run partitions, limits, or cancellation.
pub fn bayesian_run_weights(
    n: usize,
    block: usize,
    runs: &[Range<usize>],
    replicate: u64,
    rng: &mut CausalRng,
    budget: &mut TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<RunWeights, AttributionError> {
    let (block, mut output) = setup(n, runs, block, budget, ctx)?;
    for (block_index, start) in (0..n).step_by(block).enumerate() {
        let end = start.saturating_add(block).min(n);
        let primary = exponential(rng);
        let mut row = start;
        let mut segment = 0;
        while row < end {
            if ctx.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            let run = run_at(runs, row);
            let until = runs[run].end.min(end);
            let weight = if segment == 0 {
                primary
            } else {
                let mut derived =
                    segment_rng(primary.to_bits(), replicate, block_index, run, segment);
                exponential(&mut derived)
            };
            for chunk in output.values[row..until].chunks_mut(1024) {
                if ctx.cancellation.is_cancelled() {
                    return Err(AttributionError::Cancelled);
                }
                chunk.fill(weight);
            }
            row = until;
            segment += 1;
        }
    }
    Ok(output)
}

/// Circular block counts, splitting disconnected run segments into independent
/// derived streams without changing the number of primary block draws.
///
/// # Errors
/// Invalid run partitions, limits, or cancellation.
pub fn bootstrap_run_counts(
    n: usize,
    block: usize,
    runs: &[Range<usize>],
    replicate: u64,
    rng: &mut CausalRng,
    budget: &mut TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<RunWeights, AttributionError> {
    let (block, mut output) = setup(n, runs, block, budget, ctx)?;
    let mut drawn = 0;
    let mut block_index = 0;
    while drawn < n {
        if ctx.cancellation.is_cancelled() {
            return Err(AttributionError::Cancelled);
        }
        let u = rng.next_f64();
        let start = ((u * n as f64) as usize).min(n - 1);
        let take = block.min(n - drawn);
        let mut consumed = 0;
        let mut segment = 0;
        while consumed < take {
            if ctx.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            let source =
                if consumed >= n - start { consumed - (n - start) } else { start + consumed };
            let run = run_at(runs, source);
            let segment_len = (runs[run].end - source).min(take - consumed);
            let chosen = if segment == 0 {
                source
            } else {
                let mut derived = segment_rng(u.to_bits(), replicate, block_index, run, segment);
                runs[run].start
                    + ((derived.next_f64() * (runs[run].end - runs[run].start) as f64) as usize)
                        .min(runs[run].end - runs[run].start - 1)
            };
            for offset in 0..segment_len {
                if ctx.cancellation.is_cancelled() {
                    return Err(AttributionError::Cancelled);
                }
                let len = runs[run].end - runs[run].start;
                let base = chosen - runs[run].start;
                let within =
                    if offset >= len - base { offset - (len - base) } else { base + offset };
                output.values[runs[run].start + within] += 1.0;
            }
            consumed += segment_len;
            segment += 1;
        }
        drawn += take;
        block_index += 1;
    }
    Ok(output)
}
/// Draw compact row positions in their block order, preserving the primary
/// stream while deriving separate streams at disconnected run boundaries.
///
/// # Errors
/// Invalid runs, resource limits, or cancellation.
pub fn bootstrap_run_rows(
    n: usize,
    block: usize,
    runs: &[Range<usize>],
    replicate: u64,
    rng: &mut CausalRng,
    budget: &mut TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<RunRows, AttributionError> {
    let (block, admitted) = setup(n, runs, block, budget, ctx)?;
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| AttributionError::Budget {
        message: "temporal resampling row allocation refused".into(),
    })?;
    let mut drawn = 0;
    let mut block_index = 0;
    while drawn < n {
        if ctx.cancellation.is_cancelled() {
            return Err(AttributionError::Cancelled);
        }
        let u = rng.next_f64();
        let start = ((u * n as f64) as usize).min(n - 1);
        let take = block.min(n - drawn);
        let mut consumed = 0;
        let mut segment = 0;
        while consumed < take {
            if ctx.cancellation.is_cancelled() {
                return Err(AttributionError::Cancelled);
            }
            let source =
                if consumed >= n - start { consumed - (n - start) } else { start + consumed };
            let run = run_at(runs, source);
            let segment_len = (runs[run].end - source).min(take - consumed);
            let chosen = if segment == 0 {
                source
            } else {
                let mut derived = segment_rng(u.to_bits(), replicate, block_index, run, segment);
                runs[run].start
                    + ((derived.next_f64() * (runs[run].end - runs[run].start) as f64) as usize)
                        .min(runs[run].end - runs[run].start - 1)
            };
            for offset in 0..segment_len {
                if ctx.cancellation.is_cancelled() {
                    return Err(AttributionError::Cancelled);
                }
                let len = runs[run].end - runs[run].start;
                let base = chosen - runs[run].start;
                let within =
                    if offset >= len - base { offset - (len - base) } else { base + offset };
                values.push(runs[run].start + within);
            }
            consumed += segment_len;
            segment += 1;
        }
        drawn += take;
        block_index += 1;
    }
    Ok(RunRows { values, _lease: admitted._lease })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmented_weights_use_derived_stream_and_keep_primary_stream_position() {
        let ctx = ExecutionContext::for_tests(4);
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let mut actual_rng = CausalRng::from_seed(41);
        let actual =
            bayesian_run_weights(8, 4, &[0..2, 2..5, 5..8], 7, &mut actual_rng, &mut budget, &ctx)
                .unwrap();
        let mut primary = CausalRng::from_seed(41);
        let first = antecedent_kernels::bayesian_bootstrap_weights(&mut primary, 1, false)[0];
        let second = antecedent_kernels::bayesian_bootstrap_weights(&mut primary, 1, false)[0];
        let seed = derive_stream(
            derive_stream(
                derive_stream(derive_stream(derive_stream(SEGMENT_DOMAIN, 7), first.to_bits()), 0),
                1,
            ),
            1,
        );
        let mut derived = CausalRng::from_seed(seed);
        let extra = exponential(&mut derived);
        assert_eq!(&actual[0..2], &[first, first]);
        assert_eq!(&actual[2..4], &[extra, extra]);
        assert_eq!(actual[4], second);
        assert_ne!(first.to_bits(), extra.to_bits());
        assert_eq!(actual_rng.next_u64(), primary.next_u64());
    }

    #[test]
    fn block_counts_preserve_mass_and_primary_draw_count_after_splits() {
        let ctx = ExecutionContext::for_tests(4);
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let mut split_rng = CausalRng::from_seed(9);
        let counts =
            bootstrap_run_counts(10, 4, &[0..3, 3..7, 7..10], 7, &mut split_rng, &mut budget, &ctx)
                .unwrap();
        assert_eq!(counts.iter().sum::<f64>(), 10.0);
        assert!(counts.iter().all(|n| *n >= 0.0 && n.fract() == 0.0));
        let mut original_rng = CausalRng::from_seed(9);
        for _ in 0..3 {
            let _ = original_rng.next_f64();
        }
        assert_eq!(split_rng.next_u64(), original_rng.next_u64());
        let mut rows_rng = CausalRng::from_seed(9);
        let rows =
            bootstrap_run_rows(10, 4, &[0..3, 3..7, 7..10], 7, &mut rows_rng, &mut budget, &ctx)
                .unwrap();
        let mut from_rows = [0.0; 10];
        for &row in rows.iter() {
            from_rows[row] += 1.0;
        }
        assert_eq!(&*counts, &from_rows);
        let _ = rows_rng.next_u64();
        assert_eq!(rows_rng.next_u64(), split_rng.next_u64());
    }
    #[test]
    fn single_run_wrap_uses_a_derived_segment_without_advancing_primary_stream() {
        let seed = (0..100_u64)
            .find(|&seed| {
                let mut r = CausalRng::from_seed(seed);
                (r.next_f64() * 3.0) as usize > 0
            })
            .unwrap();
        let ctx = ExecutionContext::for_tests(0);
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let mut rng = CausalRng::from_seed(seed);
        let result = bootstrap_run_counts(3, 3, &[0..3], 5, &mut rng, &mut budget, &ctx).unwrap();
        let mut primary = CausalRng::from_seed(seed);
        let u = primary.next_f64();
        let start = ((u * 3.0) as usize).min(2);
        let mut expected = [0.0; 3];
        for slot in &mut expected[start..] {
            *slot += 1.0;
        }
        let key = derive_stream(
            derive_stream(
                derive_stream(derive_stream(derive_stream(SEGMENT_DOMAIN, 5), u.to_bits()), 0),
                0,
            ),
            1,
        );
        let mut derived = CausalRng::from_seed(key);
        let first = ((derived.next_f64() * 3.0) as usize).min(2);
        for k in 0..start {
            expected[(first + k) % 3] += 1.0;
        }
        assert_eq!(&*result, &expected);
        assert_eq!(rng.next_u64(), primary.next_u64());
    }

    #[test]
    fn one_row_runs_and_retained_lease_remain_bounded() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 4096);
        let ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let before = pool.available().bytes();
        let mut rng = CausalRng::from_seed(17);
        let result =
            bootstrap_run_counts(4, 4, &[0..1, 1..2, 2..3, 3..4], 2, &mut rng, &mut budget, &ctx)
                .unwrap();
        assert_eq!(&*result, &[1.0; 4]);
        assert!(pool.available().bytes() < before);
        drop(result);
        assert_eq!(pool.available().bytes(), before);
        assert!(matches!(
            bootstrap_run_counts(500, 1, &[0..500], 2, &mut rng, &mut budget, &ctx),
            Err(AttributionError::Budget { .. })
        ));
        assert_eq!(pool.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            bayesian_run_weights(4, 2, &[0..4], 0, &mut rng, &mut budget, &ctx),
            Err(AttributionError::Cancelled)
        ));
    }
}
