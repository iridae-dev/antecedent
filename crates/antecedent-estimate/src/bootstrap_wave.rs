//! Bounded IID bootstrap index waves.
//!
//! A wave retains `n` row indexes for each of `width` replicates. `width` is the
//! worker count or another caller-chosen bound, so the buffer is not an `n` by
//! replicates matrix. Replicate `i` is drawn from [`StreamDomain::Resample`] at
//! `stream_base ^ i`: the same seed and replicate id always produce the same rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_kernels::unbiased_index;

use crate::error::EstimationError;

/// Slots in an addressable `n × workers` index buffer.
///
/// The product is the retained index memory. It does not grow with the replicate count.
///
/// # Errors
///
/// `n * workers` indexes of `u64` do not fit in an addressable allocation.
pub(crate) fn retained_index_slots(n: usize, workers: usize) -> Result<usize, EstimationError> {
    n.checked_mul(workers)
        .filter(|&slots| {
            slots.checked_mul(size_of::<u64>()).is_some_and(|bytes| isize::try_from(bytes).is_ok())
        })
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap index allocation exceeds addressable capacity")
        })
}

/// Retained IID bootstrap indexes for at most `width` replicates.
///
/// The buffer length is `n * width`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct BootstrapIndexWave {
    indexes: Vec<u64>,
    n: usize,
    width: usize,
    filled: usize,
}

#[cfg(test)]
impl BootstrapIndexWave {
    /// Allocate a buffer for `width` replicates of length `n`.
    ///
    /// # Errors
    ///
    /// `n` or `width` is zero, or [`retained_index_slots`] refuses the product.
    pub(crate) fn allocate(n: usize, width: usize) -> Result<Self, EstimationError> {
        if n == 0 || width == 0 {
            return Err(EstimationError::data_msg(
                "bootstrap index wave needs a positive row count and width",
            ));
        }
        let slots = retained_index_slots(n, width)?;
        let mut indexes = Vec::new();
        indexes.try_reserve_exact(slots).map_err(|e| {
            EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
        })?;
        indexes.resize(slots, 0);
        Ok(Self { indexes, n, width, filled: 0 })
    }

    /// Number of row indexes the wave retains (`n * width`).
    #[must_use]
    pub(crate) fn index_len(&self) -> usize {
        self.indexes.len()
    }

    /// Draw replicates `origin .. origin + count` into this buffer.
    ///
    /// `count` must be at most the allocated width. Replicate `origin + local` uses
    /// [`StreamDomain::Resample`] keyed by `stream_base ^ replicate`.
    ///
    /// # Errors
    ///
    /// `count` exceeds the width, a replicate id does not fit in `u64`, or a draw does
    /// not fit in a row index.
    pub(crate) fn fill(
        &mut self,
        ctx: &ExecutionContext,
        stream_base: u64,
        origin: u64,
        count: usize,
    ) -> Result<(), EstimationError> {
        if count > self.width {
            return Err(EstimationError::data_msg(
                "bootstrap index wave count exceeds the allocated width",
            ));
        }
        if self.n == 0 {
            return Err(EstimationError::data_msg("bootstrap replicate needs n > 0"));
        }
        let n = self.n;
        self.filled = 0;
        for (local, row) in self.indexes.chunks_mut(n).take(count).enumerate() {
            let local_id = u64::try_from(local).map_err(|_| replicate_id_error())?;
            let replicate = origin.checked_add(local_id).ok_or_else(replicate_id_error)?;
            let mut at = 0;
            with_replicate_draws(n, replicate, ctx, stream_base, |index| {
                row[at] = u64::try_from(index).map_err(|_| row_index_error())?;
                at += 1;
                Ok(())
            })?;
        }
        self.filled = count;
        Ok(())
    }

    /// Row indexes of local slot `local` from the last successful [`Self::fill`].
    #[must_use]
    pub(crate) fn replicate(&self, local: usize) -> Option<&[u64]> {
        if local >= self.filled {
            return None;
        }
        let start = local.checked_mul(self.n)?;
        let end = start.checked_add(self.n)?;
        self.indexes.get(start..end)
    }
}

/// Write one replicate's row indexes into `out`, replacing its previous contents.
///
/// The draw is [`StreamDomain::Resample`] at `stream_base ^ replicate`. `out` ends at length `n`.
///
/// # Errors
///
/// `n` is zero, the buffer cannot be reserved, or a replicate draw fails.
pub(crate) fn fill_worker_replicate(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    stream_base: u64,
    out: &mut Vec<usize>,
) -> Result<(), EstimationError> {
    out.clear();
    out.try_reserve_exact(n).map_err(|e| {
        EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
    })?;
    with_replicate_draws(n, replicate, ctx, stream_base, |index| {
        out.push(index);
        Ok(())
    })
}

#[inline]
fn with_replicate_draws(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    stream_base: u64,
    mut store: impl FnMut(usize) -> Result<(), EstimationError>,
) -> Result<(), EstimationError> {
    if n == 0 {
        return Err(EstimationError::data_msg("bootstrap replicate needs n > 0"));
    }
    let mut rng = ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate);
    for _ in 0..n {
        store(unbiased_index(&mut rng, n))?;
    }
    Ok(())
}

#[cfg(test)]
fn replicate_id_error() -> EstimationError {
    EstimationError::data_msg("bootstrap replicate id exceeds u64")
}

#[cfg(test)]
fn row_index_error() -> EstimationError {
    EstimationError::data_msg(
        "row position does not fit in u64; rows are addressed by GlobalId and \
         ChunkIndex only spans one bounded chunk",
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use antecedent_core::ExecutionContext;
    use antecedent_data::ResamplingPlan;

    use super::*;
    use crate::util::{bootstrap_se, finalize_bootstrap_se};

    fn reference_ids(n: usize, replicates: usize, seed: u64, stream_base: u64) -> Vec<u64> {
        let ctx = ExecutionContext::for_tests(seed);
        let mut row = Vec::new();
        let mut full = Vec::with_capacity(n * replicates);
        for replicate in 0..replicates {
            antecedent_data::resample::fill_resample_index_replicate(
                ResamplingPlan::IidBootstrap,
                n,
                None,
                &ctx,
                stream_base,
                u64::try_from(replicate).unwrap(),
                &mut row,
            )
            .unwrap();
            full.extend(row.iter().map(|id| id.get()));
        }
        full
    }

    fn as_rows(flat: &[u64], n: usize) -> Vec<Vec<usize>> {
        flat.chunks(n)
            .map(|row| row.iter().map(|&id| usize::try_from(id).unwrap()).collect())
            .collect()
    }

    fn row_sum(idx: &[usize]) -> f64 {
        let sum = idx.iter().fold(0u32, |acc, &row| acc + u32::try_from(row).unwrap());
        f64::from(sum)
    }

    fn sums_of(rows: &[Vec<usize>]) -> Vec<f64> {
        rows.iter().map(|idx| row_sum(idx)).collect()
    }

    fn assert_bootstrap_matches(
        n: usize,
        replicates: u32,
        stream_base: u64,
        seed: u64,
        threads: u32,
        expected: &[Vec<usize>],
    ) {
        let ctx = if threads == 1 {
            ExecutionContext::for_tests(seed)
        } else {
            ExecutionContext::production(seed, threads)
        };
        let recorded = Mutex::new(Vec::new());
        let result = bootstrap_se(replicates, &ctx, stream_base, n, |idx| {
            recorded.lock().unwrap().push(idx.to_vec());
            Ok(Some(row_sum(idx)))
        })
        .unwrap();
        let mut got = recorded.into_inner().unwrap();
        let mut expect = expected.to_vec();
        if threads > 1 {
            got.sort();
            expect.sort();
        }
        assert_eq!(got, expect);
        let reference = finalize_bootstrap_se(&sums_of(expected), replicates);
        let got_se = result.se.expect("bootstrap se");
        let expect_se = reference.se.expect("reference se");
        assert!((got_se - expect_se).abs() < 1e-12);
        assert_eq!(result.replicates_ok, replicates);
        assert!(!result.cancelled);
    }

    #[test]
    fn wave_matches_full_index_reference() {
        let n = 5usize;
        let replicates = 11u32;
        let n_rep = replicates as usize;
        let stream_base = 0x51u64;
        let seed = 99u64;
        let reference = reference_ids(n, n_rep, seed, stream_base);
        let expected = as_rows(&reference, n);
        let ctx = ExecutionContext::for_tests(seed);
        let width = 3usize;
        let mut wave = BootstrapIndexWave::allocate(n, width).unwrap();
        let mut origin = 0usize;
        while origin < n_rep {
            let count = width.min(n_rep - origin);
            wave.fill(&ctx, stream_base, u64::try_from(origin).unwrap(), count).unwrap();
            for local in 0..count {
                let got = wave.replicate(local).expect("filled slot");
                let start = (origin + local) * n;
                assert_eq!(got, &reference[start..start + n]);
            }
            origin += count;
        }
        let mut alone = BootstrapIndexWave::allocate(n, 1).unwrap();
        alone.fill(&ctx, stream_base, 7, 1).unwrap();
        assert_eq!(alone.replicate(0).expect("replicate 7"), &reference[7 * n..8 * n]);
        assert_bootstrap_matches(n, replicates, stream_base, seed, 1, &expected);
        assert_bootstrap_matches(n, replicates, stream_base, seed, 4, &expected);
    }

    #[test]
    fn index_buffer_len_is_bounded_by_workers() {
        let n = 17usize;
        let workers = 4usize;
        let replicates = 200usize;
        let mut wave = BootstrapIndexWave::allocate(n, workers).unwrap();
        assert_eq!(wave.index_len(), n * workers);
        assert!(wave.index_len() < n * replicates);
        let ctx = ExecutionContext::for_tests(3);
        let mut origin = 0u64;
        let mut seen = 0usize;
        while seen < replicates {
            let count = workers.min(replicates - seen);
            wave.fill(&ctx, 11, origin, count).unwrap();
            assert_eq!(wave.index_len(), n * workers);
            origin += u64::try_from(count).unwrap();
            seen += count;
        }
    }
}
