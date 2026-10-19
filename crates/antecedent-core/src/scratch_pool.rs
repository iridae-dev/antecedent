//! Bounded worker scratch for row, draw, grid, fold, and replicate batches.
//!
//! [`ScratchPool::try_acquire`] is the producer admission. The returned guard is
//! the consumer's hold on one buffer. Dropping the guard returns that buffer
//! and is the only release. A consumer that keeps the guard is stalled:
//! further acquires return [`ScratchFull`] immediately.
//!
//! The pool accounts for `capacity_buffers * bytes_per_buffer` and does not
//! allocate the numeric payload. [`ScratchPool::peak_live`] is the high-water
//! mark of live guards. It is never greater than the capacity. Admission does
//! not wait, sleep, or spawn workers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use thiserror::Error;

/// Batch whose scratch buffers share one bounded pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScratchKind {
    /// Row batch scratch.
    Row,
    /// Draw batch scratch.
    Draw,
    /// Grid batch scratch.
    Grid,
    /// Fold batch scratch.
    Fold,
    /// Replicate batch scratch.
    Replicate,
}

impl ScratchKind {
    /// Row, draw, grid, fold, and replicate, in that order.
    pub const ALL: [Self; 5] = [Self::Row, Self::Draw, Self::Grid, Self::Fold, Self::Replicate];

    /// Stable `snake_case` name used in admission errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Draw => "draw",
            Self::Grid => "grid",
            Self::Fold => "fold",
            Self::Replicate => "replicate",
        }
    }
}

impl fmt::Display for ScratchKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Construction refusal for a scratch pool.
///
/// The pool is not built. No buffer is admitted and no payload is allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ScratchPoolError {
    /// A capacity of zero admits no buffer.
    ///
    /// Zero is a declared empty pool, so construction stops.
    #[error(
        "scratch pool `{kind}` refuses a capacity of zero buffers; next attempt: declare a \
         positive buffer capacity"
    )]
    ZeroCapacity {
        /// Batch that was refused.
        kind: ScratchKind,
    },
    /// `capacity_buffers * bytes_per_buffer` does not fit in `u64`.
    ///
    /// The product is not reported as zero and is not allocated.
    #[error(
        "scratch pool `{kind}` refuses {capacity_buffers} buffers of {bytes_per_buffer} bytes \
         because the product overflows u64; next attempt: fewer buffers or fewer bytes per buffer"
    )]
    ByteProductOverflow {
        /// Batch that was refused.
        kind: ScratchKind,
        /// Requested live-buffer ceiling.
        capacity_buffers: u64,
        /// Requested account of one buffer, in bytes.
        bytes_per_buffer: u64,
    },
}

/// The pool already holds `capacity` live buffers.
///
/// This is the backpressure signal for a stalled consumer. The acquire did not
/// wait for a guard to drop.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[error(
    "scratch pool `{kind}` is full with {live} live buffers at capacity {capacity}; next \
     attempt: drop a held guard before acquiring another"
)]
pub struct ScratchFull {
    kind: ScratchKind,
    capacity: u64,
    live: u64,
}

impl ScratchFull {
    /// Batch whose pool is full.
    #[must_use]
    pub const fn kind(self) -> ScratchKind {
        self.kind
    }

    /// Live-buffer ceiling that was reached.
    #[must_use]
    pub const fn capacity(self) -> u64 {
        self.capacity
    }

    /// Guards held when admission was refused.
    #[must_use]
    pub const fn live(self) -> u64 {
        self.live
    }
}

#[derive(Debug)]
struct PoolState {
    live: u64,
    peak: u64,
}

#[derive(Debug)]
struct Shared {
    kind: ScratchKind,
    capacity: u64,
    bytes_per_buffer: u64,
    accounted_bytes: u64,
    state: Mutex<PoolState>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Bounded account of live scratch buffers for one batch kind.
///
/// Clones share the live count, the high-water mark, and the byte account.
/// The pool stores that account. It does not store the numeric payload.
#[derive(Clone, Debug)]
pub struct ScratchPool {
    shared: Arc<Shared>,
}

impl ScratchPool {
    /// Bound a pool for `kind`.
    ///
    /// `capacity_buffers` is the most live guards the pool admits.
    /// `bytes_per_buffer` is the accounted size of one buffer. The product is
    /// taken with [`u64::checked_mul`]. A product that fits, including a
    /// declared zero byte width, is the pool's byte account.
    ///
    /// # Errors
    ///
    /// [`ScratchPoolError::ZeroCapacity`] when `capacity_buffers` is zero.
    /// [`ScratchPoolError::ByteProductOverflow`] when the byte product does not
    /// fit in `u64`. That refusal happens before any buffer is admitted.
    pub fn bounded(
        kind: ScratchKind,
        capacity_buffers: u64,
        bytes_per_buffer: u64,
    ) -> Result<Self, ScratchPoolError> {
        if capacity_buffers == 0 {
            return Err(ScratchPoolError::ZeroCapacity { kind });
        }
        let Some(accounted_bytes) = capacity_buffers.checked_mul(bytes_per_buffer) else {
            return Err(ScratchPoolError::ByteProductOverflow {
                kind,
                capacity_buffers,
                bytes_per_buffer,
            });
        };
        Ok(Self {
            shared: Arc::new(Shared {
                kind,
                capacity: capacity_buffers,
                bytes_per_buffer,
                accounted_bytes,
                state: Mutex::new(PoolState { live: 0, peak: 0 }),
            }),
        })
    }

    /// Batch this pool accounts for.
    #[must_use]
    pub fn kind(&self) -> ScratchKind {
        self.shared.kind
    }

    /// Most live guards this pool admits.
    #[must_use]
    pub fn capacity_buffers(&self) -> u64 {
        self.shared.capacity
    }

    /// Accounted size of one buffer, in bytes.
    #[must_use]
    pub fn bytes_per_buffer(&self) -> u64 {
        self.shared.bytes_per_buffer
    }

    /// `capacity_buffers * bytes_per_buffer`, fixed at construction.
    #[must_use]
    pub fn accounted_bytes(&self) -> u64 {
        self.shared.accounted_bytes
    }

    /// Guards currently held.
    #[must_use]
    pub fn live(&self) -> u64 {
        self.shared.lock().live
    }

    /// High-water mark of live buffers.
    ///
    /// The mark advances when a guard is admitted and stays when a guard is
    /// dropped. It is never greater than [`Self::capacity_buffers`].
    #[must_use]
    pub fn peak_live(&self) -> u64 {
        self.shared.lock().peak
    }

    /// Admit one live buffer, or refuse because the pool is full.
    ///
    /// A full pool is backpressure. This returns immediately. It does not wait
    /// for a consumer to drop a guard, and it does not sleep.
    ///
    /// # Errors
    ///
    /// [`ScratchFull`] when `capacity_buffers` guards are already live. Drop one
    /// held guard to admit another. The high-water mark stays at capacity.
    pub fn try_acquire(&self) -> Result<ScratchGuard, ScratchFull> {
        let mut state = self.shared.lock();
        if state.live >= self.shared.capacity {
            return Err(ScratchFull {
                kind: self.shared.kind,
                capacity: self.shared.capacity,
                live: state.live,
            });
        }
        state.live += 1;
        state.peak = state.peak.max(state.live);
        drop(state);
        Ok(ScratchGuard { shared: Arc::clone(&self.shared) })
    }
}

/// One admitted scratch buffer.
///
/// Dropping the guard returns the buffer to the pool. The guard does not own
/// a numeric payload; [`Self::bytes`] is the accounted width.
#[derive(Debug)]
#[must_use = "dropping the guard releases the scratch buffer"]
pub struct ScratchGuard {
    shared: Arc<Shared>,
}

impl ScratchGuard {
    /// Batch this buffer was acquired for.
    #[must_use]
    pub fn kind(&self) -> ScratchKind {
        self.shared.kind
    }

    /// Accounted size of this buffer, in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.shared.bytes_per_buffer
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.live = state.live.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    fn acquire_all(pool: &ScratchPool) -> Vec<ScratchGuard> {
        let capacity = pool.capacity_buffers();
        let mut guards = Vec::new();
        for expected_live in 1..=capacity {
            let guard = match pool.try_acquire() {
                Ok(guard) => guard,
                Err(full) => {
                    panic!("{} full at {} while filling {expected_live}", full.kind(), full.live())
                }
            };
            assert_eq!(guard.kind(), pool.kind());
            assert_eq!(guard.bytes(), pool.bytes_per_buffer());
            assert_eq!(pool.live(), expected_live);
            assert_eq!(pool.peak_live(), expected_live);
            assert!(pool.peak_live() <= capacity);
            guards.push(guard);
        }
        assert_eq!(pool.peak_live(), capacity);
        guards
    }

    struct WaitOnDrop<'a> {
        barrier: &'a Barrier,
    }

    impl Drop for WaitOnDrop<'_> {
        fn drop(&mut self) {
            let _leader = self.barrier.wait();
        }
    }

    #[test]
    fn each_kind_peak_equals_capacity_when_consumer_holds_every_buffer() {
        for kind in ScratchKind::ALL {
            let pool = ScratchPool::bounded(kind, 5, 24).expect("pool");
            assert_eq!(pool.accounted_bytes(), 120);
            let guards = acquire_all(&pool);
            assert_eq!(guards.len(), 5);
            match pool.try_acquire() {
                Err(full) => {
                    assert_eq!(full.kind(), kind);
                    assert_eq!(full.capacity(), 5);
                    assert_eq!(full.live(), 5);
                    let text = full.to_string();
                    assert!(text.contains(kind.as_str()), "{text}");
                    assert!(text.contains("capacity"), "{text}");
                }
                Ok(_) => panic!("{kind} admitted a buffer past capacity"),
            }
            assert_eq!(pool.live(), 5);
            assert_eq!(pool.peak_live(), 5);
            drop(guards);
            assert_eq!(pool.live(), 0);
            assert_eq!(pool.peak_live(), 5);
        }
    }

    #[test]
    fn each_kind_one_release_admits_one_more() {
        for kind in ScratchKind::ALL {
            let pool = ScratchPool::bounded(kind, 4, 16).expect("pool");
            let worker = pool.clone();
            let mut guards = acquire_all(&worker);
            assert!(pool.try_acquire().is_err());
            drop(guards.remove(0));
            assert_eq!(pool.live(), 3);
            assert_eq!(pool.peak_live(), 4);
            let admitted = match pool.try_acquire() {
                Ok(guard) => guard,
                Err(full) => panic!("{kind} stayed full at {} after one release", full.live()),
            };
            assert_eq!(admitted.kind(), kind);
            assert_eq!(admitted.bytes(), 16);
            assert_eq!(pool.live(), 4);
            assert_eq!(pool.peak_live(), 4);
            assert!(pool.try_acquire().is_err());
            assert_eq!(pool.peak_live(), 4);
            drop(admitted);
            drop(guards);
            assert_eq!(pool.live(), 0);
            assert_eq!(pool.peak_live(), 4);
        }
    }

    #[test]
    fn each_kind_byte_product_overflow_is_refused() {
        let overflow_capacity = u64::MAX.checked_div(3).and_then(|part| part.checked_add(1));
        let overflow_capacity = overflow_capacity.expect("overflow shape");
        let products = [(u64::MAX, 2_u64), (2_u64, u64::MAX), (overflow_capacity, 3_u64)];
        for kind in ScratchKind::ALL {
            for (capacity_buffers, bytes_per_buffer) in products {
                match ScratchPool::bounded(kind, capacity_buffers, bytes_per_buffer) {
                    Err(ScratchPoolError::ByteProductOverflow {
                        kind: refused,
                        capacity_buffers: refused_capacity,
                        bytes_per_buffer: refused_bytes,
                    }) => {
                        assert_eq!(refused, kind);
                        assert_eq!(refused_capacity, capacity_buffers);
                        assert_eq!(refused_bytes, bytes_per_buffer);
                        let text = ScratchPoolError::ByteProductOverflow {
                            kind,
                            capacity_buffers,
                            bytes_per_buffer,
                        }
                        .to_string();
                        assert!(text.contains("overflow"), "{text}");
                        assert!(text.contains(kind.as_str()), "{text}");
                    }
                    Err(other) => panic!("{kind} refused with {other}"),
                    Ok(_) => panic!("{kind} admitted an overflowing byte product"),
                }
            }
            let bytes = u64::MAX.checked_div(5).expect("divisor");
            let pool = ScratchPool::bounded(kind, 5, bytes).expect("product fits");
            let product = bytes.checked_mul(5).expect("product");
            assert_eq!(pool.accounted_bytes(), product);
            assert_eq!(pool.bytes_per_buffer(), bytes);
            let empty = ScratchPool::bounded(kind, 1, 0).expect("zero width is an account");
            assert_eq!(empty.accounted_bytes(), 0);
            assert_eq!(empty.try_acquire().expect("empty buffer").bytes(), 0);
        }
    }

    #[test]
    fn each_kind_zero_capacity_is_refused_at_construction() {
        for kind in ScratchKind::ALL {
            for bytes in [0, 8, u64::MAX] {
                match ScratchPool::bounded(kind, 0, bytes) {
                    Err(ScratchPoolError::ZeroCapacity { kind: refused }) => {
                        assert_eq!(refused, kind);
                        let text = ScratchPoolError::ZeroCapacity { kind }.to_string();
                        assert!(text.contains("zero"), "{text}");
                        assert!(text.contains(kind.as_str()), "{text}");
                    }
                    Err(other) => panic!("{kind} refused with {other}"),
                    Ok(_) => panic!("{kind} admitted a capacity of zero"),
                }
            }
        }
    }

    #[test]
    fn each_kind_stalled_consumer_keeps_peak_at_capacity_under_concurrency() {
        const WORKERS: usize = 8;
        const CAPACITY: u64 = 4;
        for kind in ScratchKind::ALL {
            let pool = ScratchPool::bounded(kind, CAPACITY, 32).expect("pool");
            let barrier = Arc::new(Barrier::new(WORKERS + 1));
            thread::scope(|scope| {
                for _ in 0..WORKERS {
                    let pool = pool.clone();
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        let held = pool.try_acquire();
                        let _leader = barrier.wait();
                        let release = WaitOnDrop { barrier: &barrier };
                        assert!(pool.try_acquire().is_err(), "{kind}");
                        assert!(pool.peak_live() <= CAPACITY, "{kind}");
                        // `held` stays live until the second wait, so the stall is visible.
                        drop(release);
                        drop(held);
                    });
                }
                let _leader = barrier.wait();
                let release = WaitOnDrop { barrier: &barrier };
                assert_eq!(pool.live(), CAPACITY, "{kind}");
                assert_eq!(pool.peak_live(), CAPACITY, "{kind}");
                match pool.try_acquire() {
                    Err(full) => {
                        assert_eq!(full.kind(), kind);
                        assert_eq!(full.capacity(), CAPACITY);
                        assert_eq!(full.live(), CAPACITY);
                    }
                    Ok(_) => panic!("{kind} admitted a buffer while consumers were stalled"),
                }
                drop(release);
            });
            assert_eq!(pool.live(), 0, "{kind}");
            assert_eq!(pool.peak_live(), CAPACITY, "{kind}");
        }
    }
}
