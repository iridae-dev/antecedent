//! Deterministic random-stream keys for a semantic operation.
//!
//! A legal stream is named by the scientific operation, the fold, the chain,
//! the replicate, and the declared numerical reduction policy. Fixed-seed
//! replay follows from that record: the same [`RngKey`] and the same seed open
//! the same [`CausalRng`] stream, and opening that key again (a retry) repeats
//! it. Worker id and batch size are scheduling facts. They may be carried on a
//! [`StreamRequest`] and they are not mixed into the seed, so a different batch
//! size or a different worker leaves the stream unchanged.
//!
//! [`ReductionPolicy::Strict`] is order-sensitive: partials combine in the
//! declared order. [`ReductionPolicy::Associative`] may reorder partials within
//! its declared tolerance. The policy is part of the key. Opening a stream does
//! not rewrite it, and a different policy is a different stream.
//!
//! A request that keys the stream by worker scheduling is a [`FailedRngAttempt`].
//! The legal key is the semantic one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use crate::execution::{CausalRng, derive_stream};

/// Domain word mixed ahead of every semantic key.
///
/// Keeps a key stream distinct from a raw [`derive_stream`] of the same seed and
/// a small integer (a worker index, in particular).
const KEY_DOMAIN: u64 = 0xA3D5_C6E7_190B_4F21;

/// Strict policy word. Distinct from [`POLICY_ASSOCIATIVE`] even at a zero tolerance.
const POLICY_STRICT: u64 = 1;

/// Associative policy word.
const POLICY_ASSOCIATIVE: u64 = 2;

/// Caller-chosen tag of the scientific operation that owns the stream.
///
/// The tag is the operation's whole contribution to the key. Two operations that
/// share a fold, chain, replicate, and policy still open different streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SemanticOperation(u64);

impl SemanticOperation {
    /// Tag `raw` as an operation.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The tag mixed into the key.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Cross-validation or bootstrap fold inside an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FoldId(u64);

impl FoldId {
    /// Fold index `raw`.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The index mixed into the key.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// MCMC or other sequential chain inside an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainId(u64);

impl ChainId {
    /// Chain index `raw`.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The index mixed into the key.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Replicate (bootstrap draw, posterior draw, Monte Carlo replicate).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ReplicateId(u64);

impl ReplicateId {
    /// Replicate index `raw`.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The index mixed into the key.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Worker that happens to execute a draw. Not part of [`RngKey`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorkerId(u64);

impl WorkerId {
    /// Worker index `raw`.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The scheduling index.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Batch cut that happens to contain a draw. Not part of [`RngKey`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BatchSize(u64);

impl BatchSize {
    /// Batch length `raw`. Zero is a scheduling fact like any other length.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The scheduling length.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Absolute tolerance declared for an associative reduction.
///
/// Stored as an IEEE-754 bit pattern so the key stays [`Eq`]. `-0.0` is stored
/// as `0.0`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeclaredTolerance(u64);

impl DeclaredTolerance {
    /// A finite, non-negative absolute bound.
    ///
    /// # Errors
    ///
    /// [`ToleranceError::NonFinite`] when `bound` is `NaN` or infinite.
    /// [`ToleranceError::Negative`] when `bound` is negative.
    #[must_use = "a non-finite or negative tolerance is refused"]
    pub const fn try_new(bound: f64) -> Result<Self, ToleranceError> {
        if !bound.is_finite() {
            return Err(ToleranceError::NonFinite);
        }
        if bound.is_sign_negative() {
            // `-0.0` is a zero tolerance.
            if bound.to_bits() == 0x8000_0000_0000_0000 {
                return Ok(Self(0));
            }
            return Err(ToleranceError::Negative);
        }
        Ok(Self(bound.to_bits()))
    }

    /// IEEE-754 bits of the bound.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// The declared absolute bound.
    #[must_use]
    pub const fn bound(self) -> f64 {
        f64::from_bits(self.0)
    }
}

/// Why a declared tolerance was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToleranceError {
    /// NaN or an infinity.
    NonFinite,
    /// A negative bound.
    Negative,
}

impl fmt::Display for ToleranceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFinite => "associative tolerance must be finite",
            Self::Negative => "associative tolerance must be non-negative",
        })
    }
}

impl std::error::Error for ToleranceError {}

/// How partial numerical results of a keyed draw may be combined.
///
/// The policy is declared when the key is built. [`RngKey::open`] does not
/// change it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ReductionPolicy {
    /// Order-sensitive. Partials combine in the declared order.
    Strict,
    /// Partials may be reordered within this declared absolute tolerance.
    Associative(DeclaredTolerance),
}

/// Semantic key of one random stream.
///
/// Components are the operation, fold, chain, replicate, and reduction policy.
/// Worker id and batch size are not fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RngKey {
    operation: SemanticOperation,
    fold: FoldId,
    chain: ChainId,
    replicate: ReplicateId,
    policy: ReductionPolicy,
}

impl RngKey {
    /// Build the legal key. The policy stored here is the policy the stream keeps.
    #[must_use]
    pub const fn new(
        operation: SemanticOperation,
        fold: FoldId,
        chain: ChainId,
        replicate: ReplicateId,
        policy: ReductionPolicy,
    ) -> Self {
        Self { operation, fold, chain, replicate, policy }
    }

    /// Operation tag.
    #[must_use]
    pub const fn operation(self) -> SemanticOperation {
        self.operation
    }

    /// Fold index.
    #[must_use]
    pub const fn fold(self) -> FoldId {
        self.fold
    }

    /// Chain index.
    #[must_use]
    pub const fn chain(self) -> ChainId {
        self.chain
    }

    /// Replicate index.
    #[must_use]
    pub const fn replicate(self) -> ReplicateId {
        self.replicate
    }

    /// Declared reduction policy. Equal to the policy passed to [`Self::new`].
    #[must_use]
    pub const fn policy(self) -> ReductionPolicy {
        self.policy
    }

    /// Open the stream for `seed`.
    ///
    /// The same key and the same seed return the same stream, including on a
    /// later retry. The seed is a non-additive mix of `seed` with the key; it
    /// does not read a worker id or a batch size.
    #[must_use]
    pub const fn open(self, seed: u64) -> CausalRng {
        CausalRng::from_seed(derive_stream(seed, mix_key(self)))
    }
}

/// A request to open a stream, either from a semantic key or from scheduling.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StreamRequest {
    /// Legal. `worker` and `batch` are recorded and are not key material.
    Semantic {
        /// Semantic key.
        key: RngKey,
        /// Worker that will run the draw.
        worker: WorkerId,
        /// Batch size the scheduler chose.
        batch: BatchSize,
    },
    /// Illegal. The stream would follow worker scheduling.
    Scheduling {
        /// Worker the refused stream was keyed by.
        worker: WorkerId,
        /// Batch size the refused stream was keyed by.
        batch: BatchSize,
    },
}

impl StreamRequest {
    /// A legal request. `worker` and `batch` do not enter [`RngKey::open`].
    #[must_use]
    pub const fn semantic(key: RngKey, worker: WorkerId, batch: BatchSize) -> Self {
        Self::Semantic { key, worker, batch }
    }

    /// A request keyed by worker scheduling. [`Self::open`] refuses it.
    #[must_use]
    pub const fn scheduling(worker: WorkerId, batch: BatchSize) -> Self {
        Self::Scheduling { worker, batch }
    }

    /// Open a semantic stream, or refuse a scheduling-dependent one.
    ///
    /// # Errors
    ///
    /// [`FailedRngAttempt`] when `self` is [`Self::Scheduling`].
    #[must_use = "a scheduling-dependent stream is a failed attempt"]
    pub const fn open(self, seed: u64) -> Result<CausalRng, FailedRngAttempt> {
        match self {
            // Worker and batch stay on the request. The seed is the key and `seed`.
            Self::Semantic { key, .. } => Ok(key.open(seed)),
            Self::Scheduling { worker, batch } => Err(FailedRngAttempt { worker, batch }),
        }
    }
}

/// Refusal of a stream keyed by worker scheduling.
///
/// The legal key is an [`RngKey`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FailedRngAttempt {
    worker: WorkerId,
    batch: BatchSize,
}

impl FailedRngAttempt {
    /// Worker the refused stream was keyed by.
    #[must_use]
    pub const fn worker(self) -> WorkerId {
        self.worker
    }

    /// Batch size the refused stream was keyed by.
    #[must_use]
    pub const fn batch(self) -> BatchSize {
        self.batch
    }
}

impl fmt::Display for FailedRngAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let worker = self.worker.get();
        let batch = self.batch.get();
        write!(formatter, "a stream that depends on worker {worker} and batch {batch}")?;
        formatter.write_str(" is a failed attempt; the legal key is the semantic one")
    }
}

impl std::error::Error for FailedRngAttempt {}

/// Non-additive mix of the semantic components, in field order.
///
/// Words: domain, operation, fold, chain, replicate, policy discriminant,
/// tolerance bits. Worker id and batch size are absent.
const fn mix_key(key: RngKey) -> u64 {
    let (policy, tolerance) = match key.policy {
        ReductionPolicy::Strict => (POLICY_STRICT, 0),
        ReductionPolicy::Associative(tolerance) => (POLICY_ASSOCIATIVE, tolerance.bits()),
    };
    let mut z = derive_stream(KEY_DOMAIN, key.operation.get());
    z = derive_stream(z, key.fold.get());
    z = derive_stream(z, key.chain.get());
    z = derive_stream(z, key.replicate.get());
    z = derive_stream(z, policy);
    derive_stream(z, tolerance)
}

#[cfg(test)]
mod tests {
    use super::{
        BatchSize, ChainId, DeclaredTolerance, FoldId, ReductionPolicy, ReplicateId, RngKey,
        SemanticOperation, StreamRequest, ToleranceError, WorkerId,
    };
    use crate::execution::CausalRng;

    const SEED: u64 = 42;

    fn tolerance(bound: f64) -> DeclaredTolerance {
        DeclaredTolerance::try_new(bound).unwrap()
    }

    const fn key_parts(
        operation: u64,
        fold: u64,
        chain: u64,
        replicate: u64,
        policy: ReductionPolicy,
    ) -> RngKey {
        RngKey::new(
            SemanticOperation::new(operation),
            FoldId::new(fold),
            ChainId::new(chain),
            ReplicateId::new(replicate),
            policy,
        )
    }

    const fn sample() -> RngKey {
        key_parts(10, 2, 1, 4, ReductionPolicy::Strict)
    }

    fn prefix(mut rng: CausalRng) -> [u64; 8] {
        core::array::from_fn(|_| rng.next_u64())
    }

    #[test]
    fn fixed_seed_replay_matches() {
        let first = sample();
        let second = sample();
        assert_eq!(first, second);
        assert_eq!(prefix(first.open(SEED)), prefix(second.open(SEED)));
        assert_ne!(prefix(first.open(SEED)), prefix(first.open(SEED + 1)));
    }

    #[test]
    fn retry_matches() {
        let key = sample();
        assert_eq!(key.policy(), ReductionPolicy::Strict);
        let first = key.open(SEED);
        assert_eq!(key.policy(), ReductionPolicy::Strict);
        assert_eq!(prefix(first), prefix(key.open(SEED)));
    }

    #[test]
    fn same_key_across_two_worker_ids_matches() {
        let key = sample();
        let batch = BatchSize::new(32);
        let left = StreamRequest::semantic(key, WorkerId::new(0), batch);
        let right = StreamRequest::semantic(key, WorkerId::new(7), batch);
        assert_ne!(left, right);
        let left_prefix = prefix(left.open(SEED).unwrap());
        let right_prefix = prefix(right.open(SEED).unwrap());
        assert_eq!(left_prefix, right_prefix);
        assert_eq!(prefix(key.open(SEED)), right_prefix);
    }

    #[test]
    fn batch_size_change_matches() {
        let key = sample();
        let worker = WorkerId::new(3);
        let small = StreamRequest::semantic(key, worker, BatchSize::new(1));
        let large = StreamRequest::semantic(key, worker, BatchSize::new(128));
        assert_ne!(small, large);
        let small_prefix = prefix(small.open(SEED).unwrap());
        let large_prefix = prefix(large.open(SEED).unwrap());
        assert_eq!(small_prefix, large_prefix);
        let zero = StreamRequest::semantic(key, worker, BatchSize::new(0)).open(SEED).unwrap();
        assert_eq!(large_prefix, prefix(zero));
    }

    #[test]
    fn different_replicate_ids_differ() {
        let left = sample();
        let right = key_parts(10, 2, 1, 5, ReductionPolicy::Strict);
        assert_ne!(left, right);
        assert_ne!(prefix(left.open(SEED)), prefix(right.open(SEED)));
    }

    #[test]
    fn fold_change_differs() {
        let left = sample();
        let right = key_parts(10, 9, 1, 4, ReductionPolicy::Strict);
        assert_ne!(left, right);
        assert_ne!(prefix(left.open(SEED)), prefix(right.open(SEED)));
    }

    #[test]
    fn chain_change_differs() {
        let left = sample();
        let right = key_parts(10, 2, 8, 4, ReductionPolicy::Strict);
        assert_ne!(left, right);
        assert_ne!(prefix(left.open(SEED)), prefix(right.open(SEED)));
    }

    #[test]
    fn operation_change_differs() {
        let left = sample();
        let right = key_parts(11, 2, 1, 4, ReductionPolicy::Strict);
        assert_ne!(left, right);
        assert_ne!(prefix(left.open(SEED)), prefix(right.open(SEED)));
    }

    #[test]
    fn reduction_policy_change_differs() {
        let strict = sample();
        let zero = key_parts(10, 2, 1, 4, ReductionPolicy::Associative(tolerance(0.0)));
        let loose = key_parts(10, 2, 1, 4, ReductionPolicy::Associative(tolerance(1e-6)));
        let tight = key_parts(10, 2, 1, 4, ReductionPolicy::Associative(tolerance(1e-8)));
        let opened = strict.open(SEED);
        assert_eq!(strict.policy(), ReductionPolicy::Strict);
        assert_eq!(prefix(opened), prefix(strict.open(SEED)));
        assert_ne!(strict, zero);
        assert_ne!(prefix(strict.open(SEED)), prefix(zero.open(SEED)));
        assert_ne!(prefix(loose.open(SEED)), prefix(tight.open(SEED)));
        assert_ne!(loose.policy(), tight.policy());
    }

    #[test]
    fn small_key_grid_derives_distinct_keys_and_streams() {
        let mut mixed = std::collections::HashSet::new();
        let mut streams = std::collections::HashSet::new();
        let mut count = 0_usize;
        for operation in 0..6 {
            for fold in 0..6 {
                for chain in 0..6 {
                    for replicate in 0..6 {
                        let key =
                            key_parts(operation, fold, chain, replicate, ReductionPolicy::Strict);
                        mixed.insert(super::mix_key(key));
                        streams.insert(prefix(key.open(SEED)));
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 1296);
        assert_eq!(mixed.len(), count, "two grid keys mixed to the same word");
        assert_eq!(streams.len(), count, "two grid keys opened the same stream");
    }

    #[test]
    fn scheduling_dependent_stream_is_a_failed_attempt() {
        let worker = WorkerId::new(4);
        let batch = BatchSize::new(64);
        let err = StreamRequest::scheduling(worker, batch).open(SEED).unwrap_err();
        assert_eq!(err.worker(), worker);
        assert_eq!(err.batch(), batch);
        assert_eq!(
            err.to_string(),
            "a stream that depends on worker 4 and batch 64 \
             is a failed attempt; the legal key is the semantic one"
        );
        let key = sample();
        let opened = StreamRequest::semantic(key, worker, batch).open(SEED).unwrap();
        assert_eq!(prefix(opened), prefix(key.open(SEED)));
    }

    #[test]
    fn associative_tolerance_rejects_non_finite_and_negative() {
        assert_eq!(DeclaredTolerance::try_new(f64::NAN), Err(ToleranceError::NonFinite));
        assert_eq!(DeclaredTolerance::try_new(f64::INFINITY), Err(ToleranceError::NonFinite));
        assert_eq!(DeclaredTolerance::try_new(f64::NEG_INFINITY), Err(ToleranceError::NonFinite));
        assert_eq!(DeclaredTolerance::try_new(-1.0), Err(ToleranceError::Negative));
        assert_eq!(DeclaredTolerance::try_new(-0.0).unwrap().bits(), 0);
        assert_eq!(DeclaredTolerance::try_new(0.0).unwrap().bits(), 0);
        let bound = tolerance(1e-8);
        assert_eq!(bound.bound().to_bits(), bound.bits());
    }
}
