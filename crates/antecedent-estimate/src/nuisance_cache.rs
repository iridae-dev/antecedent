//! Certified nuisance fits and blocked out-of-fold predictions.
//!
//! AIPW, DML, and policy estimation share one cache. A fit is reused only when
//! the cache key matches in every field. A changed field is a miss. The cache
//! does not return a fit stored under another key.
//!
//! A key built from a prepared fit is the hash of that fit's outcome, treatment,
//! design, options, and weights, plus the seed. Callers pass those values.
//! They do not choose the digests. [`map_indexed`] hands each worker that same
//! cache. The thread-local scope of [`with_certified_cache`] is not the lookup
//! those workers use.
//!
//! Each blocked prediction belongs to the evaluation fold it was held out for.
//! A prediction for a training row of that fold is refused. A held-out
//! evaluation row that sits in a training fold is refused, and so is a request
//! to restore a fold prediction onto a held-out row. Restoring a cached
//! prediction writes each value back onto the row id it was stored with.
//!
//! This module does not fit a nuisance model. Callers store predictions they
//! have already certified. Row ids are `u64`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use antecedent_core::{
    CanonicalWriter, DesignDependencyId, ExecutionContext, NumericalPolicy, ProviderId,
    SemanticDigest, SourceDependencyId,
};
use thiserror::Error;

/// Why a certified nuisance was not stored or restored.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum NuisanceCacheError {
    /// No fit is stored under this key.
    #[error("nuisance cache miss; next attempt: reuse only when every key field matches")]
    Miss,
    /// The row was in the training set of the evaluation fold.
    #[error(
        "out-of-fold prediction for training row {row_id} of fold {fold} is refused; \
         next attempt: predict a row held out of that fold"
    )]
    TrainingLeakage {
        /// Evaluation fold whose training set contains the row.
        fold: u32,
        /// Training row that was asked for or stored.
        row_id: u64,
    },
    /// A held-out evaluation row was used as a training or fold row.
    #[error(
        "held-out row {row_id} leaks into a nuisance fold; \
         next attempt: keep that row out of training and out of the fold blocks"
    )]
    HeldOutLeakage {
        /// Held-out row id.
        row_id: u64,
    },
    /// The same key and role already store a different prediction.
    #[error(
        "this key and role already store a different certified prediction; \
         next attempt: reuse the stored block or change a key field"
    )]
    ConflictingFit,
    /// The fold assignment listed no rows.
    #[error("fold assignment has no rows; next attempt: assign at least one row")]
    EmptyFolds,
    /// Cross-fitting needs two folds so an evaluation fold has a training complement.
    #[error("fold count {fold_count} is below 2; next attempt: declare at least two folds")]
    TooFewFolds {
        /// Declared fold count.
        fold_count: u32,
    },
    /// The same row id was listed twice.
    #[error("row {row_id} is listed twice; next attempt: keep one assignment per row")]
    DuplicateRow {
        /// Repeated row id.
        row_id: u64,
    },
    /// A fold index is outside `0..fold_count`.
    #[error("fold {fold} is outside 0..{fold_count}; next attempt: use a declared fold")]
    FoldOutOfRange {
        /// Requested fold index.
        fold: u32,
        /// Declared fold count.
        fold_count: u32,
    },
    /// A name was empty after trimming.
    #[error("{field} is blank; next attempt: name it")]
    Blank {
        /// Which name was blank.
        field: &'static str,
    },
    /// A stored prediction was not finite.
    #[error("prediction for row {row_id} is not finite; next attempt: store a finite value")]
    NonFinitePrediction {
        /// Row whose prediction was refused.
        row_id: u64,
    },
    /// A numerical dependency was not finite.
    #[error("numerical dependency {name} is not finite; next attempt: store a finite parameter")]
    NonFiniteDependency {
        /// Dependency name.
        name: String,
    },
    /// Outcome, treatment, design, options, or weights contained a non-finite value.
    #[error("{field} is not finite; next attempt: pass finite values")]
    NonFiniteInput {
        /// Which input was not finite.
        field: &'static str,
    },
    /// The row is not a member of the evaluation fold.
    #[error("row {row_id} is not in fold {fold}; next attempt: restore a row of that fold")]
    RowNotInFold {
        /// Evaluation fold.
        fold: u32,
        /// Requested row id.
        row_id: u64,
    },
    /// No block was stored for the evaluation fold.
    #[error("fold {fold} has no stored block; next attempt: store that fold's predictions")]
    UnknownFold {
        /// Evaluation fold.
        fold: u32,
    },
    /// The key is stored, and this role is not.
    #[error("this role is not stored under the key; next attempt: store that nuisance role")]
    UnknownRole,
    /// The same numerical dependency name was given twice.
    #[error(
        "numerical dependency {name} is listed twice; next attempt: keep one value for that name"
    )]
    DuplicateNumericalDependency {
        /// Repeated dependency name.
        name: String,
    },
    /// A fold block contained no predictions.
    #[error("fold {fold} has no predictions; next attempt: store its held-out rows")]
    EmptyBlock {
        /// Evaluation fold.
        fold: u32,
    },
    /// The store listed no fold blocks.
    #[error("no out-of-fold block was given; next attempt: pass the held-out fold blocks")]
    EmptyBlocks,
    /// Two blocks named the same evaluation fold.
    #[error("fold {fold} was given twice; next attempt: keep one block per evaluation fold")]
    DuplicateFold {
        /// Repeated evaluation fold.
        fold: u32,
    },
    /// The evaluation fold has no rows outside it, so the block is not out of fold.
    #[error(
        "fold {fold} has no training rows outside it; \
         next attempt: hold that fold out of a non-empty training set"
    )]
    EmptyTraining {
        /// Evaluation fold.
        fold: u32,
    },
    /// The row is in the evaluation fold and has no stored prediction.
    #[error("row {row_id} has no stored prediction; next attempt: store that row in the block")]
    MissingPrediction {
        /// Evaluation-fold row with no stored value.
        row_id: u64,
    },
    /// A batch named no prepared request.
    #[error("no prepared request was given; next attempt: pass at least one request")]
    EmptyRequests,
    /// A request named no nuisance role.
    #[error(
        "a prepared request names no nuisance role; \
         next attempt: name the treatment or outcome role"
    )]
    EmptyRoles,
    /// A family executor was given another family's request.
    #[error(
        "this request is {request} and this executor is {expected}; \
         next attempt: execute the request with its family"
    )]
    WrongFamily {
        /// Family on the request.
        request: &'static str,
        /// Family the executor serves.
        expected: &'static str,
    },
    /// Aligned prediction inputs had different lengths.
    #[error(
        "prediction rows, folds, and values differ in length; \
         next attempt: pass one fold and one value per row"
    )]
    UnalignedPredictions,
}

/// One field of a [`NuisanceCacheKey`].
///
/// [`NuisanceCacheKey::mismatch`] reports the first field that differs, in this order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum KeyField {
    /// Scientific target.
    Target,
    /// Numeric design identity.
    Design,
    /// Fold assignment and held-out rows.
    Folds,
    /// Treatment the nuisance conditions on.
    Treatment,
    /// Provider identity and version.
    Provider,
    /// Numerical policy and named parameters.
    Numerical,
    /// Outcome column the nuisance was fit to.
    Outcome,
    /// Sampling weights. Absence is distinct from any present vector.
    Weights,
    /// Seed of the fit.
    Seed,
}

/// Scientific target the nuisance was certified for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TargetDependency {
    digest: SemanticDigest,
}

impl TargetDependency {
    /// Target identity. The digest is not interpreted here.
    #[must_use]
    pub const fn new(digest: SemanticDigest) -> Self {
        Self { digest }
    }

    /// Target digest.
    #[must_use]
    pub const fn digest(self) -> SemanticDigest {
        self.digest
    }
}

/// Design identity the nuisance was certified for.
///
/// The id is not a design matrix. This module does not read columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DesignDependency {
    id: DesignDependencyId,
}

impl DesignDependency {
    /// Design identity the caller has already bound to the numeric layout.
    #[must_use]
    pub const fn new(id: DesignDependencyId) -> Self {
        Self { id }
    }

    /// Design id.
    #[must_use]
    pub const fn id(self) -> DesignDependencyId {
        self.id
    }
}

/// Treatment the nuisance conditions on.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TreatmentDependency {
    name: String,
    coding: SemanticDigest,
}

impl TreatmentDependency {
    /// Treatment name and coding identity.
    ///
    /// The stored name is trimmed. A blank name is refused.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Blank`] when `name` is empty after trimming.
    #[must_use = "a refused treatment dependency is not a cache key"]
    pub fn try_new(name: &str, coding: SemanticDigest) -> Result<Self, NuisanceCacheError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(NuisanceCacheError::Blank { field: "treatment" });
        }
        Ok(Self { name: name.to_owned(), coding })
    }

    /// Trimmed treatment name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Coding identity.
    #[must_use]
    pub const fn coding(&self) -> SemanticDigest {
        self.coding
    }
}

/// Provider that produced the nuisance, including its version.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProviderDependency {
    id: ProviderId,
    version: SemanticDigest,
}

impl ProviderDependency {
    /// Provider id and version digest.
    #[must_use]
    pub const fn new(id: ProviderId, version: SemanticDigest) -> Self {
        Self { id, version }
    }

    /// Provider id.
    #[must_use]
    pub const fn id(self) -> ProviderId {
        self.id
    }

    /// Provider version.
    #[must_use]
    pub const fn version(self) -> SemanticDigest {
        self.version
    }
}

/// One named numerical parameter. Equality is bitwise.
///
/// Negative zero is stored as positive zero. A non-finite value is refused.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NumericalDependency {
    name: String,
    bits: u64,
}

impl NumericalDependency {
    /// Finite parameter. The stored name is trimmed.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Blank`] when `name` is empty after trimming.
    /// [`NuisanceCacheError::NonFiniteDependency`] when `value` is not finite.
    #[must_use = "a refused numerical dependency is not a cache key"]
    pub fn try_new(name: &str, value: f64) -> Result<Self, NuisanceCacheError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(NuisanceCacheError::Blank { field: "numerical dependency" });
        }
        if !value.is_finite() {
            return Err(NuisanceCacheError::NonFiniteDependency { name: name.to_owned() });
        }
        Ok(Self { name: name.to_owned(), bits: canonical_bits(value) })
    }

    /// Trimmed parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Finite parameter value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// Canonical bits of [`Self::value`].
    #[must_use]
    pub const fn bits(&self) -> u64 {
        self.bits
    }
}

/// Numerical policy and named parameters the fit depends on.
///
/// Strict policy is not a zero tolerance. Named parameters are ordered by name;
/// input order is not part of the key.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NumericalDependencies {
    policy: NumericalPolicy,
    dependencies: Box<[NumericalDependency]>,
}

impl NumericalDependencies {
    /// Policy plus named parameters.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::DuplicateNumericalDependency`] when two parameters share a name.
    #[must_use = "a refused numerical dependency set is not a cache key"]
    pub fn try_new(
        policy: NumericalPolicy,
        dependencies: &[NumericalDependency],
    ) -> Result<Self, NuisanceCacheError> {
        let mut dependencies = dependencies.to_vec();
        dependencies.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        for pair in dependencies.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(NuisanceCacheError::DuplicateNumericalDependency {
                    name: pair[0].name.clone(),
                });
            }
        }
        Ok(Self { policy, dependencies: dependencies.into_boxed_slice() })
    }

    /// Numerical policy. Strict is distinct from a zero tolerance.
    #[must_use]
    pub const fn policy(&self) -> NumericalPolicy {
        self.policy
    }

    /// Named parameters, in name order.
    #[must_use]
    pub const fn dependencies(&self) -> &[NumericalDependency] {
        &self.dependencies
    }
}

impl Ord for NumericalDependencies {
    fn cmp(&self, other: &Self) -> Ordering {
        policy_key(self.policy)
            .cmp(&policy_key(other.policy))
            .then_with(|| self.dependencies.cmp(&other.dependencies))
    }
}

impl PartialOrd for NumericalDependencies {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// One estimation row and the fold it belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FoldRow {
    row_id: u64,
    fold: u32,
}

impl FoldRow {
    /// Row id and fold index. The assignment checks the fold range.
    #[must_use]
    pub const fn new(row_id: u64, fold: u32) -> Self {
        Self { row_id, fold }
    }

    /// Row id.
    #[must_use]
    pub const fn row_id(&self) -> u64 {
        self.row_id
    }

    /// Fold index.
    #[must_use]
    pub const fn fold(&self) -> u32 {
        self.fold
    }
}

/// Fold assignment and the held-out evaluation rows that must stay out of it.
///
/// Held-out row ids and fold rows are disjoint. At least two folds are present,
/// and at least two of them are used, so an evaluation fold can have a training
/// complement.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FoldAssignment {
    fold_count: u32,
    rows: Box<[FoldRow]>,
    held_out: Box<[u64]>,
}

impl FoldAssignment {
    /// Assign `rows` to folds `0..fold_count`, and keep `held_out` out of that assignment.
    ///
    /// Row order and held-out order are not part of the key. Duplicate ids are refused.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::TooFewFolds`], [`NuisanceCacheError::EmptyFolds`],
    /// [`NuisanceCacheError::DuplicateRow`], [`NuisanceCacheError::FoldOutOfRange`],
    /// [`NuisanceCacheError::EmptyTraining`] when every row shares one fold, and
    /// [`NuisanceCacheError::HeldOutLeakage`] when a held-out row is also assigned.
    #[must_use = "a refused fold assignment is not a cache key"]
    pub fn try_new(
        fold_count: u32,
        rows: &[FoldRow],
        held_out: &[u64],
    ) -> Result<Self, NuisanceCacheError> {
        if fold_count < 2 {
            return Err(NuisanceCacheError::TooFewFolds { fold_count });
        }
        if rows.is_empty() {
            return Err(NuisanceCacheError::EmptyFolds);
        }
        let mut rows = rows.to_vec();
        rows.sort_unstable_by_key(FoldRow::row_id);
        for pair in rows.windows(2) {
            if pair[0].row_id == pair[1].row_id {
                return Err(NuisanceCacheError::DuplicateRow { row_id: pair[0].row_id });
            }
        }
        for row in &rows {
            if row.fold >= fold_count {
                return Err(NuisanceCacheError::FoldOutOfRange { fold: row.fold, fold_count });
            }
        }
        let first = rows[0].fold;
        if rows.iter().all(|row| row.fold == first) {
            return Err(NuisanceCacheError::EmptyTraining { fold: first });
        }
        let mut held_out = held_out.to_vec();
        held_out.sort_unstable();
        for pair in held_out.windows(2) {
            if pair[0] == pair[1] {
                return Err(NuisanceCacheError::DuplicateRow { row_id: pair[0] });
            }
        }
        for row_id in &held_out {
            if rows.binary_search_by_key(row_id, FoldRow::row_id).is_ok() {
                return Err(NuisanceCacheError::HeldOutLeakage { row_id: *row_id });
            }
        }
        Ok(Self {
            fold_count,
            rows: rows.into_boxed_slice(),
            held_out: held_out.into_boxed_slice(),
        })
    }

    /// Declared fold count.
    #[must_use]
    pub const fn fold_count(&self) -> u32 {
        self.fold_count
    }

    /// Assigned rows, in row-id order.
    #[must_use]
    pub const fn rows(&self) -> &[FoldRow] {
        &self.rows
    }

    /// Held-out evaluation rows, in row-id order.
    #[must_use]
    pub const fn held_out(&self) -> &[u64] {
        &self.held_out
    }

    /// Fold of an assigned row.
    #[must_use]
    pub fn fold_of(&self, row_id: u64) -> Option<u32> {
        let index = self.rows.binary_search_by_key(&row_id, FoldRow::row_id).ok()?;
        Some(self.rows[index].fold)
    }

    /// Whether `row_id` is a held-out evaluation row.
    #[must_use]
    pub fn is_held_out(&self, row_id: u64) -> bool {
        self.held_out.binary_search(&row_id).is_ok()
    }

    /// Whether `row_id` is assigned to a fold other than `evaluation_fold`.
    #[must_use]
    pub fn is_training_row(&self, evaluation_fold: u32, row_id: u64) -> bool {
        self.fold_of(row_id).is_some_and(|fold| fold != evaluation_fold)
    }
}

/// Which nuisance the stored predictions are.
///
/// The role is not a key field. Propensity and outcome blocks share a key and
/// stay in separate slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum NuisanceRole {
    /// Treatment or propensity fit.
    Treatment,
    /// Outcome fit at one treatment level.
    Outcome {
        /// Treatment level the outcome regression conditions on.
        level: u64,
    },
}

/// One finite prediction bound to the row it belongs to.
///
/// Equality is bitwise, including negative zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PredictedRow {
    row_id: u64,
    bits: u64,
}

impl PredictedRow {
    /// Finite prediction for `row_id`.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::NonFinitePrediction`] when `value` is not finite.
    #[must_use = "a refused prediction is not stored"]
    pub const fn try_new(row_id: u64, value: f64) -> Result<Self, NuisanceCacheError> {
        if !value.is_finite() {
            return Err(NuisanceCacheError::NonFinitePrediction { row_id });
        }
        Ok(Self { row_id, bits: value.to_bits() })
    }

    /// Row id.
    #[must_use]
    pub const fn row_id(&self) -> u64 {
        self.row_id
    }

    /// Stored prediction.
    #[must_use]
    pub const fn value(&self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// Bits of [`Self::value`], including the sign of zero.
    #[must_use]
    pub const fn bits(&self) -> u64 {
        self.bits
    }
}

/// Out-of-fold predictions for one evaluation fold.
///
/// Rows are stored in row-id order. The block does not include that fold's
/// training rows.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BlockedOutOfFoldPredictions {
    evaluation_fold: u32,
    rows: Box<[PredictedRow]>,
}

impl BlockedOutOfFoldPredictions {
    /// Predictions for rows held out in `evaluation_fold`.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::EmptyBlock`] when `rows` is empty.
    /// [`NuisanceCacheError::DuplicateRow`] when a row id is repeated.
    #[must_use = "a refused block is not an out-of-fold prediction"]
    pub fn try_new(
        evaluation_fold: u32,
        rows: &[PredictedRow],
    ) -> Result<Self, NuisanceCacheError> {
        if rows.is_empty() {
            return Err(NuisanceCacheError::EmptyBlock { fold: evaluation_fold });
        }
        let mut rows = rows.to_vec();
        rows.sort_unstable_by_key(PredictedRow::row_id);
        for pair in rows.windows(2) {
            if pair[0].row_id == pair[1].row_id {
                return Err(NuisanceCacheError::DuplicateRow { row_id: pair[0].row_id });
            }
        }
        Ok(Self { evaluation_fold, rows: rows.into_boxed_slice() })
    }

    /// Fold these predictions were held out of.
    #[must_use]
    pub const fn evaluation_fold(&self) -> u32 {
        self.evaluation_fold
    }

    /// Predictions in row-id order.
    #[must_use]
    pub const fn rows(&self) -> &[PredictedRow] {
        &self.rows
    }

    fn bits(&self, row_id: u64) -> Option<u64> {
        let index = self.rows.binary_search_by_key(&row_id, PredictedRow::row_id).ok()?;
        Some(self.rows[index].bits)
    }
}

/// A certified fit stored under one key and role.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedNuisanceFit {
    key: NuisanceCacheKey,
    role: NuisanceRole,
    blocks: Box<[BlockedOutOfFoldPredictions]>,
}

impl CertifiedNuisanceFit {
    /// Key the fit was certified under.
    #[must_use]
    pub const fn key(&self) -> &NuisanceCacheKey {
        &self.key
    }

    /// Nuisance role.
    #[must_use]
    pub const fn role(&self) -> NuisanceRole {
        self.role
    }

    /// Fold blocks, in evaluation-fold order.
    #[must_use]
    pub const fn blocks(&self) -> &[BlockedOutOfFoldPredictions] {
        &self.blocks
    }

    /// Put each stored prediction on the requested row id.
    ///
    /// `row_ids` chooses the output order. A training row of `evaluation_fold`
    /// is refused, including when that row has no stored prediction. A held-out
    /// row is refused.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::FoldOutOfRange`], [`NuisanceCacheError::HeldOutLeakage`],
    /// [`NuisanceCacheError::TrainingLeakage`], [`NuisanceCacheError::RowNotInFold`],
    /// [`NuisanceCacheError::UnknownFold`], and [`NuisanceCacheError::MissingPrediction`].
    #[must_use = "a refused restoration is not the stored row values"]
    pub fn restore_rows(
        &self,
        evaluation_fold: u32,
        row_ids: &[u64],
    ) -> Result<Vec<f64>, NuisanceCacheError> {
        if evaluation_fold >= self.key.folds.fold_count {
            return Err(NuisanceCacheError::FoldOutOfRange {
                fold: evaluation_fold,
                fold_count: self.key.folds.fold_count,
            });
        }
        for &row_id in row_ids {
            refuse_leakage(&self.key.folds, evaluation_fold, row_id)?;
        }
        let block = self
            .blocks
            .iter()
            .find(|block| block.evaluation_fold == evaluation_fold)
            .ok_or(NuisanceCacheError::UnknownFold { fold: evaluation_fold })?;
        let mut restored = Vec::with_capacity(row_ids.len());
        for &row_id in row_ids {
            let bits =
                block.bits(row_id).ok_or(NuisanceCacheError::MissingPrediction { row_id })?;
            restored.push(f64::from_bits(bits));
        }
        Ok(restored)
    }
}

/// Exact-key cache of certified nuisance predictions.
///
/// A hit requires every key field to match. This cache does not fit a model.
#[derive(Clone, Debug, Default)]
pub struct NuisanceCache {
    entries: BTreeMap<NuisanceCacheKey, BTreeMap<NuisanceRole, CertifiedNuisanceFit>>,
}

impl NuisanceCache {
    /// An empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: BTreeMap::new() }
    }

    /// Store blocked out-of-fold predictions under `key` and `role`.
    ///
    /// The same predictions may be stored again. A different prediction for the
    /// same key and role is refused. Training rows and held-out rows are refused
    /// before anything is retained.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::EmptyBlocks`], [`NuisanceCacheError::DuplicateFold`],
    /// [`NuisanceCacheError::TrainingLeakage`], [`NuisanceCacheError::HeldOutLeakage`],
    /// [`NuisanceCacheError::EmptyTraining`], [`NuisanceCacheError::FoldOutOfRange`],
    /// [`NuisanceCacheError::RowNotInFold`], and [`NuisanceCacheError::ConflictingFit`].
    #[must_use = "a refused store is not a certified fit"]
    pub fn store_certified(
        &mut self,
        key: &NuisanceCacheKey,
        role: NuisanceRole,
        blocks: &[BlockedOutOfFoldPredictions],
    ) -> Result<CertifiedNuisanceFit, NuisanceCacheError> {
        let prepared = prepare_blocks(key.folds(), blocks)?;
        match self.entries.get(key).and_then(|roles| roles.get(&role)) {
            Some(existing) if existing.blocks.as_ref() == prepared.as_ref() => Ok(existing.clone()),
            Some(_) => Err(NuisanceCacheError::ConflictingFit),
            None => {
                let certified = CertifiedNuisanceFit { key: key.clone(), role, blocks: prepared };
                self.entries.entry(key.clone()).or_default().insert(role, certified.clone());
                Ok(certified)
            }
        }
    }

    /// The fit stored under this key and role, borrowed from the cache.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Miss`] when no fit uses this key.
    /// [`NuisanceCacheError::UnknownRole`] when the key is stored and the role is not.
    #[must_use = "a cache miss is not a certified fit"]
    pub fn certified(
        &self,
        key: &NuisanceCacheKey,
        role: NuisanceRole,
    ) -> Result<&CertifiedNuisanceFit, NuisanceCacheError> {
        let roles = self.entries.get(key).ok_or(NuisanceCacheError::Miss)?;
        roles.get(&role).ok_or(NuisanceCacheError::UnknownRole)
    }

    /// Restore `role` under `key` onto `row_ids`.
    ///
    /// A key that differs in any field is a miss, before row checks. On a hit,
    /// restoration follows [`CertifiedNuisanceFit::restore_rows`].
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Miss`], [`NuisanceCacheError::UnknownRole`], and the
    /// restoration errors of [`CertifiedNuisanceFit::restore_rows`].
    #[must_use = "a refused restoration is not the stored row values"]
    pub fn restore_rows(
        &self,
        key: &NuisanceCacheKey,
        role: NuisanceRole,
        evaluation_fold: u32,
        row_ids: &[u64],
    ) -> Result<Vec<f64>, NuisanceCacheError> {
        self.certified(key, role)?.restore_rows(evaluation_fold, row_ids)
    }
}

/// Outcome, treatment, design, options, and weights a prepared fit reads.
///
/// The slices are hashed by [`NuisanceCacheKey::from_inputs`]. Their lengths do
/// not have to agree; a different length is a different key.
#[derive(Clone, Copy, Debug)]
pub struct NuisanceColumns<'a> {
    outcome: &'a [f64],
    treatment: &'a [f64],
    design: &'a [f64],
    options: &'a [f64],
    weights: Option<&'a [f64]>,
}

impl<'a> NuisanceColumns<'a> {
    /// Columns of one nuisance fit.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::NonFiniteInput`] when any value is not finite.
    #[must_use = "a refused column set is not a cache key"]
    pub fn try_new(
        outcome: &'a [f64],
        treatment: &'a [f64],
        design: &'a [f64],
        options: &'a [f64],
        weights: Option<&'a [f64]>,
    ) -> Result<Self, NuisanceCacheError> {
        let columns = Self { outcome, treatment, design, options, weights };
        columns.refuse_non_finite()?;
        Ok(columns)
    }

    /// Outcome column.
    #[must_use]
    pub const fn outcome(&self) -> &[f64] {
        self.outcome
    }

    /// Treatment column.
    #[must_use]
    pub const fn treatment(&self) -> &[f64] {
        self.treatment
    }

    /// Design matrix, in the caller's layout.
    #[must_use]
    pub const fn design(&self) -> &[f64] {
        self.design
    }

    /// Numerical options of the learners, in the caller's order.
    #[must_use]
    pub const fn options(&self) -> &[f64] {
        self.options
    }

    /// Sampling weights, when the fit reads them.
    #[must_use]
    pub const fn weights(&self) -> Option<&[f64]> {
        self.weights
    }

    fn refuse_non_finite(&self) -> Result<(), NuisanceCacheError> {
        finite("outcome", self.outcome)?;
        finite("treatment", self.treatment)?;
        finite("design", self.design)?;
        finite("options", self.options)?;
        if let Some(weights) = self.weights {
            finite("weights", weights)?;
        }
        Ok(())
    }
}

/// Prepared fit inputs that [`NuisanceCacheKey::from_inputs`] hashes.
#[derive(Clone, Copy, Debug)]
pub struct NuisanceInputs<'a> {
    columns: NuisanceColumns<'a>,
    seed: u64,
    folds: &'a FoldAssignment,
    treatment_name: &'a str,
    policy: NumericalPolicy,
}

impl<'a> NuisanceInputs<'a> {
    /// Bind the columns, seed, folds, treatment name, and numerical policy.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Blank`] when `treatment_name` is empty after trimming.
    #[must_use = "a refused input set is not a cache key"]
    pub fn try_new(
        columns: NuisanceColumns<'a>,
        seed: u64,
        folds: &'a FoldAssignment,
        treatment_name: &'a str,
        policy: NumericalPolicy,
    ) -> Result<Self, NuisanceCacheError> {
        if treatment_name.trim().is_empty() {
            return Err(NuisanceCacheError::Blank { field: "treatment" });
        }
        Ok(Self { columns, seed, folds, treatment_name, policy })
    }

    /// Columns hashed into the key.
    #[must_use]
    pub const fn columns(&self) -> NuisanceColumns<'_> {
        self.columns
    }

    /// Fit seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Fold assignment copied into the key.
    #[must_use]
    pub const fn folds(&self) -> &FoldAssignment {
        self.folds
    }

    /// Treatment name stored on the key.
    #[must_use]
    pub const fn treatment_name(&self) -> &str {
        self.treatment_name
    }

    /// Numerical policy stored on the key.
    #[must_use]
    pub const fn policy(&self) -> NumericalPolicy {
        self.policy
    }
}

/// Cache key. A hit requires every field to match.
///
/// [`Self::new`] records dependencies the caller already checked and does not
/// bind an outcome, a weight vector, or a seed. [`Self::from_inputs`] is the
/// constructor for a prepared fit: those values are hashed into the key.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NuisanceCacheKey {
    target: TargetDependency,
    design: DesignDependency,
    folds: FoldAssignment,
    treatment: TreatmentDependency,
    provider: ProviderDependency,
    numerical: NumericalDependencies,
    outcome: SemanticDigest,
    weights: SemanticDigest,
    seed: u64,
    design_payload: SemanticDigest,
    treatment_payload: SemanticDigest,
    options: SemanticDigest,
}

impl NuisanceCacheKey {
    /// Assemble a key from dependencies that have already been checked.
    #[must_use]
    pub const fn new(
        target: TargetDependency,
        design: DesignDependency,
        folds: FoldAssignment,
        treatment: TreatmentDependency,
        provider: ProviderDependency,
        numerical: NumericalDependencies,
    ) -> Self {
        Self {
            target,
            design,
            folds,
            treatment,
            provider,
            numerical,
            outcome: UNBOUND,
            weights: UNBOUND,
            seed: 0,
            design_payload: UNBOUND,
            treatment_payload: UNBOUND,
            options: UNBOUND,
        }
    }

    /// Key of one prepared fit.
    ///
    /// Each column is a BLAKE3 digest of a domain-separated
    /// [`CanonicalWriter`] stream. Reals are written with [`CanonicalWriter::reals`],
    /// so equality is bitwise and `-0.0` differs from `0.0`.
    ///
    /// - outcome: tag `antecedent.estimate.nuisance.outcome`
    /// - treatment: tag `antecedent.estimate.nuisance.treatment`, also stored as the
    ///   treatment coding
    /// - design: tag `antecedent.estimate.nuisance.design`; the design id is the
    ///   first eight little-endian bytes, and the full digest is kept beside it
    /// - options: tag `antecedent.estimate.nuisance.options`, the numerical-policy
    ///   words, then the option reals
    /// - weights: tag `antecedent.estimate.nuisance.weights`, a presence word, then
    ///   the reals when weights are present
    /// - seed: the `u64` itself
    ///
    /// The target digest and the provider version are the digest of an empty
    /// stream under `antecedent.estimate.nuisance.derived`. They mark a derived
    /// key. They do not carry the outcome, the weights, or the seed.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Blank`] when the treatment name is empty after trimming.
    #[must_use = "a refused key is not a cache lookup"]
    pub fn from_inputs(inputs: &NuisanceInputs<'_>) -> Result<Self, NuisanceCacheError> {
        let outcome = digest_reals("antecedent.estimate.nuisance.outcome", inputs.columns.outcome);
        let treatment_payload =
            digest_reals("antecedent.estimate.nuisance.treatment", inputs.columns.treatment);
        let design_payload =
            digest_reals("antecedent.estimate.nuisance.design", inputs.columns.design);
        let options = digest_options(inputs.policy, inputs.columns.options);
        let weights = digest_weights(inputs.columns.weights);
        let marker = digest_reals("antecedent.estimate.nuisance.derived", &[]);
        let treatment = TreatmentDependency::try_new(inputs.treatment_name, treatment_payload)?;
        let numerical = NumericalDependencies::try_new(inputs.policy, &[])?;
        Ok(Self {
            target: TargetDependency::new(marker),
            design: DesignDependency::new(DesignDependencyId::from_raw(leading_word(
                design_payload,
            ))),
            folds: FoldAssignment::clone(inputs.folds),
            treatment,
            provider: ProviderDependency::new(ProviderId::from_raw(1), marker),
            numerical,
            outcome,
            weights,
            seed: inputs.seed,
            design_payload,
            treatment_payload,
            options,
        })
    }

    /// Target dependency.
    #[must_use]
    pub const fn target(&self) -> TargetDependency {
        self.target
    }

    /// Design dependency.
    #[must_use]
    pub const fn design(&self) -> DesignDependency {
        self.design
    }

    /// Fold assignment.
    #[must_use]
    pub const fn folds(&self) -> &FoldAssignment {
        &self.folds
    }

    /// Treatment dependency.
    #[must_use]
    pub const fn treatment(&self) -> &TreatmentDependency {
        &self.treatment
    }

    /// Provider dependency.
    #[must_use]
    pub const fn provider(&self) -> ProviderDependency {
        self.provider
    }

    /// Numerical dependencies.
    #[must_use]
    pub const fn numerical(&self) -> &NumericalDependencies {
        &self.numerical
    }

    /// Digest of the outcome column.
    ///
    /// All zeros when the key came from [`Self::new`] and no outcome was bound.
    #[must_use]
    pub const fn outcome_digest(&self) -> SemanticDigest {
        self.outcome
    }

    /// Digest of the sampling weights.
    ///
    /// All zeros when the key came from [`Self::new`] and no weights were bound.
    /// That zero digest is not the digest of absent weights.
    #[must_use]
    pub const fn weight_digest(&self) -> SemanticDigest {
        self.weights
    }

    /// Fit seed. Zero when the key came from [`Self::new`] and did not bind one.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Digest of the numerical policy and option vector.
    #[must_use]
    pub const fn options_digest(&self) -> SemanticDigest {
        self.options
    }

    /// First field that differs, in [`KeyField`] order.
    ///
    /// `None` means every field matches.
    #[must_use]
    pub fn mismatch(&self, other: &Self) -> Option<KeyField> {
        if self.target != other.target {
            return Some(KeyField::Target);
        }
        if self.design != other.design || self.design_payload != other.design_payload {
            return Some(KeyField::Design);
        }
        if self.folds != other.folds {
            return Some(KeyField::Folds);
        }
        if self.treatment != other.treatment || self.treatment_payload != other.treatment_payload {
            return Some(KeyField::Treatment);
        }
        if self.provider != other.provider {
            return Some(KeyField::Provider);
        }
        if self.numerical != other.numerical || self.options != other.options {
            return Some(KeyField::Numerical);
        }
        if self.outcome != other.outcome {
            return Some(KeyField::Outcome);
        }
        if self.weights != other.weights {
            return Some(KeyField::Weights);
        }
        if self.seed != other.seed {
            return Some(KeyField::Seed);
        }
        None
    }
}

/// Estimator family that may reuse a certified nuisance.
///
/// The family is not a cache-key field. Two requests share a subplan when their
/// certified source and [`NuisanceCacheKey`] match, including across these families.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EstimatorFamily {
    /// Cross-fitted, cell, or population AIPW.
    Aipw,
    /// Partially linear or AIPW double machine learning, including cluster DML.
    Dml,
    /// Policy value from a certified nuisance.
    Policy,
}

impl EstimatorFamily {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Aipw => "aipw",
            Self::Dml => "dml",
            Self::Policy => "policy",
        }
    }
}

/// Column names a request mentions.
///
/// Names are not a compatibility key. Two requests with the same names and a
/// different certified source or cache key do not share a subplan.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RequestColumns {
    outcome: String,
    treatment: String,
    covariates: Box<[String]>,
}

impl RequestColumns {
    /// Outcome, treatment, and covariate names.
    ///
    /// Stored names are trimmed. Covariate order is kept.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::Blank`] when a name is empty after trimming.
    #[must_use = "a refused column list is not part of a prepared request"]
    pub fn try_new(
        outcome: &str,
        treatment: &str,
        covariates: &[&str],
    ) -> Result<Self, NuisanceCacheError> {
        let outcome = trim_name(outcome, "outcome column")?;
        let treatment = trim_name(treatment, "treatment column")?;
        let mut stored = Vec::with_capacity(covariates.len());
        for name in covariates {
            stored.push(trim_name(name, "covariate column")?);
        }
        Ok(Self { outcome, treatment, covariates: stored.into_boxed_slice() })
    }

    /// Trimmed outcome name.
    #[must_use]
    pub fn outcome(&self) -> &str {
        &self.outcome
    }

    /// Trimmed treatment name.
    #[must_use]
    pub fn treatment(&self) -> &str {
        &self.treatment
    }

    /// Covariate names, in the order they were given.
    #[must_use]
    pub fn covariates(&self) -> &[String] {
        &self.covariates
    }
}

/// Certified source identity.
///
/// The id is not the rows. `content` distinguishes two snapshots that reuse an
/// id. Column names are not this identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CertifiedSource {
    id: SourceDependencyId,
    content: SemanticDigest,
}

impl CertifiedSource {
    /// Source id and content digest.
    #[must_use]
    pub const fn new(id: SourceDependencyId, content: SemanticDigest) -> Self {
        Self { id, content }
    }

    /// Source id.
    #[must_use]
    pub const fn id(self) -> SourceDependencyId {
        self.id
    }

    /// Content digest.
    #[must_use]
    pub const fn content(self) -> SemanticDigest {
        self.content
    }
}

/// One prepared or batch request.
///
/// Compatibility is the certified source plus every [`NuisanceCacheKey`] field.
/// [`RequestColumns`] are recorded and are not compared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedFamilyRequest {
    family: EstimatorFamily,
    columns: RequestColumns,
    source: CertifiedSource,
    key: NuisanceCacheKey,
    roles: Box<[NuisanceRole]>,
}

impl PreparedFamilyRequest {
    /// Assemble a request.
    ///
    /// Duplicate roles are kept once. Role order is treatment, then outcome level.
    ///
    /// # Errors
    ///
    /// [`NuisanceCacheError::EmptyRoles`] when `roles` is empty.
    #[must_use = "a refused request is not compiled into a subplan"]
    pub fn try_new(
        family: EstimatorFamily,
        columns: RequestColumns,
        source: CertifiedSource,
        key: NuisanceCacheKey,
        roles: &[NuisanceRole],
    ) -> Result<Self, NuisanceCacheError> {
        if roles.is_empty() {
            return Err(NuisanceCacheError::EmptyRoles);
        }
        Ok(Self { family, columns, source, key, roles: unique_roles(roles) })
    }

    /// Estimator family.
    #[must_use]
    pub const fn family(&self) -> EstimatorFamily {
        self.family
    }

    /// Column names. Not a compatibility key.
    #[must_use]
    pub const fn columns(&self) -> &RequestColumns {
        &self.columns
    }

    /// Certified source.
    #[must_use]
    pub const fn source(&self) -> CertifiedSource {
        self.source
    }

    /// Certified nuisance key.
    #[must_use]
    pub const fn key(&self) -> &NuisanceCacheKey {
        &self.key
    }

    /// Nuisance roles this request needs, in role order.
    #[must_use]
    pub const fn roles(&self) -> &[NuisanceRole] {
        &self.roles
    }
}

/// Requests that share one certified source and one nuisance key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedSubplan {
    source: CertifiedSource,
    key: NuisanceCacheKey,
    families: Box<[EstimatorFamily]>,
    members: Box<[usize]>,
    roles: Box<[NuisanceRole]>,
}

impl SharedSubplan {
    /// Certified source scanned once for every member.
    #[must_use]
    pub const fn source(&self) -> CertifiedSource {
        self.source
    }

    /// Cache key of the shared fit.
    #[must_use]
    pub const fn key(&self) -> &NuisanceCacheKey {
        &self.key
    }

    /// Families of the members, in first-seen order.
    #[must_use]
    pub const fn families(&self) -> &[EstimatorFamily] {
        &self.families
    }

    /// Indexes into the request slice that compiled this plan, in input order.
    #[must_use]
    pub const fn members(&self) -> &[usize] {
        &self.members
    }

    /// Roles fitted once for every member.
    #[must_use]
    pub const fn roles(&self) -> &[NuisanceRole] {
        &self.roles
    }

    fn accepts(&self, request: &PreparedFamilyRequest) -> bool {
        self.source == request.source && self.key.mismatch(&request.key).is_none()
    }

    fn push(&mut self, index: usize, request: &PreparedFamilyRequest) {
        let mut members = self.members.to_vec();
        members.push(index);
        self.members = members.into_boxed_slice();
        if !self.families.contains(&request.family) {
            let mut families = self.families.to_vec();
            families.push(request.family);
            self.families = families.into_boxed_slice();
        }
        self.roles = unique_roles(&[self.roles.as_ref(), request.roles.as_ref()].concat());
    }
}

/// Subplans compiled from one prepared or batch request list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledSubplans {
    plans: Box<[SharedSubplan]>,
}

impl CompiledSubplans {
    /// Subplans in first-seen order.
    #[must_use]
    pub const fn plans(&self) -> &[SharedSubplan] {
        &self.plans
    }
}

/// Scans, role fits, and certified-cache hits of one execution.
///
/// A scan is one certified source read. A fit is one nuisance role computed
/// for a subplan. Members of a subplan do not multiply either count. A later
/// execution that only restores certified fits counts hits and no scan or fit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SubplanReceipt {
    /// Certified sources read to compute a missing fit.
    pub scans: u64,
    /// Nuisance roles computed.
    pub fits: u64,
    /// Nuisance roles restored from the certified cache.
    pub hits: u64,
}

/// Certified fits produced for one compiled batch, and the work that produced them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchExecution {
    plans: usize,
    fits: Box<[CertifiedNuisanceFit]>,
    receipt: SubplanReceipt,
}

impl BatchExecution {
    /// Number of subplans executed.
    #[must_use]
    pub const fn plans(&self) -> usize {
        self.plans
    }

    /// Fits in subplan order, and role order within a subplan.
    #[must_use]
    pub const fn fits(&self) -> &[CertifiedNuisanceFit] {
        &self.fits
    }

    /// Scan, fit, and hit counts.
    #[must_use]
    pub const fn receipt(&self) -> SubplanReceipt {
        self.receipt
    }
}

/// Compile requests that share a certified source and nuisance key.
///
/// Matching column names are not enough. A different source, or any different
/// [`NuisanceCacheKey`] field, stays on its own subplan.
///
/// # Errors
///
/// [`NuisanceCacheError::EmptyRequests`] when `requests` is empty.
#[must_use = "a refused compilation is not a shared subplan"]
pub fn compile_shared_subplans(
    requests: &[PreparedFamilyRequest],
) -> Result<CompiledSubplans, NuisanceCacheError> {
    if requests.is_empty() {
        return Err(NuisanceCacheError::EmptyRequests);
    }
    let mut plans: Vec<SharedSubplan> = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        if let Some(plan) = plans.iter_mut().find(|plan| plan.accepts(request)) {
            plan.push(index, request);
        } else {
            plans.push(SharedSubplan {
                source: request.source,
                key: request.key.clone(),
                families: Box::new([request.family]),
                members: Box::new([index]),
                roles: request.roles.clone(),
            });
        }
    }
    Ok(CompiledSubplans { plans: plans.into_boxed_slice() })
}

/// Execute compiled subplans against the certified cache.
///
/// The returned execution is the restored or newly stored fits.
///
/// A role already stored under the plan's key and source is restored. A miss
/// calls `produce` once and stores that block. A key already bound to a
/// different certified source in this batch is a miss and is not written back
/// over the other source's fit. The second execution of a compatible plan
/// restores and does not call `produce`.
///
/// # Errors
///
/// Producer errors, and the store errors of [`NuisanceCache::store_certified`].
#[must_use = "a refused execution is not a certified fit"]
pub fn execute_compiled_subplans(
    cache: &mut NuisanceCache,
    compiled: &CompiledSubplans,
    mut produce: impl FnMut(
        &NuisanceCacheKey,
        NuisanceRole,
    ) -> Result<Vec<BlockedOutOfFoldPredictions>, NuisanceCacheError>,
) -> Result<BatchExecution, NuisanceCacheError> {
    let mut fits = Vec::new();
    let mut receipt = SubplanReceipt::default();
    let mut scanned = BTreeSet::new();
    let mut bindings: BTreeMap<NuisanceCacheKey, CertifiedSource> = BTreeMap::new();
    for plan in compiled.plans() {
        let foreign_source = bindings.get(plan.key()).is_some_and(|bound| bound != &plan.source);
        let mut missed = false;
        for &role in plan.roles() {
            let stored = if foreign_source {
                None
            } else {
                match cache.certified(plan.key(), role) {
                    Ok(fit) => Some(fit.clone()),
                    Err(NuisanceCacheError::Miss | NuisanceCacheError::UnknownRole) => None,
                    Err(error) => return Err(error),
                }
            };
            if let Some(fit) = stored {
                receipt.hits = receipt.hits.saturating_add(1);
                fits.push(fit);
                continue;
            }
            let blocks = produce(plan.key(), role)?;
            missed = true;
            if foreign_source {
                let prepared = prepare_blocks(plan.key().folds(), &blocks)?;
                fits.push(CertifiedNuisanceFit { key: plan.key().clone(), role, blocks: prepared });
            } else {
                fits.push(cache.store_certified(plan.key(), role, &blocks)?);
            }
            receipt.fits = receipt.fits.saturating_add(1);
        }
        if missed && scanned.insert(plan.source) {
            receipt.scans = receipt.scans.saturating_add(1);
        }
        bindings.entry(plan.key().clone()).or_insert(plan.source);
    }
    Ok(BatchExecution { plans: compiled.plans().len(), fits: fits.into_boxed_slice(), receipt })
}

/// Compile `requests` and execute them as one batch.
///
/// # Errors
///
/// [`NuisanceCacheError::WrongFamily`] when a request is not `expected`.
/// The compilation and execution errors of [`compile_shared_subplans`] and
/// [`execute_compiled_subplans`].
#[must_use = "a refused family execution is not a certified fit"]
pub fn execute_family_requests(
    expected: EstimatorFamily,
    cache: &mut NuisanceCache,
    requests: &[PreparedFamilyRequest],
    produce: impl FnMut(
        &NuisanceCacheKey,
        NuisanceRole,
    ) -> Result<Vec<BlockedOutOfFoldPredictions>, NuisanceCacheError>,
) -> Result<BatchExecution, NuisanceCacheError> {
    if let Some(request) = requests.iter().find(|request| request.family != expected) {
        return Err(NuisanceCacheError::WrongFamily {
            request: request.family.as_str(),
            expected: expected.as_str(),
        });
    }
    let compiled = compile_shared_subplans(requests)?;
    execute_compiled_subplans(cache, &compiled, produce)
}

/// Execute each request on its own empty cache.
///
/// The predictions match a shared batch when `produce` is a function of the key
/// and role. Scan and fit counts are not shared across requests.
///
/// # Errors
///
/// The errors of [`execute_family_requests`], checked per request. An empty
/// slice is an empty execution, not [`NuisanceCacheError::EmptyRequests`].
#[must_use = "a refused independent execution is not a certified fit"]
pub fn execute_independently(
    requests: &[PreparedFamilyRequest],
    mut produce: impl FnMut(
        &NuisanceCacheKey,
        NuisanceRole,
    ) -> Result<Vec<BlockedOutOfFoldPredictions>, NuisanceCacheError>,
) -> Result<BatchExecution, NuisanceCacheError> {
    let mut fits = Vec::new();
    let mut receipt = SubplanReceipt::default();
    for request in requests {
        let family = request.family;
        let mut cache = NuisanceCache::new();
        let execution = execute_family_requests(
            family,
            &mut cache,
            std::slice::from_ref(request),
            &mut produce,
        )?;
        receipt.scans = receipt.scans.saturating_add(execution.receipt.scans);
        receipt.fits = receipt.fits.saturating_add(execution.receipt.fits);
        receipt.hits = receipt.hits.saturating_add(execution.receipt.hits);
        fits.extend(execution.fits.iter().cloned());
    }
    Ok(BatchExecution { plans: requests.len(), fits: fits.into_boxed_slice(), receipt })
}

/// Group aligned rows into out-of-fold blocks.
///
/// `row_ids`, `folds`, and `values` are one prediction in the same order.
///
/// # Errors
///
/// [`NuisanceCacheError::UnalignedPredictions`] when the slices differ in length.
/// [`PredictedRow::try_new`] and [`BlockedOutOfFoldPredictions::try_new`] errors.
#[must_use = "a refused block is not an out-of-fold prediction"]
pub fn blocks_by_fold(
    row_ids: &[u64],
    folds: &[u32],
    values: &[f64],
) -> Result<Vec<BlockedOutOfFoldPredictions>, NuisanceCacheError> {
    if row_ids.len() != folds.len() || folds.len() != values.len() {
        return Err(NuisanceCacheError::UnalignedPredictions);
    }
    let mut grouped: BTreeMap<u32, Vec<PredictedRow>> = BTreeMap::new();
    for (index, &row_id) in row_ids.iter().enumerate() {
        grouped
            .entry(folds[index])
            .or_default()
            .push(PredictedRow::try_new(row_id, values[index])?);
    }
    let mut blocks = Vec::with_capacity(grouped.len());
    for (fold, rows) in grouped {
        blocks.push(BlockedOutOfFoldPredictions::try_new(fold, &rows)?);
    }
    Ok(blocks)
}

fn trim_name(name: &str, field: &'static str) -> Result<String, NuisanceCacheError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(NuisanceCacheError::Blank { field });
    }
    Ok(name.to_owned())
}

fn unique_roles(roles: &[NuisanceRole]) -> Box<[NuisanceRole]> {
    let mut roles = roles.to_vec();
    roles.sort_unstable();
    roles.dedup();
    roles.into_boxed_slice()
}

struct ActiveCertification {
    cache: Arc<Mutex<NuisanceCache>>,
    key: NuisanceCacheKey,
    scans: u64,
    fits: u64,
    hits: u64,
}

std::thread_local! {
    static CERTIFIED: std::cell::RefCell<Option<ActiveCertification>> =
        const { std::cell::RefCell::new(None) };
}

struct CertificationGuard<'a> {
    destination: &'a mut NuisanceCache,
    shared: Arc<Mutex<NuisanceCache>>,
    previous: Option<ActiveCertification>,
    receipt: SubplanReceipt,
    finished: bool,
}

impl CertificationGuard<'_> {
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let active = CERTIFIED.with(|slot| {
            let current = slot.borrow_mut().take();
            *slot.borrow_mut() = self.previous.take();
            current
        });
        if let Some(active) = active {
            self.receipt =
                SubplanReceipt { scans: active.scans, fits: active.fits, hits: active.hits };
        }
        let mut guard = self.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *self.destination = std::mem::take(&mut guard);
    }
}

impl Drop for CertificationGuard<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Run `work` with `cache` and `key` active on this thread.
///
/// The scope is thread-local. [`map_indexed`] does not install it and does not
/// read it: a worker receives the cache as an argument. Estimators that consult
/// this scope do so only on the thread that entered it. An empty cache misses
/// and the estimator fits as it does with no scope. A panic restores `cache`
/// and the enclosing scope.
#[must_use = "the receipt counts scans, fits, and certified hits"]
pub fn with_certified_cache<R>(
    cache: &mut NuisanceCache,
    key: &NuisanceCacheKey,
    work: impl FnOnce() -> R,
) -> (R, SubplanReceipt) {
    let shared = Arc::new(Mutex::new(std::mem::take(cache)));
    let previous = CERTIFIED.with(|slot| {
        slot.borrow_mut().replace(ActiveCertification {
            cache: Arc::clone(&shared),
            key: key.clone(),
            scans: 0,
            fits: 0,
            hits: 0,
        })
    });
    let mut guard = CertificationGuard {
        destination: cache,
        shared,
        previous,
        receipt: SubplanReceipt::default(),
        finished: false,
    };
    let result = work();
    guard.finish();
    (result, guard.receipt)
}

/// Run `work` once per index.
///
/// `ctx` schedules the indexes. When more than one thread is available, each
/// worker runs on its own thread and receives `cache` as an argument. This
/// function does not open [`with_certified_cache`] on the calling thread or on
/// a worker, and the worker does not read that thread-local scope.
///
/// # Errors
///
/// Returns the lowest-index error produced by `work`.
#[must_use = "a worker error is not a completed indexed fit"]
pub fn map_indexed<T, E, F>(
    ctx: &ExecutionContext,
    cache: &NuisanceCache,
    n: usize,
    work: F,
) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize, &ExecutionContext, &NuisanceCache) -> Result<T, E> + Sync,
{
    ctx.map_indexed(n, |index, inner| work(index, inner, cache))
}

/// Whether [`with_certified_cache`] is active on this thread.
#[must_use]
pub fn certification_requested() -> bool {
    CERTIFIED.with(|slot| slot.borrow().is_some())
}

fn active_cache_and_key() -> Option<(Arc<Mutex<NuisanceCache>>, NuisanceCacheKey)> {
    CERTIFIED.with(|slot| {
        slot.borrow().as_ref().map(|active| (Arc::clone(&active.cache), active.key.clone()))
    })
}

/// Restore one role from the active certified cache.
///
/// `Ok(None)` is an inactive scope, a key miss, or a role that is not stored.
/// Leakage and fold errors are returned.
///
/// # Errors
///
/// Restoration errors other than [`NuisanceCacheError::Miss`] and
/// [`NuisanceCacheError::UnknownRole`].
pub fn restore_active(
    role: NuisanceRole,
    evaluation_fold: u32,
    row_ids: &[u64],
) -> Result<Option<Vec<f64>>, NuisanceCacheError> {
    let Some((cache, key)) = active_cache_and_key() else {
        return Ok(None);
    };
    let guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match guard.restore_rows(&key, role, evaluation_fold, row_ids) {
        Ok(values) => Ok(Some(values)),
        Err(NuisanceCacheError::Miss | NuisanceCacheError::UnknownRole) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Store one role in the active certified cache.
///
/// An inactive scope stores nothing.
///
/// # Errors
///
/// The errors of [`NuisanceCache::store_certified`] when a scope is active.
pub fn store_active(
    role: NuisanceRole,
    blocks: &[BlockedOutOfFoldPredictions],
) -> Result<(), NuisanceCacheError> {
    let Some((cache, key)) = active_cache_and_key() else {
        return Ok(());
    };
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.store_certified(&key, role, blocks)?;
    Ok(())
}

/// Store roles that were just fitted and count one source scan.
///
/// An inactive scope stores nothing and leaves the receipt unchanged.
///
/// # Errors
///
/// The errors of [`store_active`].
pub fn store_active_roles(
    roles: &[(NuisanceRole, Vec<BlockedOutOfFoldPredictions>)],
) -> Result<(), NuisanceCacheError> {
    if !certification_requested() {
        return Ok(());
    }
    for (role, blocks) in roles {
        store_active(*role, blocks)?;
    }
    let fits = u64::try_from(roles.len()).unwrap_or(u64::MAX);
    if fits > 0 {
        record_active_work(1, fits, 0);
    }
    Ok(())
}

/// Fold assignment of the active certified scope, when one is open.
#[must_use]
pub fn active_folds() -> Option<FoldAssignment> {
    CERTIFIED.with(|slot| slot.borrow().as_ref().map(|active| active.key.folds().clone()))
}

/// Count certified hits on the active scope.
pub fn record_active_hits(hits: u64) {
    record_active_work(0, 0, hits);
}

/// Count role fits that read the certified source on the active scope.
pub fn record_certified_fits(fits: u64) {
    if fits > 0 {
        record_active_work(1, fits, 0);
    }
}

fn record_active_work(scans: u64, fits: u64, hits: u64) {
    CERTIFIED.with(|slot| {
        if let Some(active) = slot.borrow_mut().as_mut() {
            active.scans = active.scans.saturating_add(scans);
            active.fits = active.fits.saturating_add(fits);
            active.hits = active.hits.saturating_add(hits);
        }
    });
}

const UNBOUND: SemanticDigest = SemanticDigest::from_bytes([0; 32]);

fn finite(field: &'static str, values: &[f64]) -> Result<(), NuisanceCacheError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NuisanceCacheError::NonFiniteInput { field });
    }
    Ok(())
}

fn digest_reals(tag: &str, values: &[f64]) -> SemanticDigest {
    SemanticDigest::from_bytes(CanonicalWriter::new(tag).reals(values).finish())
}

fn digest_weights(weights: Option<&[f64]>) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.estimate.nuisance.weights");
    match weights {
        Some(values) => {
            writer.word(1).reals(values);
        }
        None => {
            writer.word(0);
        }
    }
    SemanticDigest::from_bytes(writer.finish())
}

fn digest_options(policy: NumericalPolicy, options: &[f64]) -> SemanticDigest {
    let (kind, absolute, relative) = policy_key(policy);
    let bytes = CanonicalWriter::new("antecedent.estimate.nuisance.options")
        .word(u64::from(kind))
        .word(absolute)
        .word(relative)
        .reals(options)
        .finish();
    SemanticDigest::from_bytes(bytes)
}

const fn leading_word(digest: SemanticDigest) -> u64 {
    let bytes = digest.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
        bytes[7],
    ])
}

const fn canonical_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits == 0x8000_0000_0000_0000 { 0 } else { bits }
}

const fn policy_key(policy: NumericalPolicy) -> (u8, u64, u64) {
    match policy {
        NumericalPolicy::Strict => (0, 0, 0),
        NumericalPolicy::Tolerance(tolerance) => {
            (1, tolerance.absolute().to_bits(), tolerance.relative().to_bits())
        }
    }
}

fn refuse_leakage(
    folds: &FoldAssignment,
    evaluation_fold: u32,
    row_id: u64,
) -> Result<(), NuisanceCacheError> {
    if folds.is_held_out(row_id) {
        return Err(NuisanceCacheError::HeldOutLeakage { row_id });
    }
    if folds.is_training_row(evaluation_fold, row_id) {
        return Err(NuisanceCacheError::TrainingLeakage { fold: evaluation_fold, row_id });
    }
    if folds.fold_of(row_id) != Some(evaluation_fold) {
        return Err(NuisanceCacheError::RowNotInFold { fold: evaluation_fold, row_id });
    }
    Ok(())
}

fn certify_block(
    folds: &FoldAssignment,
    block: &BlockedOutOfFoldPredictions,
) -> Result<(), NuisanceCacheError> {
    let evaluation_fold = block.evaluation_fold;
    if evaluation_fold >= folds.fold_count {
        return Err(NuisanceCacheError::FoldOutOfRange {
            fold: evaluation_fold,
            fold_count: folds.fold_count,
        });
    }
    if !folds.rows.iter().any(|row| row.fold != evaluation_fold) {
        return Err(NuisanceCacheError::EmptyTraining { fold: evaluation_fold });
    }
    // `FoldAssignment::try_new` refuses any held-out id that is also an assigned row, so no
    // training row can be held out here.
    debug_assert!(folds.rows().iter().all(|row| !folds.is_held_out(row.row_id())));
    for predicted in block.rows() {
        refuse_leakage(folds, evaluation_fold, predicted.row_id())?;
    }
    Ok(())
}

fn prepare_blocks(
    folds: &FoldAssignment,
    blocks: &[BlockedOutOfFoldPredictions],
) -> Result<Box<[BlockedOutOfFoldPredictions]>, NuisanceCacheError> {
    if blocks.is_empty() {
        return Err(NuisanceCacheError::EmptyBlocks);
    }
    let mut ordered = blocks.to_vec();
    ordered.sort_unstable_by_key(BlockedOutOfFoldPredictions::evaluation_fold);
    for pair in ordered.windows(2) {
        if pair[0].evaluation_fold == pair[1].evaluation_fold {
            return Err(NuisanceCacheError::DuplicateFold { fold: pair[0].evaluation_fold });
        }
    }
    for block in &ordered {
        certify_block(folds, block)?;
    }
    Ok(ordered.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{
        ExecutionContext, NonZeroThreadCount, NumericalPolicy, Parallelism, PhysicalTolerance,
    };

    use super::{
        BatchExecution, BlockedOutOfFoldPredictions, CertifiedSource, DesignDependency,
        DesignDependencyId, EstimatorFamily, FoldAssignment, FoldRow, KeyField, NuisanceCache,
        NuisanceCacheError, NuisanceCacheKey, NuisanceColumns, NuisanceInputs, NuisanceRole,
        NumericalDependencies, NumericalDependency, PredictedRow, PreparedFamilyRequest,
        ProviderDependency, ProviderId, RequestColumns, SemanticDigest, SourceDependencyId,
        SubplanReceipt, TargetDependency, TreatmentDependency, compile_shared_subplans,
        execute_compiled_subplans, execute_family_requests, map_indexed,
    };

    fn digest(byte: u8) -> SemanticDigest {
        let mut bytes = [0u8; 32];
        bytes[0] = byte;
        SemanticDigest::from_bytes(bytes)
    }

    fn estimation_folds() -> FoldAssignment {
        FoldAssignment::try_new(
            2,
            &[FoldRow::new(10, 0), FoldRow::new(20, 0), FoldRow::new(30, 0), FoldRow::new(40, 1)],
            &[99],
        )
        .unwrap()
    }

    fn clip(policy: NumericalPolicy, value: f64) -> NumericalDependencies {
        let dependency = NumericalDependency::try_new("clip", value).unwrap();
        NumericalDependencies::try_new(policy, &[dependency]).unwrap()
    }

    fn assemble(
        target: u8,
        design: u64,
        folds: &FoldAssignment,
        treatment: &str,
        provider: u64,
        numerical: &NumericalDependencies,
    ) -> NuisanceCacheKey {
        NuisanceCacheKey::new(
            TargetDependency::new(digest(target)),
            DesignDependency::new(DesignDependencyId::from_raw(design)),
            folds.clone(),
            TreatmentDependency::try_new(treatment, digest(6)).unwrap(),
            ProviderDependency::new(ProviderId::from_raw(provider), digest(5)),
            numerical.clone(),
        )
    }

    fn base_key() -> NuisanceCacheKey {
        let folds = estimation_folds();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        assemble(1, 7, &folds, "arm", 4, &numerical)
    }

    fn fold_zero_predictions() -> BlockedOutOfFoldPredictions {
        BlockedOutOfFoldPredictions::try_new(
            0,
            &[
                PredictedRow::try_new(10, 0.25).unwrap(),
                PredictedRow::try_new(20, -1.5).unwrap(),
                PredictedRow::try_new(30, 4.0).unwrap(),
            ],
        )
        .unwrap()
    }

    fn stored_bits() -> [u64; 3] {
        [4.0f64.to_bits(), 0.25f64.to_bits(), (-1.5f64).to_bits()]
    }

    fn cached() -> NuisanceCache {
        let mut cache = NuisanceCache::new();
        cache
            .store_certified(&base_key(), NuisanceRole::Treatment, &[fold_zero_predictions()])
            .unwrap();
        cache
    }

    fn value_bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn expect_miss(cache: &NuisanceCache, key: &NuisanceCacheKey, field: KeyField) {
        assert_eq!(base_key().mismatch(key), Some(field));
        let error = cache.restore_rows(key, NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::Miss);
        let restored =
            cache.restore_rows(&base_key(), NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap();
        assert_eq!(value_bits(&restored), stored_bits());
    }

    #[test]
    fn exact_key_is_a_hit() {
        let mut cache = cached();
        assert_eq!(base_key().mismatch(&base_key()), None);
        let fit = cache.certified(&base_key(), NuisanceRole::Treatment).unwrap();
        assert_eq!(fit.role(), NuisanceRole::Treatment);
        assert_eq!(fit.key().mismatch(&base_key()), None);
        cache
            .store_certified(&base_key(), NuisanceRole::Treatment, &[fold_zero_predictions()])
            .unwrap();
        let other = BlockedOutOfFoldPredictions::try_new(
            0,
            &[
                PredictedRow::try_new(10, 0.5).unwrap(),
                PredictedRow::try_new(20, -1.5).unwrap(),
                PredictedRow::try_new(30, 4.0).unwrap(),
            ],
        )
        .unwrap();
        let error =
            cache.store_certified(&base_key(), NuisanceRole::Treatment, &[other]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::ConflictingFit);
        let restored =
            cache.restore_rows(&base_key(), NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap();
        assert_eq!(value_bits(&restored), stored_bits());
        let missing = cache.certified(&base_key(), NuisanceRole::Outcome { level: 1 }).unwrap_err();
        assert_eq!(missing, NuisanceCacheError::UnknownRole);
    }

    #[test]
    fn changed_fold_is_a_miss() {
        let cache = cached();
        let folds = FoldAssignment::try_new(
            2,
            &[FoldRow::new(10, 1), FoldRow::new(20, 0), FoldRow::new(30, 0), FoldRow::new(40, 1)],
            &[99],
        )
        .unwrap();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        let key = assemble(1, 7, &folds, "arm", 4, &numerical);
        expect_miss(&cache, &key, KeyField::Folds);
    }

    #[test]
    fn changed_treatment_is_a_miss() {
        let cache = cached();
        let folds = estimation_folds();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        let key = assemble(1, 7, &folds, "dose", 4, &numerical);
        expect_miss(&cache, &key, KeyField::Treatment);
    }

    #[test]
    fn changed_provider_is_a_miss() {
        let cache = cached();
        let folds = estimation_folds();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        let key = assemble(1, 7, &folds, "arm", 8, &numerical);
        expect_miss(&cache, &key, KeyField::Provider);
    }

    #[test]
    fn changed_numerical_dependency_is_a_miss() {
        let cache = cached();
        let folds = estimation_folds();
        let changed_clip = clip(NumericalPolicy::strict(), 2.0);
        let key = assemble(1, 7, &folds, "arm", 4, &changed_clip);
        expect_miss(&cache, &key, KeyField::Numerical);

        let zero = PhysicalTolerance::new(0.0, 0.0).unwrap();
        let zero_tolerance = clip(NumericalPolicy::tolerance(zero), 1.0);
        let key = assemble(1, 7, &folds, "arm", 4, &zero_tolerance);
        expect_miss(&cache, &key, KeyField::Numerical);
    }

    #[test]
    fn changed_target_is_a_miss() {
        let cache = cached();
        let folds = estimation_folds();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        let key = assemble(2, 7, &folds, "arm", 4, &numerical);
        expect_miss(&cache, &key, KeyField::Target);
    }

    #[test]
    fn changed_design_is_a_miss() {
        let cache = cached();
        let folds = estimation_folds();
        let numerical = clip(NumericalPolicy::strict(), 1.0);
        let key = assemble(1, 9, &folds, "arm", 4, &numerical);
        expect_miss(&cache, &key, KeyField::Design);
    }

    #[test]
    fn out_of_fold_prediction_for_a_training_row_is_refused() {
        let mut cache = NuisanceCache::new();
        let key = base_key();
        let leaked =
            BlockedOutOfFoldPredictions::try_new(0, &[PredictedRow::try_new(40, 1.25).unwrap()])
                .unwrap();
        let error = cache.store_certified(&key, NuisanceRole::Treatment, &[leaked]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::TrainingLeakage { fold: 0, row_id: 40 });

        cache.store_certified(&key, NuisanceRole::Treatment, &[fold_zero_predictions()]).unwrap();
        let error = cache.restore_rows(&key, NuisanceRole::Treatment, 0, &[40]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::TrainingLeakage { fold: 0, row_id: 40 });
    }

    #[test]
    fn held_out_row_in_training_is_refused() {
        let error = FoldAssignment::try_new(2, &[FoldRow::new(10, 0), FoldRow::new(99, 1)], &[99])
            .unwrap_err();
        assert_eq!(error, NuisanceCacheError::HeldOutLeakage { row_id: 99 });

        let mut cache = NuisanceCache::new();
        let key = base_key();
        let leaked = BlockedOutOfFoldPredictions::try_new(
            0,
            &[PredictedRow::try_new(10, 0.25).unwrap(), PredictedRow::try_new(99, 0.5).unwrap()],
        )
        .unwrap();
        let error = cache.store_certified(&key, NuisanceRole::Treatment, &[leaked]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::HeldOutLeakage { row_id: 99 });

        cache.store_certified(&key, NuisanceRole::Treatment, &[fold_zero_predictions()]).unwrap();
        let error = cache.restore_rows(&key, NuisanceRole::Treatment, 0, &[99]).unwrap_err();
        assert_eq!(error, NuisanceCacheError::HeldOutLeakage { row_id: 99 });
    }

    #[test]
    fn restore_puts_each_value_on_its_stored_row() {
        let cache = cached();
        let restored =
            cache.restore_rows(&base_key(), NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap();
        assert_eq!(value_bits(&restored), stored_bits());
        let forward =
            cache.restore_rows(&base_key(), NuisanceRole::Treatment, 0, &[10, 20, 30]).unwrap();
        assert_eq!(
            value_bits(&forward),
            [0.25f64.to_bits(), (-1.5f64).to_bits(), 4.0f64.to_bits()]
        );
    }

    fn columns() -> RequestColumns {
        RequestColumns::try_new("outcome", "arm", &["z"]).unwrap()
    }

    fn source(byte: u8) -> CertifiedSource {
        CertifiedSource::new(SourceDependencyId::from_raw(3), digest(byte))
    }

    fn predictions_for(key: &NuisanceCacheKey) -> Vec<BlockedOutOfFoldPredictions> {
        let folds = key.folds();
        let mut blocks = Vec::new();
        for fold in 0..folds.fold_count() {
            let rows: Vec<PredictedRow> = folds
                .rows()
                .iter()
                .filter(|row| row.fold() == fold)
                .map(|row| {
                    let value = match row.row_id() {
                        10 => 0.25,
                        20 => -1.5,
                        30 => 4.0,
                        40 => 0.5,
                        _ => 0.125,
                    };
                    PredictedRow::try_new(row.row_id(), value).unwrap()
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            blocks.push(BlockedOutOfFoldPredictions::try_new(fold, &rows).unwrap());
        }
        blocks
    }

    fn both_blocks() -> Vec<BlockedOutOfFoldPredictions> {
        vec![
            fold_zero_predictions(),
            BlockedOutOfFoldPredictions::try_new(1, &[PredictedRow::try_new(40, 0.5).unwrap()])
                .unwrap(),
        ]
    }

    fn request(
        family: EstimatorFamily,
        source_byte: u8,
        key: NuisanceCacheKey,
    ) -> PreparedFamilyRequest {
        PreparedFamilyRequest::try_new(
            family,
            columns(),
            source(source_byte),
            key,
            &[NuisanceRole::Treatment],
        )
        .unwrap()
    }

    fn produce(
        calls: &std::cell::Cell<u64>,
    ) -> impl FnMut(
        &NuisanceCacheKey,
        NuisanceRole,
    ) -> Result<Vec<BlockedOutOfFoldPredictions>, NuisanceCacheError> {
        move |key, _role| {
            calls.set(calls.get().saturating_add(1));
            Ok(predictions_for(key))
        }
    }

    fn assert_receipt(execution: &BatchExecution, scans: u64, fits: u64, hits: u64) {
        assert_eq!(execution.receipt(), SubplanReceipt { scans, fits, hits });
    }

    #[test]
    fn shared_subplan_reuses_one_fit_and_matches_independent_execution() {
        let first = request(EstimatorFamily::Aipw, 1, base_key());
        let second = PreparedFamilyRequest::try_new(
            EstimatorFamily::Dml,
            RequestColumns::try_new("other_outcome", "arm", &["z", "w"]).unwrap(),
            source(1),
            base_key(),
            &[NuisanceRole::Treatment, NuisanceRole::Outcome { level: 0 }],
        )
        .unwrap();
        assert_ne!(first.columns(), second.columns());
        let compiled = compile_shared_subplans(&[first.clone(), second]).unwrap();
        assert_eq!(compiled.plans().len(), 1, "same certified source and key share one subplan");
        assert_eq!(compiled.plans()[0].members(), &[0, 1]);
        assert_eq!(
            compiled.plans()[0].roles(),
            &[NuisanceRole::Treatment, NuisanceRole::Outcome { level: 0 }]
        );

        let calls = std::cell::Cell::new(0u64);
        let mut cache = NuisanceCache::new();
        let shared = execute_compiled_subplans(&mut cache, &compiled, produce(&calls)).unwrap();
        assert_eq!(calls.get(), 2, "each role is fit once for the shared subplan");
        assert_receipt(&shared, 1, 2, 0);
        calls.set(0);
        let again = execute_compiled_subplans(&mut cache, &compiled, produce(&calls)).unwrap();
        assert_eq!(calls.get(), 0, "a second compatible execution restores the certified fit");
        assert_receipt(&again, 0, 0, 2);
        assert_eq!(shared.fits(), again.fits());

        let calls = std::cell::Cell::new(0u64);
        let requests = [first, request(EstimatorFamily::Aipw, 1, base_key())];
        let independent = super::execute_independently(&requests, produce(&calls)).unwrap();
        assert_eq!(calls.get(), 2);
        assert_receipt(&independent, 2, 2, 0);
        assert_eq!(&independent.fits()[0], &shared.fits()[0]);
        assert_eq!(&independent.fits()[1], &shared.fits()[0]);
    }

    #[test]
    fn matching_columns_miss_when_a_key_field_or_source_changes() {
        let base = request(EstimatorFamily::Aipw, 1, base_key());
        let folds = estimation_folds();
        let strict = clip(NumericalPolicy::strict(), 1.0);
        let changed_folds = FoldAssignment::try_new(
            2,
            &[FoldRow::new(10, 1), FoldRow::new(20, 0), FoldRow::new(30, 0), FoldRow::new(40, 1)],
            &[99],
        )
        .unwrap();
        let changed = [
            (assemble(2, 7, &folds, "arm", 4, &strict), KeyField::Target),
            (assemble(1, 9, &folds, "arm", 4, &strict), KeyField::Design),
            (assemble(1, 7, &changed_folds, "arm", 4, &strict), KeyField::Folds),
            (assemble(1, 7, &folds, "dose", 4, &strict), KeyField::Treatment),
            (assemble(1, 7, &folds, "arm", 8, &strict), KeyField::Provider),
            (
                assemble(1, 7, &folds, "arm", 4, &clip(NumericalPolicy::strict(), 2.0)),
                KeyField::Numerical,
            ),
        ];
        for (key, field) in changed {
            let other = request(EstimatorFamily::Aipw, 1, key.clone());
            assert_eq!(base.columns(), other.columns());
            assert_eq!(base_key().mismatch(&key), Some(field));
            let compiled = compile_shared_subplans(&[base.clone(), other]).unwrap();
            assert_eq!(compiled.plans().len(), 2, "{field:?} is not shared");
            let calls = std::cell::Cell::new(0u64);
            let mut cache = NuisanceCache::new();
            let execution =
                execute_compiled_subplans(&mut cache, &compiled, produce(&calls)).unwrap();
            assert_eq!(calls.get(), 2);
            assert_receipt(&execution, 1, 2, 0);
            assert!(cache.certified(&key, NuisanceRole::Treatment).is_ok());
        }

        let other_source = request(EstimatorFamily::Aipw, 2, base_key());
        assert_eq!(base.columns(), other_source.columns());
        let compiled = compile_shared_subplans(&[base, other_source]).unwrap();
        assert_eq!(compiled.plans().len(), 2, "a different certified source is not shared");
        let calls = std::cell::Cell::new(0u64);
        let mut cache = NuisanceCache::new();
        let execution = execute_compiled_subplans(&mut cache, &compiled, produce(&calls)).unwrap();
        assert_eq!(calls.get(), 2, "the second source does not reuse the first source's fit");
        assert_receipt(&execution, 2, 2, 0);
    }

    #[test]
    fn family_executor_refuses_another_family_and_keeps_leakage() {
        let prepared = request(EstimatorFamily::Dml, 1, base_key());
        let error = execute_family_requests(
            EstimatorFamily::Aipw,
            &mut NuisanceCache::new(),
            &[prepared],
            |_key, _role| Ok(both_blocks()),
        )
        .unwrap_err();
        assert_eq!(error, NuisanceCacheError::WrongFamily { request: "dml", expected: "aipw" });

        let leaked = request(EstimatorFamily::Aipw, 1, base_key());
        let error = execute_family_requests(
            EstimatorFamily::Aipw,
            &mut NuisanceCache::new(),
            &[leaked],
            |_key, _role| {
                Ok(vec![
                    BlockedOutOfFoldPredictions::try_new(
                        0,
                        &[PredictedRow::try_new(40, 1.25).unwrap()],
                    )
                    .unwrap(),
                ])
            },
        )
        .unwrap_err();
        assert_eq!(error, NuisanceCacheError::TrainingLeakage { fold: 0, row_id: 40 });
    }

    fn parallel_context() -> ExecutionContext {
        let mut ctx = ExecutionContext::for_tests(11)
            .with_admission(AdmissionPool::fixed(2, u64::MAX));
        ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(2).unwrap());
        ctx
    }

    fn derived(
        outcome: &[f64],
        weights: Option<&[f64]>,
        seed: u64,
        folds: &FoldAssignment,
    ) -> NuisanceCacheKey {
        let treatment = [0.0f64, 1.0, 0.0, 1.0];
        let design = [1.0f64, 1.0, 1.0, 1.0, 0.2, 0.4, 0.6, 0.8];
        let options = [0.25f64];
        let columns =
            NuisanceColumns::try_new(outcome, &treatment, &design, &options, weights).unwrap();
        let inputs =
            NuisanceInputs::try_new(columns, seed, folds, "arm", NumericalPolicy::strict())
                .unwrap();
        NuisanceCacheKey::from_inputs(&inputs).unwrap()
    }

    fn expect_indexed_miss(base: &NuisanceCacheKey, changed: &NuisanceCacheKey, field: KeyField) {
        assert_eq!(base.mismatch(changed), Some(field));
        let mut cache = NuisanceCache::new();
        cache
            .store_certified(base, NuisanceRole::Treatment, &[fold_zero_predictions()])
            .unwrap();
        let ctx = parallel_context();
        map_indexed(&ctx, &cache, 2, |_index, _inner, cache| {
            assert!(!super::certification_requested());
            let error = cache
                .restore_rows(changed, NuisanceRole::Treatment, 0, &[30, 10, 20])
                .unwrap_err();
            assert_eq!(error, NuisanceCacheError::Miss);
            Ok::<(), NuisanceCacheError>(())
        })
        .unwrap();
        let restored =
            cache.restore_rows(base, NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap();
        assert_eq!(value_bits(&restored), stored_bits());
    }

    #[test]
    fn y_change_misses_under_map_indexed() {
        let folds = estimation_folds();
        let outcome = [1.0f64, 2.0, 3.0, 4.0];
        let changed_outcome = [1.0f64, 2.0, 3.0, 5.0];
        let base = derived(&outcome, None, 11, &folds);
        let changed = derived(&changed_outcome, None, 11, &folds);
        assert_ne!(base.outcome_digest(), changed.outcome_digest());
        assert_eq!(base.weight_digest(), changed.weight_digest());
        assert_eq!(base.seed(), changed.seed());
        expect_indexed_miss(&base, &changed, KeyField::Outcome);
    }

    #[test]
    fn weights_change_misses_under_map_indexed() {
        let folds = estimation_folds();
        let outcome = [1.0f64, 2.0, 3.0, 4.0];
        let weights = [1.0f64, 1.0, 1.0, 2.0];
        let base = derived(&outcome, None, 11, &folds);
        let changed = derived(&outcome, Some(&weights), 11, &folds);
        assert_eq!(base.outcome_digest(), changed.outcome_digest());
        assert_ne!(base.weight_digest(), changed.weight_digest());
        assert_eq!(base.seed(), changed.seed());
        expect_indexed_miss(&base, &changed, KeyField::Weights);
    }

    #[test]
    fn seed_change_misses_under_map_indexed() {
        let folds = estimation_folds();
        let outcome = [1.0f64, 2.0, 3.0, 4.0];
        let base = derived(&outcome, None, 11, &folds);
        let changed = derived(&outcome, None, 29, &folds);
        assert_eq!(base.outcome_digest(), changed.outcome_digest());
        assert_eq!(base.weight_digest(), changed.weight_digest());
        assert_ne!(base.seed(), changed.seed());
        expect_indexed_miss(&base, &changed, KeyField::Seed);
    }

    #[test]
    fn parallel_worker_sees_the_cache_it_was_passed() {
        let folds = estimation_folds();
        let outcome = [1.0f64, 2.0, 3.0, 4.0];
        let key = derived(&outcome, None, 11, &folds);
        let mut passed = NuisanceCache::new();
        passed
            .store_certified(&key, NuisanceRole::Treatment, &[fold_zero_predictions()])
            .unwrap();
        let decoy_block = BlockedOutOfFoldPredictions::try_new(
            0,
            &[
                PredictedRow::try_new(10, 9.0).unwrap(),
                PredictedRow::try_new(20, 8.0).unwrap(),
                PredictedRow::try_new(30, 7.0).unwrap(),
            ],
        )
        .unwrap();
        let mut decoy = NuisanceCache::new();
        decoy.store_certified(&key, NuisanceRole::Treatment, &[decoy_block]).unwrap();
        let ctx = parallel_context();
        let (seen, receipt) = super::with_certified_cache(&mut decoy, &key, || {
            map_indexed(&ctx, &passed, 2, |_index, inner, cache| {
                assert!(!super::certification_requested());
                assert_eq!(inner.parallelism.max_threads.get(), 1u32);
                cache.restore_rows(&key, NuisanceRole::Treatment, 0, &[30, 10, 20])
            })
            .unwrap()
        });
        assert_eq!(receipt, SubplanReceipt::default());
        assert_eq!(value_bits(&seen[0]), stored_bits());
        assert_eq!(value_bits(&seen[1]), stored_bits());
        let decoy_values =
            decoy.restore_rows(&key, NuisanceRole::Treatment, 0, &[30, 10, 20]).unwrap();
        assert_ne!(value_bits(&decoy_values), stored_bits());
    }
}
