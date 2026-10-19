//! Certified nuisance fits and blocked out-of-fold predictions.
//!
//! AIPW, DML, and policy estimation reuse a fit only when the cache key matches
//! in every field: target, design, folds, treatment, provider, and numerical
//! dependencies. A changed field is a miss. The cache does not return a fit
//! stored under another key.
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
use std::collections::BTreeMap;

use antecedent_core::{DesignDependencyId, NumericalPolicy, ProviderId, SemanticDigest};
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

    /// The fit stored under this key and role.
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
    ) -> Result<CertifiedNuisanceFit, NuisanceCacheError> {
        let roles = self.entries.get(key).ok_or(NuisanceCacheError::Miss)?;
        roles.get(&role).cloned().ok_or(NuisanceCacheError::UnknownRole)
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

/// Cache key. A hit requires every field to match.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NuisanceCacheKey {
    target: TargetDependency,
    design: DesignDependency,
    folds: FoldAssignment,
    treatment: TreatmentDependency,
    provider: ProviderDependency,
    numerical: NumericalDependencies,
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
        Self { target, design, folds, treatment, provider, numerical }
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

    /// First field that differs, in [`KeyField`] order.
    ///
    /// `None` means every field matches.
    #[must_use]
    pub fn mismatch(&self, other: &Self) -> Option<KeyField> {
        if self.target != other.target {
            return Some(KeyField::Target);
        }
        if self.design != other.design {
            return Some(KeyField::Design);
        }
        if self.folds != other.folds {
            return Some(KeyField::Folds);
        }
        if self.treatment != other.treatment {
            return Some(KeyField::Treatment);
        }
        if self.provider != other.provider {
            return Some(KeyField::Provider);
        }
        if self.numerical != other.numerical {
            return Some(KeyField::Numerical);
        }
        None
    }
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
    for row in folds.rows() {
        if row.fold() != evaluation_fold && folds.is_held_out(row.row_id()) {
            return Err(NuisanceCacheError::HeldOutLeakage { row_id: row.row_id() });
        }
    }
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
    use antecedent_core::{NumericalPolicy, PhysicalTolerance};

    use super::{
        BlockedOutOfFoldPredictions, DesignDependency, DesignDependencyId, FoldAssignment, FoldRow,
        KeyField, NuisanceCache, NuisanceCacheError, NuisanceCacheKey, NuisanceRole,
        NumericalDependencies, NumericalDependency, PredictedRow, ProviderDependency, ProviderId,
        SemanticDigest, TargetDependency, TreatmentDependency,
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
}
