//! Semantic dependency keys for reusable products.
//!
//! Each [`ReusableProduct`] names the dependency classes it is valid under:
//! target and identification, source snapshot, rows/history/folds, model/prior/provider,
//! intervention targets, joint alignment, inference, and numerical policy.
//! [`DependencyKey::seal`] writes every declared class and nothing else.
//! A missing declared class is a refusal. A class the product does not declare is a
//! refusal too, so an extra dependency is not dropped. Two different values of a
//! declared dependency do not seal to the same digest.
//!
//! A present class is complete. Target and identification are both fields of one
//! class, and model, prior, and provider are all fields of one class. Row ids are
//! `u64`. Numerical policy keeps strict distinct from a zero tolerance.
//!
//! This is the reuse key. It does not replace the score-reuse wire in
//! [`crate::identity`], including `batch_share`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, NumericalPolicy, SemanticDigest};

/// Canonical-writer tag for every sealed key.
pub const DEPENDENCY_KEY_TAG: &str = "antecedent.dependency_key.v1";

/// Encoding format mixed into every sealed key. Bump when the field rules change.
pub const DEPENDENCY_KEY_FORMAT: u16 = 1;

const CLASS_COUNT: usize = 8;

/// One dependency class a reusable product may declare.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DependencyClass {
    /// Full target digest and identification digest.
    TargetIdentification,
    /// Source snapshot the product was built from.
    SourceSnapshot,
    /// Selected rows, the row-aligned payload, history, and fold assignment.
    RowsHistoryFolds,
    /// Model, prior, and provider, each present.
    ModelPriorProvider,
    /// Intervention specification and the outcome targets it is applied to.
    InterventionTargets,
    /// Joint-law axis alignment.
    JointAlignment,
    /// Inference realization and binding.
    Inference,
    /// Strict or tolerance numerical policy.
    NumericalPolicy,
}

impl DependencyClass {
    /// Closed set, in the order a key writes them.
    pub const ALL: [Self; CLASS_COUNT] = [
        Self::TargetIdentification,
        Self::SourceSnapshot,
        Self::RowsHistoryFolds,
        Self::ModelPriorProvider,
        Self::InterventionTargets,
        Self::JointAlignment,
        Self::Inference,
        Self::NumericalPolicy,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TargetIdentification => "target_identification",
            Self::SourceSnapshot => "source_snapshot",
            Self::RowsHistoryFolds => "rows_history_folds",
            Self::ModelPriorProvider => "model_prior_provider",
            Self::InterventionTargets => "intervention_targets",
            Self::JointAlignment => "joint_alignment",
            Self::Inference => "inference",
            Self::NumericalPolicy => "numerical_policy",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::TargetIdentification => 0,
            Self::SourceSnapshot => 1,
            Self::RowsHistoryFolds => 2,
            Self::ModelPriorProvider => 3,
            Self::InterventionTargets => 4,
            Self::JointAlignment => 5,
            Self::Inference => 6,
            Self::NumericalPolicy => 7,
        }
    }
}

impl std::fmt::Display for DependencyClass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A cached product whose reuse key is the digest of its declared dependencies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ReusableProduct {
    /// Identification product. Target and identification only.
    Identification,
    /// Source snapshot reused as itself.
    SourceSnapshot,
    /// Row, history, and fold assignment reused across queries.
    RowHistoryFolds,
    /// Model, prior, and provider binding.
    ModelPriorProvider,
    /// Intervention specification and its outcome targets.
    InterventionTargets,
    /// Joint-law alignment reused across consumers that share axes.
    JointAlignment,
    /// Inference binding.
    Inference,
    /// Numerical policy. Separate from the scientific target.
    NumericalPolicy,
    /// Query-specific score table. Includes the target, and not joint alignment.
    ScoreTable,
    /// Design shared across queries. Snapshot, rows, and numerical policy only.
    BatchDesign,
    /// Frozen fit. No target and no intervention coordinates.
    FrozenFit,
    /// Provider result bound to the snapshot it read.
    ProviderReplay,
    /// Aligned joint law.
    JointLaw,
    /// Intervention evaluation, including a source snapshot when one was used.
    InterventionEvaluation,
}

impl ReusableProduct {
    /// Closed set. A new variant belongs here and in [`Self::declared`].
    pub const ALL: [Self; 14] = [
        Self::Identification,
        Self::SourceSnapshot,
        Self::RowHistoryFolds,
        Self::ModelPriorProvider,
        Self::InterventionTargets,
        Self::JointAlignment,
        Self::Inference,
        Self::NumericalPolicy,
        Self::ScoreTable,
        Self::BatchDesign,
        Self::FrozenFit,
        Self::ProviderReplay,
        Self::JointLaw,
        Self::InterventionEvaluation,
    ];

    /// Stable `snake_case` name. Part of the sealed digest.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identification => "identification",
            Self::SourceSnapshot => "source_snapshot",
            Self::RowHistoryFolds => "row_history_folds",
            Self::ModelPriorProvider => "model_prior_provider",
            Self::InterventionTargets => "intervention_targets",
            Self::JointAlignment => "joint_alignment",
            Self::Inference => "inference",
            Self::NumericalPolicy => "numerical_policy",
            Self::ScoreTable => "score_table",
            Self::BatchDesign => "batch_design",
            Self::FrozenFit => "frozen_fit",
            Self::ProviderReplay => "provider_replay",
            Self::JointLaw => "joint_law",
            Self::InterventionEvaluation => "intervention_evaluation",
        }
    }

    /// Classes this product is valid under, in [`DependencyClass::ALL`] order.
    #[must_use]
    pub const fn declared(self) -> &'static [DependencyClass] {
        match self {
            Self::Identification => &[DependencyClass::TargetIdentification],
            Self::SourceSnapshot => &[DependencyClass::SourceSnapshot],
            Self::RowHistoryFolds => &[DependencyClass::RowsHistoryFolds],
            Self::ModelPriorProvider => &[DependencyClass::ModelPriorProvider],
            Self::InterventionTargets => &[DependencyClass::InterventionTargets],
            Self::JointAlignment => &[DependencyClass::JointAlignment],
            Self::Inference => &[DependencyClass::Inference],
            Self::NumericalPolicy => &[DependencyClass::NumericalPolicy],
            Self::ScoreTable => &[
                DependencyClass::TargetIdentification,
                DependencyClass::SourceSnapshot,
                DependencyClass::RowsHistoryFolds,
                DependencyClass::ModelPriorProvider,
                DependencyClass::InterventionTargets,
                DependencyClass::Inference,
                DependencyClass::NumericalPolicy,
            ],
            Self::BatchDesign => &[
                DependencyClass::SourceSnapshot,
                DependencyClass::RowsHistoryFolds,
                DependencyClass::NumericalPolicy,
            ],
            Self::FrozenFit => &[
                DependencyClass::SourceSnapshot,
                DependencyClass::RowsHistoryFolds,
                DependencyClass::ModelPriorProvider,
                DependencyClass::Inference,
                DependencyClass::NumericalPolicy,
            ],
            Self::ProviderReplay => &[
                DependencyClass::SourceSnapshot,
                DependencyClass::ModelPriorProvider,
                DependencyClass::Inference,
                DependencyClass::NumericalPolicy,
            ],
            Self::JointLaw => &[
                DependencyClass::TargetIdentification,
                DependencyClass::JointAlignment,
                DependencyClass::Inference,
                DependencyClass::NumericalPolicy,
            ],
            Self::InterventionEvaluation => &DependencyClass::ALL,
        }
    }

    /// Whether `class` is in [`Self::declared`].
    #[must_use]
    pub const fn declares(self, class: DependencyClass) -> bool {
        let declared = self.declared();
        let mut index = 0;
        while index < declared.len() {
            if declared[index].index() == class.index() {
                return true;
            }
            index += 1;
        }
        false
    }
}

impl std::fmt::Display for ReusableProduct {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why a dependency key was not sealed.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DependencyKeyError {
    /// A class in [`ReusableProduct::declared`] was not bound.
    #[error("reusable product {product} omits declared dependency {class}")]
    Omitted {
        /// Product that was being sealed.
        product: ReusableProduct,
        /// Class that was required and absent.
        class: DependencyClass,
    },
    /// A bound class is not in [`ReusableProduct::declared`].
    #[error("reusable product {product} was given undeclared dependency {class}")]
    Undeclared {
        /// Product that was being sealed.
        product: ReusableProduct,
        /// Class that is not a dependency of `product`.
        class: DependencyClass,
    },
    /// The same class was bound twice. The first value is not dropped.
    #[error("reusable product {product} was given dependency {class} twice")]
    Duplicate {
        /// Product that was being bound.
        product: ReusableProduct,
        /// Class of the second value.
        class: DependencyClass,
    },
    /// Row ids and fold ids are not the same length, so they are not one assignment.
    #[error("row ids ({rows}) and fold ids ({folds}) have different lengths")]
    RowFoldLengthMismatch {
        /// Number of row ids.
        rows: u64,
        /// Number of fold ids.
        folds: u64,
    },
}

/// Selected rows, their payload digest, history, and folds.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RowHistoryFoldBody {
    row_ids: Vec<u64>,
    row_payload: SemanticDigest,
    history: SemanticDigest,
    fold_ids: Vec<u32>,
    n_folds: u32,
}

/// One bound dependency. Every field of the class is present.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyRecord {
    body: RecordBody,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecordBody {
    TargetIdentification { target: SemanticDigest, identification: SemanticDigest },
    SourceSnapshot { snapshot: SemanticDigest },
    RowsHistoryFolds(Box<RowHistoryFoldBody>),
    ModelPriorProvider { model: SemanticDigest, prior: SemanticDigest, provider: SemanticDigest },
    InterventionTargets { interventions: SemanticDigest, targets: SemanticDigest },
    JointAlignment { alignment: SemanticDigest },
    Inference { inference: SemanticDigest },
    NumericalPolicy { policy: NumericalPolicy },
}

impl DependencyRecord {
    /// Target and identification. Both digests are required.
    #[must_use]
    pub const fn target_identification(
        target: SemanticDigest,
        identification: SemanticDigest,
    ) -> Self {
        Self { body: RecordBody::TargetIdentification { target, identification } }
    }

    /// Source snapshot digest.
    #[must_use]
    pub const fn source_snapshot(snapshot: SemanticDigest) -> Self {
        Self { body: RecordBody::SourceSnapshot { snapshot } }
    }

    /// Rows, the digest of the row-aligned payload, history, and folds.
    ///
    /// `row_ids` stay `u64`. Order is part of the assignment. `row_payload` is the
    /// digest of whatever is aligned with those rows (values, masks, or weights).
    ///
    /// # Errors
    ///
    /// [`DependencyKeyError::RowFoldLengthMismatch`] when `row_ids` and `fold_ids`
    /// differ in length.
    pub fn rows_history_folds(
        row_ids: &[u64],
        row_payload: SemanticDigest,
        history: SemanticDigest,
        fold_ids: &[u32],
        n_folds: u32,
    ) -> Result<Self, DependencyKeyError> {
        if row_ids.len() != fold_ids.len() {
            return Err(DependencyKeyError::RowFoldLengthMismatch {
                rows: count_word(row_ids.len()),
                folds: count_word(fold_ids.len()),
            });
        }
        Ok(Self {
            body: RecordBody::RowsHistoryFolds(Box::new(RowHistoryFoldBody {
                row_ids: row_ids.to_vec(),
                row_payload,
                history,
                fold_ids: fold_ids.to_vec(),
                n_folds,
            })),
        })
    }

    /// Model, prior, and provider. Omitting one of the three is not a value of this class.
    #[must_use]
    pub const fn model_prior_provider(
        model: SemanticDigest,
        prior: SemanticDigest,
        provider: SemanticDigest,
    ) -> Self {
        Self { body: RecordBody::ModelPriorProvider { model, prior, provider } }
    }

    /// Intervention specification and the outcome targets.
    #[must_use]
    pub const fn intervention_targets(
        interventions: SemanticDigest,
        targets: SemanticDigest,
    ) -> Self {
        Self { body: RecordBody::InterventionTargets { interventions, targets } }
    }

    /// Joint-alignment digest.
    #[must_use]
    pub const fn joint_alignment(alignment: SemanticDigest) -> Self {
        Self { body: RecordBody::JointAlignment { alignment } }
    }

    /// Inference digest.
    #[must_use]
    pub const fn inference(inference: SemanticDigest) -> Self {
        Self { body: RecordBody::Inference { inference } }
    }

    /// Numerical policy. Strict is not a zero tolerance.
    #[must_use]
    pub const fn numerical_policy(policy: NumericalPolicy) -> Self {
        Self { body: RecordBody::NumericalPolicy { policy } }
    }

    /// Class this record binds.
    #[must_use]
    pub fn class(&self) -> DependencyClass {
        match &self.body {
            RecordBody::TargetIdentification { .. } => DependencyClass::TargetIdentification,
            RecordBody::SourceSnapshot { .. } => DependencyClass::SourceSnapshot,
            RecordBody::RowsHistoryFolds(_) => DependencyClass::RowsHistoryFolds,
            RecordBody::ModelPriorProvider { .. } => DependencyClass::ModelPriorProvider,
            RecordBody::InterventionTargets { .. } => DependencyClass::InterventionTargets,
            RecordBody::JointAlignment { .. } => DependencyClass::JointAlignment,
            RecordBody::Inference { .. } => DependencyClass::Inference,
            RecordBody::NumericalPolicy { .. } => DependencyClass::NumericalPolicy,
        }
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        match &self.body {
            RecordBody::TargetIdentification { target, identification } => {
                writer.text("target").bytes(target.as_bytes());
                writer.text("identification").bytes(identification.as_bytes());
            }
            RecordBody::SourceSnapshot { snapshot } => {
                writer.text("snapshot").bytes(snapshot.as_bytes());
            }
            RecordBody::RowsHistoryFolds(body) => write_rows(writer, body),
            RecordBody::ModelPriorProvider { model, prior, provider } => {
                writer.text("model").bytes(model.as_bytes());
                writer.text("prior").bytes(prior.as_bytes());
                writer.text("provider").bytes(provider.as_bytes());
            }
            RecordBody::InterventionTargets { interventions, targets } => {
                writer.text("interventions").bytes(interventions.as_bytes());
                writer.text("targets").bytes(targets.as_bytes());
            }
            RecordBody::JointAlignment { alignment } => {
                writer.text("alignment").bytes(alignment.as_bytes());
            }
            RecordBody::Inference { inference } => {
                writer.text("inference").bytes(inference.as_bytes());
            }
            RecordBody::NumericalPolicy { policy } => write_policy(writer, *policy),
        }
    }
}

fn write_rows(writer: &mut CanonicalWriter, body: &RowHistoryFoldBody) {
    writer.text("row_ids").word(count_word(body.row_ids.len()));
    for id in &body.row_ids {
        writer.word(*id);
    }
    writer.text("row_payload").bytes(body.row_payload.as_bytes());
    writer.text("history").bytes(body.history.as_bytes());
    writer.text("fold_ids").word(count_word(body.fold_ids.len()));
    for id in &body.fold_ids {
        writer.word(u64::from(*id));
    }
    writer.text("n_folds").word(u64::from(body.n_folds));
}

fn write_policy(writer: &mut CanonicalWriter, policy: NumericalPolicy) {
    match policy.declared_tolerance() {
        None => {
            writer.text("policy").word(1);
        }
        Some(tolerance) => {
            writer
                .text("policy")
                .word(2)
                .word(tolerance.absolute().to_bits())
                .word(tolerance.relative().to_bits());
        }
    }
}

fn count_word(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Unsealed bindings for one [`ReusableProduct`].
#[derive(Clone, Debug)]
pub struct DependencyKey {
    product: ReusableProduct,
    slots: [Option<DependencyRecord>; CLASS_COUNT],
}

impl DependencyKey {
    /// Start a key for `product` with no dependencies bound.
    #[must_use]
    pub fn new(product: ReusableProduct) -> Self {
        Self { product, slots: [None, None, None, None, None, None, None, None] }
    }

    /// Product these bindings are for.
    #[must_use]
    pub const fn product(&self) -> ReusableProduct {
        self.product
    }

    /// Bind `record`. A second value for the same class is refused.
    ///
    /// # Errors
    ///
    /// [`DependencyKeyError::Duplicate`] when `record`'s class is already bound.
    pub fn bind(mut self, record: DependencyRecord) -> Result<Self, DependencyKeyError> {
        let class = record.class();
        let slot = &mut self.slots[class.index()];
        if slot.is_some() {
            return Err(DependencyKeyError::Duplicate { product: self.product, class });
        }
        *slot = Some(record);
        Ok(self)
    }

    /// Seal the key.
    ///
    /// # Errors
    ///
    /// [`DependencyKeyError::Omitted`] when a declared class is unbound.
    /// [`DependencyKeyError::Undeclared`] when a bound class is not declared.
    /// Declared omissions are reported before undeclared extras.
    #[must_use = "an omitted or undeclared dependency is a refusal, not a key"]
    pub fn seal(self) -> Result<SealedDependencyKey, DependencyKeyError> {
        for class in DependencyClass::ALL {
            if self.product.declares(class) && self.slots[class.index()].is_none() {
                return Err(DependencyKeyError::Omitted { product: self.product, class });
            }
        }
        for class in DependencyClass::ALL {
            if self.slots[class.index()].is_some() && !self.product.declares(class) {
                return Err(DependencyKeyError::Undeclared { product: self.product, class });
            }
        }
        let mut writer = CanonicalWriter::new(DEPENDENCY_KEY_TAG);
        writer.word(u64::from(DEPENDENCY_KEY_FORMAT)).text(self.product.as_str());
        let declared = self.product.declared();
        writer.word(count_word(declared.len()));
        for class in declared {
            writer.text(class.as_str());
            match &self.slots[class.index()] {
                Some(record) => record.write(&mut writer),
                None => {
                    return Err(DependencyKeyError::Omitted {
                        product: self.product,
                        class: *class,
                    });
                }
            }
        }
        Ok(SealedDependencyKey {
            product: self.product,
            digest: SemanticDigest::from_bytes(writer.finish()),
        })
    }
}

/// Digest of one reusable product under its declared dependencies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SealedDependencyKey {
    product: ReusableProduct,
    digest: SemanticDigest,
}

impl SealedDependencyKey {
    /// Product this key is for.
    #[must_use]
    pub const fn product(self) -> ReusableProduct {
        self.product
    }

    /// Domain-separated digest. Equality is identity under [`DEPENDENCY_KEY_TAG`].
    #[must_use]
    pub const fn digest(self) -> SemanticDigest {
        self.digest
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{NumericalPolicy, PhysicalTolerance, SemanticDigest};

    use super::{
        DependencyClass, DependencyKey, DependencyKeyError, DependencyRecord, ReusableProduct,
    };

    fn digest(fill: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([fill; 32])
    }

    fn rows(
        row_ids: &[u64],
        payload: u8,
        history: u8,
        fold_ids: &[u32],
        n_folds: u32,
    ) -> DependencyRecord {
        DependencyRecord::rows_history_folds(
            row_ids,
            digest(payload),
            digest(history),
            fold_ids,
            n_folds,
        )
        .expect("aligned rows")
    }

    fn baseline(class: DependencyClass) -> DependencyRecord {
        match class {
            DependencyClass::TargetIdentification => {
                DependencyRecord::target_identification(digest(0x11), digest(0x12))
            }
            DependencyClass::SourceSnapshot => DependencyRecord::source_snapshot(digest(0x21)),
            DependencyClass::RowsHistoryFolds => rows(&[1, 2, 3], 0x31, 0x32, &[0, 1, 0], 2),
            DependencyClass::ModelPriorProvider => {
                DependencyRecord::model_prior_provider(digest(0x41), digest(0x42), digest(0x43))
            }
            DependencyClass::InterventionTargets => {
                DependencyRecord::intervention_targets(digest(0x51), digest(0x52))
            }
            DependencyClass::JointAlignment => DependencyRecord::joint_alignment(digest(0x61)),
            DependencyClass::Inference => DependencyRecord::inference(digest(0x71)),
            DependencyClass::NumericalPolicy => {
                DependencyRecord::numerical_policy(NumericalPolicy::strict())
            }
        }
    }

    fn alterations(class: DependencyClass) -> Vec<DependencyRecord> {
        match class {
            DependencyClass::TargetIdentification => vec![
                DependencyRecord::target_identification(digest(0x19), digest(0x12)),
                DependencyRecord::target_identification(digest(0x11), digest(0x19)),
                DependencyRecord::target_identification(digest(0x12), digest(0x11)),
            ],
            DependencyClass::SourceSnapshot => {
                vec![DependencyRecord::source_snapshot(digest(0x29))]
            }
            DependencyClass::RowsHistoryFolds => vec![
                rows(&[1, 2, 4], 0x31, 0x32, &[0, 1, 0], 2),
                rows(&[3, 2, 1], 0x31, 0x32, &[0, 1, 0], 2),
                rows(&[1, 2, 3, 4], 0x31, 0x32, &[0, 1, 0, 1], 2),
                rows(&[1_u64 << 32], 0x31, 0x32, &[0], 1),
                rows(&[(1_u64 << 32) | 1], 0x31, 0x32, &[0], 1),
                rows(&[1, 2, 3], 0x39, 0x32, &[0, 1, 0], 2),
                rows(&[1, 2, 3], 0x31, 0x39, &[0, 1, 0], 2),
                rows(&[1, 2, 3], 0x31, 0x32, &[0, 1, 1], 2),
                rows(&[1, 2, 3], 0x31, 0x32, &[0, 1, 0], 3),
                rows(&[], 0x31, 0x32, &[], 0),
            ],
            DependencyClass::ModelPriorProvider => vec![
                DependencyRecord::model_prior_provider(digest(0x49), digest(0x42), digest(0x43)),
                DependencyRecord::model_prior_provider(digest(0x41), digest(0x49), digest(0x43)),
                DependencyRecord::model_prior_provider(digest(0x41), digest(0x42), digest(0x49)),
                DependencyRecord::model_prior_provider(digest(0x43), digest(0x42), digest(0x41)),
            ],
            DependencyClass::InterventionTargets => vec![
                DependencyRecord::intervention_targets(digest(0x59), digest(0x52)),
                DependencyRecord::intervention_targets(digest(0x51), digest(0x59)),
                DependencyRecord::intervention_targets(digest(0x52), digest(0x51)),
            ],
            DependencyClass::JointAlignment => {
                vec![DependencyRecord::joint_alignment(digest(0x69))]
            }
            DependencyClass::Inference => vec![DependencyRecord::inference(digest(0x79))],
            DependencyClass::NumericalPolicy => {
                let zero = PhysicalTolerance::new(0.0, 0.0).expect("zero tolerance");
                let shifted = PhysicalTolerance::new(1.0, 0.0).expect("shifted tolerance");
                vec![
                    DependencyRecord::numerical_policy(NumericalPolicy::tolerance(zero)),
                    DependencyRecord::numerical_policy(NumericalPolicy::tolerance(shifted)),
                ]
            }
        }
    }

    fn seal_declared(product: ReusableProduct) -> SemanticDigest {
        let mut key = DependencyKey::new(product);
        for class in product.declared() {
            key = key.bind(baseline(*class)).expect("first bind");
        }
        key.seal().expect("complete declaration").digest()
    }

    #[test]
    fn catalog_names_are_unique_and_declared_in_class_order() {
        let mut product_names = std::collections::BTreeSet::new();
        let mut class_names = std::collections::BTreeSet::new();
        assert_eq!(DependencyClass::ALL.len(), 8);
        assert_eq!(ReusableProduct::ALL.len(), 14);
        for (index, class) in DependencyClass::ALL.iter().enumerate() {
            assert!(class_names.insert(class.as_str()));
            assert_eq!(class.index(), index);
        }
        for product in ReusableProduct::ALL {
            assert!(product_names.insert(product.as_str()));
            assert!(!product.declared().is_empty());
            let mut previous = None;
            for class in product.declared() {
                assert!(product.declares(*class));
                if let Some(before) = previous {
                    assert!(before < class.index());
                }
                previous = Some(class.index());
            }
            for class in DependencyClass::ALL {
                assert_eq!(product.declares(class), product.declared().contains(&class));
            }
        }
    }

    #[test]
    fn every_variant_is_in_the_closed_sets() {
        for class in DependencyClass::ALL {
            match class {
                DependencyClass::TargetIdentification
                | DependencyClass::SourceSnapshot
                | DependencyClass::RowsHistoryFolds
                | DependencyClass::ModelPriorProvider
                | DependencyClass::InterventionTargets
                | DependencyClass::JointAlignment
                | DependencyClass::Inference
                | DependencyClass::NumericalPolicy => {}
            }
        }
        for product in ReusableProduct::ALL {
            match product {
                ReusableProduct::Identification
                | ReusableProduct::SourceSnapshot
                | ReusableProduct::RowHistoryFolds
                | ReusableProduct::ModelPriorProvider
                | ReusableProduct::InterventionTargets
                | ReusableProduct::JointAlignment
                | ReusableProduct::Inference
                | ReusableProduct::NumericalPolicy
                | ReusableProduct::ScoreTable
                | ReusableProduct::BatchDesign
                | ReusableProduct::FrozenFit
                | ReusableProduct::ProviderReplay
                | ReusableProduct::JointLaw
                | ReusableProduct::InterventionEvaluation => {}
            }
        }
    }

    #[test]
    fn omitted_declared_dependency_refuses_for_every_product() {
        for product in ReusableProduct::ALL {
            let sealed = seal_declared(product);
            assert_eq!(sealed.as_bytes().len(), 32);
            for omitted in product.declared() {
                let mut key = DependencyKey::new(product);
                for class in product.declared() {
                    if class == omitted {
                        continue;
                    }
                    key = key.bind(baseline(*class)).expect("bind");
                }
                let error = key.seal().expect_err("omission is a refusal");
                assert_eq!(
                    error,
                    DependencyKeyError::Omitted { product, class: *omitted },
                    "{product} omitted {omitted}"
                );
            }
        }
    }

    #[test]
    fn different_dependencies_do_not_share_a_key() {
        for product in ReusableProduct::ALL {
            let base = seal_declared(product);
            for class in product.declared() {
                for variant in alterations(*class) {
                    let mut key = DependencyKey::new(product);
                    for declared in product.declared() {
                        let record =
                            if declared == class { variant.clone() } else { baseline(*declared) };
                        key = key.bind(record).expect("bind");
                    }
                    let digest = key.seal().expect("variant seals").digest();
                    assert_ne!(base, digest, "{product} collided on {class}");
                }
            }
        }
    }

    #[test]
    fn row_ids_beyond_u32_do_not_collapse() {
        let product = ReusableProduct::RowHistoryFolds;
        let seal_rows = |ids: &[u64]| {
            DependencyKey::new(product)
                .bind(rows(ids, 0x31, 0x32, &[0], 1))
                .expect("bind")
                .seal()
                .expect("seal")
                .digest()
        };
        let wide = seal_rows(&[1_u64 << 32]);
        assert_ne!(wide, seal_rows(&[(1_u64 << 32) | 1]));
        assert_ne!(wide, seal_rows(&[0]));
    }

    #[test]
    fn strict_numerical_policy_is_not_a_zero_tolerance() {
        let product = ReusableProduct::NumericalPolicy;
        let strict = seal_declared(product);
        let zero = PhysicalTolerance::new(0.0, 0.0).expect("zero");
        let negative_zero = PhysicalTolerance::new(-0.0, 0.0).expect("negative zero");
        let zero_key = DependencyKey::new(product)
            .bind(DependencyRecord::numerical_policy(NumericalPolicy::tolerance(zero)))
            .unwrap()
            .seal()
            .unwrap()
            .digest();
        let negative_key = DependencyKey::new(product)
            .bind(DependencyRecord::numerical_policy(NumericalPolicy::tolerance(negative_zero)))
            .unwrap()
            .seal()
            .unwrap()
            .digest();
        assert_ne!(strict, zero_key);
        assert_eq!(zero_key, negative_key);
    }

    #[test]
    fn undeclared_dependency_refuses_instead_of_being_dropped() {
        for product in ReusableProduct::ALL {
            let Some(extra) =
                DependencyClass::ALL.into_iter().find(|class| !product.declares(*class))
            else {
                continue;
            };
            let mut key = DependencyKey::new(product);
            for class in product.declared() {
                key = key.bind(baseline(*class)).expect("bind");
            }
            key = key.bind(baseline(extra)).expect("extra binds");
            assert_eq!(
                key.seal().expect_err("extra"),
                DependencyKeyError::Undeclared { product, class: extra }
            );
        }
        assert!(ReusableProduct::ScoreTable.declares(DependencyClass::TargetIdentification));
        assert!(!ReusableProduct::ScoreTable.declares(DependencyClass::JointAlignment));
        assert!(!ReusableProduct::BatchDesign.declares(DependencyClass::Inference));
        assert!(!ReusableProduct::FrozenFit.declares(DependencyClass::TargetIdentification));
    }

    #[test]
    fn duplicate_dependency_refuses_instead_of_dropping_the_first() {
        let product = ReusableProduct::SourceSnapshot;
        let key =
            DependencyKey::new(product).bind(baseline(DependencyClass::SourceSnapshot)).unwrap();
        let error = key.bind(DependencyRecord::source_snapshot(digest(0x29))).unwrap_err();
        assert_eq!(
            error,
            DependencyKeyError::Duplicate { product, class: DependencyClass::SourceSnapshot }
        );
    }

    #[test]
    fn bind_order_does_not_change_the_key() {
        for product in ReusableProduct::ALL {
            let forward = seal_declared(product);
            let mut backward = DependencyKey::new(product);
            for class in product.declared().iter().rev() {
                backward = backward.bind(baseline(*class)).expect("bind");
            }
            assert_eq!(forward, backward.seal().expect("seal").digest());
        }
    }

    #[test]
    fn distinct_products_do_not_share_a_key() {
        let mut seen = std::collections::BTreeSet::new();
        for product in ReusableProduct::ALL {
            assert!(seen.insert(seal_declared(product).to_hex()), "{product}");
        }
    }

    #[test]
    fn misaligned_rows_and_folds_refuse() {
        let error =
            DependencyRecord::rows_history_folds(&[1, 2], digest(0x31), digest(0x32), &[0], 2)
                .expect_err("length");
        assert_eq!(error, DependencyKeyError::RowFoldLengthMismatch { rows: 2, folds: 1 });
    }

    #[test]
    fn omitted_class_is_reported_before_an_undeclared_extra() {
        let product = ReusableProduct::BatchDesign;
        let key = DependencyKey::new(product)
            .bind(baseline(DependencyClass::SourceSnapshot))
            .unwrap()
            .bind(baseline(DependencyClass::NumericalPolicy))
            .unwrap()
            .bind(baseline(DependencyClass::Inference))
            .unwrap();
        assert_eq!(
            key.seal().unwrap_err(),
            DependencyKeyError::Omitted { product, class: DependencyClass::RowsHistoryFolds }
        );
    }
}
