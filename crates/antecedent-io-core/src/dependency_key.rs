//! Dependency classes and the classes each reusable product declares.
//!
//! Each [`ReusableProduct`] names the dependency classes it is valid under:
//! target and identification, source snapshot, rows/history/folds, model/prior/provider,
//! intervention targets, joint alignment, inference, and numerical policy.
//! The staleness report reads these declarations to say which class changed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// Provider result bound to the snapshot and the rows it read.
    ProviderReplay,
    /// Aligned joint law. Bound to the data and model it was computed from, so two laws
    /// over the same axes on different data do not share a key.
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

    /// Stable `snake_case` name. Part of the declared catalog.
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
            Self::FrozenFit | Self::ProviderReplay => &[
                DependencyClass::SourceSnapshot,
                DependencyClass::RowsHistoryFolds,
                DependencyClass::ModelPriorProvider,
                DependencyClass::Inference,
                DependencyClass::NumericalPolicy,
            ],
            Self::JointLaw => &[
                DependencyClass::TargetIdentification,
                DependencyClass::SourceSnapshot,
                DependencyClass::RowsHistoryFolds,
                DependencyClass::ModelPriorProvider,
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

#[cfg(test)]
mod tests {
    use super::{DependencyClass, ReusableProduct};

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
    fn joint_law_and_provider_replay_declare_their_data() {
        for class in [
            DependencyClass::SourceSnapshot,
            DependencyClass::RowsHistoryFolds,
            DependencyClass::ModelPriorProvider,
        ] {
            assert!(ReusableProduct::JointLaw.declares(class), "joint law lacks {class}");
        }
        assert!(ReusableProduct::ProviderReplay.declares(DependencyClass::RowsHistoryFolds));
    }
}
