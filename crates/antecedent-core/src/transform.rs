//! Transformation-effect reports for incremental compilation.
//!
//! Labels are a set, not a total order: an operation can preserve
//! identification, invalidate support, and require estimation at once.
//! Composition unions layer effects and concatenates unresolved
//! obligations; it cannot drop an earlier unresolved obligation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::identity::{IdentityRef, SemanticDigest};
use crate::obligation::ObligationRecord;

/// One semantic layer a transformation can affect.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum SemanticLayer {
    /// Causal target / query identity.
    Target,
    /// Structural identification.
    Identification,
    /// Licensed program (premises + products + inferential commitments).
    Program,
    /// Inference binding (priors, numeric knobs, validation).
    Inference,
    /// Empirical support.
    Support,
    /// Bound data / snapshot.
    Data,
    /// Physical execution resources.
    Execution,
    /// Published results.
    Results,
}

impl SemanticLayer {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Identification => "identification",
            Self::Program => "program",
            Self::Inference => "inference",
            Self::Support => "support",
            Self::Data => "data",
            Self::Execution => "execution",
            Self::Results => "results",
        }
    }
}

/// Effect of a transformation on one layer.
///
/// Not a total order — several variants may apply to the same layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum TransformEffect {
    /// Layer meaning is unchanged.
    Preserves,
    /// Layer meaning is weaker (e.g. a lossy view).
    Weakens,
    /// Identification products must be recomputed.
    RequiresReidentification,
    /// Estimation / scores / posterior must be recomputed.
    RequiresReestimation,
    /// Layer products are no longer valid.
    Invalidates,
    /// The transformation is refused for this layer.
    Refused,
}

impl TransformEffect {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Preserves => "preserves",
            Self::Weakens => "weakens",
            Self::RequiresReidentification => "requires_reidentification",
            Self::RequiresReestimation => "requires_reestimation",
            Self::Invalidates => "invalidates",
            Self::Refused => "refused",
        }
    }
}

/// Effects recorded for one semantic layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerEffect {
    /// Layer these effects apply to.
    pub layer: SemanticLayer,
    /// Distinct effects; order is canonical (sorted).
    pub effects: Arc<[TransformEffect]>,
}

impl LayerEffect {
    /// Construct from an iterator of effects (deduplicated and sorted).
    #[must_use]
    pub fn new(layer: SemanticLayer, effects: impl IntoIterator<Item = TransformEffect>) -> Self {
        let mut effects: Vec<TransformEffect> = effects.into_iter().collect();
        effects.sort_unstable();
        effects.dedup();
        Self { layer, effects: Arc::from(effects) }
    }

    /// Union with another report on the same layer.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.layer, other.layer);
        Self::new(self.layer, self.effects.iter().copied().chain(other.effects.iter().copied()))
    }

    /// Whether this layer was refused.
    #[must_use]
    pub fn is_refused(&self) -> bool {
        self.effects.contains(&TransformEffect::Refused)
    }
}

/// Caller intent for a transformation preview.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TransformIntent {
    /// Display precision or a view that retains the full contract reference.
    DisplayPrecision,
    /// Compatible data append/replace (same observation contract).
    CompatibleDataReplace,
    /// Licensed prepared `retarget`.
    Retarget,
    /// Display-only filter; not a new population or query.
    FilterDisplay,
    /// Declared population reweighting.
    FilterPopulation,
    /// New conditional causal query.
    NewConditionalQuery,
    /// Change graph, query, intervention policy, horizon, or structural assumption.
    ChangeGraph,
    /// Change prior or its mapping.
    ChangePrior,
    /// Change only physical execution policy.
    ChangePhysicalPolicy,
    /// Average an unweighted CPDAG/PAG class into a probability-weighted mixture.
    AverageUnweightedClass,
}

impl TransformIntent {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DisplayPrecision => "display_precision",
            Self::CompatibleDataReplace => "compatible_data_replace",
            Self::Retarget => "retarget",
            Self::FilterDisplay => "filter_display",
            Self::FilterPopulation => "filter_population",
            Self::NewConditionalQuery => "new_conditional_query",
            Self::ChangeGraph => "change_graph",
            Self::ChangePrior => "change_prior",
            Self::ChangePhysicalPolicy => "change_physical_policy",
            Self::AverageUnweightedClass => "average_unweighted_class",
        }
    }
}

/// Inspectable effect of one transformation, including input identities.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TransformationReport {
    /// Intent that produced this report.
    pub intent: TransformIntent,
    /// Input identities the preview is bound to.
    pub input_identities: Arc<[IdentityRef]>,
    /// Per-layer effects.
    pub layer_effects: Arc<[LayerEffect]>,
    /// Obligations introduced or retained.
    pub obligations: Arc<[ObligationRecord]>,
    /// Whether any layer refused the transformation.
    pub refused: bool,
}

impl TransformationReport {
    /// Construct a report with canonical layer order.
    #[must_use]
    pub fn new(
        intent: TransformIntent,
        input_identities: impl Into<Arc<[IdentityRef]>>,
        layer_effects: impl IntoIterator<Item = LayerEffect>,
        obligations: impl Into<Arc<[ObligationRecord]>>,
    ) -> Self {
        let mut layer_effects: Vec<LayerEffect> = layer_effects.into_iter().collect();
        layer_effects.sort_by_key(|layer| layer.layer);
        let refused = layer_effects.iter().any(LayerEffect::is_refused);
        Self {
            intent,
            input_identities: input_identities.into(),
            layer_effects: Arc::from(layer_effects),
            obligations: obligations.into(),
            refused,
        }
    }

    /// Effects recorded for `layer`, if any.
    #[must_use]
    pub fn layer(&self, layer: SemanticLayer) -> Option<&LayerEffect> {
        self.layer_effects.iter().find(|entry| entry.layer == layer)
    }

    /// Whether this preview still matches the supplied program identity.
    #[must_use]
    pub fn binds_program(&self, program: SemanticDigest) -> bool {
        self.input_identities
            .iter()
            .any(|id| id.domain == crate::identity::IdentityDomain::Program && id.digest == program)
    }

    /// Whether all frozen contract inputs still match, including the data snapshot.
    ///
    /// Program equality alone cannot detect a refresh that preserves the target
    /// and identification. This is a freshness check, not an execution license.
    #[must_use]
    pub fn binds_contract(&self, identities: &crate::identity::ContractIdentities) -> bool {
        self.input_identities == identities.refs()
    }

    /// Compose `self` then `next`. Unresolved obligations are concatenated;
    /// layer effects are unioned. A later `Preserves` cannot erase an earlier
    /// `Invalidates` or unresolved obligation.
    #[must_use]
    pub fn compose(&self, next: &Self) -> Self {
        let mut layers: Vec<LayerEffect> = self.layer_effects.iter().cloned().collect();
        for incoming in next.layer_effects.iter() {
            if let Some(existing) = layers.iter_mut().find(|layer| layer.layer == incoming.layer) {
                *existing = existing.union(incoming);
            } else {
                layers.push(incoming.clone());
            }
        }
        let mut obligations: Vec<ObligationRecord> = self.obligations.iter().cloned().collect();
        for obligation in next.obligations.iter() {
            if !obligations.iter().any(|existing| existing.id == obligation.id) {
                obligations.push(obligation.clone());
            }
        }
        Self::new(next.intent, next.input_identities.clone(), layers, obligations)
    }
}

/// Standard per-intent layer effects used by preview and apply.
#[must_use]
pub fn intent_effects(intent: TransformIntent) -> Arc<[LayerEffect]> {
    use SemanticLayer::{
        Data, Execution, Identification, Inference, Program, Results, Support, Target,
    };
    use TransformEffect::{
        Invalidates, Preserves, RequiresReestimation, RequiresReidentification, Weakens,
    };
    let layers = match intent {
        TransformIntent::DisplayPrecision | TransformIntent::FilterDisplay => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [Preserves]),
            LayerEffect::new(Program, [Preserves]),
            LayerEffect::new(Results, [Weakens]),
        ],
        TransformIntent::CompatibleDataReplace => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [Preserves]),
            LayerEffect::new(Program, [Preserves]),
            LayerEffect::new(Support, [Invalidates, RequiresReestimation]),
            LayerEffect::new(Data, [Invalidates]),
            LayerEffect::new(Inference, [Invalidates, RequiresReestimation]),
            LayerEffect::new(Results, [Invalidates]),
        ],
        TransformIntent::Retarget => vec![
            LayerEffect::new(Target, [Invalidates]),
            LayerEffect::new(Identification, [Preserves]),
            LayerEffect::new(Program, [Invalidates]),
            LayerEffect::new(Support, [RequiresReestimation]),
            LayerEffect::new(Results, [Invalidates]),
        ],
        TransformIntent::FilterPopulation | TransformIntent::NewConditionalQuery => vec![
            LayerEffect::new(Target, [Invalidates]),
            LayerEffect::new(Identification, [RequiresReidentification]),
            LayerEffect::new(Program, [Invalidates]),
            LayerEffect::new(Results, [Invalidates]),
        ],
        TransformIntent::ChangeGraph => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [RequiresReidentification, Invalidates]),
            LayerEffect::new(Program, [Invalidates]),
            LayerEffect::new(Results, [Invalidates]),
        ],
        TransformIntent::ChangePrior => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [Preserves]),
            LayerEffect::new(Program, [Preserves]),
            LayerEffect::new(Inference, [Invalidates, RequiresReestimation]),
            LayerEffect::new(Results, [Invalidates]),
        ],
        TransformIntent::ChangePhysicalPolicy => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [Preserves]),
            LayerEffect::new(Program, [Preserves]),
            LayerEffect::new(Execution, [Invalidates]),
        ],
        TransformIntent::AverageUnweightedClass => vec![
            LayerEffect::new(Target, [Preserves]),
            LayerEffect::new(Identification, [TransformEffect::Refused, Preserves]),
            LayerEffect::new(Program, [TransformEffect::Refused]),
            LayerEffect::new(Results, [TransformEffect::Refused]),
        ],
    };
    Arc::from(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assumption::{AssumptionSource, AssumptionStatus};
    use crate::identity::{IdentityDomain, SemanticDigest};
    use crate::obligation::{ObligationKind, ObligationRecord, ObligationScope};

    fn digest(byte: u8) -> SemanticDigest {
        let mut bytes = [0u8; 32];
        bytes[0] = byte;
        SemanticDigest::from_bytes(bytes)
    }

    fn obligation(id: &str) -> ObligationRecord {
        ObligationRecord::new(
            id,
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            ObligationKind::CheckNotRun,
            AssumptionStatus::Declared,
            "pending check",
        )
    }

    #[test]
    fn composition_cannot_drop_unresolved_obligations() {
        let program = digest(1);
        let input = [IdentityRef::new(IdentityDomain::Program, program)];
        let first = TransformationReport::new(
            TransformIntent::CompatibleDataReplace,
            input,
            intent_effects(TransformIntent::CompatibleDataReplace).iter().cloned(),
            [obligation("overlap")],
        );
        let second = TransformationReport::new(
            TransformIntent::DisplayPrecision,
            input,
            intent_effects(TransformIntent::DisplayPrecision).iter().cloned(),
            [],
        );
        let chained = first.compose(&second);
        assert!(chained.obligations.iter().any(|o| &*o.id == "overlap"));
        assert!(chained.obligations.iter().any(ObligationRecord::is_unresolved));
        let support = chained.layer(SemanticLayer::Support).expect("support layer");
        assert!(support.effects.contains(&TransformEffect::Invalidates));
        assert!(support.effects.contains(&TransformEffect::RequiresReestimation));
        let identification = chained.layer(SemanticLayer::Identification).expect("id layer");
        assert!(identification.effects.contains(&TransformEffect::Preserves));
    }

    #[test]
    fn display_precision_does_not_reidentify() {
        let effects = intent_effects(TransformIntent::DisplayPrecision);
        let identification = effects.iter().find(|e| e.layer == SemanticLayer::Identification);
        assert!(
            identification
                .is_some_and(|e| e.effects.as_ref() == [TransformEffect::Preserves].as_slice())
        );
    }
}
