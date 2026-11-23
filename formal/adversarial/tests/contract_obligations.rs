//! Generated chains of contract transformations against
//! `TransformationReport::compose`.
//!
//! The expected outcome comes from the composition rules proved in
//! `formal/Antecedent/Contract.lean`: a chain never drops an unresolved obligation of
//! any step, never erases an effect a step recorded on a layer, and is refused when any
//! step was refused.

use antecedent_contract::{
    LayerEffect, ObligationKind, ObligationRecord, ObligationScope, SemanticLayer, TransformEffect,
    TransformIntent, TransformationReport,
};
use antecedent_core::{AssumptionSource, AssumptionStatus};
use proptest::prelude::*;

const LAYERS: [SemanticLayer; 8] = [
    SemanticLayer::Target,
    SemanticLayer::Identification,
    SemanticLayer::Program,
    SemanticLayer::Inference,
    SemanticLayer::Support,
    SemanticLayer::Data,
    SemanticLayer::Execution,
    SemanticLayer::Results,
];
const EFFECTS: [TransformEffect; 6] = [
    TransformEffect::Preserves,
    TransformEffect::Weakens,
    TransformEffect::RequiresReidentification,
    TransformEffect::RequiresReestimation,
    TransformEffect::Invalidates,
    TransformEffect::Refused,
];
const KINDS: [ObligationKind; 6] = [
    ObligationKind::UserAssertion,
    ObligationKind::GraphicalImplication,
    ObligationKind::EmpiricalDiagnostic,
    ObligationKind::FailedCheck,
    ObligationKind::Uncheckable,
    ObligationKind::CheckNotRun,
];
const STATUSES: [AssumptionStatus; 4] = [
    AssumptionStatus::Declared,
    AssumptionStatus::Supported,
    AssumptionStatus::Contradicted,
    AssumptionStatus::Untestable,
];

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

/// An obligation with one of a few ids, so that later steps reuse earlier ids.
fn obligation() -> impl Strategy<Value = ObligationRecord> {
    (0u8..4, 0usize..KINDS.len(), 0usize..STATUSES.len()).prop_map(|(id, kind, status)| {
        ObligationRecord::new(
            format!("obligation_{id}"),
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            KINDS[kind],
            STATUSES[status],
            "generated",
        )
    })
}

fn report() -> impl Strategy<Value = TransformationReport> {
    (
        prop::collection::vec(
            (0usize..LAYERS.len(), prop::collection::vec(0usize..EFFECTS.len(), 1..3)),
            0..4,
        ),
        prop::collection::vec(obligation(), 0..4),
    )
        .prop_map(|(layers, obligations)| {
            let mut effects: Vec<LayerEffect> = Vec::new();
            for (layer, chosen) in layers {
                let layer = LAYERS[layer];
                let incoming = LayerEffect::new(layer, chosen.into_iter().map(|e| EFFECTS[e]));
                match effects.iter_mut().find(|entry| entry.layer == layer) {
                    Some(entry) => *entry = entry.union(&incoming),
                    None => effects.push(incoming),
                }
            }
            TransformationReport::new(TransformIntent::Retarget, Vec::new(), effects, obligations)
        })
}

fn chain(reports: &[TransformationReport]) -> TransformationReport {
    let mut composed = reports[0].clone();
    for next in &reports[1..] {
        composed = composed.compose(next);
    }
    composed
}

proptest! {
    #![proptest_config(config())]

    /// Every unresolved obligation of every step is in the composed chain.
    #[test]
    fn chain_keeps_unresolved_obligations(reports in prop::collection::vec(report(), 1..5)) {
        let composed = chain(&reports);
        for report in &reports {
            for obligation in report.obligations.iter().filter(|o| o.is_unresolved()) {
                prop_assert!(
                    composed.obligations.contains(obligation),
                    "unresolved {} ({:?}, {:?}) dropped",
                    obligation.id, obligation.kind, obligation.status
                );
            }
        }
    }

    /// Every effect a step recorded on a layer is recorded by the chain, so a later
    /// `Preserves` never erases an earlier `Invalidates` or `Refused`.
    #[test]
    fn chain_keeps_layer_effects(reports in prop::collection::vec(report(), 1..5)) {
        let composed = chain(&reports);
        for report in &reports {
            for layer in report.layer_effects.iter() {
                let kept = composed.layer(layer.layer).expect("layer kept");
                for effect in layer.effects.iter() {
                    prop_assert!(kept.effects.contains(effect), "{:?} lost on {:?}", effect, layer.layer);
                }
            }
        }
    }

    /// A chain with a refused step is refused.
    #[test]
    fn chain_with_a_refused_step_is_refused(reports in prop::collection::vec(report(), 1..5)) {
        let composed = chain(&reports);
        prop_assert_eq!(composed.refused, reports.iter().any(|r| r.refused));
    }
}
