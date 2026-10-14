//! Transported surrogate bridge against the hand-computed trial-level
//! regression, with the selection-diagram premise check and its refusals.
//!
//! Oracle: validation trials with effects `(0, 0.1), (0.5, 0.5), (1, 0.9),
//! (1.5, 1.5)` give `Sxx = 1.25`, `Sxy = 1.15`, slope `0.92`, intercept
//! `0.06`. A surrogate effect of `0.8` maps to `0.06 + 0.92 * 0.8 = 0.796`.

use std::sync::Arc;

use antecedent_contract::bridge::{
    BridgeRefusal, BridgeScope, MappingKind, TrialEffectPair, TrialLevelValidation,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use antecedent_design::transported_bridge::{
    BridgeSelectionDiagram, DETAIL_DIAGRAM_SCOPE_MISMATCH, DETAIL_PREMISE_CONTRADICTED,
    SURROGATE_TRANSPORT_PREMISE_ID, TransportedSurrogateBridge,
};

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(variable: &str, units: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: "do(treatment=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(value: f64) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from("mmol_per_l")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        [Arc::<str>::from("prior:p0")],
    )
}

fn validation() -> TrialLevelValidation {
    TrialLevelValidation {
        trials: [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
            .into_iter()
            .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                surrogate_effect,
                terminal_effect,
            })
            .collect(),
        min_trials: 4,
        min_r_squared: 0.9,
    }
}

fn bridge(claim_population: &str, selection_targets: &[&str]) -> TransportedSurrogateBridge {
    TransportedSurrogateBridge {
        surrogate: quantity("trial:biomarker", "mmol_per_l", claim_population),
        terminal: quantity("trial:survival", "log_hazard", claim_population),
        validation: validation(),
        validated_in: BridgeScope {
            population_id: "adults".into(),
            regime_id: "do(treatment=1)".into(),
        },
        validation_evidence: "meta-analysis:ckd-2024".into(),
        diagram: BridgeSelectionDiagram {
            source_population: "adults".into(),
            target_population: claim_population.into(),
            selection_targets: selection_targets.iter().map(|s| (*s).to_owned()).collect(),
        },
        terminal_variable: "survival".into(),
        relation_modifiers: vec!["age".into()],
    }
}

fn refused<T: std::fmt::Debug>(
    result: Result<T, BridgeRefusal>,
    code: &str,
    detail: &str,
) -> BridgeRefusal {
    let refusal = result.expect_err("the bridge must be refused");
    assert_eq!(refusal.code, code, "{refusal:?}");
    assert_eq!(refusal.detail, detail, "{refusal:?}");
    assert_eq!(refusal.validate(), Ok(()), "{refusal:?}");
    assert!(refusal.remedy.is_some(), "{refusal:?}");
    refusal
}

#[test]
fn a_surrogate_validated_elsewhere_is_carried_when_the_diagram_leaves_the_relation_invariant() {
    // Selection into the surrogate itself (children respond differently to the
    // drug) does not contradict the premise: the surrogate effect in children
    // is the claim being bridged.
    let bridged = bridge("children", &["biomarker"]).derive(&claim(0.8)).unwrap();
    assert!((bridged.value - 0.796).abs() < 1e-12, "{}", bridged.value);
    assert!(bridged.transported);
    assert_eq!(bridged.mapping, MappingKind::TrialLevelLinear);
    assert!(
        bridged
            .receipt
            .unresolved
            .iter()
            .any(|item| &**item == format!("transportability:{SURROGATE_TRANSPORT_PREMISE_ID}"))
    );
    // Still point-only: no interval is licensed for the transported bridge.
    assert!(bridged.request_interval().is_err());
}

#[test]
fn the_premise_is_stated_in_the_contract_and_bound_to_the_diagram() {
    let contract = bridge("children", &[]).contract().unwrap();
    let premise = contract.transport.as_ref().unwrap();
    assert_eq!(premise.id, SURROGATE_TRANSPORT_PREMISE_ID);
    assert_eq!(premise.established_in.population_id, "adults");
    assert_eq!(premise.applies_to.population_id, "children");
    let evidence = premise.evidence.as_deref().unwrap();
    assert_eq!(
        evidence,
        format!("selection_diagram:{}", bridge("children", &[]).diagram.identity_hex())
    );

    // A different diagram is a different contract identity at unchanged shape.
    let other = bridge("children", &["unrelated"]).contract().unwrap();
    assert_ne!(contract.identity(), other.identity());
    // Order and repetition of selection targets are not identity.
    let a = bridge("children", &["a", "b"]).diagram.identity_hex();
    let b = bridge("children", &["b", "a", "a"]).diagram.identity_hex();
    assert_eq!(a, b);
}

#[test]
fn a_selection_arrow_into_the_terminal_outcome_refuses_naming_the_premise() {
    let refusal = refused(
        bridge("children", &["survival"]).derive(&claim(0.8)),
        "bridge_assumption_missing",
        DETAIL_PREMISE_CONTRADICTED,
    );
    assert_eq!(refusal.offending.as_deref(), Some("transportability_premise"));
    assert!(refusal.expected.unwrap().contains("survival"));
}

#[test]
fn a_selection_arrow_into_a_relation_modifier_refuses_the_same_premise() {
    let refusal = refused(
        bridge("children", &["age"]).derive(&claim(0.8)),
        "bridge_assumption_missing",
        DETAIL_PREMISE_CONTRADICTED,
    );
    assert!(refusal.expected.unwrap().contains("age"));
}

#[test]
fn a_diagram_between_other_populations_is_not_the_premise() {
    let mut wrong = bridge("children", &[]);
    wrong.diagram.source_population = "elderly".into();
    refused(wrong.derive(&claim(0.8)), "bridge_binding_mismatch", DETAIL_DIAGRAM_SCOPE_MISMATCH);
}

#[test]
fn a_claim_in_the_validated_population_needs_no_premise_and_ignores_the_diagram() {
    // Nothing is transported: the contract states no premise, and a diagram
    // that would contradict one is not consulted.
    let bridged = bridge("adults", &["survival"]).derive(&claim(0.8)).unwrap();
    assert!(!bridged.transported);
    assert!((bridged.value - 0.796).abs() < 1e-12);
    assert!(bridge("adults", &["survival"]).contract().unwrap().transport.is_none());
}

#[test]
fn a_surrogate_effect_outside_the_validated_range_is_refused_after_the_premise() {
    // The validated surrogate range is [0, 1.5]; the contract's own refusal
    // stands for a transported claim.
    let refusal = bridge("children", &[]).derive(&claim(9.0)).unwrap_err();
    assert_eq!(refusal.validate(), Ok(()));
    assert!(refusal.remedy.is_some());
}
