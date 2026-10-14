//! Transported surrogate bridge: a trial-level surrogate-to-terminal relation
//! validated in one population, carried to a claim about another.
//!
//! The bridge contract already requires a transportability premise whenever
//! its evidence was established for another population or regime than the
//! claim's. This module states what that premise is for a surrogate bridge and
//! ties it to a declared selection diagram, so it is a checked statement and
//! not free text:
//!
//! > The relation between the effect on the surrogate and the effect on the
//! > terminal outcome, validated across trials in population `P`, is the same
//! > in population `Q`.
//!
//! That relation is determined by the mechanism of the terminal outcome given
//! the surrogate and by the variables that modify it. It is invariant between
//! `P` and `Q` only when the declared selection diagram has no selection node
//! pointing into the terminal outcome or into any declared modifier of the
//! relation. A selection node into one of them contradicts the premise, and
//! the bridge is refused naming the premise. A selection node into the
//! *surrogate* does not: the surrogate's effect in `Q` is the claim being
//! bridged and is measured there.
//!
//! The route is the contract's: point-only, a mean, one trial-level linear
//! mapping inside its validated surrogate range, no interval and no
//! substitution for an identified causal effect. The premise stays an
//! unresolved obligation of the derived claim; this module checks it against
//! the diagram, it does not discharge it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeRefusal, BridgeScope, BridgeSource, BridgedClaim, QuantityClass, TransportabilityPremise,
    TrialLevelValidation, ValueKind, derive_bridged_claim,
};
use antecedent_contract::{ClaimEnvelope, ScientificQuantity};
use antecedent_core::identity::CanonicalWriter;

/// Version of the transported surrogate bridge declaration semantics.
pub const TRANSPORTED_BRIDGE_VERSION: u32 = 1;

/// Identity of the premise every transported surrogate bridge states.
pub const SURROGATE_TRANSPORT_PREMISE_ID: &str = "surrogate_terminal_relation_invariant";
/// A selection node points into the terminal outcome or a modifier of the
/// relation, contradicting the premise.
pub const DETAIL_PREMISE_CONTRADICTED: &str =
    "scientific_bridge.transport_premise_contradicted_by_selection_diagram";
/// The declared diagram is not between the scopes the bridge moves across.
pub const DETAIL_DIAGRAM_SCOPE_MISMATCH: &str =
    "scientific_bridge.transport_diagram_scope_mismatch";

const TAG_DIAGRAM: &str = "antecedent.design.transported_bridge.selection_diagram";

/// A declared selection diagram between two populations, by name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeSelectionDiagram {
    /// Population the relation was validated in.
    pub source_population: String,
    /// Population the claim is about.
    pub target_population: String,
    /// Every variable a selection node points into.
    pub selection_targets: Vec<String>,
}

impl BridgeSelectionDiagram {
    /// Identity: populations and the sorted, distinct selection targets.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut targets = self.selection_targets.clone();
        targets.sort();
        targets.dedup();
        let mut w = CanonicalWriter::new(TAG_DIAGRAM);
        w.word(u64::from(TRANSPORTED_BRIDGE_VERSION))
            .text(&self.source_population)
            .text(&self.target_population)
            .texts(&targets);
        antecedent_core::identity::to_hex(&w.finish())
    }

    fn points_into(&self, variable: &str) -> bool {
        self.selection_targets.iter().any(|t| t == variable)
    }
}

/// A surrogate bridge for a claim about a population the validation trials
/// were not run in.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedSurrogateBridge {
    /// The surrogate-effect quantity of the claim, in the claim's population.
    pub surrogate: ScientificQuantity,
    /// The terminal-effect quantity the bridge delivers, in the same
    /// population.
    pub terminal: ScientificQuantity,
    /// The trials the relation was validated on.
    pub validation: TrialLevelValidation,
    /// Population and regime the trials were run in.
    pub validated_in: BridgeScope,
    /// The validation study identity (the evidence of the surrogate
    /// validation assumption).
    pub validation_evidence: String,
    /// The selection diagram between `validated_in` and the claim's scope.
    pub diagram: BridgeSelectionDiagram,
    /// Identity of the terminal-outcome variable in the diagram.
    pub terminal_variable: String,
    /// Variables that modify the surrogate-terminal relation; a selection
    /// node into any of them contradicts the premise.
    pub relation_modifiers: Vec<String>,
}

fn contradicted(variable: &str, role: &str) -> BridgeRefusal {
    BridgeRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        "bind",
        DETAIL_PREMISE_CONTRADICTED,
    )
    .with_offending("transportability_premise")
    .with_expected(format!("no selection node into the {role} `{variable}`"))
    .with_supplied(format!("a selection node into `{variable}`"))
    .with_remedy(
        "the relation between surrogate and terminal effects is not invariant between these populations; validate the surrogate in the claim's population, or declare a diagram without a selection node into the terminal outcome and its modifiers",
    )
}

fn scope_mismatch(expected: String, supplied: String) -> BridgeRefusal {
    BridgeRefusal::new(
        antecedent_core::reason_code!("bridge_binding_mismatch"),
        "bind",
        DETAIL_DIAGRAM_SCOPE_MISMATCH,
    )
    .with_offending("selection_diagram")
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(
        "declare the selection diagram between the population the surrogate was validated in and the population the claim is about",
    )
}

impl TransportedSurrogateBridge {
    /// The bridge contract, with the transportability premise stated and
    /// checked against the diagram. When the claim is about the scope the
    /// trials were run in, no premise is stated and the diagram is not used.
    ///
    /// # Errors
    /// `bridge_assumption_missing` /
    /// `scientific_bridge.transport_premise_contradicted_by_selection_diagram`
    /// when a selection node points into the terminal outcome or a modifier;
    /// `bridge_binding_mismatch` /
    /// `scientific_bridge.transport_diagram_scope_mismatch` when the diagram
    /// is not between the two populations.
    pub fn contract(&self) -> Result<BridgeContract, BridgeRefusal> {
        let claim_scope = BridgeScope {
            population_id: self.surrogate.population_id.clone(),
            regime_id: self.surrogate.regime_id.clone(),
        };
        let moved = claim_scope != self.validated_in;
        let transport = if moved {
            if self.diagram.source_population != self.validated_in.population_id
                || self.diagram.target_population != claim_scope.population_id
            {
                return Err(scope_mismatch(
                    format!("{} -> {}", self.validated_in.population_id, claim_scope.population_id),
                    format!(
                        "{} -> {}",
                        self.diagram.source_population, self.diagram.target_population
                    ),
                ));
            }
            if self.diagram.points_into(&self.terminal_variable) {
                return Err(contradicted(&self.terminal_variable, "terminal outcome"));
            }
            if let Some(modifier) =
                self.relation_modifiers.iter().find(|m| self.diagram.points_into(m))
            {
                return Err(contradicted(modifier, "relation modifier"));
            }
            Some(TransportabilityPremise {
                id: SURROGATE_TRANSPORT_PREMISE_ID.to_owned(),
                statement: format!(
                    "the relation between the surrogate and terminal effects validated in population `{}` under regime `{}` is the same in population `{}` under regime `{}`: no selection node points into the terminal outcome `{}` or into {} declared modifier(s) of the relation",
                    self.validated_in.population_id,
                    self.validated_in.regime_id,
                    claim_scope.population_id,
                    claim_scope.regime_id,
                    self.terminal_variable,
                    self.relation_modifiers.len(),
                ),
                established_in: self.validated_in.clone(),
                applies_to: claim_scope,
                evidence: Some(format!("selection_diagram:{}", self.diagram.identity_hex())),
            })
        } else {
            None
        };
        Ok(BridgeContract {
            kind: BridgeKind::SurrogateTerminalOutcome,
            source: BridgeEndpoint {
                quantity: self.surrogate.clone(),
                class: QuantityClass::SurrogateEndpoint,
            },
            target: BridgeEndpoint {
                quantity: self.terminal.clone(),
                class: QuantityClass::TerminalOutcome,
            },
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: self.validated_in.clone(),
            mapping: BridgeMapping::TrialLevelLinear(self.validation.clone()),
            assumptions: vec![BridgeAssumption {
                id: "surrogate_validation".to_owned(),
                kind: AssumptionKind::SurrogateValidation,
                statement: "the surrogate is validated for the terminal outcome at the trial level"
                    .to_owned(),
                evidence: Some(self.validation_evidence.clone()),
            }],
            transport,
        })
    }

    /// Derive the terminal-outcome claim from a surrogate-effect claim.
    ///
    /// # Errors
    /// The refusals of [`Self::contract`], then the contract's own: an
    /// unlicensed bridge, a source about another quantity or unit, a source
    /// that is not a finite point, and a surrogate effect outside the
    /// validated range.
    pub fn derive(&self, claim: &ClaimEnvelope) -> Result<BridgedClaim, BridgeRefusal> {
        let contract = self.contract()?;
        derive_bridged_claim(&contract, &BridgeSource { claim, quantity: &self.surrogate })
    }
}
