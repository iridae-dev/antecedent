//! Complete target-bound compiled proof representation, distinct from current authority.
// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::analysis_wire::{DiagnosticWire, diagnostic_from_wire, diagnostic_to_wire};
use crate::expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire};
use crate::trace::{
    AssumptionRecordWire, DerivationStepWire, assumptions_from_wire, assumptions_to_wire,
};
use crate::{IdentifiedEstimandWire, IoError};
use antecedent_core::{ScientificTarget, ScientificWorldId, VariableId};
use antecedent_identify::{
    AssembledFunctionalObligation, CompiledIdentificationProduct, CompiledProductParts,
    DerivationTrace, IdentificationClaimClass, IdentificationOutcome, RequiredFactor,
    UnresolvedObligation, WorldObligation,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Full historical proof, including original expression ownership and all support gaps.
/// Decoding validates structure/identity and grants no current scientific permission.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledIdentificationWire {
    version: u16,
    identity: [u8; 32],
    target: [u8; 32],
    estimands: Vec<IdentifiedEstimandWire>,
    arena: ExprArenaWire,
    derivation: Vec<DerivationStepWire>,
    assumptions: Vec<AssumptionRecordWire>,
    diagnostics: Vec<DiagnosticWire>,
    factors: Vec<(String, Vec<u32>)>,
    unresolved: Vec<(String, String)>,
    outcome: u8,
    claim: u8,
    value: Option<u64>,
    worlds: Vec<WorldWire>,
    assembled: Option<ObligationWire>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObligationWire {
    outcome: u8,
    claim: u8,
    unresolved: Vec<(String, String)>,
    value: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldWire {
    world: u32,
    obligation: ObligationWire,
}
impl CompiledIdentificationWire {
    /// Encode the actual retained proof without repeating identification.
    /// # Errors
    /// Unbound/mismatched targets or expression encoding errors.
    pub fn from_domain(
        proof: &CompiledIdentificationProduct,
        target: &ScientificTarget,
    ) -> Result<Self, IoError> {
        proof.validate_target(target).map_err(invalid)?;
        Ok(Self {
            version: 1,
            identity: *proof.identity().as_bytes(),
            target: *target.identity().as_bytes(),
            estimands: proof
                .identified()
                .iter()
                .map(|estimand| IdentifiedEstimandWire {
                    method: estimand.method.to_string(),
                    adjustment_set: variables(&estimand.adjustment_set),
                    instruments: variables(&estimand.instruments),
                    mediators: variables(&estimand.mediators),
                    functional: estimand.functional.raw(),
                    rd_design: estimand.rd_design.map(|design| {
                        crate::analysis_wire::RdDesignWire {
                            running_variable: design.running_variable.raw(),
                            cutoff: design.cutoff,
                            bandwidth: design.bandwidth,
                        }
                    }),
                })
                .collect(),
            arena: expr_arena_to_wire(proof.arena())?,
            derivation: proof
                .derivation()
                .steps
                .iter()
                .map(|step| DerivationStepWire {
                    rule: step.rule.to_string(),
                    detail: step.detail.to_string(),
                })
                .collect(),
            assumptions: assumptions_to_wire(proof.assumptions()),
            diagnostics: proof
                .bounded_search_diagnostics()
                .iter()
                .map(diagnostic_to_wire)
                .collect(),
            factors: proof
                .required_factors()
                .iter()
                .map(|factor| (factor.name().into(), variables(factor.variables())))
                .collect(),
            unresolved: gaps(proof.unresolved_obligations()),
            outcome: outcome_tag(proof.outcome()),
            claim: claim_tag(proof.claim_class()),
            value: proof.numerical_value().map(f64::to_bits),
            worlds: proof
                .world_obligations()
                .iter()
                .map(|world| WorldWire {
                    world: world.world().raw(),
                    obligation: ObligationWire {
                        outcome: outcome_tag(world.outcome()),
                        claim: claim_tag(world.claim_class()),
                        unresolved: gaps(world.unresolved()),
                        value: world.numerical_value().map(f64::to_bits),
                    },
                })
                .collect(),
            assembled: proof.assembled_obligation().map(|assembled| ObligationWire {
                outcome: outcome_tag(assembled.outcome()),
                claim: claim_tag(assembled.claim_class()),
                unresolved: gaps(assembled.unresolved()),
                value: assembled.numerical_value().map(f64::to_bits),
            }),
        })
    }
    /// Reconstruct historical proof through native coherence/target checks, then verify identity.
    /// # Errors
    /// Unknown version/tags, malformed expressions, incoherent support or changed identity.
    pub fn to_domain(
        &self,
        target: &ScientificTarget,
    ) -> Result<CompiledIdentificationProduct, IoError> {
        if self.version != 1 || self.target != *target.identity().as_bytes() {
            return Err(invalid("compiled proof version or target mismatch"));
        }
        for estimand in &self.estimands {
            if estimand.functional as usize >= self.arena.nodes.len() {
                return Err(invalid("compiled estimand root is outside its expression arena"));
            }
            if estimand.rd_design.as_ref().is_some_and(|design| {
                !design.cutoff.is_finite()
                    || !design.bandwidth.is_finite()
                    || design.bandwidth <= 0.0
            }) {
                return Err(invalid("invalid compiled RD design"));
            }
        }
        let mut parts = CompiledProductParts::new(outcome(self.outcome)?, claim(self.claim)?);
        parts.required_factors = self
            .factors
            .iter()
            .map(|(name, ids)| {
                RequiredFactor::new(
                    name.as_str(),
                    ids.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
                )
            })
            .collect();
        parts.unresolved_obligations = ungap(&self.unresolved);
        parts.numerical_value = self.value.map(f64::from_bits);
        parts.world_obligations = self
            .worlds
            .iter()
            .map(|world| {
                Ok(WorldObligation::new(
                    ScientificWorldId::from_raw(world.world),
                    outcome(world.obligation.outcome)?,
                    claim(world.obligation.claim)?,
                    ungap(&world.obligation.unresolved),
                    world.obligation.value.map(f64::from_bits),
                ))
            })
            .collect::<Result<_, IoError>>()?;
        parts.assembled = self
            .assembled
            .as_ref()
            .map(|assembled| -> Result<_, IoError> {
                Ok(AssembledFunctionalObligation::new(
                    outcome(assembled.outcome)?,
                    claim(assembled.claim)?,
                    ungap(&assembled.unresolved),
                    assembled.value.map(f64::from_bits),
                ))
            })
            .transpose()?;
        let estimands = self
            .estimands
            .iter()
            .map(|estimand| {
                antecedent_expr::IdentifiedEstimand::new(
                    estimand.method.as_str(),
                    ids(&estimand.adjustment_set),
                    ids(&estimand.instruments),
                    ids(&estimand.mediators),
                    antecedent_expr::ExprId::from_raw(estimand.functional),
                    estimand.rd_design.as_ref().map(|design| {
                        antecedent_expr::RdDesignParams::new(
                            VariableId::from_raw(design.running_variable),
                            design.cutoff,
                            design.bandwidth,
                        )
                    }),
                )
            })
            .collect();
        let mut derivation = DerivationTrace::default();
        for step in &self.derivation {
            derivation.push(step.rule.as_str(), step.detail.as_str());
        }
        let product = CompiledIdentificationProduct::new(
            estimands,
            expr_arena_from_wire(&self.arena)?,
            derivation,
            assumptions_from_wire(&self.assumptions)?,
            self.diagnostics.iter().map(diagnostic_from_wire).collect::<Result<_, _>>()?,
            parts,
        )
        .map_err(invalid)?
        .bind_target(target)
        .map_err(invalid)?;
        if self.identity != *product.identity().as_bytes() {
            return Err(invalid("compiled proof identity mismatch"));
        }
        Ok(product)
    }
}
fn variables(ids: &[VariableId]) -> Vec<u32> {
    ids.iter().map(|id| id.raw()).collect()
}
fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into()
}
fn gaps(unresolved: &[UnresolvedObligation]) -> Vec<(String, String)> {
    unresolved.iter().map(|gap| (gap.code().into(), gap.detail().into())).collect()
}
fn ungap(gaps: &[(String, String)]) -> Vec<UnresolvedObligation> {
    gaps.iter()
        .map(|(code, detail)| UnresolvedObligation::new(code.as_str(), detail.as_str()))
        .collect()
}
fn outcome_tag(value: IdentificationOutcome) -> u8 {
    match value {
        IdentificationOutcome::Identified => 0,
        IdentificationOutcome::Unidentified => 1,
        IdentificationOutcome::Unsupported => 2,
        IdentificationOutcome::TimedOut => 3,
        IdentificationOutcome::NumericallyInfeasible => 4,
        IdentificationOutcome::Uncalibrated => 5,
    }
}
fn outcome(value: u8) -> Result<IdentificationOutcome, IoError> {
    match value {
        0 => Ok(IdentificationOutcome::Identified),
        1 => Ok(IdentificationOutcome::Unidentified),
        2 => Ok(IdentificationOutcome::Unsupported),
        3 => Ok(IdentificationOutcome::TimedOut),
        4 => Ok(IdentificationOutcome::NumericallyInfeasible),
        5 => Ok(IdentificationOutcome::Uncalibrated),
        _ => Err(invalid("unknown identification outcome")),
    }
}
fn claim_tag(value: IdentificationClaimClass) -> u8 {
    match value {
        IdentificationClaimClass::IdentifiedClaim => 0,
        IdentificationClaimClass::ModelConditional => 1,
        IdentificationClaimClass::UnsupportedExtrapolation => 2,
    }
}
fn claim(value: u8) -> Result<IdentificationClaimClass, IoError> {
    match value {
        0 => Ok(IdentificationClaimClass::IdentifiedClaim),
        1 => Ok(IdentificationClaimClass::ModelConditional),
        2 => Ok(IdentificationClaimClass::UnsupportedExtrapolation),
        _ => Err(invalid("unknown identification claim class")),
    }
}
fn invalid(message: impl std::fmt::Display) -> IoError {
    IoError::Convert(message.to_string())
}
