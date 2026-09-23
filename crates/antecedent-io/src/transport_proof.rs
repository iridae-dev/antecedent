//! Portable classical transport proofs. Decoding never reruns identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{
    IoError,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
};
use antecedent_core::ExecutionContext;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::sid::SidDerivationRecord;
use antecedent_identify::{ClassicalTransportDerivation, ClassicalTransportQuery, SidLimits};
use serde::{Deserialize, Serialize};

/// Untrusted expression plus typed local premises. Successful deserialization
/// alone is not verification; consumers must call `check` against their inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportProofWire {
    /// Versioned theoretical evidence scope, query and rule premises.
    pub proof: SidDerivationRecord,
    /// Original certified functional, never the optimized physical plan.
    pub expression: ExprArenaWire,
}
impl TransportProofWire {
    /// Encode a native checked proof.
    ///
    /// # Errors
    /// Expression exceeds wire capacity.
    pub fn from_checked(proof: &ClassicalTransportDerivation) -> Result<Self, IoError> {
        Ok(Self { proof: proof.to_record(), expression: expr_arena_to_wire(proof.arena())? })
    }
    /// Independently verify every local premise, expression and input identity.
    /// Does not fetch data, fit models, or rerun identification.
    ///
    /// # Errors
    /// Malformed expression, substituted query or invalid derivation.
    pub fn check(
        &self,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<ClassicalTransportDerivation, IoError> {
        if self.proof.steps.len() > limits.steps || ctx.cancellation.is_cancelled() {
            return Err(IoError::Convert("transport proof budget/cancellation".into()));
        }
        let bytes = self
            .expression
            .nodes
            .len()
            .saturating_mul(128)
            .saturating_add(
                self.expression
                    .var_sets
                    .iter()
                    .map(|v| v.len().saturating_mul(8))
                    .fold(0usize, usize::saturating_add),
            )
            .saturating_add(
                self.expression
                    .lists
                    .iter()
                    .map(|v| v.len().saturating_mul(8))
                    .fold(0usize, usize::saturating_add),
            )
            .saturating_add(
                self.expression
                    .interventions
                    .iter()
                    .map(|v| v.len().saturating_mul(64))
                    .fold(0usize, usize::saturating_add),
            );
        if ctx
            .memory
            .hard_limit_bytes
            .is_some_and(|limit| u64::try_from(bytes).map_or(true, |bytes| bytes > limit))
        {
            return Err(IoError::Convert("transport proof memory budget".into()));
        }
        ClassicalTransportDerivation::from_record_checked(
            self.proof.clone(),
            expr_arena_from_wire(&self.expression)?,
            diagram,
            query,
            limits,
            ctx,
        )
        .map_err(|e| IoError::Convert(e.to_string()))
    }
}
