//! Uncertainty carried through a chain of transformations: compile, execute,
//! change, export and independently consume.
//!
//! The single-step operators each propagate one step: a pool, a bridge, a
//! transport, a composition, a decision. A chain runs three or more of them in
//! sequence, and the uncertainty has to survive the sequence. This module
//! routes a declared chain through the layers that own each rule and adds none
//! of its own:
//!
//! 0. **Compile** ([`compile_propagation_chain`]). The chain compiles to a
//!    [`LoweredProgram`] of the six compiler layers (the chain's output as the
//!    scientific problem, the dependence contract as the identification of
//!    overlap, the stages as the statistical and evidence program, one physical
//!    operator per stage, the claim demand), viewable as a [`PlanView`] on the
//!    `PropagationChain` route. Nothing is computed at compile. Execution reads
//!    only the retained program, runs inside its dispatch scope (a nested
//!    compile is `public_builder_reentry`), and stamps the program identity on
//!    the result. [`PropagationChain::seal`], [`PropagationChain::withdraw`]
//!    and [`PropagationChain::reweight`] are thin wrappers over that path.
//! 1. **Check** (`antecedent_contract::check_propagation_chain`). Every source
//!    declares the rows, clusters, models, priors, disturbances and periods it
//!    uses; a stage inherits what its inputs use. An independent pool of
//!    overlapping evidence is refused with `evidence_double_counted` and
//!    names the dependent pool; an undeclared overlap is
//!    `evidence_overlap_unknown`. Nonlinear stages read the joint law of their
//!    inputs from one aligned draw table, and draws of unrelated tables are
//!    never paired by index (`joint_law_required`). Where the representation
//!    cannot carry the uncertainty (a point asked for a spread, an interval
//!    for a posterior, a sampling law for a posterior, an interval for a
//!    pool), the refusal names the representation that would.
//! 2. **Execute** (`antecedent_stats::propagation_chain`). Stages run row by
//!    row on the aligned table, so `g(E[X]) != E[g(X)]` holds by construction;
//!    intervals run by interval arithmetic into an enclosure that holds under
//!    every coupling; a dependent pool weights by the joint covariance of the
//!    aligned draws; a preposterior update simulates the future study per row,
//!    updates the posterior draws by its likelihood and re-evaluates the
//!    decision under them.
//! 3. **Retain** (`propagation_chain_v1`, the `analysis_result.propagation`
//!    section). The record seals the declaration, the draws, the future-study
//!    draws and the appended changes under one identity, with the stored
//!    results. [`consume_propagation_chain`] checks the rule again and replays
//!    every stage to identical bits.
//!
//! # Changed assumptions
//!
//! [`PropagationChain::withdraw`] and [`PropagationChain::reweight`] append a
//! change and return a new chain. A withdrawn source invalidates the stages
//! that depend on it; a re-weighted source (a prior or model assumption that
//! changed) re-weights them; every other stage keeps its stored numbers bit
//! for bit. The chain it was called on is not modified.
//!
//! # What a value means
//!
//! A value under a licensed chain is labelled `point_only`; its spread is
//! `uncalibrated_spread`; posterior draws are `posterior_propagation` only
//! when every source is posterior draws; an enclosure is
//! `propagation_enclosure`. No label is a coverage statement.
//! [`PropagationChain::interval`] refuses without a measured coverage record
//! at the composed coordinate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::MEASURE_COMPOSED_COVERAGE;
pub use antecedent_contract::propagation_chain::{
    ChainPlan, ChainReading, ChainRefusal, LawMeaning, NEXT_ATTEMPTS, StageKind, StagePlan,
    TableConstruction,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
pub use antecedent_io::propagation_chain_record::{
    ChainChangeWire, ChainCoefficientWire, ChainFactorWire, ChainOpWire, ChainSourceWire,
    ChainStageResultWire, ChainStageWire, ChainUnaryWire, ConsumedPropagationChain,
    PROPAGATION_CHAIN_SECTION, PropagationChainError, PropagationChainRecordWire, STATUS_CURRENT,
    STATUS_INVALIDATED, STATUS_REWEIGHTED,
};
use std::sync::Arc;

use super::lowering::chain::{declaration_identity, lower_chain};
use crate::planner::lowered::{ChainProgram, LoweredProgram};
use crate::planner::{PlanStage, PlanView};

fn refusal(error: &PropagationChainError) -> ChainRefusal {
    error.to_refusal()
}

/// A compiled chain: the lowered program and the chain it was lowered from.
///
/// Compile checks the chain rule and lowers all six layers; nothing is computed.
/// [`Self::execute`] seals the record the program carries.
#[derive(Clone, Debug)]
pub struct PropagationChainPlan {
    lowered: Arc<LoweredProgram>,
    program: ChainProgram,
}

/// Compile a declared chain.
///
/// `draft` carries the sources, stages, future-study draws and any appended
/// changes; its results, label and identity are ignored and recomputed.
///
/// # Errors
///
/// The typed refusal of the chain rule, with its registered code and next
/// attempt; `public_builder_reentry` inside a lowered dispatch; cancellation.
pub fn compile_propagation_chain(
    mut draft: PropagationChainRecordWire,
    ctx: &ExecutionContext,
) -> Result<PropagationChainPlan, ChainRefusal> {
    if ctx.lowered_dispatch().is_some() {
        return Err(ChainRefusal::new(
            antecedent_core::reason_code!("public_builder_reentry"),
            "propagation_chain.reentry",
            "a lowered dispatch executes its retained program; it does not compile another chain",
            None,
            antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION,
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ChainRefusal::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "propagation_chain.cancelled",
            "compile was cancelled; cancellation is not a verdict on the chain",
            None,
            antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION,
        ));
    }
    draft.version = antecedent_io::propagation_chain_record::PROPAGATION_CHAIN_VERSION;
    draft.required_features =
        vec![antecedent_io::propagation_chain_record::PROPAGATION_CHAIN_FEATURE.to_owned()];
    draft.results.clear();
    draft.identity.clear();
    draft.label.clear();
    let plan = draft.plan().map_err(|error| refusal(&error))?;
    let lowered = lower_chain(draft, plan)?;
    let program = lowered.propagation_chain().cloned().ok_or_else(|| {
        ChainRefusal::new(
            antecedent_core::reason_code!("route_not_lowered"),
            "propagation_chain.not_lowered",
            "the chain lowered without a program",
            None,
            antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION,
        )
    })?;
    Ok(PropagationChainPlan { lowered: Arc::new(lowered), program })
}

impl PropagationChainPlan {
    /// The lowered program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// One view over every layer of the chain's program. Reading it computes nothing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Identity of the lowered program.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        self.lowered.identity()
    }

    /// What the chain rule decided for every stage.
    #[must_use]
    pub const fn plan(&self) -> &ChainPlan {
        self.program.plan()
    }

    /// The stages that depend on `source`: the ones a change to it reaches.
    #[must_use]
    pub fn dependents_of(&self, source: &str) -> Vec<&str> {
        self.program.dependents_of(source)
    }

    /// Execute the retained program: evaluate every stage on the aligned table and
    /// seal the record, inside the program's dispatch scope.
    ///
    /// The sealed record's identity is the declaration identity the program was
    /// lowered with, so a result is always the record of its own program.
    ///
    /// # Errors
    ///
    /// The typed refusal of the arithmetic (a singular pool, a transformation
    /// leaving its domain, updated draws too few to represent the posterior),
    /// or cancellation.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<PropagationChain, ChainRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.identity());
        let record = PropagationChainRecordWire::seal(self.program.draft().clone(), &scoped)
            .map_err(|error| refusal(&error))?;
        if record.identity != self.program.declaration_identity() {
            return Err(ChainRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "propagation_chain.program_mismatch",
                "the sealed record is not the one the program was lowered from",
                None,
                antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION,
            ));
        }
        Ok(PropagationChain { record, program_identity: self.identity() })
    }

    /// Compile the chain with `change` appended: a withdrawn source, or re-weighted
    /// rows. This plan is not modified.
    ///
    /// # Errors
    ///
    /// As [`compile_propagation_chain`].
    pub fn with_change(
        &self,
        change: ChainChangeWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, ChainRefusal> {
        let mut draft = self.program.draft().clone();
        draft.changes.push(change);
        compile_propagation_chain(draft, ctx)
    }
}

/// A sealed propagation chain: the checked declaration, the draws, the changes
/// appended so far and the results of every stage, stamped with the identity of
/// the lowered program that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagationChain {
    record: PropagationChainRecordWire,
    program_identity: SemanticDigest,
}

impl PropagationChain {
    /// Compile `draft` and execute the program: a thin wrapper over
    /// [`compile_propagation_chain`] and [`PropagationChainPlan::execute`].
    ///
    /// # Errors
    ///
    /// The typed refusal of the chain rule or of the arithmetic, with its
    /// registered code and next attempt.
    pub fn seal(
        draft: PropagationChainRecordWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, ChainRefusal> {
        compile_propagation_chain(draft, ctx)?.execute(ctx)
    }

    /// Start a draft chain to add sources and stages to.
    #[must_use]
    pub fn draft(
        construction: TableConstruction,
        reading: ChainReading,
        output: &str,
        sources: Vec<ChainSourceWire>,
        stages: Vec<ChainStageWire>,
    ) -> PropagationChainRecordWire {
        PropagationChainRecordWire::draft(construction, reading, output, sources, stages)
    }

    /// The sealed identity of the record.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.record.identity
    }

    /// The identity of the lowered program that produced this chain.
    #[must_use]
    pub const fn lowered_program_identity(&self) -> SemanticDigest {
        self.program_identity
    }

    /// The label of the answer.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.record.label
    }

    /// The retained record.
    #[must_use]
    pub const fn record(&self) -> &PropagationChainRecordWire {
        &self.record
    }

    /// The stored result of a stage.
    #[must_use]
    pub fn stage(&self, id: &str) -> Option<&ChainStageResultWire> {
        self.record.results.iter().find(|result| result.stage == id)
    }

    /// The stored result of the stage whose output is the answer.
    #[must_use]
    pub fn answer(&self) -> Option<&ChainStageResultWire> {
        self.stage(&self.record.output)
    }

    /// The results of every stage, in stage order.
    #[must_use]
    pub fn results(&self) -> &[ChainStageResultWire] {
        &self.record.results
    }

    /// The plan the chain rule yielded.
    ///
    /// # Errors
    ///
    /// A refusal of the rule; a sealed chain does not have one.
    pub fn plan(&self) -> Result<ChainPlan, ChainRefusal> {
        self.record.plan().map_err(|error| refusal(&error))
    }

    /// The stages that depend on `source`: the ones a change to it reaches.
    ///
    /// # Errors
    ///
    /// A refusal of the rule; a sealed chain does not have one.
    pub fn dependents_of(&self, source: &str) -> Result<Vec<String>, ChainRefusal> {
        self.record.dependents_of(source).map_err(|error| refusal(&error))
    }

    fn changed(
        &self,
        change: ChainChangeWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, ChainRefusal> {
        let mut draft = self.record.clone();
        draft.changes.push(change);
        Self::seal(draft, ctx)
    }

    /// Withdraw `source`: every stage that depends on it is invalidated and
    /// carries no number; the rest keep their stored numbers. The change is
    /// compiled and executed as a new program.
    ///
    /// # Errors
    ///
    /// A source the chain does not have, or one already withdrawn.
    pub fn withdraw(&self, source: &str, ctx: &ExecutionContext) -> Result<Self, ChainRefusal> {
        self.changed(ChainChangeWire::withdraw(source), ctx)
    }

    /// Re-weight the rows of `source` by `log_weights` (an assumption about it
    /// changed): every stage that depends on it is summarized under the new
    /// weights; the rest keep their stored numbers. The change is compiled and
    /// executed as a new program.
    ///
    /// # Errors
    ///
    /// A source that is not draws, one that is withdrawn, or log weights that
    /// are not one finite value per row of the draw table.
    pub fn reweight(
        &self,
        source: &str,
        log_weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Self, ChainRefusal> {
        self.changed(ChainChangeWire::reweight(source, log_weights), ctx)
    }

    /// An interval of the composed quantity. Refused: it needs a measured
    /// coverage record at the composed coordinate, and a component interval
    /// does not cover the composition. Read the spread meanwhile.
    ///
    /// # Errors
    ///
    /// Always `uncertainty_representation_insufficient`.
    pub fn interval(&self) -> Result<(), ChainRefusal> {
        Err(ChainRefusal::new(
            antecedent_core::reason_code!("uncertainty_representation_insufficient"),
            "propagation_chain.interval_not_measured",
            "an interval of a composed quantity needs a measured coverage record at its composed coordinate",
            Some(&self.record.output),
            MEASURE_COMPOSED_COVERAGE,
        ))
    }

    /// Whether the answer is posterior draws of the output quantity.
    ///
    /// # Errors
    ///
    /// A chain not read as a posterior: the draws are a sampling law, or the
    /// chain carries intervals or points.
    pub fn posterior(&self) -> Result<(), ChainRefusal> {
        if self.record.reading == ChainReading::Posterior.as_str() {
            Ok(())
        } else {
            Err(ChainRefusal::new(
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                "propagation_chain.not_read_as_posterior",
                "the chain was not sealed with a posterior reading",
                Some(&self.record.output),
                antecedent_contract::propagation_chain::SUPPLY_POSTERIOR_DRAWS,
            ))
        }
    }

    /// Encode the retained record.
    ///
    /// # Errors
    ///
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ChainRefusal> {
        self.record.export().map_err(|error| refusal(&error))
    }

    /// The declaration identity a program lowered from this chain's record
    /// carries: the record identity, by construction.
    ///
    /// # Errors
    ///
    /// A refusal of the rule; a sealed chain does not have one.
    pub fn declaration_identity(&self) -> Result<String, ChainRefusal> {
        declaration_identity(&self.record, &self.plan()?)
    }
}

/// Decode and independently consume an exported record: check the rule again,
/// replay every stage and accept only results that equal the stored bits.
///
/// The consumer takes only bytes; it does not compile or read a lowered program.
/// `expected_identity`, when given, is the identity the caller retained
/// independently of the bytes.
///
/// # Errors
///
/// A record that does not decode, a foreign version, an identity or result
/// that does not replay, or a refusal of the rule.
pub fn consume_propagation_chain(
    bytes: &[u8],
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedPropagationChain, ChainRefusal> {
    PropagationChainRecordWire::decode(bytes)
        .and_then(|record| record.consume(expected_identity, ctx))
        .map_err(|error| refusal(&error))
}

/// The independent product of separately drawn columns as one aligned table:
/// the build for independent sources the refusal of unaligned draws names.
///
/// # Errors
///
/// No columns, an empty or non-finite column, or a product larger than
/// `max_rows`.
pub fn independent_product_table(
    columns: &[Vec<f64>],
    max_rows: usize,
) -> Result<Vec<Vec<f64>>, ChainRefusal> {
    antecedent_stats::independent_product_table(columns, max_rows).map_err(|error| {
        ChainRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            "propagation_chain.product_table_refused",
            error.to_string(),
            None,
            antecedent_contract::propagation_chain::RAISE_BUDGET_OR_THIN_DRAWS,
        )
    })
}

#[cfg(test)]
#[path = "propagation_chain_tests.rs"]
mod tests;
