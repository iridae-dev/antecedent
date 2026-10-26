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
    ChainChangeWire, ChainCoefficientWire, ChainExprTermWire, ChainFactorWire, ChainOpWire,
    ChainSourceWire, ChainStageResultWire, ChainStageWire, ChainUnaryWire,
    ConsumedPropagationChain, PROPAGATION_CHAIN_SECTION, PropagationChainError,
    PropagationChainRecordWire, STATUS_CURRENT, STATUS_INVALIDATED, STATUS_REWEIGHTED,
};
use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::dependence::{
    DependenceContract, DependenceError, DependenceSource, SharedFactorSets,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, LedgerError, SourceId, SourceStatus};

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

// ------------------------------------------------------------ family adapters

/// Most estimates one family table may join.
pub const MAX_FAMILY_ESTIMATES: usize = 64;

fn family_refusal(
    detail: &'static str,
    message: impl Into<String>,
    offending: Option<&str>,
) -> ChainRefusal {
    ChainRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        message,
        offending,
        antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION,
    )
}

fn dependence_refusal(error: &DependenceError) -> ChainRefusal {
    ChainRefusal::new(
        error.code(),
        "propagation_chain.family_dependence",
        format!("{}: {}", error.detail(), error.message()),
        None,
        antecedent_contract::propagation_chain::DECLARE_SHARED_FACTORS,
    )
}

fn authority_refusal(error: &LedgerError, source: &str) -> ChainRefusal {
    ChainRefusal::new(
        error.code,
        "propagation_chain.family_authority",
        error.message.clone(),
        Some(source),
        antecedent_contract::propagation_chain::RERUN_WITHOUT_WITHDRAWN_SOURCE,
    )
}

/// One estimate a method family produced: the number, its standard error, the
/// law it is read as, the evidence it used and, when the caller holds one, the
/// ledger source that authorizes it.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyEstimate {
    /// Identity of the chain source this becomes.
    pub id: String,
    /// The estimate.
    pub estimate: f64,
    /// Its standard error, from the family's own uncertainty procedure.
    pub standard_error: f64,
    /// Whether the uncertainty is a sampling law or a posterior.
    pub meaning: LawMeaning,
    /// Evidence factors the estimate used.
    pub factors: Vec<(SharedFactorKind, String)>,
    /// Whether the factor list may be partial.
    pub unknown_overlap: bool,
    /// The ledger source authorizing the estimate.
    pub authority: Option<SourceId>,
}

impl FamilyEstimate {
    /// An estimate that declares no factors and no authority yet.
    #[must_use]
    pub fn new(id: &str, estimate: f64, standard_error: f64, meaning: LawMeaning) -> Self {
        Self {
            id: id.to_owned(),
            estimate,
            standard_error,
            meaning,
            factors: Vec::new(),
            unknown_overlap: false,
            authority: None,
        }
    }

    /// Declare that the estimate used a factor.
    #[must_use]
    pub fn with_factor(mut self, kind: SharedFactorKind, id: &str) -> Self {
        self.factors.push((kind, id.to_owned()));
        self
    }

    /// Declare that the factor list may be partial.
    #[must_use]
    pub fn with_unknown_overlap(mut self) -> Self {
        self.unknown_overlap = true;
        self
    }

    /// Bind the estimate to the ledger source that authorizes it.
    #[must_use]
    pub fn with_authority(mut self, source: SourceId) -> Self {
        self.authority = Some(source);
        self
    }
}

/// How the estimates of a family table depend on one another.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FamilyDependence {
    /// Claimed independent. Refused when the declared factors share evidence or
    /// leave the overlap unknown.
    Independent,
    /// A declared joint covariance, row-major in estimate order. Its diagonal is
    /// the squared standard errors; a nonzero off-diagonal term names shared
    /// evidence through a declared factor, and a shared factor carries a nonzero
    /// term.
    Covariance(Vec<f64>),
    /// Read from the ledger: the rows two authorizing sources share give the
    /// correlation `shared / sqrt(rows_a rows_b)` of equal-weight mean-type
    /// estimators, and a row factor naming the shared rows is attached to both
    /// chain sources so the chain rule sees the overlap.
    LedgerOverlap,
}

/// Chain sources bound to the ledger sources that authorize them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChainAuthority {
    bindings: Vec<(String, SourceId)>,
}

impl ChainAuthority {
    /// Chain source identity and the ledger source behind it.
    #[must_use]
    pub fn bindings(&self) -> &[(String, SourceId)] {
        &self.bindings
    }

    /// The bound chain sources whose ledger source is no longer the current,
    /// active one, with the reason.
    #[must_use]
    pub fn retired(&self, ledger: &EvidenceLedger) -> Vec<(String, String)> {
        self.bindings
            .iter()
            .filter_map(|(chain_source, ledger_source)| {
                let reason = match ledger.source(*ledger_source).map(|source| &source.status) {
                    None => "absent from the ledger".to_owned(),
                    Some(SourceStatus::Active) => return None,
                    Some(SourceStatus::Withdrawn { reason, .. }) => format!("withdrawn: {reason}"),
                    Some(SourceStatus::Superseded { .. }) => "superseded".to_owned(),
                    Some(SourceStatus::Replaced { .. }) => "replaced".to_owned(),
                    Some(_) => "no longer the current revision".to_owned(),
                };
                Some((chain_source.clone(), reason))
            })
            .collect()
    }

    /// Bring `chain` up to the ledger: every bound source the ledger no longer
    /// holds as active is withdrawn from the chain, so exactly the stages that
    /// depend on it lose their numbers and the rest keep theirs bit for bit.
    /// Sources the chain already withdrew are not withdrawn twice.
    ///
    /// # Errors
    /// A refusal of the withdrawal.
    pub fn apply(
        &self,
        chain: &PropagationChain,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PropagationChain, ChainRefusal> {
        let mut current = chain.clone();
        for (source, _reason) in self.retired(ledger) {
            let already = current.record().changes.iter().any(
                |change| matches!(change, ChainChangeWire::Withdraw { source: named } if *named == source),
            );
            if !already {
                current = current.withdraw(&source, ctx)?;
            }
        }
        Ok(current)
    }
}

/// The chain sources a family produced, the covariance they carry and the
/// authority behind them.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyTable {
    /// One aligned draw column per estimate, with its declared factors.
    pub sources: Vec<ChainSourceWire>,
    /// Row-major joint covariance the draws carry exactly.
    pub covariance: Vec<f64>,
    /// The ledger sources behind the chain sources.
    pub authority: ChainAuthority,
}

impl FamilyTable {
    /// Draft a joint-draw chain over these sources.
    #[must_use]
    pub fn draft(
        &self,
        reading: ChainReading,
        output: &str,
        stages: Vec<ChainStageWire>,
    ) -> PropagationChainRecordWire {
        PropagationChain::draft(
            TableConstruction::JointDraws,
            reading,
            output,
            self.sources.clone(),
            stages,
        )
    }
}

fn dependence_source(
    estimate: &FamilyEstimate,
    derived: &[(SharedFactorKind, String)],
) -> Result<DependenceSource, ChainRefusal> {
    let mut sets = SharedFactorSets::empty();
    for (kind, id) in estimate.factors.iter().chain(derived) {
        sets.try_insert(*kind, id).map_err(|error| dependence_refusal(&error))?;
    }
    let built = if estimate.unknown_overlap {
        DependenceSource::try_with_unknown_overlap(&estimate.id, sets)
    } else {
        DependenceSource::try_declared(&estimate.id, sets)
    };
    built.map_err(|error| dependence_refusal(&error))
}

/// Turn family estimates into one aligned joint-draw table with the dependence
/// they share.
///
/// Each estimate becomes a chain source whose draws are normal with the
/// estimate's mean and standard error. The draws carry the declared joint
/// covariance exactly, so a pool, a derived quantity or a nonlinear utility
/// downstream sees the shared evidence once, in the joint law, and the chain
/// rule sees the same evidence as declared factors.
///
/// # Errors
/// An estimate that is not finite or has a negative standard error; duplicate
/// or blank ids; an independence claim the factors do not support; a
/// covariance that disagrees with the standard errors or with the declared
/// factors; a ledger overlap request without authorizing sources, with a
/// retired source or with undeclared observations; a table the arithmetic
/// refuses.
pub fn family_table(
    alignment: &str,
    estimates: &[FamilyEstimate],
    dependence: &FamilyDependence,
    ledger: Option<&EvidenceLedger>,
    rows: usize,
    seed: u64,
) -> Result<FamilyTable, ChainRefusal> {
    let k = estimates.len();
    if k == 0 || k > MAX_FAMILY_ESTIMATES {
        return Err(family_refusal(
            "propagation_chain.family_shape",
            "a family table joins between one and 64 estimates",
            None,
        ));
    }
    let mut seen = BTreeSet::new();
    for estimate in estimates {
        if !seen.insert(estimate.id.as_str()) || estimate.id.trim().is_empty() {
            return Err(family_refusal(
                "propagation_chain.family_shape",
                "each estimate names a distinct, non-blank source",
                Some(&estimate.id),
            ));
        }
        if !estimate.estimate.is_finite()
            || !estimate.standard_error.is_finite()
            || estimate.standard_error < 0.0
        {
            return Err(family_refusal(
                "propagation_chain.family_estimate",
                "an estimate is finite with a finite non-negative standard error",
                Some(&estimate.id),
            ));
        }
    }
    let standard_error: Vec<f64> = estimates.iter().map(|e| e.standard_error).collect();
    let mut covariance = vec![0.0; k * k];
    for i in 0..k {
        covariance[i * k + i] = standard_error[i] * standard_error[i];
    }
    let mut derived: Vec<Vec<(SharedFactorKind, String)>> = vec![Vec::new(); k];
    match dependence {
        FamilyDependence::Independent => {}
        FamilyDependence::Covariance(matrix) => {
            if matrix.len() != k * k || matrix.iter().any(|value| !value.is_finite()) {
                return Err(family_refusal(
                    "propagation_chain.family_covariance",
                    "a declared covariance is a finite k by k matrix",
                    None,
                ));
            }
            for i in 0..k {
                let declared = matrix[i * k + i];
                let published = covariance[i * k + i];
                if (declared - published).abs() > 1e-9 * published.max(declared.abs()).max(1e-300) {
                    return Err(family_refusal(
                        "propagation_chain.family_covariance",
                        "the covariance diagonal is the squared standard error the family published",
                        Some(&estimates[i].id),
                    ));
                }
            }
            covariance.clone_from(matrix);
        }
        FamilyDependence::LedgerOverlap => {
            apply_family_ledger_overlap(
                estimates,
                ledger,
                &standard_error,
                &mut covariance,
                &mut derived,
            )?;
        }
    }
    let sources: Vec<DependenceSource> = estimates
        .iter()
        .zip(&derived)
        .map(|(estimate, extra)| dependence_source(estimate, extra))
        .collect::<Result<_, _>>()?;
    check_family_dependence(estimates, &sources, dependence, &standard_error, &covariance)?;
    let means: Vec<f64> = estimates.iter().map(|e| e.estimate).collect();
    let table = antecedent_stats::moment_matched_normal_table(&means, &covariance, rows, seed)
        .map_err(|error| {
            ChainRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "propagation_chain.family_table_refused",
                error.to_string(),
                None,
                antecedent_contract::propagation_chain::RAISE_BUDGET_OR_THIN_DRAWS,
            )
        })?;
    let mut wires = Vec::with_capacity(k);
    let mut bindings = Vec::new();
    for ((estimate, column), extra) in estimates.iter().zip(&table).zip(&derived) {
        let mut wire = ChainSourceWire::draws(&estimate.id, estimate.meaning, alignment, column);
        for (kind, id) in estimate.factors.iter().chain(extra) {
            wire = wire.with_factor(*kind, id);
        }
        if estimate.unknown_overlap {
            wire = wire.with_unknown_overlap();
        }
        wires.push(wire);
        if let Some(source) = estimate.authority {
            bindings.push((estimate.id.clone(), source));
        }
    }
    Ok(FamilyTable { sources: wires, covariance, authority: ChainAuthority { bindings } })
}

impl PropagationChain {
    /// A stored value of a stage, for a decision or composition consumer.
    ///
    /// A stage that depends on a withdrawn source carries no number, and this
    /// refuses it and names the sources; a re-weighted stage returns its
    /// re-weighted number.
    ///
    /// # Errors
    /// An unknown stage or value, or an invalidated stage.
    pub fn require_current_value(&self, stage: &str, name: &str) -> Result<f64, ChainRefusal> {
        let result = self.stage(stage).ok_or_else(|| {
            family_refusal(
                "propagation_chain.unknown_stage",
                "the chain has no such stage",
                Some(stage),
            )
        })?;
        if result.status == STATUS_INVALIDATED {
            return Err(ChainRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "propagation_chain.stage_invalidated",
                format!(
                    "the stage depends on withdrawn source(s) {}; it carries no number",
                    result.invalidated_by.join(", ")
                ),
                Some(stage),
                antecedent_contract::propagation_chain::RERUN_WITHOUT_WITHDRAWN_SOURCE,
            ));
        }
        result.value(name).ok_or_else(|| {
            family_refusal(
                "propagation_chain.unknown_value",
                "the stage stores no such value",
                Some(stage),
            )
        })
    }
}

/// Derive covariance and shared row factors from the authorizing ledger, in estimate order.
fn apply_family_ledger_overlap(
    estimates: &[FamilyEstimate],
    ledger: Option<&EvidenceLedger>,
    standard_error: &[f64],
    covariance: &mut [f64],
    derived: &mut [Vec<(SharedFactorKind, String)>],
) -> Result<(), ChainRefusal> {
    let k = estimates.len();
    let ledger = ledger.ok_or_else(|| {
        family_refusal(
            "propagation_chain.family_authority",
            "a ledger overlap needs the ledger that holds the authorizing sources",
            None,
        )
    })?;
    let mut rows_used = Vec::with_capacity(k);
    for estimate in estimates {
        let id = estimate.authority.ok_or_else(|| {
            family_refusal(
                "propagation_chain.family_authority",
                "a ledger overlap needs the authorizing source of every estimate",
                Some(&estimate.id),
            )
        })?;
        let source = ledger.source(id).ok_or_else(|| {
            family_refusal(
                "propagation_chain.family_authority",
                "the authorizing source is not in the ledger",
                Some(&estimate.id),
            )
        })?;
        if !matches!(source.status, SourceStatus::Active) {
            return Err(ChainRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "propagation_chain.family_authority",
                "a retired source cannot seed a chain",
                Some(&estimate.id),
                antecedent_contract::propagation_chain::RERUN_WITHOUT_WITHDRAWN_SOURCE,
            ));
        }
        let known = source.effective.known().ok_or_else(|| {
            ChainRefusal::new(
                antecedent_core::reason_code!("evidence_overlap_unknown"),
                "propagation_chain.family_dependence",
                "the authorizing source declares no observations, so its overlap is unknown",
                Some(&estimate.id),
                antecedent_contract::propagation_chain::DECLARE_SHARED_FACTORS,
            )
        })?;
        rows_used.push(known.len());
    }
    for i in 0..k {
        for j in (i + 1)..k {
            let (a, b) = (estimates[i].authority, estimates[j].authority);
            let (Some(a), Some(b)) = (a, b) else { continue };
            let shared = ledger
                .shared_observations(a, b)
                .map_err(|error| authority_refusal(&error, &estimates[i].id))?
                .len();
            let correlation =
                antecedent_stats::shared_row_correlation(rows_used[i], rows_used[j], shared)
                    .map_err(|error| {
                        family_refusal(
                            "propagation_chain.family_covariance",
                            error.to_string(),
                            Some(&estimates[i].id),
                        )
                    })?;
            let term = correlation * standard_error[i] * standard_error[j];
            covariance[i * k + j] = term;
            covariance[j * k + i] = term;
            if shared > 0 {
                let (low, high) = if a.to_hex() <= b.to_hex() {
                    (a.to_hex(), b.to_hex())
                } else {
                    (b.to_hex(), a.to_hex())
                };
                let factor = format!("ledger-rows:{low}:{high}");
                derived[i].push((SharedFactorKind::Row, factor.clone()));
                derived[j].push((SharedFactorKind::Row, factor));
            }
        }
    }
    Ok(())
}

/// Check the declared covariance against the same dependence factors used by the chain rule.
fn check_family_dependence(
    estimates: &[FamilyEstimate],
    sources: &[DependenceSource],
    dependence: &FamilyDependence,
    standard_error: &[f64],
    covariance: &[f64],
) -> Result<(), ChainRefusal> {
    let k = estimates.len();
    match dependence {
        FamilyDependence::Independent => {
            DependenceContract::try_claim_independent(sources)
                .map_err(|error| dependence_refusal(&error))?;
        }
        FamilyDependence::Covariance(_) => {
            for i in 0..k {
                for j in (i + 1)..k {
                    let pair =
                        DependenceContract::assess(&[sources[i].clone(), sources[j].clone()])
                            .map_err(|error| dependence_refusal(&error))?;
                    let shares = !pair.shared_factors().is_empty();
                    let term = covariance[i * k + j];
                    let live = standard_error[i] > 0.0 && standard_error[j] > 0.0;
                    if live && term != 0.0 && !shares {
                        return Err(ChainRefusal::new(
                            antecedent_core::reason_code!("evidence_overlap_unknown"),
                            "propagation_chain.family_dependence",
                            "a nonzero covariance names the evidence the two estimates share; declare the shared factor",
                            Some(&estimates[i].id),
                            antecedent_contract::propagation_chain::DECLARE_SHARED_FACTORS,
                        ));
                    }
                    if live && term == 0.0 && shares {
                        return Err(ChainRefusal::new(
                            antecedent_core::reason_code!("evidence_double_counted"),
                            "propagation_chain.family_dependence",
                            "two estimates share declared evidence but the covariance counts it as independent",
                            Some(&estimates[i].id),
                            antecedent_contract::propagation_chain::POOL_WITH_JOINT_COVARIANCE,
                        ));
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "propagation_chain_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "propagation_chain_family_tests.rs"]
mod family_tests;
