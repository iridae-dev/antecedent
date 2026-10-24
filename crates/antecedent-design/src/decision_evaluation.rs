//! Decision evaluation against the source demand a contract already declares.
//!
//! [`crate::decision_eval::evaluate_contract_on_means`] classifies that demand.
//! This module does not classify it again. It names the missing predictions or
//! draws on the refusal that classification already returned. The expectation of
//! an affine utility is the utility of the means, so that evaluation requests
//! neither predictions nor draws.
//!
//! [`BatchRecord`] and [`RetentionRecord`] are records of what was requested.
//! They are not a thread pool, not a scratch pool, and not a fit. Nothing here
//! forms a posterior: a mean is not a belief, and an identified set is not
//! given probabilities.
//!
//! [`execute_demand`] runs the demand against a [`DemandSource`] under a
//! [`DemandBudget`]. The contract's own classification chooses the carrier: a
//! demand a mean answers asks the source for means once and never touches its
//! draws. A demand that needs draws asks for them in batches sized from the
//! scratch budget, folds each batch into per-action utilities and constraint
//! tallies in one reused scratch buffer, checks the retained utilities against
//! the retention budget before allocating them, and ranks through the same
//! table as the joint-law evaluation. The record it returns is measured: the
//! batches issued, the rows per batch, and the scratch and retained bytes used.
//!
//! Scoring these declared actions does not build a
//! [`crate::decision_contract::ProgramSearchContract`] and does not search.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::{CancellationToken, ExecutionContext};
use thiserror::Error;

use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DrawAlignment,
};

use crate::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion, SourceRepresentation,
};
use crate::decision_eval::{
    ConstraintExclusion, DecisionEvalError, DecisionResult, MeanEstimate, MeanSource,
    SourceReceipt, Table, constraint_fails, evaluate_contract_on_means, rank_table,
};

const ONLY_PREDICTIONS: &[MissingCarrier] = &[MissingCarrier::Predictions];
const ONLY_DRAWS: &[MissingCarrier] = &[MissingCarrier::Draws];
const PREDICTIONS_OR_DRAWS: &[MissingCarrier] =
    &[MissingCarrier::Predictions, MissingCarrier::Draws];

/// What an evaluation asked a source to supply.
///
/// A mean evaluation asks for [`Self::Means`] only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestedCarrier {
    /// Means of the coordinates the utilities read.
    Means,
    /// A predictive law or predictive simulation. Not draws, and not a posterior.
    Predictions,
    /// Marginal or joint draws. Not a posterior.
    Draws,
}

impl RequestedCarrier {
    /// Stable name: `means`, `predictions`, or `draws`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Means => "means",
            Self::Predictions => "predictions",
            Self::Draws => "draws",
        }
    }
}

/// A carrier that means do not provide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingCarrier {
    /// A predictive law or predictive simulation.
    Predictions,
    /// Marginal or joint draws.
    Draws,
}

impl MissingCarrier {
    /// Stable name: `predictions` or `draws`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Predictions => "predictions",
            Self::Draws => "draws",
        }
    }
}

/// The batch the evaluation asked for.
///
/// One record, one carrier. `batches` is how many batches that request is,
/// and `items` is how many distinct coordinates it holds. Shared coordinates
/// are one item. This admits no worker and allocates no scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchRecord {
    /// Carrier requested.
    pub carrier: RequestedCarrier,
    /// Batches in the request. A mean evaluation is one batch.
    pub batches: usize,
    /// Distinct coordinates in the batch.
    pub items: usize,
}

impl BatchRecord {
    /// One batch of `items` means.
    #[must_use]
    pub const fn means(items: usize) -> Self {
        Self { carrier: RequestedCarrier::Means, batches: 1, items }
    }
}

/// What the evaluation asked to keep.
///
/// [`Self::means_summary`] keeps the mean summary. `fits` is zero: retaining
/// that summary is not a fit, and it does not retain predictions or draws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionRecord {
    /// Retention mode of the request.
    pub mode: OutputRetention,
    /// Fits requested. Zero for a mean summary.
    pub fits: usize,
}

impl RetentionRecord {
    /// Summary of the means. Not a fit and not a draw store.
    #[must_use]
    pub const fn means_summary() -> Self {
        Self { mode: OutputRetention::SummaryOnly, fits: 0 }
    }
}

/// An affine decision scored from means, with the request that scoring made.
#[derive(Clone, Debug, PartialEq)]
pub struct DemandEvaluation {
    /// Decision under the mean source. No draws, no regret, no posterior.
    pub decision: DecisionResult,
    /// Batch that was requested.
    pub batching: BatchRecord,
    /// Retention that was requested.
    pub retention: RetentionRecord,
    /// Predictive simulations requested. Zero when means answer the utility.
    pub predictions_requested: usize,
    /// Draws requested. Zero when means answer the utility.
    pub draws_requested: usize,
}

/// Why a mean evaluation refused.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    /// The contract is invalid.
    #[error("decision contract refused: {0}")]
    Contract(DecisionContractError),
    /// The demand accepts means, and the mean source itself refused.
    #[error("decision evaluation refused: {0}")]
    Evaluation(DecisionEvalError),
    /// Means cannot answer the declared demand.
    ///
    /// `missing_label` is `predictions`, `draws`, or `predictions or draws`.
    #[error("means alone cannot answer this demand; missing {missing_label}")]
    MeansInsufficient {
        /// Carriers the demand still needs.
        missing: &'static [MissingCarrier],
        /// Source representations that would have sufficed.
        needed: Vec<SourceRepresentation>,
        /// Words the refusal uses for `missing`.
        missing_label: &'static str,
    },
    /// The demand is an identified set or a set of structures.
    ///
    /// Means are not that set, and this refusal does not put a probability on it.
    #[error("means alone are not an identified set or a set of structures")]
    IdentifiedSet,
    /// The source cannot supply any carrier that suffices for the demand.
    #[error("the source supplies none of the representations the demand needs")]
    CarrierUnavailable {
        /// Representations that would have sufficed.
        needed: Vec<SourceRepresentation>,
    },
    /// One draw row does not fit the scratch budget.
    #[error("one batch row needs {needed} scratch bytes; the budget is {budget}")]
    ScratchTooSmall {
        /// Bytes of one row of draws.
        needed: u64,
        /// Scratch bytes granted.
        budget: u64,
    },
    /// The per-action utilities do not fit the retention budget.
    #[error("the utilities need {needed} retained bytes; the budget is {budget}")]
    RetentionExceeded {
        /// Bytes the retained utilities and weights need.
        needed: u64,
        /// Retained bytes granted.
        budget: u64,
    },
    /// This executor retains full reusable state or a summary only.
    #[error("retention {mode} is not offered by the demand executor")]
    RetentionUnsupported {
        /// Mode that was asked for.
        mode: OutputRetention,
    },
    /// The source failed or returned an inconsistent batch.
    #[error("demand source failed: {message}")]
    Source {
        /// The source's message.
        message: String,
    },
    /// The shared admission pool could not grant the scratch and retained bytes.
    #[error("admission refused the demand's scratch and retained bytes: {message}")]
    Admission {
        /// The pool's refusal.
        message: String,
    },
    /// Cancellation was requested between batches.
    #[error("decision evaluation cancelled after {batches} batches")]
    Cancelled {
        /// Batches completed.
        batches: usize,
    },
}

/// Evaluate `contract` on `source` using the contract's declared source demand.
///
/// An affine expected utility or expected loss, with no hard constraint, is
/// scored from the means. The returned record requests no predictions and no
/// draws. Any other declared demand is refused before that scoring.
///
/// # Errors
///
/// [`EvaluationError::MeansInsufficient`] when the demand needs predictions or
/// draws. [`EvaluationError::IdentifiedSet`] when the demand is a set. Contract,
/// coordinate, meaning, and non-finite failures pass through after a demand that
/// does accept means.
#[must_use = "a demand that means cannot answer is refused"]
pub fn evaluate_on_means(
    contract: &DecisionContract,
    source: &MeanSource,
) -> Result<DemandEvaluation, EvaluationError> {
    let decision = evaluate_contract_on_means(contract, source).map_err(lift_eval)?;
    Ok(DemandEvaluation {
        decision,
        batching: BatchRecord::means(distinct_inputs(contract)),
        retention: RetentionRecord::means_summary(),
        predictions_requested: 0,
        draws_requested: 0,
    })
}

/// The source of a demanded carrier failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DemandSourceError {
    message: String,
}

impl DemandSourceError {
    /// Failure carrying `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for DemandSourceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DemandSourceError {}

/// A source that supplies the carriers a decision demand names, on request.
///
/// Nothing is produced until the executor asks. `means` is one call that returns one mean
/// per coordinate. `fill_draws` writes a bounded slice of the joint draws into a buffer the
/// executor owns, so the source never materializes the law.
pub trait DemandSource {
    /// Coordinates, in the order of a mean and of a draw row.
    fn coordinates(&self) -> &[ScientificQuantity];

    /// Lineage copied onto the result.
    fn receipt(&self) -> SourceReceipt;

    /// Whether `means` are posterior means or frequentist point estimates.
    fn mean_estimate(&self) -> MeanEstimate;

    /// One mean per coordinate.
    fn means(&mut self) -> Result<Vec<f64>, DemandSourceError>;

    /// Whether this source can supply means. A stored law supplies draws only.
    fn supplies_means(&self) -> bool {
        true
    }

    /// Whether this source can supply draws at all.
    fn supplies_draws(&self) -> bool;

    /// Whether the draw rows are paired joint realizations.
    fn joint(&self) -> bool;

    /// Whether the draws are an exact finite law, so values carry no sampling error.
    fn exact_law(&self) -> bool {
        false
    }

    /// Whether outcome-law inputs may read this source's draws.
    fn answers_outcome_law(&self) -> bool {
        false
    }

    /// Whether state-by-state comparison needs paired rows whatever the demand reads, as a
    /// stored joint law does.
    fn requires_joint_rows(&self) -> bool {
        false
    }

    /// Whether the coordinate at `column` is masked as unsupported.
    fn unsupported(&self, _column: usize) -> bool {
        false
    }

    /// Normalized draw weights, one per draw, when the draws are weighted. Unweighted draws
    /// weigh one over the draw count.
    fn weights(&self) -> Option<&[f64]> {
        None
    }

    /// Draws available.
    fn draw_count(&self) -> usize;

    /// Write draws `start..start + rows` row-major into `out`, which holds exactly
    /// `rows * coordinates().len()` values.
    fn fill_draws(
        &mut self,
        start: usize,
        rows: usize,
        out: &mut [f64],
    ) -> Result<(), DemandSourceError>;
}

/// Memory a demand may use, and what it keeps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DemandBudget {
    /// Bytes of the one reused buffer a batch of draws is written into.
    pub scratch_bytes: u64,
    /// Bytes the per-action utilities and weights may occupy.
    pub retained_bytes: u64,
    /// [`OutputRetention::SummaryOnly`] drops the utilities after ranking.
    /// [`OutputRetention::FullReusable`] returns them.
    pub retention: OutputRetention,
}

/// What an executed demand measured.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionRecord {
    /// Carrier the source was asked for.
    pub carrier: RequestedCarrier,
    /// Source calls issued: one for means, one per batch of draws.
    pub batches: usize,
    /// Distinct coordinates the actions read.
    pub items: usize,
    /// Rows in a full batch of draws. Zero for means.
    pub batch_rows: usize,
    /// Draws written by the source. Zero for means.
    pub draws_drawn: usize,
    /// Scratch bytes allocated.
    pub scratch_bytes: u64,
    /// Peak bytes of retained utilities and weights.
    pub retained_peak_bytes: u64,
    /// Bytes still retained when the call returned.
    pub retained_bytes: u64,
    /// Retention mode that applied.
    pub retention: OutputRetention,
    /// Predictive simulations requested. Zero: the executor asks for means or draws.
    pub predictions_requested: usize,
}

/// A demand run through a source under a budget.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutedDemand {
    /// The decision.
    pub decision: DecisionResult,
    /// What the execution measured.
    pub record: ExecutionRecord,
    /// Per-action utilities per draw, when the budget retained full reusable state.
    pub utilities: Option<Vec<Vec<f64>>>,
}

/// Execute the demand `contract` declares against `source` under `budget`.
///
/// The contract's source requirement decides the carrier. When a mean suffices the source is
/// asked for means once and its draws are never requested. Otherwise draws are requested in
/// batches of as many rows as the scratch budget holds, and the retained utilities are
/// checked against the retention budget before they are allocated.
///
/// # Errors
///
/// [`EvaluationError::IdentifiedSet`] for a set demand; [`EvaluationError::MeansInsufficient`]
/// when the demand needs draws and the source supplies none;
/// [`EvaluationError::ScratchTooSmall`] and [`EvaluationError::RetentionExceeded`] for a budget
/// that cannot hold one row or the utilities; [`EvaluationError::Cancelled`] between batches;
/// [`EvaluationError::Source`] when the source fails or returns a malformed batch; and the
/// contract, coordinate, meaning and non-finite refusals of the evaluation itself.
pub fn execute_demand(
    contract: &DecisionContract,
    source: &mut dyn DemandSource,
    budget: DemandBudget,
    cancellation: Option<&CancellationToken>,
) -> Result<ExecutedDemand, EvaluationError> {
    if !matches!(budget.retention, OutputRetention::FullReusable | OutputRetention::SummaryOnly) {
        return Err(EvaluationError::RetentionUnsupported { mode: budget.retention });
    }
    contract.validate().map_err(EvaluationError::Contract)?;
    let requirement = contract
        .validated_source_requirement()
        .map_err(EvaluationError::Contract)?
        .ok_or(EvaluationError::Contract(
        DecisionContractError::InvalidDeclaration("a decision compares at least two actions"),
    ))?;
    if requirement.is_identified_set() {
        return Err(EvaluationError::IdentifiedSet);
    }
    if requirement.any_of.contains(&SourceRepresentation::Mean) && source.supplies_means() {
        return execute_on_means(contract, source, budget);
    }
    let draws = requirement.any_of.iter().any(|representation| {
        matches!(
            representation,
            SourceRepresentation::MarginalDraws | SourceRepresentation::JointDraws
        )
    }) || (source.supplies_draws() && !source.supplies_means());
    if !draws {
        return Err(EvaluationError::CarrierUnavailable { needed: requirement.any_of });
    }
    if !source.supplies_draws() {
        return Err(means_insufficient(requirement.any_of));
    }
    let needs_joint = !requirement.any_of.contains(&SourceRepresentation::MarginalDraws);
    execute_on_draws(contract, source, budget, cancellation, needs_joint)
}

/// [`execute_demand`] with the draw path's scratch and retained bytes leased from the
/// context's shared admission pool for the length of the call, and the context's
/// cancellation token polled between batches.
///
/// A demand a mean answers leases nothing: it holds no scratch and retains no utilities.
///
/// # Errors
///
/// [`EvaluationError::Admission`] when the pool cannot grant the bytes, otherwise those of
/// [`execute_demand`].
pub fn execute_demand_admitted(
    contract: &DecisionContract,
    source: &mut dyn DemandSource,
    budget: DemandBudget,
    ctx: &ExecutionContext,
) -> Result<ExecutedDemand, EvaluationError> {
    contract.validate().map_err(EvaluationError::Contract)?;
    let by_mean = source.supplies_means()
        && contract
            .validated_source_requirement()
            .map_err(EvaluationError::Contract)?
            .is_some_and(|requirement| requirement.any_of.contains(&SourceRepresentation::Mean));
    let _lease = if by_mean {
        None
    } else {
        let bytes = budget.scratch_bytes.saturating_add(budget.retained_bytes);
        Some(
            ctx.acquire_admission(0, bytes)
                .map_err(|error| EvaluationError::Admission { message: error.to_string() })?,
        )
    };
    execute_demand(contract, source, budget, Some(&ctx.cancellation))
}

/// A mean source offered as a demand source: it supplies its means and no draws.
pub struct MeanDemand<'a> {
    source: &'a MeanSource,
}

impl<'a> MeanDemand<'a> {
    /// `source` as a demand source.
    #[must_use]
    pub const fn new(source: &'a MeanSource) -> Self {
        Self { source }
    }
}

impl DemandSource for MeanDemand<'_> {
    fn coordinates(&self) -> &[ScientificQuantity] {
        &self.source.coordinates
    }

    fn receipt(&self) -> SourceReceipt {
        SourceReceipt {
            provider_id: self.source.provider_id.clone(),
            snapshot_id: self.source.snapshot_id.clone(),
            rng_id: self.source.rng_id.clone(),
            causal_contract_id: self.source.causal_contract_id.clone(),
        }
    }

    fn mean_estimate(&self) -> MeanEstimate {
        self.source.estimate
    }

    fn means(&mut self) -> Result<Vec<f64>, DemandSourceError> {
        Ok(self.source.means.clone())
    }

    fn supplies_draws(&self) -> bool {
        false
    }

    fn joint(&self) -> bool {
        false
    }

    fn draw_count(&self) -> usize {
        0
    }

    fn fill_draws(
        &mut self,
        _start: usize,
        _rows: usize,
        _out: &mut [f64],
    ) -> Result<(), DemandSourceError> {
        Err(DemandSourceError::new("a mean source holds no draws"))
    }
}

/// A stored joint law offered as a demand source: its draws are read in batches, weighted by
/// its own weights, and it supplies no means.
pub struct ArtifactDemand<'a> {
    artifact: &'a DistributionArtifact,
    coordinates: Vec<ScientificQuantity>,
    columns: Vec<usize>,
    weights: Option<Vec<f64>>,
    outcome_law: bool,
}

impl<'a> ArtifactDemand<'a> {
    /// `artifact` as a demand source. A coordinate that does not parse as a scientific
    /// quantity is left out, as it is when the artifact is evaluated whole.
    #[must_use]
    pub fn new(artifact: &'a DistributionArtifact) -> Self {
        let mut coordinates = Vec::new();
        let mut columns = Vec::new();
        for (column, wire) in artifact.quantities().iter().enumerate() {
            if let Ok(quantity) = ScientificQuantity::try_from(wire.clone()) {
                coordinates.push(quantity);
                columns.push(column);
            }
        }
        let weights = artifact.metadata().weights.as_ref().map(|raw| {
            let total: f64 = raw.iter().sum();
            raw.iter().map(|weight| weight / total).collect()
        });
        let outcome_law = antecedent_contract::DistributionMeaning::from(artifact.semantic())
            .answers_interventional_outcome_threshold();
        Self { artifact, coordinates, columns, weights, outcome_law }
    }
}

impl DemandSource for ArtifactDemand<'_> {
    fn coordinates(&self) -> &[ScientificQuantity] {
        &self.coordinates
    }

    fn receipt(&self) -> SourceReceipt {
        let identity = &self.artifact.metadata().identity;
        SourceReceipt {
            provider_id: identity.provider_id.clone(),
            snapshot_id: identity.snapshot_id.clone(),
            rng_id: identity.rng_id.clone(),
            causal_contract_id: identity.causal_contract_id.clone(),
        }
    }

    fn mean_estimate(&self) -> MeanEstimate {
        MeanEstimate::Frequentist
    }

    fn means(&mut self) -> Result<Vec<f64>, DemandSourceError> {
        Err(DemandSourceError::new("a stored law is read through its draws"))
    }

    fn supplies_means(&self) -> bool {
        false
    }

    fn supplies_draws(&self) -> bool {
        true
    }

    fn joint(&self) -> bool {
        matches!(self.artifact.metadata().identity.alignment, DrawAlignment::Joint)
    }

    fn requires_joint_rows(&self) -> bool {
        true
    }

    fn exact_law(&self) -> bool {
        self.artifact.metadata().calibration == DistributionCalibration::Exact
    }

    fn answers_outcome_law(&self) -> bool {
        self.outcome_law
    }

    fn unsupported(&self, column: usize) -> bool {
        self.artifact
            .metadata()
            .supported
            .as_deref()
            .is_some_and(|mask| !mask[self.columns[column]])
    }

    fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    fn draw_count(&self) -> usize {
        self.artifact.n_draws()
    }

    fn fill_draws(
        &mut self,
        start: usize,
        rows: usize,
        out: &mut [f64],
    ) -> Result<(), DemandSourceError> {
        let [_, width] = self.artifact.shape();
        let draws = self.artifact.draws();
        let kept = self.columns.len();
        for row in 0..rows {
            let base = (start + row) * width;
            for (slot, column) in self.columns.iter().enumerate() {
                out[row * kept + slot] = draws[base + column];
            }
        }
        Ok(())
    }
}

/// The scratch a draw batch may take when the context sets no tighter limit.
const BATCH_SCRATCH_CEILING: u64 = 1 << 20;

/// The budget a decision draws from `ctx`: a batch of up to a mebibyte of scratch (a quarter
/// of the context's memory limit when it has one), and exactly the bytes the per-action
/// utilities and weights of `draws` draws need.
fn admitted_budget(
    contract: &DecisionContract,
    draws: usize,
    width: usize,
    ctx: &ExecutionContext,
) -> DemandBudget {
    let row = (width as u64).saturating_mul(8).max(8);
    let mut scratch = row.saturating_mul(draws as u64).clamp(row, BATCH_SCRATCH_CEILING.max(row));
    if let Some(limit) = ctx.memory_limit_bytes() {
        scratch = scratch.min(limit / 4).max(row);
    }
    let retained =
        (contract.actions.len() as u64 + 1).saturating_mul(draws as u64).saturating_mul(8);
    DemandBudget {
        scratch_bytes: scratch,
        retained_bytes: retained,
        retention: OutputRetention::SummaryOnly,
    }
}

/// Lower an executor refusal to the refusal the whole-source evaluators raise.
fn lower_eval(error: EvaluationError) -> DecisionEvalError {
    match error {
        EvaluationError::Contract(error) => DecisionEvalError::Contract(error),
        EvaluationError::Evaluation(error) => error,
        EvaluationError::MeansInsufficient { needed, .. }
        | EvaluationError::CarrierUnavailable { needed } => {
            DecisionEvalError::MeanSourceInsufficient { needed }
        }
        EvaluationError::IdentifiedSet => {
            DecisionEvalError::MeanSourceInsufficient { needed: Vec::new() }
        }
        EvaluationError::ScratchTooSmall { .. }
        | EvaluationError::RetentionExceeded { .. }
        | EvaluationError::Admission { .. } => {
            DecisionEvalError::Contract(DecisionContractError::InvalidParameter(
                "the decision evaluation does not fit the execution context's memory budget",
            ))
        }
        EvaluationError::Cancelled { .. } => DecisionEvalError::Contract(
            DecisionContractError::InvalidParameter("the decision evaluation was cancelled"),
        ),
        EvaluationError::RetentionUnsupported { .. } | EvaluationError::Source { .. } => {
            DecisionEvalError::Contract(DecisionContractError::InvalidParameter(
                "the decision source failed or returned an inconsistent batch",
            ))
        }
    }
}

/// Evaluate `contract` on a mean source through the demand executor, leasing nothing: a mean
/// answers the demand or the source is refused as insufficient, exactly as
/// [`evaluate_contract_on_means`] refuses it.
///
/// # Errors
///
/// The refusals of [`evaluate_contract_on_means`].
pub fn evaluate_mean_source_admitted(
    contract: &DecisionContract,
    source: &MeanSource,
    ctx: &ExecutionContext,
) -> Result<DecisionResult, DecisionEvalError> {
    let mut demand = MeanDemand::new(source);
    let budget = admitted_budget(contract, 0, source.coordinates.len(), ctx);
    execute_demand_admitted(contract, &mut demand, budget, ctx)
        .map(|executed| executed.decision)
        .map_err(lower_eval)
}

/// Evaluate `contract` on a stored joint law through the demand executor: the draws are read
/// in batches under the context's admission pool, with the artifact's own weights, support
/// mask and meaning.
///
/// # Errors
///
/// The refusals of [`crate::decision_eval::evaluate_contract`], and a refusal naming the
/// memory budget when the pool cannot hold the batch and the per-action utilities.
pub fn evaluate_artifact_admitted(
    contract: &DecisionContract,
    artifact: &DistributionArtifact,
    ctx: &ExecutionContext,
) -> Result<DecisionResult, DecisionEvalError> {
    if matches!(
        contract.criterion,
        DecisionCriterion::MinimaxOverIdentifiedSet | DecisionCriterion::MaximinOverStructures
    ) {
        return Err(DecisionEvalError::StructureInputsRequired(
            "needs identified-set or structure inputs, not one draw source",
        ));
    }
    let mut demand = ArtifactDemand::new(artifact);
    let budget = admitted_budget(contract, artifact.n_draws(), demand.coordinates.len(), ctx);
    execute_demand_admitted(contract, &mut demand, budget, ctx)
        .map(|executed| executed.decision)
        .map_err(lower_eval)
}

fn source_failure(error: &DemandSourceError) -> EvaluationError {
    EvaluationError::Source { message: error.message.clone() }
}

fn execute_on_means(
    contract: &DecisionContract,
    source: &mut dyn DemandSource,
    budget: DemandBudget,
) -> Result<ExecutedDemand, EvaluationError> {
    let means = source.means().map_err(|error| source_failure(&error))?;
    let coordinates = source.coordinates().to_vec();
    if means.len() != coordinates.len() {
        return Err(EvaluationError::Source {
            message: format!(
                "{} means for {} coordinates; a source supplies one mean per coordinate",
                means.len(),
                coordinates.len()
            ),
        });
    }
    let receipt = source.receipt();
    let mean_source = MeanSource {
        coordinates,
        means,
        estimate: source.mean_estimate(),
        provider_id: receipt.provider_id,
        snapshot_id: receipt.snapshot_id,
        causal_contract_id: receipt.causal_contract_id,
        rng_id: receipt.rng_id,
    };
    let evaluated = evaluate_on_means(contract, &mean_source)?;
    Ok(ExecutedDemand {
        decision: evaluated.decision,
        record: ExecutionRecord {
            carrier: RequestedCarrier::Means,
            batches: 1,
            items: evaluated.batching.items,
            batch_rows: 0,
            draws_drawn: 0,
            scratch_bytes: 0,
            retained_peak_bytes: 0,
            retained_bytes: 0,
            retention: budget.retention,
            predictions_requested: 0,
        },
        utilities: None,
    })
}

fn locate_column(
    coordinates: &[ScientificQuantity],
    action: &str,
    input: usize,
    wanted: &ScientificQuantity,
) -> Result<usize, EvaluationError> {
    coordinates
        .iter()
        .position(|candidate| candidate.require_same_coordinate(wanted).is_ok())
        .ok_or_else(|| {
            EvaluationError::Evaluation(DecisionEvalError::QuantityNotFound {
                action: action.to_owned(),
                input,
            })
        })
}

#[allow(
    clippy::too_many_lines,
    reason = "one demand in order: bind columns, size the batch, stream, tally, rank"
)]
fn execute_on_draws(
    contract: &DecisionContract,
    source: &mut dyn DemandSource,
    budget: DemandBudget,
    cancellation: Option<&CancellationToken>,
    needs_joint: bool,
) -> Result<ExecutedDemand, EvaluationError> {
    if (needs_joint || source.requires_joint_rows()) && !source.joint() {
        return Err(EvaluationError::Evaluation(DecisionEvalError::JointLawRequired {
            action: contract
                .actions
                .iter()
                .find(|action| action.utility.inputs_used().len() > 1)
                .map(|action| action.id.clone()),
            supplied_alignment: "independent_marginals",
        }));
    }
    let identity = contract.identity().map_err(EvaluationError::Contract)?;
    let coordinates = source.coordinates().to_vec();
    let width = coordinates.len();
    let mut columns: Vec<Vec<usize>> = Vec::with_capacity(contract.actions.len());
    for action in &contract.actions {
        let mut bound = Vec::with_capacity(action.inputs.len());
        for (input, wanted) in action.inputs.iter().enumerate() {
            let column = locate_column(&coordinates, &action.id, input, wanted)?;
            if source.unsupported(column) {
                return Err(EvaluationError::Evaluation(
                    DecisionEvalError::UnsupportedCoordinate { action: action.id.clone(), input },
                ));
            }
            bound.push(column);
            if wanted.functional_id == "outcome" && !source.answers_outcome_law() {
                return Err(EvaluationError::Evaluation(DecisionEvalError::MeaningMismatch {
                    action: action.id.clone(),
                    input,
                }));
            }
        }
        columns.push(bound);
    }
    let total = source.draw_count();
    if total == 0 || width == 0 {
        return Err(EvaluationError::Source {
            message: "the source holds no draws of any coordinate".into(),
        });
    }
    let row_bytes = (width as u64).checked_mul(8).ok_or(EvaluationError::ScratchTooSmall {
        needed: u64::MAX,
        budget: budget.scratch_bytes,
    })?;
    let batch_rows =
        usize::try_from(budget.scratch_bytes / row_bytes).unwrap_or(usize::MAX).min(total);
    if batch_rows == 0 {
        return Err(EvaluationError::ScratchTooSmall {
            needed: row_bytes,
            budget: budget.scratch_bytes,
        });
    }
    let actions = contract.actions.len() as u64;
    let retained_peak = actions
        .checked_add(1)
        .and_then(|columns| columns.checked_mul(total as u64))
        .and_then(|cells| cells.checked_mul(8))
        .filter(|needed| *needed <= budget.retained_bytes)
        .ok_or_else(|| EvaluationError::RetentionExceeded {
            needed: (actions + 1).saturating_mul(total as u64).saturating_mul(8),
            budget: budget.retained_bytes,
        })?;
    let mut scratch = vec![0.0_f64; batch_rows * width];
    let mut utilities: Vec<Vec<f64>> =
        (0..contract.actions.len()).map(|_| Vec::with_capacity(total)).collect();
    let uniform = 1.0 / total as f64;
    let weights: Vec<f64> = match source.weights() {
        Some(weights) if weights.len() == total => weights.to_vec(),
        Some(weights) => {
            return Err(EvaluationError::Source {
                message: format!("{} weights for {total} draws", weights.len()),
            });
        }
        None => vec![uniform; total],
    };
    // held[constraint][action] is the weight of the draws within the bound; violated is any
    // draw of positive weight beyond it.
    let mut held = vec![vec![0.0_f64; contract.actions.len()]; contract.constraints.len()];
    let mut violated = vec![vec![false; contract.actions.len()]; contract.constraints.len()];
    let mut inputs: Vec<f64> = Vec::new();
    let mut batches = 0_usize;
    let mut start = 0_usize;
    while start < total {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(EvaluationError::Cancelled { batches });
        }
        let rows = batch_rows.min(total - start);
        let window = &mut scratch[..rows * width];
        source.fill_draws(start, rows, window).map_err(|error| source_failure(&error))?;
        batches += 1;
        for (offset, row) in window.chunks_exact(width).enumerate() {
            let weight = weights[start + offset];
            for (index, action) in contract.actions.iter().enumerate() {
                inputs.clear();
                inputs.extend(columns[index].iter().map(|column| row[*column]));
                let value = action.utility.evaluate(&inputs).map_err(EvaluationError::Contract)?;
                if !value.is_finite() {
                    return Err(EvaluationError::Evaluation(DecisionEvalError::NonFiniteUtility {
                        action: action.id.clone(),
                    }));
                }
                utilities[index].push(value);
                for (slot, constraint) in contract.constraints.iter().enumerate() {
                    if !constraint.applies_to.is_empty()
                        && !constraint.applies_to.contains(&action.id)
                    {
                        continue;
                    }
                    let measured =
                        constraint.expr.evaluate(&inputs).map_err(EvaluationError::Contract)?;
                    if !measured.is_finite() {
                        return Err(EvaluationError::Evaluation(
                            DecisionEvalError::NonFiniteUtility { action: action.id.clone() },
                        ));
                    }
                    if measured <= constraint.bound {
                        held[slot][index] += weight;
                    } else if weight > 0.0 {
                        violated[slot][index] = true;
                    }
                }
            }
        }
        start += rows;
    }
    let mut exclusions: Vec<Vec<ConstraintExclusion>> = vec![Vec::new(); contract.actions.len()];
    for (slot, constraint) in contract.constraints.iter().enumerate() {
        for (index, action) in contract.actions.iter().enumerate() {
            if !constraint.applies_to.is_empty() && !constraint.applies_to.contains(&action.id) {
                continue;
            }
            let probability = held[slot][index];
            if constraint_fails(constraint.min_probability, probability, violated[slot][index]) {
                exclusions[index].push(ConstraintExclusion {
                    constraint_id: constraint.id.clone(),
                    probability,
                    required: constraint.min_probability,
                });
            }
        }
    }
    let table = Table { utilities, weights };
    let decision =
        rank_table(contract, identity, &table, exclusions, source.exact_law(), source.receipt())
            .map_err(lift_eval)?;
    let keep = budget.retention == OutputRetention::FullReusable;
    Ok(ExecutedDemand {
        decision,
        record: ExecutionRecord {
            carrier: RequestedCarrier::Draws,
            batches,
            items: distinct_inputs(contract),
            batch_rows,
            draws_drawn: total,
            scratch_bytes: (scratch.len() as u64) * 8,
            retained_peak_bytes: retained_peak,
            retained_bytes: if keep { retained_peak } else { 0 },
            retention: budget.retention,
            predictions_requested: 0,
        },
        utilities: keep.then_some(table.utilities),
    })
}

fn means_insufficient(needed: Vec<SourceRepresentation>) -> EvaluationError {
    let predictions = needed.iter().any(|representation| {
        matches!(representation, SourceRepresentation::Cdf | SourceRepresentation::QuantileFunction)
    });
    let draws = needed.iter().any(|representation| {
        matches!(
            representation,
            SourceRepresentation::MarginalDraws | SourceRepresentation::JointDraws
        )
    });
    let (missing, missing_label) = match (predictions, draws) {
        (true, true) => (PREDICTIONS_OR_DRAWS, "predictions or draws"),
        (true, false) => (ONLY_PREDICTIONS, "predictions"),
        (false, true) => (ONLY_DRAWS, "draws"),
        (false, false) => {
            return EvaluationError::MeansInsufficient {
                missing: &[],
                needed,
                missing_label: "a sufficient representation other than a mean",
            };
        }
    };
    EvaluationError::MeansInsufficient { missing, needed, missing_label }
}

fn lift_eval(error: DecisionEvalError) -> EvaluationError {
    match error {
        DecisionEvalError::Contract(error) => EvaluationError::Contract(error),
        DecisionEvalError::MeanSourceInsufficient { needed } if needed.is_empty() => {
            EvaluationError::IdentifiedSet
        }
        DecisionEvalError::MeanSourceInsufficient { needed } => means_insufficient(needed),
        other => EvaluationError::Evaluation(other),
    }
}

fn distinct_inputs(contract: &DecisionContract) -> usize {
    let mut coordinates = Vec::new();
    for action in &contract.actions {
        coordinates.extend(action.inputs.iter().cloned());
    }
    antecedent_contract::quantity::deduped_coordinate_positions(&coordinates).len()
}

#[cfg(test)]
mod tests {
    use antecedent_contract::{QuantityRole, ScientificQuantity};

    use super::*;
    use crate::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, HardConstraint, StructuralPolicy,
        UtilityExpr,
    };
    use crate::decision_eval::Verdict;

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    /// `E[Y | do(a)] = 1 + 2a` on `a = 0, 1, 2`.
    fn source() -> MeanSource {
        MeanSource {
            coordinates: vec![quantity("do(a=0)"), quantity("do(a=1)"), quantity("do(a=2)")],
            means: vec![1.0, 3.0, 5.0],
            provider_id: "lab".into(),
            snapshot_id: "snap-9".into(),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
            estimate: MeanEstimate::Frequentist,
        }
    }

    fn affine() -> UtilityExpr {
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            UtilityExpr::Const(1.0),
        )
    }

    fn contract_with(utility: UtilityExpr) -> DecisionContract {
        DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "wait".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=0)")],
                    utility: affine(),
                },
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=2)")],
                    utility,
                },
            ],
            utility_units: "utility".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        }
    }

    fn contract() -> DecisionContract {
        contract_with(affine())
    }

    fn assert_missing(error: &EvaluationError, label: &str, carriers: &[MissingCarrier]) {
        let EvaluationError::MeansInsufficient { missing, missing_label, needed } = error else {
            panic!("expected MeansInsufficient, got {error:?}");
        };
        assert_eq!(*missing_label, label);
        assert_eq!(*missing, carriers);
        assert!(!needed.iter().any(|representation| *representation == SourceRepresentation::Mean));
        let text = error.to_string();
        assert!(text.contains(label), "{text}");
    }

    #[test]
    fn affine_utility_from_means_requests_neither_predictions_nor_draws() {
        let record = evaluate_on_means(&contract(), &source()).unwrap();
        // 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9.
        assert!((record.decision.actions[0].expected_utility - 1.0).abs() < 1e-12);
        assert!((record.decision.actions[1].expected_utility - 9.0).abs() < 1e-12);
        assert_eq!(record.decision.verdict, Verdict::UniquelyOptimal("treat".into()));
        assert_eq!(record.predictions_requested, 0);
        assert_eq!(record.draws_requested, 0);
        assert_eq!(record.batching.carrier, RequestedCarrier::Means);
        assert_eq!(record.decision.n_draws, 0);
        assert!(record.decision.evpi.is_none());
        assert!(record.decision.actions.iter().all(|outcome| {
            outcome.standard_error.is_none()
                && outcome.expected_regret.is_none()
                && outcome.max_regret.is_none()
        }));
        assert!(record.decision.assumptions.iter().any(|note| note.contains("means only")));

        let mut loss = contract();
        loss.criterion = DecisionCriterion::PosteriorExpectedLoss;
        let loss = evaluate_on_means(&loss, &source()).unwrap();
        assert_eq!(loss.decision.verdict, Verdict::UniquelyOptimal("wait".into()));
        assert_eq!(loss.predictions_requested, 0);
        assert_eq!(loss.draws_requested, 0);
    }

    #[test]
    fn nonlinear_utility_given_only_means_names_the_missing_draws() {
        let squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
        let error = evaluate_on_means(&contract_with(squared), &source()).unwrap_err();
        assert_missing(&error, "draws", ONLY_DRAWS);
        let text = error.to_string();
        assert!(!text.contains("predictions"), "{text}");
        let EvaluationError::MeansInsufficient { needed, .. } = &error else {
            panic!("expected MeansInsufficient, got {error:?}");
        };
        assert!(needed.contains(&SourceRepresentation::JointDraws));
    }

    #[test]
    fn a_risk_demand_given_only_means_names_the_missing_predictions_or_draws() {
        let mut quantile = contract_with(UtilityExpr::Input(0));
        quantile.actions[0].utility = UtilityExpr::Input(0);
        quantile.criterion = DecisionCriterion::Quantile { p: 0.5 };
        let error = evaluate_on_means(&quantile, &source()).unwrap_err();
        assert_missing(&error, "predictions or draws", PREDICTIONS_OR_DRAWS);

        let mut constrained = contract();
        constrained.constraints.push(HardConstraint {
            id: "cap".into(),
            expr: UtilityExpr::Input(0),
            bound: 10.0,
            min_probability: 0.9,
            units: "mmHg".into(),
            applies_to: vec![],
        });
        let error = evaluate_on_means(&constrained, &source()).unwrap_err();
        assert_missing(&error, "predictions or draws", PREDICTIONS_OR_DRAWS);
        let text = error.to_string();
        assert!(text.contains("predictions"), "{text}");
        assert!(text.contains("draws"), "{text}");

        let mut regret = contract();
        regret.criterion = DecisionCriterion::ExpectedRegret;
        let error = evaluate_on_means(&regret, &source()).unwrap_err();
        assert_missing(&error, "draws", ONLY_DRAWS);
    }

    #[test]
    fn batching_and_retention_record_what_was_requested() {
        let record = evaluate_on_means(&contract(), &source()).unwrap();
        assert_eq!(record.batching, BatchRecord::means(2));
        assert_eq!(record.retention, RetentionRecord::means_summary());
        assert_eq!(record.retention.mode, OutputRetention::SummaryOnly);
        assert_eq!(record.retention.fits, 0);
        assert_eq!(record.predictions_requested, 0);
        assert_eq!(record.draws_requested, 0);

        let mut tied = contract();
        tied.actions[1].inputs = tied.actions[0].inputs.clone();
        let tied = evaluate_on_means(&tied, &source()).unwrap();
        assert_eq!(tied.batching.items, 1);
        assert_eq!(tied.batching.batches, 1);
        assert_eq!(tied.retention.fits, 0);
    }

    #[test]
    fn an_identified_set_is_not_formed_from_means() {
        for criterion in
            [DecisionCriterion::MinimaxOverIdentifiedSet, DecisionCriterion::MaximinOverStructures]
        {
            let mut robust = contract();
            robust.criterion = criterion;
            let error = evaluate_on_means(&robust, &source()).unwrap_err();
            assert_eq!(error, EvaluationError::IdentifiedSet);
            let text = error.to_string();
            assert!(!text.contains("predictions"), "{text}");
            assert!(!text.contains("draws"), "{text}");
            assert!(!text.contains("posterior"), "{text}");
        }
    }

    #[test]
    fn a_missing_mean_coordinate_is_not_a_request_for_draws() {
        let mut missing = source();
        missing.coordinates.pop();
        missing.means.pop();
        let error = evaluate_on_means(&contract(), &missing).unwrap_err();
        assert!(matches!(
            error,
            EvaluationError::Evaluation(DecisionEvalError::QuantityNotFound { .. })
        ));
    }

    /// A source that counts every carrier asked of it. Draw `i` of `do(a=0)` is
    /// `1 + 0.5 sin i` and of `do(a=2)` is `5 + 0.8 cos 0.7 i`.
    struct Grid {
        coordinates: Vec<ScientificQuantity>,
        total: usize,
        supplies_draws: bool,
        joint: bool,
        mean_calls: usize,
        draw_calls: usize,
        largest_batch: usize,
    }

    impl Grid {
        fn new(total: usize) -> Self {
            Self {
                coordinates: vec![quantity("do(a=0)"), quantity("do(a=2)")],
                total,
                supplies_draws: true,
                joint: true,
                mean_calls: 0,
                draw_calls: 0,
                largest_batch: 0,
            }
        }

        fn x0(i: usize) -> f64 {
            1.0 + 0.5 * (i as f64).sin()
        }

        fn x2(i: usize) -> f64 {
            5.0 + 0.8 * (0.7 * i as f64).cos()
        }
    }

    impl DemandSource for Grid {
        fn coordinates(&self) -> &[ScientificQuantity] {
            &self.coordinates
        }

        fn receipt(&self) -> SourceReceipt {
            SourceReceipt {
                provider_id: "lab".into(),
                snapshot_id: "snap-9".into(),
                rng_id: "none:grid".into(),
                causal_contract_id: "checked-contract".into(),
            }
        }

        fn mean_estimate(&self) -> MeanEstimate {
            MeanEstimate::Frequentist
        }

        fn means(&mut self) -> Result<Vec<f64>, DemandSourceError> {
            self.mean_calls += 1;
            let n = self.total;
            Ok(vec![
                (0..n).map(Self::x0).sum::<f64>() / n as f64,
                (0..n).map(Self::x2).sum::<f64>() / n as f64,
            ])
        }

        fn supplies_draws(&self) -> bool {
            self.supplies_draws
        }

        fn joint(&self) -> bool {
            self.joint
        }

        fn draw_count(&self) -> usize {
            self.total
        }

        fn fill_draws(
            &mut self,
            start: usize,
            rows: usize,
            out: &mut [f64],
        ) -> Result<(), DemandSourceError> {
            self.draw_calls += 1;
            self.largest_batch = self.largest_batch.max(rows);
            assert_eq!(out.len(), rows * 2);
            for row in 0..rows {
                out[row * 2] = Self::x0(start + row);
                out[row * 2 + 1] = Self::x2(start + row);
            }
            Ok(())
        }
    }

    fn roomy(retention: OutputRetention) -> DemandBudget {
        DemandBudget { scratch_bytes: 1 << 20, retained_bytes: 1 << 20, retention }
    }

    fn squared_contract() -> DecisionContract {
        contract_with(UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)))
    }

    #[test]
    fn an_affine_demand_asks_the_source_for_means_once_and_never_for_draws() {
        let mut grid = Grid::new(100);
        // A zero budget cannot hold a draw row; the mean path never needs one.
        let budget = DemandBudget {
            scratch_bytes: 0,
            retained_bytes: 0,
            retention: OutputRetention::SummaryOnly,
        };
        let run = execute_demand(&contract(), &mut grid, budget, None).unwrap();
        assert_eq!(grid.mean_calls, 1);
        assert_eq!(grid.draw_calls, 0);
        assert_eq!(run.record.carrier, RequestedCarrier::Means);
        assert_eq!(run.record.batches, 1);
        assert_eq!(run.record.draws_drawn, 0);
        assert_eq!(run.record.scratch_bytes, 0);
        assert_eq!(run.record.retained_peak_bytes, 0);
        assert_eq!(run.record.predictions_requested, 0);
        assert!(run.utilities.is_none());
        assert_eq!(run.decision.n_draws, 0);
        let means = Grid::new(100).means().unwrap();
        assert!((run.decision.actions[0].expected_utility - (2.0 * means[0] - 1.0)).abs() < 1e-12);
        assert!((run.decision.actions[1].expected_utility - (2.0 * means[1] - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn a_nonlinear_demand_streams_draws_in_batches_the_scratch_budget_holds() {
        let total = 100;
        let direct_wait = (0..total).map(|i| 2.0 * Grid::x0(i) - 1.0).sum::<f64>() / total as f64;
        let direct_treat = (0..total).map(|i| Grid::x2(i).powi(2)).sum::<f64>() / total as f64;

        let mut grid = Grid::new(total);
        let budget = DemandBudget {
            scratch_bytes: 7 * 2 * 8,
            retained_bytes: 1 << 20,
            retention: OutputRetention::SummaryOnly,
        };
        let run = execute_demand(&squared_contract(), &mut grid, budget, None).unwrap();
        assert_eq!(grid.mean_calls, 0, "draws, not means, answer a nonlinear utility");
        assert_eq!(run.record.carrier, RequestedCarrier::Draws);
        assert_eq!(run.record.batch_rows, 7);
        assert_eq!(run.record.batches, total.div_ceil(7));
        assert_eq!(grid.draw_calls, total.div_ceil(7));
        assert_eq!(grid.largest_batch, 7);
        assert_eq!(run.record.draws_drawn, total);
        assert_eq!(run.record.scratch_bytes, 7 * 2 * 8);
        assert_eq!(run.decision.n_draws, total);
        assert!((run.decision.actions[0].expected_utility - direct_wait).abs() < 1e-12);
        assert!((run.decision.actions[1].expected_utility - direct_treat).abs() < 1e-12);
        assert_eq!(run.decision.verdict, Verdict::UniquelyOptimal("treat".into()));

        // The batch size does not move the answer.
        let mut whole = Grid::new(total);
        let all = execute_demand(
            &squared_contract(),
            &mut whole,
            roomy(OutputRetention::SummaryOnly),
            None,
        )
        .unwrap();
        assert_eq!(all.record.batches, 1);
        let mut single = Grid::new(total);
        let one_row = DemandBudget { scratch_bytes: 16, ..roomy(OutputRetention::SummaryOnly) };
        let rows = execute_demand(&squared_contract(), &mut single, one_row, None).unwrap();
        assert_eq!(rows.record.batches, total);
        assert_eq!(all.decision, run.decision);
        assert_eq!(rows.decision, run.decision);
    }

    #[test]
    fn retention_is_budgeted_before_the_utilities_are_allocated() {
        let total = 100;
        let needed = 3 * total as u64 * 8;
        let mut grid = Grid::new(total);
        let short =
            DemandBudget { retained_bytes: needed - 1, ..roomy(OutputRetention::FullReusable) };
        let error = execute_demand(&squared_contract(), &mut grid, short, None).unwrap_err();
        assert_eq!(error, EvaluationError::RetentionExceeded { needed, budget: needed - 1 });
        assert_eq!(grid.draw_calls, 0, "nothing is drawn for a budget that cannot hold the result");

        let tiny = DemandBudget { scratch_bytes: 8, ..roomy(OutputRetention::FullReusable) };
        let error = execute_demand(&squared_contract(), &mut grid, tiny, None).unwrap_err();
        assert_eq!(error, EvaluationError::ScratchTooSmall { needed: 16, budget: 8 });

        let kept = execute_demand(
            &squared_contract(),
            &mut grid,
            DemandBudget { retained_bytes: needed, ..roomy(OutputRetention::FullReusable) },
            None,
        )
        .unwrap();
        let utilities = kept.utilities.expect("full reusable state is returned");
        assert_eq!(utilities.len(), 2);
        assert!(utilities.iter().all(|column| column.len() == total));
        assert_eq!(kept.record.retained_bytes, needed);
        assert!((utilities[1][3] - Grid::x2(3).powi(2)).abs() < 1e-12);

        let summary = execute_demand(
            &squared_contract(),
            &mut grid,
            roomy(OutputRetention::SummaryOnly),
            None,
        )
        .unwrap();
        assert!(summary.utilities.is_none());
        assert_eq!(summary.record.retained_bytes, 0);
        assert_eq!(summary.record.retained_peak_bytes, needed);

        let disk = execute_demand(
            &squared_contract(),
            &mut grid,
            roomy(OutputRetention::DiskBacked),
            None,
        );
        assert_eq!(
            disk.unwrap_err(),
            EvaluationError::RetentionUnsupported { mode: OutputRetention::DiskBacked }
        );
    }

    #[test]
    fn a_hard_constraint_is_tallied_across_batches() {
        let total = 100;
        let mut constrained = contract();
        constrained.constraints.push(HardConstraint {
            id: "cap".into(),
            expr: UtilityExpr::Input(0),
            bound: 5.5,
            min_probability: 0.9,
            units: "mmHg".into(),
            applies_to: vec![],
        });
        let within = (0..total).filter(|i| Grid::x2(*i) <= 5.5).count() as f64 / total as f64;
        assert!(within < 0.9, "the fixture must break the constraint for the treat action");
        let mut grid = Grid::new(total);
        let budget = DemandBudget { scratch_bytes: 13 * 16, ..roomy(OutputRetention::SummaryOnly) };
        let run = execute_demand(&constrained, &mut grid, budget, None).unwrap();
        assert_eq!(grid.mean_calls, 0, "a probability constraint cannot be answered by a mean");
        assert!(run.record.batches > 1);
        assert!(run.decision.actions[0].admissible);
        let treat = &run.decision.actions[1];
        assert!(!treat.admissible);
        assert!((treat.exclusions[0].probability - within).abs() < 1e-12);
        assert_eq!(run.decision.verdict, Verdict::UniquelyOptimal("wait".into()));
    }

    #[test]
    fn cancellation_between_batches_stops_drawing() {
        let mut grid = Grid::new(100);
        let token = CancellationToken::cancel_after_checks(2);
        let budget = DemandBudget { scratch_bytes: 7 * 16, ..roomy(OutputRetention::SummaryOnly) };
        let error =
            execute_demand(&squared_contract(), &mut grid, budget, Some(&token)).unwrap_err();
        let EvaluationError::Cancelled { batches } = error else {
            panic!("expected Cancelled, got {error:?}");
        };
        assert!(batches < 15);
        assert_eq!(grid.draw_calls, batches);
    }

    #[test]
    fn a_nonlinear_demand_against_a_means_only_source_requests_nothing() {
        let mut grid = Grid::new(100);
        grid.supplies_draws = false;
        let error = execute_demand(
            &squared_contract(),
            &mut grid,
            roomy(OutputRetention::SummaryOnly),
            None,
        )
        .unwrap_err();
        assert_missing(&error, "draws", ONLY_DRAWS);
        assert_eq!(grid.mean_calls, 0);
        assert_eq!(grid.draw_calls, 0);
    }

    #[test]
    fn a_regret_demand_needs_paired_draws() {
        let mut regret = contract();
        regret.criterion = DecisionCriterion::ExpectedRegret;
        let mut grid = Grid::new(50);
        grid.joint = false;
        let error = execute_demand(&regret, &mut grid, roomy(OutputRetention::SummaryOnly), None)
            .unwrap_err();
        assert!(matches!(
            error,
            EvaluationError::Evaluation(DecisionEvalError::JointLawRequired { .. })
        ));
        assert_eq!(grid.draw_calls, 0);

        grid.joint = true;
        let run =
            execute_demand(&regret, &mut grid, roomy(OutputRetention::SummaryOnly), None).unwrap();
        assert!(run.decision.evpi.is_some());
        assert!(run.decision.actions.iter().all(|outcome| outcome.expected_regret.is_some()));
    }

    #[test]
    fn the_draw_path_leases_its_bytes_from_the_shared_pool_and_a_mean_leases_none() {
        use antecedent_core::execution::AdmissionPool;
        let total = 100;
        let budget = DemandBudget {
            scratch_bytes: 7 * 16,
            retained_bytes: 3 * total as u64 * 8,
            retention: OutputRetention::SummaryOnly,
        };
        let wanted = budget.scratch_bytes + budget.retained_bytes;

        let small = AdmissionPool::fixed(4, wanted - 1);
        let ctx = ExecutionContext::for_tests(1).with_admission(small.clone());
        let mut grid = Grid::new(total);
        let refused = execute_demand_admitted(&squared_contract(), &mut grid, budget, &ctx);
        assert!(matches!(refused, Err(EvaluationError::Admission { .. })));
        assert_eq!(grid.draw_calls, 0);

        let pool = AdmissionPool::fixed(4, wanted);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let run = execute_demand_admitted(&squared_contract(), &mut grid, budget, &ctx).unwrap();
        assert_eq!(run.record.carrier, RequestedCarrier::Draws);
        assert_eq!(pool.available().bytes(), wanted, "the lease is released on return");

        let mut mean_grid = Grid::new(total);
        let none = AdmissionPool::fixed(4, 0);
        let ctx = ExecutionContext::for_tests(1).with_admission(none);
        let run = execute_demand_admitted(&contract(), &mut mean_grid, budget, &ctx).unwrap();
        assert_eq!(run.record.carrier, RequestedCarrier::Means);
        assert_eq!(mean_grid.draw_calls, 0);
    }
}
