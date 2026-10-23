//! Estimate clicks and sealed-operation dispatch for a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CheckedAipwOperation, CheckedFrontDoorOperation, CheckedGlmAdjustmentOperation,
    CheckedIvOperation, CheckedLinearOperation, CheckedRdOperation, PreparedExecution,
    PreparedStudy, overlay_prepared_score_functional, query_reads_score_table,
};
use crate::analysis::CheckedConditionalOperation;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::CheckedCellAipwResponseOperation;
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use antecedent_estimate::ScoreTable;
use std::sync::Arc;

/// What a tabular estimate click learned besides its result, and how it
/// should bind the result.
///
/// A plain estimate stamps the executed contract on its result as it always
/// has. A refresh asks for the contract to be deferred, because the contract
/// names the handle state the result belongs to and the refresh only knows
/// that state once the click has succeeded; it then installs what the click
/// derived from the new rows (the rebound operation, any frozen scores)
/// instead of deriving it again, and binds the contract once.
///
/// The notes travel beside the result rather than wrapping it: the sealed
/// operations have many return sites and each result-sized slot is 11 KiB in
/// a debug frame that already runs close to the default test thread's stack.
#[derive(Default)]
pub(crate) struct ClickNotes {
    /// Leave `executed_contract` unset; the caller binds it.
    defer_contract: bool,
    /// The operation rebound to the click's rows, for the routes whose
    /// receipts bind rows and are replaced by a refresh.
    pub(super) rebound: Option<Rebound>,
    /// The frozen cross-fitted scores of the click's study when the click
    /// built them. The scores are a function of the study snapshot and the
    /// context seed alone, so a refresh over the same snapshot and context may
    /// keep them.
    pub(super) scores: ClickScores,
}

/// Whether a click built its study's frozen scores, and what it built.
#[derive(Debug, Default)]
pub(super) enum ClickScores {
    /// The click did not build them.
    #[default]
    NotBuilt,
    /// The click built them; `None` when the plan keeps no table.
    Built(Option<ScoreTable>),
}

impl ClickNotes {
    /// Notes of a click whose contract the caller binds.
    pub(super) fn deferred() -> Self {
        Self { defer_contract: true, ..Self::default() }
    }

    /// Whether the click leaves `executed_contract` to the caller.
    pub(super) fn defers_contract(&self) -> bool {
        self.defer_contract
    }

    /// Record the rebound operation the click executed with.
    pub(super) fn record(&mut self, rebound: Rebound) {
        self.rebound = Some(rebound);
    }
}

/// The sealed operation a click rebound to its rows.
///
/// Only the routes whose receipts bind complete-case rows appear here: a
/// refresh must replace exactly these, and no other retained operation, so
/// the procedure chosen at preparation cannot change under it. Operations are
/// boxed so a click frame holds one small value however many routes it names.
pub(crate) enum Rebound {
    Linear(Box<CheckedLinearOperation>),
    GlmAdjustment(Box<CheckedGlmAdjustmentOperation>),
    Rd(Box<CheckedRdOperation>),
    Aipw(Box<CheckedAipwOperation>),
    FrontDoor(Box<CheckedFrontDoorOperation>),
    Iv(Box<CheckedIvOperation>),
    Conditional(Box<CheckedConditionalOperation>),
    CellAipwResponse(Box<CheckedCellAipwResponseOperation>),
    /// The receipts [`PreparedExecution::rebind_for_dispatch`] rebinds for the
    /// shared tabular dispatcher.
    Dispatch(Box<PreparedExecution>),
}

impl Rebound {
    #[inline(never)]
    pub(super) fn into_execution(self) -> PreparedExecution {
        match self {
            Self::Linear(operation) => PreparedExecution::CheckedLinear(*operation),
            Self::GlmAdjustment(operation) => PreparedExecution::CheckedGlmAdjustment(*operation),
            Self::Rd(operation) => PreparedExecution::CheckedRd(*operation),
            Self::Aipw(operation) => PreparedExecution::CheckedAipw(*operation),
            Self::FrontDoor(operation) => PreparedExecution::FrontDoorLinear(*operation),
            Self::Iv(operation) => PreparedExecution::Iv(*operation),
            Self::Conditional(operation) => PreparedExecution::CheckedConditional(*operation),
            Self::CellAipwResponse(operation) => PreparedExecution::CellAipwResponse(*operation),
            Self::Dispatch(execution) => *execution,
        }
    }
}

impl PreparedStudy {
    /// Re-estimate on `data` without recompiling the physical plan.
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let shared = self
            .analysis
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(data).map(Arc::new))
            .transpose()?;
        self.estimate_with_shared(data, shared, ctx)
    }

    /// Execute the retained binding with native observations on success or failure.
    pub fn estimate_retained_observed(
        &self,
        ctx: &ExecutionContext,
    ) -> antecedent_core::execution_attempt::ExecutionAttempt<StudyResult, CausalError> {
        antecedent_core::execution_attempt::observe_with_context(ctx, || {
            self.estimate_retained(ctx)
        })
    }

    pub(crate) fn estimate_with_shared(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A stats-layer refusal of a regression, GLM, IV or matching design fit learns here
        // which plan it was for and, for a rank refusal, which columns are dependent on the
        // table the fit read.
        let mut observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            self.estimate_with_shared_unnamed(data, shared, ctx, &mut ClickNotes::default())
                .map_err(|error| self.name_stats_refusal(data, error, ctx))
        });
        if let Ok(result) = &mut observed.result {
            result.performance.native_execution.clone_from(&observed.facts);
        }
        observed.result
    }

    /// Name a stats-layer refusal for this plan and the table the fit read.
    pub(super) fn name_stats_refusal(
        &self,
        data: &TabularData,
        error: CausalError,
        ctx: &ExecutionContext,
    ) -> CausalError {
        crate::analysis::preflight::name_stats_refusal(&self.analysis, Some(data), error, ctx)
    }

    /// The click on `data`, recording in `notes` what it derived. A deferring
    /// `notes` leaves the result's executed contract for the caller to bind.
    ///
    /// Every tabular click, sealed or dispatched, passes here: the plan's lowered
    /// program is required, the click runs inside its dispatch scope, and the
    /// result carries its identity.
    pub(super) fn estimate_with_shared_unnamed(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<StudyResult, CausalError> {
        let lowered = self.plan.require_lowered(None)?.identity();
        let scoped = ctx.enter_lowered_dispatch(lowered);
        let mut result = if crate::analysis::missing_imputation::imputes(self.study()) {
            self.estimate_imputed(data, shared, &scoped, notes)?
        } else {
            self.estimate_click(data, shared, &scoped, notes)?
        };
        result.lowered_identity = Some(lowered);
        crate::analysis::target_validation::stamp_route_refutations(&mut result);
        Ok(result)
    }

    /// The click under a declared multiple imputation: the retained operation runs on every
    /// completed table and the fits are pooled. The contract is bound once, to the submitted
    /// (incomplete) rows and the declared imputation, unless `notes` defers it. A refresh
    /// keeps the retained operation: no per-table rebinding is recorded.
    #[inline(never)]
    fn estimate_imputed(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
        notes: &ClickNotes,
    ) -> Result<StudyResult, CausalError> {
        let mut result = crate::analysis::missing_imputation::impute_fit_pool(
            self.study(),
            data,
            ctx,
            |table, ctx| {
                let mut inner = ClickNotes::deferred();
                self.estimate_click(table, shared.clone(), ctx, &mut inner)
            },
        )?;
        if !notes.defers_contract() {
            result.executed_contract = Some(self.executed_contract(
                &DataInput::Tabular(data.clone()),
                self.analysis.refute,
                None,
            )?);
        }
        Ok(result)
    }

    /// The click itself; see [`Self::estimate_with_shared_unnamed`].
    fn estimate_click(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        // The retained operation executes itself; each implementation runs in
        // its own frame, so this one holds a single result slot.
        if let Some(result) = self.execution.operation().estimate_tabular(self, data, ctx, notes)? {
            return Ok(result);
        }
        let mut click_analysis = self.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.interference =
            click_analysis.interference.as_ref().map(|spec| spec.bound_to(data)).transpose()?;
        click_analysis.shared_batch_design = shared;
        let rebound = self.rebound_dispatch_execution(data)?;
        let execution = rebound.as_deref().unwrap_or(&self.execution);
        let mut result = click_analysis.execute_tabular(data, &self.plan, execution, ctx)?;
        // `execute_tabular` bypasses `Study::execute_on`, which is where fresh runs
        // record which refutation reports are caller-attested. Without the names,
        // the claim would drop custom-validator evidence from `attested` and from
        // the claim identity.
        result.custom_validator_names = click_analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect();
        // Only a non-mean functional is read from the frozen scores; a mean click keeps
        // the estimator's own value, so refitting the cross-fit table would be discarded.
        let click_scores = if query_reads_score_table(&self.analysis.query) {
            ClickScores::Built(click_analysis.prepare_score_table(ctx)?)
        } else {
            ClickScores::NotBuilt
        };
        overlay_prepared_score_functional(
            &self.analysis.query,
            match &click_scores {
                ClickScores::Built(table) => table.as_ref(),
                ClickScores::NotBuilt => None,
            },
            &mut result,
        )?;
        notes.rebound = rebound.map(Rebound::Dispatch);
        notes.scores = click_scores;
        self.stamp_deferrable(&click_analysis.data, result, notes.defers_contract())
    }

    /// Rebind the row-bound dispatch receipts to `data`. Boxed and built out
    /// of line so a click frame holds a pointer rather than a second
    /// execution value.
    #[inline(never)]
    fn rebound_dispatch_execution(
        &self,
        data: &TabularData,
    ) -> Result<Option<Box<PreparedExecution>>, CausalError> {
        Ok(self.execution.rebind_for_dispatch(data)?.map(Box::new))
    }
}
