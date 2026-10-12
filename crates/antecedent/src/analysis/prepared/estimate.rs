//! Estimate clicks and sealed-operation dispatch for a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    PreparedExecution, PreparedStudy, overlay_prepared_score_functional, query_reads_score_table,
};
use crate::analysis::builder::DataInput;
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use std::sync::Arc;

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

    pub(crate) fn estimate_with_shared(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A stats-layer refusal of a regression, GLM, IV or matching design fit learns here
        // which plan it was for and, for a rank refusal, which columns are dependent on the
        // table the fit read.
        self.estimate_with_shared_unnamed(data, shared, ctx).map_err(|error| {
            crate::analysis::preflight::name_stats_refusal(&self.analysis, Some(data), error, ctx)
        })
    }

    fn estimate_with_shared_unnamed(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        // The retained operation executes itself; each implementation runs in
        // its own frame, so this one holds a single result slot.
        if let Some(result) = self.execution.operation().estimate_tabular(self, data, ctx)? {
            return Ok(result);
        }
        let mut click_analysis = self.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.interference =
            click_analysis.interference.as_ref().map(|spec| spec.bound_to(data)).transpose()?;
        click_analysis.shared_batch_design = shared;
        let rebound = self.execution.rebind_for_dispatch(data)?;
        let execution = rebound.as_ref().unwrap_or(&self.execution);
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
            click_analysis.prepare_score_table(ctx)?
        } else {
            None
        };
        overlay_prepared_score_functional(
            &self.analysis.query,
            click_scores.as_ref(),
            &mut result,
        )?;
        self.stamp(&click_analysis.data, result)
    }

    /// Rebind the row-bound dispatch receipts to `data`. Boxed and out of
    /// line so the refresh frame never holds a second execution value.
    #[inline(never)]
    pub(super) fn rebound_dispatch_execution(
        &self,
        data: &DataInput,
    ) -> Result<Option<Box<PreparedExecution>>, CausalError> {
        let DataInput::Tabular(data) = data else {
            return Ok(None);
        };
        Ok(self.execution.rebind_for_dispatch(data)?.map(Box::new))
    }

    #[inline(never)]
    pub(super) fn install_execution(&mut self, execution: Box<PreparedExecution>) {
        self.execution = *execution;
    }
}
