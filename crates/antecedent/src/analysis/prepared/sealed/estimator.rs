//! Sealed estimator routes: Bayesian g-computation, basis, robust, and
//! conditional effects, and GLM, RD, propensity, and conditional adjustment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, stamp_tabular};
use crate::analysis::CheckedConditionalOperation;
use crate::analysis::checked_propensity::CheckedPropensityOperation;
use crate::analysis::execute::{
    CheckedBayesianBasisAteExecution, CheckedBayesianBasisCateExecution,
    CheckedBayesianConditionalOperation, CheckedBayesianDagAteExecution,
    CheckedBayesianRobustAteExecution,
};
use crate::analysis::prepared::estimate::{ClickNotes, Rebound};
use crate::analysis::prepared::{
    CheckedGlmAdjustmentOperation, CheckedRdOperation, PreparedExecution, PreparedStudy,
};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use std::sync::Arc;

impl SealedOperation for CheckedBayesianDagAteExecution {
    unbound_program!();

    fn set_stage_sink(&mut self, sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>) {
        Self::set_stage_sink(self, sink);
    }

    fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        Self::set_custom_validators(self, validators.to_vec());
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let mut result = self.execute(data, ctx)?;
        result.extras.custom_validator_names = self.custom_validator_names();
        handle
            .stamp_with_likelihood(
                data,
                self.operation().inference(),
                result,
                notes.defers_contract(),
                ctx,
            )
            .map(Some)
    }
}

impl SealedOperation for CheckedBayesianBasisAteExecution {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        handle
            .stamp_with_likelihood(data, &self.inference(), result, notes.defers_contract(), ctx)
            .map(Some)
    }
}

impl SealedOperation for CheckedBayesianBasisCateExecution {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        stamp_tabular(handle, data, result, notes, ctx)
    }
}

impl SealedOperation for CheckedBayesianRobustAteExecution {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        stamp_tabular(handle, data, result, notes, ctx)
    }
}

impl SealedOperation for CheckedBayesianConditionalOperation {
    unbound_program!();

    fn set_stage_sink(&mut self, sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>) {
        Self::set_stage_sink(self, sink);
    }

    fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        Self::set_custom_validators(self, validators.to_vec());
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        handle
            .stamp_with_likelihood(data, self.inference(), result, notes.defers_contract(), ctx)
            .map(Some)
    }
}

impl SealedOperation for CheckedGlmAdjustmentOperation {
    fn adjusted_reuse(&self) -> Option<super::AdjustedReuse<'_>> {
        self.adjusted_reuse_descriptor()
    }
    fn adjusted_retarget_ready(&self) -> bool {
        self.has_adjusted_fit()
    }
    fn retarget_adjusted(
        &self,
        weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Option<super::AdjustedRetargetOutput>, CausalError> {
        self.retarget_fitted(weights, ctx).map(Some)
    }

    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::CheckedGlmAdjustment(self.rebind(data)?)))
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        if !self.sealed_for_direct_execution() {
            return Ok(None);
        }
        let rebound = self.rebind(data)?;
        let run = handle.execute_checked_glm(data, &rebound, ctx)?;
        let result = stamp_tabular(handle, data, run.result, notes, ctx)?;
        if let (Some(capture), Some(effect)) =
            (run.capture, result.as_ref().and_then(|r| r.estimate.as_effect()))
        {
            rebound.publish_fitted_point(capture.point, effect, capture.retention, ctx)?;
        }
        notes.record(Rebound::GlmAdjustment(Box::new(rebound)));
        Ok(result)
    }
}

impl SealedOperation for CheckedRdOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::CheckedRd(self.rebind(data)?)))
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_rd(data, &rebound, ctx)?;
        let result = stamp_tabular(handle, data, result, notes, ctx);
        notes.record(Rebound::Rd(Box::new(rebound)));
        result
    }
}

impl SealedOperation for CheckedPropensityOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_propensity(data, &rebound, ctx)?;
        stamp_tabular(handle, data, result, notes, ctx)
    }
}

impl SealedOperation for CheckedConditionalOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_conditional(data, &rebound, ctx)?;
        let result = stamp_tabular(handle, data, result, notes, ctx);
        notes.record(Rebound::Conditional(Box::new(rebound)));
        result
    }
}
