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
    ) -> Result<Option<StudyResult>, CausalError> {
        let mut result = self.execute(data, ctx)?;
        result.custom_validator_names = self.custom_validator_names();
        handle.stamp_with_likelihood(data, self.operation().inference(), result).map(Some)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        handle.stamp_with_likelihood(data, &self.inference(), result).map(Some)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        stamp_tabular(handle, data, result)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        stamp_tabular(handle, data, result)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let result = self.execute(data, ctx)?;
        handle.stamp_with_likelihood(data, self.inference(), result).map(Some)
    }
}

impl SealedOperation for CheckedGlmAdjustmentOperation {
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
    ) -> Result<Option<StudyResult>, CausalError> {
        if !self.sealed_for_direct_execution() {
            return Ok(None);
        }
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_glm(data, &rebound, ctx)?;
        stamp_tabular(handle, data, result)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_rd(data, &rebound, ctx)?;
        stamp_tabular(handle, data, result)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_propensity(data, &rebound, ctx)?;
        stamp_tabular(handle, data, result)
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
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_conditional(data, &rebound, ctx)?;
        stamp_tabular(handle, data, result)
    }
}
