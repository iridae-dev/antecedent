//! Sealed program routes: linear and AIPW adjustment, ADMG response curves,
//! path-specific and functional effects, distributions, front-door, IV, and the
//! nested counterfactual.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, stamp_tabular};
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::prepared::estimate::{ClickNotes, Rebound};
use crate::analysis::prepared::{
    CheckedAdmgResponseCurveOperation, CheckedAipwOperation, CheckedDistributionOperation,
    CheckedFrontDoorOperation, CheckedFunctionalEffectOperation, CheckedIvOperation,
    CheckedLinearOperation, CheckedPathSpecificEffectOperation, CheckedProgramBinding,
    PreparedExecution, PreparedStudy,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use std::sync::Arc;

/// Stamp the result of a rebound functional-effect family operation: its
/// caller-attested validator names, the likelihood disclosure of its inference,
/// and the contract it executed under.
fn stamp_functional(
    handle: &PreparedStudy,
    data: &TabularData,
    mut result: StudyResult,
    custom_validator_names: &[Arc<str>],
    inference: &InferenceMode,
    refute: RefuteSuite,
    defer: bool,
) -> Result<Option<StudyResult>, CausalError> {
    let input = DataInput::Tabular(data.clone());
    result.custom_validator_names = custom_validator_names.to_vec();
    crate::analysis::execute::push_gaussian_likelihood_disclosure(&mut result, inference, &input);
    if !defer {
        result.executed_contract = Some(handle.executed_contract(&input, refute, None)?);
    }
    Ok(Some(result))
}

impl SealedOperation for CheckedLinearOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::Linear(self)
    }

    // The operation owns the executable copy of the click controls. Updating
    // only the study would leave a sealed route unable to stream.
    fn set_stage_sink(&mut self, sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>) {
        self.stage_sink = sink;
    }

    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::CheckedLinear(self.rebind(data)?)))
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
        let rebound = self.rebind_with_context(data, ctx)?;
        let result = rebound.execute(data, ctx)?;
        let result = handle.stamp_linear(
            &DataInput::Tabular(data.clone()),
            &rebound,
            result,
            notes.defers_contract(),
        )?;
        notes.record(Rebound::Linear(Box::new(rebound)));
        Ok(Some(result))
    }
}

impl SealedOperation for CheckedAdmgResponseCurveOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::FunctionalResponse(&self.members)
    }
    study_owned_stage_sink!();
    study_owned_validators!();

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
        let result = rebound.execute(data, ctx)?;
        stamp_functional(
            handle,
            data,
            result,
            &rebound.custom_validator_names,
            &rebound.inference,
            rebound.refute,
            notes.defers_contract(),
        )
    }
}

impl SealedOperation for CheckedPathSpecificEffectOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::FunctionalEffect(self.prepared.program())
    }
    study_owned_stage_sink!();
    study_owned_validators!();

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
        let result = rebound.execute(data, ctx)?;
        stamp_functional(
            handle,
            data,
            result,
            &rebound.custom_validator_names,
            &rebound.inference,
            rebound.refute,
            notes.defers_contract(),
        )
    }
}

impl SealedOperation for CheckedFunctionalEffectOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::FunctionalEffect(self.prepared.program())
    }
    study_owned_stage_sink!();
    study_owned_validators!();

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
        let result = rebound.execute(data, ctx)?;
        stamp_functional(
            handle,
            data,
            result,
            &rebound.custom_validator_names,
            &rebound.inference,
            rebound.refute,
            notes.defers_contract(),
        )
    }
}

impl SealedOperation for CheckedAipwOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::Aipw(&self.preparation)
    }
    study_owned_stage_sink!();
    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::CheckedAipw(self.rebind(data)?)))
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
        let mut result = rebound.execute(data, ctx)?;
        result.custom_validator_names = rebound.custom_validator_names.to_vec();
        let result = handle.stamp_aipw(
            &DataInput::Tabular(data.clone()),
            &rebound,
            result,
            notes.defers_contract(),
        )?;
        notes.record(Rebound::Aipw(Box::new(rebound)));
        Ok(Some(result))
    }
}

impl SealedOperation for CheckedDistributionOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::Distribution(self.prepared.program())
    }
    study_owned_stage_sink!();
    study_owned_validators!();

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
        let mut result = self.execute(data, ctx)?;
        result.custom_validator_names = self.custom_validator_names.to_vec();
        handle
            .stamp_distribution(
                &DataInput::Tabular(data.clone()),
                self,
                result,
                notes.defers_contract(),
            )
            .map(Some)
    }
}

impl SealedOperation for CheckedFrontDoorOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::FrontDoor(&self.preparation)
    }
    study_owned_stage_sink!();
    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::FrontDoorLinear(self.rebind(data)?)))
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_frontdoor(data, &rebound, ctx)?;
        let result = stamp_tabular(handle, data, result, notes);
        notes.record(Rebound::FrontDoor(Box::new(rebound)));
        result
    }
}

impl SealedOperation for CheckedIvOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::Iv(self.preparation())
    }
    study_owned_stage_sink!();
    study_owned_validators!();

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(Some(PreparedExecution::Iv(self.rebind(data)?)))
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let rebound = self.rebind(data)?;
        let result = handle.execute_checked_iv(data, &rebound, ctx)?;
        let result = stamp_tabular(handle, data, result, notes);
        notes.record(Rebound::Iv(Box::new(rebound)));
        result
    }
}

impl SealedOperation for crate::gcm::NestedCounterfactualOperation {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        CheckedProgramBinding::NestedCounterfactual(self)
    }
    study_owned_stage_sink!();
    study_owned_validators!();
}
