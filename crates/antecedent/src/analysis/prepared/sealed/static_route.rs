//! Sealed static routes: counterfactual, tiered average, derivative and DAG
//! responses, joint cells, mediation, attribution, and the randomized-design
//! families.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, SeriesCall, stamp_tabular};
use crate::analysis::execute::{
    CheckedAttributionOperation, CheckedBayesianStaticMediationOperation,
    CheckedCellAipwResponseOperation, CheckedContinuousDoseOperation, CheckedCounterfactualPlan,
    CheckedDerivativeResponseOperation, CheckedHistoryMatchOperation, CheckedInterferenceOperation,
    CheckedLocalPolynomialRatioOperation, CheckedLongitudinalRegimeOperation,
    CheckedPanelDidOperation, CheckedPolicyValueOperation, CheckedRandomizedOperation,
    CheckedSequentialContrastOperation, CheckedStaticDagResponseOperation,
    CheckedStaticMediationOperation, CheckedSurvivalOperation, CheckedSyntheticControlOperation,
};
use crate::analysis::prepared::estimate::{ClickNotes, Rebound};
use crate::analysis::prepared::{
    CheckedStaticResponseCurve, CheckedUnknownTieredAverageOperation, PreparedStudy,
};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;

/// Operations that execute on the click's rows exactly as prepared and are
/// stamped as plain tabular results.
macro_rules! executes_as_prepared {
    ($($operation:ty),+ $(,)?) => {$(
        impl SealedOperation for $operation {
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
                stamp_tabular(handle, data, result, notes)
            }
        }
    )+};
}

executes_as_prepared!(
    CheckedCounterfactualPlan,
    CheckedStaticMediationOperation,
    CheckedBayesianStaticMediationOperation,
    CheckedInterferenceOperation,
    CheckedRandomizedOperation,
    CheckedPolicyValueOperation,
    CheckedContinuousDoseOperation,
    CheckedPanelDidOperation,
    CheckedSyntheticControlOperation,
    CheckedLocalPolynomialRatioOperation,
    CheckedSurvivalOperation,
    CheckedLongitudinalRegimeOperation,
    CheckedSequentialContrastOperation,
    CheckedHistoryMatchOperation,
);

impl SealedOperation for CheckedAttributionOperation {
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
        stamp_tabular(handle, data, result, notes)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        self.execute_series(call.series, call.ctx).map(Some)
    }
}

impl SealedOperation for CheckedUnknownTieredAverageOperation {
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
        let result = handle.analysis.execute_checked_unknown_tiered_average(
            data,
            &self.query,
            &self.physical,
            ctx,
            crate::analysis::execute::CheckedUnknownTiered {
                background: &self.background,
                identification: self.cache.identification.clone(),
            },
            crate::analysis::execute::Strategy::new(self.identifier, self.estimator),
        )?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedDerivativeResponseOperation {
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
        let result =
            handle.analysis.execute_checked_derivative_response(data, &handle.plan, ctx, self)?;
        stamp_tabular(handle, data, result, notes)
    }
}

impl SealedOperation for CheckedStaticDagResponseOperation {
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
        let result =
            handle.analysis.execute_checked_static_dag_response(data, &handle.plan, ctx, self)?;
        handle
            .stamp_with_likelihood(data, self.inference(), result, notes.defers_contract())
            .map(Some)
    }
}

impl SealedOperation for CheckedCellAipwResponseOperation {
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
        let rebound =
            self.refresh(&handle.analysis.graph, handle.analysis.tiered.as_ref(), data)?;
        let result = handle.analysis.execute_checked_cell_aipw_response(
            data,
            &handle.plan,
            ctx,
            &rebound,
        )?;
        let result = stamp_tabular(handle, data, result, notes);
        notes.record(Rebound::CellAipwResponse(Box::new(rebound)));
        result
    }
}

/// The static response curve executes through the shared tabular dispatcher,
/// which reads it back through `PreparedExecution::response_curve`.
impl SealedOperation for CheckedStaticResponseCurve {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();
}
