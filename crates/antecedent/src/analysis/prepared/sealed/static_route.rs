//! Sealed static routes: counterfactual, tiered average, derivative and DAG
//! responses, joint cells, mediation, attribution, and the randomized-design
//! families.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealedOperation, SeriesCall, stamp_tabular};
use crate::analysis::builder::DataInput;
use crate::analysis::execute::finish_family_result;
use crate::analysis::execute::{
    CheckedAssociationOperation, CheckedAttributionOperation,
    CheckedBayesianStaticMediationOperation, CheckedCellAipwResponseOperation,
    CheckedContinuousDoseOperation, CheckedCounterfactualPlan, CheckedDerivativeResponseOperation,
    CheckedHistoryMatchOperation, CheckedInterferenceOperation, CheckedIvBoundsOperation,
    CheckedLocalPolynomialRatioOperation, CheckedLongitudinalRegimeOperation,
    CheckedPanelDidOperation, CheckedPolicyValueOperation, CheckedRandomizedOperation,
    CheckedRouteOperation, CheckedSequentialContrastOperation, CheckedStaticDagResponseOperation,
    CheckedStaticMediationOperation, CheckedSurvivalOperation, CheckedSyntheticControlOperation,
    CheckedTemporalDesignOperation,
};
use crate::analysis::prepared::estimate::{ClickNotes, Rebound};
use crate::analysis::prepared::{
    CheckedStaticResponseCurve, CheckedUnknownTieredAverageOperation, PreparedStudy,
};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;

use crate::analysis::execute::CheckedDiscoveryOperation;

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
    CheckedTemporalDesignOperation,
    CheckedIvBoundsOperation,
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
        self.execute_series(call.series, call.provenance, None, None, call.ctx).map(Some)
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
            self.refresh(&handle.analysis.graph, handle.analysis.declared.tiered.as_ref(), data)?;
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

/// A route sealed to its lowered program: structure discovery, panel and
/// multi-environment temporal effects, temporal cross-world queries and temporal
/// ADMG effects. Each execution requires the plan to still carry the program the
/// operation retained; the click then runs inside that program's dispatch scope.
impl SealedOperation for CheckedRouteOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        _data: &TabularData,
        _ctx: &ExecutionContext,
        _notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&handle.plan)?;
        Ok(None)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&call.handle.plan)?;
        Ok(None)
    }
}

/// The association route: the click runs the association executor on the click's rows or
/// series, under the program this operation was sealed to.
impl SealedOperation for CheckedAssociationOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&handle.plan)?;
        let mut result = handle.analysis.execute_association_tabular(data, &handle.plan, ctx)?;
        finish_family_result(
            &handle.analysis,
            &DataInput::Tabular(data.clone()),
            &handle.plan,
            &mut result,
        )?;
        stamp_tabular(handle, data, result, notes)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&call.handle.plan)?;
        let mut result =
            call.study.execute_association_series(call.series, &call.handle.plan, call.ctx)?;
        finish_family_result(call.study, call.input, &call.handle.plan, &mut result)?;
        Ok(Some(result))
    }
}

/// The structure-discovery route: the click runs the search on the click's data, under the
/// program this operation was sealed to.
impl SealedOperation for CheckedDiscoveryOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&handle.plan)?;
        let mut result = handle.analysis.execute_discover_tabular(data, &handle.plan, ctx)?;
        finish_family_result(
            &handle.analysis,
            &DataInput::Tabular(data.clone()),
            &handle.plan,
            &mut result,
        )?;
        stamp_tabular(handle, data, result, notes)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&call.handle.plan)?;
        let mut result = call.study.execute_discover(call.input, &call.handle.plan, call.ctx)?;
        finish_family_result(call.study, call.input, &call.handle.plan, &mut result)?;
        Ok(Some(result))
    }
}

/// Mixed/class cross-world clicks execute their own scientific family directly.
impl SealedOperation for crate::analysis::execute::CheckedStaticCrossWorldOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        self.require_program(&handle.plan)?;
        let rebound = self.rebind(data)?;
        let mut result = rebound.execute_checked_static_cross_world(
            &handle.analysis,
            data,
            &handle.plan,
            ctx,
        )?;
        finish_family_result(
            &handle.analysis,
            &DataInput::Tabular(data.clone()),
            &handle.plan,
            &mut result,
        )?;
        let stamped = stamp_tabular(handle, data, result, notes);
        if stamped.is_ok() {
            notes.record(Rebound::Dispatch(Box::new(
                crate::analysis::prepared::PreparedExecution::StaticCrossWorld(rebound),
            )));
        }
        stamped
    }
}

impl SealedOperation for crate::analysis::execute::CheckedClassAttributionOperation {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();

    fn require_retained_program(
        &self,
        plan: &crate::planner::PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        let mut result = self.execute_checked_class_attribution_table(
            &handle.analysis,
            data,
            &handle.plan,
            ctx,
        )?;
        finish_family_result(
            &handle.analysis,
            &DataInput::Tabular(data.clone()),
            &handle.plan,
            &mut result,
        )?;
        stamp_tabular(handle, data, result, notes)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        let mut result = self.execute_checked_class_attribution_series(
            call.study,
            call.series,
            &call.handle.plan,
            call.provenance,
            call.ctx,
        )?;
        finish_family_result(call.study, call.input, &call.handle.plan, &mut result)?;
        Ok(Some(result))
    }
}
