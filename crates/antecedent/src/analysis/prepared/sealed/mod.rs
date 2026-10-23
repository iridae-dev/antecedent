//! What every sealed operation shares: the one trait a prepared handle
//! dispatches through, and the families that implement it.
//!
//! A [`PreparedExecution`](super::PreparedExecution) holds at most one sealed
//! operation. Everything the handle does with that operation (binding its
//! program to a result contract, refreshing its click controls, rebinding it to
//! new rows, executing it on tabular or series data) is a method of
//! [`SealedOperation`], reached through the exhaustive projection the
//! `sealed_executions!` list generates. No call site matches on the variants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::estimate::ClickNotes;
use super::{CheckedProgramBinding, PreparedExecution, PreparedStudy};
use crate::analysis::builder::DataInput;
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use antecedent_core::ExecutionContext;
use antecedent_data::{TabularData, TimeSeriesData};
use std::sync::Arc;

/// The operation reads the stage sink from the study at execution, so a sink
/// swap has nothing to refresh on it.
macro_rules! study_owned_stage_sink {
    () => {
        fn set_stage_sink(
            &mut self,
            _sink: Option<std::sync::Arc<dyn crate::analysis::stage::StageResultSink>>,
        ) {
        }
    };
}

/// The operation reads caller validators from the study at execution, so a
/// validator rebind has nothing to refresh on it.
macro_rules! study_owned_validators {
    () => {
        fn set_custom_validators(
            &mut self,
            _validators: &[std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>],
        ) {
        }
    };
}

/// The operation contributes no checked program to a result contract.
macro_rules! unbound_program {
    () => {
        fn program_binding(&self) -> crate::analysis::prepared::CheckedProgramBinding<'_> {
            crate::analysis::prepared::CheckedProgramBinding::None
        }
    };
}

mod class;
mod estimator;
mod program;
mod static_route;
mod temporal;

/// A sealed operation that has a public inspection record.
///
/// The record is built by the operation itself, from the state it retains, and
/// the public `checked_*_info` accessors of [`PreparedStudy`] are generated
/// from the list that binds each accessor to its sealed route. An operation
/// without a public record does not implement this trait.
pub(crate) trait Inspect {
    /// The public `Checked*Info` record of this operation.
    type Info;

    /// Read the record from the retained operation.
    fn info(&self) -> Self::Info;
}

/// A retained, checked operation of a prepared handle.
///
/// The three control hooks are required so each operation states how it holds
/// its click controls and program; `study_owned_stage_sink!`,
/// `study_owned_validators!` and `unbound_program!` state the common answer.
/// The execution hooks default to "this route has no sealed direct execution
/// on that modality": the shared tabular dispatcher (or, for series, the
/// ordinary executor) runs it from the retained study instead.
pub(super) trait SealedOperation {
    /// The checked program this operation binds into a result contract.
    fn program_binding(&self) -> CheckedProgramBinding<'_>;

    /// Push a swapped stage sink into the executable copy of the click
    /// controls this operation owns.
    fn set_stage_sink(&mut self, sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>);

    /// Push rebound caller validators into the operation's frozen copy.
    fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    );

    /// Rebind only the checked receipts that still execute through the shared
    /// tabular dispatcher; `None` for every operation that executes as
    /// prepared. Keeping the selection here prevents a refreshed receipt from
    /// changing the procedure selected at preparation.
    fn rebind_for_dispatch(
        &self,
        _data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        Ok(None)
    }

    /// Execute from the sealed route on tabular `data`. `None` leaves the
    /// click to the shared tabular dispatcher.
    ///
    /// An operation that rebinds itself to the click's rows records the
    /// rebound operation in `notes`, and leaves the result's executed contract
    /// unset when `notes` defers it.
    fn estimate_tabular(
        &self,
        _handle: &PreparedStudy,
        _data: &TabularData,
        _ctx: &ExecutionContext,
        _notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        Ok(None)
    }

    /// Execute from the sealed route over series data. `None` leaves the click
    /// to the ordinary executor.
    fn estimate_series(&self, _call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        Ok(None)
    }

    /// Require that `plan` still carries the lowered program this operation was
    /// sealed to. An operation that retains no program of its own accepts any plan
    /// (the handle's dispatch already requires one).
    fn require_retained_program(&self, _plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        Ok(())
    }
}

/// A boxed operation is the operation: the large variants are boxed to keep
/// the enum small, not to change what they do.
impl<T: SealedOperation + ?Sized> SealedOperation for Box<T> {
    fn program_binding(&self) -> CheckedProgramBinding<'_> {
        (**self).program_binding()
    }

    fn set_stage_sink(&mut self, sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>) {
        (**self).set_stage_sink(sink);
    }

    fn set_custom_validators(
        &mut self,
        validators: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    ) {
        (**self).set_custom_validators(validators);
    }

    fn rebind_for_dispatch(
        &self,
        data: &TabularData,
    ) -> Result<Option<PreparedExecution>, CausalError> {
        (**self).rebind_for_dispatch(data)
    }

    fn estimate_tabular(
        &self,
        handle: &PreparedStudy,
        data: &TabularData,
        ctx: &ExecutionContext,
        notes: &mut ClickNotes,
    ) -> Result<Option<StudyResult>, CausalError> {
        (**self).estimate_tabular(handle, data, ctx, notes)
    }

    fn estimate_series(&self, call: &SeriesCall<'_>) -> Result<Option<StudyResult>, CausalError> {
        (**self).estimate_series(call)
    }

    fn require_retained_program(&self, plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        (**self).require_retained_program(plan)
    }
}

/// The route no sealed operation admits: the ordinary dispatcher executes it.
#[derive(Clone, Copy, Debug)]
pub(super) struct OrdinaryDispatch;

impl SealedOperation for OrdinaryDispatch {
    unbound_program!();
    study_owned_stage_sink!();
    study_owned_validators!();
}

/// Which entry point is running a series operation. It selects the wording of
/// each refusal; the refusal's cause is the same.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SeriesEntry {
    /// [`PreparedStudy::estimate_series`], over the prepared study.
    Estimate,
    /// [`PreparedStudy::refresh_series`], over the refreshed clone.
    Refresh,
}

/// One refusal of a series operation, worded for each entry point.
#[derive(Clone, Copy, Debug)]
pub(super) struct SeriesRefusal {
    estimate: &'static str,
    refresh: &'static str,
}

impl SeriesRefusal {
    pub(super) const fn new(estimate: &'static str, refresh: &'static str) -> Self {
        Self { estimate, refresh }
    }
}

/// Everything a sealed series operation reads at execution.
///
/// `study` is the prepared study (estimate) or its refreshed clone (refresh);
/// `input` is the series data wrapped in the modality the handle was prepared
/// with.
pub(super) struct SeriesCall<'a> {
    pub(super) handle: &'a PreparedStudy,
    pub(super) study: &'a Study,
    pub(super) input: &'a DataInput,
    pub(super) series: &'a TimeSeriesData,
    pub(super) ctx: &'a ExecutionContext,
    pub(super) entry: SeriesEntry,
}

impl<'a> SeriesCall<'a> {
    /// The compile error for `refusal`, worded for this entry point.
    pub(super) fn refusal(&self, refusal: SeriesRefusal) -> CausalError {
        CausalError::Compile {
            message: match self.entry {
                SeriesEntry::Estimate => refusal.estimate,
                SeriesEntry::Refresh => refusal.refresh,
            }
            .into(),
        }
    }

    /// The study's `TemporalDag`, or `lost` when it has none.
    pub(super) fn temporal_dag(
        &self,
        lost: SeriesRefusal,
    ) -> Result<&'a crate::graph::TemporalDag, CausalError> {
        self.study.graph.as_temporal_dag().ok_or_else(|| self.refusal(lost))
    }
}

/// Stamp `result` as executed on tabular `data` by `handle`.
pub(super) fn stamp_tabular(
    handle: &PreparedStudy,
    data: &TabularData,
    result: StudyResult,
    notes: &ClickNotes,
) -> Result<Option<StudyResult>, CausalError> {
    handle
        .stamp_deferrable(&DataInput::Tabular(data.clone()), result, notes.defers_contract())
        .map(Some)
}
