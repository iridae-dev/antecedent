//! The one shape a compiled program-family handle takes.
//!
//! A program family compiles to a handle that retains its lowered program. Through
//! [`CompiledProgramHandle`] each such handle states that program, its plan view and the support
//! decision of the unit its route runs under, read from the route and never from the handle.
//! [`gate_program`] is the support gate the handle's execution passes through: a release build
//! refuses a route whose unit is not supported, or that no unit is bound to, and a development
//! build records what a release build refuses and runs the route. A support decision never
//! enters a program's identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, NotSupported};
use antecedent_io::method_provenance::MethodProvenanceWire;

use crate::planner::lowered::LoweredProgram;
use crate::planner::{PlanStage, PlanView};
use crate::support::unit::{UnitDecision, require_decided, route_decision};

/// What a route with no bound unit is missing, in a refusal.
const NO_UNIT: &str = "support_unit";

/// A compiled program-family handle: a retained lowered program and what is read from it.
pub(crate) trait CompiledProgramHandle {
    /// The lowered program the handle retains.
    fn program(&self) -> &Arc<LoweredProgram>;

    /// Every layer of the retained program.
    fn lowered(&self) -> &LoweredProgram {
        self.program()
    }

    /// Resolve the operation's own methods and inherit the snapshots on the evidence it read.
    fn result_method_provenance<'a>(
        &self,
        inputs: impl IntoIterator<Item = &'a MethodProvenanceWire>,
    ) -> MethodProvenanceWire
    where
        Self: Sized,
    {
        let mut provenance = self.lowered().method_provenance();
        provenance.inherit(inputs);
        provenance
    }

    /// The plan view of the retained program, as compiled.
    fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(self.program()), PlanStage::Compiled)
    }

    /// The support decision of the unit the route runs under, `None` while no unit is bound
    /// to the route.
    fn support(&self) -> Option<UnitDecision> {
        route_decision(self.lowered().route())
    }

    /// The calibration state of the unit the route runs under, `not_measured` while no unit is
    /// bound to the route. Reading it records no support decision.
    fn unit_calibration(&self) -> &'static str {
        crate::support::units_generated::unit_of(self.lowered().route())
            .map_or(crate::support::NOT_MEASURED, |unit| {
                crate::support::unit_calibration(&crate::support::UnitRef::Capability(unit))
            })
    }
}

/// Pass the support gate of `handle`'s route, with the dispatch-scoped context, before the
/// handle executes. The route names the entry point in a refusal.
///
/// # Errors
///
/// [`NotSupported`] in a release build when the route's unit is not supported, or when no unit
/// is bound to the route.
pub(crate) fn gate_program<H>(ctx: &ExecutionContext, handle: &H) -> Result<(), NotSupported>
where
    H: CompiledProgramHandle + ?Sized,
{
    let route = handle.lowered().route();
    let decision = route_decision(route);
    let result = match &decision {
        Some(decision) => require_decided(ctx, decision, route),
        None => antecedent_core::support_gate::require_supported(ctx, route, &[NO_UNIT]),
    };
    let returned = decision
        .as_ref()
        .filter(|decision| {
            result.is_ok() && decision.status == crate::support::CellStatus::Supported
        })
        .map(|decision| &decision.unit);
    crate::support::trace_route(route, returned);
    result
}

/// [`gate_program`] when `gate` is set. A family whose gate is still off runs its route whatever
/// the unit decided; the decision is recorded all the same, so the evidence the run produces
/// names the unit it ran under.
///
/// # Errors
///
/// As [`gate_program`], only when `gate` is set.
pub(crate) fn gate_program_when<H>(
    gate: bool,
    ctx: &ExecutionContext,
    handle: &H,
) -> Result<(), NotSupported>
where
    H: CompiledProgramHandle + ?Sized,
{
    if gate {
        return gate_program(ctx, handle);
    }
    let route = handle.lowered().route();
    let decision = route_decision(route);
    let returned = decision
        .as_ref()
        .filter(|decision| decision.status == crate::support::CellStatus::Supported)
        .map(|decision| &decision.unit);
    crate::support::trace_route(route, returned);
    Ok(())
}
