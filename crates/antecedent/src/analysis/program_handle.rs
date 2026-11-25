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

use crate::planner::lowered::LoweredProgram;
use crate::planner::{PlanStage, PlanView};
use crate::support::unit::{SupportKey, require_program};
use crate::support::units_generated::unit_of;

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

    /// The plan view of the retained program, as compiled.
    fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(self.program()), PlanStage::Compiled)
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
    match unit_of(route) {
        Some(unit) => require_program(ctx, &SupportKey::Unit(unit), route),
        None => antecedent_core::support_gate::require_supported(ctx, route, &[NO_UNIT]),
    }
}
