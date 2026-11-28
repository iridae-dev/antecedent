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

/// Identifies a gated program until its output can be published.
#[must_use = "publish the successful result, or name the admission when this is only preparation"]
#[derive(Debug)]
pub(crate) struct ProgramPublication {
    #[cfg(feature = "evidence-trace")]
    route: Option<&'static str>,
    #[cfg(feature = "evidence-trace")]
    unit: Option<crate::support::UnitRef>,
}

impl ProgramPublication {
    /// Nested work publishes under its enclosing program.
    pub(crate) const fn nested() -> Self {
        Self {
            #[cfg(feature = "evidence-trace")]
            route: None,
            #[cfg(feature = "evidence-trace")]
            unit: None,
        }
    }

    /// Finish an execution. A refusal or a failed kernel returned no scientific result.
    pub(crate) fn finish<T, E>(self, result: Result<T, E>) -> Result<T, E> {
        #[cfg(feature = "evidence-trace")]
        let result = self.finish_with(result, |route, unit| {
            crate::support::trace_route(route, unit);
            // Standalone program outputs lack the complete evidence encoding of StudyResult.
            crate::support::trace_output(|| None);
        });
        result
    }

    #[cfg(feature = "evidence-trace")]
    fn finish_with<T, E>(
        self,
        result: Result<T, E>,
        returned: impl FnOnce(&'static str, Option<&crate::support::UnitRef>),
    ) -> Result<T, E> {
        result.inspect(|_| {
            if let Some(route) = self.route {
                returned(route, self.unit.as_ref());
            }
        })
    }
}

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
pub(crate) fn gate_program<H>(
    ctx: &ExecutionContext,
    handle: &H,
) -> Result<ProgramPublication, NotSupported>
where
    H: CompiledProgramHandle + ?Sized,
{
    gate_named_program(ctx, handle.lowered().route())
}

/// Apply the same gate when a replay has only the route name.
pub(crate) fn gate_named_program(
    ctx: &ExecutionContext,
    route: &'static str,
) -> Result<ProgramPublication, NotSupported> {
    gate_with_decision(ctx, route, route_decision(route))
}

pub(crate) fn gate_with_decision(
    ctx: &ExecutionContext,
    route: &'static str,
    decision: Option<UnitDecision>,
) -> Result<ProgramPublication, NotSupported> {
    match &decision {
        Some(decision) => require_decided(ctx, decision, route),
        None => antecedent_core::support_gate::require_supported(ctx, route, &[NO_UNIT]),
    }?;
    Ok(ProgramPublication {
        #[cfg(feature = "evidence-trace")]
        route: Some(route),
        #[cfg(feature = "evidence-trace")]
        unit: decision.map(|decision| decision.unit),
    })
}

#[cfg(all(test, feature = "evidence-trace"))]
mod tests {
    use super::*;
    use crate::support::{CellStatus, UnitId, UnitRef};

    fn candidate(status: CellStatus) -> UnitDecision {
        UnitDecision {
            status,
            unit: UnitRef::Capability(UnitId::new("fixture.program")),
            missing_evidence: (status == CellStatus::NotSupported)
                .then_some("fixture.oracle")
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn a_program_gate_cannot_publish_a_failed_kernel() {
        let ctx = ExecutionContext::for_tests(2);
        let publication =
            gate_with_decision(&ctx, "Fixture", Some(candidate(CellStatus::Supported)))
                .expect("the supported fixture passes its gate");
        let result: Result<(), &str> =
            publication.finish_with(Err("kernel failed"), |_, _| panic!("nothing was returned"));
        assert_eq!(result, Err("kernel failed"));
    }

    #[test]
    fn an_owed_route_returns_its_unit_only_if_development_runs_it() {
        let ctx = ExecutionContext::for_tests(2);
        let result = gate_with_decision(&ctx, "Fixture", Some(candidate(CellStatus::NotSupported)));
        #[cfg(not(feature = "development"))]
        assert_eq!(result.unwrap_err().missing, ["fixture.oracle"]);
        #[cfg(feature = "development")]
        {
            let publication = result.expect("development can execute an owed fixture");
            let returned_unit = candidate(CellStatus::NotSupported).unit;
            let mut observed = false;
            let returned: Result<u32, ()> = publication.finish_with(Ok(7), |route, unit| {
                assert_eq!(route, "Fixture");
                assert_eq!(unit, Some(&returned_unit));
                observed = true;
            });
            assert_eq!(returned, Ok(7));
            assert!(observed);
            assert_eq!(ctx.would_refuse()[0].missing, ["fixture.oracle"]);
        }
    }
}
