//! Dispatch entry and release support gate shared by every joint-panel compiler route.
//!
//! A public execution entry point of a joint-panel handle runs, in order:
//!
//! 1. [`enter`]: refuse a context already inside another program's lowered dispatch, then
//!    enter the dispatch of the program the handle retains;
//! 2. [`require_release_support`] with the scoped context and the handle's route;
//! 3. the unchanged engine with the scoped context. A result keeps the caller's context, not
//!    the scoped one.
//!
//! A public preparation entry point calls [`Entry::admit`] with [`Entry::Public`] before it
//! lowers. A lowering takes no admission lease.
//!
//! A child program lowers and executes only inside the dispatch of the program that encloses
//! it: a completed table of a multiple imputation, the retargeted response a continuous search
//! evaluates, the panel response an experiment update conditions. Its entry is
//! [`Entry::Child`] with the enclosing program's identity: [`Entry::admit`] before it lowers,
//! [`Entry::enter`] before it executes. Any other context is refused with
//! `public_builder_reentry`. The enclosing program has already passed the release gate, so a
//! child does not pass it again.
//!
//! # Release support
//!
//! Every route gates on the capability record that owns the joint-panel module and its earned
//! evidence, `joint_response.interval_readings` in `parity/capability_records.toml`; its unit
//! id in `antecedent-io-core/data/unit_support_status.json` is the same string. The record
//! output that covers each route:
//!
//! - `JointPanelResponse`, `JointPanelTimeVarying`, `JointPanelImputation`:
//!   `joint_response.interval_readings.values.1`, the fitted joint panel (`JointPanelResult`),
//!   its change-point and smooth selections and its parameter law;
//! - `JointPanelProgramSearch`: `joint_response.interval_readings.values.1`, program search,
//!   certificates and best-found programs;
//! - `JointPanelProgramSelection`: `joint_response.interval_readings.selected_value`, and
//!   `values.1` for training values and optimism diagnostics;
//! - `JointPanelExperiment`: `joint_response.interval_readings.updated_contrast`, and
//!   `values.1` for the terminal decision and the next-study ranking;
//! - `JointDesign`: `joint_response.interval_readings.native_contrast` and
//!   `joint_response.interval_readings.reference_region`.
//!
//! The `multiple_imputation` refusal namespace the imputation route raises is owned by
//! `estimate.randomized_missing_outcomes`; that record names the namespace's sources and does
//! not gate the route.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SemanticDigest};

use super::JointPanelError;
use crate::analysis::lowering::refuse_reentry;
use crate::error::CausalError;
use crate::planner::lowered::{JointPanelRoute, LoweredProgram};
use crate::support::CellStatus;

/// Capability unit that owns the joint-panel module and its earned evidence.
pub(crate) const JOINT_RESPONSE_CAPABILITY: &str = "joint_response.interval_readings";

/// The capability unit whose supported status gates `route` in a release build.
pub(crate) const fn capability(route: JointPanelRoute) -> &'static str {
    match route {
        JointPanelRoute::Response
        | JointPanelRoute::TimeVarying
        | JointPanelRoute::Imputation
        | JointPanelRoute::ProgramSearch
        | JointPanelRoute::ProgramSelection
        | JointPanelRoute::Experiment
        | JointPanelRoute::Design => JOINT_RESPONSE_CAPABILITY,
    }
}

/// Whether the capability unit gating `route` is supported in this build's derived status.
#[must_use]
pub(crate) fn release_supported(route: JointPanelRoute) -> bool {
    crate::support::capability_supported(capability(route))
}

/// The support status a lowering of `route` records on its scientific problem: supported
/// exactly when the route's capability unit is. It does not enter the program identity.
#[must_use]
pub(crate) fn support_status(route: JointPanelRoute) -> CellStatus {
    if release_supported(route) { CellStatus::Supported } else { CellStatus::NotSupported }
}

/// Pass the release support gate of `route` with the dispatch-scoped context. A release
/// build refuses a route whose capability unit is not supported; a development build records
/// what a release build refuses and runs the route.
///
/// # Errors
/// [`JointPanelError::Compiler`] carrying the `not_supported` refusal, in a release build.
pub(crate) fn require_release_support(
    ctx: &ExecutionContext,
    route: JointPanelRoute,
) -> Result<(), JointPanelError> {
    crate::support::require_capability_route(ctx, capability(route), route.cell().as_str())
        .map_err(|refusal| JointPanelError::Compiler(CausalError::from(refusal)))
}

/// How a joint-panel entry point is reached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Entry {
    /// A public call; it is refused inside any lowered dispatch.
    Public,
    /// A child of the enclosing program with this identity; it is admitted only inside that
    /// program's dispatch.
    Child(SemanticDigest),
}

impl Entry {
    /// Admit lowering or preparation through this entry.
    ///
    /// # Errors
    /// `public_builder_reentry`: a public call inside a lowered dispatch, or a child call
    /// outside the dispatch of its enclosing program.
    pub(crate) fn admit(self, ctx: &ExecutionContext) -> Result<(), JointPanelError> {
        match self {
            Self::Public => refuse_reentry(ctx)?,
            Self::Child(parent) => {
                if ctx.lowered_dispatch() != Some(parent) {
                    return Err(crate::unsupported_reason!(
                        "public_builder_reentry",
                        "a joint-panel child program lowers and executes only inside the \
                         dispatch of the program that encloses it"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Admit execution through this entry and enter the dispatch of `program`. The engines
    /// run with the returned context.
    ///
    /// # Errors
    /// As [`Self::admit`].
    pub(crate) fn enter(
        self,
        ctx: &ExecutionContext,
        program: &LoweredProgram,
    ) -> Result<ExecutionContext, JointPanelError> {
        self.admit(ctx)?;
        Ok(ctx.enter_lowered_dispatch(program.identity()))
    }
}

/// Enter the dispatch of `program` from a public execution entry point.
///
/// # Errors
/// `public_builder_reentry` when `ctx` is already inside a lowered dispatch.
pub(crate) fn enter(
    ctx: &ExecutionContext,
    program: &LoweredProgram,
) -> Result<ExecutionContext, JointPanelError> {
    Entry::Public.enter(ctx, program)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ExecutionContext, SemanticDigest};

    use super::{Entry, capability, release_supported};
    use crate::planner::lowered::JointPanelRoute;

    #[test]
    fn every_route_gates_on_a_supported_capability_unit() {
        for route in JointPanelRoute::ALL {
            let cell = route.cell().as_str();
            assert!(release_supported(route), "{cell} gates on {}", capability(route));
        }
    }

    #[test]
    fn a_child_is_admitted_only_inside_its_enclosing_dispatch() {
        let ctx = ExecutionContext::for_tests(1);
        let parent = SemanticDigest::from_bytes([1; 32]);
        let other = SemanticDigest::from_bytes([2; 32]);
        assert!(Entry::Public.admit(&ctx).is_ok());
        assert!(Entry::Child(parent).admit(&ctx).is_err());
        let inside = ctx.enter_lowered_dispatch(parent);
        assert!(Entry::Child(parent).admit(&inside).is_ok());
        assert!(Entry::Child(other).admit(&inside).is_err());
        assert!(Entry::Public.admit(&inside).is_err());
    }
}
