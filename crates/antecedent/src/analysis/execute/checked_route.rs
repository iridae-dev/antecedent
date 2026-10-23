//! A route sealed by the lowered program compile attached to its plan.
//!
//! Structure discovery, panel and multi-environment temporal effects, temporal
//! cross-world queries and temporal ADMG effects execute from the study the handle
//! froze, inside the dispatch scope of the lowered program. This operation retains
//! the route cell and the identity of that program at preparation. Every execution
//! first requires that the plan still carries that program, lowered for that cell, so
//! a plan swapped for another program, or no program at all, is refused with
//! `route_not_lowered` before any work runs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::SemanticDigest;

use super::{CausalError, PhysicalExecutionPlan, Study};
use crate::analysis::lowering::{RouteCell, expected_cell};

/// The route cell and the lowered program a prepared handle executes.
#[derive(Clone, Debug)]
pub(crate) struct CheckedRouteOperation {
    cell: RouteCell,
    lowered: SemanticDigest,
}

impl CheckedRouteOperation {
    /// Seal `study` to the program `physical` carries for `cell`.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the plan carries no program lowered for `cell`, or the
    /// study executes as another route.
    pub(crate) fn checked(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
        cell: RouteCell,
    ) -> Result<Self, CausalError> {
        let lowered = physical.require_lowered(Some(cell))?.identity();
        if expected_cell(study) != Some(cell) {
            return Err(crate::compile_reason!(
                "route_not_lowered",
                "the study does not execute as {}",
                cell.as_str()
            ));
        }
        Ok(Self { cell, lowered })
    }

    /// Require that `physical` still carries the program this operation retained.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the plan carries no program, one lowered for another
    /// route, or another program of the same route.
    pub(crate) fn require_program(
        &self,
        physical: &PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        let lowered = physical.require_lowered(Some(self.cell))?.identity();
        if lowered == self.lowered {
            Ok(())
        } else {
            Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan carries another lowered program than the one {} was sealed to",
                self.cell.as_str()
            ))
        }
    }
}
