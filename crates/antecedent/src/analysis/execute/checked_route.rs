//! A route sealed by the lowered program compile attached to its plan.
//!
//! Family operations execute from the study the handle froze, inside the dispatch
//! scope of the lowered program. The shared route guard retains
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
    cell: Option<RouteCell>,
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
        Ok(Self { cell: Some(cell), lowered })
    }

    /// Seal `study` to whatever program `physical` carries for the route the study
    /// executes as. This is the operation of a route no specialised sealer admits, and
    /// of a fresh run (`Study::execute`, one-shot `run`): it runs the route's executor
    /// from the study inside the program's dispatch scope, and a result it produces
    /// carries no prepared-click stamp.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the plan carries no lowered program, or one lowered for
    /// another route than the study executes as.
    pub(crate) fn lowered(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let cell = expected_cell(study);
        let lowered = physical.require_lowered(cell)?.identity();
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
        let lowered = physical.require_lowered(self.cell)?.identity();
        if lowered == self.lowered {
            Ok(())
        } else {
            Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan carries another lowered program than the one {} was sealed to",
                self.cell.map_or("the study", RouteCell::as_str)
            ))
        }
    }
}

/// Run the closing steps the shared dispatcher applies to a family result: the result carries
/// the identity of the lowered program it ran under, the disclosures the study owes, the
/// caller-attested validator names and the route's refutation stamps.
///
/// # Errors
///
/// `route_not_lowered` when `plan` carries no lowered program.
pub(crate) fn finish_family_result(
    study: &Study,
    input: &crate::analysis::builder::DataInput,
    plan: &PhysicalExecutionPlan,
    result: &mut crate::result::StudyResult,
) -> Result<(), CausalError> {
    let lowered = plan.require_lowered(expected_cell(study))?.identity();
    super::dispatch::push_gaussian_likelihood_disclosure(result, &study.inference, input);
    super::dispatch::push_lag_zero_average_effect_disclosure(study, result);
    result.custom_validator_names = study
        .custom_validators
        .iter()
        .map(|validator| Arc::<str>::from(validator.name()))
        .collect();
    result.lowered_identity = Some(lowered);
    crate::analysis::target_validation::stamp_route_refutations(result);
    Ok(())
}

/// The association route's retained operation. Its click runs the association executor
/// itself, under the program it was sealed to.
#[derive(Clone, Debug)]
pub(crate) struct CheckedAssociationOperation {
    route: CheckedRouteOperation,
}

impl CheckedAssociationOperation {
    /// Seal `study` to the program `physical` carries for `cell`.
    ///
    /// # Errors
    ///
    /// `route_not_lowered`, as [`CheckedRouteOperation::checked`].
    pub(crate) fn checked(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
        cell: RouteCell,
    ) -> Result<Self, CausalError> {
        CheckedRouteOperation::checked(study, physical, cell).map(|route| Self { route })
    }

    /// Require that `physical` still carries the program this operation retained.
    ///
    /// # Errors
    ///
    /// `route_not_lowered`, as [`CheckedRouteOperation::require_program`].
    pub(crate) fn require_program(
        &self,
        physical: &PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.route.require_program(physical)
    }
}

/// The structure-discovery route's retained operation. Its click runs the search itself,
/// under the program it was sealed to.
#[derive(Clone, Debug)]
pub(crate) struct CheckedDiscoveryOperation {
    route: CheckedRouteOperation,
}

impl CheckedDiscoveryOperation {
    /// Seal `study` to the program `physical` carries for `cell`.
    ///
    /// # Errors
    ///
    /// `route_not_lowered`, as [`CheckedRouteOperation::checked`].
    pub(crate) fn checked(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
        cell: RouteCell,
    ) -> Result<Self, CausalError> {
        CheckedRouteOperation::checked(study, physical, cell).map(|route| Self { route })
    }

    /// Require that `physical` still carries the program this operation retained.
    ///
    /// # Errors
    ///
    /// `route_not_lowered`, as [`CheckedRouteOperation::require_program`].
    pub(crate) fn require_program(
        &self,
        physical: &PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.route.require_program(physical)
    }
}

/// The immutable scientific target a concrete family operation is sealed to.
///
/// The canonical graph identity covers posterior atom structures, weights and
/// temporal metadata as well as single accepted graphs; data refresh is excluded.
#[derive(Clone, Debug)]
pub(crate) struct CheckedScientificRoute {
    route: CheckedRouteOperation,
    query: Arc<[u8]>,
    graph: antecedent_io::GraphIdentityWire,
}

impl CheckedScientificRoute {
    pub(crate) fn checked(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
        cell: Option<RouteCell>,
    ) -> Result<Self, CausalError> {
        let route = match cell {
            Some(cell) => CheckedRouteOperation::checked(study, physical, cell)?,
            None => CheckedRouteOperation::lowered(study, physical)?,
        };
        Ok(Self {
            route,
            query: crate::analysis::contract_identity::canonical_query_bytes(&study.query)?,
            graph: crate::analysis::contract::graph_identity(study)?,
        })
    }

    pub(crate) fn require_program(&self, plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        self.route.require_program(plan)
    }

    pub(crate) fn require(
        &self,
        study: &Study,
        plan: &PhysicalExecutionPlan,
    ) -> Result<(), CausalError> {
        self.require_program(plan)?;
        if self.query != crate::analysis::contract_identity::canonical_query_bytes(&study.query)?
            || self.graph != crate::analysis::contract::graph_identity(study)?
        {
            return Err(crate::compile_reason!(
                "route_not_lowered",
                "the retained scientific operation requires its frozen query and graph identity"
            ));
        }
        Ok(())
    }
}
