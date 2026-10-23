//! Layer four of a route lowering: what compile started from, and how the
//! program's evidence combines.
//!
//! A decision or value-of-information goal states the carrier it demands. The
//! lowering checks that demand against the route's statistical realization and
//! refuses, naming the next attempt, when the realization cannot publish it. A
//! planning goal carries the declared request of a future trial analysed as the
//! route; its identity enters layer four.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{LoweringContext, RouteCell};
use crate::analysis::execute::require_carrier;
use crate::error::CausalError;
use crate::planner::lowered::{
    Composition, DecisionProgram, EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef,
    PlanningProgram,
};
use crate::planner::statistical::StatisticalProgram;

/// What a compile was asked to produce.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CompileGoal {
    /// The route's estimate, with no decision attached.
    Estimate,
    /// A terminal decision over the route's answer.
    Decision(DecisionProgram),
    /// A preposterior value-of-information request. The program's
    /// [`DecisionProgram::information_value`] is always present.
    Evsi(DecisionProgram),
    /// An experiment-planning request for a future trial analysed as this route.
    Planning(PlanningProgram),
}

impl CompileGoal {
    /// The decision the goal carries, when it carries one.
    pub(crate) const fn decision(&self) -> Option<&DecisionProgram> {
        match self {
            Self::Estimate | Self::Planning(_) => None,
            Self::Decision(program) | Self::Evsi(program) => Some(program),
        }
    }

    /// The planning request the goal carries, when it carries one.
    pub(crate) const fn planning(&self) -> Option<&PlanningProgram> {
        match self {
            Self::Planning(program) => Some(program),
            Self::Estimate | Self::Decision(_) | Self::Evsi(_) => None,
        }
    }
}

/// Evidence composition of the study, and the decision when compile started from one.
///
/// # Errors
///
/// A decision goal demands a carrier `statistical` cannot publish. The refusal
/// is a typed [`crate::analysis::DecisionCarrierError`] naming the next attempt.
pub(super) fn evidence_program(
    cx: &LoweringContext<'_>,
    source_digest: Option<[u8; 32]>,
    statistical: &StatisticalProgram,
) -> Result<EvidenceDecisionProgram, CausalError> {
    let composition = match cx.cell {
        RouteCell::GraphPosterior(_) => Composition::AtomMixture {
            atoms: cx.study.graph_posterior.as_ref().map_or(0, |posterior| posterior.n_graphs),
        },
        RouteCell::TransportMixed => Composition::Transported,
        _ => Composition::Single,
    };
    let decision = cx.goal.decision().cloned();
    if let Some(program) = &decision {
        require_carrier(cx.cell, statistical.realization(), program)
            .map_err(CausalError::rejected)?;
    }
    Ok(EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: source_digest,
        }]),
        composition,
        decision,
        planning: cx.goal.planning().cloned(),
    })
}
