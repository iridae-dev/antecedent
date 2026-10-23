//! Layer four of a route lowering: what compile started from, and how the
//! program's evidence combines.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{LoweringContext, RouteCell};
use crate::planner::lowered::{
    Composition, EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef,
};

/// What a compile was asked to produce.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompileGoal {
    /// The route's estimate, with no decision attached.
    Estimate,
}

/// Evidence composition of the study, and the decision when compile started from one.
pub(super) fn evidence_program(
    cx: &LoweringContext<'_>,
    source_digest: Option<[u8; 32]>,
) -> EvidenceDecisionProgram {
    let composition = match cx.cell {
        RouteCell::GraphPosterior(_) => Composition::AtomMixture {
            atoms: cx.study.graph_posterior.as_ref().map_or(0, |posterior| posterior.n_graphs),
        },
        RouteCell::TransportMixed => Composition::Transported,
        _ => Composition::Single,
    };
    let decision = match cx.goal {
        CompileGoal::Estimate => None,
    };
    EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: source_digest,
        }]),
        composition,
        decision,
    }
}
