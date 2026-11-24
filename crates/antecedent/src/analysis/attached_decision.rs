//! The parent program of a decision attached to compiled analyses.
//!
//! A decision attached to a study, to several studies joined by a chain, to the structures of a
//! graph posterior or to a compiled design reads the quantities those programs estimate. Each of
//! them is a child program: it keeps its own route, its own unit and its own support gate. The
//! decision is a parent program composed over them (ADR 0028 amends ADR 0022: a program may be
//! composed over child programs). It is a six-layer program of its own family, so:
//!
//! * its route is bound to the decision's own support unit in `parity/compiled_units.toml`, and
//!   [`PlanView::unit_support`] of its plan view reports that unit;
//! * its identity is the canonical encoding ([`crate::analysis::supplied_identity`]) of the
//!   children's program identities, in the order the children were declared, and of the
//!   decision's own declaration, under a versioned tag per family; layer four lists one source
//!   per child, carrying the child's program identity;
//! * an execution passes its gate ([`ParentDecisionProgram::gate`]) before any child runs, and
//!   every child then passes its own. A release build refuses a parent whose unit is not
//!   supported; a development build records what a release build refuses
//!   ([`antecedent_core::WouldRefuse`] in the context's shared log) and runs.
//!
//! No support decision enters the identity: the parent of a decision is the same program
//! whether its unit is supported or not.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{ExecutionContext, NotSupported, SemanticDigest};

use crate::analysis::program_handle::CompiledProgramHandle;
use crate::analysis::supplied_identity::Declaration;
use crate::analysis::supplied_terminal::{Reading, TerminalSpec, lower_over};
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    LoweredProgram, ProgramCell,
};
use crate::planner::{PlanStage, PlanView};
use crate::support::unit::{UnitDecision, require_decided};

/// What a parent whose route no unit is bound to is missing, in a refusal.
const NO_UNIT: &str = "support_unit";

/// What structure a parent program's inputs carry: compiled child programs.
const GRAPH_CLASS: &str = "attached_decision_programs";

/// The layers of a parent decision program, and what its identity covers.
pub(crate) struct ParentSpec<'a> {
    /// The parent family's program cell; its route is bound to the decision's unit.
    pub(crate) cell: ProgramCell,
    /// Versioned plan id of the family.
    pub(crate) plan_id: &'static str,
    /// Versioned tag of the parent's declaration.
    pub(crate) tag: &'static str,
    /// The child programs in the order they were declared: `None` for a declared child that
    /// compiled no program (a structure under which the decision's quantities are not
    /// identified).
    pub(crate) children: &'a [Option<SemanticDigest>],
    /// Identity of the decision's own declaration (contract, bindings and whatever else the
    /// decision reads besides its children), written by the caller with
    /// [`Declaration`].
    pub(crate) decision: [u8; 32],
    /// The kernel evaluator the parent runs, by name.
    pub(crate) evaluator: &'static str,
    /// How the children combine into the decision's source.
    pub(crate) composition: Composition,
    /// Layer four's decision.
    pub(crate) program: DecisionProgram,
    /// The kind of claim the decision is.
    pub(crate) claim: Option<ClaimKind>,
}

/// Identity of a parent's declaration: the children's program identities in declared order,
/// then the decision's own declaration, under `tag`. Reordering the children, dropping one or
/// changing the decision changes it.
pub(crate) fn parent_identity(
    tag: &str,
    children: &[Option<SemanticDigest>],
    decision: [u8; 32],
) -> [u8; 32] {
    Declaration::new(tag)
        .field("children", children)
        .field("decision", &SemanticDigest::from_bytes(decision))
        .finish()
}

/// How the parent reads the carrier its children publish.
const fn reading(carrier: CarrierDemand) -> Reading {
    match carrier {
        CarrierDemand::Mean => Reading::Exact,
        CarrierDemand::ParameterPosterior => Reading::Posterior,
        CarrierDemand::OutcomeLaw | CarrierDemand::PredictiveLaw => Reading::Predictive,
        CarrierDemand::IdentifiedSet => Reading::Envelope,
        CarrierDemand::DecisionDraws => Reading::MonteCarlo,
    }
}

/// Seal `spec` into the parent program.
///
/// # Errors
/// A physical operator that cannot be declared.
pub(crate) fn lower_parent(spec: ParentSpec<'_>) -> Result<ParentDecisionProgram, CausalError> {
    let question = parent_identity(spec.tag, spec.children, spec.decision);
    let sources: Arc<[EvidenceSourceRef]> = spec
        .children
        .iter()
        .map(|child| EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: child.as_ref().map(|digest| *digest.as_bytes()),
        })
        .collect();
    let lowered = lower_over(
        TerminalSpec {
            cell: spec.cell,
            plan_id: spec.plan_id,
            functional: FunctionalKind::PolicyValue,
            graph_class: GRAPH_CLASS,
            question,
            evaluator: spec.evaluator,
            reading: reading(spec.program.carrier),
            role: EvidenceRole::ChainSource,
            composition: spec.composition,
            decision: Some(spec.program),
            claim: spec.claim,
        },
        sources,
    )?;
    Ok(ParentDecisionProgram {
        lowered,
        #[cfg(test)]
        decided: None,
    })
}

/// The parent program of a decision attached to compiled child programs.
#[derive(Clone, Debug)]
pub(crate) struct ParentDecisionProgram {
    lowered: Arc<LoweredProgram>,
    /// A unit decision a test reads in place of the route's own, so that a supported child
    /// can be run under a parent whose unit is not supported.
    #[cfg(test)]
    decided: Option<UnitDecision>,
}

impl CompiledProgramHandle for ParentDecisionProgram {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl ParentDecisionProgram {
    /// The plan view of the parent program: its route, its layers and, through
    /// [`PlanView::unit_support`], the support decision of the decision's own unit.
    pub(crate) fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Identity of the parent program.
    pub(crate) fn identity(&self) -> SemanticDigest {
        self.lowered.identity()
    }

    /// Pass the parent's support gate with the caller's context, before any child runs. The
    /// route names the entry point in a refusal.
    ///
    /// # Errors
    /// [`NotSupported`] in a release build when the decision's unit is not supported, or when
    /// no unit is bound to the parent's route.
    pub(crate) fn gate(&self, ctx: &ExecutionContext) -> Result<(), NotSupported> {
        let route = self.lowered.route();
        match self.decision() {
            Some(decided) => require_decided(ctx, &decided, route),
            None => antecedent_core::support_gate::require_supported(ctx, route, &[NO_UNIT]),
        }
    }

    /// The support decision of the decision's own unit, read from the parent's route; `None`
    /// while no unit is bound to it.
    fn decision(&self) -> Option<UnitDecision> {
        #[cfg(test)]
        if let Some(decided) = &self.decided {
            return Some(decided.clone());
        }
        self.support()
    }

    /// The same parent, gated by `decided` in place of its route's unit.
    #[cfg(test)]
    pub(crate) fn deciding(mut self, decided: UnitDecision) -> Self {
        self.decided = Some(decided);
        self
    }
}

#[cfg(test)]
#[path = "attached_decision_tests.rs"]
mod tests;
