//! The joint law of separately estimated outcomes, declared and checked at compile.
//!
//! When a decision across several studies needs one aligned law of their outcomes, the
//! propagation chain draws it from a family table matched to each study's posterior moments:
//! one draw source per coordinate of each study's response, each carrying the content
//! identity of its study's table as its row factor, and one derived stage per source whose outputs are the joint
//! outputs the decision reads. Everything about that chain except the numbers is known before
//! anything is fitted: its sources, stages, factors, alignment, reading and the declared
//! dependence (each study's covariance kept within it, plus any declared cross-study block).
//! The compile builds that declaration and runs the chain rule on it, so the plan states the
//! chain's inputs, dependence and joint outputs and the rule's refusals arise at compile.
//!
//! The chain's lowered program itself is lowered at execution: its sources are the studies'
//! draws, and a chain's declaration identity binds the draws it reads.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainDecl, ChainPlan, ChainReading, ChainSourceDecl, ChainStageDecl, FactorDecl, LawMeaning,
    OperatorClass, Representation, StageKind, TableConstruction, check_propagation_chain,
};
use antecedent_core::CanonicalWriter;
use antecedent_design::decision_contract::SourceRepresentation;

use super::evidence::CrossStudyCovariance;
use super::outcomes::OutcomeDependence;
use super::{BoundStudy, PreferenceDemand, PreferenceRouteError, STAGE_COMPILE};
use crate::planner::lowered::{PreferenceChainInputFact, PreferenceJointLawFact};

const DECLARATION_TAG: &str = "antecedent.preference_route.joint_law_declaration.v1";

/// Identity of the chain source of column `column` of study `study`.
pub(super) fn source_id(study: usize, column: usize) -> String {
    format!("{}_source", stage_id(study, column))
}

/// Identity of the stage whose output is column `column` of study `study`.
pub(super) fn stage_id(study: usize, column: usize) -> String {
    format!("study_{study}_column_{column}")
}

/// The chain declared for a decision across separately estimated outcomes, checked by the
/// chain rule at compile.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceJointLaw {
    fact: PreferenceJointLawFact,
    plan: ChainPlan,
}

impl PreferenceJointLaw {
    /// What the plan states: the chain's inputs, the declared dependence and its joint
    /// outputs.
    #[must_use]
    pub const fn fact(&self) -> &PreferenceJointLawFact {
        &self.fact
    }

    /// What the chain rule decided for every stage of the declaration.
    #[must_use]
    pub const fn plan(&self) -> &ChainPlan {
        &self.plan
    }
}

fn chain_refusal(
    refusal: &antecedent_contract::propagation_chain::ChainRefusal,
) -> PreferenceRouteError {
    ExternalRefusal::new(refusal.code(), STAGE_COMPILE, refusal.detail())
        .with_supplied(refusal.message_with_next_attempt())
        .into()
}

/// The chain declaration of `studies` with `alignment` as its draw table: one draw source
/// per coordinate of each study, read as a posterior with the content identity of the
/// study's table as its row factor, and one derived stage per source.
fn declaration(
    studies: &[BoundStudy],
    alignment: &str,
) -> (ChainDecl, Vec<PreferenceChainInputFact>) {
    let mut sources = Vec::new();
    let mut stages = Vec::new();
    let mut inputs = Vec::new();
    for (study, bound) in studies.iter().enumerate() {
        for (column, quantity) in bound.coordinates.iter().enumerate() {
            let source = source_id(study, column);
            let stage = stage_id(study, column);
            sources.push(ChainSourceDecl {
                id: source.clone(),
                representation: Representation::Draws,
                meaning: Some(LawMeaning::Posterior),
                alignment: Some(alignment.to_owned()),
                complete_disclosure: true,
                factors: vec![FactorDecl {
                    kind: SharedFactorKind::Row,
                    id: bound.observations.clone(),
                }],
            });
            stages.push(ChainStageDecl {
                id: stage.clone(),
                kind: StageKind::DerivedQuantity,
                class: OperatorClass::Affine,
                inputs: vec![source.clone()],
                own_factors: Vec::new(),
            });
            inputs.push(PreferenceChainInputFact {
                source: Arc::from(source),
                stage: Arc::from(stage),
                study,
                quantity: Arc::from(quantity.canonical_identity()),
                factor: Arc::from(bound.observations.as_str()),
            });
        }
    }
    let output = stages.first().map(|stage| stage.id.clone()).unwrap_or_default();
    let decl = ChainDecl {
        construction: TableConstruction::JointDraws,
        sources,
        stages,
        output,
        reading: ChainReading::Posterior,
    };
    (decl, inputs)
}

/// Declare and check the chain that joins the studies' laws, when the demand needs one.
///
/// `None` for a single study, or when means suffice. The declaration's digest binds every
/// input, the dependence, the declared cross-study blocks, the rows and the seed, and enters
/// the identity of every study's decision program.
///
/// # Errors
/// The chain rule's refusal of the declaration, at compile.
pub(super) fn declare(
    studies: &[BoundStudy],
    demand: &PreferenceDemand,
    dependence: OutcomeDependence,
    cross: &[CrossStudyCovariance],
    alignment: &str,
) -> Result<Option<PreferenceJointLaw>, PreferenceRouteError> {
    if studies.len() < 2 || demand.representation == SourceRepresentation::Mean {
        return Ok(None);
    }
    let (name, draws, seed) = match dependence {
        OutcomeDependence::DistinctEvidence { draws, seed } => ("distinct_evidence", draws, seed),
        OutcomeDependence::SharedEvidence { draws, seed } => ("shared_evidence", draws, seed),
        OutcomeDependence::Undeclared => return Ok(None),
    };
    let (decl, inputs) = declaration(studies, alignment);
    let plan = check_propagation_chain(&decl).map_err(|refusal| chain_refusal(&refusal))?;
    let mut writer = CanonicalWriter::new(DECLARATION_TAG);
    writer.text(alignment).text(name).word(draws as u64).word(seed).word(inputs.len() as u64);
    for input in &inputs {
        writer
            .text(&input.source)
            .text(&input.stage)
            .word(input.study as u64)
            .text(&input.quantity)
            .text(&input.factor);
    }
    writer.word(cross.len() as u64);
    for block in cross {
        writer
            .word(block.first as u64)
            .word(block.second as u64)
            .word(block.covariance.len() as u64);
        for term in &block.covariance {
            writer.real(*term);
        }
    }
    let fact = PreferenceJointLawFact {
        inputs: Arc::from(inputs),
        dependence: Arc::from(name),
        cross_blocks: Arc::from(
            cross.iter().map(|block| (block.first, block.second)).collect::<Vec<_>>(),
        ),
        draws,
        seed,
        reading: Arc::from(plan.reading.as_str()),
        label: Arc::from(plan.label),
        declaration: Arc::from(writer.finish_hex()),
    };
    Ok(Some(PreferenceJointLaw { fact, plan }))
}
