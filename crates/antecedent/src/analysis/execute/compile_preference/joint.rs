//! The joint law of separately estimated outcomes, declared, checked and lowered at compile.
//!
//! When a decision across several studies needs one aligned law of their outcomes, the
//! propagation chain draws it from a family table matched to each study's posterior moments:
//! one draw source per coordinate of each study's response, each carrying its study's row
//! factors (see [`super::evidence::row_factors`]), and one derived stage per source whose
//! outputs are the joint outputs the decision reads. Everything about that chain except the
//! numbers is known before anything is fitted: its sources, stages, factors, alignment,
//! reading, rows, seed and the declared dependence (each study's covariance kept within it,
//! plus any declared cross-study block). The compile builds that declaration, runs the chain
//! rule on it and compiles the chain with its family table deferred, so the plan states the
//! chain's inputs, dependence, joint outputs and lowered program, and the rule's refusals and
//! the table's work budget arise at compile. Execution binds each study's posterior moments to
//! the deferred table ([`crate::analysis::propagation_chain::PropagationChainPlan::execute_bound`]).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainDecl, ChainReading, ChainRefusal, ChainSourceDecl, ChainStageDecl, FactorDecl, LawMeaning,
    OperatorClass, Representation, StageKind, TableConstruction, check_propagation_chain,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainStageWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

use super::evidence::{CrossStudyCovariance, StudyOverlaps, row_factors};
use super::outcomes::OutcomeDependence;
use super::{BoundStudy, PreferenceDemand, PreferenceRouteError, STAGE_COMPILE};
use crate::analysis::propagation_chain::{
    ChainPlan, DeferredCrossBlock, FamilyEstimateDecl, PropagationChainPlan,
    compile_deferred_propagation_chain, deferred_family_table,
};
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
/// chain rule and compiled with its family table deferred.
#[derive(Clone, Debug)]
pub struct PreferenceJointLaw {
    fact: PreferenceJointLawFact,
    chain: PropagationChainPlan,
}

impl PartialEq for PreferenceJointLaw {
    fn eq(&self, other: &Self) -> bool {
        self.fact == other.fact && self.chain.identity() == other.chain.identity()
    }
}

impl PreferenceJointLaw {
    /// What the plan states: the chain's inputs, the declared dependence, its joint outputs
    /// and the identity of its compiled program.
    #[must_use]
    pub const fn fact(&self) -> &PreferenceJointLawFact {
        &self.fact
    }

    /// What the chain rule decided for every stage of the declaration.
    #[must_use]
    pub const fn plan(&self) -> &ChainPlan {
        self.chain.plan()
    }

    /// The chain compiled before anything is fitted, its family table bound at execution.
    #[must_use]
    pub const fn program(&self) -> &PropagationChainPlan {
        &self.chain
    }
}

fn chain_refusal(refusal: &ChainRefusal) -> PreferenceRouteError {
    ExternalRefusal::new(refusal.code(), STAGE_COMPILE, refusal.detail())
        .with_supplied(refusal.message_with_next_attempt())
        .into()
}

/// The chain declaration of `studies` with `alignment` as its draw table: one draw source
/// per coordinate of each study, read as a posterior with its study's row factors, and one
/// derived stage per source.
fn declaration(
    studies: &[BoundStudy],
    factors: &[Vec<String>],
    alignment: &str,
) -> (ChainDecl, Vec<PreferenceChainInputFact>) {
    let mut sources = Vec::new();
    let mut stages = Vec::new();
    let mut inputs = Vec::new();
    for (study, bound) in studies.iter().enumerate() {
        let own = factors.get(study).cloned().unwrap_or_default();
        let (basis, namespace) = match &bound.units {
            Some(units) => ("unit_key", Some(Arc::clone(&units.namespace))),
            None => ("table_content", None),
        };
        for (column, quantity) in bound.coordinates.iter().enumerate() {
            let source = source_id(study, column);
            let stage = stage_id(study, column);
            sources.push(ChainSourceDecl {
                id: source.clone(),
                representation: Representation::Draws,
                meaning: Some(LawMeaning::Posterior),
                alignment: Some(alignment.to_owned()),
                complete_disclosure: true,
                factors: own
                    .iter()
                    .map(|id| FactorDecl { kind: SharedFactorKind::Row, id: id.clone() })
                    .collect(),
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
                factor: Arc::from(own.first().map_or("", String::as_str)),
                factors: Arc::from(own.iter().map(|id| Arc::from(id.as_str())).collect::<Vec<_>>()),
                basis: Arc::from(basis),
                namespace: namespace.clone(),
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

/// The chain over `studies` with its family table deferred: one estimate per coordinate,
/// grouped by study, each study's covariance bound at execution and the declared cross-study
/// blocks between groups; one affine stage per source; the coordinates as joint outputs.
fn deferred_chain(
    studies: &[BoundStudy],
    factors: &[Vec<String>],
    (draws, seed, cross): (usize, u64, &[CrossStudyCovariance]),
    alignment: &str,
    ctx: &ExecutionContext,
) -> Result<PropagationChainPlan, PreferenceRouteError> {
    let mut estimates = Vec::new();
    let mut groups = Vec::with_capacity(studies.len());
    let mut outputs = Vec::new();
    for (study, bound) in studies.iter().enumerate() {
        let mut group = Vec::with_capacity(bound.coordinates.len());
        for (column, quantity) in bound.coordinates.iter().enumerate() {
            let mut estimate =
                FamilyEstimateDecl::new(&source_id(study, column), LawMeaning::Posterior);
            for factor in factors.get(study).into_iter().flatten() {
                estimate = estimate.with_factor(SharedFactorKind::Row, factor);
            }
            group.push(estimates.len());
            estimates.push(estimate);
            outputs.push(ChainJointOutputWire {
                stage: stage_id(study, column),
                quantity: ScientificQuantityWire::from(quantity),
            });
        }
        groups.push(group);
    }
    let blocks: Vec<DeferredCrossBlock> = cross
        .iter()
        .map(|block| DeferredCrossBlock {
            first: block.first,
            second: block.second,
            covariance: block.covariance.clone(),
        })
        .collect();
    let table = deferred_family_table(alignment, &estimates, &groups, &blocks, draws, seed)
        .map_err(|refusal| chain_refusal(&refusal))?;
    let stages = outputs
        .iter()
        .map(|output| {
            ChainStageWire::new(
                &output.stage,
                StageKind::DerivedQuantity,
                &[format!("{}_source", output.stage).as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        })
        .collect();
    let Some((output, quantity)) =
        outputs.first().map(|first| (first.stage.clone(), first.quantity.clone()))
    else {
        return Err(super::outcomes::unpaired("no coordinate to join"));
    };
    let mut draft = table.draft(ChainReading::Posterior, &output, stages);
    draft.record = draft.record.with_output_quantity(quantity).with_joint_outputs(outputs);
    compile_deferred_propagation_chain(draft, ctx).map_err(|refusal| chain_refusal(&refusal))
}

/// Declare, check and compile the chain that joins the studies' laws, when the demand needs
/// one.
///
/// `None` for a single study, or when means suffice. The declaration's digest binds every
/// input, the dependence, the declared cross-study blocks, the rows and the seed and, for
/// studies whose overlap was judged by unit key, the basis, namespace and factors of every
/// input; it and the compiled chain's program identity enter the identity of every study's
/// decision program.
///
/// # Errors
/// The chain rule's refusal of the declaration, and the deferred table's refusals of its
/// shape, work budget and declared blocks, at compile.
pub(super) fn declare(
    studies: &[BoundStudy],
    demand: &PreferenceDemand,
    (dependence, cross): (OutcomeDependence, &[CrossStudyCovariance]),
    overlaps: Option<&StudyOverlaps>,
    alignment: &str,
    ctx: &ExecutionContext,
) -> Result<Option<PreferenceJointLaw>, PreferenceRouteError> {
    if studies.len() < 2 || demand.representation == SourceRepresentation::Mean {
        return Ok(None);
    }
    let (name, draws, seed) = match dependence {
        OutcomeDependence::DistinctEvidence { draws, seed } => ("distinct_evidence", draws, seed),
        OutcomeDependence::SharedEvidence { draws, seed } => ("shared_evidence", draws, seed),
        OutcomeDependence::Undeclared => return Ok(None),
    };
    // The declared cross-study blocks are a set keyed by their pair of studies (the deferred
    // table refuses a repeated pair): taken in pair order, so the order the caller listed them
    // in reaches neither the declaration's digest, the plan's facts nor the chain.
    let mut cross = cross.to_vec();
    cross.sort_by_key(|block| (block.first, block.second));
    let cross = cross.as_slice();
    let none = StudyOverlaps::default();
    let factors = row_factors(studies, overlaps.unwrap_or(&none));
    let (decl, inputs) = declaration(studies, &factors, alignment);
    let plan = check_propagation_chain(&decl).map_err(|refusal| chain_refusal(&refusal))?;
    let chain = deferred_chain(studies, &factors, (draws, seed, cross), alignment, ctx)?;
    let mut writer = CanonicalWriter::new(DECLARATION_TAG);
    writer.text(alignment).text(name).word(draws as u64).word(seed).word(inputs.len() as u64);
    for input in &inputs {
        writer
            .text(&input.source)
            .text(&input.stage)
            .word(input.study as u64)
            .text(&input.quantity)
            .text(&input.factor);
        // An input whose overlap was judged by unit key states its basis, namespace and
        // factors after a presence word; an unkeyed input is named by its first factor alone
        // and writes the absent word.
        match &input.namespace {
            Some(namespace) => {
                writer.word(1).text(&input.basis).text(namespace).texts(&input.factors);
            }
            None => {
                writer.word(0);
            }
        }
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
    let pairs: Vec<(usize, usize, Arc<str>)> = overlaps.map_or_else(Vec::new, |overlaps| {
        overlaps
            .bases
            .iter()
            .map(|&(first, second, basis)| (first, second, Arc::from(basis)))
            .collect()
    });
    let fact = PreferenceJointLawFact {
        inputs: Arc::from(inputs),
        pairs: Arc::from(pairs),
        dependence: Arc::from(name),
        cross_blocks: Arc::from(
            cross.iter().map(|block| (block.first, block.second)).collect::<Vec<_>>(),
        ),
        draws,
        seed,
        reading: Arc::from(plan.reading.as_str()),
        label: Arc::from(plan.label),
        declaration: Arc::from(writer.finish_hex()),
        program: chain.identity(),
    };
    Ok(Some(PreferenceJointLaw { fact, chain }))
}
