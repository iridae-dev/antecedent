//! Propagation-chain lowering: every compiled chain carries a lowered program.
//!
//! A chain compiles through its own entry
//! (`compile_propagation_chain`), not through `Study::compile`, because it binds
//! no study: its sources are declared draws, intervals and points. It lowers to
//! the same six layers:
//!
//! 1. the chain's output as a typed functional keyed by the digest of the
//!    declared chain, so two chains lower to different programs;
//! 2. the dependence contract as the identification of overlap: which rows,
//!    clusters, models, priors, disturbances and periods each stage inherits and
//!    what every combining stage shares. Nothing is identified from
//!    observations;
//! 3. the statistical program from the law the output carries (a sampling law, a
//!    posterior, or an exact enclosure or point) and one model slot per stage;
//! 4. the evidence: one source per chain source, the stages in sequence, and the
//!    declared chain itself (the record identity the execution seals);
//! 5. one physical operator per stage, with the output bytes the table admits;
//! 6. the claim demand.
//!
//! Lowering computes no number and grants no licence: a chain value is a labelled
//! point or uncalibrated spread, and an interval of a composed quantity stays
//! closed without a measured coverage record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::propagation_chain::{
    ChainPlan, ChainRefusal, FIX_CHAIN_DECLARATION, LawMeaning, NodeRepresentation, StageKind,
};
use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, LikelihoodModel, OrdinaryBayesianLikelihood, PhysicalEffect,
    PhysicalOperator, Prior, ScoreFrequentist, ScoreModel,
};
use antecedent_io::propagation_chain_record::{
    PropagationChainRecordWire, propagation_chain_identity,
};

use crate::planner::lowered::{
    ChainLoweredParts, ChainProgram, ClaimDemand, Composition, DesignShape,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ScientificProblem,
    UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};

/// Why a chain's design shape is not a quoted adjustment design.
const CHAIN_DESIGN_SHAPE: &str = "a propagation chain combines declared sources; it fits no \
                                  adjustment design";

const QUERY_TAG: &str = "antecedent.propagation_chain.lowered_query";
const SOURCE_TAG: &str = "antecedent.propagation_chain.lowered_source";

fn refuse(reason: impl std::fmt::Display) -> ChainRefusal {
    ChainRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        "propagation_chain.lowering",
        format!("propagation chain lowering: {reason}"),
        None,
        FIX_CHAIN_DECLARATION,
    )
}

/// The identity a sealed record of `draft` carries: the declaration, the draws, the
/// future-study draws and the changes, with the label the rule yields and no results.
///
/// # Errors
///
/// A stored name that is not one of the format's names.
pub(crate) fn declaration_identity(
    draft: &PropagationChainRecordWire,
    plan: &ChainPlan,
) -> Result<String, ChainRefusal> {
    let mut declared = draft.clone();
    declared.label = plan.label.to_owned();
    propagation_chain_identity(&declared).map_err(|error| error.to_refusal())
}

fn uncertainty(plan: &ChainPlan) -> UncertaintyReading {
    match plan.stage(&plan.output) {
        Some(output) => match (output.representation, output.meaning) {
            (NodeRepresentation::Enclosure, _) => UncertaintyReading::Envelope,
            (NodeRepresentation::Point, _) => UncertaintyReading::None,
            (NodeRepresentation::Draws, Some(LawMeaning::Posterior)) => {
                UncertaintyReading::Posterior
            }
            (NodeRepresentation::Draws, _) => UncertaintyReading::Sampling,
            _ => UncertaintyReading::None,
        },
        None => UncertaintyReading::None,
    }
}

fn source_digest(source: &antecedent_io::propagation_chain_record::ChainSourceWire) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(SOURCE_TAG);
    writer.text(&source.id).text(&source.representation).word(source.values.len() as u64);
    for value in &source.values {
        writer.word(*value);
    }
    writer.finish()
}

fn statistical_program(
    draft: &PropagationChainRecordWire,
    plan: &ChainPlan,
    uncertainty: UncertaintyReading,
) -> Result<StatisticalProgram, ChainRefusal> {
    let name = format!("propagation_chain.{}", plan.reading.as_str());
    let realization = match uncertainty {
        UncertaintyReading::Sampling => {
            Realization::Frequentist(ScoreFrequentist::new(ScoreModel::new(name).map_err(refuse)?))
        }
        UncertaintyReading::Posterior => Realization::Bayesian(OrdinaryBayesianLikelihood::new(
            LikelihoodModel::new(name).map_err(refuse)?,
            Prior::new("propagation_chain.declared_posterior_sources".to_owned())
                .map_err(refuse)?,
        )),
        UncertaintyReading::Envelope | UncertaintyReading::None => {
            Realization::ExactPoint { evaluator: Arc::from(name) }
        }
    };
    let models: Vec<ModelSlot> = draft
        .stages
        .iter()
        .map(|stage| ModelSlot {
            role: if stage.kind == StageKind::PreposteriorUpdate.as_str() {
                ModelRole::Outcome
            } else {
                ModelRole::Mechanism
            },
            family: Arc::from(format!("propagation_chain.{}.{}", stage.kind, stage.id)),
            inputs: Arc::from(Vec::new()),
            output: None,
        })
        .collect();
    Ok(StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from(models),
        folds: None,
        resampling: None,
        posterior: None,
        design: DesignShape::not_design_based(CHAIN_DESIGN_SHAPE),
        operators: Arc::from(Vec::new()),
    })
}

fn operator(effect: PhysicalEffect, output: Option<u64>) -> Result<PhysicalOperator, ChainRefusal> {
    let mut builder = PhysicalOperator::declare(effect);
    if let Some(point) = effect.required_cancellation_point() {
        builder = builder.cancellation_point(point);
    }
    if let Some(bytes) = output {
        builder = builder.estimates(ByteEstimates::absent().with_output(bytes));
    }
    builder.build().map_err(refuse)
}

/// One read of the sources, then one operator per stage; the last carries the bytes
/// of the table a stage column occupies.
fn physical_layer(draft: &PropagationChainRecordWire) -> Result<PhysicalLayer, ChainRefusal> {
    let rows = draft.sources.iter().map(|s| s.values.len() as u64).max().unwrap_or(1);
    let nodes = (draft.sources.len() + draft.stages.len()) as u64;
    let table_bytes = rows.saturating_mul(nodes).saturating_mul(8);
    let mut effects = vec![PhysicalEffect::Read];
    for stage in &draft.stages {
        effects.push(match StageKind::from_name(&stage.kind) {
            Some(StageKind::Transport) => PhysicalEffect::Transport,
            Some(StageKind::PreposteriorUpdate) => PhysicalEffect::Prior,
            _ => PhysicalEffect::Condition,
        });
    }
    let last = effects.len() - 1;
    let operators = effects
        .into_iter()
        .enumerate()
        .map(|(index, effect)| operator(effect, (index == last).then_some(table_bytes)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PhysicalLayer {
        operators: Arc::from(operators),
        design: DesignShape::not_design_based(CHAIN_DESIGN_SHAPE),
    })
}

/// Lower a checked chain.
///
/// `draft` is the declared chain with no results, `plan` what the chain rule
/// decided for it. Nothing is evaluated.
///
/// # Errors
///
/// A name the format does not know, a model name the statistical vocabulary
/// refuses, or a physical operator that does not declare its cancellation point.
pub(crate) fn lower_chain(
    draft: PropagationChainRecordWire,
    plan: ChainPlan,
) -> Result<LoweredProgram, ChainRefusal> {
    let identity = declaration_identity(&draft, &plan)?;
    let query = CanonicalWriter::digest_bytes(QUERY_TAG, identity.as_bytes());
    let uncertainty = uncertainty(&plan);
    let reading = plan.reading.as_str();
    let scientific = ScientificProblem {
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            query,
        },
        plan_id: Arc::from(format!("propagation_chain.{reading}")),
        data: DataClassification::Tabular,
        graph_class: "none",
        query_variables: Arc::from(Vec::new()),
        source_digest: None,
        support_coordinate: None,
        licensed: false,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = statistical_program(&draft, &plan, uncertainty)?;
    let physical = physical_layer(&draft)?;
    let sources: Vec<EvidenceSourceRef> = draft
        .sources
        .iter()
        .map(|source| EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(source_digest(source)),
        })
        .collect();
    let stages = draft.stages.len();
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(sources),
        composition: Composition::Chained { stages },
        decision: None,
        planning: None,
        chain: Some(ChainProgram::new(draft, plan, identity)),
    };
    let claim = ClaimDemand {
        uncertainty,
        kind: Some(match uncertainty {
            UncertaintyReading::Envelope => ClaimKind::Bounds,
            _ => ClaimKind::Point,
        }),
    };
    Ok(ChainLoweredParts {
        reading,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
