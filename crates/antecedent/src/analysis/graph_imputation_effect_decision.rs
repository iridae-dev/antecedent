//! Graph-selection, missing-row imputation, child effect fitting and utility.
//!
//! The graph-imputation compiler owns the whole upstream schedule and its
//! native replay. This outer compiler binds that retained program and an exact
//! binary average-effect decision before execution. Its posterior law is a
//! model-conditional causal-functional mixture, never a predictive outcome
//! law or an earned sampling confidence distribution.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_io::IoError;
use antecedent_io::measured_inference::graph_candidate::{GraphPairedWorldReading, replay_paired_worlds};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{
    CompiledGraphImputation, GraphImputationError, GraphImputationExpectation,
    GraphImputationPoint, consume as consume_graph_imputation,
};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision,
    consume_chain_decision, compile_action_world_chain_decision,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification,
    PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "graph_imputation_effect_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Refusal of an upstream mixture, downstream contract or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationEffectDecisionError {
    /// A coordinate, carrier or artifact identity is invalid.
    #[error("graph imputation effect decision: {0}")]
    Invalid(&'static str),
    /// The graph-imputation compiler or native consumer refused.
    #[error("graph imputation effect producer: {0}")]
    Producer(#[from] GraphImputationError),
    /// Decision contract refused.
    #[error("graph imputation effect contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Chain or utility compiler refused.
    #[error("graph imputation effect chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or decoding refused.
    #[error("graph imputation effect artifact: {0}")]
    Io(#[from] IoError),
}

/// The retained graph-imputation producer and the exact effect decision.
pub struct CompiledGraphImputationEffectDecision {
    producer: CompiledGraphImputation,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Whole upstream effect mixture with downstream utility on its aligned draws.
pub struct GraphImputationEffectDecisionResult {
    /// Native graph-imputation fit and effect draw receipt.
    pub producer: GraphImputationPoint,
    /// Chain and multi-action utility over the effect posterior.
    pub decision: ChainDecisionResult,
    /// Full outer compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// No whole decision reliability record has been earned.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationEffectDecisionWire {
    version: u32,
    required_feature: String,
    producer_source: Vec<u8>,
    producer_program: [u8; 32],
    producer_data: [u8; 32],
    producer_receipt: [u8; 32],
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
    calibration: String,
}

fn check_decision(
    quantity: &ScientificQuantity, decision: &DecisionContract,
) -> Result<(), GraphImputationEffectDecisionError> {
    decision.validate()?;
    if quantity.population_id != decision.target_population
        || quantity.horizon != decision.horizon
        || decision.actions.is_empty()
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1
                || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(GraphImputationEffectDecisionError::Invalid("every action must read the exact graph-mixture effect coordinate"));
    }
    Ok(())
}

fn lower_outer(
    producer: SemanticDigest, source: SemanticDigest,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
) -> Result<Arc<LoweredProgram>, GraphImputationEffectDecisionError> {
    check_decision(quantity, decision)?;
    let quantity_bytes = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let decision_identity = decision.identity()?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation_effect_decision.question.v1");
    writer.bytes(producer.as_bytes()).bytes(source.as_bytes())
        .bytes(&quantity_bytes).text(&decision_identity);
    let question = writer.finish();
    let design = DesignShape::quoted(1, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue, outcome: None, query: question,
        },
        plan_id: Arc::from("graph_imputation_effect_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "bounded_graph_imputation_effect_mixture",
        query_variables: Arc::from([]), source_digest: Some(*source.as_bytes()),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource, digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(), constraints: decision.constraints.len(),
            chance_threshold: decision.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(decision_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("native_graph_imputation_effect_draws_v1")),
        }),
        planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(quantity_bytes.len() as u64))
                .build().map_err(|_| GraphImputationEffectDecisionError::Invalid("invalid producer receipt read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build().map_err(|_| GraphImputationEffectDecisionError::Invalid("invalid graph mixture operator"))?,
        ]), design,
    };
    Ok(Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::GraphImputationEffectDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Posterior,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish()))
}

/// Compose an already compiled graph-imputation producer with an effect
/// posterior decision. The upstream compiler's exact program identity is a
/// dependency of this outer program; no imputation or fitting runs here.
///
/// # Errors
/// An effect coordinate that differs from the producer, a decision mismatch,
/// or lowering refusal.
pub fn compile_graph_imputation_effect_decision(
    producer: CompiledGraphImputation, quantity: ScientificQuantity,
    decision: DecisionContract,
) -> Result<CompiledGraphImputationEffectDecision, GraphImputationEffectDecisionError> {
    if !producer.matches_effect_quantity(&quantity)? {
        return Err(GraphImputationEffectDecisionError::Invalid("quantity differs from the compiled binary graph effect"));
    }
    let lowered = lower_outer(
        producer.lowered().identity(), producer.source_identity(),
        &quantity, &decision,
    )?;
    Ok(CompiledGraphImputationEffectDecision { producer, quantity, decision, lowered })
}

fn effect_chain(
    point: &GraphImputationPoint, quantity: &ScientificQuantity,
) -> PropagationChainRecordWire {
    let receipt = SemanticDigest::from_bytes(point.receipt).to_hex();
    let alignment = format!("graph_imputation:{receipt}");
    let source = ChainSourceWire::draws(
        "graph.effect.posterior", LawMeaning::Posterior,
        &alignment, point.effect_draws(),
    ).with_factor(SharedFactorKind::Model, &receipt);
    let stage = ChainStageWire::new(
        "graph.effect", StageKind::DerivedQuantity, &["graph.effect.posterior"],
        ChainOpWire::affine(
            ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0),
        ),
    );
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws, ChainReading::Posterior,
        "graph.effect", vec![source], vec![stage],
    ).with_output_quantity(ScientificQuantityWire::from(quantity))
}

impl CompiledGraphImputationEffectDecision {
    /// The six-layer outer program that binds the complete upstream schedule.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained program without fitting or scoring.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Rerun graph selection and missing-row imputation through the producer,
    /// then compile utility over its retained effect posterior draws.
    ///
    /// # Errors
    /// An upstream child failure, mismatched effect or decision refusal.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<GraphImputationEffectDecisionResult, GraphImputationEffectDecisionError> {
        let point = self.producer.execute(ctx)?;
        point.effect_law(self.quantity.clone())?;
        let decision = compile_chain_decision(
            effect_chain(&point, &self.quantity), self.decision.clone(), ctx,
        )?.execute(ctx)?;
        Ok(GraphImputationEffectDecisionResult {
            producer: point, decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Export the source-bound native producer and chain decision after replay.
    ///
    /// # Errors
    /// Changed graph mixture, utility, identity or artifact encoding.
    pub fn export(
        &self, result: &GraphImputationEffectDecisionResult,
        artifact_id: &str, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, GraphImputationEffectDecisionError> {
        let replay = self.producer.replay(result.producer.receipt, ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || replay.effect_draws() != result.producer.effect_draws()
        {
            return Err(GraphImputationEffectDecisionError::Invalid("graph mixture differs on replay"));
        }
        let graph = self.producer.export(&result.producer, ctx)
            .map_err(GraphImputationError::from)?;
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&replay, &self.quantity), self.decision.clone(), ctx,
        )?;
        let live = chain.execute(ctx)?;
        if live.decision != result.decision.decision
            || live.chain.identity() != result.decision.chain.identity()
        {
            return Err(GraphImputationEffectDecisionError::Invalid("utility differs on replay"));
        }
        let expected = graph.expectation();
        let wire = GraphImputationEffectDecisionWire {
            version: 1, required_feature: FEATURE.into(),
            producer_source: graph.bytes().to_vec(),
            producer_program: expected.program,
            producer_data: expected.source,
            producer_receipt: expected.receipt,
            quantity: ScientificQuantityWire::from(&self.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&result.decision, artifact_id)?,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(GraphImputationEffectDecisionError::Invalid("combined artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Independently replay the original graph-imputation source and its child
/// fits, then recompile and compare the final effect utility. The outer
/// program identity must be retained separately from the artifact.
///
/// # Errors
/// A changed upstream schedule, model-conditional effect law, utility or
/// compiler identity refuses.
pub fn consume_graph_imputation_effect_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<GraphImputationEffectDecisionResult, GraphImputationEffectDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(GraphImputationEffectDecisionError::Invalid("combined artifact exceeds its byte bound"));
    }
    let wire: GraphImputationEffectDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.calibration != "not_measured"
    {
        return Err(GraphImputationEffectDecisionError::Invalid("combined feature or identity differs"));
    }
    let expectation = GraphImputationExpectation {
        program: wire.producer_program, source: wire.producer_data,
        receipt: wire.producer_receipt,
    };
    let point = consume_graph_imputation(&wire.producer_source, expectation, ctx)?;
    let quantity: ScientificQuantity = wire.quantity.try_into()
        .map_err(GraphImputationEffectDecisionError::Invalid)?;
    point.effect_law(quantity.clone())?;
    let contract = DecisionContractArtifact::from_bytes(
        &wire.decision_artifact, &wire.decision_identity,
    )?;
    let lowered = lower_outer(
        SemanticDigest::from_bytes(wire.producer_program),
        SemanticDigest::from_bytes(wire.producer_data), &quantity,
        contract.contract(),
    )?;
    if lowered.identity() != expected_program {
        return Err(GraphImputationEffectDecisionError::Invalid("recompiled outer program differs"));
    }
    let decision = compile_chain_decision(
        effect_chain(&point, &quantity), contract.contract().clone(), ctx,
    )?.execute(ctx)?;
    let replay = consume_chain_decision(
        &wire.chain_artifact, Some(decision.lowered_program_identity), ctx,
    )?;
    if replay.chain.identity() != decision.chain.identity()
        || replay.decision != decision.decision
    {
        return Err(GraphImputationEffectDecisionError::Invalid("graph effect utility differs on replay"));
    }
    Ok(GraphImputationEffectDecisionResult {
        producer: point, decision, lowered_program_identity: lowered.identity(),
        calibration: "not_measured",
    })
}

fn lower_paired_outer(
    producer: SemanticDigest, source: SemanticDigest,
    quantities: &[ScientificQuantity; 2],
    decision: &DecisionContract,
) -> Result<Arc<LoweredProgram>, GraphImputationEffectDecisionError> {
    decision.validate()?;
    let quantity_bytes = antecedent_io::to_cbor(&quantities.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>())?;
    let decision_identity = decision.identity()?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation_paired_world_decision.question.v1");
    writer.bytes(producer.as_bytes()).bytes(source.as_bytes())
        .bytes(&quantity_bytes).text(&decision_identity);
    let question = writer.finish();
    let design = DesignShape::quoted(1, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue, outcome: None, query: question,
        },
        plan_id: Arc::from("graph_imputation_paired_world_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "bounded_graph_imputation_paired_world_mixture",
        query_variables: Arc::from([]), source_digest: Some(source),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource, digest: Some(source),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(), constraints: decision.constraints.len(),
            chance_threshold: decision.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(decision_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("native_graph_imputation_paired_world_draws_v1")),
        }),
        planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(quantity_bytes.len() as u64))
                .build().map_err(|_| GraphImputationEffectDecisionError::Invalid("invalid producer receipt read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build().map_err(|_| GraphImputationEffectDecisionError::Invalid("invalid graph mixture operator"))?,
        ]), design,
    };
    Ok(Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::GraphImputationEffectDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Posterior,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish()))
}


/// Compiler-owned graph, imputation and shared-coefficient treatment-world decision.
pub struct CompiledGraphImputationWorldDecision {
    producer: CompiledGraphImputation,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Model-conditional decision on paired potential-outcome mean worlds.
pub struct GraphImputationWorldDecisionResult {
    /// Shared graph, imputation and coefficient world pairs.
    pub producer: GraphPairedWorldReading,
    /// Utility on those paired parameter-world rows.
    pub decision: ChainDecisionResult,
    /// Identity of the enclosing compiler program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationWorldDecisionWire {
    version: u32,
    feature: String,
    source: Vec<u8>,
    producer_program: [u8; 32],
    source_identity: [u8; 32],
    receipt: [u8; 32],
    quantities: Vec<ScientificQuantityWire>,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
}

fn validate_world_coordinates(source: &[u8], quantities: &[ScientificQuantity; 2], decision: &DecisionContract)
    -> Result<(), GraphImputationEffectDecisionError> {
    decision.validate()?;
    let source = GraphImputationSource::decode(source).map_err(GraphImputationEffectDecisionError::Invalid)?;
    let parts = source.parts();
    let treatment = &parts.names[parts.treatment as usize];
    let outcome = &parts.names[parts.outcome as usize];
    for (i, quantity) in quantities.iter().enumerate() {
        if quantity.variable_id != *outcome || quantity.role != QuantityRole::Outcome
            || quantity.transform_id != "identity" || !quantity.conditioning.is_empty()
            || quantity.units != quantities[0].units || quantity.functional_id != "mean"
            || quantity.regime_id != format!("do({treatment}={i})")
            || quantity.population_id != decision.target_population || quantity.horizon != 0
            || decision.horizon != 0
        {
            return Err(GraphImputationEffectDecisionError::Invalid("world coordinates differ from source treatment or outcome"));
        }
    }
    if decision.actions.len() != 2 || decision.actions.iter().any(|action|
            !matches!(action.kind, ActionKind::Intervention | ActionKind::Regime | ActionKind::Policy))
        || quantities.iter().any(|quantity|
        !decision.actions.iter().any(|action| action.inputs.len() == 1
            && action.inputs[0].require_same_coordinate(quantity).is_ok()))
    {
        return Err(GraphImputationEffectDecisionError::Invalid("decision must name both distinct treatment worlds"));
    }
    Ok(())
}

fn paired_world_chain(reading: &GraphPairedWorldReading, quantities: &[ScientificQuantity; 2]) -> PropagationChainRecordWire {
    let alignment = SemanticDigest::from_bytes(reading.receipt).to_hex();
    let mut sources = Vec::new();
    let mut stages = Vec::new();
    let mut outputs = Vec::new();
    for (i, quantity) in quantities.iter().enumerate() {
        let source = format!("graph.world.source.{i}");
        let stage = format!("graph.world.mean.{i}");
        let values: Vec<_> = reading.worlds.iter().map(|pair| pair[i]).collect();
        sources.push(ChainSourceWire::draws(&source, LawMeaning::Posterior, &alignment, &values)
            .with_factor(SharedFactorKind::Model, &alignment));
        stages.push(ChainStageWire::new(&stage, StageKind::DerivedQuantity, &[&source],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0))));
        outputs.push(ChainJointOutputWire { stage, quantity: ScientificQuantityWire::from(quantity) });
    }
    PropagationChainRecordWire::draft(TableConstruction::JointDraws, ChainReading::Posterior,
        &outputs[0].stage, sources, stages).with_output_quantity(outputs[0].quantity.clone()).with_joint_outputs(outputs)
}

/// Compile the paired-world model and decision before fitting any child.
/// Both worlds are mean potential outcomes, not individual predictive draws.
///
/// # Errors
/// Mismatched outcome, treatment, regimes, actions or decision contract.
pub fn compile_graph_imputation_world_decision(
    producer: CompiledGraphImputation, quantities: [ScientificQuantity; 2], decision: DecisionContract,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    validate_world_coordinates(producer.source_bytes(), &quantities, &decision)?;
    let lowered = lower_paired_outer(producer.lowered().identity(), producer.source_identity(), &quantities, &decision)?;
    Ok(CompiledGraphImputationWorldDecision { producer, quantities, decision, lowered })
}

impl CompiledGraphImputationWorldDecision {
    /// Retained six-layer joint decision program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }
    /// Inspect the compiled program.
    #[must_use]
    pub fn plan_view(&self) -> PlanView { PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled) }
    /// Refit shared graph/imputation worlds and evaluate the action decision.
    ///
    /// # Errors
    /// Any graph, child, pairing or utility refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<GraphImputationWorldDecisionResult, GraphImputationEffectDecisionError> {
        let reading = replay_paired_worlds(self.producer.source_bytes(), ctx)?;
        let chain = compile_action_world_chain_decision(paired_world_chain(&reading, &self.quantities), self.decision.clone(), ctx)?;
        let decision = chain.execute(ctx)?;
        Ok(GraphImputationWorldDecisionResult { producer: reading, decision,
            lowered_program_identity: self.lowered.identity(), calibration: "not_measured" })
    }
    /// Recompute and seal the paired schedule and decision for replay.
    ///
    /// # Errors
    /// Changed result, nested refusal or oversized artifact.
    pub fn export(&self, result: &GraphImputationWorldDecisionResult, artifact_id: &str, ctx: &ExecutionContext)
        -> Result<Vec<u8>, GraphImputationEffectDecisionError> {
        let live = self.execute(ctx)?;
        if result.producer.receipt != live.producer.receipt || result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured" || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity() {
            return Err(GraphImputationEffectDecisionError::Invalid("paired world decision changed"));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_action_world_chain_decision(paired_world_chain(&live.producer, &self.quantities), self.decision.clone(), ctx)?;
        let bytes = antecedent_io::to_cbor(&GraphImputationWorldDecisionWire {
            version: 1, feature: "graph_imputation_paired_world_decision_v1".into(),
            source: self.producer.source_bytes().to_vec(), producer_program: *self.producer.lowered().identity().as_bytes(),
            source_identity: *self.producer.source_identity().as_bytes(), receipt: live.producer.receipt,
            quantities: self.quantities.iter().map(ScientificQuantityWire::from).collect(),
            decision_identity: contract.identity().into(), decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?, program_identity: *self.lowered.identity().as_bytes(),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES { return Err(GraphImputationEffectDecisionError::Invalid("paired world artifact exceeds its bound")); }
        Ok(bytes)
    }
}

/// Independently reconstruct the source, paired world rows and action decision.
///
/// # Errors
/// Changed program, source, receipt, quantity, decision or paired replay.
pub fn consume_graph_imputation_world_decision(bytes: &[u8], expected: SemanticDigest, ctx: &ExecutionContext)
    -> Result<GraphImputationWorldDecisionResult, GraphImputationEffectDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES { return Err(GraphImputationEffectDecisionError::Invalid("paired world artifact exceeds its bound")); }
    let wire: GraphImputationWorldDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.feature != "graph_imputation_paired_world_decision_v1" || wire.program_identity != *expected.as_bytes() {
        return Err(GraphImputationEffectDecisionError::Invalid("paired world feature or identity differs"));
    }
    let reading = replay_paired_worlds(&wire.source, ctx)?;
    if reading.program != wire.producer_program || reading.source != wire.source_identity || reading.receipt != wire.receipt {
        return Err(GraphImputationEffectDecisionError::Invalid("paired world source or receipt differs"));
    }
    let quantities: Vec<ScientificQuantity> = wire.quantities.into_iter().map(|q| q.try_into().map_err(GraphImputationEffectDecisionError::Invalid)).collect::<Result<_, _>>()?;
    let quantities: [ScientificQuantity; 2] = quantities.try_into().map_err(|_| GraphImputationEffectDecisionError::Invalid("paired world coordinates missing"))?;
    let contract = DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    validate_world_coordinates(&wire.source, &quantities, contract.contract())?;
    let lowered = lower_paired_outer(SemanticDigest::from_bytes(reading.program), SemanticDigest::from_bytes(reading.source), &quantities, contract.contract())?;
    if lowered.identity() != expected { return Err(GraphImputationEffectDecisionError::Invalid("paired world outer program differs")); }
    let decision = compile_action_world_chain_decision(paired_world_chain(&reading, &quantities), contract.contract().clone(), ctx)?.execute(ctx)?;
    let replay = consume_chain_decision(&wire.chain_artifact, Some(decision.lowered_program_identity), ctx)?;
    if replay.chain.identity() != decision.chain.identity() || replay.decision != decision.decision {
        return Err(GraphImputationEffectDecisionError::Invalid("paired world decision replay differs"));
    }
    Ok(GraphImputationWorldDecisionResult { producer: reading, decision, lowered_program_identity: expected, calibration: "not_measured" })
}

#[cfg(test)]
mod paired_world_tests {
    use super::*;
    use antecedent_core::{VariableId, graph_imputation_contract::GraphSupport};
    use antecedent_data::{ImputationSpec, TabularData};
    use antecedent_design::decision_contract::{DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr};
    use super::super::graph_imputation::GraphImputationSpec;

    #[test]
    #[ignore = "source-only paired graph-imputation world decision; reliability not earned"]
    fn paired_world_decision_replays() {
        let treatment: Vec<_> = (0..80).map(|i| f64::from(u8::from(i % 4 == 0 || i % 4 == 3))).collect();
        let covariate: Vec<_> = (0..80).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let outcome: Vec<_> = (0..80).map(|i| if i % 13 == 0 { f64::NAN } else {
            0.2 * covariate[i] + 0.5 * treatment[i] + 0.1 * (i % 5) as f64
        }).collect();
        let table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()), ("y", outcome.as_slice()), ("x", covariate.as_slice()),
        ]).unwrap();
        let ctx = ExecutionContext::for_tests(19);
        let producer = CompiledGraphImputation::compile(table, &GraphImputationSpec {
            treatment: VariableId::from_raw(0), outcome: VariableId::from_raw(1),
            support: GraphSupport::TreatmentRootPreTreatmentCovariates,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from(vec![1, 0, 1]), posterior_draws: 100, prior_scale: 1.0, execution_seed: 19,
        }, &ctx).unwrap();
        let quantity = |a| ScientificQuantity {
            variable_id: "y".into(), variable_name: "outcome".into(), role: QuantityRole::Outcome,
            units: "score".into(), population_id: "source".into(), regime_id: format!("do(t={a})"),
            horizon: 0, functional_id: "mean".into(), conditioning: Vec::new(), transform_id: "identity".into(),
        };
        let quantities = [quantity(0), quantity(1)];
        let decision = DecisionContract {
            actions: quantities.iter().enumerate().map(|(i, q)| DecisionAction {
                id: format!("action-{i}"), kind: ActionKind::Intervention,
                inputs: vec![q.clone()], utility: UtilityExpr::Input(0),
            }).collect(),
            utility_units: "score".into(), criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(), target_population: "source".into(), horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let plan = compile_graph_imputation_world_decision(producer, quantities.clone(), decision).unwrap();
        let result = plan.execute(&ctx).unwrap();
        let joint = paired_world_chain(&result.producer, &quantities);
        assert_eq!(joint.sources[0].alignment, joint.sources[1].alignment);
        assert_eq!(joint.sources[0].factors, joint.sources[1].factors);
        for (i, pair) in result.producer.worlds.iter().enumerate() {
            assert_eq!(joint.sources[0].values[i], pair[0].to_bits());
            assert_eq!(joint.sources[1].values[i], pair[1].to_bits());
        }
        assert_eq!(result.calibration, "not_measured");
        let artifact = plan.export(&result, "paired-world", &ctx).unwrap();
        let replay = consume_graph_imputation_world_decision(&artifact, plan.lowered().identity(), &ctx).unwrap();
        assert_eq!(result.producer.receipt, replay.producer.receipt);
        assert_eq!(result.producer.worlds, replay.producer.worlds);
        assert_eq!(result.decision.decision, replay.decision.decision);
        assert!(result.producer.worlds.iter().any(|pair| pair[0] != pair[1]));
        let mut wire: GraphImputationWorldDecisionWire = antecedent_io::from_cbor(&artifact).unwrap();
        wire.receipt[0] ^= 1;
        let tampered = antecedent_io::to_cbor(&wire).unwrap();
        assert!(consume_graph_imputation_world_decision(&tampered, plan.lowered().identity(), &ctx).is_err());
    }
}
