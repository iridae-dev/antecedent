//! One compiler program for a predictive bridge, transport premise and utility.
//!
//! The bridge's target coordinate and transport premise are checked before
//! mapping draws. Execution derives the terminal-outcome law, lowers that
//! stage through the chain compiler, and evaluates nonlinear utility on the
//! retained rows. Its decision is conditional on the bridge's explicitly
//! declared transport premise; this route does not estimate that premise or
//! acquire coverage from the source.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::bridge::{BridgeContract, ValueKind};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion,
};
use antecedent_io::IoError;
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::propagation_chain_record::{
    ChainExprTermWire, ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use serde::{Deserialize, Serialize};

use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision,
    consume_chain_decision,
};
use super::propagation_chain::PropagationChain;
use super::propagation_chain_robust_decision::{
    ChainRobustDecisionPlan, ChainRobustDecisionResult,
    compile_chain_robust_decision, consume_chain_robust_decision,
};
use super::scientific_bridge::{
    PreparedScientificBridgeDistribution, ScientificBridgeDistributionResult,
    ScientificBridgeRefusal, consume_scientific_bridge_distribution_artifact,
};
use super::bridge_transport_premise_assessment::{
    BridgeTransportPremiseAssessmentError, BridgeTransportPremiseAssessmentResult,
    CompiledBridgeTransportPremiseAssessment, recompile_bridge_transport_premise_assessment,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification,
    PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, NestedFitLaw, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "bridge_transport_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;
const ASSESSED_FEATURE: &str = "assessed_bridge_transport_decision_v1";

/// Refusal of a bridge, coordinate, decision or independent artifact replay.
#[derive(Debug, thiserror::Error)]
pub enum BridgeTransportDecisionError {
    /// A required predictive or transport declaration is absent or mismatched.
    #[error("bridge transport decision: {0}")]
    Invalid(&'static str),
    /// Bridge preparation, derivation or replay refused.
    #[error("bridge transport decision bridge: {0}")]
    Bridge(#[from] ScientificBridgeRefusal),
    /// Decision contract refused.
    #[error("bridge transport decision contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Chain or utility compiler refused.
    #[error("bridge transport decision chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or decoding refused.
    #[error("bridge transport decision artifact: {0}")]
    Io(#[from] IoError),
}

/// Checked declaration, aligned source law and retained six-layer program.
#[derive(Clone, Debug)]
pub struct CompiledBridgeTransportDecision {
    bridge: PreparedScientificBridgeDistribution,
    source: DistributionArtifact,
    seed: u64,
    decision: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Mapped terminal-outcome law and decision on the exact target coordinate.
#[derive(Clone, Debug)]
pub struct BridgeTransportDecisionResult {
    /// Replayed bridge mapping and transported law.
    pub bridge: ScientificBridgeDistributionResult,
    /// Compiler-retained output law and nonlinear decision.
    pub decision: ChainDecisionResult,
    /// Identity of the complete bridge-to-utility program.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BridgeTransportDecisionWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    bridge_identity: String,
    bridge_artifact: Vec<u8>,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
}

/// Compile a validated predictive source law, scientific bridge with an
/// explicit transport premise, and a nonlinear decision as one program.
/// The actions must read precisely the bridge's terminal-outcome quantity.
/// The bridge mapping is executed later, under this retained program.
///
/// # Errors
/// An unsupported law kind, absent transport premise, mismatched coordinate,
/// invalid contract or physical lowering refusal.
pub fn compile_bridge_transport_decision(
    contract: BridgeContract,
    source: DistributionArtifact,
    seed: u64,
    decision: DecisionContract,
) -> Result<CompiledBridgeTransportDecision, BridgeTransportDecisionError> {
    decision.validate()?;
    if contract.transport.is_none() || contract.source_value != ValueKind::PredictiveLaw
        || contract.target_value != ValueKind::PredictiveLaw
        || !matches!(source.semantic(),
            DistributionMeaningWire::PosteriorPredictive
                | DistributionMeaningWire::InterventionalPredictive
                | DistributionMeaningWire::EmpiricalOutcome)
        || !source.quantities().contains(&ScientificQuantityWire::from(&contract.source.quantity))
        || source.n_draws() < 2
        || contract.target.quantity.population_id != decision.target_population
        || contract.target.quantity.horizon != decision.horizon
        || decision.actions.is_empty()
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1
                || action.inputs[0].require_same_coordinate(&contract.target.quantity).is_err()
        })
    {
        return Err(BridgeTransportDecisionError::Invalid(
            "a predictive terminal bridge, transport premise and exact decision coordinate are required",
        ));
    }
    let bridge = PreparedScientificBridgeDistribution::prepare(contract)?;
    let decision_identity = decision.identity()?;
    let source_bytes = source.to_bytes("bridge-transport-source")?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.bridge_transport_decision.source.v1", &source_bytes,
    );
    let mut writer = CanonicalWriter::new("antecedent.bridge_transport_decision.question.v1");
    writer.bytes(&source_digest).bytes(bridge.contract_identity().as_bytes())
        .text(&decision_identity).word(seed);
    let query = writer.finish();
    let design = crate::planner::lowered::DesignShape::quoted(source.n_draws() as u64, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue, outcome: None, query,
        },
        plan_id: Arc::from("bridge_transport_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "declared_transportable_bridge",
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::DeclaredPredictive {
            evaluator: Arc::from("scientific_bridge.predictive_transport"),
        },
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource, digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::PredictiveLaw,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(), constraints: decision.constraints.len(),
            chance_threshold: decision.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(decision_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("transported_bridge_joint_draws_v1")),
        }),
        planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| BridgeTransportDecisionError::Invalid(
                    "bridge source intake cannot be lowered",
                ))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .estimates(ByteEstimates::absent().with_output(
                    (source.n_draws() as u64).saturating_mul(8),
                )).build().map_err(|_| BridgeTransportDecisionError::Invalid(
                    "bridge transport cannot be lowered",
                ))?,
        ]),
        design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::BridgeTransportDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Predictive,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish());
    Ok(CompiledBridgeTransportDecision { bridge, source, seed, decision, lowered })
}

impl CompiledBridgeTransportDecision {
    /// All compiler layers of the bridge, premise and utility.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained program without mapping or scoring draws.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    fn chain_draft(
        &self, result: &ScientificBridgeDistributionResult,
    ) -> Result<PropagationChainRecordWire, BridgeTransportDecisionError> {
        result.require_value_kind(ValueKind::PredictiveLaw)?;
        let parts = result.chain_stage("bridged.outcome")?;
        let output = parts.stage.id.clone();
        Ok(PropagationChainRecordWire::draft(
            TableConstruction::JointDraws, parts.reading, &output,
            parts.sources, vec![parts.stage],
        ).with_output_quantity(parts.output_quantity))
    }

    /// Derive the transported terminal-outcome law and score its utility.
    /// The reported spread stays uncalibrated and the transport premise stays
    /// a declared assumption.
    ///
    /// # Errors
    /// A bridge mapping, chain or decision refusal.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<BridgeTransportDecisionResult, BridgeTransportDecisionError> {
        let bridge = self.bridge.derive(&self.source, None, self.seed)?;
        let draft = self.chain_draft(&bridge)?;
        let decision = compile_chain_decision(draft, self.decision.clone(), ctx)?
            .execute(ctx)?;
        Ok(BridgeTransportDecisionResult {
            bridge, decision, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Export the independently replayable bridge and chain decision after a
    /// fresh execution matches this result.
    ///
    /// # Errors
    /// Changed derivation, decision or artifact encoding failure.
    pub fn export(
        &self, result: &BridgeTransportDecisionResult, artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, BridgeTransportDecisionError> {
        let replay = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.bridge.inspect() != replay.bridge.inspect()
            || result.decision.decision != replay.decision.decision
            || result.decision.chain.identity() != replay.decision.chain.identity()
        {
            return Err(BridgeTransportDecisionError::Invalid("bridge decision differs on replay"));
        }
        let bridge = result.bridge.export(artifact_id)?;
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(self.chain_draft(&replay.bridge)?,
            self.decision.clone(), ctx)?;
        let wire = BridgeTransportDecisionWire {
            version: 1, required_feature: FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(),
            bridge_identity: bridge.identity, bridge_artifact: bridge.bytes,
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&result.decision, artifact_id)?,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(BridgeTransportDecisionError::Invalid("combined artifact exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently replay the bridge mapping, transport premise and utility
/// from their embedded artifacts under an externally retained program id.
///
/// # Errors
/// A changed bridge, contract, law, decision or program identity.
pub fn consume_bridge_transport_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<BridgeTransportDecisionResult, BridgeTransportDecisionError> {
    Ok(recompile_bridge_transport_decision(bytes, expected_program, ctx)?.1)
}

fn recompile_bridge_transport_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<(CompiledBridgeTransportDecision, BridgeTransportDecisionResult), BridgeTransportDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(BridgeTransportDecisionError::Invalid("combined artifact exceeds its bound"));
    }
    let wire: BridgeTransportDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(BridgeTransportDecisionError::Invalid("combined feature or identity differs"));
    }
    let bridge = consume_scientific_bridge_distribution_artifact(
        &wire.bridge_artifact, Some(&wire.bridge_identity),
    )?;
    let contract = DecisionContractArtifact::from_bytes(
        &wire.decision_artifact, &wire.decision_identity,
    )?;
    let plan = compile_bridge_transport_decision(
        bridge.bridged.claim().contract.clone(), bridge.bridged.source().clone(),
        bridge.bridged.claim().stream, contract.contract().clone(),
    )?;
    if plan.lowered.identity() != expected_program {
        return Err(BridgeTransportDecisionError::Invalid("recompiled bridge decision differs"));
    }
    let live = plan.execute(ctx)?;
    if live.bridge.inspect().claim_id != bridge.bridged.claim().claim.claim_id.to_hex() {
        return Err(BridgeTransportDecisionError::Invalid("mapped bridge law changed on replay"));
    }
    let inner = consume_chain_decision(
        &wire.chain_artifact, Some(live.decision.lowered_program_identity), ctx,
    )?;
    if inner.chain.identity() != live.decision.chain.identity()
        || inner.decision != live.decision.decision
    {
        return Err(BridgeTransportDecisionError::Invalid("utility changed on replay"));
    }
    Ok((plan, live))
}

/// One compiler program retaining independent target validation and the
/// conditional predictive bridge decision. The assessment is diagnostic;
/// its screen does not establish transport or calibrate decision risk.
#[derive(Clone, Debug)]
pub struct CompiledAssessedBridgeTransportDecision {
    assessment: CompiledBridgeTransportPremiseAssessment,
    bridge: CompiledBridgeTransportDecision,
    lowered: Arc<LoweredProgram>,
}

/// The matched premise diagnostic and conditional target decision.
#[derive(Clone, Debug)]
pub struct AssessedBridgeTransportDecisionResult {
    /// Independent source-target trial mapping assessment.
    pub assessment: BridgeTransportPremiseAssessmentResult,
    /// Predictive decision conditional on the declared transport premise.
    pub bridge: BridgeTransportDecisionResult,
    /// Identity of the combined compiler schedule.
    pub lowered_program_identity: SemanticDigest,
}

/// Refusal of the combined assessment, bridge, lowering or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum AssessedBridgeTransportDecisionError {
    /// The assessment and bridge do not name identical scientific premises.
    #[error("assessed bridge decision: {0}")]
    Invalid(&'static str),
    /// The downstream decision contract refused identity construction.
    #[error("assessed bridge decision contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Validation fitting or its independent replay refused.
    #[error("assessed bridge decision assessment: {0}")]
    Assessment(#[from] BridgeTransportPremiseAssessmentError),
    /// Predictive bridge or utility refused.
    #[error("assessed bridge decision bridge: {0}")]
    Bridge(#[from] BridgeTransportDecisionError),
    /// Artifact encoding or decoding refused.
    #[error("assessed bridge decision artifact: {0}")]
    Io(#[from] IoError),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AssessedBridgeDecisionWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    assessment_identity: [u8; 32],
    assessment_artifact: Vec<u8>,
    bridge_identity: [u8; 32],
    bridge_artifact: Vec<u8>,
}

/// Bind source and target validation evidence to precisely the bridge premise,
/// source trial mapping and predictive utility under a new six-layer program.
///
/// # Errors
/// The assessment does not match the bridge or physical lowering refuses.
pub fn compile_assessed_bridge_transport_decision(
    assessment: CompiledBridgeTransportPremiseAssessment,
    bridge: CompiledBridgeTransportDecision,
) -> Result<CompiledAssessedBridgeTransportDecision, AssessedBridgeTransportDecisionError> {
    if !assessment.matches_bridge(bridge.bridge.contract()) {
        return Err(AssessedBridgeTransportDecisionError::Invalid(
            "assessment must match the exact source mapping and transport premise",
        ));
    }
    let assessment_id = assessment.lowered().identity();
    let bridge_id = bridge.lowered().identity();
    let mut writer = CanonicalWriter::new("antecedent.assessed_bridge_transport_decision.question.v1");
    writer.bytes(assessment_id.as_bytes()).bytes(bridge_id.as_bytes());
    let query = writer.finish();
    let design = crate::planner::lowered::DesignShape::quoted(
        bridge.source.n_draws() as u64, 2, true,
    );
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue, outcome: None, query,
        },
        plan_id: Arc::from("assessed_bridge_transport_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "assessed_declared_transportable_bridge",
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(*assessment_id.as_bytes()),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::DeclaredPredictive {
            evaluator: Arc::from("assessed_bridge.predictive_transport"),
        },
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(*assessment_id.as_bytes()) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(*bridge_id.as_bytes()) },
        ]),
        composition: Composition::Chained { stages: 2 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::PredictiveLaw,
            utility: DecisionProgram::utility_form(None),
            actions: bridge.decision.actions.len(),
            constraints: bridge.decision.constraints.len(),
            chance_threshold: bridge.decision.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(bridge.decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(bridge.decision.identity()?.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("assessed_bridge_joint_draws_v1")),
        }),
        planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .build().map_err(|_| AssessedBridgeTransportDecisionError::Invalid(
                    "assessment intake cannot be lowered",
                ))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .estimates(ByteEstimates::absent().with_output(
                    (bridge.source.n_draws() as u64).saturating_mul(8),
                )).build().map_err(|_| AssessedBridgeTransportDecisionError::Invalid(
                    "assessed bridge transport cannot be lowered",
                ))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::AssessedBridgeTransportDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Predictive,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish());
    Ok(CompiledAssessedBridgeTransportDecision { assessment, bridge, lowered })
}

impl CompiledAssessedBridgeTransportDecision {
    /// All compiler layers of the matched validation and decision schedule.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the combined schedule before fitting or mapping.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Refit both validation scopes, then derive and score the conditional law.
    /// A failed equivalence screen remains visible and does not silently change
    /// or discharge the premise used by the decision.
    ///
    /// # Errors
    /// Validation, bridge or decision execution refused.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<AssessedBridgeTransportDecisionResult, AssessedBridgeTransportDecisionError> {
        let assessment = self.assessment.execute(ctx)?;
        let bridge = self.bridge.execute(ctx)?;
        Ok(AssessedBridgeTransportDecisionResult {
            assessment, bridge, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Export both independently replayable child receipts after a fresh match.
    ///
    /// # Errors
    /// A child changed or artifact encoding refused.
    pub fn export(
        &self, result: &AssessedBridgeTransportDecisionResult, artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, AssessedBridgeTransportDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.assessment != live.assessment
            || result.bridge.bridge.inspect() != live.bridge.bridge.inspect()
            || result.bridge.decision.decision != live.bridge.decision.decision
            || result.bridge.decision.chain.identity() != live.bridge.decision.chain.identity()
        {
            return Err(AssessedBridgeTransportDecisionError::Invalid(
                "combined result differs on replay",
            ));
        }
        let wire = AssessedBridgeDecisionWire {
            version: 1, required_feature: ASSESSED_FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(),
            assessment_identity: *self.assessment.lowered().identity().as_bytes(),
            assessment_artifact: self.assessment.export(&result.assessment, ctx)?,
            bridge_identity: *self.bridge.lowered().identity().as_bytes(),
            bridge_artifact: self.bridge.export(&result.bridge, artifact_id, ctx)?,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(AssessedBridgeTransportDecisionError::Invalid(
                "combined artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recompile the two child receipts, check their shared premise again and
/// execute the outer program under an independently retained identity.
///
/// # Errors
/// A foreign feature, changed child, premise or combined result.
pub fn consume_assessed_bridge_transport_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<AssessedBridgeTransportDecisionResult, AssessedBridgeTransportDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(AssessedBridgeTransportDecisionError::Invalid(
            "combined artifact exceeds its byte bound",
        ));
    }
    let wire: AssessedBridgeDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != ASSESSED_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(AssessedBridgeTransportDecisionError::Invalid(
            "combined feature or program identity differs",
        ));
    }
    let (assessment, assessed) = recompile_bridge_transport_premise_assessment(
        &wire.assessment_artifact, SemanticDigest::from_bytes(wire.assessment_identity), ctx,
    )?;
    let (bridge, bridged) = recompile_bridge_transport_decision(
        &wire.bridge_artifact, SemanticDigest::from_bytes(wire.bridge_identity), ctx,
    )?;
    let plan = compile_assessed_bridge_transport_decision(assessment, bridge)?;
    if plan.lowered.identity() != expected_program {
        return Err(AssessedBridgeTransportDecisionError::Invalid(
            "recompiled combined program differs",
        ));
    }
    let live = plan.execute(ctx)?;
    if live.assessment != assessed
        || live.bridge.bridge.inspect() != bridged.bridge.inspect()
        || live.bridge.decision.decision != bridged.decision.decision
        || live.bridge.decision.chain.identity() != bridged.decision.chain.identity()
    {
        return Err(AssessedBridgeTransportDecisionError::Invalid(
            "combined child result changed on replay",
        ));
    }
    Ok(live)
}

const ASSESSED_ROBUST_FEATURE: &str = "assessed_bridge_discrepancy_robust_decision_v1";

/// A separate conservative decision that carries the assessed target-minus-
/// source mapping discrepancy into every predictive bridge row. The
/// assessment band is an unmeasured candidate, so this is an assumption
/// envelope, not a confidence or posterior decision.
#[derive(Clone, Debug)]
pub struct CompiledAssessedBridgeRobustDecision {
    assessed: CompiledAssessedBridgeTransportDecision,
    contract: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// The source-target assessment, conditional bridge and robust utility bounds.
#[derive(Clone, Debug)]
pub struct AssessedBridgeRobustDecisionResult {
    /// Both prerequisite compiler results.
    pub assessed: AssessedBridgeTransportDecisionResult,
    /// Conservative utility and feasibility bounds over the shift envelope.
    pub robust: ChainRobustDecisionResult,
    /// Outer six-layer identity binding validation, bridge and robust utility.
    pub lowered_program_identity: SemanticDigest,
    /// No complete-route reliability or coverage record has been earned.
    pub calibration: &'static str,
}

/// Refusal of the discrepancy envelope, robust utility or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum AssessedBridgeRobustDecisionError {
    /// A source draw leaves assessed support or a binding differs.
    #[error("assessed bridge robust decision: {0}")]
    Invalid(&'static str),
    /// The assessment or conditional bridge refused.
    #[error("assessed bridge robust prerequisite: {0}")]
    Assessed(#[from] AssessedBridgeTransportDecisionError),
    /// The robust decision contract refused.
    #[error("assessed bridge robust contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// The assumption-envelope chain or utility refused.
    #[error("assessed bridge robust chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or independent replay refused.
    #[error("assessed bridge robust artifact: {0}")]
    Io(#[from] IoError),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AssessedBridgeRobustWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    assessment_identity: [u8; 32],
    assessment_artifact: Vec<u8>,
    bridge_identity: [u8; 32],
    bridge_artifact: Vec<u8>,
    robust_contract_identity: String,
    robust_contract_artifact: Vec<u8>,
    robust_identity: [u8; 32],
    robust_artifact: Vec<u8>,
}

/// Compile a matched assessment and predictive bridge with a separate
/// maximin contract over the bridge's exact target outcome coordinate.
///
/// # Errors
/// A nonmatching coordinate, nonrobust criterion or lowering refusal.
pub fn compile_assessed_bridge_robust_decision(
    assessed: CompiledAssessedBridgeTransportDecision,
    contract: DecisionContract,
) -> Result<CompiledAssessedBridgeRobustDecision, AssessedBridgeRobustDecisionError> {
    contract.validate()?;
    let target = &assessed.bridge.bridge.contract().target.quantity;
    if !matches!(contract.criterion, DecisionCriterion::MaximinOverStructures)
        || contract.target_population != target.population_id
        || contract.horizon != target.horizon
        || contract.actions.iter().any(|action| action.inputs.iter().any(|input| {
            input.require_same_coordinate(target).is_err()
        })) {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "robust actions must read the assessed bridge target and use maximin",
        ));
    }
    let contract_id = contract.identity()?;
    let child_id = assessed.lowered.identity();
    let mut writer = CanonicalWriter::new("antecedent.assessed_bridge_robust.question.v1");
    writer.bytes(child_id.as_bytes()).text(&contract_id)
        .text("global_overlap_grid_discrepancy_envelope_v1");
    let query = writer.finish();
    let design = crate::planner::lowered::DesignShape::quoted(
        assessed.bridge.source.n_draws() as u64, 2, true,
    );
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue, outcome: None, query,
        },
        plan_id: Arc::from("assessed_bridge_discrepancy_robust.v1"),
        data: DataClassification::Tabular,
        graph_class: "assessed_bridge_discrepancy_envelope",
        query_variables: Arc::from(Vec::new()), source_digest: Some(*child_id.as_bytes()),
        support_coordinate: None, supported: false, support_status: None,
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource, digest: Some(*child_id.as_bytes()),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::IdentifiedSet,
            utility: DecisionProgram::utility_form(None),
            actions: contract.actions.len(), constraints: contract.constraints.len(),
            chance_threshold: contract.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(contract.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract_id.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("unmeasured_global_discrepancy_envelope_v1")),
        }), planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .build().map_err(|_| AssessedBridgeRobustDecisionError::Invalid(
                    "robust bridge assessment read cannot be lowered",
                ))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .estimates(ByteEstimates::absent().with_output(
                    (assessed.bridge.source.n_draws() as u64).saturating_mul(16),
                )).build().map_err(|_| AssessedBridgeRobustDecisionError::Invalid(
                    "robust bridge shift cannot be lowered",
                ))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::AssessedBridgeTransportDecision,
        scientific,
        identification: IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SharedEvidenceContract { contract: query },
        )),
        statistical: StatisticalProgram {
            identifier: None, estimator: None,
            realization: Realization::DeclaredAssumptionEnvelope {
                evaluator: Arc::from("assessed_bridge.global_discrepancy_envelope"),
                draw_meaning: LawMeaning::Predictive,
            },
            models: Arc::from([]), folds: None, resampling: None,
            posterior: None, design, operators: Arc::from([]),
        },
        evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::AssumptionEnvelope,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish());
    Ok(CompiledAssessedBridgeRobustDecision { assessed, contract, lowered })
}

fn discrepancy_robust_plan(
    assessed: &AssessedBridgeTransportDecisionResult,
    contract: &DecisionContract, ctx: &ExecutionContext,
) -> Result<ChainRobustDecisionPlan, AssessedBridgeRobustDecisionError> {
    let grid = &assessed.assessment.grid;
    let (Some(first), Some(last)) = (grid.first(), grid.last()) else {
        return Err(AssessedBridgeRobustDecisionError::Invalid("assessment grid is empty"));
    };
    let lower = grid.iter().map(|point| point.lower).fold(f64::INFINITY, f64::min);
    let upper = grid.iter().map(|point| point.upper).fold(f64::NEG_INFINITY, f64::max);
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed discrepancy envelope is invalid",
        ));
    }
    let bridged = assessed.bridge.bridge.bridged();
    let source = bridged.source();
    let mapped = bridged.distribution();
    let column = bridged.column();
    let width = source.quantities().len();
    if width == 0 || column >= width || mapped.quantities().len() != width
        || source.n_draws() != mapped.n_draws() {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "source and mapped bridge rows are not aligned",
        ));
    }
    let mut values = Vec::with_capacity(source.n_draws());
    for row in 0..source.n_draws() {
        let offset = row * width + column;
        let x = source.draws()[offset];
        let y = mapped.draws()[offset];
        if !x.is_finite() || !y.is_finite()
            || x < first.surrogate_effect || x > last.surrogate_effect {
            return Err(AssessedBridgeRobustDecisionError::Invalid(
                "a source draw leaves the assessed surrogate overlap grid",
            ));
        }
        values.push(y);
    }
    let target = &assessed.bridge.bridge.claim().contract.target.quantity;
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws, ChainReading::AssumptionEnvelope,
        "target_terminal", vec![
            ChainSourceWire::draws("source_mapped", LawMeaning::Predictive,
                bridged.alignment_identity(), &values),
            ChainSourceWire::interval("target_minus_source", lower, upper),
        ], vec![ChainStageWire::new(
            "target_terminal", StageKind::DerivedQuantity,
            &["source_mapped", "target_minus_source"], ChainOpWire::binary("sum"),
        )],
    ).with_output_quantity(ScientificQuantityWire::from(target));
    Ok(compile_chain_robust_decision(draft, contract.clone(), ctx)?)
}

impl CompiledAssessedBridgeRobustDecision {
    /// Complete compiler program of the assessed discrepancy decision.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the bound graph, validation, mapping and utility before execution.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Refit validation, derive the predictive bridge, enclose every mapped
    /// row under the unmeasured discrepancy band and score robust utility.
    ///
    /// # Errors
    /// A child, support, chain or utility refusal.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<AssessedBridgeRobustDecisionResult, AssessedBridgeRobustDecisionError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let assessed = self.assessed.execute(&scoped)?;
        let robust = discrepancy_robust_plan(&assessed, &self.contract, &scoped)?.execute(&scoped)?;
        Ok(AssessedBridgeRobustDecisionResult {
            assessed, robust, lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Seal the child evidence and the independently replayable robust chain.
    ///
    /// # Errors
    /// Changed result, child or artifact encoding refusal.
    pub fn export(&self, result: &AssessedBridgeRobustDecisionResult,
        artifact_id: &str, ctx: &ExecutionContext)
        -> Result<Vec<u8>, AssessedBridgeRobustDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != live.lowered_program_identity
            || result.calibration != "not_measured"
            || result.assessed.assessment != live.assessed.assessment
            || result.robust != live.robust {
            return Err(AssessedBridgeRobustDecisionError::Invalid(
                "assessed robust decision differs on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let robust_plan = discrepancy_robust_plan(&live.assessed, &self.contract, ctx)?;
        let wire = AssessedBridgeRobustWire {
            version: 1, required_feature: ASSESSED_ROBUST_FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(),
            assessment_identity: *self.assessed.assessment.lowered().identity().as_bytes(),
            assessment_artifact: self.assessed.assessment.export(&live.assessed.assessment, ctx)?,
            bridge_identity: *self.assessed.bridge.lowered().identity().as_bytes(),
            bridge_artifact: self.assessed.bridge.export(
                &live.assessed.bridge, artifact_id, ctx,
            )?,
            robust_contract_identity: contract.identity().into(),
            robust_contract_artifact: contract.to_bytes(artifact_id)?,
            robust_identity: *robust_plan.lowered().identity().as_bytes(),
            robust_artifact: robust_plan.export(&live.robust, artifact_id)?,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(AssessedBridgeRobustDecisionError::Invalid(
                "assessed robust artifact exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recompile the validation and bridge children, regenerate the discrepancy
/// envelope and independently replay robust utility under a retained identity.
///
/// # Errors
/// A foreign feature, changed child, support domain, contract or result.
pub fn consume_assessed_bridge_robust_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<AssessedBridgeRobustDecisionResult, AssessedBridgeRobustDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed robust artifact exceeds its bound",
        ));
    }
    let wire: AssessedBridgeRobustWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != ASSESSED_ROBUST_FEATURE
        || wire.program_identity != *expected_program.as_bytes() {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed robust feature or identity differs",
        ));
    }
    let (assessment, assessed) = recompile_bridge_transport_premise_assessment(
        &wire.assessment_artifact, SemanticDigest::from_bytes(wire.assessment_identity), ctx,
    ).map_err(AssessedBridgeTransportDecisionError::from)?;
    let (bridge, bridged) = recompile_bridge_transport_decision(
        &wire.bridge_artifact, SemanticDigest::from_bytes(wire.bridge_identity), ctx,
    ).map_err(AssessedBridgeTransportDecisionError::from)?;
    let contract = DecisionContractArtifact::from_bytes(
        &wire.robust_contract_artifact, &wire.robust_contract_identity,
    )?;
    let child = compile_assessed_bridge_transport_decision(assessment, bridge)?;
    let plan = compile_assessed_bridge_robust_decision(child, contract.contract().clone())?;
    if plan.lowered().identity() != expected_program {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed robust compiler changed",
        ));
    }
    let live = plan.execute(ctx)?;
    if live.assessed.assessment != assessed
        || live.assessed.bridge.bridge.inspect() != bridged.bridge.inspect()
        || live.robust.lowered_program_identity != SemanticDigest::from_bytes(wire.robust_identity) {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed robust child result changed",
        ));
    }
    let replay = consume_chain_robust_decision(
        &wire.robust_artifact, Some(live.robust.lowered_program_identity), ctx,
    )?;
    if replay != live.robust {
        return Err(AssessedBridgeRobustDecisionError::Invalid(
            "assessed robust utility changed on replay",
        ));
    }
    Ok(live)
}

/// Scientific premises for the independent target-line posterior predictive route.
/// Target trial outcomes follow a linear normal model with independent,
/// homoscedastic residuals conditional on the observed surrogate effects. The
/// target coefficients and variance use the reference prior proportional to
/// inverse variance; a singular or zero-residual fit is refused.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeTargetPosteriorPremises {
    /// Identity of the independently collected target validation design.
    pub validation_design: String,
    /// Target trial pairs are independent of the entire source predictive schedule.
    pub target_validation_independent: bool,
    /// The original source predictive innovation and its source-row dependence
    /// transfer to the target population, independently of the target line fit.
    pub source_innovation_transfers: bool,
}

/// Compiled target-line posterior predictive bridge and expected-regret decision.
#[derive(Clone, Debug)]
pub struct CompiledAssessedBridgePosteriorDecision {
    assessed: CompiledAssessedBridgeTransportDecision,
    premises: BridgeTargetPosteriorPremises,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Joint target-posterior predictive law and decision; reliability is unmeasured.
#[derive(Clone, Debug)]
pub struct AssessedBridgePosteriorDecisionResult {
    /// Original assessment and bridge, including the retained source innovation.
    pub assessed: AssessedBridgeTransportDecisionResult,
    /// Predictive chain and nonlinear expected-regret decision.
    pub decision: ChainDecisionResult,
    /// Full outer program identity.
    pub lowered_program_identity: SemanticDigest,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AssessedBridgePosteriorWire {
    version: u32, feature: String, program_identity: [u8; 32], seed: u64,
    premises: BridgeTargetPosteriorPremises,
    assessed_identity: [u8; 32], assessed_artifact: Vec<u8>,
    chain_identity: [u8; 32], chain_artifact: Vec<u8>,
}

/// Compile a matched bridge with an independent normal target-line posterior.
/// Expected regret is evaluated under the posterior predictive distribution;
/// no sampling distribution is promoted to a posterior or outcome law.
///
/// # Errors
/// Missing premises, incompatible criterion, or lowering refusal.
pub fn compile_assessed_bridge_posterior_decision(
    assessed: CompiledAssessedBridgeTransportDecision, premises: BridgeTargetPosteriorPremises,
    ctx: &ExecutionContext,
) -> Result<CompiledAssessedBridgePosteriorDecision, AssessedBridgeRobustDecisionError> {
    if premises.validation_design.trim().is_empty() || !premises.target_validation_independent
        || !premises.source_innovation_transfers
        || !matches!(assessed.bridge.decision.criterion, DecisionCriterion::ExpectedRegret) {
        return Err(AssessedBridgeRobustDecisionError::Invalid("independent target validation, transferred innovation and expected-regret criterion are required"));
    }
    let seed = ctx.rng.master_seed();
    let child = assessed.lowered.identity();
    let mut writer = CanonicalWriter::new("antecedent.assessed_bridge.target_posterior.question.v1");
    writer.bytes(child.as_bytes()).bytes(&antecedent_io::to_cbor(&premises)?).word(seed)
        .text("normal_line_reference_prior_variance_inverse_v1");
    let query = writer.finish();
    let mut scientific = assessed.lowered.scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional { kind: FunctionalKind::PolicyValue, outcome: None, query };
    scientific.plan_id = Arc::from("assessed_bridge_target_posterior_decision.v1");
    scientific.graph_class = "independent_target_normal_trial_posterior";
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = assessed.lowered.evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::ChainSource, digest: Some(*child.as_bytes()) });
    evidence.sources = Arc::from(sources);
    let mut operators = assessed.lowered.physical().operators().to_vec();
    operators.push(PhysicalOperator::declare(PhysicalEffect::Fit).build()
        .map_err(|_| AssessedBridgeRobustDecisionError::Invalid("target posterior cannot be lowered"))?);
    let mut statistical = assessed.lowered.statistical().clone();
    statistical.realization = Realization::NestedFit {
        producer_program: child,
        method: Arc::from("independent_target_normal_reference_posterior_retained_source_innovation_v1"),
        law: NestedFitLaw::Predictive,
    };
    let mut models = statistical.models.to_vec();
    models.push(ModelSlot { role: ModelRole::Outcome,
        family: Arc::from("target_normal_linear_fixed_design_reference_variance_inverse"),
        inputs: Arc::from([]), output: None });
    statistical.models = Arc::from(models);
    let prior = antecedent_core::Prior::new("target_line_reference_prior_inverse_variance")
        .map_err(|_| AssessedBridgeRobustDecisionError::Invalid("target posterior prior identity is invalid"))?;
    statistical.posterior = Some(PosteriorPlan { algorithm: PosteriorAlgorithm::Conjugate,
        draws: assessed.bridge.source.n_draws(), prior });
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::AssessedBridgeTransportDecision, scientific, identification,
        statistical, evidence,
        physical: PhysicalLayer { operators: Arc::from(operators), design: assessed.lowered.physical().design() },
        claim: ClaimDemand { uncertainty: UncertaintyReading::Predictive, kind: Some(ClaimKind::PolicyValue) },
    }.finish());
    Ok(CompiledAssessedBridgePosteriorDecision { assessed, premises, seed, lowered })
}

impl CompiledAssessedBridgePosteriorDecision {
    /// Outer compiler identity, including posterior prior and dependence premises.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }
    /// Inspect all retained layers before posterior sampling.
    #[must_use]
    pub fn plan_view(&self) -> PlanView { PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled) }

    fn posterior_plan(&self, result: &AssessedBridgeTransportDecisionResult, ctx: &ExecutionContext)
        -> Result<super::propagation_chain_decision::ChainDecisionPlan, AssessedBridgeRobustDecisionError> {
        let bridged = result.bridge.bridge.bridged();
        let source = bridged.source_column();
        let residuals = bridged.affine_columns().and_then(|columns| columns.residuals.as_ref())
            .ok_or(AssessedBridgeRobustDecisionError::Invalid("source predictive innovation is absent"))?;
        if source.len() != residuals.len() { return Err(AssessedBridgeRobustDecisionError::Invalid("source rows and innovations differ")); }
        let first = result.assessment.grid.first().ok_or(AssessedBridgeRobustDecisionError::Invalid("validation grid absent"))?.surrogate_effect;
        let last = result.assessment.grid.last().ok_or(AssessedBridgeRobustDecisionError::Invalid("validation grid absent"))?.surrogate_effect;
        if source.iter().any(|x| !x.is_finite() || *x < first || *x > last) {
            return Err(AssessedBridgeRobustDecisionError::Invalid("source draws leave validation overlap"));
        }
        let mut writer = CanonicalWriter::new("antecedent.assessed_bridge.target_posterior.seed.v1");
        writer.bytes(self.lowered.identity().as_bytes()).word(self.seed);
        let digest = writer.finish();
        let mut seed_bytes = [0u8; 8]; seed_bytes.copy_from_slice(&digest[..8]);
        let target = self.assessed.assessment.target_line_posterior(source.len(), u64::from_le_bytes(seed_bytes), ctx)
            .map_err(AssessedBridgeTransportDecisionError::from)?;
        let intercept: Vec<_> = target.iter().map(|draw| draw.intercept).collect();
        let slope: Vec<_> = target.iter().map(|draw| draw.slope).collect();
        let alignment = format!("target-posterior:{}", self.lowered.identity().to_hex());
        let draft = PropagationChain::draft(TableConstruction::JointDraws, ChainReading::Predictive,
            "target_terminal", vec![
                ChainSourceWire::draws("source", LawMeaning::Predictive, &alignment, &source)
                    .with_factor(SharedFactorKind::Model, "retained_source_predictive_pair"),
                ChainSourceWire::draws("target_slope", LawMeaning::Posterior, &alignment, &slope)
                    .with_factor(SharedFactorKind::Parameter, "independent_target_line_posterior"),
                ChainSourceWire::draws("target_intercept", LawMeaning::Posterior, &alignment, &intercept)
                    .with_factor(SharedFactorKind::Parameter, "independent_target_line_posterior"),
                ChainSourceWire::draws("source_innovation", LawMeaning::Innovation, &alignment, residuals)
                    .with_factor(SharedFactorKind::Model, "retained_source_predictive_pair"),
            ], vec![ChainStageWire::new("target_terminal", StageKind::PredictiveMapping,
                &["source", "target_slope", "target_intercept", "source_innovation"],
                ChainOpWire::expression(vec![ChainExprTermWire::input(0), ChainExprTermWire::input(1),
                    ChainExprTermWire::Multiply, ChainExprTermWire::input(2), ChainExprTermWire::Add,
                    ChainExprTermWire::input(3), ChainExprTermWire::Add]))],
        ).with_output_quantity(ScientificQuantityWire::from(&self.assessed.bridge.bridge.contract().target.quantity));
        Ok(compile_chain_decision(draft, self.assessed.bridge.decision.clone(), ctx)?)
    }

    /// Fit target posterior and evaluate nonlinear utility on aligned predictive rows.
    ///
    /// # Errors
    /// Changed seed, improper posterior, unsupported overlap or nested refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<AssessedBridgePosteriorDecisionResult, AssessedBridgeRobustDecisionError> {
        if ctx.rng.master_seed() != self.seed { return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior seed changed")); }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let assessed = self.assessed.execute(&scoped)?;
        let decision = self.posterior_plan(&assessed, &scoped)?.execute(&scoped)?;
        Ok(AssessedBridgePosteriorDecisionResult { assessed, decision, lowered_program_identity: self.lowered.identity(), calibration: "not_measured" })
    }

    /// Seal source-bound child artifacts and the posterior predictive decision.
    ///
    /// # Errors
    /// Changed result, failed replay or oversized artifact.
    pub fn export(&self, result: &AssessedBridgePosteriorDecisionResult, artifact_id: &str, ctx: &ExecutionContext)
        -> Result<Vec<u8>, AssessedBridgeRobustDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity() || result.calibration != "not_measured"
            || result.assessed.assessment != live.assessed.assessment
            || result.decision.chain.identity() != live.decision.chain.identity() || result.decision.decision != live.decision.decision {
            return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior decision differs on replay"));
        }
        let chain = self.posterior_plan(&live.assessed, ctx)?;
        let bytes = antecedent_io::to_cbor(&AssessedBridgePosteriorWire {
            version: 1, feature: "assessed_bridge_target_posterior_decision_v1".into(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(), premises: self.premises.clone(),
            assessed_identity: *self.assessed.lowered.identity().as_bytes(),
            assessed_artifact: self.assessed.export(&live.assessed, artifact_id, ctx)?,
            chain_identity: *chain.lowered().identity().as_bytes(), chain_artifact: chain.export(&live.decision, artifact_id)?,
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES { return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior artifact exceeds its bound")); }
        Ok(bytes)
    }
}

/// Independently recompile the raw validation and bridge children, then refit
/// the target posterior and replay the complete predictive decision.
///
/// # Errors
/// Changed feature, seed, child, compiled identity, posterior or decision.
pub fn consume_assessed_bridge_posterior_decision(bytes: &[u8], expected: SemanticDigest, ctx: &ExecutionContext)
    -> Result<AssessedBridgePosteriorDecisionResult, AssessedBridgeRobustDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES { return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior artifact exceeds its bound")); }
    let wire: AssessedBridgePosteriorWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.feature != "assessed_bridge_target_posterior_decision_v1"
        || wire.program_identity != *expected.as_bytes() || wire.seed != ctx.rng.master_seed() {
        return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior feature, seed or identity differs"));
    }
    let original = consume_assessed_bridge_transport_decision(&wire.assessed_artifact,
        SemanticDigest::from_bytes(wire.assessed_identity), ctx)?;
    let child: AssessedBridgeDecisionWire = antecedent_io::from_cbor(&wire.assessed_artifact)?;
    let (assessment, _) = recompile_bridge_transport_premise_assessment(&child.assessment_artifact,
        SemanticDigest::from_bytes(child.assessment_identity), ctx).map_err(AssessedBridgeTransportDecisionError::from)?;
    let (bridge, _) = recompile_bridge_transport_decision(&child.bridge_artifact,
        SemanticDigest::from_bytes(child.bridge_identity), ctx).map_err(AssessedBridgeTransportDecisionError::from)?;
    let assessed = compile_assessed_bridge_transport_decision(assessment, bridge)?;
    let plan = compile_assessed_bridge_posterior_decision(assessed, wire.premises, ctx)?;
    if plan.lowered.identity() != expected { return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior compiler differs")); }
    let live = plan.execute(ctx)?;
    if live.assessed.lowered_program_identity != original.lowered_program_identity
        || live.assessed.assessment != original.assessment
        || live.assessed.bridge.lowered_program_identity != original.bridge.lowered_program_identity
        || live.assessed.bridge.bridge.inspect() != original.bridge.bridge.inspect()
        || live.assessed.bridge.decision.chain.identity() != original.bridge.decision.chain.identity()
        || live.assessed.bridge.decision.decision != original.bridge.decision.decision
        || live.decision.lowered_program_identity != SemanticDigest::from_bytes(wire.chain_identity) {
        return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior children differ"));
    }
    let replay = consume_chain_decision(&wire.chain_artifact, Some(live.decision.lowered_program_identity), ctx)?;
    if replay.chain.identity() != live.decision.chain.identity() || replay.decision != live.decision.decision {
        return Err(AssessedBridgeRobustDecisionError::Invalid("target posterior utility differs"));
    }
    Ok(live)
}

/// A prospective exact full-route calibration coordinate, not an earned record.
pub struct AssessedBridgePosteriorCalibrationCandidate {
    /// Complete posterior predictive decision execution.
    pub result: AssessedBridgePosteriorDecisionResult,
    /// Source-bound method and validation-premise coordinate.
    pub basis: antecedent_core::CalibrationBasis,
}

impl CompiledAssessedBridgePosteriorDecision {
    /// Form a prospective unmeasured reliability candidate.
    ///
    /// # Errors
    /// Full-route execution failure.
    pub fn calibration_candidate(&self, ctx: &ExecutionContext)
        -> Result<AssessedBridgePosteriorCalibrationCandidate, AssessedBridgeRobustDecisionError> {
        let result = self.execute(ctx)?;
        let functional = format!("predictive_expected_regret:assessed={}:design={}:source_draws={}:target_trials={}",
            self.assessed.lowered.identity().to_hex(), self.premises.validation_design,
            self.assessed.bridge.source.n_draws(), self.assessed.assessment.target_trial_count());
        let basis = antecedent_core::CalibrationBasis::new([
            "PolicyValue", "independent_target_validation", "fixed", "tabular", "Bayesian",
            "assessed_bridge_target_posterior_decision_v1", "none", "joint_normal_line_reference_posterior",
            "independent_target_transferred_source_innovation", "reference_variance_inverse", functional.as_str(),
        ].map(Arc::from), 0.95, Arc::from("point"), self.assessed.bridge.source.n_draws() as u64,
            None, None, 0.0).with_validation("none");
        Ok(AssessedBridgePosteriorCalibrationCandidate { result, basis })
    }
}
