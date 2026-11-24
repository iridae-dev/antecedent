//! Fitted source-target effect posterior and target decision in one compiler program.
//!
//! The transported quantity is the target *average treatment effect*, not a
//! future outcome. The exact conjugate fit retains the source/target parameter
//! coupling, checks prior-data ownership and target support, and supplies the
//! effect posterior to the ordinary chain decision compiler. No population
//! outcome risk or measured decision reliability is inferred from these draws.

use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, LikelihoodModel, OrdinaryBayesianLikelihood,
    PhysicalEffect, PhysicalOperator, Prior, SemanticDigest, TransportQuery, VariableId,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{DecisionContract, DecisionContractError};
use antecedent_estimate::joint_bayesian_transport::{
    JointTransportFit, JointTransportModel, JointTransportOptions, SourceData, TargetData,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::joint_bayesian_transport_artifact::{
    JointBayesianArtifactWire, JointBayesianConsumeLimits, JointBayesianInputWire,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::{
    AdmgWire, IoError, TransportQueryWire, admg_from_wire, admg_to_wire, transport_query_from_wire,
    transport_query_to_wire,
};
use serde::{Deserialize, Serialize};

use super::program_handle::{CompiledProgramHandle, gate_program};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision, consume_chain_decision,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram,
};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "joint_transport_effect_decision_v1";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// The complete structural and numerical declaration. The quantity must name
/// the target average effect `E[Y|do(A=1)] - E[Y|do(A=0)]` on the raw outcome id.
#[derive(Clone, Debug)]
pub struct JointTransportEffectDecisionRequest {
    /// Fixed selection diagram used for structural identification.
    pub diagram: SelectionDiagram,
    /// Binary treatment response transport query and evidence catalog.
    pub query: TransportQuery,
    /// Shared and source-varying Gaussian mechanism with priors and overlap rule.
    pub model: JointTransportModel,
    /// Independent source trials with observation identities.
    pub sources: Vec<SourceData>,
    /// Target covariate sample with observation identity.
    pub target: TargetData,
    /// Exact conjugate draw count and seed.
    pub options: JointTransportOptions,
    /// Target average-effect coordinate; a parameter functional, not outcome risk.
    pub quantity: ScientificQuantity,
    /// Decision utility over that exact effect coordinate.
    pub decision: DecisionContract,
}

/// Refusal of the structural, fitting, decision or portable replay contract.
#[derive(Debug, thiserror::Error)]
pub enum JointTransportEffectDecisionError {
    /// A premise or replay identity is invalid.
    #[error("joint transport effect decision: {0}")]
    Invalid(&'static str),
    /// Structural identification failed.
    #[error("joint transport effect identification: {0}")]
    Identify(String),
    /// Decision contract failed.
    #[error("joint transport effect contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Posterior effect chain or decision failed.
    #[error("joint transport effect chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Producer or combined artifact failed.
    #[error("joint transport effect artifact: {0}")]
    Io(#[from] IoError),
}

/// Retained compiler program, before any source or target fit runs.
#[derive(Clone, Debug)]
pub struct CompiledJointTransportEffectDecision {
    request: JointTransportEffectDecisionRequest,
    identification: TransportIdentification,
    input: JointBayesianInputWire,
    graph: AdmgWire,
    query: TransportQueryWire,
    lowered: Arc<LoweredProgram>,
}

/// Fitted, aligned effect posterior and utility evaluated on the target effect.
#[derive(Clone, Debug)]
pub struct JointTransportEffectDecisionResult {
    /// The exact conjugate fit, including source-target coupled effects.
    pub fit: JointTransportFit,
    /// Independently replayable producer fit artifact.
    pub producer: JointBayesianArtifactWire,
    /// Effect posterior chain and decision result.
    pub decision: ChainDecisionResult,
    /// Outer six-layer compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// This composed decision has no earned coverage or reliability record.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JointTransportEffectDecisionWire {
    version: u32,
    required_feature: String,
    graph: AdmgWire,
    selections: Vec<u32>,
    query: TransportQueryWire,
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    producer_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
    calibration: String,
}

fn check_coordinate(
    request: &JointTransportEffectDecisionRequest,
) -> Result<VariableId, JointTransportEffectDecisionError> {
    let Some((treatment, outcome)) = request.query.response.functional.primary_pair() else {
        return Err(JointTransportEffectDecisionError::Invalid(
            "transport query has no treatment/outcome pair",
        ));
    };
    let quantity = &request.quantity;
    if quantity.variable_id != outcome.raw().to_string()
        || quantity.role != QuantityRole::Outcome
        || quantity.population_id != request.query.target_population.as_ref()
        || quantity.population_id != request.decision.target_population
        || quantity.regime_id != format!("do({}=1)-do({}=0)", treatment.raw(), treatment.raw())
        || quantity.functional_id != "average_effect"
        || quantity.horizon != 0
        || quantity.horizon != request.decision.horizon
        || !quantity.conditioning.is_empty()
        || quantity.transform_id != "identity"
        || request.decision.actions.is_empty()
        || request.decision.actions.iter().any(|action| {
            action.inputs.len() != 1 || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(JointTransportEffectDecisionError::Invalid(
            "decision must read the exact target binary average-effect coordinate",
        ));
    }
    Ok(outcome)
}

/// Compile raw selection, model, prior, unit rows, target population, overlap,
/// posterior seed, effect coordinate and utility before any fit executes.
///
/// # Errors
/// A mismatched effect coordinate, unsupported structural query, malformed
/// contract or encoding refusal.
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn compile_joint_transport_effect_decision(
    request: JointTransportEffectDecisionRequest,
) -> Result<CompiledJointTransportEffectDecision, JointTransportEffectDecisionError> {
    request.decision.validate()?;
    let outcome = check_coordinate(&request)?;
    let graph = admg_to_wire(request.diagram.causal_graph())?;
    let query = transport_query_to_wire(&request.query)?;
    let input = JointBayesianInputWire::from_declaration(
        &request.model,
        &request.sources,
        &request.target,
        request.options,
    );
    let identification = TransportIdentifier::new()
        .identify(&request.diagram, &request.query)
        .map_err(|error| JointTransportEffectDecisionError::Identify(error.to_string()))?;
    let contract_identity = request.decision.identity()?;
    let selections: Vec<u32> =
        request.diagram.selection_targets().iter().map(|v| v.raw()).collect();
    let input_bytes = antecedent_io::to_cbor(&input)?;
    let graph_bytes = antecedent_io::to_cbor(&graph)?;
    let query_bytes = antecedent_io::to_cbor(&query)?;
    let coordinate_bytes =
        antecedent_io::to_cbor(&ScientificQuantityWire::from(&request.quantity))?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.joint_transport_effect_decision.inputs.v1",
        &input_bytes,
    );
    let mut writer = CanonicalWriter::new("antecedent.joint_transport_effect_decision.question.v1");
    writer
        .bytes(&graph_bytes)
        .words(&selections.iter().map(|v| u64::from(*v)).collect::<Vec<_>>())
        .bytes(&query_bytes)
        .bytes(&input_bytes)
        .bytes(&coordinate_bytes)
        .text(&contract_identity);
    let question = writer.finish();
    let rows = request
        .sources
        .iter()
        .map(|s| s.outcome.len() as u64)
        .sum::<u64>()
        .saturating_add(request.target.rows as u64);
    let design = DesignShape::quoted(rows, request.model.features.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(outcome),
            query: question,
        },
        plan_id: Arc::from("joint_transport_effect_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "fixed_selection_dag_binary_trial",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification_layer = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let prior = Prior::new(format!(
        "joint_transport_gaussian:{}",
        SemanticDigest::from_bytes(source_digest).to_hex()
    ))
    .map_err(|_| JointTransportEffectDecisionError::Invalid("invalid joint prior identity"))?;
    let likelihood =
        LikelihoodModel::new("joint_transport_known_variance_normal").map_err(|_| {
            JointTransportEffectDecisionError::Invalid("invalid joint likelihood identity")
        })?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Bayesian(OrdinaryBayesianLikelihood::new(
            likelihood,
            prior.clone(),
        )),
        models: Arc::from([ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from("joint_source_target_effect"),
            inputs: Arc::from([]),
            output: Some(outcome),
        }]),
        folds: None,
        resampling: None,
        posterior: Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::Conjugate,
            draws: request.options.draws,
            prior,
        }),
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(question) },
        ]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: request.decision.actions.len(),
            constraints: request.decision.constraints.len(),
            chance_threshold: request
                .decision
                .constraints
                .iter()
                .map(|c| c.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(request.decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("joint_transport_target_effect_posterior_v1")),
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(input_bytes.len() as u64))
                .build()
                .map_err(|_| {
                    JointTransportEffectDecisionError::Invalid("invalid producer read operator")
                })?,
            PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| {
                JointTransportEffectDecisionError::Invalid("invalid producer fit operator")
            })?,
            PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(|_| {
                JointTransportEffectDecisionError::Invalid("invalid target transport operator")
            })?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::JointTransportEffectDecision,
            scientific,
            identification: identification_layer,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Posterior,
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    );
    Ok(CompiledJointTransportEffectDecision {
        request,
        identification,
        input,
        graph,
        query,
        lowered,
    })
}

fn effect_chain(
    fit: &JointTransportFit,
    quantity: &ScientificQuantity,
) -> Result<PropagationChainRecordWire, JointTransportEffectDecisionError> {
    let Some(column) = fit.draws.column("effect.target") else {
        return Err(JointTransportEffectDecisionError::Invalid("fit lost target effect draws"));
    };
    let values = fit.draws.coordinate(column);
    let source = ChainSourceWire::draws(
        "target.effect.posterior",
        LawMeaning::Posterior,
        &fit.draws.rng_id,
        &values,
    )
    .with_factor(SharedFactorKind::Parameter, &fit.model_identity);
    let stage = ChainStageWire::new(
        "target.effect",
        StageKind::DerivedQuantity,
        &["target.effect.posterior"],
        ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
    );
    Ok(PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Posterior,
        "target.effect",
        vec![source],
        vec![stage],
    )
    .with_output_quantity(ScientificQuantityWire::from(quantity)))
}

impl CompiledProgramHandle for CompiledJointTransportEffectDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledJointTransportEffectDecision {
    /// All six compiler layers of the fitted producer and decision.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the retained program without fitting or scoring.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Fit source and target data under the retained program, then compile and
    /// execute the target effect posterior decision on aligned rows.
    ///
    /// # Errors
    /// A fit, overlap, ownership, effect law or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointTransportEffectDecisionResult, JointTransportEffectDecisionError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)
                .map_err(|refusal| JointTransportEffectDecisionError::Invalid(refusal.message()))?;
        }
        let (producer, fit) = JointBayesianArtifactWire::build(
            &self.identification,
            &self.request.model,
            &self.request.sources,
            &self.request.target,
            self.request.options,
            &scoped,
        )?;
        if JointBayesianInputWire::from_artifact(&producer) != self.input {
            return Err(JointTransportEffectDecisionError::Invalid(
                "producer changed its compiled inputs",
            ));
        }
        let draft = effect_chain(&fit, &self.request.quantity)?;
        let decision =
            compile_chain_decision(draft, self.request.decision.clone(), ctx)?.execute(ctx)?;
        Ok(JointTransportEffectDecisionResult {
            fit,
            producer,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Export a combined receipt after reexecuting the exact compiled route.
    ///
    /// # Errors
    /// Any changed producer, decision, identity or artifact payload.
    pub fn export(
        &self,
        result: &JointTransportEffectDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointTransportEffectDecisionError> {
        let replay = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.fit != replay.fit
            || result.decision.decision != replay.decision.decision
            || result.decision.chain.identity() != replay.decision.chain.identity()
            || result.producer.export()? != replay.producer.export()?
        {
            return Err(JointTransportEffectDecisionError::Invalid(
                "fit or decision changed on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.request.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&replay.fit, &self.request.quantity)?,
            self.request.decision.clone(),
            ctx,
        )?;
        let wire = JointTransportEffectDecisionWire {
            version: 1,
            required_feature: FEATURE.into(),
            graph: self.graph.clone(),
            selections: self.request.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: self.query.clone(),
            quantity: ScientificQuantityWire::from(&self.request.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            producer_artifact: replay.producer.export()?,
            chain_artifact: chain.export(&result.decision, artifact_id)?,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(JointTransportEffectDecisionError::Invalid(
                "combined artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Reidentify the raw diagram, refit the stored source and target units,
/// recompile their effect posterior decision and compare the complete result.
/// The caller must retain the outer program identity outside the artifact.
///
/// # Errors
/// A foreign feature, changed structural certificate, input, fit, law,
/// decision or program identity refuses.
pub fn consume_joint_transport_effect_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<JointTransportEffectDecisionResult, JointTransportEffectDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(JointTransportEffectDecisionError::Invalid(
            "combined artifact exceeds its byte bound",
        ));
    }
    let wire: JointTransportEffectDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.calibration != "not_measured"
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(JointTransportEffectDecisionError::Invalid(
            "combined feature or identity differs",
        ));
    }
    let (producer, fit) = JointBayesianArtifactWire::consume_with_limits(
        &wire.producer_artifact,
        JointBayesianConsumeLimits::default(),
        ctx,
    )?;
    let graph = admg_from_wire(&wire.graph)?;
    let diagram = SelectionDiagram::try_new(
        graph,
        wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
    )
    .map_err(|_| JointTransportEffectDecisionError::Invalid("invalid stored selection diagram"))?;
    let query = transport_query_from_wire(&wire.query)?;
    let checked = TransportIdentifier::new()
        .identify(&diagram, &query)
        .map_err(|error| JointTransportEffectDecisionError::Identify(error.to_string()))?;
    if antecedent_io::transport_identification_to_wire(&checked) != producer.identification {
        return Err(JointTransportEffectDecisionError::Invalid(
            "stored fit certificate differs from raw graph identification",
        ));
    }
    let input = JointBayesianInputWire::from_artifact(&producer);
    let (model, sources, target, options) = input.to_declaration();
    let quantity: ScientificQuantity =
        wire.quantity.try_into().map_err(JointTransportEffectDecisionError::Invalid)?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let plan = compile_joint_transport_effect_decision(JointTransportEffectDecisionRequest {
        diagram,
        query,
        model,
        sources,
        target,
        options,
        quantity,
        decision: contract.contract().clone(),
    })?;
    if plan.lowered.identity() != expected_program {
        return Err(JointTransportEffectDecisionError::Invalid("recompiled program differs"));
    }
    let live = plan.execute(ctx)?;
    if live.fit != fit || live.producer.export()? != producer.export()? {
        return Err(JointTransportEffectDecisionError::Invalid(
            "producer fit differs on complete replay",
        ));
    }
    let chain = consume_chain_decision(
        &wire.chain_artifact,
        Some(live.decision.lowered_program_identity),
        ctx,
    )?;
    if chain.decision != live.decision.decision
        || chain.chain.identity() != live.decision.chain.identity()
    {
        return Err(JointTransportEffectDecisionError::Invalid(
            "target effect utility differs on replay",
        ));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_estimate::joint_bayesian_transport::{
        DataIdentity, GaussianPrior, JointPriors, PriorProvenance, SourceDependence, SourceSharing,
        TransportGraphClass, VaryingBlock,
    };

    // Known-truth fixture; deliberately unearned until builds and reference
    // assertions are permitted. The effect is exactly 2 at every target unit.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn fitted_target_effect_and_utility_replay_against_known_truth() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let source_node = graph.dense_of_static(treatment).unwrap();
        let target_node = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(source_node, target_node).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [treatment],
        );
        let quantity = ScientificQuantity {
            variable_id: "2".into(),
            variable_name: "outcome effect".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(0=1)-do(0=0)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0),
                        UtilityExpr::Const(1.0),
                    ),
                },
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let rows = 40;
        let source = SourceData {
            id: "trial-a".into(),
            identity: DataIdentity {
                snapshot_digest: "trial-snapshot".into(),
                datum_ids: (0..rows).map(|i| format!("trial-{i}")).collect(),
            },
            treatment: (0..rows).map(|i| i % 2 == 0).collect(),
            outcome: (0..rows).map(|i| 3.0 + if i % 2 == 0 { 2.0 } else { 0.0 }).collect(),
            covariates: Vec::new(),
            noise_variance: 0.1,
        };
        let target = TargetData {
            identity: DataIdentity {
                snapshot_digest: "target-snapshot".into(),
                datum_ids: (0..rows).map(|i| format!("target-{i}")).collect(),
            },
            rows,
            covariates: Vec::new(),
        };
        let model = JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features: Vec::new(),
            varying: VaryingBlock::Intercept,
            sharing: SourceSharing::IndependentVaryingBlocks,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        };
        let ctx = ExecutionContext::for_tests(91);
        let plan = compile_joint_transport_effect_decision(JointTransportEffectDecisionRequest {
            diagram,
            query,
            model,
            sources: vec![source],
            target,
            options: JointTransportOptions { draws: 1024, seed: 91 },
            quantity,
            decision,
        })
        .unwrap();
        let result = plan.execute(&ctx).unwrap();
        assert!((result.fit.target_effect_mean - 2.0).abs() < 0.05);
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "known-truth-joint-transport", &ctx).unwrap();
        let replay =
            consume_joint_transport_effect_decision(&bytes, result.lowered_program_identity, &ctx)
                .unwrap();
        assert_eq!(replay.fit, result.fit);
        assert_eq!(replay.decision.decision, result.decision.decision);
    }

    /// The known-truth declaration above, rebuilt for each identity the site test compares.
    #[allow(clippy::too_many_lines, reason = "one declaration read top to bottom")]
    fn effect_request() -> JointTransportEffectDecisionRequest {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let source_node = graph.dense_of_static(treatment).unwrap();
        let target_node = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(source_node, target_node).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [treatment],
        );
        let quantity = ScientificQuantity {
            variable_id: "2".into(),
            variable_name: "outcome effect".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(0=1)-do(0=0)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0),
                        UtilityExpr::Const(1.0),
                    ),
                },
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let rows = 40;
        let source = SourceData {
            id: "trial-a".into(),
            identity: DataIdentity {
                snapshot_digest: "trial-snapshot".into(),
                datum_ids: (0..rows).map(|i| format!("trial-{i}")).collect(),
            },
            treatment: (0..rows).map(|i| i % 2 == 0).collect(),
            outcome: (0..rows).map(|i| 3.0 + if i % 2 == 0 { 2.0 } else { 0.0 }).collect(),
            covariates: Vec::new(),
            noise_variance: 0.1,
        };
        let target = TargetData {
            identity: DataIdentity {
                snapshot_digest: "target-snapshot".into(),
                datum_ids: (0..rows).map(|i| format!("target-{i}")).collect(),
            },
            rows,
            covariates: Vec::new(),
        };
        let model = JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features: Vec::new(),
            varying: VaryingBlock::Intercept,
            sharing: SourceSharing::IndependentVaryingBlocks,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        };
        JointTransportEffectDecisionRequest {
            diagram,
            query,
            model,
            sources: vec![source],
            target,
            options: JointTransportOptions { draws: 1024, seed: 91 },
            quantity,
            decision,
        }
    }

    #[test]
    fn site_joint_transport_effect_decision_question_binds_rows_options_and_contract() {
        let question = |request: JointTransportEffectDecisionRequest| {
            let plan = compile_joint_transport_effect_decision(request).ok()?;
            match plan.lowered().scientific().target() {
                ProblemTarget::QueryFunctional { query, .. } => Some(*query),
                _ => None,
            }
        };
        let mut seed = effect_request();
        seed.options.seed = 92;
        let mut draws = effect_request();
        draws.options.draws = 2048;
        // Source rows are positional: the posterior schedule and the datum identities read them
        // in declared order.
        let mut swapped = effect_request();
        swapped.sources[0].treatment.swap(0, 1);
        swapped.sources[0].outcome.swap(0, 1);
        let mut outcome = effect_request();
        outcome.sources[0].outcome[0] = 5.5;
        let mut units = effect_request();
        units.decision.utility_units = "points".into();
        let mut target = effect_request();
        target.target.identity.snapshot_digest = "another-target-snapshot".into();
        crate::identity_laws::all_distinct(&[
            question(effect_request()),
            question(seed),
            question(draws),
            question(swapped),
            question(outcome),
            question(units),
            question(target),
        ]);
    }
}
