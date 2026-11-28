//! A decision whose paired intervention outcomes come from one cyclic SCM.
//!
//! The compiler checks the complete model, hard-clamp worlds, shared
//! disturbance rows, output coordinates and decision contract before solving.
//! Execution then derives each joint output column from the same row of the
//! specified model in every world. The source is an empirical disturbance
//! scenario law conditional on this model and table, not a population
//! predictive or calibrated causal effect. The portable consumer replays the
//! cyclic model and the chain decision independently.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_io::IoError;
use antecedent_io::cyclic_system_artifact::{
    CyclicArtifactError, CyclicExecutionReport, CyclicExecutionRequest,
    consume_cyclic_artifact_with_request,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_action_world_chain_decision,
    consume_chain_decision,
};
use super::{PreparedStudy, Study};
use crate::analysis::cyclic::{CyclicPreparedState, gate_decision};
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, EvidenceRole, EvidenceSourceRef,
    FunctionalKind, LoweredProgram, ProblemTarget, ProgramCell, StandaloneCompositionLoweredParts,
    UncertaintyReading,
};

const FEATURE: &str = "cyclic_derived_action_world_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;
/// Question identity: the source program, the contract, the outputs and the disturbance rows.
const QUESTION_TAG: &str = "antecedent.cyclic_action_world_decision.question.v2";
/// The released question identity, without the disturbance rows. Reproduced only to verify a
/// handoff exported under it.
const QUESTION_TAG_V1: &str = "antecedent.cyclic_action_world_decision.question.v1";

/// One intervention world and one endogenous model coordinate, published as a
/// separate scientific quantity in the joint decision law.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicDecisionOutput {
    /// Index in the cyclic request's hard-clamp worlds.
    pub world: usize,
    /// Dense index of the endogenous equation read in every disturbance row.
    pub equation: usize,
    /// Exact output coordinate, including world regime and population.
    pub quantity: ScientificQuantityWire,
}

/// Refusal of model/world binding, an inner compiler or independent replay.
#[derive(Debug, thiserror::Error)]
pub enum CyclicActionDecisionError {
    /// World, quantity or paired-row semantics are unsupported.
    ///
    /// A support-gate refusal of the route's unit (`gate_decision`) carries
    /// the gate's `reason=not_supported: …` message here and keeps that class: it displays
    /// and reports the `cyclic_decision.not_supported` detail and the `not_supported` code.
    #[error("{}: {}", crate::analysis::cyclic::decision_detail(.0), .0)]
    Invalid(&'static str),
    /// A cyclic model or solve was refused.
    #[error("cyclic action decision source: {0}")]
    Cyclic(#[from] CyclicArtifactError),
    /// Decision contract was refused.
    #[error("cyclic action decision contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// The joint chain or utility was refused.
    #[error("cyclic action decision chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or decoding was refused.
    #[error("cyclic action decision artifact: {0}")]
    Io(#[from] IoError),
}

impl CyclicActionDecisionError {
    /// Stable refusal detail, when the refusal originated in this route rather than in a
    /// wrapped model, contract, chain or artifact refusal.
    #[must_use]
    pub fn detail(&self) -> Option<&'static str> {
        match self {
            Self::Invalid(message) => Some(crate::analysis::cyclic::decision_detail(message)),
            _ => None,
        }
    }

    /// Stable top-level refusal code of [`Self::detail`].
    #[must_use]
    pub fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Invalid(message) => Some(crate::analysis::cyclic::invalid_code(message)),
            _ => None,
        }
    }
}

/// The model and decision are one six-layer program; numerical results are
/// still produced by their checked inner compiler programs.
#[derive(Clone, Debug)]
pub struct CompiledCyclicActionDecision {
    source: PreparedStudy<CyclicPreparedState>,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Complete paired model result and downstream utility at one program identity.
#[derive(Clone, Debug)]
pub struct CyclicActionDecisionResult {
    /// Original solved worlds, with their row order and solver certificates.
    pub cyclic: CyclicExecutionReport,
    /// Compiler-retained joint law and multi-action decision.
    pub decision: ChainDecisionResult,
    /// Identity of the complete model-to-decision compiler program.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CyclicActionDecisionWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    source_identity: String,
    cyclic_artifact: Vec<u8>,
    outputs: Vec<CyclicDecisionOutput>,
    contract_identity: String,
    contract_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
}

/// Compile paired action worlds from one checked cyclic structural model.
/// Every action must name its own distinct hard-clamp world and exactly one
/// declared outcome coordinate. Fitting is excluded from this first method:
/// the aligned variation is solely the explicit shared disturbance table.
///
/// # Errors
/// An unpaired world, repeated regime, mismatched coordinate, unsupported
/// action or cyclic compiler refusal.
pub fn compile_cyclic_action_world_decision(
    request: CyclicExecutionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicActionDecision, CyclicActionDecisionError> {
    compile_cyclic_action_world_decision_encoded(
        request,
        outputs,
        contract,
        QuestionEncoding::Current,
        ctx,
    )
}

/// Which encoding of the question identity a compile writes. Compiles write
/// [`Self::Current`]; a consumer whose recompilation differs from the stored program identity
/// recompiles under [`Self::V1`], the released encoding, so a handoff exported before the
/// disturbance rows were bound still verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuestionEncoding {
    /// The shared disturbance rows are part of the question.
    Current,
    /// The released encoding: the disturbance rows are not bound.
    V1,
}

#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
fn compile_cyclic_action_world_decision_encoded(
    request: CyclicExecutionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    encoding: QuestionEncoding,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicActionDecision, CyclicActionDecisionError> {
    contract.validate()?;
    if request.fit.is_some()
        || request.factual.is_some()
        || request.noise.len() < 2
        || outputs.len() < 2
        || outputs.len() != request.worlds.len()
        || outputs.len() != contract.actions.len()
        || request.worlds.iter().any(Vec::is_empty)
    {
        return Err(CyclicActionDecisionError::Invalid(
            "this route requires specified mechanisms, paired disturbance rows and distinct hard-clamp worlds",
        ));
    }
    let mut seen_worlds = BTreeSet::new();
    let mut seen_regimes = BTreeSet::new();
    for output in &outputs {
        let equation =
            request.model.equations.get(output.equation).ok_or(
                CyclicActionDecisionError::Invalid("output equation is outside the model"),
            )?;
        if !seen_worlds.insert(output.world)
            || request.worlds.get(output.world).is_none()
            || equation.id != output.quantity.variable_id
            || output.quantity.role != "outcome"
            || output.quantity.population_id != contract.target_population
            || output.quantity.horizon != contract.horizon
            || !seen_regimes.insert(output.quantity.regime_id.as_str())
        {
            return Err(CyclicActionDecisionError::Invalid(
                "every world needs a distinct regime and a matching model outcome coordinate",
            ));
        }
    }
    if request
        .worlds
        .iter()
        .enumerate()
        .any(|(index, world)| request.worlds[..index].contains(world))
    {
        return Err(CyclicActionDecisionError::Invalid("two action worlds have identical clamps"));
    }
    let mut action_regimes = BTreeSet::new();
    for action in &contract.actions {
        if !matches!(action.kind, ActionKind::Intervention | ActionKind::Regime)
            || action.inputs.len() != 1
            || !action_regimes.insert(action.inputs[0].regime_id.as_str())
            || !outputs.iter().any(|output| {
                let Ok(quantity) = ScientificQuantity::try_from(output.quantity.clone()) else {
                    return false;
                };
                action.inputs[0].require_same_coordinate(&quantity).is_ok()
            })
        {
            return Err(CyclicActionDecisionError::Invalid(
                "each action must read exactly one matching intervened outcome",
            ));
        }
    }
    if action_regimes != seen_regimes {
        return Err(CyclicActionDecisionError::Invalid(
            "the decision actions do not cover every distinct model world",
        ));
    }
    let noise = request.noise.clone();
    let source = Study::prepare_cyclic(request, ctx)?;
    let contract_identity = contract.identity()?;
    let outputs_bytes = antecedent_io::to_cbor(&outputs)?;
    let mut writer = CanonicalWriter::new(match encoding {
        QuestionEncoding::Current => QUESTION_TAG,
        QuestionEncoding::V1 => QUESTION_TAG_V1,
    });
    writer
        .bytes(source.lowered().identity().as_bytes())
        .text(&contract_identity)
        .bytes(&outputs_bytes);
    if encoding == QuestionEncoding::Current {
        // The decision is taken over these disturbance rows draw by draw, so a different row
        // set is a different question even where the model and worlds are the same.
        writer.word(noise.len() as u64);
        for row in &noise {
            writer.reals(row);
        }
    }
    let digest = writer.finish();
    let mut scientific = source.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PolicyValue,
        outcome: None,
        query: digest,
    };
    scientific.plan_id = Arc::from("cyclic_action_world_decision.v1");
    scientific.source_digest = Some(digest);
    let mut evidence = source.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::JointScenarios, digest: Some(digest) });
    evidence.sources = Arc::from(sources);
    evidence.composition = Composition::Chained { stages: 2 };
    evidence.decision = Some(DecisionProgram {
        carrier: CarrierDemand::PredictiveLaw,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: Some(Arc::from(contract_identity.as_str())),
        information_value: None,
        scenario_source: Some(Arc::from("cyclic_shared_disturbance_worlds_v1")),
    });
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::CyclicActionWorldDecision,
            scientific,
            identification: source.lowered().identification_layer().clone(),
            statistical: source.lowered().statistical().clone(),
            evidence,
            physical: source.lowered().physical().clone(),
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Predictive,
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    );
    let plan = CompiledCyclicActionDecision { source, outputs, contract, lowered };
    gate_decision(ctx, &plan)
        .map_err(|refusal| CyclicActionDecisionError::Invalid(refusal.message()))?;
    Ok(plan)
}

impl crate::analysis::program_handle::CompiledProgramHandle for CompiledCyclicActionDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledCyclicActionDecision {
    /// The complete model/world/decision program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect all compiler layers before solving or evaluating utility.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        crate::analysis::program_handle::CompiledProgramHandle::plan_view(self)
    }

    fn joint_chain(
        &self,
        report: &CyclicExecutionReport,
    ) -> Result<PropagationChainRecordWire, CyclicActionDecisionError> {
        if report.worlds.len() != self.source.request().worlds.len()
            || report.worlds.iter().any(|world| world.len() != self.source.request().noise.len())
        {
            return Err(CyclicActionDecisionError::Invalid(
                "cyclic execution lost paired disturbance rows",
            ));
        }
        let alignment = self.source.lowered().identity().to_hex();
        let mut sources = Vec::with_capacity(self.outputs.len());
        let mut stages = Vec::with_capacity(self.outputs.len());
        let mut joint = Vec::with_capacity(self.outputs.len());
        for (index, output) in self.outputs.iter().enumerate() {
            let values: Vec<f64> = report.worlds[output.world]
                .iter()
                .map(|row| {
                    row.values.get(output.equation).copied().ok_or(
                        CyclicActionDecisionError::Invalid("a solved world lost its model output"),
                    )
                })
                .collect::<Result<_, _>>()?;
            if report.worlds[output.world].iter().any(|row| {
                !row.globally_unique || !row.existence_certified || !row.numerical_converged
            }) {
                return Err(CyclicActionDecisionError::Invalid(
                    "every retained world row needs a unique certified equilibrium",
                ));
            }
            let source_id = format!("cyclic.world.{index}");
            let stage_id = format!("cyclic.outcome.{index}");
            sources.push(
                ChainSourceWire::draws(&source_id, LawMeaning::Predictive, &alignment, &values)
                    .with_factor(SharedFactorKind::Model, &alignment)
                    .with_factor(SharedFactorKind::Disturbance, &alignment),
            );
            stages.push(ChainStageWire::new(
                &stage_id,
                StageKind::DerivedQuantity,
                &[source_id.as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            ));
            joint.push(ChainJointOutputWire { stage: stage_id, quantity: output.quantity.clone() });
        }
        Ok(PropagationChainRecordWire::draft(
            TableConstruction::JointDraws,
            ChainReading::Predictive,
            &joint[0].stage,
            sources,
            stages,
        )
        .with_output_quantity(joint[0].quantity.clone())
        .with_joint_outputs(joint))
    }

    /// Solve all hard-clamp worlds on the same disturbance rows, then compile
    /// and evaluate the resulting paired law and utility.
    ///
    /// # Errors
    /// Uncertified equilibrium, missing row or inner compiler refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicActionDecisionResult, CyclicActionDecisionError> {
        gate_decision(ctx, self)
            .map_err(|refusal| CyclicActionDecisionError::Invalid(refusal.message()))?;
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let cyclic = self.source.estimate(&scoped)?;
        let draft = self.joint_chain(&cyclic)?;
        let decision =
            compile_action_world_chain_decision(draft, self.contract.clone(), ctx)?.execute(ctx)?;
        Ok(CyclicActionDecisionResult {
            cyclic,
            decision,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Export both executable sources only after checking the result against a
    /// fresh model solve and its derived joint decision.
    ///
    /// # Errors
    /// A changed result, an invalid artifact or an inner replay refusal.
    pub fn export(
        &self,
        result: &CyclicActionDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicActionDecisionError> {
        if result.lowered_program_identity != self.lowered.identity() {
            return Err(CyclicActionDecisionError::Invalid("outer program identity differs"));
        }
        let replay = self.execute(ctx)?;
        if result.cyclic != replay.cyclic
            || result.decision.chain.identity() != replay.decision.chain.identity()
            || result.decision.decision != replay.decision.decision
        {
            return Err(CyclicActionDecisionError::Invalid(
                "model-derived decision differs on replay",
            ));
        }
        let (cyclic, cyclic_artifact) = self.source.export(artifact_id, ctx)?;
        if cyclic != result.cyclic {
            return Err(CyclicActionDecisionError::Invalid("cyclic source changed during export"));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let inner = compile_action_world_chain_decision(
            self.joint_chain(&cyclic)?,
            self.contract.clone(),
            ctx,
        )?;
        let chain_artifact = inner.export(&result.decision, artifact_id)?;
        let wire = CyclicActionDecisionWire {
            version: 1,
            required_feature: FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(),
            source_identity: cyclic.identity,
            cyclic_artifact,
            outputs: self.outputs.clone(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CyclicActionDecisionError::Invalid("combined artifact exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently decode and replay the cyclic model and its derived joint
/// decision under an identity retained outside the artifact.
///
/// # Errors
/// A foreign feature, altered source or any model/decision replay mismatch.
pub fn consume_cyclic_action_world_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CyclicActionDecisionResult, CyclicActionDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(CyclicActionDecisionError::Invalid("combined artifact exceeds its bound"));
    }
    let wire: CyclicActionDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(CyclicActionDecisionError::Invalid(
            "combined artifact feature or identity differs",
        ));
    }
    let (request, original_cyclic) =
        consume_cyclic_artifact_with_request(&wire.cyclic_artifact, &wire.source_identity, ctx)?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.contract_artifact, &wire.contract_identity)?;
    // A handoff exported under the released question encoding recompiles under it.
    let compile = |encoding| {
        compile_cyclic_action_world_decision_encoded(
            request.clone(),
            wire.outputs.clone(),
            contract.contract().clone(),
            encoding,
            ctx,
        )
    };
    let mut plan = compile(QuestionEncoding::Current)?;
    if plan.lowered.identity() != expected_program {
        plan = compile(QuestionEncoding::V1)?;
    }
    if plan.lowered.identity() != expected_program {
        return Err(CyclicActionDecisionError::Invalid("recompiled model and decision differ"));
    }
    let result = plan.execute(ctx)?;
    if result.cyclic != original_cyclic {
        return Err(CyclicActionDecisionError::Invalid("model-derived worlds changed on replay"));
    }
    let inner = consume_chain_decision(
        &wire.chain_artifact,
        Some(result.decision.lowered_program_identity),
        ctx,
    )?;
    if inner.chain.identity() != result.decision.chain.identity()
        || inner.decision != result.decision.decision
    {
        return Err(CyclicActionDecisionError::Invalid("joint action decision changed on replay"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::QuantityRole;
    use antecedent_design::decision_contract::{
        DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_model::nonlinear_equilibrium::{
        Activation, Clamp, Equation, EquilibriumModel, SolvePolicy,
    };

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "wishlists".into(),
            variable_name: "wishlists".into(),
            role: QuantityRole::Outcome,
            units: "count".into(),
            population_id: "game".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn request() -> CyclicExecutionRequest {
        CyclicExecutionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "wishlists".into(),
                        intercept: 0.0,
                        weights: vec![0.1, 0.3],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "traffic".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                ],
                noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            },
            fit: None,
            worlds: vec![
                vec![Clamp { variable: 1, value: 1.0 }],
                vec![Clamp { variable: 1, value: 2.0 }],
            ],
            noise: vec![vec![-0.2, 0.0], vec![0.0, 0.0], vec![0.2, 0.0], vec![0.4, 0.0]],
            initial: vec![0.0, 0.0],
            policy: SolvePolicy::default(),
            factual: None,
        }
    }

    fn outputs() -> Vec<CyclicDecisionOutput> {
        ["ordinary", "promoted"]
            .iter()
            .enumerate()
            .map(|(world, regime)| CyclicDecisionOutput {
                world,
                equation: 0,
                quantity: ScientificQuantityWire::from(&quantity(regime)),
            })
            .collect()
    }

    fn contract() -> DecisionContract {
        DecisionContract {
            actions: ["ordinary", "promoted"]
                .iter()
                .map(|regime| DecisionAction {
                    id: (*regime).into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity(regime)],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "wishlist_count".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: Vec::new(),
            target_population: "game".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        }
    }

    #[test]
    fn paired_equilibrium_worlds_are_derived_and_replayed() {
        crate::evidence_unit!("capability", "cyclic.action_world_decisions");
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_cyclic_action_world_decision(request(), outputs(), contract(), &ctx)
            .expect("model and decision compiler");
        assert_eq!(plan.plan_view().route().as_str(), "CyclicActionWorldDecision");
        // The program reports the decision of the unit its route is bound to, and the plan view
        // reads the same decision.
        let handle: &dyn crate::analysis::program_handle::CompiledProgramHandle = &plan;
        let decision = handle.support();
        assert_eq!(
            plan.plan_view().unit_support().map(|unit| unit.status()),
            decision.as_ref().map(|decision| decision.status)
        );
        // The decision gate is inert until it is switched on; once on, it consults the same
        // decision, so a development build records the would-refuse of an unsupported unit.
        if !crate::analysis::cyclic::COMPILED_DECISION_GATE_LIVE {
            assert!(crate::analysis::cyclic::gate_decision(&ctx, &plan).is_ok());
        }
        let result = plan.execute(&ctx).expect("paired model-derived worlds");
        assert_eq!(result.decision.decision.n_draws, 4);
        assert_eq!(result.cyclic.worlds.len(), 2);
        assert!(
            result.cyclic.worlds[1]
                .iter()
                .zip(&result.cyclic.worlds[0])
                .all(|(promoted, ordinary)| promoted.values[0] > ordinary.values[0])
        );
        // Independent hand solution: with traffic clamped at c the wishlists equation
        // w = 0.1 w + 0.3 c + e gives w = (0.3 c + e) / 0.9 on every disturbance row, and the
        // clamped coordinate equals c exactly.
        let noise = request().noise;
        for (world, clamp) in [1.0_f64, 2.0].into_iter().enumerate() {
            assert_eq!(result.cyclic.worlds[world].len(), noise.len());
            for (draw, row) in result.cyclic.worlds[world].iter().zip(&noise) {
                let truth = (0.3 * clamp + row[0]) / 0.9;
                assert!((draw.values[0] - truth).abs() < 1e-7, "world {world}");
                assert!((draw.values[1] - clamp).abs() < 1e-12);
            }
        }
        // The promotion effect (0.3 / 0.9 per draw) is positive on every draw.
        assert!(matches!(
            &result.decision.decision.verdict,
            antecedent_design::decision_eval::Verdict::UniquelyOptimal(action)
                if action == "promoted"
        ));
        let mut repeated = request();
        repeated.worlds[1] = repeated.worlds[0].clone();
        assert!(
            compile_cyclic_action_world_decision(repeated, outputs(), contract(), &ctx,).is_err()
        );
    }

    #[test]
    fn the_paired_worlds_artifact_replays_under_the_held_program_identity() {
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_cyclic_action_world_decision(request(), outputs(), contract(), &ctx)
            .expect("model and decision compiler");
        let result = plan.execute(&ctx).expect("paired model-derived worlds");
        let bytes = plan.export(&result, "cyclic-paired-worlds", &ctx).expect("combined artifact");
        let replay = consume_cyclic_action_world_decision(&bytes, plan.lowered().identity(), &ctx)
            .expect("independent model and decision replay");
        assert_eq!(replay.decision.decision, result.decision.decision);
        assert_eq!(replay.cyclic, result.cyclic);
        // The identity is held outside the artifact: another program's identity is refused.
        let mut other = request();
        other.noise[0][0] = -0.3;
        let other_plan = compile_cyclic_action_world_decision(other, outputs(), contract(), &ctx)
            .expect("another disturbance row set");
        assert!(
            consume_cyclic_action_world_decision(&bytes, other_plan.lowered().identity(), &ctx)
                .is_err()
        );
    }

    #[test]
    fn a_handoff_exported_under_the_released_question_encoding_still_verifies() {
        let ctx = ExecutionContext::for_tests(47);
        let released = compile_cyclic_action_world_decision_encoded(
            request(),
            outputs(),
            contract(),
            QuestionEncoding::V1,
            &ctx,
        )
        .expect("the released encoding compiles");
        let current = compile_cyclic_action_world_decision(request(), outputs(), contract(), &ctx)
            .expect("the current encoding compiles");
        // The two encodings name the same program differently, so the released identity is
        // reproduced only to verify a handoff exported under it.
        assert_ne!(released.lowered().identity(), current.lowered().identity());
        let result = released.execute(&ctx).expect("paired model-derived worlds");
        let bytes = released.export(&result, "cyclic-released", &ctx).expect("combined artifact");
        let replay =
            consume_cyclic_action_world_decision(&bytes, released.lowered().identity(), &ctx)
                .expect("a released handoff verifies");
        assert_eq!(replay.decision.decision, result.decision.decision);
        assert_eq!(replay.cyclic, result.cyclic);
    }

    /// Once the decision gate is live, a release build refuses an unsupported unit with
    /// `NotSupported`; every cyclic decision route carries that refusal in its `Invalid`
    /// payload, and it keeps the `not_supported` class and its own detail instead of reading
    /// as an invalid request.
    #[test]
    fn a_gate_refusal_keeps_its_not_supported_class() {
        use crate::analysis::cyclic_component_composition::JointComponentPosteriorError;
        use crate::analysis::cyclic_posterior_action_decision::CyclicPosteriorDecisionError;
        use crate::analysis::cyclic_target_feedback_decision::CyclicTargetFeedbackError;
        use antecedent_core::reason_details::{cyclic_component, cyclic_decision};
        // The refusal `gate_decision` returns in a release build for an unsupported unit.
        let refusal = antecedent_core::NotSupported {
            detail: "CyclicActionWorldDecision",
            missing: vec!["unit_evidence".to_owned()],
        };
        let message = refusal.message();
        let not_supported = antecedent_core::reason_code!("not_supported");
        let shown = format!("{}: {message}", cyclic_decision::NOT_SUPPORTED);
        let action = CyclicActionDecisionError::Invalid(message);
        assert_eq!(action.reason_code(), Some(not_supported));
        assert_eq!(action.detail(), Some(cyclic_decision::NOT_SUPPORTED));
        assert_eq!(action.to_string(), shown);
        let posterior = CyclicPosteriorDecisionError::Invalid(message);
        assert_eq!(posterior.reason_code(), Some(not_supported));
        assert_eq!(posterior.detail(), Some(cyclic_decision::NOT_SUPPORTED));
        assert_eq!(posterior.to_string(), shown);
        let target = CyclicTargetFeedbackError::Invalid(message);
        assert_eq!(target.reason_code(), Some(not_supported));
        assert_eq!(target.detail(), Some(cyclic_decision::NOT_SUPPORTED));
        assert_eq!(target.to_string(), shown);
        let component = JointComponentPosteriorError::Invalid(message);
        assert_eq!(component.reason_code(), Some(not_supported));
        assert_eq!(component.detail(), Some(cyclic_component::NOT_SUPPORTED));
        assert_eq!(
            component.to_string(),
            format!("{}: {message}", cyclic_component::NOT_SUPPORTED)
        );
        // An invalid declaration keeps its own class.
        let invalid = CyclicActionDecisionError::Invalid("output equation is outside the model");
        assert_eq!(invalid.reason_code(), Some(antecedent_core::reason_code!("invalid_argument")));
        assert_eq!(invalid.detail(), Some(cyclic_decision::INVALID_REQUEST));
        assert_eq!(
            invalid.to_string(),
            "cyclic_decision.invalid_request: output equation is outside the model"
        );
        let invalid = JointComponentPosteriorError::Invalid("output port is absent");
        assert_eq!(invalid.reason_code(), Some(antecedent_core::reason_code!("invalid_argument")));
        assert_eq!(invalid.detail(), Some(cyclic_component::INVALID_REQUEST));
    }

    fn query_of(lowered: &LoweredProgram) -> [u8; 32] {
        match lowered.scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => *query,
            _ => panic!("the program target is a query functional"),
        }
    }

    #[test]
    fn site_cyclic_action_world_decision_question_orders_outputs_and_binds_parts() {
        let ctx = ExecutionContext::for_tests(47);
        let compile = |request: CyclicExecutionRequest,
                       outputs: &[CyclicDecisionOutput],
                       contract: DecisionContract| {
            compile_cyclic_action_world_decision(request, outputs.to_vec(), contract, &ctx)
                .ok()
                .map(|plan| query_of(plan.lowered()))
        };
        // Output i becomes chain stage `cyclic.outcome.i` and the first output is the chain's
        // reported quantity, so the outputs are read in declared order.
        crate::identity_laws::ordered_laws(&outputs(), |outputs: &[CyclicDecisionOutput]| {
            compile(request(), outputs, contract())
        });
        let mut units = contract();
        units.utility_units = "wishlist_value".into();
        let mut shifted = request();
        shifted.model.noise_covariance[0] = 2.0;
        // The decision is taken over the disturbance rows, so a different row set is a
        // different question over the same model, worlds and outputs.
        let mut disturbed = request();
        disturbed.noise[0][0] = -0.3;
        let mut fewer = request();
        fewer.noise.pop();
        crate::identity_laws::all_distinct(&[
            compile(request(), &outputs(), contract()),
            compile(request(), &outputs(), units),
            compile(shifted, &outputs(), contract()),
            compile(disturbed, &outputs(), contract()),
            compile(fewer, &outputs(), contract()),
        ]);
    }
}
