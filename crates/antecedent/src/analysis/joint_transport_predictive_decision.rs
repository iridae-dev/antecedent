//! Fitted source-target outcome prediction under an explicit shared-baseline premise.
//!
//! The source likelihood identifies treatment effects under the checked selection
//! diagram. Predicting an absolute target outcome additionally assumes that the
//! source-varying baseline block and known residual variance apply in the target.
//! Paired-world utility additionally needs a declared Gaussian correlation
//! between potential-outcome disturbances; marginal outcome laws do not fix it.
//! This is a new, model-conditional predictive route, not a reinterpretation of
//! the effect posterior or an earned decision reliability claim.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, SemanticDigest,
    TransportQuery, VariableId,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_estimate::joint_bayesian_transport::{
    JointTransportFit, JointTransportModel, JointTransportOptions, SourceData, SourceSharing,
    TargetData, VaryingBlock,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::joint_bayesian_transport_artifact::{
    JointBayesianArtifactWire, JointBayesianConsumeLimits, JointBayesianInputWire,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::{
    AdmgWire, IoError, TransportQueryWire, admg_from_wire, admg_to_wire, transport_query_from_wire,
    transport_query_to_wire,
};
use antecedent_kernels::standard_normal;
use serde::{Deserialize, Serialize};

use super::program_handle::{CompiledProgramHandle, gate_program};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_action_world_chain_decision,
    consume_chain_decision,
};
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram,
};

const FEATURE: &str = "joint_transport_predictive_decision_v2";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Complete source fit, target-frame prediction and paired decision declaration.
#[derive(Clone, Debug)]
pub struct JointTransportPredictiveDecisionRequest {
    /// Fixed causal graph with selection targets.
    pub diagram: SelectionDiagram,
    /// Identified binary treatment effect query.
    pub query: TransportQuery,
    /// Joint Gaussian source fit; requires a source-shared varying block.
    pub model: JointTransportModel,
    /// Source randomized trials.
    pub sources: Vec<SourceData>,
    /// Empirical target covariate frame.
    pub target: TargetData,
    /// Exact conjugate posterior draw schedule.
    pub options: JointTransportOptions,
    /// Known target residual variance, explicitly assumed transportable.
    pub target_noise_variance: f64,
    /// Correlation of the two target potential-outcome disturbances, in [-1,1].
    /// This is a cross-world premise, not identified by the source likelihood.
    pub target_noise_correlation: f64,
    /// Untreated and treated target outcome coordinates, in that order.
    pub worlds: [ScientificQuantity; 2],
    /// Utility over exactly those two predictive action worlds.
    pub decision: DecisionContract,
}

/// Structural, fit, predictive, decision or portable-replay refusal.
#[derive(Debug, thiserror::Error)]
pub enum JointTransportPredictiveDecisionError {
    /// Declaration or replay identity is inconsistent.
    #[error("joint transport predictive decision: {0}")]
    Invalid(&'static str),
    /// Graph identification failed.
    #[error("joint transport predictive identification: {0}")]
    Identify(String),
    /// Decision contract failed.
    #[error("joint transport predictive contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Paired predictive chain or utility failed.
    #[error("joint transport predictive chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Producer or portable artifact failed.
    #[error("joint transport predictive artifact: {0}")]
    Io(#[from] IoError),
}

/// Six-layer program retained before a fit or a prediction runs.
#[derive(Clone, Debug)]
pub struct CompiledJointTransportPredictiveDecision {
    request: JointTransportPredictiveDecisionRequest,
    identification: TransportIdentification,
    input: JointBayesianInputWire,
    graph: AdmgWire,
    query: TransportQueryWire,
    lowered: Arc<LoweredProgram>,
}

/// Paired target outcome predictions and the decision on their joint rows.
#[derive(Clone, Debug)]
pub struct JointTransportPredictiveDecisionResult {
    /// Refitted joint source-target producer.
    pub fit: JointTransportFit,
    /// Portable source fit and source-target identification receipt.
    pub producer: JointBayesianArtifactWire,
    /// Predictive outcome chain and utility result.
    pub decision: ChainDecisionResult,
    /// Outer compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// No whole-route decision reliability has been measured.
    pub calibration: &'static str,
}

/// Prospective coverage input for a fresh treated target outcome. It is not
/// an earned interval or a reliability claim for the action choice.
#[derive(Clone, Debug)]
pub struct JointTransportPredictiveCalibrationCandidate {
    /// Complete compiled-route result.
    pub result: JointTransportPredictiveDecisionResult,
    /// Nearest-rank central 95% interval of the treated predictive draws.
    pub interval: (f64, f64),
    /// Exact execution coordinate and model scope for a future record.
    pub basis: CalibrationBasis,
    /// Prospective action-choice scope, including the predictive coupling and utility.
    pub action_choice_basis: SemanticDigest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictiveDecisionWire {
    version: u32,
    required_feature: String,
    graph: AdmgWire,
    selections: Vec<u32>,
    query: TransportQueryWire,
    worlds: [ScientificQuantityWire; 2],
    target_noise_variance: f64,
    target_noise_correlation: f64,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    producer_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
    calibration: String,
}

fn check_worlds(
    request: &JointTransportPredictiveDecisionRequest,
) -> Result<VariableId, JointTransportPredictiveDecisionError> {
    let Some((treatment, outcome)) = request.query.response.functional.primary_pair() else {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "query lacks a treatment/outcome pair",
        ));
    };
    if request.model.sharing != SourceSharing::SharedVaryingBlock
        || !request.target_noise_variance.is_finite()
        || request.target_noise_variance <= 0.0
        || !request.target_noise_correlation.is_finite()
        || !(-1.0..=1.0).contains(&request.target_noise_correlation)
        || request.target.rows == 0
    {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "target outcome prediction requires a shared baseline and positive known target variance",
        ));
    }
    for (world, quantity) in request.worlds.iter().enumerate() {
        if quantity.variable_id != outcome.raw().to_string()
            || quantity.role != QuantityRole::Outcome
            || quantity.population_id != request.query.target_population.as_ref()
            || quantity.population_id != request.decision.target_population
            || quantity.regime_id != format!("do({}={world})", treatment.raw())
            || quantity.functional_id != "outcome"
            || quantity.transform_id != "identity"
            || quantity.horizon != 0
            || quantity.horizon != request.decision.horizon
            || !quantity.conditioning.is_empty()
        {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "target outcome world coordinate differs",
            ));
        }
    }
    if request.worlds[0].units != request.worlds[1].units || request.decision.actions.len() != 2 {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "paired outcome worlds or decision actions differ",
        ));
    }
    let mut regimes = BTreeSet::new();
    for action in &request.decision.actions {
        if !matches!(action.kind, ActionKind::Intervention | ActionKind::Regime)
            || action.inputs.len() != 1
            || !regimes.insert(action.inputs[0].regime_id.as_str())
            || !request
                .worlds
                .iter()
                .any(|world| action.inputs[0].require_same_coordinate(world).is_ok())
        {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "each action must read one distinct paired outcome world",
            ));
        }
    }
    Ok(outcome)
}

/// Bind the graph, source rows, target frame, shared-baseline and variance
/// premises, posterior schedule, paired outcome coordinates and utility.
///
/// # Errors
/// An invalid premise, graph query, coordinate or decision is refused.
pub fn compile_joint_transport_predictive_decision(
    request: JointTransportPredictiveDecisionRequest,
) -> Result<CompiledJointTransportPredictiveDecision, JointTransportPredictiveDecisionError> {
    compile_joint_transport_predictive_decision_as(request, Question::Current)
}

/// Which question digest a program binds: the current one, or the released one (selection
/// targets in declared order) a retained artifact was compiled under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Question {
    Current,
    Released,
}

/// The selection targets as the question writes them. They are a set: identification reads
/// them by membership and the diagram refuses a repeated target, so the current question
/// writes them sorted; the released one kept the declared order.
fn selection_words(diagram: &SelectionDiagram, version: Question) -> Vec<u64> {
    let mut words: Vec<u64> =
        diagram.selection_targets().iter().map(|v| u64::from(v.raw())).collect();
    if version == Question::Current {
        words.sort_unstable();
    }
    words
}

#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
fn compile_joint_transport_predictive_decision_as(
    request: JointTransportPredictiveDecisionRequest,
    version: Question,
) -> Result<CompiledJointTransportPredictiveDecision, JointTransportPredictiveDecisionError> {
    request.decision.validate()?;
    let outcome = check_worlds(&request)?;
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
        .map_err(|error| JointTransportPredictiveDecisionError::Identify(error.to_string()))?;
    let contract_identity = request.decision.identity()?;
    let selections = selection_words(&request.diagram, version);
    let input_bytes = antecedent_io::to_cbor(&input)?;
    let graph_bytes = antecedent_io::to_cbor(&graph)?;
    let query_bytes = antecedent_io::to_cbor(&query)?;
    let worlds_bytes =
        antecedent_io::to_cbor(&request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)))?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.joint_transport_predictive_decision.inputs.v2",
        &input_bytes,
    );
    let mut writer = CanonicalWriter::new(match version {
        Question::Current => "antecedent.joint_transport_predictive_decision.question.v3",
        Question::Released => "antecedent.joint_transport_predictive_decision.question.v2",
    });
    writer
        .bytes(&graph_bytes)
        .words(&selections)
        .bytes(&query_bytes)
        .bytes(&input_bytes)
        .bytes(&worlds_bytes)
        .word(request.target_noise_variance.to_bits())
        .word(request.target_noise_correlation.to_bits())
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
        plan_id: Arc::from("joint_transport_predictive_decision.v2"),
        data: DataClassification::Tabular,
        graph_class: "fixed_selection_dag_shared_baseline_predictive",
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
        "joint_transport_predictive_gaussian:{}",
        SemanticDigest::from_bytes(source_digest).to_hex()
    ))
    .map_err(|_| JointTransportPredictiveDecisionError::Invalid("invalid prior identity"))?;
    let likelihood = LikelihoodModel::new("joint_transport_shared_baseline_known_variance_normal")
        .map_err(|_| {
            JointTransportPredictiveDecisionError::Invalid("invalid likelihood identity")
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
            family: Arc::from("joint_source_target_predictive_outcome"),
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
            EvidenceSourceRef { role: EvidenceRole::JointScenarios, digest: Some(question) },
        ]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::PredictiveLaw,
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
            scenario_source: Some(Arc::from(
                "joint_transport_shared_baseline_target_prediction_v2",
            )),
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
                    JointTransportPredictiveDecisionError::Invalid("invalid read operator")
                })?,
            PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| {
                JointTransportPredictiveDecisionError::Invalid("invalid fit operator")
            })?,
            PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(|_| {
                JointTransportPredictiveDecisionError::Invalid("invalid transport operator")
            })?,
            PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(|_| {
                JointTransportPredictiveDecisionError::Invalid("invalid intervention operator")
            })?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::JointTransportPredictiveDecision,
            scientific,
            identification: identification_layer,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Predictive,
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    );
    Ok(CompiledJointTransportPredictiveDecision {
        request,
        identification,
        input,
        graph,
        query,
        lowered,
    })
}

fn paired_outcomes(
    fit: &JointTransportFit,
    model: &JointTransportModel,
    target: &TargetData,
    noise_variance: f64,
    noise_correlation: f64,
    ctx: &ExecutionContext,
) -> Result<[Vec<f64>; 2], JointTransportPredictiveDecisionError> {
    let p = model.features.len();
    let expected_parameters = 2 + 2 * p;
    let width = fit.draws.width();
    if fit.parameter_names.len() != expected_parameters
        || target.rows == 0
        || target.covariates.len() != p
        || target.covariates.iter().any(|column| column.len() != target.rows)
        || fit.draws.n_draws == 0
        || width < expected_parameters
        || fit.draws.n_draws.checked_mul(width) != Some(fit.draws.values.len())
    {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "fit parameter or target-frame layout differs",
        ));
    }
    let mut rng = ctx.rng.stream(0x5452_5052_4544_5631);
    let mut outcomes =
        [Vec::with_capacity(fit.draws.n_draws), Vec::with_capacity(fit.draws.n_draws)];
    for draw in 0..fit.draws.n_draws {
        let row = uniform_target_row(&mut rng, target.rows);
        let values = &fit.draws.values[draw * width..draw * width + expected_parameters];
        let mut untreated = values[expected_parameters
            - (1 + if model.varying == VaryingBlock::InterceptAndCovariates { p } else { 0 })];
        let mut effect = values[0];
        for j in 0..p {
            let x = target.covariates[j][row];
            effect += values[1 + j] * x;
            untreated += match model.varying {
                VaryingBlock::Intercept => values[1 + p + j] * x,
                VaryingBlock::InterceptAndCovariates => values[2 + p + j] * x,
            };
        }
        let z0 = standard_normal(&mut rng);
        let z1 = standard_normal(&mut rng);
        let sd = noise_variance.sqrt();
        let y0 = untreated + sd * z0;
        let y1 = untreated
            + effect
            + sd * (noise_correlation * z0
                + (1.0 - noise_correlation * noise_correlation).max(0.0).sqrt() * z1);
        if !y0.is_finite() || !y1.is_finite() {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "predictive outcome is nonfinite",
            ));
        }
        outcomes[0].push(y0);
        outcomes[1].push(y1);
    }
    Ok(outcomes)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the remainder is below a bound that came from a usize"
)]
fn uniform_target_row(rng: &mut antecedent_core::CausalRng, rows: usize) -> usize {
    let bound = rows as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let draw = rng.next_u64();
        if draw >= threshold {
            return (draw % bound) as usize;
        }
    }
}

fn predictive_chain(
    outcomes: &[Vec<f64>; 2],
    worlds: &[ScientificQuantity; 2],
    alignment: &SemanticDigest,
) -> PropagationChainRecordWire {
    let alignment = alignment.to_hex();
    let mut sources = Vec::with_capacity(2);
    let mut stages = Vec::with_capacity(2);
    let mut joint = Vec::with_capacity(2);
    for world in 0..2 {
        let source_id = format!("target.predictive.world.{world}");
        let stage_id = format!("target.predictive.outcome.{world}");
        sources.push(
            ChainSourceWire::draws(
                &source_id,
                LawMeaning::Predictive,
                &alignment,
                &outcomes[world],
            )
            .with_factor(SharedFactorKind::Parameter, &alignment)
            .with_factor(SharedFactorKind::Disturbance, &alignment),
        );
        stages.push(ChainStageWire::new(
            &stage_id,
            StageKind::DerivedQuantity,
            &[source_id.as_str()],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        ));
        joint.push(ChainJointOutputWire {
            stage: stage_id,
            quantity: ScientificQuantityWire::from(&worlds[world]),
        });
    }
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Predictive,
        &joint[0].stage,
        sources,
        stages,
    )
    .with_output_quantity(joint[0].quantity.clone())
    .with_joint_outputs(joint)
}

impl CompiledProgramHandle for CompiledJointTransportPredictiveDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledJointTransportPredictiveDecision {
    /// The complete six-layer predictive program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the program without fitting or predicting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Refit the producer, draw a target-frame unit and correlated Gaussian
    /// disturbances per posterior row, and score both action worlds jointly.
    ///
    /// # Errors
    /// Changed producer inputs, a failed fit, prediction or decision refuses.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointTransportPredictiveDecisionResult, JointTransportPredictiveDecisionError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| {
                JointTransportPredictiveDecisionError::Invalid(refusal.message())
            })?;
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
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "producer changed its compiled inputs",
            ));
        }
        let outcomes = paired_outcomes(
            &fit,
            &self.request.model,
            &self.request.target,
            self.request.target_noise_variance,
            self.request.target_noise_correlation,
            &scoped,
        )?;
        let decision = compile_action_world_chain_decision(
            predictive_chain(&outcomes, &self.request.worlds, &self.lowered.identity()),
            self.request.decision.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(JointTransportPredictiveDecisionResult {
            fit,
            producer,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Expose a source-only candidate for prospective treated-outcome
    /// prediction coverage. The target covariate distribution is the empirical
    /// frame; an independent future outcome supplies truth in the harness.
    ///
    /// # Errors
    /// Any route execution refusal or invalid final predictive table.
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointTransportPredictiveCalibrationCandidate, JointTransportPredictiveDecisionError>
    {
        let result = self.execute(ctx)?;
        let record = result.decision.chain.record();
        let treated = record.sources.get(1).ok_or(
            JointTransportPredictiveDecisionError::Invalid("treated outcome column is absent"),
        )?;
        let mut ordered: Vec<f64> = treated.values.iter().copied().map(f64::from_bits).collect();
        if ordered.len() < 40 || ordered.iter().any(|value| !value.is_finite()) {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "predictive interval needs finite draws",
            ));
        }
        ordered.sort_by(f64::total_cmp);
        let rank = |numerator: usize| (ordered.len() * numerator).div_ceil(1000).saturating_sub(1);
        let interval = (ordered[rank(25)], ordered[rank(975)]);
        let producer_basis = result.fit.target_calibration_basis(0.95).map_err(|_| {
            JointTransportPredictiveDecisionError::Invalid(
                "producer calibration scope is unavailable",
            )
        })?;
        let functional = format!(
            "target_treated_outcome:features={}:varying={:?}:shared_baseline:target_noise_bits={:016x}:cross_world_rho_bits={:016x}:empirical_target_frame:draws={}",
            self.request.model.features.len(),
            self.request.model.varying,
            self.request.target_noise_variance.to_bits(),
            self.request.target_noise_correlation.to_bits(),
            self.request.options.draws,
        );
        let basis = CalibrationBasis::new(
            [
                "OutcomePrediction",
                "Dag",
                "fixed",
                "multi_env",
                "Bayesian",
                FEATURE,
                "nearest_rank_95",
                "",
                "independent_source_target_future_units",
                producer_basis.posterior.as_ref(),
                functional.as_str(),
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            producer_basis.row_count,
            None,
            producer_basis.posterior_draws,
            0.0,
        )
        .with_validation("none");
        let mut action =
            CanonicalWriter::new("antecedent.joint_transport_predictive.action_choice_study.v1");
        action
            .text(FEATURE)
            .text(producer_basis.posterior.as_ref())
            .text(&functional)
            .text(&self.request.decision.identity()?)
            .bytes(&antecedent_io::to_cbor(
                &self.request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)),
            )?);
        Ok(JointTransportPredictiveCalibrationCandidate {
            result,
            interval,
            basis,
            action_choice_basis: SemanticDigest::from_bytes(action.finish()),
        })
    }

    /// Reexecute before sealing the complete fitted prediction and decision.
    ///
    /// # Errors
    /// A changed fit, action-world law or utility refuses.
    pub fn export(
        &self,
        result: &JointTransportPredictiveDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointTransportPredictiveDecisionError> {
        let replay = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.fit != replay.fit
            || result.producer.export()? != replay.producer.export()?
            || result.decision.decision != replay.decision.decision
            || result.decision.chain.identity() != replay.decision.chain.identity()
        {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "predictive decision changed on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.request.decision.clone())?;
        let chain = compile_action_world_chain_decision(
            replay.decision.chain.record().clone(),
            self.request.decision.clone(),
            ctx,
        )?;
        let wire = PredictiveDecisionWire {
            version: 2,
            required_feature: FEATURE.into(),
            graph: self.graph.clone(),
            selections: self.request.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: self.query.clone(),
            worlds: self.request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)),
            target_noise_variance: self.request.target_noise_variance,
            target_noise_correlation: self.request.target_noise_correlation,
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            producer_artifact: replay.producer.export()?,
            chain_artifact: chain.export(&result.decision, artifact_id)?,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(JointTransportPredictiveDecisionError::Invalid(
                "predictive artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Reidentify, refit and redraw the complete paired target outcome law from
/// the portable receipt. The expected outer identity must come from the caller.
///
/// # Errors
/// Any altered premise, producer, predictive row or decision refuses.
pub fn consume_joint_transport_predictive_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<JointTransportPredictiveDecisionResult, JointTransportPredictiveDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "predictive artifact exceeds its byte bound",
        ));
    }
    let wire: PredictiveDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 2
        || wire.required_feature != FEATURE
        || wire.calibration != "not_measured"
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "predictive feature or identity differs",
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
    .map_err(|_| {
        JointTransportPredictiveDecisionError::Invalid("invalid stored selection diagram")
    })?;
    let query = transport_query_from_wire(&wire.query)?;
    let checked = TransportIdentifier::new()
        .identify(&diagram, &query)
        .map_err(|error| JointTransportPredictiveDecisionError::Identify(error.to_string()))?;
    if antecedent_io::transport_identification_to_wire(&checked) != producer.identification {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "fit certificate differs from raw graph",
        ));
    }
    let input = JointBayesianInputWire::from_artifact(&producer);
    let (model, sources, target, options) = input.to_declaration();
    let worlds: [ScientificQuantity; 2] = [
        wire.worlds[0]
            .clone()
            .try_into()
            .map_err(JointTransportPredictiveDecisionError::Invalid)?,
        wire.worlds[1]
            .clone()
            .try_into()
            .map_err(JointTransportPredictiveDecisionError::Invalid)?,
    ];
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let request = JointTransportPredictiveDecisionRequest {
        diagram,
        query,
        model,
        sources,
        target,
        options,
        target_noise_variance: wire.target_noise_variance,
        target_noise_correlation: wire.target_noise_correlation,
        worlds,
        decision: contract.contract().clone(),
    };
    let mut plan = compile_joint_transport_predictive_decision(request.clone())?;
    if plan.lowered.identity() != expected_program {
        // An artifact retained under the released question still verifies: it is recompiled
        // under that question and must match it exactly.
        plan = compile_joint_transport_predictive_decision_as(request, Question::Released)?;
    }
    if plan.lowered.identity() != expected_program {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "recompiled predictive program differs",
        ));
    }
    let live = plan.execute(ctx)?;
    if live.fit != fit || live.producer.export()? != producer.export()? {
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "producer fit differs on replay",
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
        return Err(JointTransportPredictiveDecisionError::Invalid(
            "predictive utility differs on replay",
        ));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};
    use antecedent_design::decision_contract::{
        DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_estimate::joint_bayesian_transport::{
        DataIdentity, GaussianPrior, JointPriors, PriorProvenance, SourceDependence,
        TransportGraphClass,
    };

    #[test]
    #[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
    fn paired_target_outcomes_replay_with_a_shared_baseline() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let from = graph.dense_of_static(treatment).unwrap();
        let to = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(from, to).unwrap();
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
        let world = |value: u8| ScientificQuantity {
            variable_id: "2".into(),
            variable_name: "target outcome".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: format!("do(0={value})"),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let worlds = [world(0), world(1)];
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![worlds[0].clone()],
                    utility: UtilityExpr::Input(0),
                },
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![worlds[1].clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0),
                        UtilityExpr::Const(1.0),
                    ),
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
            sharing: SourceSharing::SharedVaryingBlock,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        };
        let ctx = ExecutionContext::for_tests(92);
        let request = JointTransportPredictiveDecisionRequest {
            diagram,
            query,
            model,
            sources: vec![source],
            target,
            options: JointTransportOptions { draws: 1024, seed: 92 },
            target_noise_variance: 0.1,
            target_noise_correlation: 1.0,
            worlds,
            decision,
        };
        let plan = compile_joint_transport_predictive_decision(request.clone()).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "JointTransportPredictiveDecision");
        let result = plan.execute(&ctx).unwrap();
        let columns = &result.decision.chain.record().sources;
        let effects: Vec<f64> = columns[0]
            .values
            .iter()
            .zip(&columns[1].values)
            .map(|(left, right)| f64::from_bits(*right) - f64::from_bits(*left))
            .collect();
        let mean = effects.iter().sum::<f64>() / effects.len() as f64;
        assert!((mean - 2.0).abs() < 0.05);
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "paired-target-outcomes", &ctx).unwrap();
        let replay = consume_joint_transport_predictive_decision(
            &bytes,
            result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(replay.fit, result.fit);
        assert_eq!(replay.decision.decision, result.decision.decision);
        let mut independent = request;
        independent.target_noise_correlation = 0.0;
        let other = compile_joint_transport_predictive_decision(independent)
            .unwrap()
            .execute(&ctx)
            .unwrap();
        assert_eq!(other.fit, result.fit);
        assert_ne!(other.decision.chain.identity(), result.decision.chain.identity());
    }

    /// The paired-outcome declaration above, rebuilt for each identity the site test compares.
    #[allow(clippy::too_many_lines, reason = "one declaration read top to bottom")]
    fn predictive_request() -> JointTransportPredictiveDecisionRequest {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let from = graph.dense_of_static(treatment).unwrap();
        let to = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(from, to).unwrap();
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
        let world = |value: u8| ScientificQuantity {
            variable_id: "2".into(),
            variable_name: "target outcome".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: format!("do(0={value})"),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let worlds = [world(0), world(1)];
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![worlds[0].clone()],
                    utility: UtilityExpr::Input(0),
                },
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![worlds[1].clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0),
                        UtilityExpr::Const(1.0),
                    ),
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
            sharing: SourceSharing::SharedVaryingBlock,
            dependence: SourceDependence::IndependentSamples,
            priors: JointPriors {
                invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
                varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            },
            max_unsupported_mass: 0.0,
            conflict_z_threshold: 3.0,
        };
        JointTransportPredictiveDecisionRequest {
            diagram,
            query,
            model,
            sources: vec![source],
            target,
            options: JointTransportOptions { draws: 1024, seed: 92 },
            target_noise_variance: 0.1,
            target_noise_correlation: 1.0,
            worlds,
            decision,
        }
    }

    fn question(request: JointTransportPredictiveDecisionRequest) -> Option<[u8; 32]> {
        let plan = compile_joint_transport_predictive_decision(request).ok()?;
        match plan.lowered().scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => Some(*query),
            _ => None,
        }
    }

    #[test]
    fn site_joint_transport_predictive_decision_question_binds_worlds_rows_and_premises() {
        let mut seed = predictive_request();
        seed.options.seed = 93;
        // Source rows are positional: the posterior schedule and the datum identities read them
        // in declared order.
        let mut swapped = predictive_request();
        swapped.sources[0].treatment.swap(0, 1);
        swapped.sources[0].outcome.swap(0, 1);
        let mut variance = predictive_request();
        variance.target_noise_variance = 0.2;
        let mut correlation = predictive_request();
        correlation.target_noise_correlation = 0.5;
        let mut units = predictive_request();
        units.decision.utility_units = "points".into();
        // The worlds are (untreated, treated): the order names which world is which.
        let mut worlds = predictive_request();
        worlds.worlds.swap(0, 1);
        crate::identity_laws::all_distinct(&[
            question(predictive_request()),
            question(seed),
            question(swapped),
            question(variance),
            question(correlation),
            question(units),
            question(worlds),
        ]);
        crate::identity_laws::bit_exact(|value| {
            let mut request = predictive_request();
            request.target_noise_correlation = value;
            question(request)
        });
    }

    fn selection_diagram(targets: &[u32]) -> Option<SelectionDiagram> {
        SelectionDiagram::try_new(
            antecedent_graph::Admg::with_variables(3),
            targets.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .ok()
    }

    #[test]
    fn site_joint_transport_predictive_decision_selection_targets_are_a_set() {
        // Identification reads the selection targets by membership and the diagram refuses a
        // repeat (the repeated case is refused and skipped).
        crate::identity_laws::set_laws(&[2_u32, 0, 1], |targets: &[u32]| {
            Some(selection_words(&selection_diagram(targets)?, Question::Current))
        });
        // The released question wrote them in declared order.
        let released = |targets: &[u32]| {
            selection_diagram(targets).map(|diagram| selection_words(&diagram, Question::Released))
        };
        assert_ne!(released(&[2, 0, 1]), released(&[0, 1, 2]));
    }

    #[test]
    fn site_joint_transport_predictive_decision_released_identity_still_verifies() {
        let released = compile_joint_transport_predictive_decision_as(
            predictive_request(),
            Question::Released,
        )
        .expect("the request compiles under the released question");
        let current = compile_joint_transport_predictive_decision(predictive_request())
            .expect("the request compiles under the current question");
        assert_ne!(released.lowered().identity(), current.lowered().identity());
        let ctx = ExecutionContext::for_tests(92);
        let result = released.execute(&ctx).expect("the released program runs");
        let bytes = released.export(&result, "released-paired-outcomes", &ctx).expect("it exports");
        let replay = consume_joint_transport_predictive_decision(
            &bytes,
            result.lowered_program_identity,
            &ctx,
        )
        .expect("an artifact retained under the released question still verifies");
        assert_eq!(replay.fit, result.fit);
    }
}
