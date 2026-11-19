//! Source-only trial-to-target IPW Bayesian-bootstrap predictive decisions.
//!
//! A shared Dirichlet posterior over trial rows induces two normalized IPW
//! empirical outcome distributions. Conditional on that mass, the two future
//! potential-outcome draws are independent by explicit declaration. This
//! cross-world coupling is not identified by the observed trial. The predictive
//! marginals additionally rely on correctly supplied selection and treatment
//! probabilities, selection and treatment exchangeability for the outcome
//! distributions, and positivity. The checked mean-query certificate alone
//! does not certify these distributional premises. Neither predictive coverage
//! nor action-choice reliability is earned.
use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{ChainReading, LawMeaning, StageKind, TableConstruction};
use antecedent_core::{ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, ResponseFunctional,
    ResponseQuery, SemanticDigest, TransportQuery, VariableId};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_estimate::{DrawSpec, trial_to_target_bayesian_bootstrap};
use antecedent_kernels::bayesian_bootstrap_weights;
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::{AdmgWire, IoError, TransportQueryWire, admg_from_wire, admg_to_wire,
    transport_query_from_wire, transport_query_to_wire};
use antecedent_io::propagation_chain_record::{ChainCoefficientWire, ChainOpWire,
    ChainSourceWire, ChainStageWire, ChainJointOutputWire, PropagationChainRecordWire};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::propagation_chain_decision::{ChainDecisionError, ChainDecisionResult,
    compile_action_world_chain_decision, consume_chain_decision};
use crate::planner::lowered::{CarrierDemand, ClaimDemand, Composition, DecisionProgram,
    DesignShape, EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading};
use crate::planner::statistical::{ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan,
    Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "trial_ipw_transport_predictive_decision_v1";
const MAX_BYTES: usize = 80 * 1024 * 1024;

/// Independent source-trial and target units with supplied design probabilities.
/// Target outcomes and assignments are ignored by the producer, but all rows
/// count toward the target sample size. Identities must be unique across rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialIpwRows {
    /// Distinct nonblank unit identities, in row order.
    pub unit_ids: Vec<String>,
    /// Outcome values in row order, including finite placeholders for target units.
    pub outcomes: Vec<f64>,
    /// Binary treatment indicator in row order.
    pub treatment: Vec<bool>,
    /// True for a source trial unit and false for a target unit.
    pub trial: Vec<bool>,
    /// Known or externally supplied trial selection probabilities, assumed correct.
    pub selection_probability: Vec<f64>,
    /// Known or externally supplied treatment probabilities, assumed correct.
    pub treatment_probability: Vec<f64>,
    /// Number of Dirichlet posterior draws, from 40 through 100,000.
    pub draws: usize,
    /// Fixed posterior random seed.
    pub seed: u64,
}

/// Raw structural, trial, paired outcome and decision declaration.
#[derive(Clone, Debug)]
pub struct TrialIpwPredictiveDecisionRequest {
    /// Selection diagram to identify anew on replay.
    pub diagram: SelectionDiagram,
    /// Unweighted mean response query at binary treatment values 0 and 1.
    pub query: TransportQuery,
    /// Raw independent trial and target rows.
    pub rows: TrialIpwRows,
    /// Untreated and treated predictive outcome coordinates.
    pub worlds: [ScientificQuantity; 2],
    /// Explicitly declares conditional independence of the two outcome innovations.
    pub independent_potential_outcomes_given_mass: bool,
    /// Utility over the declared predictive outcome worlds.
    pub decision: DecisionContract,
}

/// Refusal of identification, inputs, utility or replay.
#[derive(Debug, thiserror::Error)]
pub enum TrialIpwPredictiveDecisionError {
    /// Invalid or mismatched declaration.
    #[error("trial IPW transport decision: {0}")]
    Invalid(&'static str),
    /// Identification or fitting refusal.
    #[error("trial IPW transport producer: {0}")]
    Producer(String),
    /// Decision contract refusal.
    #[error("trial IPW transport contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Decision chain refusal.
    #[error("trial IPW transport chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding refusal.
    #[error("trial IPW transport artifact: {0}")]
    Io(#[from] IoError),
}
type Error = TrialIpwPredictiveDecisionError;

/// Retained six-layer raw-evidence compiler program.
pub struct CompiledTrialIpwPredictiveDecision {
    request: TrialIpwPredictiveDecisionRequest,
    identification: TransportIdentification,
    graph: AdmgWire,
    query: TransportQueryWire,
    lowered: Arc<LoweredProgram>,
}

/// Predictive outcome draws and decision; no reliability is earned.
pub struct TrialIpwPredictiveDecisionResult {
    /// Aligned predictive draws from the two empirical outcome supports.
    pub predictive_draws: [Vec<f64>; 2],
    /// Final decision on those outcome draws.
    pub decision: ChainDecisionResult,
    /// Identity of the complete compiled program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `not_measured` for this route.
    pub calibration: &'static str,
}

/// Source-only candidate for prospective treated-outcome interval coverage and
/// action-choice reliability studies. Neither coordinate is an earned record.
pub struct TrialIpwPredictiveCalibrationCandidate {
    /// Complete fitted predictive decision, labelled `not_measured`.
    pub result: TrialIpwPredictiveDecisionResult,
    /// Nearest-rank equal-tail 95% treated-outcome predictive interval.
    pub interval: (f64, f64),
    /// Exact interval construction and sampling scope for prospective validation.
    pub interval_basis: CalibrationBasis,
    /// Exact action-choice study coordinate, including the decision contract.
    /// Scoring requires independently specified true utilities and optimum;
    /// this digest is not a reliability record or a probability of correctness.
    pub action_choice_basis: SemanticDigest,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    version: u32, feature: String, graph: AdmgWire, selections: Vec<u32>,
    query: TransportQueryWire, rows: TrialIpwRows, worlds: [ScientificQuantityWire; 2],
    independent_potential_outcomes_given_mass: bool,
    decision_identity: String, decision_artifact: Vec<u8>, chain_artifact: Vec<u8>,
    predictive_draws: [Vec<f64>; 2], program_identity: [u8; 32], calibration: String,
}

/// Compile and identify the raw trial-to-target predictive law and its utility.
///
/// # Errors
/// Invalid row ownership, outcome coordinates, structural query or contract.
pub fn compile_trial_ipw_predictive_decision(
    request: TrialIpwPredictiveDecisionRequest,
) -> Result<CompiledTrialIpwPredictiveDecision, Error> {
    request.decision.validate()?;
    let ResponseFunctional::MeanCurve { outcome, treatment } = &request.query.response.functional
        else { return Err(Error::Invalid("binary mean response query required")); };
    if treatment.grid.values().map_err(|_| Error::Invalid("invalid treatment grid"))?.as_slice() != &[0.0, 1.0]
        || request.query.response != ResponseQuery::new(request.query.response.functional.clone())
        || request.query.weighted_target.is_some()
        || request.query.source_population.is_empty() || request.query.target_population.is_empty()
        || request.query.source_population == request.query.target_population
    { return Err(Error::Invalid("only the complete unweighted binary mean response is supported")); }
    if !request.independent_potential_outcomes_given_mass {
        return Err(Error::Invalid("conditional independence of potential outcomes must be declared"));
    }
    for (world, q) in request.worlds.iter().enumerate() {
        if q.role != QuantityRole::Outcome || q.variable_id != outcome.raw().to_string()
            || q.population_id != request.query.target_population.as_ref()
            || q.population_id != request.decision.target_population
            || q.regime_id != format!("do({}={world})", treatment.variable.raw())
            || q.functional_id != "outcome" || q.horizon != 0
            || q.horizon != request.decision.horizon || !q.conditioning.is_empty()
            || q.transform_id != "identity"
        { return Err(Error::Invalid("exact target predictive outcome worlds required")); }
    }
    if request.worlds[0].units != request.worlds[1].units || request.decision.actions.len() != 2
        || request.decision.actions[0].inputs.first().map(|q| &q.regime_id)
            == request.decision.actions[1].inputs.first().map(|q| &q.regime_id)
        || request.decision.actions.iter().any(|a| !matches!(a.kind, ActionKind::Intervention | ActionKind::Regime | ActionKind::Policy)
            || a.inputs.len() != 1
            || !request.worlds.iter().any(|q| a.inputs[0].require_same_coordinate(q).is_ok()))
    { return Err(Error::Invalid("two action worlds with matching predictive coordinates required")); }
    let rows = &request.rows;
    let n = rows.unit_ids.len();
    let mut ids = BTreeSet::new();
    if n < 3 || n > 1_000_000 || !(40..=100_000).contains(&rows.draws)
        || n.checked_mul(rows.draws).is_none_or(|work| work > 50_000_000)
        || [rows.outcomes.len(), rows.treatment.len(), rows.trial.len(),
            rows.selection_probability.len(), rows.treatment_probability.len()].iter().any(|&len| len != n)
        || rows.unit_ids.iter().any(|id| id.trim().is_empty() || id.len() > 256 || !ids.insert(id))
        || rows.outcomes.iter().any(|x| !x.is_finite())
        || rows.selection_probability.iter().any(|x| !x.is_finite() || *x <= 0.0 || *x >= 1.0)
        || rows.treatment_probability.iter().zip(&rows.trial)
            .any(|(x, trial)| *trial && (!x.is_finite() || *x <= 0.0 || *x >= 1.0))
        || !rows.trial.iter().any(|trial| !trial)
        || !rows.trial.iter().zip(&rows.treatment).any(|(&trial, &a)| trial && a)
        || !rows.trial.iter().zip(&rows.treatment).any(|(&trial, &a)| trial && !a)
    { return Err(Error::Invalid("invalid independent trial/target rows, identities or draw count")); }
    let graph = admg_to_wire(request.diagram.causal_graph())?;
    let query = transport_query_to_wire(&request.query)?;
    let identification = TransportIdentifier::new().identify(&request.diagram, &request.query)
        .map_err(|e| Error::Producer(e.to_string()))?;
    let graph_bytes = antecedent_io::to_cbor(&graph)?;
    let query_bytes = antecedent_io::to_cbor(&query)?;
    let row_bytes = antecedent_io::to_cbor(rows)?;
    let coordinate = antecedent_io::to_cbor(&request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)))?;
    let contract = request.decision.identity()?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.trial_ipw_predictive_decision.rows.v1", &row_bytes);
    let mut writer = CanonicalWriter::new("antecedent.trial_ipw_predictive_decision.question.v1");
    writer.bytes(&graph_bytes).words(&request.diagram.selection_targets().iter()
        .map(|v| u64::from(v.raw())).collect::<Vec<_>>())
        .bytes(&query_bytes).bytes(&row_bytes).bytes(&coordinate)
        .word(u64::from(request.independent_potential_outcomes_given_mass)).text(&contract);
    let question = writer.finish();
    let design = DesignShape::quoted(n as u64, 5, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional { kind: FunctionalKind::PolicyValue,
            outcome: Some(*outcome), query: question },
        plan_id: Arc::from(FEATURE), data: DataClassification::Tabular,
        graph_class: "fixed_selection_trial_ipw", query_variables: Arc::from([]),
        source_digest: Some(source_digest), support_coordinate: None,
        supported: false, support_status: None,
    };
    let prior = Prior::new("trial_ipw_dirichlet_unit_weights")
        .map_err(|_| Error::Invalid("invalid empirical-support prior"))?;
    let likelihood = LikelihoodModel::new("trial_ipw_empirical_support")
        .map_err(|_| Error::Invalid("invalid empirical likelihood"))?;
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior.clone())),
        models: Arc::from([ModelSlot { role: ModelRole::Outcome,
            family: Arc::from("trial_ipw_empirical_support"), inputs: Arc::from([]), output: Some(*outcome) }]),
        folds: None, resampling: None,
        posterior: Some(PosteriorPlan { algorithm: PosteriorAlgorithm::BayesianBootstrap,
            draws: rows.draws, prior }), design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(question) }]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram { carrier: CarrierDemand::PredictiveLaw,
            utility: DecisionProgram::utility_form(None), actions: request.decision.actions.len(),
            constraints: request.decision.constraints.len(),
            chance_threshold: request.decision.constraints.iter().map(|c| c.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(request.decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract.as_str())), information_value: None,
            scenario_source: Some(Arc::from("trial_ipw_target_predictive_independent_worlds_v1")) }),
        planning: None, chain: None,
    };
    let physical = PhysicalLayer { operators: Arc::from([
        PhysicalOperator::declare(PhysicalEffect::Read).estimates(ByteEstimates::absent()
            .with_output(row_bytes.len() as u64)).build().map_err(|_| Error::Invalid("invalid row read"))?,
        PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| Error::Invalid("invalid fit"))?,
        PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(|_| Error::Invalid("invalid transport"))?,
    ]), design };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::TrialIpwPredictiveDecision, scientific,
        identification: IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SharedEvidenceContract { contract: question })),
        statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Predictive, kind: Some(ClaimKind::PolicyValue) },
    }.finish());
    Ok(CompiledTrialIpwPredictiveDecision { request, identification, graph, query, lowered })
}

fn predictive_chain(draws: &[Vec<f64>; 2], identity: SemanticDigest, worlds: &[ScientificQuantity; 2]) -> PropagationChainRecordWire {
    let id = identity.to_hex();
    let mut sources = Vec::new();
    let mut stages = Vec::new();
    let mut joint = Vec::new();
    for world in 0..2 {
        let source = format!("target.predictive.source.{world}");
        let stage = format!("target.predictive.outcome.{world}");
        sources.push(ChainSourceWire::draws(&source, LawMeaning::Predictive, &id, &draws[world])
            .with_factor(SharedFactorKind::Parameter, &id));
        stages.push(ChainStageWire::new(&stage, StageKind::DerivedQuantity, &[&source],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0))));
        joint.push(ChainJointOutputWire { stage, quantity: ScientificQuantityWire::from(&worlds[world]) });
    }
    PropagationChainRecordWire::draft(TableConstruction::JointDraws, ChainReading::Predictive,
        &joint[0].stage, sources, stages).with_output_quantity(joint[0].quantity.clone()).with_joint_outputs(joint)
}

// Each arm uses the same Dirichlet mass. Sampling independently from the two
// normalized, selection-and-assignment-weighted supports is the declared
// conditional cross-world assumption; the trial does not identify that joint law.
fn predictive_draws(rows: &TrialIpwRows) -> Result<[Vec<f64>; 2], Error> {
    let source: Vec<usize> = rows.trial.iter().enumerate().filter_map(|(i, &v)| v.then_some(i)).collect();
    let mut mass_rng = antecedent_core::CausalRng::from_seed(rows.seed);
    let mut innovation_rng = antecedent_core::CausalRng::from_seed(rows.seed ^ 0x7472_6961_6c70_7265);
    let mut result = [Vec::with_capacity(rows.draws), Vec::with_capacity(rows.draws)];
    for _ in 0..rows.draws {
        let mass = bayesian_bootstrap_weights(&mut mass_rng, source.len(), false);
        for arm in 0..2 {
            let weights: Vec<f64> = source.iter().zip(&mass).map(|(&i, &w)| {
                if rows.treatment[i] != (arm == 1) { return 0.0; }
                let selection = rows.selection_probability[i];
                let assignment = if arm == 1 { rows.treatment_probability[i] } else { 1.0 - rows.treatment_probability[i] };
                w * ((1.0 - selection) / selection) / assignment
            }).collect();
            let total: f64 = weights.iter().sum();
            if !total.is_finite() || total <= 0.0 { return Err(Error::Invalid("invalid predictive IPW mass")); }
            let mut threshold = innovation_rng.next_f64() * total;
            let mut selected = None;
            for (&i, &w) in source.iter().zip(&weights) {
                if w > 0.0 { selected = Some(i); }
                if w > 0.0 && threshold < w { selected = Some(i); break; }
                threshold -= w;
            }
            result[arm].push(rows.outcomes[selected.ok_or(Error::Invalid("empty predictive arm"))?]);
        }
    }
    Ok(result)
}

impl CompiledTrialIpwPredictiveDecision {
    /// Inspect all six compiler layers.
    #[must_use] pub fn lowered(&self) -> &LoweredProgram { &self.lowered }
    /// Inspect the plan without fitting.
    #[must_use] pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }
    /// Refit the trial posterior and evaluate the decision.
    ///
    /// # Errors
    /// Unsupported certificate, inputs, resources or utility.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TrialIpwPredictiveDecisionResult, Error> {
        let rows = &self.request.rows;
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let _lease = scoped.acquire_admission(0, (rows.draws as u64).saturating_mul(16)
            .saturating_add((rows.unit_ids.len() as u64).saturating_mul(32)))
            .map_err(|_| Error::Invalid("trial IPW admission refused"))?;
        if scoped.cancellation.is_cancelled() { return Err(Error::Invalid("trial IPW cancelled")); }
        // This producer checks the identified formula and the same row-support contract.
        trial_to_target_bayesian_bootstrap(&self.identification, &rows.outcomes,
            &rows.treatment, &rows.trial, &rows.selection_probability, &rows.treatment_probability,
            DrawSpec::new(2, rows.seed)).map_err(|e| Error::Producer(e.to_string()))?;
        let draws = predictive_draws(rows)?;
        if scoped.cancellation.is_cancelled() { return Err(Error::Invalid("trial IPW cancelled")); }
        let decision = compile_action_world_chain_decision(predictive_chain(&draws, self.lowered.identity(), &self.request.worlds),
            self.request.decision.clone(), ctx)?.execute(ctx)?;
        Ok(TrialIpwPredictiveDecisionResult { predictive_draws: draws, decision,
            lowered_program_identity: self.lowered.identity(), calibration: "not_measured" })
    }
    /// Execute the complete route and expose unmeasured study coordinates for
    /// treated-outcome coverage and action-choice reliability. An external study
    /// must supply a fresh outcome and independently specified optimal action.
    ///
    /// # Errors
    /// A producer, decision or finite-draw refusal.
    pub fn calibration_candidate(&self, ctx: &ExecutionContext)
        -> Result<TrialIpwPredictiveCalibrationCandidate, Error> {
        let result = self.execute(ctx)?;
        let mut ordered = result.predictive_draws[1].clone();
        if ordered.len() < 40 || ordered.iter().any(|draw| !draw.is_finite()) {
            return Err(Error::Invalid("trial IPW interval needs finite predictive draws"));
        }
        ordered.sort_by(f64::total_cmp);
        let rank = |numerator: usize| (ordered.len() * numerator).div_ceil(1000).saturating_sub(1);
        let interval = (ordered[rank(25)], ordered[rank(975)]);
        let rows = &self.request.rows;
        let trial_count = rows.trial.iter().filter(|&&trial| trial).count();
        let mut design = CanonicalWriter::new("antecedent.trial_ipw_transport.calibration_design.v1");
        design.bytes(&antecedent_io::to_cbor(&self.graph)?)
            .words(&self.request.diagram.selection_targets().iter()
                .map(|v| u64::from(v.raw())).collect::<Vec<_>>())
            .bytes(&antecedent_io::to_cbor(&self.query)?);
        for ((&trial, &selection), &assignment_probability) in rows.trial.iter()
            .zip(&rows.selection_probability).zip(&rows.treatment_probability) {
            design.word(if trial { 1 } else { 0 })
                .word(selection.to_bits()).word(assignment_probability.to_bits());
        }
        let design = SemanticDigest::from_bytes(design.finish()).to_hex();
        let functional = format!(
            "target_treated_outcome:conditional_independent_potential_outcomes:normalized_ipw:design={design}:trial_n={trial_count}:target_n={}:draws={}:known_probabilities:fixed_target_frame",
            rows.trial.len() - trial_count, rows.draws,
        );
        let basis = CalibrationBasis::new([
            "OutcomePrediction", "Dag", "fixed", "multi_env", "Bayesian", FEATURE,
            "nearest_rank_95", "", "independent_trial_target_units",
            "trial_dirichlet_normalized_ipw_predictive_support", &functional,
        ].map(Arc::from), 0.95, Arc::from("model_conditional"), rows.trial.len() as u64,
            None, Some(rows.draws as u32), 0.0).with_validation("none");
        let mut action = CanonicalWriter::new("antecedent.trial_ipw_transport.action_choice_study.v1");
        action.text(FEATURE).text(&functional).text(&self.request.decision.identity()?)
            .bytes(&antecedent_io::to_cbor(&self.request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)))?);
        Ok(TrialIpwPredictiveCalibrationCandidate { result, interval, interval_basis: basis,
            action_choice_basis: SemanticDigest::from_bytes(action.finish()) })
    }

    /// Reexecute and export raw inputs and the final utility artifact.
    ///
    /// # Errors
    /// Changed result or artifact encoding refusal.
    pub fn export(&self, result: &TrialIpwPredictiveDecisionResult, artifact_id: &str,
        ctx: &ExecutionContext) -> Result<Vec<u8>, Error> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity() || result.calibration != "not_measured"
            || result.predictive_draws != live.predictive_draws || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        { return Err(Error::Invalid("trial IPW result changed on replay")); }
        let contract = DecisionContractArtifact::new(self.request.decision.clone())?;
        let chain = compile_action_world_chain_decision(predictive_chain(&live.predictive_draws, self.lowered.identity(),
            &self.request.worlds), self.request.decision.clone(), ctx)?;
        let wire = Wire { version: 1, feature: FEATURE.into(), graph: self.graph.clone(),
            selections: self.request.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: self.query.clone(), rows: self.request.rows.clone(),
            worlds: self.request.worlds.clone().map(|q| ScientificQuantityWire::from(&q)),
            independent_potential_outcomes_given_mass: self.request.independent_potential_outcomes_given_mass,
            decision_identity: contract.identity().into(), decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?, predictive_draws: live.predictive_draws,
            program_identity: *self.lowered.identity().as_bytes(), calibration: "not_measured".into() };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_BYTES { return Err(Error::Invalid("trial IPW artifact too large")); }
        Ok(bytes)
    }
}

/// Independently reidentify, refit raw trial rows and replay utility.
///
/// # Errors
/// Changed source, certificate, identity, draws, utility or artifact.
pub fn consume_trial_ipw_predictive_decision(bytes: &[u8], expected: SemanticDigest,
    ctx: &ExecutionContext) -> Result<TrialIpwPredictiveDecisionResult, Error> {
    if bytes.len() > MAX_BYTES { return Err(Error::Invalid("trial IPW artifact too large")); }
    let wire: Wire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.feature != FEATURE || wire.calibration != "not_measured"
        || wire.program_identity != *expected.as_bytes() { return Err(Error::Invalid("foreign trial IPW artifact")); }
    let diagram = SelectionDiagram::try_new(admg_from_wire(&wire.graph)?,
        wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>())
        .map_err(|_| Error::Invalid("invalid stored selection diagram"))?;
    let contract = DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let plan = compile_trial_ipw_predictive_decision(TrialIpwPredictiveDecisionRequest {
        diagram, query: transport_query_from_wire(&wire.query)?, rows: wire.rows,
        worlds: [wire.worlds[0].clone().try_into().map_err(Error::Invalid)?,
            wire.worlds[1].clone().try_into().map_err(Error::Invalid)?],
        independent_potential_outcomes_given_mass: wire.independent_potential_outcomes_given_mass, decision: contract.contract().clone(),
    })?;
    if plan.lowered.identity() != expected { return Err(Error::Invalid("trial IPW compiler changed")); }
    let live = plan.execute(ctx)?;
    if live.predictive_draws != wire.predictive_draws { return Err(Error::Invalid("trial IPW predictive draws changed")); }
    let chain = consume_chain_decision(&wire.chain_artifact, Some(live.decision.lowered_program_identity), ctx)?;
    if chain.chain.identity() != live.decision.chain.identity() || chain.decision != live.decision.decision {
        return Err(Error::Invalid("trial IPW decision changed"));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{ContinuousDomain, GridSpec};
    use antecedent_design::decision_contract::{DecisionAction, DecisionCriterion,
        StructuralPolicy, UtilityExpr};

    #[test]
    #[ignore = "source-only predictive route; no build or calibration earned"]
    fn predictive_worlds_refit_and_replay_raw_evidence() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let a = graph.dense_of_static(treatment).unwrap();
        let y = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(a, y).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let query = TransportQuery::new(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome, treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
        }), "trial", "target", [treatment]);
        let world = |arm| ScientificQuantity { variable_id: "2".into(), variable_name: "outcome".into(),
            role: QuantityRole::Outcome, units: "score".into(), population_id: "target".into(),
            regime_id: format!("do(0={arm})"), horizon: 0, functional_id: "outcome".into(),
            conditioning: Vec::new(), transform_id: "identity".into() };
        let worlds = [world(0), world(1)];
        let decision = DecisionContract { actions: vec![
            DecisionAction { id: "hold".into(), kind: ActionKind::Intervention,
                inputs: vec![worlds[0].clone()], utility: UtilityExpr::Input(0) },
            DecisionAction { id: "treat".into(), kind: ActionKind::Intervention,
                inputs: vec![worlds[1].clone()], utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)) },
        ], utility_units: "score".into(), criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(), target_population: "target".into(), horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction };
        let rows = TrialIpwRows { unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
            outcomes: (0..80).map(|i| if i < 40 { (i % 4) as f64 } else { 0.0 }).collect(),
            treatment: (0..80).map(|i| i % 2 == 0).collect(), trial: (0..80).map(|i| i < 40).collect(),
            selection_probability: vec![0.5; 80], treatment_probability: vec![0.5; 80], draws: 100, seed: 31 };
        let request = TrialIpwPredictiveDecisionRequest { diagram, query, rows, worlds, decision,
            independent_potential_outcomes_given_mass: true };
        let plan = compile_trial_ipw_predictive_decision(request.clone()).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "TrialIpwPredictiveDecision");
        let ctx = ExecutionContext::for_tests(31);
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.interval_basis.validation.as_ref(), "none");
        assert_eq!(candidate.result.calibration, "not_measured");
        assert!(candidate.result.predictive_draws[0].iter().all(|x| *x == 1.0 || *x == 3.0));
        assert!(candidate.result.predictive_draws[1].iter().all(|x| *x == 0.0 || *x == 2.0));
        let bytes = plan.export(&candidate.result, "trial-ipw-predictive", &ctx).unwrap();
        let replay = consume_trial_ipw_predictive_decision(&bytes, candidate.result.lowered_program_identity, &ctx).unwrap();
        assert_eq!(replay.predictive_draws, candidate.result.predictive_draws);
        assert_eq!(replay.decision.decision, candidate.result.decision.decision);
        let mut wire: Wire = antecedent_io::from_cbor(&bytes).unwrap();
        wire.rows.outcomes[0] += 1.0;
        assert!(consume_trial_ipw_predictive_decision(&antecedent_io::to_cbor(&wire).unwrap(),
            candidate.result.lowered_program_identity, &ctx).is_err());
        let mut no_premise = request;
        no_premise.independent_potential_outcomes_given_mass = false;
        assert!(compile_trial_ipw_predictive_decision(no_premise).is_err());
    }
}
