//! Source-only trial-to-target IPW Bayesian-bootstrap effect decisions.
//!
//! The posterior is a Dirichlet empirical-support law over independent trial
//! rows, conditional on supplied selection and assignment probabilities and
//! target sample size. It is an average-effect law, not future-outcome risk.
//! Neither interval coverage nor final decision reliability is earned.
use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, ResponseFunctional,
    ResponseQuery, SemanticDigest, TransportQuery, VariableId,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{DecisionContract, DecisionContractError};
use antecedent_estimate::{DrawSpec, trial_to_target_bayesian_bootstrap};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};
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

const FEATURE: &str = "trial_ipw_transport_effect_decision_v1";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;
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
    /// Known or externally supplied trial selection probabilities.
    pub selection_probability: Vec<f64>,
    /// Known or externally supplied treatment probabilities.
    pub treatment_probability: Vec<f64>,
    /// Number of Dirichlet posterior draws, from 40 through 100,000.
    pub draws: usize,
    /// Fixed posterior random seed.
    pub seed: u64,
}

/// Raw structural, trial, effect-coordinate and decision declaration.
#[derive(Clone, Debug)]
pub struct TrialIpwTransportDecisionRequest {
    /// Selection diagram to identify anew on replay.
    pub diagram: SelectionDiagram,
    /// Unweighted mean response query at binary treatment values 0 and 1.
    pub query: TransportQuery,
    /// Raw independent trial and target rows.
    pub rows: TrialIpwRows,
    /// Target average treatment effect coordinate, not a predictive outcome.
    pub quantity: ScientificQuantity,
    /// Utility over the exact effect coordinate.
    pub decision: DecisionContract,
}

/// Refusal of identification, inputs, utility or replay.
#[derive(Debug, thiserror::Error)]
pub enum TrialIpwTransportDecisionError {
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
type Error = TrialIpwTransportDecisionError;

/// Retained six-layer raw-evidence compiler program.
pub struct CompiledTrialIpwTransportDecision {
    request: TrialIpwTransportDecisionRequest,
    identification: TransportIdentification,
    graph: AdmgWire,
    query: TransportQueryWire,
    lowered: Arc<LoweredProgram>,
}

/// Posterior effect draws and decision; no reliability is earned.
pub struct TrialIpwTransportDecisionResult {
    /// Aligned empirical-support posterior effect draws.
    pub effect_draws: Vec<f64>,
    /// Final decision on those effect draws.
    pub decision: ChainDecisionResult,
    /// Identity of the complete compiled program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `not_measured` for this route.
    pub calibration: &'static str,
}

/// Source-only candidate for prospective effect-interval coverage and
/// action-choice reliability studies. Neither coordinate is an earned record.
pub struct TrialIpwCalibrationCandidate {
    /// Complete fitted effect decision, labelled `not_measured`.
    pub result: TrialIpwTransportDecisionResult,
    /// Nearest-rank equal-tail 95% target-effect posterior interval.
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
    version: u32,
    feature: String,
    graph: AdmgWire,
    selections: Vec<u32>,
    query: TransportQueryWire,
    rows: TrialIpwRows,
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    effect_draws: Vec<f64>,
    program_identity: [u8; 32],
    calibration: String,
}

/// Compile and identify the raw trial-to-target effect and its utility.
///
/// # Errors
/// Invalid row ownership, effect coordinate, structural query or contract.
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn compile_trial_ipw_transport_decision(
    request: TrialIpwTransportDecisionRequest,
) -> Result<CompiledTrialIpwTransportDecision, Error> {
    request.decision.validate()?;
    let ResponseFunctional::MeanCurve { outcome, treatment } = &request.query.response.functional
    else {
        return Err(Error::Invalid("binary mean response query required"));
    };
    if treatment.grid.values().map_err(|_| Error::Invalid("invalid treatment grid"))?.as_slice()
        != [0.0, 1.0]
        || request.query.response != ResponseQuery::new(request.query.response.functional.clone())
        || request.query.weighted_target.is_some()
        || request.query.source_population.is_empty()
        || request.query.target_population.is_empty()
        || request.query.source_population == request.query.target_population
    {
        return Err(Error::Invalid(
            "only the complete unweighted binary mean response is supported",
        ));
    }
    let q = &request.quantity;
    if q.role != QuantityRole::Outcome
        || q.variable_id != outcome.raw().to_string()
        || q.population_id != request.query.target_population.as_ref()
        || q.population_id != request.decision.target_population
        || q.regime_id
            != format!("do({}=1)-do({}=0)", treatment.variable.raw(), treatment.variable.raw())
        || q.functional_id != "average_effect"
        || q.horizon != 0
        || q.horizon != request.decision.horizon
        || !q.conditioning.is_empty()
        || q.transform_id != "identity"
        || request.decision.actions.is_empty()
        || request
            .decision
            .actions
            .iter()
            .any(|a| a.inputs.len() != 1 || a.inputs[0].require_same_coordinate(q).is_err())
    {
        return Err(Error::Invalid("utility must read the exact target average-effect coordinate"));
    }
    let rows = &request.rows;
    let n = rows.unit_ids.len();
    let mut ids = BTreeSet::new();
    if !(3..=1_000_000).contains(&n)
        || !(40..=100_000).contains(&rows.draws)
        || n.checked_mul(rows.draws).is_none_or(|work| work > 50_000_000)
        || [
            rows.outcomes.len(),
            rows.treatment.len(),
            rows.trial.len(),
            rows.selection_probability.len(),
            rows.treatment_probability.len(),
        ]
        .iter()
        .any(|&len| len != n)
        || rows.unit_ids.iter().any(|id| id.trim().is_empty() || id.len() > 256 || !ids.insert(id))
        || rows.outcomes.iter().any(|x| !x.is_finite())
        || rows.selection_probability.iter().any(|x| !x.is_finite() || *x <= 0.0 || *x >= 1.0)
        || rows
            .treatment_probability
            .iter()
            .zip(&rows.trial)
            .any(|(x, trial)| *trial && (!x.is_finite() || *x <= 0.0 || *x >= 1.0))
        || !rows.trial.iter().any(|trial| !trial)
        || !rows.trial.iter().zip(&rows.treatment).any(|(&trial, &a)| trial && a)
        || !rows.trial.iter().zip(&rows.treatment).any(|(&trial, &a)| trial && !a)
    {
        return Err(Error::Invalid(
            "invalid independent trial/target rows, identities or draw count",
        ));
    }
    let graph = admg_to_wire(request.diagram.causal_graph())?;
    let query = transport_query_to_wire(&request.query)?;
    let identification = TransportIdentifier::new()
        .identify(&request.diagram, &request.query)
        .map_err(|e| Error::Producer(e.to_string()))?;
    let graph_bytes = antecedent_io::to_cbor(&graph)?;
    let query_bytes = antecedent_io::to_cbor(&query)?;
    let row_bytes = antecedent_io::to_cbor(rows)?;
    let coordinate = antecedent_io::to_cbor(&ScientificQuantityWire::from(q))?;
    let contract = request.decision.identity()?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.trial_ipw_transport_decision.rows.v1",
        &row_bytes,
    );
    let mut writer = CanonicalWriter::new("antecedent.trial_ipw_transport_decision.question.v1");
    writer
        .bytes(&graph_bytes)
        .words(
            &request
                .diagram
                .selection_targets()
                .iter()
                .map(|v| u64::from(v.raw()))
                .collect::<Vec<_>>(),
        )
        .bytes(&query_bytes)
        .bytes(&row_bytes)
        .bytes(&coordinate)
        .text(&contract);
    let question = writer.finish();
    let design = DesignShape::quoted(n as u64, 5, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(*outcome),
            query: question,
        },
        plan_id: Arc::from(FEATURE),
        data: DataClassification::Tabular,
        graph_class: "fixed_selection_trial_ipw",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let prior = Prior::new("trial_ipw_dirichlet_unit_weights")
        .map_err(|_| Error::Invalid("invalid empirical-support prior"))?;
    let likelihood = LikelihoodModel::new("trial_ipw_empirical_support")
        .map_err(|_| Error::Invalid("invalid empirical likelihood"))?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Bayesian(OrdinaryBayesianLikelihood::new(
            likelihood,
            prior.clone(),
        )),
        models: Arc::from([ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from("trial_ipw_empirical_support"),
            inputs: Arc::from([]),
            output: Some(*outcome),
        }]),
        folds: None,
        resampling: None,
        posterior: Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::BayesianBootstrap,
            draws: rows.draws,
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
            contract_identity: Some(Arc::from(contract.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("trial_ipw_target_effect_posterior_v1")),
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(row_bytes.len() as u64))
                .build()
                .map_err(|_| Error::Invalid("invalid row read"))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|_| Error::Invalid("invalid fit"))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .build()
                .map_err(|_| Error::Invalid("invalid transport"))?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::TrialIpwTransportDecision,
            scientific,
            identification: IdentificationLayer::NonGraphical(Arc::new(
                NonGraphicalIdentification::SharedEvidenceContract { contract: question },
            )),
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
    Ok(CompiledTrialIpwTransportDecision { request, identification, graph, query, lowered })
}

fn effect_chain(
    draws: &[f64],
    identity: SemanticDigest,
    quantity: &ScientificQuantity,
) -> PropagationChainRecordWire {
    let id = identity.to_hex();
    let source =
        ChainSourceWire::draws("target.effect.posterior", LawMeaning::Posterior, &id, draws)
            .with_factor(SharedFactorKind::Parameter, &id);
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Posterior,
        "target.effect",
        vec![source],
        vec![ChainStageWire::new(
            "target.effect",
            StageKind::DerivedQuantity,
            &["target.effect.posterior"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        )],
    )
    .with_output_quantity(ScientificQuantityWire::from(quantity))
}

impl CompiledProgramHandle for CompiledTrialIpwTransportDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTrialIpwTransportDecision {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect the plan without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }
    /// Refit the trial posterior and evaluate the decision.
    ///
    /// # Errors
    /// Unsupported certificate, inputs, resources or utility.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TrialIpwTransportDecisionResult, Error> {
        let rows = &self.request.rows;
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| Error::Invalid(refusal.message()))?;
        }
        let _lease = scoped
            .acquire_admission(
                0,
                (rows.draws as u64)
                    .saturating_mul(16)
                    .saturating_add((rows.unit_ids.len() as u64).saturating_mul(32)),
            )
            .map_err(|_| Error::Invalid("trial IPW admission refused"))?;
        if scoped.cancellation.is_cancelled() {
            return Err(Error::Invalid("trial IPW cancelled"));
        }
        let draws = trial_to_target_bayesian_bootstrap(
            &self.identification,
            &rows.outcomes,
            &rows.treatment,
            &rows.trial,
            &rows.selection_probability,
            &rows.treatment_probability,
            DrawSpec::new(rows.draws, rows.seed),
        )
        .map_err(|e| Error::Producer(e.to_string()))?;
        if scoped.cancellation.is_cancelled() {
            return Err(Error::Invalid("trial IPW cancelled"));
        }
        let decision = compile_chain_decision(
            effect_chain(&draws, self.lowered.identity(), &self.request.quantity),
            self.request.decision.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(TrialIpwTransportDecisionResult {
            effect_draws: draws,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }
    /// Execute the complete route and expose unmeasured study coordinates for
    /// effect coverage and action-choice reliability. An external known-truth
    /// study must supply the effect and the optimal action for each replicate.
    ///
    /// # Errors
    /// A producer, decision or finite-draw refusal.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the draw count is validated to at most 100000 before it is recorded"
    )]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TrialIpwCalibrationCandidate, Error> {
        let result = self.execute(ctx)?;
        let mut ordered = result.effect_draws.clone();
        if ordered.len() < 40 || ordered.iter().any(|draw| !draw.is_finite()) {
            return Err(Error::Invalid("trial IPW interval needs finite posterior draws"));
        }
        ordered.sort_by(f64::total_cmp);
        let rank = |numerator: usize| (ordered.len() * numerator).div_ceil(1000).saturating_sub(1);
        let interval = (ordered[rank(25)], ordered[rank(975)]);
        let rows = &self.request.rows;
        // The construction is the graph, selection diagram and query; the rows,
        // their known probabilities, their count and the seed are not.
        let mut construction =
            CanonicalWriter::new("antecedent.trial_ipw_transport.calibration_construction.v1");
        construction
            .bytes(&antecedent_io::to_cbor(&self.graph)?)
            .words(
                &self
                    .request
                    .diagram
                    .selection_targets()
                    .iter()
                    .map(|v| u64::from(v.raw()))
                    .collect::<Vec<_>>(),
            )
            .bytes(&antecedent_io::to_cbor(&self.query)?);
        let construction = SemanticDigest::from_bytes(construction.finish()).to_hex();
        let functional = format!(
            "target_average_effect:construction={construction}:draws={}:known_probabilities:fixed_target_frame",
            rows.draws,
        );
        let basis = CalibrationBasis::new(
            [
                "AverageEffect",
                "Dag",
                "fixed",
                "multi_env",
                "Bayesian",
                FEATURE,
                "nearest_rank_95",
                "",
                "independent_trial_target_units",
                "trial_dirichlet_empirical_support",
                &functional,
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            rows.trial.len() as u64,
            None,
            Some(rows.draws as u32),
            0.0,
        )
        .with_validation("none");
        let mut action =
            CanonicalWriter::new("antecedent.trial_ipw_transport.action_choice_study.v1");
        action
            .text(FEATURE)
            .text(&functional)
            .text(&self.request.decision.identity()?)
            .bytes(&antecedent_io::to_cbor(&ScientificQuantityWire::from(&self.request.quantity))?);
        Ok(TrialIpwCalibrationCandidate {
            result,
            interval,
            interval_basis: basis,
            action_choice_basis: SemanticDigest::from_bytes(action.finish()),
        })
    }

    /// Reexecute and export raw inputs and the final utility artifact.
    ///
    /// # Errors
    /// Changed result or artifact encoding refusal.
    pub fn export(
        &self,
        result: &TrialIpwTransportDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.effect_draws != live.effect_draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(Error::Invalid("trial IPW result changed on replay"));
        }
        let contract = DecisionContractArtifact::new(self.request.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.effect_draws, self.lowered.identity(), &self.request.quantity),
            self.request.decision.clone(),
            ctx,
        )?;
        let wire = Wire {
            version: 1,
            feature: FEATURE.into(),
            graph: self.graph.clone(),
            selections: self.request.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: self.query.clone(),
            rows: self.request.rows.clone(),
            quantity: ScientificQuantityWire::from(&self.request.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?,
            effect_draws: live.effect_draws,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_BYTES {
            return Err(Error::Invalid("trial IPW artifact too large"));
        }
        Ok(bytes)
    }
}

/// Independently reidentify, refit raw trial rows and replay utility.
///
/// # Errors
/// Changed source, certificate, identity, draws, utility or artifact.
pub fn consume_trial_ipw_transport_decision(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<TrialIpwTransportDecisionResult, Error> {
    if bytes.len() > MAX_BYTES {
        return Err(Error::Invalid("trial IPW artifact too large"));
    }
    let wire: Wire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.feature != FEATURE
        || wire.calibration != "not_measured"
        || wire.program_identity != *expected.as_bytes()
    {
        return Err(Error::Invalid("foreign trial IPW artifact"));
    }
    let diagram = SelectionDiagram::try_new(
        admg_from_wire(&wire.graph)?,
        wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
    )
    .map_err(|_| Error::Invalid("invalid stored selection diagram"))?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let plan = compile_trial_ipw_transport_decision(TrialIpwTransportDecisionRequest {
        diagram,
        query: transport_query_from_wire(&wire.query)?,
        rows: wire.rows,
        quantity: wire.quantity.try_into().map_err(Error::Invalid)?,
        decision: contract.contract().clone(),
    })?;
    if plan.lowered.identity() != expected {
        return Err(Error::Invalid("trial IPW compiler changed"));
    }
    let live = plan.execute(ctx)?;
    if live.effect_draws != wire.effect_draws {
        return Err(Error::Invalid("trial IPW posterior changed"));
    }
    let chain = consume_chain_decision(
        &wire.chain_artifact,
        Some(live.decision.lowered_program_identity),
        ctx,
    )?;
    if chain.chain.identity() != live.decision.chain.identity()
        || chain.decision != live.decision.decision
    {
        return Err(Error::Invalid("trial IPW decision changed"));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{ContinuousDomain, GridSpec};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn trial_effect_is_refit_and_replayed_from_raw_rows() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let source_node = graph.dense_of_static(treatment).unwrap();
        let outcome_node = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(source_node, outcome_node).unwrap();
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
            variable_name: "effect".into(),
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
                    utility: UtilityExpr::Input(0),
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
        let rows = TrialIpwRows {
            unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
            outcomes: (0..80).map(|i| if i < 40 && i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
            treatment: (0..80).map(|i| i % 2 == 0).collect(),
            trial: (0..80).map(|i| i < 40).collect(),
            selection_probability: vec![0.5; 80],
            treatment_probability: vec![0.5; 80],
            draws: 100,
            seed: 31,
        };
        let request = TrialIpwTransportDecisionRequest { diagram, query, rows, quantity, decision };
        let plan = compile_trial_ipw_transport_decision(request.clone()).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "TrialIpwTransportDecision");
        let ctx = ExecutionContext::for_tests(31);
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert!(candidate.interval.0 <= candidate.interval.1);
        assert_eq!(candidate.interval_basis.validation.as_ref(), "none");
        let result = candidate.result;
        assert_eq!(result.effect_draws.len(), 100);
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "trial-ipw-decision", &ctx).unwrap();
        let replay =
            consume_trial_ipw_transport_decision(&bytes, result.lowered_program_identity, &ctx)
                .unwrap();
        assert_eq!(replay.effect_draws, result.effect_draws);
        assert_eq!(replay.decision.decision, result.decision.decision);
        let mut wire: Wire = antecedent_io::from_cbor(&bytes).unwrap();
        wire.rows.outcomes[0] += 1.0;
        let altered = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_trial_ipw_transport_decision(&altered, result.lowered_program_identity, &ctx)
                .is_err()
        );

        // Small prospective grid: IID Bernoulli assignment with true mean
        // effect 1 and uniquely optimal action "treat". Outcomes are fixed
        // potential outcomes Y(0)=0, Y(1)=1; selection and assignment
        // probabilities are both one half. Count outcomes without asserting
        // empirical coverage or reliability from this tiny grid.
        let mut covered = 0_usize;
        let mut correct = 0_usize;
        for seed in 1_u64..=4 {
            let mut replicate = request.clone();
            let mut state = seed;
            for i in 0..40 {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let treated = ((z ^ (z >> 31)) & 1) != 0;
                replicate.rows.treatment[i] = treated;
                replicate.rows.outcomes[i] = if treated { 1.0 } else { 0.0 };
            }
            let candidate = compile_trial_ipw_transport_decision(replicate)
                .unwrap()
                .calibration_candidate(&ctx)
                .unwrap();
            covered += usize::from(candidate.interval.0 <= 1.0 && 1.0 <= candidate.interval.1);
            correct += usize::from(matches!(candidate.result.decision.decision.verdict,
                antecedent_design::decision_eval::Verdict::UniquelyOptimal(ref id) if id == "treat"));
            assert_eq!(candidate.result.calibration, "not_measured");
        }
        assert!(covered <= 4 && correct <= 4);
    }
}
