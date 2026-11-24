//! A declared multi-action decision over a compiler-retained chain law.
//!
//! The chain and the decision contract are compiled together before numerical
//! execution. The law is read from the chain's replayable output column, never
//! reconstructed from its mean or variance. An outcome-risk input requires a
//! predictive reading; a parameter posterior does not become outcome risk.

use std::sync::Arc;

use antecedent_contract::ScientificQuantity;
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, ChainRefusal, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::composition_boundary::SupportPolicy;
use antecedent_design::decision_artifact::{DecisionContractArtifact, decision_result_fingerprint};
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_design::decision_eval::{DecisionResult, MeanEstimate, SourceReceipt};
use antecedent_design::decision_evaluation::{
    DemandBudget, DemandSource, DemandSourceError, EvaluationError, execute_demand_admitted,
};
use antecedent_design::decision_transported_source::{TransportedResponse, TransportedValue};
use antecedent_io::IoError;
use antecedent_io::propagation_chain_record::{
    ChainChangeWire, ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire,
    ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use serde::{Deserialize, Serialize};

use super::lowering::chain::lower_chain_with_decision;
use super::propagation_chain::{PropagationChain, PropagationChainPlan, compile_propagation_chain};
use crate::planner::lowered::{CarrierDemand, DecisionProgram, LoweredProgram};
use crate::planner::{PlanStage, PlanView};

/// Refusal of the chain-and-decision compiler or its retained execution.
#[derive(Debug, thiserror::Error)]
pub enum ChainDecisionError {
    /// The chain compiler refused its source declaration or operation.
    #[error("chain compiler: {0}")]
    Chain(#[from] ChainRefusal),
    /// The multi-action decision contract is invalid.
    #[error("decision contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// The decision evaluator refused the retained law.
    #[error("decision evaluation: {0}")]
    Evaluation(#[from] EvaluationError),
    /// An artifact could not be encoded, decoded or independently consumed.
    #[error("chain decision artifact: {0}")]
    Io(#[from] IoError),
    /// A scientific coordinate or law meaning cannot support the decision.
    #[error("chain decision: {0}")]
    Coordinate(&'static str),
}

/// A decision and its chain, executed from one compiled declaration.
#[derive(Clone, Debug)]
pub struct ChainDecisionResult {
    /// Replayed-output-law chain that supplied the decision state.
    pub chain: PropagationChain,
    /// Multi-action utility and constraint evaluation over the aligned rows.
    pub decision: DecisionResult,
    /// The six-layer program identity binding the chain and decision contract.
    pub lowered_program_identity: SemanticDigest,
}

/// Retained plan of a chain law and a decision contract. Compilation evaluates
/// neither the chain stages nor any utility or constraint.
#[derive(Clone, Debug)]
pub struct ChainDecisionPlan {
    chain: PropagationChainPlan,
    contract: DecisionContract,
    coordinates: Vec<ScientificQuantity>,
    lowered: Arc<LoweredProgram>,
    action_worlds: bool,
}

/// Maximum encoded chain-decision artifact bytes accepted before decoding.
pub const MAX_CHAIN_DECISION_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;
const CHAIN_DECISION_FEATURE: &str = "chain_decision_v1";
const ACTION_WORLD_DECISION_FEATURE: &str = "chain_action_world_decision_v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChainDecisionArtifactWire {
    version: u32,
    required_feature: String,
    chain_record: Vec<u8>,
    chain_identity: String,
    contract: Vec<u8>,
    contract_identity: String,
    decision_fingerprint: String,
    lowered_program_identity: [u8; 32],
}

/// Compile a chain and a multi-action decision over its retained output law.
/// The contract may use nonlinear utilities, probability/quantile criteria,
/// hard chance constraints and regret. Every action input must name one of the
/// chain's exact output quantities. Multiple quantities require a version-4
/// joint output law, so the utility reads their aligned rows.
///
/// # Errors
/// The chain, decision contract, carrier or coordinate is unsupported.
pub fn compile_chain_decision(
    draft: PropagationChainRecordWire,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<ChainDecisionPlan, ChainDecisionError> {
    compile_chain_decision_with_worlds(draft, contract, ctx, false)
}

fn compile_chain_decision_with_worlds(
    draft: PropagationChainRecordWire,
    contract: DecisionContract,
    ctx: &ExecutionContext,
    action_worlds: bool,
) -> Result<ChainDecisionPlan, ChainDecisionError> {
    let chain = compile_propagation_chain(draft, ctx)?;
    contract.validate()?;
    if !matches!(chain.plan().reading, ChainReading::Posterior | ChainReading::Predictive) {
        return Err(ChainDecisionError::Coordinate(
            "a multi-action decision needs a retained posterior or predictive output law",
        ));
    }
    let retained = chain
        .lowered()
        .propagation_chain()
        .ok_or(ChainDecisionError::Coordinate("the chain has no retained program"))?;
    let quantity =
        retained.draft().output_quantity.clone().ok_or(ChainDecisionError::Coordinate(
            "the decision source must declare its scientific output coordinate",
        ))?;
    let primary: ScientificQuantity = quantity
        .try_into()
        .map_err(|_| ChainDecisionError::Coordinate("invalid decision source coordinate"))?;
    let coordinates: Vec<ScientificQuantity> = if retained.draft().joint_outputs.is_empty() {
        vec![primary]
    } else {
        retained
            .draft()
            .joint_outputs
            .iter()
            .map(|output| {
                output.quantity.clone().try_into().map_err(|_| {
                    ChainDecisionError::Coordinate("invalid joint decision source coordinate")
                })
            })
            .collect::<Result<_, _>>()?
    };
    for action in &contract.actions {
        for input in &action.inputs {
            if !coordinates
                .iter()
                .any(|coordinate| input.require_same_coordinate(coordinate).is_ok())
            {
                return Err(ChainDecisionError::Coordinate(
                    "every action input must match a retained chain law coordinate",
                ));
            }
            if input.functional_id == "outcome" && chain.plan().reading != ChainReading::Predictive
            {
                return Err(ChainDecisionError::Coordinate(
                    "outcome risk requires a predictive law; posterior parameter draws are insufficient",
                ));
            }
        }
    }
    if contract.source_requirement()?.is_some_and(|required| required.is_identified_set()) {
        return Err(ChainDecisionError::Coordinate(
            "an identified-set decision needs joint scenarios, not one output law",
        ));
    }
    let contract_identity = contract.identity()?;
    let carrier = if chain.plan().reading == ChainReading::Predictive {
        CarrierDemand::PredictiveLaw
    } else {
        CarrierDemand::ParameterPosterior
    };
    let program = DecisionProgram {
        carrier,
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
        scenario_source: action_worlds.then(|| Arc::from("declared_joint_action_worlds_v1")),
    };
    let lowered = lower_chain_with_decision(
        retained.draft().clone(),
        retained.deferred().cloned(),
        chain.plan().clone(),
        Some(program),
    )?;
    let lowered = Arc::new(lowered);
    let chain = chain.within_lowered(Arc::clone(&lowered))?;
    Ok(ChainDecisionPlan { chain, contract, coordinates, lowered, action_worlds })
}

/// Compile an action-world decision only when the caller supplies one
/// explicitly paired joint draw table and a shared latent factor. Each action
/// must read a quantity from its own distinct intervention regime. This
/// declares a cross-world coupling for regret and nonlinear utility; it does
/// not identify that coupling from marginal laws or earn calibration.
///
/// # Errors
/// Unpaired sources, undisclosed or absent common factors, repeated action
/// regimes, or any chain and decision refusal.
pub fn compile_action_world_chain_decision(
    draft: PropagationChainRecordWire,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<ChainDecisionPlan, ChainDecisionError> {
    if draft.construction != "joint_draws"
        || draft.joint_outputs.len() < 2
        || draft.sources.len() < 2
        || draft.sources.iter().any(|source| {
            source.representation != "draws"
                || !source.complete_disclosure
                || source.alignment.is_none()
        })
    {
        return Err(ChainDecisionError::Coordinate(
            "action worlds require a jointly aligned, fully disclosed draw table",
        ));
    }
    let alignment = draft.sources[0].alignment.as_ref();
    if draft.sources.iter().any(|source| source.alignment.as_ref() != alignment) {
        return Err(ChainDecisionError::Coordinate(
            "action-world source rows have different alignment identities",
        ));
    }
    let common = draft.sources[0].factors.iter().any(|factor| {
        matches!(
            SharedFactorKind::from_name(&factor.kind),
            Some(
                SharedFactorKind::Model
                    | SharedFactorKind::Parameter
                    | SharedFactorKind::Disturbance
            )
        ) && draft.sources[1..].iter().all(|source| source.factors.contains(factor))
    });
    if !common {
        return Err(ChainDecisionError::Coordinate(
            "action worlds need a declared shared model, parameter or disturbance factor",
        ));
    }
    let mut regimes = std::collections::BTreeSet::new();
    for action in &contract.actions {
        if !matches!(
            action.kind,
            ActionKind::Intervention | ActionKind::Regime | ActionKind::Policy
        ) || action.inputs.is_empty()
        {
            return Err(ChainDecisionError::Coordinate(
                "each action world must be an intervention, regime or policy with an input",
            ));
        }
        let regime = &action.inputs[0].regime_id;
        if !regimes.insert(regime.as_str())
            || action.inputs.iter().any(|input| input.regime_id != regime.as_str())
            || !draft
                .joint_outputs
                .iter()
                .any(|output| output.quantity.regime_id == regime.as_str())
        {
            return Err(ChainDecisionError::Coordinate(
                "each action must name a distinct retained world regime",
            ));
        }
    }
    if regimes.len() < 2 {
        return Err(ChainDecisionError::Coordinate(
            "an action-world comparison needs at least two regimes",
        ));
    }
    compile_chain_decision_with_worlds(draft, contract, ctx, true)
}

/// Compile a checked transported law and a multi-action decision through the
/// chain compiler. The response's receipt identity is bound to every source
/// column and remains visible in the decision source receipt. The transported
/// law must be jointly aligned and accepted by `policy`. Any nonnegative row
/// weights are normalized by the chain's retained reweighting schedule; zero
/// weight rows are omitted before compilation. Other transport laws
/// remain with their existing direct decision evaluator. This route treats the
/// already produced transport as declared evidence, without refitting its
/// source model or upgrading its extrapolative support or calibration.
///
/// # Errors
/// A point response, marginal or malformed weighted law, unsupported law meaning,
/// missing coordinates, or any chain/decision compiler refusal.
#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
pub fn compile_transported_law_chain_decision(
    response: &TransportedResponse,
    contract: DecisionContract,
    policy: &SupportPolicy,
    ctx: &ExecutionContext,
) -> Result<ChainDecisionPlan, ChainDecisionError> {
    let support = response.receipt().support_status().ok_or(ChainDecisionError::Coordinate(
        "transport receipt has no recognized support label",
    ))?;
    if support.severity() > policy.weakest_support.severity() {
        return Err(ChainDecisionError::Coordinate(
            "transport support is below the decision policy",
        ));
    }
    if contract.target_population != response.receipt().target_population {
        return Err(ChainDecisionError::Coordinate(
            "transport target population differs from the decision contract",
        ));
    }
    let TransportedValue::Law(law) = response.value() else {
        return Err(ChainDecisionError::Coordinate("transported decision needs an aligned law"));
    };
    if law.metadata().supported.as_ref().is_some_and(|mask| mask.iter().any(|supported| !supported))
    {
        return Err(ChainDecisionError::Coordinate(
            "transported chain law requires every declared coordinate to be supported",
        ));
    }
    let alignment = law.alignment_identity()?;
    let (reading, meaning) = match law.semantic() {
        DistributionMeaningWire::ParameterPosterior
        | DistributionMeaningWire::CausalFunctionalPosterior => {
            (ChainReading::Posterior, LawMeaning::Posterior)
        }
        DistributionMeaningWire::PosteriorPredictive
        | DistributionMeaningWire::InterventionalPredictive
        | DistributionMeaningWire::EmpiricalOutcome => {
            (ChainReading::Predictive, LawMeaning::Predictive)
        }
        DistributionMeaningWire::EstimatorSampling | DistributionMeaningWire::Bootstrap => {
            return Err(ChainDecisionError::Coordinate(
                "an estimator sampling law is not a posterior or predictive outcome law",
            ));
        }
    };
    let quantities = law.quantities();
    if quantities.is_empty() {
        return Err(ChainDecisionError::Coordinate("transported law has no output coordinate"));
    }
    let width = quantities.len();
    let retained_rows: Vec<usize> = if let Some(weights) = &law.metadata().weights {
        if weights.len() != law.n_draws()
            || weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(ChainDecisionError::Coordinate(
                "transported row weights must be finite, nonnegative and aligned",
            ));
        }
        (0..weights.len()).filter(|&row| weights[row] > 0.0).collect()
    } else {
        (0..law.n_draws()).collect()
    };
    if retained_rows.len() < 2 {
        return Err(ChainDecisionError::Coordinate(
            "transported decision needs at least two positive-mass joint rows",
        ));
    }
    let alignment = if retained_rows.len() == law.n_draws() {
        alignment
    } else {
        let mut writer =
            CanonicalWriter::new("antecedent.transported_law.positive_row_alignment.v1");
        writer.text(&alignment).word(retained_rows.len() as u64);
        for &row in &retained_rows {
            writer.word(row as u64);
        }
        SemanticDigest::from_bytes(writer.finish()).to_hex()
    };
    let receipt = format!("transport_receipt:{}", response.receipt().canonical_identity());
    let weighted = law.metadata().weights.is_some();
    let carrier_id = "transported.row_weight_carrier";
    let mut sources = Vec::with_capacity(width + usize::from(weighted));
    let mut stages = Vec::with_capacity(width);
    let mut outputs = Vec::with_capacity(width);
    for (column, quantity) in quantities.iter().enumerate() {
        let source_id = format!("transported.column.{column}");
        let stage_id = format!("transported.output.{column}");
        let values: Vec<f64> =
            retained_rows.iter().map(|&row| law.draws()[row * width + column]).collect();
        sources.push(
            ChainSourceWire::draws(&source_id, meaning, &alignment, &values)
                .with_factor(SharedFactorKind::Model, &receipt),
        );
        let stage = if weighted {
            ChainStageWire::new(
                &stage_id,
                StageKind::DerivedQuantity,
                &[source_id.as_str(), carrier_id],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::input(1),
                ),
            )
        } else {
            ChainStageWire::new(
                &stage_id,
                StageKind::DerivedQuantity,
                &[source_id.as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        };
        stages.push(stage);
        outputs.push(ChainJointOutputWire { stage: stage_id, quantity: quantity.clone() });
    }
    if weighted {
        sources.push(
            ChainSourceWire::draws(
                carrier_id,
                meaning,
                &alignment,
                &vec![0.0; retained_rows.len()],
            )
            .with_factor(SharedFactorKind::Model, &receipt),
        );
    }
    let mut draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        reading,
        &outputs[0].stage,
        sources,
        stages,
    )
    .with_output_quantity(quantities[0].clone());
    if outputs.len() > 1 {
        draft = draft.with_joint_outputs(outputs);
    }
    if let Some(weights) = &law.metadata().weights {
        let log_weights: Vec<f64> = retained_rows.iter().map(|&row| weights[row].ln()).collect();
        draft.changes.push(ChainChangeWire::reweight(carrier_id, &log_weights));
    }
    compile_chain_decision(draft, contract, ctx)
}

fn declared_transport_receipt(chain: &PropagationChain) -> Option<&str> {
    chain
        .record()
        .sources
        .iter()
        .flat_map(|source| &source.factors)
        .find(|factor| {
            factor.kind == SharedFactorKind::Model.as_str()
                && factor.id.starts_with("transport_receipt:")
        })
        .map(|factor| factor.id.as_str())
}

impl ChainDecisionPlan {
    /// The six-layer program binding chain inputs and the decision contract.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect every compiler layer without executing the chain or decision.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Execute the retained chain, then score the contract on its aligned
    /// output law under the context's memory admission and cancellation rules.
    ///
    /// # Errors
    /// A chain execution refusal, invalidated output law or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ChainDecisionResult, ChainDecisionError> {
        let chain = self.chain.execute(ctx)?;
        let mut source = ChainLawDemand::new(&chain, self.coordinates.clone())?;
        let row_bytes = (self.coordinates.len() as u64).saturating_mul(8).max(8);
        let draws = source.draw_count() as u64;
        let scratch = row_bytes.saturating_mul(draws).clamp(row_bytes, (1 << 20).max(row_bytes));
        let retained =
            (self.contract.actions.len() as u64 + 1).saturating_mul(draws).saturating_mul(8);
        let budget = DemandBudget {
            scratch_bytes: scratch,
            retained_bytes: retained,
            retention: OutputRetention::SummaryOnly,
        };
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let mut decision =
            execute_demand_admitted(&self.contract, &mut source, budget, &scoped)?.decision;
        decision.assumptions.push("model-conditional chain law; no measured coverage".into());
        decision.assumptions.push(
            "each action's outcome effect must already be represented by its declared input coordinates and aligned worlds".into(),
        );
        if declared_transport_receipt(&chain).is_some() {
            decision.assumptions.push(
                "transport receipt and law are declared inputs; source transport fitting is not replayed by this decision".into(),
            );
        }
        Ok(ChainDecisionResult {
            chain,
            decision,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Export the exact decision execution with its chain record and decision
    /// contract. The independent consumer reexecutes both before returning it.
    ///
    /// # Errors
    /// A result of another plan or an encoding failure refuses.
    pub fn export(
        &self,
        result: &ChainDecisionResult,
        artifact_id: &str,
    ) -> Result<Vec<u8>, ChainDecisionError> {
        if result.lowered_program_identity != self.lowered.identity()
            || result.chain.lowered_program_identity() != self.chain.identity()
            || result.decision.contract_identity != self.contract.identity()?
        {
            return Err(ChainDecisionError::Coordinate(
                "the result was not produced by this compiled chain decision",
            ));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let wire = ChainDecisionArtifactWire {
            version: 1,
            required_feature: if self.action_worlds {
                ACTION_WORLD_DECISION_FEATURE
            } else {
                CHAIN_DECISION_FEATURE
            }
            .into(),
            chain_record: result.chain.export()?,
            chain_identity: result.chain.identity().to_owned(),
            contract: contract.to_bytes(artifact_id)?,
            contract_identity: contract.identity().to_owned(),
            decision_fingerprint: decision_result_fingerprint(
                &result.decision,
                result.chain.identity(),
            )?,
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_CHAIN_DECISION_ARTIFACT_BYTES {
            return Err(ChainDecisionError::Coordinate(
                "chain decision artifact exceeds its byte limit",
            ));
        }
        Ok(bytes)
    }
}

/// Independently consume a chain decision: verify the chain record and
/// contract, recompile their joint six-layer program, reexecute its law and
/// decision, and compare the complete decision fingerprint.
///
/// `expected_program` is an optional identity retained outside the artifact.
///
/// # Errors
/// A foreign version or feature, a changed declaration/result, or any
/// compiler or execution refusal.
pub fn consume_chain_decision(
    bytes: &[u8],
    expected_program: Option<SemanticDigest>,
    ctx: &ExecutionContext,
) -> Result<ChainDecisionResult, ChainDecisionError> {
    if bytes.len() > MAX_CHAIN_DECISION_ARTIFACT_BYTES {
        return Err(ChainDecisionError::Coordinate(
            "chain decision artifact exceeds its byte limit",
        ));
    }
    let wire: ChainDecisionArtifactWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || !matches!(
            wire.required_feature.as_str(),
            CHAIN_DECISION_FEATURE | ACTION_WORLD_DECISION_FEATURE
        )
    {
        return Err(ChainDecisionError::Coordinate(
            "unsupported chain decision version or feature",
        ));
    }
    let recorded = PropagationChainRecordWire::decode(&wire.chain_record)
        .map_err(|error| ChainDecisionError::Chain(error.to_refusal()))?;
    recorded
        .consume(Some(&wire.chain_identity), ctx)
        .map_err(|error| ChainDecisionError::Chain(error.to_refusal()))?;
    let contract = DecisionContractArtifact::from_bytes(&wire.contract, &wire.contract_identity)?;
    let plan = if wire.required_feature == ACTION_WORLD_DECISION_FEATURE {
        compile_action_world_chain_decision(recorded, contract.contract().clone(), ctx)?
    } else {
        compile_chain_decision(recorded, contract.contract().clone(), ctx)?
    };
    let identity = plan.lowered().identity();
    if wire.lowered_program_identity != *identity.as_bytes()
        || expected_program.is_some_and(|expected| expected != identity)
    {
        return Err(ChainDecisionError::Coordinate(
            "chain decision program identity differs from the independently retained identity",
        ));
    }
    let result = plan.execute(ctx)?;
    if result.chain.identity() != wire.chain_identity
        || decision_result_fingerprint(&result.decision, result.chain.identity())?
            != wire.decision_fingerprint
    {
        return Err(ChainDecisionError::Coordinate(
            "chain decision replay differs from the stored result",
        ));
    }
    Ok(result)
}

struct ChainLawDemand<'a> {
    chain: &'a PropagationChain,
    coordinates: Vec<ScientificQuantity>,
    weights: Option<Vec<f64>>,
}

impl<'a> ChainLawDemand<'a> {
    fn new(
        chain: &'a PropagationChain,
        coordinates: Vec<ScientificQuantity>,
    ) -> Result<Self, ChainDecisionError> {
        let weights = if let Some((columns, weights)) = chain.joint_output_law() {
            if columns.len() != coordinates.len() {
                return Err(ChainDecisionError::Coordinate(
                    "joint law width differs from its coordinates",
                ));
            }
            weights
        } else {
            if coordinates.len() != 1 {
                return Err(ChainDecisionError::Coordinate(
                    "the joint output law was withdrawn or is unavailable",
                ));
            }
            chain
                .output_law()
                .ok_or(ChainDecisionError::Coordinate(
                    "the chain output law was withdrawn or is unavailable",
                ))?
                .1
        };
        Ok(Self {
            chain,
            coordinates,
            weights: weights
                .map(|values| values.iter().map(|bits| f64::from_bits(*bits)).collect()),
        })
    }
}

impl DemandSource for ChainLawDemand<'_> {
    fn coordinates(&self) -> &[ScientificQuantity] {
        &self.coordinates
    }

    fn receipt(&self) -> SourceReceipt {
        SourceReceipt {
            provider_id: "antecedent.propagation_chain".into(),
            snapshot_id: self.chain.identity().to_owned(),
            rng_id: if self.chain.record().construction == "independent_product" {
                "declared_independent_product"
            } else {
                "declared_aligned_draws"
            }
            .into(),
            causal_contract_id: declared_transport_receipt(self.chain)
                .unwrap_or("none_declared")
                .to_owned(),
        }
    }

    fn mean_estimate(&self) -> MeanEstimate {
        if self.chain.record().reading == ChainReading::Posterior.as_str() {
            MeanEstimate::Posterior
        } else {
            MeanEstimate::Frequentist
        }
    }

    fn means(&mut self) -> Result<Vec<f64>, DemandSourceError> {
        Err(DemandSourceError::new("a retained chain law is read through its draws"))
    }

    fn supplies_means(&self) -> bool {
        false
    }
    fn supplies_draws(&self) -> bool {
        true
    }
    fn joint(&self) -> bool {
        true
    }
    fn requires_joint_rows(&self) -> bool {
        true
    }
    fn answers_outcome_law(&self) -> bool {
        self.chain.record().reading == ChainReading::Predictive.as_str()
    }
    fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }
    fn draw_count(&self) -> usize {
        self.chain.joint_output_law().map_or_else(
            || self.chain.output_law().map_or(0, |(draws, _)| draws.len()),
            |(columns, _)| columns.first().map_or(0, Vec::len),
        )
    }

    fn fill_draws(
        &mut self,
        start: usize,
        rows: usize,
        out: &mut [f64],
    ) -> Result<(), DemandSourceError> {
        let end =
            start.checked_add(rows).ok_or_else(|| DemandSourceError::new("draw range overflow"))?;
        let width = self.coordinates.len();
        if out.len() != rows.saturating_mul(width) {
            return Err(DemandSourceError::new("one cell per coordinate and draw is required"));
        }
        if let Some((columns, _)) = self.chain.joint_output_law() {
            for (column, draws) in columns.iter().enumerate() {
                let selected = draws
                    .get(start..end)
                    .ok_or_else(|| DemandSourceError::new("draw range outside joint law"))?;
                for (row, bits) in selected.iter().enumerate() {
                    out[row * width + column] = f64::from_bits(*bits);
                }
            }
        } else {
            let (draws, _) = self
                .chain
                .output_law()
                .ok_or_else(|| DemandSourceError::new("output law unavailable"))?;
            let selected = draws
                .get(start..end)
                .ok_or_else(|| DemandSourceError::new("draw range outside retained law"))?;
            for (slot, bits) in out.iter_mut().zip(selected) {
                *slot = f64::from_bits(*bits);
            }
        }
        Ok(())
    }
}
