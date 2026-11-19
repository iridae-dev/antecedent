//! Joint mechanism-posterior and disturbance-predictive action worlds.
//!
//! This additive method fits all selected cyclic parameters from the same
//! intervention-labelled training rows. Each retained posterior parameter
//! vector is paired with one disturbance drawn from the model covariance;
//! every hard-clamp action world reads that same vector and disturbance. The
//! resulting joint outcome law enters the ordinary action-world decision
//! compiler. Its claim is conditional on the specified equations, covariance,
//! training likelihood and prior. No coverage or decision reliability is
//! promoted from the parameter sampler.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, ScientificQuantity};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, SemanticDigest,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{
    ActionKind, DecisionContract, DecisionContractError, DecisionCriterion,
};
use antecedent_estimate::cyclic_likelihood::{
    CyclicParameter, CyclicPosterior, sample_cyclic_posterior, with_parameters,
};
use antecedent_io::IoError;
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_kernels::{cholesky_spd_into, standard_normal};
use antecedent_model::nonlinear_equilibrium::{
    Clamp, EquilibriumError, EquilibriumModel, RegimeRow, SolvePolicy,
};
use antecedent_prob::metropolis::MetropolisOptions;
use serde::{Deserialize, Serialize};

use super::cyclic_action_world_decision::CyclicDecisionOutput;
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_action_world_chain_decision,
    consume_chain_decision,
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

const FEATURE: &str = "cyclic_posterior_action_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// The complete source-fixed fit, prior, posterior simulation and action
/// schedule. A decision is compared at one draw per fitted parameter vector.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicPosteriorDecisionRequest {
    /// Specified structural equation template and fixed disturbance covariance.
    pub model: EquilibriumModel,
    /// Jointly sampled parameter coordinates.
    pub parameters: Vec<CyclicParameter>,
    /// Independent equilibrium rows under known intervention regimes.
    pub training: Vec<RegimeRow>,
    /// Zero-mean independent normal prior standard deviation.
    pub prior_sd: f64,
    /// Independent MCMC chains.
    pub chains: usize,
    /// Discarded draws per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws_per_chain: usize,
    /// Isotropic random-walk proposal standard deviation.
    pub proposal_scale: f64,
    /// Distinct hard-clamp action worlds.
    pub worlds: Vec<Vec<Clamp>>,
    /// Numerical starting state, with no equilibrium-selection meaning.
    pub initial: Vec<f64>,
    /// Certified-solve policy used for each world and draw.
    pub solve_policy: SolvePolicy,
}

/// Refusal of a fitted joint model, world, decision or independent replay.
#[derive(Debug, thiserror::Error)]
pub enum CyclicPosteriorDecisionError {
    /// The declared schedule is outside this method's finite support.
    #[error("cyclic posterior decision: {0}")]
    Invalid(&'static str),
    /// Cyclic model, posterior likelihood or equilibrium refusal.
    #[error("cyclic posterior decision model: {0}")]
    Model(#[from] EquilibriumError),
    /// Multi-action contract refusal.
    #[error("cyclic posterior decision contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Derived joint-chain or utility refusal.
    #[error("cyclic posterior decision chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or decoding refusal.
    #[error("cyclic posterior decision artifact: {0}")]
    Io(#[from] IoError),
}

/// Retained six-layer program before fitting or predictive simulation.
#[derive(Clone, Debug)]
pub struct CompiledCyclicPosteriorDecision {
    request: CyclicPosteriorDecisionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Joint parameter and outcome-predictive law with its decision. Diagnostic
/// failure refuses publication; no nominal interval is claimed.
#[derive(Clone, Debug)]
pub struct CyclicPosteriorDecisionResult {
    /// Joint parameter posterior, including its convergence diagnostics.
    pub posterior: CyclicPosterior,
    /// Paired predictive action-world law and nonlinear utility decision.
    pub decision: ChainDecisionResult,
    /// Identity of the complete fitted model-to-decision program.
    pub lowered_program_identity: SemanticDigest,
}

/// Prospective population-outcome coverage input for one paired action world.
/// The final decision's reliability remains unmeasured.
#[derive(Clone, Debug)]
pub struct CyclicPosteriorPredictiveCalibrationCandidate {
    /// Complete fitted posterior, world solves and decision.
    pub result: CyclicPosteriorDecisionResult,
    /// Nearest-rank central 95% interval of one predictive outcome.
    pub interval: (f64, f64),
    /// Exact model, sampler, world and output scope for a future record.
    pub basis: CalibrationBasis,
}

/// Prospective action-selection reliability observation for a fixed design.
/// It is one complete fit, certified world solve and decision, not a measured
/// reliability claim. A future experiment must supply the true best action and
/// retain ties, refusals and diagnostic failures in its denominator.
#[derive(Clone, Debug)]
pub struct CyclicPosteriorDecisionReliabilityCandidate {
    /// Complete fitted model, joint worlds and decision, including its verdict.
    pub result: CyclicPosteriorDecisionResult,
    /// Fixed model template, training intervention design, sampler, worlds,
    /// coordinates and decision contract; excludes observed training values
    /// and the execution seed so independent replicates can share a scope.
    pub scope: SemanticDigest,
    /// Number of independent intervention-labelled training rows.
    pub training_rows: usize,
    /// Always `not_measured`: no reliability record is earned by this method.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CyclicPosteriorDecisionWire {
    version: u32,
    required_feature: String,
    request: CyclicPosteriorDecisionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    seed: u64,
    program_identity: [u8; 32],
    contract_identity: String,
    contract_artifact: Vec<u8>,
    posterior_digest: [u8; 32],
    chain_artifact: Vec<u8>,
}

/// Compile a joint cyclic parameter posterior, paired predictive action worlds
/// and the target-population decision into one six-layer program. The prior,
/// likelihood, RNG seed, solve policy, actions and training rows enter its
/// identity before any sampler or solver runs.
///
/// # Errors
/// Invalid worlds, coordinates, posterior design, model or decision contract.
pub fn compile_cyclic_posterior_action_decision(
    request: CyclicPosteriorDecisionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicPosteriorDecision, CyclicPosteriorDecisionError> {
    compile_cyclic_posterior_action_decision_checked(request, outputs, contract, ctx, false)
}

// Only the component compiler calls this after checking every output coordinate
// against its corresponding original produced port.
pub(super) fn compile_cyclic_component_posterior_decision(
    request: CyclicPosteriorDecisionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicPosteriorDecision, CyclicPosteriorDecisionError> {
    compile_cyclic_posterior_action_decision_checked(request, outputs, contract, ctx, true)
}

fn compile_cyclic_posterior_action_decision_checked(
    request: CyclicPosteriorDecisionRequest,
    outputs: Vec<CyclicDecisionOutput>,
    contract: DecisionContract,
    ctx: &ExecutionContext,
    checked_component_outputs: bool,
) -> Result<CompiledCyclicPosteriorDecision, CyclicPosteriorDecisionError> {
    contract.validate()?;
    request.model.validate()?;
    let n = request.model.order();
    let draws = request
        .chains
        .checked_mul(request.draws_per_chain)
        .ok_or(CyclicPosteriorDecisionError::Invalid("posterior draw count overflow"))?;
    if request.parameters.is_empty()
        || request.parameters.len() > 128
        || draws.checked_mul(request.parameters.len()).is_none_or(|cells| cells > 1_000_000)
        || request.training.is_empty()
        || !request.prior_sd.is_finite()
        || request.prior_sd <= 0.0
        || request.chains < 2
        || request.draws_per_chain < 100
        || draws > 8192
        || request.warmup == 0
        || !request.proposal_scale.is_finite()
        || request.proposal_scale <= 0.0
        || request.initial.len() != n
        || request.initial.iter().any(|x| !x.is_finite())
        || outputs.len() < 2
        || outputs.len() != request.worlds.len()
        || outputs.len() != contract.actions.len()
        || request.worlds.iter().any(Vec::is_empty)
    {
        return Err(CyclicPosteriorDecisionError::Invalid("invalid fitted cyclic action schedule"));
    }
    antecedent_estimate::cyclic_likelihood::parameter_values(&request.model, &request.parameters)?;
    let mut seen_worlds = BTreeSet::new();
    let mut seen_regimes = BTreeSet::new();
    for output in &outputs {
        let equation = request
            .model
            .equations
            .get(output.equation)
            .ok_or(CyclicPosteriorDecisionError::Invalid("output equation is absent"))?;
        if !seen_worlds.insert(output.world)
            || request.worlds.get(output.world).is_none()
            || (!checked_component_outputs && equation.id != output.quantity.variable_id)
            || output.quantity.role != "outcome"
            || output.quantity.population_id != contract.target_population
            || output.quantity.horizon != contract.horizon
            || !seen_regimes.insert(output.quantity.regime_id.as_str())
        {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "output world or coordinate differs",
            ));
        }
    }
    if request.worlds.iter().enumerate().any(|(i, world)| request.worlds[..i].contains(world)) {
        return Err(CyclicPosteriorDecisionError::Invalid("action worlds have identical clamps"));
    }
    let mut action_regimes = BTreeSet::new();
    for action in &contract.actions {
        if !matches!(action.kind, ActionKind::Intervention | ActionKind::Regime)
            || action.inputs.len() != 1
            || !action_regimes.insert(action.inputs[0].regime_id.as_str())
            || !outputs.iter().any(|output| {
                ScientificQuantity::try_from(output.quantity.clone()).is_ok_and(|quantity| {
                    action.inputs[0].require_same_coordinate(&quantity).is_ok()
                })
            })
        {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "decision action does not name one paired world",
            ));
        }
    }
    if action_regimes != seen_regimes {
        return Err(CyclicPosteriorDecisionError::Invalid("decision omits a paired world"));
    }
    let request_bytes = antecedent_io::to_cbor(&request)?;
    let outputs_bytes = antecedent_io::to_cbor(&outputs)?;
    let contract_identity = contract.identity()?;
    let seed = ctx.rng.master_seed();
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.cyclic_posterior_action_decision.source.v1",
        &request_bytes,
    );
    let mut writer =
        CanonicalWriter::new("antecedent.cyclic_posterior_action_decision.question.v1");
    writer.bytes(&source_digest).bytes(&outputs_bytes).text(&contract_identity).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(request.training.len() as u64, n as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query,
        },
        plan_id: Arc::from("cyclic_posterior_action_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "specified_cyclic_scm_joint_likelihood",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let prior =
        Prior::new(format!("cyclic_normal_parameter_prior:{:016x}", request.prior_sd.to_bits()))
            .map_err(|_| CyclicPosteriorDecisionError::Invalid("invalid prior identity"))?;
    let likelihood = LikelihoodModel::new("joint_cyclic_equilibrium_regime_likelihood")
        .map_err(|_| CyclicPosteriorDecisionError::Invalid("invalid likelihood identity"))?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Bayesian(OrdinaryBayesianLikelihood::new(
            likelihood,
            prior.clone(),
        )),
        models: Arc::from([ModelSlot {
            role: ModelRole::Mechanism,
            family: Arc::from("joint_cyclic_parameter_posterior"),
            inputs: Arc::from([]),
            output: None,
        }]),
        folds: None,
        resampling: None,
        posterior: Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::RandomWalkMetropolis,
            draws,
            prior,
        }),
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::JointScenarios, digest: Some(query) },
        ]),
        composition: Composition::Chained { stages: 3 },
        decision: Some(DecisionProgram {
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
            scenario_source: Some(Arc::from("joint_mechanism_posterior_and_shared_disturbance_v1")),
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(request_bytes.len() as u64))
                .build()
                .map_err(|_| {
                    CyclicPosteriorDecisionError::Invalid("invalid training read operator")
                })?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|_| CyclicPosteriorDecisionError::Invalid("invalid joint fit operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(|_| {
                CyclicPosteriorDecisionError::Invalid("invalid action-world operator")
            })?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::CyclicPosteriorActionDecision,
            scientific,
            identification,
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
    Ok(CompiledCyclicPosteriorDecision { request, outputs, contract, seed, lowered })
}

impl CompiledCyclicPosteriorDecision {
    /// Complete compiled model, fitted uncertainty, action worlds and utility.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the retained six-layer program before sampling or solving.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    pub(super) fn joint_chain(
        &self,
        outcomes: &[Vec<f64>],
    ) -> Result<PropagationChainRecordWire, CyclicPosteriorDecisionError> {
        self.joint_chain_with_alignment(outcomes, &self.lowered.identity().to_hex())
    }

    pub(super) fn joint_chain_with_alignment(
        &self,
        outcomes: &[Vec<f64>],
        alignment: &str,
    ) -> Result<PropagationChainRecordWire, CyclicPosteriorDecisionError> {
        let mut sources = Vec::with_capacity(self.outputs.len());
        let mut stages = Vec::with_capacity(self.outputs.len());
        let mut joint = Vec::with_capacity(self.outputs.len());
        for (index, output) in self.outputs.iter().enumerate() {
            let values = outcomes.get(index).ok_or(CyclicPosteriorDecisionError::Invalid(
                "a paired world lost its outcome column",
            ))?;
            let source_id = format!("cyclic.posterior.world.{index}");
            let stage_id = format!("cyclic.posterior.outcome.{index}");
            sources.push(
                ChainSourceWire::draws(&source_id, LawMeaning::Predictive, alignment, values)
                    .with_factor(SharedFactorKind::Model, alignment)
                    .with_factor(SharedFactorKind::Disturbance, alignment),
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

    /// Sample the joint mechanism posterior, draw one population disturbance
    /// per retained parameter vector and solve all action worlds on that same
    /// pair. Diagnostics and certified uniqueness must pass before a decision
    /// is published. No coverage or frequentist reliability is asserted.
    ///
    /// # Errors
    /// Posterior, covariance, equilibrium or downstream decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicPosteriorDecisionResult, CyclicPosteriorDecisionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(CyclicPosteriorDecisionError::Invalid("posterior execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let request = &self.request;
        let posterior = sample_cyclic_posterior(
            &request.model,
            &request.parameters,
            &request.training,
            request.prior_sd,
            MetropolisOptions {
                chains: request.chains,
                warmup: request.warmup,
                draws: request.draws_per_chain,
                proposal_scale: request.proposal_scale,
            },
            &scoped,
        )?;
        if !posterior.diagnostics_pass {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "joint mechanism posterior diagnostics did not pass",
            ));
        }
        let n = request.model.order();
        let mut chol = vec![0.0; n * n];
        cholesky_spd_into(&request.model.noise_covariance, n, &mut chol).map_err(|_| {
            CyclicPosteriorDecisionError::Invalid("disturbance covariance is not positive definite")
        })?;
        let mut rng = scoped.rng.stream(0xC1C1_5EED);
        let mut outcomes = vec![Vec::with_capacity(posterior.samples.len()); self.outputs.len()];
        for parameters in &posterior.samples {
            let model = with_parameters(&request.model, &request.parameters, parameters)?;
            let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let noise: Vec<f64> =
                (0..n).map(|i| (0..=i).map(|j| chol[i * n + j] * z[j]).sum()).collect();
            for (index, output) in self.outputs.iter().enumerate() {
                let solved = model.solve(
                    &noise,
                    &request.worlds[output.world],
                    &request.initial,
                    request.solve_policy,
                    &scoped,
                )?;
                if !solved.globally_unique
                    || !solved.existence_certified
                    || !solved.numerical_converged
                {
                    return Err(CyclicPosteriorDecisionError::Invalid(
                        "a predictive action world lacks a certified unique equilibrium",
                    ));
                }
                outcomes[index].push(*solved.values.get(output.equation).ok_or(
                    CyclicPosteriorDecisionError::Invalid("a solved world lost its outcome"),
                )?);
            }
        }
        let decision = compile_action_world_chain_decision(
            self.joint_chain(&outcomes)?,
            self.contract.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(CyclicPosteriorDecisionResult {
            posterior,
            decision,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Fixed design identity for a prospective action-selection experiment.
    /// It is available before fitting, so a refused replicate remains in the
    /// same experiment rather than being silently discarded.
    ///
    /// # Errors
    /// The criterion or constraints are outside this prospective design.
    pub fn decision_reliability_scope(
        &self,
    ) -> Result<SemanticDigest, CyclicPosteriorDecisionError> {
        if !matches!(
            self.contract.criterion,
            DecisionCriterion::PosteriorExpectedUtility | DecisionCriterion::ExpectedRegret
        ) || !self.contract.constraints.is_empty()
        {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "action reliability needs unconstrained expected utility or expected regret",
            ));
        }
        let mut writer = CanonicalWriter::new(
            "antecedent.cyclic_posterior_action_decision.reliability_scope.v1",
        );
        writer
            .bytes(&antecedent_io::to_cbor(&self.request.model)?)
            .bytes(&antecedent_io::to_cbor(&self.request.parameters)?)
            .word(self.request.prior_sd.to_bits())
            .word(self.request.chains as u64)
            .word(self.request.warmup as u64)
            .word(self.request.draws_per_chain as u64)
            .word(self.request.proposal_scale.to_bits())
            .word(self.request.training.len() as u64);
        for row in &self.request.training {
            writer.bytes(&antecedent_io::to_cbor(&row.clamps)?);
        }
        writer
            .bytes(&antecedent_io::to_cbor(&self.request.worlds)?)
            .bytes(&antecedent_io::to_cbor(&self.outputs)?)
            .bytes(&antecedent_io::to_cbor(&self.request.initial)?)
            .bytes(&antecedent_io::to_cbor(&self.request.solve_policy)?)
            .text(&self.contract.identity()?);
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }

    /// Execute the whole fitted action route as a prospective reliability
    /// observation. This supports unconstrained expected utility or expected
    /// regret, whose best action can be compared with a separately specified
    /// population truth. No confidence or selection guarantee is asserted.
    ///
    /// # Errors
    /// An unsupported criterion, hard constraint, fit, diagnostic or world solve.
    pub fn decision_reliability_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicPosteriorDecisionReliabilityCandidate, CyclicPosteriorDecisionError> {
        let scope = self.decision_reliability_scope()?;
        let result = self.execute(ctx)?;
        Ok(CyclicPosteriorDecisionReliabilityCandidate {
            result,
            scope,
            training_rows: self.request.training.len(),
            calibration: "not_measured",
        })
    }

    /// Expose the final predictive outcome for a prespecified action world as
    /// a prospective whole-route coverage candidate. The interval has no
    /// measured status, and this does not calibrate the action choice.
    ///
    /// # Errors
    /// A fit, diagnostic, world solve or malformed predictive table refuses.
    pub fn calibration_candidate(
        &self,
        output_index: usize,
        ctx: &ExecutionContext,
    ) -> Result<CyclicPosteriorPredictiveCalibrationCandidate, CyclicPosteriorDecisionError> {
        if output_index >= self.outputs.len() {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "predictive calibration output is absent",
            ));
        }
        let result = self.execute(ctx)?;
        let source = result.decision.chain.record().sources.get(output_index).ok_or(
            CyclicPosteriorDecisionError::Invalid("predictive action world source is absent"),
        )?;
        let mut ordered: Vec<f64> = source.values.iter().copied().map(f64::from_bits).collect();
        if ordered.len() < 200 || ordered.iter().any(|value| !value.is_finite()) {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "predictive calibration needs finite joint draws",
            ));
        }
        ordered.sort_by(f64::total_cmp);
        let rank = |numerator: usize| (ordered.len() * numerator).div_ceil(1000).saturating_sub(1);
        let interval = (ordered[rank(25)], ordered[rank(975)]);
        let mut writer =
            CanonicalWriter::new("antecedent.cyclic_posterior_predictive.calibration_scope.v1");
        writer
            .bytes(&antecedent_io::to_cbor(&self.request.model)?)
            .bytes(&antecedent_io::to_cbor(&self.request.parameters)?)
            .word(self.request.prior_sd.to_bits())
            .word(self.request.chains as u64)
            .word(self.request.warmup as u64)
            .word(self.request.draws_per_chain as u64)
            .word(self.request.proposal_scale.to_bits())
            .bytes(&antecedent_io::to_cbor(&self.request.worlds)?)
            .bytes(&antecedent_io::to_cbor(&self.outputs)?)
            .bytes(&antecedent_io::to_cbor(&self.request.initial)?)
            .bytes(&antecedent_io::to_cbor(&self.request.solve_policy)?)
            .text(&self.contract.identity()?);
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("population_outcome:world={output_index}:scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "OutcomePrediction",
                "Cyclic",
                "fixed",
                "multi_env",
                "Bayesian",
                FEATURE,
                "nearest_rank_95",
                "",
                "independent_training_future_units",
                "joint_cyclic_metropolis_gaussian_disturbance_v1",
                functional.as_str(),
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            self.request.training.len() as u64,
            None,
            Some(ordered.len() as u32),
            0.0,
        )
        .with_validation("none");
        Ok(CyclicPosteriorPredictiveCalibrationCandidate { result, interval, basis })
    }

    /// Re-evaluate before sealing the complete model and decision as one
    /// portable artifact. A fresh consumer re-runs the posterior and solves.
    ///
    /// # Errors
    /// Changed results, unsupported artifact or replay refusal.
    pub fn export(
        &self,
        result: &CyclicPosteriorDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicPosteriorDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.posterior != live.posterior
            || result.decision.chain.identity() != live.decision.chain.identity()
            || result.decision.decision != live.decision.decision
        {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "posterior action decision differs on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let inner = compile_action_world_chain_decision(
            live.decision.chain.record().clone(),
            self.contract.clone(),
            ctx,
        )?;
        let posterior_digest = CanonicalWriter::digest_bytes(
            "antecedent.cyclic_posterior_action_decision.posterior.v1",
            &antecedent_io::to_cbor(&live.posterior)?,
        );
        let wire = CyclicPosteriorDecisionWire {
            version: 1,
            required_feature: FEATURE.into(),
            request: self.request.clone(),
            outputs: self.outputs.clone(),
            seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes(artifact_id)?,
            posterior_digest,
            chain_artifact: inner.export(&live.decision, artifact_id)?,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CyclicPosteriorDecisionError::Invalid(
                "posterior action artifact exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recompile the fitted mechanism and action schedule from bytes, regenerate
/// the paired predictive worlds and independently consume the decision under
/// a program identity held outside the artifact.
///
/// # Errors
/// A foreign feature, altered premise, diagnostic or decision replay.
pub fn consume_cyclic_posterior_action_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CyclicPosteriorDecisionResult, CyclicPosteriorDecisionError> {
    consume_cyclic_posterior_action_decision_checked(bytes, expected_program, ctx, false)
}

pub(super) fn consume_cyclic_component_posterior_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CyclicPosteriorDecisionResult, CyclicPosteriorDecisionError> {
    consume_cyclic_posterior_action_decision_checked(bytes, expected_program, ctx, true)
}

fn consume_cyclic_posterior_action_decision_checked(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
    checked_component_outputs: bool,
) -> Result<CyclicPosteriorDecisionResult, CyclicPosteriorDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(CyclicPosteriorDecisionError::Invalid(
            "posterior action artifact exceeds its bound",
        ));
    }
    let wire: CyclicPosteriorDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(CyclicPosteriorDecisionError::Invalid(
            "posterior action feature, program or seed differs",
        ));
    }
    let contract =
        DecisionContractArtifact::from_bytes(&wire.contract_artifact, &wire.contract_identity)?;
    let plan = compile_cyclic_posterior_action_decision_checked(
        wire.request,
        wire.outputs,
        contract.contract().clone(),
        ctx,
        checked_component_outputs,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(CyclicPosteriorDecisionError::Invalid(
            "recompiled posterior action program differs",
        ));
    }
    let result = plan.execute(ctx)?;
    let posterior_digest = CanonicalWriter::digest_bytes(
        "antecedent.cyclic_posterior_action_decision.posterior.v1",
        &antecedent_io::to_cbor(&result.posterior)?,
    );
    let inner = consume_chain_decision(
        &wire.chain_artifact,
        Some(result.decision.lowered_program_identity),
        ctx,
    )?;
    if posterior_digest != wire.posterior_digest
        || inner.chain.identity() != result.decision.chain.identity()
        || inner.decision != result.decision.decision
    {
        return Err(CyclicPosteriorDecisionError::Invalid(
            "fitted posterior or paired decision changed on replay",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_design::decision_contract::{
        DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_io::quantity_wire::ScientificQuantityWire;
    use antecedent_model::nonlinear_equilibrium::{Activation, Equation};

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

    fn request() -> CyclicPosteriorDecisionRequest {
        let training = (0..260)
            .map(|i| {
                let ad = (i % 5) as f64 * 0.4;
                let ew = (i as f64 * 0.19).sin() * 0.2;
                let et = (i as f64 * 0.23).cos() * 0.2;
                let wishlists = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06);
                let traffic = 0.2 * wishlists + et;
                RegimeRow {
                    values: vec![wishlists, traffic, ad],
                    clamps: vec![Clamp { variable: 2, value: ad }],
                }
            })
            .collect();
        CyclicPosteriorDecisionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "wishlists".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.3, 0.2],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "traffic".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "ads".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.0, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                ],
                noise_covariance: vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
            },
            parameters: vec![CyclicParameter::Weight { equation: 0, parent: 1 }],
            training,
            prior_sd: 1.0,
            chains: 2,
            warmup: 300,
            draws_per_chain: 300,
            proposal_scale: 0.08,
            worlds: vec![
                vec![Clamp { variable: 2, value: 0.0 }],
                vec![Clamp { variable: 2, value: 1.0 }],
            ],
            initial: vec![0.0; 3],
            solve_policy: SolvePolicy::default(),
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
    #[ignore = "prospective source-only reliability grid; no selection record is earned"]
    fn prospective_action_selection_grid_keeps_failures_and_design_scope() {
        use antecedent_design::decision_eval::Verdict;
        let mut refused = 0_usize;
        let mut selected_true_best = 0_usize;
        let mut tied_or_wrong = 0_usize;
        let mut scope = None;
        for replicate in 0..4_u64 {
            let mut declaration = request();
            let mut rng = antecedent_core::CausalRng::from_seed(10_000 + replicate);
            for row in &mut declaration.training {
                let ad = row.clamps[0].value;
                let ew = standard_normal(&mut rng) * 0.2;
                let et = standard_normal(&mut rng) * 0.2;
                let wishlist = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06);
                row.values = vec![wishlist, 0.2 * wishlist + et, ad];
            }
            // The planted linear system has positive mean promotion effect;
            // training outcomes and fit streams vary across replicates.
            let ctx = ExecutionContext::for_tests(47 + replicate);
            let plan =
                compile_cyclic_posterior_action_decision(declaration, outputs(), contract(), &ctx)
                    .expect("compile fixed grid design");
            let design_scope = plan.decision_reliability_scope().expect("fixed design scope");
            if let Some(previous) = scope {
                assert_eq!(design_scope, previous);
            }
            scope = Some(design_scope);
            match plan.decision_reliability_candidate(&ctx) {
                Ok(candidate) => {
                    assert_eq!(candidate.calibration, "not_measured");
                    assert_eq!(candidate.training_rows, 260);
                    assert_eq!(candidate.scope, design_scope);
                    match &candidate.result.decision.decision.verdict {
                        Verdict::UniquelyOptimal(action) if action == "promoted" => {
                            selected_true_best += 1;
                        }
                        _ => tied_or_wrong += 1,
                    }
                }
                Err(_) => refused += 1,
            }
        }
        // Failures and ties remain observations, never silently dropped.
        assert_eq!(selected_true_best + tied_or_wrong + refused, 4);
    }

    #[test]
    fn joint_cyclic_parameter_and_disturbance_worlds_replay() {
        let ctx = ExecutionContext::for_tests(47);
        let plan = compile_cyclic_posterior_action_decision(request(), outputs(), contract(), &ctx)
            .expect("posterior action compiler");
        assert_eq!(plan.plan_view().route().as_str(), "CyclicPosteriorActionDecision");
        let result = plan.execute(&ctx).expect("joint fitted predictive worlds");
        assert!(result.posterior.diagnostics_pass);
        assert_eq!(result.decision.decision.n_draws, 600);
        let bytes =
            plan.export(&result, "cyclic-fitted-worlds", &ctx).expect("portable fitted decision");
        let replay =
            consume_cyclic_posterior_action_decision(&bytes, plan.lowered().identity(), &ctx)
                .expect("independent fitted decision replay");
        assert_eq!(replay.decision.decision, result.decision.decision);
    }
}
