//! Jointly fitted feedback mechanisms standardized to a target baseline frame.
//!
//! Selection is confined to exogenous baseline coordinates. The mechanism and
//! disturbance law are invariant across populations; this is a declared model
//! premise, not a general transport identification theorem. Every action uses
//! the same posterior parameter, target row, and disturbance draw.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_estimate::cyclic_likelihood::{
    CyclicPosterior, sample_cyclic_posterior, with_parameters,
};
use antecedent_io::IoError;
use antecedent_kernels::{cholesky_spd_into, standard_normal};
use antecedent_model::nonlinear_equilibrium::{Clamp, EquilibriumError};
use antecedent_prob::metropolis::MetropolisOptions;
use serde::{Deserialize, Serialize};

use super::cyclic_action_world_decision::CyclicDecisionOutput;
use super::cyclic_posterior_action_decision::{
    CompiledCyclicPosteriorDecision, CyclicPosteriorDecisionError, CyclicPosteriorDecisionRequest,
    compile_cyclic_posterior_action_decision,
};
use super::propagation_chain_decision::{
    ChainDecisionResult, compile_action_world_chain_decision, consume_chain_decision,
};
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramCell, ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{NestedFitLaw, Realization};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "cyclic_target_feedback_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Independent target units with complete values of the selected exogenous
/// baseline coordinates. The ordered frame is empirical and unweighted.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetFeedbackFrame {
    /// Name of the source sample; it must differ from the target sample.
    pub source_sample: String,
    /// Name of the target sample the frame was drawn from.
    pub target_sample: String,
    /// Target population the frame represents; it must match the decision contract.
    pub target_population: String,
    /// Ascending dense indices of the only mechanisms permitted to differ in
    /// distribution between source and target populations.
    pub baseline_indices: Vec<usize>,
    /// Identifier of each target unit, in row order.
    pub unit_ids: Vec<String>,
    /// One complete baseline vector per target unit, in index order.
    pub rows: Vec<Vec<f64>>,
}

/// One shared-data cyclic fit, target standardization, paired worlds and utility.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicTargetFeedbackRequest {
    /// The shared-data cyclic posterior decision request fitted on the source population.
    pub source: CyclicPosteriorDecisionRequest,
    /// The target frame whose baseline coordinates are standardized over.
    pub target: TargetFeedbackFrame,
    /// Output equations whose paired-world outcomes feed the decision.
    pub outputs: Vec<CyclicDecisionOutput>,
}

/// Failure to compile, execute or export a cyclic target feedback decision.
#[derive(Debug, thiserror::Error)]
pub enum CyclicTargetFeedbackError {
    /// The request, target frame or execution state is invalid.
    #[error("cyclic target feedback: {0}")]
    Invalid(&'static str),
    /// The source cyclic posterior decision refused.
    #[error("cyclic target feedback source: {0}")]
    Source(#[from] CyclicPosteriorDecisionError),
    /// The cyclic model could not be solved.
    #[error("cyclic target feedback model: {0}")]
    Model(#[from] EquilibriumError),
    /// Reading or writing the artifact failed.
    #[error("cyclic target feedback artifact: {0}")]
    Io(#[from] IoError),
}

/// Compiled cyclic target feedback decision bound to its request and seed.
pub struct CompiledCyclicTargetFeedbackDecision {
    request: CyclicTargetFeedbackRequest,
    contract: DecisionContract,
    child: CompiledCyclicPosteriorDecision,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Posterior, decision and lowered-program identity from one execution.
pub struct CyclicTargetFeedbackResult {
    /// Cyclic posterior fitted on the source data.
    pub posterior: CyclicPosterior,
    /// Decision over the target-standardized paired intervention worlds.
    pub decision: ChainDecisionResult,
    /// Identity of the lowered program that produced this result.
    pub lowered_program_identity: SemanticDigest,
    /// Calibration status of the result; always `"not_measured"`.
    pub calibration: &'static str,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    version: u32,
    feature: String,
    request: CyclicTargetFeedbackRequest,
    contract_identity: String,
    contract_artifact: Vec<u8>,
    seed: u64,
    program_identity: [u8; 32],
    posterior_digest: [u8; 32],
    chain_artifact: Vec<u8>,
}

#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
fn validate_frame(
    request: &CyclicTargetFeedbackRequest,
    contract: &DecisionContract,
) -> Result<(), CyclicTargetFeedbackError> {
    let frame = &request.target;
    let n = request.source.model.order();
    if frame.source_sample.trim().is_empty()
        || frame.target_sample.trim().is_empty()
        || frame.source_sample == frame.target_sample
        || frame.target_population != contract.target_population
        || frame.baseline_indices.is_empty()
        || frame.rows.len() < 2
        || frame.rows.len() != frame.unit_ids.len()
        || frame.rows.len() > 1_000_000
    {
        return Err(CyclicTargetFeedbackError::Invalid("target frame or population is invalid"));
    }
    let mut previous = None;
    for &index in &frame.baseline_indices {
        if index >= n || previous.is_some_and(|last| index <= last) {
            return Err(CyclicTargetFeedbackError::Invalid(
                "baseline indices must be ascending and unique",
            ));
        }
        previous = Some(index);
        let equation = &request.source.model.equations[index];
        if !equation.reads()?.is_empty()
            || request.source.parameters.iter().any(|parameter| match parameter {
                antecedent_estimate::cyclic_likelihood::CyclicParameter::Intercept {
                    equation: e,
                }
                | antecedent_estimate::cyclic_likelihood::CyclicParameter::ExpressionParameter {
                    equation: e,
                    ..
                }
                | antecedent_estimate::cyclic_likelihood::CyclicParameter::Weight {
                    equation: e,
                    ..
                } => *e == index,
            })
        {
            return Err(CyclicTargetFeedbackError::Invalid(
                "selected baseline is not an exogenous fixed mechanism",
            ));
        }
        for other in 0..n {
            if other != index
                && (request.source.model.noise_covariance[index * n + other] != 0.0
                    || request.source.model.noise_covariance[other * n + index] != 0.0)
            {
                return Err(CyclicTargetFeedbackError::Invalid(
                    "baseline and feedback disturbance covariance must be zero",
                ));
            }
        }
        if request.source.training.iter().any(|row| {
            row.values.get(index).is_none_or(|value| !value.is_finite())
                || !row.clamps.iter().any(|clamp| {
                    clamp.variable == index && clamp.value.to_bits() == row.values[index].to_bits()
                })
        }) {
            return Err(CyclicTargetFeedbackError::Invalid(
                "source baseline must be observed and clamped in every training row",
            ));
        }
        if request
            .source
            .worlds
            .iter()
            .any(|world| world.iter().any(|clamp| clamp.variable == index))
        {
            return Err(CyclicTargetFeedbackError::Invalid(
                "actions may not replace target baselines",
            ));
        }
        if request.outputs.iter().any(|output| output.equation == index) {
            return Err(CyclicTargetFeedbackError::Invalid(
                "reported feedback outcome may not be a selected baseline",
            ));
        }
    }
    let mut ids = BTreeSet::new();
    for (row, id) in frame.rows.iter().zip(&frame.unit_ids) {
        if id.trim().is_empty()
            || !ids.insert(id)
            || row.len() != frame.baseline_indices.len()
            || row.iter().any(|value| !value.is_finite())
        {
            return Err(CyclicTargetFeedbackError::Invalid(
                "target units or covariates are malformed",
            ));
        }
        for (&value, &index) in row.iter().zip(&frame.baseline_indices) {
            let mut source = request.source.training.iter().map(|unit| unit.values[index]);
            let first = source
                .next()
                .ok_or(CyclicTargetFeedbackError::Invalid("source baseline is empty"))?;
            let (lo, hi) = source.fold((first, first), |(lo, hi), x| (lo.min(x), hi.max(x)));
            if value < lo || value > hi {
                return Err(CyclicTargetFeedbackError::Invalid(
                    "target baseline exceeds observed source range",
                ));
            }
        }
    }
    Ok(())
}

/// Compile one target-baseline-shifted joint feedback fit and paired decision.
/// The target rows and selection restriction are bound before fitting.
///
/// # Errors
/// A source or frame that is invalid, a contract that is refused, or a request that
/// fails to encode.
pub fn compile_cyclic_target_feedback_decision(
    request: CyclicTargetFeedbackRequest,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicTargetFeedbackDecision, CyclicTargetFeedbackError> {
    let child = compile_cyclic_posterior_action_decision(
        request.source.clone(),
        request.outputs.clone(),
        contract.clone(),
        ctx,
    )?;
    validate_frame(&request, &contract)?;
    let bytes = antecedent_io::to_cbor(&request)?;
    let source_digest =
        CanonicalWriter::digest_bytes("antecedent.cyclic_target_feedback.source.v1", &bytes);
    let mut writer = CanonicalWriter::new("antecedent.cyclic_target_feedback.question.v1");
    writer
        .bytes(child.lowered().identity().as_bytes())
        .bytes(&source_digest)
        .text(&contract.identity().map_err(CyclicPosteriorDecisionError::Contract)?)
        .word(ctx.rng.master_seed());
    let query = writer.finish();
    let mut scientific: ScientificProblem = child.lowered().scientific().clone();
    scientific.target =
        ProblemTarget::QueryFunctional { kind: FunctionalKind::PolicyValue, outcome: None, query };
    scientific.plan_id = Arc::from(FEATURE);
    scientific.data = DataClassification::Tabular;
    scientific.graph_class = "target_baseline_shift_cyclic_joint_likelihood";
    scientific.source_digest = Some(source_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut statistical = child.lowered().statistical().clone();
    let design = DesignShape::quoted(
        request.source.training.len().saturating_add(request.target.rows.len()) as u64,
        request.source.model.order() as u64,
        true,
    );
    statistical.design = design;
    statistical.realization = Realization::NestedFit {
        producer_program: child.lowered().identity(),
        method: Arc::from("cyclic_posterior_action_decision_v1"),
        law: NestedFitLaw::Predictive,
    };
    let mut evidence = child.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources
        .push(EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) });
    evidence.sources = Arc::from(sources);
    evidence.composition = Composition::Chained { stages: 4 };
    let mut operators = child.lowered().physical().operators().to_vec();
    operators.push(
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(bytes.len() as u64))
            .build()
            .map_err(|_| {
                CyclicTargetFeedbackError::Invalid("target read operator cannot be lowered")
            })?,
    );
    operators.push(PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(|_| {
        CyclicTargetFeedbackError::Invalid("target world operator cannot be lowered")
    })?);
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::CyclicTargetFeedbackDecision,
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
    Ok(CompiledCyclicTargetFeedbackDecision {
        request,
        contract,
        child,
        seed: ctx.rng.master_seed(),
        lowered,
    })
}

impl CompiledCyclicTargetFeedbackDecision {
    /// The lowered program this decision executes.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Plan view of the lowered program at the compiled stage.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Fit the cyclic posterior, standardize over the target frame and score the decision.
    ///
    /// # Errors
    /// A changed execution seed, a source or model refusal, or an invalid target frame.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the draw is in [0, 1) and the row index is clamped to the last target row"
    )]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicTargetFeedbackResult, CyclicTargetFeedbackError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(CyclicTargetFeedbackError::Invalid("execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let source = &self.request.source;
        let posterior = sample_cyclic_posterior(
            &source.model,
            &source.parameters,
            &source.training,
            source.prior_sd,
            MetropolisOptions {
                chains: source.chains,
                warmup: source.warmup,
                draws: source.draws_per_chain,
                proposal_scale: source.proposal_scale,
            },
            &scoped,
        )?;
        if !posterior.diagnostics_pass {
            return Err(CyclicTargetFeedbackError::Invalid(
                "joint posterior diagnostics did not pass",
            ));
        }
        let n = source.model.order();
        let mut chol = vec![0.0; n * n];
        cholesky_spd_into(&source.model.noise_covariance, n, &mut chol).map_err(|_| {
            CyclicTargetFeedbackError::Invalid("disturbance covariance is not positive definite")
        })?;
        let mut rng = scoped.rng.stream(0xC1C1_7A29);
        let mut outcomes =
            vec![Vec::with_capacity(posterior.samples.len()); self.request.outputs.len()];
        for parameters in &posterior.samples {
            let model = with_parameters(&source.model, &source.parameters, parameters)?;
            let target_index = ((rng.next_f64() * self.request.target.rows.len() as f64) as usize)
                .min(self.request.target.rows.len() - 1);
            let target_row = &self.request.target.rows[target_index];
            let baseline: Vec<Clamp> = self
                .request
                .target
                .baseline_indices
                .iter()
                .zip(target_row)
                .map(|(&variable, &value)| Clamp { variable, value })
                .collect();
            let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
            let noise: Vec<f64> =
                (0..n).map(|i| (0..=i).map(|j| chol[i * n + j] * z[j]).sum()).collect();
            for (index, output) in self.request.outputs.iter().enumerate() {
                let mut clamps = source.worlds[output.world].clone();
                clamps.extend_from_slice(&baseline);
                let solved =
                    model.solve(&noise, &clamps, &source.initial, source.solve_policy, &scoped)?;
                if !solved.globally_unique
                    || !solved.existence_certified
                    || !solved.numerical_converged
                {
                    return Err(CyclicTargetFeedbackError::Invalid(
                        "target action world lacks certified unique equilibrium",
                    ));
                }
                outcomes[index].push(*solved.values.get(output.equation).ok_or(
                    CyclicTargetFeedbackError::Invalid("target outcome equation is absent"),
                )?);
            }
        }
        let chain =
            self.child.joint_chain_with_alignment(&outcomes, &self.lowered.identity().to_hex())?;
        let decision = compile_action_world_chain_decision(chain, self.contract.clone(), ctx)
            .map_err(CyclicPosteriorDecisionError::Chain)?
            .execute(ctx)
            .map_err(CyclicPosteriorDecisionError::Chain)?;
        Ok(CyclicTargetFeedbackResult {
            posterior,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Fixed design identity for an unearned action-choice study. It omits
    /// observed source values, target values, and the execution seed.
    ///
    /// # Errors
    /// A model, parameter or contract that fails to encode.
    pub fn decision_reliability_scope(&self) -> Result<SemanticDigest, CyclicTargetFeedbackError> {
        let source = &self.request.source;
        let mut writer =
            CanonicalWriter::new("antecedent.cyclic_target_feedback.reliability_scope.v1");
        writer
            .bytes(&antecedent_io::to_cbor(&source.model)?)
            .bytes(&antecedent_io::to_cbor(&source.parameters)?)
            .word(source.prior_sd.to_bits())
            .word(source.chains as u64)
            .word(source.warmup as u64)
            .word(source.draws_per_chain as u64)
            .word(source.proposal_scale.to_bits())
            .word(source.training.len() as u64)
            .word(self.request.target.rows.len() as u64)
            .bytes(&antecedent_io::to_cbor(&self.request.target.baseline_indices)?)
            .text(&self.request.target.target_population)
            .bytes(&antecedent_io::to_cbor(&source.worlds)?)
            .bytes(&antecedent_io::to_cbor(&self.request.outputs)?)
            .bytes(&antecedent_io::to_cbor(&source.initial)?)
            .bytes(&antecedent_io::to_cbor(&source.solve_policy)?)
            .text(&self.contract.identity().map_err(CyclicPosteriorDecisionError::Contract)?);
        for row in &source.training {
            let fixed: Vec<_> = row
                .clamps
                .iter()
                .filter(|clamp| !self.request.target.baseline_indices.contains(&clamp.variable))
                .collect();
            writer.bytes(&antecedent_io::to_cbor(&fixed)?);
            let baselines: Vec<_> = row
                .clamps
                .iter()
                .filter(|clamp| self.request.target.baseline_indices.contains(&clamp.variable))
                .map(|clamp| clamp.variable as u64)
                .collect();
            writer.words(&baselines);
        }
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }

    /// Export the result as a portable artifact after re-executing and matching it.
    ///
    /// # Errors
    /// A result that differs from the live execution, an oversized artifact or an
    /// encoding failure.
    pub fn export(
        &self,
        result: &CyclicTargetFeedbackResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicTargetFeedbackError> {
        let live = self.execute(ctx)?;
        if result.calibration != "not_measured"
            || result.lowered_program_identity != live.lowered_program_identity
            || result.posterior != live.posterior
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(CyclicTargetFeedbackError::Invalid(
                "target feedback result differs on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let chain = compile_action_world_chain_decision(
            live.decision.chain.record().clone(),
            self.contract.clone(),
            ctx,
        )
        .map_err(CyclicPosteriorDecisionError::Chain)?;
        let wire = Wire {
            version: 1,
            feature: FEATURE.into(),
            request: self.request.clone(),
            contract_identity: contract.identity().into(),
            contract_artifact: contract.to_bytes(artifact_id)?,
            seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            posterior_digest: CanonicalWriter::digest_bytes(
                "antecedent.cyclic_target_feedback.posterior.v1",
                &antecedent_io::to_cbor(&live.posterior)?,
            ),
            chain_artifact: chain
                .export(&live.decision, artifact_id)
                .map_err(CyclicPosteriorDecisionError::Chain)?,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CyclicTargetFeedbackError::Invalid("artifact exceeds bound"));
        }
        Ok(bytes)
    }
}

/// Recompile, refit, solve and reevaluate a target feedback decision from raw
/// embedded evidence under an independently retained compiler identity.
///
/// # Errors
/// An oversized or undecodable artifact, a feature, program or seed that differs, or a
/// posterior or decision that differs on replay.
pub fn consume_cyclic_target_feedback_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CyclicTargetFeedbackResult, CyclicTargetFeedbackError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(CyclicTargetFeedbackError::Invalid("artifact exceeds bound"));
    }
    let wire: Wire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(CyclicTargetFeedbackError::Invalid("feature, program or seed differs"));
    }
    let contract =
        DecisionContractArtifact::from_bytes(&wire.contract_artifact, &wire.contract_identity)?;
    let plan =
        compile_cyclic_target_feedback_decision(wire.request, contract.contract().clone(), ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(CyclicTargetFeedbackError::Invalid("compiler program differs"));
    }
    let live = plan.execute(ctx)?;
    if CanonicalWriter::digest_bytes(
        "antecedent.cyclic_target_feedback.posterior.v1",
        &antecedent_io::to_cbor(&live.posterior)?,
    ) != wire.posterior_digest
    {
        return Err(CyclicTargetFeedbackError::Invalid("posterior differs on replay"));
    }
    let chain = compile_action_world_chain_decision(
        live.decision.chain.record().clone(),
        plan.contract.clone(),
        ctx,
    )
    .map_err(CyclicPosteriorDecisionError::Chain)?;
    let consumed =
        consume_chain_decision(&wire.chain_artifact, Some(chain.lowered().identity()), ctx)
            .map_err(CyclicPosteriorDecisionError::Chain)?;
    if consumed.decision != live.decision.decision
        || consumed.chain.identity() != live.decision.chain.identity()
    {
        return Err(CyclicTargetFeedbackError::Invalid("decision differs on replay"));
    }
    Ok(live)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_estimate::cyclic_likelihood::CyclicParameter;
    use antecedent_io::quantity_wire::ScientificQuantityWire;
    use antecedent_model::nonlinear_equilibrium::{
        Activation, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
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

    #[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
    pub(crate) fn fixture() -> (CyclicTargetFeedbackRequest, DecisionContract) {
        let training = (0..260)
            .map(|i| {
                let ad = f64::from(i % 5) * 0.4;
                let baseline = f64::from(i % 9) * 0.2;
                let ew = (f64::from(i) * 0.19).sin() * 0.2;
                let et = (f64::from(i) * 0.23).cos() * 0.2;
                let wishlists = (0.2 * ad + 0.4 * baseline + ew + 0.3 * et) / (1.0 - 0.06);
                RegimeRow {
                    values: vec![wishlists, 0.2 * wishlists + et, ad, baseline],
                    clamps: vec![
                        Clamp { variable: 2, value: ad },
                        Clamp { variable: 3, value: baseline },
                    ],
                }
            })
            .collect();
        let source = CyclicPosteriorDecisionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation {
                        id: "wishlists".into(),
                        intercept: 0.0,
                        weights: vec![0.0, 0.3, 0.2, 0.4],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "traffic".into(),
                        intercept: 0.0,
                        weights: vec![0.2, 0.0, 0.0, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "ads".into(),
                        intercept: 0.0,
                        weights: vec![0.0; 4],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                    Equation {
                        id: "baseline".into(),
                        intercept: 0.0,
                        weights: vec![0.0; 4],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: Vec::new(),
                    },
                ],
                noise_covariance: vec![
                    0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0,
                    0.04,
                ],
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
            initial: vec![0.0; 4],
            solve_policy: SolvePolicy::default(),
        };
        let outputs = ["ordinary", "promoted"]
            .iter()
            .enumerate()
            .map(|(world, regime)| CyclicDecisionOutput {
                world,
                equation: 0,
                quantity: ScientificQuantityWire::from(&quantity(regime)),
            })
            .collect();
        let decision = DecisionContract {
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
        };
        let target = TargetFeedbackFrame {
            source_sample: "trial".into(),
            target_sample: "target-frame".into(),
            target_population: "game".into(),
            baseline_indices: vec![3],
            unit_ids: vec!["t0".into(), "t1".into()],
            rows: vec![vec![0.2], vec![1.4]],
        };
        (CyclicTargetFeedbackRequest { source, target, outputs }, decision)
    }

    #[test]
    #[ignore = "source-only compiler and support assertions; no calibration earned"]
    fn compiler_identity_population_support_and_study_scope() {
        let ctx = ExecutionContext::for_tests(47);
        let (request, contract) = fixture();
        let plan = compile_cyclic_target_feedback_decision(request.clone(), contract.clone(), &ctx)
            .unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "CyclicTargetFeedbackDecision");
        let scope = plan.decision_reliability_scope().unwrap();
        let mut replicate = request.clone();
        for row in &mut replicate.source.training {
            row.values[3] = 1.6 - row.values[3];
            row.clamps[1].value = row.values[3];
        }
        replicate.target.rows = vec![vec![0.4], vec![1.2]];
        let other =
            compile_cyclic_target_feedback_decision(replicate, contract.clone(), &ctx).unwrap();
        assert_ne!(plan.lowered().identity(), other.lowered().identity());
        assert_eq!(scope, other.decision_reliability_scope().unwrap());
        let mut invalid_population = request.clone();
        invalid_population.target.target_population = "another-population".into();
        assert!(
            compile_cyclic_target_feedback_decision(invalid_population, contract.clone(), &ctx)
                .is_err()
        );
        let mut outside_support = request;
        outside_support.target.rows[0][0] = 2.0;
        assert!(compile_cyclic_target_feedback_decision(outside_support, contract, &ctx).is_err());
    }

    #[test]
    #[ignore = "source-only fitted predictive replay; no scientific record earned"]
    fn paired_target_worlds_refit_and_replay() {
        let ctx = ExecutionContext::for_tests(47);
        let (request, contract) = fixture();
        let plan = compile_cyclic_target_feedback_decision(request, contract, &ctx).unwrap();
        let result = plan.execute(&ctx).unwrap();
        assert_eq!(result.calibration, "not_measured");
        let columns = &result.decision.chain.record().sources;
        assert_eq!(columns.len(), 2);
        // Shared target baseline and disturbances cancel. The remaining
        // action effect depends only on each paired posterior feedback weight.
        for ((left, right), parameters) in
            columns[0].values.iter().zip(&columns[1].values).zip(&result.posterior.samples)
        {
            let effect = f64::from_bits(*right) - f64::from_bits(*left);
            let expected = 0.2 / (1.0 - 0.2 * parameters[0]);
            assert!((effect - expected).abs() < 1e-7);
        }
        let bytes = plan.export(&result, "target-feedback", &ctx).unwrap();
        let replay =
            consume_cyclic_target_feedback_decision(&bytes, plan.lowered().identity(), &ctx)
                .unwrap();
        assert_eq!(replay.posterior, result.posterior);
        assert_eq!(replay.decision.chain.identity(), result.decision.chain.identity());
        assert_eq!(replay.decision.decision, result.decision.decision);
    }
}
