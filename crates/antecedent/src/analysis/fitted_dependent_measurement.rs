//! Conjugate uncertainty for a complete dependent-indicator error table.
//!
//! Validation counts are joint indicator cells labelled by the latent
//! construct. Each construct row has a symmetric Dirichlet prior and a
//! multinomial likelihood. Draws of the complete conditional table feed the
//! existing compiler-routed finite observation query. This is a posterior
//! conditional on the composed mechanism model and the labelled validation
//! design; it is not a calibrated sampling interval.

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, CausalRng, ExecutionContext, LikelihoodModel,
    OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator, Prior, SemanticDigest,
};
use antecedent_kernels::{quantile_type7_sorted, sample_gamma};
use antecedent_io::IoError;
use serde::{Deserialize, Serialize};

use super::observation_query::{
    DependentMeasuredObservationRequest, JointMeasuredObservationPlan, ObservationQueryRefusal,
};
use super::PreparedMechanismComposition;
use crate::planner::lowered::{
    ClaimDemand, Composition, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    LoweredProgram, ProblemTarget, ProgramCell, StandaloneCompositionLoweredParts,
    UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization,
};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "fitted_dependent_measurement_v1";
const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Joint validation counts in the exact table order of `observation.error_law`.
#[derive(Clone, Debug)]
pub struct FittedDependentMeasurementRequest {
    /// The outcome, evidence, interventions and scoped joint-error table shape.
    /// Supplied probabilities are replaced by the posterior plug-in table.
    pub observation: DependentMeasuredObservationRequest,
    /// Row-major `[construct level][joint indicator cell]` labelled counts.
    pub counts: Vec<u64>,
    /// Concentration of each joint indicator cell in each independent row prior.
    pub prior_alpha: f64,
    /// Number of aligned posterior table draws.
    pub draws: u32,
    /// Seed of the complete joint table draw schedule.
    pub seed: u64,
}

/// Outcome-level probabilities under the plug-in table and each posterior
/// table draw. Draw `i` contains every outcome level in `levels` order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FittedDependentMeasurementReport {
    /// Outcome levels, in the finite model's order.
    pub levels: Vec<String>,
    /// Conditional probabilities at the posterior mean error table; this is
    /// a plug-in value, not the mean of the posterior outcome draws.
    pub plug_in: Vec<f64>,
    /// Joint posterior outcome-probability rows.
    pub posterior_draws: Vec<Vec<f64>>,
    /// Identity of the common draw schedule.
    pub alignment_identity: String,
    /// Always `not_measured`; no repeated-sampling coverage is inherited.
    pub calibration: String,
}

/// Equal-tail posterior interval of one outcome-level probability, conditional
/// on the composed model and validation design. Coverage is not measured.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FittedDependentMeasurementInterval {
    /// Outcome level whose probability is bounded.
    pub outcome_level: String,
    /// Posterior mass inside the interval.
    pub nominal_level: f64,
    /// Lower equal-tail quantile.
    pub lower: f64,
    /// Upper equal-tail quantile.
    pub upper: f64,
    /// Always `not_measured` until a new whole-route record is earned.
    pub calibration: String,
}

impl FittedDependentMeasurementReport {
    /// Read the type-7 equal-tail interval of the retained posterior draws.
    /// It is a posterior credible interval, with unmeasured sampling coverage.
    ///
    /// # Errors
    /// An unknown outcome level, malformed draw table or invalid level.
    pub fn nominal_interval(
        &self, outcome_level: &str, level: f64,
    ) -> Result<FittedDependentMeasurementInterval, FittedDependentMeasurementError> {
        if !level.is_finite() || !(0.0..1.0).contains(&level) {
            return Err(FittedDependentMeasurementError::Invalid("nominal interval level is invalid"));
        }
        let position = self.levels.iter().position(|name| name == outcome_level)
            .ok_or(FittedDependentMeasurementError::Invalid("outcome level is unknown"))?;
        if self.posterior_draws.len() < 2
            || self.posterior_draws.iter().any(|row| row.len() != self.levels.len())
        {
            return Err(FittedDependentMeasurementError::Invalid("posterior draw table is malformed"));
        }
        let mut column: Vec<f64> = self.posterior_draws.iter().map(|row| row[position]).collect();
        column.sort_by(f64::total_cmp);
        let tail = (1.0 - level) / 2.0;
        let lower = quantile_type7_sorted(&column, tail);
        let upper = quantile_type7_sorted(&column, 1.0 - tail);
        if !lower.is_finite() || !upper.is_finite() {
            return Err(FittedDependentMeasurementError::Invalid("posterior quantile is nonfinite"));
        }
        Ok(FittedDependentMeasurementInterval {
            outcome_level: outcome_level.into(), nominal_level: level,
            lower, upper, calibration: "not_measured".into(),
        })
    }
}

/// Refusal of the fitted table declaration, observation compiler or artifact.
#[derive(Debug, thiserror::Error)]
pub enum FittedDependentMeasurementError {
    /// The validation counts, prior or draw schedule are invalid.
    #[error("fitted dependent measurement: {0}")]
    Invalid(&'static str),
    /// The finite observation query refused.
    #[error("fitted dependent measurement observation: {0}")]
    Observation(#[from] ObservationQueryRefusal),
    /// Artifact encoding or decoding refused.
    #[error("fitted dependent measurement artifact: {0}")]
    Io(#[from] IoError),
}

/// Retained six-layer program for validation-data fitting and observation.
#[derive(Clone, Debug)]
pub struct CompiledFittedDependentMeasurement {
    composition: PreparedMechanismComposition,
    request: FittedDependentMeasurementRequest,
    plug_in: JointMeasuredObservationPlan,
    lowered: Arc<LoweredProgram>,
    cells: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FittedDependentMeasurementWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    report: FittedDependentMeasurementReport,
}

impl PreparedMechanismComposition {
    /// Compile labelled joint-error fitting and finite-model conditioning as
    /// one posterior program. Counts and prior are bound before any draw.
    ///
    /// # Errors
    /// A malformed table, unsupported prior or observation-query refusal.
    pub fn compile_fitted_dependent_measurement(
        &self, request: FittedDependentMeasurementRequest,
    ) -> Result<CompiledFittedDependentMeasurement, FittedDependentMeasurementError> {
        let law = &request.observation.error_law;
        let cells = law.indicator_levels.iter().try_fold(1_usize, |product, levels| {
            product.checked_mul(levels.len())
        }).ok_or(FittedDependentMeasurementError::Invalid("indicator table size overflows"))?;
        let expected = cells.checked_mul(law.construct_levels.len())
            .ok_or(FittedDependentMeasurementError::Invalid("joint table size overflows"))?;
        if cells == 0 || law.construct_levels.is_empty() || expected == 0
            || expected > 4096 || request.counts.len() != expected
            || request.counts.iter().any(|count| *count > 1_000_000_000)
            || !request.prior_alpha.is_finite() || !(0.5..=1_000_000.0).contains(&request.prior_alpha)
            || !(2..=2048).contains(&request.draws)
        {
            return Err(FittedDependentMeasurementError::Invalid(
                "joint counts, positive Dirichlet prior or draw count are outside the checked bounds",
            ));
        }
        let mut plug_in_request = request.observation.clone();
        plug_in_request.error_law.probability_bits = request.counts.chunks_exact(cells)
            .flat_map(|row| {
                let denominator: f64 = row.iter()
                    .map(|count| *count as f64 + request.prior_alpha).sum();
                row.iter().map(move |count| ((*count as f64 + request.prior_alpha) / denominator).to_bits())
            }).collect();
        let plug_in = self.compile_dependent_measured_observation(&plug_in_request)?;
        let mut writer = CanonicalWriter::new("antecedent.fitted_dependent_measurement.question.v1");
        writer.bytes(plug_in.lowered().identity().as_bytes())
            .word(request.prior_alpha.to_bits()).word(request.draws as u64).word(request.seed)
            .word(request.counts.len() as u64);
        for count in &request.counts { writer.word(*count); }
        let digest = writer.finish();
        let mut scientific = plug_in.lowered().scientific().clone();
        scientific.target = ProblemTarget::QueryFunctional {
            kind: FunctionalKind::ConditionalObservation, outcome: None, query: digest,
        };
        scientific.plan_id = Arc::from("fitted_dependent_measurement.v1");
        scientific.source_digest = Some(digest);
        let mut statistical = plug_in.lowered().statistical().clone();
        let prior = Prior::new(format!("symmetric_dirichlet_joint_error:{:016x}", request.prior_alpha.to_bits()))
            .map_err(|_| FittedDependentMeasurementError::Invalid("invalid prior identity"))?;
        let likelihood = LikelihoodModel::new("labelled_joint_indicator_multinomial")
            .map_err(|_| FittedDependentMeasurementError::Invalid("invalid likelihood identity"))?;
        statistical.realization = Realization::Bayesian(
            OrdinaryBayesianLikelihood::new(likelihood, prior.clone()),
        );
        statistical.posterior = Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::Conjugate,
            draws: request.draws as usize,
            prior,
        });
        let mut models = statistical.models.to_vec();
        models.push(ModelSlot {
            role: ModelRole::Mechanism, family: Arc::from("dependent_joint_error_dirichlet"),
            inputs: Arc::from([]), output: None,
        });
        statistical.models = Arc::from(models);
        let mut evidence = plug_in.lowered().evidence().clone();
        let mut sources = evidence.sources.to_vec();
        sources.push(EvidenceSourceRef {
            role: EvidenceRole::Observations, digest: Some(digest),
        });
        evidence.sources = Arc::from(sources);
        evidence.composition = Composition::Chained { stages: 3 };
        let mut physical = plug_in.lowered().physical().clone();
        let mut operators = physical.operators.to_vec();
        operators.insert(1, PhysicalOperator::declare(PhysicalEffect::Fit)
            .estimates(ByteEstimates::absent().with_output(
                (expected as u64).saturating_mul(request.draws as u64).saturating_mul(8),
            )).build().map_err(|_| FittedDependentMeasurementError::Invalid(
                "fitted table physical operator is invalid",
            ))?);
        physical.operators = Arc::from(operators);
        let lowered = Arc::new(StandaloneCompositionLoweredParts {
            cell: ProgramCell::FittedDependentMeasuredObservation,
            scientific,
            identification: plug_in.lowered().identification_layer().clone(),
            statistical, evidence, physical,
            claim: ClaimDemand { uncertainty: UncertaintyReading::Posterior,
                kind: Some(ClaimKind::Point) },
        }.finish());
        Ok(CompiledFittedDependentMeasurement {
            composition: self.clone(), request, plug_in, lowered, cells,
        })
    }
}

impl CompiledFittedDependentMeasurement {
    /// Every layer of the fitted measurement and observation program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained program without drawing parameters.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Draw the joint error table row by row and answer the finite query in
    /// every draw under the retained compiler. No coverage claim is made.
    ///
    /// # Errors
    /// Cancellation, invalid numerical draw or observation-query refusal.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<FittedDependentMeasurementReport, FittedDependentMeasurementError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let point = self.plug_in.execute(&scoped)?;
        let levels = point.conditional_levels().ok_or(FittedDependentMeasurementError::Invalid(
            "fitted measurement query did not return conditional levels",
        ))?;
        let names: Vec<_> = levels.iter().map(|(level, _)| level.clone()).collect();
        let plug_in: Vec<_> = levels.into_iter().map(|(_, probability)| probability).collect();
        let mut rng = CausalRng::from_seed(self.request.seed);
        let mut posterior_draws = Vec::with_capacity(self.request.draws as usize);
        for _ in 0..self.request.draws {
            if scoped.cancellation.is_cancelled() {
                return Err(ObservationQueryRefusal::of_cancelled().into());
            }
            let mut sampled = self.request.observation.clone();
            sampled.error_law.probability_bits.clear();
            for row in self.request.counts.chunks_exact(self.cells) {
                let gamma: Vec<f64> = row.iter().map(|count| {
                    sample_gamma(*count as f64 + self.request.prior_alpha, 1.0, &mut rng)
                }).collect();
                let total: f64 = gamma.iter().sum();
                if !total.is_finite() || total <= 0.0 || gamma.iter().any(|value| !value.is_finite()) {
                    return Err(FittedDependentMeasurementError::Invalid("a posterior table draw is nonfinite"));
                }
                sampled.error_law.probability_bits.extend(
                    gamma.into_iter().map(|value| (value / total).to_bits()),
                );
            }
            let plan = self.composition.compile_dependent_measured_observation(&sampled)?;
            let outcome = plan.execute(&scoped)?.conditional_levels()
                .ok_or(FittedDependentMeasurementError::Invalid(
                    "posterior measurement query did not return conditional levels",
                ))?;
            if outcome.iter().map(|(level, _)| level).ne(names.iter()) {
                return Err(FittedDependentMeasurementError::Invalid(
                    "a posterior draw changed the outcome-level coordinate",
                ));
            }
            posterior_draws.push(outcome.into_iter().map(|(_, value)| value).collect());
        }
        Ok(FittedDependentMeasurementReport {
            levels: names, plug_in, posterior_draws,
            alignment_identity: self.lowered.identity().to_hex(),
            calibration: "not_measured".into(),
        })
    }

    /// Seal the complete posterior report after reexecuting its program.
    ///
    /// # Errors
    /// Changed report or artifact encoding refusal.
    pub fn export(
        &self, report: &FittedDependentMeasurementReport, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, FittedDependentMeasurementError> {
        if self.execute(ctx)? != *report {
            return Err(FittedDependentMeasurementError::Invalid("fitted measurement report differs"));
        }
        let bytes = antecedent_io::to_cbor(&FittedDependentMeasurementWire {
            version: 1, required_feature: FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(), report: report.clone(),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(FittedDependentMeasurementError::Invalid("fitted measurement artifact exceeds its bound"));
        }
        Ok(bytes)
    }

    /// Independently rerun the retained program and compare every posterior
    /// draw against the artifact under an externally retained program identity.
    ///
    /// # Errors
    /// Foreign feature, changed program or changed posterior result.
    pub fn consume(
        &self, bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
    ) -> Result<FittedDependentMeasurementReport, FittedDependentMeasurementError> {
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(FittedDependentMeasurementError::Invalid("fitted measurement artifact exceeds its bound"));
        }
        let wire: FittedDependentMeasurementWire = antecedent_io::from_cbor(bytes)?;
        if wire.version != 1 || wire.required_feature != FEATURE
            || wire.program_identity != *expected_program.as_bytes()
            || expected_program != self.lowered.identity()
        {
            return Err(FittedDependentMeasurementError::Invalid("fitted measurement program differs"));
        }
        let live = self.execute(ctx)?;
        if wire.report != live {
            return Err(FittedDependentMeasurementError::Invalid("fitted measurement posterior differs on replay"));
        }
        Ok(live)
    }
}
