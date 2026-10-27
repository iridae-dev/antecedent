//! Joint-world execution lowered from the common prepared response compiler.
//!
//! This is the Gaussian-linear realization of an already identified static response.
//! Preparation freezes one design and one inference procedure; every requested world is a
//! linear functional of that same fit. Scheduled and mediation targets keep their own
//! compiler contracts and are refused here, rather than treated as static hard sets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::execution::AdmissionLease;
use antecedent_core::physical_choice::{
    AvailableCapabilities, CostQuotes, DeclaredCost, PhysicalAlternative, PhysicalChoice,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus, CausalQuery,
    CausalSchema, ExecutionContext, GraphFamily, InferenceSelection, Intervention,
    InterventionWorld, ModelAssumptions, ObservationSpec, ParametricAssumption, ScientificTarget,
    ScientificWorldId, TargetForm, TargetPopulation, VariableId,
};
use antecedent_data::TableView;
use antecedent_estimate::response::{
    JointLinearError, JointLinearEvaluation, JointLinearInference, PreparedJointLinear,
};
use antecedent_estimate::{ContinuousResponseEstimator, PreparedBayesianProblem};
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, IdentificationResult, RequiredFactor, UnresolvedObligation,
};
use antecedent_stats::DesignColumnRole;

use super::PreparedStudy;
use crate::analysis::builder::DataInput;
use crate::{AcceptedGraph, CausalError, GraphClass, InferenceMode};

const LINEAR_ASSUMPTION: &str = "response.joint.gaussian_linear";

/// Retained joint-world program lowered from an existing prepared response.
///
/// All worlds share one complete-case design and one fitted coefficient law. Retargeting
/// changes only world projections; changing data, prior or kernel dependencies requires
/// compiling a new program from the corresponding prepared study.
#[derive(Clone, Debug)]
pub struct PreparedJointResponse {
    source: Arc<JointResponseSource>,
    target: Arc<ScientificTarget>,
    identification: Arc<CompiledIdentificationProduct>,
    world_ids: Arc<[ScientificWorldId]>,
    world_weights: Arc<[Vec<f64>]>,
    contrast: Option<Arc<[f64]>>,
    retention: Arc<AdmissionLease>,
}

#[derive(Debug)]
struct JointResponseSource {
    numerical: PreparedJointLinear,
    identification: IdentificationResult,
    query: antecedent_core::ResponseQuery,
    graph: AcceptedGraph,
    schema: CausalSchema,
    treatments: Arc<[VariableId]>,
    digest: [u8; 32],
    graph_family: GraphFamily,
    choice: PhysicalChoice,
    retention: Arc<AdmissionLease>,
}

/// One joint numerical result and the exact compiled scientific program that produced it.
#[derive(Clone, Debug)]
pub struct JointResponseResult {
    program: PreparedJointResponse,
    evaluation: JointLinearEvaluation,
    context: ExecutionContext,
}

impl PreparedStudy {
    /// Lower a scientific collection or contrast of worlds from this prepared response.
    ///
    /// Uses the existing Gaussian-linear response design and frequentist or Bayesian
    /// g-computation engine. This explicit realization retains the linear-Gaussian,
    /// fixed-empirical-covariate premise in the target. Compile performs no fitting or new
    /// identification. Unsupported joint support remains model-conditional; a numerical
    /// result is never a new calibration or causal license.
    ///
    /// # Errors
    /// Requires a completely observed, all-observed static hard-intervention response with
    /// an actual retained common-adjustment proof. Refuses changed scientific coordinates,
    /// non-Gaussian likelihoods, scheduled/mediation targets, resource failures or cancellation.
    pub fn compile_joint_response(
        &self,
        target: &ScientificTarget,
        ctx: &ExecutionContext,
    ) -> Result<PreparedJointResponse, CausalError> {
        cancelled(ctx)?;
        joint_target_admission(target)?;
        let CausalQuery::Response(query) = &self.analysis.query else {
            return Err(unsupported(
                "joint response compilation requires a prepared response query",
            ));
        };
        let antecedent_core::ResponseFunctional::InterventionResponse { outcome, interventions } =
            &query.functional
        else {
            return Err(unsupported(
                "joint response compilation requires intervention-response identification",
            ));
        };
        if query.temporal.is_some()
            || query.observation != ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || !query.outcome_functional.is_mean()
        {
            return Err(unsupported(
                "joint Gaussian response requires a complete static mean over the observed population",
            ));
        }
        let cache = self.analysis.static_cache().ok_or_else(|| {
            unsupported("joint response compilation requires retained point identification")
        })?;
        if cache.identification.query != self.analysis.query
            || cache.identification.status
                != antecedent_core::IdentificationStatus::NonparametricallyIdentified
            || !cache.estimand.is_adjustment_shaped()
        {
            return Err(unsupported(
                "joint response compilation requires the matching identified common-adjustment functional",
            ));
        }
        let treatments = source_treatments(interventions)?;
        let graph_family = match self.analysis.graph.class() {
            GraphClass::Dag => GraphFamily::Dag,
            GraphClass::Admg => GraphFamily::Admg,
            _ => {
                return Err(unsupported(
                    "joint Gaussian response requires a retained point-graph proof; incomplete graph mass cannot be dropped",
                ));
            }
        };
        let DataInput::Tabular(data) = &self.analysis.data else {
            return Err(unsupported("static joint response requires tabular source data"));
        };
        let p = treatments
            .len()
            .checked_add(cache.estimand.adjustment_set.len())
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| resource("joint design width overflow"))?;
        let source_bytes = data
            .row_count()
            .checked_mul(p.checked_add(2).ok_or_else(|| resource("joint design width overflow"))?)
            .and_then(|n| n.checked_mul(24))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| resource("joint source admission quote overflow"))?;
        let reservation = Arc::new(admit(ctx, source_bytes)?);
        let fitter = ContinuousResponseEstimator::new(Arc::clone(&cache.estimand.adjustment_set));
        let problem = antecedent_core::numerical_admission::with_context(ctx, || {
            antecedent_core::numerical_admission::with_reservation(Arc::clone(&reservation), || {
                fitter.prepare_linear_response_problem(data, *outcome, &treatments)
            })
        })?;
        let inference = inference(&self.analysis.inference, ctx)?;
        let numerical = PreparedJointLinear::new(problem, inference, ctx.kernel_policy)
            .map_err(numerical_error)?;
        let resident = source_resident_bytes(numerical.problem())?;
        let retention = Arc::new(
            reservation.acquire_child(0, resident).map_err(|error| resource(&error.to_string()))?,
        );
        let source = Arc::new(JointResponseSource {
            numerical,
            identification: cache.identification.clone(),
            query: query.clone(),
            graph: self.analysis.graph.as_ref().clone(),
            schema: data.schema().clone(),
            treatments: treatments.into(),
            digest: data.storage().content_digest(),
            graph_family,
            choice: materialized_choice(source_bytes)?,
            retention,
        });
        PreparedJointResponse::lower(source, target, ctx)
    }
}

impl PreparedJointResponse {
    fn lower(
        source: Arc<JointResponseSource>,
        target: &ScientificTarget,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        cancelled(ctx)?;
        let worlds = target_worlds(target)?;
        // The world count is bounded by the memory admission below, not by a literal.
        if worlds.is_empty() {
            return Err(unsupported("joint response needs at least one world"));
        }
        let bytes = worlds
            .len()
            .checked_mul(
                source
                    .numerical
                    .problem()
                    .design
                    .ncols
                    .checked_add(16)
                    .ok_or_else(|| resource("joint world width overflow"))?,
            )
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| resource("joint world quote overflow"))?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let mut world_ids = Vec::with_capacity(worlds.len());
        let mut weights = Vec::with_capacity(worlds.len());
        let mut obligations = Vec::with_capacity(worlds.len());
        for world in worlds {
            let (weight, established) = compile_world(&source, world)?;
            world_ids.push(world.id());
            weights.push(weight);
            obligations.push(IdentificationObligation::for_world(world.id(),
                if established { IdentificationOutcome::Identified } else { IdentificationOutcome::Unidentified },
                if established { IdentificationClaimClass::IdentifiedClaim } else { IdentificationClaimClass::ModelConditional },
                if established { vec![] } else { vec![UnresolvedObligation::new("response.joint_support_unverified",
                    "marginal treatment ranges do not establish conditional joint-world support; the Gaussian response is model-conditional")] }, None));
        }
        let established = obligations.iter().all(IdentificationObligation::is_identified_claim);
        let contrast = contrast_coefficients(target, &world_ids)?;
        if let Some(coefficients) = &contrast {
            require_estimable_under_additivity(coefficients, &weights)?;
        }
        let target = retained_target(&source, target)?;
        let mut parts = CompiledProductParts::new(
            IdentificationOutcome::Identified,
            if established {
                IdentificationClaimClass::IdentifiedClaim
            } else {
                IdentificationClaimClass::ModelConditional
            },
        );
        parts.world_obligations = obligations;
        if contrast.is_some() {
            parts.assembled = Some(IdentificationObligation::assembled(
                if established {
                    IdentificationOutcome::Identified
                } else {
                    IdentificationOutcome::Unidentified
                },
                if established {
                    IdentificationClaimClass::IdentifiedClaim
                } else {
                    IdentificationClaimClass::ModelConditional
                },
                vec![],
                None,
            ));
        }
        let mut variables = source.treatments.to_vec();
        variables.extend(source.numerical.problem().adjustment_set.iter().copied());
        variables.extend(source.query.functional.outcome_ids());
        parts
            .required_factors
            .push(RequiredFactor::new("shared_linear_response_mechanism", variables));
        let identification = CompiledIdentificationProduct::retain_for_target(
            &source.identification,
            parts,
            &target,
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        Ok(Self {
            source,
            target: Arc::new(target),
            identification: Arc::new(identification),
            world_ids: world_ids.into(),
            world_weights: weights.into(),
            contrast: contrast.map(Arc::from),
            retention,
        })
    }

    /// The exact scientific target, including Gaussian-linear realization premises.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Retained source proof and world/assembled support obligations.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        &self.identification
    }
    /// Source table content identity; a changed source cannot reuse this program as a new fit.
    #[must_use]
    pub fn source_digest(&self) -> [u8; 32] {
        self.source.digest
    }
    /// Original point identification, without reconstructed method names or builders.
    #[must_use]
    pub fn source_identification(&self) -> &IdentificationResult {
        &self.source.identification
    }
    /// Original query whose common-adjustment functional covers the intervention coordinates.
    #[must_use]
    pub fn source_query(&self) -> &antecedent_core::ResponseQuery {
        &self.source.query
    }
    /// Original accepted point graph.
    #[must_use]
    pub fn source_graph(&self) -> &AcceptedGraph {
        &self.source.graph
    }
    /// Original schema whose coordinate and unit identities are retained.
    #[must_use]
    pub fn source_schema(&self) -> &CausalSchema {
        &self.source.schema
    }
    /// Shared materialized design, inference configuration and synchronized fit owner.
    #[must_use]
    pub fn numerical_program(&self) -> &PreparedJointLinear {
        &self.source.numerical
    }
    /// Ordered world identities of numerical output columns.
    #[must_use]
    pub fn world_ids(&self) -> &[ScientificWorldId] {
        &self.world_ids
    }
    /// Mean design rows after each intervention overlay, in world output order.
    #[must_use]
    pub fn world_weights(&self) -> &[Vec<f64>] {
        &self.world_weights
    }
    /// Coefficients of the one assembled functional, when a contrast was requested.
    #[must_use]
    pub fn contrast_coefficients(&self) -> Option<&[f64]> {
        self.contrast.as_deref()
    }
    /// Actual executable materialization choice, including unavailable alternatives.
    #[must_use]
    pub fn physical_choice(&self) -> &PhysicalChoice {
        &self.source.choice
    }

    /// Validate a historical artifact against this caller-supplied current compiled authority.
    ///
    /// Independent numerical replay alone grants no current scientific permission. This
    /// compares the complete target, source, proof, prior, numerical policy and projection
    /// dependencies before returning a result carrying this live program's support.
    ///
    /// # Errors
    /// Replay failure, cancellation, resource refusal or any current-dependency mismatch.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<
        JointResponseResult,
        antecedent_io::joint_response_artifact::JointResponseArtifactError,
    > {
        use antecedent_io::joint_response_artifact::{
            JointResponseArtifactError, consume_joint_response, joint_response_identity,
        };
        self.identification
            .validate_target(&self.target)
            .map_err(|error| JointResponseArtifactError::Invalid(error.to_string()))?;
        let historical = consume_joint_response(bytes, ctx)?;
        if historical.identity() != joint_response_identity(self.artifact_source())? {
            return Err(JointResponseArtifactError::Mismatch);
        }
        Ok(JointResponseResult {
            program: self.clone(),
            evaluation: historical.into_evaluation(),
            context: ctx.clone(),
        })
    }

    fn artifact_source(
        &self,
    ) -> antecedent_io::joint_response_artifact::JointResponseArtifactSource<'_> {
        antecedent_io::joint_response_artifact::JointResponseArtifactSource {
            target: &self.target,
            numerical: &self.source.numerical,
            world_weights: &self.world_weights,
            contrast: self.contrast.as_deref(),
            source_digest: self.source.digest,
            source_identification: &self.source.identification,
            compiled_identification: &self.identification,
        }
    }

    /// Recompile world projections while retaining the exact same source, prior and fit.
    ///
    /// # Errors
    /// Refuses changed source coordinates/population, unsupported scheduling, resource
    /// failures or cancellation. A new numerical dependency requires a new study compilation.
    pub fn retarget(
        &self,
        target: &ScientificTarget,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::lower(Arc::clone(&self.source), target, ctx)
    }

    /// Fit once if necessary, then jointly evaluate all worlds through the retained engine.
    ///
    /// # Errors
    /// Numerical fit failure, unsupported kernel-policy change, admission refusal or
    /// cancellation. Failed posterior slots remain aligned across world/contrast outputs.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<JointResponseResult, CausalError> {
        self.identification
            .validate_target(&self.target)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        self.source.retention.observe_current();
        self.retention.observe_current();
        let evaluation = self
            .source
            .numerical
            .execute(&self.world_weights, self.contrast.as_deref(), ctx)
            .map_err(numerical_error)?;
        Ok(JointResponseResult { program: self.clone(), evaluation, context: ctx.clone() })
    }
}

impl JointResponseResult {
    /// Retained program, usable after the original builder and prepared study are discarded.
    #[must_use]
    pub fn program(&self) -> &PreparedJointResponse {
        &self.program
    }
    /// Scientific target of these exact output coordinates.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        self.program.target()
    }
    /// Actual identification and per-world support; inspect claim class before causal use.
    pub fn identification(&self) -> &CompiledIdentificationProduct {
        self.program.identification()
    }
    /// Source data identity.
    #[must_use]
    pub fn source_digest(&self) -> [u8; 32] {
        self.program.source_digest()
    }
    /// World identities in mean/covariance/draw column order.
    #[must_use]
    pub fn world_ids(&self) -> &[ScientificWorldId] {
        self.program.world_ids()
    }
    /// Numerical joint law; covariance is never assembled from marginal intervals.
    #[must_use]
    pub fn evaluation(&self) -> &JointLinearEvaluation {
        &self.evaluation
    }
    /// Export exact compiled inputs and the original joint output for independent replay.
    ///
    /// # Errors
    /// Unsupported future semantics, encoding bounds, cancellation or resource refusal.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, antecedent_io::joint_response_artifact::JointResponseArtifactError> {
        antecedent_io::joint_response_artifact::encode_joint_response(
            self.program.artifact_source(),
            &self.evaluation,
            ctx,
        )
    }

    /// Export licensed mean-functional coordinates for the existing affine decision engine.
    ///
    /// # Errors
    /// Unverified world support or malformed source coordinates. No outcome law is implied.
    pub fn numerical_mean_source(
        &self,
    ) -> Result<antecedent_design::decision_eval::MeanSource, CausalError> {
        if !self.identification().is_identified_claim() {
            return Err(unsupported(
                "current causal mean consumption requires established joint-world support",
            ));
        }
        Ok(antecedent_design::decision_eval::MeanSource {
            coordinates: self.quantities()?,
            means: self.evaluation.means.to_vec(),
            estimate: match self.program.numerical_program().inference() {
                JointLinearInference::Bayesian(_) => {
                    antecedent_design::decision_eval::MeanEstimate::Posterior
                }
                JointLinearInference::Frequentist => {
                    antecedent_design::decision_eval::MeanEstimate::Frequentist
                }
            },
            provider_id: "native:joint_gaussian_response".into(),
            snapshot_id: antecedent_core::to_hex(&self.source_digest()),
            causal_contract_id: self.target().identity().to_hex(),
            rng_id: "none:retained_mean_functionals".into(),
        })
    }

    /// Supply the existing joint distribution consumer with aligned posterior mean functionals.
    ///
    /// # Errors
    /// Missing posterior, failed slots, unverified support, cancellation, resource limits or
    /// invalid artifact coordinates. Failed slots are never dropped or renormalized.
    pub fn numerical_joint_law(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::distribution_artifact::DistributionArtifact, CausalError> {
        use antecedent_io::distribution_artifact::{
            DistributionArtifact, DistributionCalibration, DistributionIdentity,
            DistributionMetadata, DistributionProvenance, DistributionTrust, DrawAlignment,
        };
        cancelled(ctx)?;
        let source = self.numerical_mean_source()?;
        let draws = self
            .evaluation
            .posterior_draws
            .as_ref()
            .ok_or_else(|| unsupported("a sampling covariance is not a posterior joint law"))?;
        if self.evaluation.failed_draws.iter().any(|failed| *failed)
            || draws.iter().any(|value| !value.is_finite())
        {
            return Err(unsupported(
                "current posterior law requires every aligned draw slot; failed mass cannot be discarded",
            ));
        }
        let bytes = draws
            .len()
            .checked_mul(16)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| resource("joint evidence quote overflow"))?;
        let _scratch = admit(ctx, bytes)?;
        let mut rows = Vec::with_capacity(draws.len());
        for draw in 0..self.evaluation.draw_count {
            for column in 0..self.world_ids().len() {
                rows.push(draws[column * self.evaluation.draw_count + draw]);
            }
        }
        let inference = self.program.numerical_program().inference();
        let rng_id = match inference {
            JointLinearInference::Bayesian(options) => {
                format!("shared_coefficient_draws:{}:{}", options.seed, options.n_draws)
            }
            JointLinearInference::Frequentist => return Err(unsupported("no posterior joint law")),
        };
        let identity = DistributionIdentity::new(
            antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior,
            &source.coordinates,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: source.snapshot_id.clone(),
                provider_id: source.provider_id,
                rng_id,
                snapshot_id: source.snapshot_id,
                causal_contract_id: source.causal_contract_id,
            },
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [self.evaluation.draw_count, self.world_ids().len()],
                weights: None,
                supported: Some(vec![true; self.world_ids().len()]),
                calibration: DistributionCalibration::Unmeasured,
                trust: DistributionTrust::NativeLicensed,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            rows,
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })
    }

    /// A mean-functional consumer retaining the complete compiled target and proof.
    ///
    /// # Errors
    /// Invalid current support, quantity binding or compiled dependency encoding.
    pub fn mean_source(
        &self,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, CausalError> {
        self.scientific_source(&self.context, false)
    }
    /// An aligned posterior consumer retaining scientific standing and exact dependencies.
    ///
    /// # Errors
    /// Missing/failed posterior slots, current support, resource or dependency failures.
    pub fn joint_law(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, CausalError> {
        self.scientific_source(ctx, true)
    }
    fn scientific_source(
        &self,
        ctx: &ExecutionContext,
        posterior: bool,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, CausalError> {
        use antecedent_design::scientific_source::{
            ScientificDecisionSource, ScientificSourcePayload,
        };
        cancelled(ctx)?;
        let bytes = self.scientific_payload_bytes(posterior)?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let payload = if posterior {
            ScientificSourcePayload::Law(Box::new(self.numerical_joint_law(ctx)?))
        } else {
            ScientificSourcePayload::Means(self.numerical_mean_source()?)
        };
        // Identity encoding uses the existing wire owner and a transient copy of
        // its numeric program. Bound that scratch separately from returned payload.
        let identity_bytes = self
            .program
            .source
            .numerical
            .problem()
            .design
            .matrix
            .len()
            .checked_add(self.program.source.numerical.problem().design.outcome.len())
            .and_then(|n| {
                self.program.world_weights.iter().try_fold(n, |n, row| n.checked_add(row.len()))
            })
            .and_then(|n| n.checked_mul(96))
            .and_then(|n| n.checked_add(1_000_000))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| resource("joint scientific identity quote overflow"))?;
        let _identity_scratch = admit(ctx, identity_bytes)?;
        let identity = antecedent_io::joint_response_artifact::joint_response_identity(
            self.program.artifact_source(),
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        ScientificDecisionSource::new_shared(
            Arc::clone(&self.program.target),
            Arc::clone(&self.program.identification),
            antecedent_core::SemanticDigest::from_bytes(identity),
            payload,
        )
        .map(|source| {
            source
                .retain_admission(Arc::clone(&self.program.retention))
                .retain_admission(Arc::clone(&self.program.source.retention))
                .retain_admission(retention)
        })
        .map_err(|error| CausalError::Compile { message: error.to_string() })
    }
    fn scientific_payload_bytes(&self, posterior: bool) -> Result<u64, CausalError> {
        // Static coordinates have no conditioning allocation. Include all dynamically
        // sized schema labels/units and bounded source/target identity strings, with
        // simultaneous native-to-wire coordinate conversion buffers.
        let mut bytes = 2048_usize;
        for world in target_worlds(self.target())? {
            let variable = self.program.source.schema.get(world.binding().outcome().outcome())?;
            let coordinate = std::mem::size_of::<antecedent_contract::ScientificQuantity>()
                .checked_add(512)
                .and_then(|n| n.checked_add(variable.name.len()))
                .and_then(|n| n.checked_add(world.binding().outcome().units().name().len()))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| resource("joint scientific coordinate quote overflow"))?;
            bytes = bytes
                .checked_add(coordinate)
                .ok_or_else(|| resource("joint scientific payload quote overflow"))?;
        }
        let values = if posterior {
            self.evaluation.posterior_draws.as_ref().map_or(0, |draws| draws.len())
        } else {
            self.evaluation.means.len()
        };
        bytes
            .checked_add(
                values
                    .checked_mul(8)
                    .ok_or_else(|| resource("joint scientific draw quote overflow"))?,
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| resource("joint scientific payload quote overflow"))
    }

    fn quantities(&self) -> Result<Vec<antecedent_contract::ScientificQuantity>, CausalError> {
        let worlds = target_worlds(self.target())?;
        let target_id = self.target().identity().to_hex();
        worlds
            .iter()
            .map(|world| {
                let outcome = world.binding().outcome();
                let variable = self.program.source.schema.get(outcome.outcome())?;
                Ok(antecedent_contract::ScientificQuantity {
                    variable_id: format!(
                        "source:{}:variable:{}",
                        antecedent_core::to_hex(&self.source_digest()),
                        variable.id.raw()
                    ),
                    variable_name: variable.name.to_string(),
                    role: antecedent_contract::QuantityRole::Outcome,
                    units: outcome.units().name().into(),
                    population_id: "all_observed".into(),
                    regime_id: format!("target:{target_id}:world:{}", world.id().raw()),
                    horizon: 0,
                    functional_id: "mean".into(),
                    conditioning: vec![],
                    transform_id: "identity".into(),
                })
            })
            .collect()
    }

    /// Scientific calibration remains unmeasured for this newly assembled procedure.
    #[must_use]
    pub const fn calibration(&self) -> &'static str {
        "unmeasured"
    }
    /// Legal native evidence representations, without converting sampling into posterior risk.
    #[must_use]
    pub fn evidence_ports(&self) -> &'static [antecedent_contract::EvidencePort] {
        use antecedent_contract::EvidencePort;
        if self.evaluation.posterior_draws.is_some() {
            &[EvidencePort::Mean, EvidencePort::ParameterPosterior]
        } else {
            &[EvidencePort::Mean, EvidencePort::CovarianceInfluence]
        }
    }
}

fn source_treatments(interventions: &[Intervention]) -> Result<Vec<VariableId>, CausalError> {
    let mut variables = Vec::with_capacity(interventions.len());
    for intervention in interventions {
        let Intervention::Set { variable, value } = intervention else {
            return Err(unsupported(
                "joint Gaussian response source must identify hard intervention coordinates",
            ));
        };
        if value.as_f64().is_none_or(|value| !value.is_finite()) || variables.contains(variable) {
            return Err(unsupported(
                "joint Gaussian response requires unique finite numeric intervention coordinates",
            ));
        }
        variables.push(*variable);
    }
    if variables.is_empty() {
        return Err(unsupported("joint Gaussian response has no intervention coordinates"));
    }
    Ok(variables)
}

/// Shared actual admission boundary of the production static joint Gaussian realization.
/// A temporal caller cannot obtain static-model permission by attaching an L2 target.
pub(crate) fn joint_target_admission(target: &ScientificTarget) -> Result<(), CausalError> {
    let worlds = target_worlds(target)?;
    if worlds.is_empty()
        || worlds.iter().any(|world| {
            !world.time().is_static() || world.action().at_decision_interventions().is_none()
        })
    {
        return Err(unsupported(
            "static joint Gaussian execution cannot discharge temporal schedules or histories",
        ));
    }
    Ok(())
}

fn target_worlds(target: &ScientificTarget) -> Result<&[InterventionWorld], CausalError> {
    match target.form() {
        TargetForm::World(world) => Ok(std::slice::from_ref(world)),
        TargetForm::Worlds(worlds) => Ok(worlds.worlds()),
        TargetForm::Contrast(contrast) => Ok(contrast.worlds().worlds()),
        TargetForm::DoseTime(_) => Err(unsupported(
            "scheduled joint dose-by-time execution needs a shared temporal response law; static Gaussian projection cannot substitute",
        )),
        TargetForm::Mediation(_) => Err(unsupported(
            "mediation requires its theorem-scoped compiled execution, not a generic world subtraction",
        )),
        _ => Err(unsupported("joint response target form requires a specialized compiler")),
    }
}

fn compile_world(
    source: &JointResponseSource,
    world: &InterventionWorld,
) -> Result<(Vec<f64>, bool), CausalError> {
    let problem = source.numerical.problem();
    let binding = world.binding();
    let outcome = source.query.functional.outcome_ids();
    if !world.time().is_static()
        || binding.population() != &source.query.target_population
        || binding.unit().name() != "row"
        || !binding.conditioning().variables().is_empty()
        || outcome.as_slice() != [binding.outcome().outcome()]
        || !binding.outcome().functional().is_mean()
    {
        return Err(unsupported(
            "joint world population, unit, outcome, conditioning or time differs from its source proof",
        ));
    }
    let source_unit =
        source.schema.get(binding.outcome().outcome())?.unit.as_deref().unwrap_or("unspecified");
    if binding.outcome().units().name() != source_unit {
        return Err(unsupported(
            "joint world outcome units differ from source schema; absent metadata requires unspecified units",
        ));
    }
    let interventions = world
        .action()
        .at_decision_interventions()
        .ok_or_else(|| unsupported("static joint response cannot execute a scheduled program"))?;
    let variables = source_treatments(interventions)?;
    if variables.len() != source.treatments.len()
        || variables.iter().any(|variable| !source.treatments.contains(variable))
    {
        return Err(unsupported(
            "every joint world must assign exactly the identified intervention coordinates",
        ));
    }
    let n = problem.design.nrows;
    let mut weight =
        problem.design.matrix.chunks(n).map(antecedent_kernels::mean).collect::<Vec<_>>();
    let mut columns = Vec::with_capacity(interventions.len());
    for intervention in interventions {
        let Intervention::Set { variable, value } = intervention else {
            return Err(unsupported("joint world requires finite hard levels"));
        };
        let column = treatment_column(problem, &source.treatments, *variable)?;
        let level =
            value.as_f64().ok_or_else(|| unsupported("joint world requires numeric levels"))?;
        let observed = &problem.design.matrix[column * n..(column + 1) * n];
        let minimum = observed.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if level < minimum || level > maximum {
            return Err(unsupported(
                "joint world is outside observed marginal support; extrapolation is not authorized",
            ));
        }
        weight[column] = level;
        columns.push((column, level));
    }
    // Marginal ranges and repeated cells do not establish positivity conditional on
    // confounders. The static prepared source does not retain a joint conditional-support
    // certificate, so its point-graph identification cannot authorize these new worlds.
    let established = source.numerical.unadjusted_world_support(&columns);
    Ok((weight, established))
}

fn treatment_column(
    problem: &PreparedBayesianProblem,
    treatments: &[VariableId],
    variable: VariableId,
) -> Result<usize, CausalError> {
    problem
        .design
        .columns
        .iter()
        .enumerate()
        .find_map(|(index, column)| match column.role {
            DesignColumnRole::Treatment if treatments.first() == Some(&variable) => Some(index),
            DesignColumnRole::Covariate(id) if id == variable => Some(index),
            _ => None,
        })
        .ok_or_else(|| CausalError::Compile {
            message: "joint world variable has no retained design column".into(),
        })
}

fn retained_target(
    source: &JointResponseSource,
    target: &ScientificTarget,
) -> Result<ScientificTarget, CausalError> {
    if target.assumptions().graph() != GraphFamily::Unspecified
        && target.assumptions().graph() != source.graph_family
    {
        return Err(unsupported("joint target graph family differs from its identification proof"));
    }
    if target
        .assumptions()
        .records()
        .entries
        .iter()
        .any(|record| record.status == AssumptionStatus::Contradicted)
    {
        return Err(unsupported("joint target declares a contradicted scientific assumption"));
    }
    let mut assumptions = source.identification.required_assumptions.clone();
    assumptions.entries.extend(target.assumptions().records().entries.iter().cloned());
    assumptions.entries.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption { id: Arc::from(LINEAR_ASSUMPTION),
            description: Arc::from("additive Gaussian-linear conditional mean; iid Gaussian innovations; joint uncertainty conditions on the fixed empirical covariate distribution; no interaction term is fitted") }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("response.joint.linear") },
        scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
    });
    let inferred = match source.numerical.inference() {
        JointLinearInference::Frequentist => InferenceSelection::Frequentist,
        JointLinearInference::Bayesian(_) => InferenceSelection::Bayesian,
    };
    if target.inference() != InferenceSelection::Unspecified && target.inference() != inferred {
        return Err(unsupported(
            "joint target inference label differs from the retained numerical realization",
        ));
    }
    Ok(ScientificTarget::new(
        target.form().clone(),
        ModelAssumptions::new(source.graph_family, assumptions),
        inferred,
    ))
}

/// Refuse a contrast the additive Gaussian mean cannot move.
///
/// The fitted mean is additive in the treatments, so a contrast whose world weights cancel
/// (the product interaction `Y(a,b) − Y(a,0) − Y(0,b) + Y(0,0)`) would report exactly zero
/// with zero standard error: an artifact of the model, not an estimate. That functional needs
/// a route that fits the cell means.
fn require_estimable_under_additivity(
    coefficients: &[f64],
    weights: &[Vec<f64>],
) -> Result<(), CausalError> {
    let width = weights.first().map_or(0, Vec::len);
    let scale = weights.iter().flatten().fold(1.0_f64, |m, w| m.max(w.abs()));
    let cancels = (0..width).all(|column| {
        let combined: f64 = coefficients.iter().zip(weights).map(|(c, row)| c * row[column]).sum();
        combined.abs() <= 8.0 * f64::EPSILON * scale
    });
    if cancels {
        return Err(unsupported(
            "the additive Gaussian response gives this contrast zero by construction; an interaction needs the cell-mean family route",
        ));
    }
    Ok(())
}

fn contrast_coefficients(
    target: &ScientificTarget,
    world_ids: &[ScientificWorldId],
) -> Result<Option<Vec<f64>>, CausalError> {
    let TargetForm::Contrast(contrast) = target.form() else { return Ok(None) };
    let mut coefficients = vec![0.0; world_ids.len()];
    for term in contrast.terms() {
        let index = world_ids.iter().position(|world| *world == term.world).ok_or_else(|| {
            CausalError::Compile { message: "joint contrast term names a missing world".into() }
        })?;
        coefficients[index] = term.coefficient;
    }
    Ok(Some(coefficients))
}

fn inference(
    mode: &InferenceMode,
    ctx: &ExecutionContext,
) -> Result<JointLinearInference, CausalError> {
    match mode {
        InferenceMode::Frequentist => Ok(JointLinearInference::Frequentist),
        InferenceMode::Bayesian(config) => {
            if config.prior_artifact.is_some()
                || config.external_compose.is_some()
                || config.design_model.is_some()
            {
                return Err(unsupported(
                    "joint response requires a resolved coefficient prior; deferred source/design hydration must be completed before compilation",
                ));
            }
            Ok(JointLinearInference::Bayesian(antecedent_estimate::BayesianGComputationAte {
                backend: config.backend,
                likelihood: config.likelihood,
                n_draws: config.n_draws,
                seed: ctx.rng.master_seed(),
                overlap: antecedent_estimate::OverlapPolicy::ExplicitOverride,
                prior_scale: config.prior_scale,
                prior: config.prior.clone(),
            }))
        }
    }
}

fn materialized_choice(bytes: u64) -> Result<PhysicalChoice, CausalError> {
    let alternative = PhysicalAlternative::MaterializedDesign;
    let available = AvailableCapabilities::empty()
        .with_operator(alternative.required_operator())
        .with_resource(alternative.required_resource());
    let cost = DeclaredCost::from_raw(bytes);
    PhysicalChoice::choose(CostQuotes::new(cost, cost, cost, cost), available)
        .map_err(|error| resource(&error.to_string()))
}
fn source_resident_bytes(problem: &PreparedBayesianProblem) -> Result<u64, CausalError> {
    problem
        .design
        .matrix
        .len()
        .checked_add(problem.design.outcome.len())
        .and_then(|n| n.checked_add(problem.design.row_selection.len()))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| resource("retained joint design quote overflow"))
}
fn admit(ctx: &ExecutionContext, bytes: u64) -> Result<AdmissionLease, CausalError> {
    let limit = ctx.memory_limit_bytes();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(resource("joint response exceeds configured memory budget"));
    }
    ctx.acquire_admission(0, bytes).map_err(|error| resource(&error.to_string()))
}
fn numerical_error(error: JointLinearError) -> CausalError {
    match error {
        JointLinearError::Estimation(error) => CausalError::from(error),
        JointLinearError::Invalid(message) => CausalError::Compile { message },
        JointLinearError::Resource(message) => CausalError::Resource { message },
        JointLinearError::Cancelled => CausalError::Cancelled { stage: "joint_response" },
    }
}
fn unsupported(message: &'static str) -> CausalError {
    CausalError::Unsupported { message }
}
fn resource(message: &str) -> CausalError {
    CausalError::Resource { message: message.into() }
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        Err(CausalError::Cancelled { stage: "joint_response" })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BayesianConfig, RefuteSuite, Study};
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    use antecedent_core::{
        AssumptionSet, Conditioning, InterventionContrast, ObservationUnit, OutcomeQuantity,
        QuantityUnits, TimeFrame, Value, WorldAction, WorldBinding, WorldCollection,
    };
    use antecedent_data::TabularData;
    use antecedent_graph::Dag;

    fn data(shift: f64) -> TabularData {
        let mut columns = [vec![], vec![], vec![], vec![]];
        for index in 0_u32..360 {
            let a = f64::from(index % 3) - 1.0;
            let b = f64::from((index / 3) % 3) - 1.0;
            let z = (f64::from(index) * 0.23).sin();
            let y =
                shift + 3.0 + 2.0 * a - 1.5 * b + 0.5 * z + 0.1 * (f64::from(index) * 0.31).cos();
            for (column, value) in columns.iter_mut().zip([a, b, y, z]) {
                column.push(value);
            }
        }
        TabularData::from_f64_columns(
            ["a", "b", "y", "z"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap()
    }
    fn prepared(mode: InferenceMode, data: TabularData, ctx: &ExecutionContext) -> PreparedStudy {
        let graph =
            Dag::from_named_edges(data.schema(), &[("a", "y"), ("b", "y"), ("z", "y")]).unwrap();
        let query = antecedent_core::ResponseQuery::new(
            antecedent_core::ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([
                    Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                    Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
                ]),
            },
        );
        Study::tabular(data)
            .graph(graph)
            .query(query)
            .inference(mode)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(ctx)
            .unwrap()
    }
    fn target(interaction: bool) -> ScientificTarget {
        let binding = WorldBinding::new(
            TargetPopulation::AllObserved,
            ObservationUnit::new("row").unwrap(),
            OutcomeQuantity::mean(
                VariableId::from_raw(2),
                QuantityUnits::new("unspecified").unwrap(),
            )
            .unwrap(),
            Conditioning::unconditional(),
        )
        .unwrap();
        let cells = if interaction {
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        } else {
            vec![(0.0, 0.0), (1.0, 1.0)]
        };
        let worlds = WorldCollection::new(cells.into_iter().enumerate().map(|(index, (a, b))| {
            InterventionWorld::new(
                ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                WorldAction::at_decision([
                    Intervention::set(VariableId::from_raw(0), Value::f64(a)),
                    Intervention::set(VariableId::from_raw(1), Value::f64(b)),
                ])
                .unwrap(),
                index == 0,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        }))
        .unwrap();
        let id = ScientificWorldId::from_raw;
        let contrast = if interaction {
            InterventionContrast::interaction(worlds, id(3), id(1), id(2), id(0)).unwrap()
        } else {
            InterventionContrast::difference(worlds, id(1), id(0), id(0)).unwrap()
        };
        ScientificTarget::new(
            TargetForm::Contrast(contrast),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Unspecified,
        )
    }

    /// The four interaction cells under a linear contrast the additive response can move.
    fn four_cell_response_target() -> ScientificTarget {
        let cell_means = target(true);
        let TargetForm::Contrast(interaction) = cell_means.form() else { unreachable!() };
        let id = ScientificWorldId::from_raw;
        ScientificTarget::new(
            TargetForm::Contrast(
                InterventionContrast::linear(
                    interaction.worlds().clone(),
                    [(id(0), -3.0), (id(1), 1.0), (id(2), 1.0), (id(3), 1.0)],
                    id(0),
                )
                .unwrap(),
            ),
            cell_means.assumptions().clone(),
            cell_means.inference(),
        )
    }

    #[test]
    fn joint_gaussian_compiles_without_fit_executes_shared_fit_and_retargets_after_builder_disposal()
     {
        let ctx = ExecutionContext::for_tests(19);
        let program = {
            let prepared = prepared(InferenceMode::Frequentist, data(0.0), &ctx);
            let attempt = observe_with_context(&ctx, || {
                prepared.compile_joint_response(&target(false), &ctx)
            });
            assert_eq!(attempt.report.counts(Operation::Identification).attempted, 0);
            assert_eq!(attempt.report.counts(Operation::LeastSquaresSolve).attempted, 0);
            attempt.result.unwrap()
        };
        assert!(!program.numerical_program().has_fit());
        assert!(program.identification().is_identified_claim());
        let first = observe_with_context(&ctx, || program.execute(&ctx));
        assert_eq!(first.report.counts(Operation::AdjustedFit).completed, 1);
        let first = first.result.unwrap();
        let estimate = first.evaluation().contrast.unwrap();
        assert!((estimate.value - 0.5).abs() < 0.03);
        let covariance = &first.evaluation().covariance;
        assert!(
            (estimate.se.powi(2)
                - (covariance.get(0, 0) + covariance.get(1, 1) - 2.0 * covariance.get(0, 1)))
            .abs()
                < 1e-12
        );
        assert!(
            program.retarget(&target(true), &ctx).is_err(),
            "an additive response must refuse an interaction contrast rather than report zero"
        );
        let cells = four_cell_response_target();
        let interaction = program.retarget(&cells, &ctx).unwrap();
        let second = observe_with_context(&ctx, || interaction.execute(&ctx));
        assert_eq!(second.report.counts(Operation::AdjustedFit).attempted, 0);
        assert_eq!(second.report.counts(Operation::Identification).attempted, 0);
        let second = second.result.unwrap();
        // The additive response cannot express an interaction, but a coordinated
        // intervention must also produce a useful nonzero, jointly licensed response contrast.
        let baseline = target(false);
        let TargetForm::Contrast(baseline_contrast) = baseline.form() else { unreachable!() };
        let linear_target = ScientificTarget::new(
            TargetForm::Contrast(
                InterventionContrast::linear(
                    baseline_contrast.worlds().clone(),
                    [(ScientificWorldId::from_raw(0), -1.0), (ScientificWorldId::from_raw(1), 1.0)],
                    ScientificWorldId::from_raw(0),
                )
                .unwrap(),
            ),
            baseline.assumptions().clone(),
            baseline.inference(),
        );
        let coordinated = observe_with_context(&ctx, || -> Result<_, CausalError> {
            let retained = interaction.retarget(&linear_target, &ctx)?;
            assert!(retained.identification().is_identified_claim());
            retained.execute(&ctx)
        });
        assert_eq!(coordinated.report.counts(Operation::Identification).attempted, 0);
        assert_eq!(coordinated.report.counts(Operation::AdjustedFit).attempted, 0);
        assert_eq!(coordinated.report.counts(Operation::LeastSquaresSolve).attempted, 0);
        let coordinated = coordinated.result.unwrap();
        assert!(coordinated.identification().is_identified_claim());
        let joint = coordinated.evaluation().contrast.unwrap();
        let fitted_a = second.evaluation().means[1] - second.evaluation().means[0];
        let fitted_b = second.evaluation().means[2] - second.evaluation().means[0];
        assert!((joint.value - (fitted_a + fitted_b)).abs() < 1e-12);
        assert!((joint.value - 0.5).abs() < 0.03);
        assert!(joint.se > 0.0);
        let joint_covariance = &coordinated.evaluation().covariance;
        assert!(
            (joint.se.powi(2)
                - (joint_covariance.get(0, 0) + joint_covariance.get(1, 1)
                    - 2.0 * joint_covariance.get(0, 1)))
            .abs()
                < 1e-12
        );
        assert_eq!(second.calibration(), "unmeasured");
        assert!(second.target().assumptions().records().entries.iter().any(|record| record.status == AssumptionStatus::Declared && matches!(&record.assumption,
            Assumption::ParametricRestriction(premise) if premise.id.as_ref() == LINEAR_ASSUMPTION)));
        drop(program);
        assert_eq!(first.world_ids().len(), 2);
        assert!(first.program().execute(&ctx).is_ok());
    }

    fn assert_gaussian_source_retention(
        source: &antecedent_design::scientific_source::ScientificDecisionSource,
        minimum_bytes: u64,
        ctx: &ExecutionContext,
    ) {
        let attempt = observe_with_context(ctx, || {
            source.authorize(
                antecedent_design::scientific_source::ScientificConsumptionScope::IdentifiedCausal,
            )
        });
        attempt.result.unwrap();
        assert!(
            attempt.report.peak_admitted_bytes().is_some_and(|bytes| bytes >= minimum_bytes),
            "standalone scientific source must retain and observe its payload/metadata admissions"
        );
    }

    fn emit_gaussian_interchange(result: &JointResponseResult, ctx: &ExecutionContext) {
        let Some(directory) = std::env::var_os("ANTECEDENT_GAUSSIAN_INTERCHANGE_DIR") else {
            return;
        };
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        let law = result.numerical_joint_law(ctx).unwrap();
        let columns = law.shape()[1];
        let expected = serde_json::json!({
            "shape": law.shape(),
            "draw_bits": law.draws().iter().map(|value|value.to_bits()).collect::<Vec<_>>(),
            "mean_bits": (0..columns).map(|column|law.mean(column).unwrap().to_bits()).collect::<Vec<_>>(),
            "covariance_bits": (0..columns).map(|left|(0..columns).map(|right|law.covariance(left,right).unwrap().to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        });
        std::fs::write(
            directory.join("law.bin"),
            law.to_bytes("joint_gaussian_interchange").unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.join("identity.json"),
            serde_json::to_vec(&law.metadata().identity).unwrap(),
        )
        .unwrap();
        std::fs::write(directory.join("expected.json"), serde_json::to_vec(&expected).unwrap())
            .unwrap();
    }

    #[test]
    fn joint_gaussian_posterior_keeps_whole_contrast_draws_aligned_and_changed_data_refits() {
        let ctx = ExecutionContext::for_tests(23);
        let mut source = prepared(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256)),
            data(0.0),
            &ctx,
        );
        let program = source.compile_joint_response(&target(false), &ctx).unwrap();
        let first = program.execute(&ctx).unwrap();
        assert_eq!(
            first.numerical_mean_source().unwrap().estimate,
            antecedent_design::decision_eval::MeanEstimate::Posterior,
        );
        emit_gaussian_interchange(&first, &ctx);
        let retained_mean = first.mean_source().unwrap();
        let retained_law = first.joint_law(&ctx).unwrap();
        let minimum_law_bytes =
            u64::try_from(first.evaluation().posterior_draws.as_ref().unwrap().len() * 8).unwrap();
        let mut tiny = ctx.clone();
        tiny.set_memory_limit(Some(1));
        assert!(first.joint_law(&tiny).is_err());
        let evaluation = first.evaluation();
        let worlds = evaluation.posterior_draws.as_ref().unwrap();
        let contrasts = evaluation.contrast_draws.as_ref().unwrap();
        for draw in 0..evaluation.draw_count {
            assert_eq!(
                contrasts[draw].to_bits(),
                (worlds[evaluation.draw_count + draw] - worlds[draw]).to_bits()
            );
        }
        let retargeted = program.retarget(&four_cell_response_target(), &ctx).unwrap();
        let attempt = observe_with_context(&ctx, || retargeted.execute(&ctx));
        assert_eq!(attempt.report.counts(Operation::AdjustedFit).attempted, 0);
        let retargeted_result = attempt.result.unwrap();
        let cell = retargeted_result.evaluation();
        let cell_worlds = cell.posterior_draws.as_ref().unwrap();
        let cell_contrasts = cell.contrast_draws.as_ref().unwrap();
        let draws = cell.draw_count;
        let mut moved = false;
        for draw in 0..draws {
            let combined = -3.0 * cell_worlds[draw]
                + cell_worlds[draws + draw]
                + cell_worlds[2 * draws + draw]
                + cell_worlds[3 * draws + draw];
            assert!((cell_contrasts[draw] - combined).abs() < 1e-9);
            moved |= cell_contrasts[draw].abs() > 1e-9;
        }
        assert!(moved, "the four-cell linear contrast must be a real draw, not an additive zero");
        crate::analysis::prepared::rebind_estimate(&mut source, data(1.0), &ctx).unwrap();
        let updated = source.compile_joint_response(&target(false), &ctx).unwrap();
        assert_ne!(program.source_digest(), updated.source_digest());
        let attempt = observe_with_context(&ctx, || updated.execute(&ctx));
        assert_eq!(attempt.report.counts(Operation::AdjustedFit).completed, 1);
        assert!(attempt.result.unwrap().evaluation().means[0] > evaluation.means[0] + 0.9);
        drop(updated);
        drop(retargeted);
        drop(first);
        drop(program);
        drop(source);
        assert_gaussian_source_retention(&retained_mean, 16, &ctx);
        assert_gaussian_source_retention(&retained_law, minimum_law_bytes, &ctx);
    }

    fn verify_gaussian_decision_ports(
        result: &JointResponseResult,
        ctx: &ExecutionContext,
    ) -> antecedent_design::decision_contract::DecisionContract {
        use antecedent_design::decision_contract::{
            ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
            UtilityExpr,
        };
        let mean_source = result.numerical_mean_source().unwrap();
        assert_eq!(mean_source.means.as_slice(), result.evaluation().means.as_ref());
        let law = result.numerical_joint_law(ctx).unwrap();
        assert_eq!(
            law.metadata().calibration,
            antecedent_io::distribution_artifact::DistributionCalibration::Unmeasured
        );
        assert_eq!(law.metadata().identity.quantities.len(), 2);
        let law_bytes = law.to_bytes("joint_response_posterior").unwrap();
        let independent_law =
            antecedent_io::distribution_artifact::DistributionArtifact::from_bytes(
                &law_bytes,
                &law.metadata().identity,
            )
            .unwrap();
        assert_eq!(independent_law.metadata().identity, law.metadata().identity);
        let contract = DecisionContract {
            actions: mean_source
                .coordinates
                .iter()
                .enumerate()
                .map(|(index, coordinate)| DecisionAction {
                    id: index.to_string(),
                    kind: ActionKind::Intervention,
                    inputs: vec![coordinate.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "unspecified".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "all_observed".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        assert!(
            antecedent_design::decision_eval::evaluate_contract_on_means(&contract, &mean_source)
                .is_ok()
        );
        contract
    }

    #[test]
    fn joint_gaussian_artifact_replays_independently_and_requires_matching_current_authority() {
        use antecedent_io::joint_response_artifact::{
            JointResponseArtifactError, consume_joint_response,
        };
        let ctx = ExecutionContext::for_tests(47);
        let program = prepared(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)),
            data(0.0),
            &ctx,
        )
        .compile_joint_response(&target(false), &ctx)
        .unwrap();
        let result = program.execute(&ctx).unwrap();
        let bytes = result.export(&ctx).unwrap();
        let contract = verify_gaussian_decision_ports(&result, &ctx);

        let path = std::env::temp_dir()
            .join(format!("antecedent-joint-response-{}.artifact", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","analysis::prepared::joint_response::tests::joint_gaussian_artifact_fresh_process_consumer","--nocapture"])
            .env("ANTECEDENT_JOINT_RESPONSE_REPLAY",&path).output().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(
            child.status.success(),
            "fresh replay failed: {}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert!(String::from_utf8_lossy(&child.stdout).contains("joint artifact replay verified"));
        let scoped = result
            .joint_law(&ctx)
            .unwrap()
            .evaluate(
                &contract,
                antecedent_design::scientific_source::ScientificConsumptionScope::IdentifiedCausal,
            )
            .unwrap();
        assert_eq!(scoped.target_identity, result.target().identity());
        assert_eq!(scoped.identification_identity, result.identification().identity());
        let replay = consume_joint_response(&bytes, &ctx).unwrap();
        assert!(!replay.grants_current_authority());
        assert_eq!(replay.target().identity(), result.target().identity());
        assert_eq!(replay.evaluation().means, result.evaluation().means);
        assert_eq!(replay.evaluation().posterior_draws, result.evaluation().posterior_draws);
        let current = program.consume_current(&bytes, &ctx).unwrap();
        assert_eq!(current.identification().identity(), result.identification().identity());
        let changed_target = program.retarget(&four_cell_response_target(), &ctx).unwrap();
        assert!(matches!(
            changed_target.consume_current(&bytes, &ctx),
            Err(JointResponseArtifactError::Mismatch)
        ));
        let changed_data = prepared(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)),
            data(1.0),
            &ctx,
        )
        .compile_joint_response(&target(false), &ctx)
        .unwrap();
        assert!(matches!(
            changed_data.consume_current(&bytes, &ctx),
            Err(JointResponseArtifactError::Mismatch)
        ));
        let changed_prior = prepared(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(3.0)),
            data(0.0),
            &ctx,
        )
        .compile_joint_response(&target(false), &ctx)
        .unwrap();
        assert!(matches!(
            changed_prior.consume_current(&bytes, &ctx),
            Err(JointResponseArtifactError::Mismatch)
        ));
        let mut tiny = ctx.clone();
        tiny.set_memory_limit(Some(1));
        assert!(matches!(
            consume_joint_response(&bytes, &tiny),
            Err(JointResponseArtifactError::Resource(_))
        ));
        drop(result);
        drop(program);
        assert!(replay.evaluation().contrast.is_some());
    }

    #[test]
    fn joint_gaussian_artifact_refuses_unknown_version_and_changed_result_or_request() {
        use antecedent_io::joint_response_artifact::{
            MAX_JOINT_RESPONSE_ARTIFACT_BYTES, consume_joint_response,
        };
        let ctx = ExecutionContext::for_tests(59);
        let program = prepared(InferenceMode::Frequentist, data(0.0), &ctx)
            .compile_joint_response(&target(false), &ctx)
            .unwrap();
        let bytes = program.execute(&ctx).unwrap().export(&ctx).unwrap();
        let sections = antecedent_io::decode_container(
            "joint_response",
            MAX_JOINT_RESPONSE_ARTIFACT_BYTES,
            &["joint_response_v1"],
            &bytes,
        )
        .unwrap();
        let original: serde_json::Value = serde_json::from_slice(&sections[0]).unwrap();
        for field in ["version", "result", "request"] {
            let mut record = original.clone();
            match field {
                "version" => record["version"] = serde_json::json!(99),
                "result" => record["result"]["means"][0] = serde_json::json!(999),
                _ => record["request"]["weights"][0][0] = serde_json::json!(999),
            }
            let payload = serde_json::to_vec(&record).unwrap();
            let tampered = antecedent_io::encode_container(
                &antecedent_io::ArtifactContainer {
                    kind: "joint_response",
                    note: "tamper_test",
                    max_bytes: MAX_JOINT_RESPONSE_ARTIFACT_BYTES,
                    artifact_id: "joint_response",
                    library_version: env!("CARGO_PKG_VERSION"),
                },
                &[antecedent_io::OutSection {
                    id: "joint_response_v1",
                    content_type: antecedent_io::CONTENT_BYTES,
                    bytes: &payload,
                }],
            )
            .unwrap();
            assert!(consume_joint_response(&tampered, &ctx).is_err(), "accepted changed {field}");
        }
    }

    #[test]
    fn joint_gaussian_artifact_fresh_process_consumer() {
        let Some(path) = std::env::var_os("ANTECEDENT_JOINT_RESPONSE_REPLAY") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let replay = antecedent_io::joint_response_artifact::consume_joint_response(
            &bytes,
            &ExecutionContext::for_tests(99),
        )
        .unwrap();
        assert!(!replay.grants_current_authority());
        assert_eq!(replay.evaluation().means.len(), 2);
        assert_eq!(replay.evaluation().draw_count, 128);
        assert!(replay.evaluation().contrast.is_some());
        println!("joint artifact replay verified");
    }

    #[test]
    fn joint_gaussian_conditional_assignment_is_not_licensed_by_repeated_marginal_cells() {
        let ctx = ExecutionContext::for_tests(53);
        let mut columns = [vec![], vec![], vec![], vec![]];
        for index in 0_u32..240 {
            let cell = index % 4;
            let a = f64::from(cell % 2);
            let b = f64::from(cell / 2);
            let z = f64::from(cell);
            let y = 1.0 + a - b + 0.3 * z + (f64::from(index) * 0.1).sin();
            for (column, value) in columns.iter_mut().zip([a, b, y, z]) {
                column.push(value);
            }
        }
        let data = TabularData::from_f64_columns(
            ["a", "b", "y", "z"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap();
        let graph = Dag::from_named_edges(
            data.schema(),
            &[("a", "y"), ("b", "y"), ("z", "a"), ("z", "b"), ("z", "y")],
        )
        .unwrap();
        let query = antecedent_core::ResponseQuery::new(
            antecedent_core::ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([
                    Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                    Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
                ]),
            },
        );
        let source = Study::tabular(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let program = source.compile_joint_response(&target(false), &ctx).unwrap();
        assert!(!program.numerical_program().problem().adjustment_set.is_empty());
        assert!(!program.identification().is_identified_claim());
        assert!(
            program.execute(&ctx).as_ref().map_or(true, |result| result.mean_source().is_err())
        );
        assert!(
            program
                .identification()
                .world_obligations()
                .iter()
                .all(|world| world.claim_class() == IdentificationClaimClass::ModelConditional)
        );
        // Every joint cell has sixty observations, yet the confounder fixes assignment.
        // A rank-deficient conditional outcome law is refused rather than rescuing permission.
        assert!(program.execute(&ctx).is_err());
    }

    #[test]
    fn joint_gaussian_keeps_unverified_joint_support_model_conditional_and_refuses_policy_changes()
    {
        let ctx = ExecutionContext::for_tests(31);
        let source = prepared(InferenceMode::Frequentist, data(0.0), &ctx);
        let mut conditional = target(false);
        let TargetForm::Contrast(contrast) = conditional.form() else { panic!("contrast") };
        let binding = contrast.worlds().worlds()[0].binding().clone();
        let world = InterventionWorld::new(
            ScientificWorldId::from_raw(8),
            WorldAction::at_decision([
                Intervention::set(VariableId::from_raw(0), Value::f64(0.123)),
                Intervention::set(VariableId::from_raw(1), Value::f64(0.345)),
            ])
            .unwrap(),
            false,
            binding,
            TimeFrame::static_decision(),
        )
        .unwrap();
        conditional = ScientificTarget::new(
            TargetForm::World(world),
            conditional.assumptions().clone(),
            InferenceSelection::Unspecified,
        );
        let program = source.compile_joint_response(&conditional, &ctx).unwrap();
        assert!(!program.identification().is_identified_claim());
        assert_eq!(
            program.identification().world_obligations()[0].claim_class(),
            IdentificationClaimClass::ModelConditional
        );
        assert!(program.execute(&ctx).is_ok());
        let mut changed = ctx.clone();
        changed.kernel_policy.force_scalar = !ctx.kernel_policy.force_scalar;
        assert!(matches!(program.execute(&changed), Err(CausalError::Compile { .. })));
        let mut tiny = ctx.clone();
        tiny.set_memory_limit(Some(1));
        assert!(matches!(program.execute(&tiny), Err(CausalError::Resource { .. })));
        let cancelled = ctx.clone();
        cancelled.cancellation.cancel();
        assert!(matches!(program.execute(&cancelled), Err(CausalError::Cancelled { .. })));
    }
}
