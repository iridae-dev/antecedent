//! Experiment consumers preserving method-native structural, cohort and donor targets.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::super::designs::{NativeJointDesignResult, NativeJointIdentification};
use super::{
    Arc, DecisionProblem, DistributionArtifact, DistributionCalibration, DistributionTrust,
    EvidenceLedger, ExecutionContext, ExperimentRankingReceipt, JointLawExperimentRequest,
    JointPanelError, JointPanelResponseFunctional, PanelExperimentDecisionState,
    PanelExperimentEvidence, PanelExperimentMapping, PanelExperimentModularPolicy,
    PanelExperimentUpdate, PanelFutureSample, PanelFutureSampleValue, SemanticDigest, SourceId,
    admit, rank_law_next_samples, source, validate_mapping_moments, validate_model_dependency,
    validate_update,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_estimate::joint_design::compiled::{JointDesignLaw, JointNativeDesignLaw};
use antecedent_io::distribution_artifact::{
    DistributionIdentity, DistributionMetadata, DistributionProvenance, DrawAlignment,
};

/// Declared global support of a local-ratio mapping law used by a Gaussian likelihood.
/// These are support assumptions on the complete mapping prior, not bounds inferred from draws.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalRatioGaussianRegularization {
    /// Finite global bound on the absolute intercept.
    pub max_abs_intercept: f64,
    /// Strict positive lower bound on the absolute mapping slope.
    pub min_abs_slope: f64,
    /// Finite global upper bound on the absolute mapping slope.
    pub max_abs_slope: f64,
    /// Explicit substantive bounded-support assumption for the entire paired mapping law.
    pub assumption: String,
}

/// Explicit linear functional of already compiled native target coordinates.
/// Combining local/structural/cohort coordinates retains that target kind; it
/// cannot silently become a population intervention-world mean.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeExperimentBinding {
    /// One finite coefficient per retained native target, with at least one nonzero.
    pub coefficients: Vec<f64>,
    /// Explicit units of the original numeric response and this linear functional.
    pub outcome_units: String,
    /// Required for a local ratio whose unconditioned moments are not established.
    #[serde(default)]
    pub local_ratio_regularization: Option<LocalRatioGaussianRegularization>,
}
/// Decision result carrying the actual native scientific target and conditional update scope.
#[derive(Clone, Debug)]
pub struct NativeExperimentDecision {
    /// Existing terminal utility/constraint evaluation.
    pub result: antecedent_design::decision_eval::DecisionResult,
    /// Exact original target kind, keys, functional, population and assumptions.
    pub identification: NativeJointIdentification,
    /// Exact original source, mapping and experimental dependency.
    pub dependency: SemanticDigest,
    /// The combined likelihood/translation is explicitly model conditional.
    pub scope: antecedent_design::scientific_source::ScientificConsumptionScope,
}
/// Fitless conditional experiment update of the actual method-native posterior.
#[derive(Clone, Debug)]
pub struct NativeJointExperimentResult {
    original: NativeJointDesignResult,
    binding: NativeExperimentBinding,
    update: Arc<PanelExperimentUpdate>,
    law: DistributionArtifact,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
/// Replayed native experiment evidence; historical replay grants no current authority.
#[derive(Clone, Debug)]
pub struct HistoricalNativeJointExperiment {
    result: NativeJointExperimentResult,
}
impl HistoricalNativeJointExperiment {
    /// Recomputed original-data posterior and experimental likelihood component.
    #[must_use]
    pub fn numerical_law(&self) -> &DistributionArtifact {
        &self.result.law
    }
    /// Historical declarations never authorize caller-current scientific use.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
}
fn native_coordinates(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
) -> Result<Vec<ScientificQuantity>, JointPanelError> {
    let proof = original.prepared().identification();
    if binding.outcome_units.trim().is_empty()
        || binding.coefficients.len() != proof.targets.len()
        || binding.coefficients.iter().any(|v| !v.is_finite())
        || binding.coefficients.iter().all(|v| *v == 0.0)
    {
        return Err(JointPanelError::compile(
            "native experimental projection needs explicit response units and one finite coefficient per exact native target",
        ));
    }
    Ok(proof
        .targets
        .iter()
        .enumerate()
        .map(|(i, key)| ScientificQuantity {
            variable_id: format!("native:{}:{i}", original.prepared().dependency()),
            variable_name: key.clone(),
            role: QuantityRole::Outcome,
            units: binding.outcome_units.clone(),
            population_id: proof.population.clone(),
            regime_id: format!("native:{}:{key}", proof.target_kind),
            horizon: 0,
            functional_id: proof.target_kind.clone(),
            conditioning: vec![],
            transform_id: "identity".into(),
        })
        .collect())
}
fn native_prior(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
) -> Result<DistributionArtifact, JointPanelError> {
    let JointNativeDesignLaw::Moments(JointDesignLaw::Bayesian(posterior)) = original.law() else {
        return Err(JointPanelError::Unsupported("native sampling moments, conformal regions and donor point fits cannot become a Bayesian experimental prior".into()));
    };
    if posterior.estimate.worlds != original.prepared().identification().targets {
        return Err(JointPanelError::compile(
            "native posterior coordinates do not match the compiled target proof",
        ));
    }
    native_draw_artifact(original, binding, &posterior.draws)
}
fn native_draw_artifact(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
    draws: &[Vec<f64>],
) -> Result<DistributionArtifact, JointPanelError> {
    let coordinates = native_coordinates(original, binding)?;
    let dependency = original.dependency_identity();
    let identity = DistributionIdentity::new(
        antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior,
        &coordinates,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: dependency.to_hex(),
            provider_id: original.prepared().identification().method.clone(),
            rng_id: "retained_native_mechanism_posterior".into(),
            snapshot_id: dependency.to_hex(),
            causal_contract_id: dependency.to_hex(),
        },
    )
    .map_err(JointPanelError::compile)?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [draws.len(), coordinates.len()],
            weights: None,
            supported: Some(vec![true; coordinates.len()]),
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws.iter().flatten().copied().collect(),
    )
    .map_err(JointPanelError::compile)
}
impl NativeJointDesignResult {
    /// Native coordinate quantities preserve structural/cohort/donor target semantics.
    /// # Errors
    /// Invalid projection dimensions or missing original response-unit declaration.
    pub fn experiment_quantities(
        &self,
        binding: &NativeExperimentBinding,
    ) -> Result<Vec<ScientificQuantity>, JointPanelError> {
        native_coordinates(self, binding)
    }
    /// Exact scalar native target answered by the experimental translation.
    /// # Errors
    /// Invalid target projection or response units.
    pub fn experiment_quantity(
        &self,
        binding: &NativeExperimentBinding,
    ) -> Result<ScientificQuantity, JointPanelError> {
        let mut quantity = native_coordinates(self, binding)?.remove(0);
        let mut writer =
            antecedent_core::CanonicalWriter::new("antecedent.native_experiment_target.v1");
        writer.bytes(self.prepared().dependency().as_bytes()).text(&binding.outcome_units);
        for c in &binding.coefficients {
            writer.real(*c);
        }
        if let Some(bounds) = &binding.local_ratio_regularization {
            writer.text(
                &serde_json::to_string(bounds)
                    .map_err(|error| JointPanelError::compile(error.to_string()))?,
            );
        }
        let projection = SemanticDigest::from_bytes(writer.finish());
        quantity.regime_id = format!("native-projection:{projection}");
        quantity.functional_id =
            format!("{}:linear:{projection}", self.prepared().identification().target_kind);
        Ok(quantity)
    }
    /// Condition a genuine native posterior without changing its original scientific target kind.
    /// # Errors
    /// Unsupported native law, mismatched translation, invalid evidence/mapping or admission.
    #[expect(
        clippy::too_many_lines,
        reason = "Keep native target, proper conditional likelihood, evidence authority and admitted retention in one transaction."
    )]
    pub fn condition_native_experiment(
        &self,
        binding: NativeExperimentBinding,
        request: JointLawExperimentRequest,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<NativeJointExperimentResult, JointPanelError> {
        source::check_cancelled(ctx)?;
        if request.bridge.target.quantity != self.experiment_quantity(&binding)? {
            return Err(JointPanelError::compile(
                "experimental bridge must target the exact method-native functional, population and units",
            ));
        }
        let update = PanelExperimentUpdate {
            bridge: request.bridge,
            functional: JointPanelResponseFunctional::TerminalMean,
            observed_mean: request.observed_mean,
            observation_variance: request.observation_variance,
            sample_size: request.sample_size,
            modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
            mapping: request.mapping,
            evidence: request.evidence,
        };
        validate_update(&update, ledger)?;
        validate_model_dependency(self.dependency_identity(), &update, ledger)?;
        let draw_count = match self.law() {
            JointNativeDesignLaw::Moments(JointDesignLaw::Bayesian(posterior)) => {
                posterior.draws.len()
            }
            JointNativeDesignLaw::LocalComplier(local) => {
                local.posterior.as_ref().map_or(0, Vec::len)
            }
            _ => 0,
        };
        if draw_count == 0 {
            return Err(JointPanelError::Unsupported("native experiment conditioning requires an actual posterior likelihood/prior, never sampling or conformal uncertainty".into()));
        }
        let cells = draw_count
            .checked_mul(binding.coefficients.len() + 32)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| JointPanelError::compile("native experiment admission overflow"))?;
        let mut binding_size = super::law::WireCounter(0);
        serde_json::to_writer(&mut binding_size, &binding)
            .map_err(|error| JointPanelError::compile(error.to_string()))?;
        let cells = cells
            .checked_add(
                u64::try_from(binding_size.0)
                    .map_err(|_| JointPanelError::compile("native binding size overflow"))?,
            )
            .ok_or_else(|| JointPanelError::compile("native binding admission overflow"))?;
        let retention = Arc::new(admit(ctx, cells)?);
        let prior = match self.law() {
            JointNativeDesignLaw::LocalComplier(local) => {
                validate_local_regularization(&binding, &update)?;
                let rows = local
                    .posterior
                    .as_ref()
                    .ok_or_else(|| {
                        JointPanelError::Unsupported(
                            "local conditioning needs an actual retained posterior".into(),
                        )
                    })?
                    .iter()
                    .map(|row| vec![row[0]])
                    .collect::<Vec<_>>();
                native_draw_artifact(self, &binding, &rows)?
            }
            _ => native_prior(self, &binding)?,
        };
        let values = antecedent_prob::finite_belief::linear_functional_rows(
            prior.draws(),
            &binding.coefficients,
        )
        .map_err(JointPanelError::compile)?;
        validate_mapping_moments(&update, &values)?;
        let weights = antecedent_prob::finite_belief::AffineGaussianExperiment {
            observed_mean: update.observed_mean,
            observation_variance: update.observation_variance,
            sample_size: update.sample_size,
            target_values: &values,
            intercepts: &update.mapping.intercepts,
            slopes: &update.mapping.slopes,
        }
        .condition(&vec![1.0 / prior.n_draws() as f64; prior.n_draws()])
        .map_err(JointPanelError::compile)?;
        let mut origins = vec![
            update.evidence.model.to_hex(),
            update.evidence.experiment.to_hex(),
            update.evidence.mapping.to_hex(),
        ];
        origins.extend(update.evidence.prior_origins.iter().map(SourceId::to_hex));
        let dependency = SemanticDigest::from_bytes(
            antecedent_io::joint_panel_experiment_artifact::experiment_update_identity(
                &prior.metadata().identity.causal_contract_id,
                self.dependency_identity(),
                update.bridge.identity(),
                update.observed_mean,
                update.observation_variance,
                update.sample_size,
                &update.mapping.intercepts,
                &update.mapping.slopes,
                &weights,
                &origins,
            ),
        );
        let mut metadata = prior.metadata().clone();
        metadata.weights = Some(weights);
        metadata.identity.causal_contract_id = dependency.to_hex();
        metadata.identity.provider_id =
            "antecedent.native_joint_design.conditional_experiment".into();
        let law = DistributionArtifact::new(metadata, prior.draws().to_vec())
            .map_err(JointPanelError::compile)?;
        Ok(NativeJointExperimentResult {
            original: self.clone(),
            binding,
            update: Arc::new(update),
            law,
            dependency,
            retention,
        })
    }
}
impl NativeJointExperimentResult {
    /// Additional global mapping support premise establishing local-ratio posterior moments.
    #[must_use]
    pub fn local_ratio_regularization(&self) -> Option<&LocalRatioGaussianRegularization> {
        self.binding.local_ratio_regularization.as_ref()
    }
    fn validate(&self, ledger: &EvidenceLedger) -> Result<(), JointPanelError> {
        self.retention.observe_current();
        validate_update(&self.update, ledger)?;
        validate_model_dependency(self.original.dependency_identity(), &self.update, ledger)
    }
    /// Actual native target retained through the experimental update.
    #[must_use]
    pub fn identification(&self) -> &NativeJointIdentification {
        self.original.prepared().identification()
    }
    /// Aligned complete native law under caller-current evidence authority.
    /// # Errors
    /// Withdrawn, overlapping, stale or mismatched evidence refuses consumption.
    pub fn numerical_law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<&DistributionArtifact, JointPanelError> {
        self.validate(ledger)?;
        Ok(&self.law)
    }
    /// Terminal utility and operational constraints over the native target vector.
    /// # Errors
    /// Current evidence, quantities, utilities or constraints do not satisfy the existing owner.
    pub fn decide(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
        ledger: &EvidenceLedger,
    ) -> Result<NativeExperimentDecision, JointPanelError> {
        self.validate(ledger)?;
        let result = antecedent_design::decision_eval::evaluate_contract(contract, &self.law)
            .map_err(JointPanelError::compile)?;
        Ok(NativeExperimentDecision {
            result,
            identification: self.identification().clone(),
            dependency: self.dependency,
            scope:
                antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
        })
    }
    /// Additional independent samples of this exact mapped native functional.
    /// # Errors
    /// Current evidence, study costs, utility, integration resources or cancellation fails.
    pub fn rank_next_samples<A: serde::Serialize>(
        &self,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        candidates: &[PanelFutureSample],
        replicates: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
        self.validate(ledger)?;
        let encoded = antecedent_io::convert::to_cbor(self.identification())
            .map_err(JointPanelError::compile)?;
        let identification =
            SemanticDigest::from_bytes(antecedent_core::CanonicalWriter::digest_bytes(
                "antecedent.native_joint_identification.v1",
                &encoded,
            ));
        let quantity = self.original.experiment_quantity(&self.binding)?;
        let target = SemanticDigest::from_bytes(antecedent_core::CanonicalWriter::digest_bytes(
            "antecedent.native_experiment_question.v1",
            quantity.canonical_identity().as_bytes(),
        ));
        let receipt = ExperimentRankingReceipt {
            target_identity: target,
            identification_identity: identification,
            dependency_identity: self.dependency,
            question_identity: target,
            projection: None,
        };
        rank_law_next_samples(
            &receipt,
            &self.law,
            &self.update,
            &self.binding.coefficients,
            problem,
            candidates,
            replicates,
            ctx,
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeExperimentWire {
    version: u16,
    native: Vec<u8>,
    binding: NativeExperimentBinding,
    bridge: antecedent_io::bridge_artifact::FullBridgeContractWire,
    ledger: antecedent_io::evidence_ledger_artifact::EvidenceLedgerMeta,
    origins: antecedent_io::joint_panel_experiment_artifact::PanelExperimentOrigins,
    observed: u64,
    variance: u64,
    sample_size: u64,
    intercepts: Vec<u64>,
    slopes: Vec<u64>,
    dependency: [u8; 32],
}
impl NativeJointExperimentResult {
    /// Full original native data/proof/statistical recipe and paired experimental inputs.
    /// # Errors
    /// Current authority, serialization or bounded resource admission refuses export.
    pub fn export(
        &self,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        self.validate(ledger)?;
        source::check_cancelled(ctx)?;
        let mut source_quote = super::law::WireCounter(0);
        serde_json::to_writer(&mut source_quote, self.original.prepared().recipe())
            .map_err(JointPanelError::compile)?;
        serde_json::to_writer(&mut source_quote, self.original.law())
            .map_err(JointPanelError::compile)?;
        let bytes = self
            .retention
            .remainder()
            .bytes()
            .checked_mul(16)
            .and_then(|n| n.checked_add(u64::try_from(source_quote.0).ok()?.checked_mul(16)?))
            .ok_or_else(|| JointPanelError::compile("native experiment export quote overflow"))?;
        let _lease = admit(ctx, bytes)?;
        let (_, history) =
            antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::seal(ledger, &[])
                .map_err(JointPanelError::compile)?;
        let wire = NativeExperimentWire {
            version: 1,
            native: self.original.export_artifact()?,
            binding: self.binding.clone(),
            bridge: antecedent_io::bridge_artifact::FullBridgeContractWire::from_domain(
                &self.update.bridge,
            )
            .map_err(JointPanelError::compile)?,
            ledger: history,
            origins: antecedent_io::joint_panel_experiment_artifact::PanelExperimentOrigins {
                model: self.update.evidence.model.to_hex(),
                experiment: self.update.evidence.experiment.to_hex(),
                mapping: self.update.evidence.mapping.to_hex(),
                priors: self.update.evidence.prior_origins.iter().map(SourceId::to_hex).collect(),
            },
            observed: self.update.observed_mean.to_bits(),
            variance: self.update.observation_variance.to_bits(),
            sample_size: self.update.sample_size,
            intercepts: self.update.mapping.intercepts.iter().map(|v| v.to_bits()).collect(),
            slopes: self.update.mapping.slopes.iter().map(|v| v.to_bits()).collect(),
            dependency: *self.dependency.as_bytes(),
        };
        let bytes = antecedent_io::convert::to_cbor(&wire).map_err(JointPanelError::compile)?;
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(JointPanelError::compile(
                "native experiment artifact exceeds bounded size",
            ));
        }
        Ok(bytes)
    }
    /// Independently re-execute original native identification/statistics and the likelihood update.
    /// # Errors
    /// Unsupported version/fields, changed native evidence or invalid translation/current history.
    pub fn replay(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<HistoricalNativeJointExperiment, JointPanelError> {
        source::check_cancelled(ctx)?;
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(JointPanelError::compile(
                "native experiment artifact exceeds bounded size",
            ));
        }
        let _lease = admit(
            ctx,
            (bytes.len() as u64).checked_mul(128).ok_or_else(|| {
                JointPanelError::compile("native experiment replay quote overflow")
            })?,
        )?;
        let wire: NativeExperimentWire =
            antecedent_io::convert::from_cbor(bytes).map_err(JointPanelError::compile)?;
        if wire.version != 1 {
            return Err(JointPanelError::compile("unknown native experiment artifact version"));
        }
        let original =
            super::super::designs::replay_native_joint_design_execution(&wire.native, ctx)?;
        let history = antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::from_meta(
            &wire.ledger,
            None,
        )
        .map_err(JointPanelError::compile)?;
        let origin = |id: &str| {
            SourceId::from_hex(id).ok_or_else(|| {
                JointPanelError::compile("invalid native experimental origin identity")
            })
        };
        let request = JointLawExperimentRequest {
            bridge: wire.bridge.to_domain().map_err(JointPanelError::compile)?,
            observed_mean: f64::from_bits(wire.observed),
            observation_variance: f64::from_bits(wire.variance),
            sample_size: wire.sample_size,
            mapping: PanelExperimentMapping {
                dependency: original.dependency_identity(),
                intercepts: wire.intercepts.iter().map(|v| f64::from_bits(*v)).collect(),
                slopes: wire.slopes.iter().map(|v| f64::from_bits(*v)).collect(),
            },
            evidence: PanelExperimentEvidence {
                model: origin(&wire.origins.model)?,
                experiment: origin(&wire.origins.experiment)?,
                mapping: origin(&wire.origins.mapping)?,
                prior_origins: wire
                    .origins
                    .priors
                    .iter()
                    .map(|id| origin(id))
                    .collect::<Result<_, _>>()?,
            },
        };
        let result =
            original.condition_native_experiment(wire.binding, request, history.ledger(), ctx)?;
        if *result.dependency.as_bytes() != wire.dependency {
            return Err(JointPanelError::compile(
                "native experiment artifact does not reproduce its declared complete dependency",
            ));
        }
        Ok(HistoricalNativeJointExperiment { result })
    }
    /// Bind historical replay to this actual native target/source and caller-current evidence.
    /// # Errors
    /// Another native source, binding, experiment, mapping or withdrawn evidence refuses.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<(), JointPanelError> {
        self.validate(ledger)?;
        let replay = Self::replay(bytes, ctx)?;
        if replay.result.original.prepared().dependency() != self.original.prepared().dependency()
            || replay.result.binding != self.binding
            || replay.result.dependency != self.dependency
            || replay.result.law.metadata() != self.law.metadata()
            || replay.result.law.draws() != self.law.draws()
        {
            return Err(JointPanelError::compile(
                "native experimental replay differs from caller-current source/target/update",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

fn validate_local_regularization(
    binding: &NativeExperimentBinding,
    update: &PanelExperimentUpdate,
) -> Result<(), JointPanelError> {
    let bounds = binding.local_ratio_regularization.as_ref().ok_or_else(||JointPanelError::Unsupported("local ratio prior moments are not established; a Gaussian experiment needs explicit global bounded-mapping support to establish posterior moments".into()))?;
    if binding.coefficients.len() != 1
        || !bounds.max_abs_intercept.is_finite()
        || bounds.max_abs_intercept < 0.0
        || !bounds.min_abs_slope.is_finite()
        || bounds.min_abs_slope <= 0.0
        || !bounds.max_abs_slope.is_finite()
        || bounds.max_abs_slope < bounds.min_abs_slope
        || bounds.assumption.trim().is_empty()
        || update.mapping.intercepts.iter().any(|a| a.abs() > bounds.max_abs_intercept)
        || update
            .mapping
            .slopes
            .iter()
            .any(|b| b.abs() < bounds.min_abs_slope || b.abs() > bounds.max_abs_slope)
    {
        return Err(JointPanelError::compile(
            "local ratio Gaussian regularization requires explicit valid global mapping support and every paired realization inside it",
        ));
    }
    // For R, a nonzero c and globally |a|<=A, 0<b_min<=|b|<=B,
    // |(cR-a)/b-y| >= |cR|/B - A/b_min - |y|. Thus every
    // |R|^k times the positive known-variance Gaussian likelihood is bounded.
    // Its normalizer is strictly positive under any proper local-ratio prior.
    // Every polynomial posterior moment therefore exists, without asserting
    // any unconditioned ratio prior moment. The bounds remain explicit premises.
    Ok(())
}
