//! Fitless experiment conditioning of an authenticated joint-design posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    Arc, BridgeContract, DecisionProblem, DistributionArtifact, DistributionCalibration,
    DistributionTrust, EvidenceLedger, ExecutionContext, ExperimentRankingReceipt, JointPanelError,
    JointPanelResponseFunctional, PanelExperimentDecisionSource, PanelExperimentDecisionState,
    PanelExperimentEvidence, PanelExperimentMapping, PanelExperimentModularPolicy,
    PanelExperimentUpdate, PanelFutureSample, PanelFutureSampleValue, SemanticDigest, SourceId,
    admit, rank_law_next_samples, source, validate_mapping_moments, validate_model_dependency,
    validate_update,
};
use antecedent_design::scientific_source::{ScientificDecisionSource, ScientificSourcePayload};

/// Declared experiment over an authenticated world-mean posterior.
#[derive(Clone, Debug)]
pub struct JointLawExperimentRequest {
    /// Licensed translation to the exact assembled contrast returned by `quantity`.
    pub bridge: BridgeContract,
    /// Independent experimental sample mean in the source units.
    pub observed_mean: f64,
    /// Known variance of one observation.
    pub observation_variance: f64,
    /// Number of independent observations.
    pub sample_size: u64,
    /// Original, aligned source/mapping slots, preserving dependent uncertainty.
    pub mapping: PanelExperimentMapping,
    /// Caller-current origins and overlap declarations.
    pub evidence: PanelExperimentEvidence,
}

/// Conditional update of a native design posterior, without a response-model refit.
#[derive(Clone, Debug)]
pub struct JointLawExperimentResult {
    original: ScientificDecisionSource,
    update: Arc<PanelExperimentUpdate>,
    law: DistributionArtifact,
    coefficients: Vec<f64>,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    native_design: Option<Arc<NativeDesignReplay>>,
}

impl JointLawExperimentResult {
    /// Derive the experimental contrast quantity from the original target and basis.
    /// # Errors
    /// Sources without one authenticated world-mean output per target world refuse.
    pub fn quantity(
        source: &ScientificDecisionSource,
    ) -> Result<antecedent_contract::ScientificQuantity, JointPanelError> {
        let (mut quantity, _) = contrast_binding(source)?;
        quantity.regime_id = format!("contrast:{}", source.target().identity());
        quantity.functional_id =
            format!("contrast:{}:{}", source.target().identity(), quantity.functional_id);
        Ok(quantity)
    }

    /// Execute the declared frozen finite-law likelihood update over native design draws.
    /// # Errors
    /// Nonposterior laws, unbound targets, unsupported bridges, changed evidence,
    /// invalid mappings and failed admission remain explicit refusals.
    #[expect(
        clippy::too_many_lines,
        reason = "One authenticated likelihood update retains source, evidence and admission checks together."
    )]
    pub fn condition(
        original: ScientificDecisionSource,
        request: JointLawExperimentRequest,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        original
            .authorize(
                antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
            )
            .map_err(super::super::search::map_scientific_error)?;
        let (_, coefficients) = contrast_binding(&original)?;
        let ScientificSourcePayload::Law(prior) = original.as_numerical_source() else {
            return Err(JointPanelError::Unsupported(
                "experimental conditioning needs an actual native posterior law".into(),
            ));
        };
        if !matches!(
            prior.semantic(),
            antecedent_io::quantity_wire::DistributionMeaningWire::ParameterPosterior
                | antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
        ) {
            return Err(JointPanelError::Unsupported(
                "design sampling draws cannot become a Bayesian experimental prior".into(),
            ));
        }
        if request.bridge.target.quantity != Self::quantity(&original)? {
            return Err(JointPanelError::compile(
                "experimental translation must target this exact native-design contrast",
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
        validate_model_dependency(original.dependency_identity(), &update, ledger)?;
        source::check_cancelled(ctx)?;
        let bytes = prior
            .draws()
            .len()
            .checked_mul(16)
            .and_then(|n| n.checked_add(prior.n_draws().checked_mul(128)?))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| JointPanelError::compile("native experiment quote overflow"))?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let values =
            antecedent_prob::finite_belief::linear_functional_rows(prior.draws(), &coefficients)
                .map_err(JointPanelError::compile)?;
        validate_mapping_moments(&update, &values)?;
        let masses = prior
            .metadata()
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0 / prior.n_draws() as f64; prior.n_draws()]);
        let weights = antecedent_prob::finite_belief::AffineGaussianExperiment {
            observed_mean: update.observed_mean,
            observation_variance: update.observation_variance,
            sample_size: update.sample_size,
            target_values: &values,
            intercepts: &update.mapping.intercepts,
            slopes: &update.mapping.slopes,
        }
        .condition(&masses)
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
                original.dependency_identity(),
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
        metadata.identity.provider_id = "antecedent.joint_design.conditional_experiment".into();
        metadata.calibration = DistributionCalibration::Unmeasured;
        metadata.trust = DistributionTrust::Unverified;
        let law = DistributionArtifact::new(metadata, prior.draws().to_vec())
            .map_err(JointPanelError::compile)?;
        Ok(Self {
            original,
            update: Arc::new(update),
            law,
            coefficients,
            dependency,
            retention,
            native_design: None,
        })
    }

    /// Actual updated joint law under caller-current evidence, with conditional scope.
    /// # Errors
    /// Withdrawal, source mismatch, overlap or invalid bridge authority refuses.
    pub fn law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        validate_update(&self.update, ledger)?;
        validate_model_dependency(self.original.dependency_identity(), &self.update, ledger)?;
        let projection = self
            .original
            .projection()
            .ok_or_else(|| JointPanelError::compile("missing native design projection"))?;
        let source = ScientificDecisionSource::new_projected(
            self.original.target().clone(),
            self.original.identification().clone(),
            self.dependency,
            ScientificSourcePayload::Law(Box::new(self.law.clone())),
            projection.clone(),
        )
        .map_err(super::super::search::map_scientific_error)?
        .retain_admission(Arc::clone(&self.retention));
        Ok(PanelExperimentDecisionSource {
            source,
            update: Arc::clone(&self.update),
            original_dependency: self.original.dependency_identity(),
            retention: Arc::clone(&self.retention),
        })
    }

    /// Evaluate a terminal decision under the declared conditional experiment law.
    /// # Errors
    /// Current evidence, source requirements or utility/feasibility semantics refuse.
    pub fn decide(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
        ledger: &EvidenceLedger,
    ) -> Result<antecedent_design::scientific_source::ScopedDecisionResult, JointPanelError> {
        self.law_source(ledger)?.evaluate(
            contract,
            antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
            ledger,
        )
    }

    /// Rank independent future samples of the same translated native estimand.
    /// # Errors
    /// Current evidence, future-study definitions, utilities, resources or cancellation refuse.
    pub fn rank_next_samples<A: serde::Serialize>(
        &self,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        candidates: &[PanelFutureSample],
        replicates: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
        let scientific = self.law_source(ledger)?;
        rank_law_next_samples(
            &ExperimentRankingReceipt::from(&scientific),
            &self.law,
            &self.update,
            &self.coefficients,
            problem,
            candidates,
            replicates,
            ctx,
        )
    }

    /// Historical numerical component; current authority must still pass first.
    /// # Errors
    /// Withdrawn, overlapping or changed evidence refuses consumption.
    pub fn numerical_law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<&DistributionArtifact, JointPanelError> {
        validate_update(&self.update, ledger)?;
        validate_model_dependency(self.original.dependency_identity(), &self.update, ledger)?;
        self.retention.observe_current();
        Ok(&self.law)
    }
}

fn contrast_binding(
    source: &ScientificDecisionSource,
) -> Result<(antecedent_contract::ScientificQuantity, Vec<f64>), JointPanelError> {
    let antecedent_core::TargetForm::Contrast(contrast) = source.target().form() else {
        return Err(JointPanelError::compile(
            "native experiment requires an assembled target contrast",
        ));
    };
    let projection = source.projection().ok_or_else(|| {
        JointPanelError::compile("native experiment needs an authenticated world-mean projection")
    })?;
    let mut coefficients = vec![0.0; projection.outputs.len()];
    let first = projection
        .outputs
        .first()
        .ok_or_else(|| JointPanelError::compile("empty native design response basis"))?
        .quantity
        .clone();
    let mut seen = std::collections::BTreeSet::new();
    for (i, output) in projection.outputs.iter().enumerate() {
        if output.terms.len() != 1 || output.terms[0].1.to_bits() != 1.0_f64.to_bits() {
            return Err(JointPanelError::Unsupported(
                "native experiment currently needs one whole-world mean per coordinate".into(),
            ));
        }
        let basis = projection
            .basis
            .get(output.terms[0].0)
            .ok_or_else(|| JointPanelError::compile("missing native design world basis"))?;
        if !seen.insert(basis.world)
            || output.quantity.population_id != first.population_id
            || output.quantity.units != first.units
            || output.quantity.horizon != first.horizon
        {
            return Err(JointPanelError::compile(
                "incompatible native world populations, units, horizons or duplicate worlds",
            ));
        }
        coefficients[i] = contrast
            .terms()
            .iter()
            .find(|term| term.world == basis.world)
            .map_or(0.0, |term| term.coefficient);
    }
    if contrast.terms().iter().any(|term| !seen.contains(&term.world)) {
        return Err(JointPanelError::compile("missing native contrast world"));
    }
    Ok((first, coefficients))
}

/// Independently replayed numerical experiment component, without current authority.
#[derive(Clone, Debug)]
pub struct HistoricalJointLawExperiment {
    law: DistributionArtifact,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    native_design_replayed: bool,
}
impl HistoricalJointLawExperiment {
    /// Historical aligned law, with no permission to consume withdrawn evidence.
    #[must_use]
    pub fn numerical_law(&self) -> &DistributionArtifact {
        self.retention.observe_current();
        &self.law
    }
    /// Whether replay executed the retained original design data and statistical recipe.
    #[must_use]
    pub const fn replays_original_design(&self) -> bool {
        self.native_design_replayed
    }

    /// A historical numerical replay never grants caller-current scientific authority.
    #[must_use]
    pub const fn grants_current_authority(&self) -> bool {
        false
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LawExperimentWire {
    version: u16,
    #[serde(default)]
    native_design: Option<Vec<u8>>,
    prior_metadata: antecedent_io::distribution_artifact::DistributionMetadata,
    prior_draws: Vec<u64>,
    coefficients: Vec<u64>,
    source_dependency: [u8; 32],
    bridge: antecedent_io::bridge_artifact::FullBridgeContractWire,
    ledger: antecedent_io::evidence_ledger_artifact::EvidenceLedgerMeta,
    origins: antecedent_io::joint_panel_experiment_artifact::PanelExperimentOrigins,
    observed: u64,
    variance: u64,
    sample_size: u64,
    intercepts: Vec<u64>,
    slopes: Vec<u64>,
}
const MAX_LAW_EXPERIMENT_BYTES: usize = 32 * 1024 * 1024;

impl JointLawExperimentResult {
    /// Retain the complete prior law, paired translation, likelihood and historical origins.
    /// # Errors
    /// Current authority, serialization, cancellation or resource admission fails.
    pub fn export(
        &self,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        let _current = self.law_source(ledger)?;
        source::check_cancelled(ctx)?;
        let ScientificSourcePayload::Law(prior) = self.original.as_numerical_source() else {
            return Err(JointPanelError::compile("missing native prior law"));
        };
        let mut metadata_quote = WireCounter(0);
        serde_json::to_writer(&mut metadata_quote, prior.metadata())
            .map_err(JointPanelError::compile)?;
        let bytes = self
            .retention
            .remainder()
            .bytes()
            .checked_mul(16)
            .and_then(|n| n.checked_add(u64::try_from(metadata_quote.0).ok()?.checked_mul(8)?))
            .ok_or_else(|| JointPanelError::compile("native experiment export quote overflow"))?;
        let native_bytes = self.native_design.as_ref().map_or(0, |n| n.bytes.len());
        let bytes = bytes
            .checked_add(
                u64::try_from(native_bytes).ok().and_then(|n| n.checked_mul(8)).ok_or_else(
                    || JointPanelError::compile("native design export quote overflow"),
                )?,
            )
            .ok_or_else(|| JointPanelError::compile("native design export quote overflow"))?;
        let _export_lease = admit(ctx, bytes)?;
        let (_, history) =
            antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::seal(ledger, &[])
                .map_err(JointPanelError::compile)?;
        let wire = LawExperimentWire {
            version: 2,
            native_design: self.native_design.as_ref().map(|n| n.bytes.clone()),
            prior_metadata: prior.metadata().clone(),
            prior_draws: prior.draws().iter().map(|v| v.to_bits()).collect(),
            coefficients: self.coefficients.iter().map(|v| v.to_bits()).collect(),
            source_dependency: *self.original.dependency_identity().as_bytes(),
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
        };
        let bytes = serde_json::to_vec(&wire).map_err(JointPanelError::compile)?;
        if bytes.len() > MAX_LAW_EXPERIMENT_BYTES {
            return Err(JointPanelError::compile(
                "native experiment artifact exceeds bounded wire",
            ));
        }
        source::check_cancelled(ctx)?;
        Ok(bytes)
    }

    /// Replay an exported experiment using only its retained numerical law and inputs.
    /// # Errors
    /// Unknown version/fields, invalid historical evidence or altered numerical inputs refuse.
    pub fn replay(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<HistoricalJointLawExperiment, JointPanelError> {
        let (wire, retention) = decode_wire(bytes, ctx)?;
        let history = antecedent_io::evidence_ledger_artifact::EvidenceLedgerArtifact::from_meta(
            &wire.ledger,
            None,
        )
        .map_err(JointPanelError::compile)?;
        let update = wire.update()?;
        validate_update(&update, history.ledger())?;
        let source_dependency = SemanticDigest::from_bytes(wire.source_dependency);
        validate_model_dependency(source_dependency, &update, history.ledger())?;
        let native_design_replayed = if let Some(bytes) = &wire.native_design {
            let native = super::super::designs::replay_joint_design_execution(bytes, ctx)?;
            let original = native.posterior_source()?;
            let ScientificSourcePayload::Law(law) = original.as_numerical_source() else {
                return Err(JointPanelError::compile(
                    "replayed original native design has no posterior law",
                ));
            };
            let (_, coefficients) = contrast_binding(&original)?;
            if original.dependency_identity() != source_dependency
                || law.metadata() != &wire.prior_metadata
                || wire.prior_draws.iter().copied().ne(law.draws().iter().map(|v| v.to_bits()))
                || wire.coefficients.iter().copied().ne(coefficients.iter().map(|v| v.to_bits()))
                || update.bridge.target.quantity != Self::quantity(&original)?
            {
                return Err(JointPanelError::compile(
                    "native design replay does not reproduce the original target, law, contrast or dependency",
                ));
            }
            true
        } else {
            false
        };
        let prior = DistributionArtifact::new(
            wire.prior_metadata,
            wire.prior_draws.iter().map(|v| f64::from_bits(*v)).collect(),
        )
        .map_err(JointPanelError::compile)?;
        if !matches!(
            prior.semantic(),
            antecedent_io::quantity_wire::DistributionMeaningWire::ParameterPosterior
                | antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
        ) {
            return Err(JointPanelError::compile(
                "native experimental prior is not a posterior law",
            ));
        }
        let coefficients = wire.coefficients.iter().map(|v| f64::from_bits(*v)).collect::<Vec<_>>();
        let values =
            antecedent_prob::finite_belief::linear_functional_rows(prior.draws(), &coefficients)
                .map_err(JointPanelError::compile)?;
        validate_mapping_moments(&update, &values)?;
        let masses = prior
            .metadata()
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0 / prior.n_draws() as f64; prior.n_draws()]);
        let weights = antecedent_prob::finite_belief::AffineGaussianExperiment {
            observed_mean: update.observed_mean,
            observation_variance: update.observation_variance,
            sample_size: update.sample_size,
            target_values: &values,
            intercepts: &update.mapping.intercepts,
            slopes: &update.mapping.slopes,
        }
        .condition(&masses)
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
                source_dependency,
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
        metadata.identity.provider_id = "antecedent.joint_design.conditional_experiment".into();
        metadata.calibration = DistributionCalibration::Unmeasured;
        metadata.trust = DistributionTrust::Unverified;
        let law = DistributionArtifact::new(metadata, prior.draws().to_vec())
            .map_err(JointPanelError::compile)?;
        source::check_cancelled(ctx)?;
        Ok(HistoricalJointLawExperiment { law, dependency, retention, native_design_replayed })
    }

    /// Bind replay to this actual source and caller-current experiment authority.
    /// # Errors
    /// Another original law, target, mapping, likelihood or withdrawn evidence refuses.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        let current = self.law_source(ledger)?;
        let (wire, _lease) = decode_wire(bytes, ctx)?;
        let ScientificSourcePayload::Law(prior) = self.original.as_numerical_source() else {
            return Err(JointPanelError::compile("missing original native law"));
        };
        if wire.native_design.as_deref() != self.native_design.as_ref().map(|n| n.bytes.as_slice())
            || wire.source_dependency != *self.original.dependency_identity().as_bytes()
            || wire.prior_metadata != *prior.metadata()
            || wire.prior_draws.iter().copied().ne(prior.draws().iter().map(|v| v.to_bits()))
            || wire.coefficients.iter().copied().ne(self.coefficients.iter().map(|v| v.to_bits()))
        {
            return Err(JointPanelError::compile(
                "artifact names another original native law or contrast",
            ));
        }
        let replay = Self::replay(bytes, ctx)?;
        if replay.dependency != self.dependency
            || replay.law.metadata() != self.law.metadata()
            || replay.law.draws() != self.law.draws()
        {
            return Err(JointPanelError::compile(
                "artifact differs from this current native experiment update",
            ));
        }
        Ok(current)
    }
}

fn decode_wire(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(LawExperimentWire, Arc<antecedent_core::execution::AdmissionLease>), JointPanelError> {
    source::check_cancelled(ctx)?;
    if bytes.is_empty() || bytes.len() > MAX_LAW_EXPERIMENT_BYTES {
        return Err(JointPanelError::compile("native experiment artifact exceeds bounded wire"));
    }
    let quote = bytes
        .len()
        .checked_mul(32)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| JointPanelError::compile("native experiment artifact quote overflow"))?;
    let retention = Arc::new(admit(ctx, quote)?);
    let wire: LawExperimentWire =
        serde_json::from_slice(bytes).map_err(JointPanelError::compile)?;
    if ![1, 2].contains(&wire.version) || (wire.version == 1 && wire.native_design.is_some()) {
        return Err(JointPanelError::compile("unknown native experiment artifact version"));
    }
    Ok((wire, retention))
}
impl LawExperimentWire {
    fn update(&self) -> Result<PanelExperimentUpdate, JointPanelError> {
        Ok(PanelExperimentUpdate {
            bridge: self.bridge.to_domain().map_err(JointPanelError::compile)?,
            functional: JointPanelResponseFunctional::TerminalMean,
            observed_mean: f64::from_bits(self.observed),
            observation_variance: f64::from_bits(self.variance),
            sample_size: self.sample_size,
            modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
            mapping: PanelExperimentMapping {
                dependency: SemanticDigest::from_bytes(self.source_dependency),
                intercepts: self.intercepts.iter().map(|v| f64::from_bits(*v)).collect(),
                slopes: self.slopes.iter().map(|v| f64::from_bits(*v)).collect(),
            },
            evidence: PanelExperimentEvidence {
                model: SourceId::from_hex(&self.origins.model)
                    .ok_or_else(|| JointPanelError::compile("invalid model origin digest"))?,
                prior_origins: self
                    .origins
                    .priors
                    .iter()
                    .map(|id| {
                        SourceId::from_hex(id)
                            .ok_or_else(|| JointPanelError::compile("invalid prior origin digest"))
                    })
                    .collect::<Result<_, _>>()?,
                experiment: SourceId::from_hex(&self.origins.experiment)
                    .ok_or_else(|| JointPanelError::compile("invalid experiment origin digest"))?,
                mapping: SourceId::from_hex(&self.origins.mapping)
                    .ok_or_else(|| JointPanelError::compile("invalid mapping origin digest"))?,
            },
        })
    }
}

#[derive(Debug)]
struct NativeDesignReplay {
    bytes: Vec<u8>,
    _retention: Arc<antecedent_core::execution::AdmissionLease>,
}

impl super::super::designs::JointDesignResult {
    /// Condition the actual native posterior and retain its complete original-data replay.
    /// # Errors
    /// No posterior, invalid translation/current evidence, resources or cancellation.
    pub fn condition_experiment(
        &self,
        request: JointLawExperimentRequest,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<JointLawExperimentResult, JointPanelError> {
        let mut result =
            JointLawExperimentResult::condition(self.posterior_source()?, request, ledger, ctx)?;
        let bytes = self.export_artifact()?;
        let quote =
            bytes.len().checked_mul(8).and_then(|n| u64::try_from(n).ok()).ok_or_else(|| {
                JointPanelError::compile("native design replay retention quote overflow")
            })?;
        let retention = Arc::new(admit(ctx, quote)?);
        result.native_design = Some(Arc::new(NativeDesignReplay { bytes, _retention: retention }));
        Ok(result)
    }
}

pub(super) struct WireCounter(pub(super) usize);
impl std::io::Write for WireCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("native experiment wire quote overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
