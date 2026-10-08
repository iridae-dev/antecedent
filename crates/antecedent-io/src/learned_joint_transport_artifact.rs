//! Independent artifact of the learned joint source-target transport row (2.3A cell X4
//! remainder and 2.3B B1 model-provider row `learned_joint_transport`). Internal and
//! non-routed: the public interval route `antecedent.transport.learned_joint` stays closed
//! (`cell_not_licensed`, `learned_joint_transport.route_frozen`) until calibration is
//! measured, so nothing here is a published interval.
//!
//! Format version 1. The artifact carries the model declaration (graph class, polynomial
//! basis degree, varying block, sharing, dependence, bounds), both priors with their
//! provenance, the checked 2.2 transport identification the model reads, the draw request,
//! the source and target identities with the observations themselves, and the result: the
//! exact posterior moments of the `antecedent-learn` fit, the aligned joint draws with their
//! RNG identity, the diagnostics (including the basis rank), the graph / provider / query
//! record and the calibration status (always `unmeasured`).
//!
//! A consumer trusts none of the result. Under its own bounds (a stored count above them
//! refuses before any work) it checks the two digests, re-fits the model from the stored
//! observations through the Rust core (which fits the outcome mechanism through
//! `antecedent-learn`), and accepts only a result identical bit for bit: moments, effects,
//! every aligned draw, diagnostics, provider record and identification record. A changed
//! prior, basis degree, source or target observation, draw count, seed or RNG id therefore
//! refuses. Prior-bank/likelihood double use refuses in the re-fit exactly as it does at
//! production. The premises digest (scientific declaration) and the data-identity digest
//! (every source and target identity and observation) are separate, so refreshed data
//! replaces only the latter.
//!
//! The artifact embeds the observations (units), not sufficient statistics, because the
//! core's fit entry takes units and a re-fit from units reproduces the draws bit for bit.
//! What replay does not protect against: a producer that supplies fabricated observations
//! consistently, and the 2.2 derivation the identification record copies, which is a
//! declared premise bound into the premises digest, not re-derived here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::joint_bayesian_transport_artifact::{
    DataIdentityWire, GaussianPriorWire, JointDrawsWire, JointOptionsWire, JointSourceWire,
    JointTargetWire, PriorProvenanceWire,
};
use crate::transport_interference_wire::{
    TransportIdentificationWire, transport_identification_from_wire,
    transport_identification_to_wire,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JOINT_TRANSPORT_MAX_DRAWS, JOINT_TRANSPORT_MAX_PARAMETERS,
    JointPriors, JointTransportCalibration, JointTransportOptions, JointTransportRefusal,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_estimate::learned_joint_transport::{
    LearnedJointFit, LearnedJointModel, fit_learned_joint_transport,
};
use antecedent_identify::TransportIdentification;
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const LEARNED_JOINT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const LEARNED_JOINT_ARTIFACT_FEATURE: &str = "learned_joint_transport_conjugate_v1";
/// The only calibration status of this cell until coverage is measured.
pub const LEARNED_JOINT_CALIBRATION: &str = "unmeasured";

/// Why a learned joint transport artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LearnedJointArtifactError {
    /// The feature marker, a tag or a claim is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored count or size exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored observations.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The stored identity is not the one the consumer expects.
    #[error("identity differs from the consumer's expectation: {0}")]
    ExpectationMismatch(&'static str),
    /// The re-fitted result differs from the stored result.
    #[error("posterior result does not replay")]
    ResultMismatch,
}

impl From<LearnedJointArtifactError> for IoError {
    fn from(error: LearnedJointArtifactError) -> Self {
        Self::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("learned joint transport artifact: {error}"),
        }
    }
}

fn engine_refusal(refusal: &JointTransportRefusal) -> IoError {
    IoError::Refused {
        code: refusal.code,
        message: format!("{}: {}", refusal.detail, refusal.message),
    }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct LearnedJointConsumeLimits {
    /// Most sources.
    pub max_sources: usize,
    /// Most source units over all sources.
    pub max_source_rows: usize,
    /// Most target units.
    pub max_target_rows: usize,
    /// Most covariates.
    pub max_features: usize,
    /// Most posterior draws.
    pub max_draws: usize,
    /// Most stored draw values (`draws * coordinates`).
    pub max_draw_cells: usize,
    /// Most stored datum ids over all identities.
    pub max_datum_ids: usize,
}

impl Default for LearnedJointConsumeLimits {
    fn default() -> Self {
        Self {
            max_sources: 64,
            max_source_rows: 200_000,
            max_target_rows: 200_000,
            max_features: 32,
            max_draws: JOINT_TRANSPORT_MAX_DRAWS,
            max_draw_cells: 4_000_000,
            max_datum_ids: 400_000,
        }
    }
}

/// An identity a consumer expects the artifact to carry; `None` skips that check.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LearnedJointExpectation {
    /// Expected premises digest (model, basis, priors, identification, request).
    pub premises_digest: Option<String>,
    /// Expected data-identity digest (source and target observations).
    pub data_digest: Option<String>,
}

/// The stored model declaration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearnedModelWire {
    /// `fixed_dag`, `admg` or `graph_posterior`.
    pub graph: String,
    /// Raw covariate variable ids.
    pub features: Vec<u32>,
    /// Polynomial degree of the basis in every covariate.
    pub basis_degree: usize,
    /// `intercept` or `intercept_and_covariates`.
    pub varying: String,
    /// `independent_varying_blocks` or `shared_varying_block`.
    pub sharing: String,
    /// `independent_samples`, `overlapping_units` or `unknown`.
    pub dependence: String,
    /// Prior of the invariant block.
    pub invariant_prior: GaussianPriorWire,
    /// Prior of every varying block.
    pub varying_prior: GaussianPriorWire,
    /// Largest accepted unsupported target mass.
    pub max_unsupported_mass: f64,
    /// Standardized disagreement above which two sources are flagged.
    pub conflict_z_threshold: f64,
}

fn identity_to_wire(identity: &DataIdentity) -> DataIdentityWire {
    DataIdentityWire {
        snapshot_digest: identity.snapshot_digest.clone(),
        datum_ids: identity.datum_ids.clone(),
    }
}

fn identity_from_wire(wire: &DataIdentityWire) -> DataIdentity {
    DataIdentity {
        snapshot_digest: wire.snapshot_digest.clone(),
        datum_ids: wire.datum_ids.clone(),
    }
}

fn prior_to_wire(prior: &GaussianPrior) -> GaussianPriorWire {
    GaussianPriorWire {
        mean: prior.mean.clone(),
        covariance: prior.covariance.clone(),
        provenance: match &prior.provenance {
            PriorProvenance::Declared => PriorProvenanceWire::Declared,
            PriorProvenance::Bank { bank_id, consumed } => PriorProvenanceWire::Bank {
                bank_id: bank_id.clone(),
                consumed: consumed.iter().map(identity_to_wire).collect(),
            },
        },
    }
}

fn prior_from_wire(wire: &GaussianPriorWire) -> GaussianPrior {
    GaussianPrior {
        mean: wire.mean.clone(),
        covariance: wire.covariance.clone(),
        provenance: match &wire.provenance {
            PriorProvenanceWire::Declared => PriorProvenance::Declared,
            PriorProvenanceWire::Bank { bank_id, consumed } => PriorProvenance::Bank {
                bank_id: bank_id.clone(),
                consumed: consumed.iter().map(identity_from_wire).collect(),
            },
        },
    }
}

fn source_to_wire(source: &SourceData) -> JointSourceWire {
    JointSourceWire {
        id: source.id.clone(),
        identity: identity_to_wire(&source.identity),
        treatment: source.treatment.clone(),
        outcome: source.outcome.clone(),
        covariates: source.covariates.clone(),
        noise_variance: source.noise_variance,
    }
}

fn source_from_wire(wire: &JointSourceWire) -> SourceData {
    SourceData {
        id: wire.id.clone(),
        identity: identity_from_wire(&wire.identity),
        treatment: wire.treatment.clone(),
        outcome: wire.outcome.clone(),
        covariates: wire.covariates.clone(),
        noise_variance: wire.noise_variance,
    }
}

fn graph_name(graph: TransportGraphClass) -> &'static str {
    match graph {
        TransportGraphClass::FixedDag => "fixed_dag",
        TransportGraphClass::Admg => "admg",
        TransportGraphClass::GraphPosterior => "graph_posterior",
    }
}

fn varying_name(varying: VaryingBlock) -> &'static str {
    match varying {
        VaryingBlock::Intercept => "intercept",
        VaryingBlock::InterceptAndCovariates => "intercept_and_covariates",
    }
}

fn sharing_name(sharing: SourceSharing) -> &'static str {
    match sharing {
        SourceSharing::IndependentVaryingBlocks => "independent_varying_blocks",
        SourceSharing::SharedVaryingBlock => "shared_varying_block",
    }
}

fn dependence_name(dependence: SourceDependence) -> &'static str {
    match dependence {
        SourceDependence::IndependentSamples => "independent_samples",
        SourceDependence::OverlappingUnits => "overlapping_units",
        SourceDependence::Unknown => "unknown",
    }
}

impl LearnedModelWire {
    fn from_model(model: &LearnedJointModel) -> Self {
        Self {
            graph: graph_name(model.graph).into(),
            features: model.features.clone(),
            basis_degree: model.basis_degree,
            varying: varying_name(model.varying).into(),
            sharing: sharing_name(model.sharing).into(),
            dependence: dependence_name(model.dependence).into(),
            invariant_prior: prior_to_wire(&model.priors.invariant),
            varying_prior: prior_to_wire(&model.priors.varying),
            max_unsupported_mass: model.max_unsupported_mass,
            conflict_z_threshold: model.conflict_z_threshold,
        }
    }

    fn to_model(&self) -> Result<LearnedJointModel, LearnedJointArtifactError> {
        let unsupported = LearnedJointArtifactError::UnsupportedSemantics;
        let graph = match self.graph.as_str() {
            "fixed_dag" => TransportGraphClass::FixedDag,
            "admg" => TransportGraphClass::Admg,
            "graph_posterior" => TransportGraphClass::GraphPosterior,
            _ => return Err(unsupported("graph class")),
        };
        let varying = match self.varying.as_str() {
            "intercept" => VaryingBlock::Intercept,
            "intercept_and_covariates" => VaryingBlock::InterceptAndCovariates,
            _ => return Err(unsupported("varying block")),
        };
        let sharing = match self.sharing.as_str() {
            "independent_varying_blocks" => SourceSharing::IndependentVaryingBlocks,
            "shared_varying_block" => SourceSharing::SharedVaryingBlock,
            _ => return Err(unsupported("source sharing")),
        };
        let dependence = match self.dependence.as_str() {
            "independent_samples" => SourceDependence::IndependentSamples,
            "overlapping_units" => SourceDependence::OverlappingUnits,
            "unknown" => SourceDependence::Unknown,
            _ => return Err(unsupported("source dependence")),
        };
        Ok(LearnedJointModel {
            graph,
            features: self.features.clone(),
            basis_degree: self.basis_degree,
            varying,
            sharing,
            dependence,
            priors: JointPriors {
                invariant: prior_from_wire(&self.invariant_prior),
                varying: prior_from_wire(&self.varying_prior),
            },
            max_unsupported_mass: self.max_unsupported_mass,
            conflict_z_threshold: self.conflict_z_threshold,
        })
    }
}

/// The stored graph / provider / query record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LearnedProviderWire {
    /// Provider crate (`antecedent-learn`).
    pub provider: String,
    /// Graph class (`fixed_dag`).
    pub graph: String,
    /// Query (`target_average_effect`).
    pub query: String,
    /// Model identity reported by the learn fit.
    pub learn_model_id: String,
    /// Backend implementation reported by the learn fit.
    pub learn_implementation: String,
    /// Basis declaration identity.
    pub basis_id: String,
    /// Basis term names in evaluation order.
    pub basis_terms: Vec<String>,
}

/// Stored diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearnedDiagnosticsWire {
    /// Sampler.
    pub sampler: String,
    /// Number of draws.
    pub draw_count: usize,
    /// Effective sample size.
    pub effective_sample_size: f64,
    /// Split R-hat of a chain path, when present.
    pub r_hat: Option<f64>,
    /// Smallest relative pivot of the weighted data information matrix.
    pub min_pivot_ratio: f64,
    /// Rank of the weighted data information matrix.
    pub basis_rank: usize,
    /// Condition lower bound of the learn fit's posterior precision.
    pub precision_condition: f64,
    /// Target mass outside the source support.
    pub unsupported_mass: f64,
    /// Declared tolerance of the unsupported mass.
    pub tolerance: f64,
    /// `(source, mean, variance)` of the target effect from each source alone.
    pub per_source: Vec<(String, f64, f64)>,
    /// `(left, right, z)` pairwise disagreement.
    pub pairs: Vec<(String, String, f64)>,
    /// Largest absolute standardized difference.
    pub max_abs_z: f64,
    /// Whether the disagreement is flagged.
    pub flagged: bool,
    /// Always `unmeasured`.
    pub calibration: String,
}

/// The stored posterior result and its identification record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearnedResultWire {
    /// Parameter names.
    pub parameter_names: Vec<String>,
    /// Exact posterior mean.
    pub posterior_mean: Vec<f64>,
    /// Exact posterior covariance, row-major.
    pub posterior_covariance: Vec<f64>,
    /// Effect names.
    pub effect_names: Vec<String>,
    /// Exact posterior mean of each effect.
    pub effect_means: Vec<f64>,
    /// Exact posterior covariance of the effects, row-major.
    pub effect_covariance: Vec<f64>,
    /// Posterior mean of the target effect.
    pub target_effect_mean: f64,
    /// Posterior variance of the target effect.
    pub target_effect_variance: f64,
    /// Aligned joint draws.
    pub draws: JointDrawsWire,
    /// Diagnostics.
    pub diagnostics: LearnedDiagnosticsWire,
    /// Graph / provider / query record.
    pub provider: LearnedProviderWire,
    /// Certificate rule of the checked derivation.
    pub identification_rule: String,
    /// `identified`: the only status a fit carries; never changed by a prior.
    pub identification_status: String,
    /// `direct` or `standardize`.
    pub identification_formula: String,
    /// Canonical identity of the model, basis, priors and bounds.
    pub model_identity: String,
}

impl LearnedResultWire {
    fn from_fit(fit: &LearnedJointFit) -> Self {
        let d = &fit.diagnostics;
        let calibration = match d.calibration {
            JointTransportCalibration::Unmeasured => LEARNED_JOINT_CALIBRATION,
        };
        let p = &fit.provider;
        Self {
            parameter_names: fit.parameter_names.clone(),
            posterior_mean: fit.posterior_mean.clone(),
            posterior_covariance: fit.posterior_covariance.clone(),
            effect_names: fit.effect_names.clone(),
            effect_means: fit.effect_means.clone(),
            effect_covariance: fit.effect_covariance.clone(),
            target_effect_mean: fit.target_effect_mean,
            target_effect_variance: fit.target_effect_variance,
            draws: JointDrawsWire {
                names: fit.draws.names.clone(),
                values: fit.draws.values.clone(),
                n_draws: fit.draws.n_draws,
                rng_id: fit.draws.rng_id.clone(),
                alignment: "joint".into(),
            },
            diagnostics: LearnedDiagnosticsWire {
                sampler: d.sampler.into(),
                draw_count: d.draw_count,
                effective_sample_size: d.effective_sample_size,
                r_hat: d.r_hat,
                min_pivot_ratio: d.min_pivot_ratio,
                basis_rank: d.basis_rank,
                precision_condition: d.precision_condition,
                unsupported_mass: d.overlap.unsupported_mass,
                tolerance: d.overlap.tolerance,
                per_source: d
                    .disagreement
                    .per_source
                    .iter()
                    .map(|s| (s.source_id.clone(), s.mean, s.variance))
                    .collect(),
                pairs: d
                    .disagreement
                    .pairs
                    .iter()
                    .map(|pair| (pair.left.clone(), pair.right.clone(), pair.z))
                    .collect(),
                max_abs_z: d.disagreement.max_abs_z,
                flagged: d.disagreement.flagged,
                calibration: calibration.into(),
            },
            provider: LearnedProviderWire {
                provider: p.provider.into(),
                graph: p.graph.into(),
                query: p.query.into(),
                learn_model_id: p.learn_model_id.clone(),
                learn_implementation: p.learn_implementation.clone(),
                basis_id: p.basis_id.clone(),
                basis_terms: p.basis_terms.clone(),
            },
            identification_rule: fit.identification.rule.clone(),
            identification_status: fit.identification.status.into(),
            identification_formula: fit.identification.formula.into(),
            model_identity: fit.model_identity.clone(),
        }
    }
}

/// Versioned learned joint transport result with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedJointArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Model declaration with basis and priors.
    pub model: LearnedModelWire,
    /// The checked 2.2 transport identification the model reads.
    pub identification: TransportIdentificationWire,
    /// Draw request.
    pub options: JointOptionsWire,
    /// Source trials.
    pub sources: Vec<JointSourceWire>,
    /// Target covariate sample.
    pub target: JointTargetWire,
    /// Posterior result.
    pub result: LearnedResultWire,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Digest of the scientific premises.
    pub premises_digest: String,
    /// Digest of the source and target data identities and observations.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    model: &'a LearnedModelWire,
    identification: &'a TransportIdentificationWire,
    options: &'a JointOptionsWire,
    sources: Vec<(&'a str, u64)>,
    bounds: (usize, usize),
}

#[derive(Serialize)]
struct DataView<'a> {
    tag: &'static str,
    sources: &'a [JointSourceWire],
    target: &'a JointTargetWire,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl LearnedJointArtifactWire {
    /// Fit the model through the Rust core and build the artifact from its inputs and
    /// result, with the fit returned alongside. Nothing is exported that a default
    /// consumer would refuse on size.
    ///
    /// # Errors
    /// An engine refusal (with its reason code and `learned_joint_transport.*` detail), a
    /// result larger than the default consumer bounds, or an encoding failure.
    pub fn build(
        id: &TransportIdentification,
        model: &LearnedJointModel,
        sources: &[SourceData],
        target: &TargetData,
        options: JointTransportOptions,
        ctx: &ExecutionContext,
    ) -> Result<(Self, LearnedJointFit), IoError> {
        let fit = fit_learned_joint_transport(id, model, sources, Some(target), &options, ctx)
            .map_err(|refusal| engine_refusal(&refusal))?;
        let mut wire = Self {
            version: LEARNED_JOINT_ARTIFACT_VERSION,
            required_features: vec![LEARNED_JOINT_ARTIFACT_FEATURE.into()],
            model: LearnedModelWire::from_model(model),
            identification: transport_identification_to_wire(id),
            options: JointOptionsWire { draws: options.draws, seed: options.seed },
            sources: sources.iter().map(source_to_wire).collect(),
            target: JointTargetWire {
                identity: identity_to_wire(&target.identity),
                rows: target.rows,
                covariates: target.covariates.clone(),
            },
            result: LearnedResultWire::from_fit(&fit),
            calibration: LEARNED_JOINT_CALIBRATION.into(),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.check_limits(&LearnedJointConsumeLimits::default())?;
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        Ok((wire, fit))
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-fits and compares the whole result.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let view = PremisesView {
            tag: "learned_joint_transport_premises_v1",
            model: &self.model,
            identification: &self.identification,
            options: &self.options,
            sources: self
                .sources
                .iter()
                .map(|s| (s.id.as_str(), s.noise_variance.to_bits()))
                .collect(),
            bounds: (JOINT_TRANSPORT_MAX_DRAWS, JOINT_TRANSPORT_MAX_PARAMETERS),
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    /// The data-identity digest the stored observations should carry.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let view = DataView {
            tag: "learned_joint_transport_data_v1",
            sources: &self.sources,
            target: &self.target,
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure or a foreign feature or claim.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != LEARNED_JOINT_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        let unsupported = LearnedJointArtifactError::UnsupportedSemantics;
        if wire.required_features != [LEARNED_JOINT_ARTIFACT_FEATURE] {
            return Err(unsupported("required features").into());
        }
        if wire.calibration != LEARNED_JOINT_CALIBRATION
            || wire.result.diagnostics.calibration != LEARNED_JOINT_CALIBRATION
        {
            return Err(unsupported("calibration is unmeasured").into());
        }
        if wire.result.draws.alignment != "joint" {
            return Err(unsupported("draws must be aligned joint draws").into());
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &LearnedJointConsumeLimits,
    ) -> Result<(), LearnedJointArtifactError> {
        let exceeded = LearnedJointArtifactError::LimitsExceeded;
        let rows: usize = self.sources.iter().map(|s| s.outcome.len()).sum();
        let priors = [&self.model.invariant_prior, &self.model.varying_prior];
        let ids: usize = self
            .sources
            .iter()
            .map(|s| s.identity.datum_ids.len())
            .chain([self.target.identity.datum_ids.len()])
            .chain(priors.iter().map(|p| match &p.provenance {
                PriorProvenanceWire::Declared => 0,
                PriorProvenanceWire::Bank { consumed, .. } => {
                    consumed.iter().map(|c| c.datum_ids.len()).sum::<usize>()
                }
            }))
            .sum();
        let prior_cells = JOINT_TRANSPORT_MAX_PARAMETERS * JOINT_TRANSPORT_MAX_PARAMETERS;
        if self.sources.len() > limits.max_sources {
            return Err(exceeded("source count"));
        }
        if rows > limits.max_source_rows {
            return Err(exceeded("source rows"));
        }
        if self.target.rows > limits.max_target_rows {
            return Err(exceeded("target rows"));
        }
        if self.model.features.len() > limits.max_features {
            return Err(exceeded("feature count"));
        }
        if self.options.draws > limits.max_draws
            || self.options.draws > JOINT_TRANSPORT_MAX_DRAWS
            || self.result.draws.n_draws > limits.max_draws
        {
            return Err(exceeded("draw count"));
        }
        if self.result.draws.values.len() > limits.max_draw_cells {
            return Err(exceeded("draw values"));
        }
        if ids > limits.max_datum_ids {
            return Err(exceeded("datum ids"));
        }
        if priors.iter().any(|p| {
            p.mean.len() > JOINT_TRANSPORT_MAX_PARAMETERS || p.covariance.len() > prior_cells
        }) {
            return Err(exceeded("prior size"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then re-fit the model
    /// from the stored observations and accept only an identical result. No external
    /// provider is accessed.
    ///
    /// # Errors
    /// A limit, digest, reconstruction or result mismatch, or the engine's own refusal
    /// (including prior-bank/likelihood double use) with its reason code and detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: LearnedJointConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, LearnedJointFit), IoError> {
        Self::consume_expecting(bytes, &LearnedJointExpectation::default(), limits, ctx)
    }

    /// [`Self::consume_with_limits`] that additionally requires the stored premises and
    /// data digests to equal the consumer's expected identity.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`], and an expectation mismatch.
    pub fn consume_expecting(
        bytes: &[u8],
        expected: &LearnedJointExpectation,
        limits: LearnedJointConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, LearnedJointFit), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(LearnedJointArtifactError::PremisesMismatch.into());
        }
        if wire.expected_data_digest()? != wire.data_digest {
            return Err(LearnedJointArtifactError::DataIdentityMismatch.into());
        }
        if expected.premises_digest.as_ref().is_some_and(|d| *d != wire.premises_digest) {
            return Err(LearnedJointArtifactError::ExpectationMismatch("premises").into());
        }
        if expected.data_digest.as_ref().is_some_and(|d| *d != wire.data_digest) {
            return Err(LearnedJointArtifactError::ExpectationMismatch("data identity").into());
        }
        let id = transport_identification_from_wire(&wire.identification);
        let model = wire.model.to_model()?;
        let sources: Vec<SourceData> = wire.sources.iter().map(source_from_wire).collect();
        let target = TargetData {
            identity: identity_from_wire(&wire.target.identity),
            rows: wire.target.rows,
            covariates: wire.target.covariates.clone(),
        };
        let options = JointTransportOptions { draws: wire.options.draws, seed: wire.options.seed };
        let fit = fit_learned_joint_transport(&id, &model, &sources, Some(&target), &options, ctx)
            .map_err(|refusal| engine_refusal(&refusal))?;
        let replayed = LearnedResultWire::from_fit(&fit);
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.result)? {
            return Err(LearnedJointArtifactError::ResultMismatch.into());
        }
        Ok((wire, fit))
    }
}
