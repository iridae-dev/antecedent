//! Independent artifact of the joint Bayesian source-target transport engine
//! (2.3A X4, record `2.3A.X4.joint_bayesian_transport`). Internal and non-routed: the
//! public interval route `antecedent.transport.joint_bayesian` stays closed
//! (`cell_not_licensed`, `bayesian_transport.route_frozen`) until calibration is
//! measured, so nothing here is a published interval.
//!
//! Format version 1. The artifact carries the model declaration (graph class, varying
//! block, sharing, dependence, bounds), both priors with their provenance (a prior bank
//! names every observation it consumed), the checked 2.2 transport identification the
//! model reads, the draw request (count and seed), the source and target data identities
//! with the observations themselves, and the result: exact posterior mean and covariance,
//! the aligned joint draws with their RNG identity, the diagnostics, and the calibration
//! status (always `unmeasured`).
//!
//! A consumer trusts none of the result. Under its own bounds (a stored count above them
//! refuses before any work) it checks the two digests, re-fits the conjugate model from
//! the stored observations through the engine, and accepts only a result identical bit
//! for bit: moments, effects, every aligned draw, diagnostics and identification record.
//! A changed prior, model, source or target observation, draw count, seed or RNG id
//! therefore refuses. Prior-bank/likelihood double use refuses in the re-fit exactly as it
//! does at production. The two digests are separate: the premises digest (scientific
//! declaration: model, priors with provenance, identification, request, noise variances,
//! bounds) and the data-identity digest (every source and target identity and
//! observation), so refreshed data replaces only the latter. A consumer holding an
//! expected identity passes it as a [`JointBayesianExpectation`].
//!
//! The artifact embeds the observations (units), not sufficient statistics: the engine's
//! only fit entry takes units, and a re-fit from units reproduces the draws bit for bit
//! under the engine's own summation order. What replay does not protect against: a
//! producer that supplies fabricated observations consistently, and the 2.2 derivation
//! the identification record copies, which is a declared premise bound into the premises
//! digest, not re-derived here (no graph is stored).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    IoError,
    transport_interference_wire::{
        TransportIdentificationWire, transport_identification_from_wire,
        transport_identification_to_wire,
    },
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JOINT_TRANSPORT_MAX_DRAWS, JOINT_TRANSPORT_MAX_PARAMETERS,
    JointPriors, JointTransportCalibration, JointTransportFit, JointTransportModel,
    JointTransportOptions, JointTransportRefusal, PriorProvenance, SourceData, SourceDependence,
    SourceSharing, TargetData, TransportGraphClass, VaryingBlock, fit_joint_bayesian_transport,
};
use antecedent_identify::TransportIdentification;
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const JOINT_BAYESIAN_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const JOINT_BAYESIAN_ARTIFACT_FEATURE: &str = "joint_bayesian_transport_conjugate_v1";
/// The only calibration status of this cell until coverage is measured.
pub const JOINT_BAYESIAN_CALIBRATION: &str = "unmeasured";

/// Why a joint Bayesian transport artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum JointBayesianArtifactError {
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

impl From<JointBayesianArtifactError> for IoError {
    fn from(error: JointBayesianArtifactError) -> Self {
        Self::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("joint bayesian transport artifact: {error}"),
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
pub struct JointBayesianConsumeLimits {
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

impl Default for JointBayesianConsumeLimits {
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
pub struct JointBayesianExpectation {
    /// Expected premises digest (model, priors, identification, request).
    pub premises_digest: Option<String>,
    /// Expected data-identity digest (source and target observations).
    pub data_digest: Option<String>,
}

/// Identity of the observations a likelihood or prior consumed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DataIdentityWire {
    /// Snapshot digest of the dataset.
    pub snapshot_digest: String,
    /// Stable ids of the individual data or units.
    pub datum_ids: Vec<String>,
}

impl DataIdentityWire {
    fn from_identity(identity: &DataIdentity) -> Self {
        Self {
            snapshot_digest: identity.snapshot_digest.clone(),
            datum_ids: identity.datum_ids.clone(),
        }
    }

    fn to_identity(&self) -> DataIdentity {
        DataIdentity {
            snapshot_digest: self.snapshot_digest.clone(),
            datum_ids: self.datum_ids.clone(),
        }
    }
}

/// Provenance of a stored prior.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorProvenanceWire {
    /// Declared without reference to data.
    Declared,
    /// Built from a prior bank; every consumed observation is named.
    Bank {
        /// Bank identity.
        bank_id: String,
        /// Observations the bank construction consumed.
        consumed: Vec<DataIdentityWire>,
    },
}

/// A stored dense Gaussian prior block.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GaussianPriorWire {
    /// Prior mean.
    pub mean: Vec<f64>,
    /// Row-major covariance.
    pub covariance: Vec<f64>,
    /// Provenance.
    pub provenance: PriorProvenanceWire,
}

impl GaussianPriorWire {
    fn from_prior(prior: &GaussianPrior) -> Self {
        Self {
            mean: prior.mean.clone(),
            covariance: prior.covariance.clone(),
            provenance: match &prior.provenance {
                PriorProvenance::Declared => PriorProvenanceWire::Declared,
                PriorProvenance::Bank { bank_id, consumed } => PriorProvenanceWire::Bank {
                    bank_id: bank_id.clone(),
                    consumed: consumed.iter().map(DataIdentityWire::from_identity).collect(),
                },
            },
        }
    }

    fn to_prior(&self) -> GaussianPrior {
        GaussianPrior {
            mean: self.mean.clone(),
            covariance: self.covariance.clone(),
            provenance: match &self.provenance {
                PriorProvenanceWire::Declared => PriorProvenance::Declared,
                PriorProvenanceWire::Bank { bank_id, consumed } => PriorProvenance::Bank {
                    bank_id: bank_id.clone(),
                    consumed: consumed.iter().map(DataIdentityWire::to_identity).collect(),
                },
            },
        }
    }
}

/// The stored model declaration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointModelWire {
    /// `fixed_dag`, `admg` or `graph_posterior`.
    pub graph: String,
    /// Raw covariate variable ids.
    pub features: Vec<u32>,
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

impl JointModelWire {
    fn from_model(model: &JointTransportModel) -> Self {
        Self {
            graph: graph_name(model.graph).into(),
            features: model.features.clone(),
            varying: varying_name(model.varying).into(),
            sharing: sharing_name(model.sharing).into(),
            dependence: dependence_name(model.dependence).into(),
            invariant_prior: GaussianPriorWire::from_prior(&model.priors.invariant),
            varying_prior: GaussianPriorWire::from_prior(&model.priors.varying),
            max_unsupported_mass: model.max_unsupported_mass,
            conflict_z_threshold: model.conflict_z_threshold,
        }
    }

    fn to_model(&self) -> Result<JointTransportModel, JointBayesianArtifactError> {
        let unsupported = JointBayesianArtifactError::UnsupportedSemantics;
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
        Ok(JointTransportModel {
            graph,
            features: self.features.clone(),
            varying,
            sharing,
            dependence,
            priors: JointPriors {
                invariant: self.invariant_prior.to_prior(),
                varying: self.varying_prior.to_prior(),
            },
            max_unsupported_mass: self.max_unsupported_mass,
            conflict_z_threshold: self.conflict_z_threshold,
        })
    }
}

/// The stored draw request.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JointOptionsWire {
    /// Number of aligned joint draws.
    pub draws: usize,
    /// Master seed of the draw stream.
    pub seed: u64,
}

/// One stored source trial.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointSourceWire {
    /// Source id.
    pub id: String,
    /// Identity of the observations.
    pub identity: DataIdentityWire,
    /// Treatment per unit.
    pub treatment: Vec<bool>,
    /// Outcome per unit.
    pub outcome: Vec<f64>,
    /// Covariate columns.
    pub covariates: Vec<Vec<f64>>,
    /// Declared known noise variance.
    pub noise_variance: f64,
}

impl JointSourceWire {
    fn from_source(source: &SourceData) -> Self {
        Self {
            id: source.id.clone(),
            identity: DataIdentityWire::from_identity(&source.identity),
            treatment: source.treatment.clone(),
            outcome: source.outcome.clone(),
            covariates: source.covariates.clone(),
            noise_variance: source.noise_variance,
        }
    }

    fn to_source(&self) -> SourceData {
        SourceData {
            id: self.id.clone(),
            identity: self.identity.to_identity(),
            treatment: self.treatment.clone(),
            outcome: self.outcome.clone(),
            covariates: self.covariates.clone(),
            noise_variance: self.noise_variance,
        }
    }
}

/// The stored target covariate sample.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointTargetWire {
    /// Identity of the observations.
    pub identity: DataIdentityWire,
    /// Number of units.
    pub rows: usize,
    /// Covariate columns.
    pub covariates: Vec<Vec<f64>>,
}

/// Stored aligned joint draws.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointDrawsWire {
    /// Coordinate names (parameters, source effects, target effect).
    pub names: Vec<String>,
    /// Draw-major values.
    pub values: Vec<f64>,
    /// Number of draws.
    pub n_draws: usize,
    /// RNG algorithm, seed and stream identity.
    pub rng_id: String,
    /// Always `joint`: one row is one realization of every coordinate.
    pub alignment: String,
}

/// Stored diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointDiagnosticsWire {
    /// Sampler.
    pub sampler: String,
    /// Number of draws.
    pub draw_count: usize,
    /// Effective sample size.
    pub effective_sample_size: f64,
    /// Split R-hat of a chain path, when present.
    pub r_hat: Option<f64>,
    /// Smallest relative pivot of the data information matrix.
    pub min_pivot_ratio: f64,
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
pub struct JointResultWire {
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
    pub diagnostics: JointDiagnosticsWire,
    /// Certificate rule of the checked derivation.
    pub identification_rule: String,
    /// `identified`: the only status a fit carries; never changed by a prior.
    pub identification_status: String,
    /// `direct` or `standardize`.
    pub identification_formula: String,
    /// Canonical identity of the model, priors and bounds.
    pub model_identity: String,
}

impl JointResultWire {
    fn from_fit(fit: &JointTransportFit) -> Self {
        let d = &fit.diagnostics;
        let calibration = match d.calibration {
            JointTransportCalibration::Unmeasured => JOINT_BAYESIAN_CALIBRATION,
        };
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
            diagnostics: JointDiagnosticsWire {
                sampler: d.sampler.into(),
                draw_count: d.draw_count,
                effective_sample_size: d.effective_sample_size,
                r_hat: d.r_hat,
                min_pivot_ratio: d.min_pivot_ratio,
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
                    .map(|p| (p.left.clone(), p.right.clone(), p.z))
                    .collect(),
                max_abs_z: d.disagreement.max_abs_z,
                flagged: d.disagreement.flagged,
                calibration: calibration.into(),
            },
            identification_rule: fit.identification.rule.clone(),
            identification_status: fit.identification.status.into(),
            identification_formula: fit.identification.formula.into(),
            model_identity: fit.model_identity.clone(),
        }
    }
}

/// Versioned joint Bayesian transport result with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointBayesianArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Model declaration with priors and provenance.
    pub model: JointModelWire,
    /// The checked 2.2 transport identification the model reads.
    pub identification: TransportIdentificationWire,
    /// Draw request.
    pub options: JointOptionsWire,
    /// Source trials.
    pub sources: Vec<JointSourceWire>,
    /// Target covariate sample.
    pub target: JointTargetWire,
    /// Posterior result.
    pub result: JointResultWire,
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
    model: &'a JointModelWire,
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

impl JointBayesianArtifactWire {
    /// Fit the model through the engine and build the artifact from its inputs and
    /// result, with the fit returned alongside. Nothing is exported that a default
    /// consumer would refuse on size.
    ///
    /// # Errors
    /// An engine refusal (with its reason code and `bayesian_transport.*` detail), a
    /// result larger than the default consumer bounds, or an encoding failure.
    pub fn build(
        id: &TransportIdentification,
        model: &JointTransportModel,
        sources: &[SourceData],
        target: &TargetData,
        options: JointTransportOptions,
        ctx: &ExecutionContext,
    ) -> Result<(Self, JointTransportFit), IoError> {
        let fit = fit_joint_bayesian_transport(id, model, sources, Some(target), &options, ctx)
            .map_err(|refusal| engine_refusal(&refusal))?;
        let mut wire = Self {
            version: JOINT_BAYESIAN_ARTIFACT_VERSION,
            required_features: vec![JOINT_BAYESIAN_ARTIFACT_FEATURE.into()],
            model: JointModelWire::from_model(model),
            identification: transport_identification_to_wire(id),
            options: JointOptionsWire { draws: options.draws, seed: options.seed },
            sources: sources.iter().map(JointSourceWire::from_source).collect(),
            target: JointTargetWire {
                identity: DataIdentityWire::from_identity(&target.identity),
                rows: target.rows,
                covariates: target.covariates.clone(),
            },
            result: JointResultWire::from_fit(&fit),
            calibration: JOINT_BAYESIAN_CALIBRATION.into(),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.check_limits(&JointBayesianConsumeLimits::default())?;
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
            tag: "joint_bayesian_transport_premises_v1",
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
            tag: "joint_bayesian_transport_data_v1",
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
        if peek.version != JOINT_BAYESIAN_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        let unsupported = JointBayesianArtifactError::UnsupportedSemantics;
        if wire.required_features != [JOINT_BAYESIAN_ARTIFACT_FEATURE] {
            return Err(unsupported("required features").into());
        }
        if wire.calibration != JOINT_BAYESIAN_CALIBRATION
            || wire.result.diagnostics.calibration != JOINT_BAYESIAN_CALIBRATION
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
        limits: &JointBayesianConsumeLimits,
    ) -> Result<(), JointBayesianArtifactError> {
        let exceeded = JointBayesianArtifactError::LimitsExceeded;
        let rows: usize = self.sources.iter().map(|s| s.outcome.len()).sum();
        let ids: usize = self
            .sources
            .iter()
            .map(|s| s.identity.datum_ids.len())
            .chain([self.target.identity.datum_ids.len()])
            .chain([&self.model.invariant_prior, &self.model.varying_prior].iter().map(|p| {
                match &p.provenance {
                    PriorProvenanceWire::Declared => 0,
                    PriorProvenanceWire::Bank { consumed, .. } => {
                        consumed.iter().map(|c| c.datum_ids.len()).sum::<usize>()
                    }
                }
            }))
            .sum();
        let prior_cells = JOINT_TRANSPORT_MAX_PARAMETERS * JOINT_TRANSPORT_MAX_PARAMETERS;
        let priors = [&self.model.invariant_prior, &self.model.varying_prior];
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
        limits: JointBayesianConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, JointTransportFit), IoError> {
        Self::consume_expecting(bytes, &JointBayesianExpectation::default(), limits, ctx)
    }

    /// [`Self::consume_with_limits`] that additionally requires the stored premises and
    /// data digests to equal the consumer's expected identity.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`], and an expectation mismatch.
    pub fn consume_expecting(
        bytes: &[u8],
        expected: &JointBayesianExpectation,
        limits: JointBayesianConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, JointTransportFit), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(JointBayesianArtifactError::PremisesMismatch.into());
        }
        if wire.expected_data_digest()? != wire.data_digest {
            return Err(JointBayesianArtifactError::DataIdentityMismatch.into());
        }
        if expected.premises_digest.as_ref().is_some_and(|d| *d != wire.premises_digest) {
            return Err(JointBayesianArtifactError::ExpectationMismatch("premises").into());
        }
        if expected.data_digest.as_ref().is_some_and(|d| *d != wire.data_digest) {
            return Err(JointBayesianArtifactError::ExpectationMismatch("data identity").into());
        }
        let id = transport_identification_from_wire(&wire.identification);
        let model = wire.model.to_model()?;
        let sources: Vec<SourceData> =
            wire.sources.iter().map(JointSourceWire::to_source).collect();
        let target = TargetData {
            identity: wire.target.identity.to_identity(),
            rows: wire.target.rows,
            covariates: wire.target.covariates.clone(),
        };
        let options = JointTransportOptions { draws: wire.options.draws, seed: wire.options.seed };
        let fit = fit_joint_bayesian_transport(&id, &model, &sources, Some(&target), &options, ctx)
            .map_err(|refusal| engine_refusal(&refusal))?;
        let replayed = JointResultWire::from_fit(&fit);
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.result)? {
            return Err(JointBayesianArtifactError::ResultMismatch.into());
        }
        Ok((wire, fit))
    }
}
