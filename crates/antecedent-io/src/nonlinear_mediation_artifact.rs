//! B4 nonlinear continuous-mediator mediation artifact (`nonlinear_mediation_v1`).
//!
//! A bounded, checksummed, sectioned container holding one natural-effects estimate:
//!
//! * `nonlinear_mediation.meta` (CBOR): version, feature marker, the estimand
//!   (`natural_effects`), the premises with their declared or checked status, the model
//!   specifications, the configuration, the NDE / NIE / total effects, the integration error of
//!   the declared `n` versus `2n` Gauss-Hermite nodes, the overlap diagnostics, the bootstrap
//!   record (seed and replicate ids, interval status closed), the identity digests, the
//!   **unmeasured** calibration coordinate and the standing caveats;
//! * `nonlinear_mediation.data` (little-endian `f64`): the compact analysis dataset
//!   (treatment, mediator, outcome, then each covariate), capped at
//!   [`NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS`] rows.
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! declarations and the embedded dataset, re-runs the core estimator (including the seeded
//! bootstrap) and refuses unless the whole stored record is bit-identical to the recomputed one.
//! It recomputes the identity digests from the same inputs and compares them with the stored ones
//! and, when the caller retained an identity of its own, with that: a changed premise, model
//! specification, dataset or result is refused even when the artifact was resealed consistently.
//!
//! What this does **not** say: bootstrap standard errors carry no coverage claim
//! (`calibration = "unmeasured"`), no public interval is produced
//! (`interval_status = "closed_calibration_unmeasured"`), and the cross-world independence and
//! sequential ignorability that identify the natural effects are declared, not tested. Unknown
//! major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_estimate::EstimationError;
use antecedent_estimate::nonlinear_mediation::{
    NONLINEAR_MEDIATION_CALIBRATION, NONLINEAR_MEDIATION_INTERVAL_STATUS, NonlinearMediationConfig,
    NonlinearMediationEstimand, NonlinearMediationEstimate, NonlinearMediationInput,
    NonlinearMediationPremises, estimate_nonlinear_mediation,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const NONLINEAR_MEDIATION_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const NONLINEAR_MEDIATION_ARTIFACT_FEATURE: &str = "nonlinear_mediation_v1";
/// Most analysis rows an artifact may embed, enforced on export and on consumption.
pub const NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS: usize = 20_000;
/// Most covariates an artifact may embed.
pub const NONLINEAR_MEDIATION_ARTIFACT_MAX_COVARIATES: usize = 64;
/// Most bootstrap replicates a producer may request and a consumer will replay.
pub const NONLINEAR_MEDIATION_ARTIFACT_MAX_BOOTSTRAP: u32 = 2_000;
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Inference claim of the artifact: point estimates and diagnostics, no interval.
pub const NONLINEAR_MEDIATION_ARTIFACT_CLAIM: &str = "point_with_diagnostics";

const ARTIFACT_KIND: &str = "nonlinear_mediation_v1";
const META_SECTION: &str = "nonlinear_mediation_meta";
const DATA_SECTION: &str = "nonlinear_mediation_data";
const ESTIMAND_NATURAL: &str = "natural_effects";
const ESTIMAND_INTERVENTIONAL: &str = "interventional_effects";
const STATUS_DECLARED: &str = "declared";
const STATUS_CHECKED: &str = "checked";

/// Standing caveats attached to every artifact.
pub const NONLINEAR_MEDIATION_ARTIFACT_CAVEATS: [&str; 2] = [
    "bootstrap standard errors are calibration unmeasured: no coverage claim is made and no \
     public interval is produced; the point estimates and diagnostics are the claim",
    "natural direct and indirect effects need cross-world independence and sequential \
     ignorability, which are declared and cannot be tested from the data",
];

/// Why a nonlinear-mediation artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NonlinearMediationArtifactError {
    /// The bytes do not decode as this format.
    #[error("nonlinear mediation artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported nonlinear mediation artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim, calibration coordinate or caveat that is not this format's.
    #[error("unsupported nonlinear mediation semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("nonlinear mediation consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed nonlinear mediation artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("nonlinear_mediation.wrong_contract: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored record differs from the recomputed one.
    #[error("stored nonlinear mediation record does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("nonlinear mediation artifact does not encode: {0}")]
    Encode(String),
    /// The core estimator refused (registered reason code; the message begins with its detail).
    #[error("{message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// `detail: explanation`.
        message: String,
    },
}

impl NonlinearMediationArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a core refusal, a changed identity field and a record that does not replay;
    /// absent for corruption, unsupported versions and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, message } => {
                let (detail, text) = message.split_once(": ").unwrap_or((message.as_str(), ""));
                Some((*code, detail.to_owned(), text.to_owned()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "nonlinear_mediation.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            Self::ResultMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "nonlinear_mediation.report_replay_mismatch".to_owned(),
                format!("the stored {what} does not replay"),
            )),
            _ => None,
        }
    }
}

impl From<EstimationError> for NonlinearMediationArtifactError {
    fn from(error: EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message }
            | EstimationError::RefusedWithFields { code, message, .. } => {
                Self::Refused { code, message }
            }
            other => Self::Malformed(other.to_string()),
        }
    }
}

impl From<crate::IoError> for NonlinearMediationArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn invalid_request(message: &str) -> NonlinearMediationArtifactError {
    NonlinearMediationArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("nonlinear_mediation.invalid_request: {message}"),
    }
}

/// Declared graph premises on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Wire struct declares four independent graph premises; each flag is its own declaration.
#[allow(clippy::struct_excessive_bools)]
pub struct MediationPremisesWire {
    /// An unmeasured treatment-outcome common cause is declared.
    pub unmeasured_treatment_outcome_confounding: bool,
    /// An unmeasured treatment-mediator common cause is declared.
    pub unmeasured_treatment_mediator_confounding: bool,
    /// An unmeasured mediator-outcome common cause is declared.
    pub unmeasured_mediator_outcome_confounding: bool,
    /// Declared mediator-outcome confounders that are descendants of the treatment.
    pub treatment_induced_mediator_outcome_confounders: Vec<String>,
    /// The cross-world independence is declared.
    pub cross_world_independence: bool,
}

/// Estimator configuration on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationConfigWire {
    /// `natural_effects` or `interventional_effects` (the latter is refused).
    pub estimand: String,
    /// Outcome polynomial degree in the standardized mediator.
    pub outcome_degree: u32,
    /// Declared Gauss-Hermite node count `n`.
    pub quadrature_nodes: u32,
    /// Largest accepted integration error, as a fraction of the outcome standard deviation.
    pub integration_tolerance: f64,
    /// Fewest rows required in each treatment arm.
    pub min_arm_count: u32,
    /// Largest accepted cross-world mediator mass outside the treated arm's range.
    pub max_support_violation: f64,
    /// Bootstrap replicates for the standard errors.
    pub bootstrap_replicates: u32,
    /// Bootstrap seed.
    pub seed: u64,
}

/// The compact analysis dataset on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationDataWire {
    /// Binary treatment coded 0 / 1.
    pub treatment: Vec<f64>,
    /// Continuous mediator.
    pub mediator: Vec<f64>,
    /// Outcome.
    pub outcome: Vec<f64>,
    /// Covariate names, in column order.
    pub covariate_names: Vec<String>,
    /// Covariate columns.
    pub covariates: Vec<Vec<f64>>,
}

/// A complete mediation request: what the producer declares and the core estimator consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationRequestWire {
    /// Declared premises.
    pub premises: MediationPremisesWire,
    /// Configuration.
    pub config: MediationConfigWire,
    /// The analysis dataset.
    pub data: MediationDataWire,
}

/// One premise with its status: `declared` (asserted by the caller) or `checked` (verified here).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationPremiseRecord {
    /// Premise name.
    pub name: String,
    /// `declared` or `checked`.
    pub status: String,
    /// Whether the premise holds (every premise holds in a sealed artifact; a violated one
    /// refuses before sealing).
    pub holds: bool,
}

/// Model specifications.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationModelSpecs {
    /// Mediator model.
    pub mediator: String,
    /// Outcome model.
    pub outcome: String,
    /// Integration rule.
    pub integration: String,
    /// Outcome polynomial degree.
    pub outcome_degree: u32,
    /// Declared quadrature nodes `n`.
    pub quadrature_nodes: u32,
    /// Covariate names.
    pub covariates: Vec<String>,
}

/// Overlap diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationOverlapWire {
    /// Treated rows.
    pub treated_count: u64,
    /// Control rows.
    pub control_count: u64,
    /// Smallest mediator value in the treated arm.
    pub treated_mediator_min: f64,
    /// Largest mediator value in the treated arm.
    pub treated_mediator_max: f64,
    /// Average cross-world mediator mass outside the treated arm's observed range.
    pub mediator_support_violation: f64,
}

/// Mediator-model diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationMediatorWire {
    /// Intercept, treatment, then covariate coefficients.
    pub coefficients: Vec<f64>,
    /// Residual variance.
    pub residual_variance: f64,
    /// Residual skewness (reported, not gated).
    pub residual_skewness: f64,
    /// Residual excess kurtosis (reported, not gated).
    pub residual_excess_kurtosis: f64,
}

/// Bootstrap record: seed, replicate ids and the closed interval status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationBootstrapWire {
    /// Seed.
    pub seed: u64,
    /// Replicates requested.
    pub replicates_requested: u32,
    /// Replicates that fit and integrated.
    pub replicates_succeeded: u32,
    /// Ids of every requested replicate (`0 .. requested`).
    pub replicate_ids: Vec<u64>,
    /// Ids of replicates that failed.
    pub failed_replicate_ids: Vec<u64>,
    /// Replicate-id scheme.
    pub id_scheme: String,
    /// Interval status: `closed_calibration_unmeasured`.
    pub interval_status: String,
}

/// The stored estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediationResultWire {
    /// Natural direct effect.
    pub natural_direct: f64,
    /// Natural indirect effect.
    pub natural_indirect: f64,
    /// Total effect.
    pub total: f64,
    /// Bootstrap standard error of the natural direct effect.
    pub natural_direct_se: Option<f64>,
    /// Bootstrap standard error of the natural indirect effect.
    pub natural_indirect_se: Option<f64>,
    /// Bootstrap standard error of the total effect.
    pub total_se: Option<f64>,
    /// `|total_rowwise - (NDE + NIE)|`.
    pub identity_residual: f64,
    /// `n` versus `2n` node disagreement as a fraction of the outcome standard deviation.
    pub integration_error: f64,
    /// Declared node count `n`.
    pub coarse_nodes: u32,
    /// Node count `2n` the point estimates use.
    pub fine_nodes: u32,
    /// Outcome polynomial degree.
    pub outcome_degree: u32,
    /// Rows used.
    pub n_rows: u64,
    /// Overlap diagnostics.
    pub overlap: MediationOverlapWire,
    /// Mediator-model diagnostics.
    pub mediator: MediationMediatorWire,
    /// Bootstrap record.
    pub bootstrap: MediationBootstrapWire,
    /// Assumption identifiers the estimate relies on.
    pub assumptions: Vec<String>,
}

/// Identity digests. Each is BLAKE3 over a canonical encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationIdentity {
    /// Premises and their status records.
    pub premises_id: String,
    /// Configuration and model specifications.
    pub config_id: String,
    /// The embedded dataset.
    pub data_id: String,
    /// The stored estimate.
    pub result_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// The CBOR metadata section; also the self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearMediationMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// `natural_effects`.
    pub estimand: String,
    /// Declared premises.
    pub premises: MediationPremisesWire,
    /// Premises with their declared or checked status.
    pub premise_records: Vec<MediationPremiseRecord>,
    /// Model specifications.
    pub model_specs: MediationModelSpecs,
    /// Configuration.
    pub config: MediationConfigWire,
    /// Rows in the embedded dataset.
    pub n_rows: u64,
    /// The stored estimate.
    pub result: MediationResultWire,
    /// Identity digests.
    pub identity: NonlinearMediationIdentity,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `closed_calibration_unmeasured`.
    pub interval_status: String,
    /// Always `point_with_diagnostics`.
    pub inference_claim: String,
    /// Standing caveats.
    pub caveats: Vec<String>,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

fn count_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn count_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn index(n: u32) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}

fn estimand_of(tag: &str) -> Result<NonlinearMediationEstimand, NonlinearMediationArtifactError> {
    match tag {
        ESTIMAND_NATURAL => Ok(NonlinearMediationEstimand::NaturalEffects),
        ESTIMAND_INTERVENTIONAL => Ok(NonlinearMediationEstimand::InterventionalEffects),
        _ => Err(invalid_request("unknown estimand tag")),
    }
}

fn premises_of(wire: &MediationPremisesWire) -> NonlinearMediationPremises {
    NonlinearMediationPremises {
        unmeasured_treatment_outcome_confounding: wire.unmeasured_treatment_outcome_confounding,
        unmeasured_treatment_mediator_confounding: wire.unmeasured_treatment_mediator_confounding,
        unmeasured_mediator_outcome_confounding: wire.unmeasured_mediator_outcome_confounding,
        treatment_induced_mediator_outcome_confounders: wire
            .treatment_induced_mediator_outcome_confounders
            .clone(),
        cross_world_independence: wire.cross_world_independence,
    }
}

fn config_of(
    wire: &MediationConfigWire,
) -> Result<NonlinearMediationConfig, NonlinearMediationArtifactError> {
    Ok(NonlinearMediationConfig {
        estimand: estimand_of(&wire.estimand)?,
        outcome_degree: index(wire.outcome_degree),
        quadrature_nodes: index(wire.quadrature_nodes),
        integration_tolerance: wire.integration_tolerance,
        min_arm_count: index(wire.min_arm_count),
        max_support_violation: wire.max_support_violation,
        bootstrap_replicates: wire.bootstrap_replicates,
        seed: wire.seed,
    })
}

fn check_limits(
    request: &NonlinearMediationRequestWire,
) -> Result<(), NonlinearMediationArtifactError> {
    if request.data.treatment.len() > NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("rows"));
    }
    if request.data.covariates.len() > NONLINEAR_MEDIATION_ARTIFACT_MAX_COVARIATES {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("covariates"));
    }
    if request.config.bootstrap_replicates > NONLINEAR_MEDIATION_ARTIFACT_MAX_BOOTSTRAP {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("bootstrap replicates"));
    }
    Ok(())
}

fn check_shape(
    request: &NonlinearMediationRequestWire,
) -> Result<(), NonlinearMediationArtifactError> {
    let data = &request.data;
    let n = data.treatment.len();
    if data.mediator.len() != n
        || data.outcome.len() != n
        || data.covariates.iter().any(|c| c.len() != n)
    {
        return Err(invalid_request("columns differ in length"));
    }
    if data.covariate_names.len() != data.covariates.len() {
        return Err(invalid_request("every covariate needs exactly one name"));
    }
    Ok(())
}

fn estimate_of(
    request: &NonlinearMediationRequestWire,
) -> Result<NonlinearMediationEstimate, NonlinearMediationArtifactError> {
    check_limits(request)?;
    check_shape(request)?;
    let covariates: Vec<&[f64]> = request.data.covariates.iter().map(Vec::as_slice).collect();
    let input = NonlinearMediationInput {
        treatment: &request.data.treatment,
        mediator: &request.data.mediator,
        outcome: &request.data.outcome,
        covariates: &covariates,
    };
    Ok(estimate_nonlinear_mediation(
        &input,
        &premises_of(&request.premises),
        &config_of(&request.config)?,
    )?)
}

fn data_bytes(data: &MediationDataWire) -> Vec<u8> {
    let mut bytes = Vec::new();
    let columns =
        [&data.treatment, &data.mediator, &data.outcome].into_iter().chain(data.covariates.iter());
    for column in columns {
        for value in column {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn request_of(
    meta: &NonlinearMediationMeta,
    numbers: &[u8],
) -> Result<NonlinearMediationRequestWire, NonlinearMediationArtifactError> {
    let n = usize::try_from(meta.n_rows).unwrap_or(usize::MAX);
    let p = meta.model_specs.covariates.len();
    if n > NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("rows"));
    }
    if p > NONLINEAR_MEDIATION_ARTIFACT_MAX_COVARIATES {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("covariates"));
    }
    let expected = (3 + p).checked_mul(n).and_then(|c| c.checked_mul(8));
    if expected != Some(numbers.len()) {
        return Err(NonlinearMediationArtifactError::Malformed("data section length".into()));
    }
    let values: Vec<f64> = numbers
        .chunks_exact(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            f64::from_le_bytes(word)
        })
        .collect();
    let column = |j: usize| values[j * n..(j + 1) * n].to_vec();
    Ok(NonlinearMediationRequestWire {
        premises: meta.premises.clone(),
        config: meta.config.clone(),
        data: MediationDataWire {
            treatment: column(0),
            mediator: column(1),
            outcome: column(2),
            covariate_names: meta.model_specs.covariates.clone(),
            covariates: (0..p).map(|j| column(3 + j)).collect(),
        },
    })
}

fn premise_records(premises: &MediationPremisesWire) -> Vec<MediationPremiseRecord> {
    let record = |name: &str, status: &str, holds: bool| MediationPremiseRecord {
        name: name.to_owned(),
        status: status.to_owned(),
        holds,
    };
    vec![
        record(
            "no_unmeasured_treatment_outcome_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_treatment_outcome_confounding,
        ),
        record(
            "no_unmeasured_treatment_mediator_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_treatment_mediator_confounding,
        ),
        record(
            "no_unmeasured_mediator_outcome_confounding",
            STATUS_DECLARED,
            !premises.unmeasured_mediator_outcome_confounding,
        ),
        record(
            "no_treatment_induced_mediator_outcome_confounder",
            STATUS_DECLARED,
            premises.treatment_induced_mediator_outcome_confounders.is_empty(),
        ),
        record("cross_world_independence", STATUS_DECLARED, premises.cross_world_independence),
        record("linear_gaussian_mediator", STATUS_DECLARED, true),
        record("polynomial_outcome_in_mediator", STATUS_DECLARED, true),
        record("treatment_binary", STATUS_CHECKED, true),
        record("arm_overlap", STATUS_CHECKED, true),
        record("mediator_support_within_bound", STATUS_CHECKED, true),
        record("integration_error_within_tolerance", STATUS_CHECKED, true),
        record("total_equals_direct_plus_indirect", STATUS_CHECKED, true),
    ]
}

fn model_specs(request: &NonlinearMediationRequestWire) -> MediationModelSpecs {
    MediationModelSpecs {
        mediator: "linear_gaussian_in_treatment_and_covariates_homoscedastic_normal_residual"
            .to_owned(),
        outcome: "polynomial_in_standardized_mediator_with_treatment_interaction_linear_in_treatment_and_covariates"
            .to_owned(),
        integration: "gauss_hermite_n_versus_2n".to_owned(),
        outcome_degree: request.config.outcome_degree,
        quadrature_nodes: request.config.quadrature_nodes,
        covariates: request.data.covariate_names.clone(),
    }
}

fn result_wire(estimate: &NonlinearMediationEstimate) -> MediationResultWire {
    let boot = &estimate.bootstrap;
    MediationResultWire {
        natural_direct: estimate.natural_direct,
        natural_indirect: estimate.natural_indirect,
        total: estimate.total,
        natural_direct_se: estimate.natural_direct_se,
        natural_indirect_se: estimate.natural_indirect_se,
        total_se: estimate.total_se,
        identity_residual: estimate.identity_residual,
        integration_error: estimate.integration_error,
        coarse_nodes: count_u32(estimate.quadrature_nodes),
        fine_nodes: count_u32(2 * estimate.quadrature_nodes),
        outcome_degree: count_u32(estimate.outcome_degree),
        n_rows: count_u64(estimate.n_rows),
        overlap: MediationOverlapWire {
            treated_count: count_u64(estimate.overlap.treated_count),
            control_count: count_u64(estimate.overlap.control_count),
            treated_mediator_min: estimate.overlap.treated_mediator_range.0,
            treated_mediator_max: estimate.overlap.treated_mediator_range.1,
            mediator_support_violation: estimate.overlap.mediator_support_violation,
        },
        mediator: MediationMediatorWire {
            coefficients: estimate.mediator.coefficients.clone(),
            residual_variance: estimate.mediator.residual_variance,
            residual_skewness: estimate.mediator.residual_skewness,
            residual_excess_kurtosis: estimate.mediator.residual_excess_kurtosis,
        },
        bootstrap: MediationBootstrapWire {
            seed: boot.seed,
            replicates_requested: boot.replicates_requested,
            replicates_succeeded: boot.replicates_succeeded,
            replicate_ids: (0..u64::from(boot.replicates_requested)).collect(),
            failed_replicate_ids: boot.failed_replicate_ids.clone(),
            id_scheme: boot.id_scheme.to_owned(),
            interval_status: estimate.interval_status.to_owned(),
        },
        assumptions: estimate.assumptions.iter().map(|a| (*a).to_owned()).collect(),
    }
}

fn digest<T: Serialize>(tag: &str, value: &T) -> Result<String, NonlinearMediationArtifactError> {
    let bytes = to_cbor(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(&count_u64(bytes.len()).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn data_digest(data: &MediationDataWire) -> Result<String, NonlinearMediationArtifactError> {
    let names = to_cbor(&data.covariate_names)?;
    let numbers = data_bytes(data);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"nonlinear_mediation_v1.data");
    hasher.update(&count_u64(names.len()).to_le_bytes());
    hasher.update(&names);
    hasher.update(&count_u64(data.treatment.len()).to_le_bytes());
    hasher.update(&numbers);
    Ok(hasher.finalize().to_hex().to_string())
}

fn build_meta(
    request: &NonlinearMediationRequestWire,
    estimate: &NonlinearMediationEstimate,
) -> Result<NonlinearMediationMeta, NonlinearMediationArtifactError> {
    let records = premise_records(&request.premises);
    let specs = model_specs(request);
    let result = result_wire(estimate);
    let premises_id = digest("nonlinear_mediation_v1.premises", &(&request.premises, &records))?;
    let config_id = digest("nonlinear_mediation_v1.config", &(&request.config, &specs))?;
    let data_id = data_digest(&request.data)?;
    let result_id = digest("nonlinear_mediation_v1.result", &result)?;
    let whole = digest(
        "nonlinear_mediation_v1.identity",
        &(&premises_id, &config_id, &data_id, &result_id),
    )?;
    Ok(NonlinearMediationMeta {
        version: NONLINEAR_MEDIATION_ARTIFACT_VERSION,
        feature: NONLINEAR_MEDIATION_ARTIFACT_FEATURE.to_owned(),
        estimand: ESTIMAND_NATURAL.to_owned(),
        premises: request.premises.clone(),
        premise_records: records,
        model_specs: specs,
        config: request.config.clone(),
        n_rows: count_u64(request.data.treatment.len()),
        result,
        identity: NonlinearMediationIdentity {
            premises_id,
            config_id,
            data_id,
            result_id,
            digest: whole,
        },
        calibration: NONLINEAR_MEDIATION_CALIBRATION.to_owned(),
        interval_status: NONLINEAR_MEDIATION_INTERVAL_STATUS.to_owned(),
        inference_claim: NONLINEAR_MEDIATION_ARTIFACT_CLAIM.to_owned(),
        caveats: NONLINEAR_MEDIATION_ARTIFACT_CAVEATS.iter().map(|c| (*c).to_owned()).collect(),
    })
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &NonlinearMediationIdentity,
    other: &NonlinearMediationIdentity,
) -> Option<&'static str> {
    if stored.premises_id != other.premises_id {
        Some("premises")
    } else if stored.config_id != other.config_id {
        Some("config")
    } else if stored.data_id != other.data_id {
        Some("data")
    } else if stored.result_id != other.result_id {
        Some("result")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

/// Encode a metadata section and a data section as a checksummed container.
///
/// Hidden: the producer path is [`NonlinearMediationArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &NonlinearMediationMeta,
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
    let encode = |e: crate::IoError| NonlinearMediationArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(NonlinearMediationArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(encode)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(encode)?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                section_descriptor(DATA_SECTION, "application/octet-stream", data),
            ],
            provenance: ProvenanceWire {
                note: "nonlinear_mediation_point_with_diagnostics_unmeasured".into(),
            },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(DATA_SECTION, data.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("artifact bytes"));
    }
    Ok(bytes)
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(NonlinearMediationMeta, Vec<u8>), NonlinearMediationArtifactError> {
    if bytes.len() > MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != DATA_SECTION
    {
        return Err(NonlinearMediationArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > count_u64(MAX_NONLINEAR_MEDIATION_ARTIFACT_BYTES)) {
        return Err(NonlinearMediationArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != NONLINEAR_MEDIATION_ARTIFACT_VERSION {
        return Err(NonlinearMediationArtifactError::UnsupportedVersion { version: peek.version });
    }
    let meta: NonlinearMediationMeta = from_cbor(meta_section.as_bytes())?;
    let data = reader.load_section(DATA_SECTION)?;
    Ok((meta, data.as_bytes().to_vec()))
}

/// A produced or consumed mediation artifact: the declared request, the recomputed estimate and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediationArtifact {
    meta: NonlinearMediationMeta,
    request: NonlinearMediationRequestWire,
    estimate: NonlinearMediationEstimate,
}

impl NonlinearMediationArtifact {
    /// Run the core estimator on `request` and seal the answer with its identity.
    ///
    /// # Errors
    /// The core estimator's refusals (`effect_not_identified` for declared confounding,
    /// `cross_world_not_identified`, `route_not_supported` for the interventional estimand, ...)
    /// and the format's bounds.
    pub fn seal(
        request: &NonlinearMediationRequestWire,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        let estimate = estimate_of(request)?;
        let meta = build_meta(request, &estimate)?;
        Ok(Self { meta, request: request.clone(), estimate })
    }

    /// The metadata section, which is also the self-describing report.
    #[must_use]
    pub fn meta(&self) -> &NonlinearMediationMeta {
        &self.meta
    }

    /// The request the estimate was computed on.
    #[must_use]
    pub fn request(&self) -> &NonlinearMediationRequestWire {
        &self.request
    }

    /// The core estimate.
    #[must_use]
    pub fn estimate(&self) -> &NonlinearMediationEstimate {
        &self.estimate
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &NonlinearMediationIdentity {
        &self.meta.identity
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
        encode_parts(&self.meta, &data_bytes(&self.request.data), artifact_id)
    }

    /// Consume an artifact by re-estimation.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every
    /// identity field must match it, so a resealed change of premises, configuration, dataset
    /// or result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`NonlinearMediationArtifactError::IdentityMismatch`]), a stored record that does not
    /// replay, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&NonlinearMediationIdentity>,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        let (meta, data) = decode_parts(bytes)?;
        let unsupported = NonlinearMediationArtifactError::UnsupportedSemantics;
        if meta.feature != NONLINEAR_MEDIATION_ARTIFACT_FEATURE {
            return Err(unsupported("feature marker"));
        }
        if meta.estimand != ESTIMAND_NATURAL {
            return Err(unsupported("estimand"));
        }
        if meta.inference_claim != NONLINEAR_MEDIATION_ARTIFACT_CLAIM {
            return Err(unsupported("inference claim"));
        }
        if meta.calibration != NONLINEAR_MEDIATION_CALIBRATION {
            return Err(unsupported("calibration"));
        }
        if meta.interval_status != NONLINEAR_MEDIATION_INTERVAL_STATUS {
            return Err(unsupported("interval status"));
        }
        if meta.caveats != NONLINEAR_MEDIATION_ARTIFACT_CAVEATS {
            return Err(unsupported("caveats"));
        }
        let request = request_of(&meta, &data)?;
        let estimate = estimate_of(&request)?;
        let rebuilt = build_meta(&request, &estimate)?;
        if let Some(field) = identity_diff(&meta.identity, &rebuilt.identity) {
            return Err(if field == "result" {
                NonlinearMediationArtifactError::ResultMismatch("estimate")
            } else {
                NonlinearMediationArtifactError::IdentityMismatch { field }
            });
        }
        if to_cbor(&meta)? != to_cbor(&rebuilt)? {
            return Err(NonlinearMediationArtifactError::ResultMismatch("record"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &rebuilt.identity) {
                return Err(NonlinearMediationArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta: rebuilt, request, estimate })
    }
}
