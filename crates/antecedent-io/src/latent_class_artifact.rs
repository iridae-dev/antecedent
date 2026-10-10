//! B4 latent-class regime-effects artifact (`latent_class_effects_v1`).
//!
//! A bounded, checksummed, sectioned container holding one finite-mixture fit:
//!
//! * `latent_class.meta` (CBOR): version, feature marker, the declared class count `K`, the
//!   canonical class order and its mapping to the raw EM components, each class's weight,
//!   effect, intercept, covariate coefficients and variance (with bootstrap standard errors),
//!   the mixture-average effect, a responsibilities summary (per-class means, hard-assignment
//!   counts, separation and a digest of the full matrix), the head and digest of the
//!   log-likelihood trace, the premises with their declared or checked status, the
//!   configuration, the identity digests and the **unmeasured** calibration coordinate;
//! * `latent_class.data` (little-endian `f64`): the compact analysis dataset (outcome,
//!   treatment, then each covariate), capped at [`LATENT_CLASS_ARTIFACT_MAX_ROWS`] rows.
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! configuration and the embedded dataset, **refits** the mixture under the stored seed and
//! configuration and refuses unless the whole canonical record (every class number, the
//! responsibilities digest and the trace) is bit-identical to the recomputed one. It also
//! compares the recomputed identity digests with the stored ones and, when the caller retained
//! an identity, with that: a changed configuration, dataset or result is refused even when the
//! artifact was resealed consistently.
//!
//! What this does **not** say: class labels are an effect ordering, not a discovered meaning;
//! bootstrap standard errors carry no coverage claim (`calibration = "unmeasured"`); the
//! within-class randomization and Gaussian-linear class model are declared, not tested. Unknown
//! major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_estimate::EstimationError;
use antecedent_estimate::latent_class_effects::{
    LATENT_CLASS_CAVEAT, LATENT_CLASS_INFERENCE_CLAIM, LatentClassConfig, LatentClassData,
    LatentClassResult, PremiseStatus, fit_latent_class_effects,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const LATENT_CLASS_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const LATENT_CLASS_ARTIFACT_FEATURE: &str = "latent_class_effects_v1";
/// Calibration coordinate of the bootstrap standard errors: coverage was not measured.
pub const LATENT_CLASS_ARTIFACT_CALIBRATION: &str = "unmeasured";
/// Most analysis rows an artifact may embed, enforced on export and on consumption.
pub const LATENT_CLASS_ARTIFACT_MAX_ROWS: usize = 10_000;
/// Most covariates an artifact may embed.
pub const LATENT_CLASS_ARTIFACT_MAX_COVARIATES: usize = 32;
/// Most bootstrap replicates a producer may request and a consumer will replay.
pub const LATENT_CLASS_ARTIFACT_MAX_BOOTSTRAP: u32 = 1_000;
/// Log-likelihood trace entries kept verbatim in the record (the full trace is digested).
pub const LATENT_CLASS_ARTIFACT_TRACE_HEAD: usize = 16;
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_LATENT_CLASS_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

const ARTIFACT_KIND: &str = "latent_class_effects_v1";
const META_SECTION: &str = "latent_class_meta";
const DATA_SECTION: &str = "latent_class_data";

/// Why a latent-class artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LatentClassArtifactError {
    /// The bytes do not decode as this format.
    #[error("latent class artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported latent class artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim, calibration coordinate or caveat that is not this format's.
    #[error("unsupported latent class semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("latent class consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed latent class artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("latent_class.wrong_contract: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored record differs from the refitted one.
    #[error("stored latent class record does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("latent class artifact does not encode: {0}")]
    Encode(String),
    /// The core fit refused (registered reason code; the message begins with its detail).
    #[error("{message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// `detail: explanation`.
        message: String,
    },
}

impl LatentClassArtifactError {
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
                "latent_class.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            Self::ResultMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "latent_class.report_replay_mismatch".to_owned(),
                format!("the stored {what} does not replay"),
            )),
            _ => None,
        }
    }
}

impl From<EstimationError> for LatentClassArtifactError {
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

impl From<crate::IoError> for LatentClassArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn invalid_request(message: &str) -> LatentClassArtifactError {
    LatentClassArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("latent_class.invalid_request: {message}"),
    }
}

/// Fit configuration on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassConfigWire {
    /// Declared number of classes `K`.
    pub classes: u32,
    /// Seed of every deterministic stream (initialization, bootstrap).
    pub seed: u64,
    /// EM restarts.
    pub restarts: u32,
    /// EM iteration bound per start.
    pub max_iterations: u32,
    /// Relative log-likelihood convergence tolerance.
    pub tolerance: f64,
    /// Smallest admissible class weight.
    pub min_class_weight: f64,
    /// Smallest admissible separation.
    pub min_separation: f64,
    /// Residual-variance floor as a fraction of the outcome variance.
    pub variance_floor: f64,
    /// Bootstrap replicates (0 reports no standard errors).
    pub bootstrap_replicates: u32,
    /// One hard initial labelling, used as the only start.
    pub initial_labels: Option<Vec<u32>>,
    /// The caller declares treatment as-if randomized within class given the covariates.
    pub assume_conditional_randomization: bool,
}

/// The compact analysis dataset on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassDataWire {
    /// Outcome.
    pub outcome: Vec<f64>,
    /// Treatment (binary or numeric).
    pub treatment: Vec<f64>,
    /// Covariate names, in column order.
    pub covariate_names: Vec<String>,
    /// Covariate columns.
    pub covariates: Vec<Vec<f64>>,
}

/// A complete latent-class request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassRequestWire {
    /// Configuration.
    pub config: LatentClassConfigWire,
    /// The analysis dataset.
    pub data: LatentClassDataWire,
}

/// One class in canonical order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassWire {
    /// Canonical index (ascending effect).
    pub index: u32,
    /// Raw EM component index of the selected fit.
    pub raw_index: u32,
    /// Class weight.
    pub weight: f64,
    /// Bootstrap standard error of the weight.
    pub weight_se: Option<f64>,
    /// Class intercept.
    pub intercept: f64,
    /// Class effect.
    pub effect: f64,
    /// Bootstrap standard error of the effect.
    pub effect_se: Option<f64>,
    /// Covariate coefficients.
    pub covariate_coefficients: Vec<f64>,
    /// Residual variance.
    pub residual_variance: f64,
    /// Effective class size.
    pub effective_n: f64,
}

/// Responsibilities summary: per-class means, hard-assignment counts, separation and a digest
/// of the full matrix and assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponsibilitiesSummaryWire {
    /// Rows.
    pub rows: u64,
    /// Mean posterior responsibility of each canonical class.
    pub class_means: Vec<f64>,
    /// Units whose most probable class is each canonical class.
    pub hard_counts: Vec<u64>,
    /// Mean largest posterior responsibility.
    pub separation: f64,
    /// BLAKE3 over every responsibility's bits and the hard assignment.
    pub digest: String,
}

/// Log-likelihood record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassLikelihoodWire {
    /// Observed-data log-likelihood of the selected fit.
    pub log_likelihood: f64,
    /// EM iterations of the selected fit.
    pub iterations: u64,
    /// First entries of the (non-decreasing) trace.
    pub trace_head: Vec<f64>,
    /// Trace length.
    pub trace_len: u64,
    /// BLAKE3 over the full trace's bits.
    pub trace_digest: String,
    /// Bayesian information criterion.
    pub bic: f64,
    /// Restarts attempted.
    pub restarts_attempted: u32,
    /// Restarts that converged to a non-degenerate fit.
    pub restarts_converged: u32,
}

/// Bootstrap bookkeeping.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassBootstrapWire {
    /// Replicates requested.
    pub requested: u32,
    /// Replicates that converged.
    pub succeeded: u32,
    /// Replicates skipped.
    pub failed: u32,
}

/// One premise with its status: `declared` or `checked`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassPremiseRecord {
    /// Premise name.
    pub name: String,
    /// `declared` or `checked`.
    pub status: String,
}

/// Identity digests. Each is BLAKE3 over a canonical encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassIdentity {
    /// Premise records.
    pub premises_id: String,
    /// Configuration.
    pub config_id: String,
    /// The embedded dataset.
    pub data_id: String,
    /// The canonical stored result.
    pub result_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// The CBOR metadata section; also the report handed to host languages (without the full
/// responsibilities matrix, which [`LatentClassReportWire`] adds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Declared class count.
    pub classes_declared: u32,
    /// Canonical class `c` came from raw EM component `class_order[c]`.
    pub class_order: Vec<u32>,
    /// Classes in canonical order.
    pub classes: Vec<LatentClassWire>,
    /// Mixture-average effect.
    pub mixture_average_effect: f64,
    /// Bootstrap standard error of the mixture-average effect.
    pub mixture_average_se: Option<f64>,
    /// Smallest gap between adjacent canonical effects.
    pub min_effect_gap: f64,
    /// Responsibilities summary.
    pub responsibilities: ResponsibilitiesSummaryWire,
    /// Log-likelihood record.
    pub likelihood: LatentClassLikelihoodWire,
    /// Bootstrap bookkeeping, when requested.
    pub bootstrap: Option<LatentClassBootstrapWire>,
    /// Premises with their status.
    pub premises: Vec<LatentClassPremiseRecord>,
    /// Configuration.
    pub config: LatentClassConfigWire,
    /// Rows in the embedded dataset.
    pub n_rows: u64,
    /// Covariate names.
    pub covariate_names: Vec<String>,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `point_with_bootstrap_se`.
    pub inference_claim: String,
    /// Standing caveat.
    pub caveat: String,
    /// Identity digests.
    pub identity: LatentClassIdentity,
}

/// The full report: the metadata plus the per-unit posterior responsibilities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentClassReportWire {
    /// The metadata record.
    pub meta: LatentClassMeta,
    /// Posterior responsibilities, `n` rows by `K` canonical columns.
    pub responsibilities: Vec<Vec<f64>>,
    /// Most probable canonical class of each unit.
    pub hard_assignment: Vec<u32>,
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

fn config_of(wire: &LatentClassConfigWire) -> LatentClassConfig {
    LatentClassConfig {
        classes: index(wire.classes),
        seed: wire.seed,
        restarts: index(wire.restarts),
        max_iterations: index(wire.max_iterations),
        tolerance: wire.tolerance,
        min_class_weight: wire.min_class_weight,
        min_separation: wire.min_separation,
        variance_floor: wire.variance_floor,
        bootstrap_replicates: index(wire.bootstrap_replicates),
        initial_labels: wire
            .initial_labels
            .as_ref()
            .map(|labels| labels.iter().map(|&l| index(l)).collect()),
        assume_conditional_randomization: wire.assume_conditional_randomization,
    }
}

fn check_limits(request: &LatentClassRequestWire) -> Result<(), LatentClassArtifactError> {
    if request.data.outcome.len() > LATENT_CLASS_ARTIFACT_MAX_ROWS {
        return Err(LatentClassArtifactError::LimitsExceeded("rows"));
    }
    if request.data.covariates.len() > LATENT_CLASS_ARTIFACT_MAX_COVARIATES {
        return Err(LatentClassArtifactError::LimitsExceeded("covariates"));
    }
    if request.config.bootstrap_replicates > LATENT_CLASS_ARTIFACT_MAX_BOOTSTRAP {
        return Err(LatentClassArtifactError::LimitsExceeded("bootstrap replicates"));
    }
    Ok(())
}

fn check_shape(request: &LatentClassRequestWire) -> Result<(), LatentClassArtifactError> {
    let data = &request.data;
    let n = data.outcome.len();
    if data.treatment.len() != n || data.covariates.iter().any(|c| c.len() != n) {
        return Err(invalid_request("columns differ in length"));
    }
    if data.covariate_names.len() != data.covariates.len() {
        return Err(invalid_request("every covariate needs exactly one name"));
    }
    Ok(())
}

fn fit_of(request: &LatentClassRequestWire) -> Result<LatentClassResult, LatentClassArtifactError> {
    check_limits(request)?;
    check_shape(request)?;
    let data = LatentClassData {
        outcome: &request.data.outcome,
        treatment: &request.data.treatment,
        covariates: &request.data.covariates,
    };
    Ok(fit_latent_class_effects(&data, &config_of(&request.config))?)
}

fn data_bytes(data: &LatentClassDataWire) -> Vec<u8> {
    let mut bytes = Vec::new();
    let columns = [&data.outcome, &data.treatment].into_iter().chain(data.covariates.iter());
    for column in columns {
        for value in column {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn request_of(
    meta: &LatentClassMeta,
    numbers: &[u8],
) -> Result<LatentClassRequestWire, LatentClassArtifactError> {
    let n = usize::try_from(meta.n_rows).unwrap_or(usize::MAX);
    let p = meta.covariate_names.len();
    if n > LATENT_CLASS_ARTIFACT_MAX_ROWS {
        return Err(LatentClassArtifactError::LimitsExceeded("rows"));
    }
    if p > LATENT_CLASS_ARTIFACT_MAX_COVARIATES {
        return Err(LatentClassArtifactError::LimitsExceeded("covariates"));
    }
    let expected = (2 + p).checked_mul(n).and_then(|c| c.checked_mul(8));
    if expected != Some(numbers.len()) {
        return Err(LatentClassArtifactError::Malformed("data section length".into()));
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
    Ok(LatentClassRequestWire {
        config: meta.config.clone(),
        data: LatentClassDataWire {
            outcome: column(0),
            treatment: column(1),
            covariate_names: meta.covariate_names.clone(),
            covariates: (0..p).map(|j| column(2 + j)).collect(),
        },
    })
}

fn hard_u32(result: &LatentClassResult) -> Vec<u32> {
    result.hard_assignment.iter().map(|&c| count_u32(c)).collect()
}

fn responsibilities_summary(result: &LatentClassResult) -> ResponsibilitiesSummaryWire {
    let k = result.classes.len();
    let rows = result.responsibilities.len();
    let rows_f = f64::from(count_u32(rows).max(1));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"latent_class_effects_v1.responsibilities");
    let mut sums = vec![0.0_f64; k];
    for row in &result.responsibilities {
        for (c, value) in row.iter().enumerate() {
            sums[c] += value;
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    let mut counts = vec![0_u64; k];
    for &assigned in &result.hard_assignment {
        counts[assigned] += 1;
        hasher.update(&count_u64(assigned).to_le_bytes());
    }
    ResponsibilitiesSummaryWire {
        rows: count_u64(rows),
        class_means: sums.iter().map(|s| s / rows_f).collect(),
        hard_counts: counts,
        separation: result.separation,
        digest: hasher.finalize().to_hex().to_string(),
    }
}

fn likelihood_record(result: &LatentClassResult) -> LatentClassLikelihoodWire {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"latent_class_effects_v1.trace");
    for value in &result.log_likelihood_trace {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    LatentClassLikelihoodWire {
        log_likelihood: result.log_likelihood,
        iterations: count_u64(result.iterations),
        trace_head: result
            .log_likelihood_trace
            .iter()
            .take(LATENT_CLASS_ARTIFACT_TRACE_HEAD)
            .copied()
            .collect(),
        trace_len: count_u64(result.log_likelihood_trace.len()),
        trace_digest: hasher.finalize().to_hex().to_string(),
        bic: result.bic,
        restarts_attempted: count_u32(result.restarts_attempted),
        restarts_converged: count_u32(result.restarts_converged),
    }
}

fn premise_records(result: &LatentClassResult) -> Vec<LatentClassPremiseRecord> {
    result
        .premises
        .iter()
        .map(|p| LatentClassPremiseRecord {
            name: p.name.to_owned(),
            status: match p.status {
                PremiseStatus::Checked => "checked",
                PremiseStatus::Declared => "declared",
            }
            .to_owned(),
        })
        .collect()
}

fn digest<T: Serialize>(tag: &str, value: &T) -> Result<String, LatentClassArtifactError> {
    let bytes = to_cbor(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(&count_u64(bytes.len()).to_le_bytes());
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn data_digest(data: &LatentClassDataWire) -> Result<String, LatentClassArtifactError> {
    let names = to_cbor(&data.covariate_names)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"latent_class_effects_v1.data");
    hasher.update(&count_u64(names.len()).to_le_bytes());
    hasher.update(&names);
    hasher.update(&count_u64(data.outcome.len()).to_le_bytes());
    hasher.update(&data_bytes(data));
    Ok(hasher.finalize().to_hex().to_string())
}

/// Everything in the record that the fit determines, in one comparable value.
#[derive(Serialize)]
struct CanonicalResult<'a> {
    class_order: &'a [u32],
    classes: &'a [LatentClassWire],
    mixture_average_effect: f64,
    mixture_average_se: Option<f64>,
    min_effect_gap: f64,
    responsibilities: &'a ResponsibilitiesSummaryWire,
    likelihood: &'a LatentClassLikelihoodWire,
    bootstrap: &'a Option<LatentClassBootstrapWire>,
}

fn build_meta(
    request: &LatentClassRequestWire,
    result: &LatentClassResult,
) -> Result<LatentClassMeta, LatentClassArtifactError> {
    let classes: Vec<LatentClassWire> = result
        .classes
        .iter()
        .map(|c| LatentClassWire {
            index: count_u32(c.index),
            raw_index: count_u32(c.raw_index),
            weight: c.weight,
            weight_se: c.weight_se,
            intercept: c.intercept,
            effect: c.effect,
            effect_se: c.effect_se,
            covariate_coefficients: c.covariate_coefficients.clone(),
            residual_variance: c.residual_variance,
            effective_n: c.effective_n,
        })
        .collect();
    let class_order: Vec<u32> = result.class_order.iter().map(|&c| count_u32(c)).collect();
    let responsibilities = responsibilities_summary(result);
    let likelihood = likelihood_record(result);
    let bootstrap = result.bootstrap.map(|b| LatentClassBootstrapWire {
        requested: count_u32(b.requested),
        succeeded: count_u32(b.succeeded),
        failed: count_u32(b.failed),
    });
    let premises = premise_records(result);
    let canonical = CanonicalResult {
        class_order: &class_order,
        classes: &classes,
        mixture_average_effect: result.mixture_average_effect,
        mixture_average_se: result.mixture_average_se,
        min_effect_gap: result.min_effect_gap,
        responsibilities: &responsibilities,
        likelihood: &likelihood,
        bootstrap: &bootstrap,
    };
    let premises_id = digest("latent_class_effects_v1.premises", &premises)?;
    let config_id = digest("latent_class_effects_v1.config", &request.config)?;
    let data_id = data_digest(&request.data)?;
    let result_id = digest("latent_class_effects_v1.result", &canonical)?;
    let whole = digest(
        "latent_class_effects_v1.identity",
        &(&premises_id, &config_id, &data_id, &result_id),
    )?;
    Ok(LatentClassMeta {
        version: LATENT_CLASS_ARTIFACT_VERSION,
        feature: LATENT_CLASS_ARTIFACT_FEATURE.to_owned(),
        classes_declared: request.config.classes,
        class_order,
        classes,
        mixture_average_effect: result.mixture_average_effect,
        mixture_average_se: result.mixture_average_se,
        min_effect_gap: result.min_effect_gap,
        responsibilities,
        likelihood,
        bootstrap,
        premises,
        config: request.config.clone(),
        n_rows: count_u64(request.data.outcome.len()),
        covariate_names: request.data.covariate_names.clone(),
        calibration: LATENT_CLASS_ARTIFACT_CALIBRATION.to_owned(),
        inference_claim: LATENT_CLASS_INFERENCE_CLAIM.to_owned(),
        caveat: LATENT_CLASS_CAVEAT.to_owned(),
        identity: LatentClassIdentity { premises_id, config_id, data_id, result_id, digest: whole },
    })
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &LatentClassIdentity,
    other: &LatentClassIdentity,
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
/// Hidden: the producer path is [`LatentClassArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &LatentClassMeta,
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, LatentClassArtifactError> {
    let encode = |e: crate::IoError| LatentClassArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(LatentClassArtifactError::Encode("missing artifact id".into()));
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
                note: "latent_class_point_with_bootstrap_se_unmeasured".into(),
            },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(DATA_SECTION, data.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_LATENT_CLASS_ARTIFACT_BYTES {
        return Err(LatentClassArtifactError::LimitsExceeded("artifact bytes"));
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
pub fn decode_parts(bytes: &[u8]) -> Result<(LatentClassMeta, Vec<u8>), LatentClassArtifactError> {
    if bytes.len() > MAX_LATENT_CLASS_ARTIFACT_BYTES {
        return Err(LatentClassArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != DATA_SECTION
    {
        return Err(LatentClassArtifactError::Malformed("unsupported container layout".into()));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > count_u64(MAX_LATENT_CLASS_ARTIFACT_BYTES)) {
        return Err(LatentClassArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != LATENT_CLASS_ARTIFACT_VERSION {
        return Err(LatentClassArtifactError::UnsupportedVersion { version: peek.version });
    }
    let meta: LatentClassMeta = from_cbor(meta_section.as_bytes())?;
    let data = reader.load_section(DATA_SECTION)?;
    Ok((meta, data.as_bytes().to_vec()))
}

/// A produced or consumed latent-class artifact: the declared request, the refitted result and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassArtifact {
    meta: LatentClassMeta,
    request: LatentClassRequestWire,
    result: LatentClassResult,
}

impl LatentClassArtifact {
    /// Fit the mixture on `request` and seal the answer with its identity.
    ///
    /// # Errors
    /// The core fit's refusals (`latent_class.randomization_not_declared`,
    /// `latent_class.weak_class`, `latent_class.degenerate_class`, ...) and the format's bounds.
    pub fn seal(request: &LatentClassRequestWire) -> Result<Self, LatentClassArtifactError> {
        let result = fit_of(request)?;
        let meta = build_meta(request, &result)?;
        Ok(Self { meta, request: request.clone(), result })
    }

    /// The metadata section, which is also the summary report.
    #[must_use]
    pub fn meta(&self) -> &LatentClassMeta {
        &self.meta
    }

    /// The request the fit was computed on.
    #[must_use]
    pub fn request(&self) -> &LatentClassRequestWire {
        &self.request
    }

    /// The core result, with the full responsibilities matrix.
    #[must_use]
    pub fn result(&self) -> &LatentClassResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &LatentClassIdentity {
        &self.meta.identity
    }

    /// The report for host languages: the record plus the full responsibilities.
    #[must_use]
    pub fn report(&self) -> LatentClassReportWire {
        LatentClassReportWire {
            meta: self.meta.clone(),
            responsibilities: self.result.responsibilities.clone(),
            hard_assignment: hard_u32(&self.result),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, LatentClassArtifactError> {
        encode_parts(&self.meta, &data_bytes(&self.request.data), artifact_id)
    }

    /// Consume an artifact by refitting under the stored seed and configuration.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every
    /// identity field must match it, so a resealed change of premises, configuration, dataset
    /// or result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`LatentClassArtifactError::IdentityMismatch`]), a canonical record that does not
    /// replay bit for bit, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&LatentClassIdentity>,
    ) -> Result<Self, LatentClassArtifactError> {
        let (meta, data) = decode_parts(bytes)?;
        let unsupported = LatentClassArtifactError::UnsupportedSemantics;
        if meta.feature != LATENT_CLASS_ARTIFACT_FEATURE {
            return Err(unsupported("feature marker"));
        }
        if meta.inference_claim != LATENT_CLASS_INFERENCE_CLAIM {
            return Err(unsupported("inference claim"));
        }
        if meta.calibration != LATENT_CLASS_ARTIFACT_CALIBRATION {
            return Err(unsupported("calibration"));
        }
        if meta.caveat != LATENT_CLASS_CAVEAT {
            return Err(unsupported("caveat"));
        }
        let request = request_of(&meta, &data)?;
        let result = fit_of(&request)?;
        let rebuilt = build_meta(&request, &result)?;
        if let Some(field) = identity_diff(&meta.identity, &rebuilt.identity) {
            return Err(if field == "result" {
                LatentClassArtifactError::ResultMismatch("canonical result")
            } else {
                LatentClassArtifactError::IdentityMismatch { field }
            });
        }
        if to_cbor(&meta)? != to_cbor(&rebuilt)? {
            return Err(LatentClassArtifactError::ResultMismatch("record"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &rebuilt.identity) {
                return Err(LatentClassArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta: rebuilt, request, result })
    }
}
