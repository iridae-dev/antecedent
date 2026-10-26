//! Latent-class regime-effects artifact (`latent_class_effects_v1`).
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
//!   treatment, then each covariate), or nothing when the record binds its dataset by digest.
//!
//! A consumer trusts none of the stored results. It rebuilds the request from the stored
//! configuration and the embedded dataset, **refits** the mixture under the stored seed and
//! configuration and refuses unless the whole canonical record (every class number, the
//! responsibilities digest and the trace) is bit-identical to the recomputed one. It also
//! compares the recomputed identity digests with the stored ones and, when the caller retained
//! an identity, with that: a changed configuration, dataset or result is refused even when the
//! artifact was resealed consistently.
//!
//! # Budgets
//!
//! The fit has no row ceiling; three budgets bound what is spent on it, each refusing with a
//! typed detail instead of truncating anything:
//!
//! * **Computation:** the [`ExecutionContext`] given to [`LatentClassArtifact::seal_with`] or
//!   [`LatentClassArtifact::from_bytes_with`] bounds the planned working set (its hard memory
//!   limit) and stops a cancelled run (`latent_class.memory_budget_exceeded`).
//! * **Export:** [`LatentClassArtifact::to_bytes_within`] embeds the dataset only when the whole
//!   artifact fits the export budget, which is at most [`MAX_LATENT_CLASS_ARTIFACT_BYTES`]
//!   (`latent_class.export_budget_exceeded`); the fitted result stays available in memory.
//! * **Replay:** a refit needs the rows. [`LatentClassArtifact::to_bytes_digest_bound`] exports
//!   the record without them, bound to the dataset by its digest (`identity.data_id`);
//!   [`LatentClassArtifact::from_bytes_with`] then needs the caller to supply the dataset and
//!   refuses it unless the digest matches.
//!
//! What this does **not** say: class labels are an effect ordering, not a discovered meaning;
//! bootstrap standard errors carry no coverage claim (`calibration = "unmeasured"`); the
//! within-class randomization and Gaussian-linear class model are declared, not tested. Unknown
//! major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::latent_class_effects::{
    LATENT_CLASS_CAVEAT, LATENT_CLASS_INFERENCE_CLAIM, LatentClassConfig, LatentClassData,
    LatentClassResult, PremiseStatus, fit_latent_class_effects,
};
use serde::{Deserialize, Serialize};

use crate::artifact_budget::{
    BudgetDetails, BudgetRefusal, admit_computation, admit_export, planned_bytes, unbounded_context,
};
use crate::convert::to_cbor;
use crate::width::lossless_u64;

/// The artifact major version this reader writes and accepts.
pub const LATENT_CLASS_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const LATENT_CLASS_ARTIFACT_FEATURE: &str = "latent_class_effects_v1";
/// Calibration coordinate of the bootstrap standard errors: coverage was not measured.
pub const LATENT_CLASS_ARTIFACT_CALIBRATION: &str = "unmeasured";
/// Most analysis rows an artifact embedded under the earlier fixed ceiling.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: the fit has no row ceiling; the export and computation budgets bound the artifact"
)]
pub const LATENT_CLASS_ARTIFACT_MAX_ROWS: usize = 10_000;
/// Most bootstrap replicates a producer may request and a consumer will replay.
pub const LATENT_CLASS_ARTIFACT_MAX_BOOTSTRAP: u32 = 1_000;
/// Log-likelihood trace entries kept verbatim in the record (the full trace is digested).
pub const LATENT_CLASS_ARTIFACT_TRACE_HEAD: usize = 16;
/// Most bytes an artifact may occupy: the largest export budget, enforced on consumption.
pub const MAX_LATENT_CLASS_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::LATENT_CLASS_EFFECTS;
const META_SECTION: &str = "latent_class_meta";
const DATA_SECTION: &str = "latent_class_data";
const BUDGET: BudgetDetails = BudgetDetails {
    cancelled: (antecedent_core::reason_code!("cancelled_no_claim"), "latent_class.cancelled"),
    memory: "latent_class.memory_budget_exceeded",
    export: "latent_class.export_budget_exceeded",
};
/// `LatentClassMeta::data_binding` of a record that holds no dataset, only its digest.
const DATA_BOUND_BY_DIGEST: &str = "digest";

crate::sectioned_artifact_error!(
    /// Why a latent-class artifact was refused or could not be produced.
    LatentClassArtifactError,
    "latent class",
    "latent_class",
    "record",
    true
);

fn budget_refusal((code, detail, explanation): BudgetRefusal) -> LatentClassArtifactError {
    LatentClassArtifactError::refused(code, detail, explanation)
}

fn invalid_request(message: &str) -> LatentClassArtifactError {
    LatentClassArtifactError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        "latent_class.invalid_request",
        message,
    )
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
    /// Rows in the dataset.
    pub n_rows: u64,
    /// Covariate names.
    pub covariate_names: Vec<String>,
    /// `digest` when the record holds no dataset and the caller supplies it on consumption;
    /// absent when the dataset is embedded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_binding: Option<String>,
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

fn count_u32(n: usize) -> Result<u32, LatentClassArtifactError> {
    u32::try_from(n).map_err(|_| LatentClassArtifactError::LimitsExceeded("count"))
}

fn index(n: u32) -> usize {
    n as usize
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
    // The covariate count is bounded by the bytes of the data section (a column is `n` reals),
    // by the byte budget of the artifact and by the replay's memory budget, not by a count.
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

/// Eight-byte words the fit holds per row: the design and its bootstrap resample, three
/// responsibility matrices and the per-row bookkeeping. A conservative planning estimate.
fn words_per_row(covariates: usize, classes: u32) -> Option<usize> {
    let design = covariates.checked_add(3)?.checked_mul(2)?;
    design.checked_add(index(classes).checked_mul(3)?)?.checked_add(4)
}

fn fit_of(
    request: &LatentClassRequestWire,
    ctx: &ExecutionContext,
) -> Result<LatentClassResult, LatentClassArtifactError> {
    check_limits(request)?;
    check_shape(request)?;
    let words = words_per_row(request.data.covariates.len(), request.config.classes);
    let planned = words.and_then(|words| planned_bytes(request.data.outcome.len(), words, 0));
    admit_computation(BUDGET, planned, ctx).map_err(budget_refusal)?;
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
    let columns = crate::width::columns_from_le(numbers, n, 2 + p)
        .ok_or_else(|| LatentClassArtifactError::Malformed("data section length".into()))?;
    Ok(LatentClassRequestWire {
        config: meta.config.clone(),
        data: data_from_columns(columns, meta.covariate_names.clone()),
    })
}

fn data_from_columns(
    mut columns: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
) -> LatentClassDataWire {
    let covariates = columns.split_off(2);
    let treatment = columns.pop().unwrap_or_default();
    let outcome = columns.pop().unwrap_or_default();
    LatentClassDataWire { outcome, treatment, covariate_names, covariates }
}

fn hard_u32(result: &LatentClassResult) -> Result<Vec<u32>, LatentClassArtifactError> {
    result.hard_assignment.iter().copied().map(count_u32).collect()
}

fn responsibilities_summary(
    result: &LatentClassResult,
) -> Result<ResponsibilitiesSummaryWire, LatentClassArtifactError> {
    let k = result.classes.len();
    let rows = result.responsibilities.len();
    let rows_f = f64::from(count_u32(rows)?.max(1));
    let mut hasher = CanonicalWriter::new("latent_class_effects_v1.responsibilities");
    let mut sums = vec![0.0_f64; k];
    for row in &result.responsibilities {
        for (c, value) in row.iter().enumerate() {
            sums[c] += value;
            hasher.real(*value);
        }
    }
    let mut counts = vec![0_u64; k];
    for &assigned in &result.hard_assignment {
        counts[assigned] += 1;
        hasher.word(lossless_u64(assigned));
    }
    Ok(ResponsibilitiesSummaryWire {
        rows: lossless_u64(rows),
        class_means: sums.iter().map(|s| s / rows_f).collect(),
        hard_counts: counts,
        separation: result.separation,
        digest: hasher.finish_hex(),
    })
}

fn likelihood_record(
    result: &LatentClassResult,
) -> Result<LatentClassLikelihoodWire, LatentClassArtifactError> {
    let mut hasher = CanonicalWriter::new("latent_class_effects_v1.trace");
    for value in &result.log_likelihood_trace {
        hasher.real(*value);
    }
    Ok(LatentClassLikelihoodWire {
        log_likelihood: result.log_likelihood,
        iterations: lossless_u64(result.iterations),
        trace_head: result
            .log_likelihood_trace
            .iter()
            .take(LATENT_CLASS_ARTIFACT_TRACE_HEAD)
            .copied()
            .collect(),
        trace_len: lossless_u64(result.log_likelihood_trace.len()),
        trace_digest: hasher.finish_hex(),
        bic: result.bic,
        restarts_attempted: count_u32(result.restarts_attempted)?,
        restarts_converged: count_u32(result.restarts_converged)?,
    })
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
    Ok(CanonicalWriter::digest_bytes_hex(tag, &bytes))
}

fn data_digest(data: &LatentClassDataWire) -> Result<String, LatentClassArtifactError> {
    let names = to_cbor(&data.covariate_names)?;
    let hasher = CanonicalWriter::new("latent_class_effects_v1.data")
        .bytes(&names)
        .word(lossless_u64(data.outcome.len()))
        .bytes(&data_bytes(data))
        .finish();
    Ok(antecedent_core::to_hex(&hasher))
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
        .map(|c| {
            Ok(LatentClassWire {
                index: count_u32(c.index)?,
                raw_index: count_u32(c.raw_index)?,
                weight: c.weight,
                weight_se: c.weight_se,
                intercept: c.intercept,
                effect: c.effect,
                effect_se: c.effect_se,
                covariate_coefficients: c.covariate_coefficients.clone(),
                residual_variance: c.residual_variance,
                effective_n: c.effective_n,
            })
        })
        .collect::<Result<_, LatentClassArtifactError>>()?;
    let class_order: Vec<u32> =
        result.class_order.iter().copied().map(count_u32).collect::<Result<_, _>>()?;
    let responsibilities = responsibilities_summary(result)?;
    let likelihood = likelihood_record(result)?;
    let bootstrap = match result.bootstrap {
        Some(b) => Some(LatentClassBootstrapWire {
            requested: count_u32(b.requested)?,
            succeeded: count_u32(b.succeeded)?,
            failed: count_u32(b.failed)?,
        }),
        None => None,
    };
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
        n_rows: lossless_u64(request.data.outcome.len()),
        covariate_names: request.data.covariate_names.clone(),
        data_binding: None,
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
    crate::first_identity_difference!(stored, other, {
        premises_id => "premises",
        config_id => "config",
        data_id => "data",
        result_id => "result",
        digest => "identity",
    })
}

/// Encode a metadata section and a data section as a checksummed container.
///
/// Hidden: the producer path is [`LatentClassArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &LatentClassMeta,
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, LatentClassArtifactError> {
    encode_parts_impl(meta, data, artifact_id, MAX_LATENT_CLASS_ARTIFACT_BYTES)
}

fn encode_parts_impl(
    meta: &LatentClassMeta,
    data: &[u8],
    artifact_id: &str,
    export_budget: usize,
) -> Result<Vec<u8>, LatentClassArtifactError> {
    // A consumer reads at most the byte cap of uncompressed sections, whatever the container
    // compresses to; the caller's budget bounds the container itself.
    let bytes = crate::artifact_codec::encode_sectioned_admitting(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "latent_class_point_with_bootstrap_se_unmeasured",
            max_bytes: MAX_LATENT_CLASS_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[crate::OutSection {
            id: DATA_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: data,
        }],
        |uncompressed| {
            admit_export(BUDGET, uncompressed, MAX_LATENT_CLASS_ARTIFACT_BYTES).map_err(budget_refusal)
        },
    )?;
    admit_export(BUDGET, bytes.len(), export_budget.min(MAX_LATENT_CLASS_ARTIFACT_BYTES))
        .map_err(budget_refusal)?;
    Ok(bytes)
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<(LatentClassMeta, Vec<u8>), LatentClassArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(LatentClassMeta, Vec<u8>), LatentClassArtifactError> {
    let (meta, parts) = crate::artifact_codec::decode_sectioned::<LatentClassArtifactError, LatentClassMeta>(
        ARTIFACT_KIND,
        MAX_LATENT_CLASS_ARTIFACT_BYTES,
        META_SECTION,
        &[DATA_SECTION],
        LATENT_CLASS_ARTIFACT_VERSION,
        bytes,
    )?;
    let Ok([payload]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(LatentClassArtifactError::Malformed("unsupported container layout".into()));
    };
    Ok((meta, payload))
}

/// A produced or consumed latent-class artifact: the declared request, the refitted result and
/// the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassArtifact {
    meta: LatentClassMeta,
    request: LatentClassRequestWire,
    result: LatentClassResult,
    /// Class index of each row. Computed when the record is sealed so a later
    /// report does not saturate an index that does not fit in `u32`.
    hard_assignment: Vec<u32>,
}

impl LatentClassArtifact {
    /// Fit the mixture on `request` and seal the answer with its identity, with no memory
    /// limit; see [`Self::seal_with`].
    ///
    /// # Errors
    /// The core fit's refusals (`latent_class.randomization_not_declared`,
    /// `latent_class.weak_class`, `latent_class.degenerate_class`, ...) and the format's bounds.
    pub fn seal(request: &LatentClassRequestWire) -> Result<Self, LatentClassArtifactError> {
        Self::seal_with(request, &unbounded_context())
    }

    /// Fit the mixture on `request` under the computation budget of `ctx` and seal the answer.
    ///
    /// The row count is not limited here: the planned working set must fit the context's hard
    /// memory limit and the context must not be cancelled. How much of the fit can be exported
    /// is decided later by [`Self::to_bytes_within`].
    ///
    /// # Errors
    /// The refusals of [`Self::seal`], `latent_class.memory_budget_exceeded` and
    /// `latent_class.cancelled`.
    pub fn seal_with(
        request: &LatentClassRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, LatentClassArtifactError> {
        let result = fit_of(request, ctx)?;
        let meta = build_meta(request, &result)?;
        let hard_assignment = hard_u32(&result)?;
        Ok(Self { meta, request: request.clone(), result, hard_assignment })
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
            hard_assignment: self.hard_assignment.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container, embedding the dataset.
    ///
    /// # Errors
    /// An empty id, an encoding failure or `latent_class.export_budget_exceeded` when the
    /// artifact would exceed [`MAX_LATENT_CLASS_ARTIFACT_BYTES`].
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, LatentClassArtifactError> {
        self.to_bytes_within(artifact_id, MAX_LATENT_CLASS_ARTIFACT_BYTES)
    }

    /// Serialize, embedding the dataset, when the whole artifact fits `export_budget` bytes (at
    /// most [`MAX_LATENT_CLASS_ARTIFACT_BYTES`]).
    ///
    /// # Errors
    /// An empty id, an encoding failure or `latent_class.export_budget_exceeded`, which names the
    /// bytes needed and the budget. The fitted result is untouched by the refusal.
    pub fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, LatentClassArtifactError> {
        encode_parts_impl(&self.meta, &data_bytes(&self.request.data), artifact_id, export_budget)
    }

    /// Serialize the record without its dataset, bound to it by the digest
    /// `identity.data_id`. A consumer must supply the dataset to
    /// [`Self::from_bytes_with`], which refuses any other.
    ///
    /// # Errors
    /// An empty id or an encoding failure.
    pub fn to_bytes_digest_bound(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, LatentClassArtifactError> {
        let mut meta = self.meta.clone();
        meta.data_binding = Some(DATA_BOUND_BY_DIGEST.to_owned());
        encode_parts_impl(&meta, &[], artifact_id, MAX_LATENT_CLASS_ARTIFACT_BYTES)
    }

    /// Consume an artifact that embeds its dataset by refitting under the stored seed and
    /// configuration, with no memory limit; see [`Self::from_bytes_with`].
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
        Self::from_bytes_with(bytes, expected, None, &unbounded_context())
    }

    /// Consume an artifact by refitting under the stored seed and configuration, within the
    /// replay budget of `ctx`.
    ///
    /// An artifact that embeds its dataset takes `data = None`. A digest-bound artifact
    /// ([`Self::to_bytes_digest_bound`]) takes the dataset to refit on, and refuses one whose
    /// digest is not the stored `identity.data_id` (`IdentityMismatch { field: "data" }`)
    /// before any work is done on it.
    ///
    /// # Errors
    /// The refusals of [`Self::from_bytes`], `latent_class.data_required` for a digest-bound
    /// artifact given no dataset, and `latent_class.memory_budget_exceeded` or
    /// `latent_class.cancelled` from the context.
    pub fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&LatentClassIdentity>,
        data: Option<&LatentClassDataWire>,
        ctx: &ExecutionContext,
    ) -> Result<Self, LatentClassArtifactError> {
        let (meta, payload) = decode_parts_impl(bytes)?;
        let unsupported = LatentClassArtifactError::UnsupportedSemantics;
        crate::artifact_codec::require_markers::<LatentClassArtifactError>(&[
            (meta.feature == LATENT_CLASS_ARTIFACT_FEATURE, "feature marker"),
            (meta.inference_claim == LATENT_CLASS_INFERENCE_CLAIM, "inference claim"),
            (meta.calibration == LATENT_CLASS_ARTIFACT_CALIBRATION, "calibration"),
            (meta.caveat == LATENT_CLASS_CAVEAT, "caveat"),
        ])?;
        let request = match (meta.data_binding.as_deref(), data) {
            (None, None) => request_of(&meta, &payload)?,
            (None, Some(_)) => {
                return Err(unsupported("a dataset was supplied for an artifact that embeds it"));
            }
            (Some(DATA_BOUND_BY_DIGEST), Some(supplied)) => {
                bound_request(&meta, &payload, supplied)?
            }
            (Some(DATA_BOUND_BY_DIGEST), None) => {
                return Err(LatentClassArtifactError::refused(
                    antecedent_core::reason_code!("route_not_supported"),
                    "latent_class.data_required",
                    format!(
                        "the artifact is bound to its dataset by digest {}; supply the dataset \
                         to replay it",
                        meta.identity.data_id
                    ),
                ));
            }
            (Some(_), _) => return Err(unsupported("data binding")),
        };
        let result = fit_of(&request, ctx)?;
        let rebuilt = build_meta(&request, &result)?;
        let hard_assignment = hard_u32(&result)?;
        if let Some(field) = identity_diff(&meta.identity, &rebuilt.identity) {
            return Err(if field == "result" {
                LatentClassArtifactError::ResultMismatch("canonical result")
            } else {
                LatentClassArtifactError::IdentityMismatch { field }
            });
        }
        // The binding describes the container, not the fit: compare the record as stored.
        let stored_form =
            LatentClassMeta { data_binding: meta.data_binding.clone(), ..rebuilt.clone() };
        if to_cbor(&meta)? != to_cbor(&stored_form)? {
            return Err(LatentClassArtifactError::ResultMismatch("record"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &rebuilt.identity) {
                return Err(LatentClassArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta: rebuilt, request, result, hard_assignment })
    }
}

/// The request of a digest-bound record: the stored configuration on the supplied dataset,
/// once the dataset's digest is the stored one.
fn bound_request(
    meta: &LatentClassMeta,
    payload: &[u8],
    supplied: &LatentClassDataWire,
) -> Result<LatentClassRequestWire, LatentClassArtifactError> {
    if !payload.is_empty() {
        return Err(LatentClassArtifactError::Malformed(
            "a digest-bound artifact embeds no data".into(),
        ));
    }
    if data_digest(supplied)? != meta.identity.data_id {
        return Err(LatentClassArtifactError::IdentityMismatch { field: "data" });
    }
    Ok(LatentClassRequestWire { config: meta.config.clone(), data: supplied.clone() })
}

#[cfg(test)]
mod data_section_tests {
    use super::data_bytes;
    use crate::latent_class_artifact::LatentClassDataWire;

    #[test]
    fn data_section_round_trips_signed_zero_and_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let data = LatentClassDataWire {
            outcome: vec![-0.0, 1.0],
            treatment: vec![nan, -1.0],
            covariate_names: vec!["x".into()],
            covariates: vec![vec![0.0, f64::MIN_POSITIVE]],
        };
        let bytes = data_bytes(&data);
        let columns = crate::width::columns_from_le(&bytes, 2, 3).unwrap();
        let again = super::data_from_columns(columns, data.covariate_names.clone());
        assert_eq!(data_bytes(&again), bytes);
        let bits = |sample: &[f64]| sample.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&again.outcome), bits(&data.outcome));
        assert_eq!(bits(&again.treatment), bits(&data.treatment));
        assert_eq!(bits(&again.covariates[0]), bits(&data.covariates[0]));
    }
}
