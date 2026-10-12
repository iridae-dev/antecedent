//! Joint vector-treatment artifact (`vector_treatment_v1`).
//!
//! A bounded, checksummed, sectioned container holding one joint fit of `k >= 2` treatments that
//! share ONE adjustment set and ONE row snapshot:
//!
//! * `vector_treatment_meta` (CBOR): version, feature marker, the frozen null, the **unmeasured**
//!   calibration coordinate, the covariance kind, the snapshot and adjustment-set declaration,
//!   the named coefficient order, the declared contrasts, the identity digests, the stored result
//!   (named coefficients, the **full** covariance, contrasts with their standard errors and Holm
//!   values, the joint Wald test) and the standing caveats;
//! * `vector_treatment_numbers` (little-endian `f64`): the embedded compact design the consumer
//!   replays. The replay mode is declared in the metadata:
//!   * `summary` (model-based covariance): `X'X` (`p x p`, row-major), `X'y` (`p`) and the
//!     residual sum of squares, with `p = 1 + adjustment + treatments`; the consumer recomputes
//!     coefficients and the full covariance from these alone, bit-identically;
//!   * `rows` (any heteroskedasticity-robust covariance, which needs the rows): the outcome, the
//!     adjustment columns and the treatment columns, replayed through the core fit. Replay above
//!     [`VECTOR_TREATMENT_HC_ROW_CAP`] rows is refused when sealing
//!     (`vector_treatment.hc_replay_row_cap_exceeded`).
//!
//! A consumer trusts none of the stored results. It rebuilds the design from the stored numbers,
//! reruns the core fit and refuses unless every stored value is bit-identical, then recomputes
//! the identity digests and compares them with the stored ones and, when the caller retained an
//! identity of its own, with that: a changed snapshot, adjustment set, coefficient order, design
//! numbers, covariance kind, contrast declaration, null or result is refused even when the
//! artifact was resealed consistently.
//!
//! What this does **not** say: every p-value (Wald, Holm, joint chi-square) is asymptotic and its
//! calibration is `unmeasured`; no coverage claim is made and no interval is produced. The caller
//! owns that the shared adjustment set is valid. Unknown major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::CanonicalWriter;
use antecedent_estimate::vector_treatment::{
    Contrast, DesignSummary, NamedColumn, TreatmentColumn, VECTOR_TREATMENT_INFERENCE_CLAIM,
    VECTOR_TREATMENT_MAX_TREATMENTS, VECTOR_TREATMENT_NULL, VectorCovariance, VectorTreatmentFit,
    VectorTreatmentInput, VectorTreatmentOptions, fit_vector_treatment,
    fit_vector_treatment_from_summary, summarize_vector_design,
};
use serde::{Deserialize, Serialize};

use crate::artifact_codec::CanonicalSize;
use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const VECTOR_TREATMENT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const VECTOR_TREATMENT_ARTIFACT_FEATURE: &str = "vector_treatment_v1";
/// Calibration coordinate: coverage, Type I error and power were not measured.
pub const VECTOR_TREATMENT_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_VECTOR_TREATMENT_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most rows a robust-covariance (`rows` replay) artifact may embed.
pub const VECTOR_TREATMENT_HC_ROW_CAP: usize = 4096;
/// Most shared adjustment columns an artifact may declare.
pub const VECTOR_TREATMENT_MAX_ADJUSTMENT: usize = 1024;
/// Standing caveat: inference is asymptotic and its calibration is unmeasured.
pub const VECTOR_TREATMENT_INFERENCE_CAVEAT: &str = "every p-value is an asymptotic Wald, Holm or chi-square value whose calibration is unmeasured; no coverage claim is made and no interval is produced";
/// Standing caveat: the adjustment set is the caller's.
pub const VECTOR_TREATMENT_ADJUSTMENT_CAVEAT: &str = "the caller owns that the shared adjustment set is valid; the artifact certifies the arithmetic of one joint regression, not identification";

const ARTIFACT_KIND: &str = crate::artifact_kinds::VECTOR_TREATMENT;
const META_SECTION: &str = "vector_treatment_meta";
const NUMBERS_SECTION: &str = "vector_treatment_numbers";
const REPLAY_SUMMARY: &str = "summary";
const REPLAY_ROWS: &str = "rows";

crate::sectioned_artifact_error!(
    /// Why a vector-treatment artifact was refused or could not be produced.
    VectorTreatmentArtifactError,
    "vector treatment",
    "vector_treatment",
    "result",
    false
);

fn invalid_request(message: &str) -> VectorTreatmentArtifactError {
    VectorTreatmentArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("vector_treatment.invalid_request: {message}"),
    }
}

/// One declared contrast of the named coefficients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastWire {
    /// Unique contrast name.
    pub name: String,
    /// `(coefficient name, weight)` pairs.
    pub weights: Vec<(String, f64)>,
}

/// A shared adjustment column.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjustmentColumnRequest {
    /// Unique column name.
    pub name: String,
    /// One value per row.
    pub values: Vec<f64>,
}

/// One treatment with the adjustment set and snapshot it was declared against.
#[derive(Clone, Debug, PartialEq)]
pub struct TreatmentRequest {
    /// Unique treatment (coefficient) name.
    pub name: String,
    /// One value per row.
    pub values: Vec<f64>,
    /// Adjustment column names this treatment was declared against.
    pub adjustment_set: Vec<String>,
    /// Row snapshot this treatment column was taken from.
    pub row_snapshot: String,
}

/// What a producer declares: one outcome, one shared adjustment block and snapshot, `k >= 2`
/// treatments, a covariance kind and the contrasts.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTreatmentRequest {
    /// Outcome, one value per row.
    pub outcome: Vec<f64>,
    /// Identity of the shared row snapshot.
    pub row_snapshot: String,
    /// The shared adjustment columns, in design order.
    pub adjustment: Vec<AdjustmentColumnRequest>,
    /// The treatments, in coefficient order.
    pub treatments: Vec<TreatmentRequest>,
    /// `model_based`, `hc0`, `hc1`, `hc2` or `hc3`.
    pub covariance: String,
    /// Declared contrasts, reported in this order.
    pub contrasts: Vec<ContrastWire>,
}

/// Identity digests of one joint fit. Each is BLAKE3 over a canonical byte encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorTreatmentIdentity {
    /// The row-snapshot identity and row count.
    pub snapshot_id: String,
    /// The shared adjustment set (order independent).
    pub adjustment_set_id: String,
    /// The named coefficient order.
    pub treatment_id: String,
    /// The replay mode and the embedded design numbers.
    pub design_id: String,
    /// The covariance estimator.
    pub covariance_id: String,
    /// The declared contrasts.
    pub contrast_id: String,
    /// The frozen null.
    pub null: String,
    /// The coefficients, covariance, contrasts and joint test.
    pub result_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// One treatment coefficient on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoefficientWire {
    /// Coefficient (treatment) name.
    pub name: String,
    /// OLS point estimate.
    pub estimate: f64,
    /// Square root of the covariance diagonal entry.
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
}

/// One declared contrast result on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastResultWire {
    /// Contrast name.
    pub name: String,
    /// `w' b`.
    pub estimate: f64,
    /// `sqrt(w' V w)` using the off-diagonals of the covariance.
    pub standard_error: f64,
    /// The standard error that wrongly treats the coefficients as independent.
    pub naive_independent_standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared family.
    pub p_holm: f64,
}

/// The joint Wald chi-square on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaldWire {
    /// The statistic.
    pub statistic: f64,
    /// Tested restrictions.
    pub degrees_of_freedom: u32,
    /// Asymptotic chi-square upper-tail p-value.
    pub p_value: f64,
}

/// The stored fit result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorResultWire {
    /// Named coefficients in declared order.
    pub coefficients: Vec<CoefficientWire>,
    /// Row-major `k x k` full covariance of the coefficient vector.
    pub covariance: Vec<f64>,
    /// Declared contrasts, in declaration order.
    pub contrasts: Vec<ContrastResultWire>,
    /// Wald test of the frozen null.
    pub joint_wald: WaldWire,
    /// Rows used.
    pub n_rows: u64,
    /// Residual degrees of freedom `n - p`.
    pub residual_df: u64,
    /// Residual variance `RSS / (n - p)`.
    pub residual_variance: f64,
}

impl VectorResultWire {
    fn from_fit(fit: &VectorTreatmentFit) -> Result<Self, VectorTreatmentArtifactError> {
        let degrees_of_freedom = u32::try_from(fit.joint_wald.degrees_of_freedom)
            .map_err(|_| VectorTreatmentArtifactError::LimitsExceeded("degrees of freedom"))?;
        Ok(Self {
            coefficients: fit
                .coefficients
                .iter()
                .map(|c| CoefficientWire {
                    name: c.name.clone(),
                    estimate: c.estimate,
                    standard_error: c.standard_error,
                    z: c.z,
                    p_value: c.p_value,
                })
                .collect(),
            covariance: fit.covariance.clone(),
            contrasts: fit
                .contrasts
                .iter()
                .map(|c| ContrastResultWire {
                    name: c.name.clone(),
                    estimate: c.estimate,
                    standard_error: c.standard_error,
                    naive_independent_standard_error: c.naive_independent_standard_error,
                    z: c.z,
                    p_value: c.p_value,
                    p_holm: c.p_holm,
                })
                .collect(),
            joint_wald: WaldWire {
                statistic: fit.joint_wald.statistic,
                degrees_of_freedom,
                p_value: fit.joint_wald.p_value,
            },
            n_rows: crate::width::lossless_u64(fit.n_rows),
            residual_df: crate::width::lossless_u64(fit.residual_df),
            residual_variance: fit.residual_variance,
        })
    }
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorTreatmentMeta {
    /// Major version ([`VECTOR_TREATMENT_ARTIFACT_VERSION`]).
    pub version: u16,
    /// Feature marker ([`VECTOR_TREATMENT_ARTIFACT_FEATURE`]).
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Always `asymptotic_wald_calibration_unmeasured`.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// `model_based`, `hc0`, `hc1`, `hc2` or `hc3`.
    pub covariance: String,
    /// `summary` or `rows`.
    pub replay: String,
    /// Identity of the shared row snapshot.
    pub row_snapshot: String,
    /// Rows of the snapshot.
    pub n_rows: u64,
    /// Shared adjustment column names, in design order.
    pub adjustment: Vec<String>,
    /// Treatment (coefficient) names, in order.
    pub treatments: Vec<String>,
    /// Declared contrasts.
    pub contrasts: Vec<ContrastWire>,
    /// Identity digests.
    pub identity: VectorTreatmentIdentity,
    /// The stored result.
    pub result: VectorResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorTreatmentReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The frozen null.
    pub null: String,
    /// Inference claim.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Covariance estimator.
    pub covariance_kind: String,
    /// Replay mode of the embedded design.
    pub replay: String,
    /// Shared row snapshot.
    pub row_snapshot: String,
    /// Shared adjustment column names.
    pub adjustment: Vec<String>,
    /// Treatment names in coefficient order.
    pub treatments: Vec<String>,
    /// Declared contrasts.
    pub declared_contrasts: Vec<ContrastWire>,
    /// The result.
    pub result: VectorResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
    /// Identity digests.
    pub identity: VectorTreatmentIdentity,
}

fn covariance_of(tag: &str) -> Result<VectorCovariance, VectorTreatmentArtifactError> {
    [
        VectorCovariance::ModelBased,
        VectorCovariance::Hc0,
        VectorCovariance::Hc1,
        VectorCovariance::Hc2,
        VectorCovariance::Hc3,
    ]
    .into_iter()
    .find(|kind| kind.as_str() == tag)
    .ok_or_else(|| invalid_request("unknown covariance kind"))
}

fn core_contrast(wire: &ContrastWire) -> Contrast {
    Contrast { name: wire.name.clone(), weights: wire.weights.clone() }
}

/// The declaration shared by sealing and replay.
struct Decl {
    row_snapshot: String,
    n_rows: usize,
    adjustment: Vec<String>,
    treatments: Vec<String>,
    covariance: String,
    contrasts: Vec<ContrastWire>,
}

impl Decl {
    fn from_meta(meta: &VectorTreatmentMeta) -> Result<Self, VectorTreatmentArtifactError> {
        let n_rows = usize::try_from(meta.n_rows)
            .map_err(|_| VectorTreatmentArtifactError::LimitsExceeded("rows"))?;
        Ok(Self {
            row_snapshot: meta.row_snapshot.clone(),
            n_rows,
            adjustment: meta.adjustment.clone(),
            treatments: meta.treatments.clone(),
            covariance: meta.covariance.clone(),
            contrasts: meta.contrasts.clone(),
        })
    }

    fn columns(&self) -> usize {
        1 + self.adjustment.len() + self.treatments.len()
    }
}

/// The embedded compact design.
enum Replay {
    Summary(DesignSummary),
    Rows { outcome: Vec<f64>, adjustment: Vec<Vec<f64>>, treatments: Vec<Vec<f64>> },
}

impl Replay {
    const fn tag(&self) -> &'static str {
        match self {
            Self::Summary(_) => REPLAY_SUMMARY,
            Self::Rows { .. } => REPLAY_ROWS,
        }
    }

    fn numbers(&self) -> Vec<f64> {
        match self {
            Self::Summary(s) => {
                s.gram.iter().chain(&s.xty).copied().chain(std::iter::once(s.rss)).collect()
            }
            Self::Rows { outcome, adjustment, treatments } => outcome
                .iter()
                .chain(adjustment.iter().flatten())
                .chain(treatments.iter().flatten())
                .copied()
                .collect(),
        }
    }

    fn from_numbers(
        decl: &Decl,
        tag: &str,
        numbers: &[f64],
    ) -> Result<Self, VectorTreatmentArtifactError> {
        let p = decl.columns();
        let malformed = |what: &str| VectorTreatmentArtifactError::Malformed(what.to_owned());
        match tag {
            REPLAY_SUMMARY => {
                if numbers.len() != p * p + p + 1 {
                    return Err(malformed("numbers section length"));
                }
                let mut rest = numbers;
                Ok(Self::Summary(DesignSummary {
                    n_rows: decl.n_rows,
                    first_treatment: 1 + decl.adjustment.len(),
                    gram: crate::width::split_reals(&mut rest, p * p).to_vec(),
                    xty: crate::width::split_reals(&mut rest, p).to_vec(),
                    rss: crate::width::split_reals(&mut rest, 1)[0],
                }))
            }
            REPLAY_ROWS => {
                if decl.n_rows > VECTOR_TREATMENT_HC_ROW_CAP {
                    return Err(VectorTreatmentArtifactError::LimitsExceeded("replay rows"));
                }
                if decl.n_rows.checked_mul(p) != Some(numbers.len()) {
                    return Err(malformed("numbers section length"));
                }
                let n = decl.n_rows;
                let mut columns = numbers.chunks_exact(n.max(1)).map(<[f64]>::to_vec);
                let outcome = columns.next().unwrap_or_default();
                let adjustment: Vec<Vec<f64>> =
                    columns.by_ref().take(decl.adjustment.len()).collect();
                let treatments: Vec<Vec<f64>> = columns.collect();
                Ok(Self::Rows { outcome, adjustment, treatments })
            }
            _ => Err(malformed("replay mode")),
        }
    }
}

fn identity_of(
    decl: &Decl,
    replay: &str,
    numbers: &[f64],
    result: &VectorResultWire,
) -> VectorTreatmentIdentity {
    let mut snapshot = CanonicalWriter::new("vector_treatment_v1.snapshot");
    snapshot.text(&decl.row_snapshot);
    snapshot.size(decl.n_rows);

    let mut adjustment = CanonicalWriter::new("vector_treatment_v1.adjustment_set");
    let sorted: BTreeSet<&str> = decl.adjustment.iter().map(String::as_str).collect();
    adjustment.size(sorted.len());
    for name in sorted {
        adjustment.text(name);
    }

    let mut treatment = CanonicalWriter::new("vector_treatment_v1.treatments");
    treatment.size(decl.treatments.len());
    for name in &decl.treatments {
        treatment.text(name);
    }

    let mut design = CanonicalWriter::new("vector_treatment_v1.design");
    design.text(replay);
    design.size(decl.adjustment.len());
    for name in &decl.adjustment {
        design.text(name);
    }
    design.size(numbers.len());
    for value in numbers {
        design.real(*value);
    }

    let mut covariance = CanonicalWriter::new("vector_treatment_v1.covariance");
    covariance.text(&decl.covariance);

    let mut contrasts = CanonicalWriter::new("vector_treatment_v1.contrasts");
    contrasts.size(decl.contrasts.len());
    for contrast in &decl.contrasts {
        contrasts.text(&contrast.name);
        let mut weights: Vec<&(String, f64)> = contrast.weights.iter().collect();
        weights.sort_by(|a, b| a.0.cmp(&b.0));
        contrasts.size(weights.len());
        for (name, weight) in weights {
            contrasts.text(name);
            contrasts.real(*weight);
        }
    }

    let mut outcome = CanonicalWriter::new("vector_treatment_v1.result");
    for c in &result.coefficients {
        outcome.text(&c.name);
        outcome.real(c.estimate);
        outcome.real(c.standard_error);
    }
    for value in &result.covariance {
        outcome.real(*value);
    }
    for c in &result.contrasts {
        outcome.text(&c.name);
        outcome.real(c.estimate);
        outcome.real(c.standard_error);
        outcome.real(c.p_holm);
    }
    outcome.real(result.joint_wald.statistic);
    outcome.real(result.joint_wald.p_value);
    outcome.real(result.residual_variance);

    let snapshot_id = snapshot.finish_hex();
    let adjustment_set_id = adjustment.finish_hex();
    let treatment_id = treatment.finish_hex();
    let design_id = design.finish_hex();
    let covariance_id = covariance.finish_hex();
    let contrast_id = contrasts.finish_hex();
    let result_id = outcome.finish_hex();
    let mut whole = CanonicalWriter::new("vector_treatment_v1.identity");
    for part in [
        snapshot_id.as_str(),
        adjustment_set_id.as_str(),
        treatment_id.as_str(),
        design_id.as_str(),
        covariance_id.as_str(),
        contrast_id.as_str(),
        VECTOR_TREATMENT_NULL,
        result_id.as_str(),
    ] {
        whole.text(part);
    }
    VectorTreatmentIdentity {
        snapshot_id,
        adjustment_set_id,
        treatment_id,
        design_id,
        covariance_id,
        contrast_id,
        null: VECTOR_TREATMENT_NULL.to_owned(),
        result_id,
        digest: whole.finish_hex(),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &VectorTreatmentIdentity,
    other: &VectorTreatmentIdentity,
) -> Option<&'static str> {
    if stored.snapshot_id != other.snapshot_id {
        Some("row_snapshot")
    } else if stored.adjustment_set_id != other.adjustment_set_id {
        Some("adjustment_set")
    } else if stored.treatment_id != other.treatment_id {
        Some("treatment_vector")
    } else if stored.design_id != other.design_id {
        Some("design")
    } else if stored.covariance_id != other.covariance_id {
        Some("covariance_kind")
    } else if stored.contrast_id != other.contrast_id {
        Some("contrasts")
    } else if stored.null != other.null {
        Some("null")
    } else if stored.result_id != other.result_id {
        Some("result")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

pub(crate) fn reals_to_bytes(values: &[f64]) -> Vec<u8> {
    crate::width::f64s_to_le_bytes(values)
}

pub(crate) fn bytes_to_reals(bytes: &[u8]) -> Vec<f64> {
    crate::width::f64s_from_le_bytes(bytes)
}

struct Computed {
    fit: VectorTreatmentFit,
    result: VectorResultWire,
    identity: VectorTreatmentIdentity,
    numbers: Vec<f64>,
}

fn check_decl(decl: &Decl) -> Result<(), VectorTreatmentArtifactError> {
    if decl.treatments.len() > VECTOR_TREATMENT_MAX_TREATMENTS {
        return Err(VectorTreatmentArtifactError::LimitsExceeded("treatments"));
    }
    if decl.adjustment.len() > VECTOR_TREATMENT_MAX_ADJUSTMENT {
        return Err(VectorTreatmentArtifactError::LimitsExceeded("adjustment columns"));
    }
    if decl.contrasts.len() > VECTOR_TREATMENT_MAX_ADJUSTMENT {
        return Err(VectorTreatmentArtifactError::LimitsExceeded("contrasts"));
    }
    Ok(())
}

fn compute(decl: &Decl, replay: &Replay) -> Result<Computed, VectorTreatmentArtifactError> {
    check_decl(decl)?;
    let options = VectorTreatmentOptions {
        covariance: covariance_of(&decl.covariance)?,
        contrasts: decl.contrasts.iter().map(core_contrast).collect(),
    };
    let fit = match replay {
        Replay::Summary(summary) => {
            if summary.n_rows != decl.n_rows {
                return Err(VectorTreatmentArtifactError::Malformed("row count".into()));
            }
            fit_vector_treatment_from_summary(summary, &decl.treatments, &options)?
        }
        Replay::Rows { outcome, adjustment, treatments } => {
            let input = VectorTreatmentInput {
                outcome: outcome.clone(),
                row_snapshot: decl.row_snapshot.clone(),
                adjustment: decl
                    .adjustment
                    .iter()
                    .zip(adjustment)
                    .map(|(name, values)| NamedColumn {
                        name: name.clone(),
                        values: values.clone(),
                    })
                    .collect(),
                treatments: decl
                    .treatments
                    .iter()
                    .zip(treatments)
                    .map(|(name, values)| TreatmentColumn {
                        name: name.clone(),
                        values: values.clone(),
                        adjustment_set: decl.adjustment.clone(),
                        row_snapshot: decl.row_snapshot.clone(),
                    })
                    .collect(),
            };
            fit_vector_treatment(&input, &options)?
        }
    };
    let result = VectorResultWire::from_fit(&fit)?;
    let numbers = replay.numbers();
    let identity = identity_of(decl, replay.tag(), &numbers, &result);
    Ok(Computed { fit, result, identity, numbers })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`VectorTreatmentArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &VectorTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
    encode_parts_impl(meta, numbers, artifact_id)
}

fn encode_parts_impl(
    meta: &VectorTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(VectorTreatmentArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes =
        to_cbor(meta).map_err(|error| VectorTreatmentArtifactError::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "vector_treatment_wald_unmeasured",
            max_bytes: MAX_VECTOR_TREATMENT_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            crate::OutSection {
                id: META_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &meta_bytes,
            },
            crate::OutSection {
                id: NUMBERS_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: numbers,
            },
        ],
    )
    .map_err(|error| crate::map_artifact_check!(error, VectorTreatmentArtifactError))
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
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(VectorTreatmentMeta, Vec<u8>), VectorTreatmentArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(VectorTreatmentMeta, Vec<u8>), VectorTreatmentArtifactError> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_VECTOR_TREATMENT_ARTIFACT_BYTES,
        &[META_SECTION, NUMBERS_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, VectorTreatmentArtifactError))?;
    let Ok([meta_bytes, payload]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(VectorTreatmentArtifactError::Malformed("unsupported container layout".into()));
    };
    let meta = crate::decode_with_version(&meta_bytes, |version| {
        if version == VECTOR_TREATMENT_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(VectorTreatmentArtifactError::UnsupportedVersion { version })
        }
    })?;
    Ok((meta, payload))
}

fn standing_caveats() -> Vec<String> {
    vec![
        VECTOR_TREATMENT_INFERENCE_CAVEAT.to_owned(),
        VECTOR_TREATMENT_ADJUSTMENT_CAVEAT.to_owned(),
    ]
}

/// A produced or consumed joint-fit artifact: the declaration, the recomputed fit and the
/// identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTreatmentArtifact {
    meta: VectorTreatmentMeta,
    numbers: Vec<f64>,
    fit: VectorTreatmentFit,
}

impl VectorTreatmentArtifact {
    /// Fit the declared treatments jointly and seal the answer with its embedded design and
    /// identity.
    ///
    /// # Errors
    /// The core fit's refusals (`route_not_supported` for a treatment on a different adjustment
    /// set, snapshot or row count; `design_rank_deficient`; `invalid_argument`), the
    /// `vector_treatment.hc_replay_row_cap_exceeded` refusal for a robust covariance above
    /// [`VECTOR_TREATMENT_HC_ROW_CAP`] rows, and the format's bounds.
    pub fn seal(request: &VectorTreatmentRequest) -> Result<Self, VectorTreatmentArtifactError> {
        let kind = covariance_of(&request.covariance)?;
        let options = VectorTreatmentOptions {
            covariance: kind,
            contrasts: request.contrasts.iter().map(core_contrast).collect(),
        };
        let input = VectorTreatmentInput {
            outcome: request.outcome.clone(),
            row_snapshot: request.row_snapshot.clone(),
            adjustment: request
                .adjustment
                .iter()
                .map(|c| NamedColumn { name: c.name.clone(), values: c.values.clone() })
                .collect(),
            treatments: request
                .treatments
                .iter()
                .map(|t| TreatmentColumn {
                    name: t.name.clone(),
                    values: t.values.clone(),
                    adjustment_set: t.adjustment_set.clone(),
                    row_snapshot: t.row_snapshot.clone(),
                })
                .collect(),
        };
        let decl = Decl {
            row_snapshot: request.row_snapshot.clone(),
            n_rows: request.outcome.len(),
            adjustment: request.adjustment.iter().map(|c| c.name.clone()).collect(),
            treatments: request.treatments.iter().map(|t| t.name.clone()).collect(),
            covariance: request.covariance.clone(),
            contrasts: request.contrasts.clone(),
        };
        check_decl(&decl)?;
        let replay = if kind == VectorCovariance::ModelBased {
            Replay::Summary(summarize_vector_design(&input)?)
        } else {
            if request.outcome.len() > VECTOR_TREATMENT_HC_ROW_CAP {
                return Err(VectorTreatmentArtifactError::Refused {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    message: format!(
                        "vector_treatment.hc_replay_row_cap_exceeded: a robust covariance replays \
                         from the rows and an artifact embeds at most {VECTOR_TREATMENT_HC_ROW_CAP}"
                    ),
                });
            }
            // The request's own per-treatment declarations are checked here; the replay below
            // reads the shared ones.
            fit_vector_treatment(&input, &options)?;
            Replay::Rows {
                outcome: request.outcome.clone(),
                adjustment: request.adjustment.iter().map(|c| c.values.clone()).collect(),
                treatments: request.treatments.iter().map(|t| t.values.clone()).collect(),
            }
        };
        let computed = compute(&decl, &replay)?;
        let meta = VectorTreatmentMeta {
            version: VECTOR_TREATMENT_ARTIFACT_VERSION,
            feature: VECTOR_TREATMENT_ARTIFACT_FEATURE.to_owned(),
            null: VECTOR_TREATMENT_NULL.to_owned(),
            inference_claim: VECTOR_TREATMENT_INFERENCE_CLAIM.to_owned(),
            calibration: VECTOR_TREATMENT_CALIBRATION.to_owned(),
            covariance: decl.covariance.clone(),
            replay: replay.tag().to_owned(),
            row_snapshot: decl.row_snapshot.clone(),
            n_rows: crate::width::lossless_u64(decl.n_rows),
            adjustment: decl.adjustment.clone(),
            treatments: decl.treatments.clone(),
            contrasts: decl.contrasts.clone(),
            identity: computed.identity,
            result: computed.result,
            caveats: standing_caveats(),
        };
        Ok(Self { meta, numbers: computed.numbers, fit: computed.fit })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &VectorTreatmentMeta {
        &self.meta
    }

    /// The core fit.
    #[must_use]
    pub fn fit(&self) -> &VectorTreatmentFit {
        &self.fit
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &VectorTreatmentIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> VectorTreatmentReportWire {
        VectorTreatmentReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            null: self.meta.null.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            calibration: self.meta.calibration.clone(),
            covariance_kind: self.meta.covariance.clone(),
            replay: self.meta.replay.clone(),
            row_snapshot: self.meta.row_snapshot.clone(),
            adjustment: self.meta.adjustment.clone(),
            treatments: self.meta.treatments.clone(),
            declared_contrasts: self.meta.contrasts.clone(),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
        encode_parts_impl(&self.meta, &reals_to_bytes(&self.numbers), artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected` is an identity the consumer retained independently (for example from the
    /// producer's result); when given, every identity field must match it, so a resealed change
    /// of snapshot, adjustment set, coefficient order, design, covariance kind, contrasts, null
    /// or result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`VectorTreatmentArtifactError::IdentityMismatch`]), a stored result that does not
    /// replay, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&VectorTreatmentIdentity>,
    ) -> Result<Self, VectorTreatmentArtifactError> {
        let (meta, raw) = decode_parts_impl(bytes)?;
        if meta.feature != VECTOR_TREATMENT_ARTIFACT_FEATURE {
            return Err(VectorTreatmentArtifactError::UnsupportedSemantics("feature marker"));
        }
        if meta.null != VECTOR_TREATMENT_NULL {
            return Err(VectorTreatmentArtifactError::IdentityMismatch { field: "null" });
        }
        if meta.inference_claim != VECTOR_TREATMENT_INFERENCE_CLAIM {
            return Err(VectorTreatmentArtifactError::UnsupportedSemantics("inference claim"));
        }
        if meta.calibration != VECTOR_TREATMENT_CALIBRATION {
            return Err(VectorTreatmentArtifactError::UnsupportedSemantics("calibration"));
        }
        if meta.caveats != standing_caveats() {
            return Err(VectorTreatmentArtifactError::UnsupportedSemantics("caveats"));
        }
        let decl = Decl::from_meta(&meta)?;
        check_decl(&decl)?;
        if raw.len() % 8 != 0 {
            return Err(VectorTreatmentArtifactError::Malformed("numbers section length".into()));
        }
        let replay = Replay::from_numbers(&decl, &meta.replay, &bytes_to_reals(&raw))?;
        let computed = compute(&decl, &replay)?;
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(VectorTreatmentArtifactError::IdentityMismatch { field });
        }
        if meta.result != computed.result {
            return Err(VectorTreatmentArtifactError::ResultMismatch(
                "coefficients, covariance, contrasts or joint test",
            ));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &computed.identity) {
                return Err(VectorTreatmentArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, numbers: computed.numbers, fit: computed.fit })
    }
}
