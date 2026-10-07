//! B4 categorical-treatment artifact (`categorical_treatment_v1`).
//!
//! A bounded, checksummed, sectioned container holding one dummy-coded categorical regime fit:
//!
//! * `categorical_treatment_meta` (CBOR): version, feature marker, the frozen omnibus null and the
//!   monotonicity null (when a monotonicity test is declared), the **unmeasured** calibration
//!   coordinate, the declared design (level order or canonical set, scale, reference, minimum
//!   level rows, requested pairs, monotonicity direction, covariance kind), the snapshot and
//!   adjustment declaration, the identity digests, the stored result (level counts, dummy
//!   coefficients, the full covariance, per-level and pairwise contrasts with their Holm values,
//!   the family size, the omnibus Wald test and the monotonicity result with its adjacent steps)
//!   and the standing caveats;
//! * `categorical_treatment_numbers` (little-endian `f64`): the embedded compact design.
//!   * `summary` (model-based covariance): `X'X`, `X'y` and the residual sum of squares of the
//!     dummy-coded design `[intercept, adjustment..., level:<non-reference>...]`, with the level
//!     counts in the metadata (they must agree with the Gram diagonal);
//!   * `rows` (robust covariance): the outcome and adjustment columns, with the row level labels
//!     in the metadata, replayed through the core fit. Replay above
//!     [`CATEGORICAL_HC_ROW_CAP`] rows is refused when sealing
//!     (`categorical_treatment.hc_replay_row_cap_exceeded`).
//!
//! A consumer trusts none of the stored results: it rebuilds the design from the stored numbers,
//! reruns the core fit and refuses unless every stored value is bit-identical, then recomputes
//! the identity digests and compares them with the stored ones and, when the caller retained an
//! identity of its own, with that. A changed level order, reference, scale, family, covariance
//! kind, design, null or result is refused even when the artifact was resealed consistently.
//!
//! What this does **not** say: every p-value (Wald, Holm, chi-square, monotonicity) is asymptotic
//! and calibration is `unmeasured`; the monotonicity p-value is a conservative union-intersection
//! bound whose rejection is evidence against monotonicity, not a proof of it when unrejected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_estimate::EstimationError;
use antecedent_estimate::categorical_treatment::{
    CATEGORICAL_INFERENCE_CLAIM, CATEGORICAL_OMNIBUS_NULL, CategoricalSummary,
    CategoricalTreatmentFit, CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelCount,
    LevelScale, MonotonicityDirection, fit_categorical_from_summary, fit_categorical_treatment,
    summarize_categorical_design,
};
use antecedent_estimate::vector_treatment::{DesignSummary, NamedColumn, VectorCovariance};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::vector_treatment_artifact::{
    AdjustmentColumnRequest, Canon, CoefficientWire, VECTOR_TREATMENT_ADJUSTMENT_CAVEAT,
    VECTOR_TREATMENT_MAX_ADJUSTMENT, WaldWire, bytes_to_reals, reals_to_bytes,
};
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const CATEGORICAL_TREATMENT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const CATEGORICAL_TREATMENT_ARTIFACT_FEATURE: &str = "categorical_treatment_v1";
/// Calibration coordinate: Type I error, coverage and power were not measured.
pub const CATEGORICAL_TREATMENT_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most rows a robust-covariance (`rows` replay) artifact may embed.
pub const CATEGORICAL_HC_ROW_CAP: usize = 4096;
/// Most declared levels an artifact may carry.
pub const CATEGORICAL_MAX_LEVELS: usize = 256;
/// Standing caveat: inference is asymptotic and its calibration is unmeasured.
pub const CATEGORICAL_INFERENCE_CAVEAT: &str = "every p-value is an asymptotic Wald, Holm or chi-square value whose calibration is unmeasured; no coverage claim is made and no interval is produced";
/// Standing caveat of the monotonicity test.
pub const CATEGORICAL_MONOTONICITY_CAVEAT: &str = "the monotonicity p-value is a conservative one-sided union-intersection (Bonferroni) bound; rejection is evidence against monotonicity in the declared direction, failing to reject does not prove it, and the test's Type I error and power are unmeasured";

const ARTIFACT_KIND: &str = "categorical_treatment_v1";
const META_SECTION: &str = "categorical_treatment_meta";
const NUMBERS_SECTION: &str = "categorical_treatment_numbers";
const REPLAY_SUMMARY: &str = "summary";
const REPLAY_ROWS: &str = "rows";

/// Why a categorical-treatment artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CategoricalTreatmentArtifactError {
    /// The bytes do not decode as this format.
    #[error("categorical treatment artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported categorical treatment artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker, claim, calibration coordinate or caveat that is not this format's.
    #[error("unsupported categorical treatment semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("categorical treatment consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed categorical treatment artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("categorical_treatment.wrong_contract: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored result differs from the recomputed one.
    #[error("stored categorical treatment result does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("categorical treatment artifact does not encode: {0}")]
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

impl CategoricalTreatmentArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a core refusal and for a changed identity field; absent for corruption,
    /// unsupported versions and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, message } => {
                let (detail, text) = message.split_once(": ").unwrap_or((message.as_str(), ""));
                Some((*code, detail.to_owned(), text.to_owned()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "categorical_treatment.wrong_contract".to_owned(),
                format!("{field} changed"),
            )),
            _ => None,
        }
    }
}

impl From<EstimationError> for CategoricalTreatmentArtifactError {
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

impl From<crate::IoError> for CategoricalTreatmentArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

fn invalid_request(message: &str) -> CategoricalTreatmentArtifactError {
    CategoricalTreatmentArtifactError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("categorical_treatment.invalid_request: {message}"),
    }
}

/// The declared design of the categorical regime on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalSpecWire {
    /// Declared level set (the scale, when ordered).
    pub declared_levels: Vec<String>,
    /// `ordered` or `unordered`.
    pub scale: String,
    /// Reference level.
    pub reference: String,
    /// Fewest rows a declared level may have.
    pub min_level_rows: u64,
    /// Requested pairwise contrasts `(from, to)`.
    pub pairwise: Vec<(String, String)>,
    /// `non_decreasing`, `non_increasing` or absent.
    pub monotonicity: Option<String>,
    /// `model_based`, `hc0`, `hc1`, `hc2` or `hc3`.
    pub covariance: String,
}

/// What a producer declares: rows with level labels, the shared adjustment block and the design.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentRequest {
    /// Outcome, one value per row.
    pub outcome: Vec<f64>,
    /// Identity of the row snapshot.
    pub row_snapshot: String,
    /// Shared adjustment columns, in design order.
    pub adjustment: Vec<AdjustmentColumnRequest>,
    /// Level label of each row.
    pub levels: Vec<String>,
    /// The declared design.
    pub spec: CategoricalSpecWire,
}

/// Identity digests of one categorical fit. Each is BLAKE3 over a canonical byte encoding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalTreatmentIdentity {
    /// The row-snapshot identity and row count.
    pub snapshot_id: String,
    /// The shared adjustment set (order independent).
    pub adjustment_set_id: String,
    /// Scale, reference and canonical level order.
    pub level_scale_id: String,
    /// Replay mode, level counts or labels and the embedded numbers.
    pub design_id: String,
    /// The covariance estimator.
    pub covariance_id: String,
    /// Requested pairs, monotonicity direction and the minimum level rows.
    pub family_id: String,
    /// The frozen omnibus null.
    pub omnibus_null: String,
    /// The frozen monotonicity null; empty when no monotonicity test is declared.
    pub monotonicity_null: String,
    /// Coefficients, covariance, contrasts, omnibus and monotonicity results.
    pub result_id: String,
    /// Digest over all of the above.
    pub digest: String,
}

/// Rows observed at a level, on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelCountWire {
    /// Level name.
    pub level: String,
    /// Rows at the level.
    pub rows: u64,
}

/// A level against the reference, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LevelContrastWire {
    /// The level.
    pub level: String,
    /// The reference level.
    pub reference: String,
    /// `effect(level) - effect(reference)`.
    pub estimate: f64,
    /// Standard error from the dummy-regression covariance.
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared family.
    pub p_holm: f64,
}

/// A requested pairwise contrast, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairContrastWire {
    /// Baseline level.
    pub from: String,
    /// Compared level.
    pub to: String,
    /// `effect(to) - effect(from)`.
    pub estimate: f64,
    /// Standard error (off-diagonals included).
    pub standard_error: f64,
    /// `estimate / standard_error`.
    pub z: f64,
    /// Two-sided asymptotic normal p-value.
    pub p_value: f64,
    /// Holm-adjusted p-value across the declared family.
    pub p_holm: f64,
}

/// One adjacent step of the ordered effects, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjacentStepWire {
    /// Lower level.
    pub from: String,
    /// Next level up.
    pub to: String,
    /// `effect(to) - effect(from)`.
    pub difference: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// `difference / standard_error`.
    pub z: f64,
}

/// The monotonicity result, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonotonicityWire {
    /// `non_decreasing` or `non_increasing`.
    pub direction: String,
    /// The frozen null of that direction.
    pub null: String,
    /// The adjacent steps.
    pub steps: Vec<AdjacentStepWire>,
    /// Oriented minimum step z-score.
    pub statistic: f64,
    /// Conservative union-intersection p-value.
    pub p_value: f64,
    /// Always `true` (Bonferroni bound).
    pub conservative: bool,
    /// Always `unmeasured`.
    pub calibration: String,
}

/// The stored fit result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalResultWire {
    /// Reference level.
    pub reference: String,
    /// `ordered` or `unordered`.
    pub scale: String,
    /// Canonical level order used.
    pub level_order: Vec<String>,
    /// Rows per level, in canonical order.
    pub counts: Vec<LevelCountWire>,
    /// Dummy coefficients (`level:<name>`), non-reference levels in canonical order.
    pub coefficients: Vec<CoefficientWire>,
    /// Row-major covariance of the dummy coefficients.
    pub covariance: Vec<f64>,
    /// Per-level contrasts against the reference.
    pub level_contrasts: Vec<LevelContrastWire>,
    /// Requested pairwise contrasts, in request order.
    pub pairwise: Vec<PairContrastWire>,
    /// Size of the Holm family.
    pub family_size: u64,
    /// Omnibus Wald test of the frozen null.
    pub omnibus: WaldWire,
    /// Monotonicity test, when declared.
    pub monotonicity: Option<MonotonicityWire>,
    /// Covariance estimator.
    pub covariance_kind: String,
    /// Always `unmeasured`.
    pub calibration: String,
}

const fn scale_tag(scale: LevelScale) -> &'static str {
    match scale {
        LevelScale::Ordered => "ordered",
        LevelScale::Unordered => "unordered",
    }
}

const fn direction_tag(direction: MonotonicityDirection) -> &'static str {
    match direction {
        MonotonicityDirection::NonDecreasing => "non_decreasing",
        MonotonicityDirection::NonIncreasing => "non_increasing",
    }
}

fn size(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

impl CategoricalResultWire {
    fn from_fit(fit: &CategoricalTreatmentFit) -> Self {
        Self {
            reference: fit.reference.clone(),
            scale: scale_tag(fit.scale).to_owned(),
            level_order: fit.level_order.clone(),
            counts: fit
                .counts
                .iter()
                .map(|c| LevelCountWire { level: c.level.clone(), rows: size(c.rows) })
                .collect(),
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
            level_contrasts: fit
                .level_contrasts
                .iter()
                .map(|c| LevelContrastWire {
                    level: c.level.clone(),
                    reference: c.reference.clone(),
                    estimate: c.estimate,
                    standard_error: c.standard_error,
                    z: c.z,
                    p_value: c.p_value,
                    p_holm: c.p_holm,
                })
                .collect(),
            pairwise: fit
                .pairwise
                .iter()
                .map(|c| PairContrastWire {
                    from: c.from.clone(),
                    to: c.to.clone(),
                    estimate: c.estimate,
                    standard_error: c.standard_error,
                    z: c.z,
                    p_value: c.p_value,
                    p_holm: c.p_holm,
                })
                .collect(),
            family_size: size(fit.family_size),
            omnibus: WaldWire {
                statistic: fit.omnibus.statistic,
                degrees_of_freedom: u32::try_from(fit.omnibus.degrees_of_freedom)
                    .unwrap_or(u32::MAX),
                p_value: fit.omnibus.p_value,
            },
            monotonicity: fit.monotonicity.as_ref().map(|m| MonotonicityWire {
                direction: direction_tag(m.direction).to_owned(),
                null: m.null.to_owned(),
                steps: m
                    .steps
                    .iter()
                    .map(|s| AdjacentStepWire {
                        from: s.from.clone(),
                        to: s.to.clone(),
                        difference: s.difference,
                        standard_error: s.standard_error,
                        z: s.z,
                    })
                    .collect(),
                statistic: m.statistic,
                p_value: m.p_value,
                conservative: m.conservative,
                calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
            }),
            covariance_kind: fit.covariance_kind.as_str().to_owned(),
            calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
        }
    }
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalTreatmentMeta {
    /// Major version ([`CATEGORICAL_TREATMENT_ARTIFACT_VERSION`]).
    pub version: u16,
    /// Feature marker ([`CATEGORICAL_TREATMENT_ARTIFACT_FEATURE`]).
    pub feature: String,
    /// The frozen omnibus null.
    pub null: String,
    /// The frozen monotonicity null, when a monotonicity test is declared.
    pub monotonicity_null: Option<String>,
    /// Always `asymptotic_wald_calibration_unmeasured`.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// `summary` or `rows`.
    pub replay: String,
    /// Identity of the row snapshot.
    pub row_snapshot: String,
    /// Rows of the snapshot.
    pub n_rows: u64,
    /// Shared adjustment column names, in design order.
    pub adjustment: Vec<String>,
    /// The declared design.
    pub spec: CategoricalSpecWire,
    /// Level counts of the embedded design (`summary` replay; empty for `rows`).
    pub counts: Vec<LevelCountWire>,
    /// Row level labels (`rows` replay; empty for `summary`).
    pub row_levels: Vec<String>,
    /// Identity digests.
    pub identity: CategoricalTreatmentIdentity,
    /// The stored result.
    pub result: CategoricalResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalTreatmentReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The frozen omnibus null.
    pub null: String,
    /// The frozen monotonicity null, when declared.
    pub monotonicity_null: Option<String>,
    /// Inference claim.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Replay mode of the embedded design.
    pub replay: String,
    /// Row snapshot.
    pub row_snapshot: String,
    /// Shared adjustment column names.
    pub adjustment: Vec<String>,
    /// The declared design.
    pub spec: CategoricalSpecWire,
    /// The result.
    pub result: CategoricalResultWire,
    /// Standing caveats.
    pub caveats: Vec<String>,
    /// Identity digests.
    pub identity: CategoricalTreatmentIdentity,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

fn covariance_of(tag: &str) -> Result<VectorCovariance, CategoricalTreatmentArtifactError> {
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

fn spec_of(
    wire: &CategoricalSpecWire,
) -> Result<CategoricalTreatmentSpec, CategoricalTreatmentArtifactError> {
    let scale = match wire.scale.as_str() {
        "ordered" => LevelScale::Ordered,
        "unordered" => LevelScale::Unordered,
        _ => return Err(invalid_request("unknown level scale")),
    };
    let monotonicity = match wire.monotonicity.as_deref() {
        None => None,
        Some("non_decreasing") => Some(MonotonicityDirection::NonDecreasing),
        Some("non_increasing") => Some(MonotonicityDirection::NonIncreasing),
        Some(_) => return Err(invalid_request("unknown monotonicity direction")),
    };
    Ok(CategoricalTreatmentSpec {
        declared_levels: wire.declared_levels.clone(),
        scale,
        reference: wire.reference.clone(),
        min_level_rows: usize::try_from(wire.min_level_rows)
            .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("min_level_rows"))?,
        pairwise: wire.pairwise.clone(),
        monotonicity,
        covariance: covariance_of(&wire.covariance)?,
    })
}

/// The declaration shared by sealing and replay.
struct Decl {
    row_snapshot: String,
    n_rows: usize,
    adjustment: Vec<String>,
    spec: CategoricalSpecWire,
}

impl Decl {
    fn from_meta(
        meta: &CategoricalTreatmentMeta,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self {
            row_snapshot: meta.row_snapshot.clone(),
            n_rows: usize::try_from(meta.n_rows)
                .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("rows"))?,
            adjustment: meta.adjustment.clone(),
            spec: meta.spec.clone(),
        })
    }

    /// Design columns: intercept, adjustment, one dummy per non-reference level.
    fn columns(&self) -> usize {
        1 + self.adjustment.len() + self.spec.declared_levels.len().saturating_sub(1)
    }
}

/// The embedded compact design.
enum Replay {
    Summary(CategoricalSummary),
    Rows { outcome: Vec<f64>, adjustment: Vec<Vec<f64>>, levels: Vec<String> },
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
            Self::Summary(s) => s
                .design
                .gram
                .iter()
                .chain(&s.design.xty)
                .copied()
                .chain(std::iter::once(s.design.rss))
                .collect(),
            Self::Rows { outcome, adjustment, .. } => {
                outcome.iter().chain(adjustment.iter().flatten()).copied().collect()
            }
        }
    }

    fn counts(&self) -> Vec<LevelCountWire> {
        match self {
            Self::Summary(s) => s
                .counts
                .iter()
                .map(|c| LevelCountWire { level: c.level.clone(), rows: size(c.rows) })
                .collect(),
            Self::Rows { .. } => Vec::new(),
        }
    }

    fn row_levels(&self) -> Vec<String> {
        match self {
            Self::Summary(_) => Vec::new(),
            Self::Rows { levels, .. } => levels.clone(),
        }
    }

    fn from_parts(
        decl: &Decl,
        meta: &CategoricalTreatmentMeta,
        numbers: &[f64],
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let malformed = |what: &str| CategoricalTreatmentArtifactError::Malformed(what.to_owned());
        let p = decl.columns();
        match meta.replay.as_str() {
            REPLAY_SUMMARY => {
                if !meta.row_levels.is_empty() {
                    return Err(malformed("a summary replay carries no row labels"));
                }
                if numbers.len() != p * p + p + 1 {
                    return Err(malformed("numbers section length"));
                }
                let counts = meta
                    .counts
                    .iter()
                    .map(|c| {
                        usize::try_from(c.rows)
                            .map(|rows| LevelCount { level: c.level.clone(), rows })
                            .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("rows"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Summary(CategoricalSummary {
                    counts,
                    design: DesignSummary {
                        n_rows: decl.n_rows,
                        first_treatment: 1 + decl.adjustment.len(),
                        gram: numbers[..p * p].to_vec(),
                        xty: numbers[p * p..p * p + p].to_vec(),
                        rss: numbers[p * p + p],
                    },
                }))
            }
            REPLAY_ROWS => {
                if decl.n_rows > CATEGORICAL_HC_ROW_CAP {
                    return Err(CategoricalTreatmentArtifactError::LimitsExceeded("replay rows"));
                }
                if !meta.counts.is_empty() || meta.row_levels.len() != decl.n_rows {
                    return Err(malformed("a rows replay carries one label per row and no counts"));
                }
                let a = decl.adjustment.len();
                if decl.n_rows.checked_mul(1 + a) != Some(numbers.len()) {
                    return Err(malformed("numbers section length"));
                }
                let mut columns = numbers.chunks_exact(decl.n_rows.max(1)).map(<[f64]>::to_vec);
                let outcome = columns.next().unwrap_or_default();
                Ok(Self::Rows {
                    outcome,
                    adjustment: columns.collect(),
                    levels: meta.row_levels.clone(),
                })
            }
            _ => Err(malformed("replay mode")),
        }
    }
}

fn canonical_levels(spec: &CategoricalSpecWire) -> Vec<String> {
    let mut levels = spec.declared_levels.clone();
    if spec.scale == "unordered" {
        levels.sort();
    }
    levels
}

/// The canonical encoding of the stored result section.
fn result_canon(result: &CategoricalResultWire) -> Canon {
    let mut outcome = Canon::new("categorical_treatment_v1.result");
    for c in &result.coefficients {
        outcome.text(&c.name);
        outcome.real(c.estimate);
        outcome.real(c.standard_error);
    }
    for value in &result.covariance {
        outcome.real(*value);
    }
    for c in &result.level_contrasts {
        outcome.text(&c.level);
        outcome.real(c.estimate);
        outcome.real(c.standard_error);
        outcome.real(c.p_holm);
    }
    for c in &result.pairwise {
        outcome.text(&c.from);
        outcome.text(&c.to);
        outcome.real(c.estimate);
        outcome.real(c.standard_error);
        outcome.real(c.p_holm);
    }
    outcome.real(result.omnibus.statistic);
    outcome.real(result.omnibus.p_value);
    if let Some(m) = &result.monotonicity {
        outcome.text(&m.direction);
        outcome.real(m.statistic);
        outcome.real(m.p_value);
        for step in &m.steps {
            outcome.real(step.difference);
            outcome.real(step.standard_error);
        }
    }
    outcome
}

fn identity_of(
    decl: &Decl,
    replay: &Replay,
    numbers: &[f64],
    result: &CategoricalResultWire,
) -> CategoricalTreatmentIdentity {
    let spec = &decl.spec;
    let mut snapshot = Canon::new("categorical_treatment_v1.snapshot");
    snapshot.text(&decl.row_snapshot);
    snapshot.size(decl.n_rows);

    let mut adjustment = Canon::new("categorical_treatment_v1.adjustment_set");
    let mut sorted: Vec<&str> = decl.adjustment.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    adjustment.size(sorted.len());
    for name in sorted {
        adjustment.text(name);
    }

    let mut scale = Canon::new("categorical_treatment_v1.level_scale");
    scale.text(&spec.scale);
    scale.text(&spec.reference);
    let levels = canonical_levels(spec);
    scale.size(levels.len());
    for level in &levels {
        scale.text(level);
    }

    let mut design = Canon::new("categorical_treatment_v1.design");
    design.text(replay.tag());
    design.size(decl.adjustment.len());
    for name in &decl.adjustment {
        design.text(name);
    }
    let counts = replay.counts();
    design.size(counts.len());
    for count in &counts {
        design.text(&count.level);
        design.word(count.rows);
    }
    let labels = replay.row_levels();
    design.size(labels.len());
    for label in &labels {
        design.text(label);
    }
    design.size(numbers.len());
    for value in numbers {
        design.real(*value);
    }

    let mut covariance = Canon::new("categorical_treatment_v1.covariance");
    covariance.text(&spec.covariance);

    let mut family = Canon::new("categorical_treatment_v1.family");
    family.word(spec.min_level_rows);
    family.size(spec.pairwise.len());
    for (from, to) in &spec.pairwise {
        family.text(from);
        family.text(to);
    }
    family.text(spec.monotonicity.as_deref().unwrap_or(""));

    let outcome = result_canon(result);

    let monotonicity_null =
        result.monotonicity.as_ref().map(|m| m.null.clone()).unwrap_or_default();
    let snapshot_id = snapshot.finish();
    let adjustment_set_id = adjustment.finish();
    let level_scale_id = scale.finish();
    let design_id = design.finish();
    let covariance_id = covariance.finish();
    let family_id = family.finish();
    let result_id = outcome.finish();
    let mut whole = Canon::new("categorical_treatment_v1.identity");
    for part in [
        snapshot_id.as_str(),
        adjustment_set_id.as_str(),
        level_scale_id.as_str(),
        design_id.as_str(),
        covariance_id.as_str(),
        family_id.as_str(),
        CATEGORICAL_OMNIBUS_NULL,
        monotonicity_null.as_str(),
        result_id.as_str(),
    ] {
        whole.text(part);
    }
    CategoricalTreatmentIdentity {
        snapshot_id,
        adjustment_set_id,
        level_scale_id,
        design_id,
        covariance_id,
        family_id,
        omnibus_null: CATEGORICAL_OMNIBUS_NULL.to_owned(),
        monotonicity_null,
        result_id,
        digest: whole.finish(),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &CategoricalTreatmentIdentity,
    other: &CategoricalTreatmentIdentity,
) -> Option<&'static str> {
    if stored.snapshot_id != other.snapshot_id {
        Some("row_snapshot")
    } else if stored.adjustment_set_id != other.adjustment_set_id {
        Some("adjustment_set")
    } else if stored.level_scale_id != other.level_scale_id {
        Some("level_scale")
    } else if stored.design_id != other.design_id {
        Some("design")
    } else if stored.covariance_id != other.covariance_id {
        Some("covariance_kind")
    } else if stored.family_id != other.family_id {
        Some("family")
    } else if stored.omnibus_null != other.omnibus_null {
        Some("null")
    } else if stored.monotonicity_null != other.monotonicity_null {
        Some("monotonicity_null")
    } else if stored.result_id != other.result_id {
        Some("result")
    } else if stored.digest != other.digest {
        Some("identity")
    } else {
        None
    }
}

struct Computed {
    fit: CategoricalTreatmentFit,
    result: CategoricalResultWire,
    identity: CategoricalTreatmentIdentity,
    numbers: Vec<f64>,
}

fn check_decl(decl: &Decl) -> Result<(), CategoricalTreatmentArtifactError> {
    if decl.spec.declared_levels.len() > CATEGORICAL_MAX_LEVELS {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("levels"));
    }
    if decl.adjustment.len() > VECTOR_TREATMENT_MAX_ADJUSTMENT {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("adjustment columns"));
    }
    if decl.spec.pairwise.len() > CATEGORICAL_MAX_LEVELS * CATEGORICAL_MAX_LEVELS {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("pairwise contrasts"));
    }
    Ok(())
}

fn compute(decl: &Decl, replay: &Replay) -> Result<Computed, CategoricalTreatmentArtifactError> {
    check_decl(decl)?;
    let spec = spec_of(&decl.spec)?;
    let fit = match replay {
        Replay::Summary(summary) => {
            if summary.design.n_rows != decl.n_rows {
                return Err(CategoricalTreatmentArtifactError::Malformed("row count".into()));
            }
            fit_categorical_from_summary(&spec, summary)?
        }
        Replay::Rows { outcome, adjustment, levels } => {
            let input = CategoricalTreatmentInput {
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
                levels: levels.clone(),
            };
            fit_categorical_treatment(&input, &spec)?
        }
    };
    let result = CategoricalResultWire::from_fit(&fit);
    let numbers = replay.numbers();
    let identity = identity_of(decl, replay, &numbers, &result);
    Ok(Computed { fit, result, identity, numbers })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`CategoricalTreatmentArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &CategoricalTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
    let encode = |e: crate::IoError| CategoricalTreatmentArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(CategoricalTreatmentArtifactError::Encode("missing artifact id".into()));
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
                section_descriptor(NUMBERS_SECTION, "application/octet-stream", numbers),
            ],
            provenance: ProvenanceWire { note: "categorical_treatment_wald_unmeasured".into() },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(NUMBERS_SECTION, numbers.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("artifact bytes"));
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
) -> Result<(CategoricalTreatmentMeta, Vec<u8>), CategoricalTreatmentArtifactError> {
    if bytes.len() > MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != NUMBERS_SECTION
    {
        return Err(CategoricalTreatmentArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    let limit = u64::try_from(MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES).unwrap_or(u64::MAX);
    if declared.is_none_or(|total| total > limit) {
        return Err(CategoricalTreatmentArtifactError::LimitsExceeded("artifact bytes"));
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != CATEGORICAL_TREATMENT_ARTIFACT_VERSION {
        return Err(CategoricalTreatmentArtifactError::UnsupportedVersion {
            version: peek.version,
        });
    }
    let meta: CategoricalTreatmentMeta = from_cbor(meta_section.as_bytes())?;
    let numbers = reader.load_section(NUMBERS_SECTION)?;
    Ok((meta, numbers.as_bytes().to_vec()))
}

fn standing_caveats() -> Vec<String> {
    vec![
        CATEGORICAL_INFERENCE_CAVEAT.to_owned(),
        VECTOR_TREATMENT_ADJUSTMENT_CAVEAT.to_owned(),
        CATEGORICAL_MONOTONICITY_CAVEAT.to_owned(),
    ]
}

/// A produced or consumed categorical artifact: the declared design, the recomputed fit and the
/// identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentArtifact {
    meta: CategoricalTreatmentMeta,
    numbers: Vec<f64>,
    fit: CategoricalTreatmentFit,
}

impl CategoricalTreatmentArtifact {
    /// Fit the declared categorical regime and seal the answer with its embedded design and
    /// identity.
    ///
    /// # Errors
    /// The core refusals (`categorical_treatment.absent_level`, `.sparse_level`,
    /// `.undeclared_level`, `.unknown_reference`, `.monotonicity_requires_ordered`, ...), the
    /// `categorical_treatment.hc_replay_row_cap_exceeded` refusal for a robust covariance above
    /// [`CATEGORICAL_HC_ROW_CAP`] rows, and the format's bounds.
    pub fn seal(
        request: &CategoricalTreatmentRequest,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let spec = spec_of(&request.spec)?;
        let input = CategoricalTreatmentInput {
            outcome: request.outcome.clone(),
            row_snapshot: request.row_snapshot.clone(),
            adjustment: request
                .adjustment
                .iter()
                .map(|c| NamedColumn { name: c.name.clone(), values: c.values.clone() })
                .collect(),
            levels: request.levels.clone(),
        };
        let decl = Decl {
            row_snapshot: request.row_snapshot.clone(),
            n_rows: request.outcome.len(),
            adjustment: request.adjustment.iter().map(|c| c.name.clone()).collect(),
            spec: request.spec.clone(),
        };
        check_decl(&decl)?;
        let replay = if spec.covariance == VectorCovariance::ModelBased {
            Replay::Summary(summarize_categorical_design(&input, &spec)?)
        } else {
            if request.outcome.len() > CATEGORICAL_HC_ROW_CAP {
                return Err(CategoricalTreatmentArtifactError::Refused {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    message: format!(
                        "categorical_treatment.hc_replay_row_cap_exceeded: a robust covariance \
                         replays from the rows and an artifact embeds at most \
                         {CATEGORICAL_HC_ROW_CAP}"
                    ),
                });
            }
            Replay::Rows {
                outcome: request.outcome.clone(),
                adjustment: request.adjustment.iter().map(|c| c.values.clone()).collect(),
                levels: request.levels.clone(),
            }
        };
        let computed = compute(&decl, &replay)?;
        let meta = CategoricalTreatmentMeta {
            version: CATEGORICAL_TREATMENT_ARTIFACT_VERSION,
            feature: CATEGORICAL_TREATMENT_ARTIFACT_FEATURE.to_owned(),
            null: CATEGORICAL_OMNIBUS_NULL.to_owned(),
            monotonicity_null: computed.result.monotonicity.as_ref().map(|m| m.null.clone()),
            inference_claim: CATEGORICAL_INFERENCE_CLAIM.to_owned(),
            calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
            replay: replay.tag().to_owned(),
            row_snapshot: decl.row_snapshot.clone(),
            n_rows: size(decl.n_rows),
            adjustment: decl.adjustment.clone(),
            spec: decl.spec.clone(),
            counts: replay.counts(),
            row_levels: replay.row_levels(),
            identity: computed.identity,
            result: computed.result,
            caveats: standing_caveats(),
        };
        Ok(Self { meta, numbers: computed.numbers, fit: computed.fit })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &CategoricalTreatmentMeta {
        &self.meta
    }

    /// The core fit.
    #[must_use]
    pub fn fit(&self) -> &CategoricalTreatmentFit {
        &self.fit
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &CategoricalTreatmentIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> CategoricalTreatmentReportWire {
        CategoricalTreatmentReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            null: self.meta.null.clone(),
            monotonicity_null: self.meta.monotonicity_null.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            calibration: self.meta.calibration.clone(),
            replay: self.meta.replay.clone(),
            row_snapshot: self.meta.row_snapshot.clone(),
            adjustment: self.meta.adjustment.clone(),
            spec: self.meta.spec.clone(),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        encode_parts(&self.meta, &reals_to_bytes(&self.numbers), artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every identity
    /// field must match it, so a resealed change of level order, reference, scale, family,
    /// covariance kind, design, null or result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity
    /// ([`CategoricalTreatmentArtifactError::IdentityMismatch`]), a stored result that does not
    /// replay, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&CategoricalTreatmentIdentity>,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let (meta, raw) = decode_parts(bytes)?;
        if meta.feature != CATEGORICAL_TREATMENT_ARTIFACT_FEATURE {
            return Err(CategoricalTreatmentArtifactError::UnsupportedSemantics("feature marker"));
        }
        if meta.null != CATEGORICAL_OMNIBUS_NULL {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch { field: "null" });
        }
        if meta.inference_claim != CATEGORICAL_INFERENCE_CLAIM {
            return Err(CategoricalTreatmentArtifactError::UnsupportedSemantics("inference claim"));
        }
        if meta.calibration != CATEGORICAL_TREATMENT_CALIBRATION {
            return Err(CategoricalTreatmentArtifactError::UnsupportedSemantics("calibration"));
        }
        if meta.caveats != standing_caveats() {
            return Err(CategoricalTreatmentArtifactError::UnsupportedSemantics("caveats"));
        }
        let decl = Decl::from_meta(&meta)?;
        check_decl(&decl)?;
        if raw.len() % 8 != 0 {
            return Err(CategoricalTreatmentArtifactError::Malformed(
                "numbers section length".into(),
            ));
        }
        let replay = Replay::from_parts(&decl, &meta, &bytes_to_reals(&raw))?;
        let computed = compute(&decl, &replay)?;
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch { field });
        }
        let stored_null = meta.monotonicity_null.clone().unwrap_or_default();
        if stored_null != computed.identity.monotonicity_null {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch {
                field: "monotonicity_null",
            });
        }
        if meta.result != computed.result {
            return Err(CategoricalTreatmentArtifactError::ResultMismatch(
                "counts, coefficients, covariance, contrasts or tests",
            ));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &computed.identity) {
                return Err(CategoricalTreatmentArtifactError::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, numbers: computed.numbers, fit: computed.fit })
    }
}
