//! Categorical-treatment artifact (`categorical_treatment_v1`).
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
//!   * `robust_summary` (HC0 to HC3): the same numbers followed by the `p x p` meat
//!     `sum_i w_i e_i^2 x_i x_i'` of the declared estimator. HC0 and HC1 need only `X'X` and the
//!     residuals; HC2 and HC3 need each leverage `h_i = x_i' (X'X)^-1 x_i`, which one pass over
//!     the rows computes once `X'X` is known, so the meat is fixed when sealing and the consumer
//!     recomputes the sandwich from these numbers alone, bit-identically;
//!   * `rows` (legacy robust artifacts): the outcome and adjustment columns, with the row level
//!     labels in the metadata, replayed through the core fit. It is read but no longer written;
//!     replay above [`CATEGORICAL_HC_ROW_CAP`] rows is refused.
//!
//! The summaries carry no row, so they are bound to the rows they came from by a digest
//! (`data_digest`, part of the design identity): [`CategoricalTreatmentArtifact::verify_data`]
//! checks rows a caller still holds against it.
//!
//! # Budgets
//!
//! The fit has no row ceiling. The **computation** budget is the [`ExecutionContext`] given to
//! [`CategoricalTreatmentArtifact::seal_with`] (its hard memory limit and cancellation); the
//! **export** budget is the artifact byte cap ([`CategoricalTreatmentArtifact::to_bytes_within`]),
//! which a summary artifact never approaches because its size does not grow with the rows; the
//! **replay** budget of a summary is the same context, as its cost does not grow with the rows
//! either.
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

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::categorical_treatment::{
    CATEGORICAL_INFERENCE_CLAIM, CATEGORICAL_OMNIBUS_NULL, CategoricalRobustSummary,
    CategoricalSummary, CategoricalTreatmentFit, CategoricalTreatmentInput,
    CategoricalTreatmentSpec, LevelCount, LevelScale, MonotonicityDirection,
    fit_categorical_from_robust_summary, fit_categorical_from_summary, fit_categorical_treatment,
    summarize_categorical_design, summarize_categorical_design_robust,
};
use antecedent_estimate::vector_treatment::{
    DesignSummary, NamedColumn, RobustDesignSummary, VectorCovariance,
};
use serde::{Deserialize, Serialize};

use crate::artifact_budget::{
    BudgetDetails, BudgetRefusal, admit_export, admit_regression, unbounded_context,
};
use crate::artifact_codec::CanonicalSize;
use crate::convert::to_cbor;
use crate::vector_treatment_artifact::{
    AdjustmentColumnRequest, CoefficientWire, VECTOR_TREATMENT_ADJUSTMENT_CAVEAT,
    VECTOR_TREATMENT_MAX_ADJUSTMENT, WaldWire, bytes_to_reals, reals_to_bytes,
};

/// The artifact major version this reader writes and accepts.
pub const CATEGORICAL_TREATMENT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const CATEGORICAL_TREATMENT_ARTIFACT_FEATURE: &str = "categorical_treatment_v1";
/// Calibration coordinate: Type I error, coverage and power were not measured.
pub const CATEGORICAL_TREATMENT_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy: the largest export budget, enforced on consumption.
pub const MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most rows a legacy robust-covariance `rows` replay artifact may carry. Sealing no longer
/// embeds rows, so this bounds only the consumption of artifacts that already do.
pub const CATEGORICAL_HC_ROW_CAP: usize = 4096;
/// Most declared levels an artifact may carry.
pub const CATEGORICAL_MAX_LEVELS: usize = 256;
/// Standing caveat: inference is asymptotic and its calibration is unmeasured.
pub const CATEGORICAL_INFERENCE_CAVEAT: &str = "every p-value is an asymptotic Wald, Holm or chi-square value whose calibration is unmeasured; no coverage claim is made and no interval is produced";
/// Standing caveat of the monotonicity test.
pub const CATEGORICAL_MONOTONICITY_CAVEAT: &str = "the monotonicity p-value is a conservative one-sided union-intersection (Bonferroni) bound; rejection is evidence against monotonicity in the declared direction, failing to reject does not prove it, and the test's Type I error and power are unmeasured";

const ARTIFACT_KIND: &str = crate::artifact_kinds::CATEGORICAL_TREATMENT;
const META_SECTION: &str = "categorical_treatment_meta";
const NUMBERS_SECTION: &str = "categorical_treatment_numbers";
const REPLAY_SUMMARY: &str = "summary";
const REPLAY_ROBUST_SUMMARY: &str = "robust_summary";
const REPLAY_ROWS: &str = "rows";
const BUDGET: BudgetDetails = BudgetDetails {
    cancelled: (
        antecedent_core::reason_code!("cancelled_no_claim"),
        "categorical_treatment.cancelled",
    ),
    memory: "categorical_treatment.memory_budget_exceeded",
    export: "categorical_treatment.export_budget_exceeded",
};

crate::sectioned_artifact_error!(
    /// Why a categorical-treatment artifact was refused or could not be produced.
    CategoricalTreatmentArtifactError,
    "categorical treatment",
    "categorical_treatment",
    "result",
    false
);

fn budget_refusal((code, message): BudgetRefusal) -> CategoricalTreatmentArtifactError {
    CategoricalTreatmentArtifactError::Refused { code, message }
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
    crate::width::lossless_u64(value)
}

impl CategoricalResultWire {
    fn from_fit(fit: &CategoricalTreatmentFit) -> Result<Self, CategoricalTreatmentArtifactError> {
        let degrees_of_freedom = u32::try_from(fit.omnibus.degrees_of_freedom)
            .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("degrees of freedom"))?;
        Ok(Self {
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
                degrees_of_freedom,
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
        })
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
    /// `summary`, `robust_summary` or (legacy) `rows`.
    pub replay: String,
    /// Identity of the row snapshot.
    pub row_snapshot: String,
    /// Rows of the snapshot.
    pub n_rows: u64,
    /// Digest of the rows a summary was computed from; absent for a `rows` replay, which embeds
    /// them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_digest: Option<String>,
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
    /// Digest of the rows a summary replay was computed from, when it records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_digest: Option<String>,
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
    RobustSummary(CategoricalRobustSummary),
    Rows { outcome: Vec<f64>, adjustment: Vec<Vec<f64>>, levels: Vec<String> },
}

impl Replay {
    const fn tag(&self) -> &'static str {
        match self {
            Self::Summary(_) => REPLAY_SUMMARY,
            Self::RobustSummary(_) => REPLAY_ROBUST_SUMMARY,
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
            Self::RobustSummary(s) => s
                .design
                .design
                .gram
                .iter()
                .chain(&s.design.design.xty)
                .copied()
                .chain(std::iter::once(s.design.design.rss))
                .chain(s.design.meat.iter().copied())
                .collect(),
            Self::Rows { outcome, adjustment, .. } => {
                outcome.iter().chain(adjustment.iter().flatten()).copied().collect()
            }
        }
    }

    fn counts(&self) -> Vec<LevelCountWire> {
        let counts = match self {
            Self::Summary(s) => &s.counts,
            Self::RobustSummary(s) => &s.counts,
            Self::Rows { .. } => return Vec::new(),
        };
        counts
            .iter()
            .map(|c| LevelCountWire { level: c.level.clone(), rows: size(c.rows) })
            .collect()
    }

    fn row_levels(&self) -> Vec<String> {
        match self {
            Self::Summary(_) | Self::RobustSummary(_) => Vec::new(),
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
            REPLAY_SUMMARY | REPLAY_ROBUST_SUMMARY => {
                let robust = meta.replay == REPLAY_ROBUST_SUMMARY;
                if !meta.row_levels.is_empty() {
                    return Err(malformed("a summary replay carries no row labels"));
                }
                let expected_len = if robust { 2 * p * p + p + 1 } else { p * p + p + 1 };
                if numbers.len() != expected_len {
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
                let mut rest = numbers;
                let design = DesignSummary {
                    n_rows: decl.n_rows,
                    first_treatment: 1 + decl.adjustment.len(),
                    gram: crate::width::split_reals(&mut rest, p * p).to_vec(),
                    xty: crate::width::split_reals(&mut rest, p).to_vec(),
                    rss: crate::width::split_reals(&mut rest, 1)[0],
                };
                if robust {
                    let meat = crate::width::split_reals(&mut rest, p * p).to_vec();
                    Ok(Self::RobustSummary(CategoricalRobustSummary {
                        counts,
                        design: RobustDesignSummary { design, meat },
                    }))
                } else {
                    Ok(Self::Summary(CategoricalSummary { counts, design }))
                }
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
                let n = decl.n_rows;
                // A zero row count is an empty payload, so there are no columns.
                let mut columns = if n == 0 {
                    Vec::new()
                } else {
                    let mut rest = numbers;
                    crate::width::take_blocks(&mut rest, n, numbers.len() / n)
                }
                .into_iter();
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
fn result_canon(result: &CategoricalResultWire) -> CanonicalWriter {
    let mut outcome = CanonicalWriter::new("categorical_treatment_v1.result");
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
    data_digest: Option<&str>,
    result: &CategoricalResultWire,
) -> CategoricalTreatmentIdentity {
    let spec = &decl.spec;
    let mut snapshot = CanonicalWriter::new("categorical_treatment_v1.snapshot");
    snapshot.text(&decl.row_snapshot);
    snapshot.size(decl.n_rows);

    let mut adjustment = CanonicalWriter::new("categorical_treatment_v1.adjustment_set");
    let mut sorted: Vec<&str> = decl.adjustment.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    adjustment.size(sorted.len());
    for name in sorted {
        adjustment.text(name);
    }

    let mut scale = CanonicalWriter::new("categorical_treatment_v1.level_scale");
    scale.text(&spec.scale);
    scale.text(&spec.reference);
    let levels = canonical_levels(spec);
    scale.size(levels.len());
    for level in &levels {
        scale.text(level);
    }

    let mut design = CanonicalWriter::new("categorical_treatment_v1.design");
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
    if let Some(digest) = data_digest {
        design.text(digest);
    }

    let mut covariance = CanonicalWriter::new("categorical_treatment_v1.covariance");
    covariance.text(&spec.covariance);

    let mut family = CanonicalWriter::new("categorical_treatment_v1.family");
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
    let snapshot_id = snapshot.finish_hex();
    let adjustment_set_id = adjustment.finish_hex();
    let level_scale_id = scale.finish_hex();
    let design_id = design.finish_hex();
    let covariance_id = covariance.finish_hex();
    let family_id = family.finish_hex();
    let result_id = outcome.finish_hex();
    let mut whole = CanonicalWriter::new("categorical_treatment_v1.identity");
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
        digest: whole.finish_hex(),
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

fn compute(
    decl: &Decl,
    replay: &Replay,
    data_digest: Option<&str>,
) -> Result<Computed, CategoricalTreatmentArtifactError> {
    check_decl(decl)?;
    let spec = spec_of(&decl.spec)?;
    let fit = match replay {
        Replay::Summary(summary) => {
            if summary.design.n_rows != decl.n_rows {
                return Err(CategoricalTreatmentArtifactError::Malformed("row count".into()));
            }
            fit_categorical_from_summary(&spec, summary)?
        }
        Replay::RobustSummary(summary) => {
            if summary.design.design.n_rows != decl.n_rows {
                return Err(CategoricalTreatmentArtifactError::Malformed("row count".into()));
            }
            fit_categorical_from_robust_summary(&spec, summary)?
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
    let result = CategoricalResultWire::from_fit(&fit)?;
    let numbers = replay.numbers();
    let identity = identity_of(decl, replay, &numbers, data_digest, &result);
    Ok(Computed { fit, result, identity, numbers })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`CategoricalTreatmentArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &CategoricalTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
    encode_parts_impl(meta, numbers, artifact_id, MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES)
}

fn encode_parts_impl(
    meta: &CategoricalTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
    export_budget: usize,
) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(CategoricalTreatmentArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta)
        .map_err(|error| CategoricalTreatmentArtifactError::Encode(error.to_string()))?;
    // A consumer reads at most the byte cap of uncompressed sections, whatever the container
    // compresses to; the caller's budget bounds the container itself.
    admit_export(
        BUDGET,
        meta_bytes.len().saturating_add(numbers.len()),
        MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
    )
    .map_err(budget_refusal)?;
    let bytes = crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "categorical_treatment_wald_unmeasured",
            max_bytes: MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
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
    .map_err(|error| crate::map_artifact_check!(error, CategoricalTreatmentArtifactError))?;
    admit_export(BUDGET, bytes.len(), export_budget.min(MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES))
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
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(CategoricalTreatmentMeta, Vec<u8>), CategoricalTreatmentArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(CategoricalTreatmentMeta, Vec<u8>), CategoricalTreatmentArtifactError> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
        &[META_SECTION, NUMBERS_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, CategoricalTreatmentArtifactError))?;
    let Ok([meta_bytes, payload]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(CategoricalTreatmentArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    };
    let meta = crate::decode_with_version(&meta_bytes, |version| {
        if version == CATEGORICAL_TREATMENT_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(CategoricalTreatmentArtifactError::UnsupportedVersion { version })
        }
    })?;
    Ok((meta, payload))
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

/// Digest of the rows a request fits: the outcome, every adjustment column with its name and the
/// level label of each row. A summary replay holds this instead of the rows.
fn data_digest(request: &CategoricalTreatmentRequest) -> String {
    let mut writer = CanonicalWriter::new("categorical_treatment_v1.data");
    writer.reals(&request.outcome);
    writer.size(request.adjustment.len());
    for column in &request.adjustment {
        writer.text(&column.name);
        writer.reals(&column.values);
    }
    writer.texts(&request.levels);
    writer.finish_hex()
}

impl CategoricalTreatmentArtifact {
    /// Fit the declared categorical regime and seal the answer with its embedded design and
    /// identity, with no memory limit; see [`Self::seal_with`].
    ///
    /// # Errors
    /// The core refusals (`categorical_treatment.absent_level`, `.sparse_level`,
    /// `.undeclared_level`, `.unknown_reference`, `.monotonicity_requires_ordered`, ...) and the
    /// format's bounds.
    pub fn seal(
        request: &CategoricalTreatmentRequest,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Self::seal_with(request, &unbounded_context())
    }

    /// Fit the declared categorical regime under the computation budget of `ctx`.
    ///
    /// The row count is not limited here: the planned working set must fit the context's hard
    /// memory limit and the context must not be cancelled. Every covariance kind seals a
    /// summary, so the artifact stays small however many rows were fitted.
    ///
    /// # Errors
    /// The refusals of [`Self::seal`], `categorical_treatment.memory_budget_exceeded` and
    /// `categorical_treatment.cancelled`.
    pub fn seal_with(
        request: &CategoricalTreatmentRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let spec = spec_of(&request.spec)?;
        let decl = Decl {
            row_snapshot: request.row_snapshot.clone(),
            n_rows: request.outcome.len(),
            adjustment: request.adjustment.iter().map(|c| c.name.clone()).collect(),
            spec: request.spec.clone(),
        };
        check_decl(&decl)?;
        admit_regression(BUDGET, decl.n_rows, decl.columns(), ctx).map_err(budget_refusal)?;
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
        let replay = if spec.covariance == VectorCovariance::ModelBased {
            Replay::Summary(summarize_categorical_design(&input, &spec)?)
        } else {
            Replay::RobustSummary(summarize_categorical_design_robust(&input, &spec)?)
        };
        let digest = data_digest(request);
        let computed = compute(&decl, &replay, Some(&digest))?;
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
            data_digest: Some(digest),
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
            data_digest: self.meta.data_digest.clone(),
            adjustment: self.meta.adjustment.clone(),
            spec: self.meta.spec.clone(),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Check rows a caller still holds against the digest this artifact's summary was computed
    /// from.
    ///
    /// # Errors
    /// [`CategoricalTreatmentArtifactError::IdentityMismatch`] (`data`) when the rows differ, and
    /// `categorical_treatment.data_not_bound` for an artifact that records no digest (a legacy
    /// `rows` replay, which embeds its rows instead).
    pub fn verify_data(
        &self,
        request: &CategoricalTreatmentRequest,
    ) -> Result<(), CategoricalTreatmentArtifactError> {
        match &self.meta.data_digest {
            Some(stored) if *stored == data_digest(request) => Ok(()),
            Some(_) => Err(CategoricalTreatmentArtifactError::IdentityMismatch { field: "data" }),
            None => Err(CategoricalTreatmentArtifactError::Refused {
                code: antecedent_core::reason_code!("route_not_supported"),
                message: "categorical_treatment.data_not_bound: this artifact embeds its rows \
                          and records no data digest"
                    .to_owned(),
            }),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or `categorical_treatment.export_budget_exceeded` when
    /// the artifact would exceed [`MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES`].
    pub fn to_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        self.to_bytes_within(artifact_id, MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES)
    }

    /// Serialize when the whole artifact fits `export_budget` bytes (at most
    /// [`MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES`]).
    ///
    /// # Errors
    /// An empty id, an encoding failure or `categorical_treatment.export_budget_exceeded`, which
    /// names the bytes needed and the budget. The fit is untouched by the refusal.
    pub fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        encode_parts_impl(&self.meta, &reals_to_bytes(&self.numbers), artifact_id, export_budget)
    }

    /// Consume an artifact by recomputation, with no memory limit; see
    /// [`Self::from_bytes_with`].
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
        Self::from_bytes_with(bytes, expected, &unbounded_context())
    }

    /// Consume an artifact by recomputation within the replay budget of `ctx`.
    ///
    /// # Errors
    /// The refusals of [`Self::from_bytes`], `categorical_treatment.memory_budget_exceeded` and
    /// `categorical_treatment.cancelled`.
    pub fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&CategoricalTreatmentIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let (meta, raw) = decode_parts_impl(bytes)?;
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
        let replayed_rows = if matches!(replay, Replay::Rows { .. }) { decl.n_rows } else { 0 };
        admit_regression(BUDGET, replayed_rows, decl.columns(), ctx).map_err(budget_refusal)?;
        let computed = compute(&decl, &replay, meta.data_digest.as_deref())?;
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

#[cfg(test)]
mod number_section_tests {
    use super::{
        AdjustmentColumnRequest, CATEGORICAL_HC_ROW_CAP, CategoricalResultWire,
        CategoricalSpecWire, CategoricalTreatmentArtifact, CategoricalTreatmentArtifactError,
        CategoricalTreatmentIdentity, CategoricalTreatmentMeta, CategoricalTreatmentRequest, Decl,
        MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES, REPLAY_ROWS, Replay, compute, encode_parts_impl,
        reals_to_bytes,
    };
    use crate::vector_treatment_artifact::WaldWire;

    fn spec() -> CategoricalSpecWire {
        CategoricalSpecWire {
            declared_levels: vec!["a".into(), "b".into()],
            scale: "unordered".into(),
            reference: "a".into(),
            min_level_rows: 1,
            pairwise: Vec::new(),
            monotonicity: None,
            covariance: "hc1".into(),
        }
    }

    fn decl(n_rows: usize) -> Decl {
        Decl { row_snapshot: String::new(), n_rows, adjustment: vec!["z".into()], spec: spec() }
    }

    fn meta(n_rows: u64, levels: Vec<String>) -> CategoricalTreatmentMeta {
        CategoricalTreatmentMeta {
            version: 1,
            feature: String::new(),
            null: String::new(),
            monotonicity_null: None,
            inference_claim: String::new(),
            calibration: String::new(),
            replay: REPLAY_ROWS.into(),
            row_snapshot: String::new(),
            n_rows,
            data_digest: None,
            adjustment: vec!["z".into()],
            spec: spec(),
            counts: Vec::new(),
            row_levels: levels,
            identity: CategoricalTreatmentIdentity {
                snapshot_id: String::new(),
                adjustment_set_id: String::new(),
                level_scale_id: String::new(),
                design_id: String::new(),
                covariance_id: String::new(),
                family_id: String::new(),
                omnibus_null: String::new(),
                monotonicity_null: String::new(),
                result_id: String::new(),
                digest: String::new(),
            },
            result: CategoricalResultWire {
                reference: String::new(),
                scale: String::new(),
                level_order: Vec::new(),
                counts: Vec::new(),
                coefficients: Vec::new(),
                covariance: Vec::new(),
                level_contrasts: Vec::new(),
                pairwise: Vec::new(),
                family_size: 0,
                omnibus: WaldWire { statistic: 0.0, degrees_of_freedom: 0, p_value: 0.0 },
                monotonicity: None,
                covariance_kind: String::new(),
                calibration: String::new(),
            },
            caveats: Vec::new(),
        }
    }

    fn bits(sample: &[f64]) -> Vec<u64> {
        sample.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn row_replay_round_trips_signed_zero_and_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let replay = Replay::Rows {
            outcome: vec![-0.0, 1.0],
            adjustment: vec![vec![nan, f64::MIN_POSITIVE]],
            levels: vec!["a".into(), "b".into()],
        };
        let numbers = replay.numbers();
        let bytes = crate::width::f64s_to_le_bytes(&numbers);
        let direct: Vec<u8> = numbers.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, direct);
        let decoded = crate::width::f64s_from_le_bytes(&bytes);
        let back =
            Replay::from_parts(&decl(2), &meta(2, vec!["a".into(), "b".into()]), &decoded).unwrap();
        assert_eq!(bits(&back.numbers()), bits(&numbers));
        assert_eq!(crate::width::f64s_to_le_bytes(&back.numbers()), bytes);
    }

    #[test]
    fn zero_row_replay_has_no_columns() {
        let back = Replay::from_parts(&decl(0), &meta(0, Vec::new()), &[]).unwrap();
        let Replay::Rows { outcome, adjustment, .. } = back else {
            panic!("rows replay");
        };
        assert!(outcome.is_empty());
        assert!(adjustment.is_empty());
    }

    /// Deterministic pseudo-random column in `[-1, 1)` (a 64-bit LCG).
    fn column(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    fn robust_request(n: usize, covariance: &str) -> CategoricalTreatmentRequest {
        CategoricalTreatmentRequest {
            outcome: column(n, 1),
            row_snapshot: "snap".into(),
            adjustment: vec![AdjustmentColumnRequest { name: "z".into(), values: column(n, 2) }],
            levels: (0..n).map(|i| ["a", "b"][i % 2].to_owned()).collect(),
            spec: CategoricalSpecWire { covariance: covariance.into(), ..spec() },
        }
    }

    /// An artifact sealed before summaries existed embedded the rows of a robust fit. Its
    /// consumer is kept, and it reaches exactly the fit the summary artifact does.
    #[test]
    fn a_legacy_rows_artifact_replays_to_the_summary_fit_and_records_no_digest() {
        for covariance in ["hc0", "hc1", "hc2", "hc3"] {
            let request = robust_request(60, covariance);
            let summary = CategoricalTreatmentArtifact::seal(&request).unwrap();
            assert_eq!(summary.meta().replay, "robust_summary");
            let decl = Decl::from_meta(summary.meta()).unwrap();
            let rows = Replay::Rows {
                outcome: request.outcome.clone(),
                adjustment: request.adjustment.iter().map(|c| c.values.clone()).collect(),
                levels: request.levels.clone(),
            };
            let computed = compute(&decl, &rows, None).unwrap();
            let mut meta = summary.meta().clone();
            meta.replay = REPLAY_ROWS.into();
            meta.data_digest = None;
            meta.counts = rows.counts();
            meta.row_levels = rows.row_levels();
            meta.identity = computed.identity;
            let bytes = encode_parts_impl(
                &meta,
                &reals_to_bytes(&computed.numbers),
                "legacy",
                MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
            )
            .unwrap();
            let legacy =
                CategoricalTreatmentArtifact::from_bytes(&bytes, Some(&meta.identity)).unwrap();
            assert_eq!(legacy.meta().replay, REPLAY_ROWS);
            assert_eq!(legacy.fit(), summary.fit(), "{covariance}");
            // Embedded rows need no digest; asking to verify against one says so.
            let error = legacy.verify_data(&request).unwrap_err();
            assert_eq!(error.refusal().unwrap().1, "categorical_treatment.data_not_bound");
        }
    }

    #[test]
    fn a_legacy_rows_replay_above_its_cap_is_refused_on_consumption() {
        let decl = decl(CATEGORICAL_HC_ROW_CAP + 1);
        assert!(matches!(
            Replay::from_parts(&decl, &meta(CATEGORICAL_HC_ROW_CAP as u64 + 1, Vec::new()), &[]),
            Err(CategoricalTreatmentArtifactError::LimitsExceeded("replay rows"))
        ));
    }
}
