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
//! # Declared rows
//!
//! A request may also declare per-row `sampling_weights`, a per-row `cluster` id (covariance
//! `cluster_robust`) or a Newey-West `serial_lag` (covariance `newey_west`);
//! see [`CategoricalRowsRequest`]. They change the fit and the covariance, so the artifact stores
//! the weighted statistics (`X'WX`, `X'Wy`, `e'We`) and the meat of the weighted-score sandwich
//! of the `robust_summary` replay, plus a `declared_rows` section recording the accounting
//! (rows used, rows of weight zero that were deleted, Kish size and sum of the weights, cluster
//! count, lag). The weights, cluster ids and lag are bound by the data digest and the covariance
//! and design identities; an artifact without them keeps the historical format and digests, and
//! one with them is refused by a reader that does not know the section. The Gaussian posterior
//! section is refused for declared rows: its likelihood is independent homoskedastic errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::categorical_treatment::{
    CATEGORICAL_INFERENCE_CLAIM, CATEGORICAL_OMNIBUS_NULL, CategoricalRobustSummary,
    CategoricalSummary, CategoricalTreatmentFit, CategoricalTreatmentInput,
    CategoricalTreatmentSpec, LevelCount, LevelScale, MonotonicityDirection,
    fit_categorical_from_robust_summary, fit_categorical_from_summary, fit_categorical_treatment,
    summarize_categorical_design, summarize_categorical_design_declared,
    summarize_categorical_design_robust,
};
use antecedent_estimate::categorical_treatment_posterior::{
    CATEGORICAL_POSTERIOR_ESTIMAND, CATEGORICAL_POSTERIOR_LIKELIHOOD, CategoricalPosteriorPrior,
    CategoricalTreatmentPosterior, ContrastPosterior, posterior_categorical_from_summary,
};
use antecedent_estimate::vector_treatment::{
    DesignSummary, NamedColumn, RobustDesignSummary, RowStructure, VectorCovariance,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::artifact_budget::{
    BudgetDetails, BudgetRefusal, BudgetedTreatmentArtifact, admit_export, admit_regression,
};
use crate::artifact_codec::CanonicalSize;
use crate::vector_treatment_artifact::{
    AdjustmentColumnRequest, CoefficientWire, VECTOR_TREATMENT_ADJUSTMENT_CAVEAT,
    VECTOR_TREATMENT_MAX_ADJUSTMENT, WaldWire, bytes_to_reals, reals_to_bytes,
};
use crate::width::lossless_u64;

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

fn budget_refusal((code, detail, explanation): BudgetRefusal) -> CategoricalTreatmentArtifactError {
    CategoricalTreatmentArtifactError::refused(code, detail, explanation)
}

/// The declared design of the categorical regime on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalSpecWire {
    /// Declared level set (the scale, when ordered).
    pub declared_levels: Vec<String>,
    /// Whether the levels are ordered.
    pub scale: LevelScaleWire,
    /// Reference level.
    pub reference: String,
    /// Fewest rows a declared level may have.
    pub min_level_rows: u64,
    /// Requested pairwise contrasts `(from, to)`.
    pub pairwise: Vec<(String, String)>,
    /// Declared monotonicity direction, when a monotonicity test is requested.
    pub monotonicity: Option<MonotonicityDirectionWire>,
    /// Covariance estimator.
    pub covariance: CovarianceKindWire,
}

/// Level scale of a categorical treatment on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelScaleWire {
    /// Levels are ordered as declared.
    Ordered,
    /// Levels are unordered.
    Unordered,
}

impl LevelScaleWire {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Unordered => "unordered",
        }
    }
}

impl From<LevelScale> for LevelScaleWire {
    fn from(scale: LevelScale) -> Self {
        match scale {
            LevelScale::Ordered => Self::Ordered,
            LevelScale::Unordered => Self::Unordered,
        }
    }
}

impl From<LevelScaleWire> for LevelScale {
    fn from(scale: LevelScaleWire) -> Self {
        match scale {
            LevelScaleWire::Ordered => Self::Ordered,
            LevelScaleWire::Unordered => Self::Unordered,
        }
    }
}

impl PartialEq<&str> for LevelScaleWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Direction of a declared monotonicity test on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonotonicityDirectionWire {
    /// Effects do not decrease with the level order.
    NonDecreasing,
    /// Effects do not increase with the level order.
    NonIncreasing,
}

impl MonotonicityDirectionWire {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonDecreasing => "non_decreasing",
            Self::NonIncreasing => "non_increasing",
        }
    }
}

impl From<MonotonicityDirection> for MonotonicityDirectionWire {
    fn from(direction: MonotonicityDirection) -> Self {
        match direction {
            MonotonicityDirection::NonDecreasing => Self::NonDecreasing,
            MonotonicityDirection::NonIncreasing => Self::NonIncreasing,
        }
    }
}

impl From<MonotonicityDirectionWire> for MonotonicityDirection {
    fn from(direction: MonotonicityDirectionWire) -> Self {
        match direction {
            MonotonicityDirectionWire::NonDecreasing => Self::NonDecreasing,
            MonotonicityDirectionWire::NonIncreasing => Self::NonIncreasing,
        }
    }
}

impl PartialEq<&str> for MonotonicityDirectionWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Covariance estimator of a categorical treatment fit on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceKindWire {
    /// Model-based covariance.
    ModelBased,
    /// Heteroskedasticity-consistent HC0.
    Hc0,
    /// HC1.
    Hc1,
    /// HC2.
    Hc2,
    /// HC3.
    Hc3,
    /// One-way cluster-robust; needs a `cluster` id per row.
    ClusterRobust,
    /// Newey-West (Bartlett) HAC; needs a `serial_lag` in the design.
    NeweyWest,
}

impl CovarianceKindWire {
    /// Stable `snake_case` name, the one [`VectorCovariance::as_str`] gives.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelBased => "model_based",
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
            Self::ClusterRobust => "cluster_robust",
            Self::NeweyWest => "newey_west",
        }
    }
}

impl From<VectorCovariance> for CovarianceKindWire {
    fn from(kind: VectorCovariance) -> Self {
        match kind {
            VectorCovariance::ModelBased => Self::ModelBased,
            VectorCovariance::Hc0 => Self::Hc0,
            VectorCovariance::Hc1 => Self::Hc1,
            VectorCovariance::Hc2 => Self::Hc2,
            VectorCovariance::Hc3 => Self::Hc3,
            VectorCovariance::Cluster => Self::ClusterRobust,
            VectorCovariance::NeweyWest => Self::NeweyWest,
        }
    }
}

impl From<CovarianceKindWire> for VectorCovariance {
    fn from(kind: CovarianceKindWire) -> Self {
        match kind {
            CovarianceKindWire::ModelBased => Self::ModelBased,
            CovarianceKindWire::Hc0 => Self::Hc0,
            CovarianceKindWire::Hc1 => Self::Hc1,
            CovarianceKindWire::Hc2 => Self::Hc2,
            CovarianceKindWire::Hc3 => Self::Hc3,
            CovarianceKindWire::ClusterRobust => Self::Cluster,
            CovarianceKindWire::NeweyWest => Self::NeweyWest,
        }
    }
}

impl PartialEq<&str> for CovarianceKindWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
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

/// Declarations of a categorical request beside the plain [`CategoricalTreatmentRequest`]:
/// sampling weights and a cluster id, each one entry per row, and the Newey-West lag.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CategoricalRowsRequest {
    /// Sampling weights (finite, non-negative, at least one positive). Rows of weight zero are
    /// deleted; the rest are normalized to mean one. Need a robust covariance.
    pub sampling_weights: Option<Vec<f64>>,
    /// Cluster id of each row; read by, and required for, the `cluster_robust` covariance.
    pub cluster: Option<Vec<u32>>,
    /// Newey-West (Bartlett) lag for rows in time order at unit spacing; read by, and required
    /// for, the `newey_west` covariance.
    pub serial_lag: Option<u64>,
}

impl CategoricalRowsRequest {
    /// Whether nothing is declared.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.sampling_weights.is_none() && self.cluster.is_none() && self.serial_lag.is_none()
    }
}

/// What the declared rows did, stored beside the unchanged result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredRowsWire {
    /// Format marker of this section.
    pub format: String,
    /// Whether sampling weights were declared.
    pub weights_declared: bool,
    /// Rows used after zero-weight rows were deleted.
    pub rows_used: u64,
    /// Rows of weight zero that were deleted.
    pub zero_weight_rows: u64,
    /// Kish effective size of the weights of the rows used, when weights were declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_effective_sample_size: Option<f64>,
    /// Sum of the declared weights of the rows used before normalization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_sum: Option<f64>,
    /// Clusters among the rows used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clusters: Option<u64>,
    /// Newey-West lag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_lag: Option<u64>,
}

/// The format marker of [`DeclaredRowsWire`].
pub const CATEGORICAL_DECLARED_ROWS_FORMAT: &str = "categorical_declared_rows_v1";

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
    /// Digest of the posterior declaration, absent when there is no posterior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_declaration_id: Option<String>,
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
    /// Direction of the declared monotonicity.
    pub direction: MonotonicityDirectionWire,
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
    /// Whether the levels are ordered.
    pub scale: LevelScaleWire,
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
    pub covariance_kind: CovarianceKindWire,
    /// Always `unmeasured`.
    pub calibration: String,
}

impl CategoricalResultWire {
    fn from_fit(fit: &CategoricalTreatmentFit) -> Result<Self, CategoricalTreatmentArtifactError> {
        let degrees_of_freedom = u32::try_from(fit.omnibus.degrees_of_freedom)
            .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("degrees of freedom"))?;
        Ok(Self {
            reference: fit.reference.clone(),
            scale: fit.scale.into(),
            level_order: fit.level_order.clone(),
            counts: fit
                .counts
                .iter()
                .map(|c| LevelCountWire { level: c.level.clone(), rows: lossless_u64(c.rows) })
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
            family_size: lossless_u64(fit.family_size),
            omnibus: WaldWire {
                statistic: fit.omnibus.statistic,
                degrees_of_freedom,
                p_value: fit.omnibus.p_value,
            },
            monotonicity: fit.monotonicity.as_ref().map(|m| MonotonicityWire {
                direction: m.direction.into(),
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
            covariance_kind: fit.covariance_kind.into(),
            calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
        })
    }
}

/// The declared prior and credible level of the optional posterior section.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalPosteriorDeclaration {
    /// Prior precision (relative to the error variance) of each level coefficient.
    pub level_precision: f64,
    /// Prior precision of the intercept and every adjustment coefficient.
    pub nuisance_precision: f64,
    /// Inverse-Gamma shape of the error variance.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of the error variance.
    pub variance_scale: f64,
    /// Credible level in `(0, 1)` of every interval.
    pub credible_level: f64,
}

/// Student-t posterior of one level or pairwise contrast, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorContrastWire {
    /// Baseline level.
    pub from: String,
    /// Compared level.
    pub to: String,
    /// Student-t location (the posterior mean).
    pub location: f64,
    /// Student-t scale.
    pub scale: f64,
    /// Student-t degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation; absent when the degrees of freedom are at most two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_deviation: Option<f64>,
    /// Equal-tailed credible interval at the declared level.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive.
    pub probability_positive: f64,
}

/// Posterior probability that every adjacent step moves in the declared direction, on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PosteriorMonotonicityWire {
    /// Direction of the declared monotonicity.
    pub direction: MonotonicityDirectionWire,
    /// Per-step marginal posterior probability of lying in the declared direction.
    pub step_probabilities: Vec<f64>,
    /// Lower (Bonferroni) bound of the joint probability.
    pub lower_bound: f64,
    /// Upper (Frechet) bound of the joint probability.
    pub upper_bound: f64,
}

/// The optional posterior section: a Bayesian counterpart of the stored fit under a declared
/// prior, recomputed from the stored design on consumption. Its estimand is the superpopulation
/// dummy coefficient contrast of the stated linear model; it is unlicensed and its calibration is
/// unmeasured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalPosteriorWire {
    /// The declared prior and level.
    pub declaration: CategoricalPosteriorDeclaration,
    /// The likelihood the posterior is exact under.
    pub likelihood: String,
    /// What the contrasts are the contrasts of.
    pub estimand: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `false`: no support license covers the posterior.
    pub licensed: bool,
    /// Posterior Inverse-Gamma shape of the error variance.
    pub variance_shape: f64,
    /// Posterior Inverse-Gamma scale of the error variance.
    pub variance_scale: f64,
    /// Posterior mean of the dummy coefficients, non-reference levels in canonical order.
    pub coefficient_means: Vec<f64>,
    /// Per-level contrasts against the reference.
    pub level_contrasts: Vec<PosteriorContrastWire>,
    /// Requested pairwise contrasts, in request order.
    pub pairwise: Vec<PosteriorContrastWire>,
    /// Declared monotonicity statement, when requested.
    pub monotonicity: Option<PosteriorMonotonicityWire>,
}

fn contrast_wire(c: &ContrastPosterior) -> PosteriorContrastWire {
    PosteriorContrastWire {
        from: c.from.clone(),
        to: c.to.clone(),
        location: c.posterior.location,
        scale: c.posterior.scale,
        degrees_of_freedom: c.posterior.degrees_of_freedom,
        mean: c.mean,
        standard_deviation: c.standard_deviation.is_finite().then_some(c.standard_deviation),
        interval: c.interval,
        probability_positive: c.probability_positive,
    }
}

fn posterior_section(
    declaration: &CategoricalPosteriorDeclaration,
    posterior: &CategoricalTreatmentPosterior,
) -> CategoricalPosteriorWire {
    CategoricalPosteriorWire {
        declaration: *declaration,
        likelihood: CATEGORICAL_POSTERIOR_LIKELIHOOD.to_owned(),
        estimand: CATEGORICAL_POSTERIOR_ESTIMAND.to_owned(),
        calibration: CATEGORICAL_TREATMENT_CALIBRATION.to_owned(),
        licensed: false,
        variance_shape: posterior.variance_shape_and_scale.0,
        variance_scale: posterior.variance_shape_and_scale.1,
        coefficient_means: posterior.coefficient_means.clone(),
        level_contrasts: posterior.level_contrasts.iter().map(contrast_wire).collect(),
        pairwise: posterior.pairwise.iter().map(contrast_wire).collect(),
        monotonicity: posterior.monotonicity.as_ref().map(|m| PosteriorMonotonicityWire {
            direction: m.direction.into(),
            step_probabilities: m.step_probabilities.clone(),
            lower_bound: m.lower_bound,
            upper_bound: m.upper_bound,
        }),
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
    /// The optional posterior section; absent for an artifact sealed without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior: Option<CategoricalPosteriorWire>,
    /// What declared sampling weights, clusters or a lag did; absent for plain rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_rows: Option<DeclaredRowsWire>,
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
    /// The optional posterior section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior: Option<CategoricalPosteriorWire>,
    /// What declared sampling weights, clusters or a lag did; absent for plain rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_rows: Option<DeclaredRowsWire>,
}

fn spec_of(
    wire: &CategoricalSpecWire,
) -> Result<CategoricalTreatmentSpec, CategoricalTreatmentArtifactError> {
    Ok(CategoricalTreatmentSpec {
        declared_levels: wire.declared_levels.clone(),
        scale: wire.scale.into(),
        reference: wire.reference.clone(),
        min_level_rows: usize::try_from(wire.min_level_rows)
            .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("min_level_rows"))?,
        pairwise: wire.pairwise.clone(),
        monotonicity: wire.monotonicity.map(Into::into),
        covariance: wire.covariance.into(),
    })
}

/// The declared lag as the estimator reads it.
fn lag_of(
    rows: &CategoricalRowsRequest,
) -> Result<Option<usize>, CategoricalTreatmentArtifactError> {
    rows.serial_lag
        .map(|lag| {
            usize::try_from(lag)
                .map_err(|_| CategoricalTreatmentArtifactError::LimitsExceeded("serial lag"))
        })
        .transpose()
}

/// The declaration shared by sealing and replay.
struct Decl {
    row_snapshot: String,
    n_rows: usize,
    adjustment: Vec<String>,
    spec: CategoricalSpecWire,
    declared: Option<DeclaredRowsWire>,
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
            declared: meta.declared_rows.clone(),
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
            .map(|c| LevelCountWire { level: c.level.clone(), rows: lossless_u64(c.rows) })
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
                        weighted: decl.declared.as_ref().is_some_and(|d| d.weights_declared),
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
    if spec.scale == LevelScaleWire::Unordered {
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
        outcome.text(m.direction.as_str());
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
    scale.text(spec.scale.as_str());
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

    if let Some(declared) = &decl.declared {
        design.text(&declared.format);
        design.word(u64::from(declared.weights_declared));
        design.word(declared.rows_used);
        design.word(declared.zero_weight_rows);
        design.word(declared.clusters.unwrap_or(0));
    }

    let mut covariance = CanonicalWriter::new("categorical_treatment_v1.covariance");
    covariance.text(spec.covariance.as_str());
    if let Some(lag) = decl.declared.as_ref().and_then(|d| d.serial_lag) {
        covariance.word(lag);
    }

    let mut family = CanonicalWriter::new("categorical_treatment_v1.family");
    family.word(spec.min_level_rows);
    family.size(spec.pairwise.len());
    for (from, to) in &spec.pairwise {
        family.text(from);
        family.text(to);
    }
    family.text(spec.monotonicity.map_or("", MonotonicityDirectionWire::as_str));

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
        posterior_declaration_id: None,
        digest: whole.finish_hex(),
    }
}

// Preserve historical v1 identities when no posterior is attached. A separate domain
// binds the original identity and the posterior declaration for posterior-bearing artifacts.
fn bind_posterior_declaration(
    identity: &mut CategoricalTreatmentIdentity,
    declaration: &CategoricalPosteriorDeclaration,
) {
    let mut posterior = CanonicalWriter::new("categorical_treatment_v1.posterior_declaration");
    posterior.real(declaration.level_precision);
    posterior.real(declaration.nuisance_precision);
    posterior.real(declaration.variance_shape);
    posterior.real(declaration.variance_scale);
    posterior.real(declaration.credible_level);
    let id = posterior.finish_hex();
    let mut whole = CanonicalWriter::new("categorical_treatment_v1.identity_with_posterior");
    whole.text(&identity.digest);
    whole.text(&id);
    identity.digest = whole.finish_hex();
    identity.posterior_declaration_id = Some(id);
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &CategoricalTreatmentIdentity,
    other: &CategoricalTreatmentIdentity,
) -> Option<&'static str> {
    crate::first_identity_difference!(stored, other, {
        snapshot_id => "row_snapshot",
        adjustment_set_id => "adjustment_set",
        level_scale_id => "level_scale",
        design_id => "design",
        covariance_id => "covariance_kind",
        family_id => "family",
        omnibus_null => "null",
        monotonicity_null => "monotonicity_null",
        result_id => "result",
        posterior_declaration_id => "posterior_declaration",
        digest => "identity",
    })
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

/// The posterior section of the design a replay holds.
fn posterior_wire(
    decl: &Decl,
    replay: &Replay,
    declaration: &CategoricalPosteriorDeclaration,
) -> Result<CategoricalPosteriorWire, CategoricalTreatmentArtifactError> {
    let spec = spec_of(&decl.spec)?;
    let summary = match replay {
        Replay::Summary(summary) => summary.clone(),
        Replay::RobustSummary(summary) => CategoricalSummary {
            counts: summary.counts.clone(),
            design: summary.design.design.clone(),
        },
        Replay::Rows { outcome, adjustment, levels } => {
            let input = CategoricalTreatmentInput {
                outcome: Cow::Borrowed(outcome),
                row_snapshot: Cow::Borrowed(&decl.row_snapshot),
                adjustment: decl
                    .adjustment
                    .iter()
                    .zip(adjustment)
                    .map(|(name, values)| NamedColumn::new(name.as_str(), values.as_slice()))
                    .collect(),
                levels: Cow::Borrowed(levels),
            };
            summarize_categorical_design(&input, &spec)?
        }
    };
    let prior = CategoricalPosteriorPrior {
        level_precision: declaration.level_precision,
        nuisance_precision: declaration.nuisance_precision,
        variance_shape: declaration.variance_shape,
        variance_scale: declaration.variance_scale,
    };
    let posterior =
        posterior_categorical_from_summary(&summary, &spec, &prior, declaration.credible_level)?;
    Ok(posterior_section(declaration, &posterior))
}

/// The Gaussian posterior's likelihood is independent homoskedastic errors; a declaration of
/// clusters or serial dependence contradicts it, and sampling weights have no likelihood
/// interpretation (they describe the design that drew the rows). Refused, not approximated.
fn reject_posterior_of_declared_rows(decl: &Decl) -> Result<(), CategoricalTreatmentArtifactError> {
    if decl.declared.is_some() {
        return Err(CategoricalTreatmentArtifactError::refused(
            antecedent_core::reason_code!("route_not_supported"),
            "categorical_treatment.posterior_declared_rows",
            "the conjugate Gaussian posterior assumes independent homoskedastic errors and \
             no sampling design, which declared sampling weights, clusters or serial \
             dependence contradict. Next attempt: read the frequentist weighted, \
             cluster-robust or Newey-West fit, or seal the posterior on plain rows",
        ));
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
                outcome: Cow::Borrowed(outcome),
                row_snapshot: Cow::Borrowed(&decl.row_snapshot),
                adjustment: decl
                    .adjustment
                    .iter()
                    .zip(adjustment)
                    .map(|(name, values)| NamedColumn::new(name.as_str(), values.as_slice()))
                    .collect(),
                levels: Cow::Borrowed(levels),
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
    // A consumer reads at most the byte cap of uncompressed sections, whatever the container
    // compresses to; the caller's budget bounds the container itself.
    let bytes = crate::artifact_codec::encode_sectioned_admitting(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "categorical_treatment_wald_unmeasured",
            max_bytes: MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[crate::OutSection {
            id: NUMBERS_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: numbers,
        }],
        |uncompressed| {
            admit_export(BUDGET, uncompressed, MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES)
                .map_err(budget_refusal)
        },
    )?;
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
    let (meta, parts) = crate::artifact_codec::decode_sectioned::<
        CategoricalTreatmentArtifactError,
        CategoricalTreatmentMeta,
    >(
        ARTIFACT_KIND,
        MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
        META_SECTION,
        &[NUMBERS_SECTION],
        CATEGORICAL_TREATMENT_ARTIFACT_VERSION,
        bytes,
    )?;
    let Ok([payload]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(CategoricalTreatmentArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    };
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
fn data_digest(request: &CategoricalTreatmentRequest, rows: &CategoricalRowsRequest) -> String {
    let mut writer = CanonicalWriter::new("categorical_treatment_v1.data");
    writer.reals(&request.outcome);
    writer.size(request.adjustment.len());
    for column in &request.adjustment {
        writer.text(&column.name);
        writer.reals(&column.values);
    }
    writer.texts(&request.levels);
    // Declarations extend the digest only when made, so plain rows keep their digests.
    if let Some(weights) = &rows.sampling_weights {
        writer.text("sampling_weights");
        writer.reals(weights);
    }
    if let Some(cluster) = &rows.cluster {
        writer.text("cluster");
        writer.size(cluster.len());
        for id in cluster {
            writer.word(u64::from(*id));
        }
    }
    writer.finish_hex()
}

impl BudgetedTreatmentArtifact for CategoricalTreatmentArtifact {
    type Request = CategoricalTreatmentRequest;
    type Identity = CategoricalTreatmentIdentity;
    type Error = CategoricalTreatmentArtifactError;
    const MAX_BYTES: usize = MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES;

    fn seal_with(
        request: &CategoricalTreatmentRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Self::seal_with(request, ctx)
    }

    fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        self.to_bytes_within(artifact_id, export_budget)
    }

    fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&CategoricalTreatmentIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Self::from_bytes_with(bytes, expected, ctx)
    }
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
        <Self as BudgetedTreatmentArtifact>::seal(request)
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
        Self::seal_declared(request, &CategoricalRowsRequest::default(), ctx)
    }

    /// Fit the declared categorical regime under declared sampling weights, clusters and (in the
    /// design) a Newey-West lag, under the computation budget of `ctx`.
    ///
    /// With nothing declared this is [`Self::seal_with`], digests included. Declared rows seal the
    /// weighted statistics and the meat of the weighted-score sandwich of the declared covariance
    /// kind, and the accounting of the declaration. A declaration its covariance kind does not
    /// read (or a kind whose declaration is missing) is refused, never dropped.
    ///
    /// # Errors
    /// The refusals of [`Self::seal_with`]; `sampling_weights_invalid`
    /// (`sampling_weights.invalid`); `option_not_applicable` with
    /// `vector_treatment.weights_need_robust_covariance`; and `vector_treatment.cluster_ids_required`,
    /// `.cluster_unread`, `.too_few_clusters`, `.lag_required`, `.lag_unread`, `.invalid_lag`.
    pub fn seal_declared(
        request: &CategoricalTreatmentRequest,
        rows: &CategoricalRowsRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let spec = spec_of(&request.spec)?;
        let lag = lag_of(rows)?;
        let mut decl = Decl {
            row_snapshot: request.row_snapshot.clone(),
            n_rows: request.outcome.len(),
            adjustment: request.adjustment.iter().map(|c| c.name.clone()).collect(),
            spec: request.spec.clone(),
            declared: None,
        };
        check_decl(&decl)?;
        admit_regression(BUDGET, decl.n_rows, decl.columns(), ctx).map_err(budget_refusal)?;
        let input = CategoricalTreatmentInput {
            outcome: Cow::Borrowed(&request.outcome),
            row_snapshot: Cow::Borrowed(&request.row_snapshot),
            adjustment: request
                .adjustment
                .iter()
                .map(|c| NamedColumn::new(c.name.as_str(), c.values.as_slice()))
                .collect(),
            levels: Cow::Borrowed(&request.levels),
        };
        let structure = RowStructure {
            sampling_weights: rows.sampling_weights.as_deref(),
            clusters: rows.cluster.as_deref(),
            lag,
        };
        let replay = if rows.is_plain() {
            if spec.covariance == VectorCovariance::ModelBased {
                Replay::Summary(summarize_categorical_design(&input, &spec)?)
            } else if matches!(
                spec.covariance,
                VectorCovariance::Cluster | VectorCovariance::NeweyWest
            ) {
                // The kind's declaration is missing: the estimator names which one.
                Replay::RobustSummary(
                    summarize_categorical_design_declared(&input, &spec, &structure)?.0,
                )
            } else {
                Replay::RobustSummary(summarize_categorical_design_robust(&input, &spec)?)
            }
        } else {
            let (summary, record) =
                summarize_categorical_design_declared(&input, &spec, &structure)?;
            decl.n_rows = summary.design.design.n_rows;
            decl.declared = Some(DeclaredRowsWire {
                format: CATEGORICAL_DECLARED_ROWS_FORMAT.to_owned(),
                weights_declared: record.weights.is_some(),
                rows_used: lossless_u64(summary.design.design.n_rows),
                zero_weight_rows: record.weights.map_or(0, |w| w.zero_weight_rows),
                weight_effective_sample_size: record.weights.map(|w| w.effective_sample_size),
                weight_sum: record.weights.map(|w| w.weight_sum),
                clusters: record.clusters.map(lossless_u64),
                serial_lag: record.lag.map(lossless_u64),
            });
            Replay::RobustSummary(summary)
        };
        let digest = data_digest(request, rows);
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
            n_rows: lossless_u64(decl.n_rows),
            data_digest: Some(digest),
            adjustment: decl.adjustment.clone(),
            spec: decl.spec.clone(),
            counts: replay.counts(),
            row_levels: replay.row_levels(),
            identity: computed.identity,
            result: computed.result,
            caveats: standing_caveats(),
            posterior: None,
            declared_rows: decl.declared.clone(),
        };
        Ok(Self { meta, numbers: computed.numbers, fit: computed.fit })
    }

    /// Attach the Bayesian counterpart of the stored fit under a declared prior.
    ///
    /// The posterior is computed from the stored sufficient statistics of the design, not from
    /// rows, and is recomputed bit for bit when the artifact is consumed. Its declaration is
    /// included in the returned artifact's identity. There is no default prior: a prior alone
    /// never identifies a level contrast.
    ///
    /// # Errors
    /// `invalid_argument` for a prior or credible level outside its range, and the stored
    /// design's refusals.
    pub fn with_posterior(
        mut self,
        declaration: &CategoricalPosteriorDeclaration,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        let decl = Decl::from_meta(&self.meta)?;
        reject_posterior_of_declared_rows(&decl)?;
        let replay = Replay::from_parts(&decl, &self.meta, &self.numbers)?;
        let posterior = posterior_wire(&decl, &replay, declaration)?;
        // Reconstruct the base identity so replacing an existing posterior never compounds it.
        let mut identity = identity_of(
            &decl,
            &replay,
            &self.numbers,
            self.meta.data_digest.as_deref(),
            &self.meta.result,
        );
        bind_posterior_declaration(&mut identity, declaration);
        self.meta.posterior = Some(posterior);
        self.meta.identity = identity;
        Ok(self)
    }

    /// The posterior section, when one was attached.
    #[must_use]
    pub fn posterior(&self) -> Option<&CategoricalPosteriorWire> {
        self.meta.posterior.as_ref()
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
            posterior: self.meta.posterior.clone(),
            declared_rows: self.meta.declared_rows.clone(),
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
        self.verify_data_declared(request, &CategoricalRowsRequest::default())
    }

    /// [`Self::verify_data`] for rows sealed with declared sampling weights or clusters.
    ///
    /// # Errors
    /// As [`Self::verify_data`]; the weights and cluster ids are part of the digest.
    pub fn verify_data_declared(
        &self,
        request: &CategoricalTreatmentRequest,
        rows: &CategoricalRowsRequest,
    ) -> Result<(), CategoricalTreatmentArtifactError> {
        match &self.meta.data_digest {
            Some(stored) if *stored == data_digest(request, rows) => Ok(()),
            Some(_) => Err(CategoricalTreatmentArtifactError::IdentityMismatch { field: "data" }),
            None => Err(CategoricalTreatmentArtifactError::refused(
                antecedent_core::reason_code!("route_not_supported"),
                "categorical_treatment.data_not_bound",
                "this artifact embeds its rows and records no data digest",
            )),
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
        <Self as BudgetedTreatmentArtifact>::to_bytes(self, artifact_id)
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
        <Self as BudgetedTreatmentArtifact>::from_bytes(bytes, expected)
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
        crate::artifact_codec::require_semantics::<CategoricalTreatmentArtifactError>(
            meta.feature == CATEGORICAL_TREATMENT_ARTIFACT_FEATURE,
            "feature marker",
        )?;
        if meta.null != CATEGORICAL_OMNIBUS_NULL {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch { field: "null" });
        }
        crate::artifact_codec::require_markers::<CategoricalTreatmentArtifactError>(&[
            (meta.inference_claim == CATEGORICAL_INFERENCE_CLAIM, "inference claim"),
            (meta.calibration == CATEGORICAL_TREATMENT_CALIBRATION, "calibration"),
            (meta.caveats == standing_caveats(), "caveats"),
        ])?;
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
        let mut computed = compute(&decl, &replay, meta.data_digest.as_deref())?;
        if let Some(posterior) = &meta.posterior {
            if meta.identity.posterior_declaration_id.is_none() {
                return Err(CategoricalTreatmentArtifactError::IdentityMismatch {
                    field: "posterior_declaration",
                });
            }
            bind_posterior_declaration(&mut computed.identity, &posterior.declaration);
        }
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch { field });
        }
        let stored_null = meta.monotonicity_null.clone().unwrap_or_default();
        if stored_null != computed.identity.monotonicity_null {
            return Err(CategoricalTreatmentArtifactError::IdentityMismatch {
                field: "monotonicity_null",
            });
        }
        // Replay contract: the consumer re-runs the same deterministic code on the stored
        // numbers, so the result must agree exactly (derived `PartialEq`: bitwise except that
        // `-0.0 == 0.0`); a tolerance would admit a result the producer never computed.
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
        if let Some(stored) = &meta.posterior {
            reject_posterior_of_declared_rows(&decl)?;
            let replayed = posterior_wire(&decl, &replay, &stored.declaration)?;
            if *stored != replayed {
                return Err(CategoricalTreatmentArtifactError::ResultMismatch(
                    "posterior contrasts, moments or monotonicity bounds",
                ));
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
        CategoricalTreatmentIdentity, CategoricalTreatmentMeta, CategoricalTreatmentRequest,
        CovarianceKindWire, Decl, LevelScaleWire, MAX_CATEGORICAL_TREATMENT_ARTIFACT_BYTES,
        REPLAY_ROWS, Replay, compute, encode_parts_impl, reals_to_bytes,
    };
    use crate::vector_treatment_artifact::WaldWire;

    fn spec() -> CategoricalSpecWire {
        CategoricalSpecWire {
            declared_levels: vec!["a".into(), "b".into()],
            scale: LevelScaleWire::Unordered,
            reference: "a".into(),
            min_level_rows: 1,
            pairwise: Vec::new(),
            monotonicity: None,
            covariance: CovarianceKindWire::Hc1,
        }
    }

    fn decl(n_rows: usize) -> Decl {
        Decl {
            row_snapshot: String::new(),
            n_rows,
            adjustment: vec!["z".into()],
            spec: spec(),
            declared: None,
        }
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
                posterior_declaration_id: None,
                digest: String::new(),
            },
            result: CategoricalResultWire {
                reference: String::new(),
                scale: LevelScaleWire::Unordered,
                level_order: Vec::new(),
                counts: Vec::new(),
                coefficients: Vec::new(),
                covariance: Vec::new(),
                level_contrasts: Vec::new(),
                pairwise: Vec::new(),
                family_size: 0,
                omnibus: WaldWire { statistic: 0.0, degrees_of_freedom: 0, p_value: 0.0 },
                monotonicity: None,
                covariance_kind: CovarianceKindWire::Hc1,
                calibration: String::new(),
            },
            caveats: Vec::new(),
            posterior: None,
            declared_rows: None,
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

    fn robust_request(n: usize, covariance: CovarianceKindWire) -> CategoricalTreatmentRequest {
        CategoricalTreatmentRequest {
            outcome: column(n, 1),
            row_snapshot: "snap".into(),
            adjustment: vec![AdjustmentColumnRequest { name: "z".into(), values: column(n, 2) }],
            levels: (0..n).map(|i| ["a", "b"][i % 2].to_owned()).collect(),
            spec: CategoricalSpecWire { covariance, ..spec() },
        }
    }

    /// An artifact sealed before summaries existed embedded the rows of a robust fit. Its
    /// consumer is kept, and it reaches exactly the fit the summary artifact does.
    #[test]
    fn a_legacy_rows_artifact_replays_to_the_summary_fit_and_records_no_digest() {
        for covariance in [
            CovarianceKindWire::Hc0,
            CovarianceKindWire::Hc1,
            CovarianceKindWire::Hc2,
            CovarianceKindWire::Hc3,
        ] {
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
            assert_eq!(legacy.fit(), summary.fit(), "{covariance:?}");
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
