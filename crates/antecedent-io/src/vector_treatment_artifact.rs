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
//!   * `robust_summary` (HC0 to HC3): the same numbers followed by the `p x p` meat
//!     `sum_i w_i e_i^2 x_i x_i'` of the declared estimator. HC0 and HC1 need only `X'X` and the
//!     residuals; HC2 and HC3 need each leverage `h_i = x_i' (X'X)^-1 x_i`, which one pass over
//!     the rows computes once `X'X` is known, so the meat is fixed when sealing and the consumer
//!     recomputes the sandwich `(X'X)^-1 meat (X'X)^-1` from these numbers alone, bit-identically;
//!   * `rows` (legacy robust artifacts): the outcome, the adjustment columns and the treatment
//!     columns, replayed through the core fit. It is read but no longer written; the declared row
//!     count must match the stored payload exactly, so the artifact byte budget is the only bound
//!     on the rows.
//!
//! The summaries carry no row, so they are bound to the rows they came from by a digest
//! (`data_digest`, part of the design identity): [`VectorTreatmentArtifact::verify_data`] checks
//! rows a caller still holds against it.
//!
//! # Budgets
//!
//! The fit has no row ceiling. The **computation** budget is the [`ExecutionContext`] given to
//! [`VectorTreatmentArtifact::seal_with`] (its hard memory limit and cancellation); the
//! **export** budget is the artifact byte cap ([`VectorTreatmentArtifact::to_bytes_within`]),
//! which a summary artifact never approaches because its size does not grow with the rows; the
//! **replay** budget of a summary is the same context, as its cost does not grow with the rows
//! either.
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

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::vector_treatment::{
    Contrast, DesignSummary, NamedColumn, RobustDesignSummary, TreatmentColumn,
    VECTOR_TREATMENT_INFERENCE_CLAIM, VECTOR_TREATMENT_MAX_TREATMENTS, VECTOR_TREATMENT_NULL,
    VectorCovariance, VectorTreatmentFit, VectorTreatmentInput, VectorTreatmentOptions,
    fit_vector_treatment, fit_vector_treatment_from_robust_summary,
    fit_vector_treatment_from_summary, summarize_vector_design, summarize_vector_design_robust,
};
use serde::{Deserialize, Serialize};

use crate::artifact_budget::{
    BudgetDetails, BudgetRefusal, admit_export, admit_regression, unbounded_context,
};
use crate::artifact_codec::CanonicalSize;

/// The artifact major version this reader writes and accepts.
pub const VECTOR_TREATMENT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const VECTOR_TREATMENT_ARTIFACT_FEATURE: &str = "vector_treatment_v1";
/// Calibration coordinate: coverage, Type I error and power were not measured.
pub const VECTOR_TREATMENT_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy: the largest export budget, enforced on consumption.
pub const MAX_VECTOR_TREATMENT_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Row count the legacy robust-covariance `rows` replay format was originally written for. It
/// is no longer applied: a declared row count must match the stored numbers exactly (so the
/// artifact byte budget bounds the rows), and the constant is kept only so existing importers
/// still resolve it.
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
const REPLAY_ROBUST_SUMMARY: &str = "robust_summary";
const REPLAY_ROWS: &str = "rows";
const BUDGET: BudgetDetails = BudgetDetails {
    cancelled: (antecedent_core::reason_code!("cancelled_no_claim"), "vector_treatment.cancelled"),
    memory: "vector_treatment.memory_budget_exceeded",
    export: "vector_treatment.export_budget_exceeded",
};

crate::sectioned_artifact_error!(
    /// Why a vector-treatment artifact was refused or could not be produced.
    VectorTreatmentArtifactError,
    "vector treatment",
    "vector_treatment",
    "result",
    false
);

fn budget_refusal((code, message): BudgetRefusal) -> VectorTreatmentArtifactError {
    VectorTreatmentArtifactError::Refused { code, message }
}

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
    /// `summary`, `robust_summary` or (legacy) `rows`.
    pub replay: String,
    /// Identity of the shared row snapshot.
    pub row_snapshot: String,
    /// Rows of the snapshot.
    pub n_rows: u64,
    /// Digest of the rows a summary was computed from; absent for a `rows` replay, which embeds
    /// them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_digest: Option<String>,
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
    /// Digest of the rows a summary replay was computed from, when it records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_digest: Option<String>,
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
    RobustSummary(RobustDesignSummary),
    Rows { outcome: Vec<f64>, adjustment: Vec<Vec<f64>>, treatments: Vec<Vec<f64>> },
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
            Self::Summary(s) => {
                s.gram.iter().chain(&s.xty).copied().chain(std::iter::once(s.rss)).collect()
            }
            Self::RobustSummary(s) => s
                .design
                .gram
                .iter()
                .chain(&s.design.xty)
                .copied()
                .chain(std::iter::once(s.design.rss))
                .chain(s.meat.iter().copied())
                .collect(),
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
            REPLAY_ROBUST_SUMMARY => {
                if numbers.len() != 2 * p * p + p + 1 {
                    return Err(malformed("numbers section length"));
                }
                let mut rest = numbers;
                Ok(Self::RobustSummary(RobustDesignSummary {
                    design: DesignSummary {
                        n_rows: decl.n_rows,
                        first_treatment: 1 + decl.adjustment.len(),
                        gram: crate::width::split_reals(&mut rest, p * p).to_vec(),
                        xty: crate::width::split_reals(&mut rest, p).to_vec(),
                        rss: crate::width::split_reals(&mut rest, 1)[0],
                    },
                    meat: crate::width::split_reals(&mut rest, p * p).to_vec(),
                }))
            }
            REPLAY_ROWS => {
                if decl.n_rows.checked_mul(p) != Some(numbers.len()) {
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
    data_digest: Option<&str>,
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
    if let Some(digest) = data_digest {
        design.text(digest);
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

fn compute(
    decl: &Decl,
    replay: &Replay,
    data_digest: Option<&str>,
) -> Result<Computed, VectorTreatmentArtifactError> {
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
        Replay::RobustSummary(summary) => {
            if summary.design.n_rows != decl.n_rows {
                return Err(VectorTreatmentArtifactError::Malformed("row count".into()));
            }
            fit_vector_treatment_from_robust_summary(summary, &decl.treatments, &options)?
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
    let identity = identity_of(decl, replay.tag(), &numbers, data_digest, &result);
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
    encode_parts_impl(meta, numbers, artifact_id, MAX_VECTOR_TREATMENT_ARTIFACT_BYTES)
}

fn encode_parts_impl(
    meta: &VectorTreatmentMeta,
    numbers: &[u8],
    artifact_id: &str,
    export_budget: usize,
) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
    // A consumer reads at most the byte cap of uncompressed sections, whatever the container
    // compresses to; the caller's budget bounds the container itself.
    let bytes = crate::artifact_codec::encode_sectioned_admitting(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "vector_treatment_wald_unmeasured",
            max_bytes: MAX_VECTOR_TREATMENT_ARTIFACT_BYTES,
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
            admit_export(BUDGET, uncompressed, MAX_VECTOR_TREATMENT_ARTIFACT_BYTES).map_err(budget_refusal)
        },
    )?;
    admit_export(BUDGET, bytes.len(), export_budget.min(MAX_VECTOR_TREATMENT_ARTIFACT_BYTES))
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
) -> Result<(VectorTreatmentMeta, Vec<u8>), VectorTreatmentArtifactError> {
    decode_parts_impl(bytes)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(VectorTreatmentMeta, Vec<u8>), VectorTreatmentArtifactError> {
    let (meta, parts) = crate::artifact_codec::decode_sectioned::<VectorTreatmentArtifactError, VectorTreatmentMeta>(
        ARTIFACT_KIND,
        MAX_VECTOR_TREATMENT_ARTIFACT_BYTES,
        META_SECTION,
        &[NUMBERS_SECTION],
        VECTOR_TREATMENT_ARTIFACT_VERSION,
        bytes,
    )?;
    let Ok([payload]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(VectorTreatmentArtifactError::Malformed("unsupported container layout".into()));
    };
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

/// Digest of the rows a request fits: the outcome, then every adjustment and treatment column
/// with its name. A summary replay holds this instead of the rows.
fn data_digest(request: &VectorTreatmentRequest) -> String {
    let mut writer = CanonicalWriter::new("vector_treatment_v1.data");
    writer.reals(&request.outcome);
    writer.size(request.adjustment.len());
    for column in &request.adjustment {
        writer.text(&column.name);
        writer.reals(&column.values);
    }
    writer.size(request.treatments.len());
    for treatment in &request.treatments {
        writer.text(&treatment.name);
        writer.reals(&treatment.values);
    }
    writer.finish_hex()
}

impl VectorTreatmentArtifact {
    /// Fit the declared treatments jointly and seal the answer with its embedded design and
    /// identity, with no memory limit; see [`Self::seal_with`].
    ///
    /// # Errors
    /// The core fit's refusals (`route_not_supported` for a treatment on a different adjustment
    /// set, snapshot or row count; `design_rank_deficient`; `invalid_argument`) and the
    /// format's bounds.
    pub fn seal(request: &VectorTreatmentRequest) -> Result<Self, VectorTreatmentArtifactError> {
        Self::seal_with(request, &unbounded_context())
    }

    /// Fit the declared treatments jointly under the computation budget of `ctx`.
    ///
    /// The row count is not limited here: the planned working set must fit the context's hard
    /// memory limit and the context must not be cancelled. Every covariance kind seals a
    /// summary, so the artifact stays small however many rows were fitted.
    ///
    /// # Errors
    /// The refusals of [`Self::seal`], `vector_treatment.memory_budget_exceeded` and
    /// `vector_treatment.cancelled`.
    pub fn seal_with(
        request: &VectorTreatmentRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, VectorTreatmentArtifactError> {
        let kind = covariance_of(&request.covariance)?;
        let decl = Decl {
            row_snapshot: request.row_snapshot.clone(),
            n_rows: request.outcome.len(),
            adjustment: request.adjustment.iter().map(|c| c.name.clone()).collect(),
            treatments: request.treatments.iter().map(|t| t.name.clone()).collect(),
            covariance: request.covariance.clone(),
            contrasts: request.contrasts.clone(),
        };
        check_decl(&decl)?;
        admit_regression(BUDGET, decl.n_rows, decl.columns(), ctx).map_err(budget_refusal)?;
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
        let replay = if kind == VectorCovariance::ModelBased {
            Replay::Summary(summarize_vector_design(&input)?)
        } else {
            Replay::RobustSummary(summarize_vector_design_robust(&input, kind)?)
        };
        let digest = data_digest(request);
        let computed = compute(&decl, &replay, Some(&digest))?;
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
            data_digest: Some(digest),
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
            data_digest: self.meta.data_digest.clone(),
            adjustment: self.meta.adjustment.clone(),
            treatments: self.meta.treatments.clone(),
            declared_contrasts: self.meta.contrasts.clone(),
            result: self.meta.result.clone(),
            caveats: self.meta.caveats.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Check rows a caller still holds against the digest this artifact's summary was computed
    /// from.
    ///
    /// # Errors
    /// [`VectorTreatmentArtifactError::IdentityMismatch`] (`data`) when the rows differ, and
    /// `vector_treatment.data_not_bound` for an artifact that records no digest (a legacy `rows`
    /// replay, which embeds its rows instead).
    pub fn verify_data(
        &self,
        request: &VectorTreatmentRequest,
    ) -> Result<(), VectorTreatmentArtifactError> {
        match &self.meta.data_digest {
            Some(stored) if *stored == data_digest(request) => Ok(()),
            Some(_) => Err(VectorTreatmentArtifactError::IdentityMismatch { field: "data" }),
            None => Err(VectorTreatmentArtifactError::Refused {
                code: antecedent_core::reason_code!("route_not_supported"),
                message: "vector_treatment.data_not_bound: this artifact embeds its rows and \
                          records no data digest"
                    .to_owned(),
            }),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or `vector_treatment.export_budget_exceeded` when the
    /// artifact would exceed [`MAX_VECTOR_TREATMENT_ARTIFACT_BYTES`].
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
        self.to_bytes_within(artifact_id, MAX_VECTOR_TREATMENT_ARTIFACT_BYTES)
    }

    /// Serialize when the whole artifact fits `export_budget` bytes (at most
    /// [`MAX_VECTOR_TREATMENT_ARTIFACT_BYTES`]).
    ///
    /// # Errors
    /// An empty id, an encoding failure or `vector_treatment.export_budget_exceeded`, which names
    /// the bytes needed and the budget. The fit is untouched by the refusal.
    pub fn to_bytes_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
        encode_parts_impl(&self.meta, &reals_to_bytes(&self.numbers), artifact_id, export_budget)
    }

    /// Consume an artifact by recomputation, with no memory limit; see [`Self::from_bytes_with`].
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
        Self::from_bytes_with(bytes, expected, &unbounded_context())
    }

    /// Consume an artifact by recomputation within the replay budget of `ctx`.
    ///
    /// # Errors
    /// The refusals of [`Self::from_bytes`], `vector_treatment.memory_budget_exceeded` and
    /// `vector_treatment.cancelled`.
    pub fn from_bytes_with(
        bytes: &[u8],
        expected: Option<&VectorTreatmentIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, VectorTreatmentArtifactError> {
        let (meta, raw) = decode_parts_impl(bytes)?;
        crate::artifact_codec::require_semantics::<VectorTreatmentArtifactError>(
            meta.feature == VECTOR_TREATMENT_ARTIFACT_FEATURE,
            "feature marker",
        )?;
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
        let replayed_rows = if matches!(replay, Replay::Rows { .. }) { decl.n_rows } else { 0 };
        admit_regression(BUDGET, replayed_rows, decl.columns(), ctx).map_err(budget_refusal)?;
        let computed = compute(&decl, &replay, meta.data_digest.as_deref())?;
        if let Some(field) = identity_diff(&meta.identity, &computed.identity) {
            return Err(VectorTreatmentArtifactError::IdentityMismatch { field });
        }
        // Replay contract: the consumer re-runs the same deterministic code on the stored
        // numbers, so the result must agree exactly (derived `PartialEq`: bitwise except that
        // `-0.0 == 0.0`); a tolerance would admit a result the producer never computed.
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

#[cfg(test)]
mod number_section_tests {
    use super::{
        AdjustmentColumnRequest, Decl, MAX_VECTOR_TREATMENT_ARTIFACT_BYTES, REPLAY_ROWS, Replay,
        TreatmentRequest, VECTOR_TREATMENT_HC_ROW_CAP, VectorTreatmentArtifact,
        VectorTreatmentArtifactError, VectorTreatmentRequest, compute, encode_parts_impl,
        reals_to_bytes,
    };

    fn decl(n_rows: usize) -> Decl {
        Decl {
            row_snapshot: String::new(),
            n_rows,
            adjustment: vec!["z".into()],
            treatments: vec!["a".into()],
            covariance: String::new(),
            contrasts: Vec::new(),
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
            adjustment: vec![vec![nan, -1.5]],
            treatments: vec![vec![0.0, f64::MIN_POSITIVE]],
        };
        let numbers = replay.numbers();
        let bytes = crate::width::f64s_to_le_bytes(&numbers);
        let direct: Vec<u8> = numbers.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, direct);
        let decoded = crate::width::f64s_from_le_bytes(&bytes);
        let back = Replay::from_numbers(&decl(2), REPLAY_ROWS, &decoded).unwrap();
        assert_eq!(bits(&back.numbers()), bits(&numbers));
        assert_eq!(crate::width::f64s_to_le_bytes(&back.numbers()), bytes);
    }

    #[test]
    fn zero_row_replay_has_no_columns() {
        let back = Replay::from_numbers(&decl(0), REPLAY_ROWS, &[]).unwrap();
        let Replay::Rows { outcome, adjustment, treatments } = back else {
            panic!("rows replay");
        };
        assert!(outcome.is_empty());
        assert!(adjustment.is_empty());
        assert!(treatments.is_empty());
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

    fn robust_request(n: usize, covariance: &str) -> VectorTreatmentRequest {
        let treatment = |name: &str, seed| TreatmentRequest {
            name: name.into(),
            values: column(n, seed),
            adjustment_set: vec!["w".into()],
            row_snapshot: "snap".into(),
        };
        VectorTreatmentRequest {
            outcome: column(n, 1),
            row_snapshot: "snap".into(),
            adjustment: vec![AdjustmentColumnRequest { name: "w".into(), values: column(n, 2) }],
            treatments: vec![treatment("t1", 3), treatment("t2", 4)],
            covariance: covariance.into(),
            contrasts: Vec::new(),
        }
    }

    /// An artifact sealed before summaries existed embedded the rows of a robust fit. Its
    /// consumer is kept, and it reaches exactly the fit the summary artifact does.
    #[test]
    fn a_legacy_rows_artifact_replays_to_the_summary_fit_and_records_no_digest() {
        for covariance in ["hc0", "hc1", "hc2", "hc3"] {
            let request = robust_request(60, covariance);
            let summary = VectorTreatmentArtifact::seal(&request).unwrap();
            assert_eq!(summary.meta().replay, "robust_summary");
            let decl = Decl::from_meta(summary.meta()).unwrap();
            let rows = Replay::Rows {
                outcome: request.outcome.clone(),
                adjustment: request.adjustment.iter().map(|c| c.values.clone()).collect(),
                treatments: request.treatments.iter().map(|t| t.values.clone()).collect(),
            };
            let computed = compute(&decl, &rows, None).unwrap();
            let mut meta = summary.meta().clone();
            meta.replay = REPLAY_ROWS.into();
            meta.data_digest = None;
            meta.identity = computed.identity;
            let bytes = encode_parts_impl(
                &meta,
                &reals_to_bytes(&computed.numbers),
                "legacy",
                MAX_VECTOR_TREATMENT_ARTIFACT_BYTES,
            )
            .unwrap();
            let legacy = VectorTreatmentArtifact::from_bytes(&bytes, Some(&meta.identity)).unwrap();
            assert_eq!(legacy.meta().replay, REPLAY_ROWS);
            assert_eq!(legacy.fit(), summary.fit(), "{covariance}");
            // Embedded rows need no digest; asking to verify against one says so.
            let error = legacy.verify_data(&request).unwrap_err();
            assert_eq!(error.refusal().unwrap().1, "vector_treatment.data_not_bound");
        }
    }

    #[test]
    fn a_legacy_rows_replay_declaring_more_rows_than_it_stores_is_malformed() {
        let decl = decl(VECTOR_TREATMENT_HC_ROW_CAP + 1);
        assert!(matches!(
            Replay::from_numbers(&decl, REPLAY_ROWS, &[]),
            Err(VectorTreatmentArtifactError::Malformed(_))
        ));
    }
}
