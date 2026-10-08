//! Independent artifact for the 2.3 B1 dose-grid functional row (X4).
//!
//! Format version 1. The artifact binds the design attestation, the one named
//! functional, the grid (or the two contrast doses), the fixed bandwidth with its
//! declared range, the minimum local effective sample size, the declared claims and
//! a compact `(dose, outcome)` table. It stores the full result: per-dose support
//! labels with local effective sample sizes, the point values with pointwise
//! standard errors and intervals (calibration `unmeasured`, smoothing bias not
//! included), the measured numerical residual of the local solves, and the closed
//! simultaneous-band marker.
//! The optional caller-attested exact quadratic mean premise requires
//! `dose_grid_quadratic_mean_v1`; polynomial reproduction accounts for zero smoothing
//! bias under that premise. The sampling calibration remains unmeasured. Omitting the
//! premise preserves the historical format and digests.
//!
//! A consumer trusts nothing in the result. It re-validates the stored request,
//! recomputes the Gaussian-kernel local-quadratic fit from the embedded table and
//! accepts only a result identical, bit for bit, to the stored one. Two digests
//! guard the stored request: the premises digest (design, functional, grid,
//! bandwidth and its range, minimum effective sample size, claims) and a separate
//! data digest (the table, bit for bit). A changed premise, table entry or stored
//! value is refused even when both digests are re-sealed, because the replayed
//! result then differs.
//!
//! Scope. Pointwise intervals are normal intervals around the smoother's level,
//! derivative or contrast with calibration unmeasured: no coverage claim is made.
//! A simultaneous band is a different claim and is closed; an artifact for it is
//! never produced and a stored band marker other than `closed` is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{IdentityDomain, SupportReport, SupportStatus};
use antecedent_estimate::EstimationError;
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, DoseGridRow, IntervalCalibration,
    PointwiseInterval, dose_support_report, estimate_dose_grid_functional,
    estimate_dose_grid_quadratic_mean,
};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const DOSE_GRID_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const DOSE_GRID_ARTIFACT_FEATURE: &str = "dose_grid_functional_row_v1";
/// Additional required semantics: caller attests an exact conditional quadratic mean.
pub const DOSE_GRID_QUADRATIC_FEATURE: &str = "dose_grid_quadratic_mean_v1";
/// Calibration status of every interval of this row.
pub const DOSE_GRID_CALIBRATION: &str = "unmeasured";
/// State of the simultaneous band of this row.
pub const DOSE_GRID_BAND_CLOSED: &str = "closed";
/// The only inference claim of this row.
pub const DOSE_GRID_INFERENCE_CLAIM: &str = "pointwise_calibration_unmeasured";

const DESIGN_RANDOMIZED: &str = "randomized_dose";
const DESIGN_OBSERVATIONAL: &str = "observational";
const KIND_LEVEL: &str = "level";
const KIND_DERIVATIVE: &str = "derivative";
const KIND_CONTRAST: &str = "contrast";

/// Why a dose-grid artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DoseGridArtifactError {
    /// The feature marker, a field spelling or the result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The row refused the stored request with its own typed reason.
    #[error("{detail}: {message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// Namespaced detail, such as `dose_grid.unsupported_dose`.
        detail: String,
        /// What was refused.
        message: String,
    },
    /// The recomputed result differs from the stored one.
    #[error("the stored result does not replay")]
    ResultMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored table.
    #[error("data digest mismatch")]
    DataMismatch,
    /// The bytes do not decode.
    #[error("artifact does not decode: {0}")]
    Undecodable(String),
}

impl DoseGridArtifactError {
    /// `(reason code, detail)` of this refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, String) {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::UnsupportedSemantics(_) => (
                antecedent_core::reason_code!("route_not_supported"),
                "dose_grid.unsupported_semantics".to_owned(),
            ),
            Self::LimitsExceeded(_) => (invalid, "dose_grid.consumer_limit_exceeded".to_owned()),
            Self::Refused { code, detail, .. } => (*code, detail.clone()),
            Self::ResultMismatch => (invalid, "dose_grid.result_replay_mismatch".to_owned()),
            Self::PremisesMismatch => (invalid, "dose_grid.premises_mismatch".to_owned()),
            Self::DataMismatch => (invalid, "dose_grid.data_identity_mismatch".to_owned()),
            Self::Undecodable(_) => (invalid, "dose_grid.undecodable".to_owned()),
        }
    }
}

impl From<DoseGridArtifactError> for IoError {
    fn from(error: DoseGridArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: dose grid artifact: {error}") }
    }
}

impl From<EstimationError> for DoseGridArtifactError {
    fn from(error: EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message }
            | EstimationError::RefusedWithFields { code, message, .. } => {
                let (detail, text) = message.split_once(": ").unwrap_or(("dose_grid.refused", ""));
                Self::Refused { code, detail: detail.to_owned(), message: text.to_owned() }
            }
            other => Self::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "dose_grid.invalid_request".to_owned(),
                message: other.to_string(),
            },
        }
    }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DoseGridConsumeLimits {
    /// Most `(dose, outcome)` rows.
    pub max_rows: usize,
    /// Most grid doses.
    pub max_grid: usize,
}

impl Default for DoseGridConsumeLimits {
    fn default() -> Self {
        Self { max_rows: 100_000, max_grid: 1_024 }
    }
}

/// The named functional on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseFunctionalWire {
    /// `level`, `derivative` or `contrast`.
    pub kind: String,
    /// Reference dose of a contrast, otherwise absent.
    pub from: Option<f64>,
    /// Comparison dose of a contrast, otherwise absent.
    pub to: Option<f64>,
}

/// The declared claims on the wire. Each is separate; none implies another.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DoseClaimsWire {
    /// A pointwise claim about `m(d)` (levels and contrasts).
    pub pointwise_level: bool,
    /// A pointwise claim about `m'(d)`.
    pub derivative: bool,
    /// A simultaneous band; always refused by the row.
    pub simultaneous_band: bool,
}

/// Everything a replay needs: the premises and the compact data table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseGridRequestWire {
    /// `randomized_dose` or `observational` (refused by the row).
    pub design: String,
    /// Caller-attested exact `E[Y|D=d] = beta0 + beta1*d + beta2*d²`.
    /// Under this premise polynomial reproduction implies zero smoothing bias.
    /// It is not inferred from the observed residuals; sampling calibration stays unmeasured.
    #[serde(default, skip_serializing_if = "is_false")]
    pub quadratic_mean: bool,
    /// The one named functional.
    pub functional: DoseFunctionalWire,
    /// Grid doses of a level or derivative; empty for a contrast.
    pub grid: Vec<f64>,
    /// Fixed Gaussian bandwidth.
    pub bandwidth: f64,
    /// Declared inclusive bandwidth range.
    pub bandwidth_range: (f64, f64),
    /// Smallest admissible Kish effective sample size at any requested dose.
    pub minimum_local_ess: f64,
    /// Declared claims.
    pub claims: DoseClaimsWire,
    /// Observed doses.
    pub dose: Vec<f64>,
    /// Observed outcomes, aligned with `dose`.
    pub outcome: Vec<f64>,
}

/// One pointwise value (a level or a derivative) on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DosePointWire {
    /// Grid dose.
    pub dose: f64,
    /// Support label of this dose.
    pub support: String,
    /// Local-quadratic level or first derivative.
    pub value: f64,
    /// Robust sandwich standard error.
    pub standard_error: f64,
    /// Kish effective sample size of the local weights.
    pub local_ess: f64,
    /// Lower endpoint of the pointwise interval.
    pub lower: f64,
    /// Upper endpoint of the pointwise interval.
    pub upper: f64,
    /// Nominal level the normal quantile was taken at; not a coverage claim.
    pub nominal_level: f64,
    /// Calibration status (`unmeasured`).
    pub calibration: String,
    /// Whether smoothing bias is accounted for: only the attested exact-quadratic scope.
    pub smoothing_bias_included: bool,
}

/// A named contrast on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseContrastWire {
    /// Reference dose.
    pub from: f64,
    /// Comparison dose.
    pub to: f64,
    /// Support label of the reference dose.
    pub from_support: String,
    /// Support label of the comparison dose.
    pub to_support: String,
    /// Contrast estimate.
    pub estimate: f64,
    /// Covariance-aware standard error.
    pub standard_error: f64,
    /// Lower endpoint of the pointwise interval.
    pub lower: f64,
    /// Upper endpoint of the pointwise interval.
    pub upper: f64,
    /// Nominal level the normal quantile was taken at; not a coverage claim.
    pub nominal_level: f64,
    /// Calibration status (`unmeasured`).
    pub calibration: String,
    /// Whether smoothing bias is accounted for: only the attested exact-quadratic scope.
    pub smoothing_bias_included: bool,
}

/// Per-dose support label on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseSupportWire {
    /// Requested dose.
    pub dose: f64,
    /// Support label.
    pub label: String,
    /// Kish effective sample size of the local weights at this dose.
    pub local_ess: f64,
}

/// The stored result of the row.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseGridResultWire {
    /// `level`, `derivative` or `contrast`.
    pub functional: String,
    /// Design attestation the result rests on.
    pub design: String,
    /// Rows used.
    pub n_rows: usize,
    /// Bandwidth used.
    pub bandwidth: f64,
    /// Pointwise levels (level functional only).
    pub levels: Vec<DosePointWire>,
    /// Pointwise derivatives (derivative functional only).
    pub derivatives: Vec<DosePointWire>,
    /// The contrast (contrast functional only).
    pub contrast: Option<DoseContrastWire>,
    /// Per-dose support labels, in request order.
    pub support: Vec<DoseSupportWire>,
    /// Worst support label.
    pub support_status: String,
    /// Local fits performed.
    pub fits: usize,
    /// Largest absolute sum of linearized influences (zero in exact arithmetic).
    pub max_abs_influence_sum: f64,
    /// Calibration status of every interval (`unmeasured`).
    pub calibration: String,
    /// Whether zero smoothing bias is derived under the attested exact-quadratic scope.
    pub smoothing_bias_included: bool,
    /// Simultaneous band state (`closed`).
    pub simultaneous_band: String,
    /// The only inference claim of this row.
    pub inference_claim: String,
}

/// Versioned dose-grid result with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseGridArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Premises and the compact data table.
    pub request: DoseGridRequestWire,
    /// The stored result.
    pub result: DoseGridResultWire,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the data table.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    design: &'a str,
    #[serde(skip_serializing_if = "is_false")]
    quadratic_mean: bool,
    kind: &'a str,
    from: Option<u64>,
    to: Option<u64>,
    grid: Vec<u64>,
    bandwidth: u64,
    bandwidth_range: (u64, u64),
    minimum_local_ess: u64,
    claims: &'a DoseClaimsWire,
    rows: usize,
}

#[derive(Serialize)]
struct DataView {
    tag: &'static str,
    dose: Vec<u64>,
    outcome: Vec<u64>,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde predicates receive field references")]
const fn is_false(value: &bool) -> bool {
    !*value
}

fn required_features(quadratic_mean: bool) -> Vec<String> {
    let mut features = vec![DOSE_GRID_ARTIFACT_FEATURE.to_owned()];
    if quadratic_mean {
        features.push(DOSE_GRID_QUADRATIC_FEATURE.to_owned());
    }
    features
}

fn status_name(label: SupportStatus) -> String {
    label.as_str().to_owned()
}

fn calibration_name(calibration: IntervalCalibration) -> String {
    match calibration {
        IntervalCalibration::Unmeasured => DOSE_GRID_CALIBRATION.to_owned(),
    }
}

fn point_wire(
    dose: f64,
    support: SupportStatus,
    value: f64,
    standard_error: f64,
    local_ess: f64,
    interval: &PointwiseInterval,
) -> DosePointWire {
    DosePointWire {
        dose,
        support: status_name(support),
        value,
        standard_error,
        local_ess,
        lower: interval.lower,
        upper: interval.upper,
        nominal_level: interval.nominal_level,
        calibration: calibration_name(interval.calibration),
        smoothing_bias_included: interval.smoothing_bias_included,
    }
}

impl DoseGridRequestWire {
    fn design(&self) -> Result<DoseDesign, DoseGridArtifactError> {
        match self.design.as_str() {
            DESIGN_RANDOMIZED => Ok(DoseDesign::RandomizedDose),
            DESIGN_OBSERVATIONAL => Ok(DoseDesign::Observational),
            _ => Err(DoseGridArtifactError::UnsupportedSemantics("design spelling")),
        }
    }

    fn functional(&self) -> Result<DoseFunctional, DoseGridArtifactError> {
        let spec = &self.functional;
        let unsupported = DoseGridArtifactError::UnsupportedSemantics;
        match (spec.kind.as_str(), spec.from, spec.to) {
            (KIND_LEVEL, None, None) if !self.grid.is_empty() => Ok(DoseFunctional::Level),
            (KIND_DERIVATIVE, None, None) if !self.grid.is_empty() => {
                Ok(DoseFunctional::Derivative)
            }
            (KIND_CONTRAST, Some(from), Some(to)) if self.grid.is_empty() => {
                Ok(DoseFunctional::Contrast { from, to })
            }
            (KIND_LEVEL | KIND_DERIVATIVE, _, _) => {
                Err(unsupported("a level or derivative needs a grid and no contrast doses"))
            }
            (KIND_CONTRAST, _, _) => {
                Err(unsupported("a contrast names its two doses and takes no grid"))
            }
            _ => Err(unsupported("functional kind")),
        }
    }

    fn premises_view(&self) -> PremisesView<'_> {
        PremisesView {
            tag: "dose_grid_functional_premises_v1",
            design: &self.design,
            quadratic_mean: self.quadratic_mean,
            kind: &self.functional.kind,
            from: self.functional.from.map(f64::to_bits),
            to: self.functional.to.map(f64::to_bits),
            grid: bits(&self.grid),
            bandwidth: self.bandwidth.to_bits(),
            bandwidth_range: (self.bandwidth_range.0.to_bits(), self.bandwidth_range.1.to_bits()),
            minimum_local_ess: self.minimum_local_ess.to_bits(),
            claims: &self.claims,
            rows: self.dose.len(),
        }
    }

    fn premises_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(IdentityDomain::Claim, &self.premises_view())?.to_hex())
    }

    fn data_digest(&self) -> Result<String, IoError> {
        let view = DataView {
            tag: "dose_grid_functional_data_v1",
            dose: bits(&self.dose),
            outcome: bits(&self.outcome),
        };
        Ok(crate::identity::digest_wire(IdentityDomain::DataSnapshot, &view)?.to_hex())
    }

    fn evaluation_points(&self, functional: DoseFunctional) -> Vec<f64> {
        match functional {
            DoseFunctional::Contrast { from, to } => vec![from, to],
            DoseFunctional::Level | DoseFunctional::Derivative => self.grid.clone(),
        }
    }

    /// Run the row on this request and render its result.
    fn evaluate(&self) -> Result<DoseGridResultWire, DoseGridArtifactError> {
        let design = self.design()?;
        let functional = self.functional()?;
        let request = DoseGridRequest {
            dose: &self.dose,
            outcome: &self.outcome,
            grid: &self.grid,
            bandwidth: self.bandwidth,
            bandwidth_range: self.bandwidth_range,
            minimum_local_ess: self.minimum_local_ess,
            functional,
            claims: ClaimSet {
                pointwise_level: self.claims.pointwise_level,
                derivative: self.claims.derivative,
                simultaneous_band: self.claims.simultaneous_band,
            },
            design,
        };
        let row = if self.quadratic_mean {
            estimate_dose_grid_quadratic_mean(&request)?
        } else {
            estimate_dose_grid_functional(&request)?
        };
        let points = self.evaluation_points(functional);
        let result = self.render(&row, &points, design);
        if !result_is_finite(&result) {
            return Err(DoseGridArtifactError::Refused {
                code: antecedent_core::reason_code!("transport_numerical_failure"),
                detail: "dose_grid.non_finite_result".to_owned(),
                message: "the local fit produced a non-finite value".to_owned(),
            });
        }
        Ok(result)
    }

    fn render(&self, row: &DoseGridRow, points: &[f64], design: DoseDesign) -> DoseGridResultWire {
        let support = support_rows(&row.support, points);
        DoseGridResultWire {
            functional: self.functional.kind.clone(),
            design: match design {
                DoseDesign::RandomizedDose => DESIGN_RANDOMIZED.to_owned(),
                DoseDesign::Observational => DESIGN_OBSERVATIONAL.to_owned(),
            },
            n_rows: row.n_rows,
            bandwidth: row.bandwidth,
            levels: row
                .levels
                .iter()
                .map(|l| {
                    point_wire(
                        l.dose,
                        l.support,
                        l.level,
                        l.standard_error,
                        l.local_ess,
                        &l.interval,
                    )
                })
                .collect(),
            derivatives: row
                .derivatives
                .iter()
                .map(|d| {
                    point_wire(
                        d.dose,
                        d.support,
                        d.derivative,
                        d.standard_error,
                        d.local_ess,
                        &d.interval,
                    )
                })
                .collect(),
            contrast: row.contrast.as_ref().map(|c| DoseContrastWire {
                from: c.from,
                to: c.to,
                from_support: status_name(c.from_support),
                to_support: status_name(c.to_support),
                estimate: c.estimate,
                standard_error: c.standard_error,
                lower: c.interval.lower,
                upper: c.interval.upper,
                nominal_level: c.interval.nominal_level,
                calibration: calibration_name(c.interval.calibration),
                smoothing_bias_included: c.interval.smoothing_bias_included,
            }),
            support,
            support_status: status_name(row.support.status),
            fits: row.numerical.fits,
            max_abs_influence_sum: row.numerical.max_abs_influence_sum,
            calibration: DOSE_GRID_CALIBRATION.to_owned(),
            smoothing_bias_included: self.quadratic_mean,
            simultaneous_band: if row.simultaneous_band.is_none() {
                DOSE_GRID_BAND_CLOSED.to_owned()
            } else {
                "open".to_owned()
            },
            inference_claim: DOSE_GRID_INFERENCE_CLAIM.to_owned(),
        }
    }

    fn check_limits(&self, limits: &DoseGridConsumeLimits) -> Result<(), DoseGridArtifactError> {
        if self.dose.len() > limits.max_rows || self.outcome.len() > limits.max_rows {
            return Err(DoseGridArtifactError::LimitsExceeded("rows"));
        }
        if self.grid.len() > limits.max_grid {
            return Err(DoseGridArtifactError::LimitsExceeded("grid doses"));
        }
        Ok(())
    }
}

fn support_rows(report: &SupportReport, points: &[f64]) -> Vec<DoseSupportWire> {
    let labels = report.point_status.as_deref().unwrap_or(&[]);
    let ess: &[f64] = report
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.local_ess")
        .map_or(&[][..], |d| &d.values[..]);
    points
        .iter()
        .zip(labels)
        .zip(ess)
        .map(|((&dose, &label), &local_ess)| DoseSupportWire {
            dose,
            label: status_name(label),
            local_ess,
        })
        .collect()
}

/// Label every requested dose without refusing: per-dose support, in request order.
///
/// This is the labelling half of the per-dose support contract. The row itself
/// refuses any dose that is not `supported`; this table is how a caller sees why.
///
/// # Errors
/// Fewer than three finite doses, empty or non-finite `points`, or a non-positive
/// bandwidth, as the row's typed `dose_grid.invalid_request` refusal.
pub fn dose_support_table(
    dose: &[f64],
    points: &[f64],
    bandwidth: f64,
    minimum_local_ess: f64,
) -> Result<Vec<DoseSupportWire>, DoseGridArtifactError> {
    let report = dose_support_report(dose, points, bandwidth, minimum_local_ess)?;
    Ok(support_rows(&report, points))
}

fn result_is_finite(result: &DoseGridResultWire) -> bool {
    let point_finite = |p: &DosePointWire| {
        [p.dose, p.value, p.standard_error, p.local_ess, p.lower, p.upper, p.nominal_level]
            .iter()
            .all(|v| v.is_finite())
    };
    result.levels.iter().chain(&result.derivatives).all(point_finite)
        && result.contrast.as_ref().is_none_or(|c| {
            [c.from, c.to, c.estimate, c.standard_error, c.lower, c.upper, c.nominal_level]
                .iter()
                .all(|v| v.is_finite())
        })
        && result.support.iter().all(|s| s.dose.is_finite() && s.local_ess.is_finite())
        && result.bandwidth.is_finite()
        && result.max_abs_influence_sum.is_finite()
}

impl DoseGridArtifactWire {
    /// Evaluate the row on `request` and seal the result with both digests.
    ///
    /// # Errors
    /// The row's typed refusal (`dose_grid.*`), a stored collection above the default
    /// consumer bounds, or an unsupported spelling.
    pub fn seal(request: DoseGridRequestWire) -> Result<Self, DoseGridArtifactError> {
        request.check_limits(&DoseGridConsumeLimits::default())?;
        let result = request.evaluate()?;
        let undecodable = |e: IoError| DoseGridArtifactError::Undecodable(e.to_string());
        let wire = Self {
            version: DOSE_GRID_ARTIFACT_VERSION,
            required_features: required_features(request.quadratic_mean),
            premises_digest: request.premises_digest().map_err(undecodable)?,
            data_digest: request.data_digest().map_err(undecodable)?,
            request,
            result,
        };
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still recomputes the whole result.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        self.request.premises_digest()
    }

    /// The data digest the stored table should carry.
    ///
    /// # Errors
    /// The table does not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        self.request.data_digest()
    }

    fn validate_shape(&self) -> Result<(), DoseGridArtifactError> {
        let unsupported = DoseGridArtifactError::UnsupportedSemantics;
        if self.required_features != required_features(self.request.quadratic_mean) {
            return Err(unsupported("required features"));
        }
        if self.result.calibration != DOSE_GRID_CALIBRATION
            || self.result.smoothing_bias_included != self.request.quadratic_mean
            || self.result.inference_claim != DOSE_GRID_INFERENCE_CLAIM
        {
            return Err(unsupported("this row publishes pointwise unmeasured intervals only"));
        }
        if self.result.simultaneous_band != DOSE_GRID_BAND_CLOSED
            || self.request.claims.simultaneous_band
        {
            return Err(unsupported("the simultaneous band is closed"));
        }
        if self.request.dose.len() != self.request.outcome.len() {
            return Err(unsupported("dose and outcome tables must align"));
        }
        Ok(())
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
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != DOSE_GRID_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Decode and recompute everything under the consumer's limits; accept only a
    /// result identical to the stored one.
    ///
    /// # Errors
    /// Any reconstruction failure, with the reason code and `dose_grid.*` detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: DoseGridConsumeLimits,
    ) -> Result<Self, IoError> {
        Self::consume_typed(bytes, limits).map_err(IoError::from)
    }

    /// [`Self::consume_with_limits`] with the typed refusal kind.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume_typed(
        bytes: &[u8],
        limits: DoseGridConsumeLimits,
    ) -> Result<Self, DoseGridArtifactError> {
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => DoseGridArtifactError::UnsupportedSemantics("shape"),
            other => DoseGridArtifactError::Undecodable(other.to_string()),
        })?;
        wire.request.check_limits(&limits)?;
        let undecodable = |e: IoError| DoseGridArtifactError::Undecodable(e.to_string());
        if wire.request.premises_digest().map_err(undecodable)? != wire.premises_digest {
            return Err(DoseGridArtifactError::PremisesMismatch);
        }
        if wire.request.data_digest().map_err(undecodable)? != wire.data_digest {
            return Err(DoseGridArtifactError::DataMismatch);
        }
        let fresh = wire.request.evaluate()?;
        let stored = crate::to_cbor(&wire.result).map_err(undecodable)?;
        let recomputed = crate::to_cbor(&fresh).map_err(undecodable)?;
        if stored != recomputed {
            return Err(DoseGridArtifactError::ResultMismatch);
        }
        Ok(wire)
    }
}
