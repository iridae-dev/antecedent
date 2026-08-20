//! Causal-response result and support vocabulary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{AssumptionSet, Diagnostic, IdentificationStatus, ResponseFunctional};

/// Empirical support classification, orthogonal to structural identification.
///
/// On a static curve, [`SupportReport::status`] is the worst label over requested
/// points. On a temporal dose × horizon surface it is a three-way summary of
/// [`SupportReport::point_status`]: every cell supported, mixed (partially
/// extrapolative), or no cell supported.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SupportStatus {
    /// Requested region is empirically supported under configured diagnostics.
    Supported,
    /// Support exists but overlap/local effective sample information is weak.
    WeakOverlap,
    /// Result relies on fitted-model extrapolation within the marginal observed range.
    ///
    /// On a temporal surface this is also the mixed-cell summary: some requested
    /// `(dose, horizon)` cells are inside that horizon's lag-aligned treatment
    /// range and some are not.
    Extrapolative,
    /// At least one requested coordinate is outside marginal empirical support.
    ///
    /// On a temporal surface this means no requested cell sits inside its
    /// horizon's lag-aligned treatment range.
    OutsideEmpiricalSupport,
}

impl SupportStatus {
    /// Stable `snake_case` spelling used on the Python and artifact wires.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::WeakOverlap => "weak_overlap",
            Self::Extrapolative => "extrapolative",
            Self::OutsideEmpiricalSupport => "outside_empirical_support",
        }
    }
}

/// Region assessed by support diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportRegion {
    /// Minimum coordinate per treatment dimension.
    pub minima: Arc<[f64]>,
    /// Maximum coordinate per treatment dimension.
    pub maxima: Arc<[f64]>,
}

/// One machine-readable empirical support diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportDiagnostic {
    /// Stable diagnostic id.
    pub id: Arc<str>,
    /// Per-grid values, if applicable.
    pub values: Arc<[f64]>,
    /// Human-readable interpretation.
    pub detail: Arc<str>,
}

/// Scientific support report retained on every response result.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportReport {
    /// Surface-level status. Static curves use worst-over-points; temporal
    /// dose × horizon surfaces use the three-way split documented on
    /// [`SupportStatus`].
    pub status: SupportStatus,
    /// Assessed query region.
    pub query_region: SupportRegion,
    /// Structured diagnostics such as local ESS and conditional density.
    pub diagnostics: Vec<SupportDiagnostic>,
    /// Non-fatal warnings.
    pub warnings: Vec<Diagnostic>,
    /// Per-cell status on a temporal surface, dose-major like the mean:
    /// `point_status[d * n_horizons + h]`. Intervention paths have length
    /// `n_horizons`. `None` on static curves.
    pub point_status: Option<Arc<[SupportStatus]>>,
}

/// Closed lower/upper identified set.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedSet<T> {
    /// Lower endpoint/envelope.
    pub lower: T,
    /// Upper endpoint/envelope.
    pub upper: T,
}

impl IdentifiedSet<f64> {
    /// Construct a finite scalar identified interval.
    ///
    /// # Errors
    ///
    /// Returns an error when either endpoint is non-finite or `lower > upper`.
    pub fn try_new(lower: f64, upper: f64) -> Result<Self, &'static str> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err("identified interval requires finite lower <= upper");
        }
        Ok(Self { lower, upper })
    }

    /// Intersect two scalar identified intervals.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        (lower <= upper).then_some(Self { lower, upper })
    }
}

/// Function-valued identified envelope on a shared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseEnvelope {
    /// Row-major grid coordinates.
    pub grid: Arc<[f64]>,
    /// Coordinate dimension.
    pub dimension: usize,
    /// Lower response at each grid row.
    pub lower: Arc<[f64]>,
    /// Upper response at each grid row.
    pub upper: Arc<[f64]>,
}

/// Numerical response payload.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseValue {
    /// Scalar functional such as an ADE or point derivative.
    Scalar(f64),
    /// Scalar-outcome curve/surface on a row-major grid.
    Surface {
        /// Grid coordinates.
        grid: Arc<[f64]>,
        /// Coordinate dimension.
        dimension: usize,
        /// Mean response per grid row.
        mean: Arc<[f64]>,
    },
    /// Outcome vector.
    Vector(Arc<[f64]>),
    /// Row-major outcomes-by-treatments Jacobian.
    Jacobian {
        /// Number of outcomes/rows.
        outcomes: usize,
        /// Number of treatments/columns.
        treatments: usize,
        /// Matrix values in row-major order.
        values: Arc<[f64]>,
    },
    /// Function-valued lower/upper envelope.
    Envelope(ResponseEnvelope),
}

/// Statistical uncertainty kind. Pointwise and simultaneous bands are never aliases.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseUncertainty {
    /// No uncertainty was requested/available.
    None,
    /// Scalar standard error and interval.
    Scalar {
        /// Standard error.
        standard_error: f64,
        /// Confidence/credible level.
        level: f64,
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
    },
    /// Per-coordinate intervals without simultaneous coverage semantics.
    PointwiseBand {
        /// Confidence level for each coordinate.
        level: f64,
        /// Lower values.
        lower: Arc<[f64]>,
        /// Upper values.
        upper: Arc<[f64]>,
    },
    /// One band calibrated for simultaneous coverage over the requested grid.
    SimultaneousBand {
        /// Simultaneous confidence level.
        level: f64,
        /// Lower values.
        lower: Arc<[f64]>,
        /// Upper values.
        upper: Arc<[f64]>,
        /// Resampling replicates used.
        replicates: u32,
    },
    /// Confidence band around an identified envelope (not the envelope itself).
    IdentifiedEnvelopeBand {
        /// Confidence level.
        level: f64,
        /// Lower confidence limit for the identified lower envelope.
        lower_outer: Arc<[f64]>,
        /// Upper confidence limit for the identified upper envelope.
        upper_outer: Arc<[f64]>,
    },
    /// Posterior draws or summaries live in a referenced posterior artifact.
    Posterior {
        /// Artifact identifier.
        artifact_id: Arc<str>,
    },
}

/// Identification payload for a response.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseIdentification {
    /// Point-identified numerical response.
    PointIdentified(ResponseValue),
    /// Partially identified numerical response/envelope.
    PartiallyIdentified(ResponseValue),
    /// Graph-conditional responses; keys are stable graph identifiers.
    GraphDependent(Vec<(u64, ResponseValue)>),
    /// No numerical response is licensed.
    Unidentified {
        /// Stable certificate/diagnostic id.
        certificate: Arc<str>,
    },
}

/// Complete causal-response artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalResponse {
    /// Requested estimand.
    pub estimand: ResponseFunctional,
    /// Structural identification status.
    pub identification_status: IdentificationStatus,
    /// Identified numerical payload.
    pub estimate: ResponseIdentification,
    /// Statistical uncertainty.
    pub uncertainty: ResponseUncertainty,
    /// Empirical support evidence.
    pub support: SupportReport,
    /// Explicit assumptions.
    pub assumptions: AssumptionSet,
    /// Stable provenance operation id.
    pub provenance_id: Arc<str>,
}

#[cfg(test)]
mod tests {
    use super::IdentifiedSet;

    #[test]
    fn identified_set_intersection_never_widens() {
        let a = IdentifiedSet::try_new(-1.0, 3.0).unwrap();
        let b = IdentifiedSet::try_new(0.0, 2.0).unwrap();
        assert_eq!(a.intersect(&b), Some(b));
        assert!(a.intersect(&IdentifiedSet::try_new(4.0, 5.0).unwrap()).is_none());
    }
}
