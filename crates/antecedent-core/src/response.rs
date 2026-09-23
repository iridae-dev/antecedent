//! Causal-response result and support vocabulary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{AssumptionSet, Diagnostic, IdentificationStatus, ResponseFunctional, TemporalNodeKey};

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

/// What an interval's level means: repeated-sampling coverage or posterior probability.
///
/// A `Credible` interval holds `level` of the posterior mass given the model and prior;
/// it makes no frequentist coverage claim, and its "standard error" is a posterior
/// standard deviation. The two are never interchangeable, so every interval-bearing
/// [`ResponseUncertainty`] variant carries the tag rather than leaving it to free text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IntervalInterpretation {
    /// Frequentist confidence interval (sampling coverage).
    Confidence,
    /// Bayesian credible interval (posterior probability).
    Credible,
}

impl IntervalInterpretation {
    /// Stable lowercase name used on the wire and in the Python API.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Confidence => "confidence",
            Self::Credible => "credible",
        }
    }
}

/// The posterior draws a credible interval's endpoints were read from.
///
/// A Bayesian response publishes its interval as equal-tailed quantiles of a
/// finite draw vector, at the level the caller asked for. Retaining that
/// vector lets a consumer re-summarize the same posterior at another level
/// (or by another rule) without a second execution: the endpoints on the
/// enclosing [`ResponseUncertainty`] are exactly the quantiles of these
/// values, *after* every transform the estimator applies before its quantile
/// step (a point derivative's robust bias correction, an additive-GAM band's
/// `sqrt(n / (n − edf))` spread inflation), so a re-summarization at the
/// published level reproduces the published endpoints bit for bit.
///
/// In-process only: the draws are not part of the wire or artifact encoding
/// of a response (`antecedent-io` drops them and decodes `None`), which stays
/// unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct CredibleDraws {
    /// Draws per coordinate.
    pub n_draws: usize,
    /// Column-major `[n_coordinates × n_draws]`: coordinate `j`'s draws are
    /// `values[j · n_draws .. (j + 1) · n_draws]`, in the order the estimator
    /// produced them (unsorted).
    pub values: Arc<[f64]>,
}

impl CredibleDraws {
    /// Retain one coordinate's draw vector.
    #[must_use]
    pub fn scalar(values: impl Into<Arc<[f64]>>) -> Self {
        let values = values.into();
        Self { n_draws: values.len(), values }
    }

    /// Retain `n_draws` draws of each of several coordinates, given
    /// coordinate-major.
    ///
    /// # Panics
    ///
    /// When a coordinate's vector is not `n_draws` long.
    #[must_use]
    pub fn columns(n_draws: usize, columns: &[Vec<f64>]) -> Self {
        let mut values = Vec::with_capacity(n_draws * columns.len());
        for column in columns {
            assert_eq!(column.len(), n_draws, "every coordinate retains n_draws draws");
            values.extend_from_slice(column);
        }
        Self { n_draws, values: Arc::from(values) }
    }

    /// Number of coordinates.
    #[must_use]
    pub fn n_coordinates(&self) -> usize {
        if self.n_draws == 0 { 0 } else { self.values.len() / self.n_draws }
    }

    /// Coordinate `j`'s draws, or `None` past the last coordinate.
    #[must_use]
    pub fn column(&self, j: usize) -> Option<&[f64]> {
        (j < self.n_coordinates()).then(|| &self.values[j * self.n_draws..(j + 1) * self.n_draws])
    }
}

/// Statistical uncertainty kind. Pointwise and simultaneous bands are never aliases.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseUncertainty {
    /// No uncertainty was requested/available.
    None,
    /// Scalar standard error and interval.
    Scalar {
        /// Standard error; the posterior standard deviation when `interpretation` is
        /// [`IntervalInterpretation::Credible`].
        standard_error: f64,
        /// Confidence/credible level.
        level: f64,
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
        /// Whether the interval is a confidence or a credible interval.
        interpretation: IntervalInterpretation,
        /// The one-coordinate draw vector `[lower, upper]` are the quantiles of,
        /// when the interval is credible and taken from retained draws.
        draws: Option<CredibleDraws>,
    },
    /// Per-coordinate intervals without simultaneous coverage semantics.
    PointwiseBand {
        /// Confidence/credible level for each coordinate.
        level: f64,
        /// Lower values.
        lower: Arc<[f64]>,
        /// Upper values.
        upper: Arc<[f64]>,
        /// Whether the band is confidence or credible.
        interpretation: IntervalInterpretation,
        /// One draw column per coordinate of `lower`/`upper`, when the band is
        /// credible and taken from retained draws.
        draws: Option<CredibleDraws>,
    },
    /// One band calibrated for simultaneous coverage over the requested grid.
    SimultaneousBand {
        /// Simultaneous confidence/credible level.
        level: f64,
        /// Lower values.
        lower: Arc<[f64]>,
        /// Upper values.
        upper: Arc<[f64]>,
        /// Resampling replicates (or posterior draws) used.
        replicates: u32,
        /// Whether the band is confidence or credible.
        interpretation: IntervalInterpretation,
    },
    /// Confidence band around an identified envelope (not the envelope itself).
    IdentifiedEnvelopeBand {
        /// Confidence/credible level.
        level: f64,
        /// Lower limit for the identified lower envelope.
        lower_outer: Arc<[f64]>,
        /// Upper limit for the identified upper envelope.
        upper_outer: Arc<[f64]>,
        /// Whether the limits are confidence or credible.
        interpretation: IntervalInterpretation,
    },
    /// Posterior draws or summaries live in a referenced posterior artifact.
    Posterior {
        /// Artifact identifier.
        artifact_id: Arc<str>,
    },
}

impl ResponseUncertainty {
    /// The retained draws behind a credible scalar interval or pointwise band.
    #[must_use]
    pub fn credible_draws(&self) -> Option<&CredibleDraws> {
        match self {
            Self::Scalar { draws, .. } | Self::PointwiseBand { draws, .. } => draws.as_ref(),
            _ => None,
        }
    }
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

/// Identification products for one requested horizon of a temporal response.
///
/// Parallel to [`crate::TemporalResponseSpec::horizons`]. A union of these
/// adjustment sets is not itself a valid adjustment set.
#[derive(Clone, Debug, PartialEq)]
pub struct HorizonIdentification {
    /// Requested horizon (same units as the response spec).
    pub horizon: u32,
    /// Structural identification status at this horizon.
    pub status: IdentificationStatus,
    /// Identifier method id (typically `temporal.backdoor.unfolded`).
    pub method: Arc<str>,
    /// Template-level adjustment nodes `(variable, offset)` for this horizon.
    pub adjustment: Arc<[TemporalNodeKey]>,
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
    /// Per-horizon identification on a temporal surface; absent on static curves.
    pub horizon_identification: Option<Arc<[HorizonIdentification]>>,
    /// Additive joint g-computation makes the interaction contrast structurally zero.
    pub interaction_structurally_zero: bool,
}

#[cfg(test)]
mod tests {
    use super::{CredibleDraws, IdentifiedSet, IntervalInterpretation, ResponseUncertainty};

    #[test]
    fn credible_draws_are_column_major_per_coordinate() {
        let draws = CredibleDraws::columns(3, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(draws.n_draws, 3);
        assert_eq!(draws.n_coordinates(), 2);
        assert_eq!(draws.column(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(draws.column(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(draws.column(2), None);
        let scalar = CredibleDraws::scalar(vec![0.5, -0.5]);
        assert_eq!((scalar.n_draws, scalar.n_coordinates()), (2, 1));
        assert_eq!(scalar.column(0), Some(&[0.5, -0.5][..]));
    }

    #[test]
    fn credible_draws_are_read_only_off_scalar_and_pointwise_uncertainty() {
        let band = ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: [0.0].into(),
            upper: [1.0].into(),
            interpretation: IntervalInterpretation::Credible,
            draws: Some(CredibleDraws::scalar(vec![0.0, 1.0])),
        };
        assert_eq!(band.credible_draws().map(|d| d.n_draws), Some(2));
        assert!(ResponseUncertainty::None.credible_draws().is_none());
        assert!(
            ResponseUncertainty::Posterior { artifact_id: "p".into() }.credible_draws().is_none()
        );
    }

    #[test]
    fn identified_set_intersection_never_widens() {
        let a = IdentifiedSet::try_new(-1.0, 3.0).unwrap();
        let b = IdentifiedSet::try_new(0.0, 2.0).unwrap();
        assert_eq!(a.intersect(&b), Some(b));
        assert!(a.intersect(&IdentifiedSet::try_new(4.0, 5.0).unwrap()).is_none());
    }
}
