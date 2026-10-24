//! Stats-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::diagnostic::RouteNativeDiagnostic;
use thiserror::Error;

/// Why a GLM fit was refused by an estimation path ([`crate::GlmFit::require_ok`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmRefusalKind {
    /// The IRLS loop did not converge.
    NonConverged,
    /// (Quasi-)complete separation: no finite maximum-likelihood estimate.
    Separated,
    /// Fitted probabilities within `1e-8` of 0 or 1 on a converged, unseparated fit.
    BoundarySaturated,
}

/// Statistical / linear algebra errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum StatsError {
    /// Shape mismatch.
    #[error("shape error: {message}")]
    Shape {
        /// Context.
        message: &'static str,
    },
    /// Rank deficiency / singular design.
    #[error("rank deficient: rank={rank} ncols={ncols}")]
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Materially non-positive variance after inclusion–exclusion (not FP noise).
    #[error("non-positive variance: {message}")]
    NonPositiveVariance {
        /// Context.
        message: &'static str,
    },
    /// Requested option is not implemented on this code path.
    #[error("unsupported: {message}")]
    Unsupported {
        /// Context.
        message: &'static str,
    },
    /// A local-polynomial fit whose kernel-weighted design is singular: fewer
    /// distinct regressor values carry weight at the evaluation point than the
    /// polynomial has coefficients (`order + 1`).
    #[error("singular local response design of order {order}")]
    SingularLocalDesign {
        /// Polynomial order of the local fit.
        order: usize,
    },
    /// Backend failure.
    #[error("backend error: {0}")]
    Backend(String),
    /// The execution context's cancellation token fired before the computation finished.
    #[error("cancelled")]
    Cancelled,
    /// A planned allocation exceeds the execution context's hard memory limit.
    #[error("resource limit: {message} needs {required_bytes} bytes, limit {limit_bytes}")]
    ResourceLimit {
        /// What was being planned.
        message: &'static str,
        /// Bytes the plan needs.
        required_bytes: u64,
        /// The hard limit in force.
        limit_bytes: u64,
    },
    /// A planned exact computation needs more work than the caller declared it may do. The
    /// quote is computed before the computation starts; nothing was allocated or run.
    #[error(
        "work limit: {message} needs {required_work} operations, limit {limit_work}; next \
         attempt: raise the work limit or request an approximation mode explicitly"
    )]
    WorkLimit {
        /// What was being planned.
        message: &'static str,
        /// Operations the exact computation needs.
        required_work: u64,
        /// The caller's declared ceiling.
        limit_work: u64,
    },
    /// A GLM fit an estimation path refuses. Renders exactly as the [`Self::Backend`] it
    /// replaces (same text), and also carries what the fit measured, so a caller reads the
    /// kind and the numbers instead of parsing the message. The margin is the float's bit
    /// pattern (this enum is `Eq`); read it with [`Self::glm_boundary_margin`].
    #[error("backend error: {message}")]
    GlmRefused {
        /// Which defect was refused.
        kind: GlmRefusalKind,
        /// The refusal text.
        message: &'static str,
        /// IRLS iterations the fit used.
        iterations: u32,
        /// Bits of the smallest `min(mu, 1 - mu)` over the rows; `None` when not measured.
        boundary_margin_bits: Option<u64>,
        /// Rows within `1e-8` of 0 or 1; `None` when not measured.
        boundary_count: Option<u64>,
    },
    /// Fewer clusters than a cluster-robust variance needs. Renders as `rendered`, the text
    /// of the error it replaces (which differs by call site), and carries the counts.
    #[error("{rendered}")]
    FewClusters {
        /// Clusters found (of the failing grouping, for a multiway subset).
        clusters: u64,
        /// Fewest clusters the variance needs.
        minimum: u64,
        /// Full rendered message.
        rendered: &'static str,
    },
    /// A route check failed. `message` is the refusal text; `diagnostic` is the record.
    #[error("{message}")]
    Route {
        /// Refusal text.
        message: &'static str,
        /// The route record. A failed check is not an unsupported option.
        diagnostic: RouteNativeDiagnostic,
    },
}

impl StatsError {
    /// Smallest fitted-probability margin of a refused GLM fit, when it was measured.
    #[must_use]
    pub fn glm_boundary_margin(&self) -> Option<f64> {
        match self {
            Self::GlmRefused { boundary_margin_bits, .. } => {
                boundary_margin_bits.map(f64::from_bits)
            }
            _ => None,
        }
    }

    /// Route record on this error, when it is a route outcome.
    #[must_use = "a failed route check is dropped if the record is discarded"]
    pub const fn route_diagnostic(&self) -> Option<RouteNativeDiagnostic> {
        match self {
            Self::Route { diagnostic, .. } => Some(*diagnostic),
            // A singular design is a failed solve, not a failed check. The error reports no
            // mass, so the mass is unreported.
            Self::RankDeficient { .. } | Self::SingularLocalDesign { .. } => {
                Some(RouteNativeDiagnostic::computational_failure(
                    antecedent_core::diagnostic::RouteMass::unreported(),
                ))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod singular_route_tests {
    use super::StatsError;

    #[test]
    fn a_singular_design_is_a_computational_failure_with_the_same_message() {
        let error = StatsError::RankDeficient { rank: 1, ncols: 2 };
        assert_eq!(error.to_string(), "rank deficient: rank=1 ncols=2");
        let record = error.route_diagnostic().expect("a singular design carries a record");
        assert!(record.is_computational_failure());
        assert!(record.unevaluable_mass().is_unreported());
        assert!(StatsError::Shape { message: "x" }.route_diagnostic().is_none());
    }
}
