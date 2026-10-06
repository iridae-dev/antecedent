//! Stats-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
}
