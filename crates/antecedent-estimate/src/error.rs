//! Estimation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::QueryError;
use antecedent_data::DataError;
use antecedent_prob::ProbError;
use antecedent_stats::{GlmRefusalKind, StatsError};
use std::borrow::Cow;

use thiserror::Error;

/// Estimation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EstimationError {
    /// Data/schema issue.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Stats backend.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// Probability / posterior backend.
    #[error(transparent)]
    Prob(#[from] ProbError),
    /// Query validation failed (`AverageEffectQuery::validate`, …).
    #[error(transparent)]
    Query(#[from] QueryError),
    /// Missing overlap override when required.
    #[error("{message}")]
    Overlap {
        /// Message.
        message: &'static str,
    },
    /// [`Self::Overlap`] raised on a fitted propensity that also carries structured
    /// diagnostics (see [`RefusalFields`]); the rendered message is that of the
    /// [`Self::Overlap`] it replaces.
    #[error("{message}")]
    OverlapWithFields {
        /// Message.
        message: &'static str,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
    /// Incompatible estimand.
    #[error("{message}")]
    IncompatibleEstimand {
        /// Message.
        message: &'static str,
    },
    /// Effect modifiers not supported on this estimator path.
    #[error("effect modifiers are not supported on this estimator path")]
    EffectModifiers,
    /// Target population not supported on this estimator path: a refusal with
    /// the registered `population_not_estimable` reason code.
    #[error(
        "{}{}: only TargetPopulation::AllObserved is supported on this estimator path",
        antecedent_core::reason_code::PREFIX,
        antecedent_core::reason_code!("population_not_estimable")
    )]
    TargetPopulation,
    /// Query options unsupported by this estimator (fixed message).
    #[error("{message}")]
    Unsupported {
        /// Explanation.
        message: &'static str,
    },
    /// [`Self::Unsupported`] that also names what the caller can change to
    /// proceed. The rendered message is the refusal alone, byte-identical to the
    /// [`Self::Unsupported`] it replaces; the remedy is a separate structured
    /// field read with [`Self::remedy`], never parsed out of the message. It
    /// carries no reason code, as [`Self::Unsupported`] does not.
    #[error("{message}")]
    UnsupportedWithRemedy {
        /// Explanation (the refusal).
        message: &'static str,
        /// What the caller can do instead.
        remedy: &'static str,
    },
    /// [`Self::Unsupported`] that also carries structured diagnostics (see
    /// [`RefusalFields`]). The rendered message is byte-identical to the
    /// [`Self::Unsupported`] with the same message and, as there, no reason code is
    /// attached; the fields are read with [`Self::refusal_fields`].
    #[error("{message}")]
    UnsupportedWithFields {
        /// Explanation (the refusal).
        message: &'static str,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
    /// The estimator refused because identification was not certified for this query.
    ///
    /// Distinct from [`Self::Data`]: the inputs are well formed, and the refusal is about
    /// what the graph licenses. Owned rather than `&'static str` because it carries the
    /// certificate's own reason and message.
    #[error("{message}")]
    NotCertified {
        /// Refusal, including the certificate's reason and message.
        message: String,
    },
    /// A banked posterior's coefficient count does not match the target design's, so
    /// its coefficients cannot be mapped one-to-one onto a prior. Callers that treat
    /// "not this mechanism's prior" as a fallback match this variant, never its text;
    /// the message still carries the registered `prior_dimension_mismatch` reason so a
    /// caller with no fallback (the prior transfer is terminal) surfaces a coded refusal.
    #[error(
        "{}prior_dimension_mismatch: posterior coefficient dimension {posterior} != expected n_coef {design}",
        antecedent_core::reason_code::PREFIX
    )]
    PriorDimensionMismatch {
        /// Coefficients in the banked posterior.
        posterior: usize,
        /// Coefficients in the target design.
        design: usize,
    },
    /// A resampled empirical law received no rows, so it has no defined table. Bootstrap
    /// loops count this as a failed replicate; anywhere else it is an ordinary error.
    #[error("{message}")]
    EmptyEmpiricalSample {
        /// Located description of the empty sample.
        message: String,
    },
    /// A refusal carrying a registered runtime reason code
    /// (`parity/reason_codes.toml`), rendered `reason=<code>: <message>` so every
    /// boundary reads the code the same way.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    Refused {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// What was refused and what the caller can do instead.
        message: String,
    },
    /// [`Self::Refused`] that also carries structured diagnostics (see
    /// [`RefusalFields`]). The rendered message is byte-identical to the
    /// [`Self::Refused`] with the same code and message; the fields are read with
    /// [`Self::refusal_fields`], never parsed out of the text.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    RefusedWithFields {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// What was refused and what the caller can do instead.
        message: String,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
}

/// A float compared by its bit pattern, so a refusal that carries diagnostics stays
/// `Eq` (NaN equals the same NaN; `0.0` and `-0.0` differ).
#[derive(Clone, Copy, Debug)]
pub struct ExactF64(pub f64);

impl PartialEq for ExactF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for ExactF64 {}

/// Structured diagnostics of a refusal: which treatment or cell failed, at which stage
/// and why, plus whatever numbers the failing step actually produced.
///
/// Every entry is optional or empty by default and stays that way unless the failing
/// step measured it. A diagnostic that needs a fitted nuisance is never invented from
/// a fit that failed before producing one. Build one with
/// `RefusalFields { stage: .., ..RefusalFields::default() }`; later fields are added
/// only as further optional or empty entries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RefusalFields {
    /// Pipeline stage that refused (`preflight`, `rank_drop`, ...).
    pub stage: Option<String>,
    /// Failing treatment or joint cell, by name.
    pub subject: Option<String>,
    /// Short statement of why it failed.
    pub reason: Option<String>,
    /// Effective sample size per arm, `(arm label, ess)`.
    pub arm_ess: Vec<(String, ExactF64)>,
    /// Smallest fitted propensity.
    pub propensity_min: Option<ExactF64>,
    /// Largest fitted propensity.
    pub propensity_max: Option<ExactF64>,
    /// Fitted propensity quantiles, `(probability, value)`.
    pub propensity_quantiles: Vec<(ExactF64, ExactF64)>,
    /// Number of clusters, when the estimator clusters.
    pub cluster_count: Option<u64>,
    /// Numerical rank of the design.
    pub numerical_rank: Option<u64>,
    /// Number of design columns the rank is out of.
    pub design_columns: Option<u64>,
    /// Columns the refusal implicates, by name.
    pub implicated_columns: Vec<String>,
    /// What the caller can change to get past the refusal.
    pub remedy: Option<String>,
    /// IRLS iterations of the refused GLM fit.
    pub glm_iterations: Option<u64>,
    /// Smallest `min(mu, 1 - mu)` over the refused binomial fit's rows (distance of the
    /// closest fitted probability to 0 or 1), at the returned coefficients.
    pub boundary_margin: Option<ExactF64>,
    /// Rows of the refused binomial fit whose fitted probability is within `1e-8` of 0 or 1.
    pub boundary_count: Option<u64>,
    /// Fewest clusters the refused cluster-robust variance needs (the failing count is
    /// [`Self::cluster_count`]).
    pub cluster_minimum: Option<u64>,
}

impl EstimationError {
    /// Fixed unsupported query option.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::Unsupported { message }
    }

    /// Fixed unsupported query option that names a remedy (see
    /// [`Self::UnsupportedWithRemedy`]).
    #[must_use]
    pub const fn unsupported_with_remedy(message: &'static str, remedy: &'static str) -> Self {
        Self::UnsupportedWithRemedy { message, remedy }
    }

    /// What the caller can change to get past this refusal, when the refusal
    /// names one. Optional and additive: `None` for every error that names no
    /// remedy, and the rendered message never includes it.
    #[must_use]
    pub const fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::UnsupportedWithRemedy { remedy, .. } => Some(remedy),
            _ => None,
        }
    }

    /// Refusal to estimate a quantity whose identification was not certified.
    #[must_use]
    pub fn not_certified(stage: &str, reason: &str, message: &str) -> Self {
        Self::NotCertified {
            message: format!(
                "{stage} refused: identification was not certified ({reason}): {message}"
            ),
        }
    }

    /// A refusal with a registered runtime reason code.
    #[must_use]
    pub fn refused(code: &'static str, message: impl Into<String>) -> Self {
        Self::Refused { code, message: message.into() }
    }

    /// A refusal with a registered runtime reason code and structured diagnostics.
    #[must_use]
    pub fn refused_with_fields(
        code: &'static str,
        message: impl Into<String>,
        fields: RefusalFields,
    ) -> Self {
        Self::RefusedWithFields { code, message: message.into(), fields: Box::new(fields) }
    }

    /// Structured diagnostics of a refusal that carries them: one built with
    /// [`Self::refused_with_fields`] or [`Self::UnsupportedWithFields`], or a stats-layer
    /// refusal whose own record holds them (a refused GLM fit, a rank-deficient design, too
    /// few clusters), read from that record and never parsed out of the message. Additive:
    /// `None` for every other error, and never part of the rendered message. A diagnostic the
    /// failing step did not measure stays absent.
    #[must_use]
    pub fn refusal_fields(&self) -> Option<Cow<'_, RefusalFields>> {
        match self {
            Self::RefusedWithFields { fields, .. }
            | Self::UnsupportedWithFields { fields, .. }
            | Self::OverlapWithFields { fields, .. } => Some(Cow::Borrowed(fields)),
            Self::Stats(error) => stats_refusal_fields(error).map(Cow::Owned),
            _ => None,
        }
    }

    /// [`Self::unsupported`] with structured diagnostics.
    #[must_use]
    pub fn unsupported_with_fields(message: &'static str, fields: RefusalFields) -> Self {
        Self::UnsupportedWithFields { message, fields: Box::new(fields) }
    }

    /// Ad-hoc data-layer message (maps to [`DataError::InvalidArgument`]).
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// Ad-hoc stats-layer message (maps to [`StatsError::Backend`]).
    #[must_use]
    pub fn stats_msg(message: impl Into<String>) -> Self {
        Self::Stats(StatsError::Backend(message.into()))
    }
}

/// The structured diagnostics a stats-layer refusal records: stage, reason, remedy and the
/// numbers its own failing step produced. A subject (treatment or cell) is not known at this
/// layer and stays absent; so does every figure the fit did not measure.
fn stats_refusal_fields(error: &StatsError) -> Option<RefusalFields> {
    match error {
        StatsError::GlmRefused { kind, iterations, boundary_count, .. } => {
            let (reason, remedy) = match kind {
                GlmRefusalKind::NonConverged => (
                    "non_converged",
                    "standardize or drop near-collinear columns, or raise the iteration limit",
                ),
                GlmRefusalKind::Separated => (
                    "separated",
                    "drop or coarsen the separating columns, or use a penalized nuisance",
                ),
                GlmRefusalKind::BoundarySaturated => (
                    "boundary_saturated",
                    "drop or coarsen the near-deterministic columns, or use a penalized nuisance",
                ),
            };
            Some(RefusalFields {
                stage: Some("glm_fit".to_string()),
                reason: Some(reason.to_string()),
                remedy: Some(remedy.to_string()),
                glm_iterations: Some(u64::from(*iterations)),
                boundary_margin: error.glm_boundary_margin().map(ExactF64),
                boundary_count: *boundary_count,
                ..RefusalFields::default()
            })
        }
        StatsError::RankDeficient { rank, ncols } => Some(RefusalFields {
            stage: Some("design_rank".to_string()),
            reason: Some("rank_deficient".to_string()),
            remedy: Some(
                "run preflight to name the dependent columns, then drop them or use \
                 estimate_with_rank_drop"
                    .to_string(),
            ),
            numerical_rank: Some(u64::try_from(*rank).unwrap_or(u64::MAX)),
            design_columns: Some(u64::try_from(*ncols).unwrap_or(u64::MAX)),
            ..RefusalFields::default()
        }),
        StatsError::FewClusters { clusters, minimum, .. } => Some(RefusalFields {
            stage: Some("cluster_variance".to_string()),
            reason: Some("too_few_clusters".to_string()),
            remedy: Some(
                "supply more clusters, coarser-grained dependence, or a non-clustered standard \
                 error"
                    .to_string(),
            ),
            cluster_count: Some(*clusters),
            cluster_minimum: Some(*minimum),
            ..RefusalFields::default()
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{EstimationError, RefusalFields};

    /// Only a refusal built with a remedy carries one; every other error reads
    /// `None`, and attaching a remedy never changes the rendered message.
    #[test]
    fn remedy_is_absent_unless_named_and_never_changes_the_message() {
        let plain = EstimationError::unsupported("refused");
        assert_eq!(plain.remedy(), None);
        assert_eq!(EstimationError::refused("route_not_supported", "no").remedy(), None);
        assert_eq!(EstimationError::stats_msg("backend").remedy(), None);
        assert_eq!(EstimationError::TargetPopulation.remedy(), None);
        assert_eq!(EstimationError::not_certified("s", "r", "m").remedy(), None);
        let remedied = EstimationError::unsupported_with_remedy("refused", "do this instead");
        assert_eq!(remedied.remedy(), Some("do this instead"));
        assert_eq!(remedied.to_string(), plain.to_string());
    }

    /// Structured fields ride beside a refusal without changing its code or message, and
    /// a plain refusal reads no fields.
    #[test]
    fn refusal_fields_are_additive_to_the_coded_message() {
        let plain = EstimationError::refused("route_not_supported", "no");
        assert!(plain.refusal_fields().is_none());
        let fields = RefusalFields {
            numerical_rank: Some(174),
            implicated_columns: vec!["z174".to_string()],
            ..RefusalFields::default()
        };
        let rich = EstimationError::refused_with_fields("route_not_supported", "no", fields);
        assert_eq!(rich.to_string(), plain.to_string());
        let read = rich.refusal_fields().expect("fields were attached");
        assert_eq!(read.numerical_rank, Some(174));
        assert!(read.stage.is_none() && read.arm_ess.is_empty() && read.propensity_min.is_none());
    }
}
