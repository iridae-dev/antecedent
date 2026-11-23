/// Why a bounded probability interval could not be formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbabilityIntervalUnavailable {
    /// The plug-in probability is exactly 0 or 1 (or not a probability): its
    /// logit is infinite and every bootstrap replicate sits on the same
    /// boundary, so no sampling spread is observed to build an interval from.
    BoundaryEstimate,
    /// Fewer than two usable bootstrap replicates, or more than half failed.
    BootstrapUnavailable,
    /// Every usable replicate reproduced the point value (zero spread).
    ZeroSpread,
}

impl ProbabilityIntervalUnavailable {
    /// Stable machine-readable reason.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BoundaryEstimate => "boundary_estimate",
            Self::BootstrapUnavailable => "bootstrap_unavailable",
            Self::ZeroSpread => "zero_spread",
        }
    }
}

/// Two-sided interval for one interventional probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbabilityInterval {
    /// Interval inside `[0, 1]` at nominal two-sided `level`.
    Bounded {
        /// Nominal two-sided coverage.
        level: f64,
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
    },
    /// No interval could be formed honestly.
    Unavailable(ProbabilityIntervalUnavailable),
}

impl ProbabilityInterval {
    /// `(lower, upper)` when the interval exists.
    #[must_use]
    pub const fn bounds(&self) -> Option<(f64, f64)> {
        match *self {
            Self::Bounded { lower, upper, .. } => Some((lower, upper)),
            Self::Unavailable(_) => None,
        }
    }
}

/// Frequentist bootstrap uncertainty of one [`DistributionAtom`](super::DistributionAtom).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomUncertainty {
    /// Bootstrap SE of the atom probability (probability scale).
    pub se_bootstrap: Option<f64>,
    /// Replicates that produced a value for this atom.
    pub replicates_ok: u32,
    /// Bounded interval at the estimator's confidence level.
    pub interval: ProbabilityInterval,
}

/// Logit-scale delta-method interval for a probability from its bootstrap SE:
/// `antecedent_kernels::sigmoid(logit(p̂) ± z·se / (p̂(1 − p̂)))`, `z = Φ⁻¹(1/2 + level/2)`.
///
/// Why this construction: a probability's sampling law is skewed toward the
/// interior near 0 and 1, so the symmetric `p̂ ± z·se` both leaves `[0, 1]` and
/// under-covers on the short side. On the logit scale the law is close to
/// symmetric, and `se / (p̂(1 − p̂))` is the delta-method SE of `logit(p̂)`
/// (the same transformation that makes the logit Wald interval for a binomial
/// proportion behave well at small counts). The back-transformed endpoints lie
/// strictly inside `(0, 1)` and the interval widens on the interior side. It
/// reuses the bootstrap SE that is already published, so the scalar SE and the
/// interval describe the same replicates; a percentile interval would need the
/// replicate probabilities to be retained and inherits the Wald-like
/// short-side under-coverage of a discrete proportion.
///
/// At `p̂ ∈ {0, 1}` the logit is infinite and the bootstrap spread is
/// degenerate, so the interval is [`ProbabilityIntervalUnavailable::BoundaryEstimate`]
/// rather than a zero-width or clamped fake.
#[must_use]
pub fn logit_probability_interval(p: f64, se: Option<f64>, level: f64) -> ProbabilityInterval {
    if !(p > 0.0 && p < 1.0) {
        return ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::BoundaryEstimate);
    }
    let Some(se) = se.filter(|s| s.is_finite() && *s >= 0.0) else {
        return ProbabilityInterval::Unavailable(
            ProbabilityIntervalUnavailable::BootstrapUnavailable,
        );
    };
    if se == 0.0 {
        return ProbabilityInterval::Unavailable(ProbabilityIntervalUnavailable::ZeroSpread);
    }
    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
    if !z.is_finite() || z <= 0.0 {
        return ProbabilityInterval::Unavailable(
            ProbabilityIntervalUnavailable::BootstrapUnavailable,
        );
    }
    let centre = (p / (1.0 - p)).ln();
    let half = z * se / (p * (1.0 - p));
    ProbabilityInterval::Bounded {
        level,
        lower: antecedent_kernels::sigmoid(centre - half),
        upper: antecedent_kernels::sigmoid(centre + half),
    }
}
