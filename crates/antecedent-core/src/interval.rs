//! Closed interval-method vocabulary recorded on a compiled claim.

/// How an execution formed its reported interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IntervalMethod {
    /// Analytic standard error.
    AnalyticSe,
    /// Bootstrap standard error.
    BootstrapSe,
    /// Equal-tailed interval formed from the empirical bootstrap quantiles.
    PercentileBootstrap,
    /// Posterior quantile interval.
    PosteriorQuantile,
    /// Per-unit posterior quantile interval: equal-tailed quantiles of one
    /// observed unit's effect draws (counterfactual `unit_effect_intervals`).
    /// Kept apart from [`Self::PosteriorQuantile`] because it covers a
    /// different target — each unit's contrast, not the scalar average — so a
    /// coverage record for one is no evidence for the other.
    UnitPosteriorQuantile,
    /// Identified-set interval.
    IdentifiedSet,
    /// Circular-block standard error.
    CircularBlockSe,
    /// Simultaneous band.
    SimultaneousBand,
    /// Anderson–Rubin confidence set. Endpoints are not a standard error:
    /// they may be infinite, and the set is not `estimate ± z·SE`.
    AndersonRubin,
    /// No interval was formed.
    None,
    /// Symmetric credible interval formed from posterior absolute deviations about the estimate.
    PosteriorAbsoluteDeviation,
}

impl IntervalMethod {
    /// Closed set of methods. A new variant fails calibration tests until listed.
    pub const ALL: [IntervalMethod; 11] = [
        Self::AnalyticSe,
        Self::BootstrapSe,
        Self::PercentileBootstrap,
        Self::PosteriorQuantile,
        Self::UnitPosteriorQuantile,
        Self::IdentifiedSet,
        Self::CircularBlockSe,
        Self::SimultaneousBand,
        Self::AndersonRubin,
        Self::None,
        Self::PosteriorAbsoluteDeviation,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnalyticSe => "analytic_se",
            Self::BootstrapSe => "bootstrap_se",
            Self::PercentileBootstrap => "percentile_bootstrap",
            Self::PosteriorQuantile => "posterior_quantile",
            Self::PosteriorAbsoluteDeviation => "posterior_absolute_deviation",
            Self::UnitPosteriorQuantile => "unit_posterior_quantile",
            Self::IdentifiedSet => "identified_set",
            Self::CircularBlockSe => "circular_block_se",
            Self::SimultaneousBand => "simultaneous_band",
            Self::AndersonRubin => "anderson_rubin",
            Self::None => "none",
        }
    }
}
