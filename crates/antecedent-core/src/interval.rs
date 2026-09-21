//! Closed interval-method vocabulary recorded on a compiled claim.

/// How an execution formed its reported interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IntervalMethod {
    /// Analytic standard error.
    AnalyticSe,
    /// Bootstrap standard error.
    BootstrapSe,
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
}

impl IntervalMethod {
    /// Closed set of methods. A new variant fails calibration tests until listed.
    pub const ALL: [IntervalMethod; 9] = [
        Self::AnalyticSe,
        Self::BootstrapSe,
        Self::PosteriorQuantile,
        Self::UnitPosteriorQuantile,
        Self::IdentifiedSet,
        Self::CircularBlockSe,
        Self::SimultaneousBand,
        Self::AndersonRubin,
        Self::None,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnalyticSe => "analytic_se",
            Self::BootstrapSe => "bootstrap_se",
            Self::PosteriorQuantile => "posterior_quantile",
            Self::UnitPosteriorQuantile => "unit_posterior_quantile",
            Self::IdentifiedSet => "identified_set",
            Self::CircularBlockSe => "circular_block_se",
            Self::SimultaneousBand => "simultaneous_band",
            Self::AndersonRubin => "anderson_rubin",
            Self::None => "none",
        }
    }
}
