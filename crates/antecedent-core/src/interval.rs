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
    /// Identified-set interval.
    IdentifiedSet,
    /// Circular-block standard error.
    CircularBlockSe,
    /// Simultaneous band.
    SimultaneousBand,
    /// No interval was formed.
    None,
}

impl IntervalMethod {
    /// Closed set of methods. A new variant fails calibration tests until listed.
    pub const ALL: [IntervalMethod; 7] = [
        Self::AnalyticSe,
        Self::BootstrapSe,
        Self::PosteriorQuantile,
        Self::IdentifiedSet,
        Self::CircularBlockSe,
        Self::SimultaneousBand,
        Self::None,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnalyticSe => "analytic_se",
            Self::BootstrapSe => "bootstrap_se",
            Self::PosteriorQuantile => "posterior_quantile",
            Self::IdentifiedSet => "identified_set",
            Self::CircularBlockSe => "circular_block_se",
            Self::SimultaneousBand => "simultaneous_band",
            Self::None => "none",
        }
    }
}
