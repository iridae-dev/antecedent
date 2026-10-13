//! Numerics for inference from adaptively collected data.
//!
//! Four methods, one per inferential paradigm, each over a different part of
//! the retained assignment history. None of them analyses the final sample as
//! if the design had been fixed, and none reports a calibrated interval: every
//! interval-like number is labelled uncalibrated by its caller and the
//! interval route stays closed until whole-procedure coverage is measured.
//!
//! * [`beta_bernoulli`]: the conjugate posterior of Bernoulli arms, the
//!   posterior probability that an arm is best and the posterior expected loss
//!   of choosing an arm. The posterior is a function of the observed counts
//!   alone, so it is the same under any assignment rule that depends on the
//!   observed history and independent randomization only (the likelihood
//!   principle); that premise is the caller's to state.
//! * [`weighting`]: augmented inverse-probability scores with a *predictable*
//!   outcome model and predictable weights, combined by an adaptively weighted
//!   mean (Hadad, Hirshberg, Zhan, Wager and Athey 2021). Each score is a
//!   martingale difference under the assignment history, so the weighted sum
//!   is a martingale however the policy adapts.
//! * [`confidence_sequence`]: a time-uniform normal-mixture Hoeffding
//!   confidence sequence for the mean of a `[0, 1]`-bounded arm (Robbins 1970;
//!   Howard, Ramdas, `McAuliffe` and Sekhon 2021), valid at every sample size
//!   at once and so under data-dependent stopping.
//! * [`group_sequential`]: alpha-spending boundaries (Lan and `DeMets`) for a
//!   pre-specified set of interim looks, by recursive numerical integration of
//!   the canonical joint distribution of the interim statistics (Armitage,
//!   `McPherson` and Rowe 1969; Lan and `DeMets` 1983; Jennison and Turnbull 2000).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod beta_bernoulli;
pub mod confidence_sequence;
pub mod group_sequential;
pub mod weighting;

use thiserror::Error;

pub use beta_bernoulli::{
    BetaPosterior, MAX_ARM_TRIALS, MAX_BANDIT_ARMS, best_arm_probabilities, expected_losses,
    probability_superior, regularized_incomplete_beta,
};
pub use confidence_sequence::{
    ArmSequence, MAX_SEQUENCE_OBSERVATIONS, bounded_mean_confidence_sequence, contrast_bounds,
    mixture_hoeffding_radius,
};
pub use group_sequential::{
    Boundaries, MAX_LOOKS, Sidedness, SpendingFunction, boundaries, crossing_probabilities,
    first_crossing,
};
pub use weighting::{
    AdaptiveEstimate, Estimand, MAX_ADAPTIVE_UNITS, MAX_WEIGHTED_ARMS, OutcomeModelPrior,
    WeightScheme, adaptive_scores, weighted_estimate,
};

/// Why an adaptive-inference routine refused its input.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum AdaptiveInferenceError {
    /// Lengths or counts do not agree, or a bound is exceeded.
    #[error("adaptive inference shape error: {0}")]
    Shape(&'static str),
    /// A parameter is outside its range or not finite.
    #[error("adaptive inference parameter is invalid: {0}")]
    InvalidParameter(&'static str),
    /// An arm of the estimand had assignment probability below the declared
    /// floor at a unit, so its inverse-probability score is not supported.
    #[error(
        "arm {arm} had assignment probability {probability} below the floor {floor} at unit {unit}"
    )]
    Positivity {
        /// Unit (enrollment order) at which the floor failed.
        unit: usize,
        /// Arm index.
        arm: usize,
        /// The probability the policy gave the arm.
        probability: f64,
        /// The declared floor.
        floor: f64,
    },
    /// A size or work bound was exceeded.
    #[error("adaptive inference limit exceeded: {0}")]
    Limit(&'static str),
    /// An iteration or root search did not converge or left its bracket.
    #[error("adaptive inference numerical failure: {0}")]
    Numerical(&'static str),
    /// The caller cancelled the computation.
    #[error("adaptive inference was cancelled")]
    Cancelled,
}

/// Number of Gauss-Legendre nodes per integration panel.
pub(crate) const PANEL_NODES: usize = 12;

/// Gauss-Legendre rule shared by the panels of the quadrature routines.
pub(crate) fn panel_rule() -> (Vec<f64>, Vec<f64>) {
    crate::special::gauss_legendre(PANEL_NODES)
}

/// Nodes and weights of a composite Gauss-Legendre rule on `[lo, hi]` with
/// `panels` equal panels.
pub(crate) fn composite_rule(
    rule: &(Vec<f64>, Vec<f64>),
    lo: f64,
    hi: f64,
    panels: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(panels * rule.0.len());
    let mut weights = Vec::with_capacity(panels * rule.0.len());
    let width = (hi - lo) / panels as f64;
    for panel in 0..panels {
        let left = lo + width * panel as f64;
        let centre = left + 0.5 * width;
        for (u, w) in rule.0.iter().zip(&rule.1) {
            nodes.push(centre + 0.5 * width * u);
            weights.push(0.5 * width * w);
        }
    }
    (nodes, weights)
}
