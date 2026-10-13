//! Group-sequential boundaries by alpha spending, and stage-wise crossing
//! probabilities, from the canonical joint distribution of the interim
//! statistics.
//!
//! With information fractions `t_1 < ... < t_K <= 1`, the standardized
//! statistics of a randomized comparison with independent increments are
//! jointly normal with `Z_k ~ N(theta sqrt(t_k), 1)` and
//! `Cov(Z_j, Z_k) = sqrt(t_j / t_k)` for `j <= k` (Jennison and Turnbull 2000,
//! ch. 3). A spending function `alpha*(t)` (Lan and `DeMets` 1983) fixes the
//! probability of first crossing at look `k` under `theta = 0` to be
//! `alpha*(t_k) - alpha*(t_{k-1})`. The boundaries are found by the recursive
//! integration of Armitage, `McPherson` and Rowe (1969): the sub-density of
//! `Z_k` on the continuation region is the convolution of the sub-density of
//! `Z_{k-1}` with the normal transition
//! `Z_k | Z_{k-1} = u ~ N(u sqrt(t_{k-1} / t_k), (t_k - t_{k-1}) / t_k)`.
//! The continuation probability of a candidate boundary is then a finite sum
//! over the previous stage's quadrature nodes of normal cumulative
//! probabilities, and the boundary solves a monotone equation by bisection.
//!
//! The spending functions are the O'Brien-Fleming-type function
//! `2 - 2 Phi(z_{alpha/2} / sqrt(t))` and the Pocock-type function
//! `alpha ln(1 + (e - 1) t)`. `alpha` is the total error spent by the rule:
//! for two-sided symmetric boundaries the probability of any crossing, for
//! one-sided boundaries the probability of an upper crossing.
//!
//! Information fractions are the actual fractions at which looks happen, so
//! unplanned look times spend alpha by the realized information (the
//! Lan-`DeMets` property). The guarantee is one of the independent-increment
//! model: a design whose assignment adapts to outcomes does not have
//! independent increments, and this module claims nothing for it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{norm_cdf, norm_inv, norm_pdf, norm_sf};

use super::{AdaptiveInferenceError as Error, composite_rule, panel_rule};

/// Most interim looks one rule accepts.
pub const MAX_LOOKS: usize = 20;

const LOWER_TRUNCATION: f64 = -9.0;
const PANEL_WIDTH: f64 = 0.5;
const BISECTIONS: usize = 100;
const BRACKET_HIGH: f64 = 12.0;
const BRACKET_LOW_ONE_SIDED: f64 = -8.5;
const BRACKET_WIDTH: f64 = 1e-14;

/// How alpha is spent over the information fractions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpendingFunction {
    /// `2 - 2 Phi(z_{alpha/2} / sqrt(t))`: little alpha early.
    OBrienFlemingType,
    /// `alpha ln(1 + (e - 1) t)`: alpha spread nearly evenly.
    PocockType,
}

impl SpendingFunction {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OBrienFlemingType => "obrien_fleming_type",
            Self::PocockType => "pocock_type",
        }
    }

    /// Cumulative alpha spent at information fraction `fraction` in `(0, 1]`
    /// for total error `alpha`.
    #[must_use]
    pub fn spend(self, fraction: f64, alpha: f64) -> f64 {
        match self {
            Self::OBrienFlemingType => {
                let z = norm_inv(1.0 - alpha / 2.0);
                2.0 * norm_sf(z / fraction.sqrt())
            }
            Self::PocockType => alpha * (1.0 + (std::f64::consts::E - 1.0) * fraction).ln(),
        }
    }
}

/// Which crossings stop the trial for efficacy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Sidedness {
    /// Stop when `Z_k >= c_k`.
    OneSidedUpper,
    /// Stop when `|Z_k| >= c_k`.
    TwoSidedSymmetric,
}

impl Sidedness {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OneSidedUpper => "one_sided_upper",
            Self::TwoSidedSymmetric => "two_sided_symmetric",
        }
    }
}

/// Efficacy boundaries and the alpha they spend.
#[derive(Clone, Debug, PartialEq)]
pub struct Boundaries {
    /// Critical value `c_k` on the standardized statistic at each look.
    pub critical: Vec<f64>,
    /// Cumulative alpha spent through each look.
    pub cumulative_spend: Vec<f64>,
    /// Alpha spent at each look alone.
    pub spend_increments: Vec<f64>,
}

fn check_fractions(fractions: &[f64]) -> Result<(), Error> {
    if fractions.is_empty() || fractions.len() > MAX_LOOKS {
        return Err(Error::Shape("a group-sequential rule has between 1 and 20 looks"));
    }
    let mut previous = 0.0;
    for &fraction in fractions {
        if !(fraction.is_finite() && fraction > previous && fraction <= 1.0) {
            return Err(Error::InvalidParameter(
                "information fractions must be strictly increasing in (0, 1]",
            ));
        }
        previous = fraction;
    }
    Ok(())
}

/// Continuation-region sub-density of the statistic at one look, on nodes.
struct Stage {
    fraction: f64,
    nodes: Vec<f64>,
    weights: Vec<f64>,
    density: Vec<f64>,
    mass: f64,
}

struct Recursion<'a> {
    fractions: &'a [f64],
    sidedness: Sidedness,
    drift: f64,
    rule: (Vec<f64>, Vec<f64>),
    stage: Option<Stage>,
}

impl<'a> Recursion<'a> {
    fn new(fractions: &'a [f64], sidedness: Sidedness, drift: f64) -> Self {
        Self { fractions, sidedness, drift, rule: panel_rule(), stage: None }
    }

    /// Probability of stopping at look `k` for the first time, given the
    /// candidate critical value.
    fn crossing(&self, k: usize, critical: f64) -> f64 {
        let t = self.fractions[k];
        match &self.stage {
            None => {
                let mean = self.drift * t.sqrt();
                match self.sidedness {
                    Sidedness::OneSidedUpper => norm_sf(critical - mean),
                    Sidedness::TwoSidedSymmetric => {
                        norm_cdf(-critical - mean) + norm_sf(critical - mean)
                    }
                }
            }
            Some(previous) => {
                let dt = t - previous.fraction;
                let sd = (dt / t).sqrt();
                let ratio = (previous.fraction / t).sqrt();
                let shift = self.drift * dt / t.sqrt();
                let mut continuing = 0.0;
                for ((&u, &w), &g) in
                    previous.nodes.iter().zip(&previous.weights).zip(&previous.density)
                {
                    let mean = u * ratio + shift;
                    let upper = norm_cdf((critical - mean) / sd);
                    let lower = match self.sidedness {
                        Sidedness::OneSidedUpper => 0.0,
                        Sidedness::TwoSidedSymmetric => norm_cdf((-critical - mean) / sd),
                    };
                    continuing += w * g * (upper - lower);
                }
                previous.mass - continuing
            }
        }
    }

    /// Replace the stored stage by the sub-density of look `k` on the
    /// continuation region implied by `critical`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the panel count is a positive ceiling of a region at most a few tens wide"
    )]
    fn advance(&mut self, k: usize, critical: f64) {
        let t = self.fractions[k];
        let (lo, hi) = match self.sidedness {
            Sidedness::OneSidedUpper => (LOWER_TRUNCATION - self.drift.abs(), critical),
            Sidedness::TwoSidedSymmetric => (-critical, critical),
        };
        let panels = (((hi - lo) / PANEL_WIDTH).ceil().max(1.0)) as usize;
        let (nodes, weights) = composite_rule(&self.rule, lo, hi, panels);
        let density: Vec<f64> = match &self.stage {
            None => {
                let mean = self.drift * t.sqrt();
                nodes.iter().map(|&z| norm_pdf(z - mean)).collect()
            }
            Some(previous) => {
                let dt = t - previous.fraction;
                let sd = (dt / t).sqrt();
                let ratio = (previous.fraction / t).sqrt();
                let shift = self.drift * dt / t.sqrt();
                nodes
                    .iter()
                    .map(|&z| {
                        previous
                            .nodes
                            .iter()
                            .zip(&previous.weights)
                            .zip(&previous.density)
                            .map(|((&u, &w), &g)| {
                                w * g * norm_pdf((z - (u * ratio + shift)) / sd) / sd
                            })
                            .sum::<f64>()
                    })
                    .collect()
            }
        };
        let mass = weights.iter().zip(&density).map(|(w, g)| w * g).sum();
        self.stage = Some(Stage { fraction: t, nodes, weights, density, mass });
    }

    fn solve(&self, k: usize, target: f64) -> Result<f64, Error> {
        let (mut low, mut high) = match self.sidedness {
            Sidedness::OneSidedUpper => (BRACKET_LOW_ONE_SIDED, BRACKET_HIGH),
            Sidedness::TwoSidedSymmetric => (0.0, BRACKET_HIGH),
        };
        if self.crossing(k, low) < target {
            return Err(Error::Numerical(
                "the alpha spent at a look exceeds the probability of continuing to it",
            ));
        }
        if self.crossing(k, high) > target {
            return Err(Error::Numerical("a boundary lies above 12 standard errors"));
        }
        for _ in 0..BISECTIONS {
            let mid = 0.5 * (low + high);
            if self.crossing(k, mid) > target {
                low = mid;
            } else {
                high = mid;
            }
            if high - low < BRACKET_WIDTH {
                break;
            }
        }
        Ok(0.5 * (low + high))
    }
}

/// Efficacy boundaries that spend `alpha` over `fractions` by `spending`.
///
/// # Errors
/// Fractions that are not strictly increasing in `(0, 1]`, more than
/// [`MAX_LOOKS`] looks, `alpha` outside `(0, 1)`, a look that spends no alpha,
/// a boundary outside the search bracket, or cancellation.
pub fn boundaries(
    fractions: &[f64],
    alpha: f64,
    spending: SpendingFunction,
    sidedness: Sidedness,
    cancelled: &dyn Fn() -> bool,
) -> Result<Boundaries, Error> {
    check_fractions(fractions)?;
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::InvalidParameter("the total alpha must lie in (0, 1)"));
    }
    let mut recursion = Recursion::new(fractions, sidedness, 0.0);
    let mut critical = Vec::with_capacity(fractions.len());
    let mut cumulative_spend = Vec::with_capacity(fractions.len());
    let mut spend_increments = Vec::with_capacity(fractions.len());
    let mut previous = 0.0;
    for (k, &fraction) in fractions.iter().enumerate() {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let cumulative = spending.spend(fraction, alpha);
        let increment = cumulative - previous;
        if !(increment.is_finite() && increment > 0.0) {
            return Err(Error::InvalidParameter("a look must spend a positive amount of alpha"));
        }
        let c = recursion.solve(k, increment)?;
        recursion.advance(k, c);
        critical.push(c);
        cumulative_spend.push(cumulative);
        spend_increments.push(increment);
        previous = cumulative;
    }
    Ok(Boundaries { critical, cumulative_spend, spend_increments })
}

/// Probability of stopping for efficacy at each look for the first time,
/// when the standardized drift is `drift` (`Z_k ~ N(drift sqrt(t_k), 1)`).
///
/// With `drift = 0` the entries are the alpha increments the boundaries were
/// built to spend; with a positive drift their sum is the power of the rule.
///
/// # Errors
/// Invalid fractions, a non-finite drift or boundary, a mismatch of lengths,
/// or cancellation.
pub fn crossing_probabilities(
    fractions: &[f64],
    critical: &[f64],
    sidedness: Sidedness,
    drift: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, Error> {
    check_fractions(fractions)?;
    if critical.len() != fractions.len() {
        return Err(Error::Shape("one critical value per look"));
    }
    if !drift.is_finite() || critical.iter().any(|c| !c.is_finite()) {
        return Err(Error::InvalidParameter("drift and critical values must be finite"));
    }
    if sidedness == Sidedness::TwoSidedSymmetric && critical.iter().any(|c| *c < 0.0) {
        return Err(Error::InvalidParameter("symmetric critical values are nonnegative"));
    }
    let mut recursion = Recursion::new(fractions, sidedness, drift);
    let mut out = Vec::with_capacity(fractions.len());
    for (k, &c) in critical.iter().enumerate() {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        out.push(recursion.crossing(k, c));
        recursion.advance(k, c);
    }
    Ok(out)
}

/// The first look at which the statistic crosses its boundary.
#[must_use]
pub fn first_crossing(statistics: &[f64], critical: &[f64], sidedness: Sidedness) -> Option<usize> {
    statistics.iter().zip(critical).position(|(z, c)| match sidedness {
        Sidedness::OneSidedUpper => z >= c,
        Sidedness::TwoSidedSymmetric => z.abs() >= *c,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn never() -> bool {
        false
    }

    #[test]
    fn one_look_is_the_fixed_sample_critical_value() {
        let b = boundaries(
            &[1.0],
            0.05,
            SpendingFunction::OBrienFlemingType,
            Sidedness::TwoSidedSymmetric,
            &never,
        )
        .unwrap();
        assert!((b.critical[0] - 1.959_963_984_540_054).abs() < 1e-9);
    }

    #[test]
    fn three_equal_looks_match_the_published_one_sided_boundaries() {
        let fractions = [1.0 / 3.0, 2.0 / 3.0, 1.0];
        let b = boundaries(
            &fractions,
            0.025,
            SpendingFunction::OBrienFlemingType,
            Sidedness::OneSidedUpper,
            &never,
        )
        .unwrap();
        for (got, want) in b.critical.iter().zip([3.7103, 2.5114, 1.9930]) {
            assert!((got - want).abs() < 2e-4, "{got} vs {want}");
        }
    }
}
