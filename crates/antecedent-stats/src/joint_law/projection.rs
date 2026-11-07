//! Interval candidates and exceedance decisions for joint-law results.

use crate::quantile::{QuantileRule, equal_tail_interval_sorted};
use crate::special::normal_ppf;

use super::{
    CALIBRATION_STATUS, JointLawError, MASS_TOLERANCE, MAX_COMPONENTS, Marginal, require_finite,
};

fn check_level(level: f64) -> Result<(), JointLawError> {
    if level.is_finite() && 0.0 < level && level < 1.0 {
        Ok(())
    } else {
        Err(JointLawError::InvalidLevel("a level is strictly between zero and one"))
    }
}

/// An interval built at a composed coordinate whose coverage is not measured.
///
/// This is a construction, not a coverage statement: it is reported with the
/// calibration status `not_measured` until a record for exactly this composed
/// coordinate and construction is attested, and the joint-law rule closes the
/// interval reading until then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalCandidate {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// The nominal level the construction was asked for.
    pub level: f64,
    /// The construction's stable name.
    pub construction: &'static str,
    /// Always [`CALIBRATION_STATUS`].
    pub calibration: &'static str,
}

/// Construction name of [`normal_interval_candidate`].
pub const NORMAL_APPROXIMATION_CONSTRUCTION: &str = "normal_approximation";
/// Construction name of [`draw_interval_candidate`] under the exchangeable
/// rank rule.
pub const DRAW_EXCHANGEABLE_RANK_CONSTRUCTION: &str = "aligned_draw_equal_tail_exchangeable_rank";
/// Construction name of [`draw_interval_candidate`] under the interpolated rule.
pub const DRAW_INTERPOLATED_CONSTRUCTION: &str = "aligned_draw_equal_tail_interpolated";
/// Construction name of [`marginal_interval_candidate`].
pub const MIXTURE_EQUAL_TAIL_CONSTRUCTION: &str = "mixture_marginal_equal_tail";

/// `mean +- z sqrt(variance)` at the composed coordinate, from the variance a
/// supported route determined (a delta-method variance of a linear map under a
/// covariance, a mixture variance, or the variance of aligned composed draws).
///
/// # Errors
/// A level outside `(0, 1)`, a non-finite mean or a negative or non-finite
/// variance.
pub fn normal_interval_candidate(
    mean: f64,
    variance: f64,
    level: f64,
) -> Result<IntervalCandidate, JointLawError> {
    check_level(level)?;
    require_finite(&[mean, variance], "interval input")?;
    if variance < 0.0 {
        return Err(JointLawError::InvalidVariance("variance is negative"));
    }
    let half = normal_ppf(0.5 + level / 2.0) * variance.sqrt();
    if !half.is_finite() {
        return Err(JointLawError::NonFiniteValue);
    }
    Ok(IntervalCandidate {
        lower: mean - half,
        upper: mean + half,
        level,
        construction: NORMAL_APPROXIMATION_CONSTRUCTION,
        calibration: CALIBRATION_STATUS,
    })
}

/// The equal-tailed interval of composed draws under a quantile rule.
///
/// The rows must be realizations of the composed quantity's law (aligned
/// draws pushed through the composition); percentiles of the component draws
/// are not the composition's.
///
/// # Errors
/// A level outside `(0, 1)`, no rows or non-finite draws.
pub fn draw_interval_candidate(
    draws: &[f64],
    level: f64,
    rule: QuantileRule,
) -> Result<IntervalCandidate, JointLawError> {
    check_level(level)?;
    if draws.is_empty() {
        return Err(JointLawError::Shape("an interval needs at least one row"));
    }
    require_finite(draws, "draw")?;
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, level, rule);
    let construction = match rule {
        QuantileRule::ExchangeableRank => DRAW_EXCHANGEABLE_RANK_CONSTRUCTION,
        QuantileRule::Interpolated => DRAW_INTERPOLATED_CONSTRUCTION,
    };
    Ok(IntervalCandidate { lower, upper, level, construction, calibration: CALIBRATION_STATUS })
}

/// The equal-tailed interval of a finite marginal (a mixture's law).
///
/// # Errors
/// A level outside `(0, 1)`.
pub fn marginal_interval_candidate(
    marginal: &Marginal,
    level: f64,
) -> Result<IntervalCandidate, JointLawError> {
    check_level(level)?;
    Ok(IntervalCandidate {
        lower: marginal.quantile((1.0 - level) / 2.0)?,
        upper: marginal.quantile((1.0 + level) / 2.0)?,
        level,
        construction: MIXTURE_EQUAL_TAIL_CONSTRUCTION,
        calibration: CALIBRATION_STATUS,
    })
}

/// The marginal law of a finite mixture of draw sets with known weights.
///
/// Each component's draws carry mass `weight_i / n_i`. A mixture selects one
/// component, so the components need no joint law and are never paired.
///
/// # Errors
/// No components, a weight vector that is not a probability vector, an empty or
/// non-finite component, or more atoms than [`MAX_MARGINAL_ATOMS`].
pub fn mixture_marginal(weights: &[f64], components: &[&[f64]]) -> Result<Marginal, JointLawError> {
    let k = weights.len();
    if k == 0 || k > MAX_COMPONENTS {
        return Err(JointLawError::Shape("components outside 1..=MAX_COMPONENTS"));
    }
    if components.len() != k {
        return Err(JointLawError::Shape("weights and components differ in count"));
    }
    require_finite(weights, "weight")?;
    if weights.iter().any(|&w| w < 0.0) {
        return Err(JointLawError::InvalidWeights("weights must be non-negative"));
    }
    if (weights.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE {
        return Err(JointLawError::InvalidWeights("weights must sum to one"));
    }
    let (mut values, mut masses) = (Vec::new(), Vec::new());
    for (weight, component) in weights.iter().zip(components) {
        if component.is_empty() {
            return Err(JointLawError::Shape("a mixture component needs at least one draw"));
        }
        require_finite(component, "mixture draw")?;
        let mass = weight / component.len() as f64;
        values.extend_from_slice(component);
        masses.resize(masses.len() + component.len(), mass);
    }
    Marginal::from_weighted(&values, &masses)
}

/// `P(composed > threshold)` read from composed draws, with its Monte Carlo
/// error band and a decision verdict against a probability cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExceedanceDecision {
    /// Number of rows.
    pub n: usize,
    /// Rows strictly above the threshold, as a fraction.
    pub probability: f64,
    /// Monte Carlo standard error of the probability, from the
    /// Agresti-Coull adjusted proportion so that a boundary fraction of zero
    /// or one does not report zero error.
    pub standard_error: f64,
    /// The half-width of the Monte Carlo band, `z * standard_error`.
    pub band: f64,
    /// The cutoff the verdict compares against.
    pub cutoff: f64,
    /// The verdict.
    pub verdict: DecisionVerdict,
    /// Always [`CALIBRATION_STATUS`].
    pub calibration: &'static str,
}

/// The decision about a composed exceedance probability.
///
/// The verdict is about the Monte Carlo error of the stated probability only:
/// it says nothing about whether the draws' law is calibrated for the composed
/// coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionVerdict {
    /// The probability exceeds the cutoff beyond the Monte Carlo band.
    Act,
    /// The probability is below the cutoff beyond the Monte Carlo band.
    Withhold,
    /// The cutoff is inside the Monte Carlo band: more draws, or no verdict.
    Unresolved,
}

impl DecisionVerdict {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Act => "act",
            Self::Withhold => "withhold",
            Self::Unresolved => "unresolved",
        }
    }
}

fn exceedance_verdict(
    n: usize,
    hits: usize,
    effective_size: Option<f64>,
    cutoff: f64,
    band_level: f64,
) -> Result<ExceedanceDecision, JointLawError> {
    check_level(cutoff)?;
    check_level(band_level)?;
    let nf = n as f64;
    let size = match effective_size {
        None => nf,
        Some(s) if s.is_finite() && s > 0.0 && s <= nf => s,
        Some(_) => return Err(JointLawError::InvalidLevel("effective size is in (0, n]")),
    };
    let z = normal_ppf(0.5 + band_level / 2.0);
    let probability = hits as f64 / nf;
    // Adjust with the same effective sample size the draws carry.
    let hits_eff = probability * size;
    let adjusted = (hits_eff + z * z / 2.0) / (size + z * z);
    let standard_error = (adjusted * (1.0 - adjusted) / (size + z * z)).sqrt();
    let band = z * standard_error;
    if !(standard_error.is_finite() && band.is_finite()) {
        return Err(JointLawError::NonFiniteValue);
    }
    let verdict = if probability - band > cutoff {
        DecisionVerdict::Act
    } else if probability + band < cutoff {
        DecisionVerdict::Withhold
    } else {
        DecisionVerdict::Unresolved
    };
    Ok(ExceedanceDecision {
        n,
        probability,
        standard_error,
        band,
        cutoff,
        verdict,
        calibration: CALIBRATION_STATUS,
    })
}

/// A posterior-style decision at the composed coordinate: `P(composed >
/// threshold)` from composed draws against a probability `cutoff`, with the
/// Monte Carlo band at `band_level`.
///
/// # Errors
/// No rows, non-finite draws or threshold, a cutoff or band level outside
/// `(0, 1)`, or an effective size outside `(0, n]`.
pub fn composed_exceedance_decision(
    draws: &[f64],
    threshold: f64,
    cutoff: f64,
    band_level: f64,
    effective_size: Option<f64>,
) -> Result<ExceedanceDecision, JointLawError> {
    if draws.is_empty() {
        return Err(JointLawError::Shape("a decision needs at least one row"));
    }
    require_finite(draws, "draw")?;
    require_finite(&[threshold], "threshold")?;
    let hits = draws.iter().filter(|d| **d > threshold).count();
    exceedance_verdict(draws.len(), hits, effective_size, cutoff, band_level)
}

/// `P(X > threshold)` of a finite mixture of draw sets with known weights, as
/// the weighted sum of the components' exceedance fractions, with a Monte
/// Carlo standard error that treats the components' draws as independently
/// produced (`sum w_i^2 p_i (1 - p_i) / n_i`).
///
/// # Errors
/// As [`mixture_marginal`], plus a non-finite threshold.
pub fn mixture_exceedance(
    weights: &[f64],
    components: &[&[f64]],
    threshold: f64,
) -> Result<(f64, f64), JointLawError> {
    // Validates the weights and the components exactly as the mixture law.
    mixture_marginal(weights, components)?;
    require_finite(&[threshold], "threshold")?;
    let (mut probability, mut variance) = (0.0, 0.0);
    for (weight, component) in weights.iter().zip(components) {
        let n = component.len() as f64;
        let p = component.iter().filter(|d| **d > threshold).count() as f64 / n;
        probability += weight * p;
        variance += weight * weight * p * (1.0 - p) / n;
    }
    Ok((probability, variance.sqrt()))
}
