//! What a callback is asked and what it may answer: the query, the output and
//! the check that an output is a law of the declared quantity.
//!
//! A returned value is data from outside. It is accepted only through
//! [`validate_output`], which checks the shape of the answer against the query,
//! finiteness, the declared support, and the properties that make a returned
//! distribution function or quantile function a law. A malformed law is
//! refused whole; nothing is repaired, sorted or clipped.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::identity::CanonicalWriter;

use super::declaration::CallbackOperation;
use super::refusal::{
    CallbackRefusal, output_not_a_law, output_not_finite, output_outside_support, output_shape,
    output_too_large, query_invalid,
};
use crate::interface_contract::SupportDomain;

/// Default bound on draws per query that a decoder of untrusted input applies. A query itself
/// carries no fixed bound: the size of an answer is bounded by the `max_values` budget passed to
/// [`validate_output`], which a caller derives from its memory budget.
pub const MAX_CALLBACK_DRAWS: u32 = 1_000_000;
/// Default bound on thresholds or quantile levels per query that a decoder of untrusted input
/// applies.
pub const MAX_CALLBACK_POINTS: usize = 65_536;

/// One question put to a callback.
#[derive(Clone, Debug, PartialEq)]
pub enum CallbackQuery {
    /// The mean.
    Mean,
    /// Marginal draws.
    Sample {
        /// Number of draws, at least one.
        draws: u32,
    },
    /// The distribution function at finite, strictly increasing thresholds.
    Cdf {
        /// Thresholds.
        thresholds: Vec<f64>,
    },
    /// The quantile function at levels strictly inside `(0, 1)`, strictly
    /// increasing.
    Quantile {
        /// Levels.
        levels: Vec<f64>,
    },
    /// The posterior after one possible observation.
    PosteriorUpdate {
        /// The possible observation, finite.
        observation: f64,
        /// Draws of the updated posterior (zero asks for the mean alone).
        draws: u32,
    },
}

impl CallbackQuery {
    /// The operation the query negotiates.
    #[must_use]
    pub const fn operation(&self) -> CallbackOperation {
        match self {
            Self::Mean => CallbackOperation::Mean,
            Self::Sample { .. } => CallbackOperation::Sample,
            Self::Cdf { .. } => CallbackOperation::Cdf,
            Self::Quantile { .. } => CallbackOperation::Quantile,
            Self::PosteriorUpdate { .. } => CallbackOperation::PosteriorUpdate,
        }
    }

    /// Number of values an answer holds.
    #[must_use]
    pub fn expected_values(&self) -> usize {
        match self {
            Self::Mean => 1,
            Self::Sample { draws } => *draws as usize,
            Self::Cdf { thresholds } => thresholds.len(),
            Self::Quantile { levels } => levels.len(),
            Self::PosteriorUpdate { draws, .. } => 1 + *draws as usize,
        }
    }

    /// Check the query is well formed.
    ///
    /// # Errors
    /// A `callback_output_invalid` refusal with a `callback_protocol.query_invalid`
    /// detail.
    pub fn validate(&self) -> Result<(), CallbackRefusal> {
        match self {
            Self::Mean => Ok(()),
            Self::Sample { draws } => {
                if *draws == 0 {
                    return Err(query_invalid("draws"));
                }
                Ok(())
            }
            Self::Cdf { thresholds } => {
                if thresholds.is_empty()
                    || thresholds.iter().any(|t| !t.is_finite())
                    || thresholds.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(query_invalid("thresholds"));
                }
                Ok(())
            }
            Self::Quantile { levels } => {
                if levels.is_empty()
                    || levels.iter().any(|p| !(*p > 0.0 && *p < 1.0))
                    || levels.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return Err(query_invalid("levels"));
                }
                Ok(())
            }
            Self::PosteriorUpdate { observation, draws } => {
                if !observation.is_finite() {
                    return Err(query_invalid("observation"));
                }
                Ok(())
            }
        }
    }

    /// Write the query into a canonical stream: a tag word, then every value
    /// as an IEEE bit pattern.
    pub fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.text(self.operation().as_str());
        match self {
            Self::Mean => {}
            Self::Sample { draws } => {
                writer.word(u64::from(*draws));
            }
            Self::Cdf { thresholds } => {
                writer.reals(thresholds);
            }
            Self::Quantile { levels } => {
                writer.reals(levels);
            }
            Self::PosteriorUpdate { observation, draws } => {
                writer.real(*observation).word(u64::from(*draws));
            }
        }
    }
}

/// What a callback returned. Unchecked until [`validate_output`] accepts it.
#[derive(Clone, Debug, PartialEq)]
pub enum CallbackOutput {
    /// The mean.
    Mean {
        /// The value.
        value: f64,
    },
    /// Marginal draws.
    Sample {
        /// The draws, in the provider's order.
        draws: Vec<f64>,
    },
    /// Distribution function values, one per threshold.
    Cdf {
        /// `F(threshold)`.
        values: Vec<f64>,
    },
    /// Quantiles, one per level.
    Quantile {
        /// `Q(level)`.
        values: Vec<f64>,
    },
    /// The updated posterior.
    PosteriorUpdate {
        /// Mean of the updated posterior.
        mean: f64,
        /// Draws of the updated posterior; exactly the requested number.
        draws: Vec<f64>,
    },
}

impl CallbackOutput {
    /// The operation the output answers.
    #[must_use]
    pub const fn operation(&self) -> CallbackOperation {
        match self {
            Self::Mean { .. } => CallbackOperation::Mean,
            Self::Sample { .. } => CallbackOperation::Sample,
            Self::Cdf { .. } => CallbackOperation::Cdf,
            Self::Quantile { .. } => CallbackOperation::Quantile,
            Self::PosteriorUpdate { .. } => CallbackOperation::PosteriorUpdate,
        }
    }

    /// Number of values returned.
    #[must_use]
    pub fn value_count(&self) -> usize {
        match self {
            Self::Mean { .. } => 1,
            Self::Sample { draws } => draws.len(),
            Self::Cdf { values } | Self::Quantile { values } => values.len(),
            Self::PosteriorUpdate { draws, .. } => 1 + draws.len(),
        }
    }

    /// Every value in answer order.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        match self {
            Self::Mean { value } => vec![*value],
            Self::Sample { draws } => draws.clone(),
            Self::Cdf { values } | Self::Quantile { values } => values.clone(),
            Self::PosteriorUpdate { mean, draws } => {
                let mut all = Vec::with_capacity(1 + draws.len());
                all.push(*mean);
                all.extend_from_slice(draws);
                all
            }
        }
    }

    /// Write the output into a canonical stream as bit patterns.
    pub fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.text(self.operation().as_str()).reals(&self.values());
    }
}

/// An output the protocol accepted for a query.
///
/// Only [`validate_output`] constructs one. Acceptance says the answer is a
/// well-formed object of the declared quantity; it says nothing about the
/// provider's numerical truth.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedOutput {
    query: CallbackQuery,
    output: CallbackOutput,
}

impl ValidatedOutput {
    /// The query answered.
    #[must_use]
    pub const fn query(&self) -> &CallbackQuery {
        &self.query
    }

    /// The accepted output.
    #[must_use]
    pub const fn output(&self) -> &CallbackOutput {
        &self.output
    }

    /// Consume into the accepted output.
    #[must_use]
    pub fn into_output(self) -> CallbackOutput {
        self.output
    }
}

/// Lower and upper bound of a support, and whether it is the two-point set.
struct Bounds {
    lower: f64,
    upper: f64,
    binary: bool,
}

fn support_bounds(support: &SupportDomain) -> Option<Bounds> {
    match support {
        SupportDomain::Real => {
            Some(Bounds { lower: f64::NEG_INFINITY, upper: f64::INFINITY, binary: false })
        }
        SupportDomain::NonNegative => {
            Some(Bounds { lower: 0.0, upper: f64::INFINITY, binary: false })
        }
        SupportDomain::Binary => Some(Bounds { lower: 0.0, upper: 1.0, binary: true }),
        SupportDomain::Interval { lower_bits, upper_bits } => Some(Bounds {
            lower: f64::from_bits(*lower_bits),
            upper: f64::from_bits(*upper_bits),
            binary: false,
        }),
        SupportDomain::Categorical { .. } => None,
    }
}

impl Bounds {
    #[allow(
        clippy::float_cmp,
        reason = "declared support endpoints are exact values; the comparison is the bit-for-bit declared rule"
    )]
    fn holds_value(&self, x: f64) -> bool {
        if self.binary { x == 0.0 || x == 1.0 } else { x >= self.lower && x <= self.upper }
    }

    fn holds_mean(&self, x: f64) -> bool {
        x >= self.lower && x <= self.upper
    }
}

fn require_finite(values: &[f64], what: &str) -> Result<(), CallbackRefusal> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(output_not_finite(what));
    }
    Ok(())
}

fn require_in_support(
    values: &[f64],
    bounds: &Bounds,
    support: &SupportDomain,
    what: &str,
) -> Result<(), CallbackRefusal> {
    if values.iter().any(|v| !bounds.holds_value(*v)) {
        return Err(output_outside_support(what, &support.describe()));
    }
    Ok(())
}

#[allow(
    clippy::float_cmp,
    reason = "a law's mass below and above its declared support must be exactly zero and one"
)]
fn check_cdf(thresholds: &[f64], values: &[f64], bounds: &Bounds) -> Result<(), CallbackRefusal> {
    if values.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(output_not_a_law("cdf.value_outside_unit_interval"));
    }
    if values.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(output_not_a_law("cdf.decreasing"));
    }
    for (threshold, value) in thresholds.iter().zip(values) {
        if *threshold < bounds.lower && *value != 0.0 {
            return Err(output_not_a_law("cdf.mass_below_support"));
        }
        if *threshold >= bounds.upper && *value != 1.0 {
            return Err(output_not_a_law("cdf.mass_missing_above_support"));
        }
    }
    Ok(())
}

/// Check an output is a well-formed answer to `query` for a quantity with the
/// declared `support`, holding at most `max_values` values.
///
/// # Errors
/// A `callback_output_invalid` refusal: a shape that is not the query's, a
/// non-finite value, a value outside the support, or a distribution or
/// quantile function that is not a law. A query beyond the bound is a
/// `callback_limit_exceeded` refusal.
pub fn validate_output(
    query: &CallbackQuery,
    output: CallbackOutput,
    support: &SupportDomain,
    max_values: usize,
) -> Result<ValidatedOutput, CallbackRefusal> {
    query.validate()?;
    if query.operation() != output.operation() {
        return Err(output_shape("operation"));
    }
    if output.value_count() > max_values {
        return Err(output_too_large(max_values));
    }
    if output.value_count() != query.expected_values() {
        return Err(output_shape("value_count"));
    }
    let bounds =
        support_bounds(support).ok_or_else(|| output_outside_support("support", "numeric"))?;
    require_finite(&output.values(), "values")?;
    match (query, &output) {
        (CallbackQuery::Mean, CallbackOutput::Mean { value }) => {
            if !bounds.holds_mean(*value) {
                return Err(output_outside_support("mean", &support.describe()));
            }
        }
        (CallbackQuery::Sample { .. }, CallbackOutput::Sample { draws }) => {
            require_in_support(draws, &bounds, support, "draws")?;
        }
        (CallbackQuery::Cdf { thresholds }, CallbackOutput::Cdf { values }) => {
            check_cdf(thresholds, values, &bounds)?;
        }
        (CallbackQuery::Quantile { .. }, CallbackOutput::Quantile { values }) => {
            if values.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(output_not_a_law("quantile.decreasing"));
            }
            require_in_support(values, &bounds, support, "quantiles")?;
        }
        (
            CallbackQuery::PosteriorUpdate { .. },
            CallbackOutput::PosteriorUpdate { mean, draws },
        ) => {
            if !bounds.holds_mean(*mean) {
                return Err(output_outside_support("posterior_mean", &support.describe()));
            }
            require_in_support(draws, &bounds, support, "draws")?;
        }
        _ => return Err(output_shape("operation")),
    }
    Ok(ValidatedOutput { query: query.clone(), output })
}
