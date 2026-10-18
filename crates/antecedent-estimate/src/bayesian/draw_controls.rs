//! Numerical controls on the posterior draws behind a published interval.
//!
//! The endpoints of a credible interval are quantiles of a finite set of draws, so each
//! carries a Monte Carlo standard error that is numerical error of the report, not
//! posterior uncertainty. Under these controls the draw count has a supported minimum, the
//! endpoint errors are reported beside the interval, and an interval whose larger endpoint
//! error exceeds [`MAX_ENDPOINT_MCSE_FRACTION`] of its width is refused. Exact samplers
//! (conjugate, Laplace) draw independently, so the error is the order-statistic error of the
//! draw count; an MCMC sampler's draws are autocorrelated, so the same error is inflated by
//! the square root of draws per effective draw, the tail effective sample size its own
//! publication gate measures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_prob::HessianFactorization;

use super::{BayesianBackendKind, CausalPosterior, HMC_MIN_DRAWS};
use crate::count_fitted_posterior::{
    MAX_ENDPOINT_MCSE_FRACTION, endpoint_mcse_width_fraction, quantile_mcse,
};
use crate::error::EstimationError;

/// Supported minimum draw count of an exact sampler under draw controls.
pub const MIN_CONTROLLED_DRAWS: usize = 2000;

/// Stable prefix of the inference-diagnostics note that records the endpoint errors
/// ([`EndpointMonteCarloError::note`]).
pub const DRAW_CONTROL_NOTE_PREFIX: &str = "draw_controls.endpoint_mcse";

const LOWER_TAIL: f64 = 0.025;
const UPPER_TAIL: f64 = 0.975;

/// The supported minimum draw count under draw controls: [`MIN_CONTROLLED_DRAWS`] for an
/// exact sampler, and the draw floor the MCMC publication gate needs for HMC.
#[must_use]
pub const fn supported_min_draws(backend: BayesianBackendKind) -> usize {
    match backend {
        BayesianBackendKind::Hmc => HMC_MIN_DRAWS,
        _ => MIN_CONTROLLED_DRAWS,
    }
}

/// Monte Carlo standard errors of the 2.5% and 97.5% endpoints of the effect's interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointMonteCarloError {
    /// Draws the endpoints were estimated from.
    pub draws: usize,
    /// Effective draws: the draw count of an exact sampler, the tail effective sample size
    /// (at most the draw count) of an MCMC sampler.
    pub effective_draws: f64,
    /// Standard error of the lower endpoint.
    pub lower_mcse: f64,
    /// Standard error of the upper endpoint.
    pub upper_mcse: f64,
    /// The larger endpoint error as a fraction of the interval width.
    pub width_fraction: f64,
}

impl EndpointMonteCarloError {
    /// Whether the endpoint error is above [`MAX_ENDPOINT_MCSE_FRACTION`] of the width (or
    /// not a number): the draws, not the posterior, set the endpoints.
    #[must_use]
    pub fn exceeds_tolerance(&self) -> bool {
        self.width_fraction.is_nan() || self.width_fraction > MAX_ENDPOINT_MCSE_FRACTION
    }

    /// The inference-diagnostics note that records these errors with the posterior.
    #[must_use]
    pub fn note(&self) -> Arc<str> {
        Arc::from(format!(
            "{DRAW_CONTROL_NOTE_PREFIX} draws={} effective_draws={:e} lower_mcse={:e} \
             upper_mcse={:e} width_fraction={:e}",
            self.draws, self.effective_draws, self.lower_mcse, self.upper_mcse, self.width_fraction
        ))
    }
}

/// The endpoint errors recorded by [`EndpointMonteCarloError::note`], if a note carries them.
#[must_use]
pub fn endpoint_mcse_from_notes(notes: &[Arc<str>]) -> Option<EndpointMonteCarloError> {
    notes.iter().find_map(|note| {
        let rest = note.strip_prefix(DRAW_CONTROL_NOTE_PREFIX)?;
        let field = |name: &str| {
            rest.split_whitespace().find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
        };
        Some(EndpointMonteCarloError {
            draws: field("draws")?.parse().ok()?,
            effective_draws: field("effective_draws")?.parse().ok()?,
            lower_mcse: field("lower_mcse")?.parse().ok()?,
            upper_mcse: field("upper_mcse")?.parse().ok()?,
            width_fraction: field("width_fraction")?.parse().ok()?,
        })
    })
}

/// Monte Carlo standard errors of the published 95% interval endpoints of the posterior's
/// effect.
///
/// # Errors
/// A posterior without an effect column, with fewer than two draws or a non-finite draw.
pub fn endpoint_monte_carlo_error(
    posterior: &CausalPosterior,
) -> Result<EndpointMonteCarloError, EstimationError> {
    let column = posterior
        .effect_column()
        .ok_or_else(|| EstimationError::stats_msg("CausalPosterior has no effect column"))?;
    let values = posterior.draws.column(column).map_err(EstimationError::from)?;
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg(
            "endpoint Monte Carlo error needs at least two finite effect draws",
        ));
    }
    let (Some(&lower), Some(&upper)) =
        (posterior.summaries.q025.get(column), posterior.summaries.q975.get(column))
    else {
        return Err(EstimationError::stats_msg("posterior summary has no effect interval"));
    };
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let draws = sorted.len();
    let draw_count = draws as f64;
    let diagnostics = &posterior.diagnostics;
    let effective_draws = if diagnostics.factorization == HessianFactorization::Mcmc {
        // Without a finite positive tail effective sample size the autocorrelation is
        // unknown: the error is unbounded rather than assumed independent.
        diagnostics
            .ess_tail_min
            .filter(|ess| ess.is_finite() && *ess > 0.0)
            .map_or(0.0, |ess| ess.min(draw_count))
    } else {
        draw_count
    };
    let inflation = (draw_count / effective_draws).sqrt();
    let lower_mcse = quantile_mcse(&sorted, LOWER_TAIL) * inflation;
    let upper_mcse = quantile_mcse(&sorted, UPPER_TAIL) * inflation;
    Ok(EndpointMonteCarloError {
        draws,
        effective_draws,
        lower_mcse,
        upper_mcse,
        width_fraction: endpoint_mcse_width_fraction(lower_mcse, upper_mcse, upper - lower),
    })
}
