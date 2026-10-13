//! Evidence required to move a posterior or decision probability past a
//! declared level.
//!
//! Each solve is exact:
//!
//! - [`required_bayes_factor`]: the Bayes factor (posterior odds over prior
//!   odds) that takes a probability from `prior` to `level`;
//! - [`required_observations_binary_signal`]: the fewest conditionally
//!   independent readings of a binary signal with declared rates;
//! - [`required_observations_conjugate`]: the fewest consecutive successes or
//!   failures that take a Beta-Bernoulli predictive probability to `level`.
//!
//! A level that no admissible evidence reaches is not an answer to be
//! rounded: the request is refused as `evidence_cannot_reach_level`, with the
//! most extreme probability that is reachable (a supremum when raising, an
//! infimum when lowering) stated beside the requested level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::InferenceBasis;
use antecedent_core::ExecutionContext;

pub use antecedent_model::observation_query::Direction;

use super::observation_query::{
    ObservationClaim, ObservationQueryRefusal, ObservationQueryWire, seal_declared_query,
};

/// The Bayes factor that moves a probability from `prior` to `level`.
///
/// # Errors
/// A prior or level outside `(0, 1)`, or cancellation.
pub fn required_bayes_factor(
    prior: f64,
    level: f64,
    direction: Direction,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal_declared_query(
        ObservationQueryWire::RequiredBayesFactor {
            prior_bits: prior.to_bits(),
            level_bits: level.to_bits(),
            direction: direction.as_str().to_owned(),
        },
        InferenceBasis::DeclaredModel,
        ctx,
    )
}

/// The fewest conditionally independent readings of a binary signal that move
/// `P(H)` from `prior` to `level`.
///
/// # Errors
/// `evidence_cannot_reach_level`, stating the most extreme probability
/// reachable, when the signal cannot move the probability that way or needs
/// more than `max_observations`; a probability outside its range; cancellation.
#[allow(
    clippy::too_many_arguments,
    reason = "the solve names the prior, the level, its direction, the signal's two declared rates, the bound, the reading and the context as separate values"
)]
pub fn required_observations_binary_signal(
    prior: f64,
    level: f64,
    direction: Direction,
    rate_given_hypothesis: f64,
    rate_given_complement: f64,
    max_observations: u64,
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal_declared_query(
        ObservationQueryWire::RequiredObservationsBinarySignal {
            prior_bits: prior.to_bits(),
            level_bits: level.to_bits(),
            direction: direction.as_str().to_owned(),
            rate_given_hypothesis_bits: rate_given_hypothesis.to_bits(),
            rate_given_complement_bits: rate_given_complement.to_bits(),
            max_observations,
        },
        basis,
        ctx,
    )
}

/// The fewest consecutive successes (raising) or failures (lowering) that move
/// a Beta-Bernoulli predictive probability to `level`.
///
/// # Errors
/// `evidence_cannot_reach_level` when more than `max_observations` are
/// needed; a level outside `(0, 1)`; cancellation.
pub fn required_observations_conjugate(
    alpha: f64,
    beta: f64,
    level: f64,
    direction: Direction,
    max_observations: u64,
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal_declared_query(
        ObservationQueryWire::RequiredObservationsConjugate {
            alpha_bits: alpha.to_bits(),
            beta_bits: beta.to_bits(),
            level_bits: level.to_bits(),
            direction: direction.as_str().to_owned(),
            max_observations,
        },
        basis,
        ctx,
    )
}
