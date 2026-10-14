//! Claims discovery reuse does not make, as typed refusals.
//!
//! Each function here is a request a user might reasonably make of a discovered graph. Each
//! is refused with `cell_not_licensed` (or `route_not_supported` for a route that does not
//! exist), and the refusal states the declaration or evidence that would license it and the
//! calibration design that would have to be measured first. None is measured here.
//!
//! # Calibration design (recorded, not measured)
//!
//! The only licensable error-control or interval claim downstream of discovery is a
//! statement about the *whole method*: discovery on the discovery rows, then the analysis on
//! its rows, with the reuse of the discovery rows (or a sample split) inside the procedure.
//!
//! * **Method.** Discovery with the declared contract, graph acceptance rule (resolved
//!   structure or declared completion), then the declared estimator and interval procedure
//!   on the declared analysis rows. For reuse of the discovery rows, the interval procedure
//!   must account for the selection of the graph (post-selection). A sample split (discovery
//!   rows disjoint from analysis rows, as the evidence ledger records) is the design the
//!   first calibration would target, because it needs no post-selection adjustment.
//! * **Data-generating processes.** Linear-Gaussian SCMs with known graphs spanning the
//!   declared variable count, edge density and effect sizes, including weakly faithful
//!   (near-cancelling) parameters, plus the misspecified-test case.
//! * **Nominal level and probes.** The nominal level of the downstream interval, the
//!   probability that the discovered class contains the true graph, and the probability that
//!   a declared completion is a member of the true class, each over the sample-size range
//!   and with their Monte Carlo error bounds.
//! * **Support range.** Sample sizes and variable counts inside the measured grid only; a
//!   request outside it is refused as outside the measured range.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::convert::Infallible;

use super::outcome::DiscoveryOutcome;
use super::refusal::SelectiveDiscoveryError;

/// Request a statement that the discovered structure recovers the true graph, or that its
/// edges carry an error rate (false discovery, type I).
///
/// # Errors
///
/// Always `cell_not_licensed`: matching a full run is an algorithmic fact, not a statement
/// about the truth.
pub fn request_recovery_claim(
    _outcome: &DiscoveryOutcome,
) -> Result<Infallible, SelectiveDiscoveryError> {
    Err(SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("cell_not_licensed"),
        "selective_discovery.recovery_claim_not_licensed",
        "no recovery, false-discovery or error-control statement is licensed for a discovered \
         graph: no whole-method calibration has been measured at this coordinate. Report the \
         structure as an estimate with its unresolved edges and bounded-search receipts, or \
         run the whole-method calibration (discovery, graph acceptance, estimation, on a \
         declared sample split) for this contract's variable count and sample size",
    ))
}

/// Request an interval or posterior for a downstream estimate that rests on a discovered
/// graph, on account of discovery uncertainty.
///
/// # Errors
///
/// Always `cell_not_licensed`.
pub fn request_downstream_interval(
    _outcome: &DiscoveryOutcome,
) -> Result<Infallible, SelectiveDiscoveryError> {
    Err(SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("cell_not_licensed"),
        "selective_discovery.downstream_interval_not_licensed",
        "an interval for an estimate that rests on a discovered graph needs a calibration of \
         the complete method, including the reuse of the discovery rows in the analysis. \
         Use disjoint discovery and analysis rows (the evidence ledger records the split) \
         and condition on a declared graph; the point estimate is available without an \
         interval",
    ))
}

/// Request reuse of local scores for a score-based search.
///
/// # Errors
///
/// Always `route_not_supported`: score caches are internal to a score-based run and no
/// injection seam is delivered.
pub fn request_score_search_reuse() -> Result<Infallible, SelectiveDiscoveryError> {
    Err(SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("route_not_supported"),
        "selective_discovery.score_search_closed",
        "selective reuse covers constraint-based discovery (PC, FCI, PCMCI, PCMCI+). \
         Score-based search recomputes its local scores each run; use PC with a retained \
         statistic store, or run the score-based algorithm directly",
    ))
}
