//! Claims discovery reuse does not make, as typed refusals.
//!
//! Each function here is a request a user might reasonably make of a discovered graph. Each
//! is refused with `cell_not_licensed` (or `route_not_supported` for a route that does not
//! exist), and the refusal states the declaration or evidence that would license it and the
//! calibration design that would have to be measured first. None is measured here.
//!
//! # What can and cannot be licensed
//!
//! * **Per-dataset recovery and error control are closed as claims.** Whether a discovered
//!   graph is the true graph rests on faithfulness and causal sufficiency, and neither can be
//!   tested from the data the discovery ran on. No statement about one dataset's graph can be
//!   licensed, however the procedure is calibrated. How often the method recovers the graph
//!   over a declared family of generating processes is a measured property of the method,
//!   published as a table in the documentation and never as a license.
//! * **A downstream interval is licensed only for rows discovery never saw.** The interval
//!   that can carry a coverage statement is the whole method's: a seeded split, discovery on
//!   one part, a resolved structure accepted, and the effect estimated on the other part,
//!   scored against the true effect over a declared family (see
//!   `antecedent::analysis::sample_split_effect`). The same rows used to discover and to
//!   estimate are a different construction with no licence.
//! * **Support range.** Variable counts and sample sizes inside the measured grid only; a
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
        "no recovery, false-discovery or error-control statement can be licensed for the graph \
         discovered on one dataset: faithfulness and causal sufficiency are not testable from \
         the data discovery ran on. Report the structure as an estimate with its unresolved \
         edges and bounded-search receipts. How often the method recovers the graph over a \
         declared family of processes is a measured table in the documentation, not a license",
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
        "an interval for an estimate that rests on a discovered graph cannot be licensed on \
         the rows discovery used: selecting the graph and estimating on the same rows is not \
         a construction with a measured coverage. Use disjoint discovery and analysis rows \
         (the evidence ledger records the split); the whole-method interval of the effect on \
         the analysis rows is attested from a coverage record by the sample-split effect route",
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
