//! Acceptance of discovered graph review artifacts.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

// --- Discovery review acceptance helpers -----------------------------------------
//
// The old `StudyBuilder` took a `DiscoveryAccept` flag (`AutoAccept` / `Review`) on each
// `.discover_*()` setter and deferred the actual review gate to `compile()` time. The new
// facade runs discovery standalone (see `antecedent::discovery`) and requires callers to
// explicitly turn the resulting review artifact into an `AcceptedGraph` — via
// `AcceptedGraph::accept(review)` (fallible; the review-artifact path) or the infallible
// `From<Dag|Admg|Pag|TemporalDag|TemporalPag>` conversions.
//
// These helpers reproduce the Python-visible `accept_discovered: bool` semantics on top of
// the new API, one per review-artifact shape:
// - DAG-shaped (`DagReview`, `TemporalGraphReview`): `true` clears pending directed edges
//   first (mirrors the old `DiscoveryAccept::AutoAccept`); `false` requires the graph to
//   already be fully oriented.
// - CPDAG-shaped (`CpdagReview`): `true` clears directed pending edges. Undirected
//   marks are the MEC and do not block acceptance (class-aware ATE consumes them).
//   `TemporalCpdagReview` is the same for undirected marks (class-aware pulse).
// - PAG-shaped (`PagReview`, `TemporalPagReview`): circle marks are informational, not
//   incompleteness (the class-aware identifiers handle them directly), so `true` accepts
//   the discovered graph as-is via the infallible `From` conversion — mirroring the old
//   unconditional `AutoAccept` success for FCI/RFCI/LPCMCI; `false` routes through the
//   review artifact, which fails with `ReviewRequired` while circles remain unreviewed.

/// DAG-shaped review (static `DagReview`; DirectLiNGAM, NOTEARS).
pub(crate) fn accept_dag_review(
    review: DagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let review = if accept_discovered { review.accept_all() } else { review };
    AcceptedGraph::accept(review)
}

/// Temporal DAG-shaped review (`TemporalGraphReview`; PCMCI).
pub(crate) fn accept_temporal_graph_review(
    review: TemporalGraphReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let pending = PendingGraphReview::new(review);
    let pending = if accept_discovered { pending.accept_all() } else { pending };
    pending.finish()
}

/// Static CPDAG-shaped review (`CpdagReview`; PC, GES).
pub(crate) fn accept_cpdag_review(
    mut review: CpdagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    if accept_discovered {
        review.pending_edges = std::sync::Arc::from([]);
    }
    AcceptedGraph::accept(review)
}

/// Temporal CPDAG-shaped review (`TemporalCpdagReview`; PCMCI+, J-PCMCI+, RPCMCI).
pub(crate) fn accept_temporal_cpdag_review(
    review: TemporalCpdagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let pending = PendingCpdagReview::new(review);
    let pending = if accept_discovered { pending.accept_all_directed() } else { pending };
    pending.finish()
}

/// Static PAG-shaped discovery output (FCI, RFCI).
pub(crate) fn accept_pag_review(
    graph: Pag,
    review: PagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    // Static PAG circle marks are information the class-aware generalized-adjustment
    // identifier consumes, so auto-accept is safe here.
    if accept_discovered { Ok(AcceptedGraph::pag(graph)) } else { AcceptedGraph::accept(review) }
}

/// Temporal PAG-shaped discovery output (LPCMCI).
pub(crate) fn accept_temporal_pag_review(
    graph: TemporalPag,
    review: TemporalPagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    // Circles are TemporalPag class information; auto-accept keeps the PAG.
    if accept_discovered {
        Ok(AcceptedGraph::temporal_pag(graph))
    } else {
        AcceptedGraph::accept(review)
    }
}

/// RPCMCI: reduce N per-regime CPDAG reviews to one accepted graph. A single accepted
/// graph is only possible when discovery found exactly one regime — multiple regimes
/// always require manual review (there is no single-graph collapse across regimes),
/// mirroring the old `compile()`'s `result.per_regime.len() == 1` gate.
pub(crate) fn accept_rpcmci_review(
    result: &antecedent::discovery::RpcmciDiscoveryResult,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let Some(first) = result.per_regime.first() else {
        return Err(RustCausalError::review_required_msg("RPCMCI discovered no regime graphs"));
    };
    if result.per_regime.len() != 1 {
        return Err(RustCausalError::review_required_msg(format!(
            "RPCMCI discovered {} regimes; a single accepted graph requires exactly one \
             (review each regime's CPDAG separately)",
            result.per_regime.len()
        )));
    }
    accept_temporal_cpdag_review(first.review.clone(), accept_discovered)
}
