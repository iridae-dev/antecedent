//! `AcceptedGraph`: the review gate moved into the type system.
//!
//! The library's committed philosophy is that identification is evaluated before
//! estimation, and that partial graphs are never silently completed. Historically that
//! guarantee was enforced by runtime checks scattered across `compile()` — all firing
//! late, long after a user had already built an analysis around an unreviewed graph.
//!
//! [`AcceptedGraph`] moves the guarantee into the type system: constructing one *is*
//! the review gate. Once a caller holds an `AcceptedGraph`, it cannot carry unresolved
//! marks, so nothing downstream needs to re-check.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_graph::{
    Admg, Cpdag, CpdagReview, Dag, DagReview, Pag, PagReview, TemporalCpdag, TemporalCpdagReview,
    TemporalDag, TemporalGraphReview, TemporalPag, TemporalPagReview,
};

use crate::error::{CausalError, ReviewKind};

/// Which graph class an [`AcceptedGraph`] holds.
///
/// Graph classes stay distinct — they are not interchangeable aliases. A [`Cpdag`]
/// that happens to be fully oriented is still class [`Self::Cpdag`], not silently
/// reinterpreted as [`Self::Dag`]; downstream code that cares about the difference
/// (e.g. identification dispatch) can still tell them apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum GraphClass {
    /// Fully directed acyclic graph.
    Dag,
    /// Acyclic directed mixed graph (directed + bidirected edges).
    Admg,
    /// Completed partial DAG (all marks directed).
    Cpdag,
    /// Partial ancestral graph (circle marks allowed).
    Pag,
    /// Fully directed temporal DAG.
    TemporalDag,
    /// Completed temporal CPDAG (all marks directed).
    TemporalCpdag,
    /// Temporal partial ancestral graph (circle marks allowed).
    TemporalPag,
}

/// Internal storage: one owned graph per supported class.
///
/// Private — callers only ever observe this through [`AcceptedGraph::class`] and the
/// `as_*` accessors, never by matching on the representation directly.
#[derive(Clone, Debug)]
enum GraphKind {
    Dag(Dag),
    Admg(Admg),
    Cpdag(Cpdag),
    Pag(Pag),
    TemporalDag(TemporalDag),
    TemporalCpdag(TemporalCpdag),
    TemporalPag(TemporalPag),
}

/// Asserted-or-accepted causal structure.
///
/// Construction is the review gate: a value of this type can never carry unresolved
/// marks that would block estimation. [`Self::dag`], [`Self::admg`], [`Self::pag`],
/// and [`Self::temporal_dag`] are infallible because those classes structurally cannot
/// be partial. [`Self::pag`] is infallible too: a *static* PAG's circle marks are
/// information the class-aware generalized-adjustment identifier is built to consume.
///
/// [`Self::cpdag`], [`Self::temporal_cpdag`], [`Self::temporal_pag`], and
/// [`Self::accept`] are fallible, because each of those *can* carry marks that block
/// estimation. Temporal PAGs are fallible where static PAGs are not for a concrete
/// reason, not symmetry: no class-aware *temporal* PAG identifier is wired, so a
/// circle mark on a temporal PAG has nothing that can consume it and genuinely blocks.
/// That asymmetry is deliberate (see the missing `From<Cpdag>` impls below).
#[derive(Clone, Debug)]
pub struct AcceptedGraph {
    kind: GraphKind,
    version: u32,
    algorithm_id: Option<Arc<str>>,
}

impl AcceptedGraph {
    fn from_kind(kind: GraphKind, algorithm_id: Option<Arc<str>>) -> Self {
        Self { kind, version: 1, algorithm_id }
    }

    /// Accept an already-directed static DAG. Cannot fail: a [`Dag`] has no undirected
    /// or circle marks by construction.
    #[must_use]
    pub fn dag(g: Dag) -> Self {
        Self::from_kind(GraphKind::Dag(g), None)
    }

    /// Accept a static ADMG. Cannot fail: bidirected edges are latent-confounder
    /// information, not pending review.
    #[must_use]
    pub fn admg(g: Admg) -> Self {
        Self::from_kind(GraphKind::Admg(g), None)
    }

    /// Accept a static PAG. Cannot fail: circle marks are information (ambiguity the
    /// class-aware identifier is designed to handle), not incompleteness.
    #[must_use]
    pub fn pag(g: Pag) -> Self {
        Self::from_kind(GraphKind::Pag(g), None)
    }

    /// Accept an already-directed temporal DAG. Cannot fail, for the same reason as
    /// [`Self::dag`].
    #[must_use]
    pub fn temporal_dag(g: TemporalDag) -> Self {
        Self::from_kind(GraphKind::TemporalDag(g), None)
    }

    /// Accept a temporal PAG, asserting no circle marks remain.
    ///
    /// Fallible where [`Self::pag`] is not: static PAG circles are consumed by the
    /// class-aware generalized-adjustment identifier, but no equivalent temporal
    /// identifier exists, so a temporal circle mark blocks estimation outright.
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when circle marks remain, carrying the count so
    /// a caller can drive a review UI rather than re-deriving it.
    pub fn temporal_pag(g: TemporalPag) -> Result<Self, CausalError> {
        let pending = TemporalPagReview::from_pag(g, "asserted");
        if !pending.is_complete() {
            return Err(CausalError::review_required(
                ReviewKind::TemporalPag.as_str(),
                None::<String>,
                pending.pending_circles.len(),
                "temporal PAG has unresolved circle marks",
                "orient the circle marks, or supply a fully directed TemporalDag \
                 (no class-aware temporal PAG identifier is wired today)",
            ));
        }
        Ok(Self::from_kind(GraphKind::TemporalPag(pending.graph), None))
    }

    /// Accept a static CPDAG, asserting it is fully oriented.
    ///
    /// Unlike [`Self::dag`] et al., a [`Cpdag`] *can* carry undirected or conflict
    /// marks, so this checks the graph itself (there is no separate review-acceptance
    /// workflow here — the caller is asserting the graph directly).
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when undirected or conflict marks remain.
    pub fn cpdag(g: Cpdag) -> Result<Self, CausalError> {
        let pending = g.undirected_edge_count() + g.conflict_edge_count();
        if pending > 0 {
            return Err(CausalError::review_required(
                ReviewKind::StaticCpdag.as_str(),
                None::<String>,
                pending,
                "CPDAG carries unresolved undirected or conflict marks",
                "orient undirected marks and resolve conflicts, or supply a fully oriented Cpdag",
            ));
        }
        Ok(Self::from_kind(GraphKind::Cpdag(g), None))
    }

    /// Accept a temporal CPDAG, asserting it is fully oriented.
    ///
    /// Same semantics as [`Self::cpdag`], for the temporal class.
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when undirected or conflict marks remain.
    pub fn temporal_cpdag(g: TemporalCpdag) -> Result<Self, CausalError> {
        let pending = g.undirected_edge_count() + g.conflict_edge_count();
        if pending > 0 {
            return Err(CausalError::review_required(
                ReviewKind::TemporalCpdag.as_str(),
                None::<String>,
                pending,
                "temporal CPDAG carries unresolved undirected or conflict marks",
                "orient undirected marks and resolve conflicts, or supply a fully oriented TemporalCpdag",
            ));
        }
        Ok(Self::from_kind(GraphKind::TemporalCpdag(g), None))
    }

    /// Complete a discovery review artifact into an accepted structure.
    ///
    /// This is the path for discovery output (`DagReview`, `CpdagReview`, `PagReview`,
    /// and their temporal counterparts): it requires the human sign-off the review
    /// artifact tracks, not just structural completeness.
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when pending edges or unresolved marks remain.
    pub fn accept<R: IntoAccepted>(review: R) -> Result<Self, CausalError> {
        review.into_accepted()
    }

    /// Which graph class this value holds.
    #[must_use]
    pub fn class(&self) -> GraphClass {
        match &self.kind {
            GraphKind::Dag(_) => GraphClass::Dag,
            GraphKind::Admg(_) => GraphClass::Admg,
            GraphKind::Cpdag(_) => GraphClass::Cpdag,
            GraphKind::Pag(_) => GraphClass::Pag,
            GraphKind::TemporalDag(_) => GraphClass::TemporalDag,
            GraphKind::TemporalCpdag(_) => GraphClass::TemporalCpdag,
            GraphKind::TemporalPag(_) => GraphClass::TemporalPag,
        }
    }

    /// Monotonic version, starting at 1. Bumped by [`Self::replace`].
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Discovery algorithm id, when this value came from a review artifact.
    /// `None` when the structure was asserted directly (`dag`, `cpdag`, …).
    #[must_use]
    pub fn algorithm_id(&self) -> Option<&str> {
        self.algorithm_id.as_deref()
    }

    /// Borrow the DAG when [`Self::class`] is [`GraphClass::Dag`].
    #[must_use]
    pub fn as_dag(&self) -> Option<&Dag> {
        match &self.kind {
            GraphKind::Dag(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the ADMG when [`Self::class`] is [`GraphClass::Admg`].
    #[must_use]
    pub fn as_admg(&self) -> Option<&Admg> {
        match &self.kind {
            GraphKind::Admg(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the CPDAG when [`Self::class`] is [`GraphClass::Cpdag`].
    #[must_use]
    pub fn as_cpdag(&self) -> Option<&Cpdag> {
        match &self.kind {
            GraphKind::Cpdag(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the PAG when [`Self::class`] is [`GraphClass::Pag`].
    #[must_use]
    pub fn as_pag(&self) -> Option<&Pag> {
        match &self.kind {
            GraphKind::Pag(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the temporal DAG when [`Self::class`] is [`GraphClass::TemporalDag`].
    #[must_use]
    pub fn as_temporal_dag(&self) -> Option<&TemporalDag> {
        match &self.kind {
            GraphKind::TemporalDag(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the temporal CPDAG when [`Self::class`] is [`GraphClass::TemporalCpdag`].
    #[must_use]
    pub fn as_temporal_cpdag(&self) -> Option<&TemporalCpdag> {
        match &self.kind {
            GraphKind::TemporalCpdag(g) => Some(g),
            _ => None,
        }
    }

    /// Borrow the temporal PAG when [`Self::class`] is [`GraphClass::TemporalPag`].
    #[must_use]
    pub fn as_temporal_pag(&self) -> Option<&TemporalPag> {
        match &self.kind {
            GraphKind::TemporalPag(g) => Some(g),
            _ => None,
        }
    }

    /// Produce a new accepted structure that supersedes this one, bumping the version.
    ///
    /// `next`'s own class, contents, and algorithm id are kept; only the version
    /// counter is derived from `self` (`self.version() + 1`). Use this to thread
    /// provenance through re-identification after an incremental structure update.
    #[must_use]
    pub fn replace(&self, next: AcceptedGraph) -> AcceptedGraph {
        AcceptedGraph {
            kind: next.kind,
            version: self.version + 1,
            algorithm_id: next.algorithm_id,
        }
    }
}

impl From<Dag> for AcceptedGraph {
    fn from(g: Dag) -> Self {
        Self::dag(g)
    }
}

impl From<Admg> for AcceptedGraph {
    fn from(g: Admg) -> Self {
        Self::admg(g)
    }
}

impl From<Pag> for AcceptedGraph {
    // Circles are information, not incompleteness: a PAG converts unconditionally.
    fn from(g: Pag) -> Self {
        Self::pag(g)
    }
}

impl From<TemporalDag> for AcceptedGraph {
    fn from(g: TemporalDag) -> Self {
        Self::temporal_dag(g)
    }
}

// Deliberately NO `From<Cpdag>` / `From<TemporalCpdag>`.
//
// Those two classes can carry unresolved undirected or conflict marks, so accepting
// one must be fallible (`AcceptedGraph::cpdag` / `AcceptedGraph::temporal_cpdag`,
// both returning `Result`) — never an infallible `From` that would let an incomplete
// CPDAG slip into an `AcceptedGraph` un-reviewed. This asymmetry with the `From` impls
// above *is* the guarantee that partial graphs are never silently completed. A future
// maintainer "fixing the inconsistency" by adding `From<Cpdag>` would silently destroy
// that guarantee — don't do it.

mod sealed {
    pub trait Sealed {}
    impl Sealed for antecedent_graph::DagReview {}
    impl Sealed for antecedent_graph::CpdagReview {}
    impl Sealed for antecedent_graph::PagReview {}
    impl Sealed for antecedent_graph::TemporalGraphReview {}
    impl Sealed for antecedent_graph::TemporalCpdagReview {}
    impl Sealed for antecedent_graph::TemporalPagReview {}
}

/// Discovery review artifacts that can complete into an [`AcceptedGraph`].
///
/// Sealed: only the review artifacts in `antecedent-graph` may be accepted — this is
/// not an extension point for arbitrary caller types.
pub trait IntoAccepted: sealed::Sealed {
    /// Complete the review into an accepted structure.
    ///
    /// # Errors
    ///
    /// Unresolved marks remain (pending edges, undirected marks, or unreviewed circles,
    /// depending on the artifact).
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError>;
}

impl IntoAccepted for DagReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            return Err(CausalError::review_required(
                ReviewKind::StaticDag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_edges.len(),
                "static DAG discovery review incomplete: pending directed edges remain",
                "accept pending directed edges or supply a fully oriented Dag",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::Dag(self.graph), Some(self.algorithm)))
    }
}

impl IntoAccepted for CpdagReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            let pending = self.pending_edges.len() + self.pending_undirected.len();
            return Err(CausalError::review_required(
                ReviewKind::StaticCpdag.as_str(),
                Some(self.algorithm.to_string()),
                pending,
                "static CPDAG review incomplete: pending edges or undirected marks remain",
                "accept pending edges and orient undirected marks before estimation",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::Cpdag(self.graph), Some(self.algorithm)))
    }
}

impl IntoAccepted for PagReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            return Err(CausalError::review_required(
                ReviewKind::StaticPag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_circles.len(),
                "static PAG review incomplete: circle-bearing edges remain unreviewed",
                "resolve circle marks, or call AcceptedGraph::pag(graph) directly — \
                 circles are safe input for generalized adjustment",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::Pag(self.graph), Some(self.algorithm)))
    }
}

impl IntoAccepted for TemporalGraphReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            return Err(CausalError::review_required(
                ReviewKind::TemporalDag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_edges.len(),
                "temporal DAG discovery review incomplete: pending edges remain",
                "accept pending edges or supply a fully oriented TemporalDag",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::TemporalDag(self.graph), Some(self.algorithm)))
    }
}

impl IntoAccepted for TemporalCpdagReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            let pending = self.pending_edges.len() + self.pending_undirected.len();
            return Err(CausalError::review_required(
                ReviewKind::TemporalCpdag.as_str(),
                Some(self.algorithm.to_string()),
                pending,
                "temporal CPDAG review incomplete: pending edges or undirected marks remain",
                "accept pending edges and orient undirected marks before estimation",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::TemporalCpdag(self.graph), Some(self.algorithm)))
    }
}

impl IntoAccepted for TemporalPagReview {
    fn into_accepted(self) -> Result<AcceptedGraph, CausalError> {
        if !self.is_complete() {
            return Err(CausalError::review_required(
                ReviewKind::TemporalPag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_circles.len(),
                "temporal PAG review incomplete: circle-bearing edges remain unreviewed",
                "resolve circle marks, or call AcceptedGraph::temporal_pag(graph) directly — \
                 circles are safe input for generalized adjustment",
            ));
        }
        Ok(AcceptedGraph::from_kind(GraphKind::TemporalPag(self.graph), Some(self.algorithm)))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::VariableId;
    use antecedent_graph::DenseNodeId;

    use super::*;

    fn toy_dag() -> Dag {
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g
    }

    #[test]
    fn from_dag_is_version_one_with_no_algorithm() {
        let accepted = AcceptedGraph::from(toy_dag());
        assert_eq!(accepted.class(), GraphClass::Dag);
        assert_eq!(accepted.version(), 1);
        assert!(accepted.algorithm_id().is_none());
        assert!(accepted.as_dag().is_some());
    }

    #[test]
    fn incomplete_cpdag_review_reports_pending_marks() {
        let mut g = Cpdag::with_variables(2);
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let review = CpdagReview::from_cpdag(g, "pc");
        let err = AcceptedGraph::accept(review).unwrap_err();
        match err {
            CausalError::ReviewRequired { kind, pending_edge_count, .. } => {
                assert_eq!(kind, ReviewKind::StaticCpdag.as_str());
                assert!(pending_edge_count > 0);
            }
            other => panic!("expected ReviewRequired, got {other:?}"),
        }
    }

    #[test]
    fn resolved_cpdag_review_accepts_with_algorithm_id() {
        let mut g = Cpdag::with_variables(2);
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let review = CpdagReview::from_cpdag(g, "pc")
            .orient_edge(VariableId::from_raw(0), VariableId::from_raw(1))
            .unwrap()
            .accept_edge(VariableId::from_raw(0), VariableId::from_raw(1));
        let accepted = AcceptedGraph::accept(review).unwrap();
        assert_eq!(accepted.class(), GraphClass::Cpdag);
        assert_eq!(accepted.algorithm_id(), Some("pc"));
    }

    #[test]
    fn replace_bumps_version() {
        let first = AcceptedGraph::from(toy_dag());
        let second = first.replace(AcceptedGraph::from(toy_dag()));
        assert_eq!(second.version(), 2);
        assert_eq!(first.version(), 1);
    }
}
