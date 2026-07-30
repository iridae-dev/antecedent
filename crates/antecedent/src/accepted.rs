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

use antecedent_core::{CausalSchema, NodeRef, TemporalNodeKey, VariableId};
use antecedent_graph::{
    Admg, Cpdag, CpdagReview, Dag, DagReview, DenseNodeId, Endpoint, Pag, PagReview, TemporalCpdag,
    TemporalCpdagReview, TemporalDag, TemporalGraphReview, TemporalPag, TemporalPagReview,
};

use crate::error::{CausalError, PendingEdge, ReviewKind};

/// Human-readable identifier for a static [`VariableId`]-addressed node (`"V3"`).
fn variable_name(id: VariableId) -> String {
    id.to_string()
}

/// Identifier for a temporal node key: `"V3@-1"`, or `"V3@0"` for contemporaneous.
fn temporal_key_name(key: TemporalNodeKey) -> String {
    format!("{}@{}", key.variable, key.offset)
}

/// Identifier for a [`NodeRef`], matching [`temporal_key_name`]'s `variable@offset`
/// scheme (used where only the pre-dense-indexing node identity is available, e.g.
/// [`TemporalPag`]'s node table).
fn node_ref_name(node: NodeRef) -> String {
    match node {
        NodeRef::Static(v) | NodeRef::Context { variable: v, .. } => variable_name(v),
        NodeRef::Lagged { variable, lag } if lag.raw() == 0 => format!("{variable}@0"),
        NodeRef::Lagged { variable, lag } => format!("{variable}@-{}", lag.raw()),
    }
}

/// Wire string for an [`Endpoint`] mark: `"tail"`, `"arrow"`, `"circle"`, or
/// `"conflict"` — the same vocabulary the Python `GraphEdge` binding uses.
const fn endpoint_str(mark: Endpoint) -> &'static str {
    match mark {
        Endpoint::Tail => "tail",
        Endpoint::Arrow => "arrow",
        Endpoint::Circle => "circle",
        Endpoint::Conflict => "conflict",
    }
}

/// [`PendingEdge`]s for `(from, to)` pairs that are always tail-at-source,
/// arrow-at-target: accepted-DAG-edge and directed-CPDAG-edge pending lists only ever
/// hold Tail-Arrow pairs (see `MarkedEdge::parent_child`), so the marks need no graph
/// lookup.
fn pending_directed(edges: &[(VariableId, VariableId)]) -> Vec<PendingEdge> {
    edges
        .iter()
        .map(|&(from, to)| {
            PendingEdge::new(variable_name(from), variable_name(to), "tail", "arrow")
        })
        .collect()
}

/// [`PendingEdge`]s for undirected `(a, b)` pairs (tail-tail at both ends).
fn pending_undirected_variable(edges: &[(VariableId, VariableId)]) -> Vec<PendingEdge> {
    edges
        .iter()
        .map(|&(a, b)| PendingEdge::new(variable_name(a), variable_name(b), "tail", "tail"))
        .collect()
}

/// Temporal counterpart of [`pending_directed`].
fn pending_directed_temporal(edges: &[(TemporalNodeKey, TemporalNodeKey)]) -> Vec<PendingEdge> {
    edges
        .iter()
        .map(|&(from, to)| {
            PendingEdge::new(temporal_key_name(from), temporal_key_name(to), "tail", "arrow")
        })
        .collect()
}

/// Temporal counterpart of [`pending_undirected_variable`].
fn pending_undirected_temporal(edges: &[(TemporalNodeKey, TemporalNodeKey)]) -> Vec<PendingEdge> {
    edges
        .iter()
        .map(|&(a, b)| PendingEdge::new(temporal_key_name(a), temporal_key_name(b), "tail", "tail"))
        .collect()
}

/// [`PendingEdge`]s for circle-bearing static PAG edges, resolving each pair's real
/// marks from the graph rather than assuming both ends are circles (one side can be a
/// definite tail or arrow with only the other circled).
fn pending_pag_circles(graph: &Pag, circles: &[(DenseNodeId, DenseNodeId)]) -> Vec<PendingEdge> {
    circles
        .iter()
        .filter_map(|&(a, b)| {
            // Pag nodes are positional: DenseNodeId(i) is VariableId(i) (see
            // AcceptedGraph's own doc comment on this invariant).
            graph.edge_between(a, b).map(|edge| {
                PendingEdge::new(
                    variable_name(VariableId::from_raw(a.raw())),
                    variable_name(VariableId::from_raw(b.raw())),
                    endpoint_str(edge.at_a),
                    endpoint_str(edge.at_b),
                )
            })
        })
        .collect()
}

/// Temporal counterpart of [`pending_pag_circles`]. Unlike the static case,
/// [`TemporalPag`] dense ids are not positional variable ids, so each endpoint's
/// [`NodeRef`] is read from the graph's node table.
fn pending_temporal_pag_circles(
    graph: &TemporalPag,
    circles: &[(DenseNodeId, DenseNodeId)],
) -> Vec<PendingEdge> {
    circles
        .iter()
        .filter_map(|&(a, b)| {
            let edge = graph.edge_between(a, b)?;
            let source = *graph.nodes().get(a.as_usize())?;
            let target = *graph.nodes().get(b.as_usize())?;
            Some(PendingEdge::new(
                node_ref_name(source),
                node_ref_name(target),
                endpoint_str(edge.at_a),
                endpoint_str(edge.at_b),
            ))
        })
        .collect()
}

/// [`PendingEdge`]s for a raw (directly asserted) [`Cpdag`]'s undirected and conflict
/// marks — used by [`AcceptedGraph::cpdag`], which asserts a graph directly rather
/// than completing a review artifact, so there is no stored pending list to draw from.
fn pending_cpdag_marks(graph: &Cpdag) -> Vec<PendingEdge> {
    graph
        .edges()
        .into_iter()
        .filter(|e| e.is_undirected() || e.is_conflict())
        .map(|e| {
            let source =
                graph.variable_id(e.a).map_or_else(|| format!("N{}", e.a.raw()), variable_name);
            let target =
                graph.variable_id(e.b).map_or_else(|| format!("N{}", e.b.raw()), variable_name);
            PendingEdge::new(source, target, endpoint_str(e.at_a), endpoint_str(e.at_b))
        })
        .collect()
}

/// Temporal counterpart of [`pending_cpdag_marks`].
fn pending_temporal_cpdag_marks(graph: &TemporalCpdag) -> Vec<PendingEdge> {
    graph
        .edges()
        .into_iter()
        .filter(|e| e.is_undirected() || e.is_conflict())
        .map(|e| {
            let source = graph
                .temporal_key(e.a)
                .map_or_else(|| format!("N{}", e.a.raw()), temporal_key_name);
            let target = graph
                .temporal_key(e.b)
                .map_or_else(|| format!("N{}", e.b.raw()), temporal_key_name);
            PendingEdge::new(source, target, endpoint_str(e.at_a), endpoint_str(e.at_b))
        })
        .collect()
}

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
    /// Ordered variable names this structure's node indices refer to, when bound via
    /// [`Self::with_schema`]. `None` when the structure was never bound (the default) —
    /// binding is opt-in so existing callers are unaffected.
    schema_names: Option<Arc<[Arc<str>]>>,
}

impl AcceptedGraph {
    fn from_kind(kind: GraphKind, algorithm_id: Option<Arc<str>>) -> Self {
        Self { kind, version: 1, algorithm_id, schema_names: None }
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
            let edges = pending_temporal_pag_circles(&pending.graph, &pending.pending_circles);
            return Err(CausalError::review_required(
                ReviewKind::TemporalPag.as_str(),
                None::<String>,
                pending.pending_circles.len(),
                edges,
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
            let edges = pending_cpdag_marks(&g);
            return Err(CausalError::review_required(
                ReviewKind::StaticCpdag.as_str(),
                None::<String>,
                pending,
                edges,
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
            let edges = pending_temporal_cpdag_marks(&g);
            return Err(CausalError::review_required(
                ReviewKind::TemporalCpdag.as_str(),
                None::<String>,
                pending,
                edges,
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
            schema_names: next.schema_names,
        }
    }

    /// Bind this structure to the schema its node indices refer to.
    ///
    /// Graph nodes are positional (`DenseNodeId(i)` is `VariableId(i)`), so a structure
    /// built against one schema is silently meaningless against another with the same
    /// shape. Binding lets [`crate::StudyBuilder::build`] refuse that.
    #[must_use]
    pub fn with_schema(mut self, schema: &CausalSchema) -> Self {
        let names: Arc<[Arc<str>]> =
            schema.variables().iter().map(|v| Arc::clone(&v.name)).collect();
        self.schema_names = Some(names);
        self
    }

    /// Ordered variable names this structure is bound to, if any.
    #[must_use]
    pub fn variable_names(&self) -> Option<&[Arc<str>]> {
        self.schema_names.as_deref()
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
                pending_directed(&self.pending_edges),
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
            let mut edges = pending_directed(&self.pending_edges);
            edges.extend(pending_undirected_variable(&self.pending_undirected));
            return Err(CausalError::review_required(
                ReviewKind::StaticCpdag.as_str(),
                Some(self.algorithm.to_string()),
                pending,
                edges,
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
            let edges = pending_pag_circles(&self.graph, &self.pending_circles);
            return Err(CausalError::review_required(
                ReviewKind::StaticPag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_circles.len(),
                edges,
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
                pending_directed_temporal(&self.pending_edges),
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
            let mut edges = pending_directed_temporal(&self.pending_edges);
            edges.extend(pending_undirected_temporal(&self.pending_undirected));
            return Err(CausalError::review_required(
                ReviewKind::TemporalCpdag.as_str(),
                Some(self.algorithm.to_string()),
                pending,
                edges,
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
            let edges = pending_temporal_pag_circles(&self.graph, &self.pending_circles);
            return Err(CausalError::review_required(
                ReviewKind::TemporalPag.as_str(),
                Some(self.algorithm.to_string()),
                self.pending_circles.len(),
                edges,
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
            CausalError::ReviewRequired { kind, pending_edge_count, pending_edges, .. } => {
                assert_eq!(kind, ReviewKind::StaticCpdag.as_str());
                assert!(pending_edge_count > 0);
                // The pending edge list must carry the real (undirected, tail-tail)
                // mark, not just the count.
                assert_eq!(pending_edges.len(), pending_edge_count);
                let edge = &pending_edges[0];
                assert_eq!(edge.at_source, "tail");
                assert_eq!(edge.at_target, "tail");
                assert!(!edge.source.is_empty());
                assert!(!edge.target.is_empty());
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
