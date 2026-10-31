//! Transport across a discovered *temporal* equivalence class at a declared horizon.
//!
//! A temporal structure is a stationary template over lagged nodes. A request names a
//! treatment, an outcome and a horizon: the effect of an intervention on the treatment at the
//! origin on the outcome `horizon_steps` later. This module unrolls every member of the class
//! over the finite window the request needs, declares the selection targets at every slice of
//! the window, and asks the structural transport identifier for the verdict, so the question
//! is the static one posed on the unrolled graph.
//!
//! # Members
//!
//! * [`TemporalClass::Dag`]: one member, the template itself (a lagged-only discovery).
//! * [`TemporalClass::Cpdag`]: the stationary DAG completions of the template. An undirected
//!   contemporaneous edge is oriented the same way at every slice, so a completion never has
//!   a backward arrow in time and never creates a collider the class does not have.
//! * [`TemporalClass::Pag`]: the stationary MAG completions that are valid in the request's
//!   window, each read as the most-confounded ADMG it is compatible with (see
//!   [`crate::discovered_transport`]). A circle endpoint is enumerated, never read as the
//!   absence of confounding.
//!
//! # The window and the history before it
//!
//! The window runs from `history` slices before the origin to the outcome slice. The process
//! before the window is not discarded: a window node that the template links to an earlier,
//! unobserved node is *exposed* to that history, and two exposed nodes may share a latent
//! ancestor there. Every pair of exposed nodes is therefore given a bidirected edge. That
//! supergraph contains the latent projection of the stationary process onto the window (a
//! bidirected edge of the projection joins two nodes that each have an arrowhead edge from
//! the earlier history), so a functional derived on it holds for the stationary process; it
//! may refuse an effect the exact projection would identify, never the reverse. An
//! autoregressive edge therefore does not make the question unanswerable.
//!
//! # Selection targets
//!
//! A selection target is a variable whose mechanism differs between the populations. It is
//! declared on the template variable and applies at every slice, the stationarity assumption
//! the declaration carries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::result_large_err,
    reason = "the public error shape is kept as released; boxing the refusal would break callers that match on it"
)]

use std::sync::Arc;

use antecedent_core::{
    ContinuousDomain, ExecutionContext, NodeRef, ResponseFunctional, ResponseQuery, SearchLimits,
    TemporalIndexer, TemporalNodeKey, TransportQuery, VariableId,
};
use antecedent_graph::{
    Admg, DenseNodeId, Endpoint, MarkedEdge, Pag, TemporalCpdag, TemporalCpdagCompletionSampler,
    TemporalDag, TemporalPag, TemporalPagCompletionSampler,
};

use crate::discovered_transport::{
    DiscoveredTransportError, EnvelopeStatus, MemberVerdict, RequestFacts, UnresolvedPair,
    ancestral_subgraph, check_request, formulas_agree_over, mag_member, refusal, status_of,
    verdict_of,
};
use crate::error::IdentificationError;
use crate::transport::TransportIdentifier;

/// A discovered temporal structure.
#[derive(Clone, Copy, Debug)]
pub enum TemporalClass<'a> {
    /// A lagged DAG: its own and only member.
    Dag(&'a TemporalDag),
    /// A temporal CPDAG.
    Cpdag(&'a TemporalCpdag),
    /// A temporal PAG.
    Pag(&'a TemporalPag),
}

/// A request for the structural transport verdict across a discovered temporal class.
#[derive(Clone, Copy, Debug)]
pub struct TemporalTransportRequest<'a> {
    /// The discovered temporal structure.
    pub class: TemporalClass<'a>,
    /// Variables whose mechanisms may differ between populations, at every slice (declared,
    /// not discovered).
    pub selection_targets: &'a [VariableId],
    /// How many search bounds discovery exhausted. Non-zero refuses.
    pub exhausted_search_bounds: usize,
    /// Most members to enumerate. Zero is refused.
    pub member_bound: usize,
    /// Operations the completion walk may charge; see
    /// [`crate::discovered_transport::PagTransportRequest::completion_operations`].
    pub completion_operations: usize,
    /// Steps from the intervention at the origin to the outcome. Zero asks for the
    /// contemporaneous effect.
    pub horizon_steps: u32,
    /// Slices of history in the window. `None` is twice the template's largest lag.
    pub history: Option<u32>,
}

/// An edge of a stationary template, `from` at `t - lag` and `to` at `t`.
///
/// A directed edge goes from `from` to `to` (it never points back in time, so `lag` is not
/// negative). A bidirected edge is written with the earlier end as `from` (and, at lag zero,
/// the smaller variable first).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LaggedEdge {
    /// The variable at the earlier end.
    pub from: VariableId,
    /// The variable at the later end.
    pub to: VariableId,
    /// Steps between the two ends.
    pub lag: u32,
}

/// One member of a temporal class and its verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalEnvelopeMember {
    /// Directed template edges of the member, ascending.
    pub directed_edges: Vec<LaggedEdge>,
    /// Bidirected template edges of a PAG member, ascending: a latent common cause.
    pub bidirected_edges: Vec<LaggedEdge>,
    /// Directed template edges of a PAG member with an instance that is not visible in the
    /// window, ascending: the member does not exclude a latent common cause of such a pair.
    pub confounded_edges: Vec<LaggedEdge>,
    /// The structural verdict for this member at the declared horizon.
    pub verdict: MemberVerdict,
    graph: Admg,
}

impl TemporalEnvelopeMember {
    /// The window graph the verdict was decided on: the unrolled member, its nodes labelled by
    /// their dense window ids, with the bidirected edges for non-visible edges and for the
    /// history before the window already added.
    #[must_use]
    pub const fn window_graph(&self) -> &Admg {
        &self.graph
    }
}

/// The unrolled question: the finite window and the query on its slots.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalWindow {
    /// Window coordinates: a node's variable id is its dense id.
    pub indexer: TemporalIndexer,
    /// The query on the window's slots.
    pub query: TransportQuery,
    /// Window slots whose mechanism may differ between populations: the declared targets at
    /// every slice.
    pub selection_targets: Arc<[VariableId]>,
    /// Slot of the treatment at the origin.
    pub treatment: VariableId,
    /// Slot of the outcome at the declared horizon.
    pub outcome: VariableId,
}

/// The structural transport verdicts across a discovered temporal class.
///
/// Like the static envelope: never a point, an interval or a probability over members.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalTransportEnvelope {
    members: Vec<TemporalEnvelopeMember>,
    unresolved: Vec<UnresolvedPair>,
    enumeration_complete: bool,
    member_bound: usize,
    history: u32,
    horizon_steps: u32,
    max_lag: u32,
    window: TemporalWindow,
}

impl TemporalTransportEnvelope {
    /// The unrolled window and query every member was decided on.
    #[must_use]
    pub const fn window(&self) -> &TemporalWindow {
        &self.window
    }

    /// What this object is, as a stable label for reports: always `structural_envelope`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        "structural_envelope"
    }

    /// The enumerated members with their verdicts.
    #[must_use]
    pub fn members(&self) -> &[TemporalEnvelopeMember] {
        &self.members
    }

    /// The template's unresolved pairs (lag-free).
    #[must_use]
    pub fn unresolved(&self) -> &[UnresolvedPair] {
        &self.unresolved
    }

    /// Whether the enumeration finished within the member bound and the search budget.
    #[must_use]
    pub const fn enumeration_complete(&self) -> bool {
        self.enumeration_complete
    }

    /// The member bound the enumeration ran under.
    #[must_use]
    pub const fn member_bound(&self) -> usize {
        self.member_bound
    }

    /// Slices of history in the window.
    #[must_use]
    pub const fn history(&self) -> u32 {
        self.history
    }

    /// Steps from the intervention to the outcome.
    #[must_use]
    pub const fn horizon_steps(&self) -> u32 {
        self.horizon_steps
    }

    /// The template's largest lag: the lag assumption the class carries.
    #[must_use]
    pub const fn max_lag(&self) -> u32 {
        self.max_lag
    }

    /// Enumerated members certified by an implemented rule.
    #[must_use]
    pub fn identified_members(&self) -> usize {
        self.members
            .iter()
            .filter(|m| matches!(m.verdict, MemberVerdict::Identified { .. }))
            .count()
    }

    /// The summary reading; see [`crate::discovered_transport`].
    #[must_use]
    pub fn status(&self) -> EnvelopeStatus {
        status_of(self.members.iter().map(|m| &m.verdict), self.enumeration_complete)
    }

    /// Whether every enumerated member is certified with the same formula.
    #[must_use]
    pub fn formulas_agree(&self) -> bool {
        formulas_agree_over(self.members.iter().map(|m| &m.verdict))
    }
}

/// A template edge with its endpoints as `(variable, offset)` and its marks.
#[derive(Clone, Copy)]
struct TemplateEdge {
    a: (VariableId, i64),
    b: (VariableId, i64),
    at_a: Endpoint,
    at_b: Endpoint,
}

fn lagged_node(nodes: &[NodeRef], id: DenseNodeId) -> Option<(VariableId, i64)> {
    match nodes.get(id.as_usize())? {
        NodeRef::Lagged { variable, lag } => Some((*variable, -i64::from(lag.raw()))),
        _ => None,
    }
}

fn template_edges(
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
) -> Vec<TemplateEdge> {
    edges
        .into_iter()
        .filter_map(|edge| {
            Some(TemplateEdge {
                a: lagged_node(nodes, edge.a)?,
                b: lagged_node(nodes, edge.b)?,
                at_a: edge.at_a,
                at_b: edge.at_b,
            })
        })
        .collect()
}

/// A bidirected edge as a [`LaggedEdge`], the earlier end first.
fn bidirected_lagged(first: (VariableId, i64), second: (VariableId, i64)) -> LaggedEdge {
    let (early, late) = if (first.1, first.0.raw()) <= (second.1, second.0.raw()) {
        (first, second)
    } else {
        (second, first)
    };
    LaggedEdge { from: early.0, to: late.0, lag: lag_between(early.1, late.1) }
}

/// A directed edge from `parent` to `child` as a [`LaggedEdge`]. A directed edge never points
/// back in time, so `child` is never earlier than `parent`.
fn directed_lagged(parent: (VariableId, i64), child: (VariableId, i64)) -> LaggedEdge {
    LaggedEdge { from: parent.0, to: child.0, lag: lag_between(parent.1, child.1) }
}

fn lag_between(early: i64, late: i64) -> u32 {
    u32::try_from(late - early).unwrap_or(u32::MAX)
}

/// The template edge as a directed [`LaggedEdge`], when it has a tail at one end and an
/// arrowhead at the other.
fn directed_of(edge: &TemplateEdge) -> Option<LaggedEdge> {
    match (edge.at_a, edge.at_b) {
        (Endpoint::Tail, Endpoint::Arrow) => Some(directed_lagged(edge.a, edge.b)),
        (Endpoint::Arrow, Endpoint::Tail) => Some(directed_lagged(edge.b, edge.a)),
        _ => None,
    }
}

fn template_max_lag(nodes: &[NodeRef]) -> u32 {
    nodes
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn class_nodes(class: TemporalClass<'_>) -> &[NodeRef] {
    match class {
        TemporalClass::Dag(graph) => graph.nodes(),
        TemporalClass::Cpdag(graph) => graph.nodes(),
        TemporalClass::Pag(graph) => graph.nodes(),
    }
}

fn unresolved_pairs(class: TemporalClass<'_>) -> Vec<UnresolvedPair> {
    let (nodes, edges): (&[NodeRef], Vec<MarkedEdge>) = match class {
        TemporalClass::Dag(_) => return Vec::new(),
        TemporalClass::Cpdag(graph) => (graph.nodes(), graph.edges()),
        TemporalClass::Pag(graph) => (graph.nodes(), graph.edges()),
    };
    let mut pairs: Vec<UnresolvedPair> = edges
        .iter()
        .filter_map(|edge| {
            let reason = match (edge.at_a, edge.at_b) {
                (Endpoint::Conflict, _) | (_, Endpoint::Conflict) => "orientation_conflict",
                (Endpoint::Circle, _) | (_, Endpoint::Circle) => "circle_endpoint",
                (Endpoint::Tail, Endpoint::Tail) => "undirected_in_equivalence_class",
                (Endpoint::Arrow, Endpoint::Arrow) => "bidirected_latent_confounding",
                _ => return None,
            };
            let (a, _) = lagged_node(nodes, edge.a)?;
            let (b, _) = lagged_node(nodes, edge.b)?;
            let (a, b) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
            Some(UnresolvedPair { a, b, reason })
        })
        .collect();
    pairs.sort_by_key(|pair| (pair.a.raw(), pair.b.raw(), pair.reason));
    pairs.dedup();
    pairs
}

/// Dense ids of the window nodes the template links, by an arrowhead edge, to a node before
/// the window.
fn exposed_nodes(
    edges: &[TemplateEdge],
    indexer: &TemporalIndexer,
) -> Result<Vec<u32>, DiscoveredTransportError> {
    let first = -i64::from(indexer.history());
    let last = i64::from(indexer.horizon()) - 1;
    let mut exposed = std::collections::BTreeSet::new();
    for edge in edges {
        for (inner, outer, at_inner) in [(edge.b, edge.a, edge.at_b), (edge.a, edge.b, edge.at_a)] {
            if at_inner != Endpoint::Arrow {
                continue;
            }
            for offset in first..=last {
                if offset + (outer.1 - inner.1) >= first {
                    continue;
                }
                let key = TemporalNodeKey {
                    variable: inner.0,
                    offset: i32::try_from(offset).map_err(|_| window_error("window offset"))?,
                };
                exposed.insert(indexer.dense_id(key).map_err(|e| window_error(&e.to_string()))?);
            }
        }
    }
    Ok(exposed.into_iter().collect())
}

fn window_error(message: &str) -> DiscoveredTransportError {
    DiscoveredTransportError::Identification(IdentificationError::invalid_input(message))
}

fn graph_error(error: &antecedent_graph::GraphError) -> DiscoveredTransportError {
    match error {
        antecedent_graph::GraphError::TemporalMaterialization(error) => {
            DiscoveredTransportError::Identification(crate::error::temporal_materialization_error(
                error.clone(),
            ))
        }
        _ => window_error(&error.to_string()),
    }
}

/// Add a bidirected edge between every pair of `nodes` that does not already have one.
fn connect_exposed(admg: &mut Admg, nodes: &[u32]) -> Result<(), DiscoveredTransportError> {
    for (i, &a) in nodes.iter().enumerate() {
        for &b in &nodes[i + 1..] {
            let (a, b) = (DenseNodeId::from_raw(a), DenseNodeId::from_raw(b));
            if !admg.bidirected_neighbors(a).contains(&b) {
                admg.insert_bidirected(a, b).map_err(|e| graph_error(&e))?;
            }
        }
    }
    Ok(())
}

/// The window ADMG of a DAG member: its unfolding, relabelled so a node's variable id is its
/// dense id.
fn dag_window(
    dag: &TemporalDag,
    indexer: &TemporalIndexer,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Admg, DiscoveredTransportError> {
    let unfolded = dag.unfold_within(indexer.clone(), budget, ctx).map_err(|e| graph_error(&e))?;
    let n = u32::try_from(indexer.dense_len()).map_err(|_| window_error("window size"))?;
    let mut admg = Admg::with_variables(n);
    for edge in unfolded.dag.edges() {
        if let Some((parent, child)) = edge.parent_child() {
            admg.insert_directed(parent, child).map_err(|e| graph_error(&e))?;
        }
    }
    Ok(admg)
}

/// A window MAG with its dense ids as static variable ids, so the identifier can address it.
fn relabelled(pag: &Pag) -> Result<Pag, DiscoveredTransportError> {
    let n = u32::try_from(pag.node_count()).map_err(|_| window_error("window size"))?;
    let mut out = Pag::with_variables(n);
    for i in 0..pag.node_count() {
        for (j, at_i, at_j) in pag.neighbors(DenseNodeId::from_index(i)) {
            if j.as_usize() > i {
                out.insert_marked(MarkedEdge {
                    a: DenseNodeId::from_index(i),
                    b: j,
                    at_a: at_i,
                    at_b: at_j,
                    middle: antecedent_graph::MiddleMark::Empty,
                })
                .map_err(|e| graph_error(&e))?;
            }
        }
    }
    Ok(out)
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "the labelling callback of mag_member is fallible; the dense id always maps"
)]
fn dense_label(id: DenseNodeId) -> Result<VariableId, DiscoveredTransportError> {
    Ok(VariableId::from_raw(id.raw()))
}

/// The edge set of a window edge named by dense ids, as a template [`LaggedEdge`].
fn window_edge(
    indexer: &TemporalIndexer,
    from: VariableId,
    to: VariableId,
) -> Result<LaggedEdge, DiscoveredTransportError> {
    let key = |id: VariableId| {
        indexer
            .key_of(id.raw())
            .map(|k| (k.variable, i64::from(k.offset)))
            .map_err(|e| window_error(&e.to_string()))
    };
    Ok(directed_lagged(key(from)?, key(to)?))
}

fn sorted_unique(mut edges: Vec<LaggedEdge>) -> Vec<LaggedEdge> {
    edges.sort_unstable();
    edges.dedup();
    edges
}

fn invalid_query(facts: &RequestFacts<'_>, message: String) -> DiscoveredTransportError {
    refusal("discovered_transport.temporal_query_invalid", facts, Vec::new(), message)
}

/// The template query unrolled onto the window: its intervention at the origin, its outcome at
/// the horizon, its experiments on the origin slot and its selection targets at every slice.
fn window_query(
    facts: &RequestFacts<'_>,
    template: &TransportQuery,
    indexer: &TemporalIndexer,
    horizon_steps: u32,
) -> Result<TemporalWindow, DiscoveredTransportError> {
    let ResponseFunctional::MeanCurve { outcome, treatment } = &template.response.functional else {
        return Err(invalid_query(
            facts,
            "a temporal transport request asks for the mean response curve of one outcome under \
             one treatment; other response functionals are not unrolled"
                .to_owned(),
        ));
    };
    let origin = |variable: VariableId, offset: i32| {
        indexer.dense_id(TemporalNodeKey { variable, offset }).map(VariableId::from_raw).map_err(
            |_| {
                invalid_query(
                    facts,
                    format!(
                        "variable v{} at offset {offset} is outside the window of the request",
                        variable.raw()
                    ),
                )
            },
        )
    };
    let outcome_offset = i32::try_from(horizon_steps)
        .map_err(|_| invalid_query(facts, "the horizon does not fit a window offset".to_owned()))?;
    let treated = origin(treatment.variable, 0)?;
    let measured = origin(*outcome, outcome_offset)?;
    if treated == measured {
        return Err(invalid_query(
            facts,
            "the treatment and the outcome are one node at a horizon of zero; ask for a different \
             outcome or a later horizon"
                .to_owned(),
        ));
    }
    if template.source_experiments.iter().any(|e| *e != treatment.variable) {
        return Err(invalid_query(
            facts,
            "a source experiment of a temporal request is an experiment on the treatment at the \
             origin; other experiments are not unrolled"
                .to_owned(),
        ));
    }
    let experiments: Vec<VariableId> =
        template.source_experiments.iter().map(|_| treated).collect();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: measured,
            treatment: ContinuousDomain::new(treated, treatment.grid.clone()),
        }),
        Arc::clone(&template.source_population),
        Arc::clone(&template.target_population),
        experiments,
    );
    let first = -i64::from(indexer.history());
    let last = i64::from(indexer.horizon()) - 1;
    let mut selection = Vec::new();
    for variable in facts.selection_targets {
        for offset in first..=last {
            let key = TemporalNodeKey {
                variable: *variable,
                offset: i32::try_from(offset).map_err(|_| window_error("window offset"))?,
            };
            let id = indexer.dense_id(key).map_err(|e| window_error(&e.to_string()))?;
            selection.push(VariableId::from_raw(id));
        }
    }
    selection.sort_unstable_by_key(|v| v.raw());
    selection.dedup();
    Ok(TemporalWindow {
        indexer: indexer.clone(),
        query,
        selection_targets: selection.into(),
        treatment: treated,
        outcome: measured,
    })
}

/// The structural transport verdict of `query` at a declared horizon across every member of a
/// discovered temporal class.
///
/// `query` is stated on the template's variables: a mean-curve response of one outcome under
/// one treatment, with the treatment's grid and the populations. It is unrolled onto the
/// window the request needs and identified in every member; see the module documentation for
/// the window, the history before it, and what a selection target means.
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] for an exhausted search bound, an orientation
/// conflict, a selection target or query variable that is not in the structure, a zero member
/// bound, a query that cannot be unrolled, a class that cannot be enumerated, or a class with
/// no member; [`DiscoveredTransportError::Identification`] when the identifier or the
/// unrolling fails.
#[expect(
    clippy::too_many_lines,
    reason = "one linear request pipeline: validate, unroll the window, enumerate the class by kind, assemble the envelope"
)]
pub fn transport_across_temporal_class(
    request: &TemporalTransportRequest<'_>,
    query: &TransportQuery,
) -> Result<TemporalTransportEnvelope, DiscoveredTransportError> {
    let facts = RequestFacts {
        selection_targets: request.selection_targets,
        exhausted_search_bounds: request.exhausted_search_bounds,
        member_bound: request.member_bound,
    };
    let nodes = class_nodes(request.class);
    let unresolved = unresolved_pairs(request.class);
    let mut known: Vec<VariableId> = nodes
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { variable, .. } => Some(*variable),
            _ => None,
        })
        .collect();
    known.sort_unstable_by_key(|v| v.raw());
    known.dedup();
    check_request(&facts, &unresolved, |target| known.contains(&target))?;

    let (treatment, outcome) = query.response.functional.primary_pair().ok_or_else(|| {
        invalid_query(&facts, "the query names no treatment and outcome".to_owned())
    })?;
    if let Some(missing) = [treatment, outcome].into_iter().find(|v| !known.contains(v)) {
        return Err(invalid_query(
            &facts,
            format!("v{} is not a variable of the discovered temporal structure", missing.raw()),
        ));
    }
    let max_lag = template_max_lag(nodes);
    let history = request.history.unwrap_or_else(|| max_lag.saturating_mul(2));
    let variable_count = match known.last() {
        Some(variable) => variable.raw().checked_add(1).ok_or_else(|| {
            invalid_query(&facts, "the temporal variable count overflows".to_owned())
        })?,
        None => 1,
    };
    let horizon = request
        .horizon_steps
        .checked_add(1)
        .ok_or_else(|| invalid_query(&facts, "the horizon overflows".to_owned()))?;
    let indexer = TemporalIndexer::new(variable_count, history, horizon)
        .map_err(|e| invalid_query(&facts, format!("the window cannot be built: {e}")))?;
    let window = window_query(&facts, query, &indexer, request.horizon_steps)?;

    let limits = SearchLimits { operations: request.completion_operations, depth: usize::MAX };
    let ctx = ExecutionContext::production(0, 1);
    let mut temporal_budget =
        antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let identifier = TransportIdentifier::new();
    let too_large = |detail: String| {
        refusal("discovered_transport.class_too_large", &facts, unresolved.clone(), detail)
    };
    let mut members = Vec::new();
    let mut enumeration_complete = true;

    match request.class {
        TemporalClass::Dag(dag) => {
            members.push(dag_member(
                dag,
                &indexer,
                &window,
                &identifier,
                &mut temporal_budget,
                &ctx,
            )?);
        }
        TemporalClass::Cpdag(cpdag) => {
            let mut sampler = TemporalCpdagCompletionSampler::with_budget(
                cpdag.clone(),
                request.member_bound,
                limits,
                &ctx,
            )
            .map_err(|error| too_large(format!("the class cannot be enumerated ({error})")))?;
            let completions: Vec<TemporalDag> =
                sampler.by_ref().map(|completion| completion.graph).collect();
            enumeration_complete = !sampler.hit_cap();
            for dag in &completions {
                members.push(dag_member(
                    dag,
                    &indexer,
                    &window,
                    &identifier,
                    &mut temporal_budget,
                    &ctx,
                )?);
            }
        }
        TemporalClass::Pag(pag) => {
            let mut sampler =
                TemporalPagCompletionSampler::for_window_with_borrowed_temporal_budget(
                    pag,
                    request.member_bound,
                    &indexer,
                    limits,
                    &mut temporal_budget,
                    &ctx,
                )
                .map_err(|error| match error {
                    antecedent_graph::GraphError::TemporalMaterialization(_) => graph_error(&error),
                    _ => too_large(format!("the class cannot be enumerated ({error})")),
                })?;
            let completions: Vec<TemporalPag> =
                sampler.by_ref().map(|completion| completion.graph).collect();
            if sampler.validation_report().ambiguous_global_class {
                return Err(refusal(
                    "discovered_transport.pag_class_ambiguous",
                    &facts,
                    unresolved,
                    "the circle endpoints admit completions from more than one Markov \
                     equivalence class in the window, so the structure is not the PAG of one \
                     class. Rerun discovery, or declare the graph"
                        .to_owned(),
                ));
            }
            enumeration_complete = !sampler.hit_cap() && !sampler.class_audit_incomplete();
            for mag in &completions {
                let Some(member) = mag_window_member(
                    mag,
                    &indexer,
                    &window,
                    &identifier,
                    &mut temporal_budget,
                    &ctx,
                )?
                else {
                    return Err(refusal(
                        "discovered_transport.pag_selection_bias_edge",
                        &facts,
                        unresolved,
                        "a completion has a tail-tail edge, which stands for selection bias \
                         in the sample. The transport rules do not model it; remove the edge \
                         with background knowledge, or declare the graph"
                            .to_owned(),
                    ));
                };
                members.push(member);
            }
        }
    }
    if members.is_empty() {
        return Err(refusal(
            "discovered_transport.no_members",
            &facts,
            unresolved,
            "the temporal structure has no stationary completion that is valid in the window \
             without a new collider, so it is not the class of any process; rerun discovery or \
             declare the graph"
                .to_owned(),
        ));
    }
    members.sort_by(|a, b| {
        (&a.directed_edges, &a.bidirected_edges).cmp(&(&b.directed_edges, &b.bidirected_edges))
    });
    Ok(TemporalTransportEnvelope {
        members,
        unresolved,
        enumeration_complete,
        member_bound: request.member_bound,
        history,
        horizon_steps: request.horizon_steps,
        max_lag,
        window,
    })
}

/// The window's selection targets that are nodes of the restricted graph.
fn kept_selection(admg: &Admg, window: &TemporalWindow) -> Arc<[VariableId]> {
    window
        .selection_targets
        .iter()
        .copied()
        .filter(|slot| admg.dense_of_static(*slot).is_some())
        .collect()
}

fn dag_member(
    dag: &TemporalDag,
    indexer: &TemporalIndexer,
    window: &TemporalWindow,
    identifier: &TransportIdentifier,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<TemporalEnvelopeMember, DiscoveredTransportError> {
    let mut admg = dag_window(dag, indexer, budget, ctx)?;
    let template = template_edges(dag.nodes(), dag.edges());
    connect_exposed(&mut admg, &exposed_nodes(&template, indexer)?)?;
    let admg = ancestral_subgraph(&admg, &[window.treatment, window.outcome])
        .map_err(|e| graph_error(&e))?;
    let verdict =
        verdict_of(admg.clone(), &kept_selection(&admg, window), identifier, &window.query)?;
    let directed = template.iter().filter_map(directed_of).collect();
    Ok(TemporalEnvelopeMember {
        directed_edges: sorted_unique(directed),
        bidirected_edges: Vec::new(),
        confounded_edges: Vec::new(),
        verdict,
        graph: admg,
    })
}

/// A completed temporal MAG as a member, or `None` when it has a selection-bias edge.
fn mag_window_member(
    mag: &TemporalPag,
    indexer: &TemporalIndexer,
    window: &TemporalWindow,
    identifier: &TransportIdentifier,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Option<TemporalEnvelopeMember>, DiscoveredTransportError> {
    let unfolded = mag.unfold_within(indexer.clone(), budget, ctx).map_err(|e| graph_error(&e))?;
    let template = template_edges(mag.nodes(), mag.edges());
    let directed: Vec<LaggedEdge> = template.iter().filter_map(directed_of).collect();
    let bidirected: Vec<LaggedEdge> = template
        .iter()
        .filter(|e| (e.at_a, e.at_b) == (Endpoint::Arrow, Endpoint::Arrow))
        .map(|e| bidirected_lagged(e.a, e.b))
        .collect();
    let window_mag = relabelled(&unfolded.pag)?;
    if !antecedent_graph::completion::is_mag_completion(&window_mag) {
        return Ok(Some(TemporalEnvelopeMember {
            directed_edges: sorted_unique(directed),
            bidirected_edges: sorted_unique(bidirected),
            confounded_edges: Vec::new(),
            verdict: MemberVerdict::NotCertified {
                reason: "transport.temporal_window_not_mag".to_owned(),
            },
            graph: Admg::with_variables(0),
        }));
    }
    let Some(mut member) = mag_member(&window_mag, dense_label)? else {
        return Ok(None);
    };
    connect_exposed(&mut member.admg, &exposed_nodes(&template, indexer)?)?;
    let graph = ancestral_subgraph(&member.admg, &[window.treatment, window.outcome])
        .map_err(|e| graph_error(&e))?;
    let confounded = member
        .confounded
        .iter()
        .map(|(from, to)| window_edge(indexer, *from, *to))
        .collect::<Result<Vec<_>, _>>()?;
    let verdict =
        verdict_of(graph.clone(), &kept_selection(&graph, window), identifier, &window.query)?;
    Ok(Some(TemporalEnvelopeMember {
        directed_edges: sorted_unique(directed),
        bidirected_edges: sorted_unique(bidirected),
        confounded_edges: sorted_unique(confounded),
        verdict,
        graph,
    }))
}
