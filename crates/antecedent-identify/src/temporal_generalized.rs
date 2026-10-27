//! `TemporalCpdag` / `TemporalPag` identification envelopes.
//!
//! CPDAG completions use temporal DAG adjustment. PAG completions retain mixed
//! endpoints and use visibility-aware MAG adjustment on a certified stationary
//! unfolding. Completions that do not identify contribute
//! unidentified mass. Multi-step Sustained is identified per completion; MAG
//! completions with bidirected edges remain unevaluable for sequential
//! estimation. Dynamic schedules stay refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::sync::Arc;

use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
use antecedent_data::TemporalIndexer;
use antecedent_graph::{
    TemporalCpdag, TemporalCpdagCompletionSampler, TemporalDag, TemporalPag,
    TemporalPagCompletionSampler,
};

use crate::envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
use crate::error::IdentificationError;
use crate::generalized::GeneralizedAdjustmentIdentifier;
use crate::result::IdentificationStatus;
use crate::temporal_backdoor::TemporalBackdoorIdentifier;

/// Graph class and endpoint semantics of a temporal completion.
#[derive(Clone, Debug)]
pub enum TemporalCompletionGraph {
    /// A causally sufficient temporal DAG completion.
    Dag(TemporalDag),
    /// A directed/bidirected temporal MAG completion.
    Mag(TemporalPag),
}

impl TemporalCompletionGraph {
    /// Stable fingerprint of this completion's marked edges: the low eight bytes of
    /// [`Self::digest`], little-endian.
    ///
    /// Keys a caller-supplied class prior. Enumeration order is not a key.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        let digest = self.digest();
        let mut low = [0u8; 8];
        low.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(low)
    }

    /// 32-byte BLAKE3 digest of this completion's marked edges, independent of dense
    /// insertion order and of Rust's unspecified `DefaultHasher`.
    ///
    /// The stream is domain-separated (`antecedent.identify.temporal_completion_graph.v1`)
    /// and encodes the semantic coordinates `(variable, lag)` of every vertex, so isolated
    /// nodes count, and every edge with its endpoint and middle marks, endpoints ordered
    /// canonically.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let (kind, nodes, marked): (u64, _, Vec<_>) = match self {
            Self::Dag(graph) => (0, graph.nodes(), graph.edges().collect()),
            Self::Mag(graph) => (1, graph.nodes(), graph.edges()),
        };
        let mut vertices: Vec<_> = nodes.iter().map(coordinate).collect();
        vertices.sort_unstable();
        let mut edges: Vec<_> = marked.iter().map(|edge| edge_record(nodes, edge)).collect();
        edges.sort_unstable();
        digest_stream(kind, vertices.len(), edges.len(), vertices, edges)
    }

    /// [`Self::digest`] computed without allocating, for callers that must fund every
    /// allocation before it happens.
    ///
    /// The canonical stream is the same: vertices and edge records are emitted in ascending
    /// order by repeated successor selection over the borrowed graph instead of a sorted copy,
    /// so the cost is quadratic in the vertex and edge counts rather than linearithmic.
    /// `should_stop` is polled once per selected record; when it returns true the traversal
    /// stops and `None` is returned. The result is otherwise byte-identical to
    /// [`Self::digest`].
    #[must_use]
    pub fn digest_without_allocation(&self, should_stop: &dyn Fn() -> bool) -> Option<[u8; 32]> {
        let stopped = Cell::new(false);
        let digest = match self {
            Self::Dag(graph) => {
                let nodes = graph.nodes();
                streamed_digest(0, nodes, || graph.edges(), should_stop, &stopped)
            }
            Self::Mag(graph) => {
                let nodes = graph.nodes();
                streamed_digest(1, nodes, || graph.edges_lazy(), should_stop, &stopped)
            }
        };
        if stopped.get() { None } else { Some(digest) }
    }

    /// [`Self::fingerprint`] computed by [`Self::digest_without_allocation`].
    #[must_use]
    pub fn fingerprint_without_allocation(&self, should_stop: &dyn Fn() -> bool) -> Option<u64> {
        let digest = self.digest_without_allocation(should_stop)?;
        let mut low = [0u8; 8];
        low.copy_from_slice(&digest[..8]);
        Some(u64::from_le_bytes(low))
    }

    /// Whether this MAG completion has a bidirected edge (no DAG sequential factorization).
    #[must_use]
    pub fn has_bidirected(&self) -> bool {
        match self {
            Self::Dag(_) => false,
            Self::Mag(graph) => graph.edges().iter().any(|edge| {
                matches!(
                    (edge.at_a, edge.at_b),
                    (antecedent_graph::Endpoint::Arrow, antecedent_graph::Endpoint::Arrow)
                )
            }),
        }
    }

    /// Directed completion that admits sequential g-computation.
    ///
    /// Bidirected MAG members are unevaluable. A fully directed MAG member may
    /// be used as a DAG without collapsing the source class.
    #[must_use]
    pub fn sequential_dag(&self) -> Option<TemporalDag> {
        match self {
            Self::Dag(graph) => Some(graph.clone()),
            Self::Mag(graph) if !self.has_bidirected() => graph.try_into_temporal_dag().ok(),
            Self::Mag(_) => None,
        }
    }
}

type Coordinate = (u64, u64);
type EdgeRecord = (Coordinate, Coordinate, u64, u64, u64);

fn coordinate(node: &antecedent_graph::NodeRef) -> Coordinate {
    match node {
        antecedent_graph::NodeRef::Lagged { variable, lag } => {
            (u64::from(variable.raw()), u64::from(lag.raw()))
        }
        _ => unreachable!("temporal graphs contain lagged nodes"),
    }
}

/// Semantic edge record: endpoints ordered by coordinate, marks and middle reversed with them.
fn edge_record(
    nodes: &[antecedent_graph::NodeRef],
    edge: &antecedent_graph::MarkedEdge,
) -> EdgeRecord {
    let a = coordinate(&nodes[edge.a.raw() as usize]);
    let b = coordinate(&nodes[edge.b.raw() as usize]);
    let (a, b, at_a, at_b, middle) = if a <= b {
        (a, b, edge.at_a, edge.at_b, middle_code(edge.middle))
    } else {
        let middle = match edge.middle {
            antecedent_graph::MiddleMark::Left => antecedent_graph::MiddleMark::Right,
            antecedent_graph::MiddleMark::Right => antecedent_graph::MiddleMark::Left,
            other => other,
        };
        (b, a, edge.at_b, edge.at_a, middle_code(middle))
    };
    (a, b, u64::from(endpoint_code(at_a)), u64::from(endpoint_code(at_b)), u64::from(middle))
}

/// The one canonical byte stream for a completion digest; callers supply sorted records.
fn digest_stream(
    kind: u64,
    vertex_count: usize,
    edge_count: usize,
    vertices: impl IntoIterator<Item = Coordinate>,
    edges: impl IntoIterator<Item = EdgeRecord>,
) -> [u8; 32] {
    let mut stream =
        antecedent_core::CanonicalWriter::new("antecedent.identify.temporal_completion_graph.v1");
    stream.word(kind);
    stream.word(vertex_count as u64);
    for (variable, lag) in vertices {
        stream.word(variable).word(lag);
    }
    stream.word(edge_count as u64);
    for (a, b, at_a, at_b, middle) in edges {
        for word in [a.0, a.1, b.0, b.1, at_a, at_b, middle] {
            stream.word(word);
        }
    }
    stream.finish()
}

/// Ascending successor selection over a re-iterable source: yields exactly the sorted
/// multiset without holding a copy. Each step rescans the source.
struct Ascending<'a, T, S> {
    source: S,
    last: Option<T>,
    run: Option<(T, usize)>,
    should_stop: &'a dyn Fn() -> bool,
    stopped: &'a Cell<bool>,
}
impl<T: Ord + Copy, I: Iterator<Item = T>, S: Fn() -> I> Iterator for Ascending<'_, T, S> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if let Some((value, remaining)) = &mut self.run {
            if *remaining > 0 {
                *remaining -= 1;
                return Some(*value);
            }
        }
        if self.stopped.get() || (self.should_stop)() {
            self.stopped.set(true);
            return None;
        }
        let mut best: Option<(T, usize)> = None;
        for item in (self.source)() {
            if self.last.is_some_and(|last| item <= last) {
                continue;
            }
            match &mut best {
                None => best = Some((item, 1)),
                Some((value, count)) => match item.cmp(value) {
                    std::cmp::Ordering::Less => best = Some((item, 1)),
                    std::cmp::Ordering::Equal => *count += 1,
                    std::cmp::Ordering::Greater => {}
                },
            }
        }
        let (value, count) = best?;
        self.last = Some(value);
        self.run = Some((value, count - 1));
        Some(value)
    }
}

fn streamed_digest<'a, E, I>(
    kind: u64,
    nodes: &'a [antecedent_graph::NodeRef],
    edges: E,
    should_stop: &'a dyn Fn() -> bool,
    stopped: &'a Cell<bool>,
) -> [u8; 32]
where
    E: Fn() -> I + 'a,
    I: Iterator<Item = antecedent_graph::MarkedEdge>,
{
    let edge_count = edges().count();
    let vertices = Ascending {
        source: || nodes.iter().map(coordinate),
        last: None,
        run: None,
        should_stop,
        stopped,
    };
    let records = Ascending {
        source: || edges().map(|edge| edge_record(nodes, &edge)),
        last: None,
        run: None,
        should_stop,
        stopped,
    };
    digest_stream(kind, nodes.len(), edge_count, vertices, records)
}

const fn endpoint_code(endpoint: antecedent_graph::Endpoint) -> u8 {
    match endpoint {
        antecedent_graph::Endpoint::Tail => 0,
        antecedent_graph::Endpoint::Arrow => 1,
        antecedent_graph::Endpoint::Circle => 2,
        antecedent_graph::Endpoint::Conflict => 3,
    }
}

const fn middle_code(middle: antecedent_graph::MiddleMark) -> u8 {
    match middle {
        antecedent_graph::MiddleMark::Unknown => 0,
        antecedent_graph::MiddleMark::Left => 1,
        antecedent_graph::MiddleMark::Right => 2,
        antecedent_graph::MiddleMark::Both => 3,
        antecedent_graph::MiddleMark::Empty => 4,
    }
}

/// Temporal identification cases with aligned coordinate maps.
#[derive(Clone, Debug)]
pub struct TemporalClassEnvelope {
    /// Per-completion identification and mass.
    pub envelope: IdentificationEnvelope<TemporalCompletionGraph>,
    /// Unfold indexers aligned with [`Self::envelope`].`cases`.
    pub indexers: Vec<TemporalIndexer>,
}

fn refuse_unsupported_class_policy(query: &TemporalEffectQuery) -> Result<(), IdentificationError> {
    match query.policy {
        TemporalPolicy::Pulse { .. } | TemporalPolicy::Sustained { .. } => Ok(()),
        _ => Err(IdentificationError::UnsupportedQuery {
            message: "class-aware TemporalCpdag/TemporalPag identification supports Pulse \
                      and Sustained only",
        }),
    }
}

impl GeneralizedAdjustmentIdentifier {
    /// Identify a temporal effect over a `TemporalCpdag` MEC envelope.
    ///
    /// # Errors
    ///
    /// Unsupported policy, conflict marks, or identification/unfold errors.
    pub fn identify_temporal_cpdag_envelope(
        &self,
        cpdag: &TemporalCpdag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        crate::class_envelope_cache::memoized_cpdag(cpdag, query, &self.config, || {
            self.compute_temporal_cpdag_envelope(cpdag, query)
        })
    }

    fn compute_temporal_cpdag_envelope(
        &self,
        cpdag: &TemporalCpdag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        refuse_unsupported_class_policy(query)?;
        let (limits, ctx) = self.config.completion_search();
        let mut sampler = TemporalCpdagCompletionSampler::with_budget(
            cpdag.clone(),
            self.config.max_completions,
            limits,
            &ctx,
        )?;
        let (envelope, indexers) = identify_temporal_completions(&mut sampler, query, self)?;
        crate::generalized::check_completion_stop(sampler.search_stop())?;
        let mut envelope = envelope;
        let n = cpdag.undirected_edge_count();
        if n > 0 {
            envelope.push_features([GraphFeature {
                kind: Arc::from("temporal_cpdag_undirected_marks"),
                detail: Arc::from(format!("{n} undirected edge(s) in source TemporalCpdag")),
            }]);
        }
        if sampler.hit_cap() {
            downgrade_capped(&mut envelope, self.config.max_completions, sampler.search_stop());
        }
        Ok(TemporalClassEnvelope { envelope, indexers })
    }

    /// Identify a temporal effect over a `TemporalPag` completion envelope.
    ///
    /// # Errors
    ///
    /// Unsupported policy, unsupported marks, invalid stationary templates, or identification errors.
    pub fn identify_temporal_pag_envelope(
        &self,
        pag: &TemporalPag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        crate::class_envelope_cache::memoized_pag(pag, query, &self.config, || {
            self.compute_temporal_pag_envelope(pag, query)
        })
    }

    pub(crate) fn compute_temporal_pag_envelope(
        &self,
        pag: &TemporalPag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        refuse_unsupported_class_policy(query)?;
        let indexer = crate::temporal_mag::window(pag, query)?;
        let (limits, ctx) = self.config.completion_search();
        let mut sampler = TemporalPagCompletionSampler::for_window_with_budget(
            pag.clone(),
            self.config.max_completions,
            &indexer,
            limits,
            &ctx,
        )?;
        let mut cases = Vec::new();
        let mut unfolded = Vec::new();
        for completion in sampler.by_ref() {
            let finite = completion.graph.unfold(indexer.clone())?;
            let result = crate::temporal_mag::identify(
                &completion.graph,
                query,
                &finite,
                self.config.max_candidates,
                self.config.max_examinations,
                self.config.context.as_ref(),
            )?;
            unfolded.push(finite.pag);
            cases.push(GraphIdentificationCase {
                graph: TemporalCompletionGraph::Mag(completion.graph),
                result,
                weight: ProbabilityMass(self.config.per_completion_weight),
            });
        }
        crate::generalized::check_completion_stop(sampler.search_stop())?;
        let indexers = vec![indexer; cases.len()];
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        let report = sampler.validation_report();
        let finite_audit = antecedent_graph::completion::audit_finite_mag_equivalence(&unfolded);
        envelope.push_features([GraphFeature { kind:Arc::from("temporal_pag_mixed_completion_audit"),detail:Arc::from(format!("stationary assignments={}, rejected_temporal_constraints={}, represented={}, ambiguous_class={}, window_audit_skipped={}, search_stop={}, finite_query_equivalence={finite_audit:?}",report.assignments_examined,report.rejected_constraints,report.represented_completions,report.ambiguous_global_class,report.equivalence_audit_skipped,sampler.search_stop().map_or("none",antecedent_core::SearchStop::code))) }]);
        if finite_audit == Some(false) {
            return Err(IdentificationError::NotCertified {
                message: "temporal PAG endpoint refinements have different m-separation models in the unfolded query window",
            });
        }
        if sampler.hit_cap() {
            downgrade_capped(&mut envelope, self.config.max_completions, sampler.search_stop());
        }
        if sampler.class_audit_incomplete() || finite_audit.is_none() {
            envelope.truncated_completions = envelope.truncated_completions.max(1);
            envelope.invariant = None;
            envelope.push_features([GraphFeature { kind:Arc::from("temporal_pag_equivalence_audit_capped"),detail:Arc::from("global m-separation audit exceeded its budget; identification covers only retained, query-window-validated completions") }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        Ok(TemporalClassEnvelope { envelope, indexers })
    }
}

fn identify_temporal_completions<I>(
    sampler: &mut I,
    query: &TemporalEffectQuery,
    id: &GeneralizedAdjustmentIdentifier,
) -> Result<
    (IdentificationEnvelope<TemporalCompletionGraph>, Vec<TemporalIndexer>),
    IdentificationError,
>
where
    I: Iterator,
    I::Item: CompletionGraph,
{
    let backdoor = TemporalBackdoorIdentifier::new();
    let w = ProbabilityMass(id.config.per_completion_weight);
    let mut cases = Vec::new();
    let mut indexers = Vec::new();
    for completion in sampler.by_ref() {
        let graph = completion.into_graph();
        let identified = backdoor.identify_temporal(&graph, query)?;
        cases.push(GraphIdentificationCase {
            graph: TemporalCompletionGraph::Dag(graph),
            result: identified.result,
            weight: w,
        });
        indexers.push(identified.indexer);
    }
    // Dense IDs include the history offset. Re-identify shorter unfoldings in
    // the shared window before comparing functionals or publishing an invariant.
    // Keep each arena intact: remapping just adjustment IDs would corrupt the
    // relationship between its expression and its temporal certificate.
    if let Some(history) = indexers.iter().map(TemporalIndexer::history).max() {
        for (case, indexer) in cases.iter_mut().zip(&mut indexers) {
            if indexer.history() != history {
                let TemporalCompletionGraph::Dag(graph) = &case.graph else {
                    unreachable!("this path only enumerates CPDAG completions");
                };
                let identified = backdoor.identify_temporal_with_history(graph, query, history)?;
                case.result = identified.result;
                *indexer = identified.indexer;
            }
        }
    }
    Ok((IdentificationEnvelope::from_cases(cases), indexers))
}

fn downgrade_capped(
    envelope: &mut IdentificationEnvelope<TemporalCompletionGraph>,
    max_completions: usize,
    stop: Option<antecedent_core::SearchStop>,
) {
    envelope.truncated_completions = envelope.truncated_completions.max(1);
    envelope.invariant = None;
    let stopped =
        stop.map_or_else(String::new, |stop| format!(" (search stopped: {})", stop.code()));
    envelope.push_features([GraphFeature {
        kind: Arc::from("completion_enumeration_capped"),
        detail: Arc::from(format!(
            "retained {} temporal completion(s) under max_completions={max_completions}{stopped}; \
             identification is established only over the deterministic retained subset",
            envelope.cases.len(),
        )),
    }]);
    if envelope.status == IdentificationStatus::NonparametricallyIdentified {
        envelope.status = IdentificationStatus::PartiallyIdentified;
    }
}

trait CompletionGraph {
    fn into_graph(self) -> TemporalDag;
}

impl CompletionGraph for antecedent_graph::TemporalCpdagCompletion {
    fn into_graph(self) -> TemporalDag {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{DynamicRuleId, Lag, VariableId};

    #[test]
    fn fingerprints_use_semantic_coordinates_and_include_isolated_nodes() {
        let build = |reverse: bool, lag: u32, isolated: bool| {
            let mut graph = TemporalDag::empty();
            let (source, target) = if reverse {
                let target = graph
                    .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
                    .unwrap();
                let source = graph
                    .add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(lag))
                    .unwrap();
                (source, target)
            } else {
                let source = graph
                    .add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(lag))
                    .unwrap();
                let target = graph
                    .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
                    .unwrap();
                (source, target)
            };
            graph.insert_directed(source, target).unwrap();
            if isolated {
                graph
                    .add_lagged(VariableId::from_raw(2), antecedent_core::Lag::CONTEMPORANEOUS)
                    .unwrap();
            }
            TemporalCompletionGraph::Dag(graph).fingerprint()
        };
        assert_eq!(build(false, 1, false), build(true, 1, false));
        assert_ne!(build(false, 1, false), build(false, 2, false));
        assert_ne!(build(false, 1, false), build(false, 1, true));
    }

    #[test]
    fn digest_golden_vector_is_pinned() {
        let mut graph = TemporalDag::empty();
        let source = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let target = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(source, target).unwrap();
        let completion = TemporalCompletionGraph::Dag(graph);
        let hex = antecedent_core::to_hex(&completion.digest());
        assert_eq!(hex, "87167be0773ad1c53a6da42994b4ec3699caf525b7c80f5003e178c551f587ce");
        // The fingerprint is the digest's low eight bytes, little-endian.
        assert_eq!(completion.fingerprint(), 0xc5d1_3a77_e07b_1687);
    }

    #[test]
    fn allocation_free_digest_is_byte_identical_on_varied_fixtures() {
        use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark, TemporalPag};
        let never = || false;
        let mut cases: Vec<TemporalCompletionGraph> = Vec::new();
        cases.push(TemporalCompletionGraph::Dag(TemporalDag::empty()));
        cases.push(TemporalCompletionGraph::Mag(TemporalPag::empty()));
        for seed in 0..40_u32 {
            let mut dag = TemporalDag::empty();
            let mut ids = Vec::new();
            // Deterministic scrambled insertion order so dense ids disagree with coordinates.
            let count = 3 + seed % 7;
            for index in 0..count {
                let scramble = (index * 5 + seed) % count;
                let variable = VariableId::from_raw((scramble * 3 + seed) % 11);
                let lag = Lag::from_raw((scramble + seed / 3) % 4);
                if let Ok(id) = dag.add_lagged(variable, lag) {
                    ids.push(id);
                }
            }
            let mut mag = TemporalPag::empty();
            let mut mag_ids = Vec::new();
            for node in dag.nodes() {
                if let antecedent_core::NodeRef::Lagged { variable, lag } = node {
                    mag_ids.push(mag.add_lagged(*variable, *lag).unwrap());
                }
            }
            let middles = [
                MiddleMark::Unknown,
                MiddleMark::Left,
                MiddleMark::Right,
                MiddleMark::Both,
                MiddleMark::Empty,
            ];
            let marks = [Endpoint::Tail, Endpoint::Arrow, Endpoint::Circle];
            for (i, &a) in ids.iter().enumerate() {
                for (j, &b) in ids.iter().enumerate().skip(i + 1) {
                    if (i * 7 + j * 3 + seed as usize) % 3 == 0 {
                        let _ = dag.insert_directed(a, b);
                    }
                    if (i * 5 + j * 11 + seed as usize) % 2 == 0 {
                        let edge = MarkedEdge {
                            a: mag_ids[i],
                            b: mag_ids[j],
                            at_a: marks[(i + seed as usize) % 3],
                            at_b: marks[(j + seed as usize / 2) % 3],
                            middle: middles[(i + j + seed as usize) % 5],
                        };
                        let _ = mag.insert_marked(edge);
                    }
                }
            }
            cases.push(TemporalCompletionGraph::Dag(dag));
            cases.push(TemporalCompletionGraph::Mag(mag));
        }
        let mut with_edges = 0;
        for case in &cases {
            let streamed = case.digest_without_allocation(&never).expect("not stopped");
            assert_eq!(streamed, case.digest());
            assert_eq!(case.fingerprint_without_allocation(&never), Some(case.fingerprint()));
            let edge_total = match case {
                TemporalCompletionGraph::Dag(graph) => graph.edges().count(),
                TemporalCompletionGraph::Mag(graph) => {
                    assert_eq!(graph.edges_lazy().count(), graph.edges().len());
                    graph.edges().len()
                }
            };
            with_edges += usize::from(edge_total > 0);
        }
        assert!(with_edges > 20, "fixtures must exercise non-empty edge sets");
        // The pinned golden vector is the same through the allocation-free traversal.
        let mut graph = TemporalDag::empty();
        let source = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let target = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(source, target).unwrap();
        let completion = TemporalCompletionGraph::Dag(graph);
        assert_eq!(
            antecedent_core::to_hex(&completion.digest_without_allocation(&never).unwrap()),
            "87167be0773ad1c53a6da42994b4ec3699caf525b7c80f5003e178c551f587ce"
        );
        assert_eq!(completion.digest_without_allocation(&|| true), None);
    }

    #[test]
    fn completion_agreement_uses_a_shared_temporal_indexer() {
        struct Completion(TemporalDag);
        impl CompletionGraph for Completion {
            fn into_graph(self) -> TemporalDag {
                self.0
            }
        }
        let mut graph = TemporalDag::empty();
        let t = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        let z = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        graph.insert_directed(z, t).unwrap();
        graph.insert_directed(z, y).unwrap();
        graph.insert_directed(t, y).unwrap();
        let mut wider = graph.clone();
        wider.add_lagged(VariableId::from_raw(2), Lag::from_raw(2)).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1));
        let backdoor = TemporalBackdoorIdentifier::new();
        let first = backdoor.identify_temporal(&graph, &query).unwrap();
        let second = backdoor.identify_temporal(&wider, &query).unwrap();
        assert_ne!(first.indexer, second.indexer);
        let (envelope, indexers) = identify_temporal_completions(
            &mut vec![Completion(graph), Completion(wider)].into_iter(),
            &query,
            &GeneralizedAdjustmentIdentifier::new(),
        )
        .unwrap();
        assert_eq!(indexers[0], indexers[1]);
        assert!(envelope.invariant.is_some());
        let z = envelope.invariant.unwrap().adjustment_set[0];
        let key = indexers[0].key_of(z.raw()).unwrap();
        assert_eq!(key.variable, VariableId::from_raw(2));
        assert_eq!(key.offset, -1);
    }

    #[test]
    fn invisible_temporal_pag_retains_directed_and_latent_cases() {
        let mut pag = TemporalPag::empty();
        let t = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = pag.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        pag.insert_circle_arrow(t, y).unwrap();
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::pulse(-1);
        let env = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&pag, &query)
            .unwrap()
            .envelope;
        assert_eq!(env.cases.len(), 2);
        assert_eq!(env.status, IdentificationStatus::NotIdentified);
        assert!(
            env.critical_graph_features
                .iter()
                .any(|f| f.kind.as_ref() == "temporal_pag_mixed_completion_audit")
        );
    }

    #[test]
    fn dynamic_schedule_is_not_silently_estimated_as_a_pulse() {
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), Arc::from([-1, 0]));
        assert!(refuse_unsupported_class_policy(&query).is_err());
    }
}
