//! Streamed / bounded PAG completion sampling.
//!
//! Completions are never retained without bound.  Before yielding anything, the
//! sampler walks the endpoint-assignment space depth-first, prunes every subtree whose
//! already-decided edges cannot be ancestral, and verifies that each survivor is
//! ancestral, maximal, agrees with the PAG's unshielded colliders, and belongs to one
//! common m-separation equivalence class.  A PAG whose marks admit multiple equivalence
//! classes is refused rather than sampled from an unproved superset.
//!
//! No count of circle endpoints or nodes bounds the walk.  A [`SearchBudget`]
//! (operations, depth, memory, cancellation) is charged for every assignment visited and
//! every m-separation query of the equivalence audit.  A stop is a typed partial result
//! ([`CompletionSampler::search_stop`]): the completions found so far are locally valid,
//! but no class-wide property may be concluded from them
//! ([`CompletionSampler::class_audit_incomplete`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SearchBudget, SearchLimits, SearchReceipt, SearchStop};

use crate::admg::Admg;
use crate::dsep::DSeparationWorkspace;
use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint};
use crate::workspace::{BitSet, GraphWorkspace};

/// Operations the default completion search may charge: one per endpoint assignment (or
/// partial orientation) visited and one per m-separation query of the equivalence audit.
pub const DEFAULT_COMPLETION_OPERATIONS: usize = 1_000_000;

/// Limits and context of the searches whose constructors take no explicit budget.
pub(crate) fn default_completion_search() -> (SearchLimits, ExecutionContext) {
    (
        SearchLimits { operations: DEFAULT_COMPLETION_OPERATIONS, depth: usize::MAX },
        ExecutionContext::production(0, 1),
    )
}

/// One circle-free completion of a PAG (MAG marks only).
#[derive(Clone, Debug)]
pub struct PagCompletion {
    /// Completed graph (no Circle endpoints).
    pub graph: Pag,
    /// Index of this completion in the stream (0-based).
    pub index: usize,
}

/// Audit counts for a bounded PAG-to-MAG completion enumeration.
///
/// The assignment counters count complete endpoint assignments, including those in
/// subtrees pruned before they were fully assigned, and saturate at `u64::MAX`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompletionValidationReport {
    /// Endpoint assignments classified (examined or pruned with their subtree).
    pub assignments_examined: u64,
    /// Ancestral assignments rejected by caller-supplied structural constraints.
    pub rejected_constraints: u64,
    /// Assignments rejected because they were not ancestral graphs.
    pub rejected_non_ancestral: u64,
    /// Ancestral assignments rejected because they were not maximal.
    pub rejected_nonmaximal: u64,
    /// MAGs rejected because an unshielded collider contradicted the source PAG.
    pub rejected_local_incompatible: u64,
    /// Whether locally compatible MAGs represented more than one global Markov class.
    pub ambiguous_global_class: bool,
    /// Whether the global m-separation class audit was skipped as too expensive.
    ///
    /// The audit is skipped when one member's m-separation signature alone would exceed the
    /// search budget's operations or memory. Local validity (ancestral, maximal,
    /// collider-preserving) still holds for every yielded completion, but they have not been
    /// proved to share one Markov class, so no caller may conclude a class-wide property from
    /// them.
    pub equivalence_audit_skipped: bool,
    /// Number of verified members of the single represented class (before the yield cap).
    /// A lower bound when [`Self::search_stop`] is set.
    pub represented_completions: usize,
    /// The bound that stopped the walk before every assignment was classified, if any. The
    /// counters above then describe a prefix of the assignment space.
    pub search_stop: Option<SearchStop>,
    /// Operations charged against the search budget.
    pub operations_consumed: usize,
}

/// Streams PAG completions with a hard cap (no unbounded retain).
#[derive(Clone, Debug)]
pub struct CompletionSampler {
    completions: Vec<Pag>,
    max_completions: usize,
    next_index: usize,
    n_circle_sites: usize,
    report: CompletionValidationReport,
    receipt: Option<SearchReceipt>,
}

/// One circle endpoint: edge `(a, b)` and whether the circle is at `a`.
type CircleSite = (DenseNodeId, DenseNodeId, bool);

impl CompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MAG completions.
    ///
    /// The walk is bounded by [`DEFAULT_COMPLETION_OPERATIONS`]; use [`Self::with_budget`] to
    /// choose the limits, memory cap and cancellation.
    ///
    /// # Errors
    ///
    /// Projection failure of the source graph.  A graph too large to enumerate or audit
    /// within the budget is not refused: the walk stops with a typed partial result
    /// ([`Self::search_stop`]), or the global audit is skipped and recorded in
    /// [`CompletionValidationReport::equivalence_audit_skipped`].
    #[allow(clippy::needless_pass_by_value, reason = "preserve the established constructor API")]
    pub fn new(pag: Pag, max_completions: usize) -> Result<Self, GraphError> {
        let (limits, ctx) = default_completion_search();
        Self::with_budget(&pag, max_completions, limits, &ctx)
    }

    /// [`Self::new`] under an explicit search budget.
    ///
    /// `limits.operations` bounds the endpoint assignments visited plus the audit's
    /// m-separation queries, `limits.depth` the circle endpoints assigned on one branch, and
    /// `ctx` supplies cancellation and the hard memory limit.
    ///
    /// # Errors
    ///
    /// Projection failure of the source graph.
    pub fn with_budget(
        pag: &Pag,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        Self::with_projection(pag, max_completions, limits, ctx, |g| Some(g.clone()))
    }

    pub(crate) fn with_projection(
        pag: &Pag,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
        project: impl Fn(&Pag) -> Option<Pag>,
    ) -> Result<Self, GraphError> {
        let audit_source = project(pag)
            .ok_or(GraphError::InvalidEndpoints { message: "PAG validation projection failed" })?;
        let mut sites = Vec::new();
        let n = pag.node_count();
        for i in 0..n {
            let a = DenseNodeId::try_from_usize(i)?;
            for (b, at_a, at_b) in pag.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) {
                    sites.push((a, b, true));
                }
                if matches!(at_b, Endpoint::Circle) {
                    sites.push((a, b, false));
                }
            }
        }
        let mut budget = match SearchBudget::new(limits, ctx) {
            Ok(budget) => budget,
            Err(receipt) => {
                let report = CompletionValidationReport {
                    search_stop: Some(receipt.stop),
                    ..Default::default()
                };
                return Ok(Self {
                    completions: Vec::new(),
                    max_completions,
                    next_index: 0,
                    n_circle_sites: sites.len(),
                    report,
                    receipt: Some(receipt),
                });
            }
        };
        // The global m-separation signature costs one d-separation query per node pair per
        // conditioning subset, so it is exponential in node count where the assignment
        // walk above is only exponential in circle sites. Refusing the whole graph on
        // that basis would withdraw a capability from every large PAG. Instead the audit is
        // skipped and recorded when one signature alone cannot fit the budget: local validation
        // still runs, and callers that would otherwise assert a class-wide property must
        // downgrade on this flag.
        let audit_class = sites.is_empty()
            || signature_fits_budget(
                audit_source.node_count(),
                limits.operations,
                budget.memory_limit_bytes(),
            );
        let mut walk = Walk {
            pag,
            sites: &sites,
            project: &project,
            audit_source: &audit_source,
            audit_class,
            max_completions,
            graph_bytes: graph_bytes(pag),
            signature_bytes: signature_bytes(audit_source.node_count()),
            report: CompletionValidationReport {
                equivalence_audit_skipped: !audit_class,
                ..Default::default()
            },
            completions: Vec::new(),
            represented_completions: 0,
            reference_signature: None,
            audit_allowance: limits.operations / 2,
        };
        let stop = walk.run(&mut budget)?;
        let Walk { mut report, mut completions, represented_completions, .. } = walk;
        if report.ambiguous_global_class {
            // A partial graph that admits multiple Markov classes is not a certified PAG for
            // this enumerator.  Keeping either class would invent membership information.
            completions.clear();
        }
        report.represented_completions =
            if report.ambiguous_global_class { 0 } else { represented_completions };
        report.search_stop = stop;
        report.operations_consumed = budget.operations();
        let receipt = stop.map(|stop| {
            let total = format!("2^{}", sites.len());
            budget.receipt(
                stop,
                vec![format!(
                    "{} of {total} endpoint assignment(s) classified, {} represented completion(s)",
                    report.assignments_examined, report.represented_completions
                )],
                vec![format!("the remaining endpoint assignments of {total}")],
            )
        });
        Ok(Self {
            completions,
            max_completions,
            next_index: 0,
            n_circle_sites: sites.len(),
            report,
            receipt,
        })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.max_completions
    }

    /// Number of circle endpoints being oriented.
    #[must_use]
    pub fn n_circle_sites(&self) -> usize {
        self.n_circle_sites
    }

    /// Exhaustive validation report for caller diagnostics and mass-scope reporting.
    #[must_use]
    pub const fn validation_report(&self) -> CompletionValidationReport {
        self.report
    }

    /// The bound that stopped the walk before every endpoint assignment was classified.
    #[must_use]
    pub const fn search_stop(&self) -> Option<SearchStop> {
        self.report.search_stop
    }

    /// The limits in force, the work consumed and the region left unexamined, once a bound
    /// has stopped the walk.
    #[must_use]
    pub const fn search_receipt(&self) -> Option<&SearchReceipt> {
        self.receipt.as_ref()
    }

    /// Whether the yielded completions are unproved as one Markov equivalence class.
    ///
    /// True when the global audit was skipped as too expensive, or when a search bound
    /// stopped the walk before every assignment was classified (the yielded set is then a
    /// prefix of the class, and an ambiguity among the unvisited assignments is unseen).
    /// Callers concluding a property "for every member of the class" must treat this exactly
    /// like [`Self::hit_cap`]: the yielded set is locally valid but not certified to be the
    /// class, so a class-wide claim is unearned.
    #[must_use]
    pub const fn class_audit_incomplete(&self) -> bool {
        self.report.equivalence_audit_skipped || self.report.search_stop.is_some()
    }

    /// Whether more verified class members exist than the retained/yielded prefix, or the
    /// walk stopped before the class was fully enumerated.
    ///
    /// Callers that reason over the *whole* Markov equivalence class — e.g. concluding an
    /// effect is identified in every member because every member they saw was identified —
    /// must consult this. The yielded completions are a deterministic low-assignment prefix
    /// rather than a representative sample.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.report.represented_completions > self.max_completions
            || self.report.search_stop.is_some()
    }
}

/// Bytes one retained PAG completion is estimated to occupy.
fn graph_bytes(pag: &Pag) -> u64 {
    let half_edges: usize =
        (0..pag.node_count()).map(|i| pag.neighbors(DenseNodeId::from_index(i)).count()).sum();
    let entries = pag.node_count().saturating_add(half_edges);
    u64::try_from(entries.saturating_mul(16)).unwrap_or(u64::MAX)
}

/// m-separation queries in one signature: every node pair under every conditioning subset
/// of the other nodes. `None` when the count does not fit `u128`.
fn signature_queries(n: usize) -> Option<u128> {
    let pairs = (n as u128).checked_mul(n.saturating_sub(1) as u128)? / 2;
    let subsets = 1u128.checked_shl(u32::try_from(n.saturating_sub(2)).ok()?)?;
    pairs.checked_mul(subsets)
}

/// Bytes of one stored signature (one bool per query).
fn signature_bytes(n: usize) -> u64 {
    signature_queries(n).and_then(|q| u64::try_from(q).ok()).unwrap_or(u64::MAX)
}

/// Whether two signatures (the reference and the candidate) can be built within the
/// operations and memory the budget allows.
fn signature_fits_budget(n: usize, operations: usize, memory_limit_bytes: u64) -> bool {
    signature_queries(n).is_some_and(|queries| queries <= operations as u128)
        && signature_bytes(n).saturating_mul(2) <= memory_limit_bytes
}

/// The depth-first walk over endpoint assignments and the state it classifies them with.
struct Walk<'a, F: Fn(&Pag) -> Option<Pag>> {
    pag: &'a Pag,
    sites: &'a [CircleSite],
    project: &'a F,
    audit_source: &'a Pag,
    audit_class: bool,
    max_completions: usize,
    graph_bytes: u64,
    signature_bytes: u64,
    report: CompletionValidationReport,
    completions: Vec<Pag>,
    represented_completions: usize,
    reference_signature: Option<Vec<bool>>,
    /// Queries the audit may still charge before it is abandoned for the rest of the walk.
    audit_allowance: usize,
}

/// Complete assignments below a node with `remaining` unassigned sites, saturating.
fn subtree_assignments(remaining: usize) -> u64 {
    u32::try_from(remaining).ok().and_then(|r| 1u64.checked_shl(r)).unwrap_or(u64::MAX)
}

impl<F: Fn(&Pag) -> Option<Pag>> Walk<'_, F> {
    /// Live bytes charged with each operation: the working graph, the retained completions
    /// and, when auditing, the reference and candidate signatures.
    fn live_bytes(&self) -> u64 {
        let retained = u64::try_from(self.completions.len().saturating_add(2)).unwrap_or(u64::MAX);
        let audit = if self.audit_class && !self.sites.is_empty() {
            self.signature_bytes.saturating_mul(2)
        } else {
            0
        };
        retained.saturating_mul(self.graph_bytes).saturating_add(audit)
    }

    /// Visit every endpoint assignment in ascending order of the mask whose bit `i` is the
    /// value of site `i` (Tail before Arrow), so the retained prefix is the low-mask prefix.
    /// The last site is therefore the outermost level of the walk.
    ///
    /// Returns the bound that stopped the walk, if any.
    fn run(&mut self, budget: &mut SearchBudget<'_>) -> Result<Option<SearchStop>, GraphError> {
        let total = self.sites.len();
        let mut graph = self.pag.clone();
        if total == 0 {
            if let Err(stop) = budget.charge(0, self.live_bytes()) {
                return Ok(Some(stop));
            }
            if !ancestral_up_to_open_marks(&graph, false) {
                self.report.rejected_non_ancestral += 1;
                self.report.assignments_examined += 1;
                return Ok(None);
            }
            return Ok(self.classify(&graph, budget).err());
        }
        // Frame `(depth, next_choice)`: `depth` sites of the suffix are assigned.
        let mut frames: Vec<(usize, u8)> = vec![(0, 0)];
        while let Some(frame) = frames.last_mut() {
            let (depth, choice) = *frame;
            let site = self.sites[total - 1 - depth];
            if choice == 2 {
                frames.pop();
                restore_circle(&mut graph, site)?;
                continue;
            }
            frame.1 += 1;
            let endpoint = if choice == 0 { Endpoint::Tail } else { Endpoint::Arrow };
            if let Err(stop) = budget.charge(depth + 1, self.live_bytes()) {
                return Ok(Some(stop));
            }
            let remaining = total - 1 - depth;
            // A graph with open endpoints is only checked on its decided edges; a violation
            // there holds for every assignment of the open ones, so the subtree is pruned.
            if !set_endpoint(&mut graph, site, endpoint)
                || !ancestral_up_to_open_marks(&graph, remaining > 0)
            {
                let pruned = subtree_assignments(remaining);
                self.report.rejected_non_ancestral =
                    self.report.rejected_non_ancestral.saturating_add(pruned);
                self.report.assignments_examined =
                    self.report.assignments_examined.saturating_add(pruned);
            } else if remaining == 0 {
                if let Err(stop) = self.classify(&graph, budget) {
                    return Ok(Some(stop));
                }
            } else {
                frames.push((depth + 1, 0));
            }
        }
        Ok(None)
    }

    /// Classify one fully assigned, ancestral graph.
    fn classify(
        &mut self,
        candidate: &Pag,
        budget: &mut SearchBudget<'_>,
    ) -> Result<(), SearchStop> {
        self.report.assignments_examined = self.report.assignments_examined.saturating_add(1);
        let Some(audited) = (self.project)(candidate) else {
            self.report.rejected_constraints += 1;
            return Ok(());
        };
        if !is_maximal_ancestral_graph(&audited) {
            self.report.rejected_nonmaximal += 1;
            return Ok(());
        }
        if !preserves_unshielded_colliders(self.audit_source, &audited) {
            self.report.rejected_local_incompatible += 1;
            return Ok(());
        }
        if self.audit_class && !self.sites.is_empty() {
            // The audit must not consume the walk's budget: when the next signature would
            // exceed its half, the audit is abandoned (as if skipped from the start), and the
            // walk continues on local validity alone.
            let cost = signature_queries(audited.node_count())
                .and_then(|q| usize::try_from(q).ok())
                .unwrap_or(usize::MAX);
            if cost > self.audit_allowance {
                self.audit_class = false;
                self.report.equivalence_audit_skipped = true;
                self.report.ambiguous_global_class = false;
                self.reference_signature = None;
            } else {
                self.audit_allowance -= cost;
            }
        }
        if self.audit_class {
            let signature = if self.sites.is_empty() {
                Vec::new()
            } else {
                let depth = self.sites.len();
                let bytes = self.live_bytes();
                m_separation_signature(&audited, || budget.charge(depth, bytes))?
            };
            if let Some(reference) = &self.reference_signature {
                if reference != &signature {
                    self.report.ambiguous_global_class = true;
                    return Ok(());
                }
            } else {
                self.reference_signature = Some(signature);
            }
        }
        self.represented_completions += 1;
        if self.completions.len() < self.max_completions {
            self.completions.push(candidate.clone());
        }
        Ok(())
    }
}

/// Set the circle at `site` to `endpoint`. `false` when the graph rejects the marks.
fn set_endpoint(graph: &mut Pag, (a, b, at_a_circle): CircleSite, endpoint: Endpoint) -> bool {
    let Some(edge) = graph.edge_between(a, b) else {
        return false;
    };
    let marks = if at_a_circle { (endpoint, edge.at_b) } else { (edge.at_a, endpoint) };
    graph.set_marks(a, b, marks.0, marks.1).is_ok()
}

/// Return the endpoint at `site` to a circle, undoing [`set_endpoint`].
fn restore_circle(graph: &mut Pag, site: CircleSite) -> Result<(), GraphError> {
    if set_endpoint(graph, site, Endpoint::Circle) {
        Ok(())
    } else {
        Err(GraphError::InvalidEndpoints {
            message: "PAG completion could not restore a circle endpoint",
        })
    }
}

/// Whether `g` is a maximal ancestral graph completion in the supported directed/bidirected
/// MAG family (selection-variable Tail–Tail edges are outside this sampler's scope).
#[must_use]
pub fn is_mag_completion(g: &Pag) -> bool {
    is_maximal_ancestral_graph(g)
}

fn is_ancestral_orientation(g: &Pag) -> bool {
    ancestral_up_to_open_marks(g, false)
}

/// Ancestral-orientation check of the decided edges of `g`.
///
/// With `allow_open` an edge that still carries a circle or conflict endpoint is skipped, so
/// a `false` result is a violation among edges whose marks are final: no assignment of the
/// open endpoints can repair it. Without it any open endpoint fails the check.
fn ancestral_up_to_open_marks(g: &Pag, allow_open: bool) -> bool {
    let n = g.node_count();
    let mut ws = GraphWorkspace::default();
    for i in 0..n {
        let a = DenseNodeId::try_from_usize(i).expect("node fit");
        for (b, at_a, at_b) in g.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(at_a, Endpoint::Circle | Endpoint::Conflict)
                || matches!(at_b, Endpoint::Circle | Endpoint::Conflict)
            {
                if allow_open {
                    continue;
                }
                return false;
            }
            // Directed MAGs (Zhang) allow → and ↔ only — not undirected —o—.
            if matches!((at_a, at_b), (Endpoint::Tail, Endpoint::Tail)) {
                return false;
            }
            if matches!((at_a, at_b), (Endpoint::Arrow, Endpoint::Arrow)) {
                // Almost-directed cycle: bidirected + directed path either way.
                if g.reaches_directed_with(&mut ws, a, b) || g.reaches_directed_with(&mut ws, b, a)
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Whether a circle-free graph is both ancestral and maximal.
#[must_use]
pub(crate) fn is_maximal_ancestral_graph(g: &Pag) -> bool {
    if !is_ancestral_orientation(g) {
        return false;
    }
    // Richardson–Spirtes Corollary 5.3 gives the separating set for every missing
    // edge in a MAG.  Conversely, a nonmaximal ancestral graph has a nonadjacent
    // pair connected given every set.  Thus this is an exact inducing-path-free test.
    let admg = as_admg(g);
    let n = g.node_count();
    let mut sep_ws = DSeparationWorkspace::default();
    // Ancestor sets once per candidate: `An(x) ∪ An(y)` is then a word-wise OR.
    let ancestors = admg.strict_ancestor_sets();
    let mut union = BitSet::with_len(n);
    for i in 0..n {
        let x = DenseNodeId::from_index(i);
        for j in (i + 1)..n {
            let y = DenseNodeId::from_index(j);
            if g.has_edge(x, y) {
                continue;
            }
            union.clone_from(&ancestors[i]);
            union.union_with(&ancestors[j]);
            union.remove(x);
            union.remove(y);
            let separating = union.to_dense_ids();
            if !admg.is_m_separated(x, y, &separating, &mut sep_ws).expect("known nodes") {
                return false;
            }
        }
    }
    true
}

fn preserves_unshielded_colliders(pag: &Pag, mag: &Pag) -> bool {
    let n = pag.node_count();
    for middle_i in 0..n {
        let middle = DenseNodeId::from_index(middle_i);
        let neighbors: Vec<_> = pag.neighbors(middle).map(|(node, mark, _)| (node, mark)).collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (left, pag_left_mark) = neighbors[i];
                let (right, pag_right_mark) = neighbors[j];
                if pag.has_edge(left, right) {
                    continue;
                }
                let mag_left_mark = mag
                    .neighbors(middle)
                    .find(|(node, _, _)| *node == left)
                    .map(|(_, mark, _)| mark)
                    .expect("same skeleton");
                let mag_right_mark = mag
                    .neighbors(middle)
                    .find(|(node, _, _)| *node == right)
                    .map(|(_, mark, _)| mark)
                    .expect("same skeleton");
                let pag_collider = matches!(pag_left_mark, Endpoint::Arrow)
                    && matches!(pag_right_mark, Endpoint::Arrow);
                let mag_collider = matches!(mag_left_mark, Endpoint::Arrow)
                    && matches!(mag_right_mark, Endpoint::Arrow);
                if pag_collider != mag_collider {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn as_admg(g: &Pag) -> Admg {
    let mut admg = Admg::with_variables(u32::try_from(g.node_count()).expect("node count fits"));
    for i in 0..g.node_count() {
        let a = DenseNodeId::from_index(i);
        for (b, at_a, at_b) in g.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    admg.insert_directed(a, b).expect("validated MAG");
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    admg.insert_directed(b, a).expect("validated MAG");
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    admg.insert_bidirected(a, b).expect("validated MAG");
                }
                _ => unreachable!("validated directed MAG marks"),
            }
        }
    }
    admg
}

/// The m-separation model of `g`: one answer per node pair and conditioning subset.
///
/// `charge` is called before each query; an `Err` abandons the signature.
pub(crate) fn m_separation_signature(
    g: &Pag,
    mut charge: impl FnMut() -> Result<(), SearchStop>,
) -> Result<Vec<bool>, SearchStop> {
    let admg = as_admg(g);
    let n = g.node_count();
    let mut signature = Vec::new();
    let mut ws = DSeparationWorkspace::default();
    for i in 0..n {
        for j in (i + 1)..n {
            let others: Vec<_> = (0..n).filter(|&k| k != i && k != j).collect();
            // Each subset is an iteration of a bitset counter over `others`, not a machine
            // word, so the node count is bounded by the caller's budget alone.
            let mut subset = BitSet::with_len(others.len());
            loop {
                charge()?;
                let z: Vec<_> = subset
                    .iter()
                    .map(|bit| DenseNodeId::from_index(others[bit.as_usize()]))
                    .collect();
                signature.push(
                    admg.is_m_separated(
                        DenseNodeId::from_index(i),
                        DenseNodeId::from_index(j),
                        &z,
                        &mut ws,
                    )
                    .expect("known nodes"),
                );
                if !increment(&mut subset) {
                    break;
                }
            }
        }
    }
    Ok(signature)
}

/// Advance `subset` to the next subset in binary counting order (bit 0 least significant).
/// `false` once every subset has been visited.
fn increment(subset: &mut BitSet) -> bool {
    for bit in 0..subset.bit_len() {
        let id = DenseNodeId::from_index(bit);
        if subset.contains(id) {
            subset.remove(id);
        } else {
            subset.insert(id);
            return true;
        }
    }
    false
}

impl Iterator for CompletionSampler {
    type Item = PagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let graph = self.completions.get(self.next_index)?.clone();
        let index = self.next_index;
        self.next_index += 1;
        Some(PagCompletion { graph, index })
    }
}

/// Compare the m-separation models of finite directed/bidirected MAGs.
///
/// Returns `None` when the audit exceeds the default search budget
/// ([`DEFAULT_COMPLETION_OPERATIONS`] m-separation queries).
/// This certifies only the supplied finite graphs, not an infinite temporal model.
#[must_use]
pub fn audit_finite_mag_equivalence(graphs: &[Pag]) -> Option<bool> {
    let reference = graphs.first()?;
    let n = reference.node_count();
    if graphs.iter().any(|g| g.node_count() != n || !is_mag_completion(g)) {
        return Some(false);
    }
    if graphs.len() == 1 {
        return Some(true);
    }
    let (limits, ctx) = default_completion_search();
    let mut budget = SearchBudget::new(limits, &ctx).ok()?;
    let bytes = signature_bytes(n).saturating_mul(2);
    let mut signature_of = |g: &Pag| m_separation_signature(g, || budget.charge(n, bytes)).ok();
    let signature = signature_of(reference)?;
    for g in &graphs[1..] {
        if signature_of(g)? != signature {
            return Some(false);
        }
    }
    Some(true)
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "test graphs have a handful of nodes, so node indices fit u32"
)]
mod tests {
    use super::*;
    use crate::pag::Pag;

    /// Reference maximality test by per-pair reachability searches (the formulation the
    /// ancestor-bitset version replaces): separating set `An({x,y}) \ {x,y}`.
    fn maximal_by_reachability(g: &Pag) -> bool {
        if !is_ancestral_orientation(g) {
            return false;
        }
        let admg = as_admg(g);
        let n = g.node_count();
        let mut graph_ws = GraphWorkspace::default();
        let mut sep_ws = DSeparationWorkspace::default();
        for i in 0..n {
            let x = DenseNodeId::from_raw(i as u32);
            for j in (i + 1)..n {
                let y = DenseNodeId::from_raw(j as u32);
                if g.has_edge(x, y) {
                    continue;
                }
                let separating: Vec<_> = (0..n)
                    .map(|k| DenseNodeId::from_raw(k as u32))
                    .filter(|&node| {
                        node != x
                            && node != y
                            && (g.reaches_directed_with(&mut graph_ws, node, x)
                                || g.reaches_directed_with(&mut graph_ws, node, y))
                    })
                    .collect();
                if !admg.is_m_separated(x, y, &separating, &mut sep_ws).unwrap() {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn ancestor_bitsets_match_reachability_and_maximality_on_random_mags() {
        let mut state = 0xD1B5_4A32_D192_ED03_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let (mut maximal, mut non_maximal) = (0, 0);
        for _ in 0..400 {
            let n = 3 + (next() % 5) as usize;
            let mut g = Pag::with_variables(n as u32);
            for a in 0..n {
                for b in (a + 1)..n {
                    let (a_id, b_id) =
                        (DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32));
                    let _ = match next() % 5 {
                        0 => g.insert_directed(a_id, b_id),
                        1 => g.insert_directed(b_id, a_id),
                        2 => g.insert_bidirected(a_id, b_id),
                        _ => Ok(()),
                    };
                }
            }
            // Ancestor sets against Admg reachability (as_admg needs an ancestral graph).
            if is_ancestral_orientation(&g) {
                let admg = as_admg(&g);
                let ancestors = admg.strict_ancestor_sets();
                for v in 0..n {
                    for u in 0..n {
                        let (uid, vid) =
                            (DenseNodeId::from_raw(u as u32), DenseNodeId::from_raw(v as u32));
                        assert_eq!(ancestors[v].contains(uid), u != v && admg.reaches(uid, vid));
                    }
                }
            }
            let got = is_maximal_ancestral_graph(&g);
            assert_eq!(got, maximal_by_reachability(&g));
            if got {
                maximal += 1;
            } else {
                non_maximal += 1;
            }
        }
        // The comparison must exercise both outcomes.
        assert!(maximal > 5 && non_maximal > 5, "maximal {maximal}, non-maximal {non_maximal}");
    }

    #[test]
    fn respects_max_completions_bound() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let sampler = CompletionSampler::new(pag, 2).unwrap();
        assert_eq!(sampler.n_circle_sites(), 2);
        let collected: Vec<_> = sampler.collect();
        // o-o has exactly three MAG completions (->, <-, <->; tail-tail is outside the
        // supported family), so the cap of 2 binds and the two lowest masks are yielded.
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].index, 0);
        assert_eq!(collected[1].index, 1);
        for c in &collected {
            assert!(is_mag_completion(&c.graph));
            let e =
                c.graph.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            assert!(!matches!(e.at_a, Endpoint::Circle));
            assert!(!matches!(e.at_b, Endpoint::Circle));
            // No undirected Tail–Tail in directed MAG completions.
            assert!(!matches!((e.at_a, e.at_b), (Endpoint::Tail, Endpoint::Tail)));
        }
    }

    #[test]
    fn no_circle_yields_single_completion() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CompletionSampler::new(pag, 10).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(is_mag_completion(&collected[0].graph));
    }

    #[test]
    fn rejects_almost_directed_cycle() {
        // a → b → c with a ↔ c: directed path a ⇝ c plus bidirected a ↔ c.
        let mut g = Pag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_bidirected(a, c).unwrap();
        assert!(!is_mag_completion(&g));
    }

    #[test]
    fn accepts_bidirected_without_directed_path() {
        let mut g = Pag::with_variables(2);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(is_mag_completion(&g));
    }

    #[test]
    fn rejects_nonmaximal_ancestral_orientation() {
        // X ↔ A ↔ B ↔ Y, A → Y, B → X has an inducing path X,A,B,Y:
        // A and B are colliders and ancestors of Y and X respectively.  X and Y are
        // nonadjacent, so this ancestral graph is not maximal (Richardson–Spirtes §3.7).
        let mut graph = Pag::with_variables(4);
        let endpoint_x = DenseNodeId::from_raw(0);
        let inner_a = DenseNodeId::from_raw(1);
        let inner_b = DenseNodeId::from_raw(2);
        let endpoint_y = DenseNodeId::from_raw(3);
        graph.insert_bidirected(endpoint_x, inner_a).unwrap();
        graph.insert_bidirected(inner_a, inner_b).unwrap();
        graph.insert_bidirected(inner_b, endpoint_y).unwrap();
        graph.insert_directed(inner_a, endpoint_y).unwrap();
        graph.insert_directed(inner_b, endpoint_x).unwrap();
        assert!(is_ancestral_orientation(&graph), "graph is ancestral but not maximal");
        assert!(!is_maximal_ancestral_graph(&graph));

        let sampler = CompletionSampler::new(graph, 8).unwrap();
        assert_eq!(sampler.validation_report().rejected_nonmaximal, 1);
        assert_eq!(sampler.count(), 0, "nonmaximal orientation must never be yielded");
    }

    #[test]
    fn rejects_unshielded_collider_not_encoded_by_pag() {
        // In a PAG, unshielded collider status is invariant.  X o-o M o-o Y does not
        // encode X *-> M <-* Y, so endpoint assignments that invent that collider must
        // be rejected even though they are otherwise maximal ancestral graphs.
        let mut pag = Pag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        pag.insert_circle_circle(x, m).unwrap();
        pag.insert_circle_circle(m, y).unwrap();
        let sampler = CompletionSampler::new(pag, 32).unwrap();
        assert!(sampler.validation_report().rejected_local_incompatible > 0);
        for completion in sampler {
            let xm = completion.graph.edge_between(x, m).unwrap();
            let my = completion.graph.edge_between(m, y).unwrap();
            let at_m_from_x = if xm.a == m { xm.at_a } else { xm.at_b };
            let at_m_from_y = if my.a == m { my.at_a } else { my.at_b };
            assert!(!(at_m_from_x == Endpoint::Arrow && at_m_from_y == Endpoint::Arrow));
        }
    }

    #[test]
    fn fails_closed_when_marks_admit_multiple_global_classes() {
        // <X,Q,B,Y> is a discriminating path for B: Q is a collider on the path,
        // Q -> Y, and X is not adjacent to Y.  Leaving the B-Y endpoint unresolved
        // admits both Q <-> B -> Y (noncollider) and Q <-> B <-> Y (collider).
        // They agree on all unshielded colliders but are not Markov equivalent.
        let mut pag = Pag::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let q = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        pag.insert_directed(x, q).unwrap();
        pag.insert_bidirected(q, b).unwrap();
        pag.insert_directed(q, y).unwrap();
        pag.insert_circle_circle(b, y).unwrap();

        let sampler = CompletionSampler::new(pag, 8).unwrap();
        assert!(sampler.validation_report().ambiguous_global_class);
        assert_eq!(sampler.count(), 0, "an underoriented PAG must fail closed");
    }

    fn search(operations: usize) -> (SearchLimits, ExecutionContext) {
        (SearchLimits { operations, depth: usize::MAX }, ExecutionContext::for_tests(0))
    }

    #[test]
    fn skips_rather_than_refuses_equivalence_audits_above_work_bound() {
        // Refusing here would withdraw completion enumeration from every graph past the
        // audit's exponential budget. The completions are still locally validated; what is
        // lost is the proof that they form a single Markov class, and that loss is recorded
        // so a caller cannot assert a class-wide property from them. One signature of a
        // 16-node graph is 120 pairs x 2^14 subsets = 1_966_080 queries, beyond the default
        // budget.
        let mut pag = Pag::with_variables(16);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let sampler = CompletionSampler::new(pag, 8).unwrap();
        assert!(sampler.class_audit_incomplete());
        assert!(sampler.validation_report().equivalence_audit_skipped);
        assert!(!sampler.validation_report().ambiguous_global_class);
        assert_eq!(sampler.search_stop(), None, "a skipped audit is not a stopped walk");
        assert!(sampler.count() > 0, "locally valid completions are still yielded");
    }

    #[test]
    fn the_audit_is_skipped_when_one_signature_exceeds_the_operation_budget() {
        // Six nodes: 15 pairs x 2^4 subsets = 240 queries per signature, over a budget of 30.
        let mut pag = Pag::with_variables(6);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let (limits, ctx) = search(30);
        let sampler = CompletionSampler::with_budget(&pag, 8, limits, &ctx).unwrap();
        assert!(sampler.validation_report().equivalence_audit_skipped);
        assert!(sampler.class_audit_incomplete());
        assert_eq!(sampler.search_stop(), None);
        assert_eq!(sampler.count(), 3, "o-o has three MAG completions");
    }

    /// Number of ancestral complete graphs on `n` nodes by an independent computation: every
    /// pair is `i -> j`, `j -> i` or `i <-> j`; the directed part must be acyclic and no
    /// bidirected pair may be ordered by a directed path (transitive closure by Floyd-Warshall).
    fn ancestral_complete_graphs(n: usize) -> usize {
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
        let mut count = 0;
        for code in 0..3usize.pow(pairs.len() as u32) {
            let kinds: Vec<usize> =
                (0..pairs.len()).map(|p| (code / 3usize.pow(p as u32)) % 3).collect();
            let mut reach = vec![vec![false; n]; n];
            for (&(i, j), &kind) in pairs.iter().zip(&kinds) {
                match kind {
                    0 => reach[i][j] = true,
                    1 => reach[j][i] = true,
                    _ => {}
                }
            }
            for m in 0..n {
                for a in 0..n {
                    for b in 0..n {
                        if reach[a][m] && reach[m][b] {
                            reach[a][b] = true;
                        }
                    }
                }
            }
            let acyclic = (0..n).all(|v| !reach[v][v]);
            let unordered = pairs
                .iter()
                .zip(&kinds)
                .all(|(&(i, j), &kind)| kind != 2 || (!reach[i][j] && !reach[j][i]));
            if acyclic && unordered {
                count += 1;
            }
        }
        count
    }

    fn all_circle_complete(n: u32) -> Pag {
        let mut pag = Pag::with_variables(n);
        for a in 0..n {
            for b in (a + 1)..n {
                pag.insert_circle_circle(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                    .unwrap();
            }
        }
        pag
    }

    #[test]
    fn twenty_circle_endpoints_complete_and_audit_beyond_the_old_sixteen_site_ceiling() {
        // K5 with every edge o-o: ten edges, twenty circle endpoints. Every member is a
        // complete ancestral graph, all of which share the empty m-separation model, so the
        // single class has exactly the independently counted number of members.
        assert_eq!(ancestral_complete_graphs(4), 219);
        let expected = ancestral_complete_graphs(5);
        assert_eq!(expected, 4231);
        let pag = all_circle_complete(5);
        let (limits, ctx) = search(10_000_000);
        let mut sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.n_circle_sites(), 20);
        let report = sampler.validation_report();
        assert_eq!(report.search_stop, None);
        assert!(!sampler.class_audit_incomplete());
        assert!(!report.ambiguous_global_class);
        assert_eq!(report.assignments_examined, 1 << 20, "every assignment is accounted for");
        assert_eq!(report.represented_completions, expected);
        assert!(!sampler.hit_cap());
        let members: Vec<_> = sampler.by_ref().collect();
        assert_eq!(members.len(), expected);
        for member in &members {
            assert!(is_mag_completion(&member.graph));
        }
    }

    #[test]
    fn a_path_of_nine_circle_edges_has_the_transfer_matrix_count() {
        // Path 0 o-o 1 o-o ... o-o 9: eighteen circle endpoints. The PAG carries no collider,
        // so a member is a sequence of nine edges in {->, <-, <->} with no node receiving
        // an arrowhead from both of its edges. Counting such sequences by transfer matrix:
        // after `->` or `<->` only `->` may follow; after `<-` anything may.
        let (mut ends_in_right, mut ends_in_left, mut ends_in_both) = (1usize, 1usize, 1usize);
        for _ in 1..9 {
            let next_right = ends_in_right + ends_in_left + ends_in_both;
            let next_left = ends_in_left;
            let next_both = ends_in_left;
            ends_in_right = next_right;
            ends_in_left = next_left;
            ends_in_both = next_both;
        }
        let expected = ends_in_right + ends_in_left + ends_in_both;
        assert_eq!(expected, 19);
        let mut pag = Pag::with_variables(10);
        for i in 0..9 {
            pag.insert_circle_circle(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1))
                .unwrap();
        }
        let (limits, ctx) = search(10_000_000);
        let sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.n_circle_sites(), 18);
        assert_eq!(sampler.search_stop(), None);
        assert!(!sampler.class_audit_incomplete());
        assert_eq!(sampler.validation_report().represented_completions, expected);
        assert_eq!(sampler.count(), expected);
    }

    #[test]
    fn the_subset_counter_visits_every_subset_once_in_binary_order() {
        for bits in [0usize, 1, 5, 70] {
            let mut subset = BitSet::with_len(bits);
            let mut visited = 1u64;
            while increment(&mut subset) {
                visited += 1;
                if visited > 4096 {
                    break; // 2^70 subsets: the counter only has to keep advancing
                }
            }
            if bits < 12 {
                assert_eq!(visited, 1 << bits);
            } else {
                assert!(visited > 4096);
            }
        }
        // Binary order, bit 0 least significant.
        let mut subset = BitSet::with_len(3);
        let mut order = Vec::new();
        loop {
            order.push(subset.iter().map(|b| 1usize << b.as_usize()).sum::<usize>());
            if !increment(&mut subset) {
                break;
            }
        }
        assert_eq!(order, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn signature_cost_is_pairs_times_subsets_without_a_word_limit() {
        assert_eq!(signature_queries(2), Some(1));
        assert_eq!(signature_queries(5), Some(10 * 8));
        assert_eq!(signature_queries(16), Some(120 * 16_384));
        // 70 nodes: the subset count does not fit a machine word, and the audit is skipped.
        assert!(signature_queries(70).is_some());
        assert!(!signature_fits_budget(70, usize::MAX, u64::MAX));
        assert_eq!(signature_queries(200), None);
    }

    #[test]
    fn an_exhausted_operation_budget_is_a_typed_partial_result() {
        let pag = all_circle_complete(5);
        let (limits, ctx) = search(2_000);
        let mut sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.search_stop(), Some(SearchStop::Operations));
        assert!(sampler.class_audit_incomplete(), "a prefix is not the class");
        assert!(sampler.hit_cap());
        let report = sampler.validation_report();
        assert!(report.assignments_examined < 1 << 20);
        assert_eq!(report.operations_consumed, 2_000);
        let receipt = sampler.search_receipt().unwrap();
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.operations_consumed, Some(2_000));
        assert!(!receipt.unevaluated.is_empty());
        let members: Vec<_> = sampler.by_ref().collect();
        assert!(members.len() < 4231);
        for member in &members {
            assert!(is_mag_completion(&member.graph), "yielded members stay locally valid");
        }
    }

    #[test]
    fn a_completed_search_has_no_stop_and_no_receipt() {
        let (limits, ctx) = search(1_000_000);
        let sampler =
            CompletionSampler::with_budget(&all_circle_complete(3), 64, limits, &ctx).unwrap();
        assert_eq!(sampler.search_stop(), None);
        assert!(sampler.search_receipt().is_none());
        assert!(!sampler.class_audit_incomplete());
        assert_eq!(sampler.validation_report().represented_completions, 19);
    }

    #[test]
    fn cancellation_and_a_memory_limit_stop_the_walk_with_a_typed_partial_result() {
        let pag = all_circle_complete(5);

        let (limits, ctx) = search(10_000_000);
        ctx.cancellation.cancel();
        let sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.search_stop(), Some(SearchStop::Cancelled));
        assert!(sampler.class_audit_incomplete());
        assert_eq!(sampler.count(), 0);

        let (limits, mut ctx) = search(10_000_000);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(300);
        let sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.search_stop(), Some(SearchStop::Cancelled));
        assert!(sampler.class_audit_incomplete());
        assert!(sampler.validation_report().operations_consumed > 0);

        let (limits, mut ctx) = search(10_000_000);
        ctx.memory.hard_limit_bytes = Some(1);
        let sampler = CompletionSampler::with_budget(&pag, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.search_stop(), Some(SearchStop::Memory));
        assert!(sampler.class_audit_incomplete());
    }

    /// The exhaustive mask enumeration the depth-first walk replaced: every endpoint
    /// assignment, in ascending mask order, through the same local checks and the same
    /// first-signature audit.
    fn exhaustive_completions(pag: &Pag) -> Vec<Pag> {
        let mut sites = Vec::new();
        for i in 0..pag.node_count() {
            let a = DenseNodeId::from_index(i);
            for (b, at_a, at_b) in pag.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) {
                    sites.push((a, b, true));
                }
                if matches!(at_b, Endpoint::Circle) {
                    sites.push((a, b, false));
                }
            }
        }
        let mut out = Vec::new();
        let mut reference: Option<Vec<bool>> = None;
        for mask in 0..(1u64 << sites.len()) {
            let mut graph = pag.clone();
            let mut ok = true;
            for (i, &(a, b, at_a_circle)) in sites.iter().enumerate() {
                let endpoint = if (mask >> i) & 1 == 1 { Endpoint::Arrow } else { Endpoint::Tail };
                let edge = graph.edge_between(a, b).unwrap();
                let marks = if at_a_circle { (endpoint, edge.at_b) } else { (edge.at_a, endpoint) };
                if graph.set_marks(a, b, marks.0, marks.1).is_err() {
                    ok = false;
                    break;
                }
            }
            if !ok
                || !is_ancestral_orientation(&graph)
                || !is_maximal_ancestral_graph(&graph)
                || !preserves_unshielded_colliders(pag, &graph)
            {
                continue;
            }
            if !sites.is_empty() {
                let signature = m_separation_signature(&graph, || Ok(())).unwrap();
                match &reference {
                    Some(r) if r != &signature => continue,
                    Some(_) => {}
                    None => reference = Some(signature),
                }
            }
            out.push(graph);
        }
        out
    }

    fn edge_list(g: &Pag) -> Vec<(u32, u32, Endpoint, Endpoint)> {
        let mut edges = Vec::new();
        for i in 0..g.node_count() {
            let a = DenseNodeId::from_index(i);
            for (b, at_a, at_b) in g.neighbors(a) {
                if a.raw() < b.raw() {
                    edges.push((a.raw(), b.raw(), at_a, at_b));
                }
            }
        }
        edges.sort_by_key(|e| (e.0, e.1));
        edges
    }

    #[test]
    fn the_walk_yields_exactly_the_exhaustive_enumeration_on_random_pags() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(101);
        let mut nonempty = 0;
        for _ in 0..150 {
            let n = 2 + u32::try_from(rng.next_u64() % 4).unwrap_or(0);
            let pag = random_pag_with_circles(&mut rng, n);
            let sampler = CompletionSampler::new(pag.clone(), usize::MAX).unwrap();
            if sampler.n_circle_sites() > 10 {
                continue;
            }
            let ambiguous = sampler.validation_report().ambiguous_global_class;
            let walked: Vec<_> = sampler.map(|c| edge_list(&c.graph)).collect();
            let expected: Vec<_> = exhaustive_completions(&pag).iter().map(edge_list).collect();
            if ambiguous {
                continue;
            }
            assert_eq!(walked, expected, "walk and exhaustive enumeration disagree on {pag:?}");
            nonempty += usize::from(!walked.is_empty());
        }
        assert!(nonempty > 20, "only {nonempty} graphs had completions");
    }

    fn random_pag_with_circles(rng: &mut antecedent_core::CausalRng, n: u32) -> Pag {
        let mut pag = Pag::with_variables(n);
        // Prefer a topological skeleton so directed inserts stay acyclic.
        let mut order: Vec<u32> = (0..n).collect();
        for i in (1..usize::try_from(n).unwrap_or(0)).rev() {
            let bound = u64::try_from(i + 1).unwrap_or(1);
            let j = usize::try_from(rng.next_u64() % bound).unwrap_or(0);
            order.swap(i, j);
        }
        let n_usize = usize::try_from(n).unwrap_or(0);
        for i in 0..n_usize {
            for j in (i + 1)..n_usize {
                if rng.next_u64() % 3 != 0 {
                    continue;
                }
                let a = DenseNodeId::from_raw(order[i]);
                let b = DenseNodeId::from_raw(order[j]);
                let kind = rng.next_u64() % 4;
                let _ = match kind {
                    0 => pag.insert_directed(a, b),
                    1 => pag.insert_circle_arrow(a, b),
                    2 => pag.insert_circle_circle(a, b),
                    _ => pag.insert_bidirected(a, b),
                };
            }
        }
        pag
    }

    /// Completions respect the yield cap and never retain circle marks.
    #[test]
    fn property_completions_respect_bound_and_no_circles() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(23);
        for _ in 0..40 {
            let n = 2 + u32::try_from(rng.next_u64() % 3).unwrap_or(0); // 2..=4
            let pag = random_pag_with_circles(&mut rng, n);
            let max_c = 1 + usize::try_from(rng.next_u64() % 4).unwrap_or(0); // 1..=4
            let Ok(sampler) = CompletionSampler::new(pag, max_c) else {
                continue; // too many circle sites
            };
            let collected: Vec<_> = sampler.collect();
            assert!(collected.len() <= max_c, "exceeded max_completions");
            for (i, c) in collected.iter().enumerate() {
                assert_eq!(c.index, i);
                assert!(is_mag_completion(&c.graph));
                for i in 0..c.graph.node_count() {
                    let a = DenseNodeId::from_index(i);
                    for (b, at_a, at_b) in c.graph.neighbors(a) {
                        if b.raw() < a.raw() {
                            continue;
                        }
                        assert!(!matches!(at_a, Endpoint::Circle | Endpoint::Conflict));
                        assert!(!matches!(at_b, Endpoint::Circle | Endpoint::Conflict));
                    }
                }
            }
        }
    }

    /// Where cheap: an active definite-status path in the PAG remains m-connecting in
    /// every MAG completion (sound direction only; PAG separation is incomplete).
    #[test]
    fn property_definite_msep_stable_across_completions() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(29);
        for _ in 0..30 {
            let n = 3u32;
            let pag = random_pag_with_circles(&mut rng, n);
            let Ok(sampler) = CompletionSampler::new(pag.clone(), 8) else {
                continue;
            };
            if sampler.n_circle_sites() > 4 {
                continue; // keep enumeration cheap
            }
            let completions: Vec<_> = sampler.collect();
            if completions.is_empty() {
                continue;
            }
            for x in 0..n {
                for y in 0..n {
                    if x == y {
                        continue;
                    }
                    let xi = DenseNodeId::from_raw(x);
                    let yi = DenseNodeId::from_raw(y);
                    // Empty Z only — cheapest definite-status check.
                    let Ok(pag_sep) = pag.is_m_separated(xi, yi, &[], 32, 6) else {
                        continue; // budget exhaustion — skip
                    };
                    if pag_sep {
                        continue; // incomplete: PAG sep ⇏ completion sep
                    }
                    for c in &completions {
                        let Ok(comp_sep) = c.graph.is_m_separated(xi, yi, &[], 32, 6) else {
                            continue;
                        };
                        assert!(
                            !comp_sep,
                            "PAG m-connected but completion {} separated {}–{}",
                            c.index, x, y
                        );
                    }
                }
            }
        }
    }
}
