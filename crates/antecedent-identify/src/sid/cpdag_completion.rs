//! Bounded DAG completions of a supplied CPDAG, decided as a finite scenario set (2.3A X2).
//!
//! Graph class: a supplied CPDAG of at most [`CPDAG_MAX_NODES`] fully observed
//! nodes. A completion is a DAG with the CPDAG's skeleton and exactly its
//! v-structures (every directed CPDAG edge kept, every undirected edge oriented
//! so that no directed cycle and no new unshielded collider appears). Selection
//! edges and latent (bidirected) edges are refused, never projected away.
//!
//! Enumeration does not identify a mixture and assigns no graph probability: it
//! lists the members of the equivalence class, gives each a canonical identity
//! (a domain-separated BLAKE3 digest of the sorted `parent -> child` list over
//! variable ids, so node and edge insertion order cannot move it), and orders
//! them by identity. Every completion is then one scenario of the existing
//! [`TransportScenarioSet`] and is decided by the same classical catalog route
//! as the 2.2 supplied-scenario row. Evidence is bound per completion
//! ([`CpdagEvidence`]): a catalog bound to one completion's identity never
//! satisfies another's, and a binding that names a different identity is
//! refused (`cpdag_scenarios.evidence_identity_mismatch`).
//!
//! One [`SearchBudget`] bounds enumeration and decisions together. Every
//! orientation attempt and retained completion is one charged operation; depth
//! is the number of undirected edges oriented so far. Storage is charged before
//! keeping each completion. On a stop (limit, memory or cancellation), found
//! completions stay listed, undecided ones are `unevaluated`, and the number
//! never enumerated is a conservative upper bound from the remaining orientation
//! masks. No search continues after the stop to compute an exact remainder.
//! A stopped enumeration is not a complete class certificate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    EvidenceCatalog, ExecutionContext, NodeRef, SearchBudget, SearchLimits, SearchReceipt,
    SearchStop, VariableId, reason_code,
};
use antecedent_graph::{Admg, Cpdag, DenseNodeId, SelectionDiagram};

use super::scenarios::{
    SCENARIO_MAX_COUNT, ScenarioCoordinate, ScenarioDecision, ScenarioDecisionLimits,
    ScenarioOutcome, ScenarioQuestion, ScenarioSetDecision, ScenarioSetRefusal, TransportScenario,
    TransportScenarioSet,
};
use super::{
    CatalogTransportResult, ClassicalTransportQuery, ClassicalTransportResult, IdentificationError,
    SearchCharge, SharedSearch, identify_catalog_transport_metered,
    identify_classical_transport_metered,
};

/// Most nodes of a supplied CPDAG.
pub const CPDAG_MAX_NODES: usize = 6;

/// Selection or latent structure was supplied.
pub const CPDAG_SELECTION_OR_LATENT_DETAIL: &str = "cpdag_scenarios.selection_or_latent";
/// More than [`CPDAG_MAX_NODES`] nodes.
pub const CPDAG_BOUNDS_EXCEEDED_DETAIL: &str = "cpdag_scenarios.bounds_exceeded";
/// The input is not a valid CPDAG.
pub const CPDAG_NOT_A_CPDAG_DETAIL: &str = "cpdag_scenarios.not_a_cpdag";
/// An evidence binding names a completion identity that does not match.
pub const CPDAG_EVIDENCE_MISMATCH_DETAIL: &str = "cpdag_scenarios.evidence_identity_mismatch";
/// Detail of a completion with no evidence bound to it.
pub const CPDAG_NO_EVIDENCE_DETAIL: &str = "cpdag_scenarios.no_evidence_binding";

/// A refusal or failure of the completion route.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CpdagScenarioError {
    /// A typed `cpdag_scenarios.*` or `scenarios.*` refusal.
    #[error(transparent)]
    Refused(#[from] ScenarioSetRefusal),
    /// An invalid query or catalog, or a failure of the decision route.
    #[error(transparent)]
    Identification(#[from] IdentificationError),
}

fn not_a_cpdag(message: impl Into<String>) -> ScenarioSetRefusal {
    ScenarioSetRefusal {
        code: reason_code!("invalid_argument"),
        detail: CPDAG_NOT_A_CPDAG_DETAIL,
        message: message.into(),
    }
}

fn unsupported(detail: &'static str, message: impl Into<String>) -> ScenarioSetRefusal {
    ScenarioSetRefusal {
        code: reason_code!("route_not_supported"),
        detail,
        message: message.into(),
    }
}

fn mismatch(message: impl Into<String>) -> ScenarioSetRefusal {
    ScenarioSetRefusal {
        code: reason_code!("invalid_argument"),
        detail: CPDAG_EVIDENCE_MISMATCH_DETAIL,
        message: message.into(),
    }
}

/// A supplied CPDAG with the structure the initial cell refuses if present.
#[derive(Clone, Debug)]
pub struct CpdagCompletionInput {
    /// The CPDAG: directed and undirected edges over static variables.
    pub cpdag: Cpdag,
    /// Selection targets; must be empty in this cell.
    pub selection_targets: Arc<[VariableId]>,
    /// Latent (bidirected) pairs; must be empty in this cell.
    pub latent_pairs: Arc<[(VariableId, VariableId)]>,
}

impl CpdagCompletionInput {
    /// A fully observed CPDAG with no selection or latent structure.
    #[must_use]
    pub fn new(cpdag: Cpdag) -> Self {
        Self { cpdag, selection_targets: Arc::from([]), latent_pairs: Arc::from([]) }
    }
}

/// One DAG completion with its canonical identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionDag {
    /// Canonical identity: 64 lowercase hex characters of the domain-separated
    /// digest of the sorted edge list. Also the scenario name.
    pub identity: Arc<str>,
    /// Sorted `(parent, child)` edges.
    pub edges: Arc<[(VariableId, VariableId)]>,
}

/// The enumerated completions of one CPDAG, in identity order.
#[derive(Clone, Debug)]
pub struct CpdagEnumeration {
    /// Identity of the supplied CPDAG (variables, directed edges, undirected edges).
    pub cpdag_identity: Arc<str>,
    /// Observed variables, ascending.
    pub variables: Vec<VariableId>,
    /// Completions found, sorted by identity.
    pub completions: Vec<CompletionDag>,
    /// Upper bound on completions the stop left unenumerated; zero when finished.
    /// A stopped search cannot know the exact remainder without more search.
    pub not_enumerated: usize,
    /// Present when a limit, memory bound or cancellation stopped the work.
    pub receipt: Option<SearchReceipt>,
}

impl CpdagEnumeration {
    /// Upper bound on the class size; exact only when `not_enumerated == 0`.
    #[must_use]
    pub fn total(&self) -> usize {
        self.completions.len() + self.not_enumerated
    }

    /// Whether the enumeration is complete and may be exported.
    #[must_use]
    pub const fn is_exportable(&self) -> bool {
        self.receipt.is_none()
    }
}

/// Evidence bound to one completion's identity.
#[derive(Clone, Debug)]
pub struct CompletionEvidenceBinding {
    /// The completion this evidence is supplied for.
    pub completion: Arc<str>,
    /// The completion identity the evidence's graph certificate names. It must
    /// equal [`Self::completion`]: evidence certified for one completion never
    /// satisfies another.
    pub certified_for: Arc<str>,
    /// Identity of the evidence (carried into the receipt).
    pub evidence_identity: Arc<str>,
    /// The catalog of laws and regimes for this completion.
    pub catalog: EvidenceCatalog,
}

/// How evidence reaches the completions.
#[derive(Clone, Debug)]
pub enum CpdagEvidence {
    /// One catalog the caller declares valid for every completion.
    Shared {
        /// Identity of the shared evidence.
        evidence_identity: Arc<str>,
        /// The catalog.
        catalog: EvidenceCatalog,
    },
    /// A catalog per completion identity; a completion with none is
    /// `missing_evidence`.
    PerCompletion(Vec<CompletionEvidenceBinding>),
}

/// A completion as the decision reports it, aligned with the decision's scenarios.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionRecord {
    /// Canonical completion identity.
    pub identity: Arc<str>,
    /// Sorted `(parent, child)` edges.
    pub edges: Arc<[(VariableId, VariableId)]>,
    /// Identity of the evidence bound to this completion, if any.
    pub evidence_identity: Option<Arc<str>>,
}

/// Every completion decided (or left unevaluated) against the same question.
#[derive(Clone, Debug)]
pub struct CpdagCompletionDecision {
    /// Identity of the supplied CPDAG.
    pub cpdag_identity: Arc<str>,
    /// Completions in identity order, aligned with `decision.decisions`.
    pub completions: Vec<CompletionRecord>,
    /// Upper bound on completions left unenumerated (their graphs are unknown).
    pub not_enumerated: usize,
    /// The scenario decision over the completions found; `None` when none was found.
    pub decision: Option<ScenarioSetDecision>,
    /// The receipt of the first stop, enumeration or decision.
    pub receipt: Option<SearchReceipt>,
}

impl CpdagCompletionDecision {
    /// Upper bound on the class size; exact only when `not_enumerated == 0`.
    #[must_use]
    pub fn total(&self) -> usize {
        self.completions.len() + self.not_enumerated
    }

    /// Whether every completion was enumerated and decided.
    #[must_use]
    pub const fn is_exportable(&self) -> bool {
        self.receipt.is_none()
    }
}

/// The CPDAG as index-addressed masks over its ascending variable order.
struct Skeleton {
    variables: Vec<VariableId>,
    fixed: Vec<(usize, usize)>,
    undirected: Vec<(usize, usize)>,
    adjacent: [u8; CPDAG_MAX_NODES],
}

impl Skeleton {
    fn from_input(input: &CpdagCompletionInput) -> Result<Self, ScenarioSetRefusal> {
        let cpdag = &input.cpdag;
        if cpdag.node_count() > CPDAG_MAX_NODES {
            return Err(unsupported(
                CPDAG_BOUNDS_EXCEEDED_DETAIL,
                format!("{} nodes exceed the bound of {CPDAG_MAX_NODES}", cpdag.node_count()),
            ));
        }
        if !input.selection_targets.is_empty() || !input.latent_pairs.is_empty() {
            return Err(unsupported(
                CPDAG_SELECTION_OR_LATENT_DETAIL,
                "selection targets and latent confounding are outside the initial complete-DAG cell",
            ));
        }
        let mut dense = Vec::with_capacity(cpdag.node_count());
        for node in cpdag.nodes() {
            match node {
                NodeRef::Static(variable) => dense.push(*variable),
                _ => return Err(not_a_cpdag("a CPDAG over static variables is required")),
            }
        }
        let mut variables = dense.clone();
        variables.sort_unstable();
        if variables.is_empty() {
            return Err(not_a_cpdag("a CPDAG needs a node"));
        }
        let index_of = |dense_id: DenseNodeId| {
            dense.get(dense_id.as_usize()).and_then(|v| variables.binary_search(v).ok())
        };
        let (mut fixed, mut undirected) = (Vec::new(), Vec::new());
        let mut adjacent = [0_u8; CPDAG_MAX_NODES];
        for edge in cpdag.edges() {
            let (Some(a), Some(b)) = (index_of(edge.a), index_of(edge.b)) else {
                return Err(not_a_cpdag("an edge names an unknown node"));
            };
            adjacent[a] |= 1 << b;
            adjacent[b] |= 1 << a;
            if let Some((from, to)) = edge.parent_child() {
                match (index_of(from), index_of(to)) {
                    (Some(from), Some(to)) => fixed.push((from, to)),
                    _ => return Err(not_a_cpdag("an edge names an unknown node")),
                }
            } else if edge.is_undirected() {
                undirected.push((a.min(b), a.max(b)));
            } else {
                return Err(not_a_cpdag("conflict or non-CPDAG edge marks are not a CPDAG"));
            }
        }
        fixed.sort_unstable();
        undirected.sort_unstable();
        let skeleton = Self { variables, fixed, undirected, adjacent };
        let orienter = Orienter::new(&skeleton);
        if skeleton.fixed.iter().any(|&(a, b)| orienter.reaches(b, a)) {
            return Err(not_a_cpdag("the directed edges contain a cycle"));
        }
        Ok(skeleton)
    }

    /// Per-completion live bytes: its edge list and graph, a fixed estimate.
    fn completion_bytes(&self) -> u64 {
        let edges = self.fixed.len() + self.undirected.len();
        u64::try_from(edges.saturating_mul(16).saturating_add(256)).unwrap_or(u64::MAX)
    }

    fn cpdag_identity(&self) -> Arc<str> {
        let mut hasher = blake3::Hasher::new_derive_key("antecedent.cpdag_completion.cpdag.v1");
        self.hash_variables(&mut hasher);
        for (tag, edges) in [(0_u8, &self.fixed), (1_u8, &self.undirected)] {
            hasher.update(&[tag]);
            hasher.update(&(edges.len() as u64).to_le_bytes());
            for (a, b) in edges {
                hasher.update(&self.variables[*a].raw().to_le_bytes());
                hasher.update(&self.variables[*b].raw().to_le_bytes());
            }
        }
        Arc::from(hasher.finalize().to_hex().to_string())
    }

    fn hash_variables(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&(self.variables.len() as u64).to_le_bytes());
        for variable in &self.variables {
            hasher.update(&variable.raw().to_le_bytes());
        }
    }

    /// The completion an orientation mask names: bit `k` set orients
    /// `undirected[k]` as `.0 -> .1`, clear as `.1 -> .0`.
    fn completion(&self, mask: u32) -> CompletionDag {
        let mut edges = self.fixed.clone();
        for (k, (a, b)) in self.undirected.iter().enumerate() {
            edges.push(if mask & (1 << k) != 0 { (*a, *b) } else { (*b, *a) });
        }
        let mut edges: Vec<(VariableId, VariableId)> =
            edges.into_iter().map(|(p, c)| (self.variables[p], self.variables[c])).collect();
        edges.sort_unstable();
        let mut hasher = blake3::Hasher::new_derive_key("antecedent.cpdag_completion.dag.v1");
        self.hash_variables(&mut hasher);
        hasher.update(&(edges.len() as u64).to_le_bytes());
        for (parent, child) in &edges {
            hasher.update(&parent.raw().to_le_bytes());
            hasher.update(&child.raw().to_le_bytes());
        }
        CompletionDag {
            identity: Arc::from(hasher.finalize().to_hex().to_string()),
            edges: edges.into(),
        }
    }
}

/// Mutable orientation state of the depth-first enumeration.
struct Orienter<'a> {
    skeleton: &'a Skeleton,
    parents: [u8; CPDAG_MAX_NODES],
    children: [u8; CPDAG_MAX_NODES],
    masks: Vec<u32>,
    bytes: u64,
}

/// The charge of one orientation attempt: `(depth, live bytes)`.
type Charge<'c> = dyn FnMut(usize, u64) -> Result<(), SearchStop> + 'c;

impl<'a> Orienter<'a> {
    fn new(skeleton: &'a Skeleton) -> Self {
        let mut state = Self {
            skeleton,
            parents: [0; CPDAG_MAX_NODES],
            children: [0; CPDAG_MAX_NODES],
            masks: Vec::new(),
            bytes: skeleton.completion_bytes(),
        };
        for (from, to) in &skeleton.fixed {
            state.children[*from] |= 1_u8 << *to;
            state.parents[*to] |= 1_u8 << *from;
        }
        state
    }

    /// Whether `to` is reachable from `from` along directed edges.
    fn reaches(&self, from: usize, to: usize) -> bool {
        let (mut seen, mut frontier) = (1_u8 << from, 1_u8 << from);
        while frontier != 0 {
            let node = frontier.trailing_zeros() as usize;
            frontier &= frontier - 1;
            let next = self.children[node] & !seen;
            seen |= next;
            frontier |= next;
        }
        seen & (1 << to) != 0
    }

    /// Whether `from -> to` keeps the graph acyclic and creates no unshielded
    /// collider at `to` (any such collider is new: `from - to` is undirected in
    /// the CPDAG, so no CPDAG v-structure uses it).
    fn admissible(&self, from: usize, to: usize) -> bool {
        if self.reaches(to, from) {
            return false;
        }
        let others = self.parents[to] & !(1 << from);
        others & !self.skeleton.adjacent[from] == 0
    }

    fn orient(&mut self, from: usize, to: usize, on: bool) {
        if on {
            self.children[from] |= 1 << to;
            self.parents[to] |= 1 << from;
        } else {
            self.children[from] &= !(1 << to);
            self.parents[to] &= !(1 << from);
        }
    }

    fn walk(&mut self, k: usize, mask: u32, charge: &mut Charge<'_>) -> Result<(), SearchStop> {
        let Some(&(a, b)) = self.skeleton.undirected.get(k) else {
            charge(k, self.bytes.saturating_add(self.skeleton.completion_bytes()))?;
            self.masks.push(mask);
            self.bytes = self.bytes.saturating_add(self.skeleton.completion_bytes());
            return Ok(());
        };
        for (from, to, bit) in [(a, b, 1_u32 << k), (b, a, 0)] {
            charge(k + 1, self.bytes)?;
            if self.admissible(from, to) {
                self.orient(from, to, true);
                let result = self.walk(k + 1, mask | bit, charge);
                self.orient(from, to, false);
                result?;
            }
        }
        Ok(())
    }
}

/// Run the orientation search, returning the masks found and the stop, if any.
fn run_walk(skeleton: &Skeleton, charge: &mut Charge<'_>) -> (Vec<u32>, Option<SearchStop>) {
    let mut orienter = Orienter::new(skeleton);
    let stop = orienter.walk(0, 0, charge).err();
    (orienter.masks, stop)
}

/// A started budget, or the receipt of a stop before entry.
fn start(limits: SearchLimits, ctx: &ExecutionContext) -> Result<SharedSearch<'_>, SearchReceipt> {
    SearchBudget::new(limits, ctx).map(SharedSearch::new)
}

/// Turn the masks and stop of a walk into an enumeration, bounding the
/// remainder of a stopped one and checking that a finished one is a CPDAG.
fn finish(
    skeleton: &Skeleton,
    masks: &[u32],
    stop: Option<SearchReceipt>,
) -> Result<CpdagEnumeration, ScenarioSetRefusal> {
    let mut completions: Vec<CompletionDag> =
        masks.iter().map(|m| skeleton.completion(*m)).collect();
    completions.sort_by(|a, b| a.identity.cmp(&b.identity));
    let cpdag_identity = skeleton.cpdag_identity();
    let variables = skeleton.variables.clone();
    if let Some(mut receipt) = stop {
        // No work may resume after an operation, memory or cancellation stop.
        // Every completion is one orientation of the undirected edges; this
        // conservative bound deliberately includes invalid/unvisited orientations.
        let total = 1_usize << skeleton.undirected.len();
        let not_enumerated = total.saturating_sub(completions.len());
        receipt.explored = completions.iter().map(|c| c.identity.to_string()).collect();
        receipt.unevaluated =
            vec![format!("cpdag_completions_not_enumerated_upper_bound:{not_enumerated}")];
        return Ok(CpdagEnumeration {
            cpdag_identity,
            variables,
            completions,
            not_enumerated,
            receipt: Some(receipt),
        });
    }
    check_maximally_oriented(skeleton, masks)?;
    Ok(CpdagEnumeration {
        cpdag_identity,
        variables,
        completions,
        not_enumerated: 0,
        receipt: None,
    })
}

/// A CPDAG has at least one completion and every undirected edge is oriented
/// both ways among them; otherwise some orientation was compelled and the
/// supplied graph is not the equivalence class's CPDAG.
fn check_maximally_oriented(skeleton: &Skeleton, masks: &[u32]) -> Result<(), ScenarioSetRefusal> {
    if masks.is_empty() {
        return Err(not_a_cpdag(
            "the graph has no consistent DAG completion (a cycle or an unsupported collider)",
        ));
    }
    // Chickering (1995), Lemma 1 and Theorem 2: equivalent DAGs connect
    // through covered-edge reversals. If a supplied fixed edge were reversible,
    // the first reversal of any fixed edge would be covered in a completion we
    // enumerated. Refuse that witness instead of silently shrinking the class.
    for &mask in masks {
        let mut parents = Orienter::new(skeleton).parents;
        for (k, &(a, b)) in skeleton.undirected.iter().enumerate() {
            let (from, to) = if mask & (1 << k) != 0 { (a, b) } else { (b, a) };
            parents[to] |= 1 << from;
        }
        if skeleton.fixed.iter().any(|&(a, b)| parents[b] == (parents[a] | (1 << a))) {
            return Err(not_a_cpdag("a directed edge is reversible within the equivalence class"));
        }
    }
    for k in 0..skeleton.undirected.len() {
        let forward = masks.iter().any(|m| m & (1 << k) != 0);
        let backward = masks.iter().any(|m| m & (1 << k) == 0);
        if !(forward && backward) {
            let (a, b) = skeleton.undirected[k];
            return Err(not_a_cpdag(format!(
                "undirected edge {} - {} is compelled in every completion; the graph is not maximally oriented",
                skeleton.variables[a].raw(),
                skeleton.variables[b].raw()
            )));
        }
    }
    Ok(())
}

/// Enumerate every DAG completion of a CPDAG under one search budget.
///
/// # Errors
/// `route_not_supported` / `cpdag_scenarios.bounds_exceeded` for more than six
/// nodes; `route_not_supported` / `cpdag_scenarios.selection_or_latent` for a
/// selection target or latent pair; `invalid_argument` /
/// `cpdag_scenarios.not_a_cpdag` for an input that is not a valid CPDAG.
/// A budget, memory or cancellation stop is not an error: it is returned in the
/// enumeration's receipt with a conservative upper bound on the unenumerated count.
pub fn enumerate_cpdag_completions(
    input: &CpdagCompletionInput,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<CpdagEnumeration, ScenarioSetRefusal> {
    let skeleton = Skeleton::from_input(input)?;
    match start(limits, ctx) {
        Err(receipt) => finish(&skeleton, &[], Some(receipt)),
        Ok(mut search) => {
            let (masks, stop) =
                run_walk(&skeleton, &mut |depth, bytes| search.charge(depth, bytes));
            let receipt = stop.map(|s| search.receipt(s, Vec::new(), Vec::new()));
            finish(&skeleton, &masks, receipt)
        }
    }
}

/// Bytes one entered completion scenario holds: the same shape the 2.2 set charges.
fn scenario_bytes(variables: usize) -> u64 {
    u64::try_from(variables.saturating_mul(variables).saturating_mul(64).saturating_add(512))
        .unwrap_or(u64::MAX)
}

/// One completion as an ADMG with no selection and no bidirected edge.
fn completion_scenario(
    completion: &CompletionDag,
    variables: &[VariableId],
    coordinates: &Arc<[ScenarioCoordinate]>,
) -> Result<TransportScenario, IdentificationError> {
    let mut graph = Admg::empty();
    for variable in variables {
        graph.add_node(NodeRef::Static(*variable))?;
    }
    let dense = |variable: &VariableId| {
        variables
            .binary_search(variable)
            .ok()
            .and_then(|i| u32::try_from(i).ok())
            .map(DenseNodeId::from_raw)
    };
    for (parent, child) in completion.edges.iter() {
        let (Some(parent), Some(child)) = (dense(parent), dense(child)) else {
            return Err(IdentificationError::invalid_input(
                "completion edge names an unknown node",
            ));
        };
        graph.insert_directed(parent, child)?;
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(Vec::new()))?;
    Ok(TransportScenario {
        name: Arc::clone(&completion.identity),
        diagram,
        weight: None,
        coordinates: Arc::clone(coordinates),
    })
}

/// The evidence of each completion, validated against the enumerated identities.
enum Slot {
    Catalog { catalog: EvidenceCatalog, identity: Arc<str> },
    Unbound,
}

fn bind_evidence(
    evidence: &CpdagEvidence,
    enumeration: &CpdagEnumeration,
    set: &TransportScenarioSet,
) -> Result<Vec<Slot>, CpdagScenarioError> {
    let check = |catalog: &EvidenceCatalog| -> Result<(), CpdagScenarioError> {
        catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
        set.check_catalog(catalog)?;
        Ok(())
    };
    match evidence {
        CpdagEvidence::Shared { evidence_identity, catalog } => {
            check(catalog)?;
            Ok(enumeration
                .completions
                .iter()
                .map(|_| Slot::Catalog {
                    catalog: catalog.clone(),
                    identity: Arc::clone(evidence_identity),
                })
                .collect())
        }
        CpdagEvidence::PerCompletion(bindings) => {
            let known =
                enumeration.completions.iter().map(|c| &c.identity).collect::<BTreeSet<_>>();
            let mut named = BTreeSet::new();
            for binding in bindings {
                if binding.certified_for != binding.completion {
                    return Err(mismatch(format!(
                        "evidence for completion {} is certified for completion {}",
                        binding.completion, binding.certified_for
                    ))
                    .into());
                }
                // A partial enumeration cannot refute an identity it has not reached.
                let may_exist = enumeration.not_enumerated > 0;
                if (!known.contains(&binding.completion) && !may_exist)
                    || !named.insert(Arc::clone(&binding.completion))
                {
                    return Err(mismatch(format!(
                        "evidence names completion {}, which is unknown or bound twice",
                        binding.completion
                    ))
                    .into());
                }
                check(&binding.catalog)?;
            }
            Ok(enumeration
                .completions
                .iter()
                .map(|c| {
                    bindings.iter().find(|b| b.completion == c.identity).map_or(
                        Slot::Unbound,
                        |b| Slot::Catalog {
                            catalog: b.catalog.clone(),
                            identity: Arc::clone(&b.evidence_identity),
                        },
                    )
                })
                .collect())
        }
    }
}

/// Decide one completion on the shared budget; a stop comes back as its bound.
fn decide_completion(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<Result<ScenarioOutcome, SearchStop>, IdentificationError> {
    Ok(Ok(
        match identify_catalog_transport_metered(diagram, query, catalog, search.meter(), ctx)? {
            CatalogTransportResult::Identified(bound) => ScenarioOutcome::Identified(bound),
            CatalogTransportResult::MissingEvidence { obligations, .. } => {
                ScenarioOutcome::MissingEvidence { obligations }
            }
            CatalogTransportResult::NotCertified { obligations, .. } => {
                match identify_classical_transport_metered(diagram, query, search.meter(), ctx)? {
                    ClassicalTransportResult::ProvenNonTransportable(hedge) => {
                        ScenarioOutcome::StructurallyUnidentified(Box::new(hedge.to_record()))
                    }
                    _ => ScenarioOutcome::NotCertified { obligations },
                }
            }
        },
    ))
}

fn unbound_outcome(identity: &str) -> ScenarioOutcome {
    ScenarioOutcome::MissingEvidence {
        obligations: Arc::from([Arc::from(format!(
            "{CPDAG_NO_EVIDENCE_DETAIL}: no evidence is bound to completion {identity}"
        ))]),
    }
}

/// Decide every enumerated completion on the shared budget, in identity order.
fn decide_set(
    set: &TransportScenarioSet,
    question: &ScenarioQuestion,
    slots: &[Slot],
    search: &mut SharedSearch<'_>,
    mut retained: u64,
    ctx: &ExecutionContext,
) -> Result<(Vec<ScenarioDecision>, Option<SearchReceipt>), IdentificationError> {
    let query = question.base();
    let total = set.scenarios().len();
    let remaining = |from: usize| -> Vec<String> {
        set.scenarios()[from..].iter().map(|s| s.name.to_string()).collect()
    };
    let entry_bytes = scenario_bytes(set.schema().len());
    let (mut decisions, mut receipt, mut explored) = (Vec::with_capacity(total), None, Vec::new());
    let mut stopped: Option<SearchStop> = None;
    for (index, (scenario, slot)) in set.scenarios().iter().zip(slots).enumerate() {
        if stopped.is_none() {
            search.begin(retained);
            if let Err(stop) = search.charge(1, entry_bytes) {
                receipt = Some(search.receipt(stop, explored.clone(), remaining(index)));
                stopped = Some(stop);
            }
        }
        let outcome = if let Some(stop) = stopped {
            ScenarioOutcome::Unevaluated { stop }
        } else {
            let decided = match slot {
                Slot::Unbound => Ok(Ok(unbound_outcome(&scenario.name))),
                Slot::Catalog { catalog, .. } => {
                    decide_completion(&scenario.diagram, query, catalog, search, ctx)
                }
            };
            match decided {
                Ok(Ok(outcome)) => {
                    explored.push(scenario.name.to_string());
                    retained = retained.saturating_add(search.peak_bytes());
                    outcome
                }
                Ok(Err(stop)) => {
                    receipt = Some(search.receipt(stop, explored.clone(), remaining(index)));
                    stopped = Some(stop);
                    ScenarioOutcome::Unevaluated { stop }
                }
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    receipt = Some(search.receipt(stop, explored.clone(), remaining(index)));
                    stopped = Some(stop);
                    ScenarioOutcome::Unevaluated { stop }
                }
                Err(error) => return Err(error),
            }
        };
        decisions.push(ScenarioDecision { scenario: scenario.clone(), outcome });
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)] // At most 720 completions.
            progress.report(decisions.len() as f64 / total as f64, "cpdag completions");
        }
    }
    Ok((decisions, receipt))
}

/// Enumerate every DAG completion of `input`, bind evidence to each, and decide
/// each through the 2.2 classical catalog route on one shared budget.
///
/// Statuses are those of the supplied-scenario row (`identified`,
/// `structurally_unidentified`, `missing_evidence`, `not_certified`,
/// `unevaluated`); `unsupported_provider` and `support_failure` arise when the
/// estimate layer prepares the laws. No completion is weighted and none is
/// dropped: completions that did not identify stay in the decision.
///
/// # Errors
/// The refusals of [`enumerate_cpdag_completions`]; `schema_mismatch` /
/// `scenarios.coordinate_mismatch` for coordinates, query or catalog outside
/// the shared schema; `invalid_argument` /
/// `cpdag_scenarios.evidence_identity_mismatch` for a binding that names an
/// unknown or different completion identity, or one completion twice; and an
/// invalid query or catalog.
pub fn decide_cpdag_completions(
    input: &CpdagCompletionInput,
    coordinates: &Arc<[ScenarioCoordinate]>,
    query: &ClassicalTransportQuery,
    evidence: &CpdagEvidence,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<CpdagCompletionDecision, CpdagScenarioError> {
    let skeleton = Skeleton::from_input(input)?;
    let request =
        Request { coordinates, question: ScenarioQuestion::Classical(query.clone()), evidence };
    match start(budget, ctx) {
        Err(receipt) => {
            // A budget that stopped before entry decides nothing; the whole
            // class is counted as unenumerated.
            let enumeration = finish(&skeleton, &[], Some(receipt))?;
            Ok(CpdagCompletionDecision {
                cpdag_identity: enumeration.cpdag_identity,
                completions: Vec::new(),
                not_enumerated: enumeration.not_enumerated,
                decision: None,
                receipt: enumeration.receipt,
            })
        }
        Ok(mut search) => {
            let (masks, stop) =
                run_walk(&skeleton, &mut |depth, bytes| search.charge(depth, bytes));
            let receipt = stop.map(|s| search.receipt(s, Vec::new(), Vec::new()));
            let enumeration = finish(&skeleton, &masks, receipt)?;
            let retained = u64::try_from(enumeration.completions.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(skeleton.completion_bytes());
            decide_enumerated(enumeration, &request, retained, budget, &mut search, ctx)
        }
    }
}

/// The premises every completion is decided against.
struct Request<'r> {
    coordinates: &'r Arc<[ScenarioCoordinate]>,
    question: ScenarioQuestion,
    evidence: &'r CpdagEvidence,
}

/// The scenario set of the completions found, with the evidence slots checked,
/// and the receipt records.
fn build_set(
    enumeration: &CpdagEnumeration,
    request: &Request<'_>,
) -> Result<(TransportScenarioSet, Vec<Slot>, Vec<CompletionRecord>), CpdagScenarioError> {
    if enumeration.completions.len() > SCENARIO_MAX_COUNT {
        return Err(unsupported(
            CPDAG_BOUNDS_EXCEEDED_DETAIL,
            format!(
                "{} completions exceed the scenario bound of {SCENARIO_MAX_COUNT}",
                enumeration.completions.len()
            ),
        )
        .into());
    }
    let (coordinates, question, evidence) =
        (request.coordinates, &request.question, request.evidence);
    let scenarios = enumeration
        .completions
        .iter()
        .map(|c| completion_scenario(c, &enumeration.variables, coordinates))
        .collect::<Result<Vec<_>, _>>()?;
    let set = TransportScenarioSet::try_new(scenarios)?;
    set.check_question(question)?;
    let slots = bind_evidence(evidence, enumeration, &set)?;
    let records = enumeration
        .completions
        .iter()
        .zip(&slots)
        .map(|(c, slot)| CompletionRecord {
            identity: Arc::clone(&c.identity),
            edges: Arc::clone(&c.edges),
            evidence_identity: match slot {
                Slot::Catalog { identity, .. } => Some(Arc::clone(identity)),
                Slot::Unbound => None,
            },
        })
        .collect();
    Ok((set, slots, records))
}

fn decide_enumerated(
    enumeration: CpdagEnumeration,
    request: &Request<'_>,
    retained: u64,
    budget: SearchLimits,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<CpdagCompletionDecision, CpdagScenarioError> {
    if enumeration.completions.is_empty() {
        return Ok(CpdagCompletionDecision {
            cpdag_identity: enumeration.cpdag_identity,
            completions: Vec::new(),
            not_enumerated: enumeration.not_enumerated,
            decision: None,
            receipt: enumeration.receipt,
        });
    }
    let (set, slots, records) = build_set(&enumeration, request)?;
    let question = &request.question;
    let (decisions, receipt) = if let Some(stopped) = &enumeration.receipt {
        // Enumeration stopped: the completions found are real but undecided.
        let outcome = ScenarioOutcome::Unevaluated { stop: stopped.stop };
        let decisions = set
            .scenarios()
            .iter()
            .map(|s| ScenarioDecision { scenario: s.clone(), outcome: outcome.clone() })
            .collect();
        (decisions, Some(stopped.clone()))
    } else {
        decide_set(&set, question, &slots, search, retained, ctx)?
    };
    let receipt = enumeration.receipt.clone().or(receipt);
    let decision = ScenarioSetDecision {
        set,
        question: question.clone(),
        decisions,
        receipt: receipt.clone(),
        limits: ScenarioDecisionLimits { budget, memory_limit_bytes: ctx.memory.hard_limit_bytes },
    };
    Ok(CpdagCompletionDecision {
        cpdag_identity: enumeration.cpdag_identity,
        completions: records,
        not_enumerated: enumeration.not_enumerated,
        decision: Some(decision),
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(n: u32, undirected: &[(u32, u32)], directed: &[(u32, u32)]) -> CpdagCompletionInput {
        let mut cpdag = Cpdag::with_variables(n);
        for (a, b) in undirected {
            cpdag
                .insert_undirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        for (a, b) in directed {
            cpdag
                .insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        CpdagCompletionInput::new(cpdag)
    }

    fn count(input: &CpdagCompletionInput) -> usize {
        let ctx = ExecutionContext::for_tests(1);
        let limits = SearchLimits { operations: 1_000_000, depth: 64 };
        enumerate_cpdag_completions(input, limits, &ctx).map_or(usize::MAX, |e| e.total())
    }

    #[test]
    fn chain_collider_and_triangle_have_the_known_class_sizes() {
        assert_eq!(count(&input(3, &[(0, 1), (1, 2)], &[])), 3);
        assert_eq!(count(&input(3, &[], &[(0, 1), (2, 1)])), 1);
        assert_eq!(count(&input(3, &[(0, 1), (1, 2), (0, 2)], &[])), 6);
    }

    #[test]
    fn a_compelled_edge_is_not_a_cpdag() {
        // a -> b - c: Meek rule 1 compels b -> c, so the graph is not maximally oriented.
        let refused = enumerate_cpdag_completions(
            &input(3, &[(1, 2)], &[(0, 1)]),
            SearchLimits { operations: 1000, depth: 64 },
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert_eq!(refused.detail, CPDAG_NOT_A_CPDAG_DETAIL);
    }
}
