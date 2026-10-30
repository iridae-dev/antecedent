//! General finite counterfactual transport on a static semi-Markovian graph.
//!
//! # Scope
//!
//! One frozen contract, [`COUNTERFACTUAL_TRANSPORT_CONTRACT`]: finitely many
//! observed variables (their number is limited only by the width of the
//! variable sets, [`COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES`]) with finitely many declared
//! levels; the numbers of variables, levels, worlds, atoms, sources,
//! controllable variables and regimes are bounded only by the search budget, which every
//! enumeration of subsets, cliques and level cells is charged to (a request the budget cannot
//! pay for ends as an exhausted search with a receipt, never a verdict),
//! an acyclic graph with bidirected edges for latent confounding, one selection
//! diagram per source population (the variables whose mechanism may differ from
//! the target's), and population- and regime-labelled evidence laws. The query is a
//! [`CounterfactualEventQuery`]: a conjunction of events over worlds that carry hard
//! interventions and edge routes (nested counterfactuals such as `Y_{x, M_{x'}}`),
//! optionally conditioned on a second conjunction. Continuous, cyclic, temporal and
//! sampled inference are separate routes and are refused by name.
//!
//! # Theorem and what is implemented
//!
//! The theorem family is Correa, Lee and Bareinboim, "Counterfactual
//! Transportability: A Formal Approach" (ICML 2022, PMLR 162): a counterfactual
//! distribution of a target population is expressed through counterfactual
//! *factors* that are single-world interventional laws, each of which is either
//! invariant (no selection node into its mechanism, so a source law transports) or
//! identified from target evidence. This module implements a sound, bounded
//! decision procedure of that shape, not the paper's algorithm verbatim:
//!
//! 1. *Unnesting and simplification.* Worlds and edge routes are unnested into a
//!    potential-outcome network whose nodes merge when one mechanism reads one input
//!    (the counterfactual graph of Shpitser and Pearl's ID*), with the consistency
//!    substitution of event-fixed nodes (`network`).
//! 2. *Ancestral components.* Unread nodes are pruned to the ancestral closure of the
//!    event; the nodes split into districts (copies of one variable and bidirected
//!    neighbours share exogenous terms).
//! 3. *Counterfactual-factor consistency.* A district is a single-world factor
//!    `Q[S] = P(S | do(parents))` exactly when it has one node per variable and one
//!    reading per outside parent; otherwise it is a cross-world factor no experiment
//!    contains (a recanting-type conflict) and is not derived.
//! 4. *Conditional reduction.* A conditioning atom whose node the event is
//!    m-separated from, given the other conditioning nodes, moves into an
//!    intervention (the IDC* reduction); the original conditioning mass stays a
//!    positivity guard where it is itself derivable.
//! 5. *Selection-aware district transport.* Each single-world factor is derived from
//!    supplied regime laws by marginalization to ancestral sets and district
//!    extraction (Tian and Pearl 2002), transferring a source's factor to the target
//!    when no selection target lies in its variables (`closure`).
//! 6. *Binary consistency complement.* `P(E, X = x) = P(E) - P(E, X = x')` for a
//!    two-level variable, used when the direct factorization conflicts.
//!
//! No independent marginals are ever paired into a cross-world law: a factor either
//! is one law of one regime (population, intervention set) or is derived from one.
//!
//! # Decisions
//!
//! [`CounterfactualTransportDecision`] separates a checked derivation, a proven
//! obstruction (a verified two-model witness), declared-but-absent evidence, an
//! out-of-contract request, "not certified", and a budget or cancellation stop. A
//! failure to derive is *never* an obstruction by itself: the obstruction is claimed
//! only when a bounded search finds two finite models of the declared class that agree
//! on every regime of the declared family and differ on the query, re-verified by
//! exact rational enumeration. Completeness of the derivation is not claimed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod closure;
mod network;
mod witness;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::search_budget::{FunctionalEnumeration, SearchBudgetError};
use antecedent_core::{
    CounterfactualEventQuery, DEFAULT_SEARCH_MEMORY_BYTES, EvidenceCatalog, ExecutionContext,
    ExogenousCoupling, RegimeId, SearchBudget, SearchLimits, SearchReceipt, SearchStop,
    TransportOutcomeKind, VariableId,
};
use antecedent_graph::Admg;
use serde::{Deserialize, Serialize};

use crate::error::IdentificationError;
use closure::{Closure, ItemSource, Seed, close, reachable, regime_measured_is_factor};
use network::{Built, Conflict, Factored, Level, WorldIdx, build_network, factorize, separated};
pub use witness::{CounterfactualTransportWitness, CounterfactualTransportWitnessCheck};
use witness::{WitnessOutcome, WitnessProblem, search_witness, verify_witness};

/// The contract a derivation attests.
pub const COUNTERFACTUAL_TRANSPORT_CONTRACT: &str =
    "static_finite_semi_markovian_counterfactual_transport_v1";
/// Most observed variables the representation holds: variable sets are `u32` bit sets (also
/// in the stored items and regimes of a derivation), and the full set of `n` variables needs
/// `n < 32`. This is the width of the representation, not a work bound: the ancestrally
/// closed subsets of every law, the cliques of the witness search, the subsets of every
/// controllable set and the cells of a summation are each charged to the search budget as
/// they are visited.
pub const COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES: usize = 31;
/// Worlds of one query under the earlier fixed ceiling. Not enforced by the decision: the
/// work grows with the request and is charged to the search budget (a bound of stored
/// artifacts only).
pub const COUNTERFACTUAL_TRANSPORT_MAX_WORLDS: usize = 6;
/// Atoms of the event and conditioning conjunctions under the earlier fixed ceiling. Not
/// enforced by the decision (a bound of stored artifacts only).
pub const COUNTERFACTUAL_TRANSPORT_MAX_ATOMS: usize = 16;
/// Nodes of one conjunction's kept network under the earlier fixed ceiling. Not enforced.
pub const COUNTERFACTUAL_TRANSPORT_MAX_NODES: usize = 24;
/// Source populations under the earlier fixed ceiling. Not enforced by the decision (a bound
/// of stored artifacts only).
pub const COUNTERFACTUAL_TRANSPORT_MAX_SOURCES: usize = 4;
/// Controllable variables per source under the earlier fixed ceiling. Not enforced by the
/// decision (a bound of stored artifacts only).
pub const COUNTERFACTUAL_TRANSPORT_MAX_CONTROLLABLE: usize = 4;
/// Catalog regimes under the earlier fixed ceiling. Not enforced by the decision (a bound of
/// stored artifacts only).
pub const COUNTERFACTUAL_TRANSPORT_MAX_REGIMES: usize = 64;
/// Former ceiling on the joint level cells of the variables a derivation sums over. No longer
/// enforced: a summation is quoted from the caller's search budget.
pub const COUNTERFACTUAL_TRANSPORT_MAX_SUM_CELLS: usize = 1 << 16;
/// Nesting of the binary consistency complement tried first. A derivation that fails with a
/// complement left untried at this nesting is retried one level deeper, on the same budget,
/// until no complement is cut off (the binary atoms bound the nesting) or the budget stops it.
const INITIAL_COMPLEMENT_DEPTH: usize = 2;
/// Default search limits: operations charged across every stage, and depth.
pub const COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS: SearchLimits =
    SearchLimits { operations: 50_000, depth: 64 };
/// Largest limits a caller may ask for.
pub const COUNTERFACTUAL_TRANSPORT_MAX_LIMITS: SearchLimits =
    SearchLimits { operations: 400_000, depth: 256 };
/// Estimated live-state cap of one decision, further limited by the context.
pub const COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES: u64 = DEFAULT_SEARCH_MEMORY_BYTES;

/// A request outside a declared bound.
pub(crate) const BOUNDS_EXCEEDED: &str = "counterfactual_transport.bounds_exceeded";
/// A malformed query, graph or catalog.
const INVALID_QUERY: &str = "counterfactual_transport.invalid_query";
const INVALID_CATALOG: &str = "counterfactual_transport.invalid_catalog";
/// The graph is outside the contract (directed cycle).
const GRAPH_OUTSIDE_CONTRACT: &str = "counterfactual_transport.graph_outside_contract";

/// Refuse a directed cycle before the transport problem is ordered.
fn refuse_transport_cycle(
    node_count: usize,
    directed: &[(usize, usize)],
) -> Result<(), IdentificationError> {
    let edges = directed_edge_ids(directed);
    if let Err(refusal) = antecedent_graph::error::refuse_directed_cycle(
        antecedent_graph::error::DagOperation::Identification,
        antecedent_graph::error::CyclicNextAttempt::Unlicensed {
            consumer: "counterfactual_transport",
            attempt: "hold the directed edges as a CyclicMixedGraph; condensation names the \
                      feedback and sigma_separates answers a conditional-independence \
                      question. This cell has no cyclic identification license",
        },
        node_count,
        &edges,
    ) {
        return Err(IdentificationError::invalid_input(format!(
            "{GRAPH_OUTSIDE_CONTRACT}: {refusal}"
        )));
    }
    Ok(())
}

/// Dense indexes of a problem whose variable count already fits the contract.
#[allow(
    clippy::cast_possible_truncation,
    reason = "node indices are bounded by the graph node limit, far below u32::MAX"
)]
fn directed_edge_ids(edges: &[(usize, usize)]) -> Vec<(u32, u32)> {
    edges.iter().map(|&(from, to)| (from as u32, to as u32)).collect()
}

/// The coupling is not the shared latent exogenous coupling.
const COUPLING_UNSUPPORTED: &str = "counterfactual_transport.coupling_unsupported";
/// A regime law of a selected sample is supplied; the contract has no selection-bias model.
const SELECTED_SAMPLE: &str = "counterfactual_transport.selected_sample_law";
const NO_FACTOR_DERIVATION: &str = "counterfactual_transport.factor_not_derivable";
const MISSING_REGIME: &str = "counterfactual_transport.missing_regime";
const CHECKED_OBSTRUCTION: &str = "counterfactual_transport.checked_obstruction";
const BUDGET: &str = "counterfactual_transport.budget";
const INVALID_DERIVATION: &str = "counterfactual_transport.invalid_derivation";
const INVALID_WITNESS: &str = "counterfactual_transport.invalid_witness";

/// The frozen `(reason code, counterfactual_transport.* detail)` pair of an error
/// raised by this route: `route_not_supported` for a bound or a graph outside the
/// contract, `invalid_argument` for an invalid query or catalog,
/// `transport_not_certified` for a derivation or witness that does not check, and
/// `transport_budget_cancel` for a stop. `None` for an error the route does not own.
#[must_use]
#[doc(hidden)]
pub fn counterfactual_transport_refusal(
    error: &IdentificationError,
) -> Option<(&'static str, &'static str)> {
    use antecedent_core::reason_code;
    match error {
        IdentificationError::UnsupportedInput { code } if *code == BOUNDS_EXCEEDED => {
            Some((reason_code!("route_not_supported"), BOUNDS_EXCEEDED))
        }
        IdentificationError::UnsupportedInput { code } if *code == COUPLING_UNSUPPORTED => {
            Some((reason_code!("route_not_supported"), COUPLING_UNSUPPORTED))
        }
        IdentificationError::InvalidInput { message }
            if message.starts_with(GRAPH_OUTSIDE_CONTRACT) =>
        {
            Some((reason_code!("route_not_supported"), GRAPH_OUTSIDE_CONTRACT))
        }
        IdentificationError::InvalidInput { message } if message.starts_with(INVALID_QUERY) => {
            Some((reason_code!("invalid_argument"), INVALID_QUERY))
        }
        IdentificationError::InvalidCatalog { message } if message.starts_with(INVALID_CATALOG) => {
            Some((reason_code!("invalid_argument"), INVALID_CATALOG))
        }
        IdentificationError::InvalidDerivation { code }
            if [INVALID_DERIVATION, INVALID_WITNESS].contains(code) =>
        {
            Some((reason_code!("transport_not_certified"), code))
        }
        IdentificationError::MissingEvidence { .. } => {
            Some((reason_code!("transport_missing_evidence"), MISSING_REGIME))
        }
        IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
            Some((reason_code!("transport_budget_cancel"), BUDGET))
        }
        _ => None,
    }
}

fn invalid_query(detail: impl std::fmt::Display) -> IdentificationError {
    IdentificationError::invalid_input(format!("{INVALID_QUERY}: {detail}"))
}

fn invalid_catalog(detail: impl std::fmt::Display) -> IdentificationError {
    IdentificationError::invalid_catalog(format!("{INVALID_CATALOG}: {detail}"))
}

const fn bounds_exceeded() -> IdentificationError {
    IdentificationError::UnsupportedInput { code: BOUNDS_EXCEEDED }
}

// ---------------------------------------------------------------------------
// The problem: variables, declared levels and the graph.
// ---------------------------------------------------------------------------

/// The frozen model class a decision is made in: observed variables with their
/// declared finite levels, directed edges and bidirected edges. Canonical:
/// variables ascending by id, edges sorted, levels as IEEE bits in declared order
/// (`-0.0` is `0.0`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportProblem {
    variables: Vec<VariableId>,
    levels: Vec<Vec<u64>>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    parents: Vec<Vec<usize>>,
    parent_mask: Vec<u32>,
    bidirected_mask: Vec<u32>,
    topo: Vec<usize>,
}

impl CounterfactualTransportProblem {
    /// A problem over `variables` (with their declared levels), `directed` edges
    /// `(parent, child)` and `bidirected` edges.
    ///
    /// # Errors
    ///
    /// `counterfactual_transport.bounds_exceeded` beyond
    /// [`COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES`] variables (the width of the variable
    /// sets; levels and joint cells are not bounded here, the decision's search budget
    /// pays for them);
    /// `counterfactual_transport.invalid_query` for no variables, a repeated
    /// variable, an empty, repeated or non-finite level list, or an edge naming an
    /// unknown variable or a self-loop;
    /// `counterfactual_transport.graph_outside_contract` for a directed cycle.
    pub fn new(
        variables: &[(VariableId, Vec<f64>)],
        directed: &[(VariableId, VariableId)],
        bidirected: &[(VariableId, VariableId)],
    ) -> Result<Self, IdentificationError> {
        let n = variables.len();
        if n == 0 {
            return Err(invalid_query("the graph has no variables"));
        }
        if n > COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES {
            return Err(bounds_exceeded());
        }
        let mut sorted: Vec<&(VariableId, Vec<f64>)> = variables.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        if sorted.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(invalid_query("a variable is declared twice"));
        }
        let ids: Vec<VariableId> = sorted.iter().map(|(id, _)| *id).collect();
        let mut levels: Vec<Vec<u64>> = Vec::with_capacity(n);
        for (id, list) in &sorted {
            let bits: Vec<u64> = list.iter().map(|x| (x + 0.0).to_bits()).collect();
            let distinct: BTreeSet<u64> = bits.iter().copied().collect();
            if list.is_empty()
                || list.iter().any(|x| !x.is_finite())
                || distinct.len() != bits.len()
            {
                return Err(invalid_query(format!(
                    "variable {} needs at least one distinct finite level",
                    id.raw()
                )));
            }
            levels.push(bits);
        }
        let index_of = |v: VariableId| ids.iter().position(|id| *id == v);
        let mut directed_idx: Vec<(usize, usize)> = Vec::with_capacity(directed.len());
        for &(a, b) in directed {
            match (index_of(a), index_of(b)) {
                (Some(i), Some(j)) => directed_idx.push((i, j)),
                _ => {
                    return Err(invalid_query("an edge names an unknown variable"));
                }
            }
        }
        let mut bidirected_idx: Vec<(usize, usize)> = Vec::with_capacity(bidirected.len());
        for &(a, b) in bidirected {
            match (index_of(a), index_of(b)) {
                (Some(i), Some(j)) if i != j => bidirected_idx.push((i.min(j), i.max(j))),
                (Some(_), Some(_)) => {
                    return Err(invalid_query("a bidirected edge cannot be a self-loop"));
                }
                _ => {
                    return Err(invalid_query("an edge names an unknown variable"));
                }
            }
        }
        directed_idx.sort_unstable();
        directed_idx.dedup();
        bidirected_idx.sort_unstable();
        bidirected_idx.dedup();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut parent_mask = vec![0u32; n];
        let mut bidirected_mask = vec![0u32; n];
        for &(a, b) in &directed_idx {
            parents[b].push(a);
            parent_mask[b] |= 1 << a;
        }
        for &(a, b) in &bidirected_idx {
            bidirected_mask[a] |= 1 << b;
            bidirected_mask[b] |= 1 << a;
        }
        refuse_transport_cycle(n, &directed_idx)?;
        // Kahn order, smallest index first. The cycle check above already returned.
        let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
        let mut placed = vec![false; n];
        let mut topo = Vec::with_capacity(n);
        while topo.len() < n {
            let Some(next) = (0..n).find(|&v| !placed[v] && indegree[v] == 0) else {
                return Err(IdentificationError::UnsupportedInput { code: GRAPH_OUTSIDE_CONTRACT });
            };
            placed[next] = true;
            topo.push(next);
            for &(a, b) in &directed_idx {
                if a == next {
                    indegree[b] -= 1;
                }
            }
        }
        Ok(Self {
            variables: ids,
            levels,
            directed: directed_idx,
            bidirected: bidirected_idx,
            parents,
            parent_mask,
            bidirected_mask,
            topo,
        })
    }

    /// The problem of an ADMG whose nodes are static variables, with the declared
    /// `levels` of each variable.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `counterfactual_transport.invalid_query` for a node
    /// that is not a static variable or a variable without declared levels.
    pub fn from_admg(
        admg: &Admg,
        levels: &[(VariableId, Vec<f64>)],
    ) -> Result<Self, IdentificationError> {
        let mut ids = Vec::with_capacity(admg.node_count());
        for dense in admg.node_ids() {
            let Some(variable) = admg.static_variable(dense) else {
                return Err(invalid_query("every node must be a static observed variable"));
            };
            ids.push(variable);
        }
        let mut declared = Vec::with_capacity(ids.len());
        for id in &ids {
            let Some((_, list)) = levels.iter().find(|(v, _)| v == id) else {
                return Err(invalid_query(format!("variable {} has no declared levels", id.raw())));
            };
            declared.push((*id, list.clone()));
        }
        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for dense in admg.node_ids() {
            let Some(a) = admg.static_variable(dense) else { continue };
            for &child in admg.children(dense) {
                if let Some(b) = admg.static_variable(child) {
                    directed.push((a, b));
                }
            }
            for &spouse in admg.bidirected_neighbors(dense) {
                if let Some(b) = admg.static_variable(spouse) {
                    bidirected.push((a, b));
                }
            }
        }
        Self::new(&declared, &directed, &bidirected)
    }

    /// Number of variables.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Variable ids, ascending (the internal index of a variable is its position).
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }

    /// Internal index of a variable id.
    #[must_use]
    pub fn index_of(&self, variable: VariableId) -> Option<usize> {
        self.variables.iter().position(|v| *v == variable)
    }

    /// Declared levels of the variable at internal index `index`, in declared order.
    #[must_use]
    pub fn levels(&self, index: usize) -> Vec<f64> {
        self.levels
            .get(index)
            .map_or_else(Vec::new, |l| l.iter().map(|b| f64::from_bits(*b)).collect())
    }

    /// Index of `level` among the declared levels of the variable at `index`.
    #[must_use]
    pub fn level_index(&self, index: usize, level: f64) -> Option<usize> {
        let bits = (level + 0.0).to_bits();
        self.levels.get(index)?.iter().position(|b| *b == bits)
    }

    /// Variables in the canonical topological order (internal indices; smallest
    /// index first among the available).
    #[must_use]
    pub fn topological_order(&self) -> &[usize] {
        &self.topo
    }

    /// The variable ids whose internal index is a set bit of `mask`, ascending.
    #[must_use]
    pub fn variables_of_mask(&self, mask: u32) -> Vec<VariableId> {
        self.mask_variables(mask)
    }

    /// Directed edges `(parent, child)` in internal indices, sorted.
    #[must_use]
    pub fn directed(&self) -> &[(usize, usize)] {
        &self.directed
    }

    /// Bidirected edges `(low, high)` in internal indices, sorted.
    #[must_use]
    pub fn bidirected(&self) -> &[(usize, usize)] {
        &self.bidirected
    }

    /// Canonical text of the graph and levels (a digest input).
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("counterfactual_transport_problem_v1;variables=[");
        for (id, levels) in self.variables.iter().zip(&self.levels) {
            let _ = write!(out, "{}:", id.raw());
            for bits in levels {
                let _ = write!(out, "{bits:016x},");
            }
            out.push(';');
        }
        out.push_str("];directed=[");
        for (a, b) in &self.directed {
            let _ = write!(out, "{}>{},", self.variables[*a].raw(), self.variables[*b].raw());
        }
        out.push_str("];bidirected=[");
        for (a, b) in &self.bidirected {
            let _ = write!(out, "{}<>{},", self.variables[*a].raw(), self.variables[*b].raw());
        }
        out.push(']');
        out
    }

    fn write_canonical(&self, out: &mut impl std::fmt::Write) -> std::fmt::Result {
        out.write_str("counterfactual_transport_problem_v1;variables=[")?;
        for (id, levels) in self.variables.iter().zip(&self.levels) {
            write!(out, "{}:", id.raw())?;
            for bits in levels {
                write!(out, "{bits:016x},")?;
            }
            out.write_char(';')?;
        }
        out.write_str("];directed=[")?;
        for (a, b) in &self.directed {
            write!(out, "{}>{},", self.variables[*a].raw(), self.variables[*b].raw())?;
        }
        out.write_str("];bidirected=[")?;
        for (a, b) in &self.bidirected {
            write!(out, "{}<>{},", self.variables[*a].raw(), self.variables[*b].raw())?;
        }
        out.write_char(']')
    }

    fn full_mask(&self) -> u32 {
        (1u32 << self.variables.len()) - 1
    }

    fn bidirected_adjacent(&self, a: usize, b: usize) -> bool {
        (self.bidirected_mask[a] >> b) & 1 == 1
    }

    fn mask_variables(&self, mask: u32) -> Vec<VariableId> {
        (0..self.variables.len())
            .filter(|i| (mask >> *i) & 1 == 1)
            .map(|i| self.variables[i])
            .collect()
    }
}

// A fallible UTF-8 sink: callback charges every emitted byte before appending.
struct CheckedCanonical<'a, F, E> {
    text: String,
    check: &'a mut F,
    error: Option<E>,
}
impl<F, E> std::fmt::Write for CheckedCanonical<'_, F, E>
where
    F: FnMut(usize) -> Result<(), E>,
{
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        let mut start = 0;
        while start < text.len() {
            let mut end = start.saturating_add(4096).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            if let Err(error) = (self.check)(end - start) {
                self.error = Some(error);
                return Err(std::fmt::Error);
            }
            self.text.push_str(&text[start..end]);
            start = end;
        }
        Ok(())
    }
}

// Stable merge sort whose comparisons and copies are caller-metered. No standard sort
// comparator can propagate cancellation, and source ordering is stable for equal populations.
fn checked_sort<T: Copy, E, F, C>(values: &[T], check: &mut F, mut cmp: C) -> Result<Vec<T>, E>
where
    F: FnMut(usize) -> Result<(), E>,
    C: FnMut(&T, &T, &mut F) -> Result<std::cmp::Ordering, E>,
{
    check(values.len().saturating_mul(2))?;
    let mut order = Vec::with_capacity(values.len());
    for chunk in values.chunks(512) {
        check(chunk.len())?;
        order.extend_from_slice(chunk);
    }
    let n = order.len();
    let mut next = Vec::with_capacity(n);
    let mut width = 1usize;
    while width < n {
        next.clear();
        for start in (0..n).step_by(width.saturating_mul(2)) {
            let mid = start.saturating_add(width).min(n);
            let end = mid.saturating_add(width).min(n);
            let (mut a, mut b) = (start, mid);
            while a < mid && b < end {
                if cmp(&order[a], &order[b], check)? != std::cmp::Ordering::Greater {
                    next.push(order[a]);
                    a += 1;
                } else {
                    next.push(order[b]);
                    b += 1;
                }
            }
            for chunk in order[a..mid].chunks(512) {
                check(chunk.len())?;
                next.extend_from_slice(chunk);
            }
            for chunk in order[b..end].chunks(512) {
                check(chunk.len())?;
                next.extend_from_slice(chunk);
            }
        }
        std::mem::swap(&mut order, &mut next);
        width = width.saturating_mul(2);
    }
    check(0)?;
    Ok(order)
}

fn checked_ids<E, F>(ids: &[VariableId], check: &mut F) -> Result<Vec<VariableId>, E>
where
    F: FnMut(usize) -> Result<(), E>,
{
    let sorted = checked_sort(ids, check, |a, b, check| {
        check(1)?;
        Ok(a.cmp(b))
    })?;
    check(sorted.len())?;
    let mut unique = Vec::with_capacity(sorted.len());
    for chunk in sorted.chunks(512) {
        check(chunk.len())?;
        for &id in chunk {
            if unique.last() != Some(&id) {
                unique.push(id);
            }
        }
    }
    Ok(unique)
}

fn checked_text_cmp<E, F>(a: &str, b: &str, check: &mut F) -> Result<std::cmp::Ordering, E>
where
    F: FnMut(usize) -> Result<(), E>,
{
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let common = a.len().min(b.len());
    for start in (0..common).step_by(512) {
        let end = start.saturating_add(512).min(common);
        check(end - start)?;
        let ordering = a[start..end].cmp(&b[start..end]);
        if ordering != std::cmp::Ordering::Equal {
            return Ok(ordering);
        }
    }
    check(1)?;
    Ok(a.len().cmp(&b.len()))
}

// ---------------------------------------------------------------------------
// The typed query.
// ---------------------------------------------------------------------------

/// One source population of a counterfactual transport query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportSource {
    /// Population identity (the catalog's regime population).
    pub population: Arc<str>,
    /// Variables whose mechanism may differ between this source and the target.
    pub selection_targets: Arc<[VariableId]>,
    /// Variables this source can experiment on (theoretical availability; the
    /// regimes actually supplied live in the catalog).
    pub controllable: Arc<[VariableId]>,
}

/// A counterfactual query in a target population from labelled evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualTransportQuery {
    /// The counterfactual event and conditioning conjunctions over worlds.
    pub event: CounterfactualEventQuery,
    /// Target population.
    pub target: Arc<str>,
    /// Source populations.
    pub sources: Arc<[CounterfactualTransportSource]>,
}

impl CounterfactualTransportQuery {
    /// Canonical form: sources ordered by population, variable sets sorted.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let sorted = |variables: &[VariableId]| {
            let mut out = variables.to_vec();
            out.sort_unstable();
            out.dedup();
            Arc::<[VariableId]>::from(out)
        };
        let mut sources: Vec<CounterfactualTransportSource> = self
            .sources
            .iter()
            .map(|s| CounterfactualTransportSource {
                population: Arc::clone(&s.population),
                selection_targets: sorted(&s.selection_targets),
                controllable: sorted(&s.controllable),
            })
            .collect();
        sources.sort_by(|a, b| a.population.cmp(&b.population));
        Self {
            event: self.event.clone(),
            target: Arc::clone(&self.target),
            sources: sources.into(),
        }
    }

    /// Canonical text (a digest input).
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let canonical = self.canonical();
        let mut out = format!("counterfactual_transport_query_v1;target={};", canonical.target);
        out.push_str(&canonical.event.canonical_text());
        for source in canonical.sources.iter() {
            let _ = write!(out, ";source={}:select=[", source.population);
            push_id_list(&mut out, source.selection_targets.iter().copied());
            out.push_str("];control=[");
            push_id_list(&mut out, source.controllable.iter().copied());
            out.push(']');
        }
        out
    }
}

/// Append each id as `raw,` — the list rendering every versioned canonical text of this
/// module shares, byte for byte.
fn push_id_list(out: &mut String, ids: impl IntoIterator<Item = VariableId>) {
    use std::fmt::Write as _;
    for id in ids {
        let _ = write!(out, "{},", id.raw());
    }
}

// ---------------------------------------------------------------------------
// Derivation types.
// ---------------------------------------------------------------------------

/// Defines an index newtype whose `Debug` is the bare number, so the canonical text of a
/// derivation (which renders derivation types with `{:?}`) is the text it had when these
/// indices were plain `usize`s.
macro_rules! index_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            /// The index `index`.
            #[must_use]
            pub const fn new(index: usize) -> Self {
                Self(index)
            }

            /// The index as a `usize`.
            #[must_use]
            pub const fn get(self) -> usize {
                self.0
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl From<usize> for $name {
            fn from(index: usize) -> Self {
                Self(index)
            }
        }

        impl From<$name> for usize {
            fn from(index: $name) -> Self {
                index.0
            }
        }
    };
}

index_newtype! {
    /// Internal index of an observed variable of a [`CounterfactualTransportProblem`] (its
    /// position in the ascending order of variable ids).
    CounterfactualVariableIndex
}

index_newtype! {
    /// Index of a declared level among the declared levels of one variable, in declared order.
    CounterfactualLevelIndex
}

index_newtype! {
    /// Identity of a summation symbol, unique within one functional.
    CounterfactualSymbolId
}

/// A level argument of a factor: a declared level index or a summation symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CounterfactualLevel {
    /// A declared level, by index into the variable's declared levels.
    Fixed(CounterfactualLevelIndex),
    /// The summation symbol of the free node with this id.
    Symbol(CounterfactualSymbolId),
}

/// A summation symbol: a free node of the unnested network and its variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CounterfactualSymbol {
    /// Symbol id, unique within one functional.
    pub id: CounterfactualSymbolId,
    /// Internal variable index whose levels the symbol ranges over.
    pub variable: CounterfactualVariableIndex,
}

/// One single-world factor `Q[S](outputs | parents)` of the target population.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualFactor {
    /// Item of the derivation that supplies the law.
    pub item: usize,
    /// `(variable, level)` of every member of the factor's district.
    pub outputs: Vec<(CounterfactualVariableIndex, CounterfactualLevel)>,
    /// `(variable, level)` of every parent outside the district.
    pub parents: Vec<(CounterfactualVariableIndex, CounterfactualLevel)>,
}

/// A sum over free nodes of a product of single-world factors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualSum {
    /// Symbols summed over their variables' declared levels.
    pub symbols: Vec<CounterfactualSymbol>,
    /// Factors multiplied (no factor is the constant one).
    pub factors: Vec<CounterfactualFactor>,
}

/// The identified probability of one conjunction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CounterfactualFunctional {
    /// Probability zero: the conjunction contradicts itself or an intervention.
    Zero,
    /// Sum over the free nodes of the product of the district factors.
    Sum(CounterfactualSum),
    /// `P(E) = P(E without atom) - P(E with the atom's variable flipped)` for a
    /// two-level variable (the consistency complement).
    Complement {
        /// Internal variable index of the complemented atom.
        variable: CounterfactualVariableIndex,
        /// The conjunction with the atom removed.
        whole: Box<CounterfactualFunctional>,
        /// The conjunction with the atom at the variable's other level.
        flipped: Box<CounterfactualFunctional>,
    },
}

/// Where a derived law comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterfactualItemSource {
    /// A supplied regime law `P(variables | do(intervened))`.
    Regime {
        /// The catalog regime.
        regime: RegimeId,
        /// Bit set of the regime's intervened variables (internal indices).
        intervened: u32,
    },
    /// A marginal of item `from` over an ancestrally closed subset.
    Marginal {
        /// Source item.
        from: usize,
    },
    /// The factor of one district of item `from`.
    Extract {
        /// Source item.
        from: usize,
    },
    /// The same law in the target population (no selection target involved).
    Transfer {
        /// Source item.
        from: usize,
    },
}

/// One law of the derivation: `Q^population[variables]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterfactualItem {
    /// Population index (0 is the target; sources follow in canonical order).
    pub population: usize,
    /// Bit set of the item's variables (internal indices).
    pub variables: u32,
    /// How the law is obtained.
    pub source: CounterfactualItemSource,
}

/// A conditioning atom the reduction moved into an intervention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterfactualMovedAtom {
    /// World index in the query.
    pub world: usize,
    /// Internal variable index.
    pub variable: CounterfactualVariableIndex,
    /// Declared level index.
    pub level: CounterfactualLevelIndex,
}

/// A cited catalog regime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualCitedRegime {
    /// The regime.
    pub regime: RegimeId,
    /// Its population.
    pub population: Arc<str>,
    /// Its intervened variables.
    pub interventions: Vec<VariableId>,
}

/// The accounting of a decision, stored with a derivation so a consumer can
/// replay the decision under the same limits and compare.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportSearchRecord {
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Effective memory cap in force, in bytes.
    pub memory_limit_bytes: u64,
    /// Operations charged by the decision.
    pub operations_consumed: usize,
    /// Deepest level charged.
    pub depth_reached: usize,
    /// `stage:<name>` of every stage run, up to and including the deciding one.
    pub stages: Vec<String>,
}

/// A checked positive derivation of a counterfactual transport query.
#[derive(Clone, Debug)]
pub struct CounterfactualTransportDerivation {
    /// The attested contract ([`COUNTERFACTUAL_TRANSPORT_CONTRACT`]).
    pub contract: &'static str,
    /// The model class the derivation was decided in.
    pub problem: CounterfactualTransportProblem,
    /// The canonical query.
    pub query: CounterfactualTransportQuery,
    /// The evidence catalog the derivation cites, as supplied.
    pub catalog: EvidenceCatalog,
    /// Population names: the target first, then sources in canonical order.
    pub populations: Vec<Arc<str>>,
    /// Derived laws; factors refer to them by index.
    pub items: Vec<CounterfactualItem>,
    /// `P(event and conditioning)` of the (possibly reduced) query.
    pub numerator: CounterfactualFunctional,
    /// `P(conditioning)` of the (possibly reduced) query; the constant one for
    /// an unconditional query.
    pub denominator: CounterfactualFunctional,
    /// `P(original conditioning)`, the positivity guard, when it is derivable and
    /// not already the denominator.
    pub positivity_guard: Option<CounterfactualFunctional>,
    /// Whether the original conditioning event's mass is checked at evaluation:
    /// `true` for an unconditional query, for a denominator that is the original
    /// conditioning mass, and when a guard was derived.
    pub conditioning_mass_checked: bool,
    /// Conditioning atoms moved into interventions by the conditional reduction.
    pub moved: Vec<CounterfactualMovedAtom>,
    /// The checked rule trace.
    pub rules: Vec<String>,
    /// Catalog regimes the formula cites, ascending.
    pub cited: Vec<CounterfactualCitedRegime>,
    /// Catalog regimes that supplied no usable law and why, ascending.
    pub ignored: Vec<(RegimeId, String)>,
    /// The success-path search receipt.
    pub search: CounterfactualTransportSearchRecord,
}

impl CounterfactualTransportDerivation {
    /// Canonical text of the whole derivation: contract, graph, query, items,
    /// functionals, reduction and receipt. Equal derivations render equal.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!(
            "{};{};{};populations=[",
            self.contract,
            self.problem.canonical_text(),
            self.query.canonical_text()
        );
        for population in &self.populations {
            let _ = write!(out, "{population},");
        }
        out.push_str("];items=[");
        for (index, item) in self.items.iter().enumerate() {
            let _ = write!(
                out,
                "{index}:p{}:{:#x}:{:?};",
                item.population, item.variables, item.source
            );
        }
        let _ = write!(out, "];numerator={:?};denominator={:?}", self.numerator, self.denominator);
        let _ = write!(out, ";guard={:?};moved={:?}", self.positivity_guard, self.moved);
        let _ = write!(out, ";checked={}", self.conditioning_mass_checked);
        for rule in &self.rules {
            let _ = write!(out, ";rule={rule}");
        }
        let _ = write!(out, ";search={:?}", self.search);
        out
    }

    /// Render the same canonical text while charging preparation and emitted bytes.
    ///
    /// # Errors
    /// Propagates a budget or cancellation error from `check`.
    pub fn try_canonical_text<E>(
        &self,
        mut check: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<String, E> {
        check(self.query.sources.len().saturating_mul(2))?;
        let mut source_ids = Vec::with_capacity(self.query.sources.len());
        for i in 0..self.query.sources.len() {
            if i % 512 == 0 {
                check(1)?;
            }
            source_ids.push(i);
        }
        let source_ids = checked_sort(&source_ids, &mut check, |a, b, check| {
            checked_text_cmp(
                &self.query.sources[*a].population,
                &self.query.sources[*b].population,
                check,
            )
        })?;
        let mut sets = Vec::with_capacity(source_ids.len());
        for &i in &source_ids {
            let source = &self.query.sources[i];
            let selected = checked_ids(&source.selection_targets, &mut check)?;
            let controlled = checked_ids(&source.controllable, &mut check)?;
            sets.push((selected, controlled));
        }
        let mut out = CheckedCanonical { text: String::new(), check: &mut check, error: None };
        let result = (|| -> std::fmt::Result {
            use std::fmt::Write as _;
            write!(out, "{};", self.contract)?;
            self.problem.write_canonical(&mut out)?;
            write!(out, ";counterfactual_transport_query_v1;target={};", self.query.target)?;
            self.query.event.write_canonical(&mut out)?;
            for (&i, (selected, controlled)) in source_ids.iter().zip(&sets) {
                write!(out, ";source={}:select=[", self.query.sources[i].population)?;
                for id in selected {
                    write!(out, "{},", id.raw())?;
                }
                out.write_str("];control=[")?;
                for id in controlled {
                    write!(out, "{},", id.raw())?;
                }
                out.write_char(']')?;
            }
            out.write_str(";populations=[")?;
            for population in &self.populations {
                write!(out, "{population},")?;
            }
            out.write_str("];items=[")?;
            for (index, item) in self.items.iter().enumerate() {
                write!(
                    out,
                    "{index}:p{}:{:#x}:{:?};",
                    item.population, item.variables, item.source
                )?;
            }
            write!(out, "];numerator={:?};denominator={:?}", self.numerator, self.denominator)?;
            write!(out, ";guard={:?};moved={:?}", self.positivity_guard, self.moved)?;
            write!(out, ";checked={}", self.conditioning_mass_checked)?;
            for rule in &self.rules {
                write!(out, ";rule={rule}")?;
            }
            write!(out, ";search={:?}", self.search)?;
            Ok(())
        })();
        if result.is_err() {
            return Err(out.error.expect("only the callback can fail the sink"));
        }
        (out.check)(0)?;
        Ok(out.text)
    }

    /// Replay the decision on the derivation's own inputs under its stored
    /// limits and require an identical derivation.
    ///
    /// # Errors
    ///
    /// `counterfactual_transport.invalid_derivation` when the replay does not
    /// identify or differs in any field; limits above the contract's maxima.
    pub fn replay_checked(&self, ctx: &ExecutionContext) -> Result<(), IdentificationError> {
        let limits = SearchLimits {
            operations: self.search.operations_limit,
            depth: self.search.depth_limit,
        };
        if self.search.memory_limit_bytes > COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES {
            return Err(bounds_exceeded());
        }
        let decision = decide_bounded(
            &self.problem,
            &self.query,
            &self.catalog,
            limits,
            self.search.memory_limit_bytes,
            ctx,
        )?;
        match decision {
            CounterfactualTransportDecision::Identified(replayed)
                if replayed.canonical_text() == self.canonical_text() =>
            {
                Ok(())
            }
            _ => Err(IdentificationError::invalid_derivation(INVALID_DERIVATION)),
        }
    }
}

/// A catalog regime a derivation would cite but the catalog lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualMissingRegime {
    /// Population the regime is a law of.
    pub population: Arc<str>,
    /// Variables the regime intervenes on (empty for an observational law).
    pub interventions: Vec<VariableId>,
}

/// A declared-but-absent evidence outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportMissing {
    /// Regimes whose laws, if supplied, license a derivation.
    pub missing: Vec<CounterfactualMissingRegime>,
    /// Stable description.
    pub detail: String,
}

/// A two-model witness that the declared evidence family does not determine the
/// query.
#[derive(Clone, Debug)]
pub struct CounterfactualTransportObstruction {
    /// The canonical query.
    pub query: CounterfactualTransportQuery,
    /// The model class the witness lives in.
    pub problem: CounterfactualTransportProblem,
    /// The catalog the family was built from.
    pub catalog: EvidenceCatalog,
    /// The portable witness.
    pub witness: CounterfactualTransportWitness,
    /// What the witness established when it was verified.
    pub check: CounterfactualTransportWitnessCheck,
    /// Why no derivation exists (the first failure of the declared-family search).
    pub reason: String,
}

impl CounterfactualTransportObstruction {
    /// Re-verify the witness on the obstruction's own inputs.
    ///
    /// # Errors
    ///
    /// `counterfactual_transport.invalid_witness` when the witness no longer
    /// verifies.
    pub fn verify(&self) -> Result<CounterfactualTransportWitnessCheck, IdentificationError> {
        verify_counterfactual_transport_witness(
            &self.problem,
            &self.query,
            &self.catalog,
            &self.witness,
        )
    }
}

/// What a search that stopped short of a formula or an obstruction explored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportInspection {
    /// Stable detail (`counterfactual_transport.factor_not_derivable`).
    pub detail: &'static str,
    /// The first failures of the derivation search, in order.
    pub failures: Vec<String>,
    /// Why no witness was found.
    pub witness: String,
    /// `stage:<name>` of every stage run.
    pub stages: Vec<String>,
}

/// A request outside the contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportUnsupported {
    /// Stable `counterfactual_transport.*` detail.
    pub detail: &'static str,
    /// What is outside the contract and what would bring it inside.
    pub message: String,
}

/// Outcome of a bounded counterfactual transport decision. Only
/// [`Self::ProvenNonTransportable`] is an impossibility claim.
#[derive(Clone, Debug)]
pub enum CounterfactualTransportDecision {
    /// A checked formula whose every cited factor binds to supplied evidence.
    Identified(Box<CounterfactualTransportDerivation>),
    /// Two verified finite models agreeing on the declared family and differing
    /// on the query.
    ProvenNonTransportable(Box<CounterfactualTransportObstruction>),
    /// A formula exists if the named declared regimes were supplied.
    MissingEvidence(Box<CounterfactualTransportMissing>),
    /// The request is outside the contract.
    UnsupportedScope(CounterfactualTransportUnsupported),
    /// The bounded search finished without a formula or a verified witness. This
    /// is not a non-transportability claim.
    NotCertified(CounterfactualTransportInspection),
    /// A limit or cancellation stopped the search; nothing is claimed.
    Exhausted(SearchReceipt),
}

impl CounterfactualTransportDecision {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`]). An out-of-contract request is
    /// reported as `not_certified` (no implemented rule applies).
    #[must_use]
    pub const fn identification_status(&self) -> TransportOutcomeKind {
        match self {
            Self::Identified(_) => TransportOutcomeKind::Identified,
            Self::ProvenNonTransportable(_) => TransportOutcomeKind::ProvenNonTransportable,
            Self::MissingEvidence(_) => TransportOutcomeKind::MissingEvidence,
            Self::UnsupportedScope(_) | Self::NotCertified(_) => TransportOutcomeKind::NotCertified,
            Self::Exhausted(_) => TransportOutcomeKind::BudgetCancel,
        }
    }

    /// Top-level reason code of a non-identified outcome; `None` when identified.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified(_) => None,
            Self::ProvenNonTransportable(_) => Some("transport_proven_non_transportable"),
            Self::MissingEvidence(_) => Some("transport_missing_evidence"),
            Self::UnsupportedScope(_) => Some("route_not_supported"),
            Self::NotCertified(_) => Some("transport_not_certified"),
            Self::Exhausted(_) => Some("transport_budget_cancel"),
        }
    }

    /// Stable `counterfactual_transport.*` detail of a non-identified outcome.
    #[must_use]
    pub fn detail_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified(_) => None,
            Self::ProvenNonTransportable(_) => Some(CHECKED_OBSTRUCTION),
            Self::MissingEvidence(_) => Some(MISSING_REGIME),
            Self::UnsupportedScope(unsupported) => Some(unsupported.detail),
            Self::NotCertified(inspection) => Some(inspection.detail),
            Self::Exhausted(_) => Some(BUDGET),
        }
    }

    /// The derivation, when identified.
    #[must_use]
    pub fn derivation(&self) -> Option<&CounterfactualTransportDerivation> {
        match self {
            Self::Identified(derivation) => Some(derivation),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation.
// ---------------------------------------------------------------------------

/// A usable catalog regime in internal indices.
#[derive(Clone, Copy, Debug)]
struct UsableRegime {
    id: RegimeId,
    population: usize,
    intervened: u32,
    measured: u32,
}

/// A query that passed the contract.
struct Validated {
    query: CounterfactualTransportQuery,
    populations: Vec<Arc<str>>,
    selected: Vec<u32>,
    controllable: Vec<u32>,
    worlds: Vec<WorldIdx>,
    event: Vec<(usize, usize, usize)>,
    given: Vec<(usize, usize, usize)>,
    usable: Vec<UsableRegime>,
    ignored: Vec<(RegimeId, String)>,
    /// `(population, intervened set)` of supplied population laws the closure
    /// cannot use (a witness must still agree on them).
    unused_family: Vec<(usize, u32)>,
}

/// Why a request does not reach the search.
enum Rejection {
    Invalid(IdentificationError),
    Unsupported(&'static str, String),
}

impl From<IdentificationError> for Rejection {
    fn from(error: IdentificationError) -> Self {
        Self::Invalid(error)
    }
}

fn atoms_map(atoms: &[(usize, usize, usize)]) -> Option<BTreeMap<(usize, usize), usize>> {
    let mut map = BTreeMap::new();
    for &(world, variable, level) in atoms {
        match map.insert((world, variable), level) {
            Some(existing) if existing != level => return None,
            _ => {}
        }
    }
    Some(map)
}

#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent request, catalog and bound validity checks, each with its own typed rejection"
)]
fn validate(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<Validated, Rejection> {
    let query = query.canonical();
    let event = &query.event;
    if event.coupling() != ExogenousCoupling::SharedLatentExogenous {
        return Err(Rejection::Unsupported(
            COUPLING_UNSUPPORTED,
            "counterfactual transport shares the exogenous terms of the semi-Markovian model \
             across worlds and marginalizes them; an abduced per-unit coupling is a different \
             route"
                .to_owned(),
        ));
    }
    event.validate().map_err(invalid_query)?;
    if query.target.trim().is_empty() {
        return Err(invalid_query("the target population must be named").into());
    }
    let mut populations: Vec<Arc<str>> = vec![Arc::clone(&query.target)];
    for source in query.sources.iter() {
        if populations.iter().any(|p| p.as_ref() == source.population.as_ref()) {
            return Err(invalid_query(
                "source populations must be distinct and none may be the target",
            )
            .into());
        }
        populations.push(Arc::clone(&source.population));
    }
    let mask_of = |variables: &[VariableId], what: &str| -> Result<u32, Rejection> {
        let mut mask = 0u32;
        for v in variables {
            let Some(i) = problem.index_of(*v) else {
                return Err(
                    invalid_query(format!("{what} names unknown variable {}", v.raw())).into()
                );
            };
            mask |= 1 << i;
        }
        Ok(mask)
    };
    let mut selected = vec![0u32];
    let mut controllable = vec![0u32];
    for source in query.sources.iter() {
        selected.push(mask_of(&source.selection_targets, "a selection diagram")?);
        let control = mask_of(&source.controllable, "a controllable set")?;
        controllable.push(control);
    }
    // Worlds.
    let mut worlds: Vec<WorldIdx> = Vec::with_capacity(event.worlds().len());
    for (index, spec) in event.worlds().iter().enumerate() {
        let mut world = WorldIdx::default();
        for (variable, value) in spec.interventions() {
            let Some(i) = problem.index_of(variable) else {
                return Err(invalid_query(format!(
                    "world {index} intervenes on unknown variable {}",
                    variable.raw()
                ))
                .into());
            };
            let Some(level) = problem.level_index(i, value) else {
                return Err(invalid_query(format!(
                    "world {index} sets variable {} to {value}, which is not a declared level",
                    variable.raw()
                ))
                .into());
            };
            world.fixed.insert(i, level);
        }
        for route in spec.routes() {
            let (Some(parent), Some(child)) =
                (problem.index_of(route.parent), problem.index_of(route.child))
            else {
                return Err(
                    invalid_query(format!("world {index} routes an unknown variable")).into()
                );
            };
            if !problem.parents[child].contains(&parent) {
                return Err(invalid_query(format!(
                    "world {index} routes {} -> {}, which is not an edge of the graph",
                    route.parent.raw(),
                    route.child.raw()
                ))
                .into());
            }
            let source = route.source.index();
            if source >= event.worlds().len() || source == index {
                return Err(invalid_query(format!(
                    "world {index} reads a parent from an unknown world or from itself"
                ))
                .into());
            }
            world.routes.insert((parent, child), source);
        }
        worlds.push(world);
    }
    // Atoms.
    let to_atoms = |atoms: &[antecedent_core::CounterfactualEvent]| -> Result<Vec<(usize, usize, usize)>, Rejection> {
        atoms
            .iter()
            .map(|a| -> Result<(usize, usize, usize), Rejection> {
                let Some(i) = problem.index_of(a.variable()) else {
                    return Err(invalid_query(format!(
                        "an event names unknown variable {}",
                        a.variable().raw()
                    ))
                    .into());
                };
                let Some(level) = problem.level_index(i, a.level()) else {
                    return Err(invalid_query(format!(
                        "variable {} is read at {}, which is not a declared level",
                        a.variable().raw(),
                        a.level()
                    ))
                    .into());
                };
                Ok((a.world().index(), i, level))
            })
            .collect()
    };
    let event_atoms = to_atoms(event.event())?;
    let given_atoms = to_atoms(event.given())?;
    if atoms_map(&given_atoms).is_none() {
        return Err(invalid_query(
            "the conditioning event reads one variable of one world at two levels: it has \
             probability zero in every model",
        )
        .into());
    }
    if let Some(given_map) = atoms_map(&given_atoms) {
        if !given_map.is_empty() {
            let impossible = matches!(
                build_network(problem, &worlds, &given_map).map_err(Rejection::Invalid)?,
                Built::Zero
            );
            if impossible {
                return Err(invalid_query(
                    "the conditioning event contradicts the query's own interventions: it has \
                     probability zero in every model",
                )
                .into());
            }
        }
    }
    // Catalog.
    catalog.validate().map_err(invalid_catalog)?;
    let mut usable: Vec<UsableRegime> = Vec::new();
    let mut ignored: Vec<(RegimeId, String)> = Vec::new();
    let mut unused_family: Vec<(usize, u32)> = Vec::new();
    for regime in catalog.regimes.iter() {
        if matches!(regime.selection, antecedent_core::SamplingSelection::SelectedOn { .. }) {
            return Err(Rejection::Unsupported(
                SELECTED_SAMPLE,
                format!(
                    "regime {} is the law of a selected subpopulation; this contract has no \
                     model of sampling selection, so it cannot be used or ignored soundly",
                    regime.id.raw()
                ),
            ));
        }
        let Some(population) =
            populations.iter().position(|p| p.as_ref() == regime.population.as_ref())
        else {
            return Err(invalid_catalog(format!(
                "regime {} belongs to population {}, which is neither the target nor a declared \
                 source",
                regime.id.raw(),
                regime.population
            ))
            .into());
        };
        let mut intervened = 0u32;
        let mut measured = 0u32;
        for v in regime.interventions.iter() {
            let Some(i) = problem.index_of(*v) else {
                return Err(invalid_catalog(format!(
                    "regime {} intervenes on unknown variable {}",
                    regime.id.raw(),
                    v.raw()
                ))
                .into());
            };
            intervened |= 1 << i;
        }
        for v in regime.measured.iter() {
            let Some(i) = problem.index_of(*v) else {
                return Err(invalid_catalog(format!(
                    "regime {} measures unknown variable {}",
                    regime.id.raw(),
                    v.raw()
                ))
                .into());
            };
            if (intervened >> i) & 1 == 0 {
                measured |= 1 << i;
            }
        }
        if !regime.supplies_population_law() {
            ignored.push((regime.id, "not a measured population law (no results, a model artifact, or a selected sample)".to_owned()));
            continue;
        }
        let needed = problem.mask_variables(measured);
        let interventions = problem.mask_variables(intervened);
        let need = antecedent_core::FactorNeed {
            population: regime.population.as_ref(),
            variables: &needed,
            conditioned_on: &[],
            interventions: &interventions,
        };
        if measured == 0
            || !regime.satisfies(&need)
            || !regime_measured_is_factor(problem, intervened, measured)
        {
            ignored.push((
                regime.id,
                "not a joint law over a variable set closed under ancestors of the \
                 non-intervened variables with an unrestricted intervention domain"
                    .to_owned(),
            ));
            unused_family.push((population, intervened));
            continue;
        }
        usable.push(UsableRegime { id: regime.id, population, intervened, measured });
    }
    usable.sort_by_key(|u| (u.population, u.intervened, u.measured, u.id));
    ignored.sort_by_key(|(id, _)| *id);
    unused_family.sort_unstable();
    unused_family.dedup();
    Ok(Validated {
        query,
        populations,
        selected,
        controllable,
        worlds,
        event: event_atoms,
        given: given_atoms,
        usable,
        ignored,
        unused_family,
    })
}

// ---------------------------------------------------------------------------
// Search state and the derivation of one conjunction.
// ---------------------------------------------------------------------------

/// Why the derivation search stopped without an answer.
enum Abort {
    Stop(SearchStop),
    Error(IdentificationError),
}

impl From<SearchStop> for Abort {
    fn from(stop: SearchStop) -> Self {
        Self::Stop(stop)
    }
}

impl From<IdentificationError> for Abort {
    fn from(error: IdentificationError) -> Self {
        Self::Error(error)
    }
}

struct Search<'c> {
    budget: SearchBudget<'c>,
    stop: Option<SearchStop>,
    stages: Vec<String>,
    failures: Vec<String>,
}

impl Search<'_> {
    fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), SearchStop> {
        self.budget.charge(depth, bytes).inspect_err(|stop| {
            self.stop.get_or_insert(*stop);
        })
    }

    /// Charge each item or enumerated leaf once; callback counts describe local enumeration
    /// position only, while the shared budget and receipt accumulate across all enumerations.
    fn work(
        &mut self,
        depth: usize,
        bytes: u64,
        _enumerated: Option<u64>,
    ) -> Result<(), SearchStop> {
        self.charge(depth, bytes)
    }

    /// Quote the summation over `cardinalities` from the declared exact-enumeration charge and
    /// check it against what the caller's budget still allows: the cells are the work, one
    /// binary64 per cell is the storage. The storage is leased from the context's pool and
    /// returned at once, which proves the pool can fund it before any evaluator allocates the
    /// table. Nothing is allocated here.
    fn admit_summation(
        &mut self,
        cardinalities: impl Iterator<Item = u64>,
    ) -> Result<(), SearchStop> {
        let mut enumeration = FunctionalEnumeration::new();
        for cardinality in cardinalities {
            enumeration = enumeration.factor(cardinality);
        }
        let estimate = match enumeration.estimate() {
            Ok(estimate) => estimate,
            // No summed symbol is one empty product term.
            Err(SearchBudgetError::MissingFactors) => return Ok(()),
            Err(_) => {
                self.stop.get_or_insert(SearchStop::Memory);
                return Err(SearchStop::Memory);
            }
        };
        let stop = if estimate.work()
            > u64::try_from(self.budget.operations_remaining()).unwrap_or(u64::MAX)
        {
            Some(SearchStop::Operations)
        } else {
            self.budget.lease_storage(estimate.memory_bytes()).err()
        };
        match stop {
            Some(stop) => {
                self.stop.get_or_insert(stop);
                Err(stop)
            }
            None => Ok(()),
        }
    }

    fn stage(&mut self, name: &str) {
        self.stages.push(format!("stage:{name}"));
    }

    fn note(&mut self, failure: String) {
        if self.failures.len() < 8 && !self.failures.contains(&failure) {
            self.failures.push(failure);
        }
    }
}

struct DeriveCtx<'a> {
    problem: &'a CounterfactualTransportProblem,
    closure: &'a Closure,
    /// Nesting of the binary consistency complement allowed in the current attempt.
    complement_depth: std::cell::Cell<usize>,
    /// Some complement was left untried because of `complement_depth`.
    complement_pruned: std::cell::Cell<bool>,
}

impl<'a> DeriveCtx<'a> {
    fn new(problem: &'a CounterfactualTransportProblem, closure: &'a Closure) -> Self {
        Self {
            problem,
            closure,
            complement_depth: std::cell::Cell::new(INITIAL_COMPLEMENT_DEPTH),
            complement_pruned: std::cell::Cell::new(false),
        }
    }
}

fn mask_text(problem: &CounterfactualTransportProblem, mask: u32) -> String {
    let names: Vec<String> =
        problem.mask_variables(mask).iter().map(|v| v.raw().to_string()).collect();
    format!("{{{}}}", names.join(","))
}

fn lift_level(level: Level) -> CounterfactualLevel {
    match level {
        Level::Fixed(l) => CounterfactualLevel::Fixed(CounterfactualLevelIndex::new(l)),
        Level::Symbol(s) => CounterfactualLevel::Symbol(CounterfactualSymbolId::new(s)),
    }
}

/// Derive the probability of one conjunction from the closure, trying the direct
/// factorization first and then the binary consistency complement.
fn derive_conjunction(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    worlds: &[WorldIdx],
    atoms: &BTreeMap<(usize, usize), usize>,
    depth: usize,
    used: &BTreeSet<(usize, usize)>,
) -> Result<Option<CounterfactualFunctional>, Abort> {
    let problem = ctx.problem;
    search.charge(depth, (atoms.len() as u64).saturating_mul(256))?;
    let net = match build_network(problem, worlds, atoms)? {
        Built::Zero => return Ok(Some(CounterfactualFunctional::Zero)),
        Built::Net(net) => net,
    };
    match factorize(problem, &net) {
        Factored::Terms { symbols, terms } => {
            let mut factors: Vec<CounterfactualFactor> = Vec::with_capacity(terms.len());
            let mut missing: Option<u32> = None;
            for term in &terms {
                if let Some(item) = ctx.closure.target_item(term.mask) {
                    factors.push(CounterfactualFactor {
                        item,
                        outputs: term
                            .outputs
                            .iter()
                            .map(|&(v, level)| {
                                (CounterfactualVariableIndex::new(v), lift_level(level))
                            })
                            .collect(),
                        parents: term
                            .parents
                            .iter()
                            .map(|&(v, level)| {
                                (CounterfactualVariableIndex::new(v), lift_level(level))
                            })
                            .collect(),
                    });
                } else {
                    missing = Some(term.mask);
                    break;
                }
            }
            match missing {
                None => {
                    search.admit_summation(
                        symbols.iter().map(|&(_, v)| problem.levels[v].len() as u64),
                    )?;
                    return Ok(Some(CounterfactualFunctional::Sum(CounterfactualSum {
                        symbols: symbols
                            .iter()
                            .map(|&(id, variable)| CounterfactualSymbol {
                                id: CounterfactualSymbolId::new(id),
                                variable: CounterfactualVariableIndex::new(variable),
                            })
                            .collect(),
                        factors,
                    })));
                }
                Some(mask) => search.note(format!(
                    "the single-world factor over variables {} is not derivable from the \
                     evidence",
                    mask_text(problem, mask)
                )),
            }
        }
        Factored::Conflict(Conflict::Copies { variable }) => search.note(format!(
            "variable {} appears with two different readings in one district: the joint of \
             its worlds is a cross-world law",
            problem.variables[variable].raw()
        )),
        Factored::Conflict(Conflict::Inputs { variable }) => search.note(format!(
            "variable {} is read at two values by one district: the factor is not a \
             single-world interventional law",
            problem.variables[variable].raw()
        )),
    }
    {
        for (&(world, variable), &level) in atoms {
            if problem.levels[variable].len() != 2
                || used.contains(&(world, variable))
                || worlds[world].fixed.contains_key(&variable)
            {
                continue;
            }
            if depth >= ctx.complement_depth.get() {
                ctx.complement_pruned.set(true);
                continue;
            }
            let mut whole = atoms.clone();
            whole.remove(&(world, variable));
            let mut flipped = atoms.clone();
            flipped.insert((world, variable), 1 - level);
            let mut used_next = used.clone();
            used_next.insert((world, variable));
            let Some(a) = derive_conjunction(ctx, search, worlds, &whole, depth + 1, &used_next)?
            else {
                continue;
            };
            let Some(b) = derive_conjunction(ctx, search, worlds, &flipped, depth + 1, &used_next)?
            else {
                continue;
            };
            return Ok(Some(CounterfactualFunctional::Complement {
                variable: CounterfactualVariableIndex::new(variable),
                whole: Box::new(a),
                flipped: Box::new(b),
            }));
        }
    }
    Ok(None)
}

/// The derived plan of a whole query.
struct QueryPlan {
    numerator: CounterfactualFunctional,
    denominator: CounterfactualFunctional,
    guard: Option<CounterfactualFunctional>,
    checked: bool,
    moved: Vec<CounterfactualMovedAtom>,
    rules: Vec<String>,
}

fn one() -> CounterfactualFunctional {
    CounterfactualFunctional::Sum(CounterfactualSum { symbols: Vec::new(), factors: Vec::new() })
}

fn merged(
    event: &BTreeMap<(usize, usize), usize>,
    given: &BTreeMap<(usize, usize), usize>,
) -> Option<BTreeMap<(usize, usize), usize>> {
    let mut out = event.clone();
    for (&key, &level) in given {
        match out.insert(key, level) {
            Some(existing) if existing != level => return None,
            _ => {}
        }
    }
    Some(out)
}

/// A reduced conditional query: worlds, event, and the atoms moved into interventions.
type ReducedConditioning =
    (Vec<WorldIdx>, BTreeMap<(usize, usize), usize>, Vec<CounterfactualMovedAtom>);

/// The conditional reduction: repeatedly move a conditioning atom into an
/// intervention (in every world its node appears in) when the event's nodes are
/// m-separated from it given the other conditioning nodes.
fn reduce_conditioning(
    problem: &CounterfactualTransportProblem,
    worlds: &[WorldIdx],
    event: &BTreeMap<(usize, usize), usize>,
    given: &BTreeMap<(usize, usize), usize>,
) -> Result<ReducedConditioning, IdentificationError> {
    let mut worlds: Vec<WorldIdx> = worlds.to_vec();
    let mut given = given.clone();
    let mut moved: Vec<CounterfactualMovedAtom> = Vec::new();
    loop {
        if given.is_empty() {
            break;
        }
        let Some(all) = merged(event, &given) else { break };
        let Built::Net(net) = build_network(problem, &worlds, &all)? else {
            break;
        };
        let event_nodes: Vec<usize> = event
            .keys()
            .filter_map(|&(w, v)| net.pair_node.get(w).and_then(|row| row[v]))
            .collect();
        if event_nodes.is_empty() {
            break;
        }
        let mut chosen: Option<((usize, usize), usize)> = None;
        for (&(world, variable), &level) in &given {
            let Some(node) = net.pair_node.get(world).and_then(|row| row[variable]) else {
                continue;
            };
            if event_nodes.contains(&node) {
                continue;
            }
            let others: Vec<usize> = given
                .keys()
                .filter(|&&key| key != (world, variable))
                .filter_map(|&(w, v)| net.pair_node.get(w).and_then(|row| row[v]))
                .filter(|&n| n != node)
                .collect();
            if others.iter().any(|n| event_nodes.contains(n)) {
                continue;
            }
            if separated(problem, &net, &event_nodes, &[node], &others) {
                chosen = Some(((world, variable), level));
                break;
            }
        }
        let Some(((world, variable), level)) = chosen else { break };
        let node = net.pair_node[world][variable];
        for (w, row) in net.pair_node.iter().enumerate() {
            for (v, id) in row.iter().enumerate() {
                if *id == node && node.is_some() {
                    worlds[w].fixed.insert(v, level);
                    moved.push(CounterfactualMovedAtom {
                        world: w,
                        variable: CounterfactualVariableIndex::new(v),
                        level: CounterfactualLevelIndex::new(level),
                    });
                }
            }
        }
        given.remove(&(world, variable));
        // Remaining atoms that now sit on an intervened pair are consumed by the
        // network build (they are true or contradictory).
        given.retain(|&(w, v), l| worlds[w].fixed.get(&v).is_none_or(|fixed| *fixed != *l));
    }
    moved.sort_by_key(|m| (m.world, m.variable));
    moved.dedup();
    Ok((worlds, given, moved))
}

/// [`derive_query_within`] with the complement nesting deepened on demand: a failure that cut
/// off a complement is retried one level deeper, so the nesting is bounded by the search
/// budget (an `Exhausted` stop) and never by a constant.
fn derive_query(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    validated: &Validated,
) -> Result<Option<QueryPlan>, Abort> {
    ctx.complement_depth.set(INITIAL_COMPLEMENT_DEPTH);
    loop {
        ctx.complement_pruned.set(false);
        let plan = derive_query_within(ctx, search, validated)?;
        if plan.is_some() || !ctx.complement_pruned.get() {
            return Ok(plan);
        }
        ctx.complement_depth.set(ctx.complement_depth.get() + 1);
    }
}

fn derive_query_within(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    validated: &Validated,
) -> Result<Option<QueryPlan>, Abort> {
    let problem = ctx.problem;
    let event = atoms_map(&validated.event);
    let given = atoms_map(&validated.given)
        .ok_or_else(|| Abort::Error(invalid_query("the conditioning event is contradictory")))?;
    let Some(event) = event else {
        // The event contradicts itself: probability zero given any possible conditioning.
        return Ok(Some(QueryPlan {
            numerator: CounterfactualFunctional::Zero,
            denominator: one(),
            guard: None,
            checked: given.is_empty(),
            moved: Vec::new(),
            rules: vec!["event.contradiction".to_owned()],
        }));
    };
    let used = BTreeSet::new();
    // Variant A: reduced conditioning (only when something moves).
    if !given.is_empty() {
        let (worlds, reduced_given, moved) =
            reduce_conditioning(problem, &validated.worlds, &event, &given)?;
        if !moved.is_empty() {
            let numerator_atoms = merged(&event, &reduced_given);
            let numerator = match numerator_atoms {
                Some(atoms) => derive_conjunction(ctx, search, &worlds, &atoms, 0, &used)?,
                None => Some(CounterfactualFunctional::Zero),
            };
            let denominator = if reduced_given.is_empty() {
                Some(one())
            } else {
                derive_conjunction(ctx, search, &worlds, &reduced_given, 0, &used)?
            };
            if let (Some(numerator), Some(denominator)) = (numerator, denominator) {
                let guard = derive_conjunction(ctx, search, &validated.worlds, &given, 0, &used)?;
                let checked = guard.is_some();
                return Ok(Some(QueryPlan {
                    numerator,
                    denominator,
                    guard,
                    checked,
                    moved,
                    rules: vec!["conditioning.reduction".to_owned()],
                }));
            }
        }
    }
    // Variant B: the ratio of the two original conjunctions.
    let numerator = match merged(&event, &given) {
        Some(atoms) => derive_conjunction(ctx, search, &validated.worlds, &atoms, 0, &used)?,
        None => Some(CounterfactualFunctional::Zero),
    };
    let Some(numerator) = numerator else {
        return Ok(None);
    };
    let denominator = if given.is_empty() {
        one()
    } else {
        let Some(den) = derive_conjunction(ctx, search, &validated.worlds, &given, 0, &used)?
        else {
            return Ok(None);
        };
        den
    };
    Ok(Some(QueryPlan {
        numerator,
        denominator,
        guard: None,
        checked: true,
        moved: Vec::new(),
        rules: vec!["conditioning.ratio".to_owned()],
    }))
}

fn collect_items(functional: &CounterfactualFunctional, out: &mut Vec<usize>) {
    match functional {
        CounterfactualFunctional::Zero => {}
        CounterfactualFunctional::Sum(sum) => out.extend(sum.factors.iter().map(|f| f.item)),
        CounterfactualFunctional::Complement { whole, flipped, .. } => {
            collect_items(whole, out);
            collect_items(flipped, out);
        }
    }
}

fn remap_items(functional: &mut CounterfactualFunctional, map: &BTreeMap<usize, usize>) {
    match functional {
        CounterfactualFunctional::Zero => {}
        CounterfactualFunctional::Sum(sum) => {
            for factor in &mut sum.factors {
                if let Some(&new) = map.get(&factor.item) {
                    factor.item = new;
                }
            }
        }
        CounterfactualFunctional::Complement { whole, flipped, .. } => {
            remap_items(whole, map);
            remap_items(flipped, map);
        }
    }
}

// ---------------------------------------------------------------------------
// The decision.
// ---------------------------------------------------------------------------

/// Decide a counterfactual transport query on `problem` from the labelled
/// evidence of `catalog`.
///
/// Stages run in a fixed order, each only when the earlier ones do not decide:
/// the derivation over the supplied regimes; the derivation over the declared
/// family (supplied regimes plus every population's observational law and every
/// source's experiments on subsets of its controllable set), whose success is
/// [`CounterfactualTransportDecision::MissingEvidence`]; and the bounded witness
/// search over the declared family, whose verified success is
/// [`CounterfactualTransportDecision::ProvenNonTransportable`]. One
/// [`SearchBudget`] is charged by every stage, so the limits bound the whole
/// decision; a stop anywhere is [`CounterfactualTransportDecision::Exhausted`],
/// never a verdict.
///
/// # Errors
///
/// Limits above [`COUNTERFACTUAL_TRANSPORT_MAX_LIMITS`]
/// (`counterfactual_transport.bounds_exceeded`), a malformed query or catalog
/// (`counterfactual_transport.invalid_query`,
/// `counterfactual_transport.invalid_catalog`), or a derivation that fails its own
/// check.
pub fn identify_counterfactual_transport(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportDecision, IdentificationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::Identification,
        || {
            crate::execution_counts::note_check();
            decide_bounded(
                problem,
                query,
                catalog,
                limits,
                COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
                ctx,
            )
        },
    )
}

/// Re-run a decision under exactly the limits a stored search record names:
/// its operation and depth limits and its effective memory cap. A consumer of a
/// stored derivation uses this to reproduce the decision (the receipt is part of
/// the derivation's canonical text, so the limits in force must be the stored
/// ones, not the replayer's defaults).
///
/// # Errors
///
/// As [`identify_counterfactual_transport`]; a stored memory cap above
/// [`COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES`] is `counterfactual_transport.bounds_exceeded`.
// doc(hidden): replay primitive of the artifact consumer, not a routed entry.
#[doc(hidden)]
pub fn replay_counterfactual_transport(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    record: &CounterfactualTransportSearchRecord,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportDecision, IdentificationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::Identification,
        || {
            crate::execution_counts::note_check();
            decide_bounded(
                problem,
                query,
                catalog,
                SearchLimits { operations: record.operations_limit, depth: record.depth_limit },
                record.memory_limit_bytes,
                ctx,
            )
        },
    )
}

fn decide_bounded(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    memory_limit_bytes: u64,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportDecision, IdentificationError> {
    if limits.operations > COUNTERFACTUAL_TRANSPORT_MAX_LIMITS.operations
        || limits.depth > COUNTERFACTUAL_TRANSPORT_MAX_LIMITS.depth
        || memory_limit_bytes > COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES
    {
        return Err(bounds_exceeded());
    }
    let validated = match validate(problem, query, catalog) {
        Ok(v) => v,
        Err(Rejection::Invalid(error)) => return Err(error),
        Err(Rejection::Unsupported(detail, message)) => {
            return Ok(CounterfactualTransportDecision::UnsupportedScope(
                CounterfactualTransportUnsupported { detail, message },
            ));
        }
    };
    let budget = match SearchBudget::with_memory(limits, memory_limit_bytes, ctx) {
        Ok(budget) => budget,
        Err(receipt) => return Ok(CounterfactualTransportDecision::Exhausted(receipt)),
    };
    let mut search = Search { budget, stop: None, stages: Vec::new(), failures: Vec::new() };
    match run_stages(problem, catalog, &validated, limits, &mut search) {
        Ok(decision) => Ok(decision),
        Err(Abort::Stop(stop)) => {
            let stop = search.stop.unwrap_or(stop);
            Ok(CounterfactualTransportDecision::Exhausted(search.budget.receipt(
                stop,
                search.stages.clone(),
                Vec::new(),
            )))
        }
        Err(Abort::Error(IdentificationError::UnsupportedInput { code }))
            if code == BOUNDS_EXCEEDED =>
        {
            Ok(CounterfactualTransportDecision::UnsupportedScope(
                CounterfactualTransportUnsupported {
                    detail: BOUNDS_EXCEEDED,
                    message: "a derivation exceeded a declared bound (network nodes or summation \
                              cells)"
                        .to_owned(),
                },
            ))
        }
        Err(Abort::Error(error)) => Err(error),
    }
}

/// Seeds of the supplied regimes.
fn supplied_seeds(validated: &Validated) -> Vec<Seed> {
    let mut seeds: Vec<Seed> = validated
        .usable
        .iter()
        .map(|u| Seed {
            population: u.population,
            intervened: u.intervened,
            measured: u.measured,
            regime: Some(u.id),
        })
        .collect();
    seeds.sort_by_key(|s| (s.population, s.intervened, s.measured, s.regime));
    seeds
}

/// Seeds of the declared family not supplied in full: every population's
/// observational law and every source's experiments on subsets of its
/// controllable set, each measuring every non-intervened variable. Every subset visited is
/// metered against the search (see [`Search::work`]), so a controllable set the budget cannot
/// enumerate stops the decision while the receipt keeps counting charged operations only.
fn candidate_seeds(
    validated: &Validated,
    problem: &CounterfactualTransportProblem,
    charge: &mut dyn FnMut(usize, u64, Option<u64>) -> Result<(), SearchStop>,
) -> Result<Vec<Seed>, SearchStop> {
    let full = problem.full_mask();
    let mut seeds: Vec<Seed> = Vec::new();
    let mut visited: u64 = 0;
    for population in 0..validated.populations.len() {
        let control = validated.controllable[population];
        // Every subset of the controllable set, the empty one included.
        let mut subset = control;
        loop {
            visited += 1;
            charge(0, (seeds.len() as u64 + 1).saturating_mul(32), Some(visited))?;
            let supplied = validated.usable.iter().any(|u| {
                u.population == population && u.intervened == subset && u.measured == full & !subset
            });
            if !supplied {
                seeds.push(Seed {
                    population,
                    intervened: subset,
                    measured: full & !subset,
                    regime: None,
                });
            }
            if subset == 0 {
                break;
            }
            subset = subset.wrapping_sub(1) & control;
        }
    }
    seeds.sort_by_key(|s| (s.population, s.intervened, s.measured, s.regime));
    Ok(seeds)
}

#[allow(
    clippy::too_many_lines,
    reason = "the staged pipeline: one flat block per stage, each charging the budget and recording its outcome"
)]
fn run_stages(
    problem: &CounterfactualTransportProblem,
    catalog: &EvidenceCatalog,
    validated: &Validated,
    limits: SearchLimits,
    search: &mut Search<'_>,
) -> Result<CounterfactualTransportDecision, Abort> {
    // Stage 1: supplied regimes only.
    let supplied = supplied_seeds(validated);
    let closure =
        close(problem, &validated.selected, &supplied, &[], &mut |depth, bytes, leaves| {
            search.work(depth, bytes, leaves)
        })?;
    search.stage("supplied_regimes");
    let ctx = DeriveCtx::new(problem, &closure);
    if let Some(plan) = derive_query(&ctx, search, validated)? {
        let derivation = assemble(problem, catalog, validated, &closure, plan, limits, search);
        return Ok(CounterfactualTransportDecision::Identified(Box::new(derivation)));
    }
    // Stage 2: the declared family.
    let candidates = candidate_seeds(validated, problem, &mut |depth, bytes, leaves| {
        search.work(depth, bytes, leaves)
    })?;
    let declared_closure = close(
        problem,
        &validated.selected,
        &supplied,
        &candidates,
        &mut |depth, bytes, leaves| search.work(depth, bytes, leaves),
    )?;
    search.stage("declared_family");
    let declared_ctx = DeriveCtx::new(problem, &declared_closure);
    if let Some(plan) = derive_query(&declared_ctx, search, validated)? {
        let mut roots = Vec::new();
        collect_items(&plan.numerator, &mut roots);
        collect_items(&plan.denominator, &mut roots);
        if let Some(guard) = &plan.guard {
            collect_items(guard, &mut roots);
        }
        let mut missing: Vec<CounterfactualMissingRegime> = Vec::new();
        for index in reachable(&declared_closure, &roots) {
            let item = declared_closure.items[index];
            if let ItemSource::Regime { regime: None, intervened } = item.source {
                missing.push(CounterfactualMissingRegime {
                    population: Arc::clone(&validated.populations[item.population]),
                    interventions: problem.mask_variables(intervened),
                });
            }
        }
        missing.sort_by(|a, b| {
            (a.population.as_ref(), &a.interventions)
                .cmp(&(b.population.as_ref(), &b.interventions))
        });
        missing.dedup();
        let detail = format!(
            "the query is derivable if the law of {} were supplied",
            missing
                .iter()
                .map(|m| {
                    let controls: Vec<String> =
                        m.interventions.iter().map(|v| v.raw().to_string()).collect();
                    if controls.is_empty() {
                        format!("the observational regime of {}", m.population)
                    } else {
                        format!("do({}) in {}", controls.join(","), m.population)
                    }
                })
                .collect::<Vec<_>>()
                .join("; ")
        );
        return Ok(CounterfactualTransportDecision::MissingEvidence(Box::new(
            CounterfactualTransportMissing { missing, detail },
        )));
    }
    // Stage 3: the witness search over the declared family.
    search.stage("witness_search");
    let mut family: Vec<(usize, u32)> = supplied
        .iter()
        .chain(&candidates)
        .map(|s| (s.population, s.intervened))
        .chain(validated.unused_family.iter().copied())
        .collect();
    family.sort_unstable();
    family.dedup();
    let event = atoms_map(&validated.event);
    let given = atoms_map(&validated.given);
    let (Some(event), Some(given)) = (event, given) else {
        return Ok(not_certified(search, "the event contradicts itself".to_owned()));
    };
    let Some(numerator) = merged(&event, &given) else {
        return Ok(not_certified(search, "the event contradicts the conditioning".to_owned()));
    };
    let wp = WitnessProblem {
        problem,
        selected: &validated.selected,
        family: &family,
        worlds: &validated.worlds,
        numerator: &numerator,
        given: &given,
        simulations: u64::try_from(search.budget.operations_remaining())
            .unwrap_or(u64::MAX)
            .saturating_mul(witness::SIMULATIONS_PER_OPERATION),
    };
    match search_witness(&wp, &mut |depth, bytes, leaves| search.work(depth, bytes, leaves))? {
        WitnessOutcome::Found(witness) => {
            let check = verify_witness(&wp, &witness).map_err(|_| {
                Abort::Error(IdentificationError::invalid_derivation(INVALID_WITNESS))
            })?;
            Ok(CounterfactualTransportDecision::ProvenNonTransportable(Box::new(
                CounterfactualTransportObstruction {
                    query: validated.query.clone(),
                    problem: problem.clone(),
                    catalog: catalog.clone(),
                    witness,
                    check,
                    reason: search
                        .failures
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "no derivation from the declared evidence".to_owned()),
                },
            )))
        }
        WitnessOutcome::NotFound(reason) => Ok(not_certified(search, reason)),
    }
}

fn not_certified(search: &mut Search<'_>, witness: String) -> CounterfactualTransportDecision {
    CounterfactualTransportDecision::NotCertified(CounterfactualTransportInspection {
        detail: NO_FACTOR_DERIVATION,
        failures: search.failures.clone(),
        witness,
        stages: search.stages.clone(),
    })
}

fn assemble(
    problem: &CounterfactualTransportProblem,
    catalog: &EvidenceCatalog,
    validated: &Validated,
    closure: &Closure,
    plan: QueryPlan,
    limits: SearchLimits,
    search: &Search<'_>,
) -> CounterfactualTransportDerivation {
    let QueryPlan { mut numerator, mut denominator, mut guard, checked, moved, rules } = plan;
    let mut roots = Vec::new();
    collect_items(&numerator, &mut roots);
    collect_items(&denominator, &mut roots);
    if let Some(guard) = &guard {
        collect_items(guard, &mut roots);
    }
    let kept = reachable(closure, &roots);
    let map: BTreeMap<usize, usize> =
        kept.iter().enumerate().map(|(new, &old)| (old, new)).collect();
    let mut items = Vec::with_capacity(kept.len());
    let mut cited: BTreeMap<RegimeId, CounterfactualCitedRegime> = BTreeMap::new();
    for &old in &kept {
        let item = closure.items[old];
        let remap = |from: usize| map.get(&from).copied().unwrap_or(from);
        let source = match item.source {
            ItemSource::Regime { regime, intervened } => {
                // Stage 1 only closes over supplied regimes, so every regime is present.
                let regime = regime.unwrap_or(RegimeId::from_raw(u32::MAX));
                cited.entry(regime).or_insert_with(|| CounterfactualCitedRegime {
                    regime,
                    population: Arc::clone(&validated.populations[item.population]),
                    interventions: problem.mask_variables(intervened),
                });
                CounterfactualItemSource::Regime { regime, intervened }
            }
            ItemSource::Marginal { from } => {
                CounterfactualItemSource::Marginal { from: remap(from) }
            }
            ItemSource::Extract { from } => CounterfactualItemSource::Extract { from: remap(from) },
            ItemSource::Transfer { from } => {
                CounterfactualItemSource::Transfer { from: remap(from) }
            }
        };
        items.push(CounterfactualItem {
            population: item.population,
            variables: item.mask,
            source,
        });
    }
    remap_items(&mut numerator, &map);
    remap_items(&mut denominator, &map);
    if let Some(guard) = &mut guard {
        remap_items(guard, &map);
    }
    let unverified = !checked;
    let mut rules = rules;
    if unverified {
        rules.push("positivity.premise_unverified".to_owned());
    }
    CounterfactualTransportDerivation {
        contract: COUNTERFACTUAL_TRANSPORT_CONTRACT,
        problem: problem.clone(),
        query: validated.query.clone(),
        catalog: catalog.clone(),
        populations: validated.populations.clone(),
        items,
        numerator,
        denominator,
        positivity_guard: guard,
        conditioning_mass_checked: checked,
        moved,
        rules,
        cited: cited.into_values().collect(),
        ignored: validated.ignored.clone(),
        search: CounterfactualTransportSearchRecord {
            operations_limit: limits.operations,
            depth_limit: limits.depth,
            memory_limit_bytes: search.budget.memory_limit_bytes(),
            operations_consumed: search.budget.operations(),
            depth_reached: search.budget.depth_reached(),
            stages: search.stages.clone(),
        },
    }
}

/// Verify a witness against its own query and catalog: both models are in the
/// declared class, agree on every regime of the declared family and give
/// different, recorded query values. No theorem is trusted.
///
/// # Errors
///
/// `counterfactual_transport.invalid_witness` when any check fails, or the
/// request's own validation errors.
pub fn verify_counterfactual_transport_witness(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    witness: &CounterfactualTransportWitness,
) -> Result<CounterfactualTransportWitnessCheck, IdentificationError> {
    let validated = match validate(problem, query, catalog) {
        Ok(v) => v,
        Err(Rejection::Invalid(error)) => return Err(error),
        Err(Rejection::Unsupported(..)) => {
            return Err(IdentificationError::invalid_derivation(INVALID_WITNESS));
        }
    };
    let bad = || IdentificationError::invalid_derivation(INVALID_WITNESS);
    let supplied = supplied_seeds(&validated);
    let candidates =
        candidate_seeds(&validated, problem, &mut |_, _, _| Ok(())).map_err(|_| bad())?;
    let mut family: Vec<(usize, u32)> = supplied
        .iter()
        .chain(&candidates)
        .map(|s| (s.population, s.intervened))
        .chain(validated.unused_family.iter().copied())
        .collect();
    family.sort_unstable();
    family.dedup();
    let event = atoms_map(&validated.event).ok_or_else(bad)?;
    let given = atoms_map(&validated.given).ok_or_else(bad)?;
    let numerator = merged(&event, &given).ok_or_else(bad)?;
    let wp = WitnessProblem {
        problem,
        selected: &validated.selected,
        family: &family,
        worlds: &validated.worlds,
        numerator: &numerator,
        given: &given,
        // A witness is verified against the most simulations any permitted search could pay for.
        simulations: u64::try_from(COUNTERFACTUAL_TRANSPORT_MAX_LIMITS.operations)
            .unwrap_or(u64::MAX)
            .saturating_mul(witness::SIMULATIONS_PER_OPERATION),
    };
    verify_witness(&wp, witness).map_err(|_| bad())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search_under(limits: SearchLimits, memory: u64, ctx: &ExecutionContext) -> Search<'_> {
        Search {
            budget: SearchBudget::with_memory(limits, memory, ctx).unwrap(),
            stop: None,
            stages: Vec::new(),
            failures: Vec::new(),
        }
    }

    #[test]
    fn closure_leaf_work_is_cumulative_and_receipts_are_deterministic() {
        let ctx = ExecutionContext::for_tests(1);
        let problem = CounterfactualTransportProblem::new(&binary(&[0, 1, 2]), &[], &[]).unwrap();
        let seeds = [Seed { population: 0, intervened: 0, measured: 0b111, regime: None }];
        let mut bounded = search_under(SearchLimits { operations: 6, depth: 10 }, u64::MAX, &ctx);
        let result = close(&problem, &[0], &seeds, &[], &mut |depth, bytes, leaves| {
            bounded.work(depth, bytes, leaves)
        });
        assert!(matches!(result, Err(SearchStop::Operations)));
        assert_eq!(bounded.budget.operations(), 6);
        assert_eq!(
            bounded.budget.receipt(SearchStop::Operations, vec![], vec![]).operations_consumed,
            Some(6)
        );
        let run = || {
            let mut search =
                search_under(SearchLimits { operations: 1000, depth: 10 }, u64::MAX, &ctx);
            let closure = close(&problem, &[0], &seeds, &[], &mut |depth, bytes, leaves| {
                search.work(depth, bytes, leaves)
            })
            .unwrap();
            (closure.items, search.budget.operations())
        };
        assert_eq!(run(), run());
    }

    /// A summation is quoted before anything exists: work against the operations the caller's
    /// budget still allows, storage against the lease the pool can fund.
    #[test]
    fn summation_is_refused_from_the_callers_budget_before_any_storage_is_leased() {
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.set_memory_limit(Some(1 << 20));
        let limits = SearchLimits { operations: 100, depth: 4 };
        // 4 * 5 * 3 = 60 cells: 60 operations and 480 bytes fit.
        let mut search = search_under(limits, 1 << 20, &ctx);
        assert_eq!(search.admit_summation([4u64, 5, 3].into_iter()), Ok(()));
        assert_eq!(ctx.admission().available().bytes(), 1 << 20);
        // Spend operations, and the same summation no longer fits the remainder.
        for _ in 0..50 {
            search.charge(1, 0).unwrap();
        }
        assert_eq!(search.admit_summation([4u64, 5, 3].into_iter()), Err(SearchStop::Operations));
        assert_eq!(search.stop, Some(SearchStop::Operations));
        // Storage above the search's memory cap is refused with the pool untouched.
        let mut capped = search_under(limits, 400, &ctx);
        assert_eq!(capped.admit_summation([4u64, 5, 3].into_iter()), Err(SearchStop::Memory));
        assert_eq!(ctx.admission().available().bytes(), 1 << 20);
        // A product that overflows u64 is refused, never taken as zero.
        let mut huge =
            search_under(SearchLimits { operations: usize::MAX, depth: 4 }, u64::MAX, &ctx);
        assert_eq!(huge.admit_summation([u64::MAX, 2].into_iter()), Err(SearchStop::Memory));
        // No summed symbol is one empty term.
        assert_eq!(huge.admit_summation(std::iter::empty()), Ok(()));
        // The pool, not only the cap, decides: a pool that cannot fund the cells refuses.
        let mut small = ExecutionContext::for_tests(1);
        small.set_memory_limit(Some(100));
        let mut pooled = search_under(limits, 1 << 20, &small);
        assert_eq!(pooled.admit_summation([4u64, 5, 3].into_iter()), Err(SearchStop::Memory));
        assert_eq!(small.admission().available().bytes(), 100);
    }
    use antecedent_core::{
        CounterfactualEvent, DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeKind,
        WorldId, WorldSpec,
    };

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn binary(ids: &[u32]) -> Vec<(VariableId, Vec<f64>)> {
        ids.iter().map(|&i| (v(i), vec![0.0, 1.0])).collect()
    }

    fn regime(
        id: u32,
        population: &str,
        interventions: &[u32],
        measured: &[u32],
    ) -> EvidenceRegime {
        let kind = if interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        };
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            kind,
            EvidenceKind::Available,
            interventions.iter().map(|&i| v(i)).collect::<Vec<_>>(),
            Vec::<antecedent_core::InterventionAssignment>::new(),
            measured.iter().map(|&i| v(i)).collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
        EvidenceCatalog::try_new(
            Vec::<antecedent_core::Environment>::new(),
            regimes,
            Vec::<antecedent_core::RegimeBinding>::new(),
            None,
        )
        .unwrap()
    }

    fn interventional_query(
        treatment: u32,
        level: f64,
        outcome: u32,
        sources: Vec<CounterfactualTransportSource>,
    ) -> CounterfactualTransportQuery {
        let event = CounterfactualEventQuery::new(
            vec![WorldSpec::new([(v(treatment), level)], []).unwrap()],
            ExogenousCoupling::SharedLatentExogenous,
            [CounterfactualEvent::new(WorldId::new(0), v(outcome), 1.0).unwrap()],
            [],
        )
        .unwrap();
        CounterfactualTransportQuery { event, target: Arc::from("target"), sources: sources.into() }
    }

    fn source(selection: &[u32], controllable: &[u32]) -> CounterfactualTransportSource {
        CounterfactualTransportSource {
            population: Arc::from("source"),
            selection_targets: selection.iter().map(|&i| v(i)).collect::<Vec<_>>().into(),
            controllable: controllable.iter().map(|&i| v(i)).collect::<Vec<_>>().into(),
        }
    }

    fn decide(
        problem: &CounterfactualTransportProblem,
        query: &CounterfactualTransportQuery,
        catalog: &EvidenceCatalog,
    ) -> CounterfactualTransportDecision {
        let ctx = ExecutionContext::for_tests(1);
        identify_counterfactual_transport(
            problem,
            query,
            catalog,
            COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap()
    }

    #[test]
    fn unconfounded_effect_identifies_from_the_target_observational_law() {
        // 0 -> 1, no confounding, no selection.
        let problem =
            CounterfactualTransportProblem::new(&binary(&[0, 1]), &[(v(0), v(1))], &[]).unwrap();
        let query = interventional_query(0, 1.0, 1, vec![]);
        let cat = catalog(vec![regime(0, "target", &[], &[0, 1])]);
        let decision = decide(&problem, &query, &cat);
        let derivation = decision.derivation().expect("identified");
        assert_eq!(derivation.cited.len(), 1);
        assert_eq!(derivation.cited[0].regime, RegimeId::from_raw(0));
        derivation.replay_checked(&ExecutionContext::for_tests(1)).unwrap();
    }

    #[test]
    fn checked_derivation_text_preserves_bytes_and_stops_on_budget() {
        let problem =
            CounterfactualTransportProblem::new(&binary(&[0, 1]), &[(v(0), v(1))], &[]).unwrap();
        let query = interventional_query(0, 1.0, 1, vec![]);
        let cat = catalog(vec![regime(0, "target", &[], &[0, 1])]);
        let decision = decide(&problem, &query, &cat);
        let mut derivation = decision.derivation().unwrap().clone();
        derivation.query.sources = vec![
            CounterfactualTransportSource {
                population: Arc::from("zé"),
                selection_targets: Arc::from([v(1), v(0), v(1)]),
                controllable: Arc::from([v(1), v(0)]),
            },
            CounterfactualTransportSource {
                population: Arc::from("a"),
                selection_targets: Arc::from([v(1)]),
                controllable: Arc::from([v(0)]),
            },
        ]
        .into();
        let mut spent = 0usize;
        let text = derivation
            .try_canonical_text(|amount| {
                spent += amount;
                Ok::<_, ()>(())
            })
            .unwrap();
        assert_eq!(text, derivation.canonical_text());
        assert!(spent >= text.len());
        let mut remaining = 4usize;
        assert_eq!(
            derivation.try_canonical_text(|amount| {
                remaining = remaining.checked_sub(amount).ok_or("budget")?;
                Ok(())
            }),
            Err("budget")
        );
        assert_eq!(derivation.try_canonical_text(|_| Err::<(), _>("cancelled")), Err("cancelled"));
    }

    #[test]
    fn confounded_effect_needs_the_declared_experiment_and_otherwise_has_a_witness() {
        // 0 -> 1 with 0 <-> 1.
        let problem =
            CounterfactualTransportProblem::new(&binary(&[0, 1]), &[(v(0), v(1))], &[(v(0), v(1))])
                .unwrap();
        let cat = catalog(vec![regime(0, "target", &[], &[0, 1])]);
        // No source can experiment: a verified two-model witness.
        let query = interventional_query(0, 1.0, 1, vec![source(&[], &[])]);
        match decide(&problem, &query, &cat) {
            CounterfactualTransportDecision::ProvenNonTransportable(obstruction) => {
                obstruction.verify().unwrap();
                assert_ne!(obstruction.check.first_value, obstruction.check.second_value);
            }
            other => panic!("expected an obstruction, got {other:?}"),
        }
        // A source that can experiment on 0 but supplied none: missing evidence.
        let query = interventional_query(0, 1.0, 1, vec![source(&[], &[0])]);
        match decide(&problem, &query, &cat) {
            CounterfactualTransportDecision::MissingEvidence(missing) => {
                assert_eq!(missing.missing.len(), 1);
                assert_eq!(missing.missing[0].interventions, vec![v(0)]);
            }
            other => panic!("expected missing evidence, got {other:?}"),
        }
        // Supplying it identifies.
        let cat = catalog(vec![regime(0, "target", &[], &[0, 1]), regime(1, "source", &[0], &[1])]);
        let decision = decide(&problem, &query, &cat);
        assert!(decision.derivation().is_some(), "{decision:?}");
        // A source allowed to intervene on every node can still perform the subset do(0),
        // leaving 1 measured. The available source law is the same in either declaration.
        let full_control = interventional_query(0, 1.0, 1, vec![source(&[], &[0, 1])]);
        let decision = decide(&problem, &full_control, &cat);
        let derivation = decision.derivation().expect("the source do(0) law identifies");
        assert!(derivation.cited.iter().any(|entry| entry.regime == RegimeId::from_raw(1)));
        derivation.replay_checked(&ExecutionContext::for_tests(1)).unwrap();
    }

    #[test]
    fn selection_on_the_outcome_blocks_the_source_experiment() {
        // 0 -> 1, 0 <-> 1, selection on 1: the source experiment cannot transport.
        let problem =
            CounterfactualTransportProblem::new(&binary(&[0, 1]), &[(v(0), v(1))], &[(v(0), v(1))])
                .unwrap();
        let cat = catalog(vec![
            regime(0, "target", &[], &[0, 1]),
            regime(1, "source", &[0], &[1]),
            regime(2, "source", &[], &[0, 1]),
        ]);
        let query = interventional_query(0, 1.0, 1, vec![source(&[1], &[0])]);
        match decide(&problem, &query, &cat) {
            CounterfactualTransportDecision::ProvenNonTransportable(obstruction) => {
                obstruction.verify().unwrap();
            }
            other => panic!("expected an obstruction, got {other:?}"),
        }
        // The same experiment transports when only the treatment's mechanism differs.
        let query = interventional_query(0, 1.0, 1, vec![source(&[0], &[0])]);
        let decision = decide(&problem, &query, &cat);
        // The route is the proposition (identified, not obstructed); the derived values are
        // checked against enumerated truth in the estimate-side transport tests.
        assert!(decision.derivation().is_some(), "{decision:?}");
    }

    #[test]
    fn out_of_contract_requests_are_typed() {
        let problem =
            CounterfactualTransportProblem::new(&binary(&[0, 1]), &[(v(0), v(1))], &[]).unwrap();
        let abducted = CounterfactualEventQuery::new(
            vec![WorldSpec::new([(v(0), 1.0)], []).unwrap()],
            ExogenousCoupling::SharedAbducedExogenous,
            [CounterfactualEvent::new(WorldId::new(0), v(1), 1.0).unwrap()],
            [],
        )
        .unwrap();
        let query = CounterfactualTransportQuery {
            event: abducted,
            target: Arc::from("target"),
            sources: Vec::new().into(),
        };
        let cat = catalog(vec![regime(0, "target", &[], &[0, 1])]);
        let decision = decide(&problem, &query, &cat);
        assert!(matches!(decision, CounterfactualTransportDecision::UnsupportedScope(_)));
        assert_eq!(decision.reason_code(), Some("route_not_supported"));
    }
}
