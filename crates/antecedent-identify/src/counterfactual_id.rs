//! Counterfactual identification on a bounded finite-discrete ADMG: the effect
//! of treatment on the treated from the observational joint.
//!
//! # What is supported
//!
//! One query shape, `P(Y_x = y | X = x')` with `x != x'` (and, through every
//! level of `Y`, `E[Y_x | X = x'] - E[Y | X = x']`), on an explicit ADMG with
//! finitely many levels per variable, from the observational joint `P(V)`.
//! Graph-level identification has no count cap on variables or levels: it is
//! bounded by the caller's [`SearchLimits`] and memory budget and stops with a
//! budget refusal, never a verdict. [`COUNTERFACTUAL_ID_MAX_VARIABLES`] and
//! [`COUNTERFACTUAL_ID_MAX_LEVELS`] are the sizes the stored-artifact intake of
//! the exact finite-state evidence stage accepts; this module does not apply them. The numerator `P(Y_x = y, X = x')` is the two-event conjunction
//! ID* (Shpitser and Pearl 2007/2008) decides; the division by `P(X = x')`
//! happens outside the algorithm. A conjunction that contradicts itself within
//! one world (a variable read at two levels in one world, or read at a level
//! its own world sets it away from) has probability zero in every model and is
//! returned as an exact zero, not refused. Every other shape is refused as
//! outside the contract, and a world that routes edges (a path-specific query)
//! is not supported by this route.
//!
//! # The algorithm
//!
//! [`decide_counterfactual_id`] runs ID* on the conjunction:
//!
//! 1. an empty conjunction is 1; an event that its own subscript sets away from
//!    its level is 0; an event its own subscript sets to its level is dropped;
//! 2. make-cg builds one world per distinct subscript, merges two copies of a
//!    variable (in topological order) when every parent pair is one merged node
//!    or two fixed parents at one level (Lemma 24; the variable's exogenous
//!    terms are shared across worlds), and keeps the ancestors of the event.
//!    Two events on one merged node at different levels make the conjunction 0.
//!    Each node's effective subscript is the fixed parents of it and of its
//!    ancestors, so `Y_{x,m}` is `Y_m` when `X` reaches `Y` only through `M`;
//! 3. more than one district: sum over the nodes the event leaves free of the
//!    product, over districts `S`, of ID* on `S`'s events with every node also
//!    set to the values of the parents of `S` outside `S` that are its
//!    ancestors;
//! 4. one district `S`: if a variable is set in a subscript of `S` and read in
//!    `S` at another (or an unknown) level, FAIL with the conflicting pair
//!    (`counterfactual_id.conflicting_subscripts`); otherwise
//!    `P_{sub(S)}(ev(S))`, with the free nodes of `S` marginalized, which the
//!    existing complete ID identifies from `P(V)` or refuses with a hedge
//!    (`counterfactual_id.counterfactual_hedge`).
//!
//! When ID* stops at a conflict or a hedge and the treatment has exactly two
//! levels, one more route is tried: by consistency,
//! `P(Y_x = y, X = x) = P(Y = y, X = x)`, and `X = x'` is the complement of
//! `X = x`, so `P(Y_x = y, X = x') = P_x(Y = y) - P(Y = y, X = x)`
//! ([`CounterfactualFunctional::ConsistencyComplement`]); it answers whenever
//! complete ID identifies `P_x(Y)` from `P(V)`, and otherwise the ID* refusal
//! stands. With three or more treatment levels no such route is tried.
//!
//! Every ID* call, every make-cg merge step and every ID recursion charges one
//! [`SearchBudget`] shared by the whole decision (every level of `Y`), with the
//! cumulative live-state bytes; a stop returns a receipt and is never a
//! nonidentification verdict.
//!
//! # What is and is not claimed
//!
//! The identified functional is sound: every term is an interventional law
//! identified by complete ID, the composition is the ID* identity or the
//! binary-treatment consistency identity above; the tests check it against
//! exact enumerated latent-variable structural models. A refusal is NOT a proof
//! of non-identifiability: a conflicting-subscript refusal says only that ID*
//! (with the binary complement where it applies) does not identify the
//! conjunction here. ID*'s completeness (Shpitser and Pearl, JMLR 2008;
//! paper-inherited) is from experimental distributions, and this implementation
//! composes it with ID from `P(V)`, a composition that is not complete: a graph
//! whose ETT is identified from `P(V)` through `P(y | do(x))` was refused by
//! ID* alone (pinned in the tests; the binary complement now answers it, and
//! with a three-level treatment it remains a known completeness gap).
//! Shpitser and Pearl (UAI 2009) characterize singleton-treatment ETT
//! identification from `P(V)` (paper-inherited); that characterization is not
//! implemented. The conflict check treats a node the event leaves free as
//! taking an unknown level, which can only refuse more, never answer wrongly.
//! Refusals therefore carry `route_not_supported`, never
//! `cross_world_not_identified`.
//!
//! Some guards of the recursion cannot fire on the effect-on-the-treated
//! shape: its top-level counterfactual graph holds natural-world copies only of
//! ancestors of `X` and treated-world copies only of descendants of `X`, so no
//! variable has two copies and no two copies see fixed parents at two levels.
//! They are kept for the general recursion and each is exercised by a
//! synthetic conjunction on the engine in this module's unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, ExecutionContext, ExogenousCoupling, NodeRef,
    SearchBudget, SearchLimits, SearchReceipt, SearchStop, Value, VariableId,
};
use antecedent_expr::{
    Assignment, CausalExprArena, DomainRef, EmpiricalTableProvider, EvalContext, ExactDiscreteLaw,
    ExprId, ExprNode, FactorSpec,
};
use antecedent_graph::algo::{components_by, implicit_closure};
use antecedent_graph::{Admg, BitSet, DenseNodeId, GraphWorkspace};
use serde::{Deserialize, Serialize};

use crate::error::IdentificationError;
use crate::hedge::HedgeCertificate;
use crate::id::{MeteredId, identify_interventional_metered};
use crate::prepared::PreparedAdmg;
use crate::sid::{SearchCharge, SharedSearch};

/// Most observed variables the exact finite-state evidence stage accepts from a stored
/// artifact. Graph-level identification in this module does not apply it.
pub const COUNTERFACTUAL_ID_MAX_VARIABLES: usize = 6;
/// Most levels per variable the exact finite-state evidence stage accepts from a stored
/// artifact. Graph-level identification in this module does not apply it.
pub const COUNTERFACTUAL_ID_MAX_LEVELS: usize = 4;
/// Largest operation limit a caller may ask for.
pub const COUNTERFACTUAL_ID_MAX_OPERATIONS: usize = 100_000;
/// Largest depth limit a caller may ask for.
pub const COUNTERFACTUAL_ID_MAX_DEPTH: usize = 64;
/// Default search limits.
pub const COUNTERFACTUAL_ID_DEFAULT_LIMITS: SearchLimits =
    SearchLimits { operations: 20_000, depth: 48 };
/// Default memory cap of the search (further limited by the context's hard limit).
pub const COUNTERFACTUAL_ID_MEMORY_BYTES: u64 = 64 * 1024 * 1024;
/// The contract a derivation attests.
pub const COUNTERFACTUAL_ID_CONTRACT: &str = "effect_on_treated_admg_v1";

/// Why a counterfactual query is refused.
///
/// No refusal of this cell claims non-identifiability: a conflicting-subscript
/// district or an ID hedge on a district term is `route_not_supported` with a
/// checkable obstruction (ID* does not identify the conjunction; this is not a
/// proof that nothing does), a query shape the contract does not implement is
/// `route_not_supported` too, and a budget stop is `transport_budget_cancel`
/// with a receipt. `cross_world_not_identified` is not used: nothing here
/// proves non-identification.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualIdRefusal {
    /// Registered reason code (`parity/reason_codes.toml`).
    pub code: &'static str,
    /// Namespaced stable detail.
    pub detail: &'static str,
    /// Explanation naming what failed.
    pub message: String,
    /// The checkable obstruction of an ID* refusal (not a non-identifiability proof).
    pub obstruction: Option<Box<CounterfactualObstruction>>,
    /// The receipt of a budget stop.
    pub receipt: Option<Box<SearchReceipt>>,
}

impl std::fmt::Display for CounterfactualIdRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}: {}", self.code, self.detail, self.message)?;
        if let Some(receipt) = &self.receipt {
            write!(f, "; {}", receipt.summary())?;
        }
        Ok(())
    }
}

impl std::error::Error for CounterfactualIdRefusal {}

impl CounterfactualIdRefusal {
    fn plain(code: &'static str, detail: &'static str, message: String) -> Self {
        Self { code, detail, message, obstruction: None, receipt: None }
    }

    /// A malformed query, law or artifact input (`invalid_argument`,
    /// `counterfactual_id.invalid_query`).
    #[must_use]
    pub fn invalid_query(message: impl Into<String>) -> Self {
        Self::plain(
            "invalid_argument",
            antecedent_core::reason_details::counterfactual_id::INVALID_QUERY,
            message.into(),
        )
    }

    /// A conditional the functional evaluates is on a zero-mass event.
    #[must_use]
    pub fn positivity(message: impl Into<String>) -> Self {
        Self::plain(
            "invalid_argument",
            antecedent_core::reason_details::counterfactual_id::POSITIVITY_VIOLATION,
            message.into(),
        )
    }

    /// A declared bound is exceeded.
    #[must_use]
    pub fn bounds(message: impl Into<String>) -> Self {
        Self::plain(
            "route_not_supported",
            antecedent_core::reason_details::counterfactual_id::BOUNDS_EXCEEDED,
            message.into(),
        )
    }

    /// A structure outside the graph contract.
    #[must_use]
    pub fn graph(message: impl Into<String>) -> Self {
        Self::plain(
            "route_not_supported",
            antecedent_core::reason_details::counterfactual_id::GRAPH_OUTSIDE_CONTRACT,
            message.into(),
        )
    }

    /// An interval or Bayesian request.
    #[must_use]
    pub fn interval_requested() -> Self {
        Self::plain(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            antecedent_core::reason_details::counterfactual_id::INTERVAL_REQUESTED,
            "the counterfactual identification cell is a point claim; no interval or posterior \
             is published"
                .into(),
        )
    }

    fn outside_contract(message: impl Into<String>) -> Self {
        Self::plain(
            antecedent_core::reason_code!("route_not_supported"),
            antecedent_core::reason_details::counterfactual_id::QUERY_OUTSIDE_CONTRACT,
            message.into(),
        )
    }

    fn path_specific() -> Self {
        Self::plain(
            antecedent_core::reason_code!("route_not_supported"),
            antecedent_core::reason_details::counterfactual_id::PATH_SPECIFIC_NOT_SUPPORTED,
            "a world routes edges: path-specific and recanting-district effects on ADMGs are \
             outside this identification route"
                .into(),
        )
    }

    fn conflict(obstruction: CounterfactualObstruction) -> Self {
        Self {
            code: antecedent_core::reason_code!("route_not_supported"),
            detail: antecedent_core::reason_details::counterfactual_id::CONFLICTING_SUBSCRIPTS,
            message: obstruction.describe(),
            obstruction: Some(Box::new(obstruction)),
            receipt: None,
        }
    }

    fn hedge(obstruction: CounterfactualObstruction) -> Self {
        Self {
            code: antecedent_core::reason_code!("route_not_supported"),
            detail: antecedent_core::reason_details::counterfactual_id::COUNTERFACTUAL_HEDGE,
            message: obstruction.describe(),
            obstruction: Some(Box::new(obstruction)),
            receipt: None,
        }
    }

    fn budget(receipt: SearchReceipt) -> Self {
        Self {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            detail: antecedent_core::reason_details::counterfactual_id::BUDGET,
            message: format!(
                "{} stopped the counterfactual identification search; this is not a \
                 nonidentification verdict",
                receipt.stop.code()
            ),
            obstruction: None,
            receipt: Some(Box::new(receipt)),
        }
    }
}

/// The graph and the variable levels a decision is made on, canonical: edges
/// sorted and deduplicated, bidirected edges oriented `(low, high)`, levels as
/// IEEE bits in the declared order (`-0.0` is `0.0`).
#[derive(Clone, Debug)]
pub struct CounterfactualIdProblem {
    levels: Vec<Vec<u64>>,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    /// Kahn order of `directed`. Omitted from [`Eq`] and [`Hash`], which stay the
    /// edge and level identity the canonical text already records.
    topo: Vec<u32>,
}

impl PartialEq for CounterfactualIdProblem {
    fn eq(&self, other: &Self) -> bool {
        self.levels == other.levels
            && self.directed == other.directed
            && self.bidirected == other.bidirected
    }
}

impl Eq for CounterfactualIdProblem {}

impl std::hash::Hash for CounterfactualIdProblem {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.levels.hash(state);
        self.directed.hash(state);
        self.bidirected.hash(state);
    }
}

impl CounterfactualIdProblem {
    /// A problem over variables `0..levels.len()`.
    ///
    /// # Errors
    ///
    /// `counterfactual_id.invalid_query` for no variables, an empty, repeated or
    /// non-finite level list or an edge naming an unknown variable or a
    /// self-loop, `counterfactual_id.graph_outside_contract` for a directed
    /// cycle.
    pub fn new(
        levels: Vec<Vec<f64>>,
        directed: &[(u32, u32)],
        bidirected: &[(u32, u32)],
    ) -> Result<Self, CounterfactualIdRefusal> {
        let n = levels.len();
        if n == 0 {
            return Err(CounterfactualIdRefusal::invalid_query("the graph has no variables"));
        }
        if u32::try_from(n).is_err() {
            return Err(CounterfactualIdRefusal::bounds(format!(
                "the graph has {n} variables; a variable id cannot name them"
            )));
        }
        let mut canonical_levels = Vec::with_capacity(n);
        for (variable, list) in levels.into_iter().enumerate() {
            let bits: Vec<u64> = list.iter().map(|x| (x + 0.0).to_bits()).collect();
            let distinct: BTreeSet<u64> = bits.iter().copied().collect();
            if list.is_empty()
                || list.iter().any(|x| !x.is_finite())
                || distinct.len() != bits.len()
            {
                return Err(CounterfactualIdRefusal::invalid_query(format!(
                    "variable {variable} needs one or more distinct finite levels"
                )));
            }
            canonical_levels.push(bits);
        }
        let bound = u32::try_from(n).unwrap_or(u32::MAX);
        let valid = |&(a, b): &(u32, u32)| a < bound && b < bound;
        if !directed.iter().all(valid) || !bidirected.iter().all(valid) {
            return Err(CounterfactualIdRefusal::invalid_query(
                "an edge names an unknown variable",
            ));
        }
        if bidirected.iter().any(|&(a, b)| a == b) {
            return Err(CounterfactualIdRefusal::invalid_query(
                "a bidirected edge cannot be a self-loop",
            ));
        }
        let mut directed = directed.to_vec();
        directed.sort_unstable();
        directed.dedup();
        let mut bidirected: Vec<(u32, u32)> =
            bidirected.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        bidirected.sort_unstable();
        bidirected.dedup();
        if let Err(refusal) = antecedent_graph::error::refuse_directed_cycle(
            antecedent_graph::error::DagOperation::Identification,
            antecedent_graph::error::CyclicNextAttempt::Unsupported {
                consumer: "counterfactual_id",
                attempt: "hold the directed edges as a CyclicMixedGraph; condensation names the \
                          feedback and sigma_separates answers a conditional-independence \
                          question. This cell has no cyclic identification evidence",
            },
            canonical_levels.len(),
            &directed,
        ) {
            return Err(CounterfactualIdRefusal::graph(refusal.to_string()));
        }
        let Some(topo) = crate::cross_world::topological_order(canonical_levels.len(), &directed)
        else {
            return Err(CounterfactualIdRefusal::graph("the directed edges contain a cycle"));
        };
        Ok(Self { levels: canonical_levels, directed, bidirected, topo })
    }

    /// The problem of an ADMG whose dense node `i` is the static variable `i`.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and `counterfactual_id.graph_outside_contract` for a
    /// node that is not the static variable of its position.
    pub fn from_admg(admg: &Admg, levels: Vec<Vec<f64>>) -> Result<Self, CounterfactualIdRefusal> {
        let n = admg.node_count();
        for (i, node) in admg.nodes().iter().enumerate() {
            if *node != NodeRef::Static(VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)))
            {
                return Err(CounterfactualIdRefusal::graph(
                    "every node must be the static variable of its position",
                ));
            }
        }
        if levels.len() != n {
            return Err(CounterfactualIdRefusal::invalid_query(format!(
                "{} level lists for a graph of {n} variables",
                levels.len()
            )));
        }
        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for i in 0..n {
            let id = DenseNodeId::from_index(i);
            directed.extend(admg.children(id).iter().map(|c| (id.raw(), c.raw())));
            bidirected.extend(admg.bidirected_neighbors(id).iter().map(|b| (id.raw(), b.raw())));
        }
        Self::new(levels, &directed, &bidirected)
    }

    /// Number of variables.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.levels.len()
    }

    /// Levels of `variable`, in declared order.
    #[must_use]
    pub fn levels(&self, variable: usize) -> Vec<f64> {
        self.levels
            .get(variable)
            .map_or_else(Vec::new, |l| l.iter().map(|b| f64::from_bits(*b)).collect())
    }

    /// Directed edges, sorted.
    #[must_use]
    pub fn directed(&self) -> &[(u32, u32)] {
        &self.directed
    }

    /// Bidirected edges as `(low, high)`, sorted.
    #[must_use]
    pub fn bidirected(&self) -> &[(u32, u32)] {
        &self.bidirected
    }

    /// Canonical text of the graph and levels (a digest input).
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("counterfactual_id_problem_v1;levels=[");
        for list in &self.levels {
            out.push('[');
            for bits in list {
                let _ = write!(out, "{bits:016x},");
            }
            out.push(']');
        }
        out.push_str("];directed=[");
        for (a, b) in &self.directed {
            let _ = write!(out, "{a}>{b},");
        }
        out.push_str("];bidirected=[");
        for (a, b) in &self.bidirected {
            let _ = write!(out, "{a}<>{b},");
        }
        out.push(']');
        out
    }

    fn level_index(&self, variable: u32, bits: u64) -> Option<usize> {
        self.levels.get(variable as usize)?.iter().position(|b| *b == bits)
    }

    fn parents(&self, variable: u32) -> Vec<u32> {
        self.directed.iter().filter(|e| e.1 == variable).map(|e| e.0).collect()
    }

    fn topological_order(&self) -> &[u32] {
        &self.topo
    }

    fn confounded(&self, a: u32, b: u32) -> bool {
        self.bidirected.contains(&(a.min(b), a.max(b)))
    }

    fn prepared(&self) -> Result<PreparedAdmg, IdentificationError> {
        let mut admg = Admg::empty();
        for i in 0..self.levels.len() {
            let raw = u32::try_from(i).map_err(|_| {
                IdentificationError::out_of_range(
                    "counterfactual graph has more nodes than a variable id can name",
                )
            })?;
            admg.add_node(NodeRef::Static(VariableId::from_raw(raw)))?;
        }
        for &(a, b) in &self.directed {
            admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))?;
        }
        for &(a, b) in &self.bidirected {
            admg.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))?;
        }
        PreparedAdmg::new(admg)
    }
}

/// One node of a counterfactual graph record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualNodeRecord {
    /// Variable.
    pub variable: u32,
    /// Level the event reads it at; `None` for a node the event leaves free.
    pub value: Option<CfValue>,
    /// Effective subscript `(variable, level)`, sorted.
    pub subscript: Vec<(u32, CfValue)>,
    /// Parent nodes (indices into the record's nodes).
    pub parents: Vec<usize>,
}

/// A counterfactual graph (after make-cg) as a checkable record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualGraphRecord {
    /// Nodes, canonical order.
    pub nodes: Vec<CounterfactualNodeRecord>,
    /// Bidirected edges `(low, high)` by node index, sorted.
    pub bidirected: Vec<(usize, usize)>,
    /// Districts (sorted node indices), sorted.
    pub districts: Vec<Vec<usize>>,
}

/// A checked reason ID* does not identify a counterfactual conjunction. It
/// re-checks its own record; it is not a proof of non-identifiability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CounterfactualObstruction {
    /// ID* FAIL: a district reads a variable at a level its subscripts set it
    /// away from.
    ConflictingSubscripts {
        /// The counterfactual graph of the failing ID* call (a single district).
        graph: CounterfactualGraphRecord,
        /// The conflicting variable.
        variable: u32,
        /// The level a subscript of the district sets it to.
        subscript: CfValue,
        /// The level the district reads it at (`None`: the node is free).
        event: Option<CfValue>,
    },
    /// A district term `P_x(ev)` is not identified from `P(V)`: an ID hedge
    /// with the graph it was found in, so it re-verifies from the definition.
    Hedge {
        /// Number of variables of the graph.
        node_count: usize,
        /// Directed edges of the graph.
        directed: Vec<(u32, u32)>,
        /// Bidirected edges of the graph.
        bidirected: Vec<(u32, u32)>,
        /// Intervened variables of the term.
        treatments: Vec<u32>,
        /// Event variables of the term.
        outcomes: Vec<u32>,
        /// `F` of the hedge.
        f: Vec<u32>,
        /// `F'` of the hedge.
        f_prime: Vec<u32>,
    },
}

impl CounterfactualObstruction {
    fn describe(&self) -> String {
        match self {
            Self::ConflictingSubscripts { variable, subscript, event, .. } => format!(
                "variable {variable} is set to {} in a subscript of a district that reads it at \
                 {}: the conjunction is not identified by ID* (conflicting subscripts); this is \
                 not a proof of non-identifiability",
                subscript.render(),
                event.map_or_else(|| "a free level".to_string(), CfValue::render)
            ),
            Self::Hedge { treatments, outcomes, .. } => format!(
                "the district term P(outcomes {outcomes:?} | do({treatments:?})) is not identified \
                 from the observational law (hedge), so ID* with ID does not identify the \
                 conjunction; this is not a proof that the conjunction is not identifiable"
            ),
        }
    }

    /// Check the obstruction on its own record: a conflict's graph is one
    /// district containing a node that reads the variable at a level other than
    /// one some node's subscript sets it to; a hedge's `F' ⊂ F` holds the named
    /// sets. Returns what failed.
    ///
    /// # Errors
    ///
    /// A message naming the inconsistency.
    pub fn verify(&self) -> Result<(), String> {
        match self {
            Self::ConflictingSubscripts { graph, variable, subscript, event } => {
                if graph.districts.len() != 1
                    || graph.districts[0].len() != graph.nodes.len()
                    || !record_is_connected(graph)
                {
                    return Err("a conflict is found in one district spanning the graph".into());
                }
                let sets = |value: CfValue| {
                    graph
                        .nodes
                        .iter()
                        .any(|n| n.subscript.iter().any(|(v, x)| v == variable && *x == value))
                };
                // The other side is a node reading the variable (at a level, or
                // free) or a second subscript setting it elsewhere.
                let other =
                    graph.nodes.iter().any(|n| n.variable == *variable && n.value == *event)
                        || event.is_some_and(sets);
                if sets(*subscript) && other && *event != Some(*subscript) {
                    Ok(())
                } else {
                    Err("the named pair does not conflict in the recorded district".into())
                }
            }
            Self::Hedge { node_count, directed, bidirected, treatments, outcomes, f, f_prime } => {
                let ids = |raw: &[u32]| -> Vec<VariableId> {
                    raw.iter().map(|&v| VariableId::from_raw(v)).collect()
                };
                let dense = |raw: &[u32]| -> Vec<DenseNodeId> {
                    raw.iter().map(|&v| DenseNodeId::from_raw(v)).collect()
                };
                let problem = crate::hedge::HedgeProblem {
                    variables: (0..*node_count)
                        .map(|v| VariableId::from_raw(u32::try_from(v).unwrap_or(u32::MAX)))
                        .collect(),
                    directed: directed.clone().into(),
                    bidirected: bidirected.clone().into(),
                    treatments: ids(treatments).into(),
                    outcomes: ids(outcomes).into(),
                };
                let certificate = HedgeCertificate {
                    f: ids(f).into(),
                    f_prime: ids(f_prime).into(),
                    f_dense: dense(f).into(),
                    f_prime_dense: dense(f_prime).into(),
                    problem: Some(problem),
                };
                certificate.verify_carried().map_err(|e| e.to_string())
            }
        }
    }
}

/// The budget accounting of a finished decision, stored with the derivation so
/// a consumer can replay it under the same limits and compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualIdSearchRecord {
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Effective memory cap in force.
    pub memory_limit_bytes: u64,
    /// Operations charged by the whole decision.
    pub operations_consumed: usize,
    /// Deepest level charged.
    pub depth_reached: usize,
}

/// What the supported query is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CounterfactualIdShape {
    /// `P(Y_x = y | X = x')`.
    EffectOnTreated {
        /// Treatment `X`.
        treatment: u32,
        /// Counterfactual level `x`.
        active: f64,
        /// Observed level `x'`.
        observed: f64,
        /// Outcome `Y`.
        outcome: u32,
        /// Requested outcome level `y`.
        outcome_level: f64,
    },
    /// A conjunction contradicting itself within one world: probability 0.
    ExactZero,
}

/// The identified derivation: the functional of `P(Y_x = y', X = x')` for every
/// level `y'` of `Y` (or the exact zero), the counterfactual graph of the
/// requested level, and the search accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualIdDerivation {
    /// What was asked.
    pub shape: CounterfactualIdShape,
    /// `(outcome level bits, functional of the numerator)` in the declared level
    /// order; one entry `(0, Zero)` for an exact zero.
    pub numerators: Vec<(u64, CounterfactualFunctional)>,
    /// The counterfactual graph make-cg built for the requested level.
    pub counterfactual_graph: Option<CounterfactualGraphRecord>,
    /// The natural-world conditioning events `(variable, level bits)`.
    pub given: Vec<(u32, u64)>,
    /// Budget accounting.
    pub search: CounterfactualIdSearchRecord,
    /// Canonical text of the query decided.
    pub query_text: String,
}

impl CounterfactualIdDerivation {
    /// Canonical text of the whole derivation: contract, query, every level's
    /// functional (with each ID term's expression), the counterfactual graph.
    /// Equal derivations render equal; the artifact consumer compares it.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        let mut out = format!("{COUNTERFACTUAL_ID_CONTRACT};{};", self.query_text);
        for (bits, functional) in &self.numerators {
            let _ = write!(out, "level {bits:016x}: ");
            functional.render(&mut out);
            out.push('\n');
        }
        if let Some(graph) = &self.counterfactual_graph {
            let _ = write!(out, "cg={graph:?}");
        }
        out
    }

    /// Every district term of every level's functional.
    #[must_use]
    pub fn terms(&self) -> Vec<&CfTerm> {
        let mut out = Vec::new();
        for (_, functional) in &self.numerators {
            functional.terms(&mut out);
        }
        out
    }
}

/// Decide `query` on `problem` under one shared search budget.
///
/// # Errors
///
/// A [`CounterfactualIdRefusal`]: `route_not_supported` with a checked
/// obstruction when ID* (and, for a binary treatment, the consistency
/// complement) does not identify the query (not a non-identifiability proof),
/// `transport_budget_cancel` with a receipt, `route_not_supported` for a shape
/// outside the contract (or deferred, or over a bound), `invalid_argument` for
/// a malformed query.
pub fn decide_counterfactual_id(
    problem: &CounterfactualIdProblem,
    query: &CounterfactualEventQuery,
    limits: SearchLimits,
    memory_limit_bytes: u64,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdDerivation, CounterfactualIdRefusal> {
    if limits.operations > COUNTERFACTUAL_ID_MAX_OPERATIONS
        || limits.depth > COUNTERFACTUAL_ID_MAX_DEPTH
    {
        return Err(CounterfactualIdRefusal::bounds(format!(
            "search limits {} operations / depth {} exceed the contract's \
             {COUNTERFACTUAL_ID_MAX_OPERATIONS} / {COUNTERFACTUAL_ID_MAX_DEPTH}",
            limits.operations, limits.depth
        )));
    }
    let shape = classify(problem, query)?;
    let given: Vec<(u32, u64)> =
        query.given().iter().map(|a| (a.variable().raw(), a.level_bits())).collect();
    let budget = SearchBudget::with_memory(limits, memory_limit_bytes, ctx)
        .map_err(CounterfactualIdRefusal::budget)?;
    decide_under(problem, query, shape, given, limits, SharedSearch::new(budget))
}

/// The decision of an already classified query on one shared budget.
fn decide_under(
    problem: &CounterfactualIdProblem,
    query: &CounterfactualEventQuery,
    shape: CounterfactualIdShape,
    given: Vec<(u32, u64)>,
    limits: SearchLimits,
    mut search: SharedSearch<'_>,
) -> Result<CounterfactualIdDerivation, CounterfactualIdRefusal> {
    search.mark_decision();
    let CounterfactualIdShape::EffectOnTreated {
        treatment,
        active,
        observed,
        outcome,
        outcome_level,
    } = shape
    else {
        search.charge(0, 0).map_err(|stop| {
            CounterfactualIdRefusal::budget(search.receipt(stop, Vec::new(), Vec::new()))
        })?;
        return Ok(CounterfactualIdDerivation {
            shape,
            numerators: vec![(0, CounterfactualFunctional::Zero)],
            counterfactual_graph: None,
            given,
            search: search_record(&search, limits),
            query_text: query.canonical_text(),
        });
    };
    let prepared = problem
        .prepared()
        .map_err(|e| CounterfactualIdRefusal::graph(format!("the graph does not prepare: {e}")))?;
    let mut engine = Engine {
        problem,
        prepared,
        parents: {
            let mut parents = Vec::with_capacity(problem.node_count());
            for v in 0..problem.node_count() {
                let id = u32::try_from(v).map_err(|_| {
                    CounterfactualIdRefusal::graph("variable index does not fit u32")
                })?;
                parents.push(problem.parents(id));
            }
            parents
        },
        search,
        next_symbol: 0,
        retained_bytes: 0,
        first_graph: None,
        workspace: GraphWorkspace::default(),
    };
    let requested = outcome_level.to_bits();
    let levels = problem.levels[outcome as usize].clone();
    let mut numerators = Vec::with_capacity(levels.len());
    let mut explored = Vec::new();
    let mut counterfactual_graph = None;
    for (index, &bits) in levels.iter().enumerate() {
        engine.first_graph = None;
        let functional = match engine.numerator(treatment, active, observed, outcome, bits) {
            Ok(functional) => functional,
            Err(Halt::Stop(stop)) => {
                let unevaluated =
                    levels[index..].iter().map(|b| format!("outcome_level {b:016x}")).collect();
                return Err(CounterfactualIdRefusal::budget(engine.search.receipt(
                    stop,
                    explored,
                    unevaluated,
                )));
            }
            Err(Halt::Obstruction(obstruction)) => {
                return Err(match *obstruction {
                    o @ CounterfactualObstruction::ConflictingSubscripts { .. } => {
                        CounterfactualIdRefusal::conflict(o)
                    }
                    o @ CounterfactualObstruction::Hedge { .. } => {
                        CounterfactualIdRefusal::hedge(o)
                    }
                });
            }
            Err(Halt::Failure(message)) => {
                return Err(CounterfactualIdRefusal::invalid_query(format!(
                    "the derivation failed: {message}"
                )));
            }
        };
        explored.push(format!("outcome_level {bits:016x}"));
        if bits == requested {
            counterfactual_graph = engine.first_graph.take();
        }
        numerators.push((bits, functional));
    }
    let search = search_record(&engine.search, limits);
    Ok(CounterfactualIdDerivation {
        shape,
        numerators,
        counterfactual_graph,
        given,
        search,
        query_text: query.canonical_text(),
    })
}

fn search_record(search: &SharedSearch<'_>, limits: SearchLimits) -> CounterfactualIdSearchRecord {
    CounterfactualIdSearchRecord {
        operations_limit: limits.operations,
        depth_limit: limits.depth,
        memory_limit_bytes: search.memory_limit_bytes(),
        operations_consumed: search.operations(),
        depth_reached: search.decision_depth(),
    }
}

/// Validate the query against the problem and decide its shape.
#[allow(clippy::too_many_lines, reason = "one linear list of contract conditions")]
fn classify(
    problem: &CounterfactualIdProblem,
    query: &CounterfactualEventQuery,
) -> Result<CounterfactualIdShape, CounterfactualIdRefusal> {
    query.validate().map_err(|e| CounterfactualIdRefusal::invalid_query(e.to_string()))?;
    let n = u32::try_from(problem.node_count()).unwrap_or(u32::MAX);
    let known = |variable: VariableId, bits: u64| -> Result<(), CounterfactualIdRefusal> {
        if variable.raw() >= n {
            return Err(CounterfactualIdRefusal::invalid_query(format!(
                "variable {} is not in the graph",
                variable.raw()
            )));
        }
        if problem.level_index(variable.raw(), bits).is_none() {
            return Err(CounterfactualIdRefusal::invalid_query(format!(
                "{} is not a level of variable {}",
                f64::from_bits(bits),
                variable.raw()
            )));
        }
        Ok(())
    };
    for world in query.worlds() {
        for (variable, level) in world.interventions() {
            known(variable, level.to_bits())?;
        }
    }
    for atom in query.event().iter().chain(query.given()) {
        known(atom.variable(), atom.level_bits())?;
    }
    if query.worlds().iter().any(|w| !w.routes().is_empty()) {
        return Err(CounterfactualIdRefusal::path_specific());
    }
    if query.coupling() != ExogenousCoupling::SharedLatentExogenous {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the contract covers worlds sharing latent exogenous terms (marginalized, not abducted)",
        ));
    }
    // A contradiction within one world: a variable read away from the level its
    // own world sets, or read at two levels in one world.
    let contradicts = |atoms: &[&CounterfactualEvent]| -> bool {
        atoms.iter().enumerate().any(|(i, a)| {
            query.worlds()[a.world().index()]
                .intervention_of(a.variable())
                .is_some_and(|set| set.to_bits() != a.level_bits())
                || atoms[i + 1..].iter().any(|b| {
                    b.world() == a.world()
                        && b.variable() == a.variable()
                        && b.level_bits() != a.level_bits()
                })
        })
    };
    let given: Vec<&CounterfactualEvent> = query.given().iter().collect();
    let all: Vec<&CounterfactualEvent> = query.event().iter().chain(query.given()).collect();
    if contradicts(&given) {
        return Err(CounterfactualIdRefusal::invalid_query(
            "the conditioning event contradicts itself: it has probability zero",
        ));
    }
    let natural = |a: &&CounterfactualEvent| {
        query.worlds()[a.world().index()].interventions().next().is_none()
    };
    if contradicts(&all) {
        if !given.iter().all(natural) {
            return Err(CounterfactualIdRefusal::outside_contract(
                "an exact zero is returned only for an observational conditioning event",
            ));
        }
        return Ok(CounterfactualIdShape::ExactZero);
    }
    // The effect-on-the-treated shape.
    let worlds = query.worlds();
    let (&[event], &[given]) = (query.event(), query.given()) else {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the contract covers one counterfactual outcome event and one observed treatment",
        ));
    };
    if worlds.len() != 2 {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the contract covers the natural world and one intervened world",
        ));
    }
    let treated = &worlds[event.world().index()];
    let set: Vec<(VariableId, f64)> = treated.interventions().collect();
    let observed_world = &worlds[given.world().index()];
    let &[(treatment, active)] = set.as_slice() else {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the outcome's world sets exactly the treatment",
        ));
    };
    if observed_world.interventions().next().is_some() || given.world() == event.world() {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the treatment is observed in the natural world",
        ));
    }
    if given.variable() != treatment {
        return Err(CounterfactualIdRefusal::outside_contract(
            "the conditioning event observes the treatment the outcome's world sets",
        ));
    }
    if event.variable() == treatment {
        return Err(CounterfactualIdRefusal::invalid_query(
            "treatment and outcome must be distinct",
        ));
    }
    if given.level_bits() == active.to_bits() {
        return Err(CounterfactualIdRefusal::invalid_query(
            "the counterfactual and observed treatment levels must differ (x = x' is the \
             observed conditional)",
        ));
    }
    Ok(CounterfactualIdShape::EffectOnTreated {
        treatment: treatment.raw(),
        active,
        observed: given.level(),
        outcome: event.variable().raw(),
        outcome_level: event.level(),
    })
}

fn record_is_connected(graph: &CounterfactualGraphRecord) -> bool {
    let n = graph.nodes.len();
    n != 0
        && components_by(n, |i| {
            graph.bidirected.iter().filter_map(move |&(a, b)| match (a == i, b == i) {
                (true, _) => Some(b),
                (_, true) => Some(a),
                _ => None,
            })
        })
        .len()
            == 1
}

/// Union-find root with path halving.
fn find(root: &mut [usize], mut i: usize) -> usize {
    while root[i] != i {
        root[i] = root[root[i]];
        i = root[i];
    }
    i
}

fn hedge_obstruction(
    problem: &CounterfactualIdProblem,
    x: &BitSet,
    y: &BitSet,
    hedge: &HedgeCertificate,
) -> CounterfactualObstruction {
    let raw = |set: &BitSet| set.to_dense_ids().iter().map(|d| d.raw()).collect::<Vec<_>>();
    CounterfactualObstruction::Hedge {
        node_count: problem.node_count(),
        directed: problem.directed.clone(),
        bidirected: problem.bidirected.clone(),
        treatments: raw(x),
        outcomes: raw(y),
        f: hedge.f.iter().map(|v| v.raw()).collect(),
        f_prime: hedge.f_prime.iter().map(|v| v.raw()).collect(),
    }
}

/// The identification error a shared-budget stop surfaces as inside ID.
const fn stop_error(stop: SearchStop) -> IdentificationError {
    match stop {
        SearchStop::Cancelled => IdentificationError::Cancelled,
        SearchStop::Memory => {
            IdentificationError::budget(crate::error::IdentificationBudget::Memory)
        }
        SearchStop::Operations | SearchStop::Depth => {
            IdentificationError::budget(crate::error::IdentificationBudget::Steps)
        }
    }
}

mod engine;
#[cfg(test)]
use engine::Atom;
use engine::{Engine, Halt};
mod functional;
pub use functional::{CfSymbol, CfTerm, CfValue, CounterfactualFunctional};

mod evaluation;
pub use evaluation::CounterfactualIdPoint;
#[doc(hidden)]
pub use evaluation::evaluate_counterfactual_functional;

#[cfg(test)]
mod tests;
