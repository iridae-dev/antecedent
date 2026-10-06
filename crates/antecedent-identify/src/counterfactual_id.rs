//! Counterfactual identification on a bounded finite-discrete ADMG: the effect
//! of treatment on the treated from the observational joint (2.2B, X8).
//!
//! # What is licensed
//!
//! One query shape, `P(Y_x = y | X = x')` with `x != x'` (and, through every
//! level of `Y`, `E[Y_x | X = x'] - E[Y | X = x']`), on an explicit ADMG of at
//! most [`COUNTERFACTUAL_ID_MAX_VARIABLES`] variables with at most
//! [`COUNTERFACTUAL_ID_MAX_LEVELS`] levels each, from the observational joint
//! `P(V)`. The numerator `P(Y_x = y, X = x')` is the two-event conjunction
//! ID* (Shpitser and Pearl 2007/2008) decides; the division by `P(X = x')`
//! happens outside the algorithm. A conjunction that contradicts itself within
//! one world (a variable read at two levels in one world, or read at a level
//! its own world sets it away from) has probability zero in every model and is
//! returned as an exact zero, not refused. Every other shape is refused as
//! outside the contract, and a world that routes edges (a path-specific query)
//! is deferred to 2.3.
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

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, ExecutionContext, ExogenousCoupling, NodeRef,
    SearchBudget, SearchLimits, SearchReceipt, SearchStop, Value, VariableId,
};
use antecedent_expr::{
    Assignment, CausalExprArena, DomainRef, EmpiricalTableProvider, EvalContext, ExactDiscreteLaw,
    ExprId, ExprNode, FactorSpec,
};
use antecedent_graph::{Admg, BitSet, DenseNodeId, GraphWorkspace};
use serde::{Deserialize, Serialize};

use crate::error::IdentificationError;
use crate::hedge::HedgeCertificate;
use crate::id::{MeteredId, identify_interventional_metered};
use crate::prepared::PreparedAdmg;
use crate::sid::{SearchCharge, SharedSearch};

/// Most observed variables the contract covers.
pub const COUNTERFACTUAL_ID_MAX_VARIABLES: usize = 6;
/// Most levels one variable may have.
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
        Self::plain("invalid_argument", "counterfactual_id.invalid_query", message.into())
    }

    /// A conditional the functional evaluates is on a zero-mass event.
    #[must_use]
    pub fn positivity(message: impl Into<String>) -> Self {
        Self::plain("invalid_argument", "counterfactual_id.positivity_violation", message.into())
    }

    /// A declared bound is exceeded.
    #[must_use]
    pub fn bounds(message: impl Into<String>) -> Self {
        Self::plain("route_not_supported", "counterfactual_id.bounds_exceeded", message.into())
    }

    /// A structure outside the graph contract.
    #[must_use]
    pub fn graph(message: impl Into<String>) -> Self {
        Self::plain("cell_not_licensed", "counterfactual_id.graph_outside_contract", message.into())
    }

    /// An interval or Bayesian request.
    #[must_use]
    pub fn interval_requested() -> Self {
        Self::plain(
            "estimator_inference_mismatch",
            "counterfactual_id.interval_requested",
            "the counterfactual identification cell is a point claim; no interval or posterior \
             is published"
                .into(),
        )
    }

    fn outside_contract(message: impl Into<String>) -> Self {
        Self::plain(
            "route_not_supported",
            "counterfactual_id.query_outside_contract",
            message.into(),
        )
    }

    fn path_specific() -> Self {
        Self::plain(
            "route_not_supported",
            "counterfactual_id.path_specific_deferred",
            "a world routes edges: path-specific and recanting-district effects on ADMGs are \
             deferred to 2.3"
                .into(),
        )
    }

    fn conflict(obstruction: CounterfactualObstruction) -> Self {
        Self {
            code: "route_not_supported",
            detail: "counterfactual_id.conflicting_subscripts",
            message: obstruction.describe(),
            obstruction: Some(Box::new(obstruction)),
            receipt: None,
        }
    }

    fn hedge(obstruction: CounterfactualObstruction) -> Self {
        Self {
            code: "route_not_supported",
            detail: "counterfactual_id.counterfactual_hedge",
            message: obstruction.describe(),
            obstruction: Some(Box::new(obstruction)),
            receipt: None,
        }
    }

    fn budget(receipt: SearchReceipt) -> Self {
        Self {
            code: "transport_budget_cancel",
            detail: "counterfactual_id.budget",
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CounterfactualIdProblem {
    levels: Vec<Vec<u64>>,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
}

impl CounterfactualIdProblem {
    /// A problem over variables `0..levels.len()`.
    ///
    /// # Errors
    ///
    /// `counterfactual_id.bounds_exceeded` beyond 6 variables or 4 levels,
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
        if n > COUNTERFACTUAL_ID_MAX_VARIABLES {
            return Err(CounterfactualIdRefusal::bounds(format!(
                "the graph has {n} variables; the contract covers at most \
                 {COUNTERFACTUAL_ID_MAX_VARIABLES}"
            )));
        }
        let mut canonical_levels = Vec::with_capacity(n);
        for (variable, list) in levels.into_iter().enumerate() {
            if list.len() > COUNTERFACTUAL_ID_MAX_LEVELS {
                return Err(CounterfactualIdRefusal::bounds(format!(
                    "variable {variable} has {} levels; the contract covers at most \
                     {COUNTERFACTUAL_ID_MAX_LEVELS}",
                    list.len()
                )));
            }
            let bits: Vec<u64> = list.iter().map(|x| (x + 0.0).to_bits()).collect();
            let distinct: BTreeSet<u64> = bits.iter().copied().collect();
            if list.is_empty()
                || list.iter().any(|x| !x.is_finite())
                || distinct.len() != bits.len()
            {
                return Err(CounterfactualIdRefusal::invalid_query(format!(
                    "variable {variable} needs one to four distinct finite levels"
                )));
            }
            canonical_levels.push(bits);
        }
        let bound = u32::try_from(n).unwrap_or(u32::MAX);
        let valid = |&(a, b): &(u32, u32)| a < bound && b < bound && a != b;
        if !directed.iter().all(valid) || !bidirected.iter().all(valid) {
            return Err(CounterfactualIdRefusal::invalid_query(
                "an edge names an unknown variable or is a self-loop",
            ));
        }
        let mut directed = directed.to_vec();
        directed.sort_unstable();
        directed.dedup();
        let mut bidirected: Vec<(u32, u32)> =
            bidirected.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        bidirected.sort_unstable();
        bidirected.dedup();
        let problem = Self { levels: canonical_levels, directed, bidirected };
        if problem.topological_order().is_none() {
            return Err(CounterfactualIdRefusal::graph("the directed edges contain a cycle"));
        }
        Ok(problem)
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
            let id = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
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

    fn topological_order(&self) -> Option<Vec<u32>> {
        crate::cross_world::topological_order(self.levels.len(), &self.directed)
    }

    fn confounded(&self, a: u32, b: u32) -> bool {
        self.bidirected.contains(&(a.min(b), a.max(b)))
    }

    fn prepared(&self) -> Result<PreparedAdmg, IdentificationError> {
        let mut admg = Admg::empty();
        for i in 0..self.levels.len() {
            admg.add_node(NodeRef::Static(VariableId::from_raw(u32::try_from(i).unwrap_or(0))))?;
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

/// A level in the counterfactual functional: a literal level (IEEE bits) or a
/// summation symbol bound by an enclosing [`CounterfactualFunctional::Sum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CfValue {
    /// A literal level.
    Level(u64),
    /// Summation symbol id.
    Symbol(u32),
}

impl CfValue {
    fn render(self) -> String {
        match self {
            Self::Level(bits) => format!("{}", f64::from_bits(bits)),
            Self::Symbol(id) => format!("s{id}"),
        }
    }
}

/// One summation symbol and the variable whose levels it ranges over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CfSymbol {
    /// Symbol id, unique within a derivation.
    pub id: u32,
    /// Variable whose levels the symbol ranges over.
    pub variable: u32,
}

/// One district term `P_{intervention}(event)`, identified from `P(V)` by ID.
#[derive(Clone, Debug, PartialEq)]
pub struct CfTerm {
    /// `(variable, level)` set by the intervention, sorted.
    pub intervention: Vec<(u32, CfValue)>,
    /// `(variable, level)` of the event, sorted.
    pub event: Vec<(u32, CfValue)>,
    /// Arena holding the ID functional; the intervened variables are left free.
    pub arena: CausalExprArena,
    /// Root of the ID functional.
    pub expression: ExprId,
}

/// The identified counterfactual functional.
#[derive(Clone, Debug, PartialEq)]
pub enum CounterfactualFunctional {
    /// Probability one (an empty conjunction).
    One,
    /// Probability zero (a contradictory conjunction).
    Zero,
    /// Sum over the symbols' levels of the product of the factors.
    Sum {
        /// Symbols summed.
        symbols: Vec<CfSymbol>,
        /// Factors multiplied.
        factors: Vec<CounterfactualFunctional>,
    },
    /// A district term.
    Term(Box<CfTerm>),
    /// A binary treatment's consistency complement,
    /// `P(Y_x = y, X = x') = P_x(Y = y) - P(Y = y, X = x)`: consistency gives
    /// `P(Y_x = y, X = x) = P(Y = y, X = x)` and `X = x'` is the complement of
    /// `X = x`. Used only when ID* does not identify the conjunction and ID
    /// identifies `P_x(Y)`.
    ConsistencyComplement {
        /// `P_x(Y = y)`, identified from `P(V)` by ID.
        interventional: Box<CfTerm>,
        /// `(variable, level)` of the observed subtrahend `P(Y = y, X = x)`,
        /// sorted.
        observed: Vec<(u32, CfValue)>,
    },
}

fn render_term(term: &CfTerm, out: &mut String) {
    out.push_str("P[do(");
    for (v, x) in &term.intervention {
        out.push_str(&format!("v{v}={},", x.render()));
    }
    out.push_str(")](");
    for (v, x) in &term.event {
        out.push_str(&format!("v{v}={},", x.render()));
    }
    out.push_str(") := ");
    out.push_str(&term.arena.pretty(term.expression));
}

impl CounterfactualFunctional {
    fn render(&self, out: &mut String) {
        match self {
            Self::One => out.push('1'),
            Self::Zero => out.push('0'),
            Self::Sum { symbols, factors } => {
                out.push_str("sum[");
                for s in symbols {
                    out.push_str(&format!("s{}:v{},", s.id, s.variable));
                }
                out.push_str("](");
                for f in factors {
                    f.render(out);
                    out.push_str(" * ");
                }
                out.push(')');
            }
            Self::Term(term) => render_term(term, out),
            Self::ConsistencyComplement { interventional, observed } => {
                out.push_str("complement(");
                render_term(interventional, out);
                out.push_str(" - P(");
                for (v, x) in observed {
                    out.push_str(&format!("v{v}={},", x.render()));
                }
                out.push_str("))");
            }
        }
    }

    fn terms<'a>(&'a self, out: &mut Vec<&'a CfTerm>) {
        match self {
            Self::One | Self::Zero => {}
            Self::Sum { factors, .. } => factors.iter().for_each(|f| f.terms(out)),
            Self::Term(term) => out.push(term),
            Self::ConsistencyComplement { interventional, .. } => out.push(interventional),
        }
    }

    /// Whether the functional answers through the binary consistency
    /// complement rather than ID* alone.
    #[must_use]
    pub fn uses_consistency_complement(&self) -> bool {
        match self {
            Self::One | Self::Zero | Self::Term(_) => false,
            Self::Sum { factors, .. } => factors.iter().any(Self::uses_consistency_complement),
            Self::ConsistencyComplement { .. } => true,
        }
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

/// What the licensed query is.
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
            out.push_str(&format!("level {bits:016x}: "));
            functional.render(&mut out);
            out.push('\n');
        }
        if let Some(graph) = &self.counterfactual_graph {
            out.push_str(&format!("cg={graph:?}"));
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
        parents: (0..problem.node_count())
            .map(|v| problem.parents(u32::try_from(v).unwrap_or(0)))
            .collect(),
        topo: problem.topological_order().unwrap_or_default(),
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

/// `var_{sub} = value`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Atom {
    var: u32,
    value: CfValue,
    sub: BTreeMap<u32, CfValue>,
}

/// Why a derivation stopped.
enum Halt {
    Stop(SearchStop),
    Obstruction(Box<CounterfactualObstruction>),
    Failure(String),
}

/// A parent of a parallel-worlds node: fixed by the world, or a node.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParentRef {
    Fixed(CfValue),
    Node(usize),
}

/// A node of the counterfactual graph (a merged class of parallel-world nodes).
#[derive(Clone, Debug)]
struct CgNode {
    var: u32,
    value: Option<CfValue>,
    eff_sub: BTreeMap<u32, CfValue>,
    parents: Vec<usize>,
}

struct Cg {
    nodes: Vec<CgNode>,
    bidirected: Vec<Vec<usize>>,
}

impl Cg {
    fn districts(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.nodes.len()];
        let mut out = Vec::new();
        for start in 0..self.nodes.len() {
            if seen[start] {
                continue;
            }
            let mut district = Vec::new();
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(i) = stack.pop() {
                district.push(i);
                for &j in &self.bidirected[i] {
                    if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            district.sort_unstable();
            out.push(district);
        }
        out
    }

    fn ancestors(&self, node: usize) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        let mut stack = self.nodes[node].parents.clone();
        while let Some(p) = stack.pop() {
            if out.insert(p) {
                stack.extend(self.nodes[p].parents.iter().copied());
            }
        }
        out
    }

    fn record(&self) -> CounterfactualGraphRecord {
        let mut bidirected = Vec::new();
        for (i, adj) in self.bidirected.iter().enumerate() {
            bidirected.extend(adj.iter().filter(|&&j| j > i).map(|&j| (i, j)));
        }
        bidirected.sort_unstable();
        CounterfactualGraphRecord {
            nodes: self
                .nodes
                .iter()
                .map(|n| CounterfactualNodeRecord {
                    variable: n.var,
                    value: n.value,
                    subscript: n.eff_sub.iter().map(|(v, x)| (*v, *x)).collect(),
                    parents: n.parents.clone(),
                })
                .collect(),
            bidirected,
            districts: self.districts(),
        }
    }
}

struct Engine<'p, 'c> {
    problem: &'p CounterfactualIdProblem,
    prepared: PreparedAdmg,
    parents: Vec<Vec<u32>>,
    topo: Vec<u32>,
    search: SharedSearch<'c>,
    next_symbol: u32,
    /// Live bytes of the finished terms (their arenas), charged with every step.
    retained_bytes: u64,
    first_graph: Option<CounterfactualGraphRecord>,
    workspace: GraphWorkspace,
}

fn event_bytes(event: &[Atom]) -> u64 {
    event.iter().map(|a| 48 + 24 * a.sub.len() as u64).sum()
}

impl Engine<'_, '_> {
    fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), Halt> {
        self.search.charge(depth, self.retained_bytes.saturating_add(bytes)).map_err(Halt::Stop)
    }

    fn id_star(
        &mut self,
        event: Vec<Atom>,
        depth: usize,
    ) -> Result<CounterfactualFunctional, Halt> {
        self.charge(depth, event_bytes(&event))?;
        // Line 1.
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        // Line 2: an event its own subscript sets away from its level.
        for atom in &event {
            if let Some(set) = atom.sub.get(&atom.var) {
                match (set, atom.value) {
                    (CfValue::Level(a), CfValue::Level(b)) if *a != b => {
                        return Ok(CounterfactualFunctional::Zero);
                    }
                    (a, b) if *a == b => {}
                    _ => {
                        return Err(Halt::Failure("a self-set event with an unknown level".into()));
                    }
                }
            }
        }
        // Line 3: drop events their own subscript sets to their level.
        let event: Vec<Atom> =
            event.into_iter().filter(|a| a.sub.get(&a.var) != Some(&a.value)).collect();
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        // Line 4.
        let Some(cg) = self.make_cg(&event, depth)? else {
            return Ok(CounterfactualFunctional::Zero);
        };
        if depth == 0 && self.first_graph.is_none() {
            self.first_graph = Some(cg.record());
        }
        let districts = cg.districts();
        if districts.len() > 1 {
            return self.line_five(&cg, &districts, depth);
        }
        self.single_district(&cg, depth)
    }

    /// Line 5: sum over free nodes of the product of district terms.
    fn line_five(
        &mut self,
        cg: &Cg,
        districts: &[Vec<usize>],
        depth: usize,
    ) -> Result<CounterfactualFunctional, Halt> {
        let mut symbols = Vec::new();
        let values: Vec<CfValue> = cg
            .nodes
            .iter()
            .map(|node| {
                node.value.unwrap_or_else(|| {
                    let id = self.next_symbol;
                    self.next_symbol += 1;
                    symbols.push(CfSymbol { id, variable: node.var });
                    CfValue::Symbol(id)
                })
            })
            .collect();
        let mut factors = Vec::with_capacity(districts.len());
        for district in districts {
            let inside: BTreeSet<usize> = district.iter().copied().collect();
            let outside_parents: BTreeSet<usize> = district
                .iter()
                .flat_map(|&c| cg.nodes[c].parents.iter().copied())
                .filter(|p| !inside.contains(p))
                .collect();
            let mut atoms = Vec::with_capacity(district.len());
            for &c in district {
                let mut sub = cg.nodes[c].eff_sub.clone();
                // ID* sets every node of the district to the values of all nodes
                // outside it; restricting that to the node's ancestors is an exact
                // simplification (a node does not depend on a non-ancestor), so
                // dropping the restriction gives an equal functional: a mutant
                // that removes it is equivalent, not an unchecked guard.
                let ancestors = cg.ancestors(c);
                for &p in outside_parents.intersection(&ancestors) {
                    let var = cg.nodes[p].var;
                    if let Some(previous) = sub.insert(var, values[p]) {
                        if previous != values[p] {
                            return Err(Halt::Failure(
                                "two copies of one variable among a node's ancestors".into(),
                            ));
                        }
                    }
                }
                atoms.push(Atom { var: cg.nodes[c].var, value: values[c], sub });
            }
            factors.push(self.id_star(atoms, depth + 1)?);
        }
        Ok(CounterfactualFunctional::Sum { symbols, factors })
    }

    /// Lines 6-9: one district.
    fn single_district(&mut self, cg: &Cg, depth: usize) -> Result<CounterfactualFunctional, Halt> {
        let mut sub: BTreeMap<u32, CfValue> = BTreeMap::new();
        for node in &cg.nodes {
            for (&var, &value) in &node.eff_sub {
                if let Some(previous) = sub.insert(var, value) {
                    if previous != value {
                        return Err(Halt::Obstruction(Box::new(
                            CounterfactualObstruction::ConflictingSubscripts {
                                graph: cg.record(),
                                variable: var,
                                subscript: previous,
                                event: Some(value),
                            },
                        )));
                    }
                }
            }
        }
        let mut event: BTreeMap<u32, CfValue> = BTreeMap::new();
        for node in &cg.nodes {
            if let Some(&set) = sub.get(&node.var) {
                // A node read at another level, or left free (`None`: its level is
                // summed, so it may differ), conflicts. The free case cannot occur
                // on the effect-on-the-treated shape; a synthetic conjunction in the
                // unit tests reaches it.
                if node.value != Some(set) {
                    return Err(Halt::Obstruction(Box::new(
                        CounterfactualObstruction::ConflictingSubscripts {
                            graph: cg.record(),
                            variable: node.var,
                            subscript: set,
                            event: node.value,
                        },
                    )));
                }
                continue;
            }
            let Some(value) = node.value else {
                // A free node of the district is marginalized.
                continue;
            };
            match event.insert(node.var, value) {
                Some(previous) if previous != value => {
                    // Two unmerged copies of one variable read at two levels with
                    // no subscript conflict: every parent pair that kept them apart
                    // is, at its root, a node read at exactly the level the other
                    // world fixes (anything else conflicted above), so by
                    // consistency the copies are equal and the conjunction has
                    // probability zero. Unreachable on the effect-on-the-treated
                    // shape (no variable has two copies); reached by a synthetic
                    // conjunction in the unit tests.
                    return match (previous, value) {
                        (CfValue::Level(_), CfValue::Level(_)) => {
                            Ok(CounterfactualFunctional::Zero)
                        }
                        _ => Err(Halt::Obstruction(Box::new(
                            CounterfactualObstruction::ConflictingSubscripts {
                                graph: cg.record(),
                                variable: node.var,
                                subscript: previous,
                                event: Some(value),
                            },
                        ))),
                    };
                }
                _ => {}
            }
        }
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        let n = self.problem.node_count();
        let mut y = BitSet::with_len(n);
        for &v in event.keys() {
            y.insert(DenseNodeId::from_raw(v));
        }
        let mut x = BitSet::with_len(n);
        for &v in sub.keys() {
            x.insert(DenseNodeId::from_raw(v));
        }
        let retained = self.retained_bytes;
        let search = &mut self.search;
        let mut charge = |id_depth: usize, bytes: u64| -> Result<(), IdentificationError> {
            search.charge(depth + id_depth, retained.saturating_add(bytes)).map_err(stop_error)
        };
        let outcome = identify_interventional_metered(
            &self.prepared,
            &y,
            &x,
            &mut self.workspace,
            &mut charge,
        );
        match outcome {
            Ok(MeteredId::Identified { arena, expression }) => {
                self.retained_bytes =
                    self.retained_bytes.saturating_add((arena.len() as u64).saturating_mul(96));
                Ok(CounterfactualFunctional::Term(Box::new(CfTerm {
                    intervention: sub.into_iter().collect(),
                    event: event.into_iter().collect(),
                    arena,
                    expression,
                })))
            }
            Ok(MeteredId::Hedge(hedge)) => {
                Err(Halt::Obstruction(Box::new(hedge_obstruction(self.problem, &x, &y, &hedge))))
            }
            Err(error) => Err(match error {
                IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
                    Halt::Stop(self.search.stop_of(&error))
                }
                other => Halt::Failure(other.to_string()),
            }),
        }
    }

    /// The functional of `P(Y_{X = active} = level, X = observed)`: ID* on the
    /// conjunction, and when ID* stops, the binary consistency complement (its
    /// refusal keeps the ID* obstruction).
    fn numerator(
        &mut self,
        treatment: u32,
        active: f64,
        observed: f64,
        outcome: u32,
        level: u64,
    ) -> Result<CounterfactualFunctional, Halt> {
        let event = vec![
            Atom {
                var: outcome,
                value: CfValue::Level(level),
                sub: BTreeMap::from([(treatment, CfValue::Level(active.to_bits()))]),
            },
            Atom {
                var: treatment,
                value: CfValue::Level(observed.to_bits()),
                sub: BTreeMap::new(),
            },
        ];
        match self.id_star(event, 0) {
            Err(Halt::Obstruction(obstruction)) => {
                match self.consistency_complement(treatment, active, outcome, level)? {
                    Some(functional) => Ok(functional),
                    None => Err(Halt::Obstruction(obstruction)),
                }
            }
            other => other,
        }
    }

    /// The binary-treatment consistency complement of
    /// `P(Y_{X = active} = level, X = x')`: `P_x(Y = level) - P(Y = level, X = x)`
    /// when `X` has exactly two levels (so `X = x'` is `X != x`) and complete ID
    /// identifies `P_x(Y)` from `P(V)`; `None` otherwise (the caller keeps the
    /// ID* refusal). Charged to the shared budget like every ID call.
    fn consistency_complement(
        &mut self,
        treatment: u32,
        active: f64,
        outcome: u32,
        level: u64,
    ) -> Result<Option<CounterfactualFunctional>, Halt> {
        if self.problem.levels[treatment as usize].len() != 2 {
            return Ok(None);
        }
        self.charge(0, 0)?;
        let n = self.problem.node_count();
        let mut y = BitSet::with_len(n);
        y.insert(DenseNodeId::from_raw(outcome));
        let mut x = BitSet::with_len(n);
        x.insert(DenseNodeId::from_raw(treatment));
        let retained = self.retained_bytes;
        let search = &mut self.search;
        let mut charge = |id_depth: usize, bytes: u64| -> Result<(), IdentificationError> {
            search.charge(id_depth, retained.saturating_add(bytes)).map_err(stop_error)
        };
        let outcome_id = identify_interventional_metered(
            &self.prepared,
            &y,
            &x,
            &mut self.workspace,
            &mut charge,
        );
        match outcome_id {
            Ok(MeteredId::Identified { arena, expression }) => {
                self.retained_bytes =
                    self.retained_bytes.saturating_add((arena.len() as u64).saturating_mul(96));
                let active = CfValue::Level(active.to_bits());
                let mut observed = vec![(treatment, active), (outcome, CfValue::Level(level))];
                observed.sort_unstable();
                Ok(Some(CounterfactualFunctional::ConsistencyComplement {
                    interventional: Box::new(CfTerm {
                        intervention: vec![(treatment, active)],
                        event: vec![(outcome, CfValue::Level(level))],
                        arena,
                        expression,
                    }),
                    observed,
                }))
            }
            Ok(MeteredId::Hedge(_)) => Ok(None),
            Err(error) => Err(match error {
                IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
                    Halt::Stop(self.search.stop_of(&error))
                }
                other => Halt::Failure(other.to_string()),
            }),
        }
    }

    /// make-cg: `None` when two events on one merged node disagree.
    #[allow(
        clippy::too_many_lines,
        reason = "the parallel-worlds construction, merge and restriction in order"
    )]
    fn make_cg(&mut self, event: &[Atom], depth: usize) -> Result<Option<Cg>, Halt> {
        // Disjoint borrows: the budget is charged while the graph's parents and
        // order are read, so neither is copied per call.
        let Engine { problem, parents: graph_parents, topo, search, retained_bytes, .. } = self;
        let problem: &CounterfactualIdProblem = problem;
        let n = problem.node_count();
        let worlds: Vec<BTreeMap<u32, CfValue>> =
            event.iter().map(|a| a.sub.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        // Parallel-world nodes: (world, variable) for every variable a world does not set.
        let mut index: HashMap<(usize, u32), usize> = HashMap::new();
        let mut pw: Vec<(usize, u32)> = Vec::new();
        for (w, sub) in worlds.iter().enumerate() {
            for v in 0..n {
                let v = u32::try_from(v).unwrap_or(0);
                if !sub.contains_key(&v) {
                    index.insert((w, v), pw.len());
                    pw.push((w, v));
                }
            }
        }
        let parent_refs = |w: usize, v: u32| -> Vec<ParentRef> {
            graph_parents[v as usize]
                .iter()
                .map(|&p| match worlds[w].get(&p) {
                    Some(&value) => ParentRef::Fixed(value),
                    None => ParentRef::Node(index[&(w, p)]),
                })
                .collect()
        };
        // Union-find, merged in topological order.
        let mut root: Vec<usize> = (0..pw.len()).collect();
        for &v in topo.iter() {
            let bytes = (pw.len() as u64).saturating_mul(64).saturating_add(event_bytes(event));
            search.charge(depth, retained_bytes.saturating_add(bytes)).map_err(Halt::Stop)?;
            let copies: Vec<usize> =
                (0..worlds.len()).filter_map(|w| index.get(&(w, v)).copied()).collect();
            for (i, &a) in copies.iter().enumerate() {
                for &b in &copies[i + 1..] {
                    let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                    if ra == rb {
                        continue;
                    }
                    let pa = parent_refs(pw[a].0, v);
                    let pb = parent_refs(pw[b].0, v);
                    // Lemma 24. Two fixed parents at different levels never meet
                    // on the effect-on-the-treated shape (one intervened world);
                    // a synthetic two-world conjunction in the unit tests does.
                    let same = pa.iter().zip(&pb).all(|(x, y)| match (x, y) {
                        (ParentRef::Node(i), ParentRef::Node(j)) => {
                            find(&mut root, *i) == find(&mut root, *j)
                        }
                        (ParentRef::Fixed(i), ParentRef::Fixed(j)) => i == j,
                        _ => false,
                    });
                    if same {
                        root[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
        // Classes, the event on them, and an inconsistent merge.
        let class_of: Vec<usize> = (0..pw.len()).map(|i| find(&mut root, i)).collect();
        let mut value_of: BTreeMap<usize, CfValue> = BTreeMap::new();
        let mut event_classes = BTreeSet::new();
        for atom in event {
            let w = worlds.iter().position(|s| *s == atom.sub).unwrap_or(0);
            let Some(&node) = index.get(&(w, atom.var)) else {
                return Err(Halt::Failure("an event on a variable its world sets".into()));
            };
            let class = class_of[node];
            event_classes.insert(class);
            match value_of.insert(class, atom.value) {
                Some(previous) if previous != atom.value => {
                    return match (previous, atom.value) {
                        (CfValue::Level(_), CfValue::Level(_)) => Ok(None),
                        _ => Err(Halt::Failure(
                            "one node read at a symbolic and another level".into(),
                        )),
                    };
                }
                _ => {}
            }
        }
        // Ancestors of the event over merged classes.
        let class_parents = |class: usize| -> Vec<usize> {
            let (w, v) = pw[class];
            let mut out: Vec<usize> = parent_refs(w, v)
                .into_iter()
                .filter_map(|p| match p {
                    ParentRef::Node(i) => Some(class_of[i]),
                    ParentRef::Fixed(_) => None,
                })
                .collect();
            out.sort_unstable();
            out.dedup();
            out
        };
        let mut keep = BTreeSet::new();
        let mut stack: Vec<usize> = event_classes.iter().copied().collect();
        while let Some(c) = stack.pop() {
            if keep.insert(c) {
                stack.extend(class_parents(c));
            }
        }
        // Canonical order: by variable, then representative world.
        let mut kept: Vec<usize> = keep.into_iter().collect();
        kept.sort_by_key(|&c| (pw[c].1, pw[c].0));
        let position: HashMap<usize, usize> =
            kept.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut nodes: Vec<CgNode> = Vec::with_capacity(kept.len());
        for &c in &kept {
            let (w, v) = pw[c];
            let parents: Vec<usize> = class_parents(c).into_iter().map(|p| position[&p]).collect();
            let fixed: BTreeMap<u32, CfValue> = graph_parents[v as usize]
                .iter()
                .filter_map(|&p| worlds[w].get(&p).map(|&x| (p, x)))
                .collect();
            nodes.push(CgNode {
                var: v,
                value: value_of.get(&c).copied(),
                eff_sub: fixed,
                parents,
            });
        }
        // Effective subscripts: fixed parents of a node and of its ancestors, in
        // topological order of the variables (a parent's variable precedes).
        let order: Vec<usize> = {
            let rank: HashMap<u32, usize> = topo.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            let mut o: Vec<usize> = (0..nodes.len()).collect();
            o.sort_by_key(|&i| rank[&nodes[i].var]);
            o
        };
        for &i in &order {
            let mut sub = nodes[i].eff_sub.clone();
            for &p in &nodes[i].parents.clone() {
                for (&var, &value) in &nodes[p].eff_sub {
                    if let Some(previous) = sub.insert(var, value) {
                        if previous != value {
                            return Err(Halt::Failure(
                                "a node's ancestors are set to two levels of one variable".into(),
                            ));
                        }
                    }
                }
            }
            nodes[i].eff_sub = sub;
        }
        // Bidirected: two copies of one variable share its exogenous term; a
        // bidirected edge of the graph joins every pair of copies. (No variable
        // has two copies on the effect-on-the-treated shape; the unit tests
        // reach the copy edge with a synthetic two-world conjunction.)
        let mut bidirected = vec![Vec::new(); nodes.len()];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                let (a, b) = (nodes[i].var, nodes[j].var);
                if a == b || problem.confounded(a, b) {
                    bidirected[i].push(j);
                    bidirected[j].push(i);
                }
            }
        }
        Ok(Some(Cg { nodes, bidirected }))
    }
}

fn record_is_connected(graph: &CounterfactualGraphRecord) -> bool {
    let n = graph.nodes.len();
    if n == 0 {
        return false;
    }
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(i) = stack.pop() {
        for &(a, b) in &graph.bidirected {
            let next = if a == i {
                b
            } else if b == i {
                a
            } else {
                continue;
            };
            if next < n && !seen[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    seen.into_iter().all(|s| s)
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

/// The exact point of an identified derivation on one joint law.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualIdPoint {
    /// `P(given)`, the conditioning probability (`P(X = x')`).
    pub conditioning_probability: f64,
    /// `(outcome level, P(Y_x = level, X = x'))` in declared level order; empty
    /// for an exact zero.
    pub numerators: Vec<(f64, f64)>,
    /// The requested probability `P(event | given)`.
    pub probability: f64,
    /// `E[Y_x | X = x']`; `None` for an exact zero.
    pub counterfactual_mean: Option<f64>,
    /// `E[Y | X = x']`; `None` for an exact zero.
    pub observed_mean: Option<f64>,
    /// `E[Y_x | X = x'] - E[Y | X = x']`; `None` for an exact zero.
    pub effect: Option<f64>,
}

/// A joint law laid out in the problem's variable and level order.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DenseJoint {
    /// Levels per variable (values).
    pub levels: Vec<Vec<f64>>,
    /// Row-major probabilities, last variable fastest.
    pub probabilities: Vec<f64>,
}

impl DenseJoint {
    /// Marginal table over `vars` (row-major in the given order, last fastest).
    #[must_use]
    pub(crate) fn marginal(&self, vars: &[usize]) -> Vec<f64> {
        let cards: Vec<usize> = self.levels.iter().map(Vec::len).collect();
        let size: usize = vars.iter().map(|&v| cards[v]).product();
        let mut out = vec![0.0; size];
        let mut index = vec![0usize; cards.len()];
        for p in &self.probabilities {
            let mut flat = 0usize;
            for &v in vars {
                flat = flat * cards[v] + index[v];
            }
            out[flat] += p;
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        out
    }

    /// Mass of the cells where `fixed` variables take the given level indices.
    #[must_use]
    pub(crate) fn mass(&self, fixed: &[(usize, usize)]) -> f64 {
        let cards: Vec<usize> = self.levels.iter().map(Vec::len).collect();
        let mut total = 0.0;
        let mut index = vec![0usize; cards.len()];
        for p in &self.probabilities {
            if fixed.iter().all(|&(v, l)| index[v] == l) {
                total += p;
            }
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        total
    }
}

/// Lay `law` out in `problem`'s variable and level order.
///
/// # Errors
///
/// `counterfactual_id.invalid_query` when the law's axes are not exactly the
/// problem's variables with the problem's levels (in any order).
#[doc(hidden)]
pub fn dense_joint(
    problem: &CounterfactualIdProblem,
    law: &ExactDiscreteLaw,
) -> Result<DenseJoint, CounterfactualIdRefusal> {
    let n = problem.node_count();
    let axes = law.axes();
    if axes.len() != n || !law.interventions().is_empty() {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "the law has {} axes (and {} interventions) for an observational problem of {n} \
             variables",
            axes.len(),
            law.interventions().len()
        )));
    }
    // axis_of[v] = axis index; level_map[v][law level index] = problem level index.
    let mut axis_of = vec![usize::MAX; n];
    let mut level_map = vec![Vec::new(); n];
    for (a, axis) in axes.iter().enumerate() {
        let v = axis.variable.raw() as usize;
        if v >= n || axis_of[v] != usize::MAX {
            return Err(CounterfactualIdRefusal::invalid_query(
                "the law's axes are not the problem's variables",
            ));
        }
        axis_of[v] = a;
        let mut map = Vec::with_capacity(axis.values.len());
        for value in axis.values.iter() {
            let bits = value.as_f64().map(|x| (x + 0.0).to_bits());
            let Some(index) =
                bits.and_then(|b| problem.level_index(u32::try_from(v).unwrap_or(u32::MAX), b))
            else {
                return Err(CounterfactualIdRefusal::invalid_query(format!(
                    "the law's levels of variable {v} are not the prepared levels"
                )));
            };
            map.push(index);
        }
        if map.len() != problem.levels[v].len() {
            return Err(CounterfactualIdRefusal::invalid_query(format!(
                "the law's levels of variable {v} are not the prepared levels"
            )));
        }
        level_map[v] = map;
    }
    let cards: Vec<usize> = (0..n).map(|v| problem.levels[v].len()).collect();
    let mut probabilities = vec![0.0; cards.iter().product()];
    let law_cards: Vec<usize> = axes.iter().map(|a| a.values.len()).collect();
    let mut law_index = vec![0usize; axes.len()];
    for &p in law.probabilities() {
        let mut flat = 0usize;
        for v in 0..n {
            flat = flat * cards[v] + level_map[v][law_index[axis_of[v]]];
        }
        probabilities[flat] = p;
        for k in (0..law_cards.len()).rev() {
            law_index[k] += 1;
            if law_index[k] < law_cards[k] {
                break;
            }
            law_index[k] = 0;
        }
    }
    Ok(DenseJoint { levels: (0..n).map(|v| problem.levels(v)).collect(), probabilities })
}

/// Evaluate `derivation` on `law`: each ID term through the existing compiled
/// evaluator over factor tables built from the joint, composed by the
/// functional's sums and products.
///
/// # Errors
///
/// `counterfactual_id.invalid_query` for a law that does not match the problem
/// or a zero-probability conditioning event, `counterfactual_id.positivity_violation`
/// when a conditional the functional evaluates is on a zero-mass event.
#[doc(hidden)]
pub fn evaluate_counterfactual_functional(
    problem: &CounterfactualIdProblem,
    derivation: &CounterfactualIdDerivation,
    law: &ExactDiscreteLaw,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdPoint, CounterfactualIdRefusal> {
    let joint = dense_joint(problem, law)?;
    let given: Vec<(usize, usize)> = derivation
        .given
        .iter()
        .map(|&(v, bits)| {
            problem
                .level_index(v, bits)
                .map(|l| (v as usize, l))
                .ok_or_else(|| CounterfactualIdRefusal::invalid_query("an unknown given level"))
        })
        .collect::<Result<_, _>>()?;
    let conditioning = joint.mass(&given);
    if conditioning <= 0.0 {
        return Err(CounterfactualIdRefusal::invalid_query(
            "the conditioning event has probability zero in the supplied law",
        ));
    }
    let CounterfactualIdShape::EffectOnTreated { outcome, outcome_level, .. } = derivation.shape
    else {
        return Ok(CounterfactualIdPoint {
            conditioning_probability: conditioning,
            numerators: Vec::new(),
            probability: 0.0,
            counterfactual_mean: None,
            observed_mean: None,
            effect: None,
        });
    };
    let mut evaluator = TermEvaluator::new(&joint);
    let mut numerators = Vec::with_capacity(derivation.numerators.len());
    for (bits, functional) in &derivation.numerators {
        let mut symbols = HashMap::new();
        let value = evaluator.functional(functional, &mut symbols, ctx)?;
        numerators.push((f64::from_bits(*bits), value));
    }
    // A cheap check of the derivation on this law: the numerators over every
    // level of Y are P(Y_x = y, X = x') for all y, so they sum to P(X = x').
    let total: f64 = numerators.iter().map(|(_, p)| p).sum();
    if !total.is_finite()
        || (total - conditioning).abs() > NUMERATOR_SUM_TOLERANCE * (1.0 + conditioning)
    {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "the derivation failed its check on this law: the numerators sum to {total}, not \
             P(X = x') = {conditioning}"
        )));
    }
    let probability = numerators
        .iter()
        .find(|(level, _)| level.to_bits() == outcome_level.to_bits())
        .map_or(f64::NAN, |(_, p)| p / conditioning);
    let counterfactual_mean =
        numerators.iter().map(|(level, p)| level * p).sum::<f64>() / conditioning;
    let observed_mean = (0..joint.levels[outcome as usize].len())
        .map(|l| {
            let mut fixed = given.clone();
            fixed.push((outcome as usize, l));
            joint.levels[outcome as usize][l] * joint.mass(&fixed)
        })
        .sum::<f64>()
        / conditioning;
    if !probability.is_finite() || !counterfactual_mean.is_finite() {
        return Err(CounterfactualIdRefusal::positivity(
            "the functional is not finite on this law",
        ));
    }
    Ok(CounterfactualIdPoint {
        conditioning_probability: conditioning,
        numerators,
        probability,
        counterfactual_mean: Some(counterfactual_mean),
        observed_mean: Some(observed_mean),
        effect: Some(counterfactual_mean - observed_mean),
    })
}

/// Evaluates district terms through the compiled evaluator, one provider per
/// term and null-event extension built from the joint.
struct TermEvaluator<'j> {
    joint: &'j DenseJoint,
    /// Per term: the factor tables under every null-event extension (one when
    /// the law has no zero-mass conditioning cell for the term), and the plan.
    compiled: HashMap<usize, (Vec<EmpiricalTableProvider>, antecedent_expr::CompiledEvaluator)>,
    polls: usize,
}

/// How a conditional on a zero-mass conditioning event is extended. The
/// functional is accepted only where its value is the same under every
/// extension tried (the uniform one and, for each `k` below the largest
/// cardinality, all mass on level `k` of every conditioned variable); otherwise
/// the law violates positivity for this functional. Every level is tried, so a
/// value that depends on the extension of one undefined conditional of one
/// variable is caught whatever its cardinality (the value is affine in that
/// extension, and constant on every point mass only if constant). Agreement is
/// a check, not a proof: the same extension is applied to every zero-mass
/// cell, and a joint conditional of several variables gets only the diagonal
/// point masses, so a dependence that only a different point mass per cell or
/// an off-diagonal configuration reveals is not detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum NullExtension {
    /// Every value of the conditioned variables equally likely.
    Uniform,
    /// All mass on level `k` of every conditioned variable (its last level
    /// when it has `k` or fewer).
    Level(usize),
}

impl NullExtension {
    /// The extensions tried on a law whose variables have at most
    /// `max_levels` levels.
    fn all(max_levels: usize) -> Vec<Self> {
        std::iter::once(Self::Uniform).chain((0..max_levels.max(1)).map(Self::Level)).collect()
    }
}

/// Relative tolerance of the agreement between the null-event extensions.
const EXTENSION_TOLERANCE: f64 = 1e-12;
/// Relative tolerance of the check that the numerators sum to `P(X = x')`.
const NUMERATOR_SUM_TOLERANCE: f64 = 1e-9;
/// Largest negative rounding a consistency complement may show before the law
/// is refused as incompatible with the graph.
const COMPLEMENT_TOLERANCE: f64 = 1e-12;

impl<'j> TermEvaluator<'j> {
    fn new(joint: &'j DenseJoint) -> Self {
        Self { joint, compiled: HashMap::new(), polls: 0 }
    }

    fn functional(
        &mut self,
        functional: &CounterfactualFunctional,
        symbols: &mut HashMap<u32, usize>,
        ctx: &ExecutionContext,
    ) -> Result<f64, CounterfactualIdRefusal> {
        match functional {
            CounterfactualFunctional::One => Ok(1.0),
            CounterfactualFunctional::Zero => Ok(0.0),
            CounterfactualFunctional::Term(term) => self.term(term, symbols, ctx),
            CounterfactualFunctional::ConsistencyComplement { interventional, observed } => {
                let interventional = self.term(interventional, symbols, ctx)?;
                let mut fixed = Vec::with_capacity(observed.len());
                for &(variable, value) in observed {
                    fixed.push((variable as usize, self.level_index(variable, value, symbols)?));
                }
                let value = interventional - self.joint.mass(&fixed);
                if value < -COMPLEMENT_TOLERANCE * (1.0 + interventional.abs()) {
                    return Err(CounterfactualIdRefusal::invalid_query(format!(
                        "the law is not compatible with the graph: the consistency complement \
                         P_x(y) - P(y, x) is {value}"
                    )));
                }
                Ok(value.max(0.0))
            }
            CounterfactualFunctional::Sum { symbols: summed, factors } => {
                let cards: Vec<usize> =
                    summed.iter().map(|s| self.joint.levels[s.variable as usize].len()).collect();
                let mut index = vec![0usize; summed.len()];
                let mut total = 0.0;
                loop {
                    for (s, &l) in summed.iter().zip(&index) {
                        symbols.insert(s.id, l);
                    }
                    let mut product = 1.0;
                    for factor in factors {
                        product *= self.functional(factor, symbols, ctx)?;
                        if product == 0.0 {
                            break;
                        }
                    }
                    total += product;
                    let mut k = summed.len();
                    loop {
                        if k == 0 {
                            for s in summed {
                                symbols.remove(&s.id);
                            }
                            return Ok(total);
                        }
                        k -= 1;
                        index[k] += 1;
                        if index[k] < cards[k] {
                            break;
                        }
                        index[k] = 0;
                    }
                }
            }
        }
    }

    fn level_index(
        &self,
        variable: u32,
        value: CfValue,
        symbols: &HashMap<u32, usize>,
    ) -> Result<usize, CounterfactualIdRefusal> {
        let levels = &self.joint.levels[variable as usize];
        match value {
            CfValue::Level(bits) => levels.iter().position(|l| l.to_bits() == bits),
            CfValue::Symbol(id) => symbols.get(&id).copied(),
        }
        .ok_or_else(|| {
            CounterfactualIdRefusal::invalid_query("a functional level is not a level of the law")
        })
    }

    fn level(
        &self,
        variable: u32,
        value: CfValue,
        symbols: &HashMap<u32, usize>,
    ) -> Result<f64, CounterfactualIdRefusal> {
        let index = self.level_index(variable, value, symbols)?;
        Ok(self.joint.levels[variable as usize][index])
    }

    fn term(
        &mut self,
        term: &CfTerm,
        symbols: &HashMap<u32, usize>,
        ctx: &ExecutionContext,
    ) -> Result<f64, CounterfactualIdRefusal> {
        self.polls += 1;
        if self.polls % 1024 == 0 && ctx.cancellation.is_cancelled() {
            return Err(CounterfactualIdRefusal::budget(SearchReceipt {
                stop: SearchStop::Cancelled,
                operations_limit: 0,
                depth_limit: 0,
                memory_limit_bytes: None,
                operations_consumed: None,
                depth_reached: None,
                explored: vec!["evaluation".into()],
                unevaluated: Vec::new(),
            }));
        }
        let key = std::ptr::from_ref(term) as usize;
        let joint = self.joint;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.compiled.entry(key) {
            let (uniform, touched_null) =
                factor_provider(joint, &term.arena, term.expression, NullExtension::Uniform);
            let mut providers = vec![uniform];
            if touched_null {
                let most = joint.levels.iter().map(Vec::len).max().unwrap_or(1);
                providers.extend(
                    NullExtension::all(most)
                        .into_iter()
                        .skip(1)
                        .map(|fill| factor_provider(joint, &term.arena, term.expression, fill).0),
                );
            }
            let compiled = term.arena.compile(term.expression).map_err(|e| {
                CounterfactualIdRefusal::invalid_query(format!("the term does not compile: {e}"))
            })?;
            slot.insert((providers, compiled));
        }
        let mut env = Vec::new();
        for &(v, value) in term.intervention.iter().chain(&term.event) {
            env.push((VariableId::from_raw(v), Value::f64(self.level(v, value, symbols)?)));
        }
        let bound: BTreeSet<VariableId> = env.iter().map(|(v, _)| *v).collect();
        let extra: Vec<VariableId> = term
            .arena
            .free_variables(term.expression)
            .into_iter()
            .filter(|v| !bound.contains(v))
            .collect();
        let (providers, compiled) = &self.compiled[&key];
        // A variable the functional leaves free beyond the treatments and the
        // event (ID line 3 then 7) does not change its value wherever the
        // conditionals exist: the first level where the term is defined is used.
        let cards: Vec<usize> =
            extra.iter().map(|v| self.joint.levels[v.raw() as usize].len()).collect();
        let mut index = vec![0usize; extra.len()];
        let mut last_error = String::new();
        loop {
            let mut full = env.clone();
            for (v, &l) in extra.iter().zip(&index) {
                full.push((*v, Value::f64(self.joint.levels[v.raw() as usize][l])));
            }
            let assignment = Assignment::from_pairs(full);
            let value = |provider: &EmpiricalTableProvider| {
                compiled.evaluate_with(&term.arena, provider, &EvalContext::default(), &assignment)
            };
            let values: Result<Vec<f64>, _> = providers.iter().map(value).collect();
            match values {
                Ok(values) => {
                    let first = values[0];
                    if values.iter().all(|&b| {
                        b.is_finite()
                            && (first - b).abs()
                                <= EXTENSION_TOLERANCE * (1.0 + first.abs().max(b.abs()))
                    }) {
                        return Ok(first);
                    }
                    "the value depends on how a conditional on a zero-mass event is extended"
                        .clone_into(&mut last_error);
                }
                Err(e) => last_error = e.to_string(),
            }
            let mut k = extra.len();
            loop {
                if k == 0 {
                    return Err(CounterfactualIdRefusal::positivity(format!(
                        "a district term is undefined on this law: {last_error}"
                    )));
                }
                k -= 1;
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
    }
}

/// Factor tables of every distribution the term's functional reads: the
/// conditional of the joint on every conditioning assignment with positive
/// mass, and `fill` on a zero-mass one. The flag says whether any zero-mass
/// conditioning assignment was met (otherwise every fill gives these tables).
fn factor_provider(
    joint: &DenseJoint,
    arena: &CausalExprArena,
    root: ExprId,
    fill: NullExtension,
) -> (EmpiricalTableProvider, bool) {
    let mut touched_null = false;
    let mut provider = EmpiricalTableProvider::new();
    for (v, levels) in joint.levels.iter().enumerate() {
        provider.set_domain(
            VariableId::from_raw(u32::try_from(v).unwrap_or(0)),
            levels.iter().map(|x| Value::f64(*x)),
        );
    }
    let mut seen = BTreeSet::new();
    let mut stack = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let vars = arena.var_set(*variables).to_vec();
                let cond = arena.var_set(*conditioned_on).to_vec();
                if seen.insert((vars.clone(), cond.clone())) {
                    touched_null |= insert_conditional(&mut provider, joint, &vars, &cond, fill);
                }
            }
            ExprNode::Product(list) => stack.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => stack.push(*expr),
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Kernel { body, .. } => stack.push(*body),
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    (provider, touched_null)
}

fn insert_conditional(
    provider: &mut EmpiricalTableProvider,
    joint: &DenseJoint,
    vars: &[VariableId],
    cond: &[VariableId],
    fill: NullExtension,
) -> bool {
    let mut touched_null = false;
    let all: Vec<usize> = vars.iter().chain(cond).map(|v| v.raw() as usize).collect();
    let cond_dense: Vec<usize> = cond.iter().map(|v| v.raw() as usize).collect();
    let joint_table = joint.marginal(&all);
    let cond_table = joint.marginal(&cond_dense);
    let cond_size = cond_table.len();
    let spec = FactorSpec {
        variables: vars,
        conditioned_on: cond,
        intervention: &[],
        domain: DomainRef::Observational,
        population: "",
        regime: None,
    };
    let cards: Vec<usize> = all.iter().map(|&v| joint.levels[v].len()).collect();
    let mut index = vec![0usize; all.len()];
    for (flat, &numerator) in joint_table.iter().enumerate() {
        // The conditioning variables are the trailing axes of `all`.
        let denominator = cond_table[flat % cond_size.max(1)];
        let value = if denominator > 0.0 {
            numerator / denominator
        } else {
            touched_null = true;
            match fill {
                // One over the number of configurations of the conditioned variables.
                NullExtension::Uniform => 1.0 / (joint_table.len() / cond_size.max(1)) as f64,
                // All mass on level `k` (clamped to the last level) of each.
                NullExtension::Level(k) => {
                    if index[..vars.len()]
                        .iter()
                        .zip(&cards)
                        .all(|(&l, &card)| l == k.min(card - 1))
                    {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        };
        {
            let assignment = Assignment::from_pairs(all.iter().zip(&index).map(|(&v, &l)| {
                (
                    VariableId::from_raw(u32::try_from(v).unwrap_or(0)),
                    Value::f64(joint.levels[v][l]),
                )
            }));
            // The key is built from the spec's own variables, which are all bound.
            let _ = provider.insert_probability(&spec, &assignment, value);
        }
        for k in (0..all.len()).rev() {
            index[k] += 1;
            if index[k] < cards[k] {
                break;
            }
            index[k] = 0;
        }
    }
    touched_null
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine<'p, 'c>(
        problem: &'p CounterfactualIdProblem,
        ctx: &'c ExecutionContext,
    ) -> Engine<'p, 'c> {
        let budget = SearchBudget::with_memory(
            COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            COUNTERFACTUAL_ID_MEMORY_BYTES,
            ctx,
        )
        .unwrap();
        Engine {
            problem,
            prepared: problem.prepared().unwrap(),
            parents: (0..problem.node_count())
                .map(|v| problem.parents(u32::try_from(v).unwrap()))
                .collect(),
            topo: problem.topological_order().unwrap(),
            search: SharedSearch::new(budget),
            next_symbol: 0,
            retained_bytes: 0,
            first_graph: None,
            workspace: GraphWorkspace::default(),
        }
    }

    fn level(x: f64) -> CfValue {
        CfValue::Level(x.to_bits())
    }

    /// The effect-on-the-treated shape never reaches an ID hedge (the sweep in
    /// `tests/counterfactual_id_search.rs` records none), so the hedge branch is
    /// exercised on the engine directly: `P(Y_x = 1)` on the bow arc is one
    /// district `{Y_x}` whose term `P_x(y)` ID refuses with a hedge.
    #[test]
    fn an_id_hedge_on_a_district_term_is_a_verifiable_obstruction() {
        let problem =
            CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 2], &[(0, 1)], &[(0, 1)]).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut engine = engine(&problem, &ctx);
        let event =
            vec![Atom { var: 1, value: level(1.0), sub: BTreeMap::from([(0, level(1.0))]) }];
        let Err(Halt::Obstruction(boxed)) = engine.id_star(event, 0) else {
            panic!("expected an obstruction")
        };
        let obstruction = *boxed;
        let CounterfactualObstruction::Hedge { treatments, outcomes, f, f_prime, .. } =
            &obstruction
        else {
            panic!("expected a hedge")
        };
        assert_eq!((treatments.as_slice(), outcomes.as_slice()), (&[0][..], &[1][..]));
        assert_eq!((f.as_slice(), f_prime.as_slice()), (&[0, 1][..], &[1][..]));
        obstruction.verify().unwrap();
        let refusal = CounterfactualIdRefusal::hedge(obstruction.clone());
        assert_eq!(
            (refusal.code, refusal.detail),
            ("route_not_supported", "counterfactual_id.counterfactual_hedge")
        );
        // A tampered hedge fails the definition check.
        let mut tampered = obstruction;
        if let CounterfactualObstruction::Hedge { f_prime, .. } = &mut tampered {
            *f_prime = vec![0];
        }
        assert!(tampered.verify().is_err());
    }

    fn run(
        problem: &CounterfactualIdProblem,
        event: Vec<Atom>,
    ) -> Result<CounterfactualFunctional, Halt> {
        let ctx = ExecutionContext::for_tests(1);
        engine(problem, &ctx).id_star(event, 0)
    }

    fn atom(var: u32, value: f64, sub: &[(u32, f64)]) -> Atom {
        Atom { var, value: level(value), sub: sub.iter().map(|&(v, x)| (v, level(x))).collect() }
    }

    fn conflict_of(result: Result<CounterfactualFunctional, Halt>) -> (u32, Option<CfValue>) {
        let Err(Halt::Obstruction(boxed)) = result else { panic!("expected an obstruction") };
        boxed.verify().unwrap();
        let CounterfactualObstruction::ConflictingSubscripts { variable, event, .. } = *boxed
        else {
            panic!("expected a conflict")
        };
        (variable, event)
    }

    /// The guards the effect-on-the-treated shape cannot reach (its top-level
    /// counterfactual graph has at most one copy per variable and one
    /// intervened world), each reached by a synthetic conjunction on the engine.
    #[test]
    fn synthetic_conjunctions_reach_the_guards_the_ett_shape_cannot() {
        let binary = |n| vec![vec![0.0, 1.0]; n];
        // X -> Y, no confounding: {Y_{x=0} = 0, Y_{x=1} = 1}. The two copies of Y
        // see fixed parents at two levels, so Lemma 24 does not merge them (a
        // merge would read one node at two levels: an exact zero), and the copies
        // share Y's exogenous term, so they form one district (split, the two
        // terms would multiply as if independent). ID* fails on X.
        let chain = CounterfactualIdProblem::new(binary(2), &[(0, 1)], &[]).unwrap();
        let event = vec![atom(1, 0.0, &[(0, 0.0)]), atom(1, 1.0, &[(0, 1.0)])];
        assert_eq!(conflict_of(run(&chain, event)).0, 0);
        // X -> Y, X -> W, X <-> Y, W <-> Y: {Y_{x=1} = 1, W = 0} keeps X as a free
        // ancestor of W in the district that sets X: a free node conflicts.
        let free =
            CounterfactualIdProblem::new(binary(3), &[(0, 1), (0, 2)], &[(0, 1), (1, 2)]).unwrap();
        let event = vec![atom(1, 1.0, &[(0, 1.0)]), atom(2, 0.0, &[])];
        assert_eq!(conflict_of(run(&free, event)), (0, None));
        // P -> V, P <-> V: {V_{p=1} = 0, V = 1, P = 1}. The copies of V stay
        // unmerged (a node parent against a fixed one), no subscript conflicts
        // (P is read at the level it is set to), and by consistency V_{p=1} = V
        // when P = 1: an exact zero.
        let pv = CounterfactualIdProblem::new(binary(2), &[(0, 1)], &[(0, 1)]).unwrap();
        let event = vec![atom(1, 0.0, &[(0, 1.0)]), atom(1, 1.0, &[]), atom(0, 1.0, &[])];
        assert!(matches!(run(&pv, event), Ok(CounterfactualFunctional::Zero)));
        // X -> Y and an isolated W: W_{x=1} and W merge (no parents), so reading
        // the one node at two levels is an exact zero.
        let isolated = CounterfactualIdProblem::new(binary(3), &[(0, 1)], &[]).unwrap();
        let event = vec![atom(2, 0.0, &[(0, 1.0)]), atom(2, 1.0, &[])];
        assert!(matches!(run(&isolated, event), Ok(CounterfactualFunctional::Zero)));
    }

    /// A term `sum_m P(m | x) P(y | m)` with `P(X = x) = 0` and a three-level `M`:
    /// its value is `sum_m q_m g(m)` for the extension `q` of `P(M | X = x)`,
    /// with `g = P(y = 1 | m) = (0.5, 0.25, 0.75)`. The uniform extension and all
    /// mass on the first level agree (0.5); all mass on level 1 gives 0.25, so
    /// the term is refused as a positivity violation.
    #[test]
    fn a_summed_three_level_conditional_on_a_zero_mass_event_is_caught() {
        let (x, m, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let mut arena = CausalExprArena::new();
        let none = arena.empty_intervention_set();
        let (xs, ms, ys) =
            (arena.intern_var_set([x]), arena.intern_var_set([m]), arena.intern_var_set([y]));
        let m_given_x = arena.intern_distribution(ms, xs, none, DomainRef::Observational);
        let y_given_m = arena.intern_distribution(ys, ms, none, DomainRef::Observational);
        let list = arena.intern_list([m_given_x, y_given_m]);
        let product = arena.intern(ExprNode::Product(list));
        let expression = arena.intern(ExprNode::SumOut { variables: ms, expr: product });
        let term = CfTerm {
            intervention: vec![(0, level(1.0))],
            event: vec![(2, level(1.0))],
            arena,
            expression,
        };
        // X = 0 always; P(m) = (0.3, 0.3, 0.4); P(y = 1 | m) = g(m). Row-major (x, m, y).
        let g = [0.5, 0.25, 0.75];
        let mut probabilities = vec![0.0; 12];
        for (level_m, (share, g)) in [0.3, 0.3, 0.4].iter().zip(g).enumerate() {
            probabilities[level_m * 2] = share * (1.0 - g);
            probabilities[level_m * 2 + 1] = share * g;
        }
        let joint = DenseJoint {
            levels: vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0], vec![0.0, 1.0]],
            probabilities,
        };
        let assignment = Assignment::from_pairs([(x, Value::f64(1.0)), (y, Value::f64(1.0))]);
        let compiled = term.arena.compile(term.expression).unwrap();
        let under = |fill| {
            let (provider, touched_null) =
                factor_provider(&joint, &term.arena, term.expression, fill);
            assert!(touched_null);
            compiled
                .evaluate_with(&term.arena, &provider, &EvalContext::default(), &assignment)
                .unwrap()
        };
        assert!((under(NullExtension::Uniform) - 0.5).abs() < 1e-12);
        assert!((under(NullExtension::Level(0)) - 0.5).abs() < 1e-12);
        assert!((under(NullExtension::Level(1)) - 0.25).abs() < 1e-12);
        let ctx = ExecutionContext::for_tests(1);
        let refusal = TermEvaluator::new(&joint).term(&term, &HashMap::new(), &ctx).unwrap_err();
        assert_eq!(refusal.detail, "counterfactual_id.positivity_violation");
    }

    #[test]
    fn a_context_cancelled_mid_decision_stops_with_a_receipt() {
        let problem =
            CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 3], &[(0, 1), (1, 2)], &[(0, 2)])
                .unwrap();
        let query = CounterfactualEventQuery::effect_on_treated(
            VariableId::from_raw(0),
            1.0,
            0.0,
            VariableId::from_raw(2),
            1.0,
        )
        .unwrap();
        let shape = classify(&problem, &query).unwrap();
        let given = vec![(0, 0.0f64.to_bits())];
        let ctx = ExecutionContext::for_tests(1);
        let full = decide_under(
            &problem,
            &query,
            shape,
            given.clone(),
            COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            SharedSearch::new(SearchBudget::new(COUNTERFACTUAL_ID_DEFAULT_LIMITS, &ctx).unwrap()),
        )
        .unwrap();
        let total = full.search.operations_consumed;
        // Cancel after the first, a middle and the last-but-one charge: each stops there.
        for after in [1, total / 2, total - 1] {
            let ctx = ExecutionContext::for_tests(1);
            let mut search = SharedSearch::new(
                SearchBudget::new(COUNTERFACTUAL_ID_DEFAULT_LIMITS, &ctx).unwrap(),
            );
            search.cancel_after = Some((after, ctx.cancellation.clone()));
            let refusal = decide_under(
                &problem,
                &query,
                shape,
                given.clone(),
                COUNTERFACTUAL_ID_DEFAULT_LIMITS,
                search,
            )
            .unwrap_err();
            let receipt = refusal.receipt.expect("a receipt");
            assert_eq!(receipt.stop, SearchStop::Cancelled);
            assert_eq!(receipt.operations_consumed, Some(after));
            assert_eq!(refusal.detail, "counterfactual_id.budget");
        }
    }
}
