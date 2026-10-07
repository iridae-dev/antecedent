//! Ordered-response recovery of a two-variable binary m-graph (2.3 B2 / X10
//! second row): graph decision, derivation and nonrecoverability witness.
//!
//! This is a second, separate observation-recovery row. The 2.2 row
//! ([`crate::recovery`]) refuses every `R -> R` edge; this row is exactly the
//! graphs it refuses because of one response chain, and it uses a different
//! identifying order (responses are factored sequentially, the tail response
//! conditioned on the head response being observed).
//!
//! # The row
//!
//! - **Graph.** An m-graph over binary `X1, X2` (partially observed), their
//!   response indicators `R1, R2` (1 observed, 0 missing) and deterministic
//!   proxies `X*_i` (`X*_i = X_i` when `R_i = 1`, `?` when `R_i = 0`), nothing
//!   else. Edges: any substantive edge between `X1` and `X2` (or none); the
//!   proxy wiring `X_i -> X*_i <- R_i`; exactly one response edge `R_h -> R_t`
//!   (head `h`, tail `t`); optionally `X_h -> R_t`. No other edge, no bidirected
//!   edge, `R_h` has no parent other than a possible self-censoring `X_h`.
//! - **Observed margins.** The pattern law `P(R1, R2, X*1, X*2)` (nine cells).
//! - **Target law.** The full law `P(X1, X2)`.
//! - **Assumptions.** The m-graph is the data-generating Markov structure;
//!   positivity: `P(R_h = 1) > 0` and every complete-case cell
//!   `P(R1 = 1, R2 = 1, X*1 = x1, X*2 = x2) > 0`.
//!
//! # Derivation (checked in the evaluator, independent of the m-graph proofs)
//!
//! The m-graph factorizes `p(x, r) = p(x) p(r_h | pa) p(r_t | r_h, [x_h])` with
//! `R_h` parentless (absent self-censoring). Therefore
//! `P(R_h = 1, R_t = 1, X*_h = x_h, X*_t = x_t) = p(x_h, x_t) P(R_h = 1) q(x_h)`,
//! where `q(x_h) = P(R_t = 1 | R_h = 1, X_h = x_h)` when `X_h -> R_t` exists and
//! `q = P(R_t = 1 | R_h = 1)` otherwise. Both are ratios of observed pattern
//! margins: on `R_h = 1` the head proxy equals `X_h`, so
//! `q(x_h) = P(R_h = 1, R_t = 1, X*_h = x_h) / P(R_h = 1, X*_h = x_h)`
//! (resp. `P(R_h = 1, R_t = 1) / P(R_h = 1)`), with `X*_t` marginalized out of
//! the cells with `R_t = 1` (the tail proxy carries no information on `R_t = 0`).
//! The recovered law is the complete-case cell divided by
//! `P(R_h = 1) q(x_h)`. This order (head response, then tail response given the
//! head) is the colluder-free case of Mohan, Pearl and Tian's sequential
//! recoverability; unlike the 2.2 row it needs no `R_i _||_ R_j | pa(R_i)`
//! independence because the tail's parents include `R_h`.
//!
//! # Nonrecoverability
//!
//! A self-censoring edge `X_i -> R_i` (either variable) with every other edge in
//! the class is refused with a verified witness: two positive models Markov to
//! the m-graph that agree on all nine observed pattern cells and differ on
//! `P(X_i)`. Every mechanism is a fair coin except `X_i ~ Bern(1/2),
//! P(R_i = 1 | X_i) = (2/5, 3/5)` against `X_i ~ Bern(2/5),
//! P(R_i = 1 | X_i) = (1/3, 3/4)`, whatever the other parents of `R_i`; both give
//! `P(R_i = 1, X_i = 1) = 3/10` and `P(R_i = 1, X_i = 0) = 1/5`, and the other
//! nodes see `X_i` only through `R_i = 1` cells. The witness is verified by exact
//! integer enumeration over `60^4`, so it is self-certifying. It shows
//! nonrecoverability over all models Markov to the m-graph (the models are not
//! faithful); it claims nothing about generic parameters.
//!
//! # What is refused as not supported (never as nonrecoverable)
//!
//! Any edge outside the class: `X_t -> R_h`, a graph with no response chain (use
//! the 2.2 row), bidirected edges, edges out of a response or proxy into a
//! substantive node, cross-wired proxies. These are `route_not_supported`: the
//! row neither proves nor disproves recoverability there. A violation of
//! positivity is `transport_support_failure`.
//!
//! The decision charges one [`SearchBudget`]: one operation per edge inspected
//! and per witness configuration enumerated. Calibration of any sampled provider
//! composed on this row is separate, measured only at the release cut, and is
//! unmeasured and closed here: the evaluator is an exact-law evaluator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_core::{
    ExecutionContext, SearchBudget, SearchLimits, SearchReceipt, SearchStop, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};

/// Version of the ordered-response rule, bound into every plan.
pub const CHAIN_RECOVERY_RULE_VERSION: &str = "b2.recovery_chain.v1";
/// Every witness mechanism probability is `k / CHAIN_WITNESS_SCALE`.
pub const CHAIN_WITNESS_SCALE: u32 = 60;
/// Search limits of one decision.
pub const CHAIN_RECOVERY_LIMITS: SearchLimits = SearchLimits { operations: 2_000, depth: 8 };

/// Why a chain-recovery request was refused. The detail and reason code are fixed
/// per kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ChainRecoveryDetail {
    /// A self-censoring edge with a verified witness.
    NonrecoverableWitness,
    /// A mechanism or graph shape outside the row.
    UnsupportedMechanism,
    /// A malformed query, role declaration or witness.
    InvalidQuery,
    /// A cell or denominator of the observed law has no mass.
    Positivity,
    /// The observed pattern law is not a probability law.
    InvalidObservedLaw,
    /// A plan or witness does not verify.
    InvalidDerivation,
    /// The shared budget stopped.
    Budget,
}

impl ChainRecoveryDetail {
    /// Stable namespaced detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::NonrecoverableWitness => "recovery_chain.nonrecoverable_witness",
            Self::UnsupportedMechanism => "recovery_chain.unsupported_mechanism",
            Self::InvalidQuery => "recovery_chain.invalid_query",
            Self::Positivity => "recovery_chain.positivity",
            Self::InvalidObservedLaw => "recovery_chain.invalid_observed_law",
            Self::InvalidDerivation => "recovery_chain.invalid_derivation",
            Self::Budget => "recovery_chain.budget",
        }
    }

    /// Registered top-level reason code.
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::NonrecoverableWitness => {
                antecedent_core::reason_code!("transport_proven_non_transportable")
            }
            Self::UnsupportedMechanism => antecedent_core::reason_code!("route_not_supported"),
            Self::InvalidQuery | Self::InvalidObservedLaw => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::Positivity => antecedent_core::reason_code!("transport_support_failure"),
            Self::InvalidDerivation => antecedent_core::reason_code!("transport_not_certified"),
            Self::Budget => antecedent_core::reason_code!("transport_budget_cancel"),
        }
    }
}

/// A typed refusal: detail, reason code, explanation and, for a budget stop, the
/// receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainRecoveryError {
    /// Why.
    pub detail: ChainRecoveryDetail,
    /// Human explanation.
    pub message: String,
    /// The receipt of a budget stop.
    pub receipt: Option<Box<SearchReceipt>>,
}

impl ChainRecoveryError {
    /// A refusal of `detail` with an explanation.
    #[must_use]
    pub fn new(detail: ChainRecoveryDetail, message: impl Into<String>) -> Self {
        Self { detail, message: message.into(), receipt: None }
    }

    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.detail.reason_code()
    }
}

impl fmt::Display for ChainRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail.detail(), self.message)?;
        if let Some(receipt) = &self.receipt {
            write!(f, "; {}", receipt.summary())?;
        }
        Ok(())
    }
}

impl std::error::Error for ChainRecoveryError {}

fn refuse(detail: ChainRecoveryDetail, message: impl Into<String>) -> ChainRecoveryError {
    ChainRecoveryError::new(detail, message)
}

/// One partially observed variable with its response indicator and proxy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainPartial {
    /// The substantive variable `X_i`.
    pub variable: VariableId,
    /// Its response indicator `R_i`.
    pub response: VariableId,
    /// Its proxy `X*_i`.
    pub proxy: VariableId,
}

/// Recovery of `P(X1, X2)`; the declaration order fixes the axis order of the
/// observed pattern law and of the recovered law.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChainRecoveryQuery {
    /// First declared variable (axis 0).
    pub first: ChainPartial,
    /// Second declared variable (axis 1).
    pub second: ChainPartial,
}

/// A checked ordered-response recovery plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainRecoveryPlan {
    /// The query the plan was decided for.
    pub query: ChainRecoveryQuery,
    /// Axis (0 or 1) of the head of the response chain `R_h -> R_t`.
    pub head: usize,
    /// Whether the tail response also depends on the head variable.
    pub tail_depends_on_head_variable: bool,
    /// Rule version.
    pub rule_version: &'static str,
    /// Checked graphical premises, in order.
    pub premises: Vec<String>,
    /// The recovery formula in words.
    pub formula: String,
    /// Operations charged by the decision.
    pub operations_consumed: usize,
}

/// One mechanism of a witness model: `P(node = 1 | parents) = numerator / 60`,
/// indexed by parent configuration (first parent most significant; parents in
/// ascending variable id).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainWitnessMechanism {
    /// Node.
    pub node: u32,
    /// Its graph parents, ascending.
    pub parents: Vec<u32>,
    /// Numerators per parent configuration.
    pub numerators: Vec<u32>,
}

/// Two models that agree on the observed pattern law and differ on the target.
/// Each lists, in order, the mechanisms of `X1, X2, R1, R2`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainRecoveryWitness {
    /// The self-censoring edge `(X_i, R_i)`.
    pub edge: (u32, u32),
    /// First model.
    pub first: Vec<ChainWitnessMechanism>,
    /// Second model.
    pub second: Vec<ChainWitnessMechanism>,
}

/// What a verified witness shows, in exact integer masses over `60^4`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainWitnessCheck {
    /// Observed pattern cells compared (all equal).
    pub observed_cells: usize,
    /// A target cell `(x1, x2)` where the two models differ.
    pub differing_cell: (u8, u8),
    /// The two target masses at that cell over [`Self::denominator`].
    pub masses: (u128, u128),
    /// Common denominator `60^4`.
    pub denominator: u128,
}

/// The outcome of a decision.
#[derive(Clone, Debug)]
pub enum ChainRecoveryDecision {
    /// A checked plan.
    Recovered(Box<ChainRecoveryPlan>),
    /// A self-censoring edge with a verified witness.
    NonRecoverable(Box<ChainRecoveryWitness>),
}

fn node_variable(graph: &Admg, id: DenseNodeId) -> Option<VariableId> {
    match graph.nodes().get(id.as_usize()) {
        Some(NodeRef::Static(v)) => Some(*v),
        _ => None,
    }
}

/// Every directed edge as a `(from, to)` variable pair, or `None` for a
/// non-static node.
fn directed_edges(graph: &Admg) -> Option<BTreeSet<(VariableId, VariableId)>> {
    let mut edges = BTreeSet::new();
    for i in 0..graph.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).ok()?);
        let from_var = node_variable(graph, from)?;
        for child in graph.children(from) {
            edges.insert((from_var, node_variable(graph, *child)?));
        }
    }
    Some(edges)
}

struct Meter<'a> {
    budget: SearchBudget<'a>,
}

impl Meter<'_> {
    fn charge(
        &mut self,
        depth: usize,
        explored: &[&str],
        left: &[&str],
    ) -> Result<(), ChainRecoveryError> {
        self.budget.charge(depth, 0).map_err(|stop: SearchStop| ChainRecoveryError {
            detail: ChainRecoveryDetail::Budget,
            message: format!("the chain recovery budget stopped ({})", stop.code()),
            receipt: Some(Box::new(self.budget.receipt(
                stop,
                explored.iter().map(|s| (*s).to_owned()).collect(),
                left.iter().map(|s| (*s).to_owned()).collect(),
            ))),
        })
    }
}

fn check_roles(graph: &Admg, query: &ChainRecoveryQuery) -> Result<(), ChainRecoveryError> {
    let roles = role_variables(query);
    let distinct: BTreeSet<_> = roles.iter().copied().collect();
    if distinct.len() != roles.len() {
        return Err(refuse(
            ChainRecoveryDetail::InvalidQuery,
            "a variable is declared in two roles",
        ));
    }
    let Some(edges) = directed_edges(graph) else {
        return Err(refuse(
            ChainRecoveryDetail::InvalidQuery,
            "the m-graph must have static nodes only",
        ));
    };
    let mut nodes = BTreeSet::new();
    for i in 0..graph.node_count() {
        let Some(v) =
            u32::try_from(i).ok().and_then(|raw| node_variable(graph, DenseNodeId::from_raw(raw)))
        else {
            return Err(refuse(
                ChainRecoveryDetail::InvalidQuery,
                "the m-graph must have static nodes only",
            ));
        };
        nodes.insert(v);
    }
    if nodes != distinct || graph.node_count() != roles.len() {
        return Err(refuse(
            ChainRecoveryDetail::InvalidQuery,
            "every m-graph node must have exactly one declared role and every role must be a node",
        ));
    }
    for p in [query.first, query.second] {
        if !edges.contains(&(p.variable, p.proxy)) || !edges.contains(&(p.response, p.proxy)) {
            return Err(refuse(
                ChainRecoveryDetail::InvalidQuery,
                "a proxy must be wired to exactly its declared variable and response",
            ));
        }
    }
    Ok(())
}

/// `[X1, X2, R1, R2, X*1, X*2]`.
fn role_variables(query: &ChainRecoveryQuery) -> [VariableId; 6] {
    [
        query.first.variable,
        query.second.variable,
        query.first.response,
        query.second.response,
        query.first.proxy,
        query.second.proxy,
    ]
}

/// The response chain head axis, or the refusal that the row does not apply.
fn chain_head(
    edges: &BTreeSet<(VariableId, VariableId)>,
    query: &ChainRecoveryQuery,
) -> Result<usize, ChainRecoveryError> {
    let forward = edges.contains(&(query.first.response, query.second.response));
    let backward = edges.contains(&(query.second.response, query.first.response));
    match (forward, backward) {
        (true, false) => Ok(0),
        (false, true) => Ok(1),
        _ => Err(refuse(
            ChainRecoveryDetail::UnsupportedMechanism,
            "this row needs exactly one response edge R -> R; a graph without it belongs to the 2.2 observation-recovery route",
        )),
    }
}

/// Every edge must be one the row models; returns the self-censoring edges.
fn classify_edges(
    meter: &mut Meter<'_>,
    graph: &Admg,
    query: &ChainRecoveryQuery,
    head: usize,
) -> Result<(Vec<(VariableId, VariableId)>, bool), ChainRecoveryError> {
    if graph.has_bidirected() {
        return Err(refuse(
            ChainRecoveryDetail::UnsupportedMechanism,
            "bidirected edges (unmeasured confounding) are outside the chain row",
        ));
    }
    let Some(edges) = directed_edges(graph) else {
        return Err(refuse(
            ChainRecoveryDetail::InvalidQuery,
            "the m-graph must have static nodes only",
        ));
    };
    let parts = [query.first, query.second];
    let (h, t) = (parts[head], parts[1 - head]);
    let mut allowed: BTreeSet<(VariableId, VariableId)> = BTreeSet::new();
    allowed.insert((parts[0].variable, parts[1].variable));
    allowed.insert((parts[1].variable, parts[0].variable));
    for p in parts {
        allowed.insert((p.variable, p.proxy));
        allowed.insert((p.response, p.proxy));
        allowed.insert((p.variable, p.response)); // self-censoring: witnessed, not supported
    }
    allowed.insert((h.response, t.response));
    allowed.insert((h.variable, t.response));
    for edge in &edges {
        meter.charge(1, &[], &["edge_classification", "formula_or_witness"])?;
        if !allowed.contains(edge) {
            return Err(refuse(
                ChainRecoveryDetail::UnsupportedMechanism,
                format!(
                    "edge {:?} -> {:?} is outside the ordered-response row; neither recoverability nor nonrecoverability is claimed",
                    edge.0, edge.1
                ),
            ));
        }
    }
    let mut censoring: Vec<_> =
        parts.iter().map(|p| (p.variable, p.response)).filter(|e| edges.contains(e)).collect();
    censoring.sort_unstable();
    let tail_dependent = edges.contains(&(h.variable, t.response));
    Ok((censoring, tail_dependent))
}

/// Decide whether `P(X1, X2)` is recoverable from the observed pattern law of the
/// ordered-response m-graph, with a verified witness when it is not.
///
/// # Errors
///
/// A typed [`ChainRecoveryError`]: invalid query, unsupported mechanism, an
/// unverifiable witness or a budget stop with its receipt.
pub fn decide_chain_recovery(
    graph: &Admg,
    query: &ChainRecoveryQuery,
    ctx: &ExecutionContext,
) -> Result<ChainRecoveryDecision, ChainRecoveryError> {
    let budget =
        SearchBudget::new(CHAIN_RECOVERY_LIMITS, ctx).map_err(|receipt| ChainRecoveryError {
            detail: ChainRecoveryDetail::Budget,
            message: format!(
                "the chain recovery budget stopped before entry ({})",
                receipt.stop.code()
            ),
            receipt: Some(Box::new(receipt)),
        })?;
    let mut meter = Meter { budget };
    check_roles(graph, query)?;
    let Some(edges) = directed_edges(graph) else {
        return Err(refuse(
            ChainRecoveryDetail::InvalidQuery,
            "the m-graph must have static nodes only",
        ));
    };
    let head = chain_head(&edges, query)?;
    let (censoring, dependent) = classify_edges(&mut meter, graph, query, head)?;
    if let Some(&(x, r)) = censoring.first() {
        let witness = build_witness(graph, query, x, r)?;
        verify_inner(graph, query, &witness, &mut meter)?;
        return Ok(ChainRecoveryDecision::NonRecoverable(Box::new(witness)));
    }
    meter.charge(2, &["edge_classification"], &["formula"])?;
    let parts = [query.first, query.second];
    let (h, t) = (parts[head], parts[1 - head]);
    let mut premises = vec![
        "m_graph_dag_no_bidirected".to_owned(),
        format!("response_chain:{}->{}", h.response.raw(), t.response.raw()),
        format!("head_response_parentless:{}", h.response.raw()),
        "no_self_censoring".to_owned(),
        format!("tail_depends_on_head_variable:{dependent}"),
    ];
    premises.push("positivity_checked_on_the_observed_law".to_owned());
    let formula = if dependent {
        "p(x_h,x_t) = P(R_h=1,R_t=1,X*_h=x_h,X*_t=x_t) / [P(R_h=1) * P(R_h=1,R_t=1,X*_h=x_h) / P(R_h=1,X*_h=x_h)]"
    } else {
        "p(x_h,x_t) = P(R_h=1,R_t=1,X*_h=x_h,X*_t=x_t) / [P(R_h=1) * P(R_h=1,R_t=1) / P(R_h=1)]"
    };
    Ok(ChainRecoveryDecision::Recovered(Box::new(ChainRecoveryPlan {
        query: *query,
        head,
        tail_depends_on_head_variable: dependent,
        rule_version: CHAIN_RECOVERY_RULE_VERSION,
        premises,
        formula: formula.to_owned(),
        operations_consumed: meter.budget.operations(),
    })))
}

fn sorted_parents(graph: &Admg, v: VariableId) -> Option<Vec<u32>> {
    for i in 0..graph.node_count() {
        let id = DenseNodeId::from_raw(u32::try_from(i).ok()?);
        if node_variable(graph, id)? == v {
            let mut out = Vec::new();
            for p in graph.parents(id) {
                out.push(node_variable(graph, *p)?.raw());
            }
            out.sort_unstable();
            return Some(out);
        }
    }
    None
}

/// Whether bit `position` (of `n`, first most significant) of `config` is set.
fn config_bit(config: usize, n: usize, position: usize) -> bool {
    (config >> (n - 1 - position)) & 1 == 1
}

fn build_witness(
    graph: &Admg,
    query: &ChainRecoveryQuery,
    censored: VariableId,
    response: VariableId,
) -> Result<ChainRecoveryWitness, ChainRecoveryError> {
    let order =
        [query.first.variable, query.second.variable, query.first.response, query.second.response];
    let mut first = Vec::new();
    let mut second = Vec::new();
    for node in order {
        let Some(parents) = sorted_parents(graph, node) else {
            return Err(refuse(
                ChainRecoveryDetail::InvalidQuery,
                "a declared node is not in the graph",
            ));
        };
        let n = parents.len();
        let configs = 1usize << n;
        let (mut a, mut b) = (vec![30u32; configs], vec![30u32; configs]);
        if node == censored {
            b = vec![24u32; configs];
        } else if node == response {
            let Some(pos) = parents.iter().position(|p| *p == censored.raw()) else {
                return Err(refuse(
                    ChainRecoveryDetail::InvalidQuery,
                    "the censoring edge is not in the graph",
                ));
            };
            for (config, (x, y)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
                let one = config_bit(config, n, pos);
                *x = if one { 36 } else { 24 };
                *y = if one { 45 } else { 20 };
            }
        }
        first.push(ChainWitnessMechanism {
            node: node.raw(),
            parents: parents.clone(),
            numerators: a,
        });
        second.push(ChainWitnessMechanism { node: node.raw(), parents, numerators: b });
    }
    Ok(ChainRecoveryWitness { edge: (censored.raw(), response.raw()), first, second })
}

/// Joint masses over `60^4` of one witness model: the observed pattern law keyed
/// `[r1, r2, x1*, x2*]` (`2` for missing) and the target law over `(x1, x2)`.
type ModelMasses = (BTreeMap<[u8; 4], u128>, BTreeMap<(u8, u8), u128>);

fn model_masses(
    graph: &Admg,
    query: &ChainRecoveryQuery,
    model: &[ChainWitnessMechanism],
) -> Result<ModelMasses, ChainRecoveryError> {
    let bad = |why: &str| refuse(ChainRecoveryDetail::InvalidDerivation, format!("witness: {why}"));
    let order = [
        query.first.variable.raw(),
        query.second.variable.raw(),
        query.first.response.raw(),
        query.second.response.raw(),
    ];
    if model.len() != 4 || model.iter().zip(order).any(|(m, node)| m.node != node) {
        return Err(bad("a model must list the mechanisms of X1, X2, R1, R2 in order"));
    }
    for m in model {
        let expected = sorted_parents(graph, VariableId::from_raw(m.node))
            .ok_or_else(|| bad("a mechanism names a node outside the graph"))?;
        if m.parents != expected
            || m.numerators.len() != 1usize << expected.len()
            || m.numerators.iter().any(|k| *k > CHAIN_WITNESS_SCALE)
            || m.parents.iter().any(|p| !order.contains(p))
        {
            return Err(bad("a mechanism does not match the graph parents or its probabilities"));
        }
    }
    let mut observed = BTreeMap::new();
    let mut target = BTreeMap::new();
    for cell in 0..16usize {
        let values = [(cell >> 3) & 1, (cell >> 2) & 1, (cell >> 1) & 1, cell & 1];
        let mut mass = 1u128;
        for m in model {
            let mut config = 0usize;
            for p in &m.parents {
                let at = order.iter().position(|o| o == p).ok_or_else(|| bad("unknown parent"))?;
                config = (config << 1) | values[at];
            }
            let k = u128::from(m.numerators[config]);
            let at = order.iter().position(|o| *o == m.node).ok_or_else(|| bad("unknown node"))?;
            mass *= if values[at] == 1 { k } else { u128::from(CHAIN_WITNESS_SCALE) - k };
        }
        let level = |bit: usize| u8::from(bit == 1);
        let key = [
            level(values[2]),
            level(values[3]),
            if values[2] == 1 { level(values[0]) } else { 2 },
            if values[3] == 1 { level(values[1]) } else { 2 },
        ];
        *observed.entry(key).or_insert(0u128) += mass;
        *target.entry((level(values[0]), level(values[1]))).or_insert(0u128) += mass;
    }
    Ok((observed, target))
}

fn verify_inner(
    graph: &Admg,
    query: &ChainRecoveryQuery,
    witness: &ChainRecoveryWitness,
    meter: &mut Meter<'_>,
) -> Result<ChainWitnessCheck, ChainRecoveryError> {
    let bad = |why: &str| refuse(ChainRecoveryDetail::InvalidDerivation, format!("witness: {why}"));
    let edges = directed_edges(graph).ok_or_else(|| bad("non-static graph"))?;
    let edge = (VariableId::from_raw(witness.edge.0), VariableId::from_raw(witness.edge.1));
    if !edges.contains(&edge) {
        return Err(bad("the named self-censoring edge is not in the graph"));
    }
    for _ in 0..2 {
        meter.charge(2, &["edge_classification"], &["witness"])?;
    }
    let (obs_a, target_a) = model_masses(graph, query, &witness.first)?;
    let (obs_b, target_b) = model_masses(graph, query, &witness.second)?;
    for _ in 0..16 {
        meter.charge(3, &["edge_classification"], &["witness"])?;
    }
    if obs_a != obs_b {
        return Err(bad("the two models disagree on the observed pattern law"));
    }
    let Some((cell, masses)) = target_a
        .iter()
        .find_map(|(cell, a)| target_b.get(cell).filter(|b| *b != a).map(|b| (*cell, (*a, *b))))
    else {
        return Err(bad("the two models agree on the target law"));
    };
    Ok(ChainWitnessCheck {
        observed_cells: obs_a.len(),
        differing_cell: cell,
        masses,
        denominator: u128::from(CHAIN_WITNESS_SCALE).pow(4),
    })
}

/// Verify a witness exactly against the graph: its mechanisms match the graph
/// parents, the two models give identical observed pattern masses (integer
/// enumeration over `60^4`) and different target masses. Charged against its own
/// budget: 18 operations.
///
/// # Errors
///
/// `invalid_derivation` when any check fails.
pub fn verify_chain_witness(
    graph: &Admg,
    query: &ChainRecoveryQuery,
    witness: &ChainRecoveryWitness,
    ctx: &ExecutionContext,
) -> Result<ChainWitnessCheck, ChainRecoveryError> {
    let budget =
        SearchBudget::new(CHAIN_RECOVERY_LIMITS, ctx).map_err(|receipt| ChainRecoveryError {
            detail: ChainRecoveryDetail::Budget,
            message: format!(
                "the chain recovery budget stopped before entry ({})",
                receipt.stop.code()
            ),
            receipt: Some(Box::new(receipt)),
        })?;
    let mut meter = Meter { budget };
    check_roles(graph, query)?;
    verify_inner(graph, query, witness, &mut meter)
}
