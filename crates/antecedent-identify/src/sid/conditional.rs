//! ADMG conditional transport on the classical complete-source family (2.2B B1).
//!
//! The query is the target population's conditional interventional distribution
//! `P*(y | do(x), w)` over a selection ADMG, answered from a source that can run
//! every experiment plus the target observational law. It extends the classical
//! sID row rather than standing beside it: a [`ConditionalTransportQuery`] wraps
//! a [`ClassicalTransportQuery`] and adds the conditioned set.
//!
//! # Reduction
//!
//! Stage 1 is IDC line 1 (Shpitser and Pearl 2006, as `crate::idc` implements it
//! for ordinary identification): while some `w` in `W` has `Y ⊥ w | X, W \ {w}`
//! in `G` with every edge into `X` (bidirected ones included) and every directed
//! edge out of `w` removed, `w` moves into the intervention set. Candidates are
//! tried in dense-node order and the scan restarts after every move, as in
//! `crate::idc`. Rule 2 of the do-calculus applied in the target population,
//! whose SCM has the causal graph `G` (a selection node is a parentless cause
//! conditioned in the target, so it opens no path), gives
//! `P*(y | do(x), w) = P*(y | do(x, w'), w'')`. Stage 2 decides the reduced
//! joint `P*(y, w'' | do(x, w'))` with the classical sID engine, and stage 3
//! binds it to the supplied catalog with the classical catalog search. The
//! conditional is the joint normalized over `y` at the requested `w''`.
//!
//! # Guarantee
//!
//! Sound, incomplete. Every decision re-checks its own reduction before it
//! returns any outcome: every move's rule-2 premise and the maximality of the
//! moved set are replayed by an independent augmented-graph separation test
//! ([`check_reduction`], a different criterion from the search's path test),
//! each test charged to the decision's one budget. The reduced joint is built
//! only through the classical sID engine's verified-derivation path, whose
//! checker replays it under the same budget. A stored record re-runs both
//! checks in [`ConditionalTransportDerivation::from_record_checked`] (the
//! preparation and artifact consumer paths). The conditional truth is
//! enumerated from latent SCMs in the test suites. That an s-hedge for
//! the reduced joint (with a maximal moved set) makes the conditional
//! non-transportable is the transport analogue of IDC completeness and is
//! paper-inherited (not verified here), so such a case is
//! [`ConditionalTransportDecision::NotCertified`] with an inspection-only
//! [`ConditionalObstructionCandidate`], never an impossibility claim.
//!
//! # Selection does not change the moves
//!
//! The rule-2 test is evaluated in the causal graph `G` alone: a selection node
//! is a parentless node whose only edge points into its target, and in the
//! target population it is conditioned on (fixed to the target's value). On any
//! path through a selection node that node is a non-collider (its only edge
//! points out of it), so conditioning on it blocks the path; and since it is
//! nobody's descendant, conditioning on it opens no collider. Adding the
//! selection nodes to `G` and to the conditioning set thus leaves the
//! m-separation of `Y` and `w` unchanged, and the moved set is independent of
//! the selection targets. `the_moved_set_is_selection_independent` in the
//! integration tests pins this on every three-node ADMG and selection pattern.
//!
//! # Extension note for the hedge/witness design (B5)
//!
//! [`ConditionalObstructionCandidate`] is additive: it pairs the unchanged
//! [`SHedgeCertificate`] of the reduced joint with the rule-2 moves that produced
//! the reduced query and the non-movable remainder. It is deliberately not a
//! variant of `SHedgeCertificate`/`HedgeCertificate`, whose shapes are unchanged.
//! A later shared conditional-obstruction witness can adopt its record
//! ([`ConditionalObstructionRecord`]: moves, remaining set, s-hedge record) and
//! upgrade it to a proof once the conditional completeness theorem is verified.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::mixed_source::{Checker, Set};
use super::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportDerivation,
    ClassicalTransportQuery, ClassicalTransportResult, SHedgeCertificate, SHedgeRecord,
    SharedSearch, SidDerivationRecord, SidLimits, identify_catalog_transport_metered,
    identify_classical_transport_metered, sid_memory_bytes,
};
use crate::IdentificationError;
use antecedent_core::{
    DEFAULT_SEARCH_MEMORY_BYTES, EvidenceCatalog, ExecutionContext, SearchBudget, SearchLimits,
    SearchReceipt, VariableId,
};
use antecedent_expr::{CausalExprArena, ExprId, ExprNode};
use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, SelectionDiagram};
use std::sync::Arc;

/// Observed variables in the selection diagram.
pub const ADMG_CONDITIONAL_MAX_OBSERVED: usize = 6;
/// Treatments in one query.
pub const ADMG_CONDITIONAL_MAX_TREATMENTS: usize = 3;
/// Conditioned variables in one query.
pub const ADMG_CONDITIONAL_MAX_CONDITIONED: usize = 3;
/// Search limits of one decision (every stage together). They are the defaults
/// and the maxima; larger limits refuse. Memory and cancellation come from the
/// execution context on every charge.
pub const ADMG_CONDITIONAL_DEFAULT_LIMITS: SearchLimits =
    SearchLimits { operations: 4096, depth: 24 };
/// Memory cap (estimated live-state bytes, cumulative across the stages of one
/// decision): the default and the maximum. The effective cap is the smaller of
/// this and the context's hard limit, so it is never absent.
pub const ADMG_CONDITIONAL_MEMORY_BYTES: u64 = DEFAULT_SEARCH_MEMORY_BYTES;

/// A declared bound: graph size, treatments, conditioned set or search limits.
const BOUNDS_EXCEEDED: &str = "admg_transport.bounds_exceeded";
/// The reduced joint was not certified.
const NOT_CERTIFIED: &str = "admg_transport.not_certified";
/// A certified joint whose searched formulas do not bind to the catalog.
const MISSING_EVIDENCE: &str = "admg_transport.missing_evidence";
/// A limit or cancellation stopped the decision.
const BUDGET: &str = "admg_transport.budget";
/// A stored derivation or candidate does not check.
const INVALID_DERIVATION: &str = "admg_transport.invalid_derivation";

/// Stage names, in evaluation order.
const STAGES: [&str; 3] = ["rule_two_reduction", "classical_sid", "catalog_binding"];

/// The frozen `(reason code, admg_transport.* detail)` pair of an error the
/// conditional route raises, as the B1 promotion record declares them. `None`
/// for an error the route does not own.
#[must_use]
#[doc(hidden)]
pub fn admg_conditional_refusal(
    error: &IdentificationError,
) -> Option<(&'static str, &'static str)> {
    use antecedent_core::reason_code;
    match error {
        IdentificationError::UnsupportedInput { code } if *code == BOUNDS_EXCEEDED => {
            Some((reason_code!("route_not_supported"), BOUNDS_EXCEEDED))
        }
        IdentificationError::InvalidInput { message }
            if message.starts_with("admg_transport.invalid_query") =>
        {
            Some((reason_code!("invalid_argument"), "admg_transport.invalid_query"))
        }
        IdentificationError::InvalidCatalog { message }
            if message.starts_with("admg_transport.invalid_catalog") =>
        {
            Some((reason_code!("invalid_argument"), "admg_transport.invalid_catalog"))
        }
        IdentificationError::InvalidDerivation { code } if *code == INVALID_DERIVATION => {
            Some((reason_code!("transport_not_certified"), INVALID_DERIVATION))
        }
        IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
            Some((reason_code!("transport_budget_cancel"), BUDGET))
        }
        _ => None,
    }
}

const fn bounds_exceeded() -> IdentificationError {
    IdentificationError::UnsupportedInput { code: BOUNDS_EXCEEDED }
}

fn invalid_query(message: &str) -> IdentificationError {
    IdentificationError::invalid_input(format!("admg_transport.invalid_query: {message}"))
}

const fn invalid_derivation() -> IdentificationError {
    IdentificationError::invalid_derivation(INVALID_DERIVATION)
}

/// Target conditional interventional query `P*(outcomes | do(treatments), conditioned_on)`
/// under the classical complete-source family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalTransportQuery {
    /// The unconditional classical query this one extends: outcomes, treatments,
    /// the source holding every experiment and the target.
    pub base: ClassicalTransportQuery,
    /// Conditioned coordinates `W` (at least one).
    pub conditioned_on: Arc<[VariableId]>,
}

/// A checked conditional derivation: the rule-2 moves, the non-movable
/// remainder and the checked sID derivation of the reduced joint.
#[derive(Clone, Debug)]
pub struct ConditionalTransportDerivation {
    query: ConditionalTransportQuery,
    moves: Arc<[VariableId]>,
    remaining: Arc<[VariableId]>,
    joint: ClassicalTransportDerivation,
    arena: CausalExprArena,
    root: ExprId,
}

/// Untrusted portable conditional derivation; it conveys no authority until
/// [`ConditionalTransportDerivation::from_record_checked`] succeeds.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalTransportRecord {
    /// Conditioned coordinates of the query, in query order.
    pub conditioned_on: Vec<u32>,
    /// Coordinates moved into the intervention set, in move order.
    pub moves: Vec<u32>,
    /// Coordinates still conditioned on, in query order.
    pub remaining: Vec<u32>,
    /// The reduced joint's sID derivation record (its arena travels separately).
    pub joint: SidDerivationRecord,
}

impl ConditionalTransportDerivation {
    /// The conditional query.
    #[must_use]
    pub const fn query(&self) -> &ConditionalTransportQuery {
        &self.query
    }
    /// Conditioned coordinates rule 2 moved into the intervention set, in move order.
    #[must_use]
    pub fn moves(&self) -> &[VariableId] {
        &self.moves
    }
    /// Conditioned coordinates that stay conditioned on, in query order.
    #[must_use]
    pub fn remaining(&self) -> &[VariableId] {
        &self.remaining
    }
    /// The reduced joint query `P*(y, w'' | do(x, w'))`.
    #[must_use]
    pub const fn reduced_query(&self) -> &ClassicalTransportQuery {
        self.joint.query()
    }
    /// The checked sID derivation of the reduced joint.
    #[must_use]
    pub const fn joint(&self) -> &ClassicalTransportDerivation {
        &self.joint
    }
    /// Arena of the conditional formula (the joint's arena plus the normalization).
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Conditional formula root: the joint over the joint summed over the outcomes,
    /// or the joint itself when every conditioned coordinate moved.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }

    /// Export the premises; the joint's arena travels as [`Self::joint`]'s arena.
    #[must_use]
    pub fn to_record(&self) -> ConditionalTransportRecord {
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect();
        ConditionalTransportRecord {
            conditioned_on: raw(&self.query.conditioned_on),
            moves: raw(&self.moves),
            remaining: raw(&self.remaining),
            joint: self.joint.to_record(),
        }
    }

    /// Re-check an untrusted record against externally supplied inputs: the
    /// query, every move's rule-2 premise, the maximality of the moved set and
    /// the reduced joint's sID proof, with independent checkers under `limits`.
    /// No search runs.
    ///
    /// # Errors
    /// `admg_transport.invalid_derivation` for any premise that does not check;
    /// a budget or cancellation error.
    pub fn from_record_checked(
        record: ConditionalTransportRecord,
        joint_arena: CausalExprArena,
        diagram: &SelectionDiagram,
        query: &ConditionalTransportQuery,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        validate(diagram, query)?;
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        if ids(&record.conditioned_on) != *query.conditioned_on {
            return Err(invalid_derivation());
        }
        let (moves, remaining) = check_reduction(
            diagram,
            query,
            &ids(&record.moves),
            &ids(&record.remaining),
            &mut || poll_cancellation(ctx),
        )?;
        let reduced = reduced_query(query, &moves, &remaining);
        let sid_limits = SidLimits { steps: limits.operations, depth: limits.depth };
        let joint = ClassicalTransportDerivation::from_record_checked(
            record.joint,
            joint_arena,
            diagram,
            &reduced,
            sid_limits,
            ctx,
        )
        .map_err(checked_error)?;
        Ok(Self::assemble(query.clone(), moves, remaining, joint))
    }

    /// Re-check this derivation against `diagram` and its own query under
    /// `limits`, as a preparation or consumer does before trusting it.
    ///
    /// # Errors
    /// As [`Self::from_record_checked`].
    pub fn recheck(
        &self,
        diagram: &SelectionDiagram,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<(), IdentificationError> {
        Self::from_record_checked(
            self.to_record(),
            self.joint.arena().clone(),
            diagram,
            &self.query,
            limits,
            ctx,
        )
        .map(|_| ())
    }

    /// Bind the reduced joint to the catalog. Failure means this formula does not
    /// bind, not that no other formula would.
    ///
    /// # Errors
    /// Invalid catalog or an unavailable leaf.
    pub fn bind_catalog(
        &self,
        catalog: &EvidenceCatalog,
    ) -> Result<BoundConditionalTransportFunctional, IdentificationError> {
        let joint = self.joint.bind_catalog(catalog)?;
        Ok(BoundConditionalTransportFunctional { derivation: self.clone(), joint })
    }

    fn assemble(
        query: ConditionalTransportQuery,
        moves: Vec<VariableId>,
        remaining: Vec<VariableId>,
        joint: ClassicalTransportDerivation,
    ) -> Self {
        let mut arena = joint.arena().clone();
        let root = if remaining.is_empty() {
            joint.root()
        } else {
            let outcomes = arena.intern_var_set(query.base.outcomes.iter().copied());
            let denominator =
                arena.intern(ExprNode::SumOut { variables: outcomes, expr: joint.root() });
            arena.intern(ExprNode::Ratio { numerator: joint.root(), denominator })
        };
        Self { query, moves: moves.into(), remaining: remaining.into(), joint, arena, root }
    }
}

/// Map an inner sID check failure to this route's detail; budget, cancellation
/// and memory keep their own kind.
fn checked_error(error: IdentificationError) -> IdentificationError {
    if error.is_budget_or_cancel() { error } else { invalid_derivation() }
}

/// A checked conditional derivation whose reduced joint is bound to a catalog.
#[derive(Clone, Debug)]
pub struct BoundConditionalTransportFunctional {
    derivation: ConditionalTransportDerivation,
    joint: BoundTransportFunctional,
}

impl BoundConditionalTransportFunctional {
    /// The checked conditional derivation.
    #[must_use]
    pub const fn derivation(&self) -> &ConditionalTransportDerivation {
        &self.derivation
    }
    /// The catalog-bound reduced joint, evaluated by the exact provider.
    #[must_use]
    pub const fn joint(&self) -> &BoundTransportFunctional {
        &self.joint
    }
    /// The frozen evidence catalog.
    #[must_use]
    pub const fn catalog(&self) -> &EvidenceCatalog {
        self.joint.catalog()
    }
    /// `(population, regime)` of every distribution leaf of the bound joint, sorted.
    #[must_use]
    pub fn cited_leaves(&self) -> Vec<(Arc<str>, Option<u32>)> {
        self.joint
            .arena()
            .leaf_bindings(self.joint.root())
            .into_iter()
            .map(|leaf| (leaf.population, leaf.regime.map(antecedent_core::RegimeId::raw)))
            .collect()
    }
}

/// Inspection-only candidate obstruction for a conditional query: the reduced
/// joint's verified s-hedge together with the rule-2 moves that produced it.
///
/// It is NOT a non-transportability proof: lifting the joint's s-hedge to the
/// conditional query needs the conditional completeness theorem, which is
/// paper-inherited. See the module's extension note.
#[derive(Clone, Debug)]
pub struct ConditionalObstructionCandidate {
    query: ConditionalTransportQuery,
    moves: Arc<[VariableId]>,
    remaining: Arc<[VariableId]>,
    reduced: ClassicalTransportQuery,
    s_hedge: SHedgeCertificate,
}

/// Untrusted portable candidate: moves, remaining set and the reduced joint's
/// s-hedge record.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalObstructionRecord {
    /// Coordinates moved into the intervention set, in move order.
    pub moves: Vec<u32>,
    /// Coordinates still conditioned on, in query order.
    pub remaining: Vec<u32>,
    /// The reduced joint's s-hedge.
    pub s_hedge: SHedgeRecord,
}

impl ConditionalObstructionCandidate {
    /// The conditional query.
    #[must_use]
    pub const fn query(&self) -> &ConditionalTransportQuery {
        &self.query
    }
    /// The reduced joint query the s-hedge obstructs.
    #[must_use]
    pub const fn reduced_query(&self) -> &ClassicalTransportQuery {
        &self.reduced
    }
    /// The reduced joint's verified s-hedge.
    #[must_use]
    pub const fn s_hedge(&self) -> &SHedgeCertificate {
        &self.s_hedge
    }
    /// Conditioned coordinates moved into the intervention set.
    #[must_use]
    pub fn moves(&self) -> &[VariableId] {
        &self.moves
    }
    /// Conditioned coordinates rule 2 cannot move.
    #[must_use]
    pub fn remaining(&self) -> &[VariableId] {
        &self.remaining
    }
    /// Export the candidate.
    #[must_use]
    pub fn to_record(&self) -> ConditionalObstructionRecord {
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect();
        ConditionalObstructionRecord {
            moves: raw(&self.moves),
            remaining: raw(&self.remaining),
            s_hedge: self.s_hedge.to_record(),
        }
    }
    /// Independently re-check an untrusted candidate: the moves and their
    /// maximality, the reduced query, and the s-hedge against it.
    ///
    /// # Errors
    /// `admg_transport.invalid_derivation` when any part does not check.
    pub fn from_record_checked(
        record: ConditionalObstructionRecord,
        diagram: &SelectionDiagram,
        query: &ConditionalTransportQuery,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        validate(diagram, query)?;
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let (moves, remaining) = check_reduction(
            diagram,
            query,
            &ids(&record.moves),
            &ids(&record.remaining),
            &mut || poll_cancellation(ctx),
        )?;
        let reduced = reduced_query(query, &moves, &remaining);
        let s_hedge =
            SHedgeCertificate::from_record_checked(record.s_hedge, diagram, &reduced, ctx)
                .map_err(checked_error)?;
        Ok(Self {
            query: query.clone(),
            moves: moves.into(),
            remaining: remaining.into(),
            reduced,
            s_hedge,
        })
    }
    /// Re-check this candidate against `diagram` and `query`.
    ///
    /// # Errors
    /// As [`Self::from_record_checked`].
    pub fn recheck(
        &self,
        diagram: &SelectionDiagram,
        query: &ConditionalTransportQuery,
        ctx: &ExecutionContext,
    ) -> Result<(), IdentificationError> {
        if &self.query != query {
            return Err(invalid_derivation());
        }
        Self::from_record_checked(self.to_record(), diagram, query, ctx).map(|_| ())
    }
}

/// Stage outcomes of a finished (not stopped) decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalStageRecord {
    /// Stage name.
    pub stage: &'static str,
    /// What the stage concluded.
    pub outcome: &'static str,
}

/// Why a decision is not certified, with what it explored.
#[derive(Clone, Debug)]
pub struct ConditionalTransportInspection {
    /// Stable detail (`admg_transport.not_certified`).
    pub detail: &'static str,
    /// Stages run, in order.
    pub stages: Vec<ConditionalStageRecord>,
    /// Moves made by the reduction.
    pub moves: Vec<VariableId>,
    /// Conditioned coordinates rule 2 cannot move.
    pub remaining: Vec<VariableId>,
    /// The reduced joint's verified s-hedge, when the sID engine produced one.
    /// Inspection only: not a non-transportability claim for the conditional.
    pub candidate: Option<Box<ConditionalObstructionCandidate>>,
}

/// Outcome of a bounded conditional transport decision. No variant is an
/// impossibility claim.
#[derive(Clone, Debug)]
pub enum ConditionalTransportDecision {
    /// A checked derivation bound to the supplied catalog.
    Identified(Box<BoundConditionalTransportFunctional>),
    /// The reduced joint is certified but no searched formula binds to the catalog.
    MissingEvidence {
        /// The certified derivation.
        derivation: Box<ConditionalTransportDerivation>,
        /// Per-strategy unmet obligations of the catalog search.
        obligations: Arc<[Arc<str>]>,
    },
    /// The reduced joint was not certified.
    NotCertified(ConditionalTransportInspection),
    /// A limit or cancellation stopped the decision; nothing is claimed.
    Exhausted(SearchReceipt),
}

impl ConditionalTransportDecision {
    /// Top-level reason code of a non-identified outcome; `None` when identified.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified(_) => None,
            Self::MissingEvidence { .. } => Some("transport_missing_evidence"),
            Self::NotCertified(_) => Some("transport_not_certified"),
            Self::Exhausted(_) => Some("transport_budget_cancel"),
        }
    }
    /// Stable `admg_transport.*` detail of a non-identified outcome.
    #[must_use]
    pub const fn detail_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified(_) => None,
            Self::MissingEvidence { .. } => Some(MISSING_EVIDENCE),
            Self::NotCertified(inspection) => Some(inspection.detail),
            Self::Exhausted(_) => Some(BUDGET),
        }
    }
}

/// Validate the query against the diagram: coordinates, disjointness, populations
/// and the declared bounds.
fn validate(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
) -> Result<(), IdentificationError> {
    let graph = diagram.causal_graph();
    if graph.node_count() > ADMG_CONDITIONAL_MAX_OBSERVED
        || query.base.treatments.len() > ADMG_CONDITIONAL_MAX_TREATMENTS
        || query.conditioned_on.len() > ADMG_CONDITIONAL_MAX_CONDITIONED
    {
        return Err(bounds_exceeded());
    }
    let base = &query.base;
    if base.outcomes.is_empty() {
        return Err(invalid_query("at least one outcome is required"));
    }
    if query.conditioned_on.is_empty() {
        return Err(invalid_query(
            "at least one conditioned variable is required; an unconditional query is the classical route",
        ));
    }
    if base.source.trim().is_empty() || base.target.trim().is_empty() || base.source == base.target
    {
        return Err(invalid_query("source and target must be distinct non-empty populations"));
    }
    let known = dense_variables(graph)?;
    let mut seen = Vec::new();
    for v in base.outcomes.iter().chain(base.treatments.iter()).chain(query.conditioned_on.iter()) {
        if !known.contains(v) {
            return Err(invalid_query("a query coordinate is not a node of the graph"));
        }
        if seen.contains(v) {
            return Err(invalid_query(
                "outcomes, treatments and conditioned variables must be distinct and disjoint",
            ));
        }
        seen.push(*v);
    }
    Ok(())
}

/// Static variable of every dense node, in dense order.
fn dense_variables(graph: &Admg) -> Result<Vec<VariableId>, IdentificationError> {
    graph
        .nodes()
        .iter()
        .map(|node| match node {
            antecedent_graph::NodeRef::Static(variable) => Ok(*variable),
            _ => Err(invalid_query("only static observed nodes are supported")),
        })
        .collect()
}

/// Dense index of `v` in `variables`.
fn dense_of(variables: &[VariableId], v: VariableId) -> Result<usize, IdentificationError> {
    variables.iter().position(|w| *w == v).ok_or_else(invalid_derivation)
}

/// The reduced joint query: outcomes `Y ++ W''`, treatments `X ++ W'` (both in
/// query order), same populations.
fn reduced_query(
    query: &ConditionalTransportQuery,
    moves: &[VariableId],
    remaining: &[VariableId],
) -> ClassicalTransportQuery {
    let moved: Vec<_> =
        query.conditioned_on.iter().copied().filter(|w| moves.contains(w)).collect();
    ClassicalTransportQuery {
        outcomes: query.base.outcomes.iter().copied().chain(remaining.iter().copied()).collect(),
        treatments: query.base.treatments.iter().copied().chain(moved).collect(),
        source: query.base.source.clone(),
        target: query.base.target.clone(),
    }
}

/// The search-side rule-2 test: every `y` is m-separated from `w` given `cond`
/// in the graph with edges into `x` (directed and bidirected) and directed edges
/// out of `w` removed. Path search over a rebuilt graph; the checker uses the
/// independent augmented-graph criterion instead.
fn rule_two_search(
    graph: &Admg,
    y: &[DenseNodeId],
    x: &[DenseNodeId],
    w: DenseNodeId,
    cond: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    let n = graph.node_count();
    let mut mutilated = Admg::with_variables(u32::try_from(n).map_err(|_| bounds_exceeded())?);
    for from in 0..n {
        let from = DenseNodeId::from_raw(u32::try_from(from).map_err(|_| bounds_exceeded())?);
        if from != w {
            for &to in graph.children(from) {
                if !x.contains(&to) {
                    mutilated.insert_directed(from, to)?;
                }
            }
        }
        for &to in graph.bidirected_neighbors(from) {
            if from.raw() < to.raw() && !x.contains(&from) && !x.contains(&to) {
                mutilated.insert_bidirected(from, to)?;
            }
        }
    }
    for &outcome in y {
        if !mutilated.is_m_separated(outcome, w, cond, ws)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The checker-side rule-2 test, by the independent augmented-graph criterion.
fn rule_two_checked(checker: &Checker, y: &Set, x: &Set, w: usize, rest: &Set) -> bool {
    let under = Set::from([w]);
    let cond = x.union(rest).copied().collect::<Set>();
    checker.separated(x, &under, y, &under, &cond)
}

/// Cancellation poll of a record check run outside a decision's budget.
fn poll_cancellation(ctx: &ExecutionContext) -> Result<(), IdentificationError> {
    if ctx.cancellation.is_cancelled() { Err(IdentificationError::Cancelled) } else { Ok(()) }
}

/// Independently replay recorded moves (each must satisfy rule 2 at its turn)
/// and require the remainder to be exactly the non-movable rest (maximality).
/// Returns the moves and the remainder in query order.
///
/// `charge` runs before every separation test. Inside a decision it charges
/// the decision's shared budget (operations, depth, memory, cancellation).
/// On the record paths ([`ConditionalTransportDerivation::from_record_checked`],
/// preparation and artifact consumption) it only polls cancellation: the check
/// is at most `|W| (|W| + 1) / 2 + |W|` separation tests over a graph of at most
/// [`ADMG_CONDITIONAL_MAX_OBSERVED`] nodes (both refused before any check), so
/// it is not metered there.
fn check_reduction(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    moves: &[VariableId],
    remaining: &[VariableId],
    charge: &mut dyn FnMut() -> Result<(), IdentificationError>,
) -> Result<(Vec<VariableId>, Vec<VariableId>), IdentificationError> {
    let graph = diagram.causal_graph();
    let variables = dense_variables(graph)?;
    let checker = Checker::new(graph)?;
    let index = |vs: &[VariableId]| -> Result<Set, IdentificationError> {
        vs.iter().map(|v| dense_of(&variables, *v)).collect()
    };
    let y = index(&query.base.outcomes)?;
    let mut x = index(&query.base.treatments)?;
    let mut rest = index(&query.conditioned_on)?;
    if moves.len() + remaining.len() != query.conditioned_on.len() {
        return Err(invalid_derivation());
    }
    for m in moves {
        charge()?;
        let w = dense_of(&variables, *m)?;
        if !rest.remove(&w) || !rule_two_checked(&checker, &y, &x, w, &rest) {
            return Err(invalid_derivation());
        }
        x.insert(w);
    }
    let expected: Vec<VariableId> =
        query.conditioned_on.iter().copied().filter(|w| !moves.contains(w)).collect();
    if expected != remaining {
        return Err(invalid_derivation());
    }
    for &w in &rest {
        charge()?;
        let mut others = rest.clone();
        others.remove(&w);
        if rule_two_checked(&checker, &y, &x, w, &others) {
            return Err(invalid_derivation());
        }
    }
    Ok((moves.to_vec(), expected))
}

/// Decide a bounded ADMG conditional transport query against a catalog.
///
/// Stages, charged to one [`SearchBudget`] under `limits`: the rule-2 reduction
/// and its independent re-check (one charge per separation test of either),
/// the classical sID decision of the reduced
/// joint (every engine step and the witness check), and the classical catalog
/// search binding it. A stop anywhere is [`ConditionalTransportDecision::Exhausted`],
/// never a verdict.
///
/// # Errors
/// Limits above [`ADMG_CONDITIONAL_DEFAULT_LIMITS`] or a query outside the bounds
/// (`admg_transport.bounds_exceeded`), an invalid query or catalog, or a
/// derivation that fails its own check.
pub fn decide_admg_conditional_transport(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<ConditionalTransportDecision, IdentificationError> {
    if limits.operations > ADMG_CONDITIONAL_DEFAULT_LIMITS.operations
        || limits.depth > ADMG_CONDITIONAL_DEFAULT_LIMITS.depth
    {
        return Err(bounds_exceeded());
    }
    validate(diagram, query)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("admg_transport.invalid_catalog: {error}"))
    })?;
    let mut selections = diagram.selection_targets().to_vec();
    selections.sort_unstable();
    for environment in catalog.environments.iter().filter(|e| e.identity == query.base.source) {
        let mut declared = environment.selection_targets.to_vec();
        declared.sort_unstable();
        if declared != selections {
            return Err(IdentificationError::invalid_catalog(
                "admg_transport.invalid_catalog: source selections disagree with the diagram",
            ));
        }
    }
    let unevaluated = || STAGES.iter().map(|s| format!("stage:{s}")).collect::<Vec<_>>();
    match SearchBudget::with_memory(limits, ADMG_CONDITIONAL_MEMORY_BYTES, ctx) {
        Ok(budget) => decide_charged(diagram, query, catalog, &mut SharedSearch::new(budget), ctx),
        Err(receipt) => Ok(ConditionalTransportDecision::Exhausted(SearchReceipt {
            unevaluated: unevaluated(),
            ..receipt
        })),
    }
}

/// Every stage of a validated decision, charged to `search`.
fn decide_charged(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<ConditionalTransportDecision, IdentificationError> {
    let stop = |search: &SharedSearch<'_>, error: IdentificationError, stage: usize| {
        if !error.is_budget_or_cancel() {
            return Err(error);
        }
        let receipt = search.receipt(
            search.stop_of(&error),
            STAGES[..stage].iter().map(|s| format!("stage:{s}")).collect(),
            STAGES[stage..].iter().map(|s| format!("stage:{s}")).collect(),
        );
        Ok(ConditionalTransportDecision::Exhausted(receipt))
    };
    search.begin_stage();
    let (moves, remaining) = match reduce_checked(diagram, query, search) {
        Ok(reduction) => reduction,
        Err(error) => return stop(search, error, 0),
    };
    search.end_stage();
    let reduced = reduced_query(query, &moves, &remaining);
    let mut stages = vec![ConditionalStageRecord {
        stage: STAGES[0],
        outcome: if moves.is_empty() { "no_move" } else { "moved" },
    }];
    search.begin_stage();
    let theory = match identify_classical_transport_metered(diagram, &reduced, search.meter(), ctx)
    {
        Ok(result) => result,
        Err(error) => return stop(search, error, 1),
    };
    search.end_stage();
    let derivation = match theory {
        ClassicalTransportResult::Identified(joint) => joint,
        ClassicalTransportResult::ProvenNonTransportable(s_hedge) => {
            stages.push(ConditionalStageRecord {
                stage: STAGES[1],
                outcome: "reduced_joint_s_hedge",
            });
            let candidate = ConditionalObstructionCandidate {
                query: query.clone(),
                moves: moves.clone().into(),
                remaining: remaining.clone().into(),
                reduced,
                s_hedge,
            };
            return Ok(ConditionalTransportDecision::NotCertified(
                ConditionalTransportInspection {
                    detail: NOT_CERTIFIED,
                    stages,
                    moves,
                    remaining,
                    candidate: Some(Box::new(candidate)),
                },
            ));
        }
        ClassicalTransportResult::NotCertified => {
            stages.push(ConditionalStageRecord { stage: STAGES[1], outcome: "not_certified" });
            return Ok(ConditionalTransportDecision::NotCertified(
                ConditionalTransportInspection {
                    detail: NOT_CERTIFIED,
                    stages,
                    moves,
                    remaining,
                    candidate: None,
                },
            ));
        }
    };
    search.begin_stage();
    let bound =
        match identify_catalog_transport_metered(diagram, &reduced, catalog, search.meter(), ctx) {
            Ok(result) => result,
            Err(error) => return stop(search, error, 2),
        };
    search.end_stage();
    match bound {
        CatalogTransportResult::Identified(bound) => {
            let derivation = ConditionalTransportDerivation::assemble(
                query.clone(),
                moves,
                remaining,
                bound.derivation().clone(),
            );
            Ok(ConditionalTransportDecision::Identified(Box::new(
                BoundConditionalTransportFunctional { derivation, joint: *bound },
            )))
        }
        CatalogTransportResult::MissingEvidence { obligations, .. }
        | CatalogTransportResult::NotCertified { obligations, .. } => {
            Ok(ConditionalTransportDecision::MissingEvidence {
                derivation: Box::new(ConditionalTransportDerivation::assemble(
                    query.clone(),
                    moves,
                    remaining,
                    *derivation,
                )),
                obligations,
            })
        }
    }
}

/// Stage 1 as a decision runs it: the search's reduction, then its independent
/// re-check ([`check_reduction`]) before any later stage sees it, each
/// separation test of both charged to `search` (one operation at depth 1 with
/// the diagram's live-state estimate). A search reduction the checker refuses
/// is `admg_transport.invalid_derivation`, never a result.
fn reduce_checked(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    search: &mut SharedSearch<'_>,
) -> Result<(Vec<VariableId>, Vec<VariableId>), IdentificationError> {
    let (moves, remaining) = reduce(diagram, query, search)?;
    verify_search_reduction(diagram, query, &moves, &remaining, search)
}

/// The independent re-check of a search reduction, charged to `search`.
fn verify_search_reduction(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    moves: &[VariableId],
    remaining: &[VariableId],
    search: &mut SharedSearch<'_>,
) -> Result<(Vec<VariableId>, Vec<VariableId>), IdentificationError> {
    use super::SearchCharge;
    let bytes = u64::try_from(sid_memory_bytes(diagram, 1)).unwrap_or(u64::MAX);
    check_reduction(diagram, query, moves, remaining, &mut || {
        search.charge(1, bytes).map_err(super::stop_error)
    })
}

/// Stage 1: IDC line 1 under the shared budget. Each separation test is one
/// charged operation at depth 1 with the diagram's live-state estimate.
fn reduce(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    search: &mut SharedSearch<'_>,
) -> Result<(Vec<VariableId>, Vec<VariableId>), IdentificationError> {
    use super::SearchCharge;
    let graph = diagram.causal_graph();
    let variables = dense_variables(graph)?;
    let dense = |v: &VariableId| -> Result<DenseNodeId, IdentificationError> {
        let i = variables
            .iter()
            .position(|w| w == v)
            .ok_or_else(|| invalid_query("unknown coordinate"))?;
        Ok(DenseNodeId::from_raw(u32::try_from(i).map_err(|_| bounds_exceeded())?))
    };
    let y = query.base.outcomes.iter().map(dense).collect::<Result<Vec<_>, _>>()?;
    let mut x = query.base.treatments.iter().map(dense).collect::<Result<Vec<_>, _>>()?;
    let mut rest = query.conditioned_on.iter().map(dense).collect::<Result<Vec<_>, _>>()?;
    rest.sort_unstable();
    let bytes = u64::try_from(sid_memory_bytes(diagram, 1)).unwrap_or(u64::MAX);
    let mut ws = DSeparationWorkspace::default();
    let mut moves = Vec::new();
    'restart: loop {
        for (i, &w) in rest.iter().enumerate() {
            search.charge(1, bytes).map_err(super::stop_error)?;
            let mut cond = x.clone();
            cond.extend(rest.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, z)| *z));
            if rule_two_search(graph, &y, &x, w, &cond, &mut ws)? {
                rest.remove(i);
                x.push(w);
                moves.push(variables[w.as_usize()]);
                continue 'restart;
            }
        }
        break;
    }
    let remaining =
        query.conditioned_on.iter().copied().filter(|w| !moves.contains(w)).collect::<Vec<_>>();
    Ok((moves, remaining))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::MemoryBudget;

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }
    fn d(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    /// Every source experiment and the target observational joint over three nodes.
    fn full_catalog() -> EvidenceCatalog {
        use antecedent_core::{
            DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeId, RegimeKind,
        };
        let mut regimes = Vec::new();
        for mask in 0..8u32 {
            let interventions: Vec<_> = (0..3).filter(|i| mask & (1 << i) != 0).map(v).collect();
            let measured: Vec<_> = (0..3).filter(|i| mask & (1 << i) == 0).map(v).collect();
            regimes.push(
                EvidenceRegime::try_new(
                    RegimeId::from_raw(mask),
                    if mask == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
                    EvidenceKind::Available,
                    interventions,
                    [],
                    measured,
                    "source",
                    DistributionAvailability::Joint,
                )
                .unwrap(),
            );
        }
        regimes.push(
            EvidenceRegime::try_new(
                RegimeId::from_raw(8),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1), v(2)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
        EvidenceCatalog::try_new([], regimes, [], None).unwrap()
    }

    /// A limit every stage fits alone exhausts the decision: the reduction, the
    /// sID decision and the catalog binding all charge the one budget.
    #[test]
    fn one_budget_is_shared_by_every_stage() {
        // X(0) -> Y(1) -> W(2), X <-> Y, selection on W: W stays conditioned.
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
        let query = ConditionalTransportQuery {
            base: ClassicalTransportQuery {
                outcomes: Arc::from([v(1)]),
                treatments: Arc::from([v(0)]),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            conditioned_on: Arc::from([v(2)]),
        };
        let catalog = full_catalog();
        let ctx = ExecutionContext::for_tests(1);
        let limits = |operations| SearchLimits { operations, depth: 24 };
        let total = (1..4096)
            .find(|ops| {
                matches!(
                    decide_admg_conditional_transport(
                        &diagram,
                        &query,
                        &catalog,
                        limits(*ops),
                        &ctx
                    )
                    .unwrap(),
                    ConditionalTransportDecision::Identified(_)
                )
            })
            .unwrap();
        let short = limits(total - 1);
        let ConditionalTransportDecision::Exhausted(receipt) =
            decide_admg_conditional_transport(&diagram, &query, &catalog, short, &ctx).unwrap()
        else {
            panic!("one operation short exhausts");
        };
        assert_eq!(receipt.stop, antecedent_core::SearchStop::Operations);
        assert_eq!(receipt.operations_consumed, Some(total - 1));
        // Each stage alone fits the short limit.
        let fresh = || SharedSearch::new(SearchBudget::new(short, &ctx).unwrap());
        let mut alone = fresh();
        let (moves, remaining) = reduce_checked(&diagram, &query, &mut alone).unwrap();
        let reduction = alone.operations();
        let reduced = reduced_query(&query, &moves, &remaining);
        let mut alone = fresh();
        let theory =
            identify_classical_transport_metered(&diagram, &reduced, alone.meter(), &ctx).unwrap();
        assert!(matches!(theory, ClassicalTransportResult::Identified(_)));
        let classical = alone.operations();
        let mut alone = fresh();
        let bound =
            identify_catalog_transport_metered(&diagram, &reduced, &catalog, alone.meter(), &ctx)
                .unwrap();
        assert!(matches!(bound, CatalogTransportResult::Identified(_)));
        let binding = alone.operations();
        assert!(reduction >= 1 && classical >= 1 && binding >= 1);
        assert!(reduction.max(classical).max(binding) < total - 1);
        assert_eq!(reduction + classical + binding, total, "every stage charges the one budget");
    }

    /// X -> Y -> W with S -> W: W is a descendant of Y and cannot move.
    #[test]
    fn the_shared_budget_charges_every_rule_two_separation_test() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
        let query = ConditionalTransportQuery {
            base: ClassicalTransportQuery {
                outcomes: Arc::from([v(1)]),
                treatments: Arc::from([v(0)]),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            conditioned_on: Arc::from([v(2)]),
        };
        let ctx = ExecutionContext::for_tests(1);
        let limits = SearchLimits { operations: 1, depth: 4 };
        let mut search = SharedSearch::new(SearchBudget::new(limits, &ctx).unwrap());
        reduce(&diagram, &query, &mut search).unwrap();
        assert_eq!(search.operations(), 1, "one separation test, one charge");
        let error = reduce(&diagram, &query, &mut search).unwrap_err();
        assert_eq!(search.stop_of(&error), antecedent_core::SearchStop::Operations);
        let mut small = ExecutionContext::for_tests(1);
        small.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(8) };
        let mut search = SharedSearch::new(SearchBudget::new(limits, &small).unwrap());
        let error = reduce(&diagram, &query, &mut search).unwrap_err();
        assert_eq!(search.stop_of(&error), antecedent_core::SearchStop::Memory);
    }

    /// X(0) -> Y(1) -> W(2), X <-> Y, selection on W: the fixture of the
    /// stage-accounting tests (W stays conditioned, the joint identifies).
    fn chain_fixture() -> (SelectionDiagram, ConditionalTransportQuery) {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
        let query = ConditionalTransportQuery {
            base: ClassicalTransportQuery {
                outcomes: Arc::from([v(1)]),
                treatments: Arc::from([v(0)]),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            conditioned_on: Arc::from([v(2)]),
        };
        (diagram, query)
    }

    /// The decision's own re-check of the search reduction refuses a corrupted
    /// move set by name and charges every separation test it runs.
    #[test]
    fn the_decision_rechecks_its_search_reduction_under_the_budget() {
        let (diagram, query) = chain_fixture();
        let ctx = ExecutionContext::for_tests(1);
        let limits = SearchLimits { operations: 64, depth: 4 };
        let fresh = || SharedSearch::new(SearchBudget::new(limits, &ctx).unwrap());
        let invalid = Some(("transport_not_certified", "admg_transport.invalid_derivation"));
        // The honest reduction (W is a descendant of Y and stays) checks, and the
        // check charges its one maximality test.
        let mut search = fresh();
        let checked = verify_search_reduction(&diagram, &query, &[], &[v(2)], &mut search).unwrap();
        assert_eq!(checked, (vec![], vec![v(2)]));
        assert_eq!(search.operations(), 1);
        // A search that moved the non-movable W is refused before any later stage.
        let error =
            verify_search_reduction(&diagram, &query, &[v(2)], &[], &mut fresh()).unwrap_err();
        assert_eq!(admg_conditional_refusal(&error), invalid);
        // A search that left a movable parent conditioned breaks maximality.
        let mut parent = Admg::with_variables(3);
        parent.insert_directed(d(0), d(1)).unwrap();
        parent.insert_directed(d(2), d(1)).unwrap();
        parent.insert_bidirected(d(0), d(1)).unwrap();
        let parent = SelectionDiagram::try_new(parent, [v(2)]).unwrap();
        let error =
            verify_search_reduction(&parent, &query, &[], &[v(2)], &mut fresh()).unwrap_err();
        assert_eq!(admg_conditional_refusal(&error), invalid);
        // The whole stage 1 of a decision is the search plus the check.
        let mut search = fresh();
        reduce_checked(&diagram, &query, &mut search).unwrap();
        assert_eq!(search.operations(), 2, "one search test and one check test");
        // The check is metered: an exhausted budget stops it, as does memory.
        let one = SearchLimits { operations: 1, depth: 4 };
        let mut search = SharedSearch::new(SearchBudget::new(one, &ctx).unwrap());
        let error = reduce_checked(&diagram, &query, &mut search).unwrap_err();
        assert_eq!(search.stop_of(&error), antecedent_core::SearchStop::Operations);
        assert_eq!(search.operations(), 1, "the search test ran, the check test did not");
        let mut small = ExecutionContext::for_tests(1);
        small.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(8) };
        let mut search = SharedSearch::new(SearchBudget::new(limits, &small).unwrap());
        let error =
            verify_search_reduction(&diagram, &query, &[], &[v(2)], &mut search).unwrap_err();
        assert_eq!(search.stop_of(&error), antecedent_core::SearchStop::Memory);
    }

    /// Memory is cumulative across the stages of one decision: each finished
    /// stage's peak is retained by every later charge. The hard limit is set
    /// one byte below the real peak (the three stage peaks summed, reached in
    /// the catalog binding) and at or above every sum a decision that dropped
    /// one stage's retention would reach, so only a cumulative decision stops.
    #[test]
    fn memory_is_cumulative_across_the_stages_of_one_decision() {
        let (diagram, query) = chain_fixture();
        let catalog = full_catalog();
        let ctx = ExecutionContext::for_tests(1);
        let limits = ADMG_CONDITIONAL_DEFAULT_LIMITS;
        let fresh = || SharedSearch::new(SearchBudget::new(limits, &ctx).unwrap());
        // Each stage's peak alone, on a fresh budget.
        let mut alone = fresh();
        let (moves, remaining) = reduce_checked(&diagram, &query, &mut alone).unwrap();
        let reduction = alone.peak_bytes();
        let reduced = reduced_query(&query, &moves, &remaining);
        let mut alone = fresh();
        identify_classical_transport_metered(&diagram, &reduced, alone.meter(), &ctx).unwrap();
        let classical = alone.peak_bytes();
        let mut alone = fresh();
        identify_catalog_transport_metered(&diagram, &reduced, &catalog, alone.meter(), &ctx)
            .unwrap();
        let binding = alone.peak_bytes();
        assert!(reduction > 0 && classical > 0 && binding > 0);
        let cumulative = reduction + classical + binding;
        // Without the reduction's retention, or without the sID stage's, or with
        // neither, the largest charge would be one of these.
        let dropped = (reduction + classical).max(classical + binding).max(reduction + binding);
        assert!(dropped < cumulative);
        let limit = cumulative - 1;
        assert!(limit >= dropped && limit >= reduction.max(classical).max(binding));
        let at = |hard: u64| {
            let mut tight = ExecutionContext::for_tests(1);
            tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(hard) };
            decide_admg_conditional_transport(&diagram, &query, &catalog, limits, &tight).unwrap()
        };
        let ConditionalTransportDecision::Exhausted(receipt) = at(limit) else {
            panic!("the cumulative peak exceeds the limit");
        };
        assert_eq!(receipt.stop, antecedent_core::SearchStop::Memory);
        assert_eq!(receipt.memory_limit_bytes, Some(limit));
        assert_eq!(
            receipt.explored,
            ["stage:rule_two_reduction", "stage:classical_sid"],
            "the stop lands in the catalog binding, after both stages were retained"
        );
        assert!(matches!(at(cumulative), ConditionalTransportDecision::Identified(_)));
    }
}
