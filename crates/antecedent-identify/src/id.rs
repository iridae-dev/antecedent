//! Shpitser & Pearl (2006) ID algorithm for semi-Markovian models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    Intervention, Value, VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DomainRef, EstimandMethod, ExprId, ExprNode, IdentifiedEstimand,
    InterventionAssignment, OutcomeExprId,
};
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;
use crate::hedge::HedgeCertificate;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::{AncestorMemo, PreparedAdmg};
use crate::result::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

mod expression;
mod recursion;
use expression::{c_component_expression, dist_marginal, sum_out_if_any};
use recursion::id_recurse;

/// Memo key: canonical (Y, X, V) plus optional hard assignment for ATE contrast sides.
///
/// Assignment must be part of the key: left (`do(T=t₁)`) and right (`do(T=t₀)`) share
/// the same (Y,X,V) geometry but produce distinct expressions.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SubproblemKey {
    y: BitSet,
    x: BitSet,
    v: BitSet,
    assign: Assign,
    dist: DistCtx,
}

/// Memo table of solved subproblems.
///
/// Lookups hash and compare the borrowed query parts, so a hit allocates and
/// clones nothing; the owned [`SubproblemKey`] is built only when a new
/// outcome is stored. Outcomes are shared as `Arc`s.
#[derive(Default)]
struct IdMemo {
    buckets: HashMap<u64, Vec<(SubproblemKey, Arc<IdOutcome>)>>,
    len: usize,
}

fn subproblem_hash(y: &BitSet, x: &BitSet, v: &BitSet, assign: &Assign, dist: &DistCtx) -> u64 {
    let mut hasher = DefaultHasher::new();
    y.hash(&mut hasher);
    x.hash(&mut hasher);
    v.hash(&mut hasher);
    assign.hash(&mut hasher);
    dist.hash(&mut hasher);
    hasher.finish()
}

impl IdMemo {
    const fn len(&self) -> usize {
        self.len
    }

    fn get(
        &self,
        hash: u64,
        y: &BitSet,
        x: &BitSet,
        v: &BitSet,
        assign: &Assign,
        dist: &DistCtx,
    ) -> Option<&Arc<IdOutcome>> {
        self.buckets.get(&hash)?.iter().find_map(|(key, outcome)| {
            (key.y == *y
                && key.x == *x
                && key.v == *v
                && key.assign == *assign
                && key.dist == *dist)
                .then_some(outcome)
        })
    }

    fn insert(&mut self, hash: u64, key: SubproblemKey, outcome: Arc<IdOutcome>) {
        let bucket = self.buckets.entry(hash).or_default();
        if let Some(slot) = bucket.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = outcome;
        } else {
            bucket.push((key, outcome));
            self.len += 1;
        }
    }
}

/// Hard-set values baked into emitted factors, one per intervened node.
///
/// The ATE contrast uses a single entry per side; temporal schedules bake
/// every treatment-time node of the unfolded window at the same level.
type Assign = Arc<[(DenseNodeId, Value)]>;

/// The distribution the current subproblem identifies against.
///
/// Shpitser–Pearl thread this explicitly; leaving it implicit silently
/// replaced line 7's `Q[S′]` with the marginal `P(S′)`, so factors lost their
/// conditioning on topological predecessors outside `S′` (front-door ADMGs
/// were assigned `∑_M P(M|T) P(Y)` instead of
/// `∑_m P(m|t) ∑_{t′} P(y|t′,m) P(t′)`).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum DistCtx {
    /// Marginal of the original observational law over the current `v`
    /// (chain rule / Tian factorization over `v` is exact — the pre-fix
    /// emission machinery is correct for this case and is reused verbatim).
    Marginal,
    /// A law `P′` produced by line 7 (and possibly marginalized by line 2).
    CFactor(Arc<Law>),
}

/// `P′ = ∑_{sumset} ∏ factors`, a law over the current `v`.
///
/// Invariant: the factors' own variables are exactly `v ∪ sumset` (disjoint),
/// one factor per variable, in topological order, and every factor is a
/// conditional law of its own variable that mentions only topological
/// predecessors. Line 7 creates a law with an empty sumset over `v = S′`;
/// line 2 moves the dropped non-ancestors from `v` into `sumset`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Law {
    sumset: BitSet,
    factors: Vec<Factor>,
}

/// One conditional `P′(v_i | v_π^{(i-1)})` of a line-7 product.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum Factor {
    /// Observational `P(var | cond)`, with `cond` frozen when the factor was
    /// formed: by Tian's telescope it includes predecessors *outside* the
    /// current `v`.
    Observed { var: DenseNodeId, cond: BitSet },
    /// Conditional of a marginalized law, `num / ∑_{var} num`, where `num` is
    /// the parent law summed over everything topologically after `var`
    /// (Tian's identity: `Q[H_i]` is a ratio of marginals of `Q[H]`).
    Conditional { var: DenseNodeId, num: Arc<Law> },
}

impl Factor {
    const fn var(&self) -> DenseNodeId {
        match self {
            Self::Observed { var, .. } | Self::Conditional { var, .. } => *var,
        }
    }
}

/// The mutable state one ID recursion threads through every line: working graph, expression
/// arena, memo table, derivation, counters, graph workspace and budget meter.
pub(crate) struct IdRun<'a, 'm> {
    prepared: &'a PreparedAdmg,
    ancestors: AncestorMemo,
    arena: &'a mut CausalExprArena,
    memo: &'a mut IdMemo,
    derivation: &'a mut DerivationTrace,
    perf: &'a mut IdentificationPerformanceRecord,
    ws: &'a mut GraphWorkspace,
    meter: &'a mut IdMeter<'m>,
}

/// One ID subproblem: the sets, the current law, and the free do-labels.
struct IdQuery<'a> {
    y: &'a BitSet,
    x: &'a BitSet,
    v: &'a BitSet,
    dist: &'a DistCtx,
    assign: Assign,
}

/// Outcome of a recursive ID call.
#[derive(Clone, Debug)]
enum IdOutcome {
    Expr(ExprId),
    Fail(HedgeCertificate),
}

/// Budget of one ID recursion. Every recursive call (memo hits included) is charged.
///
/// Public entry points build [`IdMeter::admitted`]: a finite call cap taken from the active
/// workflow context (`ExecutionContext::id_work_limit`, the finite default when none is
/// installed) and a cancellation poll on that context. A caller that already holds its own
/// budget builds [`IdMeter::metered`], which reports `(recursion depth, estimated live bytes)`
/// to it instead; an error from the hook stops the recursion.
pub(crate) struct IdMeter<'m> {
    charge: Option<&'m mut (dyn FnMut(usize, u64) -> Result<(), IdentificationError> + 'm)>,
    depth: usize,
    calls: u64,
    /// Finite call cap and cancellation poll of the active workflow; `None` when a caller
    /// budget (`charge`) owns both.
    admitted_cap: Option<u64>,
}

impl<'m> IdMeter<'m> {
    /// A meter capped by the active workflow's general-ID work limit and its cancellation.
    fn admitted() -> Self {
        Self {
            charge: None,
            depth: 0,
            calls: 0,
            admitted_cap: Some(antecedent_core::numerical_admission::id_work_limit()),
        }
    }

    /// A meter reporting every call to the caller's own budget.
    fn metered(
        charge: &'m mut (dyn FnMut(usize, u64) -> Result<(), IdentificationError> + 'm),
    ) -> Self {
        Self { charge: Some(charge), depth: 0, calls: 0, admitted_cap: None }
    }

    /// Enter one recursive call and charge it: the live state is the expression
    /// arena and the memo table, estimated per node and per entry.
    fn enter(
        &mut self,
        arena_nodes: usize,
        memo_entries: usize,
    ) -> Result<(), IdentificationError> {
        self.depth += 1;
        self.calls = self.calls.saturating_add(1);
        if let Some(cap) = self.admitted_cap {
            if antecedent_core::numerical_admission::is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            if self.calls > cap {
                return Err(IdentificationError::budget(crate::error::IdentificationBudget::Steps));
            }
        }
        let bytes = (arena_nodes as u64)
            .saturating_mul(96)
            .saturating_add((memo_entries as u64).saturating_mul(192));
        match self.charge.as_mut() {
            Some(charge) => charge(self.depth, bytes),
            None => Ok(()),
        }
    }

    const fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

/// Result of [`identify_interventional_metered`].
#[allow(
    clippy::large_enum_variant,
    reason = "built once per district term and moved, never stored in bulk"
)]
pub(crate) enum MeteredId {
    /// `P_x(y)` as a functional of the observational law. The functional leaves
    /// the treatments free (no do-labels are baked in): evaluate it with the
    /// treatments bound to their levels.
    Identified {
        /// Arena holding the functional.
        arena: CausalExprArena,
        /// Root of the functional.
        expression: ExprId,
    },
    /// ID line 5: the hedge, carrying its problem so it re-verifies anywhere.
    Hedge(HedgeCertificate),
}

/// Complete ID of `P(y | do(x))` on `prepared`, charging every recursive call to
/// `charge` (the counterfactual-identification route's shared budget).
///
/// # Errors
///
/// The error `charge` returns when the budget stops the recursion, or an
/// identification failure (overlapping sets, unknown variables).
pub(crate) fn identify_interventional_metered(
    prepared: &PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    ws: &mut GraphWorkspace,
    charge: &mut (dyn FnMut(usize, u64) -> Result<(), IdentificationError> + '_),
) -> Result<MeteredId, IdentificationError> {
    require_disjoint(y, x)?;
    let mut arena = CausalExprArena::new();
    let mut derivation = DerivationTrace::default();
    let mut memo = IdMemo::default();
    let mut perf = IdentificationPerformanceRecord::default();
    let active = BitSet::full(prepared.admg().node_count());
    let mut meter = IdMeter::metered(charge);
    let mut run = IdRun {
        prepared,
        ancestors: AncestorMemo::default(),
        arena: &mut arena,
        memo: &mut memo,
        derivation: &mut derivation,
        perf: &mut perf,
        ws,
        meter: &mut meter,
    };
    match id_recurse(
        &mut run,
        IdQuery { y, x, v: &active, dist: &DistCtx::Marginal, assign: Arc::from([]) },
    )? {
        IdOutcome::Expr(expression) => Ok(MeteredId::Identified { arena, expression }),
        IdOutcome::Fail(hedge) => Ok(MeteredId::Hedge(
            hedge.with_problem(crate::hedge::HedgeProblem::capture(prepared, x, y)?),
        )),
    }
}

/// One ID subproblem with its own hard-set binding: `P(y | do(x))` where the free
/// do-labels are `assign`.
pub(crate) struct AssignedProblem {
    /// Outcome set.
    pub(crate) y: BitSet,
    /// Intervened set.
    pub(crate) x: BitSet,
    /// Hard-set values baked into the emitted factors.
    pub(crate) assign: Arc<[(DenseNodeId, Value)]>,
}

/// Result of identifying a list of [`AssignedProblem`]s in one run.
pub(crate) enum AssignedIds {
    /// One functional per problem, in order, all in the caller's arena.
    Identified(Vec<ExprId>),
    /// ID line 5 for the first problem that failed, carrying that problem.
    Hedge(HedgeCertificate),
}

/// Identify each problem with the complete ID algorithm over one shared memo and arena.
///
/// Each problem keeps its own `assign`: the same intervened node can be bound to a
/// different level in each emitted functional. The district formula of a path-specific
/// effect needs this, since each district's kernel reads the treatment at the level of the
/// treatment edges that enter that district.
///
/// # Errors
///
/// Overlapping or empty sets, or unknown variables.
pub(crate) fn identify_assigned_problems(
    prepared: &PreparedAdmg,
    arena: &mut CausalExprArena,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
    problems: &[AssignedProblem],
) -> Result<AssignedIds, IdentificationError> {
    let mut memo = IdMemo::default();
    let active = BitSet::full(prepared.admg().node_count());
    let mut meter = IdMeter::admitted();
    let mut run = IdRun {
        prepared,
        ancestors: AncestorMemo::default(),
        arena,
        memo: &mut memo,
        derivation,
        perf,
        ws,
        meter: &mut meter,
    };
    let mut functionals = Vec::with_capacity(problems.len());
    for problem in problems {
        require_disjoint(&problem.y, &problem.x)?;
        match id_recurse(
            &mut run,
            IdQuery {
                y: &problem.y,
                x: &problem.x,
                v: &active,
                dist: &DistCtx::Marginal,
                assign: Arc::clone(&problem.assign),
            },
        )? {
            IdOutcome::Expr(expression) => functionals.push(expression),
            IdOutcome::Fail(hedge) => {
                return Ok(AssignedIds::Hedge(hedge.with_problem(
                    crate::hedge::HedgeProblem::capture(prepared, &problem.x, &problem.y)?,
                )));
            }
        }
    }
    Ok(AssignedIds::Identified(functionals))
}

/// The two sides of a hard-set schedule contrast: the scheduled treatment nodes and the
/// level each side sets them all to.
#[derive(Debug, Clone, Copy)]
pub struct ScheduleContrast<'a> {
    /// Treatment nodes intervened on, one per scheduled time point.
    pub schedule: &'a [VariableId],
    /// Level every scheduled node is set to on the active side.
    pub active_level: &'a Value,
    /// Level every scheduled node is set to on the control side.
    pub control_level: &'a Value,
}

/// Identifier implementing the complete ID algorithm on ADMGs.
///
/// Every valid query over a valid ADMG ends in exactly one of two ways: an
/// identified functional of the observational law, or
/// [`IdentificationStatus::NotIdentified`](crate::result::IdentificationStatus)
/// with a [`HedgeCertificate`] that [`HedgeCertificate::verify`] accepts for
/// the original query. Lines 2, 6 and 7 operate on the *current* law `P′`, so
/// derivations that marginalize a line-7 C-factor and then factorize it again
/// (the napkin graph is the smallest) are carried through as ratios of
/// marginals of `P′`.
///
/// The functional may keep free variables beyond the treatments and outcomes:
/// line 3 intervenes on variables whose value cannot affect `Y`, and when such
/// a variable is later fixed outside `S′` by line 7 it stays free (the
/// napkin's `z`). The identity holds for every value of those variables with
/// positive support.
#[derive(Clone, Debug, Default)]
pub struct IdIdentifier;

impl IdIdentifier {
    /// Create the identifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Prepare an ADMG.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare(&self, graph: &Admg) -> Result<PreparedAdmg, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare an ADMG with declared assumptions.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Admg,
        assumptions: AssumptionSet,
    ) -> Result<PreparedAdmg, IdentificationError> {
        PreparedAdmg::with_assumptions(graph.clone(), assumptions)
    }

    /// Prepare by embedding a DAG (no latent confounding).
    ///
    /// # Errors
    ///
    /// Graph construction failure.
    pub fn prepare_dag(&self, graph: &Dag) -> Result<PreparedAdmg, IdentificationError> {
        PreparedAdmg::from_dag(graph)
    }

    /// Identify `P(Y | do(X))` (and ATE contrasts for average-effect queries).
    ///
    /// # Errors
    ///
    /// Unsupported query or unknown variables.
    pub fn identify(
        &self,
        prepared: &PreparedAdmg,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        match query {
            CausalQuery::AverageEffect(q) => self.identify_ate(prepared, q, workspace),
            CausalQuery::Distribution(q) => {
                // Unconditional interventional distribution via ID.
                // Nonempty conditioning belongs to IdcIdentifier / AutoIdentifier.
                validate_distribution_query(q)?;
                if !q.conditioning.is_empty() {
                    return Err(IdentificationError::unsupported(
                        "conditional Distribution requires IdcIdentifier (or AutoIdentifier)",
                    ));
                }
                crate::intervention_support::require_hard_set_interventions(
                    q.interventions.iter(),
                    "general ID",
                )?;
                // Flatten Sequence-of-Sets / Soft(constant) reductions for multi-do.
                let normalized = crate::intervention_support::normalize_intervention_list(
                    q.interventions.iter().cloned(),
                )?;
                let mut x = BitSet::with_len(prepared.admg().node_count());
                for intervention in &normalized {
                    let v = intervention.primary_variable().ok_or(
                        IdentificationError::unsupported("intervention missing primary variable"),
                    )?;
                    x.insert(prepared.var_to_dense(v)?);
                }
                let mut y = BitSet::with_len(prepared.admg().node_count());
                for &o in q.outcomes.iter() {
                    y.insert(prepared.var_to_dense(o)?);
                }
                Self::identify_sets(prepared, &y, &x, query.clone(), workspace, Arc::from([]))
            }
            CausalQuery::Response(response) => {
                self.identify_response(prepared, response, workspace)
            }
            _ => Err(IdentificationError::unsupported(
                "IdIdentifier supports AverageEffect, Distribution, and Response queries",
            )),
        }
    }

    /// Identify the requested intervention mean, retaining the actual Set levels.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per response functional, each assembling its own estimand from the shared prepared graph"
    )]
    pub(crate) fn identify_response(
        &self,
        prepared: &PreparedAdmg,
        response: &antecedent_core::ResponseQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        if let antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } =
            &response.functional
        {
            let levels = treatment.grid.values().map_err(|_| {
                IdentificationError::unsupported(
                    "MeanCurve general ID requires a finite evaluation grid",
                )
            })?;
            if levels.is_empty() {
                return Err(IdentificationError::unsupported(
                    "MeanCurve general ID requires a finite evaluation grid",
                ));
            }
            // Whether the mean is identified does not depend on the level, but the functional
            // does: every grid level gets its own estimand, in grid order, so the result never
            // describes the whole curve with one literal level.
            let mut merged: Option<IdentificationResult> = None;
            for &level in &levels {
                let mut level_query = response.clone();
                level_query.functional =
                    antecedent_core::ResponseFunctional::InterventionResponse {
                        outcome: *outcome,
                        interventions: Arc::from([Intervention::set(
                            treatment.variable,
                            Value::f64(level),
                        )]),
                    };
                let mut at_level = self.identify_response(prepared, &level_query, workspace)?;
                if at_level.estimands.is_empty() {
                    at_level.query = CausalQuery::Response(response.clone());
                    return Ok(at_level);
                }
                match merged.as_mut() {
                    None => merged = Some(at_level),
                    Some(curve) => {
                        for estimand in &at_level.estimands {
                            let functional =
                                curve.arena.import(&at_level.arena, estimand.functional);
                            let mut copy = estimand.clone();
                            copy.functional = functional;
                            curve.estimands.push(copy);
                        }
                        curve.performance.candidates_examined = curve
                            .performance
                            .candidates_examined
                            .saturating_add(at_level.performance.candidates_examined);
                        curve.performance.sets_returned = curve
                            .performance
                            .sets_returned
                            .saturating_add(at_level.performance.sets_returned);
                    }
                }
            }
            let mut result = merged.expect("the grid is non-empty");
            result.query = CausalQuery::Response(response.clone());
            result.derivation.push(
                "identify.response.general_id",
                format!(
                    "MeanCurve: one identified intervention mean per grid level, {} estimand(s) \
                     in grid order",
                    levels.len()
                ),
            );
            return Ok(result);
        }
        let antecedent_core::ResponseFunctional::InterventionResponse { outcome, interventions } =
            &response.functional
        else {
            let witness = crate::response_id::response_ate_witness(response)?;
            let mut result = self.identify_ate(prepared, &witness, workspace)?;
            result.query = CausalQuery::Response(response.clone());
            result.derivation.push(
                "identify.response.general_id",
                "binary contrast is an identification witness only; general-ID curve estimation is not supported",
            );
            return Ok(result);
        };
        crate::intervention_support::require_hard_set_interventions(
            interventions.iter(),
            "general ID response",
        )?;
        let normalized = crate::intervention_support::normalize_intervention_list(
            interventions.iter().cloned(),
        )?;
        let mut x = BitSet::with_len(prepared.admg().node_count());
        let mut assignments = Vec::new();
        for intervention in &normalized {
            let variable = intervention
                .primary_variable()
                .ok_or(IdentificationError::unsupported("intervention missing primary variable"))?;
            let dense = prepared.var_to_dense(variable)?;
            x.insert(dense);
            assignments.push((dense, intervention_value(intervention)?));
        }
        let mut y = BitSet::with_len(prepared.admg().node_count());
        y.insert(prepared.var_to_dense(*outcome)?);
        let mut result = Self::identify_sets(
            prepared,
            &y,
            &x,
            CausalQuery::Response(response.clone()),
            workspace,
            Arc::from(assignments),
        )?;
        for estimand in &mut result.estimands {
            estimand.functional = result.arena.intern(ExprNode::Expectation {
                function: OutcomeExprId::identity(*outcome),
                distribution: estimand.functional,
            });
        }
        result.derivation.push(
            "identify.response.general_id",
            "Shpitser–Pearl ID of the intervention mean at the requested Set levels",
        );
        Ok(result)
    }

    /// Identify an average treatment effect via ID on `{treatment}` → `{outcome}`.
    ///
    /// # Errors
    ///
    /// Unknown variables or identification failure plumbing.
    pub fn identify_ate(
        &self,
        prepared: &PreparedAdmg,
        query: &AverageEffectQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::Identification,
            || Self::identify_ate_impl(prepared, query, workspace),
        )
    }

    fn identify_ate_impl(
        prepared: &PreparedAdmg,
        query: &AverageEffectQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        crate::execution_counts::note_check();
        query
            .validate()
            .map_err(|_| IdentificationError::unsupported("invalid average-effect query"))?;
        crate::identification_run::require_marginal_target(query)?;
        let t = prepared.var_to_dense(query.treatment)?;
        let y = prepared.var_to_dense(query.outcome)?;
        let mut y_set = BitSet::with_len(prepared.admg().node_count());
        y_set.insert(y);
        let mut x_set = BitSet::with_len(prepared.admg().node_count());
        x_set.insert(t);

        let mut arena = CausalExprArena::new();
        let mut derivation = DerivationTrace::default();
        derivation.push("general.id", "Shpitser–Pearl ID for ATE contrast");
        let mut memo = IdMemo::default();
        let mut perf = IdentificationPerformanceRecord::default();

        let active = BitSet::full(prepared.admg().node_count());
        let active_level = intervention_value(&query.active)?;
        let control_level = intervention_value(&query.control)?;

        // One budget across both sides of the contrast.
        let mut meter = IdMeter::admitted();
        let mut side = |assign: Assign,
                        prepared: &PreparedAdmg,
                        arena: &mut CausalExprArena,
                        memo: &mut IdMemo,
                        derivation: &mut DerivationTrace,
                        perf: &mut IdentificationPerformanceRecord|
         -> Result<IdOutcome, IdentificationError> {
            let mut run = IdRun {
                prepared,
                ancestors: AncestorMemo::default(),
                arena,
                memo,
                derivation,
                perf,
                ws: &mut workspace.graph,
                meter: &mut meter,
            };
            id_recurse(
                &mut run,
                IdQuery { y: &y_set, x: &x_set, v: &active, dist: &DistCtx::Marginal, assign },
            )
        };
        let left = match side(
            Arc::from([(t, active_level)]),
            prepared,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
        )? {
            IdOutcome::Expr(e) => e,
            IdOutcome::Fail(hedge) => {
                return Ok(not_identified_with_hedge(
                    CausalQuery::AverageEffect(query.clone()),
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                    hedge,
                ));
            }
        };
        let right = match side(
            Arc::from([(t, control_level)]),
            prepared,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
        )? {
            IdOutcome::Expr(e) => e,
            IdOutcome::Fail(hedge) => {
                return Ok(not_identified_with_hedge(
                    CausalQuery::AverageEffect(query.clone()),
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                    hedge,
                ));
            }
        };

        let functional = expectation_contrast(&mut arena, query.outcome, left, right)?;
        let estimand = IdentifiedEstimand::new(
            Arc::from(EstimandMethod::GeneralId.as_str()),
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
            functional,
            None,
        );
        Ok(IdentificationResult::identified(
            CausalQuery::AverageEffect(query.clone()),
            vec![estimand],
            arena,
            derivation,
            with_causal_markov(prepared, "general.id"),
            perf,
        ))
    }

    /// Identify the two-sided contrast
    /// `E[Y | do(X = active)] − E[Y | do(X = control)]` for a multi-node
    /// hard-set schedule (temporal unfoldings intervene on one treatment node
    /// per scheduled time point, all at the same level per side).
    ///
    /// Mirrors the single-treatment ATE path: two ID passes with the level
    /// baked into emitted factors, combined by an expectation contrast. The
    /// historical sustained/dynamic path identified only the active side and
    /// relabeled the one-sided distribution as a temporal effect.
    ///
    /// # Errors
    ///
    /// Unknown variables or empty schedules.
    pub fn identify_schedule_contrast(
        &self,
        prepared: &PreparedAdmg,
        outcome: VariableId,
        contrast: &ScheduleContrast<'_>,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let ScheduleContrast { schedule, active_level, control_level } = *contrast;
        if schedule.is_empty() {
            return Err(IdentificationError::EmptyTreatmentSchedule);
        }
        let mut arena = CausalExprArena::new();
        let mut derivation = DerivationTrace::default();
        derivation.push("general.id", "Shpitser–Pearl ID for schedule contrast");
        let mut memo = IdMemo::default();
        let mut perf = IdentificationPerformanceRecord::default();
        let active_set = BitSet::full(prepared.admg().node_count());
        let (x_set, schedule_dense, y_set) = schedule_sets(prepared, schedule, outcome)?;

        let assign_for = |level: &Value| -> Assign {
            schedule_dense.iter().map(|&d| (d, level.clone())).collect()
        };
        // One budget across both sides of the contrast.
        let mut meter = IdMeter::admitted();
        let mut side = |assign: Assign,
                        prepared: &PreparedAdmg,
                        arena: &mut CausalExprArena,
                        memo: &mut IdMemo,
                        derivation: &mut DerivationTrace,
                        perf: &mut IdentificationPerformanceRecord|
         -> Result<IdOutcome, IdentificationError> {
            let mut run = IdRun {
                prepared,
                ancestors: AncestorMemo::default(),
                arena,
                memo,
                derivation,
                perf,
                ws: &mut workspace.graph,
                meter: &mut meter,
            };
            id_recurse(
                &mut run,
                IdQuery { y: &y_set, x: &x_set, v: &active_set, dist: &DistCtx::Marginal, assign },
            )
        };
        let left = match side(
            assign_for(active_level),
            prepared,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
        )? {
            IdOutcome::Expr(e) => e,
            IdOutcome::Fail(hedge) => {
                return Ok(not_identified_with_hedge(
                    query,
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                    hedge,
                ));
            }
        };
        let right = match side(
            assign_for(control_level),
            prepared,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
        )? {
            IdOutcome::Expr(e) => e,
            IdOutcome::Fail(hedge) => {
                return Ok(not_identified_with_hedge(
                    query,
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                    hedge,
                ));
            }
        };

        let functional = expectation_contrast(&mut arena, outcome, left, right)?;
        let estimand = IdentifiedEstimand::new(
            Arc::from(EstimandMethod::GeneralId.as_str()),
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
            functional,
            None,
        );
        Ok(IdentificationResult::identified(
            query,
            vec![estimand],
            arena,
            derivation,
            with_causal_markov(prepared, "general.id"),
            perf,
        ))
    }

    pub(crate) fn identify_sets(
        prepared: &PreparedAdmg,
        y: &BitSet,
        x: &BitSet,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
        assignments: Arc<[(DenseNodeId, Value)]>,
    ) -> Result<IdentificationResult, IdentificationError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::Identification,
            || {
                crate::execution_counts::note_check();
                require_disjoint(y, x)?;
                let mut arena = CausalExprArena::new();
                let mut derivation = DerivationTrace::default();
                derivation.push("general.id", "Shpitser–Pearl ID");
                let mut memo = IdMemo::default();
                let mut perf = IdentificationPerformanceRecord::default();
                let active = BitSet::full(prepared.admg().node_count());
                let mut run = IdRun {
                    prepared,
                    ancestors: AncestorMemo::default(),
                    arena: &mut arena,
                    memo: &mut memo,
                    derivation: &mut derivation,
                    perf: &mut perf,
                    ws: &mut workspace.graph,
                    meter: &mut IdMeter::admitted(),
                };
                match id_recurse(
                    &mut run,
                    IdQuery { y, x, v: &active, dist: &DistCtx::Marginal, assign: assignments },
                )? {
                    IdOutcome::Expr(functional) => {
                        let estimand = IdentifiedEstimand::new(
                            Arc::from(EstimandMethod::GeneralId.as_str()),
                            Arc::from([]),
                            Arc::from([]),
                            Arc::from([]),
                            functional,
                            None,
                        );
                        Ok(IdentificationResult::identified(
                            query,
                            vec![estimand],
                            arena,
                            derivation,
                            with_causal_markov(prepared, "general.id"),
                            perf,
                        ))
                    }
                    IdOutcome::Fail(hedge) => {
                        let hedge = hedge
                            .with_problem(crate::hedge::HedgeProblem::capture(prepared, x, y)?);
                        Ok(not_identified_with_hedge(
                            query,
                            derivation,
                            prepared.declared_assumptions().clone(),
                            perf,
                            hedge,
                        ))
                    }
                }
            },
        )
    }
}

/// Declared assumptions plus the Causal Markov condition, attributed to
/// `algorithm`: truncated-factorization / g-formula identification (every
/// success path of general ID) depends on it, so it belongs alongside
/// whatever the caller declared on every identified result.
pub(crate) fn with_causal_markov(prepared: &PreparedAdmg, algorithm: &str) -> AssumptionSet {
    let mut assumptions = prepared.declared_assumptions().clone();
    assumptions.push(crate::assumptions::causal_markov(algorithm));
    assumptions
}

/// Reject a malformed distribution query before any graph work.
pub(crate) fn validate_distribution_query(
    q: &antecedent_core::InterventionalDistributionQuery,
) -> Result<(), IdentificationError> {
    q.validate().map_err(|e| IdentificationError::InvalidQuery { message: e.to_string() })
}

/// ID is defined for disjoint `Y` and `X` with `Y ≠ ∅`.
///
/// `P(A | do(A))` is the point mass the query itself fixes; running the
/// recursion on it would reach `V \ X = ∅` and hand back the *observational*
/// `P(A)` as if it were the interventional law. This guard sits on the dense
/// sets so that it also covers entries with no query object to validate and
/// interventions that only resolve to a variable after normalization.
/// Treatment set, dense schedule and outcome set of a schedule contrast; the outcome must not
/// be scheduled.
fn schedule_sets(
    prepared: &PreparedAdmg,
    schedule: &[VariableId],
    outcome: VariableId,
) -> Result<(BitSet, Vec<DenseNodeId>, BitSet), IdentificationError> {
    let mut x_set = BitSet::with_len(prepared.admg().node_count());
    let mut schedule_dense = Vec::with_capacity(schedule.len());
    for &t in schedule {
        let d = prepared.var_to_dense(t)?;
        x_set.insert(d);
        schedule_dense.push(d);
    }
    let mut y_set = BitSet::with_len(prepared.admg().node_count());
    y_set.insert(prepared.var_to_dense(outcome)?);
    require_disjoint(&y_set, &x_set)?;
    Ok((x_set, schedule_dense, y_set))
}

fn require_disjoint(y: &BitSet, x: &BitSet) -> Result<(), IdentificationError> {
    if !y.any() {
        return Err(IdentificationError::unsupported("general ID requires at least one outcome"));
    }
    if y.iter().any(|node| x.contains(node)) {
        return Err(IdentificationError::unsupported(
            "general ID requires outcomes disjoint from the intervened variables",
        ));
    }
    Ok(())
}

/// Assemble `E[outcome | left] − E[outcome | right]` and simplify.
///
/// A dead sum/integral here means the assembled functional is ill-formed,
/// which must surface rather than be silently rewritten away.
fn expectation_contrast(
    arena: &mut CausalExprArena,
    outcome: VariableId,
    left: ExprId,
    right: ExprId,
) -> Result<ExprId, IdentificationError> {
    let left_exp = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: left,
    });
    let right_exp = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: right,
    });
    let contrast = arena.intern(ExprNode::Contrast {
        left: left_exp,
        right: right_exp,
        op: ContrastOp::Difference,
    });
    arena.simplify(contrast).map_err(IdentificationError::expression)
}

fn intervention_value(intervention: &Intervention) -> Result<Value, IdentificationError> {
    crate::intervention_support::require_set_value(intervention, "general ID ATE")
}

fn not_identified_with_hedge(
    query: CausalQuery,
    mut derivation: DerivationTrace,
    assumptions: AssumptionSet,
    performance: IdentificationPerformanceRecord,
    hedge: HedgeCertificate,
) -> IdentificationResult {
    derivation.push(
        "general.id.hedge",
        format!(
            "hedge F={:?} F'={:?}",
            hedge.f.iter().map(|v| v.raw()).collect::<Vec<_>>(),
            hedge.f_prime.iter().map(|v| v.raw()).collect::<Vec<_>>()
        ),
    );
    let diagnostics = vec![Diagnostic {
        code: Arc::from("identify.hedge"),
        kind: DiagnosticKind::Scientific,
        severity: DiagnosticSeverity::Error,
        message: Arc::from(format!(
            "effect not identifiable; hedge F size {} / F' size {}",
            hedge.f.len(),
            hedge.f_prime.len()
        )),
        artifact_id: None,
        fields: Arc::from([
            (
                Arc::from("f"),
                Arc::from(
                    hedge.f.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(","),
                ),
            ),
            (
                Arc::from("f_prime"),
                Arc::from(
                    hedge.f_prime.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(","),
                ),
            ),
        ]),
    }];
    IdentificationResult::not_identified_hedge(
        query,
        derivation,
        assumptions,
        performance,
        hedge,
        diagnostics,
    )
}

#[cfg(test)]
mod tests;
