//! Shpitser & Pearl (2006) ID algorithm for semi-Markovian models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments, clippy::unused_self)]

use std::collections::HashMap;
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
use crate::prepared::PreparedAdmg;
use crate::result::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

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

/// Outcome of a recursive ID call.
#[derive(Clone, Debug)]
enum IdOutcome {
    Expr(ExprId),
    Fail(HedgeCertificate),
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
                self.identify_sets(prepared, &y, &x, query.clone(), workspace, Arc::from([]))
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
    #[allow(clippy::too_many_lines)]
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
                "binary contrast is an identification witness only; general-ID curve estimation is not licensed",
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
        let mut result = self.identify_sets(
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
        query
            .validate()
            .map_err(|_| IdentificationError::unsupported("invalid average-effect query"))?;
        let t = prepared.var_to_dense(query.treatment)?;
        let y = prepared.var_to_dense(query.outcome)?;
        let mut y_set = BitSet::with_len(prepared.admg().node_count());
        y_set.insert(y);
        let mut x_set = BitSet::with_len(prepared.admg().node_count());
        x_set.insert(t);

        let mut prepared = prepared.clone();
        let mut arena = CausalExprArena::new();
        let mut derivation = DerivationTrace::default();
        derivation.push("general.id", "Shpitser–Pearl ID for ATE contrast");
        let mut memo: HashMap<SubproblemKey, IdOutcome> = HashMap::new();
        let mut perf = IdentificationPerformanceRecord::default();

        let active = full_nodes(prepared.admg().node_count());
        let active_level = intervention_value(&query.active)?;
        let control_level = intervention_value(&query.control)?;

        let left = match id_recurse(
            &mut prepared,
            &y_set,
            &x_set,
            &active,
            &DistCtx::Marginal,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
            &mut workspace.graph,
            Arc::from([(t, active_level)]),
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
        let right = match id_recurse(
            &mut prepared,
            &y_set,
            &x_set,
            &active,
            &DistCtx::Marginal,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
            &mut workspace.graph,
            Arc::from([(t, control_level)]),
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
            with_causal_markov(&prepared, "general.id"),
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
        schedule: &[VariableId],
        active_level: &Value,
        control_level: &Value,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        if schedule.is_empty() {
            return Err(IdentificationError::msg("empty treatment schedule"));
        }
        let mut prepared = prepared.clone();
        let mut arena = CausalExprArena::new();
        let mut derivation = DerivationTrace::default();
        derivation.push("general.id", "Shpitser–Pearl ID for schedule contrast");
        let mut memo: HashMap<SubproblemKey, IdOutcome> = HashMap::new();
        let mut perf = IdentificationPerformanceRecord::default();
        let active_set = full_nodes(prepared.admg().node_count());

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

        let assign_for = |level: &Value| -> Assign {
            schedule_dense.iter().map(|&d| (d, level.clone())).collect()
        };
        let mut side = |assign: Assign,
                        prepared: &mut PreparedAdmg,
                        arena: &mut CausalExprArena,
                        memo: &mut HashMap<SubproblemKey, IdOutcome>,
                        derivation: &mut DerivationTrace,
                        perf: &mut IdentificationPerformanceRecord|
         -> Result<IdOutcome, IdentificationError> {
            id_recurse(
                prepared,
                &y_set,
                &x_set,
                &active_set,
                &DistCtx::Marginal,
                arena,
                memo,
                derivation,
                perf,
                &mut workspace.graph,
                assign,
            )
        };
        let left = match side(
            assign_for(active_level),
            &mut prepared,
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
            &mut prepared,
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
            with_causal_markov(&prepared, "general.id"),
            perf,
        ))
    }

    fn identify_sets(
        &self,
        prepared: &PreparedAdmg,
        y: &BitSet,
        x: &BitSet,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
        assignments: Arc<[(DenseNodeId, Value)]>,
    ) -> Result<IdentificationResult, IdentificationError> {
        require_disjoint(y, x)?;
        let mut prepared = prepared.clone();
        let mut arena = CausalExprArena::new();
        let mut derivation = DerivationTrace::default();
        derivation.push("general.id", "Shpitser–Pearl ID");
        let mut memo: HashMap<SubproblemKey, IdOutcome> = HashMap::new();
        let mut perf = IdentificationPerformanceRecord::default();
        let active = full_nodes(prepared.admg().node_count());
        match id_recurse(
            &mut prepared,
            y,
            x,
            &active,
            &DistCtx::Marginal,
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
            &mut workspace.graph,
            assignments,
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
                    with_causal_markov(&prepared, "general.id"),
                    perf,
                ))
            }
            IdOutcome::Fail(hedge) => {
                let hedge =
                    hedge.with_problem(crate::hedge::HedgeProblem::capture(&prepared, x, y)?);
                Ok(not_identified_with_hedge(
                    query,
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                    hedge,
                ))
            }
        }
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
fn require_disjoint(y: &BitSet, x: &BitSet) -> Result<(), IdentificationError> {
    if !y.any() {
        return Err(IdentificationError::unsupported("general ID requires at least one outcome"));
    }
    if y.to_dense_ids().iter().any(|node| x.contains(*node)) {
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
    arena.simplify(contrast).map_err(|e| IdentificationError::msg(e.to_string()))
}

fn full_nodes(n: usize) -> BitSet {
    let mut b = BitSet::with_len(n);
    for i in 0..n {
        b.insert(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
    }
    b
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

/// Run ID; returns expression for `P_x(y)` over observational factors in `arena`.
fn id_recurse(
    prepared: &mut PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    v: &BitSet,
    dist: &DistCtx,
    arena: &mut CausalExprArena,
    memo: &mut HashMap<SubproblemKey, IdOutcome>,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
    assign: Assign,
) -> Result<IdOutcome, IdentificationError> {
    perf.candidates_examined = perf.candidates_examined.saturating_add(1);
    let key = SubproblemKey {
        y: y.clone(),
        x: x.clone(),
        v: v.clone(),
        assign: assign.clone(),
        dist: dist.clone(),
    };
    if let Some(hit) = memo.get(&key) {
        perf.sets_returned = perf.sets_returned.saturating_add(1);
        return Ok(hit.clone());
    }

    let outcome = id_body(prepared, y, x, v, dist, arena, memo, derivation, perf, ws, assign)?;
    memo.insert(key, outcome.clone());
    Ok(outcome)
}

fn id_body(
    prepared: &mut PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    v: &BitSet,
    dist: &DistCtx,
    arena: &mut CausalExprArena,
    memo: &mut HashMap<SubproblemKey, IdOutcome>,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
    assign: Assign,
) -> Result<IdOutcome, IdentificationError> {
    // Line 1: x = ∅ → ∑_{v\y} of the *current* distribution
    if !x.any() {
        derivation.push("general.id.line1", "empty intervention; marginal of current dist");
        return Ok(IdOutcome::Expr(dist_marginal(prepared, dist, y, v, arena, assign)?));
    }

    // Line 2: restrict to An(Y)_G; the current distribution marginalizes over
    // the removed set (a Marginal stays a Marginal; a CFactor grows its sumset).
    let an_y = prepared.ancestors_within(y, v, ws);
    if !v.equal_set(&an_y) {
        let mut x2 = x.clone();
        x2.intersect_with(&an_y);
        let mut removed = v.clone();
        removed.difference_with(&an_y);
        let dist2 = dist.marginalize(&removed);
        derivation.push("general.id.line2", "restrict to ancestral set of Y");
        return id_recurse(
            prepared, y, &x2, &an_y, &dist2, arena, memo, derivation, perf, ws, assign,
        );
    }

    // Line 3: W = (V\X) \ An(Y)_{G_{\bar X}} — only X changes; dist unchanged.
    let mut v_minus_x = v.clone();
    v_minus_x.difference_with(x);
    let an_bar = prepared.ancestors_bar_x(y, v, x, ws);
    let mut w = v_minus_x.clone();
    w.difference_with(&an_bar);
    if w.any() {
        let mut x2 = x.clone();
        x2.union_with(&w);
        derivation.push("general.id.line3", "add superfluous interventions");
        return id_recurse(prepared, y, &x2, v, dist, arena, memo, derivation, perf, ws, assign);
    }

    // Line 4 / 5–7: C-components of G[V\X]
    let comps = prepared.c_components(&v_minus_x);
    if comps.is_empty() {
        // Not a published line, and unreachable: every entry point enforces
        // ∅ ≠ Y, Y ∩ X = ∅, and each line preserves Y ⊆ V \ X (line 2 keeps
        // An(Y) ⊇ Y, line 3 only adds non-ancestors of Y to X, line 4 recurses
        // on Y = S_i with X = V \ S_i, line 7 keeps Y ⊆ S ⊆ S′). An empty
        // V \ X would mean Y ⊆ X, where the answer is a point mass at x and
        // never a marginal of the current law, so nothing is emitted for it.
        return Err(IdentificationError::InvariantViolated {
            message: "general ID: outcomes are not disjoint from interventions",
        });
    }

    if comps.len() > 1 {
        derivation
            .push("general.id.line4", format!("C-component factorization ({} parts)", comps.len()));
        let mut factors = Vec::with_capacity(comps.len());
        for s_i in &comps {
            let mut x_i = v.clone();
            x_i.difference_with(s_i);
            match id_recurse(
                prepared,
                s_i,
                &x_i,
                v,
                dist,
                arena,
                memo,
                derivation,
                perf,
                ws,
                assign.clone(),
            )? {
                IdOutcome::Expr(e) => factors.push(e),
                fail @ IdOutcome::Fail(_) => return Ok(fail),
            }
        }
        let product = {
            let list = arena.intern_list(factors);
            arena.intern(ExprNode::Product(list))
        };
        // ∑_{v \ (y ∪ x)}
        let mut sum_vars = v.clone();
        sum_vars.difference_with(y);
        sum_vars.difference_with(x);
        let expr = if sum_vars.any() {
            let vs = intern_nodes(prepared, &sum_vars, arena)?;
            arena.intern(ExprNode::SumOut { variables: vs, expr: product })
        } else {
            product
        };
        return Ok(IdOutcome::Expr(expr));
    }

    // Single C-component S of G[V\X]
    let s = &comps[0];
    // Line 5: C(G) = {G} → FAIL
    if prepared.is_single_c_component(v) {
        derivation.push("general.id.line5", "hedge: G is a single C-component");
        // A certificate names real variables or is not issued: a node without
        // a variable identity fails the call rather than being given a made-up id.
        for node in v.to_dense_ids() {
            prepared.dense_to_var(node)?;
        }
        let hedge = HedgeCertificate::from_sets(v, s, |d| {
            prepared.dense_to_var(d).expect("every node of F was resolved above")
        });
        return Ok(IdOutcome::Fail(hedge));
    }

    id_lines_5_to_7(prepared, y, x, v, s, dist, arena, memo, derivation, perf, ws, assign)
}

/// Lines 6–7 dispatch for the single-C-component case (line 5 handled above).
#[allow(clippy::too_many_arguments)]
fn id_lines_5_to_7(
    prepared: &mut PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    v: &BitSet,
    s: &BitSet,
    dist: &DistCtx,
    arena: &mut CausalExprArena,
    memo: &mut HashMap<SubproblemKey, IdOutcome>,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
    assign: Assign,
) -> Result<IdOutcome, IdentificationError> {
    // Districts of G (on V)
    let g_comps = prepared.c_components(v);
    // Line 6: S ∈ C(G) — emit the C-factor of the *current* distribution.
    if g_comps.iter().any(|c| c.equal_set(s)) {
        derivation.push("general.id.line6", "S is a C-component of G; factorize current dist");
        let expr = match dist {
            DistCtx::Marginal => c_component_expression(prepared, s, y, v, arena, assign)?,
            DistCtx::CFactor(law) => {
                // ∑_{s\y} ∏_{Vi∈S} P′(vi | v_π^{(i-1)}): with an empty sumset the
                // telescope collapses each conditional to its own frozen factor;
                // after a line-2 marginalization they are ratios of marginals.
                let mut sum_vars = s.clone();
                sum_vars.difference_with(y);
                let sub = Law { sumset: sum_vars, factors: law.conditionals(prepared, s, v)? };
                sub.emit(prepared, &BitSet::with_len(v.bit_len()), arena, &assign)?
            }
        };
        return Ok(IdOutcome::Expr(expr));
    }

    // Line 7: ∃ S' ⊃ S, S' ∈ C(G). Recurse on G_{S'} against
    // P′ = ∏_{Vi∈S′} P(vi | v_π^{(i-1)} ∩ S′, v_π^{(i-1)} \ S′), the C-factor
    // of the current distribution — its factors keep conditioning on
    // topological predecessors *outside* S' (Tian's telescope).
    if let Some(s_prime) = g_comps.iter().find(|c| s.is_subset_of(c) && !c.equal_set(s)) {
        derivation.push("general.id.line7", "recurse into containing C-component S' against Q[S']");
        let mut x2 = x.clone();
        x2.intersect_with(s_prime);
        let factors = match dist {
            DistCtx::Marginal => marginal_conditionals(prepared, s_prime, v),
            DistCtx::CFactor(law) => law.conditionals(prepared, s_prime, v)?,
        };
        let dist2 =
            DistCtx::CFactor(Arc::new(Law { sumset: BitSet::with_len(v.bit_len()), factors }));
        return id_recurse(
            prepared, y, &x2, s_prime, &dist2, arena, memo, derivation, perf, ws, assign,
        );
    }

    Err(IdentificationError::InvariantViolated {
        message: "general ID: no line applies to the current C-component state",
    })
}

/// Chain-rule conditionals `P(vi | v_π^{(i-1)})` of the observational marginal
/// over `v`, for `vi ∈ s`, in topological order.
fn marginal_conditionals(prepared: &PreparedAdmg, s: &BitSet, v: &BitSet) -> Vec<Factor> {
    let mut factors = Vec::new();
    let mut preceding = BitSet::with_len(v.bit_len());
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        if s.contains(vi) {
            factors.push(Factor::Observed { var: vi, cond: preceding.clone() });
        }
        preceding.insert(vi);
    }
    factors
}

impl DistCtx {
    /// Marginalize the current distribution over `removed` (line 2).
    fn marginalize(&self, removed: &BitSet) -> Self {
        match self {
            // A marginal of the observational marginal is still a marginal.
            Self::Marginal => Self::Marginal,
            Self::CFactor(law) => {
                let mut sumset = law.sumset.clone();
                sumset.union_with(removed);
                Self::CFactor(Arc::new(Law { sumset, factors: law.factors.clone() }))
            }
        }
    }
}

impl Law {
    /// Conditionals `P′(vi | v_π^{(i-1)})` of this law over `v`, for `vi ∈ s`,
    /// in topological order.
    ///
    /// With an empty sumset every factor after `vi` sums to one without
    /// touching the earlier factors, so `∑_{later} P′ = ∏_{j ≤ i} f_j` and the
    /// conditional is `vi`'s own factor. Once line 2 has summed variables out
    /// of the product that telescope no longer holds node-wise, and the
    /// conditional is formed from its definition,
    /// `∑_{later} P′ / ∑_{vi} ∑_{later} P′`.
    fn conditionals(
        &self,
        prepared: &PreparedAdmg,
        s: &BitSet,
        v: &BitSet,
    ) -> Result<Vec<Factor>, IdentificationError> {
        if !self.sumset.any() {
            let kept: Vec<Factor> =
                self.factors.iter().filter(|f| s.contains(f.var())).cloned().collect();
            if kept.len() != s.to_dense_ids().len() {
                return Err(IdentificationError::InvariantViolated {
                    message: "general ID: current law does not cover the requested C-component",
                });
            }
            return Ok(kept);
        }
        let mut out = Vec::new();
        let mut later = v.clone();
        for &vi in prepared.topo() {
            if !v.contains(vi) {
                continue;
            }
            later.remove(vi);
            if s.contains(vi) {
                let mut sumset = self.sumset.clone();
                sumset.union_with(&later);
                let num = Arc::new(Self { sumset, factors: self.factors.clone() });
                out.push(Factor::Conditional { var: vi, num });
            }
        }
        Ok(out)
    }

    /// Emit `∑_{sumset} ∏ factors`.
    ///
    /// `bound` holds the variables already bound by enclosing sums of this
    /// emission. `do(·)` labels are applied only to factors whose assigned
    /// variable is *free* — a bound occurrence is the sum's dummy variable,
    /// not the do-value.
    fn emit(
        &self,
        prepared: &PreparedAdmg,
        bound: &BitSet,
        arena: &mut CausalExprArena,
        assign: &Assign,
    ) -> Result<ExprId, IdentificationError> {
        let mut bound = bound.clone();
        bound.union_with(&self.sumset);
        let mut exprs = Vec::with_capacity(self.factors.len());
        for factor in &self.factors {
            exprs.push(factor.emit(prepared, &bound, arena, assign)?);
        }
        let product = if exprs.len() == 1 {
            exprs[0]
        } else {
            let list = arena.intern_list(exprs);
            arena.intern(ExprNode::Product(list))
        };
        if self.sumset.any() {
            let vs = intern_nodes(prepared, &self.sumset, arena)?;
            Ok(arena.intern(ExprNode::SumOut { variables: vs, expr: product }))
        } else {
            Ok(product)
        }
    }
}

impl Factor {
    fn emit(
        &self,
        prepared: &PreparedAdmg,
        bound: &BitSet,
        arena: &mut CausalExprArena,
        assign: &Assign,
    ) -> Result<ExprId, IdentificationError> {
        match self {
            Self::Observed { var, cond } => {
                let effective_assign: Assign =
                    assign.iter().filter(|(t, _)| !bound.contains(*t)).cloned().collect();
                let var_i = prepared.dense_to_var(*var)?;
                let vars = arena.intern_var_set([var_i]);
                let cond_vars: Result<Vec<_>, _> =
                    cond.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
                let cond_vars = cond_vars?;
                let conditioned_on = arena.intern_var_set(cond_vars.clone());
                let (intervention, domain) =
                    intervention_for_factor(arena, prepared, &effective_assign, *var, &cond_vars)?;
                Ok(arena.intern_distribution(vars, conditioned_on, intervention, domain))
            }
            Self::Conditional { var, num } => {
                let numerator = num.emit(prepared, bound, arena, assign)?;
                // The denominator is literally `∑_{var}` of the numerator node,
                // so an evaluator can recognize a conditional on a null event
                // (0/0 with a bounded extension) instead of a generic 0/0. Only
                // when `var` carries a free do-label must the body be re-emitted
                // with `var` bound, since the label would otherwise pin the
                // summation variable.
                let labelled = !bound.contains(*var) && assign.iter().any(|(t, _)| t == var);
                let body = if labelled {
                    let mut inner = bound.clone();
                    inner.insert(*var);
                    num.emit(prepared, &inner, arena, assign)?
                } else {
                    numerator
                };
                let variables = arena.intern_var_set([prepared.dense_to_var(*var)?]);
                let denominator = arena.intern(ExprNode::SumOut { variables, expr: body });
                Ok(arena.intern(ExprNode::Ratio { numerator, denominator }))
            }
        }
    }
}

/// Marginal `∑_{v\y}` of the current distribution (lines 1 and degenerate).
fn dist_marginal(
    prepared: &PreparedAdmg,
    dist: &DistCtx,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Assign,
) -> Result<ExprId, IdentificationError> {
    match dist {
        DistCtx::Marginal => observational_marginal(prepared, y, v, arena, assign),
        DistCtx::CFactor(law) => {
            // Sum over everything in v except y, plus the carried sumset.
            let mut sumset = v.clone();
            sumset.difference_with(y);
            sumset.union_with(&law.sumset);
            let marginal = Law { sumset, factors: law.factors.clone() };
            marginal.emit(prepared, &BitSet::with_len(v.bit_len()), arena, &assign)
        }
    }
}

fn intern_nodes(
    prepared: &PreparedAdmg,
    nodes: &BitSet,
    arena: &mut CausalExprArena,
) -> Result<antecedent_expr::VarSetId, IdentificationError> {
    let vars: Result<Vec<_>, _> =
        nodes.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
    Ok(arena.intern_var_set(vars?))
}

fn observational_marginal(
    prepared: &PreparedAdmg,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Assign,
) -> Result<ExprId, IdentificationError> {
    // Tian / Shpitser–Pearl: P(V) = ∏_{S ∈ C(G[V])} Q[S], Q[S] = ∏_{Vi∈S} P(Vi | V^π_<i).
    // On a DAG, C-components are singletons and this reduces to the usual Markov product
    // (conditioning on extra predecessors is redundant given pa(Vi)). On an ADMG with
    // bidirected edges, ∏ P(vi | pa(vi)) is not the observational joint.
    let comps = prepared.c_components(v);
    let factors = if comps.is_empty() {
        markov_product(prepared, v, arena, assign)?
    } else if comps.len() == 1 {
        q_component_product(prepared, &comps[0], v, arena, assign)?
    } else {
        let mut parts = Vec::with_capacity(comps.len());
        for s in &comps {
            parts.push(q_component_product(prepared, s, v, arena, assign.clone())?);
        }
        if parts.len() == 1 {
            parts[0]
        } else {
            let list = arena.intern_list(parts);
            arena.intern(ExprNode::Product(list))
        }
    };
    let mut sum_vars = v.clone();
    sum_vars.difference_with(y);
    if sum_vars.any() {
        let vs = intern_nodes(prepared, &sum_vars, arena)?;
        Ok(arena.intern(ExprNode::SumOut { variables: vs, expr: factors }))
    } else {
        Ok(factors)
    }
}

fn c_component_expression(
    prepared: &PreparedAdmg,
    s: &BitSet,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Assign,
) -> Result<ExprId, IdentificationError> {
    // ∑_{s\y} ∏_{Vi∈S} P(vi | v^{π}_{<i})
    let product = q_component_product(prepared, s, v, arena, assign)?;
    let mut sum_vars = s.clone();
    sum_vars.difference_with(y);
    if sum_vars.any() {
        let vs = intern_nodes(prepared, &sum_vars, arena)?;
        Ok(arena.intern(ExprNode::SumOut { variables: vs, expr: product }))
    } else {
        Ok(product)
    }
}

fn q_component_product(
    prepared: &PreparedAdmg,
    s: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Assign,
) -> Result<ExprId, IdentificationError> {
    let empty_i = arena.empty_intervention_set();
    let mut factors = Vec::new();
    let mut preceding = BitSet::with_len(v.bit_len());
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        if s.contains(vi) {
            let var_i = prepared.dense_to_var(vi)?;
            let vars = arena.intern_var_set([var_i]);
            let cond_vars: Result<Vec<_>, _> =
                preceding.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
            let cond_vars = cond_vars?;
            let conditioned_on = arena.intern_var_set(cond_vars.clone());
            let (intervention, domain) =
                intervention_for_factor(arena, prepared, &assign, vi, &cond_vars)?;
            factors.push(arena.intern_distribution(vars, conditioned_on, intervention, domain));
        }
        preceding.insert(vi);
    }
    if factors.is_empty() {
        let y = intern_nodes(prepared, s, arena)?;
        let empty = arena.empty_var_set();
        return Ok(arena.intern_distribution(y, empty, empty_i, DomainRef::Observational));
    }
    if factors.len() == 1 {
        return Ok(factors[0]);
    }
    let list = arena.intern_list(factors);
    Ok(arena.intern(ExprNode::Product(list)))
}

fn markov_product(
    prepared: &PreparedAdmg,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Assign,
) -> Result<ExprId, IdentificationError> {
    let empty_i = arena.empty_intervention_set();
    let mut factors = Vec::new();
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        let var_i = prepared.dense_to_var(vi)?;
        let vars = arena.intern_var_set([var_i]);
        let parents: Result<Vec<_>, _> = prepared
            .admg()
            .parents(vi)
            .iter()
            .copied()
            .filter(|p| v.contains(*p))
            .map(|p| prepared.dense_to_var(p))
            .collect();
        let parents = parents?;
        let conditioned_on = arena.intern_var_set(parents.clone());
        let (intervention, domain) =
            intervention_for_factor(arena, prepared, &assign, vi, &parents)?;
        factors.push(arena.intern_distribution(vars, conditioned_on, intervention, domain));
    }
    if factors.is_empty() {
        let empty = arena.empty_var_set();
        return Ok(arena.intern_distribution(empty, empty, empty_i, DomainRef::Observational));
    }
    if factors.len() == 1 {
        return Ok(factors[0]);
    }
    let list = arena.intern_list(factors);
    Ok(arena.intern(ExprNode::Product(list)))
}

/// Bake `do(T=t)` into the factor that generates `T`, or into factors that condition on `T`.
fn intervention_for_factor(
    arena: &mut CausalExprArena,
    prepared: &PreparedAdmg,
    assign: &Assign,
    vi: DenseNodeId,
    conditioned_on: &[VariableId],
) -> Result<(antecedent_expr::InterventionSetId, DomainRef), IdentificationError> {
    let mut touching = Vec::new();
    for (t, val) in assign.iter() {
        let t_var = prepared.dense_to_var(*t)?;
        if *t == vi || conditioned_on.iter().any(|&v| v == t_var) {
            touching.push(InterventionAssignment { variable: t_var, value: val.clone() });
        }
    }
    if touching.is_empty() {
        return Ok((arena.empty_intervention_set(), DomainRef::Observational));
    }
    let intervention = arena.intern_intervention_assignments(touching);
    Ok((intervention, DomainRef::Interventional))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, ContinuousDomain, GridSpec, Intervention,
        MechanismOverride, ResponseFunctional, ResponseQuery, TargetPopulation, Value, VariableId,
    };
    use antecedent_graph::{Admg, Dag, DenseNodeId};
    use std::sync::Arc;

    use super::*;
    use crate::error::IdentificationError;
    use crate::identifier::IdentificationWorkspace;
    use crate::oracle_dot::{admg_from_oracle_dot, dag_from_oracle_dot};
    use crate::result::IdentificationStatus;

    fn chain_dag() -> Dag {
        let mut dag = Dag::with_variables(3);
        // 0 -> 1 -> 2  (T -> M -> Y) but use T=0, Z=1, Y=2 with Z confounder style:
        // backdoor chain: Z -> T -> Y, Z -> Y  => nodes 0=Z, 1=T, 2=Y
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        dag
    }

    #[test]
    fn backdoor_chain_identified() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/general_id_backdoor_chain/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["case"], "identifiable_backdoor");
        assert_eq!(fixture["treatment"], "t");
        assert_eq!(fixture["outcome"], "y");
        assert_eq!(fixture["expected_status_family"], "identified");
        assert!(
            fixture["reference"]["outputs"]["estimand"]
                .as_str()
                .unwrap()
                .contains("Estimand name: backdoor"),
            "external oracle must identify the frozen Z -> T, Z -> Y, T -> Y graph by backdoor"
        );

        let (dag, nodes) = dag_from_oracle_dot(fixture["graph_dot"].as_str().unwrap());
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = AverageEffectQuery::binary_ate(
            nodes.id(fixture["treatment"].as_str().unwrap()),
            nodes.id(fixture["outcome"].as_str().unwrap()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].method_kind().unwrap(), EstimandMethod::GeneralId);
    }

    #[test]
    fn general_id_records_causal_markov() {
        // Truncated-factorization / g-formula identification is only valid
        // under the Causal Markov condition on the graph; every other
        // identifier in the crate (backdoor, IV, RD) records its structural
        // assumptions, and general ID must not be the exception.
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(
            res.required_assumptions
                .entries
                .iter()
                .any(|r| matches!(r.assumption, antecedent_core::Assumption::CausalMarkov)),
            "general ID must record the Causal Markov assumption it relies on"
        );
    }

    #[test]
    fn hedge_not_identified() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/id_hedge/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][1]["certificate"].as_str(), Some("hedge"));
        let external: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/general_id_hedge/expected.json"
        ))
        .unwrap();
        assert_eq!(external["case"], "nonidentifiable_or_unidentified");
        assert_eq!(external["treatment"], "t");
        assert_eq!(external["outcome"], "y");
        assert_eq!(external["expected_status_family"], "unidentified");
        assert!(
            external["reference"]["outputs"]["estimand"]
                .as_str()
                .unwrap()
                .contains("No such variable(s) found!"),
            "external oracle must refuse every standard identification strategy on the bow arc"
        );
        let (g, nodes) = admg_from_oracle_dot(external["graph_dot"].as_str().unwrap());
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = AverageEffectQuery::binary_ate(
            nodes.id(external["treatment"].as_str().unwrap()),
            nodes.id(external["outcome"].as_str().unwrap()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.diagnostics.iter().any(|d| d.code.as_ref() == "identify.hedge"));
    }

    #[test]
    fn frontdoor_admg_identified() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/general_id_frontdoor/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["expected_status_family"], "identified");
        let (g, nodes) = admg_from_oracle_dot(fixture["graph_dot"].as_str().unwrap());
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = AverageEffectQuery::binary_ate(
            nodes.id(fixture["treatment"].as_str().unwrap()),
            nodes.id(fixture["outcome"].as_str().unwrap()),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    }

    #[test]
    fn soft_constant_and_shift_ate_reduce_to_set() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let mut ws = IdentificationWorkspace::default();

        let soft = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            Arc::from([]),
            Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            Intervention::soft(VariableId::from_raw(1), MechanismOverride::constant(1.0)),
            TargetPopulation::AllObserved,
        ));
        let res = id.identify(&prep, &soft, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);

        let shift = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            Arc::from([]),
            Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            Intervention::shift(VariableId::from_raw(1), Value::f64(1.0)),
            TargetPopulation::AllObserved,
        ));
        let err = id.identify(&prep, &shift, &mut ws).unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Shift")),
            "{err}"
        );
    }

    #[test]
    fn admg_observational_line1_uses_c_component_factorization() {
        // A ↔ B, no directed edges: Markov product would be P(A)P(B); Tian Q is P(A)P(B|A).
        let mut g = Admg::with_variables(2);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                VariableId::from_raw(0),
                Arc::from([]),
            )
            .with_outcomes(Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)])),
        );
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let functional = res.estimands[0].functional;
        assert!(
            distribution_has_nonempty_cond(&res.arena, functional),
            "line 1 on a bidirected ADMG must use Q-component factors, not ∏ P(vi)"
        );
    }

    fn distribution_has_nonempty_cond(arena: &CausalExprArena, id: ExprId) -> bool {
        match arena.node(id) {
            ExprNode::Distribution { conditioned_on, .. } => {
                !arena.var_set(*conditioned_on).is_empty()
            }
            ExprNode::Product(list) => {
                arena.list(*list).iter().any(|&e| distribution_has_nonempty_cond(arena, e))
            }
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                distribution_has_nonempty_cond(arena, *expr)
            }
            ExprNode::Ratio { numerator, denominator } => {
                distribution_has_nonempty_cond(arena, *numerator)
                    || distribution_has_nonempty_cond(arena, *denominator)
            }
            _ => false,
        }
    }

    /// With ID complete, an `Err` is never a verdict on identifiability, so each remaining
    /// one carries its own kind: a malformed query, and a broken internal invariant.
    #[test]
    fn remaining_errors_are_typed_by_cause() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let y = VariableId::from_raw(2);
        let outcome_intervened =
            CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
                y,
                [Intervention::set(y, Value::f64(1.0))],
            ));
        let err = id
            .identify(&prep, &outcome_intervened, &mut IdentificationWorkspace::default())
            .unwrap_err();
        assert!(matches!(err, IdentificationError::InvalidQuery { .. }), "{err:?}");

        // A law that lacks a factor for a node of the requested C-component cannot arise
        // from the recursion; if it did, that is a defect, not an unsupported query.
        let nodes = prep.admg().node_count();
        let law = Law { sumset: BitSet::with_len(nodes), factors: Vec::new() };
        let mut s = BitSet::with_len(nodes);
        s.insert(DenseNodeId::from_raw(2));
        let err = law.conditionals(&prep, &s, &full_nodes(nodes)).unwrap_err();
        assert!(matches!(err, IdentificationError::InvariantViolated { .. }), "{err:?}");
    }

    /// Napkin `W -> Z -> X -> Y`, `W <-> X`, `W <-> Y`: line 7 → line 2 → line 6.
    /// The conditional of the marginalized C-factor must be emitted as
    /// `N / ∑_y N` with the denominator summing the *same* numerator node, the
    /// shape an exact evaluator recognizes as a conditional on a null event
    /// when positivity fails (rather than an unlocated 0/0).
    #[test]
    fn napkin_conditional_is_a_ratio_over_its_own_marginal() {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 2), (0, 3)] {
            g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
            VariableId::from_raw(3),
            [Intervention::set(VariableId::from_raw(2), Value::f64(1.0))],
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let rules: Vec<&str> = res.derivation.steps.iter().map(|s| s.rule.as_ref()).collect();
        assert_eq!(
            rules,
            [
                "general.id",
                "general.id.line3",
                "general.id.line7",
                "general.id.line2",
                "general.id.line6"
            ]
        );
        let ExprNode::Ratio { numerator, denominator } =
            res.arena.node(res.estimands[0].functional).clone()
        else {
            panic!("expected a ratio, got {}", res.arena.pretty(res.estimands[0].functional));
        };
        let ExprNode::SumOut { variables, expr } = res.arena.node(denominator).clone() else {
            panic!("denominator must be a sum");
        };
        assert_eq!(expr, numerator, "denominator must marginalize the numerator node itself");
        assert_eq!(res.arena.var_set(variables), [VariableId::from_raw(3)]);
        // Numerator: ∑_w P(w) P(x | w, z) P(y | w, z, x).
        let ExprNode::SumOut { variables, expr } = res.arena.node(numerator).clone() else {
            panic!("numerator must sum out w");
        };
        assert_eq!(res.arena.var_set(variables), [VariableId::from_raw(0)]);
        let ExprNode::Product(list) = res.arena.node(expr).clone() else {
            panic!("numerator body must be the carried C-factor");
        };
        assert_eq!(res.arena.list(list).len(), 3);
    }

    /// `Z -> T`, `Z -> Y`, `T -> Y` and a four-point grid: the mean curve is one identified
    /// intervention mean per grid level, not the first level standing for the whole curve.
    #[test]
    fn mean_curve_general_id_identifies_every_grid_level() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let (t, y) = (VariableId::from_raw(1), VariableId::from_raw(2));
        let grid = [0.0, 0.5, 1.0, 2.0];
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from(grid))),
        });
        let res = id
            .identify_response(&prep, &response, &mut IdentificationWorkspace::default())
            .unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(matches!(res.query, CausalQuery::Response(_)));
        assert_eq!(res.estimands.len(), grid.len());
        for estimand in &res.estimands {
            assert_eq!(estimand.method_kind().unwrap(), EstimandMethod::GeneralId);
            let ExprNode::Expectation { function, .. } = res.arena.node(estimand.functional) else {
                panic!("each grid level is the mean of Y");
            };
            assert_eq!(function.variable(), y);
        }
        // The level is part of the functional, so no two grid points share one.
        for (i, a) in res.estimands.iter().enumerate() {
            for b in &res.estimands[i + 1..] {
                assert_ne!(a.functional, b.functional);
            }
        }
    }

    /// `P(A | do(A = 1))` is the point mass at 1, not the observational `P(A)`: an outcome that
    /// is also an intervention target is an invalid query for ID, whole or partial overlap.
    #[test]
    fn outcome_that_is_an_intervention_target_is_refused() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let (t, y) = (VariableId::from_raw(1), VariableId::from_raw(2));
        let mut ws = IdentificationWorkspace::default();
        let whole =
            CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
                t,
                [Intervention::set(t, Value::f64(1.0))],
            ));
        let partial = CausalQuery::Distribution(
            antecedent_core::InterventionalDistributionQuery::new(
                y,
                [Intervention::set(t, Value::f64(1.0))],
            )
            .with_outcomes([y, t]),
        );
        for query in [whole, partial] {
            let err = id.identify(&prep, &query, &mut ws).unwrap_err();
            assert!(matches!(err, IdentificationError::InvalidQuery { .. }), "{err}");
        }
        // Disjoint outcome and intervention stay valid.
        let fine =
            CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
                y,
                [Intervention::set(t, Value::f64(1.0))],
            ));
        assert!(id.identify(&prep, &fine, &mut ws).is_ok());
    }

    #[test]
    fn soft_linear_gaussian_still_unsupported() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let mut ws = IdentificationWorkspace::default();
        let soft = CausalQuery::AverageEffect(AverageEffectQuery::new(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            Arc::from([]),
            Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            Intervention::soft(
                VariableId::from_raw(1),
                MechanismOverride::named("linear_gaussian", vec![1.0, 0.0]),
            ),
            TargetPopulation::AllObserved,
        ));
        let err = id.identify(&prep, &soft, &mut ws).unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Soft")),
            "{err}"
        );
    }
}
