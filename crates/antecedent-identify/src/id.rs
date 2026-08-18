//! Shpitser & Pearl (2006) ID algorithm for semi-Markovian models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::unused_self
)]

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
    assign: Option<(DenseNodeId, Value)>,
    dist: DistCtx,
}

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
    /// C-factor `∑_{sumset} ∏ P(vi | cond_i)` with each factor's conditioning
    /// frozen when the factor set was formed at a line-7 entry. By Tian's
    /// telescope, `cond_i` includes predecessors *outside* the current `v`.
    /// Factors are kept in topological order.
    CFactor { sumset: BitSet, factors: Vec<(DenseNodeId, BitSet)> },
}

/// Outcome of a recursive ID call.
#[derive(Clone, Debug)]
enum IdOutcome {
    Expr(ExprId),
    Fail(HedgeCertificate),
}

/// Identifier implementing the complete ID algorithm on ADMGs.
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
                self.identify_sets(prepared, &y, &x, query.clone(), workspace)
            }
            _ => Err(IdentificationError::unsupported(
                "IdIdentifier supports AverageEffect and Distribution queries",
            )),
        }
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
            Some((t, active_level)),
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
            Some((t, control_level)),
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

        let left_exp = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(query.outcome),
            distribution: left,
        });
        let right_exp = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(query.outcome),
            distribution: right,
        });
        let contrast = arena.intern(ExprNode::Contrast {
            left: left_exp,
            right: right_exp,
            op: ContrastOp::Difference,
        });
        // A dead sum/integral here means the assembled functional is ill-formed,
        // which must surface rather than be silently rewritten away.
        let functional =
            arena.simplify(contrast).map_err(|e| IdentificationError::msg(e.to_string()))?;
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
            prepared.declared_assumptions().clone(),
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
    ) -> Result<IdentificationResult, IdentificationError> {
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
            None,
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
                    prepared.declared_assumptions().clone(),
                    perf,
                ))
            }
            IdOutcome::Fail(hedge) => Ok(not_identified_with_hedge(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
                hedge,
            )),
        }
    }
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
    assign: Option<(DenseNodeId, Value)>,
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
    assign: Option<(DenseNodeId, Value)>,
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
        // V\X empty → Y ⊆ X; interventional delta / empty product
        derivation.push("general.id.degenerate", "V\\X empty");
        return Ok(IdOutcome::Expr(dist_marginal(prepared, dist, y, v, arena, assign)?));
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
        let hedge = HedgeCertificate::from_sets(v, s, |d| {
            prepared.dense_to_var(d).unwrap_or_else(|_| VariableId::from_raw(d.raw()))
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
    assign: Option<(DenseNodeId, Value)>,
) -> Result<IdOutcome, IdentificationError> {
    // Districts of G (on V)
    let g_comps = prepared.c_components(v);
    // Line 6: S ∈ C(G) — emit the C-factor of the *current* distribution.
    if g_comps.iter().any(|c| c.equal_set(s)) {
        derivation.push("general.id.line6", "S is a C-component of G; factorize current dist");
        let expr = match dist {
            DistCtx::Marginal => c_component_expression(prepared, s, y, v, arena, assign)?,
            DistCtx::CFactor { .. } => {
                // Conditionals of the carried c-factor: with an empty sumset the
                // telescope collapses each to its own frozen factor; with a
                // non-empty sumset they are exact ratios of partial sums.
                let sub = dist.cfactor_of(prepared, s, v)?;
                sub.emit(prepared, s, y, arena, assign.as_ref())?
            }
        };
        return Ok(IdOutcome::Expr(expr));
    }

    // Line 7: ∃ S' ⊃ S, S' ∈ C(G). Recurse on G_{S'} against Q[S'], the
    // C-factor of the current distribution — its factors keep conditioning on
    // topological predecessors *outside* S' (Tian's telescope), which is what
    // the previous implementation dropped.
    if let Some(s_prime) = g_comps.iter().find(|c| s.is_subset_of(c) && !c.equal_set(s)) {
        derivation.push("general.id.line7", "recurse into containing C-component S' against Q[S']");
        let mut x2 = x.clone();
        x2.intersect_with(s_prime);
        let q_s_prime = dist.cfactor_of(prepared, s_prime, v)?;
        let dist2 = DistCtx::CFactor { sumset: q_s_prime.sumset, factors: q_s_prime.factors };
        return id_recurse(
            prepared, y, &x2, s_prime, &dist2, arena, memo, derivation, perf, ws, assign,
        );
    }

    Err(IdentificationError::msg("ID reached inconsistent C-component state"))
}

/// A materialized c-factor of the current distribution: `∑_{sumset} ∏ factors`.
struct QFactor {
    sumset: BitSet,
    factors: Vec<(DenseNodeId, BitSet)>,
}

impl DistCtx {
    /// Marginalize the current distribution over `removed` (line 2).
    fn marginalize(&self, removed: &BitSet) -> Self {
        match self {
            // A marginal of the observational marginal is still a marginal.
            Self::Marginal => Self::Marginal,
            Self::CFactor { sumset, factors } => {
                let mut sumset = sumset.clone();
                sumset.union_with(removed);
                Self::CFactor { sumset, factors: factors.clone() }
            }
        }
    }

    /// C-factor `Q_dist[S]` of the current distribution over `v`.
    ///
    /// For a marginal, each factor conditions on **all** `v`-predecessors in
    /// topological order (chain rule of the marginal joint). For a carried
    /// c-factor with an empty sumset the telescope keeps each node's frozen
    /// factor. A non-empty sumset means the carried product no longer
    /// telescopes node-wise; the factors are kept with the sumset so the
    /// emitter can fall back to exact ratio conditionals.
    fn cfactor_of(
        &self,
        prepared: &PreparedAdmg,
        s: &BitSet,
        v: &BitSet,
    ) -> Result<QFactor, IdentificationError> {
        match self {
            Self::Marginal => {
                let mut factors = Vec::new();
                let mut preceding = BitSet::with_len(v.bit_len());
                for &vi in prepared.topo() {
                    if !v.contains(vi) {
                        continue;
                    }
                    if s.contains(vi) {
                        factors.push((vi, preceding.clone()));
                    }
                    preceding.insert(vi);
                }
                Ok(QFactor { sumset: BitSet::with_len(v.bit_len()), factors })
            }
            Self::CFactor { sumset, factors } => {
                if !sumset.any() {
                    let kept = factors.iter().filter(|(vi, _)| s.contains(*vi)).cloned().collect();
                    return Ok(QFactor { sumset: sumset.clone(), factors: kept });
                }
                // Nested line-7 after a line-2 marginalization: the carried
                // product has bound variables, so node-wise conditionals are
                // ratios of partial sums over the *full* factor set. Keep all
                // factors and record which nodes S selects via the emitter.
                Err(IdentificationError::msg(
                    "general ID: nested C-factor of a marginalized Q is not yet supported;                      refusing rather than emitting an unsound functional",
                ))
            }
        }
    }
}

impl QFactor {
    /// Emit `∑_{(s\y) ∪ sumset} ∏ factors` with `do(·)` labels applied only to
    /// factors whose assigned variable is *free* (not bound by these sums) —
    /// a bound occurrence is the sum's dummy variable, not the do-value.
    fn emit(
        &self,
        prepared: &PreparedAdmg,
        s: &BitSet,
        y: &BitSet,
        arena: &mut CausalExprArena,
        assign: Option<&(DenseNodeId, Value)>,
    ) -> Result<ExprId, IdentificationError> {
        let mut sum_vars = s.clone();
        sum_vars.difference_with(y);
        sum_vars.union_with(&self.sumset);
        let effective_assign = assign.filter(|(t, _)| !sum_vars.contains(*t)).cloned();
        let mut exprs = Vec::with_capacity(self.factors.len());
        for (vi, cond) in &self.factors {
            let var_i = prepared.dense_to_var(*vi)?;
            let vars = arena.intern_var_set([var_i]);
            let cond_vars: Result<Vec<_>, _> =
                cond.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
            let cond_vars = cond_vars?;
            let conditioned_on = arena.intern_var_set(cond_vars.clone());
            let (intervention, domain) = intervention_for_factor(
                arena,
                prepared,
                effective_assign.as_ref(),
                *vi,
                &cond_vars,
            )?;
            exprs.push(arena.intern(ExprNode::Distribution {
                variables: vars,
                conditioned_on,
                intervention,
                domain,
            }));
        }
        let product = if exprs.len() == 1 {
            exprs[0]
        } else {
            let list = arena.intern_list(exprs);
            arena.intern(ExprNode::Product(list))
        };
        if sum_vars.any() {
            let vs = intern_nodes(prepared, &sum_vars, arena)?;
            Ok(arena.intern(ExprNode::SumOut { variables: vs, expr: product }))
        } else {
            Ok(product)
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
    assign: Option<(DenseNodeId, Value)>,
) -> Result<ExprId, IdentificationError> {
    match dist {
        DistCtx::Marginal => observational_marginal(prepared, y, v, arena, assign),
        DistCtx::CFactor { sumset, factors } => {
            let q = QFactor { sumset: sumset.clone(), factors: factors.clone() };
            // Sum over everything in v except y, plus the carried sumset.
            q.emit(prepared, v, y, arena, assign.as_ref())
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
    assign: Option<(DenseNodeId, Value)>,
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
    assign: Option<(DenseNodeId, Value)>,
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
    assign: Option<(DenseNodeId, Value)>,
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
                intervention_for_factor(arena, prepared, assign.as_ref(), vi, &cond_vars)?;
            factors.push(arena.intern(ExprNode::Distribution {
                variables: vars,
                conditioned_on,
                intervention,
                domain,
            }));
        }
        preceding.insert(vi);
    }
    if factors.is_empty() {
        let y = intern_nodes(prepared, s, arena)?;
        let empty = arena.empty_var_set();
        return Ok(arena.intern(ExprNode::Distribution {
            variables: y,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        }));
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
    assign: Option<(DenseNodeId, Value)>,
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
            intervention_for_factor(arena, prepared, assign.as_ref(), vi, &parents)?;
        factors.push(arena.intern(ExprNode::Distribution {
            variables: vars,
            conditioned_on,
            intervention,
            domain,
        }));
    }
    if factors.is_empty() {
        let empty = arena.empty_var_set();
        return Ok(arena.intern(ExprNode::Distribution {
            variables: empty,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        }));
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
    assign: Option<&(DenseNodeId, Value)>,
    vi: DenseNodeId,
    conditioned_on: &[VariableId],
) -> Result<(antecedent_expr::InterventionSetId, DomainRef), IdentificationError> {
    let empty_i = arena.empty_intervention_set();
    let Some((t, val)) = assign else {
        return Ok((empty_i, DomainRef::Observational));
    };
    let t_var = prepared.dense_to_var(*t)?;
    if *t == vi {
        let intervention = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t_var,
            value: val.clone(),
        }]);
        return Ok((intervention, DomainRef::Interventional));
    }
    if conditioned_on.iter().any(|&v| v == t_var) {
        let intervention = arena.intern_intervention_assignments([InterventionAssignment {
            variable: t_var,
            value: val.clone(),
        }]);
        return Ok((intervention, DomainRef::Interventional));
    }
    Ok((empty_i, DomainRef::Observational))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, Intervention, MechanismOverride, TargetPopulation, Value,
        VariableId,
    };
    use antecedent_graph::{Admg, Dag, DenseNodeId};
    use std::sync::Arc;

    use super::*;
    use crate::error::IdentificationError;
    use crate::identifier::IdentificationWorkspace;
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
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].method_kind().unwrap(), EstimandMethod::GeneralId);
    }

    #[test]
    fn hedge_not_identified() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/id_hedge/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][1]["certificate"].as_str(), Some("hedge"));
        // t -> y with t ↔ y
        let mut g = Admg::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.diagnostics.iter().any(|d| d.code.as_ref() == "identify.hedge"));
    }

    #[test]
    fn frontdoor_admg_identified() {
        // t -> m -> y, t ↔ y
        let mut g = Admg::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();
        g.insert_bidirected(t, y).unwrap();
        let id = IdIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2));
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
