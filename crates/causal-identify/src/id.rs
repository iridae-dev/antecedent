//! Shpitser–Pearl ID algorithm for semi-Markovian models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    Intervention, Value, VariableId,
};
use causal_expr::{
    CausalExprArena, ContrastOp, DomainRef, EstimandMethod, ExprId, ExprNode, IdentifiedEstimand,
    InterventionAssignment, OutcomeExprId,
};
use causal_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;
use crate::hedge::HedgeCertificate;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
};

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
                if let Err(e) = crate::intervention_support::require_hard_set_interventions(
                    q.interventions.iter(),
                    "general ID",
                ) {
                    return Err(e);
                }
                // Flatten Sequence-of-Sets / Soft(constant) reductions for multi-do.
                let normalized = crate::intervention_support::normalize_intervention_list(
                    q.interventions.iter().cloned(),
                )?;
                let mut x = BitSet::with_len(prepared.admg().node_count());
                for intervention in &normalized {
                    let v = intervention.primary_variable().ok_or(IdentificationError::unsupported(
                        "intervention missing primary variable",
                    ))?;
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
        query.validate().map_err(|_| IdentificationError::unsupported("invalid average-effect query"))?;
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
        let functional = arena.simplify(contrast);
        let estimand = IdentifiedEstimand {
            method: Arc::from(EstimandMethod::GeneralId.as_str()),
            adjustment_set: Arc::from([]),
            instruments: Arc::from([]),
            mediators: Arc::from([]),
            functional,
            rd_design: None,
        };
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
            &mut arena,
            &mut memo,
            &mut derivation,
            &mut perf,
            &mut workspace.graph,
            None,
        )? {
            IdOutcome::Expr(functional) => {
                let estimand = IdentifiedEstimand {
                    method: Arc::from(EstimandMethod::GeneralId.as_str()),
                    adjustment_set: Arc::from([]),
                    instruments: Arc::from([]),
                    mediators: Arc::from([]),
                    functional,
                    rd_design: None,
                };
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
                Arc::from(hedge.f.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",")),
            ),
            (
                Arc::from("f_prime"),
                Arc::from(
                    hedge
                        .f_prime
                        .iter()
                        .map(|v| v.raw().to_string())
                        .collect::<Vec<_>>()
                        .join(","),
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
    };
    if let Some(hit) = memo.get(&key) {
        perf.sets_returned = perf.sets_returned.saturating_add(1);
        return Ok(hit.clone());
    }

    let outcome = id_body(prepared, y, x, v, arena, memo, derivation, perf, ws, assign)?;
    memo.insert(key, outcome.clone());
    Ok(outcome)
}

fn id_body(
    prepared: &mut PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    memo: &mut HashMap<SubproblemKey, IdOutcome>,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
    assign: Option<(DenseNodeId, Value)>,
) -> Result<IdOutcome, IdentificationError> {
    // Line 1: x = ∅ → ∑_{v\y} P(v)
    if !x.any() {
        derivation.push("general.id.line1", "empty intervention; observational marginal");
        return Ok(IdOutcome::Expr(observational_marginal(prepared, y, v, arena, assign)?));
    }

    // Line 2: restrict to An(Y)_G
    let an_y = prepared.ancestors_within(y, v, ws);
    if !v.equal_set(&an_y) {
        let mut x2 = x.clone();
        x2.intersect_with(&an_y);
        derivation.push("general.id.line2", "restrict to ancestral set of Y");
        return id_recurse(prepared, y, &x2, &an_y, arena, memo, derivation, perf, ws, assign);
    }

    // Line 3: W = (V\X) \ An(Y)_{G_{\bar X}}
    let mut v_minus_x = v.clone();
    v_minus_x.difference_with(x);
    let an_bar = prepared.ancestors_bar_x(y, v, x, ws);
    let mut w = v_minus_x.clone();
    w.difference_with(&an_bar);
    if w.any() {
        let mut x2 = x.clone();
        x2.union_with(&w);
        derivation.push("general.id.line3", "add superfluous interventions");
        return id_recurse(prepared, y, &x2, v, arena, memo, derivation, perf, ws, assign);
    }

    // Line 4 / 5–7: C-components of G[V\X]
    let comps = prepared.c_components(&v_minus_x);
    if comps.is_empty() {
        // V\X empty → Y ⊆ X; interventional delta / empty product
        derivation.push("general.id.degenerate", "V\\X empty");
        return Ok(IdOutcome::Expr(observational_marginal(prepared, y, v, arena, assign)?));
    }

    if comps.len() > 1 {
        derivation.push("general.id.line4", format!("C-component factorization ({} parts)", comps.len()));
        let mut factors = Vec::with_capacity(comps.len());
        for s_i in &comps {
            let mut x_i = v.clone();
            x_i.difference_with(s_i);
            match id_recurse(
                prepared,
                s_i,
                &x_i,
                v,
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

    // Districts of G (on V)
    let g_comps = prepared.c_components(v);
    // Line 6: S ∈ C(G)
    if g_comps.iter().any(|c| c.equal_set(s)) {
        derivation.push("general.id.line6", "S is a C-component of G; observational factorization");
        return Ok(IdOutcome::Expr(c_component_expression(prepared, s, y, v, arena, assign)?));
    }

    // Line 7: ∃ S' ⊃ S, S' ∈ C(G)
    if let Some(s_prime) = g_comps.iter().find(|c| s.is_subset_of(c) && !c.equal_set(s)) {
        derivation.push("general.id.line7", "recurse into containing C-component S'");
        let mut x2 = x.clone();
        x2.intersect_with(s_prime);
        // Distribution Q[S'] = ∏_{Vi∈S'} P(Vi | …) — encode as the observational
        // C-component factor and continue ID on G_{S'}.
        return id_recurse(prepared, y, &x2, s_prime, arena, memo, derivation, perf, ws, assign);
    }

    Err(IdentificationError::msg("ID reached inconsistent C-component state"))
}

fn intern_nodes(
    prepared: &PreparedAdmg,
    nodes: &BitSet,
    arena: &mut CausalExprArena,
) -> Result<causal_expr::VarSetId, IdentificationError> {
    let vars: Result<Vec<_>, _> = nodes.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
    Ok(arena.intern_var_set(vars?))
}

fn observational_marginal(
    prepared: &PreparedAdmg,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: Option<(DenseNodeId, Value)>,
) -> Result<ExprId, IdentificationError> {
    // ∏_{vi ∈ V} P(vi | pa) then sum out V\Y — Markov factorization on directed edges.
    // For ADMGs without latents this is exact; with latents, line 1 only runs when X=∅
    // after ancestral restriction so districts are handled elsewhere.
    let factors = markov_product(prepared, v, arena, assign)?;
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
) -> Result<(causal_expr::InterventionSetId, DomainRef), IdentificationError> {
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
    use causal_core::{
        AverageEffectQuery, CausalQuery, Intervention, MechanismOverride, TargetPopulation, Value,
        VariableId,
    };
    use causal_graph::{Admg, Dag, DenseNodeId};
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

        let soft = CausalQuery::AverageEffect(AverageEffectQuery {
            treatment: VariableId::from_raw(1),
            outcome: VariableId::from_raw(2),
            effect_modifiers: Arc::from([]),
            control: Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            active: Intervention::soft(VariableId::from_raw(1), MechanismOverride::constant(1.0)),
            target_population: TargetPopulation::AllObserved,
        });
        let res = id.identify(&prep, &soft, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);

        let shift = CausalQuery::AverageEffect(AverageEffectQuery {
            treatment: VariableId::from_raw(1),
            outcome: VariableId::from_raw(2),
            effect_modifiers: Arc::from([]),
            control: Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            active: Intervention::shift(VariableId::from_raw(1), Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
        });
        let res = id.identify(&prep, &shift, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    }

    #[test]
    fn soft_linear_gaussian_still_unsupported() {
        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&chain_dag()).unwrap();
        let mut ws = IdentificationWorkspace::default();
        let soft = CausalQuery::AverageEffect(AverageEffectQuery {
            treatment: VariableId::from_raw(1),
            outcome: VariableId::from_raw(2),
            effect_modifiers: Arc::from([]),
            control: Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
            active: Intervention::soft(
                VariableId::from_raw(1),
                MechanismOverride::named("linear_gaussian", vec![1.0, 0.0]),
            ),
            target_population: TargetPopulation::AllObserved,
        });
        let err = id.identify(&prep, &soft, &mut ws).unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Soft")),
            "{err}"
        );
    }
}
