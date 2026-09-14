//! Path-restricted natural-effect identification (Avin, Shpitser & Pearl 2005).
//!
//! Enumerate directed paths π from treatment to outcome and refuse when a
//! recanting witness exists. When every directed path is in π the
//! path-specific effect is the total effect: surgically delete treatment
//! out-edges not on any path and run general ID for the active/control
//! contrast. When a complementary path exists, emit the edge g-formula
//! (Shpitser 2013): the treatment enters each child's factor at the active
//! level if that edge starts a path in π and at the control level otherwise,
//! minus the all-control g-formula.
//!
//! # Recanting witness
//!
//! Avin, Shpitser & Pearl (2005) define a recanting witness for π as a node
//! `W ≠ T` with a directed path `q` from `T` to `W` and two directed paths
//! `s₁`, `s₂` from `W` to `Y` such that `q·s₁ ∈ π` and `q·s₂ ∉ π`: `W`'s
//! response to the shared prefix `q` would be needed at the active level along
//! one continuation and at the control level along the other, a cross-world
//! quantity no observational law pins down. The effect is identified in a DAG
//! iff no witness exists. Taking `W` to be the first node of `q` after `T`
//! shows the witness exists iff some child `C` of `T` starts both a path in π
//! and a path outside it (conversely, a child that starts both is itself a
//! witness with `q = T → C`). The check is therefore exact: it compares the
//! first hops of the selected and the complementary paths. Without a witness,
//! π is exactly the set of paths leaving `T` through an edge that starts a
//! selected path, which is the edge assignment the edge g-formula binds.
//!
//! A node reached from `T` along both kinds of path but through *different*
//! children (`T → A → W → Y` selected, `T → B → W → Y` not) is not a witness:
//! `W` takes `A` from the active world and `B` from the control world through
//! one ordinary factor `P(w | a, b)`.
//!
//! On an ADMG a directed witness still refuses (latent variables cannot remove
//! one). The district-level criterion (Shpitser 2013's recanting district) is
//! not implemented: without a witness, a bidirected edge on an outcome ancestor
//! is refused as unsupported by the edge g-formula.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, PathSpecificEffectQuery, Value, VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DerivationMeta, DomainRef, EstimandMethod, ExprId, ExprNode,
    IdentifiedEstimand, InterventionAssignment, OutcomeExprId,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};

use crate::error::IdentificationError;
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Path-restricted natural-effect identifier.
#[derive(Clone, Debug, Default)]
pub struct PathSpecificIdentifier {
    /// Underlying general ID engine.
    pub inner: IdIdentifier,
}

impl PathSpecificIdentifier {
    /// Create the identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a DAG as an ADMG.
    ///
    /// # Errors
    ///
    /// Graph construction failure.
    pub fn prepare_dag(&self, graph: &Dag) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare_dag(graph)
    }

    /// Prepare an ADMG.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare(&self, graph: &Admg) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare(graph)
    }

    /// Identify a path-specific natural effect.
    ///
    /// # Errors
    ///
    /// Unsupported query shape, unknown variables, or path enumeration failure.
    pub fn identify(
        &self,
        prepared: &PreparedAdmg,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let mediation_path;
        let q = match query {
            CausalQuery::PathSpecific(q) => q,
            CausalQuery::Mediation(m) => {
                m.validate()
                    .map_err(|_| IdentificationError::unsupported("invalid mediation query"))?;
                mediation_path = PathSpecificEffectQuery {
                    treatment: m.treatment,
                    outcome: m.outcome,
                    path_nodes: Arc::clone(&m.mediators),
                    control: m.control.clone(),
                    active: m.active.clone(),
                    target_population: m.target_population.clone(),
                    ..PathSpecificEffectQuery::binary(m.treatment, m.outcome)
                };
                &mediation_path
            }
            _ => {
                return Err(IdentificationError::unsupported(
                    "PathSpecificIdentifier requires path-specific or static mediation query",
                ));
            }
        };
        q.validate()
            .map_err(|_| IdentificationError::unsupported("invalid path-specific query"))?;
        self.identify_path_specific(prepared, q, query.clone(), workspace)
    }

    #[allow(clippy::too_many_lines)]
    fn identify_path_specific(
        &self,
        prepared: &PreparedAdmg,
        q: &PathSpecificEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let mut derivation = DerivationTrace::default();
        derivation.push("path_specific", "Avin–Shpitser–Pearl path-restricted natural effect");
        let mut perf = IdentificationPerformanceRecord::default();

        let t = prepared.var_to_dense(q.treatment)?;
        let y = prepared.var_to_dense(q.outcome)?;
        let path_filter: HashSet<DenseNodeId> =
            q.path_nodes.iter().map(|&v| prepared.var_to_dense(v)).collect::<Result<_, _>>()?;

        let admg = prepared.admg();
        // Enumerate on the directed skeleton (DAG view of directed edges).
        let dag = admg_to_dag(admg)?;
        let (raw_paths, paths_truncated) = dag
            .directed_paths_with_budget(t, y, q.max_paths, q.max_len)
            .map_err(IdentificationError::from)?;
        perf.candidates_examined = perf.candidates_examined.saturating_add(raw_paths.len() as u64);

        // The recanting-witness test below concludes "identifiable" from the *absence* of a
        // witness. That inference is only valid over the complete path set: a witness sitting
        // on a path the budget dropped would be missed, and the effect declared
        // nonparametrically identified when Avin, Shpitser & Pearl (2005) say it is not. Fail closed.
        if paths_truncated {
            derivation.push(
                "path_specific.truncated",
                format!(
                    "path enumeration hit its budget (max_paths={}, max_len={}); the recanting-witness \
                     check cannot certify absence over an incomplete path set",
                    q.max_paths, q.max_len
                ),
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }

        // π and its complement come from one enumeration: re-running it would only repeat the
        // same budgeted search, and the two sets must partition the *same* path set for the
        // recanting check below to mean anything.
        let (pi, complement_paths): (Vec<Vec<DenseNodeId>>, Vec<Vec<DenseNodeId>>) =
            raw_paths.into_iter().partition(|path| {
                if let CausalQuery::Mediation(m) = &query {
                    use antecedent_core::MediationContrast;
                    let mediated = path
                        .iter()
                        .skip(1)
                        .take(path.len().saturating_sub(2))
                        .any(|node| path_filter.contains(node));
                    match m.contrast {
                        MediationContrast::Total => true,
                        MediationContrast::Direct | MediationContrast::NaturalDirect => !mediated,
                        MediationContrast::Mediated | MediationContrast::NaturalIndirect => {
                            mediated
                        }
                    }
                } else {
                    path_matches_filter(path, &path_filter)
                }
            });
        if pi.is_empty() {
            derivation.push("path_specific.empty", "no directed paths match path_nodes filter");
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
        derivation
            .push("path_specific.paths", format!("{} path(s) retained after filter", pi.len()));

        if let Some(w) = recanting_witness(&pi, &complement_paths) {
            derivation.push(
                "path_specific.recanting",
                format!(
                    "recanting witness dense={} (a treatment child starting both a selected and a \
                     complementary path) blocks nonparametric path-specific ID",
                    w.raw()
                ),
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
        derivation.push("path_specific.recanting", "no recanting witness");

        // Edges on π that leave the treatment.
        let mut keep_out: HashSet<DenseNodeId> = HashSet::new();
        for path in &pi {
            if path.len() >= 2 && path[0] == t {
                keep_out.insert(path[1]);
            }
        }

        // A complementary path means the contrast is not the total effect: the
        // treatment must reach the outcome at the active level along π and at the
        // control level along every other path. General ID on the surgical graph
        // cannot express that (its factors condition on the treatment as a
        // predecessor, so one level would reach every factor), so use the edge
        // g-formula instead.
        if !complement_paths.is_empty() {
            let (arena, functional) = edge_g_formula(prepared, q, t, y, &keep_out)?;
            derivation.push(
                "path_specific.edge_gformula",
                format!(
                    "{} complementary path(s): edge g-formula with the active level on {} \
                     treatment out-edge(s) on π and the control level on the rest",
                    complement_paths.len(),
                    keep_out.len()
                ),
            );
            let estimand = IdentifiedEstimand::new(
                Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
                Arc::from([]),
                Arc::from([]),
                Arc::clone(&q.path_nodes),
                functional,
                None,
            );
            return Ok(IdentificationResult::identified(
                query,
                vec![estimand],
                arena,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }

        let surgical = surgical_admg(admg, t, &keep_out)?;
        let surgical_prep =
            PreparedAdmg::with_assumptions(surgical, prepared.declared_assumptions().clone())?;
        derivation.push(
            "path_specific.surgery",
            format!(
                "kept {} treatment out-edge(s) on π; deleted complementary out-edges",
                keep_out.len()
            ),
        );

        let ate = AverageEffectQuery::new(
            q.treatment,
            q.outcome,
            Arc::from([]),
            q.control.clone(),
            q.active.clone(),
            q.target_population.clone(),
        );
        let mut id_res = self.inner.identify_ate(&surgical_prep, &ate, workspace)?;
        id_res.derivation.steps.splice(0..0, derivation.steps);
        id_res.performance.candidates_examined =
            id_res.performance.candidates_examined.saturating_add(perf.candidates_examined);
        id_res.query = query;
        if id_res.status == IdentificationStatus::NonparametricallyIdentified {
            for est in &mut id_res.estimands {
                est.method = Arc::from(EstimandMethod::PathSpecificNatural.as_str());
                // Surface path intermediates as mediators metadata.
                est.mediators = Arc::clone(&q.path_nodes);
            }
        }
        Ok(id_res)
    }
}

fn path_matches_filter(path: &[DenseNodeId], filter: &HashSet<DenseNodeId>) -> bool {
    if filter.is_empty() {
        return true;
    }
    // Intermediates only (exclude endpoints).
    let mid: HashSet<DenseNodeId> =
        path.iter().copied().skip(1).take(path.len().saturating_sub(2)).collect();
    filter.iter().all(|n| mid.contains(n))
}

/// A recanting witness for π, if one exists: a child of the treatment that
/// starts both a selected and a complementary directed path to the outcome.
///
/// Every path begins at the treatment, so `path[1]` is the treatment child the
/// path leaves through. By the argument in the module docs this is exactly the
/// Avin–Shpitser–Pearl criterion over the complete path set; callers must fail
/// closed on a truncated enumeration before relying on a `None`.
fn recanting_witness(
    pi: &[Vec<DenseNodeId>],
    complement: &[Vec<DenseNodeId>],
) -> Option<DenseNodeId> {
    let selected_first_hops: HashSet<DenseNodeId> =
        pi.iter().filter_map(|path| path.get(1).copied()).collect();
    complement
        .iter()
        .filter_map(|path| path.get(1).copied())
        .filter(|child| selected_first_hops.contains(child))
        .min_by_key(|child| child.raw())
}

/// Edge g-formula for a path-specific natural effect (Avin, Shpitser & Pearl
/// 2005; Shpitser 2013) on a graph without latent confounding among the
/// outcome's ancestors.
///
/// With `V* = An(Y)` in the graph with the treatment's incoming edges removed,
/// minus the treatment itself, each side of the contrast is
///
/// `E[Y] = Σ_{v* \ y} Π_{V ∈ V*} P(v | pa(v))`,
///
/// where a factor whose parents include the treatment `T` binds `T` to the
/// active level if `T → V` starts a path in π (`keep_out`) and to the control
/// level otherwise. The reference side binds every treatment edge to the
/// control level (the ordinary g-formula for `do(T = control)`). The per-factor
/// binding is carried as that factor's intervention set, so any evaluator of the
/// expression IR (the discrete plug-in and its shared Dirichlet row law
/// included) evaluates exactly this functional.
///
/// Callers must have ruled out a recanting witness: then every treatment child
/// that is an ancestor of `Y` lies on π-paths only or on complementary paths
/// only, so the edge assignment is well defined and equals the path-specific
/// effect.
fn edge_g_formula(
    prepared: &PreparedAdmg,
    q: &PathSpecificEffectQuery,
    t: DenseNodeId,
    y: DenseNodeId,
    keep_out: &HashSet<DenseNodeId>,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let admg = prepared.admg();
    // Ancestors of Y (Y included) reached without passing through T.
    let mut v_star: HashSet<DenseNodeId> = HashSet::new();
    let mut stack = vec![y];
    while let Some(node) = stack.pop() {
        if node == t || !v_star.insert(node) {
            continue;
        }
        stack.extend(admg.parents(node).iter().copied());
    }
    // `P(v | pa(v))` is the district factor only when `v` is a singleton
    // district; latent confounding would need Shpitser's district formula.
    if v_star.iter().any(|&v| !admg.bidirected_neighbors(v).is_empty()) {
        return Err(IdentificationError::unsupported(
            "path-specific edge g-formula requires no bidirected edge on an ancestor of the \
             outcome other than the treatment (the district-level formula for latent confounding \
             is not implemented)",
        ));
    }

    let t_var = prepared.dense_to_var(t)?;
    let active = crate::intervention_support::require_set_value(&q.active, "path-specific")?;
    let control = crate::intervention_support::require_set_value(&q.control, "path-specific")?;

    let mut arena = CausalExprArena::new();
    let empty_i = arena.empty_intervention_set();
    let order: Vec<DenseNodeId> =
        prepared.topo().iter().copied().filter(|v| v_star.contains(v)).collect();
    let side = |edge_level: &dyn Fn(DenseNodeId) -> Value,
                arena: &mut CausalExprArena|
     -> Result<ExprId, IdentificationError> {
        let mut factors = Vec::with_capacity(order.len());
        for &vi in &order {
            let parents = admg.parents(vi);
            let cond: Vec<VariableId> =
                parents.iter().map(|&p| prepared.dense_to_var(p)).collect::<Result<_, _>>()?;
            let (intervention, domain) = if parents.contains(&t) {
                let assignment = InterventionAssignment { variable: t_var, value: edge_level(vi) };
                (arena.intern_intervention_assignments([assignment]), DomainRef::Interventional)
            } else {
                (empty_i, DomainRef::Observational)
            };
            let variables = arena.intern_var_set([prepared.dense_to_var(vi)?]);
            let conditioned_on = arena.intern_var_set(cond);
            factors.push(arena.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
            }));
        }
        let body = if factors.len() == 1 {
            factors[0]
        } else {
            let list = arena.intern_list(factors);
            arena.intern(ExprNode::Product(list))
        };
        let summed: Vec<VariableId> = order
            .iter()
            .filter(|&&v| v != y)
            .map(|&v| prepared.dense_to_var(v))
            .collect::<Result<_, _>>()?;
        let distribution = if summed.is_empty() {
            body
        } else {
            let variables = arena.intern_var_set(summed);
            arena.intern(ExprNode::SumOut { variables, expr: body })
        };
        Ok(arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(q.outcome),
            distribution,
        }))
    };
    let path_level = |vi: DenseNodeId| {
        if keep_out.contains(&vi) { active.clone() } else { control.clone() }
    };
    let left = side(&path_level, &mut arena)?;
    let right = side(&|_| control.clone(), &mut arena)?;
    let contrast = arena.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
    arena.set_derivation(
        contrast,
        DerivationMeta {
            rule: Arc::from("path_specific.edge_gformula"),
            note: Some(Arc::from(
                "active level on treatment edges that start a selected path; control elsewhere",
            )),
        },
    );
    let functional =
        arena.simplify(contrast).map_err(|e| IdentificationError::msg(e.to_string()))?;
    Ok((arena, functional))
}

fn surgical_admg(
    admg: &Admg,
    treatment: DenseNodeId,
    keep_out: &HashSet<DenseNodeId>,
) -> Result<Admg, IdentificationError> {
    let mut out = Admg::empty();
    for node in admg.nodes() {
        out.add_node(*node).map_err(IdentificationError::from)?;
    }
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for &to in admg.children(from) {
            if from == treatment && !keep_out.contains(&to) {
                continue;
            }
            out.insert_directed(from, to).map_err(IdentificationError::from)?;
        }
    }
    // Bidirected edges unchanged.
    for i in 0..admg.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for &b in admg.bidirected_neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            out.insert_bidirected(a, b).map_err(IdentificationError::from)?;
        }
    }
    Ok(out)
}

fn admg_to_dag(admg: &Admg) -> Result<Dag, IdentificationError> {
    let n = u32::try_from(admg.node_count()).expect("fit");
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for &to in admg.children(from) {
            dag.insert_directed(from, to).map_err(IdentificationError::from)?;
        }
    }
    Ok(dag)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Intervention, MediationContrast, MediationQuery, Value, VariableId};
    use antecedent_graph::DenseNodeId;

    use super::*;
    use crate::identifier::IdentificationWorkspace;
    use crate::result::IdentificationStatus;

    fn chain_with_direct() -> Dag {
        // T → M → Y and T → Y
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        dag.insert_directed(t, m).unwrap();
        dag.insert_directed(m, y).unwrap();
        dag.insert_directed(t, y).unwrap();
        dag
    }

    #[test]
    fn mediated_only_path_identifies() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/path_specific/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["method"].as_str(), Some("path_specific.natural"));
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].method.as_ref(), EstimandMethod::PathSpecificNatural.as_str());
    }

    /// A truncated path enumeration must never yield an "identified" verdict.
    ///
    /// The recanting-witness rule concludes identifiability from the *absence* of a witness.
    /// On the graph below `w` is a genuine recanting witness (`w→a` on π, `w→b` off it), so
    /// the correct answer is `NotIdentified`. Before the fix, `max_paths = 1` enumerated only
    /// the π path, left the complement empty, found no witness, and returned
    /// `NonparametricallyIdentified` — licensing an estimand Avin–Shpitser–Pearl forbids.
    #[test]
    fn truncated_path_enumeration_never_claims_identification() {
        // 0=t 1=w 2=a 3=b 4=c 5=y;  t→c→y, t→w→b→y, t→w→a→y.  π = paths through `a`.
        let mut dag = Dag::with_variables(6);
        for (u, v) in [(0, 4), (4, 5), (0, 1), (1, 3), (3, 5), (1, 2), (2, 5)] {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();

        for max_paths in [1usize, 2, 3, 64] {
            let q =
                PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(5))
                    .with_path_nodes([VariableId::from_raw(2)])
                    .with_max_paths(max_paths);
            let cq = CausalQuery::PathSpecific(q);
            let mut ws = IdentificationWorkspace::default();
            let res = id.identify(&prep, &cq, &mut ws).unwrap();
            assert_eq!(
                res.status,
                IdentificationStatus::NotIdentified,
                "max_paths={max_paths} must not identify past a recanting witness"
            );
        }

        // The untruncated run must reach the recanting rule itself, not merely fail closed.
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(5))
            .with_path_nodes([VariableId::from_raw(2)]);
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert!(
            res.derivation
                .steps
                .iter()
                .any(|s| s.rule.contains("recanting") && s.detail.contains("blocks")),
            "full enumeration should cite the recanting witness, got {:?}",
            res.derivation.steps
        );
    }

    fn has_rule(res: &IdentificationResult, rule: &str) -> bool {
        res.derivation.steps.iter().any(|s| s.rule.as_ref() == rule)
    }

    /// With a complementary direct path the functional is the edge g-formula:
    /// the mediator factor binds the treatment to the active level and the
    /// outcome factor binds it to the control level on the left side.
    #[test]
    fn complementary_path_uses_edge_g_formula_levels() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &CausalQuery::PathSpecific(q), &mut ws).unwrap();
        assert!(has_rule(&res, "path_specific.edge_gformula"), "{:?}", res.derivation.steps);

        let (left, right) = contrast_levels(&res);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0)], "M at active, Y at control");
        assert_eq!(right, vec![(1, 0.0), (2, 0.0)], "reference side all control");
    }

    /// `(factor variable, bound treatment level)` pairs on each side of the
    /// edge g-formula contrast, sorted by variable.
    fn contrast_levels(res: &IdentificationResult) -> (Vec<(u32, f64)>, Vec<(u32, f64)>) {
        let ExprNode::Contrast { left, right, .. } = res.arena.node(res.estimands[0].functional)
        else {
            panic!("expected a contrast");
        };
        let levels = |root: ExprId| {
            let mut out = Vec::new();
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                match res.arena.node(id) {
                    ExprNode::Distribution { variables, intervention, .. } => {
                        for a in res.arena.intervention_assignments(*intervention) {
                            out.push((
                                res.arena.var_set(*variables)[0].raw(),
                                a.value.as_f64().unwrap(),
                            ));
                        }
                    }
                    ExprNode::Product(list) => stack.extend(res.arena.list(*list)),
                    ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                        stack.push(*expr);
                    }
                    ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
                    other => panic!("unexpected node {other:?}"),
                }
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        (levels(*left), levels(*right))
    }

    fn dag_from(n: u32, edges: &[(u32, u32)]) -> Dag {
        let mut dag = Dag::with_variables(n);
        for &(u, v) in edges {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        dag
    }

    fn identify_on(dag: &Dag, query: CausalQuery) -> IdentificationResult {
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(dag).unwrap();
        let mut ws = IdentificationWorkspace::default();
        id.identify(&prep, &query, &mut ws).unwrap()
    }

    fn path_query(t: u32, y: u32, path_nodes: &[u32]) -> CausalQuery {
        CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(VariableId::from_raw(t), VariableId::from_raw(y))
                .with_path_nodes(
                    path_nodes.iter().map(|&v| VariableId::from_raw(v)).collect::<Vec<_>>(),
                ),
        )
    }

    fn mediation_query(
        t: u32,
        y: u32,
        mediators: &[u32],
        contrast: MediationContrast,
    ) -> CausalQuery {
        CausalQuery::Mediation(MediationQuery::binary(
            VariableId::from_raw(t),
            VariableId::from_raw(y),
            mediators.iter().map(|&v| VariableId::from_raw(v)).collect::<Vec<_>>(),
            contrast,
        ))
    }

    fn witness(res: &IdentificationResult) -> Option<String> {
        res.derivation
            .steps
            .iter()
            .find(|s| s.rule.as_ref() == "path_specific.recanting" && s.detail.contains("blocks"))
            .map(|s| s.detail.to_string())
    }

    /// `T → A → W → Y` selected, `T → B → W → Y` not. `W` is on both paths but
    /// they leave `T` through different children, so `W` is not a recanting
    /// witness: `Σ P(a|t₁) P(b|t₀) P(w|a,b) P(y|w)` identifies the effect.
    #[test]
    fn shared_descendant_through_distinct_children_identifies() {
        // 0=T 1=A 2=B 3=W 4=Y
        let dag = dag_from(5, &[(0, 1), (1, 3), (0, 2), (2, 3), (3, 4)]);
        let res = identify_on(&dag, path_query(0, 4, &[1]));
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(witness(&res).is_none(), "{:?}", res.derivation.steps);
        assert!(has_rule(&res, "path_specific.edge_gformula"));
        let (left, right) = contrast_levels(&res);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0)], "A at active, B at control");
        assert_eq!(right, vec![(1, 0.0), (2, 0.0)]);
    }

    /// Natural direct effect with a post-mediator node: `T → M1 → W → Y`,
    /// `T → W → Y`, `T → Y`, mediator `M1`. `W` is on a mediated and an
    /// unmediated path, but `T`'s children `M1`, `W`, `Y` each start one kind
    /// only, so both natural effects are identified.
    #[test]
    fn mediation_with_post_mediator_node_identifies_both_natural_effects() {
        // 0=T 1=M1 2=W 3=Y
        let dag = dag_from(4, &[(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]);
        let nde = identify_on(&dag, mediation_query(0, 3, &[1], MediationContrast::NaturalDirect));
        assert_eq!(nde.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(has_rule(&nde, "path_specific.edge_gformula"));
        let (left, _) = contrast_levels(&nde);
        assert_eq!(left, vec![(1, 0.0), (2, 1.0), (3, 1.0)], "M1 control; W and Y active");

        let nie =
            identify_on(&dag, mediation_query(0, 3, &[1], MediationContrast::NaturalIndirect));
        assert_eq!(nie.status, IdentificationStatus::NonparametricallyIdentified);
        let (left, _) = contrast_levels(&nie);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0), (3, 0.0)], "M1 active; W and Y control");
    }

    /// `T → L → M → Y`, `T → L → Y`, `T → M → Y`: `L` is a child of `T` that
    /// starts a mediated and an unmediated path, a genuine recanting witness for
    /// both natural effects and for the path set through `M`.
    #[test]
    fn treatment_child_starting_both_kinds_of_path_refuses() {
        // 0=T 1=L 2=M 3=Y
        let dag = dag_from(4, &[(0, 1), (1, 2), (1, 3), (0, 2), (2, 3)]);
        for query in [
            path_query(0, 3, &[2]),
            mediation_query(0, 3, &[2], MediationContrast::NaturalIndirect),
            mediation_query(0, 3, &[2], MediationContrast::NaturalDirect),
        ] {
            let res = identify_on(&dag, query);
            assert_eq!(res.status, IdentificationStatus::NotIdentified);
            let detail = witness(&res).expect("recanting witness step");
            assert!(detail.contains("dense=1"), "L is the witness: {detail}");
        }
    }

    /// A selection that is not closed under the path edges: only
    /// `T → A → R → Z → Y` is selected, so `A` starts it and the unselected
    /// `T → A → R → Y`.
    #[test]
    fn selection_splitting_below_a_treatment_child_refuses() {
        // 0=T 1=A 2=B 3=R 4=Z 5=Y
        let dag = dag_from(6, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 5), (3, 4), (4, 5)]);
        let res = identify_on(&dag, path_query(0, 5, &[1, 4]));
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(witness(&res).expect("witness").contains("dense=1"));
    }

    /// Every case frozen in `conformance/identify/path_specific`.
    #[test]
    fn frozen_identification_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/path_specific/expected.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let mut names: Vec<String> = Vec::new();
            let mut index = |name: &str| {
                names.iter().position(|n| n == name).unwrap_or_else(|| {
                    names.push(name.to_owned());
                    names.len() - 1
                })
            };
            let edges: Vec<(u32, u32)> = case["dag"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    let (u, v) = e.as_str().unwrap().split_once("->").unwrap();
                    (index(u.trim()) as u32, index(v.trim()) as u32)
                })
                .collect();
            let path_nodes: Vec<u32> = case["included_path_nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| index(v.as_str().unwrap()) as u32)
                .collect();
            let (t, y) = (index("T") as u32, index("Y") as u32);
            let witness_node = case["witness"].as_str().map(&mut index);
            let dag = dag_from(names.len() as u32, &edges);
            let res = identify_on(&dag, path_query(t, y, &path_nodes));
            match case["status"].as_str().unwrap() {
                "identified" => {
                    assert_eq!(
                        res.status,
                        IdentificationStatus::NonparametricallyIdentified,
                        "{id}"
                    );
                    if let Some(method) = case["method"].as_str() {
                        assert_eq!(res.estimands[0].method.as_ref(), method, "{id}");
                    }
                }
                "not_identified" => {
                    assert_eq!(res.status, IdentificationStatus::NotIdentified, "{id}");
                    let w = witness_node.expect("not_identified case names its witness");
                    let detail = witness(&res).unwrap_or_default();
                    assert!(detail.contains(&format!("dense={w} ")), "{id}: {detail}");
                }
                other => panic!("{id}: unknown status {other}"),
            }
        }
    }

    /// Latent confounding on an outcome ancestor needs the district formula,
    /// which is not implemented: refuse rather than emit a node-wise product.
    #[test]
    fn edge_g_formula_refuses_bidirected_ancestor() {
        let mut admg = Admg::with_variables(3);
        let (t, m, y) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        admg.insert_directed(t, m).unwrap();
        admg.insert_directed(m, y).unwrap();
        admg.insert_directed(t, y).unwrap();
        admg.insert_bidirected(m, y).unwrap();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare(&admg).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let mut ws = IdentificationWorkspace::default();
        let err = id.identify(&prep, &CausalQuery::PathSpecific(q), &mut ws).unwrap_err();
        assert!(err.to_string().contains("bidirected"), "{err}");
    }

    #[test]
    fn all_paths_identifies_total() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2));
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!has_rule(&res, "path_specific.edge_gformula"), "no complement: total effect");
    }

    #[test]
    fn set_interventions_required() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let mut q =
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2));
        q.control = Intervention::set(VariableId::from_raw(0), Value::f64(0.0));
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    }
}
