//! Shpitser–Pearl IDC for conditional interventional distributions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::large_enum_variant, clippy::too_many_arguments)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, CausalQuery, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use causal_expr::{CausalExprArena, EstimandMethod, ExprNode, IdentifiedEstimand};
use causal_graph::{Admg, BitSet, DSeparationWorkspace, Dag, DenseNodeId};

use crate::error::IdentificationError;
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Identifier for conditional interventional distributions via IDC.
#[derive(Clone, Debug, Default)]
pub struct IdcIdentifier {
    /// Underlying unconditional ID engine.
    pub inner: IdIdentifier,
}

impl IdcIdentifier {
    /// Create the identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare an ADMG.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare(&self, graph: &Admg) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare(graph)
    }

    /// Prepare from a DAG (no bidirected edges).
    ///
    /// # Errors
    ///
    /// Graph construction failure.
    pub fn prepare_dag(&self, graph: &Dag) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare_dag(graph)
    }

    /// Prepare with assumptions.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Admg,
        assumptions: AssumptionSet,
    ) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare_with_assumptions(graph, assumptions)
    }

    /// Identify a distribution query. Empty conditioning delegates to ID;
    /// nonempty conditioning runs IDC.
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
            CausalQuery::Distribution(q) => {
                if q.conditioning.is_empty() {
                    return self.inner.identify(prepared, query, workspace);
                }
                crate::intervention_support::require_hard_set_interventions(
                    q.interventions.iter(),
                    "IDC",
                )?;
                let mut result = self.identify_conditional(
                    prepared,
                    q.outcomes.as_ref(),
                    q.interventions.as_ref(),
                    q.conditioning.as_ref(),
                    workspace,
                )?;
                // Preserve the caller's query (values + conditioning) on success.
                result.query = query.clone();
                Ok(result)
            }
            CausalQuery::AverageEffect(_) => self.inner.identify(prepared, query, workspace),
            _ => Err(IdentificationError::unsupported(
                "IdcIdentifier supports Distribution and AverageEffect queries",
            )),
        }
    }

    /// Identify `P(Y | do(X), Z)` via IDC.
    ///
    /// # Errors
    ///
    /// Unknown variables or ID failure plumbing.
    pub fn identify_conditional(
        &self,
        prepared: &PreparedAdmg,
        outcomes: &[VariableId],
        interventions: &[Intervention],
        conditioning: &[VariableId],
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        crate::intervention_support::require_hard_set_interventions(interventions, "IDC")?;
        let n = prepared.admg().node_count();
        let mut y = BitSet::with_len(n);
        for &v in outcomes {
            y.insert(prepared.var_to_dense(v)?);
        }
        let mut x = BitSet::with_len(n);
        let mut known_values = std::collections::BTreeMap::<VariableId, Value>::new();
        for iv in interventions {
            let v = iv
                .primary_variable()
                .ok_or(IdentificationError::unsupported("intervention missing primary variable"))?;
            x.insert(prepared.var_to_dense(v)?);
            if let Intervention::Set { value, .. } = iv {
                known_values.insert(v, value.clone());
            }
        }
        let mut z = BitSet::with_len(n);
        for &v in conditioning {
            z.insert(prepared.var_to_dense(v)?);
        }

        let mut derivation = DerivationTrace::default();
        derivation.push("general.idc", "Shpitser–Pearl IDC");
        let mut perf = IdentificationPerformanceRecord::default();

        let expr_result = idc_recurse(
            &self.inner,
            prepared,
            &y,
            &x,
            &z,
            &known_values,
            workspace,
            &mut derivation,
            &mut perf,
        )?;

        match expr_result {
            IdcOk::Identified { functional, arena } => {
                let estimand = IdentifiedEstimand::new(
                    Arc::from(EstimandMethod::GeneralId.as_str()),
                    Arc::from([]),
                    Arc::from([]),
                    Arc::from([]),
                    functional,
                    None,
                );
                Ok(IdentificationResult::identified(
                    CausalQuery::Distribution(InterventionalDistributionQuery {
                        outcomes: Arc::from(outcomes.to_vec()),
                        interventions: Arc::from(interventions.to_vec()),
                        conditioning: Arc::from(conditioning.to_vec()),
                        target_population: causal_core::TargetPopulation::AllObserved,
                    }),
                    vec![estimand],
                    arena,
                    derivation,
                    prepared.declared_assumptions().clone(),
                    perf,
                ))
            }
            IdcOk::NotIdentified(res) => {
                let mut out = res;
                out.derivation.steps.splice(0..0, derivation.steps);
                out.performance.candidates_examined =
                    out.performance.candidates_examined.saturating_add(perf.candidates_examined);
                Ok(out)
            }
        }
    }
}

enum IdcOk {
    Identified { functional: causal_expr::ExprId, arena: CausalExprArena },
    NotIdentified(IdentificationResult),
}

fn idc_recurse(
    id: &IdIdentifier,
    prepared: &PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    z: &BitSet,
    known_values: &std::collections::BTreeMap<VariableId, Value>,
    workspace: &mut IdentificationWorkspace,
    derivation: &mut DerivationTrace,
    perf: &mut IdentificationPerformanceRecord,
) -> Result<IdcOk, IdentificationError> {
    perf.candidates_examined = perf.candidates_examined.saturating_add(1);

    // Line 1: move Z∈Z into intervention when Y ⊥ Z | X, Z\{Z} in G_{\bar X \underline Z}
    for z_node in z.to_dense_ids() {
        let mut z_rest = z.clone();
        z_rest.remove(z_node);
        let mut cond = x.clone();
        cond.union_with(&z_rest);
        if independent_in_mutilated(prepared.admg(), y, z_node, &cond, x, z, &mut workspace.dsep)? {
            derivation.push(
                "general.idc.line1",
                format!("insert Z={} into intervention (rule 2)", z_node.raw()),
            );
            let mut x2 = x.clone();
            x2.insert(z_node);
            return idc_recurse(
                id,
                prepared,
                y,
                &x2,
                &z_rest,
                known_values,
                workspace,
                derivation,
                perf,
            );
        }
    }

    // Line 2: P' = ID(Y∪Z, X); return P'/∑_Y P'
    derivation.push("general.idc.line2", "reduce to ID on Y∪Z");
    let mut yz = y.clone();
    yz.union_with(z);
    let q = distribution_query_from_sets(prepared, &yz, x, known_values)?;
    let id_res = id.identify(prepared, &q, workspace)?;
    if id_res.status != IdentificationStatus::NonparametricallyIdentified {
        return Ok(IdcOk::NotIdentified(id_res));
    }
    let functional = id_res.estimands[0].functional;
    let mut arena = id_res.arena;
    let y_vars = intern_bitset_vars(prepared, y, &mut arena)?;
    let denom = arena.intern(ExprNode::SumOut { variables: y_vars, expr: functional });
    let ratio = arena.intern(ExprNode::Ratio { numerator: functional, denominator: denom });
    let functional = arena.simplify(ratio);
    Ok(IdcOk::Identified { functional, arena })
}

fn distribution_query_from_sets(
    prepared: &PreparedAdmg,
    y: &BitSet,
    x: &BitSet,
    known_values: &std::collections::BTreeMap<VariableId, Value>,
) -> Result<CausalQuery, IdentificationError> {
    let outcomes: Result<Vec<_>, _> =
        y.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
    let interventions: Result<Vec<_>, _> = x
        .to_dense_ids()
        .into_iter()
        .map(|d| {
            prepared.dense_to_var(d).map(|variable| {
                // Prefer caller Set values; for IDC-moved conditioning vars use a finite
                // structure-only placeholder (ID cares about variable sets, not levels).
                let value = known_values.get(&variable).cloned().unwrap_or(Value::Int64(0));
                Intervention::set(variable, value)
            })
        })
        .collect();
    Ok(CausalQuery::Distribution(InterventionalDistributionQuery {
        outcomes: Arc::from(outcomes?),
        interventions: Arc::from(interventions?),
        conditioning: Arc::from([]),
        target_population: causal_core::TargetPopulation::AllObserved,
    }))
}

fn intern_bitset_vars(
    prepared: &PreparedAdmg,
    nodes: &BitSet,
    arena: &mut CausalExprArena,
) -> Result<causal_expr::VarSetId, IdentificationError> {
    let vars: Result<Vec<_>, _> =
        nodes.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect();
    Ok(arena.intern_var_set(vars?))
}

/// Y ⊥ `z_node` | cond in `G_{\overline{x_nodes} \underline{z_nodes}}`.
fn independent_in_mutilated(
    admg: &Admg,
    y: &BitSet,
    z_node: DenseNodeId,
    cond: &BitSet,
    x_nodes: &BitSet,
    z_nodes: &BitSet,
    dsep: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    let mutilated = mutilate_bar_x_underline_z(admg, x_nodes, z_nodes)?;
    let cond_ids = cond.to_dense_ids();
    // Y may be a set — require every y ∈ Y independent of z_node.
    for y_node in y.to_dense_ids() {
        if !mutilated
            .is_m_separated(y_node, z_node, &cond_ids, dsep)
            .map_err(IdentificationError::from)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn mutilate_bar_x_underline_z(
    admg: &Admg,
    x_nodes: &BitSet,
    z_nodes: &BitSet,
) -> Result<Admg, IdentificationError> {
    let mut out = Admg::empty();
    for node in admg.nodes() {
        out.add_node(*node).map_err(IdentificationError::from)?;
    }
    // Directed: drop incoming to X, drop outgoing from Z.
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        if z_nodes.contains(from) {
            continue; // no outgoing from Z
        }
        for &to in admg.children(from) {
            if x_nodes.contains(to) {
                continue; // no incoming to X
            }
            out.insert_directed(from, to).map_err(IdentificationError::from)?;
        }
    }
    // Bidirected unchanged (except endpoints still present).
    let mut seen = BitSet::with_len(admg.node_count());
    for i in 0..admg.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for &b in admg.bidirected_neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if seen.contains(a) && seen.contains(b) {
                continue;
            }
            out.insert_bidirected(a, b).map_err(IdentificationError::from)?;
        }
        seen.insert(a);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use causal_core::VariableId;
    use causal_graph::Dag;

    use super::*;
    use crate::identifier::IdentificationWorkspace;

    #[test]
    fn unconditional_matches_id() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let idc = IdcIdentifier::new();
        let prep = idc.prepare_dag(&dag).unwrap();
        let mut ws = IdentificationWorkspace::default();
        let res = idc
            .identify_conditional(
                &prep,
                &[VariableId::from_raw(2)],
                &[Intervention::set(VariableId::from_raw(1), Value::f64(1.0))],
                &[],
                &mut ws,
            )
            .unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        if let CausalQuery::Distribution(q) = &res.query {
            for iv in q.interventions.iter() {
                if let Intervention::Set { value, .. } = iv {
                    assert!(
                        value.as_f64().is_none_or(|x| !x.is_nan()),
                        "IDC must not invent NaN Set values"
                    );
                }
            }
        }
    }
}
