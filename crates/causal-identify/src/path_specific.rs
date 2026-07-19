//! Path-restricted natural-effect identification (Avin–Shpitser–Pearl).
//!
//! Enumerate directed paths π from treatment to outcome, reject recanting
//! descendants, surgically delete treatment out-edges not on π, then run
//! general ID for the active/control contrast on the modified ADMG.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use causal_core::{AverageEffectQuery, CausalQuery, PathSpecificEffectQuery};
use causal_expr::EstimandMethod;
use causal_graph::{Admg, Dag, DenseNodeId};

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
        let CausalQuery::PathSpecific(q) = query else {
            return Err(IdentificationError::unsupported(
                "PathSpecificIdentifier supports PathSpecific queries only",
            ));
        };
        q.validate().map_err(|_| IdentificationError::unsupported("invalid path-specific query"))?;
        self.identify_path_specific(prepared, q, query.clone(), workspace)
    }

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
        let path_filter: HashSet<DenseNodeId> = q
            .path_nodes
            .iter()
            .map(|&v| prepared.var_to_dense(v))
            .collect::<Result<_, _>>()?;

        let admg = prepared.admg();
        // Enumerate on the directed skeleton (DAG view of directed edges).
        let dag = admg_to_dag(admg)?;
        let raw_paths = dag
            .directed_paths(t, y, q.max_paths, q.max_len)
            .map_err(IdentificationError::from)?;
        perf.candidates_examined =
            perf.candidates_examined.saturating_add(raw_paths.len() as u64);

        let pi: Vec<Vec<DenseNodeId>> = raw_paths
            .into_iter()
            .filter(|path| path_matches_filter(path, &path_filter))
            .collect();
        if pi.is_empty() {
            derivation.push("path_specific.empty", "no directed paths match path_nodes filter");
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
        derivation.push(
            "path_specific.paths",
            format!("{} path(s) retained after filter", pi.len()),
        );

        // All directed paths (for complementary / recanting check).
        let all_paths = dag
            .directed_paths(t, y, q.max_paths, q.max_len)
            .map_err(IdentificationError::from)?;
        let pi_set: HashSet<Vec<DenseNodeId>> = pi.iter().cloned().collect();
        let complement: Vec<&Vec<DenseNodeId>> =
            all_paths.iter().filter(|p| !pi_set.contains(*p)).collect();

        if let Some(w) = recanting_descendant(t, y, &pi, &complement) {
            derivation.push(
                "path_specific.recanting",
                format!(
                    "recanting descendant dense={} blocks nonparametric path-specific ID",
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
        derivation.push("path_specific.recanting", "no recanting descendants");

        // Edges on π that leave the treatment.
        let mut keep_out: HashSet<DenseNodeId> = HashSet::new();
        for path in &pi {
            if path.len() >= 2 && path[0] == t {
                keep_out.insert(path[1]);
            }
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

        let ate = AverageEffectQuery {
            treatment: q.treatment,
            outcome: q.outcome,
            control: q.control.clone(),
            active: q.active.clone(),
            effect_modifiers: Arc::from([]),
            target_population: q.target_population.clone(),
        };
        let mut id_res = self.inner.identify_ate(&surgical_prep, &ate, workspace)?;
        id_res.derivation.steps.splice(0..0, derivation.steps);
        id_res.performance.candidates_examined = id_res
            .performance
            .candidates_examined
            .saturating_add(perf.candidates_examined);
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
    let mid: HashSet<DenseNodeId> = path.iter().copied().skip(1).take(path.len().saturating_sub(2)).collect();
    filter.iter().all(|n| mid.contains(n))
}

/// Proper descendants of `t` that appear on both a π path and a complementary path (≠ y).
fn recanting_descendant(
    t: DenseNodeId,
    y: DenseNodeId,
    pi: &[Vec<DenseNodeId>],
    complement: &[&Vec<DenseNodeId>],
) -> Option<DenseNodeId> {
    let mut on_pi: HashSet<DenseNodeId> = HashSet::new();
    for path in pi {
        for &n in path.iter().skip(1) {
            if n != y && n != t {
                on_pi.insert(n);
            }
        }
    }
    let mut on_comp: HashSet<DenseNodeId> = HashSet::new();
    for path in complement {
        for &n in path.iter().skip(1) {
            if n != y && n != t {
                on_comp.insert(n);
            }
        }
    }
    on_pi.intersection(&on_comp).next().copied()
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
    use causal_core::{Intervention, Value, VariableId};
    use causal_graph::DenseNodeId;

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
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(
            res.estimands[0].method.as_ref(),
            EstimandMethod::PathSpecificNatural.as_str()
        );
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
    }

    #[test]
    fn set_interventions_required() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let mut q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2));
        q.control = Intervention::set(VariableId::from_raw(0), Value::f64(0.0));
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    }
}
