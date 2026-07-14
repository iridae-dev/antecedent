//! Backdoor adjustment identification for DAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use causal_expr::CausalExprArena;
use causal_graph::{BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for adjustment-set enumeration.
#[derive(Clone, Debug)]
pub struct AdjustmentSearchConfig {
    /// Maximum number of adjustment sets to return.
    pub max_results: usize,
    /// Variables forbidden from appearing in Z.
    pub forbidden: Arc<[VariableId]>,
    /// If true, only return inclusion-minimal sets.
    pub minimal_only: bool,
}

impl Default for AdjustmentSearchConfig {
    fn default() -> Self {
        Self { max_results: 64, forbidden: Arc::from([]), minimal_only: true }
    }
}

/// Prepared DAG for identification (cached ancestry helpers via workspaces).
#[derive(Clone, Debug)]
pub struct PreparedIdentificationGraph {
    dag: Dag,
}

impl PreparedIdentificationGraph {
    /// Wrap a DAG.
    #[must_use]
    pub fn new(dag: Dag) -> Self {
        Self { dag }
    }

    /// Borrow the DAG.
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.dag
    }
}

/// Identifier for static DAGs (Phase 1: backdoor).
#[derive(Clone, Debug, Default)]
pub struct BackdoorIdentifier {
    /// Search configuration.
    pub config: AdjustmentSearchConfig,
}

impl BackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph (Phase 1: identity wrap).
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::new(graph.clone()))
    }

    /// Identify an average-effect query via backdoor adjustment.
    ///
    /// At most `config.max_results` adjustment sets are returned; when more
    /// qualifying sets exist the result is truncated (noted in the derivation
    /// trace) rather than treated as an error.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or a candidate pool too large
    /// for exact enumeration.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "Phase 1 only supports AverageEffect",
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone())
    }

    #[allow(clippy::too_many_lines)]
    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let dag = prepared.dag();
        let t = var_to_dense(ate.treatment, dag)?;
        let y = var_to_dense(ate.outcome, dag)?;
        let mut forbidden = BitSet::with_len(dag.node_count());
        for &v in self.config.forbidden.iter() {
            forbidden.insert(var_to_dense(v, dag)?);
        }
        forbidden.insert(t);
        forbidden.insert(y);

        // Descendants of T cannot be in Z.
        let mut desc = BitSet::with_len(dag.node_count());
        let mut gws = GraphWorkspace::default();
        dag.descendants_of(&[t], &mut desc, &mut gws);
        for i in 0..dag.node_count() {
            let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            if desc.contains(id) {
                forbidden.insert(id);
            }
        }

        // G underbar T: remove outgoing edges from T.
        let mutilated = remove_outgoing(dag, t)?;

        let candidates: Vec<DenseNodeId> = (0..dag.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
            .filter(|id| !forbidden.contains(*id))
            .collect();

        let mut dsep_ws = DSeparationWorkspace::default();
        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut examined = 0u64;
        let mut truncated = false;

        // Enumerate subsets by increasing size for minimal-first. When more
        // than `max_results` qualifying sets exist, the first `max_results`
        // (in size-then-mask order) are returned and the truncation is noted
        // in the derivation trace.
        let m = candidates.len();
        if m > 20 {
            return Err(IdentificationError::NotIdentified {
                message: "candidate set too large for exact Phase 1 enumeration (>20)",
            });
        }
        let total_masks = 1usize << m;
        'sizes: for size in 0..=m {
            for mask in 0..total_masks {
                if mask.count_ones() as usize != size {
                    continue;
                }
                examined += 1;
                let z: Vec<DenseNodeId> =
                    (0..m).filter(|i| (mask & (1 << i)) != 0).map(|i| candidates[i]).collect();
                if !is_backdoor_adjustment(&mutilated, t, y, &z, &mut dsep_ws)? {
                    continue;
                }
                if self.config.minimal_only
                    && valid.iter().any(|prev| is_subset(prev, &z) && prev.len() < z.len())
                {
                    continue;
                }
                // Also skip if this set is a superset of an already found minimal set.
                if self.config.minimal_only && valid.iter().any(|prev| is_subset(prev, &z)) {
                    continue;
                }
                valid.push(z);
                if valid.len() >= self.config.max_results {
                    truncated = true;
                    break 'sizes;
                }
            }
            // After finishing a size class, if we have minimal sets, stop growing.
            if self.config.minimal_only && !valid.is_empty() {
                break;
            }
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("backdoor"));

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "backdoor.criterion",
            "Z blocks all backdoor paths and contains no descendants of T",
        );
        if truncated {
            derivation.push(
                "backdoor.enumeration",
                format!(
                    "result limit reached; returning first {} adjustment sets (more exist)",
                    valid.len()
                ),
            );
        }

        if valid.is_empty() {
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord {
                    candidates_examined: examined,
                    sets_returned: 0,
                },
            ));
        }

        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        for z in &valid {
            let vars: Vec<VariableId> =
                z.iter().map(|d| dense_to_var(*d, dag)).collect::<Result<_, _>>()?;
            let (active, control) = match (&ate.active, &ate.control) {
                (
                    causal_core::Intervention::Set { value: active, .. },
                    causal_core::Intervention::Set { value: control, .. },
                ) => (active.clone(), control.clone()),
                _ => {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: "Phase 1 backdoor ATE requires Set interventions",
                    });
                }
            };
            let functional = arena.backdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
            estimands.push(IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from(vars),
                functional,
            ));
            derivation.push("backdoor.adjustment_set", format!("|Z|={}", z.len()));
        }

        Ok(IdentificationResult::identified(
            query,
            estimands,
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: u64::try_from(valid.len()).unwrap_or(u64::MAX),
            },
        ))
    }
}

fn is_subset(small: &[DenseNodeId], big: &[DenseNodeId]) -> bool {
    small.iter().all(|s| big.contains(s))
}

pub(crate) fn is_backdoor_adjustment(
    mutilated: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    z: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    mutilated.is_d_separated(t, y, z, ws).map_err(IdentificationError::from)
}

pub(crate) fn remove_outgoing(dag: &Dag, t: DenseNodeId) -> Result<Dag, IdentificationError> {
    remove_outgoing_set(dag, &[t])
}

/// `G` with outgoing edges from every node in `nodes` removed (node ids preserved).
pub(crate) fn remove_outgoing_set(
    dag: &Dag,
    nodes: &[DenseNodeId],
) -> Result<Dag, IdentificationError> {
    let n = u32::try_from(dag.node_count()).map_err(|_| IdentificationError::msg("n"))?;
    let mut out = Dag::with_variables(n);
    for e in dag.edges() {
        let (from, to) = e.parent_child().expect("dag");
        if nodes.contains(&from) {
            continue;
        }
        out.insert_directed(from, to).map_err(IdentificationError::from)?;
    }
    Ok(out)
}

/// `G` with every node in `nodes` fully removed (both incoming and outgoing
/// edges dropped; node ids and count preserved).
pub(crate) fn remove_nodes(dag: &Dag, nodes: &[DenseNodeId]) -> Result<Dag, IdentificationError> {
    let n = u32::try_from(dag.node_count()).map_err(|_| IdentificationError::msg("n"))?;
    let mut out = Dag::with_variables(n);
    for e in dag.edges() {
        let (from, to) = e.parent_child().expect("dag");
        if nodes.contains(&from) || nodes.contains(&to) {
            continue;
        }
        out.insert_directed(from, to).map_err(IdentificationError::from)?;
    }
    Ok(out)
}

pub(crate) fn var_to_dense(id: VariableId, dag: &Dag) -> Result<DenseNodeId, IdentificationError> {
    let dense = DenseNodeId::from_raw(id.raw());
    if dense.as_usize() >= dag.node_count() {
        return Err(IdentificationError::UnknownVariable { id });
    }
    Ok(dense)
}

pub(crate) fn dense_to_var(id: DenseNodeId, dag: &Dag) -> Result<VariableId, IdentificationError> {
    match dag.nodes().get(id.as_usize()) {
        Some(causal_graph::NodeRef::Static(v)) => Ok(*v),
        _ => Err(IdentificationError::msg("expected static node")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::AverageEffectQuery;
    use crate::result::IdentificationStatus;

    #[test]
    fn confounding_requires_z() {
        // T <- Z -> Y, T -> Y
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    }

    #[test]
    fn result_limit_truncates_instead_of_erroring() {
        // T <- A -> B -> C -> Y, T -> Y: minimal singletons {A}, {B}, {C}.
        // With max_results = 2 the first two are returned, not an error.
        let mut g = Dag::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_directed(c, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let mut id = BackdoorIdentifier::new();
        id.config.max_results = 2;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 2);
        assert!(res.derivation.steps.iter().any(|s| s.rule.as_ref() == "backdoor.enumeration"));
    }

    #[test]
    fn empty_adjustment_when_no_backdoor() {
        // T -> Y only
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands[0].adjustment_set.is_empty());
    }
}
