//! Front-door criterion identification for DAGs.
//!
//! # Search completeness
//!
//! The mediator search is deliberately not exhaustive. Only two families of
//! candidate sets are tested against the front-door criterion:
//!
//! 1. every singleton `{v}` with `v ∉ {T, Y}`, and
//! 2. the full set `children(T) \ {Y}` (when it has more than one member).
//!
//! Valid mediator sets outside these families are missed: in particular,
//! multi-node sets that are proper subsets of `children(T) \ {Y}`, sets mixing
//! children of `T` with downstream intermediates, and sets of non-child
//! intermediates that jointly intercept all directed `T -> Y` paths where no
//! single node does. A `NotIdentified` result therefore means "no candidate in
//! the searched families qualifies", not that no front-door set exists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use causal_expr::CausalExprArena;
use causal_graph::{DSeparationWorkspace, Dag, DenseNodeId};

use crate::backdoor::{
    PreparedIdentificationGraph, dense_to_var, remove_nodes, remove_outgoing, remove_outgoing_set,
    var_to_dense,
};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for front-door mediator-set search.
#[derive(Clone, Debug)]
pub struct FrontDoorSearchConfig {
    /// Maximum number of mediator sets to return.
    pub max_results: usize,
}

impl Default for FrontDoorSearchConfig {
    fn default() -> Self {
        Self { max_results: 64 }
    }
}

/// Front-door criterion identifier for static DAGs.
#[derive(Clone, Debug, Default)]
pub struct FrontDoorIdentifier {
    /// Search configuration.
    pub config: FrontDoorSearchConfig,
}

impl FrontDoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph for front-door identification.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::new(graph.clone()))
    }

    /// Identify an average-effect query via the front-door criterion.
    ///
    /// Only singleton mediator sets and the full `children(T) \ {Y}` set are
    /// searched (see the module docs for what this can miss); a
    /// `NotIdentified` status is relative to those candidate families.
    ///
    /// # Errors
    ///
    /// Unsupported query or unknown variables.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "front-door identification only supports AverageEffect",
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone())
    }

    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let dag = prepared.dag();
        let t = var_to_dense(ate.treatment, dag)?;
        let y = var_to_dense(ate.outcome, dag)?;

        // Candidate mediator sets: every singleton excluding T,Y, plus the
        // full set of T's children (excluding Y) when it has more than one
        // member. This is intentionally not a full subset search; valid
        // intermediate multi-node sets outside these families are missed
        // (see module docs).
        let mut candidates: Vec<Vec<DenseNodeId>> = Vec::new();
        for i in 0..dag.node_count() {
            let v = DenseNodeId::from_raw(u32::try_from(i).expect("node id fits u32"));
            if v == t || v == y {
                continue;
            }
            candidates.push(vec![v]);
        }
        let children_of_t: Vec<DenseNodeId> =
            dag.children(t).iter().copied().filter(|&c| c != y).collect();
        if children_of_t.len() > 1 {
            candidates.push(children_of_t);
        }

        let mut dsep_ws = DSeparationWorkspace::default();
        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut examined = 0u64;

        for m in &candidates {
            examined += 1;
            if is_frontdoor_set(dag, t, y, m, &mut dsep_ws)? {
                valid.push(m.clone());
                if valid.len() >= self.config.max_results {
                    break;
                }
            }
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("frontdoor"));

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "frontdoor.criterion",
            "M intercepts all directed T->Y paths; T-M and M-Y backdoors blocked",
        );

        if valid.is_empty() {
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }

        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        for m in &valid {
            let vars: Vec<VariableId> =
                m.iter().map(|d| dense_to_var(*d, dag)).collect::<Result<_, _>>()?;
            let (active, control) = match (&ate.active, &ate.control) {
                (
                    causal_core::Intervention::Set { value: active, .. },
                    causal_core::Intervention::Set { value: control, .. },
                ) => (active.clone(), control.clone()),
                _ => {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: "front-door ATE requires Set interventions",
                    });
                }
            };
            let functional =
                arena.frontdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
            estimands.push(IdentifiedEstimand::frontdoor("frontdoor", Arc::from(vars), functional));
            derivation.push("frontdoor.mediator_set", format!("|M|={}", m.len()));
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

/// Whether `m` satisfies the front-door criterion relative to `t` and `y`.
fn is_frontdoor_set(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    m: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    if m.is_empty() {
        return Ok(false);
    }

    // 1. M intercepts all directed paths from T to Y: removing M entirely
    // must disconnect T from Y via directed edges.
    let without_m = remove_nodes(dag, m)?;
    if without_m.reaches(t, y) {
        return Ok(false);
    }

    // 2. No unblocked backdoor path from T to any m in M: in G with T's
    // outgoing edges removed, T must be d-separated from every m given ∅.
    let t_mutilated = remove_outgoing(dag, t)?;
    for &mi in m {
        let sep = t_mutilated.is_d_separated(t, mi, &[], ws).map_err(IdentificationError::from)?;
        if !sep {
            return Ok(false);
        }
    }

    // 3. All backdoor paths from M to Y are blocked by T: in G with every
    // m's outgoing edges removed, each m must be d-separated from Y given {T}.
    let m_mutilated = remove_outgoing_set(dag, m)?;
    for &mi in m {
        let sep = m_mutilated.is_d_separated(mi, y, &[t], ws).map_err(IdentificationError::from)?;
        if !sep {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use causal_core::AverageEffectQuery;

    #[test]
    fn classic_frontdoor_with_unmeasured_confounder() {
        // U -> T, U -> Y, T -> M -> Y (U unmeasured; here modeled but never
        // adjustable so the direct T<-U->Y backdoor is unblockable without M).
        let mut g = Dag::with_variables(4);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands.iter().any(|e| e.mediators.as_ref() == [VariableId::from_raw(1)]));
    }

    #[test]
    fn chain_without_confounder_still_identifies() {
        // T -> M -> Y only.
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands.iter().any(|e| e.mediators.as_ref() == [VariableId::from_raw(1)]));
    }

    #[test]
    fn direct_edge_with_no_mediator_is_not_identified() {
        // T -> Y directly, no mediator can block the direct path.
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let w = DenseNodeId::from_raw(2);
        g.insert_directed(t, y).unwrap();
        g.insert_directed(t, w).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }
}
