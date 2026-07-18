//! Front-door criterion identification for DAGs.
//!
//! # Search completeness
//!
//! Candidate mediator sets include:
//!
//! 1. every singleton `{v}` with `v ∉ {T, Y}`,
//! 2. every non-empty subset of `children(T) \ {Y}` up to
//!    [`FrontDoorSearchConfig::max_mediator_set_size`] (cardinality bound), and
//! 3. the full set `children(T) \ {Y}` when it exceeds that cardinality bound
//!    (still tested as one candidate).
//!
//! Valid mediator sets outside these families are still missed: sets mixing
//! children of `T` with non-child intermediates, and sets of non-child
//! intermediates that jointly intercept all directed `T -> Y` paths. A
//! `NotIdentified` result therefore means "no candidate in the searched
//! families qualifies", not that no front-door set exists.
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
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for front-door mediator-set search.
#[derive(Clone, Debug)]
pub struct FrontDoorSearchConfig {
    /// Maximum number of mediator sets to return.
    pub max_results: usize,
    /// Maximum cardinality of subsets of `children(T)\{Y}` enumerated as candidates.
    ///
    /// Singletons outside that child set are still tested. Subsets larger than this
    /// bound are skipped except the full child set (tested once when oversized).
    pub max_mediator_set_size: usize,
}

impl Default for FrontDoorSearchConfig {
    fn default() -> Self {
        Self { max_results: 64, max_mediator_set_size: 4 }
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
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::with_assumptions(graph.clone(), assumptions))
    }

    /// Identify an average-effect query via the front-door criterion.
    ///
    /// Searches singletons and bounded subsets of `children(T) \ {Y}` (see module
    /// docs); a `NotIdentified` status is relative to those candidate families.
    ///
    /// # Errors
    ///
    /// Unsupported query or unknown variables.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "front-door identification only supports AverageEffect",
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone(), workspace)
    }

    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let dag = prepared.dag();
        let t = var_to_dense(ate.treatment, dag)?;
        let y = var_to_dense(ate.outcome, dag)?;

        let mut candidates: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut seen = std::collections::BTreeSet::<Vec<u32>>::new();
        let mut push_candidate = |m: Vec<DenseNodeId>, candidates: &mut Vec<Vec<DenseNodeId>>| {
            let mut key: Vec<u32> = m.iter().map(|d| d.raw()).collect();
            key.sort_unstable();
            if seen.insert(key) {
                candidates.push(m);
            }
        };

        // Singletons excluding T,Y.
        for i in 0..dag.node_count() {
            let v = DenseNodeId::from_raw(u32::try_from(i).expect("node id fits u32"));
            if v == t || v == y {
                continue;
            }
            push_candidate(vec![v], &mut candidates);
        }

        // Non-empty subsets of children(T)\{Y} up to max_mediator_set_size.
        let children_of_t: Vec<DenseNodeId> =
            dag.children(t).iter().copied().filter(|&c| c != y).collect();
        let max_k = self.config.max_mediator_set_size.min(children_of_t.len());
        for k in 1..=max_k {
            for subset in combinations(&children_of_t, k) {
                push_candidate(subset, &mut candidates);
            }
        }
        // If the full child set exceeds the cardinality bound, still test it once.
        if children_of_t.len() > max_k {
            push_candidate(children_of_t, &mut candidates);
        }

        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut examined = 0u64;

        for m in &candidates {
            examined += 1;
            if is_frontdoor_set(dag, t, y, m, &mut workspace.dsep)? {
                valid.push(m.clone());
                if valid.len() >= self.config.max_results {
                    break;
                }
            }
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("frontdoor"));
        for record in &prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }

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

/// Lexicographic combinations of `items` choose `k`.
fn combinations(items: &[DenseNodeId], k: usize) -> Vec<Vec<DenseNodeId>> {
    let n = items.len();
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        // Find rightmost index that can be incremented.
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] < n - k + i {
                idx[i] += 1;
                for j in (i + 1)..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
    }
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }

    #[test]
    fn proper_subset_of_children_identifies_when_full_set_fails_backdoor() {
        // T→M1→Y, T→M2→Y, T→M3 with W confounding M3–Y and U confounding T–Y.
        // No singleton intercepts both directed paths; {M1,M2} is FD; {M1,M2,M3} fails
        // criterion 3 for M3. Pre-fix search (singletons + full only) missed {M1,M2}.
        let mut g = Dag::with_variables(6);
        let t = DenseNodeId::from_raw(0);
        let m1 = DenseNodeId::from_raw(1);
        let m2 = DenseNodeId::from_raw(2);
        let m3 = DenseNodeId::from_raw(3);
        let y = DenseNodeId::from_raw(4);
        let w = DenseNodeId::from_raw(5);
        g.insert_directed(t, m1).unwrap();
        g.insert_directed(m1, y).unwrap();
        g.insert_directed(t, m2).unwrap();
        g.insert_directed(m2, y).unwrap();
        g.insert_directed(t, m3).unwrap();
        g.insert_directed(w, m3).unwrap();
        g.insert_directed(w, y).unwrap();
        g.insert_directed(w, t).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(4),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let want = [VariableId::from_raw(1), VariableId::from_raw(2)];
        assert!(
            res.estimands.iter().any(|e| {
                let mut m = e.mediators.to_vec();
                m.sort_by_key(|v| v.raw());
                m == want
            }),
            "expected mediators {{M1,M2}}; got {:?}",
            res.estimands.iter().map(|e| e.mediators.clone()).collect::<Vec<_>>()
        );
    }
}
