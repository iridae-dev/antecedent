//! Efficient backdoor adjustment-set selection.
//!
//! Selects the *optimal* adjustment set `O` (Henckel, Perković & Maathuis;
//! the construction `DoWhy`'s efficient backdoor implements): with
//! `cn = de(T) ∩ an(Y) \ {T}` (the nodes on proper causal paths from `T` to
//! `Y`, including `Y`), `forb = de(cn) ∪ {T}`, the O-set is
//! `O = pa(cn) \ forb`. Among all valid backdoor sets in a fully observed
//! DAG, `O` yields the smallest asymptotic variance for the adjusted effect
//! estimate. The candidate `O` is still validated against the backdoor
//! criterion (d-separation in the treatment-outgoing-mutilated graph, no
//! descendants of `T`, no configured forbidden variables) before being
//! returned.
//!
//! When `O` fails validation (e.g. a member is configured as forbidden), the
//! identifier falls back to a minimum-cardinality search: subsets are
//! enumerated by increasing size, stopping at the first size class with a
//! valid set, with ties broken by maximizing `|Z ∩ Pa(T)|` then lexicographic
//! variable ids. The fallback keeps at most `max_results` valid sets; hitting
//! that limit truncates collection rather than erroring.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use causal_expr::CausalExprArena;
use causal_graph::{BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace};

use crate::backdoor::{
    AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph, dense_to_var,
    is_backdoor_adjustment, remove_outgoing, var_to_dense,
};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Identifier that returns one efficient backdoor adjustment set.
#[derive(Clone, Debug, Default)]
pub struct EfficientBackdoorIdentifier {
    /// Shared search limits / forbidden variables.
    pub config: AdjustmentSearchConfig,
}

impl EfficientBackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph (same wrap as [`BackdoorIdentifier`]).
    ///
    /// # Errors
    ///
    /// Propagates prepare failures.
    pub fn prepare(
        &self,
        graph: &causal_graph::Dag,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        BackdoorIdentifier { config: self.config.clone() }.prepare(graph)
    }

    /// Identify via efficient backdoor; at most one estimand is returned,
    /// preferring the O-set (optimal adjustment set) and falling back to a
    /// minimum-cardinality search when the O-set fails validation.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or a candidate pool too large
    /// for the exact fallback enumeration.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "efficient backdoor only supports AverageEffect",
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
        let mut forbidden = BitSet::with_len(dag.node_count());
        for &v in self.config.forbidden.iter() {
            forbidden.insert(var_to_dense(v, dag)?);
        }
        forbidden.insert(t);
        forbidden.insert(y);

        let mut desc = BitSet::with_len(dag.node_count());
        let mut gws = GraphWorkspace::default();
        dag.descendants_of(&[t], &mut desc, &mut gws);
        for i in 0..dag.node_count() {
            let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            if desc.contains(id) {
                forbidden.insert(id);
            }
        }

        let mutilated = remove_outgoing(dag, t)?;
        let candidates: Vec<DenseNodeId> = (0..dag.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
            .filter(|id| !forbidden.contains(*id))
            .collect();

        let mut dsep_ws = DSeparationWorkspace::default();
        let mut examined = 0u64;

        // Primary rule: the O-set (optimal adjustment set, Henckel et al.).
        let o_set = optimal_adjustment_set(dag, t, y);
        examined += 1;
        if o_set.iter().all(|v| !forbidden.contains(*v))
            && is_backdoor_adjustment(&mutilated, t, y, &o_set, &mut dsep_ws)?
        {
            return Self::finish(ate, query, dag, &[o_set], examined, "optimal (O-set)");
        }

        // Fallback: minimum-cardinality search, sizes ascending, stopping at
        // the first size class that yields a valid set. Collection is capped
        // at `max_results`; hitting the cap truncates rather than errors.
        let m = candidates.len();
        if m > 20 {
            return Err(IdentificationError::NotIdentified {
                message: "candidate set too large for exact efficient enumeration (>20)",
            });
        }

        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let total_masks = 1usize << m;
        'sizes: for size in 0..=m {
            for mask in 0..total_masks {
                if mask.count_ones() as usize != size {
                    continue;
                }
                examined += 1;
                let z: Vec<DenseNodeId> =
                    (0..m).filter(|i| (mask & (1 << i)) != 0).map(|i| candidates[i]).collect();
                if is_backdoor_adjustment(&mutilated, t, y, &z, &mut dsep_ws)? {
                    valid.push(z);
                    if valid.len() >= self.config.max_results {
                        break 'sizes;
                    }
                }
            }
            if !valid.is_empty() {
                break;
            }
        }

        if valid.is_empty() {
            return Ok(IdentificationResult::not_identified(
                query,
                {
                    let mut d = DerivationTrace::default();
                    d.push("backdoor.efficient", "no valid adjustment set");
                    d
                },
                default_assumptions(),
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }

        let parent_set: BitSet = {
            let mut b = BitSet::with_len(dag.node_count());
            for p in dag.parents(t) {
                b.insert(*p);
            }
            b
        };
        valid.sort_by(|a, b| {
            let pa = a.iter().filter(|n| parent_set.contains(**n)).count();
            let pb = b.iter().filter(|n| parent_set.contains(**n)).count();
            a.len().cmp(&b.len()).then_with(|| pb.cmp(&pa)).then_with(|| {
                let mut aa: Vec<_> = a.iter().map(|n| n.as_usize()).collect();
                let mut bb: Vec<_> = b.iter().map(|n| n.as_usize()).collect();
                aa.sort_unstable();
                bb.sort_unstable();
                aa.cmp(&bb)
            })
        });
        let best = vec![valid.into_iter().next().expect("non-empty")];
        Self::finish(ate, query, dag, &best, examined, "min_cardinality")
    }

    fn finish(
        ate: &AverageEffectQuery,
        query: CausalQuery,
        dag: &causal_graph::Dag,
        chosen: &[Vec<DenseNodeId>],
        examined: u64,
        rule: &str,
    ) -> Result<IdentificationResult, IdentificationError> {
        let z = &chosen[0];
        let vars: Vec<VariableId> =
            z.iter().map(|d| dense_to_var(*d, dag)).collect::<Result<_, _>>()?;
        let (active, control) = match (&ate.active, &ate.control) {
            (
                causal_core::Intervention::Set { value: active, .. },
                causal_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "efficient backdoor ATE requires Set interventions",
                });
            }
        };
        let mut arena = CausalExprArena::new();
        let functional = arena.backdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
        let mut derivation = DerivationTrace::default();
        derivation.push("backdoor.efficient", format!("selected via {rule}; |Z|={}", vars.len()));
        Ok(IdentificationResult::identified(
            query,
            vec![IdentifiedEstimand::backdoor("backdoor.efficient", Arc::from(vars), functional)],
            arena,
            derivation,
            default_assumptions(),
            IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
        ))
    }
}

/// The O-set of Henckel, Perković & Maathuis: `pa(cn) \ (de(cn) ∪ {T})`,
/// where `cn = de(T) ∩ an(Y) \ {T}` are the nodes on proper causal paths from
/// `T` to `Y` (including `Y` itself). In a fully observed DAG this is the
/// valid backdoor set with the smallest asymptotic variance.
fn optimal_adjustment_set(dag: &Dag, t: DenseNodeId, y: DenseNodeId) -> Vec<DenseNodeId> {
    let n = dag.node_count();
    let mut gws = GraphWorkspace::default();
    let mut de_t = BitSet::with_len(n);
    dag.descendants_of(&[t], &mut de_t, &mut gws);
    let mut an_y = BitSet::with_len(n);
    dag.ancestors_of(&[y], &mut an_y, &mut gws);

    let cn: Vec<DenseNodeId> = (0..n)
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
        .filter(|&v| v != t && de_t.contains(v) && an_y.contains(v))
        .collect();

    let mut forb = BitSet::with_len(n);
    dag.descendants_of(&cn, &mut forb, &mut gws);
    forb.insert(t);

    let mut seen = BitSet::with_len(n);
    let mut o_set = Vec::new();
    for &c in &cn {
        for &p in dag.parents(c) {
            if !forb.contains(p) && !seen.contains(p) {
                seen.insert(p);
                o_set.push(p);
            }
        }
    }
    o_set.sort_unstable_by_key(|d| d.as_usize());
    o_set
}

fn default_assumptions() -> AssumptionSet {
    let mut assumptions = AssumptionSet::new();
    assumptions.push(crate::assumptions::causal_markov("backdoor.efficient"));
    assumptions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use causal_graph::Dag;

    #[test]
    fn o_set_selects_confounder() {
        // T <- Z -> Y, T -> Y (O = pa(Y) \ {T, Y} = {Z})
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(&*res.estimands[0].method, "backdoor.efficient");
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("optimal")));
    }

    #[test]
    fn o_set_prefers_outcome_parent_over_treatment_parent() {
        // W -> T -> Y, Q -> Y: Pa(T) = {W} is valid but anti-efficient; the
        // O-set is {Q} (the outcome's other parent).
        let mut g = Dag::with_variables(4);
        let w = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let q_node = DenseNodeId::from_raw(3);
        g.insert_directed(w, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(q_node, y).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(3)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("optimal")));
    }

    #[test]
    fn many_disconnected_covariates_no_longer_hit_result_limit() {
        // T -> Y plus 8 disconnected covariates: every one of the 2^8 = 256
        // covariate subsets is a valid backdoor set, which previously tripped
        // the 64-set ResultLimitExceeded error during full enumeration.
        let mut g = Dag::with_variables(10);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands[0].adjustment_set.is_empty());
    }

    #[test]
    fn fallback_search_when_o_set_forbidden() {
        // A -> T -> Y, A -> B -> Y plus disconnected covariates. O = {B}, but
        // B is forbidden, so the fallback must find the minimal valid set {A}
        // without erroring on the many valid supersets.
        let mut g = Dag::with_variables(10);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, y).unwrap();

        let mut id = EfficientBackdoorIdentifier::new();
        id.config.forbidden = Arc::from([VariableId::from_raw(3)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("min_cardinality")));
    }

    #[test]
    fn empty_when_no_backdoor_paths() {
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert!(res.estimands[0].adjustment_set.is_empty());
        assert_eq!(&*res.estimands[0].method, "backdoor.efficient");
    }
}
