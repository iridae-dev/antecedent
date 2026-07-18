//! Efficient backdoor adjustment-set selection.
//!
//! Selects the *optimal* adjustment set `O` (Henckel, Perković & Maathuis;
//! the construction `pinned baseline`'s efficient backdoor implements): with
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
use causal_graph::{BitSet, Dag, DenseNodeId, GraphWorkspace};

use crate::backdoor::{
    AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph, dense_to_var,
    is_backdoor_adjustment, remove_outgoing, var_to_dense,
};
use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
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
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Propagates prepare failures.
    pub fn prepare_with_assumptions(
        &self,
        graph: &causal_graph::Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        BackdoorIdentifier { config: self.config.clone() }
            .prepare_with_assumptions(graph, assumptions)
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
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "efficient backdoor only supports AverageEffect",
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
        let mut forbidden = BitSet::with_len(dag.node_count());
        for &v in self.config.forbidden.iter() {
            forbidden.insert(var_to_dense(v, dag)?);
        }
        forbidden.insert(t);
        forbidden.insert(y);

        let mut desc = BitSet::with_len(dag.node_count());
        dag.descendants_of(&[t], &mut desc, &mut workspace.graph);
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

        let mut examined = 0u64;

        // Primary rule: the O-set (optimal adjustment set, Henckel et al.).
        let o_set = optimal_adjustment_set(dag, t, y, &mut workspace.graph);
        examined += 1;
        if o_set.iter().all(|v| !forbidden.contains(*v))
            && is_backdoor_adjustment(&mutilated, t, y, &o_set, &mut workspace.dsep)?
        {
            return Self::finish(ate, query, prepared, dag, &[o_set], examined, "optimal (O-set)");
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
        'sizes: for size in 0..=m {
            let mut early_stop = false;
            let mut enum_err: Option<IdentificationError> = None;
            crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
                if enum_err.is_some() {
                    return true;
                }
                examined += 1;
                match is_backdoor_adjustment(&mutilated, t, y, z, &mut workspace.dsep) {
                    Ok(true) => {
                        valid.push(z.to_vec());
                        if valid.len() >= self.config.max_results {
                            early_stop = true;
                            return true;
                        }
                    }
                    Ok(false) => {}
                    Err(e) => {
                        enum_err = Some(e);
                        return true;
                    }
                }
                false
            });
            if let Some(e) = enum_err {
                return Err(e);
            }
            if early_stop {
                break 'sizes;
            }
            if !valid.is_empty() {
                break;
            }
        }

        if valid.is_empty() {
            let mut assumptions = default_assumptions();
            for record in &prepared.declared_assumptions().entries {
                assumptions.push(record.clone());
            }
            return Ok(IdentificationResult::not_identified(
                query,
                {
                    let mut d = DerivationTrace::default();
                    d.push("backdoor.efficient", "no valid adjustment set");
                    d
                },
                assumptions,
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
        Self::finish(ate, query, prepared, dag, &best, examined, "min_cardinality")
    }

    fn finish(
        ate: &AverageEffectQuery,
        query: CausalQuery,
        prepared: &PreparedIdentificationGraph,
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
        let mut assumptions = default_assumptions();
        for record in &prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }
        Ok(IdentificationResult::identified(
            query,
            vec![IdentifiedEstimand::backdoor("backdoor.efficient", Arc::from(vars), functional)],
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
        ))
    }
}

/// The O-set of Henckel, Perković & Maathuis: `pa(cn) \ (de(cn) ∪ {T})`,
/// where `cn = de(T) ∩ an(Y) \ {T}` are the nodes on proper causal paths from
/// `T` to `Y` (including `Y` itself). In a fully observed DAG this is the
/// valid backdoor set with the smallest asymptotic variance.
fn optimal_adjustment_set(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    gws: &mut GraphWorkspace,
) -> Vec<DenseNodeId> {
    let n = dag.node_count();
    let mut de_t = BitSet::with_len(n);
    dag.descendants_of(&[t], &mut de_t, gws);
    let mut an_y = BitSet::with_len(n);
    dag.ancestors_of(&[y], &mut an_y, gws);

    let cn: Vec<DenseNodeId> = (0..n)
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
        .filter(|&v| v != t && de_t.contains(v) && an_y.contains(v))
        .collect();

    let mut forb = BitSet::with_len(n);
    dag.descendants_of(&cn, &mut forb, gws);
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
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
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert!(res.estimands[0].adjustment_set.is_empty());
        assert_eq!(&*res.estimands[0].method, "backdoor.efficient");
    }
}
