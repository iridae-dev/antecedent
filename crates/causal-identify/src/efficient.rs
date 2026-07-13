//! Efficient backdoor adjustment-set selection.
//!
//! Selects a single adjustment set among valid backdoors: prefer the parental
//! set `Pa(T)` when it satisfies the backdoor criterion; otherwise choose the
//! valid set of minimum cardinality, breaking ties by maximizing `|Z ∩ Pa(T)|`
//! then lexicographic variable ids.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, VariableId,
};
use causal_expr::CausalExprArena;
use causal_graph::{BitSet, DSeparationWorkspace, DenseNodeId, GraphWorkspace};

use crate::backdoor::{
    AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph, dense_to_var,
    is_backdoor_adjustment, remove_outgoing, var_to_dense,
};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand,
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
    pub fn prepare(&self, graph: &causal_graph::Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        BackdoorIdentifier { config: self.config.clone() }.prepare(graph)
    }

    /// Identify via efficient backdoor; at most one estimand is returned.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or enumeration limits.
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

        // Prefer parental adjustment set when valid.
        let parents: Vec<DenseNodeId> = dag
            .parents(t)
            .iter()
            .copied()
            .filter(|p| !forbidden.contains(*p))
            .collect();
        examined += 1;
        if is_backdoor_adjustment(&mutilated, t, y, &parents, &mut dsep_ws)? {
            return Self::finish(ate, query, dag, &[parents], examined, "parental");
        }

        let m = candidates.len();
        if m > 20 {
            return Err(IdentificationError::NotIdentified {
                message: "candidate set too large for exact efficient enumeration (>20)",
            });
        }

        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let total_masks = 1usize << m;
        for mask in 0..total_masks {
            examined += 1;
            let z: Vec<DenseNodeId> =
                (0..m).filter(|i| (mask & (1 << i)) != 0).map(|i| candidates[i]).collect();
            if is_backdoor_adjustment(&mutilated, t, y, &z, &mut dsep_ws)? {
                valid.push(z);
            }
            if valid.len() >= self.config.max_results {
                return Err(IdentificationError::ResultLimitExceeded {
                    limit: self.config.max_results,
                });
            }
        }

        if valid.is_empty() {
            return Ok(IdentificationResult {
                status: IdentificationStatus::NotIdentified,
                query,
                estimands: Vec::new(),
                arena: CausalExprArena::new(),
                derivation: {
                    let mut d = DerivationTrace::default();
                    d.push("backdoor.efficient", "no valid adjustment set");
                    d
                },
                required_assumptions: default_assumptions(),
                diagnostics: Vec::new(),
                performance: IdentificationPerformanceRecord {
                    candidates_examined: examined,
                    sets_returned: 0,
                },
            });
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
            a.len()
                .cmp(&b.len())
                .then_with(|| pb.cmp(&pa))
                .then_with(|| {
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
        derivation.push(
            "backdoor.efficient",
            format!("selected via {rule}; |Z|={}", vars.len()),
        );
        Ok(IdentificationResult {
            status: IdentificationStatus::NonparametricallyIdentified,
            query,
            estimands: vec![IdentifiedEstimand::backdoor(
                "backdoor.efficient",
                Arc::from(vars),
                functional,
            )],
            arena,
            derivation,
            required_assumptions: default_assumptions(),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: 1,
            },
        })
    }
}

fn default_assumptions() -> AssumptionSet {
    let mut assumptions = AssumptionSet::new();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::CausalMarkov,
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("backdoor.efficient"),
        },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    assumptions
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_graph::Dag;

    #[test]
    fn prefers_parental_when_valid() {
        // T <- Z -> Y, T -> Y  (Pa(T)={Z} is valid and efficient)
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
        assert!(
            res.derivation
                .steps
                .iter()
                .any(|s| s.detail.contains("parental"))
        );
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
