//! Instrumental-variable identification for DAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery,
};
use causal_expr::CausalExprArena;
use causal_graph::{DSeparationWorkspace, Dag, DenseNodeId};

use crate::backdoor::{PreparedIdentificationGraph, dense_to_var, remove_outgoing, var_to_dense};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for instrument search.
#[derive(Clone, Debug)]
pub struct InstrumentSearchConfig {
    /// Maximum number of instruments to return.
    pub max_results: usize,
}

impl Default for InstrumentSearchConfig {
    fn default() -> Self {
        Self { max_results: 64 }
    }
}

/// Instrumental-variable identifier for static DAGs.
#[derive(Clone, Debug, Default)]
pub struct InstrumentalVariableIdentifier {
    /// Search configuration.
    pub config: InstrumentSearchConfig,
}

impl InstrumentalVariableIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph for IV identification.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::new(graph.clone()))
    }

    /// Identify an average-effect query via a valid instrument.
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
                message: "IV identification only supports AverageEffect",
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

        // Candidates: all nodes except T,Y, with parents of T checked first.
        let parents_of_t: Vec<DenseNodeId> =
            dag.parents(t).iter().copied().filter(|&p| p != y).collect();
        let mut candidates: Vec<DenseNodeId> = parents_of_t.clone();
        for i in 0..dag.node_count() {
            let v = DenseNodeId::from_raw(u32::try_from(i).expect("node id fits u32"));
            if v == t || v == y || candidates.contains(&v) {
                continue;
            }
            candidates.push(v);
        }

        let mut dsep_ws = DSeparationWorkspace::default();
        let mut valid: Vec<DenseNodeId> = Vec::new();
        let mut examined = 0u64;

        for &z in &candidates {
            examined += 1;
            if is_valid_instrument(dag, z, t, y, &mut dsep_ws)? {
                valid.push(z);
                if valid.len() >= self.config.max_results {
                    break;
                }
            }
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("iv"));

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "iv.criterion",
            "Z relevant to T given ∅ and d-separated from Y in G with T's out-edges cut",
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
        for &z in &valid {
            let z_var = dense_to_var(z, dag)?;
            assumptions.push(AssumptionRecord {
                assumption: Assumption::ExclusionRestriction { instrument: z_var },
                source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("iv") },
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
            let (active, control) = match (&ate.active, &ate.control) {
                (
                    causal_core::Intervention::Set { value: active, .. },
                    causal_core::Intervention::Set { value: control, .. },
                ) => (active.clone(), control.clone()),
                _ => {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: "IV ATE requires Set interventions",
                    });
                }
            };
            let functional = arena.iv_wald(ate.treatment, ate.outcome, &[z_var], active, control);
            estimands.push(IdentifiedEstimand::instrumental("iv", Arc::from([z_var]), functional));
            derivation.push("iv.instrument", format!("Z={}", z_var.raw()));
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

/// Whether `z` is a valid instrument for `t` -> `y`.
fn is_valid_instrument(
    dag: &Dag,
    z: DenseNodeId,
    t: DenseNodeId,
    y: DenseNodeId,
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    if z == t || z == y {
        return Ok(false);
    }

    // 1. Relevance: Z reaches T and is not d-separated from T given ∅.
    if !dag.reaches(z, t) {
        return Ok(false);
    }
    let independent_of_t = dag.is_d_separated(z, t, &[], ws).map_err(IdentificationError::from)?;
    if independent_of_t {
        return Ok(false);
    }

    // 2. Exclusion + no Z-Y confounding: with T's outgoing edges cut (so the
    // legitimate T -> Y channel is removed), Z must be d-separated from Y
    // given ∅. Conditioning on T directly would instead open the T-collider
    // between Z and any T-Y confounder, so we mutilate rather than condition.
    let t_mutilated = remove_outgoing(dag, t)?;
    t_mutilated.is_d_separated(z, y, &[], ws).map_err(IdentificationError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use causal_core::{AverageEffectQuery, VariableId};

    #[test]
    fn confounded_treatment_with_valid_instrument() {
        // Z -> T -> Y, U -> T, U -> Y (U unmeasured confounder).
        let mut g = Dag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(0)]));
        // The confounder U itself must never be reported as a valid instrument.
        assert!(!res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(3)]));
    }

    #[test]
    fn direct_edge_to_outcome_rejects_instrument() {
        // Z -> T -> Y, Z -> Y (direct edge violates exclusion), U -> T, U -> Y.
        let mut g = Dag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let res = id.identify(&prep, &q).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }
}
