//! The scaffolding every graph-search identifier shares.
//!
//! The back-door, efficient back-door, front-door and instrumental-variable searches differ in
//! which sets they test, not in how they resolve the query's treatment and outcome, assemble
//! the algorithm's assumption records, or turn a dense node set into the variables an estimand
//! carries. That common preamble lives here once.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, AverageEffectQuery, Intervention, Value, VariableId};
use antecedent_graph::{Dag, DenseNodeId};

use crate::backdoor::{PreparedIdentificationGraph, dense_to_var};
use crate::error::IdentificationError;

/// One identification of an average-effect query on a prepared DAG.
pub(crate) struct IdentificationRun<'p> {
    prepared: &'p PreparedIdentificationGraph,
    ate: &'p AverageEffectQuery,
    algorithm: &'static str,
    /// Treatment node.
    pub(crate) t: DenseNodeId,
    /// Outcome node.
    pub(crate) y: DenseNodeId,
}

impl<'p> IdentificationRun<'p> {
    /// Resolve the query's treatment and outcome on the prepared graph.
    ///
    /// # Errors
    ///
    /// Unknown treatment or outcome variable.
    pub(crate) fn open(
        prepared: &'p PreparedIdentificationGraph,
        ate: &'p AverageEffectQuery,
        algorithm: &'static str,
    ) -> Result<Self, IdentificationError> {
        let t = prepared.node_map().dense(ate.treatment)?;
        let y = prepared.node_map().dense(ate.outcome)?;
        Ok(Self { prepared, ate, algorithm, t, y })
    }

    /// The prepared DAG.
    pub(crate) fn dag(&self) -> &'p Dag {
        self.prepared.dag()
    }

    /// The query being identified.
    pub(crate) fn ate(&self) -> &'p AverageEffectQuery {
        self.ate
    }

    /// The algorithm's default Markov assumption followed by the caller-declared assumptions.
    pub(crate) fn base_assumptions(&self) -> AssumptionSet {
        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov(self.algorithm));
        for record in &self.prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }
        assumptions
    }

    /// Variables carried by the dense nodes `z`.
    ///
    /// # Errors
    ///
    /// A node that carries no identification variable.
    pub(crate) fn variables(
        &self,
        z: &[DenseNodeId],
    ) -> Result<Vec<VariableId>, IdentificationError> {
        z.iter().map(|d| dense_to_var(*d, self.dag())).collect()
    }

    /// The active and control levels when both sides are hard `Set` interventions.
    ///
    /// # Errors
    ///
    /// Any other intervention form; `message` names the algorithm's requirement.
    pub(crate) fn hard_set_levels(
        &self,
        message: &'static str,
    ) -> Result<(Value, Value), IdentificationError> {
        match (&self.ate.active, &self.ate.control) {
            (Intervention::Set { value: active, .. }, Intervention::Set { value: control, .. }) => {
                Ok((active.clone(), control.clone()))
            }
            _ => Err(IdentificationError::UnsupportedQuery { message }),
        }
    }
}
