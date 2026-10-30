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

/// The marginal contrast does not encode the distribution of treated or untreated units.
pub(crate) fn require_marginal_target(ate: &AverageEffectQuery) -> Result<(), IdentificationError> {
    if matches!(
        ate.target_population,
        antecedent_core::TargetPopulation::Treated | antecedent_core::TargetPopulation::Untreated
    ) {
        return Err(IdentificationError::UnsupportedQuery {
            message: "this marginal identifier does not identify treated or untreated population effects",
        });
    }
    Ok(())
}

#[cfg(test)]
mod target_population_tests {
    use super::*;
    use crate::{
        BackdoorIdentifier, EfficientBackdoorIdentifier, FrontDoorIdentifier, IdIdentifier,
        IdentificationWorkspace, InstrumentalVariableIdentifier,
    };
    use antecedent_core::{CausalQuery, TargetPopulation};
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    fn marginal_strategies_refuse_treated_and_untreated_targets() {
        // W ~ Bernoulli(.5), P(X=1 | W=1)=.8, P(X=1 | W=0)=.2, Y=X W.
        // Marginal ATE is E[W]=.5; ATT is E[W | X=1]=.8.
        let ate: f64 = 0.5;
        let treated: f64 = (0.5 * 0.8) / (0.5 * 0.8 + 0.5 * 0.2);
        assert!((ate - 0.5).abs() < 1e-12 && (treated - 0.8).abs() < 1e-12);
        let mut dag = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let id = IdIdentifier::new();
        let prepared_id = id.prepare_dag(&dag).unwrap();
        let front = FrontDoorIdentifier::new();
        let efficient = EfficientBackdoorIdentifier::new();
        let iv = InstrumentalVariableIdentifier::new();
        let backdoor = BackdoorIdentifier::new();
        let prepared = backdoor.prepare(&dag).unwrap();
        for population in [TargetPopulation::Treated, TargetPopulation::Untreated] {
            let ate =
                AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
                    .with_target_population(population);
            let query = CausalQuery::AverageEffect(ate.clone());
            let mut ws = IdentificationWorkspace::default();
            assert!(matches!(
                id.identify_ate(&prepared_id, &ate, &mut ws),
                Err(IdentificationError::UnsupportedQuery { .. })
            ));
            assert!(matches!(
                front.identify(&prepared, &query, &mut ws),
                Err(IdentificationError::UnsupportedQuery { .. })
            ));
            assert!(matches!(
                efficient.identify(&prepared, &query, &mut ws),
                Err(IdentificationError::UnsupportedQuery { .. })
            ));
            assert!(matches!(
                iv.identify(&prepared, &query, &mut ws),
                Err(IdentificationError::UnsupportedQuery { .. })
            ));
            assert!(backdoor.identify(&prepared, &query, &mut ws).is_ok());
        }
    }
}
