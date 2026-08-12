//! Identification of response functionals through pairwise backdoor adjustment.
//!
//! The response estimand remains function-valued; the synthetic unit contrasts built in the
//! expression arena are identification witnesses for the required treatment/outcome pairs, not
//! a claim that a continuous response is itself a binary ATE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, ObservationAssumption, ObservationSpec,
    ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::Dag;

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Backdoor identifier for scalar/vector continuous-response functionals.
#[derive(Clone, Debug, Default)]
pub struct ResponseIdentifier {
    /// Pairwise adjustment-set identifier.
    pub backdoor: BackdoorIdentifier,
}

impl ResponseIdentifier {
    /// Construct with default pairwise adjustment search.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a DAG with declared assumptions.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        self.backdoor.prepare_with_assumptions(graph, assumptions)
    }

    /// Identify every treatment/outcome pair required by a response functional.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::Response(response) = query else {
            return Err(IdentificationError::unsupported(
                "ResponseIdentifier only supports Response queries",
            ));
        };
        response
            .validate()
            .map_err(|_| IdentificationError::unsupported("invalid continuous-response query"))?;
        require_observation_claim(response)?;

        let pairs = response_pairs(&response.functional)?;
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(pairs.len());
        let mut derivation = DerivationTrace::default();
        let mut performance = IdentificationPerformanceRecord::default();
        let mut assumptions = prepared.declared_assumptions().clone();
        append_observation_assumptions(response, &mut assumptions);

        for (treatment, outcome) in pairs {
            let witness = AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0)
                .with_target_population(response.target_population.clone());
            let witness_query = CausalQuery::AverageEffect(witness.clone());
            let result = self.backdoor.identify(prepared, &witness_query, workspace)?;
            performance.candidates_examined = performance
                .candidates_examined
                .saturating_add(result.performance.candidates_examined);
            performance.sets_returned =
                performance.sets_returned.saturating_add(result.performance.sets_returned);
            if result.status != IdentificationStatus::NonparametricallyIdentified {
                derivation.push(
                    "response.backdoor",
                    format!("pair ({treatment},{outcome}) was not identified"),
                );
                return Ok(IdentificationResult::not_identified(
                    query.clone(),
                    derivation,
                    assumptions,
                    performance,
                ));
            }
            let Some(first) = result.estimands.first() else {
                return Ok(IdentificationResult::not_identified(
                    query.clone(),
                    derivation,
                    assumptions,
                    performance,
                ));
            };
            let functional = arena.backdoor_ate(
                treatment,
                outcome,
                first.adjustment_set.as_ref(),
                Value::f64(1.0),
                Value::f64(0.0),
            );
            estimands.push(IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::clone(&first.adjustment_set),
                functional,
            ));
            derivation.push(
                "response.backdoor",
                format!(
                    "identified response pair ({treatment},{outcome}) with adjustment set {:?}",
                    first.adjustment_set
                ),
            );
        }

        Ok(IdentificationResult::identified(
            query.clone(),
            estimands,
            arena,
            derivation,
            assumptions,
            performance,
        ))
    }
}

fn response_pairs(
    functional: &ResponseFunctional,
) -> Result<Vec<(VariableId, VariableId)>, IdentificationError> {
    let pairs = match functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            vec![(treatment.variable, *outcome)]
        }
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            vec![(*treatment, *outcome)]
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            treatments.iter().flat_map(|t| outcomes.iter().map(move |y| (*t, *y))).collect()
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => interventions
            .iter()
            .filter_map(|intervention| intervention.primary_variable().map(|t| (t, *outcome)))
            .collect(),
    };
    if pairs.is_empty() {
        return Err(IdentificationError::unsupported(
            "response functional has no identifiable treatment/outcome pair",
        ));
    }
    Ok(pairs)
}

fn require_observation_claim(response: &ResponseQuery) -> Result<(), IdentificationError> {
    if !matches!(response.observation, ObservationSpec::Complete)
        && response.observation_assumptions.is_empty()
    {
        return Err(IdentificationError::unsupported(
            "an incomplete observation process requires an explicit ObservationAssumption",
        ));
    }
    Ok(())
}

fn append_observation_assumptions(response: &ResponseQuery, assumptions: &mut AssumptionSet) {
    for claim in response.observation_assumptions.iter() {
        let (id, description) = match claim {
            ObservationAssumption::IndependentGiven(vars) => (
                "observation.independent_given",
                format!("observation/censoring independent given {vars:?}"),
            ),
            ObservationAssumption::OutcomeIndependentGiven(vars) => (
                "observation.outcome_independent_given",
                format!("observation independent of latent outcome given {vars:?}"),
            ),
            ObservationAssumption::Structural(model) => {
                ("observation.structural", format!("structural observation model {model}"))
            }
        };
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom { id: id.into(), description: description.into() },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Untestable,
        });
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ContinuousDomain, GridSpec};
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;

    #[test]
    fn identifies_curve_pair_and_preserves_response_query() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Linspace { start: 0.0, end: 1.0, points: 5 },
            ),
        });
        let identifier = ResponseIdentifier::new();
        let prepared = identifier.prepare_with_assumptions(&dag, AssumptionSet::new()).unwrap();
        let result = identifier
            .identify(
                &prepared,
                &CausalQuery::Response(response),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(matches!(result.query, CausalQuery::Response(_)));
    }
}
