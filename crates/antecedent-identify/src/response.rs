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
    #[allow(clippy::too_many_lines)]
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
        if response.temporal.is_some() {
            return Err(IdentificationError::unsupported(
                "static ResponseIdentifier does not identify temporal response queries; use temporal backdoor identification once per horizon",
            ));
        }
        require_observation_claim(response)?;

        if matches!(response.functional, ResponseFunctional::InterventionResponse { .. })
            && response.functional.treatment_ids().len() > 1
        {
            let mut result = crate::GeneralizedAdjustmentIdentifier::new()
                .identify_joint_dag_response(prepared.dag(), response)?;
            for record in &prepared.declared_assumptions().entries {
                if !result.required_assumptions.entries.contains(record) {
                    result.required_assumptions.push(record.clone());
                }
            }
            return Ok(result);
        }

        let pairs = response_pairs(&response.functional)?;
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(pairs.len());
        let mut derivation = DerivationTrace::default();
        let mut performance = IdentificationPerformanceRecord::default();
        let mut assumptions = prepared.declared_assumptions().clone();
        append_observation_assumptions(response, &mut assumptions);
        let mut diagnostics = Vec::new();

        for (treatment, outcome) in pairs {
            let witness = AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0)
                .with_target_population(response.target_population.clone());
            let witness_query = CausalQuery::AverageEffect(witness.clone());
            let result = self.backdoor.identify(prepared, &witness_query, workspace)?;
            for record in &result.required_assumptions.entries {
                if !assumptions.entries.contains(record) {
                    assumptions.push(record.clone());
                }
            }
            performance.candidates_examined = performance
                .candidates_examined
                .saturating_add(result.performance.candidates_examined);
            performance.sets_returned =
                performance.sets_returned.saturating_add(result.performance.sets_returned);
            if result.status != IdentificationStatus::NonparametricallyIdentified {
                // Back-door failed for this pair: general ID decides it. A response naming one
                // pair is identified as itself; a Jacobian or directional derivative names many,
                // and each pair gets its own ID run whose functional joins the shared arena.
                let general = if matches!(
                    response.functional,
                    ResponseFunctional::InterventionResponse { .. }
                        | ResponseFunctional::MeanCurve { .. }
                ) {
                    crate::response_id::identify_dag_via_id(prepared.dag(), response)?
                } else {
                    crate::response_id::identify_dag_pair_via_id(
                        prepared.dag(),
                        treatment,
                        outcome,
                    )?
                };
                if general.status != IdentificationStatus::NonparametricallyIdentified
                    || general.estimands.is_empty()
                {
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
                derivation.push(
                    "identify.response.general_id",
                    format!(
                        "pair ({treatment},{outcome}) identified by Shpitser–Pearl ID after back-door failed"
                    ),
                );
                derivation.steps.extend(general.derivation.steps.iter().cloned());
                for record in &general.required_assumptions.entries {
                    if !assumptions.entries.contains(record) {
                        assumptions.push(record.clone());
                    }
                }
                performance.candidates_examined = performance
                    .candidates_examined
                    .saturating_add(general.performance.candidates_examined);
                performance.sets_returned =
                    performance.sets_returned.saturating_add(general.performance.sets_returned);
                diagnostics.extend(general.diagnostics.iter().cloned());
                for estimand in &general.estimands {
                    let mut copy = estimand.clone();
                    copy.functional = arena.import(&general.arena, estimand.functional);
                    estimands.push(copy);
                }
                continue;
            }
            let Some(first) = result.estimands.first() else {
                return Ok(IdentificationResult::not_identified(
                    query.clone(),
                    derivation,
                    assumptions,
                    performance,
                ));
            };
            let functional = match &response.functional {
                ResponseFunctional::InterventionResponse { interventions, .. } => {
                    let level = interventions.iter().find_map(|iv| match iv {
                        antecedent_core::Intervention::Set { variable, value }
                            if *variable == treatment =>
                        {
                            Some(value.clone())
                        }
                        _ => None,
                    });
                    match level {
                        Some(level) => arena.backdoor_mean(
                            treatment,
                            outcome,
                            first.adjustment_set.as_ref(),
                            level,
                        ),
                        None => arena.backdoor_ate(
                            treatment,
                            outcome,
                            first.adjustment_set.as_ref(),
                            Value::f64(1.0),
                            Value::f64(0.0),
                        ),
                    }
                }
                _ => arena.backdoor_ate(
                    treatment,
                    outcome,
                    first.adjustment_set.as_ref(),
                    Value::f64(1.0),
                    Value::f64(0.0),
                ),
            };
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

        let mut identified = IdentificationResult::identified(
            query.clone(),
            estimands,
            arena,
            derivation,
            assumptions,
            performance,
        );
        identified.diagnostics = diagnostics;
        Ok(identified)
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

/// Append one [`AssumptionRecord`] per declared [`ObservationAssumption`] of
/// `response`.
///
/// The single owner of the observation-assumption id and description table:
/// every path that identifies a response under an incomplete observation
/// process — static or temporal — records the same ids and text.
pub fn append_observation_assumptions(response: &ResponseQuery, assumptions: &mut AssumptionSet) {
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
        assert!(result.required_assumptions.entries.iter().any(|record| {
            record.assumption == Assumption::CausalMarkov
                && record.scope == AssumptionScope::Identification
        }));
    }

    /// `Y1 -> T -> Y2`. For the pair (T, Y1) the outcome is a parent of the treatment, so
    /// no adjustment set exists and general ID decides it; the pair (T, Y2) is back-door
    /// identified with the empty set. Both pairs must appear, in order, in one result.
    #[test]
    fn jacobian_keeps_every_pair_when_one_needs_general_id() {
        let mut dag = Dag::with_variables(3);
        let (t, y1, y2) =
            (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let response = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([y1, y2]),
            treatments: Arc::from([t]),
            at: Arc::from([0.5]),
            scale: antecedent_core::DerivativeScale::Identity,
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
        assert_eq!(result.estimands.len(), 2, "one estimand per (treatment, outcome) pair");
        assert_eq!(
            result.estimands[0].method_kind().unwrap(),
            antecedent_expr::EstimandMethod::GeneralId
        );
        assert_eq!(
            result.estimands[1].method_kind().unwrap(),
            antecedent_expr::EstimandMethod::BackdoorAdjustment
        );
        assert!(result.estimands[1].adjustment_set.is_empty());
        // The general-ID functional lives in the shared arena and is about the first pair's
        // outcome, Y1, not the second's.
        let antecedent_expr::ExprNode::Contrast { left, .. } =
            result.arena.node(result.estimands[0].functional)
        else {
            panic!("a pair's identification witness is a contrast");
        };
        let antecedent_expr::ExprNode::Expectation { function, .. } = result.arena.node(*left)
        else {
            panic!("each side of the contrast is a mean");
        };
        assert_eq!(function.variable(), y1);
        // ID's assumptions are carried into the merged result.
        assert!(result.required_assumptions.entries.iter().any(|r| {
            r.assumption == Assumption::CausalMarkov && r.scope == AssumptionScope::Identification
        }));
    }

    #[test]
    fn static_identifier_refuses_temporal_response_attachment() {
        let dag = Dag::with_variables(2);
        let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            antecedent_core::TemporalResponseSpec::new(
                [1u32],
                antecedent_core::TemporalPolicy::pulse(0),
                None,
            )
            .unwrap(),
        );
        let identifier = ResponseIdentifier::new();
        let prepared = identifier.prepare_with_assumptions(&dag, AssumptionSet::new()).unwrap();
        let error = identifier
            .identify(
                &prepared,
                &CausalQuery::Response(response),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("temporal response"));
    }
}
