//! Identification of response functionals: pairwise back-door adjustment for single-treatment
//! functionals, joint adjustment for functionals over several treatments.
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
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per response functional, each assembling its own estimand from the shared prepared graph"
    )]
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

        if let Some(result) = self.identify_multi_treatment_joint(prepared, query, response)? {
            return Ok(result);
        }
        if matches!(response.functional, ResponseFunctional::InterventionResponse { .. })
            && response.functional.treatment_ids().len() > 1
        {
            // Joint adjustment owns its existing scientific search limits; it shares the
            // response caller's cancellation token and admission pool.
            let mut joint = crate::GeneralizedAdjustmentIdentifier::new();
            joint.config.context.clone_from(&self.backdoor.config.context);
            let mut result = joint.identify_joint_dag_response(prepared.dag(), response)?;
            for record in &prepared.declared_assumptions().entries {
                if !result.required_assumptions.entries.contains(record) {
                    result.required_assumptions.push(record.clone());
                }
            }
            return Ok(result);
        }

        if let ResponseFunctional::InterventionResponse { interventions, .. } = &response.functional
        {
            let treatments = response.functional.treatment_ids();
            let set_level = treatments.first().is_some_and(|treatment| {
                interventions.iter().any(|iv| {
                    matches!(iv, antecedent_core::Intervention::Set { variable, .. }
                        if variable == treatment)
                })
            });
            if !set_level {
                return Err(IdentificationError::unsupported(
                    "single-treatment intervention responses are identified only for hard Set interventions",
                ));
            }
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

impl ResponseIdentifier {
    /// Joint identification of a Jacobian or directional derivative over two or more treatments.
    ///
    /// The partial derivative of `E[Y | do(t_1..t_k)]` in one coordinate holds the other
    /// treatments fixed by intervention, so it is a property of the joint response surface and
    /// not of any single-treatment effect `E[Y | do(T_j)]` (which also carries the path through
    /// the other treatments). Each outcome is therefore decided by the joint back-door search
    /// over all treatments; if no common set exists the query is reported not identified rather
    /// than answered with pairwise sets. `None` when the functional is not such a derivative.
    fn identify_multi_treatment_joint(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        response: &ResponseQuery,
    ) -> Result<Option<IdentificationResult>, IdentificationError> {
        let mut joint = crate::GeneralizedAdjustmentIdentifier::new();
        joint.config.context.clone_from(&self.backdoor.config.context);
        multi_treatment_joint_route(
            query,
            response,
            prepared.declared_assumptions().clone(),
            |surface| joint.identify_joint_dag_response(prepared.dag(), surface),
        )
    }
}

/// The joint route for a Jacobian or directional derivative over two or more treatments, on any
/// graph class: `identify_surface` decides the joint response surface of one outcome over all
/// treatments (a joint-adjustment search on the caller's graph). `None` when the functional is
/// not such a derivative; not identified as soon as one outcome has no common joint set.
pub(crate) fn multi_treatment_joint_route(
    query: &CausalQuery,
    response: &ResponseQuery,
    declared: AssumptionSet,
    mut identify_surface: impl FnMut(
        &ResponseQuery,
    ) -> Result<IdentificationResult, IdentificationError>,
) -> Result<Option<IdentificationResult>, IdentificationError> {
    {
        let (outcomes, treatments) = match &response.functional {
            ResponseFunctional::Jacobian { outcomes, treatments, .. }
            | ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. } => {
                (outcomes, treatments)
            }
            _ => return Ok(None),
        };
        if treatments.len() < 2 {
            return Ok(None);
        }
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(outcomes.len());
        let mut derivation = DerivationTrace::default();
        let mut assumptions = declared;
        let mut performance = IdentificationPerformanceRecord::default();
        let mut diagnostics = Vec::new();
        for &outcome in outcomes.iter() {
            // The joint search certifies the response surface over all treatments for one
            // outcome; the surrogate only names that surface, its levels are not used.
            let mut surrogate = response.clone();
            surrogate.functional = ResponseFunctional::InterventionResponse {
                outcome,
                interventions: treatments
                    .iter()
                    .map(|&t| antecedent_core::Intervention::set(t, Value::f64(0.0)))
                    .collect(),
            };
            let mut result = identify_surface(&surrogate)?;
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
            diagnostics.append(&mut result.diagnostics);
            if result.status != IdentificationStatus::NonparametricallyIdentified
                || result.estimands.is_empty()
            {
                derivation.push(
                    "response.joint_adjustment",
                    format!(
                        "no common adjustment set identifies the joint response of {outcome} \
                         over the {} treatments; pairwise sets would identify single-treatment \
                         effects, not the partial derivative of the joint response",
                        treatments.len()
                    ),
                );
                let mut refused = IdentificationResult::not_identified(
                    query.clone(),
                    derivation,
                    assumptions,
                    performance,
                );
                refused.diagnostics = diagnostics;
                return Ok(Some(refused));
            }
            derivation.steps.extend(result.derivation.steps.iter().cloned());
            for estimand in &result.estimands {
                let mut copy = estimand.clone();
                copy.functional = arena.import(&result.arena, estimand.functional);
                estimands.push(copy);
            }
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
        Ok(Some(identified))
    }
}

pub(crate) fn response_pairs(
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
    fn shift_intervention_response_is_not_answered_as_unit_contrast() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([antecedent_core::Intervention::Shift {
                variable: VariableId::from_raw(0),
                delta: Value::f64(0.5),
            }]),
        });
        let identifier = ResponseIdentifier::new();
        let prepared = identifier.prepare_with_assumptions(&dag, AssumptionSet::new()).unwrap();
        let result = identifier.identify(
            &prepared,
            &CausalQuery::Response(response),
            &mut IdentificationWorkspace::default(),
        );
        assert!(matches!(result, Err(IdentificationError::UnsupportedQuery { .. })), "{result:?}");
    }

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

    fn jacobian_over(treatments: &[u32], outcomes: &[u32], directional: bool) -> CausalQuery {
        let tr: Arc<[VariableId]> = treatments.iter().map(|&t| VariableId::from_raw(t)).collect();
        let out: Arc<[VariableId]> = outcomes.iter().map(|&y| VariableId::from_raw(y)).collect();
        let at: Arc<[f64]> = treatments.iter().map(|_| 0.0).collect();
        let functional = if directional {
            ResponseFunctional::DirectionalDerivative {
                outcomes: out,
                treatments: tr,
                at,
                direction: treatments.iter().map(|_| 1.0).collect(),
            }
        } else {
            ResponseFunctional::Jacobian {
                outcomes: out,
                treatments: tr,
                at,
                scale: antecedent_core::DerivativeScale::Identity,
            }
        };
        CausalQuery::Response(ResponseQuery::new(functional))
    }

    fn identify_query(dag: &Dag, query: &CausalQuery) -> IdentificationResult {
        let identifier = ResponseIdentifier::new();
        let prepared = identifier.prepare_with_assumptions(dag, AssumptionSet::new()).unwrap();
        identifier.identify(&prepared, query, &mut IdentificationWorkspace::default()).unwrap()
    }

    /// `T1 -> T2 -> Y`. The single-treatment effect of T1 on Y is the total effect through T2,
    /// but the partial derivative of the joint response holds T2 fixed. The result must be the
    /// joint response surface (one estimand per outcome, a mean conditional on both
    /// treatments), never the pairwise total-effect contrasts.
    #[test]
    fn chain_jacobian_is_the_joint_response_not_the_total_effect() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        for directional in [false, true] {
            let result = identify_query(&dag, &jacobian_over(&[0, 1], &[2], directional));
            assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
            assert_eq!(result.estimands.len(), 1, "one joint estimand per outcome, not per pair");
            assert!(
                matches!(
                    result.arena.node(result.estimands[0].functional),
                    antecedent_expr::ExprNode::Expectation { .. }
                ),
                "the joint response is a mean over both treatments, not a one-treatment contrast"
            );
            assert!(
                result.derivation.steps.iter().any(|s| &*s.rule == "response.joint_adjustment")
            );
            assert!(!result.derivation.steps.iter().any(|s| &*s.rule == "response.backdoor"));
            assert!(matches!(result.query, CausalQuery::Response(_)));
        }
    }

    /// Two independent treatments and two outcomes: one joint estimand per outcome, in order.
    #[test]
    fn independent_treatments_get_one_joint_estimand_per_outcome() {
        let mut dag = Dag::with_variables(4);
        for (from, to) in [(0, 2), (1, 2), (0, 3), (1, 3)] {
            dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let result = identify_query(&dag, &jacobian_over(&[0, 1], &[2, 3], false));
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands.len(), 2);
        assert!(result.estimands.iter().all(|e| e.adjustment_set.is_empty()));
    }

    /// `T1 -> U -> T2`, `U -> Y`, `T2 -> Y`. Pairwise, T1 and T2 each have a back-door set
    /// ({} and {U}); jointly, U descends from T1 so it cannot be adjusted for and the T2-Y
    /// back-door path through U stays open. The partial derivative is therefore not identified
    /// by joint adjustment, and pairwise sets must not be reported in its place.
    #[test]
    fn pairwise_sets_do_not_identify_a_joint_derivative_they_cannot_certify_together() {
        let mut dag = Dag::with_variables(4);
        // 0 = T1, 1 = T2, 2 = U, 3 = Y
        for (from, to) in [(0, 2), (2, 1), (2, 3), (1, 3)] {
            dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for directional in [false, true] {
            let result = identify_query(&dag, &jacobian_over(&[0, 1], &[3], directional));
            assert_eq!(result.status, IdentificationStatus::NotIdentified);
            assert!(result.estimands.is_empty());
        }
        // A single-treatment derivative of T2 alone is still identified pairwise with {U}.
        let single = identify_query(&dag, &jacobian_over(&[1], &[3], false));
        assert_eq!(single.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(single.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
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
