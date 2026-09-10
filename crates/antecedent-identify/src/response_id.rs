//! General response identification: adjustment first, then Shpitser–Pearl ID.
//!
//! This is complete ID of `P(Y | do(A))` on each MAG-as-ADMG (or DAG-as-ADMG).
//! It is not PAG-native ID/IDC — circle marks are completed, then ID runs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, CausalQuery, IdentificationStatus, Intervention, ResponseQuery, Value,
};
use antecedent_graph::{Cpdag, Dag, Pag};

use crate::envelope::{GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass};
use crate::error::IdentificationError;
use crate::generalized::{
    GeneralizedAdjustmentIdentifier, identify_on_mag_completion, mag_to_admg, pag_var_to_dense,
};
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::result::IdentificationResult;

/// Binary ATE witness for a single-treatment response.
pub(crate) fn response_ate_witness(
    query: &ResponseQuery,
) -> Result<AverageEffectQuery, IdentificationError> {
    let treatments = query.functional.treatment_ids();
    if treatments.len() != 1 {
        return Err(IdentificationError::unsupported(
            "general response ID on a MAG/DAG currently requires one treatment target",
        ));
    }
    let (treatment, outcome) = query.functional.primary_pair().ok_or_else(|| {
        IdentificationError::unsupported("response functional has no treatment/outcome pair")
    })?;
    Ok(AverageEffectQuery::binary_ate(treatment, outcome))
}

/// Identify a MAG completion: generalized adjustment, then ADMG ID.
pub(crate) fn identify_mag_response(
    mag: &Pag,
    query: &ResponseQuery,
    max_candidates: usize,
) -> Result<IdentificationResult, IdentificationError> {
    let witness = response_ate_witness(query)?;
    let t = witness.treatment;
    let y = witness.outcome;
    let t_d = pag_var_to_dense(mag, t)?;
    let y_d = pag_var_to_dense(mag, y)?;
    let (active, control) = set_levels(&witness)?;
    let mut result = identify_on_mag_completion(
        mag,
        t,
        y,
        t_d,
        y_d,
        active.clone(),
        control.clone(),
        max_candidates,
    )?;
    if !is_identified(&result) {
        result = identify_admg_response(mag, &witness, query)?;
    }
    result.query = CausalQuery::Response(query.clone());
    Ok(result)
}

/// Identify a DAG response: backdoor first (caller), then ID on the DAG-as-ADMG.
pub(crate) fn identify_dag_via_id(
    dag: &Dag,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let prepared = IdIdentifier::new().prepare_dag(dag)?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = IdIdentifier::new().identify_ate(&prepared, query, &mut workspace)?;
    result.derivation.push(
        "identify.response.general_id",
        "back-door search failed; Shpitser–Pearl ID on the DAG-as-ADMG",
    );
    Ok(result)
}

fn identify_admg_response(
    mag: &Pag,
    witness: &AverageEffectQuery,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let Some(admg) = mag_to_admg(mag) else {
        return Ok(crate::generalized::not_identified(
            CausalQuery::Response(query.clone()),
            "completion is not a directed/bidirected MAG; general ID was not attempted",
        ));
    };
    let prepared = IdIdentifier::new().prepare(&admg)?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = IdIdentifier::new().identify_ate(&prepared, witness, &mut workspace)?;
    result.derivation.push(
        "identify.response.general_id",
        "generalized adjustment failed; Shpitser–Pearl ID on the MAG-as-ADMG",
    );
    result.query = CausalQuery::Response(query.clone());
    Ok(result)
}

fn set_levels(query: &AverageEffectQuery) -> Result<(Value, Value), IdentificationError> {
    match (&query.active, &query.control) {
        (Intervention::Set { value: active, .. }, Intervention::Set { value: control, .. }) => {
            Ok((active.clone(), control.clone()))
        }
        _ => {
            Err(IdentificationError::unsupported("general response ID requires Set interventions"))
        }
    }
}

fn is_identified(result: &IdentificationResult) -> bool {
    matches!(
        result.status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    ) && !result.estimands.is_empty()
}

/// Identify a single-treatment PAG response by adjustment, then general ID.
///
/// # Errors
///
/// Invalid query or graph errors.
pub fn identify_pag_response_general(
    pag: &Pag,
    query: &ResponseQuery,
) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
    let id = GeneralizedAdjustmentIdentifier::new();
    if !pag_has_circles(pag) {
        let result = identify_mag_response(pag, query, id.config.max_candidates)?;
        return Ok(IdentificationEnvelope::from_cases(vec![GraphIdentificationCase {
            graph: pag.clone(),
            result,
            weight: ProbabilityMass(1.0),
        }]));
    }
    id.pag_envelope_with(pag, |mag| identify_mag_response(mag, query, id.config.max_candidates))
}

fn pag_has_circles(pag: &Pag) -> bool {
    for i in 0..pag.node_count() {
        let a = antecedent_graph::DenseNodeId::from_raw(i as u32);
        for (_, at_a, at_b) in pag.neighbors(a) {
            if matches!(at_a, antecedent_graph::Endpoint::Circle)
                || matches!(at_b, antecedent_graph::Endpoint::Circle)
            {
                return true;
            }
        }
    }
    false
}

/// Identify a single-treatment CPDAG response by back-door, then general ID.
///
/// # Errors
///
/// Invalid query or graph errors.
pub fn identify_cpdag_response_general(
    cpdag: &Cpdag,
    query: &ResponseQuery,
) -> Result<IdentificationEnvelope<Dag>, IdentificationError> {
    let witness = response_ate_witness(query)?;
    let cq = CausalQuery::AverageEffect(witness.clone());
    let backdoor = crate::BackdoorIdentifier::new();
    let mut workspace = IdentificationWorkspace::default();
    let id = GeneralizedAdjustmentIdentifier::new();
    id.cpdag_envelope_with(cpdag, |dag| {
        let prepared = backdoor.prepare(dag)?;
        let mut result = backdoor.identify(&prepared, &cq, &mut workspace)?;
        if !is_identified(&result) {
            result = identify_dag_via_id(dag, &witness)?;
        }
        result.query = CausalQuery::Response(query.clone());
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value, VariableId,
    };
    use antecedent_graph::{DenseNodeId, Pag};

    use super::*;
    use crate::generalized::identify_on_mag_completion;

    fn front_door_mag() -> Pag {
        let mut pag = Pag::with_variables(3);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        pag.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        pag
    }

    fn response() -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        })
    }

    #[test]
    fn front_door_mag_adjustment_fails_id_identifies() {
        let mag = front_door_mag();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(2);
        let adj = identify_on_mag_completion(
            &mag,
            t,
            y,
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(2),
            Value::f64(1.0),
            Value::f64(0.0),
            16,
        )
        .unwrap();
        assert_eq!(adj.status, IdentificationStatus::NotIdentified, "{:?}", adj.derivation);
        let env = identify_pag_response_general(&mag, &response()).unwrap();
        assert!(
            env.identified_weight.0 > 0.0,
            "cases={} status={:?} first={:?}",
            env.cases.len(),
            env.status,
            env.cases.first().map(|c| (
                &c.result.status,
                &c.result.derivation,
                c.result.hedge.as_ref().map(|h| format!("{h:?}"))
            ))
        );
        assert!(
            env.cases.iter().any(|c| {
                c.result.status == IdentificationStatus::NonparametricallyIdentified
                    && c.result
                        .derivation
                        .steps
                        .iter()
                        .any(|s| s.rule.as_ref() == "identify.response.general_id")
            }),
            "front-door MAG must be identified by general ID, not adjustment"
        );
        assert_eq!(env.cases[0].result.estimands[0].method.as_ref(), "general.id");
    }
}
