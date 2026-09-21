//! General response identification: adjustment first, then Shpitser–Pearl ID.
//!
//! On a DAG, ID runs on the DAG read as an ADMG. On a MAG a directed edge rules
//! out a latent common cause only when it is visible (Zhang 2008), so ID runs on
//! the ADMG that keeps every MAG edge and adds `A <-> B` beside each invisible
//! `A -> B`. Every DAG the MAG represents projects to an edge-subgraph of that
//! ADMG, so a functional derived there holds for all of them. This is sound and
//! strictly stronger than generalized adjustment, but not complete for MAGs.
//!
//! A PAG is handled by enumerating its valid MAG completions and identifying
//! each; it is not PAG-native ID. The complete algorithm for PAGs is IDP
//! (Jaber, Zhang & Bareinboim 2019), which this module does not implement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    IdentificationStatus, Intervention, ResponseQuery, Value,
};
use antecedent_graph::{Cpdag, Dag, Pag};

use crate::envelope::{GraphFeature, IdentificationEnvelope};
use crate::error::IdentificationError;
use crate::generalized::{
    GeneralizedAdjustmentIdentifier, identify_on_mag_completion, identify_on_mag_completion_mean,
    invisible_directed_edges, mag_dense_to_var, mag_to_confounded_admg, pag_var_to_dense,
};
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::result::IdentificationResult;

/// Diagnostic code on a MAG completion that neither generalized adjustment nor
/// visibility-aware ID identifies. The reduction is sound, not complete, so the
/// code marks a refusal rather than a proof of non-identifiability.
pub const MAG_ID_REFUSED_DIAGNOSTIC_CODE: &str = "identify.response.mag_id_refused";

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

/// Requested Set level for a single-treatment `InterventionResponse`.
pub(crate) fn intervention_response_set(
    query: &ResponseQuery,
) -> Option<(antecedent_core::VariableId, antecedent_core::VariableId, Value)> {
    match &query.functional {
        antecedent_core::ResponseFunctional::InterventionResponse { outcome, interventions }
            if interventions.len() == 1 =>
        {
            match &interventions[0] {
                Intervention::Set { variable, value } => Some((*variable, *outcome, value.clone())),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Identify a valid MAG completion: generalized adjustment, then visibility-aware ID.
pub(crate) fn identify_mag_response(
    mag: &Pag,
    query: &ResponseQuery,
    max_candidates: usize,
) -> Result<IdentificationResult, IdentificationError> {
    if let Some((t, y, level)) = intervention_response_set(query) {
        let t_d = pag_var_to_dense(mag, t)?;
        let y_d = pag_var_to_dense(mag, y)?;
        let mut result =
            identify_on_mag_completion_mean(mag, t, y, t_d, y_d, level, max_candidates, query)?;
        if !is_identified(&result) {
            result = identify_admg_response(mag, query)?;
        }
        return Ok(result);
    }
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
        result = identify_admg_response(mag, query)?;
    }
    result.query = CausalQuery::Response(query.clone());
    Ok(result)
}

/// Identify a DAG response: backdoor first (caller), then ID on the DAG-as-ADMG.
pub(crate) fn identify_dag_via_id(
    dag: &Dag,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let prepared = IdIdentifier::new().prepare_dag(dag)?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = IdIdentifier::new().identify_response(&prepared, query, &mut workspace)?;
    result.derivation.push(
        "identify.response.general_id",
        "back-door search failed; Shpitser–Pearl ID on the DAG-as-ADMG",
    );
    Ok(result)
}

fn identify_admg_response(
    mag: &Pag,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let Some(admg) = mag_to_confounded_admg(mag) else {
        return Ok(crate::generalized::not_identified(
            CausalQuery::Response(query.clone()),
            "completion is not a directed/bidirected MAG; general ID was not attempted",
        ));
    };
    let invisible = invisible_directed_edges(mag);
    let prepared = IdIdentifier::new().prepare(&admg)?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = IdIdentifier::new().identify_response(&prepared, query, &mut workspace)?;
    result.derivation.push(
        "identify.response.general_id",
        format!(
            "generalized adjustment failed; Shpitser–Pearl ID on the MAG with each of its {} \
             invisible directed edge(s) also read as latent-confounded",
            invisible.len()
        ),
    );
    if !is_identified(&result) {
        if !invisible.is_empty() {
            // The hedge lives in the confounded supergraph, which need not be a
            // member of the MAG's class: it is not a proof about the MAG.
            result.hedge = None;
            result.diagnostics.retain(|d| d.code.as_ref() != "identify.hedge");
        }
        let edges = invisible
            .iter()
            .map(|&(a, b)| {
                Ok(format!("{} -> {}", mag_dense_to_var(mag, a)?, mag_dense_to_var(mag, b)?))
            })
            .collect::<Result<Vec<_>, IdentificationError>>()?;
        result.diagnostics.push(Diagnostic::new(
            MAG_ID_REFUSED_DIAGNOSTIC_CODE,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "no adjustment set and no ID functional on this MAG once its invisible directed \
                 edges [{}] are allowed a latent common cause (Zhang 2008); the reduction is \
                 sound but not complete, so this is a refusal, not a proof of non-identifiability",
                edges.join(", ")
            ),
        ));
    }
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
/// Every route goes through the completion sampler, so a circle-free input is
/// checked to be a maximal ancestral graph before anything is identified on it;
/// a graph that is not (a directed or almost-directed cycle, or an inducing
/// path between non-adjacent nodes) yields no case and no identified mass.
/// Hold such a graph as an `Admg` instead.
///
/// # Errors
///
/// Invalid query or graph errors.
pub fn identify_pag_response_general(
    pag: &Pag,
    query: &ResponseQuery,
) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
    let id = GeneralizedAdjustmentIdentifier::new();
    let mut envelope = id.pag_envelope_with(pag, |mag| {
        identify_mag_response(mag, query, id.config.max_candidates)
    })?;
    downgrade_divergent_functionals(&mut envelope);
    Ok(envelope)
}

/// General-ID estimands carry no adjustment set, so two completions can share a
/// method tag while their functionals differ. A class-wide point claim needs
/// one functional; otherwise the class is only partially identified.
fn downgrade_divergent_functionals(envelope: &mut IdentificationEnvelope<Pag>) {
    let mut functionals = envelope
        .cases
        .iter()
        .filter(|case| is_identified(&case.result))
        .map(|case| case.result.arena.pretty(case.result.estimands[0].functional));
    let Some(first) = functionals.next() else { return };
    if functionals.all(|other| other == first) {
        return;
    }
    envelope.invariant = None;
    if envelope.status == IdentificationStatus::NonparametricallyIdentified {
        envelope.status = IdentificationStatus::PartiallyIdentified;
    }
    envelope.push_features([GraphFeature {
        kind: Arc::from("completion_functionals_differ"),
        detail: Arc::from(
            "identified MAG completions yield different functionals; no single estimand holds across the class",
        ),
    }]);
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
            result = identify_dag_via_id(dag, query)?;
        } else if let Some((t, y, level)) = intervention_response_set(query) {
            if let Some(first) = result.estimands.first() {
                let z = Arc::clone(&first.adjustment_set);
                let functional = result.arena.backdoor_mean(t, y, &z, level);
                result.estimands[0].functional = functional;
            }
        }
        result.query = CausalQuery::Response(query.clone());
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    // Envelope weights here are exact counts of unit-weight cases.
    #![allow(clippy::float_cmp)]
    use std::sync::Arc;

    use antecedent_core::{
        IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value, VariableId,
    };
    use antecedent_graph::{DenseNodeId, Pag};

    use super::*;
    use crate::generalized::identify_on_mag_completion;

    fn n(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    fn response() -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        })
    }

    // `T -> M -> Y` with `T <-> Y` is an ADMG, not an ancestral graph: `T` is an
    // ancestor of its spouse `Y`. No MAG has these marks, so nothing is identified.
    #[test]
    fn front_door_admg_stored_as_a_pag_is_not_a_mag_and_identifies_nothing() {
        let mut pag = Pag::with_variables(3);
        pag.insert_directed(n(0), n(1)).unwrap();
        pag.insert_directed(n(1), n(2)).unwrap();
        pag.insert_bidirected(n(0), n(2)).unwrap();
        assert!(!antecedent_graph::is_mag_completion(&pag));
        let env = identify_pag_response_general(&pag, &response()).unwrap();
        assert!(env.cases.is_empty());
        assert_eq!(env.status, IdentificationStatus::NotIdentified);
        assert!(env.identified_weight.0 == 0.0);
        assert!(
            env.critical_graph_features.iter().any(|f| {
                f.kind.as_ref() == "pag_completion_validation"
                    && f.detail.contains("rejected_non_ancestral=1")
            }),
            "{:?}",
            env.critical_graph_features
        );
    }

    // The valid MAG of the front-door DAG `T -> M -> Y`, `T <- L -> Y` keeps the
    // inducing path as an invisible `T -> Y`. The latent may confound it, so the
    // MAG alone does not identify the effect: front-door needs the ADMG.
    #[test]
    fn front_door_dag_has_an_invisible_edge_in_its_mag_and_is_refused() {
        let mut mag = Pag::with_variables(3);
        mag.insert_directed(n(0), n(1)).unwrap();
        mag.insert_directed(n(1), n(2)).unwrap();
        mag.insert_directed(n(0), n(2)).unwrap();
        assert!(antecedent_graph::is_mag_completion(&mag));
        let adj = identify_on_mag_completion(
            &mag,
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            n(0),
            n(2),
            Value::f64(1.0),
            Value::f64(0.0),
            16,
        )
        .unwrap();
        assert_eq!(adj.status, IdentificationStatus::NotIdentified, "{:?}", adj.derivation);
        let env = identify_pag_response_general(&mag, &response()).unwrap();
        assert_eq!(env.status, IdentificationStatus::NotIdentified);
        let result = &env.cases[0].result;
        assert!(result.hedge.is_none(), "a hedge in the confounded supergraph proves nothing here");
        let refusal = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == MAG_ID_REFUSED_DIAGNOSTIC_CODE)
            .expect("typed refusal");
        assert_eq!(refusal.kind, DiagnosticKind::Scientific);
        assert!(refusal.message.contains("V0 -> V2"), "{}", refusal.message);
    }

    // `T o-> Y` completes to `T -> Y` (invisible) and `T <-> Y`: the first is
    // refused, the second has no effect to identify, so the class is split.
    #[test]
    fn circle_arrow_pair_never_reports_the_class_identified() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_arrow(n(0), n(1)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        });
        let env = identify_pag_response_general(&pag, &query).unwrap();
        assert_eq!(env.cases.len(), 2);
        assert_eq!(env.status, IdentificationStatus::GraphDependent);
        assert!(env.unidentified_weight.0 == 1.0 && env.identified_weight.0 == 1.0);
    }

    #[test]
    fn invisible_direct_edge_mag_response_is_refused() {
        // L -> T, L -> Y (L latent) has the MAG `T -> Y`: no adjacent witness makes
        // the edge visible, so E[Y | do(T)] is not a function of P(T, Y).
        let mut mag = Pag::with_variables(2);
        mag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        });
        let env = identify_pag_response_general(&mag, &query).unwrap();
        assert_eq!(env.status, IdentificationStatus::NotIdentified, "{:?}", env.cases[0].result);
        assert!(env.identified_weight.0 == 0.0);
        assert!(
            env.cases[0]
                .result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == MAG_ID_REFUSED_DIAGNOSTIC_CODE)
        );
    }

    fn adjustment_mag() -> Pag {
        // R→T witnesses visibility of T→Y. Z is the backdoor (Z→T, Z→Y).
        let mut pag = Pag::with_variables(4);
        pag.insert_directed(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        pag
    }

    #[test]
    fn adjustment_mag_response_stages_single_arm_mean() {
        let mag = adjustment_mag();
        for level in [0.0, 1.0] {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(1),
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(level),
                )]),
            });
            let env = identify_pag_response_general(&mag, &query).unwrap();
            let result = &env.cases[0].result;
            assert!(matches!(result.query, CausalQuery::Response(_)));
            assert!(
                result.estimands[0].method.as_ref().starts_with("generalized.adjustment"),
                "method={} status={:?} derivation={:?}",
                result.estimands[0].method,
                result.status,
                result.derivation
            );
            let id = result.estimands[0].functional;
            assert!(
                !matches!(result.arena.node(id), antecedent_expr::ExprNode::Contrast { .. }),
                "adjustment MAG must not persist an ATE contrast as a Response"
            );
            let pretty = result.arena.pretty(id);
            assert!(pretty.contains("do("), "{pretty}");
            assert!(
                pretty.contains(&level.to_string()) || pretty.contains('0') || pretty.contains('1'),
                "{pretty}"
            );
            assert!(!pretty.contains('−'), "{pretty}");
        }
    }
}
