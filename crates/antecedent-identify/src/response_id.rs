//! General response identification: adjustment first, then Shpitser–Pearl ID.
//!
//! **What ID establishes.** [`IdIdentifier`] is the complete ID algorithm on the ADMG it is
//! given: every valid query ends in an identified functional or in a hedge that
//! [`crate::HedgeCertificate::verify`] accepts. That is the published theorem (Shpitser &
//! Pearl 2006; Huang & Valtorta 2006); for this implementation it is evidenced, not
//! proved: every single-treatment, single-outcome query on every ADMG with at most four
//! nodes, sampled five- and six-node joint queries, and the napkin family are checked
//! against exactly enumerated latent SCMs (`tests/id_completeness.rs`), with no query
//! ending in an error or an unverified refusal. A `NotIdentified` from ID on an ADMG is
//! therefore a scientific negative about that ADMG. An identified functional may keep a
//! variable outside the query free (the napkin's `z`); it holds at each of its supported
//! values and consumers must evaluate it there, never sum over it.
//!
//! **What the MAG and PAG routes establish.** On a DAG, ID runs on the DAG read as an
//! ADMG, and the statement above applies unchanged. On a MAG a directed edge rules out a
//! latent common cause only when it is visible (Zhang 2008), so ID runs on the ADMG that
//! keeps every MAG edge and adds `A <-> B` beside each invisible `A -> B`. Every DAG the
//! MAG represents projects to an edge-subgraph of that ADMG, so a functional derived there
//! holds for all of them: the route is sound. It is not complete. The confounded ADMG is a
//! worst case that need not belong to the MAG's class, so completeness of ID on that ADMG
//! says nothing about the MAG: a refusal here is not a proof of non-identifiability, the
//! hedge found in the supergraph is withheld when an invisible edge was confounded, and
//! the refusal is reported as [`MAG_ID_REFUSED_DIAGNOSTIC_CODE`]. The route is strictly
//! stronger than generalized adjustment. The subgraph property and the exactness of every
//! identified mean, at every value of any free variable, are checked by brute force over
//! all DAGs with at most four observed and two latent binary nodes and sampled
//! five-observed DAGs (`mag_id_bruteforce`), with no exception found; on at most four
//! observed nodes the gain over adjustment is confined to null effects, and non-null gains
//! start at five.
//!
//! A PAG is handled by enumerating its valid MAG completions and identifying each; it is
//! not PAG-native ID, and inherits the MAG route's incompleteness. Completions identified
//! through different functionals share no class-wide estimand, which the envelope reports
//! as partial identification. The complete algorithm for PAGs is IDP (Jaber, Zhang &
//! Bareinboim, "Causal Identification under Markov Equivalence: Completeness Results",
//! ICML 2019), which this module does not implement.
//!
//! **Errors.** With ID complete, an `Err` from these routes is never a statement about
//! identifiability. What remains is an unsupported or invalid query
//! ([`IdentificationError::UnsupportedQuery`], [`IdentificationError::InvalidQuery`]), a
//! variable or graph the input does not define ([`IdentificationError::UnknownVariable`],
//! [`IdentificationError::Graph`]), and a broken internal invariant of the ID recursion
//! ([`IdentificationError::InvariantViolated`], unreachable in every enumerated case). ID
//! has no search budget and takes no cancellation token; the bounded parts of these routes
//! are the adjustment search and the completion enumeration, which report an exhausted
//! budget as a bounded-search diagnostic on an `Ok` result, not as an error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    IdentificationStatus, Intervention, ResponseQuery, Value,
};
use antecedent_graph::{Cpdag, Dag, Pag};

use crate::envelope::IdentificationEnvelope;
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
    max_examinations: u64,
) -> Result<IdentificationResult, IdentificationError> {
    if let Some((t, y, level)) = intervention_response_set(query) {
        let t_d = pag_var_to_dense(mag, t)?;
        let y_d = pag_var_to_dense(mag, y)?;
        let mut result = identify_on_mag_completion_mean(
            mag,
            t,
            y,
            t_d,
            y_d,
            level,
            max_candidates,
            max_examinations,
            query,
        )?;
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
        max_examinations,
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

/// General ID of one treatment/outcome pair of a multi-pair response on the DAG-as-ADMG.
///
/// The binary contrast is the identification witness of that pair: whether `treatment`'s
/// effect on `outcome` is identified, and by which functional. A Jacobian or directional
/// derivative names many pairs, and each needs its own answer.
pub(crate) fn identify_dag_pair_via_id(
    dag: &Dag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<IdentificationResult, IdentificationError> {
    let identifier = IdIdentifier::new();
    let prepared = identifier.prepare_dag(dag)?;
    let mut workspace = IdentificationWorkspace::default();
    let mut result = identifier.identify_ate(
        &prepared,
        &AverageEffectQuery::binary_ate(treatment, outcome),
        &mut workspace,
    )?;
    result.derivation.push(
        "identify.response.general_id",
        format!(
            "back-door search failed for pair ({treatment},{outcome}); Shpitser–Pearl ID of its \
             binary contrast, an identification witness only"
        ),
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
    id.pag_envelope_with(pag, |mag| {
        identify_mag_response(mag, query, id.config.max_candidates, id.config.max_examinations)
    })
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
    #![cfg_attr(
        test,
        allow(
            clippy::float_cmp,
            reason = "test fixtures compare exact constants and index with small literals"
        )
    )]
    use std::sync::Arc;

    use antecedent_core::{
        IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value, VariableId,
    };
    use antecedent_graph::{DenseNodeId, Endpoint, Pag};

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
            1_000_000,
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

    #[test]
    fn mag_visibility_id_conformance_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/mag_visibility_id/expected.json"
        ))
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 8);
        for case in cases {
            let id = case["id"].as_str().unwrap();
            let nodes: Vec<&str> =
                case["nodes"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            let index = |name: &str| {
                u32::try_from(nodes.iter().position(|n| *n == name).expect("named node")).unwrap()
            };
            let mut pag = Pag::with_variables(u32::try_from(nodes.len()).unwrap());
            for edge in case["edges"].as_array().unwrap() {
                let edge = edge.as_str().unwrap();
                let (a, b, at_a, at_b) = ["<->", "o->", "o-o", "->"]
                    .iter()
                    .find_map(|sep| {
                        let (a, b) = edge.split_once(sep)?;
                        let (at_a, at_b) = match *sep {
                            "<->" => (Endpoint::Arrow, Endpoint::Arrow),
                            "o->" => (Endpoint::Circle, Endpoint::Arrow),
                            "o-o" => (Endpoint::Circle, Endpoint::Circle),
                            _ => (Endpoint::Tail, Endpoint::Arrow),
                        };
                        Some((n(index(a)), n(index(b)), at_a, at_b))
                    })
                    .unwrap_or_else(|| panic!("{id}: unparsed edge {edge}"));
                let mut marked = antecedent_graph::MarkedEdge::directed(a, b);
                marked.at_a = at_a;
                marked.at_b = at_b;
                pag.insert_marked(marked).unwrap();
            }
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(index("Y")),
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(index("T")),
                    Value::f64(1.0),
                )]),
            });
            let env = identify_pag_response_general(&pag, &query).unwrap();
            assert_eq!(
                env.identified_weight.0,
                case["identified_weight"].as_f64().unwrap(),
                "{id}"
            );
            assert_eq!(
                env.unidentified_weight.0,
                case["unidentified_weight"].as_f64().unwrap(),
                "{id}"
            );
            let expected = match case["status"].as_str().unwrap() {
                "identified" => IdentificationStatus::NonparametricallyIdentified,
                "graph_dependent" => IdentificationStatus::GraphDependent,
                "refused" | "not_a_mag" => IdentificationStatus::NotIdentified,
                other => panic!("{id}: unknown status {other}"),
            };
            assert_eq!(env.status, expected, "{id}");
            assert_eq!(env.cases.is_empty(), case["status"] == "not_a_mag", "{id}");
            if let Some(method) = case["method"].as_str() {
                assert_eq!(env.cases[0].result.estimands[0].method.as_ref(), method, "{id}");
            }
            if case["status"] == "refused" {
                assert!(
                    env.cases[0]
                        .result
                        .diagnostics
                        .iter()
                        .any(|d| d.code.as_ref() == MAG_ID_REFUSED_DIAGNOSTIC_CODE),
                    "{id}: a refusal is typed"
                );
            }
        }
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
