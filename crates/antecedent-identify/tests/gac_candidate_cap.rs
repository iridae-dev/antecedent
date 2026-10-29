//! Overflow pins for the GAC candidate cap. These live off the calibration
//! surface so a test-only 16-vs-40 pin cannot owe a coverage remasurement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, DiagnosticKind, IdentificationStatus, Intervention, ResponseFunctional,
    ResponseQuery, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, Pag};
use antecedent_identify::{
    GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
    JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE, MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};

fn n(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn joint(t1: VariableId, t2: VariableId, y: VariableId) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([
            Intervention::set(t1, Value::f64(1.0)),
            Intervention::set(t2, Value::f64(1.0)),
        ]),
    })
}

/// A declared ceiling skips the search for a smaller joint set, not the decision: one test on
/// the maximal candidate set already certified it, so it is returned, flagged as the result
/// of a bounded search.
#[test]
fn joint_candidate_cap_keeps_the_certified_set_and_is_execution_not_scientific() {
    let mut admg = Admg::with_variables(20);
    let t1 = n(0);
    let t2 = n(1);
    let y = n(2);
    admg.insert_directed(t1, y).unwrap();
    admg.insert_directed(t2, y).unwrap();
    for i in 3..20 {
        let z = n(i);
        admg.insert_directed(z, t1).unwrap();
        admg.insert_directed(z, t2).unwrap();
        admg.insert_directed(z, y).unwrap();
    }
    let query = joint(VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig { max_candidates: 16, ..Default::default() },
    }
    .identify_joint_admg_response(&admg, &query)
    .unwrap();
    assert_eq!(
        id.status,
        IdentificationStatus::NonparametricallyIdentified,
        "the ceiling does not withdraw the certified set: {:?}",
        id.derivation
    );
    assert_eq!(id.estimands[0].adjustment_set.len(), 17);
    assert!(
        id.diagnostics.iter().any(|d| {
            d.code.as_ref() == JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE
                && d.kind == DiagnosticKind::Execution
        }),
        "cap must be an execution diagnostic: {:?}",
        id.diagnostics
    );
    assert!(
        !id.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Scientific),
        "cap must not be stamped scientific: {:?}",
        id.diagnostics
    );
}

/// The joint search's budget is the configured `max_examinations`: three separation tests
/// cannot reach a smaller set than all seventeen common causes, so the result is the maximal
/// set one test already certified, flagged as a bounded search the envelope counts as truncated.
#[test]
fn joint_search_budget_comes_from_the_config() {
    let mut admg = Admg::with_variables(20);
    let t1 = n(0);
    let t2 = n(1);
    let y = n(2);
    admg.insert_directed(t1, y).unwrap();
    admg.insert_directed(t2, y).unwrap();
    for i in 3..20 {
        let z = n(i);
        admg.insert_directed(z, t1).unwrap();
        admg.insert_directed(z, t2).unwrap();
        admg.insert_directed(z, y).unwrap();
    }
    let query = joint(VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig { max_examinations: 3, ..Default::default() },
    }
    .identify_joint_admg_response(&admg, &query)
    .unwrap();
    assert_eq!(id.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(id.performance.candidates_examined, 3);
    assert!(
        id.diagnostics.iter().any(|d| {
            d.code.as_ref() == JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE
                && d.kind == DiagnosticKind::Execution
        }),
        "{:?}",
        id.diagnostics
    );
    assert!(!id.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Scientific));
    assert!(antecedent_identify::search_truncated(&id));
}

/// `T <-> Y` with thirty measured common causes: the bidirected edge is an open path no
/// adjustment set can block, so adjustment cannot identify the effect. One m-separation test
/// on the ancestral candidates proves it; the old subset search would have enumerated 2^30
/// sets with no work budget at all. The result is a proof, so nothing is truncated.
#[test]
fn mag_without_an_adjustment_set_is_refuted_by_one_test() {
    let n_causes = 30u32;
    let mut mag = Pag::with_variables(2 + n_causes);
    mag.insert_bidirected(n(0), n(1)).unwrap();
    for i in 0..n_causes {
        mag.insert_directed(n(2 + i), n(0)).unwrap();
        mag.insert_directed(n(2 + i), n(1)).unwrap();
    }
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = GeneralizedAdjustmentIdentifier::new().identify_pag_envelope(&mag, &q).unwrap();
    assert_eq!(env.cases.len(), 1);
    let case = &env.cases[0].result;
    assert_eq!(case.status, IdentificationStatus::NotIdentified);
    assert_eq!(case.performance.candidates_examined, 1);
    assert_eq!(env.truncated_completions, 0);
    assert!(case.diagnostics.is_empty(), "{:?}", case.diagnostics);
}

/// `Z -> T`, `Z -> Y`, `T -> Y`, `R -> T`: `R` is nonadjacent to `Y` and points into `T`, so
/// `T -> Y` is visible. The ancestral candidates `{Z, R}` separate `T` from `Y`. A budget of
/// one test is spent before the search for smaller sets starts, so the effect is identified
/// with the ancestral set and the unfinished search is an Execution diagnostic; with the
/// default budget the smallest set `{Z}` comes back and nothing is reported.
#[test]
fn mag_search_budget_keeps_the_certified_ancestral_set() {
    let mut mag = Pag::with_variables(4);
    mag.insert_directed(n(0), n(1)).unwrap();
    mag.insert_directed(n(0), n(2)).unwrap();
    mag.insert_directed(n(1), n(2)).unwrap();
    mag.insert_directed(n(3), n(1)).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let bounded = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig { max_examinations: 1, ..Default::default() },
    };
    let env = bounded.identify_pag_envelope(&mag, &q).unwrap();
    let case = &env.cases[0].result;
    assert_eq!(case.status, IdentificationStatus::NonparametricallyIdentified);
    let mut adjustment: Vec<_> = case.estimands[0].adjustment_set.iter().map(|v| v.raw()).collect();
    adjustment.sort_unstable();
    assert_eq!(adjustment, vec![0, 3]);
    assert!(case.diagnostics.iter().any(|d| {
        d.code.as_ref() == MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE && d.kind == DiagnosticKind::Execution
    }));

    let env = GeneralizedAdjustmentIdentifier::new().identify_pag_envelope(&mag, &q).unwrap();
    let case = &env.cases[0].result;
    let adjustment: Vec<_> = case.estimands[0].adjustment_set.iter().map(|v| v.raw()).collect();
    assert_eq!(adjustment, vec![0]);
    assert!(case.diagnostics.is_empty());
}

/// A PAG is a class of MAGs, not a DAG: tails and arrowheads fix ancestry, not the absence of a
/// latent confounder. In the complete directed triangle z -> t, z -> y, t -> y every pair is
/// adjacent, so t -> y is not visible (no node outside y's neighbourhood points into t), the
/// class contains the latent-confounded t <-> y world, and no adjustment set identifies the
/// effect. Adding r -> t with r not adjacent to y makes the edge visible and {z} identifies.
#[test]
fn a_complete_directed_triangle_refuses_and_the_visible_edge_version_identifies() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let (z, t, y, r) = (n(0), n(1), n(2), n(3));
    let mut triangle = Pag::with_variables(3);
    triangle.insert_directed(z, t).unwrap();
    triangle.insert_directed(z, y).unwrap();
    triangle.insert_directed(t, y).unwrap();
    let refused =
        GeneralizedAdjustmentIdentifier::new().identify_pag_envelope(&triangle, &query).unwrap();
    assert_eq!(refused.status, IdentificationStatus::NotIdentified);
    assert!(refused.identified_weight.0 <= 0.0);
    assert!(
        refused.cases[0].result.derivation.steps.iter().any(|step| step.detail.contains("invisible edge")),
        "the refusal names the invisible edge"
    );

    let mut visible = Pag::with_variables(4);
    visible.insert_directed(z, t).unwrap();
    visible.insert_directed(z, y).unwrap();
    visible.insert_directed(t, y).unwrap();
    visible.insert_directed(r, t).unwrap();
    let found =
        GeneralizedAdjustmentIdentifier::new().identify_pag_envelope(&visible, &query).unwrap();
    assert_eq!(found.status, IdentificationStatus::NonparametricallyIdentified);
    let set = &found.cases[0].result.estimands[0].adjustment_set;
    assert!(set.contains(&VariableId::from_raw(0)), "{set:?}");
}
