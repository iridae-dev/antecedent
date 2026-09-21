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
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, GeneralizedAdjustmentConfig,
    GeneralizedAdjustmentIdentifier, JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
    MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
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

#[test]
fn joint_candidate_cap_is_execution_not_scientific() {
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
        IdentificationStatus::NotIdentified,
        "cap keeps NotIdentified: {:?}",
        id.derivation
    );
    assert!(
        id.diagnostics.iter().any(|d| {
            d.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE
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
/// cannot reach the only valid joint set (all seventeen common causes), so the result is an
/// undecided search that the envelope counts as truncated, never a proof.
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
    assert_eq!(id.status, IdentificationStatus::NotIdentified);
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
