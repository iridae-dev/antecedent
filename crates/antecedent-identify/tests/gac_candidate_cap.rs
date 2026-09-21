//! Overflow pins for the GAC candidate cap. These live off the calibration
//! surface so a test-only 16-vs-40 pin cannot owe a coverage remasurement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DiagnosticKind, IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value,
    VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
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
