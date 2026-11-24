//! The nested counterfactual simulation target: its encoding, its validation and the variables
//! it reads.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::*;
use crate::query::{ModelOperation, ModelOperationFault};
use crate::value::Value;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn set(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

fn digest<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("t");
    value.encode(&mut writer);
    writer.finish()
}

fn mediation_target() -> NestedCounterfactualTarget {
    NestedCounterfactualTarget {
        outer: Arc::from([set(0, 0.0)]),
        inner: Arc::from([set(0, 1.0)]),
        frozen: Arc::from([v(1)]),
        outcome: v(2),
    }
}

#[test]
fn the_nested_target_is_tagged_apart_from_every_other_simulation_and_from_its_operation() {
    let target = mediation_target();
    let nested = SimulateQuery::new(SimulationTarget::NestedCounterfactual(target.clone()));
    let mut expected = CanonicalWriter::new("t");
    expected.word(9);
    target.encode(&mut expected);
    assert_eq!(digest(&nested), expected.finish());
    // The same interventions as one simultaneous world are another question.
    let simultaneous = SimulateQuery::new(SimulationTarget::Counterfactual {
        interventions: target.outer.iter().chain(target.inner.iter()).cloned().collect(),
        outcomes: Arc::from([target.outcome]),
    });
    assert_ne!(digest(&nested), digest(&simultaneous));
    assert_ne!(digest(&nested), digest(&ModelOperation::NestedCounterfactual(target)));
    assert_eq!(nested.target.as_str(), "nested_counterfactual");
}

#[test]
fn every_field_of_the_nested_target_reaches_the_encoding() {
    let base = mediation_target();
    let edits: [&dyn Fn(&mut NestedCounterfactualTarget); 4] = [
        &|t| t.outer = Arc::from([set(0, 0.5)]),
        &|t| t.inner = Arc::from([set(0, 2.0)]),
        &|t| t.frozen = Arc::from([v(3)]),
        &|t| t.outcome = v(4),
    ];
    let wrap = |t: NestedCounterfactualTarget| {
        digest(&SimulateQuery::new(SimulationTarget::NestedCounterfactual(t)))
    };
    let base_digest = wrap(base.clone());
    for edit in edits {
        let mut edited = base.clone();
        edit(&mut edited);
        assert_ne!(wrap(edited), base_digest);
    }
}

#[test]
fn the_nested_query_refuses_a_frozen_outcome_or_inner_target_and_reads_its_endpoints() {
    let ok = SimulateQuery::new(SimulationTarget::NestedCounterfactual(mediation_target()));
    assert!(ok.validate().is_ok());
    assert_eq!(ok.endpoints(), (Some(v(0)), Some(v(2))));
    for frozen in [v(2), v(0)] {
        let refused = SimulateQuery::new(SimulationTarget::NestedCounterfactual(
            NestedCounterfactualTarget { frozen: Arc::from([frozen]), ..mediation_target() },
        ));
        assert_eq!(
            refused.validate(),
            Err(QueryError::InvalidModelOperation(ModelOperationFault::NestedFrozenConflict))
        );
    }
}
