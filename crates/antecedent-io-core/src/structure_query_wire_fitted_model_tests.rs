//! The nested counterfactual simulation and the population do-contrast attribution round-trip
//! through the query wire, name their variables and interventions, and carry their own labels.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::Value;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn set(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

fn round_trip(query: &CausalQuery) -> CausalQueryWire {
    let wire = causal_query_to_wire(query).expect("encode");
    let json = serde_json::to_string(&wire).expect("serialize");
    let back: CausalQueryWire = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&causal_query_from_wire(&back).expect("decode"), query);
    back
}

fn nested() -> CausalQuery {
    CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::NestedCounterfactual(
        NestedCounterfactualTarget {
            outer: Arc::from([set(0, 0.0)]),
            inner: Arc::from([set(3, 1.0)]),
            frozen: Arc::from([v(1)]),
            outcome: v(2),
        },
    )))
}

fn contrast(max_units: Option<usize>) -> CausalQuery {
    CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::PopulationDoContrast(
        PopulationDoContrastTarget { parent: v(0), child: v(2), delta: 0.5, max_units, seed: 41 },
    )))
}

fn label(labels: &[(String, String)], key: &str) -> String {
    labels.iter().find(|(k, _)| k == key).map(|(_, value)| value.clone()).unwrap_or_default()
}

#[test]
fn the_nested_counterfactual_round_trips_and_names_both_worlds() {
    let wire = round_trip(&nested());
    let CausalQueryWire::Simulate(simulate) = &wire else {
        panic!("a simulation query has the simulate wire");
    };
    assert!(matches!(simulate.target, SimulationTargetWire::NestedCounterfactual { .. }));
    assert_eq!(simulate.variable_ids(), vec![1, 2]);
    assert_eq!(simulate.interventions().len(), 1, "the outer world's interventions");
    assert_eq!(simulate.all_interventions().len(), 2, "both worlds' interventions");
    let labels = crate::executed_functional_labels(&wire);
    assert_eq!(label(&labels, "query_kind"), "simulate");
    assert_eq!(label(&labels, "simulation"), "nested_counterfactual");
    let json = serde_json::to_value(&wire).expect("serialize");
    assert!(json.to_string().contains("\"nested_counterfactual\""));
}

#[test]
fn a_frozen_outcome_is_refused_when_the_wire_is_decoded() {
    let wire = causal_query_to_wire(&nested()).expect("encode");
    let CausalQueryWire::Simulate(mut simulate) = wire else {
        panic!("a simulation query has the simulate wire");
    };
    let SimulationTargetWire::NestedCounterfactual { frozen, .. } = &mut simulate.target else {
        panic!("the nested target");
    };
    *frozen = vec![2];
    assert!(simulate.to_domain().is_err());
}

#[test]
fn the_population_do_contrast_round_trips_with_and_without_a_row_cap() {
    for max_units in [None, Some(10)] {
        let wire = round_trip(&contrast(max_units));
        let CausalQueryWire::Attribute(attribute) = &wire else {
            panic!("an attribution query has the attribute wire");
        };
        assert_eq!(attribute.variable_ids(), vec![0, 2]);
        assert!(attribute.embedded().is_none());
        let labels = crate::executed_functional_labels(&wire);
        assert_eq!(label(&labels, "attribution"), "population_do_contrast");
        let json = serde_json::to_value(&wire).expect("serialize").to_string();
        assert_eq!(json.contains("max_units"), max_units.is_some());
    }
}

#[test]
fn a_self_contrast_is_refused_when_the_wire_is_decoded() {
    let wire = causal_query_to_wire(&contrast(None)).expect("encode");
    let CausalQueryWire::Attribute(mut attribute) = wire else {
        panic!("an attribution query has the attribute wire");
    };
    let AttributeTargetWire::PopulationDoContrast { child, .. } = &mut attribute.target else {
        panic!("the contrast target");
    };
    *child = 0;
    assert!(attribute.to_domain().is_err());
}
