//! The JSON binding wire the Python facade builds, exercised from Rust.
//!
//! Uses the same closed-form fixture as `python/tests/test_external_response_binding.py`
//! so both surfaces assert the same values, trust, lineage and refusal fields.

use antecedent_io::external_binding_wire::{ContractWire, ResponseWire, bind_response};
use antecedent_io::external_claim_artifact::ExternalClaimTrust;
use serde_json::{Value, json};

fn quantity(dose: u32, units: &str) -> Value {
    json!({
        "version": 1, "variable_id": "y", "variable_name": "y", "role": "outcome",
        "units": units, "population_id": "target", "regime_id": format!("do(a={dose})"),
        "horizon": 0, "functional_id": "mean", "conditioning": [], "transform_id": "identity"
    })
}

fn contract() -> ContractWire {
    serde_json::from_value(json!({
        "graph_id": "graph:g", "identification": "nonparametrically_identified",
        "estimand": [quantity(0, "mmHg"), quantity(1, "mmHg"), quantity(2, "mmHg")],
        "accepted_meanings": ["interventional_predictive"],
        "required_evidence_ids": ["factor:z"], "required_assumption_ids": ["ignorability"],
        "equivalences": []
    }))
    .unwrap()
}

fn response(units: &str, probes: &Value) -> ResponseWire {
    serde_json::from_value(json!({
        "provider": {
            "provider_id": "lab", "object_id": "curve", "version_id": "v3",
            "snapshot_id": "snap-9", "request_id": "req-1",
            "meaning": "interventional_predictive", "capabilities": ["mean"]
        },
        "graph_id": "graph:g",
        "quantities": [quantity(0, "mmHg"), quantity(1, units), quantity(2, "mmHg")],
        // Closed form: E[Y | do(a)] = 1 + 2a.
        "values": [1.0, 3.0, 5.0],
        "evidence_ids": ["factor:z"], "assumption_ids": ["ignorability"],
        "attestor": "lab", "probes": probes, "uncertainty_method": null, "point_support": null
    }))
    .unwrap()
}

#[test]
fn attested_binding_matches_the_python_fixture() {
    let artifact = bind_response(&contract(), &response("mmHg", &Value::Null), "c").unwrap();
    assert_eq!(artifact.values(), [1.0, 3.0, 5.0]);
    let identity = &artifact.metadata().identity;
    assert_eq!(identity.trust, ExternalClaimTrust::ExternallyAttested);
    assert_eq!(identity.point_status, ["missing_evidence"; 3]);
    assert!(!artifact.metadata().native_estimation);
    let stages = identity.provenance_chain().unwrap().stages_behind("claim").unwrap();
    assert_eq!(stages.len(), 4);
    assert_eq!(artifact.provenance_label(), "external:lab/curve@v3#snap-9");
}

#[test]
fn verified_trust_uses_the_shared_provider_vocabulary() {
    let probes: Vec<_> = ["shape", "support", "moments", "known_truth"]
        .iter()
        .map(|kind| json!({"kind": kind, "observed": 1.0, "expected": 1.0, "tolerance": 0.0}))
        .collect();
    let artifact = bind_response(&contract(), &response("mmHg", &json!(probes)), "c").unwrap();
    assert_eq!(artifact.metadata().identity.trust, ExternalClaimTrust::ExactRequestVerified);
    let wire = serde_json::to_value(artifact.metadata()).unwrap();
    assert_eq!(wire["identity"]["trust"], "verified_extension");
}

#[test]
fn units_mismatch_refuses_with_the_same_fields_python_reads() {
    let refusal = bind_response(&contract(), &response("kPa", &Value::Null), "c").unwrap_err();
    assert_eq!(refusal.code, "quantity_semantics_mismatch");
    assert_eq!(refusal.stage, "bind");
    assert_eq!(refusal.detail, "external_response_binding.coordinate_units");
    assert_eq!(refusal.offending.as_deref(), Some("coordinate[1]"));
    assert_eq!(refusal.expected.as_deref(), Some("mmHg"));
    assert_eq!(refusal.supplied.as_deref(), Some("kPa"));
}

fn risk(dose: u32) -> Value {
    json!({
        "version": 1, "variable_id": "y", "variable_name": "y", "role": "outcome",
        "units": "probability", "population_id": "target", "regime_id": format!("do(a={dose})"),
        "horizon": 0, "functional_id": "risk", "conditioning": [], "transform_id": "identity"
    })
}

fn bernoulli_contract() -> ContractWire {
    serde_json::from_value(json!({
        "graph_id": "graph:binary", "identification": "nonparametrically_identified",
        "estimand": [risk(0), risk(1)],
        "accepted_meanings": ["interventional_predictive"],
        "required_evidence_ids": [], "required_assumption_ids": [], "equivalences": []
    }))
    .unwrap()
}

fn bernoulli_response(graph: &str, quantities: &Value, values: &Value) -> ResponseWire {
    serde_json::from_value(json!({
        "provider": {
            "provider_id": "lab", "object_id": "risk-grid", "version_id": "v1",
            "snapshot_id": "snap-1", "request_id": "req-1",
            "meaning": "interventional_predictive", "capabilities": ["mean"]
        },
        "graph_id": graph, "quantities": quantities, "values": values,
        "evidence_ids": [], "assumption_ids": [],
        "attestor": "lab", "probes": null, "uncertainty_method": null, "point_support": null
    }))
    .unwrap()
}

/// F3 frozen boundary: the identified binary do(A) query with supplied grid
/// P(Y=1|do(A=0)) = 1/4 and P(Y=1|do(A=1)) = 3/4 binds with a risk difference of 1/2.
#[test]
fn bernoulli_risk_difference_binds_under_exact_identity() {
    let response =
        bernoulli_response("graph:binary", &json!([risk(0), risk(1)]), &json!([0.25, 0.75]));
    let artifact = bind_response(&bernoulli_contract(), &response, "binary-contract").unwrap();
    let values = artifact.values();
    assert!((values[0] - 0.25).abs() < 1e-12);
    assert!((values[1] - 0.75).abs() < 1e-12);
    assert!(((values[1] - values[0]) - 0.5).abs() < 1e-12, "bound risk difference is 1/2");
    let identity = &artifact.metadata().identity;
    assert_eq!(identity.graph_id, "graph:binary");
    assert_eq!(identity.quantities.len(), 2);
    assert_eq!(identity.snapshot_id, "snap-1");
    assert_eq!(artifact.provenance_label(), "external:lab/risk-grid@v1#snap-1");
    assert!(!artifact.metadata().native_estimation);
}

/// F3 frozen boundary: observational P(Y|A), a wrong graph, a wrong population and a
/// missing grid coordinate all refuse before a claim exists.
#[test]
fn observational_wrong_graph_population_and_missing_coordinate_refuse() {
    let good = json!([risk(0), risk(1)]);
    let values = json!([0.25, 0.75]);

    let wrong_graph = bernoulli_response("graph:other", &good, &values);
    let refusal = bind_response(&bernoulli_contract(), &wrong_graph, "c").unwrap_err();
    assert_eq!(
        (refusal.code.as_str(), refusal.detail.as_str()),
        ("external_binding_mismatch", "external_response_binding.graph")
    );

    let mut population = good.clone();
    population[1]["population_id"] = json!("source");
    let refusal = bind_response(
        &bernoulli_contract(),
        &bernoulli_response("graph:binary", &population, &values),
        "c",
    )
    .unwrap_err();
    assert_eq!(refusal.code, "quantity_semantics_mismatch");
    assert_eq!(refusal.detail, "external_response_binding.coordinate_population");
    assert_eq!(refusal.offending.as_deref(), Some("coordinate[1]"));

    let mut observational = good.clone();
    observational[0]["regime_id"] = json!("observational");
    observational[0]["conditioning"] = json!([{"variable_id": "a", "value_id": "0"}]);
    let refusal = bind_response(
        &bernoulli_contract(),
        &bernoulli_response("graph:binary", &observational, &values),
        "c",
    )
    .unwrap_err();
    assert_eq!(refusal.code, "quantity_semantics_mismatch");
    assert!(refusal.remedy.as_deref().is_some_and(|r| r.contains("checked equivalence")));

    let missing = bernoulli_response("graph:binary", &json!([risk(0)]), &json!([0.25]));
    let refusal = bind_response(&bernoulli_contract(), &missing, "c").unwrap_err();
    assert_eq!(refusal.detail, "external_response_binding.dimension");
    assert_eq!(refusal.expected.as_deref(), Some("2"));
    assert_eq!(refusal.supplied.as_deref(), Some("1"));
}
