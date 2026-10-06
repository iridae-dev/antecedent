//! The JSON binding wire the Python facade builds, exercised from Rust.
//!
//! Uses the same closed-form fixture as `python/tests/test_external_binding.py`
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
    assert_eq!(refusal.detail, "external_binding.coordinate.units");
    assert_eq!(refusal.offending.as_deref(), Some("coordinate[1]"));
    assert_eq!(refusal.expected.as_deref(), Some("mmHg"));
    assert_eq!(refusal.supplied.as_deref(), Some("kPa"));
}
