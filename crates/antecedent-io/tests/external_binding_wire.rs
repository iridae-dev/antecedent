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

// ---- Artifact fixtures for the provider contract, capabilities, verification and trust ----

use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, VerificationProbeWire,
};

fn provider_response(capabilities: &Value, request: &str, probes: &Value) -> ResponseWire {
    serde_json::from_value(json!({
        "provider": {
            "provider_id": "lab", "object_id": "risk-grid", "version_id": "v1",
            "snapshot_id": "snap-1", "request_id": request,
            "meaning": "interventional_predictive", "capabilities": capabilities
        },
        "graph_id": "graph:binary", "quantities": [risk(0), risk(1)], "values": [0.25, 0.75],
        "evidence_ids": [], "assumption_ids": [],
        "attestor": "lab", "probes": probes, "uncertainty_method": null, "point_support": null
    }))
    .unwrap()
}

fn bind(capabilities: &Value, request: &str, probes: &Value) -> ExternalClaimArtifact {
    bind_response(&bernoulli_contract(), &provider_response(capabilities, request, probes), "c")
        .unwrap()
}

fn reseal(
    artifact: &ExternalClaimArtifact,
    edit: impl FnOnce(&mut ExternalClaimIdentity),
) -> Result<Vec<u8>, antecedent_io::error::IoError> {
    let mut meta = artifact.metadata().clone();
    edit(&mut meta.identity);
    ExternalClaimArtifact::new(meta, artifact.values().to_vec())?.to_bytes("claim")
}

fn reload(bytes: &[u8], expected: &ExternalClaimIdentity) -> bool {
    ExternalClaimArtifact::from_bytes(bytes, expected).is_ok()
}

/// F2 frozen boundary: a fresh reader retains the law kind, operation
/// capabilities, exact request and provider identity.
#[test]
fn provider_contract_identity_is_retained_by_the_artifact() {
    let artifact = bind(&json!(["sample", "mean"]), "req-1", &Value::Null);
    let identity = artifact.metadata().identity.clone();
    assert_eq!(identity.provider_meaning, "interventional_predictive");
    assert_eq!(identity.capabilities, ["mean", "sample"], "sorted");
    assert_eq!(
        (identity.provider_id.as_str(), identity.object_id.as_str(), identity.request_id.as_str()),
        ("lab", "risk-grid", "req-1")
    );
    let bytes = artifact.to_bytes("claim").unwrap();
    assert!(reload(&bytes, &identity));
    // An edit that leaves the fingerprint stale is not a coherent identity.
    assert!(reseal(&artifact, |i| i.provider_meaning = "posterior_predictive".into()).is_err());
    // An edit resealed with a recomputed fingerprint differs from the retained identity.
    let resealed = reseal(&artifact, |i| {
        i.provider_meaning = "posterior_predictive".into();
        i.provider_fingerprint = i.compute_fingerprint();
    })
    .unwrap();
    assert!(!reload(&resealed, &identity));
}

/// F20 frozen boundary: the exact operation set and request fingerprint are retained.
#[test]
fn capability_set_and_request_fingerprint_are_retained() {
    let mean = bind(&json!(["mean"]), "req-1", &Value::Null);
    let both = bind(&json!(["sample", "mean"]), "req-1", &Value::Null);
    let swapped = bind(&json!(["mean", "sample"]), "req-1", &Value::Null);
    let other_request = bind(&json!(["mean"]), "req-2", &Value::Null);
    let fingerprint =
        |a: &ExternalClaimArtifact| a.metadata().identity.provider_fingerprint.clone();
    assert_eq!(fingerprint(&both), fingerprint(&swapped), "operation order is not identity");
    assert_ne!(fingerprint(&mean), fingerprint(&both), "one more operation is another object");
    assert_ne!(
        fingerprint(&mean),
        fingerprint(&other_request),
        "another request is another object"
    );
    let bytes = both.to_bytes("claim").unwrap();
    assert!(reload(&bytes, &both.metadata().identity));
    assert!(!reload(&bytes, &mean.metadata().identity));
    assert!(!reload(&bytes, &other_request.metadata().identity));
}

fn verified_probes() -> Value {
    json!(
        ["shape", "support", "moments", "known_truth"]
            .iter()
            .map(
                |kind| json!({"kind": kind, "observed": 0.25, "expected": 0.25, "tolerance": 1e-12})
            )
            .collect::<Vec<_>>()
    )
}

/// F22 frozen boundary: a fresh reader retains the checks, object kind, fingerprint
/// and the verified-extension trust scope.
#[test]
fn verification_checks_are_retained_and_cover_only_the_fingerprint() {
    let verified = bind(&json!(["mean"]), "req-1", &verified_probes());
    let identity = verified.metadata().identity.clone();
    let checks = identity.verification.clone().unwrap();
    let kinds: Vec<_> = checks.iter().map(|p| p.kind.as_str()).collect();
    assert_eq!(kinds, ["shape", "support", "moments", "known_truth"]);
    assert_eq!(identity.provider_meaning, "interventional_predictive");
    let bytes = verified.to_bytes("claim").unwrap();
    assert!(reload(&bytes, &identity));

    // A receipt cannot be edited into a failure, dropped, or attached to an attested claim.
    assert!(
        reseal(&verified, |i| {
            i.verification.as_mut().unwrap()[0] = VerificationProbeWire {
                kind: "shape".into(),
                observed: 2.0,
                expected: 1.0,
                tolerance: 0.0,
            };
        })
        .is_err()
    );
    assert!(reseal(&verified, |i| i.verification = None).is_err());
    let attested = bind(&json!(["mean"]), "req-1", &Value::Null);
    assert!(reseal(&attested, |i| i.verification = Some(checks.clone())).is_err());
    // The same checks under another request are another fingerprint, not a reused one.
    let other = bind(&json!(["mean"]), "req-2", &verified_probes());
    assert_ne!(other.metadata().identity.provider_fingerprint, identity.provider_fingerprint);
    assert!(!reload(&bytes, &other.metadata().identity));
}

/// F23 frozen boundary: the exact trust tier and provenance chain survive a fresh
/// reader without upgrade.
#[test]
fn trust_tier_and_lineage_survive_without_upgrade() {
    let attested = bind(&json!(["mean"]), "req-1", &Value::Null);
    let verified = bind(&json!(["mean"]), "req-1", &verified_probes());
    let wire = |a: &ExternalClaimArtifact| {
        serde_json::to_value(a.metadata()).unwrap()["identity"]["trust"].clone()
    };
    assert_eq!(wire(&attested), "externally_attested");
    assert_eq!(wire(&verified), "verified_extension");
    for artifact in [&attested, &verified] {
        assert!(!artifact.metadata().native_estimation);
        assert_ne!(wire(artifact), "native_licensed");
        let stages = artifact
            .metadata()
            .identity
            .provenance_chain()
            .unwrap()
            .stages_behind("claim")
            .unwrap();
        assert!(stages.contains(&antecedent_core::CompositionStage::ExternalProvider));
    }
    // Upgrading an attested claim to verified, even with fabricated passing probes
    // and a coherent artifact, differs from the retained identity.
    let attested_identity = attested.metadata().identity.clone();
    let upgraded = reseal(&attested, |i| {
        i.trust = antecedent_io::external_claim_artifact::ExternalClaimTrust::ExactRequestVerified;
        i.verification = Some(vec![VerificationProbeWire {
            kind: "known_truth".into(),
            observed: 0.25,
            expected: 0.25,
            tolerance: 0.0,
        }]);
    })
    .unwrap();
    assert!(!reload(&upgraded, &attested_identity));
}
