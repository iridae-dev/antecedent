//! Fuzz the decision contract and result artifacts; the first 32 bytes seed the expected identity.
//!
//! A contract artifact loads only under the identity recomputed from its body.
#![no_main]

use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, contract_from_json,
};
use antecedent_design::decision_robust_artifact::RobustResultArtifact;
use antecedent_design::decision_structural_artifact::StructuralResultArtifact;
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn quantity(regime: &str) -> String {
    format!(
        concat!(
            r#"{{"version":1,"variable_id":"y","variable_name":"y","role":"outcome","#,
            r#""units":"units","population_id":"target","regime_id":"{regime}","#,
            r#""horizon":0,"functional_id":"mean","conditioning":[],"transform_id":"identity"}}"#
        ),
        regime = regime,
    )
}

fn declaration(stored_identity: &str) -> String {
    format!(
        concat!(
            r#"{{"version":1,"identity":"{identity}","actions":["#,
            r#"{{"id":"A","kind":"intervention","inputs":[{qa}],"utility":{{"input":0}}}},"#,
            r#"{{"id":"B","kind":"intervention","inputs":[{qb}],"utility":{{"input":0}}}}],"#,
            r#""utility_units":"units","criterion":"posterior_expected_utility","#,
            r#""constraints":[],"target_population":"target","horizon":0,"#,
            r#""structural_policy":"report_only"}}"#
        ),
        identity = stored_identity,
        qa = quantity("do(a=1)"),
        qb = quantity("do(a=0)"),
    )
}

/// The consumer's retained identity is the load key. A stored JSON identity and
/// a truncated container are not.
fn retained_identity_is_required() {
    let contract = contract_from_json(&declaration("forged-identity")).expect("declaration");
    let canonical = contract.identity().expect("identity");
    assert_ne!(canonical, "forged-identity");
    let artifact = DecisionContractArtifact::new(contract).expect("artifact");
    assert_eq!(artifact.identity(), canonical);
    let bytes = artifact.to_bytes("decision").expect("bytes");
    let loaded = DecisionContractArtifact::from_bytes(&bytes, &canonical).expect("load");
    assert_eq!(loaded.identity(), canonical);
    assert!(DecisionContractArtifact::from_bytes(&bytes, "forged-identity").is_err());
    assert!(DecisionContractArtifact::from_bytes(&bytes[..bytes.len() - 1], &canonical).is_err());
    assert!(DecisionContractArtifact::from_bytes(&[], &canonical).is_err());
    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x01;
    assert!(DecisionContractArtifact::from_bytes(&flipped, &canonical).is_err());
}

fuzz_target!(|data: &[u8]| {
    retained_identity_is_required();
    let (head, body) = data.split_at(data.len().min(32));
    let identity = hex(head);
    let zero = "0".repeat(64);
    if let Ok(artifact) = DecisionContractArtifact::from_bytes(body, &identity) {
        assert_eq!(artifact.identity(), identity);
    }
    let _ = DecisionResultArtifact::from_bytes(body, &identity, &zero);
    let _ = RobustResultArtifact::from_bytes(body, &identity, &[]);
    let _ = StructuralResultArtifact::from_bytes(body, &identity, &[]);
});
