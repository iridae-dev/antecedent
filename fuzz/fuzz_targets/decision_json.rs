//! Fuzz the decision contract JSON declaration entry points (length cap, unknown fields,
//! stored identity ignored).
#![no_main]

use std::sync::OnceLock;

use antecedent_design::decision_artifact::{
    MAX_DECISION_ARTIFACT_BYTES, contract_from_json, contract_from_json_refusal, contract_to_json,
};
use antecedent_io::IoError;
use libfuzzer_sys::fuzz_target;

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

fn oversized_declaration() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        let mut json = String::from(r#"{"version":1,"pad":""#);
        json.extend(std::iter::repeat_n('x', MAX_DECISION_ARTIFACT_BYTES));
        json.push_str(r#""}"#);
        json
    })
    .as_str()
}

/// A stored identity is not the contract identity. An unknown field and a
/// declaration past the length cap are refusals.
fn stored_identity_is_ignored() {
    let forged = contract_from_json(&declaration("forged-identity")).expect("declaration");
    let other = contract_from_json(&declaration("other-forged")).expect("declaration");
    let canonical = forged.identity().expect("identity");
    assert_eq!(other.identity().expect("identity"), canonical);
    assert_ne!(canonical, "forged-identity");
    assert_ne!(canonical, "other-forged");
    let exported = contract_to_json(&forged).expect("export");
    assert!(exported.contains(&canonical));
    assert!(!exported.contains("forged-identity"));
    let round_trip = contract_from_json(&exported).expect("round trip");
    assert_eq!(round_trip.identity().expect("identity"), canonical);
    assert!(contract_from_json_refusal(&declaration("forged-identity")).is_ok());

    let unknown = declaration("ignored").replacen(r#""version":1"#, r#""version":1,"extra":1"#, 1);
    assert!(contract_from_json(&unknown).is_err());
    assert!(contract_from_json_refusal(&unknown).is_err());
    assert!(contract_from_json("{").is_err());
    assert!(contract_from_json_refusal("{").is_err());
    assert!(matches!(contract_from_json(oversized_declaration()), Err(IoError::TooLarge)));
    assert!(contract_from_json_refusal(oversized_declaration()).is_err());
}

fuzz_target!(|data: &[u8]| {
    stored_identity_is_ignored();
    let Ok(json) = std::str::from_utf8(data) else {
        return;
    };
    let parsed = contract_from_json(json);
    let refused = contract_from_json_refusal(json);
    assert_eq!(parsed.is_ok(), refused.is_ok());
    if let (Ok(parsed), Ok(refused)) = (parsed, refused) {
        assert_eq!(
            parsed.identity().expect("parsed identity"),
            refused.identity().expect("refusal identity")
        );
    }
});
