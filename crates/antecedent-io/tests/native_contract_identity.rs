//! Independent Python canonical JSON SHA256 contract identity oracles.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_io::external_binding_wire::native_contract_identity;
#[test]
fn original_ascii_and_utf16_contract_identity_bytes_are_preserved() {
    // hashlib.sha256(json.dumps(six_field_premises,sort_keys=True).encode()).
    // Default Python comma/colon spaces and ensure_ascii=True are part of the old identity.
    for (graph, expected) in [
        ("graph:test", "contract:07e48f4eb038b97256e2de44e9af247e678ffec7eef5a31d9cfd728cf43646ed"),
        ("graph:α", "contract:55e44c3af7e24037e229eed32686e09570088d96f908b18339d9a0b3c1e7e579"),
    ] {
        assert_eq!(
            native_contract_identity(graph, "nonparametrically_identified", None).unwrap(),
            expected
        );
    }
    assert_eq!(
        native_contract_identity(
            "graph:test",
            "nonparametrically_identified",
            Some(r#"{"graph_id":"other","identification":"nonparametrically_identified"}"#)
        )
        .unwrap(),
        "invalid:premise_substitution"
    );
}
