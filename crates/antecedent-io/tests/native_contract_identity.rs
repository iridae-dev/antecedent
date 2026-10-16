//! The contract identity is the canonical digest every host computes the same way.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_io::canonical_json::canonical_json_digest;
use antecedent_io::external_binding_wire::{CONTRACT_PREMISES_DOMAIN, native_contract_identity};

fn premises(graph: &str) -> String {
    format!(
        r#"{{"graph_id":"{graph}","identification":"nonparametrically_identified","accepted_meanings":["interventional_predictive"],"required_evidence_ids":[],"required_assumption_ids":[],"equivalences":[]}}"#
    )
}

#[test]
fn identity_is_the_canonical_digest_of_the_premises_under_the_contract_domain() {
    for graph in ["graph:test", "graph:α"] {
        let expected = format!(
            "contract:{}",
            canonical_json_digest(CONTRACT_PREMISES_DOMAIN, &premises(graph)).unwrap()
        );
        assert_eq!(
            native_contract_identity(graph, "nonparametrically_identified", None).unwrap(),
            expected
        );
    }
    // Pinned so a change of encoding or domain is deliberate: Python's
    // `ExternalSpec._content_id` computes the same value from the same premises.
    assert_eq!(
        native_contract_identity("graph:test", "nonparametrically_identified", None).unwrap(),
        "contract:1b7a8952623cafbc22f551eae06c7487778c917a0c0d625fac87c8593db2f8f8"
    );
}

#[test]
fn key_order_and_spacing_of_supplied_premises_do_not_change_the_identity() {
    let reordered = r#"{ "equivalences": [], "required_assumption_ids": [], "required_evidence_ids": [],
        "accepted_meanings": ["interventional_predictive"],
        "identification": "nonparametrically_identified", "graph_id": "graph:test" }"#;
    assert_eq!(
        native_contract_identity("graph:test", "nonparametrically_identified", Some(reordered))
            .unwrap(),
        native_contract_identity("graph:test", "nonparametrically_identified", None).unwrap()
    );
}

#[test]
fn a_substituted_graph_or_extra_field_is_not_a_contract_identity() {
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
