//! Engineering checks for external-refusal projection.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::external_refusal_parts;
use antecedent_contract::ExternalRefusal;

#[test]
fn preserves_code_and_detail_and_uses_remedy_or_route_fallback() {
    let refusal = || ExternalRefusal::new("invalid_argument", "verify", "example.detail");
    assert_eq!(
        external_refusal_parts(refusal().with_remedy("repair it"), "fallback"),
        ("invalid_argument", "example.detail".to_owned(), "repair it".to_owned()),
    );
    for fallback in [
        "the callback result is not well formed",
        "the transport handoff is not well formed",
        "the bridge is not supported",
        "the interface contract is not well formed",
    ] {
        assert_eq!(
            external_refusal_parts(refusal(), fallback),
            ("invalid_argument", "example.detail".to_owned(), fallback.to_owned()),
        );
    }
}
