//! The exact law of a declared adaptive procedure against exact rational path enumeration
//! (`oracle_data/procedure_law_reference.py`, built on `procedure_exact.py`).
//!
//! The reference expands every path of assigned arm, delivered arm, outcome and retention for
//! every unit in order, with `Fraction` probabilities and Beta exceedance probabilities from a
//! binomial-tail identity, then applies the stopping rule and selection at each look; the product
//! merges states by observed counts. The cases cover each assignment rule (fixed, epsilon-greedy
//! with ties, Thompson, floored Thompson at extreme truths, UCB with three arms), delay, outcome
//! loss that depends on the outcome, exposure failure under both attributions, every stopping
//! rule, both final selections, three arms with ties, and the prior predictive.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::enumerate_law;
use serde_json::Value;
use support::{assert_atoms, kernel, never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_law_reference.json"))
        .expect("frozen reference")
}

#[test]
fn every_terminal_state_matches_the_exact_path_enumeration() {
    let reference = reference();
    let cases = reference["cases"].as_array().expect("cases");
    assert!(cases.len() >= 16);
    let mut states = 0;
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let spec = spec(&case["declaration"]);
        let law = enumerate_law(&spec, &kernel(&case["kernel"]), &never).expect("a law");
        assert_atoms(&law, &case["atoms"], 1e-12, name);
        states += law.atoms.len();
    }
    assert!(states > 1000);
}
