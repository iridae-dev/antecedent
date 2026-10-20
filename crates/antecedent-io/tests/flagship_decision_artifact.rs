//! The mixed decision artifact, consumed without the facade crate.
//!
//! `rust_flagship_decision.bin` is the joint combination of a cluster margin, a
//! temporal effect, and a cost contrast under a declared cross-covariance. It
//! is written by the ignored `regenerate_rust_fixtures` test of
//! `crates/antecedent/tests/flagship_decision_fixture.rs`; a missing file fails
//! loudly. Nothing here builds the decision: the artifact replays from its own
//! stored numbers, and the hand values below are recomputed from them.
//!
//! Hand-derived truth for coefficients `(1, 1, -1)` and covariance `S`:
//! mean `= m0 + m1 - m2` and variance `= sum c_i c_j S_ij
//! = S00 + S11 + S22 + 2 (S01 - S02 - S12)`. The off-diagonal terms are not
//! zero, so the variance differs from the sum of the diagonal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_io::joint_law_propagation_artifact::{
    JointLawPropagationArtifactWire, PropagationRequestWire,
};
use antecedent_test_support::{conformance_dir, read_fixture_bytes};

const HINT: &str = "run `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent --test \
    flagship_decision_fixture -- --ignored regenerate_rust_fixtures` from the repo root";

fn fixture() -> Vec<u8> {
    read_fixture_bytes(&conformance_dir("cross_surface"), "rust_flagship_decision.bin", HINT)
}

fn retained_identity() -> String {
    let bytes = read_fixture_bytes(
        &conformance_dir("cross_surface"),
        "rust_flagship_decision.identity.json",
        HINT,
    );
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["identity"].as_str().unwrap().to_owned()
}

fn context() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|b| f64::from_bits(*b)).collect()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

/// Positive: the committed bytes replay under the retained identity and report
/// the hand-derived mean and variance of the stored declaration.
#[test]
fn committed_decision_replays_to_the_hand_derived_moments() {
    let identity = retained_identity();
    let wire = JointLawPropagationArtifactWire::decode(&fixture()).unwrap();
    let consumed = wire.consume(Some(&identity), &context()).unwrap();
    assert_eq!(consumed.identity, identity);
    assert_eq!(consumed.route, "linear_moments");
    assert_eq!(wire.covariance_id.as_deref(), Some("flagship-cross"));
    let PropagationRequestWire::LinearCombination { coefficients, means, covariance } =
        &wire.request
    else {
        panic!("the decision is a linear combination");
    };
    let (c, m, s) = (floats(coefficients), floats(means), floats(covariance));
    assert_eq!(c, [1.0, 1.0, -1.0]);
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert!(s[i * 3 + j] != 0.0, "cross term {i}{j} is declared");
        assert_eq!(s[i * 3 + j].to_bits(), s[j * 3 + i].to_bits());
    }
    let expected_variance = s[0] + s[4] + s[8] + 2.0 * (s[1] - s[2] - s[5]);
    let diagonal = s[0] + s[4] + s[8];
    assert!(near(consumed.value("variance").unwrap(), expected_variance));
    assert!((consumed.value("variance").unwrap() - diagonal).abs() > 1e-9 * diagonal);
    assert!(near(consumed.value("mean").unwrap(), m[0] + m[1] - m[2]));
    assert!(!consumed.is_envelope());
}

/// Negative: another retained identity or an edited byte refuses.
#[test]
fn a_different_identity_or_an_edited_byte_refuses() {
    let bytes = fixture();
    let wire = JointLawPropagationArtifactWire::decode(&bytes).unwrap();
    assert!(wire.consume(Some("not-the-retained-identity"), &context()).is_err());
    let mut edited = bytes;
    let last = edited.len() - 1;
    edited[last] ^= 0xff;
    let outcome = JointLawPropagationArtifactWire::decode(&edited)
        .and_then(|wire| wire.consume(Some(&retained_identity()), &context()));
    assert!(outcome.is_err());
}
