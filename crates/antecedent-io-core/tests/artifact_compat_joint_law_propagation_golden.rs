//! Golden fixture for the joint-law propagation artifact (`joint_law_propagation_v1`).
//!
//! The golden lives at `tests/fixtures/joint_law_propagation_v1.bin` and holds the earlier
//! layout, whose result is a list of named entries (format version 1). It is a fixed
//! fixture, never regenerated: the loader decodes it, maps the stored result into the typed
//! result, consumes it as a fresh consumer (the rule decides again from the stored
//! declaration and the result is replayed rather than trusted) and requires the identity a
//! producer derives today, so artifacts stored in the earlier layout keep replaying. The
//! writer below produces the current layout.
//!
//! Oracle: the contrast of effects with means 5 and 2, variances 4 and 9 and covariance 3
//! has mean 3 and variance 4 + 9 - 2 * 3 = 7.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading,
};
use antecedent_core::ExecutionContext;
use antecedent_io_core::joint_law_propagation_artifact::{
    JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE, JOINT_LAW_PROPAGATION_ARTIFACT_VERSION,
    JOINT_LAW_PROPAGATION_LEGACY_ARTIFACT_VERSION, JointLawPropagationArtifactError,
    JointLawPropagationArtifactWire, PropagationRequestWire, PropagationResultWire, float_bits,
};
use antecedent_test_support::read_fixture_bytes;

const GOLDEN: &str = "joint_law_propagation_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn summary(id: &str, evidence: &str) -> PropagationInput {
    PropagationInput {
        id: id.to_owned(),
        evidence_id: evidence.to_owned(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    }
}

fn contrast() -> JointLawPropagationArtifactWire {
    JointLawPropagationArtifactWire::seal(
        &PropagationRequest {
            operation: PropagationOperation::Contrast,
            inputs: vec![summary("effect_a", "snapshot_1"), summary("effect_b", "snapshot_1")],
            dependence: PropagationDependence::Covariance { id: "cov_ab".into() },
            reading: RequestedReading::Point,
        },
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(&[1.0, -1.0]),
            means: float_bits(&[5.0, 2.0]),
            covariance: float_bits(&[4.0, 3.0, 3.0, 9.0]),
        },
        &ctx(),
    )
    .expect("seal")
}

#[test]
fn the_joint_law_propagation_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, "the earlier-layout fixture is checked in");
    let fresh = contrast();
    let decoded = JointLawPropagationArtifactWire::decode(&bytes).expect("decodes");
    assert_eq!(decoded.version, JOINT_LAW_PROPAGATION_LEGACY_ARTIFACT_VERSION);
    assert_eq!(
        decoded.result,
        PropagationResultWire::Moments { mean: 3.0_f64.to_bits(), variance: 7.0_f64.to_bits() }
    );
    assert_eq!(decoded.result, fresh.result);
    assert_eq!(decoded.identity, fresh.identity);
    // The earlier layout is read-only: it is not re-written under its old version.
    assert_eq!(
        decoded.export(),
        Err(JointLawPropagationArtifactError::UnsupportedVersion {
            version: JOINT_LAW_PROPAGATION_LEGACY_ARTIFACT_VERSION
        })
    );
    let consumed = decoded
        .consume(Some(&fresh.identity), &ctx())
        .expect("a fresh consumer replays the stored propagation");
    assert_eq!(consumed.identity, fresh.identity);
    assert_eq!(consumed.value("mean"), Some(3.0));
    assert!((consumed.value("variance").expect("variance") - 7.0).abs() < 1e-12);
    assert!(!consumed.is_envelope());

    let mut newer = decoded.clone();
    newer.version = JOINT_LAW_PROPAGATION_ARTIFACT_VERSION + 1;
    let newer_bytes = antecedent_io_core::to_cbor(&newer).expect("encode");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&newer_bytes),
        Err(JointLawPropagationArtifactError::UnsupportedVersion {
            version: JOINT_LAW_PROPAGATION_ARTIFACT_VERSION + 1
        })
    );
    let mut feature = decoded;
    feature.required_features = vec!["future_feature".into()];
    let feature_bytes = antecedent_io_core::to_cbor(&feature).expect("encode");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&feature_bytes),
        Err(JointLawPropagationArtifactError::UnsupportedSemantics("required features"))
    );
    assert_eq!(JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE, "joint_law_propagation_v1");
}

#[test]
fn the_current_layout_round_trips_and_names_the_typed_result() {
    let fresh = contrast();
    assert_eq!(fresh.version, JOINT_LAW_PROPAGATION_ARTIFACT_VERSION);
    let decoded = JointLawPropagationArtifactWire::decode(&fresh.export().expect("export"))
        .expect("decodes");
    assert_eq!(decoded, fresh);
    let consumed = decoded.consume(Some(&fresh.identity), &ctx()).expect("replays");
    assert_eq!(consumed.value("mean"), Some(3.0));
    assert_eq!(consumed.n_rows(), None);
    assert_eq!(consumed.verdict(), None);
}
