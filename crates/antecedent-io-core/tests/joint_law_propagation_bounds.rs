//! `joint_law_propagation_v1` is bare CBOR, so the reader bounds it before it
//! allocates: bytes before parsing, and every item and string right after the
//! typed decode, not only when the artifact is consumed.

use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading,
};
use antecedent_core::ExecutionContext;
use antecedent_io_core::joint_law_propagation_artifact::{
    JointLawPropagationArtifactError, JointLawPropagationArtifactWire,
    MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES, MAX_JOINT_LAW_PROPAGATION_SHARED_EVIDENCE,
    MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES, PropagationRequestWire, float_bits,
};
use antecedent_io_core::to_cbor;

fn sealed() -> JointLawPropagationArtifactWire {
    let input = |id: &str| PropagationInput {
        id: id.to_owned(),
        evidence_id: "snapshot_1".to_owned(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    };
    JointLawPropagationArtifactWire::seal(
        &PropagationRequest {
            operation: PropagationOperation::Contrast,
            inputs: vec![input("effect_a"), input("effect_b")],
            dependence: PropagationDependence::Covariance { id: "cov_ab".into() },
            reading: RequestedReading::Point,
        },
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(&[1.0, -1.0]),
            means: float_bits(&[5.0, 2.0]),
            covariance: float_bits(&[4.0, 3.0, 3.0, 9.0]),
        },
        &ExecutionContext::for_tests(11),
    )
    .expect("seal")
}

#[test]
fn an_oversized_artifact_is_refused_before_it_is_parsed() {
    let bytes = vec![0xff_u8; MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES + 1];
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&bytes).expect_err("too large"),
        JointLawPropagationArtifactError::LimitsExceeded("artifact bytes")
    );
}

#[test]
fn a_long_string_is_refused_at_decode_not_only_at_consume() {
    let wire = sealed();
    let ok = to_cbor(&wire).expect("encodes");
    JointLawPropagationArtifactWire::decode(&ok).expect("the sealed artifact decodes");

    let mut long_label = wire.clone();
    long_label.label = "x".repeat(MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES + 1);
    let bytes = to_cbor(&long_label).expect("encodes");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&bytes).expect_err("label too long"),
        JointLawPropagationArtifactError::LimitsExceeded("text")
    );

    let mut long_input = wire;
    long_input.inputs[0].input_id = "y".repeat(MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES + 1);
    let bytes = to_cbor(&long_input).expect("encodes");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&bytes).expect_err("input id too long"),
        JointLawPropagationArtifactError::LimitsExceeded("text")
    );
}

#[test]
fn shared_evidence_is_bounded() {
    let mut wire = sealed();
    wire.shared_evidence = vec!["e".to_owned(); MAX_JOINT_LAW_PROPAGATION_SHARED_EVIDENCE + 1];
    let bytes = to_cbor(&wire).expect("encodes");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&bytes).expect_err("too many"),
        JointLawPropagationArtifactError::LimitsExceeded("shared evidence")
    );
}
