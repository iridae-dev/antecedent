//! Independent CPU consumers of the committed cyclic family fixtures.
//!
//! Every test reads committed bytes from `parity/cyclic/` with the separately
//! committed expected identity, runs on its own CPU execution context (scalar
//! kernels, another seed than the producer) and never calls the producer. The
//! consumer verifies the version-two contract from bytes, re-derives every payload
//! from the request, replays the answer, and must reproduce the committed report
//! bit for bit. The analytic truths below are written out here, not read from the
//! fixture.
//!
//! Generate the fixtures with
//! `cargo test -p antecedent --test cyclic_family_fixtures -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_core::{ExecutionContext, KernelPolicy};
use antecedent_io::contract_section::cyclic::{
    CYCLIC_CONTRACT_SECTION, CyclicContractWire, CyclicIdentificationWire, CyclicUnresolvedWire,
};
use antecedent_io::cyclic_family_artifact::{
    CyclicFamilyAnswer, CyclicFamilyReport, CyclicWorldOutcomeAnswer, DynamicInferenceAnswer,
    RecoveredDrawOriginWire, RecoveredDrawStatusWire, consume_cyclic_family,
    consume_cyclic_family_contract,
};
use antecedent_io::cyclic_system_artifact::{CyclicArtifactError, MAX_CYCLIC_ARTIFACT_BYTES};
use antecedent_io::{
    ArtifactContainer, CONTENT_BYTES, CONTENT_CBOR, OutSection, decode_container, encode_container,
    from_cbor, to_cbor,
};
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str =
    "cargo test -p antecedent --test cyclic_family_fixtures -- --ignored write_";
/// Record section of a version-two family artifact.
const RECORD_SECTION: &str = "cyclic_family_v2";

/// A CPU execution context unrelated to the producer's: scalar kernels and another seed.
fn cpu() -> ExecutionContext {
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.kernel_policy = KernelPolicy::scalar_only();
    ctx
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected} (±{tolerance})");
}

/// Committed bytes, expected identity and report of one fixture.
struct Committed {
    bytes: Vec<u8>,
    identity: String,
    report: String,
}

fn committed(name: &str) -> Committed {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../parity/cyclic").join(name);
    let hint = format!("generate it with `{GENERATOR}`");
    let bytes = read_fixture_bytes(&dir, "family.ant", &hint);
    let expected: serde_json::Value =
        serde_json::from_slice(&read_fixture_bytes(&dir, "expected-identity.json", &hint)).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&read_fixture_bytes(&dir, "manifest.json", &hint)).unwrap();
    assert_eq!(
        manifest["artifact_blake3"].as_str(),
        Some(antecedent_core::blake3_hex(&bytes).as_str()),
        "{name}: the committed artifact differs from its manifest"
    );
    let report = String::from_utf8(read_fixture_bytes(&dir, "report.json", &hint)).unwrap();
    Committed { bytes, identity: expected["identity"].as_str().unwrap().to_owned(), report }
}

fn pretty(report: &CyclicFamilyReport) -> String {
    serde_json::to_string_pretty(report).unwrap() + "\n"
}

/// Verify the contract, replay on the CPU context, and compare bit for bit.
fn replay(name: &str) -> (CyclicContractWire, CyclicFamilyReport) {
    let fixture = committed(name);
    let (contract, report) =
        consume_cyclic_family_contract(&fixture.bytes, &fixture.identity, &cpu())
            .unwrap_or_else(|e| panic!("{name}: the committed record does not replay: {e}"));
    assert_eq!(report.identity, fixture.identity);
    assert_eq!(contract.request_identity, fixture.identity);
    assert_eq!(pretty(&report), fixture.report, "{name}: replay differs from the committed report");
    assert!(report.lowered_program_identity().is_none(), "a replay is not a prepared execution");
    (contract, report)
}

fn worlds(
    report: &CyclicFamilyReport,
) -> &[antecedent_io::cyclic_family_artifact::CyclicWorldAnswer] {
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else {
        panic!("typed worlds expected, got {:?}", report.answer);
    };
    worlds
}

fn gaussian(answer: &antecedent_io::cyclic_family_artifact::CyclicWorldAnswer) -> (&[f64], &[f64]) {
    match &answer.outcome {
        CyclicWorldOutcomeAnswer::GaussianLaw { mean, covariance, .. } => {
            (mean.as_slice(), covariance.as_slice())
        }
        other => panic!("world {} was {other:?}", answer.world),
    }
}

#[test]
fn the_specified_feedback_pair_replays_its_three_distinct_quantities() {
    let (contract, report) = replay("specified-feedback-pair");
    assert!(matches!(
        contract.identification,
        CyclicIdentificationWire::SpecifiedModelConditional { .. }
    ));
    let worlds = worlds(&report);
    assert_eq!(worlds.iter().map(|w| w.world).collect::<Vec<_>>(), [0, 1, 2, 3]);
    for world in worlds {
        assert!(world.certificate.certified, "{world:?}");
        close(world.failed_mass, 0.0, 0.0);
        assert_eq!(world.physical_stability, "unclaimed");
    }
    // Observational association of B on A: Cov(A, B) / Var(A) = 1.40625 / 1.8125.
    let (_, covariance) = gaussian(&worlds[0]);
    let association = covariance[1] / covariance[0];
    close(association, 1.40625 / 1.8125, 1e-12);
    close(association, 0.776, 5e-4);
    // Hard-clamp derivative of B in A.
    close(gaussian(&worlds[2]).0[1] - gaussian(&worlds[1]).0[1], 0.5, 1e-12);
    // A unit mean shift of U_A keeps the feedback active.
    let (shifted, _) = gaussian(&worlds[3]);
    close(shifted[0], 1.25, 1e-12);
    close(shifted[1], 0.625, 1e-12);
    assert_eq!(worlds[1].actions[0].kind, "hard_clamp");
    assert_eq!(worlds[3].actions[0].kind, "additive_shock");
}

#[test]
fn the_overlapping_loop_clamp_is_retained_as_failed_mass() {
    let (_, report) = replay("overlapping-loops");
    let worlds = worlds(&report);
    // The full system solves; the determinant alone does not certify uniqueness.
    assert!(!worlds[0].certificate.certified);
    assert!(worlds[0].certificate.next_attempt.is_some());
    // do(C = 1) leaves 0 = 1 + U_A + U_B: no unique solution, kept as failed mass.
    assert!(matches!(worlds[1].outcome, CyclicWorldOutcomeAnswer::NoSolution { .. }));
    close(worlds[1].failed_mass, 1.0, 0.0);
}

fn recovered(
    report: &CyclicFamilyReport,
) -> &antecedent_io::cyclic_family_artifact::ExperimentalRecoveryReport {
    let CyclicFamilyAnswer::ExperimentalRecovery(recovered) = &report.answer else {
        panic!("experimental recovery expected, got {:?}", report.answer);
    };
    recovered
}

#[test]
fn recovered_laws_keep_their_draw_origin_through_an_independent_replay() {
    for (name, draws) in [("recovered-asymptotic-normal", 64), ("recovered-bootstrap", 12)] {
        let (contract, report) = replay(name);
        let CyclicIdentificationWire::ExperimentalRecovery { unresolved, .. } =
            &contract.identification
        else {
            panic!("{name}: experimental recovery identification expected");
        };
        assert!(unresolved.is_empty(), "{name}: the complete design leaves nothing unresolved");
        assert!(contract.recovery.is_some());
        let law = recovered(&report);
        assert_eq!(law.law_meaning.as_deref(), Some("sampling_law"), "{name}");
        assert_eq!(law.law.draws.len(), draws, "{name}");
        let origin = &law.law.origin;
        let expected = if draws == 64 {
            matches!(origin, RecoveredDrawOriginWire::AsymptoticNormal { draws: 64, seed: 7 })
        } else {
            matches!(origin, RecoveredDrawOriginWire::Bootstrap { replicates: 12, seed: 4, .. })
        };
        assert!(expected, "{name}: origin {origin:?}");
        assert_eq!(law.worlds.iter().map(|w| w.world).collect::<Vec<_>>(), [1, 2]);
        // do(A = 2) on simulated data from B = 0.5 A + U_B.
        let clamped = &law.worlds[1].answer;
        close(clamped.mixture_mean.unwrap(), 1.0, 0.25);
        close(clamped.withheld_mass, 0.0, 1e-12);
    }
}

#[test]
fn an_unresolved_direction_stays_unresolved_and_withholds_its_mass() {
    let (contract, report) = replay("recovered-unresolved-direction");
    let CyclicIdentificationWire::ExperimentalRecovery { unresolved, .. } =
        &contract.identification
    else {
        panic!("experimental recovery identification expected");
    };
    assert_eq!(unresolved, &[CyclicUnresolvedWire { target: 0, missing_clamps: vec![1] }]);
    let law = recovered(&report);
    assert!(
        law.law
            .draw_status
            .iter()
            .all(|status| matches!(status, RecoveredDrawStatusWire::Unresolved { .. }))
    );
    let clamped = &law.worlds[1].answer;
    assert!(clamped.mixture_mean.is_none());
    close(clamped.withheld_mass, 1.0, 1e-12);
}

#[test]
fn the_dynamic_inference_family_replays_its_seeded_sampling_law() {
    let (contract, report) = replay("dynamic-inference-ou");
    assert!(matches!(
        contract.identification,
        CyclicIdentificationWire::SpecifiedModelConditional { .. }
    ));
    let CyclicFamilyAnswer::DynamicInference { inference, .. } = &report.answer else {
        panic!("dynamic inference expected");
    };
    let DynamicInferenceAnswer::SamplingLaw(law) = inference.as_ref() else {
        panic!("bootstrap sampling law expected");
    };
    assert_eq!(law.replicates.len(), 20);
}

#[test]
fn the_previous_record_format_still_replays_and_carries_no_contract() {
    let fixture = committed("previous-record-format");
    let report = consume_cyclic_family(&fixture.bytes, &fixture.identity, &cpu()).unwrap();
    assert_eq!(pretty(&report), fixture.report);
    let refused = consume_cyclic_family_contract(&fixture.bytes, &fixture.identity, &cpu())
        .expect_err("a version-one record carries no contract");
    assert!(matches!(
        refused,
        CyclicArtifactError::Invalid(message) if message.contains("next attempt")
    ));
}

/// Decode the two sections of a version-two record.
fn sections(bytes: &[u8]) -> (serde_json::Value, CyclicContractWire) {
    let parts = decode_container(
        "cyclic_family",
        MAX_CYCLIC_ARTIFACT_BYTES,
        &[RECORD_SECTION, CYCLIC_CONTRACT_SECTION],
        bytes,
    )
    .expect("a version-two container");
    (serde_json::from_slice(&parts[0]).unwrap(), from_cbor(&parts[1]).unwrap())
}

/// Re-encode a record and contract in a well-formed container.
fn reencode(record: &serde_json::Value, contract: &CyclicContractWire) -> Vec<u8> {
    let record = serde_json::to_vec(record).unwrap();
    let contract = to_cbor(contract).unwrap();
    encode_container(
        &ArtifactContainer {
            kind: "cyclic_family",
            note: "specified_dynamic_or_constraint_model",
            max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
            artifact_id: "cyclic_family",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            OutSection { id: RECORD_SECTION, content_type: CONTENT_BYTES, bytes: &record },
            OutSection {
                id: CYCLIC_CONTRACT_SECTION,
                content_type: CONTENT_CBOR,
                bytes: &contract,
            },
        ],
    )
    .unwrap()
}

/// Perturb the first floating-point number under `value`.
fn perturb_first_float(value: &mut serde_json::Value) -> bool {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => {
            let shifted = number.as_f64().unwrap() + 1e-9;
            *value = serde_json::json!(shifted);
            true
        }
        serde_json::Value::Array(items) => items.iter_mut().any(perturb_first_float),
        serde_json::Value::Object(fields) => fields.values_mut().any(perturb_first_float),
        _ => false,
    }
}

#[test]
fn identity_report_contract_and_version_tampering_are_refused() {
    let fixture = committed("specified-feedback-pair");
    let other = committed("overlapping-loops");
    let ctx = cpu();

    // A wrong expected identity: a malformed one, and another fixture's.
    assert!(consume_cyclic_family(&fixture.bytes, &"0".repeat(64), &ctx).is_err());
    assert!(matches!(
        consume_cyclic_family(&fixture.bytes, &other.identity, &ctx),
        Err(CyclicArtifactError::Mismatch)
    ));
    // A corrupted byte inside the checksummed body fails the container checks.
    let mut flipped = fixture.bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0x01;
    assert!(consume_cyclic_family(&flipped, &fixture.identity, &ctx).is_err());

    // Re-encoding the unchanged sections still replays, so each refusal below is
    // caused by its own edit.
    let (record, contract) = sections(&fixture.bytes);
    let control = reencode(&record, &contract);
    assert_eq!(
        pretty(&consume_cyclic_family(&control, &fixture.identity, &ctx).unwrap()),
        fixture.report
    );

    // A tampered report: the replay no longer equals the stored answer.
    let mut tampered = record.clone();
    assert!(perturb_first_float(&mut tampered["report"]["answer"]));
    assert!(matches!(
        consume_cyclic_family(&reencode(&tampered, &contract), &fixture.identity, &ctx),
        Err(CyclicArtifactError::Mismatch)
    ));

    // A tampered contract: the byte-level verification fails.
    let mut edited = contract.clone();
    edited.observation.variables.push(99);
    assert!(consume_cyclic_family(&reencode(&record, &edited), &fixture.identity, &ctx).is_err());
    // Another fixture's self-consistent contract does not bind this request.
    let (_, foreign) = sections(&other.bytes);
    assert!(consume_cyclic_family(&reencode(&record, &foreign), &fixture.identity, &ctx).is_err());

    // An unknown record version is refused before the request is interpreted.
    let mut newer = record;
    newer["version"] = serde_json::json!(3);
    let refused = consume_cyclic_family(&reencode(&newer, &contract), &fixture.identity, &ctx)
        .expect_err("an unknown version");
    assert!(matches!(
        refused,
        CyclicArtifactError::Invalid(message) if message.contains("unsupported cyclic family version")
    ));
}
