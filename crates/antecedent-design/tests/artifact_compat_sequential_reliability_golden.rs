//! Golden fixtures for the sequential evaluation artifact (`sequential_evaluation_v1`) and
//! the decision reliability artifact (`decision_reliability_v1`).
//!
//! Each golden lives at `tests/fixtures/<family>_v1.bin`. A loader consumes it as a fresh
//! consumer (re-running the evaluation, or recomputing the report from the carried draws,
//! rather than trusting the stored numbers) under the identities a producer derives today
//! from the same declaration. A missing golden fails with the command that writes it. A
//! changed golden is a compatibility decision and needs a migration note.
//!
//! Generate or refresh both with
//! `cargo test -p antecedent-design --test artifact_compat_sequential_reliability_golden -- --ignored write_`
//!
//! Oracle for the sequential evaluation: prior (0.75, 0.25) over (sound, faulty) and a
//! single commit action `ship` paying +10 if sound and -10 if faulty, so shipping now is
//! worth 0.75 * 10 - 0.25 * 10 = 5.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    BatchPlan, DecisionReliabilityArtifact, DecisionReliabilityExpectation, KnownOptimumDesign,
    NoiseCoupling, RankingRequest, ReliabilityRequest, StreamId, StreamPurpose, ValidationKind,
    ValidationPlan, assess, consume as consume_reliability, simulate_selection_draws,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey,
};
use antecedent_design::sequential_evaluation_artifact::{
    ExpectedEvaluation, SEQUENTIAL_EVALUATION_ARTIFACT_FEATURE,
    SEQUENTIAL_EVALUATION_ARTIFACT_VERSION, SequentialEvaluationArtifact,
    consume as consume_sequential,
};
use antecedent_io_core::IoError;
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str = "cargo test -p antecedent-design --test artifact_compat_sequential_reliability_golden -- --ignored write_";
const SEQUENTIAL_GOLDEN: &str = "sequential_evaluation_v1.bin";
const RELIABILITY_GOLDEN: &str = "decision_reliability_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Vec<u8> {
    read_fixture_bytes(&fixture_dir(), name, &format!("generate it with `{GENERATOR}`"))
}

fn write(name: &str, bytes: &[u8]) {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(name), bytes).expect("write the golden");
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

// -- sequential evaluation -----------------------------------------------------------------------

fn ship_model() -> FiniteSequentialModel {
    FiniteSequentialModel {
        label: "ship_now".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.75, 0.25],
        horizon: 1,
        actions: vec![ActionSpec {
            label: "ship".into(),
            role: ActionRole::Commit,
            utility_cost: 0.0,
            resource_use: 0.0,
            transition: vec![1.0, 0.0, 0.0, 1.0],
            channel: None,
            immediate_utility: vec![10.0, -10.0],
            delayed: None,
        }],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(1),
    }
}

fn ship_policy() -> ContingentPolicy {
    ContingentPolicy::new("ship_now_v1").decide(HistoryKey::from_steps([]), 0)
}

fn sequential_artifact() -> SequentialEvaluationArtifact {
    SequentialEvaluationArtifact::new(
        &ship_model(),
        &ship_policy(),
        None,
        None,
        EvaluationLimits::default(),
        &ctx(),
    )
    .expect("seals")
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_sequential_evaluation_golden() {
    let bytes = sequential_artifact().to_bytes("golden-sequential-evaluation").expect("encodes");
    write(SEQUENTIAL_GOLDEN, &bytes);
}

#[test]
fn the_sequential_evaluation_golden_is_replayed_and_gated() {
    let bytes = load(SEQUENTIAL_GOLDEN);
    let fresh = sequential_artifact();
    let identities = fresh.identities();
    let expected = ExpectedEvaluation {
        model: Some(identities.model.clone()),
        policy: Some(identities.policy.clone()),
        evaluation: Some(identities.evaluation.clone()),
    };
    let consumed = consume_sequential(&bytes, Some(&expected), &ctx())
        .expect("a fresh consumer re-runs the stored evaluation");
    assert_eq!(consumed.value().to_bits(), fresh.value().to_bits());
    assert!((consumed.value() - 5.0).abs() < 1e-12);
    assert_eq!(consumed.body.version, u32::from(SEQUENTIAL_EVALUATION_ARTIFACT_VERSION));
    assert_eq!(consumed.body.inference, "point_only");
    assert_eq!(SEQUENTIAL_EVALUATION_ARTIFACT_FEATURE, "sequential_evaluation_v1");

    // A newer body version, resealed so only the version gate can refuse it, is an
    // unsupported version and not an identity mismatch or a replay failure.
    let mut newer = SequentialEvaluationArtifact::from_bytes(&bytes).expect("decodes");
    newer.body.version = u32::from(SEQUENTIAL_EVALUATION_ARTIFACT_VERSION) + 1;
    newer.seal().expect("reseals");
    let newer_bytes = newer.to_bytes("newer").expect("encodes");
    match consume_sequential(&newer_bytes, None, &ctx()) {
        Err(IoError::UnsupportedVersion { version }) => assert_eq!(version, 2),
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

// -- decision reliability -----------------------------------------------------------------------

fn stream(purpose: StreamPurpose) -> StreamId {
    StreamId::new(20_240, purpose, 0, "test_stream")
}

fn reliability_artifact() -> (DecisionReliabilityArtifact, DecisionReliabilityExpectation) {
    let design = KnownOptimumDesign {
        true_means: vec![0.2, 0.0, 0.4],
        noise_sd: 1.0,
        n_draws: 300,
        coupling: NoiseCoupling::SharedStream { correlation: 0.4 },
    };
    let selection = simulate_selection_draws(&design, &stream(StreamPurpose::Selection)).unwrap();
    let validation = simulate_selection_draws(&design, &stream(StreamPurpose::Validation)).unwrap();
    let request = ReliabilityRequest {
        validation: ValidationPlan::ProvidedStream {
            kind: ValidationKind::IndependentSimulationStream,
            gap: 0,
        },
        ranking: Some(RankingRequest {
            baseline: Some("c00001".into()),
            top_k: vec![2],
            batch_plan: BatchPlan::SquareRoot,
        }),
        ..ReliabilityRequest::expected_utility()
    };
    let expected = DecisionReliabilityExpectation {
        draws_digest: Some(selection.digest_hex()),
        candidate_set_digest: Some(selection.candidate_set_digest()),
        selection_stream: Some(stream(StreamPurpose::Selection)),
        validation_stream: Some(stream(StreamPurpose::Validation)),
        utility_unit: Some("utility".into()),
        decided_id: None,
    };
    let artifact =
        DecisionReliabilityArtifact::new(selection, Some(validation), request, &ctx()).unwrap();
    (artifact, expected)
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_decision_reliability_golden() {
    let (artifact, _) = reliability_artifact();
    let bytes = artifact.to_bytes("golden-decision-reliability", &ctx()).expect("encodes");
    write(RELIABILITY_GOLDEN, &bytes);
}

#[test]
fn the_decision_reliability_golden_is_recomputed_and_gated() {
    let bytes = load(RELIABILITY_GOLDEN);
    let (fresh, fresh_expected) = reliability_artifact();
    // The golden carries draws sampled through the platform's libm, so their digest is taken
    // from the golden itself; streams and unit are platform independent and stay gated.
    let structural = DecisionReliabilityExpectation {
        draws_digest: None,
        candidate_set_digest: None,
        ..fresh_expected
    };
    let consumed = consume_reliability(&bytes, &structural, &ctx())
        .expect("a fresh consumer recomputes the report from the carried draws");
    let expected = DecisionReliabilityExpectation {
        draws_digest: Some(consumed.draws.digest_hex()),
        candidate_set_digest: Some(consumed.draws.candidate_set_digest()),
        ..structural
    };
    consume_reliability(&bytes, &expected, &ctx())
        .expect("the golden's own digests gate the same bytes");
    assert_eq!(consumed.request, fresh.request);
    assert_eq!(consumed.version, 1);

    // A newer body version is its own refusal, before the digest is considered.
    let mut newer = consumed.clone();
    newer.version = 2;
    let body = antecedent_io_core::convert::to_cbor(&newer).expect("encodes");
    let newer_bytes = antecedent_io_core::encode_container(
        &antecedent_io_core::ArtifactContainer {
            kind: "decision_reliability_v1",
            note: "compat",
            max_bytes: 256 * 1024 * 1024,
            artifact_id: "newer",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[antecedent_io_core::OutSection {
            id: "reliability_body",
            content_type: antecedent_io_core::CONTENT_CBOR,
            bytes: &body,
        }],
    )
    .expect("container");
    match consume_reliability(&newer_bytes, &expected, &ctx()) {
        Err(IoError::UnsupportedVersion { version }) => assert_eq!(version, 2),
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

#[test]
fn historical_closure_snapshot_is_conservative_and_cannot_change_the_report() {
    let bytes = load(RELIABILITY_GOLDEN);
    let original = DecisionReliabilityArtifact::from_bytes(&bytes).unwrap();
    let current =
        assess(&original.draws, original.validation.as_ref(), &original.request, &ctx()).unwrap();
    original.verify(&DecisionReliabilityExpectation::default(), &ctx()).unwrap();
    // Newly earned licenses do not rewrite or open the historical report's refusals.
    assert!(original.report.closed_routes.len() > current.closed_routes.len());
    let consumed =
        consume_reliability(&bytes, &DecisionReliabilityExpectation::default(), &ctx()).unwrap();
    assert_eq!(consumed.digest, original.digest);
    assert_eq!(consumed.report.closed_routes, original.report.closed_routes);

    let rejected = |mut artifact: DecisionReliabilityArtifact| {
        artifact.digest = artifact.recomputed_digest().unwrap();
        assert!(
            artifact
                .verify(&DecisionReliabilityExpectation::default(), &ctx())
                .unwrap_err()
                .to_string()
                .contains("report_mismatch")
        );
    };
    // A currently closed route cannot be omitted, even in a consistently resealed body.
    let mut changed = original.clone();
    changed.report.closed_routes.retain(|route| route != &current.closed_routes[0]);
    rejected(changed);
    let mut changed = original.clone();
    changed.report.closed_routes[0].coordinate.push_str("/foreign");
    rejected(changed);
    let mut changed = original.clone();
    changed.report.closed_routes[0].remedy = "this route is licensed".into();
    rejected(changed);
    let mut changed = original.clone();
    changed.report.closed_routes[0].code = "invalid_argument".into();
    rejected(changed);
    let mut changed = original.clone();
    changed.report.closed_routes.push(changed.report.closed_routes[0].clone());
    rejected(changed);
    let mut changed = original.clone();
    changed.report.closed_routes.swap(0, 1);
    rejected(changed);
    let mut changed = original.clone();
    let se = &mut changed.report.monte_carlo_error.decision.standard_error;
    *se = f64::from_bits(se.to_bits() + 1);
    rejected(changed);
    let mut changed = original;
    changed.report.decided_id = "not_the_selected_candidate".into();
    rejected(changed);
}
