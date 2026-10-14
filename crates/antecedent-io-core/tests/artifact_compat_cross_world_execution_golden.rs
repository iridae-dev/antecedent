//! Golden fixture for the cross-world execution artifact (`cross_world_execution_v1`).
//!
//! The golden lives at `tests/fixtures/cross_world_execution_v1.bin`. The loader consumes
//! it as a fresh consumer (re-running the abduction and the worlds from the stored
//! mechanism rather than trusting the stored number) and requires the identity a producer
//! derives today. A missing golden fails with the command that writes it.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent-io-core --test artifact_compat_cross_world_execution_golden -- --ignored write_`
//!
//! Oracle: response types of `y` given `x` with masses (never, complier, defier, always)
//! `[0.2, 0.5, 0.1, 0.2]` and `P(u_x = 1) = 0.4`. With evidence `x = 1, y = 1`,
//! `P(Y_{x0} = 0 | e) = 5/7` and `P(e) = 0.28`.
//!
//! The newer-version check reseals the stored metadata and needs `--features test-util`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::{
    ExogenousInference, ExogenousVariable, FactualUnit, FiniteExecutionRequest, FiniteQuantity,
    FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling, WorldAtom,
};
use antecedent_io_core::container_kind;
use antecedent_io_core::cross_world_execution_artifact::{
    CROSS_WORLD_EXECUTION_ARTIFACT_VERSION, CrossWorldExecutionArtifact,
};
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str = "cargo test -p antecedent-io-core --test artifact_compat_cross_world_execution_golden -- --ignored write_";
const GOLDEN: &str = "cross_world_execution_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.6, 0.4] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.2, 0.5, 0.1, 0.2] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
        ],
    }
}

fn do_x(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

fn necessity() -> FiniteExecutionRequest {
    let scm = scm();
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: vec![FactualUnit { unit: "case".to_owned(), observed: vec![(0, 1), (1, 1)] }],
        worlds: vec![do_x("x0", 0), do_x("x1", 1)],
        quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_cross_world_execution_golden() {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let sealed = CrossWorldExecutionArtifact::seal(&necessity(), &ctx()).expect("seals");
    std::fs::write(
        dir.join(GOLDEN),
        sealed.to_bytes("golden-cross-world-execution").expect("bytes"),
    )
    .expect("write the golden");
}

#[test]
fn the_cross_world_execution_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    assert_eq!(container_kind(&bytes).as_deref(), Some("cross_world_execution_v1"));

    let fresh = CrossWorldExecutionArtifact::seal(&necessity(), &ctx()).expect("seals");
    let consumed = CrossWorldExecutionArtifact::from_bytes(&bytes, Some(fresh.identity()), &ctx())
        .expect("a fresh consumer re-derives the stored execution");
    assert_eq!(consumed.identity(), fresh.identity());
    assert!((consumed.result().value - 5.0 / 7.0).abs() < 1e-12);
    assert!((consumed.result().units[0].evidence_probability - 0.28).abs() < 1e-12);
    assert_eq!(consumed.meta().version, CROSS_WORLD_EXECUTION_ARTIFACT_VERSION);
    assert_eq!(consumed.meta().inference_claim, "point_only");
}

#[cfg(feature = "test-util")]
#[test]
fn a_newer_cross_world_execution_golden_is_an_unsupported_version() {
    use antecedent_io_core::cross_world_execution_artifact::{
        CrossWorldExecutionArtifactError, decode_parts, encode_parts,
    };
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    let (mut meta, values) = decode_parts(&bytes).expect("decodes");
    meta.version = CROSS_WORLD_EXECUTION_ARTIFACT_VERSION + 1;
    let newer = encode_parts(&meta, &values, "newer").expect("encodes");
    assert!(matches!(
        CrossWorldExecutionArtifact::from_bytes(&newer, None, &ctx()),
        Err(CrossWorldExecutionArtifactError::UnsupportedVersion { version: 2 })
    ));
}
