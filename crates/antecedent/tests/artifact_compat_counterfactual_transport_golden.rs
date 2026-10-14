//! Golden fixture for the counterfactual transport artifact
//! (`counterfactual_transport_v1`).
//!
//! The golden lives at `tests/fixtures/counterfactual_transport_v1.bin`. The loader
//! consumes it as a fresh consumer (re-deciding identification and re-evaluating the
//! stored laws rather than trusting the stored point) under the identity and the
//! enumerated truth a producer derives today from the same fixture. A missing golden
//! fails with the command that writes it.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent --test artifact_compat_counterfactual_transport_golden -- --ignored write_`
//!
//! The expected value is enumerated by the independent finite-SCM oracle used by the
//! lifecycle tests; the fixtures are included by path so one oracle serves every layer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    dead_code,
    reason = "test fixtures: enumerated probabilities and exact bit comparisons are the assertion"
)]

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::path::PathBuf;

use antecedent::{
    CounterfactualTransportOptions, CounterfactualTransportPreparation,
    PreparedCounterfactualTransport, prepare_counterfactual_transport,
};
use antecedent_core::ExecutionContext;
use antecedent_io::counterfactual_transport_artifact::{
    ConsumedCounterfactualTransport, ConsumedCounterfactualTransportOutcome,
    CounterfactualTransportConsumeLimits as Limits, consume_counterfactual_transport_with_limits,
};
use antecedent_io::counterfactual_transport_wire::{
    COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION, CounterfactualTransportArtifactError as E,
    CounterfactualTransportArtifactWire as Wire, CounterfactualTransportExpectedIdentity,
};
use antecedent_test_support::read_fixture_bytes;
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, ExpectedClass, all_cases};

const GENERATOR: &str = "cargo test -p antecedent --test artifact_compat_counterfactual_transport_golden -- --ignored write_";
const GOLDEN: &str = "counterfactual_transport_v1.bin";
const CASE: &str = "nonlinear_mediation_covariate_shift";
const TARGET: &str = "Y_{x0,M_{x1}}";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn case() -> Case {
    all_cases().into_iter().find(|c| c.name == CASE).expect("fixture exists")
}

fn prepared(case: &Case) -> PreparedCounterfactualTransport {
    let target = case.targets.iter().find(|t| t.label == TARGET).expect("target exists");
    assert_eq!(target.class, ExpectedClass::Identified);
    let problem = adapter::problem(case);
    let adapted = adapter::catalog(case);
    let sources = adapter::sources(case, &adapter::evidence_controllable(case));
    let query = adapter::query(case, &target.query, sources);
    let laws = adapter::laws(case, &adapted);
    let options = CounterfactualTransportOptions {
        variable_names: case.graph.observed.iter().map(|n| (*n).to_owned()).collect(),
        ..Default::default()
    };
    match prepare_counterfactual_transport(
        &problem,
        &query,
        &adapted.catalog,
        Some(laws),
        &options,
        &ctx(),
    )
    .expect("prepares")
    {
        CounterfactualTransportPreparation::Identified(prepared) => *prepared,
        other @ CounterfactualTransportPreparation::ProvenNonTransportable(_) => {
            panic!("expected a derivation, got {other:?}")
        }
    }
}

fn consume(
    bytes: &[u8],
    expected: &CounterfactualTransportExpectedIdentity,
) -> Result<ConsumedCounterfactualTransport, E> {
    consume_counterfactual_transport_with_limits(bytes, expected, Limits::default(), &ctx())
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_counterfactual_transport_golden() {
    let case = case();
    let prepared = prepared(&case);
    let result = prepared.estimate(&ctx()).expect("estimates");
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(GOLDEN), result.export_artifact().expect("export"))
        .expect("write the golden");
}

#[test]
fn the_counterfactual_transport_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    let case = case();
    let fresh_prepared = prepared(&case);
    let fresh = fresh_prepared.estimate(&ctx()).expect("estimates");
    let identity = fresh.identity().expect("identity");
    let truth = case
        .targets
        .iter()
        .find(|t| t.label == TARGET)
        .and_then(|t| t.expected_value)
        .expect("an identified target is defined");

    let consumed = consume(&bytes, &identity)
        .expect("a fresh consumer re-decides and re-evaluates the stored artifact");
    let ConsumedCounterfactualTransportOutcome::Identified(identified) = &consumed.outcome else {
        panic!("expected an identified artifact");
    };
    assert!((identified.point.probability - truth).abs() <= 1e-12);
    assert_eq!(consumed.wire.version, COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION);

    // A newer version is refused before the payload is read.
    let mut newer: Wire = consumed.wire.clone();
    newer.version = COUNTERFACTUAL_TRANSPORT_ARTIFACT_VERSION + 1;
    let none = CounterfactualTransportExpectedIdentity::default();
    let error = consume(&antecedent_io::to_cbor(&newer).expect("encode"), &none).unwrap_err();
    assert_eq!(error, E::UnsupportedVersion { version: 2 });
    // An unknown required feature is refused, not guessed.
    let mut feature = consumed.wire;
    feature.required_features = vec!["counterfactual_transport_v2".into()];
    let error = consume(&antecedent_io::to_cbor(&feature).expect("encode"), &none).unwrap_err();
    assert_eq!(error, E::UnsupportedSemantics("required_features"));
}
