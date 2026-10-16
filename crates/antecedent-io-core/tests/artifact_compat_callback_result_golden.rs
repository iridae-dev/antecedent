//! Golden fixture for the callback result artifact (`callback_result_v1`).
//!
//! The golden lives at `tests/fixtures/callback_result_v1.bin`. The loader consumes it as
//! a fresh consumer (rebuilding the declaration, re-validating the request and the output
//! and recomputing every identity rather than trusting the stored ones) and requires the
//! identity a producer derives today. A missing golden fails with the command that writes
//! it. Decoding never executes anything: the artifact is data and the consumer has no
//! provider.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent-io-core --test artifact_compat_callback_result_golden -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::path::PathBuf;

use antecedent_contract::callback_protocol::{
    CallbackDeclaration, CallbackDependency, CallbackKind, CallbackOperation, CallbackOutput,
    CallbackQuery, CallbackRequest, CallbackTrust, DependencyDeclaration, DependencyKind,
    ReplayContract, validate_output,
};
use antecedent_contract::interface_contract::{EvidenceBasis, SupportDomain};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_io_core::callback_result_artifact::{
    CALLBACK_RESULT_ARTIFACT_FEATURE, CALLBACK_RESULT_ARTIFACT_VERSION, CallbackOutcome,
    CallbackResultArtifact, CallbackResultArtifactError, CallbackResultRecord,
};
use antecedent_io_core::container_kind;
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str = "cargo test -p antecedent-io-core --test artifact_compat_callback_result_golden -- --ignored write_";
const GOLDEN: &str = "callback_result_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn declaration() -> CallbackDeclaration {
    CallbackDeclaration {
        component: "gaussian_law".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "gaussian".into(),
            version_id: "1.0".into(),
            snapshot_id: "snap-1".into(),
            request_id: "req-1".into(),
        },
        quantity: ScientificQuantity {
            variable_id: "schema:systolic".into(),
            variable_name: "systolic".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "trial".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        },
        semantic_identity: "estimand:outcome_law".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [
            CallbackOperation::Mean,
            CallbackOperation::Sample,
            CallbackOperation::Cdf,
            CallbackOperation::Quantile,
        ]
        .into_iter()
        .collect(),
        update: None,
        inputs: Vec::new(),
        dependencies: DependencyDeclaration::Complete(vec![
            CallbackDependency::new(DependencyKind::Model, "parameters", "mu=2;sd=1.5"),
            CallbackDependency::new(DependencyKind::File, "weights", "sha:abc"),
        ]),
        replay: ReplayContract::SeededStochastic { stream_id: "outcome_draws".into() },
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

/// A Gaussian(2, 1.5) CDF at 0.5, 2.0 and 3.5: Phi(-1), 1/2 and Phi(1).
fn cdf_record() -> CallbackResultRecord {
    let declaration = declaration();
    let query = CallbackQuery::Cdf { thresholds: vec![0.5, 2.0, 3.5] };
    let output = validate_output(
        &query,
        CallbackOutput::Cdf { values: vec![0.158_655_253_931_457_1, 0.5, 0.841_344_746_068_542_9] },
        &declaration.support,
        1_000,
    )
    .expect("valid");
    CallbackResultRecord {
        declaration,
        request: CallbackRequest::new(query, 9),
        outcome: CallbackOutcome::Computed { output, invocations: 1, state_identity: None },
    }
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_callback_result_golden() {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    let bytes =
        CallbackResultArtifact::to_bytes(&cdf_record(), "golden-callback-result").expect("bytes");
    std::fs::write(dir.join(GOLDEN), bytes).expect("write the golden");
}

#[test]
fn the_callback_result_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    assert_eq!(container_kind(&bytes).as_deref(), Some("callback_result_v1"));

    let record = cdf_record();
    let (fresh, fresh_meta) = CallbackResultArtifact::seal(&record).expect("seals");
    let consumed = CallbackResultArtifact::from_bytes(&bytes, Some(fresh.identity()))
        .expect("a fresh consumer re-derives the stored result");
    assert_eq!(consumed.identity(), fresh.identity());
    assert_eq!(consumed.record(), &record);
    assert_eq!(consumed.declaration_identity(), record.declaration.canonical_identity().to_hex());

    let meta = CallbackResultArtifact::inspect(&bytes).expect("inspectable");
    assert_eq!(meta.version, CALLBACK_RESULT_ARTIFACT_VERSION);
    assert_eq!(meta.feature, CALLBACK_RESULT_ARTIFACT_FEATURE);

    // A newer version is refused even under a consistent identity.
    let values = [0.158_655_253_931_457_1, 0.5, 0.841_344_746_068_542_9];
    let mut newer = fresh_meta.clone();
    newer.version = CALLBACK_RESULT_ARTIFACT_VERSION + 1;
    CallbackResultArtifact::reseal_identity(&mut newer);
    assert!(matches!(
        CallbackResultArtifact::from_meta(&newer, &values, None),
        Err(CallbackResultArtifactError::UnsupportedVersion { version: 2 })
    ));
    let newer_bytes =
        CallbackResultArtifact::meta_to_bytes(&newer, &values, "newer").expect("bytes");
    assert!(matches!(
        CallbackResultArtifact::from_bytes(&newer_bytes, None),
        Err(CallbackResultArtifactError::UnsupportedVersion { version: 2 })
    ));
}
