//! Compatibility class of recovery, mixed-source and study-plan artifacts sealed by the
//! previous release.
//!
//! The bytes in `tests/fixtures/earlier_release_artifacts/` were produced by the
//! previous release's shipped code (see the README there) and are immutable evidence:
//! they are never regenerated with current code.
//!
//! Class: requires regeneration, inspectable. Each artifact carries a rule or ranking
//! version spelling inside digested bytes (`x10.recovery.v1`, `x9.rules.v1`,
//! `x6.ranking.v1`, `b2.recovery_chain.v1`). The current rules are named by behavior and
//! no consumer accepts the earlier spellings, so a consumer that re-decides the
//! derivation refuses them. The stored wire still decodes, so the content (laws, rows,
//! plan, options) can be read and the artifact re-exported by the current producer from
//! the same inputs. Catalog descriptor v1 bindings can now refuse before the
//! historical rule-version check, because their identities require reissue too.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::consume_mixed_source_artifact;
use antecedent::design::{StudyPlanArtifactWire, StudyPlanConsumeLimits};
use antecedent_core::ExecutionContext;
use antecedent_io::mixed_source_artifact::{MixedSourceArtifactWire, MixedSourceConsumeLimits};
use antecedent_io::recovery_artifact::{RecoveryArtifactWire, RecoveryConsumeLimits};
use antecedent_io::recovery_chain_artifact::RecoveryChainArtifactWire;
use antecedent_io::sampled_recovery_artifact::{
    SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits,
};

const OBSERVATION_RECOVERY: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/observation_recovery_point.cbor");
const MIXED_SOURCE: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/mixed_source_point.cbor");
const RECOVERY_CHAIN: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/recovery_chain_recovered.cbor");
const SAMPLED_RECOVERY: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/sampled_recovery.cbor");
const STUDY_PLAN_MZ: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/study_plan_mz.json");
const STUDY_PLAN_MIXED: &[u8] =
    include_bytes!("fixtures/earlier_release_artifacts/study_plan_mixed.json");

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

#[test]
fn an_earlier_observation_recovery_artifact_is_inspectable_and_refused() {
    let wire = RecoveryArtifactWire::decode(OBSERVATION_RECOVERY)
        .expect("the stored wire is still readable");
    assert_eq!(wire.derivation.rule_version, "x10.recovery.v1");
    let outcome = RecoveryArtifactWire::consume_typed(
        OBSERVATION_RECOVERY,
        RecoveryConsumeLimits::default(),
        &ctx(),
    );
    let error = format!("{:?}", outcome.err().expect("the earlier rule spelling is not accepted"));
    assert!(
        error.contains(r#"ProofMismatch(RecoveryError { detail: InvalidDerivation"#),
        "{error}"
    );
}

#[test]
fn an_earlier_sampled_recovery_artifact_is_inspectable_and_refused() {
    let wire = SampledRecoveryArtifactWire::decode(SAMPLED_RECOVERY)
        .expect("the stored wire is still readable");
    assert_eq!(wire.derivation.rule_version, "x10.recovery.v1");
    let outcome = SampledRecoveryArtifactWire::consume_with_limits(
        SAMPLED_RECOVERY,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    );
    assert!(
        wire.derivation
            .margins
            .iter()
            .any(|margin| margin.identity.starts_with("catalog_distribution.v1|"))
    );
    let error =
        format!("{:?}", outcome.err().expect("the earlier catalog binding is not accepted"));
    assert!(error.contains("recovery.invalid_derivation: recovery formula check: a leaf's margin binding disagrees with the catalog distribution"), "{error}");
}

#[test]
fn an_earlier_recovery_chain_artifact_is_inspectable_and_refused() {
    let wire = RecoveryChainArtifactWire::decode(RECOVERY_CHAIN)
        .expect("the stored wire is still readable");
    assert_eq!(wire.plan.as_ref().expect("a plan").rule_version, "b2.recovery_chain.v1");
    let outcome = RecoveryChainArtifactWire::consume_typed(RECOVERY_CHAIN, &ctx());
    let error = format!("{:?}", outcome.err().expect("the earlier rule spelling is not accepted"));
    assert!(error.contains(r#"DecisionMismatch("plan")"#), "{error}");
}

#[test]
fn an_earlier_mixed_source_artifact_is_inspectable_and_refused() {
    let wire =
        MixedSourceArtifactWire::decode(MIXED_SOURCE).expect("the stored wire is still readable");
    assert_eq!(wire.proof.rule_set, "x9.rules.v1");
    let outcome = consume_mixed_source_artifact(
        MIXED_SOURCE,
        MixedSourceConsumeLimits::default(),
        &ExecutionContext::for_tests(3),
    );
    let error = format!("{:?}", outcome.err().expect("the earlier rule spelling is not accepted"));
    assert!(error.contains(r#"mixed_search.invalid_derivation"#), "{error}");
}

#[test]
fn earlier_study_plans_are_inspectable_and_refused() {
    for (name, bytes, ranking) in [
        ("mz-transport plan", STUDY_PLAN_MZ, "x6.ranking.v1"),
        ("mixed-source plan", STUDY_PLAN_MIXED, "x6.ranking.v1"),
    ] {
        let wire: StudyPlanArtifactWire =
            serde_json::from_slice(bytes).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(wire.premises.ranking, ranking, "{name}");
        let outcome = wire.consume_with_limits(
            StudyPlanConsumeLimits::default(),
            &ExecutionContext::for_tests(9),
        );
        let error = format!("{:?}", outcome.err().unwrap_or_else(|| panic!("{name}: accepted")));
        assert!(
            error.contains("study_plan.invalid_artifact")
                && error.contains("another rule set or ranking version"),
            "{name}: {error}"
        );
    }
}
