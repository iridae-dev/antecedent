//! A declared exact-request verification is not confirmed by a result artifact:
//! no receipt travels with it, so the result stands as diagnostic, as it does on
//! the live path, and an edit that stamps it as licensed is refused.

use std::collections::BTreeSet;

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
    CallbackOutcome, CallbackResultArtifact, CallbackResultArtifactError, CallbackResultMeta,
    CallbackResultRecord,
};

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
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
    }
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
        quantity: quantity(),
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

fn computed(
    declaration: CallbackDeclaration,
    query: CallbackQuery,
    output: CallbackOutput,
) -> CallbackResultRecord {
    let output = validate_output(&query, output, &declaration.support, 1_000).expect("valid");
    let request = CallbackRequest::new(query, 9);
    CallbackResultRecord {
        declaration,
        request,
        outcome: CallbackOutcome::Computed { output, invocations: 1, state_identity: None },
    }
}

fn cdf_record() -> CallbackResultRecord {
    computed(
        declaration(),
        CallbackQuery::Cdf { thresholds: vec![0.5, 2.0, 3.5] },
        CallbackOutput::Cdf { values: vec![0.158_655_253_931_457_1, 0.5, 0.841_344_746_068_542_9] },
    )
}

fn values(record: &CallbackResultRecord) -> Vec<f64> {
    match &record.outcome {
        CallbackOutcome::Computed { output, .. } => output.output().values(),
        CallbackOutcome::Unavailable { .. } => Vec::new(),
    }
}

fn resealed(
    mut meta: CallbackResultMeta,
    values: &[f64],
    edit: impl FnOnce(&mut CallbackResultMeta),
) -> Result<CallbackResultArtifact, CallbackResultArtifactError> {
    edit(&mut meta);
    CallbackResultArtifact::reseal_identity(&mut meta);
    CallbackResultArtifact::from_meta(&meta, values, None)
}

#[test]
fn a_verified_declaration_without_a_receipt_consumes_as_diagnostic() {
    let mut record = cdf_record();
    record.declaration.trust =
        CallbackTrust::ExactRequestVerified { receipt_digest: "any-receipt-string".into() };
    let vals = values(&record);
    let (artifact, meta) = CallbackResultArtifact::seal(&record).expect("seals");
    assert_eq!(meta.standing, "diagnostic");
    let bytes = CallbackResultArtifact::to_bytes(&record, "verified").expect("bytes");
    let consumed = CallbackResultArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .expect("consumes");
    assert_eq!(consumed.record(), &record);

    // Stamping the label the unconfirmed declaration did not earn is refused.
    let error = resealed(meta, &vals, |m| m.standing = "licensed_external".into())
        .expect_err("standing edited");
    match error {
        CallbackResultArtifactError::IdentityMismatch { field } => assert_eq!(field, "standing"),
        other => panic!("expected a standing mismatch, got {other:?}"),
    }
}

#[test]
fn an_attested_declaration_keeps_its_licensed_external_label() {
    let record = cdf_record();
    let (_, meta) = CallbackResultArtifact::seal(&record).expect("seals");
    assert_eq!(meta.standing, "licensed_external");
}
