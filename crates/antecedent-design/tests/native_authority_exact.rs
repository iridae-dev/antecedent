//! Native authority: which of several native-labelled sources governs a decision, derived from
//! the rule that only a matching native execution record grants native standing.
//!
//! Two joint laws over (p at do(a=1), s at do(a=0)) both carry the artifact label `native` and
//! `exact`. Source N has the draws (1,3), (3,5), (2,3), (2,5): E[p] = 2 and E[s] = 4. Source E has
//! (9,3), (11,5), (10,3), (10,5): E[p] = 10 and E[s] = 4. Only N has a native execution record
//! for its provider and snapshot.
//!
//! Hand derivation, with the actions go = p and stay = s (utility = the coordinate):
//! - requiring native, E is refused as a metadata-only native claim; only N enters, so go = 2
//!   and stay = 4 and stay is uniquely optimal.
//! - unrestricted, with E listed first, E answers both actions (the first input covering them):
//!   go = 10, stay = 4, and go is uniquely optimal. Trust does not reorder the inputs.
//! - E is stored unverified with its `exact` calibration claim unused; N keeps `exact`.
//! - a record for another snapshot, or with a blank execution id, proves nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::{
    ActionStatus, BoundaryError, CalibrationStatus, DecisionInput, NativeExecutionRecord,
    ProviderKind, ReceiptRef, SupportPolicy, SupportedVerdict, TrustEvidence, TrustRequirement,
    evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity { regime_id: regime.into(), ..common::quantity(variable) }
}

fn artifact(provider: &str, snapshot: &str, rows: [(f64, f64); 4]) -> DistributionArtifact {
    let columns = [quantity("p", "do(a=1)"), quantity("s", "do(a=0)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: format!("src-{provider}"),
            provider_id: provider.into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::NativeSupported,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.iter().flat_map(|(p, s)| [*p, *s]).collect(),
    )
    .unwrap()
}

fn source_n() -> DistributionArtifact {
    artifact("native-prov", "snapN", [(1.0, 3.0), (3.0, 5.0), (2.0, 3.0), (2.0, 5.0)])
}

fn source_e() -> DistributionArtifact {
    artifact("ext-prov", "snapE", [(9.0, 3.0), (11.0, 5.0), (10.0, 3.0), (10.0, 5.0)])
}

fn record(provider: &str, snapshot: &str, execution: &str) -> TrustEvidence {
    TrustEvidence::NativeExecution(NativeExecutionRecord {
        execution_id: execution.into(),
        provider_id: provider.into(),
        snapshot_id: snapshot.into(),
    })
}

fn contract() -> DecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(variable, regime)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("go", "p", "do(a=1)"), action("stay", "s", "do(a=0)")],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn build(
    id: &str,
    artifact: &DistributionArtifact,
    evidence: &TrustEvidence,
    requirement: TrustRequirement,
) -> Result<DecisionInput, BoundaryError> {
    DecisionInput::from_distribution_artifact(id, artifact, evidence, requirement)
}

fn utilities(inputs: &[DecisionInput]) -> (Vec<(String, f64)>, SupportedVerdict) {
    let decision =
        evaluate_with_support(&contract(), inputs, &SupportPolicy::compare_supported()).unwrap();
    assert!(decision.dispositions.iter().all(|d| d.status == ActionStatus::Evaluated));
    (
        decision.outcomes.iter().map(|o| (o.id.clone(), o.expected_utility)).collect(),
        decision.verdict,
    )
}

#[test]
fn requiring_native_leaves_only_the_executed_source_and_its_hand_values_decide() {
    let n_evidence = record("native-prov", "snapN", "exec-1");
    let n = build("N", &source_n(), &n_evidence, TrustRequirement::Native).unwrap();
    let refused = build("E", &source_e(), &TrustEvidence::None, TrustRequirement::Native);
    assert_eq!(refused.unwrap_err(), BoundaryError::MetadataOnlyNativeClaim { input: "E".into() });

    let (values, verdict) = utilities(&[n]);
    assert_eq!(values, vec![("go".to_owned(), 2.0), ("stay".to_owned(), 4.0)]);
    assert_eq!(verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("stay".into())));
}

#[test]
fn unrestricted_the_first_listed_source_governs_and_trust_does_not_reorder() {
    let n = build("N", &source_n(), &record("native-prov", "snapN", "exec-1"), TrustRequirement::Unrestricted)
        .unwrap();
    let e = build("E", &source_e(), &TrustEvidence::None, TrustRequirement::Unrestricted).unwrap();
    let (values, verdict) = utilities(&[e.clone(), n.clone()]);
    assert_eq!(values, vec![("go".to_owned(), 10.0), ("stay".to_owned(), 4.0)]);
    assert_eq!(verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("go".into())));
    let (values, verdict) = utilities(&[n, e]);
    assert_eq!(values, vec![("go".to_owned(), 2.0), ("stay".to_owned(), 4.0)]);
    assert_eq!(verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("stay".into())));
}

#[test]
fn only_a_matching_nonblank_record_grants_native_standing_and_keeps_the_exact_claim() {
    let n = build("N", &source_n(), &record("native-prov", "snapN", "exec-1"), TrustRequirement::Unrestricted)
        .unwrap();
    let p = n.provenance();
    assert_eq!(p.provider_kind, ProviderKind::Native);
    assert_eq!(p.trust, DistributionTrust::NativeSupported);
    assert_eq!(p.calibration, CalibrationStatus::Exact);
    assert_eq!(p.receipt, Some(ReceiptRef::NativeExecution { execution_id: "exec-1".into() }));

    let unproven = [
        TrustEvidence::None,
        record("native-prov", "other-snapshot", "exec-1"),
        record("other-prov", "snapN", "exec-1"),
        record("native-prov", "snapN", "  "),
    ];
    for evidence in &unproven {
        let input = build("N", &source_n(), evidence, TrustRequirement::Unrestricted).unwrap();
        let p = input.provenance();
        assert_eq!(p.provider_kind, ProviderKind::ExternalAttested);
        assert_eq!(p.trust, DistributionTrust::Unverified);
        assert_eq!(p.calibration, CalibrationStatus::ExactClaimUnverified);
        assert_eq!(p.receipt, None);
        assert_eq!(
            build("N", &source_n(), evidence, TrustRequirement::Native).unwrap_err(),
            BoundaryError::MetadataOnlyNativeClaim { input: "N".into() }
        );
    }
    // Every coordinate of an unmasked law is supported regardless of trust.
    assert_eq!(
        n.support().status_of(&quantity("p", "do(a=1)")),
        Some(SupportStatus::Supported)
    );
}
