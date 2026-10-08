//! Composition boundary: trust carriage, per-action support and source dependence.
//!
//! Hand-derived oracles. The joint fixture has four equally weighted draws with
//! `x0 = [1, 2, 3, 2]` (mean 2) and `x1 = [4, 0, 2, 6]` (mean 3), so with
//! identity utilities `E[U(wait)] = 2`, `E[U(treat)] = 3`, `E[max] = (4 + 2 + 3 + 6) / 4
//! = 3.75` and `EVPI = 0.75`. The mean fixture has `E[Y | do(a=0)] = 1` and
//! `E[Y | do(a=1)] = 3` under the affine utility `2x - 1`, so `wait = 1`, `treat = 5`.

use antecedent_core::{
    DistributionMeaning, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    LawProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity, SupportStatus,
    VerificationProbe, VerificationProbeKind, verify_external_object,
};
use antecedent_design::composition_boundary::{
    ActionStatus, AtomCombination, AtomPlan, BoundaryError, CalibrationStatus,
    CompositionOperation, CoordinateIssue, DecisionInput, DependenceRoute, EvidenceRelation,
    InputSource, NativeExecutionRecord, PairRelation, ProviderKind, ReceiptRef, RouteKind,
    ScalarClaim, SupportPolicy, SupportedVerdict, TrustEvidence, TrustRequirement,
    UnevaluatedReason, UnsupportedReason, check_atom_combination, check_composition,
    check_paired_draws, evaluate_functional_on_input, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, MeanSource, Verdict};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn q(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// Four equally weighted joint draws over `do(a=0)`, `do(a=1)`, `do(a=2)`.
fn joint(
    provider: &str,
    snapshot: &str,
    trust: DistributionTrust,
    calibration: DistributionCalibration,
    mask: Option<Vec<bool>>,
) -> DistributionArtifact {
    let columns = vec![q("do(a=0)", "outcome"), q("do(a=1)", "outcome"), q("do(a=2)", "outcome")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: provider.into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let x0 = [1.0, 2.0, 3.0, 2.0];
    let x1 = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([x0[i], x1[i], 0.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: mask,
            calibration,
            trust,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn labelled_native() -> DistributionArtifact {
    joint(
        "engine",
        "snap-n",
        DistributionTrust::NativeLicensed,
        DistributionCalibration::Exact,
        None,
    )
}

fn attested() -> TrustEvidence {
    TrustEvidence::External(ExternalTrustState::ExternallyAttested { attestor: "lab".into() })
}

fn native_record(provider: &str, snapshot: &str) -> TrustEvidence {
    TrustEvidence::NativeExecution(NativeExecutionRecord {
        execution_id: "exec-1".into(),
        provider_id: provider.into(),
        snapshot_id: snapshot.into(),
    })
}

fn verified(snapshot: &str) -> TrustEvidence {
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "law-1".into(),
            version_id: "v1".into(),
            snapshot_id: snapshot.into(),
            request_id: "req-1".into(),
        },
        quantities: vec![q("do(a=0)", "outcome")],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
    });
    let probes: Vec<VerificationProbe> = [
        VerificationProbeKind::KnownTruth,
        VerificationProbeKind::Shape,
        VerificationProbeKind::Support,
        VerificationProbeKind::Moments,
        VerificationProbeKind::SeededBehavior,
    ]
    .into_iter()
    .map(|kind| VerificationProbe { kind, observed: 1.0, expected: 1.0, tolerance: 0.0 })
    .collect();
    let receipt = verify_external_object(&object, &probes).unwrap();
    TrustEvidence::External(ExternalTrustState::ExactRequestVerified(Box::new(receipt)))
}

fn mean_input(
    id: &str,
    snapshot: &str,
    coordinates: &[(&str, f64)],
    statuses: &[SupportStatus],
) -> DecisionInput {
    DecisionInput::from_mean_source(
        id,
        MeanSource {
            coordinates: coordinates.iter().map(|(regime, _)| q(regime, "mean")).collect(),
            means: coordinates.iter().map(|(_, mean)| *mean).collect(),
            provider_id: "lab".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
        },
        statuses,
        &attested(),
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

fn joint_input(
    id: &str,
    provider: &str,
    snapshot: &str,
    evidence: &TrustEvidence,
) -> DecisionInput {
    DecisionInput::from_distribution_artifact(
        id,
        &joint(
            provider,
            snapshot,
            DistributionTrust::Unverified,
            DistributionCalibration::Unmeasured,
            None,
        ),
        evidence,
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

fn action(id: &str, regime: &str, functional: &str, utility: UtilityExpr) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![q(regime, functional)],
        utility,
    }
}

fn contract(actions: Vec<DecisionAction>) -> DecisionContract {
    DecisionContract {
        actions,
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// `2 * x0 - 1`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
        UtilityExpr::Const(1.0),
    )
}

fn mean_contract() -> DecisionContract {
    contract(vec![
        action("wait", "do(a=0)", "mean", affine()),
        action("treat", "do(a=1)", "mean", affine()),
        action("extend", "do(a=2)", "mean", affine()),
    ])
}

fn joint_contract() -> DecisionContract {
    contract(vec![
        action("wait", "do(a=0)", "outcome", UtilityExpr::Input(0)),
        action("treat", "do(a=1)", "outcome", UtilityExpr::Input(0)),
        action("extend", "do(a=2)", "outcome", UtilityExpr::Input(0)),
    ])
}

const SUPPORTED_TWO: [SupportStatus; 2] = [SupportStatus::Supported, SupportStatus::Supported];

// ---------------------------------------------------------------- trust carriage

#[test]
fn c1_boundary_metadata_only_native_label_is_refused_as_native() {
    let error = DecisionInput::from_distribution_artifact(
        "native",
        &labelled_native(),
        &TrustEvidence::None,
        TrustRequirement::Native,
    )
    .unwrap_err();
    assert_eq!(error, BoundaryError::MetadataOnlyNativeClaim { input: "native".into() });
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "composition_boundary.metadata_only_native_claim");
    assert_eq!(refusal.code, "attested_not_reverifiable");
    assert_eq!(refusal.offending.as_deref(), Some("native"));
    assert!(refusal.validate().is_ok());

    // A record for another snapshot proves nothing about this artifact.
    let wrong = DecisionInput::from_distribution_artifact(
        "native",
        &labelled_native(),
        &native_record("engine", "other-snapshot"),
        TrustRequirement::Native,
    )
    .unwrap_err();
    assert_eq!(wrong, BoundaryError::MetadataOnlyNativeClaim { input: "native".into() });
}

#[test]
fn c1_boundary_unbacked_label_is_stored_unverified_and_its_exact_claim_unused() {
    let input = DecisionInput::from_distribution_artifact(
        "native",
        &labelled_native(),
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let provenance = input.provenance();
    assert_eq!(provenance.provider_kind, ProviderKind::ExternalAttested);
    assert_eq!(provenance.trust, DistributionTrust::Unverified);
    assert_eq!(provenance.receipt, None);
    assert_eq!(provenance.calibration, CalibrationStatus::ExactClaimUnverified);
    let InputSource::JointLaw(stored) = input.source() else { panic!("a joint law") };
    assert_eq!(stored.metadata().calibration, DistributionCalibration::Unmeasured);
    assert_eq!(stored.metadata().trust, DistributionTrust::Unverified);

    // The same artifact with a matching native record keeps its label and exactness,
    // and the lineage digest tells the two apart.
    let native = DecisionInput::from_distribution_artifact(
        "native",
        &labelled_native(),
        &native_record("engine", "snap-n"),
        TrustRequirement::Native,
    )
    .unwrap();
    assert_eq!(native.provenance().provider_kind, ProviderKind::Native);
    assert_eq!(native.provenance().trust, DistributionTrust::NativeLicensed);
    assert_eq!(native.provenance().calibration, CalibrationStatus::Exact);
    assert_eq!(
        native.provenance().receipt,
        Some(ReceiptRef::NativeExecution { execution_id: "exec-1".into() })
    );
    assert_ne!(native.provenance().lineage_digest, provenance.lineage_digest);
}

#[test]
fn c1_boundary_exact_request_verification_needs_a_matching_receipt() {
    let claimed = joint(
        "lab",
        "snap-x",
        DistributionTrust::VerifiedExtension,
        DistributionCalibration::Exact,
        None,
    );
    let build = |evidence: &TrustEvidence| {
        DecisionInput::from_distribution_artifact(
            "ext",
            &claimed,
            evidence,
            TrustRequirement::ExactRequestVerified,
        )
    };
    for missing in [TrustEvidence::None, attested(), verified("another-snapshot")] {
        let error = build(&missing).unwrap_err();
        assert_eq!(error, BoundaryError::VerificationReceiptMissing { input: "ext".into() });
        assert_eq!(error.to_refusal().detail, "composition_boundary.verification_receipt_missing");
        assert_eq!(error.reason_code(), "external_verification_failed");
    }
    let input = build(&verified("snap-x")).unwrap();
    assert_eq!(input.provenance().provider_kind, ProviderKind::ExternalExactRequestVerified);
    assert_eq!(input.provenance().trust, DistributionTrust::VerifiedExtension);
    assert_eq!(input.provenance().calibration, CalibrationStatus::Exact);
    assert_eq!(
        input.provenance().receipt,
        Some(ReceiptRef::Verification { object_id: "law-1".into(), request_id: "req-1".into() })
    );
    // Verified is not native: a native requirement still refuses it.
    let native_needed = DecisionInput::from_distribution_artifact(
        "ext",
        &claimed,
        &verified("snap-x"),
        TrustRequirement::Native,
    )
    .unwrap_err();
    assert!(matches!(native_needed, BoundaryError::NativeRequired { .. }));
}

// ----------------------------------------------------------- per-action support

#[test]
fn c1_boundary_missing_coordinate_beside_supported_action_mean_source() {
    let input = mean_input("grid", "snap-x", &[("do(a=0)", 1.0), ("do(a=1)", 3.0)], &SUPPORTED_TWO);
    let decision =
        evaluate_with_support(&mean_contract(), &[input], &SupportPolicy::compare_supported())
            .unwrap();
    assert_eq!(decision.outcomes.len(), 2);
    assert!(near(decision.outcomes[0].expected_utility, 1.0));
    assert!(near(decision.outcomes[1].expected_utility, 5.0));
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_eq!(decision.evpi, None);
    assert_eq!(decision.dispositions[2].id, "extend");
    assert_eq!(
        decision.dispositions[2].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "grid".into(),
                coordinate: 0,
                issue: CoordinateIssue::NotInSource,
            }],
        }
    );
    assert_eq!(decision.dispositions[0].input_id.as_deref(), Some("grid"));
    assert_eq!(CoordinateIssue::NotInSource.detail(), "composition_boundary.coordinate_missing");

    // When the contract demands every action, the same inputs refuse.
    let input = mean_input("grid", "snap-x", &[("do(a=0)", 1.0), ("do(a=1)", 3.0)], &SUPPORTED_TWO);
    let error = evaluate_with_support(&mean_contract(), &[input], &SupportPolicy::require_all())
        .unwrap_err();
    assert_eq!(error, BoundaryError::UnsupportedActions { actions: vec!["extend".into()] });
    assert_eq!(error.to_refusal().detail, "composition_boundary.unsupported_action_not_comparable");
}

#[test]
fn c1_boundary_missing_coordinate_beside_supported_action_joint_source() {
    let native = DecisionInput::from_distribution_artifact(
        "native",
        &joint(
            "engine",
            "snap-n",
            DistributionTrust::NativeLicensed,
            DistributionCalibration::Exact,
            Some(vec![true, true, false]),
        ),
        &native_record("engine", "snap-n"),
        TrustRequirement::Native,
    )
    .unwrap();
    let decision = evaluate_with_support(
        &joint_contract(),
        std::slice::from_ref(&native),
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert!(near(decision.outcomes[0].expected_utility, 2.0));
    assert!(near(decision.outcomes[1].expected_utility, 3.0));
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    // E[max(x0, x1)] = (4 + 2 + 3 + 6) / 4 and max_a E[U] = 3.
    assert!(near(decision.evpi.unwrap(), 0.75));
    assert_eq!(decision.outcomes[0].standard_error, Some(0.0), "exact native law");
    assert_eq!(
        decision.dispositions[2].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "native".into(),
                coordinate: 0,
                issue: CoordinateIssue::BelowSupport(SupportStatus::OutsideEmpiricalSupport),
            }],
        }
    );
    assert_eq!(decision.results.len(), 1);
    let error = evaluate_with_support(&joint_contract(), &[native], &SupportPolicy::require_all())
        .unwrap_err();
    assert!(matches!(error, BoundaryError::UnsupportedActions { .. }));
}

#[test]
fn c1_boundary_missing_evidence_status_excludes_only_that_action() {
    let input = mean_input(
        "grid",
        "snap-x",
        &[("do(a=0)", 1.0), ("do(a=1)", 3.0)],
        &[SupportStatus::Supported, SupportStatus::MissingEvidence],
    );
    let decision = evaluate_with_support(
        &mean_contract(),
        std::slice::from_ref(&input),
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    // Only `wait` is left: it is reported, but nothing was compared.
    assert_eq!(decision.verdict, SupportedVerdict::OnlyOneEvaluated("wait".into()));
    assert_eq!(decision.outcomes.len(), 1);
    assert!(near(decision.outcomes[0].expected_utility, 1.0));
    assert!(decision.dispositions.iter().skip(1).all(|d| matches!(
        &d.status,
        ActionStatus::Unsupported { reasons } if !reasons.is_empty()
    )));
    assert_eq!(
        decision.dispositions[1].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "grid".into(),
                coordinate: 0,
                issue: CoordinateIssue::BelowSupport(SupportStatus::MissingEvidence),
            }],
        }
    );
}

#[test]
fn c1_boundary_weak_overlap_is_admitted_only_when_the_policy_allows_it() {
    let make = || {
        mean_input(
            "grid",
            "snap-x",
            &[("do(a=0)", 1.0), ("do(a=1)", 3.0)],
            &[SupportStatus::WeakOverlap, SupportStatus::Supported],
        )
    };
    let strict =
        evaluate_with_support(&mean_contract(), &[make()], &SupportPolicy::compare_supported())
            .unwrap();
    assert_eq!(strict.verdict, SupportedVerdict::OnlyOneEvaluated("treat".into()));
    let mut lenient_policy = SupportPolicy::compare_supported();
    lenient_policy.weakest_support = SupportStatus::WeakOverlap;
    let lenient = evaluate_with_support(&mean_contract(), &[make()], &lenient_policy).unwrap();
    assert_eq!(
        lenient.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_eq!(lenient.outcomes.len(), 2);
}

#[test]
fn c1_boundary_all_actions_unsupported_is_a_state_not_an_error() {
    let input = mean_input("grid", "snap-x", &[("do(a=9)", 7.0)], &[SupportStatus::Supported]);
    let decision = evaluate_with_support(
        &mean_contract(),
        std::slice::from_ref(&input),
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(decision.verdict, SupportedVerdict::NoSupportedAction);
    assert!(decision.outcomes.is_empty());
    assert!(decision.results.is_empty());
    assert_eq!(decision.dispositions.len(), 3);
    assert!(
        decision
            .dispositions
            .iter()
            .all(|d| matches!(d.status, ActionStatus::Unsupported { .. }) && d.input_id.is_none())
    );

    // The same holds when every coordinate exists but is masked.
    let masked = DecisionInput::from_distribution_artifact(
        "native",
        &joint(
            "engine",
            "snap-n",
            DistributionTrust::Unverified,
            DistributionCalibration::Unmeasured,
            Some(vec![false, false, false]),
        ),
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let decision =
        evaluate_with_support(&joint_contract(), &[masked], &SupportPolicy::compare_supported())
            .unwrap();
    assert_eq!(decision.verdict, SupportedVerdict::NoSupportedAction);

    // Demanding every action turns it into a refusal.
    let error = evaluate_with_support(&mean_contract(), &[input], &SupportPolicy::require_all())
        .unwrap_err();
    assert!(matches!(error, BoundaryError::UnsupportedActions { actions } if actions.len() == 3));
}

#[test]
fn c1_boundary_actions_answered_by_different_inputs_compare_by_value_only() {
    // `wait` is in the first source only, `treat` in the second only.
    let first = mean_input("first", "snap-1", &[("do(a=0)", 1.0)], &[SupportStatus::Supported]);
    let second = mean_input("second", "snap-2", &[("do(a=1)", 3.0)], &[SupportStatus::Supported]);
    let decision = evaluate_with_support(
        &mean_contract(),
        &[first.clone(), second.clone()],
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_eq!(decision.dispositions[0].input_id.as_deref(), Some("first"));
    assert_eq!(decision.dispositions[1].input_id.as_deref(), Some("second"));
    assert_eq!(decision.evpi, None);

    // A state-aligned criterion cannot span two sources.
    let mut regret = mean_contract();
    regret.criterion = DecisionCriterion::ExpectedRegret;
    let error =
        evaluate_with_support(&regret, &[first, second], &SupportPolicy::compare_supported())
            .unwrap_err();
    assert_eq!(error, BoundaryError::PairedDrawsAcrossSources);
    assert_eq!(error.reason_code(), "joint_law_required");
}

// ------------------------------------------------ mean versus outcome law versus joint

#[test]
fn c1_boundary_a_probability_is_never_answered_from_a_mean() {
    let contract = contract(vec![
        action("wait", "do(a=0)", "mean", UtilityExpr::Input(0)),
        action("treat", "do(a=1)", "mean", UtilityExpr::Input(0)),
    ]);
    let grid = mean_input("grid", "snap-x", &[("do(a=0)", 1.0), ("do(a=1)", 3.0)], &SUPPORTED_TWO);
    let weakest = SupportStatus::Supported;

    // The expectation of an affine utility is the utility of the mean.
    let value = evaluate_functional_on_input(
        &contract,
        "treat",
        DecisionFunctional::Expectation,
        &grid,
        weakest,
    )
    .unwrap();
    assert!(near(value.value, 3.0));
    assert_eq!(value.standard_error, None);

    for functional in [
        DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
        DecisionFunctional::Quantile { p: 0.5 },
        DecisionFunctional::Variance,
        DecisionFunctional::TailExpectation { p: 0.5, tail: Tail::Lower },
    ] {
        let error = evaluate_functional_on_input(&contract, "treat", functional, &grid, weakest)
            .unwrap_err();
        assert_eq!(error, BoundaryError::MeanIsNotADistribution { action: "treat".into() });
        assert_eq!(error.to_refusal().detail, "composition_boundary.mean_is_not_a_distribution");
    }

    // A scalar claim is a mean too.
    let scalar = DecisionInput::from_scalar_claim(
        "scalar",
        ScalarClaim {
            coordinate: q("do(a=1)", "mean"),
            value: 3.0,
            provider_id: "lab".into(),
            snapshot_id: "snap-s".into(),
            causal_contract_id: "checked-contract".into(),
        },
        SupportStatus::Supported,
        &attested(),
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let error = evaluate_functional_on_input(
        &contract,
        "treat",
        DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
        &scalar,
        weakest,
    )
    .unwrap_err();
    assert!(matches!(error, BoundaryError::MeanIsNotADistribution { .. }));
}

#[test]
fn c1_boundary_an_aligned_joint_law_answers_the_probability_a_mean_cannot() {
    let contract = contract(vec![
        action("wait", "do(a=0)", "outcome", UtilityExpr::Input(0)),
        action("treat", "do(a=1)", "outcome", UtilityExpr::Input(0)),
    ]);
    let unbacked = DecisionInput::from_distribution_artifact(
        "law",
        &labelled_native(),
        &TrustEvidence::None,
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    // P(x1 >= 2) over [4, 0, 2, 6] is 3 / 4. The unbacked exact label is not used.
    let probability = evaluate_functional_on_input(
        &contract,
        "treat",
        DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
        &unbacked,
        SupportStatus::Supported,
    )
    .unwrap();
    assert!(near(probability.value, 0.75));
    let expectation = evaluate_functional_on_input(
        &contract,
        "treat",
        DecisionFunctional::Expectation,
        &unbacked,
        SupportStatus::Supported,
    )
    .unwrap();
    assert!(near(expectation.value, 3.0));
    assert_eq!(expectation.standard_error, None, "no exactness without a native record");

    let backed = DecisionInput::from_distribution_artifact(
        "law",
        &labelled_native(),
        &native_record("engine", "snap-n"),
        TrustRequirement::Native,
    )
    .unwrap();
    let exact = evaluate_functional_on_input(
        &contract,
        "treat",
        DecisionFunctional::Expectation,
        &backed,
        SupportStatus::Supported,
    )
    .unwrap();
    assert_eq!(exact.standard_error, Some(0.0));
}

#[test]
fn c1_boundary_mean_sources_leave_nonlinear_and_outcome_law_actions_unevaluated() {
    let grid = mean_input("grid", "snap-x", &[("do(a=0)", 1.0), ("do(a=1)", 3.0)], &SUPPORTED_TWO);
    let nonlinear = DecisionAction {
        id: "risky".into(),
        kind: ActionKind::Intervention,
        inputs: vec![q("do(a=0)", "mean"), q("do(a=1)", "mean")],
        utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
    };
    let outcome_law = action("law", "do(a=0)", "outcome", UtilityExpr::Input(0));
    let contract = contract(vec![
        action("wait", "do(a=0)", "mean", UtilityExpr::Input(0)),
        nonlinear,
        outcome_law,
    ]);
    // The outcome-law coordinate is absent from a mean grid, so it is unsupported,
    // not silently read as a mean.
    let decision = evaluate_with_support(
        &contract,
        std::slice::from_ref(&grid),
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(decision.verdict, SupportedVerdict::OnlyOneEvaluated("wait".into()));
    assert_eq!(
        decision.dispositions[1].status,
        ActionStatus::Unevaluated { reason: UnevaluatedReason::NonAffineUtilityNeedsJointLaw }
    );
    assert!(matches!(decision.dispositions[2].status, ActionStatus::Unsupported { .. }));
    assert_eq!(
        UnevaluatedReason::NonAffineUtilityNeedsJointLaw.detail(),
        "composition_boundary.non_affine_needs_joint_law"
    );
    let error =
        evaluate_with_support(&contract, &[grid], &SupportPolicy::require_all()).unwrap_err();
    assert!(matches!(error, BoundaryError::UnsupportedActions { actions } if actions.len() == 2));

    // A mean supplied for an outcome-law coordinate is present but cannot answer it.
    let labelled_outcome = DecisionInput::from_mean_source(
        "outcome-grid",
        MeanSource {
            coordinates: vec![q("do(a=0)", "outcome"), q("do(a=1)", "mean")],
            means: vec![1.0, 3.0],
            provider_id: "lab".into(),
            snapshot_id: "snap-y".into(),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
        },
        &SUPPORTED_TWO,
        &attested(),
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let decision = evaluate_with_support(
        &contract_of_wait_and_law(),
        &[labelled_outcome],
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(decision.verdict, SupportedVerdict::OnlyOneEvaluated("treat".into()));
    assert_eq!(
        decision.dispositions[0].status,
        ActionStatus::Unevaluated { reason: UnevaluatedReason::OutcomeLawNeedsJointLaw }
    );
}

fn contract_of_wait_and_law() -> DecisionContract {
    contract(vec![
        action("law", "do(a=0)", "outcome", UtilityExpr::Input(0)),
        action("treat", "do(a=1)", "mean", UtilityExpr::Input(0)),
    ])
}

#[test]
fn c1_boundary_a_mean_source_cannot_supply_paired_draws() {
    let error = check_paired_draws(
        &[
            joint_input("native", "engine", "snap-n", &native_record("engine", "snap-n")),
            mean_input("grid", "snap-x", &[("do(a=0)", 1.0)], &[SupportStatus::Supported]),
        ],
        &[],
    )
    .unwrap_err();
    assert!(matches!(error, BoundaryError::Eval(DecisionEvalError::JointLawRequired { .. })));
    assert_eq!(error.reason_code(), "joint_law_required");
}

#[test]
fn c1_boundary_composition_rejects_duplicate_inputs_and_unequal_draw_counts() {
    let first = joint_input("first", "engine", "snap-a", &attested());
    let duplicate = joint_input("first", "lab", "snap-b", &attested());
    let error = check_composition(
        &[first.clone(), duplicate],
        &[],
        Some(CompositionOperation::StatisticalPooling),
    )
    .unwrap_err();
    assert_eq!(error, BoundaryError::InvalidInput("duplicate input id"));

    let mut short = joint(
        "lab",
        "snap-b",
        DistributionTrust::Unverified,
        DistributionCalibration::Unmeasured,
        None,
    );
    let mut metadata = short.metadata().clone();
    metadata.shape[0] -= 1;
    short = DistributionArtifact::new(metadata, short.draws()[..9].to_vec()).unwrap();
    let second = DecisionInput::from_distribution_artifact(
        "second",
        &short,
        &attested(),
        TrustRequirement::Unrestricted,
    )
    .unwrap();
    let relation = PairRelation {
        left: "first".into(),
        right: "second".into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    };
    assert!(
        check_composition(
            &[first.clone(), second.clone()],
            std::slice::from_ref(&relation),
            Some(CompositionOperation::StatisticalPooling),
        )
        .is_ok()
    );
    let error = check_paired_draws(&[first, second], &[relation]).unwrap_err();
    assert_eq!(error, BoundaryError::InvalidInput("paired draw count mismatch"));
}

// ----------------------------------------------------- source overlap and dependence

fn shared_data(left: &str, right: &str, route: Option<DependenceRoute>) -> PairRelation {
    PairRelation {
        left: left.into(),
        right: right.into(),
        relation: EvidenceRelation::SharedData { ids: vec!["trial-1".into()] },
        route,
    }
}

fn native_and_external_mean() -> [DecisionInput; 2] {
    [
        joint_input("native", "engine", "snap-n", &native_record("engine", "snap-n")),
        mean_input("external", "snap-x", &[("do(a=0)", 1.0)], &[SupportStatus::Supported]),
    ]
}

#[test]
fn c1_boundary_shared_data_refuses_independent_pooling() {
    let inputs = native_and_external_mean();
    let error = check_composition(
        &inputs,
        &[shared_data("native", "external", None)],
        Some(CompositionOperation::StatisticalPooling),
    )
    .unwrap_err();
    assert_eq!(
        error,
        BoundaryError::SharedEvidenceNotIndependent {
            left: "native".into(),
            right: "external".into(),
        }
    );
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "composition_boundary.shared_evidence_not_independent");
    assert_eq!(refusal.code, "scenario_aggregate_not_licensed");
    assert_eq!(refusal.offending.as_deref(), Some("native~external"));
    assert!(refusal.validate().is_ok());
}

#[test]
fn c1_boundary_shared_data_refuses_paired_draws_unless_a_joint_law_route_is_licensed() {
    let native = joint_input("native", "engine", "snap-n", &native_record("engine", "snap-n"));
    let external = joint_input("external", "lab", "snap-x", &attested());
    let pair = [native, external];
    let error = check_paired_draws(&pair, &[shared_data("native", "external", None)]).unwrap_err();
    assert!(matches!(error, BoundaryError::SharedEvidenceNotIndependent { .. }));
    // Order of the pair does not matter.
    let reversed =
        check_paired_draws(&pair, &[shared_data("external", "native", None)]).unwrap_err();
    assert!(matches!(reversed, BoundaryError::SharedEvidenceNotIndependent { .. }));

    let route = DependenceRoute {
        kind: RouteKind::JointLaw,
        id: "joint-law-1".into(),
        source_input: "native".into(),
    };
    let receipt =
        check_paired_draws(&pair, &[shared_data("native", "external", Some(route))]).unwrap();
    assert_eq!(receipt.operation, CompositionOperation::StatisticalPooling);
    assert!(!receipt.independence_assumed);
    assert_eq!(receipt.routes, vec!["joint-law-1".to_owned()]);
    assert_eq!(receipt.shared_evidence, vec!["trial-1".to_owned()]);

    let covariance = DependenceRoute {
        kind: RouteKind::Covariance,
        id: "cov-1".into(),
        source_input: "external".into(),
    };
    let receipt =
        check_paired_draws(&pair, &[shared_data("native", "external", Some(covariance))]).unwrap();
    assert_eq!(receipt.routes, vec!["cov-1".to_owned()]);
}

#[test]
fn c1_boundary_a_route_must_be_an_aligned_joint_law_of_the_pair() {
    let inputs = native_and_external_mean();
    let on_mean = DependenceRoute {
        kind: RouteKind::Covariance,
        id: "cov-1".into(),
        source_input: "external".into(),
    };
    let stranger = DependenceRoute {
        kind: RouteKind::JointLaw,
        id: "joint-law-1".into(),
        source_input: "elsewhere".into(),
    };
    let blank_id = DependenceRoute {
        kind: RouteKind::JointLaw,
        id: " ".into(),
        source_input: "native".into(),
    };
    for route in [on_mean, stranger, blank_id] {
        let error = check_composition(
            &inputs,
            &[shared_data("native", "external", Some(route))],
            Some(CompositionOperation::StatisticalPooling),
        )
        .unwrap_err();
        assert_eq!(
            error,
            BoundaryError::DependenceRouteNotLicensed {
                left: "native".into(),
                right: "external".into(),
            }
        );
        assert_eq!(error.to_refusal().detail, "composition_boundary.dependence_route_not_licensed");
    }
    // Where the mean source is the native joint law's partner, a licensed route passes.
    let ok = DependenceRoute {
        kind: RouteKind::JointLaw,
        id: "joint-law-1".into(),
        source_input: "native".into(),
    };
    assert!(
        check_composition(
            &inputs,
            &[shared_data("native", "external", Some(ok))],
            Some(CompositionOperation::StatisticalPooling),
        )
        .is_ok()
    );
}

#[test]
fn c1_boundary_declared_independence_is_overridden_by_a_shared_snapshot() {
    let first =
        mean_input("first", "same-snapshot", &[("do(a=0)", 1.0)], &[SupportStatus::Supported]);
    let second =
        mean_input("second", "same-snapshot", &[("do(a=0)", 1.0)], &[SupportStatus::Supported]);
    let relation = PairRelation {
        left: "first".into(),
        right: "second".into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    };
    let error = check_composition(
        &[first, second],
        &[relation],
        Some(CompositionOperation::StatisticalPooling),
    )
    .unwrap_err();
    assert!(matches!(error, BoundaryError::SharedEvidenceNotIndependent { .. }));
}

#[test]
fn c1_boundary_unknown_dependence_is_never_independence() {
    let inputs = native_and_external_mean();
    let unknown = PairRelation {
        left: "native".into(),
        right: "external".into(),
        relation: EvidenceRelation::UnknownDependence,
        route: None,
    };
    for operation in [
        CompositionOperation::StatisticalPooling,
        CompositionOperation::BayesianBorrowing,
        CompositionOperation::CausalTransport,
        CompositionOperation::EvidenceReuse,
    ] {
        let error = check_composition(&inputs, &[unknown.clone()], Some(operation)).unwrap_err();
        assert_eq!(
            error,
            BoundaryError::UnknownDependence { left: "native".into(), right: "external".into() }
        );
        assert_eq!(
            error.to_refusal().detail,
            "composition_boundary.unknown_dependence_is_not_independence"
        );
        // A pair that was never declared is the same as unknown.
        let silent = check_composition(&inputs, &[], Some(operation)).unwrap_err();
        assert_eq!(silent, error);
    }
    assert_eq!(
        BoundaryError::UnknownDependence { left: "a".into(), right: "b".into() }.reason_code(),
        "sampling_dependence_unknown"
    );
}

#[test]
fn c1_boundary_an_undeclared_operation_is_refused() {
    let inputs = native_and_external_mean();
    let independent = PairRelation {
        left: "native".into(),
        right: "external".into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    };
    let error = check_composition(&inputs, &[independent.clone()], None).unwrap_err();
    assert_eq!(error, BoundaryError::OperationNotDeclared);
    assert_eq!(error.to_refusal().detail, "composition_boundary.operation_not_declared");
    // Independent sources pool, and pooling records that independence was assumed.
    let receipt =
        check_composition(&inputs, &[independent], Some(CompositionOperation::StatisticalPooling))
            .unwrap();
    assert!(receipt.independence_assumed);
    assert!(receipt.routes.is_empty());
    assert!(receipt.shared_evidence.is_empty());

    // Malformed declarations are refused rather than guessed at.
    let stranger = PairRelation {
        left: "native".into(),
        right: "ghost".into(),
        relation: EvidenceRelation::IndependentSources,
        route: None,
    };
    assert_eq!(
        check_composition(&inputs, &[stranger], Some(CompositionOperation::StatisticalPooling))
            .unwrap_err(),
        BoundaryError::UnknownInput("ghost".into())
    );
    assert!(matches!(
        check_composition(&inputs[..1], &[], Some(CompositionOperation::StatisticalPooling)),
        Err(BoundaryError::InvalidInput(_))
    ));
}

#[test]
fn c1_boundary_borrowing_pooling_transport_and_reuse_are_declared_separately() {
    let inputs = native_and_external_mean();
    let relation = |relation: EvidenceRelation| {
        [PairRelation { left: "native".into(), right: "external".into(), relation, route: None }]
    };
    let check = |relation: &[PairRelation], operation| {
        check_composition(&inputs, relation, Some(operation))
    };

    // A shared prior is what Bayesian borrowing is; pooling would count it twice.
    let prior = relation(EvidenceRelation::SharedPrior { id: "prior-7".into() });
    assert!(matches!(
        check(&prior, CompositionOperation::StatisticalPooling),
        Err(BoundaryError::SharedEvidenceNotIndependent { .. })
    ));
    let borrowed = check(&prior, CompositionOperation::BayesianBorrowing).unwrap();
    assert!(!borrowed.independence_assumed);
    assert_eq!(borrowed.shared_evidence, vec!["prior-7".to_owned()]);
    assert!(check(&prior, CompositionOperation::CausalTransport).is_ok());
    assert!(check(&prior, CompositionOperation::EvidenceReuse).is_ok());

    // Shared data is double counting for borrowing and pooling, not for transport or reuse.
    let data =
        relation(EvidenceRelation::SharedData { ids: vec!["trial-2".into(), "trial-1".into()] });
    assert!(matches!(
        check(&data, CompositionOperation::StatisticalPooling),
        Err(BoundaryError::SharedEvidenceNotIndependent { .. })
    ));
    assert!(matches!(
        check(&data, CompositionOperation::BayesianBorrowing),
        Err(BoundaryError::SharedEvidenceNotIndependent { .. })
    ));
    let transported = check(&data, CompositionOperation::CausalTransport).unwrap();
    assert!(!transported.independence_assumed);
    let reused = check(&data, CompositionOperation::EvidenceReuse).unwrap();
    assert_eq!(reused.operation, CompositionOperation::EvidenceReuse);
    assert_eq!(reused.shared_evidence, vec!["trial-1".to_owned(), "trial-2".to_owned()]);

    // A shared fitted model is refused by both statistical operations.
    let model = relation(EvidenceRelation::SharedFittedModel { id: "fit-3".into() });
    for operation in
        [CompositionOperation::StatisticalPooling, CompositionOperation::BayesianBorrowing]
    {
        assert!(matches!(
            check(&model, operation),
            Err(BoundaryError::SharedEvidenceNotIndependent { .. })
        ));
    }
    assert!(check(&model, CompositionOperation::EvidenceReuse).is_ok());
}

// ------------------------------------------------------------ conflicting atoms

fn atom(id: &str, probability: Option<f64>) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability, evidence: AtomEvidence::Unidentified }
}

#[test]
fn c1_boundary_conflicting_atoms_never_become_a_silent_average() {
    let atoms = [atom("graph-a", None), atom("graph-b", None)];
    let error = check_atom_combination(&atoms, AtomCombination::WeightedByDeclaredProbabilities)
        .unwrap_err();
    assert_eq!(error, BoundaryError::ConflictingAtomsNotAveraged);
    assert_eq!(error.to_refusal().detail, "composition_boundary.conflicting_atoms_not_averaged");
    assert_eq!(error.reason_code(), "scenario_aggregate_not_licensed");

    // Probabilities on only some atoms are still not a license to average.
    let partial = [atom("graph-a", Some(0.5)), atom("graph-b", None)];
    assert_eq!(
        check_atom_combination(&partial, AtomCombination::WeightedByDeclaredProbabilities)
            .unwrap_err(),
        BoundaryError::ConflictingAtomsNotAveraged
    );

    // Reporting each atom and the worst case need no probability.
    assert_eq!(
        check_atom_combination(&atoms, AtomCombination::ReportEach).unwrap(),
        AtomPlan::ReportEach
    );
    assert_eq!(
        check_atom_combination(&atoms, AtomCombination::WorstCase).unwrap(),
        AtomPlan::WorstCase
    );

    // Declared probabilities license the weighted plan; missing mass is not renormalized.
    let declared = [atom("graph-a", Some(0.25)), atom("graph-b", Some(0.5))];
    assert_eq!(
        check_atom_combination(&declared, AtomCombination::WeightedByDeclaredProbabilities)
            .unwrap(),
        AtomPlan::Weighted(vec![("graph-a".into(), 0.25), ("graph-b".into(), 0.5)])
    );

    // Invalid probabilities and repeated identities refuse.
    let over = [atom("graph-a", Some(0.75)), atom("graph-b", Some(0.5))];
    assert_eq!(
        check_atom_combination(&over, AtomCombination::WeightedByDeclaredProbabilities)
            .unwrap_err(),
        BoundaryError::InvalidAtomProbabilities
    );
    let repeated = [atom("graph-a", None), atom("graph-a", None)];
    assert!(matches!(
        check_atom_combination(&repeated, AtomCombination::ReportEach),
        Err(BoundaryError::InvalidInput(_))
    ));
}
