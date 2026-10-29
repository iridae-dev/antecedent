//! The callback protocol: what is decided from declarations, identities and
//! stored records alone, before any provider runs.
//!
//! Every expectation is a hand-enumerated outcome of the rule it names. There
//! is no provider in this file: nothing here can invoke a callback.

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::callback_protocol::{
    CALLBACK_OUTPUT_PORT, CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind,
    CallbackOperation, CallbackOutput, CallbackPlan, CallbackQuery, CallbackRequest, CallbackTrust,
    DependencyDeclaration, DependencyKind, InputSupplier, RecomputeReason, ReplayContract,
    ResultStanding, RetryContract, StoredCallback, UpdateDeclaration, callback_source_declaration,
    plan_reuse, semantic_stream, standing_label, standing_of, validate_output,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, SupportDomain, check_connection,
};
use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalCapabilityRequest, ExternalContractError,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};

fn quantity(population: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn identity(version: &str) -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "gaussian".into(),
        version_id: version.into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    }
}

fn gaussian() -> CallbackDeclaration {
    CallbackDeclaration {
        component: "gaussian_law".into(),
        kind: CallbackKind::Law,
        identity: identity("1.0"),
        quantity: quantity("adults", "do(a=1)"),
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
            CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
        ]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn with_ops(ops: &[CallbackOperation]) -> CallbackDeclaration {
    let mut declaration = gaussian();
    declaration.operations = ops.iter().copied().collect();
    declaration
}

fn posterior() -> CallbackDeclaration {
    let mut declaration = gaussian();
    declaration.component = "posterior".into();
    declaration.kind = CallbackKind::Posterior;
    declaration.meaning = DistributionMeaning::CausalFunctionalPosterior;
    declaration.operations =
        [CallbackOperation::Mean, CallbackOperation::Sample, CallbackOperation::PosteriorUpdate]
            .into_iter()
            .collect();
    declaration.update = Some(UpdateDeclaration {
        candidate_id: "study-a".into(),
        prior_id: "prior-0".into(),
        observation: quantity("adults", "observational"),
    });
    declaration
}

fn detail(refusal: &antecedent_contract::ExternalRefusal) -> &str {
    &refusal.detail
}

fn refusal_of(declaration: &CallbackDeclaration) -> antecedent_contract::ExternalRefusal {
    declaration.validate().expect_err("the declaration must be refused")
}

// -- declaration -----------------------------------------------------------------------------------

#[test]
fn a_complete_declaration_validates_and_is_an_instance_of_the_interface_contract() {
    let declaration = gaussian();
    declaration.validate().expect("well formed");
    let contract = declaration.interface_contract();
    contract.validate().expect("the projection is a valid interface contract");
    assert_eq!(contract.component, "gaussian_law");
    let port = contract.produced(CALLBACK_OUTPUT_PORT).expect("one produced port");
    // mean -> mean, sample -> draws, cdf and quantile -> law
    let expected: BTreeSet<ContractOperation> =
        [ContractOperation::Mean, ContractOperation::Draws, ContractOperation::Law]
            .into_iter()
            .collect();
    assert_eq!(port.operations, expected);
    assert_eq!(port.calibration, CalibrationStatus::Unmeasured);
    assert_eq!(
        port.uncertainty.map(|u| u.meaning),
        Some(DistributionMeaning::InterventionalPredictive)
    );
    // The callback never holds a measured calibration record.
    assert!(!port.calibration.is_measured());
}

#[test]
fn a_mean_only_callback_is_point_only_and_carries_no_uncertainty() {
    let declaration = with_ops(&[CallbackOperation::Mean]);
    declaration.validate().expect("well formed");
    let port = declaration.interface_contract().produces.remove(0);
    assert_eq!(port.calibration, CalibrationStatus::PointOnly);
    assert!(port.uncertainty.is_none());
    assert_eq!(port.operations, [ContractOperation::Mean].into_iter().collect());
}

#[test]
fn operations_are_negotiated_separately_and_none_implies_another() {
    let object_of = |ops: &[CallbackOperation]| with_ops(ops).provider_object();
    let request = |operation: ExternalCapability| ExternalCapabilityRequest {
        expected_identity: identity("1.0"),
        operation,
    };
    // Sampling does not supply a distribution function or a quantile function.
    let sample_only = object_of(&[CallbackOperation::Sample]);
    for missing in [ExternalCapability::Cdf, ExternalCapability::Quantile, ExternalCapability::Mean]
    {
        assert_eq!(
            sample_only.negotiate(&request(missing)).unwrap_err(),
            ExternalContractError::MissingCapability(missing)
        );
    }
    assert!(sample_only.negotiate(&request(ExternalCapability::Sample)).is_ok());
    // A mean supplies no law and no draws.
    let mean_only = object_of(&[CallbackOperation::Mean]);
    for missing in
        [ExternalCapability::Sample, ExternalCapability::Cdf, ExternalCapability::Quantile]
    {
        assert_eq!(
            mean_only.negotiate(&request(missing)).unwrap_err(),
            ExternalContractError::MissingCapability(missing)
        );
    }
    // A distribution function does not supply a quantile function, and the
    // protocol keeps them apart although both disclose a law.
    let cdf_only = with_ops(&[CallbackOperation::Cdf]);
    assert!(cdf_only.operations.contains(&CallbackOperation::Cdf));
    assert!(!cdf_only.operations.contains(&CallbackOperation::Quantile));
    assert_eq!(
        cdf_only.provider_object().negotiate(&request(ExternalCapability::Quantile)).unwrap_err(),
        ExternalContractError::MissingCapability(ExternalCapability::Quantile)
    );
    assert_eq!(
        CallbackOperation::Cdf.contract_operation(),
        CallbackOperation::Quantile.contract_operation()
    );
    // A different request identity is refused at negotiation.
    let mut other = request(ExternalCapability::Mean);
    other.expected_identity.request_id = "req-2".into();
    assert_eq!(
        gaussian().provider_object().negotiate(&other).unwrap_err(),
        ExternalContractError::RequestMismatch
    );
}

#[test]
fn a_posterior_update_is_a_separate_signal_that_needs_its_observation_law() {
    let declaration = posterior();
    declaration.validate().expect("well formed");
    let signal = declaration.update_object().expect("the update bridges to a signal object");
    signal.validate().expect("a valid signal contract");
    assert!(
        signal
            .negotiate(&ExternalCapabilityRequest {
                expected_identity: identity("1.0"),
                operation: ExternalCapability::Update,
            })
            .is_ok()
    );
    // The posterior object itself answers mean and sample, not the update.
    assert_eq!(
        declaration
            .provider_object()
            .negotiate(&ExternalCapabilityRequest {
                expected_identity: identity("1.0"),
                operation: ExternalCapability::Update,
            })
            .unwrap_err(),
        ExternalContractError::MissingCapability(ExternalCapability::Update)
    );
    // An update without the observation law is refused, as in the provider contract.
    let mut no_law = posterior();
    no_law.operations.remove(&CallbackOperation::Sample);
    assert_eq!(detail(&refusal_of(&no_law)), "callback_protocol.update_declaration_mismatch");
}

#[test]
fn malformed_declarations_are_refused_with_registered_details() {
    let code = antecedent_core::reason_code!("callback_protocol_invalid");
    let mut cases: Vec<(CallbackDeclaration, &str)> = Vec::new();
    let mut blank = gaussian();
    blank.identity.version_id = "  ".into();
    cases.push((blank, "callback_protocol.blank_field"));
    let mut none = gaussian();
    none.operations.clear();
    cases.push((none, "callback_protocol.no_operations"));
    let mut law_update = gaussian();
    law_update.operations.insert(CallbackOperation::PosteriorUpdate);
    cases.push((law_update, "callback_protocol.operation_not_for_kind"));
    let mut posterior_cdf = posterior();
    posterior_cdf.operations.insert(CallbackOperation::Cdf);
    cases.push((posterior_cdf, "callback_protocol.operation_not_for_kind"));
    let mut law_meaning = gaussian();
    law_meaning.meaning = DistributionMeaning::ParameterPosterior;
    cases.push((law_meaning, "callback_protocol.meaning_not_for_kind"));
    let mut posterior_meaning = posterior();
    posterior_meaning.meaning = DistributionMeaning::EstimatorSampling;
    cases.push((posterior_meaning, "callback_protocol.meaning_not_for_kind"));
    let mut update_missing = posterior();
    update_missing.update = None;
    cases.push((update_missing, "callback_protocol.update_declaration_mismatch"));
    let mut stray_update = gaussian();
    stray_update.update = posterior().update;
    cases.push((stray_update, "callback_protocol.update_declaration_mismatch"));
    let mut unnamed_stream = gaussian();
    unnamed_stream.replay = ReplayContract::SeededStochastic { stream_id: " ".into() };
    cases.push((unnamed_stream, "callback_protocol.replay_contract_incomplete"));
    let mut zero_attempts = gaussian();
    zero_attempts.replay =
        ReplayContract::SideEffecting { retry: RetryContract::IdempotentKey { max_attempts: 0 } };
    cases.push((zero_attempts, "callback_protocol.replay_contract_incomplete"));
    let mut duplicate = gaussian();
    duplicate.dependencies = DependencyDeclaration::Complete(vec![
        CallbackDependency::new(DependencyKind::Data, "table", "v1"),
        CallbackDependency::new(DependencyKind::Data, "table", "v2"),
    ]);
    cases.push((duplicate, "callback_protocol.duplicate_name"));
    let mut categorical = gaussian();
    categorical.support = SupportDomain::Categorical { levels: vec!["a".into(), "b".into()] };
    cases.push((categorical, "callback_protocol.invalid_support"));
    let mut nested_without_inputs = gaussian();
    nested_without_inputs.kind = CallbackKind::Nested;
    cases.push((nested_without_inputs, "callback_protocol.nested_inputs_mismatch"));
    let mut trust_blank = gaussian();
    trust_blank.trust = CallbackTrust::ExternallyAttested { attestor: " ".into() };
    cases.push((trust_blank, "callback_protocol.trust_incomplete"));
    for (declaration, expected) in cases {
        let refusal = refusal_of(&declaration);
        assert_eq!(refusal.code, code, "{expected}");
        assert_eq!(detail(&refusal), expected);
        assert!(refusal.remedy.is_some(), "{expected} names what would license it");
        refusal.validate().expect("a well-formed refusal envelope");
    }
}

#[test]
fn identity_binds_dependency_versions_replay_and_provider_but_not_order() {
    let base = gaussian();
    let id = base.canonical_identity();
    // A changed dependency version is a different callback.
    let mut changed = gaussian();
    changed.dependencies = DependencyDeclaration::Complete(vec![
        CallbackDependency::new(DependencyKind::Model, "parameters", "mu=2;sd=1.6"),
        CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
    ]);
    assert_ne!(changed.canonical_identity(), id);
    // Dependency declaration order does not matter.
    let mut reordered = gaussian();
    reordered.dependencies = DependencyDeclaration::Complete(vec![
        CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
        CallbackDependency::new(DependencyKind::Model, "parameters", "mu=2;sd=1.5"),
    ]);
    assert_eq!(reordered.canonical_identity(), id);
    // Replay contract, provider version and request, trust and evidence all bind.
    let mut seeded = gaussian();
    seeded.replay = ReplayContract::SeededStochastic { stream_id: "draws".into() };
    assert_ne!(seeded.canonical_identity(), id);
    let mut replaced = gaussian();
    replaced.identity = identity("2.0");
    assert_ne!(replaced.canonical_identity(), id);
    let mut other_request = gaussian();
    other_request.identity.request_id = "req-2".into();
    assert_ne!(other_request.canonical_identity(), id);
    let mut verified = gaussian();
    verified.trust = CallbackTrust::ExactRequestVerified { receipt_digest: "d".repeat(64) };
    assert_ne!(verified.canonical_identity(), id);
    let mut opaque = gaussian();
    opaque.dependencies = DependencyDeclaration::Opaque;
    let mut undeclared = gaussian();
    undeclared.dependencies = DependencyDeclaration::Undeclared;
    assert_ne!(opaque.canonical_identity(), undeclared.canonical_identity());
}

// -- requests, streams and reuse ---------------------------------------------------------------------

fn stored_of(
    declaration: &CallbackDeclaration,
    request: &CallbackRequest,
    state: Option<&str>,
) -> StoredCallback {
    StoredCallback {
        declaration_identity: declaration.canonical_identity(),
        request_identity: request.identity(declaration),
        dependencies: declaration
            .dependencies
            .complete()
            .map(<[CallbackDependency]>::to_vec)
            .unwrap_or_default(),
        state_identity: state.map(str::to_owned),
    }
}

#[test]
fn request_identity_ignores_the_seed_of_a_deterministic_callback_only() {
    let pure = gaussian();
    let a = CallbackRequest::new(CallbackQuery::Mean, 1);
    let b = CallbackRequest::new(CallbackQuery::Mean, 2);
    assert_eq!(a.identity(&pure), b.identity(&pure));
    let mut seeded = gaussian();
    seeded.replay = ReplayContract::SeededStochastic { stream_id: "draws".into() };
    assert_ne!(a.identity(&seeded), b.identity(&seeded));
    // The query is part of the identity down to the bit.
    let cdf = |t: f64| CallbackRequest::new(CallbackQuery::Cdf { thresholds: vec![t] }, 1);
    assert_ne!(
        cdf(1.0).identity(&pure),
        cdf(f64::from_bits(1.0_f64.to_bits() + 1)).identity(&pure)
    );
    // Input digests and the idempotency key bind.
    let mut with_input = a.clone();
    with_input.input_digests.insert("inner".into(), "aa".into());
    assert_ne!(with_input.identity(&pure), a.identity(&pure));
    let mut keyed = a.clone();
    keyed.idempotency_key = Some("k".into());
    assert_ne!(keyed.identity(&pure), a.identity(&pure));
}

#[test]
fn semantic_streams_depend_on_the_seed_and_the_identity_not_on_call_order() {
    let draws = |seed: u64, stream: &str, skip_calls: usize| {
        for _ in 0..skip_calls {
            let _ = semantic_stream(seed, "unrelated").next_u64();
        }
        let mut rng = semantic_stream(seed, stream);
        (0..4).map(|_| rng.next_u64()).collect::<Vec<_>>()
    };
    let base = draws(7, "outcome_draws", 0);
    assert_eq!(base, draws(7, "outcome_draws", 5), "call order does not move a stream");
    assert_ne!(base, draws(8, "outcome_draws", 0), "another seed is another stream");
    assert_ne!(base, draws(7, "noise_draws", 0), "another semantic identity is another stream");
}

#[test]
fn reuse_follows_the_replay_contract_and_every_identity() {
    let request = CallbackRequest::new(CallbackQuery::Mean, 1);
    // Deterministic pure.
    let pure = gaussian();
    assert_eq!(
        plan_reuse(&pure, &request, None, None),
        CallbackPlan::Recompute(RecomputeReason::NoPriorResult)
    );
    let stored = stored_of(&pure, &request, None);
    assert_eq!(plan_reuse(&pure, &request, Some(&stored), None), CallbackPlan::Reuse);
    // A changed dependency version names the dependency and invalidates.
    let mut changed = gaussian();
    changed.dependencies = DependencyDeclaration::Complete(vec![
        CallbackDependency::new(DependencyKind::Model, "parameters", "mu=3;sd=1.5"),
        CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
    ]);
    assert_eq!(
        plan_reuse(&changed, &request, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::DependencyChanged {
            name: "parameters".into(),
            was: "mu=2;sd=1.5".into(),
            now: "mu=3;sd=1.5".into(),
        })
    );
    // A replaced provider is a changed declaration no dependency explains.
    let mut replaced = gaussian();
    replaced.identity = identity("2.0");
    assert_eq!(
        plan_reuse(&replaced, &request, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::DeclarationChanged)
    );
    // A different request recomputes.
    let other = CallbackRequest::new(CallbackQuery::Sample { draws: 3 }, 1);
    assert_eq!(
        plan_reuse(&pure, &other, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::RequestChanged)
    );
}

#[test]
fn opaque_or_undeclared_dependencies_make_reuse_ineligible() {
    let request = CallbackRequest::new(CallbackQuery::Mean, 1);
    for dependencies in [DependencyDeclaration::Opaque, DependencyDeclaration::Undeclared] {
        let mut declaration = gaussian();
        declaration.dependencies = dependencies;
        // Even an identical stored record is not reused: there is no stable identity.
        let stored = stored_of(&declaration, &request, None);
        assert_eq!(
            plan_reuse(&declaration, &request, Some(&stored), None),
            CallbackPlan::Recompute(RecomputeReason::OpaqueDependencies)
        );
        // A side-effecting callback is never recomputed to find out.
        declaration.replay = ReplayContract::SideEffecting {
            retry: RetryContract::IdempotentKey { max_attempts: 3 },
        };
        let plan = plan_reuse(&declaration, &request, Some(&stored), None);
        let CallbackPlan::Refuse(refusal) = plan else { panic!("must refuse, got {plan:?}") };
        assert_eq!(refusal.detail, "callback_protocol.opaque_dependencies");
        assert_eq!(refusal.code, antecedent_core::reason_code!("callback_reuse_ineligible"));
    }
}

#[test]
fn a_seeded_callback_is_reused_only_for_the_same_seed() {
    let mut seeded = gaussian();
    seeded.replay = ReplayContract::SeededStochastic { stream_id: "draws".into() };
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 11);
    let stored = stored_of(&seeded, &request, None);
    assert_eq!(plan_reuse(&seeded, &request, Some(&stored), None), CallbackPlan::Reuse);
    let other_seed = CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 12);
    assert_eq!(
        plan_reuse(&seeded, &other_seed, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::RequestChanged)
    );
}

#[test]
fn a_stateful_callback_is_reused_only_while_its_reported_state_is_unchanged() {
    let request = CallbackRequest::new(CallbackQuery::Mean, 1);
    let mut stateful = gaussian();
    stateful.replay = ReplayContract::Stateful { exposes_state_identity: true };
    let stored = stored_of(&stateful, &request, Some("weights-v1"));
    assert_eq!(
        plan_reuse(&stateful, &request, Some(&stored), Some("weights-v1")),
        CallbackPlan::Reuse
    );
    assert_eq!(
        plan_reuse(&stateful, &request, Some(&stored), Some("weights-v2")),
        CallbackPlan::Recompute(RecomputeReason::StateChanged)
    );
    assert_eq!(
        plan_reuse(&stateful, &request, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::StateUnknown)
    );
    // Hidden state that the protocol cannot observe is never shown current.
    let mut hidden = gaussian();
    hidden.replay = ReplayContract::Stateful { exposes_state_identity: false };
    let stored = stored_of(&hidden, &request, None);
    assert_eq!(
        plan_reuse(&hidden, &request, Some(&stored), None),
        CallbackPlan::Recompute(RecomputeReason::HiddenStateUnobservable)
    );
}

#[test]
fn a_side_effecting_callback_is_reused_only_under_an_idempotent_contract_and_key() {
    let mut no_retry = gaussian();
    no_retry.replay = ReplayContract::SideEffecting { retry: RetryContract::NoRetry };
    let request = CallbackRequest::new(CallbackQuery::Mean, 1);
    let stored = stored_of(&no_retry, &request, None);
    let CallbackPlan::Refuse(refusal) = plan_reuse(&no_retry, &request, Some(&stored), None) else {
        panic!("a side-effecting callback with no retry contract is refused");
    };
    assert_eq!(refusal.detail, "callback_protocol.no_idempotency_contract");
    // First execution, with nothing stored, is a recompute decision; the
    // executor's policy decides whether effects are allowed.
    assert_eq!(
        plan_reuse(&no_retry, &request, None, None),
        CallbackPlan::Recompute(RecomputeReason::NoPriorResult)
    );
    let mut keyed = gaussian();
    keyed.replay =
        ReplayContract::SideEffecting { retry: RetryContract::IdempotentKey { max_attempts: 2 } };
    let mut with_key = CallbackRequest::new(CallbackQuery::Mean, 1);
    with_key.idempotency_key = Some("order-17".into());
    let stored = stored_of(&keyed, &with_key, None);
    assert_eq!(plan_reuse(&keyed, &with_key, Some(&stored), None), CallbackPlan::Reuse);
    // The same declaration without a key has no idempotency to rely on.
    let no_key = CallbackRequest::new(CallbackQuery::Mean, 1);
    let stored = stored_of(&keyed, &no_key, None);
    assert!(matches!(plan_reuse(&keyed, &no_key, Some(&stored), None), CallbackPlan::Refuse(_)));
}

#[test]
fn standing_is_diagnostic_unless_replayable_complete_and_attested() {
    assert_eq!(standing_label(&gaussian(), false), ResultStanding::LicensedExternal);
    let mut seeded = gaussian();
    seeded.replay = ReplayContract::SeededStochastic { stream_id: "draws".into() };
    assert_eq!(standing_label(&seeded, false), ResultStanding::LicensedExternal);
    let diagnostic = |mutate: &dyn Fn(&mut CallbackDeclaration), reason: &str| {
        let mut declaration = gaussian();
        mutate(&mut declaration);
        let refusal = standing_of(&declaration, true).expect_err("diagnostic");
        assert_eq!(refusal.detail, "callback_protocol.diagnostic_only");
        assert_eq!(refusal.offending.as_deref(), Some(reason));
        assert_eq!(standing_label(&declaration, true), ResultStanding::Diagnostic);
    };
    diagnostic(&|d| d.dependencies = DependencyDeclaration::Opaque, "opaque_dependencies");
    diagnostic(
        &|d| d.replay = ReplayContract::Stateful { exposes_state_identity: true },
        "stateful",
    );
    diagnostic(
        &|d| d.replay = ReplayContract::SideEffecting { retry: RetryContract::NoRetry },
        "side_effecting",
    );
    diagnostic(&|d| d.trust = CallbackTrust::Unverified, "unverified");
    // A declared verification is believed only when confirmed against a receipt.
    let mut verified = gaussian();
    verified.trust = CallbackTrust::ExactRequestVerified { receipt_digest: "d".repeat(64) };
    assert_eq!(standing_label(&verified, false), ResultStanding::Diagnostic);
    assert_eq!(standing_label(&verified, true), ResultStanding::LicensedExternal);
}

// -- output validation ------------------------------------------------------------------------

fn accepted(query: &CallbackQuery, output: CallbackOutput, support: &SupportDomain) -> bool {
    validate_output(query, output, support, 1_000).is_ok()
}

fn refused_detail(
    query: &CallbackQuery,
    output: CallbackOutput,
    support: &SupportDomain,
) -> String {
    validate_output(query, output, support, 1_000).expect_err("refused").detail
}

#[test]
fn a_well_formed_law_is_accepted_and_every_malformed_law_is_refused() {
    let real = SupportDomain::Real;
    let cdf = CallbackQuery::Cdf { thresholds: vec![-1.0, 0.0, 1.0] };
    assert!(accepted(&cdf, CallbackOutput::Cdf { values: vec![0.1, 0.5, 0.9] }, &real));
    assert_eq!(
        refused_detail(&cdf, CallbackOutput::Cdf { values: vec![0.1, 0.9, 0.5] }, &real),
        "callback_protocol.output_not_a_law"
    );
    assert_eq!(
        refused_detail(&cdf, CallbackOutput::Cdf { values: vec![0.1, 0.5, 1.2] }, &real),
        "callback_protocol.output_not_a_law"
    );
    assert_eq!(
        refused_detail(&cdf, CallbackOutput::Cdf { values: vec![0.1, f64::NAN, 0.9] }, &real),
        "callback_protocol.output_not_finite"
    );
    assert_eq!(
        refused_detail(&cdf, CallbackOutput::Cdf { values: vec![0.1, 0.5] }, &real),
        "callback_protocol.output_shape"
    );
    assert_eq!(
        refused_detail(&cdf, CallbackOutput::Mean { value: 0.0 }, &real),
        "callback_protocol.output_shape"
    );
    // A distribution function must put no mass below the support.
    let positive = SupportDomain::NonNegative;
    let below = CallbackQuery::Cdf { thresholds: vec![-1.0, 1.0] };
    assert!(accepted(&below, CallbackOutput::Cdf { values: vec![0.0, 0.4] }, &positive));
    assert_eq!(
        refused_detail(&below, CallbackOutput::Cdf { values: vec![0.2, 0.4] }, &positive),
        "callback_protocol.output_not_a_law"
    );
    // And all of it at or above the upper end of a bounded support.
    let unit = SupportDomain::interval(0.0, 1.0);
    let top = CallbackQuery::Cdf { thresholds: vec![0.5, 1.0] };
    assert!(accepted(&top, CallbackOutput::Cdf { values: vec![0.5, 1.0] }, &unit));
    assert_eq!(
        refused_detail(&top, CallbackOutput::Cdf { values: vec![0.5, 0.9] }, &unit),
        "callback_protocol.output_not_a_law"
    );
}

#[test]
fn quantiles_draws_and_means_are_checked_against_the_support() {
    let unit = SupportDomain::interval(0.0, 1.0);
    let quantile = CallbackQuery::Quantile { levels: vec![0.1, 0.5, 0.9] };
    assert!(accepted(&quantile, CallbackOutput::Quantile { values: vec![0.1, 0.5, 0.9] }, &unit));
    assert_eq!(
        refused_detail(&quantile, CallbackOutput::Quantile { values: vec![0.5, 0.4, 0.9] }, &unit),
        "callback_protocol.output_not_a_law"
    );
    assert_eq!(
        refused_detail(&quantile, CallbackOutput::Quantile { values: vec![0.1, 0.5, 1.5] }, &unit),
        "callback_protocol.output_outside_support"
    );
    let sample = CallbackQuery::Sample { draws: 2 };
    assert_eq!(
        refused_detail(&sample, CallbackOutput::Sample { draws: vec![0.2, -0.1] }, &unit),
        "callback_protocol.output_outside_support"
    );
    // A binary quantity takes only 0 and 1; its mean is a probability.
    let binary = SupportDomain::Binary;
    assert!(accepted(&sample, CallbackOutput::Sample { draws: vec![0.0, 1.0] }, &binary));
    assert_eq!(
        refused_detail(&sample, CallbackOutput::Sample { draws: vec![0.0, 0.5] }, &binary),
        "callback_protocol.output_outside_support"
    );
    assert!(accepted(&CallbackQuery::Mean, CallbackOutput::Mean { value: 0.3 }, &binary));
    assert_eq!(
        refused_detail(&CallbackQuery::Mean, CallbackOutput::Mean { value: 1.5 }, &binary),
        "callback_protocol.output_outside_support"
    );
    let update = CallbackQuery::PosteriorUpdate { observation: 2.0, draws: 2 };
    assert!(accepted(
        &update,
        CallbackOutput::PosteriorUpdate { mean: 1.6, draws: vec![1.0, 2.0] },
        &SupportDomain::Real
    ));
    assert_eq!(
        refused_detail(
            &update,
            CallbackOutput::PosteriorUpdate { mean: 1.6, draws: vec![1.0] },
            &SupportDomain::Real
        ),
        "callback_protocol.output_shape"
    );
}

#[test]
fn queries_and_sizes_are_bounded_and_checked() {
    let real = SupportDomain::Real;
    for query in [
        CallbackQuery::Sample { draws: 0 },
        CallbackQuery::Cdf { thresholds: vec![1.0, 1.0] },
        CallbackQuery::Cdf { thresholds: vec![f64::INFINITY] },
        CallbackQuery::Quantile { levels: vec![0.0] },
        CallbackQuery::Quantile { levels: vec![0.5, 0.4] },
        CallbackQuery::PosteriorUpdate { observation: f64::NAN, draws: 0 },
    ] {
        let refusal = query.validate().expect_err("malformed query");
        assert_eq!(refusal.detail, "callback_protocol.query_invalid");
        let attempt =
            validate_output(&query, CallbackOutput::Mean { value: 0.0 }, &real, 10).unwrap_err();
        assert_eq!(attempt.detail, "callback_protocol.query_invalid");
    }
    // A larger answer than the bound is a limit refusal, not an invalid law.
    let query = CallbackQuery::Sample { draws: 11 };
    let refusal =
        validate_output(&query, CallbackOutput::Sample { draws: vec![0.0; 11] }, &real, 10)
            .expect_err("too large");
    assert_eq!(refusal.detail, "callback_protocol.output_too_large");
    assert_eq!(refusal.code, antecedent_core::reason_code!("callback_limit_exceeded"));
}

// -- nested callbacks and the evidence ledger -------------------------------------------------------------

fn observations(label: &str, rows: &[&str]) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(
        rows.iter().map(|row| ObservationId::from_key("test.callbacks", row.as_bytes())),
    )
}

fn nested_over(
    component: &str,
    inner: &CallbackDeclaration,
    own: EvidenceBasis,
) -> CallbackDeclaration {
    let mut outer = gaussian();
    outer.component = component.into();
    outer.kind = CallbackKind::Nested;
    outer.operations = [CallbackOperation::Mean].into_iter().collect();
    outer.evidence = own;
    outer.inputs = vec![CallbackInput {
        name: "inner".into(),
        quantity: inner.quantity.clone(),
        semantic_identity: inner.semantic_identity.clone(),
        support: inner.support.clone(),
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::Callback {
            identity: inner.canonical_identity().to_hex(),
            evidence: inner.effective_evidence(),
        },
    }];
    outer
}

fn register(
    ledger: &mut EvidenceLedger,
    declaration: &CallbackDeclaration,
    inner: &BTreeMap<String, SourceId>,
) -> SourceId {
    declaration.validate().expect("registrable");
    let source = callback_source_declaration(declaration, inner).expect("declaration");
    ledger.add_source(source).expect("ledger accepts it")
}

#[test]
fn nested_callbacks_are_registered_with_their_evidence_overlap() {
    let mut ledger = EvidenceLedger::new();
    let trial = ledger.add_source(observations("trial", &["r1", "r2", "r3"])).unwrap();
    let registry = ledger.add_source(observations("registry", &["g1", "g2"])).unwrap();
    // Two inner callbacks: one fitted on the trial, one on the registry.
    let mut inner_trial = gaussian();
    inner_trial.component = "inner_trial".into();
    inner_trial.evidence = EvidenceBasis::Declared([trial].into_iter().collect());
    let mut inner_registry = gaussian();
    inner_registry.component = "inner_registry".into();
    inner_registry.evidence = EvidenceBasis::Declared([registry].into_iter().collect());
    let trial_source = register(&mut ledger, &inner_trial, &BTreeMap::new());
    let registry_source = register(&mut ledger, &inner_registry, &BTreeMap::new());
    let inner_ids: BTreeMap<String, SourceId> = [
        (inner_trial.canonical_identity().to_hex(), trial_source),
        (inner_registry.canonical_identity().to_hex(), registry_source),
    ]
    .into_iter()
    .collect();
    // Two outer callbacks over the same inner callback share its observations;
    // one over the registry callback is independent of them.
    let outer_a = nested_over("outer_a", &inner_trial, EvidenceBasis::Declared(BTreeSet::new()));
    let outer_b = nested_over("outer_b", &inner_trial, EvidenceBasis::Declared(BTreeSet::new()));
    let outer_c = nested_over("outer_c", &inner_registry, EvidenceBasis::Declared(BTreeSet::new()));
    // The effective evidence of an outer callback is its inner callbacks'.
    assert_eq!(
        outer_a.effective_evidence(),
        EvidenceBasis::Declared([trial].into_iter().collect())
    );
    let a = register(&mut ledger, &outer_a, &inner_ids);
    let b = register(&mut ledger, &outer_b, &inner_ids);
    let c = register(&mut ledger, &outer_c, &inner_ids);
    assert_eq!(
        ledger.overlap(a, b).unwrap(),
        Overlap::SharedObservations { shared: 3 },
        "outer callbacks over one inner callback share its three trial rows"
    );
    assert_eq!(ledger.overlap(a, c).unwrap(), Overlap::Independent);
    assert_eq!(ledger.overlap(a, trial_source).unwrap(), Overlap::SharedObservations { shared: 3 });
    // An outer callback with undeclared own evidence overlaps with unknown
    // dependence, never independence.
    let outer_u = nested_over("outer_u", &inner_trial, EvidenceBasis::Undeclared);
    assert_eq!(outer_u.effective_evidence(), EvidenceBasis::Undeclared);
    let u = register(&mut ledger, &outer_u, &inner_ids);
    assert!(matches!(ledger.overlap(u, c).unwrap(), Overlap::Unknown { .. }));
}

#[test]
fn an_unregistered_inner_callback_is_refused_not_assumed_independent() {
    let inner = gaussian();
    let outer = nested_over("outer", &inner, EvidenceBasis::Declared(BTreeSet::new()));
    let refusal = callback_source_declaration(&outer, &BTreeMap::new())
        .expect_err("the inner callback is not registered");
    assert_eq!(refusal.detail, "callback_protocol.request_mismatch");
    assert_eq!(refusal.offending.as_deref(), Some("inner"));
}

// -- the interface contract connects callbacks ---------------------------------------------------------

#[test]
fn a_callback_connects_to_a_consumer_only_for_its_declared_quantity() {
    use antecedent_contract::interface_contract::{ConsumedPort, InterfaceContract};
    let declaration = gaussian();
    let consumer = |population: &str, regime: &str| InterfaceContract {
        component: "transport_source".into(),
        produces: vec![],
        consumes: vec![
            ConsumedPort::new(
                "source_law",
                quantity(population, regime),
                "estimand:outcome_law",
                SupportDomain::Real,
                [ContractOperation::Draws],
            )
            .with_uncertainty(DistributionMeaning::InterventionalPredictive),
        ],
    };
    let producer = declaration.interface_contract();
    let ok = check_connection(
        &producer,
        CALLBACK_OUTPUT_PORT,
        &consumer("adults", "do(a=1)"),
        "source_law",
    );
    assert!(ok.is_ok(), "{ok:?}");
    let wrong_population = check_connection(
        &producer,
        CALLBACK_OUTPUT_PORT,
        &consumer("children", "do(a=1)"),
        "source_law",
    )
    .unwrap_err();
    assert!(wrong_population.iter().any(|r| r.detail == "interface_contract.population_mismatch"));
    let wrong_regime = check_connection(
        &producer,
        CALLBACK_OUTPUT_PORT,
        &consumer("adults", "observational"),
        "source_law",
    )
    .unwrap_err();
    assert!(wrong_regime.iter().any(|r| r.detail == "interface_contract.regime_mismatch"));
    // A mean-only callback cannot feed a consumer that needs draws.
    let mean_only = with_ops(&[CallbackOperation::Mean]);
    let operation = check_connection(
        &mean_only.interface_contract(),
        CALLBACK_OUTPUT_PORT,
        &consumer("adults", "do(a=1)"),
        "source_law",
    )
    .unwrap_err();
    assert!(operation.iter().any(|r| r.detail.starts_with("interface_contract.operation")));
}
