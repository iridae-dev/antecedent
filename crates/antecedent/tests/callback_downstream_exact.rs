//! Callback results consumed by downstream computation, with hand-derived outcomes.
//!
//! * A portable result is exported, consumed from its bytes and resumed; its draws are then
//!   scored by the library's ranked-probability score. The expected scores are worked out by
//!   hand from the empirical law of the stored draws.
//! * A callback reaches a consumer, a claim reaches a callback input, and a callback law reaches
//!   the predictive assessment exactly as the connection and standing rules say, tabulated by hand.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use antecedent::analysis::provider_callbacks::{
    CallbackContext, CallbackDeclaration, CallbackDependency, CallbackExecutionPolicy,
    CallbackFailure, CallbackInput, CallbackKind, CallbackOperation, CallbackOutcome,
    CallbackOutput, CallbackQuery, CallbackRequest, CallbackResume, CallbackSession, CallbackTrust,
    DependencyDeclaration, DependencyKind, FnCallback, InputSupplier, ReplayContract,
    RetryContract, ValidatedOutput, check_claim_to_callback, consume_callback_result,
    export_callback_result, forecast_law_from_sample, inspect_callback_result,
    predictive_law_source,
};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, InterfaceContract, ProducedPort,
    SupportDomain,
};
use antecedent_contract::evidence_ledger::SourceId;
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_validate::predictive_assessment::{
    EnsembleCrpsForm, ForecastLaw, LawSource, Outcome, ProviderTrustLabel, ScoreKind, score_case,
};

fn near(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-12, "{what}: {actual} vs {expected}");
}

fn quantity(population: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:count".into(),
        variable_name: "count".into(),
        role: QuantityRole::Outcome,
        units: "events".into(),
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
        object_id: "counts".into(),
        version_id: version.into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    }
}

fn declaration(
    replay: ReplayContract,
    operations: &[CallbackOperation],
    dependencies: DependencyDeclaration,
    trust: CallbackTrust,
) -> CallbackDeclaration {
    CallbackDeclaration {
        component: "count_law".into(),
        kind: CallbackKind::Law,
        identity: identity("1.0"),
        quantity: quantity("trial", "do(a=1)"),
        semantic_identity: "estimand:count_law".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: operations.iter().copied().collect(),
        update: None,
        inputs: Vec::new(),
        dependencies,
        replay,
        trust,
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn complete() -> DependencyDeclaration {
    DependencyDeclaration::Complete(vec![CallbackDependency::new(
        DependencyKind::Model,
        "table",
        "counts-v1",
    )])
}

fn attested() -> CallbackTrust {
    CallbackTrust::ExternallyAttested { attestor: "lab".into() }
}

fn sampler() -> CallbackDeclaration {
    declaration(
        ReplayContract::SeededStochastic { stream_id: "counts".into() },
        &[CallbackOperation::Mean, CallbackOperation::Sample],
        complete(),
        attested(),
    )
}

/// A provider whose sample is always `fixed`, whatever the stream.
fn fixed_provider(
    version: &str,
    fixed: &'static [f64],
    calls: Rc<Cell<u64>>,
) -> FnCallback<
    impl FnMut(&CallbackQuery, &mut CallbackContext<'_>) -> Result<CallbackOutput, CallbackFailure>,
> {
    FnCallback::new(identity(version), move |query, _| {
        calls.set(calls.get() + 1);
        match query {
            CallbackQuery::Sample { draws } => {
                Ok(CallbackOutput::Sample { draws: fixed[..*draws as usize].to_vec() })
            }
            _ => Err(CallbackFailure::new("sample only")),
        }
    })
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(42)
}

fn no_inputs() -> BTreeMap<String, ValidatedOutput> {
    BTreeMap::new()
}

fn crps(law: &ForecastLaw, y: f64, form: EnsembleCrpsForm) -> f64 {
    score_case(ScoreKind::Crps, form, law, &Outcome::Real(y)).expect("scores")
}

fn law_of(outcome: &CallbackOutcome) -> ForecastLaw {
    let CallbackOutcome::Computed { output, .. } = outcome else { panic!("a computed result") };
    forecast_law_from_sample(output).expect("a sample")
}

/// The provider's draws are {0, 1, 1, 4}: the empirical CDF is 1/4 on [0,1), 3/4 on [1,4) and 1
/// from 4. The CRPS of that law at y is `(1/4) sum |x - y| - (1/32) sum_ij |x_i - x_j|`, and the
/// ordered-pair sum is 2 * (1 + 1 + 4 + 0 + 3 + 3) = 24, so the second term is 3/4:
///
/// * y = 0: 6/4 - 3/4 = 3/4;  y = 1: 4/4 - 3/4 = 1/4;  y = 2: 6/4 - 3/4 = 3/4;
/// * y = 3: 8/4 - 3/4 = 5/4;  y = 5: 14/4 - 3/4 = 11/4.
///
/// (Cross-checked against the integral of `(F(t) - 1{t >= y})^2`; at y = 2 that is
/// 1/16 + 9/16 + 2 * 1/16 = 3/4.) The fair form divides the pair sum by 2 m (m - 1) = 24, so at
/// y = 2 it is 3/2 - 1 = 1/2. The same hand scores must come from the stored result, the
/// consumed artifact and a resumed answer, none of which runs the provider again.
#[test]
fn a_portable_result_is_scored_by_the_hand_crps_of_its_stored_draws() {
    const DRAWS: &[f64] = &[0.0, 1.0, 1.0, 4.0];
    let calls = Rc::new(Cell::new(0));
    let mut provider = fixed_provider("1.0", DRAWS, Rc::clone(&calls));
    let declaration = sampler();
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 3);
    let mut producer = CallbackSession::new();
    producer
        .execute(
            &mut provider,
            &declaration,
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("produced");
    assert_eq!(calls.get(), 1);
    let bytes = export_callback_result(&producer.history()[0], "counts-result").expect("export");

    let expected = [(0.0, 0.75), (1.0, 0.25), (2.0, 0.75), (3.0, 1.25), (5.0, 2.75)];
    let artifact = consume_callback_result(&bytes, None).expect("consumes");
    let law = law_of(&artifact.record().outcome);
    let ForecastLaw::EmpiricalSample { draws } = &law else { panic!("empirical sample") };
    assert_eq!(draws, DRAWS);
    for (y, score) in expected {
        near(crps(&law, y, EnsembleCrpsForm::EmpiricalLaw), score, &format!("crps at {y}"));
    }
    near(crps(&law, 2.0, EnsembleCrpsForm::Fair), 0.5, "fair crps at 2");
    assert_eq!(inspect_callback_result(&bytes).expect("inspect").stage.value_count, 4);

    // A fresh session with a compatible provider answers from the portable payload.
    let counter = Rc::new(Cell::new(0));
    let mut compatible = fixed_provider("1.0", DRAWS, Rc::clone(&counter));
    let mut fresh = CallbackSession::new();
    let CallbackResume::Answered(answer) = fresh
        .resume(
            &bytes,
            Some(artifact.identity()),
            Some(&mut compatible),
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("resume")
    else {
        panic!("answered");
    };
    assert!(answer.reused);
    assert_eq!(counter.get(), 0);
    let answered = forecast_law_from_sample(&answer.output).expect("a sample");
    for (y, score) in expected {
        near(crps(&answered, y, EnsembleCrpsForm::EmpiricalLaw), score, &format!("resumed {y}"));
    }
}

/// Two providers answered the same sample request, one with {0, 1, 1, 4} and one with
/// {2, 2, 2, 2}. Observing y = 2, the constant sample has CRPS 0 (all mass at the outcome), and
/// the spread one 3/4, so after export and consumption the constant provider ranks first; at
/// y = 5 the constant sample scores |2 - 5| = 3 against 11/4 and the ranking reverses.
#[test]
fn consumed_results_rank_providers_by_their_hand_scores() {
    const SPREAD: &[f64] = &[0.0, 1.0, 1.0, 4.0];
    const CONSTANT: &[f64] = &[2.0, 2.0, 2.0, 2.0];
    let mut laws = Vec::new();
    for (fixed, version) in [(SPREAD, "1.0"), (CONSTANT, "2.0")] {
        let mut declaration = sampler();
        declaration.identity = identity(version);
        let mut provider = fixed_provider(version, fixed, Rc::new(Cell::new(0)));
        let mut session = CallbackSession::new();
        session
            .execute(
                &mut provider,
                &declaration,
                &CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 1),
                &no_inputs(),
                &CallbackExecutionPolicy::default(),
                &ctx(),
            )
            .expect("executes");
        let bytes = export_callback_result(&session.history()[0], "result").expect("export");
        let artifact = consume_callback_result(&bytes, None).expect("consumes");
        laws.push(law_of(&artifact.record().outcome));
    }
    let score = |law: &ForecastLaw, y| crps(law, y, EnsembleCrpsForm::EmpiricalLaw);
    near(score(&laws[0], 2.0), 0.75, "spread at 2");
    near(score(&laws[1], 2.0), 0.0, "constant at 2");
    near(score(&laws[0], 5.0), 2.75, "spread at 5");
    near(score(&laws[1], 5.0), 3.0, "constant at 5");
    assert!(score(&laws[1], 2.0) < score(&laws[0], 2.0));
    assert!(score(&laws[0], 5.0) < score(&laws[1], 5.0));
}

fn transport_consumer(population: &str) -> InterfaceContract {
    use antecedent_contract::interface_contract::ConsumedPort;
    InterfaceContract {
        component: "transport_source".into(),
        produces: vec![],
        consumes: vec![
            ConsumedPort::new(
                "source_law",
                quantity(population, "do(a=1)"),
                "estimand:count_law",
                SupportDomain::Real,
                [ContractOperation::Draws],
            )
            .with_uncertainty(DistributionMeaning::InterventionalPredictive),
        ],
    }
}

/// A callback connected to a consumer produces a sample that the assessment scores as its
/// empirical law. Draws {2, 2, 3, 5} scored at y = 3: `(1+1+0+2)/4 - 20/32 = 1 - 5/8 = 3/8`
/// (ordered pairs: 2 * (0+1+3+1+3+2) = 20). The integral of `(F(t) - 1{t >= 3})^2` agrees: F is
/// 1/2 on [2,3), contributing 1/4, and 3/4 on [3,5) against the step, contributing 2 * 1/16.
#[test]
fn an_executed_consumer_route_yields_the_hand_scored_empirical_law() {
    const DRAWS: &[f64] = &[2.0, 2.0, 3.0, 5.0];
    let calls = Rc::new(Cell::new(0));
    let mut provider = fixed_provider("1.0", DRAWS, Rc::clone(&calls));
    let mut session = CallbackSession::new();
    let (connection, execution) = session
        .execute_for_consumer(
            &mut provider,
            &sampler(),
            &transport_consumer("trial"),
            "source_law",
            &CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 1),
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("connected and executed");
    assert_eq!(
        connection.operations,
        [ContractOperation::Draws].into_iter().collect::<BTreeSet<_>>()
    );
    let law = forecast_law_from_sample(&execution.output).expect("a sample");
    near(crps(&law, 3.0, EnsembleCrpsForm::EmpiricalLaw), 0.375, "crps at 3");
    // The consumer for another population is refused and the provider is not entered again.
    let before = calls.get();
    assert!(
        session
            .execute_for_consumer(
                &mut provider,
                &sampler(),
                &transport_consumer("target"),
                "source_law",
                &CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 2),
                &no_inputs(),
                &CallbackExecutionPolicy::default(),
                &ctx(),
            )
            .is_err()
    );
    assert_eq!(calls.get(), before);
}

/// The predictive assessment is offered a callback law exactly when it is replayable (pure or
/// seeded), its dependencies are complete, and it is attested or verified with a confirmed
/// receipt. The table is the product of two dependency declarations, four replay contracts and
/// three trust levels with confirmation on and off: 48 cases, of which 6 are offered.
#[test]
fn a_callback_law_is_offered_exactly_when_replayable_complete_and_attested_or_confirmed() {
    let replays = [
        ("pure", ReplayContract::DeterministicPure),
        ("seeded", ReplayContract::SeededStochastic { stream_id: "s".into() }),
        ("stateful", ReplayContract::Stateful { exposes_state_identity: true }),
        ("side effecting", ReplayContract::SideEffecting { retry: RetryContract::NoRetry }),
    ];
    let trusts = [
        ("unverified", CallbackTrust::Unverified),
        ("attested", attested()),
        ("verified", CallbackTrust::ExactRequestVerified { receipt_digest: "d".repeat(64) }),
    ];
    let mut offered = 0;
    let mut cases = 0;
    for complete_dependencies in [true, false] {
        for (replay_name, replay) in &replays {
            for (trust_name, trust) in &trusts {
                for confirmed in [false, true] {
                    let dependencies =
                        if complete_dependencies { complete() } else { DependencyDeclaration::Opaque };
                    let law = declaration(
                        replay.clone(),
                        &[CallbackOperation::Mean],
                        dependencies,
                        trust.clone(),
                    );
                    let replayable = matches!(*replay_name, "pure" | "seeded");
                    let verified = *trust_name == "verified";
                    let expected = complete_dependencies
                        && replayable
                        && (*trust_name == "attested" || (verified && confirmed));
                    cases += 1;
                    let label =
                        format!("{complete_dependencies}/{replay_name}/{trust_name}/{confirmed}");
                    match predictive_law_source(&law, confirmed) {
                        Ok(LawSource::Provider { identity: provider, trust }) => {
                            assert!(expected, "{label} must be refused");
                            offered += 1;
                            assert_eq!(provider.version_id, "1.0", "{label}");
                            let label_expected = if verified {
                                ProviderTrustLabel::ExactRequestVerified
                            } else {
                                ProviderTrustLabel::ExternallyAttested
                            };
                            assert_eq!(trust, label_expected, "{label}");
                        }
                        Ok(other) => panic!("{label}: not a provider law: {other:?}"),
                        Err(refusal) => {
                            assert!(!expected, "{label} must be offered");
                            assert_eq!(refusal.code, "callback_reuse_ineligible", "{label}");
                        }
                    }
                }
            }
        }
    }
    // Offered: complete dependencies, 2 replayable contracts, attested (x2 confirmations) and
    // verified only when confirmed (x1): 2 * (2 + 1) = 6.
    assert_eq!(cases, 48);
    assert_eq!(offered, 6);
}

fn consuming_callback() -> CallbackDeclaration {
    let mut consumer = declaration(
        ReplayContract::DeterministicPure,
        &[CallbackOperation::Mean],
        complete(),
        attested(),
    );
    consumer.component = "external_consumer".into();
    consumer.inputs = vec![CallbackInput {
        name: "effect".into(),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:effect".into(),
        support: SupportDomain::Real,
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::External,
    }];
    consumer
}

fn claim(
    change: impl FnOnce(&mut ScientificQuantity, &mut String, &mut Vec<ContractOperation>),
    evidence: Option<usize>,
) -> InterfaceContract {
    let mut quantity = quantity("target", "do(a=1)");
    let mut semantic = "estimand:effect".to_owned();
    let mut operations = vec![ContractOperation::Mean];
    change(&mut quantity, &mut semantic, &mut operations);
    let mut port = ProducedPort::new("effect", quantity, semantic, SupportDomain::Real, operations)
        .with_calibration(CalibrationStatus::PointOnly);
    if let Some(count) = evidence {
        let sources = (0..count)
            .map(|k| SourceId::from_digest(SemanticDigest::from_bytes([k as u8 + 1; 32])));
        port = port.with_evidence(sources);
    }
    InterfaceContract { component: "claim".into(), produces: vec![port], consumes: vec![] }
}

fn claim_details(producer: &InterfaceContract) -> Result<(), Vec<String>> {
    check_claim_to_callback(producer, "effect", &consuming_callback(), "effect")
        .map(|_| ())
        .map_err(|refusals| refusals.into_iter().map(|r| r.detail).collect())
}

/// A claim reaches a callback input exactly when the claim's port is the input's quantity,
/// offers its operation and declares its evidence. One field is changed at a time; the expected
/// refusal is the single matching detail, and a claim offering more operations or a narrower
/// support still connects.
#[test]
fn a_claim_reaches_a_callback_input_only_as_the_quantity_it_declares_with_evidence() {
    assert_eq!(claim_details(&claim(|_, _, _| {}, Some(1))), Ok(()));
    let cases: Vec<(&str, InterfaceContract, Vec<&str>)> = vec![
        (
            "population",
            claim(|q, _, _| q.population_id = "trial".into(), Some(1)),
            vec!["population_mismatch"],
        ),
        (
            "regime",
            claim(|q, _, _| q.regime_id = "do(a=0)".into(), Some(1)),
            vec!["regime_mismatch"],
        ),
        ("units", claim(|q, _, _| q.units = "kPa".into(), Some(1)), vec!["units_mismatch"]),
        (
            "semantic identity",
            claim(|_, s, _| *s = "estimand:other".into(), Some(1)),
            vec!["semantic_identity_mismatch"],
        ),
        (
            "operation",
            claim(|_, _, o| *o = vec![ContractOperation::Point], Some(1)),
            vec!["operation_unavailable"],
        ),
        ("undeclared evidence", claim(|_, _, _| {}, None), vec!["dependence_unknown"]),
        ("empty evidence", claim(|_, _, _| {}, Some(0)), vec!["dependence_unknown"]),
        (
            "two defects",
            claim(|q, _, _| {
                q.population_id = "trial".into();
                q.units = "kPa".into();
            }, Some(1)),
            vec!["units_mismatch", "population_mismatch"],
        ),
    ];
    for (label, producer, expected) in cases {
        let expected: Vec<String> =
            expected.into_iter().map(|d| format!("interface_contract.{d}")).collect();
        assert_eq!(claim_details(&producer), Err(expected), "{label}");
    }
    // More than needed is not a defect: extra operations, and two evidence sources.
    let richer = claim(
        |_, _, o| *o = vec![ContractOperation::Mean, ContractOperation::Point],
        Some(2),
    );
    assert_eq!(claim_details(&richer), Ok(()));
}
