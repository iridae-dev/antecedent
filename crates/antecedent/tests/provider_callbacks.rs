//! Arbitrary law, posterior and nested callbacks through the facade executor.
//!
//! The reference callback is a Gaussian law whose mean, distribution function,
//! quantile function and draws are compared with values that do not come from
//! the provider's own code: tabulated standard-normal values and the
//! conjugate-normal posterior mean. Every refusal asserts the registered code
//! and the invocation counts that show what actually ran.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use antecedent::analysis::provider_callbacks::{
    CALLBACK_OUTPUT_PORT, CallbackConnectionError, CallbackContext, CallbackDeclaration,
    CallbackDependency, CallbackExecutionPolicy, CallbackFailure, CallbackInput, CallbackKind,
    CallbackOperation, CallbackOutcome, CallbackOutput, CallbackPhase, CallbackPlan, CallbackQuery,
    CallbackRequest, CallbackResume, CallbackSession, CallbackTrust, DependencyDeclaration,
    DependencyKind, FnCallback, InputSupplier, ProviderCallback, RecomputeReason, ReplayContract,
    ResultStanding, RetryContract, ValidatedOutput, check_callback_connection,
    check_claim_to_callback, consume_callback_result, export_callback_result,
    forecast_law_from_sample, inspect_callback_result, output_values_digest, predictive_law_source,
    refuse_identification_licence, refuse_interval, refuse_joint_law, refuse_native_trust,
    refuse_posterior_guarantee, register_callback_source, supersede_callback_source,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind, SourceStatus,
};
use antecedent_contract::interface_contract::{
    ConsumedPort, ContractOperation, EvidenceBasis, InterfaceContract, ProducedPort, SupportDomain,
};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_validate::predictive_assessment::{ForecastLaw, LawSource, ProviderTrustLabel};

const MU: f64 = 2.0;
const SD: f64 = 1.5;

// -- fixtures ---------------------------------------------------------------------------------------

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

fn declaration(replay: ReplayContract) -> CallbackDeclaration {
    CallbackDeclaration {
        component: "gaussian_law".into(),
        kind: CallbackKind::Law,
        identity: identity("1.0"),
        quantity: quantity("trial", "do(a=1)"),
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
        replay,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn seeded() -> CallbackDeclaration {
    declaration(ReplayContract::SeededStochastic { stream_id: "outcome_draws".into() })
}

fn pure() -> CallbackDeclaration {
    declaration(ReplayContract::DeterministicPure)
}

fn mean_only(replay: ReplayContract) -> CallbackDeclaration {
    let mut declaration = declaration(replay);
    declaration.operations = [CallbackOperation::Mean].into_iter().collect();
    declaration
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(42)
}

fn no_inputs() -> BTreeMap<String, ValidatedOutput> {
    BTreeMap::new()
}

// -- the reference callback -------------------------------------------------------------------------

/// `erf` by its Maclaurin series: accurate to about 1e-11 for |z| <= 3.5.
fn erf(z: f64) -> f64 {
    let mut term = z;
    let mut sum = z;
    for n in 1..80_u32 {
        term *= -z * z / f64::from(n);
        sum += term / f64::from(2 * n + 1);
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

fn phi(x: f64) -> f64 {
    if x > 8.0 {
        1.0
    } else if x < -8.0 {
        0.0
    } else {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }
}

fn probit(p: f64) -> f64 {
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if phi(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn gaussian_provider(
    version: &str,
    mu: f64,
    calls: Rc<Cell<u64>>,
) -> FnCallback<
    impl FnMut(&CallbackQuery, &mut CallbackContext<'_>) -> Result<CallbackOutput, CallbackFailure>,
> {
    FnCallback::new(identity(version), move |query, ctx| {
        calls.set(calls.get() + 1);
        match query {
            CallbackQuery::Mean => Ok(CallbackOutput::Mean { value: mu }),
            CallbackQuery::Sample { draws } => {
                let rng = ctx
                    .rng
                    .as_mut()
                    .ok_or_else(|| CallbackFailure::new("a seeded callback receives a stream"))?;
                let mut out = Vec::with_capacity(*draws as usize);
                for _ in 0..*draws {
                    let u1 = 1.0 - rng.next_f64();
                    let u2 = rng.next_f64();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    out.push(mu + SD * z);
                }
                Ok(CallbackOutput::Sample { draws: out })
            }
            CallbackQuery::Cdf { thresholds } => Ok(CallbackOutput::Cdf {
                values: thresholds.iter().map(|t| phi((t - mu) / SD)).collect(),
            }),
            CallbackQuery::Quantile { levels } => Ok(CallbackOutput::Quantile {
                values: levels.iter().map(|p| mu + SD * probit(*p)).collect(),
            }),
            CallbackQuery::PosteriorUpdate { .. } => {
                Err(CallbackFailure::new("not a posterior callback"))
            }
        }
    })
}

fn near(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

fn values_of(execution: &antecedent::analysis::CallbackExecution) -> Vec<f64> {
    execution.output.output().values()
}

#[allow(clippy::result_large_err, reason = "a test helper forwards the executor error unchanged")]
fn run(
    session: &mut CallbackSession,
    provider: &mut dyn ProviderCallback,
    declaration: &CallbackDeclaration,
    request: &CallbackRequest,
) -> Result<antecedent::analysis::CallbackExecution, antecedent::analysis::CallbackExecutionError> {
    session.execute(
        provider,
        declaration,
        request,
        &no_inputs(),
        &CallbackExecutionPolicy::default(),
        &ctx(),
    )
}

// -- reference values -------------------------------------------------------------------------------

#[test]
fn the_gaussian_reference_callback_matches_tabulated_values() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let declaration = seeded();
    let mut session = CallbackSession::new();
    // Mean.
    let mean = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Mean, 5),
    )
    .expect("mean");
    assert_eq!(values_of(&mean), vec![MU]);
    assert_eq!(mean.standing, ResultStanding::LicensedExternal);
    // Distribution function: Phi(-1.96) = 0.0249978951482205, Phi(0) = 0.5,
    // Phi(1) = 0.8413447460685429 (standard-normal tables).
    let thresholds = vec![MU - 1.96 * SD, MU, MU + SD];
    let cdf = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Cdf { thresholds }, 5),
    )
    .expect("cdf");
    let expected = [0.024_997_895_148_220_5, 0.5, 0.841_344_746_068_542_9];
    for (actual, expected) in values_of(&cdf).iter().zip(expected) {
        near(*actual, expected, 1e-9, "cdf");
    }
    // Quantile function: z(0.5) = 0, z(0.9) = 1.2815515655446004, z(0.975) = 1.959963984540054.
    let quantile = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Quantile { levels: vec![0.5, 0.9, 0.975] }, 5),
    )
    .expect("quantile");
    let expected = [MU, MU + SD * 1.281_551_565_544_600_4, MU + SD * 1.959_963_984_540_054];
    for (actual, expected) in values_of(&quantile).iter().zip(expected) {
        near(*actual, expected, 1e-9, "quantile");
    }
    // Draws: the sample mean and variance fall inside predeclared five-sigma bands.
    let n = 40_000_u32;
    let sample = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Sample { draws: n }, 5),
    )
    .expect("sample");
    let draws = values_of(&sample);
    let count = f64::from(n);
    let mean_hat = draws.iter().sum::<f64>() / count;
    let var_hat = draws.iter().map(|x| (x - mean_hat).powi(2)).sum::<f64>() / (count - 1.0);
    near(mean_hat, MU, 5.0 * SD / count.sqrt(), "sample mean");
    near(var_hat, SD * SD, 5.0 * SD * SD * (2.0 / (count - 1.0)).sqrt(), "sample variance");
    // Each query invoked the provider exactly once.
    assert_eq!(calls.get(), 4);
    assert_eq!(session.invocations(), 4);
}

#[test]
fn a_posterior_update_matches_the_conjugate_normal_posterior() {
    // Prior N(0, 4), noise variance 1, observation y = 2:
    // precision = 1/4 + 1 = 1.25, posterior mean = 2 / 1.25 = 1.6, variance = 0.8.
    let mut declaration = seeded();
    declaration.component = "posterior".into();
    declaration.kind = CallbackKind::Posterior;
    declaration.meaning = DistributionMeaning::CausalFunctionalPosterior;
    declaration.operations =
        [CallbackOperation::Mean, CallbackOperation::Sample, CallbackOperation::PosteriorUpdate]
            .into_iter()
            .collect();
    declaration.update = Some(antecedent::analysis::provider_callbacks::UpdateDeclaration {
        candidate_id: "study-a".into(),
        prior_id: "prior-0".into(),
        observation: quantity("trial", "observational"),
    });
    let mut provider = FnCallback::new(identity("1.0"), |query, ctx| match query {
        CallbackQuery::PosteriorUpdate { observation, draws } => {
            let (prior_mean, prior_var, noise_var) = (0.0_f64, 4.0_f64, 1.0_f64);
            let precision = 1.0 / prior_var + 1.0 / noise_var;
            let mean = (prior_mean / prior_var + observation / noise_var) / precision;
            let sd = (1.0 / precision).sqrt();
            let rng = ctx.rng.as_mut().expect("seeded");
            let mut out = Vec::new();
            for _ in 0..*draws {
                let (u1, u2) = (1.0 - rng.next_f64(), rng.next_f64());
                out.push(
                    mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos(),
                );
            }
            Ok(CallbackOutput::PosteriorUpdate { mean, draws: out })
        }
        _ => Err(CallbackFailure::new("only the update is implemented")),
    });
    let mut session = CallbackSession::new();
    let n = 20_000_u32;
    let execution = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::PosteriorUpdate { observation: 2.0, draws: n }, 3),
    )
    .expect("update");
    let values = values_of(&execution);
    near(values[0], 1.6, 1e-12, "posterior mean");
    let draws = &values[1..];
    let count = f64::from(n);
    let mean_hat = draws.iter().sum::<f64>() / count;
    let var_hat = draws.iter().map(|x| (x - mean_hat).powi(2)).sum::<f64>() / (count - 1.0);
    near(mean_hat, 1.6, 5.0 * 0.8_f64.sqrt() / count.sqrt(), "posterior draws mean");
    near(var_hat, 0.8, 5.0 * 0.8 * (2.0 / (count - 1.0)).sqrt(), "posterior draws variance");
    // The update is negotiated on its own: a mean query of a provider that only
    // implements the update fails and nothing else is inferred.
    let failure = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Mean, 3),
    )
    .expect_err("not implemented");
    assert_eq!(
        failure.refusal.code,
        antecedent_core::reason_code!("callback_provider_unavailable")
    );
    assert_eq!(failure.invocations, 1);
    // An operation that was never declared is refused before any invocation.
    let before = session.invocations();
    let quantile = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Quantile { levels: vec![0.5] }, 3),
    )
    .expect_err("not declared");
    assert_eq!(quantile.refusal.code, antecedent_core::reason_code!("external_capability_missing"));
    assert_eq!(quantile.invocations, 0);
    assert_eq!(session.invocations(), before);
}

// -- replay, reuse and invalidation -----------------------------------------------------------------

#[test]
fn deterministic_replay_reuses_without_invoking_until_a_dependency_changes() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let declaration = pure();
    let mut session = CallbackSession::new();
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let first = run(&mut session, &mut provider, &declaration, &request).expect("first");
    assert!(!first.reused);
    assert_eq!(first.invocations, 1);
    // Planning is a pure read: it takes no provider and says reuse.
    assert_eq!(session.plan(&declaration, &request, None).unwrap(), CallbackPlan::Reuse);
    let second = run(&mut session, &mut provider, &declaration, &request).expect("second");
    assert!(second.reused);
    assert_eq!(second.invocations, 0);
    assert_eq!(calls.get(), 1, "a replay does not invoke the provider");
    assert_eq!(
        values_of(&first).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        values_of(&second).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // A changed model parameter is a changed declaration: recompute, with the
    // reason named, and the old result stays in the history.
    let mut changed = pure();
    changed.dependencies = DependencyDeclaration::Complete(vec![
        CallbackDependency::new(DependencyKind::Model, "parameters", "mu=3;sd=1.5"),
        CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
    ]);
    assert_eq!(
        session.plan(&changed, &request, None).unwrap(),
        CallbackPlan::Recompute(RecomputeReason::DependencyChanged {
            name: "parameters".into(),
            was: "mu=2;sd=1.5".into(),
            now: "mu=3;sd=1.5".into(),
        })
    );
    let mut provider_v1 = gaussian_provider("1.0", 3.0, Rc::clone(&calls));
    let third = run(&mut session, &mut provider_v1, &changed, &request).expect("third");
    assert!(!third.reused);
    assert_eq!(values_of(&third), vec![3.0]);
    assert_eq!(calls.get(), 2);
    assert_eq!(session.history().len(), 2);
}

#[test]
fn cached_callback_outputs_respect_the_current_output_limit() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let declaration = seeded();
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 5 }, 0);
    let mut session = CallbackSession::new();
    run(&mut session, &mut provider, &declaration, &request).expect("initial sample");
    let policy =
        CallbackExecutionPolicy { max_output_values: 3, ..CallbackExecutionPolicy::default() };
    let refused = session
        .execute(&mut provider, &declaration, &request, &no_inputs(), &policy, &ctx())
        .expect_err("cached sample exceeds the new bound");
    assert_eq!(refused.refusal.detail, "callback_protocol.output_too_large");
    assert_eq!(refused.invocations, 0);
    assert_eq!(calls.get(), 1, "refusal must neither recompute nor invoke the provider");
    assert_eq!(session.history().len(), 1);
    let reused = run(&mut session, &mut provider, &declaration, &request).expect("original bound");
    assert!(reused.reused, "the refused read does not invalidate the stored sample");
    assert_eq!(calls.get(), 1);
}

#[test]
fn a_changed_inner_callback_invalidates_the_dependent_results() {
    let calls = Rc::new(Cell::new(0));
    let mut inner_provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let inner = pure();
    let mut session = CallbackSession::new();
    let inner_execution = run(
        &mut session,
        &mut inner_provider,
        &inner,
        &CallbackRequest::new(CallbackQuery::Mean, 0),
    )
    .expect("inner");
    // outer = inner mean + 3, a nested law callback.
    let mut outer = pure();
    outer.component = "shifted".into();
    outer.kind = CallbackKind::Nested;
    outer.identity = ProviderObjectIdentity { object_id: "shifted".into(), ..identity("1.0") };
    outer.operations = [CallbackOperation::Mean].into_iter().collect();
    outer.inputs = vec![CallbackInput {
        name: "inner".into(),
        quantity: inner.quantity.clone(),
        semantic_identity: inner.semantic_identity.clone(),
        support: SupportDomain::Real,
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::Callback {
            identity: inner.canonical_identity().to_hex(),
            evidence: inner.effective_evidence(),
        },
    }];
    let mut inputs = BTreeMap::new();
    inputs.insert("inner".to_owned(), inner_execution.output.clone());
    let mut request = CallbackRequest::new(CallbackQuery::Mean, 0);
    request
        .input_digests
        .insert("inner".into(), output_values_digest(&inner_execution.output.output().values()));
    let outer_calls = Rc::new(Cell::new(0_u64));
    let counter = Rc::clone(&outer_calls);
    let mut outer_provider = FnCallback::new(outer.identity.clone(), move |_, ctx| {
        counter.set(counter.get() + 1);
        let inner = ctx.inputs.get("inner").ok_or_else(|| CallbackFailure::new("input missing"))?;
        match inner.output() {
            CallbackOutput::Mean { value } => Ok(CallbackOutput::Mean { value: value + 3.0 }),
            _ => Err(CallbackFailure::new("expected a mean")),
        }
    });
    let policy = CallbackExecutionPolicy::default();
    let execution = session
        .execute(&mut outer_provider, &outer, &request, &inputs, &policy, &ctx())
        .expect("outer");
    assert_eq!(values_of(&execution), vec![5.0]);
    // A missing or tampered input is refused before the outer provider runs.
    let mut tampered = request.clone();
    tampered.input_digests.insert("inner".into(), "00".into());
    let refusal = session
        .execute(&mut outer_provider, &outer, &tampered, &inputs, &policy, &ctx())
        .expect_err("digest mismatch");
    assert_eq!(refusal.refusal.code, antecedent_core::reason_code!("external_binding_mismatch"));
    let refusal = session
        .execute(&mut outer_provider, &outer, &request, &no_inputs(), &policy, &ctx())
        .expect_err("input missing");
    assert_eq!(refusal.refusal.detail, "callback_protocol.request_mismatch");
    assert_eq!(outer_calls.get(), 1);
    // Dependents of the inner callback are found, so a change in its weights,
    // files or remote version invalidates them.
    let affected = session.dependents_of(&inner.canonical_identity().to_hex());
    assert_eq!(affected, vec![1]);
    assert_eq!(session.history()[1].declaration.canonical_identity(), outer.canonical_identity());
    // Nothing depends on the outer callback.
    assert!(session.dependents_of(&outer.canonical_identity().to_hex()).is_empty());
}

// -- stochastic streams -----------------------------------------------------------------------------

#[test]
fn stochastic_draws_follow_the_semantic_stream_rules() {
    let draws = |declaration: &CallbackDeclaration, seed: u64, warm_up: bool| {
        let calls = Rc::new(Cell::new(0));
        let mut provider = gaussian_provider("1.0", MU, calls);
        let mut session = CallbackSession::new();
        if warm_up {
            // Earlier work in the same session, with another seed and query.
            run(
                &mut session,
                &mut provider,
                declaration,
                &CallbackRequest::new(CallbackQuery::Mean, 99),
            )
            .expect("warm up");
            run(
                &mut session,
                &mut provider,
                declaration,
                &CallbackRequest::new(CallbackQuery::Sample { draws: 7 }, 98),
            )
            .expect("warm up");
        }
        let execution = run(
            &mut session,
            &mut provider,
            declaration,
            &CallbackRequest::new(CallbackQuery::Sample { draws: 6 }, seed),
        )
        .expect("draws");
        values_of(&execution).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    };
    let base = draws(&seeded(), 11, false);
    assert_eq!(base.len(), 6);
    assert_eq!(base, draws(&seeded(), 11, false), "the same seed replays bit for bit");
    assert_eq!(base, draws(&seeded(), 11, true), "earlier calls do not move the stream");
    assert_ne!(base, draws(&seeded(), 12, false), "another seed is another stream");
    let mut other_stream =
        declaration(ReplayContract::SeededStochastic { stream_id: "noise_draws".into() });
    other_stream.component = "gaussian_law".into();
    assert_ne!(base, draws(&other_stream, 11, false), "another semantic stream is independent");
    // A deterministic callback is given no stream at all.
    let mut probe = FnCallback::new(identity("1.0"), |_, ctx| {
        assert!(ctx.rng.is_none(), "a deterministic callback has no randomness");
        Ok(CallbackOutput::Mean { value: MU })
    });
    run(
        &mut CallbackSession::new(),
        &mut probe,
        &pure(),
        &CallbackRequest::new(CallbackQuery::Mean, 0),
    )
    .expect("pure");
}

// -- hidden state -----------------------------------------------------------------------------------

struct StatefulProvider {
    identity: ProviderObjectIdentity,
    version: Rc<Cell<u32>>,
    calls: Rc<Cell<u64>>,
}

impl ProviderCallback for StatefulProvider {
    fn identity(&self) -> &ProviderObjectIdentity {
        &self.identity
    }

    fn invoke(
        &mut self,
        _query: &CallbackQuery,
        _ctx: &mut CallbackContext<'_>,
    ) -> Result<CallbackOutput, CallbackFailure> {
        self.calls.set(self.calls.get() + 1);
        Ok(CallbackOutput::Mean { value: 10.0 * f64::from(self.version.get()) })
    }

    fn state_identity(&self) -> Option<String> {
        Some(format!("weights-v{}", self.version.get()))
    }
}

#[test]
fn a_changed_hidden_state_is_detected_through_the_protocol_and_never_guessed() {
    let version = Rc::new(Cell::new(1));
    let calls = Rc::new(Cell::new(0));
    let mut provider = StatefulProvider {
        identity: identity("1.0"),
        version: Rc::clone(&version),
        calls: Rc::clone(&calls),
    };
    let declaration = mean_only(ReplayContract::Stateful { exposes_state_identity: true });
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let mut session = CallbackSession::new();
    let first = run(&mut session, &mut provider, &declaration, &request).expect("first");
    assert_eq!(values_of(&first), vec![10.0]);
    assert_eq!(first.state_identity.as_deref(), Some("weights-v1"));
    // Unchanged state: reused, no invocation.
    let again = run(&mut session, &mut provider, &declaration, &request).expect("again");
    assert!(again.reused);
    assert_eq!(calls.get(), 1);
    // The weights change: the protocol's state identity changes and the stored
    // result is recomputed, not reused.
    version.set(2);
    let changed = run(&mut session, &mut provider, &declaration, &request).expect("changed");
    assert!(!changed.reused);
    assert_eq!(changed.recomputed_because, Some(RecomputeReason::StateChanged));
    assert_eq!(values_of(&changed), vec![20.0]);
    assert_eq!(calls.get(), 2);
    // A stateful result is diagnostic: it cannot back a licensed claim.
    assert_eq!(changed.standing, ResultStanding::Diagnostic);
    // State that the protocol does not expose is never shown current.
    let hidden = mean_only(ReplayContract::Stateful { exposes_state_identity: false });
    let mut session = CallbackSession::new();
    let before = calls.get();
    run(&mut session, &mut provider, &hidden, &request).expect("hidden first");
    let second = run(&mut session, &mut provider, &hidden, &request).expect("hidden second");
    assert!(!second.reused);
    assert_eq!(second.recomputed_because, Some(RecomputeReason::HiddenStateUnobservable));
    assert_eq!(calls.get(), before + 2);
}

// -- provider replacement ---------------------------------------------------------------------------

#[test]
fn a_replacement_provider_is_a_different_callback_and_the_history_stays() {
    let calls = Rc::new(Cell::new(0));
    let mut v1 = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let mut v2 = gaussian_provider("2.0", 4.0, Rc::clone(&calls));
    let declaration_v1 = pure();
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let mut session = CallbackSession::new();
    run(&mut session, &mut v1, &declaration_v1, &request).expect("v1");
    // The replacement is not silently accepted under the old declaration.
    let refusal = run(&mut session, &mut v2, &declaration_v1, &request).expect_err("replaced");
    assert_eq!(refusal.refusal.code, antecedent_core::reason_code!("external_binding_mismatch"));
    assert_eq!(refusal.refusal.detail, "callback_protocol.provider_identity_mismatch");
    assert_eq!(refusal.invocations, 0);
    assert_eq!(calls.get(), 1);
    // Declared and verified as its own callback it recomputes; the old result
    // stays inspectable.
    let mut declaration_v2 = pure();
    declaration_v2.identity = identity("2.0");
    let replaced = run(&mut session, &mut v2, &declaration_v2, &request).expect("v2");
    assert!(!replaced.reused);
    assert_eq!(values_of(&replaced), vec![4.0]);
    assert_eq!(session.history().len(), 2);
    assert_eq!(session.history()[0].declaration.identity.version_id, "1.0");
    // In the ledger the replacement supersedes the old claim.
    let mut ledger = EvidenceLedger::new();
    let old = register_callback_source(&mut ledger, &declaration_v1, &BTreeMap::new()).unwrap();
    let new =
        supersede_callback_source(&mut ledger, old, &declaration_v2, &BTreeMap::new()).unwrap();
    assert_ne!(old, new);
    assert!(matches!(ledger.source(old).expect("kept").status, SourceStatus::Superseded { .. }));
}

// -- failures, limits, cancellation -----------------------------------------------------------------

#[test]
fn malformed_laws_are_refused_whole_and_leave_no_record() {
    let calls = Rc::new(Cell::new(0_u64));
    let counter = Rc::clone(&calls);
    let mut provider = FnCallback::new(identity("1.0"), move |query, _| {
        counter.set(counter.get() + 1);
        match query {
            // Decreasing in the threshold: not a distribution function.
            CallbackQuery::Cdf { thresholds } => Ok(CallbackOutput::Cdf {
                values: vec![0.9; thresholds.len() - 1].into_iter().chain([0.2]).collect(),
            }),
            CallbackQuery::Sample { draws } => {
                Ok(CallbackOutput::Sample { draws: vec![f64::NAN; *draws as usize] })
            }
            _ => Ok(CallbackOutput::Mean { value: 0.0 }),
        }
    });
    let declaration = seeded();
    let mut session = CallbackSession::new();
    let cdf = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Cdf { thresholds: vec![0.0, 1.0] }, 0),
    )
    .expect_err("not a law");
    assert_eq!(cdf.refusal.code, antecedent_core::reason_code!("callback_output_invalid"));
    assert_eq!(cdf.refusal.detail, "callback_protocol.output_not_a_law");
    assert_eq!(cdf.invocations, 1, "the call happened; the answer was refused");
    let nan = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Sample { draws: 3 }, 0),
    )
    .expect_err("not finite");
    assert_eq!(nan.refusal.detail, "callback_protocol.output_not_finite");
    // A mean where a quantile function was asked for is the wrong shape.
    let wrong = run(
        &mut session,
        &mut provider,
        &declaration,
        &CallbackRequest::new(CallbackQuery::Quantile { levels: vec![0.5] }, 0),
    )
    .expect_err("wrong shape");
    assert_eq!(wrong.refusal.detail, "callback_protocol.output_shape");
    assert!(session.history().is_empty(), "no refused answer becomes a result");
    assert_eq!(calls.get(), 3);
}

#[test]
fn failure_cancellation_and_limits_are_typed_and_counted() {
    let calls = Rc::new(Cell::new(0));
    let declaration = pure();
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    // A provider failure is typed, counted and not retried.
    let counter = Rc::clone(&calls);
    let mut failing = FnCallback::new(identity("1.0"), move |_, _| {
        counter.set(counter.get() + 1);
        Err(CallbackFailure::new("backend down"))
    });
    let mut session = CallbackSession::new();
    let failure = run(&mut session, &mut failing, &declaration, &request).expect_err("failed");
    assert_eq!(
        failure.refusal.code,
        antecedent_core::reason_code!("callback_provider_unavailable")
    );
    assert_eq!(failure.refusal.detail, "callback_protocol.provider_failed");
    assert_eq!(failure.invocations, 1);
    assert_eq!(calls.get(), 1);
    assert!(session.history().is_empty());
    // Cancellation before the call: nothing runs.
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let refusal = session
        .execute(
            &mut provider,
            &declaration,
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &cancelled,
        )
        .expect_err("cancelled");
    assert_eq!(refusal.refusal.code, antecedent_core::reason_code!("cancelled_no_claim"));
    assert_eq!(refusal.invocations, 0);
    assert_eq!(calls.get(), 1);
    // Cancellation during the call: the call happened, the result is discarded.
    let during = ctx();
    let token = during.cancellation.clone();
    let mut cancelling = FnCallback::new(identity("1.0"), move |_, _| {
        token.cancel();
        Ok(CallbackOutput::Mean { value: MU })
    });
    let refusal = session
        .execute(
            &mut cancelling,
            &declaration,
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &during,
        )
        .expect_err("cancelled during");
    assert_eq!(refusal.refusal.detail, "callback_protocol.cancelled");
    assert_eq!(refusal.invocations, 1);
    assert!(session.history().is_empty());
    // A zero invocation bound refuses before any invocation.
    let policy =
        CallbackExecutionPolicy { max_invocations: 0, ..CallbackExecutionPolicy::default() };
    let refusal = session
        .execute(&mut provider, &declaration, &request, &no_inputs(), &policy, &ctx())
        .expect_err("no invocations allowed");
    assert_eq!(refusal.refusal.code, antecedent_core::reason_code!("callback_limit_exceeded"));
    assert_eq!(refusal.invocations, 0);
    // The wall-clock bound is enforced after the call: the late result is discarded.
    let mut slow = FnCallback::new(identity("1.0"), |_, _| {
        std::thread::sleep(std::time::Duration::from_millis(30));
        Ok(CallbackOutput::Mean { value: MU })
    });
    let policy =
        CallbackExecutionPolicy { wall_limit_ms: Some(1), ..CallbackExecutionPolicy::default() };
    let refusal = session
        .execute(&mut slow, &declaration, &request, &no_inputs(), &policy, &ctx())
        .expect_err("too slow");
    assert_eq!(refusal.refusal.detail, "callback_protocol.wall_limit_exceeded");
    assert_eq!(refusal.invocations, 1);
    // An output larger than the declared bound is a limit refusal.
    let policy =
        CallbackExecutionPolicy { max_output_values: 3, ..CallbackExecutionPolicy::default() };
    let mut seeded_provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let refusal = session
        .execute(
            &mut seeded_provider,
            &seeded(),
            &CallbackRequest::new(CallbackQuery::Sample { draws: 5 }, 0),
            &no_inputs(),
            &policy,
            &ctx(),
        )
        .expect_err("too large");
    assert_eq!(refusal.refusal.detail, "callback_protocol.output_too_large");
}

// -- side effects -----------------------------------------------------------------------------------

#[test]
fn importing_a_callback_record_never_lowers_the_known_attempt_count() {
    let mut declaration = pure();
    // Opaque dependencies require recomputation. The invocation must be
    // counted after the largest known attempt number from either history.
    declaration.dependencies = DependencyDeclaration::Opaque;
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let policy = CallbackExecutionPolicy::default();
    let mut provider =
        FnCallback::new(identity("1.0"), |_, _| Ok(CallbackOutput::Mean { value: MU }));
    let mut session = CallbackSession::new();
    session
        .execute(&mut provider, &declaration, &request, &no_inputs(), &policy, &ctx())
        .expect("one local attempt");
    let mut imported = session.history()[0].clone();
    if let CallbackOutcome::Computed { invocations, .. } = &mut imported.outcome {
        *invocations = 3;
    }
    let bytes = export_callback_result(&imported, "exhausted").expect("portable history");
    let resumed = session
        .resume(&bytes, None, Some(&mut provider), &no_inputs(), &policy, &ctx())
        .expect("recompute after the imported history");
    assert!(matches!(resumed, CallbackResume::Answered(_)));
    assert_eq!(session.request_attempts(&request.identity(&declaration).to_hex()), 4);
    assert_eq!(session.invocations(), 2);
}

#[test]
fn an_imported_attempt_count_cannot_overflow_during_recomputation() {
    let mut declaration = pure();
    declaration.dependencies = DependencyDeclaration::Opaque;
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let mut provider =
        FnCallback::new(identity("1.0"), |_, _| Ok(CallbackOutput::Mean { value: MU }));
    let mut session = CallbackSession::new();
    run(&mut session, &mut provider, &declaration, &request).expect("local record");
    let mut imported = session.history()[0].clone();
    if let CallbackOutcome::Computed { invocations, .. } = &mut imported.outcome {
        *invocations = u64::MAX;
    }
    let bytes = export_callback_result(&imported, "max-attempts").expect("portable history");
    let refused = session
        .resume(
            &bytes,
            None,
            Some(&mut provider),
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect_err("the next invocation cannot be counted");
    assert_eq!(refused.refusal.detail, "callback_protocol.invocation_limit");
    assert_eq!(refused.invocations, 0);
    assert_eq!(refused.request_attempts, u64::MAX);
    assert_eq!(session.invocations(), 1);
}

#[test]
fn a_side_effecting_callback_is_never_invoked_outside_execution_or_retried_outside_its_policy() {
    let calls = Rc::new(Cell::new(0_u64));
    let fail_first = Rc::new(Cell::new(true));
    let make_provider = |calls: Rc<Cell<u64>>, fail_first: Rc<Cell<bool>>| {
        FnCallback::new(identity("1.0"), move |_, ctx| {
            calls.set(calls.get() + 1);
            if fail_first.replace(false) {
                return Err(CallbackFailure::new("timeout after the effect may have happened"));
            }
            assert!(ctx.idempotency_key.is_some() || ctx.attempt == 1);
            Ok(CallbackOutput::Mean { value: MU })
        })
    };
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let no_retry = mean_only(ReplayContract::SideEffecting { retry: RetryContract::NoRetry });
    let mut provider = make_provider(Rc::clone(&calls), Rc::clone(&fail_first));
    let mut session = CallbackSession::new();
    // Planning and verification never invoke it.
    for phase in [CallbackPhase::Planning, CallbackPhase::Verification] {
        let policy = CallbackExecutionPolicy {
            phase,
            allow_side_effects: true,
            ..CallbackExecutionPolicy::default()
        };
        let refusal = session
            .execute(&mut provider, &no_retry, &request, &no_inputs(), &policy, &ctx())
            .expect_err("forbidden phase");
        assert_eq!(
            refusal.refusal.code,
            antecedent_core::reason_code!("callback_side_effect_forbidden")
        );
        assert_eq!(refusal.refusal.detail, "callback_protocol.invocation_outside_execution");
    }
    assert_eq!(calls.get(), 0);
    // Without an explicit opt-in the default policy does not run effects.
    let refusal = run(&mut session, &mut provider, &no_retry, &request).expect_err("no opt-in");
    assert_eq!(refusal.refusal.detail, "callback_protocol.side_effects_not_allowed");
    assert_eq!(calls.get(), 0);
    // Planning reads the declaration only.
    assert!(matches!(
        session.plan(&no_retry, &request, None).unwrap(),
        CallbackPlan::Recompute(RecomputeReason::NoPriorResult)
    ));
    // Opted in, the call runs once and fails; the retry is outside the policy.
    let allow =
        CallbackExecutionPolicy { allow_side_effects: true, ..CallbackExecutionPolicy::default() };
    let failure = session
        .execute(&mut provider, &no_retry, &request, &no_inputs(), &allow, &ctx())
        .expect_err("first attempt fails");
    assert_eq!(failure.invocations, 1);
    assert_eq!(failure.request_attempts, 1);
    let retry = session
        .execute(&mut provider, &no_retry, &request, &no_inputs(), &allow, &ctx())
        .expect_err("retry outside the policy");
    assert_eq!(retry.refusal.detail, "callback_protocol.retry_outside_policy");
    assert_eq!(retry.invocations, 0);
    assert_eq!(calls.get(), 1, "the effect was not run a second time");
    // Under a declared idempotent contract with a key, one retry is allowed,
    // within its bound, and a success is then reused without running again.
    let calls = Rc::new(Cell::new(0_u64));
    let mut provider = make_provider(Rc::clone(&calls), Rc::new(Cell::new(true)));
    let keyed = mean_only(ReplayContract::SideEffecting {
        retry: RetryContract::IdempotentKey { max_attempts: 2 },
    });
    let mut keyed_request = CallbackRequest::new(CallbackQuery::Mean, 0);
    keyed_request.idempotency_key = Some("order-17".into());
    let mut session = CallbackSession::new();
    let first = session
        .execute(&mut provider, &keyed, &keyed_request, &no_inputs(), &allow, &ctx())
        .expect_err("first attempt fails");
    assert_eq!(first.request_attempts, 1);
    let second = session
        .execute(&mut provider, &keyed, &keyed_request, &no_inputs(), &allow, &ctx())
        .expect("the declared retry succeeds");
    assert_eq!(second.invocations, 1);
    assert_eq!(second.standing, ResultStanding::Diagnostic);
    let third = session
        .execute(&mut provider, &keyed, &keyed_request, &no_inputs(), &allow, &ctx())
        .expect("a stored result under an idempotent key is reused");
    assert!(third.reused);
    assert_eq!(calls.get(), 2);
    // Verification may replay a pure callback, but never a stateful one.
    let verify = CallbackExecutionPolicy {
        phase: CallbackPhase::Verification,
        ..CallbackExecutionPolicy::default()
    };
    let verifying = Rc::new(Cell::new(0_u64));
    let mut pure_provider = gaussian_provider("1.0", MU, Rc::clone(&verifying));
    session
        .execute(&mut pure_provider, &pure(), &request, &no_inputs(), &verify, &ctx())
        .expect("a pure callback can be replayed in verification");
    assert_eq!(verifying.get(), 1);
    let stateful = mean_only(ReplayContract::Stateful { exposes_state_identity: true });
    let refusal = session
        .execute(&mut pure_provider, &stateful, &request, &no_inputs(), &verify, &ctx())
        .expect_err("stateful in verification");
    assert_eq!(refusal.refusal.detail, "callback_protocol.invocation_outside_execution");
    assert_eq!(verifying.get(), 1);
}

#[test]
fn opaque_dependencies_make_reuse_ineligible_and_the_result_diagnostic() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let mut opaque = pure();
    opaque.dependencies = DependencyDeclaration::Opaque;
    let request = CallbackRequest::new(CallbackQuery::Mean, 0);
    let mut session = CallbackSession::new();
    let first = run(&mut session, &mut provider, &opaque, &request).expect("first");
    let second = run(&mut session, &mut provider, &opaque, &request).expect("second");
    assert!(!second.reused);
    assert_eq!(second.recomputed_because, Some(RecomputeReason::OpaqueDependencies));
    assert_eq!(calls.get(), 2, "no stable identity: recomputed every time");
    assert_eq!(first.standing, ResultStanding::Diagnostic);
    // A side-effecting callback with opaque dependencies is not run at all.
    let mut effects = mean_only(ReplayContract::SideEffecting {
        retry: RetryContract::IdempotentKey { max_attempts: 2 },
    });
    effects.dependencies = DependencyDeclaration::Undeclared;
    let policy =
        CallbackExecutionPolicy { allow_side_effects: true, ..CallbackExecutionPolicy::default() };
    let refusal = session
        .execute(&mut provider, &effects, &request, &no_inputs(), &policy, &ctx())
        .expect_err("not eligible");
    assert_eq!(refusal.refusal.code, antecedent_core::reason_code!("callback_reuse_ineligible"));
    assert_eq!(calls.get(), 2);
}

// -- closed routes ----------------------------------------------------------------------------------

#[test]
fn interval_posterior_guarantee_native_trust_identification_and_joint_law_stay_closed() {
    let closed = antecedent_core::reason_code!("cell_not_licensed");
    for (refusal, detail) in [
        (refuse_interval(), "callback_protocol.interval_not_licensed"),
        (refuse_posterior_guarantee(), "callback_protocol.posterior_guarantee_not_licensed"),
        (refuse_native_trust(), "callback_protocol.native_trust_not_conferred"),
        (refuse_identification_licence(), "callback_protocol.identification_not_conferred"),
    ] {
        assert_eq!(refusal.code, closed);
        assert_eq!(refusal.detail, detail);
        assert!(refusal.remedy.is_some());
        refusal.validate().expect("well-formed refusal");
    }
    let joint = refuse_joint_law();
    assert_eq!(joint.code, antecedent_core::reason_code!("joint_law_required"));
    // An accepted callback law carries no calibration: its interface projection
    // is unmeasured, never a measured record.
    let projection = seeded().interface_contract();
    let port = projection.produced(CALLBACK_OUTPUT_PORT).expect("port");
    assert!(!port.calibration.is_measured());
}

// -- transported claims as callback inputs and outputs ----------------------------------------------

fn transport_consumer(population: &str) -> InterfaceContract {
    InterfaceContract {
        component: "transport_source".into(),
        produces: vec![],
        consumes: vec![
            ConsumedPort::new(
                "source_law",
                quantity(population, "do(a=1)"),
                "estimand:outcome_law",
                SupportDomain::Real,
                [ContractOperation::Draws],
            )
            .with_uncertainty(DistributionMeaning::InterventionalPredictive),
        ],
    }
}

#[test]
fn a_provider_output_feeds_transport_only_for_the_declared_population_and_regime() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let declaration = seeded();
    let mut session = CallbackSession::new();
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 8 }, 1);
    // Executed route: the callback's trial population is the transport source.
    let (connection, execution) = session
        .execute_for_consumer(
            &mut provider,
            &declaration,
            &transport_consumer("trial"),
            "source_law",
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("connected and executed");
    assert_eq!(connection.consumed_port, "source_law");
    assert_eq!(execution.invocations, 1);
    assert_eq!(execution.output.output().value_count(), 8);
    // Refusal: the consumer needs another population; the provider never runs.
    let before = calls.get();
    let error = session
        .execute_for_consumer(
            &mut provider,
            &declaration,
            &transport_consumer("target"),
            "source_law",
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect_err("population differs");
    let CallbackConnectionError::Connection(refusals) = error else {
        panic!("the connection is refused before execution");
    };
    assert!(refusals.iter().any(|r| r.detail == "interface_contract.population_mismatch"));
    assert!(
        refusals.iter().all(
            |r| r.code == antecedent_core::reason_code!("interface_contract_quantity_mismatch")
        )
    );
    assert_eq!(calls.get(), before);
    // The same check without executing.
    assert!(
        check_callback_connection(&declaration, &transport_consumer("trial"), "source_law").is_ok()
    );
}

#[test]
fn a_transported_claim_reaches_a_callback_only_as_the_population_it_declares() {
    // A callback consuming a transported effect for the target population.
    let mut consumer = mean_only(ReplayContract::DeterministicPure);
    consumer.component = "external_consumer".into();
    consumer.inputs = vec![CallbackInput {
        name: "transported_effect".into(),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:transported_effect".into(),
        support: SupportDomain::Real,
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::External,
    }];
    consumer.validate().expect("well formed");
    let claim = |population: &str| InterfaceContract {
        component: "transport".into(),
        produces: vec![
            ProducedPort::new(
                "transported_effect",
                quantity(population, "do(a=1)"),
                "estimand:transported_effect",
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_calibration(antecedent_contract::interface_contract::CalibrationStatus::PointOnly)
            .with_evidence([SourceId::from_digest(SemanticDigest::from_bytes([7; 32]))]),
        ],
        consumes: vec![],
    };
    // Executed route: the claim is for the target population the consumer names.
    let connection = check_claim_to_callback(
        &claim("target"),
        "transported_effect",
        &consumer,
        "transported_effect",
    )
    .expect("connects");
    assert_eq!(connection.producer, "transport");
    // Refusal: a claim about the source population is not a target claim.
    let refusals = check_claim_to_callback(
        &claim("trial"),
        "transported_effect",
        &consumer,
        "transported_effect",
    )
    .expect_err("population differs");
    assert!(refusals.iter().any(|r| r.detail == "interface_contract.population_mismatch"));
}

// -- registration of evidence -----------------------------------------------------------------------

#[test]
fn registered_callbacks_expose_their_overlap_through_the_ledger() {
    let mut ledger = EvidenceLedger::new();
    let rows = |label: &str, keys: &[&str]| {
        SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(
            keys.iter().map(|k| ObservationId::from_key("test.provider_callbacks", k.as_bytes())),
        )
    };
    let shared = ledger.add_source(rows("trial", &["a", "b"])).unwrap();
    let other = ledger.add_source(rows("registry", &["c"])).unwrap();
    let mut first = pure();
    first.component = "first".into();
    first.evidence = EvidenceBasis::Declared([shared].into_iter().collect());
    let mut second = pure();
    second.component = "second".into();
    second.evidence = EvidenceBasis::Declared([shared].into_iter().collect());
    let mut third = pure();
    third.component = "third".into();
    third.evidence = EvidenceBasis::Declared([other].into_iter().collect());
    let a = register_callback_source(&mut ledger, &first, &BTreeMap::new()).unwrap();
    let b = register_callback_source(&mut ledger, &second, &BTreeMap::new()).unwrap();
    let c = register_callback_source(&mut ledger, &third, &BTreeMap::new()).unwrap();
    assert_eq!(ledger.overlap(a, b).unwrap(), Overlap::SharedObservations { shared: 2 });
    assert_eq!(ledger.overlap(a, c).unwrap(), Overlap::Independent);
}

// -- portable results and fresh-process resume -------------------------------------------------------

#[test]
fn a_result_resumes_with_a_compatible_provider_and_is_unavailable_without_one() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, Rc::clone(&calls));
    let declaration = seeded();
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 5 }, 9);
    let mut producer = CallbackSession::new();
    let produced = run(&mut producer, &mut provider, &declaration, &request).expect("produced");
    let bytes = export_callback_result(&producer.history()[0], "result-1").expect("export");
    let artifact = consume_callback_result(&bytes, None).expect("consumes");
    let identity_hex = artifact.identity().to_owned();
    // Inspection reads the historical result without executing anything.
    let meta = inspect_callback_result(&bytes).expect("inspect");
    assert_eq!(meta.stage.status, "computed");
    assert_eq!(meta.stage.value_count, 5);
    assert_eq!(meta.standing, "licensed_external");
    // A fresh process with no provider: the stage is recorded unavailable and
    // nothing runs; the historical result stays in the history.
    let mut fresh = CallbackSession::new();
    let resumed = fresh
        .resume(
            &bytes,
            Some(&identity_hex),
            None,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("resume");
    let CallbackResume::Unavailable { record, refusal } = resumed else {
        panic!("without a provider the stage is unavailable");
    };
    assert_eq!(refusal.code, antecedent_core::reason_code!("callback_provider_unavailable"));
    assert_eq!(fresh.history().len(), 2);
    assert!(matches!(fresh.history()[record].outcome, CallbackOutcome::Unavailable { .. }));
    assert_eq!(fresh.invocations(), 0);
    // With a supplied compatible provider the seeded result is reused: the
    // portable payload answers it and the provider does not run.
    let counter = Rc::new(Cell::new(0));
    let mut compatible = gaussian_provider("1.0", MU, Rc::clone(&counter));
    let mut fresh = CallbackSession::new();
    let CallbackResume::Answered(answer) = fresh
        .resume(
            &bytes,
            Some(&identity_hex),
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
    assert_eq!(values_of(&answer), values_of(&produced));
    // A provider that is not the declared object is refused, not substituted.
    let mut replaced = gaussian_provider("2.0", 7.0, Rc::clone(&counter));
    let mut fresh = CallbackSession::new();
    let error = fresh
        .resume(
            &bytes,
            None,
            Some(&mut replaced),
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect_err("replaced provider");
    assert_eq!(error.refusal.code, antecedent_core::reason_code!("external_binding_mismatch"));
    assert_eq!(counter.get(), 0);
    // A retained identity of another result refuses; a flipped byte refuses.
    assert!(consume_callback_result(&bytes, Some(&"0".repeat(64))).is_err());
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0x40;
    assert!(consume_callback_result(&corrupt, None).is_err());
}

// -- the predictive assessment consumes a callback law as a provider law ----------------------------

#[test]
fn a_callback_law_reaches_the_predictive_assessment_only_as_a_replayable_provider_law() {
    let source = predictive_law_source(&seeded(), false).expect("replayable and attested");
    let LawSource::Provider { identity, trust } = source else { panic!("a provider law") };
    assert_eq!(identity.version_id, "1.0");
    assert_eq!(trust, ProviderTrustLabel::ExternallyAttested);
    // A stateful, opaque or unverified callback could not be reproduced from
    // its declaration: it is not offered to the assessment.
    let mut opaque = seeded();
    opaque.dependencies = DependencyDeclaration::Opaque;
    assert!(predictive_law_source(&opaque, false).is_err());
    let mut unverified = seeded();
    unverified.trust = CallbackTrust::Unverified;
    assert!(predictive_law_source(&unverified, false).is_err());
    let stateful = mean_only(ReplayContract::Stateful { exposes_state_identity: true });
    assert!(predictive_law_source(&stateful, false).is_err());
    // An accepted sample becomes its empirical law.
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, calls);
    let mut session = CallbackSession::new();
    let execution = run(
        &mut session,
        &mut provider,
        &seeded(),
        &CallbackRequest::new(CallbackQuery::Sample { draws: 4 }, 2),
    )
    .expect("sample");
    let law = forecast_law_from_sample(&execution.output).expect("a sample");
    let ForecastLaw::EmpiricalSample { draws } = law else { panic!("empirical sample") };
    assert_eq!(draws, values_of(&execution));
    let mean =
        run(&mut session, &mut provider, &seeded(), &CallbackRequest::new(CallbackQuery::Mean, 2))
            .expect("mean");
    assert!(forecast_law_from_sample(&mean.output).is_err());
}

#[test]
fn a_connected_result_exports_and_is_consumed_against_its_retained_identity() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, calls);
    let declaration = seeded();
    let mut session = CallbackSession::new();
    let request = CallbackRequest::new(CallbackQuery::Sample { draws: 6 }, 4);
    session
        .execute_for_consumer(
            &mut provider,
            &declaration,
            &transport_consumer("trial"),
            "source_law",
            &request,
            &no_inputs(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("connected and executed");
    let bytes = export_callback_result(&session.history()[0], "connected-result").expect("export");
    let retained = consume_callback_result(&bytes, None).expect("consumes").identity().to_owned();
    let artifact = consume_callback_result(&bytes, Some(&retained)).expect("retained identity");
    // The portable declaration is the interface contract the connection was
    // checked against: a consumer connects the consumed record, not a promise.
    let contract = artifact.record().declaration.interface_contract();
    assert!(
        antecedent_contract::interface_contract::check_connection(
            &contract,
            CALLBACK_OUTPUT_PORT,
            &transport_consumer("trial"),
            "source_law"
        )
        .is_ok()
    );
    assert!(
        antecedent_contract::interface_contract::check_connection(
            &contract,
            CALLBACK_OUTPUT_PORT,
            &transport_consumer("target"),
            "source_law"
        )
        .is_err()
    );
    assert_eq!(artifact.record(), &session.history()[0]);
}

#[test]
fn an_exported_callback_result_carries_no_calibration_record_and_no_interval() {
    let calls = Rc::new(Cell::new(0));
    let mut provider = gaussian_provider("1.0", MU, calls);
    let mut session = CallbackSession::new();
    for (declaration, query, label) in [
        (seeded(), CallbackQuery::Sample { draws: 3 }, "unmeasured"),
        (mean_only(ReplayContract::DeterministicPure), CallbackQuery::Mean, "point_only"),
    ] {
        run(&mut session, &mut provider, &declaration, &CallbackRequest::new(query, 1))
            .expect("executes");
        let record = session.history().last().expect("a record");
        let bytes = export_callback_result(record, "no-calibration").expect("export");
        let meta = inspect_callback_result(&bytes).expect("inspect");
        let port = &meta.contract.produces[0];
        assert_eq!(port.calibration.status, label);
        assert!(port.calibration.record_id.is_none());
        assert!(!port.operations.iter().any(|op| op == "interval"));
        assert!(!port.operations.iter().any(|op| op == "posterior"));
        consume_callback_result(&bytes, None).expect("consumes");
    }
}
