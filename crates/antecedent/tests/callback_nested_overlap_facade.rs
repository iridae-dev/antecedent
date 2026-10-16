//! Nested callbacks refused, through `CallbackSession::execute`, when the overlap of their
//! inner callbacks cannot be shown to be zero.
//!
//! The outer callback combines two inner callbacks by sum. The ledger decides whether the two
//! inner callbacks rest on observations in common; the executor refuses before invoking the
//! outer provider unless it shows they do not. Each refusal asserts the registered reason
//! code, the namespaced detail, the offending inputs, and that no provider ran and no record
//! was added to the session history.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;

use antecedent::analysis::provider_callbacks::{
    CallbackDeclaration, CallbackDependency, CallbackExecutionPolicy, CallbackFailure,
    CallbackInput, CallbackKind, CallbackOperation, CallbackOutput, CallbackQuery, CallbackRequest,
    CallbackSession, CallbackTrust, DependencyDeclaration, DependencyKind, FnCallback,
    InputSupplier, ReplayContract, ValidatedOutput, output_values_digest,
};
use antecedent_contract::interface_contract::{EvidenceBasis, SupportDomain};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;

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

fn identity(object: &str) -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: object.into(),
        version_id: "1.0".into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    }
}

/// A deterministic mean-only law callback named `component`.
fn law(component: &str, evidence: EvidenceBasis) -> CallbackDeclaration {
    CallbackDeclaration {
        component: component.into(),
        kind: CallbackKind::Law,
        identity: identity(component),
        quantity: quantity(),
        semantic_identity: "estimand:outcome_law".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [CallbackOperation::Mean].into_iter().collect(),
        update: None,
        inputs: Vec::new(),
        dependencies: DependencyDeclaration::Complete(vec![
            CallbackDependency::new(DependencyKind::Model, "parameters", component),
            CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
        ]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence,
    }
}

fn declared_nothing() -> EvidenceBasis {
    EvidenceBasis::Declared(std::collections::BTreeSet::new())
}

fn input(name: &str, inner: &CallbackDeclaration) -> CallbackInput {
    CallbackInput {
        name: name.into(),
        quantity: inner.quantity.clone(),
        semantic_identity: inner.semantic_identity.clone(),
        support: SupportDomain::Real,
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::Callback {
            identity: inner.canonical_identity().to_hex(),
            evidence: inner.effective_evidence(),
        },
    }
}

/// The outer callback: the sum of its `left` and `right` inner means.
fn outer(left: &CallbackDeclaration, right: &CallbackDeclaration) -> CallbackDeclaration {
    let mut outer = law("sum", declared_nothing());
    outer.kind = CallbackKind::Nested;
    outer.inputs = vec![input("left", left), input("right", right)];
    outer
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

struct Fixture {
    session: CallbackSession,
    inputs: BTreeMap<String, ValidatedOutput>,
    request: CallbackRequest,
    outer_calls: Rc<Cell<u64>>,
}

/// Execute each inner callback once (they are mean callbacks answering `left_value` and
/// `right_value`) and bind their outputs to the outer request's inputs.
fn fixture(
    left: &CallbackDeclaration,
    left_value: f64,
    right: &CallbackDeclaration,
    right_value: f64,
) -> Fixture {
    let mut session = CallbackSession::new();
    let mut inputs = BTreeMap::new();
    let mut request = CallbackRequest::new(CallbackQuery::Mean, 0);
    for (name, declaration, value) in [("left", left, left_value), ("right", right, right_value)] {
        let mut provider = FnCallback::new(declaration.identity.clone(), move |_, _| {
            Ok(CallbackOutput::Mean { value })
        });
        let execution = session
            .execute(
                &mut provider,
                declaration,
                &CallbackRequest::new(CallbackQuery::Mean, 0),
                &BTreeMap::new(),
                &CallbackExecutionPolicy::default(),
                &ctx(),
            )
            .expect("inner callback executes");
        request
            .input_digests
            .insert(name.to_owned(), output_values_digest(&execution.output.output().values()));
        inputs.insert(name.to_owned(), execution.output);
    }
    Fixture { session, inputs, request, outer_calls: Rc::new(Cell::new(0)) }
}

fn run_outer(
    fixture: &mut Fixture,
    outer: &CallbackDeclaration,
) -> Result<
    antecedent::analysis::CallbackExecution,
    Box<antecedent::analysis::CallbackExecutionError>,
> {
    let counter = Rc::clone(&fixture.outer_calls);
    let mut provider = FnCallback::new(outer.identity.clone(), move |_, ctx| {
        counter.set(counter.get() + 1);
        let mean = |name: &str| match ctx.inputs.get(name).map(ValidatedOutput::output) {
            Some(CallbackOutput::Mean { value }) => Ok(*value),
            _ => Err(CallbackFailure::new("a mean input is required")),
        };
        Ok(CallbackOutput::Mean { value: mean("left")? + mean("right")? })
    });
    fixture
        .session
        .execute(
            &mut provider,
            outer,
            &fixture.request,
            &fixture.inputs,
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .map_err(Box::new)
}

#[test]
fn an_unregistered_inner_callback_is_refused_and_registration_then_licenses_the_call() {
    let left = law("inner_left", declared_nothing());
    let right = law("inner_right", declared_nothing());
    let outer = outer(&left, &right);
    let mut f = fixture(&left, 2.0, &right, 3.0);
    let history = f.session.history().len();
    let invocations = f.session.invocations();

    // Neither inner callback is registered in the session ledger: overlap cannot be computed
    // and is never assumed absent.
    let refused = run_outer(&mut f, &outer).expect_err("unregistered inner callbacks");
    assert_eq!(refused.refusal.code, antecedent_core::reason_code!("sampling_dependence_unknown"));
    assert_eq!(refused.refusal.detail, "callback_protocol.nested_input_unregistered");
    assert_eq!(refused.refusal.offending.as_deref(), Some("left"));
    assert!(refused.refusal.remedy.is_some());
    assert_eq!(refused.invocations, 0);
    assert_eq!(f.outer_calls.get(), 0, "the outer provider must not run");
    assert_eq!(f.session.history().len(), history);
    assert_eq!(f.session.invocations(), invocations);

    // Registering only the first still leaves the second unnamed in the ledger.
    f.session.register(&left).expect("registers");
    let refused = run_outer(&mut f, &outer).expect_err("one inner callback unregistered");
    assert_eq!(refused.refusal.detail, "callback_protocol.nested_input_unregistered");
    assert_eq!(refused.refusal.offending.as_deref(), Some("right"));
    assert_eq!(f.outer_calls.get(), 0);

    // Both registered, each resting on no observations: independent, so the sum executes.
    f.session.register(&right).expect("registers");
    let digest_before = f.session.ledger().state_digest().to_hex();
    let accepted = run_outer(&mut f, &outer).expect("independent inner callbacks");
    assert_eq!(accepted.output.output().values(), vec![5.0]);
    assert_eq!(accepted.invocations, 1);
    assert_eq!(f.outer_calls.get(), 1);
    assert_eq!(accepted.evidence_state_digest.as_deref(), Some(digest_before.as_str()));
    // The same request is now answered from the stored result under the same ledger state.
    let again = run_outer(&mut f, &outer).expect("reused");
    assert!(again.reused);
    assert_eq!(f.outer_calls.get(), 1);
    assert_eq!(again.evidence_state_digest.as_deref(), Some(digest_before.as_str()));
}

#[test]
fn one_inner_callback_feeding_two_inputs_counts_its_evidence_twice() {
    let only = law("inner_only", declared_nothing());
    let outer = outer(&only, &only);
    let mut f = fixture(&only, 2.0, &only, 2.0);
    f.session.register(&only).expect("registers");
    let history = f.session.history().len();

    let refused = run_outer(&mut f, &outer).expect_err("the same inner callback twice");
    assert_eq!(refused.refusal.code, antecedent_core::reason_code!("evidence_double_counted"));
    assert_eq!(refused.refusal.detail, "callback_protocol.nested_inputs_share_evidence");
    assert_eq!(refused.refusal.offending.as_deref(), Some("left, right"));
    assert_eq!(refused.refusal.supplied.as_deref(), Some("1 shared observation(s)"));
    assert!(refused.refusal.remedy.is_some());
    assert_eq!(refused.invocations, 0);
    assert_eq!(f.outer_calls.get(), 0);
    assert_eq!(f.session.history().len(), history);
}

#[test]
fn inner_callbacks_with_undeclared_evidence_are_unknown_overlap_not_independence() {
    let left = law("inner_left", EvidenceBasis::Undeclared);
    let right = law("inner_right", EvidenceBasis::Undeclared);
    let outer = outer(&left, &right);
    let mut f = fixture(&left, 2.0, &right, 3.0);
    f.session.register(&left).expect("registers");
    f.session.register(&right).expect("registers");
    let history = f.session.history().len();

    let refused = run_outer(&mut f, &outer).expect_err("overlap of undeclared evidence is unknown");
    assert_eq!(refused.refusal.code, antecedent_core::reason_code!("sampling_dependence_unknown"));
    assert_eq!(refused.refusal.detail, "callback_protocol.nested_input_overlap_unknown");
    assert_eq!(refused.refusal.offending.as_deref(), Some("left, right"));
    assert!(refused.refusal.remedy.is_some());
    assert_eq!(refused.invocations, 0);
    assert_eq!(f.outer_calls.get(), 0);
    assert_eq!(f.session.history().len(), history);
}

#[test]
fn a_single_callback_input_has_no_pair_to_decide_and_is_not_refused() {
    // One callback input and one external input: there is no pair of callback inputs, so the
    // ledger is not consulted and no evidence digest is reported.
    let left = law("inner_left", declared_nothing());
    let mut single = law("single", declared_nothing());
    single.kind = CallbackKind::Nested;
    single.inputs = vec![input("left", &left)];
    let mut session = CallbackSession::new();
    let mut inner_provider =
        FnCallback::new(left.identity.clone(), |_, _| Ok(CallbackOutput::Mean { value: 4.0 }));
    let inner = session
        .execute(
            &mut inner_provider,
            &left,
            &CallbackRequest::new(CallbackQuery::Mean, 0),
            &BTreeMap::new(),
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("inner");
    let mut request = CallbackRequest::new(CallbackQuery::Mean, 0);
    request
        .input_digests
        .insert("left".into(), output_values_digest(&inner.output.output().values()));
    let inputs: BTreeMap<String, ValidatedOutput> =
        [("left".to_owned(), inner.output)].into_iter().collect();
    let mut provider = FnCallback::new(single.identity.clone(), |_, ctx| {
        match ctx.inputs.get("left").map(ValidatedOutput::output) {
            Some(CallbackOutput::Mean { value }) => Ok(CallbackOutput::Mean { value: value + 1.0 }),
            _ => Err(CallbackFailure::new("a mean input is required")),
        }
    });
    let accepted = session
        .execute(
            &mut provider,
            &single,
            &request,
            &inputs,
            &CallbackExecutionPolicy::default(),
            &ctx(),
        )
        .expect("one callback input");
    assert_eq!(accepted.output.output().values(), vec![5.0]);
    assert_eq!(accepted.evidence_state_digest, None);
}
