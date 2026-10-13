//! General finite counterfactual transport end to end: prepare once, evaluate
//! and refresh on exact laws, export the `counterfactual_transport_v1`
//! artifact, replay it from bytes and an expected identity in this process and
//! in a fresh one, and refuse every negative gate by name.
//!
//! Every expected number is enumerated by the independent finite-SCM oracle
//! from the target population's own structural model; the route only ever sees
//! the supplied regime laws. The fixtures live with the identifier's tests and
//! are included by path so one oracle serves every layer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    dead_code,
    reason = "test fixtures: dyadic enumerated probabilities and exact bit comparisons are the assertion"
)]

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use antecedent::{
    CounterfactualTransportConsumption, CounterfactualTransportOptions,
    CounterfactualTransportPreparation, PreparedCounterfactualTransport,
    consume_counterfactual_transport_artifact, prepare_counterfactual_transport,
};
use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_io::IoError;
use antecedent_io::counterfactual_transport_artifact::{
    ConsumedCounterfactualTransport, ConsumedCounterfactualTransportOutcome,
    CounterfactualTransportConsumeLimits as Limits, consume_counterfactual_transport_with_limits,
};
use antecedent_io::counterfactual_transport_wire::{
    CounterfactualTransportArtifactError as E, CounterfactualTransportArtifactWire as Wire,
    CounterfactualTransportExpectedIdentity,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, ExpectedClass, Target, all_cases};

const TOLERANCE: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).expect("fixture exists")
}

fn target_labelled<'a>(case: &'a Case, label: &str) -> &'a Target {
    case.targets.iter().find(|t| t.label == label).expect("target exists")
}

fn controllable_for(case: &Case) -> Vec<&'static str> {
    if case.name == "missing_interventional_evidence" {
        vec!["X"]
    } else {
        adapter::evidence_controllable(case)
    }
}

fn names_of(case: &Case) -> Vec<String> {
    case.graph.observed.iter().map(|n| (*n).to_owned()).collect()
}

fn options_of(case: &Case) -> CounterfactualTransportOptions {
    CounterfactualTransportOptions { variable_names: names_of(case), ..Default::default() }
}

fn prepare(
    case: &Case,
    target: &Target,
    with_evidence: bool,
    options: &CounterfactualTransportOptions,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportPreparation, IoError> {
    let problem = adapter::problem(case);
    let adapted = adapter::catalog(case);
    let sources = adapter::sources(case, &controllable_for(case));
    let query = adapter::query(case, &target.query, sources);
    let evidence = with_evidence.then(|| adapter::laws(case, &adapted));
    prepare_counterfactual_transport(&problem, &query, &adapted.catalog, evidence, options, ctx)
}

fn identified(case: &Case, target: &Target) -> PreparedCounterfactualTransport {
    match prepare(case, target, true, &options_of(case), &ctx())
        .unwrap_or_else(|e| panic!("{} / {}: {e}", case.name, target.label))
    {
        CounterfactualTransportPreparation::Identified(prepared) => *prepared,
        other => panic!("{} / {}: expected a derivation, got {other:?}", case.name, target.label),
    }
}

fn refusal(error: &IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => (*code, message.clone()),
        other => panic!("expected a coded refusal, got {other:?}"),
    }
}

fn consume_typed(
    bytes: &[u8],
    expected: &CounterfactualTransportExpectedIdentity,
    ctx: &ExecutionContext,
) -> Result<ConsumedCounterfactualTransport, E> {
    consume_counterfactual_transport_with_limits(bytes, expected, Limits::default(), ctx)
}

fn consumed_probability(consumed: &ConsumedCounterfactualTransport) -> f64 {
    match &consumed.outcome {
        ConsumedCounterfactualTransportOutcome::Identified(identified) => {
            identified.point.probability
        }
        ConsumedCounterfactualTransportOutcome::Obstructed(_) => {
            panic!("expected an identified artifact")
        }
    }
}

/// A producer's exported artifact together with the independent truth.
struct Exported {
    case: Case,
    bytes: Vec<u8>,
    identity: CounterfactualTransportExpectedIdentity,
    truth: f64,
}

fn export_case(name: &str, label: &str) -> Exported {
    let case = case_named(name);
    let target = target_labelled(&case, label).clone();
    let prepared = identified(&case, &target);
    let result = prepared.estimate(&ctx()).unwrap();
    Exported {
        bytes: result.export_artifact().unwrap(),
        identity: result.identity().unwrap(),
        truth: target.expected_value.expect("an identified target is defined"),
        case,
    }
}

/// The two-source conditional fixture most mutations act on.
fn joint_case() -> Exported {
    export_case("joint_counterfactual_two_source_regimes", "Y_{x1}=1,X=0")
}

fn reseal(mut wire: Wire) -> Vec<u8> {
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    antecedent_io::to_cbor(&wire).unwrap()
}

// ---------------------------------------------------------------------------
// Positive: producer -> artifact -> independent consumer equals enumerated truth.
// ---------------------------------------------------------------------------

#[test]
fn identified_oracle_cases_export_and_replay_to_the_enumerated_truth() {
    let mut checked = 0;
    for case in all_cases() {
        let names = names_of(&case);
        for target in case.targets.iter().filter(|t| {
            t.class == ExpectedClass::Identified && t.query.population == case.target_population
        }) {
            let label = format!("{} / {}", case.name, target.label);
            let prepared = identified(&case, target);
            let result = prepared.estimate(&ctx()).unwrap();
            let truth = target.expected_value.expect("an identified target is defined");
            assert!(
                (result.probability() - truth).abs() <= TOLERANCE,
                "{label}: {} against enumerated {truth}",
                result.probability()
            );
            assert!(result.cells_read() > 0, "{label}");
            assert!(result.conditioning_mass_checked(), "{label}");
            assert!(!result.cited_laws().is_empty(), "{label}");

            let bytes = result.export_artifact().unwrap();
            let identity = result.identity().unwrap();
            // The typed independent consumer re-decides and re-evaluates.
            let typed = consume_typed(&bytes, &identity, &ctx())
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(consumed_probability(&typed).to_bits(), result.probability().to_bits());
            assert!((consumed_probability(&typed) - truth).abs() <= TOLERANCE, "{label}");
            assert_eq!(typed.wire.variable_names, names, "{label}");
            // The facade consumer returns the same point and the same laws read.
            let CounterfactualTransportConsumption::Identified(again) =
                consume_counterfactual_transport_artifact(
                    &bytes,
                    &identity,
                    Limits::default(),
                    &ctx(),
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"))
            else {
                panic!("{label}: expected an identified consumption");
            };
            assert_eq!(again.probability().to_bits(), result.probability().to_bits());
            assert_eq!(again.numerator().to_bits(), result.numerator().to_bits());
            assert_eq!(again.denominator().to_bits(), result.denominator().to_bits());
            assert_eq!(again.cited_laws(), result.cited_laws(), "{label}");
            assert_eq!(again.identity().unwrap(), identity, "{label}");
            checked += 1;
        }
    }
    assert!(checked >= 7, "the named positive queries were exercised ({checked})");
}

#[test]
fn natural_direct_and_indirect_effects_replay_to_the_enumerated_contrasts() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let value = |label: &str| {
        let export = export_case(case.name, label);
        // Replayed from bytes only, against the independent enumeration.
        let consumed = consume_typed(&export.bytes, &export.identity, &ctx()).unwrap();
        let probability = consumed_probability(&consumed);
        assert!((probability - export.truth).abs() <= TOLERANCE, "{label}");
        (probability, export.truth)
    };
    let (y10, t10) = value("Y_{x1,M_{x0}}");
    let (y00, t00) = value("Y_{x0,M_{x0}}");
    let (y01, t01) = value("Y_{x0,M_{x1}}");
    let nde = y10 - y00;
    let nie = y01 - y00;
    assert!((nde - (t10 - t00)).abs() <= TOLERANCE, "NDE {nde} vs {}", t10 - t00);
    assert!((nie - (t01 - t00)).abs() <= TOLERANCE, "NIE {nie} vs {}", t01 - t00);
    // Not degenerate: both effects are nonzero in a nonlinear mediation model.
    assert!((t10 - t00).abs() > 1e-3 && (t01 - t00).abs() > 1e-3);
}

#[test]
fn two_source_regimes_conditional_evidence_and_latent_districts_cite_what_they_read() {
    // Two source regimes: the source observational and do(X) laws plus the
    // target marginal, jointly.
    let case = case_named("joint_counterfactual_two_source_regimes");
    let prepared = identified(&case, &case.targets[0]);
    let result = prepared.estimate(&ctx()).unwrap();
    let cited: Vec<(&str, usize)> = result
        .cited_laws()
        .iter()
        .map(|law| (law.population.as_ref(), law.interventions.len()))
        .collect();
    assert!(cited.contains(&("source", 0)), "{cited:?}");
    assert!(cited.contains(&("source", 1)), "{cited:?}");
    assert!(cited.contains(&("target", 0)), "{cited:?}");

    // Conditional factual evidence: the effect of treatment on the treated is
    // the ratio of two derived probabilities, equal to the enumerated ratio.
    let case = case_named("effect_of_treatment_on_treated_with_selection");
    let prepared = identified(&case, &case.targets[0]);
    let result = prepared.estimate(&ctx()).unwrap();
    let truth = case.targets[0].expected_value.unwrap();
    assert!(result.denominator() > 0.0 && result.denominator() < 1.0);
    assert!((result.numerator() / result.denominator() - truth).abs() <= TOLERANCE);
    assert_eq!(result.probability().to_bits(), (result.numerator() / result.denominator()).to_bits());

    // A transportable latent district: the target never measures Y.
    let case = case_named("transportable_latent_district");
    let prepared = identified(&case, &case.targets[0]);
    let result = prepared.estimate(&ctx()).unwrap();
    let truth = case.targets[0].expected_value.unwrap();
    assert!((result.probability() - truth).abs() <= TOLERANCE);
    let outcome = adapter::variable_id(&case, "Y");
    let target_laws: Vec<&ExactDiscreteLaw> =
        result.data().laws().iter().filter(|l| l.population() == "target").collect();
    assert!(!target_laws.is_empty());
    assert!(
        target_laws.iter().all(|law| law.axes().iter().all(|axis| axis.variable != outcome)),
        "the target supplies no law of Y"
    );
}

// ---------------------------------------------------------------------------
// Lifecycle: prepare once, refresh rebuilds numbers, failures keep state.
// ---------------------------------------------------------------------------

fn shifted_mediation() -> Case {
    // The target population's covariate law changes; the structure does not.
    let case = case_named("nonlinear_mediation_covariate_shift");
    let source = finite_scm_oracle::population(&case.populations, "source").clone();
    let new_target = finite_scm_oracle::population(&case.populations, "target")
        .with_exo_law("Uz", finite_scm_oracle::bern(0.5));
    let mut shifted = case;
    shifted.populations = vec![
        finite_scm_oracle::Population::new("source", source),
        finite_scm_oracle::Population::new("target", new_target),
    ];
    shifted
}

#[test]
fn prepare_once_then_refresh_rebuilds_the_numbers_and_a_refused_refresh_keeps_state() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    // Y_{x0,M_{x0}} depends on the covariate law (Y_{x1,M_{x0}} does not).
    let target = target_labelled(&case, "Y_{x0,M_{x0}}").clone();
    let before_truth = target.expected_value.unwrap();

    // Prepared without evidence: decided once, nothing bound, no number.
    let unbound = match prepare(&case, &target, false, &options_of(&case), &ctx()).unwrap() {
        CounterfactualTransportPreparation::Identified(prepared) => *prepared,
        other => panic!("expected a derivation, got {other:?}"),
    };
    assert!(unbound.data().is_none());
    let error = unbound.estimate(&ctx()).unwrap_err();
    assert_eq!(refusal(&error).0, "transport_missing_provider");
    let canonical = unbound.derivation().canonical_text();
    let consumed_ops = unbound.derivation().search.operations_consumed;
    assert!(consumed_ops > 0);

    // Evaluate on supplied evidence without binding it.
    let adapted = adapter::catalog(&case);
    let original = adapter::laws(&case, &adapted);
    let one_off = unbound.evaluate(&original, &ctx()).unwrap();
    assert!((one_off.probability() - before_truth).abs() <= TOLERANCE);
    assert!(unbound.data().is_none(), "evaluate does not bind");

    // Refresh binds the evidence and builds the numerical state.
    let first = unbound.refresh(original.clone(), &ctx()).unwrap();
    let state = first.estimate(&ctx()).unwrap().probability();
    assert!((state - before_truth).abs() <= TOLERANCE);

    // Same shape, changed numbers: the new enumerated truth, not the stale one.
    let shifted = shifted_mediation();
    let shifted_truth = target.query.value(&shifted.populations).unwrap();
    assert!((shifted_truth - before_truth).abs() > 1e-3, "the shift moves the target value");
    let shifted_catalog = adapter::catalog(&shifted);
    let shifted_laws = adapter::laws(&shifted, &shifted_catalog);
    let second = first.refresh(shifted_laws, &ctx()).unwrap();
    let moved = second.estimate(&ctx()).unwrap().probability();
    assert!((moved - shifted_truth).abs() <= TOLERANCE, "{moved} vs {shifted_truth}");
    // The derivation is the one decided once: no search happened again.
    assert_eq!(second.derivation().canonical_text(), canonical);
    assert_eq!(second.derivation().search.operations_consumed, consumed_ops);
    // The earlier prepared value is untouched.
    assert_eq!(first.estimate(&ctx()).unwrap().probability().to_bits(), state.to_bits());

    // A refused refresh (a counted law) and a cancelled one leave the state.
    let mut laws = original.laws().to_vec();
    let law = &laws[0];
    laws[0] = ExactDiscreteLaw::try_empirical(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap();
    let counted = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let error = second.refresh(counted, &ctx()).unwrap_err();
    assert_eq!(refusal(&error).0, "cell_not_licensed");
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = second.refresh(original, &cancelled).unwrap_err();
    assert_eq!(refusal(&error).0, "transport_budget_cancel");
    assert_eq!(second.estimate(&ctx()).unwrap().probability().to_bits(), moved.to_bits());

    // Refreshing exported evidence changes the data identity, not the premises.
    let first_identity = first.estimate(&ctx()).unwrap().identity().unwrap();
    let second_identity = second.estimate(&ctx()).unwrap().identity().unwrap();
    assert_eq!(first_identity.premises_digest, second_identity.premises_digest);
    assert_ne!(first_identity.data_digest, second_identity.data_digest);
}

#[test]
fn evaluation_is_deterministic_and_invariant_to_the_order_of_supplied_laws() {
    let case = case_named("effect_of_treatment_on_treated_with_selection");
    let prepared = identified(&case, &case.targets[0]);
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let first = prepared.evaluate(&data, &ctx()).unwrap();
    let again = prepared.evaluate(&data, &ctx()).unwrap();
    assert_eq!(first.probability().to_bits(), again.probability().to_bits());
    let mut laws = data.laws().to_vec();
    laws.reverse();
    let reversed = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let permuted = prepared.evaluate(&reversed, &ctx()).unwrap();
    assert_eq!(first.probability().to_bits(), permuted.probability().to_bits());
    // Exports of the permuted laws are the same artifact: canonical order.
    assert_eq!(first.export_artifact().unwrap(), permuted.export_artifact().unwrap());
}

// ---------------------------------------------------------------------------
// Obstructions: a replayed two-model witness, never a mere failure to derive.
// ---------------------------------------------------------------------------

#[test]
fn obstructions_export_and_replay_their_two_model_witness_from_bytes() {
    for name in ["recanting_witness_without_selection", "selection_on_outcome_mechanism"] {
        let case = case_named(name);
        let target = &case.targets[0];
        let report = match prepare(&case, target, false, &options_of(&case), &ctx()).unwrap() {
            CounterfactualTransportPreparation::ProvenNonTransportable(report) => *report,
            other => panic!("{name}: expected an obstruction, got {other:?}"),
        };
        // The fixture's own alternative model family witnesses the same query,
        // independently of the identifier's witness.
        let alternative = case.alternative.as_ref().expect("an obstruction fixture");
        let oracle = finite_scm_oracle::verify_non_identifiability_witness(
            &case.populations,
            alternative,
            &case.evidence,
            &target.query,
            1e-15,
            1e-6,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(oracle.gap > 1e-6, "{name}");
        let check = report.verify().unwrap();
        assert_ne!(check.first_value, check.second_value, "{name}");
        assert!(check.cells_compared > 0, "{name}");

        let bytes = report.export_artifact().unwrap();
        let wire = Wire::decode(&bytes).unwrap();
        assert!(wire.laws.is_empty(), "an obstruction carries no law table");
        let consumed = consume_typed(&bytes, &wire.identity(), &ctx())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ConsumedCounterfactualTransportOutcome::Obstructed(obstructed) = &consumed.outcome
        else {
            panic!("{name}: expected an obstruction");
        };
        assert_eq!(obstructed.check, check, "{name}");
        // The facade consumer returns the verified obstruction.
        let CounterfactualTransportConsumption::ProvenNonTransportable(again) =
            consume_counterfactual_transport_artifact(
                &bytes,
                &wire.identity(),
                Limits::default(),
                &ctx(),
            )
            .unwrap()
        else {
            panic!("{name}: expected an obstruction consumption");
        };
        assert_eq!(again.verify().unwrap(), check, "{name}");

        // A tampered witness, resealed, is refused in the obstruction slot.
        let mut tampered = wire.clone();
        let stored = tampered.obstruction.as_mut().unwrap();
        stored.witness.second = stored.witness.first.clone();
        let error = consume_typed(&reseal(tampered), &CounterfactualTransportExpectedIdentity::default(), &ctx())
            .unwrap_err();
        assert!(matches!(error, E::ObstructionInvalid(_)), "{name}: {error:?}");
        assert_eq!(error.slot(), "obstruction");
        let (code, message) = error.refusal();
        assert_eq!(code, "transport_not_certified");
        assert!(message.starts_with("counterfactual_transport.invalid_witness"), "{message}");

        // Stored values that the witness does not establish are refused too.
        let mut tampered = wire;
        tampered.obstruction.as_mut().unwrap().first_value = "1/1".into();
        let error = consume_typed(&reseal(tampered), &CounterfactualTransportExpectedIdentity::default(), &ctx())
            .unwrap_err();
        assert_eq!(error.slot(), "obstruction", "{name}: {error:?}");
    }
}

// ---------------------------------------------------------------------------
// Negative gates of the producer: missing factors, zero support, labels, bounds.
// ---------------------------------------------------------------------------

#[test]
fn refusals_name_what_would_license_the_request() {
    // Missing factors: the source experiment is declared available but not
    // supplied. Identification is possible if it were: it is named.
    let case = case_named("missing_interventional_evidence");
    let error =
        prepare(&case, &case.targets[0], false, &options_of(&case), &ctx()).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.starts_with("counterfactual_transport.missing_regime"), "{message}");
    assert!(message.contains("experiment on"), "{message}");
    assert!(message.contains("source"), "{message}");

    // Zero support: the conditioning event has probability zero in the target.
    let case = case_named("zero_support_conditioning_event");
    let target = &case.targets[1];
    assert_eq!(target.class, ExpectedClass::UndefinedZeroSupport);
    let error = prepare(&case, target, true, &options_of(&case), &ctx()).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("counterfactual_transport.zero_conditioning_mass"), "{message}");
    // The same query is defined, and enumerated, in the source population.
    let mut in_source = case.clone();
    in_source.target_population = "source";
    let prepared = identified(&in_source, &case.targets[0]);
    let value = prepared.estimate(&ctx()).unwrap().probability();
    let truth = case.targets[0].expected_value.unwrap();
    assert!((truth - 0.25).abs() <= TOLERANCE);
    assert!((value - truth).abs() <= TOLERANCE);

    // Budget and cancellation are coded stops, never verdicts.
    let case = case_named("nonlinear_mediation_covariate_shift");
    let target = &case.targets[0];
    let starved = CounterfactualTransportOptions {
        search: SearchLimits { operations: 1, depth: 8 },
        ..options_of(&case)
    };
    let error = prepare(&case, target, false, &starved, &ctx()).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_budget_cancel");
    assert!(message.starts_with("counterfactual_transport.budget"), "{message}");
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = prepare(&case, target, false, &options_of(&case), &cancelled).unwrap_err();
    assert_eq!(refusal(&error).0, "transport_budget_cancel");
    // Limits above the contract's maxima are refused by bound.
    let oversized = CounterfactualTransportOptions {
        search: SearchLimits { operations: usize::MAX, depth: 8 },
        ..options_of(&case)
    };
    let error = prepare(&case, target, false, &oversized, &ctx()).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("counterfactual_transport.bounds_exceeded"), "{message}");
    // Names that do not cover the variables are refused, not truncated.
    let misnamed = CounterfactualTransportOptions {
        variable_names: vec!["only".into()],
        ..Default::default()
    };
    let error = prepare(&case, target, false, &misnamed, &ctx()).unwrap_err();
    assert_eq!(refusal(&error).0, "invalid_argument");
}

fn rebuilt(law: &ExactDiscreteLaw, population: &str) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        population,
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap()
}

#[test]
fn absent_mislabelled_and_counted_evidence_is_refused_before_a_number() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let prepared = identified(&case, &case.targets[0]);
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    prepared.evaluate(&data, &ctx()).unwrap();

    // The source experiment at one level is withheld: its factor has no law.
    let kept: Vec<ExactDiscreteLaw> = data
        .laws()
        .iter()
        .filter(|law| {
            !(law.population() == "source"
                && law.interventions().iter().any(|a| a.value.as_f64() == Some(1.0)))
        })
        .cloned()
        .collect();
    assert!(kept.len() < data.laws().len());
    let thin = ExactTransportData::try_new(kept, 1 << 16).unwrap();
    let error = prepared.evaluate(&thin, &ctx()).unwrap_err();
    assert_eq!(refusal(&error).0, "transport_missing_provider");

    // A source law relabelled as the target's: the catalog does not bind that
    // population to that regime.
    let mut laws = data.laws().to_vec();
    let first_source = laws.iter().position(|l| l.population() == "source").unwrap();
    laws[first_source] = rebuilt(&laws[first_source], "target");
    let swapped = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    assert!(prepared.evaluate(&swapped, &ctx()).is_err());

    // An empirical plug-in is not licensed: sampled inference is another route.
    let mut laws = data.laws().to_vec();
    let law = &laws[0];
    laws[0] = ExactDiscreteLaw::try_empirical(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap();
    let empirical = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let error = prepared.evaluate(&empirical, &ctx()).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("counterfactual_transport.sampled_inference_closed"), "{message}");
}

// ---------------------------------------------------------------------------
// The consumer trusts nothing: resealed semantic mutations by slot.
// ---------------------------------------------------------------------------

#[test]
fn a_resealed_unmutated_artifact_still_replays() {
    let export = joint_case();
    let wire = Wire::decode(&export.bytes).unwrap();
    let consumed = consume_typed(&reseal(wire.clone()), &wire.identity(), &ctx()).unwrap();
    assert!((consumed_probability(&consumed) - export.truth).abs() <= TOLERANCE);
}

#[test]
fn resealed_query_selection_table_and_derivation_mutations_are_refused_by_slot() {
    let export = joint_case();
    let wire = Wire::decode(&export.bytes).unwrap();
    let outcome = adapter::variable_id(&export.case, "Y").raw();
    let none = CounterfactualTransportExpectedIdentity::default();

    // Query: the event's level moved. The stored derivation is another
    // query's, so the re-decision does not reproduce it.
    let mut mutated = wire.clone();
    let atom = mutated
        .query
        .event
        .event
        .iter_mut()
        .find(|atom| atom.2 == 1.0f64.to_bits())
        .expect("the event asks for a level one");
    atom.2 = 0.0f64.to_bits();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::DerivationMismatch(_)), "{error:?}");
    assert_eq!(error.slot(), "derivation");

    // Selection: the source's mechanism of the outcome may also differ, so the
    // source experiment cannot be borrowed and the stored derivation is not
    // reproduced (the decision is no longer an identification).
    let mut mutated = wire.clone();
    mutated.query.sources[0].selection_targets.push(outcome);
    mutated.query.sources[0].selection_targets.sort_unstable();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::DerivationMismatch(_)), "{error:?}");
    assert_eq!(error.slot(), "derivation");
    let (code, message) = error.refusal();
    assert_eq!(code, "transport_not_certified");
    assert!(message.starts_with("counterfactual_transport.invalid_derivation"), "{message}");

    // Selection removed: the target marginal is no longer needed, so the
    // derivation differs (or the declaration is refused by name).
    let mut mutated = wire.clone();
    mutated.query.sources[0].selection_targets.clear();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error.slot(), "derivation" | "replay"), "{error:?}");

    // Target population relabelled.
    let mut mutated = wire.clone();
    mutated.query.target = "elsewhere".into();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error.slot(), "derivation" | "replay"), "{error:?}");

    // Table: the target's covariate marginal is swapped. The tables are
    // internally valid exact laws, so the point differs, and only the result
    // slot says so.
    let mut mutated = wire.clone();
    let index = mutated
        .laws
        .iter()
        .position(|law| {
            law.population == "target"
                && law.axes.len() == 1
                && law.probabilities.len() == 2
                && law.probabilities[0] != law.probabilities[1]
        })
        .expect("the target marginal of the covariate");
    mutated.laws[index].probabilities.swap(0, 1);
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::ResultMismatch { .. }), "{error:?}");
    assert_eq!(error.slot(), "result");

    // Derivation: the stored text or receipt is edited; the replay is not.
    let mut mutated = wire.clone();
    mutated.derivation.as_mut().unwrap().canonical_text.push_str(";edited");
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::DerivationMismatch(_)), "{error:?}");
    let mut mutated = wire.clone();
    mutated.derivation.as_mut().unwrap().search.operations_consumed += 1;
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::DerivationMismatch(_)), "{error:?}");

    // Result: an edited number or law identity never survives the replay,
    // resealed or not.
    let mut mutated = wire.clone();
    mutated.result.as_mut().unwrap().probability = (export.truth + 0.125).to_bits();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::ResultMismatch { field: "probability" });
    let mut mutated = wire.clone();
    mutated.result.as_mut().unwrap().cited_laws[0].snapshot_identity.push('x');
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::ResultMismatch { field: "cited_laws" });

    // Labels: a regime relabelled to a population the query does not declare,
    // and a law relabelled to another population.
    let mut mutated = wire.clone();
    let regime = mutated.catalog.regimes.iter_mut().find(|r| r.population == "source").unwrap();
    regime.population = "nowhere".into();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error.slot(), "replay" | "catalog"), "{error:?}");
    let (code, message) = error.refusal();
    assert_eq!(code, "invalid_argument");
    assert!(
        message.contains("counterfactual_transport.invalid_catalog")
            || message.contains("counterfactual_transport.invalid_artifact"),
        "{message}"
    );
    let mut mutated = wire.clone();
    let index = mutated.laws.iter().position(|l| l.population == "source").unwrap();
    mutated.laws[index].population = "target".into();
    let error = consume_typed(&reseal(mutated), &none, &ctx()).unwrap_err();
    assert!(matches!(error.slot(), "replay" | "laws" | "result"), "{error:?}");
}

#[test]
fn edits_that_are_not_resealed_identity_versions_and_limits_are_refused() {
    let export = joint_case();
    let wire = Wire::decode(&export.bytes).unwrap();
    let none = CounterfactualTransportExpectedIdentity::default();
    consume_typed(&export.bytes, &export.identity, &ctx()).unwrap();

    // A table edit without resealing is a data-identity mismatch; a premise edit
    // without resealing is a premises mismatch.
    let mut edited = wire.clone();
    edited.laws[0].snapshot.push('x');
    let error = consume_typed(&antecedent_io::to_cbor(&edited).unwrap(), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::DataIdentityMismatch);
    assert_eq!(error.slot(), "data_digest");
    let mut edited = wire.clone();
    edited.query.target = "elsewhere".into();
    let error = consume_typed(&antecedent_io::to_cbor(&edited).unwrap(), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::PremisesMismatch);

    // A self-consistent artifact that is not the one the caller expects.
    let other = export_case("nonlinear_mediation_covariate_shift", "Y_{x0,M_{x0}}");
    let error = consume_typed(&export.bytes, &other.identity, &ctx()).unwrap_err();
    assert!(matches!(error, E::ExpectedIdentityMismatch { .. }), "{error:?}");
    assert_eq!(error.slot(), "expected_identity");
    // Either digest alone is enough to bind the expectation.
    let data_only = CounterfactualTransportExpectedIdentity {
        premises_digest: None,
        data_digest: Some(other.identity.data_digest.clone().unwrap()),
    };
    assert!(matches!(
        consume_typed(&export.bytes, &data_only, &ctx()).unwrap_err(),
        E::ExpectedIdentityMismatch { slot: "data" }
    ));

    // Another version and another required feature are refused, not guessed.
    let mut future = wire.clone();
    future.version = 2;
    let error = consume_typed(&antecedent_io::to_cbor(&future).unwrap(), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::UnsupportedVersion { version: 2 });
    let mut foreign = wire.clone();
    foreign.required_features = vec!["counterfactual_transport_v2".into()];
    let error = consume_typed(&antecedent_io::to_cbor(&foreign).unwrap(), &none, &ctx()).unwrap_err();
    assert_eq!(error, E::UnsupportedSemantics("required_features"));
    let mut extra = wire.clone();
    extra.required_features.push("another".into());
    let error = consume_typed(&antecedent_io::to_cbor(&extra).unwrap(), &none, &ctx()).unwrap_err();
    assert_eq!(error.slot(), "required_features");

    // An unknown field is a decoding failure of the whole artifact.
    let mut value: ciborium::Value = ciborium::from_reader(&export.bytes[..]).unwrap();
    let ciborium::Value::Map(entries) = &mut value else { panic!("an artifact is a map") };
    entries.push((ciborium::Value::Text("unexpected".into()), ciborium::Value::Integer(1.into())));
    let mut bytes = Vec::new();
    ciborium::into_writer(&value, &mut bytes).unwrap();
    let error = consume_typed(&bytes, &none, &ctx()).unwrap_err();
    assert!(matches!(error, E::Decode(_)), "{error:?}");
    assert_eq!(error.slot(), "artifact");

    // Consumer limits are checked before any work and are retryable bounds.
    let small_search = Limits {
        search: SearchLimits { operations: 1, depth: 1 },
        ..Limits::default()
    };
    let error =
        consume_counterfactual_transport_with_limits(&export.bytes, &none, small_search, &ctx())
            .unwrap_err();
    assert_eq!(error, E::LimitsExceeded("search operation limit"));
    assert_eq!(error.refusal().0, "route_not_supported");
    let small_evaluation = Limits {
        evaluation: ExactEvaluationLimits { operations: 1, depth: 1 },
        ..Limits::default()
    };
    let error =
        consume_counterfactual_transport_with_limits(&export.bytes, &none, small_evaluation, &ctx())
            .unwrap_err();
    assert_eq!(error, E::LimitsExceeded("operation limit"));
    let few_laws = Limits { max_laws: 1, ..Limits::default() };
    let error = consume_counterfactual_transport_with_limits(&export.bytes, &none, few_laws, &ctx())
        .unwrap_err();
    assert_eq!(error, E::LimitsExceeded("law count"));
    let small_memory = Limits { search_memory_bytes: 1, ..Limits::default() };
    let error =
        consume_counterfactual_transport_with_limits(&export.bytes, &none, small_memory, &ctx())
            .unwrap_err();
    assert_eq!(error, E::LimitsExceeded("search memory limit"));
    // The facade surfaces the same refusal as a coded error.
    let error = consume_counterfactual_transport_artifact(&export.bytes, &none, small_search, &ctx())
        .unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("counterfactual_transport.bounds_exceeded: limits:"), "{message}");

    // A cancelled context stops the replay with a coded budget refusal.
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = consume_typed(&export.bytes, &export.identity, &cancelled).unwrap_err();
    assert!(matches!(error, E::Replay { code: "transport_budget_cancel", .. }), "{error:?}");
    assert_eq!(error.slot(), "replay");
}

// ---------------------------------------------------------------------------
// A fresh process holding only the file and the expected identity.
// ---------------------------------------------------------------------------

#[test]
fn a_fresh_process_consumes_only_the_bytes_and_matches_the_oracle() {
    const PATH: &str = "ANTECEDENT_COUNTERFACTUAL_TRANSPORT_REPLAY_PATH";
    const PREMISES: &str = "ANTECEDENT_COUNTERFACTUAL_TRANSPORT_REPLAY_PREMISES";
    const DATA: &str = "ANTECEDENT_COUNTERFACTUAL_TRANSPORT_REPLAY_DATA";
    const TRUTH: &str = "ANTECEDENT_COUNTERFACTUAL_TRANSPORT_REPLAY_TRUTH_BITS";
    if let Ok(path) = std::env::var(PATH) {
        // The consumer holds the bytes, the expected identity and the oracle's
        // enumerated value: no producer, fixture or model is built here.
        let bytes = std::fs::read(path).unwrap();
        let expected = CounterfactualTransportExpectedIdentity {
            premises_digest: Some(std::env::var(PREMISES).unwrap()),
            data_digest: Some(std::env::var(DATA).unwrap()),
        };
        let truth = f64::from_bits(std::env::var(TRUTH).unwrap().parse::<u64>().unwrap());
        let consumed = consume_typed(&bytes, &expected, &ctx()).unwrap();
        assert!((consumed_probability(&consumed) - truth).abs() <= TOLERANCE);
        // Another expected identity is refused in the fresh process too.
        let wrong = CounterfactualTransportExpectedIdentity {
            premises_digest: Some("not-the-identity".into()),
            data_digest: None,
        };
        assert!(consume_typed(&bytes, &wrong, &ctx()).is_err());
        return;
    }
    let export = export_case("nonlinear_mediation_covariate_shift", "Y_{x0,M_{x1}}");
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir()
        .join(format!("antecedent-counterfactual-transport-{}-{unique}.bin", std::process::id()));
    std::fs::write(&path, &export.bytes).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("a_fresh_process_consumes_only_the_bytes_and_matches_the_oracle")
        .env(PATH, &path)
        .env(PREMISES, export.identity.premises_digest.clone().unwrap())
        .env(DATA, export.identity.data_digest.clone().unwrap())
        .env(TRUTH, export.truth.to_bits().to_string())
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
