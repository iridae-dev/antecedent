//! Exact execution of one finite two-step temporal transport sequence (2.2A, X5).
//!
//! The fixture is the known temporal SCM of `common::temporal_fixture`: a
//! baseline `b`, step covariates `l1` and `l2` (the latter a time-varying
//! confounder affected by the first action), two actions and an outcome, with
//! source/target mechanism differences at the initial state and at step 2 and
//! two latent confounders. Every expected value is enumerated from the
//! structural model, never from the formula under test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::{ExecutionContext, SearchLimits, SearchStop, Value};
use antecedent_estimate::EstimationError;
use antecedent_estimate::temporal_transport::{
    PreparedTemporalSequence, TEMPORAL_INFERENCE_CLAIM, history_support, prepare_temporal_sequence,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::sid::temporal_sequence::{
    MechanismAssumption, TEMPORAL_HORIZON_DETAIL, TEMPORAL_INVALID_SEQUENCE, TemporalOutcome,
    TemporalSequenceDecision, TemporalSequenceError, decide_temporal_transport_sequence,
};
use common::temporal_fixture::{
    A1, A2, B, L1, L2, Y, action_catalog, action_law, baseline_only_spec, baseline_shift_source,
    catalog, catalog_with_unused_regime, data_of, laws, observational_source_law, sequence,
    source_scm, spec, target_law, target_scm, truth, unrolled, v,
};

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn decide(seq: &[Value], budget: SearchLimits) -> TemporalSequenceDecision {
    decide_temporal_transport_sequence(&spec(), seq, "source", "target", &catalog(), budget, &ctx())
        .unwrap()
}

fn prepare(
    seq: &[Value],
    skip: &[(u8, u8, u8)],
) -> Result<PreparedTemporalSequence, EstimationError> {
    prepare_temporal_sequence(
        decide(seq, BUDGET),
        laws(&source_scm(), &target_scm(), skip),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
}

fn refusal(result: Result<PreparedTemporalSequence, EstimationError>) -> (&'static str, String) {
    match result {
        Err(EstimationError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn the_whole_sequence_matches_the_target_interventional_truth() {
    for (a1, a2) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
        let seq = sequence(f64::from(a1), f64::from(a2));
        let report = prepare(&seq, &[]).unwrap().evaluate(&ctx()).unwrap();
        let expected = truth(&target_scm(), [a1, a2]);
        assert!((report.mean - expected).abs() < 1e-12, "{a1}{a2}: {} vs {expected}", report.mean);
        assert_eq!((report.horizon, report.inference_claim), (2, TEMPORAL_INFERENCE_CLAIM));
        // The mechanism change matters: the source's own answer is different.
        let source = truth(&source_scm(), [a1, a2]);
        assert!((source - expected).abs() > 0.05, "{a1}{a2}: source {source} vs target {expected}");
    }
}

#[test]
fn the_sequence_is_one_intervention_and_never_per_step_products() {
    // Two comparators that treat the steps separately, both far from the truth:
    // (1) transport each step alone and combine: the target's own `P*(l2 | a1)`
    //     times the source's response to `do(a2)` given `l2` in its natural
    //     course, ignoring that `l2` follows the first action and confounds the
    //     second; (2) the sequential g-formula read straight off the target's
    //     observational law, which ignores the latent confounding of each action.
    let (source, target) = (source_scm(), target_scm());
    let joint = target.law(&[], &[B, L1, A1, L2, A2, Y]);
    // Mass of the cells whose coordinates `b, l1, a1, l2, a2, y` satisfy `keep`.
    let mass = |keep: &dyn Fn([usize; 6]) -> bool| -> f64 {
        (0..64usize)
            .filter(|i| {
                keep([(i >> 5) & 1, (i >> 4) & 1, (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
            })
            .map(|i| joint[i])
            .sum()
    };
    let (mut narrowest_step, mut widest_g) = (f64::MAX, 0.0f64);
    for (a1, a2) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
        let expected = truth(&target, [a1, a2]);
        let l2_given_a1 = target.law(&[(A1, a1)], &[L2]);
        let by_step = source.law(&[(A2, a2)], &[L2, Y]);
        let per_step: f64 = (0..2usize)
            .map(|l2| {
                let mass = by_step[l2 * 2] + by_step[l2 * 2 + 1];
                l2_given_a1[l2] * by_step[l2 * 2 + 1] / mass
            })
            .sum();
        // Sequential g-formula in the target: sum P(b, l1) P(l2 | b, l1, a1) P(y | history).
        let (a1u, a2u) = (usize::from(a1), usize::from(a2));
        let mut g = 0.0;
        for b in 0..2 {
            for l1 in 0..2 {
                let start = mass(&|c| c[0] == b && c[1] == l1);
                let first = mass(&|c| c[0] == b && c[1] == l1 && c[2] == a1u);
                for l2 in 0..2 {
                    let at_l2 = mass(&|c| c[..4] == [b, l1, a1u, l2]);
                    let at_a2 = mass(&|c| c[..5] == [b, l1, a1u, l2, a2u]);
                    let at_y = mass(&|c| c == [b, l1, a1u, l2, a2u, 1]);
                    g += start * (at_l2 / first) * (at_y / at_a2);
                }
            }
        }
        narrowest_step = narrowest_step.min((per_step - expected).abs());
        widest_g = widest_g.max((g - expected).abs());
        let report = prepare(&sequence(f64::from(a1), f64::from(a2)), &[])
            .unwrap()
            .evaluate(&ctx())
            .unwrap();
        assert!((report.mean - expected).abs() < 1e-12);
    }
    // The per-step product is off by a wide margin at every one of the four
    // sequences; the target-only g-formula, which ignores the latent confounding,
    // is off by a wide margin at its worst sequence.
    assert!(narrowest_step > 0.05, "per-step product gap {narrowest_step}");
    assert!(widest_g > 0.05, "target-only g-formula gap {widest_g}");
}

#[test]
fn sequence_order_matters() {
    let ab = prepare(&sequence(0.0, 1.0), &[]).unwrap().evaluate(&ctx()).unwrap();
    let ba = prepare(&sequence(1.0, 0.0), &[]).unwrap().evaluate(&ctx()).unwrap();
    assert!((ab.mean - truth(&target_scm(), [0, 1])).abs() < 1e-12);
    assert!((ba.mean - truth(&target_scm(), [1, 0])).abs() < 1e-12);
    assert!(
        (ab.mean - ba.mean).abs() > 0.2,
        "[a,b] and [b,a] must differ by a wide margin: {} {}",
        ab.mean,
        ba.mean
    );
    assert_ne!(ab.sequence[0].as_f64(), ba.sequence[0].as_f64());
    // The two orders are different questions with different identities.
    let (dab, dba) = (decide(&sequence(0.0, 1.0), BUDGET), decide(&sequence(1.0, 0.0), BUDGET));
    assert_ne!(dab.sequence, dba.sequence);
}

#[test]
fn time_varying_confounding_and_every_invariance_are_explicit() {
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    // l2 follows a1, confounds a2 and y: identified jointly, not per step.
    assert_eq!(decision.time_varying_confounders, vec![v(L2)]);
    let by = |variable: usize| {
        decision.invariances.iter().find(|i| i.variable == v(variable)).unwrap().clone()
    };
    assert_eq!((by(B).slice, by(B).assumption), (0, MechanismAssumption::DiffersBySelection));
    assert_eq!((by(L1).slice, by(L1).assumption), (1, MechanismAssumption::Invariant));
    assert_eq!((by(L2).slice, by(L2).assumption), (2, MechanismAssumption::DiffersBySelection));
    assert_eq!((by(Y).slice, by(Y).assumption), (2, MechanismAssumption::Invariant));
    // The outcome's mechanism is borrowed from the source; the shifted ones are not.
    assert!(by(Y).borrowed_from_source);
    assert!(!by(B).borrowed_from_source && !by(L2).borrowed_from_source);
    // Actions carry no mechanism assumption: the sequence sets them.
    assert!(decision.invariances.iter().all(|i| i.variable != v(A1) && i.variable != v(A2)));
    // Which evidence exists at each step.
    let touching = |slice: u8| {
        decision
            .evidence
            .iter()
            .filter(|e| e.slice == slice)
            .map(|e| e.population.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(touching(0), ["source", "target"]);
    assert_eq!(touching(2), ["source", "target"]);
    // Without l2 -> a2 the second action has no time-varying confounder.
    let mut edges = vec![
        (B, L1),
        (L1, A1),
        (B, L2),
        (A1, L2),
        (A1, A2),
        (B, Y),
        (L1, Y),
        (A1, Y),
        (L2, Y),
        (A2, Y),
    ];
    edges.retain(|e| *e != (L2, A2));
    let plain = unrolled(&[B, L2], &edges);
    let decision = decide_temporal_transport_sequence(
        &plain,
        &sequence(1.0, 0.0),
        "source",
        "target",
        &catalog(),
        BUDGET,
        &ctx(),
    )
    .unwrap();
    assert!(decision.time_varying_confounders.is_empty());
}

#[test]
fn a_reached_history_without_source_support_refuses() {
    // The source supplies no outcome experiment for (b = 1, l1 = 0, l2 = 1).
    let (code, message) = refusal(prepare(&sequence(1.0, 1.0), &[(1, 0, 1)]));
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("temporal_transport.history_outside_support"), "{message}");
    assert!(message.contains("b=1") && message.contains("l2=1"), "{message}");
}

#[test]
fn a_history_the_target_never_reaches_still_needs_source_support() {
    // The exact evaluator enumerates the whole history lattice, so a complete
    // history with no target mass still reads the source's experiment under it:
    // the report must not exempt it, and the evaluator agrees.
    let mut target = target_scm();
    target.exo_p[0] = 0.0; // the target never has b = 1
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    let skip = [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)];
    let gapped = laws(&source_scm(), &target, &skip);
    let report = history_support(&decision, &gapped);
    assert_eq!(report.outside().len(), 4, "{:?}", report.outside());
    assert!(report.outside().iter().all(|r| r.step == 2 && r.history[0] == Value::f64(1.0)));
    assert!(report.outside().iter().all(|r| r.target_mass == Some(0.0)));
    assert_report_matches_evaluator(&decision, &gapped);
    let (code, message) = refusal(prepare_temporal_sequence(
        decision.clone(),
        gapped,
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("temporal_transport.history_outside_support"), "{message}");
    // Served, the unreached histories are reported as such and the answer is the truth.
    let served = laws(&source_scm(), &target, &[]);
    let prepared =
        prepare_temporal_sequence(decision, served, ExactEvaluationLimits::default(), &ctx())
            .unwrap();
    let unreached = prepared.support().rows.iter().filter(|r| r.status == "unreached").count();
    assert_eq!(unreached, 2 + 4);
    let report = prepared.evaluate(&ctx()).unwrap();
    assert!((report.mean - truth(&target, [1, 0])).abs() < 1e-12);
}

#[test]
fn a_first_action_the_target_never_takes_at_an_initial_state_refuses() {
    // Zero the target cells with (b = 1, l1 = 1, a1 = 1): the initial state stays
    // reached but the first action has no target support there.
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    let full = laws(&source_scm(), &target_scm(), &[]);
    let target = full.laws().iter().find(|l| l.population() == "target").unwrap();
    let mut probabilities = target.probabilities().to_vec();
    for (cell, p) in probabilities.iter_mut().enumerate() {
        // Axes b, l1, a1, l2, a2, y with the last fastest.
        if (cell >> 5) & 1 == 1 && (cell >> 4) & 1 == 1 && (cell >> 3) & 1 == 1 {
            *p = 0.0;
        }
    }
    let total: f64 = probabilities.iter().sum();
    probabilities.iter_mut().for_each(|p| *p /= total);
    let edited = antecedent_expr::ExactDiscreteLaw::try_new(
        "target",
        target.regime(),
        [],
        target.axes().to_vec(),
        probabilities,
        target.snapshot_identity(),
        antecedent_expr::LawTolerance::default(),
    )
    .unwrap();
    let mut all = vec![edited];
    all.extend(full.laws().iter().filter(|l| l.population() != "target").cloned());
    let data = antecedent_expr::ExactTransportData::try_new(all, 4096).unwrap();
    let result =
        prepare_temporal_sequence(decision, data, ExactEvaluationLimits::default(), &ctx());
    let (code, message) = refusal(result);
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("step 1") && message.contains("b=1, l1=1"), "{message}");
}

#[test]
fn the_support_report_is_local_to_each_history_and_step() {
    let prepared = prepare(&sequence(0.0, 1.0), &[]).unwrap();
    let support = prepared.support();
    assert!(support.target_law_used);
    // Four initial states and eight complete histories, every one supported.
    assert_eq!(support.rows.iter().filter(|r| r.step == 1).count(), 4);
    assert_eq!(support.rows.iter().filter(|r| r.step == 2).count(), 8);
    assert!(support.rows.iter().all(|r| r.status == "supported"));
    assert!(support.rows.iter().all(|r| r.target_mass.is_some_and(|m| m > 0.0)));
}

#[test]
fn invalid_horizon_sequence_and_bounds_refuse_before_any_search() {
    let refused = |result: Result<TemporalSequenceDecision, TemporalSequenceError>| match result {
        Err(TemporalSequenceError::Refused(r)) => (r.code, r.detail),
        other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
    };
    let ask = |seq: &[Value]| {
        decide_temporal_transport_sequence(
            &spec(),
            seq,
            "source",
            "target",
            &catalog(),
            BUDGET,
            &ctx(),
        )
    };
    assert_eq!(
        refused(ask(&sequence(1.0, 0.0)[..1])),
        ("invalid_argument", TEMPORAL_INVALID_SEQUENCE)
    );
    assert_eq!(refused(ask(&sequence(1.0, 2.0))), ("invalid_argument", TEMPORAL_INVALID_SEQUENCE));
    let mut three = sequence(1.0, 0.0);
    three.push(Value::f64(1.0));
    assert_eq!(refused(ask(&three)), ("route_not_supported", TEMPORAL_HORIZON_DETAIL));
    // A specification with a horizon of three, or too many actions, never builds.
    let diagram = common::z_scm::diagram(6, &[], &[], &[]);
    let horizon = antecedent_identify::sid::temporal_sequence::TemporalSequenceSpec::try_new(
        3,
        common::temporal_fixture::slots(),
        diagram,
        common::temporal_fixture::coordinates(),
    );
    assert_eq!(horizon.unwrap_err().detail, TEMPORAL_HORIZON_DETAIL);
}

/// Smallest operation limit under which the decision completes.
fn completion_limit() -> usize {
    (1..2_000)
        .find(|operations| {
            let budget = SearchLimits { operations: *operations, depth: 256 };
            let d = decide(&sequence(1.0, 0.0), budget);
            matches!(d.outcome, TemporalOutcome::Identified(_))
        })
        .expect("the decision completes under a modest budget")
}

#[test]
fn one_shared_budget_charges_identification_then_every_history_state() {
    let complete = completion_limit();
    // Identification alone needs fewer operations: growing the 4 initial states
    // and 8 complete histories is charged to the same budget.
    let stopped =
        decide(&sequence(1.0, 0.0), SearchLimits { operations: complete - 1, depth: 256 });
    let TemporalOutcome::Stopped { stop } = stopped.outcome else {
        panic!("{:?}", stopped.outcome)
    };
    assert_eq!(stop, SearchStop::Operations);
    let receipt = stopped.receipt.expect("a stop leaves a receipt");
    assert_eq!(receipt.operations_limit, complete - 1);
    // Identification and the whole first history step finished; only step 2 did not.
    assert_eq!(receipt.explored, ["identification", "history_step_1"]);
    assert_eq!(receipt.unevaluated, ["history_step_2"]);
    assert_eq!(receipt.operations_consumed, Some(complete - 1));
    // Twelve states: the limit that admits identification but not the histories.
    let finished = decide(&sequence(1.0, 0.0), SearchLimits { operations: complete, depth: 256 });
    assert!(finished.receipt.is_none());
    assert_eq!((finished.histories.initial.len(), finished.histories.complete.len()), (4, 8));
    let tight = complete - 12;
    let early = decide(&sequence(1.0, 0.0), SearchLimits { operations: tight, depth: 256 });
    assert!(matches!(early.outcome, TemporalOutcome::Stopped { .. }));
    // The stop is a receipt: preparing it refuses as a budget stop, never as non-identification.
    let (code, message) = refusal(prepare_temporal_sequence(
        stopped_decision(),
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_budget_cancel");
    assert!(
        message.starts_with("temporal_transport.history_budget: search.operations"),
        "{message}"
    );
}

fn stopped_decision() -> TemporalSequenceDecision {
    let complete = completion_limit();
    decide(&sequence(1.0, 0.0), SearchLimits { operations: complete - 1, depth: 256 })
}

#[test]
fn depth_memory_and_cancellation_stop_the_shared_budget_too() {
    let seq = sequence(1.0, 0.0);
    let depth = decide(&seq, SearchLimits { operations: 100_000, depth: 1 });
    assert!(matches!(depth.outcome, TemporalOutcome::Stopped { stop: SearchStop::Depth }));
    let mut small = ExecutionContext::for_tests(1);
    small.memory = antecedent_core::MemoryBudget { hard_limit_bytes: Some(64), ..small.memory };
    let memory = decide_temporal_transport_sequence(
        &spec(),
        &seq,
        "source",
        "target",
        &catalog(),
        BUDGET,
        &small,
    )
    .unwrap();
    assert!(matches!(memory.outcome, TemporalOutcome::Stopped { stop: SearchStop::Memory }));
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let stopped = decide_temporal_transport_sequence(
        &spec(),
        &seq,
        "source",
        "target",
        &catalog(),
        BUDGET,
        &cancelled,
    )
    .unwrap();
    assert!(matches!(stopped.outcome, TemporalOutcome::Stopped { stop: SearchStop::Cancelled }));
    assert_eq!(
        stopped.receipt.unwrap().unevaluated,
        ["identification", "history_step_1", "history_step_2"]
    );
}

#[test]
fn a_structural_obstruction_and_missing_evidence_refuse_with_their_own_reasons() {
    // A mechanism difference at the outcome cannot be transported.
    let shifted = unrolled(
        &[Y],
        &[
            (B, L1),
            (L1, A1),
            (B, L2),
            (A1, L2),
            (L2, A2),
            (A1, A2),
            (B, Y),
            (L1, Y),
            (A1, Y),
            (L2, Y),
            (A2, Y),
        ],
    );
    let decision = decide_temporal_transport_sequence(
        &shifted,
        &sequence(1.0, 0.0),
        "source",
        "target",
        &catalog(),
        BUDGET,
        &ctx(),
    )
    .unwrap();
    let error = prepare_temporal_sequence(
        decision,
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    );
    let (code, message) = refusal(error);
    // An independently verified s-hedge on the unrolled diagram.
    assert_eq!(code, "transport_proven_non_transportable", "{message}");
    assert!(message.starts_with("temporal_transport.checked_obstruction:"), "{message}");
    // Without the source's history experiments the formula cannot bind.
    let no_source = antecedent_core::EvidenceCatalog::try_new(
        [],
        catalog().regimes[..1].to_vec(),
        catalog().bindings[..1].to_vec(),
        None,
    )
    .unwrap();
    let decision = decide_temporal_transport_sequence(
        &spec(),
        &sequence(1.0, 0.0),
        "source",
        "target",
        &no_source,
        BUDGET,
        &ctx(),
    )
    .unwrap();
    let (code, message) = refusal(prepare_temporal_sequence(
        decision,
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.starts_with("temporal_transport.missing_evidence"), "{message}");
}

/// Whether the exact evaluator itself accepts `data` for the decided sequence,
/// independent of the support report.
fn evaluator_accepts(decision: &TemporalSequenceDecision, data: &ExactTransportData) -> bool {
    let TemporalOutcome::Identified(bound) = &decision.outcome else { panic!("not identified") };
    let slots = decision.spec.slots();
    let request = Assignment::from_pairs(
        slots.actions.iter().copied().zip(decision.sequence.iter().cloned()),
    );
    antecedent_estimate::transport::prepare_exact_transport(
        bound,
        data.clone(),
        request,
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .and_then(|plan| plan.evaluate(&ctx()))
    .is_ok()
}

fn assert_report_matches_evaluator(decision: &TemporalSequenceDecision, data: &ExactTransportData) {
    let report = history_support(decision, data);
    assert_eq!(
        report.outside().is_empty(),
        evaluator_accepts(decision, data),
        "the support report and the evaluator must agree: {:?}",
        report.outside().len()
    );
}

#[test]
fn a_source_law_without_interventions_supports_no_history() {
    // An observational source law is not a source experiment under any history:
    // it may not be counted as vacuously consistent with every history.
    let (source, target) = (source_scm(), target_scm());
    let data = data_of(vec![target_law(&target), observational_source_law(&source)]);
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    let report = history_support(&decision, &data);
    let step2 = report.rows.iter().filter(|r| r.step == 2).collect::<Vec<_>>();
    assert_eq!(step2.len(), 8);
    assert!(step2.iter().all(|r| r.status == "outside_support"), "{step2:?}");
    // Step 1 is target-only and stays supported: the failure is history-local.
    assert!(report.rows.iter().filter(|r| r.step == 1).all(|r| r.status == "supported"));
    assert_report_matches_evaluator(&decision, &data);
    let (code, message) = refusal(prepare_temporal_sequence(
        decision,
        data,
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("temporal_transport.history_outside_support"), "{message}");
    assert!(message.contains("8 of 12 histories") && message.contains("step 2 (b="), "{message}");
}

#[test]
fn a_source_law_intervening_only_on_the_actions_supports_no_complete_history() {
    // The proof cites the source's outcome under every complete history; an
    // experiment on {a1, a2} alone does not serve that leaf at any history.
    let (source, target) = (source_scm(), target_scm());
    let data = data_of(vec![target_law(&target), action_law(&source, [1, 0])]);
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    let report = history_support(&decision, &data);
    assert!(report.rows.iter().filter(|r| r.step == 2).all(|r| r.status == "outside_support"));
    assert_report_matches_evaluator(&decision, &data);
    let (code, message) = refusal(prepare_temporal_sequence(
        decision,
        data,
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("temporal_transport.history_outside_support"), "{message}");
}

#[test]
fn the_support_report_never_promises_what_the_evaluator_refuses() {
    let decision = decide(&sequence(1.0, 0.0), BUDGET);
    let (source, target) = (source_scm(), target_scm());
    // Every history served: both agree it is fine.
    let full = laws(&source, &target, &[]);
    assert!(history_support(&decision, &full).outside().is_empty());
    assert_report_matches_evaluator(&decision, &full);
    // Exactly one reached history unserved: one row outside, both agree it is not.
    let one = laws(&source, &target, &[(1, 0, 1)]);
    let report = history_support(&decision, &one);
    assert_eq!(report.outside().len(), 1);
    assert_eq!(report.outside()[0].history, [Value::f64(1.0), Value::f64(0.0), Value::f64(1.0)]);
    assert_report_matches_evaluator(&decision, &one);
    // A source law of the right shape but the wrong regime serves nothing.
    let mut shifted = vec![target_law(&target)];
    for law in full.laws().iter().filter(|l| l.population() == "source") {
        shifted.push(
            antecedent_expr::ExactDiscreteLaw::try_new(
                "source",
                antecedent_core::RegimeId::from_raw(7),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                law.probabilities().to_vec(),
                law.snapshot_identity(),
                antecedent_expr::LawTolerance::default(),
            )
            .unwrap(),
        );
    }
    let wrong = data_of(shifted);
    assert!(!history_support(&decision, &wrong).outside().is_empty());
    assert_report_matches_evaluator(&decision, &wrong);
    // The shapes the auditor found: observational-only and actions-only source laws.
    let observational = data_of(vec![target_law(&target), observational_source_law(&source)]);
    assert_report_matches_evaluator(&decision, &observational);
    let actions = data_of(vec![target_law(&target), action_law(&source, [1, 0])]);
    assert_report_matches_evaluator(&decision, &actions);
}

#[test]
fn a_derivation_citing_other_regimes_is_supported_by_those_regimes() {
    // With the source's experiment on the actions alone and a selection at the
    // baseline, the proof cites P^s(y | b, do(a1, a2)) and P*(b): one law under the
    // requested sequence serves every reached history, and complete-history
    // experiments (which this derivation does not read) serve none.
    let shifted = baseline_shift_source();
    let decision = decide_temporal_transport_sequence(
        &baseline_only_spec(),
        &sequence(1.0, 0.0),
        "source",
        "target",
        &action_catalog(),
        BUDGET,
        &ctx(),
    )
    .unwrap();
    assert!(matches!(decision.outcome, TemporalOutcome::Identified(_)));
    let data = data_of(vec![target_law(&target_scm()), action_law(&shifted, [1, 0])]);
    let report = history_support(&decision, &data);
    assert!(report.outside().is_empty(), "{:?}", report.outside());
    assert_report_matches_evaluator(&decision, &data);
    let point =
        prepare_temporal_sequence(decision.clone(), data, ExactEvaluationLimits::default(), &ctx())
            .unwrap()
            .evaluate(&ctx())
            .unwrap();
    assert!((point.mean - truth(&target_scm(), [1, 0])).abs() < 1e-12);
    // The experiment under the other sequence is not the requested one.
    let wrong = data_of(vec![target_law(&target_scm()), action_law(&shifted, [0, 1])]);
    assert!(!history_support(&decision, &wrong).outside().is_empty());
    assert_report_matches_evaluator(&decision, &wrong);
    // Complete-history experiments serve a different leaf than the one cited.
    let complete = laws(&shifted, &target_scm(), &[]);
    assert!(!history_support(&decision, &complete).outside().is_empty());
    assert_report_matches_evaluator(&decision, &complete);
    // A source under which a reached baseline never occurs cannot condition on it.
    let mut never = baseline_shift_source();
    never.exo_p[0] = 0.0;
    let unconditioned = data_of(vec![target_law(&target_scm()), action_law(&never, [1, 0])]);
    let report = history_support(&decision, &unconditioned);
    assert!(!report.outside().is_empty());
    assert_report_matches_evaluator(&decision, &unconditioned);
}

#[test]
fn evidence_is_cited_only_when_the_derivation_reads_it() {
    let decision = decide_temporal_transport_sequence(
        &spec(),
        &sequence(1.0, 0.0),
        "source",
        "target",
        &catalog_with_unused_regime(),
        BUDGET,
        &ctx(),
    )
    .unwrap();
    let cited = |regime: u32| {
        let rows =
            decision.evidence.iter().filter(|e| e.regime.raw() == regime).collect::<Vec<_>>();
        assert!(!rows.is_empty(), "regime {regime} is in the catalog");
        rows.iter().all(|e| e.cited_by_derivation)
    };
    assert!(cited(0) && cited(1), "the target law and the complete-history experiments are read");
    assert!(
        !cited(2),
        "an observational source regime the proof never reads is not evidence it needs"
    );
    // A derivation of another shape cites what it reads.
    let other = decide_temporal_transport_sequence(
        &baseline_only_spec(),
        &sequence(1.0, 0.0),
        "source",
        "target",
        &action_catalog(),
        BUDGET,
        &ctx(),
    )
    .unwrap();
    assert!(other.evidence.iter().all(|e| e.cited_by_derivation));
}

#[test]
fn operations_stop_inside_each_history_step_with_a_receipt_of_what_finished() {
    let complete = completion_limit();
    // Identification takes `complete - 12` operations; the 4 initial states and 8
    // complete histories are charged after it.
    let identified = complete - 12;
    let at = |operations: usize| {
        let d = decide(&sequence(1.0, 0.0), SearchLimits { operations, depth: 256 });
        let TemporalOutcome::Stopped { stop } = d.outcome else { panic!("{:?}", d.outcome) };
        assert_eq!(stop, SearchStop::Operations);
        (d.receipt.expect("a stop leaves a receipt"), d.histories)
    };
    // Inside identification: nothing finished.
    let (receipt, histories) = at(identified - 1);
    assert!(receipt.explored.is_empty());
    assert!(histories.initial.is_empty() && histories.complete.is_empty());
    // Inside history step 1: identification finished, no history did.
    let (receipt, _) = at(identified);
    assert_eq!(receipt.explored, ["identification"]);
    assert_eq!(receipt.unevaluated, ["history_step_1", "history_step_2"]);
    let (receipt, _) = at(identified + 3);
    assert_eq!(receipt.explored, ["identification"]);
    // Inside history step 2: step 1 finished as well.
    let (receipt, _) = at(identified + 4);
    assert_eq!(receipt.explored, ["identification", "history_step_1"]);
    assert_eq!(receipt.unevaluated, ["history_step_2"]);
    let (receipt, _) = at(complete - 1);
    assert_eq!(receipt.explored, ["identification", "history_step_1"]);
    // Every one of these preparations refuses as a budget stop.
    let (code, message) = refusal(prepare_temporal_sequence(
        decide(&sequence(1.0, 0.0), SearchLimits { operations: identified + 4, depth: 256 }),
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_budget_cancel");
    assert!(
        message.starts_with("temporal_transport.history_budget: search.operations"),
        "{message}"
    );
}

/// The wide lattice decided under a hard memory limit.
fn wide_decision(memory: Option<u64>) -> TemporalSequenceDecision {
    let (spec, catalog) = common::temporal_fixture::wide(5);
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.memory = antecedent_core::MemoryBudget { hard_limit_bytes: memory, ..ctx.memory };
    decide_temporal_transport_sequence(
        &spec,
        &sequence(1.0, 0.0),
        "source",
        "target",
        &catalog,
        SearchLimits { operations: 1_000_000, depth: 256 },
        &ctx,
    )
    .unwrap()
}

#[test]
fn memory_stops_inside_the_history_growth_and_keeps_the_finished_steps() {
    // Five three-level baseline covariates make the lattice (972 initial states,
    // 1944 complete histories) retain more bytes than identification does, so the
    // memory limit can bite while the histories grow.
    let free = wide_decision(None);
    assert_eq!((free.histories.initial.len(), free.histories.complete.len()), (972, 1944));
    assert!(free.receipt.is_none());
    // The smallest limit under which the whole decision fits.
    let (mut low, mut high) = (1u64, 2_000_000u64);
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if wide_decision(Some(mid)).receipt.is_none() {
            high = mid;
        } else {
            low = mid;
        }
    }
    let stopped = wide_decision(Some(high - 1));
    let TemporalOutcome::Stopped { stop } = stopped.outcome else {
        panic!("{:?}", stopped.outcome)
    };
    assert_eq!(stop, SearchStop::Memory);
    let receipt = stopped.receipt.clone().expect("a stop leaves a receipt");
    // Identification and the initial states were retained; the complete histories were not.
    assert_eq!(receipt.explored, ["identification", "history_step_1"]);
    assert_eq!(receipt.unevaluated, ["history_step_2"]);
    assert!(stopped.histories.complete.is_empty());
    let (code, message) = refusal(prepare_temporal_sequence(
        stopped,
        common::temporal_fixture::data_of(vec![]),
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_budget_cancel");
    assert!(message.starts_with("temporal_transport.history_budget: search.memory"), "{message}");
}

#[test]
fn the_depth_limit_binds_identification_because_history_growth_is_charged_at_the_horizon() {
    // History growth charges depth 1 (initial states) and 2 (the horizon); the
    // identification search itself reaches depth 3, so any limit that admits
    // identification admits the history growth. A depth stop is therefore always
    // an identification stop, and its receipt says nothing finished.
    let minimal = (1..10usize)
        .find(|depth| {
            let d =
                decide(&sequence(1.0, 0.0), SearchLimits { operations: 100_000, depth: *depth });
            d.receipt.is_none()
        })
        .expect("a modest depth admits the decision");
    assert!(minimal > 2, "identification needs depth {minimal}, above the history charges");
    let d = decide(&sequence(1.0, 0.0), SearchLimits { operations: 100_000, depth: minimal - 1 });
    let TemporalOutcome::Stopped { stop } = d.outcome else { panic!("{:?}", d.outcome) };
    assert_eq!(stop, SearchStop::Depth);
    let receipt = d.receipt.unwrap();
    assert!(receipt.explored.is_empty());
    assert_eq!(&receipt.unevaluated[..3], ["identification", "history_step_1", "history_step_2"]);
}

#[test]
fn a_lagged_template_unrolled_through_the_adr_0021_unfolding_executes_end_to_end() {
    use common::temporal_fixture::template;
    // The specification comes from `unroll_two_slice` over a lagged `TemporalDag`
    // template (the ADR 0021 unfolding), then identifies, prepares and evaluates
    // against the enumerated truth of a known structural model.
    let (source, target) = (template::source(), template::target());
    let data = template::laws(&source, &target);
    for (a1, a2) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
        let decision = decide_temporal_transport_sequence(
            &template::spec(),
            &sequence(f64::from(a1), f64::from(a2)),
            "source",
            "target",
            &template::catalog(),
            BUDGET,
            &ctx(),
        )
        .unwrap();
        assert!(
            matches!(decision.outcome, TemporalOutcome::Identified(_)),
            "{:?}",
            decision.outcome.status()
        );
        // Time-varying confounding is read off the unrolled template.
        assert_eq!(decision.time_varying_confounders, vec![v(template::L2)]);
        let report = prepare_temporal_sequence(
            decision,
            data.clone(),
            ExactEvaluationLimits::default(),
            &ctx(),
        )
        .unwrap()
        .evaluate(&ctx())
        .unwrap();
        let expected = target.truth([a1, a2]);
        assert!((report.mean - expected).abs() < 1e-12, "{a1}{a2}: {} vs {expected}", report.mean);
        assert!((source.truth([a1, a2]) - expected).abs() > 0.02, "{a1}{a2}");
        assert!(report.support.outside().is_empty());
        // Step 2 histories carry the period-1 outcome as a covariate.
        assert_eq!(report.support.history_coordinates.len(), 4);
    }
}

#[test]
fn a_three_level_action_alphabet_executes_through_the_whole_route() {
    use common::temporal_fixture::categorical;
    let (source, target) = (categorical::source(), categorical::target());
    let data = categorical::laws(&source, &target);
    let mut widest = 0.0f64;
    for (a1, a2) in [(0u8, 1u8), (2, 1), (1, 2), (2, 0)] {
        let decision = decide_temporal_transport_sequence(
            &categorical::spec(),
            &sequence(f64::from(a1), f64::from(a2)),
            "source",
            "target",
            &categorical::evidence(),
            BUDGET,
            &ctx(),
        )
        .unwrap();
        assert!(
            matches!(decision.outcome, TemporalOutcome::Identified(_)),
            "{}",
            decision.outcome.status()
        );
        let prepared = prepare_temporal_sequence(
            decision,
            data.clone(),
            ExactEvaluationLimits::default(),
            &ctx(),
        )
        .unwrap();
        let report = prepared.evaluate(&ctx()).unwrap();
        let expected = target.truth([a1, a2]);
        assert!((report.mean - expected).abs() < 1e-12, "{a1}{a2}: {} vs {expected}", report.mean);
        widest = widest.max((source.truth([a1, a2]) - expected).abs());
        assert!(report.support.outside().is_empty());
    }
    assert!(widest > 0.02, "the source's own answer must differ from the target's: {widest}");
    // An action outside the three-level alphabet is refused before any search.
    let refused = decide_temporal_transport_sequence(
        &categorical::spec(),
        &sequence(3.0, 0.0),
        "source",
        "target",
        &categorical::evidence(),
        BUDGET,
        &ctx(),
    );
    assert!(
        matches!(refused, Err(TemporalSequenceError::Refused(r)) if r.detail == TEMPORAL_INVALID_SEQUENCE)
    );
}

#[test]
fn a_decision_the_rules_could_not_certify_refuses_as_search_incomplete_never_as_an_obstruction() {
    // The licensed rules are sound and incomplete: a diagram no rule certifies and
    // no s-hedge refutes is reported as such. No diagram of this fixture reaches
    // that outcome (the rules certify or find a hedge for every unrolling tried),
    // so the refusal surface is exercised on a decision whose outcome is set to it.
    let mut decision = decide(&sequence(1.0, 0.0), BUDGET);
    decision.outcome = TemporalOutcome::NotCertified {
        obligations: std::sync::Arc::from([std::sync::Arc::<str>::from("no rule applied")]),
    };
    let (code, message) = refusal(prepare_temporal_sequence(
        decision,
        laws(&source_scm(), &target_scm(), &[]),
        ExactEvaluationLimits::default(),
        &ctx(),
    ));
    assert_eq!(code, "transport_not_certified");
    assert!(
        message.starts_with("temporal_transport.search_incomplete: no rule applied"),
        "{message}"
    );
    assert!(!message.contains("checked_obstruction"), "{message}");
}
