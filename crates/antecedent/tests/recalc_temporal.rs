//! Checked whole-history recalculation against an independently enumerated longitudinal SCM.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_receipt::RecalcRunError;
use antecedent::analysis::recalc_temporal::{
    TemporalFunctional, TemporalHistoryWire, TemporalOutcome, TemporalRequest, TemporalRunError,
    TemporalSession, TemporalUnitWire, consume_temporal_recalc_artifact,
    execute_temporal_with_receipt,
};
use antecedent_core::{ExecutionContext, recalc::ResumeContext};
use antecedent_expr::execution_counts::count_static_work;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(731)
}
fn pin(group: &str, key: &str) -> f64 {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/recalculation/temporal_history/expected.json"
    ))
    .unwrap();
    oracle[group][key].as_f64().unwrap()
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} versus {expected}");
}
// Exogenous UL is uniform on 0..4, UY uniform on 0..20. Each unit supplies
// two identical histories: treating those histories as independent is incorrect.
fn request(functional: TemporalFunctional) -> TemporalRequest {
    let mut units = Vec::new();
    for s0 in 0..2 {
        for a1 in 0..2 {
            for a2 in 0..2 {
                for ul in 0..4 {
                    for uy in 0..20 {
                        let l2 = u32::from(ul < s0 + a1 + 1);
                        let y = f64::from(uy < 2 + 4 * a1 + 3 * l2 + 5 * a2 + 2 * s0 + 2 * s0 * a1);
                        let history = TemporalHistoryWire { time_id: 0, s0, a1, l2, a2, y };
                        units.push(TemporalUnitWire {
                            unit_id: u64::try_from(units.len()).unwrap(),
                            histories: vec![
                                history.clone(),
                                TemporalHistoryWire { time_id: 2, ..history },
                            ],
                        });
                    }
                }
            }
        }
    }
    TemporalRequest {
        edges: vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        bidirected: vec![],
        selection_targets: vec![0],
        horizon: 2,
        lag_alignment: [0, 1, 2, 2, 2],
        period: (0, 5),
        units,
        snapshot_id: "enumerated-dependent-source".into(),
        initial_state: [0.2, 0.8],
        initial_state_id: "given-target-law".into(),
        functional,
        benefit_per_unit: 2.,
        cost: 0.5,
    }
}
fn effect() -> TemporalRequest {
    request(TemporalFunctional::Effect { active: [1, 1], control: [0, 0] })
}
// Integrate the structural conditional probabilities, without invoking either engine.
fn mean(a1: f64, a2: f64, p1: f64) -> f64 {
    0.1375 + 0.2375 * a1 + 0.25 * a2 + 0.1375 * p1 + 0.1 * p1 * a1
}
fn run(session: &mut TemporalSession, request: &TemporalRequest) -> TemporalOutcome {
    let ((result, measured), checks) =
        antecedent_identify::execution_counts::count_checked_identifications(|| {
            count_static_work(|| execute_temporal_with_receipt(session, request, &ctx()))
        });
    let result = result.unwrap();
    let totals = result.recalc.receipt.totals();
    assert_eq!(totals.identifications, checks);
    assert_eq!(totals.factor_builds, measured.factor_builds);
    assert_eq!(totals.program_compilations, measured.program_compilations);
    assert_eq!(totals.provider_bindings, measured.provider_bindings);
    assert_eq!(totals.factor_evaluations, measured.factor_evaluations);
    assert_eq!(totals.integrations, measured.integrations);
    assert_eq!(totals.provider_calls, measured.provider_calls);
    assert_eq!(totals.law_summaries, measured.law_summaries);
    assert_eq!(totals.model_fits, 0);
    assert_eq!(totals.reweights, 0);
    assert!(result.recalc.law.std_error.is_nan());
    result
}

#[test]
fn temporal_history_response_and_effect_match_enumerated_longitudinal_truth() {
    for sequence in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        let r = request(TemporalFunctional::Response { sequence });
        let result = run(&mut TemporalSession::new(), &r);
        close(result.means[0], mean(f64::from(sequence[0]), f64::from(sequence[1]), 0.8));
        close(
            result.means[0],
            pin("target_p1_0_8", &format!("response_{}{}", sequence[0], sequence[1])),
        );
        close(result.recalc.law.ate, result.means[0]);
        assert_eq!(result.recalc.receipt.totals().factor_builds, 2);
        assert!(result.recalc.receipt.totals().integrations > 0);
    }
    let result = run(&mut TemporalSession::new(), &effect());
    close(result.means[0], pin("target_p1_0_8", "response_11"));
    close(result.means[1], pin("target_p1_0_8", "response_00"));
    close(result.recalc.law.ate, pin("target_p1_0_8", "effect_11_vs_00"));
    assert_eq!(result.recalc.receipt.totals().factor_builds, 3);
}

#[test]
fn temporal_history_unchanged_utility_and_initial_law_reuse_actual_source_fits() {
    let r = effect();
    let mut session = TemporalSession::new();
    run(&mut session, &r);
    let source = std::ptr::from_ref(session.source_factor().unwrap());
    let active = std::ptr::from_ref(session.sequence_fit([1, 1]).unwrap());
    let unchanged = run(&mut session, &r);
    assert_eq!(unchanged.recalc.receipt.totals().total(), 0);
    let mut utility = r.clone();
    utility.cost = 1.;
    let changed = run(&mut session, &utility);
    close(changed.recalc.decision.net_benefit, 2. * 0.5675 - 1.);
    assert_eq!(changed.recalc.receipt.totals().factor_builds, 0);
    assert_eq!(changed.recalc.receipt.totals().integrations, 0);
    let mut target = utility;
    target.initial_state = [0.75, 0.25];
    target.initial_state_id = "new-given-target-law".into();
    let updated = run(&mut session, &target);
    close(updated.means[0], pin("target_p1_0_25", "response_11"));
    close(updated.means[1], pin("target_p1_0_25", "response_00"));
    close(updated.recalc.law.ate, pin("target_p1_0_25", "effect_11_vs_00"));
    assert_eq!(updated.recalc.receipt.totals().identifications, 0);
    assert_eq!(updated.recalc.receipt.totals().factor_builds, 0);
    assert!(updated.recalc.receipt.totals().integrations > 0);
    assert_eq!(source, std::ptr::from_ref(session.source_factor().unwrap()));
    assert_eq!(active, std::ptr::from_ref(session.sequence_fit([1, 1]).unwrap()));
}

#[test]
fn temporal_history_new_sequence_and_new_raw_histories_invalidate_only_real_work() {
    let mut r = effect();
    let mut session = TemporalSession::new();
    run(&mut session, &r);
    let source = std::ptr::from_ref(session.source_factor().unwrap());
    r.functional = TemporalFunctional::Effect { active: [1, 0], control: [0, 1] };
    let changed = run(&mut session, &r);
    close(changed.recalc.law.ate, 0.0675);
    assert_eq!(changed.recalc.receipt.totals().factor_builds, 2);
    assert_eq!(source, std::ptr::from_ref(session.source_factor().unwrap()));
    r = effect();
    for history in &mut r.units[0].histories {
        history.y = 0.;
    }
    let updated = run(&mut session, &r);
    close(updated.recalc.law.ate, 0.57);
    assert_eq!(updated.recalc.receipt.totals().factor_builds, 3);
    let fresh = run(&mut TemporalSession::new(), &r);
    close(updated.recalc.law.ate, fresh.recalc.law.ate);
    for unit in &mut r.units {
        unit.histories.push(TemporalHistoryWire { time_id: 4, ..unit.histories[0].clone() });
    }
    r.period = (0, 7);
    let appended = run(&mut session, &r);
    assert_eq!(appended.recalc.receipt.totals().factor_builds, 3);
    assert_eq!(appended.recalc.receipt.totals().identifications, 0);
    close(appended.recalc.law.ate, 0.57);
}

#[test]
fn temporal_history_portable_flags_cannot_supply_fits_and_raw_resume_rebuilds() {
    let r = effect();
    let mut producer = TemporalSession::new();
    let expected = run(&mut producer, &r);
    let mut absent = TemporalSession::resume(
        producer.identities().clone(),
        ResumeContext {
            portable_fit: true,
            portable_scores: true,
            scores_snapshot_bound: true,
            supplied_provider: true,
            supplied_data: false,
        },
    );
    let (result, measured) =
        count_static_work(|| execute_temporal_with_receipt(&mut absent, &r, &ctx()));
    assert!(matches!(result, Err(TemporalRunError::Recalc(RecalcRunError::Refused(_)))));
    assert_eq!(measured.factor_builds, 0);
    let mut supplied = TemporalSession::resume(
        producer.identities().clone(),
        ResumeContext { supplied_data: true, ..ResumeContext::default() },
    );
    let rebuilt = run(&mut supplied, &r);
    assert_eq!(rebuilt.recalc.receipt.totals().factor_builds, 3);
    close(expected.recalc.law.ate, rebuilt.recalc.law.ate);
}

#[test]
fn temporal_history_empirical_artifact_is_independently_checked_from_raw_units() {
    let mut producer = TemporalSession::new();
    let r = effect();
    let expected = run(&mut producer, &r);
    let bytes = producer.export_result().unwrap();
    let (replayed, measured) =
        count_static_work(|| consume_temporal_recalc_artifact(&bytes, &ctx()));
    let (_, result) = replayed.unwrap();
    close(result.recalc.law.ate, expected.recalc.law.ate);
    assert_eq!(measured.factor_builds, 3);
    assert_eq!(result.recalc.receipt.totals().factor_builds, measured.factor_builds);
    assert!(consume_temporal_recalc_artifact(&bytes[..bytes.len() / 2], &ctx()).is_err());
    let (wrong_seed, measured) = count_static_work(|| {
        consume_temporal_recalc_artifact(&bytes, &ExecutionContext::for_tests(732))
    });
    assert!(wrong_seed.is_err());
    assert_eq!(measured.factor_builds, 0);
}

#[test]
fn temporal_history_resource_window_and_support_refusals_preserve_previous_fit() {
    let r = effect();
    let mut session = TemporalSession::new();
    run(&mut session, &r);
    let source = std::ptr::from_ref(session.source_factor().unwrap());
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let (result, work) =
        count_static_work(|| execute_temporal_with_receipt(&mut session, &r, &cancelled));
    assert!(matches!(
        result,
        Err(TemporalRunError::Recalc(RecalcRunError::Request("recalc.cancelled")))
    ));
    assert_eq!(work.factor_builds, 0);
    let mut changed = r.clone();
    changed.units[0].histories[0].y = 0.;
    let mut small = ctx();
    small.memory.hard_limit_bytes = Some(1);
    let (result, work) =
        count_static_work(|| execute_temporal_with_receipt(&mut session, &changed, &small));
    assert!(result.is_err());
    assert_eq!(work.factor_builds, 0);
    let mut horizon = r.clone();
    horizon.horizon = 3;
    let mut lag = r.clone();
    lag.lag_alignment[2] = 1;
    let mut incomplete_window = r.clone();
    incomplete_window.period.1 = 4;
    let mut missing_units = r.clone();
    missing_units.units.clear();
    let mut unsupported = r.clone();
    unsupported.units[0].histories[0].y = 0.5;
    let mut missing_state = r.clone();
    missing_state.units.retain(|u| u.histories[0].s0 == 0);
    let mut wrong_selection = r.clone();
    wrong_selection.selection_targets = vec![2];
    for invalid in [
        horizon,
        lag,
        incomplete_window,
        missing_units,
        unsupported,
        missing_state,
        wrong_selection,
    ] {
        assert!(execute_temporal_with_receipt(&mut session, &invalid, &ctx()).is_err());
        assert_eq!(source, std::ptr::from_ref(session.source_factor().unwrap()));
        assert_eq!(run(&mut session, &r).recalc.receipt.totals().total(), 0);
    }
}

#[test]
fn temporal_history_artifact_rechecks_resealed_data_and_binds_unit_ownership() {
    use antecedent::analysis::recalc_temporal::TemporalRecalcArtifactWire;
    let mut session = TemporalSession::new();
    run(&mut session, &effect());
    let wire = TemporalRecalcArtifactWire::decode(&session.export_result().unwrap()).unwrap();
    let mut ownership = wire.clone();
    ownership.request.units[0].unit_id = 10000;
    assert!(ownership.export().is_err(), "unit ownership is bound by the data digest");
    let mut changed = wire.request.clone();
    for history in &mut changed.units[0].histories {
        history.y = 0.;
    }
    let forged =
        TemporalRecalcArtifactWire::seal(changed, wire.seed, wire.reports.clone(), wire.value)
            .unwrap()
            .export()
            .unwrap();
    let (result, work) = count_static_work(|| consume_temporal_recalc_artifact(&forged, &ctx()));
    assert!(matches!(
        result,
        Err(TemporalRunError::Recalc(RecalcRunError::Request("recalc.temporal_artifact_mismatch")))
    ));
    assert_eq!(
        work.factor_builds, 3,
        "a resealed envelope must still execute the checked source consumer"
    );
    let mut graph = wire.request.clone();
    graph.edges.retain(|edge| *edge != (0, 4));
    let graph_forgery =
        TemporalRecalcArtifactWire::seal(graph, wire.seed, wire.reports.clone(), wire.value)
            .unwrap()
            .export()
            .unwrap();
    assert!(consume_temporal_recalc_artifact(&graph_forgery, &ctx()).is_err());
    let mut wrong_program = wire.clone();
    wrong_program.reports[0].source_proof.executable_root = 0;
    assert!(consume_temporal_recalc_artifact(&wrong_program.export().unwrap(), &ctx()).is_err());
    let mut wrong_trace = wire.clone();
    wrong_trace.reports[0].source_proof.derivation.clear();
    assert!(consume_temporal_recalc_artifact(&wrong_trace.export().unwrap(), &ctx()).is_err());
}

#[test]
fn temporal_history_source_program_uses_only_reachable_actual_observational_source_leaves() {
    use antecedent::analysis::recalc_temporal::TemporalRecalcArtifactWire;
    use antecedent_expr::{DomainRef, ExprId, ExprNode};
    let mut session = TemporalSession::new();
    run(&mut session, &effect());
    let wire = TemporalRecalcArtifactWire::decode(&session.export_result().unwrap()).unwrap();
    for report in wire.reports {
        let proof = report.source_proof;
        assert_eq!(proof.method, "general_id");
        assert_eq!(proof.population, "source");
        let arena =
            antecedent_io::expr_wire::expr_arena_from_wire(&proof.executable_expression).unwrap();
        let mut pending = vec![ExprId::from_raw(proof.executable_root)];
        let mut visited = std::collections::BTreeSet::new();
        let mut leaves = 0;
        while let Some(id) = pending.pop() {
            if !visited.insert(id.raw()) {
                continue;
            }
            match arena.node(id) {
                ExprNode::Distribution { domain, population, regime, intervention, .. } => {
                    leaves += 1;
                    assert_eq!(*domain, DomainRef::Observational);
                    assert_eq!(arena.population(*population), "source");
                    assert_eq!(*regime, Some(antecedent_core::RegimeId::from_raw(0)));
                    assert!(arena.intervention_assignments(*intervention).is_empty());
                }
                ExprNode::Kernel { body, .. } => pending.push(*body),
                ExprNode::Product(list) => pending.extend(arena.list(*list)),
                ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                    pending.push(*expr)
                }
                ExprNode::Ratio { numerator, denominator } => {
                    pending.extend([*numerator, *denominator])
                }
                _ => panic!("source proof must execute a joint distribution"),
            }
        }
        assert!(leaves > 0);
        assert!(
            visited.len() < arena.len(),
            "original unbound nodes remain inspectable but cannot execute"
        );
    }
}

#[test]
fn temporal_history_uncertainty_rejects_unmeasured_scope_and_checks_graph() {
    use antecedent_estimate::temporal_dependent_interval::DependentIntervalConfig;
    let r = effect();
    let mut session = TemporalSession::new();
    run(&mut session, &r);
    let error = session
        .dependent_interval(&DependentIntervalConfig {
            replicates: 20,
            ..DependentIntervalConfig::default()
        })
        .unwrap_err();
    assert!(error.to_string().contains("only the measured complete binary-history"));
    let mut confounded = r.clone();
    confounded.bidirected.push((1, 4));
    let (result, work) =
        count_static_work(|| execute_temporal_with_receipt(&mut session, &confounded, &ctx()));
    assert!(
        result.is_err(),
        "the actual checked whole-sequence proof must reject latent action/outcome confounding"
    );
    assert_eq!(
        work.factor_builds, 0,
        "identification refusal precedes fitting empirical history laws"
    );
    assert_eq!(run(&mut session, &r).recalc.receipt.totals().total(), 0);
}

#[cfg(feature = "calibration-internal")]
#[test]
fn temporal_history_closed_candidate_replays_whole_unit_identities_without_coverage_claim() {
    use antecedent_estimate::temporal_dependent_interval::{
        DependentIntervalConfig, IntervalMethod,
    };
    let r = effect();
    let mut session = TemporalSession::new();
    run(&mut session, &r);
    let config = DependentIntervalConfig {
        method: IntervalMethod::Percentile,
        replicates: 20,
        seed: 901,
        ..DependentIntervalConfig::default()
    };
    let (result, work) = count_static_work(|| session.candidate_interval_internal(&config, &ctx()));
    let result = result.unwrap();
    close(result.point, 0.5675);
    assert_eq!(result.units, 640);
    assert_eq!(result.calibration, "unmeasured");
    assert_eq!(result.replicates.len(), 20);
    assert_eq!(
        work.factor_builds, 63,
        "one actual source joint and two sequence mechanisms for the point and each paired unit draw"
    );
    let replay = session.candidate_interval_internal(&config, &ctx()).unwrap();
    assert_eq!(result, replay);
    let mut relabeled = r.clone();
    relabeled.units[0].unit_id = 10000;
    let mut other = TemporalSession::new();
    run(&mut other, &relabeled);
    let changed = other.candidate_interval_internal(&config, &ctx()).unwrap();
    close(changed.point, result.point);
    assert_ne!(changed.panel_digest, result.panel_digest);
    assert_ne!(changed.replicate_digest(), result.replicate_digest());
    assert!(session.dependent_interval(&config).is_err());
}

#[cfg(feature = "calibration-internal")]
#[test]
fn checked_response_retired_intervals_refuse_before_resampling() {
    use antecedent_estimate::temporal_dependent_interval::{
        DependentIntervalConfig, IntervalMethod,
    };
    let mut session = TemporalSession::new();
    run(&mut session, &request(TemporalFunctional::Response { sequence: [0, 0] }));
    for method in [IntervalMethod::Percentile, IntervalMethod::Basic] {
        let config = DependentIntervalConfig {
            method,
            replicates: 20,
            seed: 901,
            ..DependentIntervalConfig::default()
        };
        let (result, work) =
            count_static_work(|| session.candidate_interval_internal(&config, &ctx()));
        assert!(
            result.unwrap_err().to_string().contains("temporal_interval.retired_response_method")
        );
        assert_eq!(work.factor_builds, 0);
        assert_eq!(work.factor_evaluations, 0);
    }
}
