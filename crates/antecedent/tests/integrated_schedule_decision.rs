//! Independent point oracle, portable replay, and final optimum set coverage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::analysis::composition::{
    BundleLimits, BundleStage, CompositionBundle, NodeStatus, SuppliedSources, standard_consumer,
};
use antecedent::design::{
    PreparedIntegratedScheduleDecision, ScheduleBundleError, ScheduleCalibrationDesign,
    ScheduleDecisionDeclaration, ScheduleDecisionError, ScheduleSourceFit, ScheduleSourceSubject,
    ScheduleStructure, ScheduleTargetCohort, compile_integrated_schedule_decision,
    compile_schedule_decision_bundle, consume_schedule_decision_bundle,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    map_replicates, n_sim, uniform,
};

const OUTCOME: [[f64; 4]; 2] = [[0.125, 0.25, 0.5, 0.875], [0.25, 0.375, 0.625, 1.0]];
const EARLY: [f64; 2] = [0.5, 0.75];
const LATE: [[f64; 2]; 2] = [[0.25, 0.75], [0.5, 0.25]];
const RETAIN: [[f64; 4]; 2] = [[0.875, 0.75, 0.75, 0.875], [0.75, 0.875, 0.875, 0.75]];
const TARGET_PREVALENCE: f64 = 0.75;

fn declaration(tie: bool) -> ScheduleDecisionDeclaration {
    let extra =
        if tie { [0.0, 0.1875, 0.460_937_5, 0.460_937_5] } else { [0.0, 0.125, 0.0625, 0.25] };
    ScheduleDecisionDeclaration {
        schedules: vec![0, 1, 2, 3],
        structures: vec![ScheduleStructure::BothActionsCausal, ScheduleStructure::FirstActionOnly],
        preference_costs: vec![[0.0, 0.0625, 0.125, 0.1875], extra],
        calibration_design: None,
    }
}

fn source_subject(baseline: usize, code: usize, endpoint: Option<f64>) -> ScheduleSourceSubject {
    ScheduleSourceSubject { baseline: baseline == 1, actions: [code >= 2, code % 2 == 1], endpoint }
}

fn exact_source() -> Vec<ScheduleSourceSubject> {
    let mut source = Vec::new();
    for x in 0..2 {
        for code in 0..4 {
            let first = if code >= 2 { EARLY[x] } else { 1.0 - EARLY[x] };
            let late = if code % 2 == 1 { LATE[x][code / 2] } else { 1.0 - LATE[x][code / 2] };
            let assigned = (8192.0 * first * late) as usize;
            let uncensored = (assigned as f64 * RETAIN[x][code]) as usize;
            source
                .extend((0..assigned).map(|i| {
                    source_subject(x, code, (i < uncensored).then_some(OUTCOME[x][code]))
                }));
        }
    }
    source
}

fn true_utilities(structure: ScheduleStructure, costs: [f64; 4]) -> [f64; 4] {
    let [zero, one] = OUTCOME;
    let values = match structure {
        ScheduleStructure::BothActionsCausal => [
            zero[0] / 4.0 + 3.0 * one[0] / 4.0,
            zero[1] / 4.0 + 3.0 * one[1] / 4.0,
            zero[2] / 4.0 + 3.0 * one[2] / 4.0,
            zero[3] / 4.0 + 3.0 * one[3] / 4.0,
        ],
        ScheduleStructure::FirstActionOnly => [0.273_437_5, 0.273_437_5, 0.734_375, 0.734_375],
    };
    std::array::from_fn(|code| values[code] - costs[code])
}

fn true_optima(structure: ScheduleStructure, costs: [f64; 4]) -> Vec<u8> {
    let utilities = true_utilities(structure, costs);
    let best = utilities.into_iter().fold(f64::NEG_INFINITY, f64::max);
    (0_u8..4).filter(|&i| utilities[i as usize] == best).collect()
}

#[test]
fn point_oracle_replays_and_changed_parent_invalidates_terminal() {
    let ctx = ExecutionContext::for_tests(0);
    let source = exact_source();
    let target: Vec<bool> = (0..1024).map(|i| i % 4 != 0).collect();
    let problem = declaration(true);
    let compiled =
        compile_integrated_schedule_decision(source.clone(), target, problem.clone()).unwrap();
    assert_eq!(compiled.plan_view().route().as_str(), "IntegratedScheduleDecision");
    assert!(compiled.plan_view().unit_support().is_some());
    let program_identity = *compiled.lowered().identity().as_bytes();
    let built = compiled.execute(&ctx).unwrap();
    assert_eq!(built.program_identity, program_identity);
    assert_eq!(built.decision.first_assignment, EARLY);
    assert_eq!(built.decision.second_assignment, LATE);
    assert_eq!(built.decision.observation_survival, RETAIN);
    assert_eq!(built.decision.target_prevalence, TARGET_PREVALENCE);
    assert_eq!(built.decision.minimum_coverage, REPORTED_LEVEL);
    assert_eq!(built.decision.optimum_set, [0, 2, 3]);
    for world in &built.decision.scenarios {
        let costs = problem.preference_costs[world.preference_state];
        let expected = true_utilities(world.structure, costs);
        assert!(world.point_utilities.iter().all(|&(c, u)| u == expected[c as usize]));
        assert!(
            true_optima(world.structure, costs).iter().all(|c| world.compatible_optima.contains(c))
        );
    }
    assert!(built.decision.coverage_scope.contains("entire optimum set"));
    let limits = BundleLimits::default();
    let bytes = built.bundle.to_bytes("fresh-schedule-decision", &limits).unwrap();
    let consumer = standard_consumer();
    let bundle_replay = consumer
        .consume(&bytes, &limits, built.bundle.identity(), &SuppliedSources::default())
        .unwrap();
    bundle_replay.require_verified().unwrap();
    for (role, program) in
        ["source", "target", "declaration"].into_iter().zip(built.child_program_identities)
    {
        let facts = &bundle_replay.node(role).unwrap().facts;
        assert_eq!(facts.get("schedule_child_program"), Some(&antecedent_core::to_hex(&program)));
    }
    assert_eq!(
        consume_schedule_decision_bundle(&bytes, built.bundle.identity()).unwrap(),
        built.decision
    );

    let changed =
        compile_schedule_decision_bundle(source.clone(), vec![false; 1024], problem.clone(), &ctx)
            .unwrap();
    assert_ne!(changed.bundle.identity(), built.bundle.identity());
    assert_eq!(changed.child_program_identities[0], built.child_program_identities[0]);
    assert_ne!(changed.child_program_identities[1], built.child_program_identities[1]);
    assert_eq!(changed.child_program_identities[2], built.child_program_identities[2]);
    assert_ne!(
        changed.decision.scenarios[0].point_utilities,
        built.decision.scenarios[0].point_utilities
    );
    let mut nodes = built.bundle.nodes().to_vec();
    *nodes.iter_mut().find(|n| n.id == "target").unwrap() =
        changed.bundle.nodes().iter().find(|n| n.id == "target").unwrap().clone();
    let edges: Vec<_> =
        built.bundle.edges().iter().map(|e| (e.from.clone(), e.to.clone())).collect();
    let substituted = CompositionBundle::new(nodes, &edges, &limits).unwrap();
    let reencoded = substituted.to_bytes("re-sealed-with-old-terminal", &limits).unwrap();
    let checked = consumer
        .consume(&reencoded, &limits, substituted.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(matches!(
        &checked.node("decision").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
    ));
    assert!(consume_schedule_decision_bundle(&reencoded, built.bundle.identity()).is_err());
    assert!(matches!(
        consume_schedule_decision_bundle(&reencoded, substituted.identity()),
        Err(ScheduleBundleError::ChangedProgram)
    ));

    let removed: Vec<_> = edges.into_iter().filter(|(from, _)| from != "target").collect();
    let incomplete =
        CompositionBundle::new(built.bundle.nodes().to_vec(), &removed, &limits).unwrap();
    let data = incomplete.to_bytes("missing-dependency", &limits).unwrap();
    let error = consume_schedule_decision_bundle(&data, incomplete.identity()).unwrap_err();
    assert!(error.to_string().contains("target baseline sample"));

    let mut changed_preferences = problem;
    changed_preferences.preference_costs[1][0] = 0.25;
    let reweighted = compile_schedule_decision_bundle(
        source,
        (0..1024).map(|i| i % 4 != 0).collect(),
        changed_preferences,
        &ctx,
    )
    .unwrap();
    assert_eq!(reweighted.child_program_identities[..2], built.child_program_identities[..2]);
    assert_ne!(reweighted.child_program_identities[2], built.child_program_identities[2]);
    assert_ne!(reweighted.program_identity, built.program_identity);
    assert_ne!(
        reweighted.decision.scenarios[1].point_utilities,
        built.decision.scenarios[1].point_utilities
    );
}

#[test]
fn missing_structure_is_actionable() {
    let mut problem = declaration(false);
    problem.structures.pop();
    let err = PreparedIntegratedScheduleDecision::compile(
        ScheduleSourceFit::fit(&exact_source()).unwrap(),
        ScheduleTargetCohort::fit(&[true; 20]).unwrap(),
        problem,
    )
    .unwrap_err();
    assert_eq!(err, ScheduleDecisionError::MissingJointEvidence);
}

#[test]
fn insufficient_scope_and_budget_are_refused() {
    let source = exact_source();
    let declaration = declaration(false);
    let target = vec![true; 32];
    let limited = ExecutionContext::for_tests(3).with_admission(AdmissionPool::fixed(1, 1));
    assert!(matches!(
        compile_schedule_decision_bundle(source.clone(), target, declaration.clone(), &limited),
        Err(ScheduleBundleError::Resource(_))
    ));
    assert!(matches!(
        compile_schedule_decision_bundle(
            source,
            vec![true; 19],
            declaration,
            &ExecutionContext::for_tests(3)
        ),
        Err(ScheduleBundleError::Procedure(ScheduleDecisionError::InadequateOverlap))
    ));
}

fn coverage(tied: bool, name: &'static str) {
    let mut problem = declaration(tied);
    problem.calibration_design = Some(if tied {
        ScheduleCalibrationDesign::ExampleTiedOptima
    } else {
        ScheduleCalibrationDesign::ExampleUniqueOptima
    });
    let target_optima: Vec<_> = problem
        .structures
        .iter()
        .flat_map(|&structure| {
            problem.preference_costs.iter().flat_map(move |&costs| true_optima(structure, costs))
        })
        .collect();
    let subjects = grid_n(4096);
    let target_subjects = subjects / 3;
    let covered = map_replicates(n_sim(), |seed| {
        let mut source_draw = uniform(seed ^ 0x21af_0063, 411);
        let mut target_draw = uniform(seed ^ 0x91b2_8f11, 412);
        let source: Vec<_> = (0..subjects)
            .map(|_| {
                let x = usize::from(source_draw() < 0.375);
                let first = usize::from(source_draw() < EARLY[x]);
                let second = usize::from(source_draw() < LATE[x][first]);
                let code = first * 2 + second;
                let endpoint = f64::from(u8::from(source_draw() < OUTCOME[x][code]));
                source_subject(x, code, (source_draw() < RETAIN[x][code]).then_some(endpoint))
            })
            .collect();
        let target: Vec<_> =
            (0..target_subjects).map(|_| target_draw() < TARGET_PREVALENCE).collect();
        let decision = compile_schedule_decision_bundle(
            source,
            target,
            problem.clone(),
            &ExecutionContext::for_tests(seed),
        )
        .unwrap()
        .decision;
        assert_eq!(decision.minimum_coverage, REPORTED_LEVEL);
        let every_world = decision.scenarios.iter().all(|row| {
            true_optima(row.structure, problem.preference_costs[row.preference_state])
                .iter()
                .all(|code| row.compatible_optima.contains(code))
        });
        let final_set = target_optima.iter().all(|code| decision.optimum_set.contains(code));
        (every_world && final_set, decision.optimum_set.len(), decision.calibration_basis.unwrap())
    });
    let dgp = if tied {
        "independent_source_and_target_two_graphs_joint_preferences_all_optima_tie"
    } else {
        "independent_source_and_target_two_graphs_joint_preferences_unique_optima"
    };
    let mut tally = CoverageTally::for_record(
        RecordKey { test: name, dgp, interval: "hoeffding_optimum_set" },
        REPORTED_LEVEL,
    );
    let mut set_size = 0;
    for (hit, size, basis) in covered {
        let key = basis.key;
        assert_eq!(
            key.functional,
            if tied {
                "integrated_schedule_example_all_tied_optima"
            } else {
                "integrated_schedule_example_unique_optima"
            }
        );
        tally.bind_validation(
            &Construction {
                query: key.query,
                graph_class: key.graph_class,
                structure: key.structure,
                modality: key.modality,
                inference: key.inference,
                estimator: key.estimator,
                interval_method: key.interval_method,
                se_kind: key.se_kind,
                dependence: key.dependence,
                posterior: key.posterior,
                functional: key.functional,
                identification: key.identification,
                reported_level: key.level,
            },
            ScopeFacts {
                row_count: basis.scope.row_count,
                replicates_ok: basis.scope.replicates_ok,
                posterior_draws: basis.scope.posterior_draws,
                unidentified_mass: basis.scope.unidentified_mass,
            },
            &key.validation,
        );
        tally.record(Some((0.0, 0.0)), if hit { 0.0 } else { 1.0 });
        set_size += size;
    }
    eprintln!("{name}: mean optimum set cardinality {:.3}", set_size as f64 / f64::from(n_sim()));
    tally.assert_one_sided(
        CoverageSide::Upper,
        "zero omitted true optima in the final joint decision and its declared worlds",
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn integrated_schedule_unique_optima_coverage() {
    coverage(false, "integrated_schedule_unique_optima_coverage");
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn integrated_schedule_all_tied_optima_coverage() {
    coverage(true, "integrated_schedule_all_tied_optima_coverage");
}
