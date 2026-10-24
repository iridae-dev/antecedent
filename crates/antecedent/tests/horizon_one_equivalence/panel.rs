//! Joint intervention contrasts and programs at horizon one.
//!
//! The static joint contrast is a second difference of cell means over static decision worlds. A
//! panel program of one step with no lag compiles to exactly those worlds, so the contrast is the
//! same scientific question; the same program with a lag order is a temporal frame whose horizon
//! is one, a different target by construction. The numerical engines differ (cell scores against
//! a fitted dynamic mechanism) and so do their licences, which is pinned.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent::analysis::joint_panel::{
    joint_panel_target, JointPanelColumns, JointPanelContinuousSearchRequest, JointPanelFunctional,
    JointPanelRequest, JointPanelResponseFunctional, JointPanelScientificScope,
    JointPanelSearchUncertainty, JointPanelWorld,
};
use antecedent::{BatchStudy, CellFamilyContrast};
use antecedent_core::{
    InferenceSelection, Intervention, ScientificWorldId, TargetForm, Value, ValueExpr,
};
use antecedent_data::{PanelData, PanelUnit};
use antecedent_design::decision_contract::{
    ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
};
use antecedent_design::program_search::{
    continuous::ContinuousSearchOptions, FiniteProgramDomain, ProgramControl,
};
use antecedent_estimate::joint_panel::{
    PanelChannelHistory, PanelChannelProgram, PanelFitOptions, PanelInference,
    PanelInteractionSpec, PanelOptimizerOptions, PanelProgram, PanelScenario,
};

const GROUPS: u32 = 4;
const STEPS: usize = 8;

/// Four groups of eight periods: channels `a`, `b`, a context `z`, a season and the outcome.
fn panel_data() -> PanelData {
    let units: Vec<PanelUnit> = (0..GROUPS)
        .map(|unit_id| {
            let offset = unit_id as usize;
            let a: Vec<f64> = (0..STEPS).map(|t| ((t + offset) % 4) as f64).collect();
            let b: Vec<f64> = (0..STEPS).map(|t| ((2 * t + offset + 1) % 4) as f64).collect();
            let z: Vec<f64> = (0..STEPS).map(|t| (t % 2) as f64).collect();
            let season = vec![0.0; STEPS];
            let y: Vec<f64> = (0..STEPS)
                .map(|t| {
                    1.0 + 0.8 * a[t]
                        + 0.5 * b[t]
                        + 0.3 * f64::from(unit_id)
                        + 0.1 * disturbance(t + 10 * offset)
                })
                .collect();
            PanelUnit {
                unit_id,
                series: TimeSeriesData::from_f64_columns(
                    [
                        ("a", a.as_slice()),
                        ("b", b.as_slice()),
                        ("y", y.as_slice()),
                        ("z", z.as_slice()),
                        ("season", season.as_slice()),
                    ],
                    100,
                )
                .unwrap(),
            }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

fn panel_columns() -> JointPanelColumns {
    JointPanelColumns {
        channels: vec![id(0), id(1)],
        outcome: id(2),
        confounder: id(3),
        season: id(4),
    }
}

/// The four cells of two binary channels, each held for `steps` periods.
fn cell_worlds(steps: usize) -> Vec<JointPanelWorld> {
    let scenarios: Vec<PanelScenario> = (0..GROUPS)
        .map(|group| PanelScenario {
            group: group.to_string(),
            history: vec![
                PanelChannelHistory { id: "0".into(), values: [0.0; 2] },
                PanelChannelHistory { id: "1".into(), values: [0.0; 2] },
            ],
            confounder: vec![0.0; steps],
            season: vec![0; steps],
        })
        .collect();
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .into_iter()
        .enumerate()
        .map(|(index, (a, b))| JointPanelWorld {
            world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            reference: index == 0,
            program: PanelProgram {
                id: format!("cell_{index}"),
                channels: vec![
                    PanelChannelProgram { id: "0".into(), values: vec![a; steps] },
                    PanelChannelProgram { id: "1".into(), values: vec![b; steps] },
                ],
                scenarios: scenarios.clone(),
            },
        })
        .collect()
}

fn interaction_target(
    data: &PanelData,
    lag_order: usize,
    worlds: &[JointPanelWorld],
) -> antecedent_core::ScientificTarget {
    joint_panel_target(
        data.schema(),
        &panel_columns(),
        lag_order,
        worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        InferenceSelection::Frequentist,
    )
    .unwrap()
}

fn panel_request(data: &PanelData, lag_order: usize, steps: usize) -> JointPanelRequest {
    let worlds = cell_worlds(steps);
    JointPanelRequest {
        columns: panel_columns(),
        interactions: vec![PanelInteractionSpec::from_saturated_features(
            "ab",
            &ValueExpr::var(id(0)).mul(ValueExpr::var(id(1))),
        )
        .unwrap()],
        seasons: 1,
        lag_order,
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 500, gradient_tolerance: 1e-5 },
            inference: PanelInference::Frequentist { draws: 20 },
            initial: None,
        },
        target: interaction_target(data, lag_order, &worlds),
        worlds,
        assignment: None,
        temporal: None,
    }
}

/// Two binary actions `a`, `d` confounded by `z`, with a null interaction.
fn static_pair_data(n: usize) -> (TabularData, Dag) {
    let z: Vec<f64> =
        (0..n).map(|i| 1.2 * (0.37 * i as f64).sin() + 0.5 * (1.1 * i as f64).cos()).collect();
    let a: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(
                fractional(i as f64 * 0.618_033_988_749_894_9 + 0.1) < sigmoid(-0.2 + 0.8 * z[i]),
            )
        })
        .collect();
    let d: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(
                fractional(i as f64 * 0.754_877_666_246_692_7 + 0.4) < sigmoid(-0.1 + 0.7 * z[i]),
            )
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 + 0.4 * z[i] + 0.3 * (2.9 * i as f64).sin()).collect();
    let data = table(&[("a", &a), ("d", &d), ("y", &y), ("z", &z)]);
    (data, dag(4, &[(3, 0), (3, 1), (3, 2), (0, 2), (1, 2)]))
}

fn joint_query(a: f64, d: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: id(2),
        interventions: Arc::from([
            Intervention::set(id(0), Value::f64(a)),
            Intervention::set(id(1), Value::f64(d)),
        ]),
    })
}

fn sorted_coefficients(target: &antecedent_core::ScientificTarget) -> Vec<f64> {
    let TargetForm::Contrast(contrast) = target.form() else { panic!("a contrast target") };
    let mut coefficients: Vec<f64> = contrast.terms().iter().map(|term| term.coefficient).collect();
    coefficients.sort_by(f64::total_cmp);
    coefficients
}

/// outcome: equivalent target (worlds, coefficients); `not_equivalent` licence, estimator and
/// the temporal one-step frame.
///
/// The static interaction contrast is `m11 - m10 - m01 + m00` over the four static decision
/// worlds. A panel interaction with one-step programs and no lag compiles to the same four
/// worlds (the same simultaneous interventions on a static time frame) and the same coefficients.
/// The numerical routes then differ: the static cells are identified from a graph and the family
/// contrast is an identified claim, where the dynamic panel mechanism is model-conditional and
/// never an identified claim. A panel with a lag order and the same one-step programs is a
/// temporal frame of horizon one, which is a different scientific target.
#[test]
fn horizon_one_joint_interaction_is_the_static_decision_world_contrast_but_not_its_licence() {
    // The static family contrast over cell scores.
    let (data, graph) = static_pair_data(800);
    let context = ExecutionContext::for_tests(271);
    let batch = BatchStudy::new(data.clone(), graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .family_contrast(Some(CellFamilyContrast::Interaction))
        .prepare_cells(&[joint_query(1.0, 1.0), joint_query(1.0, 0.0)], &context)
        .unwrap()
        .estimate(&data, &context)
        .unwrap();
    let static_effect = batch[0].estimate.as_effect().unwrap();
    let binding = static_effect.family.compiled_contrast.as_ref().expect("a compiled contrast");
    assert!(binding.identification().is_identified_claim());
    let TargetForm::Contrast(static_contrast) = binding.target().form() else {
        panic!("the family contrast is a contrast target")
    };
    let static_worlds = static_contrast.worlds().worlds();
    assert_eq!(static_worlds.len(), 4);
    assert!(static_worlds.iter().all(|world| world.time().is_static()));

    // The panel contrast with one-step programs and no lag.
    let panel = panel_data();
    let worlds = cell_worlds(1);
    let static_frame = interaction_target(&panel, 0, &worlds);
    let TargetForm::Contrast(panel_contrast) = static_frame.form() else { panic!("a contrast") };
    let panel_worlds = panel_contrast.worlds().worlds();
    assert_eq!(panel_worlds.len(), static_worlds.len());
    assert!(panel_worlds.iter().all(|world| world.time().is_static()), "no lag, one step: static");
    for world in panel_worlds {
        assert!(
            static_worlds.iter().any(|candidate| candidate.action() == world.action()),
            "panel world {:?} is not a static cell world",
            world.action()
        );
    }
    assert_eq!(sorted_coefficients(&static_frame), sorted_coefficients(binding.target()));
    assert_eq!(sorted_coefficients(&static_frame), vec![-1.0, -1.0, 1.0, 1.0]);

    // The same programs under a lag order are a temporal frame of horizon one: another target.
    let temporal_frame = interaction_target(&panel, 1, &worlds);
    let TargetForm::Contrast(temporal_contrast) = temporal_frame.form() else {
        panic!("a contrast")
    };
    for world in temporal_contrast.worlds().worlds() {
        assert!(!world.time().is_static());
        assert_eq!(world.time().horizon_steps(), Some(1));
    }
    assert_ne!(static_frame.identity(), temporal_frame.identity());

    // The licence of the numerical route: the fitted dynamic mechanism identifies nothing.
    let prepared =
        Study::prepare_joint_panel(panel.clone(), panel_request(&panel, 0, 1), &context).unwrap();
    assert_eq!(prepared.inspect().fits_performed, 0);
    assert!(!prepared.inspect().identification.is_identified_claim());
}

/// outcome: equivalent frame and fit reuse; the search is local polling, not a certificate.
///
/// The bounded continuous program search over a panel fitted at lag zero with one-step programs
/// acts on static decision worlds. Its first proposal is the reference program, whose utility is
/// the fitted terminal mean of the reference world; every proposal controls exactly the one-step
/// coordinates, the search reuses the single fit, stays inside its budget, and certifies neither
/// a global optimum nor a gap. The same objective on a lag-ordered panel would search a temporal
/// frame, which is not interchangeable with this one (see the target identities above).
#[test]
fn horizon_one_continuous_program_search_polls_static_decision_worlds_on_one_reused_fit() {
    let context = ExecutionContext::for_tests(1);
    let panel = panel_data();
    let prepared =
        Study::prepare_joint_panel(panel.clone(), panel_request(&panel, 0, 1), &context).unwrap();
    let result = prepared.estimate(&context).unwrap();
    let TargetForm::Contrast(contrast) = result.request().target.form() else {
        panic!("a contrast")
    };
    assert!(contrast.worlds().worlds().iter().all(|world| world.time().is_static()));
    assert_eq!(result.point.len(), 4);
    assert!(result.point.iter().all(|response| response.means.len() == 1));

    let budget = 3.0;
    let search = result
        .compile_continuous_search(
            JointPanelContinuousSearchRequest {
                base_world: ScientificWorldId::from_raw(0),
                scope: JointPanelScientificScope::ModelConditional,
                functional: JointPanelResponseFunctional::TerminalMean,
                uncertainty: JointPanelSearchUncertainty::PointMeans,
                contract: ProgramSearchContract {
                    space: ProgramSearchSpace {
                        id: "panel_response".into(),
                        variables: vec!["response".into()],
                    },
                    objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
                    constraints: vec![],
                },
                domain: FiniteProgramDomain {
                    controls: (0..2)
                        .map(|channel| ProgramControl {
                            variable: id(channel),
                            offset: 0,
                            lower: 0.0,
                            upper: 2.5,
                            unit_cost: 1.0,
                        })
                        .collect(),
                    budget: Some(budget),
                    restrictions: vec![],
                },
                utility_units: "response".into(),
                options: ContinuousSearchOptions {
                    max_evaluations: 30,
                    max_iterations: 3,
                    initial_step: 0.35,
                    step_tolerance: 1e-4,
                    objective_tolerance: 1e-10,
                },
                initial: vec![0.0; 2],
            },
            &context,
        )
        .unwrap();
    let found = search.execute(&context).unwrap();
    let first = found.search.attempts[0].row.as_ref().unwrap().value.unwrap();
    assert!(
        (first - result.point[0].means[0]).abs() < 1e-9,
        "the reference proposal is the fitted reference world: {first} vs {}",
        result.point[0].means[0]
    );
    assert!(found.search.attempts.iter().all(|attempt| attempt.controls.len() == 2));
    let best = found.search.best.as_ref().expect("a feasible program");
    assert!(best.cost <= budget + 1e-12);
    assert!(best.value.unwrap() >= first);
    assert!(!found.search.global_certificate);
    assert_eq!(found.search.gap, None);
    assert_eq!(found.search.reliability, "unmeasured");
    assert_eq!(prepared.fits_performed(), 1);
}
