// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use antecedent_core::{InferenceSelection, ScientificWorldId, ValueExpr};
use antecedent_data::{PanelUnit, TableView, TimeSeriesData};
use antecedent_estimate::joint_panel::{PanelInference, PanelOptimizerOptions, PanelScenario};
/// Sixty rows per unit with pseudo-random unit-specific doses and confounder and a response
/// built from lagged, saturating exposure to every channel plus deterministic noise, so that the
/// mechanism's likelihood is regular: a fit on it converges to a point with positive definite
/// curvature, which a program search over point means requires. The first two rows are the
/// pre-period, with zero doses and no outcome.
fn regular_columns(unit: u32) -> [Vec<f64>; 6] {
    let n = 60_usize;
    let mut state = 0x2545_F491_4F6C_DD1D_u64 ^ (u64::from(unit) + 1).wrapping_mul(0x9E37_79B9);
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f64 / (1_u64 << 31) as f64
    };
    let mut doses = |n: usize| -> Vec<f64> {
        (0..n).map(|t| if t < 2 || next() < 0.2 { 0.0 } else { 3.0 * next() }).collect()
    };
    let a = doses(n);
    let b = doses(n);
    let c = doses(n);
    let z: Vec<f64> =
        (0..n).map(|t| if t < 2 { f64::NAN } else { next() * 2.0 - 1.0 }).collect();
    let lagged = |x: &[f64], t: usize| 0.6 * x[t] + 0.3 * x[t - 1] + 0.1 * x[t - 2];
    let y: Vec<f64> = (0..n)
        .map(|t| {
            if t < 2 {
                return f64::NAN;
            }
            let (ea, eb, ec) = (lagged(&a, t), lagged(&b, t), lagged(&c, t));
            let noise = 0.1 * (next() - 0.5);
            1.0 + 1.5 * ea / (ea + 1.0) + 1.0 * eb / (eb + 1.0) + 0.6 * ec / (ec + 1.0)
                + 0.4 * z[t]
                + noise
        })
        .collect();
    [a, b, c, y, z, vec![0.0; n]]
}
pub(super) fn fixture() -> (PanelData, JointPanelRequest) {
    let units = (0..4)
        .map(|unit_id| {
            let [a, b, _, y, z, season] = regular_columns(unit_id);
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
        .collect::<Vec<_>>();
    let source = PanelData::try_new(Arc::from(units)).unwrap();
    let columns = JointPanelColumns {
        channels: vec![VariableId::from_raw(0), VariableId::from_raw(1)],
        outcome: VariableId::from_raw(2),
        confounder: VariableId::from_raw(3),
        season: VariableId::from_raw(4),
    };
    let scenarios = (0..4)
        .map(|id| PanelScenario {
            group: id.to_string(),
            history: vec![
                engine::PanelChannelHistory { id: "0".into(), values: [0.0; 2] },
                engine::PanelChannelHistory { id: "1".into(), values: [0.0; 2] },
            ],
            confounder: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            season: vec![0; 6],
        })
        .collect::<Vec<_>>();
    let worlds = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .into_iter()
        .enumerate()
        .map(|(index, (a, b))| JointPanelWorld {
            world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            reference: index == 0,
            program: PanelProgram {
                id: format!("p{index}"),
                channels: vec![
                    engine::PanelChannelProgram { id: "0".into(), values: vec![a; 6] },
                    engine::PanelChannelProgram { id: "1".into(), values: vec![b; 6] },
                ],
                scenarios: scenarios.clone(),
            },
        })
        .collect::<Vec<_>>();
    let target = joint_panel_target(
        source.schema(),
        &columns,
        2,
        &worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let request = JointPanelRequest {
        columns,
        interactions: vec![
            engine::PanelInteractionSpec::from_saturated_features(
                "ab",
                &ValueExpr::var(VariableId::from_raw(0))
                    .mul(ValueExpr::var(VariableId::from_raw(1))),
            )
            .unwrap(),
        ],
        seasons: 1,
        lag_order: 2,
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 3000, gradient_tolerance: 1e-4 },
            inference: PanelInference::Frequentist { draws: 0 },
            initial: None,
        },
        family: engine::PanelFamily::default(),
        worlds,
        target,
        assignment: None,
        temporal: None,
    };
    (source, request)
}
#[test]
fn inspection_retains_actual_failed_design_and_exact_history_without_fitting() {
    let (source, request) = fixture();
    let prepared =
        Study::prepare_joint_panel(source, request, &ExecutionContext::for_tests(1)).unwrap();
    let info = prepared.inspect();
    assert_eq!(info.fits_performed, 0);
    assert!(!info.identification.is_identified_claim());
    assert_eq!(info.design_attempts.len(), 3);
    assert!(info.design_attempts[0].detail.contains("static"));
    assert!(info.design_attempts[2].next_attempt.is_some());
    assert_eq!(
        prepared.model_data().groups[0].history[0].map(f64::to_bits),
        [0.0_f64.to_bits(); 2]
    );
}
#[test]
fn changed_program_cannot_borrow_old_scientific_target() {
    let (source, mut request) = fixture();
    request.worlds[3].program.channels[0].values[0] = 2.0;
    match Study::prepare_joint_panel(source, request, &ExecutionContext::for_tests(1)) {
        Err(JointPanelError::Compile(message)) => {
            assert!(message.contains("differs from the retained program"), "{message}");
        }
        other => {
            panic!("a changed program must be refused as differing, got {:?}", other.map(|_| ()))
        }
    }
}
#[test]
fn memory_refusal_precedes_source_materialization() {
    let (source, request) = fixture();
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(8));
    assert!(matches!(
        Study::prepare_joint_panel(source, request, &ctx),
        Err(JointPanelError::Engine(engine::PanelError::Resource(_)))
    ));
}

fn three_channel_data() -> PanelData {
    let units = (0..4)
        .map(|unit_id| {
            // Three channels that are not affine functions of one another: the fit refuses
            // channels whose separate effects the data cannot identify.
            let [a, b, c, outcome, confounder, seasons] = regular_columns(unit_id);
            PanelUnit {
                unit_id,
                series: TimeSeriesData::from_f64_columns(
                    [
                        ("a", a.as_slice()),
                        ("b", b.as_slice()),
                        ("c", c.as_slice()),
                        ("y", outcome.as_slice()),
                        ("z", confounder.as_slice()),
                        ("season", seasons.as_slice()),
                    ],
                    100,
                )
                .unwrap(),
            }
        })
        .collect::<Vec<_>>();
    PanelData::try_new(Arc::from(units)).unwrap()
}
fn three_channel_prepared() -> PreparedStudy<JointPanelState> {
    let source = three_channel_data();
    let columns = JointPanelColumns {
        channels: (0..3).map(VariableId::from_raw).collect(),
        outcome: VariableId::from_raw(3),
        confounder: VariableId::from_raw(4),
        season: VariableId::from_raw(5),
    };
    let scenarios = (0..4)
        .map(|group| PanelScenario {
            group: group.to_string(),
            history: (0..3)
                .map(|channel| engine::PanelChannelHistory {
                    id: channel.to_string(),
                    values: [0.0; 2],
                })
                .collect(),
            confounder: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            season: vec![0; 6],
        })
        .collect::<Vec<_>>();
    let worlds = (0..8_u32)
        .map(|subset| JointPanelWorld {
            world: ScientificWorldId::from_raw(subset),
            reference: subset == 0,
            program: PanelProgram {
                id: format!("subset_{subset}"),
                channels: (0..3)
                    .map(|channel| engine::PanelChannelProgram {
                        id: channel.to_string(),
                        values: vec![f64::from((subset >> channel) & 1); 6],
                    })
                    .collect(),
                scenarios: scenarios.clone(),
            },
        })
        .collect::<Vec<_>>();
    let terms = (0..8_u32)
        .map(|subset| {
            (
                ScientificWorldId::from_raw(subset),
                if subset.count_ones() % 2 == 1 { 1.0 } else { -1.0 },
            )
        })
        .collect();
    let target = joint_panel_target(
        source.schema(),
        &columns,
        2,
        &worlds,
        JointPanelFunctional::Linear { terms, reference: ScientificWorldId::from_raw(0) },
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let (_, base) = fixture();
    Study::prepare_joint_panel(
        source,
        JointPanelRequest {
            columns,
            interactions: vec![
                engine::PanelInteractionSpec::from_saturated_features(
                    "ab",
                    &ValueExpr::var(VariableId::from_raw(0))
                        .mul(ValueExpr::var(VariableId::from_raw(1))),
                )
                .unwrap(),
                engine::PanelInteractionSpec::from_saturated_features(
                    "abc",
                    &ValueExpr::var(VariableId::from_raw(0))
                        .mul(ValueExpr::var(VariableId::from_raw(1)))
                        .mul(ValueExpr::var(VariableId::from_raw(2))),
                )
                .unwrap(),
            ],
            seasons: 1,
            lag_order: 2,
            options: base.options,
            family: engine::PanelFamily::default(),
            worlds,
            target,
            assignment: None,
            temporal: None,
        },
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}
#[test]
fn three_channel_third_order_contrast_uses_the_same_public_compiler() {
    let prepared = three_channel_prepared();
    assert_eq!(prepared.model_data().channels.len(), 3);
    let interaction = prepared.model_data().interactions[1].expression.to_domain();
    let axis = (0..3).map(VariableId::from_raw).collect::<Vec<_>>();
    let program =
        antecedent_model::program::NodeProgram::compile_columns(&axis, &interaction).unwrap();
    assert_eq!(
        program.evaluate_columns(&[0.5, 0.25, 0.75], None).unwrap().to_bits(),
        (0.5_f64 * 0.25 * 0.75).to_bits()
    );
    assert_eq!(prepared.inspect().fits_performed, 0);
    assert_eq!(prepared.inspect().design_attempts[2].worlds.len(), 8);
    assert!(!prepared.inspect().identification.is_identified_claim());
}

fn assert_temporal_projection_binding(result: &JointPanelResult) {
    let pool = result.context.admission();
    let before = pool.available().bytes();
    let terminal = result.project_mean_source(&JointPanelResponseFunctional::TerminalMean).unwrap();
    assert_eq!(terminal.target().identity(), result.request().target.identity());
    assert_eq!(terminal.identification().identity(), result.identification().identity());
    let authenticated = terminal.projection().expect("terminal whole-world projection is retained");
    assert_eq!(authenticated.basis.len(), result.request().worlds.len());
    assert!(authenticated.outputs.iter().enumerate().all(|(i, o)| o.terms == vec![(i, 1.0)]));
    assert_eq!(
        super::JointLawExperimentResult::quantity(&terminal).unwrap(),
        result.experiment_quantity(&JointPanelResponseFunctional::TerminalMean).unwrap()
    );
    assert!(terminal.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    let retained = pool.available().bytes();
    assert!(retained < before);
    let shared = terminal.clone();
    assert!(std::ptr::eq(terminal.as_numerical_source(), shared.as_numerical_source()));
    drop(terminal);
    assert_eq!(pool.available().bytes(), retained);
    drop(shared);
    assert_eq!(pool.available().bytes(), before);
    let full = result.mean_source().unwrap();
    let window = result
        .project_mean_source(&JointPanelResponseFunctional::WindowAverage { from: 0, until: 2 })
        .unwrap();
    let later = result
        .project_mean_source(&JointPanelResponseFunctional::WindowAverage { from: 3, until: 5 })
        .unwrap();
    assert_ne!(window.target().identity(), result.request().target.identity());
    assert_ne!(window.question_identity(), later.question_identity());
    assert_eq!(window.projection().unwrap().basis.len(), 48);
    assert_eq!(window.projection().unwrap().outputs.len(), 8);
    assert!(
        window
            .projection()
            .unwrap()
            .outputs
            .iter()
            .all(|row| row.terms.len() == 3 && row.quantity.horizon == 3)
    );
    assert_eq!(full.projection().unwrap().outputs.len(), 48);
    assert_eq!(full.projection().unwrap().basis[0].quantity.horizon, 1);
    assert!(window.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    let mut malformed = window.projection().unwrap().clone();
    malformed.basis[0].quantity.horizon = 6;
    assert!(
        antecedent_design::scientific_source::ScientificDecisionSource::new_projected(
            window.target().clone(),
            window.identification().clone(),
            window.dependency_identity(),
            window.as_numerical_source().clone(),
            malformed
        )
        .is_err()
    );
    let mut tiny_projection = result.clone();
    tiny_projection.context.set_memory_limit(Some(8));
    assert!(matches!(
        tiny_projection.mean_source(),
        Err(JointPanelError::Engine(engine::PanelError::Resource(_)))
    ));
}

#[test]
fn continuous_three_channel_search_reuses_fit_and_improves_a_budget_feasible_nonbinary_program() {
    use antecedent_design::decision_contract::{
        ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
    };
    use antecedent_design::program_search::{
        FiniteProgramDomain, ProgramControl, continuous::ContinuousSearchOptions,
    };
    let ctx = ExecutionContext::for_tests(1);
    let prepared = three_channel_prepared();
    let result = prepared.estimate(&ctx).unwrap();
    assert_temporal_projection_binding(&result);
    let domain = FiniteProgramDomain {
        controls: (0..6)
            .flat_map(|offset| {
                (0..3).map(move |channel| ProgramControl {
                    variable: VariableId::from_raw(channel),
                    offset,
                    lower: 0.0,
                    upper: 2.5,
                    unit_cost: 1.0,
                })
            })
            .collect(),
        budget: Some(3.0),
        restrictions: vec![],
    };
    let contract = ProgramSearchContract {
        space: ProgramSearchSpace {
            id: "panel_response".into(),
            variables: vec!["response".into()],
        },
        objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
        constraints: vec![],
    };
    let search = result
        .compile_continuous_search(
            JointPanelContinuousSearchRequest {
                base_world: ScientificWorldId::from_raw(0),
                scope: JointPanelScientificScope::ModelConditional,
                functional: JointPanelResponseFunctional::TerminalMean,
                uncertainty: JointPanelSearchUncertainty::PointMeans,
                contract,
                domain,
                utility_units: "response".into(),
                options: ContinuousSearchOptions {
                    max_evaluations: 40,
                    max_iterations: 3,
                    initial_step: 0.35,
                    step_tolerance: 1e-4,
                    objective_tolerance: 1e-10,
                },
                initial: vec![0.0; 18],
            },
            &ctx,
        )
        .unwrap();
    let found = search.execute(&ctx).unwrap();
    let first = found.search.attempts[0].row.as_ref().unwrap().value.unwrap();
    let best = found.search.best.as_ref().unwrap();
    assert!(best.value.unwrap() > first, "{:?}", found.search);
    assert!(best.cost <= 3.0);
    assert!(best.controls.iter().any(|value| value.fract() != 0.0));
    assert!(!found.search.global_certificate);
    assert_eq!(found.search.gap, None);
    assert_eq!(prepared.fits_performed(), 1);
    assert!(
        found
            .best_source
            .as_ref()
            .unwrap()
            .authorize(JointPanelScientificScope::IdentifiedCausal)
            .is_err()
    );
    let mut tiny = ctx.clone();
    tiny.set_memory_limit(Some(8));
    assert!(matches!(
        search.execute(&tiny),
        Err(JointPanelError::Engine(engine::PanelError::Resource(_)))
    ));
    let cancelled = ctx.clone();
    cancelled.cancellation.cancel();
    assert!(matches!(
        search.execute(&cancelled),
        Err(JointPanelError::Engine(engine::PanelError::Cancelled))
    ));
}

#[test]
fn certified_panel_search_derives_its_objective_and_preserves_the_actual_source() {
    use antecedent_design::decision_contract::{
        ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
    };
    use antecedent_design::program_search::{
        FiniteProgramDomain, ProgramControl,
        certified::{CertificateOptions, CertificateStop},
        continuous::ContinuousSearchOptions,
    };
    let ctx = ExecutionContext::for_tests(1);
    let prepared = three_channel_prepared();
    let result = prepared.estimate(&ctx).unwrap();
    let search = result
        .compile_certified_search(
            JointPanelContinuousSearchRequest {
                base_world: ScientificWorldId::from_raw(0),
                scope: JointPanelScientificScope::ModelConditional,
                functional: JointPanelResponseFunctional::TerminalMean,
                uncertainty: JointPanelSearchUncertainty::PointMeans,
                contract: ProgramSearchContract {
                    space: ProgramSearchSpace {
                        id: "actual_fitted_response".into(),
                        variables: vec!["response".into()],
                    },
                    objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
                    constraints: vec![],
                },
                domain: FiniteProgramDomain {
                    controls: (0..6)
                        .flat_map(|offset| {
                            (0..3).map(move |channel| ProgramControl {
                                variable: VariableId::from_raw(channel),
                                offset,
                                lower: 0.0,
                                upper: if offset == 5 && channel == 0 { 2.5 } else { 0.0 },
                                unit_cost: if offset == 5 && channel == 0 { 1.0 } else { 0.0 },
                            })
                        })
                        .collect(),
                    budget: Some(2.0),
                    restrictions: vec![],
                },
                utility_units: "response".into(),
                initial: vec![0.0; 18],
                options: ContinuousSearchOptions {
                    max_evaluations: 10,
                    max_iterations: 2,
                    initial_step: 0.5,
                    step_tolerance: 1e-6,
                    objective_tolerance: 1e-10,
                },
            },
            &ctx,
        )
        .unwrap();
    let output = search
        .execute(CertificateOptions { absolute_gap: 1e-3, max_splits: 10_000 }, &ctx)
        .unwrap();
    assert_eq!(output.certificate.stop, CertificateStop::Certified);
    assert!(output.certificate.gap <= 1e-3);
    assert!(output.certificate.controls[15] <= 2.0);
    assert_eq!(output.fitted_dependency, result.dependency_identity());
    assert_eq!(output.parent_target, result.request().target.identity());
    assert!(output.best_source.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    let bytes =
        search.export(CertificateOptions { absolute_gap: 1e-3, max_splits: 10_000 }, &ctx).unwrap();
    let replay = search.consume_current(&bytes, &ctx).unwrap();
    assert_eq!(replay.identity, output.certificate.identity);
    assert_eq!(replay.global_upper.to_bits(), output.certificate.global_upper.to_bits());
    let (_, historical) =
        replay_panel_certificate(&bytes, &ExecutionContext::for_tests(99)).unwrap();
    assert_eq!(historical.controls, output.certificate.controls);
    assert_eq!(prepared.fits_performed(), 1);
}

fn static_source_data() -> (PanelData, antecedent_data::TabularData) {
    use antecedent_data::TabularData;
    let mut all = [vec![], vec![], vec![], vec![], vec![]];
    let mut units = Vec::new();
    for group in 0..4_u32 {
        let mut columns = [vec![], vec![], vec![], vec![], vec![]];
        for index in 0..40_u32 {
            let a = f64::from(index % 2);
            let b = f64::from((index / 2) % 2);
            let y = 1.0
                + 2.0 * a
                + 0.5 * b
                + 0.1 * f64::from(group)
                + 0.01 * (f64::from(index) * 0.31).sin();
            for (column, value) in columns.iter_mut().zip([a, b, y, 0.0, 0.0]) {
                column.push(value);
            }
        }
        for (target, values) in all.iter_mut().zip(&columns) {
            target.extend_from_slice(values);
        }
        units.push(PanelUnit {
            unit_id: group,
            series: TimeSeriesData::from_f64_columns(
                ["a", "b", "y", "z", "season"].into_iter().zip(columns.iter().map(Vec::as_slice)),
                100,
            )
            .unwrap(),
        });
    }
    let data = PanelData::try_new(Arc::from(units)).unwrap();
    let tabular = TabularData::from_f64_columns(
        ["a", "b", "y", "z", "season"].into_iter().zip(all.iter().map(Vec::as_slice)),
    )
    .unwrap();
    (data, tabular)
}
fn static_source_fixture()
-> (PreparedStudy<JointPanelState>, crate::PreparedJointResponse, antecedent_data::TabularData) {
    use antecedent_core::{
        AssumptionSet, Conditioning, GraphFamily, Intervention, InterventionWorld,
        ModelAssumptions, ObservationUnit, ScientificTarget, TargetForm, TimeFrame, Value,
        WorldBinding, WorldCollection,
    };
    let (data, tabular) = static_source_data();
    let (_, mut request) = fixture();
    request.lag_order = 0;
    for world in &mut request.worlds {
        for channel in &mut world.program.channels {
            channel.values.truncate(1);
        }
        for scenario in &mut world.program.scenarios {
            scenario.confounder = vec![0.0];
            scenario.season = vec![0];
        }
    }
    request.target = joint_panel_target(
        data.schema(),
        &request.columns,
        0,
        &request.worlds,
        JointPanelFunctional::Worlds,
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let source_worlds = contracts::target_worlds(&request.target)
        .unwrap()
        .iter()
        .map(|world| {
            InterventionWorld::new(
                world.id(),
                world.action().clone(),
                world.is_reference(),
                WorldBinding::new(
                    world.binding().population().clone(),
                    ObservationUnit::new("row").unwrap(),
                    world.binding().outcome().clone(),
                    Conditioning::unconditional(),
                )
                .unwrap(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let source_target = ScientificTarget::new(
        TargetForm::Worlds(WorldCollection::new(source_worlds).unwrap()),
        ModelAssumptions::new(GraphFamily::Unspecified, AssumptionSet::default()),
        InferenceSelection::Frequentist,
    );
    let ctx = ExecutionContext::for_tests(1);
    let graph =
        antecedent_graph::Dag::from_named_edges(tabular.schema(), &[("a", "y"), ("b", "y")])
            .unwrap();
    let query = antecedent_core::ResponseQuery::new(
        antecedent_core::ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([
                Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            ]),
        },
    );
    let proof = Study::tabular(tabular.clone())
        .graph(graph)
        .query(query)
        .inference(crate::InferenceMode::Frequentist)
        .refute(crate::RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .compile_joint_response(&source_target, &ctx)
        .unwrap();
    (Study::prepare_joint_panel(data, request, &ctx).unwrap(), proof, tabular)
}
#[test]
fn static_panel_reuses_independent_graph_proof_without_gaussian_restriction_or_extra_work() {
    use antecedent_core::execution_attempt::{Operation, observe_with_context};
    let (panel, proof, data) = static_source_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let result = observe_with_context(&ctx, || panel.with_static_source_proof(proof, &data, &ctx));
    assert_eq!(result.report.counts(Operation::Identification).attempted, 0);
    assert_eq!(result.report.counts(Operation::AdjustedFit).attempted, 0);
    let bound = result.result.unwrap();
    assert!(bound.inspect().identification.is_identified_claim());
    assert!(!bound.static_source_proof().unwrap().numerical_program().has_fit());
    assert!(!bound.inspect().identification.assumptions().entries.iter().any(|record|matches!(&record.assumption,antecedent_core::Assumption::ParametricRestriction(premise) if premise.id.as_ref()=="response.joint.gaussian_linear")));
}
#[test]
fn static_panel_refuses_source_proof_for_another_future_context_law() {
    let (panel, proof, data) = static_source_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let mut worlds = panel.request().worlds.clone();
    worlds[0].program.scenarios[0].confounder[0] = 1.0;
    let target = joint_panel_target(
        panel.source_data().schema(),
        &panel.request().columns,
        0,
        &worlds,
        JointPanelFunctional::Worlds,
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let changed = panel.retarget(worlds, target, &ctx).unwrap();
    let error = changed.with_static_source_proof(proof, &data, &ctx).unwrap_err();
    assert!(
        matches!(error,JointPanelError::Unsupported(message) if message.contains("context law differs"))
    );
}

#[test]
fn unrolled_graph_attempt_records_the_general_id_verdict_without_claiming_identification() {
    use antecedent_identify::panel_unrolled::{PanelNode, UnrolledPanelGraph, UnrolledPanelSpec};
    let (data, request) = fixture();
    let ctx = ExecutionContext::for_tests(1);
    let panel = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let spec = |latent| UnrolledPanelSpec {
        periods: 6,
        channels: 2,
        contexts: 1,
        carryover_reach: 1,
        outcome_persistence: true,
        history_feedback: true,
        latent_confounding: latent,
        ..UnrolledPanelSpec::default()
    };
    let attempt = |panel: &PreparedStudy<JointPanelState>| {
        panel
            .inspect()
            .design_attempts
            .iter()
            .find(|a| a.id == "joint_adjustment.observational")
            .unwrap()
            .clone()
    };
    // No unmeasured cause: a DAG over the unrolled variables is identified by general ID, but the
    // fitted Gaussian/AR panel is not thereby identified.
    let identified = panel
        .with_unrolled_graph_attempt(&UnrolledPanelGraph::build(spec(vec![])).unwrap(), &[5], &ctx)
        .unwrap();
    let recorded = attempt(&identified);
    assert!(recorded.detail.contains("general ID identifies"), "{}", recorded.detail);
    assert_eq!(recorded.outcome, JointPanelDesignOutcome::Unsupported);
    assert!(recorded.proof.is_none());
    assert_eq!(
        identified.inspect().identification.identity(),
        panel.inspect().identification.identity()
    );
    // A bow between a channel and the outcome it drives is refused with the hedge.
    let bow = vec![(PanelNode::Channel { channel: 0, period: 2 }, PanelNode::Outcome(2))];
    let refused = panel
        .with_unrolled_graph_attempt(&UnrolledPanelGraph::build(spec(bow)).unwrap(), &[5], &ctx)
        .unwrap();
    assert!(attempt(&refused).detail.contains("hedge"), "{}", attempt(&refused).detail);
    // A graph of another shape than the program is refused.
    let wrong = UnrolledPanelSpec { periods: 3, channels: 2, ..UnrolledPanelSpec::default() };
    assert!(
        panel.with_unrolled_graph_attempt(&UnrolledPanelGraph::build(wrong).unwrap(), &[2], &ctx).is_err()
    );
}

#[test]
fn retarget_admits_changed_programs_retains_and_releases_their_lease_without_fitting() {
    let (data, request) = fixture();
    let ctx = ExecutionContext::for_tests(3);
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let before = ctx.admission().available().bytes();
    let next = prepared
        .retarget(prepared.request().worlds.clone(), prepared.request().target.clone(), &ctx)
        .unwrap();
    let retained = ctx.admission().available().bytes();
    assert!(retained < before);
    let copy = next.clone();
    drop(next);
    assert_eq!(ctx.admission().available().bytes(), retained);
    drop(copy);
    assert_eq!(ctx.admission().available().bytes(), before);
    let mut worlds = prepared.request().worlds.clone();
    for world in &mut worlds {
        for channel in &mut world.program.channels {
            channel.values.resize(600, 0.0);
        }
        for scenario in &mut world.program.scenarios {
            scenario.confounder.resize(600, 0.0);
            scenario.season.resize(600, 0);
        }
    }
    let target = joint_panel_target(
        prepared.source_data().schema(),
        &prepared.request().columns,
        2,
        &worlds,
        JointPanelFunctional::Worlds,
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let mut limited = ctx.clone();
    limited.set_memory_limit(Some(before - retained));
    assert!(matches!(
        prepared.retarget(worlds, target, &limited),
        Err(JointPanelError::Engine(engine::PanelError::Resource(_)))
    ));
    assert_eq!(ctx.admission().available().bytes(), before);
    assert_eq!(prepared.fits_performed(), 0);
}

fn dose_time_fixture(
    multiple_channels: bool,
    policy: &antecedent_core::TemporalPolicy,
) -> (PanelData, JointPanelRequest, antecedent_core::ScientificTarget) {
    use antecedent_core::{
        DoseTimeSurface, Intervention, InterventionWorld, ScientificTarget, TargetForm, Value,
        WorldAction, WorldCollection,
    };
    let (data, mut request) = fixture();
    request.interactions.clear();
    if !multiple_channels {
        request.columns.channels.truncate(1);
    }
    for (index, world) in request.worlds.iter_mut().enumerate() {
        let horizon = index % 2 + 1;
        let level = if index < 2 { 0.0 } else { 1.0 };
        if !multiple_channels {
            world.program.channels.truncate(1);
        }
        for channel in &mut world.program.channels {
            channel.values = vec![level; horizon];
        }
        for scenario in &mut world.program.scenarios {
            if !multiple_channels {
                scenario.history.truncate(1);
            }
            scenario.confounder.truncate(horizon);
            scenario.season.truncate(horizon);
        }
    }
    request.options.optimizer =
        PanelOptimizerOptions { max_iterations: 1, gradient_tolerance: 1e-30 };
    request.options.inference = PanelInference::Frequentist { draws: 2 };
    request.target = joint_panel_target(
        data.schema(),
        &request.columns,
        2,
        &request.worlds,
        JointPanelFunctional::Worlds,
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let worlds = contracts::target_worlds(&request.target)
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, world)| {
            InterventionWorld::new(
                world.id(),
                WorldAction::schedule(
                    Intervention::set(
                        request.columns.channels[0],
                        Value::f64(if index < 2 { 0.0 } else { 1.0 }),
                    ),
                    policy.clone(),
                )
                .unwrap(),
                world.is_reference(),
                world.binding().clone(),
                world.time().clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let surface = DoseTimeSurface::new(
        WorldCollection::new(worlds).unwrap(),
        request.columns.channels[0],
        [0.0, 1.0],
        [1, 2],
        (0..4).map(ScientificWorldId::from_raw),
    )
    .unwrap();
    let target = ScientificTarget::new(
        TargetForm::DoseTime(surface),
        request.target.assumptions().clone(),
        request.target.inference(),
    );
    (data, request, target)
}
#[test]
fn prepared_scalar_dose_time_surface_reuses_frozen_fit_and_retains_failed_slots() {
    let (data, request, target) =
        dose_time_fixture(false, &antecedent_core::TemporalPolicy::sustained(0, 1));
    let ctx = ExecutionContext::for_tests(9);
    let frozen = {
        let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
        prepared.estimate(&ctx).unwrap()
    };
    assert_eq!(frozen.draws.len(), 2);
    assert!(frozen.draws.iter().all(Option::is_none));
    let worlds = frozen.request().worlds.clone();
    let result = frozen.retarget_frozen(worlds, target.clone(), &ctx).unwrap();
    drop(frozen);
    assert_eq!(result.prepared().fits_performed(), 1);
    assert_eq!(result.request().target.identity(), target.identity());
    assert_eq!(result.point.iter().map(|row| row.means.len()).collect::<Vec<_>>(), [1, 2, 1, 2]);
    assert!(result.identification().assembled_obligation().is_some());
    assert!(result.draws.iter().all(Option::is_none));
    assert!(result.law_source().is_err());
    let window = result
        .project_mean_source(&JointPanelResponseFunctional::WindowAverage { from: 0, until: 0 })
        .unwrap();
    assert_ne!(window.target().identity(), target.identity());
    assert!(window.projection().unwrap().outputs.iter().all(|output| output.quantity.horizon == 1));
    assert!(window.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    assert_eq!(result.prepared().fits_performed(), 1);
}
#[test]
fn dose_time_refuses_implicit_extra_channel_and_unassigned_exposure_continuation() {
    for (multiple_channels, policy) in [
        (true, antecedent_core::TemporalPolicy::sustained(0, 1)),
        (false, antecedent_core::TemporalPolicy::pulse(0)),
    ] {
        let (data, mut request, target) = dose_time_fixture(multiple_channels, &policy);
        request.target = target;
        assert!(matches!(
            Study::prepare_joint_panel(data, request, &ExecutionContext::for_tests(9)),
            Err(JointPanelError::Unsupported(_))
        ));
    }
}

#[test]
fn reference_decomposition_inherits_exact_hybrid_proofs_and_refuses_unearned_transport() {
    use super::explanation::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
    };
    let (panel, proof, data) = static_source_fixture();
    let ctx = ExecutionContext::for_tests(71);
    let result =
        panel.with_static_source_proof(proof, &data, &ctx).unwrap().estimate(&ctx).unwrap();
    let worlds = &result.request().worlds;
    let groups = result.prepared().model_data().groups.len();
    let mut request = PanelDecompositionRequest {
        before: DecompositionEndpoint {
            program: worlds[0].program.clone(),
            population_weights: vec![1.0 / groups as f64; groups],
        },
        after: DecompositionEndpoint {
            program: worlds[worlds.len() - 1].program.clone(),
            population_weights: vec![1.0 / groups as f64; groups],
        },
        time_weights: vec![1.0],
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let periods = MechanismPeriodMap { before: vec![0], after: vec![0] };
    let checked = result.decompose_reference_checked(&request, &periods, &ctx).unwrap();
    checked.require_identified_reference().unwrap();
    assert_eq!(checked.hybrids.len(), 16);
    assert!(checked.hybrids.iter().all(|world| world.source_world.is_some()));
    assert!(
        (checked.decomposition.point.shapley[0] - checked.decomposition.point.total).abs() < 1e-10
    );
    for index in [1, 2, 3] {
        assert!(checked.decomposition.point.shapley[index].abs() < 1e-10);
    }
    request.after.population_weights.fill(0.0);
    request.after.population_weights[0] = 1.0;
    let unsupported = result.decompose_reference_checked(&request, &periods, &ctx).unwrap();
    assert!(!unsupported.is_identified_reference());
    assert!(unsupported.require_identified_reference().is_err());
    assert!(unsupported.hybrids.iter().filter(|world| world.mask & 4 != 0).all(|world| {
        world.unresolved.iter().any(|premise| premise.contains("population transport"))
    }));
}

#[test]
fn design_adaptive_reference_executes_identified_steps_and_names_the_design_for_the_rest() {
    use super::explanation::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
        ReferenceObligation,
    };
    let (panel, proof, data) = static_source_fixture();
    let ctx = ExecutionContext::for_tests(74);
    let result =
        panel.with_static_source_proof(proof, &data, &ctx).unwrap().estimate(&ctx).unwrap();
    let worlds = &result.request().worlds;
    let groups = result.prepared().model_data().groups.len();
    let mut request = PanelDecompositionRequest {
        before: DecompositionEndpoint {
            program: worlds[0].program.clone(),
            population_weights: vec![1.0 / groups as f64; groups],
        },
        after: DecompositionEndpoint {
            program: worlds[worlds.len() - 1].program.clone(),
            population_weights: vec![1.0 / groups as f64; groups],
        },
        time_weights: vec![1.0],
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let periods = MechanismPeriodMap { before: vec![0], after: vec![0] };
    // Everything identified: every step is available and nothing is left to design.
    let complete =
        result.attempt_design_adaptive_reference(&request, &periods, None, &ctx).unwrap();
    assert!(complete.is_complete());
    assert!(complete.next_designs.is_empty());
    assert!(complete.unidentified_components.is_empty());
    assert_eq!(complete.identified_steps.len(), 32);
    assert!(complete.selection.is_none());
    // A changed population is not identified by the design, but the exposure change at the
    // reference population still is, and it is executed on every aligned law slot.
    request.after.population_weights.fill(0.0);
    request.after.population_weights[0] = 1.0;
    let partial = result.attempt_design_adaptive_reference(&request, &periods, None, &ctx).unwrap();
    assert!(!partial.is_complete());
    assert!(partial.coarser.is_none());
    assert_eq!(partial.unidentified_components, vec![ChangeComponent::Population]);
    let exposure = partial
        .identified_steps
        .iter()
        .filter(|step| step.component == ChangeComponent::Exposure)
        .collect::<Vec<_>>();
    assert_eq!(exposure.len(), 4);
    assert!(partial.identified_steps.iter().all(|step| step.context_mask & 4 == 0));
    let hybrids = &partial.requested.decomposition.point.hybrids;
    let at_reference = exposure.iter().find(|step| step.context_mask == 0).unwrap();
    assert!((at_reference.point - (hybrids[1] - hybrids[0])).abs() < 1e-12);
    assert_eq!(at_reference.draws.len(), partial.requested.decomposition.draws.len());
    for (step, draw) in at_reference.draws.iter().zip(&partial.requested.decomposition.draws) {
        assert_eq!(step.is_some(), draw.is_some());
    }
    let transport = partial
        .next_designs
        .iter()
        .find(|step| step.obligation == ReferenceObligation::ChangedPopulationTransport)
        .unwrap();
    assert_eq!(transport.components, vec![ChangeComponent::Population]);
    assert!(transport.hybrids.iter().all(|mask| mask & 4 != 0) && transport.hybrids.len() == 8);
    assert!(transport.design.contains("population"));
    // Every executed step is reproduced from the stored fit by an independent consumer.
    let bytes = result.export(&ctx).unwrap();
    let historical =
        antecedent_io::joint_panel_artifact::consume_panel_execution(&bytes, &ctx).unwrap();
    let replay = historical.program().decompose_reference(&request, &periods, &ctx).unwrap();
    for step in &partial.identified_steps {
        let bit = match step.component {
            ChangeComponent::Exposure => 0,
            ChangeComponent::Saturation => 1,
            ChangeComponent::Population => 2,
            ChangeComponent::Mechanism => 3,
        };
        let stepped = usize::from(step.context_mask | (1 << bit));
        let context = usize::from(step.context_mask);
        assert_eq!(step.point, replay.point.hybrids[stepped] - replay.point.hybrids[context]);
    }
}

#[test]
fn compiled_component_bindings_consume_identified_worlds_and_reject_hybrid_and_overlap_errors() {
    use super::explanation::{ChangeComponent, ReferenceComponentBindings};
    use antecedent_design::scientific_source::ScientificConsumptionScope;
    let (panel, proof, data) = static_source_fixture();
    let ctx = ExecutionContext::for_tests(72);
    let result =
        panel.with_static_source_proof(proof, &data, &ctx).unwrap().estimate(&ctx).unwrap();
    let worlds = &result.request().worlds;
    let before = worlds[0].world;
    let after = worlds[worlds.len() - 1].world;
    let mut bindings = ReferenceComponentBindings {
        exposure: result.request().columns.channels.clone(),
        saturation: vec![],
        mechanism: vec![],
        hybrids: std::array::from_fn(|mask| if mask & 1 == 0 { before } else { after }),
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let checked = result
        .decompose_compiled_components(
            &bindings,
            ScientificConsumptionScope::IdentifiedCausal,
            false,
            &ctx,
        )
        .unwrap();
    assert_eq!(checked.question_identity, checked.source.question_identity());
    let bytes = result.export(&ctx).unwrap();
    let current = result.consume_current(&bytes, &ctx).unwrap();
    let replayed = current
        .decompose_compiled_components(
            &bindings,
            ScientificConsumptionScope::IdentifiedCausal,
            false,
            &ctx,
        )
        .unwrap();
    assert_eq!(checked.decomposition.point, replayed.decomposition.point);
    let historical =
        antecedent_io::joint_panel_artifact::consume_panel_execution(&bytes, &ctx).unwrap();
    let historical_values = std::array::from_fn(|mask| {
        let index = worlds.iter().position(|world| world.world == bindings.hybrids[mask]).unwrap();
        *historical.point()[index].means.last().unwrap()
    });
    let independent =
        engine::decomposition::decompose_values(historical_values, bindings.order).unwrap();
    assert_eq!(checked.decomposition.point, independent);

    assert!(
        (checked.decomposition.point.shapley[0] - checked.decomposition.point.total).abs() < 1e-10
    );
    for index in [1, 2, 3] {
        assert!(checked.decomposition.point.shapley[index].abs() < 1e-10);
    }
    bindings.hybrids[1] = before;
    assert!(
        result
            .decompose_compiled_components(
                &bindings,
                ScientificConsumptionScope::IdentifiedCausal,
                false,
                &ctx
            )
            .is_err()
    );
    bindings.hybrids[1] = after;
    bindings.mechanism.push(bindings.exposure[0]);
    assert!(
        result
            .decompose_compiled_components(
                &bindings,
                ScientificConsumptionScope::IdentifiedCausal,
                false,
                &ctx
            )
            .is_err()
    );
}

#[test]
fn unidentified_finer_decomposition_executes_actual_coarser_response_and_preserves_refusal() {
    use super::explanation::{
        ChangeComponent, DecompositionEndpoint, IdentifiedReferenceAttempt,
        JointPanelExplanationRequest, MechanismPeriodMap, PanelDecompositionRequest,
    };
    let (data, request) = fixture();
    let ctx = ExecutionContext::for_tests(73);
    let result = Study::prepare_joint_panel(data, request, &ctx).unwrap().estimate(&ctx).unwrap();
    let worlds = &result.request().worlds;
    let request = PanelDecompositionRequest {
        before: DecompositionEndpoint {
            program: worlds[0].program.clone(),
            population_weights: vec![0.25; 4],
        },
        after: DecompositionEndpoint {
            program: worlds[3].program.clone(),
            population_weights: vec![0.25; 4],
        },
        time_weights: vec![1.0 / 6.0; 6],
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let fallback = JointPanelExplanationRequest {
        explanation: engine::time_varying::PanelExplanation::Mechanism,
        world: worlds[3].world,
        reference_world: worlds[0].world,
        before: (0, 1),
        after: (4, 5),
    };
    let attempt = result
        .attempt_identified_reference_decomposition(
            &request,
            &MechanismPeriodMap { before: vec![0; 6], after: vec![0; 6] },
            fallback,
            &ctx,
        )
        .unwrap();
    let IdentifiedReferenceAttempt::Coarser { requested, executed } = attempt else {
        panic!("unidentified finer claim was accepted");
    };
    assert!(!requested.is_identified_reference());
    assert!(executed.point.is_finite());
    assert!(executed.require_unique_attribution().is_err());
    assert_eq!(executed.compiled_response().dependency_identity(), result.dependency_identity());
    assert_eq!(result.prepared().fits_performed(), 1);
}

#[test]
fn broader_family_runs_through_the_compiler_and_the_frozen_artifact_replays_every_slot() {
    let (source, mut request) = fixture();
    let spatial = engine::PanelSpatial {
        kernel: engine::PanelSpatialKernel::Exponential,
        groups: (0..4).map(|group| group.to_string()).collect(),
        distances: (0..4_i32)
            .flat_map(|a| (0..4_i32).map(move |b| f64::from((a - b).abs())))
            .collect(),
    };
    request.family = engine::PanelFamily {
        adstock: engine::PanelAdstock::Geometric,
        saturation: engine::PanelSaturation::Hill,
        scope: engine::PanelSaturationScope::Classes { assignment: vec![0, 1, 1, 0] },
        spatial: Some(spatial),
    };
    let geometric = |theta: f64| {
        let sum = 1.0 + theta + theta * theta;
        [1.0 / sum, theta / sum, theta * theta / sum]
    };
    let initial = engine::PanelParameters {
        intercept: 1.0,
        channels: vec![
            engine::PanelChannelParameters {
                id: "0".into(),
                coefficient: 1.0,
                lag: geometric(0.5),
            },
            engine::PanelChannelParameters {
                id: "1".into(),
                coefficient: 0.8,
                lag: geometric(0.3),
            },
        ],
        interactions: vec![engine::PanelInteractionParameter { id: "ab".into(), coefficient: 0.5 }],
        confounder: 0.2,
        season_effects: vec![0.0],
        half_saturation: 1.5,
        rho: 0.2,
        innovation_sd: 0.5,
        group_sd: 0.5,
        family: engine::PanelFamilyParameters {
            hill_shape: vec![1.5, 1.0],
            class_half_saturation: vec![2.5],
            spatial_range: 2.0,
        },
    };
    // A permissive stopping rule fixes the declared start, so every slot is exercised.
    request.options = PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 2, gradient_tolerance: 1e100 },
        inference: PanelInference::FrequentistBootstrap { draws: 2 },
        initial: Some(initial.clone()),
    };
    let ctx = ExecutionContext::for_tests(7);
    let prepared = Study::prepare_joint_panel(source, request, &ctx).unwrap();
    assert_eq!(prepared.model_data().family.classes(), 2);
    assert!(prepared.model_data().family.spatial.is_some());
    assert!(!prepared.inspect().identification.is_identified_claim());
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else {
        panic!("an invariant mechanism was requested");
    };
    let family = &fit.point.parameters.family;
    assert_eq!(family.hill_shape, initial.family.hill_shape);
    assert_eq!(family.class_half_saturation.len(), 1);
    assert!((family.spatial_range - 2.0).abs() < 1e-12);
    assert!(result.draws.iter().all(Option::is_some));
    for draw in fit.draws.iter().flatten() {
        assert_eq!(draw.group_intercepts.len(), 4);
        assert!(draw.parameters.family.spatial_range > 0.0);
    }
    // The frozen program carries the family; an independent consumer reproduces every slot.
    let bytes = result.export(&ctx).unwrap();
    let historical =
        antecedent_io::joint_panel_artifact::consume_panel_execution(&bytes, &ctx).unwrap();
    assert_eq!(historical.point(), result.point.as_slice());
    assert_eq!(historical.draws(), result.draws.as_slice());
    assert!(!historical.grants_current_authority());
    let current = result.consume_current(&bytes, &ctx).unwrap();
    assert_eq!(current.point, result.point);
}

#[test]
fn reviewed_family_is_the_request_default_and_a_family_change_refits() {
    let (source, request) = fixture();
    assert!(request.family.is_reviewed());
    let reviewed = request.clone();
    let mut hill = request;
    hill.family.saturation = engine::PanelSaturation::Hill;
    let ctx = ExecutionContext::for_tests(3);
    let (a, b) = (
        Study::prepare_joint_panel(source.clone(), reviewed, &ctx).unwrap(),
        Study::prepare_joint_panel(source, hill, &ctx).unwrap(),
    );
    // The family is part of the source identity, so a cached reviewed fit is never reused.
    assert_ne!(a.inspect().source_identity, b.inspect().source_identity);
    assert!(a.model_data().family.is_reviewed() && !b.model_data().family.is_reviewed());
}

fn declared_series(columns: &[Vec<f64>; 5]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [
            ("a", columns[0].as_slice()),
            ("b", columns[1].as_slice()),
            ("y", columns[2].as_slice()),
            ("z", columns[3].as_slice()),
            ("season", columns[4].as_slice()),
        ],
        100,
    )
    .unwrap()
}
fn declared_columns() -> [Vec<f64>; 5] {
    [
        vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 2.0],
        vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
        vec![f64::NAN, f64::NAN, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0; 8],
    ]
}
fn declared_panel_with(
    ids: [u32; 4],
    edit: impl Fn(u32, &mut [Vec<f64>; 5]),
    wrap: impl Fn(TimeSeriesData) -> TimeSeriesData,
) -> PanelData {
    let units = ids
        .into_iter()
        .enumerate()
        .map(|(position, unit_id)| {
            let mut columns = declared_columns();
            edit(u32::try_from(position).unwrap(), &mut columns);
            PanelUnit { unit_id, series: wrap(declared_series(&columns)) }
        })
        .collect::<Vec<_>>();
    PanelData::try_new(Arc::from(units)).unwrap()
}
fn declared_panel(edit: impl Fn(u32, &mut [Vec<f64>; 5])) -> PanelData {
    declared_panel_with([0, 1, 2, 3], edit, |series| series)
}
fn prepare_declared(
    data: PanelData,
    edit: impl FnOnce(&mut JointPanelRequest),
) -> Result<PreparedStudy<JointPanelState>, JointPanelError> {
    let (_, mut request) = fixture();
    edit(&mut request);
    Study::prepare_joint_panel(data, request, &ExecutionContext::for_tests(1))
}
#[test]
fn panel_source_refuses_weights_masks_and_missing_cells_with_typed_errors() {
    use antecedent_data::ValidityBitmap;
    assert!(prepare_declared(declared_panel(|_, _| {}), |_| {}).is_ok());
    // A weighted or masked series is a declaration the engine reads or refuses by name.
    let weighted = declared_panel_with(
        [0, 1, 2, 3],
        |_, _| {},
        |series| {
            series.with_weights(Arc::from(vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])).unwrap()
        },
    );
    let masked = declared_panel_with(
        [0, 1, 2, 3],
        |_, _| {},
        |series| {
            let keep = [true, true, true, true, true, true, true, false];
            series.with_analysis_mask(ValidityBitmap::from_bools(&keep)).unwrap()
        },
    );
    // Weights that vary over a unit have no pseudo-likelihood and are refused by name; a mask
    // is read, hiding the row and every period whose exposure window reads it.
    match prepare_declared(weighted, |_| {}) {
        Err(JointPanelError::Compile(message)) => {
            assert!(message.contains("one strictly positive finite value over the unit"), "{message}");
        }
        other => panic!("expected a typed compile refusal, got {other:?}"),
    }
    let hidden = prepare_declared(masked, |_| {}).unwrap();
    let group = &hidden.model_data().groups[0];
    assert!(group.outcome[5].is_nan() && group.outcome[..5].iter().all(|v| v.is_finite()));
    assert_eq!(hidden.data_accounting().units[0].masked_rows, vec![7]);
    // A missing cell in any read column of the fitted rows is refused, never dropped or filled.
    for (role, column, row) in [("outcome", 2, 4), ("confounder", 3, 4), ("channel", 0, 5)] {
        let data = declared_panel(|unit, columns| {
            if unit == 2 {
                columns[column][row] = f64::NAN;
            }
        });
        assert!(
            matches!(
                prepare_declared(data, |_| {}),
                Err(JointPanelError::Engine(engine::PanelError::Invalid(_)))
            ),
            "a missing {role} cell must be refused"
        );
    }
    // A missing pre-period exposure is refused too: history is never imputed.
    let history = declared_panel(|unit, columns| {
        if unit == 1 {
            columns[1][0] = f64::NAN;
        }
    });
    assert!(matches!(
        prepare_declared(history, |_| {}),
        Err(JointPanelError::Engine(engine::PanelError::Invalid(_)))
    ));
    let season = declared_panel(|unit, columns| {
        if unit == 3 {
            columns[4][4] = f64::NAN;
        }
    });
    assert!(matches!(prepare_declared(season, |_| {}), Err(JointPanelError::Compile(_))));
}
#[test]
fn panel_source_refuses_a_categorical_channel_instead_of_coding_it() {
    use antecedent_core::{CategoryDomainId, CausalSchemaBuilder, MeasurementSpec};
    use antecedent_core::{SmallRoleSet, ValueType};
    use antecedent_data::{
        CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column,
        OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, UnknownCategoryPolicy,
        ValidityBitmap,
    };
    let columns = declared_columns();
    let mut builder = CausalSchemaBuilder::new();
    for (index, name) in ["a", "b", "y", "z", "season"].into_iter().enumerate() {
        builder
            .add_variable(
                name,
                if index == 0 { ValueType::Categorical } else { ValueType::Continuous },
                SmallRoleSet::empty(),
                None,
                (index == 0).then(|| CategoryDomainId::from_raw(0)),
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let domain = Arc::new(
        CategoryDomain::try_new(
            CategoryDomainId::from_raw(0),
            ["none", "low", "high", "peak"]
                .map(|label| CategoryLevel { label: Arc::from(label) })
                .to_vec(),
            false,
            None,
            UnknownCategoryPolicy::Fail,
        )
        .unwrap(),
    );
    let units = (0..4)
        .map(|unit_id| {
            let mut owned = vec![OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    VariableId::from_raw(0),
                    columns[0]
                        .iter()
                        .map(|level| CategoryCode::from_raw(*level as u32))
                        .collect::<Vec<_>>(),
                    ValidityBitmap::all_valid(8),
                    Arc::clone(&domain),
                )
                .unwrap(),
            )];
            for (index, values) in columns.iter().enumerate().skip(1) {
                owned.push(OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(index).unwrap()),
                        Arc::from(values.clone()),
                        ValidityBitmap::from_bools(
                            &values.iter().map(|v| !v.is_nan()).collect::<Vec<_>>(),
                        ),
                    )
                    .unwrap(),
                ));
            }
            PanelUnit {
                unit_id,
                series: TimeSeriesData::try_new(
                    OwnedColumnarStorage::try_new(schema.clone(), owned, None, None).unwrap(),
                    TimeIndex {
                        regularity: SamplingRegularity::Regular { interval_ns: 100 },
                        length: 8,
                    },
                )
                .unwrap(),
            }
        })
        .collect::<Vec<_>>();
    let data = PanelData::try_new(Arc::from(units)).unwrap();
    assert!(matches!(prepare_declared(data, |_| {}), Err(JointPanelError::Compile(_))));
}
#[test]
fn panel_source_refuses_repeated_clusters_and_lag_orders_it_does_not_model() {
    // Each unit is one cluster of the group-intercept law; a repeated label is not a cluster.
    let repeated = declared_panel_with([0, 1, 1, 3], |_, _| {}, |series| series);
    assert!(matches!(
        prepare_declared(repeated, |_| {}),
        Err(JointPanelError::Engine(engine::PanelError::Invalid(_)))
    ));
    // The serial law is the stationary AR(1) with two pre-period rows, or none at all.
    for lag_order in [1, 3, 5] {
        match prepare_declared(declared_panel(|_, _| {}), |request| request.lag_order = lag_order) {
            Err(JointPanelError::Compile(message)) => {
                assert!(message.contains("only lag orders zero and two"), "{message}");
            }
            other => panic!("lag order {lag_order} must be refused, got {other:?}"),
        }
    }
}
#[test]
fn panel_source_refuses_constant_and_collinear_channels_and_a_constant_outcome() {
    let constant = declared_panel(|_, columns| columns[0] = vec![1.0; 8]);
    let collinear = declared_panel(|_, columns| {
        columns[1] = columns[0].iter().map(|value| 2.0 * value + 1.0).collect();
    });
    for refused in [constant, collinear] {
        match prepare_declared(refused, |_| {}) {
            Err(JointPanelError::Engine(engine::PanelError::Invalid(message))) => {
                assert!(message.contains("not identified"), "{message}");
            }
            other => panic!("expected an identification refusal, got {other:?}"),
        }
    }
    let flat_outcome = declared_panel(|_, columns| {
        columns[2] = vec![f64::NAN, f64::NAN, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    });
    let prepared = prepare_declared(flat_outcome, |request| {
        request.options.optimizer.max_iterations = 300;
    })
    .unwrap();
    assert!(matches!(
        prepared.estimate(&ExecutionContext::for_tests(1)),
        Err(JointPanelError::Engine(_))
    ));
}
#[test]
fn continuous_search_grants_only_conditional_use_and_refuses_a_causal_scope_for_new_programs() {
    use antecedent_design::decision_contract::{
        ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
    };
    use antecedent_design::program_search::{
        FiniteProgramDomain, ProgramControl, continuous::ContinuousSearchOptions,
    };
    let ctx = ExecutionContext::for_tests(1);
    let (source, request) = fixture();
    let result = Study::prepare_joint_panel(source, request, &ctx).unwrap().estimate(&ctx).unwrap();
    let search_request = |scope| JointPanelContinuousSearchRequest {
        base_world: ScientificWorldId::from_raw(0),
        scope,
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
            controls: (0..6)
                .flat_map(|offset| {
                    (0..2).map(move |channel| ProgramControl {
                        variable: VariableId::from_raw(channel),
                        offset,
                        lower: 0.0,
                        upper: 2.5,
                        unit_cost: 1.0,
                    })
                })
                .collect(),
            budget: Some(3.0),
            restrictions: vec![],
        },
        utility_units: "response".into(),
        options: ContinuousSearchOptions {
            max_evaluations: 30,
            max_iterations: 2,
            initial_step: 0.35,
            step_tolerance: 1e-4,
            objective_tolerance: 1e-10,
        },
        initial: vec![0.0; 12],
    };
    match result.compile_continuous_search(
        search_request(JointPanelScientificScope::IdentifiedCausal),
        &ctx,
    ) {
        Err(JointPanelError::Unsupported(message)) => {
            assert!(message.contains("own causal design proof"), "{message}");
        }
        other => {
            panic!("a causal scope for a new program must be refused, got {:?}", other.map(|_| ()))
        }
    }
    let granted = result
        .compile_continuous_search(
            search_request(JointPanelScientificScope::ModelConditional),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let source = granted.best_source.expect("a feasible best program is retained");
    assert!(source.authorize(JointPanelScientificScope::ModelConditional).is_ok());
    assert!(source.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    assert!(!granted.search.global_certificate);
}
