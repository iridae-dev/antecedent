// SPDX-License-Identifier: MIT OR Apache-2.0
//! Public retained execution on the independent X06 panel; no calibration claim.
#[path = "joint_panel_experiment_chain.rs"]
mod experiment_chain;
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;
use antecedent::{Study, analysis::joint_panel::JointPanelFit};
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_panel::{PanelInference, PanelPrior};

#[test]
fn actual_frequentist_panel_fit_retains_aligned_worlds_and_reuses_after_builder_disposal() {
    let ctx = cached_context(20_261_021);
    let prepared = {
        let data = fixture::source(false);
        let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 100 });
        Study::prepare_joint_panel(data, request, &ctx).unwrap()
    };
    assert_eq!(prepared.inspect().fits_performed, 0);
    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(
        result.numerical_mean_source().unwrap().estimate,
        antecedent_design::decision_eval::MeanEstimate::Frequentist,
    );
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("invariant mechanism") };
    assert!(fit.diagnostics.optimizer_converged, "{:?}", fit.diagnostics);
    assert!(fit.diagnostics.curvature_available, "{:?}", fit.diagnostics);
    assert!(result.draws.iter().any(Option::is_some), "{:?}", fit.diagnostics);
    assert_eq!(result.point.len(), 4);
    assert!(result.point.iter().all(|response| response.means.len() == 24
        && response.means.iter().all(|value| value.is_finite())));
    for row in result.draws.iter().flatten() {
        assert_eq!(row.len(), 4);
        assert!(row.iter().all(|response| response.means.len() == 24));
    }
    let next = prepared
        .retarget(prepared.request().worlds.clone(), prepared.request().target.clone(), &ctx)
        .unwrap();
    assert_eq!(next.estimate(&ctx).unwrap().point, result.point);
    assert_eq!(prepared.fits_performed(), 1);
    assert_eq!(next.fits_performed(), 1);
    assert!(!prepared.inspect().identification.is_identified_claim());
    exercise_reference_decomposition(&result, &ctx);
    assert_streamed_projection_matches(&result, &ctx);
    // The streamed route fits through the same cache: no second fit is performed.
    let projected = prepared
        .estimate_projected(
            &antecedent::analysis::joint_panel::JointPanelResponseFunctional::TerminalMean,
            16,
            &ctx,
        )
        .unwrap();
    assert_eq!(prepared.fits_performed(), 1);
    assert_eq!(projected.draws().len(), result.draws.len());
    assert_fresh_panel_replay(&result, &ctx);
}

/// The tiled reduction must be the full-output route followed by the same projection:
/// same draw order and failed slots, same joint alignment across worlds, same bits.
fn assert_streamed_projection_matches(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    use antecedent::analysis::joint_panel::JointPanelResponseFunctional;
    use antecedent_estimate::joint_panel::project_panel_responses;
    let functional = JointPanelResponseFunctional::WindowChange { before: (0, 5), after: (18, 23) };
    let draws = result.draws.len();
    for tile in [1, 7, draws.max(1)] {
        let streamed = result.project_streamed(&functional, tile, ctx).unwrap();
        assert_eq!(
            streamed.point(),
            project_panel_responses(&result.point, streamed.weights()).unwrap()
        );
        assert_eq!(streamed.draws().len(), draws);
        for (reduced, full) in streamed.draws().iter().zip(&result.draws) {
            let expected = full
                .as_deref()
                .map(|rows| project_panel_responses(rows, streamed.weights()).unwrap());
            assert_eq!(reduced, &expected);
        }
        assert!(streamed.receipt().peak_tile_draws <= tile);
        assert_eq!(streamed.receipt().tiles, draws.div_ceil(tile));
    }
    assert!(result.project_streamed(&functional, 0, ctx).is_err());
    let worlds = result.request().worlds.clone();
    let target = result.request().target.clone();
    let frozen = result
        .project_frozen_streamed(worlds.clone(), target.clone(), &functional, 5, ctx)
        .unwrap();
    let retained = result.retarget_frozen(worlds, target, ctx).unwrap();
    assert_eq!(frozen.draws().len(), retained.draws.len());
    for (reduced, full) in frozen.draws().iter().zip(&retained.draws) {
        let expected =
            full.as_deref().map(|rows| project_panel_responses(rows, frozen.weights()).unwrap());
        assert_eq!(reduced, &expected);
    }
    assert_eq!(result.prepared().fits_performed(), 1);
}

fn exercise_reference_decomposition(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    use antecedent::analysis::joint_panel::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
    };
    let worlds = &result.request().worlds;
    let horizon = worlds[0].horizon();
    let weights = vec![
        1.0 / result.prepared().model_data().groups.len() as f64;
        result.prepared().model_data().groups.len()
    ];
    let request = PanelDecompositionRequest {
        before: DecompositionEndpoint {
            program: worlds[0].program.clone(),
            population_weights: weights.clone(),
        },
        after: DecompositionEndpoint {
            program: worlds[3].program.clone(),
            population_weights: weights,
        },
        time_weights: vec![1.0 / horizon as f64; horizon],
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let parameter_periods = match &result.fit {
        JointPanelFit::Invariant(_) => vec![0; horizon],
        JointPanelFit::Varying(_) => (0..horizon).collect(),
    };
    let periods =
        MechanismPeriodMap { before: parameter_periods.clone(), after: parameter_periods };
    let decomposition = result.decompose_reference(&request, &periods, ctx).unwrap();
    let expected = result.point[3]
        .means
        .iter()
        .zip(&result.point[0].means)
        .map(|(a, b)| (a - b) / horizon as f64)
        .sum::<f64>();
    assert!((decomposition.point.total - expected).abs() < 1e-10);
    assert!((decomposition.point.shapley[0] - expected).abs() < 1e-10);
    for j in 1..4 {
        assert!(decomposition.point.shapley[j].abs() < 1e-10);
    }
    for (draw, parent) in decomposition.draws.iter().zip(&result.draws) {
        assert_eq!(draw.is_some(), parent.is_some());
        if let Some(draw) = draw {
            assert!((draw.shapley.iter().sum::<f64>() - draw.total).abs() < 1e-10);
        }
    }
    // The streamed reduction repeats every draw's evaluation, so the posterior-sized law
    // is left to the cheaper estimator-level equality test.
    if decomposition.draws.len() <= 1_000 {
        let streamed = result.decompose_reference_streamed(&request, &periods, ctx).unwrap();
        assert_eq!(streamed.point, decomposition.point);
        assert_eq!(streamed.joint, decomposition.joint_uncertainty());
        assert_eq!(streamed.requested, decomposition.draws.len());
        assert_eq!(streamed.failed, decomposition.draws.iter().filter(|d| d.is_none()).count());
    }
    match &result.fit {
        JointPanelFit::Invariant(_) => {
            assert!(result.decomposition_selection(&decomposition).unwrap().is_none());
        }
        JointPanelFit::Varying(fit) => {
            // Change-point and smoothing selection is carried through the decomposed law:
            // every complete replicate is accounted for under the specification it selected.
            let split = result.decomposition_selection(&decomposition).unwrap().unwrap();
            let complete = decomposition.draws.iter().flatten().count();
            assert_eq!((split.requested, split.complete), (decomposition.draws.len(), complete));
            assert!((split.specifications.iter().map(|s| s.mass).sum::<f64>() - 1.0).abs() < 1e-12);
            assert_eq!(split.specifications.iter().map(|s| s.slots).sum::<usize>(), complete);
            for share in &split.specifications {
                let counted = fit
                    .selection_counts
                    .iter()
                    .find(|(choice, _)| *choice == share.choice)
                    .map(|(_, count)| *count);
                assert_eq!(counted, Some(share.slots));
            }
            for j in 0..4 {
                assert!(split.between_variance[j] >= 0.0 && split.within_variance[j] >= 0.0);
                assert!((0.0..=1.0).contains(&split.selection_share[j]));
            }
            // The fixture's design earns no proof: the finer claim stays model-conditional,
            // no component step is called identified, and the missing design is named.
            let adaptive =
                result.attempt_design_adaptive_reference(&request, &periods, None, ctx).unwrap();
            assert!(!adaptive.is_complete());
            assert!(adaptive.identified_steps.is_empty());
            assert_eq!(adaptive.unidentified_components.len(), 4);
            assert_eq!(adaptive.selection.as_ref(), Some(&split));
            assert!(adaptive.next_designs.iter().any(|step| {
                step.obligation
                    == antecedent::analysis::joint_panel::ReferenceObligation::MissingIdentificationProof
            }));
        }
    }
    let bytes = result.export(ctx).unwrap();
    let independent =
        antecedent_io::joint_panel_artifact::consume_panel_execution(&bytes, ctx).unwrap();
    let replay = independent.program().decompose_reference(&request, &periods, ctx).unwrap();
    assert_eq!(decomposition.point, replay.point);
    assert_eq!(decomposition.draws, replay.draws);
    assert_eq!(result.prepared().fits_performed(), 1);
}

#[test]
fn static_full_procedure_bootstrap_decomposition_executes_and_replays_publicly() {
    use antecedent::analysis::joint_panel::{JointPanelFunctional, joint_panel_target};
    use antecedent_core::{InferenceSelection, ScientificWorldId};
    let ctx = cached_context(20261024);
    let data = fixture::source_static();
    let mut request =
        fixture::request(&data, false, PanelInference::FrequentistBootstrap { draws: 3 });
    request.lag_order = 0;
    request.target = joint_panel_target(
        data.schema(),
        &request.columns,
        0,
        &request.worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else {
        panic!("invariant bootstrap");
    };
    assert!(fit.diagnostics.optimizer_converged, "{:?}", fit.diagnostics);
    assert_eq!(fit.draws.len(), 3);
    assert!(
        fit.draws.iter().any(Option::is_some),
        "full-procedure bootstrap needs successful refits"
    );
    assert!(fit.diagnostics.uncertainty.contains("parametric bootstrap"));
    exercise_reference_decomposition(&result, &ctx);
}

#[test]
fn actual_bayesian_panel_fit_preserves_mixing_diagnostics_and_shared_draw_axes() {
    let ctx = cached_context(20_261_021);
    let data = fixture::source(false);
    let request = fixture::request(
        &data,
        false,
        PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: fixture::posterior_sampling(),
        },
    );
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(
        result.numerical_mean_source().unwrap().estimate,
        antecedent_design::decision_eval::MeanEstimate::Frequentist,
    );
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("invariant mechanism") };
    assert_eq!(result.draws.len(), 48_000);
    assert!(result.draws.iter().any(Option::is_some));
    assert!(fit.diagnostics.max_rhat.is_some());
    assert_eq!(fit.diagnostics.posterior_converged, Some(true), "{:?}", fit.diagnostics);
    assert_eq!(fit.diagnostics.calibration, "unmeasured");
    assert!(
        result
            .draws
            .iter()
            .flatten()
            .all(|row| row.len() == 4 && row.iter().all(|response| response.means.len() == 24))
    );
    assert_eq!(prepared.estimate(&ctx).unwrap().point, result.point);
    assert_eq!(prepared.fits_performed(), 1);
    assert!(!prepared.inspect().identification.is_identified_claim());
    exercise_reference_decomposition(&result, &ctx);
    exercise_selected_posterior(&result, &ctx);
    experiment_chain::exercise(&prepared, &result, &ctx);
}

fn lag_zero_terminal_interaction(row: &[antecedent_estimate::joint_panel::PanelResponse]) -> f64 {
    let last = |world: usize| *row[world].means.last().unwrap();
    last(3) - last(1) - last(2) + last(0)
}

/// The lag-zero (static) panel through the Bayesian fit and the experiment-to-decision chain.
///
/// The panel is simulated from a retained known truth (`joint_panel_fixture::lag_zero_truth`)
/// with no carryover and no serial error. The posterior is sampled under the strict policy
/// (four chains, rank-normalized R-hat at most 1.01, bulk and tail ESS at least 400) and must
/// mix under it. The interaction contrast has a closed-form truth,
/// `b_ab * s(4)^2 = 1.5 * (4 / 7)^2`, because the group intercepts, the confounder and the
/// season cancel in the joint minus both marginals plus the reference.
#[test]
fn build_joint_panel_lag_zero_bayesian_chain_mixes_and_recovers_the_known_interaction() {
    let ctx = cached_context(20_261_029);
    let data = fixture::source_lag_zero();
    let request = fixture::request_lag_zero(
        &data,
        PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: fixture::lag_zero_sampling(),
        },
    );
    assert_eq!(request.lag_order, 0);
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("invariant mechanism") };
    let diagnostics = &fit.diagnostics;
    let rhat = diagnostics.max_rhat.expect("sampled chains report R-hat");
    let bulk = diagnostics.min_bulk_ess.unwrap();
    let tail = diagnostics.min_tail_ess.unwrap();
    assert!(rhat < 1.01, "R-hat {rhat}");
    assert!(bulk >= 400.0 && tail >= 400.0, "ESS bulk {bulk} tail {tail}");
    assert_eq!(diagnostics.posterior_converged, Some(true), "{diagnostics:?}");
    assert!(diagnostics.optimizer_converged && diagnostics.curvature_available);
    assert_eq!(diagnostics.calibration, "unmeasured");
    assert_eq!(result.draws.len(), 100_000);
    assert_eq!(diagnostics.failed_draws, 0);

    // Known answer: the interaction contrast is b_ab * s(4)^2 at every period.
    let truth = fixture::lag_zero_truth();
    let known = truth.interaction * (4.0_f64 / (4.0 + truth.half_saturation)).powi(2);
    assert!((known - 1.5 * 16.0 / 49.0).abs() < 1e-15);
    let draws: Vec<f64> = result.draws.iter().flatten().map(|row| lag_zero_terminal_interaction(row)).collect();
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let sd = (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    assert!(
        (mean - known).abs() < 3.0 * sd,
        "posterior interaction {mean} +- {sd} does not cover the known {known}"
    );
    assert!(sd < 0.5, "the posterior is informative: sd {sd}");
    let point = lag_zero_terminal_interaction(&result.point);
    assert!((point - known).abs() < 3.0 * sd, "point {point} vs known {known}");
    assert_eq!(prepared.fits_performed(), 1);
    experiment_chain::exercise_core(&prepared, &result, &ctx);
}

fn exercise_selected_posterior(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    use antecedent::analysis::joint_panel::{
        JointPanelContinuousSearchRequest, JointPanelProgramSearchResult,
        JointPanelResponseFunctional, JointPanelScientificScope, JointPanelSearchUncertainty,
    };
    use antecedent_design::{
        decision_contract::{
            ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
        },
        program_search::{
            FiniteProgramDomain, ProgramControl, certified::CertificateOptions,
            continuous::ContinuousSearchOptions,
        },
        scientific_source::ScientificSourcePayload,
    };
    let request = result.request();
    let base = &request.worlds[0];
    let mut initial = Vec::new();
    let mut controls = Vec::new();
    for (channel, variable) in base.program.channels.iter().zip(&request.columns.channels) {
        for (period, value) in channel.values.iter().copied().enumerate() {
            initial.push(value);
            controls.push(ProgramControl {
                variable: *variable,
                offset: i32::try_from(period).unwrap(),
                lower: value,
                upper: value,
                unit_cost: 0.0,
            });
        }
    }
    let searched = result
        .search_program(
            JointPanelContinuousSearchRequest {
                base_world: base.world,
                scope: JointPanelScientificScope::ModelConditional,
                functional: JointPanelResponseFunctional::TerminalMean,
                uncertainty: JointPanelSearchUncertainty::AlignedLaw,
                contract: ProgramSearchContract {
                    space: ProgramSearchSpace {
                        id: "selected_posterior_response".into(),
                        variables: vec!["response".into()],
                    },
                    objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
                    constraints: vec![],
                },
                domain: FiniteProgramDomain { controls, budget: None, restrictions: vec![] },
                utility_units: "outcome".into(),
                initial,
                options: ContinuousSearchOptions {
                    max_evaluations: 1,
                    max_iterations: 1,
                    initial_step: 0.5,
                    step_tolerance: 1e-6,
                    objective_tolerance: 0.0,
                },
            },
            CertificateOptions { absolute_gap: 0.001, max_splits: 0 },
            ctx,
        )
        .unwrap();
    let JointPanelProgramSearchResult::BestFound { result: searched, .. } = searched else {
        panic!("the point-response certificate must not substitute for a posterior objective")
    };
    let source = searched.best_source.unwrap();
    let ScientificSourcePayload::Law(law) = source.as_numerical_source() else {
        panic!("selected program must retain the genuine complete posterior")
    };
    assert_eq!(
        law.metadata().identity.semantic,
        antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
    );
    assert_eq!(law.draws().len(), result.draws.len());
    for (value, original) in law.draws().iter().zip(&result.draws) {
        assert!((value - original.as_ref().unwrap()[0].means.last().unwrap()).abs() < 1e-12);
    }
    assert!(!searched.search.global_certificate);
}

/// Interaction cells in request order: joint, A only, B only, reference.
fn terminal_interaction(row: &[antecedent_estimate::joint_panel::PanelResponse]) -> f64 {
    let last = |index: usize| *row[index].means.last().unwrap();
    last(3) - last(1) - last(2) + last(0)
}

#[test]
fn scheduled_contrast_is_assembled_on_every_shared_draw_for_both_inferences() {
    use antecedent::analysis::joint_panel::{
        JointPanelContrastUncertainty, JointPanelResponseFunctional,
    };
    let ctx = cached_context(20_261_021);
    for (inference, uncertainty) in [
        (
            PanelInference::Frequentist { draws: 100 },
            JointPanelContrastUncertainty::EstimatorSampling,
        ),
        (
            PanelInference::Bayesian {
                prior: PanelPrior::default(),
                sampler: fixture::posterior_sampling(),
            },
            JointPanelContrastUncertainty::Posterior,
        ),
    ] {
        let data = fixture::source(false);
        let request = fixture::request(&data, false, inference);
        let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        let contrast =
            result.scheduled_contrast(&JointPanelResponseFunctional::TerminalMean, 0.95).unwrap();
        assert_eq!(contrast.uncertainty, uncertainty);
        assert!((contrast.point - terminal_interaction(&result.point)).abs() < 1e-12);
        assert_eq!(contrast.draws.len(), result.draws.len());
        // One contrast per shared parameter draw: never a difference of marginal intervals.
        for (value, row) in contrast.draws.iter().zip(&result.draws) {
            match (value, row) {
                (Some(value), Some(row)) => {
                    assert!((value - terminal_interaction(row)).abs() < 1e-12);
                }
                (None, _) => {}
                (Some(_), None) => panic!("a failed slot cannot carry a contrast"),
            }
        }
        if let Some((low, high)) = contrast.interval {
            assert!(low <= high && contrast.standard_error.unwrap() >= 0.0);
        }
        // A window outside the actual programs is refused, not clipped.
        assert!(
            result
                .scheduled_contrast(
                    &JointPanelResponseFunctional::WindowAverage { from: 0, until: 10_000 },
                    0.95,
                )
                .is_err()
        );
        assert!(
            result.scheduled_contrast(&JointPanelResponseFunctional::TerminalMean, 1.5).is_err()
        );
    }
}

#[test]
fn a_panel_whose_channels_never_move_apart_is_refused_as_collinear() {
    let ctx = cached_context(20_261_021);
    // The retained confounded panel assigns both channels together in every observed period,
    // so their separate effects are not identified and the panel is refused before any claim.
    let together = fixture::source(true);
    let refusal = Study::prepare_joint_panel(
        together.clone(),
        fixture::request(&together, true, PanelInference::Frequentist { draws: 0 }),
        &ctx,
    )
    .err()
    .expect("channels that never move separately are refused");
    assert!(
        refusal.to_string().contains(
            "channel 1 is exactly collinear with the earlier channels; separate effects are not identified"
        ),
        "{refusal}"
    );
}

#[test]
fn confounded_panel_does_not_acquire_a_causal_license_from_a_finite_mechanism() {
    use antecedent::analysis::joint_panel::{
        JointPanelDesignOutcome, declare_joint_panel_exogenous_context,
    };
    let ctx = cached_context(20_261_021);
    // Both channels are driven by an unobserved context and move separately (independent
    // noise), so the interaction fit is well-posed yet biased by that context.
    let data = fixture::source_confounded_varied();
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.options.optimizer =
        antecedent_estimate::joint_panel::PanelOptimizerOptions {
            max_iterations: 600,
            gradient_tolerance: 1e-2,
        };
    let prepared = Study::prepare_joint_panel(data, request.clone(), &ctx).unwrap();
    let inspection = prepared.inspect();
    assert_eq!(inspection.fits_performed, 0);
    assert!(!inspection.identification.is_identified_claim());
    assert!(!inspection.design_attempts.is_empty());
    let observational = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "joint_adjustment.observational")
        .unwrap();
    assert!(matches!(observational.outcome, JointPanelDesignOutcome::Unsupported));
    assert!(observational.detail.contains("not identified through general ID"));
    assert!(observational.detail.contains("accepted point graph"));
    // The fit runs, and a fitted mechanism earns no causal license.
    let fitted = prepared.estimate(&ctx).unwrap();
    assert!(fitted.require_identified().is_err());
    // The same family on the same panel gets none either.
    let mut broader = request;
    broader.family = broader_family();
    let fitted = Study::prepare_joint_panel(fixture::source_confounded_varied(), broader, &ctx)
        .unwrap()
        .estimate(&ctx)
        .unwrap();
    assert!(fitted.require_identified().is_err());
    // The discriminating twin: whole-program randomized assignment of the same outcome model
    // does earn the design-based claim through the same machinery.
    let data = fixture::whole_program_source();
    let mut randomized = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    randomized.target = declare_joint_panel_exogenous_context(&randomized.target);
    randomized.assignment = Some(fixture::whole_program_randomization());
    let twin = Study::prepare_joint_panel(data, randomized, &ctx).unwrap();
    assert!(twin.inspect().identification.is_identified_claim());
}

#[test]
fn period_randomization_does_not_certify_the_held_out_whole_program() {
    let ctx = cached_context(20_261_021);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    let assignment = fixture::period_randomization();
    assignment.validate().unwrap();
    request.assignment = Some(assignment);
    request.target =
        antecedent::analysis::joint_panel::declare_joint_panel_exogenous_context(&request.target);
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let inspection = prepared.inspect();
    assert!(!inspection.identification.is_identified_claim());
    let attempt = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .unwrap();
    assert!(matches!(
        attempt.outcome,
        antecedent::analysis::joint_panel::JointPanelDesignOutcome::Rejected
    ));
    assert!(attempt.detail.contains("independent source groups"));
    assert!(attempt.next_attempt.is_some());
    assert_eq!(inspection.fits_performed, 0);
}

#[test]
fn exact_whole_program_assignment_earns_the_world_claim_without_promoting_model_uncertainty() {
    use antecedent::analysis::joint_panel::{
        JointPanelDesignOutcome, declare_joint_panel_exogenous_context,
    };
    let ctx = cached_context(20_261_021);
    let data = fixture::whole_program_source();
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.target = declare_joint_panel_exogenous_context(&request.target);
    let design = fixture::whole_program_randomization();
    design.validate().unwrap();
    request.assignment = Some(design);
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let inspection = prepared.inspect();
    assert_eq!(inspection.fits_performed, 0);
    assert!(inspection.identification.is_identified_claim());
    let experimental = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .unwrap();
    assert!(matches!(experimental.outcome, JointPanelDesignOutcome::Identified));
    assert!(
        experimental.worlds.iter().all(|world| world.identified && world.matched_groups.len() == 1)
    );
    assert!(experimental.proof.is_some());
}

/// Before/after endpoints of the first and last declared worlds, read at the terminal period.
fn terminal_decomposition(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
) -> (
    antecedent::analysis::joint_panel::PanelDecompositionRequest,
    antecedent::analysis::joint_panel::MechanismPeriodMap,
) {
    use antecedent::analysis::joint_panel::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
    };
    let worlds = &result.request().worlds;
    let horizon = worlds[0].horizon();
    let groups = result.prepared().model_data().groups.len();
    let weights = vec![1.0 / groups as f64; groups];
    let mut time_weights = vec![0.0; horizon];
    time_weights[horizon - 1] = 1.0;
    let request = PanelDecompositionRequest {
        before: DecompositionEndpoint {
            program: worlds[0].program.clone(),
            population_weights: weights.clone(),
        },
        after: DecompositionEndpoint {
            program: worlds[3].program.clone(),
            population_weights: weights,
        },
        time_weights,
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let axis = match &result.fit {
        JointPanelFit::Invariant(_) => vec![0; horizon],
        JointPanelFit::Varying(_) => (0..horizon).collect(),
    };
    (request, MechanismPeriodMap { before: axis.clone(), after: axis })
}

#[test]
fn flagship_mechanism_explanation_is_complete_under_a_supported_design_and_names_the_next_design_otherwise()
 {
    use antecedent::analysis::joint_panel::{
        ChangeComponent, JointPanelExplanationRequest, ReferenceObligation,
        declare_joint_panel_exogenous_context,
    };
    let ctx = cached_context(20_261_031);
    let bit = |component: ChangeComponent| match component {
        ChangeComponent::Exposure => 0,
        ChangeComponent::Saturation => 1,
        ChangeComponent::Population => 2,
        ChangeComponent::Mechanism => 3,
    };

    // Supported: exact whole-program assignment identifies every reference world.
    let data = fixture::whole_program_source();
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 4 });
    request.target = declare_joint_panel_exogenous_context(&request.target);
    request.assignment = Some(fixture::whole_program_randomization());
    let supported = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    assert!(supported.inspect().identification.is_identified_claim());
    let result = supported.estimate(&ctx).unwrap();
    let (decomposition, periods) = terminal_decomposition(&result);
    let adaptive =
        result.attempt_design_adaptive_reference(&decomposition, &periods, None, &ctx).unwrap();
    assert!(adaptive.is_complete(), "{:?}", adaptive.next_designs);
    assert!(adaptive.next_designs.is_empty() && adaptive.unidentified_components.is_empty());
    assert!(adaptive.coarser.is_none(), "a complete finer claim needs no coarser fallback");
    assert_eq!(adaptive.identified_steps.len(), 32);
    let total = adaptive.requested.decomposition.point.total;
    let exposure = adaptive
        .identified_steps
        .iter()
        .find(|step| step.component == ChangeComponent::Exposure && step.context_mask == 0)
        .unwrap();
    assert!((exposure.point - total).abs() < 1e-10, "only exposure differs between endpoints");
    // An independent consumer rebuilds every executed step from the stored fit alone.
    let bytes = result.export(&ctx).unwrap();
    let historical =
        antecedent_io::joint_panel_artifact::consume_panel_execution(&bytes, &ctx).unwrap();
    let replay = historical.program().decompose_reference(&decomposition, &periods, &ctx).unwrap();
    for step in &adaptive.identified_steps {
        let stepped = usize::from(step.context_mask | (1 << bit(step.component)));
        let context = usize::from(step.context_mask);
        assert_eq!(step.point, replay.point.hybrids[stepped] - replay.point.hybrids[context]);
        assert_eq!(step.draws.len(), replay.draws.len());
        for (actual, stored) in step.draws.iter().zip(&replay.draws) {
            assert_eq!(
                *actual,
                stored.as_ref().map(|row| row.hybrids[stepped] - row.hybrids[context])
            );
        }
    }
    assert!(!historical.grants_current_authority());

    // Unsupported: period randomization identifies nothing for the whole program. The finer
    // decomposition is still computed as model-conditional, no step is called identified,
    // the missing design is named, and the coarser total contrast is actually executed.
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 4 });
    let assignment = fixture::period_randomization();
    assignment.validate().unwrap();
    request.assignment = Some(assignment);
    request.target = declare_joint_panel_exogenous_context(&request.target);
    let unsupported = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    assert!(!unsupported.inspect().identification.is_identified_claim());
    let result = unsupported.estimate(&ctx).unwrap();
    let (decomposition, periods) = terminal_decomposition(&result);
    let worlds = &result.request().worlds;
    let fallback = JointPanelExplanationRequest {
        explanation: antecedent_estimate::joint_panel::time_varying::PanelExplanation::Mechanism,
        world: worlds[3].world,
        reference_world: worlds[0].world,
        before: (0, 5),
        after: (18, 23),
    };
    let adaptive = result
        .attempt_design_adaptive_reference(&decomposition, &periods, Some(fallback), &ctx)
        .unwrap();
    assert!(!adaptive.is_complete());
    assert!(adaptive.identified_steps.is_empty());
    assert_eq!(adaptive.unidentified_components.len(), 4);
    assert!(
        adaptive
            .next_designs
            .iter()
            .any(|step| step.obligation == ReferenceObligation::MissingIdentificationProof)
    );
    assert!(adaptive.requested.decomposition.point.total.is_finite());
    let coarser = adaptive.coarser.as_ref().expect("the next coarser contrast is executed");
    assert!(coarser.point.is_finite());
    assert!(coarser.require_unique_attribution().is_err());
    assert_eq!(unsupported.fits_performed(), 1);
}

#[test]
fn temporal_panel_public_preparation_retains_attempts_and_reuses_selected_mechanism() {
    use antecedent_estimate::joint_panel::time_varying::{PanelMeanParameter, TemporalPanelSpec};
    let ctx = cached_context(20_261_021);
    let prepared = {
        let data = fixture::source(false);
        let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
        request.temporal = Some(TemporalPanelSpec {
            varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
            change_points: vec![8, 12, 16],
            minimum_segment: 4,
            weak_change_gap: 4.0,
            smooth_knots: vec![0, 8, 16, 23],
            smoothing: vec![1.0],
            path_prior_sd: 1.0,
        });
        Study::prepare_joint_panel(data, request, &ctx).unwrap()
    };
    assert_eq!(prepared.fits_performed(), 0);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("retained temporal program") };
    assert_eq!(fit.point.parameters.len(), 24);
    assert!(fit.attempts.iter().any(|attempt| attempt.converged));
    assert_eq!(fit.calibration, "unmeasured");
    assert_eq!(fit.claim, "model_conditional_temporal_mechanism_only");
    assert!(!prepared.inspect().identification.is_identified_claim());
    assert!(
        result
            .point
            .iter()
            .all(|world| world.means.len() == 24
                && world.means.iter().all(|value| value.is_finite()))
    );
    assert_eq!(prepared.estimate(&ctx).unwrap().point, result.point);
    assert_eq!(prepared.fits_performed(), 1);
}

#[test]
fn temporal_panel_bootstrap_reruns_mechanism_selection_per_replicate() {
    use antecedent_estimate::joint_panel::time_varying::{PanelMeanParameter, TemporalPanelSpec};
    let ctx = cached_context(20_261_022);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 6 });
    request.temporal = Some(TemporalPanelSpec {
        varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    });
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("retained temporal program") };
    let selected: usize = fit.selection_counts.iter().map(|(_, count)| count).sum();
    let successful = fit.draws.iter().flatten().count();
    assert!(successful > 0, "bootstrap must retain successful replicates");
    assert_eq!(selected, successful, "every retained replicate records its own selected model");
    exercise_reference_decomposition(&result, &ctx);
    assert_streamed_projection_matches(&result, &ctx);
}

fn assert_fresh_panel_replay(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    let bytes = result.export(ctx).unwrap();
    assert_eq!(result.consume_current(&bytes, ctx).unwrap().point, result.point);
    let directory =
        std::env::temp_dir().join(format!("antecedent-panel-replay-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("execution.bin"), bytes).unwrap();
    let bits = |responses: &[antecedent_estimate::joint_panel::PanelResponse]| {
        responses
            .iter()
            .map(|world| world.means.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let expected = (
        bits(&result.point),
        result.draws.iter().map(|row| row.as_deref().map(bits)).collect::<Vec<_>>(),
    );
    std::fs::write(directory.join("expected.json"), serde_json::to_vec(&expected).unwrap())
        .unwrap();
    // The child must select this very test; its libtest name carries the module path.
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name = format!("{prefix}fresh_process_panel_artifact_consumer");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture"])
        .env("ANTECEDENT_PANEL_REPLAY_DIRECTORY", &directory)
        .output()
        .unwrap();
    let status = output.status;
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        directory.join("consumer-ran").exists(),
        "fresh process must select and execute its consumer test"
    );
    let _ = std::fs::remove_dir_all(directory);
    assert!(
        status.success(),
        "fresh consumer must replay frozen shared draws without the builder or a fit"
    );
}

#[test]
fn fresh_process_panel_artifact_consumer() {
    let Some(directory) = std::env::var_os("ANTECEDENT_PANEL_REPLAY_DIRECTORY") else { return };
    let directory = std::path::PathBuf::from(directory);
    let expected: (Vec<Vec<u64>>, Vec<Option<Vec<Vec<u64>>>>) =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let replay = antecedent_io::joint_panel_artifact::consume_panel_execution(
        &std::fs::read(directory.join("execution.bin")).unwrap(),
        &cached_context(20_261_021),
    )
    .unwrap();
    let bits = |responses: &[antecedent_estimate::joint_panel::PanelResponse]| {
        responses
            .iter()
            .map(|world| world.means.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(replay.point()), expected.0);
    assert_eq!(
        replay.draws().iter().map(|row| row.as_deref().map(bits)).collect::<Vec<_>>(),
        expected.1
    );
    assert!(!replay.grants_current_authority());
    std::fs::write(directory.join("consumer-ran"), b"replayed").unwrap();
}

#[test]
fn actual_fit_cache_invalidates_changed_history_initialization_seed_and_kernel_policy() {
    // This deliberately failed one-step fit isolates dependency behavior; successful
    // inference is required separately by the representative public workflow tests.
    let ctx = cached_context(51);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 2 });
    request.options.optimizer.max_iterations = 1;
    request.options.optimizer.gradient_tolerance = 1e-20;
    let options = request.options.clone();
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let first = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &first.fit else { panic!("invariant mechanism") };
    assert!(!fit.diagnostics.optimizer_converged);
    exercise_coarser_explanation(&first, &ctx);
    let mut changed_initial = options.clone();
    changed_initial.initial = Some(fit.point.parameters.clone());
    let initialized = prepared.refit_data(fixture::source(false), changed_initial, &ctx).unwrap();
    assert_ne!(initialized.inspect().options_identity, prepared.inspect().options_identity);
    let initialized_result = initialized.estimate(&ctx).unwrap();
    assert_ne!(initialized_result.dependency_identity(), first.dependency_identity());
    assert_eq!(prepared.fits_performed(), 2);
    let changed = prepared.refit_data(fixture::source_changed_history(), options, &ctx).unwrap();
    assert_ne!(changed.inspect().source_identity, prepared.inspect().source_identity);
    assert_ne!(changed.estimate(&ctx).unwrap().dependency_identity(), first.dependency_identity());
    assert_eq!(changed.fits_performed(), 3);
    let next_seed = cached_context(52);
    let seed_result = changed.estimate(&next_seed).unwrap();
    assert_eq!(changed.fits_performed(), 4);
    let mut next_kernel = cached_context(52);
    next_kernel.kernel_policy = antecedent_core::execution::KernelPolicy::default_policy();
    assert_ne!(
        changed.estimate(&next_kernel).unwrap().dependency_identity(),
        seed_result.dependency_identity()
    );
    assert_eq!(changed.fits_performed(), 5);
}

#[test]
fn programmable_feature_constant_semantics_and_forbidden_reads_are_checked_before_fit() {
    use antecedent_core::{ValueExpr, VariableId};
    use antecedent_estimate::joint_panel::PanelInteractionSpec;
    let ctx = cached_context(91);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    // A declared constant feature is legal model syntax, even though it duplicates
    // the intercept and therefore cannot establish a unique frequentist coefficient.
    request.interactions = vec![
        PanelInteractionSpec::from_saturated_features("constant", &ValueExpr::constant(2.0))
            .unwrap(),
    ];
    let prepared = Study::prepare_joint_panel(data.clone(), request.clone(), &ctx).unwrap();
    assert_eq!(prepared.fits_performed(), 0);
    assert!(!prepared.inspect().identification.is_identified_claim());
    for (position, expression) in [
        ValueExpr::var(VariableId::from_raw(99)),
        ValueExpr::generated(),
        ValueExpr::lag(VariableId::from_raw(0), 1),
    ]
    .into_iter()
    .enumerate()
    {
        request.interactions =
            vec![PanelInteractionSpec::from_saturated_features("forbidden", &expression).unwrap()];
        let refusal = Study::prepare_joint_panel(data.clone(), request.clone(), &ctx)
            .err()
            .expect("a feature that reads outside the lagged saturated channels is refused");
        let text = refusal.to_string();
        assert!(
            text.starts_with("joint panel invalid input: invalid programmable feature: "),
            "expression {position}: {text}"
        );
        if position == 0 {
            assert!(text.contains("intervention_program.variable_not_in_model"), "{text}");
            assert!(text.contains("column expression reads absent variable V99"), "{text}");
        }
    }
}

#[test]
fn recoverable_callback_feature_is_refused_by_public_panel_compilation() {
    use antecedent_estimate::joint_panel::{PanelFeatureInputs, PanelInteractionSpec};
    use antecedent_io::query_wire::{CallbackOutputWire, CallbackRecordWire, ValueExprWire};
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.interactions = vec![PanelInteractionSpec {
        id: "callback".into(),
        inputs: PanelFeatureInputs::LaggedSaturatedChannels,
        expression: ValueExprWire::Callback {
            record: CallbackRecordWire {
                name: "refused_callback".into(),
                inputs: vec![],
                output: CallbackOutputWire::Real,
                source_sha256: "0".repeat(64),
                entry_point: "unavailable".into(),
                dependencies: vec![],
            },
            args: vec![],
        },
    }];
    let refusal = Study::prepare_joint_panel(data, request, &cached_context(1))
        .err()
        .expect("a callback feature cannot be compiled into a retained panel");
    assert_eq!(
        refusal.to_string(),
        "joint panel compilation refused: panel features require callback-free common expressions"
    );
}

fn cached_context(seed: u64) -> ExecutionContext {
    // Four threads let the sampler's chains overlap; the draws do not depend on the budget.
    let mut ctx = ExecutionContext::for_tests(seed)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(4, u64::MAX));
    ctx.parallelism =
        antecedent_core::Parallelism::bounded(antecedent_core::NonZeroThreadCount::new(4).unwrap());
    ctx.cache_policy = antecedent_core::execution::CachePolicy::enabled(None);
    ctx
}
#[test]
fn explicitly_disabled_cache_repeats_the_numerical_fit() {
    let ctx = ExecutionContext::for_tests(3);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.options.optimizer.max_iterations = 1;
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    prepared.estimate(&ctx).unwrap();
    prepared.estimate(&ctx).unwrap();
    assert_eq!(prepared.fits_performed(), 2);
}

fn exercise_coarser_explanation(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    use antecedent::analysis::joint_panel::{
        JointPanelExplanationRequest, JointPanelScientificScope,
    };
    use antecedent_design::{
        decision_contract::{
            ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
            UtilityExpr,
        },
        scientific_source::ScientificSourcePayload,
    };
    use antecedent_estimate::joint_panel::time_varying::PanelExplanation;
    let attempt = result
        .attempt_explanation(
            JointPanelExplanationRequest {
                explanation: PanelExplanation::Mechanism,
                world: antecedent_core::ScientificWorldId::from_raw(3),
                reference_world: antecedent_core::ScientificWorldId::from_raw(0),
                before: (0, 1),
                after: (22, 23),
            },
            ctx,
        )
        .unwrap();
    let difference = |world: usize| {
        (result.point[world].means[22] + result.point[world].means[23]
            - result.point[world].means[0]
            - result.point[world].means[1])
            / 2.0
    };
    assert!((attempt.point - (difference(3) - difference(0))).abs() < 1e-12);
    assert_eq!(attempt.draws.len(), result.draws.len());
    assert!(attempt.draws.iter().all(Option::is_none));
    assert_eq!(attempt.parent_target, result.request().target.identity());
    assert_ne!(attempt.source.target().identity(), attempt.parent_target);
    assert!(attempt.require_unique_attribution().is_err());
    assert!(attempt.source.authorize(JointPanelScientificScope::IdentifiedCausal).is_err());
    let derived = attempt.compiled_response();
    assert!(derived.request().worlds.iter().any(|world| world.horizon() == 1));
    assert!(derived.request().worlds.iter().any(|world| world.horizon() == 24));
    assert_eq!(
        derived.request().worlds[0].program.scenarios[0].history,
        result.request().worlds[0].program.scenarios[0].history
    );
    let ScientificSourcePayload::Means(means) = attempt.source.as_numerical_source() else {
        panic!("coarser mean source")
    };
    let quantity = means.coordinates[0].clone();
    let contract = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Policy,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "respond".into(),
                kind: ActionKind::Policy,
                inputs: vec![quantity.clone()],
                // Common baseline plus the response contrast preserves the ranking.
                utility: UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            },
        ],
        utility_units: quantity.units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: quantity.population_id.clone(),
        horizon: quantity.horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let decision =
        attempt.source.evaluate(&contract, JointPanelScientificScope::ModelConditional).unwrap();
    assert_eq!(decision.target_identity, attempt.source.target().identity());
    assert_eq!(result.prepared().fits_performed(), 1);
}

#[test]
fn static_panel_coarser_prefix_keeps_the_original_temporal_clock() {
    use antecedent::analysis::joint_panel::{JointPanelFunctional, joint_panel_target};
    use antecedent_core::{InferenceSelection, ScientificWorldId, TargetForm};
    let ctx = cached_context(123);
    let data = fixture::source_static();
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 2 });
    request.lag_order = 0;
    request.options.optimizer.max_iterations = 1;
    request.options.optimizer.gradient_tolerance = 1e-20;
    request.target = joint_panel_target(
        data.schema(),
        &request.columns,
        0,
        &request.worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    exercise_coarser_explanation(&result, &ctx);
    let attempt = result
        .attempt_explanation(
            antecedent::analysis::joint_panel::JointPanelExplanationRequest {
                explanation:
                    antecedent_estimate::joint_panel::time_varying::PanelExplanation::Mechanism,
                world: ScientificWorldId::from_raw(3),
                reference_world: ScientificWorldId::from_raw(0),
                before: (0, 1),
                after: (22, 23),
            },
            &ctx,
        )
        .unwrap();
    let TargetForm::Contrast(contrast) = attempt.source.target().form() else {
        panic!("derived Linear contrast")
    };
    assert!(contrast.worlds().worlds().iter().all(|world| !world.time().is_static()));
    let prefix_means = attempt.compiled_response().numerical_mean_source().unwrap();
    assert_eq!(prefix_means.coordinates[0].horizon, 1);
    assert!(prefix_means.coordinates.iter().all(|quantity| quantity.horizon > 0));
    let incompatible = contrast
        .worlds()
        .worlds()
        .iter()
        .map(|world| {
            let time = antecedent_core::TimeFrame::temporal(
                world.time().horizon_steps().unwrap(),
                world.time().history().unwrap(),
                antecedent_core::InitialLaw::named("unrelated_history_law").unwrap(),
            )
            .unwrap();
            antecedent_core::InterventionWorld::new(
                world.id(),
                world.action().clone(),
                world.is_reference(),
                world.binding().clone(),
                time,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let invalid = antecedent_core::ScientificTarget::new(
        TargetForm::Contrast(
            antecedent_core::InterventionContrast::linear(
                antecedent_core::WorldCollection::new(incompatible).unwrap(),
                contrast.terms().iter().map(|term| (term.world, term.coefficient)),
                contrast.reference(),
            )
            .unwrap(),
        ),
        attempt.source.target().assumptions().clone(),
        attempt.source.target().inference(),
    );
    let derived = attempt.compiled_response();
    assert!(
        derived.prepared().retarget(derived.request().worlds.clone(), invalid, &ctx).is_err(),
        "an explicit temporal h1 clock must not accept an unrelated initial-law identity"
    );
    assert_eq!(prepared.fits_performed(), 1);
}

fn broader_family() -> antecedent_estimate::joint_panel::PanelFamily {
    use antecedent_estimate::joint_panel::{PanelAdstock, PanelFamily, PanelSaturation};
    PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelFamily::per_group_saturation(4),
        spatial: None,
    }
}

#[test]
fn broader_families_inherit_the_randomized_design_claim_and_leave_confounded_panels_unlicensed() {
    use antecedent::analysis::joint_panel::{
        JointPanelDesignOutcome, declare_joint_panel_exogenous_context,
    };
    use antecedent_estimate::joint_panel::PanelOptimizerOptions;
    let ctx = cached_context(20_261_021);
    let bounded = PanelOptimizerOptions { max_iterations: 600, gradient_tolerance: 1e-2 };
    let prepare = |family: Option<antecedent_estimate::joint_panel::PanelFamily>| {
        let data = fixture::whole_program_source();
        let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
        request.target = declare_joint_panel_exogenous_context(&request.target);
        request.assignment = Some(fixture::whole_program_randomization());
        request.options.optimizer = bounded;
        if let Some(family) = family {
            request.family = family;
        }
        Study::prepare_joint_panel(data, request, &ctx).unwrap()
    };
    let reviewed = prepare(None);
    let broader = prepare(Some(broader_family()));
    // The design-based claim concerns the declared worlds and assignment, not the mechanism
    // family: the same attempt identifies the same worlds under either family.
    for prepared in [&reviewed, &broader] {
        let inspection = prepared.inspect();
        assert!(inspection.identification.is_identified_claim());
        let attempt = inspection
            .design_attempts
            .iter()
            .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
            .unwrap();
        assert!(matches!(attempt.outcome, JointPanelDesignOutcome::Identified));
        assert!(attempt.worlds.iter().all(|world| world.identified));
    }
    assert_ne!(reviewed.inspect().source_identity, broader.inspect().source_identity);
    let result = broader.estimate(&ctx).unwrap();
    result.require_identified().unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("invariant mechanism") };
    let family = &fit.point.parameters.family;
    assert_eq!((family.hill_shape.len(), family.class_half_saturation.len()), (2, 3));
    for channel in &fit.point.parameters.channels {
        assert!((channel.lag[1] / channel.lag[0] - channel.lag[2] / channel.lag[1]).abs() < 1e-8);
    }
    assert!(result.point.iter().all(|world| world.means.iter().all(|value| value.is_finite())));
    assert_eq!(broader.estimate(&ctx).unwrap().point, result.point);
    assert_eq!(broader.fits_performed(), 1);
    // The same family on the confounded panel (channels driven by an unobserved context, moving
    // separately) still earns no causal license from its fit.
    let data = fixture::source_confounded_varied();
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.family = broader_family();
    request.options.optimizer = bounded;
    let confounded = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    assert!(!confounded.inspect().identification.is_identified_claim());
    let fitted = confounded.estimate(&ctx).unwrap();
    assert!(fitted.require_identified().is_err());
}

#[test]
fn temporal_panel_fits_a_broader_family_through_the_same_selection_procedure() {
    use antecedent_estimate::joint_panel::{
        PanelOptimizerOptions,
        time_varying::{PanelMeanParameter, TemporalPanelSpec},
    };
    let ctx = cached_context(20_261_023);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.family = broader_family();
    request.options.optimizer =
        PanelOptimizerOptions { max_iterations: 600, gradient_tolerance: 1e-2 };
    request.temporal = Some(TemporalPanelSpec {
        // A shape change is a temporal mechanism of the Hill family; the shared lag and
        // class structure stay fixed across the change.
        varying: vec![PanelMeanParameter::HillShape { channel: "0".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    });
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    assert_eq!(prepared.fits_performed(), 0);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("retained temporal program") };
    assert_eq!(fit.point.parameters.len(), 24);
    assert!(fit.attempts.iter().any(|attempt| attempt.converged));
    assert!(fit.point.parameters.iter().all(|parameters| {
        parameters.family.hill_shape.len() == 2
            && parameters.family.class_half_saturation.len() == 3
    }));
    // Only the declared coordinate may move along the path.
    let first = &fit.point.parameters[0];
    assert!(
        fit.point.parameters.iter().all(|p| p.family.hill_shape[1] == first.family.hill_shape[1]
            && p.family.class_half_saturation == first.family.class_half_saturation
            && p.channels == first.channels)
    );
    assert_eq!(fit.claim, "model_conditional_temporal_mechanism_only");
    assert!(!prepared.inspect().identification.is_identified_claim());
    assert!(result.point.iter().all(|world| world.means.len() == 24));
}

fn temporal_frame_spec() -> antecedent_estimate::joint_panel::time_varying::TemporalPanelSpec {
    use antecedent_estimate::joint_panel::time_varying::{PanelMeanParameter, TemporalPanelSpec};
    TemporalPanelSpec {
        varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    }
}

/// Reference worlds chosen on the time-varying fit: the early and the late fitted mechanism
/// periods stand for the before and after worlds of one unchanged program. The component
/// blocks are checked against arithmetic that does not use the decomposition owner: the
/// retained world responses, the efficiency of the exact orderings, antisymmetry under
/// swapping the endpoints, and invariance of every hybrid that does not take the changed block.
fn exercise_temporal_reference_worlds(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    ctx: &ExecutionContext,
) {
    use antecedent::analysis::joint_panel::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
    };
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("time-varying fit") };
    let worlds = &result.request().worlds;
    let horizon = worlds[0].horizon();
    let groups = result.prepared().model_data().groups.len();
    let weights = vec![1.0 / groups as f64; groups];
    let endpoint = |index: usize| DecompositionEndpoint {
        program: worlds[index].program.clone(),
        population_weights: weights.clone(),
    };
    let mut terminal = vec![0.0; horizon];
    terminal[horizon - 1] = 1.0;
    let order = [
        ChangeComponent::Exposure,
        ChangeComponent::Saturation,
        ChangeComponent::Population,
        ChangeComponent::Mechanism,
    ];
    let request = |before: usize, after: usize| PanelDecompositionRequest {
        before: endpoint(before),
        after: endpoint(after),
        time_weights: terminal.clone(),
        order,
    };
    let identity: Vec<usize> = (0..horizon).collect();
    // The late-period mechanism replaces the early one at every response period.
    let early = vec![0; horizon];
    let late = vec![horizon - 1; horizon];
    let changed = MechanismPeriodMap { before: early.clone(), after: late.clone() };
    let same = MechanismPeriodMap { before: identity.clone(), after: identity.clone() };
    let forward = result.decompose_reference(&request(0, 3), &changed, ctx).unwrap();
    let swapped = result
        .decompose_reference(
            &request(3, 0),
            &MechanismPeriodMap { before: late.clone(), after: early.clone() },
            ctx,
        )
        .unwrap();
    let held = result.decompose_reference(&request(0, 3), &same, ctx).unwrap();
    assert_eq!(forward.draws.len(), result.draws.len(), "no slot is dropped or renumbered");
    let check = |a: &antecedent::analysis::joint_panel::ReferenceDecomposition,
                 b: &antecedent::analysis::joint_panel::ReferenceDecomposition| {
        assert!((a.shapley.iter().sum::<f64>() - a.total).abs() < 1e-9, "efficiency");
        assert!((a.total + b.total).abs() < 1e-9, "swapping the endpoints negates the total");
        for j in 0..4 {
            assert!((a.shapley[j] + b.shapley[j]).abs() < 1e-9, "antisymmetry");
        }
        // Complementing every block of a hybrid is the same world seen from the other endpoint.
        for mask in 0..16usize {
            assert!((a.hybrids[mask] - b.hybrids[mask ^ 15]).abs() < 1e-9, "hybrid {mask}");
        }
    };
    check(&forward.point, &swapped.point);
    for (f, s) in forward.draws.iter().zip(&swapped.draws) {
        assert_eq!(f.is_some(), s.is_some());
        if let (Some(f), Some(s)) = (f, s) {
            check(f, s);
        }
    }
    // Independent of the owner: the fit's own point worlds. With the identity map and an
    // exposure-only endpoint change the all-after hybrid is the fitted terminal response.
    let terminal_of = |world: usize| *result.point[world].means.last().unwrap();
    assert!((held.point.hybrids[0] - terminal_of(0)).abs() < 1e-9);
    assert!((held.point.hybrids[15] - terminal_of(3)).abs() < 1e-9);
    // The change block moves the response exactly when the fitted early and late periods differ.
    let early_p = &fit.point.parameters[0];
    let late_p = &fit.point.parameters[horizon - 1];
    let moved = format!("{early_p:?}") != format!("{late_p:?}");
    let mechanism_effect = forward.point.hybrids[8] - forward.point.hybrids[0];
    if !moved {
        assert!(mechanism_effect.abs() < 1e-9, "an unchanged mechanism has no change effect");
    }
    // The change-point/smoothing selection is carried through the decomposed law.
    let split = result.decomposition_selection(&forward).unwrap().unwrap();
    let complete = forward.draws.iter().flatten().count();
    assert_eq!((split.requested, split.complete), (forward.draws.len(), complete));
    assert_eq!(split.specifications.iter().map(|s| s.slots).sum::<usize>(), complete);
    assert!((split.specifications.iter().map(|s| s.mass).sum::<f64>() - 1.0).abs() < 1e-12);
    // This design earns no proof of the mechanism block: the finer claim is executed as
    // model-conditional and names the next design instead of calling any step identified.
    let adaptive =
        result.attempt_design_adaptive_reference(&request(0, 3), &changed, None, ctx).unwrap();
    assert!(!adaptive.is_complete() && adaptive.identified_steps.is_empty());
    assert!(adaptive.next_designs.iter().any(|step| {
        step.obligation
            == antecedent::analysis::joint_panel::ReferenceObligation::MissingIdentificationProof
    }));
}

#[test]
fn temporal_frequentist_frame_decomposes_on_reference_worlds_and_translates_experiments() {
    let ctx = cached_context(20_261_041);
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 8 });
    request.temporal = Some(temporal_frame_spec());
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("retained temporal program") };
    assert_eq!(fit.calibration, "unmeasured");
    assert!(result.draws.iter().flatten().count() > 0);
    assert!(!prepared.inspect().identification.is_identified_claim());
    exercise_reference_decomposition(&result, &ctx);
    exercise_temporal_reference_worlds(&result, &ctx);
    experiment_chain::exercise_frequentist_boundary(&prepared, &result, &ctx);
}

#[test]
fn temporal_bayesian_frame_decomposes_on_reference_worlds_and_translates_experiments() {
    let ctx = cached_context(20_261_042);
    let data = fixture::source(false);
    // Effort and thresholds are stated for a deliberately small debug-time panel; they do not
    // change the production defaults, and the fit keeps its own convergence verdict.
    let sampler = antecedent_estimate::joint_panel::PanelSamplingOptions {
        chains: 2,
        warmup: 512,
        draws: 512,
        // This routing fixture must retain enough movement for both CDF-tail
        // indicators. Keep the diagnostic thresholds, and increase effort
        // rather than treating a zero tail ESS as converged.
        proposal_scale: 0.01,
        max_rhat: 5.0,
        min_ess: 1.0,
    };
    let mut request = fixture::request(
        &data,
        false,
        PanelInference::Bayesian { prior: PanelPrior::default(), sampler },
    );
    request.temporal = Some(temporal_frame_spec());
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("retained temporal program") };
    assert_eq!(result.draws.len(), 1024);
    assert!(fit.posterior_diagnostics.is_some());
    assert_eq!(
        fit.posterior_converged,
        Some(true),
        "experimental updating requires the declared diagnostics: {:?}",
        fit.posterior_diagnostics
    );
    assert!(!prepared.inspect().identification.is_identified_claim());
    exercise_reference_decomposition(&result, &ctx);
    exercise_temporal_reference_worlds(&result, &ctx);
    experiment_chain::exercise(&prepared, &result, &ctx);
}
