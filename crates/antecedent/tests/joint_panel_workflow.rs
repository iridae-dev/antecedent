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
    assert_fresh_panel_replay(&result, &ctx);
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
    experiment_chain::exercise(&prepared, &result, &ctx);
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
fn confounded_panel_does_not_acquire_a_causal_license_from_a_finite_mechanism() {
    let ctx = cached_context(20_261_021);
    let data = fixture::source(true);
    let request = fixture::request(&data, true, PanelInference::Frequentist { draws: 0 });
    let prepared = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let inspection = prepared.inspect();
    assert_eq!(inspection.fits_performed, 0);
    assert!(!inspection.identification.is_identified_claim());
    assert!(!inspection.design_attempts.is_empty());
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
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "joint_panel_workflow::fresh_process_panel_artifact_consumer",
            "--nocapture",
        ])
        .env("ANTECEDENT_PANEL_REPLAY_DIRECTORY", &directory)
        .status()
        .unwrap();
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
    for expression in [
        ValueExpr::var(VariableId::from_raw(99)),
        ValueExpr::generated(),
        ValueExpr::lag(VariableId::from_raw(0), 1),
    ] {
        request.interactions =
            vec![PanelInteractionSpec::from_saturated_features("forbidden", &expression).unwrap()];
        assert!(Study::prepare_joint_panel(data.clone(), request.clone(), &ctx).is_err());
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
    assert!(Study::prepare_joint_panel(data, request, &cached_context(1)).is_err());
}

fn cached_context(seed: u64) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(seed);
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
