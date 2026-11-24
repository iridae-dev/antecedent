// SPDX-License-Identifier: MIT OR Apache-2.0
//! The joint-panel engine functions the facade re-exports are the same computation as the
//! compiled joint-panel routes that call them.
//!
//! Each test prepares a joint panel (or a native joint design) through the compiler, runs the
//! compiled route, and then calls the engine function directly on exactly the inputs the route
//! hands it: the source the handle retains, the fitted point and draws the result carries, the
//! world programs of the request and the weights of the projected functional. The values are
//! compared bit for bit (through `to_bits` or the round-trip `Debug` form of every float).
//!
//! - `JointPanelResponse`: `fit_panel` and `evaluate_panel_law` are compared in
//!   `joint_panel_compiled.rs`; here `evaluate_panel_programs`, `stream_panel_law`,
//!   `project_panel_law`, `project_panel_responses`, `decompose_invariant_law`,
//!   `decompose_invariant_law_streamed`, `decompose_panel_law`, `decompose_values`,
//!   `decompose_weighted_hybrid_law` and `panel_restriction_diagnostics`;
//! - `JointPanelTimeVarying`: `fit_temporal_panel`, `evaluate_temporal_law`,
//!   `evaluate_temporal_programs`, `stream_temporal_law`, `project_temporal_law`,
//!   `decompose_temporal_law`, `decompose_temporal_law_streamed` and `decomposition_selection`;
//! - `JointDesign`: `JointDesignRecipe::execute` (the moment law `execute_native` falls back
//!   to). `execute_native` is compared in `joint_panel_compiled.rs`; `decompose_hybrid_law`
//!   and `propagate_hybrid_covariance` on a design law are compared in the crate's
//!   `explanation::scientific` twin tests, whose fixture is crate-private.
//!
//! The routes run the engines with the dispatch-scoped context of the lowered program; that
//! context bounds admission and cancellation and does not change any number. The engine
//! functions carry no calibration key of their own: every route gates on, and records its
//! coverage under, the support unit `joint_response.interval_readings`, which the routes'
//! own tests check.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "test file: fixtures unwrap and bits are compared exactly"
)]

#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use std::fmt::Debug;

use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
use antecedent::analysis::joint_panel::{
    ChangeComponent, DecompositionEndpoint, JointPanelDesignOutcome, JointPanelFit,
    JointPanelResponseFunctional, JointPanelResult, JointPanelState, MechanismPeriodMap,
    PanelDecompositionRequest, ReferenceComponentBindings,
    declare_joint_panel_gaussian_restriction,
};
use antecedent::planner::PlanRoute;
use antecedent::{PreparedStudy, Study};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::CachePolicy;
use antecedent_design::scientific_source::ScientificConsumptionScope;
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use antecedent_estimate::joint_design::randomized::{
    JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior,
};
use antecedent_estimate::joint_panel::time_varying::{PanelMeanParameter, TemporalPanelSpec};
use antecedent_estimate::joint_panel::{self as engine, PanelInference, PanelResponse};
use antecedent_identify::panel_unrolled::{UnrolledPanelGraph, UnrolledPanelSpec};

/// A cache-enabled context, so the projections and decompositions reuse the result's fit.
fn context() -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(20_261_124);
    ctx.cache_policy = CachePolicy::enabled(None);
    ctx
}

/// The round-trip `Debug` form: equal strings mean equal bits for every finite float.
fn same<T: Debug + ?Sized>(label: &str, direct: &T, routed: &T) {
    assert_eq!(format!("{direct:?}"), format!("{routed:?}"), "{label}");
}

fn bits(rows: &[PanelResponse]) -> Vec<Vec<u64>> {
    rows.iter().map(|row| row.means.iter().map(|value| value.to_bits()).collect()).collect()
}

fn law_bits(rows: &[Option<Vec<PanelResponse>>]) -> Vec<Option<Vec<Vec<u64>>>> {
    rows.iter().map(|row| row.as_deref().map(bits)).collect()
}

/// The piecewise and smooth time-varying declaration over the interaction coefficient.
fn temporal_spec() -> TemporalPanelSpec {
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

/// The reference panel compiled through the main compiler, stationary or time-varying.
fn prepare(temporal: bool, ctx: &ExecutionContext) -> PreparedStudy<JointPanelState> {
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 8 });
    if temporal {
        request.temporal = Some(temporal_spec());
    }
    let prepared = Study::prepare_joint_panel(data, request, ctx).unwrap();
    let route =
        if temporal { PlanRoute::JointPanelTimeVarying } else { PlanRoute::JointPanelResponse };
    assert_eq!(prepared.plan_view().route(), route);
    prepared
}

fn programs(result: &JointPanelResult) -> Vec<engine::PanelProgram> {
    result.request().worlds.iter().map(|world| world.program.clone()).collect()
}

/// Reference minus joint world over the whole horizon, with the result's mechanism periods.
fn decomposition_request(
    result: &JointPanelResult,
) -> (PanelDecompositionRequest, MechanismPeriodMap) {
    let worlds = &result.request().worlds;
    let horizon = worlds[0].horizon();
    let groups = result.prepared().model_data().groups.len();
    let weights = vec![1.0 / groups as f64; groups];
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
    let periods: Vec<usize> = match &result.fit {
        JointPanelFit::Invariant(_) => vec![0; horizon],
        JointPanelFit::Varying(_) => (0..horizon).collect(),
    };
    (request, MechanismPeriodMap { before: periods.clone(), after: periods })
}

/// A tile consumer the streaming engines accept.
type TileSink<'a> =
    dyn FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), engine::PanelError> + Send + 'a;

/// Tiles of a streamed law, collected back into the full draw order.
fn collect_tiles(
    stream: impl FnOnce(&mut TileSink<'_>) -> engine::PanelStreamReceipt,
) -> (engine::PanelStreamReceipt, Vec<Option<Vec<PanelResponse>>>) {
    let mut rows = Vec::new();
    let receipt = stream(&mut |first: usize, tile: &[Option<Vec<PanelResponse>>]| {
        assert_eq!(first, rows.len(), "tiles arrive in draw order");
        rows.extend_from_slice(tile);
        Ok(())
    });
    (receipt, rows)
}

// --- JointPanelResponse ----------------------------------------------------------------------

#[test]
fn stationary_point_programs_equal_the_response_route() {
    let ctx = context();
    let prepared = prepare(false, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("stationary fit") };
    let direct = engine::evaluate_panel_programs(
        prepared.model_data(),
        &fit.point,
        &programs(&result),
        &ctx,
    )
    .unwrap();
    assert_eq!(bits(&direct), bits(&result.point));
}

#[test]
fn stationary_streamed_law_equals_the_response_route() {
    let ctx = context();
    let prepared = prepare(false, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("stationary fit") };
    let programs = programs(&result);
    for tile in [1, 3, result.draws.len().max(1)] {
        let (receipt, rows) = collect_tiles(|sink| {
            engine::stream_panel_law(
                prepared.model_data(),
                &fit.point,
                &fit.draws,
                &programs,
                tile,
                &ctx,
                |first, rows| sink(first, rows),
            )
            .unwrap()
        });
        assert_eq!(bits(&receipt.point), bits(&result.point), "tile {tile}: point");
        assert_eq!(law_bits(&rows), law_bits(&result.draws), "tile {tile}: draws");
    }
}

#[test]
fn stationary_projected_law_equals_the_response_route() {
    let ctx = context();
    let prepared = prepare(false, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("stationary fit") };
    let functional = JointPanelResponseFunctional::WindowChange { before: (0, 5), after: (18, 23) };
    for tile in [1, 4] {
        let routed = result.project_streamed(&functional, tile, &ctx).unwrap();
        let direct = engine::project_panel_law(
            prepared.model_data(),
            &fit.point,
            &fit.draws,
            &programs(&result),
            routed.weights(),
            tile,
            &ctx,
        )
        .unwrap();
        same("projected point", direct.point(), routed.point());
        same("projected draws", direct.draws(), routed.draws());
        same("stream receipt", direct.receipt(), routed.receipt());
        let projected = engine::project_panel_responses(&result.point, routed.weights()).unwrap();
        same("projected responses", projected.as_slice(), routed.point());
    }
}

#[test]
fn stationary_reference_decomposition_equals_the_response_route() {
    let ctx = context();
    let prepared = prepare(false, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Invariant(fit) = &result.fit else { panic!("stationary fit") };
    let data = prepared.model_data();
    let (request, periods) = decomposition_request(&result);

    let routed = result.decompose_reference(&request, &periods, &ctx).unwrap();
    let direct = engine::decomposition::decompose_invariant_law(
        data, &request, &periods, &fit.point, &fit.draws, &ctx,
    )
    .unwrap();
    same("invariant point", &direct.point, &routed.point);
    same("invariant draws", &direct.draws, &routed.draws);
    same("invariant weights", &direct.weights, &routed.weights);

    // The route pairs each fitted draw with itself through the period map.
    let point = periods.from_invariant(&fit.point).unwrap();
    let paired = fit
        .draws
        .iter()
        .map(|draw| draw.as_ref().map(|draw| periods.from_invariant(draw)).transpose())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let panel =
        engine::decomposition::decompose_panel_law(data, &request, &point, &paired, &ctx).unwrap();
    same("paired point", &panel.point, &routed.point);
    same("paired draws", &panel.draws, &routed.draws);

    let values =
        engine::decomposition::decompose_values(routed.point.hybrids, request.order).unwrap();
    same("hybrid values", &values, &routed.point);

    let streamed = result.decompose_reference_streamed(&request, &periods, &ctx).unwrap();
    let direct_streamed = engine::decomposition::decompose_invariant_law_streamed(
        data, &request, &periods, &fit.point, &fit.draws, &ctx,
    )
    .unwrap();
    same("invariant streamed", &direct_streamed, &streamed);
    assert!(result.decomposition_selection(&routed).unwrap().is_none());
}

#[test]
fn compiled_component_bindings_equal_the_weighted_hybrid_law() {
    let ctx = context();
    let prepared = prepare(false, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let worlds = &result.request().worlds;
    let before = worlds[0].world;
    let after = worlds[worlds.len() - 1].world;
    let bindings = ReferenceComponentBindings {
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
    let indices: [usize; 16] = std::array::from_fn(|mask| {
        worlds.iter().position(|world| world.world == bindings.hybrids[mask]).unwrap()
    });

    // Means only: the point is the terminal mean of each hybrid world, no draws.
    let routed = result
        .decompose_compiled_components(
            &bindings,
            ScientificConsumptionScope::ModelConditional,
            false,
            &ctx,
        )
        .unwrap();
    let point = indices.map(|index| *result.point[index].means.last().unwrap());
    let direct =
        engine::decomposition::decompose_hybrid_law(point, &[], bindings.order, &ctx).unwrap();
    same("means point", &direct.point, &routed.decomposition.point);
    same("means draws", &direct.draws, &routed.decomposition.draws);

    // With the aligned law: the point is the law's (weighted) coordinate mean.
    let routed = result
        .decompose_compiled_components(
            &bindings,
            ScientificConsumptionScope::ModelConditional,
            true,
            &ctx,
        )
        .unwrap();
    let law =
        result.numerical_project_law_source(&JointPanelResponseFunctional::TerminalMean).unwrap();
    let width = law.quantities().len();
    let weights = law.metadata().weights.clone();
    let total_weight =
        weights.as_ref().map_or(law.metadata().shape[0] as f64, |weights| weights.iter().sum());
    let mut point = [0.0; 16];
    let draws = law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(index, row)| {
            let values: [f64; 16] = std::array::from_fn(|j| row[indices[j]]);
            for j in 0..16 {
                point[j] += values[j]
                    * (weights.as_ref().map_or(1.0, |weights| weights[index]) / total_weight);
            }
            Some(values)
        })
        .collect::<Vec<_>>();
    let direct = engine::decomposition::decompose_weighted_hybrid_law(
        point,
        &draws,
        weights.as_deref(),
        bindings.order,
        &ctx,
    )
    .unwrap();
    same("law point", &direct.point, &routed.decomposition.point);
    same("law draws", &direct.draws, &routed.decomposition.draws);
    same("law weights", &direct.weights, &routed.decomposition.weights);
}

#[test]
fn declared_restriction_checks_are_the_restriction_diagnostics_of_the_fitted_point() {
    const HORIZON: usize = 24;
    let spec = UnrolledPanelSpec {
        periods: HORIZON,
        channels: 2,
        contexts: 1,
        carryover_reach: 0,
        outcome_persistence: false,
        history_feedback: false,
        latent_confounding: vec![],
        ..UnrolledPanelSpec::default()
    };
    let ctx = context();
    for (label, data) in [
        ("lag-zero", fixture::source_lag_zero()),
        (
            "skewed",
            fixture::source_lag_zero_departing(fixture::LagZeroDeparture::SkewedInnovations),
        ),
    ] {
        let mut request =
            fixture::request_lag_zero(&data, PanelInference::Frequentist { draws: 0 });
        request.target = declare_joint_panel_gaussian_restriction(&request.target);
        let panel = Study::prepare_joint_panel(data, request, &ctx).unwrap();
        let graph = UnrolledPanelGraph::build(spec.clone()).unwrap();
        let bound = panel.with_unrolled_graph_attempt(&graph, &[HORIZON - 1], &ctx).unwrap();
        assert_eq!(bound.plan_view().route(), PlanRoute::JointPanelResponse);
        let result = bound.estimate(&ctx).unwrap();
        let JointPanelFit::Invariant(fit) = &result.fit else { panic!("stationary fit") };
        let diagnostics = engine::restriction::panel_restriction_diagnostics(
            bound.model_data(),
            &fit.point,
            0.01,
        )
        .unwrap();
        let attempt = result
            .design_attempts()
            .iter()
            .find(|attempt| attempt.id == "joint_adjustment.gaussian_restriction")
            .unwrap();
        let expected = if diagnostics.flagged() {
            JointPanelDesignOutcome::Rejected
        } else {
            JointPanelDesignOutcome::Identified
        };
        assert_eq!(attempt.outcome, expected, "{label}: {}", attempt.detail);
        if let Some(check) = &diagnostics.normality {
            let text = format!(
                "innovation normality Jarque-Bera {:.3} (p {:.4}, n {})",
                check.jarque_bera, check.p_value, check.count
            );
            assert!(attempt.detail.contains(&text), "{label}: {text} in {}", attempt.detail);
        }
        for check in &diagnostics.linearity {
            let text = format!("{} lack of fit p {:.4}", check.feature, check.p_value);
            assert!(attempt.detail.contains(&text), "{label}: {text} in {}", attempt.detail);
        }
    }
}

// --- JointPanelTimeVarying -------------------------------------------------------------------

#[test]
fn temporal_fit_and_law_equal_the_time_varying_route() {
    let ctx = context();
    let prepared = prepare(true, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(routed) = &result.fit else { panic!("time-varying fit") };
    let data = prepared.model_data();
    let spec = prepared.request().temporal.as_ref().unwrap();
    let fit =
        engine::time_varying::fit_temporal_panel(data, spec, &prepared.request().options, &ctx)
            .unwrap();
    same("temporal point", &fit.point, &routed.point);
    same("temporal draws", &fit.draws, &routed.draws);
    same("temporal selection", &fit.selection_counts, &routed.selection_counts);

    let programs = programs(&result);
    let (point, draws, _retention) =
        engine::time_varying::evaluate_temporal_law(data, &fit.point, &fit.draws, &programs, &ctx)
            .unwrap()
            .into_parts();
    assert_eq!(bits(&point), bits(&result.point));
    assert_eq!(law_bits(&draws), law_bits(&result.draws));

    let single =
        engine::time_varying::evaluate_temporal_programs(data, &routed.point, &programs, &ctx)
            .unwrap();
    assert_eq!(bits(&single), bits(&result.point));
}

#[test]
fn temporal_streamed_and_projected_law_equal_the_time_varying_route() {
    let ctx = context();
    let prepared = prepare(true, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("time-varying fit") };
    let data = prepared.model_data();
    let programs = programs(&result);
    for tile in [1, 3] {
        let (receipt, rows) = collect_tiles(|sink| {
            engine::stream_temporal_law(
                data,
                &fit.point,
                &fit.draws,
                &programs,
                tile,
                &ctx,
                |first, rows| sink(first, rows),
            )
            .unwrap()
        });
        assert_eq!(bits(&receipt.point), bits(&result.point), "tile {tile}: point");
        assert_eq!(law_bits(&rows), law_bits(&result.draws), "tile {tile}: draws");
    }
    let functional = JointPanelResponseFunctional::WindowChange { before: (0, 5), after: (18, 23) };
    let routed = result.project_streamed(&functional, 2, &ctx).unwrap();
    let direct = engine::project_temporal_law(
        data,
        &fit.point,
        &fit.draws,
        &programs,
        routed.weights(),
        2,
        &ctx,
    )
    .unwrap();
    same("projected point", direct.point(), routed.point());
    same("projected draws", direct.draws(), routed.draws());
    same("stream receipt", direct.receipt(), routed.receipt());
}

#[test]
fn temporal_reference_decomposition_and_selection_equal_the_time_varying_route() {
    let ctx = context();
    let prepared = prepare(true, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let JointPanelFit::Varying(fit) = &result.fit else { panic!("time-varying fit") };
    let data = prepared.model_data();
    let (request, periods) = decomposition_request(&result);

    let routed = result.decompose_reference(&request, &periods, &ctx).unwrap();
    let direct = engine::decomposition::decompose_temporal_law(
        data, &request, &periods, &fit.point, &fit.draws, &ctx,
    )
    .unwrap();
    same("temporal point", &direct.point, &routed.point);
    same("temporal draws", &direct.draws, &routed.draws);

    let streamed = result.decompose_reference_streamed(&request, &periods, &ctx).unwrap();
    let direct_streamed = engine::decomposition::decompose_temporal_law_streamed(
        data, &request, &periods, &fit.point, &fit.draws, &ctx,
    )
    .unwrap();
    same("temporal streamed", &direct_streamed, &streamed);

    let selection = result.decomposition_selection(&routed).unwrap().unwrap();
    let direct_selection =
        engine::selection_uncertainty::decomposition_selection(&routed, &fit.point, &fit.draws)
            .unwrap();
    same("selection split", &direct_selection, &selection);
}

// --- JointDesign -----------------------------------------------------------------------------

#[test]
fn design_moment_law_equals_the_design_route() {
    let ctx = context();
    let recipe = || JointDesignRecipe::Randomized {
        design: JointRandomizedDesign {
            worlds: vec!["reference".into(), "active".into()],
            sampling_unit: "independent_cluster".into(),
            population: "declared_cluster_population".into(),
            consistency: "keys denote exact complete assigned programs".into(),
            between_unit_isolation: "independent randomized clusters without spillover".into(),
        },
        units: (0..16)
            .map(|i| JointRandomizedUnit {
                id: format!("unit:{i}"),
                arm: i % 2,
                probabilities: vec![0.5; 2],
                outcome: (if i % 2 == 0 { 1.0 } else { 2.0 }) + if i % 4 < 2 { 0.1 } else { -0.1 },
                population_weight: 1.0,
            })
            .collect(),
    };
    let prepared = PreparedNativeJointDesign::compile(recipe(), &ctx).unwrap();
    assert_eq!(prepared.plan_view().route(), PlanRoute::JointDesign);
    for inference in [
        JointDesignInference::Frequentist,
        JointDesignInference::RandomizedGaussian {
            prior: RandomizedGaussianPrior {
                mean: vec![0.0; 2],
                variance: vec![4.0; 2],
                outcome_variance: vec![1.0; 16],
                draws: 32,
                seed: 91,
            },
        },
    ] {
        let routed = prepared.execute(inference.clone(), &ctx).unwrap();
        let direct = recipe().execute(&inference, &ctx).unwrap();
        same("design law", &JointNativeDesignLaw::Moments(direct), routed.law());
    }
}
