// SPDX-License-Identifier: MIT OR Apache-2.0
//! The joint-panel model families run through the main compiler: every prepared handle names
//! its route and plan view, its results carry the program identity, retargeting keeps the fit
//! while changing the program, a child program runs only inside its parent's dispatch, and the
//! compiled route returns exactly what the engines return when they are called directly.

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test file: fixtures unwrap")]

#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use std::sync::Arc;

use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
use antecedent::analysis::joint_panel::selection::{
    PreparedRandomizedSelection, RandomizedCandidateGeneration, RandomizedSelectionPolicy,
};
use antecedent::analysis::joint_panel::{
    JointPanelFunctional, JointPanelResponseFunctional, JointPanelScientificScope,
    JointPanelSearchRequest, JointPanelSearchUncertainty, JointPanelState, PanelExperimentEvidence,
    PanelExperimentMapping, PanelExperimentModularPolicy, PanelExperimentUpdate,
    joint_panel_target,
};
use antecedent::planner::PlanRoute;
use antecedent::{PreparedStudy, Study};
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, EstimandBridge, ExposureSetting, MappingUncertainty, QuantityClass,
    SourceMappingDependence, ValueKind,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceKind,
};
use antecedent_core::execution::CachePolicy;
use antecedent_core::{CausalRng, ExecutionContext, InferenceSelection};
use antecedent_design::decision_contract::{
    ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
};
use antecedent_design::program_search::{FiniteProgramDomain, ProgramControl};
use antecedent_estimate::joint_design::compiled::{JointDesignInference, JointDesignRecipe};
use antecedent_estimate::joint_design::randomized::{
    JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior,
};
use antecedent_estimate::joint_panel::time_varying::{PanelMeanParameter, TemporalPanelSpec};
use antecedent_estimate::joint_panel::{self as engine, PanelInference, PanelPrior, PanelResponse};

/// A cache-enabled context, so a second estimate may reuse the first one's fit.
fn context() -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(20_261_119);
    ctx.cache_policy = CachePolicy::enabled(None);
    ctx
}

/// The reference panel prepared under `inference`, without fitting it.
fn prepare(inference: PanelInference, ctx: &ExecutionContext) -> PreparedStudy<JointPanelState> {
    let data = fixture::source(false);
    let request = fixture::request(&data, false, inference);
    Study::prepare_joint_panel(data, request, ctx).unwrap()
}

/// Every mean of every response, as raw bits, so equality is bit equality.
fn bits(rows: &[PanelResponse]) -> Vec<Vec<u64>> {
    rows.iter().map(|row| row.means.iter().map(|value| value.to_bits()).collect()).collect()
}

/// The refusal text of a call that must be refused.
fn refusal<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("the call must be refused"),
    }
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

/// The static response names its route, carries its identity into the result, and returns the
/// fitted point and every draw bit for bit as the engines do when called without the compiler.
#[test]
fn response_route_names_its_plan_and_matches_the_engines_bit_for_bit() {
    let ctx = context();
    let prepared = prepare(PanelInference::Frequentist { draws: 8 }, &ctx);
    assert_eq!(prepared.plan_view().route(), PlanRoute::JointPanelResponse);
    let unit = prepared.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(prepared.lowered().identity(), prepared.program_identity());
    assert_eq!(prepared.fits_performed(), 0, "reading the plan fits nothing");

    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(result.program_identity(), prepared.program_identity());
    assert_eq!(result.point.len(), 4);
    assert!(result.point.iter().all(|row| row.means.len() == 24));

    let data = prepared.model_data();
    let fit = engine::fit_panel(data, &prepared.request().options, &ctx).unwrap();
    let programs: Vec<_> =
        prepared.request().worlds.iter().map(|world| world.program.clone()).collect();
    let (point, draws, _retention) =
        engine::evaluate_panel_law(data, &fit.point, &fit.draws, &programs, &ctx)
            .unwrap()
            .into_parts();
    assert_eq!(bits(&result.point), bits(&point));
    assert_eq!(result.draws.len(), draws.len());
    for (routed, direct) in result.draws.iter().zip(&draws) {
        assert_eq!(routed.as_deref().map(bits), direct.as_deref().map(bits));
    }
}

/// A time-varying declaration is its own route with its own program identity.
#[test]
fn time_varying_declaration_names_its_own_route() {
    let ctx = context();
    let data = fixture::source(false);
    let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 8 });
    request.temporal = Some(temporal_spec());
    let varying = Study::prepare_joint_panel(data, request, &ctx).unwrap();
    let invariant = prepare(PanelInference::Frequentist { draws: 8 }, &ctx);
    assert_eq!(varying.plan_view().route(), PlanRoute::JointPanelTimeVarying);
    let unit = varying.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(varying.lowered().identity(), varying.program_identity());
    assert_ne!(varying.program_identity(), invariant.program_identity());
}

/// Retargeting lowers a new program but keeps the fit dependency: the program identity
/// changes, the fit is reused, and the response is bit-identical.
#[test]
fn retarget_changes_the_program_identity_and_keeps_the_fit() {
    let ctx = context();
    let prepared = prepare(PanelInference::Frequentist { draws: 8 }, &ctx);
    let first = prepared.estimate(&ctx).unwrap();

    let worlds = prepared.request().worlds.clone();
    let target = joint_panel_target(
        prepared.source_data().schema(),
        &prepared.request().columns,
        2,
        &worlds,
        JointPanelFunctional::Worlds,
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let next = prepared.retarget(worlds, target, &ctx).unwrap();
    assert_eq!(next.plan_view().route(), PlanRoute::JointPanelResponse);
    let unit = next.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_ne!(next.program_identity(), prepared.program_identity());

    let second = next.estimate(&ctx).unwrap();
    assert_eq!(second.program_identity(), next.program_identity());
    assert_eq!(second.dependency_identity(), first.dependency_identity());
    assert_eq!(prepared.fits_performed(), 1);
    assert_eq!(next.fits_performed(), 1, "the retargeted program reuses the fit");
    assert_eq!(bits(&second.point), bits(&first.point));
}

/// A public entry point refuses to run inside another program's dispatch, and the program
/// that encloses a child program executes it from a public context.
#[test]
fn reentry_is_refused_inside_a_dispatch() {
    let ctx = context();
    let prepared = prepare(PanelInference::Frequentist { draws: 8 }, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let scoped = ctx.enter_lowered_dispatch(prepared.program_identity());
    assert_eq!(scoped.lowered_dispatch(), Some(prepared.program_identity()));

    let data = fixture::source(false);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 8 });
    for (what, text) in [
        ("prepare", refusal(Study::prepare_joint_panel(data, request, &scoped))),
        ("estimate", refusal(prepared.estimate(&scoped))),
        (
            "retarget",
            refusal(prepared.retarget(
                prepared.request().worlds.clone(),
                prepared.request().target.clone(),
                &scoped,
            )),
        ),
        (
            "frozen retarget",
            refusal(result.retarget_frozen(
                prepared.request().worlds.clone(),
                prepared.request().target.clone(),
                &scoped,
            )),
        ),
    ] {
        assert!(text.contains("public_builder_reentry"), "{what} inside a dispatch: {text}");
    }
    // The same calls from the caller's own context are admitted.
    assert!(prepared.estimate(&ctx).is_ok());
}

/// Multiple imputation is a parent program whose completed tables run as children: the
/// pooled response is the mean of the fits, the result carries the parent identity, and the
/// parent itself is refused inside another dispatch.
#[test]
fn imputation_runs_its_children_inside_its_own_dispatch() {
    use antecedent_data::{ImputationDomain, ImputationSpec};
    use antecedent_estimate::multiple_imputation::PooledVectorEstimate;

    let ctx = context();
    let data = fixture::source(false);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 8 });
    let spec = ImputationSpec::new().with_imputations(3).with_iterations(2).with_seed(11);
    let domains = [(Arc::<str>::from("confounder"), ImputationDomain::Continuous)];
    let prepared = Study::prepare_joint_panel_imputed_with_domains(
        data.clone(),
        request.clone(),
        spec,
        &domains,
        &ctx,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(prepared.plan_view().route(), PlanRoute::JointPanelImputation);
    let unit = prepared.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(prepared.lowered().identity(), prepared.program_identity());
    let response = prepare(PanelInference::Frequentist { draws: 8 }, &ctx);
    assert_ne!(prepared.program_identity(), response.program_identity());

    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(result.program_identity(), prepared.program_identity());
    let PooledVectorEstimate::Rubin(pooled) = &result.pooled else {
        panic!("frequentist fits are pooled by Rubin's rules")
    };
    let m = result.fits.len() as f64;
    for coordinate in 0..pooled.dim {
        let mean = result.fits.iter().map(|fit| fit[coordinate]).sum::<f64>() / m;
        assert!((pooled.point[coordinate] - mean).abs() < 1e-12);
    }
    let again = prepared.estimate(&ctx).unwrap();
    assert_eq!(
        result.fits.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(),
        again.fits.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let scoped = ctx.enter_lowered_dispatch(response.program_identity());
    assert!(refusal(prepared.estimate(&scoped)).contains("public_builder_reentry"));
}

/// The finite program search over a fitted response names its route and returns the rows the
/// search engine returns when called directly on the retained plan.
#[test]
fn finite_search_names_its_route_and_matches_the_search_engine() {
    let ctx = context();
    let prepared = prepare(PanelInference::Frequentist { draws: 0 }, &ctx);
    let result = prepared.estimate(&ctx).unwrap();
    let horizon = result.request().worlds[0].horizon();
    let controls = (0..horizon)
        .flat_map(|offset| {
            result.request().columns.channels.iter().map(move |&channel| ProgramControl {
                variable: channel,
                offset: i32::try_from(offset).unwrap(),
                lower: 0.0,
                upper: 4.0,
                unit_cost: 0.0,
            })
        })
        .collect();
    let request = JointPanelSearchRequest {
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
        domain: FiniteProgramDomain { controls, budget: None, restrictions: vec![] },
        utility_units: "response".into(),
        max_candidates: 8,
    };
    let search = result.compile_search(&request, &ctx).unwrap();
    assert_eq!(search.plan_view().route(), PlanRoute::JointPanelProgramSearch);
    let unit = search.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(search.lowered().identity(), search.program_identity());
    assert_ne!(search.program_identity(), prepared.program_identity());

    let routed = search.execute(&ctx).unwrap();
    assert_eq!(routed.result.rows.len(), 4);
    assert!(routed.result.exhaustive);
    let direct = search
        .scientific_source()
        .search(search.numerical_plan(), JointPanelScientificScope::ModelConditional, &ctx)
        .unwrap();
    assert_eq!(format!("{:?}", routed.result.rows), format!("{:?}", direct.result.rows));
}

/// The held-out randomized selection names its route, and executing it twice from the same
/// compiled inputs selects the same program with the same held-out value.
#[test]
fn randomized_selection_names_its_route_and_is_reproducible() {
    let ctx = context();
    let compile = || {
        let mut rng = CausalRng::from_seed(41);
        let mut units = |stage: &str| {
            (0..30)
                .map(|i| {
                    let arm = usize::try_from(rng.next_u64() % 3).unwrap();
                    let noise = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt()
                        * (std::f64::consts::TAU * rng.next_f64()).cos();
                    JointRandomizedUnit {
                        id: format!("{stage}:{i}"),
                        arm,
                        probabilities: vec![1.0 / 3.0; 3],
                        outcome: if arm == 1 { 1.0 + noise } else { noise },
                        population_weight: 1.0,
                    }
                })
                .collect::<Vec<_>>()
        };
        let training = units("train");
        let validation = units("validation");
        PreparedRandomizedSelection::compile(
            JointRandomizedDesign {
                worlds: vec!["a".into(), "b".into(), "c".into()],
                sampling_unit: "independently sampled randomized cluster".into(),
                population: "common sampling frame".into(),
                consistency: "exact complete program and common outcome measure".into(),
                between_unit_isolation: "no interference across sampled clusters".into(),
            },
            training,
            validation,
            RandomizedSelectionPolicy {
                independent_sampling_frame: "IID clusters from the same predeclared frame".into(),
                utility_units: "outcome units".into(),
                benefit_multiplier: 1.0,
                utility_costs: vec![0.0; 3],
                resource_costs: vec![1.0; 3],
                budget: 2.0,
                generation: RandomizedCandidateGeneration::TrainingScreen { top_k: 2 },
            },
        )
        .unwrap()
    };
    let (first, second) = (compile(), compile());
    assert_eq!(first.plan_view().route(), PlanRoute::JointPanelProgramSelection);
    let unit = first.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(first.lowered().identity(), first.program_identity());
    assert_eq!(first.program_identity(), second.program_identity());

    let one = first.execute(4, 0.01, &ctx).unwrap();
    let two = second.execute(4, 0.01, &ctx).unwrap();
    assert_eq!(one.program_identity(), first.program_identity());
    let (one, two) = (&one.procedure().observed, &two.procedure().observed);
    assert_eq!(one.action, two.action);
    assert_eq!(one.heldout.value.to_bits(), two.heldout.value.to_bits());
}

/// The mapping and bridge of a conditional experiment on the reference panel's terminal mean.
fn experiment_update(
    original: &antecedent::analysis::joint_panel::JointPanelResult,
) -> (PanelExperimentUpdate, EvidenceLedger) {
    let functional = JointPanelResponseFunctional::TerminalMean;
    let target = original.experiment_quantity(&functional).unwrap();
    let mut source = target.clone();
    source.functional_id = "experimental_sample_mean".into();
    let setting = ExposureSetting {
        population_id: target.population_id.clone(),
        time_start: "program_period:23".into(),
        time_end: "program_period:23".into(),
        dose: 1.0,
        dose_units: "declared_joint_program".into(),
        duration: 1.0,
        duration_units: "panel_period".into(),
        exposure_regime: target.regime_id.clone(),
    };
    let bridge = BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint { quantity: source, class: QuantityClass::ModelParameter },
        target: BridgeEndpoint {
            quantity: target.clone(),
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: target.population_id.clone(),
            regime_id: target.regime_id.clone(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: 1.0 },
        assumptions: vec![BridgeAssumption {
            id: "declared_experiment_likelihood".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: "The experimental Gaussian mean maps to this exact joint-response contrast"
                .into(),
            evidence: Some("geo_experiment".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "paired_mapping".into(),
            intercept_variance: 0.0,
            slope_variance: 0.0,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.0,
            evidence: "retained_joint_slots".into(),
        }),
        estimand_bridge: Some(EstimandBridge {
            id: "exact_program_translation".into(),
            experiment_id: "geo_experiment".into(),
            experiment: setting.clone(),
            target: setting,
            recency_delta: 0.0,
            recency_units: "panel_period".into(),
            spend_delta: 0.0,
            spend_units: "declared_joint_program".into(),
        }),
    };
    bridge.check_support().unwrap();
    let mut ledger = EvidenceLedger::new();
    let source_of = |ledger: &mut EvidenceLedger, id: String, kind, key: &str| {
        ledger
            .add_source(
                SourceDeclaration::new(id, kind)
                    .with_observations([ObservationId::from_key(key, b"source")]),
            )
            .unwrap()
    };
    let model = source_of(
        &mut ledger,
        original.dependency_identity().to_hex(),
        SourceKind::FittedModel,
        "panel",
    );
    let mapping = source_of(&mut ledger, "mapping".into(), SourceKind::FittedModel, "mapping");
    let experiment = source_of(&mut ledger, "geo_experiment".into(), SourceKind::Experiment, "geo");
    let update = PanelExperimentUpdate {
        bridge,
        functional,
        observed_mean: 0.1,
        observation_variance: 1e-3,
        sample_size: 100,
        modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
        mapping: PanelExperimentMapping {
            dependency: original.dependency_identity(),
            intercepts: vec![0.0; 4],
            slopes: vec![1.0; 4],
        },
        evidence: PanelExperimentEvidence { model, prior_origins: vec![], experiment, mapping },
    };
    (update, ledger)
}

/// Conditioning a Bayesian panel on an experiment compiles an experiment program over the
/// panel's own program; the experiment program has its own identity and route.
#[test]
fn experiment_names_its_route_over_the_panel_program() {
    let ctx = context();
    let original = prepare(PanelInference::Frequentist { draws: 8 }, &ctx).estimate(&ctx).unwrap();
    let (update, ledger) = experiment_update(&original);
    let bayesian = prepare(
        PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: fixture::posterior_sampling(),
        },
        &ctx,
    );
    let program = bayesian.with_experiment(update, &ledger).unwrap();
    assert_eq!(program.plan_view().route(), PlanRoute::JointPanelExperiment);
    let unit = program.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(program.lowered().identity(), program.program_identity());
    assert_ne!(program.program_identity(), bayesian.program_identity());
    assert_eq!(bayesian.fits_performed(), 0, "compiling an experiment fits nothing");
}

/// A native randomized design names the design route and returns the law the design engine
/// returns when called directly on the same recipe.
#[test]
fn native_design_names_its_route_and_matches_the_design_engine() {
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
    let inference = JointDesignInference::RandomizedGaussian {
        prior: RandomizedGaussianPrior {
            mean: vec![0.0; 2],
            variance: vec![4.0; 2],
            outcome_variance: vec![1.0; 16],
            draws: 32,
            seed: 91,
        },
    };
    let prepared = PreparedNativeJointDesign::compile(recipe(), &ctx).unwrap();
    assert_eq!(prepared.plan_view().route(), PlanRoute::JointDesign);
    let unit = prepared.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "joint_response.interval_readings");
    assert!(unit.supported());
    assert_eq!(prepared.lowered().identity(), prepared.program_identity());
    let again = PreparedNativeJointDesign::compile(recipe(), &ctx).unwrap();
    assert_eq!(prepared.program_identity(), again.program_identity());

    let result = prepared.execute(inference.clone(), &ctx).unwrap();
    assert_eq!(result.program_identity(), prepared.program_identity());
    let direct = recipe().execute_native(&inference, &ctx).unwrap();
    assert_eq!(format!("{:?}", result.law()), format!("{direct:?}"));

    let scoped = ctx.enter_lowered_dispatch(prepared.program_identity());
    assert!(refusal(prepared.execute(inference, &scoped)).contains("public_builder_reentry"));
}
