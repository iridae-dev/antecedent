//! The facade over the chain: seal, change, export and consume, end to end on a
//! chain of measurement mapping, transport, derived quantity, nonlinear
//! utility and a preposterior update.

use antecedent_stats::normal_ppf;

use super::*;

const ROWS: usize = 1597;
const STRIDE: usize = 987;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn grid(mean: f64, sd: f64) -> Vec<f64> {
    (0..ROWS).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / ROWS as f64)).collect()
}

fn partner(mean: f64, sd: f64) -> Vec<f64> {
    let base = grid(mean, sd);
    (0..ROWS).map(|i| base[(i * STRIDE) % ROWS]).collect()
}

fn draft(reading: ChainReading, meaning: LawMeaning) -> PropagationChainRecordWire {
    let theta = grid(0.5, 0.6);
    PropagationChain::draft(
        TableConstruction::JointDraws,
        reading,
        "update",
        vec![
            ChainSourceWire::draws("effect", meaning, "t", &theta)
                .with_factor(SharedFactorKind::Row, "trial-rows"),
            ChainSourceWire::draws("cost", meaning, "t", &partner(0.2, 0.3))
                .with_factor(SharedFactorKind::Prior, "cost-prior"),
        ],
        vec![
            ChainStageWire::new(
                "mapped",
                StageKind::MeasurementMapping,
                &["effect"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(0.8),
                    ChainCoefficientWire::fixed(0.1),
                ),
            ),
            ChainStageWire::new(
                "transported",
                StageKind::Transport,
                &["mapped"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(0.9),
                    ChainCoefficientWire::fixed(0.0),
                ),
            ),
            ChainStageWire::new(
                "net",
                StageKind::DerivedQuantity,
                &["transported", "cost"],
                ChainOpWire::binary("difference"),
            ),
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["net"],
                ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
            )
            .with_decision_reference(0.0),
            ChainStageWire::new(
                "update",
                StageKind::PreposteriorUpdate,
                &["effect", "utility"],
                ChainOpWire::preposterior(1.0, 4.0, 0.0, &partner(0.0, 1.0)),
            ),
        ],
    )
}

#[test]
fn assumption_envelope_is_a_distinct_compiled_carrier() {
    use crate::planner::lowered::UncertaintyReading;
    use crate::planner::statistical::Realization;
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws("estimate", LawMeaning::Sampling, "table", &[1.0, 2.0, 3.0]),
            ChainSourceWire::interval("bias", 0.0, 1.0),
        ],
        vec![ChainStageWire::new(
            "answer",
            StageKind::DerivedQuantity,
            &["estimate", "bias"],
            ChainOpWire::binary("sum"),
        )],
    )
    .with_output_quantity(antecedent_io::quantity_wire::ScientificQuantityWire {
        version: 1,
        variable_id: "study.estimate_plus_bias".into(),
        variable_name: "Estimate plus bias".into(),
        role: "outcome".into(),
        units: "points".into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 1,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let plan = compile_propagation_chain(draft, &ctx()).expect("compiles");
    assert_eq!(plan.lowered().claim().uncertainty(), UncertaintyReading::AssumptionEnvelope);
    assert!(matches!(
        plan.lowered().statistical().realization(),
        Realization::DeclaredAssumptionEnvelope { draw_meaning: LawMeaning::Sampling, .. }
    ));
    let chain = plan.execute(&ctx()).expect("executes retained program");
    assert_eq!(chain.lowered_program_identity(), plan.identity());
    assert_eq!(chain.record().output_lower_bounds.as_ref().map(Vec::len), Some(3));
}

#[test]
fn assumption_envelope_reference_decision_uses_compiled_set_carrier() {
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "utility",
        vec![
            ChainSourceWire::draws("estimate", LawMeaning::Posterior, "table", &[1.0, 2.0]),
            ChainSourceWire::interval("assumption", 0.0, 1.0),
        ],
        vec![
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["estimate", "assumption"],
                ChainOpWire::binary("sum"),
            )
            .with_decision_reference(2.0),
        ],
    )
    .with_output_quantity(antecedent_io::quantity_wire::ScientificQuantityWire {
        version: 1,
        variable_id: "study.utility".into(),
        variable_name: "Utility".into(),
        role: "utility".into(),
        units: "points".into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 1,
        functional_id: "expected_utility".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let plan = compile_propagation_chain(draft, &ctx()).expect("compiles");
    let decision = plan.lowered().evidence().decision().expect("set decision");
    assert_eq!(decision.carrier(), crate::planner::lowered::CarrierDemand::IdentifiedSet);
    assert!(decision.scenario_source_identity().is_some());
    let chain = plan.execute(&ctx()).expect("executes retained program");
    assert_eq!(chain.record().results[0].count("robust_verdict_code"), Some(2));
}

#[test]
fn a_five_stage_chain_seals_exports_and_is_consumed_independently() {
    let chain = PropagationChain::seal(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
        .expect("seals");
    assert_eq!(chain.label(), "uncalibrated_spread");
    let bytes = chain.export().expect("exports");
    let consumed =
        consume_propagation_chain(&bytes, Some(chain.identity()), &ctx()).expect("consumes");
    assert_eq!(consumed.results, chain.results());
    // The update re-evaluated the decision after a simulated study of the effect.
    let update = chain.answer().expect("answer");
    assert!(update.value("value_of_information").expect("voi") >= -2e-2);
    assert!(update.value("probability_decision_changes").is_some());
}

#[test]
fn a_changed_assumption_reaches_only_the_stages_that_depend_on_it() {
    let chain = PropagationChain::seal(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
        .expect("seals");
    assert_eq!(
        chain.dependents_of("cost").expect("plan"),
        vec!["net".to_owned(), "utility".to_owned(), "update".to_owned()]
    );
    let withdrawn = chain.withdraw("cost", &ctx()).expect("changes");
    for id in ["mapped", "transported"] {
        assert_eq!(withdrawn.stage(id).expect("stage").status, STATUS_CURRENT);
        assert_eq!(withdrawn.stage(id), chain.stage(id));
    }
    for id in ["net", "utility", "update"] {
        assert_eq!(withdrawn.stage(id).expect("stage").status, STATUS_INVALIDATED);
    }
    // The original chain is unchanged.
    assert_eq!(chain.stage("net").expect("stage").status, STATUS_CURRENT);
    let tilt: Vec<f64> = grid(0.5, 0.6).iter().map(|e| 0.5 * e).collect();
    let reweighted = chain.reweight("effect", &tilt, &ctx()).expect("changes");
    assert_eq!(reweighted.stage("mapped").expect("stage").status, STATUS_REWEIGHTED);
    let bytes = reweighted.export().expect("exports");
    consume_propagation_chain(&bytes, Some(reweighted.identity()), &ctx()).expect("replays");
}

#[test]
fn the_interval_reading_is_not_supported_and_names_the_measured_coverage_attempt() {
    let chain = PropagationChain::seal(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
        .expect("seals");
    let refusal = chain.interval().expect_err("closed");
    assert_eq!(refusal.code(), "uncertainty_representation_insufficient");
    assert_eq!(refusal.next_attempt().name(), "measure_composed_coverage");
    assert!(chain.posterior().is_err());
}

#[test]
fn a_sampling_law_cannot_be_updated_or_read_as_a_posterior() {
    let refusal = PropagationChain::seal(draft(ChainReading::Spread, LawMeaning::Sampling), &ctx())
        .expect_err("refused");
    assert_eq!(refusal.code(), "uncertainty_representation_insufficient");
    assert_eq!(refusal.next_attempt().name(), "supply_posterior_draws");
}

#[test]
fn independent_sources_are_joined_by_their_product_not_by_index() {
    let table = independent_product_table(&[vec![0.0, 2.0], vec![0.0, 2.0]], 16).expect("table");
    let chain = PropagationChain::seal(
        PropagationChain::draft(
            TableConstruction::IndependentProduct,
            ChainReading::Spread,
            "product",
            vec![
                ChainSourceWire::draws("x", LawMeaning::Posterior, "product-table", &table[0]),
                ChainSourceWire::draws("y", LawMeaning::Posterior, "product-table", &table[1]),
            ],
            vec![ChainStageWire::new(
                "product",
                StageKind::DerivedQuantity,
                &["x", "y"],
                ChainOpWire::binary("product"),
            )],
        ),
        &ctx(),
    )
    .expect("seals");
    // Independent: E[XY] = 1. Paired by index the same marginals would give 2.
    let mean = chain.answer().expect("answer").value("mean").expect("mean");
    assert!((mean - 1.0).abs() < 1e-12);
    assert!(independent_product_table(&[vec![1.0; 10], vec![1.0; 10]], 50).is_err());
}

#[test]
fn a_chain_compiles_to_a_six_layer_program_before_anything_is_computed() {
    use crate::planner::lowered::Composition;
    let plan =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    let lowered = plan.lowered();
    assert_eq!(lowered.route(), "PropagationChain");
    assert_eq!(lowered.chain_reading(), Some("spread"));
    assert!(matches!(lowered.evidence().composition(), Composition::Chained { stages: 5 }));
    assert_eq!(lowered.evidence().sources().len(), 2);
    assert!(lowered.evidence().decision().is_none(), "version one keeps its program identity");
    let program = lowered.propagation_chain().expect("the program carries the chain");
    assert_eq!(program.dependents_of("cost"), vec!["net", "utility", "update"]);
    let view = plan.plan_view();
    assert_eq!(view.route(), crate::planner::PlanRoute::PropagationChain("spread"));
    assert_eq!(view.family(), crate::planner::RouteFamily::PropagationChain);
    crate::planner::plan_view::checks::assert_operations_derive("chain", &view);
    let operations = view.operations();
    let stages = operations.iter().filter(|operation| operation.name() == "propagate").count();
    assert_eq!(stages, 5, "every chain stage is an evidence operation");
    assert_eq!(view.identity(), plan.identity());
}

#[test]
fn execution_reads_the_retained_program_and_stamps_its_identity() {
    let plan =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    let chain = plan.execute(&ctx()).expect("executes");
    assert_eq!(chain.lowered_program_identity(), plan.identity());
    // The sealed record is the record of its own program.
    assert_eq!(chain.declaration_identity().expect("plan"), chain.identity());
    // The same declaration is the same program; a changed one is not.
    let again =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    assert_eq!(again.identity(), plan.identity());
    let other =
        compile_propagation_chain(draft(ChainReading::Point, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    assert_ne!(other.identity(), plan.identity());
    // A change is a new program, compiled and executed.
    let withdrawn = chain.withdraw("cost", &ctx()).expect("changes");
    assert_ne!(withdrawn.lowered_program_identity(), chain.lowered_program_identity());
}

#[test]
fn predictive_law_is_lowered_before_its_output_draws_are_published() {
    use crate::planner::lowered::UncertaintyReading;
    use crate::planner::statistical::Realization;

    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::Predictive,
        "utility",
        vec![
            ChainSourceWire::draws(
                "theta",
                LawMeaning::Posterior,
                "predictive-table",
                &[1.0, 2.0, 3.0],
            ),
            ChainSourceWire::draws(
                "epsilon",
                LawMeaning::Innovation,
                "predictive-table",
                &[0.5, -0.5, 0.5],
            ),
        ],
        vec![
            ChainStageWire::new(
                "y",
                StageKind::PredictiveMapping,
                &["theta", "epsilon"],
                ChainOpWire::binary("sum"),
            ),
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["y"],
                ChainOpWire::unary(ChainUnaryWire::Square),
            )
            .with_decision_reference(2.0),
        ],
    )
    .with_output_quantity(antecedent_io::quantity_wire::ScientificQuantityWire {
        version: 1,
        variable_id: "study.predictive_utility".into(),
        variable_name: "Predictive utility".into(),
        role: "utility".into(),
        units: "utility_points".into(),
        population_id: "target_population".into(),
        regime_id: "declared_regime".into(),
        horizon: 1,
        functional_id: "decision_utility".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let plan = compile_propagation_chain(draft, &ctx()).expect("compiles");
    assert_eq!(plan.lowered().claim().uncertainty(), UncertaintyReading::Predictive);
    assert!(matches!(
        plan.lowered().statistical().realization(),
        Realization::DeclaredPredictive { .. }
    ));
    let terminal = plan.lowered().evidence().decision().expect("compiled terminal comparison");
    assert_eq!(terminal.actions(), 2);
    assert_eq!(terminal.carrier(), crate::planner::lowered::CarrierDemand::Mean);
    assert_eq!(terminal.utility_unit(), Some("utility_points"));
    let chain = plan.execute(&ctx()).expect("executes retained program");
    assert_eq!(chain.lowered_program_identity(), plan.identity());
    assert_eq!(
        chain.record().version,
        antecedent_io::propagation_chain_record::PROPAGATION_CHAIN_LAW_VERSION
    );
    assert_eq!(chain.record().output_draws.as_ref().map(Vec::len), Some(3));
    let consumed =
        consume_propagation_chain(&chain.export().expect("export"), Some(chain.identity()), &ctx())
            .expect("replay");
    assert_eq!(consumed.output_draws, chain.record().output_draws);
    let mut different_quantity = chain.record().clone();
    different_quantity.output_quantity.as_mut().expect("quantity").units = "thousands".into();
    let changed = compile_propagation_chain(different_quantity, &ctx()).expect("new coordinate");
    assert_ne!(changed.identity(), plan.identity());
}

#[test]
fn compile_inside_a_lowered_dispatch_is_refused() {
    let plan =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    let scoped = ctx().enter_lowered_dispatch(plan.identity());
    let refusal =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &scoped)
            .expect_err("refused");
    assert_eq!(refusal.code(), "public_builder_reentry");
    assert!(
        PropagationChain::seal(draft(ChainReading::Spread, LawMeaning::Posterior), &scoped)
            .is_err()
    );
}

#[test]
fn a_refused_chain_does_not_compile() {
    let refusal =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Sampling), &ctx())
            .expect_err("refused");
    assert_eq!(refusal.next_attempt().name(), "supply_posterior_draws");
}

#[test]
fn the_chain_reads_the_support_decision_of_its_unit_and_execution_consults_it() {
    let plan =
        compile_propagation_chain(draft(ChainReading::Spread, LawMeaning::Posterior), &ctx())
            .expect("compiles");
    let decision = plan.support().expect("the chain route is bound to a unit");
    let view = plan.plan_view().unit_support().expect("the plan view reads the unit");
    assert_eq!(view.status(), decision.status);
    assert_eq!(view.unit(), decision.unit.label());
    let executed = plan.execute(&ctx());
    if RELEASE_SUPPORT_GATE_LIVE && !view.supported() && !cfg!(feature = "development") {
        assert_eq!(executed.expect_err("refused").code(), "not_supported");
    } else {
        executed.expect("the chain runs");
    }
}
