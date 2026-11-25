//! Ordinary decision contracts compiled with the study that estimates their quantities.
//!
//! Fixture. A dose table `y = 0.5 + 1.5 t + noise` on `t` in `[0, 2]` with the graph `t -> y`,
//! and a mean response curve of `y` under `do(t = 0.5)` and `do(t = 1.5)`. The actions `low` and
//! `high` each read one dose's mean benefit (points) through their own utility, the benefit
//! less a known cost of 0.5 and 1.5 points, so `high` leads on the means.
//!
//! The ordinary contract compiles on the same route as a decision across preferences: the plan
//! names its quantities and carrier before anything is fitted, a binding refusal names the
//! action and the input position, and the executed decision is the contract's own evaluation
//! on the source the compiled execution read, with the same result and contract identity, on
//! means and on joint draws, with and without a hard constraint.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{NativeDecisionSource, NativeResponseClaim};
use antecedent::analysis::preference_decision::{
    MeasureOrigin, PreferenceEvaluationOptions, PreferenceMeasure,
};
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    SourceRepresentation, SourceRequirement, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionResult, Verdict, evaluate_contract, evaluate_contract_on_means,
};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

const LOW_DOSE: f64 = 0.5;
const HIGH_DOSE: f64 = 1.5;

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

/// Deterministic noise in `[-1, 1]`.
fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn dose_table() -> TabularData {
    let n = 240_u32;
    let t: Vec<f64> = (0..n).map(|i| 2.0 * f64::from(i) / f64::from(n - 1)).collect();
    let y: Vec<f64> =
        (0..n).zip(&t).map(|(i, dose)| 0.5 + 1.5 * dose + 0.8 * wobble(i, 1.7)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn response_query() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([LOW_DOSE, HIGH_DOSE])),
        ),
    }))
}

fn study(data: &TabularData, inference: InferenceMode) -> Study {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(response_query())
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

/// The mean of `y` under `do(t = dose)` in the target population, as the study names it.
fn quantity(dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "points".into(),
        population_id: "target".into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - cost`, each action with its own utility over its own input.
fn action(id: &str, read: ScientificQuantity, cost: f64) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![read],
        utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(cost)),
    }
}

fn contract_with(
    (low, high): (ScientificQuantity, ScientificQuantity),
    criterion: DecisionCriterion,
    constraints: Vec<HardConstraint>,
) -> DecisionContract {
    DecisionContract {
        actions: vec![action("low", low, 0.5), action("high", high, 1.5)],
        utility_units: "points".into(),
        criterion,
        constraints,
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn contract(criterion: DecisionCriterion) -> DecisionContract {
    contract_with((quantity("0.5"), quantity("1.5")), criterion, Vec::new())
}

/// The contract's own evaluation on `source`.
fn evaluated(contract: &DecisionContract, source: &NativeDecisionSource) -> DecisionResult {
    match source {
        NativeDecisionSource::Mean(means) => evaluate_contract_on_means(contract, means),
        NativeDecisionSource::JointLaw(law) => evaluate_contract(contract, law),
    }
    .unwrap()
}

/// The source the study's own run hands a decision in `representation`, through the claim
/// conversion under the compiler's binding.
fn source_of_run(
    study: &Study,
    data: &TabularData,
    context: &antecedent::analysis::native_claims::NativeResponseContext,
    representation: SourceRepresentation,
    ctx: &ExecutionContext,
) -> NativeDecisionSource {
    let result = study.run(ctx).unwrap();
    let claim =
        NativeResponseClaim::from_response(result.response().unwrap(), data.schema(), context)
            .unwrap();
    let requirement =
        SourceRequirement { any_of: vec![representation], sampled_needs_error_receipt: false };
    claim.decision_source(&requirement).unwrap().source
}

#[test]
fn a_prepared_effect_supplies_a_decision_a_recorded_claim_and_a_resumable_study() {
    use antecedent::analysis::decision_result::apply_decision;
    use antecedent::io::{
        RetainedArtifactIdentity, compiled_study, load_analysis_result_expecting,
    };
    use antecedent::prelude::AverageEffectQuery;

    let data = TabularData::from_f64_columns([
        ("a", &[0.0, 0.0, 1.0, 1.0][..]),
        ("y", &[0.0, 0.0, 2.0, 2.0][..]),
    ])
    .unwrap();
    let schema = data.schema().clone();
    let graph = Dag::from_named_edges(&schema, &[("a", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("a").unwrap(), schema.id_of("y").unwrap());
    let ctx = ExecutionContext::for_tests(11);
    let prepared = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate_retained(&ctx).unwrap();
    let contrast = ScientificQuantity {
        functional_id: "mean_difference".into(),
        regime_id: "contrast:do(a=1)-do(a=0)".into(),
        ..quantity("1")
    };
    let contract = DecisionContract {
        actions: vec![
            action("act", contrast.clone(), 0.0),
            DecisionAction {
                id: "wait".into(),
                kind: ActionKind::Intervention,
                inputs: vec![contrast],
                utility: UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
            },
        ],
        ..contract(DecisionCriterion::PosteriorExpectedUtility)
    };
    let evaluated = apply_decision(&contract, &result, &schema, &ctx).unwrap();
    assert_eq!(evaluated.ordinary().unwrap().verdict, Verdict::UniquelyOptimal("act".into()));

    let claim = prepared.encode_contracted_result(&result, "observed-effect", &ctx).unwrap();
    let retained = RetainedArtifactIdentity::of_bytes(&claim);
    let recorded = load_analysis_result_expecting(&claim, &retained).unwrap();
    assert!(recorded.is_intact());
    assert_eq!(recorded.body.estimate, Some(result.effect()));
    assert!(recorded.body.standard_error.is_none());
    assert!(recorded.body.interval_lower.is_none() && recorded.body.interval_upper.is_none());

    let saved = prepared.save_compiled_study(&result, &ctx, None).unwrap();
    let reopened =
        compiled_study::open(&saved.bytes, &saved.expected_identity, &|_| None, false).unwrap();
    assert!(reopened.resumable);
    assert_eq!(reopened.estimate, recorded.body.estimate);
}

#[test]
fn the_plan_names_an_ordinary_contracts_quantities_and_carrier_before_anything_is_fitted() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let declared = contract(DecisionCriterion::PosteriorExpectedUtility);
    let compiled = study.compile_for_contract_decision(&declared, &ctx).unwrap();

    assert!(compiled.contract().is_none());
    assert_eq!(compiled.decision_contract(), Some(&declared));
    assert_eq!(compiled.contract_identity(), declared.identity().unwrap());
    let demand = compiled.demand();
    assert!(demand.route.is_none());
    assert_eq!(demand.representation, SourceRepresentation::Mean);
    assert_eq!(demand.carrier, Some(CarrierDemand::Mean));
    assert_eq!(demand.quantities.len(), 2);
    assert_eq!(demand.quantities[0].quantity, quantity("0.5"));
    assert_eq!(demand.quantities[0].read_by, [("low".to_owned(), "input 0".to_owned())]);
    assert_eq!(demand.quantities[1].read_by, [("high".to_owned(), "input 0".to_owned())]);

    let view = &compiled.plan_views()[0];
    let decision = view.evidence().decision().expect("the decision is in layer four");
    assert_eq!(decision.carrier(), CarrierDemand::Mean);
    assert!(decision.utility().means_suffice());
    assert_eq!(decision.actions(), 2);
    assert_eq!(decision.constraints(), 0);
    assert_eq!(decision.utility_unit(), Some("points"));
    assert_eq!(decision.contract_identity(), Some(compiled.bound_identity()));
    let read: Vec<&str> = decision.utility().quantities().iter().map(|q| &**q).collect();
    assert_eq!(read, [quantity("0.5").canonical_identity(), quantity("1.5").canonical_identity()]);
    let facts = decision.utility().preference_facts().expect("the quantities are stated");
    assert!(facts.parameters().is_empty(), "an ordinary contract states no trade-offs");

    // Another contract is another decision, so another program.
    let other = study
        .compile_for_contract_decision(&contract(DecisionCriterion::PosteriorExpectedLoss), &ctx)
        .unwrap();
    assert_ne!(other.plan_views()[0].identity(), view.identity());
    assert_ne!(other.bound_identity(), compiled.bound_identity());
}

#[test]
fn means_answer_an_ordinary_contract_exactly_as_the_contract_evaluates_them() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let declared = contract(DecisionCriterion::PosteriorExpectedUtility);
    let compiled = study.compile_for_contract_decision(&declared, &ctx).unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();

    let result = executed.evaluation.ordinary().expect("the contract's own evaluation");
    assert!(executed.evaluation.single().is_none());
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("high".into()));
    assert_eq!(executed.inputs[0].representation, SourceRepresentation::Mean);
    let source = executed.source.as_ref().expect("one aligned source");
    assert!(matches!(source, NativeDecisionSource::Mean(_)));

    // The same beliefs, read from the study's own run, give the same result and identity.
    let context = compiled.response_context(0, &ctx).expect("the compiler's binding");
    let run = source_of_run(&study, &data, &context, SourceRepresentation::Mean, &ctx);
    assert_eq!(source, &run);
    let by_hand = evaluated(&declared, &run);
    assert_eq!(result, &by_hand);
    assert_eq!(result.contract_identity, declared.identity().unwrap());
    assert_eq!(executed.receipt.contract_identity, declared.identity().unwrap());
    assert_eq!(executed.receipt.bound_identity, compiled.bound_identity());
}

#[test]
fn joint_draws_answer_a_quantile_exactly_as_the_contract_evaluates_them() {
    let data = dose_table();
    let study = study(&data, bayes());
    let ctx = ExecutionContext::for_tests(23);
    let declared = contract(DecisionCriterion::Quantile { p: 0.1 });
    let compiled = study.compile_for_contract_decision(&declared, &ctx).unwrap();
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    assert_eq!(compiled.demand().carrier, Some(CarrierDemand::ParameterPosterior));
    assert!(!compiled.plan_views()[0].evidence().decision().unwrap().utility().is_affine());

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let result = executed.evaluation.ordinary().expect("the contract's own evaluation");
    let source = executed.source.as_ref().expect("one aligned source");
    assert!(matches!(source, NativeDecisionSource::JointLaw(_)));
    assert!(result.n_draws > 1);

    let context = compiled.response_context(0, &ctx).unwrap();
    let run = source_of_run(&study, &data, &context, SourceRepresentation::JointDraws, &ctx);
    assert_eq!(source, &run);
    assert_eq!(result, &evaluated(&declared, &run));
    assert_eq!(result.contract_identity, declared.identity().unwrap());
}

#[test]
fn a_hard_constraint_excludes_an_action_exactly_as_the_contract_evaluates_it() {
    let data = dose_table();
    let study = study(&data, bayes());
    let ctx = ExecutionContext::for_tests(31);
    // `high` reads a mean benefit near 2.75; requiring it below 2 with probability 0.9 excludes it.
    let ceiling = HardConstraint {
        id: "benefit_ceiling".into(),
        expr: UtilityExpr::Input(0),
        bound: 2.0,
        min_probability: 0.9,
        units: "points".into(),
        applies_to: vec!["high".into()],
    };
    let declared = contract_with(
        (quantity("0.5"), quantity("1.5")),
        DecisionCriterion::PosteriorExpectedUtility,
        vec![ceiling],
    );
    let compiled = study.compile_for_contract_decision(&declared, &ctx).unwrap();
    // A hard constraint is a probability a mean cannot answer.
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    let decision = compiled.plan_views()[0].evidence().decision().cloned().unwrap();
    assert_eq!(decision.constraints(), 1);
    assert!((decision.chance_threshold() - 0.9).abs() < f64::EPSILON);

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let result = executed.evaluation.ordinary().unwrap();
    let high = result.actions.iter().find(|a| a.id == "high").unwrap();
    assert!(!high.admissible);
    assert_eq!(high.exclusions.len(), 1);
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("low".into()));

    let context = compiled.response_context(0, &ctx).unwrap();
    let run = source_of_run(&study, &data, &context, SourceRepresentation::JointDraws, &ctx);
    assert_eq!(result, &evaluated(&declared, &run));
}

#[test]
fn the_receipt_names_the_decision_program_that_ran() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let declared = contract(DecisionCriterion::PosteriorExpectedUtility);
    let compiled = study.compile_for_contract_decision(&declared, &ctx).unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let stated = compiled.plan_views()[0].identity();
    assert_eq!(executed.receipt.programs, [stated]);
    assert_eq!(executed.receipt.executed, [Some(stated)]);
    assert!(executed.receipt.measure.is_none());

    // A prepared handle compiles and runs the same decision.
    let prepared = study.prepare(&ctx).unwrap();
    let held = prepared.compile_for_contract_decision(&declared, &ctx).unwrap();
    assert_eq!(held.bound_identity(), compiled.bound_identity());
    let from_held = held.execute(None, options(), &ctx).unwrap();
    assert_eq!(from_held.evaluation, executed.evaluation);
}

#[test]
fn a_measure_over_preferences_is_refused_for_an_ordinary_contract() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let compiled = study
        .compile_for_contract_decision(&contract(DecisionCriterion::PosteriorExpectedUtility), &ctx)
        .unwrap();
    let measure = PreferenceMeasure {
        points: vec![Vec::new()],
        weights: None,
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "assumption.viewpoint".into() },
        unresolved: Vec::new(),
    };
    let refused = compiled.execute(Some(&measure), options(), &ctx).unwrap_err();
    let refusal = refused.refusal().expect("a route refusal");
    assert_eq!(refusal.code, "option_not_applicable");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::preference_compile::ORDINARY_MEASURE_NOT_READ
    );
}

#[test]
fn a_binding_refusal_names_the_action_and_the_input_position() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let off_grid = contract_with(
        (quantity("0.7"), quantity("1.5")),
        DecisionCriterion::PosteriorExpectedUtility,
        Vec::new(),
    );
    let refused = study.compile_for_contract_decision(&off_grid, &ctx).unwrap_err();
    let refusal = refused.refusal().expect("a binding refusal");
    assert_eq!(refusal.code, "quantity_semantics_mismatch");
    assert_eq!(refusal.stage, "bind");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::program_binding::DOSE_GRID_CHANGED
    );
    assert_eq!(refusal.offending.as_deref(), Some("action low attribute input 0"));
}

#[test]
fn a_criterion_over_structures_is_refused_at_compile() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let refused = study
        .compile_for_contract_decision(&contract(DecisionCriterion::MinimaxOverIdentifiedSet), &ctx)
        .unwrap_err();
    let refusal = refused.refusal().expect("a route refusal");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::decision_evaluation::STRUCTURE_INPUTS_REQUIRED
    );
}

fn mask(edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, 2, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

#[test]
fn an_ordinary_contract_over_a_graph_posterior_is_refused_before_anything_is_fitted() {
    let data = dose_table();
    let posterior = GraphPosterior::from_atoms(
        2,
        vec![0.7, 0.3],
        vec![mask(&[(0, 1)]), mask(&[(1, 0)])],
        vec![0.0; 4],
        vec![0.0; 4],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("ordinary_structures"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag);
    let study = Study::tabular(data)
        .graph_posterior(posterior)
        .query(response_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let refused = study
        .compile_for_contract_decision(&contract(DecisionCriterion::PosteriorExpectedUtility), &ctx)
        .unwrap_err();
    let refusal = refused.refusal().expect("a route refusal");
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.stage, "compile");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::preference_compile::ORDINARY_STRUCTURES_NOT_COMPILED
    );
}
