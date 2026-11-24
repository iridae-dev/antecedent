//! Decisions across a range of preferences compiled with the study that estimates them.
//!
//! Fixture. A dose table `y = 0.5 + 1.5 t + noise` on `t` in `[0, 2]` with the graph `t -> y`,
//! and a mean response curve of `y` under `do(t = 0.5)` and `do(t = 1.5)`. The actions `low` and
//! `high` read one dose's mean benefit (points) each and carry a known cost (usd) of 0.5 and
//! 1.5. The value is `benefit - price * cost` with the price a declared range, so with
//! `gain = m(1.5) - m(0.5)` the action `high` leads exactly when `price < gain`.
//!
//! The decision is compiled with the study: the plan names its quantities and carrier before
//! anything is fitted, a quantity the study does not target is refused at bind time, a value
//! that needs a joint law is refused at compile on a frequentist study, and the compiled
//! execution agrees with the result-level functions on the same data.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::NativeResponseClaim;
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute, DecisionCriterion,
    PreferenceAction, PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions,
    PreferenceExpr, PreferenceParameter, PreferencePolicy, PreferenceRoute, PreferenceSet,
    PreferenceVerdict, StructuralPolicy, StructureIdentification, ValueModel, shape,
};
use antecedent::design::decision_contract::SourceRepresentation;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

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

fn study(data: &TabularData, inference: InferenceMode) -> Study {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([LOW_DOSE, HIGH_DOSE])),
            ),
        })))
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

/// `benefit - price * cost`.
fn priced(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower,
                upper,
                assumption_id: "assumption.price_of_benefit".into(),
            }],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        utility_units: "points".into(),
    }
}

/// `benefit - price * cost - weight * max(floor - benefit, 0)`.
fn kinked(floor: f64, price_range: (f64, f64)) -> ValueModel {
    let mut model = priced(price_range.0, price_range.1);
    model.preferences.parameters.push(PreferenceParameter {
        id: "shortfall_weight".into(),
        units: "points/points".into(),
        lower: 0.0,
        upper: 2.0,
        assumption_id: "assumption.weight_of_shortfall".into(),
    });
    model.value = PreferenceExpr::difference(
        model.value,
        PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::shortfall(0, floor)),
    );
    model
}

fn actions(low: ScientificQuantity, high: ScientificQuantity) -> Vec<PreferenceAction> {
    let action = |id: &str, quantity: ScientificQuantity, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    vec![action("low", low, 0.5), action("high", high, 1.5)]
}

fn contract_over(
    model: ValueModel,
    actions: Vec<PreferenceAction>,
    criterion: DecisionCriterion,
) -> PreferenceContract {
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions,
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn contract(model: ValueModel) -> PreferenceContract {
    contract_over(
        model,
        actions(quantity("0.5"), quantity("1.5")),
        DecisionCriterion::PosteriorExpectedUtility,
    )
}

#[test]
fn the_plan_names_the_decisions_quantities_and_carrier_before_anything_is_fitted() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let compiled =
        study.compile_for_preference_decision(&contract(priced(0.2, 1.0)), &ctx).unwrap();

    let demand = compiled.demand();
    assert_eq!(demand.route.as_ref().unwrap().route, PreferenceRoute::Exact);
    assert_eq!(demand.representation, SourceRepresentation::Mean);
    assert_eq!(demand.carrier, Some(CarrierDemand::Mean));
    assert_eq!(demand.quantities.len(), 2);
    assert_eq!(demand.quantities[0].quantity, quantity("0.5"));
    assert_eq!(demand.quantities[0].read_by, [("low".to_owned(), "benefit".to_owned())]);
    assert_eq!(demand.quantities[1].quantity, quantity("1.5"));

    let [structure] = compiled.structures() else {
        panic!("one structure");
    };
    assert!(matches!(
        structure.identification(),
        StructureIdentification::Identified(_) | StructureIdentification::IdentifiedAtPreparation
    ));
    let view = structure.plan_view().expect("a compiled plan");
    let decision = view.evidence().decision().expect("the decision is in layer four");
    assert_eq!(decision.carrier(), CarrierDemand::Mean);
    assert!(decision.utility().is_affine());
    assert_eq!(decision.actions(), 2);
    assert_eq!(decision.contract_identity(), Some(compiled.bound_identity()));
    let read: Vec<&str> = decision.utility().quantities().iter().map(|q| &**q).collect();
    assert_eq!(read, [quantity("0.5").canonical_identity(), quantity("1.5").canonical_identity()]);

    // Another preference set is another decision, so another program.
    let other = study.compile_for_preference_decision(&contract(priced(0.2, 2.0)), &ctx).unwrap();
    assert_ne!(other.plan_views()[0].identity(), view.identity());
    assert_ne!(other.bound_identity(), compiled.bound_identity());
}

#[test]
fn a_quantity_the_study_does_not_target_is_refused_when_bound() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);

    let off_grid = contract_over(
        priced(0.2, 1.0),
        actions(quantity("0.7"), quantity("1.5")),
        DecisionCriterion::PosteriorExpectedUtility,
    );
    let refused = study.bind_preference_contract(&off_grid).unwrap_err();
    let refusal = refused.refusal().expect("a binding refusal");
    assert_eq!(refusal.code, "quantity_semantics_mismatch");
    assert_eq!(refusal.stage, "bind");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::program_binding::DOSE_GRID_CHANGED
    );
    assert_eq!(refusal.offending.as_deref(), Some("action low attribute benefit"));
    let compile = study.compile_for_preference_decision(&off_grid, &ctx).unwrap_err();
    assert_eq!(compile.refusal().map(|r| r.detail.as_str()), Some(refusal.detail.as_str()));

    let mut elsewhere = quantity("1.5");
    elsewhere.variable_id = "w".into();
    elsewhere.variable_name = "w".into();
    let other_outcome = contract_over(
        priced(0.2, 1.0),
        actions(quantity("0.5"), elsewhere),
        DecisionCriterion::PosteriorExpectedUtility,
    );
    let refused = study.bind_preference_contract(&other_outcome).unwrap_err();
    let refusal = refused.refusal().expect("a binding refusal");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::program_binding::TREATMENT_OUTCOME_SUBSTITUTION
    );
    assert_eq!(refusal.offending.as_deref(), Some("action high attribute benefit"));

    let bound = study.bind_preference_contract(&contract(priced(0.2, 1.0))).unwrap();
    assert_eq!(bound.coordinates(0), Some(&[quantity("0.5"), quantity("1.5")][..]));
}

#[test]
fn a_value_that_needs_draws_is_refused_at_compile_on_a_frequentist_study() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let refused = study
        .compile_for_preference_decision(&contract(kinked(2.0, (0.2, 1.0))), &ctx)
        .unwrap_err();
    assert!(refused.compiler().is_some(), "{refused}");
    assert!(refused.to_string().contains("Bayesian"), "{refused}");
}

#[test]
fn means_answer_an_affine_value_and_agree_with_the_result_level_functions() {
    let data = dose_table();
    let study = study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let result = study.run(&ctx).unwrap();
    let probe = study.compile_for_preference_decision(&contract(priced(0.2, 1.0)), &ctx).unwrap();
    let context = probe.response_context(0, &ctx).expect("the compiler's binding");
    let claim =
        NativeResponseClaim::from_response(result.response().unwrap(), data.schema(), &context)
            .unwrap();
    let gain = claim.means()[1] - claim.means()[0];
    assert!(gain > 1.0 && gain < 2.0, "{gain}");

    for (range, expected) in [
        ((0.2, 0.8 * gain), PreferenceVerdict::NecessarilyBest("high".into())),
        (
            (0.5 * gain, 1.5 * gain),
            PreferenceVerdict::PreferenceDependent {
                possibly_best: vec!["low".into(), "high".into()],
            },
        ),
    ] {
        let decided = contract(priced(range.0, range.1));
        let compiled = study.compile_for_preference_decision(&decided, &ctx).unwrap();
        let executed = compiled.execute(None, options(), &ctx).unwrap();
        let evaluation = executed.evaluation.single().expect("one source");
        assert_eq!(evaluation.exact().unwrap().verdict, expected);
        assert_eq!(executed.inputs[0].representation, SourceRepresentation::Mean);
        assert_eq!(executed.receipt.bound_identity, compiled.bound_identity());
        // The receipt states the compiled decision program, and the executed result reported
        // that program's identity: the decision program is what ran.
        let stated = compiled.plan_views()[0].identity();
        assert_eq!(executed.receipt.programs, [stated]);
        assert_eq!(executed.receipt.executed, [Some(stated)]);

        let by_hand = pd::decide_claim(&decided, &claim, None, options()).unwrap();
        assert_eq!(evaluation, &by_hand.evaluation);
        assert_eq!(executed.source.as_ref(), Some(&by_hand.input.source));
    }
}

#[test]
fn joint_draws_answer_a_kinked_value_and_a_quantile_on_a_bayesian_study() {
    let data = dose_table();
    let study = study(&data, bayes());
    let ctx = ExecutionContext::for_tests(23);
    let result = study.run(&ctx).unwrap();
    let probe = study.compile_for_preference_decision(&contract(priced(0.2, 1.0)), &ctx).unwrap();
    let context = probe.response_context(0, &ctx).unwrap();
    let claim =
        NativeResponseClaim::from_response(result.response().unwrap(), data.schema(), &context)
            .unwrap();
    let gain = claim.means()[1] - claim.means()[0];
    let floor = claim.means()[0];

    for criterion in
        [DecisionCriterion::PosteriorExpectedUtility, DecisionCriterion::Quantile { p: 0.1 }]
    {
        let decided = contract_over(
            kinked(floor, (0.2, 0.8 * gain)),
            actions(quantity("0.5"), quantity("1.5")),
            criterion,
        );
        let compiled = study.compile_for_preference_decision(&decided, &ctx).unwrap();
        assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
        assert_eq!(compiled.demand().carrier, Some(CarrierDemand::ParameterPosterior));
        let view = &compiled.plan_views()[0];
        assert!(!view.evidence().decision().unwrap().utility().is_affine());

        let executed = compiled.execute(None, options(), &ctx).unwrap();
        let evaluation = executed.evaluation.single().unwrap();
        assert_eq!(evaluation.chosen_action(), Some("high"));
        let expected_route = if criterion == DecisionCriterion::PosteriorExpectedUtility {
            PreferenceRoute::Exact
        } else {
            PreferenceRoute::Certified
        };
        assert_eq!(evaluation.route(), expected_route);
        let by_hand = pd::decide_claim(&decided, &claim, None, options()).unwrap();
        assert_eq!(evaluation, &by_hand.evaluation);
    }
}

#[test]
fn a_prepared_study_compiles_the_same_decision() {
    let data = dose_table();
    let study = study(&data, bayes());
    let ctx = ExecutionContext::for_tests(29);
    let decided = contract(kinked(2.0, (0.2, 0.8)));
    let fresh = study.compile_for_preference_decision(&decided, &ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let held = prepared.compile_for_preference_decision(&decided, &ctx).unwrap();
    assert_eq!(held.bound_identity(), fresh.bound_identity());
    let from_fresh = fresh.execute(None, options(), &ctx).unwrap();
    let from_held = held.execute(None, options(), &ctx).unwrap();
    assert_eq!(from_held.evaluation, from_fresh.evaluation);
    assert_eq!(from_held.inputs[0].coordinates, from_fresh.inputs[0].coordinates);
}

#[test]
fn a_quantity_not_identified_under_the_structure_is_refused_at_compile() {
    // `t -> y` with a latent common cause `t <-> y`: the mean curve is not identified.
    let data = dose_table();
    let mut graph = antecedent_graph::Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([LOW_DOSE, HIGH_DOSE])),
            ),
        })))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let refused =
        study.compile_for_preference_decision(&contract(priced(0.2, 1.0)), &ctx).unwrap_err();
    let refusal = refused.refusal().expect("an identification refusal of the route");
    assert_eq!(refusal.code, "effect_not_identified", "{refused}");
    assert_eq!(refusal.stage, "compile");
    assert_eq!(
        refusal.offending.as_deref(),
        Some("action low attribute benefit, action high attribute benefit")
    );
    assert!(refusal.supplied.as_deref().is_some_and(|s| s.starts_with("unidentified mass 1")));
}
