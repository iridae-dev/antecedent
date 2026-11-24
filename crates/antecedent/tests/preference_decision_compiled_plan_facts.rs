//! What the compiled plan states about a decision.
//!
//! An ordinary decision plan states its carrier, utility form, actions, constraints, units
//! and contract exactly as it always has. A decision across preferences states, besides,
//! every quantity it reads (canonical identity, schema variable, regime, units, population,
//! horizon, functional and scale), the representation it needs of its source and the ones
//! its contract accepts, the trade-off parameters with their declared ranges, and the bound
//! contract identity, all before anything is fitted.
//!
//! Fixture. A dose table `y = 0.5 + 1.5 t + noise` with the graph `t -> y`, and a mean
//! response curve of `y` under `do(t = 0.5)` and `do(t = 1.5)`; actions `low` and `high` read
//! one dose's mean benefit each and carry a known cost of 0.5 and 1.5, valued
//! `benefit - price * cost`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::DecisionCompileRequest;
use antecedent::analysis::native_claims::NativeResponseClaim;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, DecisionCriterion, PreferenceAction,
    PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr,
    PreferenceParameter, PreferencePolicy, PreferenceSet, StructuralPolicy, ValueModel, shape,
};
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{BayesianConfig, Dag, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext,
    GridSpec, ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_design::{AffineUtility, DecisionProblem};
use antecedent_graph::DenseNodeId;
use serde_json::json;

#[allow(dead_code, reason = "each test file uses only some of the hand computations")]
#[path = "common/preference_kernels.rs"]
mod kernels;

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

fn dose_study(data: &TabularData, inference: InferenceMode) -> Study {
    let mut graph = antecedent_graph::Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

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

fn contract(model: ValueModel) -> PreferenceContract {
    let action = |id: &str, dose: &str, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity(dose)),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions: vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

#[test]
fn an_ordinary_decision_plan_states_exactly_the_facts_it_always_did() {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y = [1.1, 3.0, 0.9, 3.1, 1.0, 2.9, 1.0, 3.0];
    let columns: [(&str, &[f64]); 2] = [("t", &t), ("y", &y)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
    let study = Study::tabular(data)
        .graph(Dag::from_named_edges(&schema, &[("t", "y")]).unwrap())
        .query(AverageEffectQuery::binary_ate(
            schema.id_of("t").unwrap(),
            schema.id_of("y").unwrap(),
        ))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(1.0e6),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, 2.0]).unwrap();
    let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]);
    let request = DecisionCompileRequest::new(CarrierDemand::ParameterPosterior)
        .with_contract_identity("launch-decision")
        .with_utility_unit("net_margin");
    let plan = study.compile_for_decision(&problem, &request, &ctx).unwrap();
    let json = plan.plan_view().unwrap().to_json();
    let decision = &json["evidence"]["decision"];
    let expected = json!({
        "carrier": "parameter_posterior",
        "affine_utility": true,
        "actions": 2,
        "constraints": 0,
        "chance_threshold": 1.0,
        "utility_unit": "net_margin",
        "contract_identity": "launch-decision",
        "scenario_source_identity": null,
        "information_value": null,
    });
    assert_eq!(decision, &expected);
    assert_eq!(
        decision.to_string(),
        "{\"actions\":2,\"affine_utility\":true,\"carrier\":\"parameter_posterior\",\
         \"chance_threshold\":1.0,\"constraints\":0,\"contract_identity\":\"launch-decision\",\
         \"information_value\":null,\"scenario_source_identity\":null,\
         \"utility_unit\":\"net_margin\"}",
        "the same bytes, key for key"
    );
    assert!(decision.get("preference").is_none());
}

#[test]
fn the_plan_states_a_preference_decisions_quantities_source_and_trade_offs() {
    let data = dose_table();
    let study = dose_study(&data, InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(11);
    let compiled =
        study.compile_for_preference_decision(&contract(priced(0.2, 1.0)), &ctx).unwrap();
    let json = compiled.plan_views()[0].to_json();
    let decision = &json["evidence"]["decision"];
    assert_eq!(decision["carrier"], "mean");
    assert_eq!(decision["contract_identity"], compiled.bound_identity());
    let preference = &decision["preference"];
    assert_eq!(preference["bound_contract_identity"], compiled.bound_identity());
    assert_eq!(preference["representation"], "mean");
    let accepted = preference["accepted"].as_array().unwrap();
    assert!(accepted.contains(&json!("mean")), "{accepted:?}");
    let y = data.schema().id_of("y").unwrap();
    let stated = preference["quantities"].as_array().unwrap();
    assert_eq!(stated.len(), 2);
    for (fact, dose) in stated.iter().zip(["0.5", "1.5"]) {
        assert_eq!(
            fact,
            &json!({
                "identity": quantity(dose).canonical_identity(),
                "variable": y.raw(),
                "variable_id": "y",
                "regime": format!("do(t={dose})"),
                "units": "points",
                "population": "target",
                "horizon": 0,
                "functional": "mean",
                "transform": "identity",
                "source": 0,
            })
        );
    }
    assert_eq!(
        preference["parameters"],
        json!([{
            "id": "price",
            "units": "points/usd",
            "lower": 0.2,
            "upper": 1.0,
            "assumption": "assumption.price_of_benefit",
        }])
    );
    assert!(preference["joint_law"].is_null(), "one study joins nothing");

    // The typed facts are the same statement.
    let facts = compiled.plan_views()[0]
        .evidence()
        .decision()
        .unwrap()
        .utility()
        .preference_facts()
        .cloned()
        .expect("a preference value states its facts");
    assert_eq!(facts.quantities()[1].identity(), quantity("1.5").canonical_identity());
    assert_eq!(facts.quantities()[1].variable(), Some(y));
    let range = (facts.parameters()[0].lower().to_bits(), facts.parameters()[0].upper().to_bits());
    assert_eq!(range, (0.2_f64.to_bits(), 1.0_f64.to_bits()));

    // The plan is stated before fitting, and the fitted answer is the result-level one.
    let executed = compiled.execute(None, PreferenceEvaluationOptions::default(), &ctx).unwrap();
    let result = study.run(&ctx).unwrap();
    let context = compiled.response_context(0, &ctx).unwrap();
    let claim =
        NativeResponseClaim::from_response(result.response().unwrap(), data.schema(), &context)
            .unwrap();
    let by_hand = kernels::decide_claim(
        &contract(priced(0.2, 1.0)),
        &claim,
        None,
        PreferenceEvaluationOptions::default(),
    )
    .unwrap();
    assert_eq!(executed.evaluation.single(), Some(&by_hand.evaluation));
}

#[test]
fn a_value_that_needs_draws_states_joint_draws_and_another_range_is_another_program() {
    let data = dose_table();
    let study = dose_study(&data, InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)));
    let ctx = ExecutionContext::for_tests(23);
    let mut kinked = priced(0.2, 1.0);
    kinked.preferences.parameters.push(PreferenceParameter {
        id: "shortfall_weight".into(),
        units: "points/points".into(),
        lower: 0.0,
        upper: 2.0,
        assumption_id: "assumption.weight_of_shortfall".into(),
    });
    kinked.value = PreferenceExpr::difference(
        kinked.value,
        PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::shortfall(0, 2.0)),
    );
    let compiled = study.compile_for_preference_decision(&contract(kinked), &ctx).unwrap();
    let json = compiled.plan_views()[0].to_json();
    let preference = &json["evidence"]["decision"]["preference"];
    assert_eq!(json["evidence"]["decision"]["carrier"], "parameter_posterior");
    assert_eq!(preference["representation"], "joint_draws");
    let ids: Vec<&str> = preference["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["price", "shortfall_weight"]);

    let other = study.compile_for_preference_decision(&contract(priced(0.2, 2.0)), &ctx).unwrap();
    let other_json = other.plan_views()[0].to_json();
    assert_eq!(other_json["evidence"]["decision"]["preference"]["parameters"][0]["upper"], 2.0);
    assert_ne!(other.plan_views()[0].identity(), compiled.plan_views()[0].identity());
}
