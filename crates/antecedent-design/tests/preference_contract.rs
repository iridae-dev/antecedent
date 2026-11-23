//! Preference contracts: validation, identity, binding to an ordinary decision contract at
//! one parameter value, and the term and probe pseudo-contracts.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::decision_contract::{
    ActionKind, DecisionContractError, DecisionCriterion, HardConstraint, StructuralPolicy,
    UtilityExpr,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferenceContractError as E,
    PreferencePolicy,
};
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as P, PreferenceModelError, PreferenceParameter,
    PreferenceSet, RelationOrigin, RelationSense, ValueModel, shape,
};

mod common;

fn quantity(variable: &str, regime: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        regime_id: regime.into(),
        units: units.into(),
        ..common::quantity(variable)
    }
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "usd/unit".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn model_with(value: P) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "events".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet {
            parameters: vec![parameter("w_benefit", 1.0, 3.0), parameter("w_cost", 0.5, 1.0)],
            relations: vec![LinearRelation {
                id: "ratio".into(),
                terms: vec![(0, 1.0), (1, -2.0)],
                sense: RelationSense::AtLeast,
                bound: 0.0,
                origin: RelationOrigin::Declared,
            }],
        },
        value,
        utility_units: "usd".into(),
    }
}

/// `w_benefit * benefit - w_cost * cost - 0.5 * max(cost - 10, 0)`.
fn model() -> ValueModel {
    model_with(P::difference(
        P::difference(
            P::product(P::Parameter(0), shape::rate(0)),
            P::product(P::Parameter(1), shape::rate(1)),
        ),
        P::product(P::Const(0.5), shape::excess(1, 10.0)),
    ))
}

fn q(variable: &str, regime: &str, units: &str) -> ActionInput {
    ActionInput::Quantity(quantity(variable, regime, units))
}

fn fixed(value: f64, units: &str) -> ActionInput {
    ActionInput::Fixed { value, units: units.into() }
}

fn action(id: &str, inputs: Vec<ActionInput>) -> PreferenceAction {
    PreferenceAction { id: id.into(), kind: ActionKind::Intervention, inputs }
}

/// `treat` reads both attributes; `wait` has no cost; `refer` has a known benefit.
fn contract_with(model: ValueModel) -> PreferenceContract {
    PreferenceContract {
        model,
        actions: vec![
            action("treat", vec![q("benefit", "do(a=1)", "events"), q("cost", "do(a=1)", "usd")]),
            action("wait", vec![q("benefit", "do(a=0)", "events"), fixed(0.0, "usd")]),
            action("refer", vec![fixed(2.0, "events"), q("cost", "do(a=2)", "usd")]),
        ],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![
            HardConstraint {
                id: "budget".into(),
                expr: UtilityExpr::Input(1),
                bound: 20.0,
                min_probability: 0.9,
                units: "usd".into(),
                applies_to: vec![],
            },
            HardConstraint {
                id: "benefit-floor".into(),
                expr: UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))),
                bound: -1.0,
                min_probability: 0.5,
                units: "events".into(),
                applies_to: vec!["treat".into(), "wait".into()],
            },
        ],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

fn contract() -> PreferenceContract {
    contract_with(model())
}

/// Full attribute values of an action from its quantity values (in attribute order).
fn attributes_of(action: &PreferenceAction, quantities: &[f64]) -> Vec<f64> {
    let mut next = quantities.iter();
    action
        .inputs
        .iter()
        .map(|input| match input {
            ActionInput::Fixed { value, .. } => *value,
            ActionInput::Quantity(_) => *next.next().unwrap(),
        })
        .collect()
}

fn quantity_count(action: &PreferenceAction) -> usize {
    action.inputs.iter().filter(|i| matches!(i, ActionInput::Quantity(_))).count()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

/// Deterministic dyadic grid points for the equality checks.
fn grid() -> Vec<f64> {
    vec![-3.5, -0.25, 0.0, 0.75, 4.0, 10.0, 12.5, 31.0]
}

#[test]
fn the_bound_utility_equals_the_model_value_for_every_action() {
    let c = contract();
    assert_eq!(c.validate(), Ok(()));
    for theta in [vec![2.0, 0.75], vec![1.0, 0.5], vec![3.0, 1.0], vec![-1.25, 7.5]] {
        let bound = c.bind(&theta).unwrap();
        assert_eq!(bound.validate(), Ok(()));
        assert_eq!(bound.utility_units, "usd");
        assert_eq!(bound.actions.len(), 3);
        for (declared, ordinary) in c.actions.iter().zip(&bound.actions) {
            assert_eq!(ordinary.id, declared.id);
            assert_eq!(ordinary.inputs.len(), quantity_count(declared));
            for first in grid() {
                for second in grid() {
                    let pair = [first, second];
                    let quantities = &pair[..quantity_count(declared)];
                    let got = ordinary.utility.evaluate(quantities).unwrap();
                    let want =
                        c.model.value_at(&attributes_of(declared, quantities), &theta).unwrap();
                    assert_eq!(got.to_bits(), want.to_bits(), "{} at {quantities:?}", declared.id);
                }
            }
        }
    }
    // Quantities are renumbered densely in attribute order.
    let bound = c.bind(&[2.0, 0.75]).unwrap();
    assert_eq!(bound.actions[2].inputs, vec![quantity("cost", "do(a=2)", "usd")]);
    assert_eq!(bound.actions[2].utility.inputs_used(), vec![0]);
    assert_eq!(bound.actions[0].utility.inputs_used(), vec![0, 1]);
}

#[test]
fn constraints_are_rewritten_per_action_and_split_only_when_they_differ() {
    let bound = contract().bind(&[2.0, 0.75]).unwrap();
    let ids: Vec<&str> = bound.constraints.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, vec!["budget@treat", "budget@wait", "budget@refer", "benefit-floor"]);
    let by_id = |id: &str| bound.constraints.iter().find(|c| c.id == id).unwrap();
    assert_eq!(by_id("budget@treat").expr, UtilityExpr::Input(1));
    assert_eq!(by_id("budget@treat").applies_to, vec!["treat".to_string()]);
    assert_eq!(by_id("budget@wait").expr, UtilityExpr::Const(0.0));
    assert_eq!(by_id("budget@refer").expr, UtilityExpr::Input(0));
    let floor = by_id("benefit-floor");
    assert_eq!(floor.expr, UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))));
    assert_eq!(floor.applies_to, vec!["treat".to_string(), "wait".to_string()]);
    assert!((floor.bound + 1.0).abs() < f64::EPSILON);
    assert_eq!(by_id("budget@refer").units, "usd");
}

#[test]
fn binding_refuses_a_wrong_point_and_an_unlowerable_value() {
    let c = contract();
    assert_eq!(c.bind(&[1.0]), Err(E::ParameterCount { expected: 2, found: 1 }));
    assert_eq!(c.bind(&[1.0, f64::NAN]), Err(E::NonFiniteParameter { index: 1 }));

    // An exponential of a quantity attribute has no decision-utility form.
    let curved = model_with(P::difference(
        P::product(P::Parameter(0), P::exp(P::product(P::Const(0.5), P::Attribute(0)))),
        P::product(P::Parameter(1), P::Attribute(1)),
    ));
    let refused = contract_with(curved.clone()).bind(&[2.0, 0.75]);
    assert_eq!(refused, Err(E::NotLowerable { action: "treat".into() }));
    assert_eq!(refused.unwrap_err().detail(), reason_details::preference_contract::NOT_LOWERABLE);

    // With the benefit known for every action the exponential folds to a constant.
    let mut known = contract_with(curved);
    for (a, value) in known.actions.iter_mut().zip([1.0, 0.5, 2.0]) {
        a.inputs[0] = fixed(value, "events");
        a.inputs[1] = q("cost", &format!("do({})", a.id), "usd");
    }
    known.constraints.pop();
    let bound = known.bind(&[2.0, 0.75]).unwrap();
    for (declared, ordinary) in known.actions.iter().zip(&bound.actions) {
        for cost in grid() {
            let got = ordinary.utility.evaluate(&[cost]).unwrap();
            let want =
                known.model.value_at(&attributes_of(declared, &[cost]), &[2.0, 0.75]).unwrap();
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }
    // The term contract pads a constant part so the ordinary contract reads an input.
    let terms = known.term_contract().unwrap().unwrap();
    let form = known.model.linear_form().unwrap();
    let pseudo = &terms.actions[1]; // treat#term1 = exp(0.5 * benefit) at benefit 1
    assert_eq!(pseudo.id, "treat#term1");
    assert_eq!(pseudo.utility.inputs_used(), vec![0]);
    assert!(close(
        pseudo.utility.evaluate(&[123.0]).unwrap(),
        form.terms[0].evaluate(&[1.0, 123.0], &[]).unwrap()
    ));

    // A non-finite folded constant refuses.
    let log = model_with(P::difference(
        P::product(P::ln(P::Parameter(0)), P::Attribute(0)),
        P::product(P::Parameter(1), P::Attribute(1)),
    ));
    let mut log_contract = contract_with(log);
    log_contract.model.preferences.parameters[0].lower = 0.5;
    assert_eq!(
        log_contract.bind(&[0.0, 0.75]),
        Err(E::Model(PreferenceModelError::NonFiniteValue))
    );
    assert!(log_contract.bind(&[1.5, 0.75]).is_ok());
}

#[test]
fn a_parameter_inside_a_shape_still_binds_but_has_no_term_contract() {
    let shaped = model_with(P::difference(
        P::product(P::exp(P::Parameter(0)), P::Attribute(0)),
        P::product(P::Parameter(1), P::Attribute(1)),
    ));
    let c = contract_with(shaped);
    assert_eq!(c.model.shape_parameters(), vec![0]);
    assert_eq!(c.term_contract(), Ok(None));
    let theta = [0.5, 0.75];
    let bound = c.bind(&theta).unwrap();
    let ordinary = &bound.actions[0];
    let got = ordinary.utility.evaluate(&[3.0, 4.0]).unwrap();
    assert_eq!(got.to_bits(), c.model.value_at(&[3.0, 4.0], &theta).unwrap().to_bits());
}

#[test]
fn term_contract_indices_line_up_with_the_linear_form() {
    let c = contract();
    let form = c.model.linear_form().unwrap();
    let terms = c.term_contract().unwrap().unwrap();
    assert_eq!(terms.validate(), Ok(()));
    assert_eq!(terms.criterion, DecisionCriterion::PosteriorExpectedUtility);
    assert!(terms.constraints.is_empty());
    let k = 1 + c.model.preferences.dimension();
    assert_eq!(terms.actions.len(), c.actions.len() * k);
    let parts: Vec<&P> = std::iter::once(&form.fixed).chain(&form.terms).collect();
    for (a, declared) in c.actions.iter().enumerate() {
        for (j, part) in parts.iter().enumerate() {
            let pseudo = &terms.actions[a * k + j];
            assert_eq!(pseudo.id, format!("{}#term{j}", declared.id));
            assert_eq!(pseudo.kind, declared.kind);
            assert_eq!(pseudo.inputs.len(), quantity_count(declared));
            for first in grid() {
                for second in grid() {
                    let pair = [first, second];
                    let quantities = &pair[..quantity_count(declared)];
                    let got = pseudo.utility.evaluate(quantities).unwrap();
                    let want = part.evaluate(&attributes_of(declared, quantities), &[]).unwrap();
                    assert!(close(got, want), "{} at {quantities:?}: {got} vs {want}", pseudo.id);
                }
            }
        }
    }
    // Recombining the parts gives the bound utility.
    let theta = [2.5, 0.5];
    let bound = c.bind(&theta).unwrap();
    for (a, declared) in c.actions.iter().enumerate() {
        let pair = [4.0, 12.0];
        let quantities = &pair[..quantity_count(declared)];
        let mut recombined = terms.actions[a * k].utility.evaluate(quantities).unwrap();
        for (j, weight) in theta.iter().enumerate() {
            recombined +=
                weight * terms.actions[a * k + j + 1].utility.evaluate(quantities).unwrap();
        }
        assert!(close(recombined, bound.actions[a].utility.evaluate(quantities).unwrap()));
    }
}

#[test]
fn probe_contract_reads_each_quantity_on_its_own() {
    let c = contract();
    let probe = c.probe_contract().unwrap();
    let (probes, index) = (probe.contract, probe.probe_index);
    assert_eq!(probes.validate(), Ok(()));
    assert_eq!(index, vec![vec![Some(0), Some(1)], vec![Some(2), None], vec![None, Some(3)]]);
    let ids: Vec<&str> = probes.actions.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(ids, vec!["treat#probe0", "treat#probe1", "wait#probe0", "refer#probe1"]);
    for (a, declared) in c.actions.iter().enumerate() {
        for (k, input) in declared.inputs.iter().enumerate() {
            match (input, index[a][k]) {
                (ActionInput::Quantity(quantity), Some(p)) => {
                    assert_eq!(probes.actions[p].inputs, vec![quantity.clone()]);
                    assert_eq!(probes.actions[p].utility, UtilityExpr::Input(0));
                }
                (ActionInput::Fixed { .. }, None) => {}
                other => panic!("probe index does not match the inputs: {other:?}"),
            }
        }
    }
}

#[test]
fn identity_ignores_declaration_order_but_not_semantics() {
    let base = contract();
    let id = base.identity().unwrap();
    let mut reordered = base.clone();
    reordered.actions.reverse();
    reordered.constraints.reverse();
    reordered.model.preferences.relations.reverse();
    assert_eq!(reordered.identity().unwrap(), id);

    #[allow(clippy::type_complexity, reason = "a list of semantic edits")]
    let edits: Vec<Box<dyn Fn(&mut PreferenceContract)>> = vec![
        Box::new(|c| {
            c.actions[0].id = "treat-2".into();
            for constraint in &mut c.constraints {
                for action in &mut constraint.applies_to {
                    if action == "treat" {
                        *action = "treat-2".into();
                    }
                }
            }
        }),
        Box::new(|c| c.actions[0].kind = ActionKind::Policy),
        Box::new(|c| c.actions[1].inputs[1] = fixed(1.0, "usd")),
        Box::new(|c| c.actions[1].inputs[1] = q("cost", "do(a=0)", "usd")),
        Box::new(|c| c.actions[0].inputs[0] = q("benefit", "do(a=3)", "events")),
        Box::new(|c| c.model.preferences.parameters[0].upper = 4.0),
        Box::new(|c| c.constraints[0].bound = 25.0),
        Box::new(|c| c.constraints[1].applies_to = vec!["treat".into()]),
        Box::new(|c| {
            c.constraints.pop();
        }),
        Box::new(|c| c.criterion = DecisionCriterion::Quantile { p: 0.25 }),
        Box::new(|c| c.structural_policy = StructuralPolicy::Maximin),
        Box::new(|c| c.preference_policy = PreferencePolicy::MinimaxRegret),
        Box::new(|c| c.preference_policy = PreferencePolicy::Weighted),
    ];
    let mut seen = std::collections::BTreeSet::from([id]);
    for edit in edits {
        let mut changed = base.clone();
        edit(&mut changed);
        assert!(seen.insert(changed.identity().unwrap()));
    }
}

#[test]
#[allow(clippy::type_complexity, reason = "a table of edits and their refusals")]
#[allow(clippy::too_many_lines, reason = "one table listing every refusal")]
fn every_contract_refusal_names_its_cause() {
    let named = |a: &str, b: &str| (a.to_string(), b.to_string());
    let cases: Vec<(Box<dyn Fn(&mut PreferenceContract)>, E)> = vec![
        (
            Box::new(|c| c.model.utility_units = String::new()),
            E::Model(PreferenceModelError::BlankUtilityUnits),
        ),
        (Box::new(|c| c.target_population = " ".into()), E::BlankTargetPopulation),
        (Box::new(|c| c.criterion = DecisionCriterion::Quantile { p: 1.0 }), E::InvalidCriterion),
        (
            Box::new(|c| {
                c.criterion = DecisionCriterion::ThresholdProbability { threshold: f64::NAN }
            }),
            E::InvalidCriterion,
        ),
        (Box::new(|c| c.actions.truncate(1)), E::TooFewActions { count: 1 }),
        (Box::new(|c| c.actions[1].id = String::new()), E::BlankActionId { position: 1 }),
        (Box::new(|c| c.actions[2].id = "treat".into()), E::DuplicateAction { id: "treat".into() }),
        (
            Box::new(|c| c.actions[0].inputs.push(fixed(1.0, "usd"))),
            E::InputCount { action: "treat".into(), expected: 2, found: 3 },
        ),
        (
            Box::new(|c| c.actions[0].inputs = vec![fixed(1.0, "events"), fixed(1.0, "usd")]),
            E::NoQuantityInput { action: "treat".into() },
        ),
        (Box::new(|c| c.actions[1].inputs[1] = fixed(f64::INFINITY, "usd")), {
            let (action, attribute) = named("wait", "cost");
            E::NonFiniteFixedInput { action, attribute }
        }),
        (Box::new(|c| c.actions[1].inputs[1] = fixed(0.0, "eur")), {
            let (action, attribute) = named("wait", "cost");
            E::FixedUnitsMismatch { action, attribute }
        }),
        (Box::new(|c| c.actions[0].inputs[0] = q("benefit", " ", "events")), {
            let (action, attribute) = named("treat", "benefit");
            E::InvalidQuantity { action, attribute }
        }),
        (Box::new(|c| c.actions[0].inputs[0] = q("benefit", "do(a=1)", "usd")), {
            let (action, attribute) = named("treat", "benefit");
            E::QuantityUnitsMismatch { action, attribute }
        }),
        (
            Box::new(|c| {
                c.actions[2].inputs[1] = ActionInput::Quantity(ScientificQuantity {
                    population_id: "other".into(),
                    ..quantity("cost", "do(a=2)", "usd")
                })
            }),
            {
                let (action, attribute) = named("refer", "cost");
                E::OutsideScope { action, attribute }
            },
        ),
        (Box::new(|c| c.horizon = 2), {
            let (action, attribute) = named("treat", "benefit");
            E::OutsideScope { action, attribute }
        }),
        (Box::new(|c| c.constraints[1].id = String::new()), E::BlankConstraintId { position: 1 }),
        (
            Box::new(|c| c.constraints[1].id = "budget".into()),
            E::DuplicateConstraint { id: "budget".into() },
        ),
        (
            Box::new(|c| c.constraints[0].units = " ".into()),
            E::BlankConstraintUnits { constraint: "budget".into() },
        ),
        (
            Box::new(|c| c.constraints[0].bound = f64::NAN),
            E::ConstraintNonFinite { constraint: "budget".into() },
        ),
        (
            Box::new(|c| {
                c.constraints[0].expr =
                    UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Const(f64::INFINITY))
            }),
            E::ConstraintNonFinite { constraint: "budget".into() },
        ),
        (
            Box::new(|c| c.constraints[0].min_probability = 0.0),
            E::ConstraintProbability { constraint: "budget".into() },
        ),
        (
            Box::new(|c| c.constraints[0].min_probability = 1.5),
            E::ConstraintProbability { constraint: "budget".into() },
        ),
        (
            Box::new(|c| c.constraints[0].expr = UtilityExpr::Input(2)),
            E::ConstraintUnknownAttribute { constraint: "budget".into(), index: 2 },
        ),
        (
            Box::new(|c| c.constraints[1].applies_to = vec!["ghost".into()]),
            E::ConstraintUnknownAction {
                constraint: "benefit-floor".into(),
                action: "ghost".into(),
            },
        ),
    ];
    let mut details = std::collections::BTreeSet::new();
    for (edit, expected) in cases {
        let mut c = contract();
        edit(&mut c);
        let refused = c.validate().unwrap_err();
        assert_eq!(refused, expected);
        assert_eq!(c.identity(), Err(refused.clone()));
        assert_eq!(c.bind(&[2.0, 0.75]), Err(refused.clone()));
        assert_eq!(c.term_contract(), Err(refused.clone()));
        assert_eq!(c.probe_contract().map(|_| ()), Err(refused.clone()));
        details.insert(refused.detail());
    }
    assert!(details.contains(reason_details::preference_model::UTILITY_UNITS_BLANK));
    assert!(
        details
            .iter()
            .filter(|d| **d != reason_details::preference_model::UTILITY_UNITS_BLANK)
            .all(|d| d.starts_with("preference_contract."))
    );
    // Distinct causes: 24 cases, four of which repeat a cause.
    assert_eq!(details.len(), 20);
}

#[test]
fn a_refusal_of_the_derived_ordinary_contract_is_reported() {
    let mut c = contract();
    let mut deep = UtilityExpr::Input(0);
    for _ in 0..70 {
        deep = UtilityExpr::Neg(Box::new(deep));
    }
    c.constraints[1].expr = deep;
    assert_eq!(c.validate(), Ok(()));
    let refused = c.bind(&[2.0, 0.75]).unwrap_err();
    assert_eq!(refused, E::Contract(DecisionContractError::InvalidParameter("constraint")));
    assert_eq!(refused.detail(), reason_details::preference_contract::DECISION_CONTRACT_REFUSED);
    for other in [
        E::ParameterCount { expected: 2, found: 1 },
        E::NonFiniteParameter { index: 0 },
        E::NotLowerable { action: "a".into() },
    ] {
        assert!(other.detail().starts_with("preference_contract."));
        assert!(!other.to_string().is_empty());
    }
}

#[test]
fn a_degenerate_contract_with_one_quantity_per_action_binds_and_probes() {
    let mut c = contract();
    c.actions = vec![
        action("low", vec![fixed(1.0, "events"), q("cost", "do(a=0)", "usd")]),
        action("high", vec![fixed(3.0, "events"), q("cost", "do(a=1)", "usd")]),
    ];
    c.constraints.clear();
    let probe = c.probe_contract().unwrap();
    let (probes, index) = (probe.contract, probe.probe_index);
    assert_eq!(probes.actions.len(), 2);
    assert_eq!(index, vec![vec![None, Some(0)], vec![None, Some(1)]]);
    let bound = c.bind(&[1.0, 1.0]).unwrap();
    assert!(close(bound.actions[1].utility.evaluate(&[12.0]).unwrap(), 3.0 - 12.0 - 1.0));
}
