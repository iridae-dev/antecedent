//! Identity properties of the decision contract: order independence of actions and
//! constraints, and sensitivity to every semantic field.

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};

type Mutation = (&'static str, Box<dyn Fn(&mut DecisionContract)>);

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn action(id: &str, kind: ActionKind, scale: f64) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind,
        inputs: vec![quantity("y")],
        utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(scale)),
    }
}

fn constraint(id: &str, bound: f64) -> HardConstraint {
    HardConstraint {
        id: id.into(),
        expr: UtilityExpr::Input(0),
        bound,
        min_probability: 1.0,
        units: "mmHg".into(),
        applies_to: vec![],
    }
}

fn base() -> DecisionContract {
    DecisionContract {
        actions: vec![
            action("treat", ActionKind::Intervention, 2.0),
            action("wait", ActionKind::Intervention, 0.5),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![constraint("cap_a", 10.0), constraint("cap_b", 20.0)],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn identity(contract: &DecisionContract) -> String {
    contract.identity().expect("valid contract")
}

#[test]
fn reordering_actions_and_constraints_leaves_identity_unchanged() {
    let reference = identity(&base());
    let mut reordered = base();
    reordered.actions.reverse();
    reordered.constraints.reverse();
    assert_eq!(identity(&reordered), reference);
}

#[test]
fn every_semantic_field_changes_identity() {
    let reference = identity(&base());
    let mutations: Vec<Mutation> = vec![
        ("action kind", Box::new(|c| c.actions[0].kind = ActionKind::Policy)),
        ("action id", Box::new(|c| c.actions[0].id = "treat_more".into())),
        (
            "action utility",
            Box::new(|c| {
                c.actions[0].utility =
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.5));
            }),
        ),
        ("action input", Box::new(|c| c.actions[0].inputs[0].units = "kPa".into())),
        ("constraint bound", Box::new(|c| c.constraints[0].bound = 11.0)),
        ("constraint probability", Box::new(|c| c.constraints[0].min_probability = 0.9)),
        ("constraint removed", Box::new(|c| drop(c.constraints.pop()))),
        ("utility units", Box::new(|c| c.utility_units = "cost".into())),
        ("criterion", Box::new(|c| c.criterion = DecisionCriterion::PosteriorExpectedLoss)),
        ("structural policy", Box::new(|c| c.structural_policy = StructuralPolicy::Maximin)),
    ];
    for (label, mutate) in mutations {
        let mut changed = base();
        mutate(&mut changed);
        assert_ne!(identity(&changed), reference, "{label} must change the identity");
    }
}

#[test]
fn identity_is_stable_across_clones_and_swapped_action_roles_differ() {
    let contract = base();
    assert_eq!(identity(&contract), identity(&contract.clone()));
    // Exchanging which action id carries which utility is a semantic change.
    let mut swapped = base();
    let (a, b) = (swapped.actions[0].utility.clone(), swapped.actions[1].utility.clone());
    swapped.actions[0].utility = b;
    swapped.actions[1].utility = a;
    assert_ne!(identity(&swapped), identity(&contract));
}
