//! The parent program of a decision attached to compiled studies.
//!
//! Fixture. Dose tables `y = a + b t + noise` on `t` in `[0, 2]` with the graph `t -> y`, and a
//! mean response curve of the outcome under `do(t = 0.5)` and `do(t = 1.5)`. A trade-off
//! contract `benefit - price * cost` over one study's means takes the exact route; a contract
//! over two separately estimated outcomes scored by a quantile takes the certified route; an
//! ordinary contract reads each dose's mean through its own utility.
//!
//! Checked: the parent's identity is the children's program identities in declared order and
//! the decision's declaration (stable, order-sensitive, presence-sensitive); the plan view of
//! the parent reports the decision's own unit while each child keeps its own; and a supported
//! child under a decision unit that is not supported is refused by a release build before
//! anything runs, and run by a development build that records what a release build refuses,
//! with the same decision as under a supported unit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    SemanticDigest, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::accepted_preferences::{AcceptedPreferences, PreferenceContractParts};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_evaluation::PreferenceEvaluationOptions;
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr, PreferenceParameter, PreferenceSet, ValueModel, shape,
};
use antecedent_graph::{Dag, DenseNodeId};

use super::parent_identity;
use crate::analysis::execute::{
    OutcomeDependence, RowProvenance, compile_preference_decision_across_studies,
};
use crate::planner::PlanView;
use crate::support::CellStatus;
use crate::support::unit::{UnitDecision, UnitId, UnitRef};
use crate::{BayesianConfig, InferenceMode, RefuteSuite, Study};

const EXACT_UNIT: &str = "preference_decision.exact";
const CERTIFIED_UNIT: &str = "preference_decision.certified";
const CONTRACT_UNIT: &str = "decision.contract";

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

fn quantity(variable: &str, units: &str, population: &str, dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn parts(
    actions: Vec<PreferenceAction>,
    population: &str,
    criterion: DecisionCriterion,
) -> PreferenceContractParts {
    PreferenceContractParts {
        actions,
        criterion,
        constraints: vec![],
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

/// `benefit - price * cost` over the mean of `y` at the two doses and a known cost: affine in
/// its trade-off and scored by an expectation, so the exact route.
fn priced() -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower: 0.1,
                upper: 0.5,
                assumption_id: "assumption.price_of_benefit".into(),
            }],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("y", "points", "target", dose)),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(parts(
        vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)],
        "target",
        DecisionCriterion::PosteriorExpectedUtility,
    ))
}

/// `benefit - harm_rate * harm - excess_weight * max(harm - 1.2, 0)` over two outcomes of
/// separately estimated studies, scored by its median: a criterion that is not an
/// expectation, so the certified route over one aligned law.
fn kinked_median() -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: PreferenceSet::new(
            vec![
                PreferenceParameter {
                    id: "harm_rate".into(),
                    units: "points/events".into(),
                    lower: 0.1,
                    upper: 0.3,
                    assumption_id: "assumption.price_of_harm".into(),
                },
                PreferenceParameter {
                    id: "excess_weight".into(),
                    units: "points/events".into(),
                    lower: 0.0,
                    upper: 2.0,
                    assumption_id: "assumption.weight_of_excess_harm".into(),
                },
            ],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::difference(
                PreferenceExpr::Attribute(0),
                PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
            ),
            PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, 1.2)),
        ),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", "patients", dose)),
            ActionInput::Quantity(quantity("h", "events", "patients", dose)),
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(parts(
        vec![action("high", "1.5"), action("low", "0.5")],
        "patients",
        DecisionCriterion::Quantile { p: 0.5 },
    ))
}

/// `benefit - cost`, each action with its own utility over its own dose's mean.
fn ordinary() -> DecisionContract {
    let action = |id: &str, dose: &str, cost: f64| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity("y", "points", "target", dose)],
        utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(cost)),
    };
    DecisionContract {
        actions: vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)],
        utility_units: "points".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn dose_table(outcome: &str, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

/// A dose-response study of column 1 of a two-column table, declaring the table as its rows'
/// source.
fn dose_study(data: &TabularData, inference: InferenceMode) -> Study {
    let mut graph = Dag::with_variables(2);
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
        .row_provenance(RowProvenance::of_table(data).unwrap())
        .build()
        .unwrap()
}

/// The child program identities layer four of `parent` lists, in order.
fn listed_children(parent: &PlanView) -> Vec<Option<[u8; 32]>> {
    parent.evidence().sources().iter().map(|source| source.digest()).collect()
}

/// The unit `view` reports, by name.
fn unit_of(view: &PlanView) -> Option<String> {
    view.unit_support().map(|unit| unit.unit().to_owned())
}

#[test]
fn the_parent_identity_is_its_children_in_declared_order_and_the_decision() {
    let tag = "antecedent.attached_decision.fixture.v1";
    let (a, b) = (SemanticDigest::from_bytes([1; 32]), SemanticDigest::from_bytes([2; 32]));
    let decision = [7; 32];
    let declared = parent_identity(tag, &[Some(a), Some(b)], decision);
    assert_eq!(declared, parent_identity(tag, &[Some(a), Some(b)], decision), "stable");
    assert_ne!(declared, parent_identity(tag, &[Some(b), Some(a)], decision), "order");
    assert_ne!(declared, parent_identity(tag, &[Some(a)], decision), "a dropped child");
    assert_ne!(
        parent_identity(tag, &[None, Some(a)], decision),
        parent_identity(tag, &[Some(a)], decision),
        "a declared child that compiled no program still holds its place"
    );
    assert_ne!(declared, parent_identity(tag, &[Some(a), Some(b)], [8; 32]), "the decision");
    assert_ne!(
        declared,
        parent_identity(
            "antecedent.attached_decision.other_fixture.v1",
            &[Some(a), Some(b)],
            decision
        ),
        "the family's tag"
    );
}

#[test]
fn the_decision_states_its_own_unit_and_each_child_keeps_its_own() {
    let ctx = ExecutionContext::for_tests(3);
    let study = dose_study(&dose_table("y", (0.5, 1.5), 1.7), InferenceMode::Frequentist);

    let tradeoff = study.compile_for_preference_decision(&priced(), &ctx).unwrap();
    let parent = tradeoff.plan_view();
    assert_eq!(parent.program().route(), "PreferenceAttachedDecision");
    assert_eq!(unit_of(&parent).as_deref(), Some(EXACT_UNIT));
    assert_eq!(parent.identity(), tradeoff.program_identity());
    let children = tradeoff.plan_views();
    let [child] = children.as_slice() else {
        panic!("one study, one child program");
    };
    assert_eq!(listed_children(&parent), [Some(*child.identity().as_bytes())]);
    assert_ne!(unit_of(child).as_deref(), Some(EXACT_UNIT), "the child keeps its own cell");
    assert_ne!(child.identity(), parent.identity());

    let again = study.compile_for_preference_decision(&priced(), &ctx).unwrap();
    assert_eq!(again.program_identity(), tradeoff.program_identity(), "stable across compiles");

    let contract = study.compile_for_contract_decision(&ordinary(), &ctx).unwrap();
    let parent = contract.plan_view();
    assert_eq!(parent.program().route(), "ContractAttachedDecision");
    assert_eq!(unit_of(&parent).as_deref(), Some(CONTRACT_UNIT));
    assert_ne!(contract.program_identity(), tradeoff.program_identity());
}

#[test]
fn a_certified_decision_across_studies_follows_the_declared_order_of_its_studies() {
    let ctx = ExecutionContext::for_tests(5);
    let benefit = dose_study(&dose_table("b", (2.0, 1.2), 1.7), bayes());
    let harm = dose_study(&dose_table("h", (0.5, 0.5), 2.9), bayes());
    let dependence = OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 };
    let compile = |studies: &[&Study]| {
        compile_preference_decision_across_studies(studies, &kinked_median(), dependence, &ctx)
            .unwrap()
    };
    let declared = compile(&[&benefit, &harm]);
    let parent = declared.plan_view();
    assert_eq!(parent.program().route(), "CertifiedPreferenceAttachedDecision");
    assert_eq!(unit_of(&parent).as_deref(), Some(CERTIFIED_UNIT));

    // The children as declared: each study's decision program in study order, then the chain
    // that joins them.
    let mut expected: Vec<Option<[u8; 32]>> =
        declared.plan_views().iter().map(|view| Some(*view.identity().as_bytes())).collect();
    let chain = declared.joint_law().expect("a declared chain").fact().program();
    expected.push(Some(*chain.as_bytes()));
    assert_eq!(listed_children(&parent), expected);

    assert_eq!(compile(&[&benefit, &harm]).program_identity(), declared.program_identity());
    assert_ne!(
        compile(&[&harm, &benefit]).program_identity(),
        declared.program_identity(),
        "the studies are declared in order"
    );
}

#[test]
fn a_supported_child_under_an_unsupported_decision_unit_is_refused_or_recorded() {
    let study = dose_study(&dose_table("y", (0.5, 1.5), 1.7), InferenceMode::Frequentist);
    let compiled =
        study.compile_for_preference_decision(&priced(), &ExecutionContext::for_tests(11)).unwrap();
    let children = compiled.plan_views();
    let [child] = children.as_slice() else {
        panic!("one study, one child program");
    };
    assert!(
        child.unit_support().is_some_and(|unit| unit.supported()),
        "the child's own cell is supported"
    );
    let unsupported = UnitDecision {
        status: CellStatus::NotSupported,
        unit: UnitRef::Capability(UnitId::new(EXACT_UNIT)),
        missing_evidence: vec!["preference_decision.exact.truth"],
    };
    let gated = compiled.clone().deciding(unsupported);
    let ctx = ExecutionContext::for_tests(11);
    let run = gated.execute(None, PreferenceEvaluationOptions::default(), &ctx);
    if cfg!(feature = "development") {
        let run = run.expect("a development build runs the decision");
        let recorded = ctx.would_refuse();
        let ours: Vec<_> =
            recorded.iter().filter(|entry| entry.detail == "PreferenceAttachedDecision").collect();
        let [entry] = ours.as_slice() else {
            panic!("the decision's unit is recorded once: {recorded:?}");
        };
        assert_eq!(entry.code, "not_supported");
        assert_eq!(entry.missing, ["preference_decision.exact.truth"]);
        assert_eq!(run.would_refuse, recorded, "the execution states what was recorded");
        // The gate changes no number: the decision is the one a supported unit gives.
        let supported = compiled
            .execute(None, PreferenceEvaluationOptions::default(), &ExecutionContext::for_tests(11))
            .unwrap();
        assert_eq!(run.evaluation, supported.evaluation);
        assert_eq!(run.receipt, supported.receipt);
    } else {
        let error = run.expect_err("a release build refuses the decision");
        assert_eq!(error.code(), Some("not_supported"));
        assert!(error.to_string().contains("PreferenceAttachedDecision"), "{error}");
        assert!(ctx.would_refuse().is_empty());
    }
}
