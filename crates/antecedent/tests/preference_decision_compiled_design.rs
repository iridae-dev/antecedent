//! A decision across preferences compiled with one design of several outcomes.
//!
//! The decision is attached to layer four of the design's own program, so the design's plan
//! shows the decision's quantities, carrier and trade-offs before anything is executed, the
//! carrier is checked against the design's realization at compile, and the execution enters
//! the decision-bearing program.
//!
//! Fixtures. Every design measures benefit `y` and harm `z = 2 y` on every unit, and the
//! treatment raises benefit by exactly 0.7, so harm rises by 1.4 and the value
//! `benefit - harm_rate * harm` switches at a harm rate of exactly one half:
//!
//! * a static-regime design (whole-subject refits supply an aligned law);
//! * a cluster-randomized design of 20 clusters of 10 rows, alternately assigned, whose
//!   baselines repeat in assigned pairs so both arms share them;
//! * a matched-pair experiment over the same clusters, paired as assigned.
//!
//! The randomized designs carry arm means with a conservative design-based covariance, not an
//! aligned law of their outcomes, so a value that needs a joint law is refused at compile
//! naming the design; an exposure with bias bounds leaves the arm means unidentified.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute,
    CompiledPreferenceDecision, DecisionCriterion, DesignBootstrap, PreferenceAction,
    PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr,
    PreferenceParameter, PreferencePolicy, PreferenceSet, PreferenceVerdict, StructuralPolicy,
    ValueModel, shape,
};
use antecedent::design::decision_contract::SourceRepresentation;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::reason::reason_details::preference_compile;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_design::cluster_decision::JointClusterRecipe;
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use antecedent_estimate::joint_design::excursion::ExcursionOutcome;
use antecedent_estimate::joint_design::experiment::{
    ExperimentExposure, ExperimentalDesign, JointExperimentalRecipe,
};
use antecedent_estimate::joint_design::static_regime::JointStaticRegimeRecipe;

const CLUSTERS: usize = 20;
const SIZE: usize = 10;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn arm(variable: &str, units: &str, regime: &str, horizon: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "patients".into(),
        regime_id: regime.into(),
        horizon,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - harm_rate * harm`, optionally with a kinked excess-harm charge.
fn model(range: (f64, f64), kink: Option<f64>) -> ValueModel {
    let mut parameters = vec![PreferenceParameter {
        id: "harm_rate".into(),
        units: "points/events".into(),
        lower: range.0,
        upper: range.1,
        assumption_id: "assumption.price_of_harm".into(),
    }];
    let mut value = PreferenceExpr::difference(
        PreferenceExpr::Attribute(0),
        PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
    );
    if let Some(cap) = kink {
        parameters.push(PreferenceParameter {
            id: "excess_weight".into(),
            units: "points/events".into(),
            lower: 0.0,
            upper: 2.0,
            assumption_id: "assumption.weight_of_excess_harm".into(),
        });
        value = PreferenceExpr::difference(
            value,
            PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, cap)),
        );
    }
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: PreferenceSet { parameters, relations: vec![] },
        value,
        utility_units: "points".into(),
    }
}

fn contract(model: ValueModel, horizon: u32) -> PreferenceContract {
    let action = |id: &str, regime: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(arm("y", "points", regime, horizon)),
            ActionInput::Quantity(arm("z", "events", regime, horizon)),
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions: vec![action("high", "on"), action("low", "off")],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "patients".into(),
        horizon,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn outcome(name: &str, units: &str, values: Vec<f64>) -> ExcursionOutcome {
    ExcursionOutcome {
        component_name: name.into(),
        variable_id: name.into(),
        units: units.into(),
        values,
    }
}

fn static_regime_design(ctx: &ExecutionContext) -> PreparedNativeJointDesign {
    let n = 120_usize;
    let assigned: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let x: Vec<f64> = (0..n).map(|i| (f64::from(u32::try_from(i).unwrap()) * 0.29).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let noise = f64::from(u32::try_from(i * 7 % 13).unwrap());
            2.0 + x[i] + if assigned[i] { 0.7 } else { 0.0 } + 0.1 * noise
        })
        .collect();
    let z: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let recipe = JointStaticRegimeRecipe {
        population: "patients".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        unit_ids: (0..n).map(|i| format!("u{i}")).collect(),
        periods: vec![0; n],
        assignments: assigned,
        confounder_ids: vec!["x".into()],
        confounders: vec![x],
        outcomes: vec![outcome("y", "points", y), outcome("z", "events", z)],
        active_path: vec![true],
        reference_path: vec![false],
        method: "g_formula".into(),
        min_probability: 0.05,
        positivity: "positivity".into(),
        exchangeability: "exchangeability".into(),
        consistency: "consistency".into(),
        followup: "followup".into(),
        model_assumption: "specified-model".into(),
        isolation: "isolation".into(),
    };
    PreparedNativeJointDesign::compile(JointDesignRecipe::StaticRegime { recipe }, ctx).unwrap()
}

/// Cluster totals of `y` (rows times the per-row mean) and of `z = 2 y`, alternately
/// assigned, with each assigned pair sharing its baseline.
fn cluster_totals() -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    let assigned: Vec<bool> = (0..CLUSTERS).map(|c| c % 2 == 0).collect();
    let size = f64::from(u32::try_from(SIZE).unwrap());
    let y: Vec<f64> = (0..CLUSTERS)
        .map(|c| {
            let baseline = 2.0 + 0.1 * f64::from(u32::try_from((c / 2) % 5).unwrap());
            size * (baseline + if assigned[c] { 0.7 } else { 0.0 })
        })
        .collect();
    let z = y.iter().map(|v| 2.0 * v).collect();
    (assigned, y, z)
}

fn cluster_design(
    exposure: ExperimentExposure,
    ctx: &ExecutionContext,
) -> PreparedNativeJointDesign {
    let (assignments, y, z) = cluster_totals();
    let recipe = JointClusterRecipe {
        population: "patients".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 2,
        cluster_ids: (0..CLUSTERS).map(|c| format!("c{c}")).collect(),
        cluster_sizes: vec![SIZE; CLUSTERS],
        assignment_law: "complete randomization of half the clusters".into(),
        treated_clusters: CLUSTERS / 2,
        consistency: "consistency".into(),
        assignments,
        outcomes: vec![outcome("y", "points", y), outcome("z", "events", z)],
        adjustments: vec![],
        exposure,
    };
    PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionCluster { recipe }, ctx).unwrap()
}

fn no_interference() -> ExperimentExposure {
    ExperimentExposure::NoInterference { assumption: "no spillover between clusters".into() }
}

fn matched_pair_design(ctx: &ExecutionContext) -> PreparedNativeJointDesign {
    let (assignments, y, z) = cluster_totals();
    let recipe = JointExperimentalRecipe {
        population: "patients".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 2,
        consistency: "consistency".into(),
        exposure: no_interference(),
        design: ExperimentalDesign::MatchedPairs {
            cluster_ids: (0..CLUSTERS).map(|c| format!("c{c}")).collect(),
            pairs: (0..CLUSTERS / 2).map(|p| (2 * p, 2 * p + 1)).collect(),
            matching_variables: vec!["baseline".into()],
            treatment_probabilities: vec![0.5; CLUSTERS],
            cluster_sizes: vec![SIZE; CLUSTERS],
            preassignment: "pairs fixed before assignment".into(),
            independent_pairs: "pairs assigned independently".into(),
        },
        assignments,
        outcomes: vec![outcome("y", "points", y), outcome("z", "events", z)],
    };
    PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionExperiment { recipe }, ctx)
        .unwrap()
}

fn verdict(compiled: &CompiledPreferenceDecision, ctx: &ExecutionContext) -> PreferenceVerdict {
    let executed = compiled.execute(None, options(), ctx).unwrap();
    executed.evaluation.single().unwrap().exact().unwrap().verdict.clone()
}

/// The three ranges around the switch at one half, `high` declared first.
fn check_switch(decide_on: impl Fn(f64, f64) -> PreferenceVerdict) {
    assert_eq!(decide_on(0.1, 0.4), PreferenceVerdict::NecessarilyBest("high".into()));
    assert_eq!(decide_on(0.6, 1.0), PreferenceVerdict::NecessarilyBest("low".into()));
    assert_eq!(
        decide_on(0.25, 0.75),
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["high".into(), "low".into()] }
    );
}

#[test]
fn the_design_plan_carries_the_decision_and_its_carrier_is_checked_at_compile() {
    let ctx = ExecutionContext::for_tests(7);
    let design = static_regime_design(&ctx);
    let probe = design
        .compile_for_preference_decision(&contract(model((0.1, 0.2), None), 1), None, &ctx)
        .unwrap();
    assert_eq!(probe.demand().carrier, Some(CarrierDemand::Mean));
    let views = probe.plan_views();
    assert_eq!(views.len(), 1);
    let view = &views[0];
    let decision = view.evidence().decision().expect("the decision is in layer four");
    assert_eq!(decision.carrier(), CarrierDemand::Mean);
    assert!(decision.utility().is_affine());
    assert_eq!(decision.contract_identity(), Some(probe.bound_identity()));
    assert_eq!(decision.utility().quantities().len(), 4);
    assert_ne!(view.identity(), design.program_identity(), "the decision changes layer four");
    assert!(design.lowered().evidence().decision().is_none(), "the design's own program is kept");
    let json = view.to_json();
    let stated = &json["evidence"]["decision"]["preference"]["quantities"];
    assert_eq!(stated.as_array().unwrap().len(), 4);
    assert!(stated[0]["variable"].is_null(), "a design names no schema variable");
    assert_eq!(stated[0]["regime"], "on");
    assert_eq!(stated[0]["horizon"], 1);

    // The execution enters the decision-bearing program.
    let executed = probe.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.programs, [view.identity()]);

    // A kinked value reads the whole-subject refits: aligned draws demanded of the design.
    let cap = 6.0;
    let refits = DesignBootstrap { replicates: 99, seed: 123 };
    let kinked = design
        .compile_for_preference_decision(
            &contract(model((0.1, 0.2), Some(cap)), 1),
            Some(refits),
            &ctx,
        )
        .unwrap();
    assert_eq!(kinked.demand().carrier, Some(CarrierDemand::DecisionDraws));
    let decision = kinked.plan_views()[0].evidence().decision().cloned().unwrap();
    assert_eq!(decision.carrier(), CarrierDemand::DecisionDraws);
    assert!(!decision.utility().is_affine());
}

#[test]
fn a_cluster_randomized_design_answers_by_its_arm_means() {
    let ctx = ExecutionContext::for_tests(9);
    let design = cluster_design(no_interference(), &ctx);
    let compile = |range: (f64, f64)| {
        design.compile_for_preference_decision(&contract(model(range, None), 2), None, &ctx)
    };
    let probe = compile((0.1, 0.2)).unwrap();
    assert_eq!(probe.demand().representation, SourceRepresentation::Mean);
    assert_eq!(probe.plan_views()[0].evidence().decision().unwrap().carrier(), CarrierDemand::Mean);
    let executed = probe.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::Mean(means)) = &executed.source else {
        panic!("the design's arm means");
    };
    // Treated benefit, harm, then control benefit, harm; the arms differ by exactly 0.7, 1.4.
    close(means.means[0] - means.means[2], 0.7, 1e-12);
    close(means.means[1] - means.means[3], 1.4, 1e-12);
    check_switch(|lower, upper| verdict(&compile((lower, upper)).unwrap(), &ctx));

    // The design's own frequentist law gives the same means, and the result-level evaluator
    // the same verdict.
    let law = design.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let JointNativeDesignLaw::DecisionCluster(arms) = law.law() else {
        panic!("a cluster law");
    };
    let own: Vec<f64> = arms.treated_means.iter().chain(&arms.control_means).copied().collect();
    assert_eq!(own, means.means);
    let decided = contract(model((0.1, 0.4), None), 2);
    let compiled = design.compile_for_preference_decision(&decided, None, &ctx).unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let source = executed.source.as_ref().unwrap();
    let by_hand = pd::decide(&decided, source, None, options()).unwrap();
    assert_eq!(executed.evaluation.single(), Some(&by_hand));
}

#[test]
fn a_matched_pair_experiment_answers_by_its_arm_means() {
    let ctx = ExecutionContext::for_tests(13);
    let design = matched_pair_design(&ctx);
    let compile = |range: (f64, f64)| {
        design.compile_for_preference_decision(&contract(model(range, None), 2), None, &ctx)
    };
    let executed = compile((0.1, 0.2)).unwrap().execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::Mean(means)) = &executed.source else {
        panic!("the experiment's arm means");
    };
    close(means.means[0] - means.means[2], 0.7, 1e-12);
    close(means.means[1] - means.means[3], 1.4, 1e-12);
    check_switch(|lower, upper| verdict(&compile((lower, upper)).unwrap(), &ctx));
    let law = design.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let JointNativeDesignLaw::DecisionExperiment(arms) = law.law() else {
        panic!("an experiment law");
    };
    assert_eq!(arms.means, means.means);
}

#[test]
fn a_joint_law_of_a_randomized_design_and_a_bounded_exposure_are_refused_at_compile() {
    let ctx = ExecutionContext::for_tests(17);
    for design in [cluster_design(no_interference(), &ctx), matched_pair_design(&ctx)] {
        let refused = design
            .compile_for_preference_decision(
                &contract(model((0.1, 0.2), Some(6.0)), 2),
                Some(DesignBootstrap { replicates: 99, seed: 1 }),
                &ctx,
            )
            .unwrap_err();
        let refusal = refused.refusal().expect("a compile refusal");
        assert_eq!(refusal.code, "joint_law_required");
        assert_eq!(refusal.stage, "compile");
        assert_eq!(refusal.detail, preference_compile::DESIGN_LAW_NOT_ALIGNED);
        assert_eq!(refusal.offending.as_deref(), Some(design.identification().method.as_str()));
    }

    let bounded = cluster_design(
        ExperimentExposure::ArmBounds {
            assumption: "spillover bounded".into(),
            treated: vec![0.1, 0.2],
            control: vec![0.1, 0.2],
        },
        &ctx,
    );
    let refused = bounded
        .compile_for_preference_decision(&contract(model((0.1, 0.2), None), 2), None, &ctx)
        .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!(refusal.code, "effect_not_identified");
    assert_eq!(refusal.stage, "compile");
    assert_eq!(refusal.detail, preference_compile::DESIGN_EXPOSURE_BOUNDED);
}
