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
//! * a matched-pair experiment over the same clusters, paired as assigned;
//! * a cluster-randomized design and a matched-pair experiment of 64 clusters (32 in each arm,
//!   32 pairs) whose per-row means carry a bounded cluster-level wobble, for the
//!   independent-unit bootstrap.
//!
//! Randomization alone does not identify the cross-arm dependence of the arm-mean estimators,
//! so a value that needs a joint law of a randomized design is refused at compile naming the
//! design unless the premise that its independent units were sampled independently from a
//! larger population is declared; then the independent-unit bootstrap supplies the law, its
//! kind, unit, refits, seed and premise enter the decision program's identity, and the premise
//! enters the evaluation's assumptions. An exposure with bias bounds leaves the arm means
//! unidentified.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute,
    CompiledPreferenceDecision, DecisionCriterion, DesignBootstrap, HardConstraint,
    PreferenceAction, PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions,
    PreferenceExpr, PreferenceParameter, PreferencePolicy, PreferenceSet, PreferenceVerdict,
    StructuralPolicy, ValueModel, shape,
};
use antecedent::design::decision_contract::{SourceRepresentation, UtilityExpr};
use antecedent::planner::lowered::CarrierDemand;
use antecedent::reason::reason_details::{decision_evaluation, preference_compile};
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
        preferences: PreferenceSet::new(parameters, vec![]),
        value,
        utility_units: "points".into(),
    }
}

fn contract(model: ValueModel, horizon: u32) -> PreferenceContract {
    contract_with(model, horizon, vec![])
}

fn contract_with(
    model: ValueModel,
    horizon: u32,
    constraints: Vec<HardConstraint>,
) -> PreferenceContract {
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
        constraints,
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

/// Clusters of the designs the independent-unit bootstrap reads: 32 in each arm, 32 pairs.
const SAMPLED: usize = 64;

/// A bounded cluster-level wobble in `[-1, 1]`.
fn wobble(c: usize) -> f64 {
    f64::from(u32::try_from(c * 7919 % 101).unwrap()) / 50.0 - 1.0
}

/// Cluster totals of `y` and `z = 2 y` over [`SAMPLED`] alternately assigned clusters: each
/// assigned pair shares its baseline, each cluster carries its own wobble, and the treatment
/// raises the per-row mean of `y` by 0.7.
fn sampled_totals() -> (Vec<bool>, Vec<f64>, Vec<f64>) {
    let assigned: Vec<bool> = (0..SAMPLED).map(|c| c % 2 == 0).collect();
    let size = f64::from(u32::try_from(SIZE).unwrap());
    let y: Vec<f64> = (0..SAMPLED)
        .map(|c| {
            let baseline = 2.0 + 0.1 * f64::from(u32::try_from((c / 2) % 5).unwrap());
            size * (baseline + 0.05 * wobble(c) + if assigned[c] { 0.7 } else { 0.0 })
        })
        .collect();
    let z = y.iter().map(|v| 2.0 * v).collect();
    (assigned, y, z)
}

fn sampled_cluster_design(ctx: &ExecutionContext) -> PreparedNativeJointDesign {
    let (assignments, y, z) = sampled_totals();
    let recipe = JointClusterRecipe {
        population: "patients".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 2,
        cluster_ids: (0..SAMPLED).map(|c| format!("c{c}")).collect(),
        cluster_sizes: vec![SIZE; SAMPLED],
        assignment_law: "complete randomization of half the clusters".into(),
        treated_clusters: SAMPLED / 2,
        consistency: "consistency".into(),
        assignments,
        outcomes: vec![outcome("y", "points", y), outcome("z", "events", z)],
        adjustments: vec![],
        exposure: no_interference(),
    };
    PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionCluster { recipe }, ctx).unwrap()
}

fn sampled_pair_design(ctx: &ExecutionContext) -> PreparedNativeJointDesign {
    let (assignments, y, z) = sampled_totals();
    let recipe = JointExperimentalRecipe {
        population: "patients".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 2,
        consistency: "consistency".into(),
        exposure: no_interference(),
        design: ExperimentalDesign::MatchedPairs {
            cluster_ids: (0..SAMPLED).map(|c| format!("c{c}")).collect(),
            pairs: (0..SAMPLED / 2).map(|p| (2 * p, 2 * p + 1)).collect(),
            matching_variables: vec!["baseline".into()],
            treatment_probabilities: vec![0.5; SAMPLED],
            cluster_sizes: vec![SIZE; SAMPLED],
            preassignment: "pairs fixed before assignment".into(),
            independent_pairs: "pairs assigned independently".into(),
        },
        assignments,
        outcomes: vec![outcome("y", "points", y), outcome("z", "events", z)],
    };
    PreparedNativeJointDesign::compile(JointDesignRecipe::DecisionExperiment { recipe }, ctx)
        .unwrap()
}

/// Refits under the declared premise that the independent units were sampled.
fn sampled_refits(replicates: usize, seed: u64) -> DesignBootstrap {
    DesignBootstrap {
        replicates,
        seed,
        sampling: Some("assumption.units_sampled_from_a_larger_population".into()),
    }
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
    let refits =
        |replicates: usize, seed: u64| DesignBootstrap { replicates, seed, sampling: None };
    let kinked_with = |bootstrap: DesignBootstrap| {
        design
            .compile_for_preference_decision(
                &contract(model((0.1, 0.2), Some(cap)), 1),
                Some(bootstrap),
                &ctx,
            )
            .unwrap()
    };
    let kinked = kinked_with(refits(99, 123));
    assert_eq!(kinked.demand().carrier, Some(CarrierDemand::DecisionDraws));
    let view = &kinked.plan_views()[0];
    let decision = view.evidence().decision().cloned().unwrap();
    assert_eq!(decision.carrier(), CarrierDemand::DecisionDraws);
    assert!(!decision.utility().is_affine());

    // The plan states the whole-subject refits, and their count and seed enter the identity.
    let law = decision.utility().preference_facts().unwrap().design_law().unwrap();
    assert_eq!((law.kind(), law.unit()), ("static_regime_refits", "subject"));
    assert_eq!((law.replicates(), law.seed(), law.sampling()), (99, 123, None));
    assert!(json["evidence"]["decision"]["preference"]["design_law"].is_null(), "means only");
    let stated = &view.to_json()["evidence"]["decision"]["preference"]["design_law"];
    assert_eq!(stated["kind"], "static_regime_refits");
    assert_eq!(stated["replicates"], 99);
    assert_eq!(view.identity(), kinked_with(refits(99, 123)).plan_views()[0].identity());
    assert_ne!(view.identity(), kinked_with(refits(99, 124)).plan_views()[0].identity());
    assert_ne!(view.identity(), kinked_with(refits(100, 123)).plan_views()[0].identity());
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
fn a_joint_law_of_a_randomized_design_needs_a_declared_sampling_premise() {
    let ctx = ExecutionContext::for_tests(17);
    let kinked = contract(model((0.1, 0.2), Some(6.0)), 2);
    for design in [sampled_cluster_design(&ctx), sampled_pair_design(&ctx)] {
        let undeclared = [
            None,
            Some(DesignBootstrap { replicates: 99, seed: 1, sampling: None }),
            Some(DesignBootstrap { replicates: 99, seed: 1, sampling: Some("  ".into()) }),
        ];
        for bootstrap in undeclared {
            let refused =
                design.compile_for_preference_decision(&kinked, bootstrap, &ctx).unwrap_err();
            let refusal = refused.refusal().expect("a compile refusal");
            assert_eq!(refusal.code, "joint_law_required");
            assert_eq!(refusal.stage, "compile");
            assert_eq!(refusal.detail, preference_compile::DESIGN_SAMPLING_UNDECLARED);
            assert_eq!(refusal.offending.as_deref(), Some(design.identification().method.as_str()));
        }
        let few = design
            .compile_for_preference_decision(&kinked, Some(sampled_refits(98, 1)), &ctx)
            .unwrap_err();
        let refusal = few.refusal().expect("a compile refusal");
        assert_eq!(refusal.code, "joint_law_required");
        assert_eq!(refusal.detail, decision_evaluation::JOINT_LAW_REQUIRED);
    }

    // Ten clusters in each arm, or ten pairs, are below the bootstrap's floor of thirty.
    for design in [cluster_design(no_interference(), &ctx), matched_pair_design(&ctx)] {
        let refused = design
            .compile_for_preference_decision(&kinked, Some(sampled_refits(99, 1)), &ctx)
            .unwrap_err();
        let refusal = refused.refusal().expect("a compile refusal");
        assert_eq!(refusal.code, "too_few_clusters");
        assert_eq!(refusal.stage, "compile");
        assert_eq!(refusal.detail, preference_compile::DESIGN_UNITS_TOO_FEW);
        assert_eq!(refusal.supplied.as_deref().map(|s| s.starts_with("10 ")), Some(true));
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

#[test]
fn a_declared_sampling_premise_reads_the_independent_unit_bootstrap_of_a_cluster_design() {
    let ctx = ExecutionContext::for_tests(19);
    let design = sampled_cluster_design(&ctx);
    let kinked = contract(model((0.1, 0.2), Some(5.8)), 2);
    let compile = |bootstrap: DesignBootstrap| {
        design.compile_for_preference_decision(&kinked, Some(bootstrap), &ctx).unwrap()
    };
    let compiled = compile(sampled_refits(99, 5));
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    assert_eq!(compiled.demand().carrier, Some(CarrierDemand::DecisionDraws));
    let view = &compiled.plan_views()[0];
    let decision = view.evidence().decision().unwrap();
    assert_eq!(decision.carrier(), CarrierDemand::DecisionDraws);
    let law = decision.utility().preference_facts().unwrap().design_law().unwrap();
    assert_eq!((law.kind(), law.unit()), ("independent_unit_bootstrap", "cluster"));
    assert_eq!(law.sampling(), Some("assumption.units_sampled_from_a_larger_population"));
    let stated = &view.to_json()["evidence"]["decision"]["preference"]["design_law"];
    assert_eq!(stated["kind"], "independent_unit_bootstrap");
    assert_eq!(stated["unit"], "cluster");
    assert_eq!(stated["replicates"], 99);
    assert_eq!(stated["seed"], 5);
    assert_eq!(stated["sampling"], "assumption.units_sampled_from_a_larger_population");

    // The refits, the seed and the premise enter the decision program's identity; the
    // design's own program is untouched.
    assert_eq!(view.identity(), compile(sampled_refits(99, 5)).plan_views()[0].identity());
    assert_ne!(view.identity(), compile(sampled_refits(99, 6)).plan_views()[0].identity());
    assert_ne!(view.identity(), compile(sampled_refits(120, 5)).plan_views()[0].identity());
    let other = DesignBootstrap {
        replicates: 99,
        seed: 5,
        sampling: Some("assumption.sites_drawn_at_random".into()),
    };
    assert_ne!(view.identity(), compile(other).plan_views()[0].identity());
    assert!(design.lowered().evidence().decision().is_none());

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.programs, [view.identity()]);
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("the independent-unit refits");
    };
    assert_eq!(law.shape(), [99, 4]);
    let identity = &law.metadata().identity;
    assert_eq!(identity.rng_id, "independent_unit_bootstrap:cluster:replicates=99:seed=5");
    assert_eq!(identity.provider_id, "antecedent.native_joint_design");
    // Every refit keeps the arms' 0.7 per-row benefit gap within the wobble's reach, and the
    // harm is exactly twice the benefit in every refit.
    let rows = law.draws().chunks(4);
    for row in rows {
        assert!((row[0] - row[2] - 0.7).abs() < 0.25, "{row:?}");
        close(row[1], 2.0 * row[0], 1e-12);
        close(row[3], 2.0 * row[2], 1e-12);
    }
    let evaluation = executed.evaluation.single().unwrap();
    assert_eq!(evaluation.chosen_action(), Some("high"));
    assert!(
        evaluation
            .assumptions()
            .iter()
            .any(|a| a == "assumption.units_sampled_from_a_larger_population")
    );
}

#[test]
fn a_kinked_value_and_a_probability_limit_run_on_a_matched_pair_bootstrap() {
    let ctx = ExecutionContext::for_tests(23);
    let design = sampled_pair_design(&ctx);
    let premise = "assumption.units_sampled_from_a_larger_population";
    let run = |contract: PreferenceContract| {
        let compiled = design
            .compile_for_preference_decision(&contract, Some(sampled_refits(199, 9)), &ctx)
            .unwrap();
        assert_eq!(compiled.demand().carrier, Some(CarrierDemand::DecisionDraws));
        let law = compiled.plan_views()[0]
            .evidence()
            .decision()
            .unwrap()
            .utility()
            .preference_facts()
            .unwrap()
            .design_law()
            .cloned()
            .unwrap();
        assert_eq!(law.unit(), "pair");
        let executed = compiled.execute(None, options(), &ctx).unwrap();
        let Some(NativeDecisionSource::JointLaw(draws)) = &executed.source else {
            panic!("the independent-unit refits");
        };
        assert_eq!(draws.shape(), [199, 4]);
        assert_eq!(
            draws.metadata().identity.rng_id,
            "independent_unit_bootstrap:pair:replicates=199:seed=9"
        );
        let evaluation = executed.evaluation.single().unwrap().clone();
        assert!(evaluation.assumptions().iter().any(|a| a == premise), "the premise is stated");
        evaluation
    };

    // A kinked charge on harm above the treated arm's harm, worth at most a small share of
    // the 0.42-point benefit margin at the highest harm rate.
    let kinked = run(contract(model((0.1, 0.2), Some(5.8)), 2));
    assert_eq!(kinked.chosen_action(), Some("high"));

    // A probability limit on the treated arm's harm: met with near certainty one event above
    // its mean, and never one event below it.
    let harm_cap = |bound: f64| HardConstraint {
        id: "harm_cap".into(),
        expr: UtilityExpr::Input(1),
        bound,
        min_probability: 0.9,
        units: "events".into(),
        applies_to: vec!["high".into()],
    };
    let treated_harm = {
        let law = design.execute(JointDesignInference::Frequentist, &ctx).unwrap();
        let JointNativeDesignLaw::DecisionExperiment(arms) = law.law() else {
            panic!("an experiment law");
        };
        arms.means[1]
    };
    let met = run(contract_with(model((0.1, 0.2), None), 2, vec![harm_cap(treated_harm + 1.0)]));
    assert_eq!(met.chosen_action(), Some("high"));
    let breached =
        run(contract_with(model((0.1, 0.2), None), 2, vec![harm_cap(treated_harm - 1.0)]));
    assert_eq!(breached.chosen_action(), Some("low"));
}
