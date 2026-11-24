//! Trade-offs between several outcomes, compiled with the analyses that estimate them.
//!
//! The value of an action is `benefit - harm_rate * harm` with the harm rate a declared range,
//! so between a treatment and its comparator the treatment leads exactly when
//! `harm_rate < d_benefit / d_harm`.
//!
//! Separately estimated outcomes. Benefit and harm are each estimated by a Bayesian
//! dose-response study of its own table. Their means meet side by side when the value is
//! affine; a kinked harm charge needs a joint law, which is refused at compile without a
//! declared dependence and, under distinct evidence, is drawn by the propagation chain from a
//! family table matched to each study's posterior moments. Two outcomes of one table are not
//! distinct evidence.
//!
//! One design, several outcomes. A static-regime design measures benefit `y` and harm
//! `z = 2 y` on every subject; the treatment raises benefit by 0.7, so the switch is at a harm
//! rate of one half. The design's arm means answer the affine value, and its aligned
//! whole-subject bootstrap refits supply one law of both outcomes under both arms.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::joint_panel::designs::PreparedNativeJointDesign;
use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, CompiledPreferenceDecision,
    DecisionCriterion, DesignBootstrap, OutcomeDependence, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceSet, PreferenceVerdict, RowProvenance, StructuralPolicy,
    ValueModel, compile_preference_decision_across_studies, shape,
};
use antecedent::design::decision_contract::SourceRepresentation;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::joint_design::compiled::JointDesignRecipe;
use antecedent_estimate::joint_design::excursion::ExcursionOutcome;
use antecedent_estimate::joint_design::static_regime::JointStaticRegimeRecipe;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::distribution_artifact::DistributionArtifact;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn quantity(
    variable: &str,
    units: &str,
    population: &str,
    regime: &str,
    horizon: u32,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - harm_rate * harm`.
fn traded(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "harm_rate".into(),
                units: "points/events".into(),
                lower,
                upper,
                assumption_id: "assumption.price_of_harm".into(),
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

/// `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`.
fn kinked(cap: f64, range: (f64, f64)) -> ValueModel {
    let mut model = traded(range.0, range.1);
    model.preferences.parameters.push(PreferenceParameter {
        id: "excess_weight".into(),
        units: "points/events".into(),
        lower: 0.0,
        upper: 2.0,
        assumption_id: "assumption.weight_of_excess_harm".into(),
    });
    model.value = PreferenceExpr::difference(
        model.value,
        PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, cap)),
    );
    model
}

fn action(id: &str, benefit: ScientificQuantity, harm: ScientificQuantity) -> PreferenceAction {
    PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![ActionInput::Quantity(benefit), ActionInput::Quantity(harm)],
    }
}

fn contract(
    model: ValueModel,
    actions: Vec<PreferenceAction>,
    population: &str,
    horizon: u32,
) -> PreferenceContract {
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: population.into(),
        horizon,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn verdict(compiled: &CompiledPreferenceDecision, ctx: &ExecutionContext) -> PreferenceVerdict {
    let executed = compiled.execute(None, options(), ctx).unwrap();
    executed.evaluation.single().unwrap().exact().unwrap().verdict.clone()
}

/// The three ranges around the switch `ratio`, with the treatment declared first.
fn check_switch(decide_on: impl Fn(f64, f64) -> PreferenceVerdict, ratio: f64) {
    assert_eq!(
        decide_on(0.2 * ratio, 0.8 * ratio),
        PreferenceVerdict::NecessarilyBest("high".into())
    );
    assert_eq!(
        decide_on(1.2 * ratio, 2.0 * ratio),
        PreferenceVerdict::NecessarilyBest("low".into())
    );
    assert_eq!(
        decide_on(0.5 * ratio, 1.5 * ratio),
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["high".into(), "low".into()] }
    );
}

fn column(law: &DistributionArtifact, j: usize) -> Vec<f64> {
    let [n, width] = law.shape();
    (0..n).map(|draw| law.draws()[draw * width + j]).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap())
}

// ------------------------------------------------------------- separately estimated outcomes

fn dose_table(outcome: &str, rows: u32, intercept: f64, slope: f64, rate: f64) -> TabularData {
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

fn dose_study(data: &TabularData, variables: u32, outcome: u32) -> Study {
    let mut graph = Dag::with_variables(variables);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(outcome)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(outcome),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        // A study of a whole table declares that table as its rows' source.
        .row_provenance(RowProvenance::of_table(data).unwrap())
        .build()
        .unwrap()
}

fn dosed(variable: &str, units: &str, dose: &str) -> ScientificQuantity {
    quantity(variable, units, "patients", &format!("do(t={dose})"), 0)
}

fn dosed_actions() -> Vec<PreferenceAction> {
    vec![
        action("high", dosed("b", "points", "1.5"), dosed("h", "events", "1.5")),
        action("low", dosed("b", "points", "0.5"), dosed("h", "events", "0.5")),
    ]
}

#[test]
fn separately_estimated_outcomes_meet_by_their_means_or_under_declared_distinct_evidence() {
    let ctx = ExecutionContext::for_tests(53);
    let benefit = dose_study(&dose_table("b", 220, 2.0, 1.2, 1.7), 2, 1);
    let harm = dose_study(&dose_table("h", 260, 0.5, 0.5, 2.9), 2, 1);
    let studies = [&benefit, &harm];
    let compile = |model: ValueModel, dependence: OutcomeDependence| {
        compile_preference_decision_across_studies(
            &studies,
            &contract(model, dosed_actions(), "patients", 0),
            dependence,
            &ctx,
        )
    };

    // Means suffice for the affine value: no dependence needs declaring.
    let probe = compile(traded(0.1, 0.2), OutcomeDependence::Undeclared).unwrap();
    assert_eq!(probe.demand().representation, SourceRepresentation::Mean);
    assert_eq!(probe.structures().len(), 2);
    let executed = probe.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::Mean(means)) = &executed.source else {
        panic!("the outcomes' means side by side");
    };
    assert_eq!(means.coordinates.len(), 4);
    let gain = means.means[1] - means.means[0];
    let extra_harm = means.means[3] - means.means[2];
    assert!(gain > 0.6 && extra_harm > 0.2, "{gain} {extra_harm}");
    let ratio = gain / extra_harm;
    check_switch(
        |lower, upper| {
            verdict(&compile(traded(lower, upper), OutcomeDependence::Undeclared).unwrap(), &ctx)
        },
        ratio,
    );

    // A kinked charge needs a joint law: refused at compile without a declared dependence.
    let cap = means.means[3];
    let refused = compile(kinked(cap, (0.2 * ratio, 0.8 * ratio)), OutcomeDependence::Undeclared)
        .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!((refusal.code, refusal.stage), ("joint_law_required", "compile"));

    // Under distinct evidence the chain draws one aligned law whose moments are the studies'.
    let distinct = OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 };
    let compiled = compile(kinked(cap, (0.2 * ratio, 0.8 * ratio)), distinct).unwrap();
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("one aligned law");
    };
    assert_eq!(law.shape(), [2000, 4]);
    assert_eq!(executed.receipt.programs.len(), 3, "two study plans and the chain");
    for j in 0..4 {
        let NativeDecisionSource::JointLaw(own) = &executed.inputs[j / 2].source else {
            panic!("each study's own posterior draws");
        };
        close(mean(&column(law, j)), mean(&column(own, j % 2)), 1e-9);
    }
    let harm_high = column(law, 3);
    let expected_excess = mean(&harm_high.iter().map(|h| (h - cap).max(0.0)).collect::<Vec<_>>());
    let decided = executed.evaluation.single().unwrap().exact().unwrap();
    close(decided.actions[0].coefficients.as_ref().unwrap()[2], -expected_excess, 1e-9);

    // Two outcomes of one table are not distinct evidence.
    let shared = shared_table();
    let first = dose_study(&shared, 3, 1);
    let second = dose_study(&shared, 3, 2);
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(kinked(cap, (0.2 * ratio, 0.8 * ratio)), dosed_actions(), "patients", 0),
        distinct,
        &ctx,
    )
    .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::preference_compile::STUDIES_SHARE_EVIDENCE
    );
}

fn shared_table() -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let b: Vec<f64> =
        (0..rows).zip(&t).map(|(i, t)| 2.0 + 1.2 * t + 0.6 * wobble(i, 1.7)).collect();
    let h: Vec<f64> =
        (0..rows).zip(&t).map(|(i, t)| 0.5 + 0.5 * t + 0.6 * wobble(i, 2.9)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("b", b.as_slice()), ("h", h.as_slice())])
        .unwrap()
}

// --------------------------------------------------------------- one design, several outcomes

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
    let outcome = |name: &str, units: &str, values: Vec<f64>| ExcursionOutcome {
        component_name: name.into(),
        variable_id: name.into(),
        units: units.into(),
        values,
    };
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

fn arm(variable: &str, units: &str, regime: &str) -> ScientificQuantity {
    quantity(variable, units, "patients", regime, 1)
}

fn arm_actions() -> Vec<PreferenceAction> {
    vec![
        action("high", arm("y", "points", "on"), arm("z", "events", "on")),
        action("low", arm("y", "points", "off"), arm("z", "events", "off")),
    ]
}

#[test]
fn one_design_answers_by_its_arm_means_or_its_aligned_refits() {
    let ctx = ExecutionContext::for_tests(7);
    let design = static_regime_design(&ctx);
    let compile = |model: ValueModel, bootstrap: Option<DesignBootstrap>| {
        design.compile_for_preference_decision(
            &contract(model, arm_actions(), "patients", 1),
            bootstrap,
            &ctx,
        )
    };

    // Arm means: the treatment raises benefit by 0.7 and harm by 1.4, so the switch is 0.5.
    let probe = compile(traded(0.1, 0.2), None).unwrap();
    assert_eq!(probe.demand().representation, SourceRepresentation::Mean);
    assert_eq!(probe.demand().carrier, Some(CarrierDemand::Mean));
    assert!(probe.structures().is_empty());
    assert_eq!(probe.plan_views().len(), 1);
    let executed = probe.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::Mean(means)) = &executed.source else {
        panic!("the design's arm means");
    };
    let gain = means.means[0] - means.means[2];
    let extra_harm = means.means[1] - means.means[3];
    close(extra_harm, 2.0 * gain, 1e-9);
    check_switch(|lower, upper| verdict(&compile(traded(lower, upper), None).unwrap(), &ctx), 0.5);

    // A kinked charge needs the refits; without them, or with too few, it is refused.
    let cap = means.means[1];
    let refused = compile(kinked(cap, (0.1, 0.2)), None).unwrap_err();
    assert_eq!(refused.code(), Some("joint_law_required"));
    let few = DesignBootstrap { replicates: 50, seed: 123, sampling: None };
    assert!(compile(kinked(cap, (0.1, 0.2)), Some(few)).is_err());

    let refits = DesignBootstrap { replicates: 99, seed: 123, sampling: None };
    let compiled = compile(kinked(cap, (0.1, 0.2)), Some(refits)).unwrap();
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("the aligned refits");
    };
    assert_eq!(law.shape(), [99, 4]);
    let (benefit, harm) = (column(law, 0), column(law, 1));
    for (b, h) in benefit.iter().zip(&harm) {
        close(*h, 2.0 * b, 1e-9);
    }
    let evaluation = executed.evaluation.single().unwrap();
    assert_eq!(evaluation.chosen_action(), Some("high"));
}
