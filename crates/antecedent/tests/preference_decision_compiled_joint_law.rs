//! Separately estimated outcomes joined into one law, declared and checked at compile.
//!
//! The value of an action is `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`:
//! the kinked charge needs one aligned law of benefit and harm. Each outcome is estimated by
//! a Bayesian dose-response study of its own table, so the law is the propagation chain's
//! family table matched to each study's posterior moments.
//!
//! * The chain is declared and checked by the chain rule at compile: the compiled decision
//!   and every study's plan state its inputs (one draw source per coordinate of each study,
//!   with the study's data snapshot as its row factor), the declared dependence and the joint
//!   outputs, before anything is fitted. Its refusals (no declared dependence, a study whose
//!   law the chain cannot read as a posterior) arise at compile.
//! * Two outcomes of one table share every row. The evidence ledger finds the overlap at
//!   compile: under distinct evidence the studies are refused; under shared evidence the
//!   declared cross-study covariance is required for the pair and is the cross block of the
//!   chain's declared covariance, which the drawn law then carries exactly.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute, CrossStudyCovariance,
    DecisionCriterion, OutcomeDependence, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceSet, SharedEvidence, StructuralPolicy, ValueModel,
    compile_preference_decision_across_studies, compile_preference_decision_with_shared_evidence,
    shape,
};
use antecedent::reason::reason_details::preference_compile;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::propagation_chain::StageKind;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::TabularData;
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

fn quantity(variable: &str, units: &str, dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "patients".into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`.
fn kinked(cap: f64) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: PreferenceSet {
            parameters: vec![
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
            relations: vec![],
        },
        value: PreferenceExpr::difference(
            PreferenceExpr::difference(
                PreferenceExpr::Attribute(0),
                PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
            ),
            PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, cap)),
        ),
        utility_units: "points".into(),
    }
}

fn contract(cap: f64) -> PreferenceContract {
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", dose)),
            ActionInput::Quantity(quantity("h", "events", dose)),
        ],
    };
    AcceptedPreferences::declared(kinked(cap)).unwrap().contract(PreferenceContractParts {
        actions: vec![action("high", "1.5"), action("low", "0.5")],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "patients".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn dose_table(outcome: &str, rows: u32, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
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

fn dose_study(data: &TabularData, variables: u32, outcome: u32, inference: InferenceMode) -> Study {
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
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

fn column(law: &DistributionArtifact, j: usize) -> Vec<f64> {
    let [n, width] = law.shape();
    (0..n).map(|draw| law.draws()[draw * width + j]).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap())
}

fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>()
        / f64::from(u32::try_from(x.len()).unwrap())
}

#[test]
fn the_chain_that_joins_separate_studies_is_declared_and_checked_at_compile() {
    let ctx = ExecutionContext::for_tests(53);
    let benefit = dose_study(&dose_table("b", 220, (2.0, 1.2), 1.7), 2, 1, bayes());
    let harm = dose_study(&dose_table("h", 260, (0.5, 0.5), 2.9), 2, 1, bayes());
    let distinct = OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 };
    let compiled = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        distinct,
        &ctx,
    )
    .unwrap();

    // Before fitting: the chain's inputs, dependence, outputs and the rule's stage plans.
    let joint = compiled.joint_law().expect("a declared joint law");
    let fact = joint.fact();
    assert_eq!(fact.dependence(), "distinct_evidence");
    assert_eq!((fact.draws(), fact.seed()), (2000, 7));
    assert_eq!(fact.reading(), "posterior");
    let declared: Vec<String> = fact.inputs().iter().map(|i| i.quantity().to_owned()).collect();
    let expected: Vec<String> = [
        quantity("b", "points", "0.5"),
        quantity("b", "points", "1.5"),
        quantity("h", "events", "0.5"),
        quantity("h", "events", "1.5"),
    ]
    .iter()
    .map(ScientificQuantity::canonical_identity)
    .collect();
    assert_eq!(declared, expected);
    assert_eq!(fact.inputs()[2].study(), 1);
    assert_ne!(fact.inputs()[0].factor(), fact.inputs()[2].factor(), "distinct snapshots");
    assert_eq!(joint.plan().stages.len(), 4);
    assert!(joint.plan().stages.iter().all(|stage| stage.kind == StageKind::DerivedQuantity));
    for view in compiled.plan_views() {
        let json = view.to_json();
        let stated = &json["evidence"]["decision"]["preference"]["joint_law"];
        assert_eq!(stated["dependence"], "distinct_evidence");
        assert_eq!(stated["inputs"].as_array().unwrap().len(), 4);
        assert_eq!(stated["outputs"][3]["quantity"], expected[3].as_str());
        assert_eq!(stated["declaration"], fact.declaration());
    }

    // Another seed is another declaration, so another program.
    let reseeded = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        OutcomeDependence::DistinctEvidence { draws: 2000, seed: 8 },
        &ctx,
    )
    .unwrap();
    assert_ne!(reseeded.plan_views()[0].identity(), compiled.plan_views()[0].identity());

    // The execution draws the declared outputs, with each study's own moments.
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("one aligned law");
    };
    assert_eq!(law.shape(), [2000, 4]);
    for j in 0..4 {
        let NativeDecisionSource::JointLaw(own) = &executed.inputs[j / 2].source else {
            panic!("each study's own posterior draws");
        };
        close(mean(&column(law, j)), mean(&column(own, j % 2)), 1e-9);
    }
    let by_hand =
        pd::decide(&contract(1.2), executed.source.as_ref().unwrap(), None, options()).unwrap();
    assert_eq!(executed.evaluation.single(), Some(&by_hand));

    // Refused at compile: no declared dependence, and a study whose law is not a posterior.
    let refused = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        OutcomeDependence::Undeclared,
        &ctx,
    )
    .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!((refusal.code, refusal.stage), ("joint_law_required", "compile"));
    let frequentist =
        dose_study(&dose_table("h", 260, (0.5, 0.5), 2.9), 2, 1, InferenceMode::Frequentist);
    let refused = compile_preference_decision_across_studies(
        &[&benefit, &frequentist],
        &contract(1.2),
        distinct,
        &ctx,
    )
    .unwrap_err();
    assert!(refused.compiler().is_some(), "the carrier refusal of the compiler: {refused}");
}

#[test]
#[allow(clippy::too_many_lines, reason = "one study pair walks every shared-evidence case")]
fn studies_that_share_observations_need_a_declared_cross_study_covariance() {
    let ctx = ExecutionContext::for_tests(61);
    let table = shared_table();
    let benefit = dose_study(&table, 3, 1, bayes());
    let harm = dose_study(&table, 3, 2, bayes());
    let studies = [&benefit, &harm];

    // Distinct evidence over one table: the ledger finds every row shared.
    let refused = compile_preference_decision_across_studies(
        &studies,
        &contract(1.2),
        OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 },
        &ctx,
    )
    .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!((refusal.code, refusal.stage), ("evidence_double_counted", "compile"));
    assert_eq!(refusal.detail, preference_compile::STUDIES_SHARE_EVIDENCE);
    assert_eq!(refusal.supplied.as_deref(), Some("200 shared observations"));

    // Shared evidence with the block missing, misshapen, or zero.
    for cross in [
        vec![],
        vec![CrossStudyCovariance { first: 0, second: 1, covariance: vec![0.01; 3] }],
        vec![CrossStudyCovariance { first: 0, second: 1, covariance: vec![0.0; 4] }],
    ] {
        let missing = cross.is_empty();
        let shared = SharedEvidence { draws: 2000, seed: 7, cross };
        let refused = compile_preference_decision_with_shared_evidence(
            &studies,
            &contract(1.2),
            &shared,
            &ctx,
        )
        .unwrap_err();
        let refusal = refused.refusal().expect("a compile refusal");
        assert_eq!(refusal.stage, "compile");
        let detail = if missing {
            preference_compile::STUDIES_SHARE_EVIDENCE
        } else {
            preference_compile::CROSS_STUDY_COVARIANCE_INVALID
        };
        assert_eq!(refusal.detail, detail);
    }

    // A block for studies of distinct tables names an overlap that is not there.
    let other = dose_study(&dose_table("h", 260, (0.5, 0.5), 2.9), 2, 1, bayes());
    let shared = SharedEvidence {
        draws: 2000,
        seed: 7,
        cross: vec![CrossStudyCovariance { first: 0, second: 1, covariance: vec![0.01; 4] }],
    };
    let refused = compile_preference_decision_with_shared_evidence(
        &[&benefit, &other],
        &contract(1.2),
        &shared,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        refused.refusal().map(|r| r.detail.as_str()),
        Some(preference_compile::CROSS_STUDY_COVARIANCE_INVALID)
    );

    // The declared block is the cross block of the drawn law. Its terms are a fifth of the
    // product of the two studies' posterior spreads, read from the studies' own draws.
    let separate_benefit = dose_study(&dose_table("b", 220, (2.0, 1.2), 1.7), 2, 1, bayes());
    let own = |studies: [&Study; 2], index: usize| {
        let probe = compile_preference_decision_across_studies(
            &studies,
            &contract(1.2),
            OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 },
            &ctx,
        )
        .unwrap();
        let executed = probe.execute(None, options(), &ctx).unwrap();
        let NativeDecisionSource::JointLaw(law) = &executed.inputs[index].source else {
            panic!("posterior draws");
        };
        law.as_ref().clone()
    };
    let benefit_law = own([&benefit, &other], 0);
    let harm_law = own([&separate_benefit, &harm], 1);
    let spread =
        |law: &DistributionArtifact, j: usize| covariance(&column(law, j), &column(law, j)).sqrt();
    let block: Vec<f64> = [(0, 0), (0, 1), (1, 0), (1, 1)]
        .iter()
        .map(|&(p, q)| 0.2 * spread(&benefit_law, p) * spread(&harm_law, q))
        .collect();
    let shared = SharedEvidence {
        draws: 4000,
        seed: 11,
        cross: vec![CrossStudyCovariance { first: 0, second: 1, covariance: block.clone() }],
    };
    let compiled =
        compile_preference_decision_with_shared_evidence(&studies, &contract(1.2), &shared, &ctx)
            .unwrap();
    assert_eq!(compiled.dependence(), OutcomeDependence::SharedEvidence { draws: 4000, seed: 11 });
    assert_eq!(compiled.cross_study_covariance(), &shared.cross[..]);
    let fact = compiled.joint_law().unwrap().fact();
    assert_eq!(fact.dependence(), "shared_evidence");
    assert_eq!(fact.cross_blocks(), &[(0, 1)]);
    assert_eq!(fact.inputs()[0].factor(), fact.inputs()[2].factor(), "one snapshot");
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("one aligned law");
    };
    for (index, (p, q)) in [(0, 2), (0, 3), (1, 2), (1, 3)].into_iter().enumerate() {
        close(covariance(&column(law, p), &column(law, q)), block[index], 1e-6);
    }
    let by_hand =
        pd::decide(&contract(1.2), executed.source.as_ref().unwrap(), None, options()).unwrap();
    assert_eq!(executed.evaluation.single(), Some(&by_hand));
}
