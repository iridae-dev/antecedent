//! Trade-offs between several estimated outcomes, decided across a range of preferences.
//!
//! The value of an action is `benefit - harm_rate * harm`, both outcomes estimated, with the
//! harm rate (points per event) a declared range. Between a treatment and its comparator the
//! value difference is `d_benefit - harm_rate * d_harm`, so with both differences positive the
//! treatment leads exactly when `harm_rate < d_benefit / d_harm`: a range left of that ratio
//! makes the treatment best everywhere, a range right of it makes the comparator best
//! everywhere, and a range around it leaves the choice to the harm rate, switching at the ratio.
//!
//! Several outcomes of one study. A cluster-randomized experiment measures benefit and harm on
//! every unit; the treatment raises benefit by 1 and harm by 0.4 per row, so the ratio is near
//! 2.5. The experiment's arm means are a mean source over both outcomes: the affine value is
//! decided exactly, and a kinked harm charge or a probability constraint, which need a joint
//! law, are refused by the mean-source rule.
//!
//! Separately fitted outcomes. Benefit and harm are each estimated by a Bayesian dose-response
//! analysis of its own table (benefit `2 + 1.2 t`, harm `0.5 + 0.5 t`, doses 0.5 and 1.5, ratio
//! near 2.4). Their draws are separate laws, and the joint-law rule refuses to pair their rows.
//! A propagation-chain family table joins the four posterior summaries under a declared
//! dependence (each study's own covariance within it, distinct evidence between them) into one
//! aligned joint law, on which the kinked charge and a cap on the probability of harm are
//! decided.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute, DecisionCriterion,
    HardConstraint, PreferenceAction, PreferenceContract, PreferenceContractParts,
    PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter, PreferencePolicy,
    PreferenceSet, PreferenceVerdict, StructuralPolicy, ValueModel, shape,
};
use antecedent::analysis::propagation_chain::{
    FamilyDependence, FamilyEstimate, compile_propagation_chain, family_table,
};
use antecedent::decision_evidence::{
    Exposure, JointExperimentEvidence, OutcomeComponent, complete_cluster_joint_contrast,
    evidence_from_cluster_contrast,
};
use antecedent::design::decision_contract::{SourceRepresentation, SourceRequirement, UtilityExpr};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::TableConstruction;
use antecedent_contract::propagation_chain::{ChainReading, LawMeaning, StageKind};
use antecedent_contract::{ProgramBinding, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::distribution_artifact::{DistributionArtifact, DistributionCalibration};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

const ROWS_PER_CLUSTER: u32 = 12;
const CLUSTERS: u32 = 80;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

/// Deterministic noise in `[-1, 1]`.
fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn attributes() -> Vec<Attribute> {
    vec![
        Attribute { id: "benefit".into(), units: "points".into() },
        Attribute { id: "harm".into(), units: "events".into() },
    ]
}

/// `benefit - harm_rate * harm`.
fn traded(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet {
            parameters: vec![PreferenceParameter {
                id: "harm_rate".into(),
                units: "points/events".into(),
                lower,
                upper,
                assumption_id: "assumption.price_of_harm".into(),
            }],
            relations: vec![],
        },
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

fn action(id: &str, benefit: &ScientificQuantity, harm: &ScientificQuantity) -> PreferenceAction {
    PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![ActionInput::Quantity(benefit.clone()), ActionInput::Quantity(harm.clone())],
    }
}

fn contract(
    model: ValueModel,
    actions: Vec<PreferenceAction>,
    constraints: Vec<HardConstraint>,
    population: &str,
    horizon: u32,
) -> PreferenceContract {
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints,
        target_population: population.into(),
        horizon,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn verdict(contract: &PreferenceContract, source: &NativeDecisionSource) -> PreferenceVerdict {
    pd::decide(contract, source, None, options()).unwrap().exact().unwrap().verdict.clone()
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

// ------------------------------------------------------------ one study, several outcomes

fn experiment() -> JointExperimentEvidence {
    let assigned: Vec<bool> = (0..CLUSTERS).map(|c| c % 2 == 0).collect();
    let (mut benefit, mut harm) = (Vec::new(), Vec::new());
    for (c, treated) in (0..CLUSTERS).zip(&assigned) {
        let shock = 0.5 * wobble(c + 1, 12.9898);
        let arm = if *treated { 1.0 } else { 0.0 };
        let (mut b, mut h) = (0.0, 0.0);
        for r in 0..ROWS_PER_CLUSTER {
            let noise = 0.2 * wobble(c * 31 + r * 17, 0.7);
            b += 5.0 + arm + shock + noise;
            h += 1.0 + 0.4 * arm + 0.4 * shock + 0.5 * noise;
        }
        benefit.push(b);
        harm.push(h);
    }
    let rows = usize::try_from(CLUSTERS * ROWS_PER_CLUSTER).unwrap();
    let contrast =
        complete_cluster_joint_contrast(&[benefit.as_slice(), harm.as_slice()], &assigned, rows)
            .unwrap();
    evidence_from_cluster_contrast(
        &contrast,
        "trial-protocol",
        "patients",
        ("new_protocol", "standard_protocol"),
        0,
        vec![
            OutcomeComponent {
                name: "benefit".into(),
                variable_id: "outcome.benefit".into(),
                units: "points".into(),
            },
            OutcomeComponent {
                name: "harm".into(),
                variable_id: "outcome.harm".into(),
                units: "events".into(),
            },
        ],
        rows,
        vec![],
        Exposure::NoBetweenUnitInterference {
            assumption_id: "assumption.no_cross_patient_spillover".into(),
        },
    )
    .unwrap()
}

#[test]
fn an_experiment_with_two_outcomes_decides_the_harm_trade_off_on_its_arm_means() {
    let evidence = experiment();
    let arm = |component: &str, treated: bool| {
        pd::experiment_arm_quantity(&evidence, component, treated).unwrap()
    };
    let actions = vec![
        action("high", &arm("benefit", true), &arm("harm", true)),
        action("low", &arm("benefit", false), &arm("harm", false)),
    ];
    let source = NativeDecisionSource::Mean(pd::experiment_mean_source(&evidence).unwrap());
    let gain = evidence.treated_means[0] - evidence.control_means[0];
    let extra_harm = evidence.treated_means[1] - evidence.control_means[1];
    assert!(gain > 0.5 && extra_harm > 0.2, "{gain} {extra_harm}");
    let ratio = gain / extra_harm;

    check_switch(
        |lower, upper| {
            verdict(
                &contract(traded(lower, upper), actions.clone(), vec![], "patients", 0),
                &source,
            )
        },
        ratio,
    );
    let around = contract(traded(0.5 * ratio, 1.5 * ratio), actions.clone(), vec![], "patients", 0);
    let decided = pd::decide(&around, &source, None, options()).unwrap();
    let switch = &decided.exact().unwrap().switches[0];
    // `v_high - v_low = gain - harm_rate * extra_harm`.
    close(switch.margin_coefficients[0], gain, 1e-9);
    close(switch.margin_coefficients[1], -extra_harm, 1e-9);
    close(-switch.margin_coefficients[0] / switch.margin_coefficients[1], ratio, 1e-9);

    // A kinked harm charge and a probability constraint need a joint law, not arm means.
    let cap = evidence.treated_means[1];
    let charged =
        contract(kinked(cap, (0.2 * ratio, 0.8 * ratio)), actions.clone(), vec![], "patients", 0);
    assert!(pd::decide(&charged, &source, None, options()).is_err());
    let capped =
        contract(traded(0.2 * ratio, 0.8 * ratio), actions, vec![harm_cap(cap)], "patients", 0);
    assert!(pd::decide(&capped, &source, None, options()).is_err());

    // Arm means under bounded spillover are assumption ranges, not means.
    let mut bounded = evidence.clone();
    bounded.exposure = Exposure::BoundedSpillover {
        assumption_id: "assumption.bounded_spillover".into(),
        bias_bounds: vec![0.1, 0.1],
    };
    assert!(pd::experiment_mean_source(&bounded).is_err());
}

fn harm_cap(cap: f64) -> HardConstraint {
    HardConstraint {
        id: "harm_cap".into(),
        expr: UtilityExpr::Input(1),
        bound: cap,
        min_probability: 0.9,
        units: "events".into(),
        applies_to: vec!["high".into()],
    }
}

// ------------------------------------------------------- separately fitted outcomes

const LOW_DOSE: f64 = 0.5;
const HIGH_DOSE: f64 = 1.5;

fn dose_table(outcome: &str, rows: u32, intercept: f64, slope: f64, rate: f64) -> TabularData {
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

fn context(outcome: &str, units: &str) -> NativeResponseContext {
    NativeResponseContext {
        program: ProgramBinding {
            graph_id: format!("graph:dose-{outcome}"),
            contract_id: format!("contract:dose-{outcome}"),
            treatment_id: "t".into(),
            outcome_id: outcome.into(),
            population_id: "patients".into(),
            intervention_kind: "do".into(),
            horizon: 0,
            dose_grid: vec![LOW_DOSE, HIGH_DOSE],
            dose_units: "mg".into(),
            outcome_units: units.into(),
            functional_id: "mean".into(),
            transform_id: "identity".into(),
        },
        snapshot_id: format!("snapshot:{outcome}"),
        rng_id: "pcg:seeded".into(),
        calibration: DistributionCalibration::NotMeasured,
    }
}

/// A Bayesian dose-response analysis of one outcome, as a claim.
fn analysed(data: &TabularData, outcome: &str, units: &str, seed: u64) -> NativeResponseClaim {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([LOW_DOSE, HIGH_DOSE])),
        ),
    }));
    let result = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let response = result.response().expect("a mean response curve");
    NativeResponseClaim::from_response(response, data.schema(), &context(outcome, units)).unwrap()
}

fn law_of(claim: &NativeResponseClaim) -> DistributionArtifact {
    let joint = SourceRequirement {
        any_of: vec![SourceRepresentation::JointDraws],
        sampled_needs_error_receipt: false,
    };
    let NativeDecisionSource::JointLaw(law) = claim.decision_source(&joint).unwrap().source else {
        panic!("a Bayesian curve retains aligned draws");
    };
    *law
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
    mean(&x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).collect::<Vec<_>>())
}

/// Four posterior summaries joined by a family table: each study's own covariance between its
/// two doses, nothing between the studies, which rest on distinct rows.
fn chain_law(
    benefit: &NativeResponseClaim,
    harm: &NativeResponseClaim,
    ctx: &ExecutionContext,
) -> DistributionArtifact {
    let studies =
        [(benefit, "benefit", "benefit-trial-rows"), (harm, "harm", "harm-registry-rows")];
    let mut estimates = Vec::new();
    let mut columns = Vec::new();
    let mut quantities = Vec::new();
    for (claim, name, rows) in studies {
        let law = law_of(claim);
        for (j, dose) in ["low", "high"].into_iter().enumerate() {
            let values = column(&law, j);
            estimates.push(
                FamilyEstimate::new(
                    &format!("{name}_{dose}_source"),
                    mean(&values),
                    covariance(&values, &values).sqrt(),
                    LawMeaning::Posterior,
                )
                .with_factor(SharedFactorKind::Row, rows),
            );
            columns.push((name, values));
            quantities.push((format!("{name}_{dose}"), claim.coordinates()[j].clone()));
        }
    }
    let k = columns.len();
    let mut declared = vec![0.0; k * k];
    for (i, (left, x)) in columns.iter().enumerate() {
        for (j, (right, y)) in columns.iter().enumerate() {
            if left == right {
                declared[i * k + j] = covariance(x, y);
            }
        }
    }
    let table = family_table(
        "dosing:declared-joint",
        &estimates,
        &FamilyDependence::Covariance(declared),
        None,
        2000,
        7,
    )
    .unwrap();
    let stages = quantities
        .iter()
        .map(|(stage, _)| {
            ChainStageWire::new(
                stage,
                StageKind::DerivedQuantity,
                &[format!("{stage}_source").as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        })
        .collect();
    let draft = table
        .draft(ChainReading::Posterior, &quantities[0].0, stages)
        .with_output_quantity(ScientificQuantityWire::from(&quantities[0].1))
        .with_joint_outputs(
            quantities
                .iter()
                .map(|(stage, quantity)| ChainJointOutputWire {
                    stage: stage.clone(),
                    quantity: ScientificQuantityWire::from(quantity),
                })
                .collect(),
        );
    let chain = compile_propagation_chain(draft, ctx).unwrap().execute(ctx).unwrap();
    pd::chain_joint_law(&chain).unwrap()
}

#[test]
fn separately_fitted_outcomes_meet_through_a_declared_dependence() {
    let ctx = ExecutionContext::for_tests(53);
    let benefit_data = dose_table("b", 220, 2.0, 1.2, 1.7);
    let harm_data = dose_table("h", 260, 0.5, 0.5, 2.9);
    let benefit = analysed(&benefit_data, "b", "points", 61);
    let harm = analysed(&harm_data, "h", "events", 67);
    let actions = vec![
        action("high", &benefit.coordinates()[1], &harm.coordinates()[1]),
        action("low", &benefit.coordinates()[0], &harm.coordinates()[0]),
    ];

    // The two analyses' draws are separate laws: their rows are never paired.
    let refused = pd::joint_law_of(&[&law_of(&benefit), &law_of(&harm)]).unwrap_err();
    assert_eq!(refused.code, "joint_law_required");
    assert!(refused.detail.ends_with("alignment_mismatch"), "{}", refused.detail);
    let one = pd::joint_law_of(&[&law_of(&benefit)]).unwrap();
    assert_eq!(one, law_of(&benefit));

    let law = chain_law(&benefit, &harm, &ctx);
    assert_eq!(law.shape()[1], 4);
    let source = NativeDecisionSource::JointLaw(Box::new(law.clone()));
    let gain = mean(&column(&law, 1)) - mean(&column(&law, 0));
    let extra_harm = mean(&column(&law, 3)) - mean(&column(&law, 2));
    assert!(gain > 0.6 && extra_harm > 0.2, "{gain} {extra_harm}");
    let ratio = gain / extra_harm;
    check_switch(
        |lower, upper| {
            verdict(
                &contract(traded(lower, upper), actions.clone(), vec![], "patients", 0),
                &source,
            )
        },
        ratio,
    );

    // A kinked charge on harm above a cap reads the joint law: its expectation over the draws
    // exceeds the charge at the mean.
    let harm_high = column(&law, 3);
    let cap = mean(&harm_high);
    let charged =
        contract(kinked(cap, (0.2 * ratio, 0.8 * ratio)), actions.clone(), vec![], "patients", 0);
    let decided = pd::decide(&charged, &source, None, options()).unwrap();
    let high = &decided.exact().unwrap().actions[0];
    let expected_excess = mean(&harm_high.iter().map(|h| (h - cap).max(0.0)).collect::<Vec<_>>());
    assert!(expected_excess > 1e-6, "{expected_excess}");
    close(high.coefficients.as_ref().unwrap()[2], -expected_excess, 1e-9);

    // A cap on the probability of harm excludes the treatment whatever the harm rate.
    let capped = contract(
        traded(0.2 * ratio, 0.8 * ratio),
        actions,
        vec![harm_cap(cap - 0.3)],
        "patients",
        0,
    );
    let decided = pd::decide(&capped, &source, None, options()).unwrap();
    let exact = decided.exact().unwrap();
    assert!(!exact.actions[0].admissible);
    assert_eq!(exact.verdict, PreferenceVerdict::NecessarilyBest("low".into()));
}

/// The `outcome` coordinate of `variable` under `regime`.
fn outcome_quantity(variable: &str, units: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: antecedent_contract::QuantityRole::Outcome,
        units: units.into(),
        population_id: "patients".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

#[test]
fn a_predictive_chain_answers_outcome_inputs() {
    let ctx = ExecutionContext::for_tests(59);
    // Aligned predictive draws of each outcome under each action: one row per simulated
    // patient, the treated rows sharing a disturbance with the untreated ones.
    let n = 400_u32;
    let shock: Vec<f64> = (0..n).map(|i| wobble(i + 1, 12.9898)).collect();
    let outputs = [
        ("treat_benefit", 4.0, 1.0, outcome_quantity("benefit", "points", "do(t=1)")),
        ("treat_harm", 2.0, 0.5, outcome_quantity("harm", "events", "do(t=1)")),
        ("wait_benefit", 2.0, 1.0, outcome_quantity("benefit", "points", "do(t=0)")),
        ("wait_harm", 1.0, 0.5, outcome_quantity("harm", "events", "do(t=0)")),
    ];
    let sources = outputs
        .iter()
        .zip(0_u32..)
        .map(|((stage, centre, spread, _), k)| {
            let values: Vec<f64> = (0..n)
                .zip(&shock)
                .map(|(i, s)| centre + spread * (s + 0.3 * wobble(i * 7 + 3 * k, 0.9)))
                .collect();
            ChainSourceWire::draws(
                &format!("{stage}_draws"),
                LawMeaning::Predictive,
                "joint:trial",
                &values,
            )
            .with_factor(SharedFactorKind::Disturbance, "trial-shock")
        })
        .collect();
    let stages = outputs
        .iter()
        .map(|(stage, ..)| {
            ChainStageWire::new(
                stage,
                StageKind::DerivedQuantity,
                &[format!("{stage}_draws").as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        })
        .collect();
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Predictive,
        outputs[0].0,
        sources,
        stages,
    )
    .with_output_quantity(ScientificQuantityWire::from(&outputs[0].3))
    .with_joint_outputs(
        outputs
            .iter()
            .map(|(stage, _, _, quantity)| ChainJointOutputWire {
                stage: (*stage).into(),
                quantity: ScientificQuantityWire::from(quantity),
            })
            .collect(),
    );
    let chain = compile_propagation_chain(draft, &ctx).unwrap().execute(&ctx).unwrap();
    let law = pd::chain_joint_law(&chain).unwrap();
    let source = NativeDecisionSource::JointLaw(Box::new(law.clone()));
    let actions = vec![
        action("high", &outputs[0].3, &outputs[1].3),
        action("low", &outputs[2].3, &outputs[3].3),
    ];
    let gain = mean(&column(&law, 0)) - mean(&column(&law, 2));
    let extra_harm = mean(&column(&law, 1)) - mean(&column(&law, 3));
    assert!(gain > 1.5 && extra_harm > 0.7, "{gain} {extra_harm}");
    check_switch(
        |lower, upper| {
            verdict(
                &contract(traded(lower, upper), actions.clone(), vec![], "patients", 0),
                &source,
            )
        },
        gain / extra_harm,
    );
}
