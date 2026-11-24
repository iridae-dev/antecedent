//! One entry point over the exact and certified verdicts: the route rule and its labels, the
//! fall-through from the exact route, mean sources, accepted preferences of every basis decided
//! end to end, a cross-check of the two routes on random affine problems, and fixed trade-offs
//! decided as the hand-written ordinary contract under every criterion, constraint and
//! structural policy, with the documented tie rules where the two differ.
//!
//! The standard law has `E[benefit, cost] = (10, 4)` under `A` and `(5.5, 1)` under `B`, and the
//! value is `benefit - w * cost`, so `v_A - v_B = 4.5 - 3w`: `A` leads below `w = 1.5` and `B`
//! above it. Over the declared `w` in `[0, 3]` the choice depends on preferences; the records
//! `r1` (`w <= 1`) and `r2` (`w >= 1/2`) narrow the set to `[1/2, 1]`, where `A` is best.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::accepted_preferences::{
    AcceptedPreferences, PreferenceContractParts, estimate_accepted, learn_compatible_set_accepted,
};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionResult, MeanEstimate, MeanSource, Verdict, evaluate_contract,
    evaluate_contract_on_means,
};
use antecedent_design::decision_structural::{
    AtomEvidence, StructuralAtom, StructuralDecisionResult, evaluate_structural,
};
use antecedent_design::preference_certified::{
    CertifiedAtomStatus, CertifiedOptions, CertifiedPreferenceResult, CertifiedVerdict,
    RegionStatus, StructuralCertifiedResult, StructuralCertifiedVerdict,
    evaluate_preference_certified,
};
use antecedent_design::preference_choice_model::ChoiceModelOptions;
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{
    MeasureOrigin, PreferenceAtomStatus, PreferenceDecisionError, PreferenceDecisionResult,
    PreferenceVerdict, StructuralPreferenceResult, StructuralPreferenceVerdict,
};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationError as E, PreferenceEvaluationOptions,
    PreferenceRoute, ROUTE_AFFINE_EXPECTATION, ROUTE_CRITERION_NOT_AFFINE,
    ROUTE_PART_NOT_LOWERABLE, ROUTE_SHAPE_PARAMETERS, RouteChoice, StructuralPreferenceEvaluation,
    decide_preferences, decide_preferences_on_means, decide_preferences_structural,
    decide_with_accepted, preference_route,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as P, PreferenceParameter, PreferenceSet, ValueModel,
};
use antecedent_design::preference_set_learning::SetLearningOptions;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

/// Small fixed-seed xorshift64* generator.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in the open interval `(0, 1)`.
    #[allow(clippy::cast_precision_loss, reason = "53-bit integers are exact in f64")]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn gumbel(&mut self) -> f64 {
        -(-self.uniform().ln()).ln()
    }
}

#[allow(clippy::cast_precision_loss, reason = "test counts are far below 2^53")]
fn as_f64(n: usize) -> f64 {
    n as f64
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "events/usd".to_owned(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.to_owned(), units: units.to_owned() }
}

fn two_attribute_model(lower: f64, upper: f64, value: P) -> ValueModel {
    ValueModel {
        attributes: vec![attribute("benefit", "events"), attribute("cost", "usd")],
        preferences: PreferenceSet::new(vec![parameter("w", lower, upper)], vec![]),
        value,
        utility_units: "events".to_owned(),
    }
}

/// `benefit - w * cost`.
fn cost_model(lower: f64, upper: f64) -> ValueModel {
    two_attribute_model(
        lower,
        upper,
        P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
    )
}

/// `benefit - w * w * cost`: `w` is a shape parameter.
fn squared_model() -> ValueModel {
    two_attribute_model(
        0.0,
        3.0,
        P::difference(
            P::Attribute(0),
            P::product(P::product(P::Parameter(0), P::Parameter(0)), P::Attribute(1)),
        ),
    )
}

/// `benefit - w * exp(cost)`: affine in `w`, but the part `exp(cost)` cannot be lowered.
fn exponential_model() -> ValueModel {
    two_attribute_model(
        0.0,
        0.1,
        P::difference(P::Attribute(0), P::product(P::Parameter(0), P::exp(P::Attribute(1)))),
    )
}

fn quantity(attribute: &Attribute, action: usize) -> ScientificQuantity {
    ScientificQuantity {
        units: attribute.units.clone(),
        regime_id: format!("do(a={action})"),
        ..common::quantity(&attribute.id)
    }
}

fn parts(
    model: &ValueModel,
    ids: &[&str],
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContractParts {
    PreferenceContractParts {
        actions: ids
            .iter()
            .enumerate()
            .map(|(a, id)| PreferenceAction {
                id: (*id).to_owned(),
                kind: ActionKind::Intervention,
                inputs: model
                    .attributes
                    .iter()
                    .map(|attribute| ActionInput::Quantity(quantity(attribute, a)))
                    .collect(),
            })
            .collect(),
        criterion,
        constraints: vec![],
        target_population: "target".to_owned(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: policy,
    }
}

fn contract(
    model: ValueModel,
    criterion: DecisionCriterion,
    policy: PreferencePolicy,
) -> PreferenceContract {
    let parts = parts(&model, &["A", "B"], criterion, policy);
    PreferenceContract {
        model,
        actions: parts.actions,
        criterion: parts.criterion,
        constraints: parts.constraints,
        target_population: parts.target_population,
        horizon: parts.horizon,
        structural_policy: parts.structural_policy,
        preference_policy: parts.preference_policy,
    }
}

fn eu() -> DecisionCriterion {
    DecisionCriterion::PosteriorExpectedUtility
}

/// Joint draws: `arms[a][k]` holds the draws of attribute `k` under action `a`.
fn law(
    model: &ValueModel,
    arms: &[Vec<Vec<f64>>],
    weights: Option<Vec<f64>>,
) -> DistributionArtifact {
    let mut columns = Vec::new();
    for a in 0..arms.len() {
        for attribute in &model.attributes {
            columns.push(quantity(attribute, a));
        }
    }
    let n = arms[0][0].len();
    let mut draws = Vec::new();
    for i in 0..n {
        for arm in arms {
            for attribute in arm {
                draws.push(attribute[i]);
            }
        }
    }
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [n, columns.len()],
            weights,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// `E[benefit, cost]` is `(10, 4)` under `A` and `(5.5, 1)` under `B`.
fn standard_law(model: &ValueModel) -> DistributionArtifact {
    let arms = vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
    ];
    law(model, &arms, Some(vec![1.0, 1.0, 2.0]))
}

fn mean_source(model: &ValueModel) -> MeanSource {
    let coordinates = (0..2)
        .flat_map(|a| {
            model
                .attributes
                .iter()
                .map(move |k| ScientificQuantity { functional_id: "mean".into(), ..quantity(k, a) })
        })
        .collect();
    MeanSource {
        coordinates,
        means: vec![10.0, 4.0, 5.5, 1.0],
        estimate: MeanEstimate::Posterior,
        provider_id: "lab".into(),
        snapshot_id: "snap-1".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    }
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

/// The same contract reading the mean of each quantity, which a mean source can supply.
fn reading_means(mut contract: PreferenceContract) -> PreferenceContract {
    for action in &mut contract.actions {
        for input in &mut action.inputs {
            if let ActionInput::Quantity(quantity) = input {
                quantity.functional_id = "mean".into();
            }
        }
    }
    contract
}

fn record(id: &str, a: [f64; 2], b: [f64; 2], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: "panel".to_owned(),
        options: [("a", a), ("b", b)]
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).to_owned(),
                belief: OptionBelief::Means(means.to_vec()),
            })
            .collect(),
        chosen: chosen.to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: format!("forecast-{id}") },
        observation_rows: vec![format!("{id}#row")],
    }
}

fn record_set(model: &ValueModel, records: Vec<ChoiceRecord>) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "review panel".to_owned(),
        records,
        pooling: Pooling::Pooled { assumption_id: "assumption.one_panel".to_owned() },
    }
}

/// `w <= 1` from `r1` and `w >= 1/2` from `r2`.
fn settling_records(model: &ValueModel) -> ChoiceRecordSet {
    record_set(
        model,
        vec![record("r1", [1.0, 1.0], [0.0, 0.0], "a"), record("r2", [1.0, 2.0], [0.0, 0.0], "b")],
    )
}

/// Logit choices with consistency one and value `benefit - cost`.
fn simulated(model: &ValueModel, n: usize, seed: u64) -> ChoiceRecordSet {
    let mut rng = Xorshift::new(seed);
    let records = (0..n)
        .map(|r| {
            let options: Vec<[f64; 2]> =
                (0..3).map(|_| [4.0 * rng.uniform(), 4.0 * rng.uniform()]).collect();
            let chosen = options
                .iter()
                .map(|o| o[0] - o[1] + rng.gumbel())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(0, |(i, _)| i);
            ChoiceRecord {
                id: format!("s{r}"),
                decider: "panel".to_owned(),
                options: options
                    .iter()
                    .enumerate()
                    .map(|(i, means)| ChoiceOption {
                        id: format!("o{i}"),
                        belief: OptionBelief::Means(means.to_vec()),
                    })
                    .collect(),
                chosen: format!("o{chosen}"),
                belief_source: BeliefSource::RecordedForecast { source_id: "forecast".into() },
                observation_rows: vec![format!("s{r}#row")],
            }
        })
        .collect();
    record_set(model, records)
}

fn exact_verdict(evaluation: &PreferenceEvaluation) -> &PreferenceVerdict {
    &evaluation.exact().expect("exact route").verdict
}

#[test]
fn the_route_follows_the_value_and_the_criterion() {
    let affine = contract(cost_model(0.0, 3.0), eu(), PreferencePolicy::ReportOnly);
    assert_eq!(
        preference_route(&affine).unwrap(),
        RouteChoice { route: PreferenceRoute::Exact, reason: ROUTE_AFFINE_EXPECTATION }
    );
    for criterion in [DecisionCriterion::PosteriorExpectedLoss, DecisionCriterion::ExpectedRegret] {
        let c = contract(cost_model(0.0, 3.0), criterion, PreferencePolicy::ReportOnly);
        assert_eq!(preference_route(&c).unwrap().route, PreferenceRoute::Exact);
    }
    for criterion in [
        DecisionCriterion::ThresholdProbability { threshold: 5.0 },
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::Regret,
    ] {
        let c = contract(cost_model(0.0, 3.0), criterion, PreferencePolicy::ReportOnly);
        assert_eq!(
            preference_route(&c).unwrap(),
            RouteChoice { route: PreferenceRoute::Certified, reason: ROUTE_CRITERION_NOT_AFFINE }
        );
    }
    let shaped = contract(squared_model(), eu(), PreferencePolicy::ReportOnly);
    assert_eq!(
        preference_route(&shaped).unwrap(),
        RouteChoice { route: PreferenceRoute::Certified, reason: ROUTE_SHAPE_PARAMETERS }
    );
    // The exponential model is affine in `w`; the exact evaluator sends it on.
    let exponential = contract(exponential_model(), eu(), PreferencePolicy::ReportOnly);
    assert_eq!(preference_route(&exponential).unwrap().route, PreferenceRoute::Exact);

    let mut single = affine;
    single.actions.truncate(1);
    let error = preference_route(&single).unwrap_err();
    assert!(matches!(&error, E::Contract(inner) if error.detail() == inner.detail()));
}

#[test]
fn each_route_answers_on_joint_draws_with_shared_accessors() {
    let model = cost_model(0.0, 3.0);
    let source = standard_law(&model);
    let c = contract(model, eu(), PreferencePolicy::RequireInvariantBestAction);
    let exact = decide_preferences(&c, &source, None, options()).unwrap();
    assert_eq!(exact.route(), PreferenceRoute::Exact);
    assert_eq!(
        exact_verdict(&exact),
        &PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    assert_eq!(exact.chosen_action(), None);
    assert!(!exact.is_necessarily_best() && !exact.is_never_beaten());
    assert_eq!(exact.undecided_fraction(), None);
    assert_eq!(
        exact.statement(),
        "The preferred action depends on the declared trade-offs: A leads when w is below 1.5 \
         and B leads above it; the admissible range of w is 0 to 3."
    );
    assert_eq!(exact.contract_identity(), c.identity().unwrap());
    assert_eq!(exact.assumptions(), &["w-judgment".to_owned()]);
    assert!(exact.certified().is_none());

    let shaped = contract(squared_model(), eu(), PreferencePolicy::RequireInvariantBestAction);
    let source = standard_law(&shaped.model);
    let certified = decide_preferences(&shaped, &source, None, options()).unwrap();
    assert_eq!(certified.route(), PreferenceRoute::Certified);
    let direct =
        evaluate_preference_certified(&shaped, &source, None, CertifiedOptions::default()).unwrap();
    assert_eq!(certified.certified(), Some(&direct));
    assert!(certified.exact().is_none());
    let fraction = certified.undecided_fraction().unwrap();
    assert!((0.0..=1.0).contains(&fraction));
    assert_eq!(
        certified.is_necessarily_best(),
        matches!(direct.verdict, CertifiedVerdict::NecessarilyBest(_))
    );
    // `v_A - v_B = 4.5 - 3 w^2` changes sign inside `[0, 3]`, so no action is best everywhere.
    assert!(!certified.is_necessarily_best() && !certified.is_never_beaten());
    assert_eq!(certified.chosen_action(), None);
    let statement = certified.statement();
    assert!(
        statement.starts_with("The preferred action depends on the declared trade-offs")
            || statement.starts_with("The certified subdivision does not settle the choice"),
        "{statement}"
    );
    if fraction > 0.0 {
        assert!(statement.contains("volume fraction"), "{statement}");
    }
    assert!(!statement.to_lowercase().contains("probab"), "{statement}");
}

#[test]
fn the_exact_route_falls_through_when_it_names_the_certified_route() {
    let model = exponential_model();
    let source = standard_law(&model);
    let c = contract(model.clone(), eu(), PreferencePolicy::ReportOnly);
    let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
    assert_eq!(evaluation.route(), PreferenceRoute::Certified);
    let error = decide_preferences_on_means(&c, &mean_source(&model), None, options()).unwrap_err();
    assert_eq!(error, E::JointDrawsRequired { reason: ROUTE_PART_NOT_LOWERABLE });
    assert_eq!(error.detail(), reason_details::preference_evaluation::JOINT_DRAWS_REQUIRED);

    // The worst case of expected regret is not a linear program over the set.
    let model = cost_model(0.0, 3.0);
    let source = standard_law(&model);
    let c = contract(model.clone(), DecisionCriterion::ExpectedRegret, PreferencePolicy::Maximin);
    assert_eq!(preference_route(&c).unwrap().route, PreferenceRoute::Exact);
    let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
    assert_eq!(evaluation.route(), PreferenceRoute::Certified);
    let c = contract(model, DecisionCriterion::ExpectedRegret, PreferencePolicy::ReportOnly);
    assert_eq!(
        decide_preferences(&c, &source, None, options()).unwrap().route(),
        PreferenceRoute::Exact
    );
}

#[test]
fn a_mean_source_answers_only_the_exact_route() {
    let model = cost_model(0.0, 3.0);
    let means = mean_source(&model);
    let c =
        reading_means(contract(model.clone(), eu(), PreferencePolicy::RequireInvariantBestAction));
    let evaluation = decide_preferences_on_means(&c, &means, None, options()).unwrap();
    assert_eq!(evaluation.route(), PreferenceRoute::Exact);
    assert_eq!(evaluation.exact().unwrap().n_draws, 0);
    assert_eq!(
        exact_verdict(&evaluation),
        &PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );

    let shaped = contract(squared_model(), eu(), PreferencePolicy::ReportOnly);
    let error = decide_preferences_on_means(&shaped, &mean_source(&shaped.model), None, options())
        .unwrap_err();
    assert_eq!(error, E::JointDrawsRequired { reason: ROUTE_SHAPE_PARAMETERS });

    let threshold = contract(
        model,
        DecisionCriterion::ThresholdProbability { threshold: 5.0 },
        PreferencePolicy::ReportOnly,
    );
    let error = decide_preferences_on_means(&threshold, &means, None, options()).unwrap_err();
    assert_eq!(error, E::JointDrawsRequired { reason: ROUTE_CRITERION_NOT_AFFINE });
}

#[test]
fn refusals_of_either_route_keep_their_own_detail() {
    let model = cost_model(0.0, 3.0);
    let source = standard_law(&model);
    let weighted = contract(model, eu(), PreferencePolicy::Weighted);
    let error = decide_preferences(&weighted, &source, None, options()).unwrap_err();
    assert!(matches!(&error, E::Exact(inner) if error.detail() == inner.detail()));
    assert!(error.detail().starts_with("preference_decision."));

    let shaped = contract(squared_model(), eu(), PreferencePolicy::Weighted);
    let error = decide_preferences(&shaped, &source, None, options()).unwrap_err();
    assert!(matches!(&error, E::Certified(inner) if error.detail() == inner.detail()));
    assert!(error.detail().starts_with("preference_certified."));
}

#[test]
fn structures_take_the_same_route() {
    let atoms = |model: &ValueModel| {
        vec![StructuralAtom {
            id: "s1".into(),
            probability: None,
            evidence: AtomEvidence::Evaluated(Box::new(standard_law(model))),
        }]
    };
    let affine = contract(cost_model(0.0, 3.0), eu(), PreferencePolicy::ReportOnly);
    let evaluation =
        decide_preferences_structural(&affine, &atoms(&affine.model), None, options()).unwrap();
    assert_eq!(evaluation.route(), PreferenceRoute::Exact);
    let shaped = contract(squared_model(), eu(), PreferencePolicy::ReportOnly);
    let evaluation =
        decide_preferences_structural(&shaped, &atoms(&shaped.model), None, options()).unwrap();
    assert_eq!(evaluation.route(), PreferenceRoute::Certified);
    let mut single = shaped;
    single.actions.truncate(1);
    assert!(matches!(
        decide_preferences_structural(&single, &atoms(&single.model), None, options()),
        Err(E::Contract(_))
    ));
}

#[test]
fn accepted_preferences_of_every_basis_are_decided_end_to_end() {
    let model = cost_model(0.0, 3.0);
    let source = standard_law(&model);
    let invariant =
        || parts(&model, &["A", "B"], eu(), PreferencePolicy::RequireInvariantBestAction);

    // The declared range leaves the choice open.
    let declared = AcceptedPreferences::declared(model.clone()).unwrap();
    let open = decide_with_accepted(&declared, invariant(), &source, options()).unwrap();
    assert_eq!(
        exact_verdict(&open),
        &PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );

    // The learned set `[1/2, 1]` settles it: `v_A - v_B` lies in `[1.5, 3]`.
    let learned = learn_compatible_set_accepted(
        &model,
        &settling_records(&model),
        None,
        SetLearningOptions::default(),
    )
    .unwrap();
    let settled = decide_with_accepted(&learned, invariant(), &source, options()).unwrap();
    assert_eq!(settled.route(), PreferenceRoute::Exact);
    assert!(settled.is_necessarily_best() && settled.is_never_beaten());
    assert_eq!(
        settled.statement(),
        "Action A is preferred for every admissible value of the declared trade-offs."
    );
    assert_eq!(settled.exact().unwrap().parameter_ranges.len(), 1);
    let (low, high) = settled.exact().unwrap().parameter_ranges[0];
    assert!((low - 0.5).abs() < 1e-9 && (high - 1.0).abs() < 1e-9, "the learned range");
    assert_eq!(settled.chosen_action(), Some("A"));
    assert_eq!(
        settled.contract_identity(),
        learned.contract(invariant()).identity().unwrap(),
        "the contract is built over the accepted model"
    );
    assert!(settled.assumptions().iter().any(|a| a == "recorded choice r1 over b"));
    let (low, high) = settled.exact().unwrap().actions[0].value_range.unwrap();
    assert!((low - 6.0).abs() < 1e-9 && (high - 8.0).abs() < 1e-9, "v_A = 10 - 4w on [1/2, 1]");

    // The estimate's measure is passed, so the weighted policy can choose.
    let records = simulated(&model, 400, 11);
    let quick = ChoiceModelOptions { n_draws: 200, ..ChoiceModelOptions::default() };
    let estimated = estimate_accepted(&model, &records, None, quick).unwrap();
    let weighted = parts(&model, &["A", "B"], eu(), PreferencePolicy::Weighted);
    let chosen = decide_with_accepted(&estimated, weighted.clone(), &source, options()).unwrap();
    let PreferenceVerdict::MeasureChoice { action, share } = exact_verdict(&chosen) else {
        panic!("{:?}", exact_verdict(&chosen))
    };
    assert_eq!(action, "A");
    assert!(*share >= 0.95, "share {share}");
    let measure = estimated.measure().unwrap();
    let mean_w = measure.points.iter().map(|p| p[0]).sum::<f64>() / as_f64(measure.points.len());
    let value_a = chosen.exact().unwrap().actions[0].measure_value.unwrap();
    assert!((value_a - (10.0 - 4.0 * mean_w)).abs() < 1e-8, "{value_a} vs w = {mean_w}");
    let estimate_identity = match &measure.origin {
        MeasureOrigin::ChoiceEstimate { estimate_identity } => estimate_identity.clone(),
        other @ MeasureOrigin::Declared { .. } => panic!("{other:?}"),
    };
    assert!(chosen.assumptions().contains(&format!("choice estimate {estimate_identity}")));

    // Without a measure the weighted policy refuses.
    let error = decide_with_accepted(&declared, weighted, &source, options()).unwrap_err();
    assert!(matches!(error, E::Exact(_)));
}

/// Coefficients `(c_0, c_1, ...)` evaluated at `theta`.
fn affine_value(coefficients: &[f64], theta: &[f64]) -> f64 {
    coefficients[0] + coefficients[1..].iter().zip(theta).map(|(c, t)| c * t).sum::<f64>()
}

/// A random affine problem: `x0 + sum_k t_k x_{k+1}` with `k` parameters, `n_actions` actions
/// and five equally weighted draws.
fn random_problem(
    rng: &mut Xorshift,
    k: usize,
    n_actions: usize,
    criterion: DecisionCriterion,
) -> (PreferenceContract, DistributionArtifact) {
    let parameters: Vec<PreferenceParameter> = (0..k)
        .map(|j| {
            let lower = -1.0 - rng.uniform();
            parameter(&format!("t{j}"), lower, lower + 0.5 + 2.0 * rng.uniform())
        })
        .collect();
    let value = (0..k).fold(P::Attribute(0), |sum, j| {
        P::sum(sum, P::product(P::Parameter(j), P::Attribute(j + 1)))
    });
    let model = ValueModel {
        attributes: (0..=k).map(|j| attribute(&format!("x{j}"), &format!("u{j}"))).collect(),
        preferences: PreferenceSet::new(parameters, vec![]),
        value,
        utility_units: "utility".to_owned(),
    };
    let arms: Vec<Vec<Vec<f64>>> = (0..n_actions)
        .map(|_| (0..=k).map(|_| (0..5).map(|_| 6.0 * rng.uniform() - 3.0).collect()).collect())
        .collect();
    let source = law(&model, &arms, None);
    let ids: Vec<String> = (0..n_actions).map(|a| format!("a{a}")).collect();
    let names: Vec<&str> = ids.iter().map(String::as_str).collect();
    let parts = parts(&model, &names, criterion, PreferencePolicy::RequireInvariantBestAction);
    let contract = PreferenceContract {
        model,
        actions: parts.actions,
        criterion,
        constraints: vec![],
        target_population: parts.target_population,
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    };
    (contract, source)
}

/// The corners and the centre of a box.
fn box_points(lower: &[f64], upper: &[f64]) -> Vec<Vec<f64>> {
    let k = lower.len();
    let mut points: Vec<Vec<f64>> = (0..1_usize << k)
        .map(|mask| {
            (0..k).map(|j| if (mask >> j) & 1 == 1 { upper[j] } else { lower[j] }).collect()
        })
        .collect();
    points.push(lower.iter().zip(upper).map(|(l, u)| 0.5 * (l + u)).collect());
    points
}

#[test]
fn the_certified_route_never_contradicts_the_exact_route_on_affine_problems() {
    let mut rng = Xorshift::new(2026);
    let certified_options = CertifiedOptions { max_boxes: 256, ..CertifiedOptions::default() };
    let mut necessarily_best_cases = 0;
    let mut switch_cases = 0;
    for case in 0..40 {
        let criterion = if case % 3 == 0 {
            DecisionCriterion::PosteriorExpectedLoss
        } else {
            DecisionCriterion::PosteriorExpectedUtility
        };
        let (c, source) = random_problem(&mut rng, 1 + case % 2, 2 + case % 2, criterion);
        let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
        let exact = evaluation.exact().expect("affine expectation takes the exact route");
        let certified =
            evaluate_preference_certified(&c, &source, None, certified_options).unwrap();
        let coefficients: Vec<Vec<f64>> =
            exact.actions.iter().map(|a| a.coefficients.clone().unwrap()).collect();
        let index = |id: &str| c.actions.iter().position(|a| a.id == id).unwrap();
        for region in &certified.regions {
            let RegionStatus::Best(best) = &region.status else { continue };
            // A never-beaten action, unique or tied somewhere, is never strictly behind a
            // rival, so no box can be certified for that rival.
            if let PreferenceVerdict::NecessarilyBest(exact_best)
            | PreferenceVerdict::NeverBeaten { action: exact_best, .. } = &exact.verdict
            {
                assert_eq!(best, exact_best, "case {case}: a certified box names a rival");
            }
            let a = index(best);
            for point in box_points(&region.lower, &region.upper) {
                for (b, rival) in coefficients.iter().enumerate().filter(|(b, _)| *b != a) {
                    let margin =
                        affine_value(&coefficients[a], &point) - affine_value(rival, &point);
                    assert!(
                        margin >= -1e-9,
                        "case {case}: box certified for {best} but a{b} leads by {} at {point:?}",
                        -margin
                    );
                }
            }
        }
        if matches!(exact.verdict, PreferenceVerdict::NecessarilyBest(_)) {
            necessarily_best_cases += 1;
        }
        // A switch between two rivals does not stop a third action from leading everywhere,
        // so the routes are compared on the verdict, not on the presence of a switch.
        if let CertifiedVerdict::NecessarilyBest(best) = &certified.verdict {
            // A certified unique winner is the exact route's unique winner, not merely an
            // action that is never beaten.
            assert_eq!(
                exact.verdict,
                PreferenceVerdict::NecessarilyBest(best.clone()),
                "case {case}: only the certified route finds a necessarily best action"
            );
            assert!(exact.actions[index(best)].uniquely_best, "case {case}: {best} not unique");
            assert!(
                exact.switches.iter().all(|s| &s.first != best && &s.second != best),
                "case {case}: the exact route reports a switch involving {best}"
            );
        }
        assert_eq!(
            evaluation.is_never_beaten(),
            exact.actions.iter().filter(|a| a.necessarily_best).count() == 1,
            "case {case}"
        );
        if matches!(exact.verdict, PreferenceVerdict::PreferenceDependent { .. }) {
            switch_cases += 1;
            assert!(
                !matches!(certified.verdict, CertifiedVerdict::NecessarilyBest(_)),
                "case {case}: the exact route finds the choice depends on the parameters"
            );
        }
    }
    // Both kinds of problem occur, so neither check is vacuous.
    assert!(necessarily_best_cases > 0 && switch_cases > 0);
}

/// `u * benefit - w * cost` with `u = 0.9` and `w = 1.2` fixed by declaration.
fn fixed_model() -> ValueModel {
    ValueModel {
        attributes: vec![attribute("benefit", "events"), attribute("cost", "usd")],
        preferences: PreferenceSet::new(
            vec![parameter("u", 0.9, 0.9), parameter("w", 1.2, 1.2)],
            vec![],
        ),
        value: P::difference(
            P::product(P::Parameter(0), P::Attribute(0)),
            P::product(P::Parameter(1), P::Attribute(1)),
        ),
        utility_units: "events".to_owned(),
    }
}

/// `benefit - cost`, with no preference parameter.
fn parameter_free_model() -> ValueModel {
    ValueModel {
        attributes: vec![attribute("benefit", "events"), attribute("cost", "usd")],
        preferences: PreferenceSet::new(vec![], vec![]),
        value: P::difference(P::Attribute(0), P::Attribute(1)),
        utility_units: "events".to_owned(),
    }
}

/// `A`, `B` and `C` under a budget that excludes `A` (it costs 5 on a draw of weight 1/4).
fn budgeted(model: ValueModel, criterion: DecisionCriterion) -> PreferenceContract {
    let parts =
        parts(&model, &["A", "B", "C"], criterion, PreferencePolicy::RequireInvariantBestAction);
    PreferenceContract {
        model,
        actions: parts.actions,
        criterion,
        constraints: vec![HardConstraint {
            id: "budget".into(),
            expr: UtilityExpr::Input(1),
            bound: 4.5,
            min_probability: 1.0,
            units: "usd".into(),
            applies_to: vec![],
        }],
        target_population: parts.target_population,
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

/// The standard law with a third action `C` at `(benefit, cost) = (4, 2)`.
fn three_action_law(model: &ValueModel) -> DistributionArtifact {
    let arms = vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
        vec![vec![4.0, 4.0, 4.0], vec![2.0, 2.0, 2.0]],
    ];
    law(model, &arms, Some(vec![1.0, 1.0, 2.0]))
}

fn expectation_criteria() -> [DecisionCriterion; 3] {
    [
        DecisionCriterion::PosteriorExpectedUtility,
        DecisionCriterion::PosteriorExpectedLoss,
        DecisionCriterion::ExpectedRegret,
    ]
}

/// The evaluation across the set agrees with the ordinary evaluation at the point `at`: the
/// same chosen action, the same expected utility of every admissible action (within 1e-12,
/// relative), and the same exclusions.
fn assert_same_as_ordinary(
    label: &str,
    evaluation: &PreferenceEvaluation,
    ordinary: &DecisionResult,
    at: &[f64],
    criterion: DecisionCriterion,
) {
    let chosen = match &ordinary.verdict {
        Verdict::UniquelyOptimal(action) => Some(action.as_str()),
        Verdict::Indistinguishable(_) | Verdict::NoAdmissibleAction => None,
    };
    assert_eq!(evaluation.chosen_action(), chosen, "{label}");
    assert_eq!(evaluation.is_necessarily_best(), chosen.is_some(), "{label}");
    let exact = evaluation.exact().expect("the exact route answers");
    let sign =
        if matches!(criterion, DecisionCriterion::PosteriorExpectedLoss) { -1.0 } else { 1.0 };
    assert_eq!(exact.actions.len(), ordinary.actions.len(), "{label}");
    for (ours, theirs) in exact.actions.iter().zip(&ordinary.actions) {
        assert_eq!(ours.id, theirs.id, "{label}");
        assert_eq!(ours.admissible, theirs.admissible, "{label} {}", ours.id);
        assert_eq!(ours.exclusions, theirs.exclusions, "{label} {}", ours.id);
        if !theirs.admissible {
            assert!(ours.coefficients.is_none(), "{label} {}", ours.id);
            continue;
        }
        let expected = sign * affine_value(ours.coefficients.as_ref().unwrap(), at);
        let tolerance = 1e-12 * (1.0 + theirs.expected_utility.abs());
        assert!(
            (expected - theirs.expected_utility).abs() <= tolerance,
            "{label} {}: {expected} vs {}",
            ours.id,
            theirs.expected_utility
        );
        if let Some((low, high)) = ours.value_range {
            assert!((low - theirs.expected_utility).abs() <= tolerance, "{label} {}", ours.id);
            assert!((high - theirs.expected_utility).abs() <= tolerance, "{label} {}", ours.id);
        }
    }
}

#[test]
fn fixed_preferences_decide_as_the_ordinary_contract_on_draws() {
    for (name, model, at) in [
        ("fixed", fixed_model(), vec![0.9, 1.2]),
        ("parameter-free", parameter_free_model(), vec![]),
    ] {
        let source = three_action_law(&model);
        for criterion in expectation_criteria() {
            let label = format!("{name} {criterion:?}");
            let c = budgeted(model.clone(), criterion);
            let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
            assert_eq!(evaluation.route(), PreferenceRoute::Exact, "{label}");
            let ordinary = evaluate_contract(&c.bind(&at).unwrap(), &source).unwrap();
            assert_same_as_ordinary(&label, &evaluation, &ordinary, &at, criterion);
            // The budget excludes A whatever the criterion.
            assert!(!evaluation.exact().unwrap().actions[0].admissible, "{label}");
        }
    }
}

#[test]
fn fixed_preferences_decide_as_the_ordinary_contract_on_means() {
    for (name, model, at) in [
        ("fixed", fixed_model(), vec![0.9, 1.2]),
        ("parameter-free", parameter_free_model(), vec![]),
    ] {
        let means = mean_source(&model);
        for criterion in expectation_criteria() {
            let label = format!("{name} {criterion:?}");
            let c = reading_means(contract(
                model.clone(),
                criterion,
                PreferencePolicy::RequireInvariantBestAction,
            ));
            let ordinary = evaluate_contract_on_means(&c.bind(&at).unwrap(), &means);
            let evaluation = decide_preferences_on_means(&c, &means, None, options());
            match ordinary {
                Ok(ordinary) => {
                    assert_same_as_ordinary(&label, &evaluation.unwrap(), &ordinary, &at, criterion)
                }
                // A criterion a mean source cannot answer is refused by both, identically.
                Err(refusal) => {
                    assert_eq!(criterion, DecisionCriterion::ExpectedRegret, "{label}");
                    match evaluation.unwrap_err() {
                        E::Exact(PreferenceDecisionError::Eval(error)) => {
                            assert_eq!(error, refusal, "{label}")
                        }
                        other => panic!("{label}: {other:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn declared_ranges_decide_without_records_accepted_preferences_or_a_measure() {
    // `v_A - v_B = 4.5 - 3w` over the declared `w` in `[0, 3]`.
    let model = cost_model(0.0, 3.0);
    let source = standard_law(&model);
    let decide = |policy| {
        decide_preferences(&contract(model.clone(), eu(), policy), &source, None, options())
    };

    let invariant = decide(PreferencePolicy::RequireInvariantBestAction).unwrap();
    assert_eq!(
        exact_verdict(&invariant),
        &PreferenceVerdict::PreferenceDependent { possibly_best: vec!["A".into(), "B".into()] }
    );
    assert_eq!(invariant.assumptions(), &["w-judgment".to_owned()]);

    // A in [-2, 10], B in [2.5, 5.5]: B has the better worst case.
    let maximin = decide(PreferencePolicy::Maximin).unwrap();
    assert_eq!(exact_verdict(&maximin), &PreferenceVerdict::WorstCaseChoice("B".into()));
    assert_eq!(maximin.chosen_action(), Some("B"));

    // Worst-case regret is 4.5 for both: A at w = 3, B at w = 0.
    let regret = decide(PreferencePolicy::MinimaxRegret).unwrap();
    assert_eq!(exact_verdict(&regret), &PreferenceVerdict::Tied(vec!["A".into(), "B".into()]));
    assert_eq!(
        regret.statement(),
        "Actions A and B share the smallest worst-case regret over the admissible values of the \
         declared trade-offs; no single action is chosen."
    );

    let report = decide(PreferencePolicy::ReportOnly).unwrap();
    assert_eq!(exact_verdict(&report), &PreferenceVerdict::ReportOnly);
    assert_eq!(report.exact().unwrap().switches.len(), 1);

    // Only the weighted policy needs a measure, and says so.
    let error = decide(PreferencePolicy::Weighted).unwrap_err();
    assert_eq!(error.detail(), reason_details::preference_decision::MEASURE_REQUIRED);
}

/// Non-test source lines of a module, comments dropped.
fn code_lines(source: &str) -> Vec<&str> {
    let code = source.split("#[cfg(test)]").next().unwrap_or(source);
    code.lines().filter(|line| !line.trim_start().starts_with("//")).collect()
}

#[test]
fn the_explicit_route_and_its_record_read_no_preference_learning_code() {
    let learning = [
        "accepted_preferences",
        "choice_record",
        "preference_choice_model",
        "preference_set_learning",
    ];
    let explicit = [
        ("preference_model", include_str!("../src/preference_model.rs")),
        ("preference_contract", include_str!("../src/preference_contract.rs")),
        ("preference_enclosure", include_str!("../src/preference_enclosure.rs")),
        ("preference_decision", include_str!("../src/preference_decision.rs")),
        ("preference_certified", include_str!("../src/preference_certified.rs")),
        ("preference_route_choice", include_str!("../src/preference_route_choice.rs")),
        ("preference_decision_artifact", include_str!("../src/preference_decision_artifact.rs")),
    ];
    for (name, source) in explicit {
        for line in code_lines(source) {
            for module in learning {
                assert!(
                    !line.contains(&format!("crate::{module}")),
                    "{name} reads {module}: {line}"
                );
            }
        }
    }
    // The shared artifact refusal names the refusal of accepted preferences and nothing else
    // that learns.
    let error = code_lines(include_str!("../src/preference_artifact_error.rs"));
    let reads: Vec<&&str> = error
        .iter()
        .filter(|line| learning.iter().any(|m| line.contains(&format!("crate::{m}"))))
        .collect();
    assert_eq!(reads, vec![&"use crate::accepted_preferences::AcceptedPreferencesError;"]);
}

// ---------------------------------------------------------------------------------------------
// Fixed trade-offs against a hand-written ordinary contract
// ---------------------------------------------------------------------------------------------

/// `0.9 benefit - 1.2 cost` written directly as an ordinary utility over `(benefit, cost)`.
fn fixed_utility() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Const(0.9), UtilityExpr::Input(0)),
        UtilityExpr::product(UtilityExpr::Const(1.2), UtilityExpr::Input(1)),
    )
}

/// `benefit - cost` written directly as an ordinary utility.
fn parameter_free_utility() -> UtilityExpr {
    UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1))
}

/// Cost at most 4.5 on every draw, for every action.
fn budget() -> HardConstraint {
    HardConstraint {
        id: "budget".into(),
        expr: UtilityExpr::Input(1),
        bound: 4.5,
        min_probability: 1.0,
        units: "usd".into(),
        applies_to: vec![],
    }
}

/// Benefit at least 5 with probability one half, for `B` and `C` only.
fn benefit_floor() -> HardConstraint {
    HardConstraint {
        id: "benefit_floor".into(),
        expr: UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))),
        bound: -5.0,
        min_probability: 0.5,
        units: "events".into(),
        applies_to: vec!["B".into(), "C".into()],
    }
}

/// The preference contract over `A`, `B`, `C` with every input a quantity.
fn three_actions(
    model: ValueModel,
    criterion: DecisionCriterion,
    constraints: Vec<HardConstraint>,
    structural_policy: StructuralPolicy,
) -> PreferenceContract {
    let parts =
        parts(&model, &["A", "B", "C"], criterion, PreferencePolicy::RequireInvariantBestAction);
    PreferenceContract {
        model,
        actions: parts.actions,
        criterion,
        constraints,
        target_population: parts.target_population,
        horizon: 0,
        structural_policy,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

/// The ordinary contract a user writes by hand for the same decision: each action reads
/// `(benefit, cost)` under its own regime and values them with `utility`; the constraints are
/// the user's own, under their own ids.
fn hand_written(
    model: &ValueModel,
    utility: &UtilityExpr,
    criterion: DecisionCriterion,
    constraints: Vec<HardConstraint>,
    structural_policy: StructuralPolicy,
) -> DecisionContract {
    DecisionContract {
        actions: ["A", "B", "C"]
            .iter()
            .enumerate()
            .map(|(a, id)| DecisionAction {
                id: (*id).to_owned(),
                kind: ActionKind::Intervention,
                inputs: model.attributes.iter().map(|k| quantity(k, a)).collect(),
                utility: utility.clone(),
            })
            .collect(),
        utility_units: model.utility_units.clone(),
        criterion,
        constraints,
        target_population: "target".to_owned(),
        horizon: 0,
        structural_policy,
    }
}

/// The two fixed-preference models with their hand-written utilities.
fn fixed_cases() -> [(&'static str, ValueModel, UtilityExpr); 2] {
    [
        ("fixed", fixed_model(), fixed_utility()),
        ("parameter-free", parameter_free_model(), parameter_free_utility()),
    ]
}

fn constraint_sets() -> [(&'static str, Vec<HardConstraint>); 4] {
    [
        ("unconstrained", vec![]),
        ("budget", vec![budget()]),
        ("floor", vec![benefit_floor()]),
        ("budget+floor", vec![budget(), benefit_floor()]),
    ]
}

/// Values agree within rounding: the exact route sums `theta_j E[part_j]`, the ordinary
/// evaluator `E[sum_j theta_j part_j]`.
fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * (1.0 + right.abs())
}

/// The ordinary verdict, read as the action it names alone or the actions it ties.
fn ordinary_choice(result: &DecisionResult) -> Result<Option<String>, Vec<String>> {
    match &result.verdict {
        Verdict::UniquelyOptimal(action) => Ok(Some(action.clone())),
        Verdict::Indistinguishable(tied) => Err(tied.clone()),
        Verdict::NoAdmissibleAction => Ok(None),
    }
}

/// The exact verdict agrees with the ordinary one: a unique optimum is the unique winner,
/// equal values are a tie, and no admissible action is no admissible action. The exclusions
/// are the same, under the same constraint ids, and every admissible action's criterion value
/// is the ordinary value.
fn assert_exact_agrees(label: &str, exact: &PreferenceDecisionResult, ordinary: &DecisionResult) {
    let expected = match ordinary_choice(ordinary) {
        Ok(Some(action)) => PreferenceVerdict::NecessarilyBest(action),
        Ok(None) => PreferenceVerdict::NoAdmissibleAction,
        Err(tied) => PreferenceVerdict::Tied(tied),
    };
    assert_eq!(exact.verdict, expected, "{label}");
    let regret = ordinary.criterion == DecisionCriterion::ExpectedRegret;
    let sign =
        if ordinary.criterion == DecisionCriterion::PosteriorExpectedLoss { -1.0 } else { 1.0 };
    assert_eq!(exact.actions.len(), ordinary.actions.len(), "{label}");
    for (ours, theirs) in exact.actions.iter().zip(&ordinary.actions) {
        let at = format!("{label} {}", ours.id);
        assert_eq!(ours.id, theirs.id, "{at}");
        assert_eq!(ours.admissible, theirs.admissible, "{at}");
        assert_eq!(ours.exclusions, theirs.exclusions, "{at}");
        if !theirs.admissible {
            assert!(ours.coefficients.is_none() && ours.value_range.is_none(), "{at}");
            continue;
        }
        let coefficients = ours.coefficients.as_ref().expect("admissible coefficients");
        let point: Vec<f64> = exact.parameter_ranges.iter().map(|range| range.0).collect();
        assert!(close(sign * affine_value(coefficients, &point), theirs.expected_utility), "{at}");
        if regret {
            // Expected regret ranks by expected-utility margins and reports no regret amount.
            assert_eq!(ours.value_range, None, "{at}");
        } else {
            let (low, high) = ours.value_range.expect("admissible value range");
            assert!(close(low, theirs.value) && close(high, theirs.value), "{at}: {low} {high}");
        }
    }
}

/// The certified verdict agrees with the ordinary one: a unique optimum is certified best, equal
/// values leave the choice among the tied actions undetermined, and no admissible action is no
/// admissible action. The exclusions are the same, and every admissible action's criterion
/// enclosure holds the ordinary value and is no wider than rounding.
fn assert_certified_agrees(
    label: &str,
    certified: &CertifiedPreferenceResult,
    ordinary: &DecisionResult,
) {
    match ordinary_choice(ordinary) {
        Ok(Some(action)) => {
            assert_eq!(certified.verdict, CertifiedVerdict::NecessarilyBest(action), "{label}")
        }
        Ok(None) => assert_eq!(certified.verdict, CertifiedVerdict::NoAdmissibleAction, "{label}"),
        Err(tied) => match &certified.verdict {
            CertifiedVerdict::Undetermined { candidates, .. } => {
                assert_eq!(candidates, &tied, "{label}")
            }
            other => panic!("{label}: {other:?}"),
        },
    }
    assert_eq!(certified.actions.len(), ordinary.actions.len(), "{label}");
    for (ours, theirs) in certified.actions.iter().zip(&ordinary.actions) {
        let at = format!("{label} {}", ours.id);
        assert_eq!(ours.id, theirs.id, "{at}");
        assert_eq!(ours.admissible, theirs.admissible, "{at}");
        assert_eq!(ours.exclusions, theirs.exclusions, "{at}");
        if !theirs.admissible {
            assert_eq!(ours.value_bounds, None, "{at}");
            continue;
        }
        let (low, high) = ours.value_bounds.expect("admissible value bounds");
        let slack = 1e-12 * (1.0 + theirs.value.abs());
        assert!(low <= theirs.value + slack && theirs.value - slack <= high, "{at}: {low} {high}");
        assert!(high - low <= 1e-9 * (1.0 + theirs.value.abs()), "{at}: {low} {high}");
    }
}

fn all_criteria() -> [DecisionCriterion; 7] {
    [
        DecisionCriterion::PosteriorExpectedUtility,
        DecisionCriterion::PosteriorExpectedLoss,
        DecisionCriterion::ExpectedRegret,
        DecisionCriterion::ThresholdProbability { threshold: 3.1 },
        DecisionCriterion::ThresholdProbability { threshold: 4.5 },
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::Regret,
    ]
}

#[test]
fn fixed_trade_offs_decide_as_a_hand_written_contract_under_every_criterion() {
    let mut exact_cases = 0;
    let mut certified_cases = 0;
    for (name, model, utility) in fixed_cases() {
        let source = three_action_law(&model);
        for (constraints_name, constraints) in constraint_sets() {
            for criterion in all_criteria() {
                let label = format!("{name} {constraints_name} {criterion:?}");
                let c = three_actions(
                    model.clone(),
                    criterion,
                    constraints.clone(),
                    StructuralPolicy::ReportOnly,
                );
                let ordinary = evaluate_contract(
                    &hand_written(
                        &model,
                        &utility,
                        criterion,
                        constraints.clone(),
                        StructuralPolicy::ReportOnly,
                    ),
                    &source,
                )
                .unwrap();
                let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
                assert_eq!(
                    evaluation.chosen_action(),
                    ordinary_choice(&ordinary).ok().flatten().as_deref(),
                    "{label}"
                );
                match evaluation.route() {
                    PreferenceRoute::Exact => {
                        exact_cases += 1;
                        assert_exact_agrees(&label, evaluation.exact().unwrap(), &ordinary);
                    }
                    PreferenceRoute::Certified => {
                        certified_cases += 1;
                        assert_certified_agrees(&label, evaluation.certified().unwrap(), &ordinary);
                    }
                }
            }
        }
    }
    // Expectation criteria take the exact route; threshold, quantile and regret the certified.
    assert_eq!((exact_cases, certified_cases), (24, 32));
}

#[test]
fn fixed_trade_offs_exclude_under_the_users_constraint_ids() {
    let model = fixed_model();
    let source = three_action_law(&model);
    for criterion in [eu(), DecisionCriterion::Quantile { p: 0.5 }] {
        let c = three_actions(
            model.clone(),
            criterion,
            vec![budget(), benefit_floor()],
            StructuralPolicy::ReportOnly,
        );
        let evaluation = decide_preferences(&c, &source, None, options()).unwrap();
        let exclusions: Vec<Vec<&str>> = match &evaluation {
            PreferenceEvaluation::Exact(result) => result
                .actions
                .iter()
                .map(|a| a.exclusions.iter().map(|e| e.constraint_id.as_str()).collect())
                .collect(),
            PreferenceEvaluation::Certified(result) => result
                .actions
                .iter()
                .map(|a| a.exclusions.iter().map(|e| e.constraint_id.as_str()).collect())
                .collect(),
        };
        // `A` breaks the budget on one draw; `C` never reaches the benefit floor.
        assert_eq!(
            exclusions,
            vec![vec!["budget"], vec![], vec!["benefit_floor"]],
            "{criterion:?}"
        );
        assert_eq!(evaluation.chosen_action(), Some("B"), "{criterion:?}");
    }
}

/// `B` and `C` with expected values `v` and `v + gap` under `benefit - cost`, and `A` out of
/// the running (excluded by the budget).
fn near_tie_law(model: &ValueModel, gap: f64) -> DistributionArtifact {
    let arms = vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
        vec![vec![9.0 + gap, 5.0 + gap, 4.0 + gap], vec![1.0, 1.0, 1.0]],
    ];
    law(model, &arms, Some(vec![1.0, 1.0, 2.0]))
}

#[test]
fn near_ties_follow_the_documented_tie_rules() {
    let model = parameter_free_model();
    let utility = parameter_free_utility();
    let decide = |gap: f64, tie_tolerance: f64| {
        let source = near_tie_law(&model, gap);
        let c = three_actions(model.clone(), eu(), vec![budget()], StructuralPolicy::ReportOnly);
        let mut options = options();
        options.exact.tie_tolerance = tie_tolerance;
        let evaluation = decide_preferences(&c, &source, None, options).unwrap();
        let ordinary = evaluate_contract(
            &hand_written(&model, &utility, eu(), vec![budget()], StructuralPolicy::ReportOnly),
            &source,
        )
        .unwrap();
        (evaluation.exact().unwrap().clone(), ordinary)
    };

    // Equal values: both report the tie, in declaration order.
    let (exact, ordinary) = decide(0.0, 1e-9);
    assert_eq!(ordinary.verdict, Verdict::Indistinguishable(vec!["B".into(), "C".into()]));
    assert_exact_agrees("equal", &exact, &ordinary);

    // A gap well above the tie tolerance: both choose `C`.
    let (exact, ordinary) = decide(1e-6, 1e-9);
    assert_eq!(ordinary.verdict, Verdict::UniquelyOptimal("C".into()));
    assert_exact_agrees("separated", &exact, &ordinary);

    // A gap inside the declared tie tolerance but above the numerical bound of the margins. The
    // ordinary evaluator compares values exactly and chooses `C`; the exact route counts margins
    // within the effective tie tolerance as ties.
    let gap = 1e-11;
    let (exact, ordinary) = decide(gap, 1e-9);
    assert_eq!(ordinary.verdict, Verdict::UniquelyOptimal("C".into()));
    assert_eq!(exact.verdict, PreferenceVerdict::Tied(vec!["B".into(), "C".into()]));
    assert!(exact.effective_tie_tolerance >= 1e-9 && exact.effective_tie_tolerance > gap);
    // With no declared tolerance the effective one is the numerical bound alone, which the gap
    // exceeds, and the two agree again.
    let (exact, ordinary) = decide(gap, 0.0);
    assert!(exact.effective_tie_tolerance > 0.0 && exact.effective_tie_tolerance < gap);
    assert_exact_agrees("declared zero", &exact, &ordinary);
}

#[test]
fn certified_near_ties_follow_the_documented_tie_rules() {
    let model = parameter_free_model();
    let utility = parameter_free_utility();
    let criterion = DecisionCriterion::Quantile { p: 0.5 };
    let decide = |gap: f64, tie_tolerance: f64| {
        let source = near_tie_law(&model, gap);
        let c =
            three_actions(model.clone(), criterion, vec![budget()], StructuralPolicy::ReportOnly);
        let mut options = options();
        options.certified.tie_tolerance = tie_tolerance;
        let evaluation = decide_preferences(&c, &source, None, options).unwrap();
        let ordinary = evaluate_contract(
            &hand_written(
                &model,
                &utility,
                criterion,
                vec![budget()],
                StructuralPolicy::ReportOnly,
            ),
            &source,
        )
        .unwrap();
        (evaluation.certified().unwrap().clone(), ordinary)
    };

    let (certified, ordinary) = decide(0.0, 1e-9);
    assert_certified_agrees("equal", &certified, &ordinary);
    let (certified, ordinary) = decide(1e-6, 1e-9);
    assert_certified_agrees("separated", &certified, &ordinary);

    // A lead inside the tie tolerance is not certified: the choice stays undetermined between
    // the ordinary winner and its rival.
    let (certified, ordinary) = decide(1e-11, 1e-9);
    assert_eq!(ordinary.verdict, Verdict::UniquelyOptimal("C".into()));
    assert!(matches!(
        &certified.verdict,
        CertifiedVerdict::Undetermined { candidates, .. } if candidates == &["B", "C"]
    ));
    // With no tolerance the outward-rounded enclosures still separate the two.
    let (certified, ordinary) = decide(1e-11, 0.0);
    assert_certified_agrees("declared zero", &certified, &ordinary);
}

#[test]
fn a_utility_on_the_threshold_is_counted_by_the_ordinary_rule_and_left_open_by_the_certified() {
    // Under `benefit - cost`, `B` reaches exactly 4 on its third draw (weight 1/2), so
    // `P(U >= 4)` is 3/4 for `B` and 1/2 for `C`, which reaches 4.5 on every draw but the
    // third. The certified enclosure of a utility on the threshold straddles it, so it cannot
    // prove `B` ahead.
    let model = parameter_free_model();
    let utility = parameter_free_utility();
    let criterion = DecisionCriterion::ThresholdProbability { threshold: 4.0 };
    let arms = vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![9.0, 5.0, 5.0], vec![1.0, 1.0, 1.0]],
        vec![vec![5.5, 5.5, 4.0], vec![1.0, 1.0, 1.0]],
    ];
    let source = law(&model, &arms, Some(vec![1.0, 1.0, 2.0]));
    let ordinary = evaluate_contract(
        &hand_written(&model, &utility, criterion, vec![budget()], StructuralPolicy::ReportOnly),
        &source,
    )
    .unwrap();
    assert_eq!(ordinary.verdict, Verdict::UniquelyOptimal("B".into()));
    let c = three_actions(model, criterion, vec![budget()], StructuralPolicy::ReportOnly);
    let certified = decide_preferences(&c, &source, None, options()).unwrap();
    let certified = certified.certified().unwrap();
    assert!(matches!(
        &certified.verdict,
        CertifiedVerdict::Undetermined { candidates, .. } if candidates == &["B", "C"]
    ));
    let (low, high) = certified.actions[1].value_bounds.unwrap();
    assert!(low <= 0.25 && high >= 0.75, "{low} {high}");
}

/// Two structures over the same three actions: `C` leads in `s2`, `B` in `s1`. `A` is
/// excluded by the budget in both.
fn structure_laws(model: &ValueModel) -> [DistributionArtifact; 2] {
    let second = vec![
        vec![vec![8.0, 12.0, 10.0], vec![3.0, 5.0, 4.0]],
        vec![vec![6.0, 6.0, 6.0], vec![1.0, 1.0, 1.0]],
        vec![vec![7.0, 7.0, 7.0], vec![1.0, 1.0, 1.0]],
    ];
    [three_action_law(model), law(model, &second, Some(vec![1.0, 1.0, 2.0]))]
}

fn atom(id: &str, probability: Option<f64>, source: &DistributionArtifact) -> StructuralAtom {
    StructuralAtom {
        id: id.into(),
        probability,
        evidence: AtomEvidence::Evaluated(Box::new(source.clone())),
    }
}

/// The verdict the preference side gives for one ordinary per-structure leader list.
fn leader_verdict(leaders: &[String]) -> PreferenceVerdict {
    match leaders {
        [] => PreferenceVerdict::NoAdmissibleAction,
        [only] => PreferenceVerdict::NecessarilyBest(only.clone()),
        tied => PreferenceVerdict::Tied(tied.to_vec()),
    }
}

/// The structural verdicts agree: an invariant, worst-case or Bayes choice is the combined
/// unique winner, a tie is a combined tie, disagreeing structures list the same per-structure
/// leaders, and missing evidence gives the same reason.
fn assert_structural_agrees(
    label: &str,
    ours: &StructuralPreferenceResult,
    theirs: &StructuralDecisionResult,
) {
    use StructuralPreferenceVerdict as P;
    use antecedent_design::decision_structural::StructuralVerdict as O;
    match (&theirs.verdict, &ours.verdict) {
        (
            O::InvariantBest(a) | O::WorstCaseChoice(a) | O::BayesChoice { action: a, .. },
            P::Combined(verdict),
        ) => {
            assert_eq!(verdict, &PreferenceVerdict::NecessarilyBest(a.clone()), "{label}");
        }
        (O::Tied(tied), P::Combined(verdict)) => {
            assert_eq!(verdict, &PreferenceVerdict::Tied(tied.clone()), "{label}");
        }
        (O::NoAdmissibleAction, P::Combined(verdict)) => {
            assert_eq!(verdict, &PreferenceVerdict::NoAdmissibleAction, "{label}");
        }
        (O::NoInvariantBest(leaders), P::StructuresDisagree(choices)) => {
            let expected: Vec<(String, PreferenceVerdict)> =
                leaders.iter().map(|(atom, ids)| (atom.clone(), leader_verdict(ids))).collect();
            let found: Vec<(String, PreferenceVerdict)> =
                choices.iter().map(|c| (c.atom.clone(), c.verdict.clone())).collect();
            assert_eq!(found, expected, "{label}");
        }
        (O::InsufficientScience(why), P::InsufficientScience(ours)) => {
            assert_eq!(ours, why, "{label}");
        }
        (O::ReportOnly, P::ReportOnly) => {}
        (theirs, ours) => panic!("{label}: ordinary {theirs:?}, preference {ours:?}"),
    }
    assert_eq!(ours.evaluated_mass, theirs.evaluated_mass, "{label}");
    assert_eq!(ours.unidentified_mass, theirs.unidentified_mass, "{label}");
    assert_eq!(ours.unevaluated_mass, theirs.unevaluated_mass, "{label}");
    for (summary, ordinary) in ours.atoms.iter().zip(&theirs.atoms) {
        assert_eq!(summary.id, ordinary.id, "{label}");
        if let (
            PreferenceAtomStatus::Evaluated(result),
            antecedent_design::decision_structural::AtomStatus::Evaluated(expected),
        ) = (&summary.status, &ordinary.status)
        {
            assert_exact_agrees(&format!("{label} {}", summary.id), result, expected);
        }
    }
    // The combined values: the probability-weighted value under Bayes and the worst value over
    // structures under the worst case.
    if let Some(combined) = &ours.combined {
        for (outcome, action) in combined.actions.iter().zip(&theirs.actions) {
            let at = format!("{label} {}", outcome.id);
            let expected = match theirs.policy {
                StructuralPolicy::BayesOverStructures => action.weighted_value,
                StructuralPolicy::Maximin
                    if combined_criterion(theirs)
                        == Some(DecisionCriterion::PosteriorExpectedLoss) =>
                {
                    action.range.map(|range| range.1)
                }
                StructuralPolicy::Maximin => action.range.map(|range| range.0),
                _ => continue,
            };
            match (outcome.value_range, expected) {
                (Some((low, high)), Some(value)) if outcome.admissible => {
                    assert!(close(low, value) && close(high, value), "{at}: {low} {high} {value}");
                }
                (None, _) if !outcome.admissible => {}
                other => panic!("{at}: {other:?}"),
            }
        }
    }
}

/// The certified verdict for one ordinary per-structure leader list.
fn certified_leader_verdict(leaders: &[String]) -> CertifiedVerdict {
    match leaders {
        [] => CertifiedVerdict::NoAdmissibleAction,
        [only] => CertifiedVerdict::NecessarilyBest(only.clone()),
        tied => {
            CertifiedVerdict::Undetermined { candidates: tied.to_vec(), undecided_fraction: 1.0 }
        }
    }
}

/// The verdict with an undetermined volume fraction dropped, which the ordinary result has no
/// counterpart for.
fn without_fraction(verdict: &CertifiedVerdict) -> CertifiedVerdict {
    match verdict {
        CertifiedVerdict::Undetermined { candidates, .. } => CertifiedVerdict::Undetermined {
            candidates: candidates.clone(),
            undecided_fraction: 1.0,
        },
        other => other.clone(),
    }
}

/// The certified structural verdict agrees with the ordinary one, with a tie read as an
/// undetermined choice among the tied actions.
fn assert_certified_structural_agrees(
    label: &str,
    ours: &StructuralCertifiedResult,
    theirs: &StructuralDecisionResult,
) {
    use StructuralCertifiedVerdict as P;
    use antecedent_design::decision_structural::StructuralVerdict as O;
    match (&theirs.verdict, &ours.verdict) {
        (
            O::InvariantBest(a) | O::WorstCaseChoice(a) | O::BayesChoice { action: a, .. },
            P::Combined(verdict),
        ) => {
            assert_eq!(verdict, &CertifiedVerdict::NecessarilyBest(a.clone()), "{label}");
        }
        (O::Tied(tied), P::Combined(verdict)) => {
            assert_eq!(without_fraction(verdict), certified_leader_verdict(tied), "{label}");
        }
        (O::NoAdmissibleAction, P::Combined(verdict)) => {
            assert_eq!(verdict, &CertifiedVerdict::NoAdmissibleAction, "{label}");
        }
        (O::NoInvariantBest(leaders), P::StructuresDisagree(choices)) => {
            let expected: Vec<(String, CertifiedVerdict)> = leaders
                .iter()
                .map(|(atom, ids)| (atom.clone(), certified_leader_verdict(ids)))
                .collect();
            let found: Vec<(String, CertifiedVerdict)> =
                choices.iter().map(|c| (c.atom.clone(), without_fraction(&c.verdict))).collect();
            assert_eq!(found, expected, "{label}");
        }
        (O::InsufficientScience(why), P::InsufficientScience(ours)) => {
            assert_eq!(ours, why, "{label}");
        }
        (O::ReportOnly, P::ReportOnly) => {}
        (theirs, ours) => panic!("{label}: ordinary {theirs:?}, preference {ours:?}"),
    }
    assert_eq!(ours.evaluated_mass, theirs.evaluated_mass, "{label}");
    assert_eq!(ours.unidentified_mass, theirs.unidentified_mass, "{label}");
    assert_eq!(ours.unevaluated_mass, theirs.unevaluated_mass, "{label}");
    for (summary, ordinary) in ours.atoms.iter().zip(&theirs.atoms) {
        assert_eq!(summary.id, ordinary.id, "{label}");
        if let (
            CertifiedAtomStatus::Evaluated(result),
            antecedent_design::decision_structural::AtomStatus::Evaluated(expected),
        ) = (&summary.status, &ordinary.status)
        {
            assert_certified_agrees(&format!("{label} {}", summary.id), result, expected);
        }
    }
    if let Some(combined) = &ours.combined {
        let larger_is_worse = matches!(
            combined_criterion(theirs),
            Some(DecisionCriterion::PosteriorExpectedLoss | DecisionCriterion::Regret)
        );
        for (outcome, action) in combined.actions.iter().zip(&theirs.actions) {
            let at = format!("{label} {}", outcome.id);
            let expected = match theirs.policy {
                StructuralPolicy::BayesOverStructures => action.weighted_value,
                StructuralPolicy::Maximin if larger_is_worse => action.range.map(|r| r.1),
                StructuralPolicy::Maximin => action.range.map(|r| r.0),
                _ => continue,
            };
            match (outcome.value_bounds, expected) {
                (Some((low, high)), Some(value)) if outcome.admissible => {
                    let slack = 1e-12 * (1.0 + value.abs());
                    assert!(low <= value + slack && value - slack <= high, "{at}: {low} {high}");
                }
                (None, _) if !outcome.admissible => {}
                other => panic!("{at}: {other:?}"),
            }
        }
    }
}

/// The criterion the ordinary structural result scored each structure with.
fn combined_criterion(result: &StructuralDecisionResult) -> Option<DecisionCriterion> {
    result.atoms.iter().find_map(|atom| match &atom.status {
        antecedent_design::decision_structural::AtomStatus::Evaluated(r) => Some(r.criterion),
        _ => None,
    })
}

#[test]
fn fixed_trade_offs_decide_across_structures_as_a_hand_written_contract() {
    let model = fixed_model();
    let utility = fixed_utility();
    let [first, second] = structure_laws(&model);
    let unidentified = StructuralAtom {
        id: "s3".into(),
        probability: Some(0.2),
        evidence: AtomEvidence::Unidentified,
    };
    let unevaluated = StructuralAtom {
        id: "s3".into(),
        probability: None,
        evidence: AtomEvidence::Unevaluated("budget spent".into()),
    };
    let structure_sets: Vec<(&str, Vec<StructuralAtom>)> = vec![
        ("disagree", vec![atom("s1", None, &first), atom("s2", None, &second)]),
        ("agree", vec![atom("s1", None, &first), atom("s2", None, &first)]),
        ("unevaluated", vec![atom("s1", None, &first), unevaluated]),
        ("weighted", vec![atom("s1", Some(0.5), &first), atom("s2", Some(0.5), &second)]),
        ("tilted", vec![atom("s1", Some(0.1), &first), atom("s2", Some(0.9), &second)]),
        (
            "unidentified",
            vec![atom("s1", Some(0.4), &first), atom("s2", Some(0.4), &second), unidentified],
        ),
    ];
    let policies = [
        StructuralPolicy::RequireInvariantBestAction,
        StructuralPolicy::Maximin,
        StructuralPolicy::BayesOverStructures,
        StructuralPolicy::ReportOnly,
    ];
    let mut compared = 0;
    for criterion in [
        eu(),
        DecisionCriterion::PosteriorExpectedLoss,
        DecisionCriterion::ThresholdProbability { threshold: 3.1 },
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::Regret,
    ] {
        for policy in policies {
            for (name, atoms) in &structure_sets {
                let label = format!("{name} {policy:?} {criterion:?}");
                let ordinary = evaluate_structural(
                    &hand_written(&model, &utility, criterion, vec![budget()], policy),
                    atoms,
                );
                let c = three_actions(model.clone(), criterion, vec![budget()], policy);
                let ours = decide_preferences_structural(&c, atoms, None, options());
                match (ordinary, ours) {
                    (Ok(theirs), Ok(StructuralPreferenceEvaluation::Exact(ours))) => {
                        compared += 1;
                        assert_structural_agrees(&label, &ours, &theirs);
                    }
                    (Ok(theirs), Ok(StructuralPreferenceEvaluation::Certified(ours))) => {
                        compared += 1;
                        assert_certified_structural_agrees(&label, &ours, &theirs);
                    }
                    // Bayes over structures without probabilities, and a mixture of quantiles
                    // or of largest regrets, are refused by both.
                    (Err(_), Err(_)) => {
                        assert_eq!(policy, StructuralPolicy::BayesOverStructures, "{label}");
                    }
                    (theirs, ours) => panic!("{label}: {theirs:?} {ours:?}"),
                }
            }
        }
    }
    // Every set under every policy, less Bayes without probabilities (three sets) and Bayes
    // over a quantile or the largest regret (every set).
    assert_eq!(compared, 3 * (4 * 6 - 3) + 2 * (3 * 6));
}

#[test]
fn structure_exclusions_follow_the_documented_rules() {
    // `A` breaks the budget under `s1` only and leads under `s2`.
    let model = fixed_model();
    let second = vec![
        vec![vec![10.0, 10.0, 10.0], vec![4.0, 4.0, 4.0]],
        vec![vec![9.0, 5.0, 4.0], vec![1.0, 1.0, 1.0]],
        vec![vec![4.0, 4.0, 4.0], vec![2.0, 2.0, 2.0]],
    ];
    let first = three_action_law(&model);
    let second = law(&model, &second, Some(vec![1.0, 1.0, 2.0]));
    let ordinary = |criterion, policy, atoms: &[StructuralAtom]| {
        evaluate_structural(
            &hand_written(&model, &fixed_utility(), criterion, vec![budget()], policy),
            atoms,
        )
        .unwrap()
    };
    let ours = |criterion, policy, atoms: &[StructuralAtom]| {
        let c = three_actions(model.clone(), criterion, vec![budget()], policy);
        decide_preferences_structural(&c, atoms, None, options()).unwrap()
    };
    let threshold = DecisionCriterion::ThresholdProbability { threshold: 3.1 };

    // Invariance. The ordinary policy compares each structure's own leader, so the structures
    // disagree (`B` in `s1`, `A` in `s2`), and so does the certified route, which decides each
    // structure on its own. The exact route excludes `A` in every structure before analysing
    // each, as its documentation states, so `B` leads in both and is chosen.
    let invariant = StructuralPolicy::RequireInvariantBestAction;
    let atoms = [atom("s1", None, &first), atom("s2", None, &second)];
    for criterion in [eu(), threshold] {
        let theirs = ordinary(criterion, invariant, &atoms);
        assert_eq!(
            theirs.verdict,
            antecedent_design::decision_structural::StructuralVerdict::NoInvariantBest(vec![
                ("s1".into(), vec!["B".into()]),
                ("s2".into(), vec!["A".into()]),
            ])
        );
        match ours(criterion, invariant, &atoms) {
            StructuralPreferenceEvaluation::Exact(result) => {
                assert_eq!(
                    result.verdict,
                    StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest(
                        "B".into()
                    ))
                );
                let combined = result.combined.as_ref().unwrap();
                assert!(!combined.actions[0].admissible);
                assert_eq!(combined.actions[0].exclusions[0].constraint_id, "budget");
            }
            StructuralPreferenceEvaluation::Certified(result) => {
                assert_certified_structural_agrees("certified invariance", &result, &theirs);
            }
        }
    }

    // Bayes weighting. A structure without probability excludes nothing, on either route, as
    // for the ordinary contract: `A` stays admissible and leads.
    let bayes = StructuralPolicy::BayesOverStructures;
    let atoms = [atom("s2", Some(1.0), &second), atom("s1", Some(0.0), &first)];
    for criterion in [eu(), threshold] {
        let theirs = ordinary(criterion, bayes, &atoms);
        assert!(matches!(
            &theirs.verdict,
            antecedent_design::decision_structural::StructuralVerdict::BayesChoice { action, .. }
                if action == "A"
        ));
        let label = format!("zero mass {criterion:?}");
        match ours(criterion, bayes, &atoms) {
            StructuralPreferenceEvaluation::Exact(result) => {
                assert_structural_agrees(&label, &result, &theirs);
                assert!(result.combined.as_ref().unwrap().actions[0].exclusions.is_empty());
            }
            StructuralPreferenceEvaluation::Certified(result) => {
                assert_certified_structural_agrees(&label, &result, &theirs);
                assert!(result.combined.as_ref().unwrap().actions[0].exclusions.is_empty());
            }
        }
    }
}
