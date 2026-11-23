//! One entry point over the exact and certified verdicts: the route rule and its labels, the
//! fall-through from the exact route, mean sources, accepted preferences of every basis decided
//! end to end, and a cross-check of the two routes on random affine problems.
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
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionResult, MeanEstimate, MeanSource, Verdict, evaluate_contract,
    evaluate_contract_on_means,
};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::preference_certified::{
    CertifiedOptions, CertifiedVerdict, RegionStatus, evaluate_preference_certified,
};
use antecedent_design::preference_choice_model::ChoiceModelOptions;
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{MeasureOrigin, PreferenceVerdict};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationError as E, PreferenceEvaluationOptions,
    PreferenceRoute, ROUTE_AFFINE_EXPECTATION, ROUTE_CRITERION_NOT_AFFINE,
    ROUTE_PART_NOT_LOWERABLE, ROUTE_SHAPE_PARAMETERS, RouteChoice, decide_preferences,
    decide_preferences_on_means, decide_preferences_structural, decide_with_accepted,
    preference_route,
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
        preferences: PreferenceSet {
            parameters: vec![parameter("w", lower, upper)],
            relations: vec![],
        },
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
        preferences: PreferenceSet { parameters, relations: vec![] },
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
        preferences: PreferenceSet {
            parameters: vec![parameter("u", 0.9, 0.9), parameter("w", 1.2, 1.2)],
            relations: vec![],
        },
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
        preferences: PreferenceSet { parameters: vec![], relations: vec![] },
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
                // A criterion a mean source cannot answer is refused by both.
                Err(_) => {
                    assert!(evaluation.is_err(), "{label}: only the ordinary contract refused")
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
