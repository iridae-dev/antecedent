//! Learning preferences under a value with shape parameters: compatible sets with an analytic
//! interval, proven conflicts, undetermined outcomes, decisions on the learned set against a
//! dense grid, value relations, and the noisy-choice estimate with its identification refusals.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::reason_details;
use antecedent_design::accepted_preferences::{
    AcceptedPreferencesError, estimate_accepted, learn_compatible_set_accepted,
};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordError, ChoiceRecordSet, OptionBelief,
    Pooling, choice_features, choice_features_at,
};
use antecedent_design::decision_contract::{ActionKind, DecisionCriterion, StructuralPolicy};
use antecedent_design::preference_artifact::AcceptedPreferencesArtifact;
use antecedent_design::preference_certified::{
    CertifiedOptions, CertifiedVerdict, evaluate_preference_certified,
};
use antecedent_design::preference_choice_model::{
    ChoiceModelError, ChoiceModelOptions, ParameterIntervalMethod, estimate_choice_model,
};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{
    CertifiedRouteReason, PreferenceDecisionError, PreferenceDecisionOptions,
    evaluate_preference_contract,
};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluationOptions, PreferenceRoute, ROUTE_VALUE_RELATIONS, decide_preferences,
    preference_route,
};
use antecedent_design::preference_model::{
    Attribute, BeliefRows, LinearRelation, PreferenceExpr as P, PreferenceModelError,
    PreferenceParameter, PreferenceSet, RelationOrigin, RelationSense, ValueModel, ValueRelation,
};
use antecedent_design::preference_set_learning::{
    CellStatus, ConflictKind, RangeBasis, SetLearningError, SetLearningOptions, SetOutcome,
    learn_compatible_set,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

const UNITS: &str = "u";

/// `r* = -ln((sqrt(5) - 1) / 2)`: choosing a sure 1 over an even lottery on 0 and 3 under
/// `v = -exp(-r x)` holds exactly when `exp(-r) <= (1 + exp(-3 r)) / 2`, that is, with
/// `u = exp(-r)`, when `(u - 1)(u^2 + u - 1) >= 0`, so for `r >= r*`.
fn risk_threshold() -> f64 {
    -((5.0_f64.sqrt() - 1.0) / 2.0).ln()
}

fn attribute(id: &str) -> Attribute {
    Attribute { id: id.into(), units: UNITS.into() }
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "u/u".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

/// `-exp(-r x)`.
fn exponential(r: usize, x: usize) -> P {
    P::negate(P::exp(P::negate(P::product(P::Parameter(r), P::Attribute(x)))))
}

/// `v(x) = -exp(-r x)` with the risk attitude `r` in `[lower, upper]`.
fn risk_model(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: vec![attribute("x")],
        preferences: PreferenceSet::new(vec![parameter("r", lower, upper)], vec![]),
        value: exponential(0, 0),
        utility_units: UNITS.into(),
    }
}

fn draws(rows: &[f64], weights: Option<Vec<f64>>) -> OptionBelief {
    OptionBelief::Draws { rows: rows.to_vec(), weights }
}

fn record(
    id: &str,
    decider: &str,
    options: Vec<(&str, OptionBelief)>,
    chosen: &str,
) -> ChoiceRecord {
    ChoiceRecord {
        id: id.into(),
        decider: decider.into(),
        options: options
            .into_iter()
            .map(|(id, belief)| ChoiceOption { id: id.into(), belief })
            .collect(),
        chosen: chosen.into(),
        belief_source: BeliefSource::RecordedForecast { source_id: format!("forecast-{id}") },
        observation_rows: vec![format!("ledger:{id}")],
    }
}

fn pooled(attributes: Vec<Attribute>, records: Vec<ChoiceRecord>) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes,
        population: "target".into(),
        records,
        pooling: Pooling::Pooled { assumption_id: "one_set_of_preferences".into() },
    }
}

/// A sure 1 chosen over an even lottery on 0 and 3: `r >= r*`.
fn prudent(id: &str) -> ChoiceRecord {
    record(
        id,
        "panel",
        vec![("sure", draws(&[1.0], None)), ("lottery", draws(&[0.0, 3.0], Some(vec![0.5, 0.5])))],
        "sure",
    )
}

/// An even lottery on 0 and 3 chosen over a sure 1.2: `r` at most about 0.275.
fn bold(id: &str) -> ChoiceRecord {
    record(
        id,
        "panel",
        vec![("sure", draws(&[1.2], None)), ("lottery", draws(&[0.0, 3.0], Some(vec![0.5, 0.5])))],
        "lottery",
    )
}

fn only_pooled(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
) -> SetOutcome {
    learn_compatible_set(model, records, options).expect("learned").pooled.expect("pooled")
}

#[test]
fn a_risk_attitude_interval_is_enclosed_from_one_recorded_choice() {
    let model = risk_model(0.1, 3.0);
    let records = pooled(vec![attribute("x")], vec![prudent("r1")]);
    let SetOutcome::Learned(learned) = only_pooled(&model, &records, SetLearningOptions::default())
    else {
        panic!("learned");
    };
    let threshold = risk_threshold();
    let range = &learned.ranges[0];
    assert_eq!(range.basis, RangeBasis::Enclosure);
    // An outer range: it holds the exact interval [r*, 3] and is at most one box wider.
    assert!(range.learned.0 <= threshold && range.learned.0 >= threshold - 1e-2, "{range:?}");
    assert!(range.learned.1.total_cmp(&3.0).is_eq());
    assert!(range.narrowed && range.varied_in_records);
    let region = learned.region.as_ref().expect("region");
    assert_eq!(
        (region.shape_parameters.clone(), region.linear_parameters.clone()),
        (vec![0], vec![])
    );
    assert!(!region.budget_exhausted);
    // Inside boxes lie inside the exact interval.
    let inside: Vec<_> = region.cells.iter().filter(|c| c.status == CellStatus::Inside).collect();
    assert!(!inside.is_empty());
    for cell in &inside {
        assert!(cell.shape_lower[0] >= threshold - 1e-9, "{cell:?}");
    }
    // The learned set holds the choice as a value relation and decides membership exactly.
    assert!(learned.binding.is_empty());
    assert_eq!(learned.set.value_relations.len(), 1);
    assert_eq!(learned.set.value_relations[0].id, "choice.r1.lottery");
    let set_model = ValueModel { preferences: learned.set.clone(), ..model.clone() };
    set_model.validate().expect("learned model validates");
    for i in 0..=290 {
        let r = 0.1 + 0.01 * f64::from(i);
        if (r - threshold).abs() > 1e-6 {
            assert_eq!(set_model.contains(&[r], 0.0), r > threshold, "r = {r}");
        }
    }
    // The declared model's identity is unchanged by learning; the learned model's differs.
    assert_ne!(set_model.identity().unwrap(), model.identity().unwrap());
}

#[test]
fn records_that_cannot_hold_together_are_proven_to_conflict() {
    let model = risk_model(0.1, 3.0);
    let records = pooled(vec![attribute("x")], vec![prudent("r1"), bold("r2")]);
    let SetOutcome::Contradictory(conflict) =
        only_pooled(&model, &records, SetLearningOptions::default())
    else {
        panic!("contradictory");
    };
    assert_eq!(conflict.records, vec!["r1".to_owned(), "r2".to_owned()]);
    assert_eq!(conflict.kind, ConflictKind::Smallest);
    assert!(conflict.tolerance_needed > 0.0 && conflict.tolerance_needed.is_finite());
    let error =
        learn_compatible_set_accepted(&model, &records, None, SetLearningOptions::default())
            .unwrap_err();
    assert!(matches!(error, AcceptedPreferencesError::Contradictory { .. }));
    // With the reported tolerance the records hold together.
    let tolerant =
        SetLearningOptions { tolerance: conflict.tolerance_needed * 1.5, ..Default::default() };
    assert!(matches!(only_pooled(&model, &records, tolerant), SetOutcome::Learned(_)));
}

#[test]
fn an_exhausted_box_budget_leaves_the_set_undetermined() {
    let model = risk_model(0.1, 3.0);
    let records = pooled(vec![attribute("x")], vec![prudent("r1")]);
    let one_box = SetLearningOptions { max_boxes: 1, ..Default::default() };
    let SetOutcome::Undetermined(undetermined) = only_pooled(&model, &records, one_box) else {
        panic!("undetermined");
    };
    assert!(undetermined.region.budget_exhausted);
    assert_eq!(undetermined.region.boxes_used, 1);
    assert!(undetermined.region.cells.iter().all(|c| c.status == CellStatus::Boundary));
    let error = learn_compatible_set_accepted(&model, &records, None, one_box).unwrap_err();
    assert_eq!(error, AcceptedPreferencesError::Undetermined { decider: None, boundary_boxes: 1 });
    assert_eq!(error.detail(), reason_details::accepted_preferences::SET_UNDETERMINED);

    let zero = SetLearningOptions { max_boxes: 0, ..Default::default() };
    assert_eq!(
        learn_compatible_set(&model, &records, zero).unwrap_err(),
        SetLearningError::RegionBudgetInvalid { option: "max_boxes" }
    );
    let wide = SetLearningOptions { min_width_fraction: 1.5, ..Default::default() };
    let error = learn_compatible_set(&model, &records, wide).unwrap_err();
    assert_eq!(error, SetLearningError::RegionBudgetInvalid { option: "min_width_fraction" });
    assert_eq!(error.detail(), reason_details::preference_set_learning::REGION_BUDGET_INVALID);
}

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        regime_id: format!("do({name}=1)"),
        units: UNITS.into(),
        ..common::quantity(name)
    }
}

/// An exact law of one column with equally weighted draws.
fn law(name: &str, values: &[f64]) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[quantity(name)],
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
            shape: [values.len(), 1],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        values.to_vec(),
    )
    .unwrap()
}

/// `safe` pays 1 for sure; `risky` pays 0 or 2.5 evenly. Under `v = -exp(-r x)`, `safe` is
/// better exactly when `exp(-r) < (1 + exp(-2.5 r)) / 2`, which holds for every `r` above about
/// 0.32 and so on all of `[r*, 3]`.
fn lottery_contract(model: ValueModel) -> PreferenceContract {
    PreferenceContract {
        model,
        actions: vec![
            PreferenceAction {
                id: "safe".into(),
                kind: ActionKind::Intervention,
                inputs: vec![ActionInput::Fixed { value: 1.0, units: UNITS.into() }],
            },
            PreferenceAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![ActionInput::Quantity(quantity("x"))],
            },
        ],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

#[test]
fn a_learned_risk_attitude_decides_on_the_certified_route_as_a_dense_grid_does() {
    let model = risk_model(0.1, 3.0);
    let records = pooled(vec![attribute("x")], vec![prudent("r1")]);
    let accepted =
        learn_compatible_set_accepted(&model, &records, None, SetLearningOptions::default())
            .expect("accepted");
    let source = law("x", &[0.0, 2.5]);
    let learned = lottery_contract(accepted.model().clone());
    let evaluation =
        decide_preferences(&learned, &source, None, PreferenceEvaluationOptions::default())
            .expect("decided");
    assert_eq!(evaluation.route(), PreferenceRoute::Certified);
    let result =
        evaluate_preference_certified(&learned, &source, None, CertifiedOptions::default())
            .expect("certified");
    assert_eq!(result.verdict, CertifiedVerdict::NecessarilyBest("safe".into()));
    // Dense-grid ground truth over the learned set: safe leads at every admissible point.
    let mut admitted = 0;
    for i in 0..=2900 {
        let r = 0.1 + 0.001 * f64::from(i);
        if !learned.model.contains(&[r], 0.0) {
            continue;
        }
        admitted += 1;
        let safe = -(-r).exp();
        let risky = -0.5 * (1.0 + (-2.5 * r).exp());
        assert!(safe > risky, "risky leads at r = {r}");
    }
    assert!(admitted > 2000);
    // Over the declared range the choice depends on the risk attitude.
    let declared = lottery_contract(model);
    let result =
        evaluate_preference_certified(&declared, &source, None, CertifiedOptions::default())
            .expect("certified");
    assert!(
        matches!(result.verdict, CertifiedVerdict::PreferenceDependent { .. }),
        "{:?}",
        result.verdict
    );
}

#[test]
fn the_learned_region_survives_the_accepted_preferences_artifact() {
    let model = risk_model(0.1, 3.0);
    let records = pooled(vec![attribute("x")], vec![prudent("r1")]);
    let options = SetLearningOptions { min_width_fraction: 1e-3, ..Default::default() };
    let accepted =
        learn_compatible_set_accepted(&model, &records, None, options).expect("accepted");
    let bytes = AcceptedPreferencesArtifact::new(accepted.clone()).to_bytes("accepted").unwrap();
    let loaded = AcceptedPreferencesArtifact::from_bytes(&bytes, accepted.identity()).unwrap();
    assert_eq!(loaded.accepted(), &accepted);
    loaded.replay(&records).expect("replayed");
}

/// `v = w x` with `w` in `[0, 2]` and the value relation `w * 1 - w * 2 >= 0`, so `w = 0`.
fn linear_with_value_relation() -> ValueModel {
    let mut preferences = PreferenceSet::new(vec![parameter("w", 0.0, 2.0)], vec![]);
    preferences.value_relations.push(ValueRelation {
        id: "v1".into(),
        origin: RelationOrigin::Choice { record_id: "r1".into(), rejected_option: "b".into() },
        chosen: BeliefRows { rows: vec![1.0], weights: vec![1.0] },
        rejected: BeliefRows { rows: vec![2.0], weights: vec![1.0] },
        tolerance: 0.0,
    });
    ValueModel {
        attributes: vec![attribute("x")],
        preferences,
        value: P::product(P::Parameter(0), P::Attribute(0)),
        utility_units: UNITS.into(),
    }
}

#[test]
fn value_relations_are_checked_identified_and_sent_to_the_certified_route() {
    let model = linear_with_value_relation();
    model.validate().expect("valid");
    assert!(model.contains(&[0.0], 0.0));
    assert!(!model.contains(&[0.5], 0.0));
    // The set alone cannot read a value relation.
    assert!(model.preferences.contains(&[0.5], 0.0));
    assert_eq!(model.value_relation_gap(&model.preferences.value_relations[0], &[0.5]), Ok(-0.5));

    let plain = ValueModel {
        preferences: PreferenceSet::new(model.preferences.parameters.clone(), vec![]),
        ..model.clone()
    };
    let explicit = ValueModel {
        preferences: PreferenceSet {
            parameters: plain.preferences.parameters.clone(),
            relations: vec![],
            value_relations: vec![],
        },
        ..plain.clone()
    };
    assert_eq!(plain.identity().unwrap(), explicit.identity().unwrap());
    assert_ne!(plain.identity().unwrap(), model.identity().unwrap());

    let invalid = |edit: &dyn Fn(&mut ValueRelation)| {
        let mut broken = model.clone();
        edit(&mut broken.preferences.value_relations[0]);
        broken.validate().unwrap_err()
    };
    assert_eq!(
        invalid(&|r| r.tolerance = -1.0),
        PreferenceModelError::ValueRelationToleranceInvalid { relation: "v1".into() }
    );
    assert_eq!(
        invalid(&|r| r.chosen.weights = vec![0.0]),
        PreferenceModelError::ValueRelationRowsInvalid { relation: "v1".into() }
    );
    assert_eq!(
        invalid(&|r| r.rejected.rows = vec![2.0, 3.0]),
        PreferenceModelError::ValueRelationRowsInvalid { relation: "v1".into() }
    );
    let mut duplicate = model.clone();
    duplicate.preferences.relations.push(LinearRelation {
        id: "v1".into(),
        terms: vec![(0, 1.0)],
        sense: RelationSense::AtLeast,
        bound: 0.0,
        origin: RelationOrigin::Declared,
    });
    assert_eq!(
        duplicate.validate().unwrap_err(),
        PreferenceModelError::DuplicateRelation { id: "v1".into() }
    );

    let contract = PreferenceContract {
        actions: vec![
            PreferenceAction {
                id: "a".into(),
                kind: ActionKind::Intervention,
                inputs: vec![ActionInput::Fixed { value: 1.0, units: UNITS.into() }],
            },
            PreferenceAction {
                id: "b".into(),
                kind: ActionKind::Intervention,
                inputs: vec![ActionInput::Quantity(quantity("x"))],
            },
        ],
        ..lottery_contract(model)
    };
    assert_eq!(preference_route(&contract).unwrap().reason, ROUTE_VALUE_RELATIONS);
    let source = law("x", &[0.0, 2.5]);
    assert_eq!(
        evaluate_preference_contract(
            &contract,
            &source,
            None,
            PreferenceDecisionOptions::default()
        )
        .unwrap_err(),
        PreferenceDecisionError::CertifiedRouteRequired {
            reason: CertifiedRouteReason::ValueRelations { relations: vec!["v1".into()] }
        }
    );
}

#[test]
fn forward_gradients_match_central_differences() {
    // max(p0 x0, ln(p1 + x1)) * exp(-p2 x0) - min(p0, p1 * p2)
    let value = P::difference(
        P::product(
            P::maximum(
                P::product(P::Parameter(0), P::Attribute(0)),
                P::ln(P::sum(P::Parameter(1), P::Attribute(1))),
            ),
            P::exp(P::negate(P::product(P::Parameter(2), P::Attribute(0)))),
        ),
        P::minimum(P::Parameter(0), P::product(P::Parameter(1), P::Parameter(2))),
    );
    let attributes = [1.3, 0.4];
    for point in [[0.2, 1.1, 0.7], [1.5, 0.3, -0.4], [0.9, 2.0, 0.1]] {
        let (at, gradient) = value.evaluate_with_gradient(&attributes, &point).unwrap();
        assert!((at - value.evaluate(&attributes, &point).unwrap()).abs() <= 0.0);
        for j in 0..3 {
            let h = 1e-6;
            let (mut up, mut down) = (point, point);
            up[j] += h;
            down[j] -= h;
            let numeric = (value.evaluate(&attributes, &up).unwrap()
                - value.evaluate(&attributes, &down).unwrap())
                / (2.0 * h);
            assert!((numeric - gradient[j]).abs() <= 1e-6 * (1.0 + numeric.abs()), "{point:?} {j}");
        }
    }
    let ln = P::ln(P::Parameter(0));
    assert_eq!(ln.evaluate_with_gradient(&[], &[-1.0]), Err(PreferenceModelError::NonFiniteValue));
}

#[test]
fn features_at_a_shape_value_substitute_it_and_agree_with_the_linear_features() {
    // A linear model: the features at any point are the linear features.
    let linear = ValueModel {
        attributes: vec![attribute("x"), attribute("h")],
        preferences: PreferenceSet::new(vec![parameter("w", 0.0, 2.0)], vec![]),
        value: P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
        utility_units: UNITS.into(),
    };
    let records = pooled(
        vec![attribute("x"), attribute("h")],
        vec![record(
            "r1",
            "panel",
            vec![
                ("a", OptionBelief::Means(vec![2.0, 1.0])),
                ("b", draws(&[1.0, 0.0, 3.0, 2.0], Some(vec![1.0, 3.0]))),
            ],
            "a",
        )],
    );
    assert_eq!(
        choice_features_at(&linear, &records, &[0.7]).unwrap(),
        choice_features(&linear, &records).unwrap()
    );
    // A risk model: the plain features refuse the free shape parameter; at a point they hold
    // the expectation of the value there.
    let risky = risk_model(0.1, 3.0);
    let lottery = pooled(vec![attribute("x")], vec![prudent("r1")]);
    assert_eq!(
        choice_features(&risky, &lottery).unwrap_err(),
        ChoiceRecordError::ShapeParametersNotFixed { parameters: vec!["r".into()] }
    );
    let features = choice_features_at(&risky, &lottery, &[0.5]).unwrap();
    let fixed = &features.sets[0].fixed;
    assert!((fixed[0] + (-0.5_f64).exp()).abs() <= 1e-15);
    assert!((fixed[1] + 0.5 * (1.0 + (-1.5_f64).exp())).abs() <= 1e-15);
    assert_eq!(features.sets[0].terms, vec![0.0, 0.0]);
    assert_eq!(
        choice_features_at(&risky, &lottery, &[0.5, 1.0]).unwrap_err(),
        ChoiceRecordError::Model(PreferenceModelError::ParameterCount { expected: 1, found: 2 })
    );
    assert_eq!(
        choice_features_at(&risky, &lottery, &[f64::NAN]).unwrap_err(),
        ChoiceRecordError::Model(PreferenceModelError::NonFiniteParameterValue { index: 0 })
    );
}

// ---------------------------------------------------------------------------------------------
// Noisy choices.
// ---------------------------------------------------------------------------------------------

/// A deterministic uniform generator.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        f64::from(u32::try_from(self.0 >> 40).unwrap()) / f64::from(1_u32 << 24)
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// `n` two-option records, each option a sure bundle of attribute values from `draw`, the first
/// chosen with logit probability `lambda * (value(first) - value(second))`.
fn simulated(
    attributes: &[&str],
    n: usize,
    seed: u64,
    lambda: f64,
    value: impl Fn(&[f64]) -> f64,
    draw: impl Fn(&mut Lcg) -> Vec<f64>,
    as_means: bool,
) -> ChoiceRecordSet {
    let mut rng = Lcg(seed);
    let records = (0..n)
        .map(|i| {
            let (first, second) = (draw(&mut rng), draw(&mut rng));
            let gap = lambda * (value(&first) - value(&second));
            let chosen = if rng.unit() < 1.0 / (1.0 + (-gap).exp()) { "a" } else { "b" };
            let belief = |values: Vec<f64>| {
                if as_means { OptionBelief::Means(values) } else { draws(&values, None) }
            };
            record(
                &format!("n{i}"),
                "panel",
                vec![("a", belief(first)), ("b", belief(second))],
                chosen,
            )
        })
        .collect();
    pooled(attributes.iter().map(|a| attribute(a)).collect(), records)
}

/// `v = -exp(-r x) - w h`: a risk attitude `r` and a harm weight `w`.
fn risk_and_harm() -> ValueModel {
    ValueModel {
        attributes: vec![attribute("x"), attribute("h")],
        preferences: PreferenceSet::new(
            vec![parameter("r", 0.05, 3.0), parameter("w", 0.0, 3.0)],
            vec![],
        ),
        value: P::difference(exponential(0, 0), P::product(P::Parameter(1), P::Attribute(1))),
        utility_units: UNITS.into(),
    }
}

fn quick() -> ChoiceModelOptions {
    ChoiceModelOptions { n_draws: 300, ..ChoiceModelOptions::default() }
}

#[test]
fn a_risk_attitude_and_a_harm_weight_are_recovered_from_noisy_choices() {
    let records = simulated(
        &["x", "h"],
        500,
        13,
        5.0,
        |v| -(-0.8 * v[0]).exp() - 0.6 * v[1],
        |rng| vec![rng.between(0.0, 4.0), rng.between(0.0, 2.0)],
        false,
    );
    let model = risk_and_harm();
    let estimate = estimate_choice_model(&model, &records, quick()).unwrap().pooled.unwrap();
    assert_eq!(estimate.shape_parameters, vec!["r".to_owned()]);
    let by_id = |id: &str| estimate.parameters.iter().find(|p| p.parameter_id == id).unwrap();
    let (r, w) = (by_id("r"), by_id("w"));
    assert!((r.estimate - 0.8).abs() < 0.3, "{r:?}");
    assert!((w.estimate - 0.6).abs() < 0.25, "{w:?}");
    assert_eq!(r.interval_method, ParameterIntervalMethod::ProfileLikelihood);
    assert_eq!(w.interval_method, ParameterIntervalMethod::Wald);
    assert!(r.interval.0 < r.estimate && r.estimate < r.interval.1);
    assert!(w.interval.0 < w.estimate && w.estimate < w.interval.1);
    assert_eq!(estimate.coefficient_covariance.len(), 9);
    assert_eq!(estimate.interval_calibration, DistributionCalibration::NotMeasured);
    assert!(estimate.not_learnable.is_empty());
    assert!(!estimate.measure.points.is_empty() && estimate.measure.unresolved.is_empty());
    for point in &estimate.measure.points {
        assert!(model.contains(point, 1e-9));
    }
    // A seed reproduces the measure; another seed draws another one.
    let again = estimate_choice_model(&model, &records, quick()).unwrap().pooled.unwrap();
    assert_eq!(again.measure, estimate.measure);
    assert_eq!(again.identity, estimate.identity);
    let other = ChoiceModelOptions { seed: 9, ..quick() };
    let reseeded = estimate_choice_model(&model, &records, other).unwrap().pooled.unwrap();
    assert_ne!(reseeded.measure.points, estimate.measure.points);
    // The accepted estimate survives its artifact and replays from the records.
    let accepted = estimate_accepted(&model, &records, None, quick()).unwrap();
    let bytes = AcceptedPreferencesArtifact::new(accepted.clone()).to_bytes("estimate").unwrap();
    let loaded = AcceptedPreferencesArtifact::from_bytes(&bytes, accepted.identity()).unwrap();
    assert_eq!(loaded.accepted(), &accepted);
    loaded.replay(&records).expect("replayed");
}

#[test]
fn a_shape_parameter_the_value_differences_never_read_is_not_learnable() {
    // v = x - w h + exp(r): exp(r) adds the same amount to every option.
    let model = ValueModel {
        attributes: vec![attribute("x"), attribute("h")],
        preferences: PreferenceSet::new(
            vec![parameter("w", 0.0, 3.0), parameter("r", 0.0, 1.0)],
            vec![],
        ),
        value: P::sum(
            P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1))),
            P::exp(P::Parameter(1)),
        ),
        utility_units: UNITS.into(),
    };
    let records = simulated(
        &["x", "h"],
        300,
        19,
        2.0,
        |v| v[0] - 1.2 * v[1],
        |rng| vec![rng.between(0.0, 3.0), rng.between(0.0, 2.0)],
        true,
    );
    let estimate = estimate_choice_model(&model, &records, quick()).unwrap().pooled.unwrap();
    assert_eq!(estimate.not_learnable, vec!["r".to_owned()]);
    assert!(estimate.shape_parameters.is_empty());
    assert_eq!(estimate.parameters.len(), 1);
    assert!((estimate.parameters[0].estimate - 1.2).abs() < 0.4);
    assert_eq!(estimate.measure.unresolved, vec![1]);
}

#[test]
fn a_barely_informed_shape_parameter_is_refused() {
    let model = risk_model(0.05, 2.0);
    let records = simulated(
        &["x"],
        300,
        17,
        2000.0,
        |v| -(-0.05 * v[0]).exp(),
        |rng| vec![rng.between(0.0, 0.02)],
        false,
    );
    let error = estimate_choice_model(&model, &records, quick()).unwrap_err();
    assert!(
        matches!(&error, ChoiceModelError::ShapeNotIdentified { parameter, .. } if parameter == "r"),
        "{error:?}"
    );
    assert_eq!(error.detail(), reason_details::preference_choice_model::SHAPE_NOT_IDENTIFIED);
}

#[test]
fn a_shape_estimate_on_a_declared_bound_refuses_the_measure() {
    // Risk-neutral choices read with a risk attitude of at least 0.3.
    let model = risk_model(0.3, 2.0);
    let records =
        simulated(&["x"], 800, 23, 2.0, |v| v[0], |rng| vec![rng.between(0.0, 5.0)], false);
    let error = estimate_choice_model(&model, &records, quick()).unwrap_err();
    let ChoiceModelError::ShapeEstimateAtBound { parameter, estimate, interval, .. } = &error
    else {
        panic!("{error:?}");
    };
    assert_eq!(parameter, "r");
    assert!((estimate - 0.3).abs() <= 1e-6);
    assert!(interval.0.total_cmp(&0.3).is_eq() && interval.1 > 0.3);
    assert_eq!(error.detail(), reason_details::preference_choice_model::SHAPE_ESTIMATE_AT_BOUND);
}

#[test]
fn a_profile_with_two_separated_maxima_is_refused() {
    // v = x + z exp(-(s - 1)^2) is symmetric in s about 1: every fit at s has its mirror image.
    let bump = P::exp(P::negate(P::product(
        P::difference(P::Parameter(0), P::Const(1.0)),
        P::difference(P::Parameter(0), P::Const(1.0)),
    )));
    let model = ValueModel {
        attributes: vec![attribute("x"), attribute("z")],
        preferences: PreferenceSet::new(vec![parameter("s", 0.0, 2.0)], vec![]),
        value: P::sum(P::Attribute(0), P::product(P::Attribute(1), bump)),
        utility_units: UNITS.into(),
    };
    let records = simulated(
        &["x", "z"],
        1500,
        3,
        5.0,
        |v| v[0] + 0.5 * v[1],
        |rng| vec![rng.between(-1.0, 1.0), rng.between(-2.0, 2.0)],
        true,
    );
    let error = estimate_choice_model(&model, &records, quick()).unwrap_err();
    assert!(
        matches!(&error, ChoiceModelError::ProfileMultimodal { parameter, modes, .. }
            if parameter == "s" && modes.0 < 0.6 && modes.1 > 1.4),
        "{error:?}"
    );
    assert_eq!(error.detail(), reason_details::preference_choice_model::PROFILE_MULTIMODAL);
}

#[test]
fn more_shape_parameters_than_the_search_covers_are_refused() {
    // v = x (exp(a) + exp(b) + exp(c) + exp(d)): four shape parameters, each read with x.
    let sum = (1..5).map(|j| P::exp(P::Parameter(j - 1))).reduce(P::sum).unwrap();
    let model = ValueModel {
        attributes: vec![attribute("x")],
        preferences: PreferenceSet::new(
            ["a", "b", "c", "d"].iter().map(|id| parameter(id, 0.0, 1.0)).collect(),
            vec![],
        ),
        value: P::product(P::Attribute(0), sum),
        utility_units: UNITS.into(),
    };
    let records = simulated(&["x"], 50, 29, 1.0, |v| v[0], |rng| vec![rng.between(0.0, 1.0)], true);
    let error = estimate_choice_model(&model, &records, quick()).unwrap_err();
    assert_eq!(
        error,
        ChoiceModelError::ShapeSearchBudget { decider: None, dimensions: 4, limit: 3 }
    );
    assert_eq!(error.detail(), reason_details::preference_choice_model::SHAPE_SEARCH_BUDGET);
}
