//! Accepted preferences: the lifecycle from declaration through learning, review and explicit
//! relearning (versions, pending review items, acknowledgements, identity), the checks that a
//! learned object belongs to the declared model, decider selection, and every refusal.
//!
//! The lifecycle model is `x0 + t0 x1 + t1 x2 + t2 x3` with every rate in `[-4, 4]`. Record `r1`
//! chose `x1 = 1` over `x0 = 1`, so `t0 >= 1 - tolerance`; record `r2` chose `x3 = 1` over
//! `x0 = -10`, so `t2 >= -10 - tolerance`, which does not narrow `[-4, 4]`; no record varies
//! `x2`, so `t1` keeps its declared range.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_design::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError as E, LearningOptions, PendingReview,
    PreferenceBasis, estimate_accepted, learn_compatible_set_accepted,
};
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::preference_choice_model::ChoiceModelOptions;
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as P, PreferenceParameter, PreferenceSet, RelationOrigin, ValueModel,
};
use antecedent_design::preference_set_learning::{ASSUMPTION_CHOICES_OPTIMAL, SetLearningOptions};

/// Small fixed-seed xorshift64* generator for simulating choices.
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

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "utility/unit".to_owned(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.to_owned(), units: units.to_owned() }
}

/// `x0 + t0 x1 + t1 x2 + t2 x3`, every rate in `[-4, 4]`.
fn rate_model() -> ValueModel {
    let value = (0..3).fold(P::Attribute(0), |sum, k| {
        P::sum(sum, P::product(P::Parameter(k), P::Attribute(k + 1)))
    });
    ValueModel {
        attributes: (0..4).map(|k| attribute(&format!("x{k}"), &format!("unit_{k}"))).collect(),
        preferences: PreferenceSet {
            parameters: (0..3).map(|k| parameter(&format!("t{k}"), -4.0, 4.0)).collect(),
            relations: vec![],
        },
        value,
        utility_units: "utility".to_owned(),
    }
}

/// `benefit - w * cost`, plus `- h * harm` when `harm` ranges are given.
fn cost_model(w: (f64, f64), harm: Option<(f64, f64)>) -> ValueModel {
    let mut attributes = vec![attribute("benefit", "events"), attribute("cost", "usd")];
    let mut parameters = vec![parameter("w", w.0, w.1)];
    let mut value = P::difference(P::Attribute(0), P::product(P::Parameter(0), P::Attribute(1)));
    if let Some((lower, upper)) = harm {
        attributes.push(attribute("harm", "events"));
        parameters.push(parameter("h", lower, upper));
        value = P::difference(value, P::product(P::Parameter(1), P::Attribute(2)));
    }
    ValueModel {
        attributes,
        preferences: PreferenceSet { parameters, relations: vec![] },
        value,
        utility_units: "events".to_owned(),
    }
}

fn record(id: &str, decider: &str, options: &[(&str, Vec<f64>)], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: decider.to_owned(),
        options: options
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).to_owned(),
                belief: OptionBelief::Means(means.clone()),
            })
            .collect(),
        chosen: chosen.to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: format!("forecast-{id}") },
        observation_rows: vec![format!("{id}#row")],
    }
}

fn record_set(model: &ValueModel, records: Vec<ChoiceRecord>, pooling: Pooling) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "review panel".to_owned(),
        records,
        pooling,
    }
}

fn pooled() -> Pooling {
    Pooling::Pooled { assumption_id: "assumption.one_panel".to_owned() }
}

/// `t0 >= 1 - tolerance` and `t2 >= -10 - tolerance`.
fn rate_records(pooling: Pooling) -> ChoiceRecordSet {
    let model = rate_model();
    record_set(
        &model,
        vec![
            record(
                "r1",
                "panel",
                &[("a", vec![0.0, 1.0, 0.0, 0.0]), ("b", vec![1.0, 0.0, 0.0, 0.0])],
                "a",
            ),
            record(
                "r2",
                "panel",
                &[("a", vec![0.0, 0.0, 0.0, 1.0]), ("b", vec![-10.0, 0.0, 0.0, 0.0])],
                "a",
            ),
        ],
        pooling,
    )
}

fn tolerant() -> SetLearningOptions {
    SetLearningOptions { tolerance: 0.25, ..SetLearningOptions::default() }
}

/// Logit choices with consistency one and value `benefit - cost` (harm, when present, is one
/// in every option and so never varies).
fn simulated(model: &ValueModel, n: usize, seed: u64) -> ChoiceRecordSet {
    let mut rng = Xorshift::new(seed);
    let with_harm = model.attributes.len() == 3;
    let records = (0..n)
        .map(|r| {
            let options: Vec<Vec<f64>> = (0..3)
                .map(|_| {
                    let mut means = vec![4.0 * rng.uniform(), 4.0 * rng.uniform()];
                    if with_harm {
                        means.push(1.0);
                    }
                    means
                })
                .collect();
            let chosen = options
                .iter()
                .map(|o| o[0] - o[1] + rng.gumbel())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(0, |(i, _)| i);
            let named: Vec<(String, Vec<f64>)> =
                options.into_iter().enumerate().map(|(i, o)| (format!("o{i}"), o)).collect();
            let borrowed: Vec<(&str, Vec<f64>)> =
                named.iter().map(|(id, o)| (id.as_str(), o.clone())).collect();
            record(&format!("s{r}"), "panel", &borrowed, &format!("o{chosen}"))
        })
        .collect();
    record_set(model, records, pooled())
}

fn quick() -> ChoiceModelOptions {
    ChoiceModelOptions { n_draws: 200, ..ChoiceModelOptions::default() }
}

fn not_varied(id: &str) -> PendingReview {
    PendingReview::ParameterNotVaried { parameter_id: id.to_owned() }
}

fn not_narrowed(id: &str) -> PendingReview {
    PendingReview::ParameterNotNarrowed { parameter_id: id.to_owned() }
}

#[test]
fn declared_preferences_start_at_version_one_with_nothing_pending() {
    let model = rate_model();
    let accepted = AcceptedPreferences::declared(model.clone()).unwrap();
    assert_eq!(accepted.version(), 1);
    assert_eq!(accepted.basis(), &PreferenceBasis::Declared);
    assert!(accepted.pending().is_empty() && accepted.acknowledged().is_empty());
    assert_eq!(accepted.model(), &model);
    assert_eq!(accepted.declared_model(), &model);
    assert_eq!(accepted.declared_identity(), model.identity().unwrap());
    assert!(accepted.measure().is_none() && accepted.learning_options().is_none());
    assert_eq!(accepted.decider(), None);
    assert_eq!(accepted.assumptions(), vec!["t0-judgment", "t1-judgment", "t2-judgment"]);
    // The identity is a function of the content: declaring the same model again gives it.
    let again = AcceptedPreferences::declared(model).unwrap();
    assert_eq!(again.identity(), accepted.identity());
    assert_eq!(accepted.identity().len(), 64);
}

#[test]
fn learning_reviewing_and_relearning_each_give_a_new_version_and_identity() {
    let model = rate_model();
    let records = rate_records(pooled());
    let declared = AcceptedPreferences::declared(model.clone()).unwrap();

    let learned = declared.relearn_compatible_set(&records, None, tolerant()).unwrap();
    assert_eq!(learned.version(), 2);
    let all = vec![
        not_varied("t1"),
        not_narrowed("t2"),
        PendingReview::ToleranceApplied { tolerance: 0.25 },
    ];
    assert_eq!(learned.pending(), all);
    assert_ne!(learned.identity(), declared.identity());
    assert_eq!(learned.learning_options(), Some(LearningOptions::CompatibleSet(tolerant())));
    // The handle the learning started from is unchanged.
    assert_eq!((declared.version(), declared.basis()), (1, &PreferenceBasis::Declared));

    // The accepted model carries the learned set: the declared parameters plus the choice
    // relations.
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };
    assert_eq!(learned.model().preferences, set.set);
    assert_eq!(learned.declared_model(), &model);
    assert!(learned.model().preferences.relations.iter().all(|r| matches!(
        &r.origin,
        RelationOrigin::Choice { record_id, .. } if record_id == "r1" || record_id == "r2"
    )));
    assert!(learned.assumptions().iter().any(|a| a == ASSUMPTION_CHOICES_OPTIMAL));
    assert!(learned.assumptions().iter().any(|a| a == "assumption.one_panel"));

    // Acknowledging one item.
    let reviewed = learned.acknowledge(&[not_varied("t1")]).unwrap();
    assert_eq!(reviewed.version(), 3);
    assert_eq!(reviewed.pending(), all[1..].to_vec());
    assert_eq!(reviewed.acknowledged(), &[not_varied("t1")]);
    assert_ne!(reviewed.identity(), learned.identity());
    assert_eq!(learned.pending().len(), 3, "the earlier handle is unchanged");

    // Acknowledged items are kept in the order pending lists them.
    let all_reviewed = reviewed
        .acknowledge(&[PendingReview::ToleranceApplied { tolerance: 0.25 }, not_narrowed("t2")])
        .unwrap();
    assert_eq!(all_reviewed.version(), 4);
    assert!(all_reviewed.pending().is_empty());
    assert_eq!(all_reviewed.acknowledged(), all.as_slice());

    // An item that is no longer pending, or never was, is refused.
    let error = reviewed.acknowledge(&[not_varied("t1")]).unwrap_err();
    assert_eq!(error, E::NotPending { item: not_varied("t1") });
    assert_eq!(error.detail(), "accepted_preferences.not_pending");
    let foreign = PendingReview::ParameterNotLearnable { parameter_id: "t0".to_owned() };
    assert_eq!(learned.acknowledge(&[foreign.clone()]), Err(E::NotPending { item: foreign }));
    let other_tolerance = PendingReview::ToleranceApplied { tolerance: 0.5 };
    assert!(learned.acknowledge(&[other_tolerance]).is_err());

    // Relearning is explicit, increases the version and clears the acknowledgements.
    let relearned = reviewed.relearn_compatible_set(&records, None, tolerant()).unwrap();
    assert_eq!(relearned.version(), 4);
    assert!(relearned.acknowledged().is_empty());
    assert_eq!(relearned.pending(), all);
    let identities = [
        declared.identity(),
        learned.identity(),
        reviewed.identity(),
        all_reviewed.identity(),
        relearned.identity(),
    ];
    for (i, a) in identities.iter().enumerate() {
        for b in &identities[i + 1..] {
            assert_ne!(a, b);
        }
    }

    // The same learning accepted directly is version one with the same items.
    let direct = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    assert_eq!(direct.version(), 1);
    assert_eq!(direct.pending(), all);
    assert_eq!(direct.basis(), learned.basis());
}

#[test]
fn acknowledging_nothing_or_one_item_twice_is_refused() {
    let learned =
        learn_compatible_set_accepted(&rate_model(), &rate_records(pooled()), None, tolerant())
            .unwrap();
    let error = learned.acknowledge(&[]).unwrap_err();
    assert_eq!(error, E::NoItems);
    assert_eq!(error.detail(), "accepted_preferences.acknowledge_empty");
    let error = learned.acknowledge(&[not_varied("t1"), not_varied("t1")]).unwrap_err();
    assert_eq!(error, E::RepeatedItem { item: not_varied("t1") });
    assert_eq!(error.detail(), "accepted_preferences.acknowledge_repeated");
}

#[test]
fn a_set_learned_without_tolerance_and_narrowing_every_parameter_has_nothing_pending() {
    // `w <= 1` from `r1` and `w >= 1/2` from `r2`.
    let model = cost_model((0.0, 3.0), None);
    let records = record_set(
        &model,
        vec![
            record("r1", "panel", &[("a", vec![1.0, 1.0]), ("b", vec![0.0, 0.0])], "a"),
            record("r2", "panel", &[("a", vec![1.0, 2.0]), ("b", vec![0.0, 0.0])], "b"),
        ],
        pooled(),
    );
    let accepted =
        learn_compatible_set_accepted(&model, &records, None, SetLearningOptions::default())
            .unwrap();
    assert!(accepted.pending().is_empty());
    let PreferenceBasis::CompatibleSet(set) = accepted.basis() else { panic!("compatible set") };
    assert!((set.ranges[0].learned.0 - 0.5).abs() < 1e-9);
    assert!((set.ranges[0].learned.1 - 1.0).abs() < 1e-9);
}

#[test]
fn estimate_review_items_follow_parameter_order() {
    // Harm is one in every option, so `h` is not learnable.
    let model = cost_model((0.0, 1.1), Some((0.0, 2.0)));
    let records = simulated(&model, 400, 7);
    let accepted = estimate_accepted(&model, &records, None, quick()).unwrap();
    assert_eq!(accepted.version(), 1);
    let PreferenceBasis::ChoiceEstimate(estimate) = accepted.basis() else { panic!("estimate") };
    let outside = PendingReview::EstimateOutsideDeclared { parameter_id: "w".into() };
    let not_learnable = PendingReview::ParameterNotLearnable { parameter_id: "h".into() };
    let mut expected = Vec::new();
    if !estimate.parameters[0].inside_declared {
        expected.push(outside.clone());
    }
    expected.push(not_learnable.clone());
    if estimate.share_outside_declared > 0.0 {
        expected
            .push(PendingReview::DrawsOutsideDeclared { share: estimate.share_outside_declared });
    }
    assert_eq!(accepted.pending(), expected);
    assert_eq!(accepted.measure(), Some(&estimate.measure));
    assert_eq!(accepted.measure().unwrap().unresolved, vec![1]);
    assert_eq!(accepted.model(), &model, "an estimate keeps the declared set");
    assert_eq!(accepted.learning_options(), Some(LearningOptions::ChoiceEstimate(quick())));

    // A point estimate outside its range and dropped draws, stated on the estimate itself.
    let mut flagged = (**estimate).clone();
    flagged.parameters[0].inside_declared = false;
    flagged.share_outside_declared = 0.25;
    let flagged = AcceptedPreferences::from_estimate(&model, flagged, quick()).unwrap();
    let dropped = PendingReview::DrawsOutsideDeclared { share: 0.25 };
    assert_eq!(flagged.pending(), vec![outside.clone(), not_learnable.clone(), dropped.clone()]);

    let reviewed = flagged.acknowledge(&[dropped.clone(), outside.clone()]).unwrap();
    assert_eq!(reviewed.version(), 2);
    assert_eq!(reviewed.pending(), vec![not_learnable]);
    assert_eq!(reviewed.acknowledged(), &[outside, dropped]);

    // Re-estimating replaces the stated estimate with the fitted one and clears the review.
    let relearned = reviewed.relearn_estimate(&records, None, quick()).unwrap();
    assert_eq!(relearned.version(), 3);
    assert_eq!(relearned.basis(), accepted.basis());
    assert_eq!(relearned.pending(), expected);
    assert!(relearned.acknowledged().is_empty());
}

#[test]
fn deciders_are_selected_by_name() {
    let model = rate_model();
    let per_decider = rate_records(Pooling::PerDecider);
    let accepted =
        learn_compatible_set_accepted(&model, &per_decider, Some("panel"), tolerant()).unwrap();
    assert_eq!(accepted.decider(), Some("panel"));

    let error = learn_compatible_set_accepted(&model, &per_decider, None, tolerant()).unwrap_err();
    assert_eq!(error, E::PooledNotAvailable);
    assert_eq!(error.detail(), "accepted_preferences.pooled_not_available");

    let error = learn_compatible_set_accepted(&model, &per_decider, Some("nobody"), tolerant())
        .unwrap_err();
    assert_eq!(error, E::DeciderUnknown { decider: "nobody".to_owned() });
    assert_eq!(error.detail(), "accepted_preferences.decider_unknown");

    // An estimate under pooled records fits no decider on its own.
    let cost = cost_model((0.0, 3.0), None);
    let records = simulated(&cost, 200, 3);
    let error = estimate_accepted(&cost, &records, Some("panel"), quick()).unwrap_err();
    assert_eq!(error, E::DeciderUnknown { decider: "panel".to_owned() });
    let per_decider = ChoiceRecordSet { pooling: Pooling::PerDecider, ..records };
    assert_eq!(estimate_accepted(&cost, &per_decider, None, quick()), Err(E::PooledNotAvailable));
    let by_name = estimate_accepted(&cost, &per_decider, Some("panel"), quick()).unwrap();
    assert_eq!(by_name.decider(), Some("panel"));
}

#[test]
fn contradictory_choices_are_refused_with_the_conflict() {
    // `t0 >= 1` and `t0 <= 1/2` cannot both hold.
    let model = rate_model();
    let records = record_set(
        &model,
        vec![
            record(
                "r1",
                "panel",
                &[("a", vec![0.0, 1.0, 0.0, 0.0]), ("b", vec![1.0, 0.0, 0.0, 0.0])],
                "a",
            ),
            record(
                "r3",
                "panel",
                &[("a", vec![0.0, 1.0, 0.0, 0.0]), ("b", vec![0.5, 0.0, 0.0, 0.0])],
                "b",
            ),
        ],
        pooled(),
    );
    let error =
        learn_compatible_set_accepted(&model, &records, None, SetLearningOptions::default())
            .unwrap_err();
    let E::Contradictory { decider, conflict } = &error else { panic!("{error:?}") };
    assert_eq!(decider, &None);
    assert_eq!(conflict.records, vec!["r1".to_owned(), "r3".to_owned()]);
    assert_eq!(error.detail(), "accepted_preferences.choices_contradictory");
}

#[test]
fn a_learned_object_of_another_model_or_other_options_is_refused() {
    let model = rate_model();
    let records = rate_records(pooled());
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };

    let mut other = model.clone();
    other.preferences.parameters[0].upper = 5.0;
    let error =
        AcceptedPreferences::from_compatible_set(&other, (**set).clone(), tolerant()).unwrap_err();
    assert_eq!(
        error,
        E::ModelMismatch { expected: other.identity().unwrap(), found: model.identity().unwrap() }
    );
    assert_eq!(error.detail(), "accepted_preferences.model_mismatch");

    let error = AcceptedPreferences::from_compatible_set(
        &model,
        (**set).clone(),
        SetLearningOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error, E::OptionsMismatch { option: "tolerance" });
    assert_eq!(error.detail(), "accepted_preferences.options_mismatch");
    let accepted =
        AcceptedPreferences::from_compatible_set(&model, (**set).clone(), tolerant()).unwrap();
    assert_eq!(accepted.identity(), learned.identity());

    let cost = cost_model((0.0, 3.0), None);
    let estimated = estimate_accepted(&cost, &simulated(&cost, 200, 5), None, quick()).unwrap();
    let PreferenceBasis::ChoiceEstimate(estimate) = estimated.basis() else { panic!("estimate") };
    let wider = cost_model((0.0, 4.0), None);
    let error =
        AcceptedPreferences::from_estimate(&wider, (**estimate).clone(), quick()).unwrap_err();
    assert!(matches!(error, E::ModelMismatch { .. }));
    let other_level = ChoiceModelOptions { level: 0.9, ..quick() };
    let error =
        AcceptedPreferences::from_estimate(&cost, (**estimate).clone(), other_level).unwrap_err();
    assert_eq!(error, E::OptionsMismatch { option: "level" });
}

#[test]
fn refusals_of_the_model_and_the_learners_keep_their_own_detail() {
    let mut empty = rate_model();
    empty.attributes.clear();
    let error = AcceptedPreferences::declared(empty).unwrap_err();
    assert!(matches!(&error, E::Model(inner) if error.detail() == inner.detail()));

    let negative = SetLearningOptions { tolerance: -1.0, ..SetLearningOptions::default() };
    let error =
        learn_compatible_set_accepted(&rate_model(), &rate_records(pooled()), None, negative)
            .unwrap_err();
    assert!(matches!(&error, E::SetLearning(inner) if error.detail() == inner.detail()));
    assert!(error.detail().starts_with("preference_set_learning."));

    let cost = cost_model((0.0, 3.0), None);
    let bad_level = ChoiceModelOptions { level: 2.0, ..quick() };
    let error = estimate_accepted(&cost, &simulated(&cost, 20, 1), None, bad_level).unwrap_err();
    assert!(matches!(&error, E::ChoiceModel(inner) if error.detail() == inner.detail()));
    assert!(error.detail().starts_with("preference_choice_model."));

    // A per-decider refusal of the estimate is surfaced for the selected decider.
    let flat: Vec<ChoiceRecord> = (0..4)
        .map(|r| {
            record(&format!("f{r}"), "panel", &[("a", vec![1.0, 1.0]), ("b", vec![1.0, 2.0])], "a")
        })
        .collect();
    let records = record_set(&cost, flat, Pooling::PerDecider);
    let error = estimate_accepted(&cost, &records, Some("panel"), quick()).unwrap_err();
    assert!(matches!(error, E::ChoiceModel(_)), "{error:?}");
}

#[test]
fn a_contract_over_accepted_preferences_uses_the_accepted_model() {
    use antecedent_design::accepted_preferences::PreferenceContractParts;
    use antecedent_design::decision_contract::{ActionKind, DecisionCriterion, StructuralPolicy};
    use antecedent_design::preference_contract::{ActionInput, PreferenceAction, PreferencePolicy};

    let model = rate_model();
    let accepted =
        learn_compatible_set_accepted(&model, &rate_records(pooled()), None, tolerant()).unwrap();
    let actions: Vec<PreferenceAction> = ["A", "B"]
        .iter()
        .map(|id| PreferenceAction {
            id: (*id).to_owned(),
            kind: ActionKind::Intervention,
            inputs: model
                .attributes
                .iter()
                .map(|a| ActionInput::Fixed { value: 1.0, units: a.units.clone() })
                .collect(),
        })
        .collect();
    let contract = accepted.contract(PreferenceContractParts {
        actions: actions.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".to_owned(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
        preference_policy: PreferencePolicy::ReportOnly,
    });
    assert_eq!(&contract.model, accepted.model());
    assert_eq!(contract.actions, actions);
    assert_eq!(contract.preference_policy, PreferencePolicy::ReportOnly);
}
