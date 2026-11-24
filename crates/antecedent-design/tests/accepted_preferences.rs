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

use antecedent_core::reason_details;
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
        preferences: PreferenceSet::new(
            (0..3).map(|k| parameter(&format!("t{k}"), -4.0, 4.0)).collect(),
            vec![],
        ),
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
        preferences: PreferenceSet::new(parameters, vec![]),
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
    assert_eq!(accepted.stated_comparisons(), 0);
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
    assert_eq!(error.detail(), reason_details::accepted_preferences::NOT_PENDING);
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
    assert_eq!(error.detail(), reason_details::accepted_preferences::ACKNOWLEDGE_EMPTY);
    let error = learned.acknowledge(&[not_varied("t1"), not_varied("t1")]).unwrap_err();
    assert_eq!(error, E::RepeatedItem { item: not_varied("t1") });
    assert_eq!(error.detail(), reason_details::accepted_preferences::ACKNOWLEDGE_REPEATED);
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
fn a_parameter_fixed_by_declaration_raises_no_review_item() {
    // `w` narrows to `[1/2, 1]`; no record varies `harm`. A fixed harm weight is a value stated
    // outright and needs no review; the same parameter given a range is flagged.
    let options = |model: &ValueModel| {
        record_set(
            model,
            vec![
                record("r1", "panel", &[("a", vec![1.0, 1.0, 0.0]), ("b", vec![0.0; 3])], "a"),
                record("r2", "panel", &[("a", vec![1.0, 2.0, 0.0]), ("b", vec![0.0; 3])], "b"),
            ],
            pooled(),
        )
    };
    let fixed = cost_model((0.0, 3.0), Some((0.5, 0.5)));
    let accepted = learn_compatible_set_accepted(
        &fixed,
        &options(&fixed),
        None,
        SetLearningOptions::default(),
    )
    .unwrap();
    assert!(accepted.pending().is_empty(), "{:?}", accepted.pending());

    let ranged = cost_model((0.0, 3.0), Some((0.0, 1.0)));
    let accepted = learn_compatible_set_accepted(
        &ranged,
        &options(&ranged),
        None,
        SetLearningOptions::default(),
    )
    .unwrap();
    assert_eq!(accepted.pending(), vec![not_varied("h")]);
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
    let flagged = AcceptedPreferences::from_estimate(&model, flagged, quick(), &records).unwrap();
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
    assert_eq!(error.detail(), reason_details::accepted_preferences::POOLED_NOT_AVAILABLE);

    let error = learn_compatible_set_accepted(&model, &per_decider, Some("nobody"), tolerant())
        .unwrap_err();
    assert_eq!(error, E::DeciderUnknown { decider: "nobody".to_owned() });
    assert_eq!(error.detail(), reason_details::accepted_preferences::DECIDER_UNKNOWN);

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
    assert_eq!(error.detail(), reason_details::accepted_preferences::CHOICES_CONTRADICTORY);
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
        AcceptedPreferences::from_compatible_set(&other, (**set).clone(), tolerant(), &records)
            .unwrap_err();
    assert_eq!(
        error,
        E::ModelMismatch { expected: other.identity().unwrap(), found: model.identity().unwrap() }
    );
    assert_eq!(error.detail(), reason_details::accepted_preferences::MODEL_MISMATCH);

    let error = AcceptedPreferences::from_compatible_set(
        &model,
        (**set).clone(),
        SetLearningOptions::default(),
        &records,
    )
    .unwrap_err();
    assert_eq!(error, E::OptionsMismatch { option: "tolerance" });
    assert_eq!(error.detail(), reason_details::accepted_preferences::OPTIONS_MISMATCH);
    let accepted =
        AcceptedPreferences::from_compatible_set(&model, (**set).clone(), tolerant(), &records)
            .unwrap();
    assert_eq!(accepted.identity(), learned.identity());

    let cost = cost_model((0.0, 3.0), None);
    let estimate_records = simulated(&cost, 200, 5);
    let estimated = estimate_accepted(&cost, &estimate_records, None, quick()).unwrap();
    let PreferenceBasis::ChoiceEstimate(estimate) = estimated.basis() else { panic!("estimate") };
    let wider = cost_model((0.0, 4.0), None);
    let error = AcceptedPreferences::from_estimate(
        &wider,
        (**estimate).clone(),
        quick(),
        &estimate_records,
    )
    .unwrap_err();
    assert!(matches!(error, E::ModelMismatch { .. }));
    let other_level = ChoiceModelOptions { level: 0.9, ..quick() };
    let error = AcceptedPreferences::from_estimate(
        &cost,
        (**estimate).clone(),
        other_level,
        &estimate_records,
    )
    .unwrap_err();
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

/// `record` read as a stated comparison from an elicitation exercise.
fn stated(mut record: ChoiceRecord) -> ChoiceRecord {
    record.belief_source = BeliefSource::Stated {
        elicitation_id: "elicitation-1".to_owned(),
        assumption_id: "assumption.statements_reflect_preferences".to_owned(),
    };
    record
}

/// The lifecycle records plus the stated comparison `r3`: `t0 >= 1/2 - tolerance`.
fn records_with_statement(pooling: Pooling) -> ChoiceRecordSet {
    let mut records = rate_records(pooling);
    records.records.push(stated(record(
        "r3",
        "panel",
        &[("a", vec![0.0, 1.0, 0.0, 0.0]), ("b", vec![0.5, 0.0, 0.0, 0.0])],
        "a",
    )));
    records
}

#[test]
fn stated_comparisons_among_the_records_are_a_review_item() {
    let model = rate_model();
    let records = records_with_statement(pooled());
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    assert_eq!(learned.stated_comparisons(), 1);
    let used = PendingReview::StatedComparisonsUsed { count: 1 };
    // The set-wide items follow the parameter items; the stated comparisons come last.
    assert_eq!(
        learned.pending(),
        vec![
            not_varied("t1"),
            not_narrowed("t2"),
            PendingReview::ToleranceApplied { tolerance: 0.25 },
            used.clone(),
        ]
    );
    let reviewed = learned.acknowledge(&[used.clone()]).unwrap();
    assert!(!reviewed.pending().contains(&used));
    assert_eq!(reviewed.acknowledged(), &[used.clone()]);

    // Relearning counts again and clears the acknowledgement.
    let relearned = reviewed.relearn_compatible_set(&records, None, tolerant()).unwrap();
    assert_eq!(relearned.stated_comparisons(), 1);
    assert!(relearned.pending().contains(&used));
    // Records without statements raise nothing, even after a handle that had them.
    let plain =
        relearned.relearn_compatible_set(&rate_records(pooled()), None, tolerant()).unwrap();
    assert_eq!(plain.stated_comparisons(), 0);
    assert!(
        !plain
            .pending()
            .iter()
            .any(|item| matches!(item, PendingReview::StatedComparisonsUsed { .. }))
    );

    // A learned set accepted directly counts from its records; the same records give the same
    // handle.
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };
    let direct =
        AcceptedPreferences::from_compatible_set(&model, (**set).clone(), tolerant(), &records)
            .unwrap();
    assert_eq!(direct.identity(), learned.identity());
    assert_eq!(direct.pending(), learned.pending());
}

#[test]
fn stated_comparisons_are_counted_in_the_selected_scope() {
    let model = rate_model();
    let mut records = records_with_statement(Pooling::PerDecider);
    // Another decider's statement is not read for "panel".
    records.records.push(stated(record(
        "q1",
        "other",
        &[("a", vec![0.0, 1.0, 0.0, 0.0]), ("b", vec![0.5, 0.0, 0.0, 0.0])],
        "a",
    )));
    let panel = learn_compatible_set_accepted(&model, &records, Some("panel"), tolerant()).unwrap();
    assert_eq!(panel.stated_comparisons(), 1);
    let other = learn_compatible_set_accepted(&model, &records, Some("other"), tolerant()).unwrap();
    assert_eq!(other.stated_comparisons(), 1);
    let pooled_records = ChoiceRecordSet { pooling: pooled(), ..records };
    let all = learn_compatible_set_accepted(&model, &pooled_records, None, tolerant()).unwrap();
    assert_eq!(all.stated_comparisons(), 2);
    assert!(all.pending().contains(&PendingReview::StatedComparisonsUsed { count: 2 }));
}

#[test]
fn an_estimate_from_stated_comparisons_raises_the_same_item() {
    let model = cost_model((0.0, 3.0), None);
    let mut records = simulated(&model, 300, 13);
    // Stated and recorded comparisons are never fitted together, so every record is stated.
    for record in &mut records.records {
        *record = stated(record.clone());
    }
    let estimated = estimate_accepted(&model, &records, None, quick()).unwrap();
    assert_eq!(estimated.stated_comparisons(), 300);
    assert_eq!(
        estimated.pending().last(),
        Some(&PendingReview::StatedComparisonsUsed { count: 300 })
    );
    let relearned = estimated.relearn_estimate(&records, None, quick()).unwrap();
    assert_eq!(relearned.stated_comparisons(), 300);
    // Accepting the estimate directly counts the statements again: they cannot be left out.
    let PreferenceBasis::ChoiceEstimate(estimate) = estimated.basis() else { panic!("estimate") };
    let direct =
        AcceptedPreferences::from_estimate(&model, (**estimate).clone(), quick(), &records)
            .unwrap();
    assert_eq!(direct.stated_comparisons(), 300);
    assert_eq!(direct.identity(), estimated.identity());
}

#[test]
fn a_learned_set_that_drops_a_declaration_is_refused() {
    let mut model = rate_model();
    model.preferences.relations.push(antecedent_design::preference_model::LinearRelation {
        id: "t1_floor".to_owned(),
        terms: vec![(1, 1.0)],
        sense: antecedent_design::preference_model::RelationSense::AtLeast,
        bound: -3.0,
        origin: RelationOrigin::Declared,
    });
    let records = rate_records(pooled());
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };
    let accept = |edited: &antecedent_design::preference_set_learning::LearnedPreferenceSet| {
        AcceptedPreferences::from_compatible_set(&model, edited.clone(), tolerant(), &records)
    };
    assert_eq!(accept(set).unwrap().identity(), learned.identity());

    // A declared relation removed: the handle would decide over a wider set than declared.
    let mut dropped = (**set).clone();
    dropped.set.relations.retain(|r| r.id != "t1_floor");
    let error = accept(&dropped).unwrap_err();
    assert_eq!(error, E::DeclarationNotKept { part: "relations" });
    assert_eq!(error.detail(), reason_details::accepted_preferences::DECLARATION_NOT_KEPT);
    // A declared range widened.
    let mut widened = (**set).clone();
    widened.set.parameters[1].upper = 9.0;
    assert_eq!(accept(&widened).unwrap_err(), E::DeclarationNotKept { part: "parameters" });

    // An estimate whose entry claims another declared range.
    let cost = cost_model((0.0, 3.0), None);
    let estimate_records = simulated(&cost, 200, 5);
    let estimated = estimate_accepted(&cost, &estimate_records, None, quick()).unwrap();
    let PreferenceBasis::ChoiceEstimate(estimate) = estimated.basis() else { panic!("estimate") };
    let mut claimed = (**estimate).clone();
    claimed.parameters[0].declared = (0.0, 30.0);
    assert_eq!(
        AcceptedPreferences::from_estimate(&cost, claimed, quick(), &estimate_records).unwrap_err(),
        E::DeclarationNotKept { part: "estimate_parameters" }
    );
}

#[test]
fn the_stated_comparison_count_is_read_from_the_records() {
    let model = rate_model();
    let mut records = rate_records(pooled());
    records.records[0].belief_source = BeliefSource::Stated {
        elicitation_id: "elicitation-1".to_owned(),
        assumption_id: "assumption.statements_reflect_preferences".to_owned(),
    };
    let learned = learn_compatible_set_accepted(&model, &records, None, tolerant()).unwrap();
    assert_eq!(learned.stated_comparisons(), 1);
    let PreferenceBasis::CompatibleSet(set) = learned.basis() else { panic!("compatible set") };

    // The count cannot be suppressed: it is counted from the records the set was learned from.
    let accepted =
        AcceptedPreferences::from_compatible_set(&model, (**set).clone(), tolerant(), &records)
            .unwrap();
    assert_eq!(accepted.stated_comparisons(), 1);
    assert!(accepted.pending().contains(&PendingReview::StatedComparisonsUsed { count: 1 }));
    assert_eq!(accepted.identity(), learned.identity());

    // Other records (here the same choices read as recorded forecasts) are not the ones the set
    // was learned from.
    let recorded = rate_records(pooled());
    let error =
        AcceptedPreferences::from_compatible_set(&model, (**set).clone(), tolerant(), &recorded)
            .unwrap_err();
    assert!(matches!(error, E::RecordsMismatch { .. }), "{error:?}");
    assert_eq!(error.detail(), reason_details::accepted_preferences::RECORDS_MISMATCH);

    // A per-decider set is checked against that decider's records.
    let per_decider = ChoiceRecordSet { pooling: Pooling::PerDecider, ..records.clone() };
    let own =
        learn_compatible_set_accepted(&model, &per_decider, Some("panel"), tolerant()).unwrap();
    let PreferenceBasis::CompatibleSet(own_set) = own.basis() else { panic!("compatible set") };
    let accepted = AcceptedPreferences::from_compatible_set(
        &model,
        (**own_set).clone(),
        tolerant(),
        &per_decider,
    )
    .unwrap();
    assert_eq!(accepted.stated_comparisons(), 1);
}
