//! Compatible preference sets from recorded choices: hand-derived ranges, binding choices,
//! conflicts and the tolerance they need, per-decider, pooled and overlapping sets, shape-rule
//! relations, recovery of a known parameter vector from simulated choices, invariance under
//! record order, unavailable options that teach nothing, stated comparisons, and every refusal.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_details;
use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
    choice_features,
};
use antecedent_design::preference_model::{
    Attribute, Curve, LinearRelation, PreferenceExpr as E, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use antecedent_design::preference_set_learning::{
    ASSUMPTION_CHOICES_OPTIMAL, BindingChoice, ConflictKind, LearnedPreferenceSet,
    SetLearningError, SetLearningOptions, SetLearningResult, SetOutcome, learn_compatible_set,
    uninformative_records,
};

fn attributes(n: usize) -> Vec<Attribute> {
    (0..n).map(|k| Attribute { id: format!("x{k}"), units: format!("unit_{k}") }).collect()
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "utility/unit".to_owned(),
        lower,
        upper,
        assumption_id: format!("declared_{id}"),
    }
}

/// `x0 + sum_k t_k * x_{k+1}` with `t_k` in the given ranges.
fn rate_model(ranges: &[(f64, f64)]) -> ValueModel {
    let parameters: Vec<PreferenceParameter> = ranges
        .iter()
        .enumerate()
        .map(|(k, (lo, hi))| parameter(&format!("t{k}"), *lo, *hi))
        .collect();
    let value = (0..parameters.len()).fold(E::Attribute(0), |sum, k| {
        E::sum(sum, E::product(E::Parameter(k), E::Attribute(k + 1)))
    });
    ValueModel {
        attributes: attributes(parameters.len() + 1),
        preferences: PreferenceSet { parameters, relations: Vec::new() },
        value,
        utility_units: "utility".to_owned(),
    }
}

fn one_parameter() -> ValueModel {
    rate_model(&[(-4.0, 4.0)])
}

fn record(id: &str, decider: &str, options: &[(&str, &[f64])], chosen: &str) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: decider.to_owned(),
        options: options
            .iter()
            .map(|(o, means)| ChoiceOption {
                id: (*o).to_owned(),
                belief: OptionBelief::Means(means.to_vec()),
            })
            .collect(),
        chosen: chosen.to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast".to_owned() },
        observation_rows: vec![format!("{id}#row")],
    }
}

/// `t >= 1`.
fn at_least_one(decider: &str) -> ChoiceRecord {
    record("r1", decider, &[("a", &[0.0, 1.0]), ("b", &[1.0, 0.0])], "a")
}

/// `t <= 3`.
fn at_most_three(decider: &str) -> ChoiceRecord {
    record("r2", decider, &[("a", &[0.0, 1.0]), ("b", &[3.0, 0.0])], "b")
}

/// `t <= 1/2`.
fn at_most_half(decider: &str) -> ChoiceRecord {
    record("r3", decider, &[("a", &[0.0, 1.0]), ("b", &[0.5, 0.0])], "b")
}

fn record_set(model: &ValueModel, records: Vec<ChoiceRecord>, pooling: Pooling) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "decision makers".to_owned(),
        records,
        pooling,
    }
}

fn learn(model: &ValueModel, records: &ChoiceRecordSet, tolerance: f64) -> SetLearningResult {
    let options = SetLearningOptions { tolerance, ..Default::default() };
    learn_compatible_set(model, records, options).expect("learned")
}

fn learned(outcome: &SetOutcome) -> &LearnedPreferenceSet {
    match outcome {
        SetOutcome::Learned(set) => set,
        SetOutcome::Contradictory(conflict) => panic!("contradictory: {conflict:?}"),
    }
}

fn only(result: &SetLearningResult) -> &SetOutcome {
    assert_eq!(result.per_decider.len(), 1);
    &result.per_decider[0].outcome
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() <= 1e-9 * (1.0 + want.abs()), "got {got}, want {want}");
}

fn range_is(set: &LearnedPreferenceSet, k: usize, lower: f64, upper: f64) {
    close(set.ranges[k].learned.0, lower);
    close(set.ranges[k].learned.1, upper);
}

fn binding(pairs: &[(&str, &str)]) -> Vec<BindingChoice> {
    pairs
        .iter()
        .map(|(r, o)| BindingChoice {
            record_id: (*r).to_owned(),
            rejected_option: (*o).to_owned(),
        })
        .collect()
}

#[test]
fn records_on_both_sides_bracket_one_parameter() {
    let model = one_parameter();
    let records =
        record_set(&model, vec![at_least_one("d"), at_most_three("d")], Pooling::PerDecider);
    let result = learn(&model, &records, 0.0);
    assert!(result.pooled.is_none() && result.overlap.is_none());
    let set = learned(only(&result));
    range_is(set, 0, 1.0, 3.0);
    let range = &set.ranges[0];
    assert_eq!(range.parameter_id, "t0");
    assert_eq!(range.declared, (-4.0, 4.0));
    assert!(range.varied_in_records && range.narrowed);
    assert_eq!(set.binding, binding(&[("r1", "b"), ("r2", "a")]));
    assert_eq!(set.decider.as_deref(), Some("d"));
    assert_eq!(set.n_records, 2);
    assert!(set.tolerance.abs() <= 0.0);
    assert_eq!(set.model_identity, model.identity().expect("identity"));
    assert_eq!(set.records_digest, records.digest().expect("digest"));
    assert_eq!(set.assumptions, [ASSUMPTION_CHOICES_OPTIMAL, "declared_t0"]);
    set.set.validate().expect("learned set validates");
    assert_eq!(set.set.parameters, model.preferences.parameters);
    let ids: Vec<&str> = set.set.relations.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["choice.r1.b", "choice.r2.a"]);
    let first = &set.set.relations[0];
    assert_eq!(first.sense, RelationSense::AtLeast);
    assert_eq!(first.terms, [(0, 1.0)]);
    close(first.bound, 1.0);
    assert_eq!(
        first.origin,
        RelationOrigin::Choice { record_id: "r1".to_owned(), rejected_option: "b".to_owned() }
    );
    assert!(set.set.contains(&[2.0], 0.0));
    assert!(!set.set.contains(&[0.5], 1e-9));
}

#[test]
fn a_record_that_varies_nothing_narrows_nothing() {
    let model = rate_model(&[(-2.0, 2.0), (-2.0, 2.0)]);
    let r = record("r1", "d", &[("a", &[0.0, 1.0, 0.5]), ("b", &[1.0, 0.0, 0.5])], "a");
    let set_records = record_set(&model, vec![r], Pooling::PerDecider);
    let result = learn(&model, &set_records, 0.0);
    let set = learned(only(&result));
    range_is(set, 0, 1.0, 2.0);
    range_is(set, 1, -2.0, 2.0);
    assert!(set.ranges[0].varied_in_records && set.ranges[0].narrowed);
    assert!(!set.ranges[1].varied_in_records && !set.ranges[1].narrowed);
    assert_eq!(set.set.relations[0].terms, [(0, 1.0)]);
}

#[test]
fn a_fixed_parameter_keeps_its_value_and_is_never_varied() {
    let model = rate_model(&[(-2.0, 2.0), (0.5, 0.5)]);
    let r = record("r1", "d", &[("a", &[0.0, 1.0, 1.0]), ("b", &[1.0, 0.0, 0.0])], "a");
    let result = learn(&model, &record_set(&model, vec![r], Pooling::PerDecider), 0.0);
    let set = learned(only(&result));
    // t0 + 0.5 >= 1.
    range_is(set, 0, 0.5, 2.0);
    range_is(set, 1, 0.5, 0.5);
    assert!(!set.ranges[1].varied_in_records && !set.ranges[1].narrowed);
}

#[test]
fn a_contradictory_pair_is_the_smallest_conflict_with_its_needed_tolerance() {
    let model = one_parameter();
    let records = record_set(
        &model,
        vec![at_least_one("d"), at_most_three("d"), at_most_half("d")],
        Pooling::PerDecider,
    );
    let SetOutcome::Contradictory(conflict) = only(&learn(&model, &records, 0.0)).clone() else {
        panic!("expected a conflict");
    };
    assert_eq!(conflict.records, ["r1", "r3"]);
    assert_eq!(conflict.kind, ConflictKind::Smallest);
    // t >= 1 - s and t <= 1/2 + s need s >= 1/4.
    close(conflict.tolerance_needed, 0.25);

    // The needed tolerance is a total: the same with part of it already declared.
    let SetOutcome::Contradictory(partial) = only(&learn(&model, &records, 0.125)).clone() else {
        panic!("expected a conflict");
    };
    close(partial.tolerance_needed, 0.25);
    assert!(partial.tolerance_needed > 0.125);

    // At the needed tolerance the records are compatible, at one point.
    let at_needed = learn(&model, &records, 0.25);
    range_is(learned(only(&at_needed)), 0, 0.75, 0.75);

    // A search limited to single records falls back to the deletion filter.
    let options = SetLearningOptions { conflict_search_size: 1, ..Default::default() };
    let result = learn_compatible_set(&model, &records, options).expect("learned");
    let SetOutcome::Contradictory(irreducible) = only(&result) else {
        panic!("expected a conflict");
    };
    assert_eq!(irreducible.kind, ConflictKind::Irreducible);
    assert_eq!(irreducible.records, ["r1", "r3"]);
}

#[test]
fn a_record_that_contradicts_the_model_alone_is_a_single_record_conflict() {
    let model = one_parameter();
    // Same term, lower fixed part chosen: 0 >= 1 whatever t is.
    let bad = record("r9", "d", &[("a", &[0.0, 1.0]), ("b", &[1.0, 1.0])], "a");
    let records = record_set(&model, vec![at_least_one("d"), bad], Pooling::PerDecider);
    let SetOutcome::Contradictory(conflict) = only(&learn(&model, &records, 0.0)).clone() else {
        panic!("expected a conflict");
    };
    assert_eq!(conflict.records, ["r9"]);
    assert_eq!(conflict.kind, ConflictKind::Smallest);
    close(conflict.tolerance_needed, 1.0);
}

#[test]
fn tolerance_widens_the_learned_range() {
    let model = one_parameter();
    let records =
        record_set(&model, vec![at_least_one("d"), at_most_three("d")], Pooling::PerDecider);
    let mut previous = (1.0, 3.0);
    for tolerance in [0.25, 0.5, 1.0] {
        let result = learn(&model, &records, tolerance);
        let set = learned(only(&result));
        range_is(set, 0, 1.0 - tolerance, 3.0 + tolerance);
        assert!(set.ranges[0].learned.0 <= previous.0 && set.ranges[0].learned.1 >= previous.1);
        close(set.set.relations[0].bound, 1.0 - tolerance);
        previous = set.ranges[0].learned;
    }
}

#[test]
fn per_decider_sets_use_their_own_records_and_pooling_uses_all() {
    let model = one_parameter();
    let pooled_records = record_set(
        &model,
        vec![at_least_one("a"), at_most_three("b")],
        Pooling::Pooled { assumption_id: "shared_preferences".to_owned() },
    );
    let result = learn(&model, &pooled_records, 0.0);
    assert!(result.overlap.is_none());
    let pooled = learned(result.pooled.as_ref().expect("pooled"));
    range_is(pooled, 0, 1.0, 3.0);
    assert!(pooled.decider.is_none());
    assert!(pooled.assumptions.contains(&"shared_preferences".to_owned()));
    assert_eq!(pooled.records_digest, pooled_records.digest().expect("digest"));
    let deciders: Vec<&str> = result.per_decider.iter().map(|d| d.decider.as_str()).collect();
    assert_eq!(deciders, ["a", "b"]);
    let a = learned(&result.per_decider[0].outcome);
    let b = learned(&result.per_decider[1].outcome);
    assert_eq!(result.per_decider[0].n_records, 1);
    range_is(a, 0, 1.0, 4.0);
    range_is(b, 0, -4.0, 3.0);
    assert!(!a.assumptions.contains(&"shared_preferences".to_owned()));
    assert_eq!(a.set.relations.len(), 1);

    let separate =
        record_set(&model, vec![at_least_one("a"), at_most_three("b")], Pooling::PerDecider);
    let result = learn(&model, &separate, 0.0);
    assert!(result.pooled.is_none());
    let overlap = result.overlap.as_ref().expect("overlap");
    range_is(learned(&overlap.common), 0, 1.0, 3.0);
    assert!(overlap.disagreeing.is_empty());
}

#[test]
fn deciders_with_disjoint_sets_are_both_named_in_the_overlap() {
    let model = one_parameter();
    let records =
        record_set(&model, vec![at_most_half("b"), at_least_one("a")], Pooling::PerDecider);
    let result = learn(&model, &records, 0.0);
    range_is(learned(&result.per_decider[0].outcome), 0, 1.0, 4.0);
    range_is(learned(&result.per_decider[1].outcome), 0, -4.0, 0.5);
    let overlap = result.overlap.as_ref().expect("overlap");
    let SetOutcome::Contradictory(conflict) = &overlap.common else {
        panic!("expected a conflict");
    };
    assert_eq!(conflict.records, ["r1", "r3"]);
    close(conflict.tolerance_needed, 0.25);
    assert_eq!(overlap.disagreeing, ["a", "b"]);
}

#[test]
fn a_curve_with_shape_rules_keeps_them_and_is_narrowed_through_them() {
    let curve = Curve { attribute: 1, breakpoints: vec![0.0, 1.0, 2.0], slopes: vec![0, 1] };
    let mut relations = curve.increasing("dose");
    relations.extend(curve.diminishing("dose"));
    let model = ValueModel {
        attributes: attributes(2),
        preferences: PreferenceSet {
            parameters: vec![parameter("s0", 0.0, 4.0), parameter("s1", 0.0, 4.0)],
            relations,
        },
        value: E::sum(E::Attribute(0), curve.expr()),
        utility_units: "utility".to_owned(),
    };
    let draws = |x0: f64, x1: f64| OptionBelief::Draws { rows: vec![x0, x1], weights: None };
    let r = ChoiceRecord {
        id: "r1".to_owned(),
        decider: "d".to_owned(),
        options: vec![
            ChoiceOption { id: "high_dose".to_owned(), belief: draws(0.0, 2.0) },
            ChoiceOption { id: "none".to_owned(), belief: draws(1.5, 0.0) },
        ],
        chosen: "high_dose".to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast".to_owned() },
        observation_rows: vec!["r1#row".to_owned()],
    };
    let result = learn(&model, &record_set(&model, vec![r], Pooling::PerDecider), 0.0);
    let set = learned(only(&result));
    // s0 + s1 >= 3/2 with 0 <= s1 <= s0 <= 4.
    range_is(set, 0, 0.75, 4.0);
    range_is(set, 1, 0.0, 4.0);
    assert!(set.ranges[0].narrowed && !set.ranges[1].narrowed);
    assert_eq!(set.set.relations.len(), 4);
    assert!(set.set.relations[..3].iter().all(|r| r.origin == RelationOrigin::ShapeRule));
    assert_eq!(set.set.relations[3].id, "choice.r1.none");
    assert_eq!(set.binding, binding(&[("r1", "none")]));
}

/// Fixed-seed linear congruential generator.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    #[allow(clippy::cast_precision_loss, reason = "a 53-bit integer is exact in f64")]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn between(&mut self, lower: f64, upper: f64) -> f64 {
        lower + (upper - lower) * self.unit()
    }
}

/// Records with 2 to 4 options of random attribute means, each choosing the option of highest
/// value under `truth` for the value `x0 + t0 * x1 + t1 * x2`.
fn simulated(rng: &mut Lcg, truth: &[f64], n: usize) -> Vec<ChoiceRecord> {
    (0..n)
        .map(|i| {
            let n_options = 2 + usize::try_from(rng.next_u64() % 3).expect("small");
            let options: Vec<ChoiceOption> = (0..n_options)
                .map(|o| ChoiceOption {
                    id: format!("o{o}"),
                    belief: OptionBelief::Means(vec![
                        rng.between(-1.0, 1.0),
                        rng.between(-1.0, 1.0),
                        rng.between(-1.0, 1.0),
                    ]),
                })
                .collect();
            let value = |o: &ChoiceOption| match &o.belief {
                OptionBelief::Means(m) => m[0] + truth[0] * m[1] + truth[1] * m[2],
                OptionBelief::Draws { .. } | OptionBelief::Unavailable { .. } => f64::NAN,
            };
            let chosen = options
                .iter()
                .max_by(|a, b| value(a).total_cmp(&value(b)))
                .map(|o| o.id.clone())
                .expect("options");
            ChoiceRecord {
                id: format!("s{i:03}"),
                decider: "d".to_owned(),
                options,
                chosen,
                belief_source: BeliefSource::RecordedForecast { source_id: "forecast".to_owned() },
                observation_rows: vec![format!("s{i:03}#row")],
            }
        })
        .collect()
}

#[test]
fn simulated_choices_recover_a_set_holding_the_truth_that_shrinks_with_records() {
    let model = rate_model(&[(-2.0, 2.0), (-2.0, 2.0)]);
    let mut rng = Lcg(20_251_123);
    let truth = [rng.between(-2.0, 2.0), rng.between(-2.0, 2.0)];
    let all = simulated(&mut rng, &truth, 320);
    let mut previous: Option<Vec<(f64, f64)>> = None;
    let mut widths = Vec::new();
    for n in [20, 80, 320] {
        let records = record_set(&model, all[..n].to_vec(), Pooling::PerDecider);
        let result = learn(&model, &records, 0.0);
        let set = learned(only(&result));
        assert!(set.set.contains(&truth, 1e-9), "truth outside the set at {n} records");
        let ranges: Vec<(f64, f64)> = set.ranges.iter().map(|r| r.learned).collect();
        for (k, (lo, hi)) in ranges.iter().enumerate() {
            assert!(*lo <= truth[k] + 1e-9 && truth[k] <= *hi + 1e-9);
        }
        if let Some(wider) = &previous {
            for ((lo, hi), (wlo, whi)) in ranges.iter().zip(wider) {
                assert!(*lo >= wlo - 1e-9 && *hi <= whi + 1e-9, "ranges not nested at {n}");
            }
        }
        widths.push(ranges.iter().map(|(lo, hi)| hi - lo).sum::<f64>());
        previous = Some(ranges);
    }
    assert!(widths[2] < widths[0], "no shrinkage: {widths:?}");
    assert!(widths[2] < 0.5, "320 records leave a wide set: {widths:?}");
}

#[test]
fn learned_ranges_do_not_depend_on_record_order() {
    let model = rate_model(&[(-2.0, 2.0), (-2.0, 2.0)]);
    let mut rng = Lcg(7);
    let truth = [rng.between(-2.0, 2.0), rng.between(-2.0, 2.0)];
    let forward = simulated(&mut rng, &truth, 60);
    let mut backward = forward.clone();
    backward.reverse();
    let a = learn(&model, &record_set(&model, forward, Pooling::PerDecider), 0.0);
    let b = learn(&model, &record_set(&model, backward, Pooling::PerDecider), 0.0);
    let (a, b) = (learned(only(&a)), learned(only(&b)));
    for (x, y) in a.ranges.iter().zip(&b.ranges) {
        close(x.learned.0, y.learned.0);
        close(x.learned.1, y.learned.1);
        assert_eq!(x.narrowed, y.narrowed);
        assert_eq!(x.varied_in_records, y.varied_in_records);
    }
    assert_eq!(a.records_digest, b.records_digest);
    let ids = |set: &LearnedPreferenceSet| {
        let mut ids: Vec<String> = set.set.relations.iter().map(|r| r.id.clone()).collect();
        ids.sort();
        ids
    };
    assert_eq!(ids(a), ids(b));
    assert_eq!(a.binding, b.binding);
}

/// Detail id of the wrapped refusal.
fn inner_detail(error: &SetLearningError) -> &'static str {
    match error {
        SetLearningError::Model(e) => e.detail(),
        SetLearningError::Records(e) => e.detail(),
        SetLearningError::Lp(e) => e.detail(),
        other => other.detail(),
    }
}

fn refusal(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
) -> SetLearningError {
    learn_compatible_set(model, records, options).expect_err("refused")
}

#[test]
fn every_refusal_names_its_cause() {
    let model = one_parameter();
    let records = record_set(&model, vec![at_least_one("d")], Pooling::PerDecider);
    let with = |tolerance: f64| SetLearningOptions { tolerance, ..Default::default() };

    let negative = refusal(&model, &records, with(-0.5));
    assert_eq!(negative.detail(), reason_details::preference_set_learning::TOLERANCE_NEGATIVE);
    let infinite = refusal(&model, &records, with(f64::NAN));
    assert_eq!(infinite.detail(), reason_details::preference_set_learning::TOLERANCE_NON_FINITE);

    let mut blank = model.clone();
    blank.utility_units = " ".to_owned();
    let invalid = refusal(&blank, &records, with(0.0));
    assert_eq!(invalid.detail(), reason_details::preference_set_learning::MODEL_INVALID);
    assert_eq!(inner_detail(&invalid), reason_details::preference_model::UTILITY_UNITS_BLANK);

    let mut empty = records.clone();
    empty.records.clear();
    let missing = refusal(&model, &empty, with(0.0));
    assert_eq!(missing.detail(), reason_details::preference_set_learning::RECORDS_INVALID);
    assert_eq!(inner_detail(&missing), reason_details::choice_record::RECORDS_MISSING);

    // `x0 + t * t * x1` has no linear form in the free parameter t.
    let mut squared = model.clone();
    squared.value = E::sum(
        E::Attribute(0),
        E::product(E::product(E::Parameter(0), E::Parameter(0)), E::Attribute(1)),
    );
    let no_form = refusal(&squared, &records, with(0.0));
    assert_eq!(no_form.detail(), reason_details::preference_set_learning::RECORDS_INVALID);
    assert_eq!(inner_detail(&no_form), reason_details::choice_record::SHAPE_PARAMETERS_NOT_FIXED);

    let declared = |id: &str, bound: f64| LinearRelation {
        id: id.to_owned(),
        terms: vec![(0, 1.0)],
        sense: RelationSense::AtLeast,
        bound,
        origin: RelationOrigin::Declared,
    };
    let mut impossible = model.clone();
    impossible.preferences.relations.push(declared("above_range", 5.0));
    let empty_set = refusal(&impossible, &records, with(0.0));
    assert_eq!(empty_set.detail(), reason_details::preference_set_learning::DECLARED_SET_EMPTY);

    let mut clashing = model.clone();
    clashing.preferences.relations.push(declared("choice.r1.b", -4.0));
    let collision = refusal(&clashing, &records, with(0.0));
    assert_eq!(collision.detail(), reason_details::preference_set_learning::RELATION_ID_COLLISION);
    assert!(
        matches!(collision, SetLearningError::RelationIdCollision { ref id } if id == "choice.r1.b")
    );

    let mut bad_lp = with(0.0);
    bad_lp.lp.feasibility_tolerance = 0.0;
    let lp = refusal(&model, &records, bad_lp);
    assert_eq!(lp.detail(), reason_details::preference_set_learning::LINEAR_PROGRAM_REFUSED);
    assert_eq!(inner_detail(&lp), "linear_program.invalid_options");
}

fn mark_unavailable(record: &mut ChoiceRecord, option: &str, constraint: &str) {
    for o in &mut record.options {
        if o.id == option {
            o.belief = OptionBelief::Unavailable { constraint_id: constraint.to_owned() };
        }
    }
}

/// Options a = (0, 1), b = (1, 0), c = (-10, 0), a chosen: against b, `t >= 1`; against c,
/// `t >= -10`, which the declared range `[-4, 4]` already implies.
fn three_options() -> ChoiceRecord {
    record("r", "d", &[("a", &[0.0, 1.0]), ("b", &[1.0, 0.0]), ("c", &[-10.0, 0.0])], "a")
}

#[test]
fn an_unavailable_option_teaches_nothing() {
    let model = one_parameter();
    let learn_one = |r: ChoiceRecord| {
        let result = learn(&model, &record_set(&model, vec![r], Pooling::PerDecider), 0.0);
        learned(only(&result)).clone()
    };

    let listed = learn_one(three_options());
    range_is(&listed, 0, 1.0, 4.0);
    assert!(listed.ranges[0].narrowed);
    assert!(listed.binding.contains(&binding(&[("r", "b")])[0]));

    let mut excluded = three_options();
    mark_unavailable(&mut excluded, "b", "budget_cap");
    let excluded = learn_one(excluded);
    range_is(&excluded, 0, -4.0, 4.0);
    assert!(!excluded.ranges[0].narrowed);
    assert!(excluded.binding.iter().all(|c| c.rejected_option != "b"));
    assert!(excluded.set.relations.iter().all(|r| r.id.rsplit('.').next() != Some("b")));

    let mut removed = three_options();
    removed.options.remove(1);
    let removed = learn_one(removed);
    assert_eq!(removed.ranges, excluded.ranges);
    assert_eq!(removed.binding, excluded.binding);
    assert_eq!(removed.set.relations, excluded.set.relations);
}

#[test]
fn a_record_with_one_available_option_changes_nothing() {
    let model = one_parameter();
    let mut single = record("s", "d", &[("a", &[0.0, 1.0]), ("b", &[1.0, 0.0])], "a");
    mark_unavailable(&mut single, "b", "regulation");
    let alone = record_set(&model, vec![at_most_three("d")], Pooling::PerDecider);
    let both = record_set(&model, vec![at_most_three("d"), single.clone()], Pooling::PerDecider);
    let alone_result = learn(&model, &alone, 0.0);
    let both_result = learn(&model, &both, 0.0);
    let (a, b) = (learned(only(&alone_result)), learned(only(&both_result)));
    assert_eq!(a.ranges, b.ranges);
    assert_eq!(a.set.relations, b.set.relations);
    assert_eq!(b.n_records, 2);
    assert_eq!(both_result.per_decider[0].n_records, 2);

    let only_single = record_set(&model, vec![single], Pooling::PerDecider);
    let result = learn(&model, &only_single, 0.0);
    let set = learned(only(&result));
    assert_eq!(set.n_records, 1);
    assert!(!set.ranges[0].varied_in_records && !set.ranges[0].narrowed);
    assert!(set.set.relations.is_empty());
    let features = choice_features(&model, &both).expect("features");
    assert_eq!(uninformative_records(&features), ["s"]);
}

fn stated(mut record: ChoiceRecord) -> ChoiceRecord {
    record.belief_source = BeliefSource::Stated {
        elicitation_id: "survey_2024".to_owned(),
        assumption_id: "stated_reflects_acted".to_owned(),
    };
    record
}

#[test]
fn a_stated_comparison_narrows_like_the_equivalent_recorded_choice() {
    let model = one_parameter();
    let recorded = record_set(&model, vec![at_least_one("d")], Pooling::PerDecider);
    let said = record_set(&model, vec![stated(at_least_one("d"))], Pooling::PerDecider);
    let recorded_result = learn(&model, &recorded, 0.0);
    let said_result = learn(&model, &said, 0.0);
    let (r, s) = (learned(only(&recorded_result)), learned(only(&said_result)));
    assert_eq!(r.ranges, s.ranges);
    assert_eq!(r.set.relations, s.set.relations);
    assert_eq!(
        s.set.relations[0].origin,
        RelationOrigin::Choice { record_id: "r1".to_owned(), rejected_option: "b".to_owned() }
    );
    assert!(s.assumptions.contains(&"stated_reflects_acted".to_owned()));
    assert!(!r.assumptions.contains(&"stated_reflects_acted".to_owned()));

    let mixed = record_set(
        &model,
        vec![stated(at_least_one("d")), at_most_three("d")],
        Pooling::PerDecider,
    );
    let result = learn(&model, &mixed, 0.0);
    let set = learned(only(&result));
    range_is(set, 0, 1.0, 3.0);
    assert_eq!(set.n_records, 2);
}
