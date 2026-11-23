//! Recorded choices: validation refusals, the canonical digest, and expected linear-form terms
//! of every option from attribute means and from weighted joint draws.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_details;
use antecedent_design::choice_record::{
    BeliefSource, ChoiceFeatures, ChoiceOption, ChoiceRecord, ChoiceRecordError, ChoiceRecordSet,
    OptionBelief, Pooling, choice_features,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as E, PreferenceParameter, PreferenceSet, ValueModel,
};

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.to_owned(), units: units.to_owned() }
}

fn attributes() -> Vec<Attribute> {
    vec![attribute("benefit", "events"), attribute("cost", "usd")]
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

#[allow(clippy::unnecessary_box_returns, reason = "expression variants hold boxed operands")]
fn b(e: E) -> Box<E> {
    Box::new(e)
}

fn mul(a: E, c: E) -> E {
    E::Mul(b(a), b(c))
}

fn add(a: E, c: E) -> E {
    E::Add(b(a), b(c))
}

fn value_model(parameters: Vec<PreferenceParameter>, value: E) -> ValueModel {
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet { parameters, relations: Vec::new() },
        value,
        utility_units: "utility".to_owned(),
    }
}

/// `t0 * benefit + t1 * cost`.
fn rate_model() -> ValueModel {
    value_model(
        vec![parameter("per_benefit", 0.0, 10.0), parameter("per_cost", -1.0, 0.0)],
        add(mul(E::Parameter(0), E::Attribute(0)), mul(E::Parameter(1), E::Attribute(1))),
    )
}

/// `t0 * benefit + t1 * max(cost - 2, 0)`.
fn kinked_model() -> ValueModel {
    value_model(
        vec![parameter("per_benefit", 0.0, 10.0), parameter("per_excess_cost", -1.0, 0.0)],
        add(
            mul(E::Parameter(0), E::Attribute(0)),
            mul(
                E::Parameter(1),
                E::Max(b(E::Sub(b(E::Attribute(1)), b(E::Const(2.0)))), b(E::Const(0.0))),
            ),
        ),
    )
}

/// `t0 * benefit + t1 * t1 * cost`, with `t1` a shape parameter of range `[lower, upper]`.
fn squared_scale_model(lower: f64, upper: f64) -> ValueModel {
    value_model(
        vec![parameter("per_benefit", 0.0, 10.0), parameter("cost_scale", lower, upper)],
        add(
            mul(E::Parameter(0), E::Attribute(0)),
            mul(mul(E::Parameter(1), E::Parameter(1)), E::Attribute(1)),
        ),
    )
}

fn means(values: &[f64]) -> OptionBelief {
    OptionBelief::Means(values.to_vec())
}

fn option(id: &str, belief: OptionBelief) -> ChoiceOption {
    ChoiceOption { id: id.to_owned(), belief }
}

fn record(id: &str, decider: &str, rows: &[&str]) -> ChoiceRecord {
    ChoiceRecord {
        id: id.to_owned(),
        decider: decider.to_owned(),
        options: vec![option("treat", means(&[3.0, 5.0])), option("wait", means(&[1.0, 0.5]))],
        chosen: "treat".to_owned(),
        belief_source: BeliefSource::RecordedForecast { source_id: "forecast_2019".to_owned() },
        observation_rows: rows.iter().map(|r| (*r).to_owned()).collect(),
    }
}

fn set() -> ChoiceRecordSet {
    let mut second = record("r2", "clinic_b", &["visits#2"]);
    second.belief_source = BeliefSource::AnalysisAsOf {
        result_identity: "analysis_a".to_owned(),
        as_of: "2020-01-01".to_owned(),
    };
    let mut third = record("r3", "clinic_a", &["visits#3", "visits#4"]);
    third.belief_source =
        BeliefSource::RealizedOutcome { assumption_id: "realized_as_expected".to_owned() };
    third.options[1].belief =
        OptionBelief::Draws { rows: vec![0.0, 1.0, 2.0, 0.0], weights: Some(vec![1.0, 3.0]) };
    ChoiceRecordSet {
        attributes: attributes(),
        population: "adults in region".to_owned(),
        records: vec![record("r1", "clinic_a", &["visits#1"]), second, third],
        pooling: Pooling::Pooled { assumption_id: "shared_preferences".to_owned() },
    }
}

fn refused(edit: impl FnOnce(&mut ChoiceRecordSet)) -> ChoiceRecordError {
    let mut s = set();
    edit(&mut s);
    s.validate().expect_err("refused")
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "got {got}, want {want}");
}

fn assert_all_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        close(*g, *w);
    }
}

#[test]
fn a_well_formed_set_validates_and_lists_deciders_and_assumptions() {
    let s = set();
    s.validate().expect("valid");
    assert_eq!(s.deciders(), ["clinic_a", "clinic_b"]);
    assert_eq!(s.assumptions(), ["realized_as_expected", "shared_preferences"]);
}

#[test]
fn every_set_level_refusal_names_its_cause() {
    let cases: Vec<(ChoiceRecordError, &str)> = vec![
        (refused(|s| s.attributes.clear()), reason_details::choice_record::ATTRIBUTES_MISSING),
        (
            refused(|s| s.attributes[1].id = " ".to_owned()),
            reason_details::choice_record::ATTRIBUTE_ID_BLANK,
        ),
        (
            refused(|s| s.attributes[1].units = String::new()),
            reason_details::choice_record::ATTRIBUTE_UNITS_BLANK,
        ),
        (
            refused(|s| s.attributes[1].id = "benefit".to_owned()),
            reason_details::choice_record::ATTRIBUTE_DUPLICATE,
        ),
        (
            refused(|s| s.population = "  ".to_owned()),
            reason_details::choice_record::POPULATION_BLANK,
        ),
        (
            refused(|s| s.pooling = Pooling::Pooled { assumption_id: String::new() }),
            reason_details::choice_record::POOLING_ASSUMPTION_BLANK,
        ),
        (refused(|s| s.records.clear()), reason_details::choice_record::RECORDS_MISSING),
        (
            refused(|s| s.records[1].id = String::new()),
            reason_details::choice_record::RECORD_ID_BLANK,
        ),
        (
            refused(|s| s.records[1].id = "r1".to_owned()),
            reason_details::choice_record::RECORD_DUPLICATE,
        ),
    ];
    for (error, detail) in cases {
        assert_eq!(error.detail(), detail, "{error:?}");
    }
    assert_eq!(
        refused(|s| s.records[1].id = "r1".to_owned()),
        ChoiceRecordError::RecordDuplicate { record: "r1".to_owned() }
    );
    assert_eq!(
        refused(|s| s.attributes[1].id = String::new()),
        ChoiceRecordError::AttributeIdBlank { index: 1 }
    );
}

#[test]
fn every_record_level_refusal_names_record_and_option() {
    let r = || "r2".to_owned();
    assert_eq!(
        refused(|s| s.records[1].decider = String::new()),
        ChoiceRecordError::DeciderBlank { record: r() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].options.pop();
        }),
        ChoiceRecordError::OptionsTooFew { record: r(), options: 1 }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].id = " ".to_owned()),
        ChoiceRecordError::OptionIdBlank { record: r(), index: 1 }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].id = "treat".to_owned()),
        ChoiceRecordError::OptionDuplicate { record: r(), option: "treat".to_owned() }
    );
    assert_eq!(
        refused(|s| s.records[1].chosen = "operate".to_owned()),
        ChoiceRecordError::ChosenUnknown { record: r(), chosen: "operate".to_owned() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].belief_source =
                BeliefSource::RecordedForecast { source_id: String::new() };
        }),
        ChoiceRecordError::ForecastSourceBlank { record: r() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].belief_source = BeliefSource::AnalysisAsOf {
                result_identity: " ".to_owned(),
                as_of: "2020-01-01".to_owned(),
            };
        }),
        ChoiceRecordError::AnalysisIdentityBlank { record: r() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].belief_source = BeliefSource::AnalysisAsOf {
                result_identity: "analysis_a".to_owned(),
                as_of: String::new(),
            };
        }),
        ChoiceRecordError::AnalysisDateBlank { record: r() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].belief_source =
                BeliefSource::RealizedOutcome { assumption_id: "\t".to_owned() };
        }),
        ChoiceRecordError::RealizedAssumptionBlank { record: r() }
    );
}

#[test]
fn every_belief_and_row_refusal_names_record_and_option() {
    let r = || "r2".to_owned();
    let o = || "wait".to_owned();
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = means(&[1.0])),
        ChoiceRecordError::MeansWidthMismatch { record: r(), option: o(), attributes: 2, means: 1 }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = means(&[1.0, f64::NAN])),
        ChoiceRecordError::BeliefNotFinite { record: r(), option: o() }
    );
    let draws = |rows: Vec<f64>, weights: Option<Vec<f64>>| OptionBelief::Draws { rows, weights };
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = draws(Vec::new(), None)),
        ChoiceRecordError::DrawsEmpty { record: r(), option: o() }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = draws(vec![1.0, 2.0, 3.0], None)),
        ChoiceRecordError::DrawsWidthMismatch {
            record: r(),
            option: o(),
            attributes: 2,
            values: 3
        }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = draws(vec![1.0, f64::INFINITY], None)),
        ChoiceRecordError::BeliefNotFinite { record: r(), option: o() }
    );
    assert_eq!(
        refused(|s| s.records[1].options[1].belief = draws(vec![1.0, 2.0], Some(vec![1.0, 1.0]))),
        ChoiceRecordError::WeightsLengthMismatch { record: r(), option: o(), rows: 1, weights: 2 }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].options[1].belief =
                draws(vec![1.0, 2.0, 3.0, 4.0], Some(vec![1.0, f64::NAN]));
        }),
        ChoiceRecordError::WeightNotFinite { record: r(), option: o(), row: 1 }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].options[1].belief = draws(vec![1.0, 2.0, 3.0, 4.0], Some(vec![2.0, -1.0]));
        }),
        ChoiceRecordError::WeightNegative { record: r(), option: o(), row: 1 }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].options[1].belief = draws(vec![1.0, 2.0, 3.0, 4.0], Some(vec![0.0, 0.0]));
        }),
        ChoiceRecordError::WeightsSumInvalid { record: r(), option: o() }
    );
    assert_eq!(
        refused(|s| {
            s.records[1].options[1].belief =
                draws(vec![1.0, 2.0, 3.0, 4.0], Some(vec![f64::MAX, f64::MAX]));
        }),
        ChoiceRecordError::WeightsSumInvalid { record: r(), option: o() }
    );
    assert_eq!(
        refused(|s| s.records[1].observation_rows.clear()),
        ChoiceRecordError::ObservationRowsMissing { record: r() }
    );
    assert_eq!(
        refused(|s| s.records[1].observation_rows.push("  ".to_owned())),
        ChoiceRecordError::ObservationRowBlank { record: r(), index: 1 }
    );
    assert_eq!(
        refused(|s| s.records[1].observation_rows.push(" visits#2".to_owned())),
        ChoiceRecordError::ObservationRowRepeated { record: r(), row: "visits#2".to_owned() }
    );
}

#[test]
fn a_row_under_two_records_is_double_counting() {
    let error = refused(|s| s.records[2].observation_rows.push("visits#1 ".to_owned()));
    assert_eq!(
        error,
        ChoiceRecordError::ObservationRowDoubleCounted {
            row: "visits#1".to_owned(),
            first_record: "r1".to_owned(),
            second_record: "r3".to_owned(),
        }
    );
    assert_eq!(error.detail(), reason_details::choice_record::OBSERVATION_ROW_DOUBLE_COUNTED);
    // Distinct spellings are distinct rows.
    let mut s = set();
    s.records[2].observation_rows.push("visits#01".to_owned());
    s.validate().expect("distinct rows");
}

#[test]
#[allow(clippy::too_many_lines, reason = "one edit per encoded field, listed in one table")]
fn digest_ignores_record_order_and_changes_with_every_field() {
    let base = set().digest().expect("digest");
    let mut reordered = set();
    reordered.records.reverse();
    assert_eq!(reordered.digest().expect("digest"), base);
    let mut respaced = set();
    respaced.records[2].observation_rows = vec![" visits#4".to_owned(), "visits#3 ".to_owned()];
    assert_eq!(respaced.digest().expect("digest"), base, "row spelling is trimmed and sorted");

    let edits: Vec<(&str, Box<dyn Fn(&mut ChoiceRecordSet)>)> = vec![
        (
            "attribute id",
            Box::new(|s: &mut ChoiceRecordSet| s.attributes[0].id = "gain".to_owned()),
        ),
        (
            "attribute units",
            Box::new(|s: &mut ChoiceRecordSet| s.attributes[1].units = "eur".to_owned()),
        ),
        ("attribute order", Box::new(|s: &mut ChoiceRecordSet| s.attributes.reverse())),
        ("population", Box::new(|s: &mut ChoiceRecordSet| s.population = "children".to_owned())),
        ("pooling", Box::new(|s: &mut ChoiceRecordSet| s.pooling = Pooling::PerDecider)),
        (
            "pooling assumption",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.pooling = Pooling::Pooled { assumption_id: "other".to_owned() }
            }),
        ),
        ("record id", Box::new(|s: &mut ChoiceRecordSet| s.records[0].id = "r0".to_owned())),
        (
            "decider",
            Box::new(|s: &mut ChoiceRecordSet| s.records[0].decider = "clinic_c".to_owned()),
        ),
        (
            "option id",
            Box::new(|s: &mut ChoiceRecordSet| s.records[0].options[1].id = "defer".to_owned()),
        ),
        ("option order", Box::new(|s: &mut ChoiceRecordSet| s.records[0].options.reverse())),
        (
            "mean value",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[0].options[1].belief = means(&[1.0, 0.25])
            }),
        ),
        (
            "draw value",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[2].options[1].belief = OptionBelief::Draws {
                    rows: vec![0.0, 1.0, 2.0, 0.5],
                    weights: Some(vec![1.0, 3.0]),
                };
            }),
        ),
        (
            "weight value",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[2].options[1].belief = OptionBelief::Draws {
                    rows: vec![0.0, 1.0, 2.0, 0.0],
                    weights: Some(vec![1.0, 2.0]),
                };
            }),
        ),
        (
            "weights absent",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[2].options[1].belief =
                    OptionBelief::Draws { rows: vec![0.0, 1.0, 2.0, 0.0], weights: None };
            }),
        ),
        ("chosen", Box::new(|s: &mut ChoiceRecordSet| s.records[0].chosen = "wait".to_owned())),
        (
            "belief source kind",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[0].belief_source =
                    BeliefSource::RealizedOutcome { assumption_id: "forecast_2019".to_owned() };
            }),
        ),
        (
            "belief source id",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[0].belief_source =
                    BeliefSource::RecordedForecast { source_id: "forecast_2020".to_owned() };
            }),
        ),
        (
            "as-of date",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[1].belief_source = BeliefSource::AnalysisAsOf {
                    result_identity: "analysis_a".to_owned(),
                    as_of: "2021-01-01".to_owned(),
                };
            }),
        ),
        (
            "observation row",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[0].observation_rows[0] = "visits#9".to_owned()
            }),
        ),
        (
            "extra row",
            Box::new(|s: &mut ChoiceRecordSet| {
                s.records[0].observation_rows.push("visits#9".to_owned())
            }),
        ),
    ];
    let mut seen = std::collections::BTreeSet::from([base]);
    for (what, edit) in edits {
        let mut s = set();
        edit(&mut s);
        let digest = s.digest().expect(what);
        assert!(seen.insert(digest), "digest did not change with {what}");
    }
}

#[test]
fn digest_of_an_invalid_set_is_refused() {
    let mut s = set();
    s.population = String::new();
    assert_eq!(s.digest(), Err(ChoiceRecordError::PopulationBlank));
}

fn features(model: &ValueModel, s: &ChoiceRecordSet) -> ChoiceFeatures {
    choice_features(model, s).expect("features")
}

#[test]
fn rate_model_features_from_means_are_the_means() {
    let mut s = set();
    s.records.truncate(2);
    s.records[1].options.push(option("refer", means(&[-2.0, 0.25])));
    s.records[1].chosen = "refer".to_owned();
    let f = features(&rate_model(), &s);
    assert_eq!(f.n_parameters, 2);
    assert_eq!(f.sets.len(), 2);
    assert_eq!(f.sets[0].record_id, "r1");
    assert_eq!(f.sets[0].decider, "clinic_a");
    assert_eq!(f.sets[0].option_ids, ["treat", "wait"]);
    assert_eq!(f.sets[0].chosen, 0);
    assert_all_close(&f.sets[0].fixed, &[0.0, 0.0]);
    assert_all_close(&f.sets[0].terms, &[3.0, 5.0, 1.0, 0.5]);
    assert_eq!(f.sets[1].chosen, 2);
    assert_all_close(&f.sets[1].terms, &[3.0, 5.0, 1.0, 0.5, -2.0, 0.25]);
}

#[test]
fn weighted_draws_give_the_weighted_mean() {
    // Record r3's second option: rows (0, 1) and (2, 0) with weights 1 and 3.
    let f = features(&rate_model(), &set());
    let third = &f.sets[2];
    assert_all_close(&third.terms[2..], &[1.5, 0.25]);
    // Unnormalised weights act like their normalised values; absent weights are uniform.
    let mut s = set();
    s.records[2].options[1].belief =
        OptionBelief::Draws { rows: vec![0.0, 1.0, 2.0, 0.0], weights: Some(vec![0.25, 0.75]) };
    assert_eq!(features(&rate_model(), &s).sets[2].terms, third.terms);
    s.records[2].options[1].belief =
        OptionBelief::Draws { rows: vec![0.0, 1.0, 2.0, 0.0], weights: None };
    assert_all_close(&features(&rate_model(), &s).sets[2].terms[2..], &[1.0, 0.5]);
    // A zero-weight row does not contribute.
    s.records[2].options[1].belief = OptionBelief::Draws {
        rows: vec![0.0, 1.0, 2.0, 0.0, 100.0, 100.0],
        weights: Some(vec![1.0, 3.0, 0.0]),
    };
    assert_all_close(&features(&rate_model(), &s).sets[2].terms[2..], &[1.5, 0.25]);
}

#[test]
fn kinked_model_is_refused_from_means_and_computed_from_draws() {
    let mut s = set();
    s.records.truncate(1);
    let error = choice_features(&kinked_model(), &s).expect_err("means cannot carry a kink");
    assert_eq!(
        error,
        ChoiceRecordError::BeliefLawRequired {
            record: "r1".to_owned(),
            option: "treat".to_owned(),
            term: Some(1),
        }
    );
    assert_eq!(error.detail(), reason_details::choice_record::BELIEF_LAW_REQUIRED);

    s.records[0].options[0].belief = OptionBelief::Draws {
        rows: vec![1.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        weights: Some(vec![1.0, 1.0, 2.0]),
    };
    s.records[0].options[1].belief =
        OptionBelief::Draws { rows: vec![0.0, 2.5, 1.0, 1.5], weights: None };
    let f = features(&kinked_model(), &s);
    // treat: benefit (1 + 2 + 8) / 4; excess (0 + 1 + 8) / 4. wait: benefit 1/2; excess 1/4.
    assert_all_close(&f.sets[0].terms, &[2.75, 2.25, 0.5, 0.25]);
    assert_all_close(&f.sets[0].fixed, &[0.0, 0.0]);
}

#[test]
fn a_nonaffine_fixed_part_needs_draws() {
    let model = value_model(
        vec![parameter("per_benefit", 0.0, 10.0)],
        add(mul(E::Parameter(0), E::Attribute(0)), E::Min(b(E::Attribute(1)), b(E::Const(1.0)))),
    );
    let mut s = set();
    s.records.truncate(1);
    assert_eq!(
        choice_features(&model, &s),
        Err(ChoiceRecordError::BeliefLawRequired {
            record: "r1".to_owned(),
            option: "treat".to_owned(),
            term: None,
        })
    );
}

#[test]
fn a_fixed_shape_parameter_is_folded_into_the_fixed_part() {
    // cost_scale fixed at 2: value = t0 * benefit + 4 * cost.
    let mut s = set();
    s.records.truncate(1);
    let f = features(&squared_scale_model(2.0, 2.0), &s);
    assert_eq!(f.n_parameters, 2);
    assert_all_close(&f.sets[0].fixed, &[20.0, 2.0]);
    assert_all_close(&f.sets[0].terms, &[3.0, 0.0, 1.0, 0.0]);
}

#[test]
fn a_fixed_linear_parameter_is_folded_into_the_fixed_part() {
    let mut model = rate_model();
    model.preferences.parameters[1] = parameter("per_cost", -0.5, -0.5);
    let mut s = set();
    s.records.truncate(1);
    let f = features(&model, &s);
    assert_all_close(&f.sets[0].fixed, &[-2.5, -0.25]);
    assert_all_close(&f.sets[0].terms, &[3.0, 0.0, 1.0, 0.0]);
}

#[test]
fn a_free_shape_parameter_is_refused_by_name() {
    let error = choice_features(&squared_scale_model(1.0, 2.0), &set()).expect_err("free shape");
    assert_eq!(
        error,
        ChoiceRecordError::ShapeParametersNotFixed { parameters: vec!["cost_scale".to_owned()] }
    );
    assert_eq!(error.detail(), reason_details::choice_record::SHAPE_PARAMETERS_NOT_FIXED);
}

#[test]
fn attributes_that_differ_from_the_model_are_refused() {
    let mut s = set();
    s.attributes[1] = attribute("cost", "eur");
    let error = choice_features(&rate_model(), &s).expect_err("units differ");
    assert_eq!(
        error,
        ChoiceRecordError::AttributesDiffer {
            index: 1,
            model: Some(attribute("cost", "usd")),
            records: Some(attribute("cost", "eur")),
        }
    );
    assert_eq!(error.detail(), reason_details::choice_record::ATTRIBUTES_DIFFER);

    let mut s = set();
    s.attributes.push(attribute("harm", "events"));
    for record in &mut s.records {
        for option in &mut record.options {
            option.belief = means(&[1.0, 2.0, 3.0]);
        }
    }
    assert_eq!(
        choice_features(&rate_model(), &s),
        Err(ChoiceRecordError::AttributesDiffer {
            index: 2,
            model: None,
            records: Some(attribute("harm", "events")),
        })
    );
}

#[test]
fn an_invalid_model_or_record_set_is_refused_before_features() {
    let mut model = rate_model();
    model.preferences.parameters[0] = parameter("per_benefit", 2.0, 1.0);
    let error = choice_features(&model, &set()).expect_err("empty range");
    assert!(matches!(error, ChoiceRecordError::Model(_)));
    assert_eq!(error.detail(), reason_details::choice_record::MODEL_INVALID);

    let mut s = set();
    s.records.clear();
    assert_eq!(choice_features(&rate_model(), &s), Err(ChoiceRecordError::RecordsMissing));
}

#[test]
fn an_overflowing_expectation_is_refused() {
    let mut s = set();
    s.records[0].options[1].belief =
        OptionBelief::Draws { rows: vec![f64::MAX, 0.0, f64::MAX, 0.0], weights: None };
    let error = choice_features(&rate_model(), &s).expect_err("overflow");
    assert_eq!(
        error,
        ChoiceRecordError::TermNotFinite { record: "r1".to_owned(), option: "wait".to_owned() }
    );
    assert_eq!(error.detail(), reason_details::choice_record::TERM_NOT_FINITE);
}

#[test]
fn features_follow_record_order_and_are_deterministic() {
    let s = set();
    let first = features(&rate_model(), &s);
    assert_eq!(first, features(&rate_model(), &s));
    let mut reversed = set();
    reversed.records.reverse();
    let second = features(&rate_model(), &reversed);
    let ids: Vec<&str> = second.sets.iter().map(|x| x.record_id.as_str()).collect();
    assert_eq!(ids, ["r3", "r2", "r1"]);
    assert_eq!(second.sets[0], first.sets[2]);
}
