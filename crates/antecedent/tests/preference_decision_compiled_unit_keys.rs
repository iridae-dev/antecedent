//! Shared evidence between separately estimated outcomes, judged by where the rows came from
//! and by the units they observe.
//!
//! The value of an action is `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`
//! (and, across three studies, minus a cost), so the decision needs one aligned law of the
//! outcomes and the evidence ledger must judge, at compile, which studies share observations.
//!
//! * Studies without declarations keep the table-content judgement and its encoding exactly.
//! * A column projection remembers the table it was taken from and shares that table's rows with
//!   every other study of it, so different columns of the same rows are not distinct evidence;
//!   row selections share exactly the rows they both took. A declared provenance replaces the
//!   remembered one.
//! * Studies of different tables keyed in one identifier space are judged by the units they
//!   share, each unit counted once and masked rows excluded; keys in different spaces, or a key
//!   on one study only, leave the overlap unknown and are refused.
//! * Each overlapping pair gets its own chain factor, so across three studies only the pairs
//!   that share units need a declared block.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, CrossStudyCovariance,
    DecisionCriterion, OutcomeDependence, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceRouteError, PreferenceSet, RowProvenance, SharedEvidence,
    StructuralPolicy, UnitKey, ValueModel, compile_preference_decision_across_studies,
    compile_preference_decision_with_shared_evidence, shape,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::reason_details::{decision_evidence, preference_compile};
use antecedent_core::{
    CanonicalWriter, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional,
    ResponseQuery, SemanticDigest, VariableId,
};
use antecedent_data::{TabularData, ValidityBitmap};
use antecedent_graph::{Dag, DenseNodeId};

const NAMESPACE: &str = "hospital_mrn";

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

fn parameters() -> PreferenceSet {
    PreferenceSet::new(
        vec![
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
        vec![],
    )
}

/// `benefit - harm_rate * harm - excess_weight * max(harm - 1.2, 0)`.
fn kinked() -> PreferenceExpr {
    PreferenceExpr::difference(
        PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, 1.2)),
    )
}

fn parts(actions: Vec<PreferenceAction>) -> PreferenceContractParts {
    PreferenceContractParts {
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "patients".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

fn contract() -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: parameters(),
        value: kinked(),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", dose)),
            ActionInput::Quantity(quantity("h", "events", dose)),
        ],
    };
    AcceptedPreferences::declared(model)
        .unwrap()
        .contract(parts(vec![action("high", "1.5"), action("low", "0.5")]))
}

/// The kinked value minus a cost in points, read from a third study.
fn costed_contract() -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
            Attribute { id: "cost".into(), units: "points".into() },
        ],
        preferences: parameters(),
        value: PreferenceExpr::difference(kinked(), PreferenceExpr::Attribute(2)),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", dose)),
            ActionInput::Quantity(quantity("h", "events", dose)),
            ActionInput::Quantity(quantity("c", "points", dose)),
        ],
    };
    AcceptedPreferences::declared(model)
        .unwrap()
        .contract(parts(vec![action("high", "1.5"), action("low", "0.5")]))
}

fn dose_table(outcome: &str, rows: u32, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

fn benefit_table() -> TabularData {
    dose_table("b", 200, (2.0, 1.2), 1.7)
}

fn harm_table() -> TabularData {
    dose_table("h", 200, (0.5, 0.5), 2.9)
}

fn cost_table() -> TabularData {
    dose_table("c", 200, (0.3, 0.2), 0.7)
}

/// One table holding `t`, `b` and `h` over the same 200 rows.
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

/// A Bayesian dose-response study of `outcome` over `variables` columns, before build.
fn dose_builder(data: &TabularData, variables: u32, outcome: u32) -> StudyBuilder {
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
}

fn study(data: &TabularData) -> Study {
    dose_builder(data, 2, 1).build().unwrap()
}

/// A study of a two-column table whose rows observe the units `labels`.
fn keyed(data: &TabularData, namespace: &str, labels: Vec<String>) -> Study {
    dose_builder(data, 2, 1).unit_key(UnitKey::canonical_labels(namespace, labels)).build().unwrap()
}

fn patients(range: impl Iterator<Item = u32>) -> Vec<String> {
    range.map(|i| format!("text:p{i}")).collect()
}

fn distinct() -> OutcomeDependence {
    OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 }
}

fn shared(cross: Vec<CrossStudyCovariance>) -> SharedEvidence {
    SharedEvidence { draws: 2000, seed: 7, cross }
}

fn block(first: usize, second: usize) -> CrossStudyCovariance {
    CrossStudyCovariance { first, second, covariance: vec![1e-5; 4] }
}

fn refusal_of(error: &PreferenceRouteError) -> (&str, &str, Option<&str>) {
    let refusal = error.refusal().expect("a route refusal");
    (refusal.code, refusal.detail.as_str(), refusal.supplied.as_deref())
}

fn digest(table: &TabularData) -> String {
    SemanticDigest::from_bytes(table.storage().content_digest()).to_hex()
}

#[test]
fn studies_without_declarations_keep_the_table_content_judgement_and_encoding() {
    let ctx = ExecutionContext::for_tests(3);
    let (benefit, harm) = (benefit_table(), harm_table());
    let compiled = compile_preference_decision_across_studies(
        &[&study(&benefit), &study(&harm)],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().expect("a declared joint law").fact();
    let factors = [digest(&benefit), digest(&harm)];
    let mut writer = CanonicalWriter::new("antecedent.preference_route.joint_law_declaration.v1");
    writer.text(compiled.bound_identity()).text("distinct_evidence").word(2000).word(7).word(4);
    for (index, (variable, units)) in [("b", "points"), ("h", "events")].into_iter().enumerate() {
        for (column, dose) in ["0.5", "1.5"].into_iter().enumerate() {
            let input = &fact.inputs()[index * 2 + column];
            assert_eq!(input.basis(), "table_content");
            assert_eq!(input.namespace(), None);
            assert_eq!(input.factor(), factors[index]);
            assert_eq!(input.factors().len(), 1);
            writer
                .text(&format!("study_{index}_column_{column}_source"))
                .text(&format!("study_{index}_column_{column}"))
                .word(index as u64)
                .text(&quantity(variable, units, dose).canonical_identity())
                .text(&factors[index]);
        }
    }
    writer.word(0); // no declared cross blocks
    assert_eq!(fact.declaration(), writer.finish_hex());

    // Two outcomes of one table still share every row, with or without keys.
    let table = shared_table();
    let first = dose_builder(&table, 3, 1).build().unwrap();
    let second = dose_builder(&table, 3, 2).build().unwrap();
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        refusal_of(&refused),
        (
            "evidence_double_counted",
            preference_compile::STUDIES_SHARE_EVIDENCE,
            Some("200 shared observations")
        )
    );
    let first = dose_builder(&table, 3, 1)
        .unit_key(UnitKey::canonical_labels(NAMESPACE, patients(0..200)))
        .build()
        .unwrap();
    let second = dose_builder(&table, 3, 2)
        .unit_key(UnitKey::canonical_labels(NAMESPACE, patients(500..700)))
        .build()
        .unwrap();
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).2, Some("200 shared observations"));
}

#[test]
fn projections_of_one_table_share_its_rows() {
    let ctx = ExecutionContext::for_tests(5);
    let source = shared_table();
    let (t, b, h) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let (benefit, _) = source.project(&[t, b]).unwrap();
    let (harm, _) = source.project(&[t, h]).unwrap();
    let same_rows = |first: &Study, second: &Study, shared: &str| {
        let refused = compile_preference_decision_across_studies(
            &[first, second],
            &contract(),
            distinct(),
            &ctx,
        )
        .unwrap_err();
        assert_eq!(
            refusal_of(&refused),
            ("evidence_double_counted", preference_compile::STUDIES_SHARE_EVIDENCE, Some(shared))
        );
    };
    let apart = |first: &Study, second: &Study| {
        let compiled = compile_preference_decision_across_studies(
            &[first, second],
            &contract(),
            distinct(),
            &ctx,
        );
        assert!(compiled.is_ok(), "{compiled:?}");
    };

    // Without any declaration, the projections remember the source and share its 200 rows,
    // with each other and with a study of the source itself.
    let (first, second) = (study(&benefit), study(&harm));
    same_rows(&first, &second, "200 shared observations");
    same_rows(&dose_builder(&source, 3, 1).build().unwrap(), &second, "200 shared observations");
    let compiled = compile_preference_decision_with_shared_evidence(
        &[&first, &second],
        &contract(),
        &shared(vec![block(0, 1)]),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    assert_eq!(fact.inputs()[0].factor(), digest(&source));
    assert_eq!(fact.inputs()[2].factor(), digest(&source));

    // Declaring the source explicitly is the same judgement.
    let projected = |table: &TabularData, provenance: RowProvenance| {
        dose_builder(table, 2, 1).row_provenance(provenance).build().unwrap()
    };
    let declared = projected(&harm, RowProvenance::of_table(&source));
    same_rows(&first, &declared, "200 shared observations");

    // Row selections share exactly the rows both took; disjoint selections are distinct.
    let rows_of = |rows: Vec<usize>, outcome: VariableId| {
        let (table, _) = source.select_rows(&rows).unwrap().project(&[t, outcome]).unwrap();
        study(&table)
    };
    let even = rows_of((0..200).step_by(2).collect(), b);
    let odd = rows_of((1..200).step_by(2).collect(), h);
    apart(&even, &odd);
    let early = rows_of((0..150).collect(), b);
    let late = rows_of((50..200).collect(), h);
    same_rows(&early, &late, "100 shared observations");

    // A declared provenance replaces the remembered one: rows taken from other positions of
    // the source are other observations.
    let elsewhere = RowProvenance::new(&digest(&source), Some((200..400).collect()));
    apart(&first, &projected(&harm, elsewhere));

    // A row index per row, of a named table.
    let short = RowProvenance::new(&digest(&source), Some(vec![0, 1, 2]));
    assert!(dose_builder(&harm, 2, 1).row_provenance(short).build().is_err());
    let blank = RowProvenance::new("  ", None);
    assert!(dose_builder(&harm, 2, 1).row_provenance(blank).build().is_err());
}

#[test]
fn keyed_studies_of_different_tables_are_judged_by_the_units_they_share() {
    let ctx = ExecutionContext::for_tests(7);
    let (benefit, harm) = (benefit_table(), harm_table());

    // 100 of the patients are in both tables.
    let first = keyed(&benefit, NAMESPACE, patients(0..200));
    let second = keyed(&harm, NAMESPACE, patients(100..300));
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        refusal_of(&refused),
        (
            "evidence_double_counted",
            preference_compile::STUDIES_SHARE_EVIDENCE,
            Some("100 shared units in namespace hospital_mrn")
        )
    );

    // A patient with several rows is one unit.
    let repeated =
        keyed(&benefit, NAMESPACE, (0..200).map(|i| format!("text:p{}", i / 2)).collect());
    let refused = compile_preference_decision_across_studies(
        &[&repeated, &keyed(&harm, NAMESPACE, patients(0..200))],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).2, Some("100 shared units in namespace hospital_mrn"));

    // With the pair's block declared, the chain names the shared units on both studies.
    let compiled = compile_preference_decision_with_shared_evidence(
        &[&first, &second],
        &contract(),
        &shared(vec![block(0, 1)]),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    let pair = fact.inputs()[0].factors().to_vec();
    assert_eq!(pair.len(), 1);
    assert!(pair[0].starts_with("shared_units:"), "{pair:?}");
    for input in fact.inputs() {
        assert_eq!(input.factors(), pair.as_slice());
        assert_eq!(input.basis(), "unit_key");
        assert_eq!(input.namespace(), Some(NAMESPACE));
    }
    let executed = compiled.execute(None, PreferenceEvaluationOptions::default(), &ctx).unwrap();
    assert!(executed.receipt.programs.contains(&fact.program()));

    // Patients seen by one table only: distinct evidence, each study naming its own units.
    let disjoint = keyed(&harm, NAMESPACE, patients(500..700));
    let compiled = compile_preference_decision_across_studies(
        &[&first, &disjoint],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    assert!(fact.inputs()[0].factor().starts_with("units:"));
    assert_ne!(fact.inputs()[0].factor(), fact.inputs()[2].factor());
    // A keyed declaration is another declaration than the same tables judged by content.
    let unkeyed = compile_preference_decision_across_studies(
        &[&study(&benefit), &study(&harm)],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    assert_ne!(unkeyed.joint_law().unwrap().fact().declaration(), fact.declaration());
}

#[test]
fn masked_rows_are_not_units_the_study_observes() {
    let ctx = ExecutionContext::for_tests(11);
    let keep: Vec<bool> = (0..200).map(|row| row < 100).collect();
    let masked = benefit_table().with_analysis_mask(ValidityBitmap::from_bools(&keep)).unwrap();
    let second = keyed(&harm_table(), NAMESPACE, patients(100..300));
    let compiled = compile_preference_decision_across_studies(
        &[&keyed(&masked, NAMESPACE, patients(0..200)), &second],
        &contract(),
        distinct(),
        &ctx,
    );
    assert!(compiled.is_ok(), "the masked patients are the only shared ones");
    let refused = compile_preference_decision_across_studies(
        &[&keyed(&benefit_table(), NAMESPACE, patients(0..200)), &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).0, "evidence_double_counted");
}

#[test]
fn keys_that_cannot_be_compared_leave_the_overlap_unknown() {
    let ctx = ExecutionContext::for_tests(13);
    let first = keyed(&benefit_table(), NAMESPACE, patients(0..200));
    for second in [keyed(&harm_table(), "insurer_id", patients(0..200)), study(&harm_table())] {
        let refused = compile_preference_decision_across_studies(
            &[&first, &second],
            &contract(),
            distinct(),
            &ctx,
        )
        .unwrap_err();
        let (code, detail, _) = refusal_of(&refused);
        assert_eq!(
            (code, detail),
            ("evidence_overlap_unknown", decision_evidence::DEPENDENCE_UNKNOWN)
        );
    }
}

/// A dose table of `outcome` with an `id` column holding each row's patient number.
fn identified_table(
    outcome: &str,
    (intercept, slope): (f64, f64),
    rate: f64,
    ids: impl Iterator<Item = u32>,
) -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    let ids: Vec<f64> = ids.map(f64::from).collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        (outcome, y.as_slice()),
        ("id", ids.as_slice()),
    ])
    .unwrap()
}

#[test]
fn a_key_column_identifies_units_and_a_malformed_key_is_refused() {
    let ctx = ExecutionContext::for_tests(17);
    let benefit = identified_table("b", (2.0, 1.2), 1.7, 0..200);
    let harm = identified_table("h", (0.5, 0.5), 2.9, 150..350);
    let key =
        |variable: u32, namespace: &str| UnitKey::column(VariableId::from_raw(variable), namespace);
    let first = dose_builder(&benefit, 3, 1).unit_key(key(2, NAMESPACE)).build().unwrap();
    let second = dose_builder(&harm, 3, 1).unit_key(key(2, NAMESPACE)).build().unwrap();
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).2, Some("50 shared units in namespace hospital_mrn"));

    // The key is not the outcome; the namespace is not blank; the column exists; one label
    // per row.
    let outcome_key = dose_builder(&benefit, 3, 1).unit_key(key(1, NAMESPACE)).build().unwrap();
    let refused = compile_preference_decision_across_studies(
        &[&outcome_key, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).1, preference_compile::UNIT_KEY_INVALID);
    assert!(dose_builder(&benefit, 3, 1).unit_key(key(2, " ")).build().is_err());
    assert!(dose_builder(&benefit, 3, 1).unit_key(key(9, NAMESPACE)).build().is_err());
    let short = UnitKey::canonical_labels(NAMESPACE, patients(0..10));
    assert!(dose_builder(&benefit, 3, 1).unit_key(short).build().is_err());
}

#[test]
fn only_the_pairs_that_share_units_need_a_declared_block() {
    let ctx = ExecutionContext::for_tests(19);
    // Benefit and harm share patients 100-199, harm and cost share 250-299, benefit and cost
    // share none.
    let studies = [
        keyed(&benefit_table(), NAMESPACE, patients(0..200)),
        keyed(&harm_table(), NAMESPACE, patients(100..300)),
        keyed(&cost_table(), NAMESPACE, patients(250..450)),
    ];
    let refs = [&studies[0], &studies[1], &studies[2]];
    let compiled = compile_preference_decision_with_shared_evidence(
        &refs,
        &costed_contract(),
        &shared(vec![block(0, 1), block(1, 2)]),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    assert_eq!(fact.cross_blocks(), &[(0, 1), (1, 2)]);
    let factors = |study: usize| fact.inputs()[study * 2].factors().to_vec();
    let (benefit, harm, cost) = (factors(0), factors(1), factors(2));
    assert_eq!(benefit.len(), 1);
    assert_eq!(cost.len(), 1);
    assert_eq!(harm, [benefit[0].clone(), cost[0].clone()]);
    assert_ne!(benefit, cost);

    // A block for the pair that shares nothing names an overlap that is not there, and a
    // missing block for a pair that shares units is refused.
    let refused = compile_preference_decision_with_shared_evidence(
        &refs,
        &costed_contract(),
        &shared(vec![block(0, 1), block(0, 2), block(1, 2)]),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).1, preference_compile::CROSS_STUDY_COVARIANCE_INVALID);
    let refused = compile_preference_decision_with_shared_evidence(
        &refs,
        &costed_contract(),
        &shared(vec![block(0, 1)]),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).1, preference_compile::STUDIES_SHARE_EVIDENCE);
}

#[test]
fn text_identifiers_holding_a_colon_are_text_units() {
    let ctx = ExecutionContext::for_tests(23);
    // Registry numbers such as `2023:017`: the text before the colon differs between years,
    // but every identifier is text, so the studies are compared by the units they share.
    let ids = |years: std::ops::Range<u32>| -> Vec<String> {
        years.flat_map(|year| (0..100).map(move |n| format!("{year}:{n:03}"))).collect()
    };
    let text = |data: &TabularData, ids: Vec<String>| {
        dose_builder(data, 2, 1).unit_key(UnitKey::text_labels(NAMESPACE, ids)).build().unwrap()
    };
    let first = text(&benefit_table(), ids(2023..2025));
    let second = text(&harm_table(), ids(2024..2026));
    let refused = compile_preference_decision_across_studies(
        &[&first, &second],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        refusal_of(&refused),
        (
            "evidence_double_counted",
            preference_compile::STUDIES_SHARE_EVIDENCE,
            Some("100 shared units in namespace hospital_mrn")
        )
    );
    // Text identifiers are the same units as the canonical text labels of the same values.
    let canonical = keyed(
        &harm_table(),
        NAMESPACE,
        ids(2024..2026).into_iter().map(|id| format!("text:{id}")).collect(),
    );
    let refused = compile_preference_decision_across_studies(
        &[&first, &canonical],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(refusal_of(&refused).2, Some("100 shared units in namespace hospital_mrn"));
    // An untagged identifier given as a canonical label is refused at build rather than read
    // as a type tag.
    let untagged = UnitKey::canonical_labels(NAMESPACE, ids(2023..2025));
    assert!(dose_builder(&benefit_table(), 2, 1).unit_key(untagged).build().is_err());
    for label in ["int64:017", "float64:3FF0000000000000", "bool:2", "text: ", "uuid:7"] {
        let labels = (0..200).map(|_| label.to_owned()).collect();
        let key = UnitKey::canonical_labels(NAMESPACE, labels);
        assert!(dose_builder(&benefit_table(), 2, 1).unit_key(key).build().is_err(), "{label}");
    }
}
