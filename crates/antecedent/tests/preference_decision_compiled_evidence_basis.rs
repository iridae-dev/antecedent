//! Which observations separately estimated outcomes share, and the law they are joined into.
//!
//! The value of an action is `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`
//! (and, across three studies, minus a cost), so the decision needs one aligned law of the
//! outcomes and the evidence ledger must judge, at compile, which studies share observations.
//!
//! * The rows of a table are the same observations whatever analysis mask or weights a study
//!   puts on them; masked rows are not observations a study rests on. A reordered copy made by
//!   a row selection is the same rows.
//! * Studies keyed in one identifier space are judged by their units even when their rows come
//!   from one table: two periods of one patient are one unit.
//! * Each shared pair gets its own chain factor, named by what the pair shares, so three
//!   studies of one table that share units or rows pairwise need exactly the pairs' blocks.
//! * Labels of one space stored under different types are refused, not read as other units.
//! * A declared block may hold a zero term; a block of zeros is refused.
//! * The plan states the basis of every pair's judgement.
//! * The joined law reproduces each study's own posterior covariance: the binding and the
//!   family table read covariance under one convention, so an equally weighted reader of the
//!   joined draws sees each study's covariance within floating-point tolerance.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, CrossStudyCovariance,
    DecisionCriterion, OutcomeDependence, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceRouteError, PreferenceSet, SharedEvidence, StructuralPolicy,
    UnitKey, ValueModel, compile_preference_decision_across_studies,
    compile_preference_decision_with_shared_evidence, shape,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::reason_details::preference_compile;
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
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

fn dose_table(outcome: &str, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

/// One table holding `t`, `b`, `h` and `c` over the same 200 rows.
fn shared_table() -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let column = |(intercept, slope): (f64, f64), rate: f64| -> Vec<f64> {
        (0..rows).zip(&t).map(|(i, t)| intercept + slope * t + 0.6 * wobble(i, rate)).collect()
    };
    let (b, h, c) = (column((2.0, 1.2), 1.7), column((0.5, 0.5), 2.9), column((0.3, 0.2), 0.7));
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("b", b.as_slice()),
        ("h", h.as_slice()),
        ("c", c.as_slice()),
    ])
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

fn keyed(data: &TabularData, labels: Vec<String>) -> Study {
    dose_builder(data, 2, 1).unit_key(UnitKey::canonical_labels(NAMESPACE, labels)).build().unwrap()
}

/// Rows `rows` of the shared table, with `t` and the outcome in column `outcome`.
fn rows_of(rows: &[usize], outcome: u32) -> TabularData {
    let picked = shared_table().select_rows(rows).unwrap();
    let (table, _) =
        picked.project(&[VariableId::from_raw(0), VariableId::from_raw(outcome)]).unwrap();
    table
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

fn refused_as_shared(studies: &[&Study], supplied: &str) {
    let ctx = ExecutionContext::for_tests(3);
    let refused =
        compile_preference_decision_across_studies(studies, &contract(), distinct(), &ctx)
            .unwrap_err();
    assert_eq!(
        refusal_of(&refused),
        ("evidence_double_counted", preference_compile::STUDIES_SHARE_EVIDENCE, Some(supplied))
    );
}

#[test]
fn a_mask_or_weights_do_not_make_one_tables_rows_other_observations() {
    let table = shared_table();
    let harm = dose_builder(&table, 4, 2).build().unwrap();

    // The first 150 rows analysed by one study and every row by the other: 150 shared.
    let keep: Vec<bool> = (0..200).map(|row| row < 150).collect();
    let masked = table.with_analysis_mask(ValidityBitmap::from_bools(&keep)).unwrap();
    refused_as_shared(
        &[&dose_builder(&masked, 4, 1).build().unwrap(), &harm],
        "150 shared observations",
    );

    // The same rows under sampling weights are the same 200 observations.
    let weights: Arc<[f64]> = (0..200).map(|row| 1.0 + f64::from(row % 3)).collect();
    let weighted = table.with_weights(weights).unwrap();
    refused_as_shared(
        &[&dose_builder(&weighted, 4, 1).build().unwrap(), &harm],
        "200 shared observations",
    );

    // A reordered copy taken by a row selection is the same rows.
    let reversed: Vec<usize> = (0..200).rev().collect();
    refused_as_shared(
        &[&study(&rows_of(&reversed, 1)), &study(&rows_of(&(0..200).collect::<Vec<_>>(), 2))],
        "200 shared observations",
    );

    // Masked rows are not observations: a study of the first 100 rows of a table masked to
    // its last 100 shares nothing with them, and the plan says the rows are disjoint.
    let late: Vec<bool> = (0..200).map(|row| row >= 100).collect();
    let (benefit, _) = shared_table()
        .with_analysis_mask(ValidityBitmap::from_bools(&late))
        .unwrap()
        .project(&[VariableId::from_raw(0), VariableId::from_raw(1)])
        .unwrap();
    let early = rows_of(&(0..100).collect::<Vec<_>>(), 2);
    let ctx = ExecutionContext::for_tests(5);
    let compiled = compile_preference_decision_across_studies(
        &[&study(&benefit), &study(&early)],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().expect("a declared chain").fact();
    assert_eq!(fact.pairs().len(), 1);
    assert_eq!(
        (fact.pairs()[0].0, fact.pairs()[0].1, &*fact.pairs()[0].2),
        (0, 1, "disjoint_rows")
    );

    // Studies of different tables without keys are distinct on the declaration alone.
    let compiled = compile_preference_decision_across_studies(
        &[&study(&dose_table("b", (2.0, 1.2), 1.7)), &study(&dose_table("h", (0.5, 0.5), 2.9))],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    assert_eq!(&*fact.pairs()[0].2, "declared_distinct");
}

#[test]
fn keyed_studies_of_one_table_are_judged_by_their_units() {
    // Rows 2k and 2k + 1 are patient k's two visits; one study reads the first visits, the
    // other the second. No row is shared, every patient is.
    let ctx = ExecutionContext::for_tests(7);
    let first: Vec<usize> = (0..200).step_by(2).collect();
    let second: Vec<usize> = (1..200).step_by(2).collect();
    let patients = |rows: &[usize]| -> Vec<String> {
        rows.iter().map(|row| format!("text:p{}", row / 2)).collect()
    };
    let visit_one = keyed(&rows_of(&first, 1), patients(&first));
    let visit_two = keyed(&rows_of(&second, 2), patients(&second));
    let refused = compile_preference_decision_across_studies(
        &[&visit_one, &visit_two],
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
    let compiled = compile_preference_decision_with_shared_evidence(
        &[&visit_one, &visit_two],
        &contract(),
        &shared(vec![block(0, 1)]),
        &ctx,
    )
    .unwrap();
    let fact = compiled.joint_law().unwrap().fact();
    assert!(fact.inputs()[0].factor().starts_with("shared_units:"));
    assert_eq!(fact.inputs()[0].factors(), fact.inputs()[2].factors());
    assert_eq!(&*fact.pairs()[0].2, "shared_units");
}

#[test]
fn three_studies_of_one_table_need_only_the_blocks_of_the_pairs_that_share() {
    // A reads rows 0-99, C rows 100-199, B every row; each row is its own patient.
    let ctx = ExecutionContext::for_tests(11);
    let low: Vec<usize> = (0..100).collect();
    let high: Vec<usize> = (100..200).collect();
    let all: Vec<usize> = (0..200).collect();
    let labels =
        |rows: &[usize]| -> Vec<String> { rows.iter().map(|row| format!("text:p{row}")).collect() };
    let a = keyed(&rows_of(&low, 1), labels(&low));
    let c = keyed(&rows_of(&high, 3), labels(&high));
    for b in [keyed(&rows_of(&all, 2), labels(&all)), study(&rows_of(&all, 2))] {
        let compiled = compile_preference_decision_with_shared_evidence(
            &[&a, &b, &c],
            &costed_contract(),
            &shared(vec![block(0, 1), block(1, 2)]),
            &ctx,
        )
        .unwrap();
        let fact = compiled.joint_law().unwrap().fact();
        assert_eq!(fact.cross_blocks(), &[(0, 1), (1, 2)]);
        let factors = |study: usize| fact.inputs()[study * 2].factors().to_vec();
        let (first, middle, last) = (factors(0), factors(1), factors(2));
        assert_eq!(first.len(), 1);
        assert_eq!(last.len(), 1);
        assert_eq!(middle, [first[0].clone(), last[0].clone()]);
        assert_ne!(first, last, "A and C share nothing");
        let bases: Vec<&str> = fact.pairs().iter().map(|(_, _, basis)| &**basis).collect();
        assert_eq!(bases.len(), 3);
        assert!(!bases.contains(&"declared_distinct"));
    }
}

#[test]
fn labels_of_one_space_under_different_types_are_refused() {
    let ctx = ExecutionContext::for_tests(13);
    let integers = (0..200).map(|i| format!("int64:{i}")).collect();
    let reals = (0..200).map(|i| format!("float64:{:016x}", f64::from(i).to_bits())).collect();
    let refused = compile_preference_decision_across_studies(
        &[
            &keyed(&dose_table("b", (2.0, 1.2), 1.7), integers),
            &keyed(&dose_table("h", (0.5, 0.5), 2.9), reals),
        ],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap_err();
    let (code, detail, _) = refusal_of(&refused);
    assert_eq!(
        (code, detail),
        ("evidence_overlap_unknown", preference_compile::UNIT_LABEL_TYPES_DIFFER)
    );
}

#[test]
fn a_declared_block_may_hold_a_zero_term_but_not_only_zeros() {
    let ctx = ExecutionContext::for_tests(17);
    let table = shared_table();
    let (benefit, harm) =
        (dose_builder(&table, 4, 1).build().unwrap(), dose_builder(&table, 4, 2).build().unwrap());
    let with_zero =
        CrossStudyCovariance { first: 0, second: 1, covariance: vec![1e-5, 0.0, 0.0, 1e-5] };
    let compiled = compile_preference_decision_with_shared_evidence(
        &[&benefit, &harm],
        &contract(),
        &shared(vec![with_zero]),
        &ctx,
    )
    .unwrap();
    assert!(compiled.execute(None, PreferenceEvaluationOptions::default(), &ctx).is_ok());

    let zeros = CrossStudyCovariance { first: 0, second: 1, covariance: vec![0.0; 4] };
    let refused = compile_preference_decision_with_shared_evidence(
        &[&benefit, &harm],
        &contract(),
        &shared(vec![zeros]),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        refusal_of(&refused),
        (
            "invalid_argument",
            preference_compile::CROSS_STUDY_COVARIANCE_INVALID,
            Some("a block whose every term is zero")
        )
    );
}

/// Sample covariance (divisor `n - 1`) of two aligned columns.
fn sample_covariance(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n - 1.0)
}

/// Column `j` of a row-major `[rows, width]` table.
fn column(draws: &[f64], [rows, width]: [usize; 2], j: usize) -> Vec<f64> {
    (0..rows).map(|row| draws[row * width + j]).collect()
}

#[test]
fn the_joined_law_reproduces_each_studys_own_covariance() {
    let ctx = ExecutionContext::for_tests(19);
    let compiled = compile_preference_decision_across_studies(
        &[&study(&dose_table("b", (2.0, 1.2), 1.7)), &study(&dose_table("h", (0.5, 0.5), 2.9))],
        &contract(),
        distinct(),
        &ctx,
    )
    .unwrap();
    let executed = compiled.execute(None, PreferenceEvaluationOptions::default(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(joined)) = &executed.source else {
        panic!("one aligned law");
    };
    assert!(joined.metadata().weights.is_none(), "the joined rows are equally weighted");
    let joined_shape = joined.shape();
    let mut compared = 0;
    for input in &executed.inputs {
        let NativeDecisionSource::JointLaw(law) = &input.source else {
            panic!("each study's posterior law");
        };
        let own_shape = law.shape();
        let at = |j: usize| -> usize {
            joined
                .quantities()
                .iter()
                .position(|quantity| *quantity == law.quantities()[j])
                .expect("the joined law carries every study's coordinates")
        };
        for p in 0..own_shape[1] {
            for q in 0..own_shape[1] {
                let own = sample_covariance(
                    &column(law.draws(), own_shape, p),
                    &column(law.draws(), own_shape, q),
                );
                let read = sample_covariance(
                    &column(joined.draws(), joined_shape, at(p)),
                    &column(joined.draws(), joined_shape, at(q)),
                );
                let scale = (sample_covariance(
                    &column(law.draws(), own_shape, p),
                    &column(law.draws(), own_shape, p),
                ) * sample_covariance(
                    &column(law.draws(), own_shape, q),
                    &column(law.draws(), own_shape, q),
                ))
                .sqrt();
                assert!((read - own).abs() <= 1e-8 * scale, "own {own}, joined {read}");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 8, "two coordinates of each of two studies");
}
