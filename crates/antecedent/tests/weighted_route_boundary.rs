//! Randomized and panel designs that have no sampling-weighted contrast refuse declared weights.
//!
//! Each design is first run without weights on a small table whose answer is written out by
//! hand (a closed-form contrast), so the refusal below it is a refusal of a route that works,
//! not of a route that was never reachable. The same table with observation weights, including
//! the constant weight one, is then refused with `option_not_applicable` and the detail
//! `sampling_weights.route_unsupported`: the weights are not silently ignored and the unweighted
//! number is not published as the weighted population's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, RandomizedEffectQuery, Study};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::TabularData;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED)
}

fn units(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}{i}"))).collect()
}

fn refusal_of(study: Study, data: &TabularData) -> String {
    let error = study.prepare(&ctx()).and_then(|prepared| prepared.estimate(data, &ctx()));
    error.map(|_| ()).unwrap_err().to_string()
}

fn assert_weights_refused(error: &str, design: &str) {
    assert!(error.starts_with("reason=option_not_applicable"), "{design}: {error}");
    assert!(error.contains("sampling_weights.route_unsupported"), "{design}: {error}");
}

fn weighted(data: &TabularData, weights: Vec<f64>) -> TabularData {
    data.with_weights(Arc::from(weights)).unwrap()
}

/// A randomized design, the table it runs on and its hand-computed effect.
struct Case {
    name: &'static str,
    data: TabularData,
    query: RandomizedEffectQuery,
    effect: f64,
}

fn cases() -> Vec<Case> {
    let mut cases = vec![];

    // Three arms, arm means 0, 2, 5: the low-versus-control contrast is exactly 2.
    let outcomes = [0.0, 2.0, 5.0, 0.0, 2.0, 5.0];
    let assignment: Arc<[usize]> = [0, 1, 2, 0, 1, 2].into();
    cases.push(Case {
        name: "multi_arm",
        data: TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap(),
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::MultiArm {
                assignment: assignment.clone(),
                probabilities: vec![vec![1.0 / 3.0; 3]; 6].into(),
                arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
            },
            VariableId::from_raw(0),
            assignment.iter().map(|&arm| arm != 0).collect::<Vec<_>>(),
            [1.0 / 3.0; 6],
            units("unit-", 6),
            units("row-", 6),
            ("control", "low"),
        ),
        effect: 2.0,
    });

    // 2x2 factorial with cell means 0, 2 / 2, 4 and 1, 3 / 5, 7 over the two factors: the
    // primary main effect, averaged over the second factor, is exactly 3.
    let outcomes = [0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0];
    cases.push(Case {
        name: "factorial_2x2",
        data: TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap(),
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: [false, false, false, false, true, true, true, true]
                    .into(),
                cell_counts: [2; 4],
                second_factor_arms: (Arc::from("off"), Arc::from("on")),
            },
            VariableId::from_raw(0),
            [false, false, true, true, false, false, true, true],
            [0.5; 8],
            units("u", 8),
            units("y", 8),
            ("control", "treated"),
        ),
        effect: 3.0,
    });

    // Four sequences of four periods; sequence s has baseline 10 s and a treatment effect
    // s + 1, so the mean treated-minus-control difference is (1 + 2 + 3 + 4) / 4 = 2.5.
    let assignment = [true, false, false, true].repeat(4);
    let outcomes: Vec<f64> = (0..4)
        .flat_map(|sequence| {
            [true, false, false, true].into_iter().map(move |treated| {
                10.0 * f64::from(sequence) + if treated { f64::from(sequence) + 1.0 } else { 0.0 }
            })
        })
        .collect();
    let sequences: Vec<Arc<str>> = (0..4)
        .flat_map(|sequence| (0..4).map(move |_| Arc::<str>::from(format!("s{sequence}"))))
        .collect();
    let periods: Vec<Arc<str>> = (0..4)
        .flat_map(|_| (0..4).map(|period| Arc::<str>::from(format!("p{period}"))))
        .collect();
    cases.push(Case {
        name: "switchback",
        data: TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap(),
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Switchback { periods: periods.into() },
            VariableId::from_raw(0),
            assignment,
            [0.5; 16],
            sequences,
            units("row-", 16),
            ("off", "on"),
        ),
        effect: 2.5,
    });

    // Two blocks of four, two treated in each, a constant effect of 3: the blocked
    // difference in means is exactly 3.
    let assignment = [true, true, false, false, true, true, false, false];
    let outcomes: Vec<f64> = assignment
        .iter()
        .enumerate()
        .map(|(i, &treated)| 1.0 + 4.0 * (i / 4) as f64 + if treated { 3.0 } else { 0.0 })
        .collect();
    let blocks: Vec<Arc<str>> = (0..8).map(|i| Arc::<str>::from(format!("b{}", i / 4))).collect();
    cases.push(Case {
        name: "stratified",
        data: TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap(),
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Stratified {
                blocks: Arc::from(blocks),
                treated_per_row: Arc::from([2; 8]),
            },
            VariableId::from_raw(0),
            assignment,
            [0.5; 8],
            units("u", 8),
            units("y", 8),
            ("control", "treated"),
        ),
        effect: 3.0,
    });
    cases
}

#[test]
fn designs_without_a_weighted_contrast_give_their_closed_form_effect_unweighted() {
    for case in cases() {
        let result = Study::tabular(case.data.clone())
            .query(case.query.clone())
            .build()
            .unwrap()
            .run(&ctx())
            .unwrap();
        let fit = result.randomized_effect().unwrap();
        assert!((fit.effect - case.effect).abs() < 1e-12, "{}: {}", case.name, fit.effect);
    }
}

#[test]
fn declared_weights_are_refused_by_every_randomized_design_without_a_weighted_contrast() {
    for case in cases() {
        let n = case.query.realized_assignment.len();
        let uneven: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        for weights in [uneven, vec![1.0; n]] {
            let data = weighted(&case.data, weights);
            let study = Study::tabular(data.clone()).query(case.query.clone()).build().unwrap();
            assert_weights_refused(&refusal_of(study, &data), case.name);
        }
    }
}

#[test]
fn a_staggered_group_time_panel_refuses_declared_weights() {
    let (mut outcome, mut subjects, mut clusters, mut periods, mut cohorts) =
        (vec![], vec![], vec![], vec![], vec![]);
    for subject in 0..8 {
        for period in 1..=4_i64 {
            let cohort = if subject < 4 { 0 } else { 3 };
            subjects.push(Arc::<str>::from(format!("s{subject}")));
            clusters.push(Arc::<str>::from(format!("c{subject}")));
            periods.push(period);
            cohorts.push(cohort);
            outcome.push(
                f64::from(subject)
                    + 2.0 * period as f64
                    + if cohort == 3 && period >= 3 { 4.0 } else { 0.0 },
            );
        }
    }
    let query = PanelDidQuery::staggered_group_time(
        VariableId::from_raw(0),
        subjects,
        clusters,
        periods,
        cohorts,
        3,
        4,
    );
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let unweighted =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&ctx()).unwrap();
    // Treated cohort 3 gains 4 from period 3 on: the group-time effect is exactly 4.
    assert!((unweighted.panel_did().unwrap().effect - 4.0).abs() < 1e-12);
    let data = weighted(&data, (0..outcome.len()).map(|i| 1.0 + (i % 4) as f64).collect());
    let study = Study::tabular(data.clone()).query(query).build().unwrap();
    assert_weights_refused(&refusal_of(study, &data), "staggered_group_time");
}

#[test]
fn an_augmented_panel_refuses_declared_weights() {
    let pre = [10.0, 12.0, 13.0, 9.0, 15.0, 11.0];
    let post = [15.0, 17.0, 18.0, 11.0, 17.0, 13.0];
    let propensity = [0.5; 6];
    let prediction = [2.0; 6];
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", propensity.as_slice()),
        ("m0", prediction.as_slice()),
    ])
    .unwrap();
    let ids = units("s", 6);
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        [true, true, true, false, false, false],
        ids.clone(),
        ids,
        true,
    );
    // Each treated unit gains 5, 5, 5 and each comparison unit 2, 2, 2: the effect is 3.
    let unweighted =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&ctx()).unwrap();
    assert!((unweighted.panel_did().unwrap().effect - 3.0).abs() < 1e-12);
    let data = weighted(&data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    let study = Study::tabular(data.clone()).query(query).build().unwrap();
    assert_weights_refused(&refusal_of(study, &data), "augmented_panel");
}
