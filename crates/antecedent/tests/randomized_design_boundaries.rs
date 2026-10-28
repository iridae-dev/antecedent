//! Row-level boundaries of the randomized designs without a sampling-weighted contrast.
//!
//! Each design (multi-arm, 2x2 factorial, switchback, blocked) has a closed-form effect on a
//! small table. Reordering the rows, with every row-aligned declaration moved with them, must
//! leave the effect unchanged; an outcome that is constant has effect exactly zero and no
//! variance (the Horvitz-Thompson bound of the multi-arm design excepted); an outcome table
//! with a missing value is refused rather than averaged over; and a design whose declared arm
//! has no realized unit is refused rather than extrapolated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::{TableView, TabularData};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0xD351)
}

fn units(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}{i}"))).collect()
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


/// The query with every row-aligned declaration reordered by `order` (new row `k` is old row
/// `order[k]`).
fn reordered(query: &RandomizedEffectQuery, order: &[usize]) -> RandomizedEffectQuery {
    let pick = |len: usize, f: &dyn Fn(usize) -> usize| -> Vec<usize> { (0..len).map(f).collect() };
    let rows = pick(order.len(), &|k| order[k]);
    let take = |values: &[bool]| rows.iter().map(|&i| values[i]).collect::<Vec<_>>();
    let take_text =
        |values: &[Arc<str>]| rows.iter().map(|&i| Arc::clone(&values[i])).collect::<Vec<_>>();
    let design = match &query.design {
        RandomizationDesign::MultiArm { assignment, probabilities, arms } => {
            RandomizationDesign::MultiArm {
                assignment: rows.iter().map(|&i| assignment[i]).collect::<Vec<_>>().into(),
                probabilities: rows.iter().map(|&i| probabilities[i].to_vec()).collect::<Vec<_>>().into(),
                arms: arms.clone(),
            }
        }
        RandomizationDesign::Factorial2x2 { second_factor_assignment, cell_counts, second_factor_arms } => {
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: take(second_factor_assignment).into(),
                cell_counts: *cell_counts,
                second_factor_arms: second_factor_arms.clone(),
            }
        }
        RandomizationDesign::Switchback { periods } => {
            RandomizationDesign::Switchback { periods: take_text(periods).into() }
        }
        RandomizationDesign::Stratified { blocks, treated_per_row } => {
            RandomizationDesign::Stratified {
                blocks: take_text(blocks).into(),
                treated_per_row: rows.iter().map(|&i| treated_per_row[i]).collect::<Vec<_>>().into(),
            }
        }
        other => other.clone(),
    };
    RandomizedEffectQuery::with_design(
        design,
        query.outcome,
        take(&query.realized_assignment),
        rows.iter().map(|&i| query.assignment_probabilities[i]).collect::<Vec<_>>(),
        take_text(&query.assignment_units),
        take_text(&query.outcome_units),
        (query.treatment_arms.0.clone(), query.treatment_arms.1.clone()),
    )
}

fn outcomes_of(case: &Case) -> Vec<f64> {
    match case.data.column(VariableId::from_raw(0)).unwrap() {
        antecedent_data::ColumnView::Float64(column) => column.values.as_slice().to_vec(),
        _ => unreachable!("the fixture outcome is continuous"),
    }
}

fn effect_of(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<(f64, f64), String> {
    let data = TabularData::from_f64_columns([("outcome", outcomes)]).unwrap();
    Study::tabular(data)
        .query(query.clone())
        .build()
        .and_then(|study| study.run(&ctx()))
        .map(|result| {
            let fit = result.randomized_effect().unwrap();
            (fit.effect, fit.variance_upper_bound)
        })
        .map_err(|error| error.to_string())
}

#[test]
fn every_design_gives_its_closed_form_effect_and_is_invariant_to_the_order_of_rows() {
    for case in cases() {
        let outcomes = outcomes_of(&case);
        let (effect, variance) = effect_of(&case.query, &outcomes).unwrap();
        assert!((effect - case.effect).abs() < 1e-12, "{}: {effect}", case.name);
        let n = outcomes.len();
        let orders: [Vec<usize>; 2] = [
            (0..n).rev().collect(),
            (0..n).map(|k| (k * 3 + 1) % n).collect::<Vec<_>>(),
        ];
        for order in orders {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            if sorted != (0..n).collect::<Vec<_>>() {
                continue;
            }
            let moved: Vec<f64> = order.iter().map(|&i| outcomes[i]).collect();
            let (moved_effect, moved_variance) =
                effect_of(&reordered(&case.query, &order), &moved).unwrap();
            assert!((moved_effect - effect).abs() < 1e-12, "{}: {moved_effect}", case.name);
            assert!((moved_variance - variance).abs() < 1e-12, "{}: {moved_variance}", case.name);
        }
    }
}

#[test]
fn a_constant_outcome_has_exactly_zero_effect_and_no_variance_in_every_design() {
    for case in cases() {
        let constant = vec![4.25; outcomes_of(&case).len()];
        let (effect, variance) = effect_of(&case.query, &constant).unwrap();
        assert_eq!(effect, 0.0, "{}", case.name);
        if case.name == "multi_arm" {
            // The Horvitz-Thompson bound prices the chance of an arm count away from its
            // expectation, so a constant outcome still carries a positive bound.
            assert!(variance > 0.0, "{}: {variance}", case.name);
        } else {
            assert_eq!(variance, 0.0, "{}", case.name);
        }
    }
}

#[test]
fn an_outcome_with_a_missing_value_is_refused_by_every_design_without_a_policy() {
    for case in cases() {
        let mut outcomes = outcomes_of(&case);
        outcomes[2] = f64::NAN;
        let error = effect_of(&case.query, &outcomes).unwrap_err();
        assert!(error.starts_with("reason="), "{}: {error}", case.name);
    }
}

/// The two-arm designs added to the four above: Bernoulli and cluster randomization.
fn two_arm_cases() -> Vec<Case> {
    let n = 12;
    let outcomes: Vec<f64> = (0..n).map(|i| 1.0 + f64::from(i % 3)).collect();
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let clusters: Vec<Arc<str>> = (0..n).map(|i| Arc::<str>::from(format!("c{}", i / 2))).collect();
    let cluster_assignment: Vec<bool> = (0..n).map(|i| (i / 2) % 2 == 0).collect();
    vec![
        Case {
            name: "bernoulli",
            data: TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap(),
            query: RandomizedEffectQuery::bernoulli_itt(
                VariableId::from_raw(0),
                assignment,
                vec![0.5; n as usize],
                units("u", n as usize),
                units("y", n as usize),
                ("control", "treated"),
            ),
            effect: 0.0,
        },
        Case {
            name: "cluster",
            data: TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap(),
            query: RandomizedEffectQuery::with_design(
                RandomizationDesign::Cluster { treated_clusters: 3 },
                VariableId::from_raw(0),
                cluster_assignment,
                vec![0.5; n as usize],
                clusters,
                units("y", n as usize),
                ("control", "treated"),
            ),
            effect: 0.0,
        },
    ]
}

#[test]
fn a_design_with_no_unit_in_an_arm_it_declares_is_refused_not_extrapolated() {
    let mut all = cases();
    all.extend(two_arm_cases());
    for case in all {
        let outcomes = outcomes_of(&case);
        let mut query = case.query.clone();
        // Every row is realized in the control arm: the treated arm was never visited.
        query.realized_assignment = vec![false; outcomes.len()].into();
        if let RandomizationDesign::MultiArm { assignment, probabilities, arms } = &query.design {
            query.design = RandomizationDesign::MultiArm {
                assignment: vec![0; assignment.len()].into(),
                probabilities: probabilities.clone(),
                arms: arms.clone(),
            };
        }
        let error = effect_of(&query, &outcomes).expect_err(case.name);
        assert!(
            error.starts_with("reason=") || error.contains("invalid randomized-effect query"),
            "{}: {error}",
            case.name
        );
    }
}
