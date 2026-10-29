//! Missing outcomes under a declared policy on the multi-arm, factorial, switchback and
//! blocked randomized designs.
//!
//! Each design runs on a table whose outcome has holes. Without a policy the study is refused.
//! Under complete case the effect is the contrast of the responders' arm means with every
//! assignment kept in its arm, computed here from the definition on the rows that responded;
//! the report names the rows dropped. Inverse probability of observation with no covariates
//! weights every responder by the inverse response rate of its arm, which is the same
//! responder contrast. Multiple imputation completes every table from the arm model and
//! publishes a finite effect with its imputation accounting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study, StudyResult};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0xD352)
}

fn names(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}{i}"))).collect()
}

/// A design, its outcomes before any hole, the rows that did not respond, and the two sets of
/// rows whose responder means the effect contrasts.
struct Case {
    name: &'static str,
    query: RandomizedEffectQuery,
    outcomes: Vec<f64>,
    holes: Vec<usize>,
    treated: Vec<usize>,
    control: Vec<usize>,
    /// Rows of the second factor's "on" level, when the design contrasts within it.
    second_factor: Option<Vec<bool>>,
}

fn multi_arm() -> Case {
    // Three arms of eight units; arm means 0, 2 and 5 and a within-arm spread.
    let n = 24;
    let assignment: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let outcomes: Vec<f64> =
        (0..n).map(|i| [0.0, 2.0, 5.0][i % 3] + 0.5 * ((i / 3) % 4) as f64).collect();
    let arms: Arc<[usize]> = assignment.clone().into();
    Case {
        name: "multi_arm",
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::MultiArm {
                assignment: arms,
                probabilities: vec![vec![1.0 / 3.0; 3]; n].into(),
                arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
            },
            VariableId::from_raw(0),
            assignment.iter().map(|&arm| arm != 0).collect::<Vec<_>>(),
            vec![1.0 / 3.0; n],
            names("unit-", n),
            names("row-", n),
            ("control", "low"),
        ),
        outcomes,
        holes: vec![1, 3, 6, 7, 11],
        treated: (0..n).filter(|i| i % 3 == 1).collect(),
        control: (0..n).filter(|i| i % 3 == 0).collect(),
        second_factor: None,
    }
}

fn factorial() -> Case {
    // Cells of six units, second factor major: (off, off), (on, off), (off, on), (on, on).
    let n = 24;
    let first: Vec<bool> = (0..n).map(|i| (i / 6) % 2 == 1).collect();
    let second: Vec<bool> = (0..n).map(|i| i / 12 == 1).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b) = (f64::from(u8::from(first[i])), f64::from(u8::from(second[i])));
            3.0 * a + 2.0 * b + a * b + 0.5 * (i % 3) as f64
        })
        .collect();
    Case {
        name: "factorial_2x2",
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: second.clone().into(),
                cell_counts: [6; 4],
                second_factor_arms: (Arc::from("off"), Arc::from("on")),
            },
            VariableId::from_raw(0),
            first.clone(),
            vec![0.5; n],
            names("u", n),
            names("y", n),
            ("control", "treated"),
        ),
        outcomes,
        holes: vec![1, 7, 13, 14, 20],
        treated: (0..n).filter(|&i| first[i]).collect(),
        control: (0..n).filter(|&i| !first[i]).collect(),
        second_factor: Some(second),
    }
}

fn switchback() -> Case {
    // Four sequences of four periods; sequence s has baseline 10 s and effect s + 1.
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
    Case {
        name: "switchback",
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Switchback { periods: periods.into() },
            VariableId::from_raw(0),
            assignment.clone(),
            vec![0.5; 16],
            sequences,
            names("row-", 16),
            ("off", "on"),
        ),
        outcomes,
        holes: vec![1, 6, 9, 14],
        treated: (0..16).filter(|&i| assignment[i]).collect(),
        control: (0..16).filter(|&i| !assignment[i]).collect(),
        second_factor: None,
    }
}

fn stratified() -> Case {
    // Two blocks of four with two treated in each and a constant effect of 3.
    let assignment = [true, true, false, false, true, true, false, false];
    let outcomes: Vec<f64> = assignment
        .iter()
        .enumerate()
        .map(|(i, &treated)| 1.0 + 4.0 * (i / 4) as f64 + if treated { 3.0 } else { 0.0 })
        .collect();
    let blocks: Vec<Arc<str>> = (0..8).map(|i| Arc::<str>::from(format!("b{}", i / 4))).collect();
    Case {
        name: "stratified",
        query: RandomizedEffectQuery::with_design(
            RandomizationDesign::Stratified {
                blocks: Arc::from(blocks),
                treated_per_row: Arc::from([2; 8]),
            },
            VariableId::from_raw(0),
            assignment,
            vec![0.5; 8],
            names("u", 8),
            names("y", 8),
            ("control", "treated"),
        ),
        outcomes,
        holes: vec![1, 6],
        treated: vec![0, 1, 4, 5],
        control: vec![2, 3, 6, 7],
        second_factor: None,
    }
}

fn cases() -> Vec<Case> {
    vec![multi_arm(), factorial(), switchback(), stratified()]
}

fn observed(case: &Case) -> Vec<f64> {
    let mut values = case.outcomes.clone();
    for &row in &case.holes {
        values[row] = f64::NAN;
    }
    values
}

fn run(case: &Case, declaration: Option<MissingDeclaration>) -> Result<StudyResult, String> {
    let values = observed(case);
    let data = TabularData::from_f64_columns([("outcome", values.as_slice())]).unwrap();
    let mut builder = Study::tabular(data).query(case.query.clone());
    if let Some(declaration) = declaration {
        builder = builder.missing(declaration);
    }
    builder
        .build()
        .and_then(|study| study.run(&ctx()))
        .map_err(|error| error.to_string())
}

/// The contrast of the responders' mean outcomes in the treated and control rows; the 2x2
/// factorial averages the contrast within the two levels of its second factor.
fn responder_contrast(case: &Case) -> f64 {
    contrast_without(case, &case.holes)
}

/// The same contrast with `holes` removed from every arm mean (none: the table before any hole).
fn contrast_without(case: &Case, holes: &[usize]) -> f64 {
    let contrast = |keep: &dyn Fn(usize) -> bool| {
        let mean = |rows: &[usize]| {
            let kept: Vec<f64> = rows
                .iter()
                .filter(|&&r| keep(r) && !holes.contains(&r))
                .map(|&r| case.outcomes[r])
                .collect();
            kept.iter().sum::<f64>() / kept.len() as f64
        };
        mean(&case.treated) - mean(&case.control)
    };
    match &case.second_factor {
        None => contrast(&|_| true),
        Some(second) => {
            0.5 * (contrast(&|r| !second[r]) + contrast(&|r| second[r]))
        }
    }
}

#[test]
fn every_design_refuses_an_outcome_with_holes_until_a_policy_is_declared() {
    for case in cases() {
        let error = run(&case, None).err().unwrap_or_else(|| panic!("{}: averaged holes", case.name));
        assert!(error.contains("missing_policy_required"), "{}: {error}", case.name);
        let first = case.holes.iter().min().unwrap();
        assert!(error.contains(&format!("row {first}")), "{}: {error}", case.name);
    }
}

#[test]
fn complete_case_is_the_responder_contrast_and_names_the_rows_it_dropped() {
    for case in cases() {
        let result = run(&case, Some(MissingDeclaration::new(MissingPolicy::CompleteCase)))
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let fit = result.randomized_effect().unwrap();
        let expected = responder_contrast(&case);
        assert!(
            (fit.effect - expected).abs() < 1e-9,
            "{}: effect {} vs responder contrast {expected}",
            case.name,
            fit.effect
        );
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.policy(), MissingPolicy::CompleteCase, "{}", case.name);
        let dropped: Vec<u64> = case.holes.iter().map(|&row| row as u64).collect();
        assert_eq!(report.accounting.dropped_rows.as_ref(), dropped.as_slice(), "{}", case.name);
        assert_eq!(report.accounting.rows_total, case.outcomes.len(), "{}", case.name);
        assert_eq!(
            report.accounting.rows_kept,
            case.outcomes.len() - case.holes.len(),
            "{}",
            case.name
        );
    }
}

#[test]
fn inverse_probability_of_observation_without_covariates_is_the_same_responder_contrast() {
    for case in cases() {
        let result = run(
            &case,
            Some(MissingDeclaration::new(MissingPolicy::InverseProbabilityOfObservation)),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let fit = result.randomized_effect().unwrap();
        let expected = responder_contrast(&case);
        assert!(
            (fit.effect - expected).abs() < 1e-9,
            "{}: effect {} vs responder contrast {expected}",
            case.name,
            fit.effect
        );
    }
}

#[test]
fn multiple_imputation_completes_the_table_and_publishes_a_finite_effect() {
    for case in cases() {
        let result = run(
            &case,
            Some(MissingDeclaration::new(MissingPolicy::MultipleImputation(ImputationSpec::new()))),
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        let fit = result.randomized_effect().unwrap();
        assert!(fit.effect.is_finite() && fit.variance_upper_bound > 0.0, "{}", case.name);
        // The imputed effect estimates the contrast of the table before any hole: it lies within
        // three standard errors (the published variance bound) of that contrast.
        let full = contrast_without(&case, &[]);
        let spread = 3.0 * fit.variance_upper_bound.sqrt();
        assert!(
            (fit.effect - full).abs() <= spread,
            "{}: effect {} vs complete-table contrast {full} (3 se = {spread})",
            case.name,
            fit.effect
        );
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert!(matches!(report.policy(), MissingPolicy::MultipleImputation(_)), "{}", case.name);
        assert_eq!(report.accounting.rows_total, case.outcomes.len(), "{}", case.name);
    }
}
