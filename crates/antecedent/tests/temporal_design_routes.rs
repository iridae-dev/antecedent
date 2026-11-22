//! Sequentially randomized unit-period designs and regime survival through the public study
//! flow, against simulated truths.
//!
//! Every fixture draws a panel from a known law: the excursion effects, the dose response, the
//! quantile shift and the regime survival contrast are known, and the routes must recover them
//! within their own reported uncertainty. The routes carry no matrix support status, a
//! `calibration = "not_measured"` section, and a refusal for every option they do not read; a
//! fit that matches a row of the graphless support table states the row in a coded diagnostic.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "small counts convert exactly")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PrimaryEstimate, Study, TemporalDesignEstimate};
use antecedent_core::{
    CausalQuery, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, VariableId,
};
use antecedent_data::TabularData;

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

/// `SplitMix64` uniform stream.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

/// A two-arm micro-randomized panel: the treatment probability depends on the previous outcome,
/// the outcome one period after a decision carries `effect` of that decision and a carryover of
/// `0.3` of the decision before it.
struct TwoArmPanel {
    units: Vec<String>,
    periods: Vec<i64>,
    assignment: Vec<bool>,
    probability: Vec<f64>,
    outcome: Vec<f64>,
}

fn two_arm_panel(units: usize, periods: usize, effect: f64, seed: u64) -> TwoArmPanel {
    let mut stream = Stream(seed);
    let mut panel = TwoArmPanel {
        units: Vec::new(),
        periods: Vec::new(),
        assignment: Vec::new(),
        probability: Vec::new(),
        outcome: Vec::new(),
    };
    for i in 0..units {
        let level = stream.next() - 0.5;
        let (mut previous_y, mut previous_a, mut before) = (0.0, 0.0, 0.0);
        for t in 0..periods {
            let p = if previous_y > 0.0 { 0.65 } else { 0.35 };
            let a = stream.next() < p;
            let y = level + effect * previous_a + 0.3 * before + (stream.next() - 0.5);
            panel.units.push(format!("u{i}"));
            panel.periods.push(i64::try_from(t).unwrap());
            panel.assignment.push(a);
            panel.probability.push(p);
            panel.outcome.push(y);
            previous_y = y;
            before = previous_a;
            previous_a = f64::from(u8::from(a));
        }
    }
    panel
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied()).unwrap()
}

fn two_arm_query(panel: &TwoArmPanel, functional: TemporalDesignFunctional) -> TemporalDesignQuery {
    TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&panel.units),
        periods: panel.periods.as_slice().into(),
        lag: 1,
        assignment: TemporalAssignment::TwoArm {
            assignment: panel.assignment.as_slice().into(),
            probability: panel.probability.as_slice().into(),
        },
        functional,
        min_probability: 0.05,
        outcome_misclassification: None,
    }
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

fn run(
    data: &TabularData,
    query: TemporalDesignQuery,
    inference: InferenceMode,
    replicates: u32,
    ctx: &ExecutionContext,
) -> antecedent::StudyResult {
    let mut builder =
        Study::tabular(data.clone()).query(CausalQuery::TemporalDesign(query)).inference(inference);
    if replicates > 0 {
        builder = builder.bootstrap_replicates(replicates);
    }
    builder.build().unwrap().run(ctx).unwrap()
}

fn section(result: &antecedent::StudyResult) -> &TemporalDesignEstimate {
    result.temporal_design().expect("a temporal design section")
}

/// Code of the diagnostic that states a temporal design's support row.
const DESIGN_ROW: &str = antecedent_io::graphless_support::TEMPORAL_DESIGN_ROW_CODE;
/// The artifact check's refusal of a stated row the stored section does not meet.
const ROW_NOT_MET: &str = "states a graphless support row";

fn within(section: &TemporalDesignEstimate, index: usize, truth: f64, multiple: f64) {
    let estimate = section.estimate[index];
    let scale = section.standard_error[index];
    assert!(
        (estimate - truth).abs() <= multiple * scale,
        "{} = {estimate} (se {scale}) vs truth {truth}",
        section.coordinates[index]
    );
}

#[test]
fn the_two_arm_excursion_recovers_the_effect_under_carryover_in_both_modes() {
    let panel = two_arm_panel(300, 10, 1.25, 7);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let ctx = ExecutionContext::for_tests(11);
    let frequentist = run(
        &data,
        two_arm_query(&panel, TemporalDesignFunctional::Mean),
        InferenceMode::Frequentist,
        0,
        &ctx,
    );
    assert_eq!(frequentist.logical_plan.estimator.as_deref(), Some("temporal_design.excursion"));
    assert!(matches!(frequentist.estimate(), PrimaryEstimate::NotAnEffect));
    assert_eq!(frequentist.support_status(), None);
    let fit = section(&frequentist);
    assert_eq!(&*fit.design, "two_arm");
    // 300 units, 2,700 pairs and probabilities of 0.35 and 0.65 meet the two-arm support row.
    assert!(frequentist.states_section_row());
    assert!(!fit.supported && &*fit.calibration == "not_measured");
    assert_eq!(&*fit.coordinates[0], "excursion_effect");
    within(fit, 0, 1.25, 4.0);
    assert!(fit.lower[0] < fit.estimate[0] && fit.estimate[0] < fit.upper[0]);
    assert_eq!(fit.clusters, 300);
    assert_eq!(fit.unpaired, 300);

    let posterior =
        run(&data, two_arm_query(&panel, TemporalDesignFunctional::Mean), bayesian(400), 0, &ctx);
    assert_eq!(
        posterior.logical_plan.estimator.as_deref(),
        Some("bayesian.temporal_design.excursion")
    );
    assert!(posterior.posterior.is_some());
    let draws = section(&posterior);
    assert_eq!(draws.draws, Some(400));
    assert!((draws.estimate[0] - fit.estimate[0]).abs() < 0.5 * fit.standard_error[0] + 0.02);
}

#[test]
fn the_prepared_route_matches_the_one_shot_route_at_any_thread_count() {
    let panel = two_arm_panel(120, 6, 0.8, 3);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let query = two_arm_query(&panel, TemporalDesignFunctional::Mean);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(query))
        .inference(bayesian(200))
        .build()
        .unwrap();
    let serial = ExecutionContext::for_tests(5);
    let one_shot = study.run(&serial).unwrap();
    let prepared = study.prepare(&serial).unwrap().estimate(&data, &serial).unwrap();
    assert_eq!(section(&one_shot), section(&prepared));
    let parallel = ExecutionContext::production(5, 4);
    let threaded = study.run(&parallel).unwrap();
    assert_eq!(section(&one_shot).estimate, section(&threaded).estimate);
}

#[test]
fn a_tiny_admission_pool_refuses_the_posterior_and_returns_its_lease() {
    let panel = two_arm_panel(60, 5, 1.0, 9);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(two_arm_query(&panel, TemporalDesignFunctional::Mean)))
        .inference(bayesian(500))
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(1))
        .unwrap();
    let tiny = ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 64));
    let before = tiny.admission.available();
    let error = prepared.estimate(&data, &tiny).unwrap_err();
    assert!(error.to_string().contains("admission"), "{error}");
    assert_eq!(tiny.admission.available(), before);
}

#[test]
fn options_the_design_does_not_read_are_refused_with_the_next_attempt() {
    let panel = two_arm_panel(40, 4, 1.0, 13);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let error = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(two_arm_query(&panel, TemporalDesignFunctional::Mean)))
        .bootstrap_replicates(99)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(1)))
        .unwrap_err();
    assert!(error.to_string().contains("Next attempt"), "{error}");
    let quantile = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(two_arm_query(
            &panel,
            TemporalDesignFunctional::Quantile { level: 0.5 },
        )))
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(1)))
        .unwrap_err();
    assert!(quantile.to_string().contains("bootstrap_replicates"), "{quantile}");
}

/// A multi-arm panel at lag 0 with arm effects `0, 0.5, 1.5` and a probability row that moves
/// with the period.
fn multi_arm(ordered: bool, categorical: bool) -> (TabularData, TemporalDesignQuery) {
    multi_arm_of(250, ordered, categorical)
}

/// [`multi_arm`] with `units` units.
fn multi_arm_of(
    units_total: usize,
    ordered: bool,
    categorical: bool,
) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(21);
    let (mut units, mut periods, mut assignment, mut probabilities, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let effects = [0.0, 0.5, 1.5];
    for i in 0..units_total {
        let level = stream.next() - 0.5;
        for t in 0..6_i64 {
            let row = if t % 2 == 0 { vec![0.5, 0.25, 0.25] } else { vec![0.2, 0.4, 0.4] };
            let u = stream.next();
            let arm = if u < row[0] {
                0
            } else if u < row[0] + row[1] {
                1
            } else {
                2
            };
            units.push(format!("u{i}"));
            periods.push(t);
            assignment.push(arm);
            probabilities.push(row);
            outcome.push(level + effects[arm] + (stream.next() - 0.5));
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::MultiArm {
            assignment: assignment.as_slice().into(),
            probabilities: probabilities.into(),
            arms: labels(&["low".into(), "mid".into(), "high".into()]),
            categorical,
            ordered,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.1,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

#[test]
fn multi_arm_and_ordered_categorical_contrasts_carry_a_band_and_a_holm_family() {
    let ctx = ExecutionContext::for_tests(17);
    let (data, query) = multi_arm(false, false);
    let result = run(&data, query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    assert_eq!(&*fit.design, "multi_arm");
    within(fit, 0, 0.5, 4.0);
    within(fit, 1, 1.5, 4.0);
    assert_eq!(&*fit.band_coordinates, &[0, 1]);
    assert!(fit.band_critical_value.unwrap() > 1.96);
    assert_eq!(fit.holm_adjusted_p.as_ref().unwrap().len(), 2);

    let (data, query) = multi_arm(true, true);
    let result = run(&data, query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    assert_eq!(&*fit.design, "categorical");
    assert_eq!(&*fit.coordinates[2], "step:high-mid");
    within(fit, 2, 1.0, 4.0);
    let posterior = run(&data, multi_arm(true, true).1, bayesian(300), 0, &ctx);
    assert!((section(&posterior).estimate[1] - 1.5).abs() < 0.2);
}

#[test]
fn the_factorial_and_interference_designs_recover_their_effects() {
    let ctx = ExecutionContext::for_tests(23);
    let mut stream = Stream(29);
    let (mut units, mut periods, mut first, mut second, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..300 {
        for t in 0..5_i64 {
            let a = stream.next() < 0.5;
            let b = stream.next() < 0.4;
            let (fa, fb) = (f64::from(u8::from(a)), f64::from(u8::from(b)));
            units.push(format!("u{i}"));
            periods.push(t);
            first.push(a);
            second.push(b);
            outcome.push(1.0 * fa + 0.5 * fb + 0.4 * fa * fb + (stream.next() - 0.5));
        }
    }
    let rows = units.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Factorial {
            first: first.as_slice().into(),
            first_probability: vec![0.5; rows].into(),
            second: second.as_slice().into(),
            second_probability: vec![0.4; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    let result =
        run(&table(&[("y", outcome.as_slice())]), query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    // Main effects average over the other factor's two levels with equal weight.
    within(fit, 0, 1.0 + 0.5 * 0.4, 4.0);
    within(fit, 1, 0.5 + 0.5 * 0.4, 4.0);
    within(fit, 2, 0.4, 4.0);

    // Pairs of units share an interference cluster; a treated peer adds 0.6.
    let (mut units, mut periods, mut clusters, mut assignment) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..300 {
        for t in 0..4_i64 {
            for member in 0..2 {
                units.push(format!("u{c}_{member}"));
                periods.push(t);
                clusters.push(format!("c{c}"));
                assignment.push(stream.next() < 0.5);
            }
        }
    }
    let rows = units.len();
    let outcome: Vec<f64> = (0..rows)
        .map(|row| {
            let peer = if row % 2 == 0 { row + 1 } else { row - 1 };
            f64::from(u8::from(assignment[row]))
                + 0.6 * f64::from(u8::from(assignment[peer]))
                + (stream.next() - 0.5)
        })
        .collect();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Interference {
            assignment: assignment.as_slice().into(),
            probability: vec![0.5; rows].into(),
            clusters: labels(&clusters),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    let result =
        run(&table(&[("y", outcome.as_slice())]), query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    assert_eq!(&*fit.design, "interference");
    within(fit, 0, 1.0, 4.0);
    within(fit, 1, 0.6, 4.0);
    assert_eq!(fit.clusters, 300);
}

/// Doses uniform on `(0, 2)` (density 0.5) at every decision; outcome `1 + d + 0.5 d^2` plus
/// noise in the same period.
fn dose_panel(target: TemporalDoseTarget) -> (TabularData, TemporalDesignQuery) {
    dose_panel_of(300, target)
}

/// [`dose_panel`] with `units` units.
fn dose_panel_of(
    units_total: usize,
    target: TemporalDoseTarget,
) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(31);
    let (mut units, mut periods, mut dose, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units_total {
        for t in 0..8_i64 {
            let d = 2.0 * stream.next();
            units.push(format!("u{i}"));
            periods.push(t);
            dose.push(d);
            outcome.push(1.0 + d + 0.5 * d * d + (stream.next() - 0.5));
        }
    }
    let rows = units.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Dose {
            dose: dose.as_slice().into(),
            density: vec![0.5; rows].into(),
            bandwidth: 0.3,
            target,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

#[test]
fn the_dose_targets_recover_the_curve_with_a_bias_aware_band() {
    let ctx = ExecutionContext::for_tests(37);
    let grid: Arc<[f64]> = Arc::from([0.5, 1.0, 1.5]);
    let (data, query) = dose_panel(TemporalDoseTarget::LocalQuadratic {
        grid: Arc::clone(&grid),
        derivative: false,
        curvature_bound: Some(0.0),
    });
    let result = run(&data, query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    for (index, d) in grid.iter().enumerate() {
        within(fit, index, 1.0 + d + 0.5 * d * d, 4.0);
    }
    assert_eq!(fit.band_coordinates.len(), 3);
    assert!(fit.smoothing_bias_bound.is_some());

    let (data, query) =
        dose_panel(TemporalDoseTarget::SmoothedContrast { active: 1.2, control: 0.8 });
    let result = run(&data, query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    // The smoothed curve adds h^2 m''/2 to both doses, which cancels in the contrast.
    let m = |d: f64| 1.0 + d + 0.5 * d * d;
    within(fit, 0, m(1.2) - m(0.8), 4.0);

    let (data, missing) = dose_panel(TemporalDoseTarget::LocalQuadratic {
        grid,
        derivative: false,
        curvature_bound: None,
    });
    let error = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(missing))
        .build()
        .and_then(|study| study.run(&ctx))
        .unwrap_err();
    assert!(error.to_string().contains("curvature_bound"), "{error}");
}

/// A two-arm panel at lag 0 whose outcome is a uniform draw plus `2` under treatment, so every
/// quantile shifts by `2`; the treatment probability alternates between 0.4 and 0.6.
fn quantile_panel_of(units_total: usize, level: f64) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(41);
    let (mut units, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units_total {
        for t in 0..6_i64 {
            let p = if t % 2 == 0 { 0.4 } else { 0.6 };
            let a = stream.next() < p;
            units.push(format!("u{i}"));
            periods.push(t);
            assignment.push(a);
            probability.push(p);
            outcome.push(stream.next() + if a { 2.0 } else { 0.0 });
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: probability.as_slice().into(),
        },
        functional: TemporalDesignFunctional::Quantile { level },
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

#[test]
fn the_quantile_excursion_recovers_a_location_shift_in_both_modes() {
    let (data, query) = quantile_panel_of(200, 0.5);
    let ctx = ExecutionContext::for_tests(43);
    let result = run(&data, query.clone(), InferenceMode::Frequentist, 199, &ctx);
    let fit = section(&result);
    assert!((fit.estimate[0] - 2.0).abs() < 0.1, "{:?}", fit.estimate);
    assert_eq!(fit.replicates.map(|(ok, _)| ok), Some(199));
    let posterior = run(&data, query, bayesian(200), 0, &ctx);
    assert!((section(&posterior).estimate[0] - 2.0).abs() < 0.1);
}

/// Subject trajectories with time-varying confounding and a horizon of three periods.
fn survival_panel_of(subjects: usize) -> (TabularData, TemporalDesignQuery) {
    // Time-varying confounding: L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + e, treatment more likely
    // when L_k is high, hazard expit(-2.5 + 0.8 L_k - 0.7 A_k), light censoring.
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let horizon = 3_u32;
    let mut stream = Stream(47);
    let (mut units, mut periods) = (Vec::new(), Vec::new());
    let (mut events, mut treated, mut censored, mut confounder) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..subjects {
        let mut current = 2.0 * stream.next() - 1.0;
        let mut previous = false;
        for k in 0..horizon {
            if k > 0 {
                current =
                    0.5 * current - 0.4 * f64::from(u8::from(previous)) + (stream.next() - 0.5);
            }
            let a = stream.next() < expit(0.8 * current);
            let c = stream.next() < 0.03;
            let y =
                !c && stream.next() < expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(a)));
            units.push(format!("s{i}"));
            periods.push(i64::from(k));
            events.push(f64::from(u8::from(y)));
            treated.push(f64::from(u8::from(a)));
            censored.push(f64::from(u8::from(c)));
            confounder.push(current);
            previous = a;
            if c || y {
                break;
            }
        }
    }
    let data = table(&[
        ("event", events.as_slice()),
        ("treated", treated.as_slice()),
        ("censored", censored.as_slice()),
        ("confounder", confounder.as_slice()),
    ]);
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::RegimeSurvival {
            treatment: VariableId::from_raw(1),
            censored: VariableId::from_raw(2),
            confounders: Arc::from([VariableId::from_raw(3)]),
            horizon,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.02,
        outcome_misclassification: None,
    };
    (data, query)
}

/// Always-treat minus never-treat survival at the horizon of [`survival_panel_of`], by Monte
/// Carlo under each static regime.
fn survival_truth() -> f64 {
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let horizon = 3_u32;
    let mut truth_stream = Stream(53);
    let mut totals = [0.0; 2];
    let draws = 200_000;
    for (index, arm) in [true, false].into_iter().enumerate() {
        for _ in 0..draws {
            let mut current = 2.0 * truth_stream.next() - 1.0;
            let mut surviving = 1.0;
            for k in 0..horizon {
                if k > 0 {
                    current = 0.5 * current - 0.4 * f64::from(u8::from(arm))
                        + (truth_stream.next() - 0.5);
                }
                surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(arm)));
            }
            totals[index] += surviving;
        }
    }
    (totals[0] - totals[1]) / f64::from(draws)
}

#[test]
fn regime_survival_recovers_the_contrast_with_both_procedures() {
    let (data, query) = survival_panel_of(1200);
    let truth = survival_truth();
    let ctx = ExecutionContext::for_tests(59);
    let frequentist = run(&data, query.clone(), InferenceMode::Frequentist, 99, &ctx);
    assert_eq!(
        frequentist.logical_plan.estimator.as_deref(),
        Some("temporal_design.regime_survival")
    );
    let fit = section(&frequentist);
    assert_eq!(&*fit.design, "regime_survival");
    assert!((fit.estimate[0] - truth).abs() < 4.0 * fit.standard_error[0] + 0.01);
    let posterior = run(&data, query, bayesian(200), 0, &ctx);
    let draws_fit = section(&posterior);
    assert!(
        (draws_fit.estimate[0] - truth).abs() < 4.0 * draws_fit.standard_error[0] + 0.01,
        "{} vs {truth}",
        draws_fit.estimate[0]
    );
}

#[test]
fn a_misclassified_binary_outcome_is_corrected_with_its_validation_uncertainty() {
    // True binary outcome with P(Y = 1) = 0.3 + 0.25 A; recorded with se = 0.9, sp = 0.85.
    let mut stream = Stream(61);
    let (mut units, mut periods, mut assignment, mut recorded) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..400 {
        for t in 0..6_i64 {
            let a = stream.next() < 0.5;
            let truth = stream.next() < 0.3 + if a { 0.25 } else { 0.0 };
            let observed = if truth { stream.next() < 0.9 } else { stream.next() >= 0.85 };
            units.push(format!("u{i}"));
            periods.push(t);
            assignment.push(a);
            recorded.push(f64::from(u8::from(observed)));
        }
    }
    let rows = units.len();
    let counts = antecedent_core::OutcomeMisclassification {
        true_positive: 900,
        false_negative: 100,
        true_negative: 850,
        false_positive: 150,
    };
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: vec![0.5; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: Some(counts),
    };
    let data = table(&[("y", recorded.as_slice())]);
    let ctx = ExecutionContext::for_tests(67);
    let result = run(&data, query.clone(), InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    assert!(fit.functional.contains("misclassification"));
    within(fit, 0, 0.25, 4.0);
    within(fit, 2, 0.3, 4.0);
    let posterior = run(&data, query, bayesian(300), 0, &ctx);
    // The uncorrected recorded contrast is 0.75 * 0.25 = 0.1875 in expectation (0.0625
    // below truth); the corrected posterior must sit well inside that gap.
    assert!((section(&posterior).estimate[0] - 0.25).abs() < 0.045);
    // The corrected point is the inverse-probability-weighted recorded contrast (known
    // probability 0.5) divided by se + sp - 1 = 0.75.
    let weighted: f64 =
        (0..rows).map(|i| recorded[i] * if assignment[i] { 1.0 } else { -1.0 } / 0.5).sum::<f64>()
            / rows as f64;
    let corrected = weighted / 0.75;
    assert!((fit.estimate[0] - corrected).abs() < 0.01, "{} vs {corrected}", fit.estimate[0]);
}

#[test]
fn the_exported_artifact_carries_the_section_and_a_fresh_consumer_rejects_forgeries() {
    let panel = two_arm_panel(120, 6, 0.9, 71);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let ctx = ExecutionContext::for_tests(73);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(two_arm_query(&panel, TemporalDesignFunctional::Mean)))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = section(&result).clone();
    let bytes = prepared.encode_contracted_result(&result, "temporal-design", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.temporal_design.as_ref().expect("the section is exported");
    assert!(!wire.supported && wire.calibration == "not_measured");
    assert_eq!(wire.design, "two_arm");
    assert_eq!(wire.estimate, fit.estimate.to_vec());
    assert_eq!(wire.lower, fit.lower.to_vec());
    // The untampered body re-encodes; the forgeries below are refused.
    antecedent_io::encode_analysis_result_artifact(
        &body,
        header.variable_names.clone(),
        "temporal-design-again",
    )
    .unwrap();
    let mut supported = body.clone();
    supported.temporal_design.as_mut().unwrap().supported = true;
    let error = antecedent_io::encode_analysis_result_artifact(
        &supported,
        header.variable_names.clone(),
        "forged-support",
    )
    .unwrap_err();
    assert!(error.to_string().contains("claims no support"), "{error}");
    let mut reversed = body.clone();
    let section = reversed.temporal_design.as_mut().unwrap();
    section.lower[0] = section.upper[0] + 1.0;
    let error = antecedent_io::encode_analysis_result_artifact(
        &reversed,
        header.variable_names.clone(),
        "forged-interval",
    )
    .unwrap_err();
    assert!(error.to_string().contains("interval is reversed"), "{error}");
    let mut stripped = body;
    stripped.temporal_design = None;
    let error = antecedent_io::encode_analysis_result_artifact(
        &stripped,
        header.variable_names,
        "missing-section",
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing its section"), "{error}");
}

/// The two-arm support row: a panel that meets its thresholds states `supported` on the retained
/// program's section and in the exported artifact; an artifact that claims the row below a
/// threshold or under another method is refused, and a thinner panel runs unsupported.
#[test]
fn supported_two_arm_excursion_interval_round_trips_and_refuses_forgery() {
    let panel = two_arm_panel(300, 10, 1.25, 7);
    let data = table(&[("y", panel.outcome.as_slice())]);
    let ctx = ExecutionContext::for_tests(11);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(two_arm_query(&panel, TemporalDesignFunctional::Mean)))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "supported_two_arm_excursion_interval",
        &prepared,
        &result,
        &ctx,
    );
    let fit = section(&result).clone();
    assert!(result.states_section_row());
    assert_eq!(&*fit.uncertainty, "cluster_robust_ratio_normal");
    assert_eq!((fit.clusters, fit.pairs, fit.unpaired), (300, 2700, 300));
    assert!(fit.minimum_probability >= 0.35 - 1e-12, "{}", fit.minimum_probability);
    within(&fit, 0, 1.25, 4.0);
    // Every coordinate publishes the normal interval of its own standard error.
    for index in 0..fit.coordinates.len() {
        let half = 1.959_963_984_540_054 * fit.standard_error[index];
        assert!((fit.lower[index] - (fit.estimate[index] - half)).abs() < 1e-9);
        assert!((fit.upper[index] - (fit.estimate[index] + half)).abs() < 1e-9);
    }

    let bytes =
        prepared.encode_contracted_result(&result, "temporal-design-supported", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.temporal_design.as_ref().expect("the section is exported");
    // The row is stated in a coded diagnostic; the section's own flag stays false.
    assert!(body.diagnostics.iter().any(|diagnostic| diagnostic.code == DESIGN_ROW));
    assert!(!wire.supported && wire.calibration == "not_measured");
    assert_eq!(wire.lower, fit.lower.to_vec());
    assert_eq!(wire.upper, fit.upper.to_vec());
    let refused = |forged: &antecedent_io::analysis_result_artifact::AnalysisResultWire| {
        antecedent_io::encode_analysis_result_artifact(
            forged,
            header.variable_names.clone(),
            "forged-support",
        )
        .unwrap_err()
        .to_string()
    };
    let mut thin = body.clone();
    thin.temporal_design.as_mut().unwrap().clusters = 149;
    assert!(refused(&thin).contains(ROW_NOT_MET));
    let mut weak = body.clone();
    weak.temporal_design.as_mut().unwrap().minimum_probability = Some(0.3);
    assert!(refused(&weak).contains(ROW_NOT_MET));
    let mut other = body.clone();
    other.temporal_design.as_mut().unwrap().uncertainty = "another_method".to_owned();
    assert!(refused(&other).contains(ROW_NOT_MET));
    // The section's own flag is never a statement: a body that sets it is refused.
    let mut flagged = body.clone();
    flagged.temporal_design.as_mut().unwrap().supported = true;
    assert!(refused(&flagged).contains("claims no support"));
    // A result may state less than its row grants; it then loads as an unsupported result.
    let mut unclaimed = body;
    unclaimed.diagnostics.retain(|diagnostic| diagnostic.code != DESIGN_ROW);
    antecedent_io::encode_analysis_result_artifact(
        &unclaimed,
        header.variable_names.clone(),
        "unclaimed-support",
    )
    .unwrap();

    // 120 units are below the row: the same route runs and states no support.
    let small = two_arm_panel(120, 10, 1.25, 7);
    let small_data = table(&[("y", small.outcome.as_slice())]);
    let unsupported = run(
        &small_data,
        two_arm_query(&small, TemporalDesignFunctional::Mean),
        InferenceMode::Frequentist,
        0,
        &ctx,
    );
    assert!(!unsupported.states_section_row());
}

/// A decision-time covariate drives both the treatment and the outcome, and no assignment
/// probability is recorded: the excursion effect on the same-period outcome is `effect`.
fn observational_panel(
    units: usize,
    effect: f64,
    seed: u64,
) -> (TabularData, TemporalDesignQuery, f64) {
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut stream = Stream(seed);
    let (mut labels_of_rows, mut periods) = (Vec::new(), Vec::new());
    let (mut outcome, mut treated, mut covariate) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..4_i64 {
            let x = 2.0 * stream.next() - 1.0;
            let a = stream.next() < expit(1.2 * x);
            let y = effect * f64::from(u8::from(a)) + 3.0 * x + (stream.next() - 0.5);
            labels_of_rows.push(format!("u{i}"));
            periods.push(t);
            outcome.push(y);
            treated.push(f64::from(u8::from(a)));
            covariate.push(x);
        }
    }
    let (mut sums, mut counts) = ([0.0_f64; 2], [0.0_f64; 2]);
    for (y, a) in outcome.iter().zip(&treated) {
        let arm = usize::from(*a > 0.5);
        sums[arm] += y;
        counts[arm] += 1.0;
    }
    let naive = sums[1] / counts[1] - sums[0] / counts[0];
    let data =
        table(&[("y", outcome.as_slice()), ("a", treated.as_slice()), ("x", covariate.as_slice())]);
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&labels_of_rows),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::ObservationalExcursion {
            treatment: VariableId::from_raw(1),
            history: Arc::from([VariableId::from_raw(2)]),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (data, query, naive)
}

#[test]
fn the_observational_excursion_fits_its_assignment_law_and_removes_confounding() {
    let (data, query, naive) = observational_panel(400, 1.0, 83);
    let ctx = ExecutionContext::for_tests(89);
    let frequentist = run(&data, query.clone(), InferenceMode::Frequentist, 99, &ctx);
    let fit = section(&frequentist);
    assert_eq!(&*fit.design, "observational_excursion");
    assert_eq!(&*fit.calibration, "not_measured");
    assert!(!fit.supported && !frequentist.states_section_row());
    assert!(fit.replicates.is_some_and(|(done, failed)| done + failed == 99));
    assert!(
        (fit.estimate[0] - 1.0).abs() < 4.0 * fit.standard_error[0] + 0.05,
        "{} (se {}) vs 1.0",
        fit.estimate[0],
        fit.standard_error[0]
    );
    // The raw contrast of treated and control means is confounded well above the effect.
    assert!(naive > 1.6, "raw contrast {naive}");
    let posterior = run(&data, query, bayesian(200), 0, &ctx);
    let draws_fit = section(&posterior);
    assert_eq!(draws_fit.draws, Some(200));
    assert!(
        (draws_fit.estimate[0] - 1.0).abs() < 4.0 * draws_fit.standard_error[0] + 0.05,
        "{} vs 1.0",
        draws_fit.estimate[0]
    );
}

#[test]
fn the_observational_excursion_refuses_what_it_cannot_identify_or_resample() {
    let (data, query, _) = observational_panel(60, 1.0, 97);
    let ctx = ExecutionContext::for_tests(101);
    // The interval is a whole-procedure bootstrap: no replicates, no interval.
    let missing = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(query.clone()))
        .build()
        .and_then(|study| study.run(&ctx));
    let message = missing.unwrap_err().to_string();
    assert!(
        message.contains("required_option_missing") && message.contains("Next attempt"),
        "{message}"
    );
    // A treatment column that is not coded 0/1 is refused, not rounded.
    let (zeros, halves) = (vec![0.0; 240], vec![0.5; 240]);
    let wrong =
        table(&[("y", zeros.as_slice()), ("a", halves.as_slice()), ("x", zeros.as_slice())]);
    let refused = Study::tabular(wrong)
        .query(CausalQuery::TemporalDesign(query.clone()))
        .bootstrap_replicates(99)
        .build()
        .and_then(|study| study.run(&ctx));
    let message = refused.unwrap_err().to_string();
    assert!(message.contains("must be coded exactly 0 or 1"), "{message}");
    // The outcome cannot also be a history column.
    let mut bad = query;
    bad.assignment = TemporalAssignment::ObservationalExcursion {
        treatment: VariableId::from_raw(1),
        history: Arc::from([VariableId::from_raw(0)]),
    };
    assert!(bad.validate().unwrap_err().to_string().contains("must be distinct"));
}

/// Two periods with treatment-confounder feedback: the baseline confounder drives the first
/// treatment, the first treatment raises the second confounder, which drives the second
/// treatment, and the outcome reads both treatments and both confounders. From the structural
/// equations (centered baseline confounder and noise) `E[Y^{a0,a1}] = 0.5 + 2.2 a0 + 2 a1`, so
/// always-treat against never-treat is `4.2` whatever the estimator.
fn regime_panel(
    units: usize,
    method: antecedent_core::TemporalRegimeMethod,
    seed: u64,
) -> (TabularData, TemporalDesignQuery, f64) {
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut stream = Stream(seed);
    let (mut labels_of_rows, mut periods) = (Vec::new(), Vec::new());
    let (mut outcome, mut treated, mut confounder) = (Vec::new(), Vec::new(), Vec::new());
    let (mut always, mut never) = ([0.0_f64; 2], [0.0_f64; 2]);
    for i in 0..units {
        let l0 = 2.0 * stream.next() - 1.0;
        let a0 = stream.next() < expit(1.5 * l0);
        let l1 = 0.6 * l0 + 0.8 * f64::from(u8::from(a0)) + (stream.next() - 0.5);
        let a1 = stream.next() < expit(1.2 * (l1 - 0.4));
        let y = 0.5
            + f64::from(u8::from(a0))
            + 2.0 * f64::from(u8::from(a1))
            + 1.5 * l1
            + 2.0 * l0
            + 0.6 * (stream.next() - 0.5);
        if a0 && a1 {
            always = [always[0] + y, always[1] + 1.0];
        } else if !a0 && !a1 {
            never = [never[0] + y, never[1] + 1.0];
        }
        for (t, (value, action, level)) in [(0.0, a0, l0), (y, a1, l1)].into_iter().enumerate() {
            labels_of_rows.push(format!("u{i}"));
            periods.push(i64::try_from(t).unwrap());
            outcome.push(value);
            treated.push(f64::from(u8::from(action)));
            confounder.push(level);
        }
    }
    let naive = always[0] / always[1] - never[0] / never[1];
    let data = table(&[
        ("y", outcome.as_slice()),
        ("a", treated.as_slice()),
        ("l", confounder.as_slice()),
    ]);
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&labels_of_rows),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::ObservationalRegime {
            treatment: VariableId::from_raw(1),
            confounders: Arc::from([VariableId::from_raw(2)]),
            active: Arc::from([true, true]),
            reference: Arc::from([false, false]),
            method,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (data, query, naive)
}

#[test]
fn the_observational_regime_removes_time_varying_confounding_under_feedback() {
    use antecedent_core::TemporalRegimeMethod;
    let ctx = ExecutionContext::for_tests(113);
    for method in [TemporalRegimeMethod::DoublyRobust, TemporalRegimeMethod::GFormula] {
        let (data, query, naive) = regime_panel(1500, method, 127);
        let frequentist = run(&data, query.clone(), InferenceMode::Frequentist, 99, &ctx);
        let fit = section(&frequentist);
        assert_eq!(&*fit.design, "observational_regime");
        assert_eq!(&*fit.calibration, "not_measured");
        assert!(!fit.supported && !frequentist.states_section_row());
        assert!(fit.replicates.is_some_and(|(done, failed)| done + failed == 99));
        assert!(
            (fit.estimate[0] - 4.2).abs() < 4.0 * fit.standard_error[0] + 0.1,
            "{method:?}: {} (se {}) vs 4.2",
            fit.estimate[0],
            fit.standard_error[0]
        );
        assert!(
            (fit.estimate[1] - 4.7).abs() < 4.0 * fit.standard_error[1] + 0.2,
            "{method:?}: active mean {}",
            fit.estimate[1]
        );
        // The raw contrast of the always- and never-treated subjects is confounded.
        assert!((naive - 4.2).abs() > 0.5, "raw contrast {naive}");
        let posterior = run(&data, query, bayesian(100), 0, &ctx);
        let draws_fit = section(&posterior);
        assert_eq!(draws_fit.draws, Some(100));
        assert!(
            (draws_fit.estimate[0] - 4.2).abs() < 4.0 * draws_fit.standard_error[0] + 0.2,
            "{method:?}: posterior {} vs 4.2",
            draws_fit.estimate[0]
        );
    }
}

#[test]
fn the_observational_regime_refuses_what_it_cannot_identify_or_resample() {
    use antecedent_core::TemporalRegimeMethod;
    let (data, query, _) = regime_panel(80, TemporalRegimeMethod::DoublyRobust, 131);
    let ctx = ExecutionContext::for_tests(137);
    // No replicates, no interval.
    let missing = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(query.clone()))
        .build()
        .and_then(|study| study.run(&ctx));
    let message = missing.unwrap_err().to_string();
    assert!(
        message.contains("required_option_missing") && message.contains("Next attempt"),
        "{message}"
    );
    // Identical paths, paths of different length, a lag and a repeated column are invalid.
    let mut bad = query.clone();
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(2)]),
        active: Arc::from([true, true]),
        reference: Arc::from([true, true]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().unwrap_err().to_string().contains("differ in some period"));
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(2)]),
        active: Arc::from([true]),
        reference: Arc::from([true, false]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().unwrap_err().to_string().contains("differ in some period"));
    let mut lagged = query.clone();
    lagged.lag = 1;
    assert!(lagged.validate().unwrap_err().to_string().contains("(lag 0)"));
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(0)]),
        active: Arc::from([true, true]),
        reference: Arc::from([false, false]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().unwrap_err().to_string().contains("must be distinct"));
    // A path longer than the follow-up is refused at execution.
    let mut long = query;
    long.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(2)]),
        active: Arc::from([true, true, true]),
        reference: Arc::from([false, false, false]),
        method: TemporalRegimeMethod::GFormula,
    };
    let refused = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(long))
        .bootstrap_replicates(99)
        .build()
        .and_then(|study| study.run(&ctx));
    let message = refused.unwrap_err().to_string();
    assert!(message.contains("each regime names one action per period"), "{message}");
}

/// A factorial panel: main effects `1.0` and `0.5` and an interaction `0.4`, the factors drawn
/// independently with probabilities `0.5` and `0.4`.
fn factorial_panel() -> (TabularData, TemporalDesignQuery) {
    factorial_panel_of(300)
}

/// [`factorial_panel`] with `units` units.
fn factorial_panel_of(units_total: usize) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(29);
    let (mut units, mut periods, mut first, mut second, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units_total {
        for t in 0..5_i64 {
            let a = stream.next() < 0.5;
            let b = stream.next() < 0.4;
            let (fa, fb) = (f64::from(u8::from(a)), f64::from(u8::from(b)));
            units.push(format!("u{i}"));
            periods.push(t);
            first.push(a);
            second.push(b);
            outcome.push(1.0 * fa + 0.5 * fb + 0.4 * fa * fb + (stream.next() - 0.5));
        }
    }
    let rows = units.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Factorial {
            first: first.as_slice().into(),
            first_probability: vec![0.5; rows].into(),
            second: second.as_slice().into(),
            second_probability: vec![0.4; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Pairs of units share an interference cluster; the own effect is `1.0` and a treated peer adds
/// `0.6`.
fn interference_panel() -> (TabularData, TemporalDesignQuery) {
    interference_panel_of(300)
}

/// [`interference_panel`] with `clusters` interference clusters.
fn interference_panel_of(clusters_total: usize) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(61);
    let (mut units, mut periods, mut clusters, mut assignment) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..clusters_total {
        for t in 0..4_i64 {
            for member in 0..2 {
                units.push(format!("u{c}_{member}"));
                periods.push(t);
                clusters.push(format!("c{c}"));
                assignment.push(stream.next() < 0.5);
            }
        }
    }
    let rows = units.len();
    let outcome: Vec<f64> = (0..rows)
        .map(|row| {
            let peer = if row % 2 == 0 { row + 1 } else { row - 1 };
            f64::from(u8::from(assignment[row]))
                + 0.6 * f64::from(u8::from(assignment[peer]))
                + (stream.next() - 0.5)
        })
        .collect();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Interference {
            assignment: assignment.as_slice().into(),
            probability: vec![0.5; rows].into(),
            clusters: labels(&clusters),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

#[test]
fn the_bayesian_bootstrap_recovers_the_multi_arm_factorial_and_interference_effects() {
    let ctx = ExecutionContext::for_tests(71);

    let (data, query) = multi_arm(false, false);
    let result = run(&data, query, bayesian(300), 0, &ctx);
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("bayesian.temporal_design.excursion")
    );
    assert!(result.posterior.is_some());
    let fit = section(&result);
    assert_eq!(&*fit.design, "multi_arm");
    assert_eq!(fit.draws, Some(300));
    within(fit, 0, 0.5, 4.0);
    within(fit, 1, 1.5, 4.0);

    let (data, query) = factorial_panel();
    let result = run(&data, query, bayesian(300), 0, &ctx);
    assert!(result.posterior.is_some());
    let fit = section(&result);
    assert_eq!(fit.draws, Some(300));
    // Main effects average over the other factor's levels with its design probability.
    within(fit, 0, 1.0 + 0.5 * 0.4, 4.0);
    within(fit, 1, 0.5 + 0.5 * 0.4, 4.0);
    within(fit, 2, 0.4, 4.0);

    let (data, query) = interference_panel();
    let result = run(&data, query, bayesian(300), 0, &ctx);
    assert!(result.posterior.is_some());
    let fit = section(&result);
    assert_eq!(&*fit.design, "interference");
    assert_eq!(fit.draws, Some(300));
    within(fit, 0, 1.0, 4.0);
    within(fit, 1, 0.6, 4.0);
}

#[test]
fn the_dose_targets_recover_the_curve_under_the_bayesian_bootstrap_and_the_smoothed_curve() {
    let ctx = ExecutionContext::for_tests(73);
    let grid: Arc<[f64]> = Arc::from([0.5, 1.0, 1.5]);
    let m = |d: f64| 1.0 + d + 0.5 * d * d;

    // The target is the Gaussian-kernel smoothing of the response over the dose support (0, 2)
    // at bandwidth 0.3, integrated here by the midpoint rule; kernel mass beyond the support is
    // not renormalized, so the curve is pulled down near the upper edge.
    let smoothed = |d: f64| {
        let steps = 20_000;
        let width = 2.0 / f64::from(steps);
        (0..steps)
            .map(|i| {
                let u = (f64::from(i) + 0.5) * width;
                let z = (d - u) / 0.3;
                (-0.5 * z * z).exp() / (0.3 * (2.0 * std::f64::consts::PI).sqrt()) * m(u) * width
            })
            .sum::<f64>()
    };
    let (data, query) = dose_panel(TemporalDoseTarget::SmoothedCurve { grid: Arc::clone(&grid) });
    let frequentist = run(&data, query.clone(), InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&frequentist);
    for (index, d) in grid.iter().enumerate() {
        let truth = smoothed(*d);
        assert!(
            (fit.estimate[index] - truth).abs() <= 4.0 * fit.standard_error[index] + 0.02,
            "smoothed curve at {d}: {} (se {}) vs {truth}",
            fit.estimate[index],
            fit.standard_error[index]
        );
    }
    assert_eq!(fit.band_coordinates.len(), 3);
    let posterior = run(&data, query, bayesian(300), 0, &ctx);
    assert!(posterior.posterior.is_some());
    let draws = section(&posterior);
    assert_eq!(draws.draws, Some(300));
    for (index, d) in grid.iter().enumerate() {
        assert!(
            (draws.estimate[index] - fit.estimate[index]).abs() <= 3.0 * fit.standard_error[index],
            "posterior vs point at {d}"
        );
    }

    let (data, query) = dose_panel(TemporalDoseTarget::LocalQuadratic {
        grid: Arc::clone(&grid),
        derivative: false,
        curvature_bound: Some(0.0),
    });
    let posterior = run(&data, query, bayesian(300), 0, &ctx);
    assert!(posterior.posterior.is_some());
    let fit = section(&posterior);
    assert_eq!(fit.draws, Some(300));
    for (index, d) in grid.iter().enumerate() {
        within(fit, index, m(*d), 4.0);
    }

    // The contrast of the smoothed response cancels the shared smoothing bias.
    let (data, query) =
        dose_panel(TemporalDoseTarget::SmoothedContrast { active: 1.2, control: 0.8 });
    let posterior = run(&data, query, bayesian(300), 0, &ctx);
    assert!(posterior.posterior.is_some());
    let fit = section(&posterior);
    assert_eq!(fit.draws, Some(300));
    within(fit, 0, m(1.2) - m(0.8), 4.0);
}

// ------------------------------------------------------------ support rows

/// The kernel-smoothed response `int_0^2 K_0.3(u - at) m(u) du` of [`dose_panel`] by Simpson's
/// rule: the target of the smoothed dose designs under the uniform dose law.
fn smoothed_dose_truth(at: f64) -> f64 {
    let integrand = |u: f64| {
        let z = (u - at) / 0.3;
        (-0.5 * z * z).exp() / (0.3 * (2.0 * std::f64::consts::PI).sqrt()) * (1.0 + u + 0.5 * u * u)
    };
    let steps = 4000_u32;
    let h = 2.0 / f64::from(steps);
    let mut sum = integrand(0.0) + integrand(2.0);
    for k in 1..steps {
        sum += integrand(h * f64::from(k)) * if k % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// The panel of a design of the support table with `units` independent sequences.
fn support_row_panel(design: &str, units: usize) -> (TabularData, TemporalDesignQuery) {
    let grid: Arc<[f64]> = Arc::from([0.5, 1.0, 1.5]);
    match design {
        "two_arm" => {
            let panel = two_arm_panel(units, 10, 1.25, 7);
            let query = two_arm_query(&panel, TemporalDesignFunctional::Mean);
            (table(&[("y", panel.outcome.as_slice())]), query)
        }
        "multi_arm" => multi_arm_of(units, false, false),
        "categorical" => multi_arm_of(units, true, true),
        "factorial" => factorial_panel_of(units),
        "interference" => interference_panel_of(units),
        "continuous_dose" => {
            dose_panel_of(units, TemporalDoseTarget::SmoothedContrast { active: 1.2, control: 0.8 })
        }
        "smoothed_dose" => dose_panel_of(units, TemporalDoseTarget::SmoothedCurve { grid }),
        "dose_grid" => dose_panel_of(
            units,
            TemporalDoseTarget::LocalQuadratic {
                grid,
                derivative: false,
                curvature_bound: Some(0.0),
            },
        ),
        "quantile" => quantile_panel_of(units, 0.5),
        "regime_survival" => survival_panel_of(units),
        "two_arm_misclassified" => misclassified_panel(units, [90, 10, 85, 15]),
        other => panic!("no support row for design {other}"),
    }
}

/// What one support row requires and the coordinate its coverage was measured on.
struct SupportRowCase {
    design: &'static str,
    posterior: bool,
    method: &'static str,
    /// Independent sequences of the supported panel and of a panel below the row.
    units: (usize, usize),
    /// The row's fewest independent sequences.
    min_blocks: u64,
    /// The row's smallest realized probability, where the section carries one.
    min_probability: Option<f64>,
    /// Index, name and known value of the scored coordinate.
    coordinate: (usize, &'static str, f64),
}

impl SupportRowCase {
    /// Posterior draws the row was measured with.
    fn draws(&self) -> usize {
        if matches!(self.design, "dose_grid" | "regime_survival") { 500 } else { 1000 }
    }

    /// Whole-procedure bootstrap replicates the row was measured with.
    fn replicates(&self) -> u32 {
        if !self.posterior && matches!(self.design, "quantile" | "regime_survival") {
            99
        } else {
            0
        }
    }

    fn inference(&self, draws: usize) -> InferenceMode {
        if self.posterior { bayesian(draws) } else { InferenceMode::Frequentist }
    }
}

/// One support row of the family: a panel that meets the row states `supported` on the retained
/// program's section and in the exported artifact; an artifact that claims the row with fewer
/// sequences, a smaller probability, fewer draws or replicates, or another method is refused;
/// and the same route below the row runs and states no support.
fn assert_support_row_round_trips_and_refuses_forgery(label: &'static str, case: &SupportRowCase) {
    type Forgery =
        fn(&mut antecedent_io::analysis_result_artifact::TemporalDesignWire, &SupportRowCase);
    let ctx = ExecutionContext::for_tests(11);
    let (data, query) = support_row_panel(case.design, case.units.0);
    let mut builder = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(query.clone()))
        .inference(case.inference(case.draws()));
    if case.replicates() > 0 {
        builder = builder.bootstrap_replicates(case.replicates());
    }
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let fit = section(&result).clone();
    assert!(result.states_section_row(), "{label}");
    assert_eq!(&*fit.uncertainty, case.method, "{label}");
    assert_eq!(fit.clusters, case.units.0 as u64, "{label}");
    assert_eq!(fit.draws, case.posterior.then(|| case.draws() as u64), "{label}");
    if case.replicates() > 0 {
        assert_eq!(fit.replicates, Some((99, 0)), "{label}");
    }
    // The scored coordinate sits at its known value within its own reported uncertainty.
    let (index, name, truth) = case.coordinate;
    assert_eq!(&*fit.coordinates[index], name, "{label}");
    let (estimate, scale) = (fit.estimate[index], fit.standard_error[index]);
    assert!((estimate - truth).abs() < 5.0 * scale + 0.01, "{label}: {estimate} vs {truth}");
    assert!(fit.lower[index] < estimate && estimate < fit.upper[index], "{label}");
    if !case.posterior && case.replicates() == 0 {
        // A normal interval of the coordinate's own standard error.
        let half = 1.959_963_984_540_054 * scale;
        assert!((fit.lower[index] - (estimate - half)).abs() < 1e-5 * scale + 1e-9, "{label}");
        assert!((fit.upper[index] - (estimate + half)).abs() < 1e-5 * scale + 1e-9, "{label}");
    }

    let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let wire = body.temporal_design.as_ref().expect("the section is exported");
    assert!(body.diagnostics.iter().any(|diagnostic| diagnostic.code == DESIGN_ROW), "{label}");
    assert!(!wire.supported && wire.calibration == "not_measured", "{label}");
    assert_eq!(wire.lower, fit.lower.to_vec(), "{label}");
    let forgeries: [(&str, Forgery); 5] = [
        ("sequences", |wire, case| wire.clusters = case.min_blocks - 1),
        ("method", |wire, _| "another_method".clone_into(&mut wire.uncertainty)),
        ("probability", |wire, case| {
            if let (Some(_), Some(floor)) = (wire.minimum_probability, case.min_probability) {
                wire.minimum_probability = Some(floor - 0.05);
            } else {
                wire.clusters = 2;
            }
        }),
        ("draws", |wire, case| match wire.draws {
            Some(_) => wire.draws = Some(case.draws() as u64 - 1),
            None => wire.clusters = 2,
        }),
        ("replicates", |wire, _| match wire.replicates {
            Some(_) => wire.replicates = Some([98, 0]),
            None => wire.clusters = 2,
        }),
    ];
    for (what, forge) in forgeries {
        let mut forged = body.clone();
        forge(forged.temporal_design.as_mut().unwrap(), case);
        let error = antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-support",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(ROW_NOT_MET), "{label} ({what}): {error}");
    }
    // A result may state less than its row grants; it then loads as an unsupported result.
    let mut unclaimed = body;
    unclaimed.diagnostics.retain(|diagnostic| diagnostic.code != DESIGN_ROW);
    antecedent_io::encode_analysis_result_artifact(
        &unclaimed,
        header.variable_names.clone(),
        "unclaimed-support",
    )
    .unwrap();

    // Fewer sequences than the row: the same route runs and states no support.
    let (small, small_query) = support_row_panel(case.design, case.units.1);
    let thin = run(&small, small_query, case.inference(case.draws()), case.replicates(), &ctx);
    assert!(!thin.states_section_row(), "{label}: below the row");
    if case.posterior {
        // Fewer posterior draws than the measured posterior: no support either.
        let few = run(&data, query, case.inference(case.draws() / 2), 0, &ctx);
        assert!(!few.states_section_row(), "{label}: fewer draws");
    }
}

const RATIO_NORMAL: &str = "cluster_robust_ratio_normal";
const UNIT_POSTERIOR: &str = "bayesian_bootstrap_over_clusters";

fn arm_row(
    design: &'static str,
    posterior: bool,
    units: (usize, usize, u64),
    min_probability: f64,
    coordinate: (usize, &'static str, f64),
) -> SupportRowCase {
    SupportRowCase {
        design,
        posterior,
        method: if posterior { UNIT_POSTERIOR } else { RATIO_NORMAL },
        units: (units.0, units.1),
        min_blocks: units.2,
        min_probability: Some(min_probability),
        coordinate,
    }
}

fn dose_row(design: &'static str, posterior: bool) -> SupportRowCase {
    let (method, coordinate) = match (design, posterior) {
        ("continuous_dose", _) => (
            if posterior { UNIT_POSTERIOR } else { RATIO_NORMAL },
            (0, "smoothed_contrast:1.2-0.8", smoothed_dose_truth(1.2) - smoothed_dose_truth(0.8)),
        ),
        ("smoothed_dose", _) => (
            if posterior { UNIT_POSTERIOR } else { RATIO_NORMAL },
            (1, "smoothed_response@1", smoothed_dose_truth(1.0)),
        ),
        (_, false) => ("cluster_robust_local_quadratic_bias_aware", (1, "level@1", 2.5)),
        (_, true) => ("bayesian_bootstrap_local_quadratic_projection", (1, "level@1", 2.5)),
    };
    SupportRowCase {
        design,
        posterior,
        method,
        units: (300, 100),
        min_blocks: 150,
        min_probability: None,
        coordinate,
    }
}

fn resampled_row(design: &'static str, posterior: bool) -> SupportRowCase {
    let survival = design == "regime_survival";
    SupportRowCase {
        design,
        posterior,
        method: match (survival, posterior) {
            (false, false) => "whole_procedure_cluster_bootstrap_percentile",
            (false, true) => "bayesian_bootstrap_over_units_weighted_quantile",
            (true, false) => {
                "whole_procedure_subject_bootstrap_refitting_hazard_and_confounder_transition_models"
            }
            (true, true) => "joint_likelihood_parametric_g_formula_posterior",
        },
        units: if survival { (1200, 400) } else { (200, 80) },
        min_blocks: if survival { 600 } else { 100 },
        min_probability: None,
        coordinate: if survival {
            (0, "survival_difference", survival_truth())
        } else {
            (0, "quantile_effect", 2.0)
        },
    }
}

#[test]
fn supported_two_arm_excursion_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_two_arm_excursion_posterior",
        &arm_row("two_arm", true, (300, 120, 150), 0.35, (0, "excursion_effect", 1.25)),
    );
}

#[test]
fn supported_multi_arm_contrast_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_multi_arm_contrast_interval",
        &arm_row("multi_arm", false, (250, 100, 125), 0.2, (0, "contrast:mid-low", 0.5)),
    );
}

#[test]
fn supported_multi_arm_contrast_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_multi_arm_contrast_posterior",
        &arm_row("multi_arm", true, (250, 100, 125), 0.2, (0, "contrast:mid-low", 0.5)),
    );
}

#[test]
fn supported_categorical_step_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_categorical_step_interval",
        &arm_row("categorical", false, (250, 100, 125), 0.2, (2, "step:high-mid", 1.0)),
    );
}

#[test]
fn supported_categorical_step_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_categorical_step_posterior",
        &arm_row("categorical", true, (250, 100, 125), 0.2, (2, "step:high-mid", 1.0)),
    );
}

#[test]
fn supported_factorial_main_effect_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_factorial_main_effect_interval",
        &arm_row("factorial", false, (300, 120, 150), 0.2, (0, "main_effect:first", 1.2)),
    );
}

#[test]
fn supported_factorial_main_effect_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_factorial_main_effect_posterior",
        &arm_row("factorial", true, (300, 120, 150), 0.2, (0, "main_effect:first", 1.2)),
    );
}

#[test]
fn supported_interference_direct_effect_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_interference_direct_effect_interval",
        &arm_row("interference", false, (300, 120, 150), 0.25, (0, "direct_effect", 1.0)),
    );
}

#[test]
fn supported_interference_direct_effect_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_interference_direct_effect_posterior",
        &arm_row("interference", true, (300, 120, 150), 0.25, (0, "direct_effect", 1.0)),
    );
}

#[test]
fn supported_continuous_dose_contrast_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_continuous_dose_contrast_interval",
        &dose_row("continuous_dose", false),
    );
}

#[test]
fn supported_continuous_dose_contrast_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_continuous_dose_contrast_posterior",
        &dose_row("continuous_dose", true),
    );
}

#[test]
fn supported_smoothed_dose_curve_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_smoothed_dose_curve_interval",
        &dose_row("smoothed_dose", false),
    );
}

#[test]
fn supported_smoothed_dose_curve_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_smoothed_dose_curve_posterior",
        &dose_row("smoothed_dose", true),
    );
}

#[test]
fn supported_dose_grid_level_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_dose_grid_level_interval",
        &dose_row("dose_grid", false),
    );
    // A positive curvature bound and a narrow bandwidth are outside the measured row.
    let ctx = ExecutionContext::for_tests(11);
    let grid: Arc<[f64]> = Arc::from([0.5, 1.0, 1.5]);
    let (data, query) = dose_panel_of(
        300,
        TemporalDoseTarget::LocalQuadratic { grid, derivative: false, curvature_bound: Some(0.5) },
    );
    assert!(!run(&data, query, InferenceMode::Frequentist, 0, &ctx).states_section_row());
    let (data, mut narrow) = support_row_panel("dose_grid", 300);
    if let TemporalAssignment::Dose { bandwidth, .. } = &mut narrow.assignment {
        *bandwidth = 0.1;
    }
    let fit = run(&data, narrow, InferenceMode::Frequentist, 0, &ctx);
    assert!((section(&fit).estimate[1] - 2.5).abs() < 0.5);
    assert!(!fit.states_section_row());
}

#[test]
fn supported_dose_grid_level_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_dose_grid_level_posterior",
        &dose_row("dose_grid", true),
    );
}

#[test]
fn supported_quantile_effect_bootstrap_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_quantile_effect_bootstrap_interval",
        &resampled_row("quantile", false),
    );
    // Only the median was measured: another level runs and states no support.
    let ctx = ExecutionContext::for_tests(11);
    let (data, query) = quantile_panel_of(200, 0.25);
    let fit = run(&data, query, InferenceMode::Frequentist, 99, &ctx);
    assert!((section(&fit).estimate[0] - 2.0).abs() < 0.2);
    assert!(!fit.states_section_row());
}

#[test]
fn supported_quantile_effect_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_quantile_effect_posterior",
        &resampled_row("quantile", true),
    );
}

#[test]
fn supported_regime_survival_bootstrap_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_regime_survival_bootstrap_interval",
        &resampled_row("regime_survival", false),
    );
}

#[test]
fn supported_regime_survival_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_regime_survival_posterior",
        &resampled_row("regime_survival", true),
    );
}

/// A two-arm panel of `units` sequences over six periods with a known assignment probability
/// of 0.5 and a binary outcome `P(Y = 1) = 0.3 + 0.25 A` recorded with sensitivity 0.9 and
/// specificity 0.85; `validation` is the validation sample (true positive, false negative,
/// true negative, false positive).
fn misclassified_panel(units: usize, validation: [u64; 4]) -> (TabularData, TemporalDesignQuery) {
    let mut stream = Stream(0x5EED_0000_0000_0031);
    let (mut names, mut periods, mut assignment, mut recorded) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..6_i64 {
            let treated = stream.next() < 0.5;
            let truth = stream.next() < 0.3 + if treated { 0.25 } else { 0.0 };
            let observed = if truth { stream.next() < 0.9 } else { stream.next() >= 0.85 };
            names.push(format!("u{i}"));
            periods.push(t);
            assignment.push(treated);
            recorded.push(f64::from(u8::from(observed)));
        }
    }
    let query = misclassified_query(&names, &periods, &assignment, validation);
    (table(&[("y", recorded.as_slice())]), query)
}

fn misclassified_query(
    names: &[String],
    periods: &[i64],
    assignment: &[bool],
    validation: [u64; 4],
) -> TemporalDesignQuery {
    TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(names),
        periods: periods.into(),
        lag: 0,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.into(),
            probability: vec![0.5; names.len()].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: Some(antecedent_core::OutcomeMisclassification {
            true_positive: validation[0],
            false_negative: validation[1],
            true_negative: validation[2],
            false_positive: validation[3],
        }),
    }
}

fn coordinate_of(fit: &TemporalDesignEstimate, name: &str) -> usize {
    fit.coordinates.iter().position(|c| &**c == name).unwrap_or_else(|| panic!("no {name}"))
}

/// The corrected two-arm means against the inverted misclassification algebra, by hand.
///
/// Frequentist. Sixteen units of two periods at assignment probability 0.5: eight units are
/// treated at both periods (sixteen decisions, nine recorded positives) and eight are control
/// (sixteen decisions, six recorded positives). An arm mean is the Horvitz-Thompson sum of
/// `Y / 0.5` over the arm's decisions divided by all 32 decisions, `2 * 9 / 32 = 9/16` and
/// `2 * 6 / 32 = 3/8`; it is the arm's sample mean only because the arms are balanced.
/// With 90 of 100 true positives and 85 of 100 true negatives recorded correctly, sensitivity
/// 0.9 and specificity 0.85 give the corrected control mean `(0.375 + 0.85 - 1) / 0.75 = 0.3`,
/// the corrected treated mean `(0.5625 - 0.15) / 0.75 = 0.55` and the contrast 0.25.
///
/// Bayesian. Twelve identical units (treated four times with outcomes 1, 1, 0, 0; control four
/// times with outcomes 1, 0, 0, 0; balanced, so the Horvitz-Thompson means `2 * 2 / 8` and
/// `2 * 1 / 8` are the arm sample means): every cluster weighting leaves them at 0.5 and 0.25,
/// so the posterior varies only through the validation draws, and a validation sample of a
/// million per margin at sensitivity and specificity 0.9 pins them. The corrected control mean
/// is `(0.25 - 0.1) / 0.8 = 0.1875` and the contrast `0.25 / 0.8 = 0.3125`.
#[test]
fn the_misclassification_corrected_two_arm_means_equal_the_inverted_error_algebra() {
    let ctx = ExecutionContext::for_tests(41);
    let (mut names, mut periods, mut assignment, mut recorded) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for unit in 0..16 {
        for period in 0..2_i64 {
            let decision = unit * 2 + usize::try_from(period).unwrap();
            let treated = unit < 8;
            // Decisions 0..16 are treated (positives at 0..9); 16..32 control (positives at 16..22).
            let positive = if treated { decision < 9 } else { decision < 22 };
            names.push(format!("u{unit}"));
            periods.push(period);
            assignment.push(treated);
            recorded.push(f64::from(u8::from(positive)));
        }
    }
    let query = misclassified_query(&names, &periods, &assignment, [90, 10, 85, 15]);
    let result =
        run(&table(&[("y", recorded.as_slice())]), query, InferenceMode::Frequentist, 0, &ctx);
    let fit = section(&result);
    assert_eq!(&*fit.uncertainty, "cluster_robust_ratio_normal_with_validation_delta");
    let (effect, control) =
        (coordinate_of(fit, "excursion_effect"), coordinate_of(fit, "control_mean"));
    assert!((fit.estimate[effect] - 0.25).abs() < 1e-9, "{}", fit.estimate[effect]);
    assert!((fit.estimate[control] - 0.3).abs() < 1e-9, "{}", fit.estimate[control]);
    // The uncorrected recorded contrast is 9/16 - 3/8 = 0.1875: the correction is not a no-op.
    assert!((fit.estimate[effect] - 0.1875).abs() > 0.05);

    let (mut names, mut periods, mut assignment, mut recorded) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for unit in 0..12 {
        for (period, (treated, positive)) in [
            (true, true),
            (true, true),
            (true, false),
            (true, false),
            (false, true),
            (false, false),
            (false, false),
            (false, false),
        ]
        .into_iter()
        .enumerate()
        {
            names.push(format!("u{unit}"));
            periods.push(i64::try_from(period).unwrap());
            assignment.push(treated);
            recorded.push(f64::from(u8::from(positive)));
        }
    }
    let query =
        misclassified_query(&names, &periods, &assignment, [900_000, 100_000, 900_000, 100_000]);
    let result = run(&table(&[("y", recorded.as_slice())]), query, bayesian(1000), 0, &ctx);
    let fit = section(&result);
    assert_eq!(&*fit.uncertainty, "bayesian_bootstrap_with_beta_validation_draws");
    for (name, truth) in [("excursion_effect", 0.3125), ("control_mean", 0.1875)] {
        let index = coordinate_of(fit, name);
        assert!((fit.estimate[index] - truth).abs() < 2e-3, "{name}: {}", fit.estimate[index]);
        assert!(fit.lower[index] <= truth && truth <= fit.upper[index], "{name}");
        assert!(fit.upper[index] - fit.lower[index] < 0.01, "{name}");
    }
}

const CORRECTED_NORMAL: &str = "cluster_robust_ratio_normal_with_validation_delta";
const CORRECTED_POSTERIOR: &str = "bayesian_bootstrap_with_beta_validation_draws";

fn corrected_row(posterior: bool) -> SupportRowCase {
    SupportRowCase {
        design: "two_arm_misclassified",
        posterior,
        method: if posterior { CORRECTED_POSTERIOR } else { CORRECTED_NORMAL },
        units: (200, 60),
        min_blocks: 100,
        min_probability: Some(0.5),
        coordinate: (0, "excursion_effect", 0.25),
    }
}

/// The corrected route states support only inside the measured validation conditions: a
/// validation margin below 100 units, or a validated Youden index below 0.6, runs unsupported.
fn assert_validation_conditions_gate_the_corrected_row(inference: &InferenceMode) {
    let ctx = ExecutionContext::for_tests(11);
    for validation in [[89, 10, 85, 15], [90, 10, 84, 15], [70, 30, 70, 30]] {
        let (data, query) = misclassified_panel(200, validation);
        let fit = run(&data, query, inference.clone(), 0, &ctx);
        assert!(!fit.states_section_row(), "{validation:?}");
        let width = section(&fit).upper[0] - section(&fit).lower[0];
        assert!(width > 0.0 && width < 1.0, "{validation:?}: the route still runs");
    }
}

#[test]
fn supported_misclassified_two_arm_interval_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_misclassified_two_arm_interval",
        &corrected_row(false),
    );
    assert_validation_conditions_gate_the_corrected_row(&InferenceMode::Frequentist);
}

#[test]
fn supported_misclassified_two_arm_posterior_round_trips_and_refuses_forgery() {
    assert_support_row_round_trips_and_refuses_forgery(
        "supported_misclassified_two_arm_posterior",
        &corrected_row(true),
    );
    assert_validation_conditions_gate_the_corrected_row(&bayesian(1000));
}
