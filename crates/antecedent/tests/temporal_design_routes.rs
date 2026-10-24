//! Sequentially randomized unit-period designs and regime survival through the public study
//! flow, against simulated truths.
//!
//! Every fixture draws a panel from a known law: the excursion effects, the dose response, the
//! quantile shift and the regime survival contrast are known, and the routes must recover them
//! within their own reported uncertainty. The routes are unlicensed: no support status, a
//! `calibration = "unmeasured"` section, and a refusal for every option they do not read.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "small counts convert exactly")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PrimaryEstimate, Study, TemporalDesignEstimate};
use antecedent_core::{
    CausalQuery, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, VariableId,
};
use antecedent_data::TabularData;

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
    assert!(matches!(frequentist.estimate, PrimaryEstimate::NotAnEffect));
    assert_eq!(frequentist.support_status, None);
    let fit = section(&frequentist);
    assert_eq!(&*fit.design, "two_arm");
    assert!(!fit.licensed && &*fit.calibration == "unmeasured");
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
    let mut stream = Stream(21);
    let (mut units, mut periods, mut assignment, mut probabilities, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let effects = [0.0, 0.5, 1.5];
    for i in 0..250 {
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
    assert!((section(&posterior).estimate[1] - 1.5).abs() < 0.3);
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
    let mut stream = Stream(31);
    let (mut units, mut periods, mut dose, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..300 {
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

#[test]
fn the_quantile_excursion_recovers_a_location_shift_in_both_modes() {
    let mut stream = Stream(41);
    let (mut units, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..200 {
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
        functional: TemporalDesignFunctional::Quantile { level: 0.5 },
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    let data = table(&[("y", outcome.as_slice())]);
    let ctx = ExecutionContext::for_tests(43);
    let result = run(&data, query.clone(), InferenceMode::Frequentist, 199, &ctx);
    let fit = section(&result);
    assert!((fit.estimate[0] - 2.0).abs() < 0.1, "{:?}", fit.estimate);
    assert_eq!(fit.replicates.map(|(ok, _)| ok), Some(199));
    let posterior = run(&data, query, bayesian(200), 0, &ctx);
    assert!((section(&posterior).estimate[0] - 2.0).abs() < 0.1);
}

#[test]
fn regime_survival_recovers_the_contrast_with_both_procedures() {
    // Time-varying confounding: L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + e, treatment more likely
    // when L_k is high, hazard expit(-2.5 + 0.8 L_k - 0.7 A_k), light censoring.
    let expit = |x: f64| 1.0 / (1.0 + (-x).exp());
    let horizon = 3_u32;
    let mut stream = Stream(47);
    let (mut units, mut periods) = (Vec::new(), Vec::new());
    let (mut events, mut treated, mut censored, mut confounder) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..1200 {
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
    // Monte Carlo truth under each static regime.
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
    let truth = (totals[0] - totals[1]) / f64::from(draws);
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
    assert!((section(&posterior).estimate[0] - 0.25).abs() < 0.08);
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
    assert!(!wire.licensed && wire.calibration == "unmeasured");
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
    let mut licensed = body.clone();
    licensed.temporal_design.as_mut().unwrap().licensed = true;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &licensed,
            header.variable_names.clone(),
            "forged-license"
        )
        .is_err()
    );
    let mut reversed = body.clone();
    let section = reversed.temporal_design.as_mut().unwrap();
    section.lower[0] = section.upper[0] + 1.0;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &reversed,
            header.variable_names.clone(),
            "forged-interval"
        )
        .is_err()
    );
    let mut stripped = body;
    stripped.temporal_design = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &stripped,
            header.variable_names,
            "missing-section"
        )
        .is_err()
    );
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
    assert_eq!(&*fit.calibration, "unmeasured");
    assert!(!fit.licensed);
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
    assert!(missing.is_err());
    // A treatment column that is not coded 0/1 is refused, not rounded.
    let (zeros, halves) = (vec![0.0; 240], vec![0.5; 240]);
    let wrong =
        table(&[("y", zeros.as_slice()), ("a", halves.as_slice()), ("x", zeros.as_slice())]);
    let refused = Study::tabular(wrong)
        .query(CausalQuery::TemporalDesign(query.clone()))
        .bootstrap_replicates(99)
        .build()
        .and_then(|study| study.run(&ctx));
    assert!(refused.is_err());
    // The outcome cannot also be a history column.
    let mut bad = query;
    bad.assignment = TemporalAssignment::ObservationalExcursion {
        treatment: VariableId::from_raw(1),
        history: Arc::from([VariableId::from_raw(0)]),
    };
    assert!(bad.validate().is_err());
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
        for (t, (value, action, level)) in
            [(0.0, a0, l0), (y, a1, l1)].into_iter().enumerate()
        {
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
        assert_eq!(&*fit.calibration, "unmeasured");
        assert!(!fit.licensed);
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
    assert!(missing.is_err());
    // Identical paths, paths of different length, a lag and a repeated column are invalid.
    let mut bad = query.clone();
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(2)]),
        active: Arc::from([true, true]),
        reference: Arc::from([true, true]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().is_err());
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(2)]),
        active: Arc::from([true]),
        reference: Arc::from([true, false]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().is_err());
    let mut lagged = query.clone();
    lagged.lag = 1;
    assert!(lagged.validate().is_err());
    bad.assignment = TemporalAssignment::ObservationalRegime {
        treatment: VariableId::from_raw(1),
        confounders: Arc::from([VariableId::from_raw(0)]),
        active: Arc::from([true, true]),
        reference: Arc::from([false, false]),
        method: TemporalRegimeMethod::GFormula,
    };
    assert!(bad.validate().is_err());
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
    assert!(refused.is_err());
}
