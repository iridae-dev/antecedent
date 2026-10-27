//! Instrumental variables, observational sequential contrasts, history matching and regime
//! survival at horizon one.
//!
//! Each test builds one panel with one decision per unit (or lag-aligned rows of one series),
//! computes the reference from the raw rows, and compares the temporal route with its static
//! twin on what they share: the point, the selected rows, the nuisance structure. Where the
//! temporal family is not a specialization of the static one, the difference is pinned with its
//! cause.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent::{IdentifierId, PrimaryEstimate};
use antecedent_core::{
    HistoryMatchQuery, ObservationAssumption, SequentialContrastQuery, SurvivalFunctional,
    SurvivalQuery, TargetPopulation,
};
use antecedent_estimate::TwoStageLeastSquares;

// ---------------------------------------------------------------------------
// Instrumental variable: lagged instrument against the static 2SLS.
// ---------------------------------------------------------------------------

const IV_ROWS: usize = 800;

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// `x_s = 0.9 z_{s-1} + 0.8 u_s + e`, `y_t = 2 x_{t-1} + u_{t-1} + e`; `z` and `u` iid.
fn iv_series() -> [Vec<f64>; 4] {
    let mut rng = Lcg(31);
    let z: Vec<f64> = (0..IV_ROWS).map(|_| rng.normal()).collect();
    let u: Vec<f64> = (0..IV_ROWS).map(|_| rng.normal()).collect();
    let mut x = vec![0.0; IV_ROWS];
    let mut y = vec![0.0; IV_ROWS];
    for s in 0..IV_ROWS {
        let lagged = if s > 0 { z[s - 1] } else { 0.0 };
        x[s] = 0.9 * lagged + 0.8 * u[s] + 0.3 * rng.normal();
    }
    for t in 1..IV_ROWS {
        y[t] = 2.0 * x[t - 1] + u[t - 1] + 0.3 * rng.normal();
    }
    [x, y, z, u]
}

/// `z_{t-2} -> x_{t-1} -> y_t` with the confounder `u_{t-1}` of treatment and outcome.
fn iv_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let mut place = |variable: u32, lag: u32| {
        ensure_lagged(&mut graph, id(variable), Lag::from_raw(lag)).unwrap()
    };
    let (z2, x1, y0, u1) = (place(2, 2), place(0, 1), place(1, 0), place(3, 1));
    graph.insert_directed(z2, x1).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(u1, x1).unwrap();
    graph.insert_directed(u1, y0).unwrap();
    graph
}

/// outcome: equivalent point and relevance diagnostic; `not_equivalent` uncertainty product.
///
/// A lagged instrument on one series is the just-identified Wald ratio on the rows
/// `(z_{i-1}, x_i, y_{i+1})`. The static continuous-instrument route (2SLS) on those rows
/// reproduces the ratio and the first-stage F to rounding. The two routes then publish different
/// uncertainty: the temporal fit a homoskedastic Wald standard error that ignores serial
/// dependence, the static fit no standard error at all (the weak-instrument-robust
/// Anderson-Rubin set instead).
#[test]
fn horizon_one_lagged_instrument_reproduces_the_static_two_stage_ratio_with_a_different_uncertainty_product()
 {
    let columns = iv_series();
    let rows = IV_ROWS - 2;
    let pick =
        |c: &[f64], offset: usize| -> Vec<f64> { (0..rows).map(|k| c[k + offset]).collect() };
    let (zs, xs, ys) = (pick(&columns[2], 0), pick(&columns[0], 1), pick(&columns[1], 2));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (zm, xm, ym) = (mean(&zs), mean(&xs), mean(&ys));
    let szz: f64 = zs.iter().map(|z| (z - zm).powi(2)).sum();
    let szx: f64 = zs.iter().zip(&xs).map(|(z, x)| (z - zm) * (x - xm)).sum();
    let szy: f64 = zs.iter().zip(&ys).map(|(z, y)| (z - zm) * (y - ym)).sum();
    let ratio = szy / szx;
    let df = (rows - 2) as f64;
    let slope = szx / szz;
    let first_stage_rss: f64 =
        zs.iter().zip(&xs).map(|(z, x)| (x - xm - slope * (z - zm)).powi(2)).sum();
    let structural_rss: f64 =
        xs.iter().zip(&ys).map(|(x, y)| (y - (ym - ratio * xm) - ratio * x).powi(2)).sum();
    let wald_se = (structural_rss / df * szz / (szx * szx)).sqrt();
    let first_stage_f = slope * slope * szz / (first_stage_rss / df);
    assert!(first_stage_f > 10.0, "the fixture needs a strong instrument: {first_stage_f}");

    let temporal = run(frequentist(
        Study::series(
            TimeSeriesData::from_f64_columns(
                [
                    ("x", columns[0].as_slice()),
                    ("y", columns[1].as_slice()),
                    ("z", columns[2].as_slice()),
                    ("u", columns[3].as_slice()),
                ],
                1,
            )
            .unwrap(),
        )
        .graph(iv_graph())
        .query(CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(id(0), id(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1),
        ))
        .identifier(IdentifierId::TemporalIvUnfolded)
        .estimator(EstimatorId::TemporalLaggedInstrument),
    ));
    // The replicate count lives on the configured estimator, so this study does not set it on
    // the builder as well.
    let static_result = run(Study::tabular(table(&[("z", &zs), ("x", &xs), ("y", &ys)]))
        .graph(dag(3, &[(0, 1), (1, 2)]))
        .query(CausalQuery::AverageEffect(AverageEffectQuery::with_levels(id(1), id(2), 0.0, 1.0)))
        .identifier(IdentifierId::Iv)
        .estimator(TwoStageLeastSquares::new().with_bootstrap_replicates(0))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap());

    let temporal_effect = temporal.estimate().as_effect().unwrap();
    let static_effect = static_result.estimate().as_effect().unwrap();
    // Same estimand, same rows, same ratio.
    assert!(close(temporal_effect.ate, ratio), "{} vs {ratio}", temporal_effect.ate);
    assert!(close(static_effect.ate, ratio), "{} vs {ratio}", static_effect.ate);
    // Same relevance diagnostic.
    let static_first_stage = static_effect.diagnostics.first_stage.as_ref().unwrap();
    assert!(
        (static_first_stage.f_statistic - first_stage_f).abs() < 1e-6 * first_stage_f,
        "{} vs {first_stage_f}",
        static_first_stage.f_statistic
    );
    // Different uncertainty: a homoskedastic Wald SE against an Anderson-Rubin set.
    assert!(close(temporal_effect.se_analytic(), wald_se));
    assert!(static_effect.se_analytic().is_nan());
    let (lower, upper, _) = static_first_stage.anderson_rubin.expect("an Anderson-Rubin set");
    assert!(lower < ratio && ratio < upper, "[{lower}, {upper}] around {ratio}");
    // Different licence: the temporal instrument is certified from the unfolded graph and
    // executes unlicensed; the static cell is licensed.
    assert_eq!(static_result.support_status(), Some(antecedent::support::CellStatus::Licensed));
    assert_ne!(temporal.support_status(), Some(antecedent::support::CellStatus::Licensed));
}

// ---------------------------------------------------------------------------
// Observational sequential contrast against the static AIPW.
// ---------------------------------------------------------------------------

const SUBJECTS: usize = 80;

fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

/// Columns `l0, a0, l1, a1, y`: baseline, first action, post-action covariate, second action, outcome.
fn sequential_panel() -> [Vec<f64>; 5] {
    let mut state = 12_345_u64;
    let mut columns: [Vec<f64>; 5] = Default::default();
    for _ in 0..SUBJECTS {
        let u1 = lcg_uniform(&mut state);
        let u2 = lcg_uniform(&mut state);
        let u3 = lcg_uniform(&mut state);
        let u4 = lcg_uniform(&mut state);
        let u5 = lcg_uniform(&mut state);
        let l0 = 2.0 * u1 - 1.0;
        let a0 = if u2 < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5;
        let a1 = if u4 < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        let y = 1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0);
        for (column, value) in columns.iter_mut().zip([l0, a0, l1, a1, y]) {
            column.push(value);
        }
    }
    columns
}

fn sequential_query() -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: id(4),
        first_treatment: id(1),
        second_treatment: id(3),
        baseline: Arc::from([id(0)]),
        intermediate: Arc::from([id(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..SUBJECTS).map(|row| u32::from(matches!(row % 4, 1 | 2))).collect(),
        fold_count: 2,
        min_probability: 0.05,
    }
}

/// outcome: `not_equivalent` (the sequential family has no one-decision member).
///
/// A sequential contrast is a regime contrast over two decisions with a nested outcome
/// regression, a graphless cross-fit Wald with no support status and no effect payload. Its
/// declaration needs two distinct action columns, so a one-decision panel is refused at
/// validation instead of reducing to a static AIPW. The static AIPW of the first action on the
/// same subjects targets the total effect of that action alone (with the second decision left to
/// follow the observed law), a different estimand, and publishes a licensed graph-based effect.
/// On this panel the two numbers differ by about the second decision's contribution.
#[test]
fn a_sequential_contrast_has_no_one_decision_member_and_differs_from_the_static_aipw_of_the_first_action()
 {
    let [l0, a0, l1, a1, y] = sequential_panel();
    let panel = TabularData::from_f64_columns([
        ("l0", l0.as_slice()),
        ("a0", a0.as_slice()),
        ("l1", l1.as_slice()),
        ("a1", a1.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();

    // The declaration cannot be made with one decision.
    let mut one_decision = sequential_query();
    one_decision.second_treatment = one_decision.first_treatment;
    assert!(one_decision.validate().is_err(), "a repeated action column is not a contrast");

    let context = ExecutionContext::for_tests(41);
    let prepared = Study::tabular(panel.clone())
        .query(CausalQuery::SequentialContrast(sequential_query()))
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let sequential = prepared.estimate(&panel, &context).unwrap();
    let fit = sequential.sequential_contrast().unwrap();
    assert!(matches!(sequential.estimate(), PrimaryEstimate::NotAnEffect));
    assert_eq!(sequential.support_status(), None);

    // The static twin of the first decision alone: adjust for the baseline covariate.
    let static_result = run(frequentist(
        Study::tabular(table(&[("l0", &l0), ("a0", &a0), ("y", &y)]))
            .graph(dag(3, &[(0, 1), (0, 2), (1, 2)]))
            .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(1), id(2))))
            .estimator(EstimatorId::Aipw),
    ));
    let total_effect = static_result.estimate().as_effect().unwrap().ate;
    assert_eq!(static_result.support_status(), Some(antecedent::support::CellStatus::Licensed));
    // Different estimands: the regime contrast adds the second action's own effect.
    assert!(
        (fit.contrast - total_effect).abs() > 0.2,
        "regime contrast {} vs total effect of the first action {total_effect}",
        fit.contrast
    );
    assert!(fit.contrast > total_effect, "the second decision adds to the regime contrast");
}

// ---------------------------------------------------------------------------
// Within-period history match against the static distance matching.
// ---------------------------------------------------------------------------

/// outcome: equivalent matched donors and raw contrast; the static estimate adds a regression
/// bias correction.
///
/// With one history value in one period the period history match is nearest-neighbour matching
/// with replacement on that scalar, so the matched donors and the raw matched-difference
/// contrast are the brute-force ones. The static distance matching (ATT) matches the same
/// donors on the standardized covariate and then subtracts the linear within-control-arm
/// regression bias `b (x_i - x_donor)`; the history match applies none, because its premise is
/// that the scalar score carries the confounding. The static estimate is the history contrast
/// minus that correction, computed here from the raw rows.
#[test]
fn horizon_one_history_match_is_the_unadjusted_matched_difference_of_the_static_distance_matching()
{
    let n = 160_usize;
    let history: Vec<f64> =
        (0..n).map(|i| 3.0 * (0.37 * i as f64).sin() + 0.5 * (1.3 * i as f64).cos()).collect();
    let treated: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(fractional(i as f64 * 0.618_033_988_749_894_9) < sigmoid(0.3 * history[i]))
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 1.5 * treated[i]
                + 0.8 * history[i]
                + 0.2 * history[i] * history[i]
                + 0.3 * (0.53 * i as f64).sin()
        })
        .collect();

    // Brute-force donors: nearest control by absolute history distance, lowest index on a tie.
    let controls: Vec<usize> = (0..n).filter(|&i| treated[i] < 0.5).collect();
    let treated_rows: Vec<usize> = (0..n).filter(|&i| treated[i] > 0.5).collect();
    assert!(controls.len() > 20 && treated_rows.len() > 20);
    let donor_of = |i: usize| -> usize {
        let mut best = controls[0];
        for &j in &controls[1..] {
            if (history[i] - history[j]).abs() < (history[i] - history[best]).abs() {
                best = j;
            }
        }
        best
    };
    let donors: Vec<usize> = treated_rows.iter().map(|&i| donor_of(i)).collect();
    let raw: f64 =
        treated_rows.iter().zip(&donors).map(|(&i, &j)| outcome[i] - outcome[j]).sum::<f64>()
            / treated_rows.len() as f64;
    let control_history: Vec<f64> = controls.iter().map(|&j| history[j]).collect();
    let control_outcome: Vec<f64> = controls.iter().map(|&j| outcome[j]).collect();
    let control_slope = ols(&[&control_history], &control_outcome)[1];
    let correction: f64 = treated_rows
        .iter()
        .zip(&donors)
        .map(|(&i, &j)| control_slope * (history[i] - history[j]))
        .sum::<f64>()
        / treated_rows.len() as f64;
    assert!(correction.abs() > 1e-3, "the regression correction must act: {correction}");

    let history_result =
        Study::tabular(table(&[("h", &history), ("y", &outcome), ("treated", &treated)]))
            .query(CausalQuery::HistoryMatch(HistoryMatchQuery {
                outcome: id(1),
                treated: id(2),
                history: Arc::from([id(0)]),
                period: None,
                weight: None,
                caliper: None,
            }))
            .build()
            .unwrap()
            .run(&ctx())
            .unwrap();
    let matched = history_result.history_match().expect("the history match payload");
    assert!(close(matched.contrast, raw), "{} vs {raw}", matched.contrast);
    let reported: Vec<usize> =
        matched.matched_donors.iter().map(|donor| usize::try_from(*donor).unwrap()).collect();
    assert_eq!(reported, donors);
    assert_eq!(history_result.support_status(), None);

    let static_result = run(frequentist(
        Study::tabular(table(&[("treated", &treated), ("y", &outcome), ("h", &history)]))
            .graph(dag(3, &[(2, 0), (2, 1), (0, 1)]))
            .query(CausalQuery::AverageEffect(
                AverageEffectQuery::binary_ate(id(0), id(1))
                    .with_target_population(TargetPopulation::Treated),
            ))
            .estimator(EstimatorId::DistanceMatching),
    ));
    let static_effect = static_result.estimate().as_effect().unwrap();
    assert!(
        (static_effect.ate - (raw - correction)).abs() < 1e-8,
        "{} vs {}",
        static_effect.ate,
        raw - correction
    );
    assert!((static_effect.ate - matched.contrast).abs() > 1e-3);
    assert!(static_effect.se_analytic().is_finite() && static_effect.se_analytic() > 0.0);
    assert!(matched.standard_error.is_some_and(|se| se.is_finite() && se > 0.0));
}

// ---------------------------------------------------------------------------
// Regime survival at one period against the static two-arm survival.
// ---------------------------------------------------------------------------

/// outcome: equivalent survival at the horizon in each arm and their difference; the interval
/// procedures are `not_equivalent`.
///
/// One period, one treatment decision, no censoring and no confounders: the inverse-probability
/// weighted discrete hazard of each regime is the event share of that arm, so
/// `S_a = 1 - events_a / n_a`, which is the product-limit survival of the randomized two-arm
/// route at the same horizon. The static interval resamples subjects within arms; the temporal
/// one resamples whole subjects and refits the treatment model in every replicate.
#[test]
fn horizon_one_regime_survival_is_the_arm_event_share_of_the_static_two_arm_survival() {
    let n = 200_usize;
    let nf = n as f64;
    let treated: Vec<f64> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
    // Twenty percent of the treated and forty percent of the controls have the event.
    let event: Vec<f64> = (0..n)
        .map(|i| {
            let hit = if i % 2 == 0 { (i / 2) % 5 == 0 } else { ((i - 1) / 2) % 5 < 2 };
            f64::from(hit)
        })
        .collect();
    let treated_events: f64 = (0..n).filter(|&i| treated[i] > 0.5).map(|i| event[i]).sum();
    let control_events: f64 = (0..n).filter(|&i| treated[i] < 0.5).map(|i| event[i]).sum();
    let survival_treated = 1.0 - treated_events / (nf / 2.0);
    let survival_control = 1.0 - control_events / (nf / 2.0);
    assert!((survival_treated - 0.8).abs() < 1e-12 && (survival_control - 0.6).abs() < 1e-12);

    // Static randomized survival: event time 1 when the event occurred, else administratively
    // censored at 2 after the horizon.
    let duration: Vec<f64> = event.iter().map(|e| if *e > 0.5 { 1.0 } else { 2.0 }).collect();
    let static_result = Study::tabular(table(&[
        ("duration", &duration),
        ("event", &event),
        ("treatment", &treated),
    ]))
    .query(CausalQuery::Survival(SurvivalQuery {
        duration: id(0),
        event: id(1),
        treatment: id(2),
        tau: 1.0,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    }))
    .bootstrap_replicates(399)
    .build()
    .unwrap()
    .run(&ctx())
    .unwrap();
    let curve = static_result.survival().expect("the static survival payload");
    let static_treated = *curve.treated.last().unwrap();
    let static_control = *curve.control.last().unwrap();
    assert!((static_treated - survival_treated).abs() < 1e-12, "{static_treated}");
    assert!((static_control - survival_control).abs() < 1e-12, "{static_control}");

    // The temporal regime survival at horizon one.
    let censored = vec![0.0; n];
    let temporal = run_design(
        table(&[("event", &event), ("treated", &treated), ("censored", &censored)]),
        TemporalDesignQuery {
            outcome: id(0),
            units: labels("subject", n).into(),
            periods: vec![0_i64; n].into(),
            lag: 0,
            assignment: TemporalAssignment::RegimeSurvival {
                treatment: id(1),
                censored: id(2),
                confounders: Arc::from([]),
                horizon: 1,
            },
            functional: TemporalDesignFunctional::Mean,
            min_probability: 0.02,
            outcome_misclassification: None,
        },
        99,
    );
    let fit = temporal.temporal_design().unwrap();
    assert_eq!(fit.design.as_ref(), "regime_survival");
    assert!((fit.estimate[1] - survival_treated).abs() < 1e-9, "{}", fit.estimate[1]);
    assert!((fit.estimate[2] - survival_control).abs() < 1e-9, "{}", fit.estimate[2]);
    assert!((fit.estimate[0] - (static_treated - static_control)).abs() < 1e-9);

    // Different interval procedures, each covering the shared difference.
    let [lower, upper] = curve.difference_at_tau_interval.expect("a static bootstrap interval");
    assert!(lower <= fit.estimate[0] && fit.estimate[0] <= upper);
    assert!(fit.lower[0] <= fit.estimate[0] && fit.estimate[0] <= fit.upper[0]);
    assert_ne!(curve.uncertainty.as_ref(), fit.uncertainty.as_ref());
    assert!(fit.replicates.is_some_and(|(done, failed)| done + failed == 99));
}
