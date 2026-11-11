//! Coverage designs for the pointwise 95% intervals of randomized trials at 300 participants.
//!
//! Each design runs the public randomized-effect route on a simulated trial and scores the
//! interval it publishes against the known finite-population average effect of the simulated
//! units (the effect is `1.4 + 0.2 * x1`, so it varies by unit, and the two baseline covariates
//! are fixed deterministic sequences). Only the assignment is random, which is the randomness
//! the intervals are about:
//!
//! * the independent-assignment score interval and the ANCOVA (HC0) interval, each under a
//!   balanced allocation and a 2:1 allocation, on 300 participants (the grid halves and doubles
//!   the size);
//! * the Neyman interval of a complete randomization with 150 of 300 units treated;
//! * the small-arm interval: a trial whose smaller arm realizes 20 of 300 participants gets the
//!   HC2 variance with a Bell-McCaffrey Student reference, for the difference in means of a
//!   complete randomization and for the ANCOVA. Its construction is read from the runtime's own
//!   calibration key, so its record is keyed apart from the normal intervals above.
//!
//! The designs are `#[ignore]`d measurement designs registered in `parity/calibration_groups.toml`
//! (`run_ri`). The fixed-seed checks of the same routes are
//! `crates/antecedent/tests/randomized_precision_intervals.rs` and
//! `crates/antecedent/tests/randomized_trial_route.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{RandomizedEffectQuery, Study, StudyResult};
use antecedent_core::{ExecutionContext, RandomizationDesign, StreamDomain, VariableId};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;

const INTERVAL: &str = "analytic_se";
const OUTCOME: VariableId = VariableId::from_raw(0);
const BASELINE_A: VariableId = VariableId::from_raw(1);
const BASELINE_B: VariableId = VariableId::from_raw(2);

/// How the interval under test adjusts the outcome.
#[derive(Clone, Copy)]
enum Adjustment {
    /// The independent-assignment inverse-probability score contrast.
    Score,
    /// ANCOVA on both baseline covariates with the HC0 variance.
    Ancova,
}

/// How a replicate assigns the units.
#[derive(Clone, Copy)]
enum Assignment {
    /// Independent Bernoulli assignment with this treatment probability.
    Bernoulli(f64),
    /// Complete randomization of exactly this many treated units.
    Complete(usize),
    /// A trial that declares independent assignment with this probability and realizes exactly
    /// `treated` treated units: the allocation a weak arm leaves.
    DeclaredBernoulli { probability: f64, treated: usize },
    /// A smaller arm of exactly this many treated units: a complete randomization for the score
    /// contrast, a Bernoulli trial declaring `treated / n` for the ANCOVA.
    WeakArm { treated: usize },
}

/// The simulated units: baseline covariates and the known average effect.
struct Units {
    x1: Vec<f64>,
    x2: Vec<f64>,
    effect: f64,
}

fn units(n: usize) -> Units {
    let x1 = (0..n).map(|i| (0.17 * i as f64).sin()).collect::<Vec<_>>();
    let x2 = (0..n).map(|i| (0.11 * i as f64).cos()).collect::<Vec<_>>();
    let effect = x1.iter().map(|x| 1.4 + 0.2 * x).sum::<f64>() / n as f64;
    Units { x1, x2, effect }
}

/// Draw the assignment of one replicate.
fn assign(n: usize, assignment: Assignment, seed: u64) -> Vec<bool> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x5201);
    match assignment {
        Assignment::Bernoulli(probability) => {
            (0..n).map(|_| rng.next_f64() < probability).collect()
        }
        Assignment::Complete(treated)
        | Assignment::DeclaredBernoulli { treated, .. }
        | Assignment::WeakArm { treated } => {
            let keys = (0..n).map(|_| rng.next_f64()).collect::<Vec<_>>();
            let mut order = (0..n).collect::<Vec<_>>();
            order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
            let mut assigned = vec![false; n];
            for &unit in &order[..treated] {
                assigned[unit] = true;
            }
            assigned
        }
    }
}

/// The prepared study and result of one simulated trial and the known average effect; a hard
/// refusal is a bug in the design and panics.
fn run_trial(
    n: usize,
    adjustment: Adjustment,
    assignment: Assignment,
    seed: u64,
) -> (Study, StudyResult, f64) {
    let units = units(n);
    let assignment = match (assignment, adjustment) {
        (Assignment::WeakArm { treated }, Adjustment::Score) => Assignment::Complete(treated),
        (Assignment::WeakArm { treated }, Adjustment::Ancova) => {
            Assignment::DeclaredBernoulli { probability: treated as f64 / n as f64, treated }
        }
        (other, _) => other,
    };
    let assigned = assign(n, assignment, seed);
    let outcome = (0..n)
        .map(|i| {
            2.0 + 1.2 * units.x1[i] - 0.7 * units.x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + f64::from(assigned[i]) * (1.4 + 0.2 * units.x1[i])
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("baseline_a", units.x1.as_slice()),
        ("baseline_b", units.x2.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    let ids = (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let rows = (0..n).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>();
    let query = match assignment {
        Assignment::Bernoulli(probability) | Assignment::DeclaredBernoulli { probability, .. } => {
            RandomizedEffectQuery::bernoulli_itt(
                OUTCOME,
                assigned,
                vec![probability; n],
                ids,
                rows,
                ("control", "treated"),
            )
        }
        Assignment::Complete(treated) | Assignment::WeakArm { treated } => {
            RandomizedEffectQuery::with_design(
                RandomizationDesign::Complete { treated_units: treated },
                OUTCOME,
                assigned,
                vec![treated as f64 / n as f64; n],
                ids,
                rows,
                ("control", "treated"),
            )
        }
    };
    let query = match adjustment {
        Adjustment::Score => query,
        Adjustment::Ancova => query.with_ancova(vec![BASELINE_A, BASELINE_B]),
    };
    let study =
        Study::tabular(data).query(query).build().expect("the trial declares a valid design");
    let result = study
        .run(&ExecutionContext::for_tests(seed))
        .expect("the route runs on the simulated trial");
    (study, result, units.effect)
}

/// The interval the route publishes for one simulated trial (`None` when it withholds it), the
/// diagnostics it reports and the known average effect.
fn published(
    n: usize,
    adjustment: Adjustment,
    assignment: Assignment,
    seed: u64,
) -> (Option<[f64; 2]>, Vec<Arc<str>>, f64) {
    let (_, result, effect) = run_trial(n, adjustment, assignment, seed);
    let fit = result.randomized_effect().expect("a randomized effect result");
    let diagnostics = result.diagnostics.iter().map(|d| Arc::<str>::from(&*d.message)).collect();
    (fit.interval_95, diagnostics, effect)
}

fn construction(estimator: &str, se_kind: &str) -> Construction {
    Construction {
        query: "RandomizedEffect".into(),
        graph_class: "RandomizedTrial".into(),
        structure: "design".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: INTERVAL.into(),
        se_kind: se_kind.into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "randomized_itt".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn measure(
    test: &'static str,
    dgp: &'static str,
    construction: &Construction,
    adjustment: Adjustment,
    assignment: Assignment,
    seed_base: u64,
) {
    let n = grid_n(300);
    // The treated count is declared for 300 participants and keeps its share of the grid's size,
    // so a halved grid point is still a proper complete randomization.
    let assignment = match assignment {
        Assignment::Complete(treated) => Assignment::Complete(treated * n / 300),
        other => other,
    };
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    let runs = map_replicates(n_sim(), |rep| {
        published(n, adjustment, assignment, grid_seed(seed_base + rep))
    });
    for (interval, _, truth) in runs {
        // An allocation that realizes too few rows in an arm publishes no interval; the route
        // withholds it, so the replicate is a skip (a miss), never a free pass.
        let Some([lower, upper]) = interval else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            construction,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

/// The simulated trial's outcomes and assignment, rebuilt from the generating equations
/// (independent of the route).
fn simulated(
    n: usize,
    adjustment: Adjustment,
    assignment: Assignment,
    seed: u64,
) -> (Vec<bool>, Vec<f64>) {
    let units = units(n);
    let assignment = match (assignment, adjustment) {
        (Assignment::WeakArm { treated }, Adjustment::Score) => Assignment::Complete(treated),
        (Assignment::WeakArm { treated }, Adjustment::Ancova) => {
            Assignment::DeclaredBernoulli { probability: treated as f64 / n as f64, treated }
        }
        (other, _) => other,
    };
    let assigned = assign(n, assignment, seed);
    let outcome = (0..n)
        .map(|i| {
            2.0 + 1.2 * units.x1[i] - 0.7 * units.x2[i]
                + 0.5 * (0.29 * i as f64).sin()
                + f64::from(assigned[i]) * (1.4 + 0.2 * units.x1[i])
        })
        .collect();
    (assigned, outcome)
}

/// Arm means and unbiased arm variances `(treated mean, control mean, treated var, control var,
/// treated n, control n)`.
fn arm_summary(assigned: &[bool], y: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let side = |flag: bool| {
        let values: Vec<f64> =
            assigned.iter().zip(y).filter(|(a, _)| **a == flag).map(|(_, v)| *v).collect();
        let m = values.len() as f64;
        let mean = values.iter().sum::<f64>() / m;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
        (mean, var, m)
    };
    let (m1, v1, n1) = side(true);
    let (m0, v0, n0) = side(false);
    (m1, m0, v1, v0, n1, n0)
}

/// OLS of `y` on `[1, assigned, x1, x2]` by Gauss-Jordan inversion of the normal matrix:
/// the treatment coefficient, its HC0 standard error and its HC2 standard error.
fn ancova_oracle(assigned: &[bool], y: &[f64]) -> (f64, f64, f64) {
    let n = y.len();
    let u = units(n);
    let rows: Vec<[f64; 4]> =
        (0..n).map(|i| [1.0, f64::from(assigned[i]), u.x1[i], u.x2[i]]).collect();
    let mut aug = [[0.0_f64; 8]; 4];
    for r in &rows {
        for i in 0..4 {
            for j in 0..4 {
                aug[i][j] += r[i] * r[j];
            }
        }
    }
    for (i, row) in aug.iter_mut().enumerate() {
        row[4 + i] = 1.0;
    }
    for col in 0..4 {
        let pivot =
            (col..4).max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs())).unwrap();
        aug.swap(col, pivot);
        let d = aug[col][col];
        for v in &mut aug[col] {
            *v /= d;
        }
        for r in 0..4 {
            if r != col {
                let f = aug[r][col];
                let pivot_row = aug[col];
                for (v, p) in aug[r].iter_mut().zip(pivot_row) {
                    *v -= f * p;
                }
            }
        }
    }
    let inv: Vec<Vec<f64>> = (0..4).map(|i| aug[i][4..].to_vec()).collect();
    let xty: Vec<f64> =
        (0..4).map(|i| rows.iter().zip(y).map(|(r, v)| r[i] * v).sum::<f64>()).collect();
    let beta: Vec<f64> = (0..4).map(|i| (0..4).map(|j| inv[i][j] * xty[j]).sum()).collect();
    let (mut meat0, mut meat2) = (0.0, 0.0);
    for (r, v) in rows.iter().zip(y) {
        let residual = v - (0..4).map(|k| beta[k] * r[k]).sum::<f64>();
        let leverage: f64 =
            (0..4).map(|i| (0..4).map(|j| r[i] * inv[i][j] * r[j]).sum::<f64>()).sum();
        let weight: f64 = (0..4).map(|i| inv[1][i] * r[i]).sum();
        meat0 += (weight * residual).powi(2);
        meat2 += (weight * residual).powi(2) / (1.0 - leverage);
    }
    (beta[1], meat0.sqrt(), meat2.sqrt())
}

/// The 97.5th percentile of the standard normal.
const Z975: f64 = 1.959_963_984_540_054;

/// Student-t quantile by bisection on the integrated density (`ln_gamma` normalization).
fn student_quantile_975(df: f64) -> f64 {
    let ln_norm = antecedent_stats::ln_gamma(f64::midpoint(df, 1.0))
        - antecedent_stats::ln_gamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln();
    let pdf = |t: f64| (ln_norm - f64::midpoint(df, 1.0) * (1.0 + t * t / df).ln()).exp();
    // P(0 < T < x) by Simpson's rule; the target is 0.475.
    let half_mass = |x: f64| {
        let steps = 4000;
        let h = x / f64::from(steps);
        let mut sum = pdf(0.0) + pdf(x);
        for k in 1..steps {
            sum += pdf(h * f64::from(k)) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * h / 3.0
    };
    let (mut lo, mut hi) = (0.0_f64, 10.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if half_mass(mid) < 0.475 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Each published interval is the estimator a reader would write down: the difference in means
/// with Neyman's variance and the normal 1.96 for a complete randomization, the
/// Horvitz-Thompson score contrast for independent assignment, ANCOVA with the HC0 sandwich, and
/// Welch-type Student intervals for a weak arm. Each is recomputed here from the generating
/// equations, so a wrong divisor, weighting, variance or reference quantile is caught to the
/// last digits, not by a band around the known effect.
#[test]
fn the_scored_intervals_equal_their_hand_computed_constructions() {
    let n = 300;
    let nf = n as f64;

    // Complete randomization, 150 treated: difference in means, Neyman variance, z = 1.96.
    let (assigned, y) = simulated(n, Adjustment::Score, Assignment::Complete(150), 7);
    let (m1, m0, v1, v0, n1, n0) = arm_summary(&assigned, &y);
    let (_, result, truth) = run_trial(n, Adjustment::Score, Assignment::Complete(150), 7);
    let fit = result.randomized_effect().expect("a randomized effect result");
    let se = (v1 / n1 + v0 / n0).sqrt();
    assert!((fit.effect - (m1 - m0)).abs() < 1e-12, "{} vs {}", fit.effect, m1 - m0);
    assert!((fit.standard_error.unwrap() - se).abs() < 1e-12, "{:?} vs {se}", fit.standard_error);
    let [lower, upper] = fit.interval_95.expect("a complete randomization publishes an interval");
    assert!(
        (lower - (m1 - m0 - Z975 * se)).abs() < 1e-12
            && (upper - (m1 - m0 + Z975 * se)).abs() < 1e-12
    );
    assert!(lower < truth && truth < upper, "[{lower}, {upper}] vs {truth}");

    // Independent assignment, p = 1/2: the Horvitz-Thompson contrast and the standard error of
    // its per-unit scores.
    let (assigned, y) = simulated(n, Adjustment::Score, Assignment::Bernoulli(0.5), 7);
    let scores: Vec<f64> =
        (0..n).map(|i| if assigned[i] { y[i] / 0.5 } else { -y[i] / 0.5 }).collect();
    let effect = scores.iter().sum::<f64>() / nf;
    let spread = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>();
    let (_, result, _) = run_trial(n, Adjustment::Score, Assignment::Bernoulli(0.5), 7);
    let fit = result.randomized_effect().unwrap();
    let se = (spread / (nf * (nf - 1.0))).sqrt();
    assert!((fit.effect - effect).abs() < 1e-12, "{} vs {effect}", fit.effect);
    assert!((fit.standard_error.unwrap() - se).abs() < 1e-12, "{:?} vs {se}", fit.standard_error);
    let [lower, upper] = fit.interval_95.unwrap();
    assert!(
        (lower - (effect - Z975 * se)).abs() < 1e-12
            && (upper - (effect + Z975 * se)).abs() < 1e-12
    );

    // ANCOVA on both baselines with the HC0 sandwich, normal reference.
    let (assigned, y) = simulated(n, Adjustment::Ancova, Assignment::Bernoulli(0.5), 7);
    let (beta, hc0, _) = ancova_oracle(&assigned, &y);
    let (_, result, truth) = run_trial(n, Adjustment::Ancova, Assignment::Bernoulli(0.5), 7);
    let fit = result.randomized_effect().unwrap();
    assert!((fit.effect - beta).abs() < 1e-9, "{} vs {beta}", fit.effect);
    assert!((fit.standard_error.unwrap() - hc0).abs() < 1e-9, "{:?} vs {hc0}", fit.standard_error);
    let [lower, upper] = fit.interval_95.unwrap();
    assert!(
        (lower - (beta - Z975 * hc0)).abs() < 1e-9 && (upper - (beta + Z975 * hc0)).abs() < 1e-9
    );
    assert!(lower < truth && truth < upper, "[{lower}, {upper}] vs {truth}");

    // Weak arm, 20 treated of 300, difference in means: Neyman's standard error and a Student
    // reference near Welch's (the Bell-McCaffrey degrees of freedom are within a unit of
    // Welch's here), far from the normal 1.96.
    let (assigned, y) = simulated(n, Adjustment::Score, Assignment::WeakArm { treated: 20 }, 7);
    let (m1, m0, v1, v0, n1, n0) = arm_summary(&assigned, &y);
    let (a, b) = (v1 / n1, v0 / n0);
    let welch = (a + b).powi(2) / (a * a / (n1 - 1.0) + b * b / (n0 - 1.0));
    let (_, result, truth) =
        run_trial(n, Adjustment::Score, Assignment::WeakArm { treated: 20 }, 7);
    let fit = result.randomized_effect().unwrap();
    assert!((fit.effect - (m1 - m0)).abs() < 1e-12);
    assert!((fit.standard_error.unwrap() - (a + b).sqrt()).abs() < 1e-12);
    let [lower, upper] = fit.interval_95.unwrap();
    let ratio = (upper - lower) / (2.0 * fit.standard_error.unwrap());
    assert!((f64::midpoint(upper, lower) - (m1 - m0)).abs() < 1e-12);
    assert!(
        (ratio - student_quantile_975(welch)).abs() < 0.01,
        "reference {ratio} vs Welch t {} at {welch} degrees of freedom",
        student_quantile_975(welch)
    );
    assert!(lower < truth && truth < upper, "[{lower}, {upper}] vs {truth}");

    // The same arm under ANCOVA: the HC2 standard error and a Student reference.
    let (assigned, y) = simulated(n, Adjustment::Ancova, Assignment::WeakArm { treated: 20 }, 7);
    let (beta, _, hc2) = ancova_oracle(&assigned, &y);
    let (_, result, _) = run_trial(n, Adjustment::Ancova, Assignment::WeakArm { treated: 20 }, 7);
    let fit = result.randomized_effect().unwrap();
    assert!((fit.effect - beta).abs() < 1e-9, "{} vs {beta}", fit.effect);
    assert!((fit.standard_error.unwrap() - hc2).abs() < 1e-9, "{:?} vs {hc2}", fit.standard_error);
    let [lower, upper] = fit.interval_95.unwrap();
    let ratio = (upper - lower) / (2.0 * hc2);
    assert!((ratio - student_quantile_975(welch)).abs() < 0.02, "reference {ratio}");
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_score_balanced_interval() {
    measure(
        "randomized_score_balanced_interval",
        "bernoulli_trial_balanced_known_finite_population_effect",
        &construction("randomized_independent_action_score", "influence"),
        Adjustment::Score,
        Assignment::Bernoulli(0.5),
        131_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_score_two_to_one_interval() {
    measure(
        "randomized_score_two_to_one_interval",
        "bernoulli_trial_two_to_one_known_finite_population_effect",
        &construction("randomized_independent_action_score", "influence"),
        Adjustment::Score,
        Assignment::Bernoulli(1.0 / 3.0),
        132_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_ancova_balanced_interval() {
    measure(
        "randomized_ancova_balanced_interval",
        "bernoulli_trial_balanced_known_finite_population_effect",
        &construction("randomized_ancova_hc0", "hc0"),
        Adjustment::Ancova,
        Assignment::Bernoulli(0.5),
        133_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn randomized_ancova_two_to_one_interval() {
    measure(
        "randomized_ancova_two_to_one_interval",
        "bernoulli_trial_two_to_one_known_finite_population_effect",
        &construction("randomized_ancova_hc0", "hc0"),
        Adjustment::Ancova,
        Assignment::Bernoulli(1.0 / 3.0),
        134_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn complete_randomization_neyman_interval() {
    // The Neyman variance is conservative under unit-level effect variation (here
    // `0.2 * x1`, a variance of 0.02 against an outcome variance near one), so the
    // design is held to nominal coverage from below and to its precision ceiling.
    measure(
        "complete_randomization_neyman_interval",
        "complete_randomization_half_treated_known_finite_population_effect",
        &construction("randomized_complete_neyman", "neyman"),
        Adjustment::Score,
        Assignment::Complete(150),
        135_000,
    );
}

/// Coverage of the small-arm interval: every replicate's construction and scope facts come from
/// the runtime's own calibration key, so the record is keyed by what the facade reports.
fn measure_small_arm(
    test: &'static str,
    dgp: &'static str,
    adjustment: Adjustment,
    treated: usize,
    seed_base: u64,
) {
    let n = grid_n(300);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    let runs = map_replicates(n_sim(), |rep| {
        let (study, result, truth) =
            run_trial(n, adjustment, Assignment::WeakArm { treated }, grid_seed(seed_base + rep));
        let interval = result.randomized_effect().expect("a randomized result").interval_95;
        let bound = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == INTERVAL);
        (interval, truth, bound)
    });
    for (interval, truth, bound) in runs {
        // A replicate that publishes no interval is a skip (a miss), never a free pass.
        let (Some([lower, upper]), Some((construction, scope, validation))) = (interval, bound)
        else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&construction, scope, validation);
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

/// A smaller arm of 20 of 300 participants is below the normal interval's support. The
/// difference in means of a complete randomization publishes the HC2 Bell-McCaffrey Student
/// interval, held to nominal coverage and to the precision ceiling.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weak_arm_difference_in_means_hc2_student_interval() {
    measure_small_arm(
        "weak_arm_difference_in_means_hc2_student_interval",
        "complete_randomization_twenty_treated_known_finite_population_effect",
        Adjustment::Score,
        20,
        136_000,
    );
}

/// The ANCOVA of a trial whose smaller arm realizes 20 of 300 participants publishes the same
/// construction over the two baseline covariates.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weak_arm_ancova_hc2_student_interval() {
    measure_small_arm(
        "weak_arm_ancova_hc2_student_interval",
        "bernoulli_trial_twenty_treated_known_finite_population_effect",
        Adjustment::Ancova,
        20,
        137_000,
    );
}

/// Fixed-seed check of the small-arm routes: a published interval, its own construction label,
/// the Student reference (wider than the normal), and a calibration key apart from the normal
/// interval's.
#[test]
fn the_small_arm_interval_is_published_under_its_own_construction() {
    for (adjustment, label) in [
        (Adjustment::Score, "complete_difference_in_means_hc2_bell_mccaffrey_student_interval"),
        (Adjustment::Ancova, "bernoulli_ancova_hc2_bell_mccaffrey_student_interval"),
    ] {
        let (study, result, truth) =
            run_trial(300, adjustment, Assignment::WeakArm { treated: 20 }, 7);
        let fit = result.randomized_effect().expect("a randomized effect result");
        let [lower, upper] = fit.interval_95.expect("a 20-participant arm publishes an interval");
        assert_eq!(fit.uncertainty.as_ref(), label);
        assert!(lower < truth + 0.5 && truth - 0.5 < upper);
        let standard_error = fit.standard_error.expect("the interval carries its standard error");
        assert!(
            (upper - lower) / (2.0 * standard_error) > 1.96,
            "a Student reference, not the normal"
        );
        let (construction, _, _) = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == INTERVAL)
            .expect("an analytic interval is reported");
        assert_eq!(construction.se_kind, "hc2");
    }
}

/// Calibrate the executed nonlinear decision-evidence interval under a fixed cluster population.
/// Each replicate randomizes clusters; the truth is the difference of functions of the known
/// finite-population arm means. The optional coefficient is fixed independently of outcomes.
#[allow(
    clippy::items_after_statements,
    clippy::too_many_lines,
    reason = "the bounded reference fixture preserves its exact calculations and local structure"
)]
fn measure_decision_margin(test: &'static str, centered: bool) {
    use antecedent::decision_evidence::*;
    let n = grid_n(160);
    let sizes = vec![5_usize; n];
    let rows = 5 * n;
    let pre: Vec<f64> = (0..n).map(|i| 5.0 * (1.0 + (i as f64 * 0.17).sin())).collect();
    let potential = |treated: bool, k: usize| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = pre[i] / 5.0;
                let shock = (i as f64 * 0.37).cos();
                let y = if k == 0 {
                    2.0 + 0.4 * x + 0.2 * shock + if treated { 0.3 } else { 0.0 }
                } else {
                    9.0 + x + 0.5 * shock + if treated { -0.4 } else { 0.0 }
                };
                5.0 * y
            })
            .collect()
    };
    let (t0, t1, c0, c1) =
        (potential(true, 0), potential(true, 1), potential(false, 0), potential(false, 1));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / rows as f64;
    let truth = mean(&t0) * mean(&t1) - mean(&c0) * mean(&c1);
    let quantity = DerivedQuantity {
        name: "calibration_margin".into(),
        units: "utility".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    let request = Request {
        population_id: "calibration_population".into(),
        treated_regime_id: "treated".into(),
        control_regime_id: "control".into(),
        horizon: 1,
        target: Target::ExperimentalContrast,
    };
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "cluster_margin_fixed_population",
            interval: "delta_method_normal_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let assigned = assign(n, Assignment::Complete(n / 2), grid_seed(0x6150_0000 + rep));
        use antecedent_estimate::joint_design::{
            cluster_decision::{FixedClusterAdjustment, JointClusterRecipe},
            excursion::ExcursionOutcome,
            experiment::ExperimentExposure,
        };
        let y0: Vec<f64> = (0..n).map(|i| if assigned[i] { t0[i] } else { c0[i] }).collect();
        let y1: Vec<f64> = (0..n).map(|i| if assigned[i] { t1[i] } else { c1[i] }).collect();
        let adjustments = if centered {
            vec![FixedClusterAdjustment {
                component_name: "frequency".into(),
                covariate: "pre".into(),
                coefficient: 0.4,
                pretreatment_totals: pre.clone(),
                centered: true,
                preassignment: "fixed_before_assignment".into(),
            }]
        } else {
            vec![]
        };
        let recipe = JointClusterRecipe {
            population: "calibration_population".into(),
            treated_regime_id: "treated".into(),
            control_regime_id: "control".into(),
            horizon: 1,
            cluster_ids: (0..n).map(|i| format!("cluster-{i}")).collect(),
            cluster_sizes: sizes.clone(),
            assignment_law: "complete_randomization".into(),
            treated_clusters: n / 2,
            consistency: "consistency".into(),
            assignments: assigned,
            outcomes: vec![
                ExcursionOutcome {
                    component_name: "frequency".into(),
                    variable_id: "frequency".into(),
                    units: "u".into(),
                    values: y0,
                },
                ExcursionOutcome {
                    component_name: "value".into(),
                    variable_id: "value".into(),
                    units: "u".into(),
                    values: y1,
                },
            ],
            adjustments,
            exposure: ExperimentExposure::NoInterference {
                assumption: "no_between_cluster_interference".into(),
            },
        };
        let ctx = ExecutionContext::for_tests(grid_seed(0x6150_0000 + rep));
        let retained =
            compile_and_evidence_from_cluster(recipe, "calibration_evidence", &ctx).unwrap();
        let ev = retained.numerical_evidence();
        let result = derive(ev, &request, &quantity).unwrap();
        let interval = result.sampling.map(|s| (s.lower, s.upper));
        let basis = decision_interval_calibration_basis(ev, &request, &quantity).unwrap();
        (interval, basis)
    }) {
        let construction = Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        };
        tally.bind_validation(
            &construction,
            ScopeFacts {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            },
            &basis.validation,
        );
        tally.record(interval, truth);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_joint_cluster_delta_interval() {
    measure_decision_margin("decision_margin_joint_cluster_delta_interval", false);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_centered_cuped_cluster_delta_interval() {
    measure_decision_margin("decision_margin_centered_cuped_cluster_delta_interval", true);
}

/// Registered but `not_measured` coverage design for the joint matched-pair margin.
///
/// The runtime's matched-pair variance treats the pairs as i.i.d. draws from a superpopulation,
/// so each replicate draws its pairs afresh: a pair level `u ~ 0.2 + Uniform(-1, 1)`, a unit
/// deviation `0.2 * (Uniform(0, 1) - 0.5)` shared by both components of a unit, independent
/// outcome noise `0.3 * (Uniform(0, 1) - 0.5)` on each component, and the within-pair coin flip.
/// The truth is the population margin, with `E[base] = 0.2` in place of the sample average.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_matched_pair_delta_interval() {
    use antecedent::decision_evidence::*;
    let pairs = grid_n(80);
    let n = 2 * pairs;
    let design = compile_matched_pair_design(MatchedPairDesign {
        cluster_ids: (0..n).map(|i| format!("cluster-{i}")).collect(),
        pairs: (0..pairs).map(|i| (2 * i, 2 * i + 1)).collect(),
        matching_variables: vec!["pre".into()],
        treatment_probabilities: vec![0.5; n],
        cluster_sizes: vec![1; n],
        preassignment_assumption_id: "fixed".into(),
        independent_pairs_assumption_id: "independent".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let q = DerivedQuantity {
        name: "paired_calibration_margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "t".into(),
        control_regime_id: "c".into(),
        horizon: 1,
        target: Target::ExperimentalContrast,
    };
    // Unit outcomes are `a = 2.0 + base + 0.3 T + e_a` and `b = 4.0 + 0.5 base + 0.5 T + e_b`,
    // with `base = u + d`: the pair level `u` (mean 0.2), the unit deviation `d` and the outcome
    // noises `e_a`, `e_b` (each mean zero) are independent of the assignment. So the arm means of
    // the population are `E[a(1)] = 2.3 + mu`, `E[b(1)] = 4.5 + 0.5 mu`, `E[a(0)] = 2.0 + mu`,
    // `E[b(0)] = 4.0 + 0.5 mu` with `mu = E[base] = 0.2`, and the margin (a difference of products
    // of arm means) is `2.5 * 4.6 - 2.2 * 4.1 = 2.48` exactly.
    let mu = 0.2;
    let truth = (2.3 + mu) * (4.5 + 0.5 * mu) - (2.0 + mu) * (4.0 + 0.5 * mu);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "decision_margin_matched_pair_delta_interval",
            dgp: "fixed_matched_pair_margin",
            interval: "delta_method_student_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let mut rng = ExecutionContext::for_tests(grid_seed(0x8810_0000 + rep))
            .rng
            .stream_for(StreamDomain::Test, 0x8810);
        let mut assigned = vec![false; n];
        for i in 0..pairs {
            assigned[2 * i] = rng.next_f64() < 0.5;
            assigned[2 * i + 1] = !assigned[2 * i];
        }
        // The pairs of this replicate, on a stream apart from the assignment's.
        let mut draw = ExecutionContext::for_tests(grid_seed(0x8810_0000 + rep))
            .rng
            .stream_for(StreamDomain::Test, 0x8811);
        let level: Vec<f64> = (0..pairs).map(|_| mu + 2.0 * draw.next_f64() - 1.0).collect();
        let base: Vec<f64> = (0..n).map(|i| level[i / 2] + 0.2 * (draw.next_f64() - 0.5)).collect();
        let a: Vec<f64> = (0..n)
            .map(|i| {
                2.0 + base[i]
                    + (if assigned[i] { 0.3 } else { 0.0 })
                    + 0.3 * (draw.next_f64() - 0.5)
            })
            .collect();
        let b: Vec<f64> = (0..n)
            .map(|i| {
                4.0 + 0.5 * base[i]
                    + (if assigned[i] { 0.5 } else { 0.0 })
                    + 0.3 * (draw.next_f64() - 0.5)
            })
            .collect();
        let components = ["frequency", "value"]
            .into_iter()
            .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
            .collect();
        let ctx = ExecutionContext::for_tests(grid_seed(0x8810_0000 + rep));
        let retained = compile_and_evidence_from_matched_pairs(
            &design,
            &[&a, &b],
            &assigned,
            "e",
            "pop",
            ("t", "c"),
            1,
            components,
            Exposure::NoBetweenUnitInterference { assumption_id: "none".into() },
            &ctx,
        )
        .unwrap();
        let ev = retained.numerical_evidence();
        let interval = derive(ev, &req, &q).unwrap().sampling.map(|s| (s.lower, s.upper));
        let basis = decision_interval_calibration_basis(ev, &req, &q).unwrap();
        (interval, basis)
    }) {
        bind_decision_candidate(&mut tally, &basis);
        tally.record(interval, truth);
    }
    tally.assert();
}

/// Registered but `not_measured` coverage design for the joint sequence-level switchback margin.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_switchback_delta_interval() {
    use antecedent::decision_evidence::*;
    let sequences = grid_n(80);
    let periods = 4;
    let n = sequences * periods;
    let design = compile_joint_switchback(JointSwitchbackDesign {
        sequence_ids: (0..sequences)
            .flat_map(|s| (0..periods).map(move |_| format!("s{s}")))
            .collect(),
        period_ids: (0..sequences).flat_map(|_| (0..periods).map(|p| format!("p{p}"))).collect(),
        treatment_probabilities: vec![0.5; n],
        no_carryover_assumption_id: "none".into(),
        washout_rule_id: "zero".into(),
        washout_periods: 0,
        independent_sequences_assumption_id: "independent".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let q = DerivedQuantity {
        name: "switchback_calibration_margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "t".into(),
        control_regime_id: "c".into(),
        horizon: 1,
        target: Target::ExperimentalContrast,
    };
    let base: Vec<f64> = (0..n)
        .map(|i| ((i / periods) as f64 * 0.19).sin() + 0.1 * (i as f64 * 0.37).cos())
        .collect();
    let avg = base.iter().sum::<f64>() / n as f64;
    let truth = (2.3 + avg) * (4.5 + 0.5 * avg) - (2.0 + avg) * (4.0 + 0.5 * avg);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "decision_margin_switchback_delta_interval",
            dgp: "fixed_switchback_margin",
            interval: "delta_method_student_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let mut rng = ExecutionContext::for_tests(grid_seed(0x8820_0000 + rep))
            .rng
            .stream_for(StreamDomain::Test, 0x8820);
        let assigned: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
        let a: Vec<f64> =
            (0..n).map(|i| 2.0 + base[i] + if assigned[i] { 0.3 } else { 0.0 }).collect();
        let b: Vec<f64> =
            (0..n).map(|i| 4.0 + 0.5 * base[i] + if assigned[i] { 0.5 } else { 0.0 }).collect();
        let components = ["frequency", "value"]
            .into_iter()
            .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
            .collect();
        let ctx = ExecutionContext::for_tests(grid_seed(0x8820_0000 + rep));
        let retained = compile_and_evidence_from_joint_switchback(
            &design,
            &[&a, &b],
            &assigned,
            "e",
            "pop",
            ("t", "c"),
            1,
            components,
            Exposure::NoBetweenUnitInterference { assumption_id: "none".into() },
            &ctx,
        )
        .unwrap();
        let ev = retained.numerical_evidence();
        let interval = derive(ev, &req, &q).unwrap().sampling.map(|s| (s.lower, s.upper));
        let basis = decision_interval_calibration_basis(ev, &req, &q).unwrap();
        (interval, basis)
    }) {
        bind_decision_candidate(&mut tally, &basis);
        tally.record(interval, truth);
    }
    tally.assert();
}

fn bind_decision_candidate(tally: &mut CoverageTally, basis: &antecedent_core::CalibrationBasis) {
    let construction = Construction {
        query: basis.query.to_string(),
        graph_class: basis.graph_class.to_string(),
        structure: basis.structure.to_string(),
        modality: basis.modality.to_string(),
        inference: basis.inference.to_string(),
        estimator: basis.estimator.to_string(),
        interval_method: basis.interval_method.to_string(),
        se_kind: basis.se_kind.to_string(),
        dependence: basis.dependence.to_string(),
        posterior: basis.posterior.to_string(),
        functional: basis.functional.to_string(),
        identification: basis.identification.to_string(),
        reported_level: basis.level,
    };
    tally.bind_validation(
        &construction,
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
        &basis.validation,
    );
}

/// Registered, `not_measured` coverage design for a lagged randomized excursion margin.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
#[allow(
    clippy::cast_possible_wrap,
    reason = "the bounded reference fixture preserves its exact calculations and local structure"
)]
fn decision_margin_lagged_excursion_delta_interval() {
    use antecedent::decision_evidence::*;
    let units = grid_n(80);
    let periods = 5;
    let n = units * periods;
    let design = compile_joint_excursion(JointExcursionDesign {
        unit_ids: (0..units).flat_map(|u| (0..periods).map(move |_| format!("u-{u}"))).collect(),
        periods: (0..units).flat_map(|_| (0..periods).map(|p| p as i64)).collect(),
        treatment_probabilities: vec![0.5; n],
        lag: 1,
        min_probability: 0.1,
        assignment_law_id: "independent-half".into(),
        independent_units_assumption_id: "independent-units".into(),
        administrative_followup_assumption_id: "end-of-study".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let req = Request {
        population_id: "pop".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 4,
        target: Target::Excursion { lag: 1, assignment_law_id: "independent-half".into() },
    };
    let q = DerivedQuantity {
        name: "lagged_excursion_calibration_margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    let baseline: Vec<f64> = (0..units).map(|u| (u as f64 * 0.19).sin()).collect();
    let avg = baseline.iter().sum::<f64>() / units as f64;
    let truth = (3.0 + avg) * (5.6 + 0.5 * avg) - (2.0 + avg) * (5.0 + 0.5 * avg);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "decision_margin_lagged_excursion_delta_interval",
            dgp: "fixed_lagged_excursion_margin",
            interval: "delta_method_normal_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let mut rng = ExecutionContext::for_tests(grid_seed(0x8830_0000 + rep))
            .rng
            .stream_for(StreamDomain::Test, 0x8830);
        let assigned: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                2.0 + baseline[i / periods]
                    + if i % periods > 0 && assigned[i - 1] { 1.0 } else { 0.0 }
            })
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|i| {
                5.0 + 0.5 * baseline[i / periods]
                    + if i % periods > 0 && assigned[i - 1] { 0.6 } else { 0.0 }
            })
            .collect();
        let components = ["frequency", "value"]
            .into_iter()
            .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
            .collect();
        let ctx = ExecutionContext::for_tests(grid_seed(0x8830_0000 + rep));
        let retained = compile_and_evidence_from_joint_excursion(
            &design,
            &[&y, &z],
            &assigned,
            "e",
            "pop",
            ("on", "off"),
            4,
            components,
            Exposure::NoBetweenUnitInterference { assumption_id: "no-spill".into() },
            &ctx,
        )
        .unwrap();
        let ev = retained.numerical_evidence();
        let interval = derive(ev, &req, &q).unwrap().sampling.map(|s| (s.lower, s.upper));
        let basis = decision_interval_calibration_basis(ev, &req, &q).unwrap();
        (interval, basis)
    }) {
        bind_decision_candidate(&mut tally, &basis);
        tally.record(interval, truth);
    }
    tally.assert();
}

/// Registered, `not_measured` joint whole-subject regime bootstrap margin design.
///
/// The whole-subject bootstrap resamples the confounder with the subject, so each replicate
/// draws it afresh: `x ~ 0.3 + Uniform(-1, 1)` i.i.d., with the Bernoulli(1/2) assignment and
/// the outcome noise. The truth is the population margin, with `E[x] = 0.3` in place of the
/// sample mean.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_static_regime_bootstrap_delta_interval() {
    use antecedent::decision_evidence::*;
    let n = grid_n(100);
    let unit_ids: Vec<String> = (0..n).map(|i| format!("u-{i}")).collect();
    // Outcomes are `y = 2.0 + x + 0.5 A + 0.2 e` and `z = 4.0 + 0.5 x + 0.8 A + 0.3 e`, with
    // `x` (mean 0.3) and the noise `e ~ Uniform(-0.5, 0.5)` (mean zero) independent of `A`. So the
    // regime means of the population are `E[y(on)] = 2.5 + mu`, `E[z(on)] = 4.8 + 0.5 mu`,
    // `E[y(off)] = 2.0 + mu`, `E[z(off)] = 4.0 + 0.5 mu` with `mu = E[x] = 0.3`, and the margin is
    // `2.8 * 4.95 - 2.3 * 4.15 = 4.315` exactly.
    let mu = 0.3;
    let truth = (2.5 + mu) * (4.8 + 0.5 * mu) - (2.0 + mu) * (4.0 + 0.5 * mu);
    let quantity = DerivedQuantity {
        name: "static_regime_margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![
            InputBinding::Measured("frequency".into()),
            InputBinding::Measured("value".into()),
        ],
    };
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "on".into(),
        control_regime_id: "off".into(),
        horizon: 1,
        target: Target::Regime {
            periods: 1,
            exchangeability_assumption_id: "randomization".into(),
        },
    };
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "decision_margin_static_regime_bootstrap_delta_interval",
            dgp: "fixed_single_period_regime_margin",
            interval: "delta_method_normal_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let ctx = ExecutionContext::for_tests(grid_seed(0x8840_0000 + rep));
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x8840);
        let assigned: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
        let noise: Vec<f64> = (0..n).map(|_| rng.next_f64() - 0.5).collect();
        // The subjects' confounder, on a stream apart from the assignment and noise.
        let mut draw = ctx.rng.stream_for(StreamDomain::Test, 0x8841);
        let x: Vec<f64> = (0..n).map(|_| mu + 2.0 * draw.next_f64() - 1.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 + x[i] + (if assigned[i] { 0.5 } else { 0.0 }) + 0.2 * noise[i])
            .collect();
        let z: Vec<f64> = (0..n)
            .map(|i| 4.0 + 0.5 * x[i] + (if assigned[i] { 0.8 } else { 0.0 }) + 0.3 * noise[i])
            .collect();
        let design = compile_joint_static_regime(JointStaticRegimeDesign {
            unit_ids: unit_ids.clone(),
            periods: vec![0; n],
            realized_assignment: assigned,
            confounder_ids: vec!["x".into()],
            active_path: vec![true],
            reference_path: vec![false],
            method: "g_formula".into(),
            min_probability: 0.05,
            positivity_assumption_id: "positivity".into(),
            exchangeability_assumption_id: "randomization".into(),
            consistency_assumption_id: "consistency".into(),
            followup_assumption_id: "complete".into(),
            model_assumption_id: "specified-model".into(),
        })
        .unwrap();
        let components = ["frequency", "value"]
            .into_iter()
            .map(|v| OutcomeComponent { name: v.into(), variable_id: v.into(), units: "u".into() })
            .collect();
        let bound = compile_and_evidence_from_joint_static_regime_bootstrap(
            &design,
            &[&y, &z],
            std::slice::from_ref(&x),
            "e",
            "pop",
            ("on", "off"),
            components,
            Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
            99,
            grid_seed(0x8850_0000 + rep),
            &ctx,
        )
        .unwrap();
        let ev = bound.numerical_evidence();
        let interval =
            derive(ev, &request, &quantity).unwrap().sampling.map(|s| (s.lower, s.upper));
        let basis = decision_interval_calibration_basis(ev, &request, &quantity).ok();
        (interval, basis)
    }) {
        if let Some(basis) = basis {
            bind_decision_candidate(&mut tally, &basis);
        }
        tally.record(interval, truth);
    }
    tally.assert();
}

/// Registered, `not_measured` coverage design for finite-history adaptive-policy jackknife margins.
///
/// Each replicate draws its episodes i.i.d.: `a ~ Bernoulli(1/2)`, the state
/// `s ~ Bernoulli(0.75 if a else 0.25)`, `b ~ Bernoulli(1/2)` and the outcome noises. The truth
/// is the population margin: the adaptive policy (`a = 1`, `b = s`) has `E[y] = 4 + 8 * 0.75 = 10`
/// and `E[z] = 5 + 0.5 * 10 = 10`, the reference (`a = b = 0`) has `E[y] = 2 + 4 * 0.25 = 3` and
/// `E[z] = 6.5`, so the margin is `10 * 10 - 3 * 6.5 = 80.5`. The base of 320 episodes keeps the
/// rarest policy-history cells, `(a, s, b) = (0, 1, 0)` and `(1, 0, 0)` at probability 1/16, at
/// an expected 10 or more episodes at the smallest grid point, so their leave-one-out cells are
/// not left empty.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn decision_margin_adaptive_regime_jackknife_delta_interval() {
    use antecedent::decision_evidence::*;
    use antecedent_estimate::joint_design::adaptive_regime::{
        AdaptiveAction, AdaptiveEpisode, AdaptivePolicy, JointAdaptiveRegimeRecipe,
    };
    let n = grid_n(320);
    let request = Request {
        population_id: "pop".into(),
        treated_regime_id: "adaptive".into(),
        control_regime_id: "reference".into(),
        horizon: 2,
        target: Target::AdaptiveRegime {
            periods: 2,
            exchangeability_assumption_id: "randomization".into(),
        },
    };
    let quantity = DerivedQuantity {
        name: "adaptive_margin".into(),
        units: "u".into(),
        expr: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
        inputs: vec![InputBinding::Measured("y".into()), InputBinding::Measured("z".into())],
    };
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "decision_margin_adaptive_regime_jackknife_delta_interval",
            dgp: "randomized_treatment_affected_binary_state",
            interval: "delta_method_normal_95",
        },
        REPORTED_LEVEL,
    );
    for (interval, basis) in map_replicates(n_sim(), |rep| {
        let ctx = ExecutionContext::for_tests(grid_seed(0x8860_0000 + rep));
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x8860);
        let mut episodes = Vec::with_capacity(n);
        for i in 0..n {
            let a = usize::from(rng.next_f64() < 0.5);
            let state = usize::from(rng.next_f64() < if a == 1 { 0.75 } else { 0.25 });
            let b = usize::from(rng.next_f64() < 0.5);
            let noise = rng.next_f64() - 0.5;
            let y = 2.0 + (2 * a + 3 * b + 4 * state + a * b) as f64 + noise;
            let z = 5.0 + 0.5 * y + 0.2 * (rng.next_f64() - 0.5);
            episodes.push(AdaptiveEpisode {
                id: format!("u{i}"),
                states: vec![0, state],
                actions: vec![a, b],
                outcomes: vec![y, z],
            });
        }
        let policy = |id: &str, first: usize, adaptive: bool| AdaptivePolicy {
            id: id.into(),
            rules: vec![
                AdaptiveAction { states: vec![0], past_actions: vec![], action: first },
                AdaptiveAction { states: vec![0, 0], past_actions: vec![first], action: 0 },
                AdaptiveAction {
                    states: vec![0, 1],
                    past_actions: vec![first],
                    action: usize::from(adaptive),
                },
            ],
        };
        let recipe = JointAdaptiveRegimeRecipe {
            population: "pop".into(),
            state_variable_id: "state".into(),
            action_variable_id: "action".into(),
            outcomes: vec![
                ("y".into(), "y".into(), "u".into()),
                ("z".into(), "z".into(), "u".into()),
            ],
            policies: [policy("adaptive", 1, true), policy("reference", 0, false)],
            episodes,
            policy_prespecification: "fixed before evaluation".into(),
            positivity: "positivity".into(),
            exchangeability: "randomization".into(),
            consistency: "consistency".into(),
            followup: "complete".into(),
            isolation: "independent".into(),
        };
        match compile_and_evidence_from_adaptive_regime(recipe, "e", &ctx) {
            Ok(bound) => {
                let ev = bound.numerical_evidence();
                (
                    derive(ev, &request, &quantity)
                        .ok()
                        .and_then(|r| r.sampling.map(|s| (s.lower, s.upper))),
                    decision_interval_calibration_basis(ev, &request, &quantity).ok(),
                )
            }
            Err(_) => (None, None),
        }
    }) {
        if let Some(basis) = basis {
            bind_decision_candidate(&mut tally, &basis);
        }
        tally.record(interval, 80.5);
    }
    tally.assert();
}
