//! Coverage designs for the numbers of post periods that the synthetic-control support rows
//! admit and `synthetic_control_calibration.rs` does not measure.
//!
//! The conformal-set row and the predictive-path row each need two published intervals: a
//! single post period gives the average (or constant-effect) interval and the period's own.
//! `synthetic_control_calibration.rs` measures them with five post periods only. These designs
//! repeat its panels with one and with three post periods and change nothing else.
//!
//! The panel is the base design's: ten donors and one treated unit whose untreated path is the
//! simplex combination `0.5 d3 + 0.3 d4 + 0.2 d5` plus Gaussian noise of standard deviation
//! `0.5`, independent over time. The treatment adds `2.0 + 0.5 k` to the `k`-th post period,
//! so with `P` post periods the effect at period `k` is that value and the average effect is
//! `2.0 + 0.25 (P - 1)`: 2.0 for one post period and 2.5 for three. The constant-effect panel
//! adds that average to every post period, which is the null the constant-effect set inverts.
//!
//! * Predictive path (`bayesian.synthetic_control_predictive_path`): the 95% posterior
//!   predictive intervals of the average effect and of the first and last post period, at
//!   `grid_n(40)` pre-periods (30, 40 and 80). With one post period the first and the last are
//!   the same period, so the average and the period are scored.
//! * Conformal sets (`quasi.synthetic_control_conformal`): the set of the first and of the last
//!   post period and the constant-effect set. The permutation p-value is a multiple of one over
//!   the permutation count `n` (pre-periods plus one for a period, pre plus post periods for
//!   the constant effect), so its size is `floor(0.05 n) / n`, exactly 5% when `n` is a
//!   multiple of twenty. The period designs use `20 k - 1` pre-periods and the constant-effect
//!   designs `20 k - P`, for `k` = 2, 4, 8, which gives `n` = 40, 80 and 160.
//!
//! A replicate that publishes no interval (a sampler whose chains disagree, a conformal set
//! reaching the searched range) is a skip, never a free pass. A design route has no graph and
//! runs no validation suite, so every record is keyed on the design axes: the query's name,
//! structure `design`, validation `not_applicable`.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, Study};
use antecedent_core::{ExecutionContext, SyntheticControlQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::rollout_resampling::DesignEffectKind;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, SampleGrid, ScopeFacts, gaussian,
    grid_n, grid_seed, map_replicates, n_sim,
};
use common::calibration_bind::constructions;

/// Query name of every record.
const QUERY: &str = "SyntheticControl";
/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";
/// Interval method of the predictive path.
const POSTERIOR: &str = "posterior_quantile";
/// Interval method of the conformal sets.
const CONFORMAL: &str = "conformal";
const DONORS: usize = 10;
/// Pre-intervention periods of the predictive path at the base grid point.
const BASE_PRE_PERIODS: usize = 40;
const NOISE_SD: f64 = 0.5;
const DRAWS: usize = 2_000;
const DGP: &str = "convex_donor_panel_iid_noise";
/// Donors that form the treated unit's untreated path, with their simplex weights.
const WEIGHTS: [(usize, f64); 3] = [(3, 0.5), (4, 0.3), (5, 0.2)];

/// The known effect on the treated unit at the `k`-th post period.
fn effect_at(k: usize) -> f64 {
    2.0 + 0.5 * k as f64
}

/// The known average effect over `post` post periods.
fn average_effect(post: usize) -> f64 {
    (0..post).map(effect_at).sum::<f64>() / post as f64
}

/// A simulated panel: outcome column, unit labels and period labels, with the treated unit
/// last in each period. `constant` adds the average effect to every post period instead of the
/// rising one.
fn simulate(
    pre: usize,
    post: usize,
    seed: u64,
    constant: bool,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>) {
    let mut z = gaussian(seed);
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=pre + post {
        let donors: Vec<f64> = (0..DONORS).map(|j| j as f64 + z()).collect();
        for (j, value) in donors.iter().enumerate() {
            y.push(*value);
            units.push(Arc::<str>::from(format!("d{j:02}")));
            periods.push(t as i64);
        }
        let mut treated: f64 =
            WEIGHTS.iter().map(|(j, w)| w * donors[*j]).sum::<f64>() + NOISE_SD * z();
        if t > pre {
            treated += if constant { average_effect(post) } else { effect_at(t - pre - 1) };
        }
        y.push(treated);
        units.push(Arc::<str>::from("treated"));
        periods.push(t as i64);
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())])
        .expect("the simulated outcome is finite and aligned");
    (data, units, periods)
}

/// One replicate: the scored interval and the record's construction.
struct Published {
    interval: (f64, f64),
    construction: Construction,
    scope: ScopeFacts,
}

/// Score every replicate's interval against `truth` under its own construction; a refusal or
/// an absent interval is a skip, never a free pass.
fn measure(
    test: &'static str,
    interval: &'static str,
    truth: f64,
    replicate: impl Fn(u64) -> Option<Published> + Sync,
) {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: DGP, interval }, REPORTED_LEVEL);
    for run in map_replicates(n_sim(), replicate) {
        let Some(run) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&run.construction, run.scope, NO_VALIDATION);
        tally.record(Some(run.interval), truth);
    }
    tally.assert();
}

// ------------------------------------------------------------------ predictive path

/// Which predictive-path interval a design scores.
#[derive(Clone, Copy)]
enum Path {
    /// The interval of the average effect.
    Average,
    /// The interval of the `k`-th post period.
    Period(usize),
}

impl Path {
    fn truth(self, post: usize) -> f64 {
        match self {
            Self::Average => average_effect(post),
            Self::Period(k) => effect_at(k),
        }
    }
}

/// Run the Bayesian synthetic control with the conformal declaration on one panel of `post`
/// post periods and read the scored predictive interval; `None` when the route refuses or
/// publishes no interval for it.
fn predictive_replicate(post: usize, path: Path, seed: u64) -> Option<Published> {
    let pre = grid_n(BASE_PRE_PERIODS);
    let (data, units, periods) = simulate(pre, post, seed, false);
    let intervention = pre as i64 + 1;
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        intervention,
    )
    .with_conformal_intervals();
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .build()
        .expect("the panel declares a valid synthetic control");
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let reported = constructions(&study.inspect().expect("inspect"), &result);
    let rollout = result.quasi_rollout().expect("the conformal declaration reports a path");
    assert_eq!(rollout.design.as_ref(), "posterior_predictive_synthetic_control");
    let interval = match path {
        Path::Average => {
            rollout
                .effects
                .iter()
                .find(|effect| effect.kind == DesignEffectKind::Overall)?
                .interval?
        }
        Path::Period(k) => {
            let period = intervention + k as i64;
            rollout
                .effects
                .iter()
                .find(|effect| {
                    effect.kind == DesignEffectKind::Period && effect.period == Some(period)
                })?
                .interval?
        }
    };
    let (construction, scope, _) = reported
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == POSTERIOR)?;
    let mut construction = construction;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    Some(Published { interval: (interval[0], interval[1]), construction, scope })
}

fn measure_predictive(test: &'static str, post: usize, path: Path, seed_base: u64) {
    measure(test, POSTERIOR, path.truth(post), |rep| {
        predictive_replicate(post, path, grid_seed(seed_base + rep))
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_one_post_period_average_interval() {
    measure_predictive(
        "bayesian_synthetic_control_predictive_path_one_post_period_average_interval",
        1,
        Path::Average,
        0x5C11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_one_post_period_period_interval() {
    measure_predictive(
        "bayesian_synthetic_control_predictive_path_one_post_period_period_interval",
        1,
        Path::Period(0),
        0x5C12_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_three_post_periods_average_interval() {
    measure_predictive(
        "bayesian_synthetic_control_predictive_path_three_post_periods_average_interval",
        3,
        Path::Average,
        0x5C13_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_three_post_periods_first_period_interval() {
    measure_predictive(
        "bayesian_synthetic_control_predictive_path_three_post_periods_first_period_interval",
        3,
        Path::Period(0),
        0x5C14_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_three_post_periods_last_period_interval() {
    measure_predictive(
        "bayesian_synthetic_control_predictive_path_three_post_periods_last_period_interval",
        3,
        Path::Period(2),
        0x5C15_0000,
    );
}

// ------------------------------------------------------------------ conformal sets

/// The interval scored when a run publishes no set: it covers nothing, so it is a miss.
const NO_SET: [f64; 2] = [f64::NAN; 2];

/// Which conformal set a design scores.
#[derive(Clone, Copy)]
enum ConformalSet {
    /// The set of the `k`-th post period, on the rising-effect panel.
    Period(usize),
    /// The constant-effect set, on the constant-effect panel.
    Constant,
}

impl ConformalSet {
    /// Pre-periods at the current grid point: the permutation count is a multiple of twenty.
    fn pre_periods(self, post: usize) -> usize {
        let twenties = 20 * SampleGrid::STANDARD.n(4);
        match self {
            Self::Period(_) => twenties - 1,
            Self::Constant => twenties - post,
        }
    }

    fn truth(self, post: usize) -> f64 {
        match self {
            Self::Period(k) => effect_at(k),
            Self::Constant => average_effect(post),
        }
    }
}

/// Run the conformal synthetic control on one panel of `post` post periods and read the scored
/// set; `None` when the route refuses. A run that publishes no set for the scored quantity is
/// scored as a miss.
fn conformal_replicate(post: usize, set: ConformalSet, seed: u64) -> Option<Published> {
    let pre = set.pre_periods(post);
    let (data, units, periods) = simulate(pre, post, seed, matches!(set, ConformalSet::Constant));
    let intervention = pre as i64 + 1;
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        intervention,
    )
    .with_conformal_intervals();
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let interval = match set {
        ConformalSet::Constant => rollout.overall()?.interval.unwrap_or(NO_SET),
        ConformalSet::Period(k) => {
            let period = intervention + k as i64;
            rollout
                .effects
                .iter()
                .find(|effect| {
                    effect.kind == DesignEffectKind::Period && effect.period == Some(period)
                })?
                .interval
                .unwrap_or(NO_SET)
        }
    };
    let (mut construction, scope, _) =
        constructions(&study.inspect().ok()?, &result).into_iter().next()?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = CONFORMAL.into();
    construction.se_kind = String::new();
    construction.dependence = "iid".into();
    construction.reported_level = REPORTED_LEVEL;
    Some(Published { interval: (interval[0], interval[1]), construction, scope })
}

fn measure_conformal(test: &'static str, post: usize, set: ConformalSet, seed_base: u64) {
    measure(test, CONFORMAL, set.truth(post), |rep| {
        conformal_replicate(post, set, grid_seed(seed_base + rep))
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_one_post_period_period_set() {
    measure_conformal(
        "conformal_synthetic_control_one_post_period_period_set",
        1,
        ConformalSet::Period(0),
        0x5C16_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_one_post_period_constant_effect_set() {
    measure_conformal(
        "conformal_synthetic_control_one_post_period_constant_effect_set",
        1,
        ConformalSet::Constant,
        0x5C17_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_three_post_periods_first_period_set() {
    measure_conformal(
        "conformal_synthetic_control_three_post_periods_first_period_set",
        3,
        ConformalSet::Period(0),
        0x5C18_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_three_post_periods_last_period_set() {
    measure_conformal(
        "conformal_synthetic_control_three_post_periods_last_period_set",
        3,
        ConformalSet::Period(2),
        0x5C19_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_three_post_periods_constant_effect_set() {
    measure_conformal(
        "conformal_synthetic_control_three_post_periods_constant_effect_set",
        3,
        ConformalSet::Constant,
        0x5C1A_0000,
    );
}
