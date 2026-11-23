//! Coverage designs for the synthetic-control intervals: the Bayesian credible and predictive
//! intervals, the conformal sets of one treated unit and the placebo-variance interval of a
//! staggered multi-unit rollout.
//!
//! Each one-treated-unit design simulates a balanced panel of ten donors and one treated unit. The treated
//! unit's untreated path is a fixed simplex combination of three donors plus Gaussian noise,
//! and the treatment adds a known effect to each of five post periods (`2.0 + 0.5 k` for the
//! `k`-th post period, so the average effect is 3.0). The realized effect on the treated unit
//! at a post period is exactly that known effect, which is the truth each interval is scored
//! against:
//!
//! * the posterior credible interval of the average post-period effect of the donor-weight
//!   posterior (`bayesian.synthetic_control`);
//! * under the conformal declaration with Bayesian inference, the posterior predictive
//!   intervals of the same average and of the first and last post period
//!   (`bayesian.synthetic_control_predictive_path`).
//!
//! * the conformal set of the first and of the last post period, and the constant-effect set
//!   (`quasi.synthetic_control_conformal`). The rows of the panel are independent over time, so
//!   the refit residuals are exchangeable and the permutation test is exact: its p-value is a
//!   multiple of one over the permutation count `n`, so its size is `floor(0.05 n) / n`. The
//!   per-period designs use `20 k - 1` pre-periods (`n = 20 k`) and the constant-effect design
//!   `20 k - 5` pre-periods with five post periods (`n = 20 k`), where that size is exactly 5%
//!   and the two-sided band applies. At any other count the set is conservative by the same
//!   discreteness. The constant-effect design adds one constant effect to every post period,
//!   which is the null that test inverts.
//!
//! The multi-unit design (`quasi.multi_unit_synthetic_placebo`) draws every unit, adopting or
//! not, from one law (a level, a common random-walk factor, independent noise), so the
//! adopting units' untreated errors are exchangeable with the donors'; three units adopt at
//! staggered periods with one constant effect, the truth of the post-weighted average.
//!
//! A replicate that publishes no interval (a sampler whose chains disagree, a conformal set
//! reaching the searched range) is a skip, never a free pass. A design route has no graph and
//! runs no validation suite, so every record is keyed on the design axes: the query's name,
//! structure `design`, validation `not_applicable`. The Bayesian records keep the runtime's
//! posterior-quantile key; the frequentist intervals ride on the rollout payload, so their
//! records name the interval construction here.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

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

const INTERVAL: &str = "posterior_quantile";
const DONORS: usize = 10;
const POST_PERIODS: usize = 5;
/// Pre-intervention periods at the base grid point.
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

/// The known average effect over the post periods.
fn average_effect() -> f64 {
    (0..POST_PERIODS).map(effect_at).sum::<f64>() / POST_PERIODS as f64
}

/// A simulated panel: outcome column, unit labels and period labels, with the treated unit
/// last in each period. `constant` adds the average effect to every post period instead of
/// the rising one.
fn simulate(pre: usize, seed: u64, constant: bool) -> (TabularData, Vec<Arc<str>>, Vec<i64>) {
    let mut z = gaussian(seed);
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=pre + POST_PERIODS {
        let donors: Vec<f64> = (0..DONORS).map(|j| j as f64 + z()).collect();
        for (j, value) in donors.iter().enumerate() {
            y.push(*value);
            units.push(Arc::<str>::from(format!("d{j:02}")));
            periods.push(i64::try_from(t).expect("the index fits in i64"));
        }
        let mut treated: f64 =
            WEIGHTS.iter().map(|(j, w)| w * donors[*j]).sum::<f64>() + NOISE_SD * z();
        if t > pre {
            treated += if constant { average_effect() } else { effect_at(t - pre - 1) };
        }
        y.push(treated);
        units.push(Arc::<str>::from("treated"));
        periods.push(i64::try_from(t).expect("the index fits in i64"));
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())])
        .expect("the simulated outcome is finite and aligned");
    (data, units, periods)
}

/// What one replicate published and the construction the runtime keyed it under.
struct Replicate {
    /// Equal-tailed interval of the posterior on the average effect.
    posterior: Option<(f64, f64)>,
    /// Interval of the rollout payload's average effect (predictive path only).
    rollout_overall: Option<(f64, f64)>,
    /// Interval of each post period's effect (predictive path only).
    periods: Vec<Option<(f64, f64)>>,
    reported: Vec<(Construction, ScopeFacts, &'static str)>,
}

/// Run the Bayesian synthetic control on one simulated panel. `None` when the route refuses to
/// publish (non-converged chains); a design that cannot be built panics.
fn run(pre: usize, conformal: bool, seed: u64) -> Option<Replicate> {
    let (data, units, periods) = simulate(pre, seed, false);
    let intervention = i64::try_from(pre).expect("the index fits in i64") + 1;
    let mut query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        intervention,
    );
    if conformal {
        query = query.with_conformal_intervals();
    }
    let study = Study::tabular(data.clone())
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .build()
        .expect("valid synthetic-control panel");
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = study.prepare(&ctx).expect("prepared synthetic control");
    let result = match prepared.estimate(&data, &ctx) {
        Ok(result) => result,
        Err(error) if error.reason_code() == Some("posterior_draws_not_converged") => return None,
        Err(error) => panic!("unexpected synthetic-control error: {error}"),
    };
    let contract =
        prepared.contract_for_result(&result).expect("executed synthetic-control contract");
    let reported = constructions(&contract, &result);

    let posterior_fit = result.posterior.as_ref()?;
    let col = posterior_fit.effect_column().expect("the posterior has an effect column");
    let posterior = Some((posterior_fit.summaries.q025[col], posterior_fit.summaries.q975[col]));

    let (mut rollout_overall, mut period_intervals) = (None, Vec::new());
    if conformal {
        let rollout = result.quasi_rollout().expect("the conformal declaration reports a path");
        assert_eq!(rollout.design.as_ref(), "posterior_predictive_synthetic_control");
        let pair = |interval: Option<[f64; 2]>| interval.map(|[lower, upper]| (lower, upper));
        rollout_overall = rollout
            .effects
            .iter()
            .find(|effect| effect.kind == DesignEffectKind::Overall)
            .and_then(|effect| pair(effect.interval));
        period_intervals = (0..POST_PERIODS)
            .map(|k| {
                let period = intervention + i64::try_from(k).expect("the index fits in i64");
                rollout
                    .effects
                    .iter()
                    .find(|effect| {
                        effect.kind == DesignEffectKind::Period && effect.period == Some(period)
                    })
                    .and_then(|effect| pair(effect.interval))
            })
            .collect();
    }
    Some(Replicate { posterior, rollout_overall, periods: period_intervals, reported })
}

/// Which published interval a design scores, with the known truth it is scored against.
#[derive(Clone, Copy)]
enum Scored {
    /// The posterior credible interval of the average effect.
    Posterior,
    /// The predictive path's average-effect interval.
    PredictiveOverall,
    /// The predictive path's interval at the `k`-th post period.
    PredictivePeriod(usize),
}

impl Scored {
    fn conformal(self) -> bool {
        !matches!(self, Self::Posterior)
    }

    fn truth(self) -> f64 {
        match self {
            Self::Posterior | Self::PredictiveOverall => average_effect(),
            Self::PredictivePeriod(k) => effect_at(k),
        }
    }

    fn interval(self, run: &Replicate) -> Option<(f64, f64)> {
        match self {
            Self::Posterior => run.posterior,
            Self::PredictiveOverall => run.rollout_overall,
            Self::PredictivePeriod(k) => run.periods.get(k).copied().flatten(),
        }
    }
}

fn measure(test: &'static str, scored: Scored, seed_base: u64) {
    let pre = grid_n(BASE_PRE_PERIODS);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: DGP, interval: INTERVAL }, REPORTED_LEVEL);
    let stream =
        if matches!(scored, Scored::PredictiveOverall) { 0x5041_5448_0000_0000 } else { 0 };
    let runs = map_replicates(n_sim(), |rep| {
        run(pre, scored.conformal(), grid_seed((seed_base + rep) ^ stream))
    });
    let truth = scored.truth();
    for replicate in runs {
        // A refusal to publish, or a missing interval, is a miss.
        let Some(replicate) = replicate else {
            tally.skip();
            continue;
        };
        let Some(interval) = scored.interval(&replicate) else {
            tally.skip();
            continue;
        };
        let (construction, scope, validation) = replicate
            .reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == INTERVAL)
            .expect("the execution reports its posterior interval");
        assert_eq!(construction.query, QUERY);
        assert_eq!(scope.row_count, pre as u64);
        let mut construction = construction.clone();
        if let Scored::PredictivePeriod(k) = scored {
            // This study reads one period from the rollout, a separate target
            // from the average which the public primary basis describes.
            construction.functional = format!("{}+period={}", construction.functional, k + 1);
        }
        tally.bind_validation(&construction, *scope, validation);
        tally.record(Some(interval), truth);
    }
    tally.assert_panel_interval();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval() {
    measure("bayesian_synthetic_control_posterior_interval", Scored::Posterior, 0x5C01_0000);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_average_interval() {
    measure(
        "bayesian_synthetic_control_predictive_path_average_interval",
        Scored::PredictiveOverall,
        0x5C02_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_first_period_interval() {
    measure(
        "bayesian_synthetic_control_predictive_path_first_period_interval",
        Scored::PredictivePeriod(0),
        0x5C03_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_last_period_interval() {
    measure(
        "bayesian_synthetic_control_predictive_path_last_period_interval",
        Scored::PredictivePeriod(POST_PERIODS - 1),
        0x5C04_0000,
    );
}

// ------------------------------------------------------------------ conformal sets

/// Interval method of the conformal records.
const CONFORMAL: &str = "conformal";
/// Interval method of the placebo-variance record.
const PLACEBO: &str = "placebo_variance";

/// One frequentist replicate: the scored interval and the record's construction.
struct Published {
    interval: (f64, f64),
    construction: Construction,
    scope: ScopeFacts,
}

/// The runtime's primary key for `result` with the interval a rollout payload carries named
/// as `method` under `dependence`.
fn published(
    study: &Study,
    result: &antecedent::StudyResult,
    interval: [f64; 2],
    method: &str,
    dependence: &str,
    replicates_ok: Option<u32>,
) -> Option<Published> {
    let (mut construction, mut scope, _) =
        constructions(&study.inspect().ok()?, result).into_iter().next()?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = method.into();
    construction.se_kind = String::new();
    construction.dependence = dependence.into();
    construction.reported_level = REPORTED_LEVEL;
    scope.replicates_ok = replicates_ok;
    Some(Published { interval: (interval[0], interval[1]), construction, scope })
}

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
    fn pre_periods(self) -> usize {
        let twenties = 20 * SampleGrid::STANDARD.n(4);
        match self {
            Self::Period(_) => twenties - 1,
            Self::Constant => twenties - POST_PERIODS,
        }
    }

    fn truth(self) -> f64 {
        match self {
            Self::Period(k) => effect_at(k),
            Self::Constant => average_effect(),
        }
    }
}

/// Run the conformal synthetic control on one panel and read the scored set; `None` when the
/// route refuses. A run that publishes no set for the scored quantity is scored as a miss.
fn conformal_replicate(set: ConformalSet, seed: u64) -> Option<Published> {
    let pre = set.pre_periods();
    let (data, units, periods) = simulate(pre, seed, matches!(set, ConformalSet::Constant));
    let intervention = i64::try_from(pre).expect("the index fits in i64") + 1;
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
            let period = intervention + i64::try_from(k).expect("the index fits in i64");
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
    published(&study, &result, interval, CONFORMAL, "iid", None)
}

fn measure_published(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    truth: f64,
    replicate: impl Fn(u64) -> Option<Published> + Sync,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    for run in map_replicates(n_sim(), replicate) {
        // A refusal or an absent interval is a miss.
        let Some(run) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&run.construction, run.scope, NO_VALIDATION);
        tally.record(Some(run.interval), truth);
    }
    tally.assert_panel_interval();
}

fn measure_conformal(test: &'static str, set: ConformalSet, seed_base: u64) {
    measure_published(test, DGP, CONFORMAL, set.truth(), |rep| {
        conformal_replicate(set, grid_seed(seed_base + rep))
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_first_period_set() {
    measure_conformal(
        "conformal_synthetic_control_first_period_set",
        ConformalSet::Period(0),
        0x5C05_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_last_period_set() {
    measure_conformal(
        "conformal_synthetic_control_last_period_set",
        ConformalSet::Period(POST_PERIODS - 1),
        0x5C06_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_constant_effect_set() {
    measure_conformal(
        "conformal_synthetic_control_constant_effect_set",
        ConformalSet::Constant,
        0x5C07_0000,
    );
}

// ------------------------------------------------------- multi-unit placebo variance

/// Periods of the multi-unit panel and the adoption period of each adopting unit.
const ROLLOUT_PERIODS: usize = 30;
const ADOPTIONS: [i64; 3] = [21, 24, 27];
/// Constant effect on every adopting unit from its adoption on.
const ROLLOUT_EFFECT: f64 = 2.0;
/// Never-treated donors at the base grid point.
const BASE_DONORS: usize = 20;
/// Placebo refits per replicate.
const PLACEBO_REFITS: u32 = 199;

/// Run the multi-unit synthetic control on one exchangeable panel and read the
/// placebo-variance interval of the post-weighted average.
fn placebo_replicate(seed: u64) -> Option<Published> {
    let donors = grid_n(BASE_DONORS);
    let mut z = gaussian(seed);
    let mut factor = Vec::with_capacity(ROLLOUT_PERIODS);
    let mut level = 0.0;
    for _ in 0..ROLLOUT_PERIODS {
        level += z();
        factor.push(level);
    }
    let adoptions: Vec<(Arc<str>, i64)> = ADOPTIONS
        .iter()
        .enumerate()
        .map(|(k, adoption)| (Arc::<str>::from(format!("t{k}")), *adoption))
        .collect();
    let mut labels: Vec<(Arc<str>, Option<i64>)> =
        (0..donors).map(|j| (Arc::<str>::from(format!("c{j:03}")), None)).collect();
    labels.extend(adoptions.iter().map(|(label, adoption)| (Arc::clone(label), Some(*adoption))));
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for (label, adoption) in &labels {
        // Every unit's level is a draw from one law, uniform on (0, 3), whether it adopts or not.
        let unit_level = 3.0 * normal_cdf(z());
        for (t, common) in factor.iter().enumerate() {
            let period = i64::try_from(t).expect("the index fits in i64") + 1;
            let effect = if adoption.is_some_and(|a| period >= a) { ROLLOUT_EFFECT } else { 0.0 };
            y.push(unit_level + common + effect + NOISE_SD * z());
            units.push(Arc::clone(label));
            periods.push(period);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).ok()?;
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::clone(&adoptions[0].0),
        adoptions[0].1,
    )
    .with_staggered_adoptions(adoptions, PLACEBO_REFITS);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let interval = rollout.overall()?.interval?;
    published(&study, &result, interval, PLACEBO, "unit", Some(rollout.replicates_ok))
}

/// The standard normal distribution function, by the Abramowitz-Stegun rational approximation
/// of the error function (absolute error below 1.5e-7): a uniform draw from a normal one.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * (x.abs() / std::f64::consts::SQRT_2));
    let polynomial = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - polynomial * (-(x * x) / 2.0).exp();
    0.5 * (1.0 + if x >= 0.0 { erf } else { -erf })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn multi_unit_synthetic_placebo_variance_interval() {
    measure_published(
        "multi_unit_synthetic_placebo_variance_interval",
        "exchangeable_units_common_factor_staggered_constant_effect",
        PLACEBO,
        ROLLOUT_EFFECT,
        |rep| placebo_replicate(grid_seed(0x5C08_0000 + rep)),
    );
}
