//! Repeated-sampling coverage of the sequential-contrast intervals at the cross-fit fold counts
//! its support rows admit and `temporal_routes_calibration.rs` does not measure.
//!
//! The rows are stated for any cross-fit with at least five folds none of which holds more than
//! a fifth of the rows (rounded up); the base designs measure five balanced folds only. These
//! designs repeat the base panel and query with two other fold splits and change nothing else:
//!
//! * ten balanced folds: row `i` is in fold `i mod 10`;
//! * an unbalanced split into eight folds holding `8, 6, 6, 5, 5, 4, 3, 3` fortieths of the
//!   rows: row `i` is in the fold that owns position `i mod 40` of the cycle, so at 200 rows the
//!   folds hold 40, 30, 30, 25, 25, 20, 15 and 15 rows. The largest fold is exactly a fifth of
//!   the rows, which the row allows, and every nuisance model trains on at least four fifths.
//!   The grid's sizes (200, 400 and 800 rows) are multiples of forty.
//!
//! The panel is the base design's: columns `l0, a0, l1, a1, y` with `l0` uniform with unit
//! variance, `a0 ~ Bernoulli(expit(.5 l0))`, `l1 = .5 l0 + .8 a0 + .5 u`, `a1 ~
//! Bernoulli(expit(.5 (l1 - .4)))` and `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e`. Under
//! sequential exchangeability given the measured history `E[Y(a0, a1)] = 1 + 1.32 a0 + 1.5 a1`,
//! so the contrast of treating at both decisions against neither is `1.32 + 1.5`. The histories
//! are bounded so every action probability lies in `[.25, .75]` and the nuisance models (logistic
//! propensities, linear conditional means) are correctly specified.
//!
//! These are design routes: they declare no graph and run no validation suite, so each record is
//! keyed on the design axes (structure `design`, validation `not_applicable`) and a Frequentist
//! record states the interval method, the standard-error kind and the dependence of the interval
//! the route publishes. A replicate that publishes no interval is a skip, never a free pass.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, Study, StudyResult};
use antecedent_core::{CausalQuery, ExecutionContext, SequentialContrastQuery, VariableId};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed, uniform,
};
use common::calibration_bind::constructions;

/// Interval method of the interval a route forms from its own standard error.
const NORMAL_INTERVAL: &str = "analytic_se";
/// Interval method of the equal-tailed posterior interval.
const POSTERIOR_INTERVAL: &str = "posterior_quantile";
/// Posterior draws of the Bayesian designs.
const DRAWS: usize = 2_000;
const BASE_N: usize = 400;
/// Contrast of the regime that treats at both decisions against the regime that treats at
/// neither: `E[Y(1, 1)] - E[Y(0, 0)] = (1 + 1.32 + 1.5) - 1`.
const CONTRAST_TRUTH: f64 = 1.32 + 1.5;

type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A draw on `[-sqrt 3, sqrt 3]` (mean zero, unit variance) from a uniform draw on `[0, 1)`.
fn bounded(unit: f64) -> f64 {
    3.0_f64.sqrt() * (2.0 * unit - 1.0)
}

/// The cross-fit split of a design.
#[derive(Clone, Copy)]
enum Folds {
    /// Ten folds, row `i` in fold `i mod 10`.
    TenBalanced,
    /// Eight folds holding `8, 6, 6, 5, 5, 4, 3, 3` fortieths of the rows.
    EightUnbalanced,
}

/// Fortieths of the rows held by each fold of the unbalanced split; the first is a fifth.
const UNBALANCED_FORTIETHS: [usize; 8] = [8, 6, 6, 5, 5, 4, 3, 3];

impl Folds {
    fn count(self) -> u32 {
        match self {
            Self::TenBalanced => 10,
            Self::EightUnbalanced => 8,
        }
    }

    /// Fold of each of `n` rows (`n` a multiple of forty).
    fn of_rows(self, n: usize) -> Vec<u32> {
        match self {
            Self::TenBalanced => (0..n).map(|row| u32::try_from(row % 10).expect("fits")).collect(),
            Self::EightUnbalanced => {
                let cycle: Vec<u32> = UNBALANCED_FORTIETHS
                    .iter()
                    .enumerate()
                    .flat_map(|(fold, fortieths)| {
                        std::iter::repeat_n(u32::try_from(fold).expect("fits"), *fortieths)
                    })
                    .collect();
                assert_eq!(cycle.len(), 40);
                (0..n).map(|row| cycle[row % 40]).collect()
            }
        }
    }
}

/// Columns `l0, a0, l1, a1, y` of the base design.
fn sequential_panel(n: usize, seed: u64) -> TabularData {
    let mut normal = gaussian(stream_seed(seed, 0x5E01));
    let mut unit = uniform(seed, 0x5E02);
    let mut history = uniform(seed, 0x5E03);
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let l0 = bounded(history());
        let a0 = f64::from(unit() < expit(0.5 * l0));
        let l1 = 0.5 * l0 + 0.8 * a0 + 0.5 * bounded(history());
        let a1 = f64::from(unit() < expit(0.5 * (l1 - 0.4)));
        let y = 1.0 + 0.5 * l0 + a0 + 1.5 * a1 + 0.4 * l1 + normal();
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
        ys.push(y);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned")
}

fn sequential_query(n: usize, folds: Folds) -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: folds.of_rows(n).into(),
        fold_count: folds.count(),
        min_probability: 0.05,
    }
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
}

/// The equal-tailed posterior interval of the effect column, as the posterior summarizes it.
fn posterior_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let column = posterior.effect_column()?;
    Some((posterior.summaries.q025[column], posterior.summaries.q975[column]))
}

/// The runtime's construction of a Frequentist execution with the interval method, the
/// standard-error kind and the dependence of the interval the route's fit publishes.
fn normal_interval_construction(reported: Reported) -> Reported {
    reported
        .into_iter()
        .take(1)
        .map(|(mut construction, scope, validation)| {
            construction.interval_method = NORMAL_INTERVAL.into();
            construction.se_kind = "crossfit_score".into();
            construction.dependence = "iid".into();
            construction.reported_level = REPORTED_LEVEL;
            (construction, scope, validation)
        })
        .collect()
}

/// One replicate that published an interval, with the runtime's constructions of the execution.
struct Scored {
    interval: (f64, f64),
    reported: Reported,
}

/// One sequential-contrast replicate; the Frequentist interval is the published Wald interval.
fn run_sequential_contrast(folds: Folds, seed: u64, bayesian_inference: bool) -> Option<Scored> {
    let n = grid_n(BASE_N);
    let data = sequential_panel(n, seed);
    let builder = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(sequential_query(n, folds)));
    let builder = if bayesian_inference { builder.inference(bayesian()) } else { builder };
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder.build().ok()?.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    let reported = constructions(&contract, &result);
    let (interval, reported) = if bayesian_inference {
        (posterior_interval(&result)?, reported)
    } else {
        let [lower, upper] = result.sequential_contrast()?.interval?;
        ((lower, upper), normal_interval_construction(reported))
    };
    Some(Scored { interval, reported })
}

/// Score `run` over the replicates against the record's interval method; a replicate that
/// publishes no interval is a skip, never a free pass.
fn measure(test: &'static str, folds: Folds, bayesian_inference: bool, seed: u64) {
    let interval = if bayesian_inference { POSTERIOR_INTERVAL } else { NORMAL_INTERVAL };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "sequential_panel", interval },
        REPORTED_LEVEL,
    );
    // Each grid point draws its own replicate data.
    let runs = map_replicates(n_sim(), |rep| {
        run_sequential_contrast(folds, grid_seed(stream_seed(seed, rep)), bayesian_inference)
    });
    for scored in runs {
        let Some(scored) = scored else {
            tally.skip();
            continue;
        };
        let (construction, scope, _) = scored
            .reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == interval)
            .unwrap_or_else(|| panic!("the execution reported no {interval} interval"));
        // A design route has no graph and runs no validation suite.
        let mut construction = construction.clone();
        construction.query = "SequentialContrast".into();
        construction.structure = "design".into();
        tally.bind_validation(&construction, *scope, "not_applicable");
        tally.record(Some(scored.interval), CONTRAST_TRUTH);
    }
    tally.assert();
}

/// The cross-fit Wald interval of the sequential contrast over ten balanced folds.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn sequential_crossfit_contrast_wald_interval_ten_folds_coverage() {
    measure(
        "sequential_crossfit_contrast_wald_interval_ten_folds_coverage",
        Folds::TenBalanced,
        false,
        0x5E1_0001,
    );
}

/// The cross-fit Wald interval of the sequential contrast over eight unbalanced folds.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn sequential_crossfit_contrast_wald_interval_unbalanced_folds_coverage() {
    measure(
        "sequential_crossfit_contrast_wald_interval_unbalanced_folds_coverage",
        Folds::EightUnbalanced,
        false,
        0x5E1_0002,
    );
}

/// The modular Bayesian-bootstrap posterior of the sequential contrast over ten balanced folds.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_crossfit_contrast_posterior_ten_folds_coverage() {
    measure(
        "bayesian_sequential_crossfit_contrast_posterior_ten_folds_coverage",
        Folds::TenBalanced,
        true,
        0x5E1_0003,
    );
}

/// The modular Bayesian-bootstrap posterior of the sequential contrast over eight unbalanced
/// folds.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_crossfit_contrast_posterior_unbalanced_folds_coverage() {
    measure(
        "bayesian_sequential_crossfit_contrast_posterior_unbalanced_folds_coverage",
        Folds::EightUnbalanced,
        true,
        0x5E1_0004,
    );
}
