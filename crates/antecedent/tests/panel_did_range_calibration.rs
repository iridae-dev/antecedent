//! Coverage designs for the parts of two panel difference-in-differences support rows that the
//! designs of `panel_did_calibration.rs` do not reach: the two-period posterior at the smallest
//! group size its row admits, and the augmented panel with predictions fitted on the analysed
//! rows.
//!
//! Each design is the matching design of `panel_did_calibration.rs` with one thing changed:
//!
//! * two-period posterior: the row admits 22 subjects per group (44 rows). The base design
//!   draws `grid_n(30)` per group, whose smallest point is 23. This design draws
//!   `grid_n(29)`, whose points are 22, 29 and 58 subjects per group, so the smallest point
//!   is exactly the row's floor. Everything else is unchanged: independent subjects,
//!   homoskedastic normal changes, the difference in mean changes is `2`, prior scale 100,
//!   2,000 draws, the equal-tailed 95% posterior interval of the effect column.
//! * augmented panel without the declaration: the row is stated for supplied predictions with
//!   or without the declaration that they were not fitted on the analysed rows. The route
//!   only records the declaration; it computes the same interval either way. Undeclared
//!   predictions are by definition fitted on the rows analysed, so this design fits them there.
//!   The data law is the base design's: propensity `0.25 + 0.5 x`, untreated change `x`, effect
//!   on the treated `1.5`, `x` uniform on `(0, 1)`. The propensity is the ordinary least
//!   squares fit of treatment on `x` (the linear probability model, correctly specified here),
//!   clipped to `[0.25, 0.75]`, the interval the true propensity lies in, because the row
//!   admits a supplied propensity only inside it. The untreated-change prediction is the
//!   least squares fit of the change on `x` among the untreated rows (the change is `x` plus
//!   noise there, so the line is correctly specified). Both fits use every analysed row.
//!
//! A replicate that publishes no interval is a skip, never a free pass. A design route has no
//! graph and runs no validation suite, so every record is keyed on the design axes: the
//! query's name, structure `design`, validation `not_applicable`.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, Study, StudyResult};
use antecedent_core::{CausalRng, DidResamplingScheme, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;

/// Interval method of the resampled route's pointwise interval.
const RESAMPLED: &str = "bootstrap_se";
/// Interval method of the Bayesian route.
const POSTERIOR: &str = "posterior_quantile";
/// Whole-cluster resamples per replicate of the augmented panel.
const RESAMPLES: u32 = 199;
/// Posterior draws per replicate: the count the posterior support row is stated from.
const DRAWS: usize = 2_000;
/// Query name of every record.
const QUERY: &str = "PanelDid";
/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";
/// Effect on the treated of the augmented panel design.
const AUGMENTED_TRUTH: f64 = 1.5;
/// Difference in mean changes of the two-period Bayesian design.
const TWO_PERIOD_TRUTH: f64 = 2.0;
const OUTCOME: VariableId = VariableId::from_raw(0);

/// One replicate's scored interval and the construction the runtime reported for it.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: Construction,
    scope: ScopeFacts,
}

/// Score `interval` against `truth` under the runtime's primary key for `result`, with the
/// interval fields a resampled route does not report declared as the pointwise interval of a
/// whole-cluster resample.
fn resampled(
    study: &Study,
    result: &StudyResult,
    interval: [f64; 2],
    truth: f64,
) -> Option<Scored> {
    let (mut construction, scope, _) =
        constructions(&study.inspect().ok()?, result).into_iter().next()?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = RESAMPLED.into();
    construction.se_kind = String::new();
    construction.dependence = "cluster".into();
    construction.reported_level = REPORTED_LEVEL;
    Some(Scored { interval: (interval[0], interval[1]), truth, construction, scope })
}

/// Score `interval` against `truth` under the runtime's posterior-quantile key for `result`.
fn posterior(
    study: &Study,
    result: &StudyResult,
    interval: (f64, f64),
    truth: f64,
) -> Option<Scored> {
    let (mut construction, scope, _) = constructions(&study.inspect().ok()?, result)
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == POSTERIOR)?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    Some(Scored { interval, truth, construction, scope })
}

/// Measure one design: every replicate is bound to the record's construction and scored.
fn measure(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    replicate: impl Fn(u64) -> Option<Scored> + Sync,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    for row in map_replicates(n_sim(), replicate) {
        // A replicate that publishes no interval is a skip (a miss), never a free pass.
        let Some(scored) = row else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&scored.construction, scored.scope, NO_VALIDATION);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
}

// -------------------------------------------------- two-period posterior at the row floor

/// Base subjects per group of this design. The grid is `3/4`, `1` and `2` times it, rounded
/// to nearest: 22, 29 and 58, so the smallest point is the 22 subjects per group (44 rows)
/// the row starts at.
const FLOOR_BASE_PER_ARM: usize = 29;

/// Normal-Inverse-Gamma posterior of the difference in mean changes of independent units:
/// the equal-tailed 95% interval of the effect column.
fn two_period_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(910_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let per_arm = grid_n(FLOOR_BASE_PER_ARM);
    let (mut subjects, mut treated, mut post, mut outcome) = (vec![], vec![], vec![], vec![]);
    for unit in 0..2 * per_arm {
        let active = unit >= per_arm;
        let before = 1.0 + standard_normal(&mut rng);
        let change =
            0.7 + if active { TWO_PERIOD_TRUTH } else { 0.0 } + 0.5 * standard_normal(&mut rng);
        for after in [false, true] {
            subjects.push(Arc::<str>::from(format!("s{unit}")));
            treated.push(active);
            post.push(after);
            outcome.push(if after { before + change } else { before });
        }
    }
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let clusters = subjects.clone();
    let query = PanelDidQuery::new(OUTCOME, treated, post, subjects, clusters);
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(100.0).n_draws(DRAWS),
        ))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let posterior_fit = result.posterior.as_ref()?;
    let column = posterior_fit.effect_column()?;
    let interval = (posterior_fit.summaries.q025[column], posterior_fit.summaries.q975[column]);
    posterior(&study, &result, interval, TWO_PERIOD_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_two_period_posterior_interval_from_the_row_floor() {
    measure(
        "bayesian_two_period_posterior_interval_from_the_row_floor",
        "two_period_independent_units_homoskedastic_normal_changes_known_difference_from_22_per_group",
        POSTERIOR,
        two_period_interval,
    );
}

// ------------------------------------- augmented panel, predictions fitted on the rows

/// Intercept and slope of the least squares line of `y` on `x` over the rows `keep` selects.
fn line(x: &[f64], y: &[f64], keep: impl Fn(usize) -> bool) -> (f64, f64) {
    let rows: Vec<usize> = (0..x.len()).filter(|&row| keep(row)).collect();
    let count = rows.len() as f64;
    let mean_x = rows.iter().map(|&row| x[row]).sum::<f64>() / count;
    let mean_y = rows.iter().map(|&row| y[row]).sum::<f64>() / count;
    let sxx: f64 = rows.iter().map(|&row| (x[row] - mean_x).powi(2)).sum();
    let sxy: f64 = rows.iter().map(|&row| (x[row] - mean_x) * (y[row] - mean_y)).sum();
    let slope = sxy / sxx;
    (mean_y - slope * mean_x, slope)
}

/// The base design's panel with the propensity and the untreated change fitted on the analysed
/// rows and supplied without the declaration that they were not fitted there.
fn augmented_in_sample_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(881_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let n = grid_n(400);
    let (mut pre, mut post, mut x, mut treated, mut ids) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let covariate = rng.next_f64();
        let d = rng.next_f64() < 0.25 + 0.5 * covariate;
        let base = standard_normal(&mut rng);
        pre.push(base);
        post.push(
            base + covariate
                + if d { AUGMENTED_TRUTH } else { 0.0 }
                + 0.2 * standard_normal(&mut rng),
        );
        x.push(covariate);
        treated.push(d);
        ids.push(Arc::<str>::from(format!("s{i:05}")));
    }
    let indicator: Vec<f64> = treated.iter().map(|d| f64::from(u8::from(*d))).collect();
    let change: Vec<f64> = post.iter().zip(&pre).map(|(after, before)| after - before).collect();
    let (p0, p1) = line(&x, &indicator, |_| true);
    let (m0, m1) = line(&x, &change, |row| !treated[row]);
    let p: Vec<f64> = x.iter().map(|x| (p0 + p1 * x).clamp(0.25, 0.75)).collect();
    let m: Vec<f64> = x.iter().map(|x| m0 + m1 * x).collect();
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", p.as_slice()),
        ("m0", m.as_slice()),
    ])
    .ok()?;
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        treated,
        ids.clone(),
        ids,
        false,
    )
    .with_resampling(DidResamplingScheme::ClusterBootstrap, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let interval = result.panel_did()?.interval_95?;
    resampled(&study, &result, interval, AUGMENTED_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn augmented_panel_in_sample_nuisance_bootstrap_interval() {
    measure(
        "augmented_panel_in_sample_nuisance_bootstrap_interval",
        "two_period_panel_propensity_and_untreated_change_fitted_on_the_analysed_rows_known_effect_on_treated",
        RESAMPLED,
        augmented_in_sample_interval,
    );
}
