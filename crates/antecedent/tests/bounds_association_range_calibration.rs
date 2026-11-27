//! Coverage of the conjugate association slope interval at prior scales other than the one
//! `bounds_association_calibration.rs` measures.
//!
//! The row is stated for any prior scale under which the posterior mean of the slope lies within
//! a hundredth of its posterior standard deviation of the least-squares slope; the base design
//! uses the built-in scale 10 only. These designs repeat the base design (rows
//! `y = 1 + 2 x + e`, `x ~ N(0, 1)`, `e ~ N(0, 1)`, the population slope `2`, at `grid_n(200)`
//! rows: 100, 200 and 400) with two other prior scales, fixed before any run.
//!
//! The conjugate prior puts precision `var(x) / s^2` on the slope (`s` the prior scale, `var(x)`
//! the sample variance of the predictor), so the posterior mean is the least-squares slope
//! shrunk by about `1 / (n s^2 + 1)`, a shift of `beta / (n s^2)`, while the posterior standard
//! deviation is about `1 / sqrt(n)`. The shift in posterior standard deviations is therefore
//! about `beta / (sqrt(n) s^2)`, which at the smallest sample (`n = 100`, `beta = 2`) is `0.2 /
//! s^2`. It meets the hundredth at `s^2 >= 20`, that is `s >= 4.47`.
//!
//! * scale 6, the smallest scale that holds the bound at every replicate of the smallest
//!   sample: it is a shift of `0.0056` standard deviations in expectation at 100 rows, and the
//!   boundary scale 4.47 would leave a share of replicates over the hundredth through the
//!   spread of the estimated slope, spread and predictor variance, which the row would then
//!   not state;
//! * scale 1000, a prior far wider than the data, a shift of about `2 * 10^-7` standard
//!   deviations.
//!
//! The row is stated per fit, so a replicate counts only when its result states it; a replicate
//! that publishes no interval or states no row is a skip, never a free pass. The runtime
//! reports no interval method for the association query, so the fields of a record's
//! construction that the runtime reports are read from its own calibration key and the
//! interval method, standard-error kind, dependence and functional are set here. The route is
//! graphless, so a record is keyed on the design axes (structure `design`, validation
//! `not_applicable`).
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{
    AssociationFunctional, AssociationQuery, BayesianConfig, CausalQuery, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::{bind_reported, constructions};

/// One replicate's construction, scope facts and validation mode as the bind helpers take them.
type Reported = (Construction, ScopeFacts, &'static str);

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

const TABLE_ROWS: usize = 200;
const DRAWS: usize = 400;
const SLOPE: f64 = 2.0;
const ASSOCIATION_FUNCTIONAL: &str = "linear_regression_slope";
const DGP: &str = "homoskedastic_line_known_slope";
const INTERVAL: &str = "posterior_equal_tailed";

/// Rows `y = 1 + 2 x + e`, `x ~ N(0, 1)`, `e ~ N(0, 1)`.
fn homoskedastic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 5));
    let mut draw_e = gaussian(stream_seed(seed, 6));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x.iter().map(|x| 1.0 + SLOPE * x + draw_e()).collect();
    (x, y)
}

/// The posterior interval one simulated study publishes for the slope under `prior_scale`, with
/// its construction; `None` when the result states none or does not state its support row.
fn slope_interval(n: usize, seed: u64, prior_scale: f64) -> Option<([f64; 2], Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let (x, y) = homoskedastic_line_known_slope(n, seed);
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
        .expect("the simulated rows are finite and aligned");
    let query =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Coefficients);
    let study = Study::tabular(data.clone())
        .query(CausalQuery::Association(query))
        .refute(RefuteSuite::None)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(prior_scale),
        ))
        .build()
        .expect("the association study is valid");
    let prepared = study.prepare(&ctx).expect("the association study prepares");
    let result = prepared.estimate(&data, &ctx).expect("the association runs");
    let payload = result.association().filter(|_| result.states_section_row())?;
    assert_eq!(payload.estimate.terms.len(), 2, "an intercept and one slope");
    let summary = payload.estimate.terms[1].posterior?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    let (base, scope, _) = constructions(&contract, &result)
        .into_iter()
        .next()
        .expect("the runtime reports a calibration basis");
    let construction = Construction {
        structure: "design".into(),
        interval_method: INTERVAL.into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        functional: ASSOCIATION_FUNCTIONAL.into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
        ..base
    };
    let scope = ScopeFacts {
        replicates_ok: None,
        posterior_draws: payload
            .estimate
            .posterior
            .filter(|record| record.draws > 0)
            .and_then(|record| u32::try_from(record.draws).ok()),
        ..scope
    };
    Some(([summary.lower, summary.upper], (construction, scope, "not_applicable")))
}

fn measure_prior_scale(test: &'static str, prior_scale: f64, seed_base: u64) {
    let n = grid_n(TABLE_ROWS);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: DGP, interval: INTERVAL }, REPORTED_LEVEL);
    let runs =
        map_replicates(n_sim(), |rep| slope_interval(n, grid_seed(seed_base + rep), prior_scale));
    for run in runs {
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), SLOPE);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_slope_credible_interval_prior_scale_six() {
    measure_prior_scale(
        "bayesian_association_static_slope_credible_interval_prior_scale_six",
        6.0,
        317_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_slope_credible_interval_prior_scale_one_thousand() {
    measure_prior_scale(
        "bayesian_association_static_slope_credible_interval_prior_scale_one_thousand",
        1000.0,
        318_000,
    );
}
