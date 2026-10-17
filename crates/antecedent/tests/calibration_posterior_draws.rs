//! Repeated-sampling coverage of the Bayesian linear-adjustment interval at supplied
//! posterior draw counts.
//!
//! The data are the linear Gaussian SCM `bayesian_linear_scm` (effect 0.4, 1,000 rows at
//! the base point of the sample-size grid): a confounded binary treatment adjusted for its
//! cause. Each design supplies a draw count (2000, 3000, 4000, 8000) under the posterior
//! draw controls and scores the facade's published 95% credible interval of the adjusted
//! effect against the known effect over at least 2000 replicates. The construction the
//! record describes (including the actual draw count) is bound from the runtime on every
//! scored replicate, so a record covers exactly the draws it was measured at.
//!
//! Beyond coverage, each design holds the precision the draw-based interval promises: at
//! least 90% of replicates report an interval, whose median width is at most 1.5 times the
//! effect, and which excludes zero in at least 75% of replicates (power against no effect).
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test scaffolding counts replicates and indexes a sorted sample"
)]

mod common;

use antecedent::{
    BayesianConfig, InferenceMode, PosteriorDrawControls, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, grid_n, map_replicates, stream_seed,
};
use common::calibration_bind::bind;
use common::reported::{n_sim_at_least, posterior_pair};
use common::static_dgp::bayesian_linear_scm;

/// The adjusted effect every replicate is generated with.
const EFFECT: f64 = 0.4;
/// Rows at the base point of the sample-size grid.
const ROWS: usize = 1000;
/// Replicates at which the design measures, unless the run sets its own count.
const MIN_REPLICATES: u32 = 2000;
/// Smallest share of replicates that must report an interval.
const MIN_REPORTING_RATE: f64 = 0.9;
/// Largest median interval width, as a multiple of the effect.
const MAX_MEDIAN_WIDTH_OVER_EFFECT: f64 = 1.5;
/// Smallest share of reported intervals that exclude zero.
const MIN_POWER: f64 = 0.75;

fn dag() -> Dag {
    let mut g = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

/// One replicate: the study, its result and the published 95% interval of the effect.
struct Replicate {
    study: Study,
    result: StudyResult,
    interval: (f64, f64),
}

/// A replicate at `draws` supplied draws, or `None` when the study is refused or reports no
/// interval (a skip, capped by the harness).
fn replicate(draws: usize, salt: u64, rep: u64) -> Option<Replicate> {
    let seed = stream_seed(salt, rep);
    let config = BayesianConfig::laplace().n_draws(draws);
    let study = Study::tabular(bayesian_linear_scm(grid_n(ROWS), EFFECT, seed))
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .draw_controls(PosteriorDrawControls::Measured)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let posterior = result.posterior.as_ref()?;
    assert_eq!(posterior.draws.n_draws, draws, "the interval is from the supplied draws");
    let interval = posterior_pair(&result, posterior.effect_column()?)[0]?;
    Some(Replicate { study, result, interval })
}

/// Score the interval at `draws` supplied draws over at least [`MIN_REPLICATES`] replicates
/// whose data derive from `salt`, and hold the precision targets.
fn coverage_at_draws(test: &'static str, draws: usize, salt: u64) {
    let replicates = n_sim_at_least(MIN_REPLICATES);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "crates/antecedent/tests/common/static_dgp.rs::bayesian_linear_scm",
            interval: "posterior_quantile",
        },
        REPORTED_LEVEL,
    );
    let (mut widths, mut excluding_zero) = (Vec::new(), 0_u32);
    for scored in map_replicates(replicates, |rep| replicate(draws, salt, rep)) {
        match scored {
            Some(Replicate { study, result, interval: (lower, upper) }) => {
                bind(&mut tally, &study, &result);
                tally.record(Some((lower, upper)), EFFECT);
                widths.push(upper - lower);
                excluding_zero += u32::from(lower > 0.0 || upper < 0.0);
            }
            None => tally.skip(),
        }
    }
    tally.assert();

    let reported = widths.len();
    let reporting_rate = reported as f64 / f64::from(replicates);
    assert!(
        reporting_rate >= MIN_REPORTING_RATE,
        "{draws} draws: {reporting_rate} of replicates reported an interval"
    );
    widths.sort_by(f64::total_cmp);
    let median = widths[reported / 2];
    assert!(
        median <= MAX_MEDIAN_WIDTH_OVER_EFFECT * EFFECT,
        "{draws} draws: median width {median} exceeds {MAX_MEDIAN_WIDTH_OVER_EFFECT} x the effect"
    );
    let power = f64::from(excluding_zero) / reported as f64;
    assert!(power >= MIN_POWER, "{draws} draws: power {power} against no effect");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn posterior_draws_2000_rows_1000_l95() {
    coverage_at_draws("posterior_draws_2000_rows_1000_l95", 2000, 3_272_793);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn posterior_draws_3000_rows_1000_l95() {
    coverage_at_draws("posterior_draws_3000_rows_1000_l95", 3000, 1_520_647);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn posterior_draws_4000_rows_1000_l95() {
    coverage_at_draws("posterior_draws_4000_rows_1000_l95", 4000, 5_482_399);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn posterior_draws_8000_rows_1000_l95() {
    coverage_at_draws("posterior_draws_8000_rows_1000_l95", 8000, 4_550_226);
}
