//! Repeated-sampling coverage of the trial-to-target transported effect on one serially
//! dependent series: the Newey-West interval (`transport.trial_ipw`) and the circular-block
//! Dirichlet posterior (`transport.trial_bayesian_bootstrap`) a temporal ADMG study publishes.
//!
//! The table is one series of consecutive rows. Trial and target rows alternate, so the trial
//! and target sizes are fixed and equal, and the supplied selection probability (one half) and
//! treatment probability (one half) are exact. Two dependence sources reach the per-row
//! influence of the contrast:
//!
//! * the arm is held in runs of `RUN` consecutive rows, each run an independent fair coin;
//! * the outcome noise is stationary AR(1) with coefficient `RHO` and unit marginal variance.
//!
//! The outcome is `y = EFFECT * a + noise` on trial rows (target rows carry no treatment or
//! outcome). The transported effect is the constant `EFFECT`: it is the same in the trial and
//! the target population, and the estimator's expectation is `(n_trial / n_target) * EFFECT`
//! with the two sizes equal. The dependence strength is fixed here, ahead of any measurement.
//!
//! The driver scores the reported 0.95 interval and the 0.90 interval of the same execution: a
//! normal interval from the published Newey-West standard error, a posterior-quantile interval
//! re-summarized from the retained draws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "row indexes are small integers")]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, CausalPosterior, InferenceMode, RefuteSuite, Study, TransportTrialSpec,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};
use common::calibration::{ar1_noise, grid_n, stream_seed, unit_uniform};
use common::reported::{posterior_pair, scalar_normal_pair};
use grid_cell::{Cell, Target, measure};

/// Rows of the series at the base grid point.
const ROWS: usize = 1000;
/// Consecutive rows over which one arm is held.
const RUN: usize = 10;
/// AR(1) coefficient of the outcome noise.
const RHO: f64 = 0.5;
/// The constant transported effect.
const EFFECT: f64 = 1.0;
/// Posterior draws of the Bayesian execution.
const DRAWS: usize = 1_000;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// One series of `rows` consecutive times; odd rows are trial rows, even rows target rows.
fn transport_series(rows: usize, seed: u64) -> TabularData {
    let noise = ar1_noise(rows, RHO, 1.0, stream_seed(seed, 0x4E01));
    let arm_of_run = |run: usize| {
        f64::from(u8::from(unit_uniform(stream_seed(seed, 0x5A00 + run as u64)) < 0.5))
    };
    let (mut a, mut y, mut trial) = (Vec::new(), Vec::new(), Vec::new());
    for (t, noise) in noise.iter().enumerate() {
        let in_trial = t % 2 == 1;
        let arm = if in_trial { arm_of_run(t / RUN) } else { 0.0 };
        a.push(arm);
        y.push(if in_trial { EFFECT * arm + noise } else { 0.0 });
        trial.push(f64::from(u8::from(in_trial)));
    }
    let half = vec![0.5; rows];
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("trial", trial.as_slice()),
        ("s", half.as_slice()),
        ("e", half.as_slice()),
    ])
    .unwrap()
}

/// `a -> y` over the five transport columns, lifted to a temporal ADMG.
fn temporal_graph() -> TemporalAdmg {
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    TemporalAdmg::from_static_contemporaneous(&admg).unwrap()
}

fn measure_transport(test: &'static str, family: u64, bayesian: bool) {
    let graph = temporal_graph();
    let (estimator, interval, inference) = if bayesian {
        (
            "transport.trial_bayesian_bootstrap",
            "posterior_quantile",
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(DRAWS)),
        )
    } else {
        ("transport.trial_ipw", "analytic_se", InferenceMode::Frequentist)
    };
    let cell = Cell { test, dgp: "transport_series_ar1_runs", interval, estimator, variants: 1 };
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let query = CausalQuery::Transport(TransportQuery::new(response, "trial", "target", [v(0)]));
    measure(
        cell,
        &Target::scalar(EFFECT),
        |rep, _| {
            let seed = stream_seed(family, rep);
            let study = Study::tabular(transport_series(grid_n(ROWS), seed))
                .graph(graph.clone())
                .query(query.clone())
                .selection_targets(Arc::from([]))
                .transport_trial(TransportTrialSpec {
                    trial: v(2),
                    selection_probability: v(3),
                    treatment_probability: v(4),
                })
                .inference(inference.clone())
                .bootstrap_replicates(0)
                .refute(RefuteSuite::None)
                .build()
                .unwrap_or_else(|e| panic!("{test}: {e}"));
            let result = study
                .run(&ExecutionContext::for_tests(seed))
                .unwrap_or_else(|e| panic!("{test}: {e}"));
            (study, result)
        },
        |result| {
            if bayesian {
                vec![
                    result
                        .posterior
                        .as_ref()
                        .and_then(CausalPosterior::effect_column)
                        .map_or([None, None], |column| posterior_pair(result, column)),
                ]
            } else {
                vec![scalar_normal_pair(result)]
            }
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn transport_series_newey_west_interval_coverage() {
    measure_transport("transport_series_newey_west_interval_coverage", 0x7C01, false);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn transport_series_block_posterior_coverage() {
    measure_transport("transport_series_block_posterior_coverage", 0x7C02, true);
}
