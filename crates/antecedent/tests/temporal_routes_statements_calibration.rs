//! Coverage of a further statement of the sequential-regime route: the Bayesian-bootstrap
//! posterior of the sequential doubly robust regime value when the Q predictions are fitted on
//! the other folds of the same sample, not fixed functions of each subject's own history.
//!
//! The trial is the one the family's first harness file draws: two decisions, each a fair coin,
//! `l1 = .5 l0 + .8 a0 + .5 e` and `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e`, so the value of
//! the regime that treats at both decisions is `3.5 + .4 (.8) = 3.82`. Each subject's two Q
//! predictions come from a sequential least-squares regression fitted on the four folds the
//! subject is not in.
//!
//! The record is keyed on the design axes (structure `design`, validation `not_applicable`),
//! every other field of the construction is the runtime's. A run that publishes no posterior
//! interval is a miss; a refused run is a skip.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "test scaffolding indexes rows and folds with small integers"
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, LongitudinalRegimeMethod, LongitudinalRegimeQuery, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed, uniform,
};
use common::calibration_bind::constructions;

/// Interval method of the equal-tailed posterior interval.
const POSTERIOR_INTERVAL: &str = "posterior_quantile";

/// Posterior draws of the design.
const DRAWS: usize = 2_000;

/// Subjects at the base grid point.
const REGIME_BASE_N: usize = 400;

/// `E[Y(1, 1)]` of the regime that treats at both decisions: `1 + 1 + 1.5 + .4 (.8)`.
const REGIME_TRUTH: f64 = 3.5 + 0.32;

/// Folds of the cross-fit: a subject's predictions are fitted on the other four.
const FOLDS: usize = 5;

/// The equal-tailed posterior interval of the effect column, as the posterior summarizes it.
fn posterior_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let column = posterior.effect_column()?;
    Some((posterior.summaries.q025[column], posterior.summaries.q975[column]))
}

/// Least-squares coefficients of `y` on the columns of `x` (each row one observation), by the
/// normal equations with partial pivoting. `None` when the design is singular.
fn least_squares<const K: usize>(x: &[[f64; K]], y: &[f64]) -> Option<[f64; K]> {
    let mut gram = [[0.0_f64; K]; K];
    let mut moment = [0.0_f64; K];
    for (row, &target) in x.iter().zip(y) {
        for a in 0..K {
            moment[a] += row[a] * target;
            for b in 0..K {
                gram[a][b] += row[a] * row[b];
            }
        }
    }
    for pivot in 0..K {
        let best =
            (pivot..K).max_by(|&a, &b| gram[a][pivot].abs().total_cmp(&gram[b][pivot].abs()))?;
        if gram[best][pivot].abs() < 1e-12 {
            return None;
        }
        gram.swap(pivot, best);
        moment.swap(pivot, best);
        for row in pivot + 1..K {
            let factor = gram[row][pivot] / gram[pivot][pivot];
            for column in pivot..K {
                gram[row][column] -= factor * gram[pivot][column];
            }
            moment[row] -= factor * moment[pivot];
        }
    }
    let mut beta = [0.0_f64; K];
    for pivot in (0..K).rev() {
        let tail: f64 = (pivot + 1..K).map(|column| gram[pivot][column] * beta[column]).sum();
        beta[pivot] = (moment[pivot] - tail) / gram[pivot][pivot];
    }
    Some(beta)
}

fn dot<const K: usize>(beta: &[f64; K], row: &[f64; K]) -> f64 {
    beta.iter().zip(row).map(|(b, x)| b * x).sum()
}

/// One draw of the trial with cross-fit Q predictions, or `None` when a fold's regression is
/// singular.
///
/// For each fold, on the subjects outside it: the outcome is regressed on `(1, l0, a0, l1)`
/// among those treated at the second decision, which gives the second-decision Q under the
/// regime; that fitted Q is then regressed on `(1, l0)` among those treated at the first
/// decision, which gives the first-decision Q. Both models are correctly specified (the true
/// Q functions are `2.5 + .5 l0 + a0 + .4 l1` and `3.82 + .7 l0`), so the only departure from
/// the measured row is that the predictions are estimated from the same sample.
fn crossfit_regime_trial(n: usize, seed: u64) -> Option<(TabularData, LongitudinalRegimeQuery)> {
    let mut normal = gaussian(stream_seed(seed, 0x4E11));
    let mut unit = uniform(seed, 0x4E12);
    let (mut l0, mut l1, mut a0, mut a1, mut outcome) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let first = normal();
        let treated_first = unit() < 0.5;
        let second = 0.5 * first + 0.8 * f64::from(treated_first) + 0.5 * normal();
        let treated_second = unit() < 0.5;
        outcome.push(
            1.0 + 0.5 * first
                + f64::from(treated_first)
                + 1.5 * f64::from(treated_second)
                + 0.4 * second
                + normal(),
        );
        l0.push(first);
        l1.push(second);
        a0.push(treated_first);
        a1.push(treated_second);
    }
    let folds: Vec<u32> = (0..n).map(|i| (i % FOLDS) as u32).collect();
    let history = |i: usize| [1.0, l0[i], f64::from(a0[i]), l1[i]];
    let mut q = vec![0.0_f64; 2 * n];
    for fold in 0..FOLDS as u32 {
        let training: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let second_rows: Vec<usize> = training.iter().copied().filter(|&i| a1[i]).collect();
        let second_x: Vec<[f64; 4]> = second_rows.iter().map(|&i| history(i)).collect();
        let second_y: Vec<f64> = second_rows.iter().map(|&i| outcome[i]).collect();
        let second_beta = least_squares(&second_x, &second_y)?;
        let first_rows: Vec<usize> = training.iter().copied().filter(|&i| a0[i]).collect();
        let first_x: Vec<[f64; 2]> = first_rows.iter().map(|&i| [1.0, l0[i]]).collect();
        let first_y: Vec<f64> =
            first_rows.iter().map(|&i| dot(&second_beta, &history(i))).collect();
        let first_beta = least_squares(&first_x, &first_y)?;
        for i in (0..n).filter(|&i| folds[i] == fold) {
            q[2 * i] = dot(&first_beta, &[1.0, l0[i]]);
            q[2 * i + 1] = dot(&second_beta, &history(i));
        }
    }
    let treatment: Vec<bool> = (0..n).flat_map(|i| [a0[i], a1[i]]).collect();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())])
        .expect("the simulated column is finite");
    let query = LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: LongitudinalRegimeMethod::SequentialDoublyRobust,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: Some(Arc::from(q)),
        observation_history: Some(Arc::from(vec![true; 2 * n])),
        prediction_fold_ids: Some(Arc::from(folds.clone())),
        periods: 2,
        treatment_history: Arc::from(treatment),
        regime_actions: Arc::from(vec![true; 2 * n]),
        treatment_probabilities: Arc::from(vec![0.5; 2 * n]),
        censoring_probabilities: Arc::from(vec![1.0; 2 * n]),
        outcome_observed: Arc::from(vec![true; n]),
        subject_ids: (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect(),
        fold_ids: Arc::from(folds),
        excluded_fold_predictions: true,
        probabilities_known_by_design: true,
        minimum_probability: 0.1,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    Some((data, query))
}

/// What one replicate published: the posterior interval (absent when the run published none)
/// and the runtime's constructions of the execution.
struct Published {
    interval: Option<(f64, f64)>,
    reported: Vec<(Construction, ScopeFacts, &'static str)>,
}

/// Run the route on one draw; `None` when the draw is unevaluable or the route refuses it.
fn run_crossfit_regime(seed: u64) -> Option<Published> {
    let (data, query) = crossfit_regime_trial(grid_n(REGIME_BASE_N), seed)?;
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    Some(Published {
        interval: posterior_interval(&result),
        reported: constructions(&contract, &result),
    })
}

/// The Bayesian-bootstrap posterior of the sequential doubly robust regime value with Q
/// predictions fitted on the other folds of the same sample.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_regime_value_fitted_crossfit_posterior_coverage() {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "bayesian_sequential_regime_value_fitted_crossfit_posterior_coverage",
            dgp: "crossfit_regime_trial",
            interval: POSTERIOR_INTERVAL,
        },
        REPORTED_LEVEL,
    );
    // Each grid point draws its own replicate data.
    let runs =
        map_replicates(n_sim(), |rep| run_crossfit_regime(grid_seed(stream_seed(0x4E0_0011, rep))));
    for run in runs {
        let Some(published) = run else {
            tally.skip();
            continue;
        };
        let reported = published
            .reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == POSTERIOR_INTERVAL);
        match reported {
            Some((construction, scope, _)) => {
                // A design route has no graph and runs no validation suite.
                let mut construction = construction.clone();
                construction.query = "LongitudinalRegime".into();
                construction.structure = "design".into();
                tally.bind_validation(&construction, *scope, "not_applicable");
            }
            None => assert!(
                published.interval.is_none(),
                "an interval was published without a {POSTERIOR_INTERVAL} construction"
            ),
        }
        // A run that executed and published no interval is a miss.
        tally.record(published.interval, REGIME_TRUTH);
    }
    tally.assert();
}
