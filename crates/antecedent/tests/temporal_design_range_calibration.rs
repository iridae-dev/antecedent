//! Coverage of the regime-survival intervals at the horizons its support rows admit and
//! `temporal_design_calibration.rs` does not measure.
//!
//! The rows are stated for any horizon up to three periods; the base designs measure a horizon
//! of three only. These designs repeat the base survival design with a horizon of one and of
//! two periods and change nothing else: subject trajectories with one continuous time-varying
//! confounder, treatment `expit(0.8 L)` given the current confounder, a hazard
//! `expit(-2.5 + 0.8 L - 0.7 A)` and about three percent censoring per period, `L_0` uniform
//! on `(-1, 1)` and `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + (u - 0.5)` with `u` uniform on `(0, 1)`.
//!
//! The scored coordinate is the always-treat minus never-treat survival difference at the
//! horizon `h`. Under a static regime `a` the confounder path is the recursion above with
//! `A_k = a`, and survival to the horizon is `S_a(h) = E[prod_{k < h} (1 - hazard_k)]`, the
//! expectation over `L_0` and the noise. The truth is `S_1(h) - S_0(h)`, computed by
//! simulating that law (two million paths per regime, the way the base design computes its
//! value; the Monte Carlo error is below `1e-4`, far under the intervals' half-width). For a
//! horizon of one it reduces to `E[expit(-2.5 + 0.8 L_0)] - E[expit(-3.2 + 0.8 L_0)]`.
//!
//! A frequentist result carries no interval binding in the runtime's calibration key, so its
//! record declares the percentile interval it publishes beside the construction the runtime
//! does report; the Bayesian record reads the interval method from the runtime. Every record is
//! keyed on the design axes (structure `design`, validation `not_applicable`). A replicate the
//! route refuses is a skip, a miss.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalAssignment, TemporalDesignFunctional,
    TemporalDesignQuery, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, uniform,
};
use common::calibration_bind::constructions;

/// Whole-procedure bootstrap replicates of the frequentist route: the route's replicate floor.
const BOOTSTRAP_REPLICATES: u32 = 99;
/// Posterior draws of the Bayesian route.
const DRAWS: usize = 500;
/// Subjects at the base grid point.
const BASE_SUBJECTS: usize = 1200;
const DGP: &str = "time_varying_confounding_logistic_hazard_known_survival_difference";

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied())
        .expect("the simulated columns are finite and aligned")
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Confounder at period `k`: `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + (u - 0.5)`.
fn next_confounder(current: f64, previous_treated: bool, noise: f64) -> f64 {
    0.5 * current - 0.4 * f64::from(u8::from(previous_treated)) + (noise - 0.5)
}

/// Subject trajectories with time-varying confounding over `horizon` periods: treatment is more
/// likely when the confounder is high, the hazard is `expit(-2.5 + 0.8 L - 0.7 A)` and
/// censoring is light.
fn survival(subjects: usize, horizon: u32, seed: u64) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D07);
    let (mut unit_labels, mut periods) = (Vec::new(), Vec::new());
    let (mut events, mut treated, mut censored, mut confounder) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..subjects {
        let mut current = 2.0 * next() - 1.0;
        let mut previous = false;
        for k in 0..horizon {
            if k > 0 {
                current = next_confounder(current, previous, next());
            }
            let a = next() < expit(0.8 * current);
            let c = next() < 0.03;
            let y = !c && next() < expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(a)));
            unit_labels.push(format!("s{i}"));
            periods.push(i64::from(k));
            events.push(f64::from(u8::from(y)));
            treated.push(f64::from(u8::from(a)));
            censored.push(f64::from(u8::from(c)));
            confounder.push(current);
            previous = a;
            if c || y {
                break;
            }
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::RegimeSurvival {
            treatment: VariableId::from_raw(1),
            censored: VariableId::from_raw(2),
            confounders: Arc::from([VariableId::from_raw(3)]),
            horizon,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.02,
        outcome_misclassification: None,
    };
    (
        table(&[
            ("event", events.as_slice()),
            ("treated", treated.as_slice()),
            ("censored", censored.as_slice()),
            ("confounder", confounder.as_slice()),
        ]),
        query,
    )
}

/// Always-treat minus never-treat survival at `horizon`, by simulating the generating law under
/// each static regime (the hazards are multiplied out, so only the confounder path is random).
fn survival_truth(horizon: u32) -> f64 {
    let draws = 2_000_000_u32;
    let mut next = uniform(0x5A17_0001, 0x7D08);
    let mut totals = [0.0_f64; 2];
    for (index, arm) in [true, false].into_iter().enumerate() {
        for _ in 0..draws {
            let mut current = 2.0 * next() - 1.0;
            let mut surviving = 1.0;
            for k in 0..horizon {
                if k > 0 {
                    current = next_confounder(current, arm, next());
                }
                surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(arm)));
            }
            totals[index] += surviving;
        }
    }
    (totals[0] - totals[1]) / f64::from(draws)
}

/// One replicate: the survival difference's published interval and the construction the
/// execution is keyed under, or `None` when the study is refused or publishes none.
fn replicate(
    horizon: u32,
    bayesian: bool,
    seed: u64,
) -> Option<((f64, f64), Construction, ScopeFacts)> {
    let (data, query) = survival(grid_n(BASE_SUBJECTS), horizon, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
    } else {
        InferenceMode::Frequentist
    };
    let mut builder =
        Study::tabular(data.clone()).query(CausalQuery::TemporalDesign(query)).inference(inference);
    if !bayesian {
        builder = builder.bootstrap_replicates(BOOTSTRAP_REPLICATES);
    }
    let study = builder.build().ok()?;
    let prepared = study.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let fit = result.temporal_design()?;
    assert_eq!(
        fit.coordinates.first().map(ToString::to_string).as_deref(),
        Some("survival_difference")
    );
    let interval = (*fit.lower.first()?, *fit.upper.first()?);
    let contract = prepared.contract_for_result(&result).ok()?;
    let reported = constructions(&contract, &result);
    let (mut construction, mut scope, _) = if bayesian {
        reported.into_iter().find(|(c, _, _)| c.interval_method == "posterior_quantile")?
    } else {
        reported.into_iter().next()?
    };
    // A design route has no graph and runs no validation suite, so its record is keyed on
    // the design axes: structure `design`, no validation mode applies.
    construction.structure = "design".into();
    if !bayesian {
        construction.interval_method = "percentile_bootstrap".into();
        construction.se_kind = "percentile".into();
        construction.dependence = "cluster".into();
        construction.reported_level = REPORTED_LEVEL;
        scope.replicates_ok = fit.replicates.and_then(|(ok, _)| u32::try_from(ok).ok());
    }
    Some((interval, construction, scope))
}

/// Coverage of the survival difference's 95% interval over `n_sim()` panels.
fn measure(test: &'static str, horizon: u32, bayesian: bool, seed_base: u64) {
    let interval = if bayesian { "posterior_quantile" } else { "percentile_bootstrap" };
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: DGP, interval }, REPORTED_LEVEL);
    let truth = survival_truth(horizon);
    for run in map_replicates(n_sim(), |rep| replicate(horizon, bayesian, seed_base + rep)) {
        // A refusal or a missing interval is a miss, never a free pass.
        let Some((interval, construction, scope)) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&construction, scope, "not_applicable");
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_horizon_one_bootstrap_interval() {
    measure(
        "temporal_design_regime_survival_horizon_one_bootstrap_interval",
        1,
        false,
        0x7D21_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_horizon_two_bootstrap_interval() {
    measure(
        "temporal_design_regime_survival_horizon_two_bootstrap_interval",
        2,
        false,
        0x7D22_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_regime_survival_horizon_one_interval() {
    measure("bayesian_temporal_design_regime_survival_horizon_one_interval", 1, true, 0x7D23_0000);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_regime_survival_horizon_two_interval() {
    measure("bayesian_temporal_design_regime_survival_horizon_two_interval", 2, true, 0x7D24_0000);
}
