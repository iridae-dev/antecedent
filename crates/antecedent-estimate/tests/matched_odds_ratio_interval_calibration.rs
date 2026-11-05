//! Whole-method coverage of the Wald interval of the conditional odds ratio of 1:1 matched
//! case-control sets, against the known conditional odds ratio of the generating law.
//!
//! The procedure scored is `conditional_odds_ratio` end to end and the log-scale Wald interval
//! `log_odds_ratio_interval(0.95)` from the observed information over the informative sets
//! (the exponentiated interval is a monotone map of the same limits, so its coverage of the odds
//! ratio is the coverage of the log odds ratio). Its construction is pinned by the closed form
//! `1/b + 1/c` in `matched_case_control.rs`; this file measures whether the interval covers.
//!
//! Design and truth: each of `n` pairs has its own baseline exposure probability
//! `p = 0.2 + 0.6 u` (`u` uniform), the control is exposed with probability `p`, and the case is
//! exposed with the odds `theta * p / (1 - p)` for `theta = 2`. Given a discordant pair, the
//! probability that the case is the exposed member is exactly `theta / (1 + theta)` whatever
//! the pair's baseline, which is the conditional likelihood the estimator maximizes, so the
//! conditional log odds ratio is exactly `ln 2`. Pairs on the three-point grid around 300
//! (150, 300, 600), `n_sim()` datasets; roughly 40 percent of pairs are discordant. A dataset
//! with no informative set is refused by the estimator and counts as a skip (a miss).
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge. The interval rests on asymptotic
//! normality of the conditional score; no small-sample exactness is claimed.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::ExecutionContext;
use antecedent_estimate::conditional_odds_ratio;
use calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
    stream_seed, unit_uniform,
};

const LEVEL: f64 = 0.95;
const BASE_PAIRS: usize = 300;
const THETA: f64 = 2.0;
const SEED_BASE: u64 = 0x3A7C_0001;

/// The 95% log-scale Wald interval of one simulated study of `n` matched pairs, or `None`
/// when the estimator refuses it.
fn interval(n: usize, seed: u64) -> Option<(f64, f64)> {
    let mut stratum = Vec::with_capacity(2 * n);
    let mut case = Vec::with_capacity(2 * n);
    let mut exposed = Vec::with_capacity(2 * n);
    for i in 0..n {
        let draw = |tag: u64| unit_uniform(stream_seed(seed, 3 * i as u64 + tag));
        let baseline = 0.2 + 0.6 * draw(0);
        let control_exposed = draw(1) < baseline;
        let odds = THETA * baseline / (1.0 - baseline);
        let case_exposed = draw(2) < odds / (1.0 + odds);
        for (y, x) in [(1.0, case_exposed), (0.0, control_exposed)] {
            stratum.push(format!("pair-{i}"));
            case.push(y);
            exposed.push(f64::from(x));
        }
    }
    let fit = conditional_odds_ratio(&stratum, &case, &exposed, &ExecutionContext::for_tests(seed))
        .ok()?;
    fit.log_odds_ratio_interval(LEVEL)
}

fn construction() -> Construction {
    Construction {
        query: "MatchedOddsRatio".into(),
        graph_class: "NotApplicable".into(),
        structure: "fixed".into(),
        modality: "matched_case_control_sets".into(),
        inference: "Frequentist".into(),
        estimator: "matched.conditional_odds_ratio".into(),
        interval_method: "wald_normal".into(),
        se_kind: "observed_information".into(),
        dependence: "independent_sets".into(),
        posterior: String::new(),
        functional: "log_conditional_odds_ratio".into(),
        identification: "point".into(),
        reported_level: LEVEL,
    }
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn matched_pairs_conditional_odds_ratio_wald_interval() {
    let n = grid_n(BASE_PAIRS);
    let truth = THETA.ln();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "matched_pairs_conditional_odds_ratio_wald_interval",
            dgp: "matched_pairs_heterogeneous_baseline_known_conditional_odds_ratio",
            interval: "wald_normal",
        },
        LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| interval(n, grid_seed(SEED_BASE + rep)));
    for run in runs {
        let Some(band) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction(),
            ScopeFacts {
                row_count: 2 * n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(band), truth);
    }
    tally.assert();
}
