//! Coverage designs for the normal-approximation interval of randomized effects with missing
//! outcomes: responder weighting under outcomes missing at random given a baseline covariate,
//! and the responder difference in means under outcomes missing completely at random.
//!
//! The routes publish no interval until a record measures one. These designs score the
//! candidate construction `effect ± z_{0.975} · sqrt(variance)` from the estimator that
//! backs each route, on 600 participants (the grid varies the size), at effect 0.3 with 15%
//! of outcomes missing at random on the baseline. They are `#[ignore]`d measurement designs
//! registered in `scripts/gate_calibration.sh` (`run_mo`); the coverage record is the only
//! missing piece. The fixed-seed truth checks of the same estimators are the unit tests of
//! `crates/antecedent-estimate/src/randomized_missing.rs` and
//! `crates/antecedent/tests/missing_data_routes.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_estimate::randomized_missing::responder_weighted_itt;
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95, grid_n,
    map_replicates, n_sim,
};

const INTERVAL: &str = "analytic_se";
const TRUTH: f64 = 0.3;

/// Which missingness mechanism a replicate draws.
#[derive(Clone, Copy)]
enum Mechanism {
    /// Responding depends on the baseline covariate, differently by arm.
    AtRandomGivenBaseline,
    /// Responding is independent of everything (15% missing).
    CompletelyAtRandom,
}

/// One replicate: the scored interval and the row count, or `None` when the fit refuses.
fn replicate(n: usize, seed: u64, mechanism: Mechanism) -> Option<((f64, f64), u64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x4D01);
    let (mut y, mut seen, mut arm, mut x) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let treated = i % 2 == 1;
        let baseline = standard_normal(&mut rng);
        y.push(
            1.0 + TRUTH * f64::from(u8::from(treated)) + 1.2 * baseline + standard_normal(&mut rng),
        );
        let p = match mechanism {
            Mechanism::AtRandomGivenBaseline => {
                let eta = 1.7 - (0.6 + 0.3 * f64::from(u8::from(treated))) * baseline;
                1.0 / (1.0 + (-eta).exp())
            }
            Mechanism::CompletelyAtRandom => 0.85,
        };
        seen.push(rng.next_f64() < p);
        arm.push(treated);
        x.push(baseline);
    }
    let covariates: &[f64] = match mechanism {
        Mechanism::AtRandomGivenBaseline => &x,
        Mechanism::CompletelyAtRandom => &[],
    };
    let ncov = usize::from(matches!(mechanism, Mechanism::AtRandomGivenBaseline));
    let fit = responder_weighted_itt(&y, &seen, &arm, covariates, ncov).ok()?;
    let radius = Z95 * fit.variance.sqrt();
    Some(((fit.effect - radius, fit.effect + radius), n as u64))
}

fn construction(estimator: &str) -> Construction {
    Construction {
        query: "RandomizedEffect".into(),
        graph_class: "RandomizedTrial".into(),
        structure: "design".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: INTERVAL.into(),
        se_kind: "influence".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "randomized_itt".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn measure(test: &'static str, dgp: &'static str, estimator: &str, mechanism: Mechanism) {
    let n = grid_n(600);
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    for row in map_replicates(n_sim(), |rep| replicate(n, 71_000 + rep, mechanism)) {
        let Some((interval, rows)) = row else {
            tally.skip();
            continue;
        };
        tally.bind(
            &construction(estimator),
            ScopeFacts {
                row_count: rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(Some(interval), TRUTH);
    }
    tally.assert();
}

/// The scored interval is finite, ordered and centred near the truth.
#[test]
fn the_scored_intervals_are_ordered_and_centred_near_the_truth() {
    for mechanism in [Mechanism::AtRandomGivenBaseline, Mechanism::CompletelyAtRandom] {
        let ((lo, hi), rows) = replicate(4_000, 71_001, mechanism).expect("a responder fit");
        assert_eq!(rows, 4_000);
        assert!(lo.is_finite() && hi.is_finite() && lo < hi);
        let centre = (lo + hi) / 2.0;
        // The half-width is about 1.96 se with se near 0.04 at 4000 rows; the centre is the effect.
        assert!((centre - TRUTH).abs() < 0.1, "centre {centre}");
        assert!(hi - lo > 0.05 && hi - lo < 0.4, "width {}", hi - lo);
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn responder_weighted_mar_interval() {
    measure(
        "responder_weighted_mar_interval",
        "randomized_trial_attrition_mar_on_baseline",
        "randomized_responder_weighted",
        Mechanism::AtRandomGivenBaseline,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn responder_complete_case_mcar_interval() {
    measure(
        "responder_complete_case_mcar_interval",
        "randomized_trial_attrition_mcar",
        "randomized_responder_complete_case",
        Mechanism::CompletelyAtRandom,
    );
}
