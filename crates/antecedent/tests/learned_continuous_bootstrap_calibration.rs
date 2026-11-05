//! Coverage of the joint outer percentile-bootstrap interval of the learned continuous-outcome
//! trial transport estimate (`joint_outer_bootstrap`) against known truth.
//!
//! The analytic influence interval of the same estimator is measured in
//! `learned_continuous_calibration.rs`; this file measures the other interval the estimator
//! reports when a bootstrap is requested. The procedure scored is the whole composed one
//! (`antecedent_estimate::learned_continuous_interval_internal`): cross-fitted outcome and
//! membership nuisances, the overlap gate, the augmented inverse-odds score, and the
//! percentile interval of the `REPLICATES` outer refits that resample whole units under the
//! declared design (trial and target rows jointly for a nested cohort, each sample for
//! independent samples) and refit every nuisance in each replicate.
//!
//! Design and truth: the `Good` scenario of `learned_continuous_dgp` (both nuisance families
//! correctly specified, good overlap) under both sampling designs, with the exactly known
//! target mean contrast `2.5` derived in that module's documentation. Sizes are on the
//! three-point grid around 300 trial and 200 target rows (`grid_n`), `n_sim()` datasets per
//! design, 199 bootstrap replicates (the replicate floor of the cell). A dataset the
//! estimator refuses, or whose bootstrap withholds the interval, is a skip and counts as a
//! miss in the tally, never a free pass.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge. Nothing here is tuned to the
//! measured outcome: the percentile construction is the one the estimator ships, and a
//! below-band result is recorded as a named boundary rather than adjusted.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "learned_continuous_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    LearnedContinuousOptions, LearnerSpec, LinearSpec, learned_continuous_interval_internal,
};
use antecedent_identify::TransportIdentifier;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed,
};
use dgp::{Design, Scenario, draw, graph};

const INTERVAL: &str = "percentile_bootstrap";
/// Bootstrap replicates per execution: the replicate floor of the cell.
const REPLICATES: u32 = 199;

fn construction() -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.learned_trial_aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_mean_contrast".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn options() -> LearnedContinuousOptions {
    LearnedContinuousOptions {
        outcome: LearnerSpec::Linear(LinearSpec::default()),
        folds: 3,
        bootstrap: REPLICATES,
        coverage_level: REPORTED_LEVEL,
        ..LearnedContinuousOptions::default()
    }
}

/// One execution of the whole estimator: the bootstrap interval and the replicates that
/// completed, or `None` when it refuses or withholds the interval.
fn execute(design: Design, sizes: (usize, usize), rep: u64) -> Option<((f64, f64), u32)> {
    let (diagram, query, _) = graph();
    let id = TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let input = draw(design, Scenario::Good, sizes.0, sizes.1, stream_seed(rep, 0x4C42));
    let ctx = ExecutionContext::for_tests(rep);
    let run = learned_continuous_interval_internal(&id, &input, &options(), &ctx).ok()?;
    let ok = u32::try_from(run.replicates.len()).unwrap_or(u32::MAX);
    run.interval.map(|band| (band, ok))
}

fn record(design: Design, test: &'static str, dgp_name: &'static str) {
    let sizes = (grid_n(300), grid_n(200));
    let truth = Scenario::Good.truth();
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: dgp_name, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    let rows = map_replicates(n_sim(), |rep| execute(design, sizes, rep));
    for row in rows {
        match row {
            Some((band, replicates_ok)) => {
                tally.bind_validation(
                    &construction(),
                    ScopeFacts {
                        row_count: (sizes.0 + sizes.1) as u64,
                        replicates_ok: Some(replicates_ok),
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                    "not_applicable",
                );
                tally.record(Some(band), truth);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn learned_trial_aipw_nested_cohort_joint_bootstrap_mean_contrast() {
    record(
        Design::NestedCohort,
        "learned_trial_aipw_nested_cohort_joint_bootstrap_mean_contrast",
        "learned_continuous_nested_cohort_good_overlap_bootstrap",
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn learned_trial_aipw_independent_samples_joint_bootstrap_mean_contrast() {
    record(
        Design::IndependentSamples,
        "learned_trial_aipw_independent_samples_joint_bootstrap_mean_contrast",
        "learned_continuous_independent_samples_good_overlap_bootstrap",
    );
}
