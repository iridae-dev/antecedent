//! Coverage of the learned continuous-outcome trial transport interval (2.2A cell X4)
//! against known truth.
//!
//! The estimator measured is the whole composed one the public route withholds:
//! cross-fitted outcome and membership nuisances, the augmented inverse-odds
//! score, the overlap gate, and the design-specific analytic influence variance
//! (`LearnedContinuousOptions::analytic_interval_internal`). The previously
//! proposed percentile bootstrap failed its sample-size grid and is not scored
//! as a licensed candidate. No component AIPW formula is scored on its own.
//!
//! * `learned_trial_aipw_nested_cohort_continuous_mean_contrast` and
//!   `learned_trial_aipw_independent_samples_continuous_mean_contrast` are the two
//!   coverage records: good overlap, both nuisance families correctly specified,
//!   sample-size grid `n/2, n, 2n` ([`grid_n`]), `n_sim()` datasets.
//! * `learned_trial_aipw_weak_overlap_boundary` measures the boundary: with a target
//!   far from the trial the estimator must refuse, not extrapolate (at least 90% of
//!   datasets refuse under both designs).
//! * `learned_trial_aipw_misspecified_nuisance_cases` measures each single-family
//!   misspecification the claimed model double robustness covers (wrong outcome model
//!   with correct membership, and the reverse). Double misspecification is outside the
//!   claim and is not measured.
//!
//! No simultaneous or CATE-wide claim is scored: each execution is one mean contrast.
//! The two non-record tests print `calibration-measurement` lines and assert only the
//! refusal property; they are not coverage records. The gate sweeps the sample-size grid
//! (`ANTECEDENT_CALIBRATION_GRID_POINT`, read by `grid_n`) for the two coverage records
//! only (`grid_group` in `scripts/gate_calibration.sh`); these two run once, at the base
//! point.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "learned_continuous_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    LEARNED_CONTINUOUS_BOUNDS, LearnedContinuousOptions, LearnerSpec, LinearSpec, TrialAipwInput,
    learned_continuous_interval_internal,
};
use antecedent_identify::TransportIdentifier;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed,
};
use dgp::{Design, Scenario, draw, graph};

/// Bootstrap replicates per execution: the frozen replicate floor of the cell.
const REPLICATES: u32 = LEARNED_CONTINUOUS_BOUNDS.1;

fn analytic_construction() -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.learned_trial_aipw".into(),
        interval_method: "analytic_se".into(),
        se_kind: "influence".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_mean_contrast".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn learned_continuous_nested_cohort_good_overlap(
    n_trial: usize,
    n_target: usize,
    seed: u64,
) -> TrialAipwInput {
    draw(Design::NestedCohort, Scenario::Good, n_trial, n_target, seed)
}

fn learned_continuous_independent_samples_good_overlap(
    n_trial: usize,
    n_target: usize,
    seed: u64,
) -> TrialAipwInput {
    draw(Design::IndependentSamples, Scenario::Good, n_trial, n_target, seed)
}

fn analytic_record(design: Design, test: &'static str, dgp_name: &'static str) {
    let n = sizes();
    let truth = Scenario::Good.truth();
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: dgp_name, interval: "analytic_se" },
        REPORTED_LEVEL,
    );
    let mut analytic_options = options();
    analytic_options.bootstrap = 0;
    let (diagram, query, _) = graph();
    let id = TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let rows = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(rep, 0x4C43);
        let input = match design {
            Design::NestedCohort => learned_continuous_nested_cohort_good_overlap(n.0, n.1, seed),
            Design::IndependentSamples => {
                learned_continuous_independent_samples_good_overlap(n.0, n.1, seed)
            }
        };
        analytic_options
            .analytic_interval_internal(&id, &input, &ExecutionContext::for_tests(rep))
            .ok()
    });
    for row in rows {
        match row {
            Some((_, band)) => {
                tally.bind(
                    &analytic_construction(),
                    ScopeFacts {
                        row_count: (n.0 + n.1) as u64,
                        replicates_ok: None,
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(Some(band), truth);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
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

/// One execution of the whole estimator: the interval, or `None` when it refuses.
fn execute(
    design: Design,
    scenario: Scenario,
    sizes: (usize, usize),
    rep: u64,
) -> Option<((f64, f64), u32)> {
    let (diagram, query, _) = graph();
    let id = TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let input = draw(design, scenario, sizes.0, sizes.1, stream_seed(rep, 0x4C43));
    let ctx = ExecutionContext::for_tests(rep);
    let run = learned_continuous_interval_internal(&id, &input, &options(), &ctx).ok()?;
    let ok = u32::try_from(run.replicates.len()).unwrap_or(u32::MAX);
    run.interval.map(|band| (band, ok))
}

fn sizes() -> (usize, usize) {
    (grid_n(300), grid_n(200))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn learned_trial_aipw_nested_cohort_continuous_mean_contrast() {
    analytic_record(
        Design::NestedCohort,
        "learned_trial_aipw_nested_cohort_continuous_mean_contrast",
        "learned_continuous_nested_cohort_good_overlap",
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn learned_trial_aipw_independent_samples_continuous_mean_contrast() {
    analytic_record(
        Design::IndependentSamples,
        "learned_trial_aipw_independent_samples_continuous_mean_contrast",
        "learned_continuous_independent_samples_good_overlap",
    );
}

/// Coverage and refusal counts of `n_sim()` datasets under one scenario, both designs.
#[expect(clippy::cast_possible_truncation, reason = "counts are at most n_sim(), a u32")]
fn measure(scenario: Scenario) -> Vec<(Design, u32, u32, u32)> {
    let n = sizes();
    [Design::NestedCohort, Design::IndependentSamples]
        .into_iter()
        .map(|design| {
            let rows = map_replicates(n_sim(), |rep| execute(design, scenario, n, rep));
            let refused = rows.iter().filter(|r| r.is_none()).count() as u32;
            let covered = rows
                .iter()
                .flatten()
                .filter(|((lo, hi), _)| *lo <= scenario.truth() && scenario.truth() <= *hi)
                .count() as u32;
            (design, n_sim(), refused, covered)
        })
        .collect()
}

fn report(label: &str, rows: &[(Design, u32, u32, u32)]) {
    for (design, total, refused, covered) in rows {
        let accepted = total - refused;
        println!(
            "calibration-measurement {label} design={design:?} n_sim={total} refused={refused} \
             accepted={accepted} covered={covered} coverage_of_accepted={:.4}",
            f64::from(*covered) / f64::from(accepted.max(1))
        );
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn learned_trial_aipw_weak_overlap_boundary() {
    // Far target: membership probabilities vanish in the tail, so refusal is the result.
    let far = measure(Scenario::WeakOverlap);
    report("weak_overlap_far", &far);
    for (design, total, refused, _) in &far {
        assert!(
            f64::from(*refused) >= 0.9 * f64::from(*total),
            "{design:?}: only {refused} of {total} weak-overlap datasets refused"
        );
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn learned_trial_aipw_misspecified_nuisance_cases() {
    for (label, scenario) in [
        ("misspecified_outcome_correct_membership", Scenario::MisspecifiedOutcome),
        ("misspecified_membership_correct_outcome", Scenario::MisspecifiedMembership),
    ] {
        report(label, &measure(scenario));
    }
}
