//! Coverage of the smoothed dose-response transport interval (2.2B cell X4) against the
//! closed-form smoothed target.
//!
//! The estimator measured is the whole composed one the public route withholds
//! (`cell_not_licensed`) until these records exist: cross-fitted outcome and membership
//! nuisances, the quadrature, every support and quadrature refusal, the augmented score
//! and the pointwise joint outer refit percentile bootstrap, grouped per design
//! (`antecedent_estimate::smoothed_dose_interval_internal`).
//!
//! * `smoothed_dose_transport_nested_cohort_psi_h` and
//!   `smoothed_dose_transport_independent_samples_psi_h` are the two coverage records:
//!   pointwise coverage of `psi_h(2)` (h = 0.5) with both nuisance families correctly
//!   specified, sample-size grid `n/2, n, 2n` ([`grid_n`]), [`REPLICATES`] bootstrap
//!   replicates (the replicate floor), `n_sim()` datasets. The truth is `psi_h`, never
//!   `psi_0`: smoothing bias is excluded from the claim.
//! * `smoothed_dose_transport_grid_points` prints the coverage at the other grid doses
//!   (1 and 3) at the base point; it emits no record.
//! * `smoothed_dose_transport_weak_overlap_boundary` asserts the far-target boundary
//!   refuses in at least 90% of datasets; it emits no record.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "smoothed_dose_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    SMOOTHED_DOSE_BOUNDS, SmoothedDoseOptions, smoothed_dose_interval_internal,
};
use antecedent_identify::TransportIdentifier;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed,
};
use dgp::{Design, Scenario, diagram, draw, query};

const INTERVAL: &str = "percentile_bootstrap";
/// Bootstrap replicates per execution: the frozen replicate floor of the cell.
const REPLICATES: u32 = SMOOTHED_DOSE_BOUNDS.min_bootstrap;
const H: f64 = 0.5;

fn construction() -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.smoothed_dose_transport_aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "smoothed_dose_response".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn options() -> SmoothedDoseOptions {
    SmoothedDoseOptions {
        folds: 3,
        bootstrap: REPLICATES,
        coverage_level: REPORTED_LEVEL,
        ..SmoothedDoseOptions::default()
    }
}

/// One execution of the whole estimator at `grid`: the pointwise intervals, or `None`
/// when it refuses or withholds.
fn execute(
    design: Design,
    scenario: Scenario,
    grid: &[f64],
    sizes: (usize, usize),
    rep: u64,
) -> Option<(Vec<(f64, f64)>, u32)> {
    let (diagram, _) = diagram();
    let q = query(grid, H);
    let id = TransportIdentifier::new().identify(&diagram, &q.transport_query()).unwrap();
    let input = draw(design, scenario, sizes.0, sizes.1, stream_seed(rep, 0x5D05));
    let ctx = ExecutionContext::for_tests(rep);
    let run = smoothed_dose_interval_internal(&id, &q, &input, &options(), &ctx).ok()?;
    let ok = u32::try_from(run.replicates.len()).unwrap_or(u32::MAX);
    let bands: Option<Vec<_>> = run.intervals.into_iter().collect();
    bands.map(|b| (b, ok))
}

fn sizes() -> (usize, usize) {
    (grid_n(1200), grid_n(900))
}

fn record(design: Design, test: &'static str, dgp_name: &'static str) {
    let n = sizes();
    let truth = Scenario::Good.truth(2.0, H);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: dgp_name, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    let rows = map_replicates(n_sim(), |rep| execute(design, Scenario::Good, &[2.0], n, rep));
    for row in rows {
        match row {
            Some((bands, replicates_ok)) => {
                tally.bind(
                    &construction(),
                    ScopeFacts {
                        row_count: (n.0 + n.1) as u64,
                        replicates_ok: Some(replicates_ok),
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(Some(bands[0]), truth);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn smoothed_dose_transport_nested_cohort_psi_h() {
    record(
        Design::NestedCohort,
        "smoothed_dose_transport_nested_cohort_psi_h",
        "smoothed_dose_nested_cohort_quadratic_curve",
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn smoothed_dose_transport_independent_samples_psi_h() {
    record(
        Design::IndependentSamples,
        "smoothed_dose_transport_independent_samples_psi_h",
        "smoothed_dose_independent_samples_quadratic_curve",
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
#[expect(clippy::cast_possible_truncation, reason = "counts are at most n_sim(), a u32")]
fn smoothed_dose_transport_grid_points() {
    let grid = [1.0, 3.0];
    for design in [Design::NestedCohort, Design::IndependentSamples] {
        let rows =
            map_replicates(n_sim(), |rep| execute(design, Scenario::Good, &grid, sizes(), rep));
        let refused = rows.iter().filter(|r| r.is_none()).count() as u32;
        for (g, a) in grid.iter().enumerate() {
            let truth = Scenario::Good.truth(*a, H);
            let covered = rows
                .iter()
                .flatten()
                .filter(|(bands, _)| bands[g].0 <= truth && truth <= bands[g].1)
                .count() as u32;
            let accepted = n_sim() - refused;
            println!(
                "calibration-measurement smoothed_dose_grid design={design:?} a={a} n_sim={} \
                 refused={refused} covered={covered} coverage_of_accepted={:.4}",
                n_sim(),
                f64::from(covered) / f64::from(accepted.max(1))
            );
        }
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
#[expect(clippy::cast_possible_truncation, reason = "counts are at most n_sim(), a u32")]
fn smoothed_dose_transport_weak_overlap_boundary() {
    for design in [Design::NestedCohort, Design::IndependentSamples] {
        let rows = map_replicates(n_sim(), |rep| {
            execute(design, Scenario::WeakOverlap, &[2.0], sizes(), rep)
        });
        let refused = rows.iter().filter(|r| r.is_none()).count() as u32;
        println!(
            "calibration-measurement smoothed_dose_weak_overlap design={design:?} n_sim={} refused={refused}",
            n_sim()
        );
        assert!(
            f64::from(refused) >= 0.9 * f64::from(n_sim()),
            "{design:?}: only {refused} of {} weak-overlap datasets refused",
            n_sim()
        );
    }
}
