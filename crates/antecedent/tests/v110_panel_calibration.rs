//! Coverage of the panel route intervals.
//!
//! Panel Pulse on a supplied `TemporalDag` publishes the Arellano cluster-by-unit
//! SE at `G − 1` degrees of freedom and a unit cluster bootstrap SE; both are
//! read as `ate ± 1.96·se` at the 0.95 interval level of the result binding.
//! They are calibrated here on the design where a truncated within-unit HAC
//! fails: treatment and outcome noise both AR(1) with φ = ρ = 0.9 inside each
//! unit, so every unit's regression score is persistently dependent
//! (`common::panel_dgp`, truth β = 0.8).
//!
//! Panel `ResponseCurve` averages unit surfaces with equal weight and publishes
//! the between-unit band `mean ± t_(N−1)·sd/√N`; it is calibrated for the
//! unit-population mean level with heterogeneous unit slopes at N = 3 and 10.
//!
//! Ignored coverage tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

mod common;

use antecedent::{RefuteSuite, Study, StudyResult};
use antecedent_core::{CausalQuery, ExecutionContext, ResponseUncertainty};
use common::calibration::{
    CoverageTally, RecordKey, Z95, gaussian, n_sim, normal_interval, stream_seed,
};
use common::calibration_bind::bind;
use common::panel_dgp::{UnitSpec, curve_query, lagged_ty_dag, panel, pulse_query, unit_series};

/// Persistent within-unit design: 30 units of 80 rows, φ = ρ = 0.9.
const PERSISTENT: UnitSpec = UnitSpec { n: 80, beta: 0.8, gz: 0.0, phi: 0.9, rho: 0.9 };
const UNITS: usize = 30;

/// The executed panel study beside its result: a coverage record binds to the
/// construction the runtime reports for that execution.
fn panel_pulse_study(
    spec: UnitSpec,
    units: usize,
    rep: u32,
    boot: u32,
    seed: u64,
) -> (Study, StudyResult) {
    let series = (0..units)
        .map(|unit| unit_series(spec, stream_seed(seed + u64::from(rep), unit as u64)))
        .collect();
    let study = Study::panel(panel(series))
        .graph(lagged_ty_dag())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(u64::from(rep) + 1)).unwrap();
    (study, result)
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn panel_pulse_cluster_se_ar09_nominal_95_coverage() {
    let mut analytic = CoverageTally::for_record(
        RecordKey {
            test: "panel_pulse_cluster_se_ar09_nominal_95_coverage",
            dgp: "crates/antecedent/tests/common/panel_dgp.rs::unit_series",
            interval: "analytic_se",
        },
        0.95,
    );
    for rep in 0..n_sim() {
        let (study, result) = panel_pulse_study(PERSISTENT, UNITS, rep, 0, 610_000);
        let est = &result.estimate;
        bind(&mut analytic, &study, &result);
        analytic.record(normal_interval(est.ate, Some(est.se_analytic), Z95), PERSISTENT.beta);
    }
    analytic.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn panel_pulse_unit_bootstrap_ar09_nominal_95_coverage() {
    let mut boot = CoverageTally::for_record(
        RecordKey {
            test: "panel_pulse_unit_bootstrap_ar09_nominal_95_coverage",
            dgp: "crates/antecedent/tests/common/panel_dgp.rs::unit_series",
            interval: "bootstrap_se",
        },
        0.95,
    );
    for rep in 0..n_sim() {
        let (study, result) = panel_pulse_study(PERSISTENT, UNITS, rep, 99, 620_000);
        let est = &result.estimate;
        assert_eq!(est.bootstrap_replicates_failed, Some(0));
        bind(&mut boot, &study, &result);
        boot.record(normal_interval(est.ate, est.se_bootstrap, Z95), PERSISTENT.beta);
    }
    boot.assert();
}

/// Equal-weight panel response over `units` units whose slopes are drawn from
/// N(0.8, 0.4²): the target is the unit-population mean level `1 + 0.8·dose`.
fn between_unit_band_coverage(test: &'static str, units: usize, seed: u64) {
    let key = RecordKey {
        test,
        dgp: "crates/antecedent/tests/common/panel_dgp.rs::unit_series",
        interval: "analytic_se",
    };
    let mut cells = [
        CoverageTally::for_record(key, 0.95).labelled(format!("N={units} dose 0")),
        CoverageTally::for_record(key, 0.95).labelled(format!("N={units} dose 1")),
    ];
    for rep in 0..n_sim() {
        let mut slope = gaussian(stream_seed(seed + u64::from(rep), 1_000));
        let series = (0..units)
            .map(|unit| {
                let spec = UnitSpec::iid(80, 0.8 + 0.4 * slope());
                unit_series(spec, stream_seed(seed + u64::from(rep), unit as u64))
            })
            .collect();
        let study = Study::panel(panel(series))
            .graph(lagged_ty_dag())
            .query(CausalQuery::Response(curve_query(vec![1])))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(u64::from(rep) + 1)).unwrap();
        let response = result.response.as_ref().unwrap();
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("panel response must publish a between-unit band");
        };
        for (cell, tally) in cells.iter_mut().enumerate() {
            bind(tally, &study, &result);
            tally.record(Some((lower[cell], upper[cell])), 1.0 + 0.8 * cell as f64);
        }
    }
    for tally in &cells {
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn panel_response_between_unit_band_n3_nominal_95_coverage() {
    between_unit_band_coverage(
        "panel_response_between_unit_band_n3_nominal_95_coverage",
        3,
        630_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn panel_response_between_unit_band_n10_nominal_95_coverage() {
    between_unit_band_coverage(
        "panel_response_between_unit_band_n10_nominal_95_coverage",
        10,
        640_000,
    );
}
