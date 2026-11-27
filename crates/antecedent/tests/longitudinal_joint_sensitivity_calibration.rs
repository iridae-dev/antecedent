//! Source-fixed joint sensitivity coverage of the full box and both decision extrema.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::analysis::longitudinal_joint_sensitivity::{
    JointSensitivityError, LongitudinalSensitivityCoverageDesign, LongitudinalSensitivitySpec,
    LongitudinalSubject, compile_longitudinal_joint_sensitivity,
};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "longitudinal_two_decision_joint_box_and_extrema_l95";
const DGP: &str = "balanced_binary_bernoulli_two_decision_selection_mnar_v1";
const INTERVAL: &str = "simultaneous_band";

fn measure(seed: u64, n: usize) -> Option<(f64, f64, Vec<CalibrationBasis>)> {
    let spec = LongitudinalSensitivitySpec {
        population: "balanced-independent-subject-population".into(),
        outcome: "binary-outcome".into(),
        unit: "proportion".into(),
        assignment_limits: [[0.0, 0.055], [0.0, 0.055]],
        mnar_limits: [0.0, 0.10],
        outcome_support: Some([0.0, 1.0]),
        replicates: 199,
        level: REPORTED_LEVEL,
        decision_threshold: Some(0.075),
        coverage_design: Some(LongitudinalSensitivityCoverageDesign::BalancedBinaryBernoulliV1),
    };
    let mut cell = uniform(seed, 0x751);
    let mut missingness = uniform(seed, 0x752);
    let mut response = uniform(seed, 0x753);
    let data = (0..n)
        .map(|i| {
            let mut bit = || u8::from(cell() > 0.5);
            let (w, a0, l1, a1) = (bit(), bit(), bit(), bit());
            let p = 0.30 + 0.20 * f64::from(a0) + 0.04 * f64::from(l1) + 0.02 * f64::from(w);
            LongitudinalSubject {
                subject: i as u64,
                baseline: w,
                treatment: [a0, a1],
                intermediate: l1,
                outcome: (missingness() < 0.75).then(|| f64::from(response() < p)),
            }
        })
        .collect::<Vec<_>>();
    let ctx = ExecutionContext::for_tests(grid_seed(seed ^ 0x754));
    let prepared = compile_longitudinal_joint_sensitivity(spec, data, &ctx).unwrap();
    let result = match prepared.execute(&ctx) {
        Ok(result) => result,
        Err(
            JointSensitivityError::Positivity
            | JointSensitivityError::RefitPositivity(_)
            | JointSensitivityError::IncoherentBox,
        ) => return None,
        Err(error) => panic!("unexpected joint sensitivity calibration failure: {error}"),
    };
    let report = result.report();
    let miss = |[lower, upper]: [f64; 2], truth: f64| {
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
        (lower - truth).max(truth - upper).max(0.0)
    };
    let whole_surface = report
        .corners
        .iter()
        .map(|corner| {
            // Independent closed form under fair-coin assignments and recording rates.
            let truth = 0.20
                - corner.departures.assignment.iter().sum::<f64>()
                - 0.5 * corner.departures.missingness;
            miss(corner.sampling_interval, truth)
        })
        .fold(0.0_f64, f64::max);
    let joint_endpoints =
        miss(report.extrema_intervals[0], 0.04).max(miss(report.extrema_intervals[1], 0.20));
    assert_eq!(report.corners.len(), 8);
    assert!(report.decision.is_some());
    Some((whole_surface, joint_endpoints, result.calibration_bases()))
}

fn bind(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    let construction = Construction {
        query: basis.query.to_string(),
        graph_class: basis.graph_class.to_string(),
        structure: basis.structure.to_string(),
        modality: basis.modality.to_string(),
        inference: basis.inference.to_string(),
        estimator: basis.estimator.to_string(),
        interval_method: basis.interval_method.to_string(),
        se_kind: basis.se_kind.to_string(),
        dependence: basis.dependence.to_string(),
        posterior: basis.posterior.to_string(),
        functional: basis.functional.to_string(),
        identification: basis.identification.to_string(),
        reported_level: basis.level,
        unit: None,
    };
    tally.bind_validation(
        &construction,
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
        &basis.validation,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn longitudinal_two_decision_joint_box_and_extrema_l95() {
    let n = grid_n(1600);
    let mut surface = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    )
    .labelled("all_corners");
    let mut extrema = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    )
    .labelled("joint_extrema");
    for measurement in map_replicates(n_sim(), |seed| measure(0x6A0D_8401 + seed, n)) {
        if let Some((corner_error, extrema_error, bases)) = measurement {
            let [surface_basis, decision_basis] = bases.as_slice() else {
                panic!("scope did not bind the two simultaneous readings")
            };
            bind(&mut surface, surface_basis);
            bind(&mut extrema, decision_basis);
            surface.record(Some((0.0, 0.0)), corner_error);
            extrema.record(Some((0.0, 0.0)), extrema_error);
        } else {
            surface.skip();
            extrema.skip();
        }
    }
    surface.assert();
    extrema.assert_one_sided(
        CoverageSide::Upper,
        "simultaneous containment of both projected decision extrema",
    );
}
