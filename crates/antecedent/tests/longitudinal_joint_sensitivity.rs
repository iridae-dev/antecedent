//! Independent fully enumerated and repeated-data studies for the joint sensitivity box.
#![allow(clippy::float_cmp, reason = "fixtures intentionally pin exact replay and declared values")]

use antecedent::analysis::composition::{self, BundleBuilder, SuppliedSources};
use antecedent::analysis::longitudinal_joint_sensitivity::{
    JointSensitivityError, LongitudinalSensitivityChoice, LongitudinalSensitivityPoint,
    LongitudinalSensitivitySpec, LongitudinalSubject, compile_longitudinal_joint_sensitivity,
    consume_longitudinal_joint_sensitivity,
};
use antecedent_core::{CausalRng, ExecutionContext};

fn declaration() -> LongitudinalSensitivitySpec {
    LongitudinalSensitivitySpec {
        population: "independent_baseline_population".into(),
        outcome: "reaction".into(),
        unit: "proportion".into(),
        assignment_limits: [[0.0, 0.055], [0.0, 0.055]],
        mnar_limits: [0.0, 0.10],
        outcome_support: Some([0.0, 1.0]),
        replicates: 199,
        level: 0.90,
        decision_threshold: Some(0.075),
        coverage_design: None,
    }
}

fn recorded_mean(w: u8, a0: u8, l1: u8) -> f64 {
    0.30 + 0.20 * f64::from(a0) + 0.04 * f64::from(l1) + 0.02 * f64::from(w)
}

fn observed_data() -> Vec<LongitudinalSubject> {
    let mut rows = Vec::new();
    for w in 0..2 {
        for a0 in 0..2 {
            for l1 in 0..2 {
                for a1 in 0..2 {
                    for copy in 0..40 {
                        rows.push(LongitudinalSubject {
                            subject: rows.len() as u64,
                            baseline: w,
                            treatment: [a0, a1],
                            intermediate: l1,
                            outcome: (copy % 4 != 0).then(|| recorded_mean(w, a0, l1)),
                        });
                    }
                }
            }
        }
    }
    rows
}

// A separate complete-data mechanism for each departure. Every observed outcome is
// held fixed. When the assigned treatment differs from the target schedule, find the
// target potential outcome by adjusting its mean among the otherwise relevant group.
fn complete_world(
    row: &LongitudinalSubject,
    target: u8,
    point: LongitudinalSensitivityPoint,
) -> f64 {
    let sign = if target == 0 { -1.0 } else { 1.0 };
    let w = row.baseline;
    let l1 = row.intermediate;
    if row.treatment[0] != target {
        let on_first_stage_schedule = (recorded_mean(w, target, 0) + recorded_mean(w, target, 1))
            / 2.0
            - sign * (0.25 * point.missingness + 0.5 * point.assignment[1]);
        return on_first_stage_schedule - sign * point.assignment[0];
    }
    if row.treatment[1] != target {
        return recorded_mean(w, target, l1)
            - sign * (0.25 * point.missingness + point.assignment[1]);
    }
    row.outcome.unwrap_or_else(|| recorded_mean(w, target, l1) - sign * point.missingness)
}

#[test]
fn two_complete_mechanisms_agree_observationally_and_exact_corners_change_the_decision() {
    let rows = observed_data();
    let spec = declaration();
    let ctx = ExecutionContext::for_tests(33);
    let plan = compile_longitudinal_joint_sensitivity(spec.clone(), rows.clone(), &ctx).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "LongitudinalJointSensitivity");
    let fit = plan.execute(&ctx).unwrap();
    let report = fit.report();
    assert_eq!(report.corners.len(), 8);
    let oracle = |point| {
        rows.iter()
            .map(|row| {
                let y0 = complete_world(row, 0, point);
                let y1 = complete_world(row, 1, point);
                assert!((0.0..=1.0).contains(&y0) && (0.0..=1.0).contains(&y1));
                if row.treatment == [0, 0] && row.outcome.is_some() {
                    assert_eq!(row.outcome, Some(y0));
                }
                if row.treatment == [1, 1] && row.outcome.is_some() {
                    assert_eq!(row.outcome, Some(y1));
                }
                y1 - y0
            })
            .sum::<f64>()
            / rows.len() as f64
    };
    for corner in &report.corners {
        assert!((corner.contrast - oracle(corner.departures)).abs() < 1e-12);
    }
    assert!((report.assumption_range[0] - 0.04).abs() < 1e-12);
    assert!((report.assumption_range[1] - 0.20).abs() < 1e-12);
    assert_eq!(
        report.decision.unwrap().assumption_choice,
        LongitudinalSensitivityChoice::Undetermined
    );
    for (assignment, mnar, minimum) in
        [([[0.0, 0.055]; 2], [0.0, 0.0], 0.09), ([[0.0, 0.0]; 2], [0.0, 0.10], 0.15)]
    {
        let single = LongitudinalSensitivitySpec {
            assignment_limits: assignment,
            mnar_limits: mnar,
            ..spec.clone()
        };
        let single = compile_longitudinal_joint_sensitivity(single, rows.clone(), &ctx)
            .unwrap()
            .execute(&ctx)
            .unwrap();
        assert!((single.report().assumption_range[0] - minimum).abs() < 1e-12);
        assert_eq!(
            single.report().decision.unwrap().assumption_choice,
            LongitudinalSensitivityChoice::PreferActive
        );
    }
    let artifact_bytes = fit.to_bytes().unwrap();
    let replay = consume_longitudinal_joint_sensitivity(
        &artifact_bytes,
        fit.identity(),
        plan.identity(),
        &ExecutionContext::for_tests(77),
    )
    .unwrap();
    assert_eq!(fit.report(), replay.report());

    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("sensitivity"), &artifact_bytes).unwrap();
    let bundle = builder.build().unwrap();
    let bytes = composition::export_bundle(&bundle, "longitudinal-joint-sensitivity").unwrap();
    let verified =
        composition::consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    verified.require_verified().unwrap();
    assert_eq!(verified.value("sensitivity", "assumption_lower"), Some(report.assumption_range[0]));
    assert_eq!(
        verified.value("sensitivity", "sampling_lower"),
        Some(report.extrema_intervals[0][0])
    );
    assert_eq!(
        verified.value("sensitivity", "sampling_upper"),
        Some(report.extrema_intervals[1][1])
    );
    assert_eq!(verified.value("sensitivity", "threshold"), Some(0.075));
    let incoherent = LongitudinalSensitivitySpec { mnar_limits: [0.0, 2.0], ..spec };
    assert!(matches!(
        compile_longitudinal_joint_sensitivity(incoherent, rows, &ctx).unwrap().execute(&ctx),
        Err(JointSensitivityError::IncoherentBox)
    ));
}

#[test]
fn shared_subject_bootstrap_matches_the_joint_supremum_order_statistic() {
    use antecedent_core::execution::StreamDomain;

    let ctx = ExecutionContext::for_tests(17);
    let spec = LongitudinalSensitivitySpec {
        level: 0.9025,
        assignment_limits: [[-0.02, 0.055], [-0.03, 0.04]],
        mnar_limits: [-0.04, 0.10],
        ..declaration()
    };
    let rows = observed_data();
    let plan = compile_longitudinal_joint_sensitivity(spec.clone(), rows.clone(), &ctx).unwrap();
    let fit = plan.execute(&ctx).unwrap();

    // Standardize each intervention mean directly from the sampled trajectories.
    // The observed means are constant within histories in this fixture. Refitting
    // still changes the baseline, assignment, history and recording frequencies.
    let means = |sample: &[&LongitudinalSubject], point: LongitudinalSensitivityPoint| {
        let mut contrast = 0.0;
        for a in 0..=1 {
            let sign = if a == 0 { -1.0 } else { 1.0 };
            for w in 0..=1 {
                let base: Vec<_> = sample.iter().filter(|r| r.baseline == w).collect();
                let first: Vec<_> = base.iter().filter(|r| r.treatment[0] == a).collect();
                let mut mean = 0.0;
                for l in 0..=1 {
                    let history: Vec<_> = first.iter().filter(|r| r.intermediate == l).collect();
                    let follows: Vec<_> = history.iter().filter(|r| r.treatment[1] == a).collect();
                    let missing = follows.iter().filter(|r| r.outcome.is_none()).count();
                    let history_mean = recorded_mean(w, a, l)
                        - sign
                            * (point.missingness * missing as f64 / follows.len() as f64
                                + point.assignment[1]
                                    * (1.0 - follows.len() as f64 / history.len() as f64));
                    mean += history_mean * history.len() as f64 / first.len() as f64;
                }
                mean -= sign * point.assignment[0] * (1.0 - first.len() as f64 / base.len() as f64);
                contrast += sign * mean * base.len() as f64 / sample.len() as f64;
            }
        }
        contrast
    };
    let original: Vec<_> = rows.iter().collect();
    let mut corners = Vec::new();
    for a0 in spec.assignment_limits[0] {
        for a1 in spec.assignment_limits[1] {
            for missingness in spec.mnar_limits {
                let point = LongitudinalSensitivityPoint { assignment: [a0, a1], missingness };
                corners.push((point, means(&original, point)));
            }
        }
    }
    let mut suprema = Vec::new();
    let n = rows.len() as u64;
    for draw in 0..199 {
        let mut rng = ctx.rng.stream_for(StreamDomain::external(0x4C4A_5345_4E53), draw);
        let sample: Vec<_> = (0..n)
            .map(|_| {
                let choice = rng.next_u64();
                // This fixed stream never hits the rejection tail of uniform sampling.
                assert!(choice < u64::MAX - u64::MAX % n);
                &rows[(choice % n) as usize]
            })
            .collect();
        let deviation = corners.iter().map(|&(p, center)| (means(&sample, p) - center).abs());
        suprema.push(deviation.fold(0.0, f64::max));
    }
    suprema.sort_by(f64::total_cmp);
    // 0.9025 * (199 + 1) is halfway between order statistics 180 and 181.
    let radius = (suprema[179] + suprema[180]) / 2.0;
    let report = fit.report();
    assert!((report.sampling_radius - radius).abs() < 1e-12);
    assert_eq!(report.corners.len(), corners.len());
    for (point, center) in &corners {
        let fitted = report.corners.iter().find(|c| c.departures == *point).unwrap();
        assert!((fitted.sampling_interval[0] - (center - radius)).abs() < 1e-12);
        assert!((fitted.sampling_interval[1] - (center + radius)).abs() < 1e-12);
    }
    let lo = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let hi = corners.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
    for (interval, center) in report.extrema_intervals.into_iter().zip([lo, hi]) {
        assert!((interval[0] - (center - radius)).abs() < 1e-12);
        assert!((interval[1] - (center + radius)).abs() < 1e-12);
    }
}

#[test]
fn repeated_studies_cover_the_entire_surface_and_both_decision_extrema_jointly() {
    const STUDIES: u64 = 64;
    const N: u64 = 1024;
    let truth = |point: LongitudinalSensitivityPoint| {
        0.2 - point.assignment[0] - point.assignment[1] - 0.5 * point.missingness
    };
    let mut full_surface = 0;
    let mut joint_extrema = 0;
    let spec = declaration();
    for replicate in 0..STUDIES {
        let mut rng = CausalRng::from_seed(2_000 + replicate);
        let rows = (0..N)
            .map(|subject| {
                let mut bit = || u8::from(rng.next_f64() < 0.5);
                let (w, a0, l1, a1) = (bit(), bit(), bit(), bit());
                let outcome = (rng.next_f64() < 0.75)
                    .then(|| f64::from(rng.next_f64() < recorded_mean(w, a0, l1)));
                LongitudinalSubject {
                    subject,
                    baseline: w,
                    treatment: [a0, a1],
                    intermediate: l1,
                    outcome,
                }
            })
            .collect::<Vec<_>>();
        let ctx = ExecutionContext::for_tests(90_001 + replicate);
        let fit = compile_longitudinal_joint_sensitivity(spec.clone(), rows, &ctx)
            .unwrap()
            .execute(&ctx)
            .unwrap();
        let report = fit.report();
        let contains = |[lo, hi]: [f64; 2], value| lo <= value && value <= hi;
        full_surface += usize::from(
            report.corners.iter().all(|c| contains(c.sampling_interval, truth(c.departures))),
        );
        joint_extrema += usize::from(
            contains(report.extrema_intervals[0], 0.04)
                && contains(report.extrema_intervals[1], 0.20),
        );
    }
    eprintln!(
        "nominal 0.90, simult. surface {full_surface}/{STUDIES}, joint decision extrema {joint_extrema}/{STUDIES}"
    );
    // A broad Monte Carlo guard for this finite simulation, not a uniform guarantee.
    assert!(full_surface >= 50 && joint_extrema >= 50);
}
