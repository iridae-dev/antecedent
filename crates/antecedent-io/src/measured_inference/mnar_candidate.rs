//! Source-bound measured adapter for the declared-shift MNAR contrast. An interval is
//! released only when an attesting coverage record covers the execution's scope.
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use antecedent_core::{CalibrationBasis, ExecutionContext, IdentityDomain, SemanticDigest};

const Z: f64 = 1.959_963_984_540_054_f64;

fn refused(code: &'static str, detail: &str, message: impl std::fmt::Display) -> IoError {
    IoError::Refused { code, message: format!("mnar_fixed_shift.{detail}: {message}") }
}

struct Reading {
    source: crate::mnar_candidate_source::ReconstructedMnarSource,
    point: f64,
    #[cfg_attr(
        not(any(test, feature = "calibration-internal")),
        allow(dead_code, reason = "used by the calibration harness")
    )]
    standard_error: f64,
    lower: f64,
    upper: f64,
    basis: CalibrationBasis,
}

// The interval about a point, refused when its standard error cannot support one.
fn interval(point: f64, standard_error: f64) -> Result<(f64, f64), IoError> {
    if !point.is_finite() || !standard_error.is_finite() || standard_error <= 0.0 {
        return Err(refused(
            antecedent_core::reason_code!("invalid_argument"),
            "degenerate_interval",
            format!("point {point} with standard error {standard_error} supports no interval"),
        ));
    }
    Ok((point - Z * standard_error, point + Z * standard_error))
}

// Single native implementation, shared by the feature-gated calibration harness
// and the measured replay. The source and declared assumptions are replayed.
fn execute_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<Reading, IoError> {
    let source = crate::mnar_candidate_source::decode_source(bytes, ctx)?;
    let data = source.source();
    let identity = data.lowered_identity().map_err(|e| {
        refused(antecedent_core::reason_code!("invalid_argument"), "invalid_source", e)
    })?;
    let scoped = ctx.enter_lowered_dispatch(identity);
    let arm = |treated: bool, shift: f64| -> Result<(f64, f64), IoError> {
        let rows = data.rows();
        let n = rows.iter().filter(|r| r.treated == treated).count() as f64;
        let observed = || rows.iter().filter(|r| r.treated == treated).filter_map(|r| r.outcome);
        let n_obs = observed().count() as f64;
        let mean = observed().sum::<f64>() / n_obs;
        let pi = n_obs / n;
        let missing_mean = mean + shift;
        let point = pi * mean + (1.0 - pi) * missing_mean;
        let combined = || observed().map(|y| pi * y + (1.0 - pi) * (y - mean));
        let combined_mean = combined().sum::<f64>() / n_obs;
        let variance = (mean - missing_mean).powi(2) * pi * (1.0 - pi) / n
            + combined().map(|v| (v - combined_mean).powi(2)).sum::<f64>() / (n_obs - 1.0) / n_obs;
        if scoped.cancellation.is_cancelled() {
            return Err(refused(
                antecedent_core::reason_code!("execution_cancelled"),
                "cancelled",
                "candidate execution cancelled",
            ));
        }
        Ok((point, variance))
    };
    let (treated, treated_var) = arm(true, data.treated_shift())?;
    let (control, control_var) = arm(false, 0.0)?;
    let point = treated - control;
    let standard_error = (treated_var + control_var).sqrt();
    let (lower, upper) = interval(point, standard_error)?;
    let a = |s: &str| std::sync::Arc::<str>::from(s);
    let functional = data.scope_functional();
    let basis = CalibrationBasis::new(
        [
            a("AverageEffect"),
            a("NotApplicable"),
            a("fixed"),
            a("tabular"),
            a("Frequentist"),
            a("mnar_declared_shift_difference_in_means"),
            a("delta_normal"),
            a("influence_function"),
            a("independent_units"),
            a(""),
            a(&functional),
        ],
        0.95,
        a("point"),
        data.rows().len() as u64,
        None,
        None,
        0.0,
    )
    .with_validation("not_applicable");
    Ok(Reading { source, point, standard_error, lower, upper, basis })
}

/// Source-bound, independently replayed declared-shift measurement.
///
/// The interval is released only when [`authorize`] finds an attesting coverage
/// record whose construction, shift and missingness band and row range cover this
/// execution; otherwise it refuses with the record resolver's reason.
/// # Errors
/// No attesting record covers the execution, the source is invalid or its interval
/// degenerate, or execution is cancelled or exceeds resource limits.
pub fn measure_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
    let reading = execute_source(bytes, ctx)?;
    let source = reading.source.source();
    let program = source.lowered_identity().map_err(|e| {
        refused(antecedent_core::reason_code!("invalid_argument"), "invalid_source", e)
    })?;
    let premises = crate::digest_hex(
        IdentityDomain::Claim,
        &(
            "mnar_fixed_shift_interval.v2",
            source.program_digest(),
            program.to_hex(),
            "one_row_per_declared_independent_subject",
        ),
    )?;
    let data = SemanticDigest::from_bytes(source.source_digest()).to_hex();
    let scope = serde_json::json!({
        "declared_model_assumptions": [
            "independent randomized arms",
            "one row per subject",
            "declared treated missing-pattern shift",
        ],
        "treated_shift": source.treated_shift(),
        "population": source.population(),
        "compiled_program": program.to_hex(),
        "calibration_scope": source.scope_functional(),
        "coverage": "released only under an attesting record covering this shift band, missingness band and row count",
    });
    authorize(
        "mnar_fixed_shift_interval",
        bytes.to_vec(),
        &premises,
        &data,
        scope,
        vec![ScalarExecution {
            name: "treated_minus_control_at_declared_shift".into(),
            point: reading.point,
            interval: (reading.lower, reading.upper),
            basis: reading.basis,
        }],
    )
}

/// Feature-gated study-only observation of the same private execution. It carries
/// no authority and cannot be compiled into normal builds.
/// # Errors
/// Invalid source, cancellation, or resource refusal.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn calibration_candidate(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(f64, f64, f64, CalibrationBasis), IoError> {
    let r = execute_source(bytes, ctx)?;
    Ok((r.standard_error, r.lower, r.upper, r.basis))
}

pub(crate) fn consume_measured(
    bytes: &[u8],
    expected: &MeasuredExpectation,
    ctx: &ExecutionContext,
) -> Result<MeasuredInference, IoError> {
    let wire = decode(bytes, expected)?;
    let measured = measure_source(&wire.source, ctx)?;
    compare_replay(&wire, &measured)?;
    Ok(measured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measured_inference::with_test_records;
    use antecedent_core::mnar_fixed_shift::{FixedShiftSource, SamplingPremise, SourceRow};
    use antecedent_io_core::coverage_records_data::CoverageRecord;

    fn rows(control: (usize, usize), treated: (usize, usize)) -> Vec<SourceRow> {
        let arm = |first: u64, is_treated: bool, (assigned, observed): (usize, usize)| {
            (0..assigned)
                .map(move |i| SourceRow {
                    subject_id: first + i as u64,
                    treated: is_treated,
                    outcome: (i < observed).then(|| {
                        f64::from(u32::try_from((i * 7) % 11).unwrap()) / 3.0
                            + f64::from(u8::from(is_treated))
                    }),
                })
                .collect::<Vec<_>>()
        };
        let mut rows = arm(0, false, control);
        rows.extend(arm(100_000, true, treated));
        rows
    }

    fn build(shift: f64, rows: Vec<SourceRow>) -> FixedShiftSource {
        FixedShiftSource::new(
            "p".into(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            shift,
            rows,
        )
        .unwrap()
    }

    fn at_sd(multiple: f64, rows: Vec<SourceRow>) -> FixedShiftSource {
        let scale = build(0.0, rows.clone()).pooled_outcome_sd();
        build(multiple * scale, rows)
    }

    fn encoded(source: &FixedShiftSource, ctx: &ExecutionContext) -> Vec<u8> {
        crate::mnar_candidate_source::encode(source, ctx).unwrap().bytes().to_vec()
    }

    // Independent derivation: the influence-corrected term of an arm has the sample
    // variance of its observed outcomes, so an arm contributes
    // shift^2 pi (1 - pi) / n + s^2 / n_obs about the mean mean + (1 - pi) shift.
    fn reference(source: &FixedShiftSource) -> (f64, f64) {
        let arm = |treated: bool, shift: f64| {
            let assigned = source.rows().iter().filter(|r| r.treated == treated).count() as f64;
            let y: Vec<f64> = source
                .rows()
                .iter()
                .filter(|r| r.treated == treated)
                .filter_map(|r| r.outcome)
                .collect();
            let n_obs = y.len() as f64;
            let mean = y.iter().sum::<f64>() / n_obs;
            let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_obs - 1.0);
            let pi = n_obs / assigned;
            (mean + (1.0 - pi) * shift, shift * shift * pi * (1.0 - pi) / assigned + var / n_obs)
        };
        let (t, tv) = arm(true, source.treated_shift());
        let (c, cv) = arm(false, 0.0);
        (t - c, (tv + cv).sqrt())
    }

    fn record(functional: &str, n_min: u64, n_max: u64) -> CoverageRecord {
        CoverageRecord {
            id: "cov.average_effect.not_applicable.frequentist.delta_normal.l95.synthetic.v2_validation_not_applicable",
            validation_schema: 2,
            validation: "not_applicable",
            query: "AverageEffect",
            graph_class: "NotApplicable",
            structure: "fixed",
            modality: "tabular",
            inference: "Frequentist",
            estimator: "mnar_declared_shift_difference_in_means",
            interval_method: "delta_normal",
            se_kind: "influence_function",
            dependence: "independent_units",
            posterior: "",
            functional: Box::leak(functional.to_owned().into_boxed_str()),
            identification: "point",
            nominal: 0.95,
            n_min,
            n_max,
            replicates_min: 0,
            posterior_draws_min: 0,
            unidentified_mass_max: 0.0,
            observed: 0.951,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
            grid: &[],
            confirmation: &[],
            confirmed_passing_range: None,
            dgp: "crates/antecedent/tests/synthetic.rs::dgp",
            test: "crates/antecedent/tests/synthetic.rs::test",
            calibration_sha: "0123456789abcdef0123456789abcdef01234567",
        }
    }

    #[test]
    fn execution_matches_hand_calculated_shifted_interval() {
        let cycle = |first: u64, treated: bool, offset: f64| -> Vec<SourceRow> {
            (0..25_u32)
                .map(|i| SourceRow {
                    subject_id: first + u64::from(i),
                    treated,
                    outcome: (i < 20).then(|| f64::from(i % 5) + offset),
                })
                .collect()
        };
        let mut data = cycle(0, false, 0.0);
        data.extend(cycle(100, true, 1.0));
        let source = build(1.0, data);
        let ctx = ExecutionContext::for_tests(4);
        let result = execute_source(&encoded(&source, &ctx), &ctx).unwrap();
        // Arms: means 2 and 3, observed share 0.8, sample variance 40/19 over 20.
        let variance: f64 = 0.8 * 0.2 / 25.0 + 2.0 * (40.0 / 19.0) / 20.0;
        assert!((result.point - 1.2).abs() < 1e-12);
        assert!((result.standard_error - variance.sqrt()).abs() < 1e-12);
        assert!((result.lower - (1.2 - Z * variance.sqrt())).abs() < 1e-12);
        assert!((result.upper - (1.2 + Z * variance.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn declared_shifts_and_unbalanced_arms_match_the_reference_derivation() {
        let ctx = ExecutionContext::for_tests(5);
        let designs = [
            rows((30, 24), (40, 28)),
            rows((15, 8), (35, 18)),
            rows((300, 270), (120, 61)),
            rows((40, 40), (60, 59)),
        ];
        for design in designs {
            for multiple in [-1.9, -0.4, 0.0, 0.3, 1.0, 2.0] {
                let source = at_sd(multiple, design.clone());
                let result = execute_source(&encoded(&source, &ctx), &ctx).unwrap();
                let (point, standard_error) = reference(&source);
                assert!((result.point - point).abs() < 1e-9);
                assert!((result.standard_error - standard_error).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn basis_carries_the_row_count_and_the_declared_scope() {
        let ctx = ExecutionContext::for_tests(6);
        let source = at_sd(-1.2, rows((30, 24), (40, 28)));
        let result = execute_source(&encoded(&source, &ctx), &ctx).unwrap();
        assert_eq!(result.basis.row_count, 70);
        assert_eq!(&*result.basis.functional, source.scope_functional().as_str());
        let near = at_sd(0.1, rows((30, 24), (40, 28)));
        assert_ne!(near.scope_functional(), source.scope_functional());
    }

    #[test]
    fn degenerate_intervals_are_refused_with_a_typed_error() {
        for (point, se) in
            [(1.0, 0.0), (1.0, -0.5), (1.0, f64::NAN), (1.0, f64::INFINITY), (f64::NAN, 1.0)]
        {
            assert!(matches!(
                interval(point, se),
                Err(IoError::Refused { code: "invalid_argument", message })
                    if message.starts_with("mnar_fixed_shift.degenerate_interval:")
            ));
        }
        let (lower, upper) = interval(2.0, 0.5).unwrap();
        assert!((lower - (2.0 - Z * 0.5)).abs() < 1e-15);
        assert!((upper - (2.0 + Z * 0.5)).abs() < 1e-15);
    }

    #[test]
    fn normal_quantile_constant_matches_stats() {
        let z = 1.959_963_984_540_054_f64;
        assert!((antecedent_stats::special::normal_ppf(0.975) - z).abs() < 1e-12);
    }

    #[test]
    fn without_a_record_the_route_refuses_through_the_resolver() {
        let ctx = ExecutionContext::for_tests(7);
        let bytes = encoded(&at_sd(0.8, rows((30, 24), (40, 28))), &ctx);
        let refusal = with_test_records(Vec::new(), || measure_source(&bytes, &ctx)).unwrap_err();
        assert!(matches!(&refusal, IoError::Refused { code: "not_supported", message }
            if message.contains("record_not_attesting")));
        assert!(matches!(
            with_test_records(Vec::new(), || measure_source(b"untrusted", &ctx)),
            Err(IoError::Refused { .. })
        ));
    }

    #[test]
    fn an_attesting_record_covering_the_scope_supports_the_measurement() {
        let ctx = ExecutionContext::for_tests(8);
        let source = at_sd(0.8, rows((30, 24), (40, 28)));
        let bytes = encoded(&source, &ctx);
        let covering = record(&source.scope_functional(), 50, 10_000);
        let (measured, replayed) = with_test_records(vec![covering], || {
            let measured = measure_source(&bytes, &ctx).unwrap();
            let replayed = consume_measured(measured.export(), measured.expectation(), &ctx);
            (measured, replayed.is_ok())
        });
        assert!(replayed);
        let report = measured.report();
        assert_eq!(report.calibration, "calibrated");
        let scalar = &report.scalars[0];
        let (point, standard_error) = reference(&source);
        assert!((scalar.point - point).abs() < 1e-9);
        assert!((scalar.upper - scalar.lower - 2.0 * Z * standard_error).abs() < 1e-9);
        assert_eq!(scalar.calibration.status, "calibrated");
    }

    #[test]
    fn a_record_for_another_scope_does_not_support_the_execution() {
        let ctx = ExecutionContext::for_tests(9);
        let source = at_sd(0.8, rows((30, 24), (40, 28)));
        let bytes = encoded(&source, &ctx);
        let refuses = |records: Vec<CoverageRecord>| {
            matches!(
                with_test_records(records, || measure_source(&bytes, &ctx)),
                Err(IoError::Refused { code: "not_supported", .. })
            )
        };
        let own = source.scope_functional();
        let other_shift = at_sd(1.7, rows((30, 24), (40, 28))).scope_functional();
        let other_missing = at_sd(0.8, rows((30, 30), (40, 40))).scope_functional();
        assert_ne!(own, other_shift);
        assert_ne!(own, other_missing);
        assert!(refuses(vec![record(&other_shift, 50, 10_000)]));
        assert!(refuses(vec![record(&other_missing, 50, 10_000)]));
        assert!(refuses(vec![record(&own, 100, 10_000)]));
        assert!(refuses(vec![record(&own, 50, 69)]));
        let mut coarser = record(&own, 50, 10_000);
        coarser.nominal = 0.9;
        assert!(refuses(vec![coarser]));
    }
}
