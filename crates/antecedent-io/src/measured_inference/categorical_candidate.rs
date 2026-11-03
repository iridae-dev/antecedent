//! Exact source-replay adapter for one bounded categorical posterior interval route.
//!
//! A decoded source is data, not authority. The adapter reconstructs the shared typed compiler
//! contract, independently executes the native conjugate posterior under that compiler
//! identity, and calls the private measured factory. The factory refuses unless a current
//! attesting coverage record covers the execution's construction, level count and row count.
use std::borrow::Cow;

use antecedent_core::categorical_candidate_contract::{
    CREDIBLE_LEVEL, CategoricalCandidateSource, admission_bytes,
};
use antecedent_core::{ExecutionContext, IdentityDomain, SemanticDigest};
use antecedent_estimate::categorical_treatment::CategoricalTreatmentInput;
use antecedent_estimate::categorical_treatment_posterior::{
    categorical_candidate_prior, categorical_candidate_spec,
    categorical_prior_predictive_calibration_basis, posterior_categorical_treatment,
};

use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;

fn invalid(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("{detail}: {message}"),
    }
}

struct ContrastReading {
    name: String,
    point: f64,
    interval: [f64; 2],
}

struct CandidateReading {
    source: CategoricalCandidateSource,
    contrasts: Vec<ContrastReading>,
    basis: antecedent_core::CalibrationBasis,
    _lease: antecedent_core::execution::AdmissionLease,
}

// This private reading is never a measured result: only the crate-private factory, after a
// matching attesting record, turns it into one.
fn execute_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<CandidateReading, IoError> {
    let source = CategoricalCandidateSource::decode(bytes)
        .map_err(|message| invalid("compiled_categorical_posterior.source_invalid", message))?;
    let lease = ctx
        .acquire_admission(0, admission_bytes(source.rows(), source.declared_levels().len()))
        .map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => IoError::Refused {
                code: antecedent_core::reason_code!("cancelled_no_claim"),
                message: "compiled_categorical_posterior.cancelled: source replay cancelled"
                    .to_owned(),
            },
            other => IoError::Refused {
                code: antecedent_core::reason_code!("execution_resource_limit"),
                message: format!("compiled_categorical_posterior.resource_limit: {other}"),
            },
        })?;
    let scoped = ctx.enter_lowered_dispatch(source.program_identity());
    let input = CategoricalTreatmentInput {
        outcome: Cow::Borrowed(source.outcomes()),
        row_snapshot: Cow::Borrowed(source.snapshot()),
        adjustment: vec![],
        levels: Cow::Borrowed(source.levels()),
    };
    let spec = categorical_candidate_spec(source.declared_levels());
    let posterior = posterior_categorical_treatment(
        &input,
        &spec,
        &categorical_candidate_prior(),
        CREDIBLE_LEVEL,
    )
    .map_err(|error| {
        invalid("compiled_categorical_posterior.execution_refused", &error.to_string())
    })?;
    if scoped.cancellation.is_cancelled() {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            message: "compiled_categorical_posterior.cancelled: candidate execution cancelled"
                .to_owned(),
        });
    }
    if posterior.level_contrasts.len() + 1 != source.declared_levels().len() {
        return Err(invalid(
            "compiled_categorical_posterior.execution_refused",
            "the reported contrasts differ from the declared levels",
        ));
    }
    let contrasts = posterior
        .level_contrasts
        .iter()
        .map(|contrast| ContrastReading {
            name: format!("{}_minus_{}", contrast.to, contrast.from),
            point: contrast.mean,
            interval: contrast.posterior.equal_tailed_interval(CREDIBLE_LEVEL),
        })
        .collect();
    let basis = categorical_prior_predictive_calibration_basis(
        source.rows(),
        source.declared_levels().len(),
    );
    Ok(CandidateReading { source, contrasts, basis, _lease: lease })
}

/// Reconstruct the shared compiler contract and execute the exact bounded source.
///
/// The cell has at least two declared levels, at least two rows of every level,
/// distinct declared independent subjects, the fixed normal-inverse-gamma prior and a 0.95
/// equal-tailed Student-t interval for every level against the first declared level. Its
/// candidate study is prior-predictive; no causal or pointwise frequentist coverage claim
/// follows. A measured result can never be created from a caller-provided basis or report.
///
/// # Errors
/// The source is malformed/out of scope, execution is cancelled or over budget, the native
/// computation refuses, or no current attesting record covers the construction, level count
/// and row count.
pub fn measure_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
    let reading = execute_source(bytes, ctx)?;
    let source = &reading.source;
    let premises = crate::digest_hex(
        IdentityDomain::Claim,
        &(
            "categorical_prior_predictive_interval.v2",
            source.query_digest(),
            source.program_identity().to_hex(),
            "one_row_per_declared_independent_subject",
        ),
    )?;
    let data = SemanticDigest::from_bytes(source.source_digest()).to_hex();
    let scope = serde_json::json!({
        "validation_design": "declared-level normal-inverse-gamma prior predictive over arm sizes and imbalance",
        "evidence_interpretation": "prior-predictive credible interval under the stated model and prior; not pointwise frequentist coverage or a causal claim",
        "declared_model_assumptions": ["independent subjects", "one row per subject", "homoskedastic Gaussian outcomes", "fixed normal-inverse-gamma prior"],
        "rows": source.rows(),
        "levels": source.declared_levels().len(),
        "reference": source.reference(),
        "reported_scalars": "level_minus_reference",
        "compiled_program": source.program_identity().to_hex(),
    });
    authorize(
        "categorical_prior_predictive_interval",
        bytes.to_vec(),
        &premises,
        &data,
        scope,
        reading
            .contrasts
            .iter()
            .map(|contrast| ScalarExecution {
                name: contrast.name.clone(),
                point: contrast.point,
                interval: (contrast.interval[0], contrast.interval[1]),
                basis: reading.basis.clone(),
            })
            .collect(),
    )
}

/// Reconsume a measured envelope by replaying its original native source and compiler contract.
/// The expected identity is separately retained; it is never evidence or authority by itself.
///
/// # Errors
/// Any source, compiled program, result, evidence or expected identity mismatch.
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
    use antecedent_core::sampling_design::SamplingDesign;
    use antecedent_io_core::coverage_records_data::CoverageRecord;

    fn source(counts: &[usize]) -> CategoricalCandidateSource {
        let declared: Vec<String> = (0..counts.len()).map(|i| format!("level-{i}")).collect();
        let levels: Vec<String> = counts
            .iter()
            .enumerate()
            .flat_map(|(i, &count)| (0..count).map(|_| declared[i].clone()).collect::<Vec<_>>())
            .collect();
        let rows = levels.len();
        CategoricalCandidateSource::new(
            (0..rows)
                .map(|i| {
                    f64::from(u32::try_from(i % 7).unwrap())
                        + 0.25 * f64::from(u32::try_from(i % 3).unwrap())
                })
                .collect(),
            levels,
            declared,
            (0..rows).map(|i| format!("subject-{i}")).collect(),
            "known".into(),
            SamplingDesign::IndependentSubjectTrajectories,
        )
        .unwrap()
    }

    // An attesting record keyed exactly as the route's own basis is, covering a row range.
    fn record(levels: usize, n_min: u64, n_max: u64) -> CoverageRecord {
        CoverageRecord {
            id: "cov.categorical_treatment_contrast.not_applicable.bayesian.posterior_quantile.l95.fixture.v2_validation_not_applicable",
            validation_schema: 2,
            validation: "not_applicable",
            query: "CategoricalTreatmentContrast",
            graph_class: "NotApplicable",
            structure: "declared_linear_model",
            modality: "tabular",
            inference: "Bayesian",
            estimator: "categorical_normal_inverse_gamma",
            interval_method: "posterior_quantile",
            se_kind: "posterior_student_t",
            dependence: "independent_rows",
            posterior: "nig_lp1_np0_5_shape2_scale1",
            functional: Box::leak(
                format!("level_contrast_prior_predictive_k{levels}_v1").into_boxed_str(),
            ),
            identification: "model_conditional",
            nominal: 0.95,
            n_min,
            n_max,
            replicates_min: 0,
            posterior_draws_min: 0,
            unidentified_mass_max: 0.0,
            observed: 0.95,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
            grid: &[],
            confirmation: &[],
            confirmed_passing_range: None,
            dgp: "crates/antecedent/tests/x.rs::dgp",
            test: "crates/antecedent/tests/x.rs::test",
            calibration_sha: "0123456789abcdef0123456789abcdef01234567",
        }
    }

    fn refusal_code(error: IoError) -> (String, String) {
        match error {
            IoError::Refused { code, message } => (code.to_string(), message),
            other => panic!("unexpected refusal: {other}"),
        }
    }

    #[test]
    fn untrusted_source_is_refused_before_any_execution() {
        let (code, message) = refusal_code(
            measure_source(b"untrusted", &ExecutionContext::for_tests(7)).unwrap_err(),
        );
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("compiled_categorical_posterior.source_invalid"));
    }

    #[test]
    fn without_a_matching_record_the_route_refuses() {
        let bytes = source(&[20, 20]).encode();
        let ctx = ExecutionContext::for_tests(7);
        let (code, message) = super::super::with_test_records(vec![], || {
            refusal_code(measure_source(&bytes, &ctx).unwrap_err())
        });
        assert_eq!(code, "not_supported");
        assert!(message.contains("measured_inference.record_not_attesting"));
        // A record of another level count or another row range does not cover this execution.
        for wrong in [record(3, 30, 4096), record(2, 100, 4096), record(2, 30, 39)] {
            let (code, _) = super::super::with_test_records(vec![wrong], || {
                refusal_code(measure_source(&bytes, &ctx).unwrap_err())
            });
            assert_eq!(code, "not_supported");
        }
    }

    #[test]
    fn a_covering_attesting_record_supports_every_level_contrast() {
        let ctx = ExecutionContext::for_tests(7);
        for (counts, levels) in [(vec![20, 20], 2), (vec![5, 12, 33], 3), (vec![4; 8], 8)] {
            let source = source(&counts);
            let bytes = source.encode();
            let rows = u64::try_from(source.rows()).unwrap();
            let measured = super::super::with_test_records(vec![record(levels, 30, 4096)], || {
                measure_source(&bytes, &ctx)
            })
            .unwrap();
            let scalars = &measured.report().scalars;
            assert_eq!(scalars.len(), levels - 1);
            assert_eq!(scalars[0].name, "level-1_minus_level-0");
            for scalar in scalars {
                assert_eq!(scalar.calibration.status, "calibrated");
                assert!(scalar.lower < scalar.point && scalar.point < scalar.upper);
                assert_eq!(
                    scalar.calibration.basis.as_ref().map(|basis| basis.scope.row_count),
                    Some(rows)
                );
            }
            // The supported envelope replays from its original source and is refused without the record.
            let again = super::super::with_test_records(vec![record(levels, 30, 4096)], || {
                consume_measured(measured.export(), measured.expectation(), &ctx)
            })
            .unwrap();
            assert_eq!(again.report().identity.seal, measured.report().identity.seal);
            let without = super::super::with_test_records(vec![], || {
                consume_measured(measured.export(), measured.expectation(), &ctx)
            });
            assert!(without.is_err());
        }
    }

    #[test]
    fn a_record_never_extrapolates_beyond_its_measured_rows() {
        let ctx = ExecutionContext::for_tests(7);
        let bytes = source(&[40, 40]).encode();
        let (code, _) = super::super::with_test_records(vec![record(2, 30, 79)], || {
            refusal_code(measure_source(&bytes, &ctx).unwrap_err())
        });
        assert_eq!(code, "not_supported");
        let inside = super::super::with_test_records(vec![record(2, 30, 80)], || {
            measure_source(&bytes, &ctx)
        });
        assert!(inside.is_ok());
    }

    #[test]
    fn checked_two_group_candidate_matches_the_independent_normal_equations() {
        // Lambda = [[40.5, 20], [20, 21]], X'y = [60, 40]. Its determinant is 450.5,
        // giving coefficient 420 / 450.5 = 840 / 901.
        let outcome = (0..40)
            .map(|i| {
                let alternating = if i % 2 == 0 { 0.0 } else { 2.0 };
                alternating + if i < 20 { 0.0 } else { 1.0 }
            })
            .collect();
        let source = CategoricalCandidateSource::new(
            outcome,
            (0..40).map(|i| if i < 20 { "control" } else { "treated" }.into()).collect(),
            vec!["control".into(), "treated".into()],
            (0..40).map(|i| format!("subject-{i}")).collect(),
            "known-normal-equations".into(),
            SamplingDesign::IndependentSubjectTrajectories,
        )
        .unwrap();
        let reading = execute_source(&source.encode(), &ExecutionContext::for_tests(7)).unwrap();
        let contrast = &reading.contrasts[0];
        assert_eq!(contrast.name, "treated_minus_control");
        assert!((contrast.point - 840.0 / 901.0).abs() < 1e-10);
        assert!(contrast.interval[0] < contrast.point && contrast.point < contrast.interval[1]);
        assert_eq!(reading.basis.functional.as_ref(), "level_contrast_prior_predictive_k2_v1");
        assert_eq!(reading.basis.row_count, 40);
        assert!(reading.basis.unidentified_mass.abs() < f64::EPSILON);
        assert_eq!(reading.source.program_identity(), source.program_identity());
    }

    #[test]
    fn checked_three_group_candidate_matches_the_independent_normal_equations() {
        // Ten rows per level with group sums s_j = 10 (j + 1) + 5. With intercept precision 1/2 and
        // level precision 1, Lambda = [[30.5, 10, 10], [10, 11, 0], [10, 0, 11]] and
        // X'y = [S, s1, s2], so m1 = (s1 - 10 m0) / 11 with
        // m0 (30.5 - 200 / 11) = S - 10 (s1 + s2) / 11.
        let declared: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        let mut levels = Vec::new();
        let mut outcomes = Vec::new();
        for (j, label) in declared.iter().enumerate() {
            for i in 0..10_u32 {
                levels.push(label.clone());
                outcomes.push(f64::from(u32::try_from(j).unwrap() + 1) + f64::from(i % 2));
            }
        }
        let source = CategoricalCandidateSource::new(
            outcomes,
            levels,
            declared,
            (0..30).map(|i| format!("subject-{i}")).collect(),
            "three-level".into(),
            SamplingDesign::IndependentSubjectTrajectories,
        )
        .unwrap();
        let (s0, s1, s2) = (15.0_f64, 25.0_f64, 35.0_f64);
        let m0 = (s0 + s1 + s2 - 10.0 * (s1 + s2) / 11.0) / (30.5 - 200.0 / 11.0);
        let reading = execute_source(&source.encode(), &ExecutionContext::for_tests(7)).unwrap();
        assert_eq!(reading.contrasts.len(), 2);
        assert!((reading.contrasts[0].point - (s1 - 10.0 * m0) / 11.0).abs() < 1e-10);
        assert!((reading.contrasts[1].point - (s2 - 10.0 * m0) / 11.0).abs() < 1e-10);
    }
}
