//! Compiler-owned declaration for a bounded categorical posterior candidate.
//!
//! Compilation retains the declared sampling design, per-row subject identities and the exact
//! observed source. No uncalibrated posterior result, causal claim or decision carrier is
//! exposed: the only interval is a measured inference that a covering coverage record supports.
use std::sync::Arc;

#[cfg(feature = "calibration-internal")]
use antecedent_core::CanonicalWriter;
use antecedent_core::ExecutionContext;
use antecedent_core::categorical_candidate_contract::{
    CREDIBLE_LEVEL, CategoricalCandidateSource, LEVEL_PRECISION, MIN_LEVEL_ROWS,
    NUISANCE_PRECISION, VARIANCE_SCALE, VARIANCE_SHAPE, admission_bytes,
};
use antecedent_core::sampling_design::SamplingDesign;
use antecedent_io::categorical_treatment_artifact::{
    CategoricalPosteriorDeclaration, CategoricalTreatmentRequest, CovarianceKindWire,
    LevelScaleWire,
};

use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;

/// Caller-declared sampling design and one immutable subject identity per row.
///
/// This declaration is an assumption, not evidence that subjects were independently sampled.
/// The compiler requires distinct identities because the bounded model treats every row as a
/// separate sampling unit. Repeated measurements are not admitted.
#[derive(Clone, Debug)]
pub struct CategoricalSamplingPremise {
    /// Explicit declared subject-level sampling design.
    pub design: SamplingDesign,
    /// Row-aligned subject IDs, one distinct nonempty identifier per input row.
    pub subject_ids: Vec<String>,
}

/// Bounded native request. Neither the prior nor the target is selected from the data.
#[derive(Clone, Debug)]
pub struct CategoricalPosteriorCandidateRequest {
    /// Original, unweighted categorical data and declared level design.
    pub data: CategoricalTreatmentRequest,
    /// Declared normal-inverse-gamma prior and requested credible level.
    pub posterior: CategoricalPosteriorDeclaration,
    /// Explicit row-to-subject binding and design assumption.
    pub sampling: CategoricalSamplingPremise,
}

/// Separately retained source and program identities; this value grants no scientific authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoricalPosteriorCompileReceipt {
    source: [u8; 32],
    program: [u8; 32],
}
impl CategoricalPosteriorCompileReceipt {
    /// Canonical identity of all source values, labels and row-aligned subjects.
    #[must_use]
    pub const fn source(&self) -> [u8; 32] {
        self.source
    }
    /// Identity of the six-layer compiler program and fixed semantic declarations.
    #[must_use]
    pub const fn program(&self) -> [u8; 32] {
        self.program
    }
}

impl CategoricalPosteriorCompileReceipt {
    /// Encode a separately retained expectation. These bytes are not scientific authority.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 65] {
        let mut bytes = [0; 65];
        bytes[0] = 1;
        bytes[1..33].copy_from_slice(&self.source);
        bytes[33..65].copy_from_slice(&self.program);
        bytes
    }

    /// Decode an untrusted expectation. The original source must still be recompiled.
    ///
    /// # Errors
    /// The receipt must have the supported version and exact length.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CausalError> {
        if bytes.len() != 65 || bytes[0] != 1 {
            return Err(invalid("candidate compile receipt has an unsupported version or length"));
        }
        let source = bytes[1..33].try_into().map_err(|_| invalid("invalid source identity"))?;
        let program = bytes[33..65].try_into().map_err(|_| invalid("invalid program identity"))?;
        Ok(Self { source, program })
    }
}

/// Immutable candidate declaration and its compiled program; no fitted result is retained.
pub struct PreparedCategoricalPosteriorCandidate {
    #[cfg_attr(
        not(feature = "calibration-internal"),
        allow(dead_code, reason = "read by the calibration harness")
    )]
    request: CategoricalPosteriorCandidateRequest,
    source_contract: CategoricalCandidateSource,
    program: Arc<LoweredProgram>,
    receipt: CategoricalPosteriorCompileReceipt,
    _lease: antecedent_core::execution::AdmissionLease,
}

fn invalid(message: &str) -> CausalError {
    CausalError::Compile { message: format!("compiled_categorical_posterior.candidate: {message}") }
}

/// Compile the bounded normal-inverse-gamma candidate without fitting it.
///
/// The declared levels (the first is the reference) number at least two, every level has at
/// least two rows, and each row carries a distinct subject. The row and level counts are
/// bounded only by the execution's resource limits.
///
/// # Errors
/// Rejects another sampling design, repeated subjects, unsupported levels/prior, or
/// resource limits.
pub fn prepare(
    request: CategoricalPosteriorCandidateRequest,
    ctx: &ExecutionContext,
) -> Result<PreparedCategoricalPosteriorCandidate, CausalError> {
    let data = &request.data;
    let posterior = &request.posterior;
    let sampling = &request.sampling;
    let equal = |a: f64, b: f64| a.to_bits() == b.to_bits();
    if sampling.design != SamplingDesign::IndependentSubjectTrajectories
        || !data.adjustment.is_empty()
        || data.spec.declared_levels.first() != Some(&data.spec.reference)
        || data.spec.scale != LevelScaleWire::Unordered
        || data.spec.min_level_rows != min_level_rows()
        || !data.spec.pairwise.is_empty()
        || data.spec.monotonicity.is_some()
        || data.spec.covariance != CovarianceKindWire::ModelBased
        || !equal(posterior.level_precision, LEVEL_PRECISION)
        || !equal(posterior.nuisance_precision, NUISANCE_PRECISION)
        || !equal(posterior.variance_shape, VARIANCE_SHAPE)
        || !equal(posterior.variance_scale, VARIANCE_SCALE)
        || !equal(posterior.credible_level, CREDIBLE_LEVEL)
    {
        return Err(invalid("the independent-subject design or model is not satisfied"));
    }
    let source_contract = CategoricalCandidateSource::new(
        data.outcome.clone(),
        data.levels.clone(),
        data.spec.declared_levels.clone(),
        sampling.subject_ids.clone(),
        data.row_snapshot.clone(),
        sampling.design,
    )
    .map_err(invalid)?;
    let lease = ctx
        .acquire_admission(
            0,
            admission_bytes(source_contract.rows(), source_contract.declared_levels().len()),
        )
        .map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "categorical posterior candidate" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
    let source = source_contract.source_digest();
    let program = Arc::new(super::lowering::categorical_posterior_candidate::lower(
        source,
        source_contract.query_digest(),
        source_contract.rows(),
        source_contract.declared_levels().len(),
    )?);
    // The lower-layer replay contract must describe the exact six-layer program that
    // was actually lowered, not just a caller-provided program name or digest.
    if program.identity() != source_contract.program_identity() {
        return Err(invalid("shared replay contract differs from the lowered compiler program"));
    }
    let receipt =
        CategoricalPosteriorCompileReceipt { source, program: *program.identity().as_bytes() };
    Ok(PreparedCategoricalPosteriorCandidate {
        request,
        source_contract,
        program,
        receipt,
        _lease: lease,
    })
}

fn min_level_rows() -> u64 {
    u64::try_from(MIN_LEVEL_ROWS).unwrap_or(u64::MAX)
}

impl PreparedCategoricalPosteriorCandidate {
    /// Inspect the real compiler lowering. The program carries no support of its own: an
    /// interval is supported only when a covering coverage record exists at measurement.
    #[must_use]
    pub fn program(&self) -> &LoweredProgram {
        &self.program
    }
    /// Identity to retain separately before a later native source replay.
    #[must_use]
    pub const fn receipt(&self) -> CategoricalPosteriorCompileReceipt {
        self.receipt
    }
    /// Encode the complete original source and subject declaration for later recompilation.
    /// The separately retained compile receipt is required for consumption.
    #[must_use]
    pub fn export_source(&self) -> Vec<u8> {
        self.source_contract.encode()
    }

    /// Replay the retained original source through measured-inference authorization.
    ///
    /// The interval is always reported; each scalar's calibration slot says whether a current
    /// attesting coverage record covers this construction, level count and row count.
    ///
    /// # Errors
    /// Source replay and execution can refuse.
    pub fn measured_interval(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::measured_inference::MeasuredInference, antecedent_io::IoError> {
        antecedent_io::measured_inference::categorical_candidate::measure_source(
            &self.export_source(),
            ctx,
        )
    }

    /// The credible interval of every level against the reference, with its calibration; the
    /// same route as [`Self::measured_interval`].
    ///
    /// # Errors
    /// As [`Self::measured_interval`].
    pub fn interval(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::measured_inference::MeasuredInference, antecedent_io::IoError> {
        self.measured_interval(ctx)
    }
}

/// Reconstruct an untrusted source and recompile it against a separately retained receipt.
/// No source bytes or caller-provided expectation can create measured authority.
///
/// # Errors
/// A malformed or changed source; an unsupported receipt; or an admission failure.
pub fn consume_source(
    bytes: &[u8],
    expected: CategoricalPosteriorCompileReceipt,
    ctx: &ExecutionContext,
) -> Result<PreparedCategoricalPosteriorCandidate, CausalError> {
    let source = CategoricalCandidateSource::decode(bytes).map_err(invalid)?;
    let lease = ctx
        .acquire_admission(0, admission_bytes(source.rows(), source.declared_levels().len()))
        .map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "categorical posterior source replay" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
    let request = CategoricalPosteriorCandidateRequest {
        data: CategoricalTreatmentRequest {
            outcome: source.outcomes().to_vec(),
            row_snapshot: source.snapshot().to_owned(),
            adjustment: vec![],
            levels: source.levels().to_vec(),
            spec: antecedent_io::categorical_treatment_artifact::CategoricalSpecWire {
                declared_levels: source.declared_levels().to_vec(),
                scale: LevelScaleWire::Unordered,
                reference: source.reference().to_owned(),
                min_level_rows: min_level_rows(),
                pairwise: vec![],
                monotonicity: None,
                covariance: CovarianceKindWire::ModelBased,
            },
        },
        posterior: CategoricalPosteriorDeclaration {
            level_precision: LEVEL_PRECISION,
            nuisance_precision: NUISANCE_PRECISION,
            variance_shape: VARIANCE_SHAPE,
            variance_scale: VARIANCE_SCALE,
            credible_level: CREDIBLE_LEVEL,
        },
        sampling: CategoricalSamplingPremise {
            design: SamplingDesign::IndependentSubjectTrajectories,
            subject_ids: source.subject_ids().to_vec(),
        },
    };
    drop(lease);
    replay(request, expected, ctx)
}

/// Separately retained identity of a calibration-only execution. It is never a claim.
#[cfg(feature = "calibration-internal")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CategoricalPosteriorCandidateRunReceipt {
    compiled: CategoricalPosteriorCompileReceipt,
    execution: [u8; 32],
}

#[cfg(feature = "calibration-internal")]
impl CategoricalPosteriorCandidateRunReceipt {
    /// Encode the separately retained expectation, not an authoritative claim.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 97] {
        let mut bytes = [0; 97];
        bytes[0] = 1;
        bytes[1..33].copy_from_slice(&self.compiled.source);
        bytes[33..65].copy_from_slice(&self.compiled.program);
        bytes[65..97].copy_from_slice(&self.execution);
        bytes
    }

    /// Decode an untrusted expectation. Authority is never obtained from these bytes.
    ///
    /// # Errors
    /// The receipt must have the exact supported version and length.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CausalError> {
        if bytes.len() != 97 || bytes[0] != 1 {
            return Err(invalid("candidate receipt has an unsupported version or length"));
        }
        let source = bytes[1..33].try_into().map_err(|_| invalid("invalid source identity"))?;
        let program = bytes[33..65].try_into().map_err(|_| invalid("invalid program identity"))?;
        let execution =
            bytes[65..97].try_into().map_err(|_| invalid("invalid execution identity"))?;
        Ok(Self { compiled: CategoricalPosteriorCompileReceipt { source, program }, execution })
    }
}

#[cfg(feature = "calibration-internal")]
impl PreparedCategoricalPosteriorCandidate {
    /// Run the candidate inside its retained lowered dispatch for calibration only.
    /// The returned intervals, one per non-reference level, are not supported.
    ///
    /// # Errors
    /// A native estimator refusal, cancellation, or resource admission failure.
    #[doc(hidden)]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::categorical_treatment_posterior::ContrastIntervals,
            antecedent_core::CalibrationBasis,
            CategoricalPosteriorCandidateRunReceipt,
        ),
        CausalError,
    > {
        use antecedent_estimate::categorical_treatment::CategoricalTreatmentInput;
        use antecedent_estimate::categorical_treatment_posterior::categorical_prior_predictive_calibration_candidate;
        let scoped = ctx.enter_lowered_dispatch(self.program.identity());
        let _lease = scoped
            .acquire_admission(
                0,
                admission_bytes(
                    self.source_contract.rows(),
                    self.source_contract.declared_levels().len(),
                ),
            )
            .map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "categorical posterior candidate execution" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
        let data = &self.request.data;
        let input = CategoricalTreatmentInput {
            outcome: std::borrow::Cow::Borrowed(&data.outcome),
            row_snapshot: std::borrow::Cow::Borrowed(&data.row_snapshot),
            adjustment: vec![],
            levels: std::borrow::Cow::Borrowed(&data.levels),
        };
        let (intervals, basis) =
            categorical_prior_predictive_calibration_candidate(&input, &data.spec.declared_levels)
                .map_err(CausalError::Estimate)?;
        if scoped.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: "categorical posterior candidate execution",
            });
        }
        let mut writer = CanonicalWriter::new("antecedent.categorical_candidate.execution.v1");
        writer.bytes(&self.receipt.source).bytes(&self.receipt.program);
        writer.word(intervals.len() as u64);
        for (level, interval) in &intervals {
            writer.text(level).reals(interval);
        }
        writer
            .text(&basis.validation)
            .text(&basis.query)
            .text(&basis.graph_class)
            .text(&basis.structure)
            .text(&basis.modality)
            .text(&basis.inference)
            .text(&basis.estimator)
            .text(&basis.interval_method)
            .text(&basis.se_kind)
            .text(&basis.dependence)
            .text(&basis.posterior)
            .text(&basis.functional)
            .text(&basis.identification)
            .reals(&[basis.level, basis.unidentified_mass])
            .word(basis.row_count)
            .opt_word(basis.replicates_ok.map(u64::from))
            .opt_word(basis.posterior_draws.map(u64::from));
        let receipt = CategoricalPosteriorCandidateRunReceipt {
            compiled: self.receipt,
            execution: writer.finish(),
        };
        Ok((intervals, basis, receipt))
    }
}

/// Recompile and rerun original source and typed premises against a separately held receipt.
/// This internal replay returns raw intervals and never grants measured authority.
///
/// # Errors
/// A source, program or execution mismatch, native refusal, or resource failure.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn replay_calibration_candidate(
    request: CategoricalPosteriorCandidateRequest,
    expected: CategoricalPosteriorCandidateRunReceipt,
    ctx: &ExecutionContext,
) -> Result<
    (
        antecedent_estimate::categorical_treatment_posterior::ContrastIntervals,
        antecedent_core::CalibrationBasis,
    ),
    CausalError,
> {
    let prepared = replay(request, expected.compiled, ctx)?;
    let (intervals, basis, observed) = prepared.calibration_candidate(ctx)?;
    if observed != expected {
        return Err(invalid("candidate execution differs from the retained receipt"));
    }
    Ok((intervals, basis))
}

/// Recompile original data and premises, comparing a separately retained compile receipt.
/// No decoded receipt or caller-constructed record creates scientific authority.
///
/// # Errors
/// Rejects invalid inputs and any changed source or program identity.
pub fn replay(
    request: CategoricalPosteriorCandidateRequest,
    expected: CategoricalPosteriorCompileReceipt,
    ctx: &ExecutionContext,
) -> Result<PreparedCategoricalPosteriorCandidate, CausalError> {
    let candidate = prepare(request, ctx)?;
    if candidate.receipt != expected {
        return Err(invalid("source or compiled program does not match the retained receipt"));
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_io::categorical_treatment_artifact::CategoricalSpecWire;

    fn shaped(counts: &[usize]) -> CategoricalPosteriorCandidateRequest {
        let declared: Vec<String> = (0..counts.len()).map(|i| format!("level-{i}")).collect();
        let levels: Vec<String> = counts
            .iter()
            .enumerate()
            .flat_map(|(i, &count)| (0..count).map(|_| declared[i].clone()).collect::<Vec<_>>())
            .collect();
        let rows = levels.len();
        CategoricalPosteriorCandidateRequest {
            data: CategoricalTreatmentRequest {
                outcome: (0..rows).map(|i| f64::from(u32::try_from(i).unwrap())).collect(),
                row_snapshot: "source-a".into(),
                adjustment: vec![],
                levels,
                spec: CategoricalSpecWire {
                    reference: declared[0].clone(),
                    declared_levels: declared,
                    scale: LevelScaleWire::Unordered,
                    min_level_rows: 2,
                    pairwise: vec![],
                    monotonicity: None,
                    covariance: CovarianceKindWire::ModelBased,
                },
            },
            posterior: CategoricalPosteriorDeclaration {
                level_precision: 1.0,
                nuisance_precision: 0.5,
                variance_shape: 2.0,
                variance_scale: 1.0,
                credible_level: 0.95,
            },
            sampling: CategoricalSamplingPremise {
                design: SamplingDesign::IndependentSubjectTrajectories,
                subject_ids: (0..rows).map(|i| format!("subject-{i}")).collect(),
            },
        }
    }

    fn request() -> CategoricalPosteriorCandidateRequest {
        shaped(&[20, 20])
    }

    #[test]
    fn candidate_compiles_a_posterior_claim_and_replays_its_source() {
        let ctx = ExecutionContext::for_tests(7);
        let input = request();
        let candidate = prepare(input.clone(), &ctx).unwrap();
        assert!(!candidate.program().scientific().supported());
        assert_eq!(candidate.program().route(), "CategoricalPosteriorCandidate");
        assert_eq!(
            candidate.program().claim().uncertainty(),
            crate::planner::lowered::UncertaintyReading::Posterior
        );
        // Calibration is decided by a covering coverage record at measurement and annotates
        // the interval: the route returns the interval whether or not a record covers it.
        let measured = candidate.measured_interval(&ctx).unwrap();
        assert!(!measured.report().scalars.is_empty());
        let expected = candidate.receipt();
        let expectation =
            CategoricalPosteriorCompileReceipt::from_bytes(&expected.to_bytes()).unwrap();
        let source = candidate.export_source();
        drop(candidate);
        let consumed = consume_source(&source, expectation, &ctx).unwrap();
        assert_eq!(consumed.receipt(), expected);
        let mut altered = source;
        altered[20] ^= 1;
        assert!(consume_source(&altered, expectation, &ctx).is_err());
        let replayed = replay(input.clone(), expected, &ctx).unwrap();
        assert_eq!(replayed.receipt(), expected);
        let mut changed = input;
        changed.sampling.subject_ids[0] = "changed".into();
        assert!(replay(changed, expected, &ctx).is_err());
    }

    #[test]
    fn general_shapes_compile_with_their_real_design_and_distinct_identities() {
        let ctx = ExecutionContext::for_tests(7);
        let mut identities = std::collections::BTreeSet::new();
        for counts in [
            vec![15, 15],
            vec![10, 10, 10],
            vec![2, 2, 26],
            vec![4; 8],
            vec![3, 2, 5, 40, 2, 7, 9, 2],
            vec![4_090, 6],
        ] {
            let input = shaped(&counts);
            let candidate = prepare(input.clone(), &ctx).unwrap();
            let design = candidate.program().physical().design();
            assert_eq!(design.rows(), Some(u64::try_from(counts.iter().sum::<usize>()).unwrap()));
            assert_eq!(design.columns(), Some(u64::try_from(counts.len()).unwrap()));
            assert_eq!(candidate.program().statistical().design().columns(), design.columns());
            let source = candidate.export_source();
            let receipt = candidate.receipt();
            identities.insert(receipt.program());
            drop(candidate);
            assert_eq!(consume_source(&source, receipt, &ctx).unwrap().receipt(), receipt);
            assert_eq!(replay(input, receipt, &ctx).unwrap().receipt(), receipt);
        }
        assert_eq!(identities.len(), 6);
    }

    #[test]
    fn candidate_refuses_each_violation() {
        let ctx = ExecutionContext::for_tests(7);
        let refused = |input: CategoricalPosteriorCandidateRequest| prepare(input, &ctx).is_err();
        assert!(prepare(request(), &ctx).is_ok());
        let mut duplicate = request();
        duplicate.sampling.subject_ids[1] = duplicate.sampling.subject_ids[0].clone();
        assert!(refused(duplicate));
        let mut dependent = request();
        dependent.sampling.design = SamplingDesign::DependentTimeSeries;
        assert!(refused(dependent));
        let mut changed = request();
        changed.posterior.level_precision = 2.0;
        assert!(refused(changed));
        let mut nonfinite = request();
        nonfinite.data.outcome[2] = f64::NAN;
        assert!(refused(nonfinite));
        // A level with one row, a row naming an undeclared level, a reference that is not first.
        assert!(refused(shaped(&[29, 1])));
        let mut undeclared = request();
        undeclared.data.levels[0] = "other".into();
        assert!(refused(undeclared));
        let mut reference = request();
        reference.data.spec.reference = "level-1".into();
        assert!(refused(reference));
        let mut minimum = request();
        minimum.data.spec.min_level_rows = 1;
        assert!(refused(minimum));
        let mut pairwise = request();
        pairwise.data.spec.pairwise = vec![("level-0".into(), "level-1".into())];
        assert!(refused(pairwise));
        // Only the estimator's minimums refuse: two levels, two rows of each.
        assert!(!refused(shaped(&[14, 15])));
        assert!(!refused(shaped(&[4_095, 2])));
        assert!(!refused(shaped(&[4; 9])));
        assert!(refused(shaped(&[1, 29])));
        assert!(refused(shaped(&[30])));
    }

    #[cfg(feature = "calibration-internal")]
    #[test]
    fn candidate_execution_replays_only_its_compiled_source() {
        let ctx = ExecutionContext::for_tests(7);
        let input = shaped(&[12, 9, 15]);
        let candidate = prepare(input.clone(), &ctx).unwrap();
        let (intervals, basis, expected) = candidate.calibration_candidate(&ctx).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(basis.dependence.as_ref(), "independent_rows");
        assert_eq!(basis.functional.as_ref(), "level_contrast_prior_predictive_k3_v1");
        assert_eq!(basis.row_count, 36);
        let encoded = expected.to_bytes();
        let expected = CategoricalPosteriorCandidateRunReceipt::from_bytes(&encoded).unwrap();
        assert!(CategoricalPosteriorCandidateRunReceipt::from_bytes(&encoded[..96]).is_err());
        drop(candidate);
        let (again, rebound_basis) =
            replay_calibration_candidate(input.clone(), expected, &ctx).unwrap();
        assert_eq!(intervals, again);
        assert_eq!(basis, rebound_basis);
        let mut forged = expected;
        forged.execution[0] ^= 1;
        assert!(replay_calibration_candidate(input.clone(), forged, &ctx).is_err());
        let mut changed = input;
        changed.data.outcome[0] += 1.0;
        assert!(replay_calibration_candidate(changed, expected, &ctx).is_err());
    }

    #[test]
    fn source_artifact_recompiles_and_rejects_tampered_source_or_expectation() {
        let ctx = ExecutionContext::for_tests(7);
        let prepared = prepare(request(), &ctx).unwrap();
        let source = prepared.export_source();
        let receipt = prepared.receipt();
        let expected = CategoricalPosteriorCompileReceipt::from_bytes(&receipt.to_bytes()).unwrap();
        drop(prepared);
        assert_eq!(consume_source(&source, expected, &ctx).unwrap().receipt(), receipt);
        let mut changed_source = source.clone();
        changed_source[20] ^= 1;
        assert!(consume_source(&changed_source, expected, &ctx).is_err());
        let mut changed_receipt = expected.to_bytes();
        changed_receipt[1] ^= 1;
        let forged = CategoricalPosteriorCompileReceipt::from_bytes(&changed_receipt).unwrap();
        assert!(consume_source(&source, forged, &ctx).is_err());
        assert!(CategoricalPosteriorCompileReceipt::from_bytes(&changed_receipt[..64]).is_err());
    }
}
