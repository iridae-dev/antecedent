//! Compiled, model-conditional candidate for a declared MNAR treated-arm shift.
//!
//! This preparation retains source and a six-layer native program. Its interval is
//! released only by the measured consumer, under an attesting coverage record, and
//! it is not an external model plugin.
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;
use antecedent_core::ExecutionContext;
use antecedent_core::mnar_fixed_shift::{FixedShiftProgram, FixedShiftSource};
#[cfg(any(test, feature = "calibration-internal"))]
use antecedent_validate::mnar_sensitivity::{
    ArmLoading, MnarData, MnarMechanism, MnarModel, MnarPoint, MnarSensitivityResult,
    MnarSensitivitySpec, mnar_sensitivity,
};
use std::sync::Arc;

#[cfg(any(test, feature = "calibration-internal"))]
fn source_error(error: antecedent_io::IoError) -> CausalError {
    match error {
        antecedent_io::IoError::Refused { code: "execution_cancelled", .. } => {
            CausalError::Cancelled { stage: "mnar source replay" }
        }
        antecedent_io::IoError::Refused { code: "execution_resource_limit", message } => {
            CausalError::Resource { message }
        }
        other => CausalError::Compile { message: other.to_string() },
    }
}

/// Source-bound compiled candidate. Its interval is released only by the measured
/// consumer, and only under a coverage record that covers this source's scope.
#[derive(Clone)]
pub struct PreparedMnarFixedShift {
    source: Arc<FixedShiftSource>,
    #[cfg_attr(
        not(any(test, feature = "calibration-internal")),
        allow(dead_code, reason = "only native replay and the calibration harness read it")
    )]
    program: FixedShiftProgram,
    lowered: Arc<LoweredProgram>,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}

impl Study {
    /// Compile the specified randomized-arm source and its declared MNAR assumption.
    /// The supplied assumptions are untested premises, not verified facts.
    ///
    /// # Errors
    /// A malformed native program, a resource refusal, or cancellation.
    pub fn prepare_mnar_fixed_shift(
        source: FixedShiftSource,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMnarFixedShift, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "mnar fixed-shift preparation" });
        }
        let bytes = (source.rows().len() as u64)
            .checked_mul(64)
            .ok_or_else(|| CausalError::Resource { message: "source size overflow".into() })?;
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "mnar fixed-shift preparation" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let program = FixedShiftProgram::compile(&source)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let lowered = crate::analysis::lowering::mnar_fixed_shift::lower(&source)?;
        let expected = source
            .lowered_identity()
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if lowered.identity() != expected {
            return Err(CausalError::Compile {
                message: "mnar fixed-shift shared compiler identity mismatch".into(),
            });
        }
        Ok(PreparedMnarFixedShift {
            source: Arc::new(source),
            program,
            lowered: Arc::new(lowered),
            _lease: Arc::new(lease),
        })
    }
}

impl PreparedMnarFixedShift {
    /// The exact checked six-layer native program; inspection executes nothing.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The retained source, including its separately declared assumptions.
    #[must_use]
    pub fn source(&self) -> &FixedShiftSource {
        &self.source
    }

    /// Execute the declared-shift sampling construction solely for the registered
    /// calibration study. The basis names this compiled candidate separately from
    /// the earlier raw-kernel candidate; no coverage or license is conferred here.
    ///
    /// # Errors
    /// Source or program mismatch, cancellation, a resource limit, or native refusal.
    #[cfg(feature = "calibration-internal")]
    #[doc(hidden)]
    pub fn interval_calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_validate::mnar_sensitivity::MnarSamplingInterval,
            antecedent_core::CalibrationBasis,
        ),
        CausalError,
    > {
        if !self.program.verifies(&self.source) {
            return Err(CausalError::Compile {
                message: "mnar source no longer matches its compiled operation".into(),
            });
        }
        let actual = self.estimate_candidate(ctx)?;
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let expected_point = declared_point(&actual, self.source.treated_shift())
            .ok_or_else(|| CausalError::Compile {
                message: "mnar declared point is absent".into(),
            })?
            .effect;
        let encoded = antecedent_io::mnar_candidate_source::encode(&self.source, &scoped)
            .map_err(source_error)?;
        let (standard_error, lower, upper, basis) =
            antecedent_io::measured_inference::mnar_candidate::calibration_candidate(
                encoded.bytes(),
                &scoped,
            )
            .map_err(source_error)?;
        if ((lower + upper) * 0.5 - expected_point).abs() > 1e-9 * expected_point.abs().max(1.0) {
            return Err(CausalError::Compile {
                message: "mnar interval candidate and retained point disagree".into(),
            });
        }
        Ok((
            antecedent_validate::mnar_sensitivity::MnarSamplingInterval {
                standard_error,
                lower,
                upper,
            },
            basis,
        ))
    }

    /// Execute only for native review, replay and the calibration harness.
    #[cfg(any(test, feature = "calibration-internal"))]
    #[doc(hidden)]
    fn estimate_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MnarSensitivityResult, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if !self.program.verifies(&self.source) {
            return Err(CausalError::Compile {
                message: "mnar source no longer matches its compiled operation".into(),
            });
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if scoped.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "mnar fixed-shift execution" });
        }
        let bytes = (self.source.rows().len() as u64)
            .checked_mul(64)
            .ok_or_else(|| CausalError::Resource { message: "source size overflow".into() })?;
        let _lease = scoped.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "mnar fixed-shift execution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let treatment: Vec<f64> =
            self.source.rows().iter().map(|r| if r.treated { 1.0 } else { 0.0 }).collect();
        let outcome: Vec<Option<f64>> = self.source.rows().iter().map(|r| r.outcome).collect();
        let spec = shift_spec(self.source.treated_shift());
        let result = mnar_sensitivity(&MnarData::new(&treatment, &outcome), &spec)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if scoped.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "mnar fixed-shift execution" });
        }
        Ok(result)
    }
}

// The validate-side range whose evaluated point is the declared shift: the treated
// arm alone loads the parameter, and the range is symmetric about missing-at-random.
#[cfg(any(test, feature = "calibration-internal"))]
fn shift_spec(shift: f64) -> MnarSensitivitySpec {
    let (reach, grid_points) =
        if shift.total_cmp(&0.0).is_eq() { (1.0, 3) } else { (shift.abs(), 2) };
    let mut spec = MnarSensitivitySpec::new(
        MnarMechanism {
            model: MnarModel::DeltaPatternMixture,
            loading: ArmLoading::treated_only(),
        },
        -reach,
        reach,
    );
    spec.grid_points = grid_points;
    spec
}

// The point of `shift_spec` at the declared shift.
#[cfg(any(test, feature = "calibration-internal"))]
fn declared_point(result: &MnarSensitivityResult, shift: f64) -> Option<&MnarPoint> {
    match shift.total_cmp(&0.0) {
        std::cmp::Ordering::Greater => result.grid.last(),
        std::cmp::Ordering::Less => result.grid.first(),
        std::cmp::Ordering::Equal => Some(&result.mar),
    }
}

// Bind every public field of the point-only result. The consumer recomputes
// this from actual execution; a digest alone never constructs an authority.
#[cfg(any(test, feature = "calibration-internal"))]
fn result_digest(result: &MnarSensitivityResult) -> Result<[u8; 32], CausalError> {
    use antecedent_core::CanonicalWriter;
    use antecedent_validate::mnar_sensitivity::{MnarPoint, MnarTipping, MnarTippingSide};
    fn side(w: &mut CanonicalWriter, s: Option<MnarTippingSide>) {
        match s {
            Some(s) => {
                w.word(1)
                    .real(s.parameter)
                    .real(s.bracket_lower)
                    .word(s.iterations as u64)
                    .word(u64::from(s.changes_to.to_le_bytes()[0]));
            }
            None => {
                w.word(0);
            }
        }
    }
    fn tipping(w: &mut CanonicalWriter, t: &MnarTipping) {
        w.text(t.kind.name())
            .text(&t.label)
            .real(t.threshold)
            .word(u64::from(t.at_mar.to_le_bytes()[0]));
        side(w, t.negative);
        side(w, t.positive);
    }
    fn point(w: &mut CanonicalWriter, p: &MnarPoint) -> Result<(), CausalError> {
        if p.sampling.is_some() {
            return Err(CausalError::Compile {
                message: "unlicensed MNAR interval in point candidate".into(),
            });
        }
        w.real(p.parameter)
            .real(p.treated_mean)
            .real(p.control_mean)
            .real(p.effect)
            .word(p.acts.len() as u64);
        for act in &p.acts {
            w.word(u64::from(u8::from(*act)));
        }
        Ok(())
    }
    let mut w = CanonicalWriter::new("antecedent.mnar.fixed_shift.result.v1");
    w.text(result.family)
        .text(result.target)
        .text(result.model.name())
        .real(result.loading.treated)
        .real(result.loading.control)
        .text(result.estimator.name());
    point(&mut w, &result.mar)?;
    w.real(result.observed_case_effect)
        .word(result.rows as u64)
        .word(result.treated_rows as u64)
        .word(result.treated_missing as u64)
        .word(result.control_missing as u64)
        .word(result.grid.len() as u64);
    for p in &result.grid {
        point(&mut w, p)?;
    }
    let range = result.assumption_range;
    w.real(range.parameter_lower)
        .real(range.parameter_upper)
        .real(range.effect_lower)
        .real(range.effect_upper)
        .word(u64::from(u8::from(range.is_confidence_interval)))
        .word(u64::from(u8::from(range.empirically_established)));
    w.word(result.tipping.len() as u64);
    for t in &result.tipping {
        tipping(&mut w, t);
    }
    w.word(result.decisions.len() as u64);
    for d in &result.decisions {
        w.text(&d.name)
            .real(d.threshold)
            .word(u64::from(u8::from(d.acts_at_mar)))
            .word(u64::from(u8::from(d.changes_within_range)));
        tipping(&mut w, &d.tipping);
    }
    w.real(result.level).real(result.tolerance).word(result.assumptions.len() as u64);
    for a in &result.assumptions {
        w.text(a.id).text(a.statement).text(a.status.as_str());
    }
    w.text(result.method)
        .text(result.interpretation)
        .text(result.inference_claim)
        .text(result.empirical_status)
        .text(result.sampling_basis)
        .text(result.sampling_status)
        .text(result.sampling_detail);
    Ok(w.finish())
}

/// Independently retained expected source and lowered program identity. This is a
/// replay constraint, never a license or a substitute for reconstruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpectedMnarFixedShift {
    /// Expected exact source and its shared native operation.
    pub source: antecedent_io::mnar_candidate_source::ExpectedMnarSource,
    /// Expected identity of the full six-layer causal lowering.
    pub lowered_program: [u8; 32],
    /// Digest of the original point-only execution, rederived during consumption.
    pub result: [u8; 32],
}

/// Source-bound point candidate plus its retained resource admission.
#[cfg(any(test, feature = "calibration-internal"))]
#[doc(hidden)]
pub struct MnarFixedShiftArtifact {
    source: antecedent_io::mnar_candidate_source::EncodedMnarSource,
    expected: ExpectedMnarFixedShift,
}
#[cfg(any(test, feature = "calibration-internal"))]
impl MnarFixedShiftArtifact {
    /// Original native source bytes.
    pub fn bytes(&self) -> &[u8] {
        self.source.bytes()
    }
    /// Independently retainable original source, lowering and point identities.
    pub const fn expectation(&self) -> &ExpectedMnarFixedShift {
        &self.expected
    }
}

impl PreparedMnarFixedShift {
    /// Encode this candidate's retained source and separately held expectation.
    /// It carries no interval, coverage record, or scientific authority.
    ///
    /// # Errors
    /// A resource, cancellation or source encoding refusal.
    #[cfg(any(test, feature = "calibration-internal"))]
    #[doc(hidden)]
    pub fn export_source(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MnarFixedShiftArtifact, CausalError> {
        let source = antecedent_io::mnar_candidate_source::encode(&self.source, ctx)
            .map_err(source_error)?;
        let result = self.estimate_candidate(ctx)?;
        let expected = ExpectedMnarFixedShift {
            source: source.expectation(),
            lowered_program: *self.lowered.identity().as_bytes(),
            result: result_digest(&result)?,
        };
        Ok(MnarFixedShiftArtifact { source, expected })
    }
}

/// Reconstruct the exact original source and native operations, independently lower
/// the whole six-layer program, and compare the result to a separately held expectation.
/// This returns a closed candidate, never a measured or licensed interval.
///
/// # Errors
/// Corrupt source, a changed expectation, resource/cancellation refusal, or lowering failure.
#[cfg(any(test, feature = "calibration-internal"))]
#[doc(hidden)]
pub fn consume_source(
    bytes: &[u8],
    expected: &ExpectedMnarFixedShift,
    ctx: &ExecutionContext,
) -> Result<PreparedMnarFixedShift, CausalError> {
    let source = antecedent_io::mnar_candidate_source::reconstruct(bytes, &expected.source, ctx)
        .map_err(source_error)?;
    let prepared = Study::prepare_mnar_fixed_shift(source.source().clone(), ctx)?;
    if prepared.lowered.identity().as_bytes() != &expected.lowered_program {
        return Err(CausalError::Compile {
            message: "mnar lowered operation differs from the retained expectation".into(),
        });
    }
    let replayed = prepared.estimate_candidate(ctx)?;
    if result_digest(&replayed)? != expected.result {
        return Err(CausalError::Compile {
            message: "mnar numerical replay differs from the retained expectation".into(),
        });
    }
    Ok(prepared)
}

impl PreparedMnarFixedShift {
    /// The calibrated interval of this exact source. The measured consumer replays the
    /// source and releases the interval only under an attesting coverage record whose
    /// construction, declared-shift and missingness band and row range cover it.
    ///
    /// # Errors
    /// `cell_not_licensed` when no such record exists, or an invalid source,
    /// cancellation or resource refusal.
    pub fn measured_interval(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::measured_inference::MeasuredInference, antecedent_io::IoError> {
        let encoded = antecedent_io::mnar_candidate_source::encode(&self.source, ctx)?;
        antecedent_io::measured_inference::mnar_candidate::measure_source(encoded.bytes(), ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::mnar_fixed_shift::{SamplingPremise, SourceRow};

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
            "population".into(),
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

    fn source() -> FixedShiftSource {
        at_sd(0.8, rows((30, 24), (40, 28)))
    }

    const MULTIPLES: [f64; 5] = [-1.9, -0.5, 0.0, 0.7, 2.0];

    #[test]
    fn declared_shift_executes_model_conditional_point_through_retained_program() {
        let ctx = ExecutionContext::for_tests(12);
        for design in [rows((30, 24), (40, 28)), rows((15, 9), (90, 60))] {
            for multiple in MULTIPLES {
                let data = at_sd(multiple, design.clone());
                let prepared = Study::prepare_mnar_fixed_shift(data.clone(), &ctx).unwrap();
                let result = prepared.estimate_candidate(&ctx).unwrap();
                let mean = |treated: bool| {
                    let y: Vec<f64> = data
                        .rows()
                        .iter()
                        .filter(|r| r.treated == treated)
                        .filter_map(|r| r.outcome)
                        .collect();
                    let assigned = data.rows().iter().filter(|r| r.treated == treated).count();
                    (y.iter().sum::<f64>() / y.len() as f64, y.len() as f64 / assigned as f64)
                };
                let (treated, share) = mean(true);
                let expected = treated + (1.0 - share) * data.treated_shift() - mean(false).0;
                let point = declared_point(&result, data.treated_shift()).unwrap();
                assert!((point.effect - expected).abs() < 1e-9);
                assert!(result.grid.iter().all(|point| point.sampling.is_none()));
                assert_eq!(result.sampling_status, "cell_not_licensed");
            }
        }
    }

    #[test]
    fn standard_error_matches_the_validated_sensitivity_construction() {
        use antecedent_validate::mnar_sensitivity::mnar_sensitivity_calibration_candidate;
        let ctx = ExecutionContext::for_tests(16);
        for design in
            [rows((30, 24), (40, 28)), rows((15, 9), (90, 60)), rows((200, 150), (60, 31))]
        {
            for multiple in MULTIPLES {
                let data = at_sd(multiple, design.clone());
                let treatment: Vec<f64> =
                    data.rows().iter().map(|r| if r.treated { 1.0 } else { 0.0 }).collect();
                let outcome: Vec<Option<f64>> = data.rows().iter().map(|r| r.outcome).collect();
                let reference = mnar_sensitivity_calibration_candidate(
                    &MnarData::new(&treatment, &outcome),
                    &shift_spec(data.treated_shift()),
                )
                .unwrap();
                let expected = declared_point(&reference, data.treated_shift())
                    .and_then(|point| point.sampling)
                    .unwrap();
                let encoded = antecedent_io::mnar_candidate_source::encode(&data, &ctx).unwrap();
                let (standard_error, lower, upper, basis) =
                    antecedent_io::measured_inference::mnar_candidate::calibration_candidate(
                        encoded.bytes(),
                        &ctx,
                    )
                    .unwrap();
                assert!((standard_error - expected.standard_error).abs() < 1e-9);
                assert!((lower - expected.lower).abs() < 1e-9);
                assert!((upper - expected.upper).abs() < 1e-9);
                assert_eq!(basis.row_count, data.rows().len() as u64);
            }
        }
    }

    #[test]
    fn a_source_below_the_row_bound_is_refused_with_a_typed_error() {
        let ctx = ExecutionContext::for_tests(15);
        assert!(Study::prepare_mnar_fixed_shift(source(), &ctx).is_ok());
        let few = FixedShiftSource::new(
            "p".into(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            0.0,
            rows((20, 16), (20, 16))[..39].to_vec(),
        );
        assert_eq!(
            few.unwrap_err(),
            antecedent_core::mnar_fixed_shift::SourceError::RowCountOutOfBounds
        );
    }

    #[test]
    fn declared_shift_replays_compiled_source() {
        let ctx = ExecutionContext::for_tests(13);
        let prepared = Study::prepare_mnar_fixed_shift(source(), &ctx).unwrap();
        let result = prepared.estimate_candidate(&ctx).unwrap();
        assert!(result.mar.sampling.is_none());
        assert!(result.grid.iter().all(|point| point.sampling.is_none()));
        let artifact = prepared.export_source(&ctx).unwrap();
        let expected = *artifact.expectation();
        let mut bytes = artifact.bytes().to_vec();
        let replayed = consume_source(&bytes, &expected, &ctx).unwrap();
        assert_eq!(prepared.lowered().identity(), replayed.lowered().identity());
        assert_eq!(
            prepared.estimate_candidate(&ctx).unwrap(),
            replayed.estimate_candidate(&ctx).unwrap()
        );
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(consume_source(&bytes, &expected, &ctx).is_err());
    }

    #[test]
    fn declared_shift_is_bound_into_the_lowered_identity() {
        let ctx = ExecutionContext::for_tests(17);
        let design = rows((30, 24), (40, 28));
        let identity = |multiple: f64| {
            *Study::prepare_mnar_fixed_shift(at_sd(multiple, design.clone()), &ctx)
                .unwrap()
                .lowered()
                .identity()
                .as_bytes()
        };
        assert_eq!(identity(0.5), identity(0.5));
        assert_ne!(identity(0.5), identity(-0.5));
        assert_ne!(identity(0.5), identity(0.6));
    }
}
