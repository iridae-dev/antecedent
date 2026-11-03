//! Native source for the declared-shift MNAR calibration candidate.
//!
//! This is a declaration and a reproducible program fingerprint, not a causal proof,
//! interval result, coverage record, or authority to publish an interval. A consumer
//! must reconstruct it from the original rows and independently verify the associated
//! compiled identification and any applicable measured-inference evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::CanonicalWriter;
use crate::operator::{
    Operator, OperatorDeclaration, OperatorEffect, OperatorError, OperatorGraph, OperatorPorts,
    OperatorRole,
};

/// Width of one declared-shift band of the calibration scope, in standard deviations.
pub const SHIFT_BAND_WIDTH: f64 = 0.5;
/// Width of one missingness band of the calibration scope.
pub const MISSING_BAND_WIDTH: f64 = 0.125;

/// A single assigned arm and an outcome which may be unobserved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceRow {
    /// Subject identity, unique across both arms within this source.
    pub subject_id: u64,
    /// The row belongs to the treated arm.
    pub treated: bool,
    /// Observed outcome, or `None` when the outcome is missing.
    pub outcome: Option<f64>,
}

/// A caller-declared premise, never inferred from the row values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingPremise {
    /// Independently sampled units in two declared randomized treatment arms.
    IndependentRandomizedArms,
}

/// The complete native declaration accepted by this single calibration candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedShiftSource {
    population: String,
    treatment: u32,
    outcome: u32,
    premise: SamplingPremise,
    treated_shift: f64,
    rows: Vec<SourceRow>,
}

/// Why the source cannot enter the bounded candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SourceError {
    /// Population or variable roles are absent or ambiguous.
    #[error("population must be 1 to 256 bytes and treatment and outcome must be distinct")]
    InvalidDeclaration,
    /// Subject identity occurs more than once in the source.
    #[error("subject identities must be distinct across rows")]
    DuplicateSubject,
    /// A retained value is non-finite, or its squares overflow the scale estimate.
    #[error("observed outcome must be finite with a finite scale")]
    InvalidOutcome,
    /// One arm contains fewer than two observed outcomes, so it has no sample variance.
    #[error("each arm must contain at least two observed outcomes")]
    InsufficientObservations,
    /// Every observed outcome in each arm is identical, so the scale is zero.
    #[error("observed outcomes must vary within at least one arm")]
    ConstantOutcome,
    /// The declared shift is not a finite number.
    #[error("treated shift must be finite")]
    InvalidShift,
}

impl SourceError {
    /// Stable refusal detail, without the route prefix.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::InvalidDeclaration => "invalid_declaration",
            Self::DuplicateSubject => "duplicate_subject",
            Self::InvalidOutcome => "invalid_outcome",
            Self::InsufficientObservations => "insufficient_observations",
            Self::ConstantOutcome => "constant_outcome",
            Self::InvalidShift => "invalid_shift",
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is finite and non-negative; a saturating index is its own band"
)]
fn band(value: f64, width: f64) -> usize {
    (value / width) as usize
}

fn band_text(lower: f64, upper: f64, places: usize) -> String {
    format!("{lower:.places$}_{upper:.places$}").replace('.', "p")
}

/// The calibration-scope label of one declared-shift band and one missingness band.
/// Records are measured, and bound at run time, per such cell.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "band indices are small counts")]
pub fn scope_functional(shift_band: usize, missing_band: usize) -> String {
    let shift = shift_band as f64 * SHIFT_BAND_WIDTH;
    let missing = missing_band as f64 * MISSING_BAND_WIDTH;
    format!(
        "mnar_treated_shift_sd_{}_missing_{}",
        band_text(shift, shift + SHIFT_BAND_WIDTH, 1),
        band_text(missing, missing + MISSING_BAND_WIDTH, 3)
    )
}

impl FixedShiftSource {
    /// Validate and retain original rows with the declared sampling premise and the
    /// declared treated-arm missing-pattern shift.
    ///
    /// `treated_shift` is the mean difference, in outcome units, between the treated
    /// arm's missing and observed outcomes; the control arm's difference is zero.
    /// Finite observed outcomes are accepted; no outcome or observation probability
    /// is assumed known or inferred from the sample. The shift and the randomized
    /// independent-arm design are model premises, not conclusions from these rows.
    /// Coverage has not been established.
    ///
    /// # Errors
    /// The source has invalid identities or shift, an invalid observed outcome, an arm
    /// with fewer than two observed outcomes, or no variation in the observed outcomes.
    pub fn new(
        population: String,
        treatment: u32,
        outcome: u32,
        premise: SamplingPremise,
        treated_shift: f64,
        rows: Vec<SourceRow>,
    ) -> Result<Self, SourceError> {
        if population.is_empty() || population.len() > 256 || treatment == outcome {
            return Err(SourceError::InvalidDeclaration);
        }
        if !treated_shift.is_finite() {
            return Err(SourceError::InvalidShift);
        }
        let mut observed = [0usize; 2];
        let mut subjects = std::collections::HashSet::with_capacity(rows.len());
        for row in &rows {
            if !subjects.insert(row.subject_id) {
                return Err(SourceError::DuplicateSubject);
            }
            let arm = usize::from(row.treated);
            if let Some(value) = row.outcome {
                if !value.is_finite() {
                    return Err(SourceError::InvalidOutcome);
                }
                observed[arm] += 1;
            }
        }
        if observed.iter().any(|count| *count < 2) {
            return Err(SourceError::InsufficientObservations);
        }
        let source = Self {
            population,
            treatment,
            outcome,
            premise,
            // Adding zero turns a negative zero into the positive one, so one
            // declaration has one digest.
            treated_shift: treated_shift + 0.0,
            rows,
        };
        let scale = source.pooled_outcome_sd();
        if !scale.is_finite() {
            return Err(SourceError::InvalidOutcome);
        }
        if scale <= 0.0 {
            return Err(SourceError::ConstantOutcome);
        }
        Ok(source)
    }

    /// Declared treated-arm shift of the missing pattern, in outcome units.
    #[must_use]
    pub const fn treated_shift(&self) -> f64 {
        self.treated_shift
    }

    /// Pooled within-arm standard deviation of the observed outcomes.
    #[must_use]
    pub fn pooled_outcome_sd(&self) -> f64 {
        let mut sum_squares = 0.0;
        let mut degrees = 0.0;
        for treated in [false, true] {
            let values: Vec<f64> = self
                .rows
                .iter()
                .filter(|row| row.treated == treated)
                .filter_map(|row| row.outcome)
                .collect();
            #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
            let count = values.len() as f64;
            let mean = values.iter().sum::<f64>() / count;
            sum_squares += values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            degrees += count - 1.0;
        }
        (sum_squares / degrees).sqrt()
    }

    /// Largest share of an arm's assigned rows whose outcome is missing.
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
    pub fn max_missing_fraction(&self) -> f64 {
        [false, true]
            .into_iter()
            .map(|treated| {
                let arm = self.rows.iter().filter(|row| row.treated == treated);
                let assigned = arm.clone().count() as f64;
                arm.filter(|row| row.outcome.is_none()).count() as f64 / assigned
            })
            .fold(0.0, f64::max)
    }

    /// The calibration-scope label this execution falls in: the magnitude of the
    /// declared shift in pooled standard deviations, and the largest arm missingness.
    #[must_use]
    pub fn scope_functional(&self) -> String {
        scope_functional(
            band(self.treated_shift.abs() / self.pooled_outcome_sd(), SHIFT_BAND_WIDTH),
            band(self.max_missing_fraction(), MISSING_BAND_WIDTH),
        )
    }

    /// Declared population identifier.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.population
    }

    /// Treatment variable identity in the declared source schema.
    #[must_use]
    pub const fn treatment(&self) -> u32 {
        self.treatment
    }

    /// Outcome variable identity in the declared source schema.
    #[must_use]
    pub const fn outcome(&self) -> u32 {
        self.outcome
    }

    /// Original rows; a verifier must reconstruct from these rather than a supplied digest.
    #[must_use]
    pub fn rows(&self) -> &[SourceRow] {
        &self.rows
    }

    /// Digest the complete original source, including missingness and assignment.
    #[must_use]
    pub fn source_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.mnar.fixed_shift.source.v3");
        writer.text(&self.population).word(u64::from(self.treatment)).word(u64::from(self.outcome));
        writer.real(self.treated_shift).word(self.rows.len() as u64);
        for row in &self.rows {
            writer.word(row.subject_id).word(u64::from(u8::from(row.treated)));
            match row.outcome {
                Some(value) => {
                    writer.word(1).real(value);
                }
                None => {
                    writer.word(0);
                }
            }
        }
        writer.finish()
    }

    /// Independently reconstruct the complete fixed six-layer compiler identity.
    /// This identity grants no scientific or measurement authority.
    ///
    /// # Errors
    /// The fixed native operator declaration cannot be reconstructed.
    pub fn lowered_identity(&self) -> Result<crate::SemanticDigest, OperatorError> {
        let native = FixedShiftProgram::compile(self)?;
        let source = self.source_digest();
        let query = native.identity();
        let mut statistical = CanonicalWriter::new("antecedent.lowered_program.statistical.v2");
        statistical
            .opt_text(None)
            .opt_text(None)
            .text("exact_point")
            .text("mnar.fixed_treated_shift")
            .word(0)
            .word(0)
            .word(0)
            .word(0)
            .opt_word(None)
            .text("not_design_based")
            .word(native.operators().len() as u64);
        let mut evidence = CanonicalWriter::new("antecedent.lowered_program.evidence");
        evidence.word(1).text("trial").word(1).bytes(&source).text("single").word(0);
        let source_hex = crate::SemanticDigest::from_bytes(source).to_hex();
        let mut writer = CanonicalWriter::new("antecedent.lowered_program");
        writer
            .text("MnarFixedShift")
            .text("complete")
            .text("query_functional")
            .text("assignment_contrast")
            .opt_word(Some(u64::from(self.outcome)))
            .bytes(query)
            .text("mnar.fixed_treated_shift")
            .text("tabular")
            .text("none")
            .word(2)
            .word(u64::from(self.treatment))
            .word(u64::from(self.outcome))
            .opt_text(Some(source_hex.as_str()))
            .opt_text(None)
            .word(0)
            .text("non_graphical")
            .text("specified_model_conditional")
            .bytes(query)
            .bytes(&statistical.finish())
            .bytes(&evidence.finish())
            .text("none");
        Ok(crate::SemanticDigest::from_bytes(writer.finish()))
    }

    /// Reconstruct the fixed statistical operation from the source and its stated premise.
    /// This digest is an identity, not proof that the sampling premise is true.
    #[must_use]
    pub fn program_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.mnar.fixed_shift.program.v2");
        writer.text(&self.population).word(u64::from(self.treatment)).word(u64::from(self.outcome));
        match self.premise {
            SamplingPremise::IndependentRandomizedArms => {
                writer.text("independent_randomized_arms");
            }
        }
        writer
            .text("delta_pattern_mixture")
            .text("difference_in_means")
            .text("treated_only")
            .real(self.treated_shift)
            .real(0.95)
            .text("conditional_delta_normal");
        writer.finish()
    }
}

/// One fixed native operation in execution order; no caller-supplied likelihood is admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedShiftOperation {
    /// Read the assigned arms, observed outcomes, and missingness flags.
    ReadObservedArms,
    /// Apply the declared treated-arm shift to the missing pattern.
    ApplyTreatedShift,
    /// Evaluate the fixed conditional delta-method construction.
    EvaluateConditionalDelta,
    /// Report the interval only under an attesting coverage record for its scope.
    ReportMeasuredInterval,
}

/// The sole identification interpretation: conditional on declared, untested premises.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedShiftIdentification {
    /// Two independently sampled randomized arms and the fixed missing-pattern model.
    /// The observed rows do not prove randomization, independence, or the missing law.
    ConditionalOnDeclaredDesignAndMissingness,
}

/// A reconstructed bounded operation, not a causal or calibration license.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedShiftProgram {
    source_digest: [u8; 32],
    identification: FixedShiftIdentification,
    operators: [Operator; 2],
    operations: [FixedShiftOperation; 4],
    identity: [u8; 32],
}

impl FixedShiftProgram {
    /// Reconstruct the only admitted operator sequence from the retained source.
    ///
    /// # Errors
    /// A native operator declaration violates the shared graph/role contract.
    pub fn compile(source: &FixedShiftSource) -> Result<Self, OperatorError> {
        let intervention = Operator::new(OperatorDeclaration::new(
            OperatorEffect::Intervene,
            OperatorGraph::Causal,
            OperatorPorts::new([OperatorRole::Population], [OperatorRole::WorldOrSchedule]),
        ))?;
        let read = Operator::new(OperatorDeclaration::new(
            OperatorEffect::Read,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Population], [OperatorRole::Outcome]),
        ))?;
        let operators = [intervention, read];
        let operations = [
            FixedShiftOperation::ReadObservedArms,
            FixedShiftOperation::ApplyTreatedShift,
            FixedShiftOperation::EvaluateConditionalDelta,
            FixedShiftOperation::ReportMeasuredInterval,
        ];
        let mut identity = CanonicalWriter::new("antecedent.mnar.fixed_shift.checked_program.v2");
        identity.bytes(&source.program_digest()).word(operators.len() as u64);
        for operator in &operators {
            identity
                .text(operator.effect().as_str())
                .text(operator.graph().as_str())
                .word(operator.inputs().len() as u64);
            for role in operator.inputs() {
                identity.text(role.as_str());
            }
            identity.word(operator.outputs().len() as u64);
            for role in operator.outputs() {
                identity.text(role.as_str());
            }
            identity
                .word(u64::from(u8::from(operator.causal_feedback())))
                .text(operator.frozen_fit_reuse().as_str());
        }
        identity.word(operations.len() as u64);
        for operation in operations {
            identity.text(match operation {
                FixedShiftOperation::ReadObservedArms => "read_observed_arms",
                FixedShiftOperation::ApplyTreatedShift => "apply_treated_shift",
                FixedShiftOperation::EvaluateConditionalDelta => "evaluate_conditional_delta",
                FixedShiftOperation::ReportMeasuredInterval => "report_measured_interval",
            });
        }
        Ok(Self {
            source_digest: source.source_digest(),
            identification: FixedShiftIdentification::ConditionalOnDeclaredDesignAndMissingness,
            operators,
            operations,
            identity: identity.finish(),
        })
    }

    /// The conditional identification derived from the one supported declaration.
    #[must_use]
    pub const fn identification(&self) -> FixedShiftIdentification {
        self.identification
    }

    /// Checked native causal and execution graph operators.
    #[must_use]
    pub fn operators(&self) -> &[Operator; 2] {
        &self.operators
    }

    /// The native statistical operation sequence.
    #[must_use]
    pub fn operations(&self) -> &[FixedShiftOperation; 4] {
        &self.operations
    }

    /// Identity derived from the declared target and all fixed construction premises; source rows have their own digest.
    #[must_use]
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Reconstruct the sequence from the actual source and compare every retained field.
    /// A supplied digest, matching or otherwise, is never itself accepted as proof.
    #[must_use]
    pub fn verifies(&self, source: &FixedShiftSource) -> bool {
        Self::compile(source).is_ok_and(|reconstructed| self == &reconstructed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(first: u64, treated: bool, assigned: usize, observed: usize) -> Vec<SourceRow> {
        (0..assigned)
            .map(|i| SourceRow {
                subject_id: first + i as u64,
                treated,
                outcome: (i < observed).then(|| {
                    f64::from(u8::try_from((i * 7) % 11).unwrap()) / 3.0
                        + f64::from(u8::from(treated))
                }),
            })
            .collect()
    }

    fn rows(control: (usize, usize), treated: (usize, usize)) -> Vec<SourceRow> {
        let mut rows = arm(0, false, control.0, control.1);
        rows.extend(arm(100_000, true, treated.0, treated.1));
        rows
    }

    fn build(shift: f64, rows: Vec<SourceRow>) -> Result<FixedShiftSource, SourceError> {
        FixedShiftSource::new(
            "population".into(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            shift,
            rows,
        )
    }

    fn source() -> FixedShiftSource {
        build(0.75, rows((30, 24), (40, 28))).unwrap()
    }

    #[test]
    fn native_program_reconstructs_from_rows_and_refuses_changed_source() {
        let original = source();
        let program = FixedShiftProgram::compile(&original).unwrap();
        assert!(program.verifies(&original));
        let mut changed = original.clone();
        changed.rows[0].outcome = None;
        assert!(!program.verifies(&changed));
        assert_eq!(program.identity(), FixedShiftProgram::compile(&changed).unwrap().identity());
        assert_ne!(original.source_digest(), changed.source_digest());
    }

    #[test]
    fn declared_shift_is_bound_into_every_identity() {
        let a = build(0.5, rows((30, 24), (40, 28))).unwrap();
        let b = build(-0.5, rows((30, 24), (40, 28))).unwrap();
        assert_ne!(a.source_digest(), b.source_digest());
        assert_ne!(a.program_digest(), b.program_digest());
        assert_ne!(
            FixedShiftProgram::compile(&a).unwrap().identity(),
            FixedShiftProgram::compile(&b).unwrap().identity()
        );
        assert_ne!(a.lowered_identity().unwrap(), b.lowered_identity().unwrap());
        let zero = build(0.0, rows((30, 24), (40, 28))).unwrap();
        let negative_zero = build(-0.0, rows((30, 24), (40, 28))).unwrap();
        assert_eq!(zero.source_digest(), negative_zero.source_digest());
    }

    #[test]
    fn source_refuses_duplicate_subjects_across_arms() {
        let mut rows = source().rows().to_vec();
        rows[40].subject_id = rows[0].subject_id;
        assert_eq!(build(0.0, rows).unwrap_err(), SourceError::DuplicateSubject);
    }

    #[test]
    fn source_refuses_each_violation_with_its_own_error() {
        let ok = || rows((30, 24), (40, 28));
        let mut nan = ok();
        nan[0].outcome = Some(f64::NAN);
        assert_eq!(build(0.0, nan).unwrap_err(), SourceError::InvalidOutcome);
        assert_eq!(build(f64::NAN, ok()).unwrap_err(), SourceError::InvalidShift);
        assert_eq!(build(f64::INFINITY, ok()).unwrap_err(), SourceError::InvalidShift);
        assert!(build(1.0e3, ok()).is_ok());
        let mut overflow = ok();
        overflow[0].outcome = Some(1.0e200);
        assert_eq!(build(0.0, overflow).unwrap_err(), SourceError::InvalidOutcome);
        assert_eq!(
            build(0.0, rows((30, 0), (40, 28))).unwrap_err(),
            SourceError::InsufficientObservations
        );
        assert_eq!(
            build(0.0, rows((30, 1), (40, 28))).unwrap_err(),
            SourceError::InsufficientObservations
        );
        assert!(build(0.0, rows((30, 14), (40, 28))).is_ok());
        let constant: Vec<SourceRow> = ok()
            .into_iter()
            .map(|row| SourceRow { outcome: row.outcome.map(|_| 3.0), ..row })
            .collect();
        assert_eq!(build(0.0, constant).unwrap_err(), SourceError::ConstantOutcome);
        let empty_population = FixedShiftSource::new(
            String::new(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            0.0,
            ok(),
        );
        assert_eq!(empty_population.unwrap_err(), SourceError::InvalidDeclaration);
        let same_roles = FixedShiftSource::new(
            "p".into(),
            1,
            1,
            SamplingPremise::IndependentRandomizedArms,
            0.0,
            ok(),
        );
        assert_eq!(same_roles.unwrap_err(), SourceError::InvalidDeclaration);
    }

    #[test]
    fn only_the_estimator_minimums_bound_the_source() {
        assert!(build(0.0, rows((2, 2), (2, 2))).is_ok());
        assert_eq!(
            build(0.0, rows((2, 2), (2, 1))).unwrap_err(),
            SourceError::InsufficientObservations
        );
        assert!(build(0.0, rows((15, 15), (200, 150))).is_ok());
        assert!(build(0.0, rows((100, 2), (100, 2))).is_ok());
        assert!(build(0.0, rows((20_000, 15_000), (20_000, 15_000))).is_ok());
        let data = rows((30, 24), (40, 28));
        let scale = build(0.0, data.clone()).unwrap().pooled_outcome_sd();
        assert!(build(5.0 * scale, data.clone()).is_ok());
        assert!(build(-50.0 * scale, data).is_ok());
    }

    fn at_sd(multiple: f64, rows: Vec<SourceRow>) -> FixedShiftSource {
        let scale = build(0.0, rows.clone()).unwrap().pooled_outcome_sd();
        build(multiple * scale, rows).unwrap()
    }

    #[test]
    fn scope_label_follows_declared_shift_and_missingness() {
        assert_eq!(scope_functional(0, 0), "mnar_treated_shift_sd_0p0_0p5_missing_0p000_0p125");
        let low = at_sd(0.0, rows((30, 30), (40, 40)));
        assert_eq!(low.scope_functional(), scope_functional(0, 0));
        let high = at_sd(-1.9, rows((30, 15), (40, 20)));
        assert_eq!(high.scope_functional(), scope_functional(3, 4));
        let beyond = at_sd(6.0, rows((30, 9), (40, 28)));
        assert_eq!(beyond.scope_functional(), scope_functional(12, 5));
        assert!((high.max_missing_fraction() - 0.5).abs() < 1e-12);
        let middle = at_sd(0.75, rows((30, 24), (40, 28)));
        assert_eq!(middle.scope_functional(), scope_functional(1, 2));
    }
}
