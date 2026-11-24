//! The exposure-error corrections as compiled program families.
//!
//! Four corrections of an exposure recorded with error are each one kernel call over inputs the
//! caller declares in full (`antecedent_estimate::measurement`):
//!
//! - `correct_exposure_misclassification`: case-control counts of a misclassified binary
//!   exposure, corrected by declared or validated error rates
//!   (`ExposureMisclassificationCorrection`, unit `measurement.exposure_misclassification`);
//! - `correct_soft_exposure_effect`: a linear outcome model on per-unit exposure probabilities
//!   (`SoftExposureCorrection`, unit `measurement.soft_exposure`);
//! - `correct_detection_dependent_prevalence`: a prevalence detected with a probability that
//!   depends on a covariate (`DetectionDependentPrevalenceCorrection`, unit
//!   `measurement.detection_dependent_prevalence`);
//! - `fit_repeat_codings`: repeat codings of one latent facet as a latent-class model
//!   (`RepeatCodingsLatentClass`, unit `measurement.repeat_codings_latent_class`).
//!
//! The kernels read borrowed slices; the owned inputs here ([`SoftExposureColumns`],
//! [`SoftExposureMethod`], [`DetectionColumns`], [`RepeatCodings`]) hold the same values and
//! hand the kernel the borrowed view it takes. Compiling binds the canonical digest of every
//! value and declaration into a six-layer program; running passes the support gate of the
//! route's unit and then makes the very call the direct path makes, so every number is the
//! direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_estimate::measurement::{
    CorrectedExposure, DetectionDependentPrevalence, MeasurementError, SoftExposureCombination,
    SoftExposureEffect, correct_detection_dependent_prevalence, correct_exposure_misclassification,
    correct_soft_exposure_effect, fit_repeat_codings,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, DetectionData, DetectionDeclaration, DetectionDirection, DetectionFunction,
    ExposureErrorStructure, ExposureOutcomeCounts, FalsePositiveRate, LatentClassConstraints,
    LatentClassData, LatentClassFit, LatentClassSettings, SoftExposureData,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const EXPOSURE_MISCLASSIFICATION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExposureMisclassificationCorrection,
    plan_id: "measurement.exposure_misclassification",
    functional: FunctionalKind::Association,
    evaluator: "misclassification_corrected_log_odds_ratio",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "case-control exposure counts corrected by declared or validated error rates",
};

const SOFT_EXPOSURE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SoftExposureCorrection,
    plan_id: "measurement.soft_exposure",
    functional: FunctionalKind::Association,
    evaluator: "soft_exposure_linear_outcome_model",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "linear outcome model on per-unit exposure probabilities",
};

const DETECTION_PREVALENCE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::DetectionDependentPrevalenceCorrection,
    plan_id: "measurement.detection_dependent_prevalence",
    functional: FunctionalKind::Association,
    evaluator: "detection_adjusted_prevalence",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "recorded detections under a declared covariate-dependent detection function",
};

const REPEAT_CODINGS: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::RepeatCodingsLatentClass,
    plan_id: "measurement.repeat_codings_latent_class",
    functional: FunctionalKind::Association,
    evaluator: "latent_class_em_repeat_codings",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "repeat codings of one latent facet as replicates of one latent level",
};

/// The support refusal as a measurement refusal: reason `not_supported` and the gate's detail.
pub(super) fn measurement_refusal(refusal: &NotSupported) -> MeasurementError {
    MeasurementError {
        code: refusal.code(),
        detail: refusal.detail,
        message: refusal.message().to_owned(),
    }
}

fn write_rates(writer: &mut CanonicalWriter, rates: &BinaryErrorRates) {
    writer
        .real(rates.sensitivity)
        .real(rates.specificity)
        .real(rates.sensitivity_variance)
        .real(rates.specificity_variance);
}

/// Owned soft-exposure columns: the outcome, the per-unit exposure probabilities and the
/// column-major covariates. [`Self::data`] is the borrowed view the kernel reads.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftExposureColumns {
    /// Outcome.
    pub outcome: Vec<f64>,
    /// Per-unit probability that the exposure is present, each in `[0, 1]`.
    pub probability: Vec<f64>,
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: Vec<f64>,
    /// Number of covariate columns.
    pub covariate_count: usize,
}

impl SoftExposureColumns {
    /// The borrowed view `correct_soft_exposure_effect` reads.
    #[must_use]
    pub fn data(&self) -> SoftExposureData<'_> {
        SoftExposureData {
            outcome: &self.outcome,
            probability: &self.probability,
            covariates: &self.covariates,
            covariate_count: self.covariate_count,
        }
    }
}

/// How the soft exposure enters the outcome model; the owned form of
/// `SoftExposureCombination`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SoftExposureMethod {
    /// The per-unit posterior mean replaces the exposure.
    PosteriorMean,
    /// Outcome-conditioned draws are generated and combined by Rubin's rules.
    MultipleImputation {
        /// Imputations, at least two.
        draw_count: usize,
        /// Seed of the draws.
        seed: u64,
    },
    /// Caller-supplied outcome-conditioned `{0, 1}` draws, draw-major, combined by Rubin's
    /// rules.
    SuppliedDraws {
        /// Draws, `draw_count * n` values.
        draws: Vec<f64>,
        /// Number of draws.
        draw_count: usize,
    },
}

impl SoftExposureMethod {
    /// The borrowed combination `correct_soft_exposure_effect` reads.
    #[must_use]
    pub fn combination(&self) -> SoftExposureCombination<'_> {
        match self {
            Self::PosteriorMean => SoftExposureCombination::PosteriorMean,
            Self::MultipleImputation { draw_count, seed } => {
                SoftExposureCombination::MultipleImputation { draw_count: *draw_count, seed: *seed }
            }
            Self::SuppliedDraws { draws, draw_count } => {
                SoftExposureCombination::SuppliedDraws { draws, draw_count: *draw_count }
            }
        }
    }
}

/// Owned detection columns: whether the condition was recorded, and the covariate driving
/// detection, per unit. [`Self::data`] is the borrowed view the kernel reads.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionColumns {
    /// Whether the condition was recorded as present, per unit.
    pub detected: Vec<bool>,
    /// Covariate driving detection, per unit.
    pub covariate: Vec<f64>,
}

impl DetectionColumns {
    /// The borrowed view `correct_detection_dependent_prevalence` reads.
    #[must_use]
    pub fn data(&self) -> DetectionData<'_> {
        DetectionData { detected: &self.detected, covariate: &self.covariate }
    }
}

/// Owned repeat codings of one latent facet. [`Self::data`] is the borrowed view the kernel
/// reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatCodings {
    /// Category index of every coding of every unit.
    pub codings: Vec<u8>,
    /// Units.
    pub units: usize,
    /// Codings per unit.
    pub coding_count: usize,
    /// Categories per coding.
    pub categories: usize,
    /// Latent levels.
    pub latent_levels: usize,
}

impl RepeatCodings {
    /// The borrowed view `fit_repeat_codings` reads.
    #[must_use]
    pub fn data(&self) -> LatentClassData<'_> {
        LatentClassData {
            codings: &self.codings,
            units: self.units,
            coding_count: self.coding_count,
            categories: self.categories,
            latent_levels: self.latent_levels,
        }
    }
}

#[derive(Clone, Debug)]
struct MisclassificationRequest {
    counts: ExposureOutcomeCounts,
    structure: ExposureErrorStructure,
    min_youden: f64,
}

#[derive(Clone, Debug)]
struct SoftExposureRequest {
    columns: SoftExposureColumns,
    method: SoftExposureMethod,
}

#[derive(Clone, Debug)]
struct DetectionRequest {
    columns: DetectionColumns,
    declaration: DetectionDeclaration,
    false_positive: FalsePositiveRate,
    min_net_detection: f64,
}

#[derive(Clone, Debug)]
struct RepeatCodingsRequest {
    codings: RepeatCodings,
    constraints: LatentClassConstraints,
    settings: LatentClassSettings,
}

fn misclassification_bytes(request: &MisclassificationRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.exposure_misclassification");
    let counts = &request.counts;
    writer.words(&[counts.exposed_cases, counts.cases, counts.exposed_controls, counts.controls]);
    match &request.structure {
        ExposureErrorStructure::Shared(rates) => {
            writer.text("shared");
            write_rates(&mut writer, rates);
        }
        ExposureErrorStructure::ByOutcome { controls, cases } => {
            writer.text("by_outcome");
            write_rates(&mut writer, controls);
            write_rates(&mut writer, cases);
        }
    }
    writer.real(request.min_youden);
    writer.finish()
}

fn soft_exposure_bytes(request: &SoftExposureRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.soft_exposure");
    let columns = &request.columns;
    writer
        .reals(&columns.outcome)
        .reals(&columns.probability)
        .reals(&columns.covariates)
        .word(columns.covariate_count as u64);
    match &request.method {
        SoftExposureMethod::PosteriorMean => {
            writer.text("posterior_mean");
        }
        SoftExposureMethod::MultipleImputation { draw_count, seed } => {
            writer.text("multiple_imputation").word(*draw_count as u64).word(*seed);
        }
        SoftExposureMethod::SuppliedDraws { draws, draw_count } => {
            writer.text("supplied_draws").reals(draws).word(*draw_count as u64);
        }
    }
    writer.finish()
}

fn detection_bytes(request: &DetectionRequest) -> [u8; 32] {
    let mut writer =
        CanonicalWriter::new("antecedent.kernel_program.detection_dependent_prevalence");
    let detected: Vec<u8> = request.columns.detected.iter().map(|&d| u8::from(d)).collect();
    writer.bytes(&detected).reals(&request.columns.covariate);
    match &request.declaration {
        DetectionDeclaration::Full(DetectionFunction::Logistic {
            intercept,
            slope,
            intercept_variance,
            covariance,
            slope_variance,
        }) => {
            writer
                .text("logistic")
                .real(*intercept)
                .real(*slope)
                .real(*intercept_variance)
                .real(*covariance)
                .real(*slope_variance);
        }
        DetectionDeclaration::Full(DetectionFunction::Table { knots, values }) => {
            writer.text("table").reals(knots).reals(values);
        }
        DetectionDeclaration::Partial { direction, anchors } => {
            writer.text("envelope").text(match direction {
                DetectionDirection::Nondecreasing => "nondecreasing",
                DetectionDirection::Nonincreasing => "nonincreasing",
            });
            writer.word(anchors.len() as u64);
            for anchor in anchors {
                writer.real(anchor.covariate).real(anchor.lower).real(anchor.upper);
            }
        }
    }
    writer
        .real(request.false_positive.rate)
        .real(request.false_positive.variance)
        .real(request.min_net_detection);
    writer.finish()
}

fn repeat_codings_bytes(request: &RepeatCodingsRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.repeat_codings_latent_class");
    let codings = &request.codings;
    writer.bytes(&codings.codings).words(&[
        codings.units as u64,
        codings.coding_count as u64,
        codings.categories as u64,
        codings.latent_levels as u64,
    ]);
    writer.word(u64::from(request.constraints.exchangeable));
    match &request.constraints.lowest_level {
        Some(level) => writer.word(1).reals(level),
        None => writer.word(0),
    };
    let settings = &request.settings;
    writer
        .word(settings.starts as u64)
        .word(settings.max_iterations as u64)
        .real(settings.tolerance)
        .real(settings.min_separation)
        .word(settings.seed);
    writer.finish()
}

/// A compiled exposure misclassification correction.
#[derive(Clone, Debug)]
pub struct CompiledExposureMisclassification {
    program: KernelProgram<MisclassificationRequest>,
}

/// A compiled soft-exposure effect.
#[derive(Clone, Debug)]
pub struct CompiledSoftExposureEffect {
    program: KernelProgram<SoftExposureRequest>,
}

/// A compiled detection-dependent prevalence.
#[derive(Clone, Debug)]
pub struct CompiledDetectionDependentPrevalence {
    program: KernelProgram<DetectionRequest>,
}

/// A compiled latent-class fit of repeat codings.
#[derive(Clone, Debug)]
pub struct CompiledRepeatCodings {
    program: KernelProgram<RepeatCodingsRequest>,
}

impl Study {
    /// Compile an exposure misclassification correction (the inputs of
    /// `correct_exposure_misclassification`) under the `ExposureMisclassificationCorrection`
    /// route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_exposure_misclassification(
        counts: ExposureOutcomeCounts,
        structure: ExposureErrorStructure,
        min_youden: f64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledExposureMisclassification, CausalError> {
        let request = MisclassificationRequest { counts, structure, min_youden };
        let canonical = misclassification_bytes(&request);
        let program =
            KernelProgram::compile(&EXPOSURE_MISCLASSIFICATION, request, &canonical, ctx)?;
        Ok(CompiledExposureMisclassification { program })
    }

    /// Compile a soft-exposure effect (the inputs of `correct_soft_exposure_effect`, owned)
    /// under the `SoftExposureCorrection` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_soft_exposure_effect(
        columns: SoftExposureColumns,
        method: SoftExposureMethod,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSoftExposureEffect, CausalError> {
        let request = SoftExposureRequest { columns, method };
        let canonical = soft_exposure_bytes(&request);
        let program = KernelProgram::compile(&SOFT_EXPOSURE, request, &canonical, ctx)?;
        Ok(CompiledSoftExposureEffect { program })
    }

    /// Compile a detection-dependent prevalence (the inputs of
    /// `correct_detection_dependent_prevalence`, owned) under the
    /// `DetectionDependentPrevalenceCorrection` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_detection_dependent_prevalence(
        columns: DetectionColumns,
        declaration: DetectionDeclaration,
        false_positive: FalsePositiveRate,
        min_net_detection: f64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDetectionDependentPrevalence, CausalError> {
        let request = DetectionRequest { columns, declaration, false_positive, min_net_detection };
        let canonical = detection_bytes(&request);
        let program = KernelProgram::compile(&DETECTION_PREVALENCE, request, &canonical, ctx)?;
        Ok(CompiledDetectionDependentPrevalence { program })
    }

    /// Compile a latent-class fit of repeat codings (the inputs of `fit_repeat_codings`,
    /// owned) under the `RepeatCodingsLatentClass` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_repeat_codings(
        codings: RepeatCodings,
        constraints: LatentClassConstraints,
        settings: LatentClassSettings,
        ctx: &ExecutionContext,
    ) -> Result<CompiledRepeatCodings, CausalError> {
        let request = RepeatCodingsRequest { codings, constraints, settings };
        let canonical = repeat_codings_bytes(&request);
        let program = KernelProgram::compile(&REPEAT_CODINGS, request, &canonical, ctx)?;
        Ok(CompiledRepeatCodings { program })
    }
}

impl CompiledExposureMisclassification {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then correct the exposure as
    /// `correct_exposure_misclassification` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the correction.
    pub fn correct(&self, ctx: &ExecutionContext) -> Result<CorrectedExposure, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        correct_exposure_misclassification(
            &request.counts,
            &request.structure,
            request.min_youden,
            ctx,
        )
    }
}

impl CompiledSoftExposureEffect {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then fit the outcome model as `correct_soft_exposure_effect`
    /// does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the fit.
    pub fn correct(&self, ctx: &ExecutionContext) -> Result<SoftExposureEffect, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        correct_soft_exposure_effect(&request.columns.data(), &request.method.combination(), ctx)
    }
}

impl CompiledDetectionDependentPrevalence {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then adjust the prevalence as
    /// `correct_detection_dependent_prevalence` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the adjustment.
    pub fn correct(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DetectionDependentPrevalence, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        correct_detection_dependent_prevalence(
            &request.columns.data(),
            &request.declaration,
            &request.false_positive,
            request.min_net_detection,
            ctx,
        )
    }
}

impl CompiledRepeatCodings {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then fit the latent-class model as `fit_repeat_codings` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the fit.
    pub fn fit(&self, ctx: &ExecutionContext) -> Result<LatentClassFit, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        fit_repeat_codings(&request.codings.data(), &request.constraints, &request.settings, ctx)
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_stats::measurement_error::DetectionAnchor;

    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};

    const COUNTS: ExposureOutcomeCounts =
        ExposureOutcomeCounts { exposed_cases: 3, cases: 10, exposed_controls: 2, controls: 12 };

    fn rates(sensitivity: f64) -> BinaryErrorRates {
        BinaryErrorRates {
            sensitivity,
            specificity: 0.9,
            sensitivity_variance: 0.0,
            specificity_variance: 0.0,
        }
    }

    fn misclassification(
        counts: ExposureOutcomeCounts,
        structure: ExposureErrorStructure,
        min_youden: f64,
    ) -> [u8; 32] {
        misclassification_bytes(&MisclassificationRequest { counts, structure, min_youden })
    }

    /// `misclassification_bytes`: the four counts are positional, the error structure is tagged
    /// and the by-outcome rates keep their outcome level; reals by their bits.
    #[test]
    fn site_exposure_measurement_misclassification_tags_the_structure() {
        let shared = ExposureErrorStructure::Shared(rates(0.8));
        let by_outcome = |controls: f64, cases: f64| ExposureErrorStructure::ByOutcome {
            controls: rates(controls),
            cases: rates(cases),
        };
        let swapped = ExposureOutcomeCounts {
            exposed_cases: 2,
            cases: 10,
            exposed_controls: 3,
            controls: 12,
        };
        all_distinct(&[
            misclassification(COUNTS, shared, 0.1),
            misclassification(swapped, shared, 0.1),
            misclassification(COUNTS, by_outcome(0.8, 0.8), 0.1),
            misclassification(COUNTS, by_outcome(0.8, 0.7), 0.1),
            misclassification(COUNTS, by_outcome(0.7, 0.8), 0.1),
            misclassification(COUNTS, shared, 0.2),
        ]);
        bit_exact(|x| {
            Some(misclassification(COUNTS, ExposureErrorStructure::Shared(rates(x)), 0.1))
        });
    }

    fn soft(outcome: Vec<f64>, probability: Vec<f64>, method: SoftExposureMethod) -> [u8; 32] {
        let columns = SoftExposureColumns {
            outcome,
            probability,
            covariates: vec![1.0, 2.0],
            covariate_count: 1,
        };
        soft_exposure_bytes(&SoftExposureRequest { columns, method })
    }

    /// `soft_exposure_bytes`: the columns are unit-ordered (index `i` of every column is unit
    /// `i`) and count-framed; the combination method is tagged.
    #[test]
    fn site_exposure_measurement_soft_columns_are_unit_ordered() {
        let probability = vec![0.2, 0.8, 0.5];
        ordered_laws(&[1.0, 2.0, 3.0], |outcome| {
            Some(soft(outcome.to_vec(), probability.clone(), SoftExposureMethod::PosteriorMean))
        });
        let imputation = |seed: u64| SoftExposureMethod::MultipleImputation { draw_count: 2, seed };
        let supplied = |draws: Vec<f64>| SoftExposureMethod::SuppliedDraws { draws, draw_count: 2 };
        let pair = || vec![0.2, 0.8];
        all_distinct(&[
            soft(vec![1.0, 2.0], pair(), SoftExposureMethod::PosteriorMean),
            soft(vec![1.0, 2.0], pair(), imputation(1)),
            soft(vec![1.0, 2.0], pair(), imputation(2)),
            soft(vec![1.0, 2.0], pair(), supplied(vec![0.0, 1.0, 1.0, 0.0])),
            soft(vec![1.0, 2.0], pair(), supplied(vec![1.0, 0.0, 0.0, 1.0])),
            // A value moved across the outcome / probability boundary.
            soft(vec![1.0], vec![2.0, 0.2, 0.8], SoftExposureMethod::PosteriorMean),
        ]);
        bit_exact(|x| Some(soft(vec![x, 2.0], pair(), SoftExposureMethod::PosteriorMean)));
    }

    fn detection(detected: Vec<bool>, declaration: DetectionDeclaration) -> [u8; 32] {
        detection_bytes(&DetectionRequest {
            columns: DetectionColumns { detected, covariate: vec![0.0, 1.0] },
            declaration,
            false_positive: FalsePositiveRate::NONE,
            min_net_detection: 0.1,
        })
    }

    fn anchor(covariate: f64) -> DetectionAnchor {
        DetectionAnchor { covariate, lower: 0.2, upper: 0.9 }
    }

    fn envelope(
        direction: DetectionDirection,
        anchors: &[DetectionAnchor],
    ) -> DetectionDeclaration {
        DetectionDeclaration::Partial { direction, anchors: anchors.to_vec() }
    }

    fn logistic(intercept: f64) -> DetectionDeclaration {
        DetectionDeclaration::Full(DetectionFunction::Logistic {
            intercept,
            slope: 1.0,
            intercept_variance: 0.0,
            covariance: 0.0,
            slope_variance: 0.0,
        })
    }

    /// `detection_bytes`: the anchors are ordered (the kernel refuses any order but increasing
    /// covariate), the declaration form and direction are tagged, the detections are unit-ordered.
    #[test]
    fn site_exposure_measurement_detection_tags_the_declaration() {
        let up = DetectionDirection::Nondecreasing;
        ordered_laws(&[anchor(0.0), anchor(1.0), anchor(2.0)], |anchors| {
            Some(detection(vec![true, false], envelope(up, anchors)))
        });
        let table = |knots: Vec<f64>, values: Vec<f64>| {
            DetectionDeclaration::Full(DetectionFunction::Table { knots, values })
        };
        all_distinct(&[
            detection(vec![true, false], logistic(0.5)),
            detection(vec![false, true], logistic(0.5)),
            detection(vec![true, false], table(vec![0.0, 1.0], vec![0.5, 0.9])),
            detection(vec![true, false], table(vec![0.0], vec![1.0, 0.5, 0.9])),
            detection(vec![true, false], envelope(up, &[anchor(0.0)])),
            detection(
                vec![true, false],
                envelope(DetectionDirection::Nonincreasing, &[anchor(0.0)]),
            ),
        ]);
        bit_exact(|x| Some(detection(vec![true, false], logistic(x))));
    }

    fn codings(codings: Vec<u8>, lowest_level: Option<Vec<f64>>) -> [u8; 32] {
        repeat_codings_bytes(&RepeatCodingsRequest {
            codings: RepeatCodings {
                codings,
                units: 2,
                coding_count: 2,
                categories: 2,
                latent_levels: 2,
            },
            constraints: LatentClassConstraints { exchangeable: true, lowest_level },
            settings: LatentClassSettings::default(),
        })
    }

    /// `repeat_codings_bytes`: the codings are unit-major and ordered; the declared lowest-level
    /// law carries a presence word.
    #[test]
    fn site_exposure_measurement_repeat_codings_are_ordered() {
        ordered_laws(&[0_u8, 1, 1, 0], |items| Some(codings(items.to_vec(), None)));
        all_distinct(&[
            codings(vec![0, 1, 1, 0], None),
            codings(vec![0, 1, 1, 0], Some(vec![])),
            codings(vec![0, 1, 1, 0], Some(vec![0.9, 0.1])),
        ]);
        bit_exact(|x| Some(codings(vec![0, 1, 1, 0], Some(vec![x, 0.1]))));
    }
}
