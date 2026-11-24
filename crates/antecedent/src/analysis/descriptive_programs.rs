//! Compiled descriptive comparisons and the matched case-control conditional odds ratio.
//!
//! Two program families, each bound to the support unit of its capability record:
//!
//! - `DescriptiveComparison` (`estimate.descriptive_comparison`): the unadjusted contrast of two
//!   treatment arms with its Welch standard error, that contrast set beside an adjusted estimate
//!   of the same estimand coding, and a mean-outcome pair put on declared reporting scales by
//!   the delta method (a pair the caller holds, or the raw arm means of a sample);
//! - `MatchedCaseControl` (`estimate.matched_case_control_odds_ratio`): the conditional maximum
//!   likelihood odds ratio of a binary exposure among matched sets sampled on the outcome.
//!
//! A request lowers to a six-layer program whose identity is the digest of the question (the
//! request kind and every input the kernel reads, rows included). The retained handle enters its
//! own dispatch, passes the support gate of its unit and calls the same kernel function with the
//! same inputs as the direct call; the kernel result is returned unchanged. Neither family
//! claims identification: each describes the observed law (the matched odds ratio is a
//! conditional association parameter among the declared sets), and the claim is associational.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    CanonicalWriter, ExecutionContext, NotSupported, PhysicalEffect, PhysicalOperator,
    ScoreFrequentist, ScoreModel,
};
use antecedent_estimate::{
    AdjustedEstimate, Availability, ConditionalOddsRatio, DescriptiveComparison, EstimationError,
    MeanPair, RawContrast, ReportingScale, ReportingTransform,
};

use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

/// One descriptive or matched case-control question, with every input its kernel reads.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DescriptiveRequest {
    /// The unadjusted contrast `mean(Y | T = 1) - mean(Y | T = 0)` with its Welch standard
    /// error (`antecedent_estimate::raw_contrast`).
    RawContrast {
        /// Outcome rows.
        outcome: Vec<f64>,
        /// Treatment rows, coded exactly 0 or 1.
        treatment: Vec<f64>,
    },
    /// The raw contrast beside an adjusted estimate of the same estimand coding, and their
    /// descriptive gap (`antecedent_estimate::compare_raw_adjusted`).
    CompareRawAdjusted {
        /// Outcome rows.
        outcome: Vec<f64>,
        /// Treatment rows, coded exactly 0 or 1.
        treatment: Vec<f64>,
        /// The adjusted estimate and its declared coding.
        adjusted: AdjustedEstimate,
    },
    /// A mean-outcome pair on declared reporting scales, with the delta-method covariance when
    /// the pair carries its joint covariance (`antecedent_estimate::transform_mean_pair`).
    TransformMeanPair {
        /// The pair and the covariance its source published.
        pair: MeanPair,
        /// The declared scales, in the order reported.
        scales: Vec<ReportingScale>,
    },
    /// The raw arm means of a sample on declared reporting scales, with their independent-groups
    /// covariance: `antecedent_estimate::raw_contrast`, then
    /// `antecedent_estimate::transform_mean_pair` of `RawContrast::mean_pair`.
    RawArmsTransform {
        /// Outcome rows.
        outcome: Vec<f64>,
        /// Treatment rows, coded exactly 0 or 1.
        treatment: Vec<f64>,
        /// The declared scales, in the order reported.
        scales: Vec<ReportingScale>,
    },
    /// The conditional odds ratio of matched case-control sets
    /// (`antecedent_estimate::conditional_odds_ratio`). The declared sampling design and
    /// estimand are checked by the caller before the request is built.
    MatchedOddsRatio {
        /// Matched-set label of each row.
        stratum: Vec<String>,
        /// Case indicator of each row, exactly 0 or 1.
        case: Vec<f64>,
        /// Exposure indicator of each row, exactly 0 or 1.
        exposed: Vec<f64>,
    },
}

/// The kernel result of a request, unchanged.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DescriptiveOutcome {
    /// [`DescriptiveRequest::RawContrast`].
    RawContrast(RawContrast),
    /// [`DescriptiveRequest::CompareRawAdjusted`].
    Comparison(DescriptiveComparison),
    /// [`DescriptiveRequest::TransformMeanPair`].
    Transform(ReportingTransform),
    /// [`DescriptiveRequest::RawArmsTransform`]: the raw contrast and the transform of its arm
    /// means.
    RawArmsTransform(RawContrast, ReportingTransform),
    /// [`DescriptiveRequest::MatchedOddsRatio`].
    MatchedOddsRatio(ConditionalOddsRatio),
}

/// Why a compiled descriptive program did not run or did not return a result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DescriptiveProgramError {
    /// The support gate of the route's unit refused (release build).
    NotSupported(NotSupported),
    /// The compiled program is inside another program's lowered dispatch.
    Reentry,
    /// The kernel refused, failed or was cancelled; its error is unchanged.
    Estimation(EstimationError),
}

/// Refusal detail of [`DescriptiveProgramError::Reentry`] (reason code
/// `public_builder_reentry`).
pub const REENTRY_DETAIL: &str = "descriptive_programs.reentry";

impl fmt::Display for DescriptiveProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported(refusal) => write!(f, "{refusal}"),
            Self::Reentry => write!(
                f,
                "public_builder_reentry ({REENTRY_DETAIL}): a lowered dispatch executes its \
                 retained program; it does not compile or run another descriptive program"
            ),
            Self::Estimation(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DescriptiveProgramError {}

impl From<EstimationError> for DescriptiveProgramError {
    fn from(error: EstimationError) -> Self {
        Self::Estimation(error)
    }
}

/// A compiled descriptive program: the request and the lowered program it retains.
#[derive(Clone, Debug)]
pub struct CompiledDescriptive {
    request: DescriptiveRequest,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledDescriptive {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Compile one descriptive or matched case-control question into its retained program.
///
/// Compiling checks nothing the kernel does not check itself; malformed rows are refused by the
/// kernel when the program runs, with its own typed error.
///
/// # Errors
/// [`DescriptiveProgramError::Reentry`] inside another program's lowered dispatch.
pub fn compile_descriptive(
    request: DescriptiveRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledDescriptive, DescriptiveProgramError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(DescriptiveProgramError::Reentry);
    }
    let lowered = Arc::new(lower(&request));
    Ok(CompiledDescriptive { request, lowered })
}

impl CompiledDescriptive {
    /// The request the program was compiled from.
    #[must_use]
    pub const fn request(&self) -> &DescriptiveRequest {
        &self.request
    }

    /// Every layer of the retained program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the retained program, as compiled.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Run the kernel of the request inside the program's dispatch, after the support gate of
    /// the route's unit.
    ///
    /// # Errors
    /// The gate's refusal in a release build, reentry, or the kernel's own error.
    pub fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DescriptiveOutcome, DescriptiveProgramError> {
        if ctx.lowered_dispatch().is_some() {
            return Err(DescriptiveProgramError::Reentry);
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        gate_program(&scoped, self).map_err(DescriptiveProgramError::NotSupported)?;
        run(&self.request, &scoped)
    }
}

/// The program family a request lowers to; each is bound to one support unit.
const fn cell_of(request: &DescriptiveRequest) -> ProgramCell {
    match request {
        DescriptiveRequest::RawContrast { .. }
        | DescriptiveRequest::CompareRawAdjusted { .. }
        | DescriptiveRequest::TransformMeanPair { .. }
        | DescriptiveRequest::RawArmsTransform { .. } => ProgramCell::DescriptiveComparison,
        DescriptiveRequest::MatchedOddsRatio { .. } => ProgramCell::MatchedCaseControl,
    }
}

/// The kernel call of each request: the same function, with the same inputs, as the direct
/// path.
fn run(
    request: &DescriptiveRequest,
    ctx: &ExecutionContext,
) -> Result<DescriptiveOutcome, DescriptiveProgramError> {
    use DescriptiveOutcome as O;
    use DescriptiveRequest as R;
    Ok(match request {
        R::RawContrast { outcome, treatment } => {
            O::RawContrast(antecedent_estimate::raw_contrast(outcome, treatment, ctx)?)
        }
        R::CompareRawAdjusted { outcome, treatment, adjusted } => O::Comparison(
            antecedent_estimate::compare_raw_adjusted(outcome, treatment, *adjusted, ctx)?,
        ),
        R::TransformMeanPair { pair, scales } => {
            O::Transform(antecedent_estimate::transform_mean_pair(pair, scales)?)
        }
        R::RawArmsTransform { outcome, treatment, scales } => {
            let raw = antecedent_estimate::raw_contrast(outcome, treatment, ctx)?;
            let transform = antecedent_estimate::transform_mean_pair(&raw.mean_pair(), scales)?;
            O::RawArmsTransform(raw, transform)
        }
        R::MatchedOddsRatio { stratum, case, exposed } => O::MatchedOddsRatio(
            antecedent_estimate::conditional_odds_ratio(stratum, case, exposed, ctx)?,
        ),
    })
}

// --- lowering ------------------------------------------------------------------------------

fn encode_scales(writer: &mut CanonicalWriter, scales: &[ReportingScale]) {
    writer.word(scales.len() as u64);
    for scale in scales {
        writer.text(scale.name());
    }
}

fn encode_pair(writer: &mut CanonicalWriter, pair: &MeanPair) {
    writer.real(pair.active).real(pair.control);
    match &pair.covariance {
        Availability::Available(values) => {
            writer.word(1).reals(values);
        }
        Availability::Unavailable(reason) => {
            writer.word(0).text(reason.detail);
        }
    }
}

fn encode_adjusted(writer: &mut CanonicalWriter, adjusted: &AdjustedEstimate) {
    writer
        .real(adjusted.estimate)
        .opt_real(adjusted.se)
        .real(adjusted.active)
        .real(adjusted.control)
        .text(adjusted.scale.name())
        .word(u64::from(adjusted.all_observed_population));
}

/// The digest of the question: the request kind and every input the kernel reads.
fn question(request: &DescriptiveRequest) -> [u8; 32] {
    use DescriptiveRequest as R;
    let mut w = CanonicalWriter::new("antecedent.descriptive_programs.question.v1");
    match request {
        R::RawContrast { outcome, treatment } => {
            w.text("raw_contrast").reals(outcome).reals(treatment);
        }
        R::CompareRawAdjusted { outcome, treatment, adjusted } => {
            w.text("compare_raw_adjusted").reals(outcome).reals(treatment);
            encode_adjusted(&mut w, adjusted);
        }
        R::TransformMeanPair { pair, scales } => {
            w.text("transform_mean_pair");
            encode_pair(&mut w, pair);
            encode_scales(&mut w, scales);
        }
        R::RawArmsTransform { outcome, treatment, scales } => {
            w.text("raw_arms_transform").reals(outcome).reals(treatment);
            encode_scales(&mut w, scales);
        }
        R::MatchedOddsRatio { stratum, case, exposed } => {
            w.text("matched_odds_ratio").word(stratum.len() as u64);
            for label in stratum {
                w.text(label);
            }
            w.reals(case).reals(exposed);
        }
    }
    w.finish()
}

/// Plan id and the frequentist score model of the request.
const fn shape(request: &DescriptiveRequest) -> (&'static str, &'static str) {
    match request {
        DescriptiveRequest::RawContrast { .. } => {
            ("descriptive_comparison.raw_contrast.v1", "descriptive_comparison.welch")
        }
        DescriptiveRequest::CompareRawAdjusted { .. } => {
            ("descriptive_comparison.raw_vs_adjusted.v1", "descriptive_comparison.welch")
        }
        DescriptiveRequest::TransformMeanPair { .. } => {
            ("descriptive_comparison.reporting_transform.v1", "descriptive_comparison.delta_method")
        }
        DescriptiveRequest::RawArmsTransform { .. } => (
            "descriptive_comparison.raw_reporting_transform.v1",
            "descriptive_comparison.delta_method",
        ),
        DescriptiveRequest::MatchedOddsRatio { .. } => (
            "matched_case_control.conditional_odds_ratio.v1",
            "matched_case_control.conditional_likelihood",
        ),
    }
}

/// The six layers of a descriptive program. The question digest is the problem's identity; no
/// identification is claimed; the kernel's sampling reading (a Welch or delta-method standard
/// error, a Wald interval of the conditional likelihood) is the uncertainty, and the claim is
/// associational.
fn lower(request: &DescriptiveRequest) -> LoweredProgram {
    let query = question(request);
    let (plan_id, score) = shape(request);
    let cell = cell_of(request);
    let design = DesignShape::not_design_based(
        "a descriptive contrast, a delta-method transform or a conditional likelihood of matched \
         sets fits no quoted adjustment design",
    );
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Association,
            outcome: None,
            query,
        },
        plan_id: Arc::from(plan_id),
        data: DataClassification::Tabular,
        graph_class: "none",
        query_variables: Arc::from([]),
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let realization = ScoreModel::new(score).map_or_else(
        |_| Realization::ExactPoint { evaluator: Arc::from(cell.as_str()) },
        |model| Realization::Frequentist(ScoreFrequentist::new(model)),
    );
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(query),
        }]),
        composition: Composition::Chained { stages: 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let operators: Arc<[PhysicalOperator]> = PhysicalOperator::declare(PhysicalEffect::Fit)
        .build()
        .map_or_else(|_| Arc::from([]), |op| Arc::from([op]));
    let physical = PhysicalLayer { operators, design };
    StandaloneCompositionLoweredParts {
        cell,
        scientific,
        identification: IdentificationLayer::NotClaimed { refusal: None },
        statistical,
        evidence,
        physical,
        claim: ClaimDemand {
            uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Associational),
        },
    }
    .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw() -> DescriptiveRequest {
        DescriptiveRequest::RawContrast {
            outcome: vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            treatment: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn the_program_identity_is_the_question_and_survives_recompilation() {
        let ctx = ExecutionContext::for_tests(0);
        let a = compile_descriptive(raw(), &ctx).unwrap();
        let b = compile_descriptive(raw(), &ctx).unwrap();
        assert_eq!(a.lowered().identity(), b.lowered().identity());
        assert_eq!(a.lowered().cell(), ProgramCell::DescriptiveComparison);
        let other = DescriptiveRequest::RawContrast {
            outcome: vec![1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            treatment: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        };
        let c = compile_descriptive(other, &ctx).unwrap();
        assert_ne!(a.lowered().identity(), c.lowered().identity());
    }

    #[test]
    fn the_matched_odds_ratio_lowers_to_its_own_family() {
        let ctx = ExecutionContext::for_tests(0);
        let request = DescriptiveRequest::MatchedOddsRatio {
            stratum: vec!["a".into(), "a".into()],
            case: vec![1.0, 0.0],
            exposed: vec![1.0, 0.0],
        };
        let compiled = compile_descriptive(request, &ctx).unwrap();
        assert_eq!(compiled.lowered().cell(), ProgramCell::MatchedCaseControl);
    }

    #[test]
    fn a_compiled_descriptive_program_refuses_reentry() {
        let ctx = ExecutionContext::for_tests(0);
        let compiled = compile_descriptive(raw(), &ctx).unwrap();
        let inside = ctx.enter_lowered_dispatch(compiled.lowered().identity());
        assert!(matches!(compiled.evaluate(&inside), Err(DescriptiveProgramError::Reentry)));
        assert!(matches!(
            compile_descriptive(raw(), &inside),
            Err(DescriptiveProgramError::Reentry)
        ));
    }
}
