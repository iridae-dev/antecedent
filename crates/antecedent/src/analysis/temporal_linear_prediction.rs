//! Compiled conditional prediction from a lagged linear regression of one series.
//!
//! One program family, `TemporalLinearPrediction`, bound to the support unit of its capability
//! record (`estimate.temporal_linear_prediction`). A request regresses a target on caller-listed
//! lagged columns (`target(t) ~ 1 + parents`, ordinary least squares on the lag-aligned rows)
//! and, for a prediction, evaluates the fitted line with every lag of one listed variable held
//! at a level and every other column at its observed value.
//!
//! The answer is the conditional expectation `E[target_t | listed lags, variable = level]` under
//! the linear fit. It is an association of the observed series: no graph is read, nothing is
//! identified, no adjustment is chosen, and it is not `do(variable = level)`. The program claims
//! no identification and its claim is associational; it claims no uncertainty.
//!
//! The request lowers to a six-layer program whose identity is the digest of the question (the
//! request kind, the series content and time index, and every input the kernel reads). The
//! retained handle enters its own dispatch, passes the support gate of its unit and calls
//! `antecedent_estimate::TemporalLinearPredictor::fit` and `predict_conditional` with the same
//! inputs as the direct call; the kernel results are returned unchanged.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    CanonicalWriter, ExecutionContext, KernelPolicy, NotSupported, PhysicalEffect,
    PhysicalOperator, VariableId,
};
use antecedent_data::{LaggedColumn, TimeSeriesData};
use antecedent_estimate::{EstimationError, TemporalLinearPredictor};

use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

/// One lagged linear prediction question, with every input the kernel reads.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TemporalPredictionRequest {
    /// Fit `target(t) ~ 1 + parents` on the lag-aligned rows
    /// (`TemporalLinearPredictor::fit`).
    Fit {
        /// The series.
        data: TimeSeriesData,
        /// Target column.
        target: VariableId,
        /// Lagged predictor columns, in coefficient order.
        parents: Arc<[LaggedColumn]>,
        /// Kernel policy of the least-squares solve.
        policy: KernelPolicy,
    },
    /// Fit as [`Self::Fit`], then evaluate the fitted line on the same rows with every lag of
    /// `held` set to `level` (`TemporalLinearPredictor::predict_conditional`).
    PredictConditional {
        /// The series.
        data: TimeSeriesData,
        /// Target column.
        target: VariableId,
        /// Lagged predictor columns, in coefficient order.
        parents: Arc<[LaggedColumn]>,
        /// The variable whose lagged columns are held.
        held: VariableId,
        /// The level they are held at.
        level: f64,
        /// Kernel policy of the solve and the row preparation.
        policy: KernelPolicy,
    },
}

impl TemporalPredictionRequest {
    /// A fit of `target` on `parents`.
    #[must_use]
    pub fn fit(
        data: TimeSeriesData,
        target: VariableId,
        parents: impl Into<Arc<[LaggedColumn]>>,
        policy: KernelPolicy,
    ) -> Self {
        Self::Fit { data, target, parents: parents.into(), policy }
    }

    /// A conditional prediction of `target` from `parents` with `held` at `level`.
    #[must_use]
    pub fn predict_conditional(
        data: TimeSeriesData,
        target: VariableId,
        parents: impl Into<Arc<[LaggedColumn]>>,
        held: VariableId,
        level: f64,
        policy: KernelPolicy,
    ) -> Self {
        Self::PredictConditional { data, target, parents: parents.into(), held, level, policy }
    }
}

/// What a compiled prediction returns: the kernel results, unchanged.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TemporalPredictionOutcome {
    /// The fitted predictor.
    Fit(TemporalLinearPredictor),
    /// The fitted predictor and its conditional prediction on each lag-aligned row.
    PredictConditional {
        /// The fitted predictor.
        predictor: TemporalLinearPredictor,
        /// One prediction per lag-aligned row, in row order.
        predictions: Arc<[f64]>,
    },
}

impl TemporalPredictionOutcome {
    /// The fitted predictor.
    #[must_use]
    pub const fn predictor(&self) -> &TemporalLinearPredictor {
        match self {
            Self::Fit(predictor) | Self::PredictConditional { predictor, .. } => predictor,
        }
    }

    /// The conditional predictions, when the request asked for them.
    #[must_use]
    pub fn predictions(&self) -> Option<&Arc<[f64]>> {
        match self {
            Self::Fit(_) => None,
            Self::PredictConditional { predictions, .. } => Some(predictions),
        }
    }
}

/// Why a compiled prediction did not run.
#[derive(Debug)]
#[non_exhaustive]
pub enum TemporalPredictionError {
    /// The support gate of the route's unit refused (release build).
    NotSupported(NotSupported),
    /// The compiled program is inside another program's lowered dispatch.
    Reentry,
    /// The execution context was cancelled before the fit or before the prediction; nothing is
    /// returned.
    Cancelled,
    /// The kernel refused or failed, with its own typed error.
    Estimation(EstimationError),
}

/// Refusal detail of [`TemporalPredictionError::Reentry`] (reason code
/// `public_builder_reentry`).
pub const REENTRY_DETAIL: &str =
    antecedent_core::reason_details::temporal_linear_prediction::REENTRY;

impl fmt::Display for TemporalPredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported(refusal) => write!(f, "{refusal}"),
            Self::Reentry => write!(
                f,
                "public_builder_reentry ({REENTRY_DETAIL}): a lowered dispatch executes its \
                 retained program; it does not compile or run another prediction"
            ),
            Self::Cancelled => write!(f, "cancelled during {CANCELLED_STAGE}"),
            Self::Estimation(error) => write!(f, "{error}"),
        }
    }
}

/// Stage id a cancelled prediction reports.
const CANCELLED_STAGE: &str = "model";

impl std::error::Error for TemporalPredictionError {}

impl From<EstimationError> for TemporalPredictionError {
    fn from(error: EstimationError) -> Self {
        Self::Estimation(error)
    }
}

impl From<TemporalPredictionError> for CausalError {
    fn from(error: TemporalPredictionError) -> Self {
        match error {
            TemporalPredictionError::NotSupported(refusal) => Self::from(refusal),
            TemporalPredictionError::Reentry => crate::unsupported_reason!(
                "public_builder_reentry",
                detail = REENTRY_DETAIL,
                "a lowered dispatch executes its retained program; it does not compile or run \
                 another prediction"
            ),
            TemporalPredictionError::Cancelled => Self::Cancelled { stage: CANCELLED_STAGE },
            TemporalPredictionError::Estimation(error) => Self::from(error),
        }
    }
}

/// A compiled prediction: the request and the lowered program it retains.
#[derive(Clone, Debug)]
pub struct CompiledTemporalPrediction {
    request: TemporalPredictionRequest,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledTemporalPrediction {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Compile one prediction question into its retained program.
///
/// Compiling checks nothing the kernel does not check itself; a malformed request is refused by
/// the kernel when the program runs, with its own typed error.
///
/// # Errors
/// [`TemporalPredictionError::Reentry`] inside another program's lowered dispatch.
pub fn compile_temporal_prediction(
    request: TemporalPredictionRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledTemporalPrediction, TemporalPredictionError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(TemporalPredictionError::Reentry);
    }
    let lowered = Arc::new(lower(&request));
    Ok(CompiledTemporalPrediction { request, lowered })
}

impl CompiledTemporalPrediction {
    /// The request the program was compiled from.
    #[must_use]
    pub const fn request(&self) -> &TemporalPredictionRequest {
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

    /// Run the kernel fit (and prediction) of the request inside the program's dispatch,
    /// after the support gate of the route's unit.
    ///
    /// # Errors
    /// The gate's refusal in a release build, reentry, or the kernel's own error.
    pub fn run(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TemporalPredictionOutcome, TemporalPredictionError> {
        if ctx.lowered_dispatch().is_some() {
            return Err(TemporalPredictionError::Reentry);
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(TemporalPredictionError::NotSupported)?;
        }
        run(&self.request, &scoped)
    }
}

/// Whether `run` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). The unit's record is new: the gate stays off until its
/// scientific assertions are earned, so no valid use refuses before then. Turning it on makes a
/// release build refuse the route while its unit is unsupported, and a development build record
/// what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// Refuse a cancelled context at a step boundary.
fn check_cancelled(ctx: &ExecutionContext) -> Result<(), TemporalPredictionError> {
    if ctx.cancellation.is_cancelled() {
        return Err(TemporalPredictionError::Cancelled);
    }
    Ok(())
}

/// The kernel calls of each request: the same functions, with the same inputs, as the direct
/// path. The kernel reads its kernel policy, never the execution context; the context is
/// polled for cancellation before the fit and before the prediction, so a cancelled run
/// returns nothing and an uncancelled one returns the kernel results unchanged.
fn run(
    request: &TemporalPredictionRequest,
    ctx: &ExecutionContext,
) -> Result<TemporalPredictionOutcome, TemporalPredictionError> {
    use TemporalPredictionRequest as R;
    check_cancelled(ctx)?;
    Ok(match request {
        R::Fit { data, target, parents, policy } => TemporalPredictionOutcome::Fit(
            TemporalLinearPredictor::fit(data, *target, Arc::clone(parents), policy)?,
        ),
        R::PredictConditional { data, target, parents, held, level, policy } => {
            let predictor =
                TemporalLinearPredictor::fit(data, *target, Arc::clone(parents), policy)?;
            check_cancelled(ctx)?;
            let predictions = predictor.predict_conditional(data, *held, *level, policy)?;
            TemporalPredictionOutcome::PredictConditional { predictor, predictions }
        }
    })
}

// --- lowering ------------------------------------------------------------------------------

/// The canonical bytes of a series: its storage content digest and its time index.
fn series_digest(data: &TimeSeriesData) -> [u8; 32] {
    let mut w = CanonicalWriter::new("antecedent.temporal_linear_prediction.series.v1");
    w.bytes(&data.storage().content_digest());
    w.text(&format!("{:?}", data.time_index().regularity));
    w.word(data.time_index().length as u64);
    w.finish()
}

fn encode_design(
    w: &mut CanonicalWriter,
    data: &TimeSeriesData,
    target: VariableId,
    parents: &[LaggedColumn],
    policy: &KernelPolicy,
) {
    w.bytes(&series_digest(data));
    w.word(u64::from(target.raw()));
    w.word(parents.len() as u64);
    for parent in parents {
        w.word(u64::from(parent.variable.raw())).word(u64::from(parent.lag.raw()));
    }
    // The policy selects the kernel arithmetic (scalar or vectorized), so it is part of the
    // question; it is plain data whose `Debug` is deterministic.
    w.text(&format!("{policy:?}"));
}

/// The digest of the question: the request kind and every input the kernel reads.
fn question(request: &TemporalPredictionRequest) -> [u8; 32] {
    use TemporalPredictionRequest as R;
    let mut w = CanonicalWriter::new("antecedent.temporal_linear_prediction.question.v1");
    match request {
        R::Fit { data, target, parents, policy } => {
            w.text("fit");
            encode_design(&mut w, data, *target, parents, policy);
        }
        R::PredictConditional { data, target, parents, held, level, policy } => {
            w.text("predict_conditional");
            encode_design(&mut w, data, *target, parents, policy);
            w.word(u64::from(held.raw())).real(*level);
        }
    }
    w.finish()
}

/// The six layers of a prediction program. The question digest is the problem's identity; the
/// regression is fitted on each execution; no identification and no uncertainty are claimed.
fn lower(request: &TemporalPredictionRequest) -> LoweredProgram {
    use TemporalPredictionRequest as R;
    let query = question(request);
    let cell = ProgramCell::TemporalLinearPrediction;
    let (data, target, parents) = match request {
        R::Fit { data, target, parents, .. }
        | R::PredictConditional { data, target, parents, .. } => (data, *target, parents),
    };
    let source = series_digest(data);
    let query_variables: Arc<[VariableId]> =
        std::iter::once(target).chain(parents.iter().map(|parent| parent.variable)).collect();
    let design = DesignShape::not_design_based(
        "a lagged least-squares fit sized by its listed columns; no quoted adjustment design",
    );
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Association,
            outcome: Some(target),
            query,
        },
        plan_id: Arc::from("temporal_linear_prediction.conditional.v1"),
        data: DataClassification::Temporal,
        graph_class: "none",
        query_variables,
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(cell.as_str()) },
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
            digest: Some(source),
        }]),
        composition: Composition::Chained { stages: 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let operators: Arc<[PhysicalOperator]> = [PhysicalEffect::Fit, PhysicalEffect::Condition]
        .into_iter()
        .filter_map(|effect| PhysicalOperator::declare(effect).build().ok())
        .collect();
    let physical = PhysicalLayer { operators, design };
    StandaloneCompositionLoweredParts {
        cell,
        scientific,
        identification: IdentificationLayer::NotClaimed { refusal: None },
        statistical,
        evidence,
        physical,
        claim: ClaimDemand {
            uncertainty: UncertaintyReading::None,
            kind: Some(ClaimKind::Associational),
        },
    }
    .finish()
}

#[cfg(test)]
mod tests {
    use antecedent_core::Lag;

    use super::*;

    fn series(shift: f64) -> TimeSeriesData {
        let x: Vec<f64> = (0..12).map(|t| f64::from(t % 5)).collect();
        let y: Vec<f64> = (0..12).map(|t| shift + f64::from(t % 3)).collect();
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
    }

    fn request(shift: f64, level: f64) -> TemporalPredictionRequest {
        TemporalPredictionRequest::predict_conditional(
            series(shift),
            VariableId::from_raw(1),
            [LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(1) }],
            VariableId::from_raw(0),
            level,
            KernelPolicy::default_policy(),
        )
    }

    #[test]
    fn the_program_identity_is_the_question_and_survives_recompilation() {
        let ctx = ExecutionContext::for_tests(0);
        let a = compile_temporal_prediction(request(0.0, 1.0), &ctx).unwrap();
        let b = compile_temporal_prediction(request(0.0, 1.0), &ctx).unwrap();
        assert_eq!(a.lowered().identity(), b.lowered().identity());
        assert_eq!(a.lowered().cell(), ProgramCell::TemporalLinearPrediction);
        let level = compile_temporal_prediction(request(0.0, 2.0), &ctx).unwrap();
        let rows = compile_temporal_prediction(request(1.0, 1.0), &ctx).unwrap();
        assert_ne!(a.lowered().identity(), level.lowered().identity());
        assert_ne!(a.lowered().identity(), rows.lowered().identity());
    }

    #[test]
    fn a_compiled_prediction_refuses_reentry() {
        let ctx = ExecutionContext::for_tests(0);
        let compiled = compile_temporal_prediction(request(0.0, 1.0), &ctx).unwrap();
        let inside = ctx.enter_lowered_dispatch(compiled.lowered().identity());
        assert!(matches!(compiled.run(&inside), Err(TemporalPredictionError::Reentry)));
        assert!(matches!(
            compile_temporal_prediction(request(0.0, 1.0), &inside),
            Err(TemporalPredictionError::Reentry)
        ));
    }
}
