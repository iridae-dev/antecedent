//! Compile and execute the native linear conditional-effect fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{borrow::Cow, sync::Arc};

use antecedent_contract::{ClaimKind, ScientificQuantity};
use antecedent_core::{
    CanonicalWriter, ExecutionContext, SemanticDigest, execution::AdmissionError,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::cohort_linear_effect::{
    FIT_PROCEDURE_ID, MAX_FIT_FEATURES, MAX_FIT_ROWS, NativeCohortFit,
};
use antecedent_io::artifact_codec::CanonicalSize;
use antecedent_io::quantity_wire::ScientificQuantityWire;

use super::compiled_measured_interval::{MeasuredIntervalProgram, frequentist};
use super::execute::Study;
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const PLAN_ID: &str = "native_cohort_fit.v1";

/// A native OLS/HC3 model produced by the bound program.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CohortModelFitRun {
    /// The fitted model, including its compact export and the retained training rows.
    pub fit: NativeCohortFit,
    /// The exact lowered program that produced this fit.
    pub lowered_program_identity: SemanticDigest,
}

/// The ordered training columns and quantity declarations bound to a native fit program.
#[derive(Clone, Debug)]
pub struct CompiledNativeCohortFit<'a> {
    input: FitInput<'a>,
    working_bytes: u64,
    lowered: Arc<LoweredProgram>,
}

#[derive(Clone, Debug)]
struct FitInput<'a> {
    outcome: Cow<'a, [f64]>,
    treatment: Cow<'a, [f64]>,
    features: Vec<Cow<'a, [f64]>>,
    response: Cow<'a, ScientificQuantityWire>,
    feature_quantities: Cow<'a, [ScientificQuantityWire]>,
}

pub(super) fn working_bytes(
    outcome: usize,
    treatment: usize,
    features: impl ExactSizeIterator<Item = usize>,
) -> Result<u64, AdmissionError> {
    if outcome > MAX_FIT_ROWS || treatment > MAX_FIT_ROWS || features.len() > MAX_FIT_FEATURES {
        return Err(AdmissionError::Overflow);
    }
    let mut values = outcome + treatment;
    for length in features {
        if length > MAX_FIT_ROWS {
            return Err(AdmissionError::Overflow);
        }
        values += length;
    }
    // Covers retained columns, canonical hashing and the bounded dense OLS/HC3 scratch.
    Ok((values as u64) * 32 + 512 * 1024)
}

/// Both model routes use the same fit operator and do not request a sampling interval.
pub(crate) fn lower_fit(
    cell: ProgramCell,
    plan_id: &'static str,
    question: [u8; 32],
    source: [u8; 32],
    rows: usize,
    features: usize,
) -> Result<Arc<LoweredProgram>, CausalError> {
    MeasuredIntervalProgram {
        cell,
        plan_id,
        kind: FunctionalKind::AssignmentContrast,
        graph_class: "linear_conditional_effect_under_declared_basis",
        question,
        source,
        source_role: EvidenceRole::Observations,
        realization: frequentist(FIT_PROCEDURE_ID, plan_id)?,
        uncertainty: UncertaintyReading::None,
        rows: rows as u64,
        columns: 2 + features as u64,
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind the ordered training columns and quantity declarations to a native fit plan.
    ///
    /// # Errors
    /// An oversized table, mismatched columns or quantity declarations, or an invalid quantity.
    pub fn compile_native_cohort_fit<'a>(
        outcome: &'a [f64],
        treatment: &'a [f64],
        features: &'a [&'a [f64]],
        response: &'a ScientificQuantityWire,
        feature_quantities: &'a [ScientificQuantityWire],
    ) -> Result<CompiledNativeCohortFit<'a>, EstimationError> {
        if features.len() > MAX_FIT_FEATURES {
            return Err(AdmissionError::Overflow.into());
        }
        FitInput {
            outcome: Cow::Borrowed(outcome),
            treatment: Cow::Borrowed(treatment),
            features: features.iter().map(|&column| Cow::Borrowed(column)).collect(),
            response: Cow::Borrowed(response),
            feature_quantities: Cow::Borrowed(feature_quantities),
        }
        .compile()
    }

    /// Move owned training columns into the same inspectable fit plan.
    ///
    /// # Errors
    /// The same input errors as [`Study::compile_native_cohort_fit`].
    pub fn compile_owned_native_cohort_fit(
        outcome: Vec<f64>,
        treatment: Vec<f64>,
        features: Vec<Vec<f64>>,
        response: ScientificQuantityWire,
        feature_quantities: Vec<ScientificQuantityWire>,
    ) -> Result<CompiledNativeCohortFit<'static>, EstimationError> {
        if features.len() > MAX_FIT_FEATURES {
            return Err(AdmissionError::Overflow.into());
        }
        FitInput {
            outcome: Cow::Owned(outcome),
            treatment: Cow::Owned(treatment),
            features: features.into_iter().map(Cow::Owned).collect(),
            response: Cow::Owned(response),
            feature_quantities: Cow::Owned(feature_quantities),
        }
        .compile()
    }
}

impl<'a> FitInput<'a> {
    fn compile(self) -> Result<CompiledNativeCohortFit<'a>, EstimationError> {
        let Self { outcome, treatment, features, response, feature_quantities } = &self;
        let working_bytes = working_bytes(
            outcome.len(),
            treatment.len(),
            features.iter().map(|column| column.len()),
        )?;
        if outcome.len() != treatment.len()
            || features.iter().any(|column| column.len() != outcome.len())
            || features.len() != feature_quantities.len()
        {
            return Err(EstimationError::Unsupported {
                message: "cohort linear effect: column lengths or declarations differ",
            });
        }
        let mut question = CanonicalWriter::new("antecedent.native_cohort_fit.question.v1");
        question.text(FIT_PROCEDURE_ID).size(feature_quantities.len());
        for wire in std::iter::once(response.as_ref()).chain(feature_quantities.iter()) {
            let quantity = ScientificQuantity::try_from(wire.clone()).map_err(|_| {
                EstimationError::Unsupported {
                    message: "cohort linear effect: invalid scientific quantity",
                }
            })?;
            quantity.write_canonical(&mut question);
        }
        let mut source = CanonicalWriter::new("antecedent.native_cohort_fit.source.v1");
        source.reals(outcome).reals(treatment).size(features.len());
        for column in features {
            source.reals(column);
        }
        let lowered = lower_fit(
            ProgramCell::NativeCohortFit,
            PLAN_ID,
            question.finish(),
            source.finish(),
            outcome.len(),
            features.len(),
        )
        .map_err(|_| EstimationError::Unsupported {
            message: "cohort linear effect: the native fit program could not be lowered",
        })?;
        Ok(CompiledNativeCohortFit { input: self, working_bytes, lowered })
    }
}

impl CompiledProgramHandle for CompiledNativeCohortFit<'_> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledNativeCohortFit<'_> {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// View the compiled plan without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Execute the admitted native fit and return its program identity.
    ///
    /// # Errors
    /// Unsupported fit, resource admission or cancellation.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<CohortModelFitRun, EstimationError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        gate_program(&scoped, self).map_err(|_| EstimationError::Unsupported {
            message: "cohort linear effect: native fit route is not supported",
        })?;
        let _lease = scoped.acquire_serial_numerical_admission(self.working_bytes)?;
        let input = &self.input;
        let columns: Vec<&[f64]> = input.features.iter().map(Cow::as_ref).collect();
        let fit = NativeCohortFit::fit_controlled(
            &input.outcome,
            &input.treatment,
            &columns,
            &input.response,
            &input.feature_quantities,
            || scoped.cancellation.is_cancelled(),
        );
        if scoped.cancellation.is_cancelled() {
            return Err(AdmissionError::Cancelled.into());
        }
        Ok(CohortModelFitRun { fit: fit?, lowered_program_identity: self.lowered.identity() })
    }
}
