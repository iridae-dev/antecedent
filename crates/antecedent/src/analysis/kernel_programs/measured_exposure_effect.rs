//! The measured-exposure imputation and its Rubin pooling as compiled program families.
//!
//! A measured exposure is imputed from its declaration, the outcome and the covariates
//! (`antecedent_estimate::measurement::impute_exposure`), the caller's analysis runs on every
//! completed table, and the per-draw estimates are pooled with Rubin's rules
//! (`antecedent_estimate::measurement::pool_measured_effect`). Both steps are calls over inputs
//! the caller declares in full, and both are bound to the `measurement.measured_exposure_effect`
//! unit: the imputation under the `MeasuredExposureImputation` route, the pooling under the
//! `MeasuredExposurePooling` route. Compiling binds the canonical digest of the inputs into a
//! six-layer program; running passes the support gate of the unit and then makes the very call
//! the direct path makes, so the draws and the pooled effect are the direct path's bit for bit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::measurement::{
    ExposureDraws, ExposureMeasurement, ImputationData, MeasuredEffect, MeasurementError,
    impute_exposure, pool_measured_effect,
};

use super::exposure_measurement::measurement_refusal;
use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const IMPUTATION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::MeasuredExposureImputation,
    plan_id: "measurement.measured_exposure_effect.imputation",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "measured_exposure_imputation",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "latent exposure drawn from its declared measurement, the outcome and covariates",
};

const POOLING: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::MeasuredExposurePooling,
    plan_id: "measurement.measured_exposure_effect.pooling",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "rubin_rules_pool",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::ChainSource,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: "per-draw estimates and variances pooled with Rubin's rules",
};

/// Owned imputation columns: the outcome and the column-major covariates. [`Self::data`] is
/// the borrowed view the kernel reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ImputationColumns {
    /// The outcome of every unit.
    pub outcome: Vec<f64>,
    /// Covariates, column-major (`covariate_count` columns of `units` values).
    pub covariates: Vec<f64>,
    /// Number of covariate columns.
    pub covariate_count: usize,
}

impl ImputationColumns {
    /// The borrowed view `impute_exposure` reads.
    #[must_use]
    pub fn data(&self) -> ImputationData<'_> {
        ImputationData {
            outcome: &self.outcome,
            covariates: &self.covariates,
            covariate_count: self.covariate_count,
        }
    }
}

#[derive(Clone, Debug)]
struct ImputationRequest {
    declaration: ExposureMeasurement,
    columns: ImputationColumns,
    draw_count: usize,
    seed: u64,
}

#[derive(Clone, Debug)]
struct PoolingRequest {
    points: Vec<f64>,
    variances: Vec<f64>,
    naive: Option<f64>,
}

/// Canonical bytes: the declaration identity (its kind, error structure and every recorded
/// value), the columns, the draw count and the seed.
fn imputation_bytes(request: &ImputationRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.measured_exposure_imputation");
    writer
        .text(&request.declaration.identity())
        .reals(&request.columns.outcome)
        .reals(&request.columns.covariates)
        .word(request.columns.covariate_count as u64)
        .word(request.draw_count as u64)
        .word(request.seed);
    writer.finish()
}

fn pooling_bytes(request: &PoolingRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.measured_exposure_pooling");
    writer.reals(&request.points).reals(&request.variances);
    match request.naive {
        Some(naive) => writer.word(1).real(naive),
        None => writer.word(0),
    };
    writer.finish()
}

/// A compiled measured-exposure imputation.
#[derive(Clone, Debug)]
pub struct CompiledExposureImputation {
    program: KernelProgram<ImputationRequest>,
}

/// A compiled Rubin pooling of per-draw measured-exposure estimates.
#[derive(Clone, Debug)]
pub struct CompiledMeasuredEffectPool {
    program: KernelProgram<PoolingRequest>,
}

impl Study {
    /// Compile the imputation of a measured exposure (the inputs of `impute_exposure`, owned)
    /// under the `MeasuredExposureImputation` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_exposure_imputation(
        declaration: ExposureMeasurement,
        columns: ImputationColumns,
        draw_count: usize,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledExposureImputation, CausalError> {
        let request = ImputationRequest { declaration, columns, draw_count, seed };
        let canonical = imputation_bytes(&request);
        let program = KernelProgram::compile(&IMPUTATION, request, &canonical, ctx)?;
        Ok(CompiledExposureImputation { program })
    }

    /// Compile the Rubin pooling of per-draw estimates and variances (the inputs of
    /// `pool_measured_effect`, owned) under the `MeasuredExposurePooling` route. Compiling
    /// executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_measured_effect_pool(
        points: Vec<f64>,
        variances: Vec<f64>,
        naive: Option<f64>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledMeasuredEffectPool, CausalError> {
        let request = PoolingRequest { points, variances, naive };
        let canonical = pooling_bytes(&request);
        let program = KernelProgram::compile(&POOLING, request, &canonical, ctx)?;
        Ok(CompiledMeasuredEffectPool { program })
    }
}

impl CompiledExposureImputation {
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

    /// Pass the support gate, then impute the latent exposure as `impute_exposure` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the imputation.
    pub fn impute(&self, ctx: &ExecutionContext) -> Result<ExposureDraws, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        impute_exposure(
            &request.declaration,
            &request.columns.data(),
            request.draw_count,
            request.seed,
            ctx,
        )
    }
}

impl CompiledMeasuredEffectPool {
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

    /// Pass the support gate, then pool the per-draw estimates as `pool_measured_effect`
    /// does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not
    /// supported, and every refusal of the pooling.
    pub fn pool(&self, ctx: &ExecutionContext) -> Result<MeasuredEffect, MeasurementError> {
        self.program.gate(ctx).map_err(|refusal| measurement_refusal(&refusal))?;
        let request = self.program.input();
        pool_measured_effect(&request.points, &request.variances, request.naive)
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};

    fn imputation(probability: Vec<f64>, outcome: Vec<f64>, seed: u64) -> [u8; 32] {
        imputation_bytes(&ImputationRequest {
            declaration: ExposureMeasurement::Soft { probability },
            columns: ImputationColumns { outcome, covariates: vec![1.0, 2.0], covariate_count: 1 },
            draw_count: 4,
            seed,
        })
    }

    /// `imputation_bytes`: the declaration enters by its own identity text (framed), the columns
    /// are unit-ordered and count-framed, the draw count and seed by value.
    #[test]
    fn site_measured_exposure_effect_imputation_orders_the_units() {
        ordered_laws(&[1.0, 2.0, 3.0], |outcome| {
            Some(imputation(vec![0.2, 0.8, 0.5], outcome.to_vec(), 1))
        });
        all_distinct(&[
            imputation(vec![0.2, 0.8], vec![1.0, 2.0], 1),
            imputation(vec![0.8, 0.2], vec![1.0, 2.0], 1),
            imputation(vec![0.2, 0.8], vec![1.0, 2.0], 2),
            imputation(vec![0.2, 0.8], vec![1.0], 1),
        ]);
        bit_exact(|x| Some(imputation(vec![0.2, 0.8], vec![x, 2.0], 1)));
    }

    fn pooling(points: &[f64], variances: &[f64], naive: Option<f64>) -> [u8; 32] {
        pooling_bytes(&PoolingRequest {
            points: points.to_vec(),
            variances: variances.to_vec(),
            naive,
        })
    }

    /// `pooling_bytes`: the per-draw points and variances are ordered (the pooled effect reports
    /// them in draw order and sums them in that order) and count-framed; the naive estimate
    /// carries a presence word.
    #[test]
    fn site_measured_exposure_effect_pooling_orders_the_draws() {
        let variances = [0.1, 0.2, 0.3];
        ordered_laws(&[1.0, 2.0, 3.0], |points| Some(pooling(points, &variances, None)));
        all_distinct(&[
            pooling(&[1.0, 2.0], &[0.1, 0.2], None),
            pooling(&[1.0, 2.0], &[0.1, 0.2], Some(0.0)),
            pooling(&[1.0, 2.0], &[0.1, 0.2], Some(1.0)),
            pooling(&[1.0], &[2.0, 0.1, 0.2], None),
        ]);
        bit_exact(|x| Some(pooling(&[1.0, 2.0], &[0.1, 0.2], Some(x))));
    }
}
