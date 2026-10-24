//! Row sources for the streamed reductions of the nuisance-model estimators.
//!
//! A source reads batches of rows straight from borrowed float64 columns through the
//! complete-case mask, and hands the weighted-Gaussian capacity engine one design row, one
//! response and one weight per row. No design is ever built: memory is one batch.
//!
//! Two modes cover every nuisance-model reduction. [`RowMode::Plain`] gives each row unit
//! weight and the outcome as the response, so the engine's moments are the Gram of the design
//! and its cross-products with the outcome (two-stage least squares reads these).
//! [`RowMode::LogisticWorking`] gives the iteratively-reweighted-least-squares working weight
//! `μ(1−μ)` and working response `η + (t−μ)/(μ(1−μ))` at a fixed coefficient vector, so one
//! pass of the engine is exactly one IRLS step of the logistic propensity model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::capacity_account::{DesignShape, WeightedCapacityError, WeightedRowSource};

/// What each row contributes to the engine.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RowMode<'a> {
    /// Unit weight; the response is the outcome column.
    Plain {
        /// Response column.
        outcome: &'a [f64],
    },
    /// Logistic working weight and response at `beta`; the response column is the treatment
    /// coded `1` at `active` and `0` otherwise.
    LogisticWorking {
        /// Treatment column as stored.
        treatment: &'a [f64],
        /// The level coded `1`.
        active: f64,
        /// Current coefficients, one per design column.
        beta: &'a [f64],
    },
}

/// Which kept rows a pass reads: a cross-fit fold's training rows, one treatment arm, or both.
///
/// Rows are numbered in the order the complete-case mask keeps them, which is the order of the
/// fold ids, so the filter is a function of the row's place among the kept rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowFilter<'a> {
    /// The fold id of every kept row.
    pub(crate) fold_ids: &'a [u32],
    /// Skip the rows of this fold (a training pass).
    pub(crate) exclude_fold: Option<u32>,
    /// Read only the rows of one arm: the treatment column as stored, the level coded `1`, and
    /// whether the arm read is the treated one.
    pub(crate) arm: Option<ArmFilter<'a>>,
}

/// One treatment arm, by the stored treatment column.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArmFilter<'a> {
    /// Treatment column as stored.
    pub(crate) treatment: &'a [f64],
    /// The level coded `1`.
    pub(crate) active: f64,
    /// Whether the arm read is the one at `active`.
    pub(crate) treated: bool,
}

/// Batches of design rows read through a complete-case mask.
pub(crate) struct MaskedRows<'a> {
    intercept: bool,
    columns: Vec<&'a [f64]>,
    mode: RowMode<'a>,
    mask: &'a [bool],
    filter: Option<RowFilter<'a>>,
    kept: usize,
    cursor: usize,
    bound: Option<usize>,
}

/// The logistic mean, clamped away from 0 and 1 exactly as the resident IRLS clamps it.
fn clamped_mean(eta: f64) -> f64 {
    antecedent_kernels::sigmoid(eta).clamp(1e-9, 1.0 - 1e-9)
}

impl<'a> MaskedRows<'a> {
    /// Rows of `[1 | columns…]` (the intercept when `intercept`) under `mode`.
    pub(crate) fn new(
        intercept: bool,
        columns: Vec<&'a [f64]>,
        mode: RowMode<'a>,
        mask: &'a [bool],
    ) -> Self {
        Self { intercept, columns, mode, mask, filter: None, kept: 0, cursor: 0, bound: None }
    }

    /// The same source reading only the rows `filter` keeps.
    pub(crate) fn filtered(mut self, filter: RowFilter<'a>) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Design columns per row.
    pub(crate) fn design_columns(&self) -> usize {
        usize::from(self.intercept) + self.columns.len()
    }
}

impl WeightedRowSource for MaskedRows<'_> {
    fn ncols(&self) -> usize {
        self.design_columns()
    }

    fn input_ownership_bytes(&self) -> u64 {
        // The mask and, under a fold filter, the fold ids are the only resident structures the
        // reduction keeps; the columns are the caller's table.
        let fold_bytes = self.filter.map_or(0, |filter| 4 * filter.fold_ids.len());
        u64::try_from(self.mask.len() + fold_bytes).unwrap_or(u64::MAX)
    }

    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
        if batch_rows == 0 { Err(WeightedCapacityError::BatchBound) } else { Ok(0) }
    }

    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        0
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.cursor = 0;
        self.kept = 0;
        Ok(())
    }

    #[allow(
        clippy::float_cmp,
        reason = "the declared treatment level is an exact category label, not a computed value"
    )]
    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError> {
        let bound = self.bound.ok_or(WeightedCapacityError::DecoderNotPrepared)?;
        if max_rows == 0 || max_rows > bound {
            return Err(WeightedCapacityError::ReadPastBatch { requested: max_rows, bound });
        }
        let ncols = self.design_columns();
        if x_rowmajor.len() < max_rows * ncols || y.len() < max_rows || weights.len() < max_rows {
            return Err(WeightedCapacityError::Shape(DesignShape::BatchWindow));
        }
        let mut taken = 0;
        while taken < max_rows && self.cursor < self.mask.len() {
            let row = self.cursor;
            self.cursor += 1;
            if !self.mask[row] {
                continue;
            }
            let place = self.kept;
            self.kept += 1;
            if let Some(filter) = &self.filter {
                if filter.exclude_fold == Some(filter.fold_ids[place]) {
                    continue;
                }
                if let Some(arm) = &filter.arm {
                    if (arm.treatment[row] == arm.active) != arm.treated {
                        continue;
                    }
                }
            }
            let cells = &mut x_rowmajor[taken * ncols..(taken + 1) * ncols];
            let mut next = 0;
            if self.intercept {
                cells[0] = 1.0;
                next = 1;
            }
            for column in &self.columns {
                cells[next] = column[row];
                next += 1;
            }
            match self.mode {
                RowMode::Plain { outcome } => {
                    y[taken] = outcome[row];
                    weights[taken] = 1.0;
                }
                RowMode::LogisticWorking { treatment, active, beta } => {
                    let eta: f64 = cells.iter().zip(beta).map(|(x, b)| x * b).sum();
                    let mean = clamped_mean(eta);
                    let variance = mean * (1.0 - mean);
                    let coded = if treatment[row] == active { 1.0 } else { 0.0 };
                    y[taken] = eta + (coded - mean) / variance;
                    weights[taken] = variance;
                }
            }
            taken += 1;
        }
        Ok(taken)
    }
}

/// What the engine passes of one reduction cost, accumulated.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EngineFacts {
    /// Passes over the source rows.
    pub(crate) passes: u64,
    /// Largest number of bytes one pass reserved against the ceiling.
    pub(crate) reserved: u64,
    /// Batches read by the last pass.
    pub(crate) batches: u64,
    /// The executed backend choice of the last pass.
    pub(crate) choice: Option<antecedent_core::physical_choice::PhysicalChoice>,
}

impl EngineFacts {
    fn absorb(&mut self, result: &antecedent_io_core::capacity_account::WeightedCapacityResult) {
        self.passes += result.source_passes;
        self.reserved = self.reserved.max(result.charges.total_bytes);
        self.batches = result.batches;
        self.choice = Some(*result.physical_choice());
    }
}

/// One pass of the weighted-Gaussian engine over `source`.
///
/// # Errors
///
/// The engine's refusal (a ceiling below the window, a rank-deficient design, a read error).
pub(crate) fn run_engine(
    source: &mut MaskedRows<'_>,
    options: &antecedent_io_core::capacity_account::CapacityOptions,
    facts: &mut EngineFacts,
) -> Result<
    antecedent_io_core::capacity_account::WeightedCapacityResult,
    crate::error::EstimationError,
> {
    use antecedent_io_core::capacity_account::{CapacityControl, run_weighted_capacity};
    let result = run_weighted_capacity(CapacityControl::Glm, source, options)
        .map_err(|error| crate::error::EstimationError::data_msg(error.to_string()))?
        .into_resident()
        .ok_or_else(|| {
            crate::error::EstimationError::data_msg("the reduction summary was not resident")
        })?;
    facts.absorb(&result);
    Ok(result)
}

/// Logistic IRLS over the rows `filter` keeps: one engine pass per iteration, from zero, with
/// the resident iteration's clamp and coefficient-change tolerance.
///
/// Returns the coefficients and the iterations taken.
///
/// # Errors
///
/// An engine refusal, a non-finite step, or no convergence within `glm.max_iter`.
#[allow(
    clippy::too_many_arguments,
    reason = "the columns, the declared levels, the row selection and the engine and fit options are independent inputs"
)]
pub(crate) fn fit_logistic_streamed(
    covariates: &[&[f64]],
    treatment: &[f64],
    active: f64,
    mask: &[bool],
    filter: Option<RowFilter<'_>>,
    options: &antecedent_io_core::capacity_account::CapacityOptions,
    glm: &antecedent_stats::GlmOptions,
    facts: &mut EngineFacts,
) -> Result<(Vec<f64>, u32), crate::error::EstimationError> {
    let ncols = 1 + covariates.len();
    let mut beta = vec![0.0; ncols];
    for iteration in 1..=glm.max_iter {
        let result = {
            let mut source = MaskedRows::new(
                true,
                covariates.to_vec(),
                RowMode::LogisticWorking { treatment, active, beta: &beta },
                mask,
            );
            if let Some(filter) = filter {
                source = source.filtered(filter);
            }
            run_engine(&mut source, options, facts)?
        };
        if result.coefficients.len() != ncols || result.coefficients.iter().any(|v| !v.is_finite())
        {
            return Err(crate::error::EstimationError::data_msg(
                "the logistic propensity step produced a non-finite coefficient",
            ));
        }
        let max_delta = result
            .coefficients
            .iter()
            .zip(&beta)
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f64, f64::max);
        beta.clone_from(&result.coefficients);
        if max_delta < glm.tol {
            return Ok((beta, iteration));
        }
    }
    Err(crate::error::EstimationError::data_msg(
        "GLM IRLS did not converge; refuse propensity/outcome scores",
    ))
}
