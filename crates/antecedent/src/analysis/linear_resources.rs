// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked design dimensions and conservative, overflow-checked allocation quotes.
use crate::CausalError;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{CheckedLinearAdjustmentAte, LinearAdjustmentAte};

pub(crate) const RESULT_ENVELOPE_BYTES: u64 =
    std::mem::size_of::<crate::StudyResult>() as u64 + 8192;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LinearResources {
    pub(crate) rows: u64,
    pub(crate) columns: u64,
    pub(crate) resident: u64,
    pub(crate) workspace: u64,
    pub(crate) output: u64,
    pub(crate) work: u64,
}
fn overflow() -> CausalError {
    CausalError::Resource { message: "checked linear resource arithmetic overflow".into() }
}
fn add(a: u64, b: u64) -> Result<u64, CausalError> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, CausalError> {
    a.checked_mul(b).ok_or_else(overflow)
}
impl LinearResources {
    pub(crate) fn new(
        rows: usize,
        columns: usize,
        replicates: u32,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let n = u64::try_from(rows).map_err(|_| overflow())?;
        let p = u64::try_from(columns).map_err(|_| overflow())?;
        let matrix = mul(mul(n, p)?, 8)?;
        // Final design, outcome, source-row indices and retained treatment values.
        // Metadata gets its own conservative allowance; none of these quotes is RSS.
        let resident = add(add(matrix, mul(n, 24)?)?, add(mul(p, 128)?, 4096)?)?;
        // QR/factor buffers, uncertainty scratch and one fit workspace per admitted worker.
        // Optimized resampling may use less, but its QR fallback must still fit.
        let per_worker =
            add(add(mul(matrix, 8)?, mul(mul(p, p)?, 128)?)?, add(mul(n, 128)?, 65536)?)?;
        let workers = u64::from(ctx.parallelism.max_threads.get().min(replicates.max(1)));
        let workspace = mul(per_worker, add(workers, 1)?)?;
        let output = add(
            add(add(mul(n, 8)?, mul(u64::from(replicates), 8)?)?, 4096)?,
            RESULT_ENVELOPE_BYTES,
        )?;
        let work = mul(mul(mul(n, p)?, p)?, add(u64::from(replicates), 1)?)?;
        Ok(Self { rows: n, columns: p, resident, workspace, output, work })
    }
    pub(crate) fn prepared(
        prepared: &CheckedLinearAdjustmentAte,
        fitter: &LinearAdjustmentAte,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let design = &prepared.problem().design;
        Self::new(design.nrows, design.ncols, fitter.bootstrap_replicates, ctx)
    }
    pub(crate) fn execution_bytes(self) -> Result<u64, CausalError> {
        add(self.workspace, self.output)
    }
    pub(crate) fn peak(self) -> Result<u64, CausalError> {
        add(self.resident, self.execution_bytes()?)
    }
    pub(crate) fn check(self, ctx: &ExecutionContext) -> Result<(), CausalError> {
        check_limit(self.peak()?, ctx)
    }
}
pub(crate) fn check_limit(bytes: u64, ctx: &ExecutionContext) -> Result<(), CausalError> {
    if let Some(limit) = ctx.memory_limit_bytes().filter(|limit| bytes > *limit) {
        return Err(CausalError::Resource {
            message: format!("checked linear memory quote {bytes} exceeds caller limit {limit}"),
        });
    }
    Ok(())
}
/// Quote the selected roles and actual contrast widths before allocating the design.
pub(crate) fn source_layout(
    data: &TabularData,
    adjustment: &[VariableId],
    fitter: &LinearAdjustmentAte,
    ctx: &ExecutionContext,
) -> Result<(LinearResources, u64, bool), CausalError> {
    let antecedent_estimate::design_width::DesignWidthQuote {
        columns,
        expansion_bytes: expansion,
        direct,
        ..
    } = quote_width(data, adjustment, fitter)?;
    let resources =
        LinearResources::new(data.row_count(), columns, fitter.bootstrap_replicates, ctx)?;
    let materialized_columns = if direct {
        0
    } else {
        // Materialized backend holds selected columns while assembling the final design.
        mul(mul(data.row_count() as u64, columns as u64)?, 8)?
    };
    // Vec-to-Arc conversion allocates the shared design while the source Vec is live.
    let shared_matrix_transfer = mul(mul(data.row_count() as u64, columns as u64)?, 8)?;
    let preparation = add(
        resources.resident,
        add(
            shared_matrix_transfer,
            add(materialized_columns, add(expansion, mul(data.row_count() as u64, 32)?)?)?,
        )?,
    )?;
    check_limit(preparation, ctx)?;
    resources.check(ctx)?;
    Ok((resources, preparation, direct))
}

/// The encoded design width of a linear adjustment over `adjustment` under the
/// fitter's contrast codings. A missing contrast or a missing column refuses
/// compilation, as it would refuse preparation.
pub(crate) fn quote_width(
    data: &TabularData,
    adjustment: &[VariableId],
    fitter: &LinearAdjustmentAte,
) -> Result<antecedent_estimate::design_width::DesignWidthQuote, CausalError> {
    antecedent_estimate::design_width::quote_adjustment_design(data, adjustment, &fitter.contrasts)
        .map_err(|error| CausalError::Compile { message: error.to_string() })
}
