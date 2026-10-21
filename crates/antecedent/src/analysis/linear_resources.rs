// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checked design dimensions and conservative, overflow-checked allocation quotes.
use crate::CausalError;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ColumnView, Contrast, TableView, TabularData};
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
    for limit in [ctx.memory.soft_limit_bytes, ctx.memory.hard_limit_bytes].into_iter().flatten() {
        if bytes > limit {
            return Err(CausalError::Resource {
                message: format!(
                    "checked linear memory quote {bytes} exceeds caller limit {limit}"
                ),
            });
        }
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
    let mut columns = 2_usize;
    let mut expansion = 0_u64;
    let mut direct = true;
    for &id in adjustment {
        let width = match data.column(id)? {
            ColumnView::Categorical(column) => {
                direct = false;
                let contrast = fitter
                    .contrasts
                    .iter()
                    .find(|(variable, _)| *variable == id)
                    .ok_or_else(|| CausalError::Compile {
                        message: "categorical adjustment requires explicit contrast coding".into(),
                    })?;
                let width = match &contrast.1 {
                    Contrast::Custom(matrix) => matrix.n_columns,
                    Contrast::FullRankIndicator => column.domain.len(),
                    _ => column.domain.len().saturating_sub(1),
                };
                let cells = (data.row_count() as u64)
                    .checked_add(column.domain.len() as u64)
                    .ok_or_else(overflow)?;
                expansion = add(expansion, mul(mul(cells, width as u64)?, 8)?)?;
                width
            }
            _ => 1,
        };
        columns = columns.checked_add(width).ok_or_else(overflow)?;
    }
    let resources =
        LinearResources::new(data.row_count(), columns, fitter.bootstrap_replicates, ctx)?;
    let preparation = add(resources.resident, add(expansion, mul(data.row_count() as u64, 32)?)?)?;
    check_limit(preparation, ctx)?;
    resources.check(ctx)?;
    Ok((resources, preparation, direct))
}
