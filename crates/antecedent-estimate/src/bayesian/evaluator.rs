//! Batched posterior functional evaluation and posterior-draw helpers.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_prob::{
    BayesLikelihood, EffectBatch, PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace,
    PosteriorQuantityKind, PosteriorSchema,
};
use antecedent_stats::GlmFamily;

use crate::error::EstimationError;
use crate::gcomp::GcompContrast;

/// Trait for batched posterior functional evaluation.
pub trait PosteriorFunctionalEvaluator {
    /// Compiled plan type.
    type Compiled;

    /// Compile against a posterior schema.
    ///
    /// # Errors
    ///
    /// Incompatible schema.
    fn compile(&self) -> Result<Self::Compiled, EstimationError>;

    /// Evaluate a batch of coefficient draws into effects.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn evaluate_batch(
        &self,
        compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}

/// Compiled g-comp ATE evaluator (finite-difference mean contrast).
#[derive(Clone, Debug)]
pub struct GCompAteEvaluator {
    /// Mean family.
    pub family: GlmFamily,
    /// Treatment column.
    pub treatment_col: usize,
    /// Active level.
    pub active: f64,
    /// Control level.
    pub control: f64,
    /// Rows.
    pub nrows: usize,
    /// Cols.
    pub ncols: usize,
    /// Design matrix (column-major).
    pub matrix: Arc<[f64]>,
}

/// Empty compiled marker (evaluator is self-contained).
#[derive(Clone, Copy, Debug, Default)]
pub struct CompiledGCompAte;

impl PosteriorFunctionalEvaluator for GCompAteEvaluator {
    type Compiled = CompiledGCompAte;

    fn compile(&self) -> Result<Self::Compiled, EstimationError> {
        if self.treatment_col >= self.ncols {
            return Err(EstimationError::stats_msg("treatment column out of range"));
        }
        Ok(CompiledGCompAte)
    }

    fn evaluate_batch(
        &self,
        _compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError> {
        const PARALLEL_WORK_FLOOR: usize = 1 << 16;
        let n_draws = posterior.len;
        workspace.prepare(n_draws, self.ncols);
        output.prepare(n_draws);

        // Coefficient columns 0..ncols from the batch (ignore extra quantities).
        let mut coef_cols: Vec<&[f64]> = Vec::with_capacity(self.ncols);
        for c in 0..self.ncols {
            let col = posterior.column(c).map_err(EstimationError::from)?;
            coef_cols.push(col);
        }

        // Non-identity links average over every row per draw (O(draws × n × p)); the
        // draws are independent and draw-order results are kept, so the pass is spread
        // over the context's thread budget once it is large enough to repay the spawn.
        let gcomp = GcompContrast {
            family: self.family,
            x_colmajor: &self.matrix,
            nrows: self.nrows,
            ncols: self.ncols,
            t_col: self.treatment_col,
            active: self.active,
            control: self.control,
        };
        let contrast = |beta: &[f64]| gcomp.mean_contrast(beta);
        let nonlinear = !matches!(self.family, GlmFamily::GaussianIdentity);
        if nonlinear && n_draws * self.nrows * self.ncols >= PARALLEL_WORK_FLOOR {
            let values = ctx.map_indexed(n_draws, |d, _| {
                let beta: Vec<f64> = coef_cols.iter().map(|col| col[d]).collect();
                Ok::<f64, EstimationError>(contrast(&beta))
            })?;
            output.values[..n_draws].copy_from_slice(&values);
            return Ok(());
        }
        for d in 0..n_draws {
            for (slot, col) in workspace.row.iter_mut().zip(&coef_cols) {
                *slot = col[d];
            }
            output.values[d] = contrast(&workspace.row[..self.ncols]);
        }
        Ok(())
    }
}

pub(super) fn likelihood_to_glm_family(l: BayesLikelihood) -> GlmFamily {
    match l {
        BayesLikelihood::GaussianIdentity => GlmFamily::GaussianIdentity,
        BayesLikelihood::BernoulliLogit => GlmFamily::BinomialLogit,
        BayesLikelihood::BernoulliProbit => GlmFamily::BinomialProbit,
        BayesLikelihood::PoissonLog => GlmFamily::PoissonLog,
    }
}

pub(super) fn prob_err(e: antecedent_prob::ProbError) -> EstimationError {
    EstimationError::from(e)
}

/// Concatenate two coefficient-only posterior draw tables (same schema).
/// Concatenate an initial draw block with follow-on blocks in one pass.
///
/// Column-major layout identical to pairwise-merging the blocks in order,
/// without the quadratic intermediate copies.
pub(super) fn concat_coefficient_draws(
    first: &PosteriorDraws,
    rest: &[PosteriorDraws],
) -> Result<PosteriorDraws, EstimationError> {
    if rest.is_empty() {
        return Ok(first.clone());
    }
    for block in rest {
        if block.schema != first.schema {
            return Err(EstimationError::stats_msg("concat_coefficient_draws: schema mismatch"));
        }
    }
    let n_q = first.schema.quantities.len();
    let n = first.n_draws + rest.iter().map(|b| b.n_draws).sum::<usize>();
    let mut values = vec![0.0; n * n_q];
    for q in 0..n_q {
        let mut offset = q * n;
        let col = first.column(q).map_err(EstimationError::from)?;
        values[offset..offset + first.n_draws].copy_from_slice(col);
        offset += first.n_draws;
        for block in rest {
            let col = block.column(q).map_err(EstimationError::from)?;
            values[offset..offset + block.n_draws].copy_from_slice(col);
            offset += block.n_draws;
        }
    }
    PosteriorDraws::from_column_major(first.schema.clone(), n, values)
        .map_err(EstimationError::from)
}

/// Keep coefficient columns only (drop residual-variance / other non-β quantities).
pub(super) fn coefficient_only_draws(
    draws: &PosteriorDraws,
) -> Result<PosteriorDraws, EstimationError> {
    let coef_idx: Vec<usize> = draws
        .schema
        .quantities
        .iter()
        .enumerate()
        .filter_map(|(i, q)| matches!(q, PosteriorQuantityKind::Coefficient { .. }).then_some(i))
        .collect();
    if coef_idx.is_empty() {
        return Err(EstimationError::stats_msg(
            "coefficient_only_draws: no coefficient quantities",
        ));
    }
    if coef_idx.len() == draws.schema.quantities.len() {
        return Ok(draws.clone());
    }
    let n = draws.n_draws;
    let n_q = coef_idx.len();
    let mut quantities = Vec::with_capacity(n_q);
    let mut values = vec![0.0; n * n_q];
    for (dest, &src) in coef_idx.iter().enumerate() {
        quantities.push(draws.schema.quantities[src].clone());
        let col = draws.column(src).map_err(EstimationError::from)?;
        values[dest * n..(dest + 1) * n].copy_from_slice(col);
    }
    PosteriorDraws::from_column_major(
        PosteriorSchema { quantities: Arc::from(quantities) },
        n,
        values,
    )
    .map_err(EstimationError::from)
}
