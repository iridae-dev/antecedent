//! Estimator contracts (DESIGN.md §14.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AverageEffectQuery, ExecutionContext};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;

use crate::adjustment::{EffectEstimate, EstimationWorkspace, PreparedEstimationProblem};
use crate::error::EstimationError;

/// Estimator preparation + fit (DESIGN §14.1).
///
/// Extension / dispatch surface. Most concrete estimators expose inherent
/// `prepare` / `fit` with estimator-specific prepared types, workspaces, and
/// assumption threading; implement this trait only when those signatures align
/// with [`PreparedEstimationProblem`] / [`EstimationWorkspace`].
///
/// `query` is required to bind intervention levels; DESIGN omits it in the sketch
/// but every ATE estimator needs it at prepare time.
pub trait Estimator<D, Q = AverageEffectQuery> {
    /// Fitted artifact type.
    type Fit;

    /// Compile data + estimand + query into a reusable prepared problem.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, data/schema issues, or unsupported query options.
    fn prepare(
        &self,
        data: &D,
        estimand: &IdentifiedEstimand,
        query: &Q,
        ctx: &ExecutionContext,
    ) -> Result<PreparedEstimationProblem, EstimationError>;

    /// Fit the prepared problem.
    ///
    /// # Errors
    ///
    /// Numerical / stats failures.
    fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, EstimationError>;
}

/// Batch estimation against a fitted estimator (DESIGN §14.1).
///
/// Reserved extension point — not part of the day-1 public surface. No concrete
/// estimator implements this yet; use inherent `fit` / bootstrap paths instead.
#[allow(dead_code)] // DESIGN §14.1 reserved contract
pub(crate) trait FittedEstimator<Q> {
    /// Estimate one or more queries into `output`.
    ///
    /// # Errors
    ///
    /// Incompatible query or numerical failure.
    fn estimate_batch(
        &self,
        queries: &[Q],
        output: &mut [EffectEstimate],
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}

/// Tabular ATE estimators that produce [`EffectEstimate`].
pub trait TabularAteEstimator:
    Estimator<TabularData, AverageEffectQuery, Fit = EffectEstimate>
{
}
