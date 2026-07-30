//! Estimator contracts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;

use crate::adjustment::{EffectEstimate, EstimationWorkspace, PreparedEstimationProblem};
use crate::error::EstimationError;

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for crate::adjustment::LinearAdjustmentAte {}

/// Estimator preparation + fit.
///
/// Extension / dispatch surface. Most concrete estimators expose inherent
/// `prepare` / `fit` with estimator-specific prepared types, workspaces, and
/// assumption threading; implement this trait only when those signatures align
/// with [`PreparedEstimationProblem`] / [`EstimationWorkspace`].
///
/// `query` is required to bind intervention levels; DESIGN omits it in the sketch
/// but every ATE estimator needs it at prepare time.
///
/// This trait is sealed: only types in this crate may implement it.
pub trait Estimator<D, Q = AverageEffectQuery>: sealed::Sealed {
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

/// Tabular ATE estimators that produce [`EffectEstimate`].
pub trait TabularAteEstimator:
    Estimator<TabularData, AverageEffectQuery, Fit = EffectEstimate>
{
}
