//! Multiple imputation: impute, fit, pool as one uncertainty path.
//!
//! A declared [`ImputationSpec`] names the imputation model, the number of completed tables,
//! the sweeps, the seed and the missingness assumption. [`Imputer`] draws completed tables
//! from a *proper* imputation model: each draw first draws the imputation model's
//! parameters from (an approximation of) their posterior, by a Bayesian bootstrap of whole
//! sampling units or a conjugate Jeffreys draw, then draws the missing cells given those
//! parameters. [`run`] fits the caller's estimator on every completed table and pools:
//!
//! * a frequentist fit (point and variance) by Rubin's rules with the Barnard–Rubin degrees
//!   of freedom ([`rubin_pool`]);
//! * a Bayesian fit (posterior draws) by stacking the draws of every completed table into
//!   one equal-weight mixture ([`stack_posterior`]), keeping the between-imputation variance
//!   as its own uncertainty axis.
//!
//! Imputation `j` opens its own keyed random stream ([`antecedent_core::rng_key::RngKey`]
//! with chain `j`), so a completed table depends only on the data, the specification and
//! `j`: never on threads, scheduling or the order in which tables are drawn. Every
//! completed-table buffer is leased from the execution context's admission pool before it is
//! allocated, and the lease travels with the table.
//!
//! The sampling design governs the unit the parameter draw resamples: rows for independent
//! rows, whole subjects or clusters for trajectories and clustered panels. A dependent time
//! series is refused: its blocks are not exchangeable units of an imputation model.
//!
//! Missing not at random is refused with a sensitivity analysis as the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod impute;
mod pool;

use std::fmt;
use std::sync::Arc;

use antecedent_core::refusal::NextAttempt;
use antecedent_core::sampling_design::ResamplingRule;
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext};
use antecedent_data::{ImputationSpec, MissingnessAssumption};

use crate::EstimationError;

pub use impute::{CompletedTable, ImputationColumn, ImputedVariable, Imputer, VariableKind};
pub use pool::{RubinPooled, StackedPosterior, rubin_pool, stack_posterior};

/// Most variables one imputation model reads.
pub const MAX_IMPUTATION_VARIABLES: usize = 64;

/// Semantic operation tag of the imputation streams.
pub const IMPUTATION_OPERATION: u64 = 0x4D49_5F49_4D50_5554;

/// Next attempt when missingness is declared not at random.
pub const SENSITIVITY_ANALYSIS: NextAttempt = NextAttempt::new(
    "sensitivity_analysis",
    "run a missing-not-at-random sensitivity analysis (delta-adjusted or selection-model \
     imputation over a declared grid of departures from missing at random) and report how the \
     effect moves; imputation under missing at random is not valid for this declaration",
);

/// Next attempt when the joint normal model meets an incomplete binary variable.
pub const USE_CHAINED_EQUATIONS: NextAttempt = NextAttempt::new(
    "chained_equations",
    "declare the chained-equations model, which draws a binary variable from a logistic \
     conditional model",
);

/// Next attempt when imputation is not identified or not applicable on the data.
pub const DECLARE_COMPLETE_CASE: NextAttempt = NextAttempt::new(
    "complete_case",
    "declare complete_case (valid when missingness is completely at random, or at random \
     given the adjustment set) or supply more observed rows for the incomplete variable",
);

/// Next attempt when a clustered design has no cluster labels.
pub const SUPPLY_CLUSTER_LABELS: NextAttempt = NextAttempt::new(
    "supply_cluster_labels",
    "supply one cluster label per row so the imputation model's parameter draw resamples \
     whole clusters, or declare complete_case",
);

/// Next attempt for a dependent time series.
pub const TEMPORAL_COMPLETE_CASE: NextAttempt = NextAttempt::new(
    "temporal_complete_case",
    "declare complete_case on the temporal route, which drops the lag windows that read a \
     missing time point and keeps time on its original grid",
);

/// Every next attempt this module names.
pub const MULTIPLE_IMPUTATION_NEXT_ATTEMPTS: [NextAttempt; 5] = [
    SENSITIVITY_ANALYSIS,
    USE_CHAINED_EQUATIONS,
    DECLARE_COMPLETE_CASE,
    SUPPLY_CLUSTER_LABELS,
    TEMPORAL_COMPLETE_CASE,
];

/// A typed refusal of a multiple-imputation request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultipleImputationError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: Option<NextAttempt>,
}

impl MultipleImputationError {
    pub(crate) fn new(
        code: &'static str,
        detail: &'static str,
        message: impl Into<String>,
        next: Option<NextAttempt>,
    ) -> Self {
        let message = message.into();
        let message = match next {
            Some(attempt) => attempt.append_to(&message),
            None => message,
        };
        Self { code, detail, message, next }
    }

    pub(crate) fn invalid(detail: &'static str, message: impl Into<String>) -> Self {
        Self::new(antecedent_core::reason_code!("invalid_argument"), detail, message, None)
    }

    pub(crate) fn not_applicable(
        detail: &'static str,
        message: impl Into<String>,
        next: NextAttempt,
    ) -> Self {
        Self::new(
            antecedent_core::reason_code!("missing_policy_not_applicable"),
            detail,
            message,
            Some(next),
        )
    }

    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `multiple_imputation.<slot>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, ending with the next attempt when one is named.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The attempt to make instead, when one is named.
    #[must_use]
    pub const fn next_attempt(&self) -> Option<NextAttempt> {
        self.next
    }
}

antecedent_core::impl_refused!(MultipleImputationError);

impl fmt::Display for MultipleImputationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for MultipleImputationError {}

impl From<MultipleImputationError> for EstimationError {
    fn from(error: MultipleImputationError) -> Self {
        Self::refused(error.code, format!("{}: {}", error.detail, error.message))
    }
}

/// Check a declared specification: its counts are in range and its missingness assumption
/// is one imputation can serve.
///
/// # Errors
///
/// Counts out of range (`invalid_argument`), or missing not at random
/// (`missing_policy_not_applicable`, next attempt [`SENSITIVITY_ANALYSIS`]).
pub fn check_spec(spec: &ImputationSpec) -> Result<(), MultipleImputationError> {
    if let Some(problem) = spec.range_problem() {
        return Err(MultipleImputationError::invalid("multiple_imputation.spec_range", problem));
    }
    match spec.assumption {
        MissingnessAssumption::MissingAtRandom => Ok(()),
        _ => Err(MultipleImputationError::not_applicable(
            "multiple_imputation.not_at_random",
            "missingness is declared not at random; an imputation model fitted on the observed \
             cells cannot recover a mechanism that depends on the missing values themselves",
            SENSITIVITY_ANALYSIS,
        )),
    }
}

/// The sampling unit an imputation's parameter draw resamples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImputationUnits {
    /// Every row is its own unit.
    Rows,
    /// Dense group index per row (subjects or clusters): rows of a group share one draw.
    Groups(Arc<[u32]>),
}

impl ImputationUnits {
    /// Number of units over `rows` rows.
    #[must_use]
    pub fn count(&self, rows: usize) -> usize {
        match self {
            Self::Rows => rows,
            Self::Groups(groups) => {
                groups.iter().copied().max().map_or(0, |max| usize::try_from(max).unwrap_or(0) + 1)
            }
        }
    }

    /// Stable name of the unit.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Rows => "row",
            Self::Groups(_) => "group",
        }
    }
}

/// Bind the declared sampling design to the unit the imputation model resamples.
///
/// No design and independent subject trajectories resample rows unless `groups` labels the
/// subjects; a clustered panel resamples whole clusters and needs `groups`; a dependent time
/// series is refused.
///
/// # Errors
///
/// A clustered panel without groups, a dependent series, or groups whose length differs
/// from `rows`.
pub fn bind_units(
    design: Option<DeclaredSamplingDesign>,
    groups: Option<Arc<[u32]>>,
    rows: usize,
) -> Result<ImputationUnits, MultipleImputationError> {
    if let Some(groups) = &groups {
        if groups.len() != rows {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.units_shape",
                format!("{} group labels for {rows} rows", groups.len()),
            ));
        }
    }
    let design_error = |error: antecedent_core::sampling_design::SamplingDesignError| {
        MultipleImputationError::invalid("multiple_imputation.sampling_design", error.to_string())
    };
    match design {
        None => Ok(groups.map_or(ImputationUnits::Rows, ImputationUnits::Groups)),
        Some(DeclaredSamplingDesign::IndependentSubjectTrajectories) => {
            DeclaredSamplingDesign::IndependentSubjectTrajectories
                .apply_resampling(ResamplingRule::RowSubjectIid)
                .map_err(design_error)?;
            Ok(groups.map_or(ImputationUnits::Rows, ImputationUnits::Groups))
        }
        Some(DeclaredSamplingDesign::ClusteredPanel) => {
            DeclaredSamplingDesign::ClusteredPanel
                .apply_resampling(ResamplingRule::WholeCluster)
                .map_err(design_error)?;
            groups.map(ImputationUnits::Groups).ok_or_else(|| {
                MultipleImputationError::not_applicable(
                    "multiple_imputation.clusters_required",
                    "the declared clustered panel resamples whole clusters and no cluster label \
                     is bound to the rows",
                    SUPPLY_CLUSTER_LABELS,
                )
            })
        }
        Some(DeclaredSamplingDesign::DependentTimeSeries) => {
            Err(MultipleImputationError::not_applicable(
                "multiple_imputation.dependent_series",
                "a dependent time series has no exchangeable units for the imputation model's \
                 parameter draw",
                TEMPORAL_COMPLETE_CASE,
            ))
        }
    }
}

/// What one completed table's fit reports.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ImputationFit {
    /// Point estimate.
    pub point: f64,
    /// Sampling variance of the point, for a frequentist fit.
    pub variance: Option<f64>,
    /// Complete-data degrees of freedom, when the estimator has a finite one.
    pub df_complete: Option<f64>,
    /// Posterior draws of the quantity, for a Bayesian fit.
    pub draws: Option<Arc<[f64]>>,
}

impl ImputationFit {
    /// A frequentist fit.
    #[must_use]
    pub const fn frequentist(point: f64, variance: f64, df_complete: Option<f64>) -> Self {
        Self { point, variance: Some(variance), df_complete, draws: None }
    }

    /// A Bayesian fit with its posterior draws; the point is their mean.
    #[must_use]
    pub fn bayesian(draws: Arc<[f64]>) -> Self {
        let n = draws.len().max(1);
        #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
        let point = draws.iter().sum::<f64>() / n as f64;
        Self { point, variance: None, df_complete: None, draws: Some(draws) }
    }
}

/// How the fits were pooled.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PooledEstimate {
    /// Rubin's rules over frequentist fits.
    Rubin(RubinPooled),
    /// Equal-weight mixture over Bayesian fits.
    Stacked(StackedPosterior),
}

impl PooledEstimate {
    /// Pooled point.
    #[must_use]
    pub const fn point(&self) -> f64 {
        match self {
            Self::Rubin(pooled) => pooled.point,
            Self::Stacked(stacked) => stacked.mean,
        }
    }

    /// Pooled total variance (Rubin's `T`, or the mixture variance).
    #[must_use]
    pub const fn total_variance(&self) -> f64 {
        match self {
            Self::Rubin(pooled) => pooled.total,
            Self::Stacked(stacked) => stacked.variance,
        }
    }

    /// Between-imputation variance.
    #[must_use]
    pub const fn between(&self) -> f64 {
        match self {
            Self::Rubin(pooled) => pooled.between,
            Self::Stacked(stacked) => stacked.between,
        }
    }

    /// Within-imputation variance.
    #[must_use]
    pub const fn within(&self) -> f64 {
        match self {
            Self::Rubin(pooled) => pooled.within,
            Self::Stacked(stacked) => stacked.within,
        }
    }
}

/// The whole impute-fit-pool procedure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MultipleImputationRun {
    /// The declared specification.
    pub spec: ImputationSpec,
    /// Unit the parameter draws resampled (`row` or `group`).
    pub unit: &'static str,
    /// Number of units resampled.
    pub units: usize,
    /// Cells imputed per variable, in the imputer's variable order.
    pub imputed_cells: Vec<(Arc<str>, usize)>,
    /// Per-imputation fits, in imputation order.
    pub fits: Vec<ImputationFit>,
    /// The pooled estimate.
    pub pooled: PooledEstimate,
}

/// A refusal of the procedure: the imputation itself, or the caller's fit on imputation
/// `imputation`.
#[derive(Clone, Debug)]
pub enum MultipleImputationFailure<E> {
    /// The imputation model refused.
    Imputation(MultipleImputationError),
    /// The fit on one completed table failed.
    Fit {
        /// Zero-based imputation index.
        imputation: u32,
        /// The fit's error.
        error: E,
    },
}

/// Impute every table of `imputer`'s specification, fit `fit` on each and pool the fits.
///
/// Tables are drawn and fitted one at a time and each is dropped (releasing its lease)
/// before the next is drawn, so at most one completed table is held. The fits must all be
/// frequentist (point and variance) or all Bayesian (draws).
///
/// # Errors
///
/// An imputation refusal (including an admission pool that cannot fund a completed table,
/// and cancellation), a fit error, or fits that cannot be pooled.
pub fn run<F, E>(
    imputer: &Imputer,
    ctx: &ExecutionContext,
    mut fit: F,
) -> Result<MultipleImputationRun, MultipleImputationFailure<E>>
where
    F: FnMut(&CompletedTable, &ExecutionContext) -> Result<ImputationFit, E>,
{
    let spec = *imputer.spec();
    let mut fits = Vec::with_capacity(usize::try_from(spec.imputations).unwrap_or(0));
    for index in 0..spec.imputations {
        let table = imputer.impute(index, ctx).map_err(MultipleImputationFailure::Imputation)?;
        let result = fit(&table, ctx)
            .map_err(|error| MultipleImputationFailure::Fit { imputation: index, error })?;
        drop(table);
        fits.push(result);
    }
    let pooled = pool_fits(&fits).map_err(MultipleImputationFailure::Imputation)?;
    Ok(MultipleImputationRun {
        spec,
        unit: imputer.units().name(),
        units: imputer.unit_count(),
        imputed_cells: imputer.imputed_cells(),
        fits,
        pooled,
    })
}

/// Pool per-imputation fits: Rubin's rules when every fit has a variance, a stacked posterior
/// when every fit has draws.
///
/// # Errors
///
/// Fits that mix the two kinds, or that the pooling rules refuse.
pub fn pool_fits(fits: &[ImputationFit]) -> Result<PooledEstimate, MultipleImputationError> {
    if fits.iter().all(|fit| fit.draws.is_some()) {
        let draws: Vec<&[f64]> = fits.iter().filter_map(|fit| fit.draws.as_deref()).collect();
        return stack_posterior(&draws).map(PooledEstimate::Stacked);
    }
    if fits.iter().all(|fit| fit.variance.is_some()) {
        let points: Vec<f64> = fits.iter().map(|fit| fit.point).collect();
        let variances: Vec<f64> = fits.iter().filter_map(|fit| fit.variance).collect();
        let df = fits
            .iter()
            .map(|fit| fit.df_complete)
            .try_fold(f64::INFINITY, |acc, df| df.map(|df| acc.min(df)));
        return rubin_pool(&points, &variances, df.filter(|df| df.is_finite()))
            .map(PooledEstimate::Rubin);
    }
    Err(MultipleImputationError::invalid(
        "multiple_imputation.mixed_fits",
        "every completed table's fit must report a variance (frequentist) or posterior draws \
         (Bayesian); the fits mix the two or report neither",
    ))
}
