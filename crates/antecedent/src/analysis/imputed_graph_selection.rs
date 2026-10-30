//! Multiple imputation with the graph selection re-run on every completed table.
//!
//! A graph posterior supplied to a study is fixed before the imputation, so the pooled effect
//! carries no uncertainty about which structure the data support once the missing cells are
//! drawn. Here the structure search is part of the imputed procedure: each completed table is
//! searched afresh for its own structure posterior, the effect study is built on that posterior
//! and run on that table, and the tables' effect posteriors, each already a mixture over its
//! own graphs with its own weights, are stacked with equal weight. The between-table spread
//! therefore contains the change in the selected structures as well as the change in the filled
//! cells.
//!
//! The search runs here, at the top level, and not inside a study's dispatch: a lowered
//! dispatch refuses to build and run another study, and the effect study of each table is a
//! study of its own.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::column::OwnedColumn;
use antecedent_data::{
    ImputationSpec, MissingPolicy, RowAccounting, TableView, TabularData, complete_float_columns,
};
use antecedent_estimate::bayesian::CausalPosterior;
use antecedent_estimate::multiple_imputation::{
    self, ImputationColumn, ImputationFit, Imputer, MultipleImputationFailure, PooledEstimate,
};

use super::StudyBuilder;
use super::missing_data::{self, MissingRoute};
use super::missing_imputation::{analysed_variables, fit_of, refused, stacked_posterior};
use crate::discovery::{BayesianDiscoverParams, GraphPosterior};
use crate::error::CausalError;
use crate::result::{ImputationReport, StudyResult};

/// The structure search re-run on every completed table: an exact (or order-MCMC above the
/// enumeration cap) posterior over directed acyclic graphs of the table's columns.
#[derive(Clone, Debug)]
pub struct GraphSelectionPerImputation {
    params: BayesianDiscoverParams,
    variables: Arc<[VariableId]>,
    domains: Arc<[(Arc<str>, antecedent_data::ImputationDomain)]>,
}

impl GraphSelectionPerImputation {
    /// A search with `params` over `variables`, which must be the table's columns in order
    /// (the posterior's atoms index the searched variables by position).
    #[must_use]
    pub fn new(params: BayesianDiscoverParams, variables: impl Into<Arc<[VariableId]>>) -> Self {
        Self { params, variables: variables.into(), domains: Arc::from([]) }
    }

    /// Declare domains of named imputation columns; duplicate and unknown names are refused.
    #[must_use]
    pub fn with_domains(
        mut self,
        domains: impl Into<Arc<[(Arc<str>, antecedent_data::ImputationDomain)]>>,
    ) -> Self {
        let mut domains = domains.into().to_vec();
        domains.sort_by(|a, b| a.0.cmp(&b.0));
        self.domains = domains.into();
        self
    }

    /// The searched variables.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }
}

/// The pooling rule label of a result whose graph selection ran on every completed table.
pub(crate) const RERUN_POOLING: &str = "stacked_graph_posterior_rerun";

/// Impute `spec.imputations` completed tables of `data`, search each for its structure
/// posterior with `selection`, run the Bayesian effect study `effect_study` builds on that
/// posterior and table, and stack the effect posteriors.
///
/// `effect_study` receives the completed table and its graph posterior and returns the study
/// to run: a Bayesian average-effect study over `graph_posterior(..)` with no missing-data
/// declaration (the table is complete). Every table must run `bayesian.gcomp`.
///
/// The result is the first table's result with the stacked effect posterior, the pooled mean
/// and standard deviation of the effect, and a missing-data report whose pooling is
/// `stacked_graph_posterior_rerun`; its claim evidence names
/// `graph_selection_rerun_per_imputation`. No interval is published.
///
/// # Errors
///
/// A specification out of range or declared not at random; a table whose columns are not all
/// float columns in the searched order; a discovery failure; the effect study's own errors; an
/// estimator other than `bayesian.gcomp`; posteriors that differ in draws or quantities.
pub fn estimate_with_graph_selection_per_imputation<F>(
    data: &TabularData,
    spec: ImputationSpec,
    selection: &GraphSelectionPerImputation,
    ctx: &ExecutionContext,
    effect_study: F,
) -> Result<StudyResult, CausalError>
where
    F: Fn(&TabularData, GraphPosterior) -> StudyBuilder,
{
    multiple_imputation::check_spec(&spec).map_err(|error| CausalError::Estimate(error.into()))?;
    let float_ids: Vec<VariableId> = data
        .storage()
        .columns()
        .iter()
        .filter_map(|column| match column {
            OwnedColumn::Float64(float) => Some(float.id),
            _ => None,
        })
        .collect();
    let all_float = float_ids.len() == data.schema().variables().len();
    if !all_float || float_ids.as_slice() != selection.variables() {
        return Err(refused(
            "a graph selection re-run on every completed table needs a table of float columns \
             only, searched in column order, because the posterior's atoms index the searched \
             variables by position",
        ));
    }
    if float_ids.len() > multiple_imputation::MAX_IMPUTATION_VARIABLES {
        return Err(refused(format!(
            "the table has {} float columns and an imputation model reads at most {}",
            float_ids.len(),
            multiple_imputation::MAX_IMPUTATION_VARIABLES
        )));
    }
    let mut columns = Vec::with_capacity(float_ids.len());
    for &id in &float_ids {
        let name = Arc::clone(&data.schema().get(id)?.name);
        let values: Arc<[f64]> = Arc::from(data.float64_values(id)?);
        columns.push(ImputationColumn::new(name, values));
    }
    let units = multiple_imputation::bind_units(None, None, data.row_count())
        .map_err(|error| CausalError::Estimate(error.into()))?;
    let imputer = Imputer::new_with_domains(columns, spec, units, &selection.domains)
        .map_err(|error| CausalError::Estimate(error.into()))?;
    let imputed: Vec<usize> = imputer
        .variables()
        .iter()
        .enumerate()
        .filter(|(_, v)| v.missing > 0)
        .map(|(j, _)| j)
        .collect();

    let mut base: Option<StudyResult> = None;
    let mut posteriors: Vec<CausalPosterior> = Vec::new();
    let run = multiple_imputation::run(&imputer, ctx, |table, ctx| {
        let replacements: Vec<(VariableId, Arc<[f64]>)> = imputed
            .iter()
            .map(|&j| (float_ids[j], Arc::clone(&table.columns()[j])))
            .collect();
        let completed = complete_float_columns(data, &replacements)?;
        // The structure is selected afresh on this table, so the pooled spread carries it.
        let graph_posterior = crate::discovery::discover_exact_dag_posterior(
            &completed,
            &selection.variables,
            &selection.params,
            ctx,
        )?;
        let result = effect_study(&completed, graph_posterior).build()?.run(ctx)?;
        let estimator = result.logical_plan.estimator.as_deref().unwrap_or("");
        if estimator != "bayesian.gcomp" {
            return Err(refused(format!(
                "a graph selection re-run on every completed table stacks `bayesian.gcomp` \
                 effect posteriors; the study ran `{estimator}`"
            )));
        }
        let fit = fit_of(&result, true)?;
        if let Some(posterior) = &result.posterior {
            posteriors.push(posterior.clone());
        }
        if base.is_none() {
            base = Some(result);
        }
        Ok::<ImputationFit, CausalError>(fit)
    })
    .map_err(|failure| match failure {
        MultipleImputationFailure::Imputation(error) => CausalError::Estimate(error.into()),
        MultipleImputationFailure::Fit { error, .. } => error,
    })?;
    let mut result = base.ok_or_else(|| refused("no completed table was fitted"))?;
    let PooledEstimate::Stacked(stacked) = &run.pooled else {
        return Err(refused("the pooling rule is not one a Bayesian effect posterior reports"));
    };
    result.posterior = Some(stacked_posterior(&posteriors)?);
    if let Some(effect) = result.estimate.as_effect_mut() {
        effect.ate = stacked.mean;
        effect.uncertainty.analytic.se = stacked.variance.sqrt();
        effect.uncertainty.analytic.kind = None;
        effect.uncertainty.analytic.reference_df = None;
    }
    let points: Arc<[f64]> = run.fits.iter().map(|fit| fit.point).collect();
    let estimator: Arc<str> = Arc::from(result.logical_plan.estimator.as_deref().unwrap_or(""));
    let report = ImputationReport {
        spec,
        unit: Arc::from(run.unit),
        units: run.units,
        imputed_cells: run.imputed_cells.clone().into(),
        estimator,
        pooling: Arc::from(RERUN_POOLING),
        points,
        within: stacked.within,
        between: stacked.between,
        total: stacked.variance,
        df: None,
        df_complete: None,
        fraction_missing_information: None,
        relative_increase: None,
        assumptions_digest: Arc::from(run.assumptions.digest()),
        imputation_domains: run.assumptions.domains.clone().into(),
        binary_boundary_variables: run.assumptions.binary_boundary_variables.clone().into(),
        lags: 0,
        vector: None,
    };
    result.extras.interval = None;
    result.executed_contract = None;
    let variables: Vec<VariableId> = analysed_variables(&result);
    let accounting = RowAccounting::imputed(
        data,
        data.storage().analysis_mask(),
        MissingPolicy::MultipleImputation(spec),
        &variables,
    )?;
    let covariates: Arc<[Arc<str>]> = Arc::from(Vec::new());
    let mut missing = missing_data::report(MissingRoute::Static, true, covariates, accounting);
    missing.imputation = Some(report);
    result.extras.missing_data = Some(missing);
    Ok(result)
}
