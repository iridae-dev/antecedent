//! Multiple imputation on the average-effect route: impute, fit, pool as one path.
//!
//! The study's declared [`ImputationSpec`] draws completed tables with
//! [`antecedent_estimate::multiple_imputation`]; the route's own estimator (the retained
//! click of a prepared study, or the ordinary dispatch of a fresh one) runs on every
//! completed table; the fits are pooled by Rubin's rules with the Barnard–Rubin degrees of
//! freedom (frequentist) or stacked into one equal-weight posterior (Bayesian). The pooled
//! result keeps the first completed table's plan, identification and diagnostics, carries
//! the pooled estimate and variance, and its missing-data report names every completed
//! table's estimate and the variance components. No interval is published until a coverage
//! record measures the whole procedure.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, ExecutionContext, VariableId};
use antecedent_data::column::OwnedColumn;
use antecedent_data::{
    ImputationSpec, MissingDeclaration, RowAccounting, TableView, TabularData,
    complete_float_columns,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::bayesian::CausalPosterior;
use antecedent_estimate::multiple_imputation::{
    self, ImputationColumn, ImputationFit, ImputationUnits, Imputer, MultipleImputationFailure,
    PooledEstimate,
};
use antecedent_prob::PosteriorDraws;

use super::execute::Study;
use super::missing_data::{self, MissingRoute};
use super::missing_support::route_missing_support;
use crate::error::CausalError;
use crate::result::{ImputationReport, StudyResult};

/// The declared imputation of `study`, when its policy is multiple imputation.
pub(crate) fn declared_imputation(study: &Study) -> Option<(&MissingDeclaration, ImputationSpec)> {
    let declaration = study.missing.as_ref()?;
    declaration.policy.imputation().map(|spec| (declaration, spec))
}

fn refused(message: impl Into<String>) -> CausalError {
    CausalError::Estimate(EstimationError::refused(
        antecedent_core::reason_code!("missing_policy_not_applicable"),
        format!("missing_data.multiple_imputation: {}", message.into()),
    ))
}

/// The variables the imputation model reads: every float column when no auxiliary variable
/// is declared, else the query's treatment and outcome and the declared auxiliary variables.
fn model_variables(
    study: &Study,
    declaration: &MissingDeclaration,
    data: &TabularData,
) -> Result<Vec<VariableId>, CausalError> {
    let float_ids: Vec<VariableId> = data
        .storage()
        .columns()
        .iter()
        .filter_map(|column| match column {
            OwnedColumn::Float64(float) => Some(float.id),
            _ => None,
        })
        .collect();
    if declaration.covariates.is_empty() {
        if float_ids.len() > multiple_imputation::MAX_IMPUTATION_VARIABLES {
            return Err(refused(format!(
                "the table has {} float columns and an imputation model reads at most {}; \
                 declare the auxiliary variables the model should read",
                float_ids.len(),
                multiple_imputation::MAX_IMPUTATION_VARIABLES
            )));
        }
        return Ok(float_ids);
    }
    let CausalQuery::AverageEffect(query) = &study.query else {
        return Err(refused("multiple imputation runs on average-effect queries"));
    };
    let mut ids = vec![query.treatment, query.outcome];
    for &id in declaration.covariates.iter() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    for &id in &ids {
        if !float_ids.contains(&id) {
            let name = data.schema().get(id).map_or_else(|_| "?".into(), |v| v.name.to_string());
            return Err(refused(format!(
                "variable `{name}` of the imputation model is not a float column"
            )));
        }
    }
    Ok(ids)
}

/// The imputer for `data` under the study's declaration.
fn build_imputer(
    study: &Study,
    declaration: &MissingDeclaration,
    spec: ImputationSpec,
    data: &TabularData,
) -> Result<(Imputer, Vec<VariableId>), CausalError> {
    let ids = model_variables(study, declaration, data)?;
    let mut columns = Vec::with_capacity(ids.len());
    for &id in &ids {
        let name = Arc::clone(&data.schema().get(id)?.name);
        let values: Arc<[f64]> = Arc::from(data.float64_slice(id)?);
        columns.push(ImputationColumn::new(name, values));
    }
    let units = multiple_imputation::bind_units(study.sampling_design, None, data.row_count())
        .map_err(|error| CausalError::Estimate(error.into()))?;
    debug_assert!(matches!(units, ImputationUnits::Rows));
    let imputer =
        Imputer::new(columns, spec, units).map_err(|error| CausalError::Estimate(error.into()))?;
    Ok((imputer, ids))
}

/// One completed table's fit, read from the route's result.
fn fit_of(result: &StudyResult, bayesian: bool) -> Result<ImputationFit, CausalError> {
    if bayesian {
        let posterior = result.posterior.as_ref().ok_or_else(|| {
            refused("the Bayesian fit of a completed table returned no posterior")
        })?;
        let column = posterior
            .effect_column()
            .ok_or_else(|| refused("the posterior of a completed table has no effect column"))?;
        let n = posterior.draws.n_draws;
        let draws = posterior.draws.values[column * n..(column + 1) * n].to_vec();
        return Ok(ImputationFit::bayesian(Arc::from(draws)));
    }
    let effect = result
        .estimate
        .as_effect()
        .ok_or_else(|| refused("the route's answer is not a scalar effect"))?;
    let analytic = &effect.uncertainty.analytic;
    let se = if analytic.se.is_finite() {
        analytic.se
    } else {
        effect.uncertainty.bootstrap.se.filter(|se| se.is_finite()).ok_or_else(|| {
            refused("the fit of a completed table reports no finite standard error to pool")
        })?
    };
    let df = analytic.reference_df.or_else(|| {
        let rows = effect.provenance.n_obs?;
        let estimator = result.logical_plan.estimator.as_deref()?;
        // A least-squares fit has n − (intercept + treatment + adjustment) residual degrees
        // of freedom; the other pooled estimators are referred to the normal law.
        (estimator == "linear.adjustment.ate").then(|| {
            let parameters = 2 + result.estimand.adjustment_set.len();
            rows as f64 - parameters as f64
        })
    });
    Ok(ImputationFit::frequentist(effect.ate, se * se, df.filter(|df| *df > 0.0)))
}

/// Stack the per-imputation posteriors column by column into one equal-weight mixture.
fn stacked_posterior(posteriors: &[CausalPosterior]) -> Result<CausalPosterior, CausalError> {
    let first = posteriors.first().ok_or_else(|| refused("no posterior to stack"))?;
    let n = first.draws.n_draws;
    let quantities = first.draws.schema.n_quantities();
    if posteriors
        .iter()
        .any(|p| p.draws.n_draws != n || p.draws.schema.n_quantities() != quantities)
    {
        return Err(refused(
            "the completed tables' posteriors differ in draws or quantities, so the mixture \
             weights would not be equal",
        ));
    }
    let m = posteriors.len();
    let mut values = Vec::with_capacity(quantities * n * m);
    for q in 0..quantities {
        for posterior in posteriors {
            values.extend_from_slice(&posterior.draws.values[q * n..(q + 1) * n]);
        }
    }
    let draws = PosteriorDraws::from_column_major(first.draws.schema.clone(), n * m, values)
        .map_err(|error| CausalError::Estimate(EstimationError::Prob(error)))?;
    let mut stacked = first.clone();
    stacked.summaries = draws.summarize();
    stacked.draws = draws;
    Ok(stacked)
}

/// Impute, fit with `click` on every completed table, and pool.
///
/// `click` runs the route's estimator on one completed table and returns its result.
///
/// # Errors
///
/// The imputation model's refusals, a fit's error, an estimator multiple imputation does
/// not pool on this route cell, or an analysed variable outside the imputation model that
/// is itself incomplete.
pub(crate) fn impute_fit_pool<F>(
    study: &Study,
    data: &TabularData,
    ctx: &ExecutionContext,
    mut click: F,
) -> Result<StudyResult, CausalError>
where
    F: FnMut(&TabularData, &ExecutionContext) -> Result<StudyResult, CausalError>,
{
    let (declaration, spec) =
        declared_imputation(study).ok_or_else(|| refused("no imputation is declared"))?;
    let cell = missing_data::route_cell(&study.query, &study.data, &study.inference);
    let row = route_missing_support(cell)
        .filter(|row| row.supports(declaration.policy))
        .ok_or_else(|| refused("this route cell does not execute multiple imputation"))?;
    let bayesian = study.inference.is_bayesian();
    let (imputer, ids) = build_imputer(study, declaration, spec, data)?;
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
        let replacements: Vec<(VariableId, Arc<[f64]>)> =
            imputed.iter().map(|&j| (ids[j], Arc::clone(&table.columns()[j]))).collect();
        let completed = complete_float_columns(data, &replacements)?;
        let result = click(&completed, ctx)?;
        let estimator = result.logical_plan.estimator.as_deref().unwrap_or("");
        if !row.imputes_with(estimator) {
            return Err(refused(format!(
                "multiple imputation pools the estimators {:?} on this route cell; the route ran \
                 `{estimator}`; next attempt: choose one of them or declare complete_case",
                row.imputation_estimators
            )));
        }
        let fit = fit_of(&result, bayesian)?;
        if bayesian {
            if let Some(posterior) = &result.posterior {
                posteriors.push(posterior.clone());
            }
        }
        if base.is_none() {
            check_analysed_variables(&result, data, &ids)?;
            base = Some(result);
        }
        Ok::<ImputationFit, CausalError>(fit)
    })
    .map_err(|failure| match failure {
        MultipleImputationFailure::Imputation(error) => CausalError::Estimate(error.into()),
        MultipleImputationFailure::Fit { error, .. } => error,
    })?;
    let mut result = base.ok_or_else(|| refused("no completed table was fitted"))?;
    let points: Arc<[f64]> = run.fits.iter().map(|fit| fit.point).collect();
    let estimator: Arc<str> = Arc::from(result.logical_plan.estimator.as_deref().unwrap_or(""));
    let report = match &run.pooled {
        PooledEstimate::Rubin(pooled) => {
            if let Some(effect) = result.estimate.as_effect_mut() {
                effect.ate = pooled.point;
                effect.uncertainty.analytic.se = pooled.standard_error();
                effect.uncertainty.analytic.kind = None;
                effect.uncertainty.analytic.reference_df =
                    pooled.df.is_finite().then_some(pooled.df);
                effect.uncertainty.bootstrap = antecedent_estimate::BootstrapAccounting::default();
            }
            ImputationReport {
                spec,
                unit: Arc::from(run.unit),
                units: run.units,
                imputed_cells: run.imputed_cells.clone().into(),
                estimator,
                pooling: Arc::from("rubin"),
                points,
                within: pooled.within,
                between: pooled.between,
                total: pooled.total,
                df: pooled.df.is_finite().then_some(pooled.df),
                df_complete: pooled.df_complete,
                fraction_missing_information: Some(pooled.fraction_missing_information),
                relative_increase: pooled
                    .relative_increase
                    .is_finite()
                    .then_some(pooled.relative_increase),
            }
        }
        PooledEstimate::Stacked(stacked) => {
            result.posterior = Some(stacked_posterior(&posteriors)?);
            if let Some(effect) = result.estimate.as_effect_mut() {
                effect.ate = stacked.mean;
                effect.uncertainty.analytic.se = stacked.variance.sqrt();
                effect.uncertainty.analytic.kind = None;
                effect.uncertainty.analytic.reference_df = None;
            }
            ImputationReport {
                spec,
                unit: Arc::from(run.unit),
                units: run.units,
                imputed_cells: run.imputed_cells.clone().into(),
                estimator,
                pooling: Arc::from("stacked_posterior"),
                points,
                within: stacked.within,
                between: stacked.between,
                total: stacked.variance,
                df: None,
                df_complete: None,
                fraction_missing_information: None,
                relative_increase: None,
            }
        }
        _ => return Err(refused("the pooling rule is not one this route reports")),
    };
    // The interval of the first completed table does not describe the pooled estimate, and
    // the pooled construction has no coverage record yet.
    result.interval = None;
    result.executed_contract = None;
    let variables: Vec<VariableId> = analysed_variables(&result);
    let mask = data.storage().analysis_mask();
    let accounting = RowAccounting::imputed(data, mask, declaration.policy, &variables)?;
    let covariates: Arc<[Arc<str>]> = declaration
        .covariates
        .iter()
        .filter_map(|&id| data.schema().get(id).ok().map(|v| Arc::clone(&v.name)))
        .collect();
    let mut missing = missing_data::report(MissingRoute::Static, true, covariates, accounting);
    missing.imputation = Some(report);
    result.missing_data = Some(missing);
    Ok(result)
}

/// The variables the result's estimand reads, in report order.
fn analysed_variables(result: &StudyResult) -> Vec<VariableId> {
    let mut variables: Vec<VariableId> = Vec::new();
    for id in result
        .treatment
        .into_iter()
        .chain(std::iter::once(result.outcome))
        .chain(result.estimand.adjustment_set.iter().copied())
    {
        if !variables.contains(&id) {
            variables.push(id);
        }
    }
    variables
}

/// Refuse when a variable the estimator reads is incomplete and outside the imputation
/// model: its estimator would drop those rows silently.
fn check_analysed_variables(
    result: &StudyResult,
    data: &TabularData,
    imputed: &[VariableId],
) -> Result<(), CausalError> {
    for id in analysed_variables(result) {
        if imputed.contains(&id) {
            continue;
        }
        let validity = data.column(id)?.validity();
        if !validity.is_all_valid() {
            let name = Arc::clone(&data.schema().get(id)?.name);
            return Err(refused(format!(
                "the estimator reads `{name}`, which has missing values and is not in the \
                 imputation model; declare it as an auxiliary variable of the imputation"
            )));
        }
    }
    Ok(())
}

/// Whether `study` declares multiple imputation on a tabular average-effect route.
pub(crate) fn imputes(study: &Study) -> bool {
    declared_imputation(study).is_some()
        && matches!(MissingRoute::of(&study.query, &study.data), Some(MissingRoute::Static))
        && matches!(study.query, CausalQuery::AverageEffect(_))
}

/// The policy name the claim evidence carries for an imputation report.
#[must_use]
pub(crate) fn evidence_entries(report: &ImputationReport) -> [String; 2] {
    [
        format!(
            "missing_imputation:{}:m={}:iterations={}:seed={}:{}",
            report.spec.model.name(),
            report.spec.imputations,
            report.spec.iterations,
            report.spec.seed,
            report.spec.assumption.name()
        ),
        format!(
            "missing_imputation_pool:{}:within={:e}:between={:e}",
            report.pooling, report.within, report.between
        ),
    ]
}
