//! Time-series model construction and impute-fit-pool execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, Collected, ExecutionContext, ImputationColumn, ImputationSpec,
    Imputer, MissingDeclaration, MissingRoute, OwnedColumn, Pooling, RowAccounting, Study,
    StudyResult, TableView, TabularData, TimeSeriesData, VariableId, add, analysed_variables,
    complete_float_columns, declared_imputation, failure, missing_data, multiple_imputation,
    pool_into_result, query_variables, refused, route_missing_support,
};

/// Neighbouring steps on each side the series' imputation model conditions on: the deepest lag
/// the query reads. A series too short to identify a model that wide is refused by the imputer.
fn series_lags(query: &CausalQuery) -> Result<usize, CausalError> {
    let wanted = match query {
        CausalQuery::TemporalEffect(q) => {
            q.horizon_steps.max(q.max_history_lag.unwrap_or(1)).max(1)
        }
        CausalQuery::Mediation(q) => q.horizons.iter().copied().max().unwrap_or(1).max(1),
        _ => 1,
    };
    usize::try_from(wanted).map_err(|_| refused("the query reads more history than fits"))
}

/// Contiguous block length of the series' parameter draw: the cube root of the length, and at
/// least the span of one conditioning neighbourhood.
fn series_block(rows: usize, lags: usize) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a cube root of a row count is a small positive integer"
    )]
    let cube_root = (rows as f64).cbrt().ceil() as usize;
    cube_root.max(2 * lags + 1)
}

fn series_float_ids(series: &TimeSeriesData) -> Vec<VariableId> {
    series
        .storage()
        .columns()
        .iter()
        .filter_map(|column| match column {
            OwnedColumn::Float64(float) => Some(float.id),
            _ => None,
        })
        .collect()
}

fn series_model_variables(
    study: &Study,
    declaration: &MissingDeclaration,
    series: &TimeSeriesData,
) -> Result<Vec<VariableId>, CausalError> {
    let float_ids = series_float_ids(series);
    if declaration.covariates.is_empty() {
        if float_ids.len() > multiple_imputation::MAX_IMPUTATION_VARIABLES {
            return Err(refused(format!(
                "the series has {} float columns and an imputation model reads at most {}; \
                 declare the auxiliary variables the model should read",
                float_ids.len(),
                multiple_imputation::MAX_IMPUTATION_VARIABLES
            )));
        }
        return Ok(float_ids);
    }
    let mut ids = query_variables(&study.query);
    if ids.is_empty() {
        return Err(refused("this query names no variable the imputation model can anchor on"));
    }
    for &id in declaration.covariates.iter() {
        add(&mut ids, id);
    }
    for &id in &ids {
        if !float_ids.contains(&id) {
            let name = series.schema().get(id).map_or_else(|_| "?".into(), |v| v.name.to_string());
            return Err(refused(format!(
                "variable `{name}` of the imputation model is not a float column"
            )));
        }
    }
    Ok(ids)
}

/// The series with the imputed columns replaced by their completed values.
fn complete_series(
    series: &TimeSeriesData,
    replacements: &[(VariableId, Arc<[f64]>)],
) -> Result<TimeSeriesData, CausalError> {
    let table = TabularData::new(series.storage().clone());
    let completed = complete_float_columns(&table, replacements)?;
    Ok(TimeSeriesData::try_new(completed.storage().clone(), series.time_index().clone())?)
}

/// Refuse when a variable the estimator reads is incomplete and outside the imputation model.
fn check_series_variables(
    result: &StudyResult,
    series: &TimeSeriesData,
    model: &[VariableId],
) -> Result<(), CausalError> {
    for id in analysed_variables(result) {
        if model.contains(&id) {
            continue;
        }
        if !series.column(id)?.validity().is_all_valid() {
            let name = Arc::clone(&series.schema().get(id)?.name);
            return Err(refused(format!(
                "the estimator reads `{name}`, which has missing values and is not in the \
                 imputation model; declare it as an auxiliary variable of the imputation"
            )));
        }
    }
    Ok(())
}

/// The lagged imputer for `series` under the study's declaration, the model's variables and
/// the neighbouring steps it conditions on.
pub(super) fn series_imputer(
    study: &Study,
    declaration: &MissingDeclaration,
    spec: ImputationSpec,
    series: &TimeSeriesData,
) -> Result<(Imputer, Vec<VariableId>, usize), CausalError> {
    let lags = series_lags(&study.query)?;
    let ids = series_model_variables(study, declaration, series)?;
    let mut columns = Vec::with_capacity(ids.len());
    for &id in &ids {
        let name = Arc::clone(&series.schema().get(id)?.name);
        columns.push(ImputationColumn::new(name, Arc::<[f64]>::from(series.float64_values(id)?)));
    }
    let rows = series.row_count();
    let units = multiple_imputation::bind_series_units(
        study.declared.sampling_design,
        rows,
        series_block(rows, lags),
    )
    .map_err(|error| CausalError::Estimate(error.into()))?;
    let imputer = Imputer::new_with_domains(columns, spec, units, &declaration.imputation_domains)
        .and_then(|imputer| imputer.with_dynamics(lags))
        .map_err(|error| CausalError::Estimate(error.into()))?;
    Ok((imputer, ids, lags))
}

/// Impute a series by a lagged model, fit with `click` on every completed series, and pool.
///
/// # Errors
///
/// As [`super::impute_fit_pool_vector`], plus a sampling design other than the dependent time series
/// and a query that reads more history than the lagged model conditions on.
#[allow(
    clippy::too_many_lines,
    reason = "impute, fit and pool share the run's borrowed state in one sequence"
)]
pub(crate) fn impute_fit_pool_series<F>(
    study: &Study,
    series: &TimeSeriesData,
    ctx: &ExecutionContext,
    mut click: F,
) -> Result<StudyResult, CausalError>
where
    F: FnMut(&TimeSeriesData, &ExecutionContext) -> Result<StudyResult, CausalError>,
{
    let (declaration, spec) =
        declared_imputation(study).ok_or_else(|| refused("no imputation is declared"))?;
    let cell = missing_data::route_cell(&study.query, &study.data, &study.settings.inference);
    let row = route_missing_support(cell)
        .filter(|row| row.supports(declaration.policy))
        .ok_or_else(|| refused("this route cell does not execute multiple imputation"))?;
    let bayesian = study.settings.inference.is_bayesian();
    let (imputer, ids, lags) = series_imputer(study, declaration, spec, series)?;
    let imputed: Vec<usize> = imputer
        .variables()
        .iter()
        .enumerate()
        .filter(|(_, v)| v.missing > 0)
        .map(|(j, _)| j)
        .collect();
    let mut collected = Collected::default();
    let run = multiple_imputation::run_vector(&imputer, ctx, |table, ctx| {
        let replacements: Vec<(VariableId, Arc<[f64]>)> =
            imputed.iter().map(|&j| (ids[j], Arc::clone(&table.columns()[j]))).collect();
        let completed = complete_series(series, &replacements)?;
        let result = click(&completed, ctx)?;
        let estimator = result.logical_plan.estimator.as_deref().unwrap_or("");
        if !row.imputes_with(estimator) {
            return Err(refused(format!(
                "multiple imputation pools the estimators {:?} on this route cell; the route ran \
                 `{estimator}`; next attempt: choose one of them or declare complete_case",
                row.imputation_estimators
            )));
        }
        collected.record(result, bayesian, |first| check_series_variables(first, series, &ids))
    })
    .map_err(failure)?;
    let mut assumptions = run.assumptions.clone();
    assumptions.design = study.declared.sampling_design;
    let estimator: Arc<str> = Arc::from(
        collected
            .base
            .as_ref()
            .and_then(|base| base.logical_plan.estimator.as_deref())
            .unwrap_or(""),
    );
    let meta = Pooling {
        unit: Arc::from(run.assumptions.unit),
        units: run.assumptions.units,
        imputed_cells: imputer.imputed_cells().into(),
        estimator,
        lags,
        spec,
        digest: Arc::from(assumptions.digest()),
        imputation_domains: assumptions.domains.clone().into(),
        binary_boundary_variables: assumptions.binary_boundary_variables.clone().into(),
    };
    let (mut result, report) = pool_into_result(study, &run, collected, &meta)?;
    let variables: Vec<VariableId> = ids.clone();
    let accounting = RowAccounting::imputed(
        series,
        series.storage().analysis_mask(),
        declaration.policy,
        &variables,
    )?;
    let covariates: Arc<[Arc<str>]> = declaration
        .covariates
        .iter()
        .filter_map(|&id| series.schema().get(id).ok().map(|v| Arc::clone(&v.name)))
        .collect();
    let mut missing = missing_data::report(MissingRoute::Temporal, true, covariates, accounting);
    missing.imputation = Some(report);
    result.extras.missing_data = Some(missing);
    crate::analysis::target_validation::reset_pooled_refutations(&mut result);
    Ok(result)
}
