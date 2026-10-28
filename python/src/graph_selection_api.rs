//! Python bridge of multiple imputation with the graph selection re-run on every completed
//! table.
//!
//! Python states the study declaratively: the data columns (missing cells as NaN), the
//! treatment and outcome columns, the Bayesian g-computation settings and the imputation
//! specification. The structure search is the exact DAG posterior over the table's columns, in
//! column order. Rust builds the effect study on each completed table and its own graph
//! posterior, runs it, and stacks the effect posteriors; the adjustment set is whatever each
//! sampled graph identifies, never a Python declaration. A refusal comes back as structured
//! JSON for the Python layer to raise as its own typed exception.
use crate::bridge::{split_detail, RefusalJson};
use crate::detach_catch;
use antecedent::discovery::BayesianDiscoverParams;
use antecedent::{
    estimate_with_graph_selection_per_imputation, BayesianConfig, CausalError,
    GraphSelectionPerImputation, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{reason_code, AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::{ImputationModel, ImputationSpec, TabularData};
use pyo3::prelude::*;
use serde_json::json;

fn refusal(code: &str, detail: &str, message: &str) -> String {
    RefusalJson::new(code, "graph_selection_imputation", detail, message).json()
}

fn error_refusal(error: &CausalError) -> String {
    let text = error.to_string();
    let (detail, message) = split_detail(&text);
    let code = error.reason_code().unwrap_or(reason_code!("invalid_argument"));
    refusal(code, detail, message)
}

/// The `level` central interval of the effect draws of a posterior.
fn central_interval(draws: &[f64], level: f64) -> (f64, f64) {
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len().saturating_sub(1);
    let tail = (1.0 - level) / 2.0;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    let at = |p: f64| sorted[((p * last as f64).round() as usize).min(last)];
    (at(tail), at(1.0 - tail))
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, treatment, outcome, *, model="chained_equations".to_owned(), imputations=20,
    iterations=10, seed=1, n_draws=1000, prior_scale=10.0, level=0.95
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn graph_selection_imputation_run(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    treatment: String,
    outcome: String,
    model: String,
    imputations: u32,
    iterations: u32,
    seed: u64,
    n_draws: usize,
    prior_scale: f64,
    level: f64,
) -> PyResult<(Option<String>, Option<String>)> {
    detach_catch(py, move || {
        let invalid = |detail: &str, message: String| {
            Ok((None, Some(refusal(reason_code!("invalid_argument"), detail, &message))))
        };
        if names.len() != columns.len() || names.is_empty() {
            return invalid(
                "graph_selection_imputation.columns",
                "names and columns must be non-empty and of equal length".to_owned(),
            );
        }
        let position = |name: &str| names.iter().position(|n| n == name);
        let (Some(t), Some(y)) = (position(&treatment), position(&outcome)) else {
            return invalid(
                "graph_selection_imputation.variables",
                format!("treatment `{treatment}` and outcome `{outcome}` must be data columns"),
            );
        };
        let Some(model) = ImputationModel::from_name(&model) else {
            return invalid(
                "graph_selection_imputation.model",
                format!("unknown imputation model `{model}`"),
            );
        };
        if !(level > 0.0 && level < 1.0) {
            return invalid(
                "graph_selection_imputation.level",
                format!("level must lie in (0, 1), got {level}"),
            );
        }
        let table = match TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, column)| (name.as_str(), column.as_slice())),
        ) {
            Ok(table) => table,
            Err(error) => return invalid("graph_selection_imputation.data", error.to_string()),
        };
        let raw = |i: usize| VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        let variables: Vec<VariableId> = (0..names.len()).map(raw).collect();
        let query = AverageEffectQuery::binary_ate(raw(t), raw(y));
        let spec = ImputationSpec::new()
            .with_model(model)
            .with_imputations(imputations)
            .with_iterations(iterations)
            .with_seed(seed);
        let selection =
            GraphSelectionPerImputation::new(BayesianDiscoverParams::default(), variables);
        let ctx = ExecutionContext::production_default(seed);
        let result = estimate_with_graph_selection_per_imputation(
            &table,
            spec,
            &selection,
            &ctx,
            |completed, posterior| {
                Study::tabular(completed.clone())
                    .graph_posterior(posterior)
                    .query(query.clone())
                    .inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale),
                    ))
                    .refute(RefuteSuite::None)
            },
        );
        let result = match result {
            Ok(result) => result,
            Err(error) => return Ok((None, Some(error_refusal(&error)))),
        };
        let Some(effect) = result.estimate().as_effect() else {
            return invalid(
                "graph_selection_imputation.result",
                "the pooled answer is not a scalar effect".to_owned(),
            );
        };
        let Some(report) = result.missing_data().as_ref().and_then(|m| m.imputation.as_ref())
        else {
            return invalid(
                "graph_selection_imputation.result",
                "the pooled result carries no imputation report".to_owned(),
            );
        };
        let interval = result.posterior.as_ref().and_then(|posterior| {
            let column = posterior.effect_column()?;
            let n = posterior.draws.n_draws;
            Some(central_interval(&posterior.draws.values[column * n..(column + 1) * n], level))
        });
        let payload = json!({
            "effect": effect.ate,
            "posterior_sd": effect.uncertainty.analytic.se,
            "credible_interval": interval.map(|(lower, upper)| json!({
                "level": level, "lower": lower, "upper": upper,
            })),
            "posterior_draws": result.posterior.as_ref().map(|p| p.draws.n_draws),
            "imputations": report.spec.imputations,
            "points": report.points.as_ref(),
            "within": report.within,
            "between": report.between,
            "total": report.total,
            "pooling": report.pooling.as_ref(),
            "estimator": report.estimator.as_ref(),
            "imputed_cells": report
                .imputed_cells
                .iter()
                .map(|(name, cells)| json!({"variable": name.as_ref(), "cells": cells}))
                .collect::<Vec<_>>(),
            "assumptions_digest": report.assumptions_digest.as_ref(),
            "assumption": report.spec.assumption.name(),
            "structure": "graph_selection_rerun_per_imputation",
        });
        Ok((Some(payload.to_string()), None))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(graph_selection_imputation_run, module)?)?;
    Ok(())
}
