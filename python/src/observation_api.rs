//! Python boundary for explicit outcome-observation mechanisms.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalQuery, ContinuousDomain, GridSpec, IdentificationStatus,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportStatus, VariableId,
};
use antecedent_data::TableView;
use antecedent_estimate::{
    ContinuousResponseEstimator, ObservationEstimatorOptions, ObservationMechanismEstimator,
    SelectedOutcomeCorrection,
};
use antecedent_identify::{IdentificationWorkspace, ResponseIdentifier};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalIdentifyError, columns_to_batch, dag_from_named_edges, detach_catch, py_err};

#[pyclass(skip_from_py_object)]
pub(crate) struct ObservationAdjustedOutcomeResult {
    #[pyo3(get)]
    values: Vec<f64>,
    #[pyo3(get)]
    weights: Vec<f64>,
    #[pyo3(get)]
    method: String,
}

#[pyclass(skip_from_py_object)]
pub(crate) struct ObservationResponseResult {
    #[pyo3(get)]
    treatments: Vec<String>,
    #[pyo3(get)]
    outcomes: Vec<String>,
    #[pyo3(get)]
    points: Vec<Vec<f64>>,
    #[pyo3(get)]
    values: Vec<Vec<f64>>,
    #[pyo3(get)]
    scalar: Option<f64>,
    #[pyo3(get)]
    matrix: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    uncertainty_kind: String,
    #[pyo3(get)]
    lower: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    upper: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    level: Option<f64>,
    #[pyo3(get)]
    standard_error: Option<f64>,
    #[pyo3(get)]
    replicates: Option<u32>,
    #[pyo3(get)]
    artifact_id: Option<String>,
    #[pyo3(get)]
    support_status: String,
    #[pyo3(get)]
    support_minima: Vec<f64>,
    #[pyo3(get)]
    support_maxima: Vec<f64>,
    #[pyo3(get)]
    diagnostic_ids: Vec<String>,
    #[pyo3(get)]
    diagnostic_values: Vec<Vec<f64>>,
    #[pyo3(get)]
    diagnostic_details: Vec<String>,
    #[pyo3(get)]
    warnings: Vec<String>,
    #[pyo3(get)]
    identification: String,
    #[pyo3(get)]
    adjustment_set: Vec<String>,
    #[pyo3(get)]
    assumptions: Vec<String>,
    #[pyo3(get)]
    provenance_id: String,
}

#[derive(Clone)]
struct ObservationArgs {
    kind: String,
    latent: String,
    observed: Option<String>,
    censoring: Option<String>,
    event: Option<String>,
    lower: Option<String>,
    upper: Option<String>,
    indicator: Option<String>,
    assumption_kind: String,
    assumption_variables: Vec<String>,
    structural_model: Option<String>,
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, treatment, outcome, observation_kind, latent, *, observed=None,
    censoring=None, event=None, lower=None, upper=None, indicator=None,
    assumption_kind, assumption_variables=Vec::new(), structural_model=None,
    delayed_entry=None, correction="aipw", observation_probability_floor=0.01,
    censoring_survival_floor=0.01, crossfit_folds=5
))]
#[allow(clippy::too_many_arguments)]
fn observation_adjusted_outcome(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    outcome: String,
    observation_kind: String,
    latent: String,
    observed: Option<String>,
    censoring: Option<String>,
    event: Option<String>,
    lower: Option<String>,
    upper: Option<String>,
    indicator: Option<String>,
    assumption_kind: String,
    assumption_variables: Vec<String>,
    structural_model: Option<String>,
    delayed_entry: Option<String>,
    correction: &str,
    observation_probability_floor: f64,
    censoring_survival_floor: f64,
    crossfit_folds: usize,
) -> PyResult<ObservationAdjustedOutcomeResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let correction = parse_correction(correction)?;
    let args = ObservationArgs {
        kind: observation_kind,
        latent,
        observed,
        censoring,
        event,
        lower,
        upper,
        indicator,
        assumption_kind,
        assumption_variables,
        structural_model,
    };
    detach_catch(py, move || {
        let loaded = antecedent_data::tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let query =
            build_query(data.schema(), &treatment, &outcome, &args, Arc::from([0.0_f64, 1.0_f64]))?;
        let delayed_entry = delayed_entry
            .as_deref()
            .map(|name| data.schema().id_of(name).map_err(py_err))
            .transpose()?;
        let estimator = ObservationMechanismEstimator::new(ObservationEstimatorOptions {
            selected_correction: correction,
            observation_probability_floor,
            censoring_survival_floor,
            crossfit_folds,
        });
        let adjusted = estimator.adjusted_outcome(&data, &query, delayed_entry).map_err(py_err)?;
        Ok(ObservationAdjustedOutcomeResult {
            values: adjusted.values,
            weights: adjusted.weights,
            method: adjusted.method.to_string(),
        })
    })
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, treatment, outcome, grid, observation_kind, latent, *,
    observed=None, censoring=None, event=None, lower=None, upper=None, indicator=None,
    assumption_kind, assumption_variables=Vec::new(), structural_model=None,
    delayed_entry=None, correction="aipw", observation_probability_floor=0.01,
    censoring_survival_floor=0.01, crossfit_folds=5
))]
#[allow(clippy::too_many_arguments)]
fn analyze_observation_response(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    grid: Vec<f64>,
    observation_kind: String,
    latent: String,
    observed: Option<String>,
    censoring: Option<String>,
    event: Option<String>,
    lower: Option<String>,
    upper: Option<String>,
    indicator: Option<String>,
    assumption_kind: String,
    assumption_variables: Vec<String>,
    structural_model: Option<String>,
    delayed_entry: Option<String>,
    correction: &str,
    observation_probability_floor: f64,
    censoring_survival_floor: f64,
    crossfit_folds: usize,
) -> PyResult<ObservationResponseResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let correction = parse_correction(correction)?;
    let args = ObservationArgs {
        kind: observation_kind,
        latent,
        observed,
        censoring,
        event,
        lower,
        upper,
        indicator,
        assumption_kind,
        assumption_variables,
        structural_model,
    };
    detach_catch(py, move || {
        let loaded = antecedent_data::tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let query = build_query(data.schema(), &treatment, &outcome, &args, Arc::from(grid))?;
        let identifier = ResponseIdentifier::new();
        let prepared = identifier
            .prepare_with_assumptions(&dag, AssumptionSet::new())
            .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
        let identification = identifier
            .identify(
                &prepared,
                &CausalQuery::Response(query.clone()),
                &mut IdentificationWorkspace::default(),
            )
            .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
        if identification.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(PyValueError::new_err(
                "observation-adjusted response is not identified by backdoor adjustment",
            ));
        }
        let first = identification
            .estimands
            .first()
            .ok_or_else(|| PyValueError::new_err("response identification returned no estimand"))?;
        if identification
            .estimands
            .iter()
            .any(|estimand| estimand.adjustment_set != first.adjustment_set)
        {
            return Err(PyValueError::new_err(
                "response estimation currently requires one common adjustment set",
            ));
        }
        let adjustment_set = first
            .adjustment_set
            .iter()
            .map(|id| {
                names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
            })
            .collect();
        let delayed_entry = delayed_entry
            .as_deref()
            .map(|name| data.schema().id_of(name).map_err(py_err))
            .transpose()?;
        let observation_estimator =
            ObservationMechanismEstimator::new(ObservationEstimatorOptions {
                selected_correction: correction,
                observation_probability_floor,
                censoring_survival_floor,
                crossfit_folds,
            });
        let response = observation_estimator
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::clone(&first.adjustment_set)),
                &data,
                &query,
                delayed_entry,
                identification.status,
                identification.required_assumptions,
            )
            .map_err(py_err)?;
        observation_response_result(response, treatment, outcome, adjustment_set)
    })
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, treatment, outcome, observation_kind, latent, means, sigma, *,
    observed=None, censoring=None, event=None, lower=None, upper=None, indicator=None,
    assumption_kind, assumption_variables=Vec::new(), structural_model=None
))]
#[allow(clippy::too_many_arguments)]
fn gaussian_observation_log_likelihood(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    outcome: String,
    observation_kind: String,
    latent: String,
    means: Vec<f64>,
    sigma: f64,
    observed: Option<String>,
    censoring: Option<String>,
    event: Option<String>,
    lower: Option<String>,
    upper: Option<String>,
    indicator: Option<String>,
    assumption_kind: String,
    assumption_variables: Vec<String>,
    structural_model: Option<String>,
) -> PyResult<f64> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let args = ObservationArgs {
        kind: observation_kind,
        latent,
        observed,
        censoring,
        event,
        lower,
        upper,
        indicator,
        assumption_kind,
        assumption_variables,
        structural_model,
    };
    detach_catch(py, move || {
        let loaded = antecedent_data::tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let query =
            build_query(data.schema(), &treatment, &outcome, &args, Arc::from([0.0_f64, 1.0_f64]))?;
        ObservationMechanismEstimator::default()
            .gaussian_log_likelihood(&data, &query, &means, sigma)
            .map_err(py_err)
    })
}

fn build_query(
    schema: &antecedent_core::CausalSchema,
    treatment: &str,
    outcome: &str,
    args: &ObservationArgs,
    grid: Arc<[f64]>,
) -> PyResult<ResponseQuery> {
    if outcome != args.latent {
        return Err(PyValueError::new_err(
            "response query outcome must equal the observation mechanism's latent outcome",
        ));
    }
    let treatment = schema.id_of(treatment).map_err(py_err)?;
    let latent = schema.id_of(&args.latent).map_err(py_err)?;
    if treatment == latent {
        return Err(PyValueError::new_err("treatment and outcome must be distinct"));
    }
    let functional = ResponseFunctional::MeanCurve {
        outcome: latent,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(grid)),
    };
    let observation = build_observation(schema, args, latent)?;
    let assumption = build_assumption(schema, args)?;
    Ok(ResponseQuery::new(functional).with_observation(observation, [assumption]))
}

fn observation_response_result(
    response: antecedent_core::CausalResponse,
    treatment: String,
    outcome: String,
    adjustment_set: Vec<String>,
) -> PyResult<ObservationResponseResult> {
    let ResponseIdentification::PointIdentified(ResponseValue::Surface {
        grid,
        dimension: 1,
        mean,
    }) = response.estimate
    else {
        return Err(PyValueError::new_err(
            "observation-adjusted MeanCurve returned an invalid response shape",
        ));
    };
    if grid.len() != mean.len() {
        return Err(PyValueError::new_err(
            "observation-adjusted MeanCurve grid/value lengths differ",
        ));
    }
    if response.uncertainty != ResponseUncertainty::None {
        return Err(PyValueError::new_err(
            "observation-adjusted response unexpectedly carried unlicensed uncertainty",
        ));
    }
    let support_status = match response.support.status {
        SupportStatus::Supported => "supported",
        SupportStatus::WeakOverlap => "weak_overlap",
        SupportStatus::Extrapolative => "extrapolative",
        SupportStatus::OutsideEmpiricalSupport => "outside_empirical_support",
    };
    Ok(ObservationResponseResult {
        treatments: vec![treatment],
        outcomes: vec![outcome],
        points: grid.iter().map(|value| vec![*value]).collect(),
        values: mean.iter().map(|value| vec![*value]).collect(),
        scalar: None,
        matrix: None,
        uncertainty_kind: "none".into(),
        lower: None,
        upper: None,
        level: None,
        standard_error: None,
        replicates: None,
        artifact_id: None,
        support_status: support_status.into(),
        support_minima: response.support.query_region.minima.to_vec(),
        support_maxima: response.support.query_region.maxima.to_vec(),
        diagnostic_ids: response
            .support
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.id.to_string())
            .collect(),
        diagnostic_values: response
            .support
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.values.to_vec())
            .collect(),
        diagnostic_details: response
            .support
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.detail.to_string())
            .collect(),
        warnings: response
            .support
            .warnings
            .iter()
            .map(|warning| format!("{}: {}", warning.code, warning.message))
            .collect(),
        identification: format!("{:?}", response.identification_status),
        adjustment_set,
        assumptions: response
            .assumptions
            .entries
            .iter()
            .map(|record| format!("{:?}", record.assumption))
            .collect(),
        provenance_id: response.provenance_id.to_string(),
    })
}

fn build_observation(
    schema: &antecedent_core::CausalSchema,
    args: &ObservationArgs,
    latent: VariableId,
) -> PyResult<ObservationSpec> {
    let required = |value: &Option<String>, label: &str| -> PyResult<VariableId> {
        let name = value
            .as_deref()
            .ok_or_else(|| PyValueError::new_err(format!("{label} is required")))?;
        schema.id_of(name).map_err(py_err)
    };
    let optional = |value: &Option<String>| -> PyResult<Option<VariableId>> {
        value.as_deref().map(|name| schema.id_of(name).map_err(py_err)).transpose()
    };
    match args.kind.as_str() {
        "complete" => Ok(ObservationSpec::Complete),
        "right_censored" => Ok(ObservationSpec::RightCensored {
            latent,
            observed: required(&args.observed, "observed")?,
            censoring: required(&args.censoring, "censoring")?,
            event: required(&args.event, "event")?,
        }),
        "left_censored" => Ok(ObservationSpec::LeftCensored {
            latent,
            observed: required(&args.observed, "observed")?,
            censoring: required(&args.censoring, "censoring")?,
            event: required(&args.event, "event")?,
        }),
        "interval_censored" => Ok(ObservationSpec::IntervalCensored {
            latent,
            lower: required(&args.lower, "lower")?,
            upper: required(&args.upper, "upper")?,
        }),
        "truncated" => Ok(ObservationSpec::Truncated {
            latent,
            observed: required(&args.observed, "observed")?,
            lower: optional(&args.lower)?,
            upper: optional(&args.upper)?,
        }),
        "selected" => Ok(ObservationSpec::Selected {
            latent,
            observed: required(&args.observed, "observed")?,
            indicator: required(&args.indicator, "indicator")?,
        }),
        _ => Err(PyValueError::new_err("unknown observation mechanism")),
    }
}

fn build_assumption(
    schema: &antecedent_core::CausalSchema,
    args: &ObservationArgs,
) -> PyResult<ObservationAssumption> {
    let variables = || {
        args.assumption_variables
            .iter()
            .map(|name| schema.id_of(name).map_err(py_err))
            .collect::<PyResult<Vec<_>>>()
            .map(Arc::from)
    };
    match args.assumption_kind.as_str() {
        "independent_given" => Ok(ObservationAssumption::IndependentGiven(variables()?)),
        "outcome_independent_given" => {
            Ok(ObservationAssumption::OutcomeIndependentGiven(variables()?))
        }
        "structural" => Ok(ObservationAssumption::Structural(Arc::from(
            args.structural_model
                .as_deref()
                .ok_or_else(|| PyValueError::new_err("structural_model is required"))?,
        ))),
        _ => Err(PyValueError::new_err("unknown observation assumption")),
    }
}

fn parse_correction(value: &str) -> PyResult<SelectedOutcomeCorrection> {
    match value {
        "ipw" => Ok(SelectedOutcomeCorrection::Ipw),
        "aipw" => Ok(SelectedOutcomeCorrection::Aipw),
        _ => Err(PyValueError::new_err("correction must be 'ipw' or 'aipw'")),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ObservationAdjustedOutcomeResult>()?;
    m.add_class::<ObservationResponseResult>()?;
    m.add_function(wrap_pyfunction!(observation_adjusted_outcome, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_observation_response, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_observation_log_likelihood, m)?)?;
    Ok(())
}
