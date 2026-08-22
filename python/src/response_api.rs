//! Python execution boundary for continuous causal-response queries.

use std::sync::Arc;

use antecedent::support::{StructureSource, refuse_if_not_applicable, support_cell};
use antecedent::{AcceptedGraph, GraphClass, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, DerivativeScale, DerivativeWeighting, GridSpec,
    IdentificationStatus, Intervention, MechanismOverride, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, StochasticPolicy,
    SupportStatus, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TableView;
use antecedent_estimate::ContinuousResponseEstimator;
use antecedent_identify::{
    GeneralizedAdjustmentIdentifier, IdentificationWorkspace, ResponseIdentifier,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    CausalIdentifyError, dag_from_named_edges, detach_catch, graphs, py_err, py_execution_context,
    series_from_tabular, suite_from_refute, temporal_dag_from_schema_edges,
};

#[pyclass(skip_from_py_object)]
pub(crate) struct ResponseAnalysisResult {
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
    support_point_status: Vec<String>,
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
    horizon_adjustment_sets: Vec<Vec<String>>,
    #[pyo3(get)]
    assumptions: Vec<String>,
    #[pyo3(get)]
    provenance_id: String,
    #[pyo3(get)]
    identified_mass: Option<f64>,
    #[pyo3(get)]
    unidentified_mass: Option<f64>,
    #[pyo3(get)]
    completion_count: Option<usize>,
    #[pyo3(get)]
    truncated_completions: Option<usize>,
    #[pyo3(get)]
    enumeration_capped: Option<bool>,
    #[pyo3(get)]
    mass_scope: Option<String>,
    /// Matrix evidence contract (`licensed` / `allowed_unlicensed`). Distinct from
    /// empirical dose-range [`Self::support_status`].
    #[pyo3(get)]
    evidence_status: Option<String>,
    #[pyo3(get)]
    allowlist_reason: Option<String>,
    #[pyo3(get)]
    allowlist_parent: Option<String>,
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, kind, treatments, outcomes, *, grid=None, at=None,
    direction=None, intervention_kinds=None, intervention_parameters=None,
    order=1, scale="identity", weighting="observed", bandwidth=None,
    simultaneous_replicates=None, confidence_level=0.95,
    multiplier_seed=0xA17E_CEDE_0500, export_row_diagnostics=false, accepted=false,
    refute=None
))]
#[allow(clippy::too_many_arguments)]
fn analyze_response(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    grid: Option<Vec<f64>>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    intervention_kinds: Option<Vec<String>>,
    intervention_parameters: Option<Vec<Vec<f64>>>,
    order: u8,
    scale: &str,
    weighting: &str,
    bandwidth: Option<f64>,
    simultaneous_replicates: Option<u32>,
    confidence_level: f64,
    multiplier_seed: u64,
    export_row_diagnostics: bool,
    accepted: bool,
    refute: Option<Bound<'_, PyAny>>,
) -> PyResult<ResponseAnalysisResult> {
    let suite = match refute.as_ref() {
        None => RefuteSuite::None,
        Some(obj) => suite_from_refute(Some(obj))?,
    };
    let (data, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let scale = parse_scale(scale)?;
    let weighting = parse_weighting(weighting)?;
    detach_catch(py, move || {
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let treatment_ids = resolve_names(data.schema(), &treatments)?;
        let outcome_ids = resolve_names(data.schema(), &outcomes)?;
        let functional = build_functional(
            &kind,
            &treatment_ids,
            &outcome_ids,
            grid,
            at,
            direction,
            intervention_kinds,
            intervention_parameters,
            order,
            scale,
            weighting,
        )?;
        let query = ResponseQuery::new(functional);
        // `analyze_response` never routes through `StudyBuilder`, so it must
        // consult the matrix itself. Pass the caller's `refute=` suite: cheap/full
        // on a function-valued estimand are n/a (`parity/support_n_a.toml`).
        let structure =
            if accepted { StructureSource::Accepted } else { StructureSource::Explicit };
        let causal_query = CausalQuery::Response(query.clone());
        let evidence = if let Some(cell) = support_cell(
            &causal_query,
            GraphClass::Dag,
            structure,
            &InferenceMode::Frequentist,
            suite,
        ) {
            Some(refuse_if_not_applicable(cell).map_err(py_err)?)
        } else {
            None
        };
        let identifier = ResponseIdentifier::new();
        let prepared = identifier
            .prepare_with_assumptions(&dag, AssumptionSet::new())
            .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
        let identification = identifier
            .identify(&prepared, &causal_query, &mut IdentificationWorkspace::default())
            .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
        if identification.status != IdentificationStatus::NonparametricallyIdentified {
            return Err(PyValueError::new_err(
                "continuous response is not identified by backdoor adjustment",
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
                "multi-response estimation currently requires one common adjustment set",
            ));
        }
        let adjustment_set = first
            .adjustment_set
            .iter()
            .map(|id| {
                names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
            })
            .collect();
        let mut estimator = ContinuousResponseEstimator::new(Arc::clone(&first.adjustment_set));
        estimator.options.bandwidth = bandwidth;
        estimator.options.simultaneous_replicates = simultaneous_replicates;
        estimator.options.confidence_level = confidence_level;
        estimator.options.multiplier_seed = multiplier_seed;
        estimator.options.export_row_diagnostics = export_row_diagnostics;
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                identification.status,
                identification.required_assumptions,
            )
            .map_err(py_err)?;
        response_result(response, treatments, outcomes, adjustment_set, &names, evidence)
    })
}

/// Mean response curve over the generalized-adjustment envelope of a PAG.
///
/// Each identified MAG completion is estimated on the caller's shared grid with its
/// completion-specific adjustment set. The pointwise min/max across those curves is the
/// reported identified envelope; completions that do not identify remain explicit mass.
#[pyfunction]
#[pyo3(signature = (names, columns, graph, treatment, outcome, grid, *, max_completions=32))]
#[allow(clippy::too_many_arguments)]
fn analyze_response_pag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    graph: graphs::Pag,
    treatment: String,
    outcome: String,
    grid: Vec<f64>,
    max_completions: usize,
) -> PyResult<ResponseAnalysisResult> {
    let (data, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    detach_catch(py, move || {
        let treatment_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let outcome_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let functional = ResponseFunctional::MeanCurve {
            outcome: outcome_id,
            treatment: antecedent_core::ContinuousDomain::new(
                treatment_id,
                GridSpec::Values(Arc::from(grid)),
            ),
        };
        let query = ResponseQuery::new(functional);
        query.validate().map_err(|error| PyValueError::new_err(error.to_string()))?;
        let witness = AverageEffectQuery::binary_ate(treatment_id, outcome_id);
        let identifier = GeneralizedAdjustmentIdentifier {
            config: antecedent_identify::GeneralizedAdjustmentConfig {
                max_completions,
                ..Default::default()
            },
        };
        let envelope = identifier
            .identify_pag_envelope(&graph.pag, &witness)
            .map_err(|error| CausalIdentifyError::new_err(error.to_string()))?;
        let total_mass = envelope.identified_weight.0 + envelope.unidentified_weight.0;
        let enumeration_capped = envelope
            .critical_graph_features
            .iter()
            .any(|feature| feature.kind.as_ref() == "completion_enumeration_capped");
        if total_mass <= 0.0 || envelope.identified_weight.0 <= 0.0 {
            return Err(PyValueError::new_err(
                "PAG response curve has no identified mass in its completion envelope",
            ));
        }

        let mut curves = Vec::<Vec<f64>>::new();
        let mut first_response = None;
        let mut estimable_mass = 0.0;
        let mut worst_support = SupportStatus::Supported;
        let mut diagnostic_ids = Vec::new();
        let mut diagnostic_values = Vec::new();
        let mut diagnostic_details = Vec::new();
        let mut warnings = Vec::new();
        for (case_index, case) in envelope.cases.iter().enumerate() {
            if !matches!(
                case.result.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            ) {
                continue;
            }
            let Some(estimand) = case.result.estimands.first() else {
                continue;
            };
            let response = ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set))
                .estimate_identified(
                    &data,
                    &query,
                    case.result.status,
                    case.result.required_assumptions.clone(),
                )
                .map_err(py_err)?;
            let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
                &response.estimate
            else {
                return Err(PyValueError::new_err(
                    "PAG MeanCurve completion did not produce a response surface",
                ));
            };
            curves.push(mean.to_vec());
            estimable_mass += case.weight.0;
            if support_rank(response.support.status) > support_rank(worst_support) {
                worst_support = response.support.status;
            }
            for diagnostic in &response.support.diagnostics {
                diagnostic_ids.push(format!("completion.{case_index}.{}", diagnostic.id));
                diagnostic_values.push(diagnostic.values.to_vec());
                diagnostic_details
                    .push(format!("MAG completion {case_index}: {}", diagnostic.detail));
            }
            warnings.extend(response.support.warnings.iter().map(|warning| {
                format!("completion.{case_index}.{}: {}", warning.code, warning.message)
            }));
            if first_response.is_none() {
                first_response = Some(response);
            }
        }
        let first = first_response.ok_or_else(|| {
            PyValueError::new_err("PAG response curve has identified mass but no estimable case")
        })?;
        // The envelope is the range of completion-specific Kennedy *point* curves. It is an
        // identified-set payload under structural uncertainty, not a confidence interval and
        // not a claim that sampling uncertainty within each completion was folded in.
        warnings.push(
            "response.pag_envelope_point_estimates: envelope bounds are the pointwise min/max of \
             completion-specific point curves; they omit within-completion sampling uncertainty \
             and are not a confidence band"
                .into(),
        );
        let width = curves.first().map_or(0, Vec::len);
        let mut lower = vec![f64::INFINITY; width];
        let mut upper = vec![f64::NEG_INFINITY; width];
        for curve in &curves {
            if curve.len() != width {
                return Err(PyValueError::new_err("PAG completion curves do not share one grid"));
            }
            for (index, value) in curve.iter().enumerate() {
                lower[index] = lower[index].min(*value);
                upper[index] = upper[index].max(*value);
            }
        }
        let ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid, dimension, ..
        }) = &first.estimate
        else {
            unreachable!("checked above")
        };
        let points = grid.chunks(*dimension).map(<[f64]>::to_vec).collect();
        let lower_rows = lower.iter().map(|value| vec![*value]).collect();
        let upper_rows = upper.iter().map(|value| vec![*value]).collect();
        let support_status = support_status_name(worst_support).to_owned();
        Ok(ResponseAnalysisResult {
            treatments: vec![treatment],
            outcomes: vec![outcome],
            points,
            values: Vec::new(),
            scalar: None,
            matrix: None,
            uncertainty_kind: "identified_set".into(),
            lower: Some(lower_rows),
            upper: Some(upper_rows),
            level: None,
            standard_error: None,
            replicates: None,
            artifact_id: None,
            support_status,
            support_minima: first.support.query_region.minima.to_vec(),
            support_maxima: first.support.query_region.maxima.to_vec(),
            support_point_status: Vec::new(),
            diagnostic_ids,
            diagnostic_values,
            diagnostic_details,
            warnings,
            identification: format!("{:?}", envelope.status),
            adjustment_set: Vec::new(),
            horizon_adjustment_sets: Vec::new(),
            assumptions: vec![
                "generalized adjustment within each identified MAG completion".into(),
            ],
            provenance_id: "estimate.response.kennedy_dr".into(),
            // Prior-restricted/partially-identified cases are not licensed point curves
            // for this estimator; preserve their weight as unidentified execution mass.
            identified_mass: Some(estimable_mass / total_mass),
            unidentified_mass: Some((total_mass - estimable_mass) / total_mass),
            completion_count: Some(envelope.cases.len()),
            truncated_completions: Some(envelope.truncated_completions),
            enumeration_capped: Some(enumeration_capped),
            mass_scope: Some(if enumeration_capped {
                "examined_completions".into()
            } else {
                "full_class".into()
            }),
            evidence_status: None,
            allowlist_reason: None,
            allowlist_parent: None,
        })
    })
}

pub(crate) fn resolve_names(
    schema: &antecedent_core::CausalSchema,
    names: &[String],
) -> PyResult<Vec<VariableId>> {
    if names.is_empty() {
        return Err(PyValueError::new_err("response variables must not be empty"));
    }
    names.iter().map(|name| schema.id_of(name).map_err(py_err)).collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_functional(
    kind: &str,
    treatments: &[VariableId],
    outcomes: &[VariableId],
    grid: Option<Vec<f64>>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    intervention_kinds: Option<Vec<String>>,
    intervention_parameters: Option<Vec<Vec<f64>>>,
    order: u8,
    scale: DerivativeScale,
    weighting: DerivativeWeighting,
) -> PyResult<ResponseFunctional> {
    let scalar = || -> PyResult<(VariableId, VariableId)> {
        match (treatments, outcomes) {
            ([treatment], [outcome]) => Ok((*treatment, *outcome)),
            _ => Err(PyValueError::new_err(
                "this response functional requires one treatment and one outcome",
            )),
        }
    };
    match kind {
        "response_curve" => {
            let (treatment, outcome) = scalar()?;
            Ok(ResponseFunctional::MeanCurve {
                outcome,
                treatment: antecedent_core::ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from(
                        grid.ok_or_else(|| PyValueError::new_err("grid is required"))?,
                    )),
                ),
            })
        }
        "average_derivative" => {
            let (treatment, outcome) = scalar()?;
            Ok(ResponseFunctional::AverageDerivative { outcome, treatment, weighting })
        }
        "point_derivative" | "elasticity" | "semi_elasticity" => {
            let (treatment, outcome) = scalar()?;
            let point = at
                .as_deref()
                .and_then(|values| values.first())
                .copied()
                .ok_or_else(|| PyValueError::new_err("at is required"))?;
            Ok(ResponseFunctional::PointDerivative { outcome, treatment, at: point, order, scale })
        }
        "directional_derivative" => Ok(ResponseFunctional::DirectionalDerivative {
            outcomes: Arc::from(outcomes),
            treatments: Arc::from(treatments),
            at: Arc::from(at.ok_or_else(|| PyValueError::new_err("at is required"))?),
            direction: Arc::from(
                direction.ok_or_else(|| PyValueError::new_err("direction is required"))?,
            ),
        }),
        "response_jacobian" => Ok(ResponseFunctional::Jacobian {
            outcomes: Arc::from(outcomes),
            treatments: Arc::from(treatments),
            at: Arc::from(at.ok_or_else(|| PyValueError::new_err("at is required"))?),
            scale,
        }),
        "intervention_response" => {
            let [outcome] = outcomes else {
                return Err(PyValueError::new_err(
                    "InterventionResponse requires exactly one outcome",
                ));
            };
            let kinds = intervention_kinds.ok_or_else(|| {
                PyValueError::new_err("InterventionResponse requires intervention kinds")
            })?;
            let parameters = intervention_parameters.ok_or_else(|| {
                PyValueError::new_err("InterventionResponse requires intervention parameters")
            })?;
            if treatments.len() != kinds.len() || kinds.len() != parameters.len() {
                return Err(PyValueError::new_err(
                    "intervention variables, kinds, and parameters must have equal lengths",
                ));
            }
            let interventions = treatments
                .iter()
                .zip(kinds)
                .zip(parameters)
                .map(|((&variable, kind), parameters)| {
                    intervention_from_parts(variable, &kind, &parameters)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(ResponseFunctional::InterventionResponse {
                outcome: *outcome,
                interventions: Arc::from(interventions),
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "response functional {kind:?} is not executable by this estimator"
        ))),
    }
}

fn intervention_from_parts(
    variable: VariableId,
    kind: &str,
    parameters: &[f64],
) -> PyResult<Intervention> {
    let exactly = |count: usize| {
        if parameters.len() == count && parameters.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{kind} intervention requires {count} finite parameter(s)"
            )))
        }
    };
    match kind {
        "set" => {
            exactly(1)?;
            Ok(Intervention::set(variable, Value::f64(parameters[0])))
        }
        "shift" => {
            exactly(1)?;
            Ok(Intervention::shift(variable, Value::f64(parameters[0])))
        }
        "bernoulli" => {
            exactly(1)?;
            Ok(Intervention::stochastic(variable, StochasticPolicy::bernoulli(parameters[0])))
        }
        "gaussian" => {
            exactly(2)?;
            Ok(Intervention::stochastic(
                variable,
                StochasticPolicy::gaussian(parameters[0], parameters[1]),
            ))
        }
        "categorical" => {
            if parameters.is_empty() || parameters.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "categorical intervention requires finite probabilities",
                ));
            }
            Ok(Intervention::stochastic(
                variable,
                StochasticPolicy::categorical(Arc::from(parameters)),
            ))
        }
        "soft_constant" => {
            exactly(1)?;
            Ok(Intervention::soft(variable, MechanismOverride::constant(parameters[0])))
        }
        "soft_additive_shift" => {
            exactly(1)?;
            Ok(Intervention::soft(variable, MechanismOverride::additive_shift(parameters[0])))
        }
        "soft" | "sequence" => Err(PyValueError::new_err(format!(
            "{kind} interventions require a structural/temporal model and are not estimable by response.intervention_gcomp"
        ))),
        _ => Err(PyValueError::new_err(format!("unknown intervention kind {kind:?}"))),
    }
}

fn parse_scale(value: &str) -> PyResult<DerivativeScale> {
    match value {
        "identity" => Ok(DerivativeScale::Identity),
        "log_treatment" => Ok(DerivativeScale::LogTreatment),
        "log_outcome" => Ok(DerivativeScale::LogOutcome),
        "log_log" => Ok(DerivativeScale::LogLog),
        _ => Err(PyValueError::new_err("unknown derivative scale")),
    }
}

fn parse_weighting(value: &str) -> PyResult<DerivativeWeighting> {
    match value {
        "observed" => Ok(DerivativeWeighting::Observed),
        _ => Err(PyValueError::new_err("the current ADE estimator supports weighting='observed'")),
    }
}

pub(crate) fn response_result(
    response: antecedent_core::CausalResponse,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    mut adjustment_set: Vec<String>,
    names: &[String],
    evidence: Option<antecedent::CellStatus>,
) -> PyResult<ResponseAnalysisResult> {
    let horizon_adjustment_sets = named_horizon_adjustments(&response, names);
    if let Some(first) = first_horizon_template_names(&response, names) {
        adjustment_set = first;
    }
    let (points, values, scalar, matrix) = match response.estimate {
        ResponseIdentification::PointIdentified(value) => value_parts(value)?,
        _ => {
            return Err(PyValueError::new_err(
                "the continuous-response estimator expected point identification",
            ));
        }
    };
    let (uncertainty_kind, lower, upper, level, standard_error, replicates, artifact_id) =
        uncertainty_parts(response.uncertainty);
    let support_status = support_status_name(response.support.status).to_owned();
    let (evidence_status, allowlist_reason, allowlist_parent) =
        crate::evidence_status_parts(evidence);
    Ok(ResponseAnalysisResult {
        treatments,
        outcomes,
        points,
        values,
        scalar,
        matrix,
        uncertainty_kind,
        lower,
        upper,
        level,
        standard_error,
        replicates,
        artifact_id,
        support_status,
        support_minima: response.support.query_region.minima.to_vec(),
        support_maxima: response.support.query_region.maxima.to_vec(),
        support_point_status: response
            .support
            .point_status
            .as_ref()
            .map(|cells| cells.iter().map(|status| status.as_str().to_owned()).collect())
            .unwrap_or_default(),
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
            .map(|warning| warning.message.to_string())
            .collect(),
        identification: format!("{:?}", response.identification_status),
        adjustment_set,
        horizon_adjustment_sets,
        assumptions: response
            .assumptions
            .entries
            .iter()
            .map(|record| format!("{:?}", record.assumption))
            .collect(),
        provenance_id: response.provenance_id.to_string(),
        identified_mass: None,
        unidentified_mass: None,
        completion_count: None,
        truncated_completions: None,
        enumeration_capped: None,
        mass_scope: None,
        evidence_status,
        allowlist_reason,
        allowlist_parent,
    })
}

fn named_horizon_adjustments(
    response: &antecedent_core::CausalResponse,
    names: &[String],
) -> Vec<Vec<String>> {
    match &response.horizon_identification {
        Some(horizons) => horizons
            .iter()
            .map(|horizon| {
                horizon
                    .adjustment
                    .iter()
                    .map(|key| {
                        let name = names
                            .get(key.variable.as_usize())
                            .cloned()
                            .unwrap_or_else(|| format!("var{}", key.variable.raw()));
                        format!("{name}@{}", key.offset)
                    })
                    .collect()
            })
            .collect(),
        None => Vec::new(),
    }
}

fn first_horizon_template_names(
    response: &antecedent_core::CausalResponse,
    names: &[String],
) -> Option<Vec<String>> {
    let first = response.horizon_identification.as_ref()?.first()?;
    Some(
        first
            .adjustment
            .iter()
            .map(|key| {
                names
                    .get(key.variable.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", key.variable.raw()))
            })
            .collect(),
    )
}

fn support_status_name(status: SupportStatus) -> &'static str {
    status.as_str()
}

const fn support_rank(status: SupportStatus) -> u8 {
    match status {
        SupportStatus::Supported => 0,
        SupportStatus::WeakOverlap => 1,
        SupportStatus::Extrapolative => 2,
        SupportStatus::OutsideEmpiricalSupport => 3,
    }
}

type ValueParts = (Vec<Vec<f64>>, Vec<Vec<f64>>, Option<f64>, Option<Vec<Vec<f64>>>);

fn value_parts(value: ResponseValue) -> PyResult<ValueParts> {
    match value {
        ResponseValue::Scalar(value) => Ok((Vec::new(), Vec::new(), Some(value), None)),
        ResponseValue::Surface { grid, dimension, mean } => {
            let points = grid.chunks(dimension).map(<[f64]>::to_vec).collect::<Vec<_>>();
            let values = mean.iter().map(|value| vec![*value]).collect();
            Ok((points, values, None, None))
        }
        ResponseValue::Vector(values) => {
            Ok((Vec::new(), Vec::new(), None, Some(vec![values.to_vec()])))
        }
        ResponseValue::Jacobian { outcomes, treatments, values } => {
            if values.len() != outcomes * treatments {
                return Err(PyValueError::new_err("invalid Jacobian result shape"));
            }
            Ok((
                Vec::new(),
                Vec::new(),
                None,
                Some(values.chunks(treatments).map(<[f64]>::to_vec).collect()),
            ))
        }
        ResponseValue::Envelope(_) => Err(PyValueError::new_err(
            "identified response envelopes use the bounds API, not this point estimator",
        )),
    }
}

type UncertaintyParts = (
    String,
    Option<Vec<Vec<f64>>>,
    Option<Vec<Vec<f64>>>,
    Option<f64>,
    Option<f64>,
    Option<u32>,
    Option<String>,
);

fn uncertainty_parts(value: ResponseUncertainty) -> UncertaintyParts {
    let rows = |values: Arc<[f64]>| values.iter().map(|value| vec![*value]).collect();
    match value {
        ResponseUncertainty::None => ("none".into(), None, None, None, None, None, None),
        ResponseUncertainty::Scalar { standard_error, level, lower, upper } => (
            "pointwise".into(),
            Some(vec![vec![lower]]),
            Some(vec![vec![upper]]),
            Some(level),
            Some(standard_error),
            None,
            None,
        ),
        ResponseUncertainty::PointwiseBand { level, lower, upper } => (
            "pointwise".into(),
            Some(rows(lower)),
            Some(rows(upper)),
            Some(level),
            None,
            None,
            None,
        ),
        ResponseUncertainty::SimultaneousBand { level, lower, upper, replicates } => (
            "simultaneous".into(),
            Some(rows(lower)),
            Some(rows(upper)),
            Some(level),
            None,
            Some(replicates),
            None,
        ),
        ResponseUncertainty::IdentifiedEnvelopeBand { level, lower_outer, upper_outer } => (
            "identified_set".into(),
            Some(rows(lower_outer)),
            Some(rows(upper_outer)),
            Some(level),
            None,
            None,
            None,
        ),
        ResponseUncertainty::Posterior { artifact_id } => {
            ("posterior".into(), None, None, None, None, None, Some(artifact_id.to_string()))
        }
    }
}

/// Temporal dose × horizon / intervention-path response (ADR 0021).
#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, kind, treatments, outcomes, *,
    grid=None, intervention_kinds=None, intervention_parameters=None,
    horizons, policy=crate::temporal_license::DEFAULT_POLICY,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG, max_history_lag=None,
    seed=1, threads=1, accepted=false, refute=None
))]
#[allow(clippy::too_many_arguments)]
fn analyze_temporal_response(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, u32, String, u32)>,
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    grid: Option<Vec<f64>>,
    intervention_kinds: Option<Vec<String>>,
    intervention_parameters: Option<Vec<Vec<f64>>>,
    horizons: Vec<u32>,
    policy: &str,
    treatment_lag: u32,
    max_history_lag: Option<u32>,
    seed: u64,
    threads: u32,
    accepted: bool,
    refute: Option<Bound<'_, PyAny>>,
) -> PyResult<ResponseAnalysisResult> {
    let suite = match refute.as_ref() {
        None => RefuteSuite::None,
        Some(obj) => suite_from_refute(Some(obj))?,
    };
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let policy = policy.to_ascii_lowercase();
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let dag = temporal_dag_from_schema_edges(series.schema(), &edges)?;
        let treatment_ids = resolve_names(series.schema(), &treatments)?;
        let outcome_ids = resolve_names(series.schema(), &outcomes)?;
        let functional = build_functional(
            &kind,
            &treatment_ids,
            &outcome_ids,
            grid,
            None,
            None,
            intervention_kinds,
            intervention_parameters,
            1,
            DerivativeScale::Identity,
            DerivativeWeighting::Observed,
        )?;
        // Match PulseEffect / SustainedEffect: non-negative lag → policy origin at
        // `-treatment_lag`. Licensed sustained is the single-step window.
        let temporal_policy = crate::temporal_license::policy_at_lag(policy, treatment_lag)?;
        let temporal = TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let query = ResponseQuery::new(functional).with_temporal(temporal);
        let causal_query = CausalQuery::Response(query.clone());
        if let Some(cell) = support_cell(
            &causal_query,
            GraphClass::TemporalDag,
            if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
            &InferenceMode::Frequentist,
            suite,
        ) {
            refuse_if_not_applicable(cell).map_err(py_err)?;
        }
        let mut builder = Study::series(series);
        builder =
            if accepted { builder.graph(AcceptedGraph::from(dag)) } else { builder.graph(dag) };
        let analysis = builder.query(causal_query).refute(suite).build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        let response = result.response.ok_or_else(|| {
            PyValueError::new_err("temporal response run did not produce a response payload")
        })?;
        let adjustment_set = result
            .estimand
            .adjustment_set
            .iter()
            .map(|id| {
                names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
            })
            .collect();
        response_result(
            response,
            treatments,
            outcomes,
            adjustment_set,
            &names,
            result.support_status,
        )
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResponseAnalysisResult>()?;
    m.add_function(wrap_pyfunction!(analyze_response, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_response_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_response, m)?)?;
    Ok(())
}
