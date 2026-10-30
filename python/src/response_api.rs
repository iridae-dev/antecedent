//! Python execution boundary for continuous causal-response queries.

use std::sync::Arc;

use antecedent_core::{
    DerivativeScale, DerivativeWeighting, GridSpec, IntervalInterpretation, Intervention,
    InterventionSequence, MechanismOverride, ResponseFunctional, ResponseIdentification,
    ResponseUncertainty, ResponseValue, SequencedIntervention, StochasticPolicy, SupportStatus,
    TemporalPolicy, Value, VariableId,
};
use pyo3::prelude::*;

use crate::py_err;

#[pyclass(skip_from_py_object)]
pub(crate) struct ResponseAnalysisResult {
    pub(crate) authority: Option<Arc<NativeResponseAuthority>>,
    #[pyo3(get)]
    certificate_json: Option<String>,
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
    /// `"confidence"` or `"credible"` for an interval-bearing uncertainty (a credible
    /// interval's `standard_error` is a posterior standard deviation); `None` otherwise.
    #[pyo3(get)]
    interval_interpretation: Option<String>,
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
    diagnostic_scopes: Vec<String>,
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
    /// Mass identified in theory but not evaluable by the estimator; separate
    /// from [`Self::unidentified_mass`].
    #[pyo3(get)]
    unevaluable_mass: Option<f64>,
    /// Mass on identified atoms the Interactive latency tier never evaluated;
    /// separate from both [`Self::unidentified_mass`] and
    /// [`Self::unevaluable_mass`].
    #[pyo3(get)]
    subsampled_out_mass: Option<f64>,
    #[pyo3(get)]
    completion_count: Option<usize>,
    #[pyo3(get)]
    truncated_completions: Option<usize>,
    #[pyo3(get)]
    enumeration_capped: Option<bool>,
    #[pyo3(get)]
    mass_scope: Option<String>,
    #[pyo3(get)]
    weight_basis: Option<String>,
    /// How identified graph atoms were combined, when the route aggregates across graphs.
    #[pyo3(get)]
    aggregation_policy: Option<String>,
    #[pyo3(get)]
    atom_keys: Vec<u64>,
    #[pyo3(get)]
    atom_weights: Vec<f64>,
    #[pyo3(get)]
    atom_statuses: Vec<String>,
    #[pyo3(get)]
    atom_values: Vec<Vec<f64>>,
    /// Matrix evidence contract (`licensed` / `allowed_unlicensed`). Distinct from
    /// empirical dose-range [`Self::support_status`].
    #[pyo3(get)]
    evidence_status: Option<String>,
    #[pyo3(get)]
    allowlist_reason: Option<String>,
    #[pyo3(get)]
    allowlist_parent: Option<String>,
    /// Study-level diagnostic codes (`code: message`), including class envelopes.
    #[pyo3(get)]
    diagnostics: Vec<String>,
    /// Logical-plan identifier when the result came through Study prepare/estimate.
    #[pyo3(get)]
    identifier: Option<String>,
    /// The missing-data report of the execution as JSON, when the route reports one.
    #[pyo3(get)]
    missing_data_json: Option<String>,
    /// Every interval this execution reported, primary first, as JSON.
    #[pyo3(get)]
    interval_bindings_json: Option<String>,
    /// What the native response fixes about each value coordinate (regime, horizon,
    /// functional), or the typed refusal that says why it has none. Units, population
    /// and scale are declared by the caller of [`Self::response_coordinates`].
    coordinates: CoordinateSkeleton,
}

pub(crate) struct NativeResponseAuthority {
    pub(crate) response: antecedent_core::CausalResponse,
    pub(crate) prepared: Arc<antecedent::PreparedStudy>,
    pub(crate) schema: antecedent_core::CausalSchema,
    pub(crate) result: Arc<antecedent::StudyResult>,
    pub(crate) context: antecedent_core::ExecutionContext,
}

impl ResponseAnalysisResult {
    pub(crate) fn claim_projection(&self) -> serde_json::Value {
        serde_json::json!({
            "treatment": self.treatments.first(), "outcome": self.outcomes.first(),
            "grid": self.points.iter().filter_map(|p| p.first()).collect::<Vec<_>>(),
            "means": self.values.iter().filter_map(|v| v.first()).collect::<Vec<_>>(),
            "identification": self.authority.as_ref().map(|a| a.response.identification_status.as_str()),
            "has_envelope": self.identified_mass.is_some(), "support_status": self.support_status,
            "point_status": if self.support_point_status.is_empty() {None} else {Some(&self.support_point_status)},
            "provenance_id": self.provenance_id,
            "data_snapshot_id": self.authority.as_ref().and_then(|a| a.result.executed_contract()).map(|c| c.identities.data_snapshot.to_hex()),
            "program_id": self.authority.as_ref().and_then(|a| a.result.executed_contract()).and_then(|c| c.identities.program).map(|id| id.to_hex()),
            "uncertainty": {"kind": self.uncertainty_kind, "lower": self.lower, "upper": self.upper,
                "level": self.level, "standard_error": self.standard_error, "replicates": self.replicates,
                "artifact_id": self.artifact_id, "interpretation": self.interval_interpretation},
        })
    }

    pub(crate) fn with_authority(
        mut self,
        prepared: &antecedent::PreparedStudy,
        schema: &antecedent_core::CausalSchema,
        result: &antecedent::StudyResult,
        context: &antecedent_core::ExecutionContext,
    ) -> Self {
        if let (Some(response), Some(_)) = (result.response(), result.executed_contract()) {
            self.authority = Some(Arc::new(NativeResponseAuthority {
                response: response.clone(),
                prepared: Arc::new(prepared.clone()),
                schema: schema.clone(),
                result: Arc::new(result.clone()),
                context: context.clone(),
            }));
        }
        self
    }
}

type CoordinateSkeleton = Result<
    Vec<antecedent_contract::ResponseCoordinateSkeleton>,
    antecedent_contract::ExternalRefusal,
>;

#[pymethods]
impl ResponseAnalysisResult {
    /// Refuter reports the study's validation stage produced for this response, one
    /// `(refuter, passed, informative, original, refuted, failure)` per report; empty
    /// when no suite ran or the result carries no study authority.
    fn refutation_reports(&self) -> Vec<(String, bool, bool, f64, f64, Option<String>)> {
        self.authority.as_ref().map_or_else(Vec::new, |authority| {
            authority
                .result
                .refutations
                .iter()
                .map(|report| {
                    (
                        report.refuter.to_string(),
                        report.passed,
                        report.informative,
                        report.original_ate,
                        report.refuted_ate,
                        report.failure_condition.as_deref().map(str::to_owned),
                    )
                })
                .collect()
        })
    }

    fn program_basis_json(&self) -> PyResult<Option<String>> {
        let Some(authority) = &self.authority else {
            return Ok(None);
        };
        let Some(contract) = authority.result.executed_contract() else {
            return Ok(None);
        };
        let antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } =
            &authority.response.estimand
        else {
            return Ok(None);
        };
        let grid = treatment.grid.values().map_err(|error| crate::value_err(error.to_string()))?;
        let named =
            |id| authority.schema.get(id).map(|v| v.name.to_string()).map_err(crate::py_err);
        Ok(Some(
            serde_json::json!({
                "graph_id": format!("graph:{}", contract.identities.identification),
                "status": authority.response.identification_status.as_str(),
                "treatment": named(treatment.variable)?, "outcome": named(*outcome)?, "grid": grid,
                "snapshot_id": contract.identities.data_snapshot.to_hex(),
                "program_id": contract.identities.program.map(|id| id.to_hex()),
                "rng_id": format!("native:seed:{}", authority.context.rng.master_seed()),
                "statement": "checked native response identification",
            })
            .to_string(),
        ))
    }

    /// The joint draws of dose number `dose_index` across the horizons of a temporal response
    /// prepared with joint horizons retained.
    ///
    /// Returns `(draws_json, refusal_json)`: exactly one is `None`. The draws carry the
    /// horizons, the point value of each horizon's cell, one column of replicates per horizon
    /// and the alignment and evidence identities.
    fn horizon_draws_json(&self, dose_index: usize) -> PyResult<(Option<String>, Option<String>)> {
        let Some(authority) = &self.authority else {
            return Err(crate::value_err(
                "this response carries no native response payload to read horizon draws from",
            ));
        };
        match antecedent::analysis::horizon_composition::horizon_draws(
            &authority.response,
            dose_index,
        ) {
            Ok(draws) => Ok((
                Some(
                    serde_json::json!({
                        "horizons": draws.horizons,
                        "center": draws.center,
                        "columns": draws.columns,
                        "alignment_id": draws.alignment_id,
                        "evidence_id": draws.evidence_id,
                    })
                    .to_string(),
                ),
                None,
            )),
            Err(refusal) => {
                let json = serde_json::to_string(
                    &antecedent_io::external_binding_wire::RefusalWire::from(refusal),
                )
                .map_err(|error| crate::value_err(error.to_string()))?;
                Ok((None, Some(json)))
            }
        }
    }

    /// The scientific coordinates of the response values, derived natively.
    ///
    /// Returns `(quantities_json, refusal_json)`: exactly one is `None`. The
    /// quantities are the `ScientificQuantityWire` objects, one per value in value
    /// order. Units are never inferred: blank units, population or scale refuse.
    #[pyo3(signature = (outcome_units, population="target", transform="identity"))]
    fn response_coordinates(
        &self,
        outcome_units: &str,
        population: &str,
        transform: &str,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let labels = antecedent_contract::ResponseCoordinateLabels {
            outcome_units,
            population_id: population,
            transform_id: transform,
        };
        let labelled = self
            .coordinates
            .clone()
            .and_then(|cells| antecedent_contract::label_coordinates(&cells, &labels));
        match labelled {
            Ok(quantities) => {
                let wires: Vec<antecedent_io::quantity_wire::ScientificQuantityWire> =
                    quantities.iter().map(Into::into).collect();
                let json = serde_json::to_string(&wires)
                    .map_err(|error| crate::value_err(error.to_string()))?;
                Ok((Some(json), None))
            }
            Err(refusal) => {
                let json = serde_json::to_string(
                    &antecedent_io::external_binding_wire::RefusalWire::from(refusal),
                )
                .map_err(|error| crate::value_err(error.to_string()))?;
                Ok((None, Some(json)))
            }
        }
    }
}

/// Coordinates a native response fixes, named with the caller's variable names.
// The refusal is the cold path of a once-per-refusal check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn coordinate_skeleton(
    response: &antecedent_core::CausalResponse,
    names: &[String],
) -> CoordinateSkeleton {
    antecedent_contract::response_coordinate_skeleton(response, &|id| {
        names.get(id.as_usize()).cloned()
    })
}

pub(crate) fn resolve_names(
    schema: &antecedent_core::CausalSchema,
    names: &[String],
) -> PyResult<Vec<VariableId>> {
    if names.is_empty() {
        return Err(crate::value_err("response variables must not be empty"));
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
            _ => Err(crate::value_err(
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
                        grid.ok_or_else(|| crate::value_err("grid is required"))?,
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
                .ok_or_else(|| crate::value_err("at is required"))?;
            Ok(ResponseFunctional::PointDerivative { outcome, treatment, at: point, order, scale })
        }
        "directional_derivative" => Ok(ResponseFunctional::DirectionalDerivative {
            outcomes: Arc::from(outcomes),
            treatments: Arc::from(treatments),
            at: Arc::from(at.ok_or_else(|| crate::value_err("at is required"))?),
            direction: Arc::from(
                direction.ok_or_else(|| crate::value_err("direction is required"))?,
            ),
        }),
        "response_jacobian" => Ok(ResponseFunctional::Jacobian {
            outcomes: Arc::from(outcomes),
            treatments: Arc::from(treatments),
            at: Arc::from(at.ok_or_else(|| crate::value_err("at is required"))?),
            scale,
        }),
        "intervention_response" => {
            let [outcome] = outcomes else {
                return Err(crate::value_err("InterventionResponse requires exactly one outcome"));
            };
            let kinds = intervention_kinds.ok_or_else(|| {
                crate::value_err("InterventionResponse requires intervention kinds")
            })?;
            let parameters = intervention_parameters.ok_or_else(|| {
                crate::value_err("InterventionResponse requires intervention parameters")
            })?;
            if treatments.len() != kinds.len() || kinds.len() != parameters.len() {
                return Err(crate::value_err(
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
        _ => Err(crate::value_err(format!(
            "response functional {kind:?} is not executable by this estimator"
        ))),
    }
}

pub(crate) fn wrap_temporal_sequence_steps(
    functional: ResponseFunctional,
    origin: i32,
) -> PyResult<ResponseFunctional> {
    let ResponseFunctional::InterventionResponse { outcome, interventions } = functional else {
        return Ok(functional);
    };
    if interventions.len() <= 1 {
        return Ok(ResponseFunctional::InterventionResponse { outcome, interventions });
    }
    let vars: Vec<VariableId> =
        interventions.iter().filter_map(Intervention::primary_variable).collect();
    if vars.len() != interventions.len() {
        return Err(crate::value_err("each Sequence step must name one treatment variable"));
    }
    let same_var = vars.iter().all(|variable| *variable == vars[0]);
    let unique = {
        let mut seen = vars.clone();
        seen.sort_by_key(|variable| variable.raw());
        seen.dedup();
        seen.len() == vars.len()
    };
    let offsets: Vec<i32> = if same_var {
        let n = i32::try_from(vars.len()).map_err(|_| crate::value_err("Sequence is too long"))?;
        (0..n)
            .map(|i| {
                origin.checked_sub(n - 1 - i).ok_or_else(|| {
                    crate::value_err("treatment_lag plus Sequence length does not fit in i32")
                })
            })
            .collect::<PyResult<Vec<_>>>()?
    } else if unique {
        vec![origin; vars.len()]
    } else {
        return Err(crate::value_err(
            "Sequence schedule is ambiguous; use one variable for a multi-step policy \
             or distinct variables for a joint single-time policy",
        ));
    };
    let steps: Vec<SequencedIntervention> = interventions
        .iter()
        .zip(offsets)
        .map(|(intervention, at)| {
            SequencedIntervention::new(intervention.clone(), TemporalPolicy::pulse(at))
        })
        .collect();
    Ok(ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from([Intervention::sequence(InterventionSequence::new(steps))]),
    })
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
            Err(crate::value_err(format!(
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
                return Err(crate::value_err(
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
        "soft_multiplicative" => {
            exactly(1)?;
            Ok(Intervention::soft(variable, MechanismOverride::named("multiplicative", parameters)))
        }
        "soft_truncated_shift" => {
            exactly(3)?;
            if parameters[1] > parameters[2] {
                return Err(crate::value_err("truncated_shift requires lower <= upper"));
            }
            Ok(Intervention::soft(
                variable,
                MechanismOverride::named("truncated_shift", parameters),
            ))
        }
        "soft" | "sequence" => Err(crate::value_err(format!(
            "{kind} interventions require a structural/temporal model and are not estimable by response.intervention_gcomp"
        ))),
        _ => Err(crate::value_err(format!("unknown intervention kind {kind:?}"))),
    }
}

pub(crate) fn parse_scale(value: &str) -> PyResult<DerivativeScale> {
    match value {
        "identity" => Ok(DerivativeScale::Identity),
        "log_treatment" => Ok(DerivativeScale::LogTreatment),
        "log_outcome" => Ok(DerivativeScale::LogOutcome),
        "log_log" => Ok(DerivativeScale::LogLog),
        _ => Err(crate::value_err("unknown derivative scale")),
    }
}

pub(crate) fn parse_weighting(value: &str) -> PyResult<DerivativeWeighting> {
    match value {
        "observed" => Ok(DerivativeWeighting::Observed),
        _ => Err(crate::value_err("the current ADE estimator supports weighting='observed'")),
    }
}

pub(crate) fn response_result(
    response: antecedent_core::CausalResponse,
    structural: Option<&antecedent::result::StructuralResponseMixture>,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    mut adjustment_set: Vec<String>,
    names: &[String],
    evidence: Option<antecedent::CellStatus>,
) -> PyResult<ResponseAnalysisResult> {
    let horizon_adjustment_sets = named_horizon_adjustments(&response, names);
    let coordinates = coordinate_skeleton(&response, names);
    if let Some(first) = first_horizon_template_names(&response, names) {
        adjustment_set = first;
    }
    let identified_set = structural.and_then(|mixture| mixture.identified_set.as_ref());
    let (points, values, scalar, matrix) = if let Some(envelope) = identified_set {
        let conditional = structural
            .and_then(|mixture| mixture.conditional_on_identified.as_ref())
            .map(|value| response_value_flat(Some(value)));
        let temporal_class = response.horizon_identification.as_ref().is_some_and(|horizons| {
            horizons
                .iter()
                .any(|horizon| horizon.method.as_ref() == "temporal_class.completion_envelope")
        });
        let legacy_values = if temporal_class {
            conditional.unwrap_or_default()
        } else {
            conditional
                .or_else(|| structural.and_then(structural_weighted_values))
                .filter(|values| values.len() == envelope.lower.len())
                .unwrap_or_else(|| envelope.lower.to_vec())
        };
        let scalar = if legacy_values.len() == 1 { Some(legacy_values[0]) } else { None };
        // A coordinate-free identified set (a scalar functional's bound) has no
        // grid: give each bound the same placeholder coordinate a scalar
        // response carries, so every row still has one point.
        let points = if envelope.dimension == 0 {
            vec![vec![0.0]; envelope.lower.len()]
        } else {
            envelope.grid.chunks(envelope.dimension).map(<[f64]>::to_vec).collect()
        };
        (points, legacy_values.into_iter().map(|value| vec![value]).collect(), scalar, None)
    } else {
        match response.estimate {
            ResponseIdentification::PointIdentified(value)
            | ResponseIdentification::PartiallyIdentified(value) => value_parts(value)?,
            ResponseIdentification::GraphDependent(_)
            | ResponseIdentification::Unidentified { .. } => {
                return Err(crate::value_err(
                    "the continuous-response estimator expected an identified payload",
                ));
            }
        }
    };
    let (
        mut uncertainty_kind,
        mut lower,
        mut upper,
        mut level,
        standard_error,
        replicates,
        artifact_id,
        mut interval_interpretation,
    ) = uncertainty_parts(response.uncertainty);
    if let Some(envelope) = identified_set {
        uncertainty_kind = "identified_set".into();
        interval_interpretation = None;
        lower = Some(envelope.lower.iter().map(|value| vec![*value]).collect());
        upper = Some(envelope.upper.iter().map(|value| vec![*value]).collect());
        level = None;
    }
    let support_status = support_status_name(response.support.status).to_owned();
    let (evidence_status, allowlist_reason, allowlist_parent) =
        crate::evidence_status_parts(evidence);
    let mut unique_treatments = Vec::new();
    for name in treatments {
        if !unique_treatments.contains(&name) {
            unique_treatments.push(name);
        }
    }
    Ok(ResponseAnalysisResult {
        authority: None,
        certificate_json: None,
        treatments: unique_treatments,
        outcomes,
        points,
        values,
        scalar,
        matrix,
        uncertainty_kind,
        interval_interpretation,
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
        diagnostic_scopes: response
            .support
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.scope.as_str().to_owned())
            .collect(),
        warnings: response
            .support
            .warnings
            .iter()
            .map(|warning| format!("{}: {}", warning.code, warning.message))
            .collect(),
        identification: format!("{:?}", response.identification_status),
        adjustment_set: crate::public_adjustment_set(
            response.identification_status,
            adjustment_set,
        ),
        horizon_adjustment_sets,
        assumptions: response
            .assumptions
            .entries
            .iter()
            .map(|record| format!("{:?}", record.assumption))
            .collect(),
        provenance_id: response.provenance_id.to_string(),
        identified_mass: structural.map(|mixture| mixture.identified_mass),
        unidentified_mass: structural.map(|mixture| mixture.unidentified_mass),
        unevaluable_mass: structural.map(|mixture| mixture.unevaluable_mass),
        subsampled_out_mass: structural.map(|mixture| mixture.subsampled_out_mass),
        completion_count: structural.map(|mixture| mixture.atoms.len()),
        truncated_completions: structural.map(|mixture| mixture.truncated_atoms),
        enumeration_capped: structural.map(|mixture| !mixture.full_mass_scope),
        mass_scope: structural.map(|mixture| {
            if mixture.full_mass_scope { "full_class" } else { "examined_completions" }.into()
        }),
        weight_basis: structural.map(|mixture| mixture.weight_basis.as_str().into()),
        aggregation_policy: structural
            .and_then(|mixture| mixture.aggregation_policy)
            .map(|policy| policy.as_str().to_owned()),
        atom_keys: structural
            .map(|mixture| mixture.atoms.iter().map(|atom| atom.graph_key).collect())
            .unwrap_or_default(),
        atom_weights: structural
            .map(|mixture| mixture.atoms.iter().map(|atom| atom.weight).collect())
            .unwrap_or_default(),
        atom_statuses: structural
            .map(|mixture| mixture.atoms.iter().map(|atom| format!("{:?}", atom.status)).collect())
            .unwrap_or_default(),
        atom_values: structural
            .map(|mixture| {
                mixture.atoms.iter().map(|atom| response_value_flat(atom.value.as_ref())).collect()
            })
            .unwrap_or_default(),
        evidence_status,
        allowlist_reason,
        allowlist_parent,
        diagnostics: Vec::new(),
        identifier: None,
        missing_data_json: None,
        interval_bindings_json: None,
        coordinates,
    })
}

fn response_value_flat(value: Option<&ResponseValue>) -> Vec<f64> {
    match value {
        Some(ResponseValue::Scalar(value)) => vec![*value],
        Some(ResponseValue::Surface { mean, .. }) => mean.to_vec(),
        Some(ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. }) => {
            values.to_vec()
        }
        Some(ResponseValue::Envelope(envelope)) => {
            envelope.lower.iter().chain(envelope.upper.iter()).copied().collect()
        }
        None => Vec::new(),
    }
}

fn structural_weighted_values(
    mixture: &antecedent::result::StructuralResponseMixture,
) -> Option<Vec<f64>> {
    let mut total_weight = 0.0;
    let mut mixed = Vec::new();
    for atom in &mixture.atoms {
        let values = response_value_flat(atom.value.as_ref());
        if values.is_empty() || atom.weight <= 0.0 {
            continue;
        }
        if mixed.is_empty() {
            mixed.resize(values.len(), 0.0);
        }
        if values.len() != mixed.len() {
            return None;
        }
        total_weight += atom.weight;
        for (target, value) in mixed.iter_mut().zip(values) {
            *target += atom.weight * value;
        }
    }
    (total_weight > 0.0).then(|| {
        for value in &mut mixed {
            *value /= total_weight;
        }
        mixed
    })
}

pub(crate) fn attach_study_response_meta(
    mut mapped: ResponseAnalysisResult,
    certificate_json: Option<String>,
    identifier: Option<String>,
    diagnostics: Vec<String>,
    missing_data_json: Option<String>,
    interval_bindings_json: Option<String>,
) -> ResponseAnalysisResult {
    mapped.missing_data_json = missing_data_json;
    mapped.interval_bindings_json = interval_bindings_json;
    mapped.certificate_json = certificate_json;
    mapped.identifier = identifier;
    mapped.diagnostics = diagnostics;
    mapped
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

type ValueParts = (Vec<Vec<f64>>, Vec<Vec<f64>>, Option<f64>, Option<Vec<Vec<f64>>>);

fn validate_surface_shape(grid: usize, dimension: usize, means: usize) -> Result<(), &'static str> {
    if dimension == 0 || grid % dimension != 0 || grid / dimension != means {
        return Err("invalid surface result shape");
    }
    Ok(())
}

fn validate_jacobian_shape(
    outcomes: usize,
    treatments: usize,
    values: usize,
) -> Result<(), &'static str> {
    if treatments == 0 || outcomes.checked_mul(treatments) != Some(values) {
        return Err("invalid Jacobian result shape");
    }
    Ok(())
}

fn value_parts(value: ResponseValue) -> PyResult<ValueParts> {
    match value {
        ResponseValue::Scalar(value) => Ok((vec![vec![0.0]], vec![vec![value]], Some(value), None)),
        ResponseValue::Surface { grid, dimension, mean } => {
            validate_surface_shape(grid.len(), dimension, mean.len()).map_err(crate::value_err)?;
            let points = grid.chunks(dimension).map(<[f64]>::to_vec).collect::<Vec<_>>();
            let values = mean.iter().map(|value| vec![*value]).collect();
            Ok((points, values, None, None))
        }
        ResponseValue::Vector(values) => {
            Ok((Vec::new(), Vec::new(), None, Some(vec![values.to_vec()])))
        }
        ResponseValue::Jacobian { outcomes, treatments, values } => {
            validate_jacobian_shape(outcomes, treatments, values.len())
                .map_err(crate::value_err)?;
            Ok((
                Vec::new(),
                Vec::new(),
                None,
                Some(values.chunks(treatments).map(<[f64]>::to_vec).collect()),
            ))
        }
        ResponseValue::Envelope(_) => Err(crate::value_err(
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
    Option<String>,
);

fn uncertainty_parts(value: ResponseUncertainty) -> UncertaintyParts {
    let rows = |values: Arc<[f64]>| values.iter().map(|value| vec![*value]).collect();
    let tag = |interpretation: IntervalInterpretation| Some(interpretation.as_str().to_owned());
    match value {
        ResponseUncertainty::None => ("none".into(), None, None, None, None, None, None, None),
        ResponseUncertainty::Scalar {
            standard_error, level, lower, upper, interpretation, ..
        } => (
            "pointwise".into(),
            Some(vec![vec![lower]]),
            Some(vec![vec![upper]]),
            Some(level),
            Some(standard_error),
            None,
            None,
            tag(interpretation),
        ),
        ResponseUncertainty::PointwiseBand { level, lower, upper, interpretation, .. } => (
            "pointwise".into(),
            Some(rows(lower)),
            Some(rows(upper)),
            Some(level),
            None,
            None,
            None,
            tag(interpretation),
        ),
        ResponseUncertainty::SimultaneousBand {
            level,
            lower,
            upper,
            replicates,
            interpretation,
        } => (
            "simultaneous".into(),
            Some(rows(lower)),
            Some(rows(upper)),
            Some(level),
            None,
            Some(replicates),
            None,
            tag(interpretation),
        ),
        ResponseUncertainty::IdentifiedEnvelopeBand {
            level,
            lower_outer,
            upper_outer,
            interpretation,
        } => (
            "identified_set".into(),
            Some(rows(lower_outer)),
            Some(rows(upper_outer)),
            Some(level),
            None,
            None,
            None,
            tag(interpretation),
        ),
        ResponseUncertainty::Posterior { artifact_id } => {
            ("posterior".into(), None, None, None, None, None, Some(artifact_id.to_string()), None)
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResponseAnalysisResult>()?;
    Ok(())
}

#[cfg(test)]
mod response_value_shape_tests {
    use super::{validate_jacobian_shape, validate_surface_shape};

    #[test]
    fn malformed_shapes_are_refused_before_chunking() {
        assert!(validate_surface_shape(0, 0, 0).is_err());
        assert!(validate_surface_shape(3, 2, 2).is_err());
        assert!(validate_surface_shape(4, 2, 1).is_err());
        assert!(validate_surface_shape(4, 2, 2).is_ok());
        assert!(validate_jacobian_shape(1, 0, 0).is_err());
        assert!(validate_jacobian_shape(usize::MAX, 2, 0).is_err());
        assert!(validate_jacobian_shape(2, 3, 5).is_err());
        assert!(validate_jacobian_shape(2, 3, 6).is_ok());
    }
}
