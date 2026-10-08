//! Scientific coordinates of a native response, read from the response itself.
//!
//! A response value is identified by its [`ScientificQuantity`], never by its
//! grid position. [`response_coordinate_skeleton`] derives, from a
//! [`CausalResponse`] alone, everything about each value coordinate that the
//! response determines: the outcome, the intervention regime, the horizon and the
//! functional. Units, population and scale are *declared by the caller* and are
//! never inferred, so [`ResponseCoordinateSkeleton::label`] and
//! [`response_coordinates`] take them explicitly.
//!
//! A response whose value cannot be given one honest coordinate per value is
//! refused with a typed `coordinate_support.*` detail instead of being scalarized
//! or relabeled. Nothing here estimates or rescales anything.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// A refusal is the cold path of a once-per-response derivation; boxing would not pay.
#![allow(clippy::result_large_err)]

use crate::{
    CausalResponse, ExternalRefusal, Intervention, QuantityRole, ResponseFunctional,
    ScientificQuantity, Value, VariableId, dose_label,
};

/// Everything a response fixes about one value coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseCoordinateSkeleton {
    /// Outcome variable name (also its stable identity, as in program bindings).
    pub variable: String,
    /// Exact intervention regime, for example `do(a=1)`.
    pub regime_id: String,
    /// Ordered causal steps to the outcome; zero on a static response.
    pub horizon: u32,
    /// Functional identity, for example `mean`.
    pub functional_id: String,
}

/// What the caller must declare about the response's values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResponseCoordinateLabels<'a> {
    /// Units of the outcome; never inferred or converted.
    pub outcome_units: &'a str,
    /// Stable identity of the population the response targets.
    pub population_id: &'a str,
    /// Declared scale of the outcome, for example `identity`.
    pub transform_id: &'a str,
}

impl ResponseCoordinateSkeleton {
    /// The full scientific coordinate under the declared units, population and scale.
    #[must_use]
    pub fn label(&self, labels: &ResponseCoordinateLabels<'_>) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: self.variable.clone(),
            variable_name: self.variable.clone(),
            role: QuantityRole::Outcome,
            units: labels.outcome_units.to_owned(),
            population_id: labels.population_id.to_owned(),
            regime_id: self.regime_id.clone(),
            horizon: self.horizon,
            functional_id: self.functional_id.clone(),
            conditioning: Vec::new(),
            transform_id: labels.transform_id.to_owned(),
        }
    }
}

fn refuse(
    code: &'static str,
    detail: &str,
    expected: Option<&str>,
    supplied: Option<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage: "bind",
        detail: detail.to_owned(),
        offending: None,
        expected: expected.map(str::to_owned),
        supplied,
        capability: None,
        remedy: Some(remedy),
    }
}

fn functional_name(functional: &ResponseFunctional) -> &'static str {
    match functional {
        ResponseFunctional::MeanCurve { .. } => "mean_curve",
        ResponseFunctional::AverageDerivative { .. } => "average_derivative",
        ResponseFunctional::PointDerivative { .. } => "point_derivative",
        ResponseFunctional::DirectionalDerivative { .. } => "directional_derivative",
        ResponseFunctional::Jacobian { .. } => "jacobian",
        ResponseFunctional::InterventionResponse { .. } => "intervention_response",
    }
}

fn named(
    id: VariableId,
    name_of: &dyn Fn(VariableId) -> Option<String>,
) -> Result<String, ExternalRefusal> {
    name_of(id).filter(|name| !name.trim().is_empty()).ok_or_else(|| {
        refuse(
            crate::reason_code!("unknown_variable"),
            "coordinate_support.unknown_variable",
            None,
            Some(id.raw().to_string()),
            "derive coordinates with the variable names the response was estimated on",
        )
    })
}

fn mean_curve_skeleton(
    response: &CausalResponse,
    name_of: &dyn Fn(VariableId) -> Option<String>,
) -> Result<Vec<ResponseCoordinateSkeleton>, ExternalRefusal> {
    let ResponseFunctional::MeanCurve { outcome, treatment } = &response.estimand else {
        return Ok(Vec::new());
    };
    let doses = treatment.grid.values().map_err(|error| {
        refuse(
            crate::reason_code!("invalid_argument"),
            "coordinate_support.invalid_grid",
            Some("a valid finite dose grid"),
            Some(error.to_string()),
            "request the response on a finite, strictly increasing dose grid",
        )
    })?;
    let variable = named(*outcome, name_of)?;
    let treatment_name = named(treatment.variable, name_of)?;
    // A temporal surface is dose-major like its mean: cell `d * n_horizons + h`.
    let horizons: Vec<u32> = match response.horizon_identification.as_deref() {
        Some(identified) if !identified.is_empty() => {
            identified.iter().map(|horizon| horizon.horizon).collect()
        }
        _ => vec![0],
    };
    let mut cells = Vec::with_capacity(doses.len() * horizons.len());
    for dose in &doses {
        for horizon in &horizons {
            cells.push(ResponseCoordinateSkeleton {
                variable: variable.clone(),
                regime_id: format!("do({treatment_name}={})", dose_label(*dose)),
                horizon: *horizon,
                functional_id: "mean".to_owned(),
            });
        }
    }
    Ok(cells)
}

fn not_describable(kind: &str) -> ExternalRefusal {
    refuse(
        crate::reason_code!("route_not_supported"),
        "coordinate_support.regime_not_describable",
        Some("hard set or additive shift to a finite numeric level"),
        Some(kind.to_owned()),
        "only a set or shift intervention at a finite numeric level names one exact regime",
    )
}

fn numeric_level(value: &Value, kind: &str) -> Result<String, ExternalRefusal> {
    let level = match value {
        Value::Float64(_) | Value::Int64(_) | Value::Bool(_) => {
            value.as_f64().filter(|level| level.is_finite())
        }
        Value::Category(_) | Value::Label(_) => None,
    };
    level.map(dose_label).ok_or_else(|| not_describable(kind))
}

fn regime_part(
    intervention: &Intervention,
    name_of: &dyn Fn(VariableId) -> Option<String>,
) -> Result<String, ExternalRefusal> {
    match intervention {
        Intervention::Set { variable, value } => {
            Ok(format!("do({}={})", named(*variable, name_of)?, numeric_level(value, "set")?))
        }
        Intervention::Shift { variable, delta } => {
            Ok(format!("shift({}={})", named(*variable, name_of)?, numeric_level(delta, "shift")?))
        }
        Intervention::Stochastic { .. } => Err(not_describable("stochastic")),
        Intervention::Soft { .. } => Err(not_describable("soft")),
        Intervention::Sequence(_) => Err(not_describable("sequence")),
    }
}

fn intervention_skeleton(
    response: &CausalResponse,
    name_of: &dyn Fn(VariableId) -> Option<String>,
) -> Result<Vec<ResponseCoordinateSkeleton>, ExternalRefusal> {
    let ResponseFunctional::InterventionResponse { outcome, interventions } = &response.estimand
    else {
        return Ok(Vec::new());
    };
    if response.horizon_identification.is_some() {
        return Err(refuse(
            crate::reason_code!("route_not_supported"),
            "coordinate_support.temporal_path_not_described",
            Some("a static intervention response"),
            Some("a horizon-indexed intervention path".to_owned()),
            "a temporal intervention path has one value per horizon; request a dose-by-horizon curve",
        ));
    }
    let parts = interventions
        .iter()
        .map(|intervention| regime_part(intervention, name_of))
        .collect::<Result<Vec<_>, _>>()?;
    // The value is one scalar answering the whole joint regime, so it has exactly
    // one coordinate; the regime names every intervention in query order.
    Ok(vec![ResponseCoordinateSkeleton {
        variable: named(*outcome, name_of)?,
        regime_id: parts.join(" & "),
        horizon: 0,
        functional_id: "mean".to_owned(),
    }])
}

/// The coordinates a response fixes, one per value, in value order.
///
/// A static mean curve has one coordinate per dose; a temporal dose-by-horizon
/// surface one per cell, dose-major like its mean; a static intervention response
/// (set or shift to finite levels, possibly joint) a single coordinate for its
/// scalar value.
///
/// # Errors
/// `route_not_supported` with `coordinate_support.functional_not_described` for a
/// derivative or Jacobian functional, `coordinate_support.regime_not_describable`
/// for a stochastic, soft, sequenced, categorical or symbolic intervention, and
/// `coordinate_support.temporal_path_not_described` for a horizon-indexed
/// intervention path; `unknown_variable` / `invalid_argument` for a variable the
/// caller cannot name or an invalid grid.
// A refusal is the cold path of a once-per-response derivation; boxing would not pay.
#[allow(clippy::result_large_err)]
pub fn response_coordinate_skeleton(
    response: &CausalResponse,
    name_of: &dyn Fn(VariableId) -> Option<String>,
) -> Result<Vec<ResponseCoordinateSkeleton>, ExternalRefusal> {
    match &response.estimand {
        ResponseFunctional::MeanCurve { .. } => mean_curve_skeleton(response, name_of),
        ResponseFunctional::InterventionResponse { .. } => intervention_skeleton(response, name_of),
        other => Err(refuse(
            crate::reason_code!("route_not_supported"),
            "coordinate_support.functional_not_described",
            Some("mean_curve or intervention_response"),
            Some(functional_name(other).to_owned()),
            "only a mean curve or a static intervention response is given scientific coordinates",
        )),
    }
}

/// Attach the declared units, population and scale to derived coordinates.
///
/// # Errors
/// `invalid_argument` with `coordinate_support.units_required` when the declared
/// units, population or scale are blank (units are never inferred or converted).
pub fn label_coordinates(
    cells: &[ResponseCoordinateSkeleton],
    labels: &ResponseCoordinateLabels<'_>,
) -> Result<Vec<ScientificQuantity>, ExternalRefusal> {
    if [labels.outcome_units, labels.population_id, labels.transform_id]
        .iter()
        .any(|field| field.trim().is_empty())
    {
        return Err(refuse(
            crate::reason_code!("invalid_argument"),
            "coordinate_support.units_required",
            Some("non-blank outcome units, population and transform"),
            None,
            "declare the outcome units; units are never inferred or converted",
        ));
    }
    Ok(cells.iter().map(|cell| cell.label(labels)).collect())
}

/// Full scientific coordinates of a response under the declared units, population
/// and scale.
///
/// # Errors
/// Everything [`label_coordinates`] and [`response_coordinate_skeleton`] refuse.
pub fn response_coordinates(
    response: &CausalResponse,
    name_of: &dyn Fn(VariableId) -> Option<String>,
    labels: &ResponseCoordinateLabels<'_>,
) -> Result<Vec<ScientificQuantity>, ExternalRefusal> {
    label_coordinates(&response_coordinate_skeleton(response, name_of)?, labels)
}
