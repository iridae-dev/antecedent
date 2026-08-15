//! Durable wire forms for causal-response queries and results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Field meanings intentionally mirror the fully documented domain types. Keeping the
// serde representation visually compact makes format review substantially easier.
#![allow(missing_docs)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, ContinuousDomain, DerivativeScale, DerivativeWeighting,
    GridSpec, IdentificationStatus, ObservationAssumption, ObservationSpec, ParametricAssumption,
    PriorAssumption, ResponseEnvelope, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion, SupportReport,
    SupportStatus, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::analysis_wire::{DiagnosticWire, diagnostic_from_wire, diagnostic_to_wire};
use crate::error::IoError;
use crate::query_wire::{InterventionWire, TargetPopulationWire};
use crate::trace::{AssumptionRecordWire, AssumptionTagWire, assumptions_to_wire};

/// Evaluation grid on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GridSpecWire {
    /// Explicit grid values.
    Values(Vec<f64>),
    /// Inclusive evenly spaced grid.
    Linspace {
        /// First point.
        start: f64,
        /// Last point.
        end: f64,
        /// Number of points.
        points: u64,
    },
}

/// Scalar continuous intervention domain on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContinuousDomainWire {
    /// Intervened variable raw id.
    pub variable: u32,
    /// Evaluation grid.
    pub grid: GridSpecWire,
}

/// Average-derivative weighting law on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeWeightingWire {
    /// Observed treatment/covariate law.
    Observed,
    /// Uniform law on an interval.
    Uniform {
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
    },
    /// Caller-supplied row weights.
    Custom(Vec<f64>),
}

/// Derivative reporting scale on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeScaleWire {
    /// `dm/da`.
    Identity,
    /// `a * dm/da`.
    LogTreatment,
    /// `(1/m) * dm/da`.
    LogOutcome,
    /// `(a/m) * dm/da`.
    LogLog,
}

/// Observation mechanism on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSpecWire {
    /// Complete observation.
    Complete,
    /// Right-censored outcome.
    RightCensored { latent: u32, observed: u32, censoring: u32, event: u32 },
    /// Left-censored outcome.
    LeftCensored { latent: u32, observed: u32, censoring: u32, event: u32 },
    /// Interval-censored outcome.
    IntervalCensored { latent: u32, lower: u32, upper: u32 },
    /// Truncated sampling.
    Truncated { latent: u32, observed: u32, lower: Option<u32>, upper: Option<u32> },
    /// Indicator-selected outcome.
    Selected { latent: u32, observed: u32, indicator: u32 },
}

/// Explicit observation assumption on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationAssumptionWire {
    /// Conditional observation/censoring independence.
    IndependentGiven(Vec<u32>),
    /// Conditional independence from the latent outcome.
    OutcomeIndependentGiven(Vec<u32>),
    /// Named structural observation model.
    Structural(String),
}

/// Response functional on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFunctionalWire {
    /// Mean response curve.
    MeanCurve { outcome: u32, treatment: ContinuousDomainWire },
    /// Average derivative.
    AverageDerivative { outcome: u32, treatment: u32, weighting: DerivativeWeightingWire },
    /// Point derivative.
    PointDerivative { outcome: u32, treatment: u32, at: f64, order: u8, scale: DerivativeScaleWire },
    /// Directional derivative.
    DirectionalDerivative {
        outcomes: Vec<u32>,
        treatments: Vec<u32>,
        at: Vec<f64>,
        direction: Vec<f64>,
    },
    /// Response Jacobian.
    Jacobian { outcomes: Vec<u32>, treatments: Vec<u32>, at: Vec<f64>, scale: DerivativeScaleWire },
    /// Mean under an existing intervention set.
    InterventionResponse { outcome: u32, interventions: Vec<InterventionWire> },
}

/// Complete response query on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResponseQueryWire {
    /// Requested functional.
    pub functional: ResponseFunctionalWire,
    /// Target population.
    pub target_population: TargetPopulationWire,
    /// Observation mechanism.
    pub observation: ObservationSpecWire,
    /// Explicit observation assumptions.
    #[serde(default)]
    pub observation_assumptions: Vec<ObservationAssumptionWire>,
}

/// Encode a response query.
///
/// # Errors
///
/// Unsupported intervention or target-population fields.
pub fn response_query_to_wire(q: &ResponseQuery) -> Result<ResponseQueryWire, IoError> {
    Ok(ResponseQueryWire {
        functional: functional_to_wire(&q.functional)?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        observation: observation_to_wire(&q.observation),
        observation_assumptions: q
            .observation_assumptions
            .iter()
            .map(observation_assumption_to_wire)
            .collect(),
    })
}

/// Decode and validate a response query.
///
/// # Errors
///
/// Invalid dimensions, values, interventions, or platform-size overflows.
pub fn response_query_from_wire(w: &ResponseQueryWire) -> Result<ResponseQuery, IoError> {
    let q = ResponseQuery {
        functional: functional_from_wire(&w.functional)?,
        target_population: w.target_population.to_domain()?,
        observation: observation_from_wire(&w.observation),
        observation_assumptions: w
            .observation_assumptions
            .iter()
            .map(observation_assumption_from_wire)
            .collect::<Vec<_>>()
            .into(),
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(q)
}

fn grid_to_wire(grid: &GridSpec) -> GridSpecWire {
    match grid {
        GridSpec::Values(values) => GridSpecWire::Values(values.to_vec()),
        GridSpec::Linspace { start, end, points } => GridSpecWire::Linspace {
            start: *start,
            end: *end,
            points: u64::try_from(*points).unwrap_or(u64::MAX),
        },
    }
}

pub(crate) fn grid_from_wire(grid: &GridSpecWire) -> Result<GridSpec, IoError> {
    Ok(match grid {
        GridSpecWire::Values(values) => GridSpec::Values(values.clone().into()),
        GridSpecWire::Linspace { start, end, points } => GridSpec::Linspace {
            start: *start,
            end: *end,
            points: usize::try_from(*points).map_err(|_| IoError::TooLarge)?,
        },
    })
}

fn scale_to_wire(scale: DerivativeScale) -> DerivativeScaleWire {
    match scale {
        DerivativeScale::Identity => DerivativeScaleWire::Identity,
        DerivativeScale::LogTreatment => DerivativeScaleWire::LogTreatment,
        DerivativeScale::LogOutcome => DerivativeScaleWire::LogOutcome,
        DerivativeScale::LogLog => DerivativeScaleWire::LogLog,
    }
}

fn scale_from_wire(scale: DerivativeScaleWire) -> DerivativeScale {
    match scale {
        DerivativeScaleWire::Identity => DerivativeScale::Identity,
        DerivativeScaleWire::LogTreatment => DerivativeScale::LogTreatment,
        DerivativeScaleWire::LogOutcome => DerivativeScale::LogOutcome,
        DerivativeScaleWire::LogLog => DerivativeScale::LogLog,
    }
}

fn functional_to_wire(f: &ResponseFunctional) -> Result<ResponseFunctionalWire, IoError> {
    Ok(match f {
        ResponseFunctional::MeanCurve { outcome, treatment } => ResponseFunctionalWire::MeanCurve {
            outcome: outcome.raw(),
            treatment: ContinuousDomainWire {
                variable: treatment.variable.raw(),
                grid: grid_to_wire(&treatment.grid),
            },
        },
        ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
            ResponseFunctionalWire::AverageDerivative {
                outcome: outcome.raw(),
                treatment: treatment.raw(),
                weighting: match weighting {
                    DerivativeWeighting::Observed => DerivativeWeightingWire::Observed,
                    DerivativeWeighting::Uniform { lower, upper } => {
                        DerivativeWeightingWire::Uniform { lower: *lower, upper: *upper }
                    }
                    DerivativeWeighting::Custom(weights) => {
                        DerivativeWeightingWire::Custom(weights.to_vec())
                    }
                },
            }
        }
        ResponseFunctional::PointDerivative { outcome, treatment, at, order, scale } => {
            ResponseFunctionalWire::PointDerivative {
                outcome: outcome.raw(),
                treatment: treatment.raw(),
                at: *at,
                order: *order,
                scale: scale_to_wire(*scale),
            }
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
            ResponseFunctionalWire::DirectionalDerivative {
                outcomes: outcomes.iter().map(|v| v.raw()).collect(),
                treatments: treatments.iter().map(|v| v.raw()).collect(),
                at: at.to_vec(),
                direction: direction.to_vec(),
            }
        }
        ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
            ResponseFunctionalWire::Jacobian {
                outcomes: outcomes.iter().map(|v| v.raw()).collect(),
                treatments: treatments.iter().map(|v| v.raw()).collect(),
                at: at.to_vec(),
                scale: scale_to_wire(*scale),
            }
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            ResponseFunctionalWire::InterventionResponse {
                outcome: outcome.raw(),
                interventions: interventions
                    .iter()
                    .map(InterventionWire::from_domain)
                    .collect::<Result<Vec<_>, _>>()?,
            }
        }
    })
}

fn functional_from_wire(f: &ResponseFunctionalWire) -> Result<ResponseFunctional, IoError> {
    Ok(match f {
        ResponseFunctionalWire::MeanCurve { outcome, treatment } => ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(*outcome),
            treatment: ContinuousDomain {
                variable: VariableId::from_raw(treatment.variable),
                grid: grid_from_wire(&treatment.grid)?,
            },
        },
        ResponseFunctionalWire::AverageDerivative { outcome, treatment, weighting } => {
            ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(*outcome),
                treatment: VariableId::from_raw(*treatment),
                weighting: match weighting {
                    DerivativeWeightingWire::Observed => DerivativeWeighting::Observed,
                    DerivativeWeightingWire::Uniform { lower, upper } => {
                        DerivativeWeighting::Uniform { lower: *lower, upper: *upper }
                    }
                    DerivativeWeightingWire::Custom(weights) => {
                        DerivativeWeighting::Custom(weights.clone().into())
                    }
                },
            }
        }
        ResponseFunctionalWire::PointDerivative { outcome, treatment, at, order, scale } => {
            ResponseFunctional::PointDerivative {
                outcome: VariableId::from_raw(*outcome),
                treatment: VariableId::from_raw(*treatment),
                at: *at,
                order: *order,
                scale: scale_from_wire(*scale),
            }
        }
        ResponseFunctionalWire::DirectionalDerivative { outcomes, treatments, at, direction } => {
            ResponseFunctional::DirectionalDerivative {
                outcomes: outcomes
                    .iter()
                    .copied()
                    .map(VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                treatments: treatments
                    .iter()
                    .copied()
                    .map(VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                at: at.clone().into(),
                direction: direction.clone().into(),
            }
        }
        ResponseFunctionalWire::Jacobian { outcomes, treatments, at, scale } => {
            ResponseFunctional::Jacobian {
                outcomes: outcomes
                    .iter()
                    .copied()
                    .map(VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                treatments: treatments
                    .iter()
                    .copied()
                    .map(VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                at: at.clone().into(),
                scale: scale_from_wire(*scale),
            }
        }
        ResponseFunctionalWire::InterventionResponse { outcome, interventions } => {
            ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(*outcome),
                interventions: interventions
                    .iter()
                    .map(InterventionWire::to_domain)
                    .collect::<Vec<_>>()
                    .into(),
            }
        }
    })
}

fn observation_to_wire(o: &ObservationSpec) -> ObservationSpecWire {
    match o {
        ObservationSpec::Complete => ObservationSpecWire::Complete,
        ObservationSpec::RightCensored { latent, observed, censoring, event } => {
            ObservationSpecWire::RightCensored {
                latent: latent.raw(),
                observed: observed.raw(),
                censoring: censoring.raw(),
                event: event.raw(),
            }
        }
        ObservationSpec::LeftCensored { latent, observed, censoring, event } => {
            ObservationSpecWire::LeftCensored {
                latent: latent.raw(),
                observed: observed.raw(),
                censoring: censoring.raw(),
                event: event.raw(),
            }
        }
        ObservationSpec::IntervalCensored { latent, lower, upper } => {
            ObservationSpecWire::IntervalCensored {
                latent: latent.raw(),
                lower: lower.raw(),
                upper: upper.raw(),
            }
        }
        ObservationSpec::Truncated { latent, observed, lower, upper } => {
            ObservationSpecWire::Truncated {
                latent: latent.raw(),
                observed: observed.raw(),
                lower: lower.map(VariableId::raw),
                upper: upper.map(VariableId::raw),
            }
        }
        ObservationSpec::Selected { latent, observed, indicator } => {
            ObservationSpecWire::Selected {
                latent: latent.raw(),
                observed: observed.raw(),
                indicator: indicator.raw(),
            }
        }
    }
}

fn observation_from_wire(o: &ObservationSpecWire) -> ObservationSpec {
    match o {
        ObservationSpecWire::Complete => ObservationSpec::Complete,
        ObservationSpecWire::RightCensored { latent, observed, censoring, event } => {
            ObservationSpec::RightCensored {
                latent: VariableId::from_raw(*latent),
                observed: VariableId::from_raw(*observed),
                censoring: VariableId::from_raw(*censoring),
                event: VariableId::from_raw(*event),
            }
        }
        ObservationSpecWire::LeftCensored { latent, observed, censoring, event } => {
            ObservationSpec::LeftCensored {
                latent: VariableId::from_raw(*latent),
                observed: VariableId::from_raw(*observed),
                censoring: VariableId::from_raw(*censoring),
                event: VariableId::from_raw(*event),
            }
        }
        ObservationSpecWire::IntervalCensored { latent, lower, upper } => {
            ObservationSpec::IntervalCensored {
                latent: VariableId::from_raw(*latent),
                lower: VariableId::from_raw(*lower),
                upper: VariableId::from_raw(*upper),
            }
        }
        ObservationSpecWire::Truncated { latent, observed, lower, upper } => {
            ObservationSpec::Truncated {
                latent: VariableId::from_raw(*latent),
                observed: VariableId::from_raw(*observed),
                lower: lower.map(VariableId::from_raw),
                upper: upper.map(VariableId::from_raw),
            }
        }
        ObservationSpecWire::Selected { latent, observed, indicator } => {
            ObservationSpec::Selected {
                latent: VariableId::from_raw(*latent),
                observed: VariableId::from_raw(*observed),
                indicator: VariableId::from_raw(*indicator),
            }
        }
    }
}

fn observation_assumption_to_wire(a: &ObservationAssumption) -> ObservationAssumptionWire {
    match a {
        ObservationAssumption::IndependentGiven(v) => {
            ObservationAssumptionWire::IndependentGiven(v.iter().map(|x| x.raw()).collect())
        }
        ObservationAssumption::OutcomeIndependentGiven(v) => {
            ObservationAssumptionWire::OutcomeIndependentGiven(v.iter().map(|x| x.raw()).collect())
        }
        ObservationAssumption::Structural(id) => {
            ObservationAssumptionWire::Structural(id.to_string())
        }
    }
}

fn observation_assumption_from_wire(a: &ObservationAssumptionWire) -> ObservationAssumption {
    match a {
        ObservationAssumptionWire::IndependentGiven(v) => ObservationAssumption::IndependentGiven(
            v.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into(),
        ),
        ObservationAssumptionWire::OutcomeIndependentGiven(v) => {
            ObservationAssumption::OutcomeIndependentGiven(
                v.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into(),
            )
        }
        ObservationAssumptionWire::Structural(id) => {
            ObservationAssumption::Structural(Arc::from(id.as_str()))
        }
    }
}

/// Empirical support status on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportStatusWire {
    Supported,
    WeakOverlap,
    Extrapolative,
    OutsideEmpiricalSupport,
}

/// Region evaluated by support diagnostics on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SupportRegionWire {
    pub minima: Vec<f64>,
    pub maxima: Vec<f64>,
}

/// One support diagnostic on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SupportDiagnosticWire {
    pub id: String,
    pub values: Vec<f64>,
    pub detail: String,
}

/// Complete empirical support report on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SupportReportWire {
    pub status: SupportStatusWire,
    pub query_region: SupportRegionWire,
    pub diagnostics: Vec<SupportDiagnosticWire>,
    pub warnings: Vec<DiagnosticWire>,
}

/// Function-valued identified envelope on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResponseEnvelopeWire {
    pub grid: Vec<f64>,
    pub dimension: u64,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Numerical response payload on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseValueWire {
    Scalar(f64),
    Surface { grid: Vec<f64>, dimension: u64, mean: Vec<f64> },
    Vector(Vec<f64>),
    Jacobian { outcomes: u64, treatments: u64, values: Vec<f64> },
    Envelope(ResponseEnvelopeWire),
}

/// Statistical response uncertainty on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseUncertaintyWire {
    None,
    Scalar { standard_error: f64, level: f64, lower: f64, upper: f64 },
    PointwiseBand { level: f64, lower: Vec<f64>, upper: Vec<f64> },
    SimultaneousBand { level: f64, lower: Vec<f64>, upper: Vec<f64>, replicates: u32 },
    IdentifiedEnvelopeBand { level: f64, lower_outer: Vec<f64>, upper_outer: Vec<f64> },
    Posterior { artifact_id: String },
}

/// Structural identification payload on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseIdentificationWire {
    PointIdentified(ResponseValueWire),
    PartiallyIdentified(ResponseValueWire),
    GraphDependent(Vec<(u64, ResponseValueWire)>),
    Unidentified { certificate: String },
}

/// Identification status on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentificationStatusWire {
    NonparametricallyIdentified,
    IdentifiedUnderParametricRestrictions,
    IdentifiedUnderPriorRestrictions,
    PartiallyIdentified,
    GraphDependent,
    NotIdentified,
}

/// Complete causal-response artifact payload on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CausalResponseWire {
    pub estimand: ResponseFunctionalWire,
    pub identification_status: IdentificationStatusWire,
    pub estimate: ResponseIdentificationWire,
    pub uncertainty: ResponseUncertaintyWire,
    pub support: SupportReportWire,
    pub assumptions: Vec<AssumptionRecordWire>,
    pub provenance_id: String,
}

/// Encode a causal response artifact payload.
///
/// # Errors
///
/// Unsupported intervention fields in the estimand.
pub fn causal_response_to_wire(r: &CausalResponse) -> Result<CausalResponseWire, IoError> {
    Ok(CausalResponseWire {
        estimand: functional_to_wire(&r.estimand)?,
        identification_status: status_to_wire(r.identification_status),
        estimate: identification_to_wire(&r.estimate),
        uncertainty: uncertainty_to_wire(&r.uncertainty),
        support: support_to_wire(&r.support),
        assumptions: assumptions_to_wire(&r.assumptions),
        provenance_id: r.provenance_id.to_string(),
    })
}

/// Decode a causal response artifact payload.
///
/// # Errors
///
/// Invalid response dimensions, query values, or platform-size overflows.
pub fn causal_response_from_wire(w: &CausalResponseWire) -> Result<CausalResponse, IoError> {
    Ok(CausalResponse {
        estimand: functional_from_wire(&w.estimand)?,
        identification_status: status_from_wire(w.identification_status),
        estimate: identification_from_wire(&w.estimate)?,
        uncertainty: uncertainty_from_wire(&w.uncertainty),
        support: support_from_wire(&w.support)?,
        assumptions: assumptions_from_wire(&w.assumptions)?,
        provenance_id: Arc::from(w.provenance_id.as_str()),
    })
}

fn status_to_wire(s: IdentificationStatus) -> IdentificationStatusWire {
    match s {
        IdentificationStatus::NonparametricallyIdentified => {
            IdentificationStatusWire::NonparametricallyIdentified
        }
        IdentificationStatus::IdentifiedUnderParametricRestrictions => {
            IdentificationStatusWire::IdentifiedUnderParametricRestrictions
        }
        IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            IdentificationStatusWire::IdentifiedUnderPriorRestrictions
        }
        IdentificationStatus::PartiallyIdentified => IdentificationStatusWire::PartiallyIdentified,
        IdentificationStatus::GraphDependent => IdentificationStatusWire::GraphDependent,
        IdentificationStatus::NotIdentified => IdentificationStatusWire::NotIdentified,
    }
}
fn status_from_wire(s: IdentificationStatusWire) -> IdentificationStatus {
    match s {
        IdentificationStatusWire::NonparametricallyIdentified => {
            IdentificationStatus::NonparametricallyIdentified
        }
        IdentificationStatusWire::IdentifiedUnderParametricRestrictions => {
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        }
        IdentificationStatusWire::IdentifiedUnderPriorRestrictions => {
            IdentificationStatus::IdentifiedUnderPriorRestrictions
        }
        IdentificationStatusWire::PartiallyIdentified => IdentificationStatus::PartiallyIdentified,
        IdentificationStatusWire::GraphDependent => IdentificationStatus::GraphDependent,
        IdentificationStatusWire::NotIdentified => IdentificationStatus::NotIdentified,
    }
}

fn value_to_wire(v: &ResponseValue) -> ResponseValueWire {
    match v {
        ResponseValue::Scalar(x) => ResponseValueWire::Scalar(*x),
        ResponseValue::Surface { grid, dimension, mean } => ResponseValueWire::Surface {
            grid: grid.to_vec(),
            dimension: u64::try_from(*dimension).unwrap_or(u64::MAX),
            mean: mean.to_vec(),
        },
        ResponseValue::Vector(v) => ResponseValueWire::Vector(v.to_vec()),
        ResponseValue::Jacobian { outcomes, treatments, values } => ResponseValueWire::Jacobian {
            outcomes: u64::try_from(*outcomes).unwrap_or(u64::MAX),
            treatments: u64::try_from(*treatments).unwrap_or(u64::MAX),
            values: values.to_vec(),
        },
        ResponseValue::Envelope(e) => ResponseValueWire::Envelope(ResponseEnvelopeWire {
            grid: e.grid.to_vec(),
            dimension: u64::try_from(e.dimension).unwrap_or(u64::MAX),
            lower: e.lower.to_vec(),
            upper: e.upper.to_vec(),
        }),
    }
}

fn value_from_wire(v: &ResponseValueWire) -> Result<ResponseValue, IoError> {
    Ok(match v {
        ResponseValueWire::Scalar(x) => ResponseValue::Scalar(*x),
        ResponseValueWire::Surface { grid, dimension, mean } => ResponseValue::Surface {
            grid: grid.clone().into(),
            dimension: usize::try_from(*dimension).map_err(|_| IoError::TooLarge)?,
            mean: mean.clone().into(),
        },
        ResponseValueWire::Vector(v) => ResponseValue::Vector(v.clone().into()),
        ResponseValueWire::Jacobian { outcomes, treatments, values } => ResponseValue::Jacobian {
            outcomes: usize::try_from(*outcomes).map_err(|_| IoError::TooLarge)?,
            treatments: usize::try_from(*treatments).map_err(|_| IoError::TooLarge)?,
            values: values.clone().into(),
        },
        ResponseValueWire::Envelope(e) => ResponseValue::Envelope(ResponseEnvelope {
            grid: e.grid.clone().into(),
            dimension: usize::try_from(e.dimension).map_err(|_| IoError::TooLarge)?,
            lower: e.lower.clone().into(),
            upper: e.upper.clone().into(),
        }),
    })
}

fn identification_to_wire(i: &ResponseIdentification) -> ResponseIdentificationWire {
    match i {
        ResponseIdentification::PointIdentified(v) => {
            ResponseIdentificationWire::PointIdentified(value_to_wire(v))
        }
        ResponseIdentification::PartiallyIdentified(v) => {
            ResponseIdentificationWire::PartiallyIdentified(value_to_wire(v))
        }
        ResponseIdentification::GraphDependent(v) => ResponseIdentificationWire::GraphDependent(
            v.iter().map(|(id, value)| (*id, value_to_wire(value))).collect(),
        ),
        ResponseIdentification::Unidentified { certificate } => {
            ResponseIdentificationWire::Unidentified { certificate: certificate.to_string() }
        }
    }
}
fn identification_from_wire(
    i: &ResponseIdentificationWire,
) -> Result<ResponseIdentification, IoError> {
    Ok(match i {
        ResponseIdentificationWire::PointIdentified(v) => {
            ResponseIdentification::PointIdentified(value_from_wire(v)?)
        }
        ResponseIdentificationWire::PartiallyIdentified(v) => {
            ResponseIdentification::PartiallyIdentified(value_from_wire(v)?)
        }
        ResponseIdentificationWire::GraphDependent(v) => ResponseIdentification::GraphDependent(
            v.iter()
                .map(|(id, value)| Ok((*id, value_from_wire(value)?)))
                .collect::<Result<Vec<_>, IoError>>()?,
        ),
        ResponseIdentificationWire::Unidentified { certificate } => {
            ResponseIdentification::Unidentified { certificate: Arc::from(certificate.as_str()) }
        }
    })
}

fn uncertainty_to_wire(u: &ResponseUncertainty) -> ResponseUncertaintyWire {
    match u {
        ResponseUncertainty::None => ResponseUncertaintyWire::None,
        ResponseUncertainty::Scalar { standard_error, level, lower, upper } => {
            ResponseUncertaintyWire::Scalar {
                standard_error: *standard_error,
                level: *level,
                lower: *lower,
                upper: *upper,
            }
        }
        ResponseUncertainty::PointwiseBand { level, lower, upper } => {
            ResponseUncertaintyWire::PointwiseBand {
                level: *level,
                lower: lower.to_vec(),
                upper: upper.to_vec(),
            }
        }
        ResponseUncertainty::SimultaneousBand { level, lower, upper, replicates } => {
            ResponseUncertaintyWire::SimultaneousBand {
                level: *level,
                lower: lower.to_vec(),
                upper: upper.to_vec(),
                replicates: *replicates,
            }
        }
        ResponseUncertainty::IdentifiedEnvelopeBand { level, lower_outer, upper_outer } => {
            ResponseUncertaintyWire::IdentifiedEnvelopeBand {
                level: *level,
                lower_outer: lower_outer.to_vec(),
                upper_outer: upper_outer.to_vec(),
            }
        }
        ResponseUncertainty::Posterior { artifact_id } => {
            ResponseUncertaintyWire::Posterior { artifact_id: artifact_id.to_string() }
        }
    }
}
fn uncertainty_from_wire(u: &ResponseUncertaintyWire) -> ResponseUncertainty {
    match u {
        ResponseUncertaintyWire::None => ResponseUncertainty::None,
        ResponseUncertaintyWire::Scalar { standard_error, level, lower, upper } => {
            ResponseUncertainty::Scalar {
                standard_error: *standard_error,
                level: *level,
                lower: *lower,
                upper: *upper,
            }
        }
        ResponseUncertaintyWire::PointwiseBand { level, lower, upper } => {
            ResponseUncertainty::PointwiseBand {
                level: *level,
                lower: lower.clone().into(),
                upper: upper.clone().into(),
            }
        }
        ResponseUncertaintyWire::SimultaneousBand { level, lower, upper, replicates } => {
            ResponseUncertainty::SimultaneousBand {
                level: *level,
                lower: lower.clone().into(),
                upper: upper.clone().into(),
                replicates: *replicates,
            }
        }
        ResponseUncertaintyWire::IdentifiedEnvelopeBand { level, lower_outer, upper_outer } => {
            ResponseUncertainty::IdentifiedEnvelopeBand {
                level: *level,
                lower_outer: lower_outer.clone().into(),
                upper_outer: upper_outer.clone().into(),
            }
        }
        ResponseUncertaintyWire::Posterior { artifact_id } => {
            ResponseUncertainty::Posterior { artifact_id: Arc::from(artifact_id.as_str()) }
        }
    }
}

fn support_to_wire(s: &SupportReport) -> SupportReportWire {
    SupportReportWire {
        status: match s.status {
            SupportStatus::Supported => SupportStatusWire::Supported,
            SupportStatus::WeakOverlap => SupportStatusWire::WeakOverlap,
            SupportStatus::Extrapolative => SupportStatusWire::Extrapolative,
            SupportStatus::OutsideEmpiricalSupport => SupportStatusWire::OutsideEmpiricalSupport,
        },
        query_region: SupportRegionWire {
            minima: s.query_region.minima.to_vec(),
            maxima: s.query_region.maxima.to_vec(),
        },
        diagnostics: s
            .diagnostics
            .iter()
            .map(|d| SupportDiagnosticWire {
                id: d.id.to_string(),
                values: d.values.to_vec(),
                detail: d.detail.to_string(),
            })
            .collect(),
        warnings: s.warnings.iter().map(diagnostic_to_wire).collect(),
    }
}
fn support_from_wire(s: &SupportReportWire) -> Result<SupportReport, IoError> {
    Ok(SupportReport {
        status: match s.status {
            SupportStatusWire::Supported => SupportStatus::Supported,
            SupportStatusWire::WeakOverlap => SupportStatus::WeakOverlap,
            SupportStatusWire::Extrapolative => SupportStatus::Extrapolative,
            SupportStatusWire::OutsideEmpiricalSupport => SupportStatus::OutsideEmpiricalSupport,
        },
        query_region: SupportRegion {
            minima: s.query_region.minima.clone().into(),
            maxima: s.query_region.maxima.clone().into(),
        },
        diagnostics: s
            .diagnostics
            .iter()
            .map(|d| SupportDiagnostic {
                id: Arc::from(d.id.as_str()),
                values: d.values.clone().into(),
                detail: Arc::from(d.detail.as_str()),
            })
            .collect(),
        warnings: s.warnings.iter().map(diagnostic_from_wire).collect::<Result<Vec<_>, _>>()?,
    })
}

fn assumptions_from_wire(records: &[AssumptionRecordWire]) -> Result<AssumptionSet, IoError> {
    let mut set = AssumptionSet::new();
    for record in records {
        let assumption = match &record.assumption {
            AssumptionTagWire::CausalMarkov => Assumption::CausalMarkov,
            AssumptionTagWire::Faithfulness => Assumption::Faithfulness,
            AssumptionTagWire::CausalSufficiency => Assumption::CausalSufficiency,
            AssumptionTagWire::Consistency => Assumption::Consistency,
            AssumptionTagWire::Positivity => Assumption::Positivity,
            AssumptionTagWire::NoInterference => Assumption::NoInterference,
            AssumptionTagWire::Stationarity => Assumption::Stationarity,
            AssumptionTagWire::PiecewiseStationarity => Assumption::PiecewiseStationarity,
            AssumptionTagWire::NoSelectionBias => Assumption::NoSelectionBias,
            AssumptionTagWire::ExclusionRestriction { instrument } => {
                Assumption::ExclusionRestriction { instrument: VariableId::from_raw(*instrument) }
            }
            AssumptionTagWire::Monotonicity => Assumption::Monotonicity,
            AssumptionTagWire::ParametricRestriction { id } => {
                Assumption::ParametricRestriction(ParametricAssumption {
                    id: Arc::from(id.as_str()),
                    description: Arc::from(id.as_str()),
                })
            }
            AssumptionTagWire::PriorRestriction { id } => {
                Assumption::PriorRestriction(PriorAssumption {
                    id: Arc::from(id.as_str()),
                    description: Arc::from(id.as_str()),
                })
            }
            AssumptionTagWire::Custom { id } => Assumption::Custom {
                id: Arc::from(id.as_str()),
                description: Arc::from(id.as_str()),
            },
        };
        let source = if record.source == "user_declared" {
            AssumptionSource::UserDeclared
        } else if record.source == "artifact" {
            AssumptionSource::Artifact
        } else if let Some(v) = record.source.strip_prefix("algorithm_default:") {
            AssumptionSource::AlgorithmDefault { algorithm: Arc::from(v) }
        } else if let Some(v) = record.source.strip_prefix("derived:") {
            AssumptionSource::Derived { from: Arc::from(v) }
        } else {
            return Err(IoError::Convert(format!("unknown assumption source `{}`", record.source)));
        };
        let scope = match record.scope.as_str() {
            "global" => AssumptionScope::Global,
            "identification" => AssumptionScope::Identification,
            "estimation" => AssumptionScope::Estimation,
            "discovery" => AssumptionScope::Discovery,
            values if values.starts_with("variables:[") && values.ends_with(']') => {
                let raw = &values[11..values.len() - 1];
                let variables = if raw.is_empty() {
                    Vec::new()
                } else {
                    raw.split(',')
                        .map(|id| {
                            id.parse::<u32>().map(VariableId::from_raw).map_err(|e| {
                                IoError::Convert(format!(
                                    "invalid variable assumption scope `{values}`: {e}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                AssumptionScope::Variables { variables: variables.into() }
            }
            other => {
                return Err(IoError::Convert(format!("unsupported assumption scope `{other}`")));
            }
        };
        let status = match record.status.as_str() {
            "declared" => AssumptionStatus::Declared,
            "supported" => AssumptionStatus::Supported,
            "contradicted" => AssumptionStatus::Contradicted,
            "untestable" => AssumptionStatus::Untestable,
            other => return Err(IoError::Convert(format!("unknown assumption status `{other}`"))),
        };
        set.push(AssumptionRecord { assumption, source, scope, status });
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
        Diagnostic, DiagnosticKind, DiagnosticSeverity, Intervention, Value,
    };

    use super::*;
    use crate::convert::{from_cbor, to_cbor};

    fn round_trip_query(query: &ResponseQuery) {
        let wire = response_query_to_wire(query).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: ResponseQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(response_query_from_wire(&decoded).unwrap(), *query);
    }

    #[test]
    fn every_response_functional_round_trips() {
        let t = VariableId::from_raw(0);
        let t2 = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let y2 = VariableId::from_raw(3);
        let functionals = vec![
            ResponseFunctional::MeanCurve {
                outcome: y,
                treatment: ContinuousDomain::new(
                    t,
                    GridSpec::Linspace { start: 0.0, end: 2.0, points: 5 },
                ),
            },
            ResponseFunctional::AverageDerivative {
                outcome: y,
                treatment: t,
                weighting: DerivativeWeighting::Custom(Arc::from([0.25, 0.75])),
            },
            ResponseFunctional::PointDerivative {
                outcome: y,
                treatment: t,
                at: 1.5,
                order: 2,
                scale: DerivativeScale::LogLog,
            },
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([y, y2]),
                treatments: Arc::from([t, t2]),
                at: Arc::from([1.0, 2.0]),
                direction: Arc::from([0.5, -0.5]),
            },
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([y, y2]),
                treatments: Arc::from([t, t2]),
                at: Arc::from([1.0, 2.0]),
                scale: DerivativeScale::Identity,
            },
            ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from([
                    Intervention::set(t, Value::f64(1.0)),
                    Intervention::shift(t2, Value::f64(0.5)),
                ]),
            },
        ];
        for functional in functionals {
            round_trip_query(&ResponseQuery::new(functional));
        }
    }

    #[test]
    fn observation_query_and_causal_query_round_trip() {
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0, 3.0])),
            ),
        })
        .with_observation(
            ObservationSpec::RightCensored {
                latent: VariableId::from_raw(2),
                observed: VariableId::from_raw(3),
                censoring: VariableId::from_raw(4),
                event: VariableId::from_raw(5),
            },
            Arc::from([ObservationAssumption::IndependentGiven(Arc::from([
                VariableId::from_raw(6),
            ]))]),
        );
        round_trip_query(&query);

        let causal = antecedent_core::CausalQuery::Response(query);
        let wire = crate::query_wire::causal_query_to_wire(&causal).unwrap();
        let back = crate::query_wire::causal_query_from_wire(&wire).unwrap();
        assert_eq!(back, causal);
    }

    #[test]
    fn complete_response_artifact_round_trips() {
        let estimand = ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0, 2.0])),
            ),
        };
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Positivity,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Supported,
        });
        let response = CausalResponse {
            estimand,
            identification_status: IdentificationStatus::NonparametricallyIdentified,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from([0.0, 1.0, 2.0]),
                dimension: 1,
                mean: Arc::from([2.0, 2.5, 3.2]),
            }),
            uncertainty: ResponseUncertainty::SimultaneousBand {
                level: 0.95,
                lower: Arc::from([1.8, 2.2, 2.8]),
                upper: Arc::from([2.2, 2.8, 3.6]),
                replicates: 500,
            },
            support: SupportReport {
                status: SupportStatus::WeakOverlap,
                query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([2.0]) },
                diagnostics: vec![SupportDiagnostic {
                    id: Arc::from("local_ess"),
                    values: Arc::from([80.0, 60.0, 20.0]),
                    detail: Arc::from("effective sample size by grid point"),
                }],
                warnings: vec![Diagnostic::new(
                    "support.weak_overlap",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "overlap weak at upper endpoint",
                )],
            },
            assumptions,
            provenance_id: Arc::from("response-fit-17"),
        };
        let wire = causal_response_to_wire(&response).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalResponseWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_response_from_wire(&decoded).unwrap(), response);
    }
}
