//! Assumption and derivation wire types for analysis artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ParametricAssumption, PriorAssumption, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Wire form of an assumption tag .
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssumptionTagWire {
    /// Causal Markov.
    CausalMarkov,
    /// Faithfulness.
    Faithfulness,
    /// Causal sufficiency.
    CausalSufficiency,
    /// Consistency.
    Consistency,
    /// Positivity.
    Positivity,
    /// No interference.
    NoInterference,
    /// Temporal stationarity.
    Stationarity,
    /// Piecewise stationarity.
    PiecewiseStationarity,
    /// No selection bias.
    NoSelectionBias,
    /// Instrument exclusion restriction (`instrument=<raw id>`).
    ExclusionRestriction {
        /// Instrument variable raw id.
        instrument: u32,
    },
    /// Monotonicity.
    Monotonicity,
    /// Parametric modeling restriction (`id=<stable id>`).
    ParametricRestriction {
        /// Restriction family id.
        id: String,
        /// Human-readable restriction description (absent in pre-0.4 payloads).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// Prior / Bayesian restriction (`id=<stable id>`).
    PriorRestriction {
        /// Prior family id.
        id: String,
        /// Human-readable restriction description (absent in pre-0.4 payloads).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// Custom extension (`id=<stable id>`).
    Custom {
        /// Stable custom id.
        id: String,
        /// Human-readable assumption description (absent in pre-0.4 payloads).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// One assumption record on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssumptionRecordWire {
    /// Assumption tag.
    pub assumption: AssumptionTagWire,
    /// Source label (e.g. `algorithm_default:backdoor`).
    pub source: String,
    /// Scope label (e.g. `identification`).
    pub scope: String,
    /// Status label (e.g. `declared`, `untestable`).
    pub status: String,
}

/// One derivation step on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivationStepWire {
    /// Rule id.
    pub rule: String,
    /// Detail text.
    pub detail: String,
}

/// Analysis identification/estimation trace embedded in artifacts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisTraceWire {
    /// Required assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Derivation steps.
    pub derivation: Vec<DerivationStepWire>,
    /// Estimand method tag.
    pub method: String,
    /// Adjustment set as dense variable indices.
    pub adjustment_set: Vec<u32>,
    /// `licensed` or `allowed_unlicensed` when the producing study was classified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub support_status: Option<String>,
    /// Allowlist reason when `support_status` is `allowed_unlicensed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowlist_reason: Option<String>,
    /// Allowlist parent family when `support_status` is `allowed_unlicensed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowlist_parent: Option<String>,
}

/// Convert an [`AssumptionSet`] to wire records.
#[must_use]
pub fn assumptions_to_wire(set: &AssumptionSet) -> Vec<AssumptionRecordWire> {
    set.entries.iter().map(assumption_record_to_wire).collect()
}

fn assumption_record_to_wire(record: &AssumptionRecord) -> AssumptionRecordWire {
    AssumptionRecordWire {
        assumption: assumption_to_tag(&record.assumption),
        source: source_label(&record.source),
        scope: scope_label(&record.scope),
        status: status_label(record.status),
    }
}

fn assumption_to_tag(a: &Assumption) -> AssumptionTagWire {
    match a {
        Assumption::CausalMarkov => AssumptionTagWire::CausalMarkov,
        Assumption::Faithfulness => AssumptionTagWire::Faithfulness,
        Assumption::CausalSufficiency => AssumptionTagWire::CausalSufficiency,
        Assumption::Consistency => AssumptionTagWire::Consistency,
        Assumption::Positivity => AssumptionTagWire::Positivity,
        Assumption::NoInterference => AssumptionTagWire::NoInterference,
        Assumption::Stationarity => AssumptionTagWire::Stationarity,
        Assumption::PiecewiseStationarity => AssumptionTagWire::PiecewiseStationarity,
        Assumption::NoSelectionBias => AssumptionTagWire::NoSelectionBias,
        Assumption::ExclusionRestriction { instrument } => {
            AssumptionTagWire::ExclusionRestriction { instrument: instrument.raw() }
        }
        Assumption::Monotonicity => AssumptionTagWire::Monotonicity,
        Assumption::ParametricRestriction(p) => AssumptionTagWire::ParametricRestriction {
            id: p.id.to_string(),
            description: Some(p.description.to_string()),
        },
        Assumption::PriorRestriction(p) => AssumptionTagWire::PriorRestriction {
            id: p.id.to_string(),
            description: Some(p.description.to_string()),
        },
        Assumption::Custom { id, description } => AssumptionTagWire::Custom {
            id: id.to_string(),
            description: Some(description.to_string()),
        },
    }
}

/// Reconstruct an assumption set without dropping source, scope, status, or descriptions.
///
/// # Errors
///
/// Returns an error for unknown labels or malformed variable scopes.
pub fn assumptions_from_wire(records: &[AssumptionRecordWire]) -> Result<AssumptionSet, IoError> {
    let mut set = AssumptionSet::new();
    for record in records {
        match &record.assumption {
            AssumptionTagWire::ParametricRestriction { id, description }
            | AssumptionTagWire::PriorRestriction { id, description }
            | AssumptionTagWire::Custom { id, description }
                if id.trim().is_empty()
                    || description.as_ref().is_some_and(|value| value.trim().is_empty()) =>
            {
                return Err(IoError::Convert(
                    "named assumptions require a non-blank id and description".into(),
                ));
            }
            _ => {}
        }
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
            AssumptionTagWire::ParametricRestriction { id, description } => {
                Assumption::ParametricRestriction(ParametricAssumption {
                    id: Arc::from(id.as_str()),
                    description: Arc::from(description.as_deref().unwrap_or(id)),
                })
            }
            AssumptionTagWire::PriorRestriction { id, description } => {
                Assumption::PriorRestriction(PriorAssumption {
                    id: Arc::from(id.as_str()),
                    description: Arc::from(description.as_deref().unwrap_or(id)),
                })
            }
            AssumptionTagWire::Custom { id, description } => Assumption::Custom {
                id: Arc::from(id.as_str()),
                description: Arc::from(description.as_deref().unwrap_or(id)),
            },
        };
        let source = if record.source == "user_declared" {
            AssumptionSource::UserDeclared
        } else if record.source == "artifact" {
            AssumptionSource::Artifact
        } else if let Some(value) = record.source.strip_prefix("algorithm_default:") {
            AssumptionSource::AlgorithmDefault { algorithm: Arc::from(value) }
        } else if let Some(value) = record.source.strip_prefix("derived:") {
            AssumptionSource::Derived { from: Arc::from(value) }
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
                            id.parse::<u32>().map(VariableId::from_raw).map_err(|error| {
                                IoError::Convert(format!(
                                    "invalid variable assumption scope `{values}`: {error}"
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

fn source_label(s: &AssumptionSource) -> String {
    match s {
        AssumptionSource::UserDeclared => "user_declared".into(),
        AssumptionSource::AlgorithmDefault { algorithm } => {
            format!("algorithm_default:{algorithm}")
        }
        AssumptionSource::Artifact => "artifact".into(),
        AssumptionSource::Derived { from } => format!("derived:{from}"),
    }
}

fn scope_label(s: &AssumptionScope) -> String {
    match s {
        AssumptionScope::Global => "global".into(),
        AssumptionScope::Identification => "identification".into(),
        AssumptionScope::Estimation => "estimation".into(),
        AssumptionScope::Discovery => "discovery".into(),
        AssumptionScope::Variables { variables } => {
            let ids: Vec<String> = variables.iter().map(|v| v.raw().to_string()).collect();
            format!("variables:[{}]", ids.join(","))
        }
    }
}

fn status_label(s: AssumptionStatus) -> String {
    match s {
        AssumptionStatus::Declared => "declared".into(),
        AssumptionStatus::Supported => "supported".into(),
        AssumptionStatus::Contradicted => "contradicted".into(),
        AssumptionStatus::Untestable => "untestable".into(),
    }
}
