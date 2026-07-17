//! Minimal wire forms for interventional-distribution and path-specific queries.
//!
//! Full `CausalQuery` artifact embedding remains deferred (serialization / TODO).
//! Only hard [`Intervention::Set`] and the built [`TargetPopulation`] variants
//! round-trip here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    EnvironmentId, Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery,
    TargetPopulation, Value, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Wire scalar value (subset of [`Value`]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ValueWire {
    /// Float64.
    Float64(f64),
    /// Int64.
    Int64(i64),
    /// Bool.
    Bool(bool),
    /// Category code.
    Category(u32),
    /// Diagnostic label.
    Label(String),
}

impl ValueWire {
    /// Encode a domain value.
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Float64(x) => Self::Float64(*x),
            Value::Int64(x) => Self::Int64(*x),
            Value::Bool(x) => Self::Bool(*x),
            Value::Category(x) => Self::Category(*x),
            Value::Label(s) => Self::Label(s.to_string()),
        }
    }

    /// Decode to domain value.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Float64(x) => Value::Float64(*x),
            Self::Int64(x) => Value::Int64(*x),
            Self::Bool(x) => Value::Bool(*x),
            Self::Category(x) => Value::Category(*x),
            Self::Label(s) => Value::Label(Arc::from(s.as_str())),
        }
    }
}

/// Hard set intervention on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SetInterventionWire {
    /// Target variable raw id.
    pub variable: u32,
    /// Assigned value.
    pub value: ValueWire,
}

/// Target population on the wire (built variants only).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetPopulationWire {
    /// All observed units.
    AllObserved,
    /// Treated units.
    Treated,
    /// Untreated units.
    Untreated,
    /// Environment-restricted.
    Environment(u32),
}

impl TargetPopulationWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// Never fails today; reserved for future variants.
    pub fn from_domain(p: &TargetPopulation) -> Result<Self, IoError> {
        Ok(match p {
            TargetPopulation::AllObserved => Self::AllObserved,
            TargetPopulation::Treated => Self::Treated,
            TargetPopulation::Untreated => Self::Untreated,
            TargetPopulation::Environment(id) => Self::Environment(id.raw()),
            other => {
                return Err(IoError::Convert(format!(
                    "unsupported TargetPopulation for query wire: {other:?}"
                )));
            }
        })
    }

    /// Decode.
    #[must_use]
    pub fn to_domain(&self) -> TargetPopulation {
        match self {
            Self::AllObserved => TargetPopulation::AllObserved,
            Self::Treated => TargetPopulation::Treated,
            Self::Untreated => TargetPopulation::Untreated,
            Self::Environment(raw) => TargetPopulation::Environment(EnvironmentId::from_raw(*raw)),
        }
    }
}

/// Wire form of [`InterventionalDistributionQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterventionalDistributionQueryWire {
    /// Outcome variable raw ids.
    pub outcomes: Vec<u32>,
    /// Hard-set interventions only.
    pub interventions: Vec<SetInterventionWire>,
    /// Target population.
    pub target_population: TargetPopulationWire,
}

/// Wire form of [`PathSpecificEffectQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PathSpecificEffectQueryWire {
    /// Treatment raw id.
    pub treatment: u32,
    /// Outcome raw id.
    pub outcome: u32,
    /// Intermediate path-node raw ids.
    pub path_nodes: Vec<u32>,
    /// Control hard set.
    pub control: SetInterventionWire,
    /// Active hard set.
    pub active: SetInterventionWire,
    /// Target population.
    pub target_population: TargetPopulationWire,
    /// Max paths.
    pub max_paths: u64,
    /// Max path length.
    pub max_len: u64,
}

fn set_to_wire(iv: &Intervention) -> Result<SetInterventionWire, IoError> {
    match iv {
        Intervention::Set { variable, value } => Ok(SetInterventionWire {
            variable: variable.raw(),
            value: ValueWire::from_value(value),
        }),
        other => Err(IoError::Convert(format!(
            "query wire only supports Intervention::Set; got {other:?}"
        ))),
    }
}

fn set_from_wire(w: &SetInterventionWire) -> Intervention {
    Intervention::set(VariableId::from_raw(w.variable), w.value.to_value())
}

/// Encode an interventional distribution query.
///
/// # Errors
///
/// Non-Set interventions or unsupported target population.
pub fn interventional_distribution_to_wire(
    q: &InterventionalDistributionQuery,
) -> Result<InterventionalDistributionQueryWire, IoError> {
    let mut interventions = Vec::with_capacity(q.interventions.len());
    for iv in q.interventions.iter() {
        interventions.push(set_to_wire(iv)?);
    }
    Ok(InterventionalDistributionQueryWire {
        outcomes: q.outcomes.iter().map(|v| v.raw()).collect(),
        interventions,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
    })
}

/// Decode an interventional distribution query.
#[must_use]
pub fn interventional_distribution_from_wire(
    w: &InterventionalDistributionQueryWire,
) -> InterventionalDistributionQuery {
    InterventionalDistributionQuery {
        outcomes: w.outcomes.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into(),
        interventions: w.interventions.iter().map(set_from_wire).collect::<Vec<_>>().into(),
        target_population: w.target_population.to_domain(),
    }
}

/// Encode a path-specific effect query.
///
/// # Errors
///
/// Non-Set interventions or unsupported target population.
pub fn path_specific_to_wire(
    q: &PathSpecificEffectQuery,
) -> Result<PathSpecificEffectQueryWire, IoError> {
    Ok(PathSpecificEffectQueryWire {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        path_nodes: q.path_nodes.iter().map(|v| v.raw()).collect(),
        control: set_to_wire(&q.control)?,
        active: set_to_wire(&q.active)?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        max_paths: u64::try_from(q.max_paths).unwrap_or(u64::MAX),
        max_len: u64::try_from(q.max_len).unwrap_or(u64::MAX),
    })
}

/// Decode a path-specific effect query.
///
/// # Errors
///
/// Limits that do not fit `usize` on this platform (extremely large wire values).
pub fn path_specific_from_wire(
    w: &PathSpecificEffectQueryWire,
) -> Result<PathSpecificEffectQuery, IoError> {
    let max_paths = usize::try_from(w.max_paths)
        .map_err(|_| IoError::Convert("max_paths does not fit usize".into()))?;
    let max_len = usize::try_from(w.max_len)
        .map_err(|_| IoError::Convert("max_len does not fit usize".into()))?;
    Ok(PathSpecificEffectQuery {
        treatment: VariableId::from_raw(w.treatment),
        outcome: VariableId::from_raw(w.outcome),
        path_nodes: w.path_nodes.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into(),
        control: set_from_wire(&w.control),
        active: set_from_wire(&w.active),
        target_population: w.target_population.to_domain(),
        max_paths,
        max_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{from_cbor, to_cbor};

    #[test]
    fn interventional_distribution_cbor_round_trip() {
        let q = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))],
        );
        let wire = interventional_distribution_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: InterventionalDistributionQueryWire = from_cbor(&bytes).unwrap();
        let back = interventional_distribution_from_wire(&decoded);
        assert_eq!(back.outcomes.as_ref(), q.outcomes.as_ref());
        assert_eq!(back.interventions.len(), 1);
        assert_eq!(back.target_population, TargetPopulation::AllObserved);
        back.validate().unwrap();
    }

    #[test]
    fn path_specific_cbor_round_trip() {
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)])
            .with_max_paths(32)
            .with_max_len(8);
        let wire = path_specific_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: PathSpecificEffectQueryWire = from_cbor(&bytes).unwrap();
        let back = path_specific_from_wire(&decoded).unwrap();
        assert_eq!(back.treatment, q.treatment);
        assert_eq!(back.outcome, q.outcome);
        assert_eq!(back.path_nodes.as_ref(), q.path_nodes.as_ref());
        assert_eq!(back.max_paths, 32);
        assert_eq!(back.max_len, 8);
        back.validate().unwrap();
    }
}
