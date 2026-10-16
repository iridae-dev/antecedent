//! Portable canonical queries of the single-source z, mixed-source and mz transport
//! routes.
//!
//! The artifacts that replay them need the estimators and live in `antecedent-io`;
//! a planner that only proposes a query needs no more than these.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{InterventionAssignment, VariableId};
use antecedent_identify::{
    MixedSourceQuery, MzTransportQuery, ZTransportQuery, ZTransportSourceSpec,
};
use serde::{Deserialize, Serialize};

use crate::query_wire::ValueWire;

/// Portable graph/query coordinates and exact intervention values.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatment variables.
    pub treatments: Vec<u32>,
    /// Controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment assignment.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
}

impl ZTransportQueryWire {
    /// Encode a query.
    #[must_use]
    pub fn from_query(query: &ZTransportQuery) -> Self {
        Self {
            outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: query.treatments.iter().map(|v| v.raw()).collect(),
            controllable: query.controllable.iter().map(|v| v.raw()).collect(),
            experiment_assignment: query
                .experiment_assignment
                .iter()
                .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                .collect(),
            source: query.source.to_string(),
            target: query.target.to_string(),
        }
    }

    /// Decode a query. Validation happens against the diagram it is used with.
    #[must_use]
    pub fn to_query(&self) -> ZTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        ZTransportQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            controllable: ids(&self.controllable).into(),
            experiment_assignment: self
                .experiment_assignment
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect::<Vec<_>>()
                .into(),
            source: Arc::from(self.source.as_str()),
            target: Arc::from(self.target.as_str()),
        }
    }
}

/// One source of a stored query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzSourceWire {
    /// Source population.
    pub population: String,
    /// Declared controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment levels.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Selection targets on this source's diagram.
    pub selection_targets: Vec<u32>,
}

/// Portable canonical mz query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MzTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Target population.
    pub target: String,
    /// Sources in canonical order.
    pub sources: Vec<MzSourceWire>,
}

impl MzTransportQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &MzTransportQuery) -> Self {
        let query = query.canonical();
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            outcomes: raw(&query.outcomes),
            treatments: raw(&query.treatments),
            target: query.target.to_string(),
            sources: query
                .sources
                .iter()
                .map(|s| MzSourceWire {
                    population: s.population.to_string(),
                    controllable: raw(&s.controllable),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect(),
                    selection_targets: raw(&s.selection_targets),
                })
                .collect(),
        }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(&self) -> MzTransportQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        MzTransportQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            target: Arc::from(self.target.as_str()),
            sources: self
                .sources
                .iter()
                .map(|s| ZTransportSourceSpec {
                    population: Arc::from(s.population.as_str()),
                    controllable: ids(&s.controllable).into(),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|(v, x)| InterventionAssignment {
                            variable: VariableId::from_raw(*v),
                            value: x.to_value(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                    selection_targets: ids(&s.selection_targets).into(),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

/// Portable canonical mixed-source query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MixedSourceQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Target population.
    pub target: String,
    /// Declared sources of the theorem-scoped routes, in canonical order.
    pub sources: Vec<MzSourceWire>,
}

impl MixedSourceQueryWire {
    /// Encode a query in canonical form.
    #[must_use]
    pub fn from_query(query: &MixedSourceQuery) -> Self {
        let query = query.canonical();
        let raw = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            outcomes: raw(&query.outcomes),
            treatments: raw(&query.treatments),
            target: query.target.to_string(),
            sources: query
                .sources
                .iter()
                .map(|s| MzSourceWire {
                    population: s.population.to_string(),
                    controllable: raw(&s.controllable),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect(),
                    selection_targets: raw(&s.selection_targets),
                })
                .collect(),
        }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(&self) -> MixedSourceQuery {
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        MixedSourceQuery {
            outcomes: ids(&self.outcomes).into(),
            treatments: ids(&self.treatments).into(),
            target: Arc::from(self.target.as_str()),
            sources: self
                .sources
                .iter()
                .map(|s| ZTransportSourceSpec {
                    population: Arc::from(s.population.as_str()),
                    controllable: ids(&s.controllable).into(),
                    experiment_assignment: s
                        .experiment_assignment
                        .iter()
                        .map(|(v, x)| InterventionAssignment {
                            variable: VariableId::from_raw(*v),
                            value: x.to_value(),
                        })
                        .collect::<Vec<_>>()
                        .into(),
                    selection_targets: ids(&s.selection_targets).into(),
                })
                .collect::<Vec<_>>()
                .into(),
        }
    }
}
