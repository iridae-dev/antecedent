//! Assumption and derivation wire types for analysis artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus,
};
use serde::{Deserialize, Serialize};

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
    /// Other / extended tag.
    Other(String),
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
        other => AssumptionTagWire::Other(format!("{other:?}")),
    }
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
