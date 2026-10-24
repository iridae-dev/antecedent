//! What a study retains of a failed identification: the not-identified product,
//! the structure it ran on and the variables measured today.
//!
//! The evidence plan reads these to rebuild the theorem routes that decide whether
//! a proposed study would repair the failure. Cheap inspection and identified
//! products retain no failure, so nothing is read from them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{IdentificationStatus, NodeRef, VariableId};
use antecedent_graph::Admg;
use antecedent_identify::IdentificationResult;

use super::{Study, contract_identification, data_schema};

/// The retained inputs of a failed identification.
#[derive(Clone, Debug)]
pub(crate) struct FailureInputs {
    /// The retained not-identified product.
    pub(crate) result: IdentificationResult,
    /// The structure the identification ran on.
    pub(crate) graph: Admg,
    /// The study's variable names, in id order.
    pub(crate) names: Vec<Arc<str>>,
    /// The graph variables the study's data hold.
    pub(crate) observed: Vec<VariableId>,
}

/// The retained failure of `study`, or `None` when it holds no not-identified
/// product on a directed or bidirected graph.
pub(crate) fn failure_inputs(study: &Study) -> Option<FailureInputs> {
    let cached = contract_identification(study)?;
    if cached.status != IdentificationStatus::NotIdentified {
        return None;
    }
    let graph = study
        .graph
        .as_admg()
        .cloned()
        .or_else(|| study.graph.as_dag().and_then(|dag| Admg::from_dag(dag).ok()))?;
    let names: Vec<Arc<str>> =
        data_schema(&study.data).variables().iter().map(|v| Arc::clone(&v.name)).collect();
    let observed = graph
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .filter(|variable| variable.as_usize() < names.len())
        .collect();
    Some(FailureInputs { result: cached.into_owned(), graph, names, observed })
}
