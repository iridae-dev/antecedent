//! Identification flags of a graph posterior, derived rather than declared.
//!
//! The identification ranking weighs candidate studies by how much posterior mass
//! on causal structures they move from "not identified" to "identified". The
//! per-structure flags are a property of each structure, not an opinion: they come
//! from running the complete ID algorithm on the structure. A study that sets a
//! variable is judged on the *mutilated* graph, the structure with every edge into
//! the intervened variables removed (the experimental graph), by the same
//! identifier. Nothing here estimates anything or invents a weight; the weights are
//! the posterior's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, IdentificationStatus, VariableId, reason_code};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{IdIdentifier, IdentificationWorkspace};
pub use antecedent_prob::GraphIdentFlag;
use antecedent_prob::WeightedGraphSamples;

/// One weighted structure of a graph posterior.
#[derive(Clone, Debug)]
pub struct GraphPosteriorSample {
    /// Posterior weight (not necessarily normalized).
    pub weight: f64,
    /// Stable key of the structure.
    pub key: u64,
    /// The structure.
    pub graph: Admg,
}

/// A refused derivation: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct StructureIdentificationError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `structure_identification.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl StructureIdentificationError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: "structure_identification.invalid_request",
            message: message.into(),
        }
    }

    fn budget(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_budget_cancel"),
            detail: "structure_identification.budget",
            message: message.into(),
        }
    }
}

/// The experimental graph of an intervention on `targets`: every directed edge
/// into a target and every bidirected edge at a target is removed; the nodes are
/// unchanged.
///
/// # Errors
/// A target that is not a static variable of `graph`, or a graph that cannot be
/// rebuilt.
pub fn mutilate(
    graph: &Admg,
    targets: &[VariableId],
) -> Result<Admg, StructureIdentificationError> {
    let mut blocked = Vec::with_capacity(targets.len());
    for target in targets {
        blocked.push(graph.dense_of_static(*target).ok_or_else(|| {
            StructureIdentificationError::invalid(format!(
                "intervention target {} is not a variable of the structure",
                target.raw()
            ))
        })?);
    }
    let rebuilt = |error: antecedent_graph::GraphError| {
        StructureIdentificationError::invalid(error.to_string())
    };
    let mut out = Admg::empty();
    for node in graph.nodes() {
        out.add_node(*node).map_err(rebuilt)?;
    }
    for (from, to) in graph.directed_edge_pairs_sorted() {
        if !blocked.contains(&DenseNodeId::from_raw(to)) {
            out.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .map_err(rebuilt)?;
        }
    }
    for (a, b) in graph.bidirected_edge_pairs_sorted() {
        if !blocked.contains(&DenseNodeId::from_raw(a))
            && !blocked.contains(&DenseNodeId::from_raw(b))
        {
            out.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(rebuilt)?;
        }
    }
    Ok(out)
}

/// Whether the complete ID algorithm identifies `P(outcome | do(treatment))` in
/// `graph`.
///
/// # Errors
/// Unknown variables, or a budget stop of the identifier (never read as
/// non-identification).
pub fn identifies(
    graph: &Admg,
    treatment: VariableId,
    outcome: VariableId,
) -> Result<GraphIdentFlag, StructureIdentificationError> {
    let identifier = IdIdentifier::new();
    let prepared = identifier
        .prepare(graph)
        .map_err(|e| StructureIdentificationError::invalid(e.to_string()))?;
    let result = identifier
        .identify_ate(
            &prepared,
            &AverageEffectQuery::binary_ate(treatment, outcome),
            &mut IdentificationWorkspace::default(),
        )
        .map_err(|e| {
            if e.is_budget_or_cancel() {
                StructureIdentificationError::budget(format!("the identifier stopped: {e}"))
            } else {
                StructureIdentificationError::invalid(e.to_string())
            }
        })?;
    Ok(if result.status == IdentificationStatus::NonparametricallyIdentified {
        GraphIdentFlag::Identified
    } else {
        GraphIdentFlag::Unidentified
    })
}

/// The posterior with each structure flagged by running the ID algorithm on it.
///
/// # Errors
/// An empty or invalid posterior (a non-finite, negative or all-zero weight), or
/// any refusal of [`identifies`].
pub fn structure_posterior(
    samples: &[GraphPosteriorSample],
    treatment: VariableId,
    outcome: VariableId,
) -> Result<WeightedGraphSamples, StructureIdentificationError> {
    let mut flags = Vec::with_capacity(samples.len());
    for sample in samples {
        flags.push(identifies(&sample.graph, treatment, outcome)?);
    }
    WeightedGraphSamples::new(
        samples.iter().map(|s| s.weight).collect::<Vec<_>>(),
        flags,
        samples.iter().map(|s| s.key).collect::<Vec<_>>(),
    )
    .map_err(|e| StructureIdentificationError::invalid(e.to_string()))
}

/// Per-structure identification after an experiment that sets `targets`: the ID
/// algorithm on each structure's [`mutilate`]d graph. This is the flag vector the
/// ranker's `identified_under_intervention` expects.
///
/// # Errors
/// As [`structure_posterior`], and an unknown target.
pub fn identified_under_intervention(
    samples: &[GraphPosteriorSample],
    treatment: VariableId,
    outcome: VariableId,
    targets: &[VariableId],
) -> Result<Vec<GraphIdentFlag>, StructureIdentificationError> {
    samples
        .iter()
        .map(|sample| identifies(&mutilate(&sample.graph, targets)?, treatment, outcome))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
        let mut graph = Admg::with_variables(nodes);
        for (a, b) in directed {
            graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        graph
    }

    /// Two structures over T=0 -> Y=1: with latent confounding (not identified)
    /// and without (identified).
    fn posterior() -> Vec<GraphPosteriorSample> {
        vec![
            GraphPosteriorSample { weight: 0.3, key: 10, graph: admg(2, &[(0, 1)], &[(0, 1)]) },
            GraphPosteriorSample { weight: 0.7, key: 11, graph: admg(2, &[(0, 1)], &[]) },
        ]
    }

    #[test]
    fn the_flags_come_from_the_identifier_not_from_a_list() {
        let samples = structure_posterior(&posterior(), v(0), v(1)).unwrap();
        assert_eq!(
            samples.identified.as_ref(),
            &[GraphIdentFlag::Unidentified, GraphIdentFlag::Identified]
        );
        assert!((samples.unidentified_mass() - 0.3).abs() < 1e-12);
        assert_eq!(samples.graph_keys.as_ref(), &[10, 11]);
    }

    #[test]
    fn an_experiment_on_the_treatment_identifies_the_confounded_structure() {
        let flags = identified_under_intervention(&posterior(), v(0), v(1), &[v(0)]).unwrap();
        assert_eq!(flags, vec![GraphIdentFlag::Identified, GraphIdentFlag::Identified]);
        // The un-confounded structure stays identified under any experiment.
        let on_outcome = identified_under_intervention(&posterior(), v(0), v(1), &[v(1)]).unwrap();
        assert_eq!(on_outcome[1], GraphIdentFlag::Identified);
    }

    #[test]
    fn mutilation_removes_every_edge_into_the_target_and_only_those() {
        let graph = admg(3, &[(0, 1), (1, 2)], &[(0, 2)]);
        let cut = mutilate(&graph, &[v(1)]).unwrap();
        assert_eq!(cut.directed_edge_pairs_sorted(), vec![(1, 2)]);
        assert_eq!(cut.bidirected_edge_pairs_sorted(), vec![(0, 2)]);
        let cut = mutilate(&graph, &[v(2)]).unwrap();
        assert_eq!(cut.directed_edge_pairs_sorted(), vec![(0, 1)]);
        assert!(cut.bidirected_edge_pairs_sorted().is_empty());
        assert_eq!(
            mutilate(&graph, &[v(9)]).unwrap_err().detail,
            "structure_identification.invalid_request"
        );
    }

    #[test]
    fn an_empty_posterior_is_refused() {
        let refused = structure_posterior(&[], v(0), v(1)).unwrap_err();
        assert_eq!(refused.detail, "structure_identification.invalid_request");
    }
}
