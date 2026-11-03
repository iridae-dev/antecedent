//! Bounded exact DAG scoring for the closed graph-imputation candidate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;
use antecedent_core::graph_imputation_contract::{GraphSupport, MAX_VARIABLES};
use antecedent_data::TabularData;
use antecedent_prob::kish_ess;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    GraphPosterior, GraphPrior, SamplingSummary, accumulate_marginals, analytic_graph_diagnostics,
    normalize_log_weights,
};
use crate::graph_score::{DagScorer, tabular_score_data};

/// A stopped bounded exact search retains the shared-budget receipt; a numerical
/// scoring failure remains a separate discovery error.
#[derive(Debug, thiserror::Error)]
pub enum BoundedExactDagError {
    /// Search stopped before all graph masks were evaluated.
    #[error("bounded DAG search stopped: {}", .0.summary())]
    Search(antecedent_core::search::SearchReceipt),
    /// DAGs without finite scores have unknown weight and cannot be discarded.
    #[error("bounded graph score: {count} DAGs have no finite score")]
    Unscorable {
        /// Number of DAGs with no finite score.
        count: u64,
    },
    /// The underlying native scoring operation refused.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
}

/// Predeclared structural support: the treatment-to-outcome edge is required, and a root
/// treatment may additionally be demanded. This states an assumption; it is not learned from
/// the scored observations.
#[must_use]
pub fn graph_imputation_supported_mask(
    mask: &AdjacencyMask,
    n: usize,
    treatment: usize,
    outcome: usize,
    support: GraphSupport,
) -> bool {
    (2..=MAX_VARIABLES).contains(&n)
        && treatment < n
        && outcome < n
        && treatment != outcome
        && mask.has_edge(n, treatment, outcome)
        && match support {
            GraphSupport::TreatmentRootDirectEffect => {
                (0..n).all(|parent| parent == treatment || !mask.has_edge(n, parent, treatment))
            }
            GraphSupport::TreatmentDirectEffect => true,
        }
}

/// Every DAG mask inside the declared support, in increasing mask order. Compiler and consumer
/// both enumerate through this, so they agree on the atoms without exchanging a list.
#[must_use]
pub fn graph_imputation_supported_masks(
    n: usize,
    treatment: usize,
    outcome: usize,
    support: GraphSupport,
) -> Vec<u64> {
    use crate::adjacency_mask::dag_from_adjacency_mask;
    if !(2..=MAX_VARIABLES).contains(&n) {
        return Vec::new();
    }
    (0..1u64 << (n * (n - 1)))
        .filter(|&word| {
            let mask = AdjacencyMask::from_word(word);
            graph_imputation_supported_mask(&mask, n, treatment, outcome, support)
                && dag_from_adjacency_mask(&mask, n).is_ok()
        })
        .collect()
}

/// Enumerate and score the predeclared direct-edge support using a caller-owned shared
/// search budget. No MCMC fallback is permitted.
///
/// `retained` lists, in increasing order, the masks of the declared support that the support
/// rule kept. Only those graphs are scored, so the weights are over the retained graphs alone.
///
/// # Errors
/// Returns a complete stop receipt or a native scoring refusal.
pub fn graph_imputation_exact_dags(
    data: &TabularData,
    variables: &[VariableId],
    treatment: VariableId,
    outcome: VariableId,
    support: GraphSupport,
    retained: &[u64],
    budget: &mut antecedent_core::search::SearchBudget<'_>,
) -> Result<GraphPosterior, BoundedExactDagError> {
    use crate::adjacency_mask::dag_from_adjacency_mask;
    let n = variables.len();
    if !(2..=MAX_VARIABLES).contains(&n)
        || treatment == outcome
        || treatment.raw() as usize >= n
        || outcome.raw() as usize >= n
        || variables.iter().enumerate().any(|(index, id)| id.raw() as usize != index)
    {
        return Err(
            DiscoveryError::unsupported("graph imputation requires 2 to 8 variables").into()
        );
    }
    let score_data = tabular_score_data(data, variables)?;
    // The uniform prior is conditioned on the declared support by enumerating
    // only support-admissible masks before normalization; this is not filtering
    // a data-derived posterior.
    let prior = GraphPrior::uniform();
    let scorer = DagScorer::new(n, &score_data, GraphScoreFamily::GaussianBic, &prior, variables);
    let mut masks = Vec::new();
    let mut log_weights = Vec::new();
    let mut rejected = 0u64;
    let end = 1u64 << (n * (n - 1));
    for mask in 0..end {
        if let Err(stop) = budget.charge(1, 25 * 1024 * 1024) {
            return Err(BoundedExactDagError::Search(budget.receipt(
                stop,
                vec![format!("masks:0..{mask}")],
                vec![format!("masks:{mask}..{end}")],
            )));
        }
        if retained.binary_search(&mask).is_err() {
            continue;
        }
        let mask = AdjacencyMask::from_word(mask);
        if !graph_imputation_supported_mask(
            &mask,
            n,
            treatment.raw() as usize,
            outcome.raw() as usize,
            support,
        ) || dag_from_adjacency_mask(&mask, n).is_err()
        {
            continue;
        }
        if let Some(value) = scorer.score(&mask).filter(|value| value.is_finite()) {
            masks.push(mask);
            log_weights.push(value);
        } else {
            rejected += 1;
        }
    }
    if let Err(stop) = budget.admits(1, 25 * 1024 * 1024, 0) {
        return Err(BoundedExactDagError::Search(budget.receipt(
            stop,
            vec![format!("masks:0..{end}")],
            vec!["finalization".into()],
        )));
    }
    if rejected > 0 {
        return Err(BoundedExactDagError::Unscorable { count: rejected });
    }
    if masks.is_empty() {
        return Err(DiscoveryError::unsupported("no finite BIC scores").into());
    }
    let weights = normalize_log_weights(&log_weights)?;
    let ess = kish_ess(&weights);
    let (edge, orient) = accumulate_marginals(n, &weights, &masks);
    let mut diagnostics = analytic_graph_diagnostics(masks.len(), ess);
    if rejected > 0 {
        diagnostics.notes.push(std::sync::Arc::from(format!(
            "{rejected} DAG(s) excluded because their BIC scores were not finite"
        )));
    }
    Ok(GraphPosterior::from_atoms(
        n,
        weights,
        masks,
        edge,
        orient,
        SamplingSummary { ess, diagnostics, rejected_invalid: rejected },
    )?
    .with_algorithm("exact_dag_posterior"))
}

#[cfg(test)]
mod graph_imputation_budget_tests {
    use super::*;
    use antecedent_core::ExecutionContext;
    use antecedent_core::search::{SearchBudget, SearchLimits, SearchStop};

    #[test]
    fn root_support_requires_the_edge_and_forbids_every_treatment_parent() {
        let root = GraphSupport::TreatmentRootDirectEffect;
        let mut mask = AdjacencyMask::from_word(0);
        assert!(!graph_imputation_supported_mask(&mask, 3, 0, 1, root));
        mask.set_edge(3, 0, 1, true);
        assert!(graph_imputation_supported_mask(&mask, 3, 0, 1, root));
        mask.set_edge(3, 2, 0, true);
        assert!(!graph_imputation_supported_mask(&mask, 3, 0, 1, root));
        mask.set_edge(3, 2, 0, false);
        mask.set_edge(3, 1, 2, true);
        assert!(graph_imputation_supported_mask(&mask, 3, 0, 1, root));
    }

    #[test]
    fn direct_support_allows_treatment_parents_but_still_requires_the_edge() {
        let direct = GraphSupport::TreatmentDirectEffect;
        let mut mask = AdjacencyMask::from_word(0);
        mask.set_edge(3, 2, 0, true);
        assert!(!graph_imputation_supported_mask(&mask, 3, 0, 1, direct));
        mask.set_edge(3, 0, 1, true);
        assert!(graph_imputation_supported_mask(&mask, 3, 0, 1, direct));
    }

    #[test]
    fn supported_graph_counts_are_the_labeled_dags_containing_the_edge() {
        let counts = |n, support| graph_imputation_supported_masks(n, 0, 1, support).len();
        // Root treatment: the DAGs on the remaining nodes, each optionally fed by treatment.
        assert_eq!(counts(2, GraphSupport::TreatmentRootDirectEffect), 1);
        assert_eq!(counts(3, GraphSupport::TreatmentRootDirectEffect), 6);
        assert_eq!(counts(4, GraphSupport::TreatmentRootDirectEffect), 100);
        assert_eq!(counts(2, GraphSupport::TreatmentDirectEffect), 1);
        assert_eq!(counts(3, GraphSupport::TreatmentDirectEffect), 8);
        assert!(counts(4, GraphSupport::TreatmentDirectEffect) <= 543);
        assert!(counts(5, GraphSupport::TreatmentDirectEffect) <= 29_281);
        assert!(counts(5, GraphSupport::TreatmentDirectEffect) > 543);
        assert!(
            graph_imputation_supported_masks(9, 0, 1, GraphSupport::TreatmentDirectEffect)
                .is_empty()
        );
    }

    #[test]
    fn only_retained_graphs_are_scored_and_weighted() {
        let t = [0., 1., 0., 1., 0., 1.];
        let y = [0.1, 1.1, -0.2, 0.8, 0.3, 1.4];
        let x = [0.5, -0.3, 0.9, 0.1, -0.7, 0.2];
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let ctx = ExecutionContext::for_tests(8);
        let variables = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let direct = GraphSupport::TreatmentDirectEffect;
        let all = graph_imputation_supported_masks(3, 0, 1, direct);
        let retained = &all[..all.len() - 1];
        let mut budget =
            SearchBudget::new(SearchLimits { operations: 1000, depth: 1 }, &ctx).unwrap();
        let posterior = graph_imputation_exact_dags(
            &data,
            &variables,
            variables[0],
            variables[1],
            direct,
            retained,
            &mut budget,
        )
        .unwrap();
        let scored: Vec<u64> =
            posterior.atoms().iter().map(|mask| mask.to_word().unwrap()).collect();
        assert_eq!(scored, retained);
        assert!((posterior.weights().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unscorable_graphs_are_not_silently_discarded() {
        let t = [0., 0., 0., 0., 0., 0.];
        let y = [0.1, 1.1, -0.2, 0.8, 0.3, 1.4];
        let data =
            TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
        let ctx = ExecutionContext::for_tests(8);
        let mut budget = SearchBudget::new(SearchLimits { operations: 4, depth: 1 }, &ctx).unwrap();
        let retained =
            graph_imputation_supported_masks(2, 0, 1, GraphSupport::TreatmentRootDirectEffect);
        assert!(matches!(graph_imputation_exact_dags(&data,
            &[VariableId::from_raw(0), VariableId::from_raw(1)], VariableId::from_raw(0), VariableId::from_raw(1), GraphSupport::TreatmentRootDirectEffect, &retained, &mut budget),
            Err(BoundedExactDagError::Unscorable { count }) if count > 0));
    }

    #[test]
    fn bounded_search_returns_precise_operation_stop_receipt() {
        let t = [0., 1., 0., 1., 0., 1.];
        let y = [0.1, 1.1, -0.2, 0.8, 0.3, 1.4];
        let data =
            TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
        let ctx = ExecutionContext::for_tests(8);
        let mut budget = SearchBudget::new(SearchLimits { operations: 1, depth: 1 }, &ctx).unwrap();
        let error = graph_imputation_exact_dags(
            &data,
            &[VariableId::from_raw(0), VariableId::from_raw(1)],
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            GraphSupport::TreatmentRootDirectEffect,
            &graph_imputation_supported_masks(2, 0, 1, GraphSupport::TreatmentRootDirectEffect),
            &mut budget,
        )
        .unwrap_err();
        let BoundedExactDagError::Search(receipt) = error else {
            panic!("expected search receipt")
        };
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.operations_consumed, Some(1));
        assert!(!receipt.unevaluated.is_empty());
    }
}
