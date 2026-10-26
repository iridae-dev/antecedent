//! Empty continuous tables shared by the static-discovery unit tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};

use crate::adjacency_mask::AdjacencyMask;
use crate::constraints::DiscoveryConstraints;
use crate::graph_posterior::{GraphPrior, GraphPriorTable};
use crate::graph_score::ChainTarget;

/// Named continuous columns, in order. Variable ids are `0..columns.len()`.
///
/// Callers keep the column values they already built, so a shared table does not
/// move an oracle edge.
pub(crate) fn continuous_table(names: &[&str], columns: Vec<Vec<f64>>) -> TabularData {
    let n = columns.first().map_or(0, Vec::len);
    let mut b = CausalSchemaBuilder::new();
    for name in names {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let owned: Vec<OwnedColumn> = columns
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    TabularData::new(storage)
}

/// `ncols` all-zero continuous columns of `nrows` rows, variables `0..ncols`.
pub(crate) fn tabular_n(ncols: usize, nrows: usize) -> TabularData {
    let names: Vec<String> = (0..ncols).map(|i| format!("v{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    continuous_table(&refs, (0..ncols).map(|_| vec![0.0; nrows]).collect())
}

/// The prior over DAGs on `n` labeled nodes that allows at most one parent per node, uniform on
/// its support, as a scored table (no data: the log target of a valid mask is `0`).
///
/// Its support is the rooted forests on `n` labeled nodes, whose law is closed-form
/// ([`rooted_forest_edge_count_law`]), so a sampler of it has an exact reference.
pub(crate) fn rooted_forest_prior_table(n: usize) -> GraphPriorTable {
    let prior = GraphPrior::uniform()
        .with_constraints(DiscoveryConstraints { max_parents: Some(1), ..Default::default() });
    let variables: Vec<VariableId> =
        (0..n).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    GraphPriorTable::new(n, &prior, &variables)
}

/// `P(e edges)` for `e = 0..n` under the uniform law on the rooted forests of `n` labeled nodes.
///
/// There are `(n + 1)^(n - 1)` such forests, and `C(n, k) k n^(n - k - 1)` of them have `k`
/// trees, hence `n - k` edges (Cayley's forest count).
pub(crate) fn rooted_forest_edge_count_law(n: usize) -> Vec<f64> {
    let nf = n as f64;
    let total = (nf + 1.0).powi(i32::try_from(n).unwrap() - 1);
    let mut choose = 1.0_f64;
    let mut law = vec![0.0; n];
    for k in 1..=n {
        choose *= (n - k + 1) as f64 / k as f64;
        law[n - k] = choose * k as f64 * nf.powi(i32::try_from(n - k).unwrap() - 1) / total;
    }
    law
}

/// Assert that per-chain samples of the [`rooted_forest_prior_table`] chain follow its exact law.
///
/// Checks, over the pooled chains: the edge-count histogram is within total-variation `0.1` of
/// [`rooted_forest_edge_count_law`]; the mean edge count and the inclusion frequency of each
/// `probe` edge (`E[edges] / (n (n - 1))` by symmetry) are within five batch-means standard
/// errors of it. Probing edges beyond the first packed word is the point of the choice of `probe`.
pub(crate) fn assert_rooted_forest_law(
    n: usize,
    samples: &[Vec<AdjacencyMask>],
    batch: usize,
    probe: &[(usize, usize)],
) {
    let law = rooted_forest_edge_count_law(n);
    let expected_edges: f64 = law.iter().enumerate().map(|(e, p)| e as f64 * p).sum();
    let counts = |mask: &AdjacencyMask| -> usize {
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| mask.has_edge(n, i, j))
            .count()
    };
    let mut histogram = vec![0.0; n];
    let mut total = 0.0;
    for chain in samples {
        for mask in chain {
            histogram[counts(mask)] += 1.0;
            total += 1.0;
        }
    }
    let tv: f64 = histogram.iter().zip(&law).map(|(h, p)| (h / total - p).abs()).sum::<f64>() / 2.0;
    assert!(tv < 0.1, "edge-count law total variation {tv}: {histogram:?} vs {law:?}");

    // Batch means of an indicator or count, pooled over chains.
    let z = |value: &dyn Fn(&AdjacencyMask) -> f64, target: f64| -> f64 {
        let means: Vec<f64> = samples
            .iter()
            .flat_map(|chain| chain.chunks_exact(batch))
            .map(|b| b.iter().map(value).sum::<f64>() / batch as f64)
            .collect();
        let k = means.len() as f64;
        let mean = means.iter().sum::<f64>() / k;
        let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (k - 1.0);
        (mean - target) / (var / k).sqrt().max(1e-12)
    };
    let z_edges = z(&|m| counts(m) as f64, expected_edges);
    assert!(
        z_edges.abs() < 5.0,
        "mean edge count is {z_edges} standard errors from {expected_edges}"
    );
    let inclusion = expected_edges / (n * (n - 1)) as f64;
    for &(from, to) in probe {
        let z_edge = z(&|m| f64::from(u8::from(m.has_edge(n, from, to))), inclusion);
        assert!(
            z_edge.abs() < 5.0,
            "P({from} -> {to}) is {z_edge} standard errors from {inclusion}"
        );
    }
}

/// A [`ChainTarget`] with the data removed: the log target of a graph is its log prior.
pub(crate) struct PriorOnlyTarget<'a>(pub(crate) &'a GraphPriorTable);

impl ChainTarget for PriorOnlyTarget<'_> {
    fn score(&mut self, mask: &AdjacencyMask) -> Option<f64> {
        self.0.log_prior(mask)
    }

    fn node_score(&mut self, node: usize, parents: &AdjacencyMask) -> Option<f64> {
        self.0.node_log_prior(node, parents)
    }
}
