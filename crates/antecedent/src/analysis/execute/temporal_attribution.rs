//! Lag-aligned unfolding of a [`TemporalDag`] into the static SCM compiler.
//!
//! Attribution scoring stays on a static [`Dag`]. A lagged graph is unfolded over
//! one contemporaneous slice plus its history, then relabeled so contemporaneous
//! nodes keep the series schema ids the query already names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Admg, Arc, CausalError, Dag, DenseNodeId, TableView, TabularData, TemporalDag, TemporalIndexer,
    TimeSeriesData, VariableId,
};

/// Static SCM and the series column each of its nodes reads.
#[derive(Clone, Debug)]
pub(super) struct LagAlignedScm {
    /// Static DAG passed to [`fit_gcm`]. Contemporaneous ids match the series schema.
    pub(super) dag: Dag,
    /// `(series variable, lag)` in static-node order. Lag `0` is contemporaneous.
    columns: Arc<[(VariableId, u32)]>,
    /// Longest template lag of the unfolding.
    history: u32,
}

impl LagAlignedScm {
    /// Unfold `graph` against `series` into a static DAG and a column recipe.
    ///
    /// History is the longest template lag. Horizon is one contemporaneous slice.
    /// Parents that fall before that window are dropped by the unfolding itself.
    ///
    /// # Errors
    ///
    /// Empty inputs, a non-lagged node, a variable outside the series, a schema
    /// that is not dense `0..n-1`, or a series too short to realize the longest lag.
    pub(super) fn unfold(
        graph: &TemporalDag,
        series: &TimeSeriesData,
    ) -> Result<Self, CausalError> {
        let variables = series.schema().variables();
        if variables.is_empty() || graph.nodes().is_empty() {
            return Err(CausalError::Compile {
                message: "temporal attribution requires a lagged graph and a series".into(),
            });
        }
        let variable_count = u32::try_from(variables.len()).map_err(|_| CausalError::Compile {
            message: "series has too many variables for a lag-aligned SCM".into(),
        })?;
        for (index, variable) in variables.iter().enumerate() {
            if variable.id.raw() != u32::try_from(index).unwrap_or(u32::MAX) {
                return Err(CausalError::Compile {
                    message: "temporal attribution requires a dense series schema".into(),
                });
            }
        }
        let mut history = 0u32;
        for node in graph.nodes() {
            let antecedent_graph::NodeRef::Lagged { variable, lag } = node else {
                return Err(CausalError::Compile {
                    message: "temporal attribution requires lagged nodes".into(),
                });
            };
            if variable.raw() >= variable_count {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal attribution variable {} is outside the series",
                        variable.raw()
                    ),
                });
            }
            history = history.max(lag.raw());
        }
        let history_rows = usize::try_from(history).unwrap_or(usize::MAX);
        if history > 0 && series.row_count() <= history_rows {
            return Err(CausalError::Compile {
                message: "temporal attribution needs at least one row after the longest lag".into(),
            });
        }
        let indexer = TemporalIndexer::new(variable_count, history, 1)?;
        let unfolded = graph.unfold(indexer.clone())?;
        let mut contemporaneous = Vec::new();
        let mut lagged = Vec::new();
        for dense in 0..indexer.dense_len() {
            let dense_id = u32::try_from(dense).map_err(|_| CausalError::Compile {
                message: "unfolded window exceeds u32 nodes".into(),
            })?;
            let key = indexer.key_of(dense_id)?;
            match key.offset.cmp(&0) {
                std::cmp::Ordering::Equal => contemporaneous.push(dense_id),
                std::cmp::Ordering::Less => lagged.push(dense_id),
                std::cmp::Ordering::Greater => {
                    return Err(CausalError::Compile {
                        message: "lag-aligned attribution unfolds one contemporaneous slice".into(),
                    });
                }
            }
        }
        let mut order = contemporaneous;
        order.extend(lagged);
        let mut to_static = vec![0u32; indexer.dense_len()];
        let mut columns = Vec::with_capacity(order.len());
        let mut dag = Dag::empty();
        for (static_index, unfolded_id) in order.iter().copied().enumerate() {
            let static_id = u32::try_from(static_index).map_err(|_| CausalError::Compile {
                message: "lag-aligned SCM exceeds u32 nodes".into(),
            })?;
            let unfolded_index = usize::try_from(unfolded_id).map_err(|_| {
                CausalError::Compile { message: "unfolded window exceeds usize".into() }
            })?;
            to_static[unfolded_index] = static_id;
            let key = indexer.key_of(unfolded_id)?;
            let lag = u32::try_from(-i64::from(key.offset)).map_err(|_| CausalError::Compile {
                message: "lag-aligned attribution found a forward offset".into(),
            })?;
            let id =
                dag.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(static_id)))?;
            if id.raw() != static_id {
                return Err(CausalError::Compile {
                    message: "lag-aligned node id drifted from schema order".into(),
                });
            }
            columns.push((key.variable, lag));
        }
        for edge in unfolded.dag.edges() {
            let Some((parent, child)) = edge.parent_child() else {
                continue;
            };
            let from = DenseNodeId::from_raw(to_static[parent.as_usize()]);
            let to = DenseNodeId::from_raw(to_static[child.as_usize()]);
            match dag.insert_directed(from, to) {
                Ok(()) | Err(antecedent_graph::GraphError::DuplicateEdge { .. }) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Self { dag, columns: Arc::from(columns), history })
    }

    /// Unfold a mixed template into the lag-aligned numbering.
    ///
    /// `window` is the template unrolled over `history + 1` periods
    /// ([`antecedent_graph::TemporalAdmg::unroll_window_over`]), whose coordinate
    /// `period * variable_count + variable` is the dense id of the unfolding of `history` earlier
    /// periods and one contemporaneous period. Returns the lag-aligned SCM over the directed part
    /// of the window and the window itself in the same static numbering, contemporaneous nodes
    /// first and carrying the series schema ids, so the table [`Self::table`] builds is the data
    /// of the mixed graph.
    ///
    /// # Errors
    ///
    /// An empty or non-dense schema, a series too short to realize the longest lag, or a window
    /// that is not the unfolding of `history` periods over the schema.
    pub(super) fn unfold_mixed(
        series: &TimeSeriesData,
        history: u32,
        window: &Admg,
    ) -> Result<(Self, Admg), CausalError> {
        let variables = series.schema().variables();
        if variables.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal attribution requires a lagged graph and a series".into(),
            });
        }
        let variable_count = u32::try_from(variables.len()).map_err(|_| CausalError::Compile {
            message: "series has too many variables for a lag-aligned SCM".into(),
        })?;
        for (index, variable) in variables.iter().enumerate() {
            if variable.id.raw() != u32::try_from(index).unwrap_or(u32::MAX) {
                return Err(CausalError::Compile {
                    message: "temporal attribution requires a dense series schema".into(),
                });
            }
        }
        let history_rows = usize::try_from(history).unwrap_or(usize::MAX);
        if history > 0 && series.row_count() <= history_rows {
            return Err(CausalError::Compile {
                message: "temporal attribution needs at least one row after the longest lag".into(),
            });
        }
        let indexer = TemporalIndexer::new(variable_count, history, 1)?;
        if window.node_count() != indexer.dense_len() {
            return Err(CausalError::Compile {
                message: "the unrolled window does not match the unfolding of the series".into(),
            });
        }
        let mut contemporaneous = Vec::new();
        let mut lagged = Vec::new();
        for dense in 0..indexer.dense_len() {
            let dense_id = u32::try_from(dense).map_err(|_| CausalError::Compile {
                message: "unfolded window exceeds u32 nodes".into(),
            })?;
            let key = indexer.key_of(dense_id)?;
            match key.offset.cmp(&0) {
                std::cmp::Ordering::Equal => contemporaneous.push(dense_id),
                std::cmp::Ordering::Less => lagged.push(dense_id),
                std::cmp::Ordering::Greater => {
                    return Err(CausalError::Compile {
                        message: "lag-aligned attribution unfolds one contemporaneous slice".into(),
                    });
                }
            }
        }
        let mut order = contemporaneous;
        order.extend(lagged);
        let mut to_static = vec![0u32; indexer.dense_len()];
        let mut columns = Vec::with_capacity(order.len());
        let mut dag = Dag::empty();
        let mut mixed = Admg::empty();
        for (static_index, unfolded_id) in order.iter().copied().enumerate() {
            let static_id = u32::try_from(static_index).map_err(|_| CausalError::Compile {
                message: "lag-aligned SCM exceeds u32 nodes".into(),
            })?;
            let unfolded_index = usize::try_from(unfolded_id).map_err(|_| {
                CausalError::Compile { message: "unfolded window exceeds usize".into() }
            })?;
            to_static[unfolded_index] = static_id;
            let key = indexer.key_of(unfolded_id)?;
            let lag = u32::try_from(-i64::from(key.offset)).map_err(|_| CausalError::Compile {
                message: "lag-aligned attribution found a forward offset".into(),
            })?;
            let node = antecedent_graph::NodeRef::Static(VariableId::from_raw(static_id));
            dag.add_node(node)?;
            mixed.add_node(node)?;
            columns.push((key.variable, lag));
        }
        let relabel = |node: DenseNodeId| DenseNodeId::from_raw(to_static[node.as_usize()]);
        for from in window.node_ids() {
            for &to in window.children(from) {
                for inserted in [
                    dag.insert_directed(relabel(from), relabel(to)),
                    mixed.insert_directed(relabel(from), relabel(to)),
                ] {
                    match inserted {
                        Ok(()) | Err(antecedent_graph::GraphError::DuplicateEdge { .. }) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            for &other in window.bidirected_neighbors(from) {
                if from.raw() < other.raw() {
                    match mixed.insert_bidirected(relabel(from), relabel(other)) {
                        Ok(()) | Err(antecedent_graph::GraphError::DuplicateEdge { .. }) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
        Ok((Self { dag, columns: Arc::from(columns), history }, mixed))
    }

    /// Bayesian block-bootstrap row weights of `n` consecutive lag-aligned rows: one
    /// `Exp(1)` weight per contiguous block of [`antecedent_data::circular_block_length`]
    /// rows (at least the unfolding's lag span plus one), shared by every row of the
    /// block. Rows of one dependent series are not exchangeable, so the Dirichlet
    /// weights fall on blocks of consecutive times, not on single rows.
    pub(super) fn block_row_weights(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
    ) -> Vec<f64> {
        block_dirichlet_row_weights(n, self.block_length(n), rng)
    }

    /// Circular-block multinomial row counts of `n` consecutive lag-aligned rows for one
    /// frequentist replicate: blocks of [`antecedent_data::circular_block_length`] rows
    /// (at least the unfolding's lag span plus one) start at uniform positions and wrap
    /// at the end of the series, until `n` rows are drawn.
    pub(super) fn block_row_counts(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
    ) -> Vec<f64> {
        block_bootstrap_counts(n, self.block_length(n), rng)
    }

    /// Number of contemporaneous nodes (one per series variable): the nodes `0..count` are the
    /// slice a rollout steps, the rest are lagged copies tied to the rollout's history.
    pub(super) fn contemporaneous_count(&self) -> usize {
        self.columns.iter().filter(|(_, lag)| *lag == 0).count()
    }

    /// The assignments that tie every lagged node `(v, L)` of the unfolding to the rollout's
    /// history: `v@L := v` read `L` steps back, active at every step of an `n_steps` rollout.
    /// Contemporaneous nodes keep their fitted mechanisms, so stepping the unfolded model
    /// rolls the dynamics forward instead of sampling the lagged copies as exogenous columns.
    pub(super) fn lag_assignments(
        &self,
        n_steps: usize,
    ) -> Vec<antecedent_core::SequencedIntervention> {
        let node_of_variable = |variable: VariableId| {
            self.columns
                .iter()
                .position(|&(v, lag)| v == variable && lag == 0)
                .and_then(|index| u32::try_from(index).ok())
                .map(VariableId::from_raw)
        };
        let until = i32::try_from(n_steps.saturating_sub(1)).unwrap_or(i32::MAX);
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, (_, lag))| *lag > 0)
            .filter_map(|(index, &(variable, lag))| {
                let node = VariableId::from_raw(u32::try_from(index).ok()?);
                let source = node_of_variable(variable)?;
                Some(antecedent_core::SequencedIntervention::new(
                    antecedent_core::Intervention::assign(
                        node,
                        antecedent_core::ValueExpr::lag(source, lag),
                    ),
                    antecedent_core::TemporalPolicy::Sustained { from: 0, until },
                ))
            })
            .collect()
    }

    /// Longest template lag of the unfolding: the history a rollout reads.
    pub(super) const fn history(&self) -> u32 {
        self.history
    }

    /// The last `history` rows of `table` as the steps before time zero of a rollout of `n_rows`
    /// parallel paths, most recent first, one batch per step over every node.
    ///
    /// # Errors
    ///
    /// A table with fewer rows than the history, or a missing column.
    pub(super) fn initial_history(
        &self,
        table: &TabularData,
        n_rows: usize,
    ) -> Result<Vec<antecedent_model::ValueBatch>, CausalError> {
        let n = table.row_count();
        let history = usize::try_from(self.history).unwrap_or(usize::MAX);
        if history > n {
            return Err(CausalError::Compile {
                message: "the rollout history is longer than the series".into(),
            });
        }
        let nodes = self.columns.len();
        let mut batches = Vec::with_capacity(history);
        for step in 1..=history {
            let row = n - step;
            let mut values = Vec::with_capacity(nodes * n_rows);
            for node in 0..nodes {
                let id = VariableId::from_raw(u32::try_from(node).map_err(|_| {
                    CausalError::Compile { message: "lag-aligned SCM exceeds u32 nodes".into() }
                })?);
                let value = table.float64_values(id)?[row];
                values.extend(std::iter::repeat_n(value, n_rows));
            }
            batches.push(antecedent_model::ValueBatch {
                n_rows,
                n_nodes: nodes,
                values: Arc::from(values),
            });
        }
        Ok(batches)
    }

    /// Block length of the unfolding's block resamples over `n` rows.
    pub(super) fn block_length(&self, n: usize) -> usize {
        let span = usize::try_from(self.history).unwrap_or(usize::MAX).saturating_add(1);
        antecedent_data::circular_block_length(span, n)
    }

    /// Build the lag-aligned table for `series`. Incomplete lag prefixes copy
    /// the first observation so the row count stays the series length.
    ///
    /// # Errors
    ///
    /// Schema mismatch or a missing float column.
    pub(super) fn table(&self, series: &TimeSeriesData) -> Result<TabularData, CausalError> {
        let variables = series.schema().variables();
        if variables.len() != self.columns.iter().filter(|(_, lag)| *lag == 0).count() {
            return Err(CausalError::Compile {
                message: "lag-aligned series schema does not match the unfolded graph".into(),
            });
        }
        let n = series.row_count();
        let mut owned = Vec::with_capacity(self.columns.len());
        for &(variable, lag) in self.columns.iter() {
            let source = series.float64_slice(variable)?;
            if source.len() != n {
                return Err(CausalError::Compile {
                    message: "lag-aligned series column length does not match the time index"
                        .into(),
                });
            }
            let lag_rows = usize::try_from(lag).map_err(|_| CausalError::Compile {
                message: "lag does not fit a row index".into(),
            })?;
            let mut values = Vec::with_capacity(n);
            for t in 0..n {
                values.push(source[t.saturating_sub(lag_rows)]);
            }
            owned.push((format!("v{}_{lag}", variable.raw()), values));
        }
        let refs: Vec<(&str, &[f64])> =
            owned.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
        Ok(TabularData::from_f64_columns(refs)?)
    }
}

/// Multinomial counts of `n` rows in time order drawn as circular blocks of `block`
/// consecutive rows: each block starts at a uniform row and wraps past the end.
pub(super) fn block_bootstrap_counts(
    n: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n];
    if n == 0 {
        return counts;
    }
    let block = block.clamp(1, n);
    let mut drawn = 0;
    while drawn < n {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform in [0, 1) times n floors to a row index below n"
        )]
        let start = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        for offset in 0..block.min(n - drawn) {
            counts[(start + offset) % n] += 1.0;
        }
        drawn += block.min(n - drawn);
    }
    counts
}

/// One `Exp(1)` weight per contiguous block of `block` rows, repeated on every row of
/// the block (the last block may be shorter), for `n` rows in time order.
pub(super) fn block_dirichlet_row_weights(
    n: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<f64> {
    let block = block.clamp(1, n.max(1));
    let blocks = n.div_ceil(block);
    let draws = antecedent_kernels::bayesian_bootstrap_weights(rng, blocks, false);
    (0..n).map(|row| draws[row / block]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_counts_draw_every_row_once_in_total_and_keep_blocks_contiguous() {
        let mut rng = antecedent_core::RngFactory::from_seed(5)
            .stream_for(antecedent_core::StreamDomain::Test, 2);
        let counts = block_bootstrap_counts(10, 4, &mut rng);
        assert_eq!(counts.len(), 10);
        assert!((counts.iter().sum::<f64>() - 10.0).abs() < 1e-12);
        assert!(counts.iter().all(|count| *count >= 0.0 && count.fract() == 0.0));
        let full = block_bootstrap_counts(7, 7, &mut rng);
        assert!(full.iter().all(|count| (*count - 1.0).abs() < 1e-12));
    }

    #[test]
    fn block_weights_are_shared_within_each_block() {
        let mut rng = antecedent_core::RngFactory::from_seed(3)
            .stream_for(antecedent_core::StreamDomain::Test, 1);
        let weights = block_dirichlet_row_weights(10, 4, &mut rng);
        assert_eq!(weights.len(), 10);
        assert!(weights[..4].iter().all(|w| (*w - weights[0]).abs() < 1e-15));
        assert!(weights[4..8].iter().all(|w| (*w - weights[4]).abs() < 1e-15));
        assert!(weights[8..].iter().all(|w| (*w - weights[8]).abs() < 1e-15));
        assert!(weights.iter().all(|w| *w > 0.0));
    }
    use antecedent_core::Lag;

    #[test]
    fn lag_one_chain_keeps_contemporaneous_schema_ids() {
        let mut graph = TemporalDag::empty();
        let parent =
            graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).expect("lagged parent");
        let child = graph
            .add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS)
            .expect("contemporaneous child");
        graph.insert_directed(parent, child).expect("template edge");
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 1.0, 1.0, 9.0];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .expect("series");
        let aligned = LagAlignedScm::unfold(&graph, &series).expect("unfold");
        assert_eq!(aligned.dag.directed_edge_pairs_sorted(), vec![(2, 1)]);
        let table = aligned.table(&series).expect("table");
        assert_eq!(table.float64_slice(VariableId::from_raw(1)).unwrap(), y.as_slice());
        assert_eq!(table.float64_slice(VariableId::from_raw(2)).unwrap(), &[0.0, 0.0, 1.0, 2.0]);
    }
}
