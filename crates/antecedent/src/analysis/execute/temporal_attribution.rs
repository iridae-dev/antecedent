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
    _temporal_lease: Option<Arc<antecedent_core::execution::AdmissionLease>>,
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
    #[cfg(test)]
    pub(super) fn unfold(
        graph: &TemporalDag,
        series: &TimeSeriesData,
    ) -> Result<Self, CausalError> {
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::unfold_within(graph, series, &mut budget, &ctx)
    }

    pub(super) fn unfold_within(
        graph: &TemporalDag,
        series: &TimeSeriesData,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        let initial_work = (graph.nodes().len() as u64)
            .checked_add(series.schema().variables().len() as u64)
            .ok_or_else(|| CausalError::Resource {
                message: "attribution input size overflow".into(),
            })?;
        budget
            .charge_work(initial_work)
            .map_err(|e| CausalError::Resource { message: e.to_string() })?;
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
        let unfolded = graph.unfold_within(indexer.clone(), budget, ctx)?;
        let nodes = indexer.dense_len() as u64;
        budget.charge_work(nodes).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let mut edges = 0u64;
        for i in 0..indexer.dense_len() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            edges = edges
                .checked_add(unfolded.dag.children(DenseNodeId::from_index(i)).len() as u64)
                .ok_or_else(|| CausalError::Resource {
                    message: "attribution edge overflow".into(),
                })?;
        }
        let work = edges
            .checked_mul(nodes.checked_add(edges).and_then(|n| n.checked_add(1)).ok_or_else(
                || CausalError::Resource { message: "attribution work overflow".into() },
            )?)
            .and_then(|n| n.checked_add(nodes))
            .ok_or_else(|| CausalError::Resource { message: "attribution work overflow".into() })?;
        budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = nodes
            .checked_mul(1024)
            .and_then(|n| edges.checked_mul(512).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| CausalError::Resource {
                message: "attribution bytes overflow".into(),
            })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "attribution byte limit".into() });
        }
        let lease = Arc::new(ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?);
        let mut contemporaneous = Vec::new();
        let mut lagged = Vec::new();
        for dense in 0..indexer.dense_len() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
        Ok(Self { dag, columns: Arc::from(columns), history, _temporal_lease: Some(lease) })
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
        lease: Arc<antecedent_core::execution::AdmissionLease>,
        ctx: &antecedent_core::ExecutionContext,
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
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
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
        Ok((
            Self { dag, columns: Arc::from(columns), history, _temporal_lease: Some(lease) },
            mixed,
        ))
    }

    /// Bayesian block-bootstrap row weights of `n` consecutive lag-aligned rows: one
    /// `Exp(1)` weight per contiguous block of [`antecedent_data::circular_block_length`]
    /// rows (at least the unfolding's lag span plus one), shared by every row of the
    /// block. Rows of one dependent series are not exchangeable, so the Dirichlet
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

    /// Construct observed initial cells and mark unavailable auxiliary lag
    /// copies. The rollout verifies each read before consulting a marked cell.
    pub(super) fn observed_initial_history(
        &self,
        input: &CompleteWindowInput,
        n_rows: usize,
        history_depth: u32,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<ObservedHistory, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let source = input.source();
        let n = source.row_count();
        let history = usize::try_from(history_depth)
            .map_err(|_| CausalError::Resource { message: "history size overflow".into() })?;
        let history = history.min(n);
        if history == 0 {
            return Ok(ObservedHistory {
                batches: Vec::new(),
                available: Vec::new(),
                _lease: None,
            });
        }
        let nodes = self.columns.len();
        let cells =
            history.checked_mul(nodes).and_then(|v| v.checked_mul(n_rows)).ok_or_else(|| {
                CausalError::Resource { message: "rollout history size overflow".into() }
            })?;
        let tail_work = if input.0.provenance.is_some() { n } else { 0 };
        let work = u64::try_from(cells)
            .ok()
            .and_then(|v| u64::try_from(tail_work).ok().and_then(|n| v.checked_add(n)))
            .ok_or_else(|| CausalError::Resource {
                message: "rollout history work overflow".into(),
            })?;
        budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = work.checked_mul(16).and_then(|v| v.checked_add(4096)).ok_or_else(|| {
            CausalError::Resource { message: "rollout history byte quote overflow".into() }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "rollout history byte limit".into() });
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut tail_start = n.saturating_sub(1);
        if let Some(p) = input.0.provenance.as_ref() {
            while tail_start > 0
                && !p.block_start()[tail_start]
                && p.original_ticks()[tail_start - 1].checked_add(1)
                    == Some(p.original_ticks()[tail_start])
            {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                tail_start -= 1;
            }
        } else {
            tail_start = 0;
        }
        let mut batches = Vec::new();
        batches.try_reserve_exact(history).map_err(|_| CausalError::Resource {
            message: "rollout history batch allocation refused".into(),
        })?;
        let mut available = Vec::new();
        available
            .try_reserve_exact(history.checked_mul(nodes).ok_or_else(|| CausalError::Resource {
                message: "rollout history mask size overflow".into(),
            })?)
            .map_err(|_| CausalError::Resource {
                message: "rollout history mask allocation refused".into(),
            })?;
        for step in 1..=history {
            let mut values = Vec::new();
            values
                .try_reserve_exact(nodes.checked_mul(n_rows).ok_or_else(|| {
                    CausalError::Resource { message: "rollout history batch size overflow".into() }
                })?)
                .map_err(|_| CausalError::Resource {
                    message: "rollout history values allocation refused".into(),
                })?;
            for &(variable, lag) in self.columns.iter() {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                let index = usize::try_from(lag)
                    .ok()
                    .and_then(|lag| step.checked_add(lag))
                    .and_then(|back| n.checked_sub(back));
                let value = index
                    .filter(|&row| row >= tail_start)
                    .map(|row| source.float64_slice(variable).map(|column| column[row]))
                    .transpose()?;
                available.push(value.is_some_and(f64::is_finite));
                values.extend(std::iter::repeat_n(
                    value.filter(|v| v.is_finite()).unwrap_or(f64::NAN),
                    n_rows,
                ));
            }
            batches.push(antecedent_model::ValueBatch {
                n_rows,
                n_nodes: nodes,
                values: Arc::from(values),
            });
        }
        Ok(ObservedHistory { batches, available, _lease: Some(lease) })
    }

    /// Block length of the unfolding's block resamples over `n` rows.
    pub(super) fn block_length(&self, n: usize) -> usize {
        let span = usize::try_from(self.history).unwrap_or(usize::MAX).saturating_add(1);
        antecedent_data::circular_block_length(span, n)
    }
}

pub(super) struct SelectedRuns {
    ranges: Vec<std::ops::Range<usize>>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl SelectedRuns {
    pub(super) fn ranges(&self) -> &[std::ops::Range<usize>] {
        &self.ranges
    }
}

pub(super) struct ObservedHistory {
    pub(super) batches: Vec<antecedent_model::ValueBatch>,
    pub(super) available: Vec<bool>,
    _lease: Option<antecedent_core::execution::AdmissionLease>,
}

/// A gathered series may repeat and reorder source ticks. Its identity and
/// block starts travel together with their resource admission; clones share
/// both the data and the lease. This is execution metadata, not evidence that
/// the source was produced by a particular sampler.
#[derive(Clone)]
pub(crate) struct TemporalRowProvenance(Arc<TemporalRowProvenanceInner>);

struct TemporalRowProvenanceInner {
    original_ticks: Vec<usize>,
    block_start: Vec<bool>,
    /// Whether source sampling weights were actually declared by the user;
    /// gathered refit bookkeeping alone must not masquerade as such a declaration.
    user_sampling_weights: bool,
    budget: antecedent_core::TemporalBudget,
    _lease: antecedent_core::execution::AdmissionLease,
}

impl TemporalRowProvenance {
    pub(crate) fn admit_fixed_blocks(
        original_ticks: &[usize],
        block_length: Option<usize>,
        user_sampling_weights: bool,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        if block_length == Some(0) {
            return Err(CausalError::Compile {
                message: "temporal refit block length must be positive".into(),
            });
        }
        Self::admit_with(
            original_ticks.len(),
            |row| Some(original_ticks[row]),
            |row| row == 0 || block_length.is_some_and(|len| row % len == 0),
            user_sampling_weights,
            budget,
            ctx,
        )
    }

    pub(crate) fn admit_contiguous_range(
        start: usize,
        len: usize,
        user_sampling_weights: bool,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::admit_with(
            len,
            |row| start.checked_add(row),
            |row| row == 0,
            user_sampling_weights,
            budget,
            ctx,
        )
    }

    fn admit_with(
        count: usize,
        original_tick: impl Fn(usize) -> Option<usize>,
        block_start: impl Fn(usize) -> bool,
        user_sampling_weights: bool,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let n = u64::try_from(count).map_err(|_| CausalError::Resource {
            message: "temporal refit provenance size overflow".into(),
        })?;
        let bytes = n.checked_mul(24).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
            CausalError::Resource {
                message: "temporal refit provenance byte quote overflow".into(),
            }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource {
                message: "temporal refit provenance byte limit".into(),
            });
        }
        budget.charge_work(n).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut ticks = Vec::new();
        ticks.try_reserve_exact(count).map_err(|_| CausalError::Resource {
            message: "temporal refit tick allocation refused".into(),
        })?;
        let mut starts = Vec::new();
        starts.try_reserve_exact(count).map_err(|_| CausalError::Resource {
            message: "temporal refit block allocation refused".into(),
        })?;
        for row in 0..count {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            ticks.push(original_tick(row).ok_or_else(|| CausalError::Resource {
                message: "temporal refit source coordinate overflow".into(),
            })?);
            starts.push(block_start(row));
        }
        Ok(Self(Arc::new(TemporalRowProvenanceInner {
            original_ticks: ticks,
            block_start: starts,
            user_sampling_weights,
            budget: budget.clone(),
            _lease: lease,
        })))
    }

    pub(super) fn original_ticks(&self) -> &[usize] {
        &self.0.original_ticks
    }
    pub(super) fn block_start(&self) -> &[bool] {
        &self.0.block_start
    }
    pub(crate) fn user_sampling_weights(&self) -> bool {
        self.0.user_sampling_weights
    }
    pub(crate) fn budget(&self) -> &antecedent_core::TemporalBudget {
        &self.0.budget
    }
}

/// Complete-window execution input with original positions, contiguous runs,
/// observed source history, and the admission retained with the owned table.
#[derive(Clone)]
pub(super) struct CompleteWindowInput(Arc<CompleteWindowInputInner>);

struct CompleteWindowInputInner {
    table: TabularData,
    /// The caller-owned original source, retained separately for actual rollout history.
    source: TimeSeriesData,
    provenance: Option<TemporalRowProvenance>,
    common_history: u32,
    /// Compact row position to original source tick.
    ticks: Vec<usize>,
    /// Original source coordinate for each compact row; may repeat or be unordered.
    original_ticks: Vec<usize>,
    /// Half-open compact-index ranges that are contiguous in original time.
    runs: Vec<std::ops::Range<usize>>,
    _lease: antecedent_core::execution::AdmissionLease,
}

impl CompleteWindowInput {
    /// Construct observed windows with a shared class-wide history boundary.
    /// Original source storage remains caller-owned; the lease covers the new
    /// table, mapping, runs and bounded construction copies, not allocator peaks.
    pub(super) fn build(
        aligned: &LagAlignedScm,
        series: &TimeSeriesData,
        common_history: u32,
        provenance: Option<&TemporalRowProvenance>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        if common_history < aligned.history {
            return Err(CausalError::Compile {
                message: "shared attribution history is shorter than the graph history".into(),
            });
        }
        let n = series.row_count();
        if provenance.is_some_and(|p| p.original_ticks().len() != n) {
            return Err(CausalError::Compile {
                message: "temporal refit provenance does not match series length".into(),
            });
        }
        let h = usize::try_from(common_history).map_err(|_| CausalError::Resource {
            message: "attribution history exceeds addressable rows".into(),
        })?;
        if n <= h {
            return Err(CausalError::Compile {
                message: "temporal attribution has no complete observed lag window".into(),
            });
        }
        if series.schema().variables().len() != aligned.contemporaneous_count() {
            return Err(CausalError::Compile {
                message: "lag-aligned series schema does not match the unfolded graph".into(),
            });
        }
        let rows = u64::try_from(n).map_err(|_| CausalError::Resource {
            message: "attribution row count overflow".into(),
        })?;
        let cols = u64::try_from(aligned.columns.len()).map_err(|_| CausalError::Resource {
            message: "attribution column count overflow".into(),
        })?;
        let work = rows
            .checked_mul(cols.checked_add(4).ok_or_else(|| CausalError::Resource {
                message: "attribution work overflow".into(),
            })?)
            .ok_or_else(|| CausalError::Resource { message: "attribution work overflow".into() })?;
        budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(32))
            .and_then(|v| rows.checked_mul(48).and_then(|r| v.checked_add(r)))
            .and_then(|v| cols.checked_mul(256).and_then(|c| v.checked_add(c)))
            .and_then(|v| v.checked_add(4096))
            .ok_or_else(|| CausalError::Resource {
                message: "attribution bytes overflow".into(),
            })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "attribution byte limit".into() });
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mask = series.storage().analysis_mask();
        let mut ticks = Vec::new();
        ticks.try_reserve_exact(n - h).map_err(|_| CausalError::Resource {
            message: "attribution row map allocation refused".into(),
        })?;
        let mut contiguous = 0_usize;
        for tick in 0..n {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            let continues = tick > 0
                && provenance.is_none_or(|p| {
                    !p.block_start()[tick]
                        && p.original_ticks()[tick - 1].checked_add(1)
                            == Some(p.original_ticks()[tick])
                });
            contiguous = if continues { contiguous.saturating_add(1) } else { 1 };
            if tick >= h && contiguous > h && mask.is_none_or(|m| m.is_valid(tick)) {
                ticks.push(tick);
            }
        }
        if ticks.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal attribution has no eligible complete windows".into(),
            });
        }
        let mut originals = Vec::new();
        originals.try_reserve_exact(ticks.len()).map_err(|_| CausalError::Resource {
            message: "attribution original row map allocation refused".into(),
        })?;
        for &tick in &ticks {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            originals.push(provenance.map_or(tick, |p| p.original_ticks()[tick]));
        }
        let mut runs = Vec::new();
        runs.try_reserve_exact(ticks.len()).map_err(|_| CausalError::Resource {
            message: "attribution run allocation refused".into(),
        })?;
        let mut start = 0;
        for position in 1..ticks.len() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            let row = ticks[position];
            if ticks[position - 1].checked_add(1) != Some(row)
                || originals[position - 1].checked_add(1) != Some(originals[position])
                || provenance.is_some_and(|p| p.block_start()[row])
            {
                runs.push(start..position);
                start = position;
            }
        }
        runs.push(start..ticks.len());
        let mut owned = Vec::new();
        owned.try_reserve_exact(aligned.columns.len()).map_err(|_| CausalError::Resource {
            message: "attribution column allocation refused".into(),
        })?;
        for &(variable, lag) in aligned.columns.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            let source = series.float64_slice(variable)?;
            if source.len() != n {
                return Err(CausalError::Compile {
                    message: "lag-aligned column length does not match the source".into(),
                });
            }
            let lag = usize::try_from(lag).map_err(|_| CausalError::Resource {
                message: "attribution lag overflow".into(),
            })?;
            if lag > h {
                return Err(CausalError::Compile {
                    message: "column lag exceeds shared history".into(),
                });
            }
            let mut values = Vec::new();
            values.try_reserve_exact(ticks.len()).map_err(|_| CausalError::Resource {
                message: "attribution value allocation refused".into(),
            })?;
            for &tick in &ticks {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                values.push(source[tick - lag]);
            }
            owned.push((format!("v{}_{lag}", variable.raw()), values));
        }
        let refs: Vec<(&str, &[f64])> =
            owned.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
        let mut table = TabularData::from_f64_columns(refs)?;
        if provenance.is_none_or(TemporalRowProvenance::user_sampling_weights) {
            if let Some(weights) = series.storage().weights() {
                let mut selected = Vec::new();
                selected.try_reserve_exact(ticks.len()).map_err(|_| CausalError::Resource {
                    message: "attribution weight allocation refused".into(),
                })?;
                for &tick in &ticks {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    selected.push(weights[tick]);
                }
                table = table.with_weights(selected.into())?;
            }
        }
        Ok(Self(Arc::new(CompleteWindowInputInner {
            table,
            source: series.clone(),
            provenance: provenance.cloned(),
            common_history,
            ticks,
            original_ticks: originals,
            runs,
            _lease: lease,
        })))
    }

    pub(super) fn table(&self) -> &TabularData {
        &self.0.table
    }
    pub(super) fn source(&self) -> &TimeSeriesData {
        &self.0.source
    }
    pub(super) fn ticks(&self) -> &[usize] {
        &self.0.ticks
    }
    pub(super) fn original_ticks(&self) -> &[usize] {
        &self.0.original_ticks
    }
    pub(super) fn block_length(&self, n: usize) -> usize {
        let span = usize::try_from(self.0.common_history).unwrap_or(usize::MAX).saturating_add(1);
        antecedent_data::circular_block_length(span, n)
    }
    pub(super) fn runs(&self) -> &[std::ops::Range<usize>] {
        &self.0.runs
    }

    pub(super) fn selected_runs(
        &self,
        positions: &[usize],
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<SelectedRuns, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let n = u64::try_from(positions.len())
            .map_err(|_| CausalError::Resource { message: "temporal run size overflow".into() })?;
        let log = u64::from(usize::BITS - self.0.runs.len().leading_zeros());
        let work = n.checked_mul(log + 2).ok_or_else(|| CausalError::Resource {
            message: "temporal run work overflow".into(),
        })?;
        budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = n.checked_mul(32).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
            CausalError::Resource { message: "temporal run bytes overflow".into() }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal run byte limit".into() });
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut ranges = Vec::new();
        ranges.try_reserve_exact(positions.len()).map_err(|_| CausalError::Resource {
            message: "temporal run allocation refused".into(),
        })?;
        let mut start = 0;
        for (index, &row) in positions.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            if row >= self.0.ticks.len() {
                return Err(CausalError::Compile {
                    message: "temporal population row is out of range".into(),
                });
            }
            if index > 0 {
                let prev = positions[index - 1];
                let run = self.0.runs.partition_point(|range| range.end <= row);
                if prev.checked_add(1) != Some(row) || self.0.runs[run].start == row {
                    ranges.push(start..index);
                    start = index;
                }
            }
        }
        if !positions.is_empty() {
            ranges.push(start..positions.len());
        }
        Ok(SelectedRuns { ranges, _lease: lease })
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
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let input =
            CompleteWindowInput::build(&aligned, &series, 1, None, &mut budget, &ctx).unwrap();
        assert_eq!(input.table().float64_slice(VariableId::from_raw(1)).unwrap(), &y[1..]);
        assert_eq!(input.table().float64_slice(VariableId::from_raw(2)).unwrap(), &[0.0, 1.0, 2.0]);
    }
    #[test]
    fn complete_windows_use_observed_values_and_original_tick_coordinates() {
        let mut graph = TemporalDag::empty();
        let parent = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let child = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(parent, child).unwrap();
        let x = [5.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];
        let y = [7.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .unwrap();
        let series = series.with_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into()).unwrap();
        let series = series
            .with_analysis_mask(antecedent_data::ValidityBitmap::from_bytes(vec![0x6f], 7).unwrap())
            .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let input =
            CompleteWindowInput::build(&aligned, &series, 2, None, &mut budget, &ctx).unwrap();
        assert_eq!(input.0.ticks, [2, 3, 5, 6]);
        assert_eq!(input.0.runs, [0..2, 2..4]);
        let selected = input.selected_runs(&[0, 2, 3], &mut budget, &ctx).unwrap();
        assert_eq!(selected.ranges(), [0..1, 1..3]);
        let repeated = input.selected_runs(&[2, 2, 3], &mut budget, &ctx).unwrap();
        assert_eq!(repeated.ranges(), [0..1, 1..3]);
        let lag_col =
            aligned.columns.iter().position(|&(v, lag)| v.raw() == 0 && lag == 2).unwrap();
        assert_eq!(
            input.0.table.float64_slice(VariableId::from_raw(lag_col as u32)).unwrap(),
            &[5.0, 11.0, 17.0, 19.0]
        );
        assert_eq!(input.original_ticks().iter().position(|&tick| tick == 5), Some(2));
        assert_eq!(input.0.table.storage().weights().unwrap(), &[3.0, 4.0, 6.0, 7.0]);
        assert!(!input.original_ticks().contains(&1));
        assert_eq!(input.0.source.float64_slice(VariableId::from_raw(0)).unwrap(), x);
    }
    #[test]
    fn refit_provenance_retains_duplicate_ticks_and_block_starts() {
        let ctx = antecedent_core::ExecutionContext::for_tests(11);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let rows = TemporalRowProvenance::admit_fixed_blocks(
            &[8, 9, 2, 3, 2],
            Some(2),
            false,
            &mut budget,
            &ctx,
        )
        .unwrap();
        let shared = rows.clone();
        drop(rows);
        assert_eq!(shared.original_ticks(), [8, 9, 2, 3, 2]);
        assert_eq!(shared.block_start(), [true, false, true, false, true]);
        assert!(!shared.user_sampling_weights());
        assert!(
            TemporalRowProvenance::admit_fixed_blocks(&[1, 2], Some(0), false, &mut budget, &ctx)
                .is_err()
        );
    }
    #[test]
    fn complete_refit_windows_keep_duplicate_source_coordinates_and_synthetic_weight_kind() {
        let mut graph = TemporalDag::empty();
        let parent = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let child = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(parent, child).unwrap();
        let x = [0., 1., 2., 0., 1., 2., 3.];
        let y = [4., 5., 6., 4., 5., 6., 7.];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .unwrap()
                .with_weights(vec![1.; 7].into())
                .unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let map = TemporalRowProvenance::admit_fixed_blocks(
            &[8, 9, 10, 8, 9, 10, 11],
            Some(3),
            false,
            &mut budget,
            &ctx,
        )
        .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let input = CompleteWindowInput::build(&aligned, &series, 2, Some(&map), &mut budget, &ctx)
            .unwrap();
        // Fixed blocks start at positions 0, 3 and 6. With history two,
        // the complete windows are [0, 1, 2] and [3, 4, 5]; the last
        // one-row block cannot supply the two preceding observations.
        assert_eq!(map.block_start(), [true, false, false, true, false, false, true]);
        assert_eq!(input.ticks(), [2, 5]);
        assert_eq!(input.original_ticks(), [10, 10]);
        assert_eq!(input.runs(), [0..1, 1..2]);
        assert!(input.table().storage().weights().is_none());
    }
    #[test]
    fn provenance_shares_admission_until_last_clone_and_honors_cancellation() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(3).with_admission(pool.clone());
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let before = pool.available().bytes();
        let rows =
            TemporalRowProvenance::admit_fixed_blocks(&[1, 2], Some(2), false, &mut budget, &ctx)
                .unwrap();
        let charged = pool.available().bytes();
        assert!(charged < before);
        let retained = rows.clone();
        drop(rows);
        assert_eq!(pool.available().bytes(), charged);
        drop(retained);
        assert_eq!(pool.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            TemporalRowProvenance::admit_fixed_blocks(&[], None, false, &mut budget, &ctx),
            Err(CausalError::Cancelled { .. })
        ));
        assert_eq!(pool.available().bytes(), before);
    }
    #[test]
    fn rollout_history_keeps_real_contemporaneous_values_when_auxiliary_history_is_absent() {
        let mut graph = TemporalDag::empty();
        let parent = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let child = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(parent, child).unwrap();
        let x = [5.0, 11.0, 13.0];
        let y = [7.0, 31.0, 37.0];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let input =
            CompleteWindowInput::build(&aligned, &series, 2, None, &mut budget, &ctx).unwrap();
        let history = aligned.observed_initial_history(&input, 1, 2, &mut budget, &ctx).unwrap();
        let x_now = aligned.columns.iter().position(|&(v, lag)| v.raw() == 0 && lag == 0).unwrap();
        let x_lag = aligned.columns.iter().position(|&(v, lag)| v.raw() == 0 && lag == 2).unwrap();
        let width = aligned.columns.len();
        assert_eq!(history.batches[0].values[x_now], 13.0);
        assert_eq!(history.batches[1].values[x_now], 11.0);
        assert!(history.available[width + x_now]);
        assert!(!history.available[width + x_lag]);
        assert!(history.batches[1].values[x_lag].is_nan());
    }
    #[test]
    fn a_missing_historical_cell_stays_missing_on_an_eligible_anchor() {
        let mut graph = TemporalDag::empty();
        let parent = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let child = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(parent, child).unwrap();
        let x = [f64::NAN, 11.0, 13.0, 17.0];
        let y = [7.0, 31.0, 37.0, 41.0];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let input =
            CompleteWindowInput::build(&aligned, &series, 2, None, &mut budget, &ctx).unwrap();
        let lag = aligned.columns.iter().position(|&(v, l)| v.raw() == 0 && l == 2).unwrap();
        let antecedent_data::OwnedColumn::Float64(column) = &input.table().storage().columns()[lag]
        else {
            panic!("expected float column")
        };
        assert_eq!(input.original_ticks(), [2, 3]);
        assert!(column.values[0].is_nan());
        assert!(!column.validity.is_valid(0));
        assert_eq!(column.values[1], 11.0);
        assert!(column.validity.is_valid(1));
    }
    #[test]
    fn rollout_history_does_not_cross_a_refit_block_boundary() {
        let mut graph = TemporalDag::empty();
        let parent = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let child = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(parent, child).unwrap();
        let x = [1., 2., 3., 4., 5., 6., 7.];
        let y = [8., 9., 10., 11., 12., 13., 14.];
        let series =
            TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                .unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let provenance = TemporalRowProvenance::admit_fixed_blocks(
            &[0, 1, 2, 3, 4, 5, 6],
            Some(4),
            false,
            &mut budget,
            &ctx,
        )
        .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let input =
            CompleteWindowInput::build(&aligned, &series, 2, Some(&provenance), &mut budget, &ctx)
                .unwrap();
        let history = aligned.observed_initial_history(&input, 1, 2, &mut budget, &ctx).unwrap();
        let x_now = aligned.columns.iter().position(|&(v, lag)| v.raw() == 0 && lag == 0).unwrap();
        let x_lag = aligned.columns.iter().position(|&(v, lag)| v.raw() == 0 && lag == 2).unwrap();
        let width = aligned.columns.len();
        assert_eq!(history.batches[1].values[x_now], 6.0);
        assert!(history.available[width + x_now]);
        assert!(!history.available[width + x_lag]);
    }
    #[test]
    fn zero_history_keeps_the_first_observation() {
        let mut graph = TemporalDag::empty();
        let x = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x, y).unwrap();
        let xs = [3.0, 5.0, 7.0];
        let ys = [11.0, 13.0, 17.0];
        let series =
            TimeSeriesData::from_f64_columns([("x", xs.as_slice()), ("y", ys.as_slice())], 1)
                .unwrap();
        let aligned = LagAlignedScm::unfold(&graph, &series).unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let input =
            CompleteWindowInput::build(&aligned, &series, 0, None, &mut budget, &ctx).unwrap();
        assert_eq!(input.original_ticks(), [0, 1, 2]);
        assert_eq!(input.table().float64_slice(VariableId::from_raw(0)).unwrap(), xs);
    }
}
