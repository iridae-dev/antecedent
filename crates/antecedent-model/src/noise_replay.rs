//! Batch exogenous-noise replay.
//!
//! A [`NoiseTable`] retains one exogenous-noise column per node. Columns are drawn from
//! deterministic streams keyed by `(draw, node, row block)`, fixed to chosen values,
//! redrawn on a fresh stream, or abducted from observed data, and a
//! [`NoiseReplayer`] evaluates the whole model from a table under an intervention
//! overlay without cloning the model. Replay uses the same per-node precedence as
//! forward sampling (hard set, stochastic policy, soft override or fitted mechanism,
//! additive shift), so a sampled run and a replay of its table are bit-identical.
//!
//! A coalition of fixed-versus-resampled noise coordinates replays from two tables and a
//! node mask ([`NoiseSource::coalition`]); nothing is copied per coalition, so the cost of
//! a coalition is one pass of the compiled kernel ([`NoiseReplayer::replay_into`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain, mix64};
use antecedent_graph::DenseNodeId;

use crate::batch::{MechanismWorkspace, ParentBatch, ValueBatch, ValueBatchMut};
use crate::compile::{CompiledCausalModel, MechanismSlot, ParentGatherPlan};
use crate::error::ModelError;
use crate::mechanism::{
    NoiseInferenceMode, categorical_noise, noise_inference_mode_of, reference_noise,
    sample_noise_column,
};
use crate::overlay::ModelView;
use crate::sample::{
    NodeEvaluation, NodeNoise, evaluate_node, refuse_cross_family_soft, soft_to_slot,
};

/// Stream domain of noise replay. Noise streams are keyed by `(draw, node, row block)` and
/// policy streams by `(salt, node)`, so a node's draws do not depend on the other nodes,
/// on the overlay, or on how many rows follow its block.
pub const NOISE_REPLAY_STREAM_DOMAIN: StreamDomain =
    StreamDomain::external(StreamDomain::EXTERNAL_BASE + 0x4E52);

/// Rows per independently seeded noise block. Mechanisms whose noise is serially
/// dependent across rows (state-space and dynamic mechanisms) are drawn as one block.
pub const NOISE_ROW_BLOCK: usize = 4096;

const NOISE_ROLE: u64 = 0x4E52_4E5A;
const POLICY_ROLE: u64 = 0x4E52_504C;
const NOISE_TABLE_BUDGET: &str = "noise replay memory budget";

fn noise_stream_index(draw: u64, node: usize, block: usize) -> u64 {
    mix64(mix64(mix64(draw ^ NOISE_ROLE) ^ node as u64) ^ block as u64)
}

fn policy_stream_index(salt: u64, node: usize) -> u64 {
    mix64(mix64(salt ^ POLICY_ROLE) ^ node as u64)
}

/// Where one node's noise column came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NoiseOrigin {
    /// Drawn from the node's replay streams on draw `draw` (`0` is the first sample).
    Sampled {
        /// Draw index of the stream family.
        draw: u64,
    },
    /// Set by the caller (a value, a reference noise, or columns of another table).
    Fixed,
    /// Recovered from observed values, with the way the noise was inferred. A
    /// [`NoiseInferenceMode::Posterior`] column rests on the rank-preservation assumption
    /// of the counterfactual disclosures.
    Abducted(NoiseInferenceMode),
    /// Drawn from the mechanism prior because the factual column was absent.
    Assumed,
    /// No noise was drawn: the node is hard-set by the overlay and its mechanism is unfitted.
    Unused,
}

/// A node-by-row exogenous-noise matrix with the provenance of every coordinate.
///
/// Column-major: `values[node * n_rows + row]`, the layout of [`ValueBatch`].
#[derive(Clone, Debug)]
pub struct NoiseTable {
    n_rows: usize,
    n_nodes: usize,
    values: Vec<f64>,
    origins: Vec<NoiseOrigin>,
}

/// What [`NoiseTable::fix`] sets the chosen noise coordinates to.
#[derive(Clone, Copy, Debug)]
pub enum NoiseFix<'a> {
    /// Each node's reference noise ([`reference_noise`]): the value that reconstructs it at
    /// its typical output.
    Reference,
    /// One noise value for every row of every listed node.
    Constant(f64),
    /// One column of `n_rows` values per listed node, concatenated in the order listed.
    Columns(&'a [f64]),
    /// The same nodes' columns of another table of the same shape.
    FromTable(&'a NoiseTable),
}

impl NoiseTable {
    /// Bytes a table of this shape occupies, or `None` when the size overflows.
    #[must_use]
    pub fn bytes_for(n_rows: usize, n_nodes: usize) -> Option<u64> {
        let bytes = n_rows.checked_mul(n_nodes)?.checked_mul(core::mem::size_of::<f64>())?;
        u64::try_from(bytes).ok()
    }

    /// Draw the noise of every node of `view`'s model on the first draw (`0`).
    ///
    /// Noise is drawn for every fitted node, including nodes the overlay hard-sets or
    /// replaces by a stochastic policy, so one table serves any overlay; a soft override
    /// draws from the override's own noise scale. A hard-set node whose mechanism is
    /// unfitted is [`NoiseOrigin::Unused`].
    ///
    /// # Errors
    ///
    /// `n_rows == 0`, an invalid overlay, an unfitted mechanism that is not hard-set, a
    /// table larger than the context's hard memory limit
    /// ([`ModelError::ResourceLimit`], checked before allocating), or cancellation.
    pub fn sample(
        view: &ModelView<'_>,
        n_rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, ModelError> {
        if n_rows == 0 {
            return Err(ModelError::Shape { message: "n_rows must be > 0".into() });
        }
        view.overlay.validate()?;
        let n_nodes = view.model.n_nodes();
        check_budget(ctx, 1, n_rows, n_nodes)?;
        let mut values = vec![0.0; n_rows * n_nodes];
        let mut origins = Vec::with_capacity(n_nodes);
        for node in 0..n_nodes {
            check_cancelled(ctx)?;
            let column = &mut values[node * n_rows..(node + 1) * n_rows];
            origins.push(draw_column(view, node, 0, ctx, column)?);
        }
        Ok(Self { n_rows, n_nodes, values, origins })
    }

    /// Wrap abduced noise (`values[node * n_rows + row]`) with its per-node inference mode.
    ///
    /// `assumed[node]` marks a column that was prior-sampled because the factual column
    /// was absent; every other column is [`NoiseOrigin::Abducted`] with the mode its
    /// mechanism family reports ([`noise_inference_mode_of`]).
    ///
    /// # Errors
    ///
    /// Shape mismatches.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "node ids are u32 by construction (DenseNodeId), so every node position fits u32"
    )]
    pub fn from_abducted(
        model: &CompiledCausalModel,
        n_rows: usize,
        values: Vec<f64>,
        assumed: &[bool],
    ) -> Result<Self, ModelError> {
        let n_nodes = model.n_nodes();
        if n_rows == 0
            || n_rows.checked_mul(n_nodes) != Some(values.len())
            || assumed.len() != n_nodes
        {
            return Err(ModelError::Shape {
                message: "abducted noise must be n_rows x n_nodes with one assumed flag per node"
                    .into(),
            });
        }
        let origins = (0..n_nodes)
            .map(|node| {
                if assumed[node] {
                    NoiseOrigin::Assumed
                } else {
                    let slot = model.mechanisms.get(DenseNodeId::from_raw(node as u32));
                    NoiseOrigin::Abducted(noise_inference_mode_of(slot))
                }
            })
            .collect();
        Ok(Self { n_rows, n_nodes, values, origins })
    }

    /// Rows.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Nodes (columns).
    #[must_use]
    pub const fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Flat column-major storage.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Noise column of `node`.
    ///
    /// # Errors
    ///
    /// Node out of range.
    pub fn column(&self, node: usize) -> Result<&[f64], ModelError> {
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "noise column out of range".into() });
        }
        Ok(&self.values[node * self.n_rows..(node + 1) * self.n_rows])
    }

    /// Provenance of every node's column, in dense node order.
    #[must_use]
    pub fn origins(&self) -> &[NoiseOrigin] {
        &self.origins
    }

    /// Provenance of `node`'s column.
    #[must_use]
    pub fn origin(&self, node: usize) -> Option<NoiseOrigin> {
        self.origins.get(node).copied()
    }

    /// How `node`'s noise was inferred, for an abducted column.
    #[must_use]
    pub fn inference_mode(&self, node: usize) -> Option<NoiseInferenceMode> {
        match self.origin(node)? {
            NoiseOrigin::Abducted(mode) => Some(mode),
            _ => None,
        }
    }

    /// Whether any column is a posterior draw, so results replayed from this table rest on
    /// the rank-preservation assumption.
    #[must_use]
    pub fn has_posterior_noise(&self) -> bool {
        self.origins.contains(&NoiseOrigin::Abducted(NoiseInferenceMode::Posterior))
    }

    /// Set the listed nodes' noise coordinates; their origin becomes [`NoiseOrigin::Fixed`].
    ///
    /// Noise must be finite, and strictly inside `(0, 1)` for a categorical node.
    ///
    /// # Errors
    ///
    /// A node out of range, a shape mismatch, or an invalid noise value; the table is
    /// unchanged on error.
    pub fn fix(
        &mut self,
        model: &CompiledCausalModel,
        nodes: &[DenseNodeId],
        fix: NoiseFix<'_>,
    ) -> Result<(), ModelError> {
        self.check_model(model)?;
        let n_rows = self.n_rows;
        match fix {
            NoiseFix::Columns(cols) if cols.len() != nodes.len().saturating_mul(n_rows) => {
                return Err(ModelError::Shape {
                    message: "fixed noise must hold n_rows values per listed node".into(),
                });
            }
            NoiseFix::FromTable(other)
                if other.n_rows != n_rows || other.n_nodes != self.n_nodes =>
            {
                return Err(ModelError::Shape {
                    message: "fixed noise table must match the table's shape".into(),
                });
            }
            _ => {}
        }
        for (i, node) in nodes.iter().enumerate() {
            let idx = node.as_usize();
            if idx >= self.n_nodes {
                return Err(ModelError::Shape { message: "fixed noise node out of range".into() });
            }
            let slot = model.mechanisms.get(*node);
            validate_fixed(slot, &fixed_source(fix, i, idx, n_rows, slot))?;
        }
        for (i, node) in nodes.iter().enumerate() {
            let idx = node.as_usize();
            let slot = model.mechanisms.get(*node);
            let column = &mut self.values[idx * n_rows..(idx + 1) * n_rows];
            match fixed_source(fix, i, idx, n_rows, slot) {
                Fixed::Constant(v) => column.fill(v),
                Fixed::Column(c) => column.copy_from_slice(c),
            }
            self.origins[idx] = NoiseOrigin::Fixed;
        }
        Ok(())
    }

    /// Redraw the listed nodes' noise on stream family `draw`, leaving every other column
    /// untouched. Draw `0` is the family [`Self::sample`] uses, so a fresh resample needs a
    /// draw of at least `1`; the same `(draw, seed)` always reproduces the same columns.
    ///
    /// # Errors
    ///
    /// `draw == 0`, a node out of range, an invalid overlay, an unfitted mechanism that is
    /// not hard-set, or cancellation; columns redrawn before the error keep their new values.
    pub fn randomize(
        &mut self,
        view: &ModelView<'_>,
        nodes: &[DenseNodeId],
        draw: u64,
        ctx: &ExecutionContext,
    ) -> Result<(), ModelError> {
        if draw == 0 {
            return Err(ModelError::Shape {
                message: "randomize needs a draw of at least 1; draw 0 reproduces the sampled \
                          noise"
                    .into(),
            });
        }
        self.check_model(view.model)?;
        view.overlay.validate()?;
        let n_rows = self.n_rows;
        for node in nodes {
            let idx = node.as_usize();
            if idx >= self.n_nodes {
                return Err(ModelError::Shape {
                    message: "randomized noise node out of range".into(),
                });
            }
            check_cancelled(ctx)?;
            let column = &mut self.values[idx * n_rows..(idx + 1) * n_rows];
            self.origins[idx] = draw_column(view, idx, draw, ctx, column)?;
        }
        Ok(())
    }

    fn check_model(&self, model: &CompiledCausalModel) -> Result<(), ModelError> {
        if model.n_nodes() == self.n_nodes {
            Ok(())
        } else {
            Err(ModelError::Shape { message: "noise table width differs from the model".into() })
        }
    }
}

/// The values `fix` assigns to the `i`-th listed node (dense index `idx`).
fn fixed_source<'a>(
    fix: NoiseFix<'a>,
    i: usize,
    idx: usize,
    n_rows: usize,
    slot: &MechanismSlot,
) -> Fixed<'a> {
    match fix {
        NoiseFix::Reference => Fixed::Constant(reference_noise(slot)),
        NoiseFix::Constant(v) => Fixed::Constant(v),
        NoiseFix::Columns(cols) => Fixed::Column(&cols[i * n_rows..(i + 1) * n_rows]),
        NoiseFix::FromTable(other) => {
            Fixed::Column(&other.values[idx * n_rows..(idx + 1) * n_rows])
        }
    }
}

enum Fixed<'a> {
    Constant(f64),
    Column(&'a [f64]),
}

fn validate_fixed(slot: &MechanismSlot, fixed: &Fixed<'_>) -> Result<(), ModelError> {
    let categorical = matches!(
        slot,
        MechanismSlot::Discrete { .. }
            | MechanismSlot::DiscreteBasis { .. }
            | MechanismSlot::TreeCategorical { .. }
    );
    let check = |v: f64| -> Result<(), ModelError> {
        if categorical {
            categorical_noise(v).map(|_| ())
        } else if v.is_finite() {
            Ok(())
        } else {
            Err(ModelError::Numerical { message: format!("fixed noise must be finite, got {v}") })
        }
    };
    match *fixed {
        Fixed::Constant(v) => check(v),
        Fixed::Column(c) => c.iter().try_for_each(|&v| check(v)),
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), ModelError> {
    if ctx.cancellation.is_cancelled() { Err(ModelError::Cancelled) } else { Ok(()) }
}

/// Refuse before allocating `tables` matrices of `n_rows x n_nodes` doubles when they exceed
/// the context's hard memory limit, and observe cancellation.
fn check_budget(
    ctx: &ExecutionContext,
    tables: usize,
    n_rows: usize,
    n_nodes: usize,
) -> Result<(), ModelError> {
    check_cancelled(ctx)?;
    let Some(limit_bytes) = ctx.memory.hard_limit_bytes else {
        return Ok(());
    };
    let required_bytes = NoiseTable::bytes_for(n_rows, n_nodes)
        .and_then(|b| b.checked_mul(tables as u64))
        .unwrap_or(u64::MAX);
    if required_bytes > limit_bytes {
        return Err(ModelError::ResourceLimit {
            message: NOISE_TABLE_BUDGET,
            required_bytes,
            limit_bytes,
        });
    }
    Ok(())
}

/// Draw `node`'s noise column on stream family `draw`, block by block.
#[allow(
    clippy::cast_possible_truncation,
    reason = "node ids are u32 by construction (DenseNodeId), so every node position fits u32"
)]
fn draw_column(
    view: &ModelView<'_>,
    node: usize,
    draw: u64,
    ctx: &ExecutionContext,
    out: &mut [f64],
) -> Result<NoiseOrigin, ModelError> {
    let model = view.model;
    let overlay = &view.overlay;
    let id = DenseNodeId::from_raw(node as u32);
    let base = model.mechanisms.get(id);
    let owned;
    let slot = if let Some(soft) = &overlay.soft[node] {
        refuse_cross_family_soft(base, soft)?;
        let n_parents = model.gather_for(id).map_or(0, ParentGatherPlan::n_parents);
        owned = soft_to_slot(soft, n_parents)?;
        &owned
    } else {
        base
    };
    if matches!(slot, MechanismSlot::Vacant | MechanismSlot::Pending { .. })
        && overlay.hard_set[node].is_some()
    {
        out.fill(0.0);
        return Ok(NoiseOrigin::Unused);
    }
    let serial = matches!(
        slot,
        MechanismSlot::LinearGaussianStateSpace { .. }
            | MechanismSlot::ConditionalLinearGaussianStateSpace { .. }
            | MechanismSlot::Dynamic { .. }
    );
    let block_len = if serial { out.len() } else { NOISE_ROW_BLOCK };
    for (block, chunk) in out.chunks_mut(block_len).enumerate() {
        let mut rng =
            ctx.rng.stream_for(NOISE_REPLAY_STREAM_DOMAIN, noise_stream_index(draw, node, block));
        sample_noise_column(slot, chunk.len(), &mut rng, chunk)?;
    }
    Ok(NoiseOrigin::Sampled { draw })
}

/// The noise a replay reads: one table, or a coalition of two.
#[derive(Clone, Copy, Debug)]
pub enum NoiseSource<'a> {
    /// Every node reads its column of one table.
    Table(&'a NoiseTable),
    /// A node in `fixed_nodes` reads its column of `fixed`, every other node reads
    /// `resampled`. Which table holds which noise is the caller's choice (typically the
    /// observed noise and an independent redraw), and no column is copied.
    Coalition {
        /// Noise of the nodes held fixed.
        fixed: &'a NoiseTable,
        /// Noise of the nodes that are resampled.
        resampled: &'a NoiseTable,
        /// Per node: `true` reads `fixed`.
        fixed_nodes: &'a [bool],
    },
}

impl<'a> NoiseSource<'a> {
    /// A coalition source.
    ///
    /// # Errors
    ///
    /// The two tables differ in shape, or the mask is not one flag per node.
    pub fn coalition(
        fixed: &'a NoiseTable,
        resampled: &'a NoiseTable,
        fixed_nodes: &'a [bool],
    ) -> Result<Self, ModelError> {
        if fixed.n_rows != resampled.n_rows
            || fixed.n_nodes != resampled.n_nodes
            || fixed_nodes.len() != fixed.n_nodes
        {
            return Err(ModelError::Shape {
                message: "coalition noise tables must share a shape and the mask one flag per \
                          node"
                    .into(),
            });
        }
        Ok(Self::Coalition { fixed, resampled, fixed_nodes })
    }

    fn shape_table(&self) -> &'a NoiseTable {
        match *self {
            Self::Table(table) | Self::Coalition { fixed: table, .. } => table,
        }
    }

    fn column(&self, node: usize) -> &'a [f64] {
        let (table, n_rows) = match *self {
            Self::Table(table) => (table, table.n_rows),
            Self::Coalition { fixed, resampled, fixed_nodes } => {
                let table = if fixed_nodes[node] { fixed } else { resampled };
                (table, table.n_rows)
            }
        };
        &table.values[node * n_rows..(node + 1) * n_rows]
    }
}

/// How far a replay got.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayOutcome {
    /// Nodes evaluated, as a prefix of the topological order. Their columns are exact; the
    /// remaining columns of the output buffer are untouched.
    pub completed_nodes: usize,
    /// Cancellation stopped the replay before every node was evaluated.
    pub cancelled: bool,
}

/// Evaluates a compiled model from retained noise, reusing its scratch buffers across
/// replays. One replayer serves any number of tables, overlays and coalitions; the model
/// is only borrowed through the [`ModelView`] and is never cloned.
#[derive(Debug, Default)]
pub struct NoiseReplayer {
    ws: MechanismWorkspace,
    parent_buf: Vec<f64>,
}

impl NoiseReplayer {
    /// A replayer with empty scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scratch growth events and retained capacity, which stay constant once a replay of
    /// the largest shape has run (the workspace-reuse gate).
    #[must_use]
    pub fn scratch_footprint(&self) -> (u32, usize) {
        (self.ws.grow_count, self.parent_buf.capacity() + self.ws.parents.capacity())
    }

    /// Replay `source` through `view` into a caller-owned column-major buffer. For a
    /// coalition pass [`NoiseSource::coalition`] and reuse one buffer and one replayer
    /// across every mask.
    ///
    /// Nodes are evaluated in topological order with the same precedence as forward
    /// sampling. A stochastic-policy node draws from a stream keyed by `(policy_salt,
    /// node)`, so replays with the same salt agree. Cancellation is observed once per
    /// node and returns the exact prefix computed so far, flagged in the outcome.
    ///
    /// # Errors
    ///
    /// A shape mismatch, an invalid overlay, or mechanism failures (a categorical node
    /// whose noise is outside `(0, 1)`, a cross-family soft override).
    pub fn replay_into(
        &mut self,
        view: &ModelView<'_>,
        source: NoiseSource<'_>,
        policy_salt: u64,
        values: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<ReplayOutcome, ModelError> {
        let model = view.model;
        let table = source.shape_table();
        let (n_rows, n_nodes) = (table.n_rows, model.n_nodes());
        if table.n_nodes != n_nodes {
            return Err(ModelError::Shape {
                message: "noise table width differs from the model".into(),
            });
        }
        view.overlay.validate()?;
        let mut values = ValueBatchMut::new(n_rows, n_nodes, values)?;
        let overlay = view.overlay.as_ref();
        let Self { ws, parent_buf } = self;
        let mut completed_nodes = 0;
        let eval = NodeEvaluation { model, overlay, ctx: Some(ctx) };
        for gather in overlay.gathers(model) {
            if ctx.cancellation.is_cancelled() {
                return Ok(ReplayOutcome { completed_nodes, cancelled: true });
            }
            let idx = gather.child.as_usize();
            let need = gather.n_parents().max(1).saturating_mul(n_rows);
            if parent_buf.len() < need {
                parent_buf.resize(need, 0.0);
            }
            gather.gather(values.values, n_rows, parent_buf);
            let parents = ParentBatch {
                n_rows,
                n_parents: gather.n_parents(),
                values: &parent_buf[..gather.n_parents().saturating_mul(n_rows)],
            };
            let mut policy_rng = ctx
                .rng
                .stream_for(NOISE_REPLAY_STREAM_DOMAIN, policy_stream_index(policy_salt, idx));
            let noise =
                NodeNoise::Column { noise: source.column(idx), policy_rng: &mut policy_rng };
            match evaluate_node(eval, gather, parents, noise, &mut values, ws) {
                // A value program polled the same token mid-node: report the exact prefix
                // of completed nodes, as a cancellation between nodes does.
                Err(ModelError::Cancelled) => {
                    return Ok(ReplayOutcome { completed_nodes, cancelled: true });
                }
                other => other?,
            }
            completed_nodes += 1;
        }
        Ok(ReplayOutcome { completed_nodes, cancelled: false })
    }

    /// Replay `source` into a new [`ValueBatch`].
    ///
    /// # Errors
    ///
    /// As [`Self::replay_into`]; a table larger than the context's hard memory limit
    /// ([`ModelError::ResourceLimit`], before allocating); and [`ModelError::Cancelled`]
    /// when cancelled, since a partial batch is not returned here (use
    /// [`Self::replay_into`] for the exact prefix).
    pub fn replay(
        &mut self,
        view: &ModelView<'_>,
        source: NoiseSource<'_>,
        policy_salt: u64,
        ctx: &ExecutionContext,
    ) -> Result<ValueBatch, ModelError> {
        let table = source.shape_table();
        let (n_rows, n_nodes) = (table.n_rows, table.n_nodes);
        check_budget(ctx, 1, n_rows, n_nodes)?;
        let mut values = vec![0.0; n_rows * n_nodes];
        let outcome = self.replay_into(view, source, policy_salt, &mut values, ctx)?;
        if outcome.cancelled {
            return Err(ModelError::Cancelled);
        }
        Ok(ValueBatch { n_rows, n_nodes, values: values.into() })
    }

    /// Forward-sample `n_rows` draws as a retained table plus its replay: draw every
    /// node's noise ([`NoiseTable::sample`]) and evaluate it with policy salt `0`. Replaying
    /// the returned table with salt `0` reproduces the returned values bit for bit.
    ///
    /// # Errors
    ///
    /// As [`NoiseTable::sample`] and [`Self::replay`]; the memory check covers the table
    /// and the values together.
    pub fn sample(
        &mut self,
        view: &ModelView<'_>,
        n_rows: usize,
        ctx: &ExecutionContext,
    ) -> Result<(ValueBatch, NoiseTable), ModelError> {
        check_budget(ctx, 2, n_rows, view.model.n_nodes())?;
        let table = NoiseTable::sample(view, n_rows, ctx)?;
        let values = self.replay(view, NoiseSource::Table(&table), 0, ctx)?;
        Ok((values, table))
    }
}
