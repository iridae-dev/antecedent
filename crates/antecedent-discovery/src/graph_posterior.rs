//! Graph prior / posterior types and Bayesian discovery engine trait.
//!
//! Scoring uses [`antecedent_state::GraphScoreFamily`] (Gaussian BIC) rather than
//! `antecedent-model::MechanismFamily`, keeping discovery above the model crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::algo::is_dag;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Endpoint, Pag};
use antecedent_prob::{
    GraphIdentFlag, HessianFactorization, InferenceDiagnostics, WeightedGraphSamples,
    all_chains_moved, parameter_mcmc_diagnostics,
};
use antecedent_state::GraphScoreFamily;

use crate::constraints::DiscoveryConstraints;
use crate::error::DiscoveryError;
use crate::result::LaggedLink;

/// Hard cap on labeled nodes for exact DAG enumeration (~3.8M DAGs at 6).
///
/// This is a product limit for [`crate::ExactDagPosterior`]. Larger `n` must use
/// MCMC (`OrderMcmc` / `StructureMcmc` / `CiScreenedPosterior`).
pub const EXACT_ENUM_MAX_NODES: usize = 6;

/// Prior over DAG structures for Bayesian discovery.
#[derive(Clone, Debug)]
pub struct GraphPrior {
    /// Discovery constraints (forbidden / required / max-parents / tiers).
    pub constraints: DiscoveryConstraints,
    /// Independent Bernoulli edge-inclusion probability (`None` = uniform over valid DAGs).
    pub edge_inclusion: Option<f64>,
}

impl Default for GraphPrior {
    fn default() -> Self {
        Self::uniform()
    }
}

impl GraphPrior {
    /// Uniform prior over constraint-valid DAGs.
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: Lag::CONTEMPORANEOUS,
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            edge_inclusion: None,
        }
    }

    /// Independent Bernoulli(`p`) edge prior (plus constraints).
    ///
    /// # Errors
    ///
    /// `p` outside `(0, 1)`.
    pub fn bernoulli_edges(p: f64) -> Result<Self, DiscoveryError> {
        if !(p > 0.0 && p < 1.0) {
            return Err(DiscoveryError::unsupported("Bernoulli edge prior requires p in (0, 1)"));
        }
        Ok(Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: Lag::CONTEMPORANEOUS,
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            edge_inclusion: Some(p),
        })
    }

    /// Attach constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }
}

/// Graph class of each [`GraphPosterior`] atom.
///
/// DAG is the default constructor shape. CPDAG/PAG/ADMG atoms are packed into
/// the same adjacency masks (undirected / bidirected = both directed bits) and
/// optional [`GraphPosterior::mark_masks`] for PAG circle marks. ADMG packing
/// is bidirected-edge semantics, not PAG circle marks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphPosteriorAtomKind {
    /// Directed acyclic atoms. The historical constructor default.
    #[default]
    Dag,
    /// Completed partial DAG atoms. Both directed bits encode an undirected edge.
    Cpdag,
    /// Partial ancestral graph atoms. Both directed bits encode a bidirected
    /// edge; [`GraphPosterior::mark_masks`] flags circle endpoints.
    Pag,
    /// Acyclic directed mixed graph atoms. Both directed bits encode a
    /// bidirected (latent-confounder) edge. Circle marks are unused: this is
    /// ADMG semantics, not PAG.
    Admg,
    /// Contemporaneous temporal ADMG atoms. Packing matches [`Self::Admg`]:
    /// both directed bits are a bidirected edge. Each variable is one lag-0
    /// node, so general identification is the static ADMG functional.
    TemporalAdmg,
}

impl GraphPosteriorAtomKind {
    /// Wire / matrix name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dag => "Dag",
            Self::Cpdag => "Cpdag",
            Self::Pag => "Pag",
            Self::Admg => "Admg",
            Self::TemporalAdmg => "TemporalAdmg",
        }
    }
}

/// Columnar graph posterior.
///
/// Edge / orientation marginals are packed length `n_vars * n_vars` (row-major
/// `from * n_vars + to`; diagonal unused / zero). Adjacency samples use packed
/// directed-edge bitmasks ([`edge_bit`]).
///
/// When the search was restricted to a candidate set (for example
/// [`crate::CiScreenedPosterior`], which proposes only pairs its CI screen kept), the
/// marginals are conditional on that restriction: a pair outside it is never sampled, so
/// its marginal reflects the screen, not the data. Such runs name the excluded pairs in
/// [`InferenceDiagnostics::notes`].
#[derive(Clone, Debug)]
pub struct GraphPosterior {
    /// Number of variables.
    pub n_vars: usize,
    /// Number of retained graph atoms (unique DAGs or MCMC samples).
    pub n_graphs: usize,
    /// Normalized posterior weights (length `n_graphs`).
    pub weights: Arc<[f64]>,
    /// Packed adjacency bitmasks (length `n_graphs`); see [`edge_bit`].
    pub adjacency: Arc<[u64]>,
    /// Opaque keys (often the adjacency mask itself).
    pub graph_keys: Arc<[u64]>,
    /// Directed-edge inclusion, length `n_vars * n_vars`.
    ///
    /// For a DAG mask this is `P(i → j)`. It is the same event as
    /// [`Self::orientation_marginals`]: the mask has no separate undirected bit.
    pub edge_marginals: Arc<[f64]>,
    /// `P(i → j)`, same values as [`Self::edge_marginals`].
    ///
    /// Kept as its own vector so a CI-screen floor can be applied without aliasing the
    /// two fields. A PAG atom's circle marks live in [`Self::mark_masks`], not here.
    pub orientation_marginals: Arc<[f64]>,
    /// Kish effective number of graphs, `(Σw)² / Σw²`, of the weight vector.
    ///
    /// A measure of how concentrated the posterior is over graphs (1 = a single graph),
    /// **not** a sampling effective sample size: a sharply identified exact posterior has a
    /// value near 1. Sampling accuracy of MCMC posteriors is in
    /// [`InferenceDiagnostics::ess_bulk_min`] and [`Self::edge_mcse`].
    pub ess: f64,
    /// Chain / analytic diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Graphs rejected as cyclic / constraint-invalid during search.
    pub rejected_invalid: u64,
    /// Monte-Carlo standard error of each entry of [`Self::edge_marginals`] (same packing),
    /// `sqrt(p(1-p) / ESS)` from the chains' bulk ESS. `None` for exact posteriors, whose
    /// marginals carry no sampling error; `0` for an indicator that never varied.
    pub edge_mcse: Option<Arc<[f64]>>,
    /// Monte-Carlo standard error of each entry of [`Self::lagged_edge_marginals`].
    pub lagged_edge_mcse: Option<Arc<[f64]>>,
    /// Optional lagged-edge marginals for DBN templates
    /// (packing `(lag - 1) * n_vars * n_vars + from * n_vars + to`, lag ≥ 1).
    pub lagged_edge_marginals: Option<Arc<[f64]>>,
    /// Max lag encoded in `lagged_edge_marginals` (`None` if static).
    pub max_lag: Option<u32>,
    /// Optional per-atom lag-edge bitmasks (DBN templates; same length as `adjacency`).
    pub lag_masks: Option<Arc<[u64]>>,
    /// Class of every atom. [`GraphPosterior::new`] leaves this as
    /// [`GraphPosteriorAtomKind::Dag`].
    pub atom_kind: GraphPosteriorAtomKind,
    /// Optional PAG circle-endpoint masks, same length as [`Self::adjacency`].
    /// Bit `(from, to)` means the `from` endpoint of that ordered pair is a
    /// circle. Unused for DAG and CPDAG atoms.
    pub mark_masks: Option<Arc<[u64]>>,
    /// Which structure-learning algorithm produced this posterior.
    ///
    /// `None` when the producer did not tag it. Carried so downstream plan records and
    /// artifacts can name the real algorithm rather than a generic label — a posterior
    /// handed to a study no longer arrives with that provenance implied by the call.
    pub algorithm: Option<Arc<str>>,
}

/// How a posterior's atoms were produced: the sampler's effective sample size, its
/// convergence diagnostics and the count of proposals rejected as invalid.
#[derive(Clone, Debug)]
pub struct SamplingSummary {
    /// Effective sample size of the weighted atoms; finite and positive.
    pub ess: f64,
    /// Inference diagnostics of the run that produced the atoms.
    pub diagnostics: InferenceDiagnostics,
    /// Proposals rejected as invalid (cyclic or constraint-violating) during sampling.
    pub rejected_invalid: u64,
}

impl GraphPosterior {
    /// Fail-closed constructor.
    ///
    /// Entries may repeat an adjacency mask (one entry per sample). They are
    /// kept as given, so `ess` and per-entry lag masks stay aligned; effect
    /// consumers treat repeated entries as one graph carrying their combined
    /// weight.
    ///
    /// # Errors
    ///
    /// Empty ensemble, length mismatch, non-finite weights, or non-positive mass.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
    )]
    pub fn new(
        n_vars: usize,
        weights: impl Into<Arc<[f64]>>,
        adjacency: impl Into<Arc<[u64]>>,
        edge_marginals: impl Into<Arc<[f64]>>,
        orientation_marginals: impl Into<Arc<[f64]>>,
        sampling: SamplingSummary,
    ) -> Result<Self, DiscoveryError> {
        let SamplingSummary { ess, diagnostics, rejected_invalid } = sampling;
        let weights = weights.into();
        let adjacency = adjacency.into();
        let edge_marginals = edge_marginals.into();
        let orientation_marginals = orientation_marginals.into();
        let n_graphs = weights.len();
        if n_graphs == 0 {
            return Err(DiscoveryError::unsupported("empty graph posterior"));
        }
        if adjacency.len() != n_graphs {
            return Err(DiscoveryError::unsupported("adjacency/weights length mismatch"));
        }
        let cell = n_vars.saturating_mul(n_vars);
        if edge_marginals.len() != cell || orientation_marginals.len() != cell {
            return Err(DiscoveryError::unsupported("edge/orientation marginal length mismatch"));
        }
        let mut total = 0.0;
        for &w in weights.iter() {
            if !w.is_finite() || w < 0.0 {
                return Err(DiscoveryError::unsupported("non-finite or negative posterior weight"));
            }
            total += w;
        }
        if !(total > 0.0) {
            return Err(DiscoveryError::unsupported("non-positive posterior mass"));
        }
        if !ess.is_finite() || ess <= 0.0 {
            return Err(DiscoveryError::unsupported("invalid ESS"));
        }
        let graph_keys = Arc::from(adjacency.as_ref().to_vec());
        Ok(Self {
            n_vars,
            n_graphs,
            weights,
            adjacency,
            graph_keys,
            edge_marginals,
            orientation_marginals,
            ess,
            diagnostics,
            rejected_invalid,
            edge_mcse: None,
            lagged_edge_mcse: None,
            lagged_edge_marginals: None,
            max_lag: None,
            lag_masks: None,
            atom_kind: GraphPosteriorAtomKind::Dag,
            mark_masks: None,
            algorithm: None,
        })
    }

    /// Attach Monte-Carlo standard errors of the edge marginals (`n_vars²` packing).
    ///
    /// # Errors
    ///
    /// Length differs from the edge marginals.
    pub fn with_edge_mcse(mut self, mcse: Vec<f64>) -> Result<Self, DiscoveryError> {
        if mcse.len() != self.edge_marginals.len() {
            return Err(DiscoveryError::unsupported("edge MCSE length mismatch"));
        }
        self.edge_mcse = Some(Arc::from(mcse));
        Ok(self)
    }

    /// Attach Monte-Carlo standard errors of the lagged-edge marginals.
    ///
    /// # Errors
    ///
    /// Length differs from the lagged marginals, or the posterior has none.
    pub fn with_lagged_edge_mcse(mut self, mcse: Vec<f64>) -> Result<Self, DiscoveryError> {
        match &self.lagged_edge_marginals {
            Some(m) if m.len() == mcse.len() => {
                self.lagged_edge_mcse = Some(Arc::from(mcse));
                Ok(self)
            }
            _ => Err(DiscoveryError::unsupported("lagged edge MCSE length mismatch")),
        }
    }

    /// Declare that every adjacency mask is a CPDAG, PAG, or ADMG atom, not a DAG.
    #[must_use]
    pub fn with_atom_kind(mut self, kind: GraphPosteriorAtomKind) -> Self {
        self.atom_kind = kind;
        self
    }

    /// Attach PAG circle-endpoint masks (same length as [`Self::adjacency`]).
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn with_mark_masks(mut self, masks: impl Into<Arc<[u64]>>) -> Result<Self, DiscoveryError> {
        let masks = masks.into();
        if masks.len() != self.n_graphs {
            return Err(DiscoveryError::unsupported("mark_masks/adjacency length mismatch"));
        }
        self.mark_masks = Some(masks);
        Ok(self)
    }

    /// Tag this posterior with the algorithm that produced it.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: impl Into<Arc<str>>) -> Self {
        self.algorithm = Some(algorithm.into());
        self
    }

    /// Attach lagged-edge marginals (DBN).
    ///
    /// # Errors
    ///
    /// Length mismatch vs `max_lag * n_vars²`.
    pub fn with_lagged_marginals(
        mut self,
        max_lag: u32,
        lagged: impl Into<Arc<[f64]>>,
    ) -> Result<Self, DiscoveryError> {
        let lagged = lagged.into();
        let expect = (max_lag as usize).saturating_mul(self.n_vars.saturating_mul(self.n_vars));
        if lagged.len() != expect {
            return Err(DiscoveryError::unsupported("lagged edge marginal length mismatch"));
        }
        self.lagged_edge_marginals.replace(lagged);
        self.max_lag = Some(max_lag);
        Ok(self)
    }

    /// Attach per-atom lag-edge bitmasks (same length as `adjacency`).
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn with_lag_masks(mut self, masks: impl Into<Arc<[u64]>>) -> Result<Self, DiscoveryError> {
        let masks = masks.into();
        if masks.len() != self.n_graphs {
            return Err(DiscoveryError::unsupported("lag_masks/adjacency length mismatch"));
        }
        self.lag_masks = Some(masks);
        Ok(self)
    }

    /// Convert to envelope ensemble (all graphs marked identified).
    ///
    /// # Errors
    ///
    /// Shape failures from [`WeightedGraphSamples`].
    pub fn to_weighted_samples(&self) -> Result<WeightedGraphSamples, DiscoveryError> {
        let identified = Arc::from(vec![GraphIdentFlag::Identified; self.n_graphs]);
        let mut g = WeightedGraphSamples::new(
            Arc::clone(&self.weights),
            identified,
            Arc::clone(&self.graph_keys),
        )?;
        g.edge_marginals = Some(Arc::clone(&self.edge_marginals));
        g.orientation_marginals = Some(Arc::clone(&self.orientation_marginals));
        Ok(g)
    }
}

/// Bayesian DAG / DBN posterior search engine.
pub trait GraphPosteriorEngine {
    /// Infer a graph posterior from tabular data.
    ///
    /// # Errors
    ///
    /// Data, score, constraint, budget, or convergence failures.
    fn infer_graphs(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError>;
}

/// Bit index for directed edge `from → to` among `n` labeled nodes (`from != to`).
#[must_use]
pub fn edge_bit(n: usize, from: usize, to: usize) -> u32 {
    debug_assert_ne!(from, to);
    debug_assert!(from < n && to < n);
    let idx = from * (n - 1) + if to < from { to } else { to - 1 };
    crate::indexing::dense_u32(idx)
}

/// Number of possible directed edges on `n` labeled nodes.
#[must_use]
pub const fn n_directed_edges(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1))
}

/// Refuse a directed adjacency that does not fit in the `u64` mask (`n(n-1) > 63`).
///
/// `message` stays the caller's: exact enumeration, order MCMC, structure MCMC, and the
/// DBN posterior each name their own engine. The predicate is the one those four guards
/// copied.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] with `message` when the mask would need bit 64 or higher.
pub(crate) fn ensure_directed_mask_fits(
    n: usize,
    message: &'static str,
) -> Result<(), DiscoveryError> {
    if n_directed_edges(n) > 63 { Err(DiscoveryError::unsupported(message)) } else { Ok(()) }
}

/// Whether bit `from → to` is set.
#[must_use]
pub fn has_edge(mask: u64, n: usize, from: usize, to: usize) -> bool {
    if from == to || from >= n || to >= n {
        return false;
    }
    (mask >> edge_bit(n, from, to)) & 1 == 1
}

/// Set or clear directed edge `from → to`.
#[must_use]
pub fn set_edge(mask: u64, n: usize, from: usize, to: usize, present: bool) -> u64 {
    let b = edge_bit(n, from, to);
    if present { mask | (1u64 << b) } else { mask & !(1u64 << b) }
}

/// Parent indices of `node` under `mask`, in increasing parent index.
#[must_use]
pub fn parents_of(mask: u64, n: usize, node: usize) -> Vec<u32> {
    let mut pa = Vec::new();
    fill_parents(mask, n, node, &mut pa);
    pa
}

/// Write parent indices of `node` into `parents`, reusing its allocation.
pub(crate) fn fill_parents(mask: u64, n: usize, node: usize, parents: &mut Vec<u32>) {
    parents.clear();
    for p in 0..n {
        if p != node && has_edge(mask, n, p, node) {
            parents.push(crate::indexing::dense_u32(p));
        }
    }
}

/// Whether `mask` encodes a DAG on `n` nodes.
#[must_use]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub fn mask_is_dag(mask: u64, n: usize) -> bool {
    if n <= MASK_DAG_FAST_NODES {
        return mask_is_dag_bits(mask, n);
    }
    let mut parents = vec![Vec::new(); n];
    let mut children = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && has_edge(mask, n, i, j) {
                let from = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
                let to = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
                children[i].push(to);
                parents[j].push(from);
            }
        }
    }
    is_dag(&parents, &children)
}

/// Largest node count whose parent sets fit one byte each (`8 * 7 = 56` edge bits).
const MASK_DAG_FAST_NODES: usize = 8;

/// Allocation-free Kahn peel over per-node parent bitsets.
fn mask_is_dag_bits(mask: u64, n: usize) -> bool {
    let mut parents = [0u8; MASK_DAG_FAST_NODES];
    for (j, pa) in parents.iter_mut().enumerate().take(n) {
        for i in 0..n {
            if i != j && has_edge(mask, n, i, j) {
                *pa |= 1 << i;
            }
        }
    }
    let mut remaining: u8 = if n == 0 { 0 } else { u8::MAX >> (8 - n) };
    loop {
        let before = remaining;
        for (v, &pa) in parents.iter().enumerate().take(n) {
            if remaining & (1 << v) != 0 && pa & remaining == 0 {
                remaining &= !(1 << v);
            }
        }
        if remaining == 0 {
            return true;
        }
        if remaining == before {
            return false;
        }
    }
}

/// Build a [`Dag`] from a packed adjacency bitmask.
///
/// # Errors
///
/// When `mask` is not a DAG on `n_vars` nodes, or `n_vars` exceeds packing capacity.
pub fn dag_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Dag, DiscoveryError> {
    if n_vars == 0 {
        return Err(DiscoveryError::data_msg("dag_from_adjacency_mask: n_vars must be > 0"));
    }
    if n_directed_edges(n_vars) > 64 {
        return Err(DiscoveryError::data_msg(format!(
            "dag_from_adjacency_mask: n_vars={n_vars} exceeds u64 adjacency packing"
        )));
    }
    if !mask_is_dag(mask, n_vars) {
        return Err(DiscoveryError::data_msg(format!(
            "adjacency mask {mask:#x} is not a DAG on {n_vars} nodes"
        )));
    }
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut dag = Dag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i != j && has_edge(mask, n_vars, i, j) {
                dag.insert_directed(
                    DenseNodeId::from_raw(crate::indexing::dense_u32(i)),
                    DenseNodeId::from_raw(crate::indexing::dense_u32(j)),
                )?;
            }
        }
    }
    Ok(dag)
}

fn require_mask_capacity(n_vars: usize, what: &str) -> Result<(), DiscoveryError> {
    if n_vars == 0 {
        return Err(DiscoveryError::data_msg(format!("{what}: n_vars must be > 0")));
    }
    if n_directed_edges(n_vars) > 64 {
        return Err(DiscoveryError::data_msg(format!(
            "{what}: n_vars={n_vars} exceeds u64 adjacency packing"
        )));
    }
    Ok(())
}

/// Build a [`Cpdag`] from a packed adjacency bitmask.
///
/// Both directed bits of an unordered pair encode an undirected edge.
///
/// # Errors
///
/// Packing overflow, or an insert the CPDAG rejects.
pub fn cpdag_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Cpdag, DiscoveryError> {
    require_mask_capacity(n_vars, "cpdag_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut cpdag = Cpdag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            if ij && ji {
                cpdag.insert_undirected(a, b)?;
            } else if ij {
                cpdag.insert_directed(a, b)?;
            } else if ji {
                cpdag.insert_directed(b, a)?;
            }
        }
    }
    Ok(cpdag)
}

/// Pack a [`Cpdag`] into a directed-edge bitmask (undirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_cpdag(cpdag: &Cpdag) -> Result<u64, DiscoveryError> {
    let n_vars = cpdag.node_count();
    require_mask_capacity(n_vars, "adjacency_mask_from_cpdag")?;
    let mut mask = 0_u64;
    for edge in cpdag.edges() {
        let i = edge.a.as_usize();
        let j = edge.b.as_usize();
        if edge.is_undirected() {
            mask = set_edge(mask, n_vars, i, j, true);
            mask = set_edge(mask, n_vars, j, i, true);
        } else if edge.at_a == Endpoint::Tail && edge.at_b == Endpoint::Arrow {
            mask = set_edge(mask, n_vars, i, j, true);
        } else if edge.at_a == Endpoint::Arrow && edge.at_b == Endpoint::Tail {
            mask = set_edge(mask, n_vars, j, i, true);
        }
    }
    Ok(mask)
}

/// Build a [`Pag`] from packed adjacency and optional circle-endpoint masks.
///
/// Both directed bits encode a bidirected edge when neither end is circled.
/// A circle bit on `(from, to)` marks the `from` endpoint as a circle.
///
/// # Errors
///
/// Packing overflow, or an insert the PAG rejects.
pub fn pag_from_adjacency_mask(
    mask: u64,
    mark_mask: u64,
    n_vars: usize,
) -> Result<Pag, DiscoveryError> {
    require_mask_capacity(n_vars, "pag_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut pag = Pag::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            if !ij && !ji {
                continue;
            }
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            let circle_a = has_edge(mark_mask, n_vars, i, j);
            let circle_b = has_edge(mark_mask, n_vars, j, i);
            if ij && ji {
                if circle_a && circle_b {
                    pag.insert_circle_circle(a, b)?;
                } else if !circle_a && !circle_b {
                    pag.insert_bidirected(a, b)?;
                } else if circle_a {
                    pag.insert_circle_arrow(a, b)?;
                } else {
                    pag.insert_circle_arrow(b, a)?;
                }
            } else if ij {
                if circle_a {
                    pag.insert_circle_arrow(a, b)?;
                } else {
                    pag.insert_directed(a, b)?;
                }
            } else if circle_b {
                pag.insert_circle_arrow(b, a)?;
            } else {
                pag.insert_directed(b, a)?;
            }
        }
    }
    Ok(pag)
}

/// Pack a [`Pag`] into adjacency and circle-endpoint masks.
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_masks_from_pag(pag: &Pag) -> Result<(u64, u64), DiscoveryError> {
    let n_vars = pag.node_count();
    require_mask_capacity(n_vars, "adjacency_masks_from_pag")?;
    let mut adj = 0_u64;
    let mut marks = 0_u64;
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            let Some(edge) = pag.edge_between(a, b) else {
                continue;
            };
            let (at_i, at_j) = if edge.a.as_usize() == i {
                (edge.at_a, edge.at_b)
            } else {
                (edge.at_b, edge.at_a)
            };
            match (at_i, at_j) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    adj = set_edge(adj, n_vars, j, i, true);
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    adj = set_edge(adj, n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Arrow) | (Endpoint::Tail, Endpoint::Circle) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    marks = set_edge(marks, n_vars, i, j, true);
                }
                (Endpoint::Arrow, Endpoint::Circle) | (Endpoint::Circle, Endpoint::Tail) => {
                    adj = set_edge(adj, n_vars, j, i, true);
                    marks = set_edge(marks, n_vars, j, i, true);
                }
                (Endpoint::Circle, Endpoint::Circle) => {
                    adj = set_edge(adj, n_vars, i, j, true);
                    adj = set_edge(adj, n_vars, j, i, true);
                    marks = set_edge(marks, n_vars, i, j, true);
                    marks = set_edge(marks, n_vars, j, i, true);
                }
                (Endpoint::Tail, Endpoint::Tail) => {
                    return Err(DiscoveryError::unsupported(
                        "PAG packing cannot represent a tail-tail edge; refusing to drop it",
                    ));
                }
                (Endpoint::Conflict, _) | (_, Endpoint::Conflict) => {
                    return Err(DiscoveryError::unsupported(
                        "PAG packing cannot represent a conflict endpoint; refusing to drop it",
                    ));
                }
            }
        }
    }
    Ok((adj, marks))
}

/// Build an [`Admg`] from a packed adjacency bitmask.
///
/// Both directed bits of an unordered pair encode a bidirected edge. A single
/// directed bit encodes a directed edge. Circle marks are ignored: this is
/// ADMG semantics, not PAG.
///
/// A pair that is both directed and bidirected cannot be packed into one
/// bitmask (same limitation as PAG packing). Front-door `T→M→Y` with `T↔Y`
/// packs cleanly because the bidirected pair is not also directed.
///
/// # Errors
///
/// Packing overflow, a directed cycle, or an insert the ADMG rejects.
pub fn admg_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Admg, DiscoveryError> {
    require_mask_capacity(n_vars, "admg_from_adjacency_mask")?;
    let n_u32 = u32::try_from(n_vars)
        .map_err(|_| DiscoveryError::data_msg("n_vars too large for DenseNodeId"))?;
    let mut admg = Admg::with_variables(n_u32);
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let ij = has_edge(mask, n_vars, i, j);
            let ji = has_edge(mask, n_vars, j, i);
            if !ij && !ji {
                continue;
            }
            let a = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            let b = DenseNodeId::from_raw(crate::indexing::dense_u32(j));
            if ij && ji {
                admg.insert_bidirected(a, b)?;
            } else if ij {
                admg.insert_directed(a, b)?;
            } else {
                admg.insert_directed(b, a)?;
            }
        }
    }
    Ok(admg)
}

/// Pack an [`Admg`] into a directed-edge bitmask (bidirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_admg(admg: &Admg) -> Result<u64, DiscoveryError> {
    let n_vars = admg.node_count();
    require_mask_capacity(n_vars, "adjacency_mask_from_admg")?;
    let mut mask = 0_u64;
    for i in 0..n_vars {
        let from = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
        for &to in admg.children(from) {
            mask = set_edge(mask, n_vars, i, to.as_usize(), true);
        }
        for &nbr in admg.bidirected_neighbors(from) {
            let j = nbr.as_usize();
            if i < j {
                mask = set_edge(mask, n_vars, i, j, true);
                mask = set_edge(mask, n_vars, j, i, true);
            }
        }
    }
    Ok(mask)
}

/// Contemporaneous `LaggedLink` for dense indices into `variables`.
#[must_use]
pub(crate) fn static_link(variables: &[VariableId], from: usize, to: usize) -> LaggedLink {
    LaggedLink {
        source: variables[from],
        source_lag: Lag::CONTEMPORANEOUS,
        target: variables[to],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

/// Whether constraints forbid directed edge `from → to`.
#[must_use]
#[allow(
    dead_code,
    reason = "single-edge predicate; mask scans use ConstraintIndex, which matches this boolean"
)]
pub(crate) fn edge_forbidden(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    to: usize,
) -> bool {
    let link = static_link(variables, from, to);
    // `is_forbidden` already applies `tier_forbids`.
    constraints.is_forbidden(link)
}

/// Lagged link `from` at `lag` → `to` contemporaneous.
#[must_use]
pub fn lagged_link(variables: &[VariableId], from: usize, lag: u32, to: usize) -> LaggedLink {
    LaggedLink {
        source: variables[from],
        source_lag: Lag::from_raw(lag),
        target: variables[to],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

/// Whether constraints forbid the lagged edge `from_{t−lag} → to_t`.
///
/// [`edge_forbidden`] pins both endpoints to [`Lag::CONTEMPORANEOUS`] via [`static_link`],
/// so it structurally cannot express a constraint on a lag > 0 edge. `DiscoveryConstraints`
/// stores `LaggedLink`s precisely so lag-specific constraints can be written, and callers
/// that score lagged structure need this variant to honor them.
#[must_use]
#[allow(
    dead_code,
    reason = "single-edge predicate; lag-mask scans use ConstraintIndex, which matches this boolean"
)]
pub(crate) fn lagged_edge_forbidden(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    lag: u32,
    to: usize,
) -> bool {
    let link = lagged_link(variables, from, lag, to);
    // `is_forbidden` already applies `tier_forbids`.
    constraints.is_forbidden(link)
}

/// Whether constraints require the lagged edge `from_{t−lag} → to_t`.
#[must_use]
pub(crate) fn lagged_edge_required(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    lag: u32,
    to: usize,
) -> bool {
    constraints.is_required(lagged_link(variables, from, lag, to))
}

/// Whether constraints require directed edge `from → to`.
#[must_use]
pub(crate) fn edge_required(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    to: usize,
) -> bool {
    constraints.is_required(static_link(variables, from, to))
}

/// Log prior for a constraint-valid DAG mask (`−∞` encoded as `None` if invalid).
#[must_use]
pub(crate) fn log_prior_mask(
    mask: u64,
    n: usize,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Option<f64> {
    GraphPriorTable::new(n, prior, variables).log_prior(mask)
}

/// Constraint and edge-prior facts of one `(n, prior, variables)` triple, resolved once so that
/// scoring many masks does not re-derive them per mask.
pub(crate) struct GraphPriorTable {
    n: usize,
    /// `bit[i * n + j]` is the packed bit of `i → j` (`0` on the diagonal).
    bit: Vec<u64>,
    forbidden: u64,
    required: u64,
    max_parents: usize,
    /// `(ln p, ln (1 - p))` of the Bernoulli edge prior, when one is set.
    edge_logs: Option<(f64, f64)>,
}

impl GraphPriorTable {
    pub(crate) fn new(n: usize, prior: &GraphPrior, variables: &[VariableId]) -> Self {
        let mut bit = vec![0u64; n * n];
        let mut forbidden = 0u64;
        let mut required = 0u64;
        let index = crate::constraints::ConstraintIndex::compile(&prior.constraints);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let b = 1u64.checked_shl(edge_bit(n, i, j)).unwrap_or(0);
                bit[i * n + j] = b;
                if index.is_forbidden(&prior.constraints, static_link(variables, i, j)) {
                    forbidden |= b;
                }
                if edge_required(&prior.constraints, variables, i, j) {
                    required |= b;
                }
            }
        }
        Self {
            n,
            bit,
            forbidden,
            required,
            max_parents: prior.constraints.max_parents.unwrap_or(n.saturating_sub(1)),
            edge_logs: prior.edge_inclusion.map(|p| (p.ln(), (1.0 - p).ln())),
        }
    }

    /// Packed bit of `from → to`.
    #[inline]
    pub(crate) fn bit(&self, from: usize, to: usize) -> u64 {
        self.bit[from * self.n + to]
    }

    /// Log prior of `mask`, `None` when it violates a constraint or is cyclic.
    pub(crate) fn log_prior(&self, mask: u64) -> Option<f64> {
        let n = self.n;
        if mask & self.forbidden != 0 || self.required & !mask != 0 {
            return None;
        }
        if self.max_parents < n.saturating_sub(1) {
            for j in 0..n {
                let in_degree = (0..n).filter(|&i| mask & self.bit(i, j) != 0).count();
                if in_degree > self.max_parents {
                    return None;
                }
            }
        }
        if !mask_is_dag(mask, n) {
            return None;
        }
        let Some((lp, lq)) = self.edge_logs else {
            return Some(0.0);
        };
        let mut s = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let b = self.bit(i, j);
                if self.forbidden & b != 0 {
                    continue;
                }
                s += if mask & b != 0 { lp } else { lq };
            }
        }
        Some(s)
    }
}

pub use antecedent_prob::kish_ess;

/// Normalize log-weights with log-sum-exp; returns normalized weights. A `-inf` log-weight is
/// a zero-mass graph; `NaN` or `+inf` is refused rather than silently dropped.
///
/// # Errors
///
/// Empty, invalid (`NaN`/`+inf`) or all `-inf` log-weights.
pub fn normalize_log_weights(log_w: &[f64]) -> Result<Vec<f64>, DiscoveryError> {
    antecedent_kernels::normalize_log_weights(log_w).map_err(|why| match why {
        antecedent_kernels::LogWeightError::Empty => {
            DiscoveryError::unsupported("no valid graphs to normalize")
        }
        antecedent_kernels::LogWeightError::Invalid => {
            DiscoveryError::unsupported("graph log-weights contain NaN or +infinity")
        }
        antecedent_kernels::LogWeightError::NoMass => {
            DiscoveryError::unsupported("no finite graph log-weights")
        }
    })
}

/// Directed-edge inclusion from weighted adjacency masks.
///
/// Both vectors are that probability. DAG masks do not record a separate orientation,
/// so splitting "skeleton" from "direction" here would change the serialized marginals.
/// The second vector is a clone so a later floor can edit one field without aliasing.
#[must_use]
pub(crate) fn accumulate_marginals(
    n: usize,
    weights: &[f64],
    masks: &[u64],
) -> (Vec<f64>, Vec<f64>) {
    let mut edge = vec![0.0; n * n];
    for (&w, &mask) in weights.iter().zip(masks.iter()) {
        for i in 0..n {
            for j in 0..n {
                if i != j && has_edge(mask, n, i, j) {
                    edge[i * n + j] += w;
                }
            }
        }
    }
    let orient = edge.clone();
    (edge, orient)
}

/// Analytic diagnostics for exact / closed-form graph posteriors.
#[must_use]
pub(crate) fn analytic_graph_diagnostics(n_graphs: usize, ess: f64) -> InferenceDiagnostics {
    InferenceDiagnostics {
        converged: true,
        iterations: crate::indexing::dense_u32(n_graphs),
        grad_inf_norm: 0.0,
        hessian_condition: 1.0,
        factorization: HessianFactorization::Analytic,
        separation_warning: false,
        notes: vec![Arc::from(format!(
            "exact_enumeration_of_bic_weights effective_graph_count={ess:.4}"
        ))],
        backend_id: Arc::from("graph_posterior_analytic"),
        n_chains: None,
        n_warmup: None,
        // Exact enumeration has no sampling ESS; the Kish graph count is a concentration index
        // (see the note above), not a chain diagnostic.
        ess_bulk_min: None,
        ess_tail_min: None,
        rhat_max: None,
        n_divergences: None,
        mean_accept_prob: None,
        n_warmup_divergences: None,
        n_postwarmup_divergences: None,
        max_abs_delta_h: None,
        all_chains_moved: None,
    }
}

/// Summary of a graph-structure MCMC run (edge-indicator traces) fed to
/// [`mcmc_graph_diagnostics`].
#[derive(Debug, Clone, Copy)]
pub struct GraphMcmcRun {
    /// Chains run.
    pub n_chains: u32,
    /// Warmup iterations per chain.
    pub n_warmup: u32,
    /// Retained draws per chain.
    pub n_draws: u32,
    /// Smallest bulk effective sample size over the indicators.
    pub ess_bulk_min: f64,
    /// Smallest tail effective sample size over the indicators.
    pub ess_tail_min: f64,
    /// Largest split R-hat over the indicators.
    pub rhat_max: f64,
    /// Divergent transitions.
    pub n_divergences: u32,
    /// Whether the run passed its convergence checks.
    pub converged: bool,
    /// Whether every chain made at least one structural move.
    pub chains_moved: bool,
}

/// MCMC diagnostics gate for graph chains (edge-indicator traces).
#[must_use]
pub fn mcmc_graph_diagnostics(run: &GraphMcmcRun) -> InferenceDiagnostics {
    let GraphMcmcRun {
        n_chains,
        n_warmup,
        n_draws,
        ess_bulk_min,
        ess_tail_min,
        rhat_max,
        n_divergences,
        converged,
        chains_moved,
    } = *run;
    InferenceDiagnostics {
        converged,
        iterations: n_draws,
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: Vec::new(),
        backend_id: Arc::from("graph_structure_mcmc"),
        n_chains: Some(n_chains),
        n_warmup: Some(n_warmup),
        ess_bulk_min: Some(ess_bulk_min),
        ess_tail_min: Some(ess_tail_min),
        rhat_max: Some(rhat_max),
        n_divergences: Some(n_divergences),
        // Not fabricated: this function has no accept/reject counts to derive a real mean
        // Metropolis acceptance probability from. The `rejected` counter threaded through
        // `FinishMaskPosterior`/`GraphPosterior.rejected_invalid` (see `structure_mcmc.rs`,
        // `order_mcmc.rs`) counts constraint-invalid proposals (e.g. would-create-a-cycle),
        // not the sampler's actual per-step Metropolis accept/reject decision -- and it
        // isn't even threaded into this function's signature. `allows_graph_posterior`
        // below does not consult `mean_accept_prob`, so this is safe for the publication
        // gate; callers reading `diagnostics.mean_accept_prob` for reporting get an honest
        // "unknown" instead of a fabricated constant.
        mean_accept_prob: None,
        n_warmup_divergences: Some(0),
        n_postwarmup_divergences: Some(n_divergences),
        max_abs_delta_h: Some(0.0),
        all_chains_moved: Some(chains_moved),
    }
}

/// Largest Monte-Carlo standard error any edge marginal may carry for a graph-MCMC posterior
/// to publish: `0.05` is the standard error of a probability-½ indicator at ESS 100, the
/// library's per-chain-scale bar. It binds where it matters (the marginals users read) rather
/// than on the ESS of near-constant indicators, whose error is small at any ESS.
pub(crate) const MAX_EDGE_MCSE: f64 = 0.05;

/// Whether graph-MCMC diagnostics are sufficient to publish a posterior.
///
/// Binary edge-indicator traces mix slower than continuous HMC parameters, so
/// the R-hat bar is `1.2` (vs `1.01` on [`InferenceDiagnostics::allows_posterior`]).
/// Divergences must be zero (not merely reported). Marginal accuracy is enforced separately
/// on the per-edge MCSE ([`MAX_EDGE_MCSE`]) by the graph-MCMC engines.
#[must_use]
pub(crate) fn allows_graph_posterior(diagnostics: &InferenceDiagnostics) -> bool {
    if diagnostics.factorization != HessianFactorization::Mcmc {
        return diagnostics.allows_posterior();
    }
    let rhat_ok = diagnostics.rhat_max.is_some_and(|r| r.is_finite() && r < 1.2);
    let ess_ok = diagnostics.ess_bulk_min.is_some_and(|e| e.is_finite() && e > 10.0);
    let div_ok =
        diagnostics.n_postwarmup_divergences.or(diagnostics.n_divergences).is_some_and(|n| n == 0);
    // A chain that never accepted a structural move has bit-identical edge traces, which
    // the chain diagnostics report as R̂ = 1.0 and ESS = `n_chains·n_draws` — perfect
    // scores that mean "nothing varied", not "everything converged". `all_chains_moved` is
    // the only signal that separates the two, so it has to be in the conjunction (as it is
    // in `InferenceDiagnostics::mcmc_publication_ok`).
    let moved_ok = diagnostics.all_chains_moved == Some(true);
    diagnostics.converged && rhat_ok && ess_ok && div_ok && moved_ok
}

/// Set `converged` from [`allows_graph_posterior`]; optionally refuse publication.
///
/// # Errors
///
/// When `require_gate` is true and the graph-MCMC diagnostics bar fails.
pub(crate) fn publish_graph_posterior(
    mut diagnostics: InferenceDiagnostics,
    require_gate: bool,
    refuse_msg: &'static str,
) -> Result<InferenceDiagnostics, DiscoveryError> {
    diagnostics.converged = true;
    let ok = allows_graph_posterior(&diagnostics);
    diagnostics.converged = ok;
    if require_gate && !ok {
        return Err(DiscoveryError::unsupported(refuse_msg));
    }
    Ok(diagnostics)
}

/// Edge-indicator chain diagnostics.
#[derive(Clone, Debug)]
pub(crate) struct GraphChainSummary {
    /// Maximum split R̂ over varying indicators. `∞` when any indicator's chains disagree
    /// without varying within a chain (the clearest non-convergence there is); never
    /// replaced by a finite value from another indicator.
    pub rhat_max: f64,
    /// Minimum bulk ESS over varying indicators.
    pub ess_bulk_min: f64,
    /// Minimum tail ESS over varying indicators.
    pub ess_tail_min: f64,
    /// Whether every chain moved on at least one indicator.
    pub all_chains_moved: bool,
    /// Monte-Carlo standard error of each input indicator's mean (`sqrt(p(1-p)/ESS_bulk)`),
    /// `0` for an indicator constant over every draw of every chain.
    pub mcse: Vec<f64>,
}

/// Edge-indicator chain diagnostics (R-hat / ESS / MCSE), dropping constant parameters.
///
/// Zero-variance indicators (never/always present) would otherwise inflate R-hat
/// to infinity and fail the diagnostics gate spuriously. Indicators that are constant
/// within every chain but differ between chains are *not* dropped: their R̂ is `∞`.
#[must_use]
pub(crate) fn graph_chain_summary(
    traces: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> GraphChainSummary {
    if n_params == 0 || n_chains == 0 || n_draws == 0 {
        return GraphChainSummary {
            rhat_max: f64::INFINITY,
            ess_bulk_min: 0.0,
            ess_tail_min: 0.0,
            all_chains_moved: false,
            mcse: vec![0.0; n_params],
        };
    }
    let mut keep = Vec::new();
    for p in 0..n_params {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for c in 0..n_chains {
            for d in 0..n_draws {
                let v = traces[(c * n_draws + d) * n_params + p];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        if max_v - min_v > 1e-12 {
            keep.push(p);
        }
    }
    if keep.is_empty() {
        // All indicators constant across chains — treat as perfect agreement.
        let n_total = (n_chains * n_draws) as f64;
        return GraphChainSummary {
            rhat_max: 1.0,
            ess_bulk_min: n_total,
            ess_tail_min: n_total,
            all_chains_moved: all_chains_moved(traces, n_chains, n_draws, n_params),
            mcse: vec![0.0; n_params],
        };
    }
    let k = keep.len();
    let mut filtered = vec![0.0; n_chains * n_draws * k];
    for c in 0..n_chains {
        for d in 0..n_draws {
            for (j, &p) in keep.iter().enumerate() {
                filtered[(c * n_draws + d) * k + j] = traces[(c * n_draws + d) * n_params + p];
            }
        }
    }
    // Same guards as `mcmc_summary`: too few chains / draws cannot support the statistic.
    let rhat_guard = n_chains < 2 || n_draws < 8;
    let ess_guard = n_draws < 8;
    let diags = parameter_mcmc_diagnostics(&filtered, n_chains, n_draws, k);
    let mut rhat_max = 0.0_f64;
    let mut ess_bulk_min = f64::INFINITY;
    let mut ess_tail_min = f64::INFINITY;
    let mut mcse = vec![0.0; n_params];
    let n_total = (n_chains * n_draws) as f64;
    for (j, d) in diags.iter().enumerate() {
        rhat_max = if d.rhat.is_finite() { rhat_max.max(d.rhat) } else { f64::INFINITY };
        ess_bulk_min = ess_bulk_min.min(d.ess_bulk);
        ess_tail_min = ess_tail_min.min(d.ess_tail);
        let mean: f64 = (0..n_chains * n_draws).map(|i| filtered[i * k + j]).sum::<f64>() / n_total;
        mcse[keep[j]] = if !ess_guard && d.ess_bulk.is_finite() && d.ess_bulk > 0.0 {
            (mean * (1.0 - mean) / d.ess_bulk).max(0.0).sqrt()
        } else {
            f64::INFINITY
        };
    }
    GraphChainSummary {
        rhat_max: if rhat_guard { f64::INFINITY } else { rhat_max },
        ess_bulk_min: if ess_guard || !ess_bulk_min.is_finite() { 0.0 } else { ess_bulk_min },
        ess_tail_min: if ess_guard || !ess_tail_min.is_finite() { 0.0 } else { ess_tail_min },
        all_chains_moved: all_chains_moved(&filtered, n_chains, n_draws, k),
        mcse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_masks_report_one_inclusion_probability() {
        let n = 2;
        let mask = set_edge(0, n, 0, 1, true);
        let (edge, orient) = accumulate_marginals(n, &[0.25, 0.75], &[0, mask]);
        assert_eq!(edge, orient);
        assert_eq!(edge[1], 0.75);
        assert_eq!(edge[n], 0.0);
    }

    /// A chain that never moved must not publish, however good its other diagnostics look.
    ///
    /// When every edge indicator is constant across every chain and draw,
    /// the chain diagnostics report R̂ = 1.0 and ESS = `n_chains·n_draws` — flawless
    /// numbers that mean "nothing varied", not "everything converged". `all_chains_moved` is
    /// the only field that separates the two cases, so the gate has to consult it.
    #[test]
    fn stuck_chain_is_refused_despite_perfect_rhat_and_ess() {
        let stuck = InferenceDiagnostics {
            converged: true,
            factorization: HessianFactorization::Mcmc,
            rhat_max: Some(1.0),
            ess_bulk_min: Some(8_000.0),
            ess_tail_min: Some(8_000.0),
            n_postwarmup_divergences: Some(0),
            all_chains_moved: Some(false),
            ..InferenceDiagnostics::analytic("test")
        };
        assert!(
            !allows_graph_posterior(&stuck),
            "a sampler that never accepted a move must not be published"
        );

        let moved = InferenceDiagnostics { all_chains_moved: Some(true), ..stuck.clone() };
        assert!(allows_graph_posterior(&moved), "an otherwise-clean moving chain must publish");

        // Absent (None) is not evidence of movement either.
        let unknown = InferenceDiagnostics { all_chains_moved: None, ..stuck };
        assert!(!allows_graph_posterior(&unknown));
    }

    /// `mcmc_graph_diagnostics` has no accept/reject counts available to it, so it must not
    /// invent a Metropolis acceptance rate. A caller reading `mean_accept_prob` for
    /// reporting/debugging should see an honest "unknown" (`None`), not a fabricated
    /// `Some(1.0)` that would misreport every real chain (which rejects some proposals) as
    /// having accepted every move.
    #[test]
    fn mcmc_graph_diagnostics_does_not_fabricate_accept_prob() {
        let d = mcmc_graph_diagnostics(&GraphMcmcRun {
            n_chains: 4,
            n_warmup: 100,
            n_draws: 500,
            ess_bulk_min: 200.0,
            ess_tail_min: 150.0,
            rhat_max: 1.01,
            n_divergences: 0,
            converged: true,
            chains_moved: true,
        });
        assert_eq!(
            d.mean_accept_prob, None,
            "mean_accept_prob must be None, not a fabricated constant: {:?}",
            d.mean_accept_prob
        );
    }

    #[test]
    fn edge_bit_roundtrip_unique() {
        let n = 4;
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let b = edge_bit(n, i, j);
                assert!(seen.insert(b));
                assert!(b < n_directed_edges(n) as u32);
            }
        }
        assert_eq!(seen.len(), n_directed_edges(n));
    }

    #[test]
    fn cycle_not_dag() {
        let n = 3;
        let mut m = 0u64;
        m = set_edge(m, n, 0, 1, true);
        m = set_edge(m, n, 1, 2, true);
        m = set_edge(m, n, 2, 0, true);
        assert!(!mask_is_dag(m, n));
    }

    #[test]
    fn cpdag_mask_roundtrips_undirected_and_directed() {
        let mut g = Cpdag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let mask = adjacency_mask_from_cpdag(&g).unwrap();
        assert!(!mask_is_dag(mask, 3));
        let back = cpdag_from_adjacency_mask(mask, 3).unwrap();
        assert_eq!(back.edges().len(), g.edges().len());
        assert!(
            back.edge_between(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))
                .is_some_and(antecedent_graph::MarkedEdge::is_undirected)
        );
    }

    #[test]
    fn pag_mask_roundtrips_directed_and_bidirected() {
        let mut g = Pag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let (adj, marks) = adjacency_masks_from_pag(&g).unwrap();
        let back = pag_from_adjacency_mask(adj, marks, 3).unwrap();
        assert!(
            back.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                .is_some_and(antecedent_graph::MarkedEdge::is_dag_directed)
        );
        assert!(
            back.edge_between(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))
                .is_some_and(antecedent_graph::MarkedEdge::is_bidirected)
        );
    }

    /// A tail–tail edge is a real adjacency. Packing used to drop it and return `Ok`,
    /// so the mask round-trip deleted the edge.
    #[test]
    fn pag_mask_refuses_a_tail_tail_edge() {
        let mut g = Pag::with_variables(2);
        g.insert_marked(antecedent_graph::MarkedEdge::undirected(
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(1),
        ))
        .unwrap();
        let err = adjacency_masks_from_pag(&g).unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message.contains("tail-tail")),
            "{err}"
        );
    }

    #[test]
    fn pag_mask_refuses_a_conflict_endpoint() {
        let mut g = Pag::with_variables(2);
        g.insert_marked(antecedent_graph::MarkedEdge {
            a: DenseNodeId::from_raw(0),
            b: DenseNodeId::from_raw(1),
            at_a: Endpoint::Conflict,
            at_b: Endpoint::Arrow,
            middle: antecedent_graph::MiddleMark::Empty,
        })
        .unwrap();
        let err = adjacency_masks_from_pag(&g).unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message.contains("conflict")),
            "{err}"
        );
    }

    #[test]
    fn admg_mask_roundtrips_directed_and_bidirected() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let mask = adjacency_mask_from_admg(&g).unwrap();
        assert!(!mask_is_dag(mask, 3), "bidirected both-bits must not look like a DAG");
        assert!(dag_from_adjacency_mask(mask, 3).is_err(), "ADMG bidirected packing is not a Dag");
        let back = admg_from_adjacency_mask(mask, 3).unwrap();
        assert!(back.has_bidirected());
        assert!(back.children(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1)));
        assert!(back.children(DenseNodeId::from_raw(1)).contains(&DenseNodeId::from_raw(2)));
        assert!(
            back.bidirected_neighbors(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(2))
        );
    }

    /// Two chains, each constant but at a different value, are the clearest
    /// non-convergence there is: R̂ must stay infinite instead of being replaced by the largest
    /// finite R̂ of some other indicator.
    #[test]
    fn disagreeing_constant_chains_keep_infinite_rhat() {
        let (n_chains, n_draws, n_params) = (2usize, 40usize, 2usize);
        let mut traces = vec![0.0; n_chains * n_draws * n_params];
        for c in 0..n_chains {
            for d in 0..n_draws {
                // Param 0: chain 0 stuck at 0, chain 1 stuck at 1.
                traces[(c * n_draws + d) * n_params] = c as f64;
                // Param 1: a well-mixed alternating indicator in both chains.
                traces[(c * n_draws + d) * n_params + 1] = ((d + c) % 2) as f64;
            }
        }
        let s = graph_chain_summary(&traces, n_chains, n_draws, n_params);
        assert!(s.rhat_max.is_infinite(), "rhat_max={}", s.rhat_max);
    }

    /// The MCSE of an indicator is `sqrt(p(1-p)/ESS)`; a constant indicator has none.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "an indicator that never varies has an MCSE of exactly zero by construction"
    )]
    fn mcse_is_zero_for_constant_and_positive_for_varying_indicators() {
        let (n_chains, n_draws, n_params) = (2usize, 64usize, 2usize);
        let mut traces = vec![0.0; n_chains * n_draws * n_params];
        for c in 0..n_chains {
            for d in 0..n_draws {
                traces[(c * n_draws + d) * n_params] = 1.0;
                traces[(c * n_draws + d) * n_params + 1] =
                    f64::from(u8::from((d * 7 + c * 3) % 5 < 2));
            }
        }
        let s = graph_chain_summary(&traces, n_chains, n_draws, n_params);
        assert_eq!(s.mcse[0], 0.0);
        assert!(s.mcse[1] > 0.0 && s.mcse[1] < 0.5, "mcse={}", s.mcse[1]);
    }

    #[test]
    fn exact_posterior_reports_no_sampling_ess() {
        let d = analytic_graph_diagnostics(25, 1.0);
        assert_eq!(d.ess_bulk_min, None);
    }

    /// `n = 8` has 56 directed edges and fits; `n = 9` has 72 and is the first refusal.
    /// The message is the caller's, unchanged from the four copied guards.
    #[test]
    fn directed_mask_fits_through_eight_nodes() {
        assert!(
            ensure_directed_mask_fits(8, "order MCMC adjacency exceeds 63 directed edges").is_ok()
        );
        let err =
            ensure_directed_mask_fits(9, "structure MCMC adjacency exceeds 63 directed edges")
                .unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message == "structure MCMC adjacency exceeds 63 directed edges"),
            "{err}"
        );
        assert_eq!(n_directed_edges(8), 56);
        assert_eq!(n_directed_edges(9), 72);
    }
}
