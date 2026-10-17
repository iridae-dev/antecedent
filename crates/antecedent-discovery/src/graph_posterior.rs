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
use antecedent_graph::{Admg, Cpdag, Dag, Pag};
use antecedent_prob::{
    GraphIdentFlag, HessianFactorization, InferenceDiagnostics, WeightedGraphSamples,
    all_chains_moved, parameter_mcmc_diagnostics,
};
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::{AdjacencyMask, MAX_POSTERIOR_NODES, parent_bits};
use crate::constraints::DiscoveryConstraints;
use crate::error::DiscoveryError;
use crate::result::LaggedLink;

/// Hard cap on labeled nodes for exact DAG enumeration (~3.8M DAGs at 6).
///
/// This is a product limit for [`crate::ExactDagPosterior`]. Larger `n` must use
/// MCMC (`OrderMcmc` / `StructureMcmc` / `CiScreenedPosterior`, up to
/// [`MAX_POSTERIOR_NODES`] variables).
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
/// the same [`AdjacencyMask`]s (undirected / bidirected = both directed bits) and
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
/// `from * n_vars + to`; diagonal unused / zero). Adjacency samples are packed
/// directed-edge sets ([`AdjacencyMask`], bit layout [`edge_bit`]) over at most
/// [`MAX_POSTERIOR_NODES`] variables.
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
    ///
    /// The single-word packing of [`Self::atoms`], the earlier representation: populated when
    /// every atom fits one word (a graph on at most eight nodes), empty otherwise. A posterior
    /// over more than eight nodes carries its graphs in [`Self::atoms`] only.
    pub adjacency: Arc<[u64]>,
    /// Packed adjacency sets (length `n_graphs`); see [`edge_bit`]. Authoritative for every
    /// node count up to [`MAX_POSTERIOR_NODES`].
    pub atoms: Arc<[AdjacencyMask]>,
    /// Opaque keys: [`AdjacencyMask::key`] of each adjacency set (the single-word mask itself
    /// for a graph on at most eight nodes). Distinct sets always carry distinct keys.
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
    /// Split R-hat of each entry of [`Self::edge_marginals`]'s indicator across the chains
    /// (same packing). `None` for exact posteriors. A cell whose indicator never varied, or was
    /// not sampled, reports `1`; an indicator the chains disagree on without moving reports `∞`.
    pub edge_rhat: Option<Arc<[f64]>>,
    /// Bulk effective sample size of each entry of [`Self::edge_marginals`]'s indicator across
    /// the chains (same packing). `None` for exact posteriors. A cell whose indicator never
    /// varied, or was not sampled, reports the total retained draw count.
    pub edge_ess: Option<Arc<[f64]>>,
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
    /// circle. Unused for DAG and CPDAG atoms. The single-word packing of
    /// [`Self::mark_atoms`]: empty when an atom needs more than one word.
    pub mark_masks: Option<Arc<[u64]>>,
    /// Optional PAG circle-endpoint sets, same length as [`Self::atoms`]; authoritative for
    /// every node count up to [`MAX_POSTERIOR_NODES`].
    pub mark_atoms: Option<Arc<[AdjacencyMask]>>,
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
    /// Fail-closed constructor over single-word adjacency bitmasks.
    ///
    /// The earlier constructor: [`Self::from_atoms`] with each `u64` read as the packing of a
    /// graph on at most eight nodes. Use [`Self::from_atoms`] for [`AdjacencyMask`]s.
    ///
    /// # Errors
    ///
    /// Those of [`Self::from_atoms`].
    pub fn new(
        n_vars: usize,
        weights: impl Into<Arc<[f64]>>,
        adjacency: impl Into<Arc<[u64]>>,
        edge_marginals: impl Into<Arc<[f64]>>,
        orientation_marginals: impl Into<Arc<[f64]>>,
        sampling: SamplingSummary,
    ) -> Result<Self, DiscoveryError> {
        let adjacency = adjacency.into();
        Self::from_atoms(
            n_vars,
            weights,
            adjacency.iter().copied(),
            edge_marginals,
            orientation_marginals,
            sampling,
        )
    }

    /// Fail-closed constructor.
    ///
    /// Entries may repeat an adjacency mask (one entry per sample). They are
    /// kept as given, so `ess` and per-entry lag masks stay aligned; effect
    /// consumers treat repeated entries as one graph carrying their combined
    /// weight. `atoms` takes [`AdjacencyMask`]s or the single-word `u64` packing of a graph on
    /// at most eight nodes.
    ///
    /// # Errors
    ///
    /// Empty ensemble, length mismatch, non-finite weights, non-positive mass, more than
    /// [`MAX_POSTERIOR_NODES`] variables, or two distinct adjacency sets with one key.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
    )]
    pub fn from_atoms(
        n_vars: usize,
        weights: impl Into<Arc<[f64]>>,
        atoms: impl IntoIterator<Item = impl Into<AdjacencyMask>>,
        edge_marginals: impl Into<Arc<[f64]>>,
        orientation_marginals: impl Into<Arc<[f64]>>,
        sampling: SamplingSummary,
    ) -> Result<Self, DiscoveryError> {
        let SamplingSummary { ess, diagnostics, rejected_invalid } = sampling;
        let weights = weights.into();
        let atoms: Arc<[AdjacencyMask]> = atoms.into_iter().map(Into::into).collect();
        let edge_marginals = edge_marginals.into();
        let orientation_marginals = orientation_marginals.into();
        let n_graphs = weights.len();
        if n_graphs == 0 {
            return Err(DiscoveryError::unsupported("empty graph posterior"));
        }
        if atoms.len() != n_graphs {
            return Err(DiscoveryError::unsupported("adjacency/weights length mismatch"));
        }
        if n_vars > MAX_POSTERIOR_NODES {
            return Err(DiscoveryError::unsupported(
                "graph posterior supports at most 16 variables (MAX_POSTERIOR_NODES)",
            ));
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
        let graph_keys = distinct_graph_keys(&atoms)?;
        let adjacency = single_words(&atoms);
        Ok(Self {
            n_vars,
            n_graphs,
            weights,
            adjacency,
            atoms,
            graph_keys,
            edge_marginals,
            orientation_marginals,
            ess,
            diagnostics,
            rejected_invalid,
            edge_mcse: None,
            edge_rhat: None,
            edge_ess: None,
            lagged_edge_mcse: None,
            lagged_edge_marginals: None,
            max_lag: None,
            lag_masks: None,
            atom_kind: GraphPosteriorAtomKind::Dag,
            mark_masks: None,
            mark_atoms: None,
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

    /// Attach the per-edge split R-hat and bulk ESS across chains (`n_vars²` packing).
    ///
    /// # Errors
    ///
    /// Either length differs from the edge marginals.
    pub fn with_edge_chain_diagnostics(
        mut self,
        rhat: Vec<f64>,
        ess: Vec<f64>,
    ) -> Result<Self, DiscoveryError> {
        if rhat.len() != self.edge_marginals.len() || ess.len() != self.edge_marginals.len() {
            return Err(DiscoveryError::unsupported("edge R-hat/ESS length mismatch"));
        }
        self.edge_rhat = Some(Arc::from(rhat));
        self.edge_ess = Some(Arc::from(ess));
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

    /// Attach PAG circle-endpoint masks (same length as [`Self::adjacency`]), single-word
    /// bitmasks: [`Self::with_mark_atoms`] for a graph on at most eight nodes.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn with_mark_masks(self, masks: impl Into<Arc<[u64]>>) -> Result<Self, DiscoveryError> {
        let masks = masks.into();
        self.with_mark_atoms(masks.iter().copied())
    }

    /// Attach PAG circle-endpoint sets (same length as [`Self::atoms`]).
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn with_mark_atoms(
        mut self,
        marks: impl IntoIterator<Item = impl Into<AdjacencyMask>>,
    ) -> Result<Self, DiscoveryError> {
        let marks: Arc<[AdjacencyMask]> = marks.into_iter().map(Into::into).collect();
        if marks.len() != self.n_graphs {
            return Err(DiscoveryError::unsupported("mark_masks/adjacency length mismatch"));
        }
        self.mark_masks = Some(single_words(&marks));
        self.mark_atoms = Some(marks);
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

/// Refuse a node count whose adjacency does not fit an [`AdjacencyMask`].
///
/// `message` stays the caller's: order MCMC, structure MCMC and the DBN posterior each name
/// their own engine.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] with `message` when `n` exceeds [`MAX_POSTERIOR_NODES`].
pub(crate) fn ensure_nodes_fit(n: usize, message: &'static str) -> Result<(), DiscoveryError> {
    if n > MAX_POSTERIOR_NODES { Err(DiscoveryError::unsupported(message)) } else { Ok(()) }
}

// The single-word functions below are the earlier `u64` API, kept for graphs on at most eight
// nodes (`n * (n - 1) <= 64` edge bits). The same operations over [`AdjacencyMask`], for up to
// `MAX_POSTERIOR_NODES` nodes, live in `crate::adjacency_mask`.

/// Refuse a node count whose directed edges do not fit one `u64` (`n * (n - 1) > 64`).
fn require_word_capacity(n_vars: usize, what: &str) -> Result<(), DiscoveryError> {
    if n_directed_edges(n_vars) > 64 {
        return Err(DiscoveryError::data_msg(format!(
            "{what}: n_vars={n_vars} exceeds u64 adjacency packing"
        )));
    }
    Ok(())
}

/// The single word of a mask built from single-word inputs on at most eight nodes.
fn into_word(mask: AdjacencyMask, what: &str) -> Result<u64, DiscoveryError> {
    mask.to_word().ok_or_else(|| {
        DiscoveryError::data_msg(format!("{what}: the adjacency set does not fit one u64"))
    })
}

/// Whether bit `from → to` is set.
#[must_use]
pub fn has_edge(mask: u64, n: usize, from: usize, to: usize) -> bool {
    crate::adjacency_mask::has_edge(mask, n, from, to)
}

/// Set or clear directed edge `from → to`.
///
/// # Panics
///
/// When the edge's bit lies past bit 63 (a graph on more than eight nodes); use
/// [`crate::adjacency_mask::set_edge`] for those.
#[must_use]
pub fn set_edge(mask: u64, n: usize, from: usize, to: usize, present: bool) -> u64 {
    crate::adjacency_mask::set_edge(mask, n, from, to, present).to_word().expect(
        "edge bit past bit 63: use adjacency_mask::set_edge for graphs on more than eight nodes",
    )
}

/// Parent indices of `node` under `mask`, in increasing parent index.
#[must_use]
pub fn parents_of(mask: u64, n: usize, node: usize) -> Vec<u32> {
    crate::adjacency_mask::parents_of(mask, n, node)
}

/// Whether `mask` encodes a DAG on `n` nodes.
#[must_use]
pub fn mask_is_dag(mask: u64, n: usize) -> bool {
    crate::adjacency_mask::mask_is_dag(mask, n)
}

/// Build a [`Dag`] from a packed adjacency bitmask.
///
/// # Errors
///
/// When `mask` is not a DAG on `n_vars` nodes, or `n_vars` exceeds packing capacity.
pub fn dag_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Dag, DiscoveryError> {
    require_word_capacity(n_vars, "dag_from_adjacency_mask")?;
    crate::adjacency_mask::dag_from_adjacency_mask(mask, n_vars)
}

/// Build a [`Cpdag`] from a packed adjacency bitmask.
///
/// Both directed bits of an unordered pair encode an undirected edge.
///
/// # Errors
///
/// Packing overflow, or an insert the CPDAG rejects.
pub fn cpdag_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Cpdag, DiscoveryError> {
    require_word_capacity(n_vars, "cpdag_from_adjacency_mask")?;
    crate::adjacency_mask::cpdag_from_adjacency_mask(mask, n_vars)
}

/// Pack a [`Cpdag`] into a directed-edge bitmask (undirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_cpdag(cpdag: &Cpdag) -> Result<u64, DiscoveryError> {
    require_word_capacity(cpdag.node_count(), "adjacency_mask_from_cpdag")?;
    into_word(crate::adjacency_mask::adjacency_mask_from_cpdag(cpdag)?, "adjacency_mask_from_cpdag")
}

/// Build a [`Pag`] from packed adjacency and optional circle-endpoint bitmasks.
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
    require_word_capacity(n_vars, "pag_from_adjacency_mask")?;
    crate::adjacency_mask::pag_from_adjacency_mask(mask, mark_mask, n_vars)
}

/// Pack a [`Pag`] into adjacency and circle-endpoint bitmasks.
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_masks_from_pag(pag: &Pag) -> Result<(u64, u64), DiscoveryError> {
    require_word_capacity(pag.node_count(), "adjacency_masks_from_pag")?;
    let (adjacency, marks) = crate::adjacency_mask::adjacency_masks_from_pag(pag)?;
    Ok((
        into_word(adjacency, "adjacency_masks_from_pag")?,
        into_word(marks, "adjacency_masks_from_pag")?,
    ))
}

/// Build an [`Admg`] from a packed adjacency bitmask.
///
/// Both directed bits of an unordered pair encode a bidirected edge; a single directed bit
/// encodes a directed edge. Circle marks are ignored: this is ADMG semantics, not PAG.
///
/// # Errors
///
/// Packing overflow, a directed cycle, or an insert the ADMG rejects.
pub fn admg_from_adjacency_mask(mask: u64, n_vars: usize) -> Result<Admg, DiscoveryError> {
    require_word_capacity(n_vars, "admg_from_adjacency_mask")?;
    crate::adjacency_mask::admg_from_adjacency_mask(mask, n_vars)
}

/// Pack an [`Admg`] into a directed-edge bitmask (bidirected = both bits).
///
/// # Errors
///
/// Packing overflow.
pub fn adjacency_mask_from_admg(admg: &Admg) -> Result<u64, DiscoveryError> {
    require_word_capacity(admg.node_count(), "adjacency_mask_from_admg")?;
    into_word(crate::adjacency_mask::adjacency_mask_from_admg(admg)?, "adjacency_mask_from_admg")
}

/// The single-word packing of every set, or an empty slice when one needs a wider word.
fn single_words(sets: &[AdjacencyMask]) -> Arc<[u64]> {
    sets.iter()
        .map(|set| set.to_word())
        .collect::<Option<Vec<u64>>>()
        .map_or_else(|| Arc::from(Vec::new()), Arc::from)
}

/// One key per adjacency set, in atom order.
///
/// A key is the set itself on at most eight nodes and a 64-bit digest beyond, so two distinct
/// sets could in principle share one. Downstream caches index atoms by key; a shared key would
/// merge two graphs, so it is refused instead.
fn distinct_graph_keys(adjacency: &[AdjacencyMask]) -> Result<Arc<[u64]>, DiscoveryError> {
    let mut owner: std::collections::HashMap<u64, AdjacencyMask> =
        std::collections::HashMap::with_capacity(adjacency.len());
    let mut keys = Vec::with_capacity(adjacency.len());
    for &mask in adjacency {
        let key = mask.key();
        if owner.insert(key, mask).is_some_and(|other| other != mask) {
            return Err(DiscoveryError::unsupported(
                "two distinct adjacency sets share one graph key",
            ));
        }
        keys.push(key);
    }
    Ok(Arc::from(keys))
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
    mask: AdjacencyMask,
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
    forbidden: AdjacencyMask,
    required: AdjacencyMask,
    /// Per node: the parents constraints forbid / require, as node bitsets.
    forbidden_parents: Vec<u16>,
    required_parents: Vec<u16>,
    max_parents: usize,
    /// `(ln p, ln (1 - p))` of the Bernoulli edge prior, when one is set.
    edge_logs: Option<(f64, f64)>,
}

impl GraphPriorTable {
    pub(crate) fn new(n: usize, prior: &GraphPrior, variables: &[VariableId]) -> Self {
        let mut forbidden = AdjacencyMask::EMPTY;
        let mut required = AdjacencyMask::EMPTY;
        let index = crate::constraints::ConstraintIndex::compile(&prior.constraints);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let b = edge_bit(n, i, j);
                if index.is_forbidden(&prior.constraints, static_link(variables, i, j)) {
                    forbidden = forbidden.with_bit(b, true);
                }
                if edge_required(&prior.constraints, variables, i, j) {
                    required = required.with_bit(b, true);
                }
            }
        }
        let node_sets = |mask: AdjacencyMask| -> Vec<u16> {
            (0..n).map(|node| parent_bits(mask, n, node)).collect()
        };
        Self {
            n,
            forbidden_parents: node_sets(forbidden),
            required_parents: node_sets(required),
            forbidden,
            required,
            max_parents: prior.constraints.max_parents.unwrap_or(n.saturating_sub(1)),
            edge_logs: prior.edge_inclusion.map(|p| (p.ln(), (1.0 - p).ln())),
        }
    }

    /// Number of nodes.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Log prior of `node` having exactly the parent set `parents`, `None` when constraints
    /// forbid it (a forbidden parent, a missing required parent, too many parents).
    ///
    /// The sum of this over the nodes is [`Self::log_prior`] of the graph they form (before the
    /// acyclicity test), term for term: a Bernoulli edge prior factorises over parent sets.
    pub(crate) fn node_log_prior(&self, node: usize, parents: u16) -> Option<f64> {
        if parents & self.forbidden_parents[node] != 0
            || self.required_parents[node] & !parents != 0
            || parents.count_ones() as usize > self.max_parents
        {
            return None;
        }
        let Some((lp, lq)) = self.edge_logs else {
            return Some(0.0);
        };
        let candidates = self.n - 1 - self.forbidden_parents[node].count_ones() as usize;
        let present = parents.count_ones() as usize;
        Some(present as f64 * lp + (candidates - present) as f64 * lq)
    }

    /// Every parent set constraints allow for `node`, as node bitsets, or `None` when there are
    /// more than `limit` of them.
    pub(crate) fn allowed_parent_sets(&self, node: usize, limit: usize) -> Option<Vec<u16>> {
        let others: Vec<usize> = (0..self.n).filter(|&p| p != node).collect();
        let mut out = Vec::new();
        for pattern in 0u32..(1u32 << others.len()) {
            if pattern.count_ones() as usize > self.max_parents {
                continue;
            }
            let set = others
                .iter()
                .enumerate()
                .filter(|&(bit, _)| (pattern >> bit) & 1 == 1)
                .fold(0u16, |acc, (_, &p)| acc | (1 << p));
            if self.node_log_prior(node, set).is_none() {
                continue;
            }
            if out.len() == limit {
                return None;
            }
            out.push(set);
        }
        Some(out)
    }

    /// Log prior of `mask`, `None` when it violates a constraint or is cyclic.
    pub(crate) fn log_prior(&self, mask: AdjacencyMask) -> Option<f64> {
        let n = self.n;
        if mask.intersects(self.forbidden) || !mask.contains_all(self.required) {
            return None;
        }
        if self.max_parents < n.saturating_sub(1)
            && (0..n).any(|j| parent_bits(mask, n, j).count_ones() as usize > self.max_parents)
        {
            return None;
        }
        if !crate::adjacency_mask::mask_is_dag(mask, n) {
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
                let b = edge_bit(n, i, j);
                if self.forbidden.bit(b) {
                    continue;
                }
                s += if mask.bit(b) { lp } else { lq };
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
    masks: &[AdjacencyMask],
) -> (Vec<f64>, Vec<f64>) {
    let mut edge = vec![0.0; n * n];
    for (&w, &mask) in weights.iter().zip(masks.iter()) {
        for i in 0..n {
            for j in 0..n {
                if i != j && crate::adjacency_mask::has_edge(mask, n, i, j) {
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
    /// Split R̂ of each input indicator; `1` for one constant over every draw of every chain.
    pub rhat: Vec<f64>,
    /// Bulk ESS of each input indicator; the total draw count for a constant one.
    pub ess_bulk: Vec<f64>,
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
            rhat: vec![f64::INFINITY; n_params],
            ess_bulk: vec![0.0; n_params],
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
            rhat: vec![1.0; n_params],
            ess_bulk: vec![n_total; n_params],
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
    let mut rhat = vec![1.0; n_params];
    let mut ess_bulk = vec![n_total; n_params];
    for (j, d) in diags.iter().enumerate() {
        rhat[keep[j]] = if !rhat_guard && d.rhat.is_finite() { d.rhat } else { f64::INFINITY };
        ess_bulk[keep[j]] = if !ess_guard && d.ess_bulk.is_finite() { d.ess_bulk } else { 0.0 };
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
        rhat,
        ess_bulk,
    }
}

#[cfg(test)]
mod tests {
    use antecedent_graph::{DenseNodeId, Endpoint};

    use super::*;
    use crate::adjacency_mask::{
        adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
        admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask, has_edge,
        mask_is_dag, pag_from_adjacency_mask, parents_of, set_edge,
    };

    #[test]
    fn directed_masks_report_one_inclusion_probability() {
        let n = 2;
        let mask = set_edge(0, n, 0, 1, true);
        let (edge, orient) = accumulate_marginals(n, &[0.25, 0.75], &[AdjacencyMask::EMPTY, mask]);
        assert_eq!(edge, orient);
        assert_eq!(edge[1].to_bits(), 0.75_f64.to_bits());
        assert_eq!(edge[n].to_bits(), 0.0_f64.to_bits());
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
        let mut m = AdjacencyMask::EMPTY;
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

    /// The packing holds 16 nodes (240 edge bits) and refuses a 17th; the message is the
    /// caller's.
    #[test]
    fn adjacency_fits_through_sixteen_nodes() {
        assert!(ensure_nodes_fit(16, "order MCMC supports at most 16 variables").is_ok());
        let err = ensure_nodes_fit(17, "structure MCMC supports at most 16 variables").unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message == "structure MCMC supports at most 16 variables"),
            "{err}"
        );
        assert_eq!(n_directed_edges(16), 240);
        assert_eq!(edge_bit(16, 15, 14), 239);
    }

    /// A stored single-word adjacency decodes to the same edges, DAG verdict and key as before.
    #[test]
    fn single_word_adjacency_decodes_unchanged() {
        let n = 4;
        let legacy: u64 =
            (1 << edge_bit(n, 0, 1)) | (1 << edge_bit(n, 1, 3)) | (1 << edge_bit(n, 3, 2));
        let mask = AdjacencyMask::from(legacy);
        assert!(has_edge(mask, n, 0, 1) && has_edge(mask, n, 1, 3) && has_edge(mask, n, 3, 2));
        assert!(!has_edge(mask, n, 1, 0) && !has_edge(mask, n, 2, 3));
        assert_eq!(parents_of(legacy, n, 3), vec![1]);
        assert!(mask_is_dag(legacy, n));
        assert_eq!(mask.key(), legacy);
        assert_eq!(mask, legacy);
        assert_eq!(set_edge(0, n, 0, 1, true), AdjacencyMask::from(1 << edge_bit(n, 0, 1)));
    }

    /// Edge bits past word 0 (nine or more nodes) round-trip and the DAG test sees them.
    #[test]
    fn sixteen_node_masks_detect_cycles_and_parents() {
        let n = 16;
        let mut chain = AdjacencyMask::EMPTY;
        for i in 0..n - 1 {
            chain = set_edge(chain, n, i, i + 1, true);
        }
        assert!(mask_is_dag(chain, n));
        assert_eq!(chain.to_word(), None);
        assert_eq!(parents_of(chain, n, 15), vec![14]);
        assert!(has_edge(chain, n, 14, 15) && !has_edge(chain, n, 15, 14));
        let cyclic = set_edge(chain, n, 15, 0, true);
        assert!(!mask_is_dag(cyclic, n));
        let dag = dag_from_adjacency_mask(chain, n).unwrap();
        assert_eq!(dag.edges().count(), 15);
        assert!(dag_from_adjacency_mask(cyclic, n).is_err());
        assert!(dag_from_adjacency_mask(chain, 17).is_err());
    }

    #[test]
    fn graph_posterior_refuses_more_than_sixteen_variables() {
        let sampling = SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("test"),
            rejected_invalid: 0,
        };
        let err = GraphPosterior::new(
            17,
            vec![1.0],
            vec![0_u64],
            vec![0.0; 289],
            vec![0.0; 289],
            sampling,
        )
        .unwrap_err();
        assert!(err.to_string().contains("at most 16 variables"), "{err}");
    }

    /// Wide masks key by digest; distinct ones keep distinct keys and repeats share one.
    #[test]
    fn graph_keys_are_distinct_per_adjacency_set() {
        let n = 12;
        let a = set_edge(AdjacencyMask::EMPTY, n, 0, 11, true);
        let b = set_edge(AdjacencyMask::EMPTY, n, 11, 0, true);
        let sampling = SamplingSummary {
            ess: 2.0,
            diagnostics: InferenceDiagnostics::analytic("test"),
            rejected_invalid: 0,
        };
        let post = GraphPosterior::from_atoms(
            n,
            vec![0.25, 0.25, 0.5],
            vec![a, b, a],
            vec![0.0; n * n],
            vec![0.0; n * n],
            sampling,
        )
        .unwrap();
        assert_ne!(post.graph_keys[0], post.graph_keys[1]);
        assert_eq!(post.graph_keys[0], post.graph_keys[2]);
        assert_eq!(post.atoms[0], a);
        assert!(post.adjacency.is_empty(), "a wide atom has no single-word packing");
    }

    /// Per-edge R̂ and ESS follow the summary: constant indicators report a perfect score and
    /// the well-mixed one a finite pair.
    #[test]
    fn chain_summary_reports_per_edge_rhat_and_ess() {
        let (n_chains, n_draws, n_params) = (2usize, 64usize, 2usize);
        let mut traces = vec![0.0; n_chains * n_draws * n_params];
        for c in 0..n_chains {
            for d in 0..n_draws {
                traces[(c * n_draws + d) * n_params + 1] =
                    f64::from(u8::from((d * 7 + c * 3) % 5 < 2));
            }
        }
        let s = graph_chain_summary(&traces, n_chains, n_draws, n_params);
        assert_eq!(s.rhat[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(s.ess_bulk[0].to_bits(), ((n_chains * n_draws) as f64).to_bits());
        assert!(s.rhat[1].is_finite() && s.rhat[1] < 1.2, "rhat={}", s.rhat[1]);
        assert!(s.ess_bulk[1] > 0.0, "ess={}", s.ess_bulk[1]);
        assert!((s.rhat_max - s.rhat[1]).abs() < 1e-12);
        assert!((s.ess_bulk_min - s.ess_bulk[1]).abs() < 1e-12);
    }
}
