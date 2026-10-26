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
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag};
use antecedent_prob::{
    GraphIdentFlag, HessianFactorization, InferenceDiagnostics, WeightedGraphSamples,
    all_chains_moved, parameter_mcmc_diagnostics,
};
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::{AdjacencyMask, LagMask, edge_bits_fit, parent_bits};
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
/// Edge marginals are packed length `n_vars * n_vars` (row-major `from * n_vars + to`; diagonal
/// unused / zero). Adjacency samples are packed directed-edge sets ([`AdjacencyMask`], bit
/// layout [`edge_bit`]) over any number of variables whose edge bits a `u32` addresses.
///
/// The fields are private: [`Self::from_atoms`] validates that every per-atom and per-cell
/// column agrees in length with the atoms and the variable count, and the builder methods keep
/// that true, so a posterior that exists is consistent. The atoms are the one canonical
/// description of the graphs; the keys, the single-word packings and the orientation
/// marginals are derived from them or from the edge marginals.
///
/// When the search was restricted to a candidate set (for example
/// [`crate::CiScreenedPosterior`], which proposes only pairs its CI screen kept), the
/// marginals are conditional on that restriction: a pair outside it is never sampled, so
/// its marginal reflects the screen, not the data. Such runs name the excluded pairs in
/// [`InferenceDiagnostics::notes`].
#[derive(Clone, Debug)]
pub struct GraphPosterior {
    n_vars: usize,
    weights: Arc<[f64]>,
    atoms: Arc<[AdjacencyMask]>,
    graph_keys: Arc<[u64]>,
    edge_marginals: Arc<[f64]>,
    ess: f64,
    diagnostics: InferenceDiagnostics,
    rejected_invalid: u64,
    edge_mcse: Option<Arc<[f64]>>,
    edge_rhat: Option<Arc<[f64]>>,
    edge_ess: Option<Arc<[f64]>>,
    lagged_edge_mcse: Option<Arc<[f64]>>,
    lagged_edge_marginals: Option<Arc<[f64]>>,
    max_lag: Option<u32>,
    lag_masks: Option<Arc<[LagMask]>>,
    lag_bidirected_masks: Option<Arc<[LagMask]>>,
    atom_kind: GraphPosteriorAtomKind,
    mark_atoms: Option<Arc<[AdjacencyMask]>>,
    algorithm: Option<Arc<str>>,
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
    /// `orientation_marginals` is the same directed-edge inclusion as `edge_marginals` (a DAG
    /// mask has no separate undirected bit); it is accepted for the call shape of the earlier
    /// constructor and refused when it differs, so a caller cannot believe two distinct
    /// quantities were stored.
    ///
    /// # Errors
    ///
    /// Empty ensemble, length mismatch, non-finite weights, non-positive mass, a variable count
    /// whose edge bits a `u32` does not address, orientation marginals that differ from the
    /// edge marginals, or two distinct adjacency sets with one key.
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
        if !edge_bits_fit(n_vars) {
            return Err(DiscoveryError::unsupported(
                "graph posterior has more edge bits than a u32 bit index addresses",
            ));
        }
        let cell = n_vars.saturating_mul(n_vars);
        if edge_marginals.len() != cell || orientation_marginals.len() != cell {
            return Err(DiscoveryError::unsupported("edge/orientation marginal length mismatch"));
        }
        if edge_marginals
            .iter()
            .zip(orientation_marginals.iter())
            .any(|(edge, orientation)| edge.to_bits() != orientation.to_bits())
        {
            return Err(DiscoveryError::unsupported(
                "orientation marginals are the directed-edge marginals and must equal them",
            ));
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
        Ok(Self {
            n_vars,
            weights,
            atoms,
            graph_keys,
            edge_marginals,
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
            lag_bidirected_masks: None,
            atom_kind: GraphPosteriorAtomKind::Dag,
            mark_atoms: None,
            algorithm: None,
        })
    }

    /// Number of variables.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of retained graph atoms (unique DAGs or MCMC samples).
    #[must_use]
    pub fn n_graphs(&self) -> usize {
        self.weights.len()
    }

    /// Normalized posterior weights (length [`Self::n_graphs`]).
    #[must_use]
    pub fn weights(&self) -> &Arc<[f64]> {
        &self.weights
    }

    /// Packed adjacency sets (length [`Self::n_graphs`]); see [`edge_bit`]. The canonical
    /// description of the graphs.
    #[must_use]
    pub fn atoms(&self) -> &Arc<[AdjacencyMask]> {
        &self.atoms
    }

    /// The single-word packing of [`Self::atoms`], the earlier representation: `None` unless
    /// every atom fits one word (a graph on at most eight nodes).
    #[must_use]
    pub fn adjacency_words(&self) -> Option<Vec<u64>> {
        self.atoms.iter().map(AdjacencyMask::to_word).collect()
    }

    /// Opaque keys: [`AdjacencyMask::key`] of each adjacency set (the single-word mask itself
    /// for a graph on at most eight nodes). Distinct sets always carry distinct keys.
    #[must_use]
    pub fn graph_keys(&self) -> &Arc<[u64]> {
        &self.graph_keys
    }

    /// Directed-edge inclusion, length `n_vars * n_vars`.
    ///
    /// For a DAG mask this is `P(i → j)`. It is the same event as
    /// [`Self::orientation_marginals`]: the mask has no separate undirected bit.
    #[must_use]
    pub fn edge_marginals(&self) -> &Arc<[f64]> {
        &self.edge_marginals
    }

    /// `P(i → j)`: the same values as [`Self::edge_marginals`]. A PAG atom's circle marks live
    /// in [`Self::mark_atoms`], not here.
    #[must_use]
    pub fn orientation_marginals(&self) -> &Arc<[f64]> {
        &self.edge_marginals
    }

    /// Kish effective number of graphs, `(Σw)² / Σw²`, of the weight vector.
    ///
    /// A measure of how concentrated the posterior is over graphs (1 = a single graph),
    /// **not** a sampling effective sample size: a sharply identified exact posterior has a
    /// value near 1. Sampling accuracy of MCMC posteriors is in
    /// [`InferenceDiagnostics::ess_bulk_min`] and [`Self::edge_mcse`].
    #[must_use]
    pub fn ess(&self) -> f64 {
        self.ess
    }

    /// Chain / analytic diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &InferenceDiagnostics {
        &self.diagnostics
    }

    /// Graphs rejected as cyclic / constraint-invalid during search.
    #[must_use]
    pub fn rejected_invalid(&self) -> u64 {
        self.rejected_invalid
    }

    /// Monte-Carlo standard error of each entry of [`Self::edge_marginals`] (same packing),
    /// `sqrt(p(1-p) / ESS)` from the chains' bulk ESS. `None` for exact posteriors, whose
    /// marginals carry no sampling error; `0` for an indicator that never varied.
    #[must_use]
    pub fn edge_mcse(&self) -> Option<&Arc<[f64]>> {
        self.edge_mcse.as_ref()
    }

    /// Split R-hat of each entry of [`Self::edge_marginals`]'s indicator across the chains
    /// (same packing). `None` for exact posteriors. A cell whose indicator never varied, or was
    /// not sampled, reports `1`; an indicator the chains disagree on without moving reports `∞`.
    #[must_use]
    pub fn edge_rhat(&self) -> Option<&Arc<[f64]>> {
        self.edge_rhat.as_ref()
    }

    /// Bulk effective sample size of each entry of [`Self::edge_marginals`]'s indicator across
    /// the chains (same packing). `None` for exact posteriors. A cell whose indicator never
    /// varied, or was not sampled, reports the total retained draw count.
    #[must_use]
    pub fn edge_ess(&self) -> Option<&Arc<[f64]>> {
        self.edge_ess.as_ref()
    }

    /// Monte-Carlo standard error of each entry of [`Self::lagged_edge_marginals`].
    #[must_use]
    pub fn lagged_edge_mcse(&self) -> Option<&Arc<[f64]>> {
        self.lagged_edge_mcse.as_ref()
    }

    /// Optional lagged-edge marginals for DBN templates
    /// (packing `(lag - 1) * n_vars * n_vars + from * n_vars + to`, lag ≥ 1).
    #[must_use]
    pub fn lagged_edge_marginals(&self) -> Option<&Arc<[f64]>> {
        self.lagged_edge_marginals.as_ref()
    }

    /// Max lag encoded in [`Self::lagged_edge_marginals`] (`None` if static).
    #[must_use]
    pub fn max_lag(&self) -> Option<u32> {
        self.max_lag
    }

    /// Optional per-atom lag-edge sets (DBN templates; same length as [`Self::atoms`]); bit
    /// `(lag - 1) * n_vars² + from * n_vars + to` is the edge `from(t - lag) → to(t)`.
    #[must_use]
    pub fn lag_masks(&self) -> Option<&Arc<[LagMask]>> {
        self.lag_masks.as_ref()
    }

    /// Optional per-atom lagged latent-confounder sets (same length and bit layout as
    /// [`Self::lag_masks`]): bit `(lag - 1) * n_vars² + from * n_vars + to` is
    /// `from(t - lag) ↔ to(t)`. Present only for lagged [`GraphPosteriorAtomKind::TemporalAdmg`]
    /// atoms, whose directed lagged edges are [`Self::lag_masks`].
    #[must_use]
    pub fn lag_bidirected_masks(&self) -> Option<&Arc<[LagMask]>> {
        self.lag_bidirected_masks.as_ref()
    }

    /// Class of every atom. [`GraphPosterior::from_atoms`] leaves this as
    /// [`GraphPosteriorAtomKind::Dag`].
    #[must_use]
    pub fn atom_kind(&self) -> GraphPosteriorAtomKind {
        self.atom_kind
    }

    /// Optional PAG circle-endpoint sets, same length as [`Self::atoms`]. Bit `(from, to)`
    /// means the `from` endpoint of that ordered pair is a circle. Unused for DAG and CPDAG
    /// atoms.
    #[must_use]
    pub fn mark_atoms(&self) -> Option<&Arc<[AdjacencyMask]>> {
        self.mark_atoms.as_ref()
    }

    /// The single-word packing of [`Self::mark_atoms`]: `None` unless there are marks and every
    /// one fits one word.
    #[must_use]
    pub fn mark_words(&self) -> Option<Vec<u64>> {
        self.mark_atoms.as_ref()?.iter().map(AdjacencyMask::to_word).collect()
    }

    /// The circle-endpoint set of atom `index`: empty when the posterior carries no marks.
    #[must_use]
    pub fn mark_atom(&self, index: usize) -> AdjacencyMask {
        self.mark_atoms
            .as_ref()
            .and_then(|marks| marks.get(index))
            .cloned()
            .unwrap_or(AdjacencyMask::EMPTY)
    }

    /// Which structure-learning algorithm produced this posterior.
    ///
    /// `None` when the producer did not tag it. Carried so downstream plan records and
    /// artifacts can name the real algorithm rather than a generic label — a posterior
    /// handed to a study no longer arrives with that provenance implied by the call.
    #[must_use]
    pub fn algorithm(&self) -> Option<&Arc<str>> {
        self.algorithm.as_ref()
    }

    /// Replace the edge marginals (the orientation marginals follow), keeping their length.
    ///
    /// For a producer that floors or otherwise adjusts the marginals after the fact; the
    /// per-edge Monte-Carlo columns are untouched.
    ///
    /// # Errors
    ///
    /// Length differs from `n_vars²`.
    pub(crate) fn replace_edge_marginals(
        &mut self,
        marginals: impl Into<Arc<[f64]>>,
    ) -> Result<(), DiscoveryError> {
        let marginals = marginals.into();
        if marginals.len() != self.edge_marginals.len() {
            return Err(DiscoveryError::unsupported("edge marginal length mismatch"));
        }
        self.edge_marginals = marginals;
        Ok(())
    }

    /// Record a note in the diagnostics of the run.
    pub(crate) fn push_note(&mut self, note: impl Into<Arc<str>>) {
        self.diagnostics.notes.push(note.into());
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

    /// Attach PAG circle-endpoint masks (same length as [`Self::atoms`]), single-word
    /// bitmasks: [`Self::with_mark_atoms`] takes them too.
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
        if marks.len() != self.n_graphs() {
            return Err(DiscoveryError::unsupported("mark_masks/adjacency length mismatch"));
        }
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

    /// Attach per-atom lag-edge sets (same length as the atoms): single-word `u64` packings or
    /// [`LagMask`]s of any width.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn with_lag_masks(
        mut self,
        masks: impl IntoIterator<Item = impl Into<LagMask>>,
    ) -> Result<Self, DiscoveryError> {
        let masks: Arc<[LagMask]> = masks.into_iter().map(Into::into).collect();
        if masks.len() != self.n_graphs() {
            return Err(DiscoveryError::unsupported("lag_masks/adjacency length mismatch"));
        }
        self.lag_masks = Some(masks);
        Ok(self)
    }

    /// Drop the lag template (lag masks, latent-confounder lag masks, lagged marginals and
    /// their errors, `max_lag`), leaving the contemporaneous atoms.
    #[must_use]
    pub fn without_lag_template(mut self) -> Self {
        self.lag_masks = None;
        self.lag_bidirected_masks = None;
        self.lagged_edge_marginals = None;
        self.lagged_edge_mcse = None;
        self.max_lag = None;
        self
    }

    /// Attach per-atom lagged latent-confounder sets to a lagged temporal-ADMG posterior.
    ///
    /// The directed lagged edges are [`Self::lag_masks`] and the contemporaneous ADMG is
    /// [`Self::atoms`]; with this the atom of position `i` is
    /// [`Self::temporal_admg_atom`]`(i)`.
    ///
    /// # Errors
    ///
    /// The posterior is not [`GraphPosteriorAtomKind::TemporalAdmg`], carries no lag masks or
    /// `max_lag` yet, or the masks differ in length from the atoms.
    pub fn with_lag_bidirected_masks(
        mut self,
        masks: impl IntoIterator<Item = impl Into<LagMask>>,
    ) -> Result<Self, DiscoveryError> {
        let masks: Arc<[LagMask]> = masks.into_iter().map(Into::into).collect();
        if self.atom_kind != GraphPosteriorAtomKind::TemporalAdmg {
            return Err(DiscoveryError::unsupported(
                "lagged latent-confounder masks belong to TemporalAdmg atoms",
            ));
        }
        if self.lag_masks.is_none() || self.max_lag.is_none() {
            return Err(DiscoveryError::unsupported(
                "lagged latent-confounder masks require the directed lag masks and max_lag first",
            ));
        }
        if masks.len() != self.n_graphs() {
            return Err(DiscoveryError::unsupported(
                "lag_bidirected_masks/adjacency length mismatch",
            ));
        }
        self.lag_bidirected_masks = Some(masks);
        Ok(self)
    }

    /// Whether the atoms are lagged temporal-ADMG templates: [`GraphPosteriorAtomKind::TemporalAdmg`]
    /// atoms that carry per-atom lag masks and a `max_lag`.
    #[must_use]
    pub fn is_lagged_temporal_admg(&self) -> bool {
        self.atom_kind == GraphPosteriorAtomKind::TemporalAdmg
            && self.lag_masks.is_some()
            && self.max_lag.is_some()
    }

    /// Atom `index` of a lagged temporal-ADMG posterior; `None` for any other posterior or an
    /// index outside it.
    #[must_use]
    pub fn temporal_admg_atom(&self, index: usize) -> Option<crate::TemporalAdmgAtom> {
        if !self.is_lagged_temporal_admg() || index >= self.n_graphs() {
            return None;
        }
        let directed = self.lag_masks.as_ref()?.get(index)?.clone();
        let bidirected = match self.lag_bidirected_masks.as_ref() {
            Some(masks) => masks.get(index)?.clone(),
            None => LagMask::EMPTY,
        };
        Some(crate::TemporalAdmgAtom {
            contemporaneous: self.atoms.get(index)?.clone(),
            lagged_directed: directed,
            lagged_bidirected: bidirected,
        })
    }

    /// Convert to envelope ensemble (all graphs marked identified).
    ///
    /// # Errors
    ///
    /// Shape failures from [`WeightedGraphSamples`].
    pub fn to_weighted_samples(&self) -> Result<WeightedGraphSamples, DiscoveryError> {
        if self.atom_kind == GraphPosteriorAtomKind::Dag {
            for atom in self.atoms.iter() {
                if !crate::adjacency_mask::mask_is_dag(atom, self.n_vars) {
                    let refusal = antecedent_graph::error::CyclicDagRefusal::unlicensed(
                        antecedent_graph::error::DagOperation::GraphPosterior,
                        "graph_posterior",
                        "a DAG-atom posterior does not drop a cyclic atom or mark it \
                         identified; hold the directed edges as a CyclicDigraph and read \
                         condensation, or ask sigma_separates for a conditional-independence \
                         question",
                    );
                    return Err(DiscoveryError::data_msg(refusal.to_string()));
                }
            }
        }
        let identified = Arc::from(vec![GraphIdentFlag::Identified; self.n_graphs()]);
        let mut g = WeightedGraphSamples::new(
            Arc::clone(&self.weights),
            identified,
            Arc::clone(&self.graph_keys),
        )?;
        g.edge_marginals = Some(Arc::clone(&self.edge_marginals));
        g.orientation_marginals = Some(Arc::clone(&self.edge_marginals));
        Ok(g)
    }

    /// The posterior over Markov equivalence classes this DAG posterior induces.
    ///
    /// Every DAG atom is replaced by its CPDAG (the essential graph: its skeleton, its
    /// v-structures and the edges Meek's rules compel) and the posterior mass of the DAGs
    /// that share a CPDAG is summed, so the class probabilities are exact functions of the
    /// DAG posterior and a class the DAG posterior favours is not split across its members.
    /// The atoms keep first-seen order and carry [`GraphPosteriorAtomKind::Cpdag`]: both
    /// directed bits of an unordered pair mark an undirected edge. The edge marginals are the
    /// class posterior's bit inclusion, so an undirected edge counts in both directions.
    ///
    /// The chain diagnostics of a sampled posterior describe the DAG draws; the Monte-Carlo
    /// standard errors, split R-hat and bulk ESS of the per-edge DAG indicators do not carry
    /// over to class bit indicators and are dropped. The Kish count is recomputed on the
    /// class weights, and a note records the collapse.
    ///
    /// # Errors
    ///
    /// A posterior whose atoms are not DAGs, a lagged template posterior, or an atom that is
    /// not acyclic.
    pub fn to_equivalence_classes(&self) -> Result<Self, DiscoveryError> {
        if self.atom_kind != GraphPosteriorAtomKind::Dag {
            return Err(DiscoveryError::unsupported(
                "only a posterior over DAG atoms collapses to equivalence classes",
            ));
        }
        if self.max_lag.is_some() || self.lag_masks.is_some() {
            return self.to_temporal_equivalence_classes();
        }
        let n = self.n_vars;
        let mut slot: std::collections::HashMap<AdjacencyMask, usize> =
            std::collections::HashMap::with_capacity(self.atoms.len());
        let mut classes: Vec<AdjacencyMask> = Vec::new();
        let mut mass: Vec<f64> = Vec::new();
        for (atom, &weight) in self.atoms.iter().zip(self.weights.iter()) {
            let dag = crate::adjacency_mask::dag_from_adjacency_mask(atom, n)?;
            let cpdag = crate::ges::dag_to_cpdag(&dag)?;
            let class = crate::adjacency_mask::adjacency_mask_from_cpdag(&cpdag)?;
            let at = *slot.entry(class.clone()).or_insert_with(|| {
                classes.push(class);
                mass.push(0.0);
                classes.len() - 1
            });
            mass[at] += weight;
        }
        let total: f64 = mass.iter().sum();
        let weights: Vec<f64> = mass.iter().map(|m| m / total).collect();
        let square: f64 = weights.iter().map(|w| w * w).sum();
        let ess = if square > 0.0 { 1.0 / square } else { 1.0 };
        let (edge, orientation) = accumulate_marginals(n, &weights, &classes);
        let mut diagnostics = self.diagnostics.clone();
        diagnostics.notes.push(Arc::from(format!(
            "equivalence_class_collapse dag_atoms={} classes={} effective_class_count={ess:.4}",
            self.n_graphs(),
            classes.len()
        )));
        let collapsed = Self::from_atoms(
            n,
            weights,
            classes,
            edge,
            orientation,
            SamplingSummary { ess, diagnostics, rejected_invalid: self.rejected_invalid },
        )?
        .with_atom_kind(GraphPosteriorAtomKind::Cpdag);
        Ok(match &self.algorithm {
            Some(algorithm) => collapsed.with_algorithm(format!("class_posterior({algorithm})")),
            None => collapsed.with_algorithm("class_posterior"),
        })
    }
}

impl GraphPosterior {
    /// The posterior over temporal equivalence classes a bounded-lag template posterior induces.
    ///
    /// Each template is unrolled to a window of `2L + 1` time slots of the `p` variables
    /// (`L` the maximum lag): the contemporaneous edges are repeated in slots `0..=L` and every
    /// lagged edge from slot `t + l` to slot `t` for `t` in `0..=L`, so each node of slots
    /// `0..=L` has its full parent set and the earliest slots are roots. Edges between slots
    /// are oriented by time, so they start directed; the unshielded colliders of the window and
    /// Meek's rules R1 to R4 then compel what they can, and the class is read off the
    /// contemporaneous edges of slot 0 (undirected edges carry both bits). Templates that share a
    /// lag mask and a contemporaneous class pool their mass.
    fn to_temporal_equivalence_classes(&self) -> Result<Self, DiscoveryError> {
        let (Some(max_lag), Some(lag_masks)) = (self.max_lag, self.lag_masks.as_ref()) else {
            return Err(DiscoveryError::unsupported(
                "a temporal collapse needs the lag masks of the template atoms",
            ));
        };
        let p = self.n_vars;
        let lags = max_lag as usize;
        let cell = lags * p * p;
        let slots = 2 * lags + 1;
        if slots * p > 255 {
            return Err(DiscoveryError::unsupported("the unrolled window is too wide to collapse"));
        }
        let mut slot: std::collections::HashMap<(AdjacencyMask, LagMask), usize> =
            std::collections::HashMap::with_capacity(self.atoms.len());
        let mut classes: Vec<AdjacencyMask> = Vec::new();
        let mut lag_keys: Vec<LagMask> = Vec::new();
        let mut mass: Vec<f64> = Vec::new();
        let node = |index: usize| DenseNodeId::from_raw(crate::indexing::dense_u32(index));
        for ((atom, lag_mask), &weight) in
            self.atoms.iter().zip(lag_masks.iter()).zip(self.weights.iter())
        {
            let mut window = Dag::with_variables(crate::indexing::dense_u32(slots * p));
            for slot in 0..=lags {
                for i in 0..p {
                    for j in 0..p {
                        if i != j && atom.has_edge(p, i, j) {
                            window.insert_directed(node(slot * p + i), node(slot * p + j))?;
                        }
                    }
                }
                for b in lag_mask.set_bits().map(|b| b as usize).filter(|&b| b < cell) {
                    let lag = b / (p * p) + 1;
                    let from = (b % (p * p)) / p;
                    let to = b % p;
                    window.insert_directed(node((slot + lag) * p + from), node(slot * p + to))?;
                }
            }
            let cpdag = crate::ges::dag_to_cpdag_with_known(&window, |from, to| {
                from.as_usize() / p != to.as_usize() / p
            })?;
            let mut class = AdjacencyMask::EMPTY;
            for edge in cpdag.edges() {
                let (i, j) = (edge.a.as_usize(), edge.b.as_usize());
                if i >= p || j >= p {
                    continue;
                }
                if edge.is_undirected() {
                    class.set_edge(p, i, j, true);
                    class.set_edge(p, j, i, true);
                } else if let Some((from, to)) = edge.parent_child() {
                    class.set_edge(p, from.as_usize(), to.as_usize(), true);
                }
            }
            let at = *slot.entry((class.clone(), lag_mask.clone())).or_insert_with(|| {
                classes.push(class);
                lag_keys.push(lag_mask.clone());
                mass.push(0.0);
                classes.len() - 1
            });
            mass[at] += weight;
        }
        let total: f64 = mass.iter().sum();
        let weights: Vec<f64> = mass.iter().map(|m| m / total).collect();
        let square: f64 = weights.iter().map(|w| w * w).sum();
        let ess = if square > 0.0 { 1.0 / square } else { 1.0 };
        let (edge, orientation) = accumulate_marginals(p, &weights, &classes);
        let mut diagnostics = self.diagnostics.clone();
        diagnostics.notes.push(Arc::from(format!(
            "temporal_equivalence_class_collapse template_atoms={} classes={} \
             window_slots={slots}",
            self.n_graphs(),
            classes.len()
        )));
        let mut collapsed = Self::from_atoms(
            p,
            weights,
            classes,
            edge,
            orientation,
            SamplingSummary { ess, diagnostics, rejected_invalid: self.rejected_invalid },
        )?
        .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
        .with_lag_masks(lag_keys)?;
        if let Some(lagged) = &self.lagged_edge_marginals {
            collapsed = collapsed.with_lagged_marginals(max_lag, Arc::clone(lagged))?;
        }
        Ok(match &self.algorithm {
            Some(algorithm) => collapsed.with_algorithm(format!("class_posterior({algorithm})")),
            None => collapsed.with_algorithm("class_posterior"),
        })
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

/// Refuse a node count whose edge bits a `u32` bit index does not address.
///
/// `message` stays the caller's: order MCMC, structure MCMC and the DBN posterior each name
/// their own engine.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] with `message` when `n * (n - 1)` exceeds `u32::MAX`.
pub(crate) fn ensure_nodes_fit(n: usize, message: &'static str) -> Result<(), DiscoveryError> {
    if edge_bits_fit(n) { Ok(()) } else { Err(DiscoveryError::unsupported(message)) }
}

// The single-word functions below are the earlier `u64` API, kept for graphs on at most eight
// nodes (`n * (n - 1) <= 64` edge bits). The same operations over [`AdjacencyMask`], for any
// node count, live in `crate::adjacency_mask`.

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
/// Packing overflow or a bow, which cannot be encoded without dropping an edge.
pub fn adjacency_mask_from_admg(admg: &Admg) -> Result<u64, DiscoveryError> {
    require_word_capacity(admg.node_count(), "adjacency_mask_from_admg")?;
    into_word(crate::adjacency_mask::adjacency_mask_from_admg(admg)?, "adjacency_mask_from_admg")
}

/// One key per adjacency set, in atom order.
///
/// A key is the set itself on at most eight nodes and a 64-bit digest beyond, so two distinct
/// sets could in principle share one. Downstream caches index atoms by key; a shared key would
/// merge two graphs, so it is refused instead.
fn distinct_graph_keys(adjacency: &[AdjacencyMask]) -> Result<Arc<[u64]>, DiscoveryError> {
    let mut owner: std::collections::HashMap<u64, &AdjacencyMask> =
        std::collections::HashMap::with_capacity(adjacency.len());
    let mut keys = Vec::with_capacity(adjacency.len());
    for mask in adjacency {
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
    mask: &AdjacencyMask,
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
    forbidden_parents: Vec<AdjacencyMask>,
    required_parents: Vec<AdjacencyMask>,
    max_parents: usize,
    /// `(ln p, ln (1 - p))` of the Bernoulli edge prior, when one is set.
    edge_logs: Option<(f64, f64)>,
}

/// The fixed inputs of one parent-set enumeration.
struct ParentSetSearch<'a> {
    node: usize,
    /// Candidate parents that are neither forbidden nor required, in increasing order.
    free: &'a [usize],
    required: &'a AdjacencyMask,
    limit: usize,
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
        let node_sets = |mask: &AdjacencyMask| -> Vec<AdjacencyMask> {
            (0..n).map(|node| parent_bits(mask, n, node)).collect()
        };
        Self {
            n,
            forbidden_parents: node_sets(&forbidden),
            required_parents: node_sets(&required),
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
    pub(crate) fn node_log_prior(&self, node: usize, parents: &AdjacencyMask) -> Option<f64> {
        if parents.intersects(&self.forbidden_parents[node])
            || !parents.contains_all(&self.required_parents[node])
            || parents.count_ones() > self.max_parents
        {
            return None;
        }
        let Some((lp, lq)) = self.edge_logs else {
            return Some(0.0);
        };
        let candidates = self.n - 1 - self.forbidden_parents[node].count_ones();
        let present = parents.count_ones();
        Some(present as f64 * lp + (candidates - present) as f64 * lq)
    }

    /// Every parent set constraints allow for `node`, as node bitsets in increasing numeric
    /// order of the set (node `p` is bit `p`), or `None` when there are more than `limit` of
    /// them. Only as many sets as the limit are ever generated.
    pub(crate) fn allowed_parent_sets(
        &self,
        node: usize,
        limit: usize,
    ) -> Option<Vec<AdjacencyMask>> {
        let required = &self.required_parents[node];
        let free: Vec<usize> = (0..self.n)
            .filter(|&p| {
                p != node
                    && !self.forbidden_parents[node].bit(crate::indexing::dense_u32(p))
                    && !required.bit(crate::indexing::dense_u32(p))
            })
            .collect();
        let Some(spare) = self.max_parents.checked_sub(required.count_ones()) else {
            return Some(Vec::new());
        };
        let mut out = Vec::new();
        let mut chosen = Vec::new();
        let complete = self.extend_parent_sets(
            &ParentSetSearch { node, free: &free, required, limit },
            free.len(),
            spare,
            &mut chosen,
            &mut out,
        );
        complete.then_some(out)
    }

    /// Emit, in increasing numeric order, the allowed sets made of `required` plus at most
    /// `spare` of the first `upto` free nodes, `chosen` holding the higher nodes already taken.
    /// `false` once more than `search.limit` sets exist.
    fn extend_parent_sets(
        &self,
        search: &ParentSetSearch<'_>,
        upto: usize,
        spare: usize,
        chosen: &mut Vec<usize>,
        out: &mut Vec<AdjacencyMask>,
    ) -> bool {
        if upto == 0 {
            let mut set = search.required.clone();
            for &p in chosen.iter() {
                set.set_bit(crate::indexing::dense_u32(p), true);
            }
            if self.node_log_prior(search.node, &set).is_none() {
                return true;
            }
            if out.len() == search.limit {
                return false;
            }
            out.push(set);
            return true;
        }
        // Sets without the highest remaining node are numerically smaller than those with it.
        if !self.extend_parent_sets(search, upto - 1, spare, chosen, out) {
            return false;
        }
        if spare == 0 {
            return true;
        }
        chosen.push(search.free[upto - 1]);
        let complete = self.extend_parent_sets(search, upto - 1, spare - 1, chosen, out);
        chosen.pop();
        complete
    }

    /// Log prior of `mask`, `None` when it violates a constraint or is cyclic.
    pub(crate) fn log_prior(&self, mask: &AdjacencyMask) -> Option<f64> {
        let n = self.n;
        if mask.intersects(&self.forbidden) || !mask.contains_all(&self.required) {
            return None;
        }
        if self.max_parents < n.saturating_sub(1) {
            let mut in_degree = vec![0usize; n];
            for (_, to) in mask.edges(n) {
                in_degree[to] += 1;
            }
            if in_degree.iter().any(|&d| d > self.max_parents) {
                return None;
            }
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
    for (&w, mask) in weights.iter().zip(masks.iter()) {
        // Only the set bits: a sparse graph costs its edge count, not `n²`.
        for (from, to) in mask.edges(n) {
            edge[from * n + to] += w;
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

/// Fewest value changes an edge indicator must make in every chain for that chain's mean to be
/// compared with the others. A two-state series with `t` changes holds about `t / 2` independent
/// excursions into its rarer state, so its mean has no usable standard error below a handful.
pub(crate) const MIN_INDICATOR_TRANSITIONS: usize = 6;

/// An indicator whose rarer value occurs fewer than this many times over all draws of all chains
/// is judged by the pooled count alone: its marginal is within `MIN_RARE_EVENTS / draws` of 0 or 1
/// whatever the chains do, so a chain that rarely changes it cannot move the answer.
const MIN_RARE_EVENTS: usize = 4 * MIN_INDICATOR_TRANSITIONS;

/// Largest standardised distance between one chain's marginal of an edge and the pooled marginal.
/// A run holds a few hundred comparisons (indicators times chains) and the batch-means error
/// behind each rests on [`AGREEMENT_BATCHES`] batches, so its ratio has a Student-t tail:
/// 6 (two-sided 1e-5 at 19 degrees of freedom) keeps the false refusal rate of a correct run
/// near 1e-3.
pub(crate) const MAX_CHAIN_DEVIATION_Z: f64 = 6.0;

/// Largest rank-normalised split R̂ of the log target along the chains. The log target is a
/// continuous summary of the whole graph, so it exposes chains sitting in different regions that
/// every individual edge indicator would hide behind a rarely flipping value.
pub(crate) const MAX_SCORE_RHAT: f64 = 1.1;

/// Batches the draws of one chain are cut into to estimate the standard error of its marginal.
const AGREEMENT_BATCHES: usize = 20;

/// Whether the chains of a graph sampler describe the same distribution, judged on what the
/// indicator R̂ and ESS cannot see.
///
/// R̂ of an indicator that a chain never changes has no within-chain variance, and the Geyer ESS
/// of an indicator that changes a few times is far too large. This asks the questions directly.
/// Per indicator: how far is each chain's marginal from the pooled marginal in units of that
/// chain's standard error (the batch-means error, which cannot be trusted when batches hold
/// few changes of the indicator; there the excursion bound `sqrt(p (1 - p) / (changes / 2))`
/// replaces it), and does every chain change the indicator often enough to
/// be judged at all. Overall: do the chains agree on the log target.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainAgreement {
    /// Largest standardised distance of a chain's marginal from the pooled one.
    pub max_deviation_z: f64,
    /// Trace parameter attaining [`Self::max_deviation_z`].
    pub worst_param: Option<usize>,
    /// Indicators some chain changes fewer than [`MIN_INDICATOR_TRANSITIONS`] times although
    /// their pooled marginal is not within the rare-event bound of 0 or 1.
    pub unresolved: usize,
    /// Rank-normalised split R̂ of the log target (`∞` when the chains cannot be compared).
    pub score_rhat: f64,
}

impl ChainAgreement {
    /// Chains agree on every judged edge marginal.
    #[must_use]
    pub(crate) fn deviation_ok(&self) -> bool {
        self.max_deviation_z.is_finite() && self.max_deviation_z <= MAX_CHAIN_DEVIATION_Z
    }

    /// Every indicator is changed often enough in every chain to be judged. Reported, not
    /// gated: an edge the data nearly fix flips rarely in a converged chain, so a transition
    /// count cannot tell a stuck chain from a decided edge; [`Self::deviation_ok`] judges
    /// agreement with the excursion error of such an indicator instead.
    #[must_use]
    pub(crate) fn resolved(&self) -> bool {
        self.unresolved == 0
    }

    /// The chains agree on the log target.
    #[must_use]
    pub(crate) fn score_ok(&self) -> bool {
        self.score_rhat.is_finite() && self.score_rhat <= MAX_SCORE_RHAT
    }

    /// The gated criteria hold: chain agreement on every edge marginal and on the log target.
    #[must_use]
    pub(crate) fn agrees(&self) -> bool {
        self.deviation_ok() && self.score_ok()
    }
}

/// [`ChainAgreement`] of `traces` (edge indicators) and `scores` (the log target at each draw),
/// both laid out chain-major as in [`graph_chain_summary`] (`scores` has one value per draw).
#[must_use]
pub(crate) fn graph_chain_agreement(
    traces: &[f64],
    scores: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> ChainAgreement {
    let unjudgeable = ChainAgreement {
        max_deviation_z: f64::INFINITY,
        worst_param: None,
        unresolved: n_params,
        score_rhat: f64::INFINITY,
    };
    if n_chains < 2
        || n_draws < 8
        || traces.len() != n_chains * n_draws * n_params
        || scores.len() != n_chains * n_draws
    {
        return unjudgeable;
    }
    let score_rhat = score_rank_rhat(scores, n_chains, n_draws);
    let total = (n_chains * n_draws) as f64;
    let batch_len = (n_draws / AGREEMENT_BATCHES).max(1);
    let n_batches = n_draws / batch_len;
    let mut max_deviation_z = 0.0_f64;
    let mut worst_param = None;
    let mut unresolved = 0usize;
    let mut means = vec![0.0; n_chains];
    let mut changes = vec![0usize; n_chains];
    let mut errors = vec![0.0; n_chains];
    for p in 0..n_params {
        for c in 0..n_chains {
            let at = |d: usize| traces[(c * n_draws + d) * n_params + p];
            let mut sum = 0.0;
            let mut flips = 0usize;
            let mut batch_means = vec![0.0; n_batches];
            for d in 0..n_draws {
                let v = at(d);
                sum += v;
                if d > 0 && v.to_bits() != at(d - 1).to_bits() {
                    flips += 1;
                }
                if d / batch_len < n_batches {
                    batch_means[d / batch_len] += v / batch_len as f64;
                }
            }
            means[c] = sum / n_draws as f64;
            changes[c] = flips;
            errors[c] = if n_batches >= 2 {
                let grand = batch_means.iter().sum::<f64>() / n_batches as f64;
                let var = batch_means.iter().map(|m| (m - grand).powi(2)).sum::<f64>()
                    / (n_batches as f64 - 1.0);
                (var / n_batches as f64).sqrt()
            } else {
                0.0
            };
        }
        let pooled = means.iter().sum::<f64>() / n_chains as f64;
        let minority = (pooled.min(1.0 - pooled) * total).round();
        if minority < 1.0 {
            continue;
        }
        let variance = pooled * (1.0 - pooled);
        for c in 0..n_chains {
            // Batches that each hold several changes carry the chain's autocorrelation; with
            // fewer the batch means are too coarse, and the independent units of a two-state
            // series are its excursions into the rarer state, about half its changes.
            let units = if changes[c] >= 5 * n_batches {
                n_draws as f64
            } else {
                (changes[c] as f64 / 2.0).max(1.0)
            };
            let error = errors[c].max((variance / units).sqrt());
            let z = (means[c] - pooled).abs() / error;
            if z > max_deviation_z {
                max_deviation_z = z;
                worst_param = Some(p);
            }
        }
        let thin_chain = changes.iter().any(|&t| t < MIN_INDICATOR_TRANSITIONS);
        if thin_chain && minority >= MIN_RARE_EVENTS as f64 {
            unresolved += 1;
        }
    }
    ChainAgreement { max_deviation_z, worst_param, unresolved, score_rhat }
}

/// Rank-normalised split R̂ (Vehtari et al., 2021) of one scalar trace; `1` when the trace is
/// constant over every draw of every chain, `∞` when it cannot be computed.
fn score_rank_rhat(scores: &[f64], n_chains: usize, n_draws: usize) -> f64 {
    let (lo, hi) = scores
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if !lo.is_finite() || !hi.is_finite() {
        return f64::INFINITY;
    }
    if hi - lo <= 1e-12 {
        return 1.0;
    }
    let d = parameter_mcmc_diagnostics(scores, n_chains, n_draws, 1)[0];
    if d.rhat.is_finite() { d.rhat } else { f64::INFINITY }
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
    fn dags_of_one_equivalence_class_pool_their_mass() {
        let n = 3;
        let graph = |edges: &[(usize, usize)]| {
            edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(a, b)| set_edge(&mask, n, a, b, true))
        };
        // The three orientations of a chain without a collider share one class; the
        // collider is its own class.
        let atoms = [
            graph(&[(0, 1), (1, 2)]),
            graph(&[(2, 1), (1, 0)]),
            graph(&[(1, 0), (1, 2)]),
            graph(&[(0, 1), (2, 1)]),
        ];
        let weights = [0.1, 0.2, 0.3, 0.4];
        let (edge, orientation) = accumulate_marginals(n, &weights, &atoms);
        let posterior = GraphPosterior::from_atoms(
            n,
            weights.to_vec(),
            atoms,
            edge,
            orientation,
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let classes = posterior.to_equivalence_classes().unwrap();
        assert_eq!(classes.atom_kind(), GraphPosteriorAtomKind::Cpdag);
        assert_eq!(classes.n_graphs(), 2);
        assert!((classes.weights()[0] - 0.6).abs() < 1e-12);
        assert!((classes.weights()[1] - 0.4).abs() < 1e-12);
        // The chain class leaves both edges undirected (both bits); the collider is directed.
        assert!(has_edge(&classes.atoms()[0], n, 0, 1) && has_edge(&classes.atoms()[0], n, 1, 0));
        assert!(has_edge(&classes.atoms()[0], n, 1, 2) && has_edge(&classes.atoms()[0], n, 2, 1));
        assert!(has_edge(&classes.atoms()[1], n, 0, 1) && !has_edge(&classes.atoms()[1], n, 1, 0));
        assert!(has_edge(&classes.atoms()[1], n, 2, 1) && !has_edge(&classes.atoms()[1], n, 1, 2));
        assert!((classes.ess() - 1.0 / (0.6 * 0.6 + 0.4 * 0.4)).abs() < 1e-12);
        assert_eq!(classes.algorithm().map(|a| &**a), Some("class_posterior"));
        assert!(classes.edge_mcse().is_none());
        // Collapsing is not repeated on classes.
        assert!(classes.to_equivalence_classes().is_err());
    }

    #[test]
    fn lagged_parents_compel_the_contemporaneous_edges_they_shield() {
        let p = 2;
        let forward = set_edge(AdjacencyMask::EMPTY, p, 0, 1, true);
        let backward = set_edge(AdjacencyMask::EMPTY, p, 1, 0, true);
        // Bit 0 is the lag-1 edge from variable 0 to variable 0.
        let atoms = [forward.clone(), backward.clone(), forward, backward];
        let lags = [0_u64, 0, 1, 1];
        let weights = [0.1, 0.2, 0.3, 0.4];
        let (edge, orientation) = accumulate_marginals(p, &weights, &atoms);
        let posterior = GraphPosterior::from_atoms(
            p,
            weights.to_vec(),
            atoms,
            edge,
            orientation,
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_lagged_marginals(1, vec![0.0; 4])
        .unwrap()
        .with_lag_masks(lags.to_vec())
        .unwrap();
        let classes = posterior.to_equivalence_classes().unwrap();
        assert_eq!(classes.atom_kind(), GraphPosteriorAtomKind::Cpdag);
        // Without the lagged parent the two orientations are one undirected class; with it
        // each is compelled.
        assert_eq!(classes.n_graphs(), 3);
        assert!((classes.weights()[0] - 0.3).abs() < 1e-12);
        assert!(has_edge(&classes.atoms()[0], p, 0, 1) && has_edge(&classes.atoms()[0], p, 1, 0));
        assert!(has_edge(&classes.atoms()[1], p, 0, 1) && !has_edge(&classes.atoms()[1], p, 1, 0));
        assert!(has_edge(&classes.atoms()[2], p, 1, 0) && !has_edge(&classes.atoms()[2], p, 0, 1));
        assert_eq!(classes.max_lag(), Some(1));
        assert_eq!(
            classes.lag_masks().map(|m| m.iter().map(|x| x.to_word().unwrap()).collect::<Vec<_>>()),
            Some(vec![0_u64, 1, 1])
        );
    }

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
        assert!(!mask_is_dag(&mask, 3));
        let back = cpdag_from_adjacency_mask(&mask, 3).unwrap();
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
        assert!(!mask_is_dag(&mask, 3), "bidirected both-bits must not look like a DAG");
        assert!(dag_from_adjacency_mask(&mask, 3).is_err(), "ADMG bidirected packing is not a Dag");
        let back = admg_from_adjacency_mask(&mask, 3).unwrap();
        assert!(back.has_bidirected());
        assert!(back.children(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1)));
        assert!(back.children(DenseNodeId::from_raw(1)).contains(&DenseNodeId::from_raw(2)));
        assert!(
            back.bidirected_neighbors(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(2))
        );
    }

    /// A two-state trace of `n_draws` draws with marginal `p` that changes value at random
    /// (geometric run lengths), from a fixed seed.
    fn blinking(n_draws: usize, p: f64, mut state: u64) -> Vec<f64> {
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut on = next() < p;
        (0..n_draws)
            .map(|_| {
                // Independent redraw with probability 1/2 keeps the marginal at `p`.
                if next() < 0.5 {
                    on = next() < p;
                }
                f64::from(u8::from(on))
            })
            .collect()
    }

    fn agreement_of(chains: &[Vec<f64>], scores: &[Vec<f64>]) -> ChainAgreement {
        let (n_chains, n_draws) = (chains.len(), chains[0].len());
        graph_chain_agreement(&chains.concat(), &scores.concat(), n_chains, n_draws, 1)
    }

    #[test]
    fn chains_drawing_one_marginal_agree() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.3, 11 + c)).collect();
        let scores: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.5, 91 + c)).collect();
        let a = agreement_of(&chains, &scores);
        assert!(a.agrees(), "{a:?}");
    }

    /// Chains that change their indicator freely but at different marginals disagree, however
    /// well each chain mixes; R̂ of a binary trace with this much within-chain variance is mild.
    #[test]
    fn chains_with_different_marginals_are_refused() {
        let chains: Vec<Vec<f64>> = [0.30, 0.30, 0.30, 0.60]
            .iter()
            .zip(0u64..)
            .map(|(&p, c)| blinking(4000, p, 5 + c))
            .collect();
        let scores: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.5, 91 + c)).collect();
        let a = agreement_of(&chains, &scores);
        assert!(!a.deviation_ok(), "{a:?}");
    }

    /// A chain that never changes an indicator the others change cannot be judged, and the
    /// pooled marginal is far from 0 or 1: publication is blocked.
    #[test]
    fn a_chain_that_never_flips_a_contested_edge_blocks_publication() {
        let mut chains: Vec<Vec<f64>> = (0..3).map(|c| blinking(4000, 0.3, 21 + c)).collect();
        chains.push(vec![0.0; 4000]);
        let scores: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.5, 91 + c)).collect();
        let a = agreement_of(&chains, &scores);
        assert!(!a.resolved(), "{a:?}");
    }

    /// An edge no chain ever includes, or that only a handful of draws include, is settled at
    /// the pooled count and does not block.
    #[test]
    fn a_rare_edge_does_not_block_publication() {
        let mut chains: Vec<Vec<f64>> = vec![vec![0.0; 4000]; 4];
        for d in 0..3 {
            chains[1][100 + d] = 1.0;
        }
        let scores: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.5, 91 + c)).collect();
        let a = agreement_of(&chains, &scores);
        assert!(a.resolved(), "{a:?}");
    }

    /// Chains whose log targets sit at different levels have not found the same region.
    #[test]
    fn chains_at_different_log_target_levels_are_refused() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| blinking(4000, 0.3, 11 + c)).collect();
        let scores: Vec<Vec<f64>> = (0..4)
            .map(|c| blinking(4000, 0.5, 91 + c).iter().map(|v| v + 3.0 * (c % 2) as f64).collect())
            .collect();
        let a = agreement_of(&chains, &scores);
        assert!(!a.score_ok(), "{a:?}");
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

    /// The packing has no node limit but the `u32` edge bit index; the message is the caller's.
    #[test]
    fn adjacency_is_limited_only_by_the_edge_bit_index() {
        assert!(ensure_nodes_fit(16, "order MCMC").is_ok());
        assert!(ensure_nodes_fit(1000, "order MCMC").is_ok());
        let err = ensure_nodes_fit(70_000, "structure MCMC edge bits").unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message == "structure MCMC edge bits"),
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
        assert!(has_edge(&mask, n, 0, 1) && has_edge(&mask, n, 1, 3) && has_edge(&mask, n, 3, 2));
        assert!(!has_edge(&mask, n, 1, 0) && !has_edge(&mask, n, 2, 3));
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
            chain = set_edge(&chain, n, i, i + 1, true);
        }
        assert!(mask_is_dag(&chain, n));
        assert_eq!(chain.to_word(), None);
        assert_eq!(parents_of(&chain, n, 15), vec![14]);
        assert!(has_edge(&chain, n, 14, 15) && !has_edge(&chain, n, 15, 14));
        let cyclic = set_edge(&chain, n, 15, 0, true);
        assert!(!mask_is_dag(&cyclic, n));
        let dag = dag_from_adjacency_mask(&chain, n).unwrap();
        assert_eq!(dag.edges().count(), 15);
        assert_eq!(
            dag.topological_order().unwrap().first().map(|id| id.raw()),
            Some(0),
            "an acyclic chain still receives a topological order"
        );
        let err = dag_from_adjacency_mask(&cyclic, n).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("is not a DAG"), "{text}");
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("not passed to Dag::insert_directed"), "{text}");
        assert!(
            super::dag_from_adjacency_mask(0, 17).is_err(),
            "the legacy u64 adapter still rejects wider graphs"
        );
    }

    #[test]
    fn a_dag_posterior_does_not_mark_a_two_cycle_identified() {
        let n = 2;
        let chain = set_edge(AdjacencyMask::EMPTY, n, 0, 1, true);
        let cycle = set_edge(&chain, n, 1, 0, true);
        let sampling = || SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("test"),
            rejected_invalid: 0,
        };
        let acyclic = GraphPosterior::from_atoms(
            n,
            vec![1.0],
            vec![chain],
            vec![0.0; n * n],
            vec![0.0; n * n],
            sampling(),
        )
        .unwrap();
        assert!(acyclic.to_weighted_samples().is_ok());
        let cyclic = GraphPosterior::from_atoms(
            n,
            vec![1.0],
            vec![cycle],
            vec![0.0; n * n],
            vec![0.0; n * n],
            sampling(),
        )
        .unwrap();
        let err = cyclic.to_weighted_samples().unwrap_err();
        let text = err.to_string();
        assert!(text.contains("graph posterior"), "{text}");
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("sigma_separates"), "{text}");
    }

    #[test]
    fn graph_posterior_accepts_seventeen_variables_without_truncating_edges() {
        let n = 17;
        let chain =
            (0..n - 1).fold(AdjacencyMask::EMPTY, |mask, i| set_edge(mask, n, i, i + 1, true));
        let posterior = GraphPosterior::from_atoms(
            n,
            vec![1.0],
            vec![chain.clone()],
            vec![0.0; n * n],
            vec![0.0; n * n],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("test"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        assert_eq!(posterior.atoms()[0], chain);
        let dag = dag_from_adjacency_mask(&posterior.atoms()[0], n).unwrap();
        assert_eq!(dag.node_count(), n);
        assert_eq!(dag.edges().count(), n - 1);
        assert_eq!(parents_of(&posterior.atoms()[0], n, 16), vec![15]);
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
            vec![a.clone(), b, a.clone()],
            vec![0.0; n * n],
            vec![0.0; n * n],
            sampling,
        )
        .unwrap();
        assert_ne!(post.graph_keys()[0], post.graph_keys()[1]);
        assert_eq!(post.graph_keys()[0], post.graph_keys()[2]);
        assert_eq!(post.atoms()[0], a);
        assert!(post.adjacency_words().is_none(), "a wide atom has no single-word packing");
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
