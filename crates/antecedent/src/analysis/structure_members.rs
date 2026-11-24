//! The members of a declared causal structure, each with its identification of one effect.
//!
//! A route that reads a structural question against a declared structure adopts this carrier
//! rather than collapsing the structure to one oriented edge. A [`DeclaredStructure`] is one of
//! the ways a structure enters a causal flow, and [`DeclaredStructure::members`] (crate API)
//! hands the route every member structure, its weight when the form has weights, its direct
//! relation between the treatment and the outcome, and its identification of the effect of the
//! treatment on the outcome:
//!
//! | form | members | weights |
//! |---|---|---|
//! | one DAG, ADMG, temporal DAG or temporal ADMG | the graph | none |
//! | a CPDAG or PAG | every completion, from the class completion samplers; a PAG's MAG completions are read as ADMGs | none |
//! | a temporal CPDAG or temporal PAG | the completions of the temporal class envelope | none |
//! | an explicit candidate set | each declared graph | the declared weights, if any |
//! | a graph posterior over DAG or ADMG atoms | each distinct atom | its posterior mass |
//! | a cyclic system (a declared one, or the directed graph a cyclic discovery holds) | the graph | none |
//!
//! Identification reuses the identification `StudyBuilder::identify_structure` runs. A static
//! member is identified by the complete identification algorithm on its DAG or ADMG, so an
//! effect that is zero because no directed path runs from the treatment to the outcome is an
//! identified zero, not an unidentified member. A temporal member is identified for a one-step
//! pulse of the treatment, by the temporal identifier of its class; a temporal class keeps the
//! per-completion identification of its class envelope. A cyclic member is identified under the
//! semantics the request names ([`CyclicReading`]): at equilibrium by the generalized cyclic
//! identification of Forré and Mooij, or, for a process that unfolds in time, with each directed
//! edge read as a one-step lag and the unrolled temporal DAG identified as a temporal member.
//!
//! [`StructureMembers::summarize`] reduces per-member values the way class envelopes report
//! them: the range over identified members, the identified weight, and, when the form has
//! weights, the mixture conditional on the identified members.
//!
//! The canonical identity of a declared structure ([`DeclaredStructure::identity`]) commits to
//! its form, the canonical graph identity of every graph with every dense node's label, the
//! variable names, the weights, the cyclic premises and the provenance digest of the search it
//! came from. A route that adopts the carrier writes that identity and its own reading mode into
//! its program identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroU32;
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, IdentificationStatus, Lag, NodeRef,
    SearchLimits, SemanticDigest, TemporalEffectQuery, VariableId,
};
use antecedent_discovery::adjacency_mask::{admg_from_adjacency_mask, dag_from_adjacency_mask};
use antecedent_discovery::{AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind};
use antecedent_graph::completion::mag_to_admg;
use antecedent_graph::{
    Admg, CompletionSampler, CpdagCompletionSampler, CyclicMixedGraph,
    DEFAULT_COMPLETION_OPERATIONS, Dag, DenseNodeId, MarkedEdge, TemporalAdmg, TemporalDag,
    TemporalPag,
};
use antecedent_identify::TemporalCompletionGraph;
use antecedent_identify::cyclic_id::{CyclicIdPremises, CyclicIdStatus, identify_cyclic};
use antecedent_io_core::identity::GraphIdentityWire;
use serde::Serialize;

use super::supplied_identity::{Canonical, Declaration, Wire};
use crate::StructureProvenance;
use crate::accepted::{AcceptedGraph, GraphRef};
use crate::error::CausalError;
use crate::identify_api::{Identification, identify_study_within};
use crate::strategy_table::IdentifierId;
use crate::support::StructureSource;

/// Most members one declared structure enumerates. A class with more completions is truncated
/// and says so ([`StructureMembers::truncated`]).
pub const MAX_MEMBERS: usize = 1024;

/// Step cap of one generalized cyclic identification.
const CYCLIC_ID_STEPS: usize = 10_000;

/// A causal structure declared to a route, in any of the forms a causal flow carries.
#[derive(Clone, Debug)]
pub struct DeclaredStructure {
    form: Form,
    provenance: Option<Arc<StructureProvenance>>,
}

#[derive(Clone, Debug)]
enum Form {
    Graph(AcceptedGraph),
    Candidates { graphs: Vec<AcceptedGraph>, weights: Option<Vec<f64>> },
    Posterior { posterior: GraphPosterior, names: Vec<Arc<str>> },
    Cyclic { graph: CyclicMixedGraph, names: Vec<Arc<str>>, premises: CyclicIdPremises },
}

/// Which form a declared structure has.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StructureForm {
    /// One graph or one equivalence class.
    Graph,
    /// An explicit candidate set.
    Candidates,
    /// A posterior over structures.
    Posterior,
    /// A cyclic system.
    Cyclic,
}

impl StructureForm {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Graph => "graph",
            Self::Candidates => "candidates",
            Self::Posterior => "posterior",
            Self::Cyclic => "cyclic",
        }
    }
}

/// How a cyclic member is read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CyclicReading {
    /// The equilibrium of a simple cyclic SCM: generalized cyclic identification.
    Equilibrium,
    /// A process that unfolds in time: each directed edge is a one-step lag.
    Lagged,
}

impl CyclicReading {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equilibrium => "equilibrium",
            Self::Lagged => "lagged",
        }
    }
}

/// The direct relation between the treatment and the outcome in one member.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DirectRelation {
    forward: bool,
    reverse: bool,
    bidirected: bool,
}

impl DirectRelation {
    /// A directed edge runs from the treatment into the outcome (at some lag).
    #[must_use]
    pub const fn forward(self) -> bool {
        self.forward
    }

    /// A directed edge runs from the outcome into the treatment (at some lag).
    #[must_use]
    pub const fn reverse(self) -> bool {
        self.reverse
    }

    /// A bidirected (latent confounding) edge joins the two.
    #[must_use]
    pub const fn bidirected(self) -> bool {
        self.bidirected
    }

    /// Directed edges run both ways: a feedback pair.
    #[must_use]
    pub const fn feedback(self) -> bool {
        self.forward && self.reverse
    }
}

/// Whether a member identifies the requested effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MemberStatus {
    /// The member's identification derived the effect (a zero effect included).
    Identified,
    /// The identification finished without the effect.
    NotIdentified,
    /// A budget stopped the identification before it decided.
    Undecided,
}

/// One member structure.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum MemberGraph {
    /// A directed acyclic graph.
    Dag(Dag),
    /// An acyclic directed mixed graph (a MAG completion included).
    Admg(Admg),
    /// A temporal DAG.
    TemporalDag(TemporalDag),
    /// A temporal maximal ancestral graph (a completion of a temporal PAG).
    TemporalMag(TemporalPag),
    /// A temporal ADMG.
    TemporalAdmg(TemporalAdmg),
    /// A cyclic directed mixed graph.
    Cyclic(CyclicMixedGraph),
}

/// One member of a declared structure.
#[derive(Clone, Debug)]
pub struct StructureMember {
    graph: MemberGraph,
    weight: Option<f64>,
    relation: DirectRelation,
    status: MemberStatus,
    identification: Option<IdentificationStatus>,
    key: [u8; 32],
}

impl StructureMember {
    /// The member structure.
    #[must_use]
    pub const fn graph(&self) -> &MemberGraph {
        &self.graph
    }

    /// Normalized weight, when the form has weights.
    #[must_use]
    pub const fn weight(&self) -> Option<f64> {
        self.weight
    }

    /// The direct relation between the treatment and the outcome.
    #[must_use]
    pub const fn relation(&self) -> DirectRelation {
        self.relation
    }

    /// Whether the member identifies the effect.
    #[must_use]
    pub const fn status(&self) -> MemberStatus {
        self.status
    }

    /// The identification status the member's identifier returned, when it returned one.
    #[must_use]
    pub const fn identification(&self) -> Option<IdentificationStatus> {
        self.identification
    }

    /// Canonical identity of the member structure and its node labels.
    #[must_use]
    pub const fn key(&self) -> [u8; 32] {
        self.key
    }

    /// Whether the member identifies the effect.
    #[must_use]
    pub fn identified(&self) -> bool {
        self.status == MemberStatus::Identified
    }
}

/// Every member of a declared structure for one requested effect.
#[derive(Clone, Debug)]
pub struct StructureMembers {
    form: StructureForm,
    identity: SemanticDigest,
    members: Vec<StructureMember>,
    weighted: bool,
    truncated: bool,
    cyclic: CyclicReading,
}

/// Per-member values reduced the way class envelopes report them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MemberSummary {
    /// Smallest and largest value over the identified members; `None` when none identifies.
    pub envelope: Option<[f64; 2]>,
    /// Weighted mean over the identified members, conditional on them; `None` for a form
    /// without weights or when none identifies.
    pub mixture: Option<f64>,
    /// Weight of the identified members: the posterior or declared mass, or for a form
    /// without weights the uniform enumeration weight a class envelope uses.
    pub identified_weight: f64,
    /// Identified members.
    pub identified: usize,
    /// All members.
    pub members: usize,
}

impl StructureMembers {
    /// The declared form.
    #[must_use]
    pub const fn form(&self) -> StructureForm {
        self.form
    }

    /// Canonical identity of the declared structure.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Members in enumeration order.
    #[must_use]
    pub fn members(&self) -> &[StructureMember] {
        &self.members
    }

    /// Whether the form carries weights.
    #[must_use]
    pub const fn weighted(&self) -> bool {
        self.weighted
    }

    /// Whether a class enumeration stopped before the class was complete.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// How cyclic members were read.
    #[must_use]
    pub const fn cyclic_reading(&self) -> CyclicReading {
        self.cyclic
    }

    /// Whether any member identifies the effect.
    #[must_use]
    pub fn any_identified(&self) -> bool {
        self.members.iter().any(StructureMember::identified)
    }

    /// The member's weight for a mixture: its declared weight, or the uniform enumeration
    /// weight for a form without weights.
    #[must_use]
    pub fn mixing_weight(&self, member: &StructureMember) -> f64 {
        member.weight.unwrap_or_else(|| 1.0 / self.members.len().max(1) as f64)
    }

    /// Reduce one value per identified member: envelope, identified weight and, for a weighted
    /// form, the mixture conditional on the identified members.
    pub fn summarize(&self, value: impl Fn(&StructureMember) -> f64) -> MemberSummary {
        let mut envelope: Option<[f64; 2]> = None;
        let (mut mass, mut total) = (0.0_f64, 0.0_f64);
        let mut identified = 0;
        for member in self.members.iter().filter(|member| member.identified()) {
            let reading = value(member);
            envelope = Some(match envelope {
                None => [reading, reading],
                Some([low, high]) => [low.min(reading), high.max(reading)],
            });
            let weight = self.mixing_weight(member);
            mass += weight;
            total += weight * reading;
            identified += 1;
        }
        MemberSummary {
            envelope,
            mixture: (self.weighted && mass > 0.0).then(|| total / mass),
            identified_weight: mass,
            identified,
            members: self.members.len(),
        }
    }
}

impl DeclaredStructure {
    /// One accepted graph: a DAG, ADMG or temporal graph is one member; a CPDAG, PAG, temporal
    /// CPDAG or temporal PAG is its completions. Its provenance is the graph's own.
    #[must_use]
    pub fn graph(graph: AcceptedGraph) -> Self {
        let provenance = graph.provenance_shared();
        Self { form: Form::Graph(graph), provenance }
    }

    /// An explicit candidate set of single graphs over the same variable names, with optional
    /// nonnegative weights (normalized when the members are read).
    #[must_use]
    pub fn candidates(graphs: Vec<AcceptedGraph>, weights: Option<Vec<f64>>) -> Self {
        Self { form: Form::Candidates { graphs, weights }, provenance: None }
    }

    /// A posterior over DAG or ADMG atoms on `names`, in the posterior's variable order.
    #[must_use]
    pub fn posterior(posterior: GraphPosterior, names: Vec<String>) -> Self {
        Self {
            form: Form::Posterior { posterior, names: names.into_iter().map(Arc::from).collect() },
            provenance: None,
        }
    }

    /// A structure posterior accepted from a compiled discovery, carrying its provenance.
    #[must_use]
    pub fn accepted_posterior(
        accepted: super::structure_route::AcceptedPosterior,
        names: Vec<String>,
    ) -> Self {
        let (posterior, provenance) = accepted.into_parts();
        Self::posterior(posterior, names).with_provenance(provenance)
    }

    /// A cyclic system on `names` (dense node order), read under the declared premises of
    /// generalized cyclic identification.
    #[must_use]
    pub fn cyclic(graph: CyclicMixedGraph, names: Vec<String>, premises: CyclicIdPremises) -> Self {
        Self {
            form: Form::Cyclic {
                graph,
                names: names.into_iter().map(Arc::from).collect(),
                premises,
            },
            provenance: None,
        }
    }

    /// A cyclic system declared by name: directed and bidirected edges between `names`, read
    /// under the declared premises of generalized cyclic identification (the graph is the
    /// induced graph of a compatible ioSCM; the joint law has a density relative to a product of
    /// strongly connected component measures).
    ///
    /// # Errors
    /// An edge names a variable outside `names`, or an edge the graph refuses (a duplicate or a
    /// self-loop).
    pub fn cyclic_named(
        names: Vec<String>,
        directed: &[(String, String)],
        bidirected: &[(String, String)],
        io_scm: bool,
        scc_product_density: bool,
    ) -> Result<Self, CausalError> {
        let index = |name: &str| {
            names.iter().position(|item| item == name).map(dense).ok_or_else(|| {
                crate::unsupported_reason!(
                    "unknown_variable",
                    "a cyclic edge names a variable outside the declared names"
                )
            })
        };
        let count = u32::try_from(names.len()).map_err(|_| {
            crate::unsupported_reason!("invalid_argument", "too many cyclic variables")
        })?;
        let mut graph = CyclicMixedGraph::with_variables(count);
        for (from, to) in directed {
            graph.insert_directed(index(from)?, index(to)?).map_err(graph_error)?;
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(index(a)?, index(b)?).map_err(graph_error)?;
        }
        Ok(Self::cyclic(graph, names, CyclicIdPremises { io_scm, scc_product_density }))
    }

    /// Attach the provenance of the search the structure came from.
    #[must_use]
    pub fn with_provenance(mut self, provenance: Arc<StructureProvenance>) -> Self {
        self.form = match self.form {
            Form::Graph(graph) => Form::Graph(graph.with_provenance(Arc::clone(&provenance))),
            other => other,
        };
        self.provenance = Some(provenance);
        self
    }

    /// The declared form.
    #[must_use]
    pub const fn form(&self) -> StructureForm {
        match self.form {
            Form::Graph(_) => StructureForm::Graph,
            Form::Candidates { .. } => StructureForm::Candidates,
            Form::Posterior { .. } => StructureForm::Posterior,
            Form::Cyclic { .. } => StructureForm::Cyclic,
        }
    }

    /// The search the structure came from, when it came from one.
    #[must_use]
    pub fn provenance(&self) -> Option<&StructureProvenance> {
        self.provenance.as_deref()
    }

    /// The single accepted graph, for the one-graph form.
    #[must_use]
    pub const fn as_graph(&self) -> Option<&AcceptedGraph> {
        match &self.form {
            Form::Graph(graph) => Some(graph),
            _ => None,
        }
    }

    /// Ordered variable names the structure is bound to: `None` when a graph is not bound to
    /// names or the candidates are bound to different names.
    #[must_use]
    pub fn variable_names(&self) -> Option<Vec<Arc<str>>> {
        match &self.form {
            Form::Graph(graph) => graph.variable_names().map(<[Arc<str>]>::to_vec),
            Form::Candidates { graphs, .. } => {
                let first = graphs.first()?.variable_names()?;
                graphs
                    .iter()
                    .all(|graph| graph.variable_names() == Some(first))
                    .then(|| first.to_vec())
            }
            Form::Posterior { names, .. } | Form::Cyclic { names, .. } => Some(names.clone()),
        }
    }

    /// Canonical identity: the form, each graph's canonical identity with its node labels, the
    /// variable names, the weights, the cyclic premises and the provenance digest.
    ///
    /// # Errors
    /// A graph whose canonical identity cannot be encoded.
    pub fn identity(&self) -> Result<SemanticDigest, CausalError> {
        let mut declaration = Declaration::new("antecedent.structure_members.declared.v1");
        declaration.field("form", self.form().as_str());
        let names: Option<Vec<String>> =
            self.variable_names().map(|names| names.iter().map(ToString::to_string).collect());
        declaration.field("names", &names);
        match &self.form {
            Form::Graph(graph) => {
                declaration.field("graph", &AcceptedIdentity::of(graph)?);
            }
            Form::Candidates { graphs, weights } => {
                let identities =
                    graphs.iter().map(AcceptedIdentity::of).collect::<Result<Vec<_>, _>>()?;
                declaration.field("graphs", &identities);
                declaration.field("weights", weights);
            }
            Form::Posterior { posterior, .. } => {
                let atoms =
                    antecedent_io_core::identity::graph_posterior_atom_identities(posterior)
                        .map_err(|error| crate::compile_reason!("invalid_argument", "{error}"))?;
                declaration.field("atom_kind", posterior.atom_kind().as_str());
                declaration.field("atoms", &Wire(&atoms));
                declaration.field("weights", &posterior.weights().to_vec());
                declaration.field("lagged", &posterior.max_lag().is_some());
            }
            Form::Cyclic { graph, premises, .. } => {
                declaration.field("cyclic", &CyclicIdentity(graph));
                declaration.field("io_scm", &premises.io_scm);
                declaration.field("scc_product_density", &premises.scc_product_density);
            }
        }
        let provenance =
            self.provenance.as_ref().map(|item| crate::accepted::digest_hex(&item.digest()));
        declaration.field("provenance", &provenance);
        Ok(SemanticDigest::from_bytes(declaration.finish()))
    }

    /// Enumerate the members for the effect of `treatment` on `outcome`, identifying each.
    ///
    /// # Errors
    /// Names the structure does not bind, an invalid declaration (mismatched candidate names,
    /// weights that are not a nonnegative finite mass, a class atom or a lagged posterior, a
    /// randomized-trial contract), cancellation or a resource refusal.
    pub(crate) fn members(
        &self,
        treatment: &str,
        outcome: &str,
        cyclic: CyclicReading,
        ctx: &ExecutionContext,
    ) -> Result<StructureMembers, CausalError> {
        let names = self.variable_names().ok_or_else(|| {
            crate::unsupported_reason!(
                "schema_mismatch",
                "every graph of a declared structure is bound to the same variable names"
            )
        })?;
        let index = |name: &str| names.iter().position(|item| item.as_ref() == name);
        let (Some(treatment), Some(outcome)) = (index(treatment), index(outcome)) else {
            return Err(crate::unsupported_reason!(
                "unknown_variable",
                "the treatment and the outcome are variables of the declared structure"
            ));
        };
        if treatment == outcome {
            return Err(crate::unsupported_reason!(
                "invalid_argument",
                "a structural member reading compares two different variables"
            ));
        }
        let pair = Pair { treatment, outcome };
        let mut raw = Raw::default();
        let weights = match &self.form {
            Form::Graph(graph) => {
                graph_members(graph, None, pair, ctx, &mut raw)?;
                None
            }
            Form::Candidates { graphs, weights } => {
                if graphs.is_empty() || weights.as_ref().is_some_and(|w| w.len() != graphs.len()) {
                    return Err(crate::unsupported_reason!(
                        "invalid_argument",
                        "a candidate set names at least one graph and one weight per graph"
                    ));
                }
                let normalized = weights.as_deref().map(normalize).transpose()?;
                for (position, graph) in graphs.iter().enumerate() {
                    if is_class(graph) {
                        return Err(crate::unsupported_reason!(
                            "invalid_argument",
                            "a candidate set lists single graphs; declare an equivalence class \
                             as one graph"
                        ));
                    }
                    let weight = normalized.as_ref().map(|weights| weights[position]);
                    graph_members(graph, weight, pair, ctx, &mut raw)?;
                }
                normalized
            }
            Form::Posterior { posterior, names } => {
                posterior_members(posterior, names.len(), pair, ctx, &mut raw)?;
                Some(Vec::new())
            }
            Form::Cyclic { graph, premises, .. } => {
                let member = MemberGraph::Cyclic(graph.clone());
                let (status, identification) =
                    identify_cyclic_member(graph, pair, *premises, cyclic, ctx)?;
                raw.push(member, None, status, identification, pair);
                None
            }
        };
        Ok(StructureMembers {
            form: self.form(),
            identity: self.identity()?,
            weighted: weights.is_some(),
            members: raw.members,
            truncated: raw.truncated,
            cyclic,
        })
    }
}

impl From<AcceptedGraph> for DeclaredStructure {
    fn from(graph: AcceptedGraph) -> Self {
        Self::graph(graph)
    }
}

impl From<&AcceptedGraph> for DeclaredStructure {
    fn from(graph: &AcceptedGraph) -> Self {
        Self::graph(graph.clone())
    }
}

impl From<&Self> for DeclaredStructure {
    fn from(structure: &Self) -> Self {
        structure.clone()
    }
}

#[derive(Clone, Copy)]
struct Pair {
    treatment: usize,
    outcome: usize,
}

#[derive(Default)]
struct Raw {
    members: Vec<StructureMember>,
    truncated: bool,
}

impl Raw {
    fn push(
        &mut self,
        graph: MemberGraph,
        weight: Option<f64>,
        status: MemberStatus,
        identification: Option<IdentificationStatus>,
        pair: Pair,
    ) {
        if self.members.len() >= MAX_MEMBERS {
            self.truncated = true;
            return;
        }
        let relation = relation(&graph, pair);
        let key = member_key(&graph);
        self.members.push(StructureMember { graph, weight, relation, status, identification, key });
    }
}

fn normalize(weights: &[f64]) -> Result<Vec<f64>, CausalError> {
    let total: f64 = weights.iter().sum();
    if weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
        || !total.is_finite()
        || total <= 0.0
    {
        return Err(crate::unsupported_reason!(
            "invalid_argument",
            "member weights are nonnegative, finite and carry positive mass"
        ));
    }
    Ok(weights.iter().map(|weight| weight / total).collect())
}

fn is_class(graph: &AcceptedGraph) -> bool {
    matches!(
        graph.graph_ref(),
        GraphRef::Cpdag(_)
            | GraphRef::Pag(_)
            | GraphRef::TemporalCpdag(_)
            | GraphRef::TemporalPag(_)
    )
}

fn limits() -> SearchLimits {
    SearchLimits { operations: DEFAULT_COMPLETION_OPERATIONS, depth: usize::MAX }
}

fn graph_error(error: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!("invalid_argument", "{error}")
}

/// The members of one accepted graph, each identified.
fn graph_members(
    graph: &AcceptedGraph,
    weight: Option<f64>,
    pair: Pair,
    ctx: &ExecutionContext,
    raw: &mut Raw,
) -> Result<(), CausalError> {
    match graph.graph_ref() {
        GraphRef::Dag(dag) => single(MemberGraph::Dag(dag.clone()), weight, pair, ctx, raw),
        GraphRef::Admg(admg) => single(MemberGraph::Admg(admg.clone()), weight, pair, ctx, raw),
        GraphRef::TemporalDag(dag) => {
            single(MemberGraph::TemporalDag(dag.clone()), weight, pair, ctx, raw)
        }
        GraphRef::TemporalAdmg(admg) => {
            single(MemberGraph::TemporalAdmg(admg.clone()), weight, pair, ctx, raw)
        }
        GraphRef::Cpdag(cpdag) => {
            let mut sampler =
                CpdagCompletionSampler::with_budget(cpdag.clone(), MAX_MEMBERS, limits(), ctx)
                    .map_err(graph_error)?;
            for completion in sampler.by_ref() {
                poll(ctx)?;
                single(MemberGraph::Dag(completion.graph), weight, pair, ctx, raw)?;
            }
            raw.truncated |= sampler.hit_cap();
            Ok(())
        }
        GraphRef::Pag(pag) => {
            let mut sampler = CompletionSampler::with_budget(pag, MAX_MEMBERS, limits(), ctx)
                .map_err(graph_error)?;
            for completion in sampler.by_ref() {
                poll(ctx)?;
                match mag_to_admg(&completion.graph) {
                    Some(admg) => single(MemberGraph::Admg(admg), weight, pair, ctx, raw)?,
                    None => raw.truncated = true,
                }
            }
            raw.truncated |= sampler.hit_cap();
            Ok(())
        }
        GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_) => {
            temporal_class_members(graph, weight, pair, ctx, raw)
        }
        GraphRef::RandomizedTrial => Err(crate::unsupported_reason!(
            "invalid_argument",
            "a randomized-trial contract carries no causal graph to enumerate"
        )),
    }
}

/// A temporal class: the completions and per-completion identification of its class envelope.
fn temporal_class_members(
    graph: &AcceptedGraph,
    weight: Option<f64>,
    pair: Pair,
    ctx: &ExecutionContext,
    raw: &mut Raw,
) -> Result<(), CausalError> {
    let identification =
        identify_study_within(graph, &temporal_query(pair), None, StructureSource::Accepted, ctx)?;
    let Identification::TemporalEnvelope { envelope, .. } = identification else {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a temporal class identifies through its class envelope"
        ));
    };
    raw.truncated |= envelope.envelope.truncated_completions > 0;
    for case in envelope.envelope.cases {
        let status = case.result.status;
        let member = match case.graph {
            TemporalCompletionGraph::Dag(dag) => MemberGraph::TemporalDag(dag),
            TemporalCompletionGraph::Mag(mag) => MemberGraph::TemporalMag(mag),
        };
        raw.push(member, weight, member_status(status), Some(status), pair);
    }
    Ok(())
}

/// The members of a DAG or ADMG posterior: each distinct atom with its combined mass.
fn posterior_members(
    posterior: &GraphPosterior,
    n_names: usize,
    pair: Pair,
    ctx: &ExecutionContext,
    raw: &mut Raw,
) -> Result<(), CausalError> {
    if posterior.n_vars() != n_names {
        return Err(crate::unsupported_reason!(
            "schema_mismatch",
            "a graph posterior names one variable per posterior node"
        ));
    }
    if posterior.lag_masks().is_some() || posterior.max_lag().is_some() {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a lagged structure posterior is not a posterior over static members"
        ));
    }
    let admg = match posterior.atom_kind() {
        GraphPosteriorAtomKind::Dag => false,
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => true,
        _ => {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                "a graph posterior's members are DAG or ADMG atoms; declare a class as one graph"
            ));
        }
    };
    let mut distinct: Vec<(&AdjacencyMask, f64)> = Vec::new();
    for (mask, weight) in posterior.atoms().iter().zip(posterior.weights().iter()) {
        match distinct.iter_mut().find(|(seen, _)| *seen == mask) {
            Some((_, total)) => *total += *weight,
            None => distinct.push((mask, *weight)),
        }
    }
    let masses: Vec<f64> = distinct.iter().map(|(_, weight)| *weight).collect();
    let normalized = normalize(&masses)?;
    for ((mask, _), weight) in distinct.into_iter().zip(normalized) {
        poll(ctx)?;
        let member = if admg {
            MemberGraph::Admg(admg_from_adjacency_mask(mask, n_names).map_err(graph_error)?)
        } else {
            MemberGraph::Dag(dag_from_adjacency_mask(mask, n_names).map_err(graph_error)?)
        };
        single(member, Some(weight), pair, ctx, raw)?;
    }
    Ok(())
}

/// Identify one single-graph member and record it.
fn single(
    member: MemberGraph,
    weight: Option<f64>,
    pair: Pair,
    ctx: &ExecutionContext,
    raw: &mut Raw,
) -> Result<(), CausalError> {
    let (status, identification) = identify_member(&member, pair, ctx)?;
    raw.push(member, weight, status, identification, pair);
    Ok(())
}

/// The identification `identify_structure` runs, on one member.
fn identify_member(
    member: &MemberGraph,
    pair: Pair,
    ctx: &ExecutionContext,
) -> Result<(MemberStatus, Option<IdentificationStatus>), CausalError> {
    let (graph, query, strategy) = match member {
        MemberGraph::Dag(dag) => (
            AcceptedGraph::dag(dag.clone()),
            static_query(dag.nodes(), pair),
            Some(IdentifierId::GeneralId),
        ),
        MemberGraph::Admg(admg) => (
            AcceptedGraph::admg(admg.clone()),
            static_query(admg.nodes(), pair),
            Some(IdentifierId::GeneralId),
        ),
        MemberGraph::TemporalDag(dag) => {
            (AcceptedGraph::temporal_dag(dag.clone()), temporal_query(pair), None)
        }
        MemberGraph::TemporalMag(mag) => {
            (AcceptedGraph::temporal_pag(mag.clone()), temporal_query(pair), None)
        }
        MemberGraph::TemporalAdmg(admg) => (
            AcceptedGraph::temporal_admg(admg.clone()),
            static_query(&[], pair),
            Some(IdentifierId::GeneralId),
        ),
        MemberGraph::Cyclic(_) => {
            return Err(crate::unsupported_reason!(
                "invalid_argument",
                "a cyclic member is identified under its declared premises"
            ));
        }
    };
    identified(identify_study_within(&graph, &query, strategy, StructureSource::Explicit, ctx))
}

/// Read an identification outcome as a member status; only an identification refusal makes a
/// member unidentified, and a capped search leaves it undecided.
fn identified(
    outcome: Result<Identification, CausalError>,
) -> Result<(MemberStatus, Option<IdentificationStatus>), CausalError> {
    match outcome {
        Ok(identification) => {
            let status = identification.status();
            Ok((member_status(status), Some(status)))
        }
        Err(CausalError::NotIdentified { status, search_capped: true, .. }) => {
            Ok((MemberStatus::Undecided, Some(status)))
        }
        Err(error) if error.is_identification_refusal() => Ok((MemberStatus::NotIdentified, None)),
        Err(error) => Err(error),
    }
}

fn member_status(status: IdentificationStatus) -> MemberStatus {
    if crate::analysis::execute::identification_status_ok_for_case(status) {
        MemberStatus::Identified
    } else {
        MemberStatus::NotIdentified
    }
}

/// A cyclic member under the requested reading.
fn identify_cyclic_member(
    graph: &CyclicMixedGraph,
    pair: Pair,
    premises: CyclicIdPremises,
    reading: CyclicReading,
    ctx: &ExecutionContext,
) -> Result<(MemberStatus, Option<IdentificationStatus>), CausalError> {
    match reading {
        CyclicReading::Equilibrium => {
            let identification = identify_cyclic(
                graph,
                &[pair.outcome],
                &[pair.treatment],
                premises,
                true,
                CYCLIC_ID_STEPS,
                ctx,
            )
            .map_err(|error| match error {
                antecedent_identify::cyclic_id::CyclicIdError::Resource(message) => {
                    CausalError::Resource { message }
                }
                antecedent_identify::cyclic_id::CyclicIdError::Invalid(message) => {
                    crate::compile_reason!("invalid_argument", "{message}")
                }
            })?;
            Ok((
                match identification.status {
                    CyclicIdStatus::Identified => MemberStatus::Identified,
                    CyclicIdStatus::NotIdentifiedByProcedure => MemberStatus::NotIdentified,
                    CyclicIdStatus::BudgetExhausted => MemberStatus::Undecided,
                },
                None,
            ))
        }
        CyclicReading::Lagged => {
            // A latent confounder has no place in the one-step unrolling.
            if !graph.bidirected_edge_pairs_sorted().is_empty() {
                return Ok((MemberStatus::NotIdentified, None));
            }
            let unrolled = unroll(graph)?;
            identified(identify_study_within(
                &AcceptedGraph::temporal_dag(unrolled),
                &temporal_query(pair),
                None,
                StructureSource::Explicit,
                ctx,
            ))
        }
    }
}

/// Each directed edge `a -> b` of a cyclic graph as `a@t-1 -> b@t`, over one lagged and one
/// contemporaneous node per variable.
fn unroll(graph: &CyclicMixedGraph) -> Result<TemporalDag, CausalError> {
    let n = graph.node_count();
    let mut unrolled = TemporalDag::empty();
    let mut past = Vec::with_capacity(n);
    let mut present = Vec::with_capacity(n);
    for index in 0..n {
        let variable = variable_id(index);
        past.push(
            unrolled.add_lagged(variable, Lag::historical(NonZeroU32::MIN)).map_err(graph_error)?,
        );
        present.push(unrolled.add_lagged(variable, Lag::CONTEMPORANEOUS).map_err(graph_error)?);
    }
    for (from, to) in graph.directed_edge_pairs_sorted() {
        unrolled.insert_directed(past[from as usize], present[to as usize]).map_err(graph_error)?;
    }
    Ok(unrolled)
}

fn variable_id(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).unwrap_or(u32::MAX))
}

/// The static average effect of the treatment on the outcome, on the member's own labels.
fn static_query(nodes: &[NodeRef], pair: Pair) -> CausalQuery {
    let label = |index: usize| match nodes.get(index) {
        Some(NodeRef::Static(variable)) => *variable,
        _ => variable_id(index),
    };
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        label(pair.treatment),
        label(pair.outcome),
    ))
}

/// A one-step pulse of the treatment, read on the outcome one step later.
fn temporal_query(pair: Pair) -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
        variable_id(pair.treatment),
        variable_id(pair.outcome),
        1.0,
    ))
}

fn poll(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::error::STAGE_IDENTIFY });
    }
    Ok(())
}

fn dense(index: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(index).unwrap_or(u32::MAX))
}

/// The direct relation between the treatment and the outcome in `graph`.
fn relation(graph: &MemberGraph, pair: Pair) -> DirectRelation {
    let (t, o) = (dense(pair.treatment), dense(pair.outcome));
    match graph {
        MemberGraph::Dag(dag) => DirectRelation {
            forward: dag.parents(o).contains(&t),
            reverse: dag.parents(t).contains(&o),
            bidirected: false,
        },
        MemberGraph::Admg(admg) => DirectRelation {
            forward: admg.parents(o).contains(&t),
            reverse: admg.parents(t).contains(&o),
            bidirected: admg.bidirected_neighbors(t).contains(&o),
        },
        MemberGraph::Cyclic(cyclic) => DirectRelation {
            forward: cyclic.parents(o).contains(&t),
            reverse: cyclic.parents(t).contains(&o),
            bidirected: cyclic.bidirected_neighbors(t).contains(&o),
        },
        MemberGraph::TemporalDag(dag) => scan(dag.nodes(), dag.edges(), pair),
        MemberGraph::TemporalMag(mag) => scan(mag.nodes(), mag.edges(), pair),
        MemberGraph::TemporalAdmg(admg) => scan(admg.nodes(), admg.edges(), pair),
    }
}

/// The direct relation over the variables of a temporal graph, at any lag.
fn scan(
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
    pair: Pair,
) -> DirectRelation {
    let mut relation = DirectRelation::default();
    for edge in edges {
        let (Some(a), Some(b)) = (variable(nodes, edge.a), variable(nodes, edge.b)) else {
            continue;
        };
        let joins = (a == pair.treatment && b == pair.outcome)
            || (a == pair.outcome && b == pair.treatment);
        if !joins {
            continue;
        }
        relation.bidirected |= edge.is_bidirected();
        if let Some((parent, child)) = edge.parent_child() {
            let (parent, child) = (variable(nodes, parent), variable(nodes, child));
            relation.forward |= parent == Some(pair.treatment) && child == Some(pair.outcome);
            relation.reverse |= parent == Some(pair.outcome) && child == Some(pair.treatment);
        }
    }
    relation
}

fn variable(nodes: &[NodeRef], id: DenseNodeId) -> Option<usize> {
    let variable = match nodes.get(id.as_usize())? {
        NodeRef::Static(variable)
        | NodeRef::Lagged { variable, .. }
        | NodeRef::Unfolded { variable, .. }
        | NodeRef::Context { variable, .. } => variable,
    };
    usize::try_from(variable.raw()).ok()
}

/// A canonical graph identity wire, or the reason it could not be encoded.
struct GraphWire(Result<GraphIdentityWire, String>);

impl Canonical for GraphWire {
    fn encode(&self, out: &mut antecedent_core::CanonicalWriter) {
        match &self.0 {
            Ok(wire) => Wire(wire).encode(out),
            Err(error) => {
                out.text("unencodable").text(error);
            }
        }
    }
}

/// An accepted graph's canonical identity and its node labels.
struct AcceptedIdentity {
    wire: GraphIdentityWire,
    nodes: Vec<NodeRef>,
}

impl AcceptedIdentity {
    fn of(graph: &AcceptedGraph) -> Result<Self, CausalError> {
        let nodes = match graph.graph_ref() {
            GraphRef::Dag(g) => g.nodes().to_vec(),
            GraphRef::Admg(g) => g.nodes().to_vec(),
            GraphRef::Cpdag(g) => g.nodes().to_vec(),
            GraphRef::Pag(g) => g.nodes().to_vec(),
            GraphRef::TemporalDag(g) => g.nodes().to_vec(),
            GraphRef::TemporalCpdag(g) => g.nodes().to_vec(),
            GraphRef::TemporalPag(g) => g.nodes().to_vec(),
            GraphRef::TemporalAdmg(g) => g.nodes().to_vec(),
            GraphRef::RandomizedTrial => Vec::new(),
        };
        Ok(Self { wire: crate::analysis::contract::accepted_graph_identity(graph)?, nodes })
    }
}

impl Canonical for AcceptedIdentity {
    fn encode(&self, out: &mut antecedent_core::CanonicalWriter) {
        Wire(&self.wire).encode(out);
        self.nodes.encode(out);
    }
}

/// A cyclic graph's sorted directed and bidirected edges and its node labels.
struct CyclicIdentity<'a>(&'a CyclicMixedGraph);

impl Canonical for CyclicIdentity<'_> {
    fn encode(&self, out: &mut antecedent_core::CanonicalWriter) {
        let directed = self.0.directed_edge_pairs_sorted();
        let bidirected = self.0.bidirected_edge_pairs_sorted();
        out.word(directed.len() as u64);
        for (from, to) in directed {
            out.word(u64::from(from)).word(u64::from(to));
        }
        out.word(bidirected.len() as u64);
        for (a, b) in bidirected {
            out.word(u64::from(a)).word(u64::from(b));
        }
        self.0.nodes().encode(out);
    }
}

/// Canonical identity of one member structure with its node labels.
fn member_key(graph: &MemberGraph) -> [u8; 32] {
    use antecedent_io_core::identity::{
        admg_identity, dag_identity, temporal_admg_identity, temporal_dag_identity,
        temporal_pag_identity,
    };
    let mut declaration = Declaration::new("antecedent.structure_members.member.v1");
    match graph {
        MemberGraph::Dag(g) => {
            declaration
                .field("dag", &GraphWire(dag_identity(g).map_err(|error| error.to_string())));
            declaration.field("nodes", g.nodes());
        }
        MemberGraph::Admg(g) => {
            declaration
                .field("admg", &GraphWire(admg_identity(g).map_err(|error| error.to_string())));
            declaration.field("nodes", g.nodes());
        }
        MemberGraph::TemporalDag(g) => {
            declaration.field(
                "temporal_dag",
                &GraphWire(temporal_dag_identity(g).map_err(|error| error.to_string())),
            );
            declaration.field("nodes", g.nodes());
        }
        MemberGraph::TemporalMag(g) => {
            declaration.field(
                "temporal_mag",
                &GraphWire(temporal_pag_identity(g).map_err(|error| error.to_string())),
            );
            declaration.field("nodes", g.nodes());
        }
        MemberGraph::TemporalAdmg(g) => {
            declaration.field(
                "temporal_admg",
                &GraphWire(temporal_admg_identity(g).map_err(|error| error.to_string())),
            );
            declaration.field("nodes", g.nodes());
        }
        MemberGraph::Cyclic(g) => {
            declaration.field("cyclic", &CyclicIdentity(g));
        }
    }
    declaration.finish()
}

#[cfg(test)]
#[path = "structure_members_tests.rs"]
mod tests;
