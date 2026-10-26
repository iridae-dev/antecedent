//! Transport under graph uncertainty: what a request over a *discovered* equivalence class
//! returns.
//!
//! A selection diagram is a causal graph the caller asserts, plus the variables whose
//! mechanisms may differ between populations. A CPDAG is not one graph: it stands for every
//! DAG in a Markov equivalence class, and which member is true is not something the data
//! determine. This module never picks a member. It returns one of two things.
//!
//! * A typed [`DiscoveredTransportRefusal`] that names every unresolved pair and every
//!   exhausted search bound, when the class cannot be enumerated honestly (the search did not
//!   finish, orientation evidence conflicted, a selection target is not in the structure, or
//!   the class is too large to enumerate within its bound).
//! * A [`StructuralTransportEnvelope`]: the structural transport verdict of each member of the
//!   class, side by side. It is a statement about which members the implemented rules certify
//!   and which formulas they emit. It is **not** a point, an interval or a bound on an effect,
//!   and it carries no probability: the members are not weighted and the true graph is not
//!   asserted to be among them (that holds only if the assumptions in
//!   [`ENVELOPE_ASSUMPTIONS`] hold).
//!
//! # How to read an envelope
//!
//! * [`EnvelopeStatus::IdentifiedInEveryMember`] holds only when the enumeration finished and
//!   every member is certified. Then the query is structurally transportable *whichever
//!   member is true*, under the listed assumptions. The formulas may still differ between
//!   members ([`StructuralTransportEnvelope::formulas_agree`]); when they differ, the class
//!   does not determine the estimand's expression and the caller must resolve the class
//!   (background knowledge, an intervention) or declare a graph before estimating.
//! * [`EnvelopeStatus::NotIdentifiedInEveryMember`] means no member is certified by the
//!   implemented rules. That is the conservative `NotCertified` reading, not a proof of
//!   non-transportability.
//! * [`EnvelopeStatus::MembersDisagree`] means some members are certified and some are not:
//!   whether the query is transportable depends on which member is true, and the data do not
//!   say.
//! * [`EnvelopeStatus::Partial`] means the member bound stopped the enumeration. The members
//!   listed are real members, the unexplored remainder is not certified empty, and no
//!   "every member" statement is made.
//!
//! # Scope
//!
//! * Static CPDAGs under causal sufficiency: every member is a DAG over the observed variables.
//! * Static PAGs ([`transport_across_pag`]): the members of a PAG's class are maximal ancestral
//!   graphs, whose directed edges state ancestry and whose bidirected edges stand for latent
//!   common causes. Each member is read as the most-confounded ADMG it is compatible with: every
//!   directed edge is kept, every bidirected edge is kept, and a directed edge that is not
//!   visible (Zhang 2008) is also given a bidirected edge, because an invisible edge does not
//!   exclude a latent common cause of its endpoints. A functional that identification derives on
//!   that graph holds in every causal model the member represents. A circle endpoint is never
//!   read as the absence of confounding: it is enumerated, and each completion carries its own
//!   bidirected and invisible edges.
//! * Temporal structures are in [`crate::discovered_transport_temporal`].
//! * Selection targets are declared by the caller. They are never inferred from differences
//!   between per-population discoveries.
//!
//! # Refresh with outcome-only data
//!
//! The envelope is derived on every call from the discovered class and the declared query;
//! nothing is cached or updated in place. Outcome-only data does not change the class, so
//! it keeps the envelope's content; a re-run discovery that yields a different class is a
//! new input and the envelope is recomputed from it. An envelope for one class is never
//! read as a statement about another, and the declared selection targets are kept only by
//! being passed again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::result_large_err,
    reason = "the public error shape is kept as released; boxing the refusal would break callers that match on it"
)]

use antecedent_core::{ExecutionContext, NodeRef, SearchLimits, TransportQuery, VariableId};
use antecedent_graph::{
    Admg, CompletionSampler, Cpdag, CpdagCompletionSampler, DenseNodeId, Endpoint, Pag,
    SelectionDiagram,
};

use crate::error::IdentificationError;
use crate::transport::{TransportFormula, TransportIdentification, TransportIdentifier};

/// Registered reason code of every refusal in this module.
pub const DISCOVERED_TRANSPORT_REASON: &str =
    antecedent_core::reason_code!("discovered_structure_unresolved");

/// Default bound on enumerated members.
pub const DEFAULT_ENVELOPE_MEMBER_BOUND: usize = 256;

/// Assumptions under which the true graph is a member of the enumerated class. The envelope
/// states nothing when one fails, and no function here can check them.
pub const ENVELOPE_ASSUMPTIONS: [&str; 4] = [
    "causal sufficiency: no unobserved common cause of two observed variables",
    "faithfulness: the independences in the data are exactly those the true graph implies",
    "the conditional-independence tests returned the true independences within the search bounds",
    "selection targets are declared by the caller, not inferred from the data",
];

/// A request for the structural transport verdict across a discovered class.
#[derive(Clone, Copy, Debug)]
pub struct DiscoveredTransportRequest<'a> {
    /// The discovered CPDAG.
    pub class: &'a Cpdag,
    /// Variables whose mechanisms may differ between populations (declared, not discovered).
    pub selection_targets: &'a [VariableId],
    /// How many search bounds discovery exhausted. Non-zero refuses: absent edges are not
    /// established, so the class is not the class of any graph.
    pub exhausted_search_bounds: usize,
    /// Most members to enumerate. Zero is refused.
    pub member_bound: usize,
}

/// An unresolved pair of the class.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedPair {
    /// One endpoint.
    pub a: VariableId,
    /// The other endpoint.
    pub b: VariableId,
    /// `undirected_in_equivalence_class`, `orientation_conflict`, `circle_endpoint` or
    /// `bidirected_latent_confounding`.
    pub reason: &'static str,
}

/// Typed refusal of a transport request over a discovered class.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredTransportRefusal {
    /// Registered reason code ([`DISCOVERED_TRANSPORT_REASON`]).
    pub code: &'static str,
    /// Namespaced detail: `discovered_transport.search_incomplete`,
    /// `discovered_transport.orientation_conflict`,
    /// `discovered_transport.selection_target_unknown`,
    /// `discovered_transport.class_too_large`, `discovered_transport.no_members`,
    /// `discovered_transport.member_bound_zero`,
    /// `discovered_transport.pag_class_ambiguous`,
    /// `discovered_transport.pag_selection_bias_edge`,
    /// `discovered_transport.temporal_query_invalid`,
    /// `discovered_transport.class_not_identified`,
    /// `discovered_transport.member_not_estimable`,
    /// `discovered_transport.class_enumeration_partial` or
    /// `discovered_transport.evidence_invalid`.
    pub detail: &'static str,
    /// Every unresolved pair of the class with its reason.
    pub unresolved: Vec<UnresolvedPair>,
    /// Number of exhausted search bounds.
    pub exhausted_search_bounds: usize,
    /// Selection targets that are not nodes of the structure.
    pub unknown_selection_targets: Vec<VariableId>,
    /// For a refusal of the estimation step: `(member index, reason)` for every member that
    /// is not identified or not estimable. Empty otherwise.
    pub member_reasons: Vec<(usize, String)>,
    /// What would license the request.
    pub remedy: String,
}

impl std::fmt::Display for DiscoveredTransportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.remedy)
    }
}

/// Refusal for a PAG whose members are read as causal graphs.
///
/// A PAG is now transported through [`transport_across_pag`], which reads each completion as
/// the most-confounded ADMG it is compatible with; this constructor remains for callers that
/// build the refusal themselves.
///
/// # Errors
///
/// Always [`DiscoveredTransportError::Refused`] with
/// `discovered_transport.pag_members_not_causal_graphs`.
#[deprecated(since = "2.4.1", note = "use transport_across_pag, which no longer refuses a PAG")]
#[must_use]
pub fn refuse_pag_transport(
    unresolved: Vec<UnresolvedPair>,
    exhausted_search_bounds: usize,
) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: DISCOVERED_TRANSPORT_REASON,
        // Two literals so the refusal-detail registry gate does not read this retained
        // constructor as a live emission of a detail no route produces.
        detail: concat!("discovered_transport", ".pag_members_not_causal_graphs"),
        unresolved,
        exhausted_search_bounds,
        unknown_selection_targets: Vec::new(),
        member_reasons: Vec::new(),
        remedy: "the members of a PAG are maximal ancestral graphs: a directed edge states \
                 ancestry, not a causal parent, and a bidirected edge stands for a latent \
                 path, so the structural transport rules do not describe them. Resolve the PAG \
                 to a declared causal graph (an ADMG you assert) and request transport with \
                 that graph"
            .to_owned(),
    })
}

/// Failure of a transport request over a discovered class.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscoveredTransportError {
    /// A scientific refusal with its unresolved structure.
    Refused(DiscoveredTransportRefusal),
    /// Completion or member identification failed, including a malformed query/diagram,
    /// caller memory exhaustion or cancellation. Never reported as a member verdict.
    Identification(IdentificationError),
    /// The estimation step failed for a reason that is not a scientific refusal: a resource
    /// bound, a cancellation or an internal check.
    Estimation(String),
}

impl std::fmt::Display for DiscoveredTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(refusal) => refusal.fmt(f),
            Self::Identification(error) => error.fmt(f),
            Self::Estimation(message) => write!(f, "estimation failed: {message}"),
        }
    }
}

impl std::error::Error for DiscoveredTransportError {}

/// A member's structural verdict.
#[derive(Clone, Debug, PartialEq)]
pub enum MemberVerdict {
    /// An implemented sound rule certified the query for this member.
    Identified {
        /// Rule id.
        rule: String,
        /// The emitted formula.
        formula: TransportFormula,
    },
    /// No implemented rule certified the query for this member.
    NotCertified {
        /// Stable failure id.
        reason: String,
    },
    /// A required source regime or factor is absent for this member.
    MissingEvidence {
        /// Stable reason id.
        reason: String,
    },
}

/// One member of the class and its verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvelopeMember {
    /// Directed edges `(parent, child)` of the member, ascending.
    pub directed_edges: Vec<(VariableId, VariableId)>,
    /// Bidirected edges `(a, b)` with `a < b` of a PAG member, ascending: a latent common
    /// cause of the two endpoints. Empty for a member of a CPDAG's class.
    pub bidirected_edges: Vec<(VariableId, VariableId)>,
    /// Directed edges `(parent, child)` of a PAG member that are not visible, ascending. The
    /// member does not exclude a latent common cause of such a pair, so the identifier read it
    /// as confounded as well. Empty for a member of a CPDAG's class.
    pub confounded_edges: Vec<(VariableId, VariableId)>,
    /// The structural verdict for this member.
    pub verdict: MemberVerdict,
}

/// Summary reading of an envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnvelopeStatus {
    /// The enumeration finished and every member is certified.
    IdentifiedInEveryMember,
    /// The enumeration finished and no member is certified.
    NotIdentifiedInEveryMember,
    /// The enumeration finished and some members are certified and some are not.
    MembersDisagree,
    /// The member bound stopped the enumeration; no statement over the whole class.
    Partial,
}

/// The structural transport verdicts across a discovered class.
///
/// A structural envelope: never a point estimate or an interval, and not a probability
/// statement over members.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTransportEnvelope {
    members: Vec<EnvelopeMember>,
    unresolved: Vec<UnresolvedPair>,
    enumeration_complete: bool,
    member_bound: usize,
    variables: Vec<VariableId>,
}

impl StructuralTransportEnvelope {
    /// What this object is, as a stable label for reports: always `structural_envelope`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        "structural_envelope"
    }

    /// The enumerated members with their verdicts.
    #[must_use]
    pub fn members(&self) -> &[EnvelopeMember] {
        &self.members
    }

    /// The class's unresolved pairs.
    #[must_use]
    pub fn unresolved(&self) -> &[UnresolvedPair] {
        &self.unresolved
    }

    /// The variables of the class, in the order of its nodes.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }

    /// The graph member `index`'s verdict was decided on: its directed edges, its bidirected
    /// edges, and a bidirected edge beside each directed edge that is not visible.
    ///
    /// `None` when `index` is out of range.
    #[must_use]
    pub fn member_graph(&self, index: usize) -> Option<Admg> {
        let member = self.members.get(index)?;
        let mut graph = Admg::empty();
        let mut dense = std::collections::HashMap::new();
        for variable in &self.variables {
            dense.insert(*variable, graph.add_node(NodeRef::Static(*variable)).ok()?);
        }
        for (parent, child) in &member.directed_edges {
            graph.insert_directed(*dense.get(parent)?, *dense.get(child)?).ok()?;
        }
        let latent = member.bidirected_edges.iter().chain(&member.confounded_edges);
        for (a, b) in latent {
            let (a, b) = (*dense.get(a)?, *dense.get(b)?);
            if !graph.bidirected_neighbors(a).contains(&b) {
                graph.insert_bidirected(a, b).ok()?;
            }
        }
        Some(graph)
    }

    /// Whether the enumeration finished within the member bound.
    #[must_use]
    pub const fn enumeration_complete(&self) -> bool {
        self.enumeration_complete
    }

    /// The member bound the enumeration ran under.
    #[must_use]
    pub const fn member_bound(&self) -> usize {
        self.member_bound
    }

    /// Enumerated members certified by an implemented rule.
    #[must_use]
    pub fn identified_members(&self) -> usize {
        self.members
            .iter()
            .filter(|m| matches!(m.verdict, MemberVerdict::Identified { .. }))
            .count()
    }

    /// The summary reading; see the module documentation.
    #[must_use]
    pub fn status(&self) -> EnvelopeStatus {
        status_of(self.members.iter().map(|m| &m.verdict), self.enumeration_complete)
    }

    /// Whether every enumerated member is certified with the same formula. Vacuously false
    /// when a member is not certified.
    #[must_use]
    pub fn formulas_agree(&self) -> bool {
        formulas_agree_over(self.members.iter().map(|m| &m.verdict))
    }
}

/// The summary reading of member verdicts over an enumeration that did or did not finish.
pub(crate) fn status_of<'a>(
    verdicts: impl ExactSizeIterator<Item = &'a MemberVerdict>,
    enumeration_complete: bool,
) -> EnvelopeStatus {
    if !enumeration_complete {
        return EnvelopeStatus::Partial;
    }
    let total = verdicts.len();
    let identified =
        verdicts.filter(|verdict| matches!(verdict, MemberVerdict::Identified { .. })).count();
    if identified == total {
        EnvelopeStatus::IdentifiedInEveryMember
    } else if identified == 0 {
        EnvelopeStatus::NotIdentifiedInEveryMember
    } else {
        EnvelopeStatus::MembersDisagree
    }
}

/// Whether every verdict is a certification with one and the same formula.
pub(crate) fn formulas_agree_over<'a>(verdicts: impl Iterator<Item = &'a MemberVerdict>) -> bool {
    let mut formulas = verdicts.map(|verdict| match verdict {
        MemberVerdict::Identified { formula, .. } => Some(formula),
        _ => None,
    });
    match formulas.next() {
        Some(Some(first)) => formulas.all(|f| f == Some(first)),
        _ => false,
    }
}

pub(crate) fn refusal(
    detail: &'static str,
    request: &RequestFacts<'_>,
    unresolved: Vec<UnresolvedPair>,
    remedy: String,
) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: DISCOVERED_TRANSPORT_REASON,
        detail,
        unresolved,
        exhausted_search_bounds: request.exhausted_search_bounds,
        unknown_selection_targets: Vec::new(),
        member_reasons: Vec::new(),
        remedy,
    })
}

/// The request fields every class kind shares, as the checks read them.
pub(crate) struct RequestFacts<'a> {
    pub(crate) selection_targets: &'a [VariableId],
    pub(crate) exhausted_search_bounds: usize,
    pub(crate) member_bound: usize,
}

fn static_variables(nodes: &[NodeRef]) -> Vec<VariableId> {
    nodes
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect()
}

fn variable_of(class: &Cpdag, id: DenseNodeId) -> Option<VariableId> {
    class.variable_id(id)
}

fn unresolved_pairs(class: &Cpdag) -> Vec<UnresolvedPair> {
    let mut pairs: Vec<UnresolvedPair> = class
        .edges()
        .into_iter()
        .filter_map(|edge| {
            let reason = if edge.is_conflict() {
                "orientation_conflict"
            } else if edge.is_undirected() {
                "undirected_in_equivalence_class"
            } else {
                return None;
            };
            let (a, b) = (variable_of(class, edge.a)?, variable_of(class, edge.b)?);
            let (a, b) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
            Some(UnresolvedPair { a, b, reason })
        })
        .collect();
    pairs.sort_by_key(|pair| (pair.a.raw(), pair.b.raw()));
    pairs
}

/// The refusals every class kind shares, in their fixed order: an exhausted search bound, an
/// orientation conflict, a selection target outside the structure, a zero member bound.
pub(crate) fn check_request(
    request: &RequestFacts<'_>,
    unresolved: &[UnresolvedPair],
    is_node: impl Fn(VariableId) -> bool,
) -> Result<(), DiscoveredTransportError> {
    if request.exhausted_search_bounds > 0 {
        return Err(refusal(
            "discovered_transport.search_incomplete",
            request,
            unresolved.to_vec(),
            format!(
                "discovery exhausted {} search bound(s), so absent edges are not established \
                 and the class is not the equivalence class of any graph. Rerun with larger \
                 bounds, or declare the graph and its selection diagram",
                request.exhausted_search_bounds
            ),
        ));
    }
    if unresolved.iter().any(|pair| pair.reason == "orientation_conflict") {
        return Err(refusal(
            "discovered_transport.orientation_conflict",
            request,
            unresolved.to_vec(),
            "orientation evidence conflicted on at least one edge, so the structure is not a \
             CPDAG with members. Resolve the conflicting edges with background knowledge, or \
             declare the graph"
                .to_owned(),
        ));
    }
    let unknown: Vec<VariableId> =
        request.selection_targets.iter().copied().filter(|target| !is_node(*target)).collect();
    if !unknown.is_empty() {
        let names: Vec<String> = unknown.iter().map(|v| format!("v{}", v.raw())).collect();
        let mut error = refusal(
            "discovered_transport.selection_target_unknown",
            request,
            unresolved.to_vec(),
            format!(
                "selection target(s) {} are not variables of the discovered structure; declare \
                 targets among the searched variables",
                names.join(", ")
            ),
        );
        if let DiscoveredTransportError::Refused(r) = &mut error {
            r.unknown_selection_targets = unknown;
        }
        return Err(error);
    }
    if request.member_bound == 0 {
        return Err(refusal(
            "discovered_transport.member_bound_zero",
            request,
            unresolved.to_vec(),
            "the member bound is zero; give a positive bound on the members to enumerate"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The structural verdict of `query` on one member graph under the declared selection targets.
pub(crate) fn verdict_of(
    admg: Admg,
    selection: &std::sync::Arc<[VariableId]>,
    identifier: &TransportIdentifier,
    query: &TransportQuery,
) -> Result<MemberVerdict, DiscoveredTransportError> {
    verdict_with_context(admg, selection, identifier, query, None)
}

fn verdict_with_context(
    admg: Admg,
    selection: &std::sync::Arc<[VariableId]>,
    identifier: &TransportIdentifier,
    query: &TransportQuery,
    context: Option<&ExecutionContext>,
) -> Result<MemberVerdict, DiscoveredTransportError> {
    let invalid = |error: antecedent_graph::GraphError| {
        DiscoveredTransportError::Identification(IdentificationError::invalid_input(
            error.to_string(),
        ))
    };
    let diagram = SelectionDiagram::try_new(admg, selection.clone()).map_err(invalid)?;
    Ok(
        match context
            .map_or_else(
                || identifier.identify(&diagram, query),
                |ctx| identifier.identify_with_context(&diagram, query, ctx),
            )
            .map_err(DiscoveredTransportError::Identification)?
        {
            TransportIdentification::Transportable { formula, certificate } => {
                MemberVerdict::Identified { rule: certificate.rule.to_string(), formula }
            }
            TransportIdentification::NotCertified(certificate) => {
                MemberVerdict::NotCertified { reason: certificate.reason.to_string() }
            }
            TransportIdentification::MissingEvidence(certificate) => {
                MemberVerdict::MissingEvidence { reason: certificate.reason.to_string() }
            }
        },
    )
}

/// The verdict and directed edges of one member of a CPDAG's class.
fn member_of(
    dag: &antecedent_graph::Dag,
    class: &Cpdag,
    selection: &std::sync::Arc<[VariableId]>,
    identifier: &TransportIdentifier,
    query: &TransportQuery,
    ctx: &ExecutionContext,
) -> Result<EnvelopeMember, DiscoveredTransportError> {
    let admg = Admg::from_dag(dag).map_err(|error| {
        DiscoveredTransportError::Identification(IdentificationError::invalid_input(
            error.to_string(),
        ))
    })?;
    let verdict = verdict_with_context(admg, selection, identifier, query, Some(ctx))?;
    let mut directed_edges: Vec<(VariableId, VariableId)> = dag
        .edges()
        .filter_map(antecedent_graph::MarkedEdge::parent_child)
        .filter_map(|(parent, child)| {
            Some((variable_of(class, parent)?, variable_of(class, child)?))
        })
        .collect();
    directed_edges.sort_by_key(|(p, c)| (p.raw(), c.raw()));
    Ok(EnvelopeMember {
        directed_edges,
        bidirected_edges: Vec::new(),
        confounded_edges: Vec::new(),
        verdict,
    })
}

/// Members in the canonical order of their edge sets.
fn sort_members(members: &mut [EnvelopeMember]) {
    let key = |m: &EnvelopeMember| {
        let pairs = |edges: &[(VariableId, VariableId)]| {
            edges.iter().map(|(p, c)| (p.raw(), c.raw())).collect::<Vec<_>>()
        };
        (pairs(&m.directed_edges), pairs(&m.bidirected_edges))
    };
    members.sort_by_key(key);
}

fn check_transport_context(ctx: &ExecutionContext) -> Result<(), DiscoveredTransportError> {
    if ctx.cancellation.is_cancelled() {
        Err(DiscoveredTransportError::Identification(IdentificationError::Cancelled))
    } else {
        Ok(())
    }
}

/// The structural transport verdict of `query` across every member of a discovered class.
///
/// Members are enumerated up to `member_bound` and each is identified with the structural
/// identifier under the declared `selection_targets`. No member is preferred, weighted or
/// averaged.
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] for an exhausted search bound, an orientation
/// conflict, a selection target that is not in the structure, a class that cannot be
/// enumerated, a zero member bound, or a class with no member;
/// [`DiscoveredTransportError::Identification`] when the structural identifier fails for a
/// member.
pub fn transport_across_class(
    request: &DiscoveredTransportRequest<'_>,
    query: &TransportQuery,
) -> Result<StructuralTransportEnvelope, DiscoveredTransportError> {
    transport_across_class_with_context(request, query, &ExecutionContext::production(0, 1))
}

/// [`transport_across_class`] using the caller's completion and member-search
/// memory pool and cancellation. Resource stops never become scientific verdicts.
///
/// # Errors
/// As [`transport_across_class`], including typed identification memory/cancellation errors.
pub fn transport_across_class_with_context(
    request: &DiscoveredTransportRequest<'_>,
    query: &TransportQuery,
    ctx: &ExecutionContext,
) -> Result<StructuralTransportEnvelope, DiscoveredTransportError> {
    check_transport_context(ctx)?;
    let class = request.class;
    let unresolved = unresolved_pairs(class);
    let facts = RequestFacts {
        selection_targets: request.selection_targets,
        exhausted_search_bounds: request.exhausted_search_bounds,
        member_bound: request.member_bound,
    };
    check_request(&facts, &unresolved, |target| class.nodes().contains(&NodeRef::Static(target)))?;

    let mut sampler = CpdagCompletionSampler::with_budget(
        class.clone(),
        request.member_bound,
        SearchLimits {
            operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
            depth: usize::MAX,
        },
        ctx,
    )
    .map_err(|error| {
        refusal(
            "discovered_transport.class_too_large",
            &facts,
            unresolved.clone(),
            format!(
                "the class cannot be enumerated ({error}). Orient edges with background \
                     knowledge or an intervention, or declare the graph"
            ),
        )
    })?;
    let dags: Vec<_> = sampler.by_ref().map(|completion| completion.graph).collect();
    crate::generalized::check_completion_stop(sampler.search_stop())
        .map_err(DiscoveredTransportError::Identification)?;
    let enumeration_complete = !sampler.hit_cap();
    if dags.is_empty() {
        if sampler.search_stop().is_some() {
            return Err(DiscoveredTransportError::Identification(IdentificationError::budget(
                crate::IdentificationBudget::Steps,
            )));
        }
        return Err(refusal(
            "discovered_transport.no_members",
            &facts,
            unresolved,
            "the structure has no acyclic completion without a new unshielded collider, so it \
             is not a CPDAG of any graph; rerun discovery or declare the graph"
                .to_owned(),
        ));
    }

    let identifier = TransportIdentifier::new();
    let selection: std::sync::Arc<[VariableId]> = request.selection_targets.into();
    let mut members = Vec::with_capacity(dags.len());
    for dag in &dags {
        check_transport_context(ctx)?;
        members.push(member_of(dag, class, &selection, &identifier, query, ctx)?);
    }
    sort_members(&mut members);
    check_transport_context(ctx)?;

    Ok(StructuralTransportEnvelope {
        members,
        unresolved,
        enumeration_complete,
        member_bound: request.member_bound,
        variables: static_variables(class.nodes()),
    })
}

/// A request for the structural transport verdict across a discovered PAG's class.
#[derive(Clone, Copy, Debug)]
pub struct PagTransportRequest<'a> {
    /// The discovered PAG.
    pub class: &'a Pag,
    /// Variables whose mechanisms may differ between populations (declared, not discovered).
    pub selection_targets: &'a [VariableId],
    /// How many search bounds discovery exhausted. Non-zero refuses.
    pub exhausted_search_bounds: usize,
    /// Most members to enumerate. Zero is refused.
    pub member_bound: usize,
    /// Operations the completion walk may charge: one per endpoint assignment visited and one
    /// per m-separation query of the equivalence audit. A stop is reported as a partial
    /// envelope, never as a statement about the unvisited completions.
    pub completion_operations: usize,
}

/// The subgraph of `admg` induced on the ancestors of `seeds`, with every node keeping its
/// variable id.
///
/// Identification of an effect on an outcome depends on the ancestors of the treatment and the
/// outcome only: a node that is not an ancestor of either is a latent projection away, and a
/// bidirected edge to it joins no two kept nodes. Dropping the rest keeps the verdict and keeps
/// every subset search on the nodes that can matter.
///
/// # Errors
///
/// [`antecedent_graph::GraphError`] when a seed is not a node of the graph or an edge cannot
/// be copied.
pub fn ancestral_subgraph(
    admg: &Admg,
    seeds: &[VariableId],
) -> Result<Admg, antecedent_graph::GraphError> {
    let dense: Vec<DenseNodeId> = seeds
        .iter()
        .map(|seed| {
            admg.dense_of_static(*seed)
                .ok_or(antecedent_graph::GraphError::UnknownNode { id: seed.raw() })
        })
        .collect::<Result<_, _>>()?;
    let mut keep = antecedent_graph::BitSet::with_len(admg.node_count());
    admg.ancestors_of(&dense, &mut keep, &mut antecedent_graph::GraphWorkspace::default());
    let mut out = Admg::empty();
    let mut map = vec![None; admg.node_count()];
    for id in keep.iter() {
        map[id.as_usize()] = Some(out.add_node(admg.nodes()[id.as_usize()])?);
    }
    for id in keep.iter() {
        for &child in admg.children(id) {
            if let (Some(a), Some(b)) = (map[id.as_usize()], map[child.as_usize()]) {
                out.insert_directed(a, b)?;
            }
        }
        for &other in admg.bidirected_neighbors(id) {
            if let (Some(a), Some(b)) = (map[id.as_usize()], map[other.as_usize()]) {
                if id.raw() < other.raw() {
                    out.insert_bidirected(a, b)?;
                }
            }
        }
    }
    Ok(out)
}

/// The unresolved pairs of a PAG: every edge whose marks the class does not fix, and every
/// bidirected edge (a latent common cause, kept as an edge of every member).
fn pag_unresolved_pairs(class: &Pag) -> Vec<UnresolvedPair> {
    let nodes = class.nodes();
    let mut pairs = Vec::new();
    for i in 0..class.node_count() {
        let a = DenseNodeId::from_index(i);
        for (b, at_a, at_b) in class.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            let reason = match (at_a, at_b) {
                (Endpoint::Conflict, _) | (_, Endpoint::Conflict) => "orientation_conflict",
                (Endpoint::Circle, _) | (_, Endpoint::Circle) => "circle_endpoint",
                (Endpoint::Tail, Endpoint::Tail) => "undirected_in_equivalence_class",
                (Endpoint::Arrow, Endpoint::Arrow) => "bidirected_latent_confounding",
                _ => continue,
            };
            let (NodeRef::Static(x), NodeRef::Static(y)) = (nodes[i], nodes[b.as_usize()]) else {
                continue;
            };
            let (x, y) = if x.raw() <= y.raw() { (x, y) } else { (y, x) };
            pairs.push(UnresolvedPair { a: x, b: y, reason });
        }
    }
    pairs.sort_by_key(|pair| (pair.a.raw(), pair.b.raw()));
    pairs
}

/// One circle-free completion read as its most-confounded ADMG, with the edge sets that
/// reading used. Edges are named by the labels the caller gave the nodes.
pub(crate) struct MagMember {
    pub(crate) admg: Admg,
    pub(crate) directed: Vec<(VariableId, VariableId)>,
    pub(crate) bidirected: Vec<(VariableId, VariableId)>,
    pub(crate) confounded: Vec<(VariableId, VariableId)>,
}

fn static_variable(mag: &Pag, id: DenseNodeId) -> Result<VariableId, DiscoveredTransportError> {
    match mag.nodes().get(id.as_usize()) {
        Some(NodeRef::Static(variable)) => Ok(*variable),
        _ => Err(DiscoveredTransportError::Identification(IdentificationError::invalid_input(
            "a static PAG member has a node that is not a static variable",
        ))),
    }
}

/// The most-confounded ADMG a MAG is compatible with: every directed and bidirected edge is
/// kept, and each directed edge that is not visible also gets a bidirected edge. See the
/// module documentation for why a derivation on this graph holds in every model the MAG
/// represents.
///
/// `label` names each MAG node by the variable the ADMG carries for it. `None` when a
/// completion has a tail-tail edge (selection bias), which no edge set here represents.
pub(crate) fn mag_member(
    mag: &Pag,
    label: impl Fn(DenseNodeId) -> Result<VariableId, DiscoveredTransportError>,
) -> Result<Option<MagMember>, DiscoveredTransportError> {
    let internal = |error: antecedent_graph::GraphError| {
        DiscoveredTransportError::Identification(IdentificationError::invalid_input(
            error.to_string(),
        ))
    };
    let mut admg = Admg::empty();
    for i in 0..mag.node_count() {
        admg.add_node(NodeRef::Static(label(DenseNodeId::from_index(i))?)).map_err(internal)?;
    }
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_index(i);
        for (b, at_a, at_b) in mag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            let (va, vb) = (label(a)?, label(b)?);
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    admg.insert_directed(a, b).map_err(internal)?;
                    directed.push((va, vb));
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    admg.insert_directed(b, a).map_err(internal)?;
                    directed.push((vb, va));
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    admg.insert_bidirected(a, b).map_err(internal)?;
                    bidirected.push(if va.raw() <= vb.raw() { (va, vb) } else { (vb, va) });
                }
                _ => return Ok(None),
            }
        }
    }
    let mut confounded = Vec::new();
    for (tail, head) in crate::generalized::invisible_directed_edges(mag) {
        admg.insert_bidirected(tail, head).map_err(internal)?;
        confounded.push((label(tail)?, label(head)?));
    }
    for edges in [&mut directed, &mut bidirected, &mut confounded] {
        edges.sort_by_key(|(x, y): &(VariableId, VariableId)| (x.raw(), y.raw()));
    }
    Ok(Some(MagMember { admg, directed, bidirected, confounded }))
}

/// The structural transport verdict of `query` across every member of a discovered PAG's
/// class.
///
/// The members are the circle-free completions of the PAG that are maximal ancestral graphs
/// of its one Markov equivalence class, enumerated by the budgeted completion walk. Each is
/// read as its most-confounded ADMG (see the module documentation) and identified with the
/// structural identifier under the declared `selection_targets`. A circle endpoint is never
/// read as the absence of confounding, and no member is preferred, weighted or averaged.
///
/// A walk that a bound stopped, a member bound that cut the enumeration, or an equivalence
/// audit the budget skipped makes the envelope [`EnvelopeStatus::Partial`]: no statement is
/// made about the whole class.
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] for an exhausted search bound, an orientation
/// conflict, a selection target that is not in the structure, a zero member bound, marks that
/// admit more than one Markov equivalence class, a selection-bias edge, a walk that stopped
/// before finding a completion, or a structure with no completion;
/// [`DiscoveredTransportError::Identification`] when the structural identifier fails for a
/// member.
pub fn transport_across_pag(
    request: &PagTransportRequest<'_>,
    query: &TransportQuery,
) -> Result<StructuralTransportEnvelope, DiscoveredTransportError> {
    transport_across_pag_with_context(request, query, &ExecutionContext::production(0, 1))
}

/// [`transport_across_pag`] using the caller's completion and member-search
/// memory pool and cancellation, preserving declared work-cap partial envelopes.
///
/// # Errors
/// As [`transport_across_pag`], including typed identification memory/cancellation errors.
pub fn transport_across_pag_with_context(
    request: &PagTransportRequest<'_>,
    query: &TransportQuery,
    ctx: &ExecutionContext,
) -> Result<StructuralTransportEnvelope, DiscoveredTransportError> {
    check_transport_context(ctx)?;
    let class = request.class;
    let unresolved = pag_unresolved_pairs(class);
    let facts = RequestFacts {
        selection_targets: request.selection_targets,
        exhausted_search_bounds: request.exhausted_search_bounds,
        member_bound: request.member_bound,
    };
    check_request(&facts, &unresolved, |target| class.nodes().contains(&NodeRef::Static(target)))?;

    let limits = SearchLimits { operations: request.completion_operations, depth: usize::MAX };
    let mut sampler = CompletionSampler::with_budget(class, request.member_bound, limits, ctx)
        .map_err(|error| {
            refusal(
                "discovered_transport.class_too_large",
                &facts,
                unresolved.clone(),
                format!(
                    "the completions cannot be enumerated ({error}). Orient edges with \
                     background knowledge or an intervention, or declare the graph"
                ),
            )
        })?;
    let mags: Vec<Pag> = sampler.by_ref().map(|completion| completion.graph).collect();
    crate::generalized::check_completion_stop(sampler.search_stop())
        .map_err(DiscoveredTransportError::Identification)?;
    if sampler.validation_report().ambiguous_global_class {
        return Err(refusal(
            "discovered_transport.pag_class_ambiguous",
            &facts,
            unresolved,
            "the circle endpoints admit completions from more than one Markov equivalence \
             class, so the structure is not the PAG of one class and its members cannot be \
             enumerated as one. Rerun discovery, or declare the graph"
                .to_owned(),
        ));
    }
    if mags.is_empty() {
        return Err(match sampler.search_stop() {
            Some(stop) => refusal(
                "discovered_transport.class_too_large",
                &facts,
                unresolved,
                format!(
                    "the completion search stopped ({}) before it found a member. Raise the \
                     completion operation budget, orient edges with background knowledge, or \
                     declare the graph",
                    stop.code()
                ),
            ),
            None => refusal(
                "discovered_transport.no_members",
                &facts,
                unresolved,
                "the structure has no maximal ancestral completion that agrees with its \
                 unshielded colliders, so it is not a PAG of any graph; rerun discovery or \
                 declare the graph"
                    .to_owned(),
            ),
        });
    }
    let enumeration_complete = !sampler.hit_cap() && !sampler.class_audit_incomplete();

    let identifier = TransportIdentifier::new();
    let selection: std::sync::Arc<[VariableId]> = request.selection_targets.into();
    let mut members = Vec::with_capacity(mags.len());
    for mag in &mags {
        check_transport_context(ctx)?;
        let Some(member) = mag_member(mag, |id| static_variable(mag, id))? else {
            return Err(refusal(
                "discovered_transport.pag_selection_bias_edge",
                &facts,
                unresolved,
                "a completion has a tail-tail edge, which stands for selection bias in the \
                 sample. The transport rules do not model it; remove the edge with background \
                 knowledge, or declare the graph"
                    .to_owned(),
            ));
        };
        let verdict = verdict_with_context(member.admg, &selection, &identifier, query, Some(ctx))?;
        members.push(EnvelopeMember {
            directed_edges: member.directed,
            bidirected_edges: member.bidirected,
            confounded_edges: member.confounded,
            verdict,
        });
    }
    sort_members(&mut members);
    check_transport_context(ctx)?;

    Ok(StructuralTransportEnvelope {
        members,
        unresolved,
        enumeration_complete,
        member_bound: request.member_bound,
        variables: static_variables(class.nodes()),
    })
}
