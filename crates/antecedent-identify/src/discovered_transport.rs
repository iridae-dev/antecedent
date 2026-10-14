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
//! * Static CPDAGs under causal sufficiency (every member is a DAG over the observed
//!   variables). A PAG's members are ancestral graphs whose directed edges are ancestral, not
//!   causal-parent relations, so the structural rules here would be applied to a graph they do
//!   not describe: that route is refused (`pag_members_not_causal_graphs`). A temporal
//!   structure is outside this module.
//! * Selection targets are declared by the caller. They are never inferred from differences
//!   between per-population discoveries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{NodeRef, TransportQuery, VariableId};
use antecedent_graph::{Admg, Cpdag, CpdagCompletionSampler, DenseNodeId, SelectionDiagram};

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
    /// `undirected_in_equivalence_class` or `orientation_conflict`.
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
    /// `discovered_transport.member_bound_zero` or
    /// `discovered_transport.pag_members_not_causal_graphs`.
    pub detail: &'static str,
    /// Every unresolved pair of the class with its reason.
    pub unresolved: Vec<UnresolvedPair>,
    /// Number of exhausted search bounds.
    pub exhausted_search_bounds: usize,
    /// Selection targets that are not nodes of the structure.
    pub unknown_selection_targets: Vec<VariableId>,
    /// What would license the request.
    pub remedy: String,
}

impl std::fmt::Display for DiscoveredTransportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.remedy)
    }
}

/// Failure of a transport request over a discovered class.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscoveredTransportError {
    /// A scientific refusal with its unresolved structure.
    Refused(DiscoveredTransportRefusal),
    /// The structural identifier failed for a member (a malformed query or diagram). Never
    /// reported as a member verdict.
    Identification(IdentificationError),
}

impl std::fmt::Display for DiscoveredTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(refusal) => refusal.fmt(f),
            Self::Identification(error) => error.fmt(f),
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
        if !self.enumeration_complete {
            return EnvelopeStatus::Partial;
        }
        let identified = self.identified_members();
        if identified == self.members.len() {
            EnvelopeStatus::IdentifiedInEveryMember
        } else if identified == 0 {
            EnvelopeStatus::NotIdentifiedInEveryMember
        } else {
            EnvelopeStatus::MembersDisagree
        }
    }

    /// Whether every enumerated member is certified with the same formula. Vacuously false
    /// when a member is not certified.
    #[must_use]
    pub fn formulas_agree(&self) -> bool {
        let mut formulas = self.members.iter().map(|m| match &m.verdict {
            MemberVerdict::Identified { formula, .. } => Some(formula),
            _ => None,
        });
        match formulas.next() {
            Some(Some(first)) => formulas.all(|f| f == Some(first)),
            _ => false,
        }
    }
}

fn refusal(
    detail: &'static str,
    request: &DiscoveredTransportRequest<'_>,
    unresolved: Vec<UnresolvedPair>,
    remedy: String,
) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: DISCOVERED_TRANSPORT_REASON,
        detail,
        unresolved,
        exhausted_search_bounds: request.exhausted_search_bounds,
        unknown_selection_targets: Vec::new(),
        remedy,
    })
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
    let class = request.class;
    let unresolved = unresolved_pairs(class);

    if request.exhausted_search_bounds > 0 {
        return Err(refusal(
            "discovered_transport.search_incomplete",
            request,
            unresolved,
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
            unresolved,
            "orientation evidence conflicted on at least one edge, so the structure is not a \
             CPDAG with members. Resolve the conflicting edges with background knowledge, or \
             declare the graph"
                .to_owned(),
        ));
    }
    let unknown: Vec<VariableId> = request
        .selection_targets
        .iter()
        .copied()
        .filter(|target| !class.nodes().contains(&NodeRef::Static(*target)))
        .collect();
    if !unknown.is_empty() {
        let names: Vec<String> = unknown.iter().map(|v| format!("v{}", v.raw())).collect();
        let mut error = refusal(
            "discovered_transport.selection_target_unknown",
            request,
            unresolved,
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
            unresolved,
            "the member bound is zero; give a positive bound on the members to enumerate"
                .to_owned(),
        ));
    }

    let mut sampler = CpdagCompletionSampler::new(class.clone(), request.member_bound)
        .map_err(|error| {
            refusal(
                "discovered_transport.class_too_large",
                request,
                unresolved.clone(),
                format!(
                    "the class cannot be enumerated ({error}). Orient edges with background \
                     knowledge or an intervention, or declare the graph"
                ),
            )
        })?;
    let dags: Vec<_> = sampler.by_ref().map(|completion| completion.graph).collect();
    let enumeration_complete = !sampler.hit_cap();
    if dags.is_empty() {
        return Err(refusal(
            "discovered_transport.no_members",
            request,
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
        let admg = Admg::from_dag(dag).map_err(|error| {
            DiscoveredTransportError::Identification(IdentificationError::invalid_input(
                error.to_string(),
            ))
        })?;
        let diagram = SelectionDiagram::try_new(admg, selection.clone()).map_err(|error| {
            DiscoveredTransportError::Identification(IdentificationError::invalid_input(
                error.to_string(),
            ))
        })?;
        let verdict = match identifier
            .identify(&diagram, query)
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
        };
        let mut directed_edges: Vec<(VariableId, VariableId)> = dag
            .edges()
            .filter_map(|edge| edge.parent_child())
            .filter_map(|(parent, child)| {
                Some((variable_of(class, parent)?, variable_of(class, child)?))
            })
            .collect();
        directed_edges.sort_by_key(|(p, c)| (p.raw(), c.raw()));
        members.push(EnvelopeMember { directed_edges, verdict });
    }
    members.sort_by(|a, b| {
        let key = |m: &EnvelopeMember| {
            m.directed_edges.iter().map(|(p, c)| (p.raw(), c.raw())).collect::<Vec<_>>()
        };
        key(a).cmp(&key(b))
    });

    Ok(StructuralTransportEnvelope {
        members,
        unresolved,
        enumeration_complete,
        member_bound: request.member_bound,
    })
}

/// Refusal for a PAG: its members are not causal graphs the structural rules describe.
///
/// # Errors
///
/// Always [`DiscoveredTransportError::Refused`] with
/// `discovered_transport.pag_members_not_causal_graphs`.
pub fn refuse_pag_transport(
    unresolved: Vec<UnresolvedPair>,
    exhausted_search_bounds: usize,
) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: DISCOVERED_TRANSPORT_REASON,
        detail: "discovered_transport.pag_members_not_causal_graphs",
        unresolved,
        exhausted_search_bounds,
        unknown_selection_targets: Vec::new(),
        remedy: "the members of a PAG are maximal ancestral graphs: a directed edge states \
                 ancestry, not a causal parent, and a bidirected edge stands for a latent \
                 path, so the structural transport rules do not describe them. Resolve the PAG \
                 to a declared causal graph (an ADMG you assert) and request transport with \
                 that graph"
            .to_owned(),
    })
}
