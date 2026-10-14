//! Transport of a response across the equivalence class a discovery run returned.
//!
//! A discovered structure is not a selection diagram: a CPDAG stands for every DAG of a
//! Markov equivalence class and the data do not say which member is true. This module is the
//! facade entry for the one thing that can honestly be said about a transport request over
//! such a class: the structural verdict of each member of the class, side by side, labelled
//! `structural_envelope`.
//!
//! * [`transport_discovered_structure`] reads a retained [`DiscoveryOutcome`] and the
//!   caller's declared selection targets and query, and returns a
//!   [`DiscoveredTransportReport`] (the envelope with the identities of the structure and
//!   result it was derived from) or a typed [`DiscoveredTransportRefusal`] that names every
//!   unresolved pair and every exhausted search bound.
//! * [`discover_and_transport`] runs static discovery on a table and then the transport
//!   request, so a request from data is one call. A refusal of the transport request does not
//!   discard the discovery: both are returned.
//! * [`discovered_transport_refusal`] renders a refusal on the shared external refusal
//!   surface.
//!
//! # What is and is not returned
//!
//! The envelope states which members the implemented structural rules certify and which
//! formulas they emit. It is not a point, an interval or a bound on an effect, and it carries
//! no probability: members are not weighted, and the true graph is a member only under
//! [`ENVELOPE_ASSUMPTIONS`], none of which this module can check. A PAG is refused (its
//! members are ancestral graphs, not causal graphs the rules describe), a temporal structure
//! is refused (outside this module), and an unfinished search, an orientation conflict, a
//! selection target outside the structure or a class too large for the member bound is
//! refused or reported `Partial`; nothing is read as a statement about one member.
//!
//! # Refresh with outcome-only data
//!
//! The report is derived on every call from the class and the declared query; nothing is
//! cached or updated in place. Outcome-only rows leave the class unchanged and so leave the
//! envelope's content unchanged; a rerun that yields another class is a new input and the
//! envelope, and its [`DiscoveredTransportReport::identity`], are recomputed from it. The
//! selection targets are kept only by being passed again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, ExecutionContext, TransportQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{
    DiscoveryOutcome, NativeResult, SelectiveDiscoveryError, SelectiveRunReport,
    StaticDiscoveryContract, StaticSelectiveDiscovery,
};
use antecedent_identify::discovered_transport::{
    DiscoveredTransportRequest, refuse_pag_transport, transport_across_class,
};

pub use antecedent_identify::discovered_transport::{
    DEFAULT_ENVELOPE_MEMBER_BOUND, DISCOVERED_TRANSPORT_REASON, DiscoveredTransportError,
    DiscoveredTransportRefusal, ENVELOPE_ASSUMPTIONS, EnvelopeMember, EnvelopeStatus,
    MemberVerdict, StructuralTransportEnvelope, UnresolvedPair,
};

/// Detail of the refusal of a temporal structure.
pub const TEMPORAL_STRUCTURE_DETAIL: &str = "discovered_transport.temporal_structure_unsupported";

const REPORT_TAG: &str = "antecedent.discovered_transport.report.v1";

/// A structural envelope with the identities of the discovery it was derived from.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredTransportReport {
    envelope: StructuralTransportEnvelope,
    structure_identity: [u8; 32],
    result_identity: [u8; 32],
    selection_targets: Vec<VariableId>,
    identity: [u8; 32],
}

impl DiscoveredTransportReport {
    /// What this object is, as a stable report label: always `structural_envelope`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        "structural_envelope"
    }

    /// The members of the class with their structural verdicts.
    #[must_use]
    pub const fn envelope(&self) -> &StructuralTransportEnvelope {
        &self.envelope
    }

    /// Identity of the discovered structure the class was read from.
    #[must_use]
    pub const fn structure_identity(&self) -> &[u8; 32] {
        &self.structure_identity
    }

    /// Identity of the discovery result the structure came from.
    #[must_use]
    pub const fn result_identity(&self) -> &[u8; 32] {
        &self.result_identity
    }

    /// The declared selection targets, ascending.
    #[must_use]
    pub fn selection_targets(&self) -> &[VariableId] {
        &self.selection_targets
    }

    /// The assumptions under which the true graph is a member of the class. Nothing here
    /// checks them.
    #[must_use]
    pub const fn assumptions(&self) -> &'static [&'static str] {
        &ENVELOPE_ASSUMPTIONS
    }

    /// Identity of the whole report: the structure and result it was read from, the declared
    /// targets, the member bound and completeness, and every member's edges and verdict.
    #[must_use]
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }
}

fn report_identity(
    envelope: &StructuralTransportEnvelope,
    structure: &[u8; 32],
    result: &[u8; 32],
    targets: &[VariableId],
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(REPORT_TAG);
    writer.bytes(structure).bytes(result).word(targets.len() as u64);
    for target in targets {
        writer.word(u64::from(target.raw()));
    }
    writer
        .word(envelope.member_bound() as u64)
        .word(u64::from(envelope.enumeration_complete()))
        .word(envelope.members().len() as u64);
    for member in envelope.members() {
        writer.word(member.directed_edges.len() as u64);
        for (parent, child) in &member.directed_edges {
            writer.word(u64::from(parent.raw())).word(u64::from(child.raw()));
        }
        match &member.verdict {
            MemberVerdict::Identified { rule, .. } => writer.word(1).text(rule),
            MemberVerdict::NotCertified { reason } => writer.word(2).text(reason),
            MemberVerdict::MissingEvidence { reason } => writer.word(3).text(reason),
        };
    }
    writer.finish()
}

fn exhausted_bounds(outcome: &DiscoveryOutcome) -> usize {
    outcome.bounded_search_receipts.len().max(usize::from(outcome.search_incomplete))
}

fn unresolved_of(outcome: &DiscoveryOutcome) -> Vec<UnresolvedPair> {
    let mut pairs: Vec<UnresolvedPair> = outcome
        .structure
        .unresolved_edges()
        .iter()
        .map(|u| {
            let (a, b) = (u.edge.a.variable_id(), u.edge.b.variable_id());
            let (a, b) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
            UnresolvedPair { a, b, reason: u.reason }
        })
        .collect();
    pairs.sort_by_key(|pair| (pair.a.raw(), pair.b.raw()));
    pairs
}

/// The structural transport verdict of `query` across the class `outcome` returned.
///
/// `selection_targets` are variables whose mechanisms may differ between the populations of
/// the query; they are declared by the caller and are never derived from the data.
/// `member_bound` bounds the enumeration of the class ([`DEFAULT_ENVELOPE_MEMBER_BOUND`] is
/// the usual value).
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] for a PAG
/// (`discovered_transport.pag_members_not_causal_graphs`), a temporal structure
/// ([`TEMPORAL_STRUCTURE_DETAIL`]), an exhausted search bound, an orientation conflict, a
/// target outside the structure, a zero member bound, or a class that cannot be enumerated;
/// [`DiscoveredTransportError::Identification`] for a malformed query or diagram.
pub fn transport_discovered_structure(
    outcome: &DiscoveryOutcome,
    selection_targets: &[VariableId],
    query: &TransportQuery,
    member_bound: usize,
) -> Result<DiscoveredTransportReport, DiscoveredTransportError> {
    let exhausted = exhausted_bounds(outcome);
    let class = match &outcome.native {
        NativeResult::StaticCpdag(result) => &result.evidence.graph,
        NativeResult::StaticPag(_) => {
            return Err(refuse_pag_transport(unresolved_of(outcome), exhausted));
        }
        _ => {
            return Err(DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
                code: DISCOVERED_TRANSPORT_REASON,
                detail: TEMPORAL_STRUCTURE_DETAIL,
                unresolved: unresolved_of(outcome),
                exhausted_search_bounds: exhausted,
                unknown_selection_targets: Vec::new(),
                remedy: "a temporal discovered structure is not enumerated here: declare the \
                         unrolled causal graph and its selection diagram and request \
                         transport through the temporal transport route"
                    .to_owned(),
            }));
        }
    };
    let request = DiscoveredTransportRequest {
        class,
        selection_targets,
        exhausted_search_bounds: exhausted,
        member_bound,
    };
    let envelope = transport_across_class(&request, query)?;
    let mut targets = selection_targets.to_vec();
    targets.sort_unstable_by_key(|t| t.raw());
    targets.dedup();
    let identity =
        report_identity(&envelope, &outcome.structure_identity, &outcome.result_identity, &targets);
    Ok(DiscoveredTransportReport {
        envelope,
        structure_identity: outcome.structure_identity,
        result_identity: outcome.result_identity,
        selection_targets: targets,
        identity,
    })
}

/// What [`discover_and_transport`] returns: the discovery and the transport request's verdict.
#[derive(Clone, Debug)]
pub struct DiscoveredTransportRun {
    /// What the discovery request did (stage plan, observed work).
    pub discovery: SelectiveRunReport,
    /// The retained discovery outcome the request was made over.
    pub outcome: DiscoveryOutcome,
    /// The envelope, or the typed refusal. A refusal here leaves the discovery above intact.
    pub transport: Result<DiscoveredTransportReport, DiscoveredTransportError>,
}

/// Run static discovery on `data` under `contract`, then request the structural transport
/// verdict across the class it returned.
///
/// The adapter is local to the call: nothing is retained for a later request, so the statistic
/// reuse of [`StaticSelectiveDiscovery`] is not exercised. Keep an adapter and call
/// [`transport_discovered_structure`] on its outcome to reuse statistics between requests.
///
/// # Errors
///
/// A contract or data discovery refuses, or an algorithm failure or cancellation. A refusal of
/// the transport request is returned inside the run, not as an error.
pub fn discover_and_transport(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
    selection_targets: &[VariableId],
    query: &TransportQuery,
    member_bound: usize,
    ctx: &ExecutionContext,
) -> Result<DiscoveredTransportRun, SelectiveDiscoveryError> {
    let mut adapter = StaticSelectiveDiscovery::new();
    let discovery = adapter.run(contract, data, ctx)?;
    let outcome = adapter.outcome().cloned().ok_or_else(|| {
        SelectiveDiscoveryError::refused(
            antecedent_core::reason_code!("invalid_argument"),
            "selective_discovery.nothing_retained",
            "the discovery run retained no outcome",
        )
    })?;
    let transport =
        transport_discovered_structure(&outcome, selection_targets, query, member_bound);
    Ok(DiscoveredTransportRun { discovery, outcome, transport })
}

fn remedy_of(detail: &str) -> &'static str {
    match detail {
        "discovered_transport.search_incomplete" => {
            "rerun discovery with larger search bounds, or declare the causal graph and its \
             selection diagram; transport is not computed across an incomplete class"
        }
        "discovered_transport.orientation_conflict" => {
            "resolve the conflicting edges with background knowledge, or declare the causal \
             graph and its selection diagram"
        }
        "discovered_transport.selection_target_unknown" => {
            "declare selection targets among the searched variables"
        }
        "discovered_transport.class_too_large" | "discovered_transport.member_bound_zero" => {
            "orient edges with background knowledge or an intervention, give a positive member \
             bound, or declare the causal graph"
        }
        "discovered_transport.no_members" => {
            "rerun discovery or declare the causal graph; the structure is not the CPDAG of any \
             graph"
        }
        "discovered_transport.pag_members_not_causal_graphs" => {
            "resolve the PAG to a declared causal graph, or run PC under causal sufficiency, \
             then request transport"
        }
        _ => {
            "declare the causal graph and its selection diagram and request transport with \
             that graph"
        }
    }
}

/// Render a refusal of a transport request over a discovered class on the shared external
/// refusal surface.
///
/// Unresolved pairs are listed in `offending` as `a~b:reason`, joined by `;`; for a refusal on
/// unknown selection targets `offending` names those targets instead.
#[must_use]
pub fn discovered_transport_refusal(refusal: &DiscoveredTransportRefusal) -> ExternalRefusal {
    let mut out = ExternalRefusal::new(refusal.code, "bind", refusal.detail)
        .with_remedy(remedy_of(refusal.detail));
    let offending = if refusal.unknown_selection_targets.is_empty() {
        refusal
            .unresolved
            .iter()
            .map(|p| format!("v{}~v{}:{}", p.a.raw(), p.b.raw(), p.reason))
            .collect::<Vec<_>>()
            .join(";")
    } else {
        refusal
            .unknown_selection_targets
            .iter()
            .map(|t| format!("v{}", t.raw()))
            .collect::<Vec<_>>()
            .join(";")
    };
    if !offending.is_empty() {
        out = out.with_offending(offending);
    }
    out
}

/// A stable label for the kind of formula a certified member emitted: `direct`,
/// `standardize` or `recursive_factorization`.
#[must_use]
pub const fn formula_kind(formula: &antecedent_identify::TransportFormula) -> &'static str {
    use antecedent_identify::TransportFormula;
    match formula {
        TransportFormula::Direct(_) => "direct",
        TransportFormula::Standardize { .. } => "standardize",
        TransportFormula::RecursiveFactorization { .. } => "recursive_factorization",
    }
}
