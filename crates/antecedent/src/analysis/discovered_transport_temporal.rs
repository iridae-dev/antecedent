//! Transport of a response at a declared horizon across a discovered temporal class.
//!
//! A temporal discovery returns a stationary template over lagged nodes: a lagged DAG (PCMCI),
//! a temporal CPDAG (PCMCI+) or a temporal PAG (LPCMCI). A request names a treatment, an
//! outcome and the number of steps between the intervention and the outcome. Every member of the
//! class is unrolled over the window the request needs (see
//! [`antecedent_identify::discovered_transport_temporal`]); time order is kept by construction,
//! the lag assumptions are the template's, and a selection target applies at every slice.
//!
//! * [`transport_discovered_temporal_structure`] reads a retained [`DiscoveryOutcome`] of a lagged
//!   DAG or a temporal CPDAG.
//! * [`transport_temporal_pag`] reads a temporal PAG, which the selective adapter does not wrap.
//! * [`estimate_discovered_temporal_transport`] turns the report into a point, a structural
//!   range or a refusal exactly as [`super::discovered_transport_estimate`] does for a static
//!   class, from two time-ordered tables.
//!
//! # Evidence
//!
//! Each table is a time series: one row per step, one column per variable, integer codes. A
//! window row is the values at the offsets the unrolled graph names, so the rows of the window
//! table overlap and are dependent. The estimate is a plug-in under stationarity, and an
//! interval on a point answer comes from a circular block bootstrap of each series' time axis:
//! blocks of consecutive steps are drawn once per population and every column, hence every arm
//! of the source experiment, is taken at those steps, so blocks are contiguous in time and arms
//! are resampled jointly. The block length defaults to twice the window's span and is conditional on the discovered class; it
//! carries no coverage claim until its complete-procedure design is measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::result_large_err,
    reason = "the public error shape is kept as released; boxing the refusal would break callers that match on it"
)]

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, TransportQuery, VariableId};
use antecedent_discovery::selective::{DiscoveredStructure, DiscoveryOutcome, NativeResult};
use antecedent_identify::TemporalCompletionGraph;
use antecedent_identify::discovered_transport_temporal::{
    TemporalEnvelopeMember, TemporalTransportRequest, TemporalWindow,
    transport_across_temporal_class,
};

pub use antecedent_identify::discovered_transport_temporal::{
    LaggedEdge, TemporalClass, TemporalTransportEnvelope,
};

use super::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRefusal, ENVELOPE_ASSUMPTIONS, UnresolvedPair,
};
use super::discovered_transport_estimate::Columns;
use super::discovered_transport_estimate::{
    ClassFacts, DiscoveredTransportEstimate, EvidenceColumns, MemberProblem, Settings,
    TransportEvidence, columns_of, estimate_class,
};
const REPORT_TAG: &str = "antecedent.discovered_transport.temporal_report.v1";

/// The declared horizon of a temporal request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalHorizon {
    /// Steps from the intervention at the origin to the outcome. Zero asks for the
    /// contemporaneous effect.
    pub horizon_steps: u32,
    /// Slices of history in the window. `None` is twice the template's largest lag.
    pub history: Option<u32>,
}

/// A temporal structural envelope with the identities of the discovery it was derived from.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredTemporalTransportReport {
    envelope: TemporalTransportEnvelope,
    structure_identity: [u8; 32],
    result_identity: Option<[u8; 32]>,
    selection_targets: Vec<VariableId>,
    identity: [u8; 32],
}

impl DiscoveredTemporalTransportReport {
    /// What this object is, as a stable report label: always `structural_envelope`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        "structural_envelope"
    }

    /// The members of the class with their structural verdicts at the declared horizon.
    #[must_use]
    pub const fn envelope(&self) -> &TemporalTransportEnvelope {
        &self.envelope
    }

    /// Identity of the discovered structure the class was read from.
    #[must_use]
    pub const fn structure_identity(&self) -> &[u8; 32] {
        &self.structure_identity
    }

    /// Identity of the discovery result, when the structure came from a retained outcome.
    #[must_use]
    pub const fn result_identity(&self) -> Option<&[u8; 32]> {
        self.result_identity.as_ref()
    }

    /// The declared selection targets, ascending.
    #[must_use]
    pub fn selection_targets(&self) -> &[VariableId] {
        &self.selection_targets
    }

    /// The assumptions under which the true process is a member of the class. Nothing here
    /// checks them.
    #[must_use]
    pub const fn assumptions(&self) -> &'static [&'static str] {
        &ENVELOPE_ASSUMPTIONS
    }

    /// Identity of the whole report: the structure, the declared targets, the horizon and the
    /// window, and every member's edges and verdict.
    #[must_use]
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }
}

fn report_identity(
    envelope: &TemporalTransportEnvelope,
    structure: &[u8; 32],
    result: Option<&[u8; 32]>,
    targets: &[VariableId],
) -> [u8; 32] {
    use antecedent_identify::discovered_transport::MemberVerdict;
    let mut writer = CanonicalWriter::new(REPORT_TAG);
    writer.bytes(structure).word(u64::from(result.is_some()));
    if let Some(result) = result {
        writer.bytes(result);
    }
    writer.word(targets.len() as u64);
    for target in targets {
        writer.word(u64::from(target.raw()));
    }
    writer
        .word(envelope.member_bound() as u64)
        .word(u64::from(envelope.enumeration_complete()))
        .word(u64::from(envelope.history()))
        .word(u64::from(envelope.horizon_steps()))
        .word(u64::from(envelope.max_lag()))
        .word(envelope.members().len() as u64);
    for member in envelope.members() {
        for edges in [&member.directed_edges, &member.bidirected_edges, &member.confounded_edges] {
            writer.word(edges.len() as u64);
            for edge in edges {
                writer
                    .word(u64::from(edge.from.raw()))
                    .word(u64::from(edge.to.raw()))
                    .word(u64::from(edge.lag));
            }
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

fn finish(
    envelope: TemporalTransportEnvelope,
    structure: [u8; 32],
    result: Option<[u8; 32]>,
    selection_targets: &[VariableId],
) -> DiscoveredTemporalTransportReport {
    let mut targets = selection_targets.to_vec();
    targets.sort_unstable_by_key(|t| t.raw());
    targets.dedup();
    let identity = report_identity(&envelope, &structure, result.as_ref(), &targets);
    DiscoveredTemporalTransportReport {
        envelope,
        structure_identity: structure,
        result_identity: result,
        selection_targets: targets,
        identity,
    }
}

#[allow(clippy::too_many_arguments, reason = "one request: class, bounds, horizon, targets, query")]
fn across(
    class: TemporalClass<'_>,
    exhausted: usize,
    member_bound: usize,
    horizon: TemporalHorizon,
    selection_targets: &[VariableId],
    query: &TransportQuery,
) -> Result<TemporalTransportEnvelope, DiscoveredTransportError> {
    transport_across_temporal_class(
        &TemporalTransportRequest {
            class,
            selection_targets,
            exhausted_search_bounds: exhausted,
            member_bound,
            completion_operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
            horizon_steps: horizon.horizon_steps,
            history: horizon.history,
        },
        query,
    )
}

/// The structural transport verdict of `query` at `horizon` across the temporal class `outcome`
/// returned (a lagged DAG or a temporal CPDAG).
///
/// `query` is stated on the template's variables: a mean-curve response of one outcome under
/// one treatment. `selection_targets` are variables whose mechanisms differ between the
/// populations, at every slice; they are declared by the caller and never derived from the
/// data.
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] for a static structure (the static entry answers it),
/// an exhausted search bound, an orientation conflict, a target or query variable outside the
/// structure, a zero member bound, a query that cannot be unrolled, or a class that cannot be
/// enumerated; [`DiscoveredTransportError::Identification`] for a malformed query or window.
pub fn transport_discovered_temporal_structure(
    outcome: &DiscoveryOutcome,
    horizon: TemporalHorizon,
    selection_targets: &[VariableId],
    query: &TransportQuery,
    member_bound: usize,
) -> Result<DiscoveredTemporalTransportReport, DiscoveredTransportError> {
    let exhausted = exhausted_bounds(outcome);
    let class = match &outcome.native {
        NativeResult::TemporalDag(result) => TemporalClass::Dag(&result.evidence.graph),
        NativeResult::TemporalCpdag(result) => TemporalClass::Cpdag(&result.evidence.graph),
        _ => {
            return Err(DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
                code: antecedent_core::reason_code!("discovered_structure_unresolved"),
                detail: "discovered_transport.temporal_query_invalid",
                unresolved: Vec::new(),
                exhausted_search_bounds: exhausted,
                unknown_selection_targets: Vec::new(),
                member_reasons: Vec::new(),
                remedy: "a static discovered structure has no horizon: request transport through \
                         the static entry"
                    .to_owned(),
            }));
        }
    };
    let envelope = across(class, exhausted, member_bound, horizon, selection_targets, query)?;
    Ok(finish(
        envelope,
        outcome.structure_identity,
        Some(outcome.result_identity),
        selection_targets,
    ))
}

/// The structural transport verdict of `query` at `horizon` across a temporal class held as a
/// graph: a lagged DAG, a temporal CPDAG, or the temporal PAG an LPCMCI discovery returns (the
/// selective adapter does not wrap that one).
///
/// A circle endpoint of a PAG is enumerated, never read as the absence of confounding; see
/// [`antecedent_identify::discovered_transport`]. `exhausted_search_bounds` is the number of
/// search bounds the discovery that produced the graph exhausted.
///
/// # Errors
///
/// As [`transport_discovered_temporal_structure`], and for PAG marks that admit more than one
/// Markov equivalence class in the window or a completion with a selection-bias edge.
pub fn transport_temporal_class(
    class: TemporalClass<'_>,
    exhausted_search_bounds: usize,
    horizon: TemporalHorizon,
    selection_targets: &[VariableId],
    query: &TransportQuery,
    member_bound: usize,
) -> Result<DiscoveredTemporalTransportReport, DiscoveredTransportError> {
    let envelope =
        across(class, exhausted_search_bounds, member_bound, horizon, selection_targets, query)?;
    let structure = match class {
        TemporalClass::Dag(graph) => DiscoveredStructure::from_temporal_dag(graph).identity(),
        TemporalClass::Cpdag(graph) => DiscoveredStructure::from_temporal_cpdag(graph).identity(),
        TemporalClass::Pag(graph) => TemporalCompletionGraph::Mag(graph.clone()).digest(),
    };
    Ok(finish(envelope, structure, None, selection_targets))
}

// -- estimate ---------------------------------------------------------------------------------

fn invalid(message: &str) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "discovered_transport.evidence_invalid",
        unresolved: Vec::new(),
        exhausted_search_bounds: 0,
        unknown_selection_targets: Vec::new(),
        member_reasons: Vec::new(),
        remedy: message.to_owned(),
    })
}

/// The window table of one population: for each window row, the value of every needed slot.
fn window_columns(
    series: &Columns,
    window: &TemporalWindow,
    needed: &BTreeSet<VariableId>,
) -> Result<Columns, DiscoveredTransportError> {
    let history = usize::try_from(window.indexer.history())
        .map_err(|_| invalid("the history does not fit a row index"))?;
    let ahead = usize::try_from(window.indexer.horizon())
        .map_err(|_| invalid("the horizon does not fit a row index"))?
        - 1;
    let steps = series.values().next().map_or(0, Vec::len);
    let usable = steps.checked_sub(history + ahead).filter(|n| *n > 0).ok_or_else(|| {
        invalid(
            "the series is shorter than the window: it needs more steps than the history and \
             the horizon together",
        )
    })?;
    let mut out = Columns::new();
    for slot in needed {
        let key = window
            .indexer
            .key_of(slot.raw())
            .map_err(|e| DiscoveredTransportError::Estimation(e.to_string()))?;
        let column = series
            .get(&key.variable)
            .ok_or_else(|| invalid("the table has no column for a variable of the structure"))?;
        // `offset` runs from `-history` to `ahead`; the row at window start `start` reads the
        // series at `start + history + offset`, which is in range for every usable start.
        let shift = i64::try_from(history).map_err(|_| invalid("history"))? + i64::from(key.offset);
        let shift = usize::try_from(shift).map_err(|_| invalid("the window offset"))?;
        out.insert(*slot, (0..usable).map(|start| column[start + shift]).collect());
    }
    Ok(out)
}

/// Estimate `query` at the report's horizon across the temporal class `report` was made over.
///
/// `evidence` holds two time-ordered tables (one row per step, one column per variable of the
/// structure). Rows of a window overlap, so an interval uses the circular block bootstrap.
///
/// # Errors
///
/// As [`super::discovered_transport_estimate::estimate_discovered_transport`], and
/// `discovered_transport.evidence_invalid` for a series shorter than its window or an interval
/// request.
pub fn estimate_discovered_temporal_transport(
    report: &DiscoveredTemporalTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredTransportEstimate, DiscoveredTransportError> {
    super::discovered_transport_estimate::validate_evidence_query(query)?;
    let envelope = report.envelope();
    let window = envelope.window();
    let antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } =
        &query.response.functional
    else {
        return Err(invalid("an estimate asks for a mean response curve"));
    };
    let antecedent_core::GridSpec::Values(levels) = &treatment.grid else {
        return Err(invalid("the treatment grid must list its levels"));
    };
    // Slots any member's graph keeps, and the template variables they read.
    let needed: BTreeSet<VariableId> = envelope
        .members()
        .iter()
        .flat_map(|m| {
            let graph = m.window_graph();
            graph
                .nodes()
                .iter()
                .filter_map(|node| match node {
                    antecedent_core::NodeRef::Static(slot) => Some(*slot),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut variables = BTreeSet::new();
    for slot in &needed {
        let key = window
            .indexer
            .key_of(slot.raw())
            .map_err(|e| DiscoveredTransportError::Estimation(e.to_string()))?;
        variables.insert(key.variable);
    }
    let variables: Vec<VariableId> = variables.into_iter().collect();
    let source = columns_of(evidence.source, &variables)?;
    let target = columns_of(evidence.target, &variables)?;
    let columns = EvidenceColumns {
        source: window_columns(&source, window, &needed)?,
        target: window_columns(&target, window, &needed)?,
    };
    let problems: Vec<MemberProblem> = envelope
        .members()
        .iter()
        .map(|member: &TemporalEnvelopeMember| {
            let graph = member.window_graph().clone();
            let selection: Arc<[VariableId]> = window
                .selection_targets
                .iter()
                .copied()
                .filter(|slot| graph.dense_of_static(*slot).is_some())
                .collect();
            MemberProblem { graph, selection, treatment: window.treatment, outcome: window.outcome }
        })
        .collect();
    // Window rows overlap, so a block of at least two windows' span keeps their dependence
    // inside a block; the default doubles the span of a window.
    let span = window.indexer.history().saturating_add(window.indexer.horizon());
    let block_length = evidence.block_length.unwrap_or_else(|| span.saturating_mul(2).max(2));
    let windowed = TransportEvidence {
        rows_independent: false,
        block_length: Some(block_length),
        ..*evidence
    };
    let settings = Settings {
        prefer_declared_selection: false,
        levels,
        source: &query.source_population,
        target: &query.target_population,
        evidence: &windowed,
    };
    let unresolved: &[UnresolvedPair] = envelope.unresolved();
    let class = ClassFacts {
        report_identity: *report.identity(),
        selection_targets: report.selection_targets(),
        unresolved,
        status: envelope.status(),
    };
    estimate_class(&class, &problems, &columns, &settings, ctx)
}
