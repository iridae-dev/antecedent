//! Explicit experimental compatibility refinement of a discovered static class.
//!
//! A randomized treatment has no effect on an outcome in a causal graph with no directed
//! treatment-to-outcome path. Such a graph implies treatment/outcome independence in the
//! treatment trial, even when they are confounded observationally. This opt-in route tests
//! that single implication with a two-sided Fisher exact test of binary trial assignments
//! and outcomes. It does not force an adjacency or choose among the remaining graphs.
//!
//! Every no-path member asserts the same null, so one test is shared by all of them. Its
//! rejection level, contingency table, p-value, original member indices and directed edges
//! are retained. The original structural report is unchanged. Intervals remain conditional
//! on discovery and refinement; complete-procedure calibration must include both steps.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::result_large_err,
    reason = "match the shared public discovered-transport refusal type without breaking callers"
)]

use antecedent_core::{CanonicalWriter, ExecutionContext, TransportQuery};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_io::IoError;
use serde::{Deserialize, Serialize};

use super::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRefusal, DiscoveredTransportReport, MemberVerdict,
};
use super::discovered_transport_estimate::{
    DiscoveredTransportEstimate, TransportAnswer, TransportEvidence, VerifiedEstimate, columns_of,
    estimate_selected_discovered_transport, validate_evidence_query, verify_trial_member_estimate,
};

const PROOF_FORMAT: &str = "antecedent.discovered_transport.trial_compatibility.v1";
const ESTIMATE_FORMAT: &str = "antecedent.discovered_transport.trial_estimate.v1";
const LEGACY_ESTIMATION_STRATEGY: &str = "minimal_pretreatment_standardization";
const SELECTION_ESTIMATION_STRATEGY: &str = "declared_selection_joint_pretreatment_standardization";

fn legacy_estimation_strategy() -> String {
    LEGACY_ESTIMATION_STRATEGY.into()
}

const MAX_TRIAL_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TRIAL_ARTIFACT_MEMBERS: usize = 4096;

/// What a verifier will decode: the largest artifact it reads and the most original members
/// it replays.
///
/// The defaults bound artifacts of hostile origin; a caller replaying its own larger receipts
/// (a wide class, a long trial) raises them through the `_within` verifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrialArtifactBudget {
    /// Largest artifact, in bytes, that is decoded.
    pub max_bytes: usize,
    /// Most original members that are replayed.
    pub max_members: usize,
}

impl Default for TrialArtifactBudget {
    fn default() -> Self {
        Self { max_bytes: MAX_TRIAL_ARTIFACT_BYTES, max_members: MAX_TRIAL_ARTIFACT_MEMBERS }
    }
}

/// Actual assignment design asserted for the source trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialRandomization {
    /// Assignment is independent of baseline causes/potential outcomes and uniform given
    /// the treated count. Common-probability Bernoulli randomization also qualifies after
    /// conditioning on that count. Stratified or covariate-adaptive allocation needs a
    /// conditional randomization test instead.
    CompletelyRandomized,
}

/// Experimental test settings declared before the trial rows are inspected.
#[derive(Clone, Copy, Debug)]
pub struct TrialCompatibilityOptions {
    /// Level of the one shared no-effect implication test.
    pub alpha: f64,
    /// Actual source assignment design, asserted by the caller rather than inferred.
    pub randomization: TrialRandomization,
}

/// One original member's tested implication and refinement decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialMemberCompatibility {
    /// Index in the original structural envelope, never a renumbered selected index.
    pub index: u32,
    /// Directed edges of this original member; latent edges cannot carry a directed effect.
    pub directed_edges: Vec<(u32, u32)>,
    /// Original latent edges, retained to replay the unchanged original report identity.
    pub bidirected_edges: Vec<(u32, u32)>,
    /// Invisible directed edges originally augmented with possible latent confounding.
    pub confounded_edges: Vec<(u32, u32)>,
    /// Original structural verdict, independently of experimental compatibility.
    pub structural_verdict: TrialStructuralVerdict,
    /// Whether the member permits a directed causal effect of treatment on outcome.
    pub has_directed_path: bool,
    /// Whether this member survived the declared experimental test.
    pub retained: bool,
}

/// Original structural verdict fields consumed by the report's canonical identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum TrialStructuralVerdict {
    /// An identified member with its original rule.
    Identified {
        /// Original identification rule.
        rule: String,
    },
    /// A member the original bounded structural route did not certify.
    NotCertified {
        /// Original reason.
        reason: String,
    },
    /// A member lacking the required original declared evidence.
    MissingEvidence {
        /// Original reason.
        reason: String,
    },
}

/// Replayable experimental test and every original member's decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialCompatibilityProof {
    format: String,
    /// Structural formula-selection policy, bound into the refined identity.
    /// Missing legacy fields retain their original minimal-standardizer identity.
    #[serde(default = "legacy_estimation_strategy")]
    pub estimation_strategy: String,
    /// Identity of the unchanged discovered structural report.
    pub original_report_identity: Vec<u8>,
    /// Original discovery structure identity.
    pub original_structure_identity: [u8; 32],
    /// Original discovery result identity.
    pub original_result_identity: [u8; 32],
    /// Original declared selection targets, in canonical order.
    pub original_selection_targets: Vec<u32>,
    /// Original enumeration bound.
    pub original_member_bound: u64,
    /// Original enumeration completeness; refinement requires true.
    pub original_enumeration_complete: bool,
    /// Randomized source treatment, coded zero or one.
    pub treatment: u32,
    /// Binary source outcome, coded zero or one.
    pub outcome: u32,
    /// The declared level of the one shared independence test.
    pub alpha: f64,
    /// Declared actual assignment mechanism the independence implication needs.
    pub randomization: TrialRandomization,
    /// Counts in order `(X=0,Y=0), (0,1), (1,0), (1,1)`.
    pub counts: [u64; 4],
    /// Exact consumed assignment/outcome rows, in their supplied order, for independent replay.
    pub source_trial_rows: Vec<[u8; 2]>,
    /// Canonical identity of the exact source columns consumed by the experimental test.
    pub source_trial_identity: [u8; 32],
    /// Conditional, two-sided Fisher exact p-value; margins are conditioned on.
    pub p_value: f64,
    /// All original members and the explicitly recorded refinement decisions.
    pub members: Vec<TrialMemberCompatibility>,
}

impl TrialCompatibilityProof {
    fn validate_replay_budget(
        &self,
        limits: ExactEvaluationLimits,
        budget: TrialArtifactBudget,
    ) -> Result<usize, IoError> {
        let exceeded = || IoError::Convert("trial compatibility replay budget exceeded".into());
        if self.members.len() > budget.max_members {
            return Err(exceeded());
        }
        let mut work = self
            .source_trial_rows
            .len()
            .checked_mul(4)
            .and_then(|rows| rows.checked_add(self.original_selection_targets.len()))
            .ok_or_else(exceeded)?;
        for member in &self.members {
            // Directed-path replay scans the edge list once per newly reachable layer.
            // At most twice the number of edges plus two vertices can be visited.
            let edges = member.directed_edges.len();
            let path_work = edges
                .checked_mul(2)
                .and_then(|vertices| vertices.checked_add(2))
                .and_then(|vertices| vertices.checked_mul(edges.max(1)))
                .ok_or_else(exceeded)?;
            work = work
                .checked_add(path_work)
                .and_then(|n| n.checked_add(member.bidirected_edges.len()))
                .and_then(|n| n.checked_add(member.confounded_edges.len()))
                .ok_or_else(exceeded)?;
        }
        if work > limits.operations {
            return Err(exceeded());
        }
        Ok(work)
    }

    /// Original envelope indices retained after the one experimental test.
    #[must_use]
    pub fn retained_members(&self) -> Vec<usize> {
        self.members.iter().filter(|m| m.retained).map(|m| m.index as usize).collect()
    }

    /// Identity binding the original report, complete test and every member decision.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new(PROOF_FORMAT);
        writer
            .bytes(&self.original_report_identity)
            .word(u64::from(self.treatment))
            .word(u64::from(self.outcome))
            .word(self.alpha.to_bits())
            .word(self.p_value.to_bits());
        writer.text("completely_randomized_binary_trial");
        for count in self.counts {
            writer.word(count);
        }
        writer.bytes(&self.source_trial_identity);
        writer.word(self.members.len() as u64);
        for member in &self.members {
            writer
                .word(u64::from(member.index))
                .word(u64::from(member.has_directed_path))
                .word(u64::from(member.retained))
                .word(member.directed_edges.len() as u64);
            for (from, to) in &member.directed_edges {
                writer.word(u64::from(*from)).word(u64::from(*to));
            }
        }
        if self.estimation_strategy != LEGACY_ESTIMATION_STRATEGY {
            writer.text(&self.estimation_strategy);
        }
        writer.finish()
    }

    fn replay_original_report_identity(&self) -> [u8; 32] {
        // Rebuild report.v1, binding every original graph and verdict rather than trusting
        // an opaque report hash beside an independently mutable exclusion list.
        let mut writer = CanonicalWriter::new("antecedent.discovered_transport.report.v1");
        writer
            .bytes(&self.original_structure_identity)
            .bytes(&self.original_result_identity)
            .word(self.original_selection_targets.len() as u64);
        for target in &self.original_selection_targets {
            writer.word(u64::from(*target));
        }
        writer
            .word(self.original_member_bound)
            .word(u64::from(self.original_enumeration_complete))
            .word(self.members.len() as u64);
        for member in &self.members {
            writer.word(member.directed_edges.len() as u64);
            for (a, b) in &member.directed_edges {
                writer.word(u64::from(*a)).word(u64::from(*b));
            }
            for edges in [&member.bidirected_edges, &member.confounded_edges] {
                if !edges.is_empty() {
                    writer.word(0x4c41_5445_4e54).word(edges.len() as u64);
                    for (a, b) in edges {
                        writer.word(u64::from(*a)).word(u64::from(*b));
                    }
                }
            }
            match &member.structural_verdict {
                TrialStructuralVerdict::Identified { rule } => writer.word(1).text(rule),
                TrialStructuralVerdict::NotCertified { reason } => writer.word(2).text(reason),
                TrialStructuralVerdict::MissingEvidence { reason } => writer.word(3).text(reason),
            };
        }
        writer.finish()
    }

    #[allow(
        clippy::float_cmp,
        reason = "independent proof replay must reproduce the stored p-value exactly"
    )]
    fn validate(&self) -> Result<(), IoError> {
        self.validate_with_empty(false)
    }

    #[allow(clippy::float_cmp, reason = "proof replay checks the exact original Fisher result")]
    fn validate_with_empty(&self, allow_empty: bool) -> Result<(), IoError> {
        let bad =
            || IoError::Convert("discovered trial compatibility proof does not replay".into());
        if self.format != PROOF_FORMAT
            || !matches!(
                self.estimation_strategy.as_str(),
                LEGACY_ESTIMATION_STRATEGY | SELECTION_ESTIMATION_STRATEGY
            )
            || self.original_report_identity.len() != 32
            || self.original_report_identity.as_slice()
                != self.replay_original_report_identity().as_slice()
            || !self.original_enumeration_complete
            || !(self.alpha > 0.0 && self.alpha < 0.5)
            || self.treatment == self.outcome
            || self.members.is_empty()
            || self.source_trial_rows.len() > 1_000_000
        {
            return Err(bad());
        }
        let mut counts = [0u64; 4];
        for [x, y] in &self.source_trial_rows {
            if *x > 1 || *y > 1 {
                return Err(bad());
            }
            counts[usize::from(*x) * 2 + usize::from(*y)] += 1;
        }
        if counts != self.counts
            || trial_identity(self.treatment, self.outcome, &self.source_trial_rows)
                != self.source_trial_identity
        {
            return Err(bad());
        }
        // Counts must first be reconstructed from the bounded rows. Untrusted
        // u64 counts must never reach Fisher's margin additions or mass loop.
        if fisher_exact(counts).map_err(IoError::from)? != self.p_value {
            return Err(bad());
        }
        for (index, member) in self.members.iter().enumerate() {
            let path = has_directed_path(&member.directed_edges, self.treatment, self.outcome);
            if member.index as usize != index
                || member.has_directed_path != path
                || member.retained != (path || self.p_value > self.alpha)
            {
                return Err(bad());
            }
        }
        if !allow_empty && self.retained_members().is_empty() {
            return Err(bad());
        }
        Ok(())
    }
}

/// A transported estimate together with the experimental refinement it actually used.
#[derive(Clone, Debug)]
pub struct TrialCompatibleTransportEstimate {
    estimate: DiscoveredTransportEstimate,
    proof: TrialCompatibilityProof,
}

impl TrialCompatibleTransportEstimate {
    /// Summary over every experimentally compatible member.
    #[must_use]
    pub const fn answer(&self) -> &TransportAnswer {
        self.estimate.answer()
    }
    /// Original class, experimental test and original retained member indices.
    #[must_use]
    pub const fn compatibility(&self) -> &TrialCompatibilityProof {
        &self.proof
    }
    /// Retained member grids, indexed by `compatibility().retained_members()`.
    #[must_use]
    pub const fn estimate(&self) -> &DiscoveredTransportEstimate {
        &self.estimate
    }
    /// Export the test proof and every retained causal/evaluation artifact together.
    ///
    /// # Errors
    /// Artifact encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        antecedent_io::to_cbor(&TrialEstimateWire {
            format: ESTIMATE_FORMAT.into(),
            proof: self.proof.clone(),
            estimate: self.estimate.export()?,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialEstimateWire {
    format: String,
    proof: TrialCompatibilityProof,
    estimate: Vec<u8>,
}

/// Replay the experimental test, all graph-path decisions and every retained member grid.
/// The operation limit bounds the compatibility proof and each retained grid separately;
/// at most 4096 grids can be replayed. Stored grid limits retain their original identity.
///
/// # Errors
/// An artifact above 16 MiB or 4096 original members, a proof exceeding the
/// caller's operation budget, or a test, decision, identity or grid that does not replay.
/// See [`verify_trial_compatible_transport_estimate_within`] to set the artifact budget.
pub fn verify_trial_compatible_transport_estimate(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(TrialCompatibilityProof, VerifiedEstimate), IoError> {
    verify_trial_compatible_transport_estimate_within(
        bytes,
        limits,
        TrialArtifactBudget::default(),
        ctx,
    )
}

/// [`verify_trial_compatible_transport_estimate`] under a caller-chosen artifact budget.
///
/// # Errors
/// As [`verify_trial_compatible_transport_estimate`], with the artifact bytes and original
/// members bounded by `budget`.
pub fn verify_trial_compatible_transport_estimate_within(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    budget: TrialArtifactBudget,
    ctx: &ExecutionContext,
) -> Result<(TrialCompatibilityProof, VerifiedEstimate), IoError> {
    if bytes.len() > budget.max_bytes {
        return Err(IoError::Convert("trial compatibility artifact byte budget exceeded".into()));
    }
    let wire: TrialEstimateWire = antecedent_io::from_cbor(bytes)?;
    if wire.format != ESTIMATE_FORMAT {
        return Err(IoError::Convert("unsupported trial estimate format".into()));
    }
    wire.proof.validate_replay_budget(limits, budget)?;
    wire.proof.validate()?;
    let retained = wire.proof.retained_members();
    let mut verified = verify_trial_member_estimate(&wire.estimate, limits, ctx, retained.len())?;
    if verified.report_identity != wire.proof.identity()
        || verified.members != (0..retained.len()).collect::<Vec<_>>()
    {
        return Err(IoError::Convert(
            "trial refinement and retained member artifacts disagree".into(),
        ));
    }
    verified.members = retained;
    Ok((wire.proof, verified))
}

/// Refine a complete static class using the actually supplied randomized binary trial,
/// then identify and estimate every retained member. The original DGP and report are unchanged.
/// Declared population-shift variables are retained in the estimation graph. Their joint
/// set is preferred for standardization only if every variable is pretreatment in that
/// member and the existing separation, catalog and independent proof checks succeed.
/// This policy depends on declared structure and evidence availability, never fitted
/// effects; its distinct strategy is bound into the replayable refinement identity.
///
/// `alpha` is declared before inspecting rows. Source trial rows must be independent; this
/// procedure makes no claim for dependent assignments or temporal windows. The source treatment
/// assignment and binary outcome must both be complete zero/one codes.
/// At most one million rows are accepted by the bounded exact-test implementation.
///
/// # Errors
/// Invalid trial evidence, an incomplete original class, rejection of every member, or the same
/// identification/evidence/budget failures as ordinary discovered transport estimation.
#[allow(
    clippy::too_many_lines,
    reason = "one auditable pipeline records the exact test inputs, every graph decision and all retained estimation artifacts"
)]
#[allow(clippy::float_cmp, reason = "trial inputs must be exact binary zero/one codes")]
pub fn estimate_trial_compatible_discovered_transport(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    ctx: &ExecutionContext,
) -> Result<TrialCompatibleTransportEstimate, DiscoveredTransportError> {
    let proof = trial_compatibility_proof(report, query, evidence, options)?;
    let retained = proof.retained_members();
    if retained.is_empty() {
        return Err(DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
            code: antecedent_core::reason_code!("discovered_structure_unresolved"),
            detail: "discovered_transport.trial_rejects_every_member",
            unresolved: report.envelope().unresolved().to_vec(),
            exhausted_search_bounds: 0,
            unknown_selection_targets: Vec::new(),
            member_reasons: Vec::new(),
            remedy: "the declared randomized trial rejects the common no-effect implication of every discovered member; revisit discovery rather than invent an edge".into(),
        }));
    }
    let estimate = estimate_selected_discovered_transport(
        report,
        query,
        evidence,
        Some(&retained),
        Some(proof.identity()),
        ctx,
    )?;
    Ok(TrialCompatibleTransportEstimate { estimate, proof })
}

#[allow(
    clippy::too_many_lines,
    reason = "retain the exact existing trial validation and refinement protocol"
)]
#[allow(clippy::float_cmp, reason = "trial inputs must be exact binary codes")]
fn trial_compatibility_proof(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
) -> Result<TrialCompatibilityProof, DiscoveredTransportError> {
    let alpha = options.alpha;
    validate_evidence_query(query)?;
    let refuse = |detail, remedy: String| {
        DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
            code: antecedent_core::reason_code!("discovered_structure_unresolved"),
            detail,
            unresolved: report.envelope().unresolved().to_vec(),
            exhausted_search_bounds: 0,
            unknown_selection_targets: Vec::new(),
            member_reasons: Vec::new(),
            remedy,
        })
    };
    if !evidence.rows_independent
        || evidence.block_length.is_some()
        || !(alpha > 0.0 && alpha < 0.5)
    {
        return Err(refuse(
            "discovered_transport.trial_compatibility_invalid",
            "declare an independence-test level in (0,.5) and independent randomized trial rows"
                .into(),
        ));
    }
    if !report.envelope().enumeration_complete() {
        return Err(refuse(
            "discovered_transport.class_enumeration_partial",
            "experimental compatibility cannot certify unenumerated members".into(),
        ));
    }
    let (treatment, outcome) = query.response.functional.primary_pair().ok_or_else(|| {
        refuse(
            "discovered_transport.trial_compatibility_invalid",
            "the query needs one treatment and outcome".into(),
        )
    })?;
    let columns = columns_of(evidence.source, &[treatment, outcome])?;
    if columns[&treatment].len() > 1_000_000 {
        return Err(refuse(
            "discovered_transport.trial_compatibility_invalid",
            "the bounded exact test accepts at most one million source rows".into(),
        ));
    }
    let mut counts = [0u64; 4];
    let mut source_trial_rows = Vec::with_capacity(columns[&treatment].len());
    for (x, y) in columns[&treatment].iter().zip(&columns[&outcome]) {
        let (Some(x), Some(y)) = (*x, *y) else {
            return Err(refuse(
                "discovered_transport.trial_compatibility_invalid",
                "the binary trial has a missing assignment or outcome".into(),
            ));
        };
        let code = |v| {
            if v == 0.0 {
                Some(0)
            } else if v == 1.0 {
                Some(1)
            } else {
                None
            }
        };
        let (Some(x), Some(y)) = (code(x), code(y)) else {
            return Err(refuse(
                "discovered_transport.trial_compatibility_invalid",
                "this exact test requires binary zero/one assignments and outcomes".into(),
            ));
        };
        counts[x * 2 + y] += 1;
        source_trial_rows.push([u8::from(x == 1), u8::from(y == 1)]);
    }
    let p_value = fisher_exact(counts)
        .map_err(|error| DiscoveredTransportError::Estimation(error.to_string()))?;
    let members = report
        .envelope()
        .members()
        .iter()
        .enumerate()
        .map(|(index, member)| {
            let directed_edges: Vec<_> =
                member.directed_edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect();
            let path = has_directed_path(&directed_edges, treatment.raw(), outcome.raw());
            TrialMemberCompatibility {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                directed_edges,
                bidirected_edges: member
                    .bidirected_edges
                    .iter()
                    .map(|(a, b)| (a.raw(), b.raw()))
                    .collect(),
                confounded_edges: member
                    .confounded_edges
                    .iter()
                    .map(|(a, b)| (a.raw(), b.raw()))
                    .collect(),
                structural_verdict: match &member.verdict {
                    MemberVerdict::Identified { rule, .. } => {
                        TrialStructuralVerdict::Identified { rule: rule.clone() }
                    }
                    MemberVerdict::NotCertified { reason } => {
                        TrialStructuralVerdict::NotCertified { reason: reason.clone() }
                    }
                    MemberVerdict::MissingEvidence { reason } => {
                        TrialStructuralVerdict::MissingEvidence { reason: reason.clone() }
                    }
                },
                has_directed_path: path,
                retained: path || p_value > alpha,
            }
        })
        .collect();
    let proof = TrialCompatibilityProof {
        format: PROOF_FORMAT.into(),
        estimation_strategy: SELECTION_ESTIMATION_STRATEGY.into(),
        original_report_identity: report.identity().to_vec(),
        original_structure_identity: *report.structure_identity(),
        original_result_identity: *report.result_identity(),
        original_selection_targets: report.selection_targets().iter().map(|v| v.raw()).collect(),
        original_member_bound: report.envelope().member_bound() as u64,
        original_enumeration_complete: report.envelope().enumeration_complete(),
        treatment: treatment.raw(),
        outcome: outcome.raw(),
        alpha,
        randomization: options.randomization,
        counts,
        source_trial_identity: trial_identity(treatment.raw(), outcome.raw(), &source_trial_rows),
        source_trial_rows,
        p_value,
        members,
    };
    Ok(proof)
}

const CONFIDENCE_SET_FORMAT: &str = "antecedent.discovered_transport.trial_confidence_set.v1";

/// Caller-declared support of every potential outcome in both populations.
/// Observed rows are checked against it; finite observations cannot prove this assumption.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportOutcomeSupport {
    /// Scientific lower support bound, not an empirical minimum.
    pub lower: f64,
    /// Scientific upper support bound, not an empirical maximum.
    pub upper: f64,
}

/// Whether a set uses an identified sampling interval or only logical outcome support.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialConfidenceSetKind {
    /// Every retained member identifies the same functional and a sampling interval exists.
    IdentifiedSamplingInterval,
    /// No point effect is asserted; the set follows from declared outcome support alone.
    LogicalSupportBound,
}

/// Scientific reason for a logical support set. Operational failures never enter this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialLogicalBoundReason {
    /// The declared trial rejected every original no-directed-effect member.
    TrialRejectsEveryMember,
    /// Completed identification did not identify the effect in every retained member.
    ClassNotIdentified,
    /// Identified retained members disagree on their empirical causal functional.
    StructuralRange,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialConfidenceSetWire {
    format: String,
    execution_seed: u64,
    input_identity: [u8; 32],
    support: TransportOutcomeSupport,
    proof: TrialCompatibilityProof,
    kind: TrialConfidenceSetKind,
    reason: Option<TrialLogicalBoundReason>,
    lower: f64,
    upper: f64,
    // Retain the actual existing member artifacts for both point and structural-range results.
    estimate: Option<Vec<u8>>,
}

/// A bounded-outcome whole-procedure set, distinct from a point transport answer.
///
/// A logical set asserts no identified point and is not a successful sampling interval.
/// An identified set retains the existing 95% sampling construction unchanged. Neither
/// branch claims complete-procedure calibration without its own currently attesting record.
#[derive(Clone, Debug)]
pub struct TrialCompatibleTransportConfidenceSet {
    wire: TrialConfidenceSetWire,
    retained_estimate: Option<TrialCompatibleTransportEstimate>,
}

impl TrialCompatibleTransportConfidenceSet {
    /// Versioned identity of this whole-procedure set construction.
    #[must_use]
    pub const fn method(&self) -> &'static str {
        CONFIDENCE_SET_FORMAT
    }
    /// Canonical seal of report, query, support, options, seed and both complete input tables.
    #[must_use]
    pub const fn input_identity(&self) -> &[u8; 32] {
        &self.wire.input_identity
    }
    /// Original bootstrap/execution seed; replay requires the same seed and scientific inputs.
    #[must_use]
    pub const fn execution_seed(&self) -> u64 {
        self.wire.execution_seed
    }
    /// The actual inferential branch; logical sets must not count as point reporting.
    #[must_use]
    pub const fn kind(&self) -> TrialConfidenceSetKind {
        self.wire.kind
    }
    /// The set endpoints for the last-minus-first binary treatment mean contrast.
    #[must_use]
    pub const fn bounds(&self) -> (f64, f64) {
        (self.wire.lower, self.wire.upper)
    }
    /// Scientific fallback reason, absent only for identified sampling intervals.
    #[must_use]
    pub const fn logical_reason(&self) -> Option<TrialLogicalBoundReason> {
        self.wire.reason
    }
    /// Caller-declared potential-outcome support used by this construction.
    #[must_use]
    pub const fn outcome_support(&self) -> TransportOutcomeSupport {
        self.wire.support
    }
    /// Original report, trial observations and all graph compatibility decisions.
    #[must_use]
    pub const fn compatibility(&self) -> &TrialCompatibilityProof {
        &self.wire.proof
    }
    /// Actual original estimate, when the retained class could be estimated.
    /// A structural-range estimate remains a range, not a point.
    #[must_use]
    pub const fn retained_estimate(&self) -> Option<&TrialCompatibleTransportEstimate> {
        self.retained_estimate.as_ref()
    }
    /// This computed set does not automatically bind a whole-procedure coverage record.
    #[must_use]
    pub const fn calibration(&self) -> &'static str {
        "not_bound_to_this_execution"
    }
    /// Canonical identity of the method, inputs, support, branch, proof and retained artifacts.
    ///
    /// # Errors
    /// Artifact encoding failure.
    pub fn identity(&self) -> Result<[u8; 32], IoError> {
        let mut writer = CanonicalWriter::new(CONFIDENCE_SET_FORMAT);
        writer.bytes(&self.export()?);
        Ok(writer.finish())
    }
    /// Export a receipt for replay **against the original report, query and datasets**.
    /// This receipt is not a self-contained data container or a calibration license.
    ///
    /// # Errors
    /// Encoding failure or the shared 16 MiB artifact bound is exceeded.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.export_within(TrialArtifactBudget::default())
    }

    /// [`Self::export`] under a caller-chosen artifact budget.
    ///
    /// # Errors
    /// Encoding failure or `budget.max_bytes` is exceeded.
    pub fn export_within(&self, budget: TrialArtifactBudget) -> Result<Vec<u8>, IoError> {
        let bytes = antecedent_io::to_cbor(&self.wire)?;
        if bytes.len() > budget.max_bytes {
            return Err(IoError::Convert(
                "trial confidence-set artifact byte budget exceeded".into(),
            ));
        }
        Ok(bytes)
    }
}

/// Cancellation, kept distinct from an invalid-evidence failure so a caller can tell them apart.
fn cancelled() -> DiscoveredTransportError {
    DiscoveredTransportError::Identification(antecedent_identify::IdentificationError::Cancelled)
}

fn confidence_set_invalid(message: impl Into<String>) -> DiscoveredTransportError {
    DiscoveredTransportError::Estimation(message.into())
}

// Validate and bind the supported request and ALL supplied rows before scientific fallback.
#[allow(
    clippy::too_many_lines,
    reason = "validate the complete bounded request and bind both supplied evidence tables before fallback"
)]
#[allow(clippy::float_cmp, reason = "the binary contrast and exact row codes define this protocol")]
fn confidence_set_input_identity(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    support: TransportOutcomeSupport,
    ctx: &ExecutionContext,
) -> Result<[u8; 32], DiscoveredTransportError> {
    use antecedent_core::{GridSpec, ResponseFunctional, ResponseQuery};
    use antecedent_data::TableView;
    query.validate().map_err(|e| confidence_set_invalid(e.to_string()))?;
    validate_evidence_query(query)?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let ResponseFunctional::MeanCurve { outcome, treatment } = &query.response.functional else {
        return Err(confidence_set_invalid(
            "confidence sets require a binary treatment mean curve",
        ));
    };
    let GridSpec::Values(levels) = &treatment.grid else {
        return Err(confidence_set_invalid(
            "confidence sets require the explicit binary grid [0,1]",
        ));
    };
    if levels.as_ref() != [0.0, 1.0]
        || query.response != ResponseQuery::new(query.response.functional.clone())
        || !support.lower.is_finite()
        || !support.upper.is_finite()
        || support.lower >= support.upper
        || !(support.upper - support.lower).is_finite()
        || evidence.bootstrap_replicates < crate::result::PERCENTILE_95_MIN_REPLICATES
        || evidence.limits.operations == 0
        || evidence.limits.depth == 0
        || evidence.sid_limits.steps == 0
        || evidence.sid_limits.depth == 0
        || evidence.target_sampling != antecedent_core::TargetSampling::RepresentativeSample
    {
        return Err(confidence_set_invalid(
            "invalid confidence-set request, support, sampling or budget",
        ));
    }
    let mut writer = CanonicalWriter::new(CONFIDENCE_SET_FORMAT);
    writer
        .bytes(report.identity())
        .text(&query.source_population)
        .text(&query.target_population)
        .word(u64::from(treatment.variable.raw()))
        .word(u64::from(outcome.raw()))
        .word(support.lower.to_bits())
        .word(support.upper.to_bits())
        .word(options.alpha.to_bits())
        .text("completely_randomized")
        .word(ctx.rng.master_seed())
        .word(u64::from(evidence.bootstrap_replicates))
        .word(evidence.limits.operations as u64)
        .word(evidence.limits.depth as u64)
        .word(evidence.sid_limits.steps as u64)
        .word(evidence.sid_limits.depth as u64);
    for (population, data) in [("source", evidence.source), ("target", evidence.target)] {
        if data.row_count() > 1_000_000
            || data
                .row_count()
                .checked_mul(data.schema().len())
                .is_none_or(|cells| cells > evidence.limits.operations)
        {
            return Err(confidence_set_invalid("confidence-set input replay budget exceeded"));
        }
        let variables: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let columns = columns_of(data, &variables)?;
        writer.text(population).word(data.row_count() as u64).word(variables.len() as u64);
        if data.row_count() == 0 {
            return Err(confidence_set_invalid("empty confidence-set evidence"));
        }
        for variable in data.schema().variables() {
            writer.word(u64::from(variable.id.raw())).text(&variable.name);
            for value in &columns[&variable.id] {
                let value = value.filter(|v| v.is_finite()).ok_or_else(|| {
                    confidence_set_invalid("confidence-set evidence requires complete finite rows")
                })?;
                if ctx.cancellation.is_cancelled() {
                    return Err(cancelled());
                }
                if value < 0.0 || value.fract() != 0.0 || value >= f64::from(u32::MAX) {
                    return Err(confidence_set_invalid(
                        "confidence-set evidence requires supported nonnegative integer codes",
                    ));
                }
                if variable.id == *outcome && (value < support.lower || value > support.upper) {
                    return Err(confidence_set_invalid(
                        "observed outcome lies outside declared potential-outcome support",
                    ));
                }
                writer.word(value.to_bits());
            }
        }
    }
    Ok(writer.finish())
}

// The legacy summary detail prefers unidentified over unestimable when both occur.
// Permit only completed scientific verdicts, never mixed provider/support/budget failures.
fn scientific_nonidentification(refusal: &DiscoveredTransportRefusal) -> bool {
    refusal.detail == "discovered_transport.class_not_identified"
        && refusal.exhausted_search_bounds == 0
        && refusal.unknown_selection_targets.is_empty()
        && !refusal.member_reasons.is_empty()
        && refusal.member_reasons.iter().all(|(_, reason)| {
            matches!(reason.as_str(), "not_certified" | "proven_non_transportable")
        })
}

/// Compute a binary-trial mean-contrast confidence set with an explicit support fallback.
///
/// The caller asserts finite potential-outcome support `[a,b]` in both populations. All
/// observed outcomes are checked, but this scientific assumption is not learned from rows.
/// Completed scientific nonidentification returns `[a-b,b-a]` with no point assertion.
/// Identified sampling intervals keep the existing bootstrap level and endpoints unchanged.
/// Report point-reporting and set widths separately when calibrating the whole procedure.
///
/// # Errors
/// Invalid/missing data, an unsupported query (this route requires the complete mean curve
/// at explicit treatment levels `[0,1]`), fewer than 40 requested bootstrap draws, invalid support, cancellation,
/// incomplete enumeration, search/evaluation limits, or a member that cannot be estimated.
#[allow(
    clippy::too_many_lines,
    reason = "explicitly distinguish sampling success from the three scientific logical-bound branches"
)]
pub fn estimate_trial_compatible_transport_confidence_set(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    support: TransportOutcomeSupport,
    ctx: &ExecutionContext,
) -> Result<TrialCompatibleTransportConfidenceSet, DiscoveredTransportError> {
    let input_identity =
        confidence_set_input_identity(report, query, evidence, options, support, ctx)?;
    let proof = trial_compatibility_proof(report, query, evidence, options)?;
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    proof
        .validate_replay_budget(evidence.limits, TrialArtifactBudget::default())
        .and_then(|_| proof.validate_with_empty(true))
        .map_err(|e| confidence_set_invalid(e.to_string()))?;
    let retained = proof.retained_members();
    let mut retained_estimate = None;
    let reason = if retained.is_empty() {
        Some(TrialLogicalBoundReason::TrialRejectsEveryMember)
    } else {
        match estimate_selected_discovered_transport(
            report,
            query,
            evidence,
            Some(&retained),
            Some(proof.identity()),
            ctx,
        ) {
            Ok(estimate) => {
                let is_range = estimate.answer().kind
                    == super::discovered_transport_estimate::AnswerKind::StructuralRange;
                if !is_range && estimate.answer().interval.is_none() {
                    return Err(confidence_set_invalid(
                        "identified point has no successful requested sampling interval",
                    ));
                }
                retained_estimate =
                    Some(TrialCompatibleTransportEstimate { estimate, proof: proof.clone() });
                is_range.then_some(TrialLogicalBoundReason::StructuralRange)
            }
            Err(DiscoveredTransportError::Refused(refusal))
                if scientific_nonidentification(&refusal) =>
            {
                Some(TrialLogicalBoundReason::ClassNotIdentified)
            }
            Err(error) => return Err(error),
        }
    };
    let (kind, lower, upper) = if reason.is_some() {
        (
            TrialConfidenceSetKind::LogicalSupportBound,
            support.lower - support.upper,
            support.upper - support.lower,
        )
    } else {
        let interval = retained_estimate
            .as_ref()
            .and_then(|e| e.answer().interval)
            .filter(|i| i.low.is_finite() && i.high.is_finite() && i.low <= i.high)
            .ok_or_else(|| {
                confidence_set_invalid("missing or invalid identified sampling interval")
            })?;
        (TrialConfidenceSetKind::IdentifiedSamplingInterval, interval.low, interval.high)
    };
    let estimate = retained_estimate
        .as_ref()
        .map(TrialCompatibleTransportEstimate::export)
        .transpose()
        .map_err(|e| confidence_set_invalid(e.to_string()))?;
    Ok(TrialCompatibleTransportConfidenceSet {
        wire: TrialConfidenceSetWire {
            format: CONFIDENCE_SET_FORMAT.into(),
            execution_seed: ctx.rng.master_seed(),
            input_identity,
            support,
            proof,
            kind,
            reason,
            lower,
            upper,
            estimate,
        },
        retained_estimate,
    })
}

/// Verify a set receipt **against the original report, query, datasets and execution seed**.
///
/// Validates the independent Fisher/graph proof and, when present, retained member-grid
/// artifacts; then reruns the bounded endpoint against the supplied inputs, including the
/// original bootstrap, so neither a forged refusal reason nor substituted interval can pass.
/// The datasets are not embedded. This operation grants no calibration authority.
///
/// # Errors
/// An artifact above 16 MiB, incompatible inputs/support/options/seed, any tampered result,
/// or the same cancellation/resource errors as estimation. Supplied evidence limits apply.
#[allow(
    clippy::too_many_arguments,
    reason = "replay requires the complete original scientific request and caller bounds"
)]
pub fn verify_trial_compatible_transport_confidence_set_against(
    bytes: &[u8],
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    support: TransportOutcomeSupport,
    ctx: &ExecutionContext,
) -> Result<TrialCompatibleTransportConfidenceSet, IoError> {
    verify_trial_compatible_transport_confidence_set_against_within(
        bytes,
        report,
        query,
        evidence,
        options,
        support,
        TrialArtifactBudget::default(),
        ctx,
    )
}

/// [`verify_trial_compatible_transport_confidence_set_against`] under a caller-chosen
/// artifact budget for the receipt and its embedded estimate.
///
/// # Errors
/// As [`verify_trial_compatible_transport_confidence_set_against`], with the artifact bytes
/// and original members bounded by `budget`.
#[allow(
    clippy::too_many_arguments,
    reason = "replay requires the complete original scientific request, caller bounds and decode budget"
)]
pub fn verify_trial_compatible_transport_confidence_set_against_within(
    bytes: &[u8],
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    support: TransportOutcomeSupport,
    budget: TrialArtifactBudget,
    ctx: &ExecutionContext,
) -> Result<TrialCompatibleTransportConfidenceSet, IoError> {
    let bad = || {
        IoError::Convert(
            "trial confidence-set receipt does not replay against supplied inputs".into(),
        )
    };
    if bytes.len() > budget.max_bytes {
        return Err(bad());
    }
    let wire: TrialConfidenceSetWire = antecedent_io::from_cbor(bytes)?;
    if wire.format != CONFIDENCE_SET_FORMAT {
        return Err(bad());
    }
    let input = confidence_set_input_identity(report, query, evidence, options, support, ctx)
        .map_err(|_| bad())?;
    if wire.input_identity != input || wire.support != support {
        return Err(bad());
    }
    wire.proof.validate_replay_budget(evidence.limits, budget)?;
    wire.proof.validate_with_empty(true)?;
    if let Some(estimate) = &wire.estimate {
        let (proof, _) = verify_trial_compatible_transport_estimate_within(
            estimate,
            evidence.limits,
            budget,
            ctx,
        )?;
        if proof != wire.proof {
            return Err(bad());
        }
    }
    let replay = estimate_trial_compatible_transport_confidence_set(
        report, query, evidence, options, support, ctx,
    )
    .map_err(|_| bad())?;
    if replay.wire != wire || replay.export_within(budget)?.as_slice() != bytes {
        return Err(bad());
    }
    Ok(replay)
}

fn has_directed_path(edges: &[(u32, u32)], treatment: u32, outcome: u32) -> bool {
    let mut children: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (from, to) in edges {
        children.entry(*from).or_default().push(*to);
    }
    let mut reached = std::collections::BTreeSet::from([treatment]);
    let mut stack = vec![treatment];
    while let Some(node) = stack.pop() {
        if node == outcome {
            return true;
        }
        for child in children.get(&node).into_iter().flatten() {
            if reached.insert(*child) {
                stack.push(*child);
            }
        }
    }
    false
}

fn trial_identity(treatment: u32, outcome: u32, rows: &[[u8; 2]]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.discovered_transport.binary_trial.v1");
    writer.word(u64::from(treatment)).word(u64::from(outcome)).word(rows.len() as u64);
    for [x, y] in rows {
        writer.word(u64::from(*x)).word(u64::from(*y));
    }
    writer.finish()
}

/// Relative tolerance with which a table counts as exactly as probable as the observed one.
///
/// Masses are generated from the mode by the hypergeometric ratio recurrence, each step of
/// which carries a few ulps of relative error: at the one-million-row capacity the
/// accumulated relative error stays below `5e-10`. Two masses within `1e-7` of each other are
/// therefore indistinguishable at this precision, and counting them as ties errs on the side
/// of the conservative (larger) p-value, as exact-arithmetic ties such as a symmetric table
/// require.
const TIE_RELATIVE_SLACK: f64 = 1e-7;

/// A contingency table the exact test cannot evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InvalidContingencyTable;

impl std::fmt::Display for InvalidContingencyTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid binary trial contingency table")
    }
}

impl std::error::Error for InvalidContingencyTable {}

impl From<InvalidContingencyTable> for IoError {
    fn from(error: InvalidContingencyTable) -> Self {
        Self::Convert(error.to_string())
    }
}

/// Two-sided Fisher exact p-value of the 2x2 table `[[a, b], [c, d]]` flattened row-major.
///
/// The probability of each table with the observed margins is proportional to
/// `C(K, k) C(N - K, draws - k)`. Rather than evaluating each through log-gamma differences,
/// the masses are generated relative to the mode (mass one there) by the exact ratio
/// recurrence `m(k + 1) / m(k) = (K - k)(draws - k) / ((k + 1)(N - K - draws + k + 1))`, whose
/// factors are integers below `2^53` here. The p-value is the compensated sum of the masses
/// not exceeding the observed one (within [`TIE_RELATIVE_SLACK`]) over the compensated sum of
/// all masses, so no normalising constant is evaluated. Masses far from the mode underflow to
/// zero, which only drops terms below the working precision.
#[allow(
    clippy::cast_precision_loss,
    reason = "all counts are bounded by one million and exactly represented as f64"
)]
fn fisher_exact(counts: [u64; 4]) -> Result<f64, InvalidContingencyTable> {
    use antecedent_kernels::numeric::CompensatedSum;
    let n = counts
        .iter()
        .try_fold(0u64, |sum, c| sum.checked_add(*c))
        .ok_or(InvalidContingencyTable)?;
    if n == 0 || n > 1_000_000 {
        return Err(InvalidContingencyTable);
    }
    let draws = counts[2] + counts[3];
    let successes = counts[1] + counts[3];
    let lo = draws.saturating_sub(n - successes);
    let hi = draws.min(successes);
    let mode = (((draws + 1) * (successes + 1)) / (n + 2)).clamp(lo, hi);
    // Mass of the table with `k + 1` successes relative to the table with `k`.
    let up = |k: u64| {
        ((successes - k) as f64 * (draws - k) as f64)
            / ((k + 1) as f64 * (n - successes + k + 1 - draws) as f64)
    };
    let observed_k = counts[3];
    let mut observed = 1.0_f64;
    if observed_k >= mode {
        for k in mode..observed_k {
            observed *= up(k);
        }
    } else {
        for k in observed_k..mode {
            observed /= up(k);
        }
    }
    let threshold = observed * (1.0 + TIE_RELATIVE_SLACK);
    let mut total = CompensatedSum::default();
    let mut tail = CompensatedSum::default();
    let mut take = |mass: f64| {
        total.add(mass);
        if mass <= threshold {
            tail.add(mass);
        }
    };
    let mut mass = 1.0_f64;
    take(mass);
    for k in mode..hi {
        mass *= up(k);
        take(mass);
    }
    mass = 1.0;
    for k in (lo..mode).rev() {
        mass /= up(k);
        take(mass);
    }
    Ok((tail.total() / total.total()).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_discovery::selective::{CiTestSpec, StaticAlgorithm, StaticDiscoveryContract};
    use std::sync::Arc;

    fn confidence_fixture(
        independent_discovery: bool,
    ) -> (DiscoveredTransportReport, TransportQuery, TabularData, ExecutionContext) {
        let x: Vec<f64> = (0..200).map(|i| f64::from(i >= 100)).collect();
        let y: Vec<f64> = (0..200).map(|i| f64::from(i < 20 || (100..180).contains(&i))).collect();
        let source =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let independent: Vec<f64> = (0..200).map(|i| f64::from(i % 2 == 0)).collect();
        let discovery =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", independent.as_slice())])
                .unwrap();
        let treatment = VariableId::from_raw(0);
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [treatment],
        );
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            vec![treatment, VariableId::from_raw(1)],
            CiTestSpec::declared(
                "gsquared_analytic",
                Arc::new(antecedent_stats::ci::GSquared::new()),
            ),
            71,
        );
        let ctx = ExecutionContext::for_tests(71);
        let report = super::super::discovered_transport::discover_and_transport(
            &contract,
            if independent_discovery { &discovery } else { &source },
            &[],
            &query,
            32,
            &ctx,
        )
        .unwrap()
        .transport
        .unwrap();
        (report, query, source, ctx)
    }
    fn confidence_options() -> TrialCompatibilityOptions {
        TrialCompatibilityOptions {
            alpha: 0.01,
            randomization: TrialRandomization::CompletelyRandomized,
        }
    }
    fn binary_support() -> TransportOutcomeSupport {
        TransportOutcomeSupport { lower: 0.0, upper: 1.0 }
    }

    fn bound_bits(set: &TrialCompatibleTransportConfidenceSet) -> [u64; 2] {
        let (low, high) = set.bounds();
        [low.to_bits(), high.to_bits()]
    }

    #[test]
    fn confidence_set_preserves_the_real_identified_sampling_interval_and_replays() {
        let (report, query, source, ctx) = confidence_fixture(false);
        let mut evidence = TransportEvidence::new(&source, &source);
        evidence.bootstrap_replicates = 64;
        let original = estimate_trial_compatible_discovered_transport(
            &report,
            &query,
            &evidence,
            confidence_options(),
            &ctx,
        )
        .unwrap();
        let set = estimate_trial_compatible_transport_confidence_set(
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
        assert_eq!(set.kind(), TrialConfidenceSetKind::IdentifiedSamplingInterval);
        let interval = original.answer().interval.unwrap();
        assert_eq!(bound_bits(&set), [interval.low.to_bits(), interval.high.to_bits()]);
        assert_eq!(set.retained_estimate().unwrap().answer(), original.answer());
        assert_eq!(set.calibration(), "not_bound_to_this_execution");
        let replay = verify_trial_compatible_transport_confidence_set_against(
            &set.export().unwrap(),
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
        assert_eq!(replay.identity().unwrap(), set.identity().unwrap());
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one real logical-bound receipt is challenged against every independent input and wire component"
    )]
    fn rejected_class_returns_declared_support_not_a_point_or_empirical_range() {
        let (report, query, source, ctx) = confidence_fixture(true);
        let mut evidence = TransportEvidence::new(&source, &source);
        evidence.bootstrap_replicates = 64;
        assert!(
            estimate_trial_compatible_discovered_transport(
                &report,
                &query,
                &evidence,
                confidence_options(),
                &ctx
            )
            .is_err()
        );
        let support = TransportOutcomeSupport { lower: -2.0, upper: 3.0 };
        let set = estimate_trial_compatible_transport_confidence_set(
            &report,
            &query,
            &evidence,
            confidence_options(),
            support,
            &ctx,
        )
        .unwrap();
        assert_eq!(set.kind(), TrialConfidenceSetKind::LogicalSupportBound);
        assert_eq!(set.logical_reason(), Some(TrialLogicalBoundReason::TrialRejectsEveryMember));
        assert_eq!(bound_bits(&set), [(-5.0_f64).to_bits(), 5.0_f64.to_bits()]);
        assert!(set.retained_estimate().is_none());
        assert!(set.compatibility().retained_members().is_empty());
        // Target outcomes need not be supplied for a valid logical bound.
        let target_x: Vec<f64> = (0..200).map(|i| f64::from(i >= 100)).collect();
        let target = TabularData::from_f64_columns([("x", target_x.as_slice())]).unwrap();
        let target_covariates_only = TransportEvidence { target: &target, ..evidence };
        let without_target_outcomes = estimate_trial_compatible_transport_confidence_set(
            &report,
            &query,
            &target_covariates_only,
            confidence_options(),
            support,
            &ctx,
        )
        .unwrap();
        assert_eq!(without_target_outcomes.kind(), TrialConfidenceSetKind::LogicalSupportBound);
        let bytes = set.export().unwrap();
        verify_trial_compatible_transport_confidence_set_against(
            &bytes,
            &report,
            &query,
            &evidence,
            confidence_options(),
            support,
            &ctx,
        )
        .unwrap();
        for which in 0..8 {
            let mut altered: TrialConfidenceSetWire = antecedent_io::from_cbor(&bytes).unwrap();
            match which {
                0 => altered.lower = -4.0,
                1 => altered.reason = Some(TrialLogicalBoundReason::ClassNotIdentified),
                2 => altered.kind = TrialConfidenceSetKind::IdentifiedSamplingInterval,
                3 => altered.format.push_str(".forged"),
                4 => altered.proof.members[0].directed_edges.push((0, 1)),
                5 => altered.support.lower = -1.0,
                6 => altered.proof.original_enumeration_complete = false,
                _ => altered.input_identity[0] ^= 1,
            }
            assert!(
                verify_trial_compatible_transport_confidence_set_against(
                    &antecedent_io::to_cbor(&altered).unwrap(),
                    &report,
                    &query,
                    &evidence,
                    confidence_options(),
                    support,
                    &ctx
                )
                .is_err()
            );
        }
        let (other_report, _, _, _) = confidence_fixture(false);
        assert!(
            verify_trial_compatible_transport_confidence_set_against(
                &bytes,
                &other_report,
                &query,
                &evidence,
                confidence_options(),
                support,
                &ctx
            )
            .is_err()
        );
        let mut other_query = query.clone();
        other_query.target_population = "another_target".into();
        assert!(
            verify_trial_compatible_transport_confidence_set_against(
                &bytes,
                &report,
                &other_query,
                &evidence,
                confidence_options(),
                support,
                &ctx
            )
            .is_err()
        );
        let wrong_seed = ExecutionContext::for_tests(72);
        assert!(
            verify_trial_compatible_transport_confidence_set_against(
                &bytes,
                &report,
                &query,
                &evidence,
                confidence_options(),
                support,
                &wrong_seed
            )
            .is_err()
        );
        // Same schema and support, different actual source/target data must not replay.
        let x = vec![0.0; 200];
        let y = vec![1.0; 200];
        let changed =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        for changed_evidence in [
            TransportEvidence { source: &changed, ..evidence },
            TransportEvidence { target: &changed, ..evidence },
        ] {
            assert!(
                verify_trial_compatible_transport_confidence_set_against(
                    &bytes,
                    &report,
                    &query,
                    &changed_evidence,
                    confidence_options(),
                    support,
                    &ctx
                )
                .is_err()
            );
        }
    }

    #[test]
    fn structural_range_is_retained_without_becoming_an_identified_point() {
        // Observational discovery returns X-Z-Y with no compelled collider. In a target
        // whose Z mechanism shifts, X->Z->Y and Z->X,Z->Y have different do(X) laws.
        // The source X trial has no marginal effect, so Fisher retains both possibilities.
        let table = |cell_counts: [usize; 4]| {
            let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
            for (cell, count) in cell_counts.into_iter().enumerate() {
                let treatment = i32::try_from(cell / 2).unwrap();
                let shift = i32::try_from(cell % 2).unwrap();
                for row in 0..count {
                    x.push(f64::from(treatment));
                    z.push(f64::from(shift));
                    y.push(f64::from(row < count * (if shift == 0 { 1 } else { 4 }) / 5));
                }
            }
            TabularData::from_f64_columns([
                ("x", x.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
            ])
            .unwrap()
        };
        // Exact factorization with ample counts in every CI conditioning cell.
        let discovery = table([800, 200, 200, 800]);
        let source = table([100, 100, 100, 100]);
        let target = table([20, 80, 80, 20]);
        let (_, query, _, ctx) = confidence_fixture(false);
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            (0..3).map(VariableId::from_raw).collect(),
            CiTestSpec::declared(
                "gsquared_analytic",
                Arc::new(antecedent_stats::ci::GSquared::new()),
            ),
            71,
        );
        let report = super::super::discovered_transport::discover_and_transport(
            &contract,
            &discovery,
            &[VariableId::from_raw(2)],
            &query,
            64,
            &ctx,
        )
        .unwrap()
        .transport
        .unwrap();
        let mut evidence = TransportEvidence::new(&source, &target);
        evidence.bootstrap_replicates = 64;
        let set = estimate_trial_compatible_transport_confidence_set(
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
        assert_eq!(set.logical_reason(), Some(TrialLogicalBoundReason::StructuralRange));
        assert_eq!(bound_bits(&set), [(-1.0_f64).to_bits(), 1.0_f64.to_bits()]);
        let answer = set.retained_estimate().unwrap().answer();
        assert_eq!(
            answer.kind,
            super::super::discovered_transport_estimate::AnswerKind::StructuralRange
        );
        assert!(answer.levels[0].low < answer.levels[0].high);
        assert!(answer.interval.is_none());
        verify_trial_compatible_transport_confidence_set_against(
            &set.export().unwrap(),
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
    }

    #[test]
    fn confidence_set_support_requests_and_resources_fail_closed() {
        let (report, query, source, ctx) = confidence_fixture(true);
        let mut evidence = TransportEvidence::new(&source, &source);
        evidence.bootstrap_replicates = 64;
        for support in [
            TransportOutcomeSupport { lower: 0.0, upper: 0.5 },
            TransportOutcomeSupport { lower: 1.0, upper: 0.0 },
            TransportOutcomeSupport { lower: f64::NEG_INFINITY, upper: 1.0 },
            TransportOutcomeSupport { lower: -f64::MAX, upper: f64::MAX },
        ] {
            assert!(
                estimate_trial_compatible_transport_confidence_set(
                    &report,
                    &query,
                    &evidence,
                    confidence_options(),
                    support,
                    &ctx
                )
                .is_err()
            );
        }
        let mut no_boot = evidence;
        no_boot.bootstrap_replicates = 0;
        assert!(
            estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &no_boot,
                confidence_options(),
                binary_support(),
                &ctx
            )
            .is_err()
        );
        let mut no_work = evidence;
        no_work.limits.operations = 0;
        assert!(
            estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &no_work,
                confidence_options(),
                binary_support(),
                &ctx
            )
            .is_err()
        );
        let mut no_sid = evidence;
        no_sid.sid_limits.steps = 0;
        assert!(
            estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &no_sid,
                confidence_options(),
                binary_support(),
                &ctx
            )
            .is_err()
        );
        let cancelled = ExecutionContext::for_tests(71);
        cancelled.cancellation.cancel();
        assert!(
            estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &evidence,
                confidence_options(),
                binary_support(),
                &cancelled
            )
            .is_err()
        );
        assert!(
            verify_trial_compatible_transport_confidence_set_against(
                &vec![0; MAX_TRIAL_ARTIFACT_BYTES + 1],
                &report,
                &query,
                &evidence,
                confidence_options(),
                binary_support(),
                &ctx
            )
            .is_err()
        );
    }

    #[test]
    fn selected_latent_cause_returns_a_scientific_set_after_completed_identification() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for treatment in 0..2 {
            for shift in 0..2 {
                for row in 0..100 {
                    x.push(f64::from(treatment));
                    z.push(f64::from(shift));
                    y.push(f64::from(row < 10 + 30 * treatment + 20 * shift));
                }
            }
        }
        let source = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let (_, query, _, ctx) = confidence_fixture(false);
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            (0..3).map(VariableId::from_raw).collect(),
            CiTestSpec::declared(
                "gsquared_analytic",
                Arc::new(antecedent_stats::ci::GSquared::new()),
            ),
            71,
        );
        let report = super::super::discovered_transport::discover_and_transport(
            &contract,
            &source,
            &[VariableId::from_raw(2)],
            &query,
            64,
            &ctx,
        )
        .unwrap()
        .transport
        .unwrap();
        let mut evidence = TransportEvidence::new(&source, &source);
        evidence.bootstrap_replicates = 64;
        let error = estimate_trial_compatible_discovered_transport(
            &report,
            &query,
            &evidence,
            confidence_options(),
            &ctx,
        )
        .unwrap_err();
        let DiscoveredTransportError::Refused(refusal) = error else {
            panic!("expected scientific refusal")
        };
        assert!(scientific_nonidentification(&refusal));
        let set = estimate_trial_compatible_transport_confidence_set(
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
        assert_eq!(set.logical_reason(), Some(TrialLogicalBoundReason::ClassNotIdentified));
        assert_eq!(set.kind(), TrialConfidenceSetKind::LogicalSupportBound);
        assert!(set.retained_estimate().is_none());
        verify_trial_compatible_transport_confidence_set_against(
            &set.export().unwrap(),
            &report,
            &query,
            &evidence,
            confidence_options(),
            binary_support(),
            &ctx,
        )
        .unwrap();
        let mut small = evidence;
        small.sid_limits.steps = 1;
        assert!(
            estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &small,
                confidence_options(),
                binary_support(),
                &ctx
            )
            .is_err()
        );
    }

    #[test]
    fn minimum_bootstrap_request_is_enforced_before_both_scientific_branches() {
        for independent_discovery in [false, true] {
            let (report, query, source, ctx) = confidence_fixture(independent_discovery);
            let mut evidence = TransportEvidence::new(&source, &source);
            evidence.bootstrap_replicates = 39;
            assert!(
                estimate_trial_compatible_transport_confidence_set(
                    &report,
                    &query,
                    &evidence,
                    confidence_options(),
                    binary_support(),
                    &ctx
                )
                .is_err()
            );
            evidence.bootstrap_replicates = 40;
            let set = estimate_trial_compatible_transport_confidence_set(
                &report,
                &query,
                &evidence,
                confidence_options(),
                binary_support(),
                &ctx,
            )
            .unwrap();
            assert_eq!(
                set.kind(),
                if independent_discovery {
                    TrialConfidenceSetKind::LogicalSupportBound
                } else {
                    TrialConfidenceSetKind::IdentifiedSamplingInterval
                }
            );
        }
    }

    #[test]
    fn mixed_missing_evidence_is_not_scientific_nonidentification() {
        let refusal = |reasons: Vec<&str>| DiscoveredTransportRefusal {
            code: antecedent_core::reason_code!("discovered_structure_unresolved"),
            detail: "discovered_transport.class_not_identified",
            unresolved: Vec::new(),
            exhausted_search_bounds: 0,
            unknown_selection_targets: Vec::new(),
            member_reasons: reasons.into_iter().enumerate().map(|(i, r)| (i, r.into())).collect(),
            remedy: String::new(),
        };
        assert!(scientific_nonidentification(&refusal(vec![
            "not_certified",
            "proven_non_transportable"
        ])));
        assert!(!scientific_nonidentification(&refusal(vec![])));
        assert!(!scientific_nonidentification(&refusal(vec![
            "not_certified",
            "no formula binds to the supplied treatment experiment"
        ])));
        assert!(!scientific_nonidentification(&refusal(vec![
            "proven_non_transportable",
            "evaluation budget exceeded"
        ])));
    }

    #[test]
    fn a_real_trial_refines_an_ambiguous_class_and_replays_every_decision() {
        let x: Vec<f64> = (0..200).map(|i| f64::from(i >= 100)).collect();
        let y: Vec<f64> = (0..200).map(|i| f64::from(i < 20 || (100..180).contains(&i))).collect();
        let source =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [treatment],
        );
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            vec![treatment, outcome],
            CiTestSpec::declared(
                "gsquared_analytic",
                Arc::new(antecedent_stats::ci::GSquared::new()),
            ),
            71,
        );
        let ctx = ExecutionContext::for_tests(71);
        let run = super::super::discovered_transport::discover_and_transport(
            &contract,
            &source,
            &[],
            &query,
            32,
            &ctx,
        )
        .unwrap();
        let report = run.transport.unwrap();
        assert!(report.envelope().members().len() > 1);
        let before = *report.identity();
        let evidence = TransportEvidence::new(&source, &source);
        let refined = estimate_trial_compatible_discovered_transport(
            &report,
            &query,
            &evidence,
            TrialCompatibilityOptions {
                alpha: 0.01,
                randomization: TrialRandomization::CompletelyRandomized,
            },
            &ctx,
        )
        .unwrap();
        assert_eq!(report.identity(), &before);
        assert!(refined.compatibility().members.iter().any(|m| !m.retained));
        assert!(refined.compatibility().p_value < 0.01);
        assert_eq!(
            refined.answer().kind,
            super::super::discovered_transport_estimate::AnswerKind::Point
        );
        assert!((refined.answer().levels[0].low - 0.2).abs() < 1e-12);
        assert!((refined.answer().levels[1].low - 0.8).abs() < 1e-12);
        let bytes = refined.export().unwrap();
        let (proof, verified) =
            verify_trial_compatible_transport_estimate(&bytes, evidence.limits, &ctx).unwrap();
        assert_eq!(verified.members, proof.retained_members());
        assert_eq!(proof, refined.proof);
        assert_eq!(proof.estimation_strategy, SELECTION_ESTIMATION_STRATEGY);
        let mut changed_strategy = proof.clone();
        changed_strategy.estimation_strategy = LEGACY_ESTIMATION_STRATEGY.into();
        assert_ne!(changed_strategy.identity(), proof.identity());
        changed_strategy.estimation_strategy = "unchecked_selection_union".into();
        assert!(changed_strategy.validate().is_err());
        assert_trial_replay_budgets(&bytes, evidence.limits, &ctx);
        assert_trial_tampering_rejected(&bytes, evidence.limits, &ctx);
    }

    fn assert_trial_tampering_rejected(
        bytes: &[u8],
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) {
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(bytes).unwrap();
        tampered.proof.counts = [u64::MAX; 4];
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                limits,
                ctx,
            )
            .is_err()
        );
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(bytes).unwrap();
        tampered.proof.source_trial_rows[0][1] ^= 1;
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                limits,
                ctx
            )
            .is_err()
        );
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(bytes).unwrap();
        tampered.proof.members[0].retained = !tampered.proof.members[0].retained;
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                limits,
                ctx
            )
            .is_err()
        );
        // Even a change to a rejected member's latent graph must remain bound to the
        // original report, rather than merely passing the directed-path decision again.
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(bytes).unwrap();
        tampered.proof.members[0].bidirected_edges.push((0, 1));
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                limits,
                ctx
            )
            .is_err()
        );
    }

    fn assert_trial_replay_budgets(
        bytes: &[u8],
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) {
        let no_work = ExactEvaluationLimits { operations: 0, ..limits };
        assert!(
            verify_trial_compatible_transport_estimate(bytes, no_work, ctx)
                .unwrap_err()
                .to_string()
                .contains("replay budget")
        );
        let mut too_many: TrialEstimateWire = antecedent_io::from_cbor(bytes).unwrap();
        too_many.proof.members =
            vec![too_many.proof.members[0].clone(); MAX_TRIAL_ARTIFACT_MEMBERS + 1];
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&too_many).unwrap(),
                limits,
                ctx,
            )
            .unwrap_err()
            .to_string()
            .contains("replay budget")
        );
    }
    #[test]
    fn trial_consumer_checks_byte_budget_before_decoding() {
        let error = verify_trial_compatible_transport_estimate(
            &vec![0; MAX_TRIAL_ARTIFACT_BYTES + 1],
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("byte budget"));
    }
    #[test]
    fn a_smaller_caller_budget_refuses_before_decoding() {
        let budget = TrialArtifactBudget { max_bytes: 8, max_members: 1 };
        let error = verify_trial_compatible_transport_estimate_within(
            &[0; 9],
            ExactEvaluationLimits::default(),
            budget,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("byte budget"));
        assert_eq!(TrialArtifactBudget::default().max_bytes, MAX_TRIAL_ARTIFACT_BYTES);
    }
    #[test]
    fn fisher_exact_matches_hand_enumerated_margin_conditioning() {
        // Margins (2,2),(2,2): masses 1/6,4/6,1/6. Extreme two-sided p=1/3.
        assert!((fisher_exact([2, 0, 0, 2]).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((fisher_exact([1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(fisher_exact([10, 0, 10, 0]).unwrap().to_bits(), 1.0_f64.to_bits());
        assert!(fisher_exact([0, 0, 0, 0]).is_err());
    }

    #[test]
    fn fisher_exact_is_stable_at_the_row_capacity() {
        let balanced = fisher_exact([250_000, 250_000, 250_000, 250_000]).unwrap();
        assert!((balanced - 1.0).abs() < 1e-9, "{balanced}");
        let separated = fisher_exact([500_000, 0, 0, 500_000]).unwrap();
        assert!((0.0..1e-12).contains(&separated), "{separated}");
        let skewed = fisher_exact([499_000, 1_000, 1_000, 499_000]).unwrap();
        assert!((0.0..1e-12).contains(&skewed), "{skewed}");
        assert!(fisher_exact([1_000_000, 1, 0, 0]).is_err());
    }

    #[test]
    fn fisher_exact_matches_integer_mass_enumeration_for_every_small_table() {
        fn integer_binomial_for_trial_oracle(n: u64, k: u64) -> u64 {
            (0..k.min(n - k)).fold(1, |value, i| value * (n - i) / (i + 1))
        }
        for n in 1..=12 {
            for a in 0..=n {
                for b in 0..=n - a {
                    for c in 0..=n - a - b {
                        let d = n - a - b - c;
                        let successes = b + d;
                        let draws = c + d;
                        let mass = |k| {
                            integer_binomial_for_trial_oracle(successes, k)
                                * integer_binomial_for_trial_oracle(n - successes, draws - k)
                        };
                        let observed = mass(d);
                        let numerator: u64 = (draws.saturating_sub(n - successes)
                            ..=draws.min(successes))
                            .map(mass)
                            .filter(|weight| *weight <= observed)
                            .sum();
                        let expected = f64::from(u32::try_from(numerator).unwrap())
                            / f64::from(
                                u32::try_from(integer_binomial_for_trial_oracle(n, draws)).unwrap(),
                            );
                        assert!(
                            (fisher_exact([a, b, c, d]).unwrap() - expected).abs() < 1e-11,
                            "table {a},{b},{c},{d}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn directed_path_never_uses_a_common_cause_as_a_treatment_effect() {
        assert!(!has_directed_path(&[(2, 0), (2, 1)], 0, 1));
        assert!(has_directed_path(&[(0, 2), (2, 1)], 0, 1));
    }
}
