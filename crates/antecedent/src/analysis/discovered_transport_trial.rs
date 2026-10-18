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
    DISCOVERED_TRANSPORT_REASON, DiscoveredTransportError, DiscoveredTransportRefusal,
    DiscoveredTransportReport, MemberVerdict,
};
use super::discovered_transport_estimate::{
    DiscoveredTransportEstimate, TransportAnswer, TransportEvidence, VerifiedEstimate, columns_of,
    estimate_selected_discovered_transport, validate_evidence_query,
    verify_discovered_transport_estimate,
};

const PROOF_FORMAT: &str = "antecedent.discovered_transport.trial_compatibility.v1";
const ESTIMATE_FORMAT: &str = "antecedent.discovered_transport.trial_estimate.v1";

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
        let bad =
            || IoError::Convert("discovered trial compatibility proof does not replay".into());
        if self.format != PROOF_FORMAT
            || self.original_report_identity.len() != 32
            || self.original_report_identity.as_slice()
                != self.replay_original_report_identity().as_slice()
            || !self.original_enumeration_complete
            || !(self.alpha > 0.0 && self.alpha < 0.5)
            || self.treatment == self.outcome
            || self.members.is_empty()
            || self.source_trial_rows.len() > 1_000_000
            || fisher_exact(self.counts)? != self.p_value
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
        for (index, member) in self.members.iter().enumerate() {
            let path = has_directed_path(&member.directed_edges, self.treatment, self.outcome);
            if member.index as usize != index
                || member.has_directed_path != path
                || member.retained != (path || self.p_value > self.alpha)
            {
                return Err(bad());
            }
        }
        if self.retained_members().is_empty() {
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
///
/// # Errors
/// A test, decision, refined identity or retained grid that does not replay.
pub fn verify_trial_compatible_transport_estimate(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(TrialCompatibilityProof, VerifiedEstimate), IoError> {
    let wire: TrialEstimateWire = antecedent_io::from_cbor(bytes)?;
    if wire.format != ESTIMATE_FORMAT {
        return Err(IoError::Convert("unsupported trial estimate format".into()));
    }
    wire.proof.validate()?;
    let mut verified = verify_discovered_transport_estimate(&wire.estimate, limits, ctx)?;
    let retained = wire.proof.retained_members();
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
#[allow(
    clippy::float_cmp,
    reason = "trial inputs must be exact binary zero/one codes"
)]
pub fn estimate_trial_compatible_discovered_transport(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    options: TrialCompatibilityOptions,
    ctx: &ExecutionContext,
) -> Result<TrialCompatibleTransportEstimate, DiscoveredTransportError> {
    let alpha = options.alpha;
    validate_evidence_query(query)?;
    let refuse = |detail, remedy: String| {
        DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
            code: DISCOVERED_TRANSPORT_REASON,
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
    let p_value =
        fisher_exact(counts).map_err(|e| DiscoveredTransportError::Estimation(e.to_string()))?;
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
    let retained = proof.retained_members();
    if retained.is_empty() {
        return Err(refuse("discovered_transport.trial_rejects_every_member", "the declared randomized trial rejects the common no-effect implication of every discovered member; revisit discovery rather than invent an edge".into()));
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

fn has_directed_path(edges: &[(u32, u32)], treatment: u32, outcome: u32) -> bool {
    let mut reached = std::collections::BTreeSet::from([treatment]);
    loop {
        let before = reached.len();
        for (from, to) in edges {
            if reached.contains(from) {
                reached.insert(*to);
            }
        }
        if reached.contains(&outcome) {
            return true;
        }
        if reached.len() == before {
            return false;
        }
    }
}

fn trial_identity(treatment: u32, outcome: u32, rows: &[[u8; 2]]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.discovered_transport.binary_trial.v1");
    writer.word(u64::from(treatment)).word(u64::from(outcome)).word(rows.len() as u64);
    for [x, y] in rows {
        writer.word(u64::from(*x)).word(u64::from(*y));
    }
    writer.finish()
}

#[allow(
    clippy::cast_precision_loss,
    reason = "all counts are bounded by one million and exactly represented as f64"
)]
fn fisher_exact(counts: [u64; 4]) -> Result<f64, IoError> {
    let invalid = || IoError::Convert("invalid binary trial contingency table".into());
    let n = counts.iter().try_fold(0u64, |sum, c| sum.checked_add(*c)).ok_or_else(invalid)?;
    if n == 0 || n > 1_000_000 {
        return Err(invalid());
    }
    let draws = counts[2] + counts[3];
    let successes = counts[1] + counts[3];
    let choose = |n: u64, k: u64| {
        antecedent_stats::ln_gamma(n as f64 + 1.0)
            - antecedent_stats::ln_gamma(k as f64 + 1.0)
            - antecedent_stats::ln_gamma((n - k) as f64 + 1.0)
    };
    let log_probability =
        |k| choose(successes, k) + choose(n - successes, draws - k) - choose(n, draws);
    let observed = log_probability(counts[3]);
    let lo = draws.saturating_sub(n - successes);
    let hi = draws.min(successes);
    let mut probability = 0.0;
    for k in lo..=hi {
        let log_p = log_probability(k);
        // Include numerical ties conservatively even at the one-million-row capacity.
        if log_p <= observed + 1e-7 {
            probability += log_p.exp();
        }
    }
    Ok(probability.min(1.0))
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
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(&bytes).unwrap();
        tampered.proof.source_trial_rows[0][1] ^= 1;
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                evidence.limits,
                &ctx
            )
            .is_err()
        );
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(&bytes).unwrap();
        tampered.proof.members[0].retained = !tampered.proof.members[0].retained;
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                evidence.limits,
                &ctx
            )
            .is_err()
        );
        // Even a change to a rejected member's latent graph must remain bound to the
        // original report, rather than merely passing the directed-path decision again.
        let mut tampered: TrialEstimateWire = antecedent_io::from_cbor(&bytes).unwrap();
        tampered.proof.members[0].bidirected_edges.push((0, 1));
        assert!(
            verify_trial_compatible_transport_estimate(
                &antecedent_io::to_cbor(&tampered).unwrap(),
                evidence.limits,
                &ctx
            )
            .is_err()
        );
    }
    #[test]
    fn fisher_exact_matches_hand_enumerated_margin_conditioning() {
        // Margins (2,2),(2,2): masses 1/6,4/6,1/6. Extreme two-sided p=1/3.
        assert!((fisher_exact([2, 0, 0, 2]).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((fisher_exact([1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(fisher_exact([10, 0, 10, 0]).unwrap(), 1.0);
        assert!(fisher_exact([0, 0, 0, 0]).is_err());
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
