//! In-memory design ranking value: the record a later wave serializes as a
//! `DesignRankingArtifact`.
//!
//! A [`RankingRecord`] binds the decision contract identity, the semantic candidate
//! ids, each candidate's signal provider identity and update mode, the source
//! distribution digests, RNG and error, and the cost mapping to a deterministic
//! canonical order. Its [`RankingRecord::identity`] is invariant to the order the
//! candidates were supplied in and changes with any altered signal, update, source,
//! cost mapping, value or order. When no probabilistic model is licensed the structural
//! ordering is preserved: verified structural sufficiency first, then lower cost
//! units, then lower sample budget, then semantic id.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::ExternalRefusal;
use antecedent_core::CanonicalWriter;

use crate::evsi::{
    CostToUtilityMap, EvsiCandidateRow, EvsiReport, IntegrationMethod, SearchReceipt,
};
use crate::signal::{RefusalBuilder, SignalTrustLabel, SignalUpdateMode};
use crate::study_planner::StudyCost;

/// What the ranking is ordered by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DesignRankingBasis {
    /// Expected value of sample information (no cost mapping).
    Evsi,
    /// Net value of information under a valid cost-to-utility mapping.
    NetValue,
    /// Verified structural sufficiency and cost (no probabilistic model).
    StructuralSufficiencyCost,
}

impl DesignRankingBasis {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Evsi => "evsi",
            Self::NetValue => "net_value",
            Self::StructuralSufficiencyCost => "structural_sufficiency_cost",
        }
    }
}

/// One probabilistically ranked candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiRankRow {
    /// Candidate semantic id.
    pub semantic_id: String,
    /// Rank (0 best).
    pub rank: usize,
    /// EVSI.
    pub evsi: f64,
    /// Standard error of the EVSI (zero unless Monte Carlo).
    pub mc_stderr: f64,
    /// Monte Carlo replicates (zero unless Monte Carlo).
    pub replicates: u64,
    /// Integration method.
    pub integration: IntegrationMethod,
    /// EVPI bound.
    pub evpi: f64,
    /// Sample size.
    pub sample_size: u64,
    /// Study cost amount.
    pub study_cost_amount: f64,
    /// Study cost unit.
    pub study_cost_unit: String,
    /// Net value, only with a valid cost mapping.
    pub net_value: Option<f64>,
    /// Neighbour gap within error or tie tolerance.
    pub rank_uncertain: bool,
    /// Identity of the signal receipt (provider, update mode, trust, request, law).
    pub signal_identity: String,
    /// Exact signal request fingerprint.
    pub request_fingerprint: String,
    /// Update mode.
    pub update_mode: SignalUpdateMode,
    /// Provider trust label.
    pub provider_trust: SignalTrustLabel,
}

impl From<&EvsiCandidateRow> for EvsiRankRow {
    fn from(row: &EvsiCandidateRow) -> Self {
        Self {
            semantic_id: row.semantic_id.clone(),
            rank: row.rank,
            evsi: row.evsi,
            mc_stderr: row.integration.stderr,
            replicates: row.integration.replicates,
            integration: row.integration.method,
            evpi: row.evpi,
            sample_size: row.sample_size,
            study_cost_amount: row.study_cost.amount,
            study_cost_unit: row.study_cost.unit.clone(),
            net_value: row.net_value,
            rank_uncertain: row.rank_uncertain,
            signal_identity: row.signal_receipt.identity(),
            request_fingerprint: row.signal_receipt.request_fingerprint.clone(),
            update_mode: row.update_mode,
            provider_trust: row.provider_trust,
        }
    }
}

/// A candidate with a verified structural verdict, for the no-model ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralCandidate {
    /// Candidate semantic id.
    pub semantic_id: String,
    /// The planner verified the candidate sufficient for the failed query.
    pub verified_sufficient: bool,
    /// Declared cost.
    pub cost: StudyCost,
}

/// One structurally ranked candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralEntry {
    /// Candidate semantic id.
    pub semantic_id: String,
    /// Rank (0 best).
    pub rank: usize,
    /// Verified sufficiency.
    pub verified_sufficient: bool,
    /// Declared cost.
    pub cost: StudyCost,
}

/// The in-memory ranking value.
#[derive(Clone, Debug, PartialEq)]
pub struct RankingRecord {
    /// Basis of the order.
    pub basis: DesignRankingBasis,
    /// Decision contract identity (`None` for a structural ranking).
    pub decision_contract_identity: Option<String>,
    /// Utility unit of the decision.
    pub utility_unit: Option<String>,
    /// Cost mapping, if any.
    pub cost_mapping: Option<CostToUtilityMap>,
    /// RNG seed of the evaluation.
    pub rng_seed: Option<u64>,
    /// Source distribution digests (sorted).
    pub source_digests: Vec<String>,
    /// Probabilistic entries in canonical order (empty for a structural ranking).
    pub entries: Vec<EvsiRankRow>,
    /// Structural entries in canonical order (empty for a probabilistic ranking).
    pub structural: Vec<StructuralEntry>,
    /// Search receipt of the evaluation.
    pub search: SearchReceipt,
}

/// Why a ranking cannot be built or replayed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RankingError {
    /// No candidates.
    Empty,
    /// Two candidates share a semantic id.
    DuplicateCandidate(String),
    /// A candidate's signal identity differs from the retained one.
    SignalMismatch(String),
    /// The source distribution digests differ.
    SourceDigestMismatch,
    /// The cost mapping (or its units) differs.
    CostMappingMismatch,
    /// The decision contract identity differs.
    ContractMismatch,
}

impl std::fmt::Display for RankingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::evsi::write_refusal(f, &self.to_refusal())
    }
}

impl std::error::Error for RankingError {}

impl RankingError {
    /// Structured refusal under the `design_ranking` namespace.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let at = |code, detail: &str| RefusalBuilder::new(code, "consume", detail);
        match self {
            Self::Empty => {
                at(antecedent_core::reason_code!("invalid_argument"), "design_ranking.empty")
                    .build()
            }
            Self::DuplicateCandidate(id) => {
                at(signal, "design_ranking.duplicate_candidate").offending(id.clone()).build()
            }
            Self::SignalMismatch(id) => {
                at(signal, "design_ranking.signal_mismatch").offending(id.clone()).build()
            }
            Self::SourceDigestMismatch => {
                at(signal, "design_ranking.source_digest_mismatch").build()
            }
            Self::CostMappingMismatch => at(
                antecedent_core::reason_code!("design_cost_units_mismatch"),
                "design_ranking.cost_units_mismatch",
            )
            .remedy("replay with the cost mapping the ranking was computed under")
            .build(),
            Self::ContractMismatch => at(signal, "design_ranking.contract_mismatch").build(),
        }
    }
}

impl RankingRecord {
    /// Build the probabilistic ranking from an EVSI report and the digests of the
    /// source distributions (prior sources and external laws) it was computed from.
    ///
    /// # Errors
    ///
    /// An empty report.
    pub fn from_evsi(report: &EvsiReport, source_digests: &[String]) -> Result<Self, RankingError> {
        if report.candidates.is_empty() {
            return Err(RankingError::Empty);
        }
        let mut entries: Vec<EvsiRankRow> =
            report.candidates.iter().map(EvsiRankRow::from).collect();
        entries.sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
        let mut digests = source_digests.to_vec();
        digests.sort();
        digests.dedup();
        Ok(Self {
            basis: match report.basis {
                crate::evsi::RankingBasis::Evsi => DesignRankingBasis::Evsi,
                crate::evsi::RankingBasis::NetValue => DesignRankingBasis::NetValue,
            },
            decision_contract_identity: Some(report.decision_contract_identity.clone()),
            utility_unit: Some(report.utility_unit.clone()),
            cost_mapping: report.cost_map.clone(),
            rng_seed: Some(report.rng_seed),
            source_digests: digests,
            entries,
            structural: Vec::new(),
            search: report.search.clone(),
        })
    }

    /// Build the structural ranking: verified sufficiency first, then fewer cost
    /// units, then smaller sample budget, then semantic id.
    ///
    /// # Errors
    ///
    /// No candidates, or a repeated semantic id.
    pub fn from_structural(candidates: &[StructuralCandidate]) -> Result<Self, RankingError> {
        if candidates.is_empty() {
            return Err(RankingError::Empty);
        }
        let mut ordered: Vec<&StructuralCandidate> = candidates.iter().collect();
        ordered.sort_by(|a, b| {
            b.verified_sufficient
                .cmp(&a.verified_sufficient)
                .then(a.cost.units.cmp(&b.cost.units))
                .then(a.cost.sample_budget.cmp(&b.cost.sample_budget))
                .then_with(|| a.semantic_id.cmp(&b.semantic_id))
        });
        if let Some(pair) = ordered.windows(2).find(|w| w[0].semantic_id == w[1].semantic_id) {
            return Err(RankingError::DuplicateCandidate(pair[0].semantic_id.clone()));
        }
        let structural: Vec<StructuralEntry> = ordered
            .iter()
            .enumerate()
            .map(|(rank, c)| StructuralEntry {
                semantic_id: c.semantic_id.clone(),
                rank,
                verified_sufficient: c.verified_sufficient,
                cost: c.cost,
            })
            .collect();
        let evaluated = structural.len();
        Ok(Self {
            basis: DesignRankingBasis::StructuralSufficiencyCost,
            decision_contract_identity: None,
            utility_unit: None,
            cost_mapping: None,
            rng_seed: None,
            source_digests: Vec::new(),
            entries: Vec::new(),
            structural,
            search: SearchReceipt {
                supplied: evaluated,
                evaluated,
                truncated: false,
                unevaluated_ids: Vec::new(),
            },
        })
    }

    /// The probabilistic ranking when a licensed model produced `probabilistic`,
    /// otherwise the preserved structural ranking.
    ///
    /// # Errors
    ///
    /// As [`Self::from_evsi`] or [`Self::from_structural`].
    pub fn with_fallback(
        probabilistic: Option<(&EvsiReport, &[String])>,
        structural: &[StructuralCandidate],
    ) -> Result<Self, RankingError> {
        match probabilistic {
            Some((report, digests)) => Self::from_evsi(report, digests),
            None => Self::from_structural(structural),
        }
    }

    /// Canonical identity (BLAKE3 hex), invariant to the order candidates were
    /// supplied in.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut hasher = CanonicalWriter::new("antecedent.design_ranking.v1");
        hasher
            .text(self.basis.as_str())
            .opt_text(self.decision_contract_identity.as_deref())
            .opt_text(self.utility_unit.as_deref());
        match &self.cost_mapping {
            Some(map) => {
                hasher
                    .word(1)
                    .text(&map.cost_unit)
                    .text(&map.utility_unit)
                    .real(map.utility_per_cost);
            }
            None => {
                hasher.word(0);
            }
        }
        hasher.opt_word(self.rng_seed).texts(&self.source_digests);
        let mut entries: Vec<&EvsiRankRow> = self.entries.iter().collect();
        entries.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        hasher.word(entries.len() as u64);
        for e in entries {
            hasher
                .text(&e.semantic_id)
                .word(e.rank as u64)
                .real(e.evsi)
                .real(e.mc_stderr)
                .word(e.replicates)
                .text(e.integration.as_str())
                .real(e.evpi)
                .word(e.sample_size)
                .real(e.study_cost_amount)
                .text(&e.study_cost_unit)
                .opt_real(e.net_value)
                .word(u64::from(e.rank_uncertain))
                .text(&e.signal_identity)
                .text(&e.request_fingerprint)
                .text(e.update_mode.as_str())
                .text(e.provider_trust.as_str());
        }
        let mut structural: Vec<&StructuralEntry> = self.structural.iter().collect();
        structural.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        hasher.word(structural.len() as u64);
        for s in structural {
            hasher
                .text(&s.semantic_id)
                .word(s.rank as u64)
                .word(u64::from(s.verified_sufficient))
                .word(s.cost.units)
                .word(s.cost.sample_budget);
        }
        hasher.word(u64::from(self.search.truncated));
        let mut unevaluated = self.search.unevaluated_ids.clone();
        unevaluated.sort();
        hasher.texts(&unevaluated);
        hasher.finish_hex()
    }

    /// Replay check against independently retained inputs: the decision contract
    /// identity, each candidate's signal-receipt identity, the source distribution
    /// digests and the cost mapping must all equal what the ranking recorded.
    ///
    /// # Errors
    ///
    /// The first difference found.
    pub fn verify_replay(
        &self,
        decision_contract_identity: &str,
        signal_identities: &BTreeMap<String, String>,
        source_digests: &[String],
        cost_mapping: Option<&CostToUtilityMap>,
    ) -> Result<(), RankingError> {
        if self.decision_contract_identity.as_deref() != Some(decision_contract_identity) {
            return Err(RankingError::ContractMismatch);
        }
        for entry in &self.entries {
            if signal_identities.get(&entry.semantic_id) != Some(&entry.signal_identity) {
                return Err(RankingError::SignalMismatch(entry.semantic_id.clone()));
            }
        }
        let mut digests = source_digests.to_vec();
        digests.sort();
        digests.dedup();
        if digests != self.source_digests {
            return Err(RankingError::SourceDigestMismatch);
        }
        if self.cost_mapping.as_ref() != cost_mapping {
            return Err(RankingError::CostMappingMismatch);
        }
        Ok(())
    }
}
