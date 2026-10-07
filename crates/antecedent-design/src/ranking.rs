//! In-memory design ranking value: the record a later wave serializes as a
//! `DesignRankingArtifact`.
//!
//! A [`DesignRanking`] binds the decision contract identity, the semantic candidate
//! ids, each candidate's signal provider identity and update mode, the source
//! distribution digests, RNG and error, and the cost mapping to a deterministic
//! canonical order. Its [`DesignRanking::identity`] is invariant to the order the
//! candidates were supplied in and changes with any altered signal, update, source,
//! cost mapping, value or order. When no probabilistic model is licensed the 2.2
//! ordering is preserved: verified structural sufficiency first, then lower cost
//! units, then lower sample budget, then semantic id.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExternalRefusal;

use crate::evsi::{CostToUtilityMap, EvsiReport, IntegrationMethod, SearchReceipt};
use crate::signal::{SignalTrustLabel, SignalUpdateMode, make_refusal};
use crate::study_planner::StudyCost;

/// What the ranking is ordered by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DesignRankingBasis {
    /// Expected value of sample information (no cost mapping).
    Evsi,
    /// Net value of information under a valid cost-to-utility mapping.
    NetValue,
    /// 2.2 verified structural sufficiency and cost (no probabilistic model).
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
pub struct RankingEntry {
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
pub struct DesignRanking {
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
    pub entries: Vec<RankingEntry>,
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

impl RankingError {
    /// Structured refusal under the `design_ranking` namespace.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let at = |code, detail: &str| make_refusal(code, "consume", detail);
        match self {
            Self::Empty => {
                at(antecedent_core::reason_code!("invalid_argument"), "design_ranking.empty")
            }
            Self::DuplicateCandidate(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "design_ranking.duplicate_candidate")
            },
            Self::SignalMismatch(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "design_ranking.signal_mismatch")
            },
            Self::SourceDigestMismatch => at(signal, "design_ranking.source_digest_mismatch"),
            Self::CostMappingMismatch => at(
                antecedent_core::reason_code!("design_cost_units_mismatch"),
                "design_ranking.cost_units_mismatch",
            ),
            Self::ContractMismatch => at(signal, "design_ranking.contract_mismatch"),
        }
    }
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn put_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_le_bytes());
}

fn put_opt_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(v) => {
            hasher.update(&[1]);
            put_f64(hasher, v);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

impl DesignRanking {
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
        let mut entries: Vec<RankingEntry> = report
            .candidates
            .iter()
            .map(|c| RankingEntry {
                semantic_id: c.semantic_id.clone(),
                rank: c.rank,
                evsi: c.evsi,
                mc_stderr: c.integration.stderr,
                replicates: c.integration.replicates,
                integration: c.integration.method,
                evpi: c.evpi,
                sample_size: c.sample_size,
                study_cost_amount: c.study_cost.amount,
                study_cost_unit: c.study_cost.unit.clone(),
                net_value: c.net_value,
                rank_uncertain: c.rank_uncertain,
                signal_identity: c.signal_receipt.identity(),
                request_fingerprint: c.signal_receipt.request_fingerprint.clone(),
                update_mode: c.update_mode,
                provider_trust: c.provider_trust,
            })
            .collect();
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

    /// Build the 2.2 structural ranking: verified sufficiency first, then fewer cost
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
    /// otherwise the preserved 2.2 structural ranking.
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
        let mut hasher = blake3::Hasher::new();
        put_str(&mut hasher, "antecedent.design_ranking.v1");
        put_str(&mut hasher, self.basis.as_str());
        put_str(&mut hasher, self.decision_contract_identity.as_deref().unwrap_or(""));
        put_str(&mut hasher, self.utility_unit.as_deref().unwrap_or(""));
        if let Some(map) = &self.cost_mapping {
            hasher.update(&[1]);
            put_str(&mut hasher, &map.cost_unit);
            put_str(&mut hasher, &map.utility_unit);
            put_f64(&mut hasher, map.utility_per_cost);
        } else {
            hasher.update(&[0]);
        }
        hasher.update(&self.rng_seed.unwrap_or(0).to_le_bytes());
        for digest in &self.source_digests {
            put_str(&mut hasher, digest);
        }
        let mut entries: Vec<&RankingEntry> = self.entries.iter().collect();
        entries.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        for e in entries {
            put_str(&mut hasher, &e.semantic_id);
            hasher.update(&(e.rank as u64).to_le_bytes());
            put_f64(&mut hasher, e.evsi);
            put_f64(&mut hasher, e.mc_stderr);
            hasher.update(&e.replicates.to_le_bytes());
            put_str(&mut hasher, e.integration.as_str());
            put_f64(&mut hasher, e.evpi);
            hasher.update(&e.sample_size.to_le_bytes());
            put_f64(&mut hasher, e.study_cost_amount);
            put_str(&mut hasher, &e.study_cost_unit);
            put_opt_f64(&mut hasher, e.net_value);
            hasher.update(&[u8::from(e.rank_uncertain)]);
            put_str(&mut hasher, &e.signal_identity);
            put_str(&mut hasher, &e.request_fingerprint);
            put_str(&mut hasher, e.update_mode.as_str());
            put_str(&mut hasher, e.provider_trust.as_str());
        }
        let mut structural: Vec<&StructuralEntry> = self.structural.iter().collect();
        structural.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        for s in structural {
            put_str(&mut hasher, &s.semantic_id);
            hasher.update(&(s.rank as u64).to_le_bytes());
            hasher.update(&[u8::from(s.verified_sufficient)]);
            hasher.update(&s.cost.units.to_le_bytes());
            hasher.update(&s.cost.sample_budget.to_le_bytes());
        }
        hasher.update(&[u8::from(self.search.truncated)]);
        let mut unevaluated = self.search.unevaluated_ids.clone();
        unevaluated.sort();
        for id in &unevaluated {
            put_str(&mut hasher, id);
        }
        hasher.finalize().to_hex().to_string()
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
