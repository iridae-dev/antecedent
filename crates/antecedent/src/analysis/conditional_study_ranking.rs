//! Conditional point study ranking from actual issued combined scientific state.
//! Original structural sufficiency/cost ordering is retained; no point means become draws.
use super::design_ranking::{StructuralCandidateDeclWire, rank_structural};
use super::recalc_composite::CompositeSession;
use antecedent_design::composition_boundary::SupportedVerdict;
use antecedent_design::decision_eval::Verdict;
use antecedent_io::recalc_receipt_artifact::{
    CountsWire, DeclaredStageWire, RecalcReceiptArtifact, decode_parts, identities_to_wire,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
const MAX_BYTES: usize = 4 * 1024 * 1024;
/// Explicit original candidate declaration; the caller owns its sufficiency verdict.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalCandidate {
    /// Original semantic study identity.
    pub semantic_id: String,
    /// Passed unchanged to original ranking; this projection never proves sufficiency.
    pub verified_sufficient: bool,
    /// Original declared cost units.
    pub cost_units: u64,
    /// Original declared sample budget.
    pub sample_budget: u64,
}
/// One declared action-dependent table.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalBranch {
    /// Actual terminal action identity.
    pub action: String,
    /// Complete bounded original structural candidate table for this action.
    pub candidates: Vec<ConditionalCandidate>,
}
/// Complete conditional rule; every actual terminal action must occur exactly once.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalStudyPolicy {
    /// Caller-owned rule/version identity.
    pub policy_id: String,
    /// At most64 action branches, at most512 total candidates.
    pub branches: Vec<ConditionalBranch>,
}
/// Registered checked request/source/artifact refusal.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConditionalRankingError {
    /// Stable reason code.
    pub code: &'static str,
    /// Exact route-owned detail.
    pub detail: &'static str,
}
fn refuse(detail: &'static str) -> ConditionalRankingError {
    let code = if detail == "conditional_study_ranking.source_unavailable"
        || detail == "conditional_study_ranking.action_not_unique"
    {
        antecedent_core::reason_code!("route_not_supported")
    } else {
        antecedent_core::reason_code!("invalid_argument")
    };
    ConditionalRankingError { code, detail }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Action {
    id: String,
    status: String,
    input_id: Option<String>,
    expected_utility: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    contract_identity: String,
    requested: Vec<DeclaredStageWire>,
    input_order: Vec<String>,
    actions: Vec<Action>,
    selected_action: String,
    native_response: Vec<u8>,
    callback_claims: Vec<Vec<u8>>,
}
/// One original structural entry; its sufficiency declaration has not been upgraded.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalRankingEntry {
    /// Original study identity.
    pub semantic_id: String,
    /// Original structural rank, zero best.
    pub rank: usize,
    /// Original declared sufficiency.
    pub verified_sufficient: bool,
    /// Original declared cost units.
    pub cost_units: u64,
    /// Original declared sample budget.
    pub sample_budget: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: u16,
    source: Source,
    historical_receipt: Vec<u8>,
    policy: ConditionalStudyPolicy,
    entries: Vec<ConditionalRankingEntry>,
    identity: String,
}
/// Bounded source-backed point ranking. Bytes alone cannot supply current scientific state.
pub struct ConditionalStudyRanking {
    payload: Payload,
}
fn counts(c: super::recalc_receipt::StageCounts) -> CountsWire {
    CountsWire {
        identifications: c.identifications,
        fold_fits: c.fold_fits,
        model_fits: c.model_fits,
        score_computations: c.score_computations,
        reweights: c.reweights,
        decisions: c.decisions,
        factor_builds: c.factor_builds,
        program_compilations: c.program_compilations,
        provider_bindings: c.provider_bindings,
        factor_evaluations: c.factor_evaluations,
        integrations: c.integrations,
        provider_calls: c.provider_calls,
        law_summaries: c.law_summaries,
        posterior_draws: c.posterior_draws,
        external_invocations: c.external_invocations,
    }
}
fn source(session: &CompositeSession) -> Result<Source, ConditionalRankingError> {
    let publication = session
        .publication()
        .ok_or_else(|| refuse("conditional_study_ranking.source_unavailable"))?;
    let decision =
        session.decision().ok_or_else(|| refuse("conditional_study_ranking.source_unavailable"))?;
    let SupportedVerdict::Compared(Verdict::UniquelyOptimal(selected)) = &decision.verdict else {
        return Err(refuse("conditional_study_ranking.action_not_unique"));
    };
    Ok(Source {
        contract_identity: decision.contract_identity.clone(),
        requested: identities_to_wire(&publication.requested),
        input_order: publication.input_order.clone(),
        selected_action: selected.clone(),
        native_response: session
            .native()
            .export_result(
                session
                    .native()
                    .producing_context()
                    .ok_or_else(|| refuse("conditional_study_ranking.source_unavailable"))?,
            )
            .map_err(|_| refuse("conditional_study_ranking.source_unavailable"))?,
        callback_claims: publication
            .branches
            .iter()
            .map(|b| {
                session
                    .callback(*b)
                    .ok_or_else(|| refuse("conditional_study_ranking.source_unavailable"))?
                    .export_output()
                    .map_err(|_| refuse("conditional_study_ranking.source_unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        actions: decision
            .dispositions
            .iter()
            .map(|a| Action {
                id: a.id.clone(),
                status: format!("{:?}", a.status),
                input_id: a.input_id.clone(),
                expected_utility: decision
                    .outcomes
                    .iter()
                    .find(|o| o.id == a.id)
                    .map(|o| o.expected_utility),
            })
            .collect(),
    })
}
fn validate(
    policy: &ConditionalStudyPolicy,
    source: &Source,
) -> Result<(), ConditionalRankingError> {
    let valid_name = |s: &str| !s.trim().is_empty() && s.len() <= 256;
    if !valid_name(&policy.policy_id)
        || policy.branches.is_empty()
        || policy.branches.len() > 64
        || policy.branches.iter().map(|b| b.candidates.len()).sum::<usize>() > 512
    {
        return Err(refuse("conditional_study_ranking.invalid_policy"));
    }
    let expected: BTreeSet<_> = source.actions.iter().map(|a| a.id.as_str()).collect();
    let mut actual = BTreeSet::new();
    for branch in &policy.branches {
        if !valid_name(&branch.action)
            || !actual.insert(branch.action.as_str())
            || branch.candidates.is_empty()
            || branch.candidates.len() > 128
        {
            return Err(refuse("conditional_study_ranking.invalid_policy"));
        }
        let mut candidates = BTreeSet::new();
        if branch
            .candidates
            .iter()
            .any(|c| !valid_name(&c.semantic_id) || !candidates.insert(c.semantic_id.as_str()))
        {
            return Err(refuse("conditional_study_ranking.invalid_policy"));
        }
    }
    if actual != expected {
        return Err(refuse("conditional_study_ranking.invalid_policy"));
    }
    Ok(())
}
fn rank(
    policy: &ConditionalStudyPolicy,
    source: &Source,
) -> Result<Vec<ConditionalRankingEntry>, ConditionalRankingError> {
    validate(policy, source)?;
    let branch = policy
        .branches
        .iter()
        .find(|b| b.action == source.selected_action)
        .ok_or_else(|| refuse("conditional_study_ranking.invalid_policy"))?;
    let candidates: Vec<_> = branch
        .candidates
        .iter()
        .map(|c| StructuralCandidateDeclWire {
            semantic_id: c.semantic_id.clone(),
            verified_sufficient: c.verified_sufficient,
            cost_units: c.cost_units,
            sample_budget: c.sample_budget,
        })
        .collect();
    let result = rank_structural(&candidates)
        .map_err(|_| refuse("conditional_study_ranking.original_ranking_refused"))?;
    Ok(result
        .structural
        .into_iter()
        .map(|e| ConditionalRankingEntry {
            semantic_id: e.semantic_id,
            rank: e.rank,
            verified_sufficient: e.verified_sufficient,
            cost_units: e.cost.units,
            sample_budget: e.cost.sample_budget,
        })
        .collect())
}
fn identity(payload: &Payload) -> Result<String, ConditionalRankingError> {
    let mut canonical = payload.clone();
    canonical.identity.clear();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
    if bytes.len() > MAX_BYTES {
        return Err(refuse("conditional_study_ranking.resource_limit"));
    }
    Ok(blake3::hash(&bytes).to_hex().to_string())
}
impl ConditionalStudyRanking {
    /// Execute original structural ranking from the current actually issued terminal verdict.
    /// # Errors
    /// Missing/ambiguous actual state, incomplete policy, budget or original ranking refusal.
    pub fn execute(
        session: &CompositeSession,
        policy: ConditionalStudyPolicy,
    ) -> Result<Self, ConditionalRankingError> {
        let source = source(session)?;
        let entries = rank(&policy, &source)?;
        let publication = session
            .publication()
            .ok_or_else(|| refuse("conditional_study_ranking.source_unavailable"))?;
        let count_map: BTreeMap<_, _> =
            publication.receipt.entries().iter().map(|e| (e.stage, counts(e.counts))).collect();
        let receipt = RecalcReceiptArtifact::seal(
            &publication.previous,
            &publication.requested,
            &publication.capabilities,
            &count_map,
        )
        .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        let historical_receipt = receipt
            .to_bytes("conditional-study-source")
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        let mut payload = Payload {
            version: 1,
            source,
            historical_receipt,
            policy,
            entries,
            identity: String::new(),
        };
        payload.identity = identity(&payload)?;
        Ok(Self { payload })
    }
    /// Source-bound artifact identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.payload.identity
    }
    /// Actual terminal action that selected the declared study table.
    #[must_use]
    pub fn selected_action(&self) -> &str {
        &self.payload.source.selected_action
    }
    /// Original point study ranking entries.
    #[must_use]
    pub fn entries(&self) -> &[ConditionalRankingEntry] {
        &self.payload.entries
    }
    /// Inspect complete rule, source request/order and full historical work without granting authority.
    /// # Errors
    /// Serialization exceeds the byte budget or fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ConditionalRankingError> {
        let bytes = serde_json::to_vec(&self.payload)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        if bytes.len() > MAX_BYTES {
            return Err(refuse("conditional_study_ranking.resource_limit"));
        }
        Ok(bytes)
    }
    /// Independently consume against an explicitly replayed actual opaque combined session.
    /// Historical receipt inspection cannot replace the executing native/provider state.
    /// # Errors
    /// Bounds/hash/version/receipt/rule/source mismatch, or original ranking does not reproduce.
    pub fn consume(
        bytes: &[u8],
        session: &CompositeSession,
        expected_identity: Option<&str>,
    ) -> Result<Self, ConditionalRankingError> {
        if bytes.len() > MAX_BYTES {
            return Err(refuse("conditional_study_ranking.resource_limit"));
        }
        let payload: Payload = serde_json::from_slice(bytes)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        if payload.version != 1
            || payload.identity != identity(&payload)?
            || expected_identity.is_some_and(|e| e != payload.identity)
        {
            return Err(refuse("conditional_study_ranking.artifact_invalid"));
        }
        let _receipt = RecalcReceiptArtifact::from_bytes(&payload.historical_receipt, None)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        let receipt_meta = decode_parts(&payload.historical_receipt)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        let current = source(session)?;
        if payload.source != current || receipt_meta.requested != current.requested {
            return Err(refuse("conditional_study_ranking.source_mismatch"));
        }
        if payload.entries != rank(&payload.policy, &current)? {
            return Err(refuse("conditional_study_ranking.artifact_invalid"));
        }
        Ok(Self { payload })
    }
}
