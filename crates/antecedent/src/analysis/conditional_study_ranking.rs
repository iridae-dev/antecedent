//! Conditional point study ranking from actual issued combined scientific state.
//! Original structural sufficiency/cost ordering is retained; no point means become draws.
use super::design_ranking::{StructuralCandidateDeclWire, rank_structural};
use super::recalc_composite::CompositeSession;
use antecedent_design::composition_boundary::SupportedVerdict;
use antecedent_design::decision_eval::Verdict;
use antecedent_io::recalc_receipt_artifact::{
    CountsWire, DeclaredStageWire, RecalcReceiptArtifact, identities_to_wire,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
/// Default most bytes of a sealed ranking payload built, decoded or hashed.
const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024;
/// Byte budget of a conditional ranking: the sealed payload (policy, source and receipt) built,
/// serialized, decoded or hashed may not exceed it, and a policy whose identifiers alone cannot
/// fit is refused before any per-candidate work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditionalRankingLimits {
    /// Largest payload, in bytes.
    pub max_bytes: usize,
}
impl Default for ConditionalRankingLimits {
    fn default() -> Self {
        Self { max_bytes: DEFAULT_MAX_BYTES }
    }
}
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
    /// One branch per actual terminal action. The table is bounded by the artifact byte
    /// budget (it is part of the sealed payload), not by a count of branches or candidates.
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
    limits: ConditionalRankingLimits,
) -> Result<(), ConditionalRankingError> {
    let valid_name = |s: &str| !s.trim().is_empty() && s.len() <= 256;
    if !valid_name(&policy.policy_id) || policy.branches.is_empty() {
        return Err(refuse("conditional_study_ranking.invalid_policy"));
    }
    // The policy is sealed into the payload, so its identifiers alone already cost this many
    // bytes of the artifact budget; refuse before any per-candidate work when they cannot fit.
    let identifier_bytes: usize = policy.policy_id.len()
        + policy
            .branches
            .iter()
            .map(|b| b.action.len() + b.candidates.iter().map(|c| c.semantic_id.len()).sum::<usize>())
            .sum::<usize>();
    if identifier_bytes > limits.max_bytes {
        return Err(refuse("conditional_study_ranking.resource_limit"));
    }
    let expected: BTreeSet<_> = source.actions.iter().map(|a| a.id.as_str()).collect();
    let mut actual = BTreeSet::new();
    for branch in &policy.branches {
        if !valid_name(&branch.action)
            || !actual.insert(branch.action.as_str())
            || branch.candidates.is_empty()
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
    limits: ConditionalRankingLimits,
) -> Result<Vec<ConditionalRankingEntry>, ConditionalRankingError> {
    validate(policy, source, limits)?;
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
/// The payload as the identity digest sees it: every field of [`Payload`] in declaration order,
/// borrowed, with `identity` fixed to the empty string. Serialising it yields exactly the bytes
/// of the payload with its identity cleared, without cloning the source and receipt bytes.
#[derive(Serialize)]
struct IdentityFreeView<'a> {
    version: u16,
    source: &'a Source,
    historical_receipt: &'a [u8],
    policy: &'a ConditionalStudyPolicy,
    entries: &'a [ConditionalRankingEntry],
    identity: &'static str,
}

/// Identity of a payload (encoding version 1): the lowercase hex BLAKE3 digest of the compact
/// `serde_json` encoding of the payload with its `identity` field set to `""`. Fields are
/// written in declaration order (`version`, `source`, `historical_receipt`, `policy`,
/// `entries`, `identity`); byte strings are JSON arrays of numbers. The digest binds the
/// encoding version through the leading `version` field, so a change of encoding is a new
/// version, never a silent change of identity.
fn identity(
    payload: &Payload,
    limits: ConditionalRankingLimits,
) -> Result<String, ConditionalRankingError> {
    let view = IdentityFreeView {
        version: payload.version,
        source: &payload.source,
        historical_receipt: &payload.historical_receipt,
        policy: &payload.policy,
        entries: &payload.entries,
        identity: "",
    };
    let bytes = serde_json::to_vec(&view)
        .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
    if bytes.len() > limits.max_bytes {
        return Err(refuse("conditional_study_ranking.resource_limit"));
    }
    Ok(antecedent_core::blake3_hex(&bytes))
}
fn retained_payload(
    bytes: &[u8],
    expected_identity: Option<&str>,
    limits: ConditionalRankingLimits,
) -> Result<Payload, ConditionalRankingError> {
    if bytes.len() > limits.max_bytes {
        return Err(refuse("conditional_study_ranking.resource_limit"));
    }
    let payload: Payload = serde_json::from_slice(bytes)
        .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
    if payload.version != 1
        || payload.identity != identity(&payload, limits)?
        || expected_identity.is_some_and(|e| e != payload.identity)
    {
        return Err(refuse("conditional_study_ranking.artifact_invalid"));
    }
    Ok(payload)
}
impl ConditionalStudyRanking {
    /// Execute original structural ranking from the current actually issued terminal verdict.
    /// # Errors
    /// Missing/ambiguous actual state, incomplete policy, budget or original ranking refusal.
    pub fn execute(
        session: &CompositeSession,
        policy: ConditionalStudyPolicy,
    ) -> Result<Self, ConditionalRankingError> {
        Self::execute_with_limits(session, policy, ConditionalRankingLimits::default())
    }

    /// [`Self::execute`] under the caller's byte budget instead of the default.
    /// # Errors
    /// Those of [`Self::execute`], with `limits` in place of the default budget.
    pub fn execute_with_limits(
        session: &CompositeSession,
        policy: ConditionalStudyPolicy,
        limits: ConditionalRankingLimits,
    ) -> Result<Self, ConditionalRankingError> {
        let source = source(session)?;
        let entries = rank(&policy, &source, limits)?;
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
        payload.identity = identity(&payload, limits)?;
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
        self.to_bytes_with_limits(ConditionalRankingLimits::default())
    }

    /// [`Self::to_bytes`] under the caller's byte budget instead of the default.
    /// # Errors
    /// Those of [`Self::to_bytes`], with `limits` in place of the default budget.
    pub fn to_bytes_with_limits(
        &self,
        limits: ConditionalRankingLimits,
    ) -> Result<Vec<u8>, ConditionalRankingError> {
        let bytes = serde_json::to_vec(&self.payload)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        if bytes.len() > limits.max_bytes {
            return Err(refuse("conditional_study_ranking.resource_limit"));
        }
        Ok(bytes)
    }
    /// Inspect a sealed historical ranking without supplying executable source state.
    ///
    /// Checks its identity, historical receipt and the structural ranking of its
    /// retained source. This does not attest that the source is the current
    /// combined native/provider state; [`Self::consume`] performs that check.
    /// # Errors
    /// Bounds, encoding, identity, receipt or retained ranking mismatch.
    pub fn inspect_retained(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ConditionalRankingError> {
        Self::inspect_retained_with_limits(
            bytes,
            expected_identity,
            ConditionalRankingLimits::default(),
        )
    }

    /// [`Self::inspect_retained`] under the caller's byte budget instead of the default.
    /// # Errors
    /// Those of [`Self::inspect_retained`], with `limits` in place of the default budget.
    pub fn inspect_retained_with_limits(
        bytes: &[u8],
        expected_identity: Option<&str>,
        limits: ConditionalRankingLimits,
    ) -> Result<Self, ConditionalRankingError> {
        let payload = retained_payload(bytes, expected_identity, limits)?;
        let receipt = RecalcReceiptArtifact::from_bytes(&payload.historical_receipt, None)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        if receipt.meta().requested != payload.source.requested
            || payload.entries != rank(&payload.policy, &payload.source, limits)?
        {
            return Err(refuse("conditional_study_ranking.artifact_invalid"));
        }
        Ok(Self { payload })
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
        Self::consume_with_limits(
            bytes,
            session,
            expected_identity,
            ConditionalRankingLimits::default(),
        )
    }

    /// [`Self::consume`] under the caller's byte budget instead of the default.
    /// # Errors
    /// Those of [`Self::consume`], with `limits` in place of the default budget.
    pub fn consume_with_limits(
        bytes: &[u8],
        session: &CompositeSession,
        expected_identity: Option<&str>,
        limits: ConditionalRankingLimits,
    ) -> Result<Self, ConditionalRankingError> {
        let payload = retained_payload(bytes, expected_identity, limits)?;
        let receipt = RecalcReceiptArtifact::from_bytes(&payload.historical_receipt, None)
            .map_err(|_| refuse("conditional_study_ranking.artifact_invalid"))?;
        let receipt_meta = receipt.meta();
        let current = source(session)?;
        if payload.source != current || receipt_meta.requested != current.requested {
            return Err(refuse("conditional_study_ranking.source_mismatch"));
        }
        if payload.entries != rank(&payload.policy, &current, limits)? {
            return Err(refuse("conditional_study_ranking.artifact_invalid"));
        }
        Ok(Self { payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(actions: &[String]) -> Source {
        Source {
            contract_identity: "contract".into(),
            requested: Vec::new(),
            input_order: vec!["native".into()],
            actions: actions
                .iter()
                .map(|id| Action {
                    id: id.clone(),
                    status: "Optimal".into(),
                    input_id: None,
                    expected_utility: Some(1.0),
                })
                .collect(),
            selected_action: actions[0].clone(),
            native_response: vec![1, 2, 3],
            callback_claims: vec![vec![4, 5]],
        }
    }

    fn candidates(prefix: &str, count: usize, width: usize) -> Vec<ConditionalCandidate> {
        (0..count)
            .map(|i| ConditionalCandidate {
                semantic_id: format!("{prefix}-{i:0>width$}"),
                verified_sufficient: i % 2 == 0,
                cost_units: i as u64,
                sample_budget: 10,
            })
            .collect()
    }

    #[test]
    fn the_identity_view_encodes_exactly_the_payload_with_its_identity_cleared() {
        let actions = vec!["wait".to_string(), "treat".to_string()];
        let policy = ConditionalStudyPolicy {
            policy_id: "rule".into(),
            branches: actions
                .iter()
                .map(|a| ConditionalBranch { action: a.clone(), candidates: candidates(a, 3, 2) })
                .collect(),
        };
        let source = source(&actions);
        let entries = rank(&policy, &source, ConditionalRankingLimits::default()).unwrap();
        let payload = Payload {
            version: 1,
            source,
            historical_receipt: vec![9, 8, 7],
            policy,
            entries,
            identity: "stale".into(),
        };
        let mut cleared = payload.clone();
        cleared.identity.clear();
        let expected = antecedent_core::blake3_hex(&serde_json::to_vec(&cleared).unwrap());
        assert_eq!(identity(&payload, ConditionalRankingLimits::default()).unwrap(), expected);
    }

    #[test]
    fn a_policy_is_bounded_by_the_byte_budget_and_not_by_counts() {
        let actions: Vec<String> = (0..100).map(|i| format!("action-{i}")).collect();
        let source = source(&actions);
        let branches = |count: usize, width: usize| {
            actions
                .iter()
                .map(|a| ConditionalBranch { action: a.clone(), candidates: candidates(a, count, width) })
                .collect::<Vec<_>>()
        };
        // 100 actions and 20_000 candidates: far past the former count caps, well inside the budget.
        let wide = ConditionalStudyPolicy { policy_id: "rule".into(), branches: branches(200, 4) };
        let limits = ConditionalRankingLimits::default();
        assert!(validate(&wide, &source, limits).is_ok());
        // The same shape with 250-byte identifiers (a 10-byte prefix and 240 digits, inside the
        // 256-byte name limit) cannot fit the artifact budget.
        let heavy = ConditionalStudyPolicy { policy_id: "rule".into(), branches: branches(200, 240) };
        assert_eq!(
            validate(&heavy, &source, limits).unwrap_err().detail,
            "conditional_study_ranking.resource_limit"
        );
        // A caller-declared budget replaces the default: the heavy policy fits a larger one.
        let larger = ConditionalRankingLimits { max_bytes: 64 * 1024 * 1024 };
        assert!(validate(&heavy, &source, larger).is_ok());
    }
}
