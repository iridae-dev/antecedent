//! Validate and consume sealed design-ranking artifacts.

use super::{
    AdjacentRank, BTreeMap, BTreeSet, Baseline, ConsumeExpectation, ConsumedCandidate,
    ConsumedDesignRanking, CostToUtilityMap, DESIGN_RANKING_ARTIFACT_KIND,
    DESIGN_RANKING_ARTIFACT_VERSION, DESIGN_RANKING_CALIBRATION, DecisionWire,
    DeclaredObservations, DesignRankingArtifactError, DesignRankingArtifactWire,
    DesignRankingBasis, DesignRankingReadLimits, EvsiCandidateWire, EvsiRankRow, LawWire,
    ObservationOverlap, PriorWire, ProviderObjectIdentity, RankingError, RankingRecord, Result,
    ScientificQuantity, SearchReceipt, SignalLimits, SignalReceipt, SignalRequest,
    SignalTrustLabel, SignalUpdateMode, UtilityWire, baseline, close, declared_overlap,
    evsi_bound_ok, is_blank, is_native, parse_method, parse_mode, parse_trust, rank_basis,
    rank_rows, recompute, rectangular,
};

fn check_bounds(wire: &DesignRankingArtifactWire, limits: &DesignRankingReadLimits) -> Result<()> {
    if wire.candidates.is_empty() {
        return Err(DesignRankingArtifactError::InvalidWire("no candidates"));
    }
    if wire.candidates.len() > limits.max_candidates
        || !usize::try_from(wire.max_candidates).is_ok_and(|m| m <= limits.max_candidates)
    {
        return Err(DesignRankingArtifactError::Bounds("candidates"));
    }
    let actions = wire.decision.action_ids.len();
    if actions == 0 || actions > limits.max_states {
        return Err(DesignRankingArtifactError::Bounds("actions"));
    }
    let states = match &wire.decision.prior {
        PriorWire::Draws { states } => states.len(),
        PriorWire::Normal { .. } => 1,
    };
    if states == 0
        || states > limits.max_states
        || states.saturating_mul(actions) > limits.max_cells
    {
        return Err(DesignRankingArtifactError::Bounds("decision_table"));
    }
    for candidate in &wire.candidates {
        let support = match &candidate.law {
            LawWire::GaussianMean { .. } => 0,
            LawWire::Binomial => usize::try_from(candidate.sample_size)
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or(DesignRankingArtifactError::Bounds("support"))?,
            LawWire::FiniteLikelihood { statistics, .. }
            | LawWire::FinitePosterior { statistics, .. } => statistics.len(),
            LawWire::DecisionValues { branch_probabilities, .. } => branch_probabilities.len(),
        };
        if support > limits.max_states || support.saturating_mul(states) > limits.max_cells {
            return Err(DesignRankingArtifactError::Bounds("signal_table"));
        }
    }
    Ok(())
}

fn check_decision(decision: &DecisionWire) -> Result<()> {
    let bad = DesignRankingArtifactError::InvalidWire;
    if is_blank(&decision.contract_identity) || is_blank(&decision.utility_unit) {
        return Err(bad("decision identity and utility unit"));
    }
    let n_actions = decision.action_ids.len();
    let unique: BTreeSet<&String> = decision.action_ids.iter().collect();
    if unique.len() != n_actions || decision.action_ids.iter().any(|id| is_blank(id)) {
        return Err(bad("action ids"));
    }
    if decision.admissible.len() != n_actions || !decision.admissible.iter().any(|ok| *ok) {
        return Err(bad("admissible actions"));
    }
    match (&decision.prior, &decision.utility) {
        (PriorWire::Draws { states }, UtilityWire::Table { rows }) => {
            if !states.iter().all(|s| s.is_finite())
                || !rectangular(rows, n_actions, states.len())
                || !rows.iter().all(|row| row.iter().all(|u| u.is_finite()))
            {
                return Err(bad("decision table"));
            }
        }
        (PriorWire::Normal { mean, variance }, UtilityWire::Affine { coefficients }) => {
            if !(mean.is_finite() && variance.is_finite() && *variance > 0.0)
                || coefficients.len() != n_actions
                || !coefficients.iter().all(|(a, b)| a.is_finite() && b.is_finite())
                || decision.admissible.iter().any(|ok| !ok)
            {
                return Err(bad("normal decision"));
            }
        }
        _ => return Err(bad("prior and utility representation disagree")),
    }
    Ok(())
}

fn check_costs(wire: &DesignRankingArtifactWire) -> Result<()> {
    let utility_unit = &wire.decision.utility_unit;
    if let Some(map) = &wire.cost_map {
        if is_blank(&map.cost_unit)
            || is_blank(&map.utility_unit)
            || !(map.utility_per_cost.is_finite() && map.utility_per_cost > 0.0)
        {
            return Err(DesignRankingArtifactError::InvalidWire("cost mapping"));
        }
        if &map.utility_unit != utility_unit {
            return Err(DesignRankingArtifactError::CostUnitsMismatch(format!(
                "map utility unit {} differs from decision utility unit {utility_unit}",
                map.utility_unit
            )));
        }
    }
    for candidate in &wire.candidates {
        if !(candidate.cost_amount.is_finite() && candidate.cost_amount >= 0.0)
            || is_blank(&candidate.cost_unit)
        {
            return Err(DesignRankingArtifactError::InvalidWire("study cost"));
        }
        if let Some(map) = &wire.cost_map {
            if candidate.cost_unit != map.cost_unit {
                return Err(DesignRankingArtifactError::CostUnitsMismatch(format!(
                    "candidate {} cost unit {} differs from map cost unit {}",
                    candidate.semantic_id, candidate.cost_unit, map.cost_unit
                )));
            }
        }
    }
    Ok(())
}

fn signal_request_of(candidate: &EvsiCandidateWire) -> Option<SignalRequest> {
    let wire = &candidate.request;
    let state = ScientificQuantity::try_from(wire.state_quantity.clone()).ok()?;
    let observation = ScientificQuantity::try_from(wire.observation_quantity.clone()).ok()?;
    Some(SignalRequest {
        candidate_id: wire.candidate_id.clone(),
        prior_id: wire.prior_id.clone(),
        state_quantity: state,
        observation_quantity: observation,
        sample_size: candidate.sample_size,
        rng_seed: wire.rng_seed,
        evidence_lineage: wire.evidence_lineage.clone(),
        conditional_independence: wire.conditional_independence.clone(),
        limits: SignalLimits {
            max_sample_size: wire.max_sample_size,
            max_support: usize::try_from(wire.max_support).ok()?,
        },
    })
}

/// Trust-limit sentence for the consumed candidate.
fn trust_limit(trust: SignalTrustLabel, attestor: Option<&str>) -> String {
    match trust {
        SignalTrustLabel::NativeSupported => {
            "native_supported: implemented and supported natively".to_owned()
        }
        SignalTrustLabel::ExternallyAttested => format!(
            "externally_attested by {}: supplier assertion only, not verified or natively replayed by Antecedent",
            attestor.unwrap_or("unnamed attestor")
        ),
        SignalTrustLabel::ExactRequestVerified => "exact_request_verified: the verification receipt is not retained in this artifact; the signal is still not native and was not natively replayed".to_owned(),
    }
}

fn trust_limit_unreceipted(attestor: Option<&str>) -> String {
    format!(
        "externally_attested by {}: the artifact stores an exact-request verification label but no receipt, and none was supplied by the consumer; supplier assertion only, not verified or natively replayed by Antecedent",
        attestor.unwrap_or("unnamed attestor")
    )
}

fn check_signal(candidate: &EvsiCandidateWire) -> Result<(SignalTrustLabel, SignalUpdateMode)> {
    let id = &candidate.semantic_id;
    let inconsistent = |slot: &'static str| DesignRankingArtifactError::SignalInconsistent {
        candidate: id.clone(),
        slot,
    };
    let request = signal_request_of(candidate).ok_or_else(|| inconsistent("request"))?;
    request.validate().map_err(|_| inconsistent("request"))?;
    if request.fingerprint() != candidate.request_fingerprint {
        return Err(inconsistent("request_fingerprint"));
    }
    let mode = parse_mode(&candidate.update_mode).ok_or_else(|| inconsistent("update_mode"))?;
    let trust = parse_trust(&candidate.trust).ok_or_else(|| inconsistent("trust"))?;
    // Mode, trust, provider and law must tell one story.
    let law_mode = match &candidate.law {
        LawWire::GaussianMean { .. } | LawWire::Binomial | LawWire::FiniteLikelihood { .. } => {
            SignalUpdateMode::NativeUpdate
        }
        LawWire::FinitePosterior { .. } => SignalUpdateMode::ExternalPosterior,
        LawWire::DecisionValues { .. } => SignalUpdateMode::ExternalDecisionValues,
    };
    if mode != law_mode {
        return Err(inconsistent("update_mode"));
    }
    if is_native(&candidate.law) {
        let family = match &candidate.law {
            LawWire::GaussianMean { .. } => "gaussian_mean",
            _ => "binomial",
        };
        let parameters: Vec<f64> = match &candidate.law {
            LawWire::GaussianMean { noise_variance } => vec![*noise_variance],
            _ => Vec::new(),
        };
        if trust != SignalTrustLabel::NativeSupported
            || candidate.provider.is_some()
            || candidate.attestor.is_some()
            || candidate.family != family
        {
            return Err(inconsistent("trust"));
        }
        if candidate.law_digest != crate::signal::native_law_digest(family, &parameters) {
            return Err(inconsistent("law_digest"));
        }
    } else {
        let Some(provider) = &candidate.provider else {
            return Err(inconsistent("provider"));
        };
        if trust == SignalTrustLabel::NativeSupported || candidate.family != "external_finite_law" {
            return Err(inconsistent("trust"));
        }
        if trust == SignalTrustLabel::ExternallyAttested
            && candidate.attestor.as_deref().is_none_or(is_blank)
        {
            return Err(inconsistent("attestor"));
        }
        if provider.request_id != candidate.request_fingerprint {
            return Err(inconsistent("provider_request"));
        }
        if is_blank(&provider.provider_id) || is_blank(&provider.object_id) {
            return Err(inconsistent("provider"));
        }
    }
    let mut lineage = candidate.request.evidence_lineage.clone();
    lineage.sort();
    let receipt = SignalReceipt {
        request_fingerprint: candidate.request_fingerprint.clone(),
        candidate_id: candidate.request.candidate_id.clone(),
        prior_id: candidate.request.prior_id.clone(),
        sample_size: candidate.sample_size,
        rng_seed: candidate.request.rng_seed,
        family: candidate.family.clone(),
        update_mode: mode,
        trust,
        attestor: candidate.attestor.clone(),
        provider_identity: candidate.provider.as_ref().map(|p| ProviderObjectIdentity {
            provider_id: p.provider_id.clone(),
            object_id: p.object_id.clone(),
            version_id: p.version_id.clone(),
            snapshot_id: p.snapshot_id.clone(),
            request_id: p.request_id.clone(),
        }),
        evidence_lineage: lineage,
        conditional_independence: candidate.request.conditional_independence.clone(),
        limits: request.limits,
        law_digest: candidate.law_digest.clone(),
    };
    if receipt.identity() != candidate.signal_identity {
        return Err(inconsistent("signal_identity"));
    }
    Ok((trust, mode))
}

fn verify_candidate(
    wire: &DesignRankingArtifactWire,
    candidate: &EvsiCandidateWire,
    base: &Baseline,
    max_states: usize,
) -> Result<ConsumedCandidate> {
    let id = &candidate.semantic_id;
    let mismatch = |quantity: &'static str| DesignRankingArtifactError::ReplayMismatch {
        candidate: id.clone(),
        quantity,
    };
    if is_blank(id) {
        return Err(DesignRankingArtifactError::InvalidWire("blank candidate id"));
    }
    let (trust, mode) = check_signal(candidate)?;
    // Source overlap: re-queried through the evidence ledger, never trusted from the
    // stored diagnostic. The wire carries declared identities only, so an undeclared
    // side cannot occur here; any ledger refusal is unknown and refuses.
    let checked = match declared_overlap(
        DeclaredObservations::Declared(&wire.prior_observation_ids),
        DeclaredObservations::Declared(&candidate.reused_observation_ids),
    ) {
        ObservationOverlap::Independent { checked } => checked,
        ObservationOverlap::Shared { ids, .. } => {
            return Err(DesignRankingArtifactError::SourceOverlap { candidate: id.clone(), ids });
        }
        ObservationOverlap::Unknown { reason } => {
            return Err(DesignRankingArtifactError::SourceOverlapUnknown {
                candidate: id.clone(),
                reason,
            });
        }
    };
    if !candidate.overlapping.is_empty() || candidate.overlap_checked != checked as u64 {
        return Err(mismatch("source_overlap"));
    }
    let method = parse_method(&candidate.integration)
        .ok_or(DesignRankingArtifactError::InvalidWire("integration method"))?;
    let recomputed =
        recompute(candidate, &wire.decision, base, method, max_states).map_err(|slot| {
            DesignRankingArtifactError::SignalInconsistent { candidate: id.clone(), slot }
        })?;
    if !close(candidate.evpi, recomputed.evpi) {
        return Err(mismatch("evpi"));
    }
    if let Some(evsi) = recomputed.evsi {
        if !close(candidate.evsi, evsi) {
            return Err(mismatch("evsi"));
        }
        if candidate.stderr.abs() > 0.0 || candidate.replicates != 0 || candidate.ess.is_some() {
            return Err(mismatch("integration_error"));
        }
    } else {
        // Monte Carlo: error and replicates retained. The EVSI bound is the shared
        // predicate below, the same one the exact branch falls through to.
        if !(candidate.stderr.is_finite() && candidate.stderr >= 0.0 && candidate.replicates > 0)
            || candidate.ess.is_none()
        {
            return Err(mismatch("integration_error"));
        }
        if candidate.converged != (candidate.stderr <= wire.mc_error_tolerance) {
            return Err(mismatch("converged"));
        }
    }
    if !evsi_bound_ok(candidate.evsi, candidate.evpi, candidate.stderr) {
        return Err(mismatch("evsi_bounds"));
    }
    match &wire.cost_map {
        Some(map) => {
            let cost_utility = map.utility_per_cost * candidate.cost_amount;
            match (candidate.study_cost_utility, candidate.net_value) {
                (Some(stored_cost), Some(stored_net))
                    if close(stored_cost, cost_utility)
                        && close(stored_net, candidate.evsi - cost_utility) => {}
                _ => return Err(mismatch("net_value")),
            }
        }
        None => {
            if candidate.study_cost_utility.is_some() || candidate.net_value.is_some() {
                return Err(mismatch("net_value"));
            }
        }
    }
    Ok(ConsumedCandidate {
        semantic_id: id.clone(),
        replay: recomputed.replay,
        trust,
        update_mode: mode,
        trust_limit: trust_limit(trust, candidate.attestor.as_deref()),
    })
}

fn rank_key(candidate: &EvsiCandidateWire) -> f64 {
    candidate.net_value.unwrap_or(candidate.evsi)
}

/// Recompute the order, rank-uncertainty flags and ties from the stored values.
fn verify_order(wire: &DesignRankingArtifactWire) -> Result<()> {
    let mut rows: Vec<&EvsiCandidateWire> = wire.candidates.iter().collect();
    rows.sort_by(|a, b| {
        rank_key(b).total_cmp(&rank_key(a)).then_with(|| a.semantic_id.cmp(&b.semantic_id))
    });
    let ranked = rank_rows(
        &rows
            .iter()
            .map(|row| AdjacentRank {
                key: rank_key(row),
                stderr: row.stderr,
                id: row.semantic_id.as_str(),
            })
            .collect::<Vec<_>>(),
        wire.tie_tolerance,
    );
    for (i, row) in rows.iter().enumerate() {
        if row.rank != i as u64 {
            return Err(DesignRankingArtifactError::RankMismatch("rank"));
        }
        if row.rank_uncertain != ranked.uncertain[i] {
            return Err(DesignRankingArtifactError::RankMismatch("rank_uncertain"));
        }
    }
    let mut stored_ties = wire.ties.clone();
    let mut ties = ranked.ties;
    stored_ties.sort();
    ties.sort();
    if stored_ties != ties {
        return Err(DesignRankingArtifactError::RankMismatch("ties"));
    }
    Ok(())
}

fn verify_search(wire: &DesignRankingArtifactWire) -> Result<()> {
    let search = &wire.search;
    let ids: BTreeSet<&str> = wire.candidates.iter().map(|c| c.semantic_id.as_str()).collect();
    if ids.len() != wire.candidates.len() {
        return Err(DesignRankingArtifactError::InvalidWire("duplicate candidate id"));
    }
    if search.evaluated != wire.candidates.len() as u64
        || search.truncated == search.unevaluated_ids.is_empty()
        || search.supplied != search.evaluated + search.unevaluated_ids.len() as u64
        || search.evaluated > wire.max_candidates
        || search.unevaluated_ids.iter().any(|id| ids.contains(id.as_str()))
    {
        return Err(DesignRankingArtifactError::RankMismatch("search_receipt"));
    }
    Ok(())
}

pub(super) fn rebuild_ranking(wire: &DesignRankingArtifactWire) -> Result<RankingRecord> {
    let basis =
        rank_basis(&wire.basis).ok_or(DesignRankingArtifactError::InvalidWire("ranking basis"))?;
    if (basis == DesignRankingBasis::NetValue) != wire.cost_map.is_some() {
        return Err(DesignRankingArtifactError::RankMismatch("basis"));
    }
    let mut entries = Vec::with_capacity(wire.candidates.len());
    for c in &wire.candidates {
        entries.push(EvsiRankRow::try_from(c)?);
    }
    entries.sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
    let mut digests = wire.source_digests.clone();
    digests.sort();
    digests.dedup();
    Ok(RankingRecord {
        basis,
        decision_contract_identity: Some(wire.decision.contract_identity.clone()),
        utility_unit: Some(wire.decision.utility_unit.clone()),
        cost_mapping: wire.cost_map.as_ref().map(|m| CostToUtilityMap {
            cost_unit: m.cost_unit.clone(),
            utility_unit: m.utility_unit.clone(),
            utility_per_cost: m.utility_per_cost,
        }),
        rng_seed: Some(wire.rng_seed),
        source_digests: digests,
        entries,
        structural: Vec::new(),
        search: SearchReceipt {
            supplied: usize::try_from(wire.search.supplied)
                .map_err(|_| DesignRankingArtifactError::InvalidWire("search receipt"))?,
            evaluated: usize::try_from(wire.search.evaluated)
                .map_err(|_| DesignRankingArtifactError::InvalidWire("search receipt"))?,
            truncated: wire.search.truncated,
            unevaluated_ids: wire.search.unevaluated_ids.clone(),
        },
    })
}

/// Verify a decoded artifact body: digest, bounds, decision, costs, every candidate's
/// signal and recomputable value, order, search receipt and the retained expectations.
///
/// # Errors
///
/// The first inconsistency found; see [`DesignRankingArtifactError`].
pub fn consume_wire(
    wire: &DesignRankingArtifactWire,
    expectation: &ConsumeExpectation,
) -> Result<ConsumedDesignRanking> {
    consume_wire_with_receipts(wire, expectation, &BTreeSet::new())
}

/// [`consume_wire`] with the candidates whose exact-request verification the
/// consumer itself holds a receipt for.
///
/// The artifact does not retain a verification receipt, so a stored
/// `exact_request_verified` label is a claim: the consumed candidate reports
/// `externally_attested` unless its semantic id is in `verified_candidates`,
/// which the consumer fills only from a receipt it checked against the same
/// provider object and request. The label inside the signal identity is
/// unchanged, so a resealed upgrade is not mistaken for a verified one and an
/// honest verified artifact still replays.
///
/// # Errors
///
/// The first inconsistency found; see [`DesignRankingArtifactError`].
pub fn consume_wire_with_receipts(
    wire: &DesignRankingArtifactWire,
    expectation: &ConsumeExpectation,
    verified_candidates: &BTreeSet<String>,
) -> Result<ConsumedDesignRanking> {
    consume_wire_with_limits(
        wire,
        expectation,
        verified_candidates,
        &DesignRankingReadLimits::default(),
    )
}

/// [`consume_wire_with_receipts`] accepting the table sizes `limits` allows instead of the
/// defaults.
///
/// # Errors
///
/// Those of [`consume_wire_with_receipts`], with `limits` in place of the default bounds.
pub fn consume_wire_with_limits(
    wire: &DesignRankingArtifactWire,
    expectation: &ConsumeExpectation,
    verified_candidates: &BTreeSet<String>,
    limits: &DesignRankingReadLimits,
) -> Result<ConsumedDesignRanking> {
    if wire.version != DESIGN_RANKING_ARTIFACT_VERSION {
        return Err(DesignRankingArtifactError::UnsupportedVersion(wire.version));
    }
    if wire.kind != DESIGN_RANKING_ARTIFACT_KIND {
        return Err(DesignRankingArtifactError::InvalidWire("kind"));
    }
    check_bounds(wire, limits)?;
    let digest = wire.compute_digest()?;
    if digest != wire.digest {
        return Err(DesignRankingArtifactError::DigestMismatch);
    }
    if let Some(expected) = &expectation.artifact_identity {
        if *expected != digest {
            return Err(DesignRankingArtifactError::ExpectedIdentityMismatch);
        }
    }
    if wire.calibration != DESIGN_RANKING_CALIBRATION {
        return Err(DesignRankingArtifactError::InvalidWire("calibration label"));
    }
    if !(wire.mc_error_tolerance.is_finite()
        && wire.mc_error_tolerance >= 0.0
        && wire.tie_tolerance.is_finite()
        && wire.tie_tolerance >= 0.0)
    {
        return Err(DesignRankingArtifactError::InvalidWire("tolerances"));
    }
    check_decision(&wire.decision)?;
    check_costs(wire)?;
    verify_search(wire)?;
    let base = baseline(&wire.decision);
    if wire.bayes_action != base.bayes as u64
        || !close(wire.prior_expected_utility, base.prior_best)
        || !close(wire.evpi, base.evpi)
    {
        return Err(DesignRankingArtifactError::ReplayMismatch {
            candidate: String::new(),
            quantity: "decision_baseline",
        });
    }
    let mut consumed = Vec::with_capacity(wire.candidates.len());
    for candidate in &wire.candidates {
        let mut one = verify_candidate(wire, candidate, &base, limits.max_states)?;
        if one.trust == SignalTrustLabel::ExactRequestVerified
            && !verified_candidates.contains(&candidate.semantic_id)
        {
            // A stored verification label with no receipt in the consumer's
            // hands is an attestation, never a verification.
            one.trust = SignalTrustLabel::ExternallyAttested;
            one.trust_limit = trust_limit_unreceipted(candidate.attestor.as_deref());
        }
        consumed.push(one);
    }
    verify_order(wire)?;
    let ranking = rebuild_ranking(wire)?;
    if ranking.identity() != wire.ranking_identity {
        return Err(DesignRankingArtifactError::RankMismatch("ranking_identity"));
    }
    let signals: BTreeMap<String, String> =
        expectation.signal_identities.clone().unwrap_or_else(|| {
            ranking
                .entries
                .iter()
                .map(|e| (e.semantic_id.clone(), e.signal_identity.clone()))
                .collect()
        });
    let digests =
        expectation.source_digests.clone().unwrap_or_else(|| ranking.source_digests.clone());
    let contract = expectation
        .decision_contract_identity
        .clone()
        .unwrap_or_else(|| wire.decision.contract_identity.clone());
    let cost_mapping =
        expectation.cost_mapping.clone().unwrap_or_else(|| ranking.cost_mapping.clone());
    ranking.verify_replay(&contract, &signals, &digests, cost_mapping.as_ref())?;
    if let Some(retained) = &expectation.signal_identities {
        let stored: BTreeSet<&String> = ranking.entries.iter().map(|e| &e.semantic_id).collect();
        if let Some(extra) = retained.keys().find(|id| !stored.contains(id)) {
            return Err(RankingError::SignalMismatch(extra.clone()).into());
        }
    }
    let rank_order: BTreeMap<&str, usize> = ranking
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.semantic_id.as_str(), index))
        .collect();
    consumed.sort_by_key(|candidate| rank_order.get(candidate.semantic_id.as_str()).copied());
    let provenance_chain = wire
        .provenance_chain()
        .map_err(|_| DesignRankingArtifactError::InvalidWire("provenance chain"))?;
    Ok(ConsumedDesignRanking {
        ranking,
        identity: digest,
        candidates: consumed,
        calibration: DESIGN_RANKING_CALIBRATION,
        provenance_chain,
    })
}

/// Decode and verify exported bytes.
///
/// # Errors
///
/// Corruption, oversize, unknown version, or any inconsistency of
/// [`consume_wire`].
pub fn consume(bytes: &[u8], expectation: &ConsumeExpectation) -> Result<ConsumedDesignRanking> {
    consume_with_limits(bytes, expectation, &DesignRankingReadLimits::default())
}

/// [`consume`] accepting the container and table sizes `limits` allows instead of the defaults.
///
/// # Errors
///
/// Those of [`consume`], with `limits` in place of the default bounds.
pub fn consume_with_limits(
    bytes: &[u8],
    expectation: &ConsumeExpectation,
    limits: &DesignRankingReadLimits,
) -> Result<ConsumedDesignRanking> {
    let wire = DesignRankingArtifactWire::from_bytes_with_limits(bytes, limits)?;
    consume_wire_with_limits(&wire, expectation, &BTreeSet::new(), limits)
}
