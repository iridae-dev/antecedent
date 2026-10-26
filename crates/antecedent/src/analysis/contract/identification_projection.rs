//! Projection of cached identification products onto the contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

// A first identified posterior atom is not the identification status of the
// entire mixture. Preserve unidentified mass and disagreeing estimands at
// compilation, as the executor does, so the program and execution agree.
pub(super) fn contract_identification(study: &Study) -> Option<Cow<'_, IdentificationResult>> {
    let Some(cached) = cached_identification(study) else {
        // TemporalCpdag / TemporalPag prepare caches a class envelope rather
        // than a per-graph result. Aggregate it the way the executor does, so
        // the prepared program binds a product instead of none at all.
        let cache = study.temporal_class_cache()?;
        return Some(Cow::Owned(crate::analysis::execute::envelope_to_identification_result_for(
            &cache.envelope.envelope,
            study.query.clone(),
        )));
    };
    let (unidentified, contributing) = if let Some(cache) = study.graph_posterior_cache() {
        (
            cache.graphs.unidentified_mass(),
            cache.atoms.iter().map(|atom| &atom.estimand).collect::<Vec<_>>(),
        )
    } else if let Some(cache) = study.dbn_posterior_cache() {
        (
            cache.graphs.unidentified_mass(),
            cache.atoms.iter().map(|atom| &atom.estimand).collect::<Vec<_>>(),
        )
    } else {
        return Some(Cow::Borrowed(cached));
    };
    let status = crate::analysis::execute::graph_posterior_mixture_status(
        unidentified,
        &contributing,
        cached.status,
    );
    if status == cached.status {
        Some(Cow::Borrowed(cached))
    } else {
        let mut aggregate = cached.clone();
        aggregate.status = status;
        Some(Cow::Owned(aggregate))
    }
}

pub(super) fn cached_identification(study: &Study) -> Option<&IdentificationResult> {
    study.identification.as_ref()?.primary()
}

/// The study's temporal identification product and its exact unfolded variable namespace,
/// however it was produced.
///
/// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
/// envelope instead cache a per-atom or per-completion result with its own unfold indexer;
/// [`dbn_projected_temporal_identification`] and [`class_projected_temporal_identification`]
/// project those onto the same [`CachedTemporalIdentification`] shape so every consumer —
/// the compiled contract and an exported `analysis_result` artifact alike — validates and
/// reports the same namespace for the same study.
pub(crate) fn full_temporal_identification(
    study: &Study,
) -> Option<Cow<'_, CachedTemporalIdentification>> {
    if let Some(cache) = study.temporal_cache() {
        return Some(Cow::Borrowed(&**cache));
    }
    dbn_projected_temporal_identification(study)
        .or_else(|| class_projected_temporal_identification(study))
        .map(Cow::Owned)
}

/// Project a DBN posterior atom onto the existing temporal-namespace owner.
///
/// Graph-posterior Pulse stores lagged-node identification on
/// the DBN-posterior product, not the temporal-backdoor product.
/// Encode reuses [`HorizonAdjustmentNodeWire`] so unfolded IDs stay inside a
/// declared variable+offset namespace.
pub(super) fn dbn_projected_temporal_identification(
    study: &Study,
) -> Option<CachedTemporalIdentification> {
    let cache = study.dbn_posterior_cache()?;
    let atom = cache.atoms.first()?;
    let horizon = query_horizon_steps(&study.query);
    Some(CachedTemporalIdentification {
        by_horizon: Arc::from([CachedTemporalHorizonIdentification {
            horizon,
            identification: atom.identification.clone(),
            estimand: atom.estimand.clone(),
            indexer: atom.indexer.clone(),
        }]),
    })
}

/// Project a TemporalCpdag/Pag envelope indexer onto the same namespace owner.
pub(super) fn class_projected_temporal_identification(
    study: &Study,
) -> Option<CachedTemporalIdentification> {
    if let Some(cache) = study.temporal_class_cache() {
        let (horizon, envelope) = cache.by_horizon.first().map_or_else(
            || (query_horizon_steps(&study.query), &cache.envelope),
            |(horizon, envelope)| (*horizon, envelope),
        );
        if let Some(projected) = project_class_envelope(horizon, envelope) {
            return Some(projected);
        }
    }
    let cache = study.temporal_class_posterior_cache()?;
    let atom = cache.class_atoms.first()?;
    project_class_envelope(query_horizon_steps(&study.query), &atom.envelope)
}

pub(super) fn project_class_envelope(
    horizon: u32,
    envelope: &antecedent_identify::TemporalClassEnvelope,
) -> Option<CachedTemporalIdentification> {
    let indexer = envelope.indexers.first()?.clone();
    let (case, estimand) = envelope
        .envelope
        .cases
        .iter()
        .find_map(|case| case.result.estimands.first().map(|estimand| (case, estimand)))?;
    Some(CachedTemporalIdentification {
        by_horizon: Arc::from([CachedTemporalHorizonIdentification {
            horizon,
            identification: case.result.clone(),
            estimand: estimand.clone(),
            indexer,
        }]),
    })
}

pub(super) fn query_horizon_steps(query: &antecedent_core::CausalQuery) -> u32 {
    match query {
        antecedent_core::CausalQuery::TemporalEffect(query) => query.horizon_steps,
        antecedent_core::CausalQuery::Response(query) => {
            query.temporal.as_ref().and_then(|spec| spec.horizons.first()).copied().unwrap_or(1)
        }
        _ => 1,
    }
}

/// Shape of the enumerated class a prepared product covers, when there is one.
pub(super) fn class_envelope_wire(study: &Study) -> Option<IdentificationEnvelopeWire> {
    if let Some(cache) = study.temporal_class_cache() {
        return Some(envelope_shape(&cache.envelope.envelope));
    }
    if let Some(cache) = study.cpdag_cache() {
        return Some(envelope_shape(&cache.envelope));
    }
    if let Some(cache) = study.pag_cache() {
        return Some(envelope_shape(&cache.envelope));
    }
    None
}

pub(super) fn envelope_shape<G>(
    envelope: &IdentificationEnvelope<G>,
) -> IdentificationEnvelopeWire {
    IdentificationEnvelopeWire {
        cases: envelope.cases.len() as u64,
        identified_weight_bits: envelope.identified_weight.0.to_bits(),
        unidentified_weight_bits: envelope.unidentified_weight.0.to_bits(),
        truncated_completions: envelope.truncated_completions as u64,
    }
}

pub(super) fn identification_search_capped(result: &IdentificationResult) -> bool {
    result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE)
}
