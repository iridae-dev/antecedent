//! TemporalCpdag/Pag graph-posterior identification caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, ExecutionContext, TemporalEffectQuery};
use antecedent_discovery::GraphPosterior;
use antecedent_discovery::dbn_posterior::{
    temporal_cpdag_from_dbn_atoms, temporal_pag_from_dbn_atoms,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::{dbn_envelope_key, map_posterior_graphs};
use crate::analysis::prepared::CachedTemporalClassIdentification;
use crate::error::CausalError;

/// Class-envelope identification for one TemporalCpdag/Pag posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorAtomIdentification {
    /// Collision-free, position-derived key used by the effect envelope.
    pub key: u64,
    /// Envelope-level identification (status, diagnostics, assumptions).
    pub identification: IdentificationResult,
    /// Shared estimand when every identified completion agrees.
    pub invariant: Option<IdentifiedEstimand>,
    /// Completions + unfold indexers for this class atom.
    pub envelope: Arc<antecedent_identify::TemporalClassEnvelope>,
    /// Envelope identified weight before posterior mixing.
    pub identified_weight: f64,
    /// Completions whose search was capped.
    pub truncated_completions: usize,
}

/// Prepare-time identification for TemporalCpdag/Pag graph-posterior Pulse/Sustained.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified class atoms. Unidentified atoms remain in [`Self::graphs`].
    pub class_atoms: Arc<[CachedTemporalClassPosteriorAtomIdentification]>,
    /// The first atom's envelope for the requested horizon, as the class product a
    /// single-graph class response reads. Only the response route sets it; the
    /// study's class accessor answers with it while this posterior is the installed product.
    pub first_atom_class: Option<Arc<CachedTemporalClassIdentification>>,
}

/// Identify every TemporalCpdag/Pag posterior atom with the class envelope.
///
/// Each atom is reconstructed from adjacency + lag masks (+ mark masks for
/// Pag). Completions that disagree keep envelope status; unidentified class
/// mass stays on [`CachedTemporalClassPosteriorIdentification::graphs`].
/// Completion enumeration is not posterior probability.
#[allow(
    clippy::too_many_lines,
    reason = "validates the temporal class posterior's lag masks and atom kind, then identifies each CPDAG or PAG atom under one configuration and walks the samples once into atoms, weights and flags"
)]
pub(crate) fn build_temporal_class_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &TemporalEffectQuery,
    max_completions: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<CachedTemporalClassPosteriorIdentification, CausalError> {
    use crate::strategy_table::{
        DEFAULT_PAG_IDENTIFIER_ID, identify_temporal_cpdag_configured,
        identify_temporal_pag_configured,
    };

    let lag_masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior.max_lag().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    if !matches!(
        posterior.atom_kind(),
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag
    ) {
        return Err(CausalError::Compile {
            message: "temporal class posterior requires Cpdag or Pag atom_kind".into(),
        });
    }
    crate::analysis::execute::report_identify_compute(ctx);
    let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
    if let Some(max) = max_completions {
        config.max_completions = max;
    }
    let mapped = map_posterior_graphs(posterior, ctx, |i, inner| {
        let mut config = config.clone();
        config.context = Some(inner.clone());
        let key = dbn_envelope_key(i)?;
        let mark = posterior.mark_atom(i);
        let cached = match posterior.atom_kind() {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = temporal_cpdag_from_dbn_atoms(
                    &posterior.atoms()[i],
                    &lag_masks[i],
                    posterior.n_vars(),
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok::<_, CausalError>(
                    temporal_class_attempt(identify_temporal_cpdag_configured(
                        DEFAULT_PAG_IDENTIFIER_ID,
                        &cpdag,
                        query,
                        config.clone(),
                    ))?
                    .map(|envelope| cache_temporal_class_atom(key, query, envelope)),
                )
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = temporal_pag_from_dbn_atoms(
                    &posterior.atoms()[i],
                    &lag_masks[i],
                    &mark,
                    posterior.n_vars(),
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok(temporal_class_attempt(identify_temporal_pag_configured(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &pag,
                    query,
                    config.clone(),
                ))?
                .map(|envelope| cache_temporal_class_atom(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut class_atoms = Vec::new();

    for (i, cached) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            progress.report(i as f64 / posterior.n_graphs().max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights()[i]);
        let Some(cached) = cached else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        if cached.identified_weight > 0.0 {
            flags.push(GraphIdentFlag::Identified);
            class_atoms.push(cached);
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }
    let graphs = WeightedGraphSamples::new(weights, flags, keys)?;
    Ok(CachedTemporalClassPosteriorIdentification {
        graphs,
        class_atoms: Arc::from(class_atoms),
        first_atom_class: None,
    })
}

/// Scientific refusals keep unidentified mass; interrupted or unfunded searches refuse the attempt.
fn temporal_class_attempt(
    attempt: Result<antecedent_identify::TemporalClassEnvelope, CausalError>,
) -> Result<Option<antecedent_identify::TemporalClassEnvelope>, CausalError> {
    match attempt {
        Ok(envelope) => Ok(Some(envelope)),
        Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => Err(error),
        Err(_) => Ok(None),
    }
}

fn cache_temporal_class_atom(
    key: u64,
    query: &TemporalEffectQuery,
    envelope: antecedent_identify::TemporalClassEnvelope,
) -> CachedTemporalClassPosteriorAtomIdentification {
    let identification = crate::analysis::execute::envelope_to_identification_result_for(
        &envelope.envelope,
        CausalQuery::TemporalEffect(query.clone()),
    );
    CachedTemporalClassPosteriorAtomIdentification {
        key,
        identification,
        invariant: envelope.envelope.invariant.clone(),
        identified_weight: envelope.envelope.identified_weight.0,
        truncated_completions: envelope.envelope.truncated_completions,
        envelope: Arc::new(envelope),
    }
}
