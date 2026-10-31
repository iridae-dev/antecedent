//! TemporalCpdag/Pag graph-posterior identification caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, ExecutionContext, TemporalEffectQuery};
use antecedent_discovery::GraphPosterior;
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

    let _lag_masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let _max_lag = posterior.max_lag().ok_or_else(|| CausalError::Compile {
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
    let budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let mapped = map_posterior_graphs(posterior, ctx, |i, inner| {
        let mut budget = budget.clone();
        let mut config = config.clone();
        config.context = Some(inner.clone());
        config.temporal_budget = Some(budget.clone());
        let key = dbn_envelope_key(i)?;
        let Some(atom) =
            super::admitted_temporal_class_atom(posterior, i, variables, &mut budget, inner)?
        else {
            return Ok(None);
        };
        let envelope = match atom {
            super::AdmittedTemporalClassAtom::Cpdag(graph, _lease) => {
                temporal_class_attempt(identify_temporal_cpdag_configured(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &graph,
                    query,
                    config,
                ))?
            }
            super::AdmittedTemporalClassAtom::Pag(graph, _lease) => temporal_class_attempt(
                identify_temporal_pag_configured(DEFAULT_PAG_IDENTIFIER_ID, &graph, query, config),
            )?,
        };
        let cached = envelope.map(|envelope| cache_temporal_class_atom(key, query, envelope));
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

#[cfg(test)]
mod temporal_class_budget_tests {
    use super::*;
    use antecedent_core::{ExecutionContext, TemporalBudget, TemporalEffectQuery, VariableId};
    use antecedent_discovery::{GraphPosteriorAtomKind, SamplingSummary};
    use antecedent_prob::InferenceDiagnostics;

    fn posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.5, 0.5],
            vec![0_u64, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            SamplingSummary {
                ess: 2.0,
                diagnostics: InferenceDiagnostics::analytic("temporal-class-budget"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(kind)
        .with_lagged_marginals(1, vec![0.0; 4])
        .unwrap()
        .with_lag_masks(vec![0_u64, 0])
        .unwrap()
    }

    #[test]
    fn class_posterior_conversion_shares_budget_and_keeps_operational_errors() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
            let posterior = posterior(kind);
            let ctx = ExecutionContext::for_tests(1);
            let mut limits = ctx.temporal_materialization_limits();
            limits.max_work = 31;
            let mut budget = TemporalBudget::new(limits);
            let first =
                super::super::admitted_temporal_class_atom(&posterior, 0, &vars, &mut budget, &ctx)
                    .unwrap();
            assert!(first.is_some());
            assert!(matches!(
                super::super::admitted_temporal_class_atom(&posterior, 1, &vars, &mut budget, &ctx),
                Err(CausalError::Resource { .. })
            ));
            drop(first);

            let mut limits = ctx.temporal_materialization_limits();
            limits.max_work = 0;
            let limited = ctx.with_temporal_materialization_limits(limits);
            let query = TemporalEffectQuery::pulse(vars[0], vars[1], 1.0);
            assert!(matches!(
                build_temporal_class_posterior_identification_cache(
                    &posterior, &vars, &query, None, &limited
                ),
                Err(CausalError::Resource { .. })
            ));
            let cancelled = ExecutionContext::for_tests(2);
            cancelled.cancellation.cancel();
            assert!(matches!(
                build_temporal_class_posterior_identification_cache(
                    &posterior, &vars, &query, None, &cancelled
                ),
                Err(CausalError::Cancelled { .. })
            ));
        }
    }
}
