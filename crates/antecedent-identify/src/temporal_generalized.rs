//! `TemporalCpdag` / `TemporalPag` identification envelopes.
//!
//! Each `TemporalDag` completion is identified with
//! [`TemporalBackdoorIdentifier`]. Completions that do not identify contribute
//! unidentified mass. Multi-step Sustained is refused: 1.4 licenses Pulse and
//! single-step Sustained only.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{TemporalEffectQuery, TemporalPolicy};
use antecedent_data::TemporalIndexer;
use antecedent_graph::{
    TemporalCpdag, TemporalCpdagCompletionSampler, TemporalDag, TemporalPag,
    TemporalPagCompletionSampler,
};

use crate::envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
use crate::error::IdentificationError;
use crate::generalized::GeneralizedAdjustmentIdentifier;
use crate::result::IdentificationStatus;
use crate::temporal_backdoor::TemporalBackdoorIdentifier;

/// Generalized-adjustment envelope over `TemporalDag` completions, plus the
/// unfold indexer each case needs for estimation.
#[derive(Clone, Debug)]
pub struct TemporalClassEnvelope {
    /// Per-completion identification and mass.
    pub envelope: IdentificationEnvelope<TemporalDag>,
    /// Unfold indexers aligned with [`Self::envelope`].`cases`.
    pub indexers: Vec<TemporalIndexer>,
}

fn refuse_multi_step(query: &TemporalEffectQuery) -> Result<(), IdentificationError> {
    if matches!(query.policy, TemporalPolicy::Sustained { from, until } if from != until) {
        return Err(IdentificationError::UnsupportedQuery {
            message: "class-aware TemporalCpdag/TemporalPag identification is Pulse \
                      and single-step Sustained only",
        });
    }
    Ok(())
}

impl GeneralizedAdjustmentIdentifier {
    /// Identify a temporal effect over a `TemporalCpdag` MEC envelope.
    ///
    /// # Errors
    ///
    /// Multi-step Sustained, conflict marks, or identification/unfold errors.
    pub fn identify_temporal_cpdag_envelope(
        &self,
        cpdag: &TemporalCpdag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        refuse_multi_step(query)?;
        let mut sampler =
            TemporalCpdagCompletionSampler::new(cpdag.clone(), self.config.max_completions)?;
        let (envelope, indexers) = identify_temporal_completions(&mut sampler, query, self)?;
        let mut envelope = envelope;
        let n = cpdag.undirected_edge_count();
        if n > 0 {
            envelope.push_features([GraphFeature {
                kind: Arc::from("temporal_cpdag_undirected_marks"),
                detail: Arc::from(format!("{n} undirected edge(s) in source TemporalCpdag")),
            }]);
        }
        if sampler.hit_cap() {
            downgrade_capped(&mut envelope, self.config.max_completions);
        }
        Ok(TemporalClassEnvelope { envelope, indexers })
    }

    /// Identify a temporal effect over a `TemporalPag` completion envelope.
    ///
    /// # Errors
    ///
    /// Multi-step Sustained, bidirected/conflict marks, or identification errors.
    pub fn identify_temporal_pag_envelope(
        &self,
        pag: &TemporalPag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        refuse_multi_step(query)?;
        let mut sampler =
            TemporalPagCompletionSampler::new(pag.clone(), self.config.max_completions)?;
        let (envelope, indexers) = identify_temporal_completions(&mut sampler, query, self)?;
        let mut envelope = envelope;
        envelope.push_features([GraphFeature {
            kind: Arc::from("temporal_pag_circle_marks"),
            detail: Arc::from("TemporalPag completions are definite directed TemporalDags"),
        }]);
        if sampler.hit_cap() {
            downgrade_capped(&mut envelope, self.config.max_completions);
        }
        Ok(TemporalClassEnvelope { envelope, indexers })
    }
}

fn identify_temporal_completions<I>(
    sampler: &mut I,
    query: &TemporalEffectQuery,
    id: &GeneralizedAdjustmentIdentifier,
) -> Result<(IdentificationEnvelope<TemporalDag>, Vec<TemporalIndexer>), IdentificationError>
where
    I: Iterator,
    I::Item: CompletionGraph,
{
    let backdoor = TemporalBackdoorIdentifier::new();
    let w = ProbabilityMass(id.config.per_completion_weight);
    let mut cases = Vec::new();
    let mut indexers = Vec::new();
    for completion in sampler.by_ref() {
        let graph = completion.into_graph();
        let identified = backdoor.identify_temporal(&graph, query)?;
        cases.push(GraphIdentificationCase { graph, result: identified.result, weight: w });
        indexers.push(identified.indexer);
    }
    Ok((IdentificationEnvelope::from_cases(cases), indexers))
}

fn downgrade_capped(envelope: &mut IdentificationEnvelope<TemporalDag>, max_completions: usize) {
    envelope.push_features([GraphFeature {
        kind: Arc::from("completion_enumeration_capped"),
        detail: Arc::from(format!(
            "retained {} temporal completion(s) under max_completions={}; \
             identification is established only over the deterministic retained subset",
            envelope.cases.len(),
            max_completions
        )),
    }]);
    if envelope.status == IdentificationStatus::NonparametricallyIdentified {
        envelope.status = IdentificationStatus::PartiallyIdentified;
    }
}

trait CompletionGraph {
    fn into_graph(self) -> TemporalDag;
}

impl CompletionGraph for antecedent_graph::TemporalCpdagCompletion {
    fn into_graph(self) -> TemporalDag {
        self.graph
    }
}

impl CompletionGraph for antecedent_graph::TemporalPagCompletion {
    fn into_graph(self) -> TemporalDag {
        self.graph
    }
}
